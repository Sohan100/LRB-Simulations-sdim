"""Diagnosis scan: how non-exponential is each LRB-D policy in the sdim 1.4.0 data?

For every configuration (one/two ancilla x SI1000/SD6, production and the
high-p extensions), every p > 0 and the policies unif 1..6 and const 0..3,
this computes on the accepted shots:

* the pipeline's statistic (mean over sequences of |sum_i omega^i P_i|) and
  its fit r_pipe (exact mirror of the production fit);
* the pooled signed polarization lambda = (3 P(0 | accepted) - 1)/2 pooled
  over sequences, with a sequence bootstrap;
* windowed rates between consecutive usable depths, a log-quadratic fit
  ln lambda = c0 - c1 m - c2 m^2, and the first/last window ratio;
* acceptance per depth.

Writes results/curvature_scan.csv and results/curves.npz. Cheap: seconds.
"""
from __future__ import annotations

import csv
import os

import numpy as np

import nmlib as L

MIN_ACC_TOTAL = 1000  # pooled accepted shots needed to use a depth (SE <~ 0.03)
POLICIES = [("unif", k) for k in range(1, 7)] + [("const", c) for c in range(0, 4)]
N_BOOT = 200


def usable(c: dict) -> np.ndarray:
    return np.isfinite(c["pooled"]) & (c["n_acc_tot"] >= MIN_ACC_TOTAL) & (c["pooled"] > 0)


def analyse(key, p_idx, kind, k):
    m = L.meta(key)
    depths = np.asarray(m.depths)
    try:
        t, rj = L.counts(key, p_idx, kind, k)
    except FileNotFoundError:
        return None
    c = L.curves(t, rj)
    b = L.bootstrap_curves(t, rj, n_boot=N_BOOT, seed=1000 * p_idx + k)
    se = np.nanstd(b["pooled"], 0)
    se_acc = np.nanstd(b["acc"], 0)
    ok = usable(c) & (se > 0)
    row = {
        "config": f"{key[0]}-{key[1]}",
        "source": key[2],
        "p": m.probs[p_idx],
        "p_idx": p_idx,
        "policy": f"{kind}{k}",
        "n_usable": int(ok.sum()),
        "depths_usable": " ".join(str(d) for d in depths[ok]),
        "lambda0": c["pooled"][0],
        "acc0": c["acc"][0],
    }
    pipe = L.fit_fixed_a(depths, c["pipeline"], c["pipeline_std"])
    row["r_pipe"] = pipe["r"] if pipe else np.nan
    row["n_pipe"] = pipe["n"] if pipe else 0
    if ok.sum() >= 2:
        fit = L.fit_fixed_a(depths[ok], c["pooled"][ok], se[ok], y_ceiling=None)
        row["r_pooled"] = fit["r"]
        mids, wr = L.windowed_rates(depths[ok], c["pooled"][ok])
        row["r_first_window"] = wr[0]
        row["r_last_window"] = wr[-1]
        row["window_ratio"] = wr[-1] / wr[0] if wr[0] > 0 else np.nan
        # bootstrap the fitted rate and the window ratio on the same depth set
        rb, wrat = [], []
        for bp in b["pooled"]:
            yy = bp[ok]
            if np.all(np.isfinite(yy)) and np.all(yy > 0):
                f = L.fit_fixed_a(depths[ok], yy, se[ok], y_ceiling=None)
                rb.append(f["r"])
                _, w = L.windowed_rates(depths[ok], yy)
                if w[0] > 0:
                    wrat.append(w[-1] / w[0])
        row["r_pooled_se"] = float(np.std(rb)) if rb else np.nan
        row["window_ratio_lo"], row["window_ratio_hi"] = (
            np.percentile(wrat, [16, 84]) if len(wrat) > 10 else (np.nan, np.nan))
    if ok.sum() >= 4:
        q = L.fit_log_poly(depths[ok], c["pooled"][ok], se[ok], deg=2)
        lin = L.fit_log_poly(depths[ok], c["pooled"][ok], se[ok], deg=1)
        row["c1"], row["c2"] = q["c"][1], q["c"][2]
        row["chi2_lin"], row["dof_lin"] = lin["chi2"], lin["dof"]
        row["chi2_quad"], row["dof_quad"] = q["chi2"], q["dof"]
        c2b = []
        for bp in b["pooled"]:
            yy = bp[ok]
            if np.all(np.isfinite(yy)) and np.all(yy > 0):
                c2b.append(L.fit_log_poly(depths[ok], yy, se[ok], deg=2)["c"][2])
        row["c2_lo"], row["c2_hi"] = np.percentile(c2b, [2.5, 97.5])
        # share of -ln lambda at the deepest usable depth due to the m^2 term
        mmax = depths[ok][-1]
        row["curv_share_at_mmax"] = q["c"][2] * mmax ** 2 / (q["c"][1] * mmax + q["c"][2] * mmax ** 2)
    # acceptance per depth step from a log-linear fit (heralded discard rate)
    okA = (c["acc"] > 0) & (c["n_acc_tot"] >= MIN_ACC_TOTAL)
    if okA.sum() >= 2:
        sl = se_acc[okA] / c["acc"][okA]
        sl = np.where(sl > 0, sl, np.nanmax(sl[sl > 0]) if np.any(sl > 0) else 1.0)
        A = np.vstack([np.ones(okA.sum()), depths[okA]]).T
        W = 1 / sl ** 2
        coef, *_ = np.linalg.lstsq(A * np.sqrt(W)[:, None], np.log(c["acc"][okA]) * np.sqrt(W), rcond=None)
        row["acc_per_step"] = float(np.exp(coef[1]))
        mids, wa = L.windowed_rates(depths[okA], c["acc"][okA])  # reuse: 2/3(1-ratio)
        row["acc_first_window"] = float(1 - 1.5 * wa[0])
        row["acc_last_window"] = float(1 - 1.5 * wa[-1])
    return row, {"depths": depths, **{f: c[f] for f in ("pooled", "pipeline", "pipeline_std", "acc", "n_acc_tot", "n_acc_min", "rice_floor", "joint", "signed_mean")}, "pooled_se": se, "acc_se": se_acc}


def main():
    rows, store = [], {}
    for key in L.RUNS:
        m = L.meta(key)
        for p_idx, p in enumerate(m.probs):
            if p == 0:
                continue
            for kind, k in POLICIES:
                res = analyse(key, p_idx, kind, k)
                if res is None:
                    continue
                row, cur = res
                rows.append(row)
                tag = f"{key[0]}-{key[1]}-{key[2]}|{p}|{kind}{k}"
                for f, v in cur.items():
                    store[f"{tag}|{f}"] = np.asarray(v, dtype=float)
    cols = sorted({c for r in rows for c in r}, key=lambda c: list(rows[0]).index(c) if c in rows[0] else 99)
    out = os.path.join(L.OUT, "results", "curvature_scan.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(os.path.join(L.OUT, "results", "curves.npz"), **store)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
