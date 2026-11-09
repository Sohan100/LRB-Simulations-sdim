"""Estimator comparison and its effect on the quoted pseudo-thresholds.

For unif k = 1..4 and every p, computes the LRB error per depth step with:

  pipe        production statistic and fit (mean over sequences of
              |sum_i omega^i P_i|, weights 1/std^2, every depth with at
              least one accepted shot in every sequence);
  pooled      signed polarization (3 P(0 | accepted) - 1)/2 pooled over
              sequences, every depth with >= 1000 accepted shots in total;
  aligned     pooled, but only depths that are multiples of k (so every
              depth ends a checked interval: no m mod k sawtooth);
  uncond      unconditional decay of J(m) = E[Re omega^residual * 1_accept]
              = A(m) lambda(m) on aligned depths: a discarded shot is scored
              as a random guess. Linear in the process, no ratio.

and the acceptance per depth step a (log-linear fit of A(m), aligned depths).
Physical RB uses the production fit (offset 0) on results/RB.

The pseudo-threshold is the p where r_LRB = r_RB, taken as the last
crossing from below (LRB stays worse above it), linearly interpolated in
log(r_LRB / r_RB). SD6 merges production (p < 0.025) with the high-p
extension (p >= 0.025), like scripts/merge_highp_threshold_extension.py.
SI1000 is production-only (its extension was ~15% complete); rows marked
"si1000+ext" use the partial extension above p = 0.015 as a cross-check.

Writes results/estimator_table.csv and results/pseudo_thresholds_by_estimator.csv.
"""
from __future__ import annotations

import csv
import os
import warnings

import numpy as np

import nmlib as L

warnings.filterwarnings("ignore", category=RuntimeWarning)
MIN_ACC = 1000
N_BOOT = 100
ESTIMATORS = ("pipe", "pooled", "aligned", "uncond")


def merged_grid(anc: str, noise: str, use_ext: bool):
    split = 0.025 if noise == "sd6" else 0.015
    prod = (anc, noise, "prod")
    pts = [(prod, i, p) for i, p in enumerate(L.meta(prod).probs) if p > 0 and (not use_ext or p < split)]
    if use_ext:
        ext = (anc, noise, "ext")
        pts += [(ext, i, p) for i, p in enumerate(L.meta(ext).probs) if p >= split]
    return pts


def lrb_rates(key, p_idx, k, seq_idx=None):
    depths = np.asarray(L.meta(key).depths)
    t, rj = L.counts(key, p_idx, "unif", k)
    c = L.curves(t, rj, seq_idx)
    out = {}
    if seq_idx is None:
        f = L.fit_fixed_a(depths, c["pipeline"], c["pipeline_std"])
        out["pipe"] = f["r"] if f else np.nan
    use = np.isfinite(c["pooled"]) & (c["n_acc_tot"] >= MIN_ACC) & (c["pooled"] > 0)
    aligned = use & (depths % k == 0)
    # every fit includes depth 0 (user rule, 2026-10-31)
    sets = {"pooled": use, "aligned": aligned}
    # equal weights in ln(lambda) would over-weight noisy deep points; use the
    # binomial SE of the pooled estimator, sqrt(9 P0 (1 - P0) / 4 N_acc)
    p0 = (2 * c["pooled"] + 1) / 3
    se = np.sqrt(np.clip(9 * p0 * (1 - p0) / 4, 1e-12, None) / np.maximum(c["n_acc_tot"], 1))
    for name, m in sets.items():
        f = L.fit_fixed_a(depths[m], c["pooled"][m], se[m], y_ceiling=None) if m.sum() >= 2 else None
        out[name] = f["r"] if f else np.nan
        out[name + "_n"] = int(m.sum())
    m = aligned & (c["joint"] > 0)
    n_tot = np.maximum(c["n_acc_tot"] / np.maximum(c["acc"], 1e-300), 1)
    sej = np.sqrt(np.clip(c["acc"] * (1 - c["acc"]), 1e-12, None) / n_tot)
    f = L.fit_fixed_a(depths[m], c["joint"][m], sej[m], y_ceiling=None) if m.sum() >= 2 else None
    out["uncond"] = f["r"] if f else np.nan
    mA = (depths % k == 0) & (c["n_acc_tot"] >= MIN_ACC)
    if mA.sum() >= 2:
        slope = np.polyfit(depths[mA], np.log(c["acc"][mA]), 1)[0]
        out["acc_per_step"] = float(np.exp(slope))
    else:
        out["acc_per_step"] = np.nan
    return out


def rb_rate(key, p_idx):
    try:
        mean, std = L.rb_curve(key, p_idx)
    except FileNotFoundError:
        # running extension: no RB output yet; physical RB is smooth in p, so
        # interpolate the production RB rate in log space
        prod = (key[0], key[1], "prod")
        ps = np.asarray(L.meta(prod).probs)[1:]
        rs = np.asarray([rb_rate(prod, i) for i in range(1, len(ps) + 1)])
        return float(np.exp(np.interp(L.meta(key).probs[p_idx], ps, np.log(rs))))
    f = L.fit_fixed_a(np.asarray(L.meta(key).depths), mean, std)
    return f["r"] if f else np.nan


def crossing(ps, r_lrb, r_rb):
    ps, r_lrb, r_rb = map(np.asarray, (ps, r_lrb, r_rb))
    ok = np.isfinite(r_lrb) & np.isfinite(r_rb) & (r_lrb > 0) & (r_rb > 0)
    ps, lr = ps[ok], np.log(r_lrb[ok] / r_rb[ok])
    if len(ps) < 2:
        return np.nan, "too few points"
    worse = lr >= 0
    if not worse[-1]:
        return np.nan, f"no crossing below p={ps[-1]:g}"
    if worse.all():
        return np.nan, f"worse at every p >= {ps[0]:g}"
    i = int(np.max(np.where(~worse)[0]))  # last p where LRB still beats RB
    p0, p1, y0, y1 = ps[i], ps[i + 1], lr[i], lr[i + 1]
    return float(p0 + (0 - y0) * (p1 - p0) / (y1 - y0)), "interp"


def main():
    rows, thr = [], []
    configs = [("one", "sd6", True, "sd6 merged"), ("two", "sd6", True, "sd6 merged"),
               ("one", "si1000", False, "si1000 prod"), ("two", "si1000", False, "si1000 prod"),
               ("one", "si1000", True, "si1000+ext"), ("two", "si1000", True, "si1000+ext")]
    rng = np.random.default_rng(7)
    for anc, noise, use_ext, label in configs:
        grid = merged_grid(anc, noise, use_ext)
        for k in (1, 2, 3, 4):
            per_p = []
            for key, i, p in grid:
                r = lrb_rates(key, i, k)
                r["rb"] = rb_rate(key, i)
                # sequence bootstrap for the pooled-family estimators
                boots = {e: [] for e in ESTIMATORS if e != "pipe"}
                n_seq = L.counts(key, i, "unif", k)[0].shape[1]
                for _ in range(N_BOOT):
                    rb_ = lrb_rates(key, i, k, rng.integers(0, n_seq, n_seq))
                    for e in boots:
                        boots[e].append(rb_[e])
                for e in boots:
                    v = np.asarray(boots[e], float)
                    r[e + "_se"] = float(np.nanstd(v)) if np.isfinite(v).sum() > 10 else np.nan
                    r[e + "_boot"] = v
                per_p.append((p, key, r))
                rows.append({"config": f"{anc}-ancilla-{noise}", "grid": label, "k": k, "p": p, "source": key[2],
                             **{c: r[c] for c in r if not c.endswith("_boot")}})
            ps = [p for p, _, _ in per_p]
            rbs = [r["rb"] for _, _, r in per_p]
            for e in ESTIMATORS:
                vals = [r[e] for _, _, r in per_p]
                pstar, how = crossing(ps, vals, rbs)
                lo = hi = np.nan
                if e != "pipe":
                    bs = []
                    for b in range(N_BOOT):
                        bv = [r[e + "_boot"][b] for _, _, r in per_p]
                        ps_b, _ = crossing(ps, bv, rbs)
                        bs.append(ps_b)
                    bs = np.asarray(bs)
                    if np.isfinite(bs).sum() > 0.8 * N_BOOT:
                        lo, hi = np.nanpercentile(bs, [16, 84])
                thr.append({"config": f"{anc}-ancilla-{noise}", "grid": label, "k": k, "estimator": e,
                            "p_star": pstar, "ci68_lo": lo, "ci68_hi": hi, "note": how})
            print(label, anc, k, {t["estimator"]: round(t["p_star"], 4) if np.isfinite(t["p_star"]) else t["note"]
                                  for t in thr[-len(ESTIMATORS):]}, flush=True)

    def dump(name, data):
        cols = list(dict.fromkeys(c for d in data for c in d))
        with open(os.path.join(L.OUT, "results", name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for d in data:
                w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in d.items()})

    dump("estimator_table.csv", rows)
    dump("pseudo_thresholds_by_estimator.csv", thr)


if __name__ == "__main__":
    main()
