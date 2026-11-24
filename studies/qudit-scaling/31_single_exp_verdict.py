"""Part 2 verdicts from results/single_exp_rows.csv.gz.

Rows are merged as in the production chain: production run for p < p_split,
extension / refinement run for p_split <= p <= p_max.

Criterion (stated in REPORT.md), applied to the pipeline fit itself (offset 0,
m = 0 in every fit).  At (configuration, k, p) a single exponential is an
acceptable description when both hold:
  (1) window stability: the fitted rate changes by at most tau between the
      shortest window from m = 0 with three usable depths and the full window,
      |D| - 2 SE(D) <= tau;
  (2) grid stability (k >= 2): the rate fitted on all depths and the rate
      fitted on the depths that are multiples of k agree,
      |G| - 2 SE(G) <= tau;
  The SEM-weighted chi2/dof and the excess relative residual are reported
  next to the verdict but do not enter it: chi2 grows with the shot count
  for any fixed deviation, and the relative residual is dominated by the
  pipeline's |sum| floor wherever the model falls towards zero.
Only p where the rate is resolved (r > 2 SE(r)) count.  A policy is usable in
a configuration when every resolved p in the range p_lo <= p <= p*(k) passes;
the table gives the smallest p_lo for which this holds (p*(4) is used for
k = 5, 6 and the terminal policies, which have no published p*).

Settings tau: strict 0.05, main 0.10, also 0.12, 0.15 and loose 0.20.

Writes results/single_exp_verdicts.csv, results/single_exp_kmax.csv,
results/single_exp_fidelities.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import qfit  # noqa: F401
import merge_highp_threshold_extension as merge

ROWS = "results/single_exp_rows.csv.gz"
ROWS5 = "results/single_exp_rows_five.csv.gz"
CANON = {
    "qutrit-one": ("pseudo-thresholds-current.csv", "one-ancilla-{n}"),
    "qubit-one": ("pseudo-thresholds-qubit-current.csv", "qubit-one-ancilla-{n}"),
    "qutrit-two": ("pseudo-thresholds-layout-b-current.csv", "layout-b-{n}"),
    "qubit-two": ("pseudo-thresholds-qubit-layout-b-current.csv", "qubit-layout-b-{n}"),
    "qutrit-one5": ("pseudo-thresholds-prod5b-current.csv", "prod5b-one-ancilla-{n}"),
    "qubit-one5": ("pseudo-thresholds-qubit-prod5b-current.csv", "qubit-prod5b-one-ancilla-{n}"),
}
SETTINGS = {"strict": 0.05, "main": 0.10, "t12": 0.12, "t15": 0.15, "loose": 0.20}
FID_P = (0.001, 0.002, 0.005, 0.01)


def pstars():
    out = {}
    for key, (fname, pat) in CANON.items():
        t = pd.read_csv(merge.COMPARISONS / fname)
        for noise in ("si1000", "sd6"):
            lab = pat.format(n=noise)
            for r in t[t.configuration == lab].itertuples():
                out[(f"{key}-{noise}", int(r.interval))] = (float(r.p_star), float(r.seq_boot_se))
    return out


def pass_mask(h, tau, tau_r):
    drift_bad = (np.abs(h.drift) - 2 * h.drift_se.fillna(0)) > tau
    grid_bad = (np.abs(h.grid_dep) - 2 * h.grid_dep_se.fillna(0)) > tau
    return ~(drift_bad.fillna(False) | grid_bad.fillna(False))


def reasons(x, tau, tau_r):
    out = []
    if np.isfinite(x.drift) and abs(x.drift) - 2 * (x.drift_se if np.isfinite(x.drift_se) else 0) > tau:
        out.append("window")
    if np.isfinite(x.get("grid_dep", np.nan)) and abs(x.grid_dep) - 2 * (
            x.grid_dep_se if np.isfinite(x.grid_dep_se) else 0) > tau:
        out.append("grid")
    return out


def merged_rows(d):
    from single_exp_configs import CONFIGS  # noqa
    keep = []
    for cfg, g in d.groupby("config"):
        fam, mlabel = CONFIGS[cfg][:2]
        camp = merge.CAMPAIGNS[fam]
        ps, pm = camp["p_split"], camp.get("p_max", np.inf)
        ext = merge.from_extension(g.p.to_numpy(), ps, pm)
        sel = np.where(g.run_kind == "extension", ext, ~ext)
        keep.append(g[sel])
    return pd.concat(keep)


def main():
    d = pd.concat([pd.read_csv(ROWS), pd.read_csv(ROWS5)]) if __import__('os').path.exists(ROWS5) else pd.read_csv(ROWS)
    d = merged_rows(d)
    d["resolved"] = np.isfinite(d.r) & (d.r > 2 * d.r_se)
    ps = pstars()
    verdicts = []
    for (cfg, pol), g in d.groupby(["config", "policy"]):
        k = int(g.k.iloc[0])
        kind = g.kind.iloc[0]
        kk = k if (kind == "unif" and k <= 4) else 4
        pstar = ps[(cfg, kk)][0]
        h = g[(g.p <= pstar) & g.resolved].sort_values("p")
        top = h[h.p >= 0.5 * pstar]
        low = h[h.p < 0.5 * pstar]
        def excess(x, col, se):
            v = (np.abs(x[col]) - 2 * x[se].fillna(0)) if col in x else pd.Series(dtype=float)
            return float(np.nanmax(v)) if len(v) and np.isfinite(v).any() else np.nan
        row_extra = dict(
            top_max_window=excess(top, "drift", "drift_se"), low_max_window=excess(low, "drift", "drift_se"),
            top_max_grid=excess(top, "grid_dep", "grid_dep_se"), low_max_grid=excess(low, "grid_dep", "grid_dep_se"),
            top_median_abs_drift=float(np.nanmedian(np.abs(top.drift))) if len(top) else np.nan,
            top_median_abs_grid=float(np.nanmedian(np.abs(top.grid_dep))) if len(top) and "grid_dep" in top else np.nan,
            top_median_chi2=float(np.nanmedian(top.chi2_sem)) if len(top) else np.nan)
        row = dict(config=cfg, code=g.code.iloc[0], layout=g.layout.iloc[0], noise=g.noise.iloc[0],
                   policy=pol, kind=kind, k=k, p_star_range=pstar, n_p=len(h),
                   max_abs_drift=float(np.nanmax(np.abs(h.drift))) if len(h) else np.nan,
                   max_chi2_sem=float(np.nanmax(h.chi2_sem)) if len(h) else np.nan,
                   median_chi2_sem=float(np.nanmedian(h.chi2_sem)) if len(h) else np.nan, **row_extra)
        for name, tau in SETTINGS.items():
            tau_r = None
            ok = pass_mask(h, tau, tau_r)
            row[f"pass_frac_{name}"] = float(ok.mean()) if len(h) else np.nan
            bad = h[~ok]
            # smallest p_lo such that every resolved p in [p_lo, p*] passes
            p_lo = float(h[h.p > bad.p.max()].p.min()) if len(bad) else (float(h.p.min()) if len(h) else np.nan)
            row[f"usable_all_{name}"] = bool(ok.all()) if len(h) else False
            row[f"usable_from_p_{name}"] = p_lo
            row[f"n_pass_top_{name}"] = int((h.p >= p_lo).sum()) if np.isfinite(p_lo) else 0
            row[f"fail_reasons_{name}"] = ";".join(sorted(set(
                r for _, x in bad.iterrows() for r in reasons(x, tau, tau_r))))
        verdicts.append(row)
    v = pd.DataFrame(verdicts)
    v.to_csv("results/single_exp_verdicts.csv", index=False)

    # largest usable k: every k' <= k usable on every resolved p <= p*(k')
    lines = []
    for cfg, g in v[v.kind == "unif"].groupby("config"):
        out = dict(config=cfg)
        for name in SETTINGS:
            kmax = 0
            for k in range(1, 7):
                r = g[g.k == k]
                if len(r) and bool(r[f"usable_all_{name}"].iloc[0]):
                    kmax = k
                else:
                    break
            out[f"k_max_all_p_{name}"] = kmax
            # same, restricted to the near-threshold half: p >= p*(k)/2
            kmax2 = 0
            for k in range(1, 7):
                r = g[g.k == k]
                if len(r) and r[f"usable_from_p_{name}"].iloc[0] <= 0.5 * r["p_star_range"].iloc[0]:
                    kmax2 = k
                else:
                    break
            out[f"k_max_upper_half_{name}"] = kmax2
        lines.append(out)
    km = pd.DataFrame(lines)
    km.to_csv("results/single_exp_kmax.csv", index=False)

    # fidelities F = 1 - r per forward+inverse pair (pipeline unit)
    fid = []
    for (cfg, pol), g in d[d.kind == "unif"].groupby(["config", "policy"]):
        for p in FID_P:
            r = g[np.isclose(g.p, p)]
            if not len(r):
                continue
            r = r.iloc[0]
            fid.append(dict(config=cfg, policy=pol, k=int(r.k), p=p, r=r.r, r_se=r.r_se,
                            F=1 - r.r, chi2_sem=r.chi2_sem, drift=r.drift, drift_se=r.get("drift_se", np.nan),
                            grid_dep=r.get("grid_dep", np.nan), grid_dep_se=r.get("grid_dep_se", np.nan),
                            rel_resid_excess=r.rel_resid_excess))
    pd.DataFrame(fid).to_csv("results/single_exp_fidelities.csv", index=False)
    pd.set_option("display.width", 250)
    print(v[["config", "policy", "p_star_range", "n_p", "top_max_window", "top_max_grid",
             "low_max_window", "low_max_grid", "top_median_chi2",
             "usable_all_main", "usable_from_p_main", "fail_reasons_main", "usable_from_p_strict",
             "usable_from_p_loose"]].round(3).to_string())
    print(km.to_string())


if __name__ == "__main__":
    main()
