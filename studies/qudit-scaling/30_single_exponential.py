"""Part 2: when is the accepted LRB-D signal a single exponential in depth?

Data: the checkpointed per-sequence tallies of the production and extension /
refinement runs of the eight paper configurations (one-ancilla and two-ancilla
layouts, qutrit and qubit codes, SI1000 and SD6), read-only.

Policies: unif k = 1..6 (postselect the rounds after C_k, C_2k, ... and the
final round; every round is executed), const1 (final round only, all rounds
executed) and const0 (terminal check: no check rounds, direct five-qudit
X readout).

For each (run, p, policy) the pipeline fit is redone exactly as production does
it (per-sequence |sum omega^i P_i|, mean and std over the 30 sequences per
depth, y = b f^m with offset 0, weights 1/std^2, every depth with an accepted
shot in every sequence, m = 0 always included; extension runs use even depths
only for k >= 2, as the production merge does).  Three model checks:

  chi2_sem   reduced chi-square of that fit with the standard error of the
             mean (std / sqrt(30)) instead of the std;
  drift      D = r(all usable depths) / r(depths 0..m3) - 1, where 0..m3 is the
             shortest window (from m = 0) with three usable depths.  Every
             window starts at m = 0 (user rule: m = 0 is in every fit); a
             single exponential gives D = 0;
  window     the fitted rate for every window 0..m_j (for figures);

  grid_dep  G = r(all fitted depths) / r(fitted depths that are multiples of k)
             - 1 for k >= 2 (the staircase: depths that are not multiples of k
             end with an unchecked partial interval); m = 0 is in both fits;
  rel_resid_excess  RMS of (mean - fit)/fit over the fitted depths with the
             statistical part (SEM/fit) removed in quadrature;

and a bootstrap over whole Clifford sequences (shared by all depths) gives the
standard errors of r, D and G.

    python 30_single_exponential.py --reps 200 --workers 8
Writes results/single_exp_rows.csv.gz (one row per run, p, policy).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import qfit  # noqa: F401  (repository paths, plotter)
import merge_highp_threshold_extension as merge  # noqa: E402

from single_exp_configs import CONFIGS  # noqa: E402
POLICIES = [("unif", k) for k in range(1, 7)] + [("const", 1), ("const", 0)]


def pol_from(t):
    t = np.asarray(t, dtype=float)
    d = t.shape[-1]
    w = np.exp(2j * np.pi * np.arange(d) / d)
    n = t.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(t @ w) / n


def fit_rate(plotter, pol, keep, return_resid=False):
    """Pipeline LRB fit; returns (r, f, b, n_points, chi2_sem/dof)."""
    stats = qfit.stats_list(pol, keep)
    fit, (x, y, e) = plotter._fit_lrb_from_stats(stats)
    if fit is None:
        if return_resid:
            e0 = np.array([], float)
            return np.nan, np.nan, np.nan, 0, np.nan, (e0, e0, e0, e0)
        return np.nan, np.nan, np.nan, 0, np.nan
    r = qfit.error_rate(plotter, fit)
    x, y, e = np.asarray(x, float), np.asarray(y, float), np.asarray(e, float)
    n_seq = pol.shape[1]
    sem = np.where(e > 0, e / np.sqrt(n_seq), np.nan)
    model = fit["b"] * np.power(fit["f"], x)
    ok = np.isfinite(sem) & (sem > 0)
    dof = int(ok.sum()) - 2
    chi2 = float(np.sum(((y[ok] - model[ok]) / sem[ok]) ** 2)) / dof if dof > 0 else np.nan
    if return_resid:
        return r, fit["f"], fit["b"], len(x), chi2, (x, y, sem, model)
    return r, fit["f"], fit["b"], len(x), chi2


def usable(pol, keep):
    with np.errstate(invalid="ignore"):
        good = np.all(np.isfinite(pol), axis=1)
    return good & keep


def windows(depths, use):
    """Window end points m_j (depth values) with >= 3 usable depths in 0..m_j."""
    d = np.asarray(depths)
    ends = []
    for m in d[use]:
        if np.sum(use & (d <= m)) >= 3:
            ends.append(int(m))
    return ends


def analyse_cell(plotter, depths, pol, keep, rng, reps, k_grid=None):
    d = np.asarray(depths)
    use = usable(pol, keep)
    r, f, b, npts, chi2, (x, y, sem, model) = fit_rate(plotter, pol, keep, return_resid=True)
    ends = windows(depths, use)
    # excess relative residual: RMS of (y - model)/model after removing the
    # statistical part (SEM/model), over the fitted depths
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (y - model) / model
        rel_sem = sem / model
    good = np.isfinite(rel) & np.isfinite(rel_sem)
    excess = (float(np.sqrt(max(0.0, np.mean(rel[good] ** 2) - np.mean(rel_sem[good] ** 2))))
              if good.sum() > 2 else np.nan)
    out = dict(r=r, f=f, b=b, n_points=npts, chi2_sem=chi2, rel_resid_excess=excess,
               rel_resid=";".join(f"{int(m)}:{v:.4f}" for m, v in zip(x, rel)),
               deepest=int(d[use].max()) if use.any() else -1, windows=len(ends))
    if len(ends) >= 2:
        win_r = [fit_rate(plotter, pol, keep & (d <= m))[0] for m in ends]
        out["drift"] = win_r[-1] / win_r[0] - 1 if win_r[0] > 0 else np.nan
        out["r_short"] = win_r[0]
        out["window_rates"] = ";".join(f"{m}:{x:.4g}" for m, x in zip(ends, win_r))
        m_short = ends[0]
    else:
        out["drift"] = np.nan
        out["r_short"] = np.nan
        out["window_rates"] = ""
        m_short = None
    # grid dependence for k >= 2: rate on all fitted depths vs only multiples of k
    keep_mult = None
    if k_grid and k_grid >= 2:
        km = keep & (d % k_grid == 0)
        if np.sum(usable(pol, km)) >= 2:
            keep_mult = km
            r_mult = fit_rate(plotter, pol, km)[0]
            out["r_mult"] = r_mult
            out["grid_dep"] = r / r_mult - 1 if r_mult > 0 else np.nan
    if reps:
        n = pol.shape[1]
        rs, ds, gs = [], [], []
        for _ in range(reps):
            s = rng.integers(0, n, n)
            pb = pol[:, s]
            rb = fit_rate(plotter, pb, keep)[0]
            rs.append(rb)
            if m_short is not None:
                r0 = fit_rate(plotter, pb, keep & (d <= m_short))[0]
                ds.append(rb / r0 - 1 if r0 > 0 else np.nan)
            if keep_mult is not None:
                rm = fit_rate(plotter, pb, keep_mult)[0]
                gs.append(rb / rm - 1 if rm > 0 else np.nan)
        rs = np.asarray(rs, float)
        out["r_se"] = float(np.nanstd(rs, ddof=1))
        if ds:
            ds = np.asarray(ds, float)
            out["drift_se"] = float(np.nanstd(ds, ddof=1))
        if gs:
            out["grid_dep_se"] = float(np.nanstd(np.asarray(gs, float), ddof=1))
    return out


def run_job(args):
    label, which, run, reps, seed = args
    fam, mlabel, code, layout, noise = CONFIGS[label]
    path = merge.RUNS_ROOT / run
    plotter = merge.plotter_for(run)
    depths = list(plotter.depths)
    probs = [float(p) for p in plotter.raw_probabilities]
    rng = np.random.default_rng(seed)
    dense = depths == [0, 1, 2, 3, 4, 6]
    rows = []
    for idx, p in enumerate(probs):
        if p == 0.0:
            continue
        ck = path / "progress" / str(idx) / ".checkpoint_current"
        for kind, k in POLICIES:
            sub = "unif_check_data" if kind == "unif" else "const_check_data"
            f = ck / sub / f"{k}.npy"
            if not f.exists():
                continue
            t = np.load(f)
            rej = np.load(ck / sub / f"{k}_rejected.npy")
            pol = pol_from(t)
            keep = np.ones(len(depths), dtype=bool)
            if kind == "unif" and k >= 2 and which == "extension" and dense:
                keep = np.asarray(depths) % 2 == 0
            res = analyse_cell(plotter, depths, pol, keep, rng, reps,
                               k_grid=k if kind == "unif" else None)
            acc = t.sum(axis=(1, 2)) / (t.sum(axis=(1, 2)) + rej.sum(axis=1))
            rows.append(dict(config=label, code=code, layout=layout, noise=noise, run_kind=which,
                             run=run, p=p, policy=f"{kind}{k}", kind=kind, k=k,
                             acc=";".join(f"{x:.3g}" for x in acc), **res))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default="results/single_exp_rows.csv.gz")
    a = ap.parse_args()
    jobs = []
    for i, label in enumerate(a.configs.split(",")):
        fam, mlabel = CONFIGS[label][:2]
        prod, ext = merge.CAMPAIGNS[fam]["pairs"][mlabel]
        jobs.append((label, "production", prod, a.reps, 1000 + 2 * i))
        jobs.append((label, "extension", ext, a.reps, 1001 + 2 * i))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_job, jobs):
            rows.extend(r)
            print(f"done {r[0]['config']} {r[0]['run_kind']}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
