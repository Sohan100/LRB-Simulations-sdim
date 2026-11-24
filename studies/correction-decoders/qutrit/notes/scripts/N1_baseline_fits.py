#!/usr/bin/env python3
"""N1: non-exponential and history-dependent behaviour in the LRB-D baseline.

Input: only the result CSVs of the baseline run (read-only)

    <RUN>/results/LRB/<p_idx>/{unif,const}_check_data/<k>.csv
    <RUN>/results/RB/<p_idx>.csv            (physical RB, p_RB = p)
    <RUN>/results/RB_comparison/<p_idx>.csv (physical RB, p_RB = 5p)
    <RUN>/results/plots/unif_lrb_vs_rb_table_all_mixed_fits.csv (cross-check)

plus the small run metadata files (depths.txt, probs.txt, ...). No DEM, no
circuit simulation. p index 0 (p = 0) is skipped.

What it does, for every p index 1..14 and every policy (unif 1-22, const 0-22)
and for physical RB (RB and RB_comparison):

 (1) Reproduces the pipeline fit y = b f^m (offset fixed at 0) of
     src/lrb/lrb_plotting.py: _mask_invalid, _apply_ceiling_cut (keep y <= 1,
     LRB only), _safe_weights (w = 1/std^2, std <= 0 replaced by
     max(1e-3 * median(std), 1e-12)), 5-pass 2001-point grid scan over
     f in [1e-8, 1]. Reports f, b, chi2 and F = (1 + 2 f)/3, and checks the
     result against the pipeline's own table.
 (2) Fits alternatives with sigma = SEM = std/sqrt(30) on "usable" points:
       A  b f^m                    (pipeline model, refitted with SEM)
       B  a + b f^m                (free offset)
       C  b1 f1^m + b2 f2^m        (double exponential; includes A and B)
       D  b exp(-c m - d m^2)      (quadratic term in the exponent)
       E  e0 + e1 m + e2 m^2       (quadratic infidelity, 1 - y = b0 + b1 m + b2 m^2)
       A0, E0: A and E fitted to m > 0 only.
     chi2, dof, goodness-of-fit p, AIC, AICc, BIC, likelihood-ratio tests.
 (3) Depth-windowed rates f_i = (y_{i+1}/y_i)^(1/dm) with error bars, the
     per-Clifford error r_i = 2(1 - f_i)/3, and per-step acceptance
     a_i = (acc_{i+1}/acc_i)^(1/dm), acc = 1 - rejected.
 (4) GLS trend tests of ln f_i and ln a_i against the window midpoint, using
     the exact covariance of the differences; checks the identity
     "window slope = 2 x log-quadratic curvature".
 (5) Signatures: const 1 quadratic infidelity, unif 1 flat fidelity versus
     decaying acceptance, the unif-1 bound on pairs inside the inverse half,
     const 0, check-spacing scaling (unif k per-step rate, unchecked-tail
     cost, const c curvature), windows in which the fidelity rises.
 (6) The same windowed analysis for physical RB as a Markovian reference.

Outputs go to docs/correction-decoders/qutrit-512/results/baseline_nonmarkov/:
  pipeline_fits.csv, pipeline_table_check.csv   (1) pipeline f, b, F, chi2
  model_fits.csv, deviation_summary.csv,
  deviation_by_policy.csv, points.csv           (2) models A-E, tests, forms
  windowed_rates.csv, trend_tests.csv           (3)/(4)
  signatures_unif1_const1_const0.csv,
  attribution_and_scaling.csv, check_spacing_unif.csv,
  check_spacing_const.csv, tail_cost_m2.csv, segment_model.csv,
  acceptance_additivity.csv, tail_growth_regression.csv,
  rising_windows.csv                            (5)
  covariance_sensitivity.csv, m0_shot_noise_check.csv,
  identity_checks.csv                           checks of the statistics
  windowed_rates.png, acceptance_hazard.png, residuals_p1e-3.png,
  curvature_map.png, spacing_and_quadratic.png  figures

Usage:  module load python/3.11 && python N1_baseline_fits.py [--no-plots]
Runtime: about one minute on one core.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"                     # login node: one core

import numpy as np  # noqa: E402
from scipy import optimize, stats  # noqa: E402

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(os.path.dirname(HERE))           # .../qutrit-512
OUT = os.path.join(STUDY, "results", "baseline_nonmarkov")

RB_SHOTS = 10_000          # NORMAL_RB_SHOTS in src/lrb/lrb_simulation.py
NACC_MIN = 1000            # usable point: mean accepted shots per sequence
RAYLEIGH_FACTOR = 10.0     # usable point: y >= 10 x sqrt(pi / (4 N_acc))
ALPHA = 1e-3               # nominal per-test significance level
Z_ALPHA = stats.norm.isf(ALPHA / 2)   # 3.29


# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------
def _read_list(name, cast):
    with open(os.path.join(RUN, name)) as fh:
        txt = fh.read().replace(",", " ").split()
    return [cast(v) for v in txt]


DEPTHS = np.array(_read_list("depths.txt", int), dtype=float)
PROBS = _read_list("probs.txt", float)
RBC_PROBS = _read_list("rb_comparison_probs.txt", float)
UNIF = _read_list("check_unif.txt", int)
CONST = _read_list("check_const.txt", int)
NSEQ = int(_read_list("num_cliffs.txt", float)[0])
SHOTS = int(_read_list("shots.txt", float)[0])
P_IDX = list(range(1, len(PROBS)))              # skip p = 0


def read_stats(path):
    """Five-row result CSV -> (prob, y, std, rejected, rejected_std)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    prob = float(rows[0][1])
    arr = [np.array([float(v) for v in r[1:]]) for r in rows[1:5]]
    return prob, arr[0], arr[1], arr[2], arr[3]


def load_curves():
    curves = []
    for pidx in P_IDX:
        for kind, ks in (("unif", UNIF), ("const", CONST)):
            sub = "unif_check_data" if kind == "unif" else "const_check_data"
            for k in ks:
                path = os.path.join(RUN, "results", "LRB", str(pidx), sub, f"{k}.csv")
                prob, y, sd, rj, rsd = read_stats(path)
                assert abs(prob - PROBS[pidx]) < 1e-15, path
                curves.append(dict(family="LRB", kind=kind, k=k, policy=f"{kind}{k}",
                                   pidx=pidx, p=PROBS[pidx], p_phys=PROBS[pidx],
                                   m=DEPTHS.copy(), y=y, std=sd, rej=rj, rej_std=rsd,
                                   shots=SHOTS))
        for fam, sub, plist in (("RB", "RB", PROBS), ("RBC", "RB_comparison", RBC_PROBS)):
            path = os.path.join(RUN, "results", sub, f"{pidx}.csv")
            prob, y, sd, rj, rsd = read_stats(path)
            assert abs(prob - plist[pidx]) < 1e-12, path
            curves.append(dict(family=fam, kind="rb", k=0, policy=fam, pidx=pidx,
                               p=PROBS[pidx], p_phys=plist[pidx], m=DEPTHS.copy(),
                               y=y, std=sd, rej=rj, rej_std=rsd, shots=RB_SHOTS))
    return curves


# ----------------------------------------------------------------------------
# policy structure (which already-executed check rounds are used)
# ----------------------------------------------------------------------------
def used_rounds(kind, k, m):
    """Intermediate rounds used (terminal round m always used). Mirrors
    LRBSimulationPipeline.postselect_uniform_interval / postselect_constant_number."""
    m = int(m)
    if kind == "unif":
        if k > 0 and m >= k:
            return [j * k - 1 for j in range(1, m // k + 1) if j * k - 1 < m]
        return []
    if kind == "const":
        if k == 0:
            return None                      # const 0: no ancilla rounds at all
        if k <= m:
            step = m // k
            return [j * step - 1 for j in range(1, k)]
        return []
    return None


def structure(kind, k, m):
    """Closed forward segments, unchecked forward tail r (forward Cliffords after
    the last used intermediate round), number of used intermediate rounds."""
    used = used_rounds(kind, k, m)
    if used is None:
        return dict(n_used=np.nan, tail=np.nan, segments="", s2=np.nan)
    edges = [-1] + sorted(used)
    segs = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    tail = int(m) - (edges[-1] + 1)
    return dict(n_used=len(used), tail=tail, segments=" ".join(map(str, segs)),
                s2=float(sum(s * s for s in segs)))


# ----------------------------------------------------------------------------
# (1) pipeline fit, mirrored from src/lrb/lrb_plotting.py
# ----------------------------------------------------------------------------
def safe_weights(errors):
    """LRBResultsPlotter._safe_weights."""
    err = np.asarray(errors, dtype=float)
    err = np.where(np.isnan(err), np.inf, err)
    finite = err[np.isfinite(err)]
    median = float(np.median(finite)) if len(finite) else 1.0
    floor = max(median * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    return 1.0 / (err * err)


def pipeline_points(m, y, e, ceiling):
    keep = ~(np.isnan(y) | np.isnan(e))                       # _mask_invalid
    x, yy, ee = m[keep], y[keep], e[keep]
    if ceiling is not None:                                   # _apply_ceiling_cut
        keep = yy <= ceiling
        x, yy, ee = x[keep], yy[keep], ee[keep]
    return x, yy, ee


def pipeline_fit_fixed_a(x, y, e, a_fixed=0.0, f_min=1e-8, f_max=1.0, min_points=2):
    """LRBResultsPlotter._fit_decay_parameter_fixed_a."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < min_points:
        return None
    w = safe_weights(e)
    y0 = y - float(a_fixed)
    low, high = float(f_min), float(f_max)
    best = None
    for _ in range(5):
        f_grid = np.linspace(low, high, 2001)
        z = np.power(f_grid[None, :], x[:, None])
        num = np.sum(w[:, None] * z * y0[:, None], axis=0)
        den = np.sum(w[:, None] * z * z, axis=0)
        den = np.where(den <= 0, np.nan, den)
        b = num / den
        resid = y0[:, None] - b[None, :] * z
        chi2 = np.sum(w[:, None] * resid * resid, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        idx = int(np.argmin(chi2))
        f_best = float(f_grid[idx])
        best = dict(f=f_best, a=float(a_fixed), b=float(b[idx]), chi2=float(chi2[idx]),
                    n_points=len(x))
        span = (high - low) / 10.0
        low = max(f_min, f_best - span)
        high = min(f_max, f_best + span)
    return best


def pipeline_fit_free_a(x, y, e, f_min=1e-8, f_max=1.0, min_points=3):
    """LRBResultsPlotter._fit_decay_parameter_free_a (code default for RB)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < min_points:
        return None
    w = safe_weights(e)
    s, sy = float(np.sum(w)), float(np.sum(w * y))
    low, high = float(f_min), float(f_max)
    best = None
    for _ in range(5):
        f_grid = np.linspace(low, high, 2001)
        z = np.power(f_grid[None, :], x[:, None])
        sz = np.sum(w[:, None] * z, axis=0)
        szz = np.sum(w[:, None] * z * z, axis=0)
        szy = np.sum(w[:, None] * z * y[:, None], axis=0)
        det = s * szz - sz * sz
        det = np.where(np.abs(det) < 1e-20, np.nan, det)
        a = (szz * sy - sz * szy) / det
        b = (s * szy - sz * sy) / det
        resid = y[:, None] - (a[None, :] + b[None, :] * z)
        chi2 = np.sum(w[:, None] * resid * resid, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        idx = int(np.argmin(chi2))
        f_best = float(f_grid[idx])
        best = dict(f=f_best, a=float(a[idx]), b=float(b[idx]), chi2=float(chi2[idx]),
                    n_points=len(x))
        span = (high - low) / 10.0
        low = max(f_min, f_best - span)
        high = min(f_max, f_best + span)
    return best


def F_of_f(f):
    return (1.0 + 2.0 * f) / 3.0


# ----------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------
def usable_points(c):
    """Points used for the SEM-based analysis."""
    nacc = (1.0 - c["rej"]) * c["shots"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ray = np.sqrt(np.pi / (4.0 * np.maximum(nacc, 1.0)))
    ok = (np.isfinite(c["y"]) & np.isfinite(c["std"]) & (nacc >= NACC_MIN)
          & (c["y"] >= RAYLEIGH_FACTOR * ray))
    return ok, nacc, ray


def sem_sigma(c, nacc):
    """SEM of the 30-sequence mean, floored at 1.5/(30 N_acc) (one logical-error
    event in the pooled sample; only matters where std = 0)."""
    sem = c["std"] / math.sqrt(NSEQ)
    floor = 1.5 / (NSEQ * np.maximum(nacc, 1.0))
    return np.maximum(np.nan_to_num(sem, nan=np.inf), floor), sem


def wls(X, y, s):
    w = 1.0 / s
    Xw, yw = X * w[:, None], y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    r = yw - Xw @ coef
    try:
        cov = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Xw.T @ Xw)
    return coef, cov, float(r @ r)


def gls(X, v, C):
    L = np.linalg.cholesky(C)
    Xw = np.linalg.solve(L, X)
    vw = np.linalg.solve(L, v)
    coef, *_ = np.linalg.lstsq(Xw, vw, rcond=None)
    r = vw - Xw @ coef
    cov = np.linalg.inv(Xw.T @ Xw)
    return coef, cov, float(r @ r)


def info_criteria(chi2, n, k):
    aic = chi2 + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    bic = chi2 + k * math.log(n)
    return aic, aicc, bic


def gof(chi2, dof):
    if dof <= 0 or not np.isfinite(chi2):
        return np.nan, np.nan
    return float(stats.chi2.sf(chi2, dof)), float(stats.chi2.logsf(chi2, dof) / math.log(10))


def lrt(chi2_small, chi2_big, ddof):
    if not (np.isfinite(chi2_small) and np.isfinite(chi2_big)):
        return np.nan, np.nan
    d = max(chi2_small - chi2_big, 0.0)
    return float(stats.chi2.sf(d, ddof)), d


def _model_record(name, m, y, s, pred, params, errors, k):
    n = len(m)
    resid = (y - pred) / s
    chi2 = float(resid @ resid)
    dof = n - k
    p, log10p = gof(chi2, dof)
    aic, aicc, bic = info_criteria(chi2, n, k)
    return dict(model=name, n=n, k=k, chi2=chi2, dof=dof, p_gof=p, log10_p_gof=log10p,
                aic=aic, aicc=aicc, bic=bic, params=params, errors=errors, pred=pred,
                max_abs_resid=float(np.max(np.abs(resid))) if n else np.nan)


def _lsq(fun, x0):
    return optimize.least_squares(fun, x0, method="lm", xtol=1e-15, ftol=1e-15,
                                  gtol=1e-15, max_nfev=50000)


def _cov_from_jac(jac):
    try:
        return np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return np.full((jac.shape[1], jac.shape[1]), np.nan)


def fit_A(m, y, s):
    """y = b f^m."""
    n = len(m)
    if n < 3:
        return None
    coef, _, _ = wls(np.column_stack([np.ones(n), m]), np.log(y), s / y)

    def fun(x):
        return (y - np.exp(x[0] + x[1] * m)) / s
    res = _lsq(fun, coef.copy())
    cov = _cov_from_jac(res.jac)
    lnb, lnf = res.x
    b, f = math.exp(lnb), math.exp(lnf)
    pred = b * f ** m
    se = np.sqrt(np.abs(np.diag(cov)))
    return _model_record("A", m, y, s, pred, dict(b=b, f=f, F=F_of_f(f), r=1 - F_of_f(f)),
                         dict(b=b * se[0], f=f * se[1], r=2 / 3 * f * se[1]), 2)


def fit_D(m, y, s, start_A=None):
    """y = b exp(-c m - d m^2); d > 0 means the decay accelerates with m."""
    n = len(m)
    if n < 4:
        return None
    coef, _, _ = wls(np.column_stack([np.ones(n), m, m * m]), np.log(y), s / y)
    starts = [np.array([coef[0], -coef[1], -coef[2]])]
    if start_A is not None:
        starts.append(np.array([math.log(start_A["params"]["b"]),
                                -math.log(start_A["params"]["f"]), 0.0]))

    def fun(x):
        return (y - np.exp(x[0] - x[1] * m - x[2] * m * m)) / s
    best = None
    for x0 in starts:
        res = _lsq(fun, x0)
        if best is None or res.cost < best.cost:
            best = res
    cov = _cov_from_jac(best.jac)
    lnb, c_, d_ = best.x
    se = np.sqrt(np.abs(np.diag(cov)))
    pred = np.exp(lnb - c_ * m - d_ * m * m)
    return _model_record("D", m, y, s, pred,
                         dict(b=math.exp(lnb), c=c_, d=d_),
                         dict(b=math.exp(lnb) * se[0], c=se[1], d=se[2]), 3)


def fit_E(m, y, s, name="E"):
    """y = e0 + e1 m + e2 m^2, reported as 1 - y = b0 + b1 m + b2 m^2."""
    n = len(m)
    if n < 4:
        return None
    X = np.column_stack([np.ones(n), m, m * m])
    coef, cov, _ = wls(X, y, s)
    se = np.sqrt(np.abs(np.diag(cov)))
    pred = X @ coef
    return _model_record(name, m, y, s, pred,
                         dict(b0=1 - coef[0], b1=-coef[1], b2=-coef[2]),
                         dict(b0=se[0], b1=se[1], b2=se[2]), 3)


_FG = np.concatenate([[0.0], 1.0 - np.logspace(0, -9, 600)[1:],
                      1.0 + np.logspace(-9, math.log10(0.3), 250)])
F_GRID_B = np.unique(_FG[(_FG != 1.0)])
_FC = np.concatenate([[0.0, 1.0], 1.0 - np.logspace(0, -8, 160)[1:],
                      1.0 + np.logspace(-8, math.log10(0.3), 70)])
F_GRID_C = np.unique(_FC)


def _profile_B(f, m, y, w):
    z = np.power(f, m)
    X = np.column_stack([np.ones_like(m), z])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = (y - X @ coef) * sw
    return float(r @ r), coef


def fit_B(m, y, s):
    """y = a + b f^m (profile least squares over f)."""
    n = len(m)
    if n < 4:
        return None
    w = 1.0 / s ** 2
    Z = np.power(F_GRID_B[None, :], m[:, None])
    S0, S1, S2 = w.sum(), (w[:, None] * Z).sum(0), (w[:, None] * Z * Z).sum(0)
    T0, T1 = (w * y).sum(), (w[:, None] * Z * y[:, None]).sum(0)
    det = S0 * S2 - S1 * S1
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (S2 * T0 - S1 * T1) / det
        b = (S0 * T1 - S1 * T0) / det
        chi2 = (w[:, None] * (y[:, None] - a[None, :] - b[None, :] * Z) ** 2).sum(0)
    chi2 = np.where(np.isfinite(chi2) & (np.abs(det) > 1e-12 * S0 * S2), chi2, np.inf)
    order = np.argsort(chi2)[:5]
    best_f, best_chi2 = None, np.inf
    for i in order:
        lo = F_GRID_B[max(i - 1, 0)]
        hi = F_GRID_B[min(i + 1, len(F_GRID_B) - 1)]
        if hi > lo:
            res = optimize.minimize_scalar(lambda f: _profile_B(f, m, y, w)[0],
                                           bounds=(lo, hi), method="bounded",
                                           options=dict(xatol=(hi - lo) * 1e-7))
            cand = [(res.fun, res.x), (_profile_B(F_GRID_B[i], m, y, w)[0], F_GRID_B[i])]
        else:
            cand = [(_profile_B(F_GRID_B[i], m, y, w)[0], F_GRID_B[i])]
        for val, f in cand:
            if val < best_chi2:
                best_chi2, best_f = val, float(f)
    _, (a_, b_) = _profile_B(best_f, m, y, w)
    errors = dict(a=np.nan, b=np.nan, f=np.nan)
    if 1e-6 < best_f < 1.29 and abs(best_f - 1.0) > 1e-9:
        def fun(x):
            return (y - x[0] - x[1] * np.power(x[2], m)) / s
        res = _lsq(fun, np.array([a_, b_, best_f]))
        if 2 * res.cost <= best_chi2 * (1 + 1e-9) + 1e-12 and 0 < res.x[2]:
            a_, b_, best_f = res.x
        se = np.sqrt(np.abs(np.diag(_cov_from_jac(res.jac))))
        errors = dict(a=se[0], b=se[1], f=se[2])
    pred = a_ + b_ * np.power(best_f, m)
    return _model_record("B", m, y, s, pred, dict(a=a_, b=b_, f=best_f), errors, 3)


def _profile_C(f1, f2, m, y, w):
    X = np.column_stack([np.power(f1, m), np.power(f2, m)])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = (y - X @ coef) * sw
    return float(r @ r), coef


def fit_C(m, y, s, nested=()):
    """y = b1 f1^m + b2 f2^m (2-D profile over f1 < f2, then local refinement).
    `nested` holds (f1, f2) candidates that reproduce fits of models A and B."""
    n = len(m)
    if n < 5:
        return None
    w = 1.0 / s ** 2
    Z = np.power(F_GRID_C[None, :], m[:, None])               # n x G
    Zw = Z * w[:, None]
    S = Z.T @ Zw                                             # G x G
    T = Zw.T @ y                                             # G
    d = np.diag(S)
    det = d[:, None] * d[None, :] - S * S
    num = d[None, :] * T[:, None] ** 2 - 2 * S * T[:, None] * T[None, :] + d[:, None] * T[None, :] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (w * y * y).sum() - num / det
    G = len(F_GRID_C)
    iu = np.triu_indices(G, 1)
    vals = chi2[iu]
    good = np.isfinite(vals) & (det[iu] > 1e-10 * (d[:, None] * d[None, :])[iu])
    vals = np.where(good, vals, np.inf)
    cands = [(F_GRID_C[iu[0][j]], F_GRID_C[iu[1][j]]) for j in np.argsort(vals)[:12]]
    cands += list(nested)
    best = (np.inf, None)
    for f1, f2 in cands:
        val, _ = _profile_C(f1, f2, m, y, w)
        if val < best[0]:
            best = (val, (f1, f2))

    def obj(x):
        return _profile_C(min(max(x[0], 0.0), 1.3), min(max(x[1], 0.0), 1.3), m, y, w)[0]
    res = optimize.minimize(obj, np.array(best[1]), method="Nelder-Mead",
                            options=dict(xatol=1e-12, fatol=1e-10, maxiter=4000))
    if res.fun < best[0]:
        best = (float(res.fun), (min(max(res.x[0], 0.0), 1.3), min(max(res.x[1], 0.0), 1.3)))
    f1, f2 = best[1]
    if f1 > f2:
        f1, f2 = f2, f1
    _, (b1, b2) = _profile_C(f1, f2, m, y, w)
    pred = b1 * np.power(f1, m) + b2 * np.power(f2, m)
    return _model_record("C", m, y, s, pred, dict(b1=b1, f1=f1, b2=b2, f2=f2),
                         dict(b1=np.nan, f1=np.nan, b2=np.nan, f2=np.nan), 4)


# ----------------------------------------------------------------------------
# (3)/(4) windowed rates and GLS trend tests
# ----------------------------------------------------------------------------
def difference_operator(m):
    n = len(m)
    dm = np.diff(m)
    D = np.zeros((n - 1, n))
    D[np.arange(n - 1), np.arange(n - 1)] = -1.0 / dm
    D[np.arange(n - 1), np.arange(1, n)] = 1.0 / dm
    return D


def windows_and_trend(m, v, sv):
    """v = ln(level) at depths m with independent errors sv. Returns the window
    values ln f_i = (v_{i+1} - v_i)/dm, their sds, and GLS tests (exact
    tridiagonal covariance D S D^T) of a constant and of a linear trend in the
    window midpoint."""
    D = difference_operator(m)
    wv = D @ v
    C = D @ np.diag(sv ** 2) @ D.T
    swv = np.sqrt(np.diag(C))
    mid = 0.5 * (m[:-1] + m[1:])
    out = dict(mid=mid, lnf=wv, sd=swv, m_lo=m[:-1], m_hi=m[1:])
    nw = len(wv)
    out.update(n_windows=nw, slope=np.nan, slope_se=np.nan, slope_z=np.nan,
               chi2_const=np.nan, dof_const=nw - 1, p_const=np.nan,
               chi2_lin=np.nan, dof_lin=nw - 2, p_lin=np.nan, intercept=np.nan)
    out.update(const=np.nan, const_se=np.nan)
    if nw >= 1:
        coef, cov, chi2c = gls(np.ones((nw, 1)), wv, C)
        out.update(const=coef[0], const_se=math.sqrt(cov[0, 0]))
        if nw >= 2:
            out.update(chi2_const=chi2c, p_const=gof(chi2c, nw - 1)[0])
    if nw >= 3:
        X = np.column_stack([np.ones(nw), mid])
        coef, cov, chi2l = gls(X, wv, C)
        out.update(slope=coef[1], slope_se=math.sqrt(cov[1, 1]),
                   slope_z=coef[1] / math.sqrt(cov[1, 1]), intercept=coef[0],
                   chi2_lin=chi2l, p_lin=gof(chi2l, nw - 2)[0])
    return out


def reml_identity_check(m, v, sv, tr):
    """Window GLS vs level WLS: slope = 2 beta2 of ln y = b0 + b1 m + b2 m^2,
    constant-rate chi2 = chi2 of the log-linear fit, linear-trend chi2 = chi2 of
    the log-quadratic fit."""
    n = len(m)
    if n < 4:
        return np.nan
    _, _, chi_lin = wls(np.column_stack([np.ones(n), m]), v, sv)
    cq, _, chi_quad = wls(np.column_stack([np.ones(n), m, m * m]), v, sv)
    devs = [abs(tr["slope"] - 2 * cq[2]) / max(abs(2 * cq[2]), 1e-300),
            abs(tr["chi2_const"] - chi_lin) / max(chi_lin, 1e-300),
            abs(tr["chi2_lin"] - chi_quad) / max(chi_quad, 1e-300)]
    return float(max(devs))


# ----------------------------------------------------------------------------
# per-cell analysis
# ----------------------------------------------------------------------------
def analyse_curve(c):
    m_all = c["m"]
    res = dict(curve=c)
    # (1) pipeline fit
    ceiling = 1.0 if c["family"] == "LRB" else None
    x, yy, ee = pipeline_points(m_all, c["y"], c["std"], ceiling)
    pf = pipeline_fit_fixed_a(x, yy, ee)
    res["pipe"] = pf
    if pf is not None:
        # same fit with SEM weights: f and b must be identical, chi2 x 30
        pf_sem = pipeline_fit_fixed_a(x, yy, ee / math.sqrt(NSEQ))
        res["pipe_sem"] = pf_sem
    else:
        res["pipe_sem"] = None
    if c["family"] != "LRB":
        res["pipe_free"] = pipeline_fit_free_a(*pipeline_points(m_all, c["y"], c["std"], None))
    # (2) alternative fits on usable points with SEM
    ok, nacc, ray = usable_points(c)
    sig, sem = sem_sigma(c, nacc)
    res.update(usable=ok, nacc=nacc, rayleigh=ray, sigma=sig, sem=sem)
    m, y, s = m_all[ok], c["y"][ok], sig[ok]
    fits = {}
    fits["A"] = fit_A(m, y, s)
    fits["D"] = fit_D(m, y, s, fits["A"])
    fits["E"] = fit_E(m, y, s)
    fits["B"] = fit_B(m, y, s)
    nested = []
    if fits["A"] is not None:
        nested.append((fits["A"]["params"]["f"], 0.0))
        nested.append((fits["A"]["params"]["f"], 1.0))
    if fits["B"] is not None:
        nested.append((min(fits["B"]["params"]["f"], 1.0), 1.0))
    fits["C"] = fit_C(m, y, s, nested)
    pos = m > 0
    fits["A0"] = fit_A(m[pos], y[pos], s[pos])
    if fits["A0"] is not None:
        fits["A0"]["model"] = "A0"
    fits["E0"] = fit_E(m[pos], y[pos], s[pos], name="E0")
    res["fits"] = fits
    # m = 0 boundary test: A0 extrapolated to m = 0
    res["z_m0"] = np.nan
    if fits["A0"] is not None and ok[0] and m_all[0] == 0:
        b0, sb0 = fits["A0"]["params"]["b"], fits["A0"]["errors"]["b"]
        res["z_m0"] = (c["y"][0] - b0) / math.hypot(sig[0], sb0)
        res["m0_offset"] = c["y"][0] - b0
        res["m0_offset_sd"] = math.hypot(sig[0], sb0)
    # (3)/(4) windowed fidelity
    if ok.sum() >= 2:
        v, sv = np.log(y), s / y
        tr = windows_and_trend(m, v, sv)
        tr["identity_dev"] = reml_identity_check(m, v, sv, tr) if len(m) >= 4 else np.nan
        res["win_fid"] = tr
        if (m > 0).sum() >= 4:
            p0 = m > 0
            res["win_fid_m0excl"] = windows_and_trend(m[p0], v[p0], sv[p0])
    # windowed acceptance (LRB only)
    if c["family"] == "LRB":
        acc = 1.0 - c["rej"]
        sacc = np.maximum(c["rej_std"] / math.sqrt(NSEQ), 1.0 / (NSEQ * c["shots"]))
        oka = np.isfinite(acc) & (acc * c["shots"] >= 100)
        if oka.sum() >= 2:
            ma = m_all[oka]
            va, sva = np.log(acc[oka]), sacc[oka] / acc[oka]
            tra = windows_and_trend(ma, va, sva)
            res["win_acc"] = tra
            p0 = ma > 0
            if p0.sum() >= 3:
                res["win_acc_m0excl"] = windows_and_trend(ma[p0], va[p0], sva[p0])
    return res


def deviation_record(r):
    c, f = r["curve"], r["fits"]
    A, B, C, D, E = f["A"], f["B"], f["C"], f["D"], f["E"]
    rec = dict(family=c["family"], policy=c["policy"], kind=c["kind"], k=c["k"],
               p_idx=c["pidx"], p=c["p"], p_phys=c["p_phys"],
               n_usable=int(r["usable"].sum()),
               m_usable_max=float(c["m"][r["usable"]].max()) if r["usable"].any() else np.nan)
    pf = r["pipe"]
    rec["F_pipeline"] = F_of_f(pf["f"]) if pf else np.nan
    rec["r_pipeline"] = 1 - F_of_f(pf["f"]) if pf else np.nan
    if A is None:
        rec["tested"] = False
        return rec
    rec["tested"] = True
    rec.update(chi2_A=A["chi2"], dof_A=A["dof"], p_A=A["p_gof"], log10p_A=A["log10_p_gof"],
               r_A=A["params"]["r"], r_A_sd=A["errors"]["r"],
               max_resid_A=A["max_abs_resid"])
    last = int(np.argmax(r["curve"]["m"][r["usable"]]))
    yu = c["y"][r["usable"]]
    su = r["sigma"][r["usable"]]
    rec["miss_last_A"] = float(yu[last] - A["pred"][last])
    rec["miss_last_A_sigma"] = float((yu[last] - A["pred"][last]) / su[last])
    for name, mod in (("B", B), ("C", C), ("D", D), ("E", E)):
        if mod is None:
            continue
        rec[f"chi2_{name}"] = mod["chi2"]
        rec[f"p_{name}"] = mod["p_gof"]
    if D is not None:
        pAD, dAD = lrt(A["chi2"], D["chi2"], 1)
        rec.update(d=D["params"]["d"], d_sd=D["errors"]["d"], p_AD=pAD,
                   z_curv=math.copysign(math.sqrt(dAD), D["params"]["d"]),
                   c_D=D["params"]["c"])
        mlast = float(c["m"][r["usable"]].max())
        cD, dD = D["params"]["c"], D["params"]["d"]
        tot = cD * mlast + dD * mlast ** 2
        # share of -ln(y(m_last)/b) carried by the quadratic term of model D
        rec["quad_share_D"] = dD * mlast ** 2 / tot if tot != 0 else np.nan
    if B is not None:
        pAB, dAB = lrt(A["chi2"], B["chi2"], 1)
        rec.update(p_AB=pAB, a_B=B["params"]["a"], f_B=B["params"]["f"])
    if C is not None:
        pAC, _ = lrt(A["chi2"], C["chi2"], 2)
        rec.update(p_AC_nominal=pAC, f1_C=C["params"]["f1"], f2_C=C["params"]["f2"],
                   b1_C=C["params"]["b1"], b2_C=C["params"]["b2"])
    if E is not None:
        rec.update(b1_E=E["params"]["b1"], b1_E_sd=E["errors"]["b1"],
                   b2_E=E["params"]["b2"], b2_E_sd=E["errors"]["b2"])
    if f["E0"] is not None:
        rec.update(b2_E0=f["E0"]["params"]["b2"], b2_E0_sd=f["E0"]["errors"]["b2"],
                   p_E0=f["E0"]["p_gof"])
    if f["A0"] is not None:
        rec.update(p_A0=f["A0"]["p_gof"], r_A0=f["A0"]["params"]["r"])
    rec["z_m0"] = r.get("z_m0", np.nan)
    # best model by AICc / BIC among A-D (E is a polynomial, listed separately)
    cand = {nm: mod for nm, mod in (("A", A), ("B", B), ("C", C), ("D", D)) if mod is not None}
    for crit in ("aicc", "bic"):
        vals = {nm: mod[crit] for nm, mod in cand.items() if np.isfinite(mod[crit])}
        rec[f"best_{crit}"] = min(vals, key=vals.get) if vals else ""
    # significance and form
    ps = [rec.get("p_A", np.nan), rec.get("p_AD", np.nan), rec.get("p_AB", np.nan)]
    ps = [q for q in ps if np.isfinite(q)]
    rec["p_min"] = min(ps) if ps else np.nan
    rec["n_tests"] = len(ps)
    sig = bool(ps) and rec["p_min"] < ALPHA
    rec["significant"] = sig
    form = ""
    if sig:
        boundary = (np.isfinite(rec.get("p_A0", np.nan)) and rec["p_A0"] >= ALPHA
                    and np.isfinite(rec["z_m0"]) and abs(rec["z_m0"]) >= Z_ALPHA)
        if boundary:
            form = "boundary(m=0)"
        elif np.isfinite(rec.get("p_AD", np.nan)) and rec["p_AD"] < ALPHA:
            form = "accelerating" if rec["d"] > 0 else "decelerating"
        else:
            form = "irregular"
        if np.isfinite(rec.get("p_D", np.nan)) and rec["p_D"] < ALPHA and form != "boundary(m=0)":
            form += "+misfit"
    rec["form"] = form
    # windowed summaries
    wf = r.get("win_fid")
    if wf is not None and wf["n_windows"] >= 1:
        rr = 2 / 3 * (1 - np.exp(wf["lnf"]))
        rec.update(r_win_first=rr[0], r_win_last=rr[-1],
                   win_slope_z=wf["slope_z"], win_p_const=wf["p_const"],
                   identity_dev=wf.get("identity_dev", np.nan))
    wf0 = r.get("win_fid_m0excl")
    if wf0 is not None:
        rec.update(win_slope_z_m0excl=wf0["slope_z"], win_p_const_m0excl=wf0["p_const"])
    wa = r.get("win_acc")
    if wa is not None:
        rec.update(acc_slope_z=wa["slope_z"], acc_p_const=wa["p_const"])
    wa0 = r.get("win_acc_m0excl")
    if wa0 is not None:
        rec.update(acc_slope_z_m0excl=wa0["slope_z"], acc_p_const_m0excl=wa0["p_const"],
                   hazard_per_p_m0excl=-wa0["const"] / c["p"] if np.isfinite(wa0.get("const", np.nan)) else np.nan)
    return rec


# ----------------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------------
def write_csv(name, rows, fields=None):
    path = os.path.join(OUT, name)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for row in rows:
            wr.writerow({k: _fmt(row.get(k, "")) for k in fields})
    return path


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
        return f"{float(v):.10g}"
    return v


# ----------------------------------------------------------------------------
# main analyses
# ----------------------------------------------------------------------------
def pipeline_table_check(results):
    """Compare the mirrored pipeline fits with the pipeline's own table."""
    path = os.path.join(RUN, "results", "plots", "unif_lrb_vs_rb_table_all_mixed_fits.csv")
    with open(path, newline="") as fh:
        table = list(csv.DictReader(fh))
    lookup = {(r["curve"]["policy"], r["curve"]["pidx"]): r for r in results}
    worst = dict(lrb_f=0.0, lrb_b=0.0, lrb_chi2_rel=0.0, rb_f=0.0, rb_b=0.0, rb_chi2_rel=0.0)
    n_lrb = n_rb = 0
    npts_mismatch = 0
    for row in table:
        pidx = int(row["p_index"])
        if pidx == 0:
            continue
        r = lookup[(f"unif{int(row['check_num'])}", pidx)]
        pf = r["pipe"]
        if pf is not None and row["lrb_f"] not in ("", "nan"):
            worst["lrb_f"] = max(worst["lrb_f"], abs(pf["f"] - float(row["lrb_f"])))
            worst["lrb_b"] = max(worst["lrb_b"], abs(pf["b"] - float(row["lrb_b"])))
            ch = float(row["lrb_chi2"])
            worst["lrb_chi2_rel"] = max(worst["lrb_chi2_rel"], abs(pf["chi2"] - ch) / max(ch, 1e-300))
            npts_mismatch += int(pf["n_points"] != int(float(row["lrb_n_points"])))
            n_lrb += 1
        rb = lookup[("RBC", pidx)]["pipe"]
        if rb is not None and row["rb_f"] not in ("", "nan"):
            worst["rb_f"] = max(worst["rb_f"], abs(rb["f"] - float(row["rb_f"])))
            worst["rb_b"] = max(worst["rb_b"], abs(rb["b"] - float(row["rb_b"])))
            ch = float(row["rb_chi2"])
            worst["rb_chi2_rel"] = max(worst["rb_chi2_rel"], abs(rb["chi2"] - ch) / max(ch, 1e-300))
            n_rb += 1
    return dict(n_lrb_rows=n_lrb, n_rb_rows=n_rb, n_points_mismatch=npts_mismatch, **worst)


def pipeline_rows(results):
    rows = []
    for r in results:
        c, pf, ps = r["curve"], r["pipe"], r["pipe_sem"]
        row = dict(family=c["family"], policy=c["policy"], p_idx=c["pidx"], p=c["p"],
                   p_phys=c["p_phys"])
        if pf is not None:
            row.update(n_points=pf["n_points"], f=pf["f"], b=pf["b"], F=F_of_f(pf["f"]),
                       r=1 - F_of_f(pf["f"]), chi2_std_weights=pf["chi2"],
                       chi2_sem_weights=ps["chi2"], f_sem_minus_f_std=ps["f"] - pf["f"],
                       chi2_ratio_sem_over_std=ps["chi2"] / pf["chi2"] if pf["chi2"] > 0 else np.nan)
        pfa = r.get("pipe_free")
        if pfa is not None:
            row.update(free_a_f=pfa["f"], free_a_a=pfa["a"], free_a_b=pfa["b"],
                       free_a_r=1 - F_of_f(pfa["f"]))
        rows.append(row)
    return rows


def model_rows(results):
    rows = []
    for r in results:
        c = r["curve"]
        for name in ("A", "B", "C", "D", "E", "A0", "E0"):
            mod = r["fits"].get(name)
            if mod is None:
                continue
            row = dict(family=c["family"], policy=c["policy"], p_idx=c["pidx"], p=c["p"],
                       model=name, n=mod["n"], k=mod["k"], chi2=mod["chi2"], dof=mod["dof"],
                       p_gof=mod["p_gof"], log10_p_gof=mod["log10_p_gof"], aic=mod["aic"],
                       aicc=mod["aicc"], bic=mod["bic"], max_abs_resid=mod["max_abs_resid"])
            for key, val in mod["params"].items():
                row[f"par_{key}"] = val
            for key, val in mod["errors"].items():
                row[f"err_{key}"] = val
            rows.append(row)
    return rows


def point_rows(results):
    rows = []
    for r in results:
        c = r["curve"]
        pred_A = None
        if r["fits"]["A"] is not None:
            pred_A = dict(zip(c["m"][r["usable"]], r["fits"]["A"]["pred"]))
        for i, m in enumerate(c["m"]):
            st = structure(c["kind"], c["k"], m) if c["family"] == "LRB" else {}
            row = dict(family=c["family"], policy=c["policy"], p_idx=c["pidx"], p=c["p"],
                       m=m, y=c["y"][i], std=c["std"][i], sem=r["sem"][i], sigma=r["sigma"][i],
                       rejected=c["rej"][i], rejected_std=c["rej_std"][i],
                       n_acc=r["nacc"][i], rayleigh=r["rayleigh"][i], usable=bool(r["usable"][i]),
                       n_used=st.get("n_used", ""), tail=st.get("tail", ""),
                       segments=st.get("segments", ""))
            if pred_A is not None and m in pred_A:
                row["resid_A_sigma"] = (c["y"][i] - pred_A[m]) / r["sigma"][i]
            rows.append(row)
    return rows


def window_rows(results):
    rows = []
    for r in results:
        c = r["curve"]
        for qty, key in (("fidelity", "win_fid"), ("acceptance", "win_acc")):
            w = r.get(key)
            if w is None:
                continue
            for i in range(w["n_windows"]):
                lnf, sd = w["lnf"][i], w["sd"][i]
                row = dict(family=c["family"], policy=c["policy"], p_idx=c["pidx"], p=c["p"],
                           quantity=qty, m_lo=w["m_lo"][i], m_hi=w["m_hi"][i],
                           m_mid=w["mid"][i], ln_rate=lnf, ln_rate_sd=sd,
                           rate=math.exp(lnf), rate_sd=math.exp(lnf) * sd)
                if qty == "fidelity":
                    row["r_per_clifford"] = 2 / 3 * (1 - math.exp(lnf))
                    row["r_per_clifford_sd"] = 2 / 3 * math.exp(lnf) * sd
                    if r["pipe"] is not None:
                        rp = 1 - F_of_f(r["pipe"]["f"])
                        row["r_over_r_pipeline"] = row["r_per_clifford"] / rp if rp > 0 else np.nan
                    if c["family"] in ("RB", "RBC"):
                        row["r_over_p_phys"] = row["r_per_clifford"] / c["p_phys"]
                    else:
                        row["r_over_p2"] = row["r_per_clifford"] / c["p"] ** 2
                else:
                    row["reject_per_step"] = 1 - math.exp(lnf)
                    row["hazard_over_p"] = -lnf / c["p"]
                    row["hazard_over_p_sd"] = sd / c["p"]
                if c["family"] == "LRB" and not (c["kind"] == "const" and c["k"] == 0):
                    lo = structure(c["kind"], c["k"], w["m_lo"][i])
                    hi = structure(c["kind"], c["k"], w["m_hi"][i])
                    row.update(tail_lo=lo["tail"], tail_hi=hi["tail"],
                               n_used_lo=lo["n_used"], n_used_hi=hi["n_used"],
                               tail_growth=(hi["tail"] - lo["tail"]) / (w["m_hi"][i] - w["m_lo"][i]))
                rows.append(row)
    return rows


def trend_rows(results):
    rows = []
    for r in results:
        c = r["curve"]
        for qty, key in (("fidelity", "win_fid"), ("fidelity_m0excl", "win_fid_m0excl"),
                         ("acceptance", "win_acc"), ("acceptance_m0excl", "win_acc_m0excl")):
            w = r.get(key)
            if w is None:
                continue
            rows.append(dict(family=c["family"], policy=c["policy"], p_idx=c["pidx"], p=c["p"],
                             quantity=qty, n_windows=w["n_windows"], slope_ln_rate=w["slope"],
                             slope_sd=w["slope_se"], slope_z=w["slope_z"],
                             p_slope=2 * stats.norm.sf(abs(w["slope_z"])) if np.isfinite(w["slope_z"]) else np.nan,
                             chi2_const=w["chi2_const"], dof_const=w["dof_const"],
                             p_const=w["p_const"], chi2_lin=w["chi2_lin"], dof_lin=w["dof_lin"],
                             p_lin=w["p_lin"], identity_dev=w.get("identity_dev", np.nan)))
    return rows


def by_policy_rows(dev):
    rows = []
    policies = []
    for d in dev:
        if d["policy"] not in policies:
            policies.append(d["policy"])
    for pol in policies:
        ds = [d for d in dev if d["policy"] == pol and d.get("tested")]
        sig = [d for d in ds if d["significant"]]
        forms = {}
        for d in sig:
            forms[d["form"]] = forms.get(d["form"], 0) + 1
        rows.append(dict(policy=pol, n_tested=len(ds), n_significant=len(sig),
                         significant_p=" ".join(f"{d['p']:g}" for d in sig),
                         forms="; ".join(f"{k}:{v}" for k, v in sorted(forms.items())),
                         max_abs_z_curv=max([abs(d.get("z_curv", 0.0)) for d in ds] or [np.nan]),
                         min_log10_p=min([d.get("log10p_A", 0.0) for d in ds] or [np.nan])))
    return rows


def lookup_dev(dev, policy, pidx):
    for d in dev:
        if d["policy"] == policy and d["p_idx"] == pidx:
            return d
    return None


def lookup_res(results, policy, pidx):
    for r in results:
        if r["curve"]["policy"] == policy and r["curve"]["pidx"] == pidx:
            return r
    return None


def signature_rows(results, dev):
    """(5) const 1 / unif 1 / const 0 coefficients in natural units."""
    rows = []
    for pol in ("const1", "unif1", "unif2", "const0"):
        for pidx in P_IDX:
            r = lookup_res(results, pol, pidx)
            d = lookup_dev(dev, pol, pidx)
            p = PROBS[pidx]
            row = dict(policy=pol, p_idx=pidx, p=p, n_usable=d["n_usable"],
                       r_pipeline=d["r_pipeline"], r_pipeline_over_p2=d["r_pipeline"] / p ** 2)
            f = r["fits"]
            for nm in ("E", "E0"):
                mod = f[nm]
                if mod is None:
                    continue
                row.update({f"{nm}_b1_over_p": mod["params"]["b1"] / p,
                            f"{nm}_b1_sd_over_p": mod["errors"]["b1"] / p,
                            f"{nm}_b2_over_p2": mod["params"]["b2"] / p ** 2,
                            f"{nm}_b2_sd_over_p2": mod["errors"]["b2"] / p ** 2,
                            f"{nm}_b0_over_p": mod["params"]["b0"] / p,
                            f"{nm}_chi2": mod["chi2"], f"{nm}_dof": mod["dof"],
                            f"{nm}_p_gof": mod["p_gof"]})
            if f["D"] is not None:
                row.update(D_d_over_p2=f["D"]["params"]["d"] / p ** 2,
                           D_d_sd_over_p2=f["D"]["errors"]["d"] / p ** 2,
                           D_p_gof=f["D"]["p_gof"], z_curv=d.get("z_curv", np.nan))
            if f["A0"] is not None:
                row.update(A0_r_over_p2=f["A0"]["params"]["r"] / p ** 2,
                           A0_r_sd_over_p2=f["A0"]["errors"]["r"] / p ** 2,
                           A0_p_gof=f["A0"]["p_gof"])
            row.update(p_A=d.get("p_A", np.nan), z_m0=d.get("z_m0", np.nan),
                       m0_offset_over_p2=r.get("m0_offset", np.nan) / p ** 2,
                       m0_offset_sd_over_p2=r.get("m0_offset_sd", np.nan) / p ** 2)
            wa = r.get("win_acc_m0excl")
            if wa is not None and np.isfinite(wa.get("const", np.nan)):
                hz = -wa["const"]                     # GLS mean of -ln a_i over m > 0 windows
                row.update(hazard_over_p=hz / p, hazard_sd_over_p=wa["const_se"] / p,
                           reject_per_step=1 - math.exp(-hz),
                           acc_slope_z_m0excl=wa["slope_z"], acc_p_const_m0excl=wa["p_const"])
                if d["r_pipeline"] > 0:
                    row["reject_per_step_over_r_pipeline"] = (1 - math.exp(-hz)) / d["r_pipeline"]
            wf = r.get("win_fid_m0excl")
            if wf is not None:
                row.update(fid_slope_z_m0excl=wf["slope_z"], fid_p_const_m0excl=wf["p_const"])
            rows.append(row)
    return rows


def weighted_mean(vals, sds):
    vals, sds = np.asarray(vals, float), np.asarray(sds, float)
    ok = np.isfinite(vals) & np.isfinite(sds) & (sds > 0)
    if not ok.any():
        return np.nan, np.nan, np.nan, 0
    w = 1 / sds[ok] ** 2
    mu = float((w * vals[ok]).sum() / w.sum())
    se = float(1 / math.sqrt(w.sum()))
    chi2 = float((w * (vals[ok] - mu) ** 2).sum())
    return mu, se, chi2, int(ok.sum())


def loglog_slope(ps, vals, sds):
    ps, vals, sds = map(lambda a: np.asarray(a, float), (ps, vals, sds))
    ok = np.isfinite(vals) & (vals > 0) & np.isfinite(sds) & (sds > 0)
    if ok.sum() < 2:
        return np.nan, np.nan
    X = np.column_stack([np.ones(ok.sum()), np.log(ps[ok])])
    coef, cov, chi2 = wls(X, np.log(vals[ok]), sds[ok] / vals[ok])
    return float(coef[1]), float(math.sqrt(cov[1, 1]))


def attribution_summary(sig_rows):
    """unif 1's m^2 coefficient bounds pairs inside the inverse half (the only
    O(p^2) source of an m^2 term for unif 1); compare with const 1 and const 0."""
    out = []
    get = lambda pol: [r for r in sig_rows if r["policy"] == pol]
    u1, c1, c0, u2 = get("unif1"), get("const1"), get("const0"), get("unif2")
    rng = [(5e-4, 5e-3), (1e-4, 5e-3), (5e-4, 2e-3)]
    for lo, hi in rng:
        sel = [r for r in u1 if lo <= r["p"] <= hi and "E0_b2_over_p2" in r]
        mu, se, chi2, n = weighted_mean([r["E0_b2_over_p2"] for r in sel],
                                        [r["E0_b2_sd_over_p2"] for r in sel])
        sel2 = [r for r in u2 if lo <= r["p"] <= hi and "E0_b2_over_p2" in r]
        mu2, se2, chi22, n2 = weighted_mean([r["E0_b2_over_p2"] for r in sel2],
                                            [r["E0_b2_sd_over_p2"] for r in sel2])
        out.append(dict(quantity="unif1_E0_b2_over_p2_weighted_mean", p_lo=lo, p_hi=hi,
                        value=mu, sd=se, chi2=chi2, n_p=n, upper95=mu + 1.645 * se))
        out.append(dict(quantity="unif2_E0_b2_over_p2_weighted_mean", p_lo=lo, p_hi=hi,
                        value=mu2, sd=se2, chi2=chi22, n_p=n2, upper95=mu2 + 1.645 * se2))
    for pol, rows, key in (("const1", c1, "E_b2_over_p2"), ("const0", c0, "E_b2_over_p2")):
        for lo, hi in ((1e-4, 5e-4), (1e-4, 1e-3), (1e-4, 2e-3)):
            sel = [r for r in rows if lo <= r["p"] <= hi and key in r]
            mu, se, chi2, n = weighted_mean([r[key] for r in sel],
                                            [r[key.replace("b2_over", "b2_sd_over")] for r in sel])
            out.append(dict(quantity=f"{pol}_{key}_weighted_mean", p_lo=lo, p_hi=hi,
                            value=mu, sd=se, chi2=chi2, n_p=n))
        for lo, hi in ((1e-4, 5e-4), (1e-4, 1e-3), (2e-4, 2e-3)):
            sel = [r for r in rows if lo <= r["p"] <= hi and key in r]
            sl, se = loglog_slope([r["p"] for r in sel], [r[key] * r["p"] ** 2 for r in sel],
                                  [r[key.replace("b2_over", "b2_sd_over")] * r["p"] ** 2 for r in sel])
            out.append(dict(quantity=f"{pol}_b2_loglog_exponent", p_lo=lo, p_hi=hi,
                            value=sl, sd=se, n_p=len(sel)))
    # per-p ratio: unif1 bound / const1 b2 at the same p
    for r1 in u1:
        rc = [r for r in c1 if r["p_idx"] == r1["p_idx"]][0]
        if "E0_b2_over_p2" in r1 and "E_b2_over_p2" in rc:
            out.append(dict(quantity="unif1_E0_b2_over_const1_E_b2", p_lo=r1["p"], p_hi=r1["p"],
                            value=r1["E0_b2_over_p2"] / rc["E_b2_over_p2"],
                            sd=r1["E0_b2_sd_over_p2"] / rc["E_b2_over_p2"], n_p=1))
    # pipeline-rate scaling with p
    for pol, rows in (("unif1", u1), ("unif2", u2), ("const1", c1), ("const0", c0)):
        sel = [r for r in rows if 2e-4 <= r["p"] <= 2e-3 and r["r_pipeline"] > 0]
        if len(sel) >= 2:
            x = np.log([r["p"] for r in sel])
            yv = np.log([r["r_pipeline"] for r in sel])
            out.append(dict(quantity=f"{pol}_r_pipeline_loglog_exponent", p_lo=2e-4, p_hi=2e-3,
                            value=float(np.polyfit(x, yv, 1)[0]), sd=np.nan, n_p=len(sel)))
    return out


def unif_spacing_rows(results):
    """Per-step infidelity growth of unif k from depths with no unchecked forward
    tail (m a multiple of k, m >= k), and the unchecked-tail cost from a
    sawtooth regression 1 - y = alpha + beta m + tau r (+ kappa r m) on m >= k."""
    rows = []
    for pidx in P_IDX:
        p = PROBS[pidx]
        for k in UNIF:
            r = lookup_res(results, f"unif{k}", pidx)
            c = r["curve"]
            ok = r["usable"]
            m, y, s = c["m"][ok], c["y"][ok], r["sigma"][ok]
            tails = np.array([structure("unif", k, mm)["tail"] for mm in m], dtype=float)
            row = dict(k=k, p_idx=pidx, p=p)
            sel = (m >= k) & (tails == 0) & (m > 0)
            if sel.sum() >= 2:
                X = np.column_stack([np.ones(sel.sum()), m[sel]])
                coef, cov, chi2 = wls(X, 1 - y[sel], s[sel])
                row.update(n_r0=int(sel.sum()), m_r0=" ".join(str(int(v)) for v in m[sel]),
                           step_rate_over_p2=coef[1] / p ** 2,
                           step_rate_sd_over_p2=math.sqrt(cov[1, 1]) / p ** 2,
                           step_rate_per_k_over_p2=coef[1] / p ** 2 / k,
                           chi2_r0=chi2, dof_r0=int(sel.sum()) - 2)
            sel = (m >= k) & (m > 0)
            X = np.column_stack([np.ones(sel.sum()), m[sel], tails[sel]])
            if (k >= 3 and sel.sum() >= 4 and len(set(tails[sel])) >= 2
                    and np.linalg.matrix_rank(X) == 3):
                coef, cov, chi2 = wls(X, 1 - y[sel], s[sel])
                row.update(n_saw=int(sel.sum()), m_saw=" ".join(str(int(v)) for v in m[sel]),
                           tails=" ".join(str(int(v)) for v in tails[sel]),
                           tau_over_p2=coef[2] / p ** 2, tau_sd_over_p2=math.sqrt(cov[2, 2]) / p ** 2,
                           beta_over_p2=coef[1] / p ** 2, chi2_saw=chi2, dof_saw=int(sel.sum()) - 3)
                X2 = np.column_stack([X, tails[sel] * m[sel]])
                if sel.sum() >= 5 and np.linalg.matrix_rank(X2) == 4:
                    coef2, cov2, chi22 = wls(X2, 1 - y[sel], s[sel])
                    row.update(kappa_rm_over_p2=coef2[3] / p ** 2,
                               kappa_rm_sd_over_p2=math.sqrt(cov2[3, 3]) / p ** 2,
                               tau2_over_p2=coef2[2] / p ** 2, chi2_saw_rm=chi22,
                               dof_saw_rm=int(sel.sum()) - 4)
                # same with a log-linear model without the tail term (single exponential)
                Xn = np.column_stack([np.ones(sel.sum()), m[sel]])
                _, _, chi2n = wls(Xn, 1 - y[sel], s[sel])
                row.update(chi2_no_tail=chi2n, dof_no_tail=int(sel.sum()) - 2)
            rows.append(row)
    return rows


def const_spacing_rows(results):
    """Quadratic infidelity coefficient of const c on depths m >= c (all c - 1
    intermediate rounds switched on); the pair picture predicts b2 ~ K/c."""
    rows = []
    for pidx in P_IDX:
        p = PROBS[pidx]
        for cc in CONST:
            if cc == 0:
                continue
            r = lookup_res(results, f"const{cc}", pidx)
            c = r["curve"]
            ok = r["usable"] & (c["m"] >= max(cc, 1))
            if ok.sum() < 4:
                continue
            mod = fit_E(c["m"][ok], c["y"][ok], r["sigma"][ok])
            rows.append(dict(c=cc, p_idx=pidx, p=p, n=int(ok.sum()),
                             m_min=float(c["m"][ok].min()), b2_over_p2=mod["params"]["b2"] / p ** 2,
                             b2_sd_over_p2=mod["errors"]["b2"] / p ** 2,
                             c_times_b2_over_p2=cc * mod["params"]["b2"] / p ** 2,
                             c_times_b2_sd_over_p2=cc * mod["errors"]["b2"] / p ** 2,
                             b1_over_p2=mod["params"]["b1"] / p ** 2, chi2=mod["chi2"],
                             dof=mod["dof"], p_gof=mod["p_gof"]))
    return rows


def tail_cost_m2_rows(results):
    """At m = 2 the policies differ only in which of rounds 0 and 1 are used:
    unif1 {0,1}, unif2 {1}, const2 {0}, const1 {} (tail 0, 0, 1, 2)."""
    rows = []
    for pidx in P_IDX:
        p = PROBS[pidx]
        ref = lookup_res(results, "unif1", pidx)
        i = int(np.where(ref["curve"]["m"] == 2)[0][0])
        if not ref["usable"][i]:
            continue
        for pol in ("unif2", "const2", "const1"):
            r = lookup_res(results, pol, pidx)
            if not r["usable"][i]:
                continue
            st = structure(r["curve"]["kind"], r["curve"]["k"], 2)
            diff = ref["curve"]["y"][i] - r["curve"]["y"][i]
            sd = math.hypot(ref["sigma"][i], r["sigma"][i])
            rows.append(dict(p_idx=pidx, p=p, policy=pol, used_rounds=" ".join(
                map(str, used_rounds(r["curve"]["kind"], r["curve"]["k"], 2))),
                tail=st["tail"], excess_infidelity_vs_unif1=diff, sd_independent=sd,
                excess_over_p2=diff / p ** 2, sd_over_p2=sd / p ** 2))
    return rows


def policy_identity_rows(curves):
    """unif k at m < k and const c at m < c use only the terminal round, i.e.
    they are const 1; the CSVs must agree exactly (same shots)."""
    by = {(c["policy"], c["pidx"]): c for c in curves}
    n_cmp, worst = 0, 0.0
    for pidx in P_IDX:
        c1 = by[("const1", pidx)]
        for kind, ks in (("unif", UNIF), ("const", CONST)):
            for k in ks:
                if (kind == "const" and k <= 1) or (kind == "unif" and k == 1):
                    continue
                c = by[(f"{kind}{k}", pidx)]
                for i, m in enumerate(DEPTHS):
                    if m < k:
                        for key in ("y", "std", "rej", "rej_std"):
                            a, b = c[key][i], c1[key][i]
                            if np.isnan(a) and np.isnan(b):
                                continue
                            worst = max(worst, abs(a - b))
                        n_cmp += 1
    # m = 0 is shared by every ordinary policy
    worst0 = 0.0
    for pidx in P_IDX:
        ref = by[("unif1", pidx)]
        for kind, ks in (("unif", UNIF), ("const", CONST)):
            for k in ks:
                if kind == "const" and k == 0:
                    continue
                c = by[(f"{kind}{k}", pidx)]
                worst0 = max(worst0, abs(c["y"][0] - ref["y"][0]), abs(c["rej"][0] - ref["rej"][0]))
    return dict(n_points_compared=n_cmp, max_abs_difference=worst,
                max_abs_difference_m0_all_policies=worst0)


def m0_noise_rows(curves):
    """m = 0: the 30 circuits are identical, so the across-sequence spread is
    shot noise. Compare with the multinomial / binomial predictions."""
    rows = []
    n = NSEQ
    c4 = math.sqrt(2 / (n - 1)) * math.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))
    expect = math.sqrt((n - 1) / n) * c4                       # E[std_ddof0]/sigma
    Df = Dr = 0.0
    for c in curves:
        if c["policy"] != "unif1":
            continue
        y0, s0, r0, rs0 = c["y"][0], c["std"][0], c["rej"][0], c["rej_std"][0]
        nacc = (1 - r0) * c["shots"]
        pf = math.sqrt((1 - y0) * (y0 + 0.5) / nacc)
        pr = math.sqrt(r0 * (1 - r0) / c["shots"])
        Df += n * (s0 / pf) ** 2
        Dr += n * (rs0 / pr) ** 2
        rows.append(dict(p_idx=c["pidx"], p=c["p"], y0=y0, std_fid=s0, pred_fid=pf,
                         ratio_fid=s0 / pf, rejected=r0, std_rej=rs0, pred_rej=pr,
                         ratio_rej=rs0 / pr, expected_ratio=expect))
    dof = len(rows) * (n - 1)
    rf = np.array([r["ratio_fid"] for r in rows])
    rr = np.array([r["ratio_rej"] for r in rows])
    summary = dict(p_idx="pooled", ratio_fid=rf.mean(),
                   ratio_fid_se=rf.std(ddof=1) / math.sqrt(len(rf)), ratio_rej=rr.mean(),
                   ratio_rej_se=rr.std(ddof=1) / math.sqrt(len(rr)), expected_ratio=expect,
                   D_fid=Df, D_rej=Dr, dof=dof,
                   P_chi2_le_D_fid=stats.chi2.cdf(Df, dof),
                   P_chi2_ge_D_rej=stats.chi2.sf(Dr, dof))
    return rows, summary


def covariance_sensitivity_rows(results):
    """Log-space GLS with a 'Brownian' covariance: shot noise (independent) plus
    the remaining across-sequence variance correlated as sqrt(min/max) across
    depths (nested sequences). Compared with the diagonal SEM treatment."""
    rows = []
    for pol in ("unif1", "unif2", "unif4", "const0", "const1", "const2", "RB", "RBC"):
        for pidx in P_IDX:
            r = lookup_res(results, pol, pidx)
            c = r["curve"]
            ok = r["usable"]
            if ok.sum() < 5:
                continue
            m, y, sd, nacc = c["m"][ok], c["y"][ok], c["std"][ok], r["nacc"][ok]
            sig = r["sigma"][ok]
            # per-sequence variance split: total = SEM^2 * NSEQ (same diagonal as the
            # diagonal analysis), shot part from the multinomial formula, rest = sequences
            vtot = sig ** 2 * NSEQ
            vshot = np.minimum(np.clip((1 - y) * (y + 0.5), 0, None) / nacc, vtot)
            vseq = vtot - vshot
            lo, hi = np.minimum.outer(m, m), np.maximum.outer(m, m)
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = np.where(hi > 0, np.sqrt(lo / np.where(hi > 0, hi, 1)), 0.0)
            np.fill_diagonal(rho, 1.0)
            C = (np.diag(vshot) + rho * np.sqrt(np.outer(vseq, vseq))) / NSEQ
            Cl = C / np.outer(y, y)
            v = np.log(y)
            row = dict(policy=pol, p_idx=pidx, p=c["p"], n=len(m),
                       shot_fraction_median=float(np.median((vshot / np.where(sd > 0, sd ** 2, np.nan))[m > 0])))
            for lab, Cov in (("diag", np.diag(sig ** 2 / y ** 2)), ("brownian", Cl)):
                X1 = np.column_stack([np.ones(len(m)), m])
                X2 = np.column_stack([X1, m * m])
                _, _, chi1 = gls(X1, v, Cov)
                cf2, cv2, chi2_ = gls(X2, v, Cov)
                row[f"{lab}_chi2_loglin"] = chi1
                row[f"{lab}_p_loglin"] = gof(chi1, len(m) - 2)[0]
                row[f"{lab}_curv_z"] = -cf2[2] / math.sqrt(cv2[2, 2])
            rows.append(row)
    return rows


def _distinct_lrb_points(results, pidx, include_m0):
    """Distinct (m, used-round set) points of all ordinary-circuit policies at one
    p (policies that use the same rounds at the same m share the same shots and
    have identical CSV entries)."""
    pts = {}
    for r in results:
        c = r["curve"]
        if c["family"] != "LRB" or c["pidx"] != pidx or (c["kind"] == "const" and c["k"] == 0):
            continue
        for i, m in enumerate(c["m"]):
            if m == 0 and not include_m0:
                continue
            used = used_rounds(c["kind"], c["k"], m)
            key = (int(m), tuple(used))
            if key in pts:
                continue
            st = structure(c["kind"], c["k"], m)
            pts[key] = dict(m=float(m), n_used=st["n_used"], s2=st["s2"], tail=st["tail"],
                            kind=c["kind"], y=c["y"][i], sigma=r["sigma"][i],
                            usable=bool(r["usable"][i]), acc=1.0 - c["rej"][i],
                            acc_sd=max(c["rej_std"][i] / math.sqrt(NSEQ), 1.0 / (NSEQ * c["shots"])),
                            shots=c["shots"])
    return list(pts.values())


SEGMENT_TERMS = ["const", "m", "n_used", "S2_closed", "tail", "tail^2", "tail*m", "m^2"]


def _segment_design(pts):
    return np.array([[1.0, q["m"], q["n_used"], q["s2"], q["tail"], q["tail"] ** 2,
                      q["tail"] * q["m"], q["m"] ** 2] for q in pts])


def segment_model_rows(results):
    """Descriptive cross-policy model of the infidelity at m > 0:
        1 - y = a0 + a1 m + a_n n_used + k_seg S2 + tau r + k_t r^2 + k_FI r m + k_II m^2
    S2 = sum of squared lengths of closed forward segments (between used rounds),
    r = unchecked forward tail before the inverse half. Pair counting motivates
    k_seg S2, k_t r^2 (pairs inside forward spans), k_FI r m (tail x inverse
    pairs), k_II m^2 (pairs inside the inverse half). Fitted on all distinct points
    and, as an out-of-sample test, on unif k only and evaluated on const c."""
    rows = []
    for pidx in P_IDX:
        p = PROBS[pidx]
        pts = [q for q in _distinct_lrb_points(results, pidx, False) if q["usable"]]
        if len(pts) < 20:
            continue
        X = _segment_design(pts)
        y = np.array([1 - q["y"] for q in pts])
        s = np.array([q["sigma"] for q in pts])
        uni = np.array([q["kind"] == "unif" for q in pts])
        row = dict(p_idx=pidx, p=p, n_points=len(pts), n_unif=int(uni.sum()))
        if np.linalg.matrix_rank(X) < X.shape[1] or np.linalg.matrix_rank(X[uni]) < X.shape[1]:
            rows.append(row)
            continue
        coef, cov, chi2 = wls(X, y, s)
        res = (y - X @ coef) / s
        _, _, chi2_0 = wls(X[:, :2], y, s)
        cu, _, _ = wls(X[uni], y[uni], s[uni])
        rc = (y[~uni] - X[~uni] @ cu) / s[~uni]
        row.update(chi2=chi2, dof=len(y) - X.shape[1], rms_resid_sigma=float(np.sqrt(np.mean(res ** 2))),
                   max_abs_resid_sigma=float(np.max(np.abs(res))), chi2_no_structure=chi2_0,
                   oos_const_rms_sigma=float(np.sqrt(np.mean(rc ** 2))),
                   oos_const_max_sigma=float(np.max(np.abs(rc))), n_const=int((~uni).sum()))
        se = np.sqrt(np.abs(np.diag(cov)))
        for name, v, e, vu in zip(SEGMENT_TERMS, coef, se, cu):
            unit = p if name == "const" else p ** 2
            row[f"{name}_over_{'p' if name == 'const' else 'p2'}"] = v / unit
            row[f"{name}_sd"] = e / unit
            row[f"{name}_unif_only"] = vu / unit
        rows.append(row)
    return rows


def acceptance_additivity_rows(results):
    """First-order acceptance law across all ordinary-circuit policies:
        -ln acc = H0 + H_p m + H_t n_used
    (a persistent detectable fault is rejected once whatever the schedule;
    a transient readout/ancilla fault only if it hits a used round)."""
    rows = []
    for pidx in P_IDX:
        p = PROBS[pidx]
        pts = [q for q in _distinct_lrb_points(results, pidx, True) if q["acc"] * q["shots"] >= 100]
        X = np.array([[1.0, q["m"], q["n_used"]] for q in pts])
        v = np.array([-math.log(q["acc"]) for q in pts])
        s = np.array([q["acc_sd"] / q["acc"] for q in pts])
        coef, cov, chi2 = wls(X, v, s)
        res = (v - X @ coef) / s
        se = np.sqrt(np.diag(cov))
        rows.append(dict(p_idx=pidx, p=p, n_points=len(pts), H0_over_p=coef[0] / p,
                         Hp_over_p=coef[1] / p, Hp_sd_over_p=se[1] / p, Ht_over_p=coef[2] / p,
                         Ht_sd_over_p=se[2] / p, chi2=chi2, dof=len(v) - 3,
                         rms_resid_sigma=float(np.sqrt(np.mean(res ** 2))),
                         max_abs_resid_sigma=float(np.max(np.abs(res)))))
    return rows


def rising_window_rows(results):
    """Windows in which the fidelity rises significantly with m."""
    rows = []
    for r in results:
        c = r["curve"]
        if c["family"] != "LRB" or (c["kind"] == "const" and c["k"] == 0):
            continue
        w = r.get("win_fid")
        if w is None:
            continue
        for i in range(w["n_windows"]):
            z = w["lnf"][i] / w["sd"][i]
            lo = structure(c["kind"], c["k"], w["m_lo"][i])
            hi = structure(c["kind"], c["k"], w["m_hi"][i])
            rows.append(dict(policy=c["policy"], p_idx=c["pidx"], p=c["p"], m_lo=w["m_lo"][i],
                             m_hi=w["m_hi"][i], z_rise=z, rising=z > Z_ALPHA,
                             tail_lo=lo["tail"], tail_hi=hi["tail"],
                             d_tail=hi["tail"] - lo["tail"], n_used_lo=lo["n_used"],
                             n_used_hi=hi["n_used"],
                             r_per_clifford=2 / 3 * (1 - math.exp(w["lnf"][i]))))
    return rows


def tail_regression_rows(win_rows_all):
    """Per p: windowed per-Clifford error against the growth of the unchecked
    forward tail per step, g = d(tail)/dm, over all distinct LRB windows."""
    out = []
    for pidx in P_IDX:
        rows = [w for w in win_rows_all if w["p_idx"] == pidx and w["quantity"] == "fidelity"
                and w["family"] == "LRB" and "tail_growth" in w and w["policy"] != "const0"]
        seen, uniq = set(), []
        for w in rows:
            key = (round(w["r_per_clifford"], 15), w["m_lo"], w["m_hi"])
            if key not in seen:
                seen.add(key)
                uniq.append(w)
        if len(uniq) < 5:
            continue
        g = np.array([w["tail_growth"] for w in uniq], float)
        rr = np.array([w["r_per_clifford"] for w in uniq])
        sd = np.array([w["r_per_clifford_sd"] for w in uniq])
        X = np.column_stack([np.ones(len(g)), g])
        coef, cov, chi2 = wls(X, rr, sd)
        rec = dict(p_idx=pidx, p=PROBS[pidx], n_windows=len(uniq),
                   slope_r_vs_g=coef[1], slope_sd=math.sqrt(cov[1, 1]),
                   slope_z=coef[1] / math.sqrt(cov[1, 1]), intercept=coef[0],
                   slope_over_p2=coef[1] / PROBS[pidx] ** 2)
        for lab, msk in (("g_neg", g < 0), ("g_0", g == 0), ("g_pos", g > 0)):
            rec[f"median_r_{lab}"] = float(np.median(rr[msk])) if msk.any() else np.nan
            rec[f"n_{lab}"] = int(msk.sum())
        rising = rr < -Z_ALPHA * sd          # fidelity rises significantly over the window
        rec["n_rising_sig"] = int(rising.sum())
        rec["n_rising_sig_with_g_neg"] = int((rising & (g < 0)).sum())
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_plots(results, dev, spacing_rows=None, sig_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "text.color": INK, "axes.labelcolor": INK2})
    # --- windowed per-Clifford error -------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), facecolor=SURFACE)
    panels = [("unif1", "unif 1 (every round used)"), ("const1", "const 1 (terminal round only)"),
              ("const0", "const 0 (no ancilla checks)"), ("RBC", "physical RB, p_RB = 5p")]
    for ax, (pol, title) in zip(axes.ravel(), panels):
        _style(ax)
        for j, pidx in enumerate((3, 4, 5)):
            r = lookup_res(results, pol, pidx)
            w = r.get("win_fid")
            if w is None:
                continue
            p = PROBS[pidx]
            scale = r["curve"]["p_phys"] if pol == "RBC" else p ** 2
            rr = 2 / 3 * (1 - np.exp(w["lnf"])) / scale
            sd = 2 / 3 * np.exp(w["lnf"]) * w["sd"] / scale
            ax.errorbar(w["mid"] + (j - 1) * 0.25, rr, yerr=sd, color=SERIES[j], lw=2,
                        marker="o", ms=4, capsize=0, elinewidth=1,
                        label=(f"p_RB = {r['curve']['p_phys']:g}" if pol == "RBC" else f"p = {p:g}"))
            if r["pipe"] is not None:
                ax.axhline((1 - F_of_f(r["pipe"]["f"])) / scale, color=SERIES[j], lw=1, ls="--")
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("window midpoint m")
        ax.set_ylabel("windowed error per Clifford / p_RB" if pol == "RBC"
                      else "windowed error per Clifford / p^2")
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Depth-windowed error rate r_i = 2(1 - f_i)/3 (dashed: pipeline single-exponential rate)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "windowed_rates.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # --- acceptance hazard -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE, sharey=False)
    for ax, (pol, title) in zip(axes, panels[:3]):
        _style(ax)
        for j, pidx in enumerate((1, 4, 6)):
            r = lookup_res(results, pol, pidx)
            w = r.get("win_acc")
            if w is None:
                continue
            p = PROBS[pidx]
            ax.errorbar(w["mid"] + (j - 1) * 0.25, -w["lnf"] / p, yerr=w["sd"] / p, color=SERIES[j],
                        lw=2, marker="o", ms=4, elinewidth=1, label=f"p = {p:g}")
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("window midpoint m")
        ax.set_ylabel("rejection hazard per step / p")
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Per-step acceptance a_i: hazard -ln(a_i)/p (flat = exponential acceptance)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "acceptance_hazard.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # --- residuals of the single exponential at p = 1e-3 ------------------
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), facecolor=SURFACE, sharey=False)
    for ax, (pol, title) in zip(axes, [("unif1", "unif 1"), ("const1", "const 1"),
                                       ("const0", "const 0"), ("RBC", "RB, p_RB = 5e-3")]):
        _style(ax)
        r = lookup_res(results, pol, 4)
        A = r["fits"]["A"]
        m = r["curve"]["m"][r["usable"]]
        res_ = (r["curve"]["y"][r["usable"]] - A["pred"]) / r["sigma"][r["usable"]]
        ax.axhline(0, color=INK2, lw=1)
        ax.axhspan(-2, 2, color=GRID, alpha=0.6, lw=0)
        ax.plot(m, res_, color=SERIES[0], lw=2, marker="o", ms=5)
        ax.set_title(f"{title}: chi2/dof = {A['chi2']:.1f}/{A['dof']}", color=INK, loc="left")
        ax.set_xlabel("depth m")
        ax.set_ylabel("(y - b f^m) / SEM")
    fig.suptitle("Residuals of the single exponential b f^m in units of the SEM, p = 1e-3 "
                 "(gray band: +-2)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "residuals_p1e-3.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # --- curvature map ------------------------------------------------------
    pols = [f"unif{k}" for k in UNIF] + [f"const{k}" for k in CONST] + ["RB", "RBC"]
    Z = np.full((len(pols), len(P_IDX)), np.nan)
    S = np.zeros_like(Z, dtype=bool)
    for i, pol in enumerate(pols):
        for j, pidx in enumerate(P_IDX):
            d = lookup_dev(dev, pol, pidx)
            if d is not None and d.get("tested") and np.isfinite(d.get("z_curv", np.nan)):
                Z[i, j] = d["z_curv"]
                S[i, j] = bool(d["significant"])
    T = np.sign(Z) * np.log10(1 + np.abs(Z))
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(7.5, 11), facecolor=SURFACE)
    vmax = math.log10(1 + 300)
    im = ax.imshow(np.ma.masked_invalid(T), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    for i in range(len(pols)):
        for j in range(len(P_IDX)):
            if S[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=6, color=INK)
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels(pols, fontsize=6.5)
    ax.set_xticks(range(len(P_IDX)))
    ax.set_xticklabels([f"{PROBS[i]:g}" for i in P_IDX], rotation=60, fontsize=7)
    ax.set_xlabel("p (SI1000)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ticks = [-300, -30, -3, 0, 3, 30, 300]
    cb.set_ticks([math.copysign(math.log10(1 + abs(t)), t) for t in ticks])
    cb.set_ticklabels([str(t) for t in ticks])
    cb.set_label("signed curvature z (red: decay accelerates)", color=INK2)
    ax.set_title("Curvature of ln y (model D vs A, SEM weights); * = deviation significant at 1e-3;\n"
                 "white = fewer than 4 usable depths", color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curvature_map.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # --- check spacing and the m^2 attribution ------------------------------
    if spacing_rows is None or sig_rows is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    for j, pidx in enumerate((3, 4, 5)):
        sel = [q for q in spacing_rows if q["p_idx"] == pidx and "step_rate_over_p2" in q]
        ks = np.array([q["k"] for q in sel], float)
        v = np.array([q["step_rate_over_p2"] for q in sel])
        e = np.array([q["step_rate_sd_over_p2"] for q in sel])
        ax.errorbar(ks + (j - 1) * 0.12, v, yerr=e, color=SERIES[j], lw=0, elinewidth=1.2,
                    marker="o", ms=5, label=f"p = {PROBS[pidx]:g}")
    kk = np.linspace(0.5, 10.5, 50)
    ax.plot(kk, 25 * kk, color=INK2, lw=1, ls="--", label="25 k (reference)")
    ax.set_xlabel("check spacing k of unif k")
    ax.set_ylabel("d(1 - y)/dm  /  p^2")
    ax.set_title("unif k, depths with no unchecked tail (m = multiple of k)", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    _style(ax)
    for j, (pol, key, lab) in enumerate((("const1", "E_b2", "const 1: b2 (all m)"),
                                         ("unif1", "E0_b2", "unif 1: b2 (m > 0)"),
                                         ("const0", "E_b2", "const 0: b2 (all m)"))):
        sel = [q for q in sig_rows if q["policy"] == pol and f"{key}_over_p2" in q and q["p"] <= 2e-3]
        ps = np.array([q["p"] for q in sel])
        v = np.array([q[f"{key}_over_p2"] for q in sel])
        e = np.array([q[f"{key}_sd_over_p2"] for q in sel])
        ax.errorbar(ps * (1 + 0.06 * (j - 1)), v, yerr=e, color=SERIES[j], lw=2, marker="o",
                    ms=5, elinewidth=1.2, label=lab)
    ax.set_xscale("log")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("p (SI1000)")
    ax.set_ylabel("quadratic coefficient b2 / p^2")
    ax.set_title("1 - y = b0 + b1 m + b2 m^2 (SEM-weighted)", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spacing_and_quadratic.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    curves = load_curves()
    results = [analyse_curve(c) for c in curves]
    dev = [deviation_record(r) for r in results]
    tested = [d for d in dev if d["tested"]]
    n_tests = sum(d["n_tests"] for d in tested)
    alpha_bonf = 0.05 / n_tests
    for d in dev:
        d["significant_bonferroni"] = bool(d.get("tested")) and d.get("p_min", 1.0) < alpha_bonf

    chk = pipeline_table_check(results)
    write_csv("pipeline_fits.csv", pipeline_rows(results))
    write_csv("pipeline_table_check.csv", [chk])
    write_csv("model_fits.csv", model_rows(results))
    write_csv("points.csv", point_rows(results))
    write_csv("deviation_summary.csv", dev)
    write_csv("deviation_by_policy.csv", by_policy_rows(dev))
    wrows = window_rows(results)
    write_csv("windowed_rates.csv", wrows)
    write_csv("trend_tests.csv", trend_rows(results))
    sig_rows = signature_rows(results, dev)
    write_csv("signatures_unif1_const1_const0.csv", sig_rows)
    write_csv("attribution_and_scaling.csv", attribution_summary(sig_rows))
    spacing_rows = unif_spacing_rows(results)
    write_csv("check_spacing_unif.csv", spacing_rows)
    write_csv("check_spacing_const.csv", const_spacing_rows(results))
    seg_rows = segment_model_rows(results)
    write_csv("segment_model.csv", seg_rows)
    acc_rows = acceptance_additivity_rows(results)
    write_csv("acceptance_additivity.csv", acc_rows)
    write_csv("tail_cost_m2.csv", tail_cost_m2_rows(results))
    write_csv("tail_growth_regression.csv", tail_regression_rows(wrows))
    rise = rising_window_rows(results)
    write_csv("rising_windows.csv", [w for w in rise if w["rising"]])
    m0rows, m0sum = m0_noise_rows(curves)
    write_csv("m0_shot_noise_check.csv", m0rows + [m0sum])
    write_csv("covariance_sensitivity.csv", covariance_sensitivity_rows(results))
    ident = policy_identity_rows(curves)
    idev = [d.get("identity_dev", np.nan) for d in dev]
    idev = [v for v in idev if np.isfinite(v)]
    ident["reml_identity_max_rel_dev"] = max(idev) if idev else np.nan
    ident["reml_identity_n_cells"] = len(idev)
    pr = pipeline_rows(results)
    ratios = [q["chi2_ratio_sem_over_std"] for q in pr if np.isfinite(q.get("chi2_ratio_sem_over_std", np.nan))]
    dfs = [abs(q["f_sem_minus_f_std"]) for q in pr if "f_sem_minus_f_std" in q]
    ident.update(sem_vs_std_max_abs_df=max(dfs), sem_vs_std_chi2_ratio_min=min(ratios),
                 sem_vs_std_chi2_ratio_max=max(ratios), n_tests_total=n_tests,
                 alpha_bonferroni=alpha_bonf)
    write_csv("identity_checks.csv", [ident])
    if not args.no_plots:
        make_plots(results, dev, spacing_rows, sig_rows)

    # console summary
    print(f"pipeline table check: {chk}")
    print(f"identity checks: {ident}")
    print(f"m=0 noise: {m0sum}")
    print(f"cells tested {len(tested)}, tests {n_tests}, Bonferroni alpha {alpha_bonf:.2e}")
    for fam in ("LRB", "RB", "RBC"):
        ds = [d for d in tested if d["family"] == fam]
        print(f"  {fam}: tested {len(ds)}, significant {sum(d['significant'] for d in ds)}, "
              f"Bonferroni {sum(d['significant_bonferroni'] for d in ds)}")
    print(f"done in {time.time() - t0:.1f} s; outputs in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
