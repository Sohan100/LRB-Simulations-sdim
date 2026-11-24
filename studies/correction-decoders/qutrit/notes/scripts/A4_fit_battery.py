#!/usr/bin/env python3
"""A4: test battery for non-exponential / history-dependent LRB-D decay.

Implements the battery of A4-nonmarkov-theory.md section 5 on the per-
sequence outcome counts of the baseline run (read-only):
    progress/<p_idx>/{unif,const}_check_data/<k>.npy          (9, 30, 3)
    progress/<p_idx>/{unif,const}_check_data/<k>_rejected.npy (9, 30)

For each (p, policy):
  T0  variance decomposition per depth: shot-noise variance of the
      per-sequence fidelity vs between-sequence variance;
  T1  model comparison on the sequence-mean curve, m >= 2 (m = 0 is a
      structurally different circuit, tested separately in T5):
        M1  b f^m                       (2 parameters)
        M4  b exp(-c m - d m^2)         (3)
        M3  b1 f1^m + b2 f2^m           (4)
      weighted chi^2 with the standard error of the mean over sequences,
      AIC = chi2 + 2k, BIC = chi2 + k ln n, nested F-test M1 vs M4;
  T2  cluster bootstrap over sequences (whole depth curves resampled, so
      the depth-to-depth correlation from shared Clifford prefixes is kept):
      95% interval for the curvature d of M4;
  T3  depth-windowed rates f_i = (F(m_{i+1})/F(m_i))^{1/dm}; bootstrap
      interval of the rate drift  Delta = ln f_last - ln f_first;
  T4  Markovian surrogate F_M(m) = B f_M^m with (B, f_M) from a weighted
      log-linear fit to the three shortest non-zero depths m = 2, 4, 6
      (per-step channel estimated where memory has had least time to act);
      signed statistic
      Delta_M = F(22) - F_M(22) with a bootstrap interval in which the
      surrogate is re-estimated on every resample (a max-|.| statistic is
      positive by construction and needs a null calibration instead);
  T5  first-point test: residual of m = 0 against M1 fitted on m >= 2.
  T6  (unif k, k not dividing all depths) sawtooth test: weighted fit of
      ln F = ln b - c m - e (m mod k); Delta chi^2 against e = 0 and a
      bootstrap interval for e (e > 0 means extra decay per unchecked
      forward step left in the final interval).

    module load python/3.11
    python .../A4_fit_battery.py            (~1 minute, one core)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as fdist

sys.path.insert(0, os.path.dirname(__file__))
from A4_exact_lrbd import RUN, DEPTHS, PROBS, OMEGA  # noqa: E402

M = np.array(DEPTHS, dtype=float)
SEL = M >= 2
RNG = np.random.default_rng(7)
NBOOT = 1000


def load(p_idx: int, kind: str, val: int):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    base = f"{RUN}/progress/{p_idx}/{sub}/{val}"
    acc = np.load(base + ".npy").astype(float)          # (D, S, 3)
    rej = np.load(base + "_rejected.npy").astype(float)
    return acc, rej


def per_sequence(acc, rej):
    n = acc.sum(-1)
    z = acc[..., 0] + OMEGA * acc[..., 1] + OMEGA ** 2 * acc[..., 2]
    F = np.abs(z) / np.maximum(n, 1)
    # shot variance of the per-sequence estimate (Re part, delta method)
    P = acc / np.maximum(n, 1)[..., None]
    re = np.array([1.0, -0.5, -0.5])
    var_shot = ((P * re ** 2).sum(-1) - (P * re).sum(-1) ** 2) / \
        np.maximum(n, 1)
    return F.T, (n / (n + rej)).T, var_shot.T          # (S, D)


def m1(par, m):
    return par[0] * par[1] ** m


def m4(par, m):
    return par[0] * np.exp(-par[1] * m - par[2] * m ** 2)


def m3(par, m):
    return par[0] * par[1] ** m + par[2] * par[3] ** m


def wfit(model, p0, m, y, s, bounds):
    r = least_squares(lambda q: (model(q, m) - y) / s, p0, bounds=bounds,
                      x_scale="jac", max_nfev=20000)
    return r.x, float(np.sum(r.fun ** 2))


def fit_all(y, s):
    m = M[SEL]
    yy, ss = y[SEL], s[SEL]
    f0 = np.clip((yy[-1] / yy[0]) ** (1 / (m[-1] - m[0])), 1e-6, 1)
    b0 = yy[0] / f0 ** m[0]
    p1, c1 = wfit(m1, [b0, f0], m, yy, ss, ([0, 0], [2, 1]))
    p4, c4 = wfit(m4, [p1[0], -np.log(max(p1[1], 1e-12)), 0.0], m, yy, ss,
                  ([0, -1, -1], [2, 10, 1]))
    best3 = None
    for frac in (0.2, 0.5, 0.8):
        for fs in (0.5, 0.9, 0.99):
            try:
                p3, c3 = wfit(m3, [p1[0] * (1 - frac), p1[1],
                                   p1[0] * frac, p1[1] * fs], m, yy, ss,
                              ([-2, 0, -2, 0], [2, 1.2, 2, 1.2]))
            except ValueError:
                continue
            if best3 is None or c3 < best3[1]:
                best3 = (p3, c3)
    return (p1, c1), (p4, c4), best3


def loglin(y, s, deg):
    """Weighted least squares of ln y on (1, m, ..., m^deg), m >= 2."""
    m = M[SEL]
    Q = np.stack([m ** k for k in range(deg + 1)], 1)
    wts = (y[SEL] / s[SEL]) ** 2
    WQ = Q * wts[:, None]
    return np.linalg.solve(Q.T @ WQ, WQ.T @ np.log(y[SEL]))


def windowed(y):
    return (y[1:] / y[:-1]) ** (1 / np.diff(M))


def analyse(p_idx: int, kind: str, val: int) -> None:
    acc, rej = load(p_idx, kind, val)
    F, A, vs = per_sequence(acc, rej)           # (S, D)
    S = F.shape[0]
    ok = np.all(A * 1e6 >= 50, axis=0)          # enough accepted shots
    if not np.all(ok[SEL]):
        print(f"  {kind}{val}: too few accepted shots at some depth; skip")
        return
    y = F.mean(0)
    s = F.std(0, ddof=1) / np.sqrt(S)
    s = np.where(s > 0, s, 1e-12)
    # T0 variance decomposition
    v_tot = F.var(0, ddof=1)
    v_shot = vs.mean(0)
    frac_shot = np.where(v_tot > 0, v_shot / v_tot, np.nan)
    (p1, c1), (p4, c4), m3fit = fit_all(y, s)
    n = SEL.sum()
    aic = {"M1": c1 + 4, "M4": c4 + 6}
    bic = {"M1": c1 + 2 * np.log(n), "M4": c4 + 3 * np.log(n)}
    if m3fit is not None:
        aic["M3"] = m3fit[1] + 8
        bic["M3"] = m3fit[1] + 4 * np.log(n)
    Fstat = (c1 - c4) / (c4 / (n - 3)) if c4 > 0 else np.inf
    pF = 1 - fdist.cdf(Fstat, 1, n - 3)
    # T3 windowed rates, T4 surrogate
    wr = windowed(y)
    i2 = list(DEPTHS).index(2)
    short = np.isin(M, [2, 4, 6])

    def surrogate(yv, sv):
        Q = np.stack([np.ones(short.sum()), M[short]], 1)
        wts = (yv[short] / sv[short]) ** 2
        WQ = Q * wts[:, None]
        cb = np.linalg.solve(Q.T @ WQ, WQ.T @ np.log(yv[short]))
        return np.exp(cb[0] + cb[1] * M)

    surr = surrogate(y, s)
    Ninf = y[-1] - surr[-1]
    # T5 first point
    res0 = (y[0] - m1(p1, 0.0)) / s[0]
    # T2-T4 bootstrap over sequences
    d_b, drift_b, N_b = [], [], []
    for _ in range(NBOOT):
        idx = RNG.integers(0, S, S)
        yb = F[idx].mean(0)
        sb = F[idx].std(0, ddof=1) / np.sqrt(S)
        sb = np.where(sb > 0, sb, s)
        pb = -loglin(yb, sb, 2)               # (-ln b, c, d)
        d_b.append(pb[2])
        w = windowed(yb)
        drift_b.append(np.log(w[-1]) - np.log(w[i2]))
        N_b.append(yb[-1] - surrogate(yb, sb)[-1])
    ci = lambda a: np.percentile(a, [2.5, 97.5])  # noqa: E731
    # T6 sawtooth (unif k)
    saw = ""
    if kind == "unif" and np.any((M[SEL] % val) != 0):
        r = (M % val).astype(float)

        def sawfit(yv, sv, with_r=True):
            cols = [np.ones(SEL.sum()), M[SEL]] + ([r[SEL]] if with_r else [])
            Q = np.stack(cols, 1)
            wts = (yv[SEL] / sv[SEL]) ** 2
            WQ = Q * wts[:, None]
            cb = np.linalg.solve(Q.T @ WQ, WQ.T @ np.log(yv[SEL]))
            res = np.log(yv[SEL]) - Q @ cb
            return cb, float(np.sum(res ** 2 * wts))

        cb, chi_r = sawfit(y, s)
        _, chi_0 = sawfit(y, s, with_r=False)
        e_b = []
        for _ in range(NBOOT):
            idx = RNG.integers(0, S, S)
            yb = F[idx].mean(0)
            sb = F[idx].std(0, ddof=1) / np.sqrt(S)
            e_b.append(-sawfit(yb, np.where(sb > 0, sb, s))[0][2])
        eci = ci(e_b)
        saw = (f" | T6 e={-cb[2]:+.2e} CI[{eci[0]:+.1e},{eci[1]:+.1e}] "
               f"chi2 {chi_0:.1f}->{chi_r:.1f}")
    dci, drci, Nci = ci(d_b), ci(drift_b), ci(N_b)
    best_aic = min(aic, key=aic.get)
    best_bic = min(bic, key=bic.get)
    print(f"  {kind}{val:<2d} T0 shot-var fraction m=2,22: "
          f"{frac_shot[1]:.2f},{frac_shot[-1]:.2f} | T1 chi2 M1={c1:8.1f} "
          f"M4={c4:7.1f}"
          + (f" M3={m3fit[1]:7.1f}" if m3fit is not None else "")
          + f" (n={n}) bestAIC={best_aic} bestBIC={best_bic} F-test p="
          f"{pF:.1e} | T2 d={p4[2]:+.2e} CI[{dci[0]:+.1e},{dci[1]:+.1e}] |"
          f" T3 drift={np.log(wr[-1]) - np.log(wr[i2]):+.2e} "
          f"CI[{drci[0]:+.1e},{drci[1]:+.1e}] | T4 DeltaM={Ninf:+.2e} "
          f"CI[{Nci[0]:+.1e},{Nci[1]:+.1e}]"
          + (f" | T5 z0={res0:+.1f}" if s[0] > 1e-11 else " | T5 n/a")
          + saw)


def main() -> None:
    policies = [("unif", 1), ("unif", 2), ("unif", 3), ("unif", 4),
                ("unif", 6), ("const", 1), ("const", 2), ("const", 3),
                ("const", 4), ("const", 0)]
    for p in (5e-4, 1e-3, 2e-3, 5e-3):
        print(f"\np = {p}")
        for kind, val in policies:
            analyse(PROBS.index(p), kind, val)


if __name__ == "__main__":
    main()
