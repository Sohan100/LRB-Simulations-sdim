#!/usr/bin/env python3
"""Adversarial critic of A4-nonmarkov-theory.md: checks on the baseline
per-sequence counts (read-only run folder), compared with exact values.

Inputs: baseline progress/<p_idx>/{unif,const}_check_data/<k>{,_rejected}.npy
(9 depths x 30 sequences x 3 outcomes), $A4_SCRATCH/A4_exact.npz (from
A4_exact_lrbd.py, sequences 0-5) and $A4_SCRATCH/critic_A4_const4.npz (from
critic_A4-nonmarkov-theory_exact.py const4, all 30 sequences).

Sections (default all):
  sanity   p = 0: no rejection, F = 1 for every policy.
  const0   C11: (1 - F)/m^2 at p = 1e-3 (both estimators).
  rise     C14: const 4 paired F(4) - F(2) per sequence, z, against the
           exact 30-sequence mean.
  t5       C15: m = 0 residual against b f^m fitted on m >= 2: z with the
           m = 0 standard error only (A4) and with a cluster bootstrap that
           refits on every replicate (includes extrapolation error);
           expected residual from the exact 6-sequence curves.
  sat      C16: const 1 acceptance and real-part fidelity at m = 22, p >= 1e-2.
  rice     C17: modulus vs real-part estimator z-scores against the exact
           values (const 1, p = 1e-2, 2e-2, sequences 0-5).
  power    C18: unif 1 curvature: exact d (6 sequences) vs bootstrap CI.
  saw      C13: sawtooth coefficient e (unif 4) vs p, e/p^2.

    module load python/3.11
    export A4_SCRATCH=<dir>
    python critic_A4-nonmarkov-theory_data.py [sections...]    (~1 minute)
"""
from __future__ import annotations

import os
import sys

import numpy as np

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
       "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22])
PROBS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 0.01, 0.0125,
         0.015, 0.02, 0.03, 0.04, 0.05]
W = np.exp(2j * np.pi / 3)
SCR = os.environ.get("A4_SCRATCH", "/tmp")
RNG = np.random.default_rng(11)


def counts(p, kind, val):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    base = f"{RUN}/progress/{PROBS.index(p)}/{sub}/{val}"
    acc = np.load(base + ".npy").astype(float)        # (D, S, 3)
    rej = np.load(base + "_rejected.npy").astype(float)
    return acc, rej


def estimators(acc, rej):
    n = acc.sum(-1)
    z = acc[..., 0] + W * acc[..., 1] + W ** 2 * acc[..., 2]
    nn = np.maximum(n, 1)
    Fabs = np.abs(z) / nn
    Fre = (acc[..., 0] - 0.5 * (acc[..., 1] + acc[..., 2])) / nn
    A = n / (n + rej)
    return Fabs.T, Fre.T, A.T, n.T                     # (S, D)


def var_re(F):
    return (1 + 2 * F) / 3 + (1 - F) / 6 - F ** 2


def sec_sanity():
    print("\n== sanity: p = 0")
    bad = 0
    for kind, vals in (("unif", (1, 2, 3, 4, 6)), ("const", (0, 1, 2, 3, 4))):
        for v in vals:
            acc, rej = counts(0.0, kind, v)
            Fa, Fr, A, n = estimators(acc, rej)
            bad += int(np.any(A != 1) or np.any(np.abs(Fr - 1) > 1e-12))
    print(f"  policies with rejection or F != 1 at p = 0: {bad}")


def sec_const0():
    print("\n== const0 (C11): (1 - F)/m^2 at p = 1e-3, mean over 30 seqs")
    acc, rej = counts(1e-3, "const", 0)
    Fa, Fr, A, n = estimators(acc, rej)
    for lab, F in (("modulus", Fa), ("real", Fr)):
        y = (1 - F[:, 1:]) / DEPTHS[1:] ** 2
        print(f"  {lab:7s}: " + ", ".join(
            f"m={m}: {y[:, i].mean() * 1e6:.2f}+-{y[:, i].std(ddof=1) / np.sqrt(30) * 1e6:.2f}"
            for i, m in enumerate(DEPTHS[1:])) + "  (x1e-6)")


def sec_rise():
    print("\n== rise (C14): const 4, paired F(4) - F(2) over 30 sequences")
    ex = np.load(os.path.join(SCR, "critic_A4_const4.npz"))
    for p in (1e-3, 5e-3):
        acc, rej = counts(p, "const", 4)
        Fa, Fr, A, n = estimators(acc, rej)
        k = list(ex["p"]).index(p)
        dex = ex["F4"][:, k] - ex["F2"][:, k]
        for lab, F in (("modulus", Fa), ("real", Fr)):
            dd = F[:, 2] - F[:, 1]
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            print(f"  p={p:.0e} {lab:7s}: data mean {dd.mean():+.2e} (SE "
                  f"{se:.1e}, z = {dd.mean() / se:+.2f}); exact mean over the"
                  f" same 30 sequences {dex.mean():+.2e}; data - exact = "
                  f"{(dd.mean() - dex.mean()) / se:+.2f} SE; per-sequence "
                  f"(data - exact) sd/shot-sd check: "
                  f"{np.std(dd - dex, ddof=1):.1e}")


def m1_fit(y, s, sel):
    m = DEPTHS[sel]
    Q = np.stack([np.ones(sel.sum()), m], 1)
    wts = (y[sel] / s[sel]) ** 2
    WQ = Q * wts[:, None]
    return np.linalg.solve(Q.T @ WQ, WQ.T @ np.log(y[sel]))


def sec_t5():
    print("\n== t5 (C15): m = 0 point vs b f^m fitted on m >= 2 (unif 1)")
    z = np.load(os.path.join(SCR, "A4_exact.npz"))
    pe = list(z["p_eval"])
    sel = DEPTHS >= 2
    for p in (5e-4, 1e-3, 2e-3):
        acc, rej = counts(p, "unif", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        F = Fa
        y, s = F.mean(0), F.std(0, ddof=1) / np.sqrt(30)
        b = m1_fit(y, s, sel)
        res0 = y[0] - np.exp(b[0])
        boot = []
        for _ in range(2000):
            idx = RNG.integers(0, 30, 30)
            yb = F[idx].mean(0)
            sb = F[idx].std(0, ddof=1) / np.sqrt(30)
            sb = np.where(sb > 0, sb, s)
            bb = m1_fit(yb, sb, sel)
            boot.append(yb[0] - np.exp(bb[0]))
        boot = np.array(boot)
        # exact expectation (6-sequence mean curve, same fit)
        Fe = z["F"][0][:, :, pe.index(p)].mean(0)
        be = m1_fit(Fe, s, sel)
        print(f"  p={p:.0e}: residual {res0:+.2e}; z (m=0 SE only, A4) = "
              f"{res0 / s[0]:+.2f}; z (cluster bootstrap incl. "
              f"extrapolation) = {res0 / boot.std(ddof=1):+.2f}; exact "
              f"expected residual (6 seqs) {Fe[0] - np.exp(be[0]):+.2e} -> "
              f"expected z {(Fe[0] - np.exp(be[0])) / boot.std(ddof=1):+.2f}")


def sec_sat():
    print("\n== sat (C16): const 1 at m = 22")
    z = np.load(os.path.join(SCR, "A4_exact.npz"))
    pe = list(z["p_eval"])
    for p in (0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05):
        acc, rej = counts(p, "const", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        ex = (f"exact A(22) = {z['A'][5][:, 8, pe.index(p)].mean():.5f}"
              if p in pe else "")
        print(f"  p={p:<6}: data A(22) mean {A[:, 8].mean():.5f} (range "
              f"{A[:, 8].min():.5f}-{A[:, 8].max():.5f}); real-part F(22) "
              f"mean {Fr[:, 8].mean():+.4f} sd {Fr[:, 8].std(ddof=1):.4f}; "
              f"modulus F(22) mean {Fa[:, 8].mean():.4f}; "
              f"sqrt(pi/(4 n_acc)) = {np.mean(np.sqrt(np.pi / 4 / n[:, 8])):.4f}"
              f"  {ex}   [1/81 = {1 / 81:.5f}]")


def sec_rice():
    print("\n== rice (C17): estimator z-scores against exact values")
    z = np.load(os.path.join(SCR, "A4_exact.npz"))
    pe = list(z["p_eval"])
    seqs = list(z["seqs"])
    for p in (0.01, 0.02):
        k = pe.index(p)
        acc, rej = counts(p, "const", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        Fx = z["F"][5][:, :, k]                      # (6, D) exact
        Fa, Fr, n = Fa[seqs], Fr[seqs], n[seqs]
        se = np.sqrt(var_re(Fx) / np.maximum(n, 1))
        za, zr = (Fa - Fx) / se, (Fr - Fx) / se
        low = Fx < 5 * se
        print(f"  p={p}: n={za.size}: modulus z mean {za.mean():+.2f} "
              f"(sd {za.std():.2f}); real z mean {zr.mean():+.2f} (sd "
              f"{zr.std():.2f}); cells with F < 5 SE: {low.sum()}, modulus "
              f"z there {za[low].mean():+.2f}, real {zr[low].mean():+.2f}")


def loglin(y, s, deg, sel):
    m = DEPTHS[sel]
    Q = np.stack([m ** k for k in range(deg + 1)], 1)
    wts = (y[sel] / s[sel]) ** 2
    WQ = Q * wts[:, None]
    return np.linalg.solve(Q.T @ WQ, WQ.T @ np.log(y[sel]))


def sec_power():
    print("\n== power (C18): unif 1 curvature d (ln F = a - c m - d m^2)")
    z = np.load(os.path.join(SCR, "A4_exact.npz"))
    pe = list(z["p_eval"])
    sel = DEPTHS >= 2
    for p in (5e-4, 1e-3, 2e-3):
        acc, rej = counts(p, "unif", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        F = Fa
        y, s = F.mean(0), F.std(0, ddof=1) / np.sqrt(30)
        d0 = -loglin(y, s, 2, sel)[2]
        db = []
        for _ in range(2000):
            idx = RNG.integers(0, 30, 30)
            yb = F[idx].mean(0)
            sb = F[idx].std(0, ddof=1) / np.sqrt(30)
            db.append(-loglin(yb, np.where(sb > 0, sb, s), 2, sel)[2])
        db = np.array(db)
        Fe = z["F"][0][:, :, pe.index(p)]            # (6, D)
        dex = -loglin(Fe.mean(0), s, 2, sel)[2]
        dper = np.array([-loglin(Fe[i], s, 2, sel)[2] for i in range(6)])
        print(f"  p={p:.0e}: data d = {d0:+.2e}, bootstrap sd {db.std():.1e},"
              f" 95% half-width {0.5 * np.diff(np.percentile(db, [2.5, 97.5]))[0]:.1e};"
              f" exact d (6-seq mean curve) {dex:+.2e} -> d/sd = "
              f"{dex / db.std():.2f}; per-sequence exact d: "
              f"{np.array2string(dper, precision=1)};"
              f" acceptance at m=22 {A[:, 8].mean():.4f}")


def sec_saw():
    print("\n== saw (C13): unif 4 sawtooth e in ln F = a - c m - e (m mod 4)")
    sel = DEPTHS >= 2
    r = (DEPTHS % 4).astype(float)
    for p in (5e-4, 1e-3, 2e-3, 5e-3):
        acc, rej = counts(p, "unif", 4)
        Fa, Fr, A, n = estimators(acc, rej)
        y, s = Fa.mean(0), Fa.std(0, ddof=1) / np.sqrt(30)
        Q = np.stack([np.ones(sel.sum()), DEPTHS[sel], r[sel]], 1)
        wts = (y[sel] / s[sel]) ** 2
        WQ = Q * wts[:, None]
        cb = np.linalg.solve(Q.T @ WQ, WQ.T @ np.log(y[sel]))
        print(f"  p={p:.0e}: e = {-cb[2]:+.2e}, e/p^2 = {-cb[2] / p ** 2:7.1f}")


SECTIONS = {"sanity": sec_sanity, "const0": sec_const0, "rise": sec_rise,
            "t5": sec_t5, "sat": sec_sat, "rice": sec_rice,
            "power": sec_power, "saw": sec_saw}

if __name__ == "__main__":
    for nm in sys.argv[1:] or list(SECTIONS):
        SECTIONS[nm]()
