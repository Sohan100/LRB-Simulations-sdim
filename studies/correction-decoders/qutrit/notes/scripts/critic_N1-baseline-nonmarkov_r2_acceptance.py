#!/usr/bin/env python3
"""Critic check (round 2) of N1 claims C17/C18: first-order acceptance law.

Independent re-implementation from the result CSVs (read-only):
  -ln acc = H0 + H_p m + H_t n_used                    (N1's model)
and an extended model motivated by the actual circuit files:
  * the m = 0 circuit has no reset layer before its terminal round, every
    m >= 1 circuit has one (reset after round m-1), so the terminal round of
    m >= 1 carries the 4 x 2p reset shifts that m = 0 lacks: term [m >= 1];
  * round 0 never has a reset layer before it (ancillas come from the ideal
    encoding), so a used round 0 lacks the 8p reset-shift flips that every
    other used intermediate round has: term [round 0 used].
Prints per-p fits, residuals by group, and the first-window hazards of
const 1 / unif 2 versus their later windows.

Usage: module load python/3.11 && python critic_N1-baseline-nonmarkov_r2_acceptance.py
"""
import csv
import math
import os

import numpy as np

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
PROBS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05]
NSEQ, SHOTS = 30, 1_000_000


def read(pidx, kind, k):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    with open(os.path.join(RUN, "results", "LRB", str(pidx), sub, f"{k}.csv")) as fh:
        rows = list(csv.reader(fh))
    return [np.array([float(v) for v in r[1:]]) for r in rows[1:5]]


def used(kind, k, m):
    # mirrors LRBSimulationPipeline.postselect_uniform_interval / postselect_constant_number
    if kind == "unif":
        return [j * k - 1 for j in range(1, m // k + 1) if j * k - 1 < m] if m >= k else []
    if k <= m:
        st = m // k
        return [j * st - 1 for j in range(1, k)]
    return []


def distinct_points(pidx):
    pts = {}
    for kind, ks in (("unif", range(1, 23)), ("const", range(1, 23))):
        for k in ks:
            y, sd, rj, rsd = read(pidx, kind, k)
            for i, m in enumerate(DEPTHS):
                u = tuple(used(kind, k, m))
                key = (m, u)
                acc = 1 - rj[i]
                if key in pts or acc * SHOTS < 100:
                    continue
                s = max(rsd[i] / math.sqrt(NSEQ), 1 / (NSEQ * SHOTS)) / acc
                pts[key] = dict(m=m, n=len(u), r0=int(0 in u), v=-math.log(acc), s=s,
                                pol=f"{kind}{k}")
    return list(pts.values())


def wls(X, v, s):
    Xw, vw = X / s[:, None], v / s
    c, *_ = np.linalg.lstsq(Xw, vw, rcond=None)
    r = vw - Xw @ c
    cov = np.linalg.inv(Xw.T @ Xw)
    return c, np.sqrt(np.diag(cov)), r


def main():
    print("p      model      H0/p    Hp/p    Ht/p    extra/p (m>=1, r0)          rms   max|res|  groups: mean res (SEM) m0 | r0-used | other")
    for pidx in (1, 2, 3, 4, 5):
        p = PROBS[pidx]
        P = distinct_points(pidx)
        m = np.array([q["m"] for q in P], float)
        n = np.array([q["n"] for q in P], float)
        r0 = np.array([q["r0"] for q in P], float)
        v = np.array([q["v"] for q in P])
        s = np.array([q["s"] for q in P])
        ge1 = (m >= 1).astype(float)
        X3 = np.column_stack([np.ones_like(m), m, n])
        X5 = np.column_stack([np.ones_like(m), m, n, ge1, r0])
        for lab, X in (("N1-3par", X3), ("ext-5par", X5)):
            c, e, r = wls(X, v, s)
            g0, g1, g2 = (m == 0), (r0 == 1), (m > 0) & (r0 == 0)
            extra = "" if X.shape[1] == 3 else f"{c[3]/p:+6.2f}+-{e[3]/p:.2f}, {c[4]/p:+6.2f}+-{e[4]/p:.2f}"
            print(f"{p:<7g}{lab:<10}{c[0]/p:7.2f} {c[1]/p:7.3f} {c[2]/p:7.3f}  {extra:<26} "
                  f"{np.sqrt(np.mean(r**2)):6.2f} {np.max(np.abs(r)):7.2f}   "
                  f"{r[g0].mean():+6.2f} | {r[g1].mean():+6.2f} (n={g1.sum()}) | {r[g2].mean():+6.2f} (n={g2.sum()})")
        print(f"         points {len(P)}; m=0 points {int((m == 0).sum())}; round-0-using points {int(r0.sum())}")
    # first-window hazards
    print("\nwindowed hazard -ln(acc_{i+1}/acc_i)/dm / p (independent-depth errors):")
    for pidx in (1, 2, 3, 4):
        p = PROBS[pidx]
        for kind, k in (("const", 1), ("unif", 2), ("unif", 1)):
            y, sd, rj, rsd = read(pidx, kind, k)
            acc = 1 - rj
            sa = rsd / math.sqrt(NSEQ) / acc
            v = -np.log(acc)
            h = np.diff(v) / np.diff(DEPTHS) / p
            hs = np.sqrt(sa[1:] ** 2 + sa[:-1] ** 2) / np.diff(DEPTHS) / p
            later = np.sum(h[1:] / hs[1:] ** 2) / np.sum(1 / hs[1:] ** 2)
            print(f"  p={p:<7g} {kind}{k:<3} first window (0->2) {h[0]:7.2f} +- {hs[0]:.2f};"
                  f" weighted mean of later windows {later:7.2f};  difference x2 = {2*(h[0]-later):+6.2f} p")


if __name__ == "__main__":
    main()
