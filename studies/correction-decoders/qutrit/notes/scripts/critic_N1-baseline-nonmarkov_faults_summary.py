#!/usr/bin/env python3
"""Critic check for N1, part 3b: summarise the exact low-order fault analysis
(critic_N1-baseline-nonmarkov_faults.py output) and compare with the data.

To second order in p, for a policy,
    P(reject)          = R1 p + O(p^2)
    P(error | accept)  = U1 p + N2(m) p^2 + O(p^3)
    1 - y(m)          ~= (3/2) P(error | accept)
so  y = b f^m  has  r = (2/3)(1 - f) = dN2/dm p^2 when N2 is linear in m, and
the E-model quadratic coefficient is b2 = (3/2) d^2N2/dm^2 / 2 p^2.
All coefficients are averages over the same 30 Clifford sequences as the data.
Usage: python critic_N1-baseline-nonmarkov_faults_summary.py <faults csv> [out csv]
"""
import csv
import math
import os
import sys

import numpy as np

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
M = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22], dtype=float)
P_REF = 1e-3
PROBS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 1e-2]


def load_faults(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        key = (r["policy"], int(float(r["m"])))
        out.setdefault(key, []).append({k: float(v) for k, v in r.items()
                                        if k not in ("policy", "m", "seq") and v != ""})
    mean = {}
    for key, lst in out.items():
        mean[key] = {k: np.mean([d[k] for d in lst]) for k in lst[0]}
        mean[key]["nseq"] = len(lst)
    return mean


def load_csv(pidx, kind, k):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    with open(os.path.join(RUN, "results", "LRB", str(pidx), sub, f"{k}.csv")) as fh:
        rows = list(csv.reader(fh))
    return [np.array([float(v) for v in r[1:]]) for r in rows[1:5]]


def series(mean, pol, key, scale):
    return np.array([mean[(pol, int(m))][key] / scale for m in M])


def polyfit(m, y, deg):
    X = np.vander(m, deg + 1, increasing=True)
    th, *_ = np.linalg.lstsq(X, y, rcond=None)
    return th


def main():
    mean = load_faults(sys.argv[1])
    print(f"sequences averaged: {mean[('unif1', 0)]['nseq']:.0f}")
    p2 = P_REF ** 2
    # ---------------- first order: hazards and m = 0 infidelity
    print("\n[first order]  (coefficients in units of p)")
    for pol, kind, k in (("unif1", "unif", 1), ("const1", "const", 1), ("const0", "const", 0),
                         ("unif2", "unif", 2)):
        R1 = series(mean, pol, "R1", P_REF)
        U1 = series(mean, pol, "U1", P_REF)
        slope = polyfit(M[1:], R1[1:], 1)[1]
        y, sd, rej, _ = load_csv(1, kind, k)                 # data at p = 1e-4
        acc = 1 - rej
        h_data = np.polyfit(M[1:], -np.log(acc[1:]), 1)[0] / PROBS[1]
        print(f"  {pol:6s}: rejection(m=0) theory {R1[0]:.2f} p, data(1e-4) {-math.log(acc[0])/PROBS[1]:.2f} p;"
              f" hazard (m>=2) theory {slope:.2f} p, data(1e-4) {h_data:.2f} p;"
              f" first window theory {(R1[1]-R1[0])/2:.2f} p;"
              f" U1 (m=0..22) {U1.min():.3f}..{U1.max():.3f} p -> 1-y(0) theory {1.5*U1[0]:.2f} p,"
              f" data(1e-4) {(1-y[0])/PROBS[1]:.2f} p")
    # ---------------- second order
    print("\n[second order]  (coefficients in units of p^2)")
    res = {}
    for pol in ("unif1", "unif2", "const1", "const0"):
        N2 = series(mean, pol, "N2", p2)
        th1 = polyfit(M[1:], N2[1:], 1)             # a + b m on m >= 2
        th2 = polyfit(M[1:], N2[1:], 2)             # a + b m + c m^2 on m >= 2
        thq = polyfit(M, N2, 2)                     # incl. m = 0
        res[pol] = dict(N2=N2, lin=th1, quad=th2, quad_all=thq)
        print(f"  {pol:6s}: N2(m) = " + " ".join(f"{v:.1f}" for v in N2))
        print(f"          linear fit m>=2: intercept {th1[0]:.1f}, slope {th1[1]:.2f}"
              f"  -> r theory = {th1[1]:.1f} p^2 per Clifford")
        print(f"          quadratic fit m>=2: c = {th2[2]:.4f} -> E-model b2 = (3/2)c = {1.5*th2[2]:.3f} p^2;"
              f" incl. m=0: b2 = {1.5*thq[2]:.3f} p^2, b1 = {1.5*thq[1]:.2f} p^2")
        if pol in ("unif1", "unif2"):
            off = 1.5 * (th1[0] - N2[0])
            print(f"          m = 0 offset y(0) - b(m>0) theory = 1.5 (intercept - N2(0)) = {off:.1f} p^2")
    # ---------------- region decomposition
    print("\n[region decomposition of N2]  regions: fwd (before last forward round), last"
          " (last forward check round + reset), inv (inverse blocks), term (terminal round)")
    regions = ("fwd", "last", "inv", "term")
    pairs = [(a, b) for i, a in enumerate(regions) for b in regions[i:]]
    for pol in ("unif1", "const1"):
        print(f"  {pol}:")
        tot_c = res[pol]["quad"][2]
        for a, b in pairs:
            key = f"N2_{a}_{b}"
            s = series(mean, pol, key, p2)
            th = polyfit(M[1:], s[1:], 2)
            th1 = polyfit(M[1:], s[1:], 1)
            print(f"    {a:4s}x{b:4s}: m=0 {s[0]:8.1f}  m=22 {s[-1]:9.1f}  quad c {th[2]:8.4f}"
                  f" ({100*th[2]/tot_c:5.1f}% of total c)  lin-intercept(m>=2) {th1[0]:8.1f}"
                  f"  -> offset share {1.5*(th1[0]-s[0]):7.1f} p^2")
    # ---------------- direct comparison with the data at p = 1e-4, 2e-4
    print("\n[zero-parameter prediction vs data]  y_th = 1 - 1.5 (U1 p + N2 p^2)")
    for pol, kind, k in (("const0", "const", 0), ("const1", "const", 1), ("unif1", "unif", 1)):
        U1 = series(mean, pol, "U1", P_REF)
        N2 = series(mean, pol, "N2", p2)
        for pidx in (1, 2, 3):
            p = PROBS[pidx]
            y, sd, rej, _ = load_csv(pidx, kind, k)
            sem = np.maximum(sd / math.sqrt(30), 1e-12)
            yth = 1 - 1.5 * (U1 * p + N2 * p * p)
            z = (y - yth) / sem
            print(f"  {pol:6s} p={p:g}: (y - y_th)/SEM = " + " ".join(f"{v:+.1f}" for v in z)
                  + f";  1-y data/theory at m=22: {(1-y[-1]):.3e}/{(1-yth[-1]):.3e}")


if __name__ == "__main__":
    main()
