#!/usr/bin/env python3
"""Critic (v2): exact offline fidelity of A3's DISTANCE-2 toy chain (n = 2).

Model as in A3 section 2.8 (re-implemented, not imported): n = 2 data trits,
x_0 = 0; per checked step each trit gets +1 or +2 w.p. q/2 each (q = 0.01);
noisy check s = x0 - x1 + m, m = +1/+2 w.p. qm/2 each; readout x[0] after the
last round (memory-noisy).  Exact enumeration of all 3^T records, T <= 13.

Why: in a distance-2 code every single data fault is ambiguous (two
explanations differing by a logical, never resolved by later rounds).  The
offline MAP decoder keeps these 50/50 posteriors forever.  Analytic prediction
for qm = 0 (to first order in q per step, Poisson number k of faults with mean
mu = 2 q T): a(record) = prod_i (1 + w^(b_i))/2, |a| = 2^-k, and MAP achieves
Re = 2^-k (k even) or 2^-(k+1) (k odd), so
    F_off(T) = (3/4) exp(-mu/2) + (1/4) exp(-3 mu/2),
a sum of two exponentials with weights 3/4 and 1/4 whose rates differ by a
factor 3, while no correction (and any decoder that commits each ambiguity
after a finite window) gives exp(-3 mu/4) = (1 - 3q/2)^T at first order.
The local per-step factor of the offline decoder therefore drifts from the
no-correction value 1 - 3q/2 toward exp(-q) over the whole depth range.

Usage: OMP_NUM_THREADS=1 python critic_A3-pauli-frame_v2_toy_d2_exact.py
"""

from __future__ import annotations

import itertools

import numpy as np

w = np.exp(2j * np.pi / 3)


def run(q, qm, Tmax=13):
    XS = np.array(list(itertools.product(range(3), repeat=2)))
    IDX = {tuple(x): i for i, x in enumerate(XS)}
    pe1 = np.array([1 - q, q / 2, q / 2])
    TC = np.zeros((9, 9))
    for i, x in enumerate(XS):
        for e in XS:
            TC[i, IDX[tuple((x + e) % 3)]] += np.prod(pe1[e])
    pm1 = np.array([1 - qm, qm / 2, qm / 2])
    LIK = np.zeros((3, 9))
    for j, x in enumerate(XS):
        for s in range(3):
            LIK[s, j] = pm1[(s - (x[0] - x[1])) % 3]
    ph = w ** XS[:, 0]
    joint = np.zeros((1, 9))
    joint[0, 0] = 1.0
    out = []
    for T in range(1, Tmax + 1):
        joint = joint @ TC
        joint = (joint[:, None, :] * LIK[None, :, :]).reshape(-1, 9)
        a = joint @ ph
        F = np.max(np.stack([(w ** (-k) * a).real for k in range(3)]), axis=0).sum()
        out.append(F)
    return np.array(out)


def main():
    q = 0.01
    for qm in (0.0, 0.006):
        F = run(q, qm)
        print(f"\nq = {q}, qm = {qm}   (none exact: (1 - 3q/2)^T, local factor {1 - 1.5 * q:.5f};"
              f" offline asymptote exp(-q) = {np.exp(-q):.5f})")
        print("  T   F_off exact   two-exp formula   none        local F_off(T)/F_off(T-1)")
        for T in range(1, len(F) + 1):
            mu = 2 * q * T
            form = 0.75 * np.exp(-mu / 2) + 0.25 * np.exp(-1.5 * mu)
            loc = "" if T == 1 else f"{F[T - 1] / F[T - 2]:.5f}"
            print(f"  {T:2d}  {F[T - 1]:.6f}     {form:.6f}         {(1 - 1.5 * q) ** T:.6f}    {loc}")
    print("\nTwo-exponential formula at larger depths (qm = 0): local factor per step")
    for T in (1, 5, 10, 20, 30, 50, 100, 200):
        mu = 2 * q * T
        f0 = 0.75 * np.exp(-mu / 2) + 0.25 * np.exp(-1.5 * mu)
        mu1 = 2 * q * (T + 1)
        f1 = 0.75 * np.exp(-mu1 / 2) + 0.25 * np.exp(-1.5 * mu1)
        print(f"  T={T:4d}: F={f0:.4f}, local factor {f1 / f0:.5f}")


if __name__ == "__main__":
    main()
