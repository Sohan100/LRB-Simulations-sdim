#!/usr/bin/env python3
"""Critic (v2) of A3-offline-causal-C4/C6: exact offline fidelity of the A3 toy chain.

Model re-implemented from the description in A3 section 2.8 (not imported):
n = 3 data trits, x_0 = 0; per checked step each trit gets +1 or +2 with
probability q/2 each (q = 0.01); noisy check s = (x0 - x1, x1 - x2) + m, each
check wrong by +1 or +2 with probability qm/2 each (qm = 0.006); the readout is
x[0] after the last (noisy) round ('memory-noisy' schedule).

The exact offline fidelity F_off(T) = sum_records max_k Re(w^-k a(record)),
a(record) = sum_x P(record, x) w^x0, is computed by full enumeration of all
9^T records (forward recursion on joint probabilities) for T = 1..6, together
with the exact no-correction value (1 - 3q/2)^T.  Local per-step factors
F(T+1)/F(T) and the implied eps = (2/3)(1 - factor) are printed for comparison
with A3's Monte Carlo fit (offline f = 0.99930(3), eps = 4.64e-4).

Usage: OMP_NUM_THREADS=1 python critic_A3-pauli-frame_v2_toy_exact.py
"""

from __future__ import annotations

import itertools

import numpy as np

w = np.exp(2j * np.pi / 3)
q, qm = 0.01, 0.006
XS = np.array(list(itertools.product(range(3), repeat=3)))
SS = np.array(list(itertools.product(range(3), repeat=2)))
IDX = {tuple(x): i for i, x in enumerate(XS)}
SIDX = {tuple(s): i for i, s in enumerate(SS)}

pe1 = np.array([1 - q, q / 2, q / 2])
T_C = np.zeros((27, 27))
for i, x in enumerate(XS):
    for e in XS:
        T_C[i, IDX[tuple((x + e) % 3)]] += np.prod(pe1[e])
pm1 = np.array([1 - qm, qm / 2, qm / 2])
LIK = np.zeros((9, 27))                     # P(s | x)
for j, x in enumerate(XS):
    hx = np.array([(x[0] - x[1]) % 3, (x[1] - x[2]) % 3])
    for k, s in enumerate(SS):
        LIK[k, j] = np.prod(pm1[(s - hx) % 3])
phase = w ** XS[:, 0]


def main():
    joint = np.zeros((1, 27))
    joint[0, 0] = 1.0
    prev = None
    print(" T   F_off(T)     none(T)    local F(T)/F(T-1)   eps=(2/3)(1-local)")
    for T in range(1, 7):
        joint = joint @ T_C                                   # (R, 27)
        joint = (joint[:, None, :] * LIK[None, :, :]).reshape(-1, 27)
        a = joint @ phase                                     # (R,)
        F = np.max(np.stack([(w ** (-k) * a).real for k in range(3)]), axis=0).sum()
        none = (1 - 1.5 * q) ** T
        if prev is None:
            print(f"{T:2d}  {F:.7f}  {none:.7f}")
        else:
            loc = F / prev
            print(f"{T:2d}  {F:.7f}  {none:.7f}   {loc:.7f}          {(2 / 3) * (1 - loc):.3e}")
        prev = F


if __name__ == "__main__":
    main()
