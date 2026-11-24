#!/usr/bin/env python3
"""Critic of A3: independent exact checks of the A3 toy chain (memory-noisy schedule).

Written from the model description in A3 section 2.8 / the toy docstring, without
importing the A3 code:
  n = 3 data trits, x in Z_3^3, x_0 = 0; per step each trit gets +1 or +2 with
  probability q/2 each; noisy check s = (x0 - x1, x1 - x2) + m, each check
  wrong by +1 or +2 with probability qm/2 each; logical = x[0];
  F = |E w^(x[0] - Lhat)|.

1. Exact transfer matrix of the memoryless committing decoder (w = 0):
   eigenvalues and weights of F(T) = e0^T M^T f.
2. Exact offline (whole-record marginal-MAP) fidelity for T = 1..6 by full
   enumeration of all 9^T records, F_off = sum_s max_k Re(w^-k a(s)),
   with a(s) = sum_x P(s, x) w^x0.  Local per-step factors F(T+1)/F(T).
3. Exact 'none' (= (1 - 3q/2)^T) and exact w = 0 values for comparison.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python critic_A3-pauli-frame_toy_exact.py
"""

from __future__ import annotations

import itertools

import numpy as np

w = np.exp(2j * np.pi / 3)
q, qm = 0.01, 0.006
X = np.array(list(itertools.product(range(3), repeat=3)))          # 27 states
S = np.array(list(itertools.product(range(3), repeat=2)))          # 9 syndromes
xi = {tuple(v): i for i, v in enumerate(X)}
si = {tuple(v): i for i, v in enumerate(S)}
p1 = np.array([1 - q, q / 2, q / 2])
pm1 = np.array([1 - qm, qm / 2, qm / 2])
Perr = np.array([np.prod(p1[v]) for v in X])
Pm = np.array([np.prod(pm1[v]) for v in S])
Hx = np.array([si[((v[0] - v[1]) % 3, (v[1] - v[2]) % 3)] for v in X])
ADD = np.array([[xi[tuple((X[a] + X[b]) % 3)] for b in range(27)] for a in range(27)])
SUB = np.array([[xi[tuple((X[a] - X[b]) % 3)] for b in range(27)] for a in range(27)])
SADD = np.array([[si[tuple((S[a] + S[b]) % 3)] for b in range(9)] for a in range(9)])
SSUB = np.array([[si[tuple((S[a] - S[b]) % 3)] for b in range(9)] for a in range(9)])
# P(s | x) for the syndrome of state x
Lik = np.array([[Pm[SSUB[s, Hx[x]]] for x in range(27)] for s in range(9)])   # (9, 27)


def lookup_ties():
    out = []
    for s in range(9):
        sc = Perr * Lik[s]
        out.append(np.flatnonzero(sc >= sc.max() * (1 - 1e-12)))
    return out


def transfer_w0():
    ties = lookup_ties()
    M = np.zeros((27, 27))
    for r in range(27):
        for e in range(27):
            re = ADD[r, e]
            for m in range(9):
                sp = SADD[Hx[re], m]
                for d in ties[sp]:
                    M[r, SUB[re, d]] += Perr[e] * Pm[m] / len(ties[sp])
    f = w ** X[:, 0]
    ev, R = np.linalg.eig(M)
    wt = R[0, :] * (np.linalg.inv(R) @ f)
    order = np.argsort(-np.abs(ev))
    print("exact w=0 transfer matrix (memory-noisy): leading eigenvalues and weights")
    seen = []
    for k in order:
        if abs(wt[k]) < 1e-9:
            continue
        print(f"   lambda = {ev[k].real:+.6f}{ev[k].imag:+.1e}j  weight = {wt[k].real:+.5f}")
        seen.append(k)
        if len(seen) >= 4:
            break
    Fw0 = []
    v = np.zeros(27); v[0] = 1
    for T in range(0, 7):
        Fw0.append(abs(v @ f))
        v = v @ M
    return Fw0


def offline_exact(Tmax=6):
    # alpha over (record, state); records enumerated in order
    T_C = np.zeros((27, 27))
    for x in range(27):
        for e in range(27):
            T_C[x, ADD[x, e]] += Perr[e]
    alpha = np.zeros((1, 27)); alpha[0, 0] = 1.0
    phase = w ** X[:, 0]
    res = []
    for T in range(1, Tmax + 1):
        pred = alpha @ T_C                                   # (R, 27)
        alpha = (pred[:, None, :] * Lik[None, :, :]).reshape(-1, 27)   # (R*9, 27)
        a = alpha @ phase                                    # (R*9,)
        vals = np.stack([np.real(w ** (-k) * a) for k in range(3)])
        F_off = vals.max(axis=0).sum()
        res.append((T, F_off, np.real(a.sum()), alpha.sum()))
    return res


if __name__ == "__main__":
    Fw0 = transfer_w0()
    res = offline_exact()
    print("\n T   F_offline(exact)  local factor   F_w0(exact)   F_none(exact)")
    prev = None
    for T, Fo, Fn, tot in res:
        loc = f"{Fo / prev:.6f}" if prev else "   -    "
        print(f" {T}   {Fo:.6f}          {loc}      {Fw0[T]:.6f}      {Fn:.6f}   (prob. mass {tot:.6f})")
        prev = Fo
    print(" A3 Monte Carlo (memory-noisy, offline): T=1 0.9909, T=2 0.9898, T=4 0.9876, T=6 0.9877;"
          " fitted f = 0.99930(3)")
