#!/usr/bin/env python3
"""Critic check of claim C6 (code-capacity [[5,1,2]]_3), in EXACT arithmetic.

Independent of A2_optimal_rule_check.py.  Noise: each data qutrit gets
X^a Z^b, (a,b) != (0,0), with probability p/8 each.  Stabilizers and logical
from the brief (notes/00-brief.md):
  S_Z1 = Z0 Z1^-1 Z3^-1, S_Z2 = Z1 Z2 Z4^-1   (detect X-components a)
  S_X1 = X0 X1 X2^-1,    S_X2 = X1^-1 X3 X4^-1 (detect Z-components b)
  L = shift of x0 + x3 = b0 + b3 (mod 3).
For each (syndrome D, label l) we count configurations by weight w, so
P(D,l) = sum_w N[D,l,w] (1-p)^(5-w) (p/8)^w exactly.  With p rational the
scaled weights are integers, and all comparisons of posteriors are exact.

Checks: q(0) >= q(D) for every D (detection on the frontier), exact ties,
three-way ties, MAP failure 1-E[q], and that ties are p-independent.

Run: module load python/3.11 && python critic_A2-optimal-postselection_codecap.py
"""
from fractions import Fraction
import itertools

import numpy as np

ab = [(a, b) for a in range(3) for b in range(3)]
cfg = np.array(list(itertools.product(range(9), repeat=5)))
A = np.array([[x[0] for x in ab]])[0][cfg]
B = np.array([[x[1] for x in ab]])[0][cfg]
wt = (cfg != 0).sum(1)
sZ1 = (A[:, 0] - A[:, 1] - A[:, 3]) % 3
sZ2 = (A[:, 1] + A[:, 2] - A[:, 4]) % 3
sX1 = (B[:, 0] + B[:, 1] - B[:, 2]) % 3
sX2 = (-B[:, 1] + B[:, 3] - B[:, 4]) % 3
lab = (B[:, 0] + B[:, 3]) % 3

for synd_name, synd in (("Z+X", [sZ1, sZ2, sX1, sX2]), ("X only", [sX1, sX2])):
    D = np.zeros(len(cfg), dtype=np.int64)
    for s in synd:
        D = 3 * D + s
    nD = 3 ** len(synd)
    counts = np.zeros((nD, 3, 6), dtype=np.int64)
    np.add.at(counts, (D, lab, wt), 1)
    # p-independent exact ties: identical weight-count vectors
    tie_struct = 0
    for d in range(nD):
        c = counts[d]
        if c.sum() == 0:
            continue
        for l1, l2 in ((0, 1), (0, 2), (1, 2)):
            if np.array_equal(c[l1], c[l2]):
                tie_struct += 1
    print(f"[{synd_name}] syndromes: {nD}; label pairs with identical weight"
          f" enumerators (ties at every p): {tie_struct}")
    for p in (Fraction(1, 1000), Fraction(1, 100), Fraction(1, 20)):
        w0, w1 = 1 - p, p / 8
        P = [[sum(int(counts[d, l, w]) * w0 ** (5 - w) * w1 ** w
                  for w in range(6)) for l in range(3)] for d in range(nD)]
        pD = [sum(r) for r in P]
        q = [max(r) / s if s else Fraction(0) for r, s in zip(P, pD)]
        q0 = q[0]
        above = [d for d in range(nD) if pD[d] and q[d] > q0]
        equal = [d for d in range(1, nD) if pD[d] and q[d] == q0]
        ties = two_way_0x = two_way_12 = three_way = 0
        for d in range(nD):
            if not pD[d]:
                continue
            mx = max(P[d])
            arg = [l for l in range(3) if P[d][l] == mx]
            if len(arg) >= 2:
                ties += 1
                if len(arg) == 3:
                    three_way += 1
                elif 0 in arg:
                    two_way_0x += 1
                else:
                    two_way_12 += 1
        Eq = sum(s * qq for s, qq in zip(pD, q))
        alpha0 = pD[0]
        S_det = P[0][0] / pD[0]
        print(f"  p={str(p):6s} alpha_det={float(alpha0):.4f} 1-S_det={float(1 - S_det):.3e}"
              f"  #D with q>q(0): {len(above)}  #D!=0 with q=q(0): {len(equal)}"
              f"  MAP 1-E[q]={float(1 - Eq):.3e}  ties: {ties} (0x:{two_way_0x},"
              f" 12:{two_way_12}, 3-way:{three_way})")
        # margin of D=0 over the best other syndrome
        others = [float(q[d]) for d in range(1, nD) if pD[d]]
        print(f"         q(0)={float(q0):.8f}  max_{{D!=0}} q(D)={max(others):.8f}")

# ---- toy per-step constants used in A2 section [R] (claim C15), p = 0.02 ----
p = 0.02
w0, w1 = 1 - p, p / 8
D = np.zeros(len(cfg), dtype=np.int64)
for s in (sZ1, sZ2, sX1, sX2):
    D = 3 * D + s
prob = w0 ** (5 - wt) * w1 ** wt
J = np.zeros((81, 3))
np.add.at(J, (D, lab), prob)
pD = J.sum(1)
post = J / pD[:, None]
a_det = pD[0]
lam_det = (3 * post[0, 0] - 1) / 2
# per-step MAP with covariant tie-breaking (ties are 0 vs one nonzero label)
digits = np.array(list(itertools.product(range(3), repeat=4)))
negidx = ((-digits) % 3) @ (3 ** np.arange(3, -1, -1))
c = np.argmax(post >= post.max(1, keepdims=True) * (1 - 1e-12), axis=1)
for d in range(81):
    if negidx[d] < d:
        c[d] = (-c[negidx[d]]) % 3
W3 = np.exp(2j * np.pi / 3)
t_hat = sum(J[d, l] * W3 ** ((l - c[d]) % 3) for d in range(81) for l in range(3))
print(f"\n[toy, p=0.02] a_DET = {a_det:.5f}; lam_DET per step = {lam_det:.7f};"
      f" LMAP t_hat = {t_hat.real:.6f}{t_hat.imag:+.1e}i")
