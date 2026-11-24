#!/usr/bin/env python3
"""Critic check (round 2) of A1-syndromes claim C12 (phenomenological model).

Own enumeration, written from the model description in A1-syndromes.md
section 7, not from A1_syndrome_classification.py:
  * R noisy rounds; before round r a data layer of iid depolarizing noise
    (each of the 8 non-identity Paulis with q/8 on each qutrit);
  * round r reports o_r = sigma(E_r) + m_r, each digit wrong with prob q_m
    (value 1 or 2 with q_m/2 each);
  * the decoder sees d_r = o_r - o_{r-1} (o_0 = 0) and, if present, the
    final perfect digits minus o_R.
Settings: F1 (late layer + perfect full final syndrome, class (a,b)),
F2b / F2ab (nothing after round R, class b / (a,b)), F3 (F2b + late data
layer, no final syndrome), F4 (late layer + perfect X digits, class b).
Coefficients are of q (single faults) and q^2 (fault pairs at distinct
locations).  Run: module load python/3.11; python critic_A1-syndromes_r2_pheno.py
"""
import itertools
from collections import defaultdict
from fractions import Fraction as Fr

import numpy as np

S = np.array([[0, 0, 0, 0, 0, 1, -1, 0, -1, 0],      # S_Z1 = Z0 Z1^-1 Z3^-1
              [0, 0, 0, 0, 0, 0, 1, 1, 0, -1],      # S_Z2 = Z1 Z2 Z4^-1
              [1, 1, -1, 0, 0, 0, 0, 0, 0, 0],      # S_X1 = X0 X1 X2^-1
              [0, -1, 0, 1, -1, 0, 0, 0, 0, 0]])    # S_X2 = X1^-1 X3 X4^-1


def sy(a, b):
    return (a[5:] @ b[:5] - a[:5] @ b[5:]) % 3


XL = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
ZL = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0])
PAULIS = []
for q in range(5):
    for x, z in itertools.product(range(3), repeat=2):
        if x or z:
            v = np.zeros(10, int)
            v[q], v[5 + q] = x, z
            PAULIS.append((q, v))


def syn(v):
    return tuple(int(sy(s, v)) for s in S)


def cls(v, mode):
    a, b = int(sy(ZL, v)), int((-sy(XL, v)) % 3)
    return b if mode == "b" else (a, b)


def addc(c1, c2, mode):
    if mode == "b":
        return (c1 + c2) % 3
    return ((c1[0] + c2[0]) % 3, (c1[1] + c2[1]) % 3)


def faults(R, setting, kappa):
    mode = {"F1": "ab", "F2b": "b", "F2ab": "ab", "F3": "b", "F4": "b"}[setting]
    late = setting in ("F1", "F3", "F4")
    fin = {"F1": [0, 1, 2, 3], "F4": [2, 3]}.get(setting, [])
    L = 4 * R + len(fin)
    fl = []
    layers = list(range(1, R + 1)) + ([R + 1] if late else [])
    for r in layers:
        for q, v in PAULIS:
            sig = np.zeros(L, int)
            s = syn(v)
            if r <= R:
                sig[4 * (r - 1):4 * r] = s
            else:
                for t, j in enumerate(fin):
                    sig[4 * R + t] = s[j]
            fl.append((("d", r, q), tuple(sig % 3), cls(v, mode), Fr(1, 8)))
    for r in range(1, R + 1):
        for j in range(4):
            for val in (1, 2):
                sig = np.zeros(L, int)
                sig[4 * (r - 1) + j] = val
                if r < R:
                    sig[4 * r + j] = -val
                elif j in fin:
                    sig[4 * R + fin.index(j)] = -val
                zero = 0 if mode == "b" else (0, 0)
                fl.append((("m", r, j), tuple(sig % 3), zero, Fr(kappa) / 2))
    return fl, mode, L


def analyse(R, setting, kappa):
    fl, mode, L = faults(R, setting, kappa)
    zero_sig = (0,) * L
    zero_c = 0 if mode == "b" else (0, 0)
    T1 = defaultdict(lambda: defaultdict(Fr))
    for loc, sig, c, w in fl:
        T1[sig][c] += w
    res = defaultdict(Fr)
    for sig, d in T1.items():
        tot = sum(d.values())
        if sig == zero_sig:
            res["undetected"] += sum(w for c, w in d.items() if c != zero_c)
            res["mld"] += sum(w for c, w in d.items() if c != zero_c)
            continue
        res["det_rej"] += tot
        res["mld"] += tot - max(d.values())
        if len([c for c, w in d.items() if w > 0]) > 1:
            res["hyb_rej"] += tot
    # pairs: hybrid failures at q^2 (accepted signature, wrong class)
    acc_class = {}
    for sig, d in T1.items():
        if sig == zero_sig:
            continue
        cl = [c for c, w in d.items() if w > 0]
        if len(cl) == 1:
            acc_class[sig] = cl[0]
    hyb2 = Fr(0)
    det2 = Fr(0)
    n = len(fl)
    sigs = [np.array(f[1]) for f in fl]
    for i in range(n):
        for j in range(i + 1, n):
            if fl[i][0] == fl[j][0]:
                continue
            s2 = tuple(((sigs[i] + sigs[j]) % 3).tolist())
            c2 = addc(fl[i][2], fl[j][2], mode)
            w = fl[i][3] * fl[j][3]
            if s2 == zero_sig:
                if c2 != zero_c:
                    hyb2 += w
                    det2 += w
            elif s2 in acc_class and acc_class[s2] != c2:
                hyb2 += w
    res["hyb_fail2_pairs"] = hyb2
    res["det_fail2_pairs"] = det2
    return res


rows = []
for setting in ("F1", "F2b", "F2ab", "F3", "F4"):
    for kappa in (Fr(1, 4), Fr(1), Fr(4)):
        for R in (1, 2, 3):
            r = analyse(R, setting, kappa)
            print(f"{setting:4s} kappa={str(kappa):4s} R={R}: det_rej {r['det_rej']}  undet {r['undetected']}"
                  f"  MLD {r['mld']}  hyb_rej {r['hyb_rej']}  hyb pairs-fail(q^2) {r['hyb_fail2_pairs']}"
                  f"  det pairs-fail(q^2) {r['det_fail2_pairs']}")
