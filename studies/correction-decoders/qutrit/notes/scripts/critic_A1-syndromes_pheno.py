#!/usr/bin/env python3
"""Critic check for A1-syndromes C12 (phenomenological model).

Independent re-implementation: signatures are the RAW round outcomes
o_r = sigma(E_1+...+E_r) + m_r (not detection events), faults are enumerated
directly, and the exact R = 1 check is a direct convolution over the joint
(syndrome, class) table of one data layer (not a forward recursion).

Model (as stated in the notes): R noisy rounds; iid depolarizing data noise q
per qutrit (q/8 per Pauli) before each round; each outcome digit shifted with
probability q_m = kappa q (value 1 or 2 equally). Finals:
  F1  late data layer, then perfect full syndrome; class (a,b)
  F2  nothing after round R; class b
  F3  F2 plus an unobserved late data layer; class b
  F4  late data layer, then perfect X-syndrome only; class b
  F2f F2 but class (a,b) (an intermediate LRB round without successor)
"""
import itertools
import os
from collections import defaultdict
from fractions import Fraction as Fr

import numpy as np

N = 5
FUN = np.array([[1, 2, 0, 2, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 2, 1],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
OUT = []


def say(*a):
    s = " ".join(str(t) for t in a)
    print(s, flush=True)
    OUT.append(s)


SINGLE = []
for q in range(N):
    for x in range(3):
        for z in range(3):
            if (x, z) == (0, 0):
                continue
            v = np.zeros(10, dtype=int)
            v[q], v[5 + q] = x, z
            s = FUN @ v % 3
            SINGLE.append((q, tuple(int(t) for t in s[:4]), int(s[4]), int(s[5])))


def fault_list(R, final, kappa):
    """Return list of (loc, signature tuple, class tuple(a,b), weight)."""
    late = final in ("F1", "F3", "F4")
    fin = {"F1": [0, 1, 2, 3], "F4": [2, 3]}.get(final, [])
    L = 4 * R + len(fin)
    out = []
    layers = list(range(1, R + 1)) + ([R + 1] if late else [])
    for r in layers:
        for (q, s, a, b) in SINGLE:
            sig = [0] * L
            for rr in range(r, R + 1):            # raw outcomes of rounds >= r
                for j in range(4):
                    sig[4 * (rr - 1) + j] = s[j]
            for t, j in enumerate(fin):           # perfect final syndrome
                sig[4 * R + t] = s[j]
            out.append((("d", r, q), tuple(sig), (a, b), Fr(1, 8)))
    for r in range(1, R + 1):
        for j in range(4):
            for v in (1, 2):
                sig = [0] * L
                sig[4 * (r - 1) + j] = v
                out.append((("m", r, j), tuple(sig), (0, 0), Fr(kappa) / 2))
    return out


def project(c, mode):
    return c if mode == "full" else (0, c[1])


def analyse(R, final, kappa, mode, order2=False):
    F = fault_list(R, final, kappa)
    zero = tuple([0] * len(F[0][1]))
    G = defaultdict(lambda: defaultdict(Fr))
    for loc, sig, c, w in F:
        G[sig][project(c, mode)] += w
    det = sum(sum(g.values()) for s, g in G.items() if s != zero)
    undet = sum(w for c, w in G.get(zero, {}).items() if c != (0, 0))
    mld = undet + sum(sum(g.values()) - max(g.values()) for s, g in G.items()
                      if s != zero)
    hyb = sum(sum(g.values()) for s, g in G.items()
              if s != zero and sum(1 for w in g.values() if w > 0) > 1)
    res = dict(det=det, undet=undet, mld=mld, hyb=hyb)
    if order2:
        dec = {zero: (0, 0)}          # zero signature is always accepted
        for s, g in G.items():
            if s == zero:
                dec[s] = (0, 0)
            elif sum(1 for w in g.values() if w > 0) == 1:
                dec[s] = max(g, key=g.get)
        A2 = Fr(0)
        nF = len(F)
        for i in range(nF):
            for j in range(i + 1, nF):
                if F[i][0] == F[j][0]:
                    continue
                s2 = tuple((x + y) % 3 for x, y in zip(F[i][1], F[j][1]))
                if s2 not in dec:
                    continue          # rejected, or pair-only (cannot fail)
                c2 = ((F[i][2][0] + F[j][2][0]) % 3, (F[i][2][1] + F[j][2][1]) % 3)
                if project(c2, mode) != dec[s2]:
                    A2 += F[i][3] * F[j][3]
        # first-order accepted failures (undetected) and their depletion
        Ttot = Fr(0)
        locw = defaultdict(Fr)
        for loc, sig, c, w in F:
            locw[loc] += w
        Ttot = sum(locw.values())
        dep = sum(w * (Ttot - locw[loc]) for loc, sig, c, w in F
                  if sig == zero and project(c, mode) != (0, 0))
        res["hyb_fa2"] = A2 - dep + undet * hyb
    return res


say("Leading coefficients (units of q) and hybrid P(f|a) q^2 coefficient")
for final, mode in (("F1", "full"), ("F2", "obs"), ("F2f", "full"),
                    ("F3", "obs"), ("F4", "obs")):
    fin = "F2" if final == "F2f" else final
    for kappa in (Fr(1, 4), Fr(1), Fr(4)):
        line = []
        for R in (1, 2, 3, 4):
            r = analyse(R, fin, kappa, mode, order2=(final == "F1" and R <= 2))
            t = f"R={R}: det {r['det']} undet {r['undet']} MLD {r['mld']} hyb {r['hyb']}"
            if "hyb_fa2" in r:
                t += f" hybf|a2 {r['hyb_fa2']}"
            line.append(t)
        say(f"  {final:3s} {mode:4s} kappa={kappa}: " + " | ".join(line))

# final-round explanation sets (F2): which classes tie on a single digit
say("F2 final round, single-digit events (R=1, kappa=1): explanations")
F = fault_list(1, "F2", 1)
G = defaultdict(list)
for loc, sig, c, w in F:
    G[sig].append((loc, c, w))
for sig, lst in sorted(G.items()):
    if sum(1 for x in sig if x) == 1:
        say(f"   sig {sig}: " + "; ".join(f"{l} class{c} w{w}" for l, c, w in lst))

# ---------------------------------------------------------------- exact R=1
say("Exact finite-q check, R = 1 (direct convolution)")
allv = np.array(list(itertools.product(range(3), repeat=10)))
wt = np.count_nonzero((allv[:, :5] != 0) | (allv[:, 5:] != 0), axis=1)
S = allv @ FUN.T % 3
sidx = S[:, 0] + 3 * S[:, 1] + 9 * S[:, 2] + 27 * S[:, 3]
cidx = S[:, 4] + 3 * S[:, 5]


def onelayer(q):
    p = (1 - q) ** (5 - wt) * (q / 8) ** wt
    T = np.zeros((81, 9))
    np.add.at(T, (sidx, cidx), p)
    return T


def sadd(s1, s2):
    d1 = np.array([(s1 // 3 ** k) % 3 for k in range(4)])
    d2 = np.array([(s2 // 3 ** k) % 3 for k in range(4)])
    d = (d1 + d2) % 3
    return int(d @ np.array([1, 3, 9, 27]))


SADD = np.array([[sadd(a, b) for b in range(81)] for a in range(81)])
CADD = np.array([[((a % 3 + b % 3) % 3) + 3 * ((a // 3 + b // 3) % 3) for b in range(9)]
                 for a in range(9)])


# vectorised version
def exact_F1_R1_vec(q, qm):
    T = onelayer(q)
    pm = np.array([np.prod([(1 - qm) if (m // 3 ** k) % 3 == 0 else qm / 2
                            for k in range(4)]) for m in range(81)])
    P = np.zeros((81, 81, 9))
    for s1 in range(81):
        for c1 in range(9):
            t = T[s1, c1]
            if t == 0:
                continue
            # second layer table shifted: s_f = s1 + s2, c = c1 + c2
            T2 = np.zeros((81, 9))
            T2[SADD[s1]] = T                      # T2[s_f] = T[s_f - s1]
            T2c = np.zeros((81, 9))
            T2c[:, CADD[c1]] = T2                 # columns reindexed by c
            # o1 = s1 + m
            P[SADD[s1], :, :] += t * pm[:, None, None] * T2c[None, :, :]
    tot = P.sum(axis=2)
    best = P.max(axis=2)
    mld = (tot - best).sum()
    acc0 = tot[0, 0]
    det_fail = tot[0, 0] - P[0, 0, 0]
    return mld, 1 - acc0, det_fail / acc0, P.sum()


for q, qm in ((1e-4, 1e-4), (1e-4, 2.5e-5)):
    mld, rej, dfa, tot = exact_F1_R1_vec(q, qm)
    say(f"  F1 R=1 q={q} qm={qm}: total {tot:.12f}  MLD/q {mld / q:.5f}  "
        f"det rej/q {rej / q:.4f}  det P(f|a)/q^2 {dfa / q ** 2:.4f}")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "critic_A1-syndromes_pheno_output.txt"), "w") as fh:
    fh.write("\n".join(OUT) + "\n")
