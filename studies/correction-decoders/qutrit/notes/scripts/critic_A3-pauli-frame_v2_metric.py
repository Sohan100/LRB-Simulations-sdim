#!/usr/bin/env python3
"""Critic (v2) of A3-offline-causal-C2/C3 and A3-boundary-C5 (exact, code capacity).

Part 1 (A3-boundary-C5): weight-1 ambiguity table of the [[5,1,2]]_3 code from the
  brief's stabilizers, plus the minimum weight of a same-syndrome partner in a
  different logical class for every weight-1 fault.
Part 2 (A3-offline-causal-C2): is the MAP decoder optimal for the brief's metric
  |E w^R|?  Exact code-capacity model: i.i.d. single-qutrit depolarizing noise of
  strength q on the five data qutrits, perfect syndrome, readout shift b0 + b3
  (the Z-powers on wires 0 and 3; brief's terminal readout).  All 9^5 error
  configurations are enumerated.  Decoders: none; MAP with symmetric random
  tie-breaking; MAP with deterministic lowest-index tie-breaking; anti-MAP
  (least likely class, symmetric ties); and the exact optimum of |E w^R| over
  all deterministic decoders, max_phi sum_s P(s) max_k Re(e^{-i phi} w^{-k} a(s)).
Part 3: first-order code-capacity failure rate of none vs MAP (for the X_L
  readout only; not the Clifford-twirled LRB quantity).

Usage: module load python/3.11; python critic_A3-pauli-frame_v2_metric.py
"""

from __future__ import annotations

import itertools

import numpy as np

w = np.exp(2j * np.pi / 3)
NQ = 5


def vec(xs=None, zs=None):
    """Weyl vector (a0..a4, b0..b4) from {wire: power} dicts."""
    v = np.zeros(2 * NQ, int)
    for q, p in (xs or {}).items():
        v[q] = p % 3
    for q, p in (zs or {}).items():
        v[NQ + q] = p % 3
    return v


def symp(u, v):
    """Symplectic form <u, v> = a_u . b_v - b_u . a_v (mod 3)."""
    return int(u[:NQ] @ v[NQ:] - u[NQ:] @ v[:NQ]) % 3


STABS = [vec(zs={0: 1, 1: -1, 3: -1}), vec(zs={1: 1, 2: 1, 4: -1}),
         vec(xs={0: 1, 1: 1, 2: -1}), vec(xs={1: -1, 3: 1, 4: -1})]
XL = vec(xs={0: 1, 3: 1})
ZL = vec(zs={0: 1, 2: 1})


def syndrome(e):
    return tuple(symp(s, e) for s in STABS)


def logical_coords(e):
    return (symp(e, ZL), symp(e, XL))


def part1():
    print("Part 1: commutation checks and weight-1 ambiguities")
    for i, s in enumerate(STABS):
        assert all(symp(s, t) == 0 for t in STABS), "stabilizers do not commute"
        assert symp(s, XL) == 0 and symp(s, ZL) == 0, "logical not in normalizer"
    print(f"  stabilizers commute; X_L, Z_L in normalizer; <X_L,Z_L> = {symp(XL, ZL)}")
    errs1 = []
    for q in range(NQ):
        for a, b in itertools.product(range(3), repeat=2):
            if (a, b) != (0, 0):
                errs1.append(((q, a, b), vec(xs={q: a}, zs={q: b})))
    errs2 = []
    for q1, q2 in itertools.combinations(range(NQ), 2):
        for a1, b1, a2, b2 in itertools.product(range(3), repeat=4):
            if (a1, b1) != (0, 0) and (a2, b2) != (0, 0):
                errs2.append(vec(xs={q1: a1, q2: a2}, zs={q1: b1, q2: b2}))
    pairs = []
    minw = {}
    for (lab, e) in errs1:
        s, l = syndrome(e), logical_coords(e)
        p1 = [lab2 for (lab2, e2) in errs1 if lab2 != lab and syndrome(e2) == s
              and logical_coords(e2) != l]
        if p1:
            minw[lab] = 1
            pairs.extend((lab, x) for x in p1)
        else:
            p2 = [e2 for e2 in errs2 if syndrome(e2) == s and logical_coords(e2) != l]
            minw[lab] = 2 if p2 else 3
    names = {1: "X", 2: "X^-1"}
    seen = set()
    print("  weight-1 same-syndrome, different-class pairs (wire, a, b):")
    for (l1, l2) in pairs:
        key = tuple(sorted([l1, l2]))
        if key in seen:
            continue
        seen.add(key)
        print(f"    {l1} ~ {l2}")
    by_type = {}
    for (q, a, b), m in minw.items():
        typ = "pure-X" if b == 0 else ("pure-Z" if a == 0 else "mixed")
        by_type.setdefault((q, typ), set()).add(m)
    print("  minimum partner weight by (wire, type):")
    for key in sorted(by_type):
        print(f"    wire {key[0]} {key[1]:6s}: {sorted(by_type[key])}")


def exact_joint(q):
    """P(s, L) for code capacity, L = b0 + b3 (readout shift)."""
    single = [(1 - q, (0, 0))] + [(q / 8, ab) for ab in
                                    itertools.product(range(3), repeat=2) if ab != (0, 0)]
    P = np.zeros((81, 3))
    for combo in itertools.product(range(9), repeat=NQ):
        prob = 1.0
        xs, zs = {}, {}
        for qq, idx in enumerate(combo):
            p, (a, b) = single[idx] if idx < len(single) else (0, (0, 0))
            prob *= p
            xs[qq], zs[qq] = a, b
        e = vec(xs, zs)
        s = syndrome(e)
        sidx = s[0] * 27 + s[1] * 9 + s[2] * 3 + s[3]
        L = (e[NQ + 0] + e[NQ + 3]) % 3
        P[sidx, L] += prob
    return P


def evaluate(P):
    ps = P.sum(axis=1)
    a = (P * w ** np.arange(3)).sum(axis=1)       # unnormalized E[w^L ; s]
    res = {}
    res["none"] = a.sum()
    tot_map_sym, tot_map_det, tot_anti = 0j, 0j, 0j
    for s in range(81):
        if ps[s] <= 0:
            continue
        mx = P[s].max()
        ties = np.flatnonzero(P[s] >= mx * (1 - 1e-12))
        tot_map_sym += np.mean([w ** (-k) for k in ties]) * a[s]
        tot_map_det += w ** (-ties[0]) * a[s]
        mn = P[s].min()
        lows = np.flatnonzero(P[s] <= mn * (1 + 1e-12) + 1e-300)
        tot_anti += np.mean([w ** (-k) for k in lows]) * a[s]
    res["MAP-sym"], res["MAP-det"], res["anti-MAP"] = tot_map_sym, tot_map_det, tot_anti
    best, bphi = 0.0, 0.0
    for phi in np.linspace(0, 2 * np.pi / 3, 2001):
        val = sum(max((np.exp(-1j * phi) * w ** (-k) * a[s]).real for k in range(3))
                  for s in range(81) if ps[s] > 0)
        if val > best:
            best, bphi = val, phi
    res["|.|-optimal"] = best * np.exp(1j * bphi)
    return res


def part2():
    print("\nPart 2: |E w^R| and Re E w^R for several decoders, exact code capacity")
    print(f"  {'q':>5s} " + "".join(f"{d:>22s}" for d in
                                   ["none", "MAP-sym", "MAP-det", "anti-MAP", "|.|-optimal"]))
    for q in (0.01, 0.05, 0.15, 0.3, 0.5, 0.7):
        P = exact_joint(q)
        r = evaluate(P)
        line = f"  {q:5.2f} "
        for d in ["none", "MAP-sym", "MAP-det", "anti-MAP", "|.|-optimal"]:
            z = r[d]
            line += f"  {abs(z):.5f}/{z.real:+.5f}"
        print(line)
    print("  (each entry: |E w^R| / Re E w^R)")


def part3():
    print("\nPart 3: first-order code-capacity failure, X_L readout, 1 - Re E w^R ~ c q")
    for q in (1e-4, 1e-3):
        r = evaluate(exact_joint(q))
        print(f"  q={q:g}: none c = {(1 - r['none'].real) / q:.3f}, "
              f"MAP-sym c = {(1 - r['MAP-sym'].real) / q:.3f}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
