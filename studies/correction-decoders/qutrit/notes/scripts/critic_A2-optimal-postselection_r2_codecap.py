#!/usr/bin/env python3
"""Critic (round 2) of A2-optimal-postselection: code-capacity claims (C6).

Independent of A2_optimal_rule_check.py:
  * syndromes and the logical shift come from symplectic products with the
    stabilizer / logical-X vectors (not hand-written linear forms), after
    checking that the vectors commute as they should;
  * P(D, l) is a polynomial in p: P = sum_w N_w(D, l) (1-p)^(5-w) (p/8)^w with
    INTEGER counts N_w, evaluated in exact rational arithmetic (fractions),
    so ties and rank comparisons are exact, not tolerance-based.

Run: module load python/3.11 && python critic_A2-optimal-postselection_r2_codecap.py
"""
from __future__ import annotations

import itertools
from fractions import Fraction as Fr

import numpy as np

# (x | z) exponent vectors on data qutrits 0..4 (mod 3)
def vec(x=(0,) * 5, z=(0,) * 5):
    return np.array(list(x) + list(z)) % 3


STABS = {
    "S_Z1 = Z0 Z1^-1 Z3^-1": vec(z=(1, -1, 0, -1, 0)),
    "S_Z2 = Z1 Z2 Z4^-1": vec(z=(0, 1, 1, 0, -1)),
    "S_X1 = X0 X1 X2^-1": vec(x=(1, 1, -1, 0, 0)),
    "S_X2 = X1^-1 X3 X4^-1": vec(x=(0, -1, 0, 1, -1)),
}
XL = vec(x=(1, 0, 0, 1, 0))     # logical X = X0 X3; readout = x0 + x3
ZL = vec(z=(1, 0, 1, 0, 0))     # logical Z = Z0 Z2


def symp(u, v):
    """Symplectic form <u, v> = u_x . v_z - u_z . v_x (mod 3)."""
    return int(u[:5] @ v[5:] - u[5:] @ v[:5]) % 3


def check_code():
    S = list(STABS.values())
    ok = all(symp(a, b) == 0 for a in S for b in S)
    ok &= all(symp(s, XL) == 0 and symp(s, ZL) == 0 for s in S)
    ok &= symp(XL, ZL) != 0
    print(f"  stabilizers commute, logicals commute with stabilizers, "
          f"<XL,ZL> != 0: {ok}")
    return ok


def counts(z_checks=True):
    """Integer counts N[w][(D, l)] over all 9^5 error configurations."""
    S = list(STABS.values())
    if not z_checks:
        S = S[2:]
    single = [(a, b) for a in range(3) for b in range(3)]
    N = [dict() for _ in range(6)]
    for conf in itertools.product(range(9), repeat=5):
        a = np.array([single[k][0] for k in conf])
        b = np.array([single[k][1] for k in conf])
        e = np.concatenate([a, b]) % 3
        D = tuple(symp(s, e) for s in S)
        l = symp(XL, e)          # shift of the X-basis readout x0 + x3
        w = sum(1 for k in conf if k != 0)
        N[w][(D, l)] = N[w].get((D, l), 0) + 1
    return N, len(S)


def joint_exact(N, p):
    P = {}
    for w in range(6):
        fac = (1 - p) ** (5 - w) * (p / 8) ** w
        for key, c in N[w].items():
            P[key] = P.get(key, 0) + c * fac
    J = {}
    for (D, l), v in P.items():
        J.setdefault(D, [Fr(0)] * 3)[l] += v
    return J


def analyse(J, nd):
    D0 = (0,) * nd
    pD = {D: sum(v) for D, v in J.items()}
    q = {D: max(v) / pD[D] for D, v in J.items()}
    # detection: accept D=0 only, c=0
    a_det = pD[D0]
    S_det = J[D0][0] / a_det
    above = sum(pD[D] for D in J if q[D] > q[D0])
    equal = sum(pD[D] for D in J if q[D] == q[D0] and D != D0)
    map_fail = 1 - sum(max(v) for v in J.values())
    ties = {"0 & one nonzero": 0, "1 & 2 only": 0, "three-way": 0}
    for D, v in J.items():
        m = max(v)
        arg = [l for l in range(3) if v[l] == m]
        if len(arg) == 3:
            ties["three-way"] += 1
        elif len(arg) == 2:
            ties["0 & one nonzero" if 0 in arg else "1 & 2 only"] += 1
    # conjugation symmetry P(D,l) = P(-D,-l), exact
    sym = all(J[tuple((-x) % 3 for x in D)][(-l) % 3] == J[D][l]
              for D in J for l in range(3))
    # frontier at alpha = 1, 0.98, 0.99 (exact)
    order = sorted(J, key=lambda D: -q[D])
    def Sstar(alpha):
        acc, val = Fr(0), Fr(0)
        for D in order:
            take = min(pD[D], alpha - acc)
            if take <= 0:
                break
            acc += take
            val += take * q[D]
        return val / alpha
    return dict(a_det=a_det, S_det=S_det, above=above, equal=equal,
                map_fail=map_fail, ties=ties, sym=sym,
                Sstar_det=Sstar(a_det), Sstar=Sstar)


def main():
    print("Code capacity [[5,1,2]]_3, i.i.d. single-qutrit depolarizing "
          "(p/8 per X^aZ^b), perfect checks, exact arithmetic")
    assert check_code()
    for z_checks in (True, False):
        N, nd = counts(z_checks)
        for p in (Fr(1, 1000), Fr(1, 100), Fr(1, 20)):
            J = joint_exact(N, p)
            r = analyse(J, nd)
            tag = "Z+X" if z_checks else "X  "
            print(f"  p={float(p):<6g} {tag} a_det={float(r['a_det']):.4f} "
                  f"1-S_det={float(1 - r['S_det']):.3e} "
                  f"1-S*(a_det)={float(1 - r['Sstar_det']):.3e} "
                  f"mass(q>q0)={float(r['above']):.1e} "
                  f"mass(q==q0, D!=0)={float(r['equal']):.1e} "
                  f"MAP fail={float(r['map_fail']):.4e} ties={r['ties']} "
                  f"P(D,l)=P(-D,-l) exactly: {r['sym']}")
            if z_checks and p == Fr(1, 100):
                print("    frontier S*: " + ", ".join(
                    f"{a}->{float(r['Sstar'](Fr(a))):.5f}"
                    for a in ("0.95", "0.98", "0.99", "1")))
                fail_zx = r["map_fail"]
            if (not z_checks) and p == Fr(1, 100):
                print(f"    ratio MAP fail (X only)/(Z+X) at p=0.01: "
                      f"{float(r['map_fail'] / fail_zx):.3f}")


if __name__ == "__main__":
    main()
