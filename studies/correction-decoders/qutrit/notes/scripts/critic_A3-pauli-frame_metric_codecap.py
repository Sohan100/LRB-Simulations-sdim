#!/usr/bin/env python3
"""Critic of A3: does MAP maximise the brief's metric |sum_k w^k P(k)|, and does
correction give a first-order gain over no correction in the [[5,1,2]]_3 code?

Part A: two-syndrome counterexample (exact).
Part B: exact code-capacity model of the folded [[5,1,2]]_3 code (stabilisers and
  logicals from the brief), i.i.d. single-qutrit depolarising noise of strength q
  on the five data qutrits (uniform over the 8 non-identity X^a Z^b), perfect
  syndrome, then the brief's readout x0 + x3 (shift = b0 + b3).  All 9^5 error
  configurations are enumerated.  Decoders: none; MAP with inversion-symmetric
  (uniform random) tie-breaking; MAP with deterministic argmax (lowest index,
  as numpy.argmax and src/lrb/correction/decoders.py do); and the decoder that
  maximises |E w^R| exactly (maximum over the direction z of
  sum_s max_k Re(conj(z) w^-k a(s))).
Part C: weight-1 ambiguity table (A3-boundary-C5) recomputed from the
  stabiliser algebra.

Usage: module load python/3.11; python critic_A3-pauli-frame_metric_codecap.py
"""

from __future__ import annotations

import itertools

import numpy as np

w = np.exp(2j * np.pi / 3)

# stabiliser exponents (brief): S_Z1 = Z0 Z1^-1 Z3^-1, S_Z2 = Z1 Z2 Z4^-1,
# S_X1 = X0 X1 X2^-1, S_X2 = X1^-1 X3 X4^-1 ; X_L = X0 X3, Z_L = Z0 Z2
SZ = np.array([[1, -1, 0, -1, 0], [0, 1, 1, 0, -1]]) % 3
SX = np.array([[1, 1, -1, 0, 0], [0, -1, 0, 1, -1]]) % 3


def syndrome(a, b):
    """Commutation exponents of E = prod X^a Z^b with the four checks."""
    # X^a Z^b vs Z^c: phase exponent a*c (sign convention irrelevant for grouping)
    sz = (a @ SZ.T) % 3          # X part detected by Z checks
    sx = (-(b @ SX.T)) % 3       # Z part detected by X checks
    return np.concatenate([sz, sx], axis=-1)


def part_a():
    print("Part A: two conjugate syndromes s, -s with P(L|s)=(1/2,1/2,0), P(L|-s)=(1/2,0,1/2)")
    post = {+1: np.array([0.5, 0.5, 0.0]), -1: np.array([0.5, 0.0, 0.5])}

    def fid(rule):
        tot = 0
        for s, ps in post.items():
            for L in range(3):
                for lhat, pl in rule(s).items():
                    tot += 0.5 * ps[L] * pl * w ** ((L - lhat) % 3)
        return tot

    sym = fid(lambda s: {0: 0.5, 1: 0.5} if s == 1 else {0: 0.5, 2: 0.5})
    det = fid(lambda s: {0: 1.0})
    ali = fid(lambda s: {0: 1.0} if s == 1 else {2: 1.0})
    for name, v in (("symmetric MAP (random ties)", sym), ("argmax (lowest index)", det),
                    ("aligned (s->0, -s->2)", ali)):
        print(f"  {name:28s}: E[w^R] = {v.real:+.4f}{v.imag:+.4f}j  |E| = {abs(v):.4f}  "
              f"Re = {v.real:+.4f}")


def part_b():
    print("\nPart B: exact code capacity, readout shift = b0 + b3")
    vecs = np.array(list(itertools.product(range(3), repeat=10)))   # (a0,b0,...,a4,b4)
    a = vecs[:, 0::2]
    b = vecs[:, 1::2]
    nz = ((a != 0) | (b != 0)).sum(axis=1)       # number of faulty qutrits
    shift = (b[:, 0] + b[:, 3]) % 3
    syn = syndrome(a, b)
    sidx = syn @ np.array([27, 9, 3, 1])
    print("  q        decoder                 P(R!=0)     Re E w^R   |E w^R|")
    for q in (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3):
        pw = (q / 8) ** nz * (1 - q) ** (5 - nz)
        joint = np.zeros((81, 3))
        np.add.at(joint, (sidx, shift), pw)
        A = joint @ (w ** np.arange(3))        # a(s) = sum_L P(s,L) w^L
        rows = []
        # none
        rows.append(("none", A.sum(), 1 - joint[:, 0].sum()))
        # MAP with inversion-symmetric (uniform random) ties, and argmax ties
        e_sym = e_det = 0
        p_sym = p_det = 0
        for s in range(81):
            pj = joint[s]
            if pj.sum() == 0:
                continue
            mx = pj.max()
            ties = np.flatnonzero(pj >= mx * (1 - 1e-12))
            e_sym += np.mean([w ** (-k) for k in ties]) * A[s]
            p_sym += mx
            k0 = int(np.argmax(pj))
            e_det += w ** (-k0) * A[s]
            p_det += pj[k0]
        rows.append(("MAP, symmetric ties", e_sym, 1 - p_sym))
        rows.append(("MAP, argmax ties", e_det, 1 - p_det))
        # |.|-optimal: max over directions z of sum_s max_k Re(conj z w^-k a(s))
        best, best_e, best_p = 0, None, None
        for psi in np.linspace(0, 2 * np.pi / 3, 2401):
            z = np.exp(1j * psi)
            vals = np.stack([np.real(np.conj(z) * w ** (-k) * A) for k in range(3)])
            kk = vals.argmax(axis=0)
            e = np.sum(w ** (-kk) * A)
            if abs(e) > best:
                best, best_e = abs(e), e
                best_p = 1 - joint[np.arange(81), kk].sum()
        rows.append(("|.|-optimal decoder", best_e, best_p))
        for name, e, pf in rows:
            print(f"  {q:<8g} {name:22s} {pf:.6e}  {e.real:+.6f}  {abs(e):.6f}")
        print(f"  {'':8s} ratio P(R!=0) none / MAP = {rows[0][2] / rows[1][2]:.3f}; "
              f"|.|-optimal minus symmetric MAP = {abs(best_e) - abs(e_sym):.3e}")


def part_c():
    print("\nPart C: weight-1 faults with a same-syndrome weight-1 partner in another shift class")
    singles = []
    for q in range(5):
        for (x, z) in itertools.product(range(3), repeat=2):
            if (x, z) == (0, 0):
                continue
            a = np.zeros(5, int); b = np.zeros(5, int)
            a[q], b[q] = x, z
            singles.append((q, x, z, tuple(syndrome(a[None], b[None])[0]), a, b))
    for q, x, z, s, a, b in singles:
        partners = []
        for q2, x2, z2, s2, a2, b2 in singles:
            if q2 == q or s2 != s:
                continue
            # residual E1 - E2: logical class from commutation with X_L and Z_L
            da, db = (a - a2) % 3, (b - b2) % 3
            lx = (db[0] + db[3]) % 3                 # vs X_L = X0 X3
            lz = (da[0] + da[2]) % 3                 # vs Z_L = Z0 Z2
            partners.append(f"X{q2}^{x2}Z{q2}^{z2}(dL={lx},{lz})")
        kind = "pure-X" if z == 0 else ("pure-Z" if x == 0 else "mixed")
        if partners:
            print(f"  X{q}^{x} Z{q}^{z} [{kind}] syndrome {s}: partners {partners}")
    print("  (faults not listed have no weight-1 partner with the same syndrome)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
