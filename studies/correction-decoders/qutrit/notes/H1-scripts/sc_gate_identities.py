"""H1 (rewrite) - gate identities used in the hardware compilation.

Matrices are taken from sdim.unitary (the repo's conventions), via cirq.

Checks:
  I1  CNOT (CSUM) = (1 x H^-1) CZ (1 x H)       and   = (1 x H) CZ^dag (1 x H^-1)
  I2  CNOT_INV    = (1 x H^-1) CZ^dag (1 x H)   and   = (1 x H) CZ (1 x H^-1)
      -> every repo two-qutrit gate can use the faster native CZ^dag
         (Goss et al. 2022: CZ^dag 580 ns, CZ 783 ns) with single-qutrit
         basis changes on the target.
  I3  P, P_INV, Z, Z_INV are diagonal (virtual-Z frame updates on transmons).
  I4  logical H token without its SWAP(2,3): (H on 0,2,3,4; H_INV on 1) maps
      the code stabilizer group onto the 2<->3-relabelled group (so the SWAP
      can be a classical relabelling), checked on the 4 generators.
  I5  SI1000 two-qutrit depolarizing with probability p has process fidelity
      1 - p (so p is directly the Weyl/process error rate reported by cycle
      benchmarking).
  I6  level bookkeeping of the IQM pulse-library reset composite
      (prx_12(pi) then f0g1) on a qutrit, versus f0g1, prx_12(pi), f0g1.

Run: module load python/3.11; python sc_gate_identities.py   (seconds)
"""
from __future__ import annotations

import itertools

import cirq
import numpy as np
import sdim.unitary as U

D = 3
H = cirq.unitary(U.GeneralizedHadamardGate(D))
Hi = H.conj().T
P = cirq.unitary(U.GeneralizedPhaseShiftGate(D))
CZ = cirq.unitary(U.GeneralizedCZGate(D))
CNOT = cirq.unitary(U.GeneralizedCNOTGate(D))
CNOTi = cirq.unitary(U.GeneralizedCNOTGateInverse(D))
I3 = np.eye(D)
W = np.exp(2j * np.pi / 3)
X = np.roll(np.eye(D), 1, axis=0)
Z = np.diag([W ** j for j in range(D)])


def close(a, b):
    return np.allclose(a, b, atol=1e-10)


def main():
    print("I1:", close(CNOT, np.kron(I3, Hi) @ CZ @ np.kron(I3, H)),
          close(CNOT, np.kron(I3, H) @ CZ.conj().T @ np.kron(I3, Hi)))
    print("I2:", close(CNOTi, np.kron(I3, Hi) @ CZ.conj().T @ np.kron(I3, H)),
          close(CNOTi, np.kron(I3, H) @ CZ @ np.kron(I3, Hi)))
    print("I3: P diagonal", close(P, np.diag(np.diag(P))), "; P =", np.round(np.diag(P), 3))

    # I4: conjugate generators (as 5-qutrit Weyl vectors) by the single-qutrit
    # layer; Weyl conjugation rules from matrices.
    def weyl(a, b):
        return np.linalg.matrix_power(X, a % 3) @ np.linalg.matrix_power(Z, b % 3)

    def conj_rule(Ug):
        rule = {}
        for a, b in itertools.product(range(3), repeat=2):
            C = Ug @ weyl(a, b) @ Ug.conj().T
            for c, d in itertools.product(range(3), repeat=2):
                if abs(abs(np.trace(weyl(c, d).conj().T @ C)) - 3) < 1e-9:
                    rule[(a, b)] = (c, d)
        return rule
    rH, rHi = conj_rule(H), conj_rule(Hi)
    gens = {"SZ1": ([0] * 5, [1, 2, 0, 2, 0]), "SZ2": ([0] * 5, [0, 1, 1, 0, 2]),
            "SX1": ([1, 1, 2, 0, 0], [0] * 5), "SX2": ([0, 2, 0, 1, 2], [0] * 5)}
    def apply_layer(v):
        x, z = list(v[0]), list(v[1])
        for q in range(5):
            r = rHi if q == 1 else rH
            x[q], z[q] = r[(x[q], z[q])]
        return x, z
    def relabel(v):
        x, z = list(v[0]), list(v[1])
        x[2], x[3] = x[3], x[2]
        z[2], z[3] = z[3], z[2]
        return x, z
    # stabilizer group spanned (mod 3) by generators: compare spans
    def span(vs):
        out = set()
        for co in itertools.product(range(3), repeat=len(vs)):
            x = tuple(sum(c * v[0][q] for c, v in zip(co, vs)) % 3 for q in range(5))
            z = tuple(sum(c * v[1][q] for c, v in zip(co, vs)) % 3 for q in range(5))
            out.add((x, z))
        return out
    target = span([relabel(v) for v in gens.values()])
    image = span([apply_layer(v) for v in gens.values()])
    print("I4: single-qutrit part of logical H maps the stabilizer group onto its"
          " 2<->3 relabelling:", image == target)
    XL = ([1, 0, 0, 1, 0], [0] * 5)
    ZL = ([0] * 5, [1, 0, 1, 0, 0])
    print("    images of X_L, Z_L under the layer:", apply_layer(XL), apply_layer(ZL))

    p = 0.05
    # uniform two-qutrit depolarizing: F_pro = sum_k |Tr K_k|^2 / d^2
    kraus = []
    for v in itertools.product(range(3), repeat=4):
        Wv = np.kron(weyl(v[0], v[1]), weyl(v[2], v[3]))
        w = (1 - p) if not any(v) else p / 80
        kraus.append(np.sqrt(w) * Wv)
    fpro = sum(abs(np.trace(K)) ** 2 for K in kraus) / 81
    print("I5: F_pro of 2-qutrit SI1000 depolarizing (p=0.05), computed:", round(fpro, 12))

    # I6 level bookkeeping: f0g1 maps |2> -> |0> (photon then decays), leaves
    # |0>,|1>; prx_12(pi) swaps |1> <-> |2> (up to phase).
    f0g1 = {0: 0, 1: 1, 2: 0}
    prx12 = {0: 0, 1: 2, 2: 1}
    comp = lambda *maps: (lambda s: [s := m[s] for m in maps][-1])
    iqm_composite = comp(prx12, f0g1)
    qutrit_reset = comp(f0g1, prx12, f0g1)
    print("I6: IQM Reset_F0G1_Composite (prx_12 then f0g1) on |0>,|1>,|2> ->",
          [iqm_composite(s) for s in range(3)],
          "; f0g1, prx_12, f0g1 ->", [qutrit_reset(s) for s in range(3)])


if __name__ == "__main__":
    main()
