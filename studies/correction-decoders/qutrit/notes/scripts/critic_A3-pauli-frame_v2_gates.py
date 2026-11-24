#!/usr/bin/env python3
"""Critic (v2) of A3-pauli-frame-C2: qutrit frame-update table versus sdim itself.

Independent of the A3 script: the unitaries are taken from sdim's own
`sdim.unitary` generators (not re-typed), CZ is taken from sdim's
`GeneralizedCZGate._unitary_`, and SWAP is rebuilt from the exact gate sequence
that `sdim/tableau/tableau_gates.py::apply_SWAP` applies for the prime-dimension
tableau (ExtendedTableau: CNOT; H a; H b; CNOT; H a; H b; CNOT; H b; H b) and
for the WeylTableau branch (CNOT; CNOT_INV(b->a); CNOT; H a; H a).

Checks
  1. Symplectic action U W(v) U^dag = phase * W(S v) of every gate, compared with
     the closed-form rules claimed in A3 section 1.1 (all 9 or 81 input vectors).
  2. Both sdim SWAP decompositions versus the standard SWAP matrix (exact
     equality, not only up to a phase or a Weyl factor).
  3. Terminal readout: frame (a,b) on wire q before H_INV q has X-power b after it.

Usage: module load python/3.11; python critic_A3-pauli-frame_v2_gates.py
"""

from __future__ import annotations

import itertools

import numpy as np
from sdim import unitary as su

D = 3
I3 = np.eye(3, dtype=complex)
X = su.generate_x_matrix(D)
Z = su.generate_z_matrix(D)
H = su.generate_h_matrix(D)
P = su.generate_p_matrix(D)
CNOT = su.generate_cnot_matrix(D)          # control = first (most significant)
CZ = su.GeneralizedCZGate(D)._unitary_()
Hd, Pd, CNOTd, CZd = (m.conj().T for m in (H, P, CNOT, CZ))


def W1(a, b):
    return np.linalg.matrix_power(X, a % 3) @ np.linalg.matrix_power(Z, b % 3)


def Wn(v):
    out = np.array([[1.0 + 0j]])
    for q in range(len(v) // 2):
        out = np.kron(out, W1(v[2 * q], v[2 * q + 1]))
    return out


def weyl_of(m, nq):
    for v in itertools.product(range(3), repeat=2 * nq):
        w = Wn(v)
        k = np.argmax(np.abs(w))
        c = m.flat[k] / w.flat[k]
        if abs(abs(c) - 1) < 1e-9 and np.allclose(m, c * w, atol=1e-9):
            return tuple(v)
    return None


# A3's claimed closed-form rules (section 1.1)
RULES1 = {
    "H": lambda a, b: (-b, a), "H_INV": lambda a, b: (b, -a),
    "P": lambda a, b: (a, a + b), "P_INV": lambda a, b: (a, b - a),
    "X": lambda a, b: (a, b), "Z": lambda a, b: (a, b),
}
RULES2 = {
    "CNOT": lambda ac, bc, at, bt: (ac, bc - bt, at + ac, bt),
    "CNOT_INV": lambda ac, bc, at, bt: (ac, bc + bt, at - ac, bt),
    "CZ": lambda ac, bc, at, bt: (ac, bc + at, at, bt + ac),
    "CZ_INV": lambda ac, bc, at, bt: (ac, bc - at, at, bt - ac),
    "SWAP": lambda ac, bc, at, bt: (at, bt, ac, bc),
}


def check(name, u, nq, rule):
    bad = 0
    for v in itertools.product(range(3), repeat=2 * nq):
        img = weyl_of(u @ Wn(v) @ u.conj().T, nq)
        want = tuple(x % 3 for x in rule(*v))
        if img != want:
            bad += 1
    print(f"  {name:9s}: {3 ** (2 * nq)} input Weyl vectors, mismatches with A3 rule: {bad}")
    return bad


def main():
    k = np.kron
    # sdim tableau SWAP, prime branch (ExtendedTableau): operations in time order
    ops_prime = [CNOT, k(H, I3), k(I3, H), CNOT, k(H, I3), k(I3, H), CNOT,
                 k(I3, H), k(I3, H)]
    # WeylTableau branch: cnot(a,b); cnot_inv(b,a); cnot(a,b); H a; H a
    SWAPm = np.zeros((9, 9), complex)
    for i in range(3):
        for j in range(3):
            SWAPm[3 * j + i, 3 * i + j] = 1
    CNOT_ba_inv = SWAPm @ CNOTd @ SWAPm      # control b, target a, inverse
    ops_weyl = [CNOT, CNOT_ba_inv, CNOT, k(H, I3), k(H, I3)]
    u_prime = np.eye(9, dtype=complex)
    for op in ops_prime:
        u_prime = op @ u_prime
    u_weyl = np.eye(9, dtype=complex)
    for op in ops_weyl:
        u_weyl = op @ u_weyl
    print("1. Symplectic action of sdim's gates versus the A3 table")
    bad = 0
    for name, u in (("H", H), ("H_INV", Hd), ("P", P), ("P_INV", Pd),
                    ("X", X), ("Z", Z)):
        bad += check(name, u, 1, RULES1[name])
    for name, u in (("CNOT", CNOT), ("CNOT_INV", CNOTd), ("CZ", CZ),
                    ("CZ_INV", CZd), ("SWAP", SWAPm)):
        bad += check(name, u, 2, RULES2[name])
    print(f"   CZ from sdim equals diag(w^(ij)): "
          f"{np.allclose(CZ, np.diag([np.exp(2j*np.pi*i*j/3) for i in range(3) for j in range(3)]))}")
    print("2. sdim tableau SWAP decompositions versus the standard SWAP")
    for lab, u in (("prime/ExtendedTableau", u_prime), ("WeylTableau", u_weyl)):
        print(f"   {lab:22s}: max |U - SWAP| = {np.max(np.abs(u - SWAPm)):.2e}")
    print("3. Terminal readout: X-power after H_INV of a frame (a,b) equals b")
    ok = all(weyl_of(Hd @ W1(a, b) @ H, 1)[0] == b % 3
             for a in range(3) for b in range(3))
    print(f"   holds for all 9 frames: {ok}")
    print(f"TOTAL rule mismatches: {bad}")


if __name__ == "__main__":
    main()
