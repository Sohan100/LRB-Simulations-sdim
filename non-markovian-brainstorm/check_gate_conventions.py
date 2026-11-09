"""Check that variant_sim's qutrit token unitaries match sdim's gate actions.

Prepares |0> (Z basis) or |+> (X basis) on one qutrit, applies a random word
of H, P, X, Z and their inverses with sdim, measures, and compares with the
numpy prediction: a deterministic prediction must give a deterministic sdim
outcome with the same value, and vice versa. A Clifford that maps both |0>
and |+> correctly equals the identity up to phase, so this pins down the
recovery-Clifford lookup used by variant_sim.py.
"""
from __future__ import annotations

import random

import numpy as np
from sdim.circuit import Circuit
from sdim.program import Program

from variant_sim import _TOKEN

U = dict(_TOKEN)
for k in ("H", "P", "X", "Z"):
    U[k + "_INV"] = _TOKEN[k].conj().T


def sdim_outcome(tokens, basis):
    c = Circuit(1, 3)
    if basis == "X":
        c.add_gate("H", 0)
    for t in tokens:
        c.add_gate(t, 0)
    if basis == "X":
        c.add_gate("H_INV", 0)
    c.add_gate("M", 0)
    pr = Program(c)
    pr.simulate(shots=1)
    return pr.measurement_results[0][0][0]


def main(n_words=400, seed=1):
    rng = random.Random(seed)
    bad = 0
    for _ in range(n_words):
        toks = [rng.choice(list(U)) for _ in range(rng.randint(1, 7))]
        M = np.eye(3, dtype=complex)
        for t in toks:
            M = U[t] @ M
        for basis in ("Z", "X"):
            psi = np.zeros(3, dtype=complex)
            psi[0] = 1
            if basis == "X":
                psi = U["H"] @ psi
            out = M @ psi
            if basis == "X":
                out = U["H_INV"] @ out
            pred = np.abs(out) ** 2
            m = sdim_outcome(toks, basis)
            det = pred.max() > 0.99
            if det != bool(m.deterministic) or (det and m.measurement_value != int(np.argmax(pred))):
                bad += 1
    print(f"{bad} mismatches in {2 * n_words} checks")


if __name__ == "__main__":
    main()
