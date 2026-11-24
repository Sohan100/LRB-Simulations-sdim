"""Exact statevector check that an encoder prepares the paper's |+_L>.

Conventions as in sdim 1.4.0: X|j> = |j+1>, CNOT c->t: |c,t> -> |c,t+c>,
CNOT_INV subtracts, H is the Fourier transform (H|0> = sum_j |j>/sqrt(d)).
The reference is the paper's encoder (code_definitions.logical_plus_initial_
state: H_INV on q2, q3, q4; CNOT_INV 2->1; CNOT 1->0; CNOT 3->0; CNOT 4->2).
Helpers must end in a product state (|0> or |+>); the check reports the
overlap of the data with the reference and the purity of the data state.
"""
import numpy as np


def apply(psi, n, d, name, *w):
    psi = psi.reshape((d,) * n)
    if name in ("CNOT", "CNOT_INV"):
        c, t = w
        s = 1 if name == "CNOT" else -1
        out = np.empty_like(psi)
        for v in range(d):
            idx = [slice(None)] * n
            idx[c] = v
            out[tuple(idx)] = np.roll(psi[tuple(idx)], s * v, axis=t if t < c else t - 1)
        return out.reshape(-1)
    W = np.exp(2j * np.pi / d)
    F = np.array([[W ** (j * k) for j in range(d)] for k in range(d)]) / np.sqrt(d)
    M = F if name == "H" else F.conj().T
    return np.moveaxis(np.tensordot(M, psi, axes=([1], [w[0]])), 0, w[0]).reshape(-1)


def run(gates, n, d):
    psi = np.zeros(d ** n, dtype=complex)
    psi[0] = 1
    for g in gates:
        psi = apply(psi, n, d, *g)
    return psi


PAPER = [("H_INV", 2), ("H_INV", 3), ("H_INV", 4), ("CNOT_INV", 2, 1),
         ("CNOT", 1, 0), ("CNOT", 3, 0), ("CNOT", 4, 2)]


def check(gates, n, d):
    """Return (fidelity of data with paper |+_L>, data purity)."""
    ref = run(PAPER, 5, d)
    psi = run(gates, n, d).reshape(d ** 5, d ** (n - 5))
    rho = psi @ psi.conj().T                      # reduced data state
    fid = np.real(ref.conj() @ rho @ ref)
    purity = np.real(np.trace(rho @ rho))
    return fid, purity


def from_search(start_plus, moves):
    """Gate list from encoder_search output: |+> on start_plus, then moves."""
    return [("H", q) for q in start_plus] + [
        ("CNOT" if s == 1 else "CNOT_INV", c, t) for c, t, s in moves]


if __name__ == "__main__":
    for d in (3, 2):
        print(f"d={d} paper encoder vs itself:", check(PAPER, 5, d))
