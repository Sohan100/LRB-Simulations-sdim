"""Compiled recovery Clifford for single-qutrit RB and LRB sequences.

Standard RB closes a random sequence C1..Cm with one recovery Clifford equal
to (Cm ... C1)^-1, instead of the mirror inverses Cm^-1 .. C1^-1 that the
repository uses by default. This module finds that recovery element in a
code definition's Clifford catalog by multiplying 3x3 unitaries, so no code
definition has to change.

Token conventions match SDIM's qutrit gates (checked against SDIM's
``Program`` in ``tests/test_compiled_recovery.py``): ``X|j> = |j+1>``,
``Z = diag(1, w, w^2)``, ``H_{jk} = w^{jk}/sqrt(3)``, ``P = diag(1, 1, w)``
with ``w = exp(2 pi i / 3)``; ``*_INV`` is the adjoint.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

_W = np.exp(2j * np.pi / 3)
_BASE_TOKENS = {
    "I": np.eye(3, dtype=complex),
    "X": np.roll(np.eye(3), 1, axis=0).astype(complex),
    "Z": np.diag([1, _W, _W ** 2]),
    "H": np.array([[_W ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(3),
    "P": np.diag([1, 1, _W]),
}
QUTRIT_TOKEN_UNITARIES = {
    **_BASE_TOKENS,
    **{f"{name}_INV": u.conj().T for name, u in _BASE_TOKENS.items() if name != "I"},
}


def gate_list_unitary(gates: Iterable[str]) -> np.ndarray:
    """Unitary of an execution-order token list (first token acts first)."""
    u = np.eye(3, dtype=complex)
    for gate in gates:
        if gate not in QUTRIT_TOKEN_UNITARIES:
            raise ValueError(f"No qutrit unitary for gate token '{gate}'.")
        u = QUTRIT_TOKEN_UNITARIES[gate] @ u
    return u


def _phase_key(u: np.ndarray) -> tuple:
    """Hashable key of ``u`` up to global phase."""
    flat = u.reshape(-1)
    pivot = flat[int(np.argmax(np.abs(flat) > 1e-9))]
    v = flat * (abs(pivot) / pivot)
    return tuple(np.round(v.real, 6)) + tuple(np.round(v.imag, 6))


class CliffordRecoveryTable:
    """Look up the catalog Clifford that inverts a product of catalog Cliffords.

    Args:
        clifford_strings: Catalog descriptors, e.g. a code definition's
            ``clifford_strings``.
        clifford_to_gate_sequence: Maps a descriptor to its execution-order
            token list, e.g. a code definition's ``clifford_to_gate_sequence``.
        dimension: Qudit dimension; only 3 is supported.
    """

    def __init__(self, clifford_strings: Sequence[str], clifford_to_gate_sequence, dimension: int = 3) -> None:
        if dimension != 3:
            raise ValueError("Compiled recovery is implemented for qutrits only (dimension 3).")
        self._to_gates = clifford_to_gate_sequence
        self._unitary = {w: gate_list_unitary(clifford_to_gate_sequence(w)) for w in clifford_strings}
        self._by_key = {}
        for word, u in self._unitary.items():
            self._by_key.setdefault(_phase_key(u), word)
        if len(self._by_key) != len(self._unitary):
            raise ValueError("Clifford catalog contains duplicate group elements up to phase.")

    def __len__(self) -> int:
        return len(self._by_key)

    def recovery_word(self, words: Sequence[str]) -> str:
        """Catalog descriptor R with R (Cm ... C1) = I up to phase; C1 = words[0]."""
        total = np.eye(3, dtype=complex)
        for word in words:
            total = self._unitary[word] @ total
        key = _phase_key(total.conj().T)
        if key not in self._by_key:
            raise ValueError("Catalog does not contain the inverse of this product; is it a group?")
        return self._by_key[key]

    def recovery_gates(self, words: Sequence[str]) -> list[str]:
        """Execution-order token list of ``recovery_word(words)``."""
        return list(self._to_gates(self.recovery_word(words)))
