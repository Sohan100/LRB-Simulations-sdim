"""Unrotated planar surface codes with fold-transversal H and S (d = 2, 3, ...).

The repo's qubit ``[[5,1,2]]`` (``lrb_qubit.code``) is the distance-2 member of
the unrotated planar surface-code family that is symmetric under the diagonal
fold (Moussa 2016, arXiv:1603.02286; Breuckmann and Burton 2024,
arXiv:2202.06647). This module builds any member of the family from its
geometry, with ``CODE13 = UnrotatedSurfaceCode(3)`` the ``[[13,1,3]]`` code
used by work package G.

Geometry. Sites ``(r, c)`` with ``0 <= r, c <= 2d - 2``:

* data qubits at ``r + c`` even (``d**2 + (d-1)**2`` of them);
* Z checks ("plaquettes") at ``r`` even, ``c`` odd;
* X checks ("stars") at ``r`` odd, ``c`` even.

Each check acts on its (up to four) nearest data neighbours. The transpose
``(r, c) -> (c, r)`` maps data to data and swaps the Z and X checks, which is
the diagonal fold. ``X_L`` is X on row 0 and ``Z_L`` is Z on column 0.

Fold-transversal gates (verified with Stim tableaus in
``tests/test_qubit512_d3.py``):

* logical H = SWAP on every mirror pair ``(q, q^T)``, then H on every data
  qubit;
* logical S = S on the diagonal qubits ``(i, i)`` with ``i`` even, S-dagger on
  those with ``i`` odd, and CZ on every mirror pair.

The CNOT order inside each check follows the compass directions that
reproduce the ``[[5,1,2]]`` check circuits of ``lrb_qubit.code`` exactly
(Z checks N, W, S, E; X checks N, W, E, S). With
``labels=FIVE_QUBIT_LABELS`` the ``d = 2`` instance has the same qubit
labels, stabilizer order, logical operators, checks and logical S, X, Z
gate lists as ``lrb_qubit.code``; its logical H applies the same gates with
the five H gates in a different order.

An operation is ``(stim_gate_name, targets)``, as in ``lrb_qubit.code``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import stim

from . import code as C

Operation = tuple[str, tuple[int, ...]]
Site = tuple[int, int]

# Compass steps. "N" is towards row 0, "W" towards column 0.
_STEPS = {"N": (-1, 0), "W": (0, -1), "S": (1, 0), "E": (0, 1)}
DEFAULT_Z_ORDER = ("N", "W", "S", "E")
DEFAULT_X_ORDER = ("N", "W", "E", "S")

# Labels under which the d = 2 instance coincides with ``lrb_qubit.code``.
FIVE_QUBIT_LABELS: dict[Site, int] = {(0, 0): 0, (1, 1): 1, (2, 0): 2,
                                       (0, 2): 3, (2, 2): 4}

_TOKEN_INVERSE_GATE = {"I": "I", "X": "X", "Z": "Z", "H": "H", "S": "S_DAG",
                       "S_DAG": "S", "SWAP": "SWAP", "CZ": "CZ"}


class UnrotatedSurfaceCode:
    """Distance-``d`` unrotated planar surface code with fold-transversal H, S.

    Args:
        distance: Code distance ``d >= 2``.
        labels: Optional map from data sites to labels ``0..n-1``. The
            default numbers the data sites row by row.
        z_order, x_order: Compass order of the CNOTs inside each Z and X
            check (permutations of ``"NWSE"``).

    Attributes:
        num_data, num_stabilizers, num_qubits: Sizes. Ancilla of stabilizer
            ``s`` is wire ``num_data + s``.
        stabilizer_paulis: Data-qubit Pauli strings, Z checks first (row-major
            sites), then X checks (row-major sites).
        z_stabilizers, x_stabilizers: Index ranges of the two types.
        logical_x, logical_z: ``X_L`` (row 0) and ``Z_L`` (column 0).
        fold_pairs: Mirror pairs ``(a, b)`` with ``a < b`` (labels).
        diagonal_gates: ``(gate, label)`` for the S-type gate on the fold line.
    """

    def __init__(self, distance: int,
                 labels: Optional[Mapping[Site, int]] = None,
                 z_order: Sequence[str] = DEFAULT_Z_ORDER,
                 x_order: Sequence[str] = DEFAULT_X_ORDER):
        if distance < 2:
            raise ValueError("distance must be at least 2.")
        for order in (z_order, x_order):
            if sorted(order) != sorted("NWSE"):
                raise ValueError("CNOT orders must be permutations of NWSE.")
        self.distance = distance
        self.size = 2 * distance - 1
        sites = [(r, c) for r in range(self.size) for c in range(self.size)
                 if (r + c) % 2 == 0]
        if labels is None:
            labels = {site: i for i, site in enumerate(sites)}
        if sorted(labels) != sorted(sites) or sorted(labels.values()) != list(
                range(len(sites))):
            raise ValueError("labels must number every data site 0..n-1.")
        self.labels = dict(labels)
        self.site_of = {label: site for site, label in self.labels.items()}
        self.num_data = len(sites)
        self.data_qubits = tuple(range(self.num_data))
        self.z_order = tuple(z_order)
        self.x_order = tuple(x_order)

        z_sites = [(r, c) for r in range(self.size) for c in range(self.size)
                   if r % 2 == 0 and c % 2 == 1]
        x_sites = [(r, c) for r in range(self.size) for c in range(self.size)
                   if r % 2 == 1 and c % 2 == 0]
        self.stabilizer_sites = tuple(z_sites + x_sites)
        self.num_stabilizers = len(self.stabilizer_sites)
        self.z_stabilizers = tuple(range(len(z_sites)))
        self.x_stabilizers = tuple(range(len(z_sites), self.num_stabilizers))
        self.stabilizer_types = tuple(["Z"] * len(z_sites) + ["X"] * len(x_sites))
        self.stabilizer_ancilla = tuple(self.num_data + s
                                        for s in range(self.num_stabilizers))
        self.ancilla_qubits = self.stabilizer_ancilla
        self.num_qubits = self.num_data + self.num_stabilizers

        self.check_supports: tuple[tuple[int, ...], ...] = tuple(
            self._support(site, z_order if kind == "Z" else x_order)
            for site, kind in zip(self.stabilizer_sites, self.stabilizer_types))
        self.stabilizer_paulis = tuple(
            self._pauli_string(kind, support)
            for kind, support in zip(self.stabilizer_types,
                                     self.check_supports))
        self.stabilizer_names = tuple(
            f"S_{kind}{site[0]}{site[1]}"
            for kind, site in zip(self.stabilizer_types, self.stabilizer_sites))

        self.logical_x_support = tuple(sorted(
            self.labels[(0, c)] for c in range(0, self.size, 2)))
        self.logical_z_support = tuple(sorted(
            self.labels[(r, 0)] for r in range(0, self.size, 2)))
        self.logical_x = self._pauli_string("X", self.logical_x_support)
        self.logical_z = self._pauli_string("Z", self.logical_z_support)

        self.fold_pairs = tuple(sorted(
            tuple(sorted((self.labels[(r, c)], self.labels[(c, r)])))
            for (r, c) in sites if r < c))
        diagonal = [(i, i) for i in range(self.size)]
        self.diagonal = tuple(self.labels[site] for site in diagonal)
        # S on even diagonal positions, S-dagger on odd ones: every X check
        # touches two consecutive diagonal sites or none, so the phases
        # cancel and X checks map to (X check) * (mirror Z check) with a +
        # sign. S on the corner (0, 0) makes X_L -> +Y_L.
        self.diagonal_gates = tuple(
            [("S", self.labels[(i, i)]) for i in range(0, self.size, 2)]
            + [("S_DAG", self.labels[(i, i)]) for i in range(1, self.size, 2)])

        self._generator_matrix = np.array(
            [_symplectic(stim.PauliString(p)) for p in self.stabilizer_paulis],
            dtype=np.uint8)
        self._pivots, self._pivot_inverse = _gf2_left_solver(
            self._generator_matrix)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _support(self, site: Site, order: Sequence[str]) -> tuple[int, ...]:
        r, c = site
        out = []
        for step in order:
            dr, dc = _STEPS[step]
            neighbour = (r + dr, c + dc)
            if neighbour in self.labels:
                out.append(self.labels[neighbour])
        return tuple(out)

    def _pauli_string(self, kind: str, support: Sequence[int]) -> str:
        chars = ["_"] * self.num_data
        for q in support:
            chars[q] = kind
        return "".join(chars)

    def transpose(self, label: int) -> int:
        """Label of the mirror image of data qubit ``label``."""
        r, c = self.site_of[label]
        return self.labels[(c, r)]

    # ------------------------------------------------------------------
    # Paulis and tableaus
    # ------------------------------------------------------------------
    def stabilizer_pauli(self, index: int) -> stim.PauliString:
        return stim.PauliString(self.stabilizer_paulis[index])

    def logical_pauli(self, which: str) -> stim.PauliString:
        """``X_L``, ``Z_L`` or ``Y_L = i X_L Z_L`` as a Stim Pauli string."""
        x = stim.PauliString(self.logical_x)
        z = stim.PauliString(self.logical_z)
        if which == "X":
            return x
        if which == "Z":
            return z
        if which == "Y":
            return x * z * 1j
        raise ValueError(which)

    def data_tableau(self, operations: Sequence[Operation]) -> stim.Tableau:
        """Tableau of the data unitary of ideal data-only ``operations``."""
        circuit = stim.Circuit()
        circuit.append("I", [self.num_data - 1])
        for gate, targets in operations:
            if any(t >= self.num_data for t in targets):
                raise ValueError("data_tableau only accepts data operations.")
            circuit.append(gate, list(targets))
        return stim.Tableau.from_circuit(circuit)

    def stabilizer_coefficients(self, pauli: stim.PauliString
                                ) -> Optional[np.ndarray]:
        """GF(2) coefficients of ``pauli`` over the generators, or ``None``.

        Signs are ignored. ``None`` means ``pauli`` is not in the stabilizer
        group (up to sign).
        """
        target = _symplectic(pauli)
        coeffs = (target[self._pivots].astype(np.int64)
                  @ self._pivot_inverse) % 2
        check = (coeffs @ self._generator_matrix.astype(np.int64)) % 2
        if not np.array_equal(check.astype(np.uint8), target):
            return None
        return coeffs.astype(np.uint8)

    def decompose_in_stabilizers(self, pauli: stim.PauliString
                                 ) -> tuple[int, ...]:
        """Generator subset whose product equals ``pauli`` up to sign."""
        coeffs = self.stabilizer_coefficients(pauli)
        if coeffs is None:
            raise ValueError(f"{pauli} is not in the stabilizer group.")
        return tuple(int(s) for s in np.flatnonzero(coeffs))

    def stabilizer_product(self, subset: Sequence[int]) -> stim.PauliString:
        product = stim.PauliString(self.num_data)
        for s in subset:
            product *= self.stabilizer_pauli(s)
        return product

    def logical_class(self, pauli: stim.PauliString) -> Optional[tuple[str, int]]:
        """Return ``(P, sign)`` with ``pauli = sign * P_L * (stabilizer)``.

        ``P`` is one of ``"I", "X", "Y", "Z"``; ``None`` if ``pauli`` is not
        in the normalizer (or the sign is imaginary).
        """
        for name in ("I", "X", "Y", "Z"):
            logical = (stim.PauliString(self.num_data) if name == "I"
                       else self.logical_pauli(name))
            rest = pauli * logical  # = sign * stabilizer if the class matches
            coeffs = self.stabilizer_coefficients(rest)
            if coeffs is None:
                continue
            product = self.stabilizer_product(np.flatnonzero(coeffs))
            if rest == product:
                return name, +1
            if rest == -product:
                return name, -1
            return None
        return None

    # ------------------------------------------------------------------
    # Circuits
    # ------------------------------------------------------------------
    def check_operations(self, index: int) -> list[Operation]:
        """Entangling operations of one check (no measurement).

        Z checks: CNOT data -> ancilla. X checks: H on the ancilla, CNOT
        ancilla -> data, H on the ancilla (the ``lrb_qubit.code`` layout).
        """
        ancilla = self.stabilizer_ancilla[index]
        support = self.check_supports[index]
        if self.stabilizer_types[index] == "Z":
            return [("CX", (q, ancilla)) for q in support]
        return ([("H", (ancilla,))] + [("CX", (ancilla, q)) for q in support]
                + [("H", (ancilla,))])

    def encoding_operations(self) -> list[Operation]:
        """Ideal unitary preparation of ``|+>_L`` from ``|0...0>``.

        Built by Stim from the stabilizers plus ``X_L``; it is applied without
        noise, so only the prepared state matters (checked in the tests).
        """
        generators = [self.stabilizer_pauli(s)
                      for s in range(self.num_stabilizers)]
        generators.append(self.logical_pauli("X"))
        tableau = stim.Tableau.from_stabilizers(generators)
        circuit = tableau.to_circuit("elimination")
        operations: list[Operation] = []
        for instruction in circuit:
            name = instruction.name
            targets = [t.value for t in instruction.targets_copy()]
            arity = 2 if stim.gate_data(name).is_two_qubit_gate else 1
            for k in range(0, len(targets), arity):
                operations.append((name, tuple(targets[k:k + arity])))
        return operations

    def logical_gate_operations(self, token: str) -> list[Operation]:
        """Physical operations of one logical gate token (``H``, ``P``...).

        ``P`` is the logical S. ``*_INV`` tokens reverse and invert the list,
        as in ``lrb_qubit.code.logical_gate_operations``.
        """
        base = token[0]
        if base == "I":
            ops: list[Operation] = [("I", (0,))]
        elif base == "X":
            ops = [("X", (q,)) for q in self.logical_x_support]
        elif base == "Z":
            ops = [("Z", (q,)) for q in self.logical_z_support]
        elif base == "H":
            ops = ([("SWAP", pair) for pair in self.fold_pairs]
                   + [("H", (q,)) for q in self.data_qubits])
        elif base == "P":
            ops = ([(gate, (q,)) for gate, q in self.diagonal_gates]
                   + [("CZ", pair) for pair in self.fold_pairs])
        else:
            raise ValueError(f"Unsupported logical gate token {token!r}.")
        if len(token) > 1:
            if not token.endswith("_INV"):
                raise ValueError(f"Unsupported logical gate token {token!r}.")
            ops = [(_TOKEN_INVERSE_GATE[g], t) for g, t in reversed(ops)]
        return ops

    def logical_clifford_operations(self, descriptor: str,
                                    inverse: bool = False) -> list[Operation]:
        """Physical operations of one logical Clifford (or its inverse).

        Descriptors are those of ``lrb_qubit.code.SINGLE_QUBIT_CLIFFORDS``
        (rightmost symbol acts first).
        """
        tokens = (C.clifford_inverse_sequence(descriptor) if inverse
                  else C.clifford_gate_sequence(descriptor))
        ops: list[Operation] = []
        for token in tokens:
            ops.extend(self.logical_gate_operations(token))
        return ops

    def describe(self) -> dict:
        """Plain-data summary (for notes and result metadata)."""
        return {
            "distance": self.distance,
            "num_data": self.num_data,
            "num_stabilizers": self.num_stabilizers,
            "stabilizers": dict(zip(self.stabilizer_names,
                                    self.stabilizer_paulis)),
            "check_supports_in_cnot_order": [list(s) for s in
                                             self.check_supports],
            "logical_x": self.logical_x,
            "logical_z": self.logical_z,
            "fold_pairs": [list(p) for p in self.fold_pairs],
            "diagonal_gates": [list(g) for g in self.diagonal_gates],
            "z_order": "".join(self.z_order),
            "x_order": "".join(self.x_order),
            "sites": {str(label): list(site)
                      for label, site in sorted(self.site_of.items())},
        }


def _symplectic(pauli: stim.PauliString) -> np.ndarray:
    x, z = pauli.to_numpy()
    return np.concatenate([x, z]).astype(np.uint8)


def _gf2_left_solver(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pivot columns and inverse for solving ``c @ matrix = v`` over GF(2).

    ``matrix`` has independent rows. Returns columns ``P`` such that
    ``matrix[:, P]`` is invertible, and its GF(2) inverse ``M``; then
    ``c = v[P] @ M``.
    """
    rows, cols = matrix.shape
    work = matrix.copy() % 2
    pivots = []
    row = 0
    reduced = work.copy()
    for col in range(cols):
        if row == rows:
            break
        candidates = np.flatnonzero(reduced[row:, col]) + row
        if len(candidates) == 0:
            continue
        pivot = candidates[0]
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for other in range(rows):
            if other != row and reduced[other, col]:
                reduced[other] ^= reduced[row]
        pivots.append(col)
        row += 1
    if len(pivots) != rows:
        raise ValueError("Stabilizer generators are not independent.")
    sub = matrix[:, pivots].astype(np.uint8) % 2
    inverse = _gf2_inverse(sub)
    return np.asarray(pivots), inverse.astype(np.int64)


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    aug = np.concatenate([matrix % 2, np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        candidates = np.flatnonzero(aug[col:, col]) + col
        if len(candidates) == 0:
            raise ValueError("Matrix is singular over GF(2).")
        pivot = candidates[0]
        aug[[col, pivot]] = aug[[pivot, col]]
        for other in range(n):
            if other != col and aug[other, col]:
                aug[other] ^= aug[col]
    return aug[:, n:]


CODE13 = UnrotatedSurfaceCode(3)
CODE5 = UnrotatedSurfaceCode(2, labels=FIVE_QUBIT_LABELS)
