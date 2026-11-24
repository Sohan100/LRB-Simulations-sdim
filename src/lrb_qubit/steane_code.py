"""The Steane ``[[7,1,3]]`` code: transversal logical Cliffords and check circuits.

Work package K. The Steane code (Steane 1996, arXiv:quant-ph/9601029) is the
CSS code of the ``[7,4,3]`` Hamming code. Every single-qubit logical Clifford
is transversal, so the logical gates need no SWAP or CZ (unlike the
fold-transversal gates of the ``[[5,1,2]]`` and ``[[13,1,3]]`` surface codes).

Labels
------
Data qubits ``0..6``. Qubit ``q`` sits at column ``q + 1`` of the Hamming
parity-check matrix, so the three supports are the rows of that matrix::

    (3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6)

Stabilizer index ``s`` (used everywhere in this package): ``0, 1, 2`` are the
Z-type checks ``S_Z1, S_Z2, S_Z3`` and ``3, 4, 5`` the X-type checks
``S_X1, S_X2, S_X3`` on the same three supports (Z checks first, as in the
repo's check rounds). The ancilla of stabilizer ``s`` is wire ``7 + s``. With
flagged extraction the flag of stabilizer ``s`` is wire ``13 + s``.

Logical operators: ``X_L = X0 X1 X2`` and ``Z_L = Z0 Z1 Z2`` (weight 3;
``X^7`` and ``Z^7`` are the same logical operators times a stabilizer), and
``Y_L = i X_L Z_L = -Y0 Y1 Y2``.

Logical gates (checked with Stim tableaus in ``tests/test_qubit512_steane.py``):

* ``H_L`` = ``H`` on all seven data qubits;
* ``S_L`` = ``S_DAG`` on all seven data qubits. Transversal ``S`` maps
  ``X_L -> -Y_L``, i.e. it implements ``S_L^dagger``, because seven is
  ``3 mod 4``;
* ``X_L`` and ``Z_L``: ``X`` (``Z``) on the weight-3 support ``{0, 1, 2}``
  (default) or on all seven qubits (``pauli_gates="transversal"``);
* the identity Clifford is an ``I`` on data qubit 0, as in ``lrb_qubit.code``.

Check circuits
--------------
Z checks: ``CNOT data -> ancilla``. X checks: ``H`` on the ancilla,
``CNOT ancilla -> data``, ``H`` (the ``lrb_qubit.code`` layout).

``extraction="naive"``: one ancilla per weight-4 stabilizer. A single fault on
the ancilla between the second and third CNOT spreads to a weight-2 data error
(a hook). In the Steane code every weight-2 error equals a weight-1 error
times a logical operator, so the circuit-level distance drops to 2.

``extraction="flag"``: one flag qubit per stabilizer (Chao and Reichardt,
arXiv:1705.02329). The flag is coupled to the syndrome ancilla after the first
and before the last data CNOT::

    X check:  H a; CX a d1; CX a f; CX a d2; CX a d3; CX a f; CX a d4; H a;
              measure a and f (f in the Z basis)
    Z check:  H f; CX d1 a; CX f a; CX d2 a; CX d3 a; CX f a; CX d4 a; H f;
              measure a and f (f in the X basis)

Without faults the flag reads 0. A fault that spreads to a data error of
weight 2 (``d3 d4``) also flips the flag, so the decoder can tell it apart
from the weight-1 error with the same syndrome. The flagged data errors of one
check (``I, d4, d3 d4, d2 d3 d4 ~ d1``) have distinct syndromes.

An operation is ``(stim_gate_name, targets)``, as in ``lrb_qubit.code``.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import stim

from . import code as C

Operation = tuple[str, tuple[int, ...]]

NUM_DATA = 7
DATA_QUBITS = tuple(range(NUM_DATA))
SUPPORTS = ((3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6))
STABILIZER_NAMES = ("S_Z1", "S_Z2", "S_Z3", "S_X1", "S_X2", "S_X3")
STABILIZER_TYPES = ("Z", "Z", "Z", "X", "X", "X")
Z_STABILIZERS = (0, 1, 2)
X_STABILIZERS = (3, 4, 5)
NUM_STABILIZERS = 6
LOGICAL_SUPPORT = (0, 1, 2)
EXTRACTIONS = ("naive", "flag")
PAULI_GATES = ("weight3", "transversal")

_INVERSE_GATE = {"I": "I", "X": "X", "Z": "Z", "H": "H", "S": "S_DAG",
                 "S_DAG": "S"}


def _pauli_string(kind: str, support: Sequence[int],
                  n: int = NUM_DATA) -> str:
    chars = ["_"] * n
    for q in support:
        chars[q] = kind
    return "".join(chars)


STABILIZER_PAULIS = tuple(
    _pauli_string(kind, SUPPORTS[s % 3])
    for s, kind in enumerate(STABILIZER_TYPES))
LOGICAL_X = _pauli_string("X", LOGICAL_SUPPORT)
LOGICAL_Z = _pauli_string("Z", LOGICAL_SUPPORT)


def _symplectic(pauli: stim.PauliString) -> np.ndarray:
    x, z = pauli.to_numpy()
    return np.concatenate([x, z]).astype(np.uint8)


@lru_cache(maxsize=None)
def _stabilizer_table() -> dict[bytes, tuple[int, ...]]:
    """Symplectic vector of every stabilizer-group element -> generator subset."""
    table: dict[bytes, tuple[int, ...]] = {}
    vectors = [_symplectic(stim.PauliString(p)) for p in STABILIZER_PAULIS]
    for mask in range(2 ** NUM_STABILIZERS):
        subset = tuple(s for s in range(NUM_STABILIZERS) if (mask >> s) & 1)
        vec = np.zeros(2 * NUM_DATA, dtype=np.uint8)
        for s in subset:
            vec ^= vectors[s]
        table[vec.tobytes()] = subset
    if len(table) != 2 ** NUM_STABILIZERS:
        raise AssertionError("Stabilizer generators are not independent.")
    return table


class SteaneCode:
    """The ``[[7,1,3]]`` code with transversal Cliffords and a check layout.

    Args:
        extraction: ``"naive"`` (one ancilla per stabilizer) or ``"flag"``
            (one ancilla plus one flag per stabilizer).
        pauli_gates: ``"weight3"`` (logical X, Z on ``{0, 1, 2}``) or
            ``"transversal"`` (on all seven qubits).
        cnot_orders: Optional data-qubit order of the CNOTs of each of the
            six checks (each a permutation of its support). Default: the
            support order of ``SUPPORTS`` for every check.

    Attributes:
        num_data, num_stabilizers, num_qubits: Sizes.
        stabilizer_ancilla: Wire of the syndrome ancilla of each stabilizer.
        flag_qubits: Wire of the flag of each stabilizer (``None`` entries
            for naive extraction).
        ancilla_qubits: Every non-data wire (reset at the end of a round).
    """

    distance = 3

    def __init__(self, extraction: str = "naive",
                 pauli_gates: str = "weight3",
                 cnot_orders: Optional[Sequence[Sequence[int]]] = None):
        if extraction not in EXTRACTIONS:
            raise ValueError(f"extraction must be one of {EXTRACTIONS}.")
        if pauli_gates not in PAULI_GATES:
            raise ValueError(f"pauli_gates must be one of {PAULI_GATES}.")
        self.extraction = extraction
        self.pauli_gates = pauli_gates
        self.num_data = NUM_DATA
        self.data_qubits = DATA_QUBITS
        self.num_stabilizers = NUM_STABILIZERS
        self.stabilizer_names = STABILIZER_NAMES
        self.stabilizer_types = STABILIZER_TYPES
        self.stabilizer_paulis = STABILIZER_PAULIS
        self.z_stabilizers = Z_STABILIZERS
        self.x_stabilizers = X_STABILIZERS
        self.logical_x = LOGICAL_X
        self.logical_z = LOGICAL_Z
        self.logical_x_support = LOGICAL_SUPPORT
        self.logical_z_support = LOGICAL_SUPPORT
        self.stabilizer_ancilla = tuple(NUM_DATA + s
                                        for s in range(NUM_STABILIZERS))
        if extraction == "flag":
            self.flag_qubits: tuple[Optional[int], ...] = tuple(
                NUM_DATA + NUM_STABILIZERS + s
                for s in range(NUM_STABILIZERS))
        else:
            self.flag_qubits = tuple(None for _ in range(NUM_STABILIZERS))
        self.ancilla_qubits = tuple(self.stabilizer_ancilla) + tuple(
            q for q in self.flag_qubits if q is not None)
        self.num_qubits = NUM_DATA + len(self.ancilla_qubits)
        if cnot_orders is None:
            cnot_orders = [SUPPORTS[s % 3] for s in range(NUM_STABILIZERS)]
        if len(cnot_orders) != NUM_STABILIZERS:
            raise ValueError("Need one CNOT order per stabilizer.")
        for s, order in enumerate(cnot_orders):
            if sorted(order) != sorted(SUPPORTS[s % 3]):
                raise ValueError(f"CNOT order {order} of stabilizer {s} is "
                                 "not a permutation of its support.")
        self.cnot_orders = tuple(tuple(int(q) for q in o) for o in cnot_orders)

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
        circuit.append("I", [NUM_DATA - 1])
        for gate, targets in operations:
            if any(t >= NUM_DATA for t in targets):
                raise ValueError("data_tableau only accepts data operations.")
            circuit.append(gate, list(targets))
        return stim.Tableau.from_circuit(circuit)

    def decompose_in_stabilizers(self, pauli: stim.PauliString
                                 ) -> tuple[int, ...]:
        """Generator subset whose product equals ``pauli`` up to sign."""
        key = _symplectic(pauli).tobytes()
        table = _stabilizer_table()
        if key not in table:
            raise ValueError(f"{pauli} is not in the stabilizer group.")
        return table[key]

    def stabilizer_product(self, subset: Sequence[int]) -> stim.PauliString:
        product = stim.PauliString(NUM_DATA)
        for s in subset:
            product *= self.stabilizer_pauli(s)
        return product

    def logical_class(self, pauli: stim.PauliString
                      ) -> Optional[tuple[str, int]]:
        """``(P, sign)`` with ``pauli = sign * P_L * (stabilizer)``, or None.

        ``P`` is ``"I", "X", "Y"`` or ``"Z"``; ``None`` if ``pauli`` is not in
        the normalizer or the sign is not real.
        """
        for name in ("I", "X", "Y", "Z"):
            logical = (stim.PauliString(NUM_DATA) if name == "I"
                       else self.logical_pauli(name))
            rest = pauli * logical
            try:
                subset = self.decompose_in_stabilizers(rest)
            except ValueError:
                continue
            product = self.stabilizer_product(subset)
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
        """Entangling operations of one check (no measurement, no reset)."""
        a = self.stabilizer_ancilla[index]
        f = self.flag_qubits[index]
        d = self.cnot_orders[index]
        is_z = self.stabilizer_types[index] == "Z"
        if f is None:
            if is_z:
                return [("CX", (q, a)) for q in d]
            return ([("H", (a,))] + [("CX", (a, q)) for q in d]
                    + [("H", (a,))])
        if is_z:
            return [("H", (f,)), ("CX", (d[0], a)), ("CX", (f, a)),
                    ("CX", (d[1], a)), ("CX", (d[2], a)), ("CX", (f, a)),
                    ("CX", (d[3], a)), ("H", (f,))]
        return [("H", (a,)), ("CX", (a, d[0])), ("CX", (a, f)),
                ("CX", (a, d[1])), ("CX", (a, d[2])), ("CX", (a, f)),
                ("CX", (a, d[3])), ("H", (a,))]

    def measured_qubits(self, index: int) -> tuple[int, ...]:
        """Wires measured for check ``index``: ancilla, then flag (if any)."""
        f = self.flag_qubits[index]
        a = self.stabilizer_ancilla[index]
        return (a,) if f is None else (a, f)

    def encoding_operations(self) -> list[Operation]:
        """Ideal unitary preparation of ``|+>_L`` from ``|0...0>``.

        Built by Stim from the stabilizers plus ``X_L``; it is applied
        without noise, so only the prepared state matters (tested).
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
        x_support = (LOGICAL_SUPPORT if self.pauli_gates == "weight3"
                     else DATA_QUBITS)
        if base == "I":
            ops: list[Operation] = [("I", (0,))]
        elif base == "X":
            ops = [("X", (q,)) for q in x_support]
        elif base == "Z":
            ops = [("Z", (q,)) for q in x_support]
        elif base == "H":
            ops = [("H", (q,)) for q in DATA_QUBITS]
        elif base == "P":
            ops = [("S_DAG", (q,)) for q in DATA_QUBITS]
        else:
            raise ValueError(f"Unsupported logical gate token {token!r}.")
        if len(token) > 1:
            if not token.endswith("_INV"):
                raise ValueError(f"Unsupported logical gate token {token!r}.")
            ops = [(_INVERSE_GATE[g], t) for g, t in reversed(ops)]
        return ops

    def logical_clifford_operations(self, descriptor: str,
                                    inverse: bool = False) -> list[Operation]:
        """Physical operations of one logical Clifford (or its inverse).

        Descriptors are those of ``lrb_qubit.code.SINGLE_QUBIT_CLIFFORDS``
        (rightmost symbol acts first); the inverse is executed token by token
        as in ``lrb_qubit.code.clifford_inverse_sequence``.
        """
        tokens = (C.clifford_inverse_sequence(descriptor) if inverse
                  else C.clifford_gate_sequence(descriptor))
        ops: list[Operation] = []
        for token in tokens:
            ops.extend(self.logical_gate_operations(token))
        return ops

    def describe(self) -> dict:
        """Plain-data summary for notes and result metadata."""
        return {
            "code": "Steane [[7,1,3]]",
            "extraction": self.extraction,
            "pauli_gates": self.pauli_gates,
            "num_qubits": self.num_qubits,
            "stabilizers": dict(zip(self.stabilizer_names,
                                    self.stabilizer_paulis)),
            "cnot_orders": [list(o) for o in self.cnot_orders],
            "stabilizer_ancilla": list(self.stabilizer_ancilla),
            "flag_qubits": [q for q in self.flag_qubits],
            "logical_x": self.logical_x,
            "logical_z": self.logical_z,
            "logical_H": "H on all 7 data qubits",
            "logical_S": "S_DAG on all 7 data qubits",
        }


def weight_two_equivalences() -> dict[tuple[int, int], int]:
    """For each data pair ``(c, d)`` the qubit ``e`` with ``X_c X_d X_e = X_L``
    up to a stabilizer (Hamming columns ``c + 1 XOR d + 1 = e + 1``)."""
    out = {}
    for c, d in itertools.combinations(DATA_QUBITS, 2):
        out[(c, d)] = ((c + 1) ^ (d + 1)) - 1
    return out
