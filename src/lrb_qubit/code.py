"""The qubit ``[[5,1,2]]`` folded surface code and its logical Clifford gates.

This is the qubit counterpart of ``FoldedQutritDetectionCode`` in
``src/lrb/code_definitions.py``. Wire layout, stabilizer supports, logical
operators, logical-gate gate lists and the stabilizer-check circuits are copied
from the qutrit ``folded_qutrit`` profile, with every qutrit gate replaced by
its qubit analogue (``H_INV -> H``, ``P -> S``, ``P_INV -> S_DAG``,
``CNOT_INV -> CNOT``, ``CZ_INV -> CZ``, ``X_INV -> X``, ``Z_INV -> Z``).

Wires 0-4 are data qubits. Ancilla 5 measures ``Z0 Z1 Z3``, ancilla 6 measures
``Z1 Z2 Z4``, ancilla 7 measures ``X0 X1 X2`` and ancilla 8 measures
``X1 X3 X4``.

An operation is a tuple ``(gate, targets)`` where ``gate`` is a Stim gate name
and ``targets`` is a tuple of one or two qubit indices. Two-qubit tuples are
ordered ``(control, target)`` for ``CX``.
"""

from __future__ import annotations

from typing import Sequence

import stim

NUM_DATA = 5
NUM_QUBITS = 9
DATA_QUBITS = (0, 1, 2, 3, 4)
ANCILLA_QUBITS = (5, 6, 7, 8)

# Stabilizer order used everywhere in this package: index 0..3.
STABILIZER_NAMES = ("S_Z1", "S_Z2", "S_X1", "S_X2")
STABILIZER_PAULIS = ("ZZ_Z_", "_ZZ_Z", "XXX__", "_X_XX")
STABILIZER_ANCILLA = (5, 6, 7, 8)
Z_STABILIZERS = (0, 1)
X_STABILIZERS = (2, 3)

LOGICAL_X = "X__X_"
LOGICAL_Z = "Z_Z__"

Operation = tuple[str, tuple[int, ...]]

# Inverses of the gate tokens used in Clifford descriptors.
_TOKEN_INVERSE = {
    "H": "H_INV",
    "H_INV": "H",
    "P": "P_INV",
    "P_INV": "P",
    "X": "X_INV",
    "X_INV": "X",
    "Z": "Z_INV",
    "Z_INV": "Z",
    "I": "I",
}

# Six representatives of the single-qubit symplectic group (S3 acting on
# {X, Y, Z}) times the four Paulis give all 24 single-qubit Cliffords.
# Descriptor strings follow the qutrit convention: the rightmost symbol acts
# first, and "S" is stored as "P".
_SYMPLECTIC_WORDS = ("", "H", "S", "HS", "SH", "HSH")
_PAULI_WORDS = ("", "X", "Z", "XZ")
SINGLE_QUBIT_CLIFFORDS_WITH_S = tuple(
    (pauli + word) or "I" for pauli in _PAULI_WORDS for word in _SYMPLECTIC_WORDS
)
SINGLE_QUBIT_CLIFFORDS = tuple(
    c.replace("S", "P") for c in SINGLE_QUBIT_CLIFFORDS_WITH_S
)


def clifford_gate_sequence(descriptor: str) -> list[str]:
    """Return execution-order gate tokens for a Clifford descriptor.

    Mirrors ``QutritCliffordLibrary.qutrit_gate_seq_list``: descriptors are
    read right-to-left.
    """
    return list(reversed(list(descriptor)))


def clifford_inverse_sequence(descriptor: str) -> list[str]:
    """Return execution-order tokens of the inverse Clifford.

    Mirrors ``QutritCliffordLibrary.build_inverse_lookup``: the inverse of a
    descriptor ``g_1 ... g_k`` (``g_k`` acts first) is executed as
    ``g_1^-1, ..., g_k^-1``.
    """
    return [_TOKEN_INVERSE[g] for g in descriptor]


def logical_gate_operations(token: str) -> list[Operation]:
    """Physical operations implementing one logical gate token.

    Mirrors ``FoldedQutritDetectionCode.
    elementary_single_qudit_logical_operator_gate_sequence``. ``*_INV`` tokens
    reverse and invert the elementary sequence.
    """
    base = token[0]
    if base == "I":
        ops: list[Operation] = [("I", (0,))]
    elif base == "X":
        ops = [("X", (0,)), ("X", (3,))]
    elif base == "Z":
        ops = [("Z", (0,)), ("Z", (2,))]
    elif base == "H":
        ops = [("SWAP", (2, 3))] + [("H", (q,)) for q in (0, 2, 3, 4)] + [
            ("H", (1,))
        ]
    elif base == "P":
        ops = [("S", (0,)), ("S", (4,)), ("S_DAG", (1,)), ("CZ", (2, 3))]
    else:
        raise ValueError(f"Unsupported logical gate token {token!r}.")
    if len(token) > 1:
        if not token.endswith("_INV"):
            raise ValueError(f"Unsupported logical gate token {token!r}.")
        inverse = {"I": "I", "X": "X", "Z": "Z", "H": "H", "S": "S_DAG",
                   "S_DAG": "S", "SWAP": "SWAP", "CZ": "CZ"}
        # The qutrit code reverses the (gate, wires) blocks, where a block can
        # cover several wires; reversing per-wire operations is equivalent
        # because operations inside one block act on distinct wires.
        ops = [(inverse[g], t) for g, t in reversed(ops)]
    return ops


def physical_gate_operations(token: str) -> list[Operation]:
    """Physical RB: one gate token on wire 0."""
    mapping = {"I": "I", "H": "H", "H_INV": "H", "P": "S", "P_INV": "S_DAG",
               "X": "X", "X_INV": "X", "Z": "Z", "Z_INV": "Z"}
    return [(mapping[token], (0,))]


def z_stabilizer_check_operations() -> list[Operation]:
    """Z checks, CNOT data->ancilla variant (qutrit ``z_ALT`` block)."""
    return [
        ("CX", (0, 5)), ("CX", (1, 5)), ("CX", (3, 5)), ("M", (5,)),
        ("CX", (1, 6)), ("CX", (2, 6)), ("CX", (4, 6)), ("M", (6,)),
    ]


def x_stabilizer_check_operations() -> list[Operation]:
    """X checks with H-conjugated ancillas (qutrit X block)."""
    return [
        ("H", (7,)), ("CX", (7, 0)), ("CX", (7, 1)), ("CX", (7, 2)),
        ("H", (7,)), ("M", (7,)),
        ("H", (8,)), ("CX", (8, 3)), ("CX", (8, 1)), ("CX", (8, 4)),
        ("H", (8,)), ("M", (8,)),
    ]


def reset_operations() -> list[Operation]:
    """Reset all four ancillas (one reset layer)."""
    return [("R", (q,)) for q in ANCILLA_QUBITS]


def check_round_operations() -> list[Operation]:
    """One full stabilizer-check round followed by the ancilla reset."""
    return (z_stabilizer_check_operations() + x_stabilizer_check_operations()
            + reset_operations())


def encoding_operations() -> list[Operation]:
    """Ideal preparation of |+>_L (all stabilizers and X_L equal +1).

    The qutrit encoder is ``H_INV[2,3,4]; CNOT_INV 2->1; CNOT 1->0;
    CNOT 3->0; CNOT 4->2``. Its qubit translation is used directly; the
    resulting state is checked in the test suite.
    """
    return [("H", (2,)), ("H", (3,)), ("H", (4,)), ("CX", (2, 1)),
            ("CX", (1, 0)), ("CX", (3, 0)), ("CX", (4, 2))]


def stabilizer_pauli(index: int) -> stim.PauliString:
    """Five-qubit Pauli string of stabilizer ``index``."""
    return stim.PauliString(STABILIZER_PAULIS[index])


def data_tableau(operations: Sequence[Operation]) -> stim.Tableau:
    """Tableau of the data-qubit unitary implemented by ideal ``operations``.

    Only operations acting purely on data qubits are allowed.
    """
    circuit = stim.Circuit()
    circuit.append("I", [NUM_DATA - 1])
    for gate, targets in operations:
        if any(t >= NUM_DATA for t in targets):
            raise ValueError("data_tableau only accepts data-qubit operations.")
        circuit.append(gate, list(targets))
    return stim.Tableau.from_circuit(circuit)


def logical_clifford_operations(descriptor: str, inverse: bool = False
                                ) -> list[Operation]:
    """Physical operations for one logical Clifford (or its inverse)."""
    tokens = (clifford_inverse_sequence(descriptor) if inverse
              else clifford_gate_sequence(descriptor))
    ops: list[Operation] = []
    for token in tokens:
        ops.extend(logical_gate_operations(token))
    return ops


def decompose_in_stabilizers(pauli: stim.PauliString) -> tuple[int, ...]:
    """Return the generator subset whose product equals ``pauli`` up to sign.

    Raises ``ValueError`` if ``pauli`` is not in the stabilizer group.
    """
    target_x, target_z = pauli.to_numpy()
    for mask in range(16):
        prod = stim.PauliString(NUM_DATA)
        for s in range(4):
            if (mask >> s) & 1:
                prod *= stabilizer_pauli(s)
        x, z = prod.to_numpy()
        if (x == target_x).all() and (z == target_z).all():
            return tuple(s for s in range(4) if (mask >> s) & 1)
    raise ValueError(f"{pauli} is not in the stabilizer group.")
