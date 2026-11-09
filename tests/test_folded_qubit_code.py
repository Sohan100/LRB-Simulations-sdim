"""The qubit [[5,1,2]] folded code implements the intended logical Cliffords.

Unitaries come from SDIM's own dimension-2 gates through its Cirq export, as
in test_qutrit_clifford_group.py, so the checks use the gate conventions the
simulations run with. Detection checks run the noiseless SDIM simulator on
the stabilizer-check blocks of each qubit profile.
"""

from __future__ import annotations

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")
from sdim.circuit import Circuit
from sdim.circuit_io import circuit_to_cirq_circuit
from sdim.program import Program

from lrb.code_definitions import (
    FoldedQubitDetectionCode,
    FoldedQubitSplitAncillaUniformDetectionCode,
    QubitCliffordLibrary,
    single_qubit_cliffords_exhaustive_strings,
)
from lrb.code_simulation_profiles import (
    FOLDED_QUBIT_CODE_NAME,
    SPLIT_UNIFORM_FOLDED_QUBIT_CODE_NAME,
    CodeSimulationProfileRegistry,
)

GROUP_ORDER = 24
CODES = (FoldedQubitDetectionCode, FoldedQubitSplitAncillaUniformDetectionCode)
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
STABILIZERS = (
    {0: "Z", 1: "Z", 3: "Z"},
    {1: "Z", 2: "Z", 4: "Z"},
    {0: "X", 1: "X", 2: "X"},
    {1: "X", 3: "X", 4: "X"},
)
LOGICAL_X = {0: "X", 3: "X"}
LOGICAL_Z = {0: "Z", 2: "Z"}


def unitary(circuit: Circuit) -> np.ndarray:
    """Unitary of an SDIM circuit on all of its wires, wire 0 most significant."""
    qubits = [cirq.LineQid(i, dimension=circuit.dimension)
              for i in range(circuit.num_qudits)]
    cirq_circuit = circuit_to_cirq_circuit(circuit)
    if not len(cirq_circuit):
        return np.eye(circuit.dimension**circuit.num_qudits, dtype=complex)
    return cirq_circuit.unitary(qubit_order=qubits)


def pauli_operator(terms: dict[int, str], num_qudits: int) -> np.ndarray:
    operator = np.array([[1.0 + 0j]])
    for wire in range(num_qudits):
        operator = np.kron(operator, PAULI[terms.get(wire, "I")])
    return operator


def phase_free(matrix: np.ndarray) -> np.ndarray:
    flat = matrix.reshape(-1)
    pivot = flat[np.argmax(np.abs(flat) > 1e-9)]
    return matrix * (abs(pivot) / pivot)


def physical_clifford(descriptor: str) -> np.ndarray:
    circuit = Circuit(1, 2)
    for gate in QubitCliffordLibrary.qubit_gate_seq_list(descriptor):
        circuit.add_gate(gate, 0)
    return unitary(circuit)


def logical_basis(code) -> tuple[np.ndarray, np.ndarray]:
    """|0_L> and |1_L> from the code's own |+_L> preparation."""
    plus = unitary(code.logical_plus_initial_state())[:, 0]
    minus = pauli_operator(LOGICAL_Z, code.num_qudits) @ plus
    return (plus + minus) / np.sqrt(2), (plus - minus) / np.sqrt(2)


def logical_matrix(code, tokens: list[str]) -> np.ndarray:
    """2x2 action of logical gate tokens (in execution order) on the code."""
    circuit = Circuit(code.num_qudits, code.dimension)
    for token in tokens:
        circuit = circuit + code.single_qudit_logical_gate_circuit(token)
    full = unitary(circuit)
    basis = np.stack(logical_basis(code), axis=1)
    return basis.conj().T @ full @ basis


@pytest.mark.parametrize("code", CODES)
def test_plus_state_is_in_the_codespace(code):
    plus = unitary(code.logical_plus_initial_state())[:, 0]
    for stabilizer in STABILIZERS:
        assert np.allclose(
            pauli_operator(stabilizer, code.num_qudits) @ plus, plus)
    assert np.allclose(
        pauli_operator(LOGICAL_X, code.num_qudits) @ plus, plus)


def test_catalog_has_24_distinct_cliffords_with_inverses():
    assert len(set(single_qubit_cliffords_exhaustive_strings)) == GROUP_ORDER
    keys = {
        tuple(np.round(phase_free(physical_clifford(c)).reshape(-1), 8))
        for c in single_qubit_cliffords_exhaustive_strings
    }
    assert len(keys) == GROUP_ORDER
    inverses = QubitCliffordLibrary.build_inverse_lookup()
    for cliff in single_qubit_cliffords_exhaustive_strings:
        circuit = Circuit(1, 2)
        for gate in (QubitCliffordLibrary.qubit_gate_seq_list(cliff)
                     + inverses[cliff]):
            circuit.add_gate(gate, 0)
        assert np.allclose(phase_free(unitary(circuit)), np.eye(2))


@pytest.mark.parametrize("code", CODES)
def test_every_encoded_clifford_and_inverse_acts_as_intended(code):
    inverses = QubitCliffordLibrary.build_inverse_lookup()
    for cliff in single_qubit_cliffords_exhaustive_strings:
        target = physical_clifford(cliff)
        encoded = logical_matrix(
            code, QubitCliffordLibrary.qubit_gate_seq_list(cliff))
        # Unitary on the code space means the gates preserve it.
        assert np.allclose(encoded.conj().T @ encoded, np.eye(2)), cliff
        assert np.allclose(phase_free(encoded), phase_free(target)), cliff
        encoded_inverse = logical_matrix(code, inverses[cliff])
        assert np.allclose(
            phase_free(encoded_inverse), phase_free(target.conj().T)), cliff


@pytest.mark.parametrize("code_name", (
    FOLDED_QUBIT_CODE_NAME, SPLIT_UNIFORM_FOLDED_QUBIT_CODE_NAME))
def test_profiles_are_dimension_two(code_name):
    profile = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
    assert profile.code_definition.dimension == 2
    assert profile.logical_dimension == 2
    assert len(profile.code_definition.clifford_strings) == GROUP_ORDER


def check_round_passes(code_name: str, error: tuple[int, str] | None) -> bool:
    """Run one noiseless check round on |+_L> with an optional data error."""
    profile = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
    definition = profile.code_definition
    spec = CodeSimulationProfileRegistry._UNPACK_SPECS[code_name]
    circuit = definition.logical_plus_initial_state()
    if error is not None:
        wire, pauli = error
        for gate in {"X": ["X"], "Z": ["Z"], "Y": ["X", "Z"]}[pauli]:
            circuit.add_gate(gate, wire)
    for factory, _ in definition.stabilizer_check_blocks:
        circuit = circuit + factory()
    measurements = Program(circuit).simulate(shots=4)
    if isinstance(measurements, tuple):
        measurements = measurements[0]
    passes = set()
    for shot in range(4):
        values = [measurements[wire][0][shot].measurement_value
                  for wire in spec.stabilizer_wires]
        passes.add(bool(spec.stabilizer_pass_fn(values)))
    assert len(passes) == 1, "noiseless syndrome must be deterministic"
    return passes.pop()


@pytest.mark.parametrize("code_name", (
    FOLDED_QUBIT_CODE_NAME, SPLIT_UNIFORM_FOLDED_QUBIT_CODE_NAME))
def test_checks_pass_on_codewords_and_flag_every_single_data_error(code_name):
    assert check_round_passes(code_name, None)
    for wire in range(5):
        for pauli in ("X", "Y", "Z"):
            assert not check_round_passes(code_name, (wire, pauli)), (
                wire, pauli)
