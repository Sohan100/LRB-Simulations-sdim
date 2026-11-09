"""The sampled single-qutrit Clifford catalog is the full 216-element group.

Every check builds circuits from SDIM's own gates and reads their unitaries
through SDIM's Cirq export, so the catalog is tested against the gate
conventions the simulations actually use.
"""

from __future__ import annotations

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")
from sdim.circuit import Circuit
from sdim.circuit_io import circuit_to_cirq_circuit

from lrb.code_definitions import (
    QutritCliffordLibrary,
    single_qutrit_cliffords_exhaustive_strings,
)


DIMENSION = 3
GROUP_ORDER = DIMENSION**3 * (DIMENSION**2 - 1)


def unitary_of_gates(gates: list[str]) -> np.ndarray:
    """Unitary of single-qutrit SDIM gates applied in list order."""
    circuit = Circuit(1, DIMENSION)
    for gate in gates:
        circuit.add_gate(gate, 0)
    cirq_circuit = circuit_to_cirq_circuit(circuit)
    if not len(cirq_circuit):
        return np.eye(DIMENSION, dtype=complex)
    return cirq.unitary(cirq_circuit)


def phase_free_key(unitary: np.ndarray) -> tuple:
    """Hashable form of a unitary with its global phase removed."""
    flat = unitary.reshape(-1)
    pivot = flat[np.argmax(np.abs(flat) > 1e-9)]
    normalized = flat * (abs(pivot) / pivot)
    return tuple(np.round(normalized, 8))


def catalog_unitaries() -> dict[str, np.ndarray]:
    return {
        cliff: unitary_of_gates(
            QutritCliffordLibrary.qutrit_gate_seq_list(cliff))
        for cliff in single_qutrit_cliffords_exhaustive_strings
    }


def test_catalog_has_216_distinct_cliffords():
    unitaries = catalog_unitaries()
    assert len(single_qutrit_cliffords_exhaustive_strings) == GROUP_ORDER
    assert len(set(single_qutrit_cliffords_exhaustive_strings)) == GROUP_ORDER
    keys = {phase_free_key(u) for u in unitaries.values()}
    assert len(keys) == GROUP_ORDER


def test_catalog_equals_generated_group():
    generators = [unitary_of_gates([gate]) for gate in ("H", "P")]
    identity = np.eye(DIMENSION, dtype=complex)
    group = {phase_free_key(identity)}
    frontier = [identity]
    while frontier:
        next_frontier = []
        for element in frontier:
            for generator in generators:
                product = generator @ element
                key = phase_free_key(product)
                if key not in group:
                    group.add(key)
                    next_frontier.append(product)
        frontier = next_frontier
    assert len(group) == GROUP_ORDER
    catalog = {phase_free_key(u) for u in catalog_unitaries().values()}
    assert catalog == group


def test_inverse_lookup_undoes_every_clifford():
    inverse_lookup = QutritCliffordLibrary.build_inverse_lookup()
    identity_key = phase_free_key(np.eye(DIMENSION, dtype=complex))
    for cliff in single_qutrit_cliffords_exhaustive_strings:
        gates = (QutritCliffordLibrary.qutrit_gate_seq_list(cliff)
                 + inverse_lookup[cliff])
        assert phase_free_key(unitary_of_gates(gates)) == identity_key, cliff
