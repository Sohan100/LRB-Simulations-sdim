"""The conventional-site two-ancilla check round measures the four stabilizers.

Noiseless SDIM runs of the profile's own check round (reset block plus
stabilizer blocks, as the generator builds it) on the qutrit and qubit codes.
"""

from __future__ import annotations

import itertools

import pytest
from sdim.program import Program

from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_definitions import FoldedQutritConventionalSitesDetectionCode
from lrb.code_simulation_profiles import (
    CONVENTIONAL_SITES_FOLDED_CODE_NAME,
    CONVENTIONAL_SITES_FOLDED_QUBIT_CODE_NAME,
    CodeSimulationProfileRegistry,
)

NAMES = (CONVENTIONAL_SITES_FOLDED_CODE_NAME,
         CONVENTIONAL_SITES_FOLDED_QUBIT_CODE_NAME)
SITES = {5: {0, 1, 2}, 6: {0, 2, 3}, 7: {1, 2, 4}, 8: {2, 3, 4}}
SHOTS = 200


def check_round(code_name):
    definition = CodeSimulationProfileRegistry.resolve_code_profile(
        code_name).code_definition
    generator = LRBCircuitGenerator(code_definition=definition,
                                    noise_model="si1000")
    return definition, generator._check_round_circuit()


def run(code_name, rounds=3, sandwich=False, error=None):
    """Return (readouts per shot, terminal X data per shot)."""
    definition, round_circuit = check_round(code_name)
    d = definition.dimension
    c = definition.logical_plus_initial_state()
    if error is not None:
        wire, gate, power = error
        for _ in range(power):
            c.add_gate(gate, wire)
    if sandwich:
        c = c + definition.logical_gate_circuit("H")
    for _ in range(rounds):
        c = c + round_circuit
    if sandwich:
        c = c + definition.logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H_INV", q)
        c.add_gate("M", q)
    counts, reads = {}, []
    for _ in range(rounds):
        for op in round_circuit.operations:
            if op.gate_name in ("M", "RESET"):
                w = int(op.qudit_index)
                if op.gate_name == "M":
                    reads.append((w, counts.get(w, 0)))
                counts[w] = counts.get(w, 0) + 1
    final = [(q, counts.get(q, 0)) for q in range(5)]
    m = Program(c).simulate(shots=SHOTS)[0]
    value = lambda w, k, s: int(m[w][k][s].measurement_value) % d
    return ([[value(w, k, s) for w, k in reads] for s in range(SHOTS)],
            [[value(w, k, s) for w, k in final] for s in range(SHOTS)], d)


@pytest.mark.parametrize("code_name", NAMES)
@pytest.mark.parametrize("sandwich", (False, True))
def test_noiseless_rounds_pass_and_keep_the_logical(code_name, sandwich):
    reads, x, d = run(code_name, sandwich=sandwich)
    assert all(v == 0 for shot in reads for v in shot)
    for a in x:
        assert (a[0] + a[1] - a[2]) % d == 0
        assert (-a[1] + a[3] - a[4]) % d == 0
        assert (a[0] + a[3]) % d == 0


@pytest.mark.parametrize("code_name", NAMES)
def test_every_weight_one_data_error_is_flagged(code_name):
    d = CodeSimulationProfileRegistry.resolve_code_profile(
        code_name).code_definition.dimension
    for wire, gate, power in itertools.product(range(5), ("X", "Z"), range(1, d)):
        reads, _, _ = run(code_name, rounds=1, error=(wire, gate, power))
        assert all(any(shot) for shot in reads), (wire, gate, power)


@pytest.mark.parametrize("code_name", NAMES)
def test_gates_only_couple_a_site_to_its_own_data(code_name):
    _, round_circuit = check_round(code_name)
    for op in round_circuit.operations:
        wires = _qudits(op)
        if len(wires) == 2:
            site = [w for w in wires if w in SITES]
            data = [w for w in wires if w not in SITES]
            assert len(site) == 1 and len(data) == 1, wires
            assert data[0] in SITES[site[0]], wires


def _qudits(op):
    generator = LRBCircuitGenerator(noise_model="si1000")
    return tuple(generator._operation_qudits(op))


def test_round_schedule_depth():
    _, round_circuit = check_round(CONVENTIONAL_SITES_FOLDED_CODE_NAME)
    moments = LRBCircuitGenerator(noise_model="si1000").schedule_moments(
        round_circuit)
    gate_moments = sum(1 for collapse, _ in moments if not collapse)
    assert gate_moments == 11
    assert len(moments) - gate_moments == 2
    assert FoldedQutritConventionalSitesDetectionCode.num_qudits == 9
