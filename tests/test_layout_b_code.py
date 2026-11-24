"""Square-lattice layout B: encoder, check round and nearest-neighbour couplers.

Noiseless SDIM runs of the profile's own encoder and check round (reset block
plus round, as the generator builds it) on the qutrit and qubit codes, a check
that every two-qudit gate sits on a layout-B coupler, the ASAP schedule, and
agreement of the detector-backed and raw unpack paths for the repeated
per-round readouts of each ancilla.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sdim.program import Program

from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_definitions import FoldedQutritLayoutBDetectionCode
from lrb.code_simulation_profiles import (
    LAYOUT_B_FOLDED_CODE_NAME,
    LAYOUT_B_FOLDED_QUBIT_CODE_NAME,
    CodeSimulationProfileRegistry,
)
from lrb.lrb_simulation import LRBSimulationPipeline, NoiseModelUtils

NAMES = (LAYOUT_B_FOLDED_CODE_NAME, LAYOUT_B_FOLDED_QUBIT_CODE_NAME)
POS = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (0, 2), 5: (1, 2), 6: (2, 1)}
COUPLERS = {frozenset((a, b)) for a in POS for b in POS
            if a < b and abs(POS[a][0] - POS[b][0]) + abs(POS[a][1] - POS[b][1]) == 1}
SHOTS = 200


def _generator():
    return LRBCircuitGenerator(noise_model="si1000")


def _qudits(op):
    return tuple(_generator()._operation_qudits(op))


def check_round(code_name):
    definition = CodeSimulationProfileRegistry.resolve_code_profile(
        code_name).code_definition
    generator = LRBCircuitGenerator(code_definition=definition,
                                    noise_model="si1000")
    return definition, generator._check_round_circuit()


def run(code_name, rounds=3, sandwich=False, error=None):
    """Return (readouts per shot, terminal X data per shot, dimension)."""
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
def test_every_two_qudit_gate_is_on_a_layout_b_coupler(code_name):
    definition, round_circuit = check_round(code_name)
    pieces = [definition.logical_plus_initial_state(), round_circuit]
    pieces += [definition.logical_gate_circuit(t)
               for t in ("H", "H_INV", "P", "P_INV", "X", "Z")]
    pieces.append(definition.terminal_measurement())
    for piece in pieces:
        for op in piece.operations:
            wires = _qudits(op)
            assert len(wires) <= 2, wires
            if len(wires) == 2:
                assert frozenset(wires) in COUPLERS, wires


def test_encoder_uses_four_two_qutrit_gates():
    enc = FoldedQutritLayoutBDetectionCode.logical_plus_initial_state()
    assert sum(1 for op in enc.operations if len(_qudits(op)) == 2) == 4


def test_round_schedule_depth():
    _, round_circuit = check_round(LAYOUT_B_FOLDED_CODE_NAME)
    moments = _generator().schedule_moments(round_circuit)
    gate_moments = sum(1 for collapse, _ in moments if not collapse)
    assert gate_moments == 19
    assert len(moments) - gate_moments == 3   # reset, two readouts
    assert FoldedQutritLayoutBDetectionCode.num_qudits == 7


@pytest.mark.parametrize("code_name", NAMES)
def test_detector_and_raw_unpack_agree(code_name):
    """Repeated per-round readouts: detector labels and raw offsets pick the same values."""
    profile = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
    generator = LRBCircuitGenerator(code_definition=profile.code_definition,
                                    noise_model="si1000")
    depth = 2
    circuits = generator.generate_lrb_clifford_sequence([depth], with_noise=True)
    circuit = circuits[-1]
    generator.update_noise_param(circuit, 0.01)
    NoiseModelUtils.ensure_noise_params(circuit)
    spec = CodeSimulationProfileRegistry._UNPACK_SPECS[code_name]
    out = LRBSimulationPipeline.simulate_circuit(circuit, 400, simulation_backend="sdim")
    det = LRBSimulationPipeline.try_unpack_detector_results_from_spec(out, depth, 400, spec)
    assert det is not None
    raw = LRBSimulationPipeline.measurement_results_from_sdim_output(out)
    dims = profile.code_definition.dimension
    slots_ok = []
    for r in range(depth + spec.check_rounds_offset):
        vals = [np.array([int(raw[w][r * spec.check_stride + off][s].measurement_value)
                          for s in range(400)])
                for w, off in zip(spec.stabilizer_wires, spec.stabilizer_raw_offsets)]
        slots_ok.append(np.all(np.stack(vals) % dims == 0, axis=0).astype(int))
    raw_pass = np.stack(slots_ok, axis=1)
    assert np.array_equal(np.asarray(det[0]), raw_pass)
