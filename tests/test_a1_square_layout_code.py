"""Layout A1 (square lattice, two ancillas, paper encoder native) is correct.

Noiseless SDIM runs of the profile's own check round (reset block plus the A1
round, as the generator builds it) on the qutrit and qubit codes; coupler
checks for every two-qudit gate the experiment uses; the detector path and the
raw-record path of the unpacker agree shot by shot on noisy generated circuits.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest
from sdim.program import Program

from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_definitions import FoldedQutritA1SquareDetectionCode
from lrb.code_simulation_profiles import (
    A1_SQUARE_FOLDED_CODE_NAME,
    A1_SQUARE_FOLDED_QUBIT_CODE_NAME,
    CodeSimulationProfileRegistry,
)
from lrb.lrb_simulation import LRBSimulationPipeline

NAMES = (A1_SQUARE_FOLDED_CODE_NAME, A1_SQUARE_FOLDED_QUBIT_CODE_NAME)
E, N = 5, 6
POS = {0: (0, 0), 1: (1, 0), E: (2, 0), 3: (0, 1), 2: (1, 1), 4: (2, 1),
       N: (1, 2)}
COUPLERS = {frozenset((a, b)) for a in POS for b in POS if a < b
            and abs(POS[a][0] - POS[b][0]) + abs(POS[a][1] - POS[b][1]) == 1}
CHECK_DATA_COUPLERS = {frozenset(p) for p in ((0, 1), (1, 2), (2, 3))}
SHOTS = 200


def check_round(code_name):
    definition = CodeSimulationProfileRegistry.resolve_code_profile(
        code_name).code_definition
    generator = LRBCircuitGenerator(code_definition=definition,
                                    noise_model="si1000")
    return definition, generator._check_round_circuit()


def _qudits(op):
    return tuple(LRBCircuitGenerator._operation_qudits(op))


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
def test_each_readout_is_one_stabilizer(code_name):
    """A weight-1 data Pauli shows up exactly in the stabilizers it
    anticommutes with. Readout order is N=G1, E=G2, N, E; the ancillas are not
    reset between their two readouts, so the second readout of an ancilla is
    its X-type stabilizer plus its first readout, and the stabilizer is the
    difference. Acceptance (every readout zero) is the same either way."""
    d = CodeSimulationProfileRegistry.resolve_code_profile(
        code_name).code_definition.dimension
    z_checks = {"G1": (1, -1, 0, -1, 0), "G2": (0, 1, 1, 0, -1)}
    x_checks = {"G3": (1, 1, -1, 0, 0), "G4": (0, -1, 0, 1, -1)}
    for wire, gate, power in itertools.product(range(5), ("X", "Z"),
                                               range(1, d)):
        reads, _, _ = run(code_name, rounds=1, error=(wire, gate, power))
        if gate == "X":   # shifts Z-type checks by +-power * coefficient
            expect = [z_checks["G1"][wire] * power, z_checks["G2"][wire] * power,
                      0, 0]
        else:             # Z error: X-type checks, sign fixed per check
            expect = [0, 0, x_checks["G3"][wire] * power,
                      x_checks["G4"][wire] * power]
        for shot in reads:
            got = [shot[0] % d, shot[1] % d, (shot[2] - shot[0]) % d,
                   (shot[3] - shot[1]) % d]
            assert [bool(v) for v in got] == [bool(e % d) for e in expect], (
                wire, gate, power, got)
            for v, e in zip(got, expect):
                assert v in (e % d, -e % d)


@pytest.mark.parametrize("code_name", NAMES)
def test_every_two_qudit_gate_is_nearest_neighbour(code_name):
    definition, round_circuit = check_round(code_name)
    pieces = {"encoder": definition.logical_plus_initial_state(),
              "round": round_circuit,
              "const0 readout": definition.terminal_const0_measurement(),
              "readout": definition.terminal_measurement()}
    for token in ("H", "H_INV", "P", "P_INV", "X", "Z"):
        pieces[token] = definition.logical_gate_circuit(token)
    for name, circuit in pieces.items():
        for op in circuit.operations:
            wires = _qudits(op)
            assert len(wires) <= 2, (name, wires)
            if len(wires) == 2:
                assert frozenset(wires) in COUPLERS, (name, wires)
                if name == "round" and all(w < 5 for w in wires):
                    assert frozenset(wires) in CHECK_DATA_COUPLERS, wires


def test_round_schedule_and_size():
    _, round_circuit = check_round(A1_SQUARE_FOLDED_CODE_NAME)
    moments = LRBCircuitGenerator(noise_model="si1000").schedule_moments(
        round_circuit)
    gate_moments = sum(1 for collapse, _ in moments if not collapse)
    assert gate_moments == 22
    assert len(moments) - gate_moments == 3    # one reset, two readouts
    two = sum(1 for op in round_circuit.operations if len(_qudits(op)) == 2)
    assert two == 25
    assert FoldedQutritA1SquareDetectionCode.num_qudits == 7


@pytest.mark.parametrize("code_name", NAMES)
def test_detector_and_raw_unpack_agree(code_name, monkeypatch):
    random.seed(1117)
    profile = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
    generator = LRBCircuitGenerator(code_definition=profile.code_definition,
                                    noise_model="si1000")
    depth, shots = 3, 4000
    cliffords = generator.generate_random_clifford_strings(depth)
    circuit = generator.generate_lrb_clifford_sequence(
        [depth], clifford_strings=cliffords)[0]
    generator.update_noise_param(circuit, 0.01)
    output = LRBSimulationPipeline.simulate_circuit(
        circuit, shots=shots, simulation_backend="sdim")
    det_checks, det_logical = profile.unpack_func(output, depth, shots)
    monkeypatch.setattr(LRBSimulationPipeline,
                        "try_unpack_detector_results_from_spec",
                        staticmethod(lambda *args, **kwargs: None))
    raw_checks, raw_logical = profile.unpack_func(output, depth, shots)
    det_checks = np.asarray(det_checks)
    raw_checks = np.asarray(raw_checks)
    assert det_checks.shape == raw_checks.shape == (shots, depth + 1)
    assert np.array_equal(det_checks, raw_checks)
    assert np.array_equal(np.asarray(det_logical) % profile.code_definition.dimension,
                          np.asarray(raw_logical) % profile.code_definition.dimension)
    # The noise does something, and some shots still pass every round.
    assert 0 < det_checks.all(axis=1).sum() < shots
