"""One-ancilla round in the gate order PROD5B (profile ``folded_qutrit_prod5b``).

Checks that the profile differs from ``folded_qutrit`` only in the order of
the check-round gates: the same gates, encoder, logical gates and readout
layout; the ASAP schedule has 5 gate moments (production: 7); noiseless rounds
pass and keep the logical; every single data Pauli gives exactly the
production round's readouts; and the detector (DEM) and raw unpack paths agree.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from sdim.program import Program

from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_simulation_profiles import (
    DEFAULT_CODE_NAME,
    PROD5B_FOLDED_CODE_NAME,
    CodeSimulationProfileRegistry as Reg,
)
from lrb.lrb_simulation import LRBSimulationPipeline, NoiseModelUtils

SHOTS = 200
# rounds.py PROD5B (hw-layouts/one-ancilla), without the resets
PROD5B_ORDER = [
    ("CNOT_INV", 1, 5), ("H", 7), ("H", 8), ("CNOT", 0, 5), ("CNOT_INV", 7, 2),
    ("CNOT_INV", 8, 1), ("CNOT_INV", 3, 5), ("CNOT", 2, 6), ("CNOT", 7, 0),
    ("CNOT_INV", 8, 4), ("CNOT_INV", 4, 6), ("CNOT", 7, 1), ("CNOT", 8, 3),
    ("CNOT", 1, 6), ("H_INV", 7), ("H_INV", 8), ("M", 5), ("M", 6), ("M", 7), ("M", 8),
]


def generator(code_name, noise="si1000"):
    return LRBCircuitGenerator(
        code_definition=Reg.resolve_code_profile(code_name).code_definition,
        noise_model=noise)


def ops(circuit):
    g = generator(DEFAULT_CODE_NAME)
    return [(op.gate_name, *g._operation_qudits(op)) for op in circuit.operations]


def wire_orders(op_list):
    out = {}
    for op in op_list:
        for w in op[1:]:
            out.setdefault(w, []).append(op)
    return out


def test_round_is_prod5b_and_uses_the_production_gates():
    prod = ops(generator(DEFAULT_CODE_NAME)._check_round_circuit())
    new = ops(generator(PROD5B_FOLDED_CODE_NAME)._check_round_circuit())
    assert new[:4] == [("RESET", w) for w in (5, 6, 7, 8)]
    assert new[4:] == PROD5B_ORDER
    assert Counter(new) == Counter(prod)


def test_schedule_depths():
    for name, gate_moments in ((DEFAULT_CODE_NAME, 7), (PROD5B_FOLDED_CODE_NAME, 5)):
        g = generator(name)
        moments = g.schedule_moments(g._check_round_circuit())
        assert sum(1 for c, _ in moments if not c) == gate_moments
        assert sum(1 for c, _ in moments if c) == 2      # reset, readout


def test_everything_but_the_round_matches_folded_qutrit():
    a = Reg.resolve_code_profile(DEFAULT_CODE_NAME)
    b = Reg.resolve_code_profile(PROD5B_FOLDED_CODE_NAME)
    da, db = a.code_definition, b.code_definition
    assert ops(da.logical_plus_initial_state()) == ops(db.logical_plus_initial_state())
    for t in ("H", "H_INV", "P", "P_INV", "X", "Z"):
        assert ops(da.logical_gate_circuit(t)) == ops(db.logical_gate_circuit(t))
    assert ops(da.terminal_measurement()) == ops(db.terminal_measurement())
    assert ops(da.terminal_const0_measurement()) == ops(db.terminal_const0_measurement())
    assert da.stabilizer_detector_wires == db.stabilizer_detector_wires
    sa, sb = Reg._UNPACK_SPECS[DEFAULT_CODE_NAME], Reg._UNPACK_SPECS[PROD5B_FOLDED_CODE_NAME]
    assert replace(sa, logical_outcome_fn=None) == replace(sb, logical_outcome_fn=None)
    for m in itertools.product(range(3), repeat=2):
        assert sa.logical_outcome_fn(list(m), 4) == sb.logical_outcome_fn(list(m), 4)
    assert a.logical_dimension == b.logical_dimension == 3


def readouts(code_name, rounds=3, sandwich=False, error=None):
    definition = Reg.resolve_code_profile(code_name).code_definition
    round_circuit = generator(code_name)._check_round_circuit()
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
    value = lambda w, k, s: int(m[w][k][s].measurement_value) % 3
    return ([tuple(value(w, k, s) for w, k in reads) for s in range(SHOTS)],
            [[value(w, k, s) for w, k in final] for s in range(SHOTS)])


@pytest.mark.parametrize("sandwich", (False, True))
def test_noiseless_rounds_pass_and_keep_the_logical(sandwich):
    reads, x = readouts(PROD5B_FOLDED_CODE_NAME, sandwich=sandwich)
    assert all(v == 0 for shot in reads for v in shot)
    for a in x:
        assert (a[0] + a[1] - a[2]) % 3 == 0
        assert (-a[1] + a[3] - a[4]) % 3 == 0
        assert (a[0] + a[3]) % 3 == 0


def test_single_data_paulis_give_the_production_syndromes():
    for wire, gate, power in itertools.product(range(5), ("X", "Z"), (1, 2)):
        new, _ = readouts(PROD5B_FOLDED_CODE_NAME, rounds=1, error=(wire, gate, power))
        old, _ = readouts(DEFAULT_CODE_NAME, rounds=1, error=(wire, gate, power))
        assert len(set(new)) == 1 and set(new) == set(old), (wire, gate, power)
        assert any(new[0]), (wire, gate, power)


def test_detector_and_raw_unpack_agree():
    profile = Reg.resolve_code_profile(PROD5B_FOLDED_CODE_NAME)
    g = generator(PROD5B_FOLDED_CODE_NAME)
    depth = 2
    circuit = g.generate_lrb_clifford_sequence([depth], with_noise=True)[-1]
    g.update_noise_param(circuit, 0.01)
    NoiseModelUtils.ensure_noise_params(circuit)
    spec = Reg._UNPACK_SPECS[PROD5B_FOLDED_CODE_NAME]
    out = LRBSimulationPipeline.simulate_circuit(circuit, 400, simulation_backend="sdim")
    det = LRBSimulationPipeline.try_unpack_detector_results_from_spec(out, depth, 400, spec)
    assert det is not None
    raw = LRBSimulationPipeline.measurement_results_from_sdim_output(out)
    passes = []   # per ancilla and round the record holds the reset, then the readout
    for r in range(depth + spec.check_rounds_offset):
        vals = [np.array([int(raw[w][r * spec.check_stride + 1][s].measurement_value)
                          for s in range(400)]) for w in spec.stabilizer_wires]
        passes.append(np.all(np.stack(vals) % 3 == 0, axis=0).astype(int))
    assert np.array_equal(np.asarray(det[0]), np.stack(passes, axis=1))
    assert np.asarray(det[0]).mean() < 1      # the noise does flip some checks
    assert profile.logical_dimension == 3
