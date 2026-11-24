"""Qubit five-moment one-ancilla round (profile ``folded_qubit_prod5b``).

The qubit counterpart of ``test_prod5b_round.py``: the profile differs from
``folded_qubit`` only in the order of the check-round gates (the same gates and
order as ``folded_qutrit_prod5b``, built in dimension two); the ASAP schedule has
5 gate moments (published round: 7); noiseless rounds pass and keep the logical;
every single data Pauli gives exactly the published round's readouts; and the
detector (DEM) and raw unpack paths agree.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import replace

import numpy as np
from sdim.program import Program

from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_simulation_profiles import (
    FOLDED_QUBIT_CODE_NAME,
    PROD5B_FOLDED_CODE_NAME,
    PROD5B_FOLDED_QUBIT_CODE_NAME,
    CodeSimulationProfileRegistry as Reg,
)
from lrb.lrb_simulation import LRBSimulationPipeline, NoiseModelUtils

SHOTS = 200


def generator(code_name, noise="si1000"):
    return LRBCircuitGenerator(
        code_definition=Reg.resolve_code_profile(code_name).code_definition,
        noise_model=noise)


def ops(circuit, code_name=FOLDED_QUBIT_CODE_NAME):
    g = generator(code_name)
    return [(op.gate_name, *g._operation_qudits(op)) for op in circuit.operations]


def test_round_is_the_qutrit_order_and_uses_the_published_gates():
    qutrit = ops(generator(PROD5B_FOLDED_CODE_NAME)._check_round_circuit(), PROD5B_FOLDED_CODE_NAME)
    new = ops(generator(PROD5B_FOLDED_QUBIT_CODE_NAME)._check_round_circuit())
    prod = ops(generator(FOLDED_QUBIT_CODE_NAME)._check_round_circuit())
    assert new == qutrit
    assert Counter(new) == Counter(prod)
    assert Reg.resolve_code_profile(PROD5B_FOLDED_QUBIT_CODE_NAME).code_definition.dimension == 2


def test_schedule_depths():
    for name, gate_moments in ((FOLDED_QUBIT_CODE_NAME, 7), (PROD5B_FOLDED_QUBIT_CODE_NAME, 5)):
        g = generator(name)
        moments = g.schedule_moments(g._check_round_circuit())
        assert sum(1 for c, _ in moments if not c) == gate_moments
        assert sum(1 for c, _ in moments if c) == 2      # reset, readout


def test_everything_but_the_round_matches_folded_qubit():
    a = Reg.resolve_code_profile(FOLDED_QUBIT_CODE_NAME)
    b = Reg.resolve_code_profile(PROD5B_FOLDED_QUBIT_CODE_NAME)
    da, db = a.code_definition, b.code_definition
    assert ops(da.logical_plus_initial_state()) == ops(db.logical_plus_initial_state())
    for t in ("H", "H_INV", "P", "P_INV", "X", "Z"):
        assert ops(da.logical_gate_circuit(t)) == ops(db.logical_gate_circuit(t))
    assert ops(da.terminal_measurement()) == ops(db.terminal_measurement())
    assert ops(da.terminal_const0_measurement()) == ops(db.terminal_const0_measurement())
    assert da.stabilizer_detector_wires == db.stabilizer_detector_wires
    assert list(da.clifford_strings) == list(db.clifford_strings)
    sa = Reg._UNPACK_SPECS[FOLDED_QUBIT_CODE_NAME]
    sb = Reg._UNPACK_SPECS[PROD5B_FOLDED_QUBIT_CODE_NAME]
    strip = dict(logical_outcome_fn=None, x_stabilizer_check_fn=None, stabilizer_pass_fn=None)
    assert replace(sa, **strip) == replace(sb, **strip)
    for m in itertools.product(range(2), repeat=2):
        assert sa.logical_outcome_fn(list(m), 4) == sb.logical_outcome_fn(list(m), 4)
    for m in itertools.product(range(2), repeat=5):
        assert sa.x_stabilizer_check_fn(list(m)) == sb.x_stabilizer_check_fn(list(m))
    for m in itertools.product(range(2), repeat=4):
        assert sa.stabilizer_pass_fn(list(m)) == sb.stabilizer_pass_fn(list(m))
    assert a.logical_dimension == b.logical_dimension == 2


def readouts(code_name, rounds=3, sandwich=False, error=None):
    definition = Reg.resolve_code_profile(code_name).code_definition
    round_circuit = generator(code_name)._check_round_circuit()
    c = definition.logical_plus_initial_state()
    if error is not None:
        wire, gate = error
        c.add_gate(gate, wire)
    if sandwich:
        c = c + definition.logical_gate_circuit("H")
    for _ in range(rounds):
        c = c + round_circuit
    if sandwich:
        c = c + definition.logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H", q)
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
    value = lambda w, k, s: int(m[w][k][s].measurement_value) % 2  # noqa: E731
    return ([tuple(value(w, k, s) for w, k in reads) for s in range(SHOTS)],
            [[value(w, k, s) for w, k in final] for s in range(SHOTS)])


def test_noiseless_rounds_pass_and_keep_the_logical():
    for sandwich in (False, True):
        reads, x = readouts(PROD5B_FOLDED_QUBIT_CODE_NAME, sandwich=sandwich)
        assert all(v == 0 for shot in reads for v in shot)
        for a in x:
            assert (a[0] + a[1] + a[2]) % 2 == 0
            assert (a[1] + a[3] + a[4]) % 2 == 0
            assert (a[0] + a[3]) % 2 == 0


def test_single_data_paulis_give_the_published_syndromes():
    for wire, gate in itertools.product(range(5), ("X", "Z")):
        new, _ = readouts(PROD5B_FOLDED_QUBIT_CODE_NAME, rounds=1, error=(wire, gate))
        old, _ = readouts(FOLDED_QUBIT_CODE_NAME, rounds=1, error=(wire, gate))
        assert len(set(new)) == 1 and set(new) == set(old), (wire, gate)
        assert any(new[0]), (wire, gate)


def test_detector_and_raw_unpack_agree():
    g = generator(PROD5B_FOLDED_QUBIT_CODE_NAME)
    depth = 2
    circuit = g.generate_lrb_clifford_sequence([depth], with_noise=True)[-1]
    g.update_noise_param(circuit, 0.01)
    NoiseModelUtils.ensure_noise_params(circuit)
    spec = Reg._UNPACK_SPECS[PROD5B_FOLDED_QUBIT_CODE_NAME]
    out = LRBSimulationPipeline.simulate_circuit(circuit, 400, simulation_backend="sdim")
    det = LRBSimulationPipeline.try_unpack_detector_results_from_spec(out, depth, 400, spec)
    assert det is not None
    raw = LRBSimulationPipeline.measurement_results_from_sdim_output(out)
    passes = []   # per ancilla and round the record holds the reset, then the readout
    for r in range(depth + spec.check_rounds_offset):
        vals = [np.array([int(raw[w][r * spec.check_stride + 1][s].measurement_value)
                          for s in range(400)]) for w in spec.stabilizer_wires]
        passes.append(np.all(np.stack(vals) % 2 == 0, axis=0).astype(int))
    assert np.array_equal(np.asarray(det[0]), np.stack(passes, axis=1))
    assert np.asarray(det[0]).mean() < 1
