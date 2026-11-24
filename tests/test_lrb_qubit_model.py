"""Validation tests for the qubit [[5,1,2]] Stim model and the exact MLD.

Run with ``module load python/3.11 && python -m pytest tests/test_lrb_qubit_model.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQUENCE = Q.random_clifford_sequence(22, np.random.default_rng(11))


def _single_qubit_tableau(descriptor: str) -> stim.Tableau:
    circuit = stim.Circuit()
    circuit.append("I", [0])
    names = {"I": "I", "H": "H", "P": "S", "X": "X", "Z": "Z"}
    for token in C.clifford_gate_sequence(descriptor):
        circuit.append(names[token], [0])
    return stim.Tableau.from_circuit(circuit)


def test_clifford_catalog_is_the_group():
    tableaus = [_single_qubit_tableau(d) for d in C.SINGLE_QUBIT_CLIFFORDS]
    keys = {str(t) for t in tableaus}
    assert len(C.SINGLE_QUBIT_CLIFFORDS) == 24 and len(keys) == 24
    assert all(str(a.then(b)) in keys for a in tableaus for b in tableaus)


def test_inverse_sequences_invert():
    for descriptor in C.SINGLE_QUBIT_CLIFFORDS:
        ops = (C.logical_clifford_operations(descriptor)
               + C.logical_clifford_operations(descriptor, inverse=True))
        assert C.data_tableau(ops) == stim.Tableau(C.NUM_DATA)


def test_encoder_prepares_plus_logical():
    sim = stim.TableauSimulator()
    for gate, targets in C.encoding_operations():
        sim.do(stim.Circuit(f"{gate} {' '.join(map(str, targets))}"))
    for s in range(4):
        assert sim.peek_observable_expectation(C.stabilizer_pauli(s)) == 1
    assert sim.peek_observable_expectation(stim.PauliString(C.LOGICAL_X)) == 1


@pytest.mark.parametrize("token,x_image,z_image", [
    ("H", "Z", "X"), ("P", "Y", "Z"), ("X", "X", "-Z"), ("Z", "-X", "Z"),
    ("I", "X", "Z")])
def test_logical_gates(token, x_image, z_image):
    tableau = C.data_tableau(C.logical_gate_operations(token))
    for s in range(4):
        image = tableau(C.stabilizer_pauli(s))
        subset = C.decompose_in_stabilizers(image)
        product = stim.PauliString(C.NUM_DATA)
        for k in subset:
            product *= C.stabilizer_pauli(k)
        assert image == product  # stabilizers map to stabilizers with + sign
    logical = {"X": stim.PauliString(C.LOGICAL_X),
               "Z": stim.PauliString(C.LOGICAL_Z)}
    logical["Y"] = logical["X"] * logical["Z"] * 1j

    def matches(image, expected):
        sign = -1 if expected.startswith("-") else 1
        target = logical[expected.lstrip("-")] * sign
        for mask in range(16):
            product = target.copy()
            for k in range(4):
                if (mask >> k) & 1:
                    product *= C.stabilizer_pauli(k)
            if product == image:
                return True
        return False

    assert matches(tableau(logical["X"]), x_image)
    assert matches(tableau(logical["Z"]), z_image)


@pytest.mark.parametrize("terminal", ["data2", "data5"])
@pytest.mark.parametrize("depth", [0, 1, 3, 22])
def test_noiseless_circuits_are_deterministic(terminal, depth):
    lc = Q.build_lrb_circuit(SEQUENCE, depth, 0.0, terminal=terminal)
    dets, obs = lc.circuit.compile_detector_sampler(seed=3).sample(
        64, separate_observables=True)
    assert not dets.any() and not obs.any()
    # Every detector must be deterministic for Stim to build a DEM.
    Q.build_lrb_circuit(SEQUENCE, depth, 0.01,
                        terminal=terminal).circuit.detector_error_model()


def test_circuit_level_distance_is_two_with_full_readout():
    lc = Q.build_lrb_circuit(SEQUENCE, 4, 0.01, terminal="data5")
    error = lc.circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False)
    assert len(error) == 2


def test_two_qubit_readout_has_first_order_undetectable_logicals():
    lc = Q.build_lrb_circuit(SEQUENCE, 2, 0.01, terminal="data2")
    decoder = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                  M.lrb_detector_slots(lc))
    assert decoder.undetectable_logical_probability > 0.01


@pytest.mark.parametrize("depth", [0, 2, 6])
def test_raw_syndromes_match_measurement_records(depth):
    lc = Q.build_lrb_circuit(SEQUENCE, depth, 0.02, terminal="data5")
    circuit = lc.circuit
    measurements = circuit.compile_sampler(seed=5).sample(3000)
    reference = circuit.reference_sample()
    flips = measurements ^ reference[None, :]
    dets, _ = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)
    raw = lc.raw_syndromes(dets)
    # Ancilla records come first: four per round, in round order.
    expected = flips[:, :4 * lc.num_rounds].reshape(-1, lc.num_rounds, 4)
    assert np.array_equal(raw, expected)


def test_policies_match_qutrit_pipeline():
    pytest.importorskip("sdim")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lrb.lrb_simulation import LRBSimulationPipeline as P
    rng = np.random.default_rng(0)
    depths = [0, 2, 4, 6, 10, 14, 18, 20, 22]
    record = (rng.random((1, len(depths), 400, 23)) > 0.08).astype(np.int64)
    for k in range(1, 23):
        for policy, reference in (("const", P.postselect_constant_number),
                                  ("unif", P.postselect_uniform_interval)):
            expected = reference(record, depths, k)
            for d_index, depth in enumerate(depths):
                rounds = B.used_rounds(policy, k, depth)
                ours = record[0, d_index][:, rounds].prod(axis=1)
                assert np.array_equal(ours, expected[0, d_index]), (
                    policy, k, depth)


@pytest.mark.parametrize("terminal", ["data2", "data5"])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("p", [0.002, 0.03])
def test_transfer_matrix_mld_matches_brute_force(terminal, depth, p):
    lc = Q.build_lrb_circuit(SEQUENCE, depth, p, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    decoder = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    joint = M.brute_force_joint(dem)
    n = dem.num_detectors
    dets, _ = lc.circuit.compile_detector_sampler(seed=9).sample(
        2000, separate_observables=True)
    index = dets.astype(np.int64) @ (1 << np.arange(n))
    expected = np.log(joint[index]) - np.log(joint[index + (1 << n)])
    assert np.allclose(decoder.log_likelihood_ratio(dets), expected,
                       atol=1e-9)


def test_const0_circuit_has_no_checks():
    lc = Q.build_lrb_const0_circuit(SEQUENCE, 6, 0.01)
    assert lc.num_rounds == 0 and lc.circuit.num_detectors == 2
    text = str(lc.circuit)
    assert "M 5" not in text and "R 5" not in text


def test_physical_rb_inverts():
    circuit = Q.build_physical_rb_circuit(SEQUENCE, 22, 0.0)
    _, obs = circuit.compile_detector_sampler(seed=1).sample(
        32, separate_observables=True)
    assert not obs.any()


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("policy,k,final", [
    ("unif", 1, False), ("unif", 1, True), ("const", 1, False),
    ("const", 1, True), ("unif", 2, False), ("const", 2, True)])
def test_exact_raw_policy_matches_brute_force(depth, policy, k, final):
    from lrb_qubit import exact as X
    lc = Q.build_lrb_circuit(SEQUENCE, depth, 0.01, terminal="data5")
    dem = lc.circuit.detector_error_model()
    decoder = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    used = [lc.rounds.index(r) for r in B.used_rounds(policy, k, depth)]
    p_acc, p_flip = X.RawPolicyEvaluator(lc, decoder).evaluate(used, final)
    joint = M.brute_force_joint(dem)
    n = dem.num_detectors
    patterns = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    raw = lc.raw_syndromes(patterns)
    ok = ~raw[:, used, :].any(axis=(1, 2))
    if final:
        ok &= ~patterns[:, 4 * lc.num_rounds:].any(axis=1)
    expected_acc = joint[:2 ** n][ok].sum() + joint[2 ** n:][ok].sum()
    expected_flip = joint[2 ** n:][ok].sum()
    assert np.isclose(p_acc, expected_acc, rtol=1e-10, atol=1e-14)
    assert np.isclose(p_flip, expected_flip, rtol=1e-10, atol=1e-14)
