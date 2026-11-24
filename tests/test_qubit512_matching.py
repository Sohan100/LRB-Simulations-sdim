"""Tests for the matching decoders of work package B (``lrb_qubit.decoders_matching``).

Run with
``module load python/3.11 && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_qubit512_matching.py``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pymatching = pytest.importorskip("pymatching")
stim = pytest.importorskip("stim")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import decoders_matching as D  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQUENCES = E.clifford_sequences(3, 22, 20261028)


def _circuit(depth: int, p: float, seq: int = 0):
    return Q.build_lrb_circuit(SEQUENCES[seq], depth, p, terminal="data5")


def _load_script():
    # The script re-execs the interpreter unless these are "1".
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    path = ROOT / "studies/correction-decoders/qubit/scripts/matching_vs_mld.py"
    spec = importlib.util.spec_from_file_location("matching_vs_mld", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- DEM utilities and PyMatching behaviour ---------------------------------

def test_error_components_split():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0 ^ D2 ^ L0")
    (instruction,) = list(dem.flattened())
    assert D.error_components(instruction) == [((0, 1), 1), ((2,), 0),
                                               ((), 1)]


def test_plain_pymatching_drops_undetectable_component():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0 ^ D1 ^ L0\nerror(0.01) D0\nerror(0.01) D1")
    edges = {(u, v): data["fault_ids"] for u, v, data in
             pymatching.Matching.from_detector_error_model(dem).edges()}
    # The mechanism flips D0 and D1 but not L0; the graph says L0.
    assert edges[(0, None)] == {0} and edges[(1, None)] == set()
    stripped, changed = D.strip_undetectable_components(dem)
    assert changed == 1
    assert str(stripped).splitlines()[0].endswith("D0 L0 ^ D1")


def test_correlated_pymatching_rejects_undetectable_component():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0 ^ D1 ^ L0\n"
                                  "error(0.01) D0 L0\nerror(0.01) D1")
    with pytest.raises(ValueError, match="undetectable component"):
        pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=True)
    decoder = D.PyMatchingDecoder(dem, correlated=True)
    assert decoder.stripped_instructions == 1


def test_first_added_observable_wins():
    dem = stim.DetectorErrorModel("error(0.01) D0\nerror(0.1) D0 L0")
    matching = pymatching.Matching.from_detector_error_model(dem)
    assert not matching.decode(np.array([1], dtype=np.uint8))[0]
    model = D.pymatching_merge_model(dem)
    assert model[(0,)].conflict and not model[(0,)].kept_flips
    assert math.isclose(model[(0,)].lost_probability, 0.1)


@pytest.mark.parametrize("depth", [0, 2, 6])
def test_merge_model_reproduces_pymatching_graph(depth):
    dem = _circuit(depth, 2e-3).circuit.detector_error_model(
        decompose_errors=True)
    model = D.pymatching_merge_model(dem)
    graph = {}
    for u, v, data in pymatching.Matching.from_detector_error_model(
            dem).edges():
        key = (u,) if v is None else tuple(sorted((u, v)))
        graph[key] = data
    assert set(graph) == set(model)
    for key, data in graph.items():
        assert (0 in data["fault_ids"]) == model[key].kept_flips
        assert math.isclose(data["error_probability"],
                            model[key].merged_probability, rel_tol=1e-9)


@pytest.mark.parametrize("depth", [2, 6])
def test_stripped_dem_gives_the_plain_graph(depth):
    dem = _circuit(depth, 2e-3, seq=0).circuit.detector_error_model(
        decompose_errors=True)
    stripped, _ = D.strip_undetectable_components(dem)
    a = pymatching.Matching.from_detector_error_model(dem).edges()
    b = pymatching.Matching.from_detector_error_model(stripped).edges()
    assert len(a) == len(b)
    for (u1, v1, d1), (u2, v2, d2) in zip(sorted(a, key=str),
                                          sorted(b, key=str)):
        assert (u1, v1, d1["fault_ids"]) == (u2, v2, d2["fault_ids"])
        assert math.isclose(d1["weight"], d2["weight"], rel_tol=1e-12)


@pytest.mark.parametrize("depth", [0, 2, 6])
def test_decomposition_preserves_mechanisms(depth):
    circuit = _circuit(depth, 5e-3, seq=1).circuit
    und = {(m.detectors, m.flips_observable): m.probability
           for m in M.dem_mechanisms(circuit.detector_error_model())}
    merged: dict = {}
    for instruction in circuit.detector_error_model(
            decompose_errors=True).flattened():
        if instruction.type != "error":
            continue
        total, mask = D._symmetric_difference(
            D.error_components(instruction))
        key = (total, bool(mask & 1))
        q = merged.get(key, 0.0)
        p = instruction.args_copy()[0]
        merged[key] = q * (1 - p) + p * (1 - q)
    assert set(merged) == set(und)
    for key, value in und.items():
        assert math.isclose(merged[key], value, rel_tol=1e-9)


def test_observable_as_detector_circuit():
    lc = _circuit(2, 0.0)
    new, index = D.observable_as_detector_circuit(lc.circuit)
    assert index == lc.circuit.num_detectors
    assert new.num_detectors == index + 1 and new.num_observables == 0
    noisy, _ = D.observable_as_detector_circuit(_circuit(2, 0.01).circuit)
    dets, obs = _circuit(2, 0.01).circuit.compile_detector_sampler(
        seed=1).sample(500, separate_observables=True)
    # Same records: the new detector is the old observable.
    dets2 = noisy.compile_detector_sampler(seed=1).sample(500)
    assert np.array_equal(dets2[:, :-1], dets)
    assert np.array_equal(dets2[:, -1], obs[:, 0])


# --- Exact min-sum transfer matrix ---------------------------------------------

def _mechanism_sets(lc):
    und = lc.circuit.detector_error_model()
    dec = lc.circuit.detector_error_model(decompose_errors=True)
    return {
        "mwe": D.mechanisms_from_dem(und),
        "pymatching": D.mechanisms_from_matching(
            pymatching.Matching.from_detector_error_model(dec)),
        "classaware": D.classaware_mechanisms(dec),
    }


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("p", [0.002, 0.03])
def test_minsum_matches_brute_force(depth, p):
    lc = _circuit(depth, p, seq=1)
    n = lc.circuit.num_detectors
    slots = M.lrb_detector_slots(lc)
    dets, _ = lc.circuit.compile_detector_sampler(seed=4).sample(
        1500, separate_observables=True)
    index = dets.astype(np.int64) @ (1 << np.arange(n))
    for name, mechanisms in _mechanism_sets(lc).items():
        weights = D.MinSumTransferMatrix(slots, mechanisms).class_weights(dets)
        table = D.brute_force_class_weights(n, mechanisms)
        expected = np.stack([table[index], table[index + (1 << n)]], axis=1)
        assert np.allclose(weights, expected, atol=1e-9), name


def test_minsum_unique_and_batching_do_not_change_results():
    lc = _circuit(6, 0.01, seq=2)
    tm = D.MinSumTransferMatrix(M.lrb_detector_slots(lc),
                                D.mechanisms_from_dem(
                                    lc.circuit.detector_error_model()))
    dets, _ = lc.circuit.compile_detector_sampler(seed=8).sample(
        700, separate_observables=True)
    a = tm.class_weights(dets)
    b = tm.class_weights(dets, batch_size=37, unique=False)
    assert np.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("depth", [2, 6, 14])
def test_minsum_on_pymatching_graph_reproduces_mwpm(depth):
    lc = _circuit(depth, 5e-3, seq=0)
    dec = lc.circuit.detector_error_model(decompose_errors=True)
    mwpm = D.PyMatchingDecoder(dec)
    tm = D.MinSumTransferMatrix(M.lrb_detector_slots(lc),
                                D.mechanisms_from_matching(mwpm.matching))
    dets, _ = lc.circuit.compile_detector_sampler(seed=6).sample(
        3000, separate_observables=True)
    prediction, weight = mwpm.predict_with_weights(dets)
    classes = tm.class_weights(dets)
    assert np.allclose(classes.min(axis=1), weight,
                       atol=D.PYMATCHING_WEIGHT_TOL)
    decided = np.abs(classes[:, 1] - classes[:, 0]) > D.PYMATCHING_WEIGHT_TOL
    argmin = classes[:, 1] < classes[:, 0]
    assert np.array_equal(prediction[decided], argmin[decided])


@pytest.mark.parametrize("depth", [0, 2, 6])
def test_observable_detector_matching_is_exact_on_its_graph(depth):
    lc = _circuit(depth, 5e-3, seq=1)
    obs = D.ObservableDetectorMatching(lc.circuit)
    mechanisms = D.obsdet_mechanisms(obs.matching, obs.obs_detector)
    tm = D.MinSumTransferMatrix(M.lrb_detector_slots(lc), mechanisms)
    dets, _ = lc.circuit.compile_detector_sampler(seed=7).sample(
        2000, separate_observables=True)
    assert np.allclose(obs.class_weights(dets), tm.class_weights(dets),
                       atol=D.PYMATCHING_WEIGHT_TOL)


# --- Decoder suite ---------------------------------------------------------------

def test_suite_zero_syndrome_and_shapes():
    lc = _circuit(6, 2e-3, seq=2)
    suite = D.DecoderSuite.build(lc)
    zero = np.zeros((3, lc.circuit.num_detectors), dtype=bool)
    pred, gaps, _ = suite.run(zero)
    assert set(pred) == set(D.SOFT_DECODERS) | set(D.HARD_DECODERS)
    for name, value in pred.items():
        assert value.shape == (3,) and not value.any(), name
    for name, value in gaps.items():
        assert (value > 1.0).all(), name


def test_first_order_mld_equals_mwe_and_mwpm_is_worse_than_none():
    """MLD and MWE coincide on single faults to first order (p -> 0).

    They can differ only on first-order ties: symmetric pairs whose
    probabilities are equal or differ by O(p) (MWE gap -> 0 as p -> 0), where
    both choices cost the same first-order mass and MLD's O(p) multi-fault
    terms decide.
    """
    lc = _circuit(2, 1e-5, seq=0)
    table = D.single_fault_table(D.DecoderSuite.build(lc))
    untied = table["gap_mwe"] >= 1e-3
    assert (~untied).any()
    assert np.array_equal(table["wrong_mld"][untied],
                          table["wrong_mwe"][untied])
    prob = table["probability"]
    c = {k[6:]: (prob * v).sum() / 1e-5 for k, v in table.items()
         if k.startswith("wrong_")}
    assert c["mld"] == pytest.approx(c["mwe"], rel=1e-6)
    # MLD minimizes the full posterior error, so on near-ties it may lose
    # O(p) relative single-fault mass to MWE.
    assert c["mld"] <= min(c.values()) * (1 + 1e-4)
    # Plain MWPM is worse than not decoding at all at first order here.
    assert c["mwpm"] > c["none"] > c["mld"]


def test_depth0_first_order_coefficients():
    """Depth 0 is sequence independent; regression values in units of p."""
    table = D.single_fault_table(D.DecoderSuite.build(_circuit(0, 1e-5)))
    prob = table["probability"]
    c = {k[6:]: (prob * v).sum() / 1e-5 for k, v in table.items()
         if k.startswith("wrong_")}
    assert c["mld"] == pytest.approx(c["mwe"])
    assert c["mld"] == pytest.approx(11.07, abs=0.02)
    assert c["mwpm"] == pytest.approx(12.80, abs=0.02)
    assert c["none"] == pytest.approx(13.33, abs=0.02)


def test_correlated_suite_builds_where_plain_correlated_fails():
    lc = _circuit(6, 2e-3, seq=0)
    dec = lc.circuit.detector_error_model(decompose_errors=True)
    _, changed = D.strip_undetectable_components(dec)
    if changed == 0:
        pytest.skip("this sequence has no undetectable components")
    with pytest.raises(ValueError):
        pymatching.Matching.from_detector_error_model(
            dec, enable_correlations=True)
    corr = D.PyMatchingDecoder(dec, correlated=True)
    dets, _ = lc.circuit.compile_detector_sampler(seed=3).sample(
        200, separate_observables=True)
    assert corr.predict(dets).shape == (200,)


def test_census_counts_are_consistent():
    census = D.decomposition_census(_circuit(6, 2e-3, seq=0).circuit)
    assert census.decomposed_lines == (
        census.graphlike_kept_lines + census.graphlike_split_lines
        + census.hyper_clean_lines + census.undetectable_component_lines)
    assert census.hyperedges > 0 and census.pm_conflict_edges > 0
    # Every observable-detector component is graphlike.
    assert census.obsdet_l_components > 0


# --- Script statistics helpers ---------------------------------------------------

def test_error_at_acceptance_and_jackknife():
    script = _load_script()
    # Bins in increasing gap: (correct, wrong).
    hist = np.array([[10, 10], [30, 0], [60, 0]])
    acc, wrong, shots = script.error_at_acceptance(hist, 1.0)
    assert (acc, wrong, shots) == (1.0, 10.0, 110.0)
    _, wrong, shots = script.error_at_acceptance(hist, 0.6)
    assert wrong == 0.0 and shots == pytest.approx(66.0)
    # Accept 100 of 110: the last 10 come from the mixed bin (half wrong).
    _, wrong, shots = script.error_at_acceptance(hist, 100 / 110)
    assert shots == pytest.approx(100.0) and wrong == pytest.approx(5.0)
    _, wrong, _ = script.error_at_acceptance(hist, 105 / 110)
    assert wrong == pytest.approx(7.5)
    values = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
    est, se = script.jackknife(values, lambda s: s[0] / s[1])
    assert est == pytest.approx(0.2)
    # Jackknife of a ratio with equal denominators equals the SEM of x/n.
    x = values[:, 0] / values[:, 1]
    assert se == pytest.approx(np.std(x, ddof=1) / np.sqrt(3))


def test_physical_rb_fidelity_is_exact():
    """(1 - 4q/3)^N with q = 0.1 p_RB matches the lead's sampled physical RB."""
    script = _load_script()
    sequence = SEQUENCES[0]
    p_rb = 0.05
    shots = 200_000
    table = E.run_physical_rb(p_rb, [sequence], [6], shots, seed=11)
    sampled = abs(1 - 2 * table.errors[0, 0] / table.shots[0, 0])
    exact = script.physical_rb_fidelity(sequence, 6, p_rb)
    assert abs(sampled - exact) < 5 * math.sqrt((1 - exact ** 2) / shots)
    assert script.physical_rb_fidelity(sequence, 22, 0.0) == 1.0
