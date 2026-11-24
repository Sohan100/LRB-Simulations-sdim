"""Tests for WP-C decoders: ldpc (BP+OSD, BP+LSD, union-find) and the sliding-window LUT.

Run with
``module load python/3.11 && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest tests/test_qubit512_ldpc.py``.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")
pytest.importorskip("ldpc")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import decoders_ldpc as DL  # noqa: E402
from lrb_qubit import decoders_lut as DT  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQUENCES = E.clifford_sequences(2, 22, 20261028)

# ---------------------------------------------------------------------------
# Check matrices
# ---------------------------------------------------------------------------

# D0 has two explanations with opposite observable effects; the *less*
# probable one (L0, p=0.1) is listed last, as can happen in a Stim DEM.
TOY_DEM = stim.DetectorErrorModel("""
error(0.2) D0
error(0.05) D0 D1
error(0.01) D1 L0
error(0.02) L0
error(0.1) D0 L0
""")


def _columns(matrices):
    h = matrices.check_matrix.toarray()
    lmat = matrices.observables_matrix.toarray()
    return {(tuple(np.flatnonzero(h[:, j])), bool(lmat[0, j])):
            matrices.priors[j] for j in range(h.shape[1])}


def test_mechanism_columns_keep_both_classes():
    m = DL.dem_check_matrices(TOY_DEM, "mechanism")
    cols = _columns(m)
    assert cols == pytest.approx({((0,), False): 0.2, ((0, 1), False): 0.05,
                                  ((1,), True): 0.01, ((0,), True): 0.1})
    assert m.undetectable_probability == pytest.approx(0.02)


def test_dominant_columns_merge_with_most_probable_class():
    m = DL.dem_check_matrices(TOY_DEM, "dominant")
    cols = _columns(m)
    assert cols[((0,), False)] == pytest.approx(0.2 + 0.1 - 2 * 0.2 * 0.1)
    assert ((0,), True) not in cols


def test_ldpc_conversion_keeps_last_listed_observable():
    """Documents the ldpc pitfall: the merged D0 column gets the L0 of the
    less probable, last-listed mechanism."""
    m = DL.dem_check_matrices(TOY_DEM, "ldpc")
    cols = _columns(m)
    assert ((0,), True) in cols and ((0,), False) not in cols
    assert cols[((0,), True)] == pytest.approx(0.26)


def test_bposd_decision_on_ambiguous_syndrome_depends_on_convention():
    syndrome = np.array([1, 0], dtype=bool)[None, :]
    flips = {}
    for mode in ("mechanism", "dominant", "ldpc"):
        decoder = DL.make_bposd(DL.dem_check_matrices(TOY_DEM, mode),
                                max_iter=5)
        flips[mode] = bool(decoder.predict(syndrome)[0])
    # Exact posterior: P(L0 | D0) ~ 0.1 / (0.1 + 0.2) < 1/2, so no flip.
    assert flips == {"mechanism": False, "dominant": False, "ldpc": True}


@pytest.mark.parametrize("depth", [0, 2, 6])
def test_mechanism_matrices_reproduce_dem(depth):
    lc = Q.build_lrb_circuit(SEQUENCES[0], depth, 3e-3, terminal="data5")
    dem = lc.circuit.detector_error_model()
    m = DL.dem_check_matrices(dem, "mechanism")
    mechanisms = M.dem_mechanisms(dem)
    assert m.num_columns == len({(x.detectors, x.flips_observable)
                                 for x in mechanisms})
    assert m.undetectable_probability == 0.0  # data5: distance 2
    assert m.num_detectors == dem.num_detectors


def test_graph_matrices_are_graphlike_and_cover_dem():
    lc = Q.build_lrb_circuit(SEQUENCES[0], 6, 3e-3, terminal="data5")
    graph = DL.graph_check_matrices(
        lc.circuit.detector_error_model(decompose_errors=True))
    weights = np.diff(graph.check_matrix.indptr)
    assert weights.min() >= 1 and weights.max() <= 2
    assert graph.check_matrix.shape[0] == lc.circuit.num_detectors
    with pytest.raises(ValueError):
        DL.graph_check_matrices(lc.circuit.detector_error_model())


def test_unique_rows_roundtrip():
    rng = np.random.default_rng(1)
    events = rng.random((500, 13)) < 0.2
    rows, inverse = DL.unique_rows(events)
    assert np.array_equal(rows[inverse], events)
    assert len({r.tobytes() for r in rows}) == len(rows)


# ---------------------------------------------------------------------------
# ldpc decoders on real circuits
# ---------------------------------------------------------------------------

def test_union_find_rejects_peeling_on_hypergraph():
    m = DL.dem_check_matrices(TOY_DEM, "mechanism")
    DL.make_union_find(m, peeling=True)  # graphlike toy: fine
    lc = Q.build_lrb_circuit(SEQUENCES[0], 2, 3e-3, terminal="data5")
    hyper = DL.dem_check_matrices(lc.circuit.detector_error_model())
    with pytest.raises(ValueError):
        DL.make_union_find(hyper, peeling=True)
    assert DL.make_union_find(hyper).name.startswith("uf_matrix")


@pytest.fixture(scope="module")
def depth2_problem():
    lc = Q.build_lrb_circuit(SEQUENCES[1], 2, 3e-3, terminal="data5")
    dem = lc.circuit.detector_error_model()
    mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    dets, obs = lc.circuit.compile_detector_sampler(seed=17).sample(
        3000, separate_observables=True)
    llr = mld.log_likelihood_ratio(dets)
    return lc, dem, dets, obs[:, 0], llr


def test_standard_decoders_satisfy_syndromes(depth2_problem):
    lc, dem, dets, flip, llr = depth2_problem
    decoders = DL.build_standard_decoders(lc.circuit, dem=dem)
    assert set(decoders) == set(DL.STANDARD_DECODERS)
    for name, decoder in decoders.items():
        pred = decoder.predict(dets)
        assert pred.shape == flip.shape
        assert decoder.unsatisfied == 0, name
        assert decoder.decode_calls <= len(DL.unique_rows(dets)[0])


def test_bposd_is_close_to_mld_and_never_better_in_expectation(
        depth2_problem):
    lc, dem, dets, flip, llr = depth2_problem
    decoder = DL.build_standard_decoders(lc.circuit, ("bposd_cs7",),
                                         dem=dem)["bposd_cs7"]
    pred = decoder.predict(dets)
    disagree = pred != (llr < 0)
    excess = np.tanh(np.abs(llr[disagree]) / 2).sum() / len(llr)
    assert excess < 0.01  # posterior-weighted excess error
    assert disagree.mean() < 0.06


def test_delfosse_nickerson_uf_on_path_graph():
    # Path c0 - c1 - c2 - c3 with boundary edges at both ends (columns 0, 4).
    h = np.zeros((4, 5), dtype=np.uint8)
    for j, (a, b) in enumerate([(0, None), (0, 1), (1, 2), (2, 3), (3, None)]):
        h[a, j] = 1
        if b is not None:
            h[b, j] = 1
    from scipy.sparse import csc_matrix
    m = DL.CheckMatrices(csc_matrix(h), csc_matrix(np.zeros((1, 5), np.uint8)),
                         np.full(5, 0.1), "graph")
    uf = DL.DelfosseNickersonUF(m)
    expected = {(1, 0, 0, 0): [1, 0, 0, 0, 0], (0, 0, 0, 1): [0, 0, 0, 0, 1],
                (0, 1, 1, 0): [0, 0, 1, 0, 0], (1, 1, 0, 0): [0, 1, 0, 0, 0]}
    for syndrome, correction in expected.items():
        assert list(uf.decode(np.array(syndrome, np.uint8))) == correction


def test_delfosse_nickerson_uf_satisfies_lrb_syndromes():
    lc = Q.build_lrb_circuit(SEQUENCES[0], 22, 1e-2, terminal="data5")
    graph = DL.graph_check_matrices(
        lc.circuit.detector_error_model(decompose_errors=True))
    dets, _ = lc.circuit.compile_detector_sampler(seed=2).sample(
        300, separate_observables=True)
    decoder = DL.make_dn_union_find(graph)
    decoder.predict(dets)
    assert decoder.unsatisfied == 0 and decoder.decode_calls > 250


def test_matching_control_runs(depth2_problem):
    lc, dem, dets, flip, llr = depth2_problem
    graph = DL.graph_check_matrices(
        lc.circuit.detector_error_model(decompose_errors=True))
    pred = DL.MatchingControl(graph).predict(dets)
    assert pred.shape == flip.shape


# ---------------------------------------------------------------------------
# Sliding-window LUT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("terminal", ["data5", "data2"])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("p", [2e-3, 3e-2])
def test_lut_with_full_window_is_exact_mld(terminal, depth, p):
    lc = Q.build_lrb_circuit(SEQUENCES[0], depth, p, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    dets, _ = lc.circuit.compile_detector_sampler(seed=4).sample(
        2000, separate_observables=True)
    lut = DT.build_lut(lc, window=mld.num_slots, dem=dem)
    assert lut.num_interior == 0
    pred, llr_lut = lut.predict(dets, return_llr=True)
    llr = mld.log_likelihood_ratio(dets)
    assert np.allclose(llr_lut, llr, atol=1e-9)
    assert np.array_equal(pred, llr < 0)


def test_kernels_are_normalized():
    lc = Q.build_lrb_circuit(SEQUENCES[0], 6, 1e-2, terminal="data5")
    kernels, slot_dets = DT.earliest_slot_kernels(
        lc.circuit.detector_error_model(), M.lrb_detector_slots(lc))
    assert len(kernels) == lc.num_rounds + 1
    assert [len(d) for d in slot_dets] == [4] * lc.num_rounds + [2]
    for kernel in kernels:
        assert kernel.sum() == pytest.approx(1.0)
        assert (kernel >= 0).all()


def _random_dem(rng, sizes, per_group=4):
    """Random DEM on consecutive slots; returns (dem, slots, mechanisms)."""
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    lines, mechanisms = [], []
    for u, n_u in enumerate(sizes):
        n_next = sizes[u + 1] if u + 1 < len(sizes) else 0
        for _ in range(per_group):
            while True:
                x = int(rng.integers(0, 2 ** n_u))
                y = int(rng.integers(0, 2 ** n_next)) if n_next else 0
                if x:  # earliest slot must be u
                    break
            dets = [offsets[u] + k for k in range(n_u) if (x >> k) & 1]
            dets += [offsets[u + 1] + k for k in range(n_next) if (y >> k) & 1]
            flips = bool(rng.integers(0, 2))
            prob = float(rng.uniform(0.02, 0.3))
            mechanisms.append((u, x, y, flips, prob))
            lines.append(f"error({prob}) " + " ".join(f"D{d}" for d in dets)
                         + (" L0" if flips else ""))
    lines.append(f"detector D{offsets[-1] - 1}")
    lines.append("logical_observable L0")
    slots = np.repeat(np.arange(len(sizes)), sizes)
    return stim.DetectorErrorModel("\n".join(lines)), slots, mechanisms


def _window_scores(mechanisms, sizes, t, last, constraints, final):
    """Brute force over the mechanisms of groups t..last.

    ``constraints[u]`` is the required syndrome of slot u (slot t: sigma).
    Returns a dict (kappa, lam) -> probability for interior steps, or
    parity -> probability when ``final``.
    """
    group = [m for m in mechanisms if t <= m[0] <= last]
    scores: dict = {}
    for picks in itertools.product([0, 1], repeat=len(group)):
        weight = 1.0
        bits = {u: 0 for u in range(t, last + 2)}
        kappa, lam, parity = 0, 0, 0
        for pick, (u, x, y, flips, prob) in zip(picks, group):
            weight *= prob if pick else 1.0 - prob
            if not pick:
                continue
            bits[u] ^= x
            bits[u + 1] ^= y
            parity ^= int(flips)
            if u == t:
                kappa ^= y
                lam ^= int(flips)
        if any(bits[u] != constraints[u] for u in range(t, last + 1)):
            continue
        key = parity if final else (kappa, lam)
        scores[key] = scores.get(key, 0.0) + weight
    return scores


@pytest.mark.parametrize("window", [1, 2, 3])
@pytest.mark.parametrize("commit", ["joint", "carry"])
def test_lut_tables_match_brute_force(window, commit):
    rng = np.random.default_rng(100 + window)
    sizes = [2, 2, 2, 1]
    dem, slots, mechanisms = _random_dem(rng, sizes)
    lut = DT.SlidingWindowLUT(dem, slots, window, commit=commit)
    for t, table in enumerate(lut.tables):
        last = t + window - 1
        width = sum(sizes[t:last + 1])
        for index in range(2 ** width):
            constraints, shift = {}, 0
            for u in range(t, last + 1):
                constraints[u] = (index >> shift) & ((1 << sizes[u]) - 1)
                shift += sizes[u]
            scores = _window_scores(mechanisms, sizes, t, last, constraints,
                                    final=False)
            if not scores:
                assert not lut.valid[t][index]
                continue
            if commit == "joint":
                best = max(scores, key=scores.get)
            else:
                carry = {}
                for (k, lam), v in scores.items():
                    carry[k] = carry.get(k, 0.0) + v
                k_best = max(carry, key=carry.get)
                lam_best = int(scores.get((k_best, 1), 0.0)
                               > scores.get((k_best, 0), 0.0))
                best = (k_best, lam_best)
            n_next = sizes[t + 1]
            out = int(table[index])
            assert (out & ((1 << n_next) - 1), out >> n_next) == best
    t = lut.final_start
    width = sum(sizes[t:])
    for index in range(2 ** width):
        constraints, shift = {}, 0
        for u in range(t, len(sizes)):
            constraints[u] = (index >> shift) & ((1 << sizes[u]) - 1)
            shift += sizes[u]
        scores = _window_scores(mechanisms, sizes, t, len(sizes) - 1,
                                constraints, final=True)
        p0, p1 = scores.get(0, 0.0), scores.get(1, 0.0)
        assert bool(lut.final_table[index]) == (p1 > p0)
        if p0 > 0 and p1 > 0:
            assert lut.final_llr[index] == pytest.approx(np.log(p0 / p1))


@pytest.mark.parametrize("window", [1, 2, 3])
def test_lut_predict_matches_reference_stepping(window):
    """Vectorized ``predict`` equals a per-syndrome brute-force run of the
    sliding-window algorithm (carry and frame updates included)."""
    rng = np.random.default_rng(7 + window)
    sizes = [2, 2, 2, 1]
    dem, slots, mechanisms = _random_dem(rng, sizes)
    lut = DT.SlidingWindowLUT(dem, slots, window)
    n_det = int(sum(sizes))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    events = np.array(list(itertools.product([0, 1], repeat=n_det)),
                      dtype=bool)[:, ::-1]
    predicted = lut.predict(events)
    for row, pred in zip(events, predicted):
        s = [int(sum(int(row[offsets[u] + k]) << k for k in range(sizes[u])))
             for u in range(len(sizes))]
        carry, frame, valid = 0, 0, True
        for t in range(lut.num_interior):
            last = t + window - 1
            constraints = {u: s[u] for u in range(t + 1, last + 1)}
            constraints[t] = s[t] ^ carry
            scores = _window_scores(mechanisms, sizes, t, last, constraints,
                                    final=False)
            if not scores:
                valid = False
                break
            carry, lam = max(scores, key=scores.get)
            frame ^= lam
        if not valid:
            continue
        t = lut.final_start
        constraints = {u: s[u] for u in range(t + 1, len(sizes))}
        constraints[t] = s[t] ^ carry
        scores = _window_scores(mechanisms, sizes, t, len(sizes) - 1,
                                constraints, final=True)
        frame ^= int(scores.get(1, 0.0) > scores.get(0, 0.0))
        assert bool(pred) == bool(frame)


@pytest.mark.parametrize("window", [2, 3])
def test_lut_on_lrb_circuit_is_valid_and_close_to_mld(window):
    lc = Q.build_lrb_circuit(SEQUENCES[1], 6, 2e-3, terminal="data5")
    dem = lc.circuit.detector_error_model()
    mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    dets, _ = lc.circuit.compile_detector_sampler(seed=8).sample(
        5000, separate_observables=True)
    llr = mld.log_likelihood_ratio(dets)
    lut = DT.build_lut(lc, window, dem=dem)
    pred = lut.predict(dets)
    assert lut.invalid_hits == 0
    excess = np.tanh(np.abs(llr[pred != (llr < 0)]) / 2).sum() / len(llr)
    assert excess < 0.02
    assert lut.table_bits() == sum(
        t.size * (lut.sizes[i + 1] + 1) for i, t in enumerate(lut.tables)
    ) + lut.final_table.size


# ---------------------------------------------------------------------------
# Script smoke test
# ---------------------------------------------------------------------------

def test_comparison_script_smoke(tmp_path):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    subprocess.run(
        [sys.executable, str(ROOT / "studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py"),
         "--out", str(tmp_path), "--p-list", "2e-3", "--depths", "0,2",
         "--num-seq", "1", "--shots", "400", "--ldpc-unique", "50",
         "--windows", "1,4", "--commits", "joint",
         "--decoders", "bposd_cs7,uf_graph", "--workers", "1"],
        check=True, env=env, capture_output=True, timeout=300)
    text = (tmp_path / "summary.csv").read_text().splitlines()
    header = text[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in text[1:]]
    names = {r["decoder"] for r in rows}
    assert {"mld", "lut_w1_joint", "lut_w4_joint", "bposd_cs7",
            "uf_graph"} <= names
    for r in rows:
        if r["decoder"] == "mld" or r["decoder"] == "lut_w4_joint":
            assert float(r["agreement"]) == 1.0  # w=4 >= T for depth <= 2
    assert (tmp_path / "calibration.csv").exists()
