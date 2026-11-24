"""Tests for WP J: hardware layout, compilation, device model, DEM learning.

Run with
``module load python/3.11 && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m pytest tests/test_qubit512_hw.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "studies" / "correction-decoders" / "qubit" / "scripts"))

import dem_learning as D  # noqa: E402
import hardware_model as H  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQS = E.clifford_sequences(4, 22, 20261028)


def _adjacent(a, b, coords):
    (r1, c1), (r2, c2) = coords[a], coords[b]
    return abs(r1 - r2) + abs(c1 - c2) == 1


# --------------------------------------------------------------------- layout

def test_tanner_graph_embeds_uniquely_as_3x3_patch():
    edges = H.tanner_edges()
    assert len(edges) == 12
    assert all(_adjacent(a, b, H.PATCH_COORDS) for a, b in edges)
    # 8 labellings of the 3x3 grid = its 8 symmetries: unique up to symmetry
    assert len(H.embeddings(H.square_grid_edges(3, 3), edges)) == 8
    # in a 5x5 grid: 9 positions x 8 symmetries, nothing else
    assert len(H.embeddings(H.square_grid_edges(5, 5), edges)) == 72


def test_logical_coupling_is_not_native():
    assert H.shortest_cycle_through(H.tanner_edges() + [(2, 3)], (2, 3)) == 5
    (r2, c2), (r3, c3) = H.PATCH_COORDS[2], H.PATCH_COORDS[3]
    assert abs(r2 - r3) + abs(c2 - c3) == 4


def test_garnet_coupling_map_and_patches():
    assert len(H.GARNET_COUPLERS) == 30
    assert len(set(H.GARNET_COORDS)) == 20
    assert all(_adjacent(a, b, H.GARNET_COORDS) for a, b in H.GARNET_COUPLERS)
    patches = H.garnet_patches()
    assert len(patches) == 4
    assert all(p["distance_q2_q3"] == 4 for p in patches)
    for patch in patches:
        labels = patch["labels"]
        couplers = {frozenset(e) for e in H.GARNET_COUPLERS}
        assert all(frozenset((labels[a], labels[b])) in couplers
                   for a, b in H.tanner_edges())


def test_frame_ancilla_roles_match_patch_neighbours():
    neighbours = {a: {q for q in H.DATA if _adjacent(q, a, H.PATCH_COORDS)}
                  for a in H.ANCILLAS}
    for frame in (0, 1):
        for s in range(4):
            support = {H.PERM[frame][q] for q in H.stabilizer_support(s)}
            assert neighbours[H.ANC_OF_STAB[frame][s]] == support


# --------------------------------------------------------------- compilation

def test_cnot_from_cz():
    cz = stim.Circuit("H 1\nCZ 0 1\nH 1")
    assert stim.Tableau.from_circuit(cz) == stim.Tableau.from_circuit(
        stim.Circuit("CX 0 1"))


def _layers_circuit(layers):
    circuit = stim.Circuit()
    circuit.append("I", [8])
    for layer in layers:
        if layer.kind in ("1q", "2q", "virtual"):
            for gate, targets in layer.ops:
                circuit.append(gate, list(targets))
    return circuit


@pytest.mark.parametrize("bridge", [0, 4])
def test_routed_cz23_equals_cz(bridge):
    layers = H.routed_cz23_layers(bridge)
    allowed = {frozenset(e) for e in H.tanner_edges()}
    for layer in layers:
        if layer.kind == "2q":
            assert all(frozenset(t) in allowed for _, t in layer.ops)
    assert H.layer_counts(layers)["cz"] == 8
    circuit = _layers_circuit(layers)
    for trial in range(25):
        tab = stim.Tableau.random(5)
        sim = stim.TableauSimulator()
        sim.do_tableau(tab, list(range(5)))
        sim.do(circuit)
        sim.do(stim.Circuit("CZ 2 3"))
        sim.do_tableau(tab.inverse(), list(range(5)))
        assert all(sim.peek_z(q) == 1 for q in range(9))


def test_gate_counts():
    rnd = H.layer_counts(H.check_round_layers(0))
    assert rnd["cz"] == 12 and rnd["pulses_1q"] == 18
    assert rnd["layers_2q"] == 6 and rnd["layers_meas"] == 1
    tokens = H.logical_token_counts("routed")
    assert tokens["H"].get("cz", 0) == 0 and tokens["H"]["pulses_1q"] == 5
    assert tokens["P"]["cz"] == 8
    assert tokens["X"].get("pulses_1q", 0) == 0  # Pauli frame
    routed = H.clifford_catalog_counts(H.IQM_GARNET, "routed")
    native = H.clifford_catalog_counts(H.IQM_GARNET, "native")
    assert routed["mean_cz"] == pytest.approx(8 * 2 / 3)
    assert native["mean_cz"] == pytest.approx(2 / 3)
    phys = H.clifford_catalog_counts(H.IQM_GARNET, "routed",
                                     virtual_swap=False)
    assert phys["mean_cz"] == pytest.approx(8 * 2 / 3 + 24 * 5 / 6)
    assert H.repo_gate_counts()["two_qubit_per_round"] == 12


@pytest.mark.parametrize("kwargs", [
    {}, {"reset": False}, {"cz23": "native"}, {"virtual_paulis": False},
    {"bridge": 4, "reset": False}])
def test_noiseless_hw_circuits_are_deterministic(kwargs):
    for seq in SEQS[:2]:
        for depth in (0, 1, 5, 22):
            for terminal in ("data5", "data2"):
                lc = H.build_hw_lrb_circuit(seq, depth, H.IQM_GARNET,
                                            terminal=terminal, noisy=False,
                                            **kwargs)
                dets, obs = lc.circuit.compile_detector_sampler(
                    seed=3).sample(32, separate_observables=True)
                assert not dets.any() and not obs.any()
    lc0 = H.build_hw_const0_circuit(SEQS[0], 12, H.IQM_GARNET, noisy=False)
    dets, obs = lc0.circuit.compile_detector_sampler(seed=4).sample(
        32, separate_observables=True)
    assert not dets.any() and not obs.any()


def test_frames_follow_logical_h_parity():
    seq = SEQS[0]
    depth = 8
    _, info = H.build_hw_lrb_circuit(seq, depth, H.IQM_GARNET, noisy=False,
                                     return_info=True)
    parity = 0
    for j in range(depth):
        parity ^= seq[j].count("H") % 2
        assert info.frames[j] == parity
    total = sum(d.count("H") for d in seq[:depth])
    assert info.frames[-1] == (2 * total) % 2 == 0


def test_hw_dem_works_with_exact_mld_and_brute_force():
    lc = H.build_hw_lrb_circuit(SEQS[1], 1, H.IQM_GARNET)
    dem = lc.circuit.detector_error_model()
    assert dem.num_detectors == 10
    decoder = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    joint = M.brute_force_joint(dem)
    n = dem.num_detectors
    dets, _ = lc.circuit.compile_detector_sampler(seed=5).sample(
        200, separate_observables=True)
    idx = (dets.astype(np.int64) * (1 << np.arange(n))).sum(axis=1)
    expected = np.log(joint[idx]) - np.log(joint[idx + (1 << n)])
    assert np.allclose(decoder.log_likelihood_ratio(dets), expected,
                       atol=1e-9)
    # no-reset variant (Pauli-frame feedback) is also MLD-compatible
    lc2 = H.build_hw_lrb_circuit(SEQS[1], 6, H.IQM_GARNET, reset=False)
    M.TransferMatrixMLD(lc2.circuit.detector_error_model(),
                        M.lrb_detector_slots(lc2))


def test_run_hw_point_smoke():
    res = H.run_hw_point(H.DEVICES["ion_helios"], SEQS[:1], [0, 2], 2000,
                         const_checks=(0, 1), unif_checks=(1,))
    for name in ("mld5", "mld2", "const0", "const0_mld", "lrbd_unif_1"):
        assert res.tables[name].shots.sum() == 4000
    sparse = H.run_hw_point(H.IQM_GARNET, SEQS[:1], [4], 1000, check_every=2,
                            const_checks=(1,), unif_checks=())
    assert "lrbd_all" in sparse.tables


def test_decay_fit_recovers_known_exponential():
    from lrb_qubit import baseline as B
    rng = np.random.default_rng(1)
    depths = np.array([0, 2, 4, 6, 10, 14])
    b, f = 0.95, 0.97
    table = B.CountTable.zeros(30, len(depths))
    for s in range(30):
        f_s = f + 0.003 * rng.standard_normal()  # sequence-to-sequence spread
        for d, m in enumerate(depths):
            p_wrong = (1 - b * f_s ** m) / 2
            table.add(s, d, 20000, 20000, int(rng.binomial(20000, p_wrong)))
    fit = H.decay_fit(table, depths, n_boot=200)
    assert abs(fit["f"] - f) < 4 * fit["f_err"] + 1e-3
    assert abs(fit["b"] - b) < 4 * fit["b_err"] + 5e-3
    assert 0 < fit["f_err"] < 0.005


# --------------------------------------------------------------- device model

def _twirl_numeric(t, t1, t2):
    """Pauli twirl of amplitude damping (gamma) followed by pure dephasing."""
    gamma = 1 - math.exp(-t / t1)
    # total coherence decay exp(-t/T2) = sqrt(1-gamma) * (1 - 2 q)
    q = (1 - math.exp(-t / t2) / math.sqrt(1 - gamma)) / 2
    k_ad = [np.array([[1, 0], [0, math.sqrt(1 - gamma)]]),
            np.array([[0, math.sqrt(gamma)], [0, 0]])]
    k_ph = [math.sqrt(1 - q) * np.eye(2), math.sqrt(q) * np.diag([1, -1])]
    kraus = [b @ a for a in k_ad for b in k_ph]
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]),
              np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    # chi-matrix diagonal: p_P = sum_k |tr(P K)/2|^2
    return [sum(abs(np.trace(P.conj().T @ K) / 2) ** 2 for K in kraus)
            for P in paulis]


@pytest.mark.parametrize("t,t1,t2", [(1000, 40, 21), (500, 100, 100),
                                     (40, 30, 10)])
def test_pauli_twirl_matches_kraus(t, t1, t2):
    pi, px, py, pz = _twirl_numeric(t / 1000, t1, t2)
    ex, ey, ez = H.pauli_twirl_idle(t, t1, t2)
    assert (px, py, pz) == pytest.approx((ex, ey, ez), abs=1e-12)
    assert pi == pytest.approx(1 - ex - ey - ez, abs=1e-12)


def test_depolarizing_from_average_infidelity():
    rng = np.random.default_rng(0)
    for n, paulis in ((1, ["X", "Y", "Z"]), (2, ["XI", "ZZ", "YX"])):
        d = 2 ** n
        psi = rng.normal(size=(4000, d)) + 1j * rng.normal(size=(4000, d))
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
        for label in paulis:
            P = stim.PauliString(label).to_unitary_matrix(endian="little")
            overlap = np.abs(np.einsum("ni,ij,nj->n", psi.conj(), P, psi)) ** 2
            assert overlap.mean() == pytest.approx(1 / (d + 1), abs=0.02)
        r = 0.01
        p = H.depolarizing_from_infidelity(r, n)
        assert p * d / (d + 1) == pytest.approx(r)


def test_si1000_equivalent_recovers_si1000():
    p = 2e-3
    dev = H.Device(name="si", r_1q=0.1 * p * 2 / 3, r_2q=p * 4 / 5,
                   e_meas=5 * p, e_reset=2 * p,
                   extra_idle=(("meas", 2 * p), ("reset", 2 * p)))
    eq = H.si1000_equivalent(dev)
    assert eq["p"] == pytest.approx(p)
    f = eq["factors"]
    assert f["one_qubit"] == pytest.approx(0.1)
    assert f["meas_idle"] == pytest.approx(2.0)
    assert f["meas_flip"] == pytest.approx(5.0)
    assert f["reset_idle"] == pytest.approx(2.0)
    assert f["reset_flip"] == pytest.approx(2.0)
    garnet = H.si1000_equivalent(H.IQM_GARNET)
    assert garnet["p"] == pytest.approx(6.25e-3)
    assert garnet["absolute"]["meas_idle"] == pytest.approx(
        sum(H.pauli_twirl_idle(1000, 40, 21.05)))


# --------------------------------------------------------------- DEM learning

def _small_tables():
    lc = H.build_hw_lrb_circuit(SEQS[2], 2, H.IQM_GARNET)
    dem = lc.circuit.detector_error_model()
    return lc, dem, D.footprint_table(dem)


def test_hyperedge_estimator_is_exact_with_exact_moments():
    _, dem, table = _small_tables()
    assert max(len(fp) for fp in table) >= 4  # genuine hyperedges present
    est = D.estimate_footprints(D.exact_moment_fn(table),
                                [fp for fp in table if fp])
    for fp, p in est.items():
        assert p == pytest.approx(D.total_probability(table[fp]), abs=1e-12)


def test_pij_formula_forms_agree_and_match_truth():
    lc, dem, table = _small_tables()
    pairs = sorted({(fp[a], fp[b]) for fp in table if len(fp) > 1
                    for a in range(len(fp)) for b in range(a + 1, len(fp))})
    exact = D.pij_from_moments(D.exact_moment_fn(table), pairs)
    truth = D.true_pij(table, pairs)
    assert max(abs(exact[k] - truth[k]) for k in pairs) < 1e-12
    dets, _ = lc.circuit.compile_detector_sampler(seed=9).sample(
        100_000, separate_observables=True)
    classic = D.pij_classic(dets, pairs)
    moment = D.pij_from_moments(D.ParityMoments(dets), pairs)
    assert max(abs(classic[k] - moment[k]) for k in pairs) < 1e-9
    big = [k for k in pairs if truth[k] > 2e-3]
    rel = np.median([abs(classic[k] - truth[k]) / truth[k] for k in big])
    assert rel < 0.15


def test_ambiguous_footprints_exist_and_split_is_invisible():
    _, dem, table = _small_tables()
    # d = 2: many footprints are caused both by faults that flip X_L and by
    # faults that do not (e.g. Z0 vs Z2 at the same location)
    ambiguous = [fp for fp, (p0, p1) in table.items() if p0 > 0 and p1 > 0]
    assert len(ambiguous) >= 10
    # moving probability between the two classes leaves every syndrome
    # moment unchanged when the footprint total is kept fixed
    swapped = {fp: [e[1], e[0]] for fp, e in table.items()}
    m1, m2 = D.exact_moment_fn(table), D.exact_moment_fn(swapped)
    subsets = [fp for fp in ambiguous[:20]] + [fp[:2] for fp in ambiguous[:20]
                                               if len(fp) > 1]
    assert all(m1(s) == pytest.approx(m2(s), abs=1e-15) for s in subsets)
    # ... while the decoder's posterior does change
    lc, _, _ = _small_tables()
    n = dem.num_detectors
    slots = M.lrb_detector_slots(lc)
    dets, _ = lc.circuit.compile_detector_sampler(seed=8).sample(
        2000, separate_observables=True)
    a = M.TransferMatrixMLD(D.dem_from_table(table, n), slots)
    b = M.TransferMatrixMLD(D.dem_from_table(swapped, n), slots)
    assert np.mean((a.log_likelihood_ratio(dets) < 0)
                   != (b.log_likelihood_ratio(dets) < 0)) > 0.01


def test_learned_dem_decodes_like_oracle():
    dev = H.DEVICES["iqm_t100"]
    row = D.study_circuit((dev, dev, SEQS[0], 1, 0, [100_000], 40_000, 21,
                           (0.5,)))
    learned = row["decoders"]["learned_100000"]
    assert abs(learned["diff_vs_oracle"]) < 4 * learned["diff_se"] + 2e-3
    assert row["missing_footprints"] == 0
