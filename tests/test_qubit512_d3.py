"""Tests for the [[13,1,3]] fold-transversal code and its LRB circuits (WP G).

Run with ``module load python/3.11 && python -m pytest tests/test_qubit512_d3.py``
(export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 first).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")
pytest.importorskip("ldpc")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import d3_circuits as G  # noqa: E402
from lrb_qubit import d3_code as D  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402

CODE = D.CODE13
SEQUENCE = Q.random_clifford_sequence(22, np.random.default_rng(11))
GRID_SEQUENCES = E.clifford_sequences(3, 22, 20261028)
S_AND_H = ["P", "H", "HP", "PH", "HPH", "P", "XZ", "H"]


def _is_logical_class(code, pauli, name, sign=+1):
    return code.logical_class(pauli) == (name, sign)


# ----------------------------------------------------------------------
# The code
# ----------------------------------------------------------------------
def test_parameters_and_commutation():
    assert CODE.num_data == 13 and CODE.num_stabilizers == 12
    assert CODE.num_qubits == 25
    stabs = [CODE.stabilizer_pauli(s) for s in range(12)]
    for a, b in itertools.combinations(stabs, 2):
        assert a.commutes(b)
    x_l, z_l = CODE.logical_pauli("X"), CODE.logical_pauli("Z")
    assert all(s.commutes(x_l) and s.commutes(z_l) for s in stabs)
    assert not x_l.commutes(z_l)
    matrix = np.array([np.concatenate(s.to_numpy()) for s in stabs], float)
    assert np.linalg.matrix_rank(matrix) == 12  # 13 - 12 = 1 logical qubit
    assert x_l.weight == 3 and z_l.weight == 3


def test_code_distance_is_three():
    """No Pauli of weight 1 or 2 is a nontrivial logical; X_L has weight 3."""
    stabs = [CODE.stabilizer_pauli(s) for s in range(12)]
    for weight in (1, 2):
        for qubits in itertools.combinations(range(13), weight):
            for paulis in itertools.product("XYZ", repeat=weight):
                p = stim.PauliString(13)
                for q, ch in zip(qubits, paulis):
                    p[q] = ch
                if all(p.commutes(s) for s in stabs):
                    assert CODE.logical_class(p)[0] == "I"


def test_diagonal_fold_symmetry():
    """The transpose maps data to data and Z checks onto X checks."""
    for q in range(CODE.num_data):
        assert CODE.transpose(CODE.transpose(q)) == q
    z_supports = {frozenset(CODE.check_supports[s]) for s in CODE.z_stabilizers}
    x_supports = {frozenset(CODE.check_supports[s]) for s in CODE.x_stabilizers}
    mirrored = {frozenset(CODE.transpose(q) for q in sup) for sup in z_supports}
    assert mirrored == x_supports
    assert {CODE.transpose(q) for q in CODE.logical_x_support} == set(
        CODE.logical_z_support)
    fixed = [q for q in range(13) if CODE.transpose(q) == q]
    assert sorted(fixed) == sorted(CODE.diagonal) and len(fixed) == 5
    paired = sorted(q for pair in CODE.fold_pairs for q in pair)
    assert paired == sorted(set(range(13)) - set(fixed))


@pytest.mark.parametrize("code", [D.CODE13, D.UnrotatedSurfaceCode(4)])
@pytest.mark.parametrize("token,x_image,z_image", [
    ("H", ("Z", 1), ("X", 1)), ("P", ("Y", 1), ("Z", 1)),
    ("X", ("X", 1), ("Z", -1)), ("Z", ("X", -1), ("Z", 1)),
    ("I", ("X", 1), ("Z", 1)), ("H_INV", ("Z", 1), ("X", 1)),
    ("P_INV", ("Y", -1), ("Z", 1))])
def test_fold_transversal_logical_gates(code, token, x_image, z_image):
    """Stabilizers map to stabilizers with + signs; logicals as expected.

    H = fold SWAPs then transversal H; S = S / S-dagger on the fold line
    plus CZ across mirror pairs. Checked with Stim tableaus (also at d=4).
    """
    tableau = code.data_tableau(code.logical_gate_operations(token))
    for s in range(code.num_stabilizers):
        image = tableau(code.stabilizer_pauli(s))
        assert code.logical_class(image) == ("I", 1)
    assert code.logical_class(tableau(code.logical_pauli("X"))) == x_image
    assert code.logical_class(tableau(code.logical_pauli("Z"))) == z_image


def test_fold_gate_structure():
    """The gate lists are fold-transversal: SWAP/CZ only on mirror pairs."""
    h_ops = CODE.logical_gate_operations("H")
    s_ops = CODE.logical_gate_operations("P")
    pairs = set(CODE.fold_pairs)
    assert {t for g, t in h_ops if g == "SWAP"} == pairs
    assert sorted(t[0] for g, t in h_ops if g == "H") == list(range(13))
    assert {t for g, t in s_ops if g == "CZ"} == pairs
    diag = {t[0]: g for g, t in s_ops if g in ("S", "S_DAG")}
    assert set(diag) == set(CODE.diagonal)
    for i in range(5):
        expected = "S" if i % 2 == 0 else "S_DAG"
        assert diag[CODE.labels[(i, i)]] == expected


def test_all_24_cliffords_act_correctly_and_invert():
    """Each descriptor acts on (X_L, Z_L) like the single-qubit Clifford."""
    names = {"I": "I", "H": "H", "P": "S", "X": "X", "Z": "Z"}
    for descriptor in C.SINGLE_QUBIT_CLIFFORDS:
        ops = CODE.logical_clifford_operations(descriptor)
        inv = CODE.logical_clifford_operations(descriptor, inverse=True)
        assert CODE.data_tableau(ops + inv) == stim.Tableau(13)
        single = stim.Circuit()
        single.append("I", [0])
        for token in C.clifford_gate_sequence(descriptor):
            single.append(names[token], [0])
        t1 = stim.Tableau.from_circuit(single)
        tableau = CODE.data_tableau(ops)
        for which in ("X", "Z"):
            expected = t1(stim.PauliString(which))
            sign = int(expected.sign.real)
            label = str(expected)[1:]
            assert CODE.logical_class(tableau(CODE.logical_pauli(which))) == (
                label, sign)


def test_encoder_prepares_plus_logical():
    sim = stim.TableauSimulator()
    for gate, targets in CODE.encoding_operations():
        sim.do(stim.Circuit(f"{gate} {' '.join(map(str, targets))}"))
    for s in range(12):
        assert sim.peek_observable_expectation(CODE.stabilizer_pauli(s)) == 1
    assert sim.peek_observable_expectation(CODE.logical_pauli("X")) == 1


def test_distance_two_member_is_the_lead_code():
    """d=2 with the lead's labels gives the lead's [[5,1,2]] exactly."""
    five = D.CODE5
    assert five.stabilizer_paulis == C.STABILIZER_PAULIS
    assert five.logical_x == C.LOGICAL_X and five.logical_z == C.LOGICAL_Z
    lead = G.LeadFiveQubitCode()
    for s in range(4):
        assert five.check_operations(s) == lead.check_operations(s)
    for token in ("P", "X", "Z", "I", "P_INV"):
        assert (five.logical_gate_operations(token)
                == C.logical_gate_operations(token))
    for token in ("H", "H_INV"):  # same gates, different order of the Hs
        assert (five.data_tableau(five.logical_gate_operations(token))
                == C.data_tableau(C.logical_gate_operations(token)))


# ----------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------
@pytest.mark.parametrize("depth", [0, 1, 3, 22])
def test_generic_builder_reproduces_lead_circuit(depth):
    lead = Q.build_lrb_circuit(SEQUENCE, depth, 0.003, terminal="data5")
    mine = G.build_lrb_circuit(G.LeadFiveQubitCode(), SEQUENCE, depth, 0.003,
                               schedule="sequential")
    assert str(lead.circuit) == str(mine.circuit)
    assert lead.rounds == mine.rounds and lead.back == mine.back


@pytest.mark.parametrize("schedule", G.SCHEDULES)
@pytest.mark.parametrize("depth", [0, 1, 4, 22])
def test_noiseless_d3_circuits_are_deterministic(schedule, depth):
    lc = G.build_lrb_circuit(CODE, SEQUENCE, depth, 0.0, schedule=schedule,
                             noise=None)
    dets, obs = lc.circuit.compile_detector_sampler(seed=2).sample(
        64, separate_observables=True)
    assert not dets.any() and not obs.any()
    noisy = G.build_lrb_circuit(CODE, SEQUENCE, depth, 0.01, schedule=schedule)
    assert noisy.circuit.num_detectors == 12 * (depth + 1) + 6
    noisy.circuit.detector_error_model()  # all detectors deterministic


def _idle_layers_on(circuit: stim.Circuit, qubit: int, prob: float) -> int:
    count = 0
    for inst in circuit.flattened():
        if (inst.name == "DEPOLARIZE1" and abs(inst.gate_args_copy()[0] - prob)
                < 1e-15 and any(t.value == qubit for t in inst.targets_copy())):
            count += 1
    return count


@pytest.mark.parametrize("schedule,layers", [("sequential", 12 + 1),
                                             ("parallel", 1 + 1)])
def test_measurement_layers_per_round(schedule, layers):
    """Idle DEPOLARIZE1(2p) layers on a data qubit per forward round."""
    p = 1e-3
    one = G.build_lrb_circuit(CODE, ["I"] * 5, 1, p, schedule=schedule)
    two = G.build_lrb_circuit(CODE, ["I"] * 5, 2, p, schedule=schedule)
    per_round = (_idle_layers_on(two.circuit, 6, 2 * p)
                 - _idle_layers_on(one.circuit, 6, 2 * p))
    assert per_round == layers


@pytest.mark.parametrize("schedule", G.SCHEDULES)
@pytest.mark.parametrize("seq", [SEQUENCE, S_AND_H])
def test_circuit_level_distance_is_three(schedule, seq):
    lc = G.build_lrb_circuit(CODE, seq, 3, 0.001, schedule=schedule)
    hyper = lc.circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False)
    assert len(hyper) == 3
    graph = lc.circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    assert len(graph) == 3


@pytest.mark.parametrize("schedule", G.SCHEDULES)
@pytest.mark.parametrize("depth", [0, 2, 5])
def test_raw_syndromes_match_measurement_records(schedule, depth):
    lc = G.build_lrb_circuit(CODE, SEQUENCE, depth, 0.02, schedule=schedule)
    circuit = lc.circuit
    measurements = circuit.compile_sampler(seed=5).sample(2000)
    flips = measurements ^ circuit.reference_sample()[None, :]
    dets, _ = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)
    raw = lc.raw_syndromes(dets)
    # Records in time order: per round, one record per stabilizer, in the
    # schedule's measurement order (Z checks then X checks).
    order = list(CODE.z_stabilizers) + list(CODE.x_stabilizers)
    for k in range(lc.num_rounds):
        for j, s in enumerate(order):
            assert np.array_equal(raw[:, k, s], flips[:, 12 * k + j])


def test_policies_use_lead_semantics():
    lc = G.build_lrb_circuit(CODE, SEQUENCE, 6, 0.004, schedule="parallel")
    dets, _ = lc.circuit.compile_detector_sampler(seed=9).sample(
        3000, separate_observables=True)
    raw = lc.raw_syndromes(dets)
    for policy, k in (("unif", 1), ("unif", 2), ("const", 1), ("const", 3)):
        rounds = B.used_rounds(policy, k, 6)
        acc = B.accept_mask(raw, lc.rounds, rounds)
        manual = np.ones(len(raw), bool)
        for r in rounds:
            manual &= ~raw[:, lc.rounds.index(r), :].any(axis=1)
        assert np.array_equal(acc, manual)


# ----------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------
@pytest.mark.parametrize("schedule", G.SCHEDULES)
def test_bposd_corrects_every_single_fault(schedule):
    """At small p every single fault is the most likely explanation of its
    own syndrome, so a distance-preserving decoder must correct all of them.
    (At p = 1e-3 the sequential schedule's idle noise makes some pairs of
    common faults likelier than one rare CZ fault, so exact MLD would also
    'fail' those; hence p = 1e-4 here.)"""
    lc = G.build_lrb_circuit(CODE, S_AND_H, 3, 1e-4, schedule=schedule)
    report = G.single_fault_report(lc.circuit, G.BpOsdDEMDecoder(lc.circuit))
    assert report["num_failing"] == 0


def test_plain_bposd_can_converge_to_absurd_solutions():
    """Why BpOsdDEMDecoder adds fallbacks: with OSD skipped after BP
    'converges', plain ldpc BP-OSD misdecodes some single faults."""
    from ldpc import BpOsdDecoder
    lc = G.build_lrb_circuit(CODE, S_AND_H, 3, 1e-5, schedule="sequential")
    H, L, priors = G.dem_matrices(lc.circuit.detector_error_model())
    keep = np.asarray(H.sum(axis=0)).ravel() > 0
    H, L, priors = H[:, keep], L[keep], priors[keep]
    plain = BpOsdDecoder(H, error_channel=list(priors), max_iter=30,
                         bp_method="product_sum", osd_method="osd_cs",
                         osd_order=10)
    wrong = sum(int(L @ plain.decode(col)) % 2 != L[j]
                for j, col in enumerate(H.toarray().T.astype(np.uint8)))
    assert wrong > 0


@pytest.mark.parametrize("schedule", G.SCHEDULES)
def test_matching_corrects_single_faults_without_s_gates(schedule):
    lc = G.build_lrb_circuit(CODE, ["X", "Z", "XZ", "I"], 4, 0.001,
                             schedule=schedule)
    report = G.single_fault_report(lc.circuit, G.MatchingDecoder(lc.circuit))
    assert report["num_failing"] == 0


@pytest.mark.parametrize("schedule", G.SCHEDULES)
def test_matching_fails_single_faults_across_fold_transversal_s(schedule):
    """Known limitation (Serra-Peralta et al.): Stim's graphlike pieces of the
    S-gate hyperedges disagree on the observable, so MWPM misdecodes some
    single faults (a first-order logical error)."""
    lc = G.build_lrb_circuit(CODE, ["P", "P"], 2, 0.001, schedule=schedule)
    report = G.single_fault_report(lc.circuit, G.MatchingDecoder(lc.circuit))
    assert report["num_failing"] > 0
    assert report["failing_probability"] > 1e-3  # > p: first order


def test_observable_frames():
    lc = G.build_lrb_circuit(CODE, ["P", "P"], 2, 0.001)
    assert G.observable_frames(CODE, lc) == {0: "Y", 1: "X", 2: "X"}
    lc = G.build_lrb_circuit(CODE, ["H", "H"], 2, 0.001)
    assert G.observable_frames(CODE, lc) == {0: "Z", 1: "X", 2: "X"}


def test_mld_wrapper_matches_lead_decoder():
    from lrb_qubit import mld as M
    lc5 = Q.build_lrb_circuit(SEQUENCE, 4, 0.004, terminal="data5")
    lead = M.TransferMatrixMLD(lc5.circuit.detector_error_model(),
                               M.lrb_detector_slots(lc5))
    mine = G.MLDDecoder(G.build_lrb_circuit(G.LeadFiveQubitCode(), SEQUENCE,
                                            4, 0.004))
    dets, _ = lc5.circuit.compile_detector_sampler(seed=4).sample(
        2000, separate_observables=True)
    assert np.array_equal(lead.log_likelihood_ratio(dets) < 0,
                          mine.predict(dets))


def test_run_sequence_counts_are_consistent():
    result = G.run_sequence(CODE, "parallel", 0.002, GRID_SEQUENCES[0],
                            [0, 2], 2000, decoders=("pm", "bposd"),
                            decode_shots={"bposd": 300}, batch_size=1000)
    unif1 = result.tables["lrbd_unif_1"]
    assert (unif1.accepted <= unif1.shots).all()
    assert (result.tables["bposd"].shots == 300).all()
    assert (result.tables["pm"].shots == 2000).all()
    assert (result.tables["const0"].shots == 2000).all()
    assert (result.tables["const0_mld"].shots == 2000).all()
    # unif=1 uses every round, so it accepts no more than const=1.
    assert (unif1.accepted <= result.tables["lrbd_const_1"].accepted).all()


# ----------------------------------------------------------------------
# Physical RB
# ----------------------------------------------------------------------
def test_exact_physical_rb_matches_sampling():
    p_rb = 0.05
    for depth in (2, 10):
        exact = G.physical_rb_fidelity(SEQUENCE, depth, p_rb)
        circuit = Q.build_physical_rb_circuit(SEQUENCE, depth, p_rb)
        _, obs = circuit.compile_detector_sampler(seed=1).sample(
            200_000, separate_observables=True)
        sampled = 1 - 2 * obs.mean()
        assert abs(sampled - exact) < 5 * np.sqrt(1 / 200_000)


def test_physical_rb_ensemble_decay():
    """Ensemble decay equals the average of exact per-sequence fidelities."""
    p_rb = 0.02
    rng = np.random.default_rng(3)
    seqs = [Q.random_clifford_sequence(1, rng) for _ in range(20000)]
    mean_f = np.mean([G.physical_rb_fidelity(s, 1, p_rb) for s in seqs])
    assert abs(mean_f - G.physical_rb_decay(p_rb)) < 2e-4
    # Mean gate count per Clifford is 61/24.
    lengths = [G.clifford_gate_count(c) for c in C.SINGLE_QUBIT_CLIFFORDS]
    assert sum(lengths) == 61


def test_run_decoder_depth_respects_shots_and_budget():
    full = G.run_decoder_depth(CODE, "parallel", 0.002, GRID_SEQUENCES[1], 2,
                               60, decoder="bposd", seed=3, chunk=20)
    assert full["shots"] == 60 and 0 <= full["errors"] <= 60
    cut = G.run_decoder_depth(CODE, "parallel", 0.002, GRID_SEQUENCES[1], 2,
                              10_000, decoder="pm", seed=3, chunk=20,
                              time_budget=0.0)
    assert cut["shots"] == 0


def test_bruteforce_mld_matches_transfer_matrix_on_const0():
    from lrb_qubit import mld as M
    lead = Q.build_lrb_const0_circuit(SEQUENCE, 6, 0.01)
    tm = M.TransferMatrixMLD(lead.circuit.detector_error_model(),
                             M.lrb_detector_slots(lead))
    dets, _ = lead.circuit.compile_detector_sampler(seed=2).sample(
        4000, separate_observables=True)
    brute = G.BruteForceMLD(lead.circuit)
    llr = tm.log_likelihood_ratio(dets)
    clear = np.abs(llr) > 1e-9  # ties may break either way
    assert np.array_equal((llr < 0)[clear], brute.predict(dets)[clear])
    # d = 3: six final detectors, exact by enumeration.
    lc0 = G.build_lrb_const0_circuit(CODE, SEQUENCE, 6, 0.002)
    assert lc0.circuit.num_detectors == 6
    G.BruteForceMLD(lc0.circuit)


def test_physical_rb_scales():
    """p_RB = p and p_RB = 5p: error per step is linear in p_RB."""
    r1 = (1 - G.physical_rb_decay(1e-3)) / 2
    r5 = (1 - G.physical_rb_decay(5e-3)) / 2
    assert abs(r5 / r1 - 5) < 0.01
    # (4/3) q (2 * 61/24) / 2 with q = 0.1 p_RB.
    assert abs(r1 - (4 / 3) * 1e-4 * 61 / 24) / r1 < 1e-3
