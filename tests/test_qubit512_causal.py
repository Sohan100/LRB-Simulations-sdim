"""Tests for the causal decoders and Pauli-frame tools (work package D).

Run with
``module load python/3.11; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_qubit512_causal.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import decoders_causal as D  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQUENCES = E.clifford_sequences(3, 22, 20261028)
SEQUENCE = SEQUENCES[0]


def _sample(lc, shots, seed):
    return lc.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)


# ---------------------------------------------------------------------------
# Brute-force reference for one commit posterior
# ---------------------------------------------------------------------------

def _brute_force_commit_joint(dem, slots, t, end, mode):
    """Exact joint over a register vector for the commit of slot ``t``.

    Registers: every observed detector bit (slots ``0..end``), the boundary
    register ``b_t`` (slot-``t`` bits of mechanisms whose latest slot is
    ``t``), the history registers and the cumulative logical ``l_t``. History
    in mode ``"state"``: ``b_{t-1}`` and ``l_{t-1}``; in mode ``"logical"``:
    ``l_0..l_{t-1}``. Mechanisms whose latest slot is ``end + 1`` only
    contribute their slot-``end`` bits (the next slot is unobserved); later
    mechanisms are irrelevant. Returns ``(joint, layout)``.
    """
    slots = np.asarray(slots)
    num_slots = int(slots.max()) + 1
    sizes = [int((slots == u).sum()) for u in range(num_slots)]
    position = np.zeros(len(slots), dtype=np.int64)
    for u in range(num_slots):
        position[slots == u] = np.arange(sizes[u])
    obs_offset = {}
    bit = 0
    for u in range(end + 1):
        obs_offset[u] = bit
        bit += sizes[u]
    b_off = bit
    bit += sizes[t]
    bprev_off = None
    hist_l_bits = {}
    if mode == "state" and t > 0:
        bprev_off = bit
        bit += sizes[t - 1]
        hist_l_bits[t - 1] = bit
        bit += 1
    elif mode == "logical":
        for j in range(t):
            hist_l_bits[j] = bit
            bit += 1
    # mode "none": no history registers (uncommitted fixed-lag marginal).
    lt_bit = bit
    bit += 1
    nbits = bit

    vectors = {}
    for mech in M.dem_mechanisms(dem):
        if not mech.detectors:
            if not mech.flips_observable:
                continue
            u = 0
        else:
            u = int(slots[list(mech.detectors)].max())
        if u > end + 1:
            continue
        v = 0
        for det in mech.detectors:
            su = int(slots[det])
            if su <= end:
                v ^= 1 << (obs_offset[su] + int(position[det]))
            if u == t and su == t:
                v ^= 1 << (b_off + int(position[det]))
            if bprev_off is not None and u == t - 1 and su == t - 1:
                v ^= 1 << (bprev_off + int(position[det]))
        if mech.flips_observable:
            for j, b in hist_l_bits.items():
                if u <= j:
                    v ^= 1 << b
            if u <= t:
                v ^= 1 << lt_bit
        if v == 0:
            continue
        q = vectors.get(v, 0.0)
        vectors[v] = q + mech.probability - 2 * q * mech.probability

    joint = np.zeros(2 ** nbits)
    joint[0] = 1.0
    index = np.arange(2 ** nbits)
    for v, prob in vectors.items():
        joint = (1 - prob) * joint + prob * joint[index ^ v]
    layout = dict(sizes=sizes, obs_offset=obs_offset, b_off=b_off,
                  bprev_off=bprev_off, hist_l_bits=hist_l_bits,
                  lt_bit=lt_bit, t=t, end=end)
    return joint, layout


def _brute_force_posterior(joint, layout, values, l_hist, b_hist):
    """Normalised ``P(l_t, b_t | observed slots, committed history)``."""
    t = layout["t"]
    fixed = 0
    for u, off in layout["obs_offset"].items():
        fixed |= int(values[u]) << off
    if layout["bprev_off"] is not None:
        fixed |= int(b_hist[t - 1]) << layout["bprev_off"]
    for j, b in layout["hist_l_bits"].items():
        fixed |= int(l_hist[j]) << b
    # With no history registers l_hist / b_hist are unused (may be None).
    post = np.zeros((2, 2 ** layout["sizes"][t]))
    for lt in range(2):
        for bt in range(2 ** layout["sizes"][t]):
            post[lt, bt] = joint[fixed | (bt << layout["b_off"])
                                 | (lt << layout["lt_bit"])]
    return post / post.sum()


@pytest.mark.parametrize("terminal,depth,mode,lookahead", [
    ("data2", 1, "state", 0), ("data2", 1, "state", 1),
    ("data5", 1, "state", 0), ("data5", 1, "state", 1),
    ("data5", 1, "logical", 0), ("data5", 1, "logical", 1),
    ("data5", 2, "state", 0), ("data5", 2, "logical", 1)])
def test_commit_posteriors_match_brute_force(terminal, depth, mode, lookahead):
    lc = Q.build_lrb_circuit(SEQUENCE, depth, 0.03, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    dec = D.CausalWindowMLD(dem, slots, lookahead, mode)
    dets, _ = _sample(lc, 400, seed=21)
    values = dec.slot_values(dets)
    # Prefer shots with several detection events.
    order = np.argsort(-dets.sum(axis=1), kind="stable")[:3]
    stream = dec.stream(len(order), record=True, keep_posteriors=True)
    for t in range(dec.num_slots):
        stream.push(values[order, t])
    last = dec.num_slots - 1
    for t in range(dec.num_slots):
        if mode == "logical" and t > 2:
            continue
        end = min(t + lookahead, last)
        joint, layout = _brute_force_commit_joint(dem, slots, t, end, mode)
        for i, shot in enumerate(order):
            expected = _brute_force_posterior(joint, layout, values[shot],
                                              stream.l_hist[i],
                                              stream.b_hist[i])
            got = stream.posteriors[t][i]
            assert np.allclose(got, expected, atol=1e-9), (shot, t)


@pytest.mark.parametrize("terminal,depth,lookahead", [
    ("data5", 1, 0), ("data5", 1, 1), ("data5", 2, 1), ("data2", 2, 0)])
def test_lagged_marginals_match_brute_force(terminal, depth, lookahead):
    lc = Q.build_lrb_circuit(SEQUENCE, depth, 0.03, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    dec = D.CausalWindowMLD(dem, slots, lookahead, "state")
    dets, _ = _sample(lc, 300, seed=13)
    values = dec.slot_values(dets)
    order = np.argsort(-dets.sum(axis=1), kind="stable")[:3]
    marg = dec.lagged_marginals(dets[order], lookahead)
    last = dec.num_slots - 1
    for t in range(dec.num_slots):
        end = min(t + lookahead, last)
        joint, layout = _brute_force_commit_joint(dem, slots, t, end, "none")
        for i, shot in enumerate(order):
            expected = _brute_force_posterior(joint, layout, values[shot],
                                              None, None)
            assert np.allclose(marg[t][i], expected, atol=1e-9), (shot, t)


def test_smoothing_last_slot_is_offline_mld_and_truth_is_consistent():
    lc = Q.build_lrb_circuit(SEQUENCE, 6, 0.01, terminal="data5")
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    dets, obs, errs = dem.compile_sampler(seed=5).sample(
        3000, return_errors=True)
    # Error columns line up with mld.dem_mechanisms.
    mechs = M.dem_mechanisms(dem)
    rebuilt = np.zeros_like(dets)
    flip = np.zeros(len(dets), dtype=bool)
    for k, mech in enumerate(mechs):
        for det in mech.detectors:
            rebuilt[:, det] ^= errs[:, k]
        if mech.flips_observable:
            flip ^= errs[:, k]
    assert np.array_equal(rebuilt, dets) and np.array_equal(flip, obs[:, 0])
    b, l = D.true_boundary_states(dem, slots, errs)
    assert np.array_equal(l[:, -1].astype(bool), obs[:, 0])
    dec = D.CausalWindowMLD(dem, slots, None, "state")
    values = dec.slot_values(dets)
    assert np.array_equal(b[:, -1], values[:, -1])   # last slot fully explained
    smooth = dec.lagged_marginals(dets, None)
    mld = M.TransferMatrixMLD(dem, slots).log_likelihood_ratio(dets) < 0
    last = smooth[-1].sum(axis=2)
    assert np.array_equal(last[:, 1] > last[:, 0], mld)


# ---------------------------------------------------------------------------
# Decoder consistency and causality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("terminal", ["data2", "data5"])
@pytest.mark.parametrize("depth", [0, 3, 12])
def test_kernel_extraction_reproduces_lead_mld(terminal, depth):
    lc = Q.build_lrb_circuit(SEQUENCE, depth, 0.005, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    base = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    dec = D.CausalWindowMLD(None, None, 0, base=base)
    dets, _ = _sample(lc, 3000, seed=5)
    if dets.shape[1] == 0:
        pytest.skip("no detectors")
    a = base.log_likelihood_ratio(dets)
    b = dec.filter_llr(dets)
    finite = np.isfinite(a)
    assert np.array_equal(finite, np.isfinite(b))
    assert np.allclose(a[finite], b[finite], atol=1e-9)


@pytest.mark.parametrize("mode", ["state", "logical"])
@pytest.mark.parametrize("lookahead", [0, 1, 2, 4])
def test_commits_never_use_later_slots(mode, lookahead):
    lc = Q.build_lrb_circuit(SEQUENCE, 8, 0.01, terminal="data5")
    dem = lc.circuit.detector_error_model()
    dec = D.CausalWindowMLD(dem, M.lrb_detector_slots(lc), lookahead, mode)
    dets, _ = _sample(lc, 2000, seed=3)
    _, l_ref, b_ref = dec.decode(dets, return_history=True)
    rng = np.random.default_rng(1)
    for t in (0, 3, 6):
        horizon = t + lookahead
        tampered = dets.copy()
        later = np.flatnonzero(M.lrb_detector_slots(lc) > horizon)
        tampered[:, later] = rng.random((len(dets), len(later))) < 0.5
        _, l_new, b_new = dec.decode(tampered, return_history=True)
        assert np.array_equal(l_new[:, :t + 1], l_ref[:, :t + 1])
        assert np.array_equal(b_new[:, :t + 1], b_ref[:, :t + 1])


def test_streaming_commits_as_soon_as_window_is_complete():
    lc = Q.build_lrb_circuit(SEQUENCE, 6, 0.01, terminal="data5")
    dec = D.CausalWindowMLD(lc.circuit.detector_error_model(),
                            M.lrb_detector_slots(lc), 2)
    dets, _ = _sample(lc, 500, seed=8)
    values = dec.slot_values(dets)
    stream = dec.stream(len(dets))
    committed = []
    for t in range(dec.num_slots):
        newly = stream.push(values[:, t])
        committed.append(newly)
        if t >= 2 and t < dec.num_slots - 1:
            assert newly == [t - 2]
    assert committed[-1] == list(range(dec.num_slots - 3, dec.num_slots))
    assert np.array_equal(stream.prediction, dec.decode(dets))


@pytest.mark.parametrize("mode", ["state", "logical"])
def test_full_lookahead_equals_unbounded_window(mode):
    lc = Q.build_lrb_circuit(SEQUENCE, 5, 0.01, terminal="data5")
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    full = D.CausalWindowMLD(dem, slots, None, mode)
    wide = D.CausalWindowMLD(dem, slots, 50, mode)
    dets, _ = _sample(lc, 3000, seed=4)
    assert np.array_equal(full.decode(dets), wide.decode(dets))


def test_causal_decoders_on_noiseless_and_single_fault_inputs():
    lc = Q.build_lrb_circuit(SEQUENCE, 6, 0.002, terminal="data5")
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    zero = np.zeros((4, dem.num_detectors), dtype=bool)
    for w in (0, 1, 2, 4, None):
        for mode in ("state", "logical"):
            dec = D.CausalWindowMLD(dem, slots, w, mode)
            assert not dec.decode(zero).any()
            assert dec.dead_shots == 0


def test_tie_breaking_is_explicit_and_roundoff_immune():
    rng = np.random.default_rng(0)
    base = rng.random((6, 2, 16)) * 0.5
    base[:, 0, 3] = 1.0
    base[:, 1, 7] = 1.0                      # exact tie between l=0 and l=1
    prev = np.array([0, 1, 0, 1, 0, 1])
    for eps in (0.0, 2e-16, -2e-16, 5e-13):
        post = base.copy()
        post[:, 1, 7] *= 1.0 + eps           # round-off-sized perturbation
        for rule, expected in (("l0", 0 * prev), ("l1", 0 * prev + 1),
                               ("keep", prev), ("flip", 1 - prev)):
            l, b, tie, dead = D.choose_state(post, prev, rule, D.TIE_TOL,
                                             np.random.default_rng(1))
            assert tie.all() and not dead.any()
            assert np.array_equal(l, expected)
            assert np.array_equal(b, np.where(expected == 0, 3, 7))
            pl = post.sum(axis=2) * 0 + post.max(axis=2)
            l2, tie2, _ = D.choose_logical(pl, prev, rule, D.TIE_TOL,
                                           np.random.default_rng(1))
            assert tie2.all() and np.array_equal(l2, expected)
    # A real gap (relative 1e-6) is never a tie.
    post = base.copy()
    post[:, 1, 7] *= 1.0 + 1e-6
    for rule in D.TIE_RULES:
        l, b, tie, _ = D.choose_state(post, prev, rule, D.TIE_TOL,
                                      np.random.default_rng(2))
        assert not tie.any() and (l == 1).all() and (b == 7).all()


@pytest.mark.parametrize("mode", ["state", "logical"])
def test_tie_rules_differ_only_after_a_tie(mode):
    lc = Q.build_lrb_circuit(SEQUENCE, 8, 0.005, terminal="data5")
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    dets, _ = _sample(lc, 4000, seed=12)
    hist = {}
    for rule in ("l0", "l1"):
        dec = D.CausalWindowMLD(dem, slots, 0, mode, tie_rule=rule)
        values = dec.slot_values(dets)
        st = dec.stream(len(dets), record=True)
        for t in range(dec.num_slots):
            st.push(values[:, t])
        hist[rule] = st
    a, b = hist["l0"], hist["l1"]
    if mode == "state":
        assert a.ties > 0          # exact (b_t, l_t) ties occur in state mode
    differs = (a.l_hist != b.l_hist) | (a.b_hist != b.b_hist)
    for shot in np.flatnonzero(differs.any(axis=1)):
        first = int(np.flatnonzero(differs[shot])[0])
        # The first divergence is a tie commit in both decoders.
        assert a.tie_hist[shot, first] and b.tie_hist[shot, first]
    # Shots without any tie are decoded identically.
    clean = ~a.tie_hist.any(axis=1)
    assert np.array_equal(a.l_hist[clean], b.l_hist[clean])


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def test_fixed_offset_fit_matches_repo_plotter():
    pytest.importorskip("matplotlib")
    try:
        from lrb.lrb_plotting import LRBPlotFitConfig, LRBResultsPlotter
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"repo plotter unavailable: {exc}")
    depths = [0, 2, 4, 6, 10, 14, 18, 20, 22]
    rng = np.random.default_rng(2)
    means = 0.93 * 0.97 ** np.asarray(depths) + rng.normal(0, 0.003, 9)
    errs = 0.01 + 0.002 * rng.random(9)
    plotter = LRBResultsPlotter.__new__(LRBResultsPlotter)
    plotter.fit_config = LRBPlotFitConfig()
    ref = plotter._fit_decay_parameter_fixed_a(depths, list(means),
                                               list(errs))
    ours = D.fit_fixed_offset(depths, means, errs)
    assert ours["f"] == pytest.approx(ref["f"], abs=1e-12)
    assert ours["b"] == pytest.approx(ref["b"], abs=1e-12)


# ---------------------------------------------------------------------------
# Pauli frame versus physical correction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("terminal", ["data2", "data5"])
@pytest.mark.parametrize("depth", [0, 1, 4])
def test_segmented_circuit_equals_lead_circuit(terminal, depth):
    seg = D.build_segmented_lrb(SEQUENCE, depth, 0.004, terminal=terminal)
    assert (D.strip_annotations(seg.concatenated())
            == D.strip_annotations(seg.lc.circuit))
    # Round records point at the ancilla measurements M5, M6, M7, M8.
    meas = [inst for inst in seg.concatenated().flattened()
            if inst.name == "M"]
    order = [t.value for inst in meas for t in inst.targets_copy()]
    for records in seg.round_records:
        assert [order[r] for r in records] == [5, 6, 7, 8]
    for q, r in seg.readout_records.items():
        assert order[r] == q


def test_frame_tracker_matches_stim_propagation():
    rng = np.random.default_rng(0)
    ops = C.logical_clifford_operations("ZHP") + C.logical_clifford_operations(
        "HP", inverse=True)
    mat = D.symplectic_matrix(ops)
    tableau = C.data_tableau(ops)
    tracker = D.FrameTracker(20)
    tracker.bits = rng.integers(0, 2, size=(20, 10)).astype(np.uint8)
    before = tracker.bits.copy()
    tracker.propagate(mat)
    for row_in, row_out in zip(before, tracker.bits):
        pauli = stim.PauliString.from_numpy(xs=row_in[:5].astype(bool),
                                            zs=row_in[5:].astype(bool))
        expected = D.pauli_to_bits(tableau(pauli))
        assert np.array_equal(expected, row_out)


@pytest.mark.parametrize("table", [D.COVARIANT_TABLE, D.NONCOVARIANT_TABLE])
@pytest.mark.parametrize("depth", [0, 2, 6])
@pytest.mark.parametrize("p", [0.002, 0.01])
def test_rule_a_feedback_equals_frame_exactly(depth, p, table):
    seg = D.build_segmented_lrb(SEQUENCE, depth, p, terminal="data5")
    physical, frame = D.rule_a_feedback_circuits(seg, table=table)
    a = D.observable_flip_probability(physical)
    b = D.observable_flip_probability(frame)
    assert a == pytest.approx(b, abs=1e-13)
    # The rule matters: no correction gives a different flip probability.
    plain = seg.concatenated()
    total = plain.num_measurements
    plain.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(seg.readout_records[q] - total)
                  for q in (0, 3)], 0)
    if depth > 0:
        assert abs(D.observable_flip_probability(plain) - a) > 1e-4


def test_rule_a_noisy_feedback_differs_from_frame():
    seg = D.build_segmented_lrb(SEQUENCE, 6, 0.002, terminal="data5")
    _, frame = D.rule_a_feedback_circuits(seg)
    noisy, _ = D.rule_a_feedback_circuits(seg, correction_noise=0.002)
    assert (D.observable_flip_probability(noisy)
            > D.observable_flip_probability(frame) + 1e-4)


@pytest.mark.parametrize("rule_factory", [
    D.LookupEveryRound, D.ConfirmedLookup,
    lambda: D.LookupEveryRound(D.NONCOVARIANT_TABLE),
    lambda: D.CausalMLDLogical(1), lambda: D.CausalMLDLogical(0, "logical")])
def test_physical_and_frame_corrections_agree_shot_by_shot(rule_factory):
    seg = D.build_segmented_lrb(SEQUENCE, 5, 0.01, terminal="data5")
    phys = D.simulate_feedback(seg, rule_factory(), 2000, seed=11,
                               mode="physical")
    frame = D.simulate_feedback(seg, rule_factory(), 2000, seed=11,
                                mode="frame")
    assert np.array_equal(phys.logical_flip, frame.logical_flip)
    for a, b in zip(phys.corrections, frame.corrections):
        assert np.array_equal(a, b)
    # Corrections actually happen.
    assert sum(int(c.any(axis=1).sum()) for c in phys.corrections) > 50


class _TerminalOnlyLookup(D.FeedbackRule):
    def after_round(self, k, raw):
        if k == self.seg.num_rounds - 1:
            return D.lookup_correction(raw)
        return D.lookup_correction(np.zeros_like(raw))


def test_covariant_lookup_forward_corrections_are_logically_inert():
    # The covariant lookup group <X0, Z0, X2, Z3> is mapped to itself by the
    # physical logical gates, so forward-round corrections never change the
    # logical output: rule A equals a terminal-only lookup shot by shot.
    # The non-covariant table does not have this property.
    seg = D.build_segmented_lrb(SEQUENCES[1], 10, 0.005, terminal="data5")
    a = D.simulate_feedback(seg, D.LookupEveryRound(), 4000, 3, "frame")
    b1 = D.simulate_feedback(seg, D.ConfirmedLookup(), 4000, 3, "frame")
    term = D.simulate_feedback(seg, _TerminalOnlyLookup(), 4000, 3, "frame")
    nc = D.simulate_feedback(seg, D.LookupEveryRound(D.NONCOVARIANT_TABLE),
                             4000, 3, "frame")
    assert np.array_equal(a.logical_flip, term.logical_flip)
    assert np.array_equal(b1.logical_flip, term.logical_flip)
    assert (nc.logical_flip != term.logical_flip).mean() > 0.01
    # Group closure, checked directly on the Pauli algebra.
    gens = []
    for pauli, q in D.COVARIANT_TABLE:
        v = np.zeros(10, dtype=np.uint8)
        v[q if pauli == "X" else 5 + q] = 1
        gens.append(v)
    span = {tuple(np.bitwise_xor.reduce([g for g, b in zip(gens, bits) if b]
                                        or [np.zeros(10, np.uint8)]))
            for bits in np.ndindex(2, 2, 2, 2)}
    for token in ("H", "P", "X", "Z", "H_INV", "P_INV"):
        mat = D.symplectic_matrix(C.logical_gate_operations(token))
        for v in span:
            image = tuple((np.asarray(v, np.int64) @ mat) % 2)
            assert image in span, token


def test_flip_simulator_rule_a_matches_stim_feedback_rate():
    seg = D.build_segmented_lrb(SEQUENCE, 4, 0.01, terminal="data5")
    physical, _ = D.rule_a_feedback_circuits(seg)
    exact = D.observable_flip_probability(physical)
    run = D.simulate_feedback(seg, D.LookupEveryRound(), 20000, seed=2,
                              mode="physical")
    rate = run.logical_flip.mean()
    sigma = np.sqrt(exact * (1 - exact) / 20000)
    assert abs(rate - exact) < 5 * sigma


def test_causal_mld_rule_reproduces_batch_decoder():
    seg = D.build_segmented_lrb(SEQUENCE, 5, 0.01, terminal="data5")
    run = D.simulate_feedback(seg, D.CausalMLDLogical(1), 1500, seed=6,
                              mode="frame")
    # The same circuit sampled by FlipSimulator, decoded in batch.
    sim = stim.FlipSimulator(batch_size=1500, num_qubits=C.NUM_QUBITS, seed=6)
    for s in seg.segments:
        sim.do(s)
    flips = sim.get_measurement_flips()
    converter = seg.lc.circuit.compile_m2d_converter()
    reference = seg.lc.circuit.reference_sample()
    dets, obs = converter.convert(measurements=flips.T ^ reference[None, :],
                                  separate_observables=True)
    dec = D.CausalWindowMLD(seg.lc.circuit.detector_error_model(),
                            M.lrb_detector_slots(seg.lc), 1)
    expected = obs[:, 0] ^ dec.decode(dets)
    assert np.array_equal(run.logical_flip, expected)


def test_nonclifford_toy_breaks_naive_frame():
    psi = D._encoded_plus_state()
    for pauli in C.STABILIZER_PAULIS + (C.LOGICAL_X,):
        mat = D._pauli_matrix(pauli)
        assert np.allclose(mat @ psi, psi, atol=1e-12)
    out = D.nonclifford_toy(0.1)
    assert out["physical"] == pytest.approx(out["frame_fixup"], abs=1e-12)
    assert out["no_error"] == pytest.approx(np.cos(np.pi / 8) ** 2, abs=1e-12)
    assert out["physical"] - out["frame_naive"] == pytest.approx(
        0.1 * np.sqrt(0.5), abs=1e-12)


@pytest.mark.parametrize("basis,seep_to", [("Z", 1), ("Z", 0), ("X", 0)])
def test_leakage_toy(basis, seep_to):
    out = D.leakage_toy(0.2, 0.1, seep_to, basis)
    no_leak = D.leakage_toy(0.2, 0.0, seep_to, basis)
    assert no_leak["physical"] == pytest.approx(no_leak["frame"], abs=1e-12)
    assert abs(out["physical"] - out["frame"]) == pytest.approx(0.02,
                                                               abs=1e-12)
