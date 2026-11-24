"""Tests for work package E: gap postselection, repo-style fits, hybrid rules.

Run with::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m pytest tests/test_qubit512_pareto.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("stim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrb_qubit import acceptance_rules as A  # noqa: E402
from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQUENCE = Q.random_clifford_sequence(22, np.random.default_rng(7))
# A sequence whose first Clifford contains both H and S tokens.
SEQUENCE_HS = ["HP", "PH", "I", "XZ"] + SEQUENCE


def _repo_plotter():
    lp = pytest.importorskip("lrb.lrb_plotting")
    stub = SimpleNamespace(fit_config=lp.LRBPlotFitConfig(),
                           _safe_weights=lp.LRBResultsPlotter._safe_weights)
    return lp.LRBResultsPlotter, stub


def _random_series(rng, n_depths=9, nan_prob=0.1, zero_prob=0.1):
    depths = [0, 2, 4, 6, 10, 14, 18, 20, 22][:n_depths]
    f = rng.uniform(0.8, 1.0)
    b = rng.uniform(0.5, 1.0)
    means = b * f ** np.asarray(depths, float) + rng.normal(0, 0.02, n_depths)
    means = np.clip(means, 0.0, 1.05)
    errs = rng.uniform(0.0, 0.05, n_depths)
    errs[rng.random(n_depths) < zero_prob] = 0.0
    means[rng.random(n_depths) < nan_prob] = np.nan
    return depths, list(means), list(errs)


# ---------------------------------------------------------------------------
# Repo fit replication
# ---------------------------------------------------------------------------

def test_fixed_offset_fit_matches_repo():
    cls, stub = _repo_plotter()
    rng = np.random.default_rng(1)
    for _ in range(40):
        depths, means, errs = _random_series(rng)
        x, y, e = cls._mask_invalid(depths, means, errs)
        x, y, e = cls._apply_ceiling_cut(x, y, e, stub.fit_config.y_ceiling)
        repo = cls._fit_decay_parameter_fixed_a(stub, x, y, e)
        ours = A.fit_fixed_offset(depths, means, errs)
        if repo is None:
            assert ours is None
            continue
        assert ours["f"] == repo["f"]
        assert ours["b"] == pytest.approx(repo["b"], rel=1e-12, abs=1e-15)
        assert ours["n_points"] == repo["n_points"]


def test_free_offset_fit_matches_repo():
    cls, stub = _repo_plotter()
    rng = np.random.default_rng(2)
    for _ in range(40):
        depths, means, errs = _random_series(rng, nan_prob=0.05)
        x, y, e = cls._mask_invalid(depths, means, errs)
        repo = cls._fit_decay_parameter_free_a(stub, x, y, e)
        ours = A.fit_free_offset(depths, means, errs)
        if repo is None:
            assert ours is None
            continue
        assert ours["f"] == repo["f"]
        assert ours["a"] == pytest.approx(repo["a"], rel=1e-12, abs=1e-15)


def test_batch_fit_matches_scalar_fit():
    rng = np.random.default_rng(3)
    rows = [_random_series(rng) for _ in range(30)]
    depths = rows[0][0]
    means = np.array([r[1] for r in rows])
    errs = np.array([r[2] for r in rows])
    batch = A.fit_fixed_offset_batch(depths, means, errs, chunk=7)
    for i in range(len(rows)):
        scalar = A.fit_fixed_offset(depths, means[i], errs[i])
        if scalar is None:
            assert np.isnan(batch[i])
        else:
            assert batch[i] == pytest.approx(scalar["f"], abs=2e-6)


def test_weighted_stats_match_direct_resample():
    rng = np.random.default_rng(4)
    acc = rng.integers(50, 100, size=(10, 4)).astype(float)
    err = rng.integers(0, 20, size=(10, 4)).astype(float)
    acc[3, 2] = 0.0  # NaN fidelity for one sequence
    mult = A.bootstrap_multiplicities(10, 25, rng)
    fid = A.fidelity(acc, err)
    means, stds = A.weighted_fidelity_stats(fid, mult)
    for b in range(25):
        idx = np.repeat(np.arange(10), mult[b])
        m_ref, s_ref = A.repo_fidelity_stats(acc[idx], err[idx])
        np.testing.assert_allclose(means[b], m_ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(stds[b], s_ref, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Curves, matching, randomized rules
# ---------------------------------------------------------------------------

def _calibrated_hist(rng, lead, edges):
    """Expected (correct, wrong) counts whose wrong fraction lies inside each
    bin's q band, as for an exact soft output."""
    bins = len(edges) - 1
    n = rng.integers(0, 200, size=tuple(lead) + (bins,))
    q_hi = A.gap_error_probability(edges[:-1])
    q_lo = A.gap_error_probability(edges[1:])
    q = q_lo + rng.random(n.shape) * (q_hi - q_lo)
    wrong = n * q          # expected counts: exactly calibrated
    return np.stack([n - wrong, wrong], axis=-1)


def test_cumulative_counts_and_matching():
    rng = np.random.default_rng(5)
    hist = rng.integers(0, 50, size=(3, 12, 2))
    acc, err = A.cumulative_counts(hist)
    assert acc.shape == (3, 13)
    assert np.all(acc[:, 0] == hist.sum(axis=(1, 2)))
    assert np.all(acc[:, -1] == 0)
    assert np.all(err[:, 4] == hist[:, 4:, 1].sum(axis=1))
    pooled_acc, pooled_err = A.cumulative_counts(hist.sum(axis=0))
    for target in np.linspace(0, pooled_acc[0], 17):
        k, lam = A.match_acceptance(pooled_acc, target)
        a, _ = A.interpolate_counts(pooled_acc, pooled_err, k, lam)
        assert a == pytest.approx(target)
    for rate in (0.0, 0.05, 0.2, 0.4, 0.5, 0.9):
        k, lam, accepted = A.match_error_rate(pooled_acc, pooled_err, rate)
        a, e = A.interpolate_counts(pooled_acc, pooled_err, k, lam)
        assert a == pytest.approx(accepted)
        if a > 0:
            assert e / a <= rate + 1e-12
        # brute force over a fine grid of randomized rules
        best = 0.0
        for kk in range(1, len(pooled_acc)):
            for ll in np.linspace(0, 1, 201):
                aa, ee = A.interpolate_counts(pooled_acc, pooled_err, kk, ll)
                if aa > 0 and ee / aa <= rate + 1e-12:
                    best = max(best, aa)
        assert accepted >= best - 1e-9


def test_pareto_curve_bounds():
    rng = np.random.default_rng(6)
    hist = _calibrated_hist(rng, (4, 5), A.GAP_EDGES)
    curve = A.ParetoCurve.from_hist(hist)
    assert curve.shots == hist.sum()
    for alpha in (0.1, 0.33, 0.5, 0.9):
        out = curve.error_at_acceptance(alpha)
        assert out["accepted"] == pytest.approx(alpha * curve.shots)
        assert out["rate_low"] <= out["rate"] + 1e-15


def test_batch_matching_equals_scalar_matching():
    rng = np.random.default_rng(8)
    edges = np.concatenate([np.arange(0, 5, 0.25), [np.inf]])
    hist = _calibrated_hist(rng, (6,), edges)
    acc, err = A.cumulative_counts(hist)
    acc = acc.astype(float)
    err = err.astype(float)
    targets = rng.uniform(0, 1, 6) * acc[:, 0]
    k, lam, rate, rate_low = A.batch_error_at_acceptance(acc, err, targets,
                                                         edges)
    for i in range(6):
        k1, lam1 = A.match_acceptance(acc[i], targets[i])
        a1, e1 = A.interpolate_counts(acc[i], err[i], k1, lam1)
        assert rate[i] == pytest.approx(e1 / a1)
        assert rate_low[i] <= rate[i] + 1e-15
    rates = rng.uniform(0, 0.6, 6)
    k, lam, accepted = A.batch_acceptance_at_error(acc, err, rates)
    for i in range(6):
        _, _, a1 = A.match_error_rate(acc[i], err[i], rates[i])
        assert accepted[i] == pytest.approx(a1)


def test_matched_gap_scorer_matches_weights():
    rng = np.random.default_rng(9)
    n = 500
    llr = np.round(rng.normal(0, 3, n), 1)          # many exact ties
    flip = rng.random(n) < 1 / (1 + np.exp(np.abs(llr)))
    flip ^= llr < 0
    scorer = A.MatchedGapScorer(llr, flip)
    wrong = (llr < 0) != flip
    for _ in range(20):
        accept = rng.random(n) < rng.random()
        rule_wrong = rng.random(n) < 0.3
        out = scorer.score(accept, rule_wrong)
        w = A.matched_gap_weights(llr, accept.sum())
        z = (accept & rule_wrong).astype(float) - w * wrong
        assert out["gap_errors"] == pytest.approx(np.sum(w * wrong))
        assert out["diff"] == pytest.approx(z.sum())
        assert out["diff_var"] == pytest.approx(
            np.sum(z * z) - z.sum() ** 2 / n)
        assert scorer.gap_errors(int(accept.sum())) == pytest.approx(
            out["gap_errors"])


def test_matched_gap_weights():
    gap = np.array([5.0, 1.0, 3.0, 3.0, 3.0, 0.5, 7.0])
    w = A.matched_gap_weights(gap, 3)
    assert w.sum() == pytest.approx(3)
    assert w[6] == 1 and w[0] == 1 and w[1] == 0 and w[5] == 0
    np.testing.assert_allclose(w[2:5], 1 / 3)
    assert A.matched_gap_weights(gap, 0).sum() == 0
    assert A.matched_gap_weights(gap, 9).sum() == 7


# ---------------------------------------------------------------------------
# Neyman-Pearson optimality on an exactly enumerable circuit
# ---------------------------------------------------------------------------

def _exact_syndrome_table(depth, p, terminal="data5"):
    lc = Q.build_lrb_circuit(SEQUENCE_HS, depth, p, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    joint = M.brute_force_joint(dem)
    n = dem.num_detectors
    p0 = joint[: 2 ** n]
    p1 = joint[2 ** n:]
    return lc, n, p0, p1


def _optimal_error(prob, q, alpha):
    """Exact optimal accepted error mass at acceptance alpha (fractional)."""
    order = np.argsort(q, kind="stable")
    cum_a = np.concatenate([[0.0], np.cumsum(prob[order])])
    cum_e = np.concatenate([[0.0], np.cumsum((prob * q)[order])])
    return np.interp(alpha, cum_a, cum_e)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_gap_threshold_is_neyman_pearson_optimal(depth):
    lc, n, p0, p1 = _exact_syndrome_table(depth, 0.01)
    prob = p0 + p1
    q = np.minimum(p0, p1) / np.where(prob > 0, prob, 1)
    rng = np.random.default_rng(depth)
    # Random (randomized) acceptance rules with MLD or raw decisions.
    for _ in range(300):
        a = rng.random(2 ** n) * (rng.random(2 ** n) < rng.random())
        alpha = float(np.sum(prob * a))
        e_mld = float(np.sum(a * prob * q))
        e_raw = float(np.sum(a * p1))
        bound = _optimal_error(prob, q, alpha)
        assert e_mld >= bound - 1e-15
        assert e_raw >= bound - 1e-15
    # LRB-D policies (raw syndrome postselection, no correction).
    events = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
              ).astype(bool)
    raw = lc.raw_syndromes(events)
    for policy, k in [("unif", 1), ("const", 1), ("unif", 2)]:
        acc = B.accept_mask(raw, lc.rounds, B.used_rounds(policy, k, depth))
        alpha = float(prob[acc].sum())
        assert float(p1[acc].sum()) >= _optimal_error(prob, q, alpha) - 1e-15
    # Threshold rules: per-shot error never exceeds q(t).
    llr = np.log(np.where(p0 > 0, p0, 1e-300)) - np.log(
        np.where(p1 > 0, p1, 1e-300))
    for t in (0.0, 1.0, 2.0, 4.0):
        acc = (np.abs(llr) >= t) & (prob > 0)
        if acc.any():
            rate = float((prob * q)[acc].sum() / prob[acc].sum())
            assert rate <= A.gap_error_probability(t) + 1e-12


def test_transfer_matrix_llr_matches_enumeration_on_samples():
    lc, n, p0, p1 = _exact_syndrome_table(2, 0.01)
    dem = lc.circuit.detector_error_model()
    mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    sampler = lc.circuit.compile_detector_sampler(seed=3)
    dets, _ = sampler.sample(2000, separate_observables=True)
    index = dets.astype(np.int64) @ (1 << np.arange(n))
    exact = np.log(p0[index]) - np.log(p1[index])
    np.testing.assert_allclose(mld.log_likelihood_ratio(dets), exact,
                               rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Hybrid rules
# ---------------------------------------------------------------------------

def _layouts(depth, p=0.002):
    lc5 = Q.build_lrb_circuit(SEQUENCE_HS, depth, p, terminal="data5")
    lc2 = Q.build_lrb_circuit(SEQUENCE_HS, depth, p, terminal="data2")
    return (lc5, lc2, A.SlotLayout.from_circuit(lc5, True),
            A.SlotLayout.from_circuit(lc5, False),
            lc5.circuit.detector_error_model(),
            lc2.circuit.detector_error_model())


@pytest.mark.parametrize("depth", [0, 1, 3, 6])
def test_time_like_rule_has_no_first_order_logical_errors(depth):
    _, _, l5, l2, dem5, dem2 = _layouts(depth)
    audit5 = A.single_fault_audit(l5, A.structural_tables(l5, False), dem5)
    assert audit5["accepted_wrong"] == 0.0
    audit2 = A.single_fault_audit(l2, A.structural_tables(l2, False), dem2)
    # Only the undetectable data-2 readout faults remain (about 4.5 p).
    assert audit2["accepted_wrong"] == pytest.approx(
        audit2["undetectable_flip"])
    assert audit2["undetectable_flip"] == pytest.approx(4.5 * 0.002, rel=0.05)
    for layout, dem in ((l5, dem5), (l2, dem2)):
        audit = A.single_fault_audit(layout, A.dem_lookup_tables(layout, dem),
                                     dem)
        assert audit["accepted_wrong"] == pytest.approx(
            audit["undetectable_flip"])


def test_qubit1_rule_first_order_errors_are_fold_gate_pair_faults():
    import stim
    _, _, l5, _, dem5, _ = _layouts(0)
    assert A.single_fault_audit(l5, A.structural_tables(l5, True),
                                dem5)["accepted_wrong"] == 0.0
    lc5, _, l5, _, dem5, _ = _layouts(1)
    tables = A.structural_tables(l5, True)
    audit = A.single_fault_audit(l5, tables, dem5)
    assert audit["accepted_wrong"] > 0
    weights = A.dem_signature_weights(l5, dem5)
    checked = 0
    for key, (w0, w1) in weights.items():
        if key[0] == "pair" and w1 > 0:
            assert tables.pair[key[1]][key[2]] < 0
        if key[0] != "single" or w1 == 0:
            continue
        if tables.single[key[1]][key[2]] < 0:
            continue
        t, code = key[1], key[2]
        dets = [int(l5.columns[t][b]) for b in range(l5.bits[t])
                if (code >> b) & 1]
        flt = stim.DetectorErrorModel(
            "error(0.1) " + " ".join(f"D{d}" for d in dets) + " L0")
        explained = lc5.circuit.explain_detector_error_model_errors(
            dem_filter=flt, reduce_to_one_representative_error=False)
        for err in explained:
            for loc in err.circuit_error_locations:
                text = str(loc)
                assert "DEPOLARIZE2" in text and " 2 3" in text, text
                checked += 1
    assert checked > 0


def test_cluster_rule_vectorized_matches_reference():
    lc5, _, l5, l2, dem5, dem2 = _layouts(4, p=0.02)
    sampler = lc5.circuit.compile_detector_sampler(seed=11)
    dets, _ = sampler.sample(3000, separate_observables=True)
    for layout, dem in ((l5, dem5), (l2, dem2)):
        codes = layout.codes(dets[:, :layout.num_detectors]
                             if not layout.has_final else dets)
        for tables in (A.structural_tables(layout, False),
                       A.structural_tables(layout, True),
                       A.dem_lookup_tables(layout, dem, 0.0),
                       A.dem_lookup_tables(layout, dem, 0.1)):
            acc, flip = A.evaluate_cluster_rule(codes, layout, tables)
            for i in range(0, len(codes), 7):
                ref_acc, ref_flip = A.reference_cluster_rule(
                    list(codes[i]), layout, tables)
                assert acc[i] == ref_acc
                if ref_acc:
                    assert flip[i] == ref_flip


def test_rules_accept_noiseless_shots_without_flip():
    lc5 = Q.build_lrb_circuit(SEQUENCE_HS, 5, 0.0, terminal="data5")
    layout = A.SlotLayout.from_circuit(lc5, True)
    dets, obs = lc5.circuit.compile_detector_sampler().sample(
        50, separate_observables=True)
    codes = layout.codes(dets)
    for tables in (A.structural_tables(layout, False),
                   A.structural_tables(layout, True)):
        acc, flip = A.evaluate_cluster_rule(codes, layout, tables)
        assert acc.all() and not flip.any() and not obs.any()


def test_unif1_is_no_check_round_events():
    lc5 = Q.build_lrb_circuit(SEQUENCE_HS, 6, 0.01, terminal="data5")
    dets, _ = lc5.circuit.compile_detector_sampler(seed=5).sample(
        4000, separate_observables=True)
    raw = lc5.raw_syndromes(dets)
    acc = B.accept_mask(raw, lc5.rounds, B.used_rounds("unif", 1, 6))
    np.testing.assert_array_equal(acc, ~dets[:, :4 * lc5.num_rounds].any(1))


def test_measurement_error_patterns_follow_stabilizer_maps():
    lc5, _, l5, _, _, _ = _layouts(3)
    for t in range(l5.num_slots - 2):
        nxt = lc5.rounds[t + 1]
        for s in range(4):
            first, second = l5.measurement_error_pattern(t, s)
            assert first == 1 << s
            expected = sum(1 << s2 for s2 in range(4)
                           if s in lc5.back[nxt][s2])
            assert second == expected
    terminal = l5.num_slots - 2
    assert l5.measurement_error_pattern(terminal, 0)[1] is None
    assert l5.measurement_error_pattern(terminal, 2)[1] == 0b01
    assert l5.measurement_error_pattern(terminal, 3)[1] == 0b10
