"""Tests for WP I: erasure conversion, herald-aware MLD, flags and leakage.

Run with::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m pytest tests/test_qubit512_leak.py -q
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import erasure as ER  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import leakage as L  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQUENCES = E.clifford_sequences(3, 22, 20261028)
SEQ = SEQUENCES[0]


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal", ["data2", "data5"])
@pytest.mark.parametrize("depth", [0, 1, 4, 22])
@pytest.mark.parametrize("mode", ["decode", "sample"])
def test_builder_reproduces_lead_circuit(terminal, depth, mode):
    lead = Q.build_lrb_circuit(SEQ, depth, 2e-3, terminal=terminal)
    mine = ER.build_instrumented_lrb(SEQ, depth, 2e-3, terminal=terminal,
                                     mode=mode)
    assert str(mine.circuit) == str(lead.circuit)
    assert mine.lrb.detector_keys == lead.detector_keys
    assert mine.lrb.rounds == lead.rounds


@pytest.mark.parametrize("depth", [0, 3])
def test_probe_signatures_reproduce_dem(depth):
    p = 1e-6
    probe = ER.build_instrumented_lrb(SEQUENCES[1], depth, p, mode="probe")
    sig = ER.probe_signatures(probe)
    faults = ER.enumerate_single_faults(probe, sig)
    ours: dict = {}
    for f in faults:
        key = (f.detectors, f.flips)
        ours[key] = ours.get(key, 0.0) + f.probability
    ours.pop(((), False), None)
    lead = Q.build_lrb_circuit(SEQUENCES[1], depth, p, terminal="data5")
    dem: dict = {}
    for mech in M.dem_mechanisms(lead.circuit.detector_error_model()):
        key = (mech.detectors, mech.flips_observable)
        dem[key] = dem.get(key, 0.0) + mech.probability
    assert set(ours) == set(dem)
    for key, value in dem.items():
        assert abs(ours[key] - value) <= 1e-4 * value


def test_erasure_circuits_are_deterministic_and_noiseless_clean():
    spec = ER.ErasureSpec(two_qubit=0.7, meas_idle=0.4)
    circuit = ER.build_instrumented_lrb(SEQ, 3, 0.0, erasure=spec).circuit
    dets, obs = circuit.compile_detector_sampler(seed=1).sample(
        64, separate_observables=True)
    assert not dets.any() and not obs.any()
    # Herald detectors are deterministic; the decode circuit has a DEM.
    ER.build_instrumented_lrb(SEQ, 3, 0.01, erasure=spec,
                              mode="decode").circuit.detector_error_model()


def test_heralded_erase_semantics():
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        HERALDED_ERASE(0.4) 0
        CX 0 1
        H 0
        M 0 1
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    s = circuit.compile_detector_sampler(seed=1).sample(200_000)
    herald = s[:, 0]
    assert abs(herald.mean() - 0.4) < 0.005
    for col in (1, 2):
        assert abs(s[herald, col].mean() - 0.5) < 0.01
        assert not s[~herald, col].any()
    assert abs((s[herald, 1] & s[herald, 2]).mean() - 0.25) < 0.01


@pytest.mark.parametrize("spec", [ER.ErasureSpec(two_qubit=1.0),
                                  ER.ErasureSpec(two_qubit=0.5, meas_idle=0.9)])
def test_erasure_conversion_preserves_pauli_channel(spec):
    """Detector and observable statistics equal the unconverted circuit."""
    p, n = 5e-3, 200_000
    lead = Q.build_lrb_circuit(SEQUENCES[1], 3, p, terminal="data5").circuit
    d0, o0 = lead.compile_detector_sampler(seed=1).sample(
        n, separate_observables=True)
    sample = ER.build_instrumented_lrb(SEQUENCES[1], 3, p, erasure=spec)
    events, o1 = sample.circuit.compile_detector_sampler(seed=2).sample(
        n, separate_observables=True)
    d1 = events[:, :sample.num_main_detectors]
    r0, r1 = d0.mean(0), d1.mean(0)
    se = np.sqrt((r0 * (1 - r0) + r1 * (1 - r1)) / n)
    assert np.abs((r0 - r1) / se).max() < 4.5
    so = np.sqrt(2 * o0.mean() * (1 - o0.mean()) / n)
    assert abs(o0.mean() - o1.mean()) < 4.5 * so
    expected = sum((16 / 15 if loc.channel == "DEPOLARIZE2" else 4 / 3)
                   * loc.herald_fraction * loc.probability
                   for loc in sample.locations if loc.herald >= 0)
    heralds = events[:, sample.num_main_detectors:].sum(axis=1)
    assert abs(heralds.mean() - expected) < 5 * np.sqrt(expected / n)


# ---------------------------------------------------------------------------
# Herald-conditioned MLD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0, 5])
def test_heralded_mld_without_heralds_equals_lead_mld(depth):
    lead = Q.build_lrb_circuit(SEQ, depth, 5e-3, terminal="data5")
    dem = lead.circuit.detector_error_model()
    a = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lead))
    b = ER.HeraldedMLD(dem, M.lrb_detector_slots(lead))
    dets, _ = lead.circuit.compile_detector_sampler(seed=2).sample(
        3000, separate_observables=True)
    assert np.allclose(a.log_likelihood_ratio(dets),
                       b.log_likelihood_ratio(dets), atol=1e-10)


def _brute_force_with_heralds(decode_dem, comps, active):
    dem = decode_dem.copy()
    for h in active:
        for dets, flips in comps[h]:
            if not dets and not flips:
                continue
            targets = [stim.target_relative_detector_id(d) for d in dets]
            if flips:
                targets.append(stim.target_logical_observable_id(0))
            dem.append("error", 0.5, targets)
    return M.brute_force_joint(dem)


@pytest.mark.parametrize("depth,spec", [
    (0, ER.ErasureSpec(two_qubit=1.0)),
    (1, ER.ErasureSpec(two_qubit=0.5, meas_idle=0.5)),
    (2, ER.ErasureSpec(two_qubit=0.9))])
def test_heralded_mld_matches_brute_force(depth, spec):
    ec = ER.prepare_erasure_circuits(SEQ, depth, 0.02, spec)
    events, _ = ec.sample.circuit.compile_detector_sampler(seed=4).sample(
        1500, separate_observables=True)
    dets, heralds = ec.sample.split(events)
    llr = ec.mld_herald.log_likelihood_ratio(dets, heralds)
    comps = ER.herald_components(ec.sample, ec.signatures)
    dem = ec.decode.circuit.detector_error_model()
    n = dem.num_detectors
    patterns: dict = {}
    for i, row in enumerate(heralds):
        patterns.setdefault(tuple(np.flatnonzero(row)), []).append(i)
    for active, rows in itertools.islice(patterns.items(), 25):
        joint = _brute_force_with_heralds(dem, comps, active)
        index = dets[rows].astype(np.int64) @ (1 << np.arange(n))
        expected = np.log(joint[index]) - np.log(joint[index + (1 << n)])
        assert np.allclose(llr[rows], expected, atol=1e-9)


def test_heralded_posterior_is_calibrated_on_the_true_channel():
    """Independent check: the conditional model matches sampled frequencies."""
    spec = ER.ErasureSpec(two_qubit=1.0)
    ec = ER.prepare_erasure_circuits(SEQ, 0, 0.03, spec)
    n = 400_000
    events, obs = ec.sample.circuit.compile_detector_sampler(seed=11).sample(
        n, separate_observables=True)
    dets, heralds = ec.sample.split(events)
    llr = ec.mld_herald.log_likelihood_ratio(dets, heralds)
    posterior = 1.0 / (1.0 + np.exp(llr))
    wrong = (llr < 0) != obs[:, 0]
    predicted = np.minimum(posterior, 1 - posterior).mean()
    assert abs(wrong.mean() - predicted) < 4 * np.sqrt(predicted / n)
    key = np.packbits(np.concatenate([dets, heralds], axis=1), axis=1)
    view = np.ascontiguousarray(key).view(
        np.dtype((np.void, key.shape[1]))).ravel()
    _, inverse, counts = np.unique(view, return_inverse=True,
                                   return_counts=True)
    chi2, dof = 0.0, 0
    for g in np.flatnonzero(counts >= 3000):
        rows = inverse.ravel() == g
        pr, k, m = posterior[rows][0], obs[rows, 0].sum(), rows.sum()
        if 1e-6 < pr < 1 - 1e-6:
            chi2 += (k - m * pr) ** 2 / (m * pr * (1 - pr))
            dof += 1
    assert dof >= 10
    assert chi2 < dof + 5 * np.sqrt(2 * dof)


def test_code_capacity_single_erasures_are_correctable():
    """X, Y, Z on any one data qubit give three distinct nonzero syndromes."""
    stabs = [stim.PauliString(s) for s in C.STABILIZER_PAULIS]
    for q in range(C.NUM_DATA):
        syndromes = set()
        for pauli in "XYZ":
            error = stim.PauliString("_" * C.NUM_DATA)
            error[q] = pauli
            syndromes.add(tuple(int(not error.commutes(s)) for s in stabs))
        assert len(syndromes) == 3 and (0, 0, 0, 0) not in syndromes


def test_every_single_heralded_gate_erasure_is_corrected():
    """Circuit level: all 16 Paulis of any one heralded 2q-gate erasure."""
    spec = ER.ErasureSpec(two_qubit=1.0)
    ec = ER.prepare_erasure_circuits(SEQ, 2, 1e-3, spec)
    n_det = ec.sample.num_main_detectors
    for h, loc_index in enumerate(ec.sample.herald_locations):
        loc = ec.sample.locations[loc_index]
        events, flips = [], []
        for pa in range(4):
            for pb in range(4):
                dets, obs = ec.signatures.pauli(
                    loc_index, {loc.qubits[0]: pa, loc.qubits[1]: pb})
                events.append(dets[:n_det])
                flips.append(obs)
        events = np.asarray(events)
        heralds = np.zeros((16, ec.sample.num_heralds), dtype=bool)
        heralds[:, h] = True
        llr = ec.mld_herald.log_likelihood_ratio(events, heralds)
        assert np.array_equal(llr < 0, np.asarray(flips)), loc
        e0, e1 = ec.explain.log_joint(events, heralds)
        assert np.all(np.isfinite(e0) ^ np.isfinite(e1))


def test_r0_methods_coincide_on_identical_shots():
    result = ER.run_erasure_point(5e-3, ER.ErasureSpec(), SEQUENCES[:1],
                                  [0, 3], 4000, seed=3)
    t = result["tables"]
    assert np.array_equal(t["hybrid"].accepted, t["lrbd_unif_1_final"].accepted)
    assert np.array_equal(t["hybrid"].errors, t["lrbd_unif_1_final"].errors)
    assert np.array_equal(t["mld_herald"].errors, t["mld"].errors)
    assert np.array_equal(t["lrbd_unif_1_final_noherald"].accepted,
                          t["lrbd_unif_1_final"].accepted)


def test_hybrid_beats_lrbd_acceptance_with_heralds():
    result = ER.run_erasure_point(5e-3, ER.ErasureSpec(two_qubit=1.0),
                                  SEQUENCES[:1], [2], 20000, seed=5)
    t = result["tables"]
    assert t["hybrid"].accepted.sum() > t["lrbd_unif_1_final"].accepted.sum()
    assert (t["lrbd_unif_1_final_noherald"].accepted.sum()
            < t["lrbd_unif_1_final"].accepted.sum())


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags", ["noisy", "ideal"])
@pytest.mark.parametrize("layout", ["standard", "wrap"])
def test_flagged_circuits_valid_and_mld_exact(flags, layout):
    noiseless = ER.build_instrumented_lrb(SEQ, 2, 0.0, flags=flags,
                                          flag_layout=layout, mode="decode")
    dets, obs = noiseless.circuit.compile_detector_sampler(seed=1).sample(
        128, separate_observables=True)
    assert not dets.any() and not obs.any()
    lc = ER.build_instrumented_lrb(SEQ, 1, 0.01, flags=flags,
                                   flag_layout=layout, mode="decode")
    dem = lc.circuit.detector_error_model()
    slots = lc.decoder_slots()
    assert np.bincount(slots).max() == 8
    decoder = M.TransferMatrixMLD(dem, slots)
    joint = M.brute_force_joint(dem)
    n = dem.num_detectors
    events, _ = lc.circuit.compile_detector_sampler(seed=3).sample(
        1000, separate_observables=True)
    index = events.astype(np.int64) @ (1 << np.arange(n))
    expected = np.log(joint[index]) - np.log(joint[index + (1 << n)])
    assert np.allclose(decoder.log_likelihood_ratio(events), expected,
                       atol=1e-9)


def _first_order(flags, layout, depth, p=1e-4):
    probe = ER.build_instrumented_lrb(SEQ, depth, p, flags=flags,
                                      flag_layout=layout, mode="probe")
    dec_c = ER.build_instrumented_lrb(SEQ, depth, p, flags=flags,
                                      flag_layout=layout, mode="decode")
    sig = ER.probe_signatures(probe)
    n = dec_c.num_main_detectors + len(dec_c.flag_detectors)
    faults = ER.enumerate_single_faults(probe, sig, n)
    decoder = M.TransferMatrixMLD(dec_c.circuit.detector_error_model(),
                                  dec_c.decoder_slots())
    keys = sorted({f.detectors for f in faults if f.detectors})
    table = ER.first_order_table(faults, ER.decide_syndromes(decoder, keys, n))
    return probe, table


def test_first_order_mld_error_and_flags():
    _, none = _first_order(None, "standard", 2)
    _, ideal = _first_order("ideal", "wrap", 2)
    probe, wrap = _first_order("noisy", "wrap", 2)
    # Exact MLD first order = minority mass of each syndrome class.
    assert abs(none["first_order_error"] - none["minority_mass"]) < 1e-12
    assert none["undetected_logical"] == 0.0
    # Ideal flags only add information; the wrap layout flags every hook.
    assert ideal["first_order_error"] < none["first_order_error"]
    assert not any(ER.is_hook(probe, f) and probe.locations[f.location].kind
                   == "cx_check" for f in wrap["miscorrected"])


# ---------------------------------------------------------------------------
# Leakage simulator
# ---------------------------------------------------------------------------


def test_bernoulli_indices_statistics():
    rng = np.random.default_rng(1)
    counts = np.array([L._bernoulli_indices(rng, 1000, 0.01).size
                       for _ in range(4000)])
    assert abs(counts.mean() - 10.0) < 0.25
    assert abs(counts.var() - 9.9) < 1.2
    idx = L._bernoulli_indices(rng, 100, 0.3)
    assert np.all(np.diff(idx) > 0) and idx.max(initial=0) < 100


@pytest.mark.parametrize("depth,p", [(4, 5e-3), (14, 2e-3)])
def test_leakage_free_simulator_matches_stim(depth, p):
    lc = Q.build_lrb_circuit(SEQUENCES[2], depth, p, terminal="data5")
    n = 200_000
    d0, o0 = lc.circuit.compile_detector_sampler(seed=3).sample(
        n, separate_observables=True)
    s = L.simulate_leakage(lc, p, L.LeakageModel(rate=0.0), n, seed=5)
    r0, r1 = d0.mean(0), s.dets.mean(0)
    se = np.sqrt((r0 * (1 - r0) + r1 * (1 - r1)) / n)
    assert np.abs((r0 - r1) / se).max() < 4.5
    so = np.sqrt(2 * o0.mean() * (1 - o0.mean()) / n)
    assert abs(o0[:, 0].mean() - s.obs.mean()) < 4.5 * so
    c0 = d0[:, :8].T.astype(float) @ d0[:, :8].astype(float) / n
    c1 = s.dets[:, :8].T.astype(float) @ s.dets[:, :8].astype(float) / n
    assert np.abs(c0 - c1).max() < 0.004
    assert not s.any_flag().any() and s.leak_events == 0


def test_leaked_data_qubit_randomizes_partners_until_lru():
    lc = Q.build_lrb_circuit(SEQ, 5, 1e-7, terminal="data5")
    n = 20000
    s = L.simulate_leakage(lc, 1e-7, L.LeakageModel(rate=0.0), n, seed=1,
                           initial_leaked=[1])
    rates = s.dets[:, :4 * lc.num_rounds].reshape(n, lc.num_rounds, 4).mean(0)
    assert np.all(np.abs(rates - 0.5) < 0.03)
    assert s.terminal_flags[:, 1].all() and not s.terminal_flags[:, 0].any()
    s = L.simulate_leakage(lc, 1e-7, L.LeakageModel(rate=0.0, lru=True), n,
                           seed=1, initial_leaked=[1])
    rates = s.dets[:, :4 * lc.num_rounds].reshape(n, lc.num_rounds, 4).mean(0)
    assert np.all(np.abs(rates[:2] - 0.5) < 0.03)
    assert np.all(rates[2:] < 0.01)
    assert s.lru_flags[:, 0, 1].all() and not s.lru_flags[:, 1:].any()
    assert not s.terminal_flags.any()


def test_leaked_ancilla_reads_fixed_value():
    lc = Q.build_lrb_circuit(SEQ, 4, 1e-7, terminal="data5")
    s = L.simulate_leakage(lc, 1e-7, L.LeakageModel(rate=0.0,
                                                     reset_returns=False),
                           2000, seed=1, initial_leaked=[5])
    raw = lc.raw_syndromes(s.dets)
    assert raw[:, :, 0].all() and s.ancilla_flags[:, :, 0].all()
    s = L.simulate_leakage(lc, 1e-7, L.LeakageModel(rate=0.0), 2000, seed=1,
                           initial_leaked=[5])
    assert lc.raw_syndromes(s.dets)[:, 0, 0].all()
    assert s.ancilla_flags[:, 0, 0].all() and not s.ancilla_flags[:, 1:].any()


def test_leaked_logical_qubit_randomizes_readout():
    lc = Q.build_lrb_circuit(SEQ, 2, 1e-7, terminal="data5")
    s = L.simulate_leakage(lc, 1e-7, L.LeakageModel(rate=0.0), 40000, seed=2,
                           initial_leaked=[0])
    assert abs(s.obs.mean() - 0.5) < 0.015


def test_leak_rate_and_persistence():
    p = 2e-3
    lc = Q.build_lrb_circuit(SEQ, 10, p, terminal="data5")
    s = L.simulate_leakage(lc, p, L.LeakageModel(rate=0.5), 20000, seed=3)
    gates = len(L._noisy_two_qubit_instructions(lc.circuit))
    expected = 2 * gates * 0.5 * p
    assert abs(s.leak_events / 20000 - expected) < 0.05 * expected
    # Without LRU the leaked-data population grows round after round.
    assert np.all(np.diff(s.leaked_data[:-1]) > 0)
    s2 = L.simulate_leakage(lc, p, L.LeakageModel(rate=0.5, lru=True), 20000,
                            seed=3)
    assert s2.leaked_data[-2] < 0.5 * s.leaked_data[-2]
