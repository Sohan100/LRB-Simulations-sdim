"""Tests for the WP-H circuit variants (``src/lrb_qubit/variants.py``).

Run with
``module load python/3.11 && OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_qubit512_variants.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

stim = pytest.importorskip("stim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402
from lrb_qubit import variants as V  # noqa: E402

SEQUENCE = Q.random_clifford_sequence(22, np.random.default_rng(11))
P_TEST = 0.004


def _det_obs(lc, shots=64, seed=3):
    return lc.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)


# ---------------------------------------------------------------------------
# Equivalence with the lead's builder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("terminal", ["data2", "data5"])
@pytest.mark.parametrize("depth,forward,inverse_every", [
    (0, None, 0), (1, None, 0), (3, None, 0), (6, None, 2), (6, [1, 4], 0),
    (6, [], 3), (10, [2, 5, 8], 1)])
@pytest.mark.parametrize("p", [0.0, P_TEST])
def test_serial_reproduces_lead_builder(terminal, depth, forward,
                                        inverse_every, p):
    ref = Q.build_lrb_circuit(SEQUENCE, depth, p, terminal=terminal,
                              forward_check_rounds=forward,
                              inverse_check_every=inverse_every)
    ours = V.build_variant_circuit(SEQUENCE, depth, p, terminal=terminal,
                                   forward_check_rounds=forward,
                                   inverse_check_every=inverse_every)
    assert str(ours.circuit) == str(ref.circuit)
    assert ours.rounds == ref.rounds
    assert ours.detector_keys == ref.detector_keys
    assert ours.back == ref.back
    assert ours.segments == ref.segments


def test_const0_spec_reproduces_lead_builder():
    spec = V.VariantSpec("c0", const0=True)
    for depth in (0, 4, 22):
        ref = Q.build_lrb_const0_circuit(SEQUENCE, depth, P_TEST)
        assert str(spec.build(SEQUENCE, depth, P_TEST).circuit) == str(
            ref.circuit)


def test_serial_round_operations_match_code():
    assert V.check_round_operations("serial") == C.check_round_operations()
    assert V.check_round_operations("serial", reset=False) == (
        C.z_stabilizer_check_operations() + C.x_stabilizer_check_operations())


@pytest.mark.parametrize("schedule", V.SCHEDULES)
def test_schedules_keep_gates_and_measurement_order(schedule):
    ops = V.check_round_operations(schedule, reset=False)
    serial = V.check_round_operations("serial", reset=False)
    assert [o for o in ops if o[0] != "M"] == [o for o in serial
                                                if o[0] != "M"]
    assert [o for o in ops if o[0] == "M"] == [("M", (q,))
                                               for q in C.ANCILLA_QUBITS]


# ---------------------------------------------------------------------------
# Noise bookkeeping
# ---------------------------------------------------------------------------

def _idle_layers_on(circuit: stim.Circuit, qubit: int, value: float) -> int:
    count = 0
    for inst in circuit:
        if inst.name == "DEPOLARIZE1" and abs(inst.gate_args_copy()[0]
                                              - value) < 1e-15:
            if qubit in [t.value for t in inst.targets_copy()]:
                count += 1
    return count


@pytest.mark.parametrize("schedule,reset,expected", [
    ("serial", True, 5), ("zx", True, 3), ("parallel", True, 2),
    ("serial", False, 4), ("zx", False, 2), ("parallel", False, 1)])
def test_data_idle_layers_per_round(schedule, reset, expected):
    p = 0.001
    circuit = stim.Circuit()
    V.emit_with_noise(circuit, V.check_round_operations(schedule, reset), p,
                      Q.SI1000())
    for q in C.DATA_QUBITS:
        assert _idle_layers_on(circuit, q, 2 * p) == expected
    assert V.data_idle_layers_per_round(schedule, reset) == expected


def test_idle_split_default_is_emit_operations():
    ops = V.check_round_operations("serial")
    a, b = stim.Circuit(), stim.Circuit()
    Q.emit_operations(a, ops, P_TEST, Q.SI1000(), C.NUM_QUBITS)
    V.emit_with_noise(b, ops, P_TEST, Q.SI1000(), meas_idle_data=1.0,
                      meas_idle_ancilla=1.0)
    assert str(a) == str(b)
    # Zeroing both halves of the split equals zeroing meas_idle.
    c, d = stim.Circuit(), stim.Circuit()
    V.emit_with_noise(c, ops, P_TEST, Q.SI1000(), meas_idle_data=0.0,
                      meas_idle_ancilla=0.0)
    Q.emit_operations(d, ops, P_TEST, Q.SI1000().scaled(meas_idle=0.0),
                      C.NUM_QUBITS)
    assert str(c) == str(d)


def test_idle_split_targets():
    # No reset layer here, so every 2p DEPOLARIZE1 comes from a measurement layer.
    ops = V.check_round_operations("serial", reset=False)
    p = 0.001
    for data_f, anc_f, expected in ((1.0, 0.0, set(C.DATA_QUBITS)),
                                    (0.0, 1.0, set(C.ANCILLA_QUBITS))):
        circuit = stim.Circuit()
        V.emit_with_noise(circuit, ops, p, Q.SI1000(), meas_idle_data=data_f,
                          meas_idle_ancilla=anc_f)
        hit = set()
        for inst in circuit:
            if inst.name == "DEPOLARIZE1" and abs(
                    inst.gate_args_copy()[0] - 2 * p) < 1e-15:
                hit |= {t.value for t in inst.targets_copy()}
        assert hit == expected


def test_perfect_noise_has_empty_dem():
    lc = V.build_variant_circuit(
        SEQUENCE, 6, P_TEST, schedule="parallel",
        noise=V.NoiseConfig(check=V._ZERO, gate=None))
    assert lc.circuit.detector_error_model().num_errors == 0


def test_ablation_table_is_consistent():
    for name in V.ABLATIONS:
        V.VariantSpec(f"a_{name}", ablation=name).build(SEQUENCE, 2, P_TEST)
    assert set(V.DISJOINT_ABLATIONS) <= set(V.ABLATIONS)


# ---------------------------------------------------------------------------
# Determinism, structure, decoders
# ---------------------------------------------------------------------------

VARIANT_KWARGS = [
    dict(schedule="parallel"), dict(schedule="zx"),
    dict(schedule="parallel", reset=False), dict(schedule="serial", reset=False),
    dict(schedule="zx", reset=False),
    dict(schedule="parallel", inverse_check_every=2),
    dict(schedule="parallel", reset=False, inverse_check_every=1),
    dict(schedule="parallel", forward_check_rounds=[1, 4]),
    dict(schedule="parallel", reset=False, forward_check_rounds=[2]),
]


@pytest.mark.parametrize("kwargs", VARIANT_KWARGS)
@pytest.mark.parametrize("terminal", ["data2", "data5"])
@pytest.mark.parametrize("depth", [0, 1, 6, 22])
def test_noiseless_variants_are_deterministic(kwargs, terminal, depth):
    kw = dict(kwargs)
    if "forward_check_rounds" in kw:
        kw["forward_check_rounds"] = [r for r in kw["forward_check_rounds"]
                                      if r < depth]
    lc = V.build_variant_circuit(SEQUENCE, depth, 0.0, terminal=terminal, **kw)
    dets, obs = _det_obs(lc)
    assert not dets.any() and not obs.any()
    noisy = V.build_variant_circuit(SEQUENCE, depth, 0.01, terminal=terminal,
                                    **kw)
    dem = noisy.circuit.detector_error_model()
    # The transfer-matrix MLD needs every mechanism within two adjacent slots.
    M.TransferMatrixMLD(dem, M.lrb_detector_slots(noisy))


@pytest.mark.parametrize("kwargs", [
    dict(schedule="parallel"), dict(schedule="zx"),
    dict(schedule="parallel", reset=False),
    dict(schedule="serial", reset=False)])
def test_circuit_distance_two_with_full_readout(kwargs):
    lc = V.build_variant_circuit(SEQUENCE, 4, 0.01, terminal="data5", **kwargs)
    error = lc.circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False)
    assert len(error) == 2


@pytest.mark.parametrize("kwargs,depth", [
    (dict(schedule="parallel"), 3), (dict(schedule="zx"), 2),
    (dict(schedule="parallel", reset=False), 3),
    (dict(schedule="serial", reset=False), 2),
    (dict(schedule="parallel", inverse_check_every=1), 2),
    (dict(schedule="parallel", forward_check_rounds=[1]), 3)])
@pytest.mark.parametrize("terminal", ["data2", "data5"])
def test_transfer_matrix_mld_matches_brute_force(kwargs, depth, terminal):
    lc = V.build_variant_circuit(SEQUENCE, depth, 0.02, terminal=terminal,
                                 **kwargs)
    dem = lc.circuit.detector_error_model()
    decoder = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    joint = M.brute_force_joint(dem)
    n = dem.num_detectors
    dets, _ = _det_obs(lc, shots=2000, seed=9)
    index = dets.astype(np.int64) @ (1 << np.arange(n))
    expected = np.log(joint[index]) - np.log(joint[index + (1 << n)])
    assert np.allclose(decoder.log_likelihood_ratio(dets), expected,
                       atol=1e-9)


@pytest.mark.parametrize("schedule", V.SCHEDULES)
@pytest.mark.parametrize("reset", [True, False])
def test_raw_syndromes_match_measurement_records(schedule, reset):
    lc = V.build_variant_circuit(SEQUENCE, 6, 0.02, terminal="data5",
                                 schedule=schedule, reset=reset,
                                 inverse_check_every=2)
    circuit = lc.circuit
    measurements = circuit.compile_sampler(seed=5).sample(3000)
    flips = measurements ^ circuit.reference_sample()[None, :]
    dets, _ = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)
    raw = lc.raw_syndromes(dets)
    expected = np.zeros_like(raw)
    for k, label in enumerate(lc.rounds):
        for s in range(4):
            value = np.zeros(len(flips), dtype=bool)
            for record in lc.raw_records[label][s]:
                value ^= flips[:, record]
            expected[:, k, s] = value
    assert np.array_equal(raw, expected)
    if reset:
        ancilla = flips[:, :4 * lc.num_rounds].reshape(-1, lc.num_rounds, 4)
        assert np.array_equal(raw, ancilla)


def test_parallel_no_reset_equals_reset_without_reset_noise():
    """Analytic claim: with one measurement layer, dropping the reset only
    removes the reset-layer noise (exact joint distribution check)."""
    no_reset_noise = V.NoiseConfig(
        check=Q.SI1000().scaled(reset_idle=0.0, reset_flip=0.0))
    for depth in (1, 2):
        a = V.build_variant_circuit(SEQUENCE, depth, 0.01, schedule="parallel",
                                    reset=False)
        b = V.build_variant_circuit(SEQUENCE, depth, 0.01, schedule="parallel",
                                    reset=True, noise=no_reset_noise)
        ja = M.brute_force_joint(a.circuit.detector_error_model())
        jb = M.brute_force_joint(b.circuit.detector_error_model())
        assert np.allclose(ja, jb, atol=1e-15, rtol=1e-10)


def test_serial_no_reset_differs_from_reset_ablation():
    # Post-measurement idle on ancillas persists without reset in the serial
    # schedule, so there the two models are not the same.
    no_reset_noise = V.NoiseConfig(
        check=Q.SI1000().scaled(reset_idle=0.0, reset_flip=0.0))
    a = V.build_variant_circuit(SEQUENCE, 1, 0.01, schedule="serial",
                                reset=False)
    b = V.build_variant_circuit(SEQUENCE, 1, 0.01, schedule="serial",
                                reset=True, noise=no_reset_noise)
    ja = M.brute_force_joint(a.circuit.detector_error_model())
    jb = M.brute_force_joint(b.circuit.detector_error_model())
    assert not np.allclose(ja, jb, atol=1e-12)


def test_parallel_two_qubit_readout_floor_is_smaller():
    """data2 readout: undetectable logical mass (analytic: serial ~ 4p + 8p/15,
    parallel ~ 8p/3 + 8p/15)."""
    p = 0.001
    floors = {}
    for schedule in ("serial", "parallel"):
        lc = V.build_variant_circuit(SEQUENCE, 2, p, terminal="data2",
                                     schedule=schedule)
        dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                  M.lrb_detector_slots(lc))
        floors[schedule] = dec.undetectable_logical_probability / p
    assert floors["serial"] == pytest.approx(4 + 8 / 15, rel=0.02)
    assert floors["parallel"] == pytest.approx(8 / 3 + 8 / 15, rel=0.02)


def test_first_order_coefficients():
    p = 1e-6
    serial = V.first_order_coefficients(
        V.build_variant_circuit(SEQUENCE, 0, p), p)
    # Lead's first_order.py values at depth 0 (sequence independent there).
    assert serial["E1_5"] == pytest.approx(11.067, abs=2e-3)
    assert serial["R1_5"] == pytest.approx(69.533, abs=2e-3)
    assert serial["R1_2"] == pytest.approx(58.866, abs=2e-3)
    assert serial["U1_2"] == pytest.approx(68 / 15, rel=1e-4)
    assert serial["U1_5"] == pytest.approx(0.0, abs=1e-9)
    parallel = V.first_order_coefficients(
        V.build_variant_circuit(SEQUENCE, 0, p, schedule="parallel"), p)
    assert parallel["U1_2"] == pytest.approx(48 / 15, rel=1e-4)
    # The data2 decoder sees a coarser symptom partition: never better.
    assert parallel["E1_2"] >= parallel["E1_5"] - 1e-9


@pytest.mark.parametrize("policy,k,depth", [("unif", 3, 10), ("unif", 1, 6),
                                            ("const", 3, 22), ("const", 1, 6),
                                            ("unif", 11, 22), ("unif", 4, 2)])
def test_policy_executed_circuits(policy, k, depth):
    spec = V.VariantSpec("x", schedule="parallel", forward=(policy, k))
    lc = spec.build(SEQUENCE, depth, P_TEST)
    assert lc.rounds == B.used_rounds(policy, k, depth)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_run_variant_noiseless_counts():
    spec = V.VariantSpec("smoke", schedule="parallel", all_policies=True)
    out = V.run_variant(spec, 0.0, SEQUENCE, [0, 2], 500, decode_shots=200)
    assert (out["raw__errors"] == 0).all()
    assert (out["lrbd__accepted"] == 500).all()
    assert (out["mld5__errors"] == 0).all() and (out["mld5__shots"] == 200).all()
    assert (out["mld2__errors"] == 0).all()


def test_run_variant_counts_consistent():
    spec = V.VariantSpec("smoke2", schedule="serial", all_policies=True)
    out = V.run_variant(spec, 0.01, SEQUENCE, [0, 3], 3000, decode_shots=1000,
                        batch_size=1000)
    assert (out["lrbd__accepted"] <= out["lrbd__shots"]).all()
    assert (out["lrbd_final__accepted"] <= out["lrbd__accepted"]).all()
    # Every forward round is executed, so unif=1 uses every round.
    for key in ("shots", "accepted", "errors"):
        assert np.array_equal(out[f"lrbd__{key}"],
                              out[f"lrbd_unif_1__{key}"])
    # const=1 uses only the terminal round, so it accepts at least as much.
    assert (out["lrbd_const_1__accepted"] >= out["lrbd__accepted"]).all()
    assert (out["mld5__shots"] == 1000).all()


def test_run_variant_const0():
    spec = V.VariantSpec("c0", const0=True)
    out = V.run_variant(spec, 0.01, SEQUENCE, [0, 4], 2000, decode_shots=500)
    assert "mld2__shots" not in out
    assert (out["mld5__shots"] == 500).all()
    assert out["num_rounds"].tolist() == [0, 0]
    assert out["raw__errors"][0] == 0  # depth 0: nothing noisy happens


def test_task_seeds_distinct():
    seeds = {V.task_seed(n, p, s, d) for n in ("a", "b") for p in (1e-3, 2e-3)
             for s in range(3) for d in (0, 2)}
    assert len(seeds) == 24


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _repo_fit(depths, means, stds):
    plotting = pytest.importorskip("lrb.lrb_plotting")
    cls = plotting.LRBResultsPlotter
    fake = SimpleNamespace(fit_config=plotting.LRBPlotFitConfig(),
                           _safe_weights=cls._safe_weights)
    x, y, e = cls._mask_invalid(list(depths), list(means), list(stds))
    x, y, e = cls._apply_ceiling_cut(x, y, e, fake.fit_config.y_ceiling)
    return cls._fit_decay_parameter_fixed_a(fake, x, y, e)


def test_fit_matches_repo_implementation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    rng = np.random.default_rng(4)
    depths = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22])
    cases = []
    for _ in range(12):
        f_true = rng.uniform(0.85, 0.999)
        means = 0.97 * f_true ** depths + rng.normal(0, 0.01, len(depths))
        stds = rng.uniform(0.005, 0.05, len(depths))
        cases.append((means, stds))
    means, stds = cases[0]
    cases.append((np.where(depths > 15, np.nan, means), stds))   # NaN tail
    cases.append((np.where(depths == 0, 1.02, means), stds))      # > ceiling
    cases.append((means, np.where(depths == 0, 0.0, stds)))       # zero std
    ours_f, ours_b, _ = V.fit_decay_batch(
        depths, np.array([c[0] for c in cases]), np.array([c[1] for c in cases]))
    for (m, s), f, b in zip(cases, ours_f, ours_b):
        ref = _repo_fit(depths, m, s)
        # Final grid spacing is 0.0016/2000 = 8e-7; summation order may differ.
        assert f == pytest.approx(ref["f"], abs=2e-6)
        assert b == pytest.approx(ref["b"], rel=1e-4, abs=1e-6)


def test_fit_too_few_points_is_nan():
    f, b, n = V.fit_decay_batch([0, 2, 4], np.array([[np.nan, np.nan, 0.5]]),
                                np.array([[0.1, 0.1, 0.1]]))
    assert np.isnan(f[0]) and n[0] == 1


def test_bootstrap_recovers_decay():
    rng = np.random.default_rng(7)
    depths = [0, 2, 4, 6, 10, 14, 18, 20, 22]
    f_true = 0.96
    per_seq = np.array([0.98 * (f_true + rng.normal(0, 0.004)) ** np.array(
        depths) for _ in range(20)])
    idx = V.bootstrap_indices(20, 300, np.random.default_rng(1))
    est = V.fit_with_bootstrap(depths, per_seq, idx)
    assert abs(est.f - f_true) < 4 * est.f_boot_std + 1e-3
    assert 0 < est.f_boot_std < 0.01
    assert est.f_boot_lo <= est.f <= est.f_boot_hi
