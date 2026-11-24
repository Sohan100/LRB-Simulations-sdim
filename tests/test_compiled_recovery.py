"""Opt-in compiled recovery Clifford and pooled r_L fits.

The default (mirror) path must stay byte-identical to what produced the
production runs; the compiled path must close every sequence exactly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import tempfile

import numpy as np
import pytest

from lrb.circuit_generator import (
    COMPILED_INVERSE_MODE,
    MIRROR_INVERSE_MODE,
    LRBCircuitGenerator,
)
from lrb.clifford_recovery import QUTRIT_TOKEN_UNITARIES, CliffordRecoveryTable, gate_list_unitary
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry
from lrb.dem_simulation import reference_shot
from lrb.experiment_setup import ExperimentSetupConfig, ExperimentSetupManager
from lrb.pooled_lrb_fits import fit_pooled_lrb, pseudo_threshold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sha256 over every file generate_tests writes for a seeded small grid,
# recorded with the generator as it was before inverse_mode existed.
GOLDEN_DEFAULT_TREE = {
    ("folded_qutrit", "sd6"): "6230bd42cdea03577adabf73f048ff59c2dbf1de15a3509e037a0e7048835dad",
    ("folded_qutrit_split_unif", "si1000"): "6da2dd2bd14a5ced40d07ffc883996444b553870b7c695ed1b946dea80ab7b84",
}
# The .chp circuit files are gitignored; LRB_RUNS_ROOT can point at a checkout that has them.
RUNS_ROOT = os.environ.get("LRB_RUNS_ROOT", os.path.join(REPO, "LRB-experiment-data-slurm"))
PRODUCTION_RUN = os.path.join(RUNS_ROOT, "Run-2026-10-28-10-24-46-folded_qutrit-single-ancilla-sd6-sdim140")


def make_generator(code: str, noise: str, **kwargs) -> LRBCircuitGenerator:
    profile = CodeSimulationProfileRegistry.resolve_code_profile(code)
    return LRBCircuitGenerator(code_definition=profile.code_definition, noise_model=noise, **kwargs)


def tree_hash(root: str) -> str:
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            path = os.path.join(d, f)
            h.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def generate_small_tree(gen: LRBCircuitGenerator, root: str) -> None:
    random.seed(20261031)
    gen.generate_tests(
        num_clifford_sequences=2,
        lrb_experiment_folder_path=os.path.join(root, "LRB"),
        rb_experiment_folder_path=os.path.join(root, "RB"),
        depths=[0, 1, 2, 4],
        probabilities=[0.0, 0.01],
        lrb_const0_experiment_folder_path=os.path.join(root, "LRB_const0"),
    )


@pytest.mark.parametrize("code,noise", sorted(GOLDEN_DEFAULT_TREE))
@pytest.mark.parametrize("kwargs", [{}, {"inverse_mode": MIRROR_INVERSE_MODE}])
def test_default_generation_is_unchanged(code, noise, kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        generate_small_tree(make_generator(code, noise, **kwargs), tmp)
        assert tree_hash(tmp) == GOLDEN_DEFAULT_TREE[(code, noise)]


def test_compiled_mode_consumes_the_same_random_draws():
    states = []
    for mode in (MIRROR_INVERSE_MODE, COMPILED_INVERSE_MODE):
        with tempfile.TemporaryDirectory() as tmp:
            generate_small_tree(make_generator("folded_qutrit", "sd6", inverse_mode=mode), tmp)
        states.append(random.getstate())
    assert states[0] == states[1]


def test_unknown_inverse_mode_is_rejected():
    with pytest.raises(ValueError, match="inverse mode"):
        make_generator("folded_qutrit", "sd6", inverse_mode="reverse")


@pytest.mark.production_data
@pytest.mark.skipif(not os.path.isdir(os.path.join(PRODUCTION_RUN, "experiments")),
                    reason="production circuit files not present (set LRB_RUNS_ROOT)")
def test_default_generator_reproduces_production_chp_files():
    spec = importlib.util.spec_from_file_location(
        "verify_default_circuits_unchanged", os.path.join(REPO, "scripts", "verify_default_circuits_unchanged.py"))
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    gen = verify.generator_for_run(PRODUCTION_RUN)
    assert verify.compare_sequence(PRODUCTION_RUN, gen, seq=0, p_indices=[0, 6]) == []


def test_recovery_table_inverts_random_products():
    cd = CodeSimulationProfileRegistry.resolve_code_profile("folded_qutrit").code_definition
    table = CliffordRecoveryTable(cd.clifford_strings, cd.clifford_to_gate_sequence)
    assert len(table) == 216
    rng = random.Random(3)
    for _ in range(300):
        words = [rng.choice(cd.clifford_strings) for _ in range(rng.randint(1, 12))]
        gates = [g for w in words for g in cd.clifford_to_gate_sequence(w)] + table.recovery_gates(words)
        u = gate_list_unitary(gates)
        assert np.allclose(u / u[0, 0], np.eye(3), atol=1e-9)


def test_token_unitaries_match_sdim():
    from sdim.circuit import Circuit
    from sdim.program import Program

    tokens = [t for t in QUTRIT_TOKEN_UNITARIES if t != "I"]
    rng = random.Random(5)
    h = QUTRIT_TOKEN_UNITARIES["H"]
    for _ in range(150):
        word = [rng.choice(tokens) for _ in range(rng.randint(1, 7))]
        for basis in ("Z", "X"):
            psi = np.zeros(3, dtype=complex)
            psi[0] = 1
            if basis == "X":
                psi = h @ psi
            out = gate_list_unitary(word) @ psi
            if basis == "X":
                out = h.conj().T @ out
            pred = np.abs(out) ** 2
            c = Circuit(1, 3)
            if basis == "X":
                c.add_gate("H", 0)
            for t in word:
                c.add_gate(t, 0)
            if basis == "X":
                c.add_gate("H_INV", 0)
            c.add_gate("M", 0)
            prog = Program(c)
            prog.simulate(shots=1)
            m = prog.measurement_results[0][0][0]
            assert bool(m.deterministic) == (pred.max() > 0.99)
            if m.deterministic:
                assert m.measurement_value == int(np.argmax(pred))


@pytest.mark.parametrize("code", ["folded_qutrit", "folded_qutrit_split_unif"])
def test_compiled_circuits_close_exactly(code):
    gen = make_generator(code, "sd6", inverse_mode=COMPILED_INVERSE_MODE)
    cd = gen.code_definition
    rng = random.Random(11)
    words = [rng.choice(cd.clifford_strings) for _ in range(5)]
    depths = [0, 1, 3, 5]
    lrb = gen.generate_lrb_clifford_sequence(depths=depths, with_noise=True, clifford_strings=words)
    const0 = gen.generate_lrb_const0_clifford_sequence(depths=depths, with_noise=True, clifford_strings=words)
    gen.generate_random_clifford_strings = lambda n: words[:n]
    rb = gen.generate_rb_clifford_sequence(depths=depths, with_noise=True)
    mirror = make_generator(code, "sd6").generate_lrb_clifford_sequence(
        depths=depths, with_noise=True, clifford_strings=words)
    for circ in lrb + const0 + rb:
        # A wrong recovery leaves the terminal X readout random; ten fresh
        # noiseless shots all reading 0 would then happen with prob 3^-10.
        for _ in range(10):
            _, ref = reference_shot(circ)
            assert all(v % 3 == 0 for v in ref["detectors"])
            assert all(v % 3 == 0 for v in ref["logicals"])
    # same check rounds, hence the same detector layout as the mirror circuits
    for c_comp, c_mirr in zip(lrb, mirror):
        n_det = [sum(op.gate_name == "DETECTOR" for op in c.operations) for c in (c_comp, c_mirr)]
        assert n_det[0] == n_det[1] > 0


def test_experiment_setup_records_only_non_default_inverse_mode():
    for mode, expect_file in ((MIRROR_INVERSE_MODE, False), (COMPILED_INVERSE_MODE, True)):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ExperimentSetupConfig(
                n_cliff=1, depths=[0, 1], n_shots=10, probabilities=[0.0],
                stab_checks_constant_numbers=[0, 1], stab_checks_uniform_interval_size=[1],
                home_folder=tmp, lrb_folder_name="lrb", code_name="folded_qutrit",
                noise_model="sd6", inverse_mode=mode)
            run = ExperimentSetupManager(config=cfg).run_setup(custom_name="t")
            path = os.path.join(run, "inverse_mode.txt")
            assert os.path.exists(path) == expect_file
            if expect_file:
                assert open(path).read().strip() == COMPILED_INVERSE_MODE


def _synthetic_counts(f, a, b=0.9, depths=(0, 1, 2, 3, 4, 6), n_seq=30, shots=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    tallies = np.zeros((len(depths), n_seq, 3))
    rejected = np.zeros((len(depths), n_seq))
    for i, m in enumerate(depths):
        lam = b * f ** m
        p = [(1 + 2 * lam) / 3, (1 - lam) / 3, (1 - lam) / 3]
        acc = rng.binomial(shots, 0.8 * a ** m, size=n_seq)
        for s in range(n_seq):
            tallies[i, s] = rng.multinomial(acc[s], p)
        rejected[i] = shots - acc
    return np.asarray(depths), tallies, rejected


def test_pooled_fit_recovers_known_rates():
    f, a = 0.97, 0.5
    depths, t, rj = _synthetic_counts(f, a)
    fit = fit_pooled_lrb(depths, t, rj, k=1, n_boot=50)
    assert fit.depths_used == [0, 1, 2, 3, 4, 6]
    assert abs(fit.r_L - 2 * (1 - f) / 3) < 4 * fit.r_L_se + 1e-4
    assert abs(fit.d_L - (1 - a)) < 0.01
    assert abs(fit.r_uncond - 2 * (1 - a * f) / 3) < 0.01
    assert fit.chi2_dof < 5


def test_pooled_fit_aligned_depths_and_shot_floor():
    depths, t, rj = _synthetic_counts(0.97, 0.5)
    fit = fit_pooled_lrb(depths, t, rj, k=2, n_boot=0)
    assert fit.depths_used == [0, 2, 4, 6]
    fit = fit_pooled_lrb(depths, t, rj, k=1, min_accepted=10**9, n_boot=0)
    assert fit.depths_used == [] and np.isnan(fit.r_L)


def test_pseudo_threshold_takes_last_crossing_from_below():
    ps = [0.01, 0.02, 0.03, 0.04]
    assert pseudo_threshold(ps, [0.5, 2.0, 0.5, 2.0], [1, 1, 1, 1])[0] == pytest.approx(0.035, abs=1e-9)
    assert np.isnan(pseudo_threshold(ps, [0.5] * 4, [1] * 4)[0])
