"""Exactness and consistency tests for the chain-model decoders.

The key test builds small SI1000 folded-qutrit LRB circuits with the
repository's own generator, compiles them with the production DEM backend,
and checks the chain model against a brute-force convolution over the full
detector space: the joint distribution of (all detectors, logical shift)
must agree to machine precision, and so must the maximum-likelihood
posteriors. The remaining tests check the exact LRB-D and blip-forgiving
evaluators against sampling and the policy masks against the pipeline's own
postselection functions.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

sdim = pytest.importorskip("sdim")

from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry  # noqa: E402
from lrb.correction import chain_dem, decoders, forward, sampler, zmod3  # noqa: E402
from lrb.dem_simulation import build_compact_detector_error_model  # noqa: E402
from lrb.code_definitions import single_qutrit_cliffords_exhaustive_strings as CATALOG  # noqa: E402
from lrb.lrb_simulation import LRBSimulationPipeline  # noqa: E402


def make_circuits(depths, clifford_strings, p):
    profile = CodeSimulationProfileRegistry.resolve_code_profile("folded_qutrit")
    generator = LRBCircuitGenerator(code_definition=profile.code_definition, noise_model="si1000")
    circuits = generator.generate_lrb_clifford_sequence(
        depths=depths, with_noise=True, clifford_strings=clifford_strings)
    for circuit in circuits:
        generator.update_noise_param(circuit, p)
    return circuits


def brute_force_joint(mechanisms, activation, n_det):
    """Full-space joint distribution over (detectors..., logical) by convolution."""
    width = n_det + 1
    size = 3**width
    digits = zmod3.index_to_digits(np.arange(size), width)
    joint = np.zeros(size)
    joint[0] = 1.0
    for mechanism, q in zip(mechanisms, activation):
        effects = np.concatenate(
            [mechanism.detector_effects, mechanism.logical_effects[:, None]], axis=1) % 3
        updated = (1.0 - q) * joint
        weight = q / len(effects)
        for effect in effects:
            shifted_index = zmod3.digits_to_index((digits - effect) % 3)
            updated += weight * joint[shifted_index]
        joint = updated
    return joint.reshape((3,) * width, order="F")  # axes: det0, det1, ..., logical


@pytest.fixture(scope="module")
def depth_one_model():
    circuits = make_circuits([1], ["HP"], 0.01)
    model = build_compact_detector_error_model(circuits[0])
    mechanisms = chain_dem.extract_mechanisms(model)
    return model, mechanisms


def test_solve_linear_map_recovers_random_map():
    rng = np.random.default_rng(0)
    matrix = rng.integers(0, 3, size=(4, 4))
    x = rng.integers(0, 3, size=(30, 4))
    y = (x @ matrix.T) % 3
    solved = zmod3.solve_linear_map(x, y)
    assert solved is not None
    assert np.array_equal((x @ solved.T) % 3, y)
    assert zmod3.solve_linear_map(x, (y + np.eye(30, 4, dtype=int)) % 3) is None


def test_chain_joint_matches_brute_force(depth_one_model):
    model, mechanisms = depth_one_model
    chain = chain_dem.build_chain_model(model, 0.01, mechanisms)
    assert chain.num_rounds == 2
    n_det = model.num_detectors
    joint = brute_force_joint(mechanisms, chain.activation, n_det)
    # Every full syndrome history and logical value: compare P(s, L).
    histories = np.array(list(product(range(3), repeat=n_det)))
    syndromes = sampler.syndrome_indices(histories.astype(np.int8), chain.layout.detector_index)
    posterior, log_evidence = forward.forward_posteriors(chain, syndromes)
    chain_joint = posterior * np.exp(log_evidence)[:, None]
    brute = np.array([joint[tuple(h)] for h in histories])  # (n, 3)
    assert np.allclose(chain_joint, brute, rtol=1e-9, atol=1e-15)
    assert abs(brute.sum() - 1.0) < 1e-12


def test_shared_forward_matches_plain(depth_one_model):
    model, mechanisms = depth_one_model
    chain = chain_dem.build_chain_model(model, 0.02, mechanisms)
    rng = np.random.default_rng(1)
    det, _ = sampler.sample_shots(mechanisms, chain.activation, 3000, rng)
    syndromes = sampler.syndrome_indices(det, chain.layout.detector_index)
    for mask in (None, np.array([False, True])):
        plain, ev_plain = forward.forward_posteriors(chain, syndromes, mask)
        shared, ev_shared = forward.forward_posteriors_shared(chain, syndromes, mask)
        assert np.allclose(plain, shared, atol=1e-13)
        assert np.allclose(ev_plain, ev_shared, atol=1e-10)


def test_exact_detection_and_bfd_match_sampling():
    circuits = make_circuits([4], [CATALOG[5], CATALOG[17], CATALOG[24], CATALOG[101]], 0.004)
    model = build_compact_detector_error_model(circuits[0])
    mechanisms = chain_dem.extract_mechanisms(model)
    chain = chain_dem.build_chain_model(model, 0.004, mechanisms)
    rng = np.random.default_rng(2)
    shots = 200000
    det, logical = sampler.sample_shots(mechanisms, chain.activation, shots, rng)
    syndromes = sampler.syndrome_indices(det, chain.layout.detector_index)
    for mask in (np.ones(5, bool), decoders.const_mask(4, 1), decoders.unif_mask(4, 2)):
        exact = decoders.lrbd_exact(chain, mask)
        accepted = np.all(syndromes[:, mask] == 0, axis=1)
        sampled = np.bincount(logical[accepted], minlength=3) / shots
        assert np.allclose(exact, sampled, atol=4 * np.sqrt(exact * (1 - exact) / shots) + 1e-4)
    exact_bfd = decoders.blip_forgiving_exact(chain)
    accepted = decoders.blip_forgiving_accept(syndromes)
    sampled = np.bincount(logical[accepted], minlength=3) / shots
    assert np.allclose(exact_bfd, sampled, atol=4 * np.sqrt(exact_bfd * (1 - exact_bfd) / shots) + 1e-4)
    # BFD accepts a superset of the all-round LRB-D shots.
    assert exact_bfd.sum() >= decoders.lrbd_exact(chain, np.ones(5, bool)).sum()


def test_policy_masks_match_pipeline_postselection():
    for depth in (0, 2, 4, 6, 10, 22):
        rounds = depth + 1
        # One synthetic shot per round that fails only that round.
        record = np.ones((1, 1, rounds, rounds), dtype=np.int64)
        for r in range(rounds):
            record[0, 0, r, r] = 0
        for interval in range(1, 23):
            table = LRBSimulationPipeline.postselect_uniform_interval(record, [depth], interval)
            used = table[0, 0] == 0  # rejected exactly when its failing round is used
            assert np.array_equal(used, decoders.unif_mask(depth, interval))
        for checks in range(1, 23):
            table = LRBSimulationPipeline.postselect_constant_number(record, [depth], checks)
            used = table[0, 0] == 0
            assert np.array_equal(used, decoders.const_mask(depth, checks))
