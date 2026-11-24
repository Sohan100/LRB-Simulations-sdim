"""Tests for protocol-variant circuits and the terminal-readout chain model."""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

sdim = pytest.importorskip("sdim")

from lrb.code_definitions import single_qutrit_cliffords_exhaustive_strings as CATALOG  # noqa: E402
from lrb.correction import chain_dem, decoders, forward, sampler, variants, zmod3  # noqa: E402
from lrb.dem_simulation import build_compact_detector_error_model  # noqa: E402

STRINGS = [CATALOG[5], CATALOG[17], CATALOG[24], CATALOG[101], CATALOG[150]]


def ops(circuit):
    out = []
    for op in circuit.operations:
        params = {} if op.params is None else {k: str(v) for k, v in op.params.items()}
        out.append((op.gate_name, op.qudit_index, op.target_index, tuple(sorted(params.items()))))
    return out


def test_flags_off_matches_production_generator():
    generator, _code = variants.folded_generator()
    depths = [0, 2, 3]
    reference = generator.generate_lrb_clifford_sequence(
        depths=depths, with_noise=True, clifford_strings=STRINGS[:3])
    for depth, ref in zip(depths, reference):
        built = variants.build_variant_circuit(STRINGS[:3], depth, generator=generator)
        assert ops(built) == ops(ref)


@pytest.mark.parametrize("flags", [
    {"checked_inverse": True},
    {"parallel_measurement": True},
    {"full_terminal_readout": True},
    {"checked_inverse": True, "parallel_measurement": True, "full_terminal_readout": True},
])
def test_variant_chain_matches_sampling(flags):
    circuit = variants.build_variant_circuit(STRINGS[:3], 3, p=0.004, **flags)
    model = build_compact_detector_error_model(circuit)
    structure = chain_dem.build_chain_structure(model)
    expected_rounds = 6 if flags.get("checked_inverse") else 4
    assert structure.layout.num_rounds == expected_rounds
    assert (structure.layout.terminal_index is not None) == bool(flags.get("full_terminal_readout"))
    chain = structure.at(0.004)
    rng = np.random.default_rng(5)
    shots = 150000
    det, logical = sampler.sample_shots(chain.mechanisms, chain.activation, shots, rng)
    syndromes = sampler.syndrome_indices(det, chain.layout.detector_index)
    accepted = np.all(syndromes == 0, axis=1)
    if chain.has_terminal:
        terminal = zmod3.digits_to_index(det[:, chain.layout.terminal_index].astype(np.int64))
        accepted &= terminal == 0
    exact = decoders.lrbd_exact(chain, np.ones(chain.num_rounds, bool))
    sampled = np.bincount(logical[accepted], minlength=3) / shots
    assert np.allclose(exact, sampled, atol=4 * np.sqrt(exact * (1 - exact) / shots) + 1e-4)


def test_terminal_chain_exact_on_sparse_noise():
    """Brute-force check of the terminal-readout chain on a sparse noise subset."""
    circuit = variants.build_variant_circuit(STRINGS[:2], 1, p=0.05, full_terminal_readout=True)
    model = build_compact_detector_error_model(circuit)
    structure = chain_dem.build_chain_structure(model)
    rng = np.random.default_rng(11)
    active = np.zeros(len(structure.mechanisms), dtype=bool)
    # Keep a few mechanisms from every slot so every code path is exercised.
    for slot in range(structure.layout.num_rounds):
        candidates = np.flatnonzero(structure.slots == slot)
        active[rng.choice(candidates, size=min(8, candidates.size), replace=False)] = True
    activation = np.where(active, [m.activation_probability(0.05) for m in structure.mechanisms], 0.0)
    chain = structure.at(0.05)
    chain.activation = activation
    chain.kernels = chain_dem.build_kernels(
        structure.mechanisms, structure.slots, structure.ab, activation,
        structure.layout.num_rounds, chain_dem.TERMINAL_WIDTH)
    n_det = model.num_detectors
    width = n_det + 1
    size = 3**width
    digits = zmod3.index_to_digits(np.arange(size), width)
    joint = np.zeros(size)
    joint[0] = 1.0
    for mechanism, q in zip(structure.mechanisms, activation):
        if q == 0:
            continue
        effects = np.concatenate([mechanism.detector_effects, mechanism.logical_effects[:, None]], axis=1) % 3
        updated = (1 - q) * joint
        for effect in effects:
            updated += q / len(effects) * joint[zmod3.digits_to_index((digits - effect) % 3)]
        joint = updated
    joint = joint.reshape((3,) * width, order="F")
    histories = np.array(list(product(range(3), repeat=n_det)))
    syndromes = sampler.syndrome_indices(histories.astype(np.int8), chain.layout.detector_index)
    terminal = zmod3.digits_to_index(histories[:, chain.layout.terminal_index])
    posterior, log_evidence = forward.forward_posteriors(chain, syndromes, terminal=terminal)
    chain_joint = posterior * np.exp(log_evidence)[:, None]
    brute = np.array([joint[tuple(h)] for h in histories])
    assert np.allclose(chain_joint, brute, rtol=1e-8, atol=1e-14)
    shared, ev_shared = forward.forward_posteriors_shared(chain, syndromes, terminal=terminal)
    assert np.allclose(shared, posterior, atol=1e-12)
