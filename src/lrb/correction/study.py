"""Per-circuit evaluation of detection, correction and hybrid decoders.

``prepare_circuit`` compiles one baseline circuit into a reusable chain
structure. ``evaluate_chain`` computes, at one noise strength:

* exact LRB-D results for every ``unif k`` and ``const c`` policy;
* exact blip-forgiving detection (BFD), observing all rounds or only the
  rounds of a policy;
* sampled results for maximum-likelihood correction (observing all rounds or
  a policy's rounds), hard-decision sliding-window decoders, and the soft
  confidence histogram from which every posterior-threshold rule follows.

Sampled decoders are scored with a Rao-Blackwellized estimator: for each
sampled syndrome history the exact posterior over the logical shift given all
rounds is known, so the expected residual distribution of any decoder output
is averaged instead of the single realized residual. This is unbiased (the
decoder model equals the simulation model) and has far lower variance at
small ``p``. Realized residual counts are stored as a cross-check.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from . import chain_dem, decoders, forward, sampler, zmod3

MAX_POLICY = 22
CONF_EDGES = np.linspace(-12.0, 0.0, 241)


@dataclass
class PreparedCircuit:
    """Chain structure of one ordinary circuit and the const0 mechanisms."""

    depth: int
    structure: chain_dem.ChainStructure
    const0_mechanisms: list | None
    const0_layout: chain_dem.DetectorLayout | None


def prepare_circuit(lrb_path: str, const0_path: str | None, depth: int,
                    p: float | None = None) -> PreparedCircuit:
    """
    Compile one baseline circuit (and its const0 sibling) into structures.

    Args:
        lrb_path (str): Path of the ordinary LRB ``.chp`` circuit.
        const0_path (str | None): Path of the matching const0 circuit.
        depth (int): Benchmark depth of the circuit.
        p (float | None): Noise strength the files were written at. Files written by
            sdim 1.4.0 need it to recover each gate's ``prob_scale`` (see
            ``compact_dem.read_run_circuit``); older files store the scales.

    Returns:
        PreparedCircuit: Reusable structures.
    """
    from .compact_dem import build_compact_detector_error_model, read_run_circuit

    model = build_compact_detector_error_model(read_run_circuit(lrb_path, p))
    structure = chain_dem.build_chain_structure(model)
    if structure.layout.num_rounds != depth + 1:
        raise ValueError(f"{lrb_path}: expected {depth + 1} rounds")
    const0_mechanisms = const0_layout = None
    if const0_path is not None:
        model0 = build_compact_detector_error_model(read_run_circuit(const0_path, p))
        const0_mechanisms = chain_dem.extract_mechanisms(model0)
        const0_layout = chain_dem.detector_layout(model0.detector_labels)
    return PreparedCircuit(depth, structure, const0_mechanisms, const0_layout)


def save_prepared(prepared: PreparedCircuit, path: str) -> None:
    with open(path, "wb") as handle:
        pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_prepared(path: str) -> PreparedCircuit:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def const0_joint(prepared: PreparedCircuit, p: float) -> np.ndarray:
    """Exact ``(3, 9)`` joint of logical shift and terminal X-syndrome."""
    return chain_dem.const0_joint_from_mechanisms(
        prepared.const0_mechanisms, prepared.const0_layout, p)


def _expected_residual(post_all: np.ndarray, l_hat: np.ndarray) -> np.ndarray:
    """Sum over shots of ``P(L = l_hat + k | all rounds)`` for k = 0, 1, 2."""
    rows = np.arange(len(l_hat))
    return np.array([post_all[rows, (l_hat + k) % 3].sum() for k in range(3)])


def evaluate_chain(
    prepared: PreparedCircuit,
    p: float,
    shots: int,
    rng: np.random.Generator | None,
    windows: tuple[int, ...] = (0, 1, 2),
    restricted_unif: tuple[int, ...] = (2, 4, 22),
    run_restricted: bool = True,
) -> dict[str, np.ndarray]:
    """
    Evaluate all strategies for one circuit at one noise strength.

    Policy sweeps (``unif k``, ``const c``) are evaluated only for circuits
    with the baseline round structure (``depth + 1`` rounds). Variant
    circuits get the all-round rules. When the circuit has terminal
    X-stabilizer detectors every rule may use them (``*_noterm`` entries
    ignore them for comparison).

    Args:
        prepared (PreparedCircuit): Prepared circuit.
        p (float): Swept noise parameter.
        shots (int): Sampled shots (0 skips the sampled decoders).
        rng (np.random.Generator | None): Random generator for sampling.
        windows (tuple[int, ...]): Lookaheads of the hard-decision decoders.
        restricted_unif (tuple[int, ...]): ``unif k`` policies whose rounds a
            restricted MLD decoder may observe (``const 1`` information, the
            terminal round only, is always included).
        run_restricted (bool): Run the restricted-observation MLD decoders.

    Returns:
        dict[str, np.ndarray]: Named result arrays (see module docstring).
    """
    depth = prepared.depth
    chain = prepared.structure.at(p)
    rounds = chain.num_rounds
    policy_sweep = rounds == depth + 1
    cache = decoders._SlotMatrices(chain)
    all_rounds = np.ones(rounds, dtype=bool)
    out: dict[str, np.ndarray] = {}
    out["lrbd_all"] = decoders.lrbd_exact(chain, all_rounds, cache)
    xtriv = [0] * (rounds - 1) + [decoders.X_PART_TRIVIAL]
    out["lrbd_xtriv"] = forward.constrained_joint(chain, xtriv, cache.terminal)
    out["bfd_all"] = decoders.blip_forgiving_exact(chain, None, cache)
    out["bfd_xtriv"] = decoders.blip_forgiving_exact(chain, None, cache, final_rule="xtriv")
    if chain.has_terminal:
        bare = decoders._SlotMatrices(chain, use_terminal=False)
        out["lrbd_all_noterm"] = decoders.lrbd_exact(chain, all_rounds, bare)
        out["bfd_all_noterm"] = decoders.blip_forgiving_exact(chain, None, bare)
    if policy_sweep:
        out["lrbd_unif"] = np.array([
            decoders.lrbd_exact(chain, decoders.unif_mask(depth, k), cache)
            for k in range(1, MAX_POLICY + 1)])
        out["lrbd_const"] = np.array([
            decoders.lrbd_exact(chain, decoders.const_mask(depth, c), cache)
            for c in range(1, MAX_POLICY + 1)])
        out["bfd_unif"] = np.array([
            decoders.blip_forgiving_exact(chain, decoders.unif_mask(depth, k), cache)
            for k in range(1, MAX_POLICY + 1)])
        out["bfd_const"] = np.array([
            decoders.blip_forgiving_exact(chain, decoders.const_mask(depth, c), cache)
            for c in range(1, MAX_POLICY + 1)])
    if prepared.const0_mechanisms is not None:
        out["const0_joint"] = const0_joint(prepared, p)
    if shots <= 0:
        return out

    det, logical = sampler.sample_shots(chain.mechanisms, chain.activation, shots, rng)
    syndromes = sampler.syndrome_indices(det, chain.layout.detector_index)
    terminal = (zmod3.digits_to_index(det[:, chain.layout.terminal_index].astype(np.int64))
                if chain.has_terminal else None)
    logical = logical.astype(np.int64)
    out["shots"] = np.array(shots)
    out["raw_counts"] = np.bincount(logical, minlength=3)

    post_all, log_evidence = forward.forward_posteriors_shared(chain, syndromes, None, terminal)
    l_mld = post_all.argmax(axis=1)
    confidence = post_all[np.arange(shots), l_mld]
    out["mld_all_rb"] = _expected_residual(post_all, l_mld)
    out["mld_all_counts"] = np.bincount((logical - l_mld) % 3, minlength=3)
    residual_rb = np.stack([post_all[np.arange(shots), (l_mld + k) % 3] for k in range(3)], axis=1)
    bins = np.searchsorted(CONF_EDGES, np.log10(np.clip(1.0 - confidence, 1e-300, None)), side="right")
    hist_rb = np.zeros((len(CONF_EDGES) + 1, 3))
    np.add.at(hist_rb, bins, residual_rb)
    out["soft_hist_rb"] = hist_rb
    out["soft_hist_counts"] = decoders.confidence_histogram(confidence, (logical - l_mld) % 3, CONF_EDGES)

    accept_bfd = decoders.blip_forgiving_accept(syndromes, None, terminal)
    out["bfd_sampled_rb"] = _expected_residual(post_all[accept_bfd], np.zeros(int(accept_bfd.sum()), dtype=np.int64))
    out["bfd_sampled_accepted"] = np.array(int(accept_bfd.sum()))

    if chain.has_terminal:
        post_bare, _ = forward.forward_posteriors_shared(chain, syndromes)
        l_bare = post_bare.argmax(axis=1)
        out["mld_noterm_rb"] = _expected_residual(post_all, l_bare)
    masks = {}
    if policy_sweep and run_restricted:
        masks["terminal"] = decoders.const_mask(depth, 1)
        for k in restricted_unif:
            masks[f"unif{k}"] = decoders.unif_mask(depth, k)
    for name, mask in masks.items():
        post_obs, _ = forward.forward_posteriors_shared(chain, syndromes, mask, terminal)
        l_obs = post_obs.argmax(axis=1)
        out[f"mld_{name}_rb"] = _expected_residual(post_all, l_obs)
        out[f"mld_{name}_counts"] = np.bincount((logical - l_obs) % 3, minlength=3)

    for lookahead in windows:
        l_win = decoders.window_decode(chain, syndromes, lookahead, terminal)
        out[f"win{lookahead}_rb"] = _expected_residual(post_all, l_win)
        out[f"win{lookahead}_counts"] = np.bincount((logical - l_win) % 3, minlength=3)
    out["mean_log_evidence"] = np.array(log_evidence.mean())
    return out
