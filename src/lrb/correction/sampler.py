"""Vectorized shot sampler for extracted DEM mechanisms.

This mirrors ``lrb.correction.compact_dem.sample_detector_error_model`` (independent
locations, each active with its own probability and then applying one of its
equally likely non-zero Weyl vectors) but uses an explicit
``numpy.random.Generator``, works at any swept ``p`` without recompiling the
model, and returns compact ``int8`` arrays.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .chain_dem import Mechanism


def sample_shots(
    mechanisms: Sequence[Mechanism],
    activation: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample raw detector shifts and logical shifts.

    Args:
        mechanisms (Sequence[Mechanism]): Extracted mechanisms.
        activation (np.ndarray): Activation probability per mechanism.
        shots (int): Number of shots.
        rng (np.random.Generator): Random generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(shots, n_det)`` detector shifts and
        ``(shots,)`` logical shifts, both ``int8`` in ``{0, 1, 2}``.
    """
    n_det = mechanisms[0].detector_effects.shape[1] if mechanisms else 0
    detectors = np.zeros((shots, n_det), dtype=np.int8)
    logical = np.zeros(shots, dtype=np.int8)
    for mechanism, q in zip(mechanisms, activation):
        if q <= 0.0:
            continue
        hits = np.flatnonzero(rng.random(shots) < q)
        if hits.size == 0:
            continue
        choice = rng.integers(0, len(mechanism.vectors), size=hits.size)
        det_effect = mechanism.detector_effects[choice].astype(np.int8)
        log_effect = mechanism.logical_effects[choice].astype(np.int8)
        nonzero_cols = np.flatnonzero(np.any(mechanism.detector_effects != 0, axis=0))
        if nonzero_cols.size:
            block = detectors[np.ix_(hits, nonzero_cols)]
            detectors[np.ix_(hits, nonzero_cols)] = (block + det_effect[:, nonzero_cols]) % 3
        logical[hits] = (logical[hits] + log_effect) % 3
    return detectors, logical


def syndrome_indices(detectors: np.ndarray, detector_index: np.ndarray) -> np.ndarray:
    """
    Convert raw detector shifts to per-round syndrome indices.

    Args:
        detectors (np.ndarray): ``(shots, n_det)`` detector shifts.
        detector_index (np.ndarray): ``(rounds, width)`` detector layout.

    Returns:
        np.ndarray: ``(shots, rounds)`` integer indices in ``0..3**width-1``.
    """
    per_round = detectors[:, detector_index].astype(np.int64)
    powers = 3 ** np.arange(detector_index.shape[1], dtype=np.int64)
    return per_round @ powers
