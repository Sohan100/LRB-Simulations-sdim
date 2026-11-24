"""LRB-D postselection policies and statistics, mirroring the qutrit pipeline.

``used_rounds`` reproduces ``LRBSimulationPipeline.postselect_constant_number``
and ``postselect_uniform_interval`` from ``src/lrb/lrb_simulation.py``: given
depth ``m`` and a policy, it returns the check-round indices (``0..m``, with
``m`` the terminal check) whose outcomes the policy uses. A shot is accepted
when every stabilizer measurement in those rounds equals its noiseless value.

The figure of merit matches ``lrb_counts_to_statistics`` for dimension 2:
``F = |P(0) - P(1)|`` over accepted shots, averaged over Clifford sequences.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def used_rounds(policy: str, k: int, depth: int) -> list[int]:
    """Check rounds used for postselection by ``const=k`` or ``unif=k``.

    ``const=0`` is a separate circuit with no check rounds and is not handled
    here.
    """
    rounds = [depth]
    if policy == "const":
        if k <= 0:
            raise ValueError("const=0 uses a different circuit.")
        if k <= depth:
            interval = depth // k
            rounds += [j * interval - 1 for j in range(1, k)]
    elif policy == "unif":
        if k <= 0:
            raise ValueError("unif requires k >= 1.")
        if depth >= k:
            for j in range(1, depth // k + 1):
                index = j * k - 1
                if index < depth:
                    rounds.append(index)
    else:
        raise ValueError(f"Unknown policy {policy!r}.")
    return sorted(set(rounds))


def accept_mask(raw: np.ndarray, round_labels: Sequence[int],
                rounds: Iterable[int]) -> np.ndarray:
    """Shots whose raw syndromes vanish in all ``rounds``.

    Args:
        raw: ``(shots, num_rounds, 4)`` raw syndrome flips from
            ``LRBCircuit.raw_syndromes``.
        round_labels: ``LRBCircuit.rounds`` (labels of the raw columns).
        rounds: Round labels that must be clean.
    """
    columns = [list(round_labels).index(r) for r in rounds]
    return ~raw[:, columns, :].any(axis=(1, 2))


def fidelity_from_counts(accepted: np.ndarray, errors: np.ndarray
                         ) -> np.ndarray:
    """``|1 - 2 * errors / accepted|``; NaN where nothing was accepted."""
    accepted = np.asarray(accepted, dtype=float)
    errors = np.asarray(errors, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(1.0 - 2.0 * errors / accepted)


@dataclass
class CountTable:
    """Per-sequence, per-depth counts for one method at one probability.

    Arrays have shape ``(num_sequences, num_depths)``.
    """

    shots: np.ndarray
    accepted: np.ndarray
    errors: np.ndarray

    @classmethod
    def zeros(cls, num_sequences: int, num_depths: int) -> "CountTable":
        shape = (num_sequences, num_depths)
        return cls(np.zeros(shape, np.int64), np.zeros(shape, np.int64),
                   np.zeros(shape, np.int64))

    def add(self, seq: int, depth_index: int, shots: int, accepted: int,
            errors: int) -> None:
        self.shots[seq, depth_index] += shots
        self.accepted[seq, depth_index] += accepted
        self.errors[seq, depth_index] += errors

    def fidelity(self) -> np.ndarray:
        return fidelity_from_counts(self.accepted, self.errors)

    def rejection(self) -> np.ndarray:
        return 1.0 - self.accepted / np.maximum(self.shots, 1)

    def summary(self) -> dict[str, np.ndarray]:
        """Mean and standard deviation over sequences, as in the repo CSVs."""
        fid = self.fidelity()
        rej = self.rejection()
        with np.errstate(invalid="ignore"):
            return {
                "fidelity_mean": np.mean(fid, axis=0),
                "fidelity_std": np.std(fid, axis=0),
                "rejection_mean": np.mean(rej, axis=0),
                "rejection_std": np.std(rej, axis=0),
            }


def write_repo_stats(path: str, probability: float,
                     table: CountTable) -> None:
    """Write the repo's five-row CSV layout (``write_stats``)."""
    summary = table.summary()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Probability", probability])
        writer.writerow(["Fidelity averages", *summary["fidelity_mean"]])
        writer.writerow(["Fidelity Standard Deviations",
                         *summary["fidelity_std"]])
        writer.writerow(["Rejected Runs", *summary["rejection_mean"]])
        writer.writerow(["Rejected Standard Deviations",
                         *summary["rejection_std"]])
