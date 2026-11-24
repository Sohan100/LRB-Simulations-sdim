"""Grid runner: LRB-D policies and decoders evaluated on identical shots.

For one noise parameter ``p`` the runner loops over Clifford sequences and
depths, samples detection events from the ``data5`` circuit (all five data
qubits read out ideally at the end; the extra readout does not perturb the
check rounds or ``x0 + x3``), and scores every method on the same shots:

``lrbd_{const,unif}_k``
    The qutrit pipeline's postselection on the raw check-round syndromes,
    ignoring the extra data readout (``data2`` semantics).
``lrbd_{const,unif}_k_final``
    The same policies, additionally requiring the two final X-stabilizer
    detectors from the full data readout to be clean.
``mld5`` / ``mld2``
    Exact maximum-likelihood correction using every check round, with
    (``mld5``) or without (``mld2``) the final data detectors. No rejection.
``const0`` / ``const0_mld``
    The separate no-check circuit, postselected on the ideal terminal X
    stabilizers (as in the qutrit pipeline) or decoded instead.

Soft outputs (``|LLR|``) of the MLD variants are histogrammed separately for
correct and wrong decisions, so acceptance-vs-error (Pareto) curves for any
gap threshold can be rebuilt later without re-sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import baseline as B
from . import circuits as Q
from . import mld as M

GAP_EDGES = np.concatenate([np.arange(0.0, 40.0, 0.25), [np.inf]])


def clifford_sequences(num_sequences: int, max_depth: int,
                       base_seed: int) -> list[list[str]]:
    """Reproducible random sequences, one RNG stream per sequence index."""
    return [Q.random_clifford_sequence(max_depth,
                                       np.random.default_rng(base_seed + i))
            for i in range(num_sequences)]


def policy_names(const_checks: Sequence[int], unif_checks: Sequence[int]
                 ) -> list[tuple[str, str, int]]:
    names = [(f"lrbd_const_{k}", "const", k) for k in const_checks if k > 0]
    names += [(f"lrbd_unif_{k}", "unif", k) for k in unif_checks]
    return names


@dataclass
class PointResult:
    """All counts for one noise parameter."""

    p: float
    depths: list[int]
    num_sequences: int
    tables: dict[str, B.CountTable] = field(default_factory=dict)
    gap_hist: dict[str, np.ndarray] = field(default_factory=dict)

    def table(self, name: str) -> B.CountTable:
        if name not in self.tables:
            self.tables[name] = B.CountTable.zeros(self.num_sequences,
                                                   len(self.depths))
        return self.tables[name]

    def hist(self, name: str) -> np.ndarray:
        if name not in self.gap_hist:
            self.gap_hist[name] = np.zeros(
                (self.num_sequences, len(self.depths), len(GAP_EDGES) - 1, 2),
                dtype=np.int64)
        return self.gap_hist[name]

    def to_npz(self, path: str) -> None:
        arrays = {"p": np.array(self.p), "depths": np.array(self.depths)}
        for name, table in self.tables.items():
            arrays[f"{name}__shots"] = table.shots
            arrays[f"{name}__accepted"] = table.accepted
            arrays[f"{name}__errors"] = table.errors
        for name, hist in self.gap_hist.items():
            arrays[f"gap__{name}"] = hist
        arrays["gap_edges"] = GAP_EDGES
        np.savez_compressed(path, **arrays)

    @classmethod
    def from_npz(cls, path: str) -> "PointResult":
        data = np.load(path)
        depths = [int(d) for d in data["depths"]]
        names = sorted({key.split("__")[0] for key in data.files
                        if key.endswith("__shots")})
        first = data[f"{names[0]}__shots"]
        result = cls(float(data["p"]), depths, first.shape[0])
        for name in names:
            result.tables[name] = B.CountTable(
                data[f"{name}__shots"], data[f"{name}__accepted"],
                data[f"{name}__errors"])
        for key in data.files:
            if key.startswith("gap__"):
                result.gap_hist[key[5:]] = data[key]
        return result


def _histogram_gap(hist: np.ndarray, gap: np.ndarray, wrong: np.ndarray
                   ) -> None:
    bins = np.clip(np.searchsorted(GAP_EDGES, gap, side="right") - 1, 0,
                   len(GAP_EDGES) - 2)
    np.add.at(hist, (bins, wrong.astype(np.int64)), 1)


def run_point(p: float,
              sequences: Sequence[Sequence[str]],
              depths: Sequence[int],
              shots: int,
              *,
              const_checks: Sequence[int] = tuple(range(0, 23)),
              unif_checks: Sequence[int] = tuple(range(1, 23)),
              final_variant_policies: Sequence[tuple[str, int]] = (
                  ("unif", 1), ("const", 1)),
              decode_shots: Optional[int] = None,
              batch_size: int = 100_000,
              seed: int = 12345,
              noise: Q.SI1000 = Q.SI1000()) -> PointResult:
    """Evaluate all methods at one ``p``.

    Args:
        decode_shots: Shots (per circuit) that are also decoded; ``None``
            decodes all of them. LRB-D statistics always use all ``shots``.
    """
    depths = list(depths)
    result = PointResult(p, depths, len(sequences))
    policies = policy_names(const_checks, unif_checks)
    decode_shots = shots if decode_shots is None else min(decode_shots, shots)
    for seq_index, sequence in enumerate(sequences):
        for d_index, depth in enumerate(depths):
            lc5 = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                                      noise=noise)
            lc2 = Q.build_lrb_circuit(sequence, depth, p, terminal="data2",
                                      noise=noise)
            mld5 = M.TransferMatrixMLD(lc5.circuit.detector_error_model(),
                                       M.lrb_detector_slots(lc5))
            mld2 = M.TransferMatrixMLD(lc2.circuit.detector_error_model(),
                                       M.lrb_detector_slots(lc2))
            n_round_dets = 4 * lc5.num_rounds
            sampler = lc5.circuit.compile_detector_sampler(
                seed=seed + 1000 * seq_index + d_index)
            done = 0
            while done < shots:
                n = min(batch_size, shots - done)
                dets, obs = sampler.sample(n, separate_observables=True)
                flip = obs[:, 0]
                raw = lc5.raw_syndromes(dets)
                final_clean = ~dets[:, n_round_dets:].any(axis=1)
                for name, policy, k in policies:
                    rounds = B.used_rounds(policy, k, depth)
                    acc = B.accept_mask(raw, lc5.rounds, rounds)
                    result.table(name).add(seq_index, d_index, n,
                                           int(acc.sum()),
                                           int((acc & flip).sum()))
                    if (policy, k) in final_variant_policies:
                        acc_f = acc & final_clean
                        result.table(name + "_final").add(
                            seq_index, d_index, n, int(acc_f.sum()),
                            int((acc_f & flip).sum()))
                n_dec = min(n, max(decode_shots - done, 0))
                if n_dec > 0:
                    for name, decoder, columns in (
                            ("mld5", mld5, slice(None)),
                            ("mld2", mld2, slice(0, n_round_dets))):
                        llr = decoder.log_likelihood_ratio(
                            dets[:n_dec, columns])
                        wrong = (llr < 0) != flip[:n_dec]
                        result.table(name).add(seq_index, d_index, n_dec,
                                               n_dec, int(wrong.sum()))
                        _histogram_gap(result.hist(name)[seq_index, d_index],
                                       np.abs(llr), wrong)
                done += n

            if 0 in const_checks:
                lc0 = Q.build_lrb_const0_circuit(sequence, depth, p,
                                                 noise=noise)
                dec0 = M.TransferMatrixMLD(lc0.circuit.detector_error_model(),
                                           M.lrb_detector_slots(lc0))
                sampler0 = lc0.circuit.compile_detector_sampler(
                    seed=seed + 7 + 1000 * seq_index + d_index)
                done = 0
                while done < shots:
                    n = min(batch_size, shots - done)
                    dets, obs = sampler0.sample(n, separate_observables=True)
                    flip = obs[:, 0]
                    acc = ~dets.any(axis=1)
                    result.table("const0").add(seq_index, d_index, n,
                                               int(acc.sum()),
                                               int((acc & flip).sum()))
                    if dets.shape[1] > 0:
                        llr = dec0.log_likelihood_ratio(dets)
                    else:
                        llr = np.full(n, np.inf)
                    wrong = (llr < 0) != flip
                    result.table("const0_mld").add(seq_index, d_index, n, n,
                                                   int(wrong.sum()))
                    _histogram_gap(result.hist("const0_mld")[seq_index,
                                                             d_index],
                                   np.abs(llr), wrong)
                    done += n
    return result


def run_physical_rb(p_rb: float, sequences: Sequence[Sequence[str]],
                    depths: Sequence[int], shots: int,
                    seed: int = 999) -> B.CountTable:
    """Matched single-qubit RB counts (no rejection)."""
    table = B.CountTable.zeros(len(sequences), len(depths))
    for seq_index, sequence in enumerate(sequences):
        for d_index, depth in enumerate(depths):
            circuit = Q.build_physical_rb_circuit(sequence, depth, p_rb)
            sampler = circuit.compile_detector_sampler(
                seed=seed + 1000 * seq_index + d_index)
            _, obs = sampler.sample(shots, separate_observables=True)
            table.add(seq_index, d_index, shots, shots, int(obs[:, 0].sum()))
    return table
