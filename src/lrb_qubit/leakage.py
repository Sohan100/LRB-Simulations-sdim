"""Leakage model and a batched shot-by-shot leakage simulator (WP I).

The simulator is a Pauli-frame simulator written in numpy (one boolean frame
per qubit and shot), stepping through an ordinary SI1000 LRB circuit from
``circuits.build_lrb_circuit`` instruction by instruction, with a per-shot,
per-qubit leakage flag kept on the side. It follows Stim's frame-simulator
conventions: frames are relative to Stim's noiseless ``reference_sample``,
and the Z frame of a qubit is randomized at initialization, at every reset
and after every measurement, so non-deterministic outcomes are sampled
correctly (needed for the fixed-value readout of a leaked data qubit). Noise
channels are sampled exactly (Bernoulli positions from geometric gaps).
``stim.FlipSimulator`` was tried first; its ``broadcast_pauli_errors`` costs
about 10 ms per call at 5e4 shots, which made per-gate leakage bookkeeping
30x slower than this simulator. With ``rate = 0`` the simulator reproduces
Stim's sampling distribution (tested).

Model (``LeakageModel``):

* **Leakage events.** After every *noisy* two-qubit gate (check CNOTs and the
  logical SWAP/CZ; the ideal encoder never leaks), each of its two qubits
  leaks independently with probability ``rate * p`` (``rate`` is in units of
  the SI1000 ``p``).
* **Leaked-qubit action.** Whenever a two-qubit gate involves a leaked qubit
  (including the gate during which it leaks), its non-leaked partner gets a
  uniformly random Pauli (X and Z frame bits each flipped with probability
  1/2). Gates on the leaked qubit itself do nothing observable.
* **Readout.** A leaked qubit reads out as the fixed value ``readout_value``.
  With three-state readout the outcome is also flagged ("L"); the flags are
  returned, and what to do with them is a policy or decoder choice.
* **Return to the qubit subspace.** The ancilla reset returns leaked ancillas
  (``reset_returns``). A leakage-reduction unit (``lru``) acts on the data
  qubits at the start of every forward round's reset layer and returns each
  leaked qubit with probability ``lru_efficiency``; ``lru_herald`` makes it
  report which qubits were leaked (a leakage-detection unit). ``seepage`` is
  a per-round spontaneous return probability. A returned qubit is in the
  maximally mixed state (random Pauli frame). Without LRU or seepage a leaked
  data qubit stays leaked until the terminal readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import stim

from . import code as C

_TWO_QUBIT = ("CX", "CZ", "SWAP")


@dataclass(frozen=True)
class LeakageModel:
    """Leakage parameters (see module docstring)."""

    rate: float = 0.1
    lru: bool = False
    lru_efficiency: float = 1.0
    lru_herald: bool = True
    reset_returns: bool = True
    seepage: float = 0.0
    readout_value: int = 1


@dataclass
class LeakageSample:
    """Output of ``simulate_leakage``.

    Attributes:
        dets: ``(shots, num_detectors)`` detection events (lead layout).
        obs: ``(shots,)`` logical flips.
        ancilla_flags: ``(shots, num_rounds, 4)``: the ancilla of stabilizer
            ``s`` was leaked when measured in round ``k`` (three-state
            readout).
        lru_flags: ``(shots, num_rounds - 1, 5)``: data qubit leaked when the
            LRU after forward round ``k`` ran (with ``lru`` and ``lru_herald``;
            otherwise an empty array).
        terminal_flags: ``(shots, 5)``: data qubit leaked at the terminal
            readout (three-state readout of the data).
        leaked_data: Fraction of shots with at least one leaked data qubit just
            before each forward reset layer (before the LRU acts) and at the
            terminal readout.
        leak_events: Total number of leakage events (diagnostic).
    """

    dets: np.ndarray
    obs: np.ndarray
    ancilla_flags: np.ndarray
    lru_flags: np.ndarray
    terminal_flags: np.ndarray
    leaked_data: np.ndarray
    leak_events: int

    def any_flag(self, use_lru: bool = True) -> np.ndarray:
        """Shots with any leakage flag (ancilla, terminal and LRU flags)."""
        flags = (self.ancilla_flags.any(axis=(1, 2))
                 | self.terminal_flags.any(axis=1))
        if use_lru and self.lru_flags.size:
            flags |= self.lru_flags.any(axis=(1, 2))
        return flags


def _bernoulli_indices(rng: np.random.Generator, n: int, q: float) -> np.ndarray:
    """Exact positions of a Bernoulli(q) process on ``range(n)``."""
    if q <= 0.0 or n == 0:
        return np.zeros(0, dtype=np.int64)
    if q >= 1.0:
        return np.arange(n)
    expected = n * q
    size = int(expected + 6.0 * np.sqrt(expected) + 16)
    out = []
    position = -1
    while True:
        gaps = rng.geometric(q, size=size)
        positions = position + np.cumsum(gaps)
        if positions[-1] >= n:
            out.append(positions[positions < n])
            break
        out.append(positions)
        position = int(positions[-1])
    return np.concatenate(out)


def _noisy_two_qubit_instructions(circuit: stim.Circuit) -> set[int]:
    """Indices of two-qubit gates followed by their DEPOLARIZE2 channel."""
    noisy = set()
    instructions = list(circuit)
    for index, inst in enumerate(instructions[:-1]):
        if inst.name in _TWO_QUBIT:
            nxt = instructions[index + 1]
            if (nxt.name == "DEPOLARIZE2"
                    and [t.value for t in nxt.targets_copy()]
                    == [t.value for t in inst.targets_copy()]):
                noisy.add(index)
    return noisy


class _FrameSimulator:
    """Minimal Pauli-frame simulator for the gate set of the LRB circuits."""

    def __init__(self, num_qubits: int, shots: int, rng: np.random.Generator):
        self.rng = rng
        self.shots = shots
        self.x = np.zeros((num_qubits, shots), dtype=bool)
        self.z = rng.random((num_qubits, shots)) < 0.5
        self.records: list[np.ndarray] = []

    def random_bits(self) -> np.ndarray:
        return self.rng.random(self.shots) < 0.5

    def randomize(self, qubit: int, which: np.ndarray) -> None:
        """Uniformly random Pauli on ``qubit`` in the shots ``which``."""
        k = int(which.sum())
        if k == 0:
            return
        rows = np.flatnonzero(which)
        self.x[qubit, rows] ^= self.rng.random(k) < 0.5
        self.z[qubit, rows] ^= self.rng.random(k) < 0.5

    def gate(self, name: str, targets: list[int]) -> None:
        x, z = self.x, self.z
        if name in ("I", "X", "Y", "Z"):
            return
        if name in ("H",):
            for q in targets:
                x[q], z[q] = z[q].copy(), x[q].copy()
            return
        if name in ("S", "S_DAG"):
            for q in targets:
                z[q] ^= x[q]
            return
        pairs = zip(targets[::2], targets[1::2])
        if name == "CX":
            for c, t in pairs:
                x[t] ^= x[c]
                z[c] ^= z[t]
            return
        if name == "CZ":
            for a, b in pairs:
                z[b] ^= x[a]
                z[a] ^= x[b]
            return
        if name == "SWAP":
            for a, b in pairs:
                x[[a, b]] = x[[b, a]]
                z[[a, b]] = z[[b, a]]
            return
        raise ValueError(f"Unsupported gate {name!r}.")

    def noise(self, name: str, targets: list[int], prob: float) -> None:
        rng, n = self.rng, self.shots
        if name == "X_ERROR":
            for q in targets:
                self.x[q, _bernoulli_indices(rng, n, prob)] ^= True
            return
        if name == "Z_ERROR":
            for q in targets:
                self.z[q, _bernoulli_indices(rng, n, prob)] ^= True
            return
        if name == "DEPOLARIZE1":
            for q in targets:
                hits = _bernoulli_indices(rng, n, prob)
                pauli = rng.integers(1, 4, size=hits.size)  # 1 X, 2 Y, 3 Z
                self.x[q, hits[pauli <= 2]] ^= True
                self.z[q, hits[pauli >= 2]] ^= True
            return
        if name == "DEPOLARIZE2":
            for a, b in zip(targets[::2], targets[1::2]):
                hits = _bernoulli_indices(rng, n, prob)
                code = rng.integers(1, 16, size=hits.size)
                pa, pb = code >> 2, code & 3
                self.x[a, hits[(pa == 1) | (pa == 2)]] ^= True
                self.z[a, hits[pa >= 2]] ^= True
                self.x[b, hits[(pb == 1) | (pb == 2)]] ^= True
                self.z[b, hits[pb >= 2]] ^= True
            return
        raise ValueError(f"Unsupported noise channel {name!r}.")

    def measure(self, qubit: int) -> np.ndarray:
        flip = self.x[qubit].copy()
        self.records.append(flip)
        self.z[qubit] = self.random_bits()
        return flip

    def reset(self, qubit: int) -> None:
        self.x[qubit] = False
        self.z[qubit] = self.random_bits()


def simulate_leakage(lrb_circuit, p: float, model: LeakageModel, shots: int,
                     seed: int = 0,
                     initial_leaked: Iterable[int] = ()) -> LeakageSample:
    """Sample ``shots`` runs of an LRB circuit with leakage.

    Args:
        lrb_circuit: ``circuits.LRBCircuit`` built at noise parameter ``p``.
        p: SI1000 parameter of the circuit (sets the leak rate).
        model: Leakage model.
        shots: Number of shots.
        seed: RNG seed.
        initial_leaked: Qubits leaked in every shot from the start (tests).
    """
    circuit = lrb_circuit.circuit
    rng = np.random.default_rng(seed)
    num_qubits = circuit.num_qubits
    frame = _FrameSimulator(num_qubits, shots, rng)
    reference = circuit.reference_sample()
    leaked = np.zeros((num_qubits, shots), dtype=bool)
    for q in initial_leaked:
        leaked[q] = True
    leak_prob = model.rate * p
    noisy_gates = _noisy_two_qubit_instructions(circuit)
    num_rounds = lrb_circuit.num_rounds
    num_resets = max(num_rounds - 1, 0)
    ancilla_flags = np.zeros((shots, num_rounds, 4), dtype=bool)
    lru_flags = np.zeros((shots, num_resets if (model.lru and model.lru_herald)
                          else 0, C.NUM_DATA), dtype=bool)
    terminal_flags = np.zeros((shots, C.NUM_DATA), dtype=bool)
    leaked_data: list[float] = []
    leak_events = 0
    ancilla_measurements = 0
    resets_seen = 0
    stabilizer_of = {a: s for s, a in enumerate(C.STABILIZER_ANCILLA)}
    detectors: list[np.ndarray] = []
    observable = np.zeros(shots, dtype=bool)

    def return_qubits() -> None:
        nonlocal resets_seen
        leaked_data.append(float(leaked[:C.NUM_DATA].any(axis=0).mean()))
        for q in range(num_qubits):
            if q >= C.NUM_DATA and model.reset_returns:
                continue
            if not leaked[q].any():
                continue
            if lru_flags.shape[1] and q < C.NUM_DATA:
                lru_flags[:, resets_seen, q] = leaked[q]
            back = np.zeros(shots, dtype=bool)
            if model.lru and model.lru_efficiency > 0:
                back |= leaked[q] & (rng.random(shots) < model.lru_efficiency)
            if model.seepage > 0:
                back |= leaked[q] & (rng.random(shots) < model.seepage)
            leaked[q] &= ~back
            frame.randomize(q, back)
        resets_seen += 1

    for index, inst in enumerate(circuit):
        name = inst.name
        targets = inst.targets_copy()
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            value = np.zeros(shots, dtype=bool)
            for t in targets:
                value ^= frame.records[len(frame.records) + t.value]
            if name == "DETECTOR":
                detectors.append(value)
            else:
                observable ^= value
            continue
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        qubits = [t.value for t in targets]
        if name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"):
            frame.noise(name, qubits, inst.gate_args_copy()[0])
            continue
        if name in _TWO_QUBIT:
            frame.gate(name, qubits)
            if index not in noisy_gates:
                continue
            for a, b in zip(qubits[::2], qubits[1::2]):
                old_a, old_b = leaked[a], leaked[b]
                if leak_prob > 0:
                    new_a = _bernoulli_indices(rng, shots, leak_prob)
                    new_b = _bernoulli_indices(rng, shots, leak_prob)
                    now_a, now_b = old_a.copy(), old_b.copy()
                    now_a[new_a] = True
                    now_b[new_b] = True
                    leak_events += int((now_a & ~old_a).sum()
                                       + (now_b & ~old_b).sum())
                else:
                    now_a, now_b = old_a, old_b
                if now_a.any() or now_b.any():
                    frame.randomize(b, now_a & ~now_b)
                    frame.randomize(a, now_b & ~now_a)
                leaked[a], leaked[b] = now_a, now_b
            continue
        if name == "M":
            for q in qubits:
                record = len(frame.records)
                which = leaked[q]
                if which.any():
                    desired = bool(model.readout_value) ^ bool(reference[record])
                    frame.x[q, which] = desired
                if q in stabilizer_of:
                    k = ancilla_measurements // 4
                    ancilla_flags[:, k, stabilizer_of[q]] = which
                    ancilla_measurements += 1
                elif q < C.NUM_DATA:
                    terminal_flags[:, q] = which
                frame.measure(q)
            continue
        if name == "R":
            if qubits and qubits[0] == C.ANCILLA_QUBITS[0]:
                return_qubits()
            for q in qubits:
                if model.reset_returns:
                    leaked[q] = False
                frame.reset(q)
            continue
        frame.gate(name, qubits)

    leaked_data.append(float(leaked[:C.NUM_DATA].any(axis=0).mean()))
    dets = (np.stack(detectors, axis=1) if detectors
            else np.zeros((shots, 0), dtype=bool))
    return LeakageSample(dets=dets, obs=observable,
                         ancilla_flags=ancilla_flags, lru_flags=lru_flags,
                         terminal_flags=terminal_flags,
                         leaked_data=np.asarray(leaked_data),
                         leak_events=leak_events)


def detection_autocorrelation(dets: np.ndarray, num_rounds: int,
                              max_lag: int = 6) -> np.ndarray:
    """Connected two-point correlation of detection events across rounds.

    Returns ``corr[lag, s]`` = mean over rounds ``k`` of
    ``<D_k,s D_k+lag,s> - <D_k,s><D_k+lag,s>`` for the round detectors
    (``corr[0]`` is NaN). Long-range (lag >= 2) correlations are the
    signature of persistent error sources such as leakage.
    """
    d = dets[:, :4 * num_rounds].reshape(dets.shape[0], num_rounds, 4)
    d = d.astype(np.float64)
    mean = d.mean(axis=0)
    out = np.full((max_lag + 1, 4), np.nan)
    for lag in range(1, min(max_lag, num_rounds - 1) + 1):
        prod = (d[:, :-lag, :] * d[:, lag:, :]).mean(axis=0)
        cov = prod - mean[:-lag] * mean[lag:]
        out[lag] = cov.mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Leakage experiment (identical shots for every method)
# ---------------------------------------------------------------------------

LEAKAGE_METHODS = (
    "lrbd_unif_1", "lrbd_unif_1_final", "lrbd_const_1",
    "lrbd_unif_1_final_flag", "lrbd_const_1_flag", "mld", "mld_flag",
)


def run_leakage_point(p: float, model: LeakageModel, sequences, depths,
                      shots: int, *, seed: int = 1, batch_size: int = 50_000,
                      max_lag: int = 6) -> dict:
    """Score LRB-D and exact-MLD correction under leakage on the same shots.

    Methods: ``lrbd_unif_1``, ``lrbd_unif_1_final``, ``lrbd_const_1`` (repo
    LRB-D, leakage flags ignored); ``*_flag`` variants also reject every shot
    with a leakage flag (three-state readout of ancillas and of the final
    data, plus LRU heralds when the model has a heralded LRU); ``mld`` is the
    lead's exact MLD for the leakage-free SI1000 model (leakage-unaware);
    ``mld_flag`` decodes only the shots without a leakage flag.

    Also returns the mean leaked-data trajectory and the detection-event
    autocorrelation (lags 1..max_lag) at the largest depth.
    """
    from . import baseline as B
    from . import circuits as Q
    from . import mld as M
    depths = list(depths)
    n_seq = len(sequences)
    tables = {name: B.CountTable.zeros(n_seq, len(depths))
              for name in LEAKAGE_METHODS}
    max_depth = max(depths)
    trajectory = np.zeros(max_depth + 2)
    autocorr = np.zeros((max_lag + 1, 4))
    autocorr_count = 0
    flagged = np.zeros((n_seq, len(depths)), dtype=np.int64)
    events = np.zeros((n_seq, len(depths)), dtype=np.int64)
    for si, sequence in enumerate(sequences):
        for di, depth in enumerate(depths):
            lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
            decoder = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                          M.lrb_detector_slots(lc))
            n_round = 4 * lc.num_rounds
            done = 0
            while done < shots:
                n = min(batch_size, shots - done)
                s = simulate_leakage(lc, p, model, n,
                                     seed=seed + 1_000_003 * si + 1009 * di
                                     + done)
                flip = s.obs
                raw = lc.raw_syndromes(s.dets[:, :n_round])
                final_clean = ~s.dets[:, n_round:].any(axis=1)
                flags = s.any_flag()
                flagged[si, di] += int(flags.sum())
                events[si, di] += s.leak_events
                acc_u1 = B.accept_mask(raw, lc.rounds,
                                       B.used_rounds("unif", 1, depth))
                acc_c1 = B.accept_mask(raw, lc.rounds,
                                       B.used_rounds("const", 1, depth))
                for name, acc in (
                        ("lrbd_unif_1", acc_u1),
                        ("lrbd_unif_1_final", acc_u1 & final_clean),
                        ("lrbd_const_1", acc_c1),
                        ("lrbd_unif_1_final_flag",
                         acc_u1 & final_clean & ~flags),
                        ("lrbd_const_1_flag", acc_c1 & ~flags)):
                    tables[name].add(si, di, n, int(acc.sum()),
                                     int((acc & flip).sum()))
                llr = decoder.log_likelihood_ratio(s.dets)
                wrong = (llr < 0) != flip
                tables["mld"].add(si, di, n, n, int(wrong.sum()))
                tables["mld_flag"].add(si, di, n, int((~flags).sum()),
                                       int((wrong & ~flags).sum()))
                if depth == max_depth:
                    trajectory[:len(s.leaked_data)] += s.leaked_data * n
                    if lc.num_rounds > 2:
                        autocorr += np.nan_to_num(detection_autocorrelation(
                            s.dets, lc.num_rounds, max_lag)) * n
                    autocorr_count += n
                done += n
    if autocorr_count:
        trajectory /= autocorr_count
        autocorr /= autocorr_count
    return {"p": p, "model": model, "depths": depths, "tables": tables,
            "leaked_trajectory": trajectory, "autocorr": autocorr,
            "flagged": flagged, "leak_events": events}
