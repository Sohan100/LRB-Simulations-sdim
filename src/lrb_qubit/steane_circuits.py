"""Steane-code LRB circuits, exact decoding and counting (work package K).

Circuits
--------
``build_lrb_circuit`` follows ``lrb_qubit.circuits.build_lrb_circuit``:

1. ideal encoding of ``|+>_L``;
2. for each forward Clifford ``C_j``: the transversal logical gates, one full
   noisy check round (Z checks, then X checks) and a reset of every ancilla
   and flag;
3. the inverses ``C_m^-1 ... C_1^-1``, each inverted token by token (mirror
   style), with no checks;
4. one final noisy check round (no reset);
5. an ideal X-basis readout of all seven data qubits. Observable 0 is the
   ``X_L = X0 X1 X2`` parity; the three X stabilizers of the readout give
   three final data detectors.

Noise is the repo's SI1000 model through ``lrb_qubit.circuits.emit_operations``
(consecutive measurements form one layer, consecutive resets one layer).
``schedule="sequential"`` measures each check (its ancilla, and its flag if
any) in its own layer right after its CNOTs, as the repo does;
``schedule="parallel"`` applies the same gates in the same order and measures
every ancilla and flag in one layer at the end of the round.

Detectors are the lead's difference detectors: detector ``(r, s)`` compares
round ``r``'s measurement of stabilizer ``s`` with the product of round
``r-1``'s generators that equals ``s`` after propagation back through the
transversal gates in between. Flag detectors are single measurements (a flag
reads 0 without faults).

Decoding
--------
``SlotMLD`` is an exact maximum-likelihood decoder. It is the transfer-matrix
decoder of ``lrb_qubit.mld`` (slots = check rounds, plus the final data
readout) with one addition: detectors can be declared *conditioning*
detectors (here the flags). Every mechanism that flips a flag of round ``r``
touches only round ``r`` and ``r+1`` syndromes, so it is folded into the
transfer kernel of the step ``r -> r+1`` with an extra parity bit per flag.
For each shot the kernel is restricted to the observed flag values, which
gives ``P(syndromes, flags, L)`` exactly while keeping 6-detector slots
(``128 x 128`` transfer matrices). Kernels are built lazily per observed flag
pattern and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import stim

from . import baseline as B
from . import circuits as Q
from . import code as C
from . import experiment as E
from . import mld as M
from .steane_code import SteaneCode

Operation = tuple[str, tuple[int, ...]]
SCHEDULES = ("sequential", "parallel")
FLAG_COORD_OFFSET = 100


# ----------------------------------------------------------------------
# Circuit construction
# ----------------------------------------------------------------------
def check_round_operations(code: SteaneCode, schedule: str = "sequential",
                           reset: bool = True
                           ) -> tuple[list[Operation], list[tuple[int, str]]]:
    """One check round and its measurement plan.

    Returns:
        ``(operations, plan)`` where ``plan[i] = (s, kind)`` names the
        ``i``-th measurement: ``kind`` is ``"a"`` (syndrome ancilla of
        stabilizer ``s``) or ``"f"`` (its flag).
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}.")
    order = list(code.z_stabilizers) + list(code.x_stabilizers)
    ops: list[Operation] = []
    plan: list[tuple[int, str]] = []
    measurements: list[Operation] = []
    for s in order:
        ops += code.check_operations(s)
        block = []
        for q, kind in zip(code.measured_qubits(s), ("a", "f")):
            block.append(("M", (q,)))
            plan.append((s, kind))
        if schedule == "sequential":
            ops += block
        else:
            measurements += block
    ops += measurements
    if reset:
        ops += [("R", (q,)) for q in code.ancilla_qubits]
    return ops, plan


@dataclass
class SteaneLRBCircuit:
    """A Stim Steane-code LRB circuit plus decoding and policy bookkeeping.

    Attributes:
        circuit: Noisy Stim circuit with detectors and observable 0 (X_L).
        depth: Benchmark depth ``m``.
        rounds: Check-round labels in time order (``0..m-1`` forward, ``m``
            terminal).
        syndrome_columns: ``(num_rounds, 6)`` detector index of each
            difference detector.
        flag_columns: ``(num_rounds, 6)`` detector index of each flag
            detector (``(num_rounds, 0)`` without flags).
        final_columns: Detector indices of the three final X-stabilizer
            detectors of the data readout.
        back: For each round label, the previous round's generator subset
            equal to each stabilizer after back-propagation.
        segments: Logical-gate operations before each round and before the
            readout.
    """

    circuit: stim.Circuit
    depth: int
    rounds: list[int]
    schedule: str
    extraction: str
    num_stabilizers: int
    syndrome_columns: np.ndarray
    flag_columns: np.ndarray
    final_columns: np.ndarray
    back: dict[int, tuple[tuple[int, ...], ...]]
    segments: list[list[Operation]] = field(default_factory=list)
    syndrome_records: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 6), dtype=np.int64))
    flag_records: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int64))

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    def raw_syndromes(self, detection_events: np.ndarray) -> np.ndarray:
        """``(shots, rounds, 6)`` raw stabilizer flips (as ``LRBCircuit``)."""
        S = self.num_stabilizers
        shots = detection_events.shape[0]
        raw = np.zeros((shots, self.num_rounds, S), dtype=bool)
        for k, label in enumerate(self.rounds):
            raw[:, k, :] = detection_events[:, self.syndrome_columns[k]]
            if k > 0:
                for s in range(S):
                    for prev_s in self.back[label][s]:
                        raw[:, k, s] ^= raw[:, k - 1, prev_s]
        return raw

    def raw_flags(self, detection_events: np.ndarray) -> np.ndarray:
        """``(shots, rounds, F)`` flag outcomes (``F = 0`` without flags)."""
        if self.flag_columns.size == 0:
            return np.zeros((detection_events.shape[0], self.num_rounds, 0),
                            dtype=bool)
        return detection_events[:, self.flag_columns]

    def round_clean(self, detection_events: np.ndarray,
                    use_flags: bool = True) -> np.ndarray:
        """``(shots, rounds)``: True where a round's checks (and flags) are
        trivial."""
        clean = ~self.raw_syndromes(detection_events).any(axis=2)
        if use_flags and self.flag_columns.size:
            clean &= ~self.raw_flags(detection_events).any(axis=2)
        return clean

    def final_clean(self, detection_events: np.ndarray) -> np.ndarray:
        return ~detection_events[:, self.final_columns].any(axis=1)

    def detector_slots(self) -> tuple[np.ndarray, np.ndarray]:
        """Slots and flag transitions for ``SlotMLD``.

        Syndrome detectors of round position ``k`` are in slot ``k``, final
        data detectors in slot ``num_rounds`` (if any), and flag detectors of
        round position ``k`` are conditioning detectors of transition
        ``k + 1``.
        """
        n = self.num_detectors
        slots = np.full(n, -2, dtype=np.int64)
        transitions = np.full(n, -1, dtype=np.int64)
        for k in range(self.num_rounds):
            slots[self.syndrome_columns[k]] = k
            if self.flag_columns.size:
                slots[self.flag_columns[k]] = -1
                transitions[self.flag_columns[k]] = k + 1
        slots[self.final_columns] = self.num_rounds
        if (slots == -2).any():
            raise AssertionError("Unassigned detector.")
        return slots, transitions


def _back_subsets(code: SteaneCode, segment: Sequence[Operation]
                  ) -> tuple[tuple[int, ...], ...]:
    inverse = code.data_tableau(segment).inverse()
    return tuple(code.decompose_in_stabilizers(
        inverse(code.stabilizer_pauli(s))) for s in range(code.num_stabilizers))


def build_lrb_circuit(code: SteaneCode,
                      cliffords: Sequence[str],
                      depth: int,
                      p: float,
                      *,
                      schedule: str = "sequential",
                      noise: Optional[Q.SI1000] = Q.SI1000(),
                      terminal_noise: bool = False,
                      forward_check_rounds: Optional[Iterable[int]] = None,
                      terminal_check: bool = True) -> SteaneLRBCircuit:
    """Encoded Steane LRB circuit of depth ``depth``.

    Arguments mirror ``lrb_qubit.circuits.build_lrb_circuit`` with the
    ``"data5"`` terminal semantics (every data qubit read out in the X
    basis), plus ``schedule``.
    """
    if len(cliffords) < depth:
        raise ValueError("Clifford sequence is shorter than the depth.")
    forward = list(cliffords[:depth])
    forward_set = (set(range(depth)) if forward_check_rounds is None
                   else {int(r) for r in forward_check_rounds})
    if any(r < 0 or r >= depth for r in forward_set):
        raise ValueError("forward_check_rounds must lie in [0, depth).")
    n_q = code.num_qubits
    S = code.num_stabilizers
    has_flags = code.extraction == "flag"

    circuit = stim.Circuit()
    Q.emit_operations(circuit, code.encoding_operations(), p, None, n_q)

    rounds: list[int] = []
    syn_records: dict[int, list[int]] = {}
    flag_records: dict[int, list[int]] = {}
    segments: list[list[Operation]] = []
    pending: list[Operation] = []

    def run_check_round(label: int, reset: bool = True) -> None:
        ops, plan = check_round_operations(code, schedule, reset=reset)
        records = Q.emit_operations(circuit, ops, p, noise, n_q,
                                    noisy_measurements=True)
        syn = [-1] * S
        flg = [-1] * S
        for (s, kind), record in zip(plan, records):
            if kind == "a":
                syn[s] = record
            else:
                flg[s] = record
        rounds.append(label)
        syn_records[label] = syn
        flag_records[label] = flg
        segments.append(list(pending))
        pending.clear()

    for j, descriptor in enumerate(forward):
        ops = code.logical_clifford_operations(descriptor)
        Q.emit_operations(circuit, ops, p, noise, n_q,
                          noisy_measurements=False)
        pending.extend(ops)
        if j in forward_set:
            run_check_round(j)

    for descriptor in reversed(forward):
        ops = code.logical_clifford_operations(descriptor, inverse=True)
        Q.emit_operations(circuit, ops, p, noise, n_q,
                          noisy_measurements=False)
        pending.extend(ops)

    if terminal_check:
        run_check_round(depth, reset=False)
    segments.append(list(pending))
    pending.clear()

    readout_ops = ([("H", (q,)) for q in code.data_qubits]
                   + [("M", (q,)) for q in code.data_qubits])
    readout_records = Q.emit_operations(
        circuit, readout_ops, p, noise if terminal_noise else None, n_q,
        noisy_measurements=True)
    data_record = dict(zip(code.data_qubits, readout_records))

    def rec(records: Sequence[int]) -> list[stim.GateTarget]:
        return [stim.target_rec(r - circuit.num_measurements) for r in records]

    det_index = 0
    syndrome_columns = np.zeros((len(rounds), S), dtype=np.int64)
    flag_columns = np.zeros((len(rounds), S if has_flags else 0),
                            dtype=np.int64)
    back: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous: Optional[list[int]] = None
    for position, label in enumerate(rounds):
        records = syn_records[label]
        if previous is None:
            subsets: tuple[tuple[int, ...], ...] = tuple(() for _ in range(S))
        else:
            subsets = _back_subsets(code, segments[position])
        back[label] = subsets
        for s in range(S):
            targets = [records[s]]
            if previous is not None:
                targets += [previous[t] for t in subsets[s]]
            circuit.append("DETECTOR", rec(targets), [s, label])
            syndrome_columns[position, s] = det_index
            det_index += 1
        if has_flags:
            for s in range(S):
                circuit.append("DETECTOR", rec([flag_records[label][s]]),
                               [FLAG_COORD_OFFSET + s, label])
                flag_columns[position, s] = det_index
                det_index += 1
        previous = records

    final_subsets = _back_subsets(code, segments[-1])
    final_columns = []
    for s in code.x_stabilizers:
        support = [q for q, ch in enumerate(code.stabilizer_paulis[s])
                   if ch != "_"]
        targets = [data_record[q] for q in support]
        if previous is not None:
            targets += [previous[t] for t in final_subsets[s]]
        circuit.append("DETECTOR", rec(targets), [s, depth + 1])
        final_columns.append(det_index)
        det_index += 1

    circuit.append("OBSERVABLE_INCLUDE",
                   rec([data_record[q] for q in code.logical_x_support]), 0)

    return SteaneLRBCircuit(
        circuit=circuit, depth=depth, rounds=rounds, schedule=schedule,
        extraction=code.extraction, num_stabilizers=S,
        syndrome_columns=syndrome_columns, flag_columns=flag_columns,
        final_columns=np.asarray(final_columns, dtype=np.int64), back=back,
        segments=segments,
        syndrome_records=np.asarray([syn_records[r] for r in rounds],
                                    dtype=np.int64).reshape(len(rounds), S),
        flag_records=np.asarray(
            [flag_records[r] for r in rounds] if has_flags else [],
            dtype=np.int64).reshape(len(rounds), S if has_flags else 0))


def build_lrb_const0_circuit(code: SteaneCode, cliffords: Sequence[str],
                             depth: int, p: float, *,
                             noise: Optional[Q.SI1000] = Q.SI1000()
                             ) -> SteaneLRBCircuit:
    """``const=0``: no check rounds, ideal X readout of all data."""
    return build_lrb_circuit(code, cliffords, depth, p, noise=noise,
                             forward_check_rounds=(), terminal_check=False)


# ----------------------------------------------------------------------
# Exact maximum-likelihood decoding with flag conditioning
# ----------------------------------------------------------------------
class _FlagGroup:
    """Flags of one transition that share mechanisms, and those mechanisms."""

    def __init__(self, flags: list[int]):
        self.flags = flags              # local bit positions in the pattern
        self.mechanisms: list[tuple[int, int, float]] = []


class SlotMLD:
    """Exact MLD over time slots, conditioning on flag detectors per shot.

    Args:
        dem: Detector error model with observable 0 as the logical.
        detector_slots: Slot of every detector (non-decreasing in time); ``-1``
            marks a conditioning (flag) detector.
        flag_transitions: For conditioning detectors, the transition ``u >= 1``
            (from slot ``u - 1`` to slot ``u``) whose kernel holds every
            mechanism that flips them. Ignored for other detectors.
        max_slot_size: Largest allowed slot (the transfer matrices are
            ``2^(n+1)`` square).

    With no conditioning detectors this is the lead's ``TransferMatrixMLD``
    (same kernels and recursion; equality is tested).
    """

    def __init__(self, dem: stim.DetectorErrorModel,
                 detector_slots: Sequence[int],
                 flag_transitions: Optional[Sequence[int]] = None,
                 max_slot_size: int = 8):
        slots = np.asarray(detector_slots, dtype=np.int64)
        n_det = dem.num_detectors
        if len(slots) != n_det:
            raise ValueError("Need one slot per detector.")
        if flag_transitions is None:
            flag_transitions = np.full(n_det, -1, dtype=np.int64)
        transitions = np.asarray(flag_transitions, dtype=np.int64)
        is_flag = slots == -1
        if (slots < -1).any():
            raise ValueError("Invalid slot index.")
        regular = ~is_flag
        self.num_slots = int(slots[regular].max()) + 1 if regular.any() else 1
        self.slot_detectors = [np.flatnonzero(slots == t)
                               for t in range(self.num_slots)]
        self.slot_sizes = [len(d) for d in self.slot_detectors]
        if max(self.slot_sizes) > max_slot_size:
            raise ValueError(f"Slots larger than {max_slot_size} detectors "
                             "are not supported.")
        position = np.zeros(n_det, dtype=np.int64)
        for dets in self.slot_detectors:
            position[dets] = np.arange(len(dets))

        # Conditioning detectors per transition.
        self.transition_flags: dict[int, np.ndarray] = {}
        flag_bit = np.full(n_det, -1, dtype=np.int64)
        for u in sorted(set(transitions[is_flag].tolist())):
            if u < 1 or u >= self.num_slots:
                raise ValueError(f"Flag transition {u} out of range.")
            dets = np.flatnonzero(is_flag & (transitions == u))
            self.transition_flags[u] = dets
            flag_bit[dets] = np.arange(len(dets))
            if len(dets) > 20:
                raise ValueError("Too many flags in one transition.")

        kernels = []
        for t in range(self.num_slots):
            n_prev = self.slot_sizes[t - 1] if t > 0 else 0
            kernel = np.zeros(2 ** (n_prev + self.slot_sizes[t] + 1))
            kernel[0] = 1.0
            kernels.append(kernel)

        flag_mechs: dict[int, list[tuple[int, int, float]]] = {
            u: [] for u in self.transition_flags}
        self.num_mechanisms = 0
        self.undetectable_logical_probability = 0.0
        for mech in M.dem_mechanisms(dem):
            if mech.probability <= 0:
                continue
            dets = list(mech.detectors)
            if not dets:
                if mech.flips_observable:
                    q = self.undetectable_logical_probability
                    self.undetectable_logical_probability = (
                        q + mech.probability - 2 * q * mech.probability)
                    M._convolve(kernels[0], 1 << self.slot_sizes[0],
                                mech.probability)
                    self.num_mechanisms += 1
                continue
            flags = [d for d in dets if is_flag[d]]
            syn = [d for d in dets if not is_flag[d]]
            if flags:
                us = set(transitions[flags].tolist())
                if len(us) != 1:
                    raise ValueError("Mechanism flips flags of several "
                                     "transitions.")
                t = us.pop()
                if syn and (min(slots[syn]) < t - 1 or max(slots[syn]) > t):
                    raise ValueError(
                        "Flag mechanism outside its transition: slots "
                        f"{sorted(set(slots[syn].tolist()))}, transition {t}.")
            else:
                syn_slots = slots[syn]
                t = int(syn_slots.max())
                if int(syn_slots.min()) < t - 1:
                    raise ValueError(
                        "Mechanism spans non-adjacent slots: "
                        f"{sorted(set(syn_slots.tolist()))}")
            n_prev = self.slot_sizes[t - 1] if t > 0 else 0
            n_cur = self.slot_sizes[t]
            vector = 0
            for det in syn:
                bit = int(position[det])
                if slots[det] == t:
                    vector ^= 1 << (n_prev + bit)
                else:
                    vector ^= 1 << bit
            if mech.flips_observable:
                vector ^= 1 << (n_prev + n_cur)
            if flags:
                mask = 0
                for det in flags:
                    mask ^= 1 << int(flag_bit[det])
                flag_mechs[t].append((vector, mask, mech.probability))
            else:
                M._convolve(kernels[t], vector, mech.probability)
            self.num_mechanisms += 1

        # Group the flags of each transition into connected components.
        self.groups: dict[int, list[_FlagGroup]] = {}
        for u, mechs in flag_mechs.items():
            n_flags = len(self.transition_flags[u])
            parent = list(range(n_flags))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for _, mask, _ in mechs:
                bits = [b for b in range(n_flags) if (mask >> b) & 1]
                for b in bits[1:]:
                    parent[find(b)] = find(bits[0])
            components: dict[int, list[int]] = {}
            for b in range(n_flags):
                components.setdefault(find(b), []).append(b)
            groups = {root: _FlagGroup(sorted(members))
                      for root, members in components.items()}
            for vector, mask, prob in mechs:
                first = next(b for b in range(n_flags) if (mask >> b) & 1)
                group = groups[find(first)]
                local = 0
                for i, b in enumerate(group.flags):
                    if (mask >> b) & 1:
                        local |= 1 << i
                group.mechanisms.append((vector, local, prob))
            self.groups[u] = list(groups.values())
            if max(len(g.flags) for g in self.groups[u]) > 8:
                raise ValueError("Flag groups larger than 8 flags.")

        self.base_kernels = kernels
        self.initial = kernels[0].copy()
        self.transfer: list[Optional[np.ndarray]] = [None]
        for t in range(1, self.num_slots):
            self.transfer.append(self._matrix_from_kernel(t, kernels[t]))
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._prefix_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    # ------------------------------------------------------------------
    def _matrix_from_kernel(self, t: int, kernel: np.ndarray) -> np.ndarray:
        n_prev = self.slot_sizes[t - 1]
        n_cur = self.slot_sizes[t]
        view = kernel.reshape(2, 2 ** n_cur, 2 ** n_prev)
        matrix = np.zeros((2 * 2 ** n_prev, 2 * 2 ** n_cur))
        for l_in in range(2):
            for l_out in range(2):
                matrix[l_in * 2 ** n_prev:(l_in + 1) * 2 ** n_prev,
                       l_out * 2 ** n_cur:(l_out + 1) * 2 ** n_cur] = (
                           view[l_in ^ l_out].T)
        return matrix

    def kernel_for_pattern(self, t: int, pattern: int) -> np.ndarray:
        """Kernel of transition ``t`` restricted to observed flags ``pattern``.

        Entry ``(c, b, l)`` is ``P(prev flips c, cur flips b, L flip l,
        flags = pattern)`` over the mechanisms of this transition.
        """
        kernel = self.base_kernels[t]
        size = kernel.size
        shift = size.bit_length() - 1
        prefix: tuple[int, ...] = ()
        for group in self.groups.get(t, []):
            local = 0
            for i, b in enumerate(group.flags):
                if (pattern >> b) & 1:
                    local |= 1 << i
            prefix = prefix + (local,)
            cached = self._prefix_cache.get((t, prefix))
            if cached is None:
                # Patterns sharing the first groups' flag values share the
                # partial kernel, so each group is folded in once per prefix.
                lifted = np.zeros(size << len(group.flags))
                lifted[:size] = kernel
                for vector, mask, prob in group.mechanisms:
                    M._convolve(lifted, vector | (mask << shift), prob)
                cached = lifted[local * size:(local + 1) * size].copy()
                self._prefix_cache[(t, prefix)] = cached
            kernel = cached
        return kernel.copy()

    def _transfer_for(self, t: int, pattern: int) -> np.ndarray:
        key = (t, pattern)
        matrix = self._cache.get(key)
        if matrix is None:
            matrix = self._matrix_from_kernel(
                t, self.kernel_for_pattern(t, pattern))
            self._cache[key] = matrix
        return matrix

    def _slot_values(self, events: np.ndarray, t: int) -> np.ndarray:
        dets = self.slot_detectors[t]
        weights = (1 << np.arange(len(dets))).astype(np.int64)
        return events[:, dets].astype(np.int64) @ weights

    def _flag_patterns(self, events: np.ndarray, t: int) -> np.ndarray:
        dets = self.transition_flags[t]
        weights = (1 << np.arange(len(dets))).astype(np.int64)
        return events[:, dets].astype(np.int64) @ weights

    def log_likelihood_ratio(self, detection_events: np.ndarray,
                             batch_size: int = 65536) -> np.ndarray:
        """``log P(L=0 | s) - log P(L=1 | s)`` for each shot."""
        detection_events = np.asarray(detection_events, dtype=bool)
        out = np.empty(detection_events.shape[0])
        for start in range(0, detection_events.shape[0], batch_size):
            chunk = detection_events[start:start + batch_size]
            out[start:start + len(chunk)] = self._llr_chunk(chunk)
        return out

    def _llr_chunk(self, events: np.ndarray) -> np.ndarray:
        shots = events.shape[0]
        psi = np.broadcast_to(self.initial, (shots, self.initial.size)).copy()
        for t in range(1, self.num_slots):
            n_prev = self.slot_sizes[t - 1]
            s = self._slot_values(events, t - 1)
            size = 2 ** n_prev
            base = np.arange(2 * size)
            cols = (base[None, :] & (size - 1)) ^ s[:, None]
            cols |= (base[None, :] >> n_prev) << n_prev
            phi = np.take_along_axis(psi, cols, axis=1)
            if t in self.groups:
                patterns = self._flag_patterns(events, t)
                order = np.argsort(patterns, kind="stable")
                sorted_patterns = patterns[order]
                bounds = np.flatnonzero(np.diff(sorted_patterns)) + 1
                starts = np.concatenate([[0], bounds])
                ends = np.concatenate([bounds, [shots]])
                new = np.empty((shots, 2 * 2 ** self.slot_sizes[t]))
                for a, b in zip(starts, ends):
                    idx = order[a:b]
                    new[idx] = phi[idx] @ self._transfer_for(
                        t, int(sorted_patterns[a]))
                psi = new
            else:
                psi = phi @ self.transfer[t]
            norm = psi.max(axis=1, keepdims=True)
            norm[norm <= 0] = 1.0
            psi /= norm
        n_last = self.slot_sizes[-1]
        s = self._slot_values(events, self.num_slots - 1)
        p0 = psi[np.arange(shots), s]
        p1 = psi[np.arange(shots), s + 2 ** n_last]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(p0) - np.log(p1)

    def decode(self, detection_events: np.ndarray) -> tuple[np.ndarray,
                                                            np.ndarray]:
        """Return ``(predicted_flip, gap)`` with ``gap = |LLR|``."""
        llr = self.log_likelihood_ratio(detection_events)
        return llr < 0, np.abs(llr)


def mld_for(lc: SteaneLRBCircuit,
            dem: Optional[stim.DetectorErrorModel] = None) -> SlotMLD:
    """Exact decoder for one Steane LRB circuit."""
    if dem is None:
        dem = lc.circuit.detector_error_model()
    slots, transitions = lc.detector_slots()
    return SlotMLD(dem, slots, transitions)


def brute_force_llr(dem: stim.DetectorErrorModel,
                    detection_events: np.ndarray) -> np.ndarray:
    """Exact LLR by enumerating the full joint (at most 22 detectors)."""
    joint = M.brute_force_joint(dem)
    n = dem.num_detectors
    weights = (1 << np.arange(n)).astype(np.int64)
    index = np.asarray(detection_events, dtype=np.int64) @ weights
    with np.errstate(divide="ignore"):
        return np.log(joint[index]) - np.log(joint[index + (1 << n)])


# ----------------------------------------------------------------------
# DEM structure: circuit distance and first-order coefficients
# ----------------------------------------------------------------------
def symptom_classes(dem: stim.DetectorErrorModel
                    ) -> dict[tuple[int, ...], list[float]]:
    """Single mechanisms grouped by symptom: ``{dets: [P0, P1]}``.

    ``P0`` (``P1``) is the total probability of the mechanisms with these
    detection events that leave (flip) the logical observable, combined as
    independent errors.
    """
    classes: dict[tuple[int, ...], list[float]] = {}
    for mech in M.dem_mechanisms(dem):
        if mech.probability <= 0:
            continue
        entry = classes.setdefault(mech.detectors, [0.0, 0.0])
        k = 1 if mech.flips_observable else 0
        old = entry[k]
        entry[k] = old + mech.probability - 2 * old * mech.probability
    return classes


def first_order_summary(dem: stim.DetectorErrorModel,
                        detectable: Optional[np.ndarray] = None) -> dict:
    """First-order quantities of the DEM (exact at small p).

    Returns ``U1`` (undetectable logical weight), ``R1`` (weight with a
    non-empty symptom: LRB-D rejection when every detector is used), ``E1``
    (``sum_s min(P0, P1)``: first-order error of exact MLD), ``A2``
    (``sum_s P0 P1``: second-order accepted error of that LRB-D) and the
    number of ambiguous symptom classes (``P0, P1 > 0``).
    """
    classes = symptom_classes(dem)
    u1 = r1 = e1 = a2 = 0.0
    ambiguous = 0
    for dets, (p0, p1) in classes.items():
        if not dets:
            u1 += p1
            continue
        r1 += p0 + p1
        e1 += min(p0, p1)
        a2 += p0 * p1
        if p0 > 0 and p1 > 0:
            ambiguous += 1
    return {"U1": u1, "R1": r1, "E1": e1, "A2": a2,
            "ambiguous_classes": ambiguous, "num_classes": len(classes)}


def distance_report(circuit: stim.Circuit, *, search: bool = True,
                    search_size: int = 6, search_degree: int = 6) -> dict:
    """Circuit-level distance evidence for one circuit.

    * ``exact_lower_bound``: 1 if a single mechanism flips the observable
      with no detection event; else 2 if two mechanisms share their
      detection events but differ in the observable flip (together they are
      an undetectable logical error); else 3. This is exact for ``d <= 3``.
    * ``stim_search``: size of the undetectable logical error found by
      ``search_for_undetectable_logical_errors`` (an upper bound).
    * ``shortest_graphlike``: size found by ``shortest_graphlike_error``
      with hyperedges ignored (an upper bound restricted to graphlike
      mechanisms), or ``None`` if it finds none.
    """
    dem = circuit.detector_error_model()
    summary = first_order_summary(dem)
    if summary["U1"] > 0:
        lower = 1
    elif summary["ambiguous_classes"] > 0:
        lower = 2
    else:
        lower = 3
    report = {"exact_lower_bound": lower,
              "ambiguous_classes": summary["ambiguous_classes"],
              "num_detectors": dem.num_detectors,
              "num_mechanisms": dem.num_errors}
    if search:
        found = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=search_size,
            dont_explore_edges_with_degree_above=search_degree,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True)
        report["stim_search"] = len(found) if found else None
        report["stim_search_example"] = [
            str(e.circuit_error_locations[0].flipped_pauli_product)
            + " @ " + str(e.circuit_error_locations[0].instruction_targets)
            if e.circuit_error_locations else str(e.dem_error_terms)
            for e in found]
        report["search_limits"] = {"size": search_size,
                                   "degree": search_degree}
    try:
        graphlike = circuit.shortest_graphlike_error(
            ignore_ungraphlike_errors=True, canonicalize_circuit_errors=True)
        report["shortest_graphlike"] = len(graphlike)
    except ValueError as exc:
        report["shortest_graphlike"] = None
        report["shortest_graphlike_error"] = str(exc).splitlines()[0][:200]
    return report


# ----------------------------------------------------------------------
# Physical RB (exact)
# ----------------------------------------------------------------------
def physical_rb_fidelity(sequence: Sequence[str], depth: int, p_rb: float,
                         noise: Q.SI1000 = Q.SI1000()) -> float:
    """Exact ``F = |P(0) - P(1)|`` of ``circuits.build_physical_rb_circuit``.

    Each gate is followed by ``DEPOLARIZE1(q)``, ``q = 0.1 p_rb``; the
    channel commutes with every unitary and shrinks the Bloch vector by
    ``1 - 4q/3``. The forward and inverse halves have the same gate count.
    """
    q = noise.one_qubit * p_rb
    gates = 2 * sum(len(C.clifford_gate_sequence(c)) for c in sequence[:depth])
    return (1.0 - 4.0 * q / 3.0) ** gates


def physical_rb_decay(p_rb: float, noise: Q.SI1000 = Q.SI1000()) -> float:
    """Ensemble decay per step: ``mean_C (1 - 4q/3)^(2 len C)``."""
    q = noise.one_qubit * p_rb
    lengths = np.array([len(C.clifford_gate_sequence(c))
                        for c in C.SINGLE_QUBIT_CLIFFORDS])
    return float(np.mean((1.0 - 4.0 * q / 3.0) ** (2 * lengths)))


# ----------------------------------------------------------------------
# Counting
# ----------------------------------------------------------------------
DEFAULT_POLICIES = ([(f"lrbd_const_{k}", "const", k) for k in range(1, 23)]
                    + [(f"lrbd_unif_{k}", "unif", k) for k in range(1, 23)])
NOFLAG_POLICIES = (("unif", 1), ("const", 1))


def _histogram_gap(hist: np.ndarray, gap: np.ndarray, wrong: np.ndarray
                   ) -> None:
    bins = np.clip(np.searchsorted(E.GAP_EDGES, gap, side="right") - 1, 0,
                   len(E.GAP_EDGES) - 2)
    np.add.at(hist, (bins, wrong.astype(np.int64)), 1)


def run_sequence(code: SteaneCode, schedule: str, p: float,
                 sequence: Sequence[str], depths: Sequence[int], shots: int,
                 *, decode_shots: Optional[int] = None,
                 const0: bool = True,
                 policies=DEFAULT_POLICIES,
                 batch_size: int = 50_000,
                 seed: int = 0,
                 noise: Q.SI1000 = Q.SI1000()) -> E.PointResult:
    """All methods for one Clifford sequence at one ``p`` (one-row tables).

    Methods (``experiment.PointResult`` tables, sequence index 0):

    ``lrbd_{const,unif}_k``
        Accept when every used round (``baseline.used_rounds``) has a
        trivial raw syndrome and, with flags, no raised flag.
    ``lrbd_{const,unif}_k_final``
        The same, also requiring the three final data detectors to be clean.
    ``lrbd_{unif,const}_1[_final]_noflag``
        Flagged circuits only: the same policies ignoring the flags.
    ``mld``
        Exact maximum-likelihood correction of every decoded shot (no
        rejection); ``|LLR|`` histograms (``gap_hist["mld"]``) split by
        correct / wrong allow any soft-output threshold later.
    ``const0`` / ``const0_mld``
        The separate no-check circuit, postselected on the final X
        stabilizers or decoded exactly.
    """
    depths = list(depths)
    result = E.PointResult(p, depths, 1)
    decode_shots = shots if decode_shots is None else min(decode_shots, shots)
    for d_index, depth in enumerate(depths):
        lc = build_lrb_circuit(code, sequence, depth, p, schedule=schedule,
                               noise=noise)
        decoder = mld_for(lc) if decode_shots > 0 else None
        sampler = lc.circuit.compile_detector_sampler(
            seed=seed + 1000 * d_index)
        done = 0
        while done < shots:
            n = min(batch_size, shots - done)
            dets, obs = sampler.sample(n, separate_observables=True)
            flip = obs[:, 0]
            final_ok = lc.final_clean(dets)
            clean = lc.round_clean(dets, use_flags=True)
            clean_nf = (lc.round_clean(dets, use_flags=False)
                        if code.extraction == "flag" else None)
            for name, policy, k in policies:
                columns = [lc.rounds.index(r)
                           for r in B.used_rounds(policy, k, depth)]
                acc = clean[:, columns].all(axis=1)
                result.table(name).add(0, d_index, n, int(acc.sum()),
                                       int((acc & flip).sum()))
                acc_f = acc & final_ok
                result.table(name + "_final").add(
                    0, d_index, n, int(acc_f.sum()), int((acc_f & flip).sum()))
                if clean_nf is not None and (policy, k) in NOFLAG_POLICIES:
                    acc_n = clean_nf[:, columns].all(axis=1)
                    result.table(name + "_noflag").add(
                        0, d_index, n, int(acc_n.sum()),
                        int((acc_n & flip).sum()))
                    acc_nf = acc_n & final_ok
                    result.table(name + "_final_noflag").add(
                        0, d_index, n, int(acc_nf.sum()),
                        int((acc_nf & flip).sum()))
            n_dec = min(n, max(decode_shots - done, 0))
            if decoder is not None and n_dec > 0:
                llr = decoder.log_likelihood_ratio(dets[:n_dec])
                wrong = (llr < 0) != flip[:n_dec]
                result.table("mld").add(0, d_index, n_dec, n_dec,
                                        int(wrong.sum()))
                _histogram_gap(result.hist("mld")[0, d_index], np.abs(llr),
                               wrong)
            done += n

        if const0:
            lc0 = build_lrb_const0_circuit(code, sequence, depth, p,
                                           noise=noise)
            dec0 = mld_for(lc0)
            sampler0 = lc0.circuit.compile_detector_sampler(
                seed=seed + 7 + 1000 * d_index)
            done = 0
            while done < shots:
                n = min(batch_size, shots - done)
                dets, obs = sampler0.sample(n, separate_observables=True)
                flip = obs[:, 0]
                acc = ~dets.any(axis=1)
                result.table("const0").add(0, d_index, n, int(acc.sum()),
                                           int((acc & flip).sum()))
                n_dec = min(n, max(decode_shots - done, 0))
                if n_dec > 0:
                    llr = dec0.log_likelihood_ratio(dets[:n_dec])
                    wrong = (llr < 0) != flip[:n_dec]
                    result.table("const0_mld").add(0, d_index, n_dec, n_dec,
                                                   int(wrong.sum()))
                    _histogram_gap(result.hist("const0_mld")[0, d_index],
                                   np.abs(llr), wrong)
                done += n
    return result
