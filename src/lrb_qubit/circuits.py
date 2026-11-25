"""Stim circuit builders for qubit ``[[5,1,2]]`` LRB and matched physical RB.

The encoded circuits follow ``LRBCircuitGenerator.generate_lrb_clifford_sequence``
as it was before the sdim 1.4.0 port (commit ``c21386f6``), which the qubit
correction-decoder study was built on. The current generator schedules every
segment into moments with idle noise on untouched wires and resets the ancillas
inside each check round, so its DEMs differ from these (same mechanisms,
different probabilities); ``studies/correction-decoders/qubit/scripts/
crosscheck_sdim.py dem`` shows the difference. The earlier schedule:

1. ideal encoding of |+>_L;
2. for each forward Clifford ``C_j``: the logical gates, one full noisy check
   round and an ancilla reset;
3. the inverses ``C_m^-1 ... C_1^-1``, each inverted individually, with no
   checks;
4. one final noisy check round;
5. an ideal terminal X-basis readout.

Noise follows the earlier ``LRBCircuitGenerator.apply_si1000_noise_to_circuit``
(since replaced by ``apply_circuit_level_noise``) with qubit channels: ``DEPOLARIZE1(0.1p)`` after one-qubit gates, ``DEPOLARIZE2(p)`` after
two-qubit gates, and for every measurement layer ``DEPOLARIZE1(2p)`` on all
other wires plus ``DEPOLARIZE1(p)`` and ``X_ERROR(5p)`` on each measured wire.
Reset layers put ``DEPOLARIZE1(2p)`` on all other wires and ``X_ERROR(2p)``
after each reset. The encoding and the terminal readout are ideal unless
``terminal_noise`` is set.

Check rounds are labelled ``0..m-1`` (after forward Clifford ``j = r + 1``) and
``m`` (after the inverse half). Detector ``(r, s)`` compares the measurement of
stabilizer ``s`` in round ``r`` with the product of the previous physical
round's measurements that equals ``s`` after propagating it back through the
logical gates in between ("difference" detectors). Their detection events are
local in time, which matching and transfer-matrix decoders need. The raw
per-round syndromes used by the LRB-D postselection policies are recovered with
``raw_syndromes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import stim

from . import code as C


@dataclass(frozen=True)
class SI1000:
    """Scale factors of the repo's generalized SI1000 model (multiples of p)."""

    one_qubit: float = 0.1
    two_qubit: float = 1.0
    meas_idle: float = 2.0
    meas_depol: float = 1.0
    meas_flip: float = 5.0
    reset_idle: float = 2.0
    reset_flip: float = 2.0

    def scaled(self, **factors: float) -> "SI1000":
        """Return a copy with some components multiplied (for ablations)."""
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for name, factor in factors.items():
            values[name] *= factor
        return SI1000(**values)


INVERSE_ROUND_OFFSET = 1000

_ONE_QUBIT_GATES = {"I", "X", "Z", "H", "S", "S_DAG"}
_TWO_QUBIT_GATES = {"CX", "CZ", "SWAP"}


def _append(circuit: stim.Circuit, name: str, targets: Sequence[int],
            arg: Optional[float] = None) -> None:
    if arg is None:
        circuit.append(name, list(targets))
    elif arg > 0.0:
        circuit.append(name, list(targets), arg)


def emit_operations(circuit: stim.Circuit,
                    operations: Sequence[C.Operation],
                    p: float,
                    noise: Optional[SI1000],
                    num_qubits: int,
                    noisy_measurements: bool = True) -> list[int]:
    """Append ``operations`` to ``circuit`` with optional SI1000 noise.

    Mirrors ``apply_si1000_noise_to_circuit``: consecutive measurements form
    one measurement layer and consecutive resets form one reset layer.

    Returns:
        Measurement-record indices of the ``M`` operations, in order.
    """
    records: list[int] = []
    noisy = noise is not None and p > 0.0
    index = 0
    while index < len(operations):
        gate, targets = operations[index]
        if gate in ("M", "R"):
            end = index
            while end < len(operations) and operations[end][0] == gate:
                end += 1
            layer = [operations[i][1][0] for i in range(index, end)]
            if gate == "M":
                if noisy and noisy_measurements:
                    idle = [q for q in range(num_qubits) if q not in layer]
                    _append(circuit, "DEPOLARIZE1", idle, noise.meas_idle * p)
                for q in layer:
                    if noisy and noisy_measurements:
                        _append(circuit, "DEPOLARIZE1", [q], noise.meas_depol * p)
                        _append(circuit, "X_ERROR", [q], noise.meas_flip * p)
                    records.append(circuit.num_measurements)
                    circuit.append("M", [q])
            else:
                if noisy:
                    idle = [q for q in range(num_qubits) if q not in layer]
                    _append(circuit, "DEPOLARIZE1", idle, noise.reset_idle * p)
                for q in layer:
                    circuit.append("R", [q])
                    if noisy:
                        _append(circuit, "X_ERROR", [q], noise.reset_flip * p)
            index = end
            continue
        circuit.append(gate, list(targets))
        if noisy:
            if gate in _ONE_QUBIT_GATES:
                _append(circuit, "DEPOLARIZE1", targets, noise.one_qubit * p)
            elif gate in _TWO_QUBIT_GATES:
                _append(circuit, "DEPOLARIZE2", targets, noise.two_qubit * p)
            else:
                raise ValueError(f"No SI1000 rule for gate {gate!r}.")
        index += 1
    return records


@dataclass
class LRBCircuit:
    """A Stim LRB circuit plus the bookkeeping decoders and policies need.

    Attributes:
        circuit: Noisy Stim circuit with detectors and observable 0 (= X_L).
        depth: Benchmark depth ``m``.
        rounds: Physical check-round labels in time order (subset of 0..m).
        detector_keys: ``(round, stabilizer)`` for every detector, in order.
            Final data-readout detectors use round label ``m + 1``.
        back: For each physical round, a tuple of four generator subsets: the
            previous physical round's stabilizers whose product equals this
            round's stabilizer ``s`` after back-propagation (empty for the
            first round, whose reference is the ideal code state).
        terminal: ``"data2"`` (data 0 and 3, as in the qutrit pipeline) or
            ``"data5"`` (all five data qubits, enables final X detectors).
        segments: Logical-gate operations between consecutive physical rounds
            (``segments[i]`` precedes ``rounds[i]``; the last entry precedes
            the terminal readout).
    """

    circuit: stim.Circuit
    depth: int
    rounds: list[int]
    detector_keys: list[tuple[int, int]]
    back: dict[int, tuple[tuple[int, ...], ...]]
    terminal: str
    segments: list[list[C.Operation]] = field(default_factory=list)

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    def round_detector_slice(self, round_label: int) -> slice:
        """Detector-column slice holding the four detectors of one round."""
        position = self.rounds.index(round_label)
        return slice(4 * position, 4 * position + 4)

    def raw_syndromes(self, detection_events: np.ndarray) -> np.ndarray:
        """Per-round raw stabilizer flips from difference detection events.

        Args:
            detection_events: ``(shots, num_detectors)`` boolean array.

        Returns:
            ``(shots, num_rounds, 4)`` boolean array; entry ``[i, k, s]`` is
            True when the round-``rounds[k]`` measurement of stabilizer ``s``
            differs from its noiseless value. This is the quantity the qutrit
            pipeline postselects on.
        """
        shots = detection_events.shape[0]
        raw = np.zeros((shots, self.num_rounds, 4), dtype=bool)
        for k, label in enumerate(self.rounds):
            det = detection_events[:, 4 * k:4 * k + 4]
            for s in range(4):
                value = det[:, s].copy()
                if k > 0:
                    for prev_s in self.back[label][s]:
                        value ^= raw[:, k - 1, prev_s]
                raw[:, k, s] = value
        return raw


def _detector_from_back(prev_records: Optional[Sequence[int]],
                        subset: Sequence[int]) -> list[int]:
    if prev_records is None:
        return []
    return [prev_records[s] for s in subset]


def _back_subsets(segment: Sequence[C.Operation],
                  stabilizers: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Generator subsets of each stabilizer back-propagated through ``segment``."""
    inverse = C.data_tableau(segment).inverse()
    return tuple(
        C.decompose_in_stabilizers(inverse(C.stabilizer_pauli(s)))
        for s in stabilizers)


def build_lrb_circuit(cliffords: Sequence[str],
                      depth: int,
                      p: float,
                      *,
                      terminal: str = "data5",
                      terminal_noise: bool = False,
                      forward_check_rounds: Optional[Iterable[int]] = None,
                      terminal_check: bool = True,
                      noise: Optional[SI1000] = SI1000(),
                      inverse_check_every: int = 0) -> LRBCircuit:
    """Build one encoded LRB circuit of depth ``depth``.

    Args:
        cliffords: Clifford descriptors; the first ``depth`` are used.
        depth: Benchmark depth ``m``.
        p: SI1000 noise parameter.
        terminal: ``"data2"`` measures data 0 and 3 only (the qutrit
            pipeline's ordinary readout); ``"data5"`` measures all five data
            qubits in the X basis, which adds two final X-stabilizer detectors
            without changing the logical observable ``x0 + x3``.
        terminal_noise: Apply SI1000 noise to the terminal readout block
            (the qutrit pipeline keeps it ideal).
        forward_check_rounds: Forward rounds (0..m-1) that are physically
            executed. ``None`` executes all of them, as the qutrit pipeline
            does for every policy.
        terminal_check: Execute the final check round after the inverse half.
        noise: SI1000 scale factors, or ``None`` for a noiseless circuit.
        inverse_check_every: If positive, also execute a check round after
            every ``inverse_check_every``-th inverse Clifford (a variant, not
            part of the qutrit protocol). These rounds get labels
            ``INVERSE_ROUND_OFFSET + j``; ``rounds`` lists all rounds in time
            order, so they appear before the terminal round ``depth``.

    Returns:
        The annotated circuit and its bookkeeping.
    """
    if terminal not in ("data2", "data5"):
        raise ValueError("terminal must be 'data2' or 'data5'.")
    if len(cliffords) < depth:
        raise ValueError("Clifford sequence is shorter than the depth.")
    forward = list(cliffords[:depth])
    forward_set = (set(range(depth)) if forward_check_rounds is None
                   else {int(r) for r in forward_check_rounds})
    if any(r < 0 or r >= depth for r in forward_set):
        raise ValueError("forward_check_rounds must lie in [0, depth).")

    circuit = stim.Circuit()
    emit_operations(circuit, C.encoding_operations(), p, None, C.NUM_QUBITS)

    rounds: list[int] = []
    round_records: dict[int, list[int]] = {}
    segments: list[list[C.Operation]] = []
    pending: list[C.Operation] = []

    def run_check_round(label: int, reset: bool = True) -> None:
        # The qutrit generator resets ancillas after every forward check round
        # but not after the terminal one (that reset layer would add idle
        # noise to the data right before the readout).
        operations = (C.check_round_operations() if reset else
                      C.z_stabilizer_check_operations()
                      + C.x_stabilizer_check_operations())
        records = emit_operations(circuit, operations, p, noise,
                                  C.NUM_QUBITS, noisy_measurements=True)
        rounds.append(label)
        round_records[label] = records
        segments.append(list(pending))
        pending.clear()

    for j, descriptor in enumerate(forward):
        ops = C.logical_clifford_operations(descriptor)
        emit_operations(circuit, ops, p, noise, C.NUM_QUBITS,
                        noisy_measurements=False)
        pending.extend(ops)
        if j in forward_set:
            run_check_round(j)

    inverse_label = INVERSE_ROUND_OFFSET
    for j, descriptor in enumerate(reversed(forward)):
        ops = C.logical_clifford_operations(descriptor, inverse=True)
        emit_operations(circuit, ops, p, noise, C.NUM_QUBITS,
                        noisy_measurements=False)
        pending.extend(ops)
        if (inverse_check_every > 0 and (j + 1) % inverse_check_every == 0
                and j + 1 < depth):
            run_check_round(inverse_label)
            inverse_label += 1

    if terminal_check:
        run_check_round(depth, reset=False)
    segments.append(list(pending))
    pending.clear()

    readout_qubits = (0, 3) if terminal == "data2" else C.DATA_QUBITS
    readout_ops = ([("H", (q,)) for q in readout_qubits]
                   + [("M", (q,)) for q in readout_qubits])
    readout_records = emit_operations(
        circuit, readout_ops, p, noise if terminal_noise else None,
        C.NUM_QUBITS, noisy_measurements=True)
    data_record = dict(zip(readout_qubits, readout_records))

    # Difference detectors, round by round.
    detector_keys: list[tuple[int, int]] = []
    back: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous: Optional[list[int]] = None
    for position, label in enumerate(rounds):
        records = round_records[label]
        if previous is None:
            subsets = tuple(() for _ in range(4))
        else:
            subsets = _back_subsets(segments[position], range(4))
        back[label] = subsets
        for s in range(4):
            targets = [records[s]] + _detector_from_back(previous, subsets[s])
            circuit.append(
                "DETECTOR",
                [stim.target_rec(r - circuit.num_measurements) for r in targets],
                [s, label])
            detector_keys.append((label, s))
        previous = records

    if terminal == "data5":
        subsets = _back_subsets(segments[-1], C.X_STABILIZERS)
        for s, subset in zip(C.X_STABILIZERS, subsets):
            support = [q for q, ch in enumerate(C.STABILIZER_PAULIS[s])
                       if ch != "_"]
            targets = [data_record[q] for q in support]
            targets += _detector_from_back(previous, subset)
            circuit.append(
                "DETECTOR",
                [stim.target_rec(r - circuit.num_measurements) for r in targets],
                [s, depth + 1])
            detector_keys.append((depth + 1, s))

    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(data_record[q] - circuit.num_measurements)
         for q in (0, 3)],
        0)

    return LRBCircuit(circuit=circuit, depth=depth, rounds=rounds,
                      detector_keys=detector_keys, back=back,
                      terminal=terminal, segments=segments)


def build_lrb_const0_circuit(cliffords: Sequence[str], depth: int, p: float,
                             *, noise: Optional[SI1000] = SI1000()
                             ) -> LRBCircuit:
    """``const=0`` protocol: no check rounds, ideal X readout of all data."""
    return build_lrb_circuit(cliffords, depth, p, terminal="data5",
                             terminal_noise=False, forward_check_rounds=(),
                             terminal_check=False, noise=noise)


def build_physical_rb_circuit(cliffords: Sequence[str], depth: int,
                              p_rb: float,
                              noise: Optional[SI1000] = SI1000()
                              ) -> stim.Circuit:
    """Single-qubit RB matched to the qutrit pipeline's physical RB.

    Every gate token is followed by ``DEPOLARIZE1(0.1 p_rb)``; the final
    Z measurement is ideal. Observable 0 is the measured bit (ideal outcome 0).
    """
    circuit = stim.Circuit()
    forward = list(cliffords[:depth])
    ops: list[C.Operation] = []
    for descriptor in forward:
        for token in C.clifford_gate_sequence(descriptor):
            ops.extend(C.physical_gate_operations(token))
    for descriptor in reversed(forward):
        for token in C.clifford_inverse_sequence(descriptor):
            ops.extend(C.physical_gate_operations(token))
    emit_operations(circuit, ops, p_rb, noise, 1, noisy_measurements=False)
    circuit.append("M", [0])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit


def random_clifford_sequence(length: int, rng: np.random.Generator
                             ) -> list[str]:
    """Uniformly random descriptors from the 24-element catalog."""
    picks = rng.integers(0, len(C.SINGLE_QUBIT_CLIFFORDS), size=length)
    return [C.SINGLE_QUBIT_CLIFFORDS[int(i)] for i in picks]
