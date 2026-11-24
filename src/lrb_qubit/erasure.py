"""Heralded erasures, flag qubits and an exact herald-conditioned MLD (WP I).

This module extends the lead's qubit ``[[5,1,2]]`` LRB model without touching
it. ``build_instrumented_lrb`` re-implements ``circuits.build_lrb_circuit``
(default protocol: every forward round checked, ``data5`` or ``data2``
terminal, ideal encoding and readout) with three optional additions:

* **Erasure conversion.** ``ErasureSpec(two_qubit=r)`` heralds a fraction
  ``r`` of the depolarizing events after every two-qubit gate (check CNOTs and
  the logical SWAP/CZ); ``ErasureSpec(meas_idle=r)`` does the same for the
  measurement-idle ``DEPOLARIZE1(2p)`` channels. ``DEPOLARIZE2(q)`` equals
  "replace both qubits by the maximally mixed state with probability
  ``16q/15``"; a fraction ``r`` of these replacement events is heralded, so the
  marginal Pauli channel is *exactly* the SI1000 channel for every ``r``.
  Only the side information changes. Two-qubit locations are sampled with a
  ``CORRELATED_ERROR``/``ELSE_CORRELATED_ERROR`` chain whose heralded terms
  also flip a dedicated herald qubit (measured with ``MR``). One-qubit
  locations use Stim's ``HERALDED_ERASE(4rq/3)`` followed by
  ``DEPOLARIZE1((1-r)q/(1-4rq/3))``, which is exactly the same channel.
  Herald records become detectors appended after the lead's detector layout.
* **Flag qubits** on all four weight-3 checks (``flags="noisy"`` with SI1000
  noise on the flag operations, ``flags="ideal"`` for noiseless flags). Flag
  qubits 9-12 belong to checks S_Z1, S_Z2, S_X1, S_X2. Each flag is coupled to
  its ancilla by two CNOTs between the first and the last data CNOT. A flag
  detector lives in its round's time slot, so the lead's ``TransferMatrixMLD``
  decodes the flagged circuit exactly (8 detectors per slot).
* **Probe mode**, used to compute the detection-event signature of every
  single-qubit Pauli component at every noise location (``probe_signatures``).

``mode="sample"`` gives the circuit to sample from (true channel plus herald
detectors), ``mode="decode"`` gives the decoder's prior circuit (every
converted channel replaced by its exact *no-herald conditional* channel) and
``mode="probe"`` gives a noiseless circuit with location markers.

``HeraldedMLD`` is the slot-aware exact decoder. Stim's own DEM of
``HERALDED_ERASE`` or ``ELSE_CORRELATED_ERROR`` needs
``approximate_disjoint_errors=True`` and treats the disjoint outcomes as
independent; we avoid it. Conditioned on the observed herald pattern, the
noise is again a product of independent mechanisms: the no-herald conditional
channels (the ``decode`` DEM) everywhere, and, at each heralded location, each
single-qubit X/Z component with probability 1/2 (uniform Pauli). Convolving a
uniform component into the slot kernel absorbs the location's base channel,
so ``P(syndrome, L | heralds)`` is exact. Herald bits are not part of the
transfer-matrix state; each herald modifies the kernels of the slots in which
its components live, and per-shot kernel variants are grouped and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import stim

from . import baseline as B
from . import circuits as Q
from . import code as C
from . import mld as M

ONE_QUBIT_GATES = frozenset({"I", "X", "Z", "H", "S", "S_DAG"})
TWO_QUBIT_GATES = frozenset({"CX", "CZ", "SWAP"})
FLAG_QUBITS = (9, 10, 11, 12)
PAULI_TARGET = {1: stim.target_x, 2: stim.target_y, 3: stim.target_z}


# ---------------------------------------------------------------------------
# Specifications and bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErasureSpec:
    """Fractions of depolarizing events converted into heralded erasures."""

    two_qubit: float = 0.0
    meas_idle: float = 0.0

    def __post_init__(self):
        for name in ("two_qubit", "meas_idle"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")

    @property
    def active(self) -> bool:
        return self.two_qubit > 0.0 or self.meas_idle > 0.0


@dataclass
class NoiseLocation:
    """One independent noise channel instance of the SI1000 circuit.

    Attributes:
        kind: ``cx_check``, ``cx_flag``, ``gate2_logical``, ``gate1_check``,
            ``gate1_logical``, ``meas_idle``, ``meas_depol``, ``meas_flip``,
            ``reset_idle``, ``reset_flip`` or ``terminal`` (a zero-probability
            marker just before the ideal terminal readout).
        channel: ``DEPOLARIZE1``, ``DEPOLARIZE2``, ``X_ERROR`` or ``NONE``.
        probability: Marginal channel parameter (the SI1000 value).
        qubits: Qubits the channel acts on.
        slot: Number of check rounds completed before this location, i.e. the
            time slot of the first round that can see it.
        gate: Gate after which (or measurement/reset before/after which) the
            channel acts.
        herald_fraction: Fraction ``r`` of depolarizing events heralded here.
        herald: Herald column (index among the herald detectors) or -1.
        step: Position of the location in the noise stream (time order).
        check: For check-round CNOT locations, ``(stabilizer, n)`` with ``n``
            the number of that check's data CNOTs done up to and including
            this gate (an ancilla fault here spreads to ``3 - n`` data
            qubits); ``(-1, -1)`` otherwise.
    """

    kind: str
    channel: str
    probability: float
    qubits: tuple[int, ...]
    slot: int
    gate: str
    herald_fraction: float = 0.0
    herald: int = -1
    step: int = 0
    check: tuple[int, int] = (-1, -1)


@dataclass
class InstrumentedLRB:
    """An instrumented LRB circuit plus bookkeeping.

    ``lrb`` is a ``circuits.LRBCircuit`` view of the same circuit (rounds,
    detector keys, back subsets), so ``lrb.raw_syndromes`` and
    ``mld.lrb_detector_slots(lrb)`` work on the first ``num_main_detectors``
    detection-event columns, which have exactly the lead's layout.
    """

    circuit: stim.Circuit
    lrb: Q.LRBCircuit
    mode: str
    locations: list[NoiseLocation]
    num_main_detectors: int
    flag_detectors: list[int] = field(default_factory=list)
    flag_slots: list[int] = field(default_factory=list)
    herald_detectors: list[int] = field(default_factory=list)
    herald_locations: list[int] = field(default_factory=list)
    markers: dict[int, list[int]] = field(default_factory=dict)

    @property
    def num_heralds(self) -> int:
        return len(self.herald_locations)

    def decoder_slots(self) -> np.ndarray:
        """Slots of the main (and flag) detectors, herald detectors excluded."""
        main = M.lrb_detector_slots(self.lrb)
        if not self.flag_detectors:
            return main
        return np.concatenate([main, np.asarray(self.flag_slots, np.int64)])

    def split(self, detection_events: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """Split sampled events into (decoder detectors, herald bits)."""
        n_dec = self.num_main_detectors + len(self.flag_detectors)
        return (detection_events[:, :n_dec],
                detection_events[:, n_dec:n_dec + self.num_heralds])


# ---------------------------------------------------------------------------
# Noise emission (mirrors circuits.emit_operations)
# ---------------------------------------------------------------------------


class _Emitter:
    """Appends operations plus SI1000 noise, optionally converted to erasures.

    In ``decode`` mode with no conversion and no flags the output is
    instruction-for-instruction identical to ``circuits.emit_operations``.
    """

    def __init__(self, circuit: stim.Circuit, p: float,
                 noise: Optional[Q.SI1000], *, noise_qubits: Sequence[int],
                 noiseless_qubits: Iterable[int], erasure: ErasureSpec,
                 mode: str, herald_qubit: int):
        self.circuit = circuit
        self.p = p
        self.noise = noise
        self.noisy = noise is not None and p > 0.0
        self.noise_qubits = list(noise_qubits)
        self.noiseless = set(noiseless_qubits)
        self.erasure = erasure
        self.mode = mode
        self.herald_qubit = herald_qubit
        self.locations: list[NoiseLocation] = []
        self.markers: dict[int, list[int]] = {}
        self.herald_records: list[int] = []
        self.herald_locations: list[int] = []
        self.slot = 0
        self.check: dict[tuple[int, ...], tuple[int, int]] = {}

    # -- locations ---------------------------------------------------------

    def _location(self, kind: str, channel: str, probability: float,
                  qubits: Sequence[int], gate: str, herald_fraction: float,
                  check: tuple[int, int] = (-1, -1)) -> int:
        loc = NoiseLocation(kind, channel, float(probability), tuple(qubits),
                            self.slot, gate, float(herald_fraction),
                            step=len(self.locations), check=check)
        if herald_fraction > 0.0:
            loc.herald = len(self.herald_locations)
            self.herald_locations.append(len(self.locations))
        self.locations.append(loc)
        return len(self.locations) - 1

    def _mark(self, location_indices: Sequence[int]) -> None:
        self.circuit.append("TICK")
        self.markers[len(self.circuit) - 1] = list(location_indices)

    def terminal_marker(self, qubits: Sequence[int]) -> None:
        locs = [self._location("terminal", "NONE", 0.0, (q,), "readout", 0.0)
                for q in qubits]
        if self.mode == "probe":
            self._mark(locs)

    # -- channels ----------------------------------------------------------

    def _depolarize1(self, kind: str, qubits: Sequence[int], prob: float,
                     gate: str, herald_fraction: float = 0.0,
                     check: tuple[int, int] = (-1, -1)) -> None:
        qubits = list(qubits)
        if prob <= 0.0 or not qubits:
            return
        locs = [self._location(kind, "DEPOLARIZE1", prob, (q,), gate,
                               herald_fraction, check) for q in qubits]
        if self.mode == "probe":
            self._mark(locs)
            return
        r = herald_fraction
        if r <= 0.0:
            self.circuit.append("DEPOLARIZE1", qubits, prob)
            return
        erase = 4.0 * r * prob / 3.0
        if erase >= 1.0:
            raise ValueError("Erasure probability exceeds 1.")
        if self.mode == "sample":
            first = self.circuit.num_measurements
            self.circuit.append("HERALDED_ERASE", qubits, erase)
            self.herald_records.extend(range(first, first + len(qubits)))
        residual = (1.0 - r) * prob / (1.0 - erase)
        if residual > 0.0:
            self.circuit.append("DEPOLARIZE1", qubits, residual)

    def _depolarize2(self, kind: str, pair: Sequence[int], prob: float,
                     gate: str, herald_fraction: float = 0.0,
                     check: tuple[int, int] = (-1, -1)) -> None:
        pair = tuple(pair)
        if prob <= 0.0:
            return
        loc = self._location(kind, "DEPOLARIZE2", prob, pair, gate,
                             herald_fraction, check)
        if self.mode == "probe":
            self._mark([loc])
            return
        r = herald_fraction
        if r <= 0.0:
            self.circuit.append("DEPOLARIZE2", list(pair), prob)
            return
        erase = 16.0 * r * prob / 15.0
        if erase >= 1.0:
            raise ValueError("Erasure probability exceeds 1.")
        if self.mode == "decode":
            residual = (1.0 - r) * prob / (1.0 - erase)
            if residual > 0.0:
                self.circuit.append("DEPOLARIZE2", list(pair), residual)
            return
        # sample mode: one exact chain of mutually exclusive outcomes.
        a, b = pair
        terms = []
        for pa in range(4):
            for pb in range(4):
                terms.append((r * prob / 15.0, True, pa, pb))
        if r < 1.0:
            for pa in range(4):
                for pb in range(4):
                    if pa or pb:
                        terms.append(((1.0 - r) * prob / 15.0, False, pa, pb))
        remaining = 1.0
        for index, (mass, heralded, pa, pb) in enumerate(terms):
            targets = [stim.target_x(self.herald_qubit)] if heralded else []
            if pa:
                targets.append(PAULI_TARGET[pa](a))
            if pb:
                targets.append(PAULI_TARGET[pb](b))
            conditional = min(1.0, mass / remaining)
            name = "CORRELATED_ERROR" if index == 0 else "ELSE_CORRELATED_ERROR"
            self.circuit.append(name, targets, conditional)
            remaining -= mass
        self.herald_records.append(self.circuit.num_measurements)
        self.circuit.append("MR", [self.herald_qubit])

    def _x_error(self, kind: str, qubit: int, prob: float, gate: str) -> None:
        if prob <= 0.0:
            return
        loc = self._location(kind, "X_ERROR", prob, (qubit,), gate, 0.0)
        if self.mode == "probe":
            self._mark([loc])
            return
        self.circuit.append("X_ERROR", [qubit], prob)

    # -- operations --------------------------------------------------------

    def _kind(self, targets: Sequence[int], two: bool) -> str:
        if any(t in FLAG_QUBITS for t in targets) and two:
            return "cx_flag"
        if any(t >= C.NUM_DATA for t in targets):
            return "cx_check" if two else "gate1_check"
        return "gate2_logical" if two else "gate1_logical"

    def emit(self, operations: Sequence[C.Operation],
             noisy_measurements: bool = True) -> list[int]:
        """Append operations with noise; return measurement-record indices."""
        records: list[int] = []
        noise = self.noise
        index = 0
        while index < len(operations):
            gate, targets = operations[index]
            if gate in ("M", "R"):
                end = index
                while end < len(operations) and operations[end][0] == gate:
                    end += 1
                layer = [operations[i][1][0] for i in range(index, end)]
                if gate == "M":
                    if self.noisy and noisy_measurements:
                        idle = [q for q in self.noise_qubits if q not in layer]
                        label = "M" + ",".join(str(q) for q in layer)
                        self._depolarize1("meas_idle", idle,
                                          noise.meas_idle * self.p, label,
                                          self.erasure.meas_idle)
                    for q in layer:
                        if (self.noisy and noisy_measurements
                                and q not in self.noiseless):
                            self._depolarize1("meas_depol", [q],
                                              noise.meas_depol * self.p, "M")
                            self._x_error("meas_flip", q,
                                          noise.meas_flip * self.p, "M")
                        records.append(self.circuit.num_measurements)
                        self.circuit.append("M", [q])
                else:
                    if self.noisy:
                        idle = [q for q in self.noise_qubits if q not in layer]
                        self._depolarize1("reset_idle", idle,
                                          noise.reset_idle * self.p, "R")
                    for q in layer:
                        self.circuit.append("R", [q])
                        if self.noisy and q not in self.noiseless:
                            self._x_error("reset_flip", q,
                                          noise.reset_flip * self.p, "R")
                index = end
                continue
            self.circuit.append(gate, list(targets))
            if self.noisy and not (set(targets) & self.noiseless):
                check = self.check.get((index,), (-1, -1))
                if gate in ONE_QUBIT_GATES:
                    self._depolarize1(self._kind(targets, False), targets,
                                      noise.one_qubit * self.p, gate,
                                      check=check)
                elif gate in TWO_QUBIT_GATES:
                    self._depolarize2(self._kind(targets, True), targets,
                                      noise.two_qubit * self.p, gate,
                                      self.erasure.two_qubit, check=check)
                else:
                    raise ValueError(f"No SI1000 rule for gate {gate!r}.")
            index += 1
        return records


# ---------------------------------------------------------------------------
# Check rounds with optional flags
# ---------------------------------------------------------------------------


def _check_supports() -> list[tuple[int, tuple[int, int, int]]]:
    """(ancilla, data CNOT order) per stabilizer, from code.py's check ops."""
    ops = C.z_stabilizer_check_operations() + C.x_stabilizer_check_operations()
    supports = []
    for s, ancilla in enumerate(C.STABILIZER_ANCILLA):
        data = []
        for gate, targets in ops:
            if gate == "CX" and ancilla in targets:
                data.append(targets[0] if targets[1] == ancilla else targets[1])
        supports.append((ancilla, tuple(data)))
    return supports


CHECK_SUPPORTS = _check_supports()


FLAG_LAYOUTS = {
    # positions of the two flag CNOTs among the three data CNOTs
    # "standard": d1 F d2 F d3 (Chao-Reichardt style: after the first and
    #   before the last data CNOT); flags ancilla faults that occur after
    #   the first flag CNOT, i.e. NOT the fault right after d1.
    # "wrap": F d1 d2 F d3; flags every ancilla fault after d1 or d2, i.e.
    #   every ancilla fault that spreads to data.
    "standard": (1, 2),
    "wrap": (0, 2),
}


def flagged_check_operations(layout: str = "standard") -> list[C.Operation]:
    """The four checks, each with one flag qubit coupled by two CNOTs.

    Z checks (CNOT data->ancilla): the flag starts in |+> (H), the flag CNOTs
    are flag->ancilla, and the flag is read in the X basis, so a Z on the
    ancilla between the two flag CNOTs flips it. X checks (CNOT
    ancilla->data): the flag starts in |0>, the flag CNOTs are
    ancilla->flag, and an X on the ancilla between them flips it. The flag
    is measured in the same layer as its ancilla.
    """
    first, second = FLAG_LAYOUTS[layout]
    ops: list[C.Operation] = []
    for s, (ancilla, data) in enumerate(CHECK_SUPPORTS):
        flag = FLAG_QUBITS[s]
        z_check = s in C.Z_STABILIZERS
        flag_cx = (flag, ancilla) if z_check else (ancilla, flag)
        body: list[C.Operation] = []
        for k, d in enumerate(data):
            if k in (first, second):
                body.append(("CX", flag_cx))
            body.append(("CX", (d, ancilla) if z_check else (ancilla, d)))
        if z_check:
            ops += [("H", (flag,))] + body + [("H", (flag,)),
                                              ("M", (flag,)), ("M", (ancilla,))]
        else:
            ops += [("H", (ancilla,))] + body + [("H", (ancilla,)),
                                                 ("M", (flag,)), ("M", (ancilla,))]
    return ops


def _check_positions(ops: Sequence[C.Operation]) -> dict[tuple[int], tuple[int, int]]:
    """Map op index -> (stabilizer, data CNOTs of that check done so far).

    For a data CNOT at position k (0-based) the count is k + 1; for a flag
    CNOT it is the number of data CNOTs before it. An ancilla fault right
    after the gate spreads to ``3 - count`` data qubits.
    """
    positions = {}
    count = {s: 0 for s in range(4)}
    for index, (gate, targets) in enumerate(ops):
        if gate != "CX":
            continue
        for s, (ancilla, data) in enumerate(CHECK_SUPPORTS):
            if ancilla in targets:
                other = targets[0] if targets[1] == ancilla else targets[1]
                if other in data:
                    count[s] += 1
                positions[(index,)] = (s, count[s])
    return positions


def _back_subsets(segment: Sequence[C.Operation],
                  stabilizers: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Same as ``circuits._back_subsets`` (re-implemented, not imported)."""
    inverse = C.data_tableau(segment).inverse()
    return tuple(C.decompose_in_stabilizers(inverse(C.stabilizer_pauli(s)))
                 for s in stabilizers)


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------


def build_instrumented_lrb(cliffords: Sequence[str], depth: int, p: float, *,
                           terminal: str = "data5",
                           noise: Optional[Q.SI1000] = Q.SI1000(),
                           erasure: Optional[ErasureSpec] = None,
                           flags: Optional[str] = None,
                           flag_layout: str = "standard",
                           mode: str = "sample") -> InstrumentedLRB:
    """LRB circuit of depth ``depth`` with erasure conversion and/or flags.

    The protocol is the lead's default one (all forward rounds checked, ideal
    encoding and terminal readout). See the module docstring for ``mode``;
    ``flag_layout`` is a key of ``FLAG_LAYOUTS``.
    """
    if mode not in ("sample", "decode", "probe"):
        raise ValueError("mode must be 'sample', 'decode' or 'probe'.")
    if flags not in (None, "noisy", "ideal"):
        raise ValueError("flags must be None, 'noisy' or 'ideal'.")
    if terminal not in ("data2", "data5"):
        raise ValueError("terminal must be 'data2' or 'data5'.")
    if len(cliffords) < depth:
        raise ValueError("Clifford sequence is shorter than the depth.")
    erasure = erasure or ErasureSpec()
    forward = list(cliffords[:depth])
    flag_qubits = FLAG_QUBITS if flags else ()
    num_qubits = C.NUM_QUBITS + len(flag_qubits)
    noise_qubits = (range(num_qubits) if flags != "ideal"
                    else range(C.NUM_QUBITS))
    circuit = stim.Circuit()
    Q.emit_operations(circuit, C.encoding_operations(), p, None, C.NUM_QUBITS)
    em = _Emitter(circuit, p, noise, noise_qubits=noise_qubits,
                  noiseless_qubits=flag_qubits if flags == "ideal" else (),
                  erasure=erasure, mode=mode, herald_qubit=num_qubits)

    rounds: list[int] = []
    round_records: dict[int, list[int]] = {}
    flag_records: dict[int, list[int]] = {}
    segments: list[list[C.Operation]] = []
    pending: list[C.Operation] = []

    def run_check_round(label: int, reset: bool = True) -> None:
        if flags:
            ops = flagged_check_operations(flag_layout)
        else:
            ops = (C.z_stabilizer_check_operations()
                   + C.x_stabilizer_check_operations())
        if reset:
            ops = ops + C.reset_operations() + [("R", (f,)) for f in flag_qubits]
        em.slot = len(rounds)
        em.check = _check_positions(ops)
        records = em.emit(ops, noisy_measurements=True)
        em.check = {}
        measured = [t[0] for g, t in ops if g == "M"]
        record_of = dict(zip(measured, records))
        round_records[label] = [record_of[a] for a in C.STABILIZER_ANCILLA]
        flag_records[label] = [record_of[f] for f in flag_qubits]
        rounds.append(label)
        segments.append(list(pending))
        pending.clear()

    for j, descriptor in enumerate(forward):
        ops = C.logical_clifford_operations(descriptor)
        em.slot = len(rounds)
        em.emit(ops, noisy_measurements=False)
        pending.extend(ops)
        run_check_round(j)
    for descriptor in reversed(forward):
        ops = C.logical_clifford_operations(descriptor, inverse=True)
        em.slot = len(rounds)
        em.emit(ops, noisy_measurements=False)
        pending.extend(ops)
    run_check_round(depth, reset=False)
    segments.append(list(pending))
    pending.clear()

    readout_qubits = (0, 3) if terminal == "data2" else C.DATA_QUBITS
    em.slot = len(rounds)
    em.terminal_marker(readout_qubits)
    readout_ops = ([("H", (q,)) for q in readout_qubits]
                   + [("M", (q,)) for q in readout_qubits])
    readout_records = Q.emit_operations(circuit, readout_ops, p, None,
                                        C.NUM_QUBITS, noisy_measurements=True)
    data_record = dict(zip(readout_qubits, readout_records))

    def rec(targets: Sequence[int]) -> list:
        return [stim.target_rec(r - circuit.num_measurements) for r in targets]

    detector_keys: list[tuple[int, int]] = []
    back: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous: Optional[list[int]] = None
    for position, label in enumerate(rounds):
        records = round_records[label]
        subsets = (tuple(() for _ in range(4)) if previous is None
                   else _back_subsets(segments[position], range(4)))
        back[label] = subsets
        for s in range(4):
            targets = [records[s]]
            if previous is not None:
                targets += [previous[k] for k in subsets[s]]
            circuit.append("DETECTOR", rec(targets), [s, label])
            detector_keys.append((label, s))
        previous = records
    if terminal == "data5":
        subsets = _back_subsets(segments[-1], C.X_STABILIZERS)
        for s, subset in zip(C.X_STABILIZERS, subsets):
            support = [q for q, ch in enumerate(C.STABILIZER_PAULIS[s])
                       if ch != "_"]
            targets = [data_record[q] for q in support]
            targets += [previous[k] for k in subset]
            circuit.append("DETECTOR", rec(targets), [s, depth + 1])
            detector_keys.append((depth + 1, s))
    num_main = len(detector_keys)

    flag_detectors: list[int] = []
    flag_slots: list[int] = []
    for position, label in enumerate(rounds):
        for k, record in enumerate(flag_records[label]):
            circuit.append("DETECTOR", rec([record]), [4 + k, label])
            flag_detectors.append(num_main + len(flag_detectors))
            flag_slots.append(position)

    herald_detectors: list[int] = []
    if mode == "sample":
        first = num_main + len(flag_detectors)
        for k, record in enumerate(em.herald_records):
            loc = em.locations[em.herald_locations[k]]
            circuit.append("DETECTOR", rec([record]), [9, loc.slot, k])
            herald_detectors.append(first + k)

    circuit.append("OBSERVABLE_INCLUDE", rec([data_record[0], data_record[3]]),
                   0)

    lrb = Q.LRBCircuit(circuit=circuit, depth=depth, rounds=rounds,
                       detector_keys=detector_keys, back=back,
                       terminal=terminal, segments=segments)
    return InstrumentedLRB(circuit=circuit, lrb=lrb, mode=mode,
                           locations=em.locations, num_main_detectors=num_main,
                           flag_detectors=flag_detectors, flag_slots=flag_slots,
                           herald_detectors=herald_detectors,
                           herald_locations=(list(em.herald_locations)
                                             if mode == "sample" else
                                             list(em.herald_locations)),
                           markers=em.markers)


# ---------------------------------------------------------------------------
# Probe: signatures of single-qubit Pauli components at every location
# ---------------------------------------------------------------------------


@dataclass
class Signatures:
    """Detection-event signature of X and Z on each qubit of each location.

    ``lanes[k] = (location, qubit, 'X' | 'Z')``; ``dets[k]`` (bool, over all
    probe-circuit detectors) and ``obs[k]`` are its detector and observable
    flips. ``lane_of[(location, qubit, pauli)] = k``.
    """

    lanes: list[tuple[int, int, str]]
    dets: np.ndarray
    obs: np.ndarray
    lane_of: dict[tuple[int, int, str], int]

    def pauli(self, location: int, paulis: dict[int, int]
              ) -> tuple[np.ndarray, bool]:
        """Signature of a Pauli (qubit -> 1 X, 2 Y, 3 Z) at a location."""
        dets = np.zeros(self.dets.shape[1], dtype=bool)
        obs = False
        for qubit, pauli in paulis.items():
            if pauli in (1, 2):
                k = self.lane_of[(location, qubit, "X")]
                dets ^= self.dets[k]
                obs ^= bool(self.obs[k])
            if pauli in (2, 3):
                k = self.lane_of[(location, qubit, "Z")]
                dets ^= self.dets[k]
                obs ^= bool(self.obs[k])
        return dets, obs


def probe_signatures(probe: InstrumentedLRB,
                     locations: Optional[Sequence[int]] = None) -> Signatures:
    """Propagate X and Z at every (or the given) location through the circuit.

    Uses ``stim.FlipSimulator`` with stabilizer randomization disabled: one
    simulation lane per (location, qubit, Pauli), the Pauli is injected at the
    location's marker, and the detector/observable flips are read at the end.
    """
    if probe.mode != "probe":
        raise ValueError("probe_signatures needs a mode='probe' circuit.")
    chosen = (range(len(probe.locations)) if locations is None
              else sorted(set(locations)))
    lanes: list[tuple[int, int, str]] = []
    lane_of: dict[tuple[int, int, str], int] = {}
    for li in chosen:
        for q in probe.locations[li].qubits:
            for pauli in ("X", "Z"):
                lane_of[(li, q, pauli)] = len(lanes)
                lanes.append((li, q, pauli))
    num_qubits = probe.circuit.num_qubits
    batch = max(1, len(lanes))
    sim = stim.FlipSimulator(batch_size=batch, num_qubits=num_qubits,
                             disable_stabilizer_randomization=True, seed=0)
    wanted = set(chosen)
    for index, instruction in enumerate(probe.circuit):
        sim.do(instruction)
        locs = probe.markers.get(index)
        if not locs:
            continue
        mask_x = np.zeros((num_qubits, batch), dtype=bool)
        mask_z = np.zeros((num_qubits, batch), dtype=bool)
        hit = False
        for li in locs:
            if li not in wanted:
                continue
            for q in probe.locations[li].qubits:
                mask_x[q, lane_of[(li, q, "X")]] = True
                mask_z[q, lane_of[(li, q, "Z")]] = True
                hit = True
        if hit:
            sim.broadcast_pauli_errors(pauli="X", mask=mask_x)
            sim.broadcast_pauli_errors(pauli="Z", mask=mask_z)
    dets = sim.get_detector_flips().T[:len(lanes)].copy()
    obs = sim.get_observable_flips()[0][:len(lanes)].copy()
    return Signatures(lanes=lanes, dets=dets, obs=obs, lane_of=lane_of)


def herald_components(sample: InstrumentedLRB, signatures: Signatures,
                      num_detectors: Optional[int] = None
                      ) -> list[list[tuple[tuple[int, ...], bool]]]:
    """Per herald column: X/Z component signatures of its location's qubits.

    Location indices agree between ``sample``, ``decode`` and ``probe``
    circuits built with the same arguments (the location stream does not
    depend on the mode).
    """
    n = (sample.num_main_detectors + len(sample.flag_detectors)
         if num_detectors is None else num_detectors)
    comps = []
    for loc_index in sample.herald_locations:
        items = []
        for q in sample.locations[loc_index].qubits:
            for pauli in ("X", "Z"):
                k = signatures.lane_of[(loc_index, q, pauli)]
                items.append((tuple(np.flatnonzero(signatures.dets[k][:n])
                                    .tolist()), bool(signatures.obs[k])))
        comps.append(items)
    return comps


# ---------------------------------------------------------------------------
# Exact herald-conditioned transfer-matrix MLD
# ---------------------------------------------------------------------------


class HeraldedMLD:
    """Exact MLD given the syndrome and a per-shot set of erased locations.

    Args:
        dem: DEM of the no-herald conditional noise (``mode="decode"``
            circuit), or ``None`` for "no noise except the erasures" (used by
            the herald-explanation policy).
        detector_slots: Slot of every decoder detector.
        components: For each herald column, the list of
            ``(detector_indices, flips_observable)`` of the independent
            uniform components of an erasure there (X and Z on each erased
            qubit).

    With no herald columns this equals ``mld.TransferMatrixMLD`` (tested).
    """

    def __init__(self, dem: Optional[stim.DetectorErrorModel],
                 detector_slots: Sequence[int],
                 components: Sequence[Sequence[tuple[Sequence[int], bool]]] = ()):
        slots = np.asarray(detector_slots, dtype=np.int64)
        if dem is not None and dem.num_detectors != len(slots):
            raise ValueError("Need one slot per DEM detector.")
        self.slots = slots
        self.num_slots = int(slots.max()) + 1 if len(slots) else 1
        self.slot_detectors = [np.flatnonzero(slots == t)
                               for t in range(self.num_slots)]
        self.slot_sizes = [len(d) for d in self.slot_detectors]
        if max(self.slot_sizes) > 8:
            raise ValueError("Slots larger than 8 detectors are not supported.")
        self.position = np.zeros(len(slots), dtype=np.int64)
        for dets in self.slot_detectors:
            self.position[dets] = np.arange(len(dets))
        self.kernels = []
        for t in range(self.num_slots):
            size = 2 ** (self._n_prev(t) + self.slot_sizes[t] + 1)
            kernel = np.zeros(size)
            kernel[0] = 1.0
            self.kernels.append(kernel)
        if dem is not None:
            for mech in M.dem_mechanisms(dem):
                if not mech.detectors and not mech.flips_observable:
                    continue
                t, vector = self._place(mech.detectors, mech.flips_observable)
                _convolve(self.kernels[t], vector, mech.probability)
        self.initial = self.kernels[0].copy()
        self.transfer = [None] + [self._transfer_matrix(t, self.kernels[t])
                                  for t in range(1, self.num_slots)]
        self.herald_vectors: list[dict[int, list[int]]] = []
        slot_heralds: list[list[int]] = [[] for _ in range(self.num_slots)]
        for h, comps in enumerate(components):
            per: dict[int, list[int]] = {}
            for dets, flips in comps:
                if not len(dets) and not flips:
                    continue
                t, vector = self._place(tuple(dets), bool(flips))
                per.setdefault(t, []).append(vector)
            self.herald_vectors.append(per)
            for t in per:
                slot_heralds[t].append(h)
        self.num_heralds = len(self.herald_vectors)
        self.slot_heralds = [np.asarray(sorted(x), dtype=np.int64)
                             for x in slot_heralds]
        self._cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    # -- construction helpers ---------------------------------------------

    def _n_prev(self, t: int) -> int:
        return self.slot_sizes[t - 1] if t > 0 else 0

    def _place(self, detectors: Sequence[int], flips: bool) -> tuple[int, int]:
        if not len(detectors):
            return 0, (1 << self.slot_sizes[0]) if flips else 0
        mech_slots = self.slots[list(detectors)]
        t = int(mech_slots.max())
        if int(mech_slots.min()) < t - 1:
            raise ValueError("Mechanism spans non-adjacent slots.")
        n_prev = self._n_prev(t)
        n_cur = self.slot_sizes[t]
        vector = 0
        for det in detectors:
            bit = int(self.position[det])
            vector ^= (1 << (n_prev + bit)) if self.slots[det] == t else (1 << bit)
        if flips:
            vector ^= 1 << (n_prev + n_cur)
        return t, vector

    def _transfer_matrix(self, t: int, kernel: np.ndarray) -> np.ndarray:
        n_prev = self._n_prev(t)
        n_cur = self.slot_sizes[t]
        k = kernel.reshape(2, 2 ** n_cur, 2 ** n_prev)
        matrix = np.zeros((2 * 2 ** n_prev, 2 * 2 ** n_cur))
        for l_in in range(2):
            for l_out in range(2):
                matrix[l_in * 2 ** n_prev:(l_in + 1) * 2 ** n_prev,
                       l_out * 2 ** n_cur:(l_out + 1) * 2 ** n_cur] = \
                    k[l_in ^ l_out].T
        return matrix

    def _modified(self, t: int, active: tuple[int, ...]) -> np.ndarray:
        key = (t, active)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kernel = self.kernels[t].copy()
        for h in active:
            for vector in self.herald_vectors[h].get(t, ()):
                _convolve(kernel, vector, 0.5)
        result = kernel if t == 0 else self._transfer_matrix(t, kernel)
        self._cache[key] = result
        return result

    def _groups(self, heralds: Optional[np.ndarray], t: int):
        if heralds is None or self.num_heralds == 0:
            return []
        cols = self.slot_heralds[t]
        if cols.size == 0:
            return []
        sub = heralds[:, cols]
        rows = np.flatnonzero(sub.any(axis=1))
        if rows.size == 0:
            return []
        patterns = sub[rows]
        packed = np.packbits(patterns, axis=1)
        keys = np.ascontiguousarray(packed).view(
            np.dtype((np.void, packed.shape[1]))).ravel()
        _, first, inverse = np.unique(keys, return_index=True,
                                      return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(inverse, kind="stable")
        bounds = np.cumsum(np.bincount(inverse, minlength=len(first)))
        out = []
        start = 0
        for u, row0 in enumerate(first):
            active = tuple(int(h) for h in cols[patterns[row0]])
            out.append((rows[order[start:bounds[u]]], active))
            start = bounds[u]
        return out

    # -- decoding ----------------------------------------------------------

    def _slot_values(self, events: np.ndarray, t: int) -> np.ndarray:
        dets = self.slot_detectors[t]
        weights = (1 << np.arange(len(dets))).astype(np.int64)
        return events[:, dets].astype(np.int64) @ weights

    def log_joint(self, detection_events: np.ndarray,
                  heralds: Optional[np.ndarray] = None,
                  batch_size: int = 65536) -> tuple[np.ndarray, np.ndarray]:
        """``log P(s, L=0 | h)`` and ``log P(s, L=1 | h)`` up to one common
        per-shot-independent constant (exact; ``-inf`` if impossible)."""
        events = np.asarray(detection_events, dtype=bool)
        if heralds is not None:
            heralds = np.asarray(heralds, dtype=bool)
            if heralds.shape[1] != self.num_heralds:
                raise ValueError("Herald matrix has the wrong width.")
        out0 = np.empty(events.shape[0])
        out1 = np.empty(events.shape[0])
        for start in range(0, events.shape[0], batch_size):
            stop = start + batch_size
            chunk_h = None if heralds is None else heralds[start:stop]
            a, b = self._chunk(events[start:stop], chunk_h)
            out0[start:start + len(a)] = a
            out1[start:start + len(b)] = b
        return out0, out1

    def log_likelihood_ratio(self, detection_events: np.ndarray,
                             heralds: Optional[np.ndarray] = None,
                             batch_size: int = 65536) -> np.ndarray:
        """``log P(L=0 | s, h) - log P(L=1 | s, h)``."""
        l0, l1 = self.log_joint(detection_events, heralds, batch_size)
        with np.errstate(invalid="ignore"):
            return l0 - l1

    def decode(self, detection_events: np.ndarray,
               heralds: Optional[np.ndarray] = None
               ) -> tuple[np.ndarray, np.ndarray]:
        llr = self.log_likelihood_ratio(detection_events, heralds)
        return llr < 0, np.abs(llr)

    def _chunk(self, events: np.ndarray, heralds: Optional[np.ndarray]):
        shots = events.shape[0]
        psi = np.broadcast_to(self.initial, (shots, self.initial.size)).copy()
        for rows, active in self._groups(heralds, 0):
            psi[rows] = self._modified(0, active)
        log_scale = np.zeros(shots)
        with np.errstate(divide="ignore"):
            for t in range(1, self.num_slots):
                n_prev = self.slot_sizes[t - 1]
                s = self._slot_values(events, t - 1)
                size = 2 ** n_prev
                base = np.arange(2 * size)
                cols = (base[None, :] & (size - 1)) ^ s[:, None]
                cols |= (base[None, :] >> n_prev) << n_prev
                phi = np.take_along_axis(psi, cols, axis=1)
                psi = phi @ self.transfer[t]
                for rows, active in self._groups(heralds, t):
                    psi[rows] = phi[rows] @ self._modified(t, active)
                peak = psi.max(axis=1)
                good = peak > 0
                psi[good] /= peak[good, None]
                log_scale += np.log(np.where(good, peak, 0.0))
            n_last = self.slot_sizes[-1]
            s = self._slot_values(events, self.num_slots - 1)
            p0 = psi[np.arange(shots), s]
            p1 = psi[np.arange(shots), s + 2 ** n_last]
            return np.log(p0) + log_scale, np.log(p1) + log_scale


def _convolve(kernel: np.ndarray, vector: int, probability: float) -> None:
    """In-place Z_2 convolution with {0: 1 - p, vector: p}."""
    if probability <= 0.0 or vector == 0:
        return
    index = np.arange(kernel.size) ^ vector
    kernel[:] = (1.0 - probability) * kernel + probability * kernel[index]


# ---------------------------------------------------------------------------
# Single-fault enumeration (first-order analysis)
# ---------------------------------------------------------------------------

_SINGLE = ((1,), (2,), (3,))
_PAIRS = tuple((a, b) for a in range(4) for b in range(4) if a or b)


@dataclass
class SingleFault:
    location: int
    paulis: dict[int, int]
    probability: float
    detectors: tuple[int, ...]
    flips: bool


def enumerate_single_faults(probe: InstrumentedLRB, signatures: Signatures,
                            num_detectors: Optional[int] = None,
                            heralded_weight: bool = False
                            ) -> list[SingleFault]:
    """Every single Pauli fault with its first-order probability.

    ``DEPOLARIZE1(q)``: X, Y, Z with ``q/3`` each; ``DEPOLARIZE2(q)``: the 15
    non-identity Paulis with ``q/15``; ``X_ERROR(q)``: X with ``q``. If
    ``heralded_weight`` is set, the probability is multiplied by
    ``1 - herald_fraction`` (the unheralded first-order part).
    """
    n = (probe.num_main_detectors + len(probe.flag_detectors)
         if num_detectors is None else num_detectors)
    faults = []
    for li, loc in enumerate(probe.locations):
        if loc.channel == "NONE" or loc.probability <= 0.0:
            continue
        if (li, loc.qubits[0], "X") not in signatures.lane_of:
            continue
        weight = (1.0 - loc.herald_fraction) if heralded_weight else 1.0
        if loc.channel == "DEPOLARIZE1":
            cases = [({loc.qubits[0]: c[0]}, loc.probability / 3.0)
                     for c in _SINGLE]
        elif loc.channel == "X_ERROR":
            cases = [({loc.qubits[0]: 1}, loc.probability)]
        elif loc.channel == "DEPOLARIZE2":
            a, b = loc.qubits
            cases = [({a: pa, b: pb}, loc.probability / 15.0)
                     for pa, pb in _PAIRS]
        else:
            raise ValueError(loc.channel)
        for paulis, prob in cases:
            paulis = {q: v for q, v in paulis.items() if v}
            dets, obs = signatures.pauli(li, paulis)
            faults.append(SingleFault(li, paulis, prob * weight,
                                      tuple(np.flatnonzero(dets[:n]).tolist()),
                                      bool(obs)))
    return faults


def first_order_table(faults: Sequence[SingleFault],
                      decisions: Optional[dict[tuple[int, ...], bool]] = None
                      ) -> dict:
    """Group faults by syndrome; minority mass is the first-order MLD error.

    Returns a dict with ``groups`` (syndrome -> [mass L=0, mass L=1]),
    ``miscorrected`` (faults the decision gets wrong; if ``decisions`` is
    None the majority class is assumed), ``undetected_logical`` (mass of
    faults with no detection event that flip L) and ``first_order_error``.
    """
    groups: dict[tuple[int, ...], list[float]] = {}
    for f in faults:
        groups.setdefault(f.detectors, [0.0, 0.0])[int(f.flips)] += f.probability
    if decisions is None:
        decisions = {key: (w[1] > w[0]) for key, w in groups.items()}
    miscorrected = [f for f in faults if f.detectors
                    and bool(decisions[f.detectors]) != f.flips]
    undetected = [f for f in faults if not f.detectors and f.flips]
    return {
        "groups": groups,
        "decisions": decisions,
        "miscorrected": miscorrected,
        "undetected_logical": sum(f.probability for f in undetected),
        "first_order_error": (sum(f.probability for f in miscorrected)
                              + sum(f.probability for f in undetected)),
        "minority_mass": sum(min(w) for key, w in groups.items() if key)
                         + groups.get((), [0.0, 0.0])[1],
    }


def decide_syndromes(decoder, keys: Sequence[tuple[int, ...]],
                     num_detectors: int) -> dict[tuple[int, ...], bool]:
    """MLD decision for each detector-index syndrome (one shot per key)."""
    keys = list(keys)
    events = np.zeros((len(keys), num_detectors), dtype=bool)
    for i, key in enumerate(keys):
        events[i, list(key)] = True
    llr = decoder.log_likelihood_ratio(events)
    return {key: bool(v < 0) for key, v in zip(keys, llr)}


def describe_fault(probe: InstrumentedLRB, fault: SingleFault) -> str:
    """Short human-readable label of a fault."""
    loc = probe.locations[fault.location]
    names = {1: "X", 2: "Y", 3: "Z"}
    pauli = "".join(f"{names[v]}{q}" for q, v in sorted(fault.paulis.items()))
    extra = ""
    if loc.check[0] >= 0:
        extra = f" {C.STABILIZER_NAMES[loc.check[0]]}@{loc.check[1]}"
    return f"{loc.kind}{extra} slot{loc.slot} {pauli}"


def is_hook(probe: InstrumentedLRB, fault: SingleFault) -> bool:
    """Check-round CNOT fault whose ancilla part spreads to two data qubits.

    Z checks (CNOT data->ancilla) spread Z/Y ancilla faults as Z onto the
    later data controls; X checks (CNOT ancilla->data) spread X/Y ancilla
    faults as X onto the later data targets. A hook is a spreading ancilla
    component right after a gate that precedes at least two data CNOTs.
    """
    loc = probe.locations[fault.location]
    if loc.kind not in ("cx_check", "cx_flag") or loc.check[0] < 0:
        return False
    s, done = loc.check
    ancilla = C.STABILIZER_ANCILLA[s]
    pauli = fault.paulis.get(ancilla, 0)
    spreading = (pauli in (2, 3)) if s in C.Z_STABILIZERS else (pauli in (1, 2))
    return spreading and 3 - done >= 2


# ---------------------------------------------------------------------------
# Erasure experiment (identical shots for every method)
# ---------------------------------------------------------------------------

ERASURE_METHODS = (
    "lrbd_unif_1", "lrbd_unif_1_final", "lrbd_const_1",
    "lrbd_unif_1_final_noherald", "hybrid", "mld", "mld_herald",
)


@dataclass
class ErasureCircuits:
    """Everything needed to sample and decode one (sequence, depth) circuit."""

    sample: InstrumentedLRB
    decode: InstrumentedLRB
    probe: InstrumentedLRB
    signatures: Signatures
    mld: M.TransferMatrixMLD
    mld_herald: HeraldedMLD
    explain: HeraldedMLD


def prepare_erasure_circuits(sequence: Sequence[str], depth: int, p: float,
                             erasure: ErasureSpec,
                             noise: Q.SI1000 = Q.SI1000()) -> ErasureCircuits:
    sample = build_instrumented_lrb(sequence, depth, p, erasure=erasure,
                                    noise=noise, mode="sample")
    decode = build_instrumented_lrb(sequence, depth, p, erasure=erasure,
                                    noise=noise, mode="decode")
    probe = build_instrumented_lrb(sequence, depth, p, erasure=erasure,
                                   noise=noise, mode="probe")
    herald_locs = sample.herald_locations
    signatures = probe_signatures(probe, herald_locs)
    comps = herald_components(sample, signatures)
    slots = decode.decoder_slots()
    lead = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                               noise=noise)
    mld = M.TransferMatrixMLD(lead.circuit.detector_error_model(),
                              M.lrb_detector_slots(lead))
    mld_herald = HeraldedMLD(decode.circuit.detector_error_model(), slots,
                             comps)
    explain = HeraldedMLD(None, slots, comps)
    return ErasureCircuits(sample, decode, probe, signatures, mld,
                           mld_herald, explain)


def _histogram_gap(hist: np.ndarray, gap: np.ndarray, wrong: np.ndarray,
                   edges: np.ndarray) -> None:
    gap = np.where(np.isnan(gap), 0.0, gap)
    bins = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0,
                   len(edges) - 2)
    np.add.at(hist, (bins, wrong.astype(np.int64)), 1)


def run_erasure_point(p: float, erasure: ErasureSpec,
                      sequences: Sequence[Sequence[str]],
                      depths: Sequence[int], shots: int, *,
                      seed: int = 1, batch_size: int = 50_000,
                      noise: Q.SI1000 = Q.SI1000(),
                      gap_edges: Optional[np.ndarray] = None) -> dict:
    """Score every erasure-study method on the same shots.

    Methods (all on identical shots): ``lrbd_unif_1``, ``lrbd_const_1`` and
    ``lrbd_unif_1_final`` ignore heralds (LRB-D as in the repo);
    ``lrbd_unif_1_final_noherald`` also rejects any shot with a herald;
    ``hybrid`` accepts a shot iff its full syndrome is explained by the
    heralded erasures alone with an unambiguous logical class (the
    "correct heralded, reject unexplained" policy) and applies that class;
    ``mld`` is the lead's exact MLD (heralds ignored); ``mld_herald`` is the
    exact herald-conditioned MLD. Gap histograms (|LLR|, correct vs wrong)
    are kept for both MLDs.
    """
    from . import experiment as E
    edges = E.GAP_EDGES if gap_edges is None else gap_edges
    depths = list(depths)
    n_seq = len(sequences)
    tables = {name: B.CountTable.zeros(n_seq, len(depths))
              for name in ERASURE_METHODS}
    hists = {name: np.zeros((n_seq, len(depths), len(edges) - 1, 2),
                            dtype=np.int64) for name in ("mld", "mld_herald")}
    herald_stats = np.zeros((n_seq, len(depths), 3))  # shots, any, total
    ambiguous = np.zeros((n_seq, len(depths)), dtype=np.int64)
    for si, sequence in enumerate(sequences):
        for di, depth in enumerate(depths):
            ec = prepare_erasure_circuits(sequence, depth, p, erasure, noise)
            lc = ec.sample.lrb
            n_round = 4 * lc.num_rounds
            sampler = ec.sample.circuit.compile_detector_sampler(
                seed=seed + 1000 * si + di)
            done = 0
            while done < shots:
                n = min(batch_size, shots - done)
                events, obs = sampler.sample(n, separate_observables=True)
                flip = obs[:, 0]
                dets, heralds = ec.sample.split(events)
                raw = lc.raw_syndromes(dets[:, :n_round])
                final_clean = ~dets[:, n_round:].any(axis=1)
                any_herald = heralds.any(axis=1)
                herald_stats[si, di] += (n, any_herald.sum(), heralds.sum())
                acc_u1 = B.accept_mask(raw, lc.rounds,
                                       B.used_rounds("unif", 1, depth))
                acc_c1 = B.accept_mask(raw, lc.rounds,
                                       B.used_rounds("const", 1, depth))
                for name, acc in (
                        ("lrbd_unif_1", acc_u1),
                        ("lrbd_const_1", acc_c1),
                        ("lrbd_unif_1_final", acc_u1 & final_clean),
                        ("lrbd_unif_1_final_noherald",
                         acc_u1 & final_clean & ~any_herald)):
                    tables[name].add(si, di, n, int(acc.sum()),
                                     int((acc & flip).sum()))
                llr = ec.mld.log_likelihood_ratio(dets)
                wrong = (llr < 0) != flip
                tables["mld"].add(si, di, n, n, int(wrong.sum()))
                _histogram_gap(hists["mld"][si, di], np.abs(llr), wrong, edges)
                llr_h = ec.mld_herald.log_likelihood_ratio(dets, heralds)
                wrong_h = (llr_h < 0) != flip
                tables["mld_herald"].add(si, di, n, n, int(wrong_h.sum()))
                _histogram_gap(hists["mld_herald"][si, di], np.abs(llr_h),
                               wrong_h, edges)
                e0, e1 = ec.explain.log_joint(dets, heralds)
                ok0, ok1 = np.isfinite(e0), np.isfinite(e1)
                accept = ok0 ^ ok1
                ambiguous[si, di] += int((ok0 & ok1).sum())
                tables["hybrid"].add(si, di, n, int(accept.sum()),
                                     int((accept & (ok1 != flip)).sum()))
                done += n
    return {"p": p, "erasure": erasure, "depths": depths, "tables": tables,
            "hists": hists, "herald_stats": herald_stats,
            "ambiguous": ambiguous, "gap_edges": edges}
