"""Literature-exact SI1000 variant with ASAP moment scheduling (qubits).

This mirrors the circuit-level noise model adopted in the main checkout on
2026-10-28: ``CircuitLevelNoiseRates`` and ``apply_circuit_level_noise`` in
``src/lrb/circuit_generator.py``, after Gidney, Newman, Fowler and Broughton,
"A Fault-Tolerant Honeycomb Memory" (2021).

* Every logical Clifford, and every check round, is one barrier-separated
  segment packed into as-soon-as-possible moments. A moment holds either
  unitary gates or collapse operations (measurements, resets). Measurements
  whose wire is not used again in the segment are deferred to one final
  measurement moment, so the four ancillas of a round read out together.
* Noise per moment: DEPOLARIZE1(0.1p) after one-qubit gates (and on ``I``);
  DEPOLARIZE2(p) after two-qubit gates; X_ERROR(2p) after resets; X_ERROR(5p)
  before and DEPOLARIZE1(p) after measurements. Every wire the moment leaves
  untouched gets DEPOLARIZE1(0.1p) in gate moments and DEPOLARIZE1(2p) in
  collapse moments.
* Ancillas are reset at the start of every check round, not after it.
* The encoding and the terminal readout are ideal, as before.

Detectors, the observable and the ``LRBCircuit`` bookkeeping are the same as
in ``circuits.build_lrb_circuit``, so ``mld``, ``exact`` and ``baseline`` work
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import stim

from . import circuits as Q
from . import code as C


@dataclass(frozen=True)
class LiteratureSI1000:
    """Multiples of p (the two-qubit gate error) for each noise location."""

    one_qubit: float = 0.1
    two_qubit: float = 1.0
    reset_flip: float = 2.0
    meas_flip: float = 5.0
    post_measurement: float = 1.0
    idle: float = 0.1
    collapse_idle: float = 2.0

    def scaled(self, **factors: float) -> "LiteratureSI1000":
        values = {name: getattr(self, name)
                  for name in self.__dataclass_fields__}
        for name, factor in factors.items():
            values[name] *= factor
        return LiteratureSI1000(**values)


def schedule_moments(operations: Sequence[C.Operation]
                     ) -> list[tuple[bool, list[C.Operation]]]:
    """ASAP moments, as ``CircuitLevelNoise.schedule_moments`` does."""
    last_use: dict[int, int] = {}
    for index, (_, targets) in enumerate(operations):
        for wire in targets:
            last_use[wire] = index
    deferred = [i for i, (gate, targets) in enumerate(operations)
                if gate == "M" and last_use[targets[0]] == i]
    deferred_set = set(deferred)
    moments: list[tuple[bool, list[C.Operation]]] = []
    ready: dict[int, int] = {}
    for index, operation in enumerate(operations):
        if index in deferred_set:
            continue
        gate, targets = operation
        collapse = gate in ("M", "R")
        slot = max(ready.get(wire, 0) for wire in targets)
        while slot < len(moments) and moments[slot][0] != collapse:
            slot += 1
        if slot == len(moments):
            moments.append((collapse, []))
        moments[slot][1].append(operation)
        for wire in targets:
            ready[wire] = slot + 1
    if deferred:
        moments.append((True, [operations[i] for i in deferred]))
    return moments


def emit_segment(circuit: stim.Circuit, operations: Sequence[C.Operation],
                 p: float, noise: Optional[LiteratureSI1000],
                 num_qubits: int) -> dict[int, int]:
    """Append one scheduled, noisy segment; return ``{wire: record}`` of Ms."""
    noisy = noise is not None and p > 0.0
    records: dict[int, int] = {}
    for collapse, moment in schedule_moments(operations):
        busy: set[int] = set()
        for gate, targets in moment:
            busy.update(targets)
            if gate == "M":
                q = targets[0]
                if noisy:
                    Q._append(circuit, "X_ERROR", [q], noise.meas_flip * p)
                records[q] = circuit.num_measurements
                circuit.append("M", [q])
                if noisy:
                    Q._append(circuit, "DEPOLARIZE1", [q],
                              noise.post_measurement * p)
            elif gate == "R":
                circuit.append("R", list(targets))
                if noisy:
                    Q._append(circuit, "X_ERROR", list(targets),
                              noise.reset_flip * p)
            elif gate == "I":
                circuit.append("I", list(targets))
                if noisy:
                    Q._append(circuit, "DEPOLARIZE1", list(targets),
                              noise.idle * p)
            elif len(targets) == 1:
                circuit.append(gate, list(targets))
                if noisy:
                    Q._append(circuit, "DEPOLARIZE1", list(targets),
                              noise.one_qubit * p)
            else:
                circuit.append(gate, list(targets))
                if noisy:
                    Q._append(circuit, "DEPOLARIZE2", list(targets),
                              noise.two_qubit * p)
        if noisy:
            idle = [w for w in range(num_qubits) if w not in busy]
            scale = noise.collapse_idle if collapse else noise.idle
            Q._append(circuit, "DEPOLARIZE1", idle, scale * p)
    return records


def build_lrb_circuit_lit(cliffords: Sequence[str], depth: int, p: float,
                          *, terminal: str = "data5",
                          forward_check_rounds: Optional[Iterable[int]] = None,
                          terminal_check: bool = True,
                          noise: Optional[LiteratureSI1000] = LiteratureSI1000()
                          ) -> Q.LRBCircuit:
    """Literature-SI1000 counterpart of ``circuits.build_lrb_circuit``."""
    if terminal not in ("data2", "data5"):
        raise ValueError("terminal must be 'data2' or 'data5'.")
    forward = list(cliffords[:depth])
    forward_set = (set(range(depth)) if forward_check_rounds is None
                   else {int(r) for r in forward_check_rounds})
    circuit = stim.Circuit()
    Q.emit_operations(circuit, C.encoding_operations(), p, None,
                      C.NUM_QUBITS)
    check_ops = (C.reset_operations() + C.z_stabilizer_check_operations()
                 + C.x_stabilizer_check_operations())
    rounds: list[int] = []
    round_records: dict[int, list[int]] = {}
    segments: list[list[C.Operation]] = []
    pending: list[C.Operation] = []

    def run_check_round(label: int) -> None:
        records = emit_segment(circuit, check_ops, p, noise, C.NUM_QUBITS)
        rounds.append(label)
        round_records[label] = [records[a] for a in C.STABILIZER_ANCILLA]
        segments.append(list(pending))
        pending.clear()

    for j, descriptor in enumerate(forward):
        ops = C.logical_clifford_operations(descriptor)
        emit_segment(circuit, ops, p, noise, C.NUM_QUBITS)
        pending.extend(ops)
        if j in forward_set:
            run_check_round(j)
    for descriptor in reversed(forward):
        ops = C.logical_clifford_operations(descriptor, inverse=True)
        emit_segment(circuit, ops, p, noise, C.NUM_QUBITS)
        pending.extend(ops)
    if terminal_check:
        run_check_round(depth)
    segments.append(list(pending))
    pending.clear()

    readout_qubits = (0, 3) if terminal == "data2" else C.DATA_QUBITS
    readout_ops = ([("H", (q,)) for q in readout_qubits]
                   + [("M", (q,)) for q in readout_qubits])
    readout_records = Q.emit_operations(circuit, readout_ops, p, None,
                                        C.NUM_QUBITS)
    data_record = dict(zip(readout_qubits, readout_records))

    detector_keys: list[tuple[int, int]] = []
    back: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous: Optional[list[int]] = None
    for position, label in enumerate(rounds):
        records = round_records[label]
        subsets = (tuple(() for _ in range(4)) if previous is None else
                   Q._back_subsets(segments[position], range(4)))
        back[label] = subsets
        for s in range(4):
            targets = [records[s]] + Q._detector_from_back(previous,
                                                           subsets[s])
            circuit.append("DETECTOR", [stim.target_rec(r - circuit.num_measurements)
                                        for r in targets], [s, label])
            detector_keys.append((label, s))
        previous = records
    if terminal == "data5":
        subsets = Q._back_subsets(segments[-1], C.X_STABILIZERS)
        for s, subset in zip(C.X_STABILIZERS, subsets):
            support = [q for q, ch in enumerate(C.STABILIZER_PAULIS[s])
                       if ch != "_"]
            targets = ([data_record[q] for q in support]
                       + Q._detector_from_back(previous, subset))
            circuit.append("DETECTOR", [stim.target_rec(r - circuit.num_measurements)
                                        for r in targets], [s, depth + 1])
            detector_keys.append((depth + 1, s))
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(data_record[q] - circuit.num_measurements)
                    for q in (0, 3)], 0)
    return Q.LRBCircuit(circuit=circuit, depth=depth, rounds=rounds,
                        detector_keys=detector_keys, back=back,
                        terminal=terminal, segments=segments)


def build_lrb_const0_circuit_lit(cliffords: Sequence[str], depth: int,
                                 p: float, *,
                                 noise: Optional[LiteratureSI1000] = (
                                     LiteratureSI1000())) -> Q.LRBCircuit:
    """``const=0`` under the literature model: no checks, ideal X readout."""
    return build_lrb_circuit_lit(cliffords, depth, p, terminal="data5",
                                 forward_check_rounds=(),
                                 terminal_check=False, noise=noise)


def build_physical_rb_circuit_lit(cliffords: Sequence[str], depth: int,
                                  p_rb: float,
                                  noise: Optional[LiteratureSI1000] = (
                                      LiteratureSI1000())) -> stim.Circuit:
    """Physical RB: each Clifford one segment, ideal final measurement."""
    circuit = stim.Circuit()
    forward = list(cliffords[:depth])
    for descriptor in forward:
        ops = []
        for token in C.clifford_gate_sequence(descriptor):
            ops.extend(C.physical_gate_operations(token))
        emit_segment(circuit, ops, p_rb, noise, 1)
    for descriptor in reversed(forward):
        ops = []
        for token in C.clifford_inverse_sequence(descriptor):
            ops.extend(C.physical_gate_operations(token))
        emit_segment(circuit, ops, p_rb, noise, 1)
    circuit.append("M", [0])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit
