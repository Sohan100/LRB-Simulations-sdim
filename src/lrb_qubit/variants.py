"""Circuit-design variants and noise ablations for qubit ``[[5,1,2]]`` LRB.

Work package H. Every builder here returns an ``LRBCircuit`` (subclass
``VariantCircuit``) with the bookkeeping conventions of
``circuits.build_lrb_circuit``: round labels ``0..m-1`` for forward rounds,
``INVERSE_ROUND_OFFSET + j`` for inverse-half rounds, ``m`` for the terminal
round and ``m + 1`` for the final data detectors; four difference detectors
per round in stabilizer order (S_Z1, S_Z2, S_X1, S_X2); observable 0 = the
X_L readout flip ``x0 + x3``. ``mld.TransferMatrixMLD(dem,
mld.lrb_detector_slots(lc))`` and ``lc.raw_syndromes`` therefore work
unchanged on every variant.

Variants
--------
``schedule``
    ``"serial"`` is the repo's check round (each ancilla measured in its own
    layer: four measurement layers, then one reset layer). ``"zx"`` measures
    the two Z ancillas together after both Z checks and the two X ancillas
    together after both X checks (two layers). ``"parallel"`` runs all twelve
    CNOTs (same order as the repo) and then measures all four ancillas in one
    layer. Under the repo's SI1000 model only measurement and reset layers
    carry idle noise, so every data qubit sees 5, 3 or 2 ``DEPOLARIZE1(2p)``
    layers per round (serial, zx, parallel).
``reset``
    ``False`` removes the ancilla reset after forward and inverse rounds
    (measure-without-reset). Each round's raw stabilizer value is then the
    XOR of the ancilla's current and previous measurement, and the detectors
    are built from those XORs. With the repo's convention that measurement
    errors are ``X_ERROR`` *before* ``M`` (a physical flip), a measurement
    error still flips one raw syndrome only.
``forward_check_rounds`` / ``inverse_check_every``
    As in ``build_lrb_circuit``: physically execute only some forward check
    rounds, and optionally add check rounds in the inverse half.
``noise``
    A ``NoiseConfig`` with separate SI1000 scale factors for the check rounds
    and for the logical Clifford gates, plus multipliers that split the
    measurement-layer idle noise between data qubits and idle ancillas.

The module also holds the analysis helpers used by
``scripts/qubit512/schedule_variants.py``: a vectorised copy of the repo's
fixed-offset LRB fit (``lrb_plotting._fit_decay_parameter_fixed_a``) and a
sequence bootstrap.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import stim

from . import baseline as B
from . import circuits as Q
from . import code as C
from . import mld as M

SCHEDULES = ("serial", "zx", "parallel")
_SCHEDULE_GROUPS = {
    "serial": ((0,), (1,), (2,), (3,)),
    "zx": ((0, 1), (2, 3)),
    "parallel": ((0, 1, 2, 3),),
}


# ---------------------------------------------------------------------------
# Check-round operation lists
# ---------------------------------------------------------------------------

def ancilla_blocks() -> list[tuple[list[C.Operation], C.Operation]]:
    """The repo's serial check round split into ``(gates, measurement)`` blocks.

    One block per ancilla 5, 6, 7, 8 (stabilizers S_Z1, S_Z2, S_X1, S_X2), taken
    from ``code.z_stabilizer_check_operations`` and
    ``code.x_stabilizer_check_operations`` so the CNOT order stays the repo's.
    """
    blocks: list[tuple[list[C.Operation], C.Operation]] = []
    gates: list[C.Operation] = []
    for op in (C.z_stabilizer_check_operations()
               + C.x_stabilizer_check_operations()):
        if op[0] == "M":
            blocks.append((gates, op))
            gates = []
        else:
            gates.append(op)
    if gates or len(blocks) != 4:
        raise RuntimeError("Unexpected check-round structure in lrb_qubit.code.")
    return blocks


def check_round_operations(schedule: str = "serial",
                           reset: bool = True) -> list[C.Operation]:
    """One check round under ``schedule`` (optionally followed by the reset).

    Measurements of a group are consecutive, so ``circuits.emit_operations``
    treats them as one measurement layer. ``schedule="serial"`` with
    ``reset=True`` equals ``code.check_round_operations()``.
    """
    if schedule not in _SCHEDULE_GROUPS:
        raise ValueError(f"Unknown schedule {schedule!r}; use {SCHEDULES}.")
    blocks = ancilla_blocks()
    ops: list[C.Operation] = []
    for group in _SCHEDULE_GROUPS[schedule]:
        for index in group:
            ops.extend(blocks[index][0])
        for index in group:
            ops.append(blocks[index][1])
    if reset:
        ops.extend(C.reset_operations())
    return ops


def measurement_layers_per_round(schedule: str) -> int:
    """Number of measurement layers in one check round."""
    return len(_SCHEDULE_GROUPS[schedule])


def data_idle_layers_per_round(schedule: str, reset: bool = True) -> int:
    """``DEPOLARIZE1(meas_idle p)`` / ``(reset_idle p)`` layers per data qubit."""
    return measurement_layers_per_round(schedule) + (1 if reset else 0)


# ---------------------------------------------------------------------------
# Noise configuration and emission
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseConfig:
    """SI1000 noise split by circuit part.

    Attributes:
        check: Scale factors for check rounds (CNOTs, ancilla H, measurement
            and reset layers). ``None`` makes check rounds noiseless.
        gate: Scale factors for the logical Clifford gates (only
            ``one_qubit`` and ``two_qubit`` matter). ``None`` = noiseless.
        meas_idle_data: Multiplier on ``check.meas_idle`` for idle *data*
            qubits during measurement layers.
        meas_idle_ancilla: Multiplier on ``check.meas_idle`` for idle
            *ancillas* during measurement layers.

    The terminal readout block uses ``check`` when ``terminal_noise`` is set.
    """

    check: Optional[Q.SI1000] = Q.SI1000()
    gate: Optional[Q.SI1000] = Q.SI1000()
    meas_idle_data: float = 1.0
    meas_idle_ancilla: float = 1.0

    @property
    def splits_idle(self) -> bool:
        return self.meas_idle_data != 1.0 or self.meas_idle_ancilla != 1.0


NoiseLike = Union[NoiseConfig, Q.SI1000, None]


def as_noise_config(noise: NoiseLike) -> NoiseConfig:
    """Normalise ``SI1000`` / ``None`` / ``NoiseConfig`` to ``NoiseConfig``."""
    if isinstance(noise, NoiseConfig):
        return noise
    return NoiseConfig(check=noise, gate=noise)


def _append(circuit: stim.Circuit, name: str, targets: Sequence[int],
            arg: float) -> None:
    if arg > 0.0 and len(targets) > 0:
        circuit.append(name, list(targets), arg)


def emit_with_noise(circuit: stim.Circuit, operations: Sequence[C.Operation],
                    p: float, noise: Optional[Q.SI1000],
                    noisy_measurements: bool = True,
                    meas_idle_data: float = 1.0,
                    meas_idle_ancilla: float = 1.0) -> list[int]:
    """``circuits.emit_operations`` with an optional data/ancilla idle split.

    Without a split this *is* ``circuits.emit_operations`` (same output).
    With a split, measurement layers put ``DEPOLARIZE1(meas_idle p x
    meas_idle_data)`` on idle data qubits and ``DEPOLARIZE1(meas_idle p x
    meas_idle_ancilla)`` on idle ancillas; everything else is delegated to
    ``circuits.emit_operations``.
    """
    if (noise is None or p <= 0.0 or not noisy_measurements
            or (meas_idle_data == 1.0 and meas_idle_ancilla == 1.0)):
        return Q.emit_operations(circuit, operations, p, noise, C.NUM_QUBITS,
                                 noisy_measurements)
    records: list[int] = []
    index = 0
    while index < len(operations):
        end = index
        if operations[index][0] == "M":
            while end < len(operations) and operations[end][0] == "M":
                end += 1
            layer = [operations[i][1][0] for i in range(index, end)]
            idle = [q for q in range(C.NUM_QUBITS) if q not in layer]
            _append(circuit, "DEPOLARIZE1",
                    [q for q in idle if q < C.NUM_DATA],
                    noise.meas_idle * p * meas_idle_data)
            _append(circuit, "DEPOLARIZE1",
                    [q for q in idle if q >= C.NUM_DATA],
                    noise.meas_idle * p * meas_idle_ancilla)
            for q in layer:
                _append(circuit, "DEPOLARIZE1", [q], noise.meas_depol * p)
                _append(circuit, "X_ERROR", [q], noise.meas_flip * p)
                records.append(circuit.num_measurements)
                circuit.append("M", [q])
        else:
            while end < len(operations) and operations[end][0] != "M":
                end += 1
            records.extend(Q.emit_operations(
                circuit, operations[index:end], p, noise, C.NUM_QUBITS,
                noisy_measurements))
        index = end
    return records


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

@dataclass
class VariantCircuit(Q.LRBCircuit):
    """``LRBCircuit`` plus the variant settings that produced it."""

    schedule: str = "serial"
    reset: bool = True
    noise_config: Optional[NoiseConfig] = None
    raw_records: dict[int, tuple[tuple[int, ...], ...]] = field(
        default_factory=dict)


def _back_subsets(segment: Sequence[C.Operation],
                  stabilizers: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    inverse = C.data_tableau(segment).inverse()
    return tuple(C.decompose_in_stabilizers(inverse(C.stabilizer_pauli(s)))
                 for s in stabilizers)


def _xor_records(*groups: Iterable[int]) -> list[int]:
    """Symmetric difference of record-index groups, in first-appearance order."""
    parity: dict[int, bool] = {}
    for group in groups:
        for record in group:
            parity[record] = not parity.get(record, False)
    return [record for record, odd in parity.items() if odd]


def build_variant_circuit(cliffords: Sequence[str],
                          depth: int,
                          p: float,
                          *,
                          schedule: str = "serial",
                          reset: bool = True,
                          terminal: str = "data5",
                          terminal_noise: bool = False,
                          forward_check_rounds: Optional[Iterable[int]] = None,
                          terminal_check: bool = True,
                          inverse_check_every: int = 0,
                          noise: NoiseLike = NoiseConfig()) -> VariantCircuit:
    """Build one LRB circuit variant (see the module docstring).

    With ``schedule="serial"``, ``reset=True`` and a plain ``SI1000`` noise
    this reproduces ``circuits.build_lrb_circuit`` instruction for
    instruction (tested).
    """
    if terminal not in ("data2", "data5"):
        raise ValueError("terminal must be 'data2' or 'data5'.")
    if schedule not in _SCHEDULE_GROUPS:
        raise ValueError(f"Unknown schedule {schedule!r}.")
    if len(cliffords) < depth:
        raise ValueError("Clifford sequence is shorter than the depth.")
    config = as_noise_config(noise)
    forward = list(cliffords[:depth])
    forward_set = (set(range(depth)) if forward_check_rounds is None
                   else {int(r) for r in forward_check_rounds})
    if any(r < 0 or r >= depth for r in forward_set):
        raise ValueError("forward_check_rounds must lie in [0, depth).")

    circuit = stim.Circuit()
    Q.emit_operations(circuit, C.encoding_operations(), p, None, C.NUM_QUBITS)

    rounds: list[int] = []
    round_records: dict[int, list[int]] = {}
    segments: list[list[C.Operation]] = []
    pending: list[C.Operation] = []

    def emit_gates(ops: Sequence[C.Operation]) -> None:
        Q.emit_operations(circuit, ops, p, config.gate, C.NUM_QUBITS,
                          noisy_measurements=False)

    def run_check_round(label: int, with_reset: bool) -> None:
        ops = check_round_operations(schedule, reset=with_reset and reset)
        records = emit_with_noise(circuit, ops, p, config.check,
                                  noisy_measurements=True,
                                  meas_idle_data=config.meas_idle_data,
                                  meas_idle_ancilla=config.meas_idle_ancilla)
        rounds.append(label)
        round_records[label] = records
        segments.append(list(pending))
        pending.clear()

    for j, descriptor in enumerate(forward):
        ops = C.logical_clifford_operations(descriptor)
        emit_gates(ops)
        pending.extend(ops)
        if j in forward_set:
            run_check_round(j, True)

    inverse_label = Q.INVERSE_ROUND_OFFSET
    for j, descriptor in enumerate(reversed(forward)):
        ops = C.logical_clifford_operations(descriptor, inverse=True)
        emit_gates(ops)
        pending.extend(ops)
        if (inverse_check_every > 0 and (j + 1) % inverse_check_every == 0
                and j + 1 < depth):
            run_check_round(inverse_label, True)
            inverse_label += 1

    if terminal_check:
        # As in the repo: no reset after the terminal round.
        run_check_round(depth, False)
    segments.append(list(pending))
    pending.clear()

    readout_qubits = (0, 3) if terminal == "data2" else C.DATA_QUBITS
    readout_ops = ([("H", (q,)) for q in readout_qubits]
                   + [("M", (q,)) for q in readout_qubits])
    readout_records = Q.emit_operations(
        circuit, readout_ops, p, config.check if terminal_noise else None,
        C.NUM_QUBITS, noisy_measurements=True)
    data_record = dict(zip(readout_qubits, readout_records))

    # Raw stabilizer values as record groups. With reset (or for the first
    # round) the ancilla starts in |0>, so the raw value is one record; without
    # reset it is the XOR with the same ancilla's previous record.
    raw_records: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous_label: Optional[int] = None
    for label in rounds:
        records = round_records[label]
        if reset or previous_label is None:
            groups = tuple((records[s],) for s in range(4))
        else:
            prev = round_records[previous_label]
            groups = tuple((records[s], prev[s]) for s in range(4))
        raw_records[label] = groups
        previous_label = label

    detector_keys: list[tuple[int, int]] = []
    back: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous: Optional[tuple[tuple[int, ...], ...]] = None
    for position, label in enumerate(rounds):
        groups = raw_records[label]
        if previous is None:
            subsets = tuple(() for _ in range(4))
        else:
            subsets = _back_subsets(segments[position], range(4))
        back[label] = subsets
        for s in range(4):
            parts = [groups[s]]
            if previous is not None:
                parts += [previous[t] for t in subsets[s]]
            targets = _xor_records(*parts)
            circuit.append(
                "DETECTOR",
                [stim.target_rec(r - circuit.num_measurements) for r in targets],
                [s, label])
            detector_keys.append((label, s))
        previous = groups

    if terminal == "data5":
        subsets = _back_subsets(segments[-1], C.X_STABILIZERS)
        for s, subset in zip(C.X_STABILIZERS, subsets):
            support = [q for q, ch in enumerate(C.STABILIZER_PAULIS[s])
                       if ch != "_"]
            parts = [[data_record[q] for q in support]]
            if previous is not None:
                parts += [previous[t] for t in subset]
            targets = _xor_records(*parts)
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

    return VariantCircuit(circuit=circuit, depth=depth, rounds=rounds,
                          detector_keys=detector_keys, back=back,
                          terminal=terminal, segments=segments,
                          schedule=schedule, reset=reset,
                          noise_config=config, raw_records=raw_records)


def policy_forward_rounds(policy: str, k: int, depth: int) -> list[int]:
    """Forward rounds a ``const=k`` / ``unif=k`` policy uses (terminal removed).

    ``policy="terminal"`` returns no forward round (terminal check only).
    """
    if policy == "terminal":
        return []
    return [r for r in B.used_rounds(policy, k, depth) if r != depth]


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------

_BASE = Q.SI1000()
_ZERO = Q.SI1000(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

ABLATIONS: dict[str, NoiseConfig] = {
    "full": NoiseConfig(),
    # idle during measurement layers (all idle wires / data only / ancillas only)
    "no_meas_idle": NoiseConfig(check=_BASE.scaled(meas_idle=0.0)),
    "no_meas_idle_data": NoiseConfig(meas_idle_data=0.0),
    "no_meas_idle_anc": NoiseConfig(meas_idle_ancilla=0.0),
    # measurement flips X_ERROR(5p) and the measured-qubit DEPOLARIZE1(p)
    "no_meas_flip": NoiseConfig(check=_BASE.scaled(meas_flip=0.0)),
    "no_meas_depol": NoiseConfig(check=_BASE.scaled(meas_depol=0.0)),
    # check-circuit gates: CNOT DEPOLARIZE2(p), ancilla H DEPOLARIZE1(0.1p)
    "no_cnot": NoiseConfig(check=_BASE.scaled(two_qubit=0.0)),
    "no_check_1q": NoiseConfig(check=_BASE.scaled(one_qubit=0.0)),
    # reset layer: data idle DEPOLARIZE1(2p) and reset flips X_ERROR(2p)
    "no_reset_idle": NoiseConfig(check=_BASE.scaled(reset_idle=0.0)),
    "no_reset_flip": NoiseConfig(check=_BASE.scaled(reset_flip=0.0)),
    # logical Clifford gates (one-qubit 0.1p and SWAP/CZ p)
    "no_gate": NoiseConfig(gate=_BASE.scaled(one_qubit=0.0, two_qubit=0.0)),
    # bounds: perfect check rounds / only check noise
    "perfect_checks": NoiseConfig(check=_ZERO),
}

# Ablations whose removed parts are disjoint and together cover every noise
# channel (used for the additivity check).
DISJOINT_ABLATIONS = ("no_meas_idle", "no_meas_flip", "no_meas_depol",
                      "no_cnot", "no_check_1q", "no_reset_idle",
                      "no_reset_flip", "no_gate")


def ablation_noise(name: str) -> NoiseConfig:
    """Noise for an ``ABLATIONS`` key, or ``"check_scale_<C>"``.

    ``check_scale_<C>`` multiplies every check-round SI1000 factor by ``C``
    and keeps the logical-gate noise at SI1000 (for check-cost scans).
    """
    if name.startswith("check_scale_"):
        scale = float(name[len("check_scale_"):])
        factors = {f: scale for f in Q.SI1000.__dataclass_fields__}
        return NoiseConfig(check=_BASE.scaled(**factors))
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation {name!r}.")
    return ABLATIONS[name]


# ---------------------------------------------------------------------------
# Variant specification and runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantSpec:
    """One circuit family evaluated at every depth.

    Attributes:
        name: Unique label (used for seeds and file names).
        schedule: ``"serial"``, ``"zx"`` or ``"parallel"``.
        reset: Reset ancillas after forward/inverse rounds.
        forward: ``None`` executes every forward round (repo protocol);
            ``("unif", k)`` / ``("const", k)`` executes only the rounds that
            policy uses; ``("terminal", 0)`` executes none.
        inverse_every: Inverse-half check interval (0 = none).
        ablation: Key of ``ABLATIONS``.
        const0: The repo's ``const=0`` circuit (no check rounds, ideal X
            readout of all data); schedule/forward/inverse are ignored.
        all_policies: Also score every ``const``/``unif`` k = 1..22 LRB-D
            policy on the same shots (meaningful when ``forward is None``).
    """

    name: str
    schedule: str = "serial"
    reset: bool = True
    forward: Optional[tuple[str, int]] = None
    inverse_every: int = 0
    ablation: str = "full"
    const0: bool = False
    all_policies: bool = False

    def forward_rounds(self, depth: int) -> Optional[list[int]]:
        if self.forward is None:
            return None
        policy, k = self.forward
        return policy_forward_rounds(policy, k, depth)

    def build(self, sequence: Sequence[str], depth: int, p: float,
              terminal: str = "data5") -> VariantCircuit:
        noise = ablation_noise(self.ablation)
        if self.const0:
            return build_variant_circuit(
                sequence, depth, p, schedule="serial", reset=True,
                terminal=terminal, forward_check_rounds=(),
                terminal_check=False, noise=noise)
        return build_variant_circuit(
            sequence, depth, p, schedule=self.schedule, reset=self.reset,
            terminal=terminal, forward_check_rounds=self.forward_rounds(depth),
            inverse_check_every=self.inverse_every, noise=noise)


def task_seed(name: str, p: float, seq_index: int, depth: int,
              base_seed: int = 0) -> int:
    """Deterministic, distinct Stim seed per (variant, p, sequence, depth)."""
    key = f"{name}|{p:.6e}|{seq_index}|{depth}|{base_seed}".encode()
    return zlib.crc32(key) * 4096 + depth


ALL_POLICIES = tuple([("const", k) for k in range(1, 23)]
                     + [("unif", k) for k in range(1, 23)])


def run_variant(spec: VariantSpec, p: float, sequence: Sequence[str],
                depths: Sequence[int], shots: int, *,
                decode_shots: Optional[int] = None,
                batch_size: int = 100_000,
                seq_index: int = 0,
                base_seed: int = 0) -> dict[str, np.ndarray]:
    """Sample one sequence at every depth and score all methods on the same shots.

    Methods (counts per depth; ``__shots``, ``__accepted``, ``__errors``):

    ``raw``
        No decoding, no postselection (X_L readout as is).
    ``lrbd``
        LRB-D: accept when every *executed* check round has a trivial raw
        syndrome (the qutrit pipeline's rule, ``data2`` semantics). For
        ``const0`` it is the repo's rule (both ideal final X stabilizers
        trivial).
    ``lrbd_final``
        ``lrbd`` and the two final data-readout X detectors clean.
    ``mld5`` / ``mld2``
        Exact MLD with / without the final data detectors, on the first
        ``decode_shots`` shots. (``mld2`` is skipped for ``const0``.)
    ``lrbd_<policy>_<k>``
        Only with ``spec.all_policies``: every const/unif policy.

    Extra per-depth diagnostics: ``undet5``/``undet2`` (probability of
    undetectable logical mechanisms), ``num_rounds``, ``num_detectors``.
    """
    depths = list(depths)
    n_depths = len(depths)
    decode_shots = shots if decode_shots is None else min(decode_shots, shots)
    out: dict[str, np.ndarray] = {}

    def add(name: str, d_index: int, n: int, acc: int, err: int) -> None:
        for suffix, value in (("shots", n), ("accepted", acc),
                              ("errors", err)):
            key = f"{name}__{suffix}"
            if key not in out:
                out[key] = np.zeros(n_depths, dtype=np.int64)
            out[key][d_index] += value

    for name in ("undet5", "undet2", "num_rounds", "num_detectors"):
        out[name] = np.zeros(n_depths)

    for d_index, depth in enumerate(depths):
        lc5 = spec.build(sequence, depth, p, terminal="data5")
        dem5 = lc5.circuit.detector_error_model()
        mld5 = M.TransferMatrixMLD(dem5, M.lrb_detector_slots(lc5))
        mld2 = None
        n_round_dets = 4 * lc5.num_rounds
        if n_round_dets > 0:
            lc2 = spec.build(sequence, depth, p, terminal="data2")
            if lc2.detector_keys != lc5.detector_keys[:n_round_dets]:
                raise RuntimeError("data2 and data5 detectors disagree.")
            mld2 = M.TransferMatrixMLD(lc2.circuit.detector_error_model(),
                                       M.lrb_detector_slots(lc2))
            out["undet2"][d_index] = mld2.undetectable_logical_probability
        out["undet5"][d_index] = mld5.undetectable_logical_probability
        out["num_rounds"][d_index] = lc5.num_rounds
        out["num_detectors"][d_index] = lc5.circuit.num_detectors
        policy_columns = []
        if spec.all_policies and not spec.const0:
            for policy, k in ALL_POLICIES:
                used = B.used_rounds(policy, k, depth)
                policy_columns.append((f"lrbd_{policy}_{k}",
                                       [lc5.rounds.index(r) for r in used]))
        sampler = lc5.circuit.compile_detector_sampler(
            seed=task_seed(spec.name, p, seq_index, depth, base_seed))
        done = 0
        while done < shots:
            n = min(batch_size, shots - done)
            dets, obs = sampler.sample(n, separate_observables=True)
            flip = obs[:, 0]
            add("raw", d_index, n, n, int(flip.sum()))
            final_clean = ~dets[:, n_round_dets:].any(axis=1)
            if spec.const0:
                acc = final_clean
                add("lrbd", d_index, n, int(acc.sum()), int((acc & flip).sum()))
            else:
                raw = lc5.raw_syndromes(dets)
                round_bad = raw.any(axis=2)
                acc = ~round_bad.any(axis=1)
                add("lrbd", d_index, n, int(acc.sum()), int((acc & flip).sum()))
                acc_f = acc & final_clean
                add("lrbd_final", d_index, n, int(acc_f.sum()),
                    int((acc_f & flip).sum()))
                for name, columns in policy_columns:
                    acc_p = ~round_bad[:, columns].any(axis=1)
                    add(name, d_index, n, int(acc_p.sum()),
                        int((acc_p & flip).sum()))
            n_dec = min(n, max(decode_shots - done, 0))
            if n_dec > 0:
                if dets.shape[1] > 0:
                    llr5 = mld5.log_likelihood_ratio(dets[:n_dec])
                else:
                    llr5 = np.full(n_dec, np.inf)
                add("mld5", d_index, n_dec, n_dec,
                    int(((llr5 < 0) != flip[:n_dec]).sum()))
                if mld2 is not None:
                    llr2 = mld2.log_likelihood_ratio(dets[:n_dec,
                                                          :n_round_dets])
                    add("mld2", d_index, n_dec, n_dec,
                        int(((llr2 < 0) != flip[:n_dec]).sum()))
            done += n
    return out


# ---------------------------------------------------------------------------
# Exact first-order DEM coefficients (same definitions as the lead's
# scripts/qubit512/first_order.py, plus the data2-restricted MLD error)
# ---------------------------------------------------------------------------

def symptom_weights(dem: stim.DetectorErrorModel,
                    detector_mask: Optional[np.ndarray] = None
                    ) -> dict[tuple[int, ...], list[float]]:
    """``{symptom: [P0, P1]}`` summed over single DEM mechanisms.

    ``detector_mask`` restricts symptoms to the selected detectors (a decoder
    that does not see the other detectors cannot tell those classes apart).
    """
    weights: dict[tuple[int, ...], list[float]] = {}
    for mech in M.dem_mechanisms(dem):
        dets = mech.detectors
        if detector_mask is not None:
            dets = tuple(d for d in dets if detector_mask[d])
        entry = weights.setdefault(dets, [0.0, 0.0])
        entry[int(mech.flips_observable)] += mech.probability
    return weights


def first_order_coefficients(lc: Q.LRBCircuit, p: float) -> dict[str, float]:
    """Leading-order coefficients of one circuit built at small ``p``.

    ``E1_5`` / ``E1_2``: first-order error of exact MLD with / without the
    final data detectors, ``sum_s min(P0, P1)`` plus undetectable logicals,
    divided by ``p``. ``R1_5`` / ``R1_2``: first-order rejection of LRB-D on
    every executed round with / without the final data check. ``U1_2``:
    undetectable logical mass with the data2 readout. ``A2_5``: second-order
    accepted error of LRB-D with every detector, divided by ``p**2``.
    """
    dem = lc.circuit.detector_error_model()
    n_round = 4 * lc.num_rounds
    mask = np.arange(dem.num_detectors) < n_round
    out = {}
    for tag, weights in (("5", symptom_weights(dem)),
                         ("2", symptom_weights(dem, mask))):
        e1 = r1 = a2 = 0.0
        for symptom, (p0, p1) in weights.items():
            if not symptom:
                e1 += p1
                continue
            e1 += min(p0, p1)
            r1 += p0 + p1
            a2 += p0 * p1
        out[f"E1_{tag}"] = e1 / p
        out[f"R1_{tag}"] = r1 / p
        out[f"A2_{tag}"] = a2 / p ** 2
        out[f"U1_{tag}"] = weights.get((), [0.0, 0.0])[1] / p
    return out


# ---------------------------------------------------------------------------
# Statistics: the repo's fixed-offset fit, vectorised, and a sequence bootstrap
# ---------------------------------------------------------------------------

def _safe_weights(errors: np.ndarray) -> np.ndarray:
    """Row-wise copy of ``LRBPlotter._safe_weights`` (NaN = zero weight)."""
    err = np.asarray(errors, dtype=float)
    err = np.where(np.isnan(err), np.inf, err)
    finite = np.where(np.isfinite(err), err, np.nan)
    with np.errstate(all="ignore"):
        median = np.nanmedian(finite, axis=-1, keepdims=True)
    median = np.where(np.isfinite(median), median, 1.0)
    floor = np.maximum(median * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    return 1.0 / (err * err)


def fit_decay_batch(depths: Sequence[float], means: np.ndarray,
                    stds: np.ndarray, *, a_fixed: float = 0.0,
                    f_min: float = 1e-8, f_max: float = 1.0,
                    y_ceiling: Optional[float] = 1.0,
                    min_points: int = 2, grid: int = 2001,
                    iterations: int = 5) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]:
    """Fit ``y = a_fixed + b f^m`` row by row, exactly as the repo does.

    Mirrors ``LRBPlotter._fit_lrb_from_stats``: points with NaN mean or std
    are dropped, points with mean above ``y_ceiling`` are dropped, weights are
    ``1/std^2`` with the repo's floor, and ``f`` is found by five rounds of a
    2001-point grid scan, each narrowed to +-1/10 of the previous span.

    Args:
        depths: ``(n,)`` depth axis.
        means, stds: ``(rows, n)`` (or ``(n,)``) sequence means and standard
            deviations.

    Returns:
        ``(f, b, n_points)`` arrays of shape ``(rows,)``; ``f`` and ``b`` are
        NaN where fewer than ``min_points`` points remain.
    """
    x = np.asarray(depths, dtype=float)
    y = np.atleast_2d(np.asarray(means, dtype=float))
    e = np.atleast_2d(np.asarray(stds, dtype=float))
    rows = y.shape[0]
    valid = np.isfinite(y) & np.isfinite(e) & np.isfinite(x)[None, :]
    if y_ceiling is not None:
        valid &= np.where(np.isfinite(y), y, np.inf) <= y_ceiling
    n_points = valid.sum(axis=1)
    # Weights: computed on the valid points only, as the repo does.
    e_valid = np.where(valid, e, np.nan)
    w = _safe_weights(e_valid)
    w = np.where(valid, w, 0.0)
    y0 = np.where(valid, y - a_fixed, 0.0)
    low = np.full(rows, float(f_min))
    high = np.full(rows, float(f_max))
    f_best = np.full(rows, np.nan)
    b_best = np.full(rows, np.nan)
    ramp = np.arange(grid, dtype=float)
    for _ in range(iterations):
        # Same arithmetic as np.linspace(low, high, grid), row by row.
        f_grid = ramp[None, :] * ((high - low) / (grid - 1))[:, None]
        f_grid += low[:, None]
        f_grid[:, -1] = high
        z = np.power(f_grid[:, :, None], x[None, None, :])
        num = np.sum(w[:, None, :] * z * y0[:, None, :], axis=2)
        den = np.sum(w[:, None, :] * z * z, axis=2)
        den = np.where(den <= 0, np.nan, den)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = num / den
            resid = y0[:, None, :] - b[:, :, None] * z
            chi2 = np.sum(w[:, None, :] * resid * resid, axis=2)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        idx = np.argmin(chi2, axis=1)
        f_best = f_grid[np.arange(rows), idx]
        b_best = b[np.arange(rows), idx]
        span = (high - low) / 10.0
        low = np.maximum(f_min, f_best - span)
        high = np.minimum(f_max, f_best + span)
    bad = n_points < min_points
    f_best = np.where(bad, np.nan, f_best)
    b_best = np.where(bad, np.nan, b_best)
    return f_best, b_best, n_points


def per_sequence_fidelity(shots: np.ndarray, accepted: np.ndarray,
                          errors: np.ndarray) -> np.ndarray:
    """``F = |1 - 2 errors/accepted|`` per sequence and depth (NaN if 0 acc)."""
    return B.fidelity_from_counts(accepted, errors)


def sequence_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std over sequences (axis 0), as in the repo CSVs."""
    with np.errstate(invalid="ignore"):
        return np.mean(values, axis=0), np.std(values, axis=0)


@dataclass
class DecayEstimate:
    """Point fit plus sequence-bootstrap summary of one decay curve."""

    f: float
    b: float
    n_points: int
    f_boot_std: float
    f_boot_lo: float
    f_boot_hi: float
    boot_samples: np.ndarray

    def as_dict(self) -> dict:
        return {"f": self.f, "b": self.b, "n_points": self.n_points,
                "f_err": self.f_boot_std, "f_lo": self.f_boot_lo,
                "f_hi": self.f_boot_hi,
                "f_boot_valid": int(np.isfinite(self.boot_samples).sum())}


def bootstrap_indices(num_sequences: int, num_boot: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Sequence-resampling indices ``(num_boot, num_sequences)``."""
    return rng.integers(0, num_sequences, size=(num_boot, num_sequences))


def fit_with_bootstrap(depths: Sequence[int], per_seq: np.ndarray,
                       indices: np.ndarray, chunk: int = 200,
                       boot_grid: int = 401, boot_iterations: int = 6
                       ) -> DecayEstimate:
    """Repo fit of the sequence mean plus the same fit on bootstrap resamples.

    The point estimate uses the repo's exact scan (2001 points, 5 rounds).
    Resamples use ``boot_grid`` points and ``boot_iterations`` rounds; the
    defaults (401, 6) end at the repo's final grid spacing (8e-7) with 4x
    fewer evaluations.

    Args:
        per_seq: ``(num_sequences, num_depths)`` per-sequence values.
        indices: ``(num_boot, num_sequences)`` resampling indices; pass the
            same array to every method to get paired bootstrap samples.
    """
    mean, std = sequence_stats(per_seq)
    f, b, n_pts = fit_decay_batch(depths, mean[None, :], std[None, :])
    samples = []
    for start in range(0, indices.shape[0], chunk):
        idx = indices[start:start + chunk]
        resampled = per_seq[idx]                       # (B, S, D)
        with np.errstate(invalid="ignore"):
            means = np.mean(resampled, axis=1)
            stds = np.std(resampled, axis=1)
        fb, _, _ = fit_decay_batch(depths, means, stds, grid=boot_grid,
                                   iterations=boot_iterations)
        samples.append(fb)
    boot = np.concatenate(samples) if samples else np.zeros(0)
    finite = boot[np.isfinite(boot)]
    if len(finite) > 1:
        boot_std = float(np.std(finite, ddof=1))
        lo, hi = (float(v) for v in np.percentile(finite, [15.865, 84.135]))
    else:
        boot_std, lo, hi = float("nan"), float("nan"), float("nan")
    return DecayEstimate(float(f[0]), float(b[0]), int(n_pts[0]), boot_std,
                         lo, hi, boot)


def error_per_step(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Average logical infidelity per depth step for a qubit: ``(1 - f) / 2``."""
    return (1.0 - np.asarray(f)) / 2.0
