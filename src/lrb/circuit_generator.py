"""Circuit-generation utilities for logical and physical RB workflows.

This module centers on ``LRBCircuitGenerator``, which owns the full workflow
for constructing RB and LRB circuits and serializing them to disk. The class
composes reusable code definitions and keeps all noise-injection logic in one
place.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .clifford_recovery import CliffordRecoveryTable
from .detector_events import (
    LRB_CONST0_LOGICAL_OBSERVABLE_LABEL,
    LRB_LOGICAL_OBSERVABLE_LABEL,
    RB_LOGICAL_OBSERVABLE_LABEL,
    expression_from_measurement_indices,
    indices_and_coefficients_from_wire_terms,
    lrb_const0_stabilizer_detector_label,
    lrb_stabilizer_detector_label,
    measurement_indices_by_wire,
)

try:
    from sdim.circuit import Circuit
    from sdim.circuit_io import write_circuit
except Exception as sdim_exc:  # pragma: no cover - environment dependent.
    Circuit = Any
    write_circuit = None
    _SDIM_IMPORT_ERROR = sdim_exc
else:
    _SDIM_IMPORT_ERROR = None


def _require_sdim_runtime() -> None:
    """Raise a clear error when SDIM generation runtime is unavailable."""
    if _SDIM_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "SDIM runtime is unavailable in this environment. "
        "Circuit generation methods require a compatible SDIM install."
    ) from _SDIM_IMPORT_ERROR


DEPOLARIZING_NOISE_MODEL = "depolarizing"
SI1000_NOISE_MODEL = "si1000"
SD6_NOISE_MODEL = "sd6"
SUPPORTED_NOISE_MODELS = (
    DEPOLARIZING_NOISE_MODEL,
    SI1000_NOISE_MODEL,
    SD6_NOISE_MODEL,
)
_NOISE_MODEL_ALIASES = {
    "default": DEPOLARIZING_NOISE_MODEL,
    "legacy": DEPOLARIZING_NOISE_MODEL,
    "legacy_depolarizing": DEPOLARIZING_NOISE_MODEL,
    "depol": DEPOLARIZING_NOISE_MODEL,
    DEPOLARIZING_NOISE_MODEL: DEPOLARIZING_NOISE_MODEL,
    SI1000_NOISE_MODEL: SI1000_NOISE_MODEL,
    SD6_NOISE_MODEL: SD6_NOISE_MODEL,
}


@dataclass(frozen=True)
class CircuitLevelNoiseRates:
    """
    Noise strengths of a circuit-level model as multiples of the swept p.

    The swept parameter p is the two-qudit gate depolarizing probability.
    Rates follow the SD6 and SI1000 models of Gidney, Newman, Fowler and
    Broughton, "A Fault-Tolerant Honeycomb Memory", Quantum 5, 605 (2021),
    generalized to qudits: DEP1/DEP2 apply a uniformly random non-identity
    one-/two-qudit Pauli, and a flip applies ``X^a`` (``Z^a`` before
    ``M_X``) with ``a`` uniform on ``1..d-1``.

    Attributes:
        one_qudit_gate (float): DEP1 after every one-qudit Clifford.
        two_qudit_gate (float): DEP2 after every two-qudit Clifford.
        reset_flip (float): Flip after every RESET.
        measurement_flip (float): Flip just before every M/M_X, which shifts
            the recorded result.
        post_measurement (float): DEP1 after every M/M_X. Every measured
            ancilla is reset before reuse, so this never reaches a detector in
            the LRB circuits; it is kept for fidelity to the model.
        idle (float): DEP1 on every qudit idle during a gate moment.
        measure_reset_idle (float): DEP1 on every qudit idle during a
            measurement/reset moment (SI1000's resonator idle).
    """
    one_qudit_gate: float
    two_qudit_gate: float
    reset_flip: float
    measurement_flip: float
    post_measurement: float
    idle: float
    measure_reset_idle: float


CIRCUIT_LEVEL_NOISE_RATES = {
    SI1000_NOISE_MODEL: CircuitLevelNoiseRates(
        one_qudit_gate=0.1,
        two_qudit_gate=1.0,
        reset_flip=2.0,
        measurement_flip=5.0,
        post_measurement=1.0,
        idle=0.1,
        measure_reset_idle=2.0,
    ),
    SD6_NOISE_MODEL: CircuitLevelNoiseRates(
        one_qudit_gate=1.0,
        two_qudit_gate=1.0,
        reset_flip=1.0,
        measurement_flip=1.0,
        post_measurement=1.0,
        idle=1.0,
        measure_reset_idle=1.0,
    ),
}
CIRCUIT_LEVEL_NOISE_MODELS = tuple(CIRCUIT_LEVEL_NOISE_RATES)

_ONE_QUDIT_IDEAL_GATES = {
    "I",
    "X",
    "X_INV",
    "Z",
    "Z_INV",
    "H",
    "H_INV",
    "P",
    "P_INV",
    "MUL",
}
_TWO_QUDIT_IDEAL_GATES = {
    "CNOT",
    "CNOT_INV",
    "CZ",
    "CZ_INV",
    "SWAP",
}
_MEASUREMENT_GATES = {"M", "M_X"}
_RESET_GATES = {"RESET"}


# How an RB/LRB sequence C1..Cm is closed. ``mirror`` (the default, used by
# every production run) appends Cm^-1..C1^-1; ``compiled`` appends one
# catalog Clifford equal to (Cm...C1)^-1, as in standard RB.
MIRROR_INVERSE_MODE = "mirror"
COMPILED_INVERSE_MODE = "compiled"
SUPPORTED_INVERSE_MODES = (MIRROR_INVERSE_MODE, COMPILED_INVERSE_MODE)


def normalize_noise_model(noise_model: str) -> str:
    """
    Normalize user-facing noise-model aliases to canonical generator names.

    Args:
        noise_model (str): Requested model name.

    Returns:
        str: Canonical model name.

    Raises:
        ValueError: If the model name is unsupported.
    """
    key = str(noise_model).strip().lower().replace("-", "_")
    if key not in _NOISE_MODEL_ALIASES:
        supported = ", ".join(SUPPORTED_NOISE_MODELS)
        raise ValueError(
            f"Unsupported noise model '{noise_model}'. "
            f"Supported choices are: {supported}."
        )
    return _NOISE_MODEL_ALIASES[key]


@dataclass(frozen=True)
class LRBCodeDefinition:
    """
    Declarative container for code-specific LRB construction callbacks.

    The dataclass captures all code-dependent pieces required by the generic
    generator so new error-correcting or error-detecting codes can plug into
    the same RB/LRB workflow.

    Attributes:
        dimension (int): Local qudit dimension.
        physical_num_qudits (int): Physical RB circuit width.
        encoded_num_qudits (int): Encoded LRB circuit width.
        clifford_strings (Sequence[str]): Clifford labels used for random
            sampling.
        clifford_to_gate_sequence (Callable[[str], list[str]]): Mapping from a
            Clifford label to a gate-token sequence.
        clifford_inverse_map (dict[str, list[str]]): Clifford inverse lookup.
        apply_physical_gate (Callable[[Circuit, str], None]): Appends one
            physical RB gate token to a circuit.
        logical_plus_initial_state (Callable[[], Circuit]): Encoded
            initialization circuit builder.
        logical_gate_circuit (Callable[[str], Circuit]): Encoded logical-gate
            circuit builder.
        affected_wires (Callable[[list[str]], set[int]]): Returns data wires
            affected by one logical Clifford block.
        stabilizer_check_blocks (Sequence[tuple[Callable[[], Circuit],
            set[int]]]): Stabilizer-check circuit builders with ancilla-wire
            metadata.
        reset_measurement_wires (Callable[[], Circuit] | None): Optional reset
            block inserted between check rounds.
        terminal_measurement (Callable[[], Circuit]): Final logical readout
            block.
        terminal_const0_measurement (Callable[[], Circuit]): Error-free
            terminal direct data readout used only for ``const=0``.
        stabilizer_detector_wires (Sequence[int]): Ancilla wires whose
            check-round measurements should be exposed as SDIM detectors.
        logical_observable_terms (Sequence[tuple[int, int]]): Terminal
            logical-observable expression as ``(wire, coefficient)`` terms.
        terminal_x_stabilizer_terms (Sequence[Sequence[tuple[int, int]]]):
            Direct terminal X-stabilizer expressions for ``const=0`` circuits.
        depth_zero_noise_wires (Sequence[int]): Wires that receive the
            depth-zero noise model.

    Methods:
        This dataclass is declarative and defines no custom methods.
    """
    dimension: int
    physical_num_qudits: int
    encoded_num_qudits: int
    clifford_strings: Sequence[str]
    clifford_to_gate_sequence: Callable[[str], list[str]]
    clifford_inverse_map: dict[str, list[str]]
    apply_physical_gate: Callable[[Circuit, str], None]
    logical_plus_initial_state: Callable[[], Circuit]
    logical_gate_circuit: Callable[[str], Circuit]
    affected_wires: Callable[[list[str]], set[int]]
    stabilizer_check_blocks: Sequence[tuple[Callable[[], Circuit], set[int]]]
    reset_measurement_wires: Callable[[], Circuit] | None
    terminal_measurement: Callable[[], Circuit]
    terminal_const0_measurement: Callable[[], Circuit]
    stabilizer_detector_wires: Sequence[int] = ()
    logical_observable_terms: Sequence[tuple[int, int]] = ()
    terminal_x_stabilizer_terms: Sequence[Sequence[tuple[int, int]]] = ()
    depth_zero_noise_wires: Sequence[int] = ()


class LRBCodeDefinitionFactory:
    """
    Utility factory for reusable code-definition helper callbacks.

    Attributes:
        This class is stateless and stores no persistent attributes.

    Methods:
        apply_single_qudit_gate_to_wire_zero: Default physical-RB gate mapper.
    """

    @staticmethod
    def apply_single_qudit_gate_to_wire_zero(circuit: Circuit, gate: str
                                             ) -> None:
        """
            Apply one physical RB gate token on wire zero.

        Args:
            circuit (Circuit): Target circuit receiving the gate.
            gate (str): Gate token to append.

        Returns:
            None: The method mutates the provided circuit.

        Raises:
            ValueError: If SDIM rejects the gate token or wire index.
        """
        circuit.add_gate(gate, 0)


@dataclass
class LRBCircuitGenerator:
    """
    Builder for RB/LRB circuit families and serialized experiment artifacts.

    The class generates physical RB and encoded LRB circuits from a pluggable
    ``LRBCodeDefinition`` and writes depth-indexed circuits to disk.

    Attributes:
        with_default_noise_channel (str): Default serialized noise-channel tag
            used for injected ``N1`` gates.
        code_definition (LRBCodeDefinition | None): Code-specific construction
            hooks used by all RB/LRB builders.
        noise_model (str): Noise model (depolarizing, si1000 or sd6).
        inverse_mode (str): How sequences are closed: ``mirror`` (default,
            Cm^-1..C1^-1) or ``compiled`` (one recovery Clifford).

    Methods:
        _add_n1_depolarizing/_add_n2_depolarizing: Append serialized noise
            gates with consistent parameter handling.
        inject_stabcheck_noise: Clone stabilizer-check blocks and insert
            ancilla-coupled noise.
        generate_random_clifford_strings/clifford_string_seq_to_list: Build
            Clifford samples and convert descriptors into gate lists.
        generate_rb_clifford_sequence: Build unencoded RB depth families.
        generate_lrb_clifford_sequence: Build folded-code encoded depth
            families.
        generate_lrb_const0_clifford_sequence: Build encoded circuits for the
            ``const=0`` direct terminal X-stabilizer protocol.
        append_lrb_detector_events: Add compact SDIM detector/logical payloads
            to ordinary encoded LRB circuits.
        append_lrb_const0_detector_events: Add compact SDIM detector/logical
            payloads to special direct terminal X-data circuits.
        append_rb_detector_events: Add compact SDIM logical payloads to
            physical RB circuits.
        update_noise_param: Rewrite noise probabilities on generated circuits.
        generate_tests: Export all circuits for all Clifford seeds and
            probabilities.
    """
    with_default_noise_channel: str = "d"
    code_definition: LRBCodeDefinition | None = None
    noise_model: str = DEPOLARIZING_NOISE_MODEL
    inverse_mode: str = MIRROR_INVERSE_MODE

    def __post_init__(self) -> None:
        """Normalize noise-model aliases after dataclass initialization."""
        self.noise_model = normalize_noise_model(self.noise_model)
        if self.inverse_mode not in SUPPORTED_INVERSE_MODES:
            raise ValueError(
                f"Unsupported inverse mode '{self.inverse_mode}'. "
                f"Supported choices are: {', '.join(SUPPORTED_INVERSE_MODES)}."
            )
        self._recovery_table: CliffordRecoveryTable | None = None

    def _closing_gate_lists(
        self,
        clifford_strings: list[str],
        full_clifford_inverse_list: list[list[str]],
        max_depth: int,
        depth: int,
    ) -> list[list[str]]:
        """
            Return the gate lists that close a depth-``depth`` sequence.

        ``mirror`` returns the inverses of the first ``depth`` Cliffords in
        reverse order (the suffix of ``full_clifford_inverse_list``);
        ``compiled`` returns one catalog Clifford equal to their inverse
        product. Depth 0 has no closing gates in either mode.
        """
        if depth == 0:
            return []
        if self.inverse_mode == MIRROR_INVERSE_MODE:
            return full_clifford_inverse_list[(max_depth - depth):]
        if self._recovery_table is None:
            code_definition = self._require_code_definition()
            self._recovery_table = CliffordRecoveryTable(
                code_definition.clifford_strings,
                code_definition.clifford_to_gate_sequence,
                dimension=code_definition.dimension,
            )
        return [self._recovery_table.recovery_gates(clifford_strings[:depth])]

    def _require_code_definition(self) -> LRBCodeDefinition:
        """
        Require code definition.
        
        Args:
            None: This method relies on object state and accepts no
                additional inputs.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        if self.code_definition is None:
            raise ValueError(
                "LRBCircuitGenerator requires an explicit code_definition."
            )
        return self.code_definition

    def _add_n1_noise(
        self,
        circuit: Circuit,
        wire: int,
        prob: float,
        noise_channel: str = "d",
        prob_scale: float | None = None,
    ) -> None:
        """
            Append a one-qudit Pauli noise gate and normalize serialized
            noise-channel metadata.

        Args:
            circuit (Circuit): Circuit receiving the inserted noise operation.
            wire (int): Target wire index for the noise gate.
            prob (float): Probability assigned to the gate.
            noise_channel (str): SDIM single-qudit channel tag.
            prob_scale (float | None): Optional multiplier applied when
                materializing a probability sweep value. SDIM rejects unknown
                gate parameters, so it is stored on the operation after
                ``add_gate`` and stripped again by ``write_circuit_file``.

        Returns:
            None: Updates the provided circuit in-place.

        Raises:
            ValueError: Propagated if SDIM rejects invalid gate arguments.
        """
        circuit.add_gate("N1", wire, noise_channel=noise_channel, prob=prob)
        if prob_scale is not None:
            circuit.operations[-1].params["prob_scale"] = prob_scale

    def _add_n1_depolarizing(
        self,
        circuit: Circuit,
        wire: int,
        prob: float,
        prob_scale: float | None = None,
    ) -> None:
        """
            Append a one-qutrit depolarizing gate.

        Args:
            circuit (Circuit): Circuit receiving the inserted noise operation.
            wire (int): Target wire index for the noise gate.
            prob (float): Depolarizing probability assigned to the gate.
            prob_scale (float | None): Optional sweep multiplier.

        Returns:
            None: Updates the provided circuit in-place.

        Raises:
            ValueError: Propagated if SDIM rejects invalid gate arguments.
        """
        self._add_n1_noise(
            circuit,
            wire,
            prob,
            noise_channel=self.with_default_noise_channel,
            prob_scale=prob_scale,
        )

    def _add_n2_depolarizing(self, circuit: Circuit, q0: int, q1: int,
                             prob: float,
                             prob_scale: float | None = None) -> None:
        """
            Append a two-qutrit depolarizing gate and enforce scalar ``prob``
            serialization for later runtime expansion.

        Args:
            circuit (Circuit): Circuit receiving the inserted noise operation.
            q0 (int): First qudit index for the two-qudit noise gate.
            q1 (int): Second qudit index for the two-qudit noise gate.
            prob (float): Depolarizing probability assigned to the gate.
            prob_scale (float | None): Optional sweep multiplier.

        Returns:
            None: Updates the provided circuit in-place.

        Raises:
            ValueError: Propagated if SDIM rejects invalid gate arguments.
        """
        circuit.add_gate("N2", q0, q1, prob=prob)
        if prob_scale is not None:
            circuit.operations[-1].params["prob_scale"] = prob_scale

    def _append_operation_copy(self, circuit: Circuit, operation: Any) -> None:
        """
            Append a copy of one SDIM operation to ``circuit``.

        Args:
            circuit (Circuit): Destination circuit.
            operation (Any): Source SDIM ``CircuitInstruction``.

        Returns:
            None: Mutates the destination circuit.

        Raises:
            ValueError: Propagated if SDIM rejects the copied operation.
        """
        params = {} if operation.params is None else dict(operation.params)
        if operation.target_index is None:
            circuit.add_gate(operation.gate_name, operation.qudit_index,
                             **params)
        else:
            circuit.add_gate(
                operation.gate_name,
                operation.qudit_index,
                operation.target_index,
                **params,
            )

    @staticmethod
    def _operation_qudits(operation: Any) -> tuple[int, ...]:
        """Return the wires one SDIM operation acts on."""
        if operation.target_index is None:
            return (int(operation.qudit_index),)
        return (int(operation.qudit_index), int(operation.target_index))

    def schedule_moments(
        self,
        source: Circuit,
    ) -> list[tuple[bool, list[Any]]]:
        """
            Pack an ideal subcircuit into as-soon-as-possible moments.

        Each operation goes into the earliest moment after the previous
        operation on each of its wires, so operations on disjoint wires run in
        parallel while the order on every wire is kept. A moment holds either
        unitary gates or collapse operations (measurements and resets), never
        both, so idle wires can receive the right idle noise. Measurements
        whose wire is not used again in ``source`` are deferred to one final
        measurement moment, so a block of checks reads out together.

        Args:
            source (Circuit): Ideal subcircuit of Clifford gates,
                measurements and resets.

        Returns:
            list[tuple[bool, list[Any]]]: ``(is_collapse_moment,
            operations)`` pairs in time order. Operations within a moment act
            on disjoint wires and keep their source order.

        Raises:
            ValueError: If ``source`` contains an operation that cannot be
                scheduled, such as a noise gate or detector annotation.
        """
        operations = list(source.operations)
        schedulable = (
            _ONE_QUDIT_IDEAL_GATES
            | _TWO_QUDIT_IDEAL_GATES
            | _MEASUREMENT_GATES
            | _RESET_GATES
        )
        last_use: dict[int, int] = {}
        for index, operation in enumerate(operations):
            if operation.gate_name not in schedulable:
                raise ValueError(
                    f"Cannot schedule {operation.gate_name} inside a "
                    "circuit-level noise segment."
                )
            for wire in self._operation_qudits(operation):
                last_use[wire] = index

        deferred = [
            index for index, operation in enumerate(operations)
            if operation.gate_name in _MEASUREMENT_GATES
            and last_use[int(operation.qudit_index)] == index
        ]
        deferred_set = set(deferred)

        moments: list[tuple[bool, list[Any]]] = []
        ready: dict[int, int] = {}
        for index, operation in enumerate(operations):
            if index in deferred_set:
                continue
            collapse = operation.gate_name in (
                _MEASUREMENT_GATES | _RESET_GATES)
            wires = self._operation_qudits(operation)
            slot = max(ready.get(wire, 0) for wire in wires)
            while slot < len(moments) and moments[slot][0] != collapse:
                slot += 1
            if slot == len(moments):
                moments.append((collapse, []))
            moments[slot][1].append(operation)
            for wire in wires:
                ready[wire] = slot + 1

        if deferred:
            moments.append((True, [operations[index] for index in deferred]))
        return moments

    def _add_scaled_depolarizing(
        self,
        circuit: Circuit,
        wire: int,
        prob_scale: float,
    ) -> None:
        """Append DEP1 at ``prob_scale * p`` unless the scale is zero."""
        if prob_scale > 0.0:
            self._add_n1_depolarizing(
                circuit, wire, 0.0, prob_scale=prob_scale)

    def apply_circuit_level_noise(
        self,
        source: Circuit,
        noisy_measurements: bool = True,
    ) -> Circuit:
        """
            Schedule an ideal subcircuit into moments and insert the
            generator's circuit-level (SI1000 or SD6) noise.

        Each moment adds DEP1 after one-qudit gates, DEP2 after two-qudit
        gates, a flip after resets, a result flip before and DEP1 after
        measurements, and idle DEP1 on every wire the moment leaves untouched
        (``measure_reset_idle`` in measurement/reset moments, ``idle``
        otherwise). The identity gate ``I`` occupies a moment and receives
        idle noise. Moments never span two calls, so each call is one
        barrier-separated segment of the protocol. All noise gates carry a
        ``prob_scale`` relative to p; ``update_noise_param`` materializes
        them.

        Args:
            source (Circuit): Ideal source subcircuit.
            noisy_measurements (bool): Whether measurements in ``source``
                receive measurement noise. Idle noise is applied either way.

        Returns:
            Circuit: Noisy, rescheduled copy of ``source``.

        Raises:
            ValueError: If the generator's noise model is not circuit-level
                or ``source`` cannot be scheduled.
        """
        _require_sdim_runtime()
        if self.noise_model not in CIRCUIT_LEVEL_NOISE_RATES:
            raise ValueError(
                f"Noise model '{self.noise_model}' has no circuit-level "
                "rates."
            )
        rates = CIRCUIT_LEVEL_NOISE_RATES[self.noise_model]
        output = Circuit(
            dimension=source.dimension,
            num_qudits=source.num_qudits,
        )

        for collapse, moment in self.schedule_moments(source):
            busy: set[int] = set()
            for operation in moment:
                busy.update(self._operation_qudits(operation))
                gate_name = operation.gate_name
                wire = int(operation.qudit_index)

                if gate_name in _MEASUREMENT_GATES:
                    if noisy_measurements:
                        self._add_n1_noise(
                            output,
                            wire,
                            0.0,
                            noise_channel=(
                                "p" if gate_name == "M_X" else "f"),
                            prob_scale=rates.measurement_flip,
                        )
                    self._append_operation_copy(output, operation)
                    if noisy_measurements:
                        self._add_scaled_depolarizing(
                            output, wire, rates.post_measurement)
                elif gate_name in _RESET_GATES:
                    self._append_operation_copy(output, operation)
                    self._add_n1_noise(
                        output,
                        wire,
                        0.0,
                        noise_channel="f",
                        prob_scale=rates.reset_flip,
                    )
                elif gate_name == "I":
                    self._append_operation_copy(output, operation)
                    self._add_scaled_depolarizing(output, wire, rates.idle)
                elif operation.target_index is None:
                    self._append_operation_copy(output, operation)
                    self._add_scaled_depolarizing(
                        output, wire, rates.one_qudit_gate)
                else:
                    self._append_operation_copy(output, operation)
                    self._add_n2_depolarizing(
                        output,
                        wire,
                        int(operation.target_index),
                        0.0,
                        prob_scale=rates.two_qudit_gate,
                    )

            idle_scale = (
                rates.measure_reset_idle if collapse else rates.idle)
            for wire in range(source.num_qudits):
                if wire not in busy:
                    self._add_scaled_depolarizing(output, wire, idle_scale)

        return output

    def inject_stabcheck_noise(
        self,
        stabcheck_circuit: Circuit,
        ancilla_wires: set[int],
        prob: float = 0.0,
    ) -> Circuit:
        """
            Copy a stabilizer-check subcircuit and inject ancilla-dependent N1
            and N2 depolarizing gates after selected operations.

        Args:
            stabcheck_circuit (Circuit): Source stabilizer-check circuit block.
            ancilla_wires (set[int]): Ancilla indices used to detect ancilla-
                data couplings for noise placement.
            prob (float): Depolarizing probability used for inserted gates.

        Returns:
            Circuit: New circuit containing original operations and injected
            noise operations.

        Raises:
            ValueError: Propagated if injected gate arguments are invalid.
        """
        _require_sdim_runtime()
        output = Circuit(
            dimension=stabcheck_circuit.dimension,
            num_qudits=stabcheck_circuit.num_qudits,
        )

        # Rebuild the block gate-by-gate so we can inject noise contextually.
        for operation in stabcheck_circuit.operations:
            params = {} if operation.params is None else dict(operation.params)

            if operation.target_index is None:
                output.add_gate(operation.gate_name, operation.qudit_index,
                                **params)
            else:
                output.add_gate(
                    operation.gate_name,
                    operation.qudit_index,
                    operation.target_index,
                    **params,
                )

            if (operation.target_index is None
                    and operation.gate_name in ("H", "H_INV")
                    and operation.qudit_index in ancilla_wires):
                # Ancilla basis-change gates get local single-qudit noise.
                self._add_n1_depolarizing(output, operation.qudit_index, prob)

            if (operation.target_index is not None
                    and operation.gate_name in ("CNOT", "CNOT_INV")
                    and ((operation.qudit_index in ancilla_wires) ^
                         (operation.target_index in ancilla_wires))):
                # Ancilla/data coupling gates get two-qudit depolarizing noise.
                self._add_n2_depolarizing(output, operation.qudit_index,
                                          operation.target_index, prob)

        return output

    def append_lrb_detector_events(self, circuit: Circuit, depth: int) -> None:
        """
            Append ordinary LRB stabilizer detectors and logical observable.

        The detector operations are added after all stabilizer and terminal
        logical measurements have already been appended. They do not alter the
        simulated quantum state; they ask SDIM to return compact vectorized
        Pauli-frame event arrays for the measurements the legacy unpacker was
        already reading from the raw measurement-result tensor.

        For ordinary LRB circuits, every check-round detector references one
        true ancilla measurement. The profile-level
        ``stabilizer_detector_wires`` ordering is the same ordering consumed by
        the runtime postselection spec, so the generated labels line up with
        the detector-aware unpack path. Reset operations are deliberately
        ignored because SDIM detector expressions count only ``M`` and ``M_X``
        operations as ``rec`` targets.

        Args:
            circuit (Circuit): Fully assembled encoded LRB circuit.
            depth (int): Benchmark depth. The circuit contains ``depth + 1``
                stabilizer-check layers, including the final post-inverse
                check.

        Returns:
            None: Mutates ``circuit`` by appending SDIM event annotations.

        Raises:
            KeyError: If a required detector wire was not measured enough
                times for the requested depth.
            ValueError: If detector-expression construction fails.
        """
        code_definition = self._require_code_definition()
        measurement_indices = measurement_indices_by_wire(circuit)

        for check_round in range(depth + 1):
            for wire in code_definition.stabilizer_detector_wires:
                measurement_index = measurement_indices[int(wire)][
                    check_round]
                circuit.add_gate(
                    "DETECTOR",
                    expr=expression_from_measurement_indices(
                        [measurement_index]),
                    label=lrb_stabilizer_detector_label(
                        check_round, int(wire)),
                )

        logical_indices, logical_coefficients = (
            indices_and_coefficients_from_wire_terms(
                measurement_indices,
                code_definition.logical_observable_terms,
                use_last_measurement=True,
            )
        )
        circuit.add_gate(
            "LOGICAL_OBSERVABLE",
            expr=expression_from_measurement_indices(
                logical_indices, logical_coefficients),
            label=LRB_LOGICAL_OBSERVABLE_LABEL,
        )

    def append_lrb_const0_detector_events(
        self,
        circuit: Circuit,
        depth: int,
    ) -> None:
        """
            Append detectors for the direct terminal-X ``const=0`` protocol.

        ``const=0`` circuits have no intermediate ancilla checks. The only
        postselection decision is placed in slot ``depth`` and is computed from
        direct X-basis data measurements at the end of the circuit. This method
        exposes each terminal X-stabilizer as a separate SDIM detector and the
        logical X readout as one SDIM logical observable, allowing runtime code
        to reconstruct the old pass/logical arrays from compact vectorized
        event data.

        Args:
            circuit (Circuit): Fully assembled encoded ``const=0`` circuit.
            depth (int): Benchmark depth. The value is accepted for API
                symmetry with ordinary LRB detector annotation; the detector
                labels themselves do not need the depth because there is only
                one terminal direct-X check layer.

        Returns:
            None: Mutates ``circuit`` by appending SDIM event annotations.

        Raises:
            KeyError: If a terminal data wire needed by the profile was not
                measured.
            ValueError: If detector-expression construction fails.
        """
        _ = depth
        code_definition = self._require_code_definition()
        measurement_indices = measurement_indices_by_wire(circuit)

        for stabilizer_index, stabilizer_terms in enumerate(
                code_definition.terminal_x_stabilizer_terms):
            indices, coefficients = indices_and_coefficients_from_wire_terms(
                measurement_indices,
                stabilizer_terms,
                use_last_measurement=True,
            )
            circuit.add_gate(
                "DETECTOR",
                expr=expression_from_measurement_indices(
                    indices, coefficients),
                label=lrb_const0_stabilizer_detector_label(
                    stabilizer_index),
            )

        logical_indices, logical_coefficients = (
            indices_and_coefficients_from_wire_terms(
                measurement_indices,
                code_definition.logical_observable_terms,
                use_last_measurement=True,
            )
        )
        circuit.add_gate(
            "LOGICAL_OBSERVABLE",
            expr=expression_from_measurement_indices(
                logical_indices, logical_coefficients),
            label=LRB_CONST0_LOGICAL_OBSERVABLE_LABEL,
        )

    @staticmethod
    def append_rb_detector_events(circuit: Circuit) -> None:
        """
            Append a physical-RB logical observable for wire-zero readout.

        Physical RB records only the terminal measurement of wire ``0``. Adding
        a single logical observable lets the runtime consume SDIM's compact
        vectorized event array when available while preserving the raw
        measurement fallback for older circuits.

        Args:
            circuit (Circuit): Fully assembled physical RB circuit.

        Returns:
            None: Mutates ``circuit`` by appending one logical observable.

        Raises:
            KeyError: If wire ``0`` was not measured before annotation.
        """
        measurement_indices = measurement_indices_by_wire(circuit)
        measurement_index = measurement_indices[0][-1]
        circuit.add_gate(
            "LOGICAL_OBSERVABLE",
            expr=expression_from_measurement_indices([measurement_index]),
            label=RB_LOGICAL_OBSERVABLE_LABEL,
        )

    def generate_random_clifford_strings(self, depth: int) -> list[str]:
        """
            Sample random single-qutrit Clifford descriptors from the
            exhaustive Clifford catalog.

        Args:
            depth (int): Number of descriptors to sample.

        Returns:
            list[str]: Randomly sampled Clifford descriptor strings.

        Raises:
            ValueError: Propagated if ``depth`` is invalid for random sampling.
        """
        code_definition = self._require_code_definition()
        return [
            random.choice(code_definition.clifford_strings)
            for _ in range(depth)
        ]

    def clifford_string_seq_to_list(
            self, clifford_strings: list[str]) -> list[list[str]]:
        """
            Convert compact Clifford descriptor strings into execution-order
            gate-sequence lists.

        Args:
            clifford_strings (list[str]): Symbolic Clifford descriptors.

        Returns:
            list[list[str]]: Gate-sequence representation for each descriptor.

        Raises:
            ValueError: Propagated if descriptor parsing fails downstream.
        """
        code_definition = self._require_code_definition()
        return [
            code_definition.clifford_to_gate_sequence(clifford)
            for clifford in clifford_strings
        ]

    def _circuit_level_noise(self) -> bool:
        """Whether the generator uses a moment-scheduled circuit-level model."""
        return self.noise_model in CIRCUIT_LEVEL_NOISE_RATES

    def _noisy_segment(self, segment: Circuit, with_noise: bool) -> Circuit:
        """Apply circuit-level noise to one protocol segment if requested."""
        if not with_noise:
            return segment
        return self.apply_circuit_level_noise(segment)

    def _clifford_and_inverse_gate_lists(
        self,
        clifford_strings: list[str],
    ) -> tuple[list[list[str]], list[list[str]]]:
        """
            Return execution-order gate lists for a maximum-depth Clifford
            sample and for its inverses in reverse (closing) order.
        """
        code_definition = self._require_code_definition()
        full_clifford_gate_list = [
            code_definition.clifford_to_gate_sequence(clifford)
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list = [
            code_definition.clifford_inverse_map[clifford]
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list.reverse()
        return full_clifford_gate_list, full_clifford_inverse_list

    def _logical_clifford_circuit(self, clifford: list[str]) -> Circuit:
        """Concatenate the encoded circuits of one Clifford's logical tokens."""
        code_definition = self._require_code_definition()
        circuit = Circuit(
            dimension=code_definition.dimension,
            num_qudits=code_definition.encoded_num_qudits,
        )
        for logical_gate in clifford:
            circuit = circuit + code_definition.logical_gate_circuit(
                logical_gate)
        return circuit

    def _check_round_circuit(self) -> Circuit:
        """
            Build one ideal stabilizer-check round for circuit-level models.

        Ancillas are reset immediately before they are used, then every
        stabilizer block runs, so the whole round is one scheduling segment:
        blocks on disjoint wires overlap and all ancilla readouts share the
        final measurement moment.
        """
        code_definition = self._require_code_definition()
        circuit = Circuit(
            dimension=code_definition.dimension,
            num_qudits=code_definition.encoded_num_qudits,
        )
        if code_definition.reset_measurement_wires is not None:
            circuit = circuit + code_definition.reset_measurement_wires()
        for stab_factory, _ in code_definition.stabilizer_check_blocks:
            circuit = circuit + stab_factory()
        return circuit

    def generate_rb_clifford_sequence(self,
                                      depths: list[int],
                                      with_noise: bool = True
                                      ) -> list[Circuit]:
        """
            Build single-qutrit RB circuits for all requested depths using a
            shared max-depth random Clifford sample and matching inverse tails.

        Under SI1000/SD6 every Clifford is one noisy scheduling segment; the
        terminal measurement stays ideal.

        Args:
            depths (list[int]): Benchmark depth values to instantiate.
            with_noise (bool): Whether to insert noise.

        Returns:
            list[Circuit]: One physical RB circuit per requested depth.

        Raises:
            ValueError: Propagated if depth-dependent slicing or gate insertion
                receives invalid values.
        """
        _require_sdim_runtime()
        subcircuits: list[Circuit] = []
        code_definition = self._require_code_definition()
        sorted_depths = sorted(depths)
        max_depth = sorted_depths[-1]

        # Sample once at max depth, then reuse prefixes for each depth.
        clifford_strings = self.generate_random_clifford_strings(max_depth)
        full_clifford_gate_list, full_clifford_inverse_list = (
            self._clifford_and_inverse_gate_lists(clifford_strings))

        for depth in sorted_depths:
            circuit = Circuit(
                dimension=code_definition.dimension,
                num_qudits=code_definition.physical_num_qudits,
            )

            if depth == 0:
                clifford_gates: list[list[str]] = []
                inverse_gates: list[list[str]] = []
            else:
                # Prefix selects forward gates; suffix selects inverse closure.
                clifford_gates = full_clifford_gate_list[:depth]
                inverse_gates = self._closing_gate_lists(
                    clifford_strings, full_clifford_inverse_list,
                    max_depth, depth)

            if self._circuit_level_noise():
                for clifford in clifford_gates + inverse_gates:
                    segment = Circuit(
                        dimension=code_definition.dimension,
                        num_qudits=code_definition.physical_num_qudits,
                    )
                    for gate in clifford:
                        code_definition.apply_physical_gate(segment, gate)
                    circuit = circuit + self._noisy_segment(
                        segment, with_noise)
            else:
                for clifford in clifford_gates:
                    for gate in clifford:
                        code_definition.apply_physical_gate(circuit, gate)
                    if with_noise:
                        circuit.add_gate(
                            "N1",
                            0,
                            noise_channel=self.with_default_noise_channel,
                            prob=0.0)

                for clifford in inverse_gates:
                    for gate in clifford:
                        code_definition.apply_physical_gate(circuit, gate)

                if with_noise:
                    circuit.add_gate(
                        "N1",
                        0,
                        noise_channel=self.with_default_noise_channel,
                        prob=0.0)
            circuit.add_gate("M", 0)
            self.append_rb_detector_events(circuit)
            subcircuits.append(circuit)

        return subcircuits

    def generate_lrb_clifford_sequence(
        self,
        depths: list[int],
        with_noise: bool = True,
        clifford_strings: list[str] | None = None,
    ) -> list[Circuit]:
        """
            Build folded-code logical RB circuits for all requested depths,
            including logical gate expansion, stabilizer checks, and optional
            noise insertion.

        Each forward logical Clifford is followed by one stabilizer-check
        round; the mirrored inverse Cliffords run back to back and are
        followed by one final check round. Under SI1000/SD6 every Clifford and
        every check round is one noisy scheduling segment, and each check
        round resets its ancillas just before use. The encoded preparation
        and terminal logical readout stay ideal.

        Args:
            depths (list[int]): Benchmark depth values to instantiate.
            with_noise (bool): Whether to insert physical noise and noisy
                stabilizer-check blocks.
            clifford_strings (list[str] | None): Optional pre-sampled maximum
                depth Clifford descriptor sequence.

        Returns:
            list[Circuit]: One encoded logical RB circuit per requested depth.

        Raises:
            ValueError: Propagated if logical-operator expansion or circuit
                assembly receives unsupported gate symbols.
        """
        _require_sdim_runtime()
        subcircuits: list[Circuit] = []
        code_definition = self._require_code_definition()
        sorted_depths = sorted(depths)
        max_depth = sorted_depths[-1]

        if clifford_strings is None:
            clifford_strings = self.generate_random_clifford_strings(max_depth)
        elif len(clifford_strings) < max_depth:
            raise ValueError(
                "Pre-sampled Clifford sequence is shorter than max depth.")
        full_clifford_gate_list, full_clifford_inverse_list = (
            self._clifford_and_inverse_gate_lists(clifford_strings))

        for depth in sorted_depths:
            circuit = Circuit(
                dimension=code_definition.dimension,
                num_qudits=code_definition.encoded_num_qudits,
            )
            # The encoded state-preparation circuit is intentionally ideal:
            # these simulations isolate the LRB-D protocol after preparation.
            circuit = circuit + code_definition.logical_plus_initial_state()

            if depth == 0:
                clifford_gates: list[list[str]] = []
                inverse_gates: list[list[str]] = []
            else:
                clifford_gates = full_clifford_gate_list[:depth]
                inverse_gates = self._closing_gate_lists(
                    clifford_strings, full_clifford_inverse_list,
                    max_depth, depth)

            if self._circuit_level_noise():
                for clifford in clifford_gates:
                    circuit = circuit + self._noisy_segment(
                        self._logical_clifford_circuit(clifford), with_noise)
                    circuit = circuit + self._noisy_segment(
                        self._check_round_circuit(), with_noise)
                for clifford in inverse_gates:
                    circuit = circuit + self._noisy_segment(
                        self._logical_clifford_circuit(clifford), with_noise)
                circuit = circuit + self._noisy_segment(
                    self._check_round_circuit(), with_noise)
            else:
                circuit = self._append_depolarizing_lrb_body(
                    circuit, depth, clifford_gates, inverse_gates, with_noise)

            circuit = circuit + code_definition.terminal_measurement()
            self.append_lrb_detector_events(circuit, depth)
            subcircuits.append(circuit)

        return subcircuits

    def _append_depolarizing_lrb_body(
        self,
        circuit: Circuit,
        depth: int,
        clifford_gates: list[list[str]],
        inverse_gates: list[list[str]],
        with_noise: bool,
    ) -> Circuit:
        """
            Append the historical depolarizing-model LRB body: per-Clifford
            N1 noise, checks then ancilla reset after every forward Clifford,
            and one final check round after the inverse block.
        """
        code_definition = self._require_code_definition()

        if depth == 0:
            # Preserve the historical special-case noise model at depth zero.
            for wire in code_definition.depth_zero_noise_wires:
                circuit.add_gate(
                    "N1",
                    wire,
                    noise_channel=self.with_default_noise_channel,
                    prob=0.0)

        for clifford in clifford_gates:
            for logical_gate in clifford:
                circuit = circuit + code_definition.logical_gate_circuit(
                    logical_gate)

            if with_noise:
                for wire in code_definition.affected_wires(clifford):
                    circuit.add_gate(
                        "N1",
                        wire,
                        noise_channel=self.with_default_noise_channel,
                        prob=0.0)

            for stab_factory, ancilla_wires in (
                    code_definition.stabilizer_check_blocks):
                if with_noise:
                    circuit = circuit + self.inject_stabcheck_noise(
                        stab_factory(),
                        ancilla_wires=set(ancilla_wires),
                        prob=0.0,
                    )
                else:
                    circuit = circuit + stab_factory()

            if code_definition.reset_measurement_wires is not None:
                circuit = circuit + code_definition.reset_measurement_wires()

        inverse_affected_wires: set[int] = set()
        for clifford in inverse_gates:
            for logical_gate in clifford:
                circuit = circuit + code_definition.logical_gate_circuit(
                    logical_gate)
            # Track touched wires so one final inverse-stage noise pass is
            # applied exactly where needed.
            inverse_affected_wires.update(
                code_definition.affected_wires(clifford))

        if with_noise:
            for wire in sorted(inverse_affected_wires):
                circuit.add_gate(
                    "N1",
                    wire,
                    noise_channel=self.with_default_noise_channel,
                    prob=0.0)

        for stab_factory, ancilla_wires in (
                code_definition.stabilizer_check_blocks):
            if with_noise:
                circuit = circuit + self.inject_stabcheck_noise(
                    stab_factory(),
                    ancilla_wires=set(ancilla_wires),
                    prob=0.0,
                )
            else:
                circuit = circuit + stab_factory()
        return circuit

    def generate_lrb_const0_clifford_sequence(
        self,
        depths: list[int],
        with_noise: bool = True,
        clifford_strings: list[str] | None = None,
    ) -> list[Circuit]:
        """
            Build encoded circuits for the special ``const=0`` protocol.

        These circuits use the same logical Clifford and inverse sequence as
        ordinary LRB, but they omit every ancilla stabilizer-check block and
        every ancilla reset. The only postselection data comes from the
        error-free terminal direct X-basis data readout, which evaluates the
        X-stabilizers and logical X observable. Under SI1000/SD6 every
        Clifford is one noisy scheduling segment.

        Args:
            depths (list[int]): Benchmark depth values to instantiate.
            with_noise (bool): Whether to insert gate noise.
            clifford_strings (list[str] | None): Optional pre-sampled maximum
                depth Clifford descriptor sequence shared with ordinary LRB.

        Returns:
            list[Circuit]: One encoded ``const=0`` circuit per requested depth.

        Raises:
            ValueError: If the pre-sampled Clifford sequence is too short.
        """
        _require_sdim_runtime()
        subcircuits: list[Circuit] = []
        code_definition = self._require_code_definition()
        sorted_depths = sorted(depths)
        max_depth = sorted_depths[-1]

        if clifford_strings is None:
            clifford_strings = self.generate_random_clifford_strings(max_depth)
        elif len(clifford_strings) < max_depth:
            raise ValueError(
                "Pre-sampled Clifford sequence is shorter than max depth.")
        full_clifford_gate_list, full_clifford_inverse_list = (
            self._clifford_and_inverse_gate_lists(clifford_strings))

        for depth in sorted_depths:
            circuit = Circuit(
                dimension=code_definition.dimension,
                num_qudits=code_definition.encoded_num_qudits,
            )
            circuit = circuit + code_definition.logical_plus_initial_state()

            if depth == 0:
                clifford_gates: list[list[str]] = []
                inverse_gates: list[list[str]] = []
            else:
                clifford_gates = full_clifford_gate_list[:depth]
                inverse_gates = self._closing_gate_lists(
                    clifford_strings, full_clifford_inverse_list,
                    max_depth, depth)

            if self._circuit_level_noise():
                for clifford in clifford_gates + inverse_gates:
                    circuit = circuit + self._noisy_segment(
                        self._logical_clifford_circuit(clifford), with_noise)
            else:
                for clifford in clifford_gates:
                    for logical_gate in clifford:
                        circuit = circuit + (
                            code_definition.logical_gate_circuit(
                                logical_gate))

                    if with_noise:
                        for wire in code_definition.affected_wires(clifford):
                            circuit.add_gate(
                                "N1",
                                wire,
                                noise_channel=self.with_default_noise_channel,
                                prob=0.0)

                inverse_affected_wires: set[int] = set()
                for clifford in inverse_gates:
                    for logical_gate in clifford:
                        circuit = circuit + (
                            code_definition.logical_gate_circuit(
                                logical_gate))
                    inverse_affected_wires.update(
                        code_definition.affected_wires(clifford))

                if with_noise:
                    for wire in sorted(inverse_affected_wires):
                        circuit.add_gate(
                            "N1",
                            wire,
                            noise_channel=self.with_default_noise_channel,
                            prob=0.0)

            circuit = circuit + code_definition.terminal_const0_measurement()
            self.append_lrb_const0_detector_events(circuit, depth)
            subcircuits.append(circuit)

        return subcircuits

    @staticmethod
    def write_circuit_file(
        circuit: Circuit,
        directory: str,
        output_file: str,
    ) -> str:
        """
            Write ``circuit`` as a CHP file without generator-only metadata.

        SDIM's reader rejects parameters a gate does not define, so the
        ``prob_scale`` kept on noise operations for ``update_noise_param`` is
        left out of the file and restored on the in-memory circuit.

        Args:
            circuit (Circuit): Circuit with materialized noise probabilities.
            directory (str): Output directory.
            output_file (str): Output file name.

        Returns:
            str: Path of the written file.
        """
        stripped = []
        for operation in circuit.operations:
            if operation.params and "prob_scale" in operation.params:
                stripped.append(
                    (operation, operation.params.pop("prob_scale")))
        try:
            return write_circuit(
                circuit=circuit,
                output_file=output_file,
                comment="",
                directory=directory,
            )
        finally:
            for operation, prob_scale in stripped:
                operation.params["prob_scale"] = prob_scale

    def update_noise_param(self, circuit: Circuit, prob: float) -> None:
        """
            Rewrite serialized probabilities for all N1 and N2 operations in
            a circuit. SI1000 gates carry a ``prob_scale`` metadata field so
            their local rate can be materialized from the swept parameter.

        Args:
            circuit (Circuit): Circuit whose noise parameters are updated.
            prob (float): New depolarizing probability value.

        Returns:
            None: Mutates circuit operations in-place.

        Raises:
            ValueError: Propagated if operation parameter structures are
                invalid.
        """
        for operation in circuit.operations:
            if operation.params is None:
                continue

            if operation.gate_name in ("N1", "N2"):
                scale = float(operation.params.get("prob_scale", 1.0))
                local_prob = float(prob) * scale
                if local_prob < 0.0 or local_prob > 1.0:
                    raise ValueError(
                        f"Noise probability {local_prob} is invalid for "
                        f"{operation.gate_name} with scale {scale} and "
                        f"swept p={prob}."
                    )
                operation.params["prob"] = local_prob

            if operation.gate_name == "N2":
                operation.params.pop("prob_dist", None)

    def generate_tests(
        self,
        num_clifford_sequences: int,
        lrb_experiment_folder_path: str,
        rb_experiment_folder_path: str,
        depths: list[int],
        probabilities: list[float],
        lrb_const0_experiment_folder_path: str | None = None,
    ) -> None:
        """
            Generate and export all RB/LRB circuit files for every Clifford
            seed, probability index, and depth in the configured sweep.

        Args:
            num_clifford_sequences (int): Number of independent Clifford seeds.
            lrb_experiment_folder_path (str): Output root for encoded circuits.
            rb_experiment_folder_path (str): Output root for physical circuits.
            depths (list[int]): Benchmark depth values.
            probabilities (list[float]): Noise probabilities to materialize.
            lrb_const0_experiment_folder_path (str | None): Optional output
                root for the special ``const=0`` encoded circuits.

        Returns:
            None: Writes circuits to disk and returns nothing.

        Raises:
            OSError: Propagated if directory creation or file writes fail.
            ValueError: Propagated if circuit-generation helpers fail.
        """
        _require_sdim_runtime()
        for clifford_index in range(num_clifford_sequences):
            lrb_clifford_round_path = os.path.join(lrb_experiment_folder_path,
                                                   str(clifford_index))
            rb_clifford_round_path = os.path.join(rb_experiment_folder_path,
                                                  str(clifford_index))

            clifford_strings = self.generate_random_clifford_strings(
                max(depths))
            lrb_experiments = self.generate_lrb_clifford_sequence(
                depths=depths,
                with_noise=True,
                clifford_strings=clifford_strings)
            lrb_const0_experiments = (
                self.generate_lrb_const0_clifford_sequence(
                    depths=depths,
                    with_noise=True,
                    clifford_strings=clifford_strings)
                if lrb_const0_experiment_folder_path is not None
                else []
            )
            rb_experiments = self.generate_rb_clifford_sequence(
                depths=depths, with_noise=True)

            # For each probability, rewrite noise values and emit CHP files.
            for probability_index, probability in enumerate(probabilities):
                lrb_prob_path = os.path.join(lrb_clifford_round_path,
                                             str(probability_index))
                lrb_const0_prob_path = (
                    os.path.join(
                        lrb_const0_experiment_folder_path,
                        str(clifford_index),
                        str(probability_index),
                    )
                    if lrb_const0_experiment_folder_path is not None
                    else None
                )
                rb_prob_path = os.path.join(rb_clifford_round_path,
                                            str(probability_index))

                os.makedirs(lrb_prob_path, exist_ok=True)
                if lrb_const0_prob_path is not None:
                    os.makedirs(lrb_const0_prob_path, exist_ok=True)
                os.makedirs(rb_prob_path, exist_ok=True)

                for depth_index, circuit in enumerate(lrb_experiments):
                    self.update_noise_param(circuit, probability)
                    self.write_circuit_file(
                        circuit, lrb_prob_path, f"{depth_index}.chp")

                for depth_index, circuit in enumerate(lrb_const0_experiments):
                    self.update_noise_param(circuit, probability)
                    self.write_circuit_file(
                        circuit, lrb_const0_prob_path, f"{depth_index}.chp")

                for depth_index, circuit in enumerate(rb_experiments):
                    self.update_noise_param(circuit, probability)
                    self.write_circuit_file(
                        circuit, rb_prob_path, f"{depth_index}.chp")

            print(f"Generated tests for Clifford round {clifford_index}.")


