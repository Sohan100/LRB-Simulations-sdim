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
SUPPORTED_NOISE_MODELS = (DEPOLARIZING_NOISE_MODEL, SI1000_NOISE_MODEL)
_NOISE_MODEL_ALIASES = {
    "default": DEPOLARIZING_NOISE_MODEL,
    "legacy": DEPOLARIZING_NOISE_MODEL,
    "legacy_depolarizing": DEPOLARIZING_NOISE_MODEL,
    "depol": DEPOLARIZING_NOISE_MODEL,
    DEPOLARIZING_NOISE_MODEL: DEPOLARIZING_NOISE_MODEL,
    SI1000_NOISE_MODEL: SI1000_NOISE_MODEL,
}

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

    def __post_init__(self) -> None:
        """Normalize noise-model aliases after dataclass initialization."""
        self.noise_model = normalize_noise_model(self.noise_model)

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
                materializing a probability sweep value.

        Returns:
            None: Updates the provided circuit in-place.

        Raises:
            ValueError: Propagated if SDIM rejects invalid gate arguments.
        """
        params: dict[str, float | str] = {
            "noise_channel": noise_channel,
            "prob": prob,
        }
        if prob_scale is not None:
            params["prob_scale"] = prob_scale
        circuit.add_gate("N1", wire, **params)
        last_gate = circuit.operations[-1]
        if last_gate.params is not None and "channel" in last_gate.params:
            last_gate.params.pop("channel", None)

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
        params: dict[str, float] = {"prob": prob}
        if prob_scale is not None:
            params["prob_scale"] = prob_scale
        circuit.add_gate("N2", q0, q1, **params)
        last_gate = circuit.operations[-1]
        if last_gate.params is not None and "prob_dist" in last_gate.params:
            last_gate.params.pop("prob_dist", None)

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

    def _add_si1000_idle_layer_noise(
        self,
        circuit: Circuit,
        active_wires: set[int],
    ) -> None:
        """
            Add SI1000 reset/measurement-layer idle noise.

        Args:
            circuit (Circuit): Circuit receiving idle noise gates.
            active_wires (set[int]): Wires occupied by reset/measurement
                operations in this layer.

        Returns:
            None: Mutates ``circuit``.
        """
        for wire in range(circuit.num_qudits):
            if wire not in active_wires:
                self._add_n1_depolarizing(circuit, wire, 0.0, prob_scale=2.0)

    def apply_si1000_noise_to_circuit(
        self,
        source: Circuit,
        noisy_measurements: bool = True,
    ) -> Circuit:
        """
            Copy a subcircuit and insert generalized SI1000 noise locations.

        Args:
            source (Circuit): Ideal source subcircuit.
            noisy_measurements (bool): Whether measurement gates in the source
                should receive SI1000 measurement noise. Terminal data readout
                passes ``False`` to remain ideal.

        Returns:
            Circuit: Noisy copy of ``source``.

        Raises:
            ValueError: Propagated if SDIM rejects generated operations.
        """
        _require_sdim_runtime()
        output = Circuit(
            dimension=source.dimension,
            num_qudits=source.num_qudits,
        )
        operations = source.operations
        operation_index = 0
        while operation_index < len(operations):
            operation = operations[operation_index]
            gate_name = operation.gate_name

            if noisy_measurements and gate_name in _MEASUREMENT_GATES:
                layer_end = operation_index
                active_wires: set[int] = set()
                while (layer_end < len(operations)
                       and operations[layer_end].gate_name
                       in _MEASUREMENT_GATES):
                    active_wires.add(int(operations[layer_end].qudit_index))
                    layer_end += 1

                self._add_si1000_idle_layer_noise(output, active_wires)
                for measurement in operations[operation_index:layer_end]:
                    self._add_n1_depolarizing(
                        output,
                        measurement.qudit_index,
                        0.0,
                        prob_scale=1.0,
                    )
                    shift_channel = (
                        "p" if measurement.gate_name == "M_X" else "f"
                    )
                    self._add_n1_noise(
                        output,
                        measurement.qudit_index,
                        0.0,
                        noise_channel=shift_channel,
                        prob_scale=5.0,
                    )
                    self._append_operation_copy(output, measurement)

                operation_index = layer_end
                continue

            if gate_name in _RESET_GATES:
                layer_end = operation_index
                active_wires = set()
                while (layer_end < len(operations)
                       and operations[layer_end].gate_name in _RESET_GATES):
                    active_wires.add(int(operations[layer_end].qudit_index))
                    layer_end += 1

                self._add_si1000_idle_layer_noise(output, active_wires)
                for reset in operations[operation_index:layer_end]:
                    self._append_operation_copy(output, reset)
                    self._add_n1_noise(
                        output,
                        reset.qudit_index,
                        0.0,
                        noise_channel="f",
                        prob_scale=2.0,
                    )

                operation_index = layer_end
                continue

            self._append_operation_copy(output, operation)
            if gate_name in _ONE_QUDIT_IDEAL_GATES:
                self._add_n1_depolarizing(
                    output,
                    operation.qudit_index,
                    0.0,
                    prob_scale=0.1,
                )
            elif (gate_name in _TWO_QUDIT_IDEAL_GATES
                  or operation.target_index is not None):
                self._add_n2_depolarizing(
                    output,
                    operation.qudit_index,
                    operation.target_index,
                    0.0,
                    prob_scale=1.0,
                )

            operation_index += 1

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

    def generate_rb_clifford_sequence(self,
                                      depths: list[int],
                                      with_noise: bool = True
                                      ) -> list[Circuit]:
        """
            Build single-qutrit RB circuits for all requested depths using a
            shared max-depth random Clifford sample and matching inverse tails.

        Args:
            depths (list[int]): Benchmark depth values to instantiate.
            with_noise (bool): Whether to insert N1 noise after each Clifford
                block and terminally before measurement.

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
        full_clifford_gate_list = [
            code_definition.clifford_to_gate_sequence(clifford)
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list = [
            code_definition.clifford_inverse_map[clifford]
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list.reverse()

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
                inverse_gates = full_clifford_inverse_list[(max_depth -
                                                            depth):]

            for clifford in clifford_gates:
                for gate in clifford:
                    if with_noise and self.noise_model == SI1000_NOISE_MODEL:
                        gate_circuit = Circuit(
                            dimension=code_definition.dimension,
                            num_qudits=code_definition.physical_num_qudits,
                        )
                        code_definition.apply_physical_gate(
                            gate_circuit, gate)
                        circuit = (
                            circuit
                            + self.apply_si1000_noise_to_circuit(
                                gate_circuit,
                                noisy_measurements=False,
                            )
                        )
                    else:
                        code_definition.apply_physical_gate(circuit, gate)
                if (with_noise
                        and self.noise_model == DEPOLARIZING_NOISE_MODEL):
                    circuit.add_gate(
                        "N1",
                        0,
                        noise_channel=self.with_default_noise_channel,
                        prob=0.0)

            for clifford in inverse_gates:
                for gate in clifford:
                    if with_noise and self.noise_model == SI1000_NOISE_MODEL:
                        gate_circuit = Circuit(
                            dimension=code_definition.dimension,
                            num_qudits=code_definition.physical_num_qudits,
                        )
                        code_definition.apply_physical_gate(
                            gate_circuit, gate)
                        circuit = (
                            circuit
                            + self.apply_si1000_noise_to_circuit(
                                gate_circuit,
                                noisy_measurements=False,
                            )
                        )
                    else:
                        code_definition.apply_physical_gate(circuit, gate)

            if (with_noise
                    and self.noise_model == DEPOLARIZING_NOISE_MODEL):
                circuit.add_gate("N1",
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
            physical depolarizing noise insertion.

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
        full_clifford_gate_list = [
            code_definition.clifford_to_gate_sequence(clifford)
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list = [
            code_definition.clifford_inverse_map[clifford]
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list.reverse()

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
                # Preserve the historical special-case noise model at depth
                # zero.
                if self.noise_model == DEPOLARIZING_NOISE_MODEL:
                    for wire in code_definition.depth_zero_noise_wires:
                        circuit.add_gate(
                            "N1",
                            wire,
                            noise_channel=self.with_default_noise_channel,
                            prob=0.0)
            else:
                clifford_gates = full_clifford_gate_list[:depth]
                inverse_gates = full_clifford_inverse_list[(max_depth -
                                                            depth):]

            for clifford in clifford_gates:
                for logical_gate in clifford:
                    gate_circuit = code_definition.logical_gate_circuit(
                        logical_gate)
                    if (with_noise
                            and self.noise_model == SI1000_NOISE_MODEL):
                        gate_circuit = self.apply_si1000_noise_to_circuit(
                            gate_circuit,
                            noisy_measurements=False,
                        )
                    circuit = circuit + gate_circuit

                if (with_noise
                        and self.noise_model == DEPOLARIZING_NOISE_MODEL):
                    for wire in code_definition.affected_wires(clifford):
                        circuit.add_gate(
                            "N1",
                            wire,
                            noise_channel=self.with_default_noise_channel,
                            prob=0.0)

                if with_noise:
                    for stab_factory, ancilla_wires in (
                            code_definition.stabilizer_check_blocks):
                        if self.noise_model == SI1000_NOISE_MODEL:
                            circuit = (
                                circuit
                                + self.apply_si1000_noise_to_circuit(
                                    stab_factory(),
                                    noisy_measurements=True,
                                )
                            )
                        else:
                            circuit = circuit + self.inject_stabcheck_noise(
                                stab_factory(),
                                ancilla_wires=set(ancilla_wires),
                                prob=0.0,
                            )
                else:
                    for stab_factory, _ in (
                            code_definition.stabilizer_check_blocks):
                        circuit = circuit + stab_factory()

                if code_definition.reset_measurement_wires is not None:
                    reset_circuit = code_definition.reset_measurement_wires()
                    if (with_noise
                            and self.noise_model == SI1000_NOISE_MODEL):
                        reset_circuit = self.apply_si1000_noise_to_circuit(
                            reset_circuit,
                            noisy_measurements=False,
                        )
                    circuit = circuit + reset_circuit

            inverse_affected_wires: set[int] = set()
            for clifford in inverse_gates:
                for logical_gate in clifford:
                    gate_circuit = code_definition.logical_gate_circuit(
                        logical_gate)
                    if (with_noise
                            and self.noise_model == SI1000_NOISE_MODEL):
                        gate_circuit = self.apply_si1000_noise_to_circuit(
                            gate_circuit,
                            noisy_measurements=False,
                        )
                    circuit = circuit + gate_circuit
                # Track touched wires so one final inverse-stage noise pass is
                # applied exactly where needed.
                inverse_affected_wires.update(
                    code_definition.affected_wires(clifford))

            if (with_noise
                    and self.noise_model == DEPOLARIZING_NOISE_MODEL):
                for wire in sorted(inverse_affected_wires):
                    circuit.add_gate(
                        "N1",
                        wire,
                        noise_channel=self.with_default_noise_channel,
                        prob=0.0)

            if with_noise:
                for stab_factory, ancilla_wires in (
                        code_definition.stabilizer_check_blocks):
                    if self.noise_model == SI1000_NOISE_MODEL:
                        circuit = (
                            circuit
                            + self.apply_si1000_noise_to_circuit(
                                stab_factory(),
                                noisy_measurements=True,
                            )
                        )
                    else:
                        circuit = circuit + self.inject_stabcheck_noise(
                            stab_factory(),
                            ancilla_wires=set(ancilla_wires),
                            prob=0.0,
                        )
            else:
                for stab_factory, _ in (
                        code_definition.stabilizer_check_blocks):
                    circuit = circuit + stab_factory()

            circuit = circuit + code_definition.terminal_measurement()
            self.append_lrb_detector_events(circuit, depth)
            subcircuits.append(circuit)

        return subcircuits

    def generate_lrb_const0_clifford_sequence(
        self,
        depths: list[int],
        with_noise: bool = True,
        clifford_strings: list[str] | None = None,
    ) -> list[Circuit]:
        """
            Build encoded LRB circuits for the special ``const=0`` protocol.

        The non-fault-tolerant state preparation is ideal. No intermediate or
        ancilla-based terminal stabilizer checks are inserted. Instead, after
        the noisy logical Clifford/inverse sequence, the circuit performs an
        error-free direct X-basis data readout containing the X stabilizers and
        logical-X observable.

        Args:
            depths (list[int]): Benchmark depth values to instantiate.
            with_noise (bool): Whether logical Clifford gates receive physical
                noise according to the selected model.
            clifford_strings (list[str] | None): Optional pre-sampled maximum
                depth Clifford descriptor sequence.

        Returns:
            list[Circuit]: One encoded const=0 circuit per requested depth.

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
        full_clifford_gate_list = [
            code_definition.clifford_to_gate_sequence(clifford)
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list = [
            code_definition.clifford_inverse_map[clifford]
            for clifford in clifford_strings
        ]
        full_clifford_inverse_list.reverse()

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
                inverse_gates = full_clifford_inverse_list[(max_depth -
                                                            depth):]

            for clifford in clifford_gates:
                for logical_gate in clifford:
                    gate_circuit = code_definition.logical_gate_circuit(
                        logical_gate)
                    if (with_noise
                            and self.noise_model == SI1000_NOISE_MODEL):
                        gate_circuit = self.apply_si1000_noise_to_circuit(
                            gate_circuit,
                            noisy_measurements=False,
                        )
                    circuit = circuit + gate_circuit

                if (with_noise
                        and self.noise_model == DEPOLARIZING_NOISE_MODEL):
                    for wire in code_definition.affected_wires(clifford):
                        circuit.add_gate(
                            "N1",
                            wire,
                            noise_channel=self.with_default_noise_channel,
                            prob=0.0)

            inverse_affected_wires: set[int] = set()
            for clifford in inverse_gates:
                for logical_gate in clifford:
                    gate_circuit = code_definition.logical_gate_circuit(
                        logical_gate)
                    if (with_noise
                            and self.noise_model == SI1000_NOISE_MODEL):
                        gate_circuit = self.apply_si1000_noise_to_circuit(
                            gate_circuit,
                            noisy_measurements=False,
                        )
                    circuit = circuit + gate_circuit
                inverse_affected_wires.update(
                    code_definition.affected_wires(clifford))

            if (with_noise
                    and self.noise_model == DEPOLARIZING_NOISE_MODEL):
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
                    write_circuit(
                        circuit=circuit,
                        output_file=f"{depth_index}.chp",
                        comment="",
                        directory=lrb_prob_path,
                    )

                for depth_index, circuit in enumerate(lrb_const0_experiments):
                    self.update_noise_param(circuit, probability)
                    write_circuit(
                        circuit=circuit,
                        output_file=f"{depth_index}.chp",
                        comment="",
                        directory=lrb_const0_prob_path,
                    )

                for depth_index, circuit in enumerate(rb_experiments):
                    self.update_noise_param(circuit, probability)
                    write_circuit(
                        circuit=circuit,
                        output_file=f"{depth_index}.chp",
                        comment="",
                        directory=rb_prob_path,
                    )

            print(f"Generated tests for Clifford round {clifford_index}.")


