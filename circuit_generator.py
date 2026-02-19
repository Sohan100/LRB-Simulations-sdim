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
from typing import Callable, Sequence

from sdim.circuit import Circuit
from sdim.circuit_io import write_circuit


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
        update_noise_param: Rewrite noise probabilities on generated circuits.
        generate_tests: Export all circuits for all Clifford seeds and
            probabilities.
    """
    with_default_noise_channel: str = "d"
    code_definition: LRBCodeDefinition | None = None

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

    def _add_n1_depolarizing(self, circuit: Circuit, wire: int,
                             prob: float) -> None:
        """
            Append a one-qutrit depolarizing gate and normalize serialized
            noise parameters by removing redundant channel metadata.

        Args:
            circuit (Circuit): Circuit receiving the inserted noise operation.
            wire (int): Target wire index for the noise gate.
            prob (float): Depolarizing probability assigned to the gate.

        Returns:
            None: Updates the provided circuit in-place.

        Raises:
            ValueError: Propagated if SDIM rejects invalid gate arguments.
        """
        circuit.add_gate("N1",
                         wire,
                         noise_channel=self.with_default_noise_channel,
                         prob=prob)
        last_gate = circuit.operations[-1]
        if last_gate.params is not None and "channel" in last_gate.params:
            last_gate.params.pop("channel", None)

    def _add_n2_depolarizing(self, circuit: Circuit, q0: int, q1: int,
                             prob: float) -> None:
        """
            Append a two-qutrit depolarizing gate and enforce scalar ``prob``
            serialization for later runtime expansion.

        Args:
            circuit (Circuit): Circuit receiving the inserted noise operation.
            q0 (int): First qudit index for the two-qudit noise gate.
            q1 (int): Second qudit index for the two-qudit noise gate.
            prob (float): Depolarizing probability assigned to the gate.

        Returns:
            None: Updates the provided circuit in-place.

        Raises:
            ValueError: Propagated if SDIM rejects invalid gate arguments.
        """
        circuit.add_gate("N2", q0, q1, prob=prob)
        last_gate = circuit.operations[-1]
        if last_gate.params is not None and "prob_dist" in last_gate.params:
            last_gate.params.pop("prob_dist", None)

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
                circuit.add_gate("N1",
                                 0,
                                 noise_channel=self.with_default_noise_channel,
                                 prob=0.0)
            circuit.add_gate("M", 0)
            subcircuits.append(circuit)

        return subcircuits

    def generate_lrb_clifford_sequence(self,
                                       depths: list[int],
                                       with_noise: bool = True
                                       ) -> list[Circuit]:
        """
            Build folded-code logical RB circuits for all requested depths,
            including logical gate expansion, stabilizer checks, and optional
            physical depolarizing noise insertion.

        Args:
            depths (list[int]): Benchmark depth values to instantiate.
            with_noise (bool): Whether to insert physical noise and noisy
                stabilizer-check blocks.

        Returns:
            list[Circuit]: One encoded logical RB circuit per requested depth.

        Raises:
            ValueError: Propagated if logical-operator expansion or circuit
                assembly receives unsupported gate symbols.
        """
        subcircuits: list[Circuit] = []
        code_definition = self._require_code_definition()
        sorted_depths = sorted(depths)
        max_depth = sorted_depths[-1]

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
                num_qudits=code_definition.encoded_num_qudits,
            )
            circuit = circuit + code_definition.logical_plus_initial_state()

            if depth == 0:
                clifford_gates: list[list[str]] = []
                inverse_gates: list[list[str]] = []
                # Preserve the historical special-case noise model at depth
                # zero.
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
                    circuit = circuit + gate_circuit

                if with_noise:
                    for wire in code_definition.affected_wires(clifford):
                        circuit.add_gate(
                            "N1",
                            wire,
                            noise_channel=self.with_default_noise_channel,
                            prob=0.0)

                if with_noise:
                    for stab_factory, ancilla_wires in (
                            code_definition.stabilizer_check_blocks):
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
                    circuit = (
                        circuit
                        + code_definition.reset_measurement_wires()
                    )

            inverse_affected_wires: set[int] = set()
            for clifford in inverse_gates:
                for logical_gate in clifford:
                    gate_circuit = code_definition.logical_gate_circuit(
                        logical_gate)
                    circuit = circuit + gate_circuit
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

            if with_noise:
                for stab_factory, ancilla_wires in (
                        code_definition.stabilizer_check_blocks):
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
            subcircuits.append(circuit)

        return subcircuits

    def update_noise_param(self, circuit: Circuit, prob: float) -> None:
        """
            Rewrite serialized depolarizing probabilities for all N1 and N2
            operations in a circuit.

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

            if operation.gate_name == "N1":
                operation.params["prob"] = prob

            if operation.gate_name == "N2":
                operation.params["prob"] = prob
                operation.params.pop("prob_dist", None)

    def generate_tests(
        self,
        num_clifford_sequences: int,
        lrb_experiment_folder_path: str,
        rb_experiment_folder_path: str,
        depths: list[int],
        probabilities: list[float],
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

        Returns:
            None: Writes circuits to disk and returns nothing.

        Raises:
            OSError: Propagated if directory creation or file writes fail.
            ValueError: Propagated if circuit-generation helpers fail.
        """
        for clifford_index in range(num_clifford_sequences):
            lrb_clifford_round_path = os.path.join(lrb_experiment_folder_path,
                                                   str(clifford_index))
            rb_clifford_round_path = os.path.join(rb_experiment_folder_path,
                                                  str(clifford_index))

            lrb_experiments = self.generate_lrb_clifford_sequence(
                depths=depths, with_noise=True)
            rb_experiments = self.generate_rb_clifford_sequence(
                depths=depths, with_noise=True)

            # For each probability, rewrite noise values and emit CHP files.
            for probability_index, probability in enumerate(probabilities):
                lrb_prob_path = os.path.join(lrb_clifford_round_path,
                                             str(probability_index))
                rb_prob_path = os.path.join(rb_clifford_round_path,
                                            str(probability_index))

                os.makedirs(lrb_prob_path, exist_ok=True)
                os.makedirs(rb_prob_path, exist_ok=True)

                for depth_index, circuit in enumerate(lrb_experiments):
                    self.update_noise_param(circuit, probability)
                    write_circuit(
                        circuit=circuit,
                        output_file=f"{depth_index}.chp",
                        comment="",
                        directory=lrb_prob_path,
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


