"""Code-specific LRB profiles wired into generic setup and runtime engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import code_definitions as defns
from circuit_generator import LRBCodeDefinition, LRBCodeDefinitionFactory

DEFAULT_CODE_NAME = "folded_qutrit"


@dataclass(frozen=True)
class _UnpackSpec:
    """
    UnpackSpec.
    
    Attributes:
        None: This class stores no persistent attributes unless
            documented by concrete fields.
    
    Methods:
        See class method definitions for the supported API.
    """
    stabilizer_wires: tuple[int, ...]
    logical_measurement_wires: tuple[int, ...]
    logical_outcome_fn: Callable[[list[int], int], int]
    check_stride: int = 2
    check_round_start: int = 0
    check_rounds_offset: int = 1
    logical_measurement_round: int = 0


@dataclass(frozen=True)
class CodeSimulationProfile:
    """
    Runtime profile for one logical code configuration.

    Attributes:
        code_name (str): Stable identifier used to select the profile.
        code_definition (LRBCodeDefinition): Circuit-construction callbacks
            consumed by the generic generator.
        logical_dimension (int): Number of logical outcomes for statistics.
        unpack_func (Callable): Measurement unpack callback for this code.

    Methods:
        This dataclass is declarative and defines no custom methods.
    """

    code_name: str
    code_definition: LRBCodeDefinition
    logical_dimension: int
    unpack_func: Callable


class CodeSimulationProfileRegistry:
    """
    Registry resolving code-specific setup/runtime profiles by name.

    Attributes:
        _CODE_DEFINITION_SPECS (dict): Per-code construction config consumed by
            ``_build_code_definition``.
        _UNPACK_SPECS (dict): Per-code unpack specs for runtime decoding.
        _LOGICAL_DIMENSIONS (dict): Per-code logical dimensions for statistics.

    Methods:
        _build_code_definition(code_name): Build one ``LRBCodeDefinition``.
        _make_unpacker(spec): Build an unpack function for one unpack spec.
        resolve_code_profile(code_name): Resolve the complete runtime profile.
    """

    # Per-code construction data used to build LRBCodeDefinition objects.
    _CODE_DEFINITION_SPECS = {
        "folded_qutrit": {
            "dimension": 3,
            "physical_num_qudits": 1,
            "encoded_num_qudits": 9,
            "clifford_strings": (
                defns.single_qutrit_cliffords_exhaustive_strings
            ),
            "clifford_to_gate_sequence": (
                defns.QutritCliffordLibrary.qutrit_gate_seq_list
            ),
            "clifford_inverse_map": (
                defns.QutritCliffordLibrary.build_inverse_lookup()
            ),
            "apply_physical_gate": (
                LRBCodeDefinitionFactory.apply_single_qudit_gate_to_wire_zero
            ),
            "logical_plus_initial_state": (
                defns.FoldedQutritDetectionCode.logical_plus_initial_state
            ),
            "logical_gate_circuit": (
                defns.FoldedQutritDetectionCode.
                single_qudit_logical_gate_circuit
            ),
            "affected_wires": defns.FoldedQutritDetectionCode.affected_wires,
            "stabilizer_check_blocks": (
                (
                    defns.FoldedQutritDetectionCode.
                    z_ALT_stabilizer_measurement_ancillae_circuit,
                    {5, 6},
                ),
                (
                    defns.FoldedQutritDetectionCode.
                    x_stabilizer_measurement_ancillae_circuit,
                    {7, 8},
                ),
            ),
            "reset_measurement_wires": (
                defns.FoldedQutritDetectionCode.reset_measurement_wires
            ),
            "terminal_measurement": (
                defns.FoldedQutritDetectionCode.
                terminal_logical_x_measurement_circuit
            ),
            "depth_zero_noise_wires": (0, 1, 2, 3, 4),
        },
        "qgrm_3_1_2": {
            "dimension": 3,
            "physical_num_qudits": 1,
            "encoded_num_qudits": 5,
            "clifford_strings": (
                defns.single_qutrit_cliffords_exhaustive_strings
            ),
            "clifford_to_gate_sequence": (
                defns.QutritCliffordLibrary.qutrit_gate_seq_list
            ),
            "clifford_inverse_map": (
                defns.QutritCliffordLibrary.build_inverse_lookup()
            ),
            "apply_physical_gate": (
                LRBCodeDefinitionFactory.apply_single_qudit_gate_to_wire_zero
            ),
            "logical_plus_initial_state": (
                defns.QGRMThreeQutritDetectionCode.logical_plus_initial_state
            ),
            "logical_gate_circuit": (
                defns.QGRMThreeQutritDetectionCode.
                single_qudit_logical_gate_circuit
            ),
            "affected_wires": (
                defns.QGRMThreeQutritDetectionCode.affected_wires
            ),
            "stabilizer_check_blocks": (
                (
                    defns.QGRMThreeQutritDetectionCode.
                    z_stabilizer_measurement_ancillae_circuit,
                    {3},
                ),
                (
                    defns.QGRMThreeQutritDetectionCode.
                    x_stabilizer_measurement_ancillae_circuit,
                    {4},
                ),
            ),
            "reset_measurement_wires": (
                defns.QGRMThreeQutritDetectionCode.reset_measurement_wires
            ),
            "terminal_measurement": (
                defns.QGRMThreeQutritDetectionCode.
                terminal_logical_x_measurement_circuit
            ),
            "depth_zero_noise_wires": (0, 1, 2),
        },
    }

    # Per-code unpack layout for decoding stabilizer and logical readout wires.
    _UNPACK_SPECS = {
        "folded_qutrit": _UnpackSpec(
            stabilizer_wires=(5, 6, 7, 8),
            logical_measurement_wires=(0, 3),
            logical_outcome_fn=(
                lambda measurements, _: (measurements[0] + measurements[1]) % 3
            ),
        ),
        "qgrm_3_1_2": _UnpackSpec(
            stabilizer_wires=(3, 4),
            logical_measurement_wires=(0, 1),
            logical_outcome_fn=(
                lambda measurements, _: (measurements[0] - measurements[1]) % 3
            ),
        ),
    }

    # Logical alphabet size used when extracting logical statistics.
    _LOGICAL_DIMENSIONS = {
        "folded_qutrit": 3,
        "qgrm_3_1_2": 3,
    }

    @classmethod
    def _build_code_definition(cls, code_name: str) -> LRBCodeDefinition:
        """
        Build code definition.
        
        Args:
            code_name (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        # Materialize a concrete code definition from the registry record.
        spec = cls._CODE_DEFINITION_SPECS[code_name]
        return LRBCodeDefinition(
            dimension=spec["dimension"],
            physical_num_qudits=spec["physical_num_qudits"],
            encoded_num_qudits=spec["encoded_num_qudits"],
            clifford_strings=spec["clifford_strings"],
            clifford_to_gate_sequence=spec["clifford_to_gate_sequence"],
            clifford_inverse_map=spec["clifford_inverse_map"],
            apply_physical_gate=spec["apply_physical_gate"],
            logical_plus_initial_state=spec["logical_plus_initial_state"],
            logical_gate_circuit=spec["logical_gate_circuit"],
            affected_wires=spec["affected_wires"],
            stabilizer_check_blocks=spec["stabilizer_check_blocks"],
            reset_measurement_wires=spec["reset_measurement_wires"],
            terminal_measurement=spec["terminal_measurement"],
            depth_zero_noise_wires=spec["depth_zero_noise_wires"],
        )

    @staticmethod
    def _make_unpacker(spec: _UnpackSpec) -> Callable:
        """
        Make unpacker.
        
        Args:
            spec (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        def _unpack(results, depth, shots):
            """
            Unpack.
            
            Args:
                results (Any): Input argument.
                depth (Any): Input argument.
                shots (Any): Input argument.
            
            Returns:
                object: Method output.
            
            Raises:
                ValueError: If supplied arguments violate this method's input
                    assumptions.
            """
            # Import locally to avoid circular import issues at module load.
            from lrb_simulation import LRBSimulationPipeline

            return LRBSimulationPipeline.unpack_measurement_results_from_spec(
                results, depth, shots, spec
            )

        return _unpack

    @classmethod
    def resolve_code_profile(cls, code_name: str) -> CodeSimulationProfile:
        """
        Resolve code profile.
        
        Args:
            code_name (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        if code_name not in cls._CODE_DEFINITION_SPECS:
            raise ValueError(f"Unsupported code_name '{code_name}'.")
        # Bundle setup-time and runtime hooks into one resolved profile.
        return CodeSimulationProfile(
            code_name=code_name,
            code_definition=cls._build_code_definition(code_name),
            logical_dimension=cls._LOGICAL_DIMENSIONS[code_name],
            unpack_func=cls._make_unpacker(cls._UNPACK_SPECS[code_name]),
        )
