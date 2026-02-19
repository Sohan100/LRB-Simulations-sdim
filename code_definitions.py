"""Definitions for qutrit Clifford operations and qutrit detection codes.

This module provides primary class abstractions used by the LRB workflow:

1. ``QutritCliffordLibrary``
   Encapsulates the single-qutrit Clifford catalog and deterministic
   helpers for converting symbolic strings (for example ``"HP"``) into
   executable gate sequences and inverse lookup tables.

2. ``FoldedQutritDetectionCode``
   Encodes circuit templates for the ``[[5,1,2]]_3`` folded surface
   error-detection code used in this repository. The class exposes methods
   that build full ``sdim`` ``Circuit`` objects for initialization, logical
   operations, stabilizer measurements, and measurement-wire reset steps.

3. ``QGRMThreeQutritDetectionCode``
   Encodes the circuit templates for the [[3,1,2]]_3 QGRM qutrit detection
   code, including initialization, logical operators, and stabilizer checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from sdim.circuit import Circuit

# ---------------------------------------------------------------------------
# Canonical gate inverses for all gate symbols used in this project.
# ---------------------------------------------------------------------------
SINGLE_QUTRIT_INVERSES: dict[str, str] = {
    "H": "H_INV",
    "H_INV": "H",
    "P": "P_INV",
    "P_INV": "P",
    "Z": "Z_INV",
    "Z_INV": "Z",
    "X": "X_INV",
    "X_INV": "X",
    "CNOT": "CNOT_INV",
    "CNOT_INV": "CNOT",
    "CZ": "CZ_INV",
    "CZ_INV": "CZ",
    "SWAP": "SWAP",
    "I": "I",
}

single_qutrit_cliffords_exhaustive_with_S_strings = [
    "I",
    "H",
    "S",
    "HH",
    "HS",
    "SH",
    "SS",
    "HHH",
    "HHS",
    "HSH",
    "HSS",
    "SHS",
    "SSH",
    "HHHS",
    "HHSH",
    "HHSS",
    "HSHS",
    "SHSH",
    "SHSS",
    "SSHS",
    "HHHSH",
    "HHSHS",
    "HSHSS",
    "SSHSH",
    "X",
    "XH",
    "XS",
    "XHH",
    "XHS",
    "XSH",
    "XSS",
    "XHHH",
    "XHHS",
    "XHSH",
    "XHSS",
    "XSHS",
    "XSSH",
    "XHHHS",
    "XHHSH",
    "XHHSS",
    "XHSHS",
    "XSHSH",
    "XSHSS",
    "XSSHS",
    "XHHHSH",
    "XHHSHS",
    "XHSHSS",
    "XSSHSH",
    "Z",
    "ZH",
    "ZS",
    "ZHH",
    "ZHS",
    "ZSH",
    "ZSS",
    "ZHHH",
    "ZHHS",
    "ZHSH",
    "ZHSS",
    "ZSHS",
    "ZSSH",
    "ZHHHS",
    "ZHHSH",
    "ZHHSS",
    "ZHSHS",
    "ZSHSH",
    "ZSHSS",
    "ZSSHS",
    "ZHHHSH",
    "ZHHSHS",
    "ZHSHSS",
    "ZSSHSH",
    "XZ",
    "XZH",
    "XZS",
    "XZHH",
    "XZHS",
    "XZSH",
    "XZSS",
    "XZHHH",
    "XZHHS",
    "XZHSH",
    "XZHS",
    "XZSHS",
    "XZSH",
    "XZHHHS",
    "XZHHSH",
    "XZHHSS",
    "XZHSHS",
    "XZSHSH",
    "XZSHSS",
    "XZSSHS",
    "XZHHHSH",
    "XZHHSHS",
    "XZHSHSS",
    "XZSSHSH",
    "XX",
    "XXH",
    "XXS",
    "XXHH",
    "XXHS",
    "XXSH",
    "XXSS",
    "XXHHH",
    "XXHHS",
    "XXHSH",
    "XXHSS",
    "XXSHS",
    "XXSSH",
    "XXHHHS",
    "XXHHSH",
    "XXHHSS",
    "XXHSHS",
    "XXSHSH",
    "XXSHSS",
    "XXSSHS",
    "XXHHHSH",
    "XXHHSHS",
    "XXHSHSS",
    "XXSSHSH",
    "ZZ",
    "ZZH",
    "ZZS",
    "ZZHH",
    "ZZHS",
    "ZZSH",
    "ZZSS",
    "ZZHHH",
    "ZZHHS",
    "ZZHSH",
    "ZZHSS",
    "ZZSHS",
    "ZZSSH",
    "ZZHHHS",
    "ZZHHSH",
    "ZZHHSS",
    "ZZHSHS",
    "ZZSHSH",
    "ZZSHSS",
    "ZZSSHS",
    "ZZHHHSH",
    "ZZHHSHS",
    "ZZHSHSS",
    "ZZSSHSH",
    "XXZZ",
    "XXZH",
    "XXZS",
    "XXZHH",
    "XXZHS",
    "XXZSH",
    "XXZSS",
    "XXZHHH",
    "XXZHHS",
    "XXZHSH",
    "XXZHS",
    "XXZSHS",
    "XXZS",
    "XXZHHHS",
    "XXZHHSH",
    "XXZHHSS",
    "XXZHSHS",
    "XXZSHSH",
    "XXZSHSS",
    "XXZSSHS",
    "XXZHHHSH",
    "XXZHHSHS",
    "XXZHSHSS",
    "XXZSSHSH",
    "ZZXX",
    "ZZXH",
    "ZZXS",
    "ZZXHH",
    "ZZXHS",
    "ZZXSH",
    "ZZXSS",
    "ZZXHHH",
    "ZZXHHS",
    "ZZXHSH",
    "ZZXHS",
    "ZZXSHS",
    "ZZXS",
    "ZZXHHHS",
    "ZZXHHSH",
    "ZZXHHSS",
    "ZZXHSHS",
    "ZZXSHSH",
    "ZZXSHSS",
    "ZZXSSHS",
    "ZZXHHHSH",
    "ZZXHHSHS",
    "ZZXHSHSS",
    "ZZXSSHSH",
    "XXZZ",
    "XXZZH",
    "XXZZS",
    "XXZZHH",
    "XXZZHS",
    "XXZZSH",
    "XXZZSS",
    "XXZZHHH",
    "XXZZHHS",
    "XXZZHSH",
    "XXZZHS",
    "XXZZSHS",
    "XXZZS",
    "XXZZHHHS",
    "XXZZHHSH",
    "XXZZHHSS",
    "XXZZHSHS",
    "XXZZSHSH",
    "XXZZSHSS",
    "XXZZSSHS",
    "XXZZHHHSH",
    "XXZZHHSHS",
    "XXZZHSHSS",
    "XXZZSSHSH",
]

# Internal catalog in terms of the P gate symbol used by SDIM.
single_qutrit_cliffords_exhaustive_strings = [
    cliff.replace("S", "P")
    for cliff in single_qutrit_cliffords_exhaustive_with_S_strings
]


class QutritCliffordLibrary:
    """
    Helper for qutrit Clifford parsing and inverse lookup generation.

    The class stores no mutable state and exposes parsing utilities that map
    string descriptors to gate sequences and inverse tables used by folded-code
    circuit generators.

    Attributes:
        None: This class is a stateless namespace of utility methods.
    Methods:
        qutrit_gate_seq_list(cliff): Convert one descriptor string into an
            execution-order gate list.
        build_inverse_lookup(): Build descriptor-to-inverse mappings for the
            exhaustive Clifford catalog.
    """
    @staticmethod
    def qutrit_gate_seq_list(cliff: str) -> list[str]:
        """
        Convert a Clifford descriptor string into execution-order gate symbols.

        Args:
            cliff (str): Symbolic descriptor where rightmost symbols act first.

        Returns:
            list[str]: Execution-order gate symbols.

        Raises:
            ValueError: If `cliff` cannot be interpreted as a gate sequence.
        """
        # Descriptor strings are written left-to-right but applied right-to-
        # left, so reverse once here.
        return list(reversed([gate for gate in cliff]))

    @classmethod
    def build_inverse_lookup(cls) -> dict[str, list[str]]:
        """
        Build inverse mappings for all descriptors in the Clifford catalog.

        Args:
            None: `build_inverse_lookup` relies on module-level lookup tables.

        Returns:
            dict[str, list[str]]: Descriptor-to-inverse mapping.

        Raises:
            KeyError: If a gate symbol is missing from
                `SINGLE_QUTRIT_INVERSES`.
        """
        inv_dict: dict[str, list[str]] = {}
        for cliff in single_qutrit_cliffords_exhaustive_strings:
            inv_seq: list[str] = []
            for gate in cliff:
                inv_seq.append(SINGLE_QUTRIT_INVERSES[gate])
            # Keep mapping aligned with raw descriptor key for fast lookup.
            inv_dict[cliff] = inv_seq
        return inv_dict


@dataclass(frozen=True)
class FoldedQutritDetectionCode:
    """
    Circuit builder for ``[[5,1,2]]_3`` folded surface-code templates.

    The class defines canonical code parameters and methods that build SDIM
    circuits for initialization, logical operators, stabilizer measurements,
    and reset routines used in encoded benchmarking workflows.

    Attributes:
        dimension (int): Local qutrit dimension used by constructed circuits.
        num_qudits (int): Total wire count for the folded code template.
    Methods:
        logical_plus_initial_state(): Build logical `|+>_L` preparation.
        single_qudit_logical_gate_circuit(operator): Compile one logical
            operator token.
        construct_elementary_single_qudit_logical_gate_circuit(
        gates, controls, targets): Assemble circuits from parsed placement
            lists.
        elementary_single_qudit_logical_operator_gate_sequence(operator):
            Parse logical operator tokens into elementary gate placements.
        affected_wires_per_operator(operator): Report touched data wires for
            one operator token.
        z_stabilizer_measurement_ancillae_circuit(): Build basis-change Z
            checks.
        z_ALT_stabilizer_measurement_ancillae_circuit(): Build CNOT-based Z
            checks.
        x_stabilizer_measurement_ancillae_circuit(): Build X checks.
        reset_measurement_wires(): Build ancilla reset block.
        affected_wires(cliff): Report all touched data wires for a sequence.
        codespace_check(z_measurements, x_measurements): Evaluate full
            stabilizer constraints from measurements.
        codespace_X_stabilizer_check(x_measurements): Evaluate only X checks.
    """

    dimension: int = 3
    num_qudits: int = 9

    @classmethod
    def logical_plus_initial_state(cls) -> Circuit:
        """
        Build folded-code initialization for logical `|+>_L`.

        Args:
            None: `logical_plus_initial_state` uses class-level code constants.

        Returns:
            Circuit: Initialization subcircuit on the nine-wire code block.

        Raises:
            ValueError: If SDIM gate insertion fails during circuit assembly.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("H_INV", [2, 3, 4])
        c.add_gate("CNOT_INV", 2, 1)
        c.add_gate("CNOT", 1, 0)
        c.add_gate("CNOT", 3, 0)
        c.add_gate("CNOT", 4, 2)
        return c

    @classmethod
    def single_qudit_logical_gate_circuit(cls, operator: str) -> Circuit:
        """
        Compile one logical operator token into a folded-code subcircuit.

        Args:
            operator (str): Logical operator token (for example `X` or
                `H_INV`).

        Returns:
            Circuit: Physical subcircuit implementing the requested operator.

        Raises:
            ValueError: Raised if the operator token is unsupported.
        """
        gates, controls, targets = (
            cls.elementary_single_qudit_logical_operator_gate_sequence(
                operator
            )
        )
        return cls.construct_elementary_single_qudit_logical_gate_circuit(
            gates, controls, targets)

    @classmethod
    def construct_elementary_single_qudit_logical_gate_circuit(
        cls,
        gates: Sequence[str],
        controls: Sequence[int | list[int]],
        targets: Sequence[int | None],
    ) -> Circuit:
        """
        Build a folded-code subcircuit from aligned gate placement lists.

        Args:
            gates (Sequence[str]): Ordered gate names.
            controls (Sequence[int | list[int]]): Primary wire placements.
            targets (Sequence[int | None]): Optional secondary wire placements.

        Returns:
            Circuit: Constructed subcircuit matching the provided placements.

        Raises:
            ValueError: Raised when two-qudit gates receive non-scalar
                controls.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        for index, gate_name in enumerate(gates):
            if targets[index] is None:
                c.add_gate(gate_name, controls[index])
            else:
                if not isinstance(controls[index], int):
                    raise ValueError(
                        "Two-qudit gates must have a single control index.")
                c.add_gate(gate_name, controls[index], targets[index])
        return c

    @classmethod
    def elementary_single_qudit_logical_operator_gate_sequence(
        cls,
        operator: str,
    ) -> tuple[list[str], list[int | list[int]], list[int | None]]:
        """
        Parse one logical operator token into elementary gate placements.

        Args:
            operator (str): Logical operator token to parse.

        Returns:
            tuple[list[str], list[int | list[int]], list[int | None]]: Parsed
            gate names and placements.

        Raises:
            ValueError: Raised if the operator token is unsupported.
        """
        gates: list[str] = []
        controls: list[int | list[int]] = []
        targets: list[int | None] = []

        op_type = operator[0]

        # Map each logical token to its physical folded-code implementation.
        if op_type == "I":
            gates = ["I"]
            controls = [0]
            targets = [None]
        elif op_type == "X":
            gates = ["X"]
            controls = [[0, 3]]
            targets = [None]
        elif op_type == "Z":
            gates = ["Z"]
            controls = [[0, 2]]
            targets = [None]
        elif op_type == "H":
            gates = ["SWAP", "H", "H_INV"]
            controls = [2, [0, 2, 3, 4], 1]
            targets = [3, None, None]
        elif op_type == "P":
            gates = ["P", "P_INV", "CZ"]
            controls = [[0, 4], 1, 2]
            targets = [None, None, 3]
        else:
            raise ValueError(
                f"Gate '{operator}' is not supported by this code definition.")

        if len(operator) > 1:
            # "_INV" operators are represented by reversing and inverting the
            # elementary gate sequence.
            gates.reverse()
            gates = [SINGLE_QUTRIT_INVERSES[gate] for gate in gates]
            controls.reverse()
            targets.reverse()

        return gates, controls, targets

    @classmethod
    def affected_wires_per_operator(cls, operator: str) -> list[int]:
        """
        Return data-wire indices touched by one logical operator token.

        Args:
            operator (str): Logical operator token.

        Returns:
            list[int]: Data-wire indices touched by that operator.

        Raises:
            ValueError: Raised if the operator token is unsupported.
        """
        if operator in ("H", "H_INV", "P", "P_INV"):
            return [0, 1, 2, 3, 4]
        if operator in ("X", "X_INV"):
            return [0, 3]
        if operator in ("Z", "Z_INV"):
            return [0, 2]
        if operator == "I":
            return []
        raise ValueError(
            f"Gate '{operator}' is not supported by this code definition.")

    @classmethod
    def z_stabilizer_measurement_ancillae_circuit(cls) -> Circuit:
        """
        Build the Z-stabilizer ancilla measurement subcircuit.

        Args:
            None: `z_stabilizer_measurement_ancillae_circuit` uses fixed code
                wiring.

        Returns:
            Circuit: Z-stabilizer ancilla measurement subcircuit.

        Raises:
            ValueError: If SDIM gate insertion fails during circuit assembly.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        c.add_gate("H", 5)
        c.add_gate("CZ", 5, 0)
        c.add_gate("CZ_INV", 5, 1)
        c.add_gate("CZ_INV", 5, 3)
        c.add_gate("H_INV", 5)
        c.add_gate("M", 5)

        c.add_gate("H", 6)
        c.add_gate("CZ", 6, 1)
        c.add_gate("CZ", 6, 2)
        c.add_gate("CZ_INV", 6, 4)
        c.add_gate("H_INV", 6)
        c.add_gate("M", 6)

        return c

    @classmethod
    def z_ALT_stabilizer_measurement_ancillae_circuit(cls) -> Circuit:
        """
        Build the alternate Z-stabilizer block using CNOT couplings.

        Args:
            None: `z_ALT_stabilizer_measurement_ancillae_circuit` uses fixed
                code wiring.

        Returns:
            Circuit: Alternative Z-stabilizer ancilla measurement subcircuit.

        Raises:
            ValueError: If SDIM gate insertion fails during circuit assembly.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        c.add_gate("CNOT", 0, 5)
        c.add_gate("CNOT_INV", 1, 5)
        c.add_gate("CNOT_INV", 3, 5)
        c.add_gate("M", 5)

        c.add_gate("CNOT", 1, 6)
        c.add_gate("CNOT", 2, 6)
        c.add_gate("CNOT_INV", 4, 6)
        c.add_gate("M", 6)

        return c

    @classmethod
    def x_stabilizer_measurement_ancillae_circuit(cls) -> Circuit:
        """
        Build the X-stabilizer ancilla measurement subcircuit.

        Args:
            None: `x_stabilizer_measurement_ancillae_circuit` uses fixed code
                wiring.

        Returns:
            Circuit: X-stabilizer ancilla measurement subcircuit.

        Raises:
            ValueError: If SDIM gate insertion fails during circuit assembly.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        c.add_gate("H", 7)
        c.add_gate("CNOT", 7, 0)
        c.add_gate("CNOT", 7, 1)
        c.add_gate("CNOT_INV", 7, 2)
        c.add_gate("H_INV", 7)
        c.add_gate("M", 7)

        c.add_gate("H", 8)
        c.add_gate("CNOT", 8, 3)
        c.add_gate("CNOT_INV", 8, 1)
        c.add_gate("CNOT_INV", 8, 4)
        c.add_gate("H_INV", 8)
        c.add_gate("M", 8)

        return c

    @classmethod
    def reset_measurement_wires(cls) -> Circuit:
        """
        Build the ancilla reset block for all measurement wires.

        Args:
            None: `reset_measurement_wires` uses fixed ancilla-wire indices.

        Returns:
            Circuit: Reset subcircuit on ancilla wires 5 through 8.

        Raises:
            ValueError: If SDIM gate insertion fails during circuit assembly.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("RESET", 5)
        c.add_gate("RESET", 6)
        c.add_gate("RESET", 7)
        c.add_gate("RESET", 8)
        return c

    @classmethod
    def terminal_logical_x_measurement_circuit(cls) -> Circuit:
        """
        Build the folded-code terminal logical-X measurement block.

        Args:
            None: Uses fixed folded-code readout wiring.

        Returns:
            Circuit: Two-wire terminal X-basis measurement subcircuit.

        Raises:
            ValueError: If SDIM gate insertion fails during circuit assembly.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("H_INV", [0, 3])
        c.add_gate("M", [0, 3])
        return c

    @classmethod
    def affected_wires(cls, cliff: Iterable[str]) -> list[int]:
        """
        Return all unique data wires touched by a logical gate sequence.

        Args:
            cliff (Iterable[str]): Logical operator token sequence.

        Returns:
            list[int]: Sorted unique data-wire indices touched by the sequence.

        Raises:
            ValueError: Propagated if any operator token is unsupported.
        """
        wires: set[int] = set()
        for gate in cliff:
            wires.update(cls.affected_wires_per_operator(gate))
        return sorted(wires)

    @staticmethod
    def codespace_check(z_measurements: Sequence[int],
                        x_measurements: Sequence[int]) -> bool:
        """
        Evaluate full folded-code stabilizer constraints from measurements.

        Args:
            z_measurements (Sequence[int]): Z-basis measurement values.
            x_measurements (Sequence[int]): X-basis measurement values.

        Returns:
            bool: ``True`` when all stabilizer checks pass modulo 3.

        Raises:
            IndexError: If input arrays do not contain required data-wire
                entries.
        """
        stabilizer_measurements = [
            (z_measurements[0] - z_measurements[1] - z_measurements[3]) % 3,
            (z_measurements[1] + z_measurements[2] - z_measurements[4]) % 3,
            (x_measurements[0] + x_measurements[1] - x_measurements[2]) % 3,
            (x_measurements[3] - x_measurements[1] - x_measurements[4]) % 3,
        ]
        # Codespace pass requires every stabilizer outcome to be zero mod 3.
        return all(stabilizer == 0 for stabilizer in stabilizer_measurements)

    @staticmethod
    def codespace_X_stabilizer_check(x_measurements: Sequence[int]) -> bool:
        """
        Evaluate only the X-stabilizer constraints from measurements.

        Args:
            x_measurements (Sequence[int]): X-basis measurement values.

        Returns:
            bool: ``True`` when both X-stabilizer checks pass modulo 3.

        Raises:
            IndexError: If the input array does not contain required data-wire
                entries.
        """
        stabilizer_measurements = [
            (x_measurements[0] + x_measurements[1] - x_measurements[2]) % 3,
            (x_measurements[3] - x_measurements[1] - x_measurements[4]) % 3,
        ]
        return all(stabilizer == 0 for stabilizer in stabilizer_measurements)


@dataclass(frozen=True)
class QGRMThreeQutritDetectionCode:
    """
    Circuit builder for the [[3,1,2]]_3 QGRM error-detection code.

    Attributes:
        dimension (int): Local qutrit dimension used by constructed circuits.
        num_qudits (int): Total wire count (3 data + 2 ancilla).

    Methods:
        logical_plus_initial_state: Build the provided encoding circuit.
        single_qudit_logical_gate_circuit: Compile one logical gate token.
        affected_wires_per_operator: Return touched data wires per operator.
        z_stabilizer_measurement_ancillae_circuit: Measure ``Z0 Z1 Z2``.
        x_stabilizer_measurement_ancillae_circuit: Measure ``X0 X1 X2``.
        reset_measurement_wires: Reset both ancilla wires.
        affected_wires: Return touched data wires for a gate sequence.
        codespace_check: Evaluate both QGRM stabilizer constraints.
        codespace_X_stabilizer_check: Evaluate only the X stabilizer.
    """

    dimension: int = 3
    num_qudits: int = 5

    @classmethod
    def logical_plus_initial_state(cls) -> Circuit:
        """
        Build the provided QGRM encoding circuit.

        Args:
            None: Uses fixed QGRM wiring.

        Returns:
            Circuit: Encoding circuit on the 3-data-wire block.

        Raises:
            ValueError: If SDIM gate insertion fails.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("H", 0)
        c.add_gate("H", 1)
        c.add_gate("CNOT_INV", 0, 1)
        c.add_gate("CNOT_INV", 0, 2)
        c.add_gate("CNOT_INV", 1, 0)
        return c

    @classmethod
    def single_qudit_logical_gate_circuit(cls, operator: str) -> Circuit:
        """
        Compile one logical operator token into a QGRM subcircuit.

        Args:
            operator (str): Logical operator token.

        Returns:
            Circuit: Physical subcircuit implementing the logical operator.

        Raises:
            ValueError: If the operator token is unsupported.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        # Logical operators are encoded from the user-provided QGRM formulas.
        if operator == "I":
            c.add_gate("I", 0)
            return c
        if operator == "X":
            c.add_gate("X", 0)
            c.add_gate("X_INV", 1)
            return c
        if operator == "X_INV":
            c.add_gate("X_INV", 0)
            c.add_gate("X", 1)
            return c
        if operator == "Z":
            # Chosen dual form to support full Clifford-token compilation.
            c.add_gate("Z", 0)
            c.add_gate("Z_INV", 1)
            return c
        if operator == "Z_INV":
            c.add_gate("Z_INV", 0)
            c.add_gate("Z", 1)
            return c
        if operator == "H":
            c.add_gate("H", [0, 1, 2])
            return c
        if operator == "H_INV":
            c.add_gate("H_INV", [0, 1, 2])
            return c
        if operator in ("P", "S"):
            c.add_gate("P", [0, 1, 2])
            return c
        if operator in ("P_INV", "S_INV"):
            c.add_gate("P_INV", [0, 1, 2])
            return c

        raise ValueError(
            f"Gate '{operator}' is not supported by the QGRM definition.")

    @classmethod
    def affected_wires_per_operator(cls, operator: str) -> list[int]:
        """
        Return data-wire indices touched by one logical operator token.

        Args:
            operator (str): Logical operator token.

        Returns:
            list[int]: Data-wire indices touched by that operator.

        Raises:
            ValueError: If the operator token is unsupported.
        """
        if operator in ("I",):
            return []
        if operator in ("X", "X_INV", "Z", "Z_INV"):
            return [0, 1]
        if operator in ("H", "H_INV", "P", "P_INV", "S", "S_INV"):
            return [0, 1, 2]
        raise ValueError(
            f"Gate '{operator}' is not supported by the QGRM definition.")

    @classmethod
    def z_stabilizer_measurement_ancillae_circuit(cls) -> Circuit:
        """
        Build ancilla measurement for stabilizer ``Z0 Z1 Z2``.

        Args:
            None: Uses fixed QGRM wiring.

        Returns:
            Circuit: Z-stabilizer measurement block on ancilla wire 3.

        Raises:
            ValueError: If SDIM gate insertion fails.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("CNOT", 0, 3)
        c.add_gate("CNOT", 1, 3)
        c.add_gate("CNOT", 2, 3)
        c.add_gate("M", 3)
        return c

    @classmethod
    def x_stabilizer_measurement_ancillae_circuit(cls) -> Circuit:
        """
        Build ancilla measurement for stabilizer ``X0 X1 X2``.

        Args:
            None: Uses fixed QGRM wiring.

        Returns:
            Circuit: X-stabilizer measurement block on ancilla wire 4.

        Raises:
            ValueError: If SDIM gate insertion fails.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("H", 4)
        c.add_gate("CNOT", 4, 0)
        c.add_gate("CNOT", 4, 1)
        c.add_gate("CNOT", 4, 2)
        c.add_gate("H_INV", 4)
        c.add_gate("M", 4)
        return c

    @classmethod
    def reset_measurement_wires(cls) -> Circuit:
        """
        Build ancilla reset block for QGRM checks.

        Args:
            None: Uses fixed ancilla-wire indices.

        Returns:
            Circuit: Reset subcircuit on wires 3 and 4.

        Raises:
            ValueError: If SDIM gate insertion fails.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("RESET", 3)
        c.add_gate("RESET", 4)
        return c

    @classmethod
    def terminal_logical_x_measurement_circuit(cls) -> Circuit:
        """
        Build the QGRM terminal logical-X measurement block.

        Args:
            None: Uses fixed QGRM readout wiring.

        Returns:
            Circuit: Two-wire terminal X-basis measurement subcircuit.

        Raises:
            ValueError: If SDIM gate insertion fails during circuit assembly.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("H_INV", [0, 1])
        c.add_gate("M", [0, 1])
        return c

    @classmethod
    def affected_wires(cls, cliff: Iterable[str]) -> list[int]:
        """
        Return all unique data wires touched by a logical gate sequence.

        Args:
            cliff (Iterable[str]): Logical operator token sequence.

        Returns:
            list[int]: Sorted unique data-wire indices touched by the sequence.

        Raises:
            ValueError: Propagated if any operator token is unsupported.
        """
        wires: set[int] = set()
        for gate in cliff:
            wires.update(cls.affected_wires_per_operator(gate))
        return sorted(wires)

    @staticmethod
    def codespace_check(z_measurements: Sequence[int],
                        x_measurements: Sequence[int]) -> bool:
        """
        Evaluate full QGRM stabilizer constraints from measurements.

        Args:
            z_measurements (Sequence[int]): Z-basis data-wire measurements
                ordered as ``[z0, z1, z2]``.
            x_measurements (Sequence[int]): X-basis data-wire measurements
                ordered as ``[x0, x1, x2]``.

        Returns:
            bool: ``True`` when both stabilizers evaluate to zero modulo 3.

        Raises:
            IndexError: If input arrays do not contain required wire entries.
        """
        stabilizer_measurements = [
            (z_measurements[0] + z_measurements[1] + z_measurements[2]) % 3,
            (x_measurements[0] + x_measurements[1] + x_measurements[2]) % 3,
        ]
        # Codespace pass requires both independent stabilizers to vanish mod 3.
        return all(stabilizer == 0 for stabilizer in stabilizer_measurements)

    @staticmethod
    def codespace_X_stabilizer_check(x_measurements: Sequence[int]) -> bool:
        """
        Evaluate only the QGRM X-stabilizer constraint from measurements.

        Args:
            x_measurements (Sequence[int]): X-basis data-wire measurements
                ordered as ``[x0, x1, x2]``.

        Returns:
            bool: ``True`` when the X stabilizer evaluates to zero modulo 3.

        Raises:
            IndexError: If the input array does not contain required entries.
        """
        stabilizer_measurement = (
            x_measurements[0] + x_measurements[1] + x_measurements[2]
        ) % 3
        return stabilizer_measurement == 0

