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

3. ``FoldedQutritSplitAncillaUniformDetectionCode``
   Encodes the split-boundary variant of the folded qutrit code in which the
   two geometrically nonlocal boundary stabilizers use a second local relay
   ancilla. The logical code and stabilizers are unchanged; only the
   stabilizer-measurement layout and syndrome/readout arithmetic differ.

4. ``QGRMThreeQutritDetectionCode``
   Encodes the circuit templates for the [[3,1,2]]_3 QGRM qutrit detection
   code, including initialization, logical operators, and stabilizer checks.

5. ``QubitCliffordLibrary``, ``FoldedQubitDetectionCode`` and
   ``FoldedQubitSplitAncillaUniformDetectionCode``
   The qubit ``[[5,1,2]]`` counterparts of the folded code: the 24 single-qubit
   Cliffords and the same circuits built in dimension two, with modulo-two
   stabilizer checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    from sdim.circuit import Circuit
except Exception:  # pragma: no cover - environment dependent.
    # Keep metadata/profile imports usable without SDIM runtime.
    Circuit = Any

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

# The single-qutrit Clifford group modulo global phase has
# d^3 (d^2 - 1) = 216 elements for d = 3. Every element factors uniquely as a
# Pauli X^a Z^b (9 choices) times a symplectic part from SL(2, Z_3)
# (24 choices). Descriptors are words whose rightmost symbol acts first, so
# the Pauli prefix is applied last.
QUTRIT_SYMPLECTIC_WORDS = (
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
)
QUTRIT_PAULI_WORDS = ("", "X", "XX", "Z", "XZ", "XXZ", "ZZ", "XZZ", "XXZZ")

single_qutrit_cliffords_exhaustive_with_S_strings = [
    pauli + word if word != "I" else (pauli or "I")
    for pauli in QUTRIT_PAULI_WORDS
    for word in QUTRIT_SYMPLECTIC_WORDS
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
        terminal_const0_logical_x_measurement_circuit(): Build direct
            terminal X-data readout for the special ``const=0`` protocol.
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
    def terminal_const0_logical_x_measurement_circuit(cls) -> Circuit:
        """
        Build the folded-code direct terminal X-data readout for ``const=0``.

        The measured data wires are exactly those needed to evaluate the two
        X stabilizers and the logical-X observable. This block is appended
        without noise by the const=0 circuit generator.

        Returns:
            Circuit: Error-free direct X-basis data measurement subcircuit.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("H_INV", [0, 1, 2, 3, 4])
        c.add_gate("M", [0, 1, 2, 3, 4])
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
class FoldedQutritSplitAncillaUniformDetectionCode(
        FoldedQutritDetectionCode):
    """
    Split-ancilla folded-code circuit templates for uniform-interval checks.

    This class describes a measurement-layout variant of the same
    ``[[5,1,2]]_3`` folded qutrit surface-code detector used by
    ``FoldedQutritDetectionCode``. The five data wires, logical operators, and
    stabilizer generators are unchanged:

    ``S_Z1 = Z0 Z1^-1 Z3^-1``
        Boundary Z-type stabilizer. In the conventional folded-code geometry,
        one ancilla cannot locally touch all three participating data qutrits,
        so this mode uses a nearby relay ancilla to collect the distant
        contribution coherently, transfers that contribution into the main
        syndrome ancilla, uncomputes the relay, and then measures both
        ancillas.

    ``S_Z2 = Z1 Z2 Z4^-1``
        Interior Z-type stabilizer. This remains a one-ancilla stabilizer
        because a single local ancilla can reach all data qutrits in the
        stabilizer support.

    ``S_X1 = X0 X1 X2^-1``
        Interior X-type stabilizer. This also remains a one-ancilla
        stabilizer for the same local-connectivity reason.

    ``S_X2 = X3 X1^-1 X4^-1``
        Boundary X-type stabilizer. This mirrors ``S_Z1`` and is split across
        the same coherent relay strategy used for ``S_Z1``.

    The class intentionally inherits the initialization, logical-gate, logical
    readout, affected-wire, and direct terminal-X routines from
    ``FoldedQutritDetectionCode``. Those inherited routines use ``cls`` for the
    circuit width, so this subclass can keep the same data-wire behavior while
    increasing the total wire count from nine to eleven.

    Attributes:
        dimension (int): Local qutrit dimension. This remains three.
        num_qudits (int): Total circuit width. Wires ``0..4`` are data qutrits;
            wires ``5..10`` are stabilizer-measurement ancillas.

    Methods:
        z_boundary_split_stabilizer_measurement_ancillae_circuit: Measure
            ``S_Z1`` with main syndrome ancilla ``5`` and relay ancilla ``9``.
        z_middle_stabilizer_measurement_ancillae_circuit: Measure ``S_Z2`` on
            ancilla ``6``.
        x_middle_stabilizer_measurement_ancillae_circuit: Measure ``S_X1`` on
            ancilla ``7``.
        x_boundary_split_stabilizer_measurement_ancillae_circuit: Measure
            ``S_X2`` with main syndrome ancilla ``8`` and relay ancilla ``10``.
        reset_measurement_wires: Reset all six split-mode ancilla wires after
            each stabilizer-check round.
        split_ancilla_stabilizer_check: Interpret the six ancilla results as
            four stabilizer syndromes plus two relay-cleanliness checks and
            return the full pass/fail decision.
    """

    dimension: int = 3
    num_qudits: int = 11

    @classmethod
    def z_boundary_split_stabilizer_measurement_ancillae_circuit(
            cls) -> Circuit:
        """
        Build the split measurement circuit for boundary stabilizer ``S_Z1``.

        The original stabilizer is ``S_Z1 = Z0 Z1^-1 Z3^-1``. In the
        single-ancilla circuit this is one qutrit-valued syndrome
        ``z0 - z1 - z3``. A tempting two-ancilla implementation is to measure
        ``z0 - z1`` and ``-z3`` separately and add the two classical outcomes,
        but that exposes non-stabilizer partial information. The coherent
        circuit below instead uses ancilla ``9`` only as a relay: it collects
        the distant ``-z3`` contribution, transfers that contribution into
        syndrome ancilla ``5`` with ``CNOT 9 5``, and then uncomputes the relay
        by applying the inverse data-relay coupling. At the end, ancilla ``5``
        carries the reconstructed ``S_Z1`` syndrome and ancilla ``9`` should
        read zero when the relay was clean.

        Args:
            None: The method uses fixed folded-code data and ancilla wires.

        Returns:
            Circuit: Two-part Z-boundary stabilizer readout circuit on the
            eleven-wire split-ancilla folded code.

        Raises:
            ValueError: If SDIM rejects a gate placement during construction.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        c.add_gate("CNOT", 0, 5)
        c.add_gate("CNOT_INV", 1, 5)
        c.add_gate("CNOT_INV", 3, 9)
        c.add_gate("CNOT", 9, 5)
        c.add_gate("CNOT", 3, 9)
        c.add_gate("M", 5)
        c.add_gate("M", 9)

        return c

    @classmethod
    def z_middle_stabilizer_measurement_ancillae_circuit(cls) -> Circuit:
        """
        Build the one-ancilla measurement circuit for interior stabilizer ``S_Z2``.

        The stabilizer is ``S_Z2 = Z1 Z2 Z4^-1``. It is geometrically local in
        this folded-code layout, so this split-ancilla mode intentionally keeps
        the same one-ancilla syndrome extraction used by the normal folded
        profile: ancilla ``6`` stores ``z1 + z2 - z4`` and must be zero modulo
        three for the check to pass.

        Args:
            None: The method uses fixed folded-code data and ancilla wires.

        Returns:
            Circuit: One-ancilla Z-interior stabilizer readout circuit.

        Raises:
            ValueError: If SDIM rejects a gate placement during construction.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        c.add_gate("CNOT", 1, 6)
        c.add_gate("CNOT", 2, 6)
        c.add_gate("CNOT_INV", 4, 6)
        c.add_gate("M", 6)

        return c

    @classmethod
    def x_middle_stabilizer_measurement_ancillae_circuit(cls) -> Circuit:
        """
        Build the one-ancilla measurement circuit for interior stabilizer ``S_X1``.

        The stabilizer is ``S_X1 = X0 X1 X2^-1``. This check remains local to a
        single conventional ancilla, so ancilla ``7`` measures the whole
        qutrit-valued syndrome ``x0 + x1 - x2``. The leading and trailing
        Hadamard basis changes convert the ancilla-mediated controlled-X style
        couplings into the X-basis stabilizer readout used by this repository.

        Args:
            None: The method uses fixed folded-code data and ancilla wires.

        Returns:
            Circuit: One-ancilla X-interior stabilizer readout circuit.

        Raises:
            ValueError: If SDIM rejects a gate placement during construction.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        c.add_gate("H", 7)
        c.add_gate("CNOT", 7, 0)
        c.add_gate("CNOT", 7, 1)
        c.add_gate("CNOT_INV", 7, 2)
        c.add_gate("H_INV", 7)
        c.add_gate("M", 7)

        return c

    @classmethod
    def x_boundary_split_stabilizer_measurement_ancillae_circuit(
            cls) -> Circuit:
        """
        Build the split measurement circuit for boundary stabilizer ``S_X2``.

        The original stabilizer is ``S_X2 = X3 X1^-1 X4^-1``. As with ``S_Z1``,
        this circuit avoids separately measuring non-stabilizer partials.
        Ancilla ``8`` is the main X-basis syndrome ancilla for the nearby
        ``x3 - x1`` contribution. Ancilla ``10`` is a relay for the distant
        ``-x4`` contribution. The ``CNOT_INV 8 10`` relay-combination step and
        the subsequent inverse data-relay coupling make the final readout
        deterministic at zero noise: ancilla ``8`` carries the reconstructed
        ``S_X2`` syndrome and ancilla ``10`` should read zero when the relay
        was clean.

        Args:
            None: The method uses fixed folded-code data and ancilla wires.

        Returns:
            Circuit: Two-part X-boundary stabilizer readout circuit on the
            eleven-wire split-ancilla folded code.

        Raises:
            ValueError: If SDIM rejects a gate placement during construction.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)

        c.add_gate("H", 8)
        c.add_gate("H", 10)
        c.add_gate("CNOT", 8, 3)
        c.add_gate("CNOT_INV", 8, 1)
        c.add_gate("CNOT_INV", 10, 4)
        c.add_gate("CNOT_INV", 8, 10)
        c.add_gate("CNOT", 10, 4)
        c.add_gate("H_INV", 8)
        c.add_gate("H_INV", 10)
        c.add_gate("M", 8)
        c.add_gate("M", 10)

        return c

    @classmethod
    def reset_measurement_wires(cls) -> Circuit:
        """
        Build the split-mode ancilla reset block.

        This variant resets all six stabilizer-measurement ancillas after each
        check round: ``5`` and ``9`` for split ``S_Z1``, ``6`` for ``S_Z2``,
        ``7`` for ``S_X1``, and ``8`` and ``10`` for split ``S_X2``. The reset
        block is intentionally separate from the stabilizer-measurement blocks
        so the generic generator can insert SI1000 reset noise around it in
        the same place where the normal folded profile resets ancillas.

        Args:
            None: The method uses fixed split-mode ancilla-wire indices.

        Returns:
            Circuit: Reset subcircuit on wires ``5`` through ``10``.

        Raises:
            ValueError: If SDIM rejects a reset gate placement.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("RESET", 5)
        c.add_gate("RESET", 6)
        c.add_gate("RESET", 7)
        c.add_gate("RESET", 8)
        c.add_gate("RESET", 9)
        c.add_gate("RESET", 10)
        return c

    @staticmethod
    def split_ancilla_stabilizer_check(
            stabilizer_measurements: Sequence[int]) -> bool:
        """
        Evaluate the full split-ancilla stabilizer layer modulo three.

        The measurement order is fixed by the corresponding split-mode runtime
        profile. The boundary entries are not independent partial stabilizer
        readouts; they are coherent syndrome-and-relay readouts:

        ``m5``
            Reconstructed syndrome for ``S_Z1`` after the ``9 -> 5`` relay
            combination.
        ``m6``
            Full one-ancilla syndrome for ``S_Z2``, equal to ``z1 + z2 - z4``.
        ``m7``
            Full one-ancilla syndrome for ``S_X1``, equal to ``x0 + x1 - x2``.
        ``m8``
            Reconstructed syndrome for ``S_X2`` after the ``8 -> 10`` relay
            combination and relay uncompute.
        ``m9``
            Relay-cleanliness readout for the ``S_Z1`` relay ancilla. In an
            ideal stabilizer measurement this is zero; nonzero values indicate
            that the relay did not uncompute cleanly and should reject the
            check layer.
        ``m10``
            Relay-cleanliness readout for the ``S_X2`` relay ancilla.

        All six values are qutrit-valued and therefore checked modulo three.
        The four stabilizer-syndrome entries ``m5``, ``m6``, ``m7``, and ``m8``
        must vanish, and the two relay-cleanliness entries ``m9`` and ``m10``
        must also vanish. Requiring the relay outputs to be zero is stricter
        than adding them classically to the syndrome outputs; it prevents a
        noisy relay from accidentally canceling a real syndrome value.

        Args:
            stabilizer_measurements (Sequence[int]): Six qutrit-valued ancilla
                measurement outcomes ordered as ``(m5, m6, m7, m8, m9, m10)``.

        Returns:
            bool: ``True`` only when all four reconstructed stabilizer
            syndromes and both relay-cleanliness readouts vanish modulo three.

        Raises:
            ValueError: If the caller supplies anything other than six
                split-mode stabilizer readouts.
        """
        if len(stabilizer_measurements) != 6:
            raise ValueError(
                "Split-ancilla folded checks require exactly six "
                "stabilizer measurement values ordered as "
                "(m5, m6, m7, m8, m9, m10)."
            )

        m5, m6, m7, m8, m9, m10 = stabilizer_measurements
        stabilizer_syndromes = [
            m5 % 3,
            m6 % 3,
            m7 % 3,
            m8 % 3,
            m9 % 3,
            m10 % 3,
        ]
        return all(syndrome == 0 for syndrome in stabilizer_syndromes)


@dataclass(frozen=True)
class FoldedQutritConventionalSitesDetectionCode(FoldedQutritDetectionCode):
    """
    Two-ancilla checks of the folded ``[[5,1,2]]_3`` code on the four
    conventional surface-code ancilla sites.

    The data sit as in the paper's patch, ``q2`` at the centre (``q0`` top
    left, ``q1`` top right, ``q3`` bottom left, ``q4`` bottom right), and the
    ancillas at the four sites between them:

    ``N = 5`` touches ``q0, q1, q2``; ``W = 6`` touches ``q0, q2, q3``;
    ``E = 7`` touches ``q1, q2, q4``; ``S = 8`` touches ``q2, q3, q4``.

    Every two-qutrit gate couples a site to one of its own data qutrits; no
    two sites are coupled. ``S_X1 = X0 X1 X2^-1`` (G3) fits ``N`` and
    ``S_Z2 = Z1 Z2 Z4^-1`` (G2) fits ``E``. No site covers
    ``S_Z1 = Z0 Z1^-1 Z3^-1`` (G1) or ``S_X2 = X1^-1 X3 X4^-1`` (G4), because
    both contain the opposite corners ``q1`` and ``q3``. They are measured by a
    data relay:

    G1
        ``N`` collects ``-z1`` and lends it to ``q0`` (``q0 -> z0 - z1``);
        ``W`` reads ``q0`` and ``-z3``; ``N`` takes the value back from ``q0``
        and clears itself against ``q1``.
    G4
        The same in the X basis: ``E`` lends its ``q1`` part to ``q4``,
        ``S`` reads ``q4`` and ``q3``, ``E`` takes it back.

    ``N`` measures G3 before lending and ``E`` lends before measuring G2, so
    each site is read once per round and no ancilla is reset mid-round. The
    readouts are ``m5 = G3``, ``m6 = G1 + G3`` (``N`` holds its G3 value while
    lending), ``m7 = G2`` and ``m8 = G4``, so all four vanish exactly when the
    four stabilizers do. A noiseless single-fault enumeration of this round
    finds no undetected logical error. The gate order is the shallowest
    correct order found by a search over orders: 11 gate moments under the
    ASAP scheduler, with one reset and one measurement moment.

    Attributes:
        dimension (int): Local qutrit dimension, three.
        num_qudits (int): Total circuit width, nine (five data, four sites).
    """

    dimension: int = 3
    num_qudits: int = 9

    @classmethod
    def conventional_sites_check_round_circuit(cls) -> Circuit:
        """
        Build one check round on the sites ``N, W, E, S`` (wires 5-8).

        Gates are listed in the moment the ASAP scheduler gives them.

        Returns:
            Circuit: Check round without the leading ancilla resets.
        """
        n, w, e, s = 5, 6, 7, 8
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        # 1: N and S start X-type checks, E starts its X-type lend.
        c.add_gate("H", e)
        c.add_gate("H", s)
        c.add_gate("H", n)
        # 2-5: G3 on N. E collects its q1 part, lends it to q4 and takes it
        # back once S has read q4; S also reads q3 and W reads -z3.
        c.add_gate("CNOT", e, 1)
        c.add_gate("CNOT_INV", s, 3)
        c.add_gate("CNOT", n, 0)
        c.add_gate("CNOT", 4, e)
        c.add_gate("CNOT_INV", 3, w)
        c.add_gate("CNOT_INV", n, 2)
        c.add_gate("CNOT", s, 4)
        c.add_gate("CNOT", n, 1)
        c.add_gate("H_INV", s)
        c.add_gate("CNOT_INV", 4, e)
        c.add_gate("H_INV", n)
        # 6-7: E clears against q1 and returns to the Z basis; N collects -z1.
        c.add_gate("CNOT_INV", e, 1)
        c.add_gate("CNOT_INV", 1, n)
        c.add_gate("H_INV", e)
        # 8-11: N lends to q0, W reads q0, N takes it back and clears
        # against q1; G2 on E.
        c.add_gate("CNOT", n, 0)
        c.add_gate("CNOT", 2, e)
        c.add_gate("CNOT_INV", 4, e)
        c.add_gate("CNOT", 0, w)
        c.add_gate("CNOT_INV", n, 0)
        c.add_gate("CNOT", 1, e)
        c.add_gate("CNOT", 1, n)
        # One measurement moment for all four sites.
        c.add_gate("M", s)
        c.add_gate("M", w)
        c.add_gate("M", e)
        c.add_gate("M", n)
        return c


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
        terminal_const0_logical_x_measurement_circuit: Build direct terminal
            X-data readout for the special ``const=0`` protocol.
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
    def terminal_const0_logical_x_measurement_circuit(cls) -> Circuit:
        """
        Build the QGRM direct terminal X-data readout for ``const=0``.

        The measured data wires are exactly those needed to evaluate the X
        stabilizer and the logical-X observable. This block is appended
        without noise by the const=0 circuit generator.

        Returns:
            Circuit: Error-free direct X-basis data measurement subcircuit.
        """
        c = Circuit(dimension=cls.dimension, num_qudits=cls.num_qudits)
        c.add_gate("H_INV", [0, 1, 2])
        c.add_gate("M", [0, 1, 2])
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



# ---------------------------------------------------------------------------
# Qubit counterpart of the folded code.
# ---------------------------------------------------------------------------
# The single-qubit Clifford group modulo global phase has
# d^3 (d^2 - 1) = 24 elements for d = 2: a Pauli X^a Z^b (4 choices) times a
# symplectic part from SL(2, Z_2), which permutes {X, Y, Z} (6 choices).
# Descriptors follow the qutrit convention: the rightmost symbol acts first,
# so the Pauli prefix is applied last.
QUBIT_SYMPLECTIC_WORDS = ("I", "H", "S", "HS", "SH", "HSH")
QUBIT_PAULI_WORDS = ("", "X", "Z", "XZ")

single_qubit_cliffords_exhaustive_with_S_strings = [
    pauli + word if word != "I" else (pauli or "I")
    for pauli in QUBIT_PAULI_WORDS
    for word in QUBIT_SYMPLECTIC_WORDS
]

# Internal catalog in terms of the P gate symbol used by SDIM.
single_qubit_cliffords_exhaustive_strings = [
    cliff.replace("S", "P")
    for cliff in single_qubit_cliffords_exhaustive_with_S_strings
]


class QubitCliffordLibrary(QutritCliffordLibrary):
    """
    Qubit counterpart of ``QutritCliffordLibrary`` over the 24 Cliffords.

    Gate tokens keep their qutrit spellings. In dimension two SDIM gives
    ``H_INV = H``, ``P = S``, ``P_INV = S^dagger``, ``X_INV = X`` and
    ``Z_INV = Z``, so ``SINGLE_QUTRIT_INVERSES`` is still the right inverse
    table and descriptors parse exactly as for qutrits.

    Methods:
        qubit_gate_seq_list(cliff): Convert one descriptor string into an
            execution-order gate list.
        build_inverse_lookup(): Build descriptor-to-inverse mappings for the
            24-element qubit catalog.
    """

    @staticmethod
    def qubit_gate_seq_list(cliff: str) -> list[str]:
        """Convert a qubit Clifford descriptor into execution-order gates."""
        return QutritCliffordLibrary.qutrit_gate_seq_list(cliff)

    @classmethod
    def build_inverse_lookup(cls) -> dict[str, list[str]]:
        """Build inverse mappings for all descriptors in the qubit catalog."""
        return {
            cliff: [SINGLE_QUTRIT_INVERSES[gate] for gate in cliff]
            for cliff in single_qubit_cliffords_exhaustive_strings
        }


class _QubitStabilizerArithmetic:
    """
    Modulo-two stabilizer checks shared by the qubit folded-code classes.

    The qubit stabilizers have the same supports as the qutrit ones; every
    ``-1`` exponent becomes ``+1`` modulo two (``S_Z1 = Z0 Z1 Z3``,
    ``S_Z2 = Z1 Z2 Z4``, ``S_X1 = X0 X1 X2``, ``S_X2 = X1 X3 X4``).
    """

    @staticmethod
    def codespace_check(z_measurements: Sequence[int],
                        x_measurements: Sequence[int]) -> bool:
        """Return ``True`` when all four stabilizers vanish modulo two."""
        stabilizer_measurements = [
            (z_measurements[0] + z_measurements[1] + z_measurements[3]) % 2,
            (z_measurements[1] + z_measurements[2] + z_measurements[4]) % 2,
            (x_measurements[0] + x_measurements[1] + x_measurements[2]) % 2,
            (x_measurements[3] + x_measurements[1] + x_measurements[4]) % 2,
        ]
        return all(stabilizer == 0 for stabilizer in stabilizer_measurements)

    @staticmethod
    def codespace_X_stabilizer_check(x_measurements: Sequence[int]) -> bool:
        """Return ``True`` when both X stabilizers vanish modulo two."""
        stabilizer_measurements = [
            (x_measurements[0] + x_measurements[1] + x_measurements[2]) % 2,
            (x_measurements[3] + x_measurements[1] + x_measurements[4]) % 2,
        ]
        return all(stabilizer == 0 for stabilizer in stabilizer_measurements)


@dataclass(frozen=True)
class FoldedQubitDetectionCode(_QubitStabilizerArithmetic,
                               FoldedQutritDetectionCode):
    """
    Circuit builder for the qubit ``[[5,1,2]]`` folded surface code.

    Wire layout, stabilizer supports, logical operators, fold-transversal
    logical gates and check circuits are those of
    ``FoldedQutritDetectionCode``, built in dimension two. Every inherited
    builder reads ``cls.dimension``, so the same gate lists give the qubit
    circuits (``H_INV -> H``, ``P -> S``, ``P_INV -> S^dagger``,
    ``CNOT_INV -> CNOT``, ``CZ_INV -> CZ``). Only the modulo arithmetic of the
    classical checks changes.

    Attributes:
        dimension (int): Local qubit dimension, two.
        num_qudits (int): Total wire count, nine (five data, four ancillas).
    """

    dimension: int = 2
    num_qudits: int = 9


@dataclass(frozen=True)
class FoldedQubitSplitAncillaUniformDetectionCode(
        _QubitStabilizerArithmetic,
        FoldedQutritSplitAncillaUniformDetectionCode):
    """
    Split-ancilla layout of the qubit ``[[5,1,2]]`` folded code.

    The qubit counterpart of ``FoldedQutritSplitAncillaUniformDetectionCode``:
    the same eleven wires and relay circuits for the boundary stabilizers
    ``S_Z1`` and ``S_X2``, built in dimension two.

    Attributes:
        dimension (int): Local qubit dimension, two.
        num_qudits (int): Total circuit width, eleven.
    """

    dimension: int = 2
    num_qudits: int = 11

    @staticmethod
    def split_ancilla_stabilizer_check(
            stabilizer_measurements: Sequence[int]) -> bool:
        """
        Evaluate the split-ancilla stabilizer layer modulo two.

        Same readout order and pass rule as the qutrit version: the four
        syndromes ``m5..m8`` and the relay-cleanliness readouts ``m9`` and
        ``m10`` must all vanish.
        """
        if len(stabilizer_measurements) != 6:
            raise ValueError(
                "Split-ancilla folded checks require exactly six "
                "stabilizer measurement values ordered as "
                "(m5, m6, m7, m8, m9, m10)."
            )
        return all(value % 2 == 0 for value in stabilizer_measurements)


@dataclass(frozen=True)
class FoldedQubitConventionalSitesDetectionCode(
        _QubitStabilizerArithmetic,
        FoldedQutritConventionalSitesDetectionCode):
    """
    Conventional-site two-ancilla checks of the qubit ``[[5,1,2]]`` code.

    The qubit counterpart of ``FoldedQutritConventionalSitesDetectionCode``:
    the same nine wires and check round, built in dimension two.

    Attributes:
        dimension (int): Local qubit dimension, two.
        num_qudits (int): Total circuit width, nine.
    """

    dimension: int = 2
    num_qudits: int = 9
