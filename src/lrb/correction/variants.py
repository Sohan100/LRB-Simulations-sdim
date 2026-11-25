"""Protocol-variant LRB circuits for the correction study.

``build_variant_circuit`` reproduces ``LRBCircuitGenerator.
generate_lrb_clifford_sequence`` (SI1000 noise, folded qutrit code, sdim 1.4.0
scheduling) exactly when every flag is off, and can change three protocol choices:

* ``checked_inverse``: also run a check round (and ancilla reset) after every
  inverse Clifford except the last, so the second half of the mirrored
  sequence is checked like the first instead of running m Cliffords blind;
* ``parallel_measurement``: entangle all four ancillas first and measure them
  in one simultaneous layer. The current generator already schedules every
  round so that all readouts share one moment; under it this flag only changes
  the order of the entangling gates (the study's results came from the earlier
  scheduler, where it saved one resonator-idle layer per ancilla);
* ``full_terminal_readout``: read all five data qutrits in the X basis at the
  end (noiselessly, as the terminal readout already is), which adds two
  noiseless terminal X-stabilizer detectors ``lrb_term_xstab_{0,1}``.

The functions only build circuits in memory; nothing is written to the run
folders and ``scripts/generate_circuits_folded.py`` is never invoked.
"""

from __future__ import annotations

from typing import Sequence

from ..circuit_generator import LRBCircuitGenerator
from ..code_definitions import FoldedQutritDetectionCode
from ..code_simulation_profiles import CodeSimulationProfileRegistry
from ..detector_events import (
    expression_from_measurement_indices,
    indices_and_coefficients_from_wire_terms,
    measurement_indices_by_wire,
)

TERMINAL_LABEL = "lrb_term_xstab_{}"


def folded_generator() -> tuple[LRBCircuitGenerator, object]:
    """Return an SI1000 generator and code definition for ``folded_qutrit``."""
    profile = CodeSimulationProfileRegistry.resolve_code_profile("folded_qutrit")
    generator = LRBCircuitGenerator(code_definition=profile.code_definition, noise_model="si1000")
    return generator, profile.code_definition


def parallel_check_round_circuit():
    """
    Ideal check round with all four ancillas measured in one layer.

    The entangling sequences per ancilla are those of the production profile
    (``z_ALT`` Z-checks, then the X-checks); only the four measurements are
    deferred to the end so they form one simultaneous measurement layer.
    """
    from sdim.circuit import Circuit

    code = FoldedQutritDetectionCode
    c = Circuit(dimension=code.dimension, num_qudits=code.num_qudits)
    c.add_gate("CNOT", 0, 5)
    c.add_gate("CNOT_INV", 1, 5)
    c.add_gate("CNOT_INV", 3, 5)
    c.add_gate("CNOT", 1, 6)
    c.add_gate("CNOT", 2, 6)
    c.add_gate("CNOT_INV", 4, 6)
    c.add_gate("H", 7)
    c.add_gate("CNOT", 7, 0)
    c.add_gate("CNOT", 7, 1)
    c.add_gate("CNOT_INV", 7, 2)
    c.add_gate("H_INV", 7)
    c.add_gate("H", 8)
    c.add_gate("CNOT", 8, 3)
    c.add_gate("CNOT_INV", 8, 1)
    c.add_gate("CNOT_INV", 8, 4)
    c.add_gate("H_INV", 8)
    for wire in (5, 6, 7, 8):
        c.add_gate("M", wire)
    return c


def build_variant_circuit(
    clifford_strings: Sequence[str],
    depth: int,
    *,
    checked_inverse: bool = False,
    parallel_measurement: bool = False,
    full_terminal_readout: bool = False,
    p: float | None = None,
    generator: LRBCircuitGenerator | None = None,
):
    """
    Build one annotated SI1000 LRB circuit for a protocol variant.

    With every flag off this is exactly ``LRBCircuitGenerator.generate_lrb_clifford_sequence``
    for the same Clifford sample: each forward Clifford is one noisy scheduling segment followed
    by one noisy check round (ancilla reset, then the stabilizer blocks, all readouts in the last
    moment), the inverse Cliffords run back to back, and one final check round closes the
    sequence.

    Args:
        clifford_strings (Sequence[str]): Max-depth Clifford descriptors; the
            first ``depth`` are the forward sequence, as in the generator.
        depth (int): Benchmark depth m.
        checked_inverse (bool): Check after every inverse Clifford too.
        parallel_measurement (bool): Use ``parallel_check_round_circuit`` for every round.
            The generator already reads all ancillas in one moment, so this changes the
            order of the entangling gates only.
        full_terminal_readout (bool): Read all five data qutrits at the end.
        p (float | None): If given, materialize noise probabilities at ``p``.
        generator (LRBCircuitGenerator | None): Optional generator instance.

    Returns:
        Circuit: Annotated circuit with ``lrb_stab_r*_w*`` detectors, optional
        terminal detectors, and the ``lrb_logical`` observable.
    """
    from sdim.circuit import Circuit

    if generator is None:
        generator, code = folded_generator()
    else:
        code = generator.code_definition
    width = code.encoded_num_qudits
    strings = list(clifford_strings)
    max_depth = len(strings)
    gate_lists, inverse_lists = generator._clifford_and_inverse_gate_lists(strings)
    forward = gate_lists[:depth] if depth else []
    inverse = (generator._closing_gate_lists(strings, inverse_lists, max_depth, depth)
               if depth else [])

    def noisy(block):
        return generator._noisy_segment(block, True)

    def check_round():
        if not parallel_measurement:
            return noisy(generator._check_round_circuit())
        block = Circuit(dimension=code.dimension, num_qudits=width)
        if code.reset_measurement_wires is not None:
            block = block + code.reset_measurement_wires()
        return noisy(block + parallel_check_round_circuit())

    circuit = Circuit(dimension=code.dimension, num_qudits=width)
    circuit = circuit + code.logical_plus_initial_state()
    rounds = 0
    for clifford in forward:
        circuit = circuit + noisy(generator._logical_clifford_circuit(clifford))
        circuit = circuit + check_round()
        rounds += 1
    for position, clifford in enumerate(inverse):
        circuit = circuit + noisy(generator._logical_clifford_circuit(clifford))
        if checked_inverse and position < len(inverse) - 1:
            circuit = circuit + check_round()
            rounds += 1
    circuit = circuit + check_round()
    rounds += 1
    circuit = circuit + (code.terminal_const0_measurement() if full_terminal_readout
                         else code.terminal_measurement())
    generator.append_lrb_detector_events(circuit, rounds - 1)
    if full_terminal_readout:
        indices_by_wire = measurement_indices_by_wire(circuit)
        for index, terms in enumerate(code.terminal_x_stabilizer_terms):
            rec, coefficients = indices_and_coefficients_from_wire_terms(
                indices_by_wire, terms, use_last_measurement=True)
            circuit.add_gate(
                "DETECTOR",
                expr=expression_from_measurement_indices(rec, coefficients),
                label=TERMINAL_LABEL.format(index),
            )
    if p is not None:
        generator.update_noise_param(circuit, p)
    return circuit
