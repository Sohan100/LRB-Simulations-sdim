"""Protocol-variant LRB circuits for the correction study.

``build_variant_circuit`` reproduces ``LRBCircuitGenerator.
generate_lrb_clifford_sequence`` (SI1000 noise, folded qutrit code) exactly
when every flag is off, and can change three protocol choices:

* ``checked_inverse``: also run a check round (and ancilla reset) after every
  inverse Clifford except the last, so the second half of the mirrored
  sequence is checked like the first instead of running m Cliffords blind;
* ``parallel_measurement``: entangle all four ancillas first and measure them
  in one simultaneous layer, so SI1000 idle noise on the data is applied once
  per round instead of once per ancilla measurement;
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

    Args:
        clifford_strings (Sequence[str]): Max-depth Clifford descriptors; the
            first ``depth`` are the forward sequence, as in the generator.
        depth (int): Benchmark depth m.
        checked_inverse (bool): Check after every inverse Clifford too.
        parallel_measurement (bool): One simultaneous ancilla readout layer.
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
    gate_lists = [code.clifford_to_gate_sequence(c) for c in clifford_strings]
    inverse_lists = [code.clifford_inverse_map[c] for c in clifford_strings]
    inverse_lists.reverse()
    forward = gate_lists[:depth] if depth else []
    inverse = inverse_lists[len(clifford_strings) - depth:] if depth else []

    def noisy(block, measurements: bool):
        return generator.apply_si1000_noise_to_circuit(block, noisy_measurements=measurements)

    def check_round():
        out = Circuit(dimension=code.dimension, num_qudits=width)
        if parallel_measurement:
            return out + noisy(parallel_check_round_circuit(), True)
        for factory, _ancillas in code.stabilizer_check_blocks:
            out = out + noisy(factory(), True)
        return out

    def reset_round():
        return noisy(code.reset_measurement_wires(), False)

    circuit = Circuit(dimension=code.dimension, num_qudits=width)
    circuit = circuit + code.logical_plus_initial_state()
    rounds = 0
    for clifford in forward:
        for gate in clifford:
            circuit = circuit + noisy(code.logical_gate_circuit(gate), False)
        circuit = circuit + check_round() + reset_round()
        rounds += 1
    for position, clifford in enumerate(inverse):
        for gate in clifford:
            circuit = circuit + noisy(code.logical_gate_circuit(gate), False)
        if checked_inverse and position < len(inverse) - 1:
            circuit = circuit + check_round() + reset_round()
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
