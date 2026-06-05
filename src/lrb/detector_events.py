"""Shared detector and logical-observable conventions for LRB circuits.

This module is intentionally small and dependency-light. The circuit generator
uses it when appending SDIM ``DETECTOR`` and ``LOGICAL_OBSERVABLE`` operations;
the simulation unpacker uses the same functions when looking those events back
up in SDIM's compact detector payload. Keeping the labels and expression
format in one place prevents a quiet mismatch where generated circuits contain
perfectly valid detector data under labels the runtime never reads.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


LRB_LOGICAL_OBSERVABLE_LABEL = "lrb_logical"
LRB_CONST0_LOGICAL_OBSERVABLE_LABEL = "lrb_const0_logical"
RB_LOGICAL_OBSERVABLE_LABEL = "rb_logical"


def lrb_stabilizer_detector_label(check_round: int, wire: int) -> str:
    """
    Build the stable label for one ordinary LRB stabilizer detector.

    Args:
        check_round (int): Stabilizer-check round counted in protocol order.
            The final post-inverse check is round ``depth``.
        wire (int): Physical ancilla wire whose measured syndrome/relay value
            is represented by this detector.

    Returns:
        str: Label stored in the SDIM detector payload.

    Raises:
        ValueError: If a negative round or wire is supplied.
    """
    if check_round < 0 or wire < 0:
        raise ValueError("Detector labels require non-negative indices.")
    return f"lrb_stab_r{check_round}_w{wire}"


def lrb_const0_stabilizer_detector_label(stabilizer_index: int) -> str:
    """
    Build the stable label for one direct terminal-X ``const=0`` detector.

    Args:
        stabilizer_index (int): Index of the terminal X-stabilizer expression
            in the code profile.

    Returns:
        str: Label stored in the SDIM detector payload.

    Raises:
        ValueError: If ``stabilizer_index`` is negative.
    """
    if stabilizer_index < 0:
        raise ValueError("Detector labels require non-negative indices.")
    return f"lrb_const0_xstab_{stabilizer_index}"


def measurement_indices_by_wire(circuit: Any) -> dict[int, list[int]]:
    """
    Return global SDIM measurement indices grouped by physical wire.

    SDIM detector expressions reference only true measurement operations,
    namely ``M`` and ``M_X`` in this repository's circuit vocabulary. Reset
    operations do produce entries in the older raw measurement-result tensor,
    but SDIM's detector parser does not count resets as ``rec`` targets. This
    helper mirrors that detector convention exactly so detector expressions
    align with SDIM's internal ``seen_measurements`` counter.

    Args:
        circuit (Any): SDIM-like circuit object with an ``operations`` list.

    Returns:
        dict[int, list[int]]: Mapping from wire index to global measurement
        indices, in the order those true measurements appear in the circuit.

    Raises:
        AttributeError: If ``circuit`` does not expose SDIM-style operations.
    """
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    next_measurement_index = 0

    for operation in circuit.operations:
        if operation.gate_name not in ("M", "M_X"):
            continue
        grouped_indices[int(operation.qudit_index)].append(
            next_measurement_index)
        next_measurement_index += 1

    return dict(grouped_indices)


def expression_from_measurement_indices(
    measurement_indices: Sequence[int],
    coefficients: Sequence[int] | None = None,
) -> str:
    """
    Convert a linear qutrit expression into SDIM detector syntax.

    SDIM 1.3.4 expects expressions to mention measurement references as
    ``rec[<index>]``. Bare ``[<index>]`` syntax is parsed into a Python list and
    then fails when SDIM applies modulo arithmetic. This helper always emits
    the working ``rec[...]`` form and supports signed integer coefficients for
    stabilizers such as ``x0 + x1 - x2``.

    Args:
        measurement_indices (Sequence[int]): Global SDIM measurement indices.
        coefficients (Sequence[int] | None): Matching integer coefficients.
            ``None`` means every coefficient is ``+1``.

    Returns:
        str: SDIM detector expression string.

    Raises:
        ValueError: If the lengths differ or an index is negative.
    """
    if coefficients is None:
        coefficients = tuple(1 for _ in measurement_indices)
    if len(measurement_indices) != len(coefficients):
        raise ValueError(
            "Detector expression indices and coefficients must align.")

    terms: list[str] = []
    for measurement_index, coefficient in zip(
            measurement_indices, coefficients):
        if measurement_index < 0:
            raise ValueError("Detector expressions require non-negative rec "
                             "indices.")
        if coefficient == 0:
            continue
        if coefficient == 1:
            terms.append(f"rec[{measurement_index}]")
        elif coefficient == -1:
            terms.append(f"-rec[{measurement_index}]")
        else:
            terms.append(f"{coefficient}*rec[{measurement_index}]")

    return "+".join(terms) if terms else "0"


def indices_and_coefficients_from_wire_terms(
    measurement_indices: dict[int, list[int]],
    wire_terms: Iterable[tuple[int, int]],
    *,
    use_last_measurement: bool,
) -> tuple[list[int], list[int]]:
    """
    Resolve profile-level wire terms into detector ``rec`` indices.

    Code profiles describe logical observables and terminal direct-X
    stabilizers in physical-wire language, for example ``[(0, 1), (3, 1)]``.
    Detector expressions need global measurement indices instead. For terminal
    data readouts the desired event is the last true measurement on that wire;
    for stabilizer rounds the caller resolves the round explicitly and should
    not use this helper.

    Args:
        measurement_indices (dict[int, list[int]]): Output from
            ``measurement_indices_by_wire``.
        wire_terms (Iterable[tuple[int, int]]): ``(wire, coefficient)`` terms.
        use_last_measurement (bool): Whether to select the last true
            measurement on each wire. The keyword is explicit so call sites
            document that they are resolving terminal readout events.

    Returns:
        tuple[list[int], list[int]]: Resolved global measurement indices and
        their matching coefficients.

    Raises:
        KeyError: If a required wire has no true measurement in the circuit.
    """
    resolved_indices: list[int] = []
    resolved_coefficients: list[int] = []

    for wire, coefficient in wire_terms:
        wire_measurements = measurement_indices[int(wire)]
        resolved_indices.append(
            wire_measurements[-1] if use_last_measurement
            else wire_measurements[0]
        )
        resolved_coefficients.append(int(coefficient))

    return resolved_indices, resolved_coefficients
