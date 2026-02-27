"""Convert generated folded-qutrit .chp circuits into Cirq circuits.

This script recursively scans an input directory for ``.chp`` files, converts
each file to a Cirq circuit using ``sdim.circuit_io``, and writes outputs while
preserving the same folder structure.

Default workflow targets the output of:
``generate_const0_noiseless_x_terminal_check_circuits.py``
"""

from __future__ import annotations

import argparse
import pickle
from functools import lru_cache
from pathlib import Path

import cirq
import numpy as np
from sdim.circuit_io import read_circuit
from sdim.unitary import (
    GeneralizedCNOTGate,
    GeneralizedCNOTGateInverse,
    GeneralizedCZGate,
    GeneralizedCZGateInverse,
    GeneralizedHadamardGate,
    GeneralizedHadamardGateInverse,
    GeneralizedPhaseShiftGate,
    GeneralizedPhaseShiftGateInverse,
    GeneralizedXPauliGate,
    GeneralizedXPauliGateInverse,
    GeneralizedZPauliGate,
    GeneralizedZPauliGateInverse,
    IdentityGate,
)


THIS_FILE = Path(__file__).resolve()
DEFAULT_INPUT_DIR = THIS_FILE.parent / "const_0_noiseless_lrb"
DEFAULT_OUTPUT_DIR = THIS_FILE.parent / "const_0_noiseless_lrb_cirq"


@lru_cache(maxsize=None)
def _two_qudit_identity_gate(dimension: int) -> cirq.MatrixGate:
    """Create an identity gate on two qudits of equal dimension."""
    matrix = np.eye(dimension * dimension, dtype=complex)
    return cirq.MatrixGate(matrix, qid_shape=(dimension, dimension))


@lru_cache(maxsize=None)
def _qudit_swap_gate(dimension: int) -> cirq.MatrixGate:
    """Create a SWAP gate for two qudits of equal dimension."""
    matrix = np.zeros((dimension * dimension, dimension * dimension),
                      dtype=complex)
    for left in range(dimension):
        for right in range(dimension):
            matrix[right * dimension + left, left * dimension + right] = 1.0
    return cirq.MatrixGate(matrix, qid_shape=(dimension, dimension))


def convert_to_cirq(circuit, include_measurements: bool) -> cirq.Circuit:
    """Convert one SDIM Circuit into a Cirq circuit with qutrit gate support."""
    qudits = [
        cirq.LineQid(index, dimension=circuit.dimension)
        for index in range(circuit.num_qudits)
    ]
    gate_map = {
        "I": IdentityGate(circuit.dimension),
        "H": GeneralizedHadamardGate(circuit.dimension),
        "P": GeneralizedPhaseShiftGate(circuit.dimension),
        "CNOT": GeneralizedCNOTGate(circuit.dimension),
        "X": GeneralizedXPauliGate(circuit.dimension),
        "Z": GeneralizedZPauliGate(circuit.dimension),
        "H_INV": GeneralizedHadamardGateInverse(circuit.dimension),
        "P_INV": GeneralizedPhaseShiftGateInverse(circuit.dimension),
        "CNOT_INV": GeneralizedCNOTGateInverse(circuit.dimension),
        "X_INV": GeneralizedXPauliGateInverse(circuit.dimension),
        "Z_INV": GeneralizedZPauliGateInverse(circuit.dimension),
        "CZ": GeneralizedCZGate(circuit.dimension),
        "CZ_INV": GeneralizedCZGateInverse(circuit.dimension),
        "SWAP": _qudit_swap_gate(circuit.dimension),
        "N1": IdentityGate(circuit.dimension),
        "N2": _two_qudit_identity_gate(circuit.dimension),
    }

    cirq_circuit = cirq.Circuit()
    for op_index, operation in enumerate(circuit.operations):
        gate_name = getattr(operation, "name", None)
        if gate_name is None:
            gate_name = getattr(operation, "gate_name")

        if gate_name == "M":
            if include_measurements:
                cirq_circuit.append(
                    cirq.measure(
                        qudits[operation.qudit_index],
                        key=f"m_{operation.qudit_index}_{op_index}",
                    )
                )
            continue

        if gate_name == "RESET":
            cirq_circuit.append(cirq.reset(qudits[operation.qudit_index]))
            continue

        gate = gate_map.get(gate_name)
        if gate is None:
            raise NotImplementedError(f"Gate {gate_name} is not supported.")

        if operation.target_index is None:
            cirq_circuit.append(gate.on(qudits[operation.qudit_index]))
        else:
            cirq_circuit.append(
                gate.on(
                    qudits[operation.qudit_index],
                    qudits[operation.target_index],
                )
            )

    touched_qudits = {
        qudit for op in cirq_circuit.all_operations() for qudit in op.qubits
    }
    for qudit in qudits:
        if qudit not in touched_qudits:
            cirq_circuit.append(IdentityGate(circuit.dimension).on(qudit))
    return cirq_circuit


def convert_file(
    chp_path: Path,
    input_root: Path,
    output_root: Path,
    output_format: str,
    include_measurements: bool,
    overwrite: bool,
) -> tuple[str, Path]:
    """Convert one .chp file and write result in the requested format."""
    circuit = read_circuit(str(chp_path.resolve()))
    cirq_circuit = convert_to_cirq(circuit, include_measurements)

    relative_path = chp_path.relative_to(input_root)
    output_stub = output_root / relative_path

    if output_format == "pickle":
        output_path = output_stub.with_suffix(".cirq.pkl")
        if output_path.exists() and not overwrite:
            return ("skipped", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as writer:
            pickle.dump(cirq_circuit, writer)
        return ("converted", output_path)

    if output_format == "diagram":
        output_path = output_stub.with_suffix(".cirq.txt")
        if output_path.exists() and not overwrite:
            return ("skipped", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagram = cirq_circuit.to_text_diagram(use_unicode_characters=False)
        output_path.write_text(diagram + "\n", encoding="utf-8")
        return ("converted", output_path)

    raise ValueError(f"Unsupported output format: {output_format}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert all .chp circuits under an input tree into Cirq circuits."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Root directory containing .chp files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory for converted Cirq outputs.",
    )
    parser.add_argument(
        "--format",
        choices=("pickle", "diagram"),
        default="pickle",
        help=(
            "Output format: 'pickle' writes .cirq.pkl files; "
            "'diagram' writes ASCII Cirq diagrams in .cirq.txt files."
        ),
    )
    parser.add_argument(
        "--exclude-measurements",
        action="store_true",
        help="If set, do not include M gates in converted Cirq circuits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of .chp files to convert (for testing).",
    )
    return parser


def main() -> None:
    """Parse args and convert all .chp files under the requested input tree."""
    parser = build_arg_parser()
    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    include_measurements = not args.exclude_measurements

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    chp_files = sorted(input_root.rglob("*.chp"))
    if args.limit is not None:
        chp_files = chp_files[: args.limit]

    if not chp_files:
        print(f"No .chp files found under: {input_root}")
        return

    converted_count = 0
    skipped_count = 0

    for index, chp_path in enumerate(chp_files, start=1):
        status, output_path = convert_file(
            chp_path=chp_path,
            input_root=input_root,
            output_root=output_root,
            output_format=args.format,
            include_measurements=include_measurements,
            overwrite=args.overwrite,
        )
        if status == "converted":
            converted_count += 1
        else:
            skipped_count += 1
        print(f"[{index}/{len(chp_files)}] {status}: {output_path}")

    print("Done converting .chp circuits to Cirq.")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Output format: {args.format}")
    print(f"Include measurements: {include_measurements}")
    print(f"Converted: {converted_count}")
    print(f"Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
