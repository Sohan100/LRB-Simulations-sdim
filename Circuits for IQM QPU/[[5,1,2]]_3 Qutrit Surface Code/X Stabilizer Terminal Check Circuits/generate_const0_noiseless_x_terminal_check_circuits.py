"""Generate noiseless const=0 folded-qutrit LRB circuits for IQM execution.

This script creates folded ``[[5,1,2]]_3`` logical RB circuits with:
- no inserted noise gates
- no ancilla stabilizer-check rounds during the sequence
- terminal checks done directly on data qutrits in X basis

Terminal X-basis data readout (wires 0..4) supports:
    S1 = X0 X1 X2^-1  -> (x0 + x1 - x2) mod 3
    S2 = X1^-1 X3 X4^-1 -> (-x1 + x3 - x4) mod 3
    Logical X = X0 X3 -> (x0 + x3) mod 3
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Locate repository root by probing for core project modules."""
    for candidate in (start, *start.parents):
        if (candidate / "code_definitions.py").exists() and (
            candidate / "circuit_generator.py"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing code_definitions.py "
        "and circuit_generator.py."
    )


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _find_repo_root(THIS_FILE.parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import code_definitions as defns  # noqa: E402
from circuit_generator import (  # noqa: E402
    LRBCircuitGenerator,
    LRBCodeDefinition,
    LRBCodeDefinitionFactory,
)
from sdim.circuit import Circuit  # noqa: E402
from sdim.circuit_io import write_circuit  # noqa: E402


DEFAULT_DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
DEFAULT_NUM_CLIFFORD_SEQUENCES = 30
DEFAULT_SEED = 12345


def parse_int_csv(text: str) -> list[int]:
    """Parse comma-separated integers."""
    if not text.strip():
        return []
    return [int(token.strip()) for token in text.split(",")]


def terminal_x_stabilizer_and_logical_x_measurement() -> Circuit:
    """Measure data qutrits directly in X basis for S1/S2 and logical X."""
    c = Circuit(
        dimension=defns.FoldedQutritDetectionCode.dimension,
        num_qudits=defns.FoldedQutritDetectionCode.num_qudits,
    )
    c.add_gate("H_INV", [0, 1, 2, 3, 4])
    c.add_gate("M", [0, 1, 2, 3, 4])
    return c


def build_const0_x_terminal_code_definition() -> LRBCodeDefinition:
    """Build folded-code definition with terminal-only X stabilizer checks."""
    return LRBCodeDefinition(
        dimension=3,
        physical_num_qudits=1,
        encoded_num_qudits=9,
        clifford_strings=defns.single_qutrit_cliffords_exhaustive_strings,
        clifford_to_gate_sequence=defns.QutritCliffordLibrary.qutrit_gate_seq_list,
        clifford_inverse_map=defns.QutritCliffordLibrary.build_inverse_lookup(),
        apply_physical_gate=(
            LRBCodeDefinitionFactory.apply_single_qudit_gate_to_wire_zero
        ),
        logical_plus_initial_state=(
            defns.FoldedQutritDetectionCode.logical_plus_initial_state
        ),
        logical_gate_circuit=(
            defns.FoldedQutritDetectionCode.single_qudit_logical_gate_circuit
        ),
        affected_wires=defns.FoldedQutritDetectionCode.affected_wires,
        stabilizer_check_blocks=(),
        reset_measurement_wires=None,
        terminal_measurement=terminal_x_stabilizer_and_logical_x_measurement,
        depth_zero_noise_wires=(),
    )


def write_metadata(
    output_root: Path,
    depths: list[int],
    num_clifford_sequences: int,
    seed: int | None,
) -> None:
    """Write human-readable and machine-readable metadata for output circuits."""
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / "depth_index_map.csv").open(
        "w", encoding="utf-8", newline=""
    ) as writer:
        csv_writer = csv.writer(writer)
        csv_writer.writerow(["depth_index", "depth"])
        for index, depth in enumerate(depths):
            csv_writer.writerow([index, depth])

    (output_root / "depths.txt").write_text(
        ",".join(str(depth) for depth in depths) + "\n",
        encoding="utf-8",
    )

    details = [
        "Noiseless const=0 folded-qutrit LRB circuits",
        f"num_clifford_sequences={num_clifford_sequences}",
        f"depths={depths}",
        f"random_seed={seed}",
        "",
        "Terminal X-basis data measurements:",
        "  measured wires: 0,1,2,3,4",
        "  S1 = (x0 + x1 - x2) mod 3",
        "  S2 = (-x1 + x3 - x4) mod 3",
        "  Logical X = (x0 + x3) mod 3",
        "",
        "Output layout:",
        "  <output_root>/<clifford_index>/<depth_index>.chp",
    ]
    (output_root / "terminal_check_definition.txt").write_text(
        "\n".join(details) + "\n",
        encoding="utf-8",
    )


def generate_noiseless_const0_lrb_circuits(
    output_root: Path,
    num_clifford_sequences: int,
    depths: list[int],
    seed: int | None,
) -> None:
    """Generate noiseless const=0 logical-RB folded-qutrit circuits."""
    if not depths:
        raise ValueError("Depth list cannot be empty.")

    sorted_depths = sorted(set(depths))
    if seed is not None:
        random.seed(seed)

    generator = LRBCircuitGenerator(
        code_definition=build_const0_x_terminal_code_definition()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for clifford_index in range(num_clifford_sequences):
        clifford_dir = output_root / str(clifford_index)
        clifford_dir.mkdir(parents=True, exist_ok=True)
        circuits = generator.generate_lrb_clifford_sequence(
            depths=sorted_depths,
            with_noise=False,
        )
        for depth_index, circuit in enumerate(circuits):
            depth = sorted_depths[depth_index]
            write_circuit(
                circuit=circuit,
                output_file=f"{depth_index}.chp",
                comment=f"noiseless,const=0,depth={depth}",
                directory=str(clifford_dir),
            )

    write_metadata(
        output_root=output_root,
        depths=sorted_depths,
        num_clifford_sequences=num_clifford_sequences,
        seed=seed,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    default_out = THIS_FILE.parent / "const_0_noiseless_lrb"
    parser = argparse.ArgumentParser(
        description=(
            "Generate noiseless const=0 folded [[5,1,2]]_3 LRB circuits "
            "with terminal X-stabilizer checks measured on data qutrits."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_out),
        help="Output directory for generated .chp circuits.",
    )
    parser.add_argument(
        "--num-clifford-sequences",
        type=int,
        default=DEFAULT_NUM_CLIFFORD_SEQUENCES,
        help="Number of random Clifford sequences to generate.",
    )
    parser.add_argument(
        "--depths",
        default=",".join(str(depth) for depth in DEFAULT_DEPTHS),
        help="Comma-separated logical RB depths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic Clifford sampling.",
    )
    return parser


def main() -> None:
    """Parse CLI args and generate circuits."""
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    depths = parse_int_csv(args.depths)

    generate_noiseless_const0_lrb_circuits(
        output_root=output_dir,
        num_clifford_sequences=args.num_clifford_sequences,
        depths=depths,
        seed=args.seed,
    )

    print("Generated noiseless const=0 folded-qutrit LRB circuits.")
    print(f"Output directory: {output_dir}")
    print("Terminal measured data wires (X basis): [0, 1, 2, 3, 4]")
    print("S1 = (x0 + x1 - x2) mod 3")
    print("S2 = (-x1 + x3 - x4) mod 3")
    print("Logical X = (x0 + x3) mod 3")


if __name__ == "__main__":
    main()
