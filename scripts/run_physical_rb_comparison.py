"""Run an RB-only comparison grid for an existing LRB result folder.

The normal LRB runs store physical RB at the raw SI1000 sweep values. For
normalized comparisons, this script adds a separate physical RB grid where
RB p is paired index-by-index with the displayed LRB coordinate, for example
LRB lambda = 5 * raw_p.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry
from lrb.experiment_setup import ExperimentSetupManager
from lrb.lrb_simulation import NORMAL_RB_SHOTS, LRBSimulationPipeline
from lrb.project_paths import DEFAULT_RUNS_ROOT


def _parse_float_csv(csv_text: str) -> list[float]:
    """Parse a comma-separated float list."""
    return [float(token.strip()) for token in csv_text.split(",")
            if token.strip()]


def _parse_index_csv(csv_text: str, count: int) -> list[int]:
    """Parse an index selector."""
    text = csv_text.strip().lower()
    if text in ("", "all"):
        return list(range(count))
    indices: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value >= count:
            raise ValueError(f"Index {value} is outside [0, {count}).")
        indices.append(value)
    return sorted(set(indices))


def _resolve_run_folder(run_arg: str | None) -> Path:
    """Resolve a run name/path or fall back to the split-unif marker."""
    runs_root = Path(DEFAULT_RUNS_ROOT)
    if run_arg:
        candidate = Path(run_arg)
        if candidate.exists():
            return candidate.resolve()
        candidate = runs_root / run_arg
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Could not resolve run folder '{run_arg}'.")

    marker = runs_root / "working-folder-folded_qutrit_split_unif.txt"
    if not marker.exists():
        raise FileNotFoundError(
            f"Missing {marker}; pass the run name/path explicitly.")
    run_name = marker.read_text().strip()
    if not run_name:
        raise ValueError(f"{marker} is empty.")
    candidate = runs_root / run_name
    if not candidate.exists():
        raise FileNotFoundError(f"Marker points to missing run {candidate}.")
    return candidate.resolve()


def _read_required_list(run_folder: Path, filename: str) -> list[str]:
    """Read one required CSV-row metadata file."""
    path = run_folder / filename
    values = ExperimentSetupManager.fetch_list(str(path))
    if not values:
        raise ValueError(f"Required metadata file is empty or missing: {path}")
    return values


def _comparison_probabilities(
    run_folder: Path,
    probabilities_arg: str | None,
    axis_scale: float,
) -> tuple[list[float], list[float]]:
    """Return raw LRB probabilities and desired RB comparison p-values."""
    raw_probabilities = [
        float(v) for v in _read_required_list(run_folder, "probs.txt")
    ]
    if probabilities_arg:
        rb_probabilities = _parse_float_csv(probabilities_arg)
    else:
        rb_probabilities = [float(axis_scale) * p for p in raw_probabilities]
    if len(rb_probabilities) != len(raw_probabilities):
        raise ValueError(
            "The RB comparison grid must have the same length as probs.txt.")
    return raw_probabilities, rb_probabilities


def _write_probability_grid(run_folder: Path,
                            rb_probabilities: list[float],
                            force: bool) -> Path:
    """Persist the comparison p grid with a consistency guard."""
    out_path = run_folder / "rb_comparison_probs.txt"
    if out_path.exists():
        existing = [
            float(v) for v in ExperimentSetupManager.fetch_list(str(out_path))
        ]
        same = (
            len(existing) == len(rb_probabilities)
            and all(abs(a - b) <= 1e-15
                    for a, b in zip(existing, rb_probabilities))
        )
        if not same and not force:
            raise ValueError(
                f"{out_path} already exists with a different grid; use "
                "--force to replace it.")
    ExperimentSetupManager.write_list(rb_probabilities, str(out_path))
    return out_path


def _read_result_probability(result_path: Path) -> float:
    """Read and validate the probability coordinate in one RB result CSV."""
    with result_path.open(newline="", encoding="utf-8") as result_file:
        first_row = next(csv.reader(result_file), [])
    if (len(first_row) < 2
            or first_row[0].strip().lower() != "probability"):
        raise ValueError(
            f"{result_path} does not start with a Probability row.")
    return float(first_row[1])


def _build_rb_experiments(
    generator: LRBCircuitGenerator,
    depths: list[int],
    probability: float,
    num_clifford_sequences: int,
    seed: int | None,
) -> list[list[object]]:
    """Generate one physical RB experiment table for a fixed probability."""
    experiments: list[list[object]] = []
    for clifford_index in range(num_clifford_sequences):
        if seed is not None:
            random.seed(int(seed) + clifford_index)
        circuits = generator.generate_rb_clifford_sequence(
            depths=depths,
            with_noise=True,
        )
        for circuit in circuits:
            generator.update_noise_param(circuit, probability)
        experiments.append(circuits)
    return experiments


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Add physical RB comparison points to an existing LRB run. By "
            "default RB p_i = axis_scale * raw_lrb_p_i."
        )
    )
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run folder name or path. Defaults to the split-unif working "
            "folder marker."
        ),
    )
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=5.0,
        help="Scale applied to probs.txt when --probabilities is omitted.",
    )
    parser.add_argument(
        "--probabilities",
        help="Explicit comma-separated RB p grid.",
    )
    parser.add_argument(
        "--indices",
        default="all",
        help="Comma-separated probability indices to run, or 'all'.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=NORMAL_RB_SHOTS,
        help="Physical RB shots per Clifford/depth circuit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314159,
        help=(
            "Base seed for reusable RB Clifford samples across probabilities; "
            "set to -1 to leave sampling unseeded."
        ),
    )
    parser.add_argument(
        "--simulation-backend",
        default=os.environ.get("LRB_SIMULATION_BACKEND"),
        choices=("sdim", "dem", None),
        help="Simulation backend. Defaults to env/backend resolver.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing comparison grid/results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned grid and exit without simulating.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the RB-only comparison workflow."""
    args = build_arg_parser().parse_args(argv)
    run_folder = _resolve_run_folder(args.run)

    depths = [int(v) for v in _read_required_list(run_folder, "depths.txt")]
    num_clifford_sequences = int(float(
        ExperimentSetupManager.fetch_single_param(
            str(run_folder / "num_cliffs.txt"))))
    code_name = ExperimentSetupManager.fetch_single_param(
        str(run_folder / "code_name.txt"))
    noise_model = ExperimentSetupManager.fetch_single_param(
        str(run_folder / "noise_model.txt"))
    _, rb_probabilities = _comparison_probabilities(
        run_folder=run_folder,
        probabilities_arg=args.probabilities,
        axis_scale=args.axis_scale,
    )
    selected_indices = _parse_index_csv(args.indices, len(rb_probabilities))

    result_root = run_folder / "results" / "RB_comparison"
    progress_root = run_folder / "progress_rb_comparison"
    probability_grid_path = run_folder / "rb_comparison_probs.txt"

    print(f"Run folder: {run_folder}")
    print(f"RB comparison grid: {probability_grid_path}")
    print(f"Results: {result_root}")
    print(f"Depths: {depths}")
    print(f"Num Clifford sequences: {num_clifford_sequences}")
    print(f"Shots per physical RB circuit: {args.shots}")
    print(f"Simulation backend: {args.simulation_backend or 'default'}")
    print("Selected indices:")
    for index in selected_indices:
        print(f"  {index}: p_RB={rb_probabilities[index]:.12g}")
    if args.dry_run:
        return 0

    result_root.mkdir(parents=True, exist_ok=True)
    progress_root.mkdir(parents=True, exist_ok=True)
    _write_probability_grid(
        run_folder, rb_probabilities, force=bool(args.force))

    profile = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
    generator = LRBCircuitGenerator(
        code_definition=profile.code_definition,
        noise_model=noise_model,
    )
    seed = None if int(args.seed) < 0 else int(args.seed)

    for index in selected_indices:
        probability = rb_probabilities[index]
        out_csv = result_root / f"{index}.csv"
        if out_csv.exists() and not args.force:
            stored_probability = _read_result_probability(out_csv)
            if abs(stored_probability - probability) > max(
                    1e-15,
                    1e-12 * max(abs(stored_probability), abs(probability)),
            ):
                raise ValueError(
                    f"{out_csv} stores p_RB={stored_probability:.12g}, but "
                    f"this matched grid requires p_RB={probability:.12g}. "
                    "Use --force to regenerate it."
                )
            print(
                f"[SKIP] {out_csv} already exists with matched "
                f"p_RB={probability:.12g}."
            )
            continue

        progress_dir = progress_root / str(index)
        progress_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        print(f"[RUN] RB comparison index {index}, p_RB={probability:.12g}")

        experiments = _build_rb_experiments(
            generator=generator,
            depths=depths.copy(),
            probability=probability,
            num_clifford_sequences=num_clifford_sequences,
            seed=seed,
        )
        measurement_record = LRBSimulationPipeline.RB(
            experiments=experiments,
            depths=depths.copy(),
            shots=int(args.shots),
            simulation_backend=args.simulation_backend,
            partial_progress_folder=str(progress_dir),
            timing_phase="RB_comparison",
            probability_index=index,
            probability=probability,
        )
        fidelity_stats, _, rejected_stats, _ = (
            LRBSimulationPipeline.extract_statistics(
                measurement_record,
                dimension=profile.code_definition.dimension,
            )
        )
        LRBSimulationPipeline.write_stats(
            str(out_csv),
            probability,
            fidelity_stats,
            rejected_stats,
        )
        elapsed = time.perf_counter() - start
        (progress_dir / "done.txt").write_text("1\n")
        (progress_dir / "elapsed_seconds.txt").write_text(f"{elapsed:.6f}\n")
        print(f"[OK] wrote {out_csv} in {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
