"""Benchmark DEM compile/sample time against direct SDIM simulation."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sdim.circuit_io import read_circuit
from sdim.program import Program

from lrb.dem_simulation import (
    build_compact_detector_error_model,
    sample_detector_error_model,
)
from lrb.lrb_simulation import NoiseModelUtils
from lrb.project_paths import DEFAULT_RUNS_ROOT


def parse_int_csv(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_str_csv(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def resolve_run_folder(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.exists():
        return candidate.resolve()

    candidate = DEFAULT_RUNS_ROOT / raw
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve run folder '{raw}' as a path or under "
        f"{DEFAULT_RUNS_ROOT}."
    )


def resolve_circuit_paths(args: argparse.Namespace) -> list[Path]:
    if args.circuit:
        return [Path(path).resolve() for path in args.circuit]

    if args.run_folder is None:
        raise ValueError("Provide --circuit or --run-folder.")

    run_folder = resolve_run_folder(args.run_folder)
    families = parse_str_csv(args.families)
    depth_file_indices = parse_int_csv(args.depth_file_indices)
    paths: list[Path] = []
    for family in families:
        for depth_index in depth_file_indices:
            path = (
                run_folder
                / "experiments"
                / family
                / str(args.clifford_index)
                / str(args.probability_index)
                / f"{depth_index}.chp"
            )
            paths.append(path.resolve())
    return paths


def time_call(function):
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def append_row(
    rows: list[dict[str, object]],
    *,
    circuit_path: Path,
    backend: str,
    phase: str,
    seconds: float,
    repeat: int,
    shots: int | None,
    operation_count: int,
    active_noise_locations: int,
    all_noise_locations: int,
    detector_count: int,
    logical_count: int,
    response_batch_size: int,
) -> None:
    rows.append(
        {
            "circuit": str(circuit_path),
            "backend": backend,
            "phase": phase,
            "shots": "" if shots is None else shots,
            "repeat": repeat,
            "seconds": f"{seconds:.9f}",
            "operation_count": operation_count,
            "active_noise_locations": active_noise_locations,
            "all_noise_locations": all_noise_locations,
            "detector_count": detector_count,
            "logical_count": logical_count,
            "response_batch_size": response_batch_size,
        }
    )


def benchmark_circuit(
    circuit_path: Path,
    *,
    shots_values: list[int],
    repeats: int,
    response_batch_size: int,
    skip_sdim: bool,
) -> list[dict[str, object]]:
    if not circuit_path.exists():
        raise FileNotFoundError(f"Missing circuit file: {circuit_path}")

    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        compile_circuit = read_circuit(str(circuit_path))
        operation_count = len(compile_circuit.operations)
        compile_seconds, model = time_call(
            lambda: build_compact_detector_error_model(
                compile_circuit,
                response_batch_size=response_batch_size,
            )
        )
        append_row(
            rows,
            circuit_path=circuit_path,
            backend="dem",
            phase="compile",
            seconds=compile_seconds,
            repeat=repeat,
            shots=None,
            operation_count=operation_count,
            active_noise_locations=len(model.locations),
            all_noise_locations=model.all_noise_locations,
            detector_count=model.num_detectors,
            logical_count=model.num_logical_observables,
            response_batch_size=response_batch_size,
        )

        for shots in shots_values:
            sample_seconds, _payload = time_call(
                lambda: sample_detector_error_model(
                    model,
                    extra_shots=max(0, shots - 1),
                )
            )
            append_row(
                rows,
                circuit_path=circuit_path,
                backend="dem",
                phase="sample",
                seconds=sample_seconds,
                repeat=repeat,
                shots=shots,
                operation_count=operation_count,
                active_noise_locations=len(model.locations),
                all_noise_locations=model.all_noise_locations,
                detector_count=model.num_detectors,
                logical_count=model.num_logical_observables,
                response_batch_size=response_batch_size,
            )

            if skip_sdim:
                continue

            sdim_circuit = read_circuit(str(circuit_path))
            NoiseModelUtils.ensure_noise_params(sdim_circuit)
            sdim_seconds, _result = time_call(
                lambda: Program(sdim_circuit).simulate(shots=shots)
            )
            append_row(
                rows,
                circuit_path=circuit_path,
                backend="sdim",
                phase="simulate",
                seconds=sdim_seconds,
                repeat=repeat,
                shots=shots,
                operation_count=operation_count,
                active_noise_locations=len(model.locations),
                all_noise_locations=model.all_noise_locations,
                detector_count=model.num_detectors,
                logical_count=model.num_logical_observables,
                response_batch_size=response_batch_size,
            )

    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "circuit",
        "backend",
        "phase",
        "shots",
        "repeat",
        "seconds",
        "operation_count",
        "active_noise_locations",
        "all_noise_locations",
        "detector_count",
        "logical_count",
        "response_batch_size",
    ]
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["circuit"]),
            str(row["backend"]),
            str(row["phase"]),
            str(row["shots"]),
        )
        grouped.setdefault(key, []).append(float(row["seconds"]))

    print("circuit,backend,phase,shots,repeats,median_s,mean_s")
    for key in sorted(grouped):
        values = grouped[key]
        circuit, backend, phase, shots = key
        circuit_label = "/".join(Path(circuit).parts[-5:])
        print(
            f"{circuit_label},{backend},{phase},"
            f"{shots},{len(values)},"
            f"{statistics.median(values):.6f},"
            f"{statistics.mean(values):.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure DEM compile/sample time and direct SDIM simulation time "
            "for detector-annotated .chp circuits."
        )
    )
    parser.add_argument(
        "--circuit",
        action="append",
        default=[],
        help="Circuit .chp path. May be supplied more than once.",
    )
    parser.add_argument(
        "--run-folder",
        default=None,
        help=(
            "Generated run folder path or name under "
            "LRB-experiment-data-slurm."
        ),
    )
    parser.add_argument("--families", default="LRB,LRB_const0,RB")
    parser.add_argument("--clifford-index", type=int, default=0)
    parser.add_argument("--probability-index", type=int, default=0)
    parser.add_argument("--depth-file-indices", default="0")
    parser.add_argument("--shots", default="100,1000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dem-response-batch-size", type=int, default=256)
    parser.add_argument(
        "--skip-sdim",
        action="store_true",
        help="Only measure DEM compile/sample phases.",
    )
    parser.add_argument(
        "--output",
        default="results/dem_vs_sdim_runtime_benchmark.csv",
        help="CSV output path.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.dem_response_batch_size < 1:
        parser.error("--dem-response-batch-size must be at least 1")

    circuit_paths = resolve_circuit_paths(args)
    shots_values = parse_int_csv(args.shots)
    rows: list[dict[str, object]] = []
    for circuit_path in circuit_paths:
        rows.extend(
            benchmark_circuit(
                circuit_path,
                shots_values=shots_values,
                repeats=args.repeats,
                response_batch_size=args.dem_response_batch_size,
                skip_sdim=args.skip_sdim,
            )
        )

    output_path = Path(args.output).resolve()
    write_rows(output_path, rows)
    print_summary(rows)
    print(f"Wrote benchmark rows to {output_path}")


if __name__ == "__main__":
    main()
