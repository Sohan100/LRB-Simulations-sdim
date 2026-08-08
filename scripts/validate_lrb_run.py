#!/usr/bin/env python3
"""Deeply validate every probability index in an LRB run folder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lrb.lrb_simulation import atomic_write_text, open_lrb_checkpoint_store
from lrb.project_paths import DEFAULT_RUNS_ROOT
from scripts.run_lrb_experiment import LRBRunCoordinator


def _read_scalar(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Missing metadata file: {path}")
    return path.read_text().strip()


def _done_value(path: Path) -> int:
    if not path.is_file():
        return 0
    value = path.read_text().strip()
    if value not in ("0", "1"):
        raise RuntimeError(f"Invalid completion marker {path}: {value!r}")
    return int(value)


def _expected_result_keys(inputs: dict, probability_indices) -> tuple[set, set]:
    rb_keys = {f"{index}.csv" for index in probability_indices}
    lrb_keys = set()
    for index in probability_indices:
        for check in inputs["stab_checks_const"]:
            lrb_keys.add(
                f"{index}/const_check_data/{int(check)}.csv"
            )
        for check in inputs["stab_checks_unif"]:
            lrb_keys.add(
                f"{index}/unif_check_data/{int(check)}.csv"
            )
    return rb_keys, lrb_keys


def validate_run(
    run_name: str,
    *,
    batch_size: int,
    initialize_fresh: bool,
    require_complete: bool,
    expected_code: str | None,
    expected_noise: str | None,
    strict_single_ancilla_si1000: bool,
) -> tuple[dict, bool]:
    run_path = Path(DEFAULT_RUNS_ROOT, run_name)
    if not run_path.is_dir():
        raise RuntimeError(f"Run folder does not exist: {run_path}")

    inputs = LRBRunCoordinator.load_run_inputs(run_name, 0)
    runtime_config = LRBRunCoordinator.resolve_code_runtime_config(
        inputs["code_name"]
    )
    runtime_config.simulation_backend = "dem"
    runtime_config.batch_size = batch_size
    probabilities = inputs["probabilities"]
    probability_indices = tuple(range(len(probabilities)))
    metadata_errors = []
    if expected_code is not None and inputs["code_name"] != expected_code:
        metadata_errors.append(
            f"code_name={inputs['code_name']!r}, expected {expected_code!r}"
        )
    noise_model = _read_scalar(run_path / "noise_model.txt")
    if expected_noise is not None and noise_model != expected_noise:
        metadata_errors.append(
            f"noise_model={noise_model!r}, expected {expected_noise!r}"
        )
    if strict_single_ancilla_si1000:
        strict_expectations = {
            "code_name": (
                inputs["code_name"],
                "folded_qutrit",
            ),
            "noise_model": (noise_model, "si1000"),
            "num_shots": (inputs["num_shots"], 1_000_000),
            "num_cliff_seq": (inputs["num_cliff_seq"], 30),
            "depths": (
                inputs["depths"],
                [0, 2, 4, 6, 10, 14, 18, 20, 22],
            ),
            "probabilities": (
                probabilities,
                [
                    0.0,
                    0.0001,
                    0.0002,
                    0.0005,
                    0.001,
                    0.002,
                    0.005,
                    0.0075,
                    0.01,
                    0.0125,
                    0.015,
                    0.02,
                    0.03,
                    0.04,
                    0.05,
                ],
            ),
            "stab_checks_const": (
                inputs["stab_checks_const"],
                list(range(23)),
            ),
            "stab_checks_unif": (
                inputs["stab_checks_unif"],
                list(range(1, 23)),
            ),
        }
        for field, (observed, expected) in strict_expectations.items():
            if observed != expected:
                metadata_errors.append(
                    f"{field}={observed!r}, expected {expected!r}"
                )

    index_reports = {}
    for probability_index, probability in enumerate(probabilities):
        index_inputs = LRBRunCoordinator.load_run_inputs(
            run_name,
            probability_index,
        )
        progress_path = Path(
            index_inputs["partial_progress_folder_path"]
        )
        done_path = Path(index_inputs["progress_file_path"])
        report = {
            "probability": probability,
            "status": "INVALID",
            "shots_remaining": None,
            "current_generation": None,
            "error": None,
        }
        try:
            done = _done_value(done_path)
            current_link = progress_path / ".checkpoint_current"
            if not current_link.is_symlink() and not initialize_fresh:
                legacy_counts = (
                    list(progress_path.glob("*_check_data/*.npy"))
                    if progress_path.is_dir()
                    else []
                )
                legacy_shots_path = progress_path / "shots_processed.txt"
                legacy_shots = (
                    int(legacy_shots_path.read_text().strip())
                    if legacy_shots_path.is_file()
                    else inputs["num_shots"]
                )
                if done == 0 and not legacy_counts \
                        and legacy_shots == inputs["num_shots"]:
                    report["status"] = "FRESH_VALID"
                    report["shots_remaining"] = inputs["num_shots"]
                    index_reports[str(probability_index)] = report
                    continue
                raise RuntimeError(
                    "No authoritative checkpoint generation exists and the "
                    "legacy state is not clean/fresh."
                )

            checkpoint_store, _, _ = open_lrb_checkpoint_store(
                partial_progress_folder_path=(
                    index_inputs["partial_progress_folder_path"]),
                num_shots=index_inputs["num_shots"],
                batch_size=batch_size,
                num_cliff_seq=index_inputs["num_cliff_seq"],
                depths=index_inputs["depths"],
                stab_checks_const=index_inputs["stab_checks_const"],
                stab_checks_unif=index_inputs["stab_checks_unif"],
                logical_dimension=runtime_config.logical_dimension,
                error_prob_ind=probability_index,
                error_prob=probability,
                backend="dem",
                runtime_profile=runtime_config.profile_name,
                filter_trivial_shots=runtime_config.filter_trivial_shots,
                lrb_experiment_folder_path=(
                    index_inputs["lrb_experiment_folder_path"]),
                lrb_const0_experiment_folder_path=(
                    index_inputs["lrb_const0_experiment_folder_path"]),
                rb_experiment_folder_path=(
                    index_inputs["rb_experiment_folder_path"]),
            )
            shots_remaining = checkpoint_store.shots_remaining
            report["shots_remaining"] = shots_remaining
            report["current_generation"] = os.path.relpath(
                checkpoint_store.current_generation,
                run_path,
            )
            if done == 1:
                if shots_remaining != 0:
                    raise RuntimeError(
                        f"done=1 with {shots_remaining} shots remaining."
                    )
                LRBRunCoordinator._validate_completed_artifacts(
                    stab_checks_const=index_inputs["stab_checks_const"],
                    stab_checks_unif=index_inputs["stab_checks_unif"],
                    probability=probability,
                    error_prob_ind=probability_index,
                    depths=index_inputs["depths"],
                    lrb_results_folder_path=(
                        index_inputs["lrb_results_folder_path"]),
                    rb_results_folder_path=(
                        index_inputs["rb_results_folder_path"]),
                    partial_progress_folder_path=(
                        index_inputs["partial_progress_folder_path"]),
                )
                LRBRunCoordinator._validate_completion_manifest(
                    stab_checks_const=index_inputs["stab_checks_const"],
                    stab_checks_unif=index_inputs["stab_checks_unif"],
                    probability=probability,
                    error_prob_ind=probability_index,
                    lrb_results_folder_path=(
                        index_inputs["lrb_results_folder_path"]),
                    rb_results_folder_path=(
                        index_inputs["rb_results_folder_path"]),
                    partial_progress_folder_path=(
                        index_inputs["partial_progress_folder_path"]),
                )
                report["status"] = "COMPLETE_VALID"
            elif shots_remaining == 0:
                report["status"] = "RESUMABLE_FINALIZATION"
            else:
                report["status"] = "RESUMABLE_VALID"
        except Exception as validation_error:
            report["error"] = (
                f"{type(validation_error).__name__}: {validation_error}"
            )
        index_reports[str(probability_index)] = report

    statuses = {
        index: report["status"]
        for index, report in index_reports.items()
    }
    complete = all(
        status == "COMPLETE_VALID" for status in statuses.values()
    )
    valid_statuses = {
        "FRESH_VALID",
        "RESUMABLE_VALID",
        "RESUMABLE_FINALIZATION",
        "COMPLETE_VALID",
    }
    valid = not metadata_errors and all(
        status in valid_statuses for status in statuses.values()
    )

    if complete:
        expected_rb, expected_lrb = _expected_result_keys(
            inputs,
            probability_indices,
        )
        rb_root = run_path / "results" / "RB"
        lrb_root = run_path / "results" / "LRB"
        actual_rb = {
            path.relative_to(rb_root).as_posix()
            for path in rb_root.rglob("*.csv")
        } if rb_root.is_dir() else set()
        actual_lrb = {
            path.relative_to(lrb_root).as_posix()
            for path in lrb_root.rglob("*.csv")
        } if lrb_root.is_dir() else set()
        if actual_rb != expected_rb:
            metadata_errors.append(
                "Global RB CSV paths differ: "
                f"missing={sorted(expected_rb - actual_rb)}, "
                f"extra={sorted(actual_rb - expected_rb)}"
            )
        if actual_lrb != expected_lrb:
            metadata_errors.append(
                "Global LRB CSV paths differ: "
                f"missing={sorted(expected_lrb - actual_lrb)}, "
                f"extra={sorted(actual_lrb - expected_lrb)}"
            )
        valid = valid and not metadata_errors

    if require_complete and not complete:
        valid = False
    report = {
        "schema_version": 1,
        "validated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "run_name": run_name,
        "run_path": str(run_path),
        "backend": "dem",
        "batch_size": batch_size,
        "code_name": inputs["code_name"],
        "noise_model": noise_model,
        "strict_single_ancilla_si1000": strict_single_ancilla_si1000,
        "probability_count": len(probabilities),
        "complete": complete,
        "valid": valid,
        "metadata_errors": metadata_errors,
        "resubmittable_indices": [
            int(index)
            for index, status in statuses.items()
            if status in {
                "FRESH_VALID",
                "RESUMABLE_VALID",
                "RESUMABLE_FINALIZATION",
            }
        ],
        "invalid_indices": [
            int(index)
            for index, status in statuses.items()
            if status == "INVALID"
        ],
        "indices": index_reports,
    }
    return report, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--initialize-fresh", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-code")
    parser.add_argument("--expected-noise")
    parser.add_argument(
        "--strict-single-ancilla-si1000",
        action="store_true",
    )
    parser.add_argument("--output")
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        report, valid = validate_run(
            arguments.run_name,
            batch_size=arguments.batch_size,
            initialize_fresh=arguments.initialize_fresh,
            require_complete=arguments.require_complete,
            expected_code=arguments.expected_code,
            expected_noise=arguments.expected_noise,
            strict_single_ancilla_si1000=(
                arguments.strict_single_ancilla_si1000),
        )
    except Exception as validation_error:
        report = {
            "schema_version": 1,
            "validated_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "run_name": arguments.run_name,
            "valid": False,
            "complete": False,
            "fatal_error": (
                f"{type(validation_error).__name__}: {validation_error}"
            ),
        }
        valid = False
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        atomic_write_text(arguments.output, rendered)
    sys.stdout.write(rendered)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
