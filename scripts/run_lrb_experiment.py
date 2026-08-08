"""CLI entrypoint for executing one probability-index LRB/RB simulation round.

The module now exposes ``LRBRunCoordinator`` so orchestration logic is explicit
and testable. The existing command-line contract is preserved:

    python scripts/run_lrb_experiment.py <run-folder-name> <error-probability-index>
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

LRB_BATCH_SIZE_ENV = "LRB_BATCH_SIZE"

from lrb.code_simulation_profiles import (
    DEFAULT_CODE_NAME,
    CodeSimulationProfileRegistry,
)
from lrb.experiment_setup import ExperimentSetupManager
from lrb.lrb_simulation import (
    DEFAULT_SIM_ENGINE,
    LRBSimulationEngine,
    atomic_write_text,
    open_lrb_checkpoint_store,
)
from lrb.project_paths import DEFAULT_RUNS_ROOT


@dataclass
class LRBRunConfig:
    """
    Runtime configuration for one probability-index LRB job.

    The dataclass stores execution parameters used by the run coordinator to
    dispatch simulation and post-processing for a selected error-probability
    index.

    Attributes:
        batch_size (int): Encoded-shot batch size used for resumable execution.
        filter_trivial_shots (bool): Whether trivial shots are filtered in
            logical-code post-processing.
        logical_dimension (int): Number of logical outcomes used in
            logical-code statistics.
        unpack_func (Callable | None): Optional unpack callback for
            code-specific measurement decoding.
        const0_unpack_func (Callable | None): Optional unpack callback for the
            special direct terminal X-data ``const=0`` protocol.
        simulation_backend (str | None): Optional circuit-sampling backend.

    Methods:
        This class is declarative and intentionally defines no custom methods.
    """
    batch_size: int = 500000
    filter_trivial_shots: bool = False
    logical_dimension: int = 3
    unpack_func: Callable | None = None
    const0_unpack_func: Callable | None = None
    simulation_backend: str | None = None
    profile_name: str = DEFAULT_CODE_NAME


class LRBRunCoordinator:
    """
    Coordinator for one resumable probability-index execution.

    The class reads progress state, invokes the simulation engine for pending
    work, and writes completion markers for batch jobs.

    Attributes:
        simulation_engine (LRBSimulationEngine): Engine used to execute RB/LRB
            workloads.
        config (LRBRunConfig): Runtime configuration for the execution.

    Methods:
        __init__: Initialize coordinator dependencies.
        compute_lrb: Run one probability-index job with progress-file guards.
        load_run_inputs: Parse run metadata and resolve filesystem paths for a
            specific probability index.
        resolve_code_runtime_config: Resolve unpack/runtime settings through
            code simulation profiles.
    """
    def __init__(self,
                 simulation_engine: LRBSimulationEngine,
                 config: LRBRunConfig | None = None) -> None:
        """
            Bind the simulation engine and runtime configuration to this
            coordinator instance.

        Args:
            simulation_engine (LRBSimulationEngine): Engine that executes the
                encoded and physical circuit workloads.
            config (LRBRunConfig | None): Optional runtime override;
                defaults to a new ``LRBRunConfig`` when omitted.

        Returns:
            None: Initializes coordinator state and returns nothing.

        Raises:
            TypeError: Propagated for invalid dependency object usage.
        """
        self.simulation_engine = simulation_engine
        self.config = config if config is not None else LRBRunConfig()

    @staticmethod
    def _validate_stats_csv(
        filename: str,
        *,
        probability: float,
        num_depths: int,
    ) -> None:
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            raise RuntimeError(f"Missing or empty result file: {filename}")
        with open(filename, "r", newline="") as result_file:
            rows = list(csv.reader(result_file))
        expected_labels = (
            "Probability",
            "Fidelity averages",
            "Fidelity Standard Deviations",
            "Rejected Runs",
            "Rejected Standard Deviations",
        )
        if len(rows) != len(expected_labels):
            raise RuntimeError(
                f"Result file {filename} has {len(rows)} rows; expected 5."
            )
        for row_index, (row, label) in enumerate(zip(rows, expected_labels)):
            expected_length = 2 if row_index == 0 else num_depths + 1
            if not row or row[0] != label or len(row) != expected_length:
                raise RuntimeError(
                    f"Malformed result row {row_index} in {filename}."
                )
        observed_probability = float(rows[0][1])
        if not math.isclose(
            observed_probability,
            probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"Result probability {observed_probability} in {filename} "
                f"does not match {probability}."
            )
        for row_index, row in enumerate(rows[1:], start=1):
            for value in row[1:]:
                if value in ("", "None"):
                    if row_index not in (2, 4):
                        raise RuntimeError(
                            f"Undefined non-deviation value in {filename}."
                        )
                    continue
                float(value)

    @staticmethod
    def _sha256_file(filename: str) -> str:
        digest = hashlib.sha256()
        with open(filename, "rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _completed_artifact_paths(
        *,
        stab_checks_const,
        stab_checks_unif,
        error_prob_ind: int,
        lrb_results_folder_path: str,
        rb_results_folder_path: str,
    ) -> dict[str, str]:
        paths = {}
        for policy_name, checks in (
            ("const_check_data", stab_checks_const),
            ("unif_check_data", stab_checks_unif),
        ):
            for check in checks:
                key = (
                    f"LRB/{error_prob_ind}/{policy_name}/"
                    f"{int(check)}.csv"
                )
                paths[key] = os.path.join(
                    lrb_results_folder_path,
                    str(error_prob_ind),
                    policy_name,
                    f"{int(check)}.csv",
                )
        paths[f"RB/{error_prob_ind}.csv"] = os.path.join(
            rb_results_folder_path,
            f"{error_prob_ind}.csv",
        )
        return paths

    @classmethod
    def _completion_manifest_payload(
        cls,
        *,
        stab_checks_const,
        stab_checks_unif,
        probability: float,
        error_prob_ind: int,
        lrb_results_folder_path: str,
        rb_results_folder_path: str,
        partial_progress_folder_path: str,
    ) -> dict:
        current_link = os.path.join(
            partial_progress_folder_path,
            ".checkpoint_current",
        )
        if not os.path.islink(current_link):
            raise RuntimeError(
                f"Missing authoritative checkpoint link: {current_link}"
            )
        current_target = os.readlink(current_link)
        checkpoint_manifest_path = os.path.join(
            partial_progress_folder_path,
            current_target,
            "manifest.json",
        )
        with open(checkpoint_manifest_path, "r") as manifest_file:
            checkpoint_manifest = json.load(manifest_file)
        artifact_paths = cls._completed_artifact_paths(
            stab_checks_const=stab_checks_const,
            stab_checks_unif=stab_checks_unif,
            error_prob_ind=error_prob_ind,
            lrb_results_folder_path=lrb_results_folder_path,
            rb_results_folder_path=rb_results_folder_path,
        )
        artifacts = {}
        for key, filename in sorted(artifact_paths.items()):
            artifacts[key] = {
                "bytes": os.path.getsize(filename),
                "sha256": cls._sha256_file(filename),
            }
        return {
            "schema_version": 1,
            "probability_index": int(error_prob_ind),
            "probability": float(probability),
            "checkpoint_current_target": current_target,
            "checkpoint_configuration_fingerprint": (
                checkpoint_manifest["configuration_fingerprint"]),
            "checkpoint_manifest_sha256": cls._sha256_file(
                checkpoint_manifest_path
            ),
            "artifacts": artifacts,
        }

    @classmethod
    def _write_completion_manifest(cls, **kwargs) -> None:
        progress_folder = kwargs["partial_progress_folder_path"]
        payload = cls._completion_manifest_payload(**kwargs)
        atomic_write_text(
            os.path.join(progress_folder, "completion_manifest.json"),
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
        )

    @classmethod
    def _validate_completion_manifest(cls, **kwargs) -> None:
        progress_folder = kwargs["partial_progress_folder_path"]
        manifest_path = os.path.join(
            progress_folder,
            "completion_manifest.json",
        )
        if not os.path.isfile(manifest_path):
            raise RuntimeError(
                f"Missing completion manifest: {manifest_path}"
            )
        with open(manifest_path, "r") as manifest_file:
            recorded = json.load(manifest_file)
        expected = cls._completion_manifest_payload(**kwargs)
        if recorded != expected:
            raise RuntimeError(
                f"Completion manifest does not match authoritative "
                f"checkpoint/results for probability index "
                f"{kwargs['error_prob_ind']}."
            )

    @classmethod
    def _validate_completed_artifacts(
        cls,
        *,
        stab_checks_const,
        stab_checks_unif,
        probability: float,
        error_prob_ind: int,
        depths,
        lrb_results_folder_path: str,
        rb_results_folder_path: str,
        partial_progress_folder_path: str,
    ) -> None:
        shots_path = os.path.join(
            partial_progress_folder_path,
            "shots_processed.txt",
        )
        if not os.path.isfile(shots_path):
            raise RuntimeError(
                f"Missing final checkpoint counter: {shots_path}"
            )
        with open(shots_path, "r") as shots_file:
            shots_remaining = int(shots_file.read().strip())
        if shots_remaining != 0:
            raise RuntimeError(
                f"Completion requested with {shots_remaining} shots remaining."
            )

        policy_directories = (
            (
                os.path.join(
                    lrb_results_folder_path,
                    str(error_prob_ind),
                    "const_check_data",
                ),
                tuple(int(check) for check in stab_checks_const),
            ),
            (
                os.path.join(
                    lrb_results_folder_path,
                    str(error_prob_ind),
                    "unif_check_data",
                ),
                tuple(int(check) for check in stab_checks_unif),
            ),
        )
        for directory, checks in policy_directories:
            expected_names = {f"{check}.csv" for check in checks}
            actual_names = {
                entry.name
                for entry in os.scandir(directory)
                if entry.is_file(follow_symlinks=False)
                and entry.name.endswith(".csv")
            } if os.path.isdir(directory) else set()
            if actual_names != expected_names:
                raise RuntimeError(
                    f"Result files differ in {directory}: "
                    f"missing={sorted(expected_names - actual_names)}, "
                    f"extra={sorted(actual_names - expected_names)}."
                )
            for filename in sorted(expected_names):
                cls._validate_stats_csv(
                    os.path.join(directory, filename),
                    probability=probability,
                    num_depths=len(depths),
                )

        cls._validate_stats_csv(
            os.path.join(
                rb_results_folder_path,
                f"{error_prob_ind}.csv",
            ),
            probability=probability,
            num_depths=len(depths),
        )

    def compute_lrb(
        self,
        stab_checks_const,
        stab_checks_unif,
        probabilities,
        error_prob_ind,
        num_cliff_seq,
        depths,
        num_shots,
        lrb_experiment_folder_path,
        lrb_const0_experiment_folder_path,
        rb_experiment_folder_path,
        lrb_results_folder_path,
        rb_results_folder_path,
        progress_file_path,
        partial_progress_folder_path,
        ) -> int:
        """
            Execute one probability-index pipeline unless a progress marker
            already indicates completion.

        Args:
            stab_checks_const (list[int]): Constant-check-count policy values.
            stab_checks_unif (list[int]): Uniform-interval policy values.
            probabilities (list[float]): Full probability sweep values.
            error_prob_ind (int): Selected probability index for this run.
            num_cliff_seq (int): Number of Clifford seeds.
            depths (list[int]): Benchmark depth values.
            num_shots (int): Total encoded-shot budget.
            lrb_experiment_folder_path (str): Input folder for encoded
                circuits.
            lrb_const0_experiment_folder_path (str): Input folder for special
                direct terminal X-data ``const=0`` circuits.
            rb_experiment_folder_path (str): Input folder for physical
                circuits.
            lrb_results_folder_path (str): Output folder for encoded results.
            rb_results_folder_path (str): Output folder for physical results.
            progress_file_path (str): Completion marker path for this index.
            partial_progress_folder_path (str): In-progress accumulation
                folder.

        Returns:
            int: ``0`` after successful completion or when already complete.

        Raises:
            OSError: Propagated if progress marker read/write fails.
            RuntimeError: Propagated from simulation-engine execution failures.
            ValueError: Propagated if invalid index/parameter combinations are
                encountered downstream.
        """
        os.makedirs(partial_progress_folder_path, exist_ok=True)
        lock_path = os.path.join(
            partial_progress_folder_path,
            ".worker.lock",
        )
        with open(lock_path, "a+") as lock_file:
            try:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as lock_error:
                raise RuntimeError(
                    f"Another worker already owns probability index "
                    f"{error_prob_ind}; refusing a concurrent update."
                ) from lock_error

            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(
                f"pid={os.getpid()} "
                f"slurm_job_id={os.environ.get('SLURM_JOB_ID', 'local')}\n"
            )
            lock_file.flush()

            error_prob = probabilities[error_prob_ind]
            done = 0
            if os.path.exists(progress_file_path):
                with open(progress_file_path, "r") as progress_reader:
                    done_text = progress_reader.read().strip()
                if done_text in ("0", "1"):
                    done = int(done_text)
                elif done_text:
                    raise RuntimeError(
                        f"Invalid completion marker {progress_file_path}: "
                        f"{done_text!r}."
                    )
            else:
                print(
                    "No progress file found -- starting from the beginning."
                )
                atomic_write_text(progress_file_path, "0")

            if done == 1:
                checkpoint_store, _, _ = open_lrb_checkpoint_store(
                    partial_progress_folder_path=(
                        partial_progress_folder_path),
                    num_shots=num_shots,
                    batch_size=self.config.batch_size,
                    num_cliff_seq=num_cliff_seq,
                    depths=depths,
                    stab_checks_const=stab_checks_const,
                    stab_checks_unif=stab_checks_unif,
                    logical_dimension=self.config.logical_dimension,
                    error_prob_ind=error_prob_ind,
                    error_prob=error_prob,
                    backend=self.config.simulation_backend,
                    runtime_profile=self.config.profile_name,
                    filter_trivial_shots=(
                        self.config.filter_trivial_shots),
                    lrb_experiment_folder_path=(
                        lrb_experiment_folder_path),
                    lrb_const0_experiment_folder_path=(
                        lrb_const0_experiment_folder_path),
                    rb_experiment_folder_path=(
                        rb_experiment_folder_path),
                )
                if checkpoint_store.shots_remaining != 0:
                    raise RuntimeError(
                        "Completion marker is set but the authoritative "
                        f"checkpoint has {checkpoint_store.shots_remaining} "
                        "shots remaining."
                    )
                self._validate_completed_artifacts(
                    stab_checks_const=stab_checks_const,
                    stab_checks_unif=stab_checks_unif,
                    probability=error_prob,
                    error_prob_ind=error_prob_ind,
                    depths=depths,
                    lrb_results_folder_path=lrb_results_folder_path,
                    rb_results_folder_path=rb_results_folder_path,
                    partial_progress_folder_path=(
                        partial_progress_folder_path),
                )
                self._validate_completion_manifest(
                    stab_checks_const=stab_checks_const,
                    stab_checks_unif=stab_checks_unif,
                    probability=error_prob,
                    error_prob_ind=error_prob_ind,
                    lrb_results_folder_path=lrb_results_folder_path,
                    rb_results_folder_path=rb_results_folder_path,
                    partial_progress_folder_path=(
                        partial_progress_folder_path),
                )
                print(
                    f"No more work to be done for error probability "
                    f"{error_prob}; completion artifacts revalidated."
                )
                return 0

            atomic_write_text(progress_file_path, "0")
            self.simulation_engine.run_round(
                stab_checks_const=stab_checks_const,
                stab_checks_unif=stab_checks_unif,
                batch_size=self.config.batch_size,
                error_prob=error_prob,
                error_prob_ind=error_prob_ind,
                num_cliff_seq=num_cliff_seq,
                depths=depths,
                num_shots=num_shots,
                filter_trivial_shots=self.config.filter_trivial_shots,
                lrb_experiment_folder_path=lrb_experiment_folder_path,
                lrb_const0_experiment_folder_path=(
                    lrb_const0_experiment_folder_path),
                rb_experiment_folder_path=rb_experiment_folder_path,
                lrb_results_folder_path=lrb_results_folder_path,
                rb_results_folder_path=rb_results_folder_path,
                partial_progress_folder_path=partial_progress_folder_path,
                unpack_func=self.config.unpack_func,
                const0_unpack_func=self.config.const0_unpack_func,
                logical_dimension=self.config.logical_dimension,
                simulation_backend=self.config.simulation_backend,
                runtime_profile=self.config.profile_name,
            )
            checkpoint_store, _, _ = open_lrb_checkpoint_store(
                partial_progress_folder_path=partial_progress_folder_path,
                num_shots=num_shots,
                batch_size=self.config.batch_size,
                num_cliff_seq=num_cliff_seq,
                depths=depths,
                stab_checks_const=stab_checks_const,
                stab_checks_unif=stab_checks_unif,
                logical_dimension=self.config.logical_dimension,
                error_prob_ind=error_prob_ind,
                error_prob=error_prob,
                backend=self.config.simulation_backend,
                runtime_profile=self.config.profile_name,
                filter_trivial_shots=self.config.filter_trivial_shots,
                lrb_experiment_folder_path=lrb_experiment_folder_path,
                lrb_const0_experiment_folder_path=(
                    lrb_const0_experiment_folder_path),
                rb_experiment_folder_path=rb_experiment_folder_path,
            )
            if checkpoint_store.shots_remaining != 0:
                raise RuntimeError(
                    "Simulation returned before its authoritative checkpoint "
                    f"completed; {checkpoint_store.shots_remaining} shots "
                    "remain."
                )
            self._validate_completed_artifacts(
                stab_checks_const=stab_checks_const,
                stab_checks_unif=stab_checks_unif,
                probability=error_prob,
                error_prob_ind=error_prob_ind,
                depths=depths,
                lrb_results_folder_path=lrb_results_folder_path,
                rb_results_folder_path=rb_results_folder_path,
                partial_progress_folder_path=partial_progress_folder_path,
            )
            self._write_completion_manifest(
                stab_checks_const=stab_checks_const,
                stab_checks_unif=stab_checks_unif,
                probability=error_prob,
                error_prob_ind=error_prob_ind,
                lrb_results_folder_path=lrb_results_folder_path,
                rb_results_folder_path=rb_results_folder_path,
                partial_progress_folder_path=partial_progress_folder_path,
            )
            self._validate_completion_manifest(
                stab_checks_const=stab_checks_const,
                stab_checks_unif=stab_checks_unif,
                probability=error_prob,
                error_prob_ind=error_prob_ind,
                lrb_results_folder_path=lrb_results_folder_path,
                rb_results_folder_path=rb_results_folder_path,
                partial_progress_folder_path=partial_progress_folder_path,
            )
            atomic_write_text(progress_file_path, "1")
            return 0

    @staticmethod
    def load_run_inputs(working_folder_name: str, error_prob_ind: int) -> dict:
        """
            Load all run parameters and paths from a prepared setup folder.

        Args:
            working_folder_name (str): Run folder name under experiment root.
            error_prob_ind (int): Probability index for this execution.

        Returns:
            dict: Parsed run parameters and resolved input/output paths.

        Raises:
            OSError: Propagated from parameter-file reads.
            ValueError: Propagated if file contents cannot be parsed.
        """
        # Build all canonical paths once so downstream calls stay simple.
        working_folder_path = os.path.join(
            str(DEFAULT_RUNS_ROOT),
            working_folder_name,
        )

        partial_progress_folder_path = os.path.join(
            working_folder_path, "progress", str(error_prob_ind)) + "/"
        progress_file_path = partial_progress_folder_path + "done.txt"

        lrb_experiment_folder_path = os.path.join(working_folder_path,
                                                  "experiments", "LRB") + "/"
        lrb_const0_experiment_folder_path = os.path.join(
            working_folder_path,
            "experiments",
            "LRB_const0",
        ) + "/"
        rb_experiment_folder_path = os.path.join(working_folder_path,
                                                 "experiments", "RB") + "/"

        lrb_results_folder_path = os.path.join(working_folder_path, "results",
                                               "LRB") + "/"
        rb_results_folder_path = os.path.join(working_folder_path, "results",
                                              "RB") + "/"

        # Load scalar/list metadata files produced during setup generation.
        depths = [
            int(value) for value in ExperimentSetupManager.fetch_list(
                os.path.join(working_folder_path, "depths.txt"))
        ]
        probabilities = [
            float(value)
            for value in ExperimentSetupManager.fetch_list(
                os.path.join(working_folder_path, "probs.txt"))
        ]

        num_shots = int(ExperimentSetupManager.fetch_single_param(
            os.path.join(working_folder_path, "shots.txt")))
        num_cliff_seq = int(ExperimentSetupManager.fetch_single_param(
            os.path.join(working_folder_path, "num_cliffs.txt")))

        stab_checks_const_raw = ExperimentSetupManager.fetch_list(
            os.path.join(working_folder_path, "check_const.txt"))
        stab_checks_unif_raw = ExperimentSetupManager.fetch_list(
            os.path.join(working_folder_path, "check_unif.txt"))

        # Convert optional check arrays to int lists only when populated.
        stab_checks_const = [] if not stab_checks_const_raw else [
            int(value) for value in stab_checks_const_raw
        ]
        stab_checks_unif = [] if not stab_checks_unif_raw else [
            int(value) for value in stab_checks_unif_raw
        ]

        code_name = ExperimentSetupManager.fetch_single_param(
            os.path.join(working_folder_path, "code_name.txt"))
        # Backward-compatible default for runs generated before code_name.txt.
        if code_name in ("", "0"):
            code_name = DEFAULT_CODE_NAME

        return {
            "stab_checks_const": stab_checks_const,
            "stab_checks_unif": stab_checks_unif,
            "probabilities": probabilities,
            "num_cliff_seq": num_cliff_seq,
            "depths": depths,
            "num_shots": num_shots,
            "code_name": code_name,
            "lrb_experiment_folder_path": lrb_experiment_folder_path,
            "lrb_const0_experiment_folder_path": (
                lrb_const0_experiment_folder_path),
            "rb_experiment_folder_path": rb_experiment_folder_path,
            "lrb_results_folder_path": lrb_results_folder_path,
            "rb_results_folder_path": rb_results_folder_path,
            "progress_file_path": progress_file_path,
            "partial_progress_folder_path": partial_progress_folder_path,
        }

    @staticmethod
    def resolve_code_runtime_config(code_name: str) -> LRBRunConfig:
        """
            Resolve runtime unpack/stat configuration for a configured code.

        Args:
            code_name (str): Logical code identifier from run metadata.

        Returns:
            LRBRunConfig: Runtime config with matching unpack behavior.

        Raises:
            ValueError: If the code identifier is unsupported.
        """
        profile = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
        return LRBRunConfig(
            profile_name=profile.code_name,
            logical_dimension=profile.logical_dimension,
            unpack_func=profile.unpack_func,
            const0_unpack_func=profile.const0_unpack_func,
        )


if __name__ == "__main__":
    # Preserve a minimal CLI interface for cluster and local workflows.
    if len(sys.argv) < 3:
        raise Exception(
            "Not enough parameters! Usage: python scripts/run_lrb_experiment.py "
            "<run-folder-name> <error-probability-index> "
            "[--simulation-backend sdim|dem]"
        )

    _, working_folder_name, error_prob_ind_arg, *extra_args = sys.argv
    error_prob_ind = int(error_prob_ind_arg)
    simulation_backend = None
    if extra_args:
        if (len(extra_args) == 2
                and extra_args[0] == "--simulation-backend"):
            simulation_backend = extra_args[1]
        else:
            raise Exception(
                "Unexpected arguments. Usage: python "
                "scripts/run_lrb_experiment.py <run-folder-name> "
                "<error-probability-index> "
                "[--simulation-backend sdim|dem]"
            )

    inputs = LRBRunCoordinator.load_run_inputs(working_folder_name,
                                               error_prob_ind)

    # Resolve code-specific runtime unpack behavior from the run metadata.
    runtime_cfg = LRBRunCoordinator.resolve_code_runtime_config(
        inputs["code_name"])
    runtime_cfg.simulation_backend = simulation_backend
    batch_size_env = os.environ.get(LRB_BATCH_SIZE_ENV)
    if batch_size_env is not None:
        runtime_cfg.batch_size = int(batch_size_env)
        if runtime_cfg.batch_size < 1:
            raise ValueError(f"{LRB_BATCH_SIZE_ENV} must be at least 1.")
    coordinator = LRBRunCoordinator(simulation_engine=DEFAULT_SIM_ENGINE,
                                    config=runtime_cfg)
    coordinator.compute_lrb(
        stab_checks_const=inputs["stab_checks_const"],
        stab_checks_unif=inputs["stab_checks_unif"],
        probabilities=inputs["probabilities"],
        error_prob_ind=error_prob_ind,
        num_cliff_seq=inputs["num_cliff_seq"],
        depths=inputs["depths"],
        num_shots=inputs["num_shots"],
        lrb_experiment_folder_path=inputs["lrb_experiment_folder_path"],
        lrb_const0_experiment_folder_path=inputs[
            "lrb_const0_experiment_folder_path"],
        rb_experiment_folder_path=inputs["rb_experiment_folder_path"],
        lrb_results_folder_path=inputs["lrb_results_folder_path"],
        rb_results_folder_path=inputs["rb_results_folder_path"],
        progress_file_path=inputs["progress_file_path"],
        partial_progress_folder_path=inputs["partial_progress_folder_path"],
    )
