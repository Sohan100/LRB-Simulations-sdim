"""CLI entrypoint for executing one probability-index LRB/RB simulation round.

The module now exposes ``LRBRunCoordinator`` so orchestration logic is explicit
and testable. The existing command-line contract is preserved:

    python run_lrb_experiment.py <run-folder-name> <error-probability-index>
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

from code_simulation_profiles import (
    DEFAULT_CODE_NAME,
    CodeSimulationProfileRegistry,
)
from experiment_setup import ExperimentSetupManager
from lrb_simulation import (
    DEFAULT_SIM_ENGINE,
    LRBSimulationEngine,
)


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

    Methods:
        This class is declarative and intentionally defines no custom methods.
    """
    batch_size: int = 500000
    filter_trivial_shots: bool = True
    logical_dimension: int = 3
    unpack_func: Callable | None = None


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
        # Honor resumable execution markers so repeated calls are cheap.
        if os.path.exists(progress_file_path):
            with open(progress_file_path, "r") as progress_reader:
                done = int(progress_reader.readline())
            if done == 1:
                print(
                    f"No more work to be done for error probability "
                    f"{probabilities[error_prob_ind]}."
                )
                return 0
        else:
            print("No progress file found -- starting from the beginning.")
            with open(progress_file_path, "w") as progress_writer:
                progress_writer.write("0")

        # Select the one physical-error point handled by this worker.
        error_prob = probabilities[error_prob_ind]

        # Run the full LRB/RB processing round for this probability index.
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
            rb_experiment_folder_path=rb_experiment_folder_path,
            lrb_results_folder_path=lrb_results_folder_path,
            rb_results_folder_path=rb_results_folder_path,
            partial_progress_folder_path=partial_progress_folder_path,
            unpack_func=self.config.unpack_func,
            logical_dimension=self.config.logical_dimension,
        )

        with open(progress_file_path, "w") as progress_writer:
            progress_writer.write("1")

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
        working_folder_path = os.path.join("LRB-experiment-data-slurm",
                                           working_folder_name)

        partial_progress_folder_path = os.path.join(
            working_folder_path, "progress", str(error_prob_ind)) + "/"
        progress_file_path = partial_progress_folder_path + "done.txt"

        lrb_experiment_folder_path = os.path.join(working_folder_path,
                                                  "experiments", "LRB") + "/"
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
            logical_dimension=profile.logical_dimension,
            unpack_func=profile.unpack_func,
        )


if __name__ == "__main__":
    # Preserve a minimal CLI interface for cluster and local workflows.
    if len(sys.argv) < 3:
        raise Exception(
            "Not enough parameters! Usage: python run_lrb_experiment.py "
            "<run-folder-name> <error-probability-index>"
        )

    _, working_folder_name, error_prob_ind_arg = sys.argv
    error_prob_ind = int(error_prob_ind_arg)

    inputs = LRBRunCoordinator.load_run_inputs(working_folder_name,
                                               error_prob_ind)

    # Resolve code-specific runtime unpack behavior from the run metadata.
    runtime_cfg = LRBRunCoordinator.resolve_code_runtime_config(
        inputs["code_name"])
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
        rb_experiment_folder_path=inputs["rb_experiment_folder_path"],
        lrb_results_folder_path=inputs["lrb_results_folder_path"],
        rb_results_folder_path=inputs["rb_results_folder_path"],
        progress_file_path=inputs["progress_file_path"],
        partial_progress_folder_path=inputs["partial_progress_folder_path"],
    )



