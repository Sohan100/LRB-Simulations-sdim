"""Experiment-setup entrypoint for code-configurable LRB studies.

This module exposes an object-oriented setup workflow through
``ExperimentSetupManager`` and ``ExperimentSetupConfig``. The manager prepares
run directories, writes parameter files, and generates all RB/LRB circuits.
"""

from __future__ import annotations

import csv
import datetime
import os
import sys
from dataclasses import dataclass, field

from code_simulation_profiles import (
    DEFAULT_CODE_NAME,
    CodeSimulationProfileRegistry,
)
from circuit_generator import LRBCodeDefinition, LRBCircuitGenerator


@dataclass
class ExperimentSetupConfig:
    """
    Configuration for one RB/LRB setup-generation run.

    The dataclass stores all setup-time parameters needed to create run
    directories, write parameter files, and pre-generate circuits.

    Attributes:
        n_cliff (int): Number of random Clifford sequences to generate.
        depths (list[int]): Benchmark depths for RB/LRB circuit families.
        n_shots (int): Total encoded-shot budget written to run metadata.
        probabilities (list[float]): Physical error probabilities to sweep.
        stab_checks_constant_numbers (list[int]): Constant-number postselection
            check settings.
        stab_checks_uniform_interval_size (list[int]): Uniform-interval
            postselection check settings.
        home_folder (str): Root directory containing all run folders.
        lrb_folder_name (str): Name of the top-level experiment directory.

    Methods:
        This class is declarative and intentionally defines no custom methods.
    """
    n_cliff: int = 30
    depths: list[int] = field(
        default_factory=lambda: [0, 2, 4, 6, 10, 14, 18, 20, 22])
    n_shots: int = int(1e6)
    probabilities: list[float] = field(default_factory=lambda: [
        3.35981829e-05,
        6.15848211e-04,
        1.12883789e-02,
        2.06130785e-02,
        2.33572147e-02,
        3.11537409e-02,
        3.62021775e-02,
        4.20687089e-02,
        4.83293024e-02,
        5.47144504e-02,
        6.35808794e-02,
        7.38841056e-02,
        8.58569606e-02,
        9.25524149e-02,
        1.00000000e-01,
        1.43844989e-01,
        2.06913808e-01,
        3.35981829e-01,
    ])
    stab_checks_constant_numbers: list[int] = field(
        default_factory=lambda: list(range(23)))
    stab_checks_uniform_interval_size: list[int] = field(
        default_factory=lambda: list(range(1, 23)))
    home_folder: str = "./"
    lrb_folder_name: str = "LRB-experiment-data-slurm"
    code_name: str = DEFAULT_CODE_NAME


class ExperimentSetupManager:
    """
    Orchestrator for end-to-end RB/LRB experiment setup.

    The manager creates the run folder tree, persists configuration metadata,
    initializes progress files, and dispatches circuit generation.

    Attributes:
        config (ExperimentSetupConfig): Setup parameters for the run.
        generator (LRBCircuitGenerator): Circuit generator used to materialize
            RB/LRB experiments.

    Methods:
        __init__: Bind setup configuration and circuit generator dependencies.
        get_working_folder: Resolve the currently active run path from marker
            metadata.
        write_single_param/fetch_single_param: Read or write scalar parameter
            files.
        write_list/fetch_list: Read or write one-row CSV list files.
        _build_run_name: Construct timestamped run-folder names.
        _prepare_run_tree: Create the full run directory hierarchy.
        _initialize_progress_files: Initialize per-probability progress files.
        _write_run_parameters: Persist run parameters used by runtime jobs.
        _write_instructions: Write run command guidance.
        run_setup: Execute end-to-end setup and return the created run path.
    """
    def __init__(self,
                 config: ExperimentSetupConfig,
                 generator: LRBCircuitGenerator | None = None,
                 code_definition: LRBCodeDefinition | None = None) -> None:
        """
            Initialize manager state with configuration and optional generator.

        Args:
            config (ExperimentSetupConfig): Setup parameter container.
            generator (LRBCircuitGenerator | None): Optional circuit generator
                dependency.

        Returns:
            None: Constructor sets internal state and returns nothing.

        Raises:
            TypeError: Propagated if invalid dependency objects are provided.
        """
        self.config = config
        # Keep dependency wiring explicit: caller can pass a generator or a
        # raw code definition, or rely on profile resolution.
        if generator is not None and code_definition is not None:
            raise ValueError(
                "Pass either generator or code_definition, not both.")
        if generator is not None:
            self.generator = generator
        elif code_definition is not None:
            self.generator = LRBCircuitGenerator(
                code_definition=code_definition)
        else:
            # Resolve default generator hooks from the configured code name.
            profile = CodeSimulationProfileRegistry.resolve_code_profile(
                self.config.code_name)
            self.generator = LRBCircuitGenerator(
                code_definition=profile.code_definition)

    @staticmethod
    def get_working_folder(working_folder_filepath: str,
                           lrb_root: str) -> str | None:
        """
            Read working-folder metadata and return the current run path when
            available.

        Args:
            working_folder_filepath (str): Path to working-folder marker file.
            lrb_root (str): Root directory containing run subfolders.

        Returns:
            str | None: Current run path if marker exists, else ``None``.

        Raises:
            OSError: Propagated if marker-file read fails.
        """
        if os.path.exists(working_folder_filepath):
            with open(working_folder_filepath, "r") as read_folder:
                current_run_name = read_folder.readline().strip()
            # Keep returned path format consistent with legacy scripts.
            current_run_path = os.path.join(lrb_root, current_run_name) + "/"
            print("Working folder found.")
            return current_run_path

        print("No working folder found.")
        return None

    @staticmethod
    def write_single_param(param, filepath: str) -> None:
        """
            Persist one scalar parameter as a plain-text file.

        Args:
            param (Any): Scalar value to serialize.
            filepath (str): Destination path for the parameter file.

        Returns:
            None: Writes data to disk and returns nothing.

        Raises:
            OSError: Propagated if file write fails.
        """
        with open(filepath, "w") as writer:
            writer.write(str(param))

    @staticmethod
    def fetch_single_param(filepath: str) -> str:
        """
            Read one scalar parameter from disk and fall back to ``'0'`` if the
            file does not exist.

        Args:
            filepath (str): Source path for the scalar parameter file.

        Returns:
            str: File contents as a stripped string, or ``'0'`` if missing.

        Raises:
            OSError: Propagated if file read fails.
        """
        if os.path.exists(filepath):
            with open(filepath, "r") as reader:
                return reader.readline().strip()
        print("No file found.")
        return "0"

    @staticmethod
    def write_list(values: list, filepath: str) -> None:
        """
            Write a list of values to disk as a single CSV row.

        Args:
            values (list): Sequence of values to serialize.
            filepath (str): Destination path for the CSV file.

        Returns:
            None: Writes data to disk and returns nothing.

        Raises:
            OSError: Propagated if CSV write fails.
        """
        with open(filepath, "w", newline="") as writer:
            csv.writer(writer).writerow(values)

    @staticmethod
    def fetch_list(file_path: str) -> list[str]:
        """
            Read a one-row CSV list file and return parsed string elements.

        Args:
            file_path (str): Source CSV file path.

        Returns:
            list[str]: Parsed list entries; empty list if file is missing.

        Raises:
            OSError: Propagated if file read fails.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                content = file.read()
                values = content.strip().split(",")
                return [] if values == [""] else values
        print("No list found.")
        return []

    @staticmethod
    def _build_run_name(code_name: str, custom_name: str) -> str:
        """
            Build a timestamped run name that includes the configured code name
            and optionally a custom suffix.

        Args:
            code_name (str): Code identifier embedded in the generated run
                folder name.
            custom_name (str): Optional user-provided suffix.

        Returns:
            str: Generated run-folder name.

        Raises:
            ValueError: Not raised directly by this method.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        base_name = f"Run-{timestamp}-{code_name}"
        if custom_name:
            return f"{base_name}-{custom_name}"
        return base_name

    def _prepare_run_tree(self, custom_name: str) -> dict[str, str]:
        """
            Create run-root directories and return a map of canonical
            paths used by setup and runtime stages.

        Args:
            custom_name (str): Optional suffix for generated run name.

        Returns:
            dict[str, str]: Dictionary of resolved run and subfolder paths.

        Raises:
            OSError: Propagated if directory creation or metadata write fails.
        """
        lrb_root = os.path.join(self.config.home_folder,
                                self.config.lrb_folder_name)
        os.makedirs(lrb_root, exist_ok=True)

        run_name = self._build_run_name(self.config.code_name, custom_name)
        run_path = os.path.join(lrb_root, run_name)
        os.makedirs(run_path, exist_ok=True)

        # Update both legacy and code-specific run markers for script lookup.
        working_folder_file = os.path.join(lrb_root, "working-folder.txt")
        code_working_folder_file = os.path.join(
            lrb_root,
            f"working-folder-{self.config.code_name}.txt",
        )
        self.write_single_param(run_name, working_folder_file)
        self.write_single_param(run_name, code_working_folder_file)

        paths = {
            "lrb_root": lrb_root,
            "run_name": run_name,
            "run_path": run_path,
            "experiments": os.path.join(run_path, "experiments"),
            "experiments_lrb": os.path.join(run_path, "experiments", "LRB"),
            "experiments_rb": os.path.join(run_path, "experiments", "RB"),
            "results": os.path.join(run_path, "results"),
            "results_lrb": os.path.join(run_path, "results", "LRB"),
            "results_rb": os.path.join(run_path, "results", "RB"),
            "progress": os.path.join(run_path, "progress"),
        }

        for key in (
                "experiments",
                "experiments_lrb",
                "experiments_rb",
                "results",
                "results_lrb",
                "results_rb",
                "progress",
        ):
            os.makedirs(paths[key], exist_ok=True)

        # Pre-create per-probability result folders consumed by workers.
        for prob_index in range(len(self.config.probabilities)):
            os.makedirs(os.path.join(paths["results_lrb"], str(prob_index)),
                        exist_ok=True)

        return paths

    def _initialize_progress_files(self, run_path: str,
                                   progress_path: str) -> None:
        """
            Ensure each probability index has a progress subfolder and ``done``
            marker file initialized.

        Args:
            run_path (str): Absolute run-folder path (retained for parity).
            progress_path (str): Progress-root path for this run.

        Returns:
            None: Writes progress files and returns nothing.

        Raises:
            OSError: Propagated if file or folder creation fails.
        """
        for prob_index in range(len(self.config.probabilities)):
            prob_progress_folder = os.path.join(progress_path, str(prob_index))
            os.makedirs(prob_progress_folder, exist_ok=True)
            # Each probability gets a "done" marker for resumable jobs.
            done_file = os.path.join(prob_progress_folder, "done.txt")
            if not os.path.exists(done_file):
                self.write_single_param(0, done_file)

    def _write_run_parameters(self, run_path: str) -> None:
        """
            Write all scalar and list parameter files consumed by runtime jobs.

        Args:
            run_path (str): Run-folder path where parameter files are written.

        Returns:
            None: Writes files and returns nothing.

        Raises:
            OSError: Propagated if parameter-file writes fail.
        """
        self.write_list(self.config.probabilities,
                        os.path.join(run_path, "probs.txt"))
        self.write_list(self.config.depths,
                        os.path.join(run_path, "depths.txt"))
        self.write_single_param(self.config.n_shots,
                                os.path.join(run_path, "shots.txt"))
        self.write_single_param(self.config.n_cliff,
                                os.path.join(run_path, "num_cliffs.txt"))
        self.write_list(self.config.stab_checks_constant_numbers,
                        os.path.join(run_path, "check_const.txt"))
        self.write_list(self.config.stab_checks_uniform_interval_size,
                        os.path.join(run_path, "check_unif.txt"))
        self.write_single_param(self.config.code_name,
                                os.path.join(run_path, "code_name.txt"))

    @staticmethod
    def _write_instructions(run_path: str) -> None:
        """
            Write quick-start runtime instructions into the run directory.

        Args:
            run_path (str): Run-folder path where instructions are written.

        Returns:
            None: Writes instructions file and returns nothing.

        Raises:
            OSError: Propagated if instruction-file write fails.
        """
        with open(os.path.join(run_path, "run_instructions.txt"),
                  "w") as script_writer:
            script_writer.writelines([
                "If you want to run a test for a specific probability, "
                "invoke the following:\n",
                "python3 run_lrb_experiment.py "
                "<experiment-folder-name> "
                "<error-probability-index>\n",
                "Though you likely want to use a slurm script to run all "
                "of these at once!",
            ])

    def run_setup(self, custom_name: str = "") -> str:
        """
            Execute full setup workflow: directory creation, parameter writes,
            circuit generation, and instruction generation.

        Args:
            custom_name (str): Optional suffix appended to
                timestamped run name.

        Returns:
            str: Absolute path to the created run directory.

        Raises:
            OSError: Propagated from filesystem operations.
            ValueError: Propagated from downstream circuit-generation failures.
        """
        paths = self._prepare_run_tree(custom_name)
        self._initialize_progress_files(paths["run_path"], paths["progress"])
        self._write_run_parameters(paths["run_path"])

        self.generator.generate_tests(
            num_clifford_sequences=self.config.n_cliff,
            lrb_experiment_folder_path=paths["experiments_lrb"],
            rb_experiment_folder_path=paths["experiments_rb"],
            depths=self.config.depths,
            probabilities=self.config.probabilities,
        )
        print("Finished test generation.")

        # Save helper instructions after all run artifacts exist.
        self._write_instructions(paths["run_path"])
        return paths["run_path"]

if __name__ == "__main__":
    custom_name = sys.argv[1] if len(sys.argv) >= 2 else ""
    manager = ExperimentSetupManager(ExperimentSetupConfig())
    created_path = manager.run_setup(custom_name=custom_name)
    print(f"Setup completed at: {created_path}")



