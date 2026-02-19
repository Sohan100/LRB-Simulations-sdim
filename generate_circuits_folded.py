"""Generate all folded-code RB/LRB circuits for a new run folder."""

from __future__ import annotations

import argparse

from experiment_setup import ExperimentSetupConfig, ExperimentSetupManager


class FoldedCircuitGenerationScript:
    """
    CLI helper for folded-code circuit generation.

    Attributes:
        CODE_NAME (str): Code profile name used by setup generation.
        DEFAULT_N_CLIFF (int): Default number of Clifford sequences.
        DEFAULT_DEPTHS (list[int]): Default benchmark depths.
        DEFAULT_N_SHOTS (int): Default number of shots.
        DEFAULT_PROBABILITIES (list[float]): Default probability sweep.
        DEFAULT_STAB_CHECKS_CONST (list[int]): Default constant-check settings.
        DEFAULT_STAB_CHECKS_UNIF (list[int]): Default uniform-check settings.
        DEFAULT_HOME_FOLDER (str): Default run-root parent folder.
        DEFAULT_LRB_FOLDER_NAME (str): Default experiment folder name.

    Methods:
        parse_int_csv(csv_text): Parse integer CSV strings.
        parse_float_csv(csv_text): Parse float CSV strings.
        build_arg_parser(): Build CLI parser with all customization options.
        run(...): Generate circuits and return created run path.
        main(): Parse CLI arguments and execute setup generation.
    """

    CODE_NAME = "folded_qutrit"
    DEFAULT_N_CLIFF = 30
    DEFAULT_DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
    DEFAULT_N_SHOTS = int(1e6)
    DEFAULT_PROBABILITIES = [
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
    ]
    DEFAULT_STAB_CHECKS_CONST = list(range(23))
    DEFAULT_STAB_CHECKS_UNIF = list(range(1, 23))
    DEFAULT_HOME_FOLDER = "./"
    DEFAULT_LRB_FOLDER_NAME = "LRB-experiment-data-slurm"

    @staticmethod
    def parse_int_csv(csv_text: str) -> list[int]:
        """
        Parse a comma-separated integer list.

        Args:
            csv_text (str): Comma-separated integer string.

        Returns:
            list[int]: Parsed integer list.

        Raises:
            ValueError: If any token cannot be parsed as an integer.
        """
        if csv_text.strip() == "":
            return []
        # Parse explicit CSV lists so CLI overrides are easy to audit.
        return [int(token.strip()) for token in csv_text.split(",")]

    @staticmethod
    def parse_float_csv(csv_text: str) -> list[float]:
        """
        Parse a comma-separated float list.

        Args:
            csv_text (str): Comma-separated float string.

        Returns:
            list[float]: Parsed float list.

        Raises:
            ValueError: If any token cannot be parsed as a float.
        """
        if csv_text.strip() == "":
            return []
        # Keep float parsing centralized for consistent CLI behavior.
        return [float(token.strip()) for token in csv_text.split(",")]

    @classmethod
    def build_arg_parser(cls) -> argparse.ArgumentParser:
        """
        Build command-line argument parser with all setup customizations.

        Args:
            None: This method relies on class constants.

        Returns:
            argparse.ArgumentParser: Configured parser instance.

        Raises:
            ValueError: Not raised directly by this method.
        """
        # Expose all setup knobs directly so runs are reproducible from CLI.
        parser = argparse.ArgumentParser(
            description="Generate folded-code RB/LRB circuits."
        )
        parser.add_argument("--custom-name", default="")
        parser.add_argument("--n-cliff", type=int, default=cls.DEFAULT_N_CLIFF)
        parser.add_argument(
            "--depths",
            default=",".join(str(v) for v in cls.DEFAULT_DEPTHS),
        )
        parser.add_argument("--n-shots", type=int, default=cls.DEFAULT_N_SHOTS)
        parser.add_argument(
            "--probabilities",
            default=",".join(str(v) for v in cls.DEFAULT_PROBABILITIES),
        )
        parser.add_argument(
            "--stab-checks-const",
            default=",".join(str(v) for v in cls.DEFAULT_STAB_CHECKS_CONST),
        )
        parser.add_argument(
            "--stab-checks-unif",
            default=",".join(str(v) for v in cls.DEFAULT_STAB_CHECKS_UNIF),
        )
        parser.add_argument("--home-folder", default=cls.DEFAULT_HOME_FOLDER)
        parser.add_argument(
            "--lrb-folder-name",
            default=cls.DEFAULT_LRB_FOLDER_NAME,
        )
        return parser

    @classmethod
    def run(
        cls,
        custom_name: str,
        n_cliff: int,
        depths: list[int],
        n_shots: int,
        probabilities: list[float],
        stab_checks_const: list[int],
        stab_checks_unif: list[int],
        home_folder: str,
        lrb_folder_name: str,
    ) -> str:
        """
        Generate folded-code circuits for one setup run.

        Args:
            custom_name (str): Optional run-name suffix.
            n_cliff (int): Number of Clifford sequences to generate.
            depths (list[int]): Benchmark depths used for generation.
            n_shots (int): Shots written into run metadata.
            probabilities (list[float]): Physical probability sweep.
            stab_checks_const (list[int]): Constant-check policy list.
            stab_checks_unif (list[int]): Uniform-check policy list.
            home_folder (str): Root folder where runs are created.
            lrb_folder_name (str): Experiment folder name under home_folder.

        Returns:
            str: Absolute path to the created run directory.

        Raises:
            OSError: Propagated from directory or file writes.
            ValueError: Propagated if setup parameters are invalid.
        """
        # Bind all CLI parameters into the shared setup config dataclass.
        config = ExperimentSetupConfig(
            n_cliff=n_cliff,
            depths=depths,
            n_shots=n_shots,
            probabilities=probabilities,
            stab_checks_constant_numbers=stab_checks_const,
            stab_checks_uniform_interval_size=stab_checks_unif,
            home_folder=home_folder,
            lrb_folder_name=lrb_folder_name,
            code_name=cls.CODE_NAME,
        )
        # Delegate filesystem setup and circuit generation to the manager.
        manager = ExperimentSetupManager(config=config)
        return manager.run_setup(custom_name=custom_name)

    @classmethod
    def main(cls) -> None:
        """
        Parse CLI arguments and execute folded-code setup generation.

        Args:
            None: Uses command-line input from the active process.

        Returns:
            None: Prints created run path and exits.

        Raises:
            ValueError: Propagated if CLI values are malformed.
            OSError: Propagated from setup filesystem operations.
        """
        # Parse CLI arguments once, then pass through the typed run wrapper.
        parser = cls.build_arg_parser()
        args = parser.parse_args()
        created_path = cls.run(
            custom_name=args.custom_name,
            n_cliff=args.n_cliff,
            depths=cls.parse_int_csv(args.depths),
            n_shots=args.n_shots,
            probabilities=cls.parse_float_csv(args.probabilities),
            stab_checks_const=cls.parse_int_csv(args.stab_checks_const),
            stab_checks_unif=cls.parse_int_csv(args.stab_checks_unif),
            home_folder=args.home_folder,
            lrb_folder_name=args.lrb_folder_name,
        )
        print(f"Created folded-code circuits at: {created_path}")


if __name__ == "__main__":
    FoldedCircuitGenerationScript.main()
