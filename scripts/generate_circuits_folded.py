"""Generate all folded-code RB/LRB circuits for a new run folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lrb.experiment_setup import ExperimentSetupConfig, ExperimentSetupManager
from lrb.code_simulation_profiles import (
    CONVENTIONAL_SITES_FOLDED_CODE_NAME,
    CONVENTIONAL_SITES_FOLDED_QUBIT_CODE_NAME,
    DEFAULT_CODE_NAME,
    FOLDED_QUBIT_CODE_NAME,
    SPLIT_UNIFORM_FOLDED_CODE_NAME,
    SPLIT_UNIFORM_FOLDED_QUBIT_CODE_NAME,
)
from lrb.circuit_generator import (
    CIRCUIT_LEVEL_NOISE_MODELS,
    DEPOLARIZING_NOISE_MODEL,
    MIRROR_INVERSE_MODE,
    SUPPORTED_INVERSE_MODES,
    SUPPORTED_NOISE_MODELS,
)


class FoldedCircuitGenerationScript:
    """
    CLI helper for folded-code circuit generation.

    Attributes:
        CODE_NAME (str): Default code profile name used by setup generation.
        ANCILLA_MODE_SINGLE (str): Conventional folded-code stabilizer
            measurement mode where each stabilizer generator has one ancilla
            readout.
        ANCILLA_MODE_SPLIT_UNIF (str): Split-boundary folded-code mode where
            the two nonlocal boundary stabilizers are measured with a local
            syndrome ancilla plus a local relay ancilla. The historical mode
            string is kept as ``split-unif`` for compatibility with existing
            notes and launchers, but the generated run now supports the same
            constant, ``const=0``, and uniform postselection policies as the
            conventional folded profile.
        ANCILLA_MODE_CONV_SITES (str): Two-ancilla mode on the four
            conventional surface-code ancilla sites (q2 at the centre): G1 and
            G4 are read through a data relay with only site-data gates, every
            site read once per round.
        ANCILLA_MODE_TO_CODE_NAME (dict[str, str]): Explicit translation from
            user-facing CLI mode names to stable runtime code-profile names.
        DEFAULT_N_CLIFF (int): Default number of Clifford sequences.
        DEFAULT_DEPTHS (list[int]): Default benchmark depths.
        DEFAULT_N_SHOTS (int): Default number of shots.
        DEFAULT_PROBABILITIES (list[float]): Default probability sweep.
        DEFAULT_CIRCUIT_LEVEL_PROBABILITIES (list[float]): Default SI1000/SD6
            sweep of the two-qudit gate error p. SI1000 assigns measurement
            flips ``5p``, so the grid stays well below ``p = 0.2``.
        DEFAULT_STAB_CHECKS_CONST (list[int]): Default constant-check settings.
        DEFAULT_STAB_CHECKS_UNIF (list[int]): Default uniform-check settings.
        DEFAULT_HOME_FOLDER (str): Default run-root parent folder.
        DEFAULT_LRB_FOLDER_NAME (str): Default experiment folder name.
        DEFAULT_NOISE_MODEL (str): Default circuit-level noise model.

    Methods:
        parse_int_csv(csv_text): Parse integer CSV strings.
        parse_float_csv(csv_text): Parse float CSV strings.
        default_probabilities_for_noise_model(noise_model): Choose the
            implicit probability sweep for the selected noise model.
        resolve_ancilla_mode_config(...): Convert the requested ancilla mode
            into a code profile and check-policy set.
        build_arg_parser(): Build CLI parser with all customization options.
        run(...): Generate circuits and return created run path.
        main(): Parse CLI arguments and execute setup generation.
    """

    CODE_NAME = DEFAULT_CODE_NAME
    ANCILLA_MODE_SINGLE = "single"
    ANCILLA_MODE_SPLIT_UNIF = "split-unif"
    ANCILLA_MODE_CONV_SITES = "conv-sites"
    ANCILLA_MODE_TO_CODE_NAME = {
        ANCILLA_MODE_SINGLE: DEFAULT_CODE_NAME,
        ANCILLA_MODE_SPLIT_UNIF: SPLIT_UNIFORM_FOLDED_CODE_NAME,
        ANCILLA_MODE_CONV_SITES: CONVENTIONAL_SITES_FOLDED_CODE_NAME,
    }
    QUBIT_ANCILLA_MODE_TO_CODE_NAME = {
        ANCILLA_MODE_SINGLE: FOLDED_QUBIT_CODE_NAME,
        ANCILLA_MODE_SPLIT_UNIF: SPLIT_UNIFORM_FOLDED_QUBIT_CODE_NAME,
        ANCILLA_MODE_CONV_SITES: CONVENTIONAL_SITES_FOLDED_QUBIT_CODE_NAME,
    }
    SUPPORTED_QUDIT_DIMENSIONS = (3, 2)
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
    DEFAULT_CIRCUIT_LEVEL_PROBABILITIES = [
        0.0,
        1.0e-4,
        2.0e-4,
        5.0e-4,
        1.0e-3,
        2.0e-3,
        5.0e-3,
        7.5e-3,
        1.0e-2,
        1.25e-2,
        1.5e-2,
        2.0e-2,
        3.0e-2,
        4.0e-2,
        5.0e-2,
    ]
    DEFAULT_STAB_CHECKS_CONST = list(range(23))
    DEFAULT_STAB_CHECKS_UNIF = list(range(1, 23))
    DEFAULT_HOME_FOLDER = str(PROJECT_ROOT)
    DEFAULT_LRB_FOLDER_NAME = "LRB-experiment-data-slurm"
    DEFAULT_NOISE_MODEL = DEPOLARIZING_NOISE_MODEL

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
    def default_probabilities_for_noise_model(cls, noise_model: str) -> list[float]:
        """
        Choose the implicit probability sweep for a noise model.

        The historical folded scripts used one long probability grid for the
        default depolarizing sweeps. The circuit-level models (SI1000, SD6)
        sweep the two-qudit gate error p; SI1000 additionally assigns
        measurement flips ``5p`` and reset flips/resonator idles ``2p``. When
        the user does not pass ``--probabilities`` explicitly, these models
        use a lower grid concentrated around the physically meaningful
        ``10^-4``-to-``5e-2`` range. Explicit probability lists are still
        honored verbatim so custom sweeps remain auditable.

        Args:
            noise_model (str): Canonical setup noise-model name.

        Returns:
            list[float]: Default probability list for that noise model.

        Raises:
            ValueError: If the noise model is not supported by this generator.
        """
        if noise_model in CIRCUIT_LEVEL_NOISE_MODELS:
            return list(cls.DEFAULT_CIRCUIT_LEVEL_PROBABILITIES)
        if noise_model == DEPOLARIZING_NOISE_MODEL:
            return list(cls.DEFAULT_PROBABILITIES)
        supported = ", ".join(SUPPORTED_NOISE_MODELS)
        raise ValueError(
            f"Unsupported noise model '{noise_model}'. "
            f"Supported choices are: {supported}."
        )

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
            default=None,
            help=(
                "Optional comma-separated physical probability sweep. If this "
                "is omitted, depolarizing generation uses the historical "
                "folded default grid, while SI1000 generation uses the lower "
                "recommended grid 0 through 0.05."
            ),
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
        parser.add_argument(
            "--noise-model",
            choices=SUPPORTED_NOISE_MODELS,
            default=cls.DEFAULT_NOISE_MODEL,
            help=(
                "Noise model. 'si1000' and 'sd6' are the moment-scheduled "
                "circuit-level models of Gidney et al. (2021), generalized to "
                "qudits; p is the two-qudit gate error."
            ),
        )
        parser.add_argument(
            "--ancilla-mode",
            choices=tuple(cls.ANCILLA_MODE_TO_CODE_NAME),
            default=cls.ANCILLA_MODE_SINGLE,
            help=(
                "'single' keeps the conventional folded-code stabilizer "
                "measurement circuits. 'split-unif' selects the 11-wire "
                "split-boundary folded mode where boundary "
                "stabilizers use two local ancillas each; this mode requires "
                "a circuit-level --noise-model (si1000 or sd6) and supports "
                "the same const, const=0, and uniform postselection policies "
                "as the single-ancilla mode. 'conv-sites' measures the "
                "split stabilizers on the four conventional ancilla sites "
                "with a data relay (9 wires, circuit-level noise only)."
            ),
        )
        parser.add_argument(
            "--qudit-dimension",
            type=int,
            choices=cls.SUPPORTED_QUDIT_DIMENSIONS,
            default=3,
            help=(
                "3 (default) builds the [[5,1,2]]_3 folded qutrit code used by "
                "all production runs; 2 builds the qubit [[5,1,2]] folded code "
                "with the same circuits, check layouts and noise model."
            ),
        )
        parser.add_argument(
            "--inverse-mode",
            choices=SUPPORTED_INVERSE_MODES,
            default=MIRROR_INVERSE_MODE,
            help=(
                "How each RB/LRB sequence C1..Cm is closed. 'mirror' (the "
                "default, used by all production runs) appends "
                "Cm^-1..C1^-1 with no checks between them; 'compiled' "
                "appends one recovery Clifford equal to (Cm...C1)^-1, as in "
                "standard RB, so depth m has m+1 Cliffords."
            ),
        )
        return parser

    @classmethod
    def resolve_ancilla_mode_config(
        cls,
        ancilla_mode: str,
        noise_model: str,
        stab_checks_const: list[int],
        stab_checks_unif: list[int],
        qudit_dimension: int = 3,
    ) -> tuple[str, list[int], list[int]]:
        """
        Resolve the CLI ancilla mode into a code profile and check policies.

        The split-ancilla geometry exists to test circuit-level noise under a local
        stabilizer-measurement geometry where the two boundary stabilizers use
        coherent syndrome-and-relay readouts instead of one long-range
        ancilla. The stabilizer measurement record produced by that circuit is
        still compatible with all existing postselection policies: constant
        number checks, the special direct-X ``const=0`` circuit family, and
        uniform interval checks. Detector/logical annotations are generated
        for all of those circuit families downstream by the shared circuit
        generator.

        The only mode-specific invariant enforced here is that split mode must
        use a circuit-level model (``si1000`` or ``sd6``), because the
        geometry question is about local circuit-level behavior.

        The conventional single-ancilla mode leaves all user-supplied check
        policies untouched.

        Args:
            ancilla_mode (str): User-facing mode string from ``--ancilla-mode``.
            noise_model (str): Canonical setup noise model.
            stab_checks_const (list[int]): Requested constant-check policies.
            stab_checks_unif (list[int]): Requested uniform-interval policies.

        Returns:
            tuple[str, list[int], list[int]]: Resolved code profile name,
            constant-check policy list, and uniform-check policy list.

        Raises:
            ValueError: If the requested mode is unsupported or violates the
                split-mode circuit-level noise requirement.
        """
        if ancilla_mode not in cls.ANCILLA_MODE_TO_CODE_NAME:
            supported = ", ".join(cls.ANCILLA_MODE_TO_CODE_NAME)
            raise ValueError(
                f"Unsupported ancilla mode '{ancilla_mode}'. "
                f"Supported choices are: {supported}."
            )

        code_name = cls.ANCILLA_MODE_TO_CODE_NAME[ancilla_mode]
        if qudit_dimension == 2:
            code_name = cls.QUBIT_ANCILLA_MODE_TO_CODE_NAME[ancilla_mode]
        elif qudit_dimension != 3:
            raise ValueError(
                f"Unsupported qudit dimension {qudit_dimension}; "
                "choose 3 or 2."
            )
        if ancilla_mode in (cls.ANCILLA_MODE_SPLIT_UNIF,
                            cls.ANCILLA_MODE_CONV_SITES):
            if noise_model not in CIRCUIT_LEVEL_NOISE_MODELS:
                raise ValueError(
                    f"The {ancilla_mode} folded ancilla mode is only defined "
                    "for circuit-level noise. Re-run with --noise-model "
                    "si1000 or sd6."
                )
            return code_name, stab_checks_const, stab_checks_unif

        return code_name, stab_checks_const, stab_checks_unif

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
        noise_model: str,
        ancilla_mode: str = ANCILLA_MODE_SINGLE,
        inverse_mode: str = MIRROR_INVERSE_MODE,
        qudit_dimension: int = 3,
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
            noise_model (str): Circuit-level noise model.
            ancilla_mode (str): Stabilizer-measurement layout mode.
            inverse_mode (str): Sequence closing, ``mirror`` or ``compiled``.
            qudit_dimension (int): 3 for the qutrit code, 2 for the qubit
                code.

        Returns:
            str: Absolute path to the created run directory.

        Raises:
            OSError: Propagated from directory or file writes.
            ValueError: Propagated if setup parameters are invalid.
        """
        code_name, stab_checks_const, stab_checks_unif = (
            cls.resolve_ancilla_mode_config(
                ancilla_mode=ancilla_mode,
                noise_model=noise_model,
                stab_checks_const=stab_checks_const,
                stab_checks_unif=stab_checks_unif,
                qudit_dimension=qudit_dimension,
            )
        )
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
            code_name=code_name,
            noise_model=noise_model,
            inverse_mode=inverse_mode,
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
        if args.probabilities is None:
            probabilities = cls.default_probabilities_for_noise_model(
                args.noise_model
            )
        else:
            probabilities = cls.parse_float_csv(args.probabilities)
        created_path = cls.run(
            custom_name=args.custom_name,
            n_cliff=args.n_cliff,
            depths=cls.parse_int_csv(args.depths),
            n_shots=args.n_shots,
            probabilities=probabilities,
            stab_checks_const=cls.parse_int_csv(args.stab_checks_const),
            stab_checks_unif=cls.parse_int_csv(args.stab_checks_unif),
            home_folder=args.home_folder,
            lrb_folder_name=args.lrb_folder_name,
            noise_model=args.noise_model,
            ancilla_mode=args.ancilla_mode,
            inverse_mode=args.inverse_mode,
            qudit_dimension=args.qudit_dimension,
        )
        print(f"Created folded-code circuits at: {created_path}")


if __name__ == "__main__":
    FoldedCircuitGenerationScript.main()
