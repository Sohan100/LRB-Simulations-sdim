# Logical Randomized Benchmarking (LRB) Framework

Logical Randomized Benchmarking (LRB) extends standard Randomized
Benchmarking (RB) from physical gates to encoded logical operations.

In standard RB, you sample random Clifford sequences of length `m`, append a
Clifford inverse that should return the state to its start, measure an
observable, and fit decay versus depth. The fit is usually modeled as
`A * f^m + B`, where `A` and `B` absorb state-preparation and measurement
(SPAM) effects, and `f` is the decay parameter of interest.

LRB follows the same idea but includes the full logical stack inside each
sequence element:

1. Encode into the code space (into a specific code word).
2. Apply logical Clifford gates.
3. Run stabilizer checks (optionally with postselection).
4. Perform terminal logical measurement.

That makes LRB a direct measure of encoded logical performance, not just
physical gate quality. In this repository, the workflow focuses on a
postselection-forward LRB style (`LRB-D` style), where rejected-run statistics
are tracked explicitly and logical decay is compared against physical RB.

For qutrits (`d = 3`), the plotting pipeline reports decay `f` and converts it
to average fidelity using:
`F = (1 + (d - 1) * f) / d`.

This project is in the qutrit (`q = 3`) setting and compares:

1. `RB`: physical-level Clifford performance.
2. `LRB`: logical-level performance under a chosen code/check strategy.

The primary built-in logical benchmark target is the
`[[5,1,2]]_3` folded surface error-detection code.

This repository provides a code-configurable framework for:

1. Generating RB and LRB circuit datasets for qutrit codes.
2. Running large simulation sweeps over physical error probabilities.
3. Post-processing and plotting logical/physical benchmarking results.

The current built-in code profiles are:

1. `folded_qutrit` (`[[5,1,2]]_3` folded surface error-detection code).
2. `qgrm_3_1_2` (`[[3,1,2]]_3` QGRM detection code).

## Design Overview

The project is organized so code-specific logic lives in definitions and
profiles, while simulation and generation remain generic:

1. Code-specific circuits and logical operators:
   `code_definitions.py`
2. Mapping from `code_name` to generic hooks:
   `code_simulation_profiles.py`
3. Generic RB/LRB circuit generation engine:
   `circuit_generator.py`
4. Run setup (folders, metadata, circuit export):
   `experiment_setup.py`
5. Runtime simulation executor for one probability index:
   `run_lrb_experiment.py`
6. Simulation and postselection backend:
   `lrb_simulation.py`
7. Plotting and analysis class API:
   `lrb_plotting.py`

## Repository Files

- `code_definitions.py`
  - Qutrit Clifford library.
  - `[[5,1,2]]_3` folded surface-code circuit templates.
  - QGRM `[[3,1,2]]_3` circuit templates.
- `code_simulation_profiles.py`
  - Registry that resolves `code_name` to:
    - `LRBCodeDefinition`
    - logical dimension
    - unpack function
- `circuit_generator.py`
  - Generic `LRBCircuitGenerator`.
  - Generates depth-indexed RB/LRB circuits.
  - Injects N1/N2 depolarizing noise placeholders.
- `experiment_setup.py`
  - Creates run folders and parameter files.
  - Generates all circuit files for all probabilities.
  - Writes working-folder markers.
- `generate_circuits_folded.py`
  - One-command folded-code setup generation.
- `generate_circuits_qgrm.py`
  - One-command QGRM setup generation.
- `run_lrb_experiment.py`
  - Runs one probability-index simulation round.
- `run_lrb_slurm_folded.sh`
  - SLURM launcher for all folded probabilities.
- `run_lrb_slurm_qgrm.sh`
  - SLURM launcher for all QGRM probabilities.
- `lrb_simulation.py`
  - Core simulation, postselection, stats read/write.
- `lrb_plotting.py`
  - Plotting class with:
    - unif-fit plots
    - const no-fit plots
    - mixed-fit CSV tables
    - unif threshold/error-rate graphs
- `Visualize LRB Stab Check Results.ipynb`
  - Output-only notebook that calls `lrb_plotting.py`.

## Prerequisites

Recommended runtime:

1. Python `3.11`.
2. Installed `sdim` package and dependencies.
3. `numpy`, `matplotlib`, and `pandas` for plotting/table workflows.
4. `jupyter` for notebook execution (optional).

Cluster scripts also expect:

1. Bash shell.
2. SLURM scheduler.
3. `module load python/3.11` available.

## End-to-End Workflow

### 1. Generate circuits for a code

Folded:

```bash
python generate_circuits_folded.py
```

QGRM:

```bash
python generate_circuits_qgrm.py
```

Each command creates a new run folder under:
`./LRB-experiment-data-slurm/`

Run name format:
`Run-YYYY-MM-DD-HH-MM-SS-<code_name>[-custom-name]`

### 2. Run simulations

Single probability index (local or manual):

```bash
python run_lrb_experiment.py <RUN_NAME> <PROB_INDEX>
```

Full probability sweep on SLURM:

```bash
sbatch run_lrb_slurm_folded.sh
sbatch run_lrb_slurm_qgrm.sh
```

### 3. Plot and analyze

Open and run notebook:
`Visualize LRB Stab Check Results.ipynb`

The notebook currently does:

1. Uniform-check summary plots (with fits).
2. Constant-check summary plots (no fits).
3. Mixed-fit unif LRB-vs-RB tables.
4. Unif threshold/error-rate plots + summary CSV.

## Working Folder Markers

Setup writes:

1. Legacy marker: `working-folder.txt`
2. Per-code marker:
   - `working-folder-folded_qutrit.txt`
   - `working-folder-qgrm_3_1_2.txt`

SLURM scripts resolve run name in this order:

1. `RUN_NAME_OVERRIDE` (if set).
2. Code-specific marker file.
3. Legacy marker file.
4. Latest `Run-*` fallback.

This allows running different codes back-to-back without marker collision.

## Run Folder Structure

For a run:
`LRB-experiment-data-slurm/Run-...-<code_name>/`

Key contents:

- `depths.txt`, `probs.txt`, `shots.txt`, `num_cliffs.txt`
- `check_const.txt`, `check_unif.txt`, `code_name.txt`
- `experiments/LRB/<cliff_idx>/<prob_idx>/<depth_idx>.chp`
- `experiments/RB/<cliff_idx>/<prob_idx>/<depth_idx>.chp`
- `progress/<prob_idx>/done.txt`
- `results/LRB/<prob_idx>/const_check_data/<check>.csv`
- `results/LRB/<prob_idx>/unif_check_data/<check>.csv`
- `results/RB/<prob_idx>.csv`
- `results/plots/*` (generated by plotting pipeline)

## Configuration Guide

### Circuit generation (Python scripts)

Both generation scripts expose the same CLI args:

- `--custom-name`
- `--n-cliff`
- `--depths` (CSV)
- `--n-shots`
- `--probabilities` (CSV)
- `--stab-checks-const` (CSV)
- `--stab-checks-unif` (CSV)
- `--home-folder`
- `--lrb-folder-name`

Example:

```bash
python generate_circuits_folded.py \
  --custom-name testA \
  --n-cliff 40 \
  --depths 0,2,4,8,12 \
  --n-shots 1000000
```

### Runtime shot controls

- Logical LRB shots come from run metadata (`shots.txt`), which is overwritten
  by `NUM_SHOTS` in SLURM scripts.
- Physical RB shots are currently fixed by:
  `NORMAL_RB_SHOTS = 10000` in `lrb_simulation.py`.

### SLURM controls

In each SLURM script:

- `RUN_NAME_OVERRIDE` sets a specific run.
- `NUM_SHOTS` overrides `shots.txt` before execution.
- `SCRIPTS_DIR` points to script location.
- `EXPECTED_CODE_NAME` guards against wrong run/code pairing.
- `PROBABILITIES` array controls how many tasks launch.

Note:
`PROBABILITIES` in SLURM should match the run's `probs.txt` ordering.

## Plotting API (`lrb_plotting.py`)

### Main classes

1. `LRBPlotFitConfig`
2. `LRBThresholdConfig`
3. `LRBResultsPlotter`

### Typical usage (Python)

```python
from lrb_plotting import (
    LRBPlotFitConfig,
    LRBThresholdConfig,
    LRBResultsPlotter,
)

plotter = LRBResultsPlotter(
    working_folder="./LRB-experiment-data-slurm/Run-.../",
    fit_config=LRBPlotFitConfig(),
)

plotter.plot_all_unif_checks(show=True)      # with fits
plotter.plot_all_const_checks(show=True)     # no fits

table_csv = plotter.build_unif_lrb_vs_rb_table_mixed_fits()
summary_csv = plotter.plot_all_unif_threshold_graphs(
    threshold_config=LRBThresholdConfig(),
    table_csv_path=table_csv,
    show=True,
)
```

### Plot outputs

Uniform fit summary:
- `results/plots/unif-<CHECK>-Summary-Graph-Fit.pdf`

Constant no-fit summary:
- `results/plots/const-<CHECK>-Summary-Graph-NoFit.pdf`

Mixed-fit tables:
- `results/plots/unif_lrb_vs_rb_table_all_mixed_fits.csv`
- `results/plots/unif-<CHECK>-lrb-vs-rb-table-mixed-fits.csv`

Threshold plots:
- `results/plots/unif-<CHECK>-error-vs-p-threshold-monotone.pdf`
- `results/plots/unif-<CHECK>-lrb-vs-rb-threshold-monotone.pdf`
- `results/plots/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`

## Stats CSV Format

`LRBSimulationPipeline.write_stats()` writes 5 rows:

1. `Probability,<p>`
2. `Fidelity averages,<d0>,<d1>,...`
3. `Fidelity Standard Deviations,<d0>,<d1>,...`
4. `Rejected Runs,<d0>,<d1>,...`
5. `Rejected Standard Deviations,<d0>,<d1>,...`

`LRBSimulationPipeline.read_stats()` expects exactly that layout.

## Adding a New Code

### Step 1: Define code circuits

In `code_definitions.py`, add a code class with methods analogous to existing
code classes:

1. `logical_plus_initial_state()`
2. `single_qudit_logical_gate_circuit(operator)`
3. `affected_wires(cliff)`
4. stabilizer measurement circuit builders
5. `reset_measurement_wires()` (or `None` via profile)
6. `terminal_logical_x_measurement_circuit()`
7. optional codespace-check helpers

### Step 2: Register profile

In `code_simulation_profiles.py`, add entries for your `code_name` in:

1. `_CODE_DEFINITION_SPECS`
2. `_UNPACK_SPECS`
3. `_LOGICAL_DIMENSIONS`

### Step 3: Add generation entrypoint (optional but recommended)

Copy one generation script and set:

1. `CODE_NAME`
2. default knobs if needed

### Step 4: Add SLURM launcher (optional)

Copy a SLURM script and set:

1. `EXPECTED_CODE_NAME`
2. `WORKING_FOLDER_FILE` suffix
3. resource requests

## Troubleshooting

### `ModuleNotFoundError: No module named 'sdim'`

Install and activate the environment that provides `sdim` before running any
generation/simulation/plotting command.

### SLURM script says code mismatch

The run folder `code_name.txt` does not match script
`EXPECTED_CODE_NAME`. Use:

1. the matching script, or
2. set `RUN_NAME_OVERRIDE` to the correct run.

### No run found

If markers are missing, SLURM falls back to latest `Run-*`. Confirm your run
exists under `LRB-experiment-data-slurm`.

### Empty plots or insufficient threshold points

Common causes:

1. missing CSVs for some checks/probabilities,
2. overly strict threshold filters,
3. partial run completion.

Try running remaining probability indices and then rerun plotting.

### Probability index confusion

`run_lrb_experiment.py` expects `<PROB_INDEX>` using `probs.txt` order.
Indexing is zero-based.

## Reproducibility Notes

1. Record full generation command line used.
2. Keep `probs.txt`, `depths.txt`, and check lists unchanged per run.
3. Keep SLURM `PROBABILITIES` array aligned with the run metadata.
4. Use `--custom-name` for traceable run IDs.

## License

See `LICENSE`.
