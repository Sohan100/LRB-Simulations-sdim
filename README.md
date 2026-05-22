# Logical Randomized Benchmarking - Detection Only (LRB-D)

This repository contains the active LRB-D simulation code and saved result
outputs for qutrit error-detection benchmarks built on `sdim`.

The public repository is intentionally scoped to the parts needed to understand,
rerun, and inspect the LRB-D project:

- core Python source under `src/lrb/`
- runnable entrypoints under `scripts/`
- SLURM launch scripts under `slurm/`
- saved run metadata and result outputs under `LRB-experiment-data-slurm/`
- the plotting notebook `Visualize LRB Stab Check Results.ipynb`

Large generated circuit inputs, progress checkpoints, cluster logs, external IQM
viewer material, zip archives, and legacy compatibility files are kept locally
and ignored by Git.

## Project Summary

Logical Randomized Benchmarking (LRB) extends standard randomized benchmarking
from physical qutrit gates to encoded logical operations. In this project, the
main workflow is detection-only LRB (`LRB-D`): stabilizer checks are inserted
during logical benchmark sequences, rejected runs are tracked explicitly, and
logical decay is compared against physical RB.

The current code profiles are:

- `folded_qutrit`: `[[5,1,2]]_3` folded surface error-detection code
- `qgrm_3_1_2`: `[[3,1,2]]_3` QGRM detection code

For qutrits (`d = 3`), fitted decay `f` is converted to average fidelity by:

```text
F = (1 + (d - 1) * f) / d
```

## Repository Layout

```text
src/lrb/
  code_definitions.py          Code circuits, qutrit Clifford gates, checks
  code_simulation_profiles.py  Registry from code_name to code hooks
  circuit_generator.py         Generic RB/LRB circuit generation
  experiment_setup.py          Run-folder setup and circuit export
  lrb_simulation.py            Simulation, postselection, result writing
  lrb_plotting.py              Plotting, fits, threshold summaries

scripts/
  generate_circuits_folded.py  Generate folded-code circuit inputs locally
  generate_circuits_qgrm.py    Generate QGRM circuit inputs locally
  run_lrb_experiment.py        Run one probability index for one run

slurm/
  run_lrb_slurm_folded.sh      Cluster launcher for folded runs
  run_lrb_slurm_qgrm.sh        Cluster launcher for QGRM runs

LRB-experiment-data-slurm/
  Run-.../                     Saved metadata and result outputs
```

## Saved Results

The repository keeps result outputs and run metadata for the current LRB-D
experiments. In each committed run folder, the important public files are:

- `code_name.txt`
- `depths.txt`
- `probs.txt`
- `shots.txt`
- `num_cliffs.txt`
- `check_const.txt`
- `check_unif.txt`
- `run_instructions.txt`
- `results/RB/*.csv`
- `results/LRB/*/*.csv`
- `results/plots/*`

Generated circuit files under `experiments/`, partial progress arrays under
`progress/`, and cluster logs under `logs_job_*/` are local working data. They
are not committed.

## Environment

Recommended runtime:

- Python 3.11
- `sdim`
- `numpy`
- `matplotlib`
- `pandas`
- `jupyter` for notebook plotting

Cluster scripts additionally expect a Bash shell, SLURM, and a Python module
compatible with the local environment.

## Reproducing Runs

Generate circuit inputs locally before running fresh simulations:

```bash
python scripts/generate_circuits_folded.py
python scripts/generate_circuits_qgrm.py
```

Each command creates a run folder under `LRB-experiment-data-slurm/`.

Run a single probability index:

```bash
python scripts/run_lrb_experiment.py <RUN_NAME> <PROB_INDEX>
```

Launch full sweeps on SLURM:

```bash
sbatch slurm/run_lrb_slurm_folded.sh
sbatch slurm/run_lrb_slurm_qgrm.sh
```

The generation scripts support:

- `--custom-name`
- `--n-cliff`
- `--depths`
- `--n-shots`
- `--probabilities`
- `--stab-checks-const`
- `--stab-checks-unif`
- `--home-folder`
- `--lrb-folder-name`

Example:

```bash
python scripts/generate_circuits_folded.py \
  --custom-name testA \
  --n-cliff 40 \
  --depths 0,2,4,8,12 \
  --n-shots 1000000
```

## Plotting

Open and run:

```text
Visualize LRB Stab Check Results.ipynb
```

The notebook uses `LRBResultsPlotter` from `src/lrb/lrb_plotting.py` to build:

- uniform-check summary plots with fits
- constant-check summary plots without fits
- mixed-fit LRB-vs-RB CSV tables
- LRB/RB threshold plots
- pseudo-threshold vs interval-check plots

Typical Python usage:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("src").resolve()))

from lrb.lrb_plotting import (
    LRBPlotFitConfig,
    LRBThresholdConfig,
    LRBResultsPlotter,
)

plotter = LRBResultsPlotter(
    working_folder="./LRB-experiment-data-slurm/Run-.../",
    fit_config=LRBPlotFitConfig(),
)

plotter.plot_all_unif_checks(show=True)
plotter.plot_all_const_checks(show=True)

table_csv = plotter.build_unif_lrb_vs_rb_table_mixed_fits()
summary_csv = plotter.plot_all_unif_threshold_graphs(
    threshold_config=LRBThresholdConfig(),
    table_csv_path=table_csv,
    show=True,
)

plotter.plot_unif_pseudo_thresholds_vs_interval_check(
    check_min=1,
    check_max=4,
    summary_csv_path=summary_csv,
    do_fit=True,
    fit_model="exp",
    show=True,
)
```

## Result CSV Format

`LRBSimulationPipeline.write_stats()` writes five rows:

1. `Probability,<p>`
2. `Fidelity averages,<d0>,<d1>,...`
3. `Fidelity Standard Deviations,<d0>,<d1>,...`
4. `Rejected Runs,<d0>,<d1>,...`
5. `Rejected Standard Deviations,<d0>,<d1>,...`

`LRBSimulationPipeline.read_stats()` expects this same layout.

## Local-Only Material

These paths may exist on a development machine but are intentionally excluded
from Git:

- `LRB-experiment-data-slurm/**/experiments/`
- `LRB-experiment-data-slurm/**/progress/`
- `LRB-experiment-data-slurm/**/logs_job_*/`
- `LRB-experiment-data-slurm/working-folder*.txt`
- `external/`
- `legacy/`
- `artifacts/`
- `.tmp/`
- Python caches and notebook checkpoints

Use `git status --ignored --short` if you need to confirm that local-only files
are being ignored rather than deleted.

## License

See `LICENSE`.
