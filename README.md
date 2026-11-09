# Logical Randomized Benchmarking - Detection Only (LRB-D)

This repository contains the active LRB-D simulation code and saved result
outputs for qutrit error-detection benchmarks built on `sdim`.

Featured folded-qutrit result, showing a uniform interval-check sweep for
postselection LRB-D on the `[[5,1,2]]_3` folded surface error-detection code:

![Uniform interval postselection LRB-D result for folded [[5,1,2]]_3](LRB-experiment-data-slurm/Run-2026-02-18-19-45-04-folded_qutrit/results/plots/gifs/unif-interval-sweep-pidx04-p2.34e-02.gif)

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
from physical qutrit gates to encoded logical operations.

In standard randomized benchmarking, one samples random Clifford sequences of
depth `m`, appends a Clifford inverse that should return the system to its
starting state, measures an observable, and fits the decay versus sequence
depth. A typical fit model is:

```text
A * f^m + B
```

Here `A` and `B` absorb state-preparation and measurement effects, while `f` is
the decay parameter used to infer the benchmarked average fidelity.

LRB uses the same decay-fitting idea, but each sequence element is lifted to the
logical-code setting. A logical benchmark sequence includes:

1. preparation of an encoded code state,
2. logical Clifford gates,
3. stabilizer-check layers,
4. optional rejection/postselection based on check outcomes,
5. terminal logical measurement.

That makes the measured decay a property of the encoded logical workflow, not
just the underlying physical Clifford gates. The physical RB data in this
repository is kept alongside the logical LRB data so the same physical error
probability sweep can be compared at both levels.

This project focuses on detection-only LRB (`LRB-D`). In this setting, the
stabilizer checks are used to detect faults and reject runs rather than to apply
active correction. The plotting pipeline therefore tracks two coupled pieces of
information:

- logical/physical fidelity decay versus benchmark depth,
- rejected-run statistics versus stabilizer-check strategy and physical error
  probability.

The main check strategies are:

- `const`: a fixed number of stabilizer checks,
- `unif`: uniformly spaced interval checks throughout the sequence.

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
  run_lrb_dem_si1000_all_checks_folded.sh
                               DEM/SI1000 launcher, one ancilla per stabilizer
  run_lrb_rb_comparison_folded.sh
                               Matched physical RB grid for DEM notebook plots

LRB-experiment-data-slurm/
  Run-.../                     Saved metadata and result outputs
```

## Saved Results

The repository keeps result outputs and run metadata for the current LRB-D
experiments. In each committed run folder, the important public files are:

- `code_name.txt`
- `noise_model.txt`
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

Representative committed plot outputs include:

- `results/plots/gifs/unif-interval-sweep-pidx*.gif`
- `results/plots/gifs/const-0-rb-lrb-vs-p.gif`
- `results/plots/unif-<CHECK>-Summary-Graph-Fit.pdf`
- `results/plots/const-<CHECK>-Summary-Graph-NoFit.pdf`
- `results/plots/unif-<CHECK>-error-vs-p-threshold-monotone.pdf`
- `results/plots/unif-<CHECK>-lrb-vs-rb-threshold-monotone.pdf`
- `results/plots/unif-<MIN>-to-<MAX>-pseudo-threshold-vs-interval-check-fit.pdf`

Generated circuit files under `experiments/`, partial progress arrays under
`progress/`, and cluster logs under `logs_job_*/` are local working data. They
are not committed.

## Environment

Recommended runtime:

- Python 3.11
- `sdim>=1.4.0` (for `sdim.dem`, the exact compact qudit detector error model)
- `numpy<2`, `numba`, `cirq-core`, `matplotlib`, `pandas`
- `jupyter` for notebook plotting

On Perlmutter the repo keeps its own virtual environment so sdim 1.4.0 does
not replace the sdim in `~/.local` that other projects use:

```bash
module load python/3.11
python -m venv --system-site-packages .venv-sdim140
PYTHONNOUSERSITE=1 .venv-sdim140/bin/python -m pip install sdim==1.4.0 \
  "cirq-core==1.4.1" "numpy==1.26.4" "pandas==2.1.4" "scipy==1.11.4" \
  "matplotlib==3.8.2" "numba==0.61.2" pytest ipykernel
PYTHONNOUSERSITE=1 .venv-sdim140/bin/python -m ipykernel install --user \
  --name lrb-sdim140 --display-name "LRB sdim 1.4.0 (.venv-sdim140)" \
  --env PYTHONNOUSERSITE 1
```

Run Python with `PYTHONNOUSERSITE=1` so packages in `~/.local` cannot shadow
the venv. Tests run with `PYTHONPATH=src:. .venv-sdim140/bin/python -m pytest tests`.

Cluster scripts additionally expect a Bash shell and SLURM.

## Simulation Model

- **Clifford group.** Random Cliffords are drawn uniformly from the full
  216-element single-qutrit Clifford group, written as one of the 9 Paulis
  `X^a Z^b` times one of 24 symplectic words in `H` and `S`
  (`tests/test_qutrit_clifford_group.py` checks the group against sdim's own
  gates).
- **Noise models.** `si1000` and `sd6` are the circuit-level models of
  Gidney, Newman, Fowler and Broughton (2021), generalized to qudits. The
  swept p is the two-qudit gate depolarizing probability. Each logical
  Clifford and each stabilizer-check round is scheduled into
  as-soon-as-possible moments, and every idle qudit receives idle noise.

  | location | SI1000 | SD6 |
  |---|---|---|
  | two-qudit gate (DEP2 after) | p | p |
  | one-qudit gate (DEP1 after) | p/10 | p |
  | idle qudit in a gate moment (DEP1) | p/10 | p |
  | idle qudit in a measure/reset moment (DEP1) | 2p | p |
  | measurement (result flip before, DEP1 after) | 5p, p | p, p |
  | reset (flip after) | 2p | p |

  Encoded preparation and the terminal logical readout are ideal. Ancillas
  are reset immediately before each check round. The historical
  `depolarizing` model keeps its original placement.
- **Sampling.** The `dem` backend compiles each circuit with
  `sdim.dem.DetectorErrorModel` and samples detector/observable shifts from
  it; `tests/test_dem_backend.py` checks it against sdim's Pauli-frame
  sampler through the LRB unpackers.

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

Launch an SI1000 or SD6 DEM run on one node (all probability indices run
concurrently and resume from checkpoints; submit from the repository root):

```bash
PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/generate_circuits_folded.py \
  --noise-model sd6 --ancilla-mode split-unif --custom-name split-ancilla-sd6-sdim140 \
  --n-cliff 30 --depths 0,2,4,6,10,14,18,20,22 --n-shots 1000000 \
  --probabilities 0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05 \
  --stab-checks-const 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
  --stab-checks-unif 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22

sbatch --export=ALL,RUN_NAME_OVERRIDE=<Run-folder> slurm/run_lrb_dem_circuit_level.sh
```

Use `--noise-model si1000` or `sd6` and `--ancilla-mode single` or
`split-unif` for the four code/noise combinations. Plot them with
`Visualize LRB DEM SI1000 SD6 Results.ipynb`, which places LRB and physical
RB on the same two-qudit gate error axis p.

Launch the historical full sweeps on SLURM:

```bash
sbatch slurm/run_lrb_slurm_folded.sh
sbatch slurm/run_lrb_slurm_qgrm.sh
```

The earlier sdim 1.3.4 single-ancilla SI1000 run (pre-moment-scheduling noise,
206-element Clifford catalog, λ = 5p axis) was produced as follows; its
launchers still expect circuits generated by that code version:

```bash
python3 scripts/generate_circuits_folded.py \
  --noise-model si1000 \
  --ancilla-mode single \
  --custom-name single-ancilla-si1000 \
  --n-cliff 30 \
  --depths 0,2,4,6,10,14,18,20,22 \
  --n-shots 1000000 \
  --probabilities 0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05 \
  --stab-checks-const 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
  --stab-checks-unif 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22

sbatch slurm/run_lrb_dem_si1000_all_checks_folded.sh

# Submit after all one-ancilla LRB probability jobs are complete. This creates
# physical RB points with p_RB = lambda_LRB = 5 * raw LRB p.
sbatch slurm/run_lrb_rb_comparison_folded.sh
```

This matches the split-ancilla DEM production grid and postselection policies;
only the folded-code profile changes to the one-ancilla stabilizer circuits.
The second job creates the RB comparison grid consumed by the DEM notebook.
For example, raw LRB `p=0.005` is displayed as `lambda_LRB=0.025` and is
paired with the physical RB result at `p_RB=0.025`.

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
- `--noise-model`
- `--ancilla-mode`

The default noise model is the historical `depolarizing` generator model.
Use `--noise-model si1000` to generate generalized SI1000 circuit-level noise.
For LRB-D, the non-fault-tolerant encoded state-preparation circuit is kept
ideal so the generated noise isolates the protocol after preparation.
Because SI1000 uses measurement-result noise at rate `5p`, choose a probability
sweep with `p <= 0.2`.

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
