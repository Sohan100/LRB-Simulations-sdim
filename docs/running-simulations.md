# Running simulations

## Environment

Python 3.11 and sdim 1.4.0 (for `sdim.dem`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks]"      # add ,memory for the folded-surface-code memory study
pytest                                  # about 3 minutes; one test skips without production circuits
```

On Perlmutter we keep a repo-local environment so sdim 1.4.0 does not replace the sdim in
`~/.local` that other projects use. Run it with `PYTHONNOUSERSITE=1`:

```bash
module load python/3.11
python -m venv --system-site-packages .venv-sdim140
PYTHONNOUSERSITE=1 .venv-sdim140/bin/python -m pip install -e ".[dev,notebooks]"
PYTHONNOUSERSITE=1 .venv-sdim140/bin/python -m ipykernel install --user \
  --name lrb-sdim140 --display-name "LRB sdim 1.4.0 (.venv-sdim140)" --env PYTHONNOUSERSITE 1
```

The test that regenerates a production run's circuits byte for byte needs that run's `experiments/`
folder, which is not in git. Point `LRB_RUNS_ROOT` at a checkout that has it:

```bash
LRB_RUNS_ROOT=/path/to/LRB-experiment-data-slurm pytest tests/test_compiled_recovery.py
```

## A campaign, start to finish

1. **Generate** the run folder. The production grid is 30 Clifford sequences, depths 0 to 22,
   10^6 shots and 15 values of `p`:

   ```bash
   python scripts/generate_circuits_folded.py \
     --ancilla-mode single-prod5b --qudit-dimension 3 --noise-model si1000 \
     --custom-name single-ancilla-si1000-sdim140 \
     --n-cliff 30 --depths 0,2,4,6,10,14,18,20,22 --n-shots 1000000 \
     --probabilities 0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05 \
     --stab-checks-const 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
     --stab-checks-unif 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
   ```

   `--ancilla-mode` picks the layout (see `layouts/README.md`): `single` and `single-prod5b`
   (one-ancilla layout, seven- and five-moment round), `layout-b` (two-ancilla layout),
   `conv-sites`, `split-unif` and `a1`. `--qudit-dimension 2` gives the qubit code.

2. **Run** every probability index on one node; the job resumes from checkpoints and resubmits
   itself before its time limit:

   ```bash
   sbatch --export=ALL,RUN_NAME_OVERRIDE=<Run-folder> slurm/run_lrb_dem_circuit_level.sh
   ```

   One index alone: `python scripts/run_lrb_experiment.py <Run-folder> <index>`.

3. **Validate**: `python scripts/validate_lrb_run.py <Run-folder> --batch-size 100000
   --require-complete --strict-production-grid`.

4. **High-p extension.** At high `p` with frequent checks almost every shot is rejected after a few
   Cliffords, so the threshold crossings need extension runs (depths 0 to 6, 2 x 10^7 shots, `p`
   above the crossing). `scripts/merge_highp_threshold_extension.py` merges them with the production
   run and writes `LRB-experiment-data-slurm/comparisons/pseudo-thresholds-*-current.{md,csv}`.

5. **Error bars**: `python scripts/sequence_bootstrap_thresholds.py <label>` adds the
   sequence-bootstrap standard error to those tables.

6. **Plot** with the matching notebook in `notebooks/`; each one finds the repository root by itself.

## Historical runs

The February 2026 depolarizing runs (`Run-2026-02-18-*`) and the June to August SI1000 runs came
from sdim 1.3.4 with the earlier noise placement and the `lambda = 5p` axis. Their launchers
(`slurm/run_lrb_slurm_*.sh`, `slurm/run_lrb_*si1000_all_checks*.sh`,
`slurm/run_lrb_rb_comparison_*.sh`) expect circuits made by that code version and are kept for the
record; `slurm/README.md` says which launcher belongs to which run.
