# LRB-D at larger local dimension

**Questions.** What happens to the LRB-D pseudo-threshold as the local dimension q grows? Up to which postselection interval k is the accepted signal a single exponential, so that a logical gate fidelity can be read from it?

**Answers.** The full write-up is [REPORT.md](REPORT.md). In short:

- The k = 1 pseudo-threshold rises with q, but more slowly at each step. In the five-moment round under SI1000 it is 0.0123, 0.0313, 0.0596 and 0.0764 for q = 2, 3, 5, 7. Under SD6 there is no quotable crossing at q = 7.
- Three effects set the shape. Undetected logical faults are suppressed as q⁻². The compiled physical Clifford gets longer. A fully scrambled shot passes the depth-1 checks with probability q⁻⁸, which caps what postselection can do.
- A model built from these effects caps p* near 0.09 to 0.125 (SI1000) and 0.08 to 0.11 (SD6) for q = 7 to 31.
- A single exponential holds for k = 1 and 2 under SI1000 up to p*. It fails for k ≥ 3, for terminal-only postselection, and under SD6 at low p, where the unchecked mirror inverse adds an m² term.

We ran q = 5 and 7 in this study with the one-ancilla layout. The q = 2 and 3 values come from the production runs. Both check rounds are reported: the five-moment round used in the paper and the seven-moment round of the published runs.

## Contents

| Path | What it holds |
|---|---|
| `REPORT.md` | the study; Section 3 lists every script, its outputs and its runtime |
| `qslib.py`, `qsim.py`, `qpairs.py`, `qfit.py`, `single_exp_configs.py` | libraries: general-q Clifford catalogue and code profile, circuit sampling, exact two-fault expansion, the production fit |
| `01_*.py` to `04_*.py` | checks against the production catalogue and circuits |
| `10_*.py` to `14_*.py` | sampling, thresholds and the p*(q) table |
| `20_*.py` to `24_*.py` | the two-fault expansion, random acceptance and the model envelope |
| `30_*.py` to `32_*.py` | the single-exponential criterion and its tables |
| `40_figures.py` | Figs. 1 to 5 |
| `run_*.sh`, `slurm_sample.sh`, `jobs_*.txt` | the run commands we used |
| `results/` | every table the report quotes |
| `figures/` | the five figures of the report |

## How to run

We run every script from this folder:

```bash
cd studies/qudit-scaling
PYTHONNOUSERSITE=1 NUMBA_NUM_THREADS=1 ../../.venv-sdim140/bin/python 01_catalogue_checks.py
```

The libraries put the repository's `src/` and `scripts/` on `sys.path`. Scripts 03, 04, 14, 30, 31 and 40 also read production runs or comparison tables under `LRB-experiment-data-slurm/`. SLURM jobs go through `slurm_sample.sh <jobs file>`, submitted from this folder after `mkdir -p logs`.

We checked the copy after the move. Scripts 01, 02, 14, 31, 32 and 40 run here, and the tables they rewrite match the committed ones byte for byte.

## What we left out

The original folder was 44 MB. We kept the scripts, the report, the figures and the tables. We left out:

- `cache/`: the sampled counts per (q, noise, p), about 27 MB. `10_sample_lrb.py` and `12_sample_rb.py` regenerate them, and they resume where they stopped.
- `logs/`: SLURM and run logs.
- `vendor/`: a frozen copy of `src/lrb` and three helper scripts. It is byte-identical to the repository's `src/lrb` and `scripts/`, so the scripts now import those directly.
- `results/superseded/` and two smoke-test tables.

Six large tables (the pair expansions, the model curves and the single-exponential rows) are stored gzip-compressed as `*.csv.gz`. The scripts read and write them under those names, and pandas handles the compression.
