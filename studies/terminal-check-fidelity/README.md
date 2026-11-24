# A logical gate fidelity from terminal-check LRB-D

**Question.** The terminal-check (const-0) protocol has no check rounds between Cliffords, and its postselected decay is not a single exponential. Can its data still give a logical gate fidelity that we can compare with physical RB?

**Answer.** Yes, but not from a decay fit of the postselected curve. The full write-up is [REPORT.md](REPORT.md). In short:

- The accepted logical infidelity grows as the square of the depth. A single-exponential fit overstates the infidelity of one Clifford pair by a factor of 6 to 11 with ideal SPAM, and by more once the encoder and readout are noisy.
- We keep all five X-basis readouts of every shot and Fourier-transform the outcome histogram. Each Fourier mode decays as an ordinary exponential, with SPAM only in its amplitude.
- A joint fit of the mode rates gives ε_L, the logical infidelity of one Clifford pair between perfect preparation and a perfect check, and d_L, the fraction of shots one pair rejects. The block-rate fit is unbiased to below 1% in simulation, with 2 to 5% statistical error from 30 sequences and about 2.5e5 shots.
- One logical Clifford followed by the check beats the physical Clifford at every simulated p. Without intermediate checks the gain falls as one over the depth.
- The block fit assumes independent, stationary errors. A correlated error between the q2-q3 pair and q0, q1 or q4 is a first-order logical fault that it misses.

## Contents

| Path | What it holds |
|---|---|
| `REPORT.md` | the study; Section 9 lists every script and its runtime |
| `tclib.py`, `tcexact.py`, `tcsteps.py`, `tcfit.py`, `tcref.py` | libraries: circuit builder, exact distributions, per-Clifford factors, the estimators (`tcfit.analyze` is the entry point for hardware data), reference values |
| `01_*.py` to `09_*.py` | one script per section or table of the report |
| `figures.py` | Figs. 1 to 7 (`09_logical_vs_physical.py` draws Fig. 8) |
| `results/` | every table of the report as CSV or Markdown, the exact curves and resampled counts as small NPZ files, and the log of the Section 2 checks |
| `figures/` | Figs. 1 to 8 |

## How to run

We run every script from this folder:

```bash
cd studies/terminal-check-fidelity
PYTHONNOUSERSITE=1 ../../.venv-sdim140/bin/python 02b_exact_checks.py
```

`tclib.py` puts the repository's `src/` on `sys.path`. Scripts 01, 02 and 06 also read the one-ancilla production runs under `LRB-experiment-data-slurm/`. Script 01 needs `TMPDIR` set to a scratch folder.

We checked the copy after the move. `02b_exact_checks.py` and `07_tables.py` run against the current `src/lrb`, and their output matches `results/02b_exact_checks.log` and `results/tables.md` line for line.

## What we left out

- `results/cache/`: the exact tables `tcexact.py` and `tcsteps.py` build on first use, about 1.5 MB. They rebuild in about a minute.
- The run logs of scripts 01, 04, 05, 06, 08 and 09. Their numbers are in the CSV files and in the report.
