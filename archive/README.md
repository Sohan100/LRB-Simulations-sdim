# Archive

The raw data of our simulation campaigns, compressed, and a record of what we threw away.

## runs/

One xz archive per run folder of [`LRB-experiment-data-slurm/`](../LRB-experiment-data-slurm/),
`runs/<run>.tar.xz`, with the run's raw data:

- `experiments/`: every circuit (`.chp`) the run simulated;
- `progress/`, `progress_rb_comparison/`: the committed checkpoint of every noise strength, which
  holds the per-sequence outcome tallies behind every fit and every sequence bootstrap;
- the run's grid and provenance files (`code_name.txt`, `probs.txt`, `run_instructions.txt`, ...).

`results/` stays where it is, in git. Circuits are highly repetitive text, so xz shrinks a run's
400 to 500 MB to about 1 MB. [`runs/index.json`](runs/index.json) lists every archive with its file
count and its size before and after compression; `python scripts/archive_runs.py list` prints it.
In total, 55 runs and 18.5 GB of raw data fit in 51 MB.

Restore a run into its folder (the results already there are untouched):

```bash
tar -xJf archive/runs/<run>.tar.xz -C LRB-experiment-data-slurm/
```

Scripts and notebooks then find the circuits and checkpoints where they were on the cluster, for
example `scripts/sequence_bootstrap_thresholds.py`, the byte-identity test in
`tests/test_compiled_recovery.py` (CI restores one run for it), or the correction-decoder study.
The runs from before sdim 1.4.0 (June and August 2026) store `prob_scale` on their noise gates;
sdim 1.4.0 reads them through `lrb.correction.compact_dem.read_run_circuit`.

Archive a new run once it is finished:

```bash
python scripts/archive_runs.py pack LRB-experiment-data-slurm/Run-...
```

## What we threw away

| data | where it was | why |
|---|---|---|
| superseded checkpoint generations | `progress/<index>/.checkpoint_generations/` | only the committed generation (`.checkpoint_current`) feeds the results |
| per-batch timing (`timing_metrics.csv`) | `progress/<index>/` | runtime telemetry; about 110 MB even compressed |
| job logs, locks | `logs_*/`, `slurm-*.out`, `*.err`, `*.lock` | job bookkeeping |
| animated plots, 275 GIFs (284 MB) | `results/plots/gifs/`, in git until 2026-11-24 | the notebooks regenerate them from `results/` (`LRBGifAnimator`); git history keeps the old ones |
| GIFs embedded in notebook outputs (64 MB) | `notebooks/` | same; rerunning the cell shows them |
| three superseded runs | `Run-2026-05-29-19-32-40-folded_qutrit-si1000-all`, `Run-2026-06-05-16-29-51-folded_qutrit_split_unif-split-unif-si1000-detectors`, `Run-2026-07-15-23-28-58-folded_qutrit-single-ancilla-si1000` | replaced by later runs, never in git |
| raw sinter output of the folded-code memory study | `layouts/folded-surface-code/memory/results/raw/` | `results/stats.csv` keeps every task's shot and error counts |
| per-circuit outputs of the correction-decoder studies | `.cache_correction/` | the study scripts rebuild them; the tables are in each study's `results/` |

The two depolarizing runs of February 2026 (`Run-2026-02-18-*`) ran on a laptop; only their
`results/` exist.

Runs with noise models from collaborators' hardware are not archived here; they are not cleared
for this public repository.
