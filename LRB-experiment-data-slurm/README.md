# Run data

One folder per simulation campaign, `Run-<date>-<time>-<code_name>-<custom name>`, plus the
cross-run analyses in `comparisons/`. The folder name is historical: every script, notebook and
checkpoint refers to it.

## Which runs belong to which layout

| folder name contains | layout | round |
|---|---|---|
| `folded_qutrit_prod5b-single-ancilla`, `folded_qubit_prod5b-single-ancilla` | one-ancilla | five-moment |
| `folded_qutrit-single-ancilla`, `folded_qubit-single-ancilla` | one-ancilla | seven-moment |
| `layout_b-layout-b` | two-ancilla | |
| `conv_sites-conv-sites` | conventional-site two-ancilla | |
| `split_unif-split-ancilla`, `split_unif-split-unif` | relay two-ancilla | |
| `Run-2026-02-18-*` | depolarizing runs of the folded code and the QGRM code (sdim 1.3.4) | |

Suffixes: `-sdim140` runs use sdim 1.4.0 and the SI1000/SD6 models of `docs/simulation-model.md`;
`-highp-ext` are high-p extensions merged into the production thresholds; `-threshold-refine` add
points near a crossing; `-compiled-recovery` close each sequence with one compiled recovery Clifford
instead of the mirror inverse.

## Inside a run folder

| file | what |
|---|---|
| `code_name.txt`, `noise_model.txt`, `probs.txt`, `depths.txt`, `shots.txt`, `num_cliffs.txt`, `check_const.txt`, `check_unif.txt` | the run's grid |
| `run_instructions.txt` | the command that generated it |
| `validation.txt` (newer runs) | output of `scripts/validate_lrb_run.py` |
| `results/LRB/<index>/{const,unif}_check_data/<k>.csv`, `results/RB/<index>.csv` | fidelities and rejection rates (format in `docs/simulation-model.md`) |
| `results/plots/` | fits, threshold plots, GIFs |

The circuits (`experiments/`), checkpoints (`progress/`) and job logs stay on the cluster and are
not in git.

## comparisons/

The canonical pseudo-thresholds are the `pseudo-thresholds-*-current.{md,csv}` files, one pair per
layout and code, written by `scripts/merge_highp_threshold_extension.py` with the sequence-bootstrap
error from `scripts/sequence_bootstrap_thresholds.py`. The subfolders hold the merged threshold
grids, the bootstraps, the layout and round comparisons (`sdim140-prod5b/`, `sdim140-layout-b/`,
`qubit-vs-qutrit-*`), the hardware layout search (`two-ancilla-checks/`) and the encoder search
(`encoder-study/`).

`local-runner/` holds the drivers that kept the campaigns going between SLURM jobs.
