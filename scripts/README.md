# Scripts

Command-line entry points. They stay in one flat folder because every run folder records the
exact command that produced it (`run_instructions.txt`), and those commands name these paths. Run
them from the repository root with the package installed (`pip install -e .`).

## Make and run a campaign

| script | what it does |
|---|---|
| `generate_circuits_folded.py` | builds a new run folder for the folded code: `--ancilla-mode` picks the layout (`single`, `single-prod5b`, `layout-b`, `conv-sites`, `split-unif`, `a1`), `--qudit-dimension 3` or `2`, `--noise-model si1000` or `sd6`, plus depths, probabilities and check schedules |
| `generate_circuits_qgrm.py` | the same for the `[[3,1,2]]` QGRM code |
| `make_round_variant.py` | copies a run with the same Clifford sequences but another profile's check round, so two rounds can be compared pair by pair |
| `run_lrb_experiment.py` | simulates one probability index of a run (resumes from checkpoints) |
| `run_lrb_dem_local_until_slurm.py` | keeps indices running on a login node until the SLURM job starts |
| `run_physical_rb_comparison.py` | adds an RB-only comparison grid to an existing run |
| `validate_lrb_run.py` | checks every index of a finished run (`--require-complete --strict-production-grid`) |
| `verify_default_circuits_unchanged.py` | checks that the generator still reproduces a run's `.chp` files byte for byte |
| `benchmark_dem_vs_sdim.py` | times the DEM backend against direct sdim sampling |

## Pseudo-thresholds

| script | what it does |
|---|---|
| `merge_highp_threshold_extension.py` | merges production and high-p extension runs into the canonical tables `LRB-experiment-data-slurm/comparisons/pseudo-thresholds-*-current.{md,csv}` |
| `sequence_bootstrap_thresholds.py` | one standard error on each `p*` by resampling whole Clifford sequences |
| `plot_sequence_bootstrap_thresholds.py` | `p*` against check interval with those error bars |
| `provisional_thresholds.py` | quick `p*` from one production run before the merge |
| `merge_highp_extension_thresholds.py` | the earlier one-run version of the merge; kept because old comparison folders were made with it |

## Comparisons between layouts and codes

| script | what it compares |
|---|---|
| `compare_round_variant_thresholds.py` | the one-ancilla layout's five-moment round against its seven-moment round |
| `compare_layout_b_thresholds.py` | one ancilla, the conventional sites, and the two-ancilla layout |
| `compare_conv_sites_thresholds.py` | one ancilla, the relay, and the conventional sites |
| `compare_qubit_qutrit_thresholds.py` | the qubit and the qutrit code |
| `check_a1_single_faults.py` | enumerates every single fault of the A1 candidate's round |

## Other analyses

| script | what it does |
|---|---|
| `fit_lrb_pooled.py` | pooled signed-polarization fits (`r_L`, `d_L`) for one run |
| `plot_compiled_recovery_results.py` | the compiled-recovery runs through the production pipeline |
| `execute_notebook.py` | runs a notebook headless with the `lrb-sdim140` kernel |

The full workflow, with example commands, is in `docs/running-simulations.md`.
