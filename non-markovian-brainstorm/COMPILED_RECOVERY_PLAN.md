# Compiled recovery Clifford: implementation and run plan

Written 2026-10-31, following REPORT.md. The user approved this "as an alternate method but dont override the current experiment". Everything below is opt-in. The default mirror protocol and the production fit are unchanged, and the next section shows how that was checked. Nothing is committed.

## What was added

| file | change |
|---|---|
| `src/lrb/clifford_recovery.py` (new) | `CliffordRecoveryTable` finds the catalog Clifford equal to (Cm...C1)^-1 by multiplying 3×3 unitaries; the token conventions are checked against sdim |
| `src/lrb/circuit_generator.py` | `LRBCircuitGenerator(inverse_mode="mirror" \| "compiled")`, default `mirror`. One helper, `_closing_gate_lists`, replaces the inverse-suffix slice in the LRB, const=0 and RB builders. In mirror mode it returns the same slice as before. |
| `src/lrb/experiment_setup.py` | `ExperimentSetupConfig.inverse_mode`, passed to the generator; `inverse_mode.txt` is written only for non-default runs, so mirror runs keep their exact file set |
| `scripts/generate_circuits_folded.py` | `--inverse-mode {mirror,compiled}`, default `mirror` |
| `src/lrb/pooled_lrb_fits.py` (new) | pooled signed-polarization fits: r_L, d_L, r_uncond, sequence-bootstrap SE, χ²/dof, early-vs-late window ratio, pseudo-thresholds, optional production + high-p merge. Every LRB and RB fit includes depth 0, with no option to leave it out |
| `scripts/fit_lrb_pooled.py` (new) | CLI for the above; writes only to `--out-dir` |
| `scripts/verify_default_circuits_unchanged.py` (new) | recovers a run's Clifford words from its own `.chp` files, regenerates them with the default generator and compares bytes |
| `tests/test_compiled_recovery.py` (new) | 15 tests, described below |

Files not touched: the six checkpoint-fingerprinted files (`scripts/run_lrb_experiment.py`, `src/lrb/{lrb_simulation,dem_simulation,detector_events,code_definitions,code_simulation_profiles}.py`), `src/lrb/lrb_plotting.py`, every run folder, `comparisons/`, the notebooks and the `working-folder*.txt` pointers (md5 checked before and after).

Compiled mode closes a depth-m sequence with one recovery Clifford: m forward Cliffords each followed by a check round, then the recovery, then the final check. That is m + 1 noisy Cliffords and m + 1 check rounds, against 2m Cliffords and m + 1 rounds for the mirror. Detector layout, postselection slots and the simulation pipeline are unchanged. RB and const=0 circuits get the same closing, so LRB and RB rates are both per Clifford. Depth 0 has no recovery gate in either mode. Every fit includes it anyway, and at high p it shows up as the depth-0 boundary effect described in REPORT.md Section 2.1.

## Evidence that the default path is unchanged

1. **Production circuits are byte-identical.** The production circuits came from an unseeded `random.choice`, so there was no seed to replay. Instead, `verify_default_circuits_unchanged.py` decodes each sequence's Clifford words from the run's own depth-22 file, regenerates the LRB, LRB_const0 and RB circuits with the default generator, and compares every depth file byte for byte.
   - I ran it on all eight sdim140 run folders (four production, four high-p extensions) at sequences 0, 17 and 29 and three p indices each.
   - Result: 24/24 identical before the edit and 24/24 identical after.
2. **The seeded generation hash is unchanged.** I hashed a seeded `generate_tests` tree (two codes, two noise models) before the edit. After the edit the hashes match, both with no `inverse_mode` argument and with `inverse_mode="mirror"`. This is in the tests, so it holds without the production data.
3. **The notebook tables reproduce.** I rebuilt `unif_lrb_vs_rb_table_all_mixed_fits.csv` with the notebook's settings (500 bootstrap reps, seed 12345, unif 1–22) into a scratch directory and compared with the stored file. The one-ancilla SI1000 table is identical (330×26, maximum difference 0); the other three runs: see the line at the end of this section. `lrb_plotting.py` was not edited, so this checks reproducibility rather than my change.
4. **Tests pass.** The 38 existing tests plus the 15 new ones pass (`PYTHONPATH=src .venv-sdim140/bin/python -m pytest tests`). The new tests cover:
   - default-path golden hashes;
   - the production byte-identity check (skipped if the run folder is absent);
   - equal random-number consumption in both modes;
   - the recovery inverting 300 random products;
   - token unitaries against sdim;
   - compiled LRB, const=0 and RB circuits reading the ideal outcome deterministically for both layouts;
   - `inverse_mode.txt` written only for compiled runs;
   - the pooled fit recovering known r and d from synthetic counts;
   - the shot floor, depth selection and threshold crossing logic.

Notebook table check, all four production runs: identical (330×26 each, `DataFrame.equals` true, maximum difference 0).

## Smoke test, done

The smoke run lived entirely in scratch, so the real working-folder pointers were untouched. One-ancilla SD6, compiled mode, 3 sequences, depths 0–4, p ∈ {0, 0.002, 0.02}, 20k shots, unif 1–2 and const 0–1.

- `scripts/run_lrb_experiment.py ... --simulation-backend dem` ran all three indices unchanged.
- At p = 0 every LRB, const=0 and RB fidelity is 1.0 with 0 rejections, so the recovery closes exactly through the real pipeline.
- `validate_lrb_run.py --require-complete` reports `valid: true`.
- `fit_lrb_pooled.py` runs on it and labels the step unit `clifford+check`.

## Proposed runs (not started)

Four new run folders, one per configuration, each with a production grid and a high-p dense grid. Every policy is postselected as now, so unif 1–22 and const 0–22 come for free.

| run | grid | shots | estimated cost |
|---|---|---|---|
| `...-single-ancilla-si1000-sdim140-compiled-recovery` | 15 p (0 to 0.05, production list), depths 0,1,2,3,4,6,10,14,18,22 | 1M | ~0.4 node-h |
| `...-single-ancilla-sd6-sdim140-compiled-recovery` | same | 1M | ~0.4 node-h |
| `...-split-ancilla-si1000-sdim140-compiled-recovery` | same | 1M | ~0.5 node-h |
| `...-split-ancilla-sd6-sdim140-compiled-recovery` | same | 1M | ~0.5 node-h |
| `...-single-ancilla-si1000-sdim140-compiled-recovery-highp` | 12 p, 0.015 to 0.05 (as the SI1000 extension), depths 0,1,2,3,4 | 20M | ~0.7 node-h |
| `...-split-ancilla-si1000-sdim140-compiled-recovery-highp` | same | 20M | ~0.9 node-h |
| `...-single-ancilla-sd6-sdim140-compiled-recovery-highp` | 10 p, 0.025 to 0.07 in steps of 0.005 | 20M | ~0.7 node-h |
| `...-split-ancilla-sd6-sdim140-compiled-recovery-highp` | same | 20M | ~0.9 node-h |
| **total** | | | **~5 node-hours** (budget 6–7 with restarts) |

Notes on the plan:

- **How the costs were estimated.** From the production timing logs: the summed sampling, compile and unpack time per index is 4–22 min for the production runs and 45–64 min for the 20M-shot extensions. All indices run concurrently on one node, so wall time is the slowest index. Compiled circuits are somewhat shorter (m + 1 Cliffords instead of 2m, same check rounds), and the high-p grid stops at depth 4 instead of 6, so these estimates are conservative.
- **Why the SD6 grid goes higher.** It reaches 0.07, against 0.06 before. At p = 0.025 the compiled LRB/RB ratio is 0.29, against 0.46 for the mirror production fit, so the SD6 crossings should move up.
- **Why the grid is dense at small depths.** Depth 0 is in every fit (user rule, 2026-10-31). The acceptance per step sets the useful depth range, so the grid is dense where the accepted shots are.

Generation, from the repository root. The generator rewrites the working-folder pointers, so save and restore them around it:

```bash
mkdir -p /tmp/wf-backup && cp LRB-experiment-data-slurm/working-folder*.txt /tmp/wf-backup/ && \
.venv-sdim140/bin/python scripts/generate_circuits_folded.py --inverse-mode compiled \
    --noise-model si1000 --ancilla-mode single --custom-name single-ancilla-si1000-sdim140-compiled-recovery \
    --n-cliff 30 --depths 0,1,2,3,4,6,10,14,18,22 --n-shots 1000000 \
    --probabilities 0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05 \
    --stab-checks-const 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
    --stab-checks-unif 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 && \
cp /tmp/wf-backup/working-folder*.txt LRB-experiment-data-slurm/
```

The other three production runs differ only in `--noise-model` and `--ancilla-mode split-unif`. The high-p runs use `--depths 0,1,2,3,4 --n-shots 20000000` and their p lists.

Submission, debug QOS, 30-minute jobs with self-resubmission from checkpoints:

```bash
sbatch -q debug -t 00:30:00 --export=ALL,RUN_NAME_OVERRIDE=<Run-folder>,LRB_THREADS_PER_INDEX=8 slurm/run_lrb_dem_circuit_level.sh
```

With two debug jobs running at a time, about 14 thirty-minute jobs cover all eight runs in roughly 4 h of wall time. The default preempt QOS in the launcher would need fewer restarts.

Analysis once done. This writes new files under each new run folder only:

```bash
.venv-sdim140/bin/python scripts/fit_lrb_pooled.py LRB-experiment-data-slurm/<prod-compiled-run> \
    --highp-run LRB-experiment-data-slurm/<highp-compiled-run> --split 0.025 \
    --out-dir LRB-experiment-data-slurm/<prod-compiled-run>/results/plots/pooled_fits
```

Use split 0.015 for SI1000, as before. Each output row carries r_L, d_L, r_uncond, r_RB per Clifford, χ²/dof and the window ratio. A window ratio far from 1 with χ²/dof ≫ 1 means the number should not be quoted as a gate fidelity.

## Launched 2026-10-31 00:47 (user approved)

The eight run folders were generated with `non-markovian-brainstorm/compiled-runs/generate_runs.sh`; the working-folder pointers were restored and md5-checked after each generation:

| job name | run folder | threads per index |
|---|---|---|
| lrbc-1-si | `Run-2026-10-31-00-39-01-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery` | 8 |
| lrbc-1-sd6 | `Run-2026-10-31-00-40-37-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery` | 8 |
| lrbc-2-si | `Run-2026-10-31-00-41-58-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery` | 8 |
| lrbc-2-sd6 | `Run-2026-10-31-00-43-31-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery` | 8 |
| lrbc-1-si-hp | `Run-2026-10-31-00-45-00-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery-highp` | 10 |
| lrbc-2-si-hp | `Run-2026-10-31-00-45-27-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery-highp` | 10 |
| lrbc-1-sd6-hp | `Run-2026-10-31-00-45-57-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery-highp` | 12 |
| lrbc-2-sd6-hp | `Run-2026-10-31-00-46-19-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery-highp` | 12 |

All eight pass the launcher's `DRY_RUN=1` validation (two were checked explicitly).

`non-markovian-brainstorm/compiled-runs/driver.sh` runs detached (`setsid nohup`, PID in `driver.pid`, log in `driver.log`):

- It keeps at most two `lrbc-*` debug jobs queued: 30 minutes each, `LRB_AUTO_RESUBMIT=0`, production grids first. It resubmits each run from its checkpoints until every `progress/*/done.txt` is 1, with at most 8 submissions per run.
- It then runs `validate_lrb_run.py --require-complete` on each run (`validate-<job>.json`).
- Finally it writes the merged pooled fits to `<production run>/results/plots/pooled_fits/` (`pooled-<run>.txt`).

To stop it, `touch non-markovian-brainstorm/compiled-runs/STOP`; running jobs finish their 30 minutes and checkpoints keep their progress. Each job's SLURM output is `slurm-<jobid>.{out,err}` inside its run folder.

## Results, 2026-10-31 (depth 0 in every fit)

**Provisional, on hold (2026-10-31).** These compiled-recovery numbers use depth m as the x-axis. In compiled mode, depth 0 has no noisy Clifford while every later depth m has m + 1 (forward plus recovery), so a fit against m through m = 0 biases the rates, and most strongly over the short 0 to 4 high-p window. Indexing by the number of noisy Cliffords (0 at depth 0, m + 1 otherwise) puts compiled RB's depth-0 point back on its curve and mostly fixes unif1 LRB. The user is deciding which x-axis to use, with m = 0 kept in every fit. Do not quote the compiled numbers until then.

All eight runs completed by 03:02 in 12 debug jobs (lrbc-1-si and lrbc-1-sd6 needed one each; the high-p runs needed two), and all validate.

- The driver's first pooled fits left out depth 0. The user then ruled that every fit includes m = 0, so `pooled_lrb_fits.py` lost the option to leave it out.
- All four merged fits were redone with the depth-0 rule. They are in each production run's `results/plots/pooled_fits/` and in `compiled-runs/pooled-<run>.txt`.
- The side-by-side comparison with the production mirror numbers, both estimators, is `compiled_vs_mirror.py` → `results/compiled_vs_mirror_thresholds.csv`, summarised in REPORT.md Section 3.6.

In short:

- At low p the compiled decays are single exponentials through m = 0.
- At high p the depth-0 boundary dominates both protocols. The compiled thresholds land between 18% below and 15% above the official mirror ones; for one-ancilla SD6 p*(1) it is 0.040 against 0.040.
- A leading check round after preparation is the protocol change that addresses the boundary while keeping m = 0 in every fit. It was tested on the login node (REPORT.md Section 2.1) but is not yet in `src/`.
