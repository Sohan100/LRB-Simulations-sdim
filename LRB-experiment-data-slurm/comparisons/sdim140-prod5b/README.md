# The five-moment one-ancilla round in LRB-D: pseudo-thresholds (2026-11-19)

The five-moment round is the published one-ancilla round (the seven-moment round) with its 12 CNOTs in a different order. The generator packs it into 5 gate moments instead of 7. Layout, ancillas, readouts, encoder, logical gates and terminal measurements are unchanged. An exhaustive search over all 31,104 correct gate orders found none with a lower leading-order logical error per round; see `../two-ancilla-checks/hw-layouts/README.md`, section "Best schedule of the current round". Its internal name is PROD5B: profiles `folded_qutrit_prod5b` and `folded_qubit_prod5b`, merge families `prod5b` and `qubit-prod5b`.

History:
- The user said "Yes, do it" to rerunning the qutrit one-ancilla LRB-D campaigns with this round.
- On 2026-11-19 they chose it for the paper and asked for the four qubit runs (relayed by the paper session).

## Qutrit result

The five-moment round raises every pseudo-threshold: by 7-9% under SI1000 and by 14-25% under SD6.

| noise | k | p* seven-moment | p* five-moment | difference (paired 95%) | ratio (paired 95%) | error of the mean, five-moment |
|---|---|---|---|---|---|---|
| SI1000 | 1 | 0.0292 | 0.0313 | +0.0021 [+0.0017, +0.0025] | 1.073 [1.061, 1.085] | 0.0006 |
| SI1000 | 2 | 0.0239 | 0.0258 | +0.0019 [+0.0015, +0.0022] | 1.078 [1.063, 1.091] | 0.0005 |
| SI1000 | 3 | 0.0166 | 0.0178 | +0.0012 [+0.0008, +0.0015] | 1.074 [1.043, 1.094] | 0.0003 |
| SI1000 | 4 | 0.0132 | 0.0144 | +0.0012 [+0.0008, +0.0019] | 1.094 [1.061, 1.145] | 0.0003 |
| SD6 | 1 | 0.0399 | 0.0501 | +0.0101 [+0.0087, +0.0136] | 1.254 [1.222, 1.331] | 0.0026 |
| SD6 | 2 | 0.0269 | 0.0326 | +0.0057 [+0.0051, +0.0198] | 1.213 [1.192, 1.743] | 0.0059 or 0.0008, see below |
| SD6 | 3 | 0.0181 | 0.0214 | +0.0033 [+0.0023, +0.0039] | 1.180 [1.119, 1.222] | 0.0003 |
| SD6 | 4 | 0.0155 | 0.0177 | +0.0021 [+0.0017, +0.0028] | 1.136 [1.109, 1.184] | 0.0005 |

- The p* values are the published ones: each run against its own RB, with production and high-p extension merged at the published p_split (SI1000 0.015, SD6 0.025). Every fit includes depth 0.
- The paired columns use one common RB curve, the seven-moment run's. This moves the five-moment p* by at most 0.0001.
- Every paired 95% interval excludes zero.
- Hand-off tables: `../pseudo-thresholds-current.md` (seven-moment) and `../pseudo-thresholds-prod5b-current.md` (five-moment).

**SD6, k = 1.** The bootstrap is skewed upward: 5% of replicates land above p = 0.055, and the 68% interval is [0.0477, 0.0530].

### Error of the mean for SD6, k = 2: decision for the user

The pipeline's value is 0.0059. `scripts/sequence_bootstrap_thresholds.py` resamples the 30 Clifford sequences 1000 times, refits every curve, and reruns the threshold estimator. It reports the error of the mean as the plain standard deviation (ddof = 1) of p* over all finite replicates, with no trimming. That is the `seq_boot_se` column of `../pseudo-thresholds-prod5b-current.csv`. Its 68% percentile interval is [0.0318, 0.0453].

Why it is that large: the replicate distribution has two modes.
- The five-moment LRB curve at k = 2 dips at p = 0.045, where a three-depth fit gives 0.2646 per Clifford, below the 0.2713 at p = 0.04 and only 0.010 above RB.
- The estimator takes the last crossing, above which LRB stays above RB (`permanent_regime`).
- In 23% of replicates the resampled dip falls below RB, so the last crossing moves to about 0.045.
- In the other 77%, p* sits at the clean first crossing: LRB is 0.035 below RB at p = 0.030 and 0.031 above it at p = 0.035. Their standard deviation is 0.0008.
- The standard deviation over both modes, 0.0059, describes neither of them.

The alternatives:
1. **0.0059, the pipeline's value.** It is the same rule as every other cell, with no post-hoc choice. It counts the 23% of resamples in which the estimator really does return a different crossing.
2. **0.0008.** This is the standard deviation of the replicates below p = 0.04, and the leave-one-sequence-out jackknife gives the same, 0.00085. It describes the uncertainty of the crossing at 0.0326, but trimming at 0.04 is a choice made after seeing the distribution.
3. **Remove the dip.** Rerun SD6 at p = 0.040-0.050 with more depths, then recompute. This is the only option that changes the data rather than the statistic.

The paired lower bound on the gain, +0.0051, is the same under every option.

## LRB error per Clifford (qutrit)

Point estimates of the ratio five-moment / seven-moment, from the merged tables:

- SI1000: 0.79-0.92 at every k for p = 0.005-0.03, mostly 0.83-0.88. The round-only prediction was 0.86; the 1M-shot memory benchmark gave 0.81-0.86.
- SD6: 0.65-0.84 at every k for p = 0.005-0.02, moving toward 1 near and above each crossing. The round-only prediction was 0.51; the memory benchmark gave 0.44-0.48.
- RB ratios are 0.99-1.01, as expected, since the RB circuits are byte-identical.

I did not test why SD6 gains less in LRB-D than in back-to-back rounds. The likely reason is that the logical Clifford between rounds adds the same noise in both runs. Under SD6 its idle noise is full strength, so it is a larger share of the total than under SI1000.

## Qubit code

All four runs completed on 2026-11-20 at 00:23, and all four validated. The five-moment round raises every qubit pseudo-threshold, by 10-14% under SI1000 and 32-38% under SD6. Every paired 95% interval excludes zero.

| noise | k | p* seven-moment | p* five-moment | difference (paired 95%) | ratio (paired 95%) | error of the mean, five-moment |
|---|---|---|---|---|---|---|
| SI1000 | 1 | 0.0112 | 0.0123 | +0.0011 [+0.0008, +0.0016] | 1.096 [1.069, 1.137] | 0.0004 |
| SI1000 | 2 | 0.0078 | 0.0087 | +0.0009 [+0.0007, +0.0010] | 1.113 [1.089, 1.130] | 0.0002 |
| SI1000 | 3 | 0.0049 | 0.0056 | +0.0007 [+0.0006, +0.0008] | 1.143 [1.119, 1.161] | 0.0001 |
| SI1000 | 4 | 0.0040 | 0.0045 | +0.0005 [+0.0005, +0.0006] | 1.139 [1.117, 1.159] | 0.0001 |
| SD6 | 1 | 0.0210 | 0.0290 | +0.0080 [+0.0064, +0.0111] | 1.381 [1.289, 1.563] | 0.0010 |
| SD6 | 2 | 0.0151 | 0.0203 | +0.0052 [+0.0045, +0.0060] | 1.342 [1.306, 1.397] | 0.0007 |
| SD6 | 3 | 0.0094 | 0.0123 | +0.0030 [+0.0025, +0.0034] | 1.316 [1.273, 1.355] | 0.0004 |
| SD6 | 4 | 0.0072 | 0.0095 | +0.0023 [+0.0021, +0.0025] | 1.317 [1.285, 1.345] | 0.0003 |

- The p* values are the published ones: each run against its own RB, with production and threshold refinement merged in the published windows. Every fit includes depth 0.
- The paired columns use one common RB curve, the seven-moment run's. This moves the five-moment p* by at most 0.0001.
- `paired_thresholds_qubit.md` prints the paired SEs to four decimals, so the smallest show as 0.0000; they are below 0.00005.
- None of the qubit cells is bimodal: the bootstrap and the jackknife agree within 0.0003 everywhere, and every replicate is valid.
- Hand-off tables: `../pseudo-thresholds-qubit-current.md` (seven-moment) and `../pseudo-thresholds-qubit-prod5b-current.md` (five-moment).

**LRB error per Clifford.** The point ratios five-moment / seven-moment are close to the qutrit ones:
- SI1000: 0.82-0.90 at every k for p = 0.001-0.02.
- SD6: 0.64-0.77 at every k for p = 0.001-0.02, moving toward 1 above p* at k = 4.
- RB ratios: 0.99-1.01.

**Why the qubit SD6 gain in p* is larger than the qutrit one.** A drop in LRB error by a factor r near p* moves the crossing by r^(-1/(s_LRB - s_RB)), where s_LRB and s_RB are the local log-log slopes of the LRB and RB curves. With the slopes and the r measured at each crossing, this predicts all 16 measured gains (qutrit and qubit) to within 0.13, and 12 of them to within 0.03. Under SD6 the qubit LRB drop near p* is a little larger (r = 0.66-0.70, against 0.70-0.81 for the qutrit). The qubit LRB curve is also shallower relative to RB there: s_LRB - s_RB = 1.1-1.8, against 1.4-2.0. Both push the crossing further.

**Qubit / qutrit p* with the five-moment round on both sides** (`../qubit-vs-qutrit-five-moment/threshold-comparison.md`):

| noise | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|
| SI1000, five-moment | 0.39 | 0.34 | 0.32 | 0.31 |
| SI1000, seven-moment (published) | 0.38 | 0.33 | 0.30 | 0.30 |
| SD6, five-moment | 0.58 | 0.62 | 0.58 | 0.54 |
| SD6, seven-moment (published) | 0.53 | 0.56 | 0.52 | 0.47 |

- The same folder has the two-ancilla layout's ratios (unchanged) and the plots.
- Its SD6 k = 2 qutrit cell uses the pipeline's 0.0059, so that row shows -2.1 sigma. With 0.0008 the qubit-qutrit difference would be about -12 sigma.

| qubit five-moment run | paired with (published) |
|---|---|
| `Run-2026-11-19-23-39-09-folded_qubit_prod5b-single-ancilla-si1000-sdim140` | `Run-2026-10-31-10-12-55-folded_qubit-single-ancilla-si1000-sdim140` |
| `Run-2026-11-19-23-39-27-folded_qubit_prod5b-single-ancilla-sd6-sdim140` | `Run-2026-10-31-10-13-03-folded_qubit-single-ancilla-sd6-sdim140` |
| `Run-2026-11-19-23-39-41-folded_qubit_prod5b-single-ancilla-si1000-sdim140-threshold-refine` | `Run-2026-10-31-11-09-27-folded_qubit-single-ancilla-si1000-sdim140-threshold-refine` |
| `Run-2026-11-19-23-39-59-folded_qubit_prod5b-single-ancilla-sd6-sdim140-threshold-refine` | `Run-2026-10-31-11-09-31-folded_qubit-single-ancilla-sd6-sdim140-threshold-refine` |

- Profile `folded_qubit_prod5b`: class `FoldedQubitProd5bDetectionCode`, generator mode `--ancilla-mode single-prod5b --qudit-dimension 2`.
- It has the same twelve gates in the same order as the qutrit five-moment round, built in dimension two, and schedules to 5 gate moments (published qubit round: 7).
- Merge windows and threshold rule are those of the published qubit merge: refinement rows inside SI1000 0.002-0.013 and SD6 0.006-0.026, scale-free crossing rule.

- The runs ran in 30-minute debug jobs (`local-runner/qubit-prod5b-campaign-driver.sh`, jobs `lrbQP*`), 2026-11-19 23:41 to 2026-11-20 00:23. `validate_lrb_run.py --require-complete` reports all four valid; the production runs also pass `--strict-production-grid`.

Checks:
- `tests/test_prod5b_qubit_round.py` (6 tests) plus the existing suite: 107 tests pass. The round equals the qutrit five-moment order, uses the published qubit gates, and gives the published syndromes for every single data Pauli. Noiseless rounds pass and keep the logical. Detector and raw unpack agree.
- DEM sampling agrees with direct sdim: SI1000 p = 0.005, depth 4, z = +0.85 and +0.73; SD6 p = 0.005, depth 2, z = +0.83 and -0.91.
- Every sequence of every run reproduced its published source byte for byte before the five-moment circuits were written.

## How the qutrit runs were made

| qutrit five-moment run | paired with (published) |
|---|---|
| `Run-2026-11-19-12-51-34-folded_qutrit_prod5b-single-ancilla-si1000-sdim140` | `Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140` |
| `Run-2026-11-19-12-51-50-folded_qutrit_prod5b-single-ancilla-sd6-sdim140` | `Run-2026-10-28-10-24-46-folded_qutrit-single-ancilla-sd6-sdim140` |
| `Run-2026-11-19-12-52-10-folded_qutrit_prod5b-single-ancilla-si1000-sdim140-highp-ext` | `Run-2026-10-30-10-01-06-folded_qutrit-single-ancilla-si1000-sdim140-highp-ext` |
| `Run-2026-11-19-12-52-21-folded_qutrit_prod5b-single-ancilla-sd6-sdim140-highp-ext` | `Run-2026-10-30-05-58-58-folded_qutrit-single-ancilla-sd6-sdim140-highp-ext` |

- Profile `folded_qutrit_prod5b`: class `FoldedQutritProd5bDetectionCode`, generator mode `--ancilla-mode single-prod5b`.
- All code is in the worktree `/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/one-ancilla-prod5b`, branch `claude/one-ancilla-prod5b`, uncommitted. Production profiles and checkpoint fingerprints are untouched.
- `scripts/make_round_variant.py` built each run from its source run:
  - It decodes the Clifford words of every sequence from the source circuit files and regenerates the source from them.
  - Every sequence matched the source files byte for byte (LRB and LRB_const0, at the first, middle and last p).
  - It then writes the five-moment circuits from the same words and copies the RB circuits byte for byte.
  - Grids, shots, check policies, noise and `d.txt` are the source's. `round_variant_manifest.json` in each run lists the words.
- The runs ran in 30-minute debug jobs (`local-runner/prod5b-campaign-driver.sh`, jobs `lrbP-*`), 12:53-14:17. `validate_lrb_run.py --require-complete` reports all four valid (`validation.txt` in each run; production runs with `--strict-production-grid`).

Checks:
- `tests/test_prod5b_round.py` passes: the round uses exactly the published gates, and every single data Pauli gives the published syndromes.
- DEM sampling agrees with direct sdim: SI1000 p = 0.01, depth 4, z = +0.24 and -1.32; SD6 p = 0.005, depth 2, z = -0.64 and -0.50.
- The worktree's analysis code reproduces all eight published seven-moment p* exactly. The sequence bootstrap's full sample reproduces the five-moment p* exactly.

## Files

- `paired_thresholds.{csv,md}` (qutrit) and `paired_thresholds_qubit.{csv,md}` (qubit, when done): paired comparisons, 1000 replicates, seed 20261119 (`scripts/compare_round_variant_thresholds.py --family qutrit|qubit`). Each replicate draws the same sequences for both rounds, separately for production and extension or refinement, and uses one common RB curve.
- `lrb-vs-rb-unif-1-prod5b-vs-current.{pdf,png,csv}` (qutrit) and `lrb-vs-rb-unif-1-qubit-five-vs-seven-moment.*` (qubit): k = 1 LRB and RB error per Clifford for both rounds, production runs only. They come from the worktree notebooks `Visualize LRB DEM PROD5B SI1000 SD6 Results.ipynb` and `Visualize LRB DEM Qubit Five-Moment SI1000 SD6 Results.ipynb`. These are review plots, not paper figures.
- `../sdim140-prod5b-{si1000,sd6}-highp-merged/` and `../sdim140-qubit-prod5b-{si1000,sd6}-refine-merged/`: merged tables and threshold summaries.
- `../sequence-bootstrap/{,qubit-}prod5b-one-ancilla-{si1000,sd6}/`: error of the mean, replicates and RB caches.
- `../pseudo-thresholds-prod5b-current.{md,csv}` and `../pseudo-thresholds-qubit-prod5b-current.{md,csv}`: hand-off tables in the usual format.
- `../qubit-vs-qutrit-five-moment/` (when done): qubit/qutrit p* ratios with the five-moment round on both sides (`scripts/compare_qubit_qutrit_thresholds.py --two-ancilla layout-b-five-moment`).

The paper drafts are not edited here; the paper session writes the text.
