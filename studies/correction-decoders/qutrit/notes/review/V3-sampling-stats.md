# V3 review: sampler fidelity and statistical methodology

Reviewer id: V3-sampling-stats. All scripts and their raw outputs are in
`notes/review/V3/` (each `v3_*.py` has a matching `v3_*_output.txt`). Every script
only reads `src/`, `.cache_correction/` and the baseline run folder. Run with
`module load python/3.11; OMP_NUM_THREADS=4 python <script>` from that folder.

## Verdict

| Item | Result |
|---|---|
| (1) `sampler.sample_shots` vs the production `sample_detector_error_model` | Faithful **[VN]**. No deviation in marginals, pairwise co-occurrences, the logical marginal or the joint (logical, all rounds trivial), in two-sample tests or against exact references. |
| (2) Rao-Blackwellized (RB) estimator (`study._expected_residual`, `soft_hist_rb`) | Unbiased, with lower variance **[VN]**. The soft-threshold *mapping* is wrong: see F1. |
| (3) `metrics.fit_decay` vs `LRBResultsPlotter` | Bit-identical to the pipeline **[VN]**. Weighting caveat for exact curves: see F3. |
| (4) `unif_mask` / `const_mask` vs `postselect_*` | Identical for depths 0 to 22 and policies 1 to 22 **[VN]**. |
| (5) Exact LRB-D vs the 1M-shot production data | Reproduced within shot noise, using a per-circuit test that is stronger than the published one **[VN]**. The published z-scores are flawed: see F2. |

Findings: 1 critical, 3 major and 4 minor.

---

## Findings

### F1 (critical): the soft-threshold labels are not the thresholds that were applied

- **Where:**
  - `src/lrb/correction/aggregate.py:55-63` (`_soft_point`), with `study.py:31` (`CONF_EDGES = linspace(-12, 0, 241)`) and `study.py:174` / `decoders.py:313` (`searchsorted(..., side="right")`).
  - The labels appear in `studies/correction-decoders/qutrit/scripts/make_tables.py:21-23`, `REPORT.md:220`, `REPORT.md:344-346` and `results/decoder_tables.md`.
- **What is wrong:**
  - `_soft_point` keeps whole bins whose upper edge is `<= log10(1 - t)`. The grid step is 0.05 in `log10(1 - c)`, so the rule actually applied is `confidence > t_eff`.
  - `t_eff` equals `t` only for 0.9, 0.99, 0.999 and 0.9999. Otherwise: 0.5 -> **0.5533**, 0.8 -> 0.80047, 0.95 -> 0.95533, 0.97 -> 0.97182, 0.98 -> 0.98005, 0.993 -> 0.99369, 0.995 -> 0.99553, 0.997 -> 0.99718, 0.998 -> 0.998005, 0.9995 -> 0.99955.
  - The comment at `aggregate.py:57` says that bin b is `(edges[b-1], edges[b]]`. With `side="right"` it is actually `[edges[b-1], edges[b])`, so acceptance is strict (`>`) even for thresholds on the grid. The report defines the rule as `>= t` (`REPORT.md:220`).
  - The confidence distribution has large point masses near 0.5. At p = 1e-3, about 6.7% of all study shots (1.80e6 of 2.7e7) fall in the single bin `c` in (0.499, 0.553] (`v3_soft_calibration_output.txt`).
- **Evidence:**
  - Exact enumeration of all 81^3 histories (m = 2, p = 5e-3, `v3_rb_d1_p0.005_output.txt`):
    - label 0.5: the literal rule accepts 0.911364 with P(correct) 0.852109; the rule applied accepts 0.872616 with P(correct) 0.866763;
    - label 0.95: 0.413710 vs 0.383624;
    - label 0.97: 0.289257 vs 0.264201.
  - Sampled with the study's own sampler and filter at p = 1e-3, sequences 0 to 2, 2e4 shots each (`v3_soft_threshold_effect_output.txt`):
    - t = 0.5 at m = 22 (depth index 8): the literal rule gives acceptance 0.827 and P(correct) 0.844; the rule applied gives 0.733 and 0.886;
    - t = 0.97 at m = 22: 0.3336 / 0.98946 vs 0.3267 / 0.98986.
  - So the `REPORT.md:346` row "Soft MLD, t = 0.5" (acceptance 0.724 and P(correct) 0.88747 at m = 22), and every "MLD, reject if confidence < 0.5" row of `decoder_tables.md`, describe t ≈ 0.553.
  - The t = 0.97 rows (`REPORT.md:59`, `REPORT.md:344`) describe t ≈ 0.972. That effect is small, about 2% relative in acceptance.
- **Not affected:**
  - The frontier itself. Each point is still a valid (acceptance, fidelity) pair, and `extra_analyses.pareto_rows` labels its points by the bin edge.
  - The t = 0.9 and 0.99 numbers.
- **Fix:** do one of the following:
  - put `log10(1 - t)` of every reported threshold on the grid (add those edges to `CONF_EDGES`);
  - relabel the rows with `t_eff`;
  - store per-shot confidences.

### F2 (major): `baseline_reproduction.csv.gz` does not test what it claims, although the claim is true

- **Where:** `studies/correction-decoders/qutrit/scripts/analyze_study.py:146-157`; quoted at `REPORT.md:243`.
- **What is wrong:**
  1. **Unpaired standard error.** Study and production use the *same* 30 circuits, so the comparison is paired and the across-sequence spread cancels. Yet `rej_se` and `fid_se` add `rej_std**2` and `fid_std**2` to the shot variance.
     - Rejection SE: inflated by a median 2.7x (range up to 265x).
     - Fidelity SE: inflated by a median 1.7x (up to 75x).
     - The published `rej_z` therefore have sd 0.47 (after excluding the degenerate rows below). A median |z| near 0.4 shows little: a true offset of one shot-noise sigma would also give |z| near 0.4.
  2. **Degenerate rows.** The SE is built from the *observed* production value and floored at `1e-15`. When production saw zero errors (`fid_prod = 1`) or zero acceptances (`rej_prod = 1`), the z explodes.
     - 314 rows have |z| > 5.
     - Examples: `p=1e-4 const0 m=2: fid_z = -7.7e7` at acceptance 0.9986; `p=0.0075 unif1 m=22: rej_z = -6.6e6`.
  3. **The Rice bias is not removed by the cut.** The `n_acc >= 1000` "comparable" cut leaves rows in which the |E ω^k| Rice bias is up to 2.7e-2 (Monte Carlo from the exact distributions).
     - This bias is about `0.886/sqrt(N_acc)` only when F ≈ 0. Otherwise it is about `(1 - F)/(4 F N_acc)`, with a median of 4.7e-7 in the comparable rows.
     - 223 of the 224 rows with |z| > 3.7 have F < 0.5, so `REPORT.md:243`'s explanation of the tail is right except for the one degenerate row.
- **Correct test:** production saved per-circuit counts (`progress/<p_idx>/{unif,const}_check_data/<k>.npy` and `_rejected.npy`, with 1e6 noisy shots per circuit; the reference shot is dropped in `LRBSimulationPipeline.LRB`).
  - A per-circuit multinomial GOF of (acc&L=0, acc&L=1, acc&L=2, rejected) against `1e6 x` the study's exact joint, pooled over 270 circuits for each of the 630 (p, policy) families: chi2/dof ≈ 1 everywhere. The minimum pooled p is 0.011; none of the 630 is below 0.01 (6.3 expected).
  - Per-policy per-circuit p-values are uniform: KS p is 0.055 to 0.94 for unif1, 2 and 5 and const0, 1 and 3 (`v3_reproduction_output.txt`, `v3_reproduction_ks_output.txt`).
  - Paired z with the predicted Rice bias:
    - rejection, 5670 rows: sd 0.88, max |z| 4.46, a Poisson case with 0.045 accepted shots expected and 1 seen;
    - fidelity, 4210 rows: sd 0.98, max |z| 3.03.
- **Recommendation:** replace the CSV's z-columns with the per-circuit test. The validation claim "Exact LRB-D vs the 1M-shot production CSVs" is then **[VN]** at shot-noise precision.

### F3 (major): for exact strategies the zero-variance m = 0 point fixes the fitted r; the CIs ignore this

- **Where:**
  - `src/lrb/correction/aggregate.py:157-195` (`fit_curve`, weights `1/std^2` across sequences);
  - `metrics.py:45-53` (`_safe_weights`);
  - results quoted at `REPORT.md:342` and in `key_summary.md` / `fits_baseline.csv.gz`.
- **What is wrong:**
  - At m = 0 all 30 sequences are the same circuit. For exact strategies the across-sequence std is 0 or about 1e-16, so its weight is 1e14 to 1e30 times the others and the fit `y = b f^m` is forced through `y(0)`.
  - The pipeline fits the same policy from 1M-shot data, where m = 0 has shot-noise std (about 8.6e-5 at p = 1e-3). Sampled strategies (soft, MLD, window) are not forced either.
  - For curves that are not exponential, r then reflects the weighting, not the data.
- **Evidence** (`v3_fit_weighting_output.txt`, `v3_fit_weighting2_output.txt`), `lrbd:const1` at p = 1e-3:

  | Fit | r |
  |---|---|
  | published (m = 0 forced) | **2.96e-4**, CI [2.53e-4, 3.49e-4] |
  | pipeline fit of the production CSV for the same circuits | **4.52e-4** (outside the CI) |
  | study data with the production m = 0 std | 4.32e-4 |
  | m = 0 excluded | 5.05e-4 |
  | unweighted log-linear | 7.77e-4 |

  - At p = 2e-3, `const1` ranges from 1.37e-3 to 5.73e-3 across these conventions, and `lrbd:const0` at p = 1e-3 from 9.1e-6 to 2.1e-5.
  - For near-exponential curves the effect is 5-12%:
    - `unif1`: 2.00e-5 published vs 1.79e-5 with m = 0 excluded;
    - BFD: 5.30e-5 vs 4.87e-5;
    - `soft:0.99`: 5.80e-5 vs 5.16e-5.
  - Under every convention tested, the strategies keep the same order.
- **Impact:** the report labels `const 1` and `const 0` as "(non-exponential)", which is good. But their single-r values and bootstrap CIs (`key_summary.md`, `fits_baseline.csv.gz`) are not comparable with the pipeline's r for the same policy, and the CIs understate the uncertainty.
- **Suggestion:** for non-exponential rules, report the windowed rates only; or fit all strategies with m >= 2.

### F4 (major): `results/key_summary.md` still uses the free-offset RB reference

- **Where:** `results/key_summary.md`, column "physical RB r at 5p", written at 12:08 and never regenerated.
- **What is wrong:** it lists 0.00226, 0.00492, 0.00885 and 0.0199 at p = 5e-4, 1e-3, 2e-3 and 5e-3. These are the free-offset fits (offset 0.17, 0.23 and so on). `rb_reference.csv`, `decoder_tables.md` and `REPORT.md` now use the offset-0 values 0.00188, 0.00376, 0.00752 and 0.0186.
- **Why offset 0 is right:** the production mixed-fit table used offset 0. `rb_a = 0` in all 330 rows, and `fit_decay(free_offset=False)` reproduces its `rb_f` exactly, while the free fit differs by 7.1x at p_RB = 5e-4 and by 1.31x at 5e-3 (`v3_fits_output.txt`).
- **Fix:** regenerate `key_summary.md` or delete it. `REPORT.md:900` lists it as a deliverable.

### F5 (minor): the quasi-static mixture is "exact" only for p-independent rules

- **Where:**
  - `studies/correction-decoders/qutrit/scripts/analyze_study.py:167-177` (docstring "the mixed curve of every strategy is exact"; `MIXTURE_STRATEGIES` includes `soft:0.99` and `mld:all`);
  - `extra_analyses.mixture_curve_rows` (`mld:all`);
  - `REPORT.md:584-586`.
- **What is wrong:** mixing the per-p results of MLD or soft MLD models a decoder that knows each shot's p, that is, an oracle.
- **Evidence** (`v3_mixture_oracle_output.txt`, sequence 0):

  | Case | Rule | Oracle | Bayes mixture-posterior decoder |
  |---|---|---|---|
  | m = 10, p in {1e-3, 5e-3} | soft:0.99 acceptance | 0.2379 | 0.2194 |
  | m = 10, p in {1e-3, 5e-3} | MLD signed | 0.52963 | 0.52940 |
  | m = 22, p in {5e-4, 2e-3} | soft:0.99 acceptance | 0.2763 | 0.2675 |
  | m = 22, p in {5e-4, 2e-3} | MLD signed | 0.62372 | 0.62363 |

- **Impact:** the MLD numbers in REPORT §5.5 hold to about 1e-4. The `soft:0.99` rows of `mixture_nonmarkov.csv` describe an oracle. LRB-D, BFD and raw are exact as claimed.

### F6 (minor): fits at p >= 0.04 are dominated by floating-point noise

- **Where:** `aggregate.fit_curve`, `metrics.fit_decay`.
- **What is wrong:** at p = 0.04 and 0.05 the signed values of several strategies at large m are about ±1e-16, with std about 1e-16 and weights about 1e32. The "survival" and "signed" fits, which are algebraically identical, then disagree by up to 6.4e-4 in f. Examples: `lrbd:const1` p = 0.04 gives f 0.16881 vs 0.16945; `bfd:all` p = 0.05 gives 0.14978 vs 0.15025 (`v3_fits_output.txt`).
- **Impact:** those `fits_baseline.csv.gz` rows are numerically meaningless. The report only quotes p <= 2e-2.

### F7 (minor): the cited mask test does not cover the claim

- **Where:** `REPORT.md:241`.
- **What is wrong:** the report cites `tests/test_correction_chain.py` for "identical for depths 0 to 22", but `test_policy_masks_match_pipeline_postselection` loops over depths (0, 2, 4, 6, 10, 22) only.
- **Status:** the claim is true. `v3_masks.py` checks all 1012 single-depth (m, k) cases plus the production depth grid passed jointly (396 cases), with 0 mismatches. The citation should point at a test that covers the claim.

### F8 (minor): inaccurate comments

- `aggregate.py:57`: the bin convention is wrong (see F1).
- `metrics.py:10-11`: `survival = (1 + 2 signed)/3` is stated "for a depolarizing logical channel". It is an identity for every residual distribution, because Re ω = Re ω² = -1/2. It was verified to 3e-16 on every curve.
- `analyze_study.py:218`: "The pipeline fits RB with a free offset". The default config does, but the pipeline's production table was made with offset 0 (see F4).

---

## Checks that found nothing wrong (with outputs)

### (1) Sampler fidelity (`v3_sampler.py`, `v3_sampler_multiseed.py`, `v3_global_rng.py`)

- **Production and study models agree.** For circuit `c00` at depth indices 2 (m = 4) and 8 (m = 22) and p = 1e-3 and 5e-3, the production circuit file at that p yields a compact DEM with unit responses identical to the study's structure (which was compiled at p_idx 4). Its location probabilities equal `p * prob_scale` to within 3.5e-18.
- **Shots per run.** 2e6 shots per sampler at m = 4 and 1e6 at m = 22. The global np.random was seeded for the production sampler, and an explicit `Generator` was used for the study's sampler.

**Production and study samplers against exact references and each other:**

| Circuit | Sampler | Per-detector chi2/dof | Pairwise co-occurrence | Logical marginal p | Joint (L, all trivial) p |
|---|---|---|---|---|---|
| d2 p=1e-3 | production vs exact | 0.68 | z²/dof 0.89 | 0.80 | 0.16 |
| d2 p=1e-3 | study vs exact | 1.12 | z²/dof 1.14 | 0.85 | 0.57 |
| d2 p=1e-3 | two-sample | 0.91 | z²/dof 0.92 | 0.99 | 0.79 |
| d8 p=1e-3 | production vs exact | 1.03 | 4278 pairs, max abs z 2.89 | 0.57 | 0.80 |
| d8 p=1e-3 | study vs exact | 1.13 | max abs z 3.26 | 0.59 | 0.056 |
| d8 p=1e-3 | two-sample | 0.87 | Bonferroni p = 1 | 0.58 | 0.13 |
| d8 p=5e-3 | all three | consistent | Bonferroni p >= 0.085 | | |

- **Multi-seed calibration.** Under H0, E[X²/dof] = 1 exactly.

  | Circuit | Sampler | Seeds x shots | Per-detector X²/dof | Pooled max abs z |
  |---|---|---|---|---|
  | d2 p=1e-3 | study | 40 x 5e5 | 1.059 ± 0.051 | 2.20 |
  | d2 p=1e-3 | production | 12 x 5e5 | 1.150 ± 0.112 | |
  | d8 p=5e-3 | study | 20 x 3e5 | 0.982 ± 0.029 | 3.57 over 4278 pairs |
  | d8 p=5e-3 | production | | 1.008 ± 0.040 | |

- **The production pipeline's RNG is safe.** DEM compilation consumes the global `np.random` stream but does not reset it (the post-compile draws depend on the prior seed), so production batches of different circuits are not correlated.

### (2) RB estimator (`v3_rb.py`, `v3_npz_checks.py`, `v3_soft_calibration.py`)

- **Exact enumeration at m = 2, p = 5e-3.**
  - The joint sums to 1 + 2e-15.
  - Enumeration BFD and LRB-D equal `blip_forgiving_exact` and `lrbd_exact` exactly.
- **Many seeds at m = 2.** 160 seeds x 2e4 shots of the real `study.evaluate_chain`, run as two independent seed bases.
  - For MLD, windows 0/1/2, BFD and every non-empty soft threshold, the RB means agree with the exact values. |z_RB| <= 2.53 in all cases but one (-3.42 for `soft0.5`, k = 1), which became +1.95 under the second seed base; the combined value is -1.0.
  - Variance ratio var(count)/var(RB): 3-8 for MLD and the windows, 8-170 for the soft error classes.
- **m = 14 at p = 1e-3.** 30 seeds x 1e4 shots, RB vs count paired: |z| <= 2.54. Variance ratios are 1.0-410; they are near 1 for the k = 0 class of soft rules, whose noise is dominated by acceptance.
- **Production study outputs.** All 1890 sampled (sequence, p, depth) entries:

  | Comparison | Mean z |
  |---|---|
  | counts - RB, 8 decoders (conservative variance) | -0.008 to +0.026 |
  | BFD RB vs exact `bfd_all` | +0.03 |

  - The BFD acceptance exact binomial tests have #p < 0.01 = 18 (18.9 expected).
  - The soft histogram is calibrated: pooled per bin over sequences and depths, 305 bins with >= 5 expected errors give mean z = +0.03, sd = 0.95, and one |z| > 3 (0.8 expected).
  - The RB and count histograms have identical bin occupancies, so `confidence_histogram` and `soft_hist_rb` bin the same way.
- **Observation, not a finding.** Pooled raw counts give L=1 at 25,295,382 vs L=2 at 25,312,292 (z = -2.4). The model is exactly symmetric, and this is one of about 15 aggregate statistics, so it is consistent with chance. My own 2.1e6 study-sampler logical errors give z = -0.6.

### (3) Fits (`v3_fits.py`)

- **`fit_decay` vs the pipeline.**
  - Against `_fit_lrb_from_stats` (offset 0, ceiling cut) on all 675 production LRB CSVs: |Δf| = |Δb| = Δchi2 = 0.
  - Against `_fit_decay_parameter_free_a` on RB and RB_comparison: also 0.
  - It reproduces `lrb_f` of all 330 rows of `unif_lrb_vs_rb_table_all_mixed_fits.csv` exactly.
- **`fit_curve` survival conversion.** Offset 1/3 and scale 1.5 are correct: signed = 1.5(survival - 1/3) to 3e-16. The two fits agree except in the degenerate cases of F6.

### (4) Masks (`v3_masks.py`)

0 mismatches in 1012 single-depth checks plus 396 production-grid checks. The existing tests pass (`python -m pytest -p no:cacheprovider tests/test_correction_chain.py tests/test_correction_variants.py`: 11 passed in 79 s).

### (5) Reproduction (`v3_reproduction.py`, `v3_reproduction_ks.py`)

See F2: the conclusion holds under the correct per-circuit test. The REPORT's "rejection: max abs difference 2.8e-4" is the row `p=2e-3 unif2 m=4`, whose paired z is -3.09 (one of 5670 correlated rows).
