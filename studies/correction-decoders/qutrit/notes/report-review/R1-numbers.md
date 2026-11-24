# R1-numbers: numbers audit of REPORT.md

Reviewer R1, 2026-10-28. Lens: every number in `REPORT.md` (executive summary,
all tables, inline figures, derived ratios and scaling coefficients) checked
against the committed result files. Sections 13.1 and 13.2 were skipped as
instructed. Nothing outside `notes/report-review/R1-numbers*` was edited.

## How the audit was done

All scripts are in `notes/report-review/R1-numbers/`. Each one reads only
`results/` (or, where stated, the raw cache and `src/` read-only). Run them
after `module load python/3.11`. Each script's stdout is saved next to it as
`*_output.txt`.

| Script | What it recomputes | Run time |
|---|---|---|
| `audit_numbers.py` | Every table entry and ratio in the executive summary and Sections 2.4, 2.5, 3.3, 4.1 to 4.8, 5.3 to 5.5, 6 and 7.2, from the `fits_*`, `curves_*`, `rb_reference`, `mixture_*`, `pareto_frontier`, `mismatch_check`, `baseline_reproduction` and `first_order_check` files | 3 s |
| `extra_checks.py` | The checks behind the main findings: dense spread definitions, distance to the Pareto frontier, const 0 MLD windows, full MLD vs physical RB at every p, the z tail of the reproduction, SI1000 acceptance at hardware p | 3 s |
| `code_capacity_enum.py` | Section 3.1 table, by brute force over all 9^5 error patterns, as exact rational coefficients | 2 s |
| `catalog_check.py` | Section 10 catalog statistics and the per-Clifford fault weight (imports `src/lrb/code_definitions.py`) | 1 s |
| `check_bootstrap_count.py` | Which bootstrap count produced the committed CIs (raw `.cache_correction` npz at p = 1e-3) | 9 s |
| `shot_budget_check.py` | The Section 7.6 shot budget | 1 s |
| `timing_check.py` | The Section 2.2 decode time (cached chain structures) | 90 s |

## Overall verdict

Most of the report's numbers reproduce exactly from the result files. All of
Section 4.1, Section 4.7, Section 5.3, Section 5.5, Section 6 (except one
rounding), Section 2.4, Section 2.5, the code-capacity table of Section 3.1
and the catalog statistics of Section 10 check out. The problems are of four
kinds:

1. two wrong numbers (full MLD "11 to 15 p, 3 to 4 times physical RB at every
   p"; "4e7 raw shots per circuit");
2. claims stated as general that hold only at p = 1e-3 or only for part of the
   p range;
3. per-step coefficients quoted in mixed units (r, 1 - f, -ln F slope, eps);
4. hardware numbers that no longer match the H1 and H2 notes. Those notes were
   rewritten at 21:07 and 21:08, after the report (20:53).

Counts: 2 critical, 10 major, 17 minor.

## Findings

### Critical

**R1-N01 (critical). Full MLD vs physical RB.**
- Quote (Section 4.3, line 393): "Full MLD | 11 to 15 p, 3 to 4 times above physical RB at every p". The same claim is in Section 9, line 820: "r is 3 to 4 times worse than physical RB at every p."
- Problem: both ranges are violated by the committed fits. At p = 1e-2, r = 17.4 p and the ratio is 4.7. At p = 5e-3 the ratio is 4.1. At p = 1e-4 it is 2.9.
- Evidence (`fits_baseline.csv.gz` `mld:all` signed_r divided by `rb_reference.csv` rb_r):

  | p | 1e-4 | 2e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 | 1e-2 |
  |---|---|---|---|---|---|---|---|
  | r/p | 10.99 | 11.08 | 11.38 | 11.85 | 12.81 | 15.21 | 17.41 |
  | r / physical RB | 2.92 | 2.99 | 3.03 | 3.15 | 3.41 | 4.09 | 4.74 |

  The bootstrap CI of the ratio is [4.04, 4.13] at 5e-3 and [4.70, 4.78] at 1e-2. A log-linear fit over m <= 6 gives 4.24 and 4.86, so the result does not depend on the fit (`extra_checks.py` section 4).
- Fix: "11 to 17 p; 2.9 to 4.7 times physical RB (3.0 to 3.4 for p <= 2e-3)". Correct Section 9 the same way. The conclusion (worse than physical RB at every simulated p) stands.

**R1-N02 (critical). Detection shot budget.**
- Quote (Section 7.6, lines 778 and 779): "Detection needs about 4e7 raw shots per circuit to resolve `unif 1` at m = 22 to 10%".
- Problem: 4e7 is the total over the 30 sequences, not a per-circuit number. The figure is 30 times too large for the stated unit.
- Evidence: `shot_budget_check.py`, unif 1, p = 1e-3, `curves_baseline.csv.gz`:
  - lambda(0) = 0.992491 and lambda(22) = 0.991853, so the decay is 6.4e-4 and the target SE is 6.4e-5;
  - the per-shot std of Re w^R is 0.106 at m = 0 and 0.110 at m = 22;
  - acceptance is 0.939 at m = 0 and 0.0807 at m = 22;
  - result: 1.33e6 raw shots per circuit, 4.0e7 in total. The sequence spread adds 2%.

  The revised H2 says the same (`H2-other-platforms.md` lines 913 to 917): "The draft's '4e7 raw shots per circuit' ... was a total over sequences". It gives 1.75e6 per circuit for 10% on r.
- Fix: "about 1.3e6 raw shots per circuit (4e7 in total over 30 sequences)". Or use H2's revised per-circuit target.

### Major

**R1-N03 (major). p range of the MLD claim.**
- Quote (executive summary, lines 44 and 45): "it is worse than physical RB at every p simulated (1e-4 to 2e-2) **[VN]**".
- Problem: the sampled rules, `mld:all` included, exist only for p = 1e-4 to 1e-2. Nothing was simulated at 2e-2, so that part of the [VN] range has no data.
- Evidence: `mld:all` p values in `fits_baseline.csv.gz` are {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2}. `run_all_on_node.sh` uses `--sampled-probs 1e-4,...,1e-2`, and report Section 2.3 lists the same set.
- Fix: "(1e-4 to 1e-2)".

**R1-N04 (major). Causal-decoder losses stated without p.**
- Quotes: line 67, "Hard-decision causal decoders with lookahead 2, 1 and 0 are 4%, 9% and 31% worse than MLD". Line 415, "the loss is 31% (no lookahead), 9% (one round) and 4% (two rounds) in r". Line 822, "they lose 4 to 31%".
- Problem: these are the p = 1e-3 values (3.9%, 8.6%, 30.7%). The loss depends strongly on p, and none of the three places says so.
- Evidence (`fits_baseline.csv.gz`, win:w r divided by mld:all r, minus 1):

  | p | 1e-4 | 2e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 | 1e-2 |
  |---|---|---|---|---|---|---|---|
  | lookahead 2 | 0.6% | 1.1% | 2.3% | 3.9% | 6.2% | 8.9% | 9.2% |
  | lookahead 1 | 1.3% | 2.4% | 5.1% | 8.6% | 12.4% | 15.6% | 13.6% |
  | lookahead 0 | 21.5% | 23.0% | 26.5% | 30.7% | 35.1% | 35.2% | 27.1% |

- Fix: add "at p = 1e-3", and give the ranges over p (about 1 to 9%, 1 to 16% and 22 to 35%).

**R1-N05 (major). Dense-study BFD spreads do not match the column definition.**
- Quote (Section 5.4, lines 564 and 565, column "windowed-rate spread (max/min, m >= 2), LRB-D / BFD"): "3.8 / 2.0" and "**1.16 / 1.49**".
- Problem: the two BFD values come from windows that include the m = 1 to 2 window. With windows that start at m >= 2, the reading Section 5.3 uses for "windows m = 2 to 22", BFD's spread is 1.62 (baseline) and 1.13 (checked inverse). The bold "1.16 / 1.49" therefore reverses the comparison. With checks inside the inverse half, BFD is flatter than LRB-D (1.13 vs 1.16), not less flat. The 1.49 comes from a single boundary window.
- Evidence (`extra_checks.py` section 1, `fits_dense.csv.gz`):
  - BFD, checked inverse, p = 1e-3: the 1 to 2 window is 1 - rate = 172e-6, while every other window lies between 116e-6 and 131e-6.
  - Windows starting at m >= 2, LRB-D / BFD: baseline 3.79 / 1.62, checked inverse 1.157 / 1.130, parallel + terminal 1.56 / 1.31.
  - Windows ending at m >= 2: baseline 3.79 / 2.01, checked inverse 1.157 / 1.489. These are the report's numbers.
  - p = 2e-3 and 5e-3 behave the same (BFD checked inverse 1.13 and 1.12 against 1.50 and 1.53).
- Fix: use 1.62 and 1.13 under the m >= 2 header. Alternatively, relabel the column "windows from m = 1" and say BFD's excess is one m = 1 to 2 window.

**R1-N06 (major). Unsupported rise of the const 0 MLD rate.**
- Quote (Section 4.6, line 445): "MLD's windowed rate rises about 10% over m = 2 to 22."
- Problem: the data do not show a 10% rise.
- Evidence (`fits_baseline.csv.gz` `mld:const0`, p = 1e-3, `extra_checks.py` section 3):
  - 1 - rate goes from 2.48e-3 in the 2 to 4 window to 2.61e-3 in the 20 to 22 window, which is +5.6%.
  - Each window's bootstrap CI is about ±8%.
  - The windows from m = 2 onward are not monotonic: 2.48, 2.39, 2.58, 2.63, 2.57, 2.45, 2.61 (e-3).
  - The curvature is c2 = -1.8e-6 [-8.6e-6, +6.7e-6], which includes 0.
- Fix: "MLD's windowed rate is flat within the ±8% window uncertainty (c2 consistent with 0)". Or drop the clause. The detection half of the sentence (m^2) is right: the const 0 LRB-D windows grow 26 times.

**R1-N07 (major). unif 1 "on the frontier" at m = 6 and 22 is not tested by the data.**
- Quote (Section 4.2, line 368): "and holds within the precision of the sampled frontier at m = 6 and 22."
- Problem: the sampled frontier does not reach LRB-D unif 1's acceptance at any p. The CSV shows only that unif 1 has both lower acceptance and lower infidelity than the last frontier point. That neither confirms nor refutes that it lies on the frontier.
- Evidence (`pareto_frontier.csv`; the lowest-acceptance point is the bin edge log10(1 - t) = -2.25):

  | p, m | frontier minimum acceptance | unif 1 acceptance |
  |---|---|---|
  | 1e-3, 6 | 0.535 | 0.4825 |
  | 1e-3, 22 | 0.093 | 0.0807 |

  The same holds at 5e-4, 2e-3 and 5e-3 (`extra_checks.py` section 2).
- Fix: rerun the confidence histogram with finer bins below -2.25, or restrict the claim to the m = 0 and 2 enumeration. The executive summary sentence "already sits on that frontier" then rests on m = 0 and 2 only.

**R1-N08 (major). "BFD sits on it" contradicts the treatment of unif 2.**
- Quote (Section 4.2, lines 370 and 371): "BFD sits on it, `const 1` is dominated". Line 369 says "`const c` and `unif 2` lie strictly below it **[VN]**".
- Problem: at equal acceptance, BFD's accepted infidelity is further above the frontier than unif 2's, yet unif 2 is called strictly dominated and BFD is said to sit on the frontier.
- Evidence (`extra_checks.py` section 2): accepted infidelity above the frontier at the same acceptance, with the frontier interpolated linearly in accepted error mass:

  | p | m = 6, BFD | m = 6, unif 2 | m = 22, BFD | m = 22, unif 2 |
  |---|---|---|---|---|
  | 1e-3 | +1.5% | +0.8% | +4.1% | +2.9% |
  | 5e-4 | +0.7% | outside range | +2.1% | +1.5% |
  | 2e-3 | +2.9% | +1.3% | +7.4% | +5.7% |
  | 5e-3 | +7.2% | +3.7% | +17.1% | +14.6% |

  At p = 1e-3 BFD's neighbouring frontier points are only 0.011 apart in acceptance, so the gap is not an interpolation artifact. Linear-in-mass interpolation is conservative: the true frontier is at least that good.
- Fix: "BFD lies close to, but strictly below, the frontier (1.5% and 4% higher accepted infidelity at p = 1e-3, m = 6 and 22)".

**R1-N09 (major). The hybrid's coefficient is 1 - f, not r.**
- Quote (Section 3.3, lines 325 and 326): "Accepted-shot errors stay second order for LRB-D (`r` about 20 p^2), BFD (about 53 p^2) and the hybrid (about 1.3e3 p^2)".
- Problem: the LRB-D and BFD values are r (`fits_baseline`: r/p^2 = 19.4 to 20.9 and 50.5 to 54.6 for p <= 2e-3). The hybrid's 1.3e3 p^2 is A1's 1 - f. In r units it is 0.85e3 to 0.89e3 p^2, about 870 p^2. No results file contains the hybrid; its rows in Section 3.3 come only from notes/A1.
- Evidence: `A1-syndromes.md` line 607, "hybrid ... ≈ 1.5 × (0.85–0.89)·10³ p² ≈ 1.3·10³ p²" in a column headed "per-depth 1 − f".
- Fix: "the hybrid (r about 870 p^2, i.e. 1 - f about 1.3e3 p^2) [VN, notes/A1]". Also say that the hybrid rows of the Section 3.3 table come from A1, not from the fits.

**R1-N10 (major). "27 p^2" is in different units from the rest of the report.**
- Quotes: Section 8, lines 797 and 798: "about 27 p^2 per step here, against an estimated 90 to 1,900 p^2 for a corrected d = 3 code". Section 9, line 828: "Estimated 90 to 1,900 p^2 vs 27 p^2."
- Problem 1: the 27 p^2 is A5's eps = (4/3) r, the depolarizing probability, computed with the pipeline-table r = 2.03e-5. Everywhere else the report gives detection as r about 20 p^2. A reader sees 20 p^2 (Sections 3.3 and 4.3), 22.7 p^2 (Section 5.3) and 27 p^2 (Sections 8 and 9) for the same per-step error, with no units stated.
- Problem 2: "estimated 90 to 1,900" merges A5's optimistic lower bound (at least 90) with its realistic estimate (460 to 1,900).
- Evidence: `A5-distance3.md` lines 36 to 39 and 437 to 444 ("eps = (4/3) r = 27.1 p^2"; "A kappa_1^2 >= ... 90"; "realistic values ... about 460 to 1900"). `fits_baseline` unif 1: r/p^2 = 20.0 at p = 1e-3.
- Fix: state the unit ("eps = 4r/3: detection 27 p^2; corrected d = 3 at least 90 p^2 (optimistic bound), 460 to 1,900 p^2 realistic"). Or convert everything to r (20 p^2 vs at least 68, realistic 345 to 1,425 p^2).

**R1-N11 (major). Hardware acceptance claims conflict with the hardware notes they cite.**
- Quotes: executive summary, lines 110 to 113: "trapped-ion qutrit gates to about 0.014. At those values the baseline protocol keeps less than 0.1% of shots by `m = 2` to 4 **[LIT, VN]**". Section 7.2, lines 704 and 705: "At these p the baseline keeps less than 0.1% of shots by m = 2 (transmons) or m = 4 (ions) **[VN, notes/H2]**".
- Problem: under a gate-only SI1000 mapping the numbers reproduce. The hardware notes' own noise models contradict the conclusion, and the report does not say the claim depends on the gate-only mapping.
- Evidence:
  - Under the SI1000 mapping, `curves_baseline` unif 1 acceptance (log-interpolated in p) is 7.4e-4 at m = 2 for p = 0.027 and 9.0e-4 at m = 4 for p = 0.014 (`extra_checks.py` section 6).
  - Revised `H2-other-platforms.md` section 0.2: the ion_now location-by-location model "behaves like SI1000 at p ≈ 3e-3 to 5e-3, not at the p ≈ 0.014".
  - H2 lines 919 to 922: "the baseline rejects more than 99.9% of shots by m = 4 on ions does not hold for the ion model, which keeps 18% at m = 4 and 8% at m = 6".
  - Revised `H1-sc-hardware.md` section 0.4: the duration-aware HW-now model "keeps 0.5-7% of shots at m = 2".
- Fix: qualify the claim as "under a two-qutrit-gate-only SI1000 mapping", and add the hardware-model numbers (transmons 0.5 to 7% at m = 2; ions 18% at m = 4).

**R1-N12 (major). Section 7 hardware figures not supported by the current H1.**
- Quotes:
  - line 689: "a waiting transmon qutrit loses 1.4 to 9% per µs";
  - lines 707 and 708: "about 12% per data qutrit per sequential round at 0.6 µs gates";
  - line 777: "About 8e6 shots per full dataset, under 1 h at 300 µs per shot on transmons [notes/H1]".
- Problem: none of these numbers appears in the cited note. H1 and H2 were rewritten at 21:07 and 21:08, after the report (20:53), so this is a re-sync item.
- Evidence (`H1-sc-hardware.md`):
  - lines 744 and 745: "loses 1.4%/µs (echo), 2.9%/µs (Ramsey) or 4.0%/µs (IQM-like assumption)", so the maximum is 4%/µs, not 9%;
  - lines 60 to 62: "a check round lasts 6-12 µs, during which a data qutrit idles at 1.4-2.9% per µs", which is about 8 to 35% per round; no 12% at 0.6 µs is stated;
  - lines 133 to 135: "Use about 100 sequences of 3,000 shots per depth: 2.7e6 shots for 9 depths, 0.07-0.75 h at 0.1-1 ms per shot".

  The report's own arithmetic is consistent (8e6 x 300 µs = 40 min).
- Fix: re-sync with the revised H1: 1.4 to 4%/µs; the idle per round as H1 states it; 2.7e6 shots and 0.07 to 0.75 h.

### Minor

**R1-N13 (minor). MLD first-order range excludes p = 1e-4.**
- Quote (line 322): "Full MLD failure | 11 p, so `1 - f` is 16.6 to 17.7 p, i.e. `r = 11.1` to `11.8 p`".
- Problem: the fits give r/p = 10.99, 11.08, 11.38 and 11.85 at p = 1e-4 to 1e-3, so 1 - f = 16.5 to 17.8 p. The stated range leaves out p = 1e-4. Also, "11 p, so" implies 1 - f = 16.5 p exactly.
- Fix: "r = 11.0 to 11.9 p (1 - f = 16.5 to 17.8 p) for p = 1e-4 to 1e-3".

**R1-N14 (minor). Two rounded ratios.**
- Quotes: line 401, "2.7 times the conditional error"; line 466, "with a 6 times lower error rate".
- Problem:
  - BFD/LRB-D is 5.3027e-5 / 2.0030e-5 = 2.65, which rounds to 2.6. The 2.7 comes from rounded inputs (5.3/2.0).
  - const 1 / BFD is 2.959e-4 / 5.303e-5 = 5.58, so "5.6 times".
- Fix: 2.6 and 5.6.

**R1-N15 (minor). Rounding and CI display.**
- Quotes: line 565, "8.6e-5, 0.875"; line 644, "2.2e-5, 0.930"; line 339, "2.0e-5 [2.0e-5, 2.0e-5]".
- Problem:
  - Dense checked-inverse BFD has r = 8.549e-5, so 8.5e-5.
  - The parallel + terminal + checked-inverse BFD acceptance is 0.92946, so 0.929. The 0.930 is double rounding via `variant_summary.md`'s 0.9295.
  - The unif 1 CI is [1.951e-5, 2.044e-5]. At 2 significant figures it collapses to a zero-width interval.
- Fix: 8.5e-5; 0.929; [1.95e-5, 2.04e-5].

**R1-N16 (minor). Soft MLD range.**
- Quote (line 392): "4 to 7 times below physical RB from p = 2e-4 to 1e-2".
- Problem: physical RB divided by soft 0.9's r is 5.12, 4.06, 3.99, 4.35, 6.38 and 7.74 at p = 2e-4, 5e-4, 1e-3, 2e-3, 5e-3 and 1e-2.
- Fix: "4 to 8 times (3.99 to 7.74)".

**R1-N17 (minor). const 0 MLD ratio stated four ways.**
- Quotes: line 74, "`r = 0.45 x` physical RB at every p"; line 394, "0.44 to 0.46 times physical RB at every p"; line 427, "0.44 times physical RB"; line 764, "0.45 times physical RB".
- Problem: the data give 0.443 to 0.450 for p <= 1e-2 (0.444 at p = 1e-3). Higher p gives 0.456 (0.02), 0.464 (0.03), 0.477 (0.04) and 0.489 (0.05), so "0.44 to 0.46 at every p" fails at 0.04 and 0.05. r/p goes from 1.67 down to 1.56.
- Fix: one value, 0.44, and "0.44 to 0.49 over all 14 p".

**R1-N18 (minor). Parallel-readout factor stated without p.**
- Quote (line 72): "cuts every strategy's error rate by 1.6 to 3 times".
- Problem: this holds at p = 1e-3 (1.63 to 3.10), but the executive summary gives no p. Over p = 5e-4 to 5e-3 the range is 1.44 (soft 0.97 at 2e-3) to 3.23 (BFD at 5e-3).
- Fix: add "at p = 1e-3".

**R1-N19 (minor). Physical RB slope.**
- Quote (line 29): "`p_RB = 5p` has `r = 3.76 p` per depth step".
- Problem: `rb_reference.csv` gives rb_r/p = 3.70 to 3.76 for p <= 2e-3, then 3.72 (5e-3), 3.67 (1e-2), 3.56 (0.02) and 3.18 (0.05).
- Fix: "r = 3.76 p at small p".

**R1-N20 (minor). Per-Clifford fault weight.**
- Quote (line 162): "while a logical Clifford carries 3.3 to 4.9 p".
- Problem: individual catalog Cliffords carry 0.1 to 8.1 p (mean 4.78 p, median 4.9 p; `catalog_check.py`). The range 3.3 to 4.9 p is A3's per-Clifford average over two sequences at m = 2 and 22 (`A3-pauli-frame-offline-causal.md` lines 472 to 481).
- Fix: "about 4.7 p per logical Clifford on average (0.1 to 8.1 p for individual Cliffords)".

**R1-N21 (minor). The Rice-bias wording is too general.**
- Quotes: line 243, "The z tail sits at low acceptance, where the pipeline's `abs(E[omega^k])` estimator is biased up by about `0.89/sqrt(N_acc)`". Lines 845 to 847, "biased upward by about `0.89/sqrt(N_acc)` when few shots are accepted. This explains every large deviation".
- Problem: the numbers are right, but the stated condition is wrong. 0.886/sqrt(N) is the bias when the true polarization is about 0. When lambda is much larger than 1/sqrt(N), the bias is O(1/N), whatever the acceptance.
- Evidence (`extra_checks.py` section 5):
  - The 4,210 / 0.43 / 3.7 values reproduce.
  - Of the 474 rows with |z| > 3, 473 have an exact polarization of about 0 (p >= 1e-2, deep circuits). For those, (production - exact) / (0.886/sqrt(N_acc)) has median 0.99 (5% to 95%: 0.85 to 1.13).
  - The largest |z| (-7.7e7; p = 1e-4, const 0, m = 2) is a zero-variance artifact: 0.99999992 vs 1.0 with production std 0. It is not Rice bias.
- Fix: "biased up by about 0.89/sqrt(N_acc) when the true fidelity is near 0"; note the degenerate row.

**R1-N22 (minor). Reproduction recipe.**
- Quotes: line 262, "95% intervals from 100 bootstrap resamples"; line 882, "exactness and consistency tests (about 3.5 min)"; line 208, "1e5 shots of a depth-22 circuit decode in 6 to 11 s on one core".
- Problem:
  - The committed CIs reproduce exactly with boot = 100 (`check_bootstrap_count.py`), but `analyze_study.py`, as run in Section 12, defaults to `--boot 200` and gives different CIs. For example, MLD becomes [1.1645e-2, 1.2033e-2] instead of [1.1650e-2, 1.1992e-2].
  - The only recorded test run, `.cache_correction/pytest_final.log`, is "11 passed in 34.98s".
  - The forward filter on 1e5 depth-22 shots took 15 to 17 s at p = 1e-3 and 30 s at p = 5e-3, single-threaded on a login node (`timing_check.py`).
- Fix: add `--boot 100` to the Section 12 command; update the test time; state the p and the hardware behind "6 to 11 s".

**R1-N23 (minor). Stale `results/key_summary.md`.**
- Quote (line 900, results list): "`results/` | `decoder_tables.md`, `variant_summary.md`, `key_summary.md`, ...".
- Problem: `key_summary.md` gives physical RB as the free-offset fit (4.92e-3 at p = 1e-3, 2.26e-3 at 5e-4), which the report itself calls ill-conditioned. The report uses 3.76e-3. With the key_summary value, "3.1 times worse" would read as 2.4. Its column is also headed "r per Clifford", but r is per depth step.
- Fix: regenerate `key_summary.md` from `rb_reference.csv` rb_r, or delete it.

**R1-N24 (minor). Unmarked lead-fetched citations.**
- Quote (line 928): "or, for the two marked entries, fetched by the lead".
- Problem: no entry in Section 14 is marked. The two entries are Hrmo2023 and Ringbauer2022. They are the only Section 14 keys missing from `bibliography_raw.csv`, which has 214 rows (188 confirmed, 26 corrected; the line 908 counts are right). They carry the executive summary's ion number, which is labelled [LIT].
- Fix: mark the two entries.

**R1-N25 (minor). Hardware table inconsistencies.**
- Quotes: line 442, "ternary readout error of 2 to 7% per qutrit" against line 698, "93 to 97%". Line 111, "(typically 0.05 to 0.18)" against line 696, "others 82 to 89% | p about 0.027 to 0.18".
- Problem:
  - 93 to 97% is a 3 to 7% error; H1 states 2 to 7% in 140 to 750 ns.
  - The 0.05 lower end of "typically 0.05 to 0.18" is Goss's CZ at 95.2% (`lit-L6-scqutrit.md` lines 52 and 921). The table omits that gate, so its "others 82 to 89%" (0.11 to 0.18) does not support 0.05.
- Fix: use 2 to 7% in both places, and add "CZ 95.2%" to the table.

**R1-N26 (minor). "Never accepts a first-order logical error" needs a qualifier.**
- Quote (lines 51 and 52, and line 401): "It never accepts a first-order logical error **[VN]**".
- Problem: BFD accepts the 4.95 p final-round floor, as every rule does. In `first_order_check.json` at p = 1e-4, m = 0, exact_bfd_logical = 4.92e-4. The json's `bfd_accepted_logical = 0` counts only forgiven errors.
- Fix: "never accepts a first-order logical error beyond the 4.95 p floor that every rule accepts".

**R1-N27 (minor). Test tolerance.**
- Quote (line 240): "within 4 sigma at 1.5e5 to 2e5 shots".
- Problem: the tests use atol = 4 sigma + 1e-4 (`tests/test_correction_chain.py` lines 125 and 129; `tests/test_correction_variants.py` line 66).
- Fix: "within 4 sigma + 1e-4".

**R1-N28 (minor). Figure p does not match the table.**
- Quote (line 568): "![Exact windowed rates at every depth, p = 2e-3](results/figures/dense_windows_p0.002.png)".
- Problem: the figure sits under the p = 1e-3 table. `dense_windows_p0.001.png` exists.
- Fix: use the p = 1e-3 figure, or say why p = 2e-3 is shown.

**R1-N29 (minor). The 22.7 slope is in different units, and exact-rule fits are pinned at m = 0.**
- Quote (line 547): "`unif k`: per-step rate about `22.7 k p^2` plus a sawtooth."
- Problem 1: 22.7 p^2 is A4's slope of -ln lambda per step, in 1 - f units, which is r of about 15 p^2. Sections 3.3 and 4.3 give unif 1 as r about 20 p^2.
- Problem 2: part of the gap comes from the fit. For exact rules the across-sequence std at m = 0 is about 1e-16, so the m = 0 point gets a weight of about 1e32. The fitted b then equals y(0) to 10 digits (`audit_numbers.py` section P), and r absorbs the m = 0 to 2 boundary transient:
  - unif 1: fitted 1 - f = 30 p^2, while the mean of the windows from m = 2 onward is 25.6 to 26.7 p^2 (r about 17 to 18 p^2);
  - BFD: fitted 53 p^2 against about 48 p^2 asymptotically.
- Fix: state the unit at line 547, and note in Section 2.5 that exact-rule fits are pinned at y(0).

## Numbers checked and found correct

These reproduce from the result files to the printed precision:

- **Executive summary.**
  - Values: r = 1.18e-2 (MLD); 3.1 times physical RB (3.15); 600 times LRB-D (592); 16 of 40; P_fail = q + (63/16) q^2; rejection per step 0.106 to 0.070; acceptance at m = 22 8.1% to 19.1%; 2.0e-5 to 5.3e-5; 70 times (70.8).
  - Soft 0.9: 3.1%, 9.4e-4, 4 times (3.99), 47.7%. Soft 0.97: 4.8%, 2.4e-4.
  - 0.99499 to 0.99998.
  - 118 p per round (118.4 p, 100.4 p final; recounted from the brief's noise model) and about 4.7 p per Clifford (4.78 catalog mean).
  - 18 times (18.01) and 26 times (26.25); -15% (0.85); flat to 16% (1.157); 10% acceptance cost (9.6%); mixture values 0.9908, 0.9958, 0.9945, 0.974 and 0.617.
- **Section 1:** 4 x 2p + 2p = 10p.
- **Section 2.1:** 270, 1,620 and 2,070 circuits (node logs); 243 states; 3 x 81 x 81.
- **Section 2.3:** shot counts (1e5, 5e4, 4e4) and the 7 sampled and 14 exact p values.
- **Section 2.4:** 5,670 rows; max rejection difference 2.8e-4 (2.78e-4); 4,210 rows; median |z| 0.43; 95th percentile 3.7; 6.315e-3 vs 6.294e-3; 1.402e-3 vs 1.407e-3; 1 - exp(-0.2516) = 0.2224 against the exact 0.2224; r 1.99e-5 vs 2.0e-5; a 0.894 vs 0.894.
- **Section 2.5:** offset 0.86 (0.855); r = 3.76e-4, which equals the production mixed-fit table's rb_error_rate of 3.7636e-4; 100 bootstrap resamples (verified).
- **Section 3.1:** all six polynomials (`code_capacity_enum.py`, "any logical" convention); 24 unique; 8 tied pairs differing by X0X3, X2X4, Z0Z2 and Z3Z4.
- **Section 3.2:** 4.95 p, and its four sources.
- **Section 3.3:** unif 1 rejection 106 to 112 p (1 - a = 105.6 to 111.0 p; -ln a = 111.6 p); BFD 70 p (69.7 to 70.7); r about 20 p^2 and about 53 p^2; physical RB 1 - f = 5.6 p (5.635).
- **Section 4.1:** all 15 rows match `decoder_tables.md` and the underlying fits and curves.
- **Section 4.2:** the confidence cap of 0.995 at p = 1e-3; soft 0.99 accepts nothing at p = 2e-3 (acceptance 0 at m = 0).
- **Section 4.6:** 9.1e-6; 0.992; 1.67e-3; 1.18e-2, 1.50e-2 and 2.01e-2.
- **Section 4.7:** all 24 entries.
- **Section 4.8:** 0.2214; 0.2216 to 0.2226; 0.54 and 3.9e-2; 0.48 and 2.5e-2; 3 sequences, m = 10 and 22, scales 1/4 to 4.
- **Section 5.3:** all window values and all c2 values with CIs; a1 = 7.43 (7.425); 2.24 m^2 p^2 (exact -ln F at m = 22, p = 1e-3, is 1.10e-3 against 1.08e-3 predicted).
- **Section 5.4:** all r, a and c2 entries except the BFD spreads (R1-N05) and the 8.6e-5 rounding (R1-N15).
- **Section 5.5:** all 12 table values; 98% (97.8%); the mixed LRB-D fit returns f = 1, r = 0.
- **Section 6:** all table entries except one rounding (R1-N15); 1.6 to 3 times (1.63 to 3.10 at p = 1e-3); 0.894 to 0.930; 5.5 times (5.51); 1.6% (1.61%).
- **Section 7.1:** 180/180, 40/40, 0.985 vs 0.941 and 0.950 vs 0.9025 match A3; FPGA loops of 0.2 to 0.6 µs match H2 (0.18 to 0.55).
- **Section 7.2:** the SI1000 conversions (0.027; 0.03 to 0.07 via process infidelity = 4/3 x RB r; 0.006 to 0.014).
- **Section 8:** break-even 1e-4 to 6e-4 and 1.3e-3 to 3.4e-3 (A5); 3^12 x 3 states; 25 qutrits and 40 two-qutrit gates.
- **Section 10:** 216 strings, 207 distinct, the nine duplicates, 206 unitaries, class multiplicities 5, 5, 9 (x19), 10, 12 and 13, TV distance 3.7% (`catalog_check.py`).
- **Section 12:** about 1,800 npz files (420 + 900 + 450 = 1,770).
- **Section 13:** 214, 188, 26 and 0 unfound.
