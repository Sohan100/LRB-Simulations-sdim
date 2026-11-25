# V2: independent re-implementation of exact maximum-likelihood decoding

Reviewer id: V2-independent-mld. Scripts, logs and JSON outputs are in
`notes/review/V2/`. Every script inserts `src/` into `sys.path` and runs with
`module load python/3.11` on at most 4 threads.

## Verdict

The exact decoding machinery is correct to machine precision. My three
independent exact methods share no code with `lrb.correction`. I compared them
against `chain_dem` + `forward` + `decoders.lrbd_exact`/`blip_forgiving_exact`
and against the production npz files. Across depths 0, 2, 4, 14 and 22, the
largest relative deviation was 6e-14 and there were no argmax disagreements.
The production Rao-Blackwellized (RB) MLD estimates are unbiased: z-scores
against the exact failure are N(0, 1) over all 30 sequences.

There is one finding in the soft-output evaluation path. Ten of the fourteen
`soft:t` strategies are evaluated at a stricter threshold than their label.
This changes the reported `t = 0.5` and `t = 0.97` numbers (severity
critical by the brief's definition; conclusions are unchanged). There are two
minor items.

## What I built (no `lrb.correction` code)

`V2/v2_core.py` reads only `build_compact_detector_error_model` output:

- **Fault distribution.** Re-derived from
  `dem_simulation.sample_location_fault_vectors`:
  - N1 `d` is uniform over the 8 non-zero pairs; `f` is component 0 in {1, 2};
    `p` is component 1 in {1, 2}.
  - N2 is weighted by the location's own `prob_dist`, decoded with column 3
    least significant.
- **Activation.** Taken from `location.probability` and `prob_dist` of the
  circuit compiled at the requested p (`p_idx` 5 for 0.002, 8 for 0.01).
  `prob_scale` is not used. The chain's `prob_scale * p` agrees with it to
  4.3e-16 relative.
- **Effects.** Linear combinations of the `component_terms` unit responses,
  mod 3.
- **`dense_joint_naive`.** Dense array over Z_3^(n_det+1). Each location is
  convolved as one cyclic shift per distinct non-zero effect. This makes no
  assumption beyond the DEM.
- **`dense_joint_factored`.** Applies the identity
  `a0 P + a1 prod_c (I + S_u_c + S_2u_c) P` for locations that are uniform on
  a coordinate subgroup.
- **`SpanFilter`.** An exact forward filter for any linear DEM. Its state is
  the coordinate vector of the accumulated effect on all not-yet-observed
  trits, in an RREF basis of the span of past effects. It needs no transfer
  maps and no chain assumption. Its state dimension came out as
  `[5, 5, ..., 1]`: 4 syndrome trits plus the logical. That independently
  confirms the 243-state chain at depths 0 to 22.
- **`my_sample`.** An independent sampler.

Self-test (`v2_selftest.log`), c00, p = 0.01, depth 2:

| Comparison | Max relative deviation |
|---|---|
| naive vs factored dense | 1.0e-14 |
| span filter vs naive dense | 1.7e-14 |

The total probability is 1 to within 4e-15.

The code under test was evaluated in two ways:

- **fresh:** `chain_dem.build_chain_model(model, p)` on the same compiled model;
- **prod:** the production pickle `.cache_correction/structures/cXX_dY.pkl`,
  evaluated with `.at(p)` (built from the `p_idx` 4 circuit).

The two were identical in every check.

## Checks and outputs

### (a) P(history) and P(L | history) vs `forward.forward_posteriors`

Sources: `v2_check_dense.py`, `v2_check_dense_c0_d0.json`, `_c0_d1.json`,
`_c16_d1.json`.

Every history has P > 1e-12 here: 81 at depth 0 and 531,441 at depth 2,
including p = 0.002.

| Circuit | P(history) max abs / rel | P(L given h) max abs | P(h, L) max rel | argmax disagreements |
|---|---|---|---|---|
| c00, m = 0, p = 0.002 | 5.0e-15 / 7.0e-15 | 4.4e-16 | 8.0e-15 | 0 |
| c00, m = 0, p = 0.01 | 2.1e-15 / 5.5e-15 | 5.6e-16 | 5.6e-15 | 0 |
| c00, m = 2, p = 0.002 | 5.2e-15 / 1.9e-14 | 3.4e-15 | 2.1e-14 | 0 |
| c00, m = 2, p = 0.01 | 6.0e-16 / 1.3e-14 | 2.8e-15 | 1.3e-14 | 0 |
| c16, m = 2, p = 0.01 | 5.6e-16 / 1.6e-14 | 3.1e-15 | 1.8e-14 | 0 |

`forward_posteriors_shared` vs plain: at most 6.7e-16. No near-ties: the top
two posteriors are never within 1e-9.

### (b) Exact LRB-D vs `decoders.lrbd_exact`

Every one of the 2^(m+1) - 1 used-round masks agrees to at most 1.4e-14
relative. The production npz entries also agree:

- all 22 `unif` and 22 `const` policies: at most 1.1e-14;
- `lrbd_xtriv`: 9.6e-15;
- `bfd_all`: 9.8e-15.

Exact values for c00:

| m | p | Rule | Acceptance | Legacy fidelity |
|---|---|---|---|---|
| 0 | 0.002 | all rounds = terminal | 0.8817482 | 0.9848129 |
| 0 | 0.01 | all rounds = terminal | 0.5385288 | 0.9171589 |
| 2 | 0.002 | all rounds | 0.5710082 | 0.9845468 |
| 2 | 0.002 | terminal only | 0.6634090 | 0.9837449 |
| 2 | 0.01 | all rounds | 0.0620336 | 0.9082976 |
| 2 | 0.01 | terminal only | 0.1398969 | 0.8736456 |

### (c) Exact MLD failure, sum_s [P(s) - max_L P(L, s)]

Compared with a fresh 2e5-shot RB estimate (`sampler.sample_shots` +
`forward_posteriors_shared`) and with the production npz (1e5 shots).

| Circuit | Exact | RB, 2e5 shots (z) | Realized, 2e5 shots (z) | Production npz RB |
|---|---|---|---|---|
| c00, m = 0, p = 0.002 | 0.0282350 | 0.0281065 ± 1.4e-4 (-0.95) | 0.028485 (+0.67) | 0.0280848 |
| c00, m = 0, p = 0.01 | 0.1381264 | 0.1380776 ± 2.6e-4 (-0.19) | 0.139095 (+1.25) | 0.1380920 |
| c00, m = 2, p = 0.002 | 0.0688616 | 0.0688301 ± 2.8e-4 (-0.11) | 0.068965 (+0.18) | 0.0692053 |
| c00, m = 2, p = 0.01 | 0.3506554 | 0.3508287 ± 4.0e-4 (+0.44) | 0.349625 (-0.97) | 0.3503318 |

The restricted-observation MLD (`mld_terminal`, `mld_unif2`), the window
decoders (`win0` to `win2`, exact expected failure of `window_decode` on every
history) and `soft:0.9` acceptance all agree with the npz within sampling
error. Example, c00, m = 2, p = 0.002:

| Rule | Exact | npz |
|---|---|---|
| `win0` | 0.081257 | 0.081515 |
| `mld_terminal` | 0.077700 | 0.078097 |

### All 30 production sequences, m = 0 and 2, p = 0.002 and 0.01

Sources: `v2_check_allseq.py`, `v2_allseq_summary.log`.

- **Exact rules vs npz.** LRB-D `unif 1`/`const 1` and BFD: at most 1.5e-14
  relative.
- **Exact rules vs `curves_baseline.csv.gz`.** Clifford-averaged acceptance
  and legacy fidelity agree to at most 5e-11 (the CSV has 10 significant
  digits).
- **RB MLD z-scores.** z of each production RB estimate against the exact
  failure uses the exact per-shot variance. Over the 30 sequences the
  z-scores have mean -0.07 to +0.31, std 0.84 to 1.07 and max |z| 2.40.
  `mld_terminal` and `soft:0.9` acceptance behave the same way.
- **Against the 1M-shot production LRB-D CSVs** (DEM sampler), using my exact
  per-sequence values averaged over 30: |z| ≤ 1.78 for both acceptance and
  fidelity, for `unif 1` and `const 1`.

### (a) at depth 4 (c00, 20 detectors)

Sources: `v2_check_depth4.py`, `v2_check_depth4.json`.

**Why a dense joint is infeasible.** A dense joint needs 3^21 = 1.05e10
doubles (84 GB). Restricting the output to histories with at most two
non-zero rounds does not shrink the state. A round that ends up trivial must
still be tracked through the convolution, because later faults can cancel
earlier ones. Its trits can only be dropped after every location touching it
is processed, and the early data faults touch all later rounds.

So I used two exact substitutes.

**1. Span filter on all histories with at most two non-zero rounds**
(1 + 5·80 + 10·80² = 64,401 histories). It agrees with `forward_posteriors`:

| p | Histories with P > 1e-12 | Mass covered | P(history) max rel | P(L given h) max abs | argmax disagreements |
|---|---|---|---|---|---|
| 0.002 | 58,511 | 70.2% | 1.7e-14 | 2.6e-15 | 0 |
| 0.01 | 64,401 | 10.8% | 2.3e-14 | 2.3e-15 | 0 |

**2. Assumption-free dense marginals over three rounds** (3^13 arrays), for
round triples {0,1,2}, {0,2,4} and {1,3,4}, against `forward_posteriors` with
`observed_rounds` set to the triple. This exercises the marginalization path
on all 531,441 observed histories:

| Quantity | Max deviation |
|---|---|
| P(observations), relative | 6.0e-14 |
| P(L given observations), absolute | 4.4e-15 |

Also at depth 4: all 31 LRB-D masks agree to at most 1.1e-14 and BFD to
9.0e-15. My slot assignment is identical to `chain_dem`'s.

### Deep circuits (c00, depths 4, 14, 22; p = 0.002 and 0.01)

Sources: `v2_check_deep.py`, `v2_check_deep.json`.

On 2e4 histories drawn by my sampler, the span filter and
`forward_posteriors_shared` agree:

| Quantity | Max deviation |
|---|---|
| posterior, absolute | 4.3e-15 |
| log-evidence | 1.3e-13 |
| argmax disagreements | 0 |

- All 44 LRB-D policies: at most 6.7e-14 relative to the npz.
- BFD: at most 6.2e-14.
- `unif_mask` and `const_mask` equal my own masks from the brief's
  definitions.
- My independent RB MLD-failure estimates vs the npz: z = -0.85, -1.08,
  +0.73, -0.98, +0.31, +2.07.

### Report claim (§4.2, notes/A2): `unif 1` is on the soft-MLD frontier at m = 0 and 2

Verified exactly (`v2_frontier_check.log`). The all-trivial history has the
largest posterior confidence:

- at m = 0, and at m = 2 for all 30 sequences;
- at p = 2e-4, 1e-3, 2e-3 and 1e-2.

The smallest margin is 6.1e-6: c13, p = 2e-4, against history [3, 0, 0], a
round-0 blip.

### Sampler

Sources: `v2_sampler_depth0.log`, `v2_zcond_many_c16_p0.01_300.log`,
`v2_zcond_allseq_p0.01.log`.

`sampler.sample_shots` matches the exact joint:

- **Depth 0, 1e7 shots.** Pearson chi-square p = 0.011 at p = 0.002 and 0.26
  at p = 0.01. My own sampler gives 0.17 and 0.75.
- **c16, depth 2, 300 seeds × 1e5 shots.** z_cond = (realized failures minus
  their conditional expectation under the exact posterior) / sd has mean
  +0.04 and std 0.98.

## Findings

### F1 (critical): soft-threshold strategies are evaluated at a stricter threshold than their label

**Where.**

| File | Lines | Role |
|---|---|---|
| `src/lrb/correction/aggregate.py` | 32, 55-63 | `SOFT_THRESHOLDS`, `_soft_point` |
| `src/lrb/correction/study.py` | 31 | `CONF_EDGES` |
| `studies/correction-decoders/qutrit/scripts/make_tables.py` | 21-23 | labels |
| `REPORT.md` | 59-60, 220, 344, 346, 402, 773 | statements using the labels |
| `results/decoder_tables.md` | rows "reject if confidence < 0.97" and "< 0.5" | tables |
| `results/variant_summary.md` | `soft:0.97` rows | tables |

**What.**

- The report defines soft MLD as "accept iff `max_L P(L | s) >= t`" (REPORT
  line 220).
- `_soft_point` accepts whole histogram bins whose upper edge, on the 0.05-step
  grid in log10(1 - conf), is at most log10(1 - t).
- So it accepts iff conf > t_eff = 1 - 10^(largest grid edge ≤ log10(1 - t)).
- Only t = 0.9, 0.99, 0.999 and 0.9999 lie on the grid.

| Label t | 0.5 | 0.8 | 0.95 | 0.97 | 0.98 | 0.993 | 0.995 | 0.997 | 0.998 | 0.9995 |
|---|---|---|---|---|---|---|---|---|---|---|
| t_eff | 0.5533 | 0.8005 | 0.9553 | 0.9718 | 0.9800 | 0.9937 | 0.9955 | 0.9972 | 0.9980 | 0.99955 |

**Evidence that production uses t_eff** (`v2_soft_npz_check.log`). Production
`soft:0.5` acceptance (c00 to c02, m = 2, p = 2e-4, 1e-3, 2e-3) matches the
exact t_eff rule within |z| ≤ 1.2. It is 19 to 82 standard errors away from
the stated rule.

**Size, m = 2** (exact, 10 sequences; `v2_soft_threshold.log`):

| p | t | Acceptance: stated / implemented | Accepted infidelity: stated / implemented |
|---|---|---|---|
| 1e-3 | 0.5 | 0.9932 / 0.9766 | 3.54e-2 / 2.77e-2 |
| 1e-2 | 0.5 | 0.634 / 0.528 | 0.285 / 0.247 |

**Size, m = 22** (c00, my sampler with exact posteriors, 2e4 shots):

| p | t | Acceptance: stated / implemented | Accepted infidelity: stated / implemented |
|---|---|---|---|
| 2e-4 | 0.5 | 0.985 / 0.946 | 4.08e-2 / 2.20e-2 |
| 1e-3 | 0.5 | 0.823 / 0.734 | 0.153 / 0.112 |
| 1e-3 | 0.97 | 0.340 / 0.333 | 1.061e-2 / 1.024e-2 |
| 2e-3 | 0.97 | 0.060 / 0.057 | 1.77e-2 / 1.70e-2 |

**Size of the decay rate** (c00, p = 1e-3, all 9 depths, 3e4 shots each;
`v2_soft_rate.log`):

| t | r per depth step: stated rule | r per depth step: implemented rule |
|---|---|---|
| 0.5 | 7.1e-3 | 4.8e-3 |
| 0.97 | 2.45e-4 | 2.28e-4 |

The implemented values reproduce the published ones: REPORT §4.1 gives
5.2e-3 for t = 0.5 and 2.4e-4 for t = 0.97 (30-sequence fits).

- For t = 0.5, the published values (r = 5.2e-3, P(correct) 0.887 and
  acceptance 0.724 at m = 22) describe "reject if conf ≤ 0.553". The stated
  rule would give roughly 1.5 times the r, P(correct) ≈ 0.85 and acceptance
  ≈ 0.82.
- For the headline "a 0.97 threshold gives 4.8% and 2.4e-4", the stated rule
  gives an r about 7% larger.

**Unaffected.** The `soft:0.9` and `soft:0.99` numbers in `key_summary.md`,
the executive summary and the figures, and the Pareto frontier (labelled by
bin edge). The qualitative conclusions do not change.

**Fix, any one of:**

1. relabel each `soft:t` as `conf > t_eff`;
2. choose `SOFT_THRESHOLDS` on the grid, `t = 1 - 10^(-0.05 k)`;
3. add the edges `log10(1 - t)` to `CONF_EDGES` and re-run the sampled stage.

### F2 (minor): "Realized counts ... agree" is true on average, but one cell is over-dispersed

**Where.** `REPORT.md` line 230; the stored `d*__mld_all_counts`.

**What.** For p = 0.01 and m = 2, the production realized MLD-failure counts
of the 30 sequences give sum z² = 64.6 on 30 degrees of freedom (p = 2.5e-4).
The RB estimates from the same shots give 28.7. The other three
(p, m) cells are normal.

**Investigation.**

- I reproduced the production shots bit for bit (seed
  `[20261028, cliff, round(p * 1e9)]`, depth index 0 drawn first). On those
  shots the realized failures deviate from their conditional expectation:
  sum z_cond² = 60.9 over 30 sequences.
- Fresh seeds on the same 30 sequences give sum z² = 136 on 150 degrees of
  freedom (p = 0.79).
- 3e7 shots on c16 give std 0.98.

So this is a statistical fluctuation of those particular draws, not a sampler
or decoder defect. No reported number uses the realized counts.

**Suggestion.** Quote the RB-vs-exact z-statistics rather than "agree".

**Scripts.** `v2_allseq_summary.py`, `v2_realized_probe.py`,
`v2_realized_probe_all.py`, `v2_zcond_allseq.py`.

### F3 (minor): wrong bin-interval comment

`src/lrb/correction/aggregate.py:57` says bin b collects
`(edges[b-1], edges[b]]`. `np.searchsorted(..., side="right")` (`study.py:174`)
gives `[edges[b-1], edges[b])`. This has no numerical effect.

## Files

| File | Role |
|---|---|
| `V2/v2_core.py` | independent toolkit: reader, two dense convolutions, span filter, Z_3 algebra |
| `V2/v2_census.py`, `.log` | location census per depth |
| `V2/v2_selftest.py`, `.log` | my methods agree with each other |
| `V2/v2_check_dense.py` | (a), (b), (c) at m = 0 and 2 vs `lrb.correction` and the npz |
| `V2/v2_check_depth4.py` | depth 4 |
| `V2/v2_check_deep.py` | depths 4, 14 and 22 |
| `V2/v2_check_allseq.py`, `V2/v2_allseq_summary.py` | 30 sequences vs npz, curves and 1M-shot CSVs |
| `V2/v2_frontier_check.py` | the frontier claim |
| `V2/v2_soft_*.py` | F1 |
| `V2/v2_realized_probe*.py`, `V2/v2_zcond_*.py`, `V2/v2_sampler_*.py` | F2 and the sampler checks |

The `v2_sampler_gof.py` G-test is miscalibrated at expected counts near 5. My
own sampler also "fails" it, so I do not use it as evidence.
