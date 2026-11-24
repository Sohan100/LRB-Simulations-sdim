# V1 review: exactness of the chain model and the forward recursion

Reviewer id: V1-exactness. Lens: mathematical exactness of `chain_dem.py` and
`forward.py` (plus the exact evaluators in `decoders.py` that are built on them).
All scripts and their raw outputs are in `notes/review/V1/` (`*.py` and `*.out`). I
did not modify anything under `src/`, `scripts/` or `tests/`; I read the main
checkout and the production cache only.

## Verdict

- When `build_chain_structure` accepts a DEM and the activation probabilities are
  right, the chain model and forward recursion are exact. I checked this by
  deriving the recursion myself, comparing every index table against loop code
  written from that derivation, and comparing `posterior * exp(log_evidence)`
  against three routes that use no slots, transfer maps or kernels (Section 3).
  The worst deviation was 2.3e-15 absolute across 1.6e6 to 1.4e7 joint entries
  per circuit, and 4.4e-15 relative on exhaustive enumerations at depths 6, 14 and
  22. **[VN]**
- A location cannot skip a round in these circuits. If one did, the code would
  raise rather than silently mis-model it (Section 4). **[VA, VN]**
- I have **three findings**. One changes reported numbers (BFD-X), one is a real
  bug that does not reach production (activation from `prob_scale`), and one is
  a test-coverage gap.

## 1. Findings

### F1 (critical, BFD-X rows only): `blip_forgiving_exact(final_rule="xtriv")` does not evaluate the rule it documents

- **Where:** `src/lrb/correction/decoders.py:172`,
  `return clean @ final + flagged @ final`.
- **What is wrong:** The docstring (`decoders.py:127-160`), the per-shot function
  `blip_forgiving_accept` (`decoders.py:178-208`, lines 200-204) and REPORT.md
  Section 2.2 ("BFD-X ... as above, but in the final round only the X-check digits
  must be trivial") all keep the rule that no two consecutive observed rounds may
  both be non-trivial. With `final_rule="xtriv"`, `final` allows a final round that
  is non-trivial in its Z-check digits only. The `flagged @ final` term therefore
  accepts shots where the second-to-last round is non-trivial and the final round
  is non-trivial too, which is two consecutive non-trivial rounds. The documented
  rule is `clean @ final_xtriv + flagged @ final_zero`. For `final_rule="zero"` the
  two expressions coincide, so plain BFD is unaffected.
- **Evidence:** `V1/check_bfd_xtriv_rule.py` → `check_bfd_xtriv_rule.out`, on
  production structures. In every row the stored production `bfd_xtriv` equals the
  code value. 4e5 sampled shots scored with `blip_forgiving_accept(final_rule="xtriv")`
  agree with the documented rule (|z| ≤ 1.9) and disagree with the code (z down to
  -68):

  | cliff, m, p | code acceptance | documented-rule acceptance | sampled per-shot | code P(L=0 given acc) | documented P(L=0 given acc) |
  |---|---|---|---|---|---|
  | 0, 2, 1e-3 | 0.852891 | 0.834426 | 0.834937 ± 0.000587 | 0.994388 | 0.994711 |
  | 0, 2, 5e-3 | 0.450232 | 0.396943 | 0.397515 ± 0.000774 | 0.958332 | 0.966706 |
  | 0, 22, 1e-3 | 0.202988 | 0.199077 | 0.198683 ± 0.000631 | 0.992508 | 0.993188 |
  | 17, 22, 5e-3 | 0.000278 | 0.000243 | 0.000245 ± 0.000025 | 0.912138 | 0.930608 |

  The extra shots the code accepts are not harmless. At cliff 0, m = 2, p = 1e-3
  they make up 0.018465 of all shots and carry a logical-error probability of
  `(0.852891*0.005612 - 0.834426*0.005289)/0.018465 = 2.0%`, about four times the
  accepted average of 0.53%. **[VN]** (My guess at the cause, a remaining `Z_L`
  factor that a final Z-only syndrome cannot rule out, is a **conjecture**.)
- **Impact:** every reported `bfd:xtriv` number: the "BFD, final round X-part
  only" rows of `results/decoder_tables.md` and the `bfd:xtriv` rows of
  `fits_*.csv` and `curves_*.csv`. Effect on the fits over all 30 x 9 production
  structures (`V1/check_bfd_xtriv_fit.py`): FIT_TABLE_PLACEHOLDER
  Plain BFD, LRB-D, LRB-D-X and every MLD number are unaffected.
- **Fix:** in the final step use
  `clean @ cache.get(slot, final_rule) + flagged @ zero`. Alternatively, if
  accepting those shots is intended, change the docstring, REPORT.md and
  `blip_forgiving_accept`. Then regenerate the `bfd:xtriv` rows either way.

### F2 (major, does not reach production): activation comes from `prob_scale` metadata with a silent default of 1.0, and is never checked against the DEM's own probability

- **Where:** `src/lrb/correction/chain_dem.py:160`
  (`prob_scale = float(location.params.get("prob_scale", 1.0))`), used by
  `Mechanism.activation_probability` (`chain_dem.py:80-85`), `ChainStructure.at`
  (`chain_dem.py:543-560`), `const0_joint_from_mechanisms` (`chain_dem.py:674`) and
  therefore `sampler.sample_shots`. `location.probability`, which the repository's
  own DEM sampler uses, is ignored.
- **Failing input:** the sdim 1.4.0 SI1000 run in the main checkout,
  `Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140/experiments/LRB/<c>/4/<d>.chp`.
  Its `.chp` files carry materialized probabilities (`prob=0.0001/0.001/0.002/0.005`)
  but no `prob_scale`. `build_chain_structure` accepts them without any warning and
  every activation becomes `p`.
  - `V1/check_families.out`: chain activation / DEM probability lies in [0.2, 10].
    At depth 4 the exact chain gives P(L, all rounds trivial) = [0.5630, 0.00231,
    0.00231], but the repository DEM sampler gives [0.6985, 0.00150, 0.00142]
    (2e5 shots, z = +122, -7.6, -8.3). Depth 14: 0.1675 against 0.3325 (z = +198).
  - `V1/check_fft_other_families.out`: on the exact full joint (depth 2), the
    largest absolute difference between the chain and the DEM truth is 7.7e-2.
    P(all trivial) is 0.7418 from the chain against 0.8158 from the DEM.
- **Production is unaffected:** all 540 production `.chp` files (`LRB` and
  `LRB_const0`, 30 sequences x 9 depths, p_index 4) carry `prob_scale` on every
  N1/N2 line. The `(prob, prob_scale)` pairs are exactly
  `(1e-4, 0.1), (1e-3, 1), (2e-3, 2), (5e-3, 5)`, and on 9 freshly compiled
  circuits `location.probability == 1e-3 * prob_scale` to 1e-15
  (`check_structure.py`). The 07-15 SI1000 run carries `prob_scale` and is exact.
  The sdim140 SD6 run and the in-memory depolarizing family carry no `prob_scale`.
  They come out exact (Section 5) only because every one of their locations has
  probability exactly p.
- **Fix:** store `location.probability` in `Mechanism` and require
  `location.probability == p_compile * prob_scale` for one common `p_compile` (raise
  otherwise). Or take `scale = location.probability / p_compile` with `p_compile`
  passed in explicitly. Either way, raise when `prob_scale` is missing and the
  probabilities are not all equal.

### F3 (minor): the shipped exactness tests never exercise syndrome transport

- **Where:** `tests/test_correction_chain.py:83-96` (depth 1, two rounds) and
  `tests/test_correction_variants.py:69-109` (depth 1). With two rounds the only
  transfer map is `T[1] = I`, applied to `h_0 = 0`, so neither test exercises a
  learned `T`, `_carry_index` with a non-identity permutation, or a multi-step
  recursion. REPORT.md Section 2.4 cites these tests as the brute-force validation
  of the chain model. The depth-4 sampling test checks only the LRB-D and BFD
  acceptance joints, at 4 sigma.
- **Status:** I closed the gap independently (Section 3: full joint at depth 2 and
  terminal depth 2, exhaustive enumeration at depths 6 to 22, and second-order
  enumeration at all depths). No error was found.
- **Suggestion:** add the depth-2 FFT comparison (`V1/check_fft_full_joint.py`,
  about 4 s per circuit) as a test.

## 2. Derivation used for the checks **[VA]**

For each slot `r` let `(A_r, B_r, L_r)` be the summed (raw round-r shift,
round-(r+1) shift or terminal shift, logical shift) of the locations assigned to
slot `r`. Locations are independent and each one belongs to exactly one slot, so
the slots are independent with joint law `K_r[l, b, a]`. A slot-`r'` location
shifts round `r > r'+1` by `T_r ... T_{r'+2} b`. Every vector's effect is checked
against this in `assign_slots`, and the check is complete because it runs over
every vector and every later round, and over the terminal detectors when they
exist. Hence

```text
s_r = h_r + A_r,  h_0 = 0,  h_{r+1} = T_{r+1} h_r + B_r,  t = Pi h_{R-1} + B_{R-1},  L = sum_r L_r
alpha_{r+1}(h', L) = sum_{h, L'} alpha_r(h, L') K_r[L - L', h' - T_{r+1} h, s_r - h]
P(s, t, L)        = sum_{h, L'} alpha_{R-1}(h, L') K_{R-1}[L - L', t - Pi h, s_{R-1} - h]
```

Code conventions, all confirmed: state `3h + L`; `_LSHIFT[L', L] = L - L'`;
`_carry_index(perm)[h, h'] = idx(h' - T h)` with `perm[h] = idx(T h)`;
`a_index[h] = idx(s - h)`; `_allowed_matrix(mask, sub, o)[a, j] = mask[idx(a + o_j)]`;
terminal `t_index[h] = idx(t - Pi h)`. `build_kernels` computes
`new[l, b, a] = (1 - q + w_000) old[l, b, a] + sum_k w_k old[l - l_k, b - b_k, a - a_k]`,
which is the correct convolution direction. The last slot has `b` width 1 (no
terminal) or 9 (terminal), with digit 0 being `lrb_term_xstab_0`. A location that
touches no round goes to the last slot with `a = 0` and `b` = its terminal shift.
That is exact even when it sits early in the circuit, because its only observable
effects are the logical shift and the terminal shift. The forward filter's
posterior is `P(L | s, t)`, so its argmax is the exact MAP of the read-out logical
class.

## 3. Checks run and their outputs

| # | Script (`notes/review/V1/`) | What it checks (independent route) | Result |
|---|---|---|---|
| 1 | `check_indexing.py` | `transition_matrix` (obs = all / value / 0 / random set, random invertible T), `final_matrix` (B = 1 and 9; every combination of all / value / set for s and t) and `build_kernels` (asymmetric shifts, zero atom, b widths 4/0/2), each against loop code built from explicit digit arithmetic | worst deviation **8.3e-17** |
| 2 | `check_fft_full_joint.py` | the full joint over all 3^(D+1) (detector, logical) values by characteristic functions, `IFFT(prod_i [(1-q_i) + q_i/n_i sum_v omega^(-chi.e_iv)])`, against chain `P(s, L)` for **every** history (shared and plain forward) | baseline depth 2 (3 rounds, so learned `T[2]`), cliffs 0/9/18 at p = 5e-3 and 3e-2: max abs **2.3e-15**, rel ≤ 1.1e-9 on entries > 1e-9 (FFT round-off). Terminal variants: depth 1, depth 1 parallel+terminal, and depth 2 parallel+terminal (1.4e7 states): max abs ≤ **8.1e-16**. Marginalizing the terminal digits agrees to ≤ 7.1e-16. |
| 3 | `check_sparse_enumeration.py` | 5 to 7 active locations spread across slots, including slot 0 (seen in every later round), one N2 mid-circuit and the last slot, with q ∈ [0.15, 0.6]; every configuration enumerated (≤ 1.6e6), against chain on every reachable history | baseline depths 6, 14, 22 (cliffs 3, 7, 11, 29; 2 draws each), variants `chk` depth 6, `par_term_chk` depth 6 and `term` depth 8: worst abs **2.8e-17**, rel **4.4e-15**, total chain mass on the reachable set = 1 to 1e-15 |
| 4 | `check_two_fault.py` | all locations active at p = 1e-6, 1e-7, 1e-8. Exact ≤ 2-location enumeration `P2(s, L)`, which needs every omitted term to be ≥ 0 and third order. Targets: the trivial history, 300 single-fault histories and 300 two-fault histories (one early and one late fault) | depths 6, 14, 22 (cliffs 3, 7, 11). Every (chain - P2)/chain is ≥ -4.1e-14 (rounding) and otherwise positive. At m = 22 the maximum falls by 10x per decade of p for single-fault histories (1.80e-3, 1.80e-4, 1.80e-5) and for two-fault histories (1.87e-1, 2.25e-2, 2.30e-3, i.e. 8.3x then 9.8x as the O(p^2) part of the remainder dies out; the minimum goes 3.98e-4, 3.98e-5, 3.98e-6). For the trivial history it falls by 100x (1.16e-7, 1.17e-9, 1.16e-11). That is what a third-order remainder must do. Entries with P2 = 0 scale as p^3 (3.5e-16, 3.5e-19, 3.5e-22). A wrong first- or second-order coefficient would leave a floor that does not move with p, and there is none (`check_two_fault.out`). |
| 5 | `check_all_production_t0.py` | all 270 production pickles: chain and `lrbd_exact` `P(all rounds trivial, L)` against `P2(0, L)` at p = 1e-7 and 1e-8; the stored production `const0_joint` against my own convolution | all 270 structures. L = 0 agrees to rounding (relative difference in [-4.3e-14, +1.8e-14]). L != 0 (the undetected logical error, first order) has a positive remainder of 1.2e-11 to 1.3e-9 at p = 1e-7 and 1.2e-13 to 1.3e-11 at p = 1e-8. Every remainder that is not pure rounding drops by at least 100.3x per decade, so the first- and second-order coefficients are exact. `lrbd_exact` equals the forward evaluation to 1e-12. The stored production `const0_joint` (p = 1e-3) matches my own dense convolution to 5.6e-14 in all 270 circuits. |
| 6 | `check_rules_enumeration.py` | on top of route 3: masked `forward_posteriors(_shared)` (terminal-only and unif-2 observation) against the marginalized enumeration; `lrbd_exact` (all / unif2 / const1), `lrbd_xtriv` and BFD (zero/xtriv, all/unif2) against the enumeration filtered by `blip_forgiving_accept` | 5 circuits (baseline m = 6, 14, 22; `term` m = 4; `par_term_chk` m = 4), each against exhaustive enumeration: masked posteriors ≤ 1.1e-15; `lrbd_exact` all/unif2/const1 ≤ 1.1e-15; `lrbd_xtriv` ≤ 7e-18; BFD `zero` ≤ 3.9e-16. BFD `xtriv` **does not match** in 7 of 10 cases (up to 6.2e-2 in acceptance); the other 3 match because their active set cannot produce the offending history (F1) |
| 7 | `check_production_reproduce.py` | every exact key of `study.evaluate_chain` from a fresh compile against the stored production npz (cliffs 0, 13, 29; p = 1e-3, 5e-3, 5e-2; 9 depths) | worst absolute difference between recomputed and stored values = **3.5e-18** over 729 arrays. The stored exact production numbers are what the current code computes. |
| 8 | `check_structure.py` | 9 baseline circuits (depths 2 to 22): my own re-extraction of the effect tables against `extract_mechanisms`; `prob == 1e-3 * prob_scale`; per-vector gaps; first-round spread; ranks; production pickle against a fresh compile | effect tables identical; 0 gap vectors; first-round spread ≤ 1 across vectors **and** across unit components; every transfer-fit rank = 4; production pickles identical |
| 9 | `check_synthetic_breaks.py` | hand-made DEMs fed to `build_chain_structure` | see Section 4 |
| 10 | `check_families.py`, `check_fft_other_families.py` | other circuit families | see Section 5 |
| 11 | `check_bfd_xtriv_rule.py`, `check_bfd_xtriv_fit.py` | F1 | see F1 |

## 4. Can a location skip a round, and what does the code do then?

- **Physics [VA]:** a location acts at one time. A data component either passes
  through a check round unchanged (data-control CNOTs copy X onto the ancilla, and
  ancilla-control CNOTs copy Z back onto the ancilla, without changing the data
  operator) or is conjugated by noiseless logical Cliffords between rounds. Those
  Cliffords preserve the stabilizer group and its normalizer, so the syndrome
  evolves by an invertible linear `T`. It is zero at round r+1 exactly when it is
  zero at every later round. An ancilla component reaches only the current
  round's outcome, plus the next round's outcome for the reset-X shift, because
  every ancilla is reset before it is reused (after the round here, before the
  round in SD6). Its back-action on the data happens only while it is coupled, and
  that back-action is a data component. So the rounds a vector touches are `{r}`, a
  contiguous tail `[r', R-1]` with `r' ∈ {r, r+1}`, or their union. Different
  components of one location first appear at most one round apart.
- **Numerically [VN]:** 9 baseline circuits, every location, every vector and every
  unit component: 0 gaps, first-round spread ≤ 1 (`check_structure.out`).
- **The code raises [VN]** (`check_synthetic_breaks.out`):

  | Case | Outcome |
  |---|---|
  | A: touches rounds 0 and 2, not 1 | `learn_transfer_maps` raises "no linear syndrome map from round 1 to 2" |
  | B: unit components first appear at rounds 0 and 2 | `assign_slots` raises "breaks the chain assumption at round 2" |
  | C: later effect is not the T-image | raises at the transfer fit |
  | E: earlier-slot terminal effect is not `Pi` of the last round | `learn_terminal_map` raises |
  | B': spread 1 | accepted; exact against enumeration (1.0e-17) |
  | D: touches no round but has terminal and logical effect | accepted in the last slot; exact (2.8e-17) |
  | F: single round with terminal detectors | accepted; exact (5.6e-17) |
  | G: 4 rounds with non-identity T | learned `T[2] = T[3] = T`; exact (1.7e-17) |

  A vector-level gap would need a unit component with a gap, because
  `T(x e0 + z e1) = x T e0 + z T e1`. The transfer fit already rejects unit
  components with a gap, so the check is complete.

## 5. Other circuit families

`check_families.out` and `check_fft_other_families.out`:

| Family (main checkout) | Outcome |
|---|---|
| Depolarizing `Run-2026-02-18-19-45-04-folded_qutrit` | Has no `experiments/` folder, so I rebuilt the family in memory with `LRBCircuitGenerator(noise_model="depolarizing")`. Builds at depths 0, 4, 14 and agrees with the DEM sampler. The exact full joint at depth 2 matches to 1.6e-15. |
| Split ancilla `Run-2026-06-05-20-34-54-...split-unif-si1000` | 6 detectors per round; rejected cleanly with `ValueError: chain model expects four detectors per round` |
| Split ancilla `...-si1000-detectors` (06-05 16-29), sdim140 split SI1000 and split SD6 | Rejected cleanly with the same error |
| `Run-2026-05-29-...-si1000-all` | No DETECTOR annotations; `build_compact_detector_error_model` raises before the chain is reached |
| 07-15 single-ancilla SI1000 | Exact (full joint at depth 2 matches to 5.8e-15) |
| sdim140 SD6 | Exact (5.3e-15), but only by accident (F2) |
| sdim140 SI1000 | Silently wrong (F2) |
