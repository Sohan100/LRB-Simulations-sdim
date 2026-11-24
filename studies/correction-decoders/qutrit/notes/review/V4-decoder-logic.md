# V4 review: decision-rule logic and protocol-variant circuits

Reviewer: V4-decoder-logic. Lens: `decoders.py` decision rules (BFD, window
decoder), `forward.constrained_joint` with allowed sets, and
`variants.build_variant_circuit`. Everything here was run on the login node
(at most 4 threads), read-only on the production run folder and on
`.cache_correction`. Scripts and raw outputs are in `review/V4/`.

## Summary

| ID | Severity | Finding |
|---|---|---|
| F1 | **critical** (only the reported BFD-X rows change; no headline conclusion uses them) | The reported "BFD, final round X-part only" (`bfd:xtriv`, `blip_forgiving_exact(final_rule="xtriv")`) is not the documented rule, disagrees with `blip_forgiving_accept(final_rule="xtriv")`, and accepts first-order logical errors. Its rationale ("an X error left at the end cannot change the X readout") is false because the residual can carry a Z-logical, which the X checks cannot see. Confirmed by an independent SDIM tableau fault injection. Under the documented rule the reported r at p = 1e-3 changes from 1.31e-4 to 8.8e-5, with non-overlapping CIs. |
| F2 | minor | "BFD never accepts a first-order logical error" (REPORT lines 51-52 and 401; `decoders.py` 136-141) needs a qualifier. BFD accepts the m-independent 4.95 p undetectable floor, as LRB-D does. It adds nothing beyond that floor. The forgiven blips are not only measurement or reset faults. |
| F3 | minor | `window_decode` drops the terminal X-syndrome from its backward (lookahead) message (`decoders.py` 247-249). On terminal-readout circuits, "full lookahead" therefore does not use all the information, and survival can drop when lookahead is added. |
| F4 | minor | With the final round unobserved, `blip_forgiving_exact` raises `AssertionError("unreachable")`, while `blip_forgiving_accept` silently treats the last observed round as final. |
| F5 | minor | No package test exercises the learned transfer maps `T[r>=2]`, the BFD adjacency logic or allowed-set observations exactly. The brute-force tests use 2-round circuits, where `T[1]` is never used. The rest is covered only by 4-sigma sampling tests, and `xtriv` is not tested at all. |

All other checks in my lens passed:
- BFD with `final_rule="zero"`, exact vs sampled, every mask, with and without the terminal.
- `constrained_joint` with allowed sets.
- `window_decode` indexing, backward message, commit step, final slot and lookahead edge cases.
- All three variant flags, including noiseless determinism of every detector.

Details follow.

## Method: an independent exact reference

`v4_common.fourier_joint` computes the exact joint distribution of every
detector, the terminal X-stabilizers and the logical shift. It uses only the
compact DEM's unit-component responses, with
`chi(k) = prod_loc [1 - q + q/n_v sum_v omega^(v . U k)]` and an n-dimensional
FFT over Z_3^n, and none of chain_dem's slot, transfer or terminal-map
construction. It is feasible up to about 15 trits (depth-2 circuits, or depth-1
circuits with terminal readout). Against a direct full-space convolution it
agrees to 1.1e-14 (`check_fourier_selftest.out`, production m = 0 and
generated m = 1 circuits, p = 1e-3 and 2e-2).

Exact checks on 6 circuits at p = 1e-3, 1e-2 and 3e-2
(`check_exact_small.py` / `.out`):
- production cliffs 0, 5 and 12 at m = 2;
- the parallel variant at m = 2;
- the terminal and parallel+terminal variants at m = 1.

| Quantity compared with brute force | max abs error |
|---|---|
| chain forward `P(L, history)`, every history (T[2] included) | 4.8e-15 |
| `lrbd_exact` (all rounds) | 5.1e-15 |
| `constrained_joint`, final round `X_PART_TRIVIAL` (= `lrbd_xtriv`) | 5.7e-15 |
| `constrained_joint`, random int / None / mask per round and terminal | 3.5e-14 |
| `blip_forgiving_exact` vs `blip_forgiving_accept`, `zero`, every distinct unif/const mask, terminal on and off | 8.1e-15 |
| `blip_forgiving_exact`, `xtriv` vs my lenient re-implementation | 2.0e-14 |
| `blip_forgiving_accept`, `xtriv` vs my strict re-implementation | 0 |
| `blip_forgiving_exact` vs `blip_forgiving_accept`, `xtriv` | **7.4e-2** (F1) |
| exact survival: any window decoder minus MLD | never positive |

## F1 (critical, scoped to the BFD-X rows): BFD-X accepts first-order logical errors

**Where.**
- `decoders.py:170-172`: `final = cache.get(slot, final_rule)` and
  `return clean @ final + flagged @ final`. With `final_rule="xtriv"` a
  non-trivial round m-1 ("flagged") followed by a final round with trivial X
  digits but non-trivial Z digits is accepted.
- `decoders.py:199-205` (`blip_forgiving_accept`) computes `consecutive` over
  the full syndromes, including the final round, so it rejects that history.
  This is also the rule documented at `decoders.py:136-137` and REPORT line 217
  ("as above, but in the final round only the X-check digits must be
  trivial").
- `study.py:137` stores the exact (lenient) version as `bfd_xtriv`.
  `results/decoder_tables.md` reports it as "BFD, final round X-part only".

**Two different rules (verified numerically, exact).** On production cliff 0,
m = 2, p = 1e-3:
- exact: acceptance 0.852891, error given acceptance 5.612e-3;
- `accept()`: acceptance 0.834426, error given acceptance 5.289e-3.

The Fourier brute force shows exact = lenient rule and `accept()` = strict rule
to 1e-14 (table above). Sampled on production cliff 0, m = 22, p = 1e-3, with
2e5 shots: exact acceptance 0.20299, `accept()` 0.19872, z = -4.8
(`check_edge_cases.out`).

**The lenient rule accepts first-order logical errors (verified numerically and
by independent simulation).** Single-fault enumeration
(`check_first_order_bfd.out`) gives the extra first-order accepted logical
error, per unit p, over LRB-D on all rounds:
- `bfd:xtriv` as reported: +0.075 p in 10 of the 20 production circuits
  enumerated (cliffs 0, 1, 2, 7 and 19 at m = 2, 6, 14 and 22). For example
  c1 m2, c1 m6, c2 m22 and c19 m22.
- strict BFD-X: 0 in all circuits.
- BFD (`zero`): 0 in all circuits, for every mask.

`check_xtriv_offender.out` gives the package's own exact evaluator at
p = 1e-5 (accepted logical-error probability divided by p):

| circuit | LRB-D | LRB-D-X | BFD | BFD-X as reported |
|---|---|---|---|---|
| c1 m2 | 4.937 | 4.940 | 4.942 | 5.023 |
| c2 m2 | 4.937 | 4.940 | 4.941 | 5.024 |

The offending fault is `N2` after `CNOT 2->6` in round m-1, with Weyl vector
`(x2, z2, x6, z6) = (1, 0, 0, 1)`: X on data qutrit 2 and Z on ancilla 6, which
propagates to Z on data 4. `check_offender_tableau.out` removes all noise from
production circuit c1 m2, inserts exactly this error and runs SDIM's tableau
simulator. Results:
- syndromes: round 1 `(0,0,0,1)`, round 2 `(0,1,0,0)`, logical shift 1,
  deterministic;
- identical to the DEM prediction;
- reported BFD-X: ACCEPT; `accept(xtriv)`: reject; BFD: reject.

The final round's X digits are trivial, so the Z part of the residual
commutes with both X stabilizers. It therefore lies in
`<S_Z1, S_Z2, Z_L>`, with `Z_L = Z0 Z2 ~ Z3 Z4`. The readout is shifted, so
the residual contains `Z_L`; its X part is what `w6` sees. A trivial X part of
the final round therefore does not mean that the residual cannot flip the X
readout. The lenient rule drops the only evidence, round m-1, that the error
persisted.

**Wider scope (unreported).** The same blind spot makes every `xtriv` rule
unsafe once the round before the final one is unobserved: LRB-D-X and BFD-X,
strict or lenient, with sparse unif/const masks accept up to +1.05 p at
first order (m = 22). The reported `lrbd:xtriv` uses all rounds and is clean
(0 extra in all 30 circuits).

**Impact on reported numbers.** `check_bfd_xtriv_impact.out` uses all 30
production sequences at p = 1e-3, with the package's slot matrices and
`aggregate.fit_curve` (signed metric). The lenient recomputation reproduces
the stored `d*__bfd_xtriv` joints exactly (max difference 0.0).

| BFD-X semantics | r per depth step [95% CI] | acceptance per step | acceptance m = 22 | 1 - signed, m = 22 |
|---|---|---|---|---|
| lenient (reported, `decoder_tables.md` row "BFD, final round X-part only") | 1.309e-4 [1.28e-4, 1.34e-4] | 0.9315 | 0.2025 | 1.133e-2 |
| strict (documented; `accept()`) | **8.76e-5** [8.62e-5, 8.88e-5] | 0.9312 | 0.1994 | 1.026e-2 |

Under the documented rule, the reported BFD-X decay rate falls by 33%, and
the confidence intervals do not overlap. The lenient rule is strictly worse:
it gains 0.0003 acceptance per step and pays 50% more r. So it changes
reported numbers: the BFD-X rows of `decoder_tables.md` (all p) and the
REPORT method-table line 217. No headline conclusion in REPORT.md uses
BFD-X.

**Fix.** Make the exact recursion end with
`clean @ M_xtriv + flagged @ M_zero` (strict), matching `accept()`. Then
re-run `bfd_xtriv`, or drop the row and the "X error left at the end"
rationale.

## F2 (minor): first-order statements about BFD

- REPORT.md lines 51-52 (executive summary, Q1 BFD bullet) and line 401
  (section 4.4 ranking table) say BFD "never accepts a first-order logical
  error". The single-fault enumeration shows BFD's first-order accepted
  logical error equals LRB-D's in every one of the 30 circuits and for every
  mask. That is the undetectable floor: 4.95 p for the production protocol,
  3.45 p with parallel readout, and 0 with full terminal readout. It is not
  zero. `results/first_order_check.json` already shows this:
  `exact_bfd_logical` is 4.9e-4 at p = 1e-4. Suggested wording: "accepts no
  first-order logical error beyond the undetectable floor that LRB-D also
  accepts".
- `decoders.py:140-141` says BFD keeps "rejecting every single data fault".
  Data faults after the last check interaction are undetected and are
  accepted; these are the floor.
- REPORT line 51 says the forgiven syndrome "is always a measurement or reset
  fault". `check_first_order_bfd.out` breaks down the forgiven (isolated-blip)
  single faults for c0 m6, 238 p in total:
  - measurement flips: 120 p;
  - reset flips: 40 p;
  - idle depolarizing on waiting ancillas: 54 p;
  - measured-ancilla depolarizing: 18 p;
  - ancilla components of CNOT faults: 4.2 p;
  - ancilla H-gate noise: 1.8 p.

  None of them shifts the logical, so the conclusion holds, but "ancilla-side
  faults" is the accurate description.

## F3 (minor): window_decode ignores the terminal X-syndrome in its lookahead

`reduced_matrix` sums the last kernel over its terminal axis
(`decoders.py:247-249`). The terminal syndrome enters only the final-slot
decision (`decoders.py:264-266`). On terminal circuits, "lookahead covering
the whole record" is therefore not the full-information hard-decision
smoother that the docstring describes.

On the terminal variant at m = 1 and p = 3e-2, exact survival is non-monotone
in lookahead (`check_exact_small.out`): w0 = 0.41149 > w1 = w2 = 0.41057 (MLD
0.41641). At p <= 1e-2 lookahead still helps.

Including the terminal in the backward message changes decisions
(`check_window.out` part C; `check_window_terminal_exact.out`). I used a
reference decoder with the terminal in the lookahead, otherwise identical.

Exact survival, m = 1, every history:

| circuit, p | MLD | w0 | w1 (package) | w1 (terminal in lookahead) |
|---|---|---|---|---|
| term, 1e-3 | 0.971776 | 0.969321 | 0.971022 | 0.971715 |
| term, 1e-2 | 0.713674 | 0.694238 | 0.702836 | 0.710080 |
| term, 3e-2 | 0.416413 | 0.411486 | 0.410571 | 0.412871 |
| par_term, 1e-3 | 0.985680 | 0.983389 | 0.985412 | 0.985660 |

At m = 1 and p = 1e-3, ignoring the terminal accounts for about 90% of the
window decoder's gap to MLD. With the terminal included, lookahead is
monotone again.

Sampled, 1500 shots:
- term m4: no decision changes at p = 1e-3. At p = 5e-3 there are 30
  changes, and Rao-Blackwellized survival with full lookahead goes from
  0.68468 to 0.68763 (MLD 0.69668).
- par_term m6, p = 5e-3: 0.76531 to 0.76705.

The effect sits in the last few slots, so it is mostly m-independent. It
affects only the `win:2` (and `win:1`) rows of the `term`, `par_term` and
`par_term_chk` variants. `win:0` and all baseline rows are unaffected.

## F4 (minor): unobserved final round

`blip_forgiving_exact(chain, [T, T, F])` raises
`AssertionError("unreachable")`. `blip_forgiving_accept` with the same mask
returns acceptance 0.506, because it uses round 1 as the final round
(`check_edge_cases.out`). The production masks always include the final round,
so no number changes. One explicit `ValueError` in both functions would make
them consistent.

## F5 (minor): test coverage

- `test_chain_joint_matches_brute_force` and
  `test_terminal_chain_exact_on_sparse_noise` use 2-round circuits. The
  learned `T[1]` is never exercised because `h_0 = 0`.
- Learned maps `T[r>=2]`, BFD adjacency and allowed-set observations
  (`_allowed_matrix`, terminal sets) are tested only by 4-sigma +1e-4
  sampling.
- `final_rule="xtriv"` has no test, which is how F1 went unnoticed.

The Fourier reference in `review/V4/v4_common.py` runs a depth-2 exact test in
under a minute (about 15-20 s for the brute force). It covers all of these and would have caught F1.

## Checks that passed

### (1) BFD exact vs sampled

- **Proof sketch, `final_rule="zero"` (verified analytically).** The
  recursion keeps two sub-normalized state vectors: "previous observed round
  trivial" and "previous observed round non-trivial".
  - An observed trivial round maps both to clean.
  - An observed non-trivial round maps only clean to flagged; flagged followed
    by non-trivial is dropped.
  - An unobserved round applies the marginal matrix to both states, so the
    flag survives the gap. This is exactly `syndromes[:, observed_rounds]`
    followed by adjacent pairs in `accept()`.
  - The final step requires a trivial final round from either state. A
    trivial round cannot form a non-trivial pair, so the two rules coincide.
- **Exact numerical check.** Agreement to 8e-15 on 6 circuits x 3 p x every
  distinct mask, with `_SlotMatrices(use_terminal=True/False)` against
  `accept(terminal=...)` and `accept(terminal=None)`.
- **Production consistency.** `check_production_npz.out` compares the sampled
  vs exact BFD acceptance in all 1890 baseline and 6480 variant entries.
  Exact binomial p-values are consistent with uniform: fraction below 0.01 is
  0.0085 and 0.0094. All are above the Bonferroni threshold. The two |z| > 11
  entries are single accepted shots against exact acceptance below 1.5e-7,
  where the normal approximation fails (binomial p = 0.015).
- **First-order claim.** The single-fault enumeration covers 20 production
  circuits and 10 variant circuits, for every distinct mask and with the
  terminal when present. No BFD (`zero`) rule and no LRB-D mask accepts any
  detected single fault that shifts the logical. Extra first-order logical
  error over LRB-D(all) is 0.000 p everywhere. First-order BFD rejection equals
  const-1 rejection, as `first_order_check.json` shows.

### (2) window_decode

- **Reference implementation.** A slow per-shot reference (`check_window.py`
  part A) uses explicit digit arithmetic, an explicit backward recursion
  `beta_f(h) = sum_h' M_f[h,h'] beta_{f+1}(h')`, an explicit argmax over
  `(l, h')` and the terminal or plain final slot. It covers 4 circuits
  (production m = 6 and 10; term and par_term_chk at m = 4), p = 5e-3 and
  2e-2, and lookaheads 0, 1, 2, 3 and 100, plus terminal-ignored runs.
  Mismatches with the vectorized decoder: **0** out of 11,000 decodes.
- **Normalization.** Dividing beta by its row maximum does not change the
  argmax. A zero row is impossible because the sampled path has nonzero
  probability.
- **Lookahead edge cases.** Negative lookahead equals 0. Lookahead beyond the
  last round equals full (`check_edge_cases.out`).
- **Survival.** Exact survival (Fourier, every history) is never above MLD.
  Full lookahead is close to MLD, for example 0.96686 vs 0.96715 (c0 m2,
  p = 1e-3) and 0.64020 vs 0.64934 at p = 1e-2. Lookahead 0 is always the
  worst on non-terminal circuits.
- **Sampled, m = 18 and 22.** Rao-Blackwellized survival satisfies
  MLD >= window shot by shot. For c0 m22 at p = 1e-3: MLD 0.7814, w100 0.7744,
  w2 0.7731, w1 0.7649, w0 0.7314.
- **Production.** 0 violations of MLD >= window in 5670 baseline and 12960
  variant comparisons. w0 <= w1 <= w2 holds in 1888 of 1890 baseline entries;
  in the two exceptions w1 exceeds w2 by at most 3e-6. The median per-circuit infidelity ratio
  to MLD is 1.18 / 1.027 / 1.012 for lookahead 0 / 1 / 2.

### (3) constrained_joint with allowed sets

`_allowed_matrix` builds `A[a, j] = allowed[a + offset_j]`, which is correct
by construction and checked exactly. It matches brute force to 5.7e-15 for
the `X_PART_TRIVIAL` final round and to 3.5e-14 for random mixtures of int,
None and boolean masks per round, including terminal int and terminal mask
on terminal circuits.

Sampled checks on deeper circuits (production c3 m10; par_term_chk m4, which
has 8 rounds plus the terminal) at p = 5e-3 also pass. The specs are random
mixes of masks, 0, None and `X_PART_TRIVIAL`, plus terminal masks.
- 16 specs x 3 logical values on one shared 3e5-shot sample per circuit give
  max z = 3.37. The specs overlap heavily, so these z values are correlated
  (`check_constrained_sampled.out`).
- Four independent 3e5-shot batches for 5 specs on c3 m10 give pooled
  |z| <= 2.06 over 1.2e6 shots, with per-batch |z| <= 2.24
  (`check_constrained_resample.out`).

### (4) variants.build_variant_circuit

Checked by `check_variants.py` / `.out`, with depth-2 circuit text in
`variant_{base,chk,par,term,par_term_chk}_m2.txt`.

- **checked_inverse.** The block pattern for m = 2 is `GQR GQR GQR GQ T`
  (G = data-only gates, Q = check round, R = reset, T = terminal readout).
  Each gate block equals the expected `logical_gate_circuit` sequence: prep +
  C1, then C2, then C2^-1, then C1^-1. There is a check+reset after every
  inverse Clifford except the last, which is followed by the final round only.
  The circuit has 2m rounds (4 for m = 2, 6 for m = 3). Every detector
  `lrb_stab_r{k}_w{w}` reads the k-th check round's measurement of wire w.
- **parallel_measurement.** There is one measurement layer per round,
  measuring (5, 6, 7, 8):
  - idle 2p `N1 d` on data wires 0-4 only;
  - then `N1 d` (1x) and `N1 f` (5x) on each ancilla;
  - the reset layer is unchanged.

  The CNOT/H order is the production order with the measurements deferred,
  which is noiselessly equivalent because the X-check CNOTs commute with the
  stored Z parities. The m = 0 floor drops from 4.95 p to 3.45 p, matching
  P(correct) 0.99655 in the variant table.
- **full_terminal_readout.** The readout is noiseless `H_INV` + `M` on 0-4.
  Terminal detectors are `x0 + x1 - x2` and `-x1 + x3 - x4`, identical to the
  production `const0` detectors. The logical is `x0 + x3`. The learned
  terminal map is `[[0,0,2,0],[0,0,0,2]]`, i.e. minus the ancilla X-syndrome,
  consistent with the circuit convention.
- **Noiseless determinism.** Noise was stripped and SDIM's tableau simulator
  (force_tableau, 12 seeds) run on base, chk, par, term and par_term_chk at
  m = 2, and on production LRB and const0 at m = 2. Every detector was 0 in
  every shot and the logical was always 0.
- **Chain structure.** The chain model of every variant builds without
  chain-assumption errors and matches brute force exactly (above).

## Scripts and outputs (all in `review/V4/`)

| script | checks | output |
|---|---|---|
| `v4_common.py` | Fourier brute force, helpers | - |
| `check_fourier_selftest.py` | brute force vs direct convolution | `.out` |
| `check_exact_small.py` | exact forward, LRB-D, constrained sets, BFD rules, window survival | `.out` |
| `check_first_order_bfd.py` | single-fault enumeration, all rules and masks | `.out` |
| `check_xtriv_offender.py` | BFD-X offending faults and exact small-p coefficients | `.out` |
| `check_offender_tableau.py` | independent tableau injection of the BFD-X offender | `.out` |
| `check_bfd_xtriv_impact.py` | lenient vs strict BFD-X on 30 production sequences | `.out` |
| `check_window.py` | window reference vs vectorized; sampled survival; terminal in lookahead | `.out` |
| `check_window_terminal_exact.py` | exact effect of terminal in lookahead | `.out` |
| `check_edge_cases.py` | unobserved final round; lookahead edges; sampled BFD-X mismatch | `.out` |
| `check_constrained_sampled.py`, `check_constrained_resample.py` | constrained_joint with allowed sets vs sampling on deeper circuits | `.out` |
| `check_production_npz.py` | production MLD >= window; BFD sampled vs exact | `.out`, `_outliers.out` |
| `check_variants.py` | variant structure, noise layers, detectors, noiseless determinism | `.out`, `variant_*_m2.txt` |
