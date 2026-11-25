# R3-physics: adversarial physics review of `REPORT.md`

Reviewer id: R3-physics. Lens: try to refute the main conclusions (a) to (f).
Sections 13.1 and 13.2 of the report were ignored as instructed. Nothing under
`src/`, `scripts/`, `tests/` or `REPORT.md` was edited.

## Checks run (all on the login node, under 1 min each, read-only inputs)

Scripts and their outputs are in `notes/report-review/R3-physics/`:

| Script | What it checks | Output |
|---|---|---|
| `first_order_attribution.py` | First-order MLD failure split by fault category, for checked circuits (full MLD) and `const 0` circuits (terminal MLD), from the cached chain structures `.cache_correction/structures/c00_d8.pkl`, `c07_d8.pkl`, `c00_d5.pkl`; per-syndrome posteriors for `const 0`; symmetric-noise counterfactual | `first_order_attribution_output.txt` |
| `mixture_floor_check.py` | 50/50 quasi-static mixture (p = 5e-4 / 2e-3) of the exact per-sequence joints, for the base protocol and for the full-terminal-readout variants (floor removed) | `mixture_floor_check_output.txt` |
| `anchor_and_convention_check.py` | (1) refit of r with the m = 0 point dropped; (2) all "versus physical RB" ratios against the run's own `results/RB` (p_RB = p) as well as `results/RB_comparison` (p_RB = 5p) | `anchor_and_convention_check_output.txt` |
| `frontier_and_shape_check.py` | BFD and `unif 2` against the sampled soft-MLD frontier; same-information shape comparison; soft-MLD windowed rates vs p; unif-1/BFD curvature vs p; checked-inverse effect on correction rules; dense-window spreads; degenerate CIs | `frontier_and_shape_check_output.txt` |

## What survives

- **(a), checked circuits.** Full MLD on the checked circuits is worse than
  physical RB under either reference (3.0 to 4.7 times at p_RB = 5p, 14 to 23
  times at p_RB = p). The first-order attribution reproduces the report's
  coefficient (11.1 to 12.1 p per depth step), and MLD is pointwise optimal
  among accept-all rules, so no other decoder can fix it.
- **(b), BFD.** BFD has no first-order term in its depth-dependent accepted
  error. Its r scales as p^2 (2.1e-6, 1.31e-5, 5.3e-5 at p = 2e-4, 5e-4,
  1e-3).
- **(b), rules suggested in the brief.** None of them changes the ranking.
  - Ignoring the Z part of the final round (`lrbd:xtriv`, `bfd:xtriv`, already
    in `results/decoder_tables.md`) gains at most 0.1% acceptance per step and
    raises r 2.5 to 2.9 times.
  - Combining BFD with soft MLD cannot beat the per-depth soft-MLD frontier.
- **(d).** The forward filter is the exact MAP of the terminal outcome. At the
  final time, filtering and smoothing coincide. The Pauli-frame equivalence
  holds for the stated Pauli model.

Twelve findings below change or qualify a stated conclusion. Six more are minor.

## Findings, most severe first

### F1 (critical): the report's own `const 0` MLD result contradicts "genuine correction needs distance 3"

**Quote** (lines 120-121): "Genuine correction (an unpostselected logical decay
below the physical qutrit's) needs distance 3." Also lines 791-793:
"`[[5,1,2]]_3` with optimal correction is never better than the bare qutrit.
... and at circuit level, where it is 3 times worse." And lines 45-47: "The
cause is structural: 16 of the 40 single-qutrit errors ... are ambiguous, so
first-order logical errors cannot be avoided."

**Problem.** In the report's own convention, `const 0` + MLD meets the
report's definition of genuine correction.
- It is accept-all (acceptance 1) and exact MLD on a distance-2 code.
- It decays at 0.44 to 0.49 times physical RB at every p from 1e-4 to 5e-2.
- The exec summary (line 74) and Section 4.6 report this result. Sections 1, 8
  and 9 then state the opposite conclusion.

"Structural" is also wrong as an explanation of the gap. Ambiguity forces a
first-order logical rate, but not a rate above the physical qutrit's.

**Evidence.**
- `results/decoder_tables.md`, p = 1e-3: "Full MLD, const 0 circuits | 0.00167".
  Physical RB is 3.76e-3.
- `anchor_and_convention_check_output.txt`, `mld:const0`: 0.44 to 0.45 times at
  p_RB = 5p, but 2.13 to 2.18 times at p_RB = p.
- So the conclusion holds under p_RB = p and fails under p_RB = 5p, the
  convention the report uses everywhere else.

**Fix.**
- Say: "With mid-circuit checks, no distance-2 decoder beats physical RB under
  either reference. Without checks, terminal MLD beats physical RB at
  p_RB = 5p (0.45 times) but not at p_RB = p (2.2 times worse)."
- Drop "structural". Give the mechanism from F3.
- Make Sections 1, 8 and 9 consistent with this.

### F2 (major): p_RB = 5p is labelled "matched noise", but it is a display normalization, and several headline comparisons flip under the run's own p_RB = p reference

**Quote** (line 29): "Physical RB at the repository's matched noise `p_RB = 5p`
has `r = 3.76 p` per depth step." Lines 58-60: "a 0.9 threshold gives 3.1%
rejection per step and `r = 9.4e-4` (4 times below physical RB ...)". Line 392:
"4 to 7 times below physical RB".

**Problem.** The repository uses 5p as a plotting normalization, not as a
physically matched reference.
- The notebook labels the axis "Normalized SI1000 Noise Strength λ = 5p".
- `scripts/run_physical_rb_comparison.py` lines 4-6 pairs RB p "with the
  displayed LRB coordinate, for example LRB lambda = 5 * raw_p".
- The same baseline run also stores physical RB at p_RB = p
  (`results/RB/<i>.csv`). That is a qutrit built from the same SI1000
  single-qutrit gates (0.1 p per gate) that the code's circuits use.
- The report's own hardware mapping (Section 7.2) puts single-qutrit Cliffords
  at p of about 0.03 to 0.07 and two-qutrit gates at about 0.027 to 0.18. That
  gives p_RB/p between about 0.2 and 2.6, never 5. It is lower still once you
  account for the simulated physical Clifford being about 5 noisy gate tokens.

**Evidence.** From `anchor_and_convention_check_output.txt`:

| Rule | Ratio at p_RB = 5p | Ratio at p_RB = p |
|---|---|---|
| Soft MLD t = 0.9 | 0.20 to 0.25 | 0.92 to 1.23 (1.23 at p = 1e-3: worse than physical RB) |
| BFD | 0.01 at p = 1e-3 ("70 times below") | 0.07 at p = 1e-3; 1.08 at p = 1e-2 |
| Full MLD | 3.15 | 15.5 |
| `const 0` MLD | 0.44 | 2.18 |

The distance-3 break-even estimates in Section 8 are measured against "2.5p per
Clifford" (`notes/A5-distance3.md`, line 426), the same 5p reference. Under
p_RB = p every p* falls 5 times: about 2e-5 to 1.3e-4 with sequential readout
and 2.6e-4 to 6.8e-4 with simultaneous readout.

**Fix.**
- Call 5p "the repository's λ = 5p display convention".
- Quote every "versus physical RB" ratio at both p_RB = 5p and p_RB = p,
  including in the executive summary and Sections 4.3, 4.4 and 8.
- State that soft MLD at t = 0.9 does not beat physical RB at p_RB = p.

### F3 (major): the mechanisms behind the full-MLD loss and the `const 0` win are misattributed

**Quote** (lines 74-77): "In this noise model the check rounds themselves (about
118 p of fault probability per round, against about 4.7 p per logical Clifford)
are the dominant error source." Lines 45-47: "The cause is structural ...".

**Evidence** (`first_order_attribution_output.txt`, depth 22, c00):

- **Checked circuits, full MLD** (11.1 p per step):
  - 70.9% comes from data idle in the four sequential measurement layers and
    the reset layer (SI1000's 2 p per layer, about 10 p per data qutrit per
    round);
  - 13.9% from check CNOTs;
  - 10.5% from the logical CZ/SWAP on qutrits 2 and 3;
  - 4.7% from single-qutrit gates;
  - 0% from the rest of the 118 p per round: measurement flips, reset flips,
    ancilla idle and ancilla depolarizing (552 p + 460 p + 176 p + 92 p of
    fault mass, all decoded correctly).
  - c07 gives 72.5%; c00 at depth 14 gives 69.9%.
- **`const 0`, terminal MLD** (1.68 p per step):
  - 100% of first-order failures come from the 0.1 p single-qutrit gate faults;
  - the 134 p of CZ/SWAP fault mass on (2,3), which is 65% of all fault mass,
    contributes 0.
  - Every ambiguous terminal syndrome has a lopsided posterior: 0.69/0.31,
    0.79/0.21, 0.77/0.23. MLD always picks the "(2,3) fault" explanation.
  - Failure fraction (failure over fault mass): 0.18 for this noise against
    0.30 for the same fault mass spread symmetrically over the data qutrits.

**Problem.** The `const 0` advantage does not come from "fewer faults" alone.
- It relies on SI1000 as implemented having no idle noise during gate layers.
  The standard SI1000 has p/10 idle per layer, and hardware has crosstalk and
  idle during the long CZ/SWAP. Either would erode the asymmetry.
- `notes/H1-sc-hardware.md` §7.1 already notes that 79-83% of single faults in
  `const 0` fall in ambiguous classes ("correction is weak"). My count is
  81.5% of detected faults.

**Fix.** Replace the "118 p" sentence with the attribution above. State that
the `const 0` result depends on the gate noise being concentrated on (2,3),
and label its hardware relevance [CONJ].

### F4 (major): "BFD never accepts a first-order logical error" is false as stated, and the [VN] tag does not support it

**Quote** (lines 51-52): "It never accepts a first-order logical error **[VN]**."
Line 401: "Never accepts a first-order logical error."

**Problem.**
- BFD accepts the depth-independent 4.95 p final-round floor, which is first
  order.
- The supporting field `bfd_accepted_logical = 0.0` in
  `results/first_order_check.json` counts only detected (blip) faults.
  `studies/correction-decoders/qutrit/scripts/verify_first_order.py` skips undetected faults with
  `continue` before the BFD check.
- In the same JSON, `exact_bfd_logical` = 4.92e-4 at p = 1e-4 (m = 0), that
  is 4.9 p.

**Fix.** Write: "BFD accepts no first-order logical error beyond the
depth-independent 4.95 p final-round floor that every rule shares. Its
depth-dependent accepted error is second order (r of about 53 p^2)."

### F5 (major): BFD "sits on" the frontier while `unif 2` is "strictly dominated", but BFD is as far from the frontier as `unif 2`, and `unif 1`'s frontier membership was not tested at m = 6 and 22

**Quote** (lines 368-372): "holds within the precision of the sampled frontier
at m = 6 and 22. `const c` and `unif 2` lie strictly below it **[VN]**. ... BFD
sits on it". Line 63: "Sparse-check LRB-D (`const c`, `unif k >= 2`) is
strictly dominated **[VN, VA]**."

**Evidence** (`frontier_and_shape_check_output.txt` §1). Infidelity divided by
the interpolated sampled-frontier infidelity at equal acceptance:

| Point | BFD | `unif 2` |
|---|---|---|
| p = 1e-3, m = 22 | 1.041 | 1.033 |
| p = 1e-3, m = 6 | 1.015 | 1.008 |
| p = 2e-3, m = 22 | 1.075 | 1.070 |
| p = 5e-4, m = 22 | 1.021 | 1.016 |

- Netting out the 4.95 p floor, BFD's depth-dependent infidelity is about 25%
  above the frontier at p = 1e-3, m = 22.
- `unif 1`'s acceptance lies below the lowest sampled frontier point (0.483 vs
  0.535 at m = 6; 0.081 vs 0.093 at m = 22), so "on the frontier" at those
  depths is an extrapolation.
- The frontier is a common threshold applied to a mean of per-sequence ratios.
  Chow's optimality holds for the pooled ratio, so percent-level comparisons
  carry that caveat too.

**Fix.**
- Say that BFD and `unif 2` both lie slightly below the frontier (1 to 8% in
  total infidelity).
- Drop "strictly dominated" for `unif 2`, or apply it equally to BFD.
- Say that `unif 1`'s optimality is proven only at m = 0 and 2.

### F6 (major): soft-MLD decay rates at p >= 2e-3 come from the fixed-threshold cap, so "4 to 7 times below physical RB" is not a property of the logical gates

**Quote** (line 392): "| Soft MLD, t = 0.9 | between first and second order; 4
to 7 times below physical RB from p = 2e-4 to 1e-2 | |".

**Problem.**
- With a fixed threshold, accepted P(correct) is at least t at every depth, so
  the decay must flatten. Section 4.2 says this, but the rate table ignores it.
- The windowed 1-f of soft MLD at t = 0.9 falls with depth
  (`frontier_and_shape_check_output.txt` §3):

  | p | Windowed 1-f, first to last window | c2 |
  |---|---|---|
  | 5e-4 | falls 14% | |
  | 1e-3 | falls 31% | +6.2e-6 |
  | 2e-3 | 3.3e-3 to 1.7e-3, falls 2 times | +4.4e-5 |
  | 5e-3 | 9.5e-3 to about 1.1e-3 to 1.9e-3, falls 5 to 9 times | +1.1e-4 |

- At p = 5e-3, accepted P(correct) at m = 22 is 0.928 against a cap of 0.9.
  The "6 to 7 times below physical RB" at p >= 5e-3 measures the cap, not the
  gates.
- Section 5.3 lists soft MLD's positive curvature as a physical "mild
  deceleration".

**Fix.**
- Quote soft MLD as accepted error at a stated m and acceptance, or use a
  depth-dependent threshold with a fixed rejection quantile per step, which
  gives a well-defined per-step rate.
- Drop the physical-RB ratio for p >= 2e-3.
- Attribute soft MLD's c2 to the cap.

### F7 (major): "LRB-D every round and BFD decay as a single exponential on the 9 production depths" is false at m = 0 and at p >= 2e-3, and r includes a 10% boundary transient

**Quote** (lines 84-86): "LRB-D with a check every round, and BFD, decay as a
single exponential within statistics on the 9 production depths." Line 400
(rank 1): "exponential decay."

**Evidence.**
- The m = 0 to 2 window is steeper at every p:
  - `unif 1` at p = 1e-3: 5.12e-5 [4.69e-5, 5.46e-5] per step, against
    2.2e-5 to 2.9e-5 in every later window (CIs do not overlap). The ratio of
    the first to the second window is 1.91 to 1.95 for every p up to 1e-2.
  - BFD: 1.20e-4 [1.13e-4, 1.26e-4] against at most 8.0e-5 (ratio 1.6 to 1.7).
- The c2 test hides this. `aggregate.curvature(..., min_depth=2)`
  (`studies/correction-decoders/qutrit/scripts/analyze_study.py` line 106) drops m = 0.
- The fit keeps m = 0 and effectively pins it. That point has zero
  across-sequence std, and `_safe_weights` gives it about 1e6 times the
  weight of any other depth.
- The anchored r is 9 to 12% higher than a refit without m = 0 for LRB-D and
  BFD (2.0e-5 vs 1.79e-5; 5.3e-5 vs 4.87e-5 at p = 1e-3), and 24% higher for
  soft MLD at t = 0.97 and p = 2e-3 (`anchor_and_convention_check_output.txt`).
- Curvature is significant on the production depths at p >= 2e-3 for LRB-D:

  | p | c2 [95% CI] |
  |---|---|
  | 2e-3 | -5.2e-7 [-9.6e-7, -7.0e-8] |
  | 5e-3 | -5.6e-6 [-8.8e-6, -2.5e-6] |
  | 1e-2 | -5.2e-5 |
  | 2e-2 | -6.9e-4 |

  The windowed rate from m = 2-4 to m = 20-22 rises 48% at p = 1e-2 and 89% at
  p = 2e-2. For BFD the curvature is significant at p >= 7.5e-3.
- The window baselines are inconsistent. const 1's "18 times" and const 0's
  "26 times" (line 89) use the m = 0 to 2 window, while `unif 1`'s flatness
  excludes it. With m >= 2 windows throughout, const 1 grows 9.9 times and
  const 0 8.4 times. With m = 0 included throughout, `unif 1`'s last window is
  0.57 times its first.

**Fix.**
- Report r both with m = 0 and from m >= 2 with a free amplitude.
- State that the dense-check rules have a 2-times boundary transient.
- Qualify "single exponential" as p <= 1e-3.
- Use one window convention for all rules.

### F8 (major): "Correction mitigates non-exponentiality in shape only" compares different information, and the cause of MLD's curvature is untested

**Quote** (lines 91-95): "MLD's windowed rate drifts by 15% (downward) where
`const 1` grows 18 times." Lines 615-616: "a significant (positive) curvature
remains, from boundary and decoder-memory effects".

**Problem.** Full MLD sees every round; `const 1` sees one. This mixes
information with correction.

**Evidence** (`frontier_and_shape_check_output.txt` §2, p = 1e-3). With the same
information:

| Pair | Detection | Correction |
|---|---|---|
| Terminal round only | `const 1`: grows 18 times | MLD: c2 = -3.04e-4 [-3.37e-4, -2.74e-4], windowed 1-f rises 2.3e-2 to 3.6e-2 (+59%, accelerating) |
| `unif 22` rounds | | MLD: 0.60 sawtooth |
| All rounds | `unif 1`: +11% (m >= 2) | MLD: -15% |

- So correction reduces the drift from 18 times to 1.6 times, but MLD with
  `const 1`'s information is still clearly non-exponential.
- With all rounds, detection and correction are about equally exponential.
- The undecoded "raw" curve has a similar positive c2: +3.95e-5
  [-2.9e-5, +9.6e-5] against MLD's +5.36e-5. A cause other than decoder memory
  is not excluded, for example the non-uniform Clifford catalog of Section 10.1
  (TV 3.7%), which makes the twirl inexact.

**Fix.** Add the same-information rows (terminal MLD vs `const 1`; restricted
MLD vs `unif k`). Relabel "from boundary and decoder-memory effects" as [CONJ].

### F9 (major): under quasi-static noise the "rise" is an amplitude (floor) effect, and MLD does not "decay normally"

**Quote** (lines 100-103): "LRB-D's accepted fidelity then *rises* from 0.9908
at `m = 0` to 0.9958 at `m = 22` ... so the fitted decay rate is meaningless.
MLD correction decays normally from 0.974 to 0.617 **[VN]**." Lines 624-626:
"**Correction removes a different, more dangerous non-Markovianity**".

**Evidence** (`mixture_floor_check_output.txt`, `results/mixture_nonmarkov.csv`):

- **The rise comes from the floor.** The rise needs the m-independent
  4.95 p floor, which differs between the two components (λ(0) = 0.99627 vs
  0.98481). With full terminal readout, which removes the floor and which the
  report recommends:
  - LRB-D mixture: no rise, 0.99994 to 0.99981 (−1.2e-4); fitted
    r = 4.7e-6, against component rates 3.98e-6 and 6.78e-5 (mean 3.6e-5);
  - BFD: fitted r = 2.8e-5 against a component mean of 1.0e-4.
- **What remains is a rate bias.** The real hardware-relevant effect is a 3.6
  to 8 times downward bias toward the quiet component, not a meaningless
  negative rate.
- **MLD's mixture is not exponential.** It is a mixture of exponentials with
  c2 = +1.68e-4 [1.28e-4, 1.99e-4] (significant). Its windowed 1-f falls 28%
  (2.39e-2 to 1.72e-2). Its fitted r (1.43e-2) is 9% below the component mean
  (1.56e-2).
- So correction removes the selection bias but not the non-exponential
  signature of quasi-static noise.

**Fix.**
- Explain the rise as floor plus selection.
- Quote the floor-free bias (factor 3.6 to 8).
- Replace "decays normally" with "decays as a non-exponential mixture
  (c2 > 0), with an r close to the shot average".
- Reword verdict item 3: correction removes the selection, not the
  non-Markovian signature.

### F10 (major): "leave no long unchecked interval" holds for detection only; for correction rules the checked inverse worsens both rate and shape

**Quote** (lines 627-629): "**The decisive fix is protocol-level:** leave no long
unchecked interval. Check inside the inverse block or compile a single recovery
Clifford". Line 654: "Checking inside the inverse half buys shape (Section
5.4), not rate."

**Evidence** (`frontier_and_shape_check_output.txt` §5, variant study):

| Rule | p | r, base to `chk` | c2, base to `chk` |
|---|---|---|---|
| Full MLD | 1e-3 | 1.19e-2 to 2.0e-2 | +3.3e-5 to +9.4e-5 [4.0e-5, 1.4e-4] |
| Full MLD | 2e-3 | | +1.0e-4 to +2.2e-4 |
| Soft MLD t = 0.9 | 1e-3 | | +7.1e-6 to +3.0e-5 (CIs disjoint) |

LRB-D and BFD do become flat.

**Fix.** Restrict verdict item 4 and Stage C to detection rules. For
correction, recommend the single compiled recovery Clifford, which is still
unsimulated [CONJ].

### F11 (major): "no ancilla reset is needed" is untested and contradicts the report's own hardware notes; BFD's rule changes without reset

**Quote** (lines 107-109): "No feedback, no decoder latency budget, and no
ancilla reset are needed **[VA, VN, LIT]**." Line 684: "The ancilla reset can
be dropped." Stage C (line 771): "no reset or unconditional reset".

**Problem.**
- No variant without reset was simulated, so [VN] has no support.
- `notes/H1-sc-hardware.md` §3.4 says that an un-reset qutrit ancilla in
  |s ≠ 0> decays with probability t_round/T1 (6-26% per round). It concludes:
  "Unconditional reset at 0.1-0.2% is preferable whenever syndromes are often
  nonzero". At hardware p of about 0.03, syndromes are non-zero almost every
  round.
- An un-reset ancilla also makes the ancilla error rate depend on the previous
  outcome (|2> decays faster than |1>). That is a history-dependent error
  source, the very kind of non-Markovianity Q2 is about.
- BFD's blip rule changes without reset. With difference detectors
  d_r = o_r - o_{r-1}:
  - a classical readout misassignment shows up in two consecutive detectors
    with opposite signs, so BFD as defined rejects it and loses most of its
    acceptance gain;
  - a pre-measurement state flip shows up once.
  - The rule must be re-derived for the hardware's measurement-error mechanism.

**Fix.** Recommend unconditional reset (or simulate the no-reset variant and
redefine BFD), and drop "no ancilla reset" from the [VN] claim.

### F12 (major): Stage A "runs on IQM-class square lattices today" is unsupported and motivated by the convention-dependent 0.45 times figure; the summary bullet has no evidence tag

**Quote** (lines 114-116): "The sustainable path is staged: 1. `const 0` LRB on
the five data qutrits with terminal decoding. It needs no mid-circuit
measurement and runs on IQM-class square lattices today." (untagged). Lines
763-764: "MLD correction of it (0.45 times physical RB in simulation)".

**Problem.**
- Section 7.5 says "Not found: native qutrit gates in the circuit API,
  multi-state discrimination, or ternary feedback".
- Stage A needs native qutrit CZ/SWAP and three-level readout of five data
  qutrits.
- Stage A also needs a noisy unitary encoder. The simulation's preparation is
  ideal, and the encoder's couplings (0-1, 0-3, 1-2, 2-4 plus the fold,
  `notes/H1-sc-hardware.md` §1 and §7.1) are not mentioned in the report.
- H1's expected Stage A numbers under its hardware model are left out:
  accepted fidelity 0.996 / 0.94 / 0.81 at m = 2 / 6 / 10, with acceptance
  0.76 / 0.39 / 0.24 (HW-now).
- The 0.45 times figure is 2.2 times worse at p_RB = p (F2) and depends on the
  noise asymmetry (F3).

**Fix.**
- Tag the bullet [CONJ].
- Replace "today" with "with pulse-level qutrit calibration".
- Mention the encoder and H1's HW-now numbers.
- Drop 0.45 times as the motivation.

### F13 (minor): "Full terminal readout is free" holds only with noiseless readout

**Quote** (line 646): "**Full terminal readout** is free, adds no noise, and
removes the 4.95 p floor."

**Problem.** On hardware, with 93-97% ternary readout (Section 7.2), reading
five qutrits instead of two adds three noisy readouts to the terminal syndrome.
That means extra rejections for LRB-D and miscorrections for MLD.

**Fix.** Add "in this noiseless-readout model" and a [CONJ] estimate of the
hardware cost.

### F14 (minor): the causal/offline wording contradicts itself, and the latency-loss numbers are mislabelled

**Quote** (lines 418-419): "Offline decoding does not change what is measured.
It measures a better logical channel than any committing decoder could
deliver." Lines 66-67: "4%, 9% and 31% worse than MLD **[VA, VN]**."

**Problem.**
- The second sentence contradicts the first. The decoded rate depends on how
  early the decoder must commit (up to +31%).
- The percentages are numerical results, not analytical.

**Fix.** Say: "Offline decoding measures the logical rate of Clifford-only,
frame-tracked computation. Rules that must commit corrections (for example
before non-Clifford gates) see rates 4 to 31% higher [VN]."

### F15 (minor): statistical reporting

**Quote** (line 262): "95% intervals from 100 bootstrap resamples of the 30
sequences". Line 562: "windowed-rate spread (max/min, m >= 2)".

**Problems.**
- **Resample count.** `analyze_study.py --boot` defaults to 200, and Section
  12's reproduction command uses the default.
- **Grid-limited CIs.** The CIs come from percentile bootstraps, and four fits
  at p = 1e-4 have degenerate intervals limited by the fit grid, for example
  LRB-D r = 2.01e-7 with CI [2.01e-7, 2.01e-7].
- **Multiple comparisons.** Dozens of c2 intervals are called significant one
  at a time. The dense LRB-D sign is at least consistent across p.
- **Spread window.** The dense spread column actually includes the m = 1 to 2
  window. With windows starting at m >= 2, BFD's checked-inverse spread is 1.13
  (not 1.49) and its base spread is 1.62 (not 2.0). BFD's windows become as
  flat as LRB-D's.

**Fix.** Correct the resample count, report grid resolution, note the
multiplicity, and fix the window definition. Also see F7 on m = 0.

### F16 (minor): stated ranges do not match the data

**Quote** (lines 44-45): "worse than physical RB at every p simulated (1e-4 to
2e-2)". Line 393: "11 to 15 p, 3 to 4 times above physical RB at every p". Line
394: "0.44 to 0.46 times physical RB at every p".

**Evidence.**
- Sampled MLD exists only up to p = 1e-2.
- At p = 1e-2, full MLD is 17.4 p and 4.74 times physical RB.
- `const 0` MLD reaches 0.48 to 0.49 at p = 4e-2 to 5e-2.

**Fix.** Correct the ranges.

### F17 (minor): refuted and missing ideas are not listed

The report computes, but never discusses, "final round X-part only" rules
(`results/decoder_tables.md`). They refute the idea of ignoring the Z part of
the final round: about 0.1% more acceptance per step for 2.5 to 2.9 times the
r. Three other ideas are missing:
- analog (IQ) readout likelihoods as erasure-style soft information for BFD and
  MLD on hardware;
- a depth-dependent soft threshold with a fixed rejection per step (see F6);
- the observation that protocol variants (`const 0` LRB-D: 9.1e-6 at 0.992;
  `par_term` LRB-D: 4.8e-6 at 0.930) beat the rank-1 rule in both r and
  acceptance.

**Fix.** Add these to Section 9 and Section 11.

### F18 (minor): `results/key_summary.md` disagrees with the report

**Problem.** Its header says "r per Clifford", and its physical-RB column uses
the free-offset fit (4.92e-3 at p = 1e-3). The report uses per depth step and
the offset-0 value 3.76e-3. Section 12 lists the file as a result.

**Fix.** Regenerate or delete it.
