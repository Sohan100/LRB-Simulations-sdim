# Critique of A2: optimal postselection and the figure of merit

Adversarial review of `notes/A2-optimal-postselection.md`, its script
`notes/scripts/A2_optimal_rule_check.py` and the stored output, 2026-10-28.
Nothing committed. Nothing written outside the notes folder. The baseline
run folder and the main checkout were only read.

This file replaces an earlier critique of the same notes written by a
previous critic run. That run's scripts
(`critic_A2-optimal-postselection_{theory,codecap,circuit,baseline}.py`) are
still in `notes/scripts/`. Every check below is my own, in new scripts with
the suffix `_r2_`. Where I reach the same conclusion as the earlier critique,
I re-derived it independently and say so.

Labels on my own statements: **[VA]** verified analytically, **[VN]**
verified numerically by a script listed below, **[LIT]** from a source I
fetched in this task (Section 5), **[CONJ]** conjecture.

## Scripts

All in `notes/scripts/`. Run with `module load python/3.11`, one core, BLAS
limited to one thread. Outputs are saved next to each script as
`*_output.txt` (the circuit and chain scripts have one output file per mode).

| script | what it checks | CPU |
|---|---|---|
| `critic_A2-optimal-postselection_r2_theory.py` | Theorem 1 by LP on tables with forced ties; frontier shape and Chow identity; metric identities for d=2..7; Examples A/B incl. covariant variants; legacy maximiser at fixed α<1 by exact vertex enumeration (loss bound on positive-loss cases); estimator variance and Rayleigh floor; biased local rule; unchecked-half and raw-detector toys; shot cost with B unknown; regret bound; Forney with unequal priors | 16 s |
| `critic_A2-optimal-postselection_r2_codecap.py` | code capacity in exact rational arithmetic from integer weight enumerators; syndromes from symplectic products | 3 s |
| `critic_A2-optimal-postselection_r2_circuit.py` | `conv`: exact P(D,L) by direct nonnegative convolution (no DFT), compared with A2's DFT, all-history ranking, Monte Carlo with the repository's DEM sampler; `dft`: 30 sequences at p=0.01, m=2 and p=0.02, 0.05 at m=0, 2; `ref`: noiseless reference values on 90 circuits | 96 s, 98 s, 13 s |
| `critic_A2-optimal-postselection_r2_baseline.py` | CSV fits with standard errors, `const 1` saturation and floor with z-scores, Rice inversion, weighted fit and repository fit table, across-sequence vs shot-noise SD | <1 s |
| `critic_A2-optimal-postselection_r2_chain.py` | exact posteriors of chosen histories at any depth by a hidden-state contraction over check rounds (validated against the full law at m=2); m = 4, 6, 10, 22 | 28 s + 276 s |

I also re-ran `A2_optimal_rule_check.py`. Its output is identical to the
stored file except for timing lines.

---

## 0. Verdicts

| id | verdict | one-line reason |
|---|---|---|
| C1 | confirmed | Re-derived; LP with forced ties matches to 2.2e-16 and LP vertices satisfy (a), (b). |
| C2 | confirmed | Re-derived; Chow identity holds with atoms (closed interval, left-continuous R). |
| C3 | needs_qualification | Fetched sources do not state the multi-class max-posterior form; Forney's form assumes equal priors, which LRB violates; Forney/NP is second-hand. |
| C4 | confirmed | Corollary of Theorem 1's optimality conditions. |
| C5 | confirmed | Exact on all histories (no cutoff needed); extended to 30 sequences and to p=0.02, 0.05. The p=0.01, m=2 numbers quoted are 8-sequence values. |
| C6 | confirmed | Exact rational arithmetic, including the tie structure (no three-way ties). |
| C7 | needs_qualification | Identity right; also fails for d=4 and also holds for d=2, so "special to d=3" (Sec. 3.2) is wrong. |
| C8 | confirmed | Last inequality needs λ>0 (the notes say so, the claim does not). |
| C9 | confirmed | Proof right. A2's numerical check is tautological (the DFT construction forces symmetry); my direct convolution gives a real check. |
| C10 | needs_qualification | Needs λ≥0 per circuit and holds for population values only; fails in the floor regime; its numerics are tautological; a related statement in Sec. 3.5 is false. |
| C11 | confirmed | Reproduced. The never-correct map in Example A is covariant; Example B needs a non-covariant rule. |
| C12 | confirmed | A2's numerical test was vacuous (loss always 0); mine has 87 positive-loss cases at fixed α<1 and no violation. |
| C13 | needs_qualification | Estimator results right, but m=14 is not at the floor (+3.3 SE). m≥18 is. |
| C14 | needs_qualification | m=14 is +3.4 SE above 3^-4; "0.012344" is the m=18 value. |
| C15 | needs_qualification | "Exactly" needs i.i.d. per-step increments with every step checked and no hidden syndrome state; LRB-D violates this. |
| C16 | needs_qualification | Numbers reproduce. Acceptance misfit is entirely the m=0 point; η is 1−a here, −ln a in the notes; r_c is fit-dependent (unweighted 1.8e-5 vs repository 2.03e-5). |
| C17 | needs_qualification | Needs exact posteriors; acceptance vanishes once t > ess sup q(m); "arbitrarily slow" only on a finite window. |
| C18 | confirmed | Proposition immediate; counterexample correct (as a statement about curves). |
| C19 | needs_qualification | Assumes B known (×1.74 if fitted) and shot noise only (true for `unif 1` at p=1e-3, false for `const` policies). |
| C20 | confirmed | Each MAP term (3q−1)/2 ≥ 0. |
| C21 | needs_qualification | "Only when" (necessity) is unproven; as worded the conditions are not sufficient either. |
| C22 | confirmed | Re-derived; no violation in 3000 tables (max ratio 0.66). |
| C23 | unverifiable | Correctly labelled. New exact evidence up to m=22: 0 of ~130,000 candidate histories outrank D=0 (min margin 3.4e-5). Exhaustive ranking still infeasible. |
| C24 | confirmed | Reproduced; also holds for 30 sequences at p=0.01, m=2 (within 1.5 SE). |
| C25 | confirmed | Fetched arXiv HTML; Fig. 3 caption gives exactly these numbers and fits. |
| C26 | confirmed | Abstracts match; minor caveats on non-MAP decoders and the third class. |

---

## 1. Reproduction and code inspection

1. **Repository facts used by the claims** [VA, code read]:
   - `dem_simulation.py` samples `N1 d` uniformly over the 8 nonzero (a,b)
     (rejection sampling), `N1 f`/`N1 p` uniformly over {1,2} on one
     component, and `N2` from an 81-entry `prob_dist`. In every circuit I
     loaded (about 200), the 80 nonidentity entries are equal; when only a
     scalar `prob` is present, `ensure_dem_noise_params` writes the uniform
     law. Responses are applied linearly (`apply_location_response`).
   - `circuit_generator.py` puts measurement noise before each ancilla `M`
     (depolarizing p, shift 5p, idle 2p on all other wires), reset shifts
     after `RESET`, 0.1p after one-qutrit gates and p after two-qutrit
     gates, as in the brief. The m=2 circuit has exactly 3 ancilla
     measurement rounds; the 5 records per ancilla in sdim's result list
     are 3 measurements plus 2 resets.
   - `postselect_uniform_interval` and `postselect_constant_number` select
     exactly the rounds A2's `used_rounds` returns.
   - The legacy metric is the mean over sequences of per-circuit
     |mean ω^K| (`extract_statistics` sets 0 for an empty circuit;
     `lrb_counts_to_statistics` gives NaN).
   - Noiseless reference values [VN, `ref`]: x0+x3 = 0 and every raw
     ancilla record = 0 in all 90 circuits checked (30 sequences each at
     m = 2, 10, 22; A2 checked six). So K = L and "accept iff D_U = 0" is the
     repository's pass condition.
2. **A2's symmetry numerics are tautological** [VA]. `exact_joint_from_dem`
   builds P̂(k) as a product of real factors
   1 − p_i·3^w/(3^w−1)·1(R_iᵀk ≠ 0), which are even in k. A real, even
   characteristic function forces P(x) = P(−x) for any response matrices.
   So "max|P(x)−P(−x)| ≤ 1.4e-16" and "|F_leg − λ| ≤ 1e-16" on 98 circuits
   test nothing beyond the uniformity assertion. My direct convolution does
   not build symmetry in: it gives max relative asymmetry 3.9e-15 on two
   m=2 circuits [VN, `conv`]. The earlier critique made the same point; I
   re-derived it.
3. **Exact law cross-checks** [VN, `conv`]. Direct convolution vs A2's DFT:
   max absolute difference ≤ 7e-16 on 6 circuits (m=0 at p = 1e-3, 1e-2,
   5e-2; m=2 at p = 1e-3, 1e-2). Monte Carlo with the repository's
   `sample_detector_error_model` (2×10^6 shots, p=1e-3, m=2):
   P(D=0, L=0) = 0.751938 vs exact 0.751690 (0.8 SE); max |z| = 2.5 over
   the 40 most likely (D,L) cells.
4. **Ranking cutoff.** A2 ranks only histories with P(D) > 1e-13. The direct
   convolution resolves all 3^12 histories of two m=2 circuits (smallest
   P(D) = 3.1e-13 at p=1e-3): none ranks above D=0. The cutoff hid nothing.

---

## 2. Claim by claim

**C1. Confirmed.**
- Both proof steps re-derived [VA]: (i) Σ_c δ(c|D)π(D,c) ≤ a(D)π(D)q(D),
  with equality iff δ(·|D) is supported on the argmax; (ii) with a* the
  threshold acceptance, Σ_D π(D)(a(D)−a*(D))(q(D)−t*) ≤ 0 termwise, with
  equality iff a = 1 on {q>t*} and a = 0 on {q<t*}.
- [VN, T1] 300 random tables whose posteriors are drawn from a pool with
  argmax ties and repeated q values, 1200 (table, α) pairs: HiGHS LP optimum
  equals the sorted-q frontier to 2.2e-16. The LP vertex solutions satisfy
  (a) and (b) with 0 violations. 20,000 random randomised rules never
  exceed the frontier.
- Minor: t* is not unique when α falls in a gap of the law of q; the set of
  optimal rules does not depend on the choice.

**C2. Confirmed** [VA; VN, T2]. Concavity, monotonicity of S*, and the
endpoints ess sup q and E[q] hold on 300 tables. The error–reject identity
E(t) = ∫_[t,1](1−s)dR(s) needs R(s) = P(q<s) (left-continuous) and the
closed interval, so the atom at s = t is included; checked with atoms.
"Slope t*" is a one-sided derivative at kinks.

**C3. Needs qualification.**
- Merhav's text (fetched) gives Forney's rule as his Eq. (9), defines E1
  (not deciding correctly) and E2 (undetected error), assumes
  equiprobable messages, and states that Forney showed, using the
  Neyman–Pearson theorem, that this rule gives the best tradeoff [LIT].
  Forney's paper itself was not read (A2 says so), so the NP statement is
  second-hand.
- The equal-prior condition does not hold in LRB: P(L=0) ≈ 0.99 at
  p=1e-3. The correspondence to Theorem 1 is with the prior-weighted ratio
  P(y,x_m)/Σ_{m'≠m}P(y,x_{m'}), not Forney's likelihood ratio. With a 0.98
  prior, the likelihood-ratio form can even pick a different class from
  the MAP class [VN, T11].
- The Chow abstract (Semantic Scholar, fetched) states that an optimum
  rejection rule exists and that "the error rate can be directly evaluated
  from the reject function" [LIT]. It does not state the rule's form.
  Grandvalet et al. state "two thresholds on posterior probabilities" for
  *binary* classification [LIT]. The notes drop "binary" and "two".
- The multi-class form (reject iff max_y η_y(x) ≤ 1−c, else output the
  argmax) is Definition 1 of Charoenphakdee et al., arXiv:2010.11748,
  which I fetched [LIT]. The earlier critique cited the same source.
- Fix: cite a multi-class statement; say that Chow's and Forney's rules
  are the cost (Lagrangian) forms and that Theorem 1 adds the
  acceptance-constrained randomisation; mark Forney/NP as via Merhav; use
  the prior-weighted ratio.

**C4. Confirmed** [VA]. For a deterministic rule, condition (b) of Theorem 1
is equivalent to min over accepted q ≥ max over rejected q (positive mass);
condition (a) is 0 ∈ argmax for accepted histories.

**C5. Confirmed** [VN].
- Direct convolution (independent method): no history of any mass ranks
  above D=0 at p=1e-3, m=2 (margin q(0) − max_{D≠0} q = 4.4e-5, runner-up
  an isolated round-0 flag on ancilla 6) or p=0.01, m=2 (2.8e-3).
- All 30 sequences at p=0.01, m=2, and 30 sequences each at p = 0.02 and
  0.05 for m = 0 and 2 (A2's DFT function, validated above): D=0 is
  top-ranked in every circuit; `unif 1` is on the frontier; `unif 2`,
  `const 1`, `const 2` are below it. At p=0.02, m=2 the `const 1` gap is
  0.073.
- The p=0.01, m=2 numbers in the claim are 8-sequence averages. With all
  30: `const 1` S = 0.9104 vs S* = 0.9241 (A2: 0.9096, 0.9236); `unif 1`
  S = S* = 0.93591; S*(1) = 0.6152 (A2: 0.6090). The earlier critique got
  the same 30-sequence values. The conclusions do not change.
- The p=1e-3, m=2 values (0.9619 MAP, 0.99493 at 75.3%) reproduce.

**C6. Confirmed** [VN, `codecap`]. Integer weight enumerators N_w(D,l)
with syndromes from symplectic products (commutation relations checked),
evaluated in exact fractions at p = 1/1000, 1/100, 1/20:
- no syndrome has q ≥ q(0) for either check set, so detection is exactly
  on the frontier;
- MAP failure 5.3037e-3 (Z+X) and 1.4916e-2 (X only), ratio 2.81;
- 30 (Z+X) and 6 (X only) tied syndromes, all two-way between 0 and one
  nonzero label, none three-way. A2's tie counter could not distinguish
  two-way from three-way ties; the exact count closes that gap.

**C7. Needs qualification** (wording). The d=3 identity is right [VA; VN,
T3]. "Fails for d≥5" omits d=4: cos(2πr/4) takes the values 0, −1, 0, so
Re E[i^R] is not a function of p0. The identity also holds for d=2
(λ = 2S−1), so "This is special to d=3" in Sec. 3.2 is wrong. Fix: "holds
for d ∈ {2,3}, fails for every d ≥ 4". (Same conclusion as the earlier
critique.)

**C8. Confirmed** [VA; VN, T3 on 20,000 random distributions, errors
≤ 7e-16, bound never exceeded]. The final inequality needs λ > 0.

**C9. Confirmed** [VA]. The proof is correct: symmetric fault laws (Section
1, item 1) plus linear responses give a sum of independent symmetric terms.
Numerical evidence: see Section 1, item 2 (A2's is tautological; mine,
3.9e-15 relative asymmetry, is not). The symmetry is a property of this
noise model, not of hardware (Section 4, item 2).

**C10. Needs qualification.**
- The covariance argument is correct [VA], and a covariant MAP-threshold
  rule exists (randomising between 1 and 2 at D=0 only if
  P(1|0) = P(2|0) > P(0|0)).
- "S = (1+2F)/3" needs λ ≥ 0 for every circuit and holds for population
  values. The CSV entries are means of |mean ω^K|, biased upward. In the
  floor regime the identity fails: `const 1`, p=0.01, m ≥ 18 gives
  F ≈ 0.008, hence "S" ≈ 0.339, while the true S is 1/3.
- "|F_leg − λ| ≤ 1e-16" is tautological (Section 1, item 2). The CSV
  comparison cannot test p1 = p2 at p=1e-3, m=2: by C8,
  F_leg − λ ≤ 9.7e-6 there, below the 2.2e-5 SE. It can at p=1e-2, m=0
  (bound 1.2e-3, observed difference 4.6e-5) [VN].
- The related statement in Sec. 3.5 is false (Section 3).

**C11. Confirmed** [VN, T4]. Examples A and B reproduce exactly. My own
random-table replication gives 27.5% and maximum loss 0.106 (A2: 27.9%,
0.124; the figure is sample-dependent). Context the claim omits:
- Example A's never-correct map (c_D, c_{−D}) = (2,1) satisfies
  c(−D) = −c(D), so it is covariant. Under symmetric noise the legacy
  pathology survives covariance, through the sign of λ (F = |λ|).
- Example B needs a non-covariant acceptance (accept D, reject −D). The
  covariant version accepts both and has F_leg = λ = 0.475 < 0.55: no
  reversal.
- The random tables are not conjugation symmetric.

**C12. Confirmed** [VA]. Aligned case: z(D) = ω^ĉ(3q−1)/2 and the
triangle inequality. Loss bound: λ* ≤ F_L and C8. A2's numerical support is
vacuous: in all 2000 "high-fidelity" tables the loss was 0. My exact vertex
enumeration of the acceptance-α polytope (4 histories, α uniform in
[0.15, 1]) found 87 of 300 tables with positive loss and no violation (max
loss/bound 0.647); the aligned statement held in all 100 aligned tables at
α < 1 [VN, T5].

**C13. Needs qualification.**
- λ̂ unbiased (given N_acc ≥ 1), variance (1+λ)/2 − λ², and the Rayleigh
  floor are right [VA]. Monte Carlo at N = 12,300: mean 0.00802 vs
  0.00799, SD 0.00419 vs 0.00418 [VN, T6].
- The baseline evidence overreaches at m=14 [VN, `baseline`]:
  - CSV mean 0.0105 is +3.3 SE above the floor 0.0080 (SE of a 30-sequence
    Rayleigh mean);
  - across-sequence SD 0.0054 vs Rayleigh SD 0.0042;
  - inverting the Rice mean gives a residual |λ| ≈ 0.0074, consistent with
    continued decay from 0.059 at m=10.
- m = 18, 20, 22 are at the floor (−1.1, +0.7, +0.3 SE). Fix: "m ≥ 18 is
  pure noise floor". (Same as the earlier critique.)

**C14. Needs qualification** [VN, `baseline`]. Acceptance vs
3^-4 = 0.0123457, z with SE = CSV std/√30:

| m | acceptance | z |
|---|---|---|
| 10 | 0.0129696 | +23.4 |
| 14 | 0.0124063 | +3.45 |
| 18 | 0.0123442 | −0.08 |
| 20 | 0.0123003 | −2.93 (−2.3 with binomial SE) |
| 22 | 0.0123449 | −0.03 |

The quoted 0.012344 is the m=18 value; m=14 is not yet saturated. The
mechanism (uniform data frame gives a uniform terminal syndrome) is right
[VA]. Fix: "m ≥ 18".

**C15. Needs qualification.**
- The derivation is right for its own model [VA]: i.i.d. per-step
  (ΔD_j, l_j), decision from ΔD_j only, scalar additive residual. The toy
  constants reproduce, and a biased local rule gives λ/F_leg = cos(mθ)
  exactly (θ = −0.0347 in my toy) [VN, T7].
- "Markovian, conjugation-symmetric Pauli noise" does not give that model:
  1. LRB-D's inverse half is unchecked and checked once at the end, so the
     final accept event depends on the sum of m syndrome increments. Toy
     with a checked half, an unchecked half and one final check:
     α(m) = a^m(1+2φ^m)/3, log-residual 8.9e-2 from one exponential at a
     7% per-step syndrome rate [VN, T7].
  2. Raw detectors with measurement errors couple the per-round accept
     events through a hidden syndrome state, so even with every round
     checked α(m) is a sum of exponentials. The effect is small:
     log-residual 4.6e-6 at p_data = 0.05, p_meas = 0.02 [VN, T7].
  3. Logical Cliffords change a data error's syndrome from round to round
     (I hit this directly: in difference coordinates one fault touched
     rounds 0, 1, 2), and the logical error is a Z_3² vector conjugated by
     the Clifford prefix. For a fixed sequence the steps are not
     identical [VA].
- Fix: state the i.i.d.-increment hypothesis explicitly, or replace
  "exactly" by "up to transients and the unchecked-half correction".

**C16. Needs qualification.** All fitted numbers reproduce [VN]. But:
- The acceptance misfit is entirely the m=0 point: per-depth z = −45.6 at
  m=0 and |z| ≤ 0.8 elsewhere (p=1e-3); −76.1 and ≤ 0.9 (p=2e-3). For
  m ≥ 2 a single exponential fits within errors (χ² = 1 for 6 dof). The
  "max log-residual 2.3e-3" is a structural m=0 offset (that circuit has
  no reset layer and no Cliffords), not scatter.
- η is quoted as 1−a (0.106, 0.200); the notes define η = −ln a (0.112,
  0.223).
- r_c depends on the fit at the 10% level. Unweighted 1.82e-5; my
  weighted fit 2.03e-5 ± 0.16e-5 (χ²/dof 2.1, SE scaled); the
  repository's own weighted fit 2.03e-5 (bootstrap SD 1.1e-6). The quoted
  value is about 2 bootstrap SD low. At p=2e-3: 7.64e-5 vs 8.60e-5
  (repository 8.59e-5 ± 0.36e-5).
- The λ residuals have χ²/dof = 19/7 (p=1e-3) and 30/7 (p=2e-3) against
  across-sequence SEs.
- "Close to exponential" stands for m ≥ 2. Quote r_c with an
  uncertainty and say which fit.

**C17. Needs qualification.** S = E[q | q ≥ t] ≥ t is right [VA], and the
toy numbers reproduce. Two conditions are missing:
- exact posteriors: a model posterior gives no such guarantee (C22 applies
  instead);
- acceptance is nonzero only while t ≤ ess sup q(m). On the repository
  circuits the top history is D=0 (C5, C23 evidence), with q(0) ≈ 0.9949
  at p=1e-3 for m ≤ 6, 0.99455 at m=22, decreasing with m [VN, `chain`].
  S cannot be pinned above q(0)(m), and "arbitrarily slow" holds only on
  a finite window with t < q(0)(m_max).

**C18. Confirmed** [VA; VN, T8]. The proposition follows from pointwise
optimality (needs λ_δ > 0). 0.99^m ≥ 0.8·0.995^m on [0,22] (the curves cross
at m = 44.3). The counterexample is a pair of curves; no pair of rules with
a common acceptance profile that produces it is exhibited (the notes say
so; the claim, read literally, is about curves).

**C19. Needs qualification.**
- Verified [VA, VN, T9]: min_x (e^x−1)(e^x+2)/(2x²) = 4.018 at x = 0.891;
  3e/2 = 4.077; 3e²/8 = 2.771; N* = 4.00e7 at m* = 17 reproduces with the
  fitted A = 0.941, a = 0.89435, B = 0.99246, f = 0.9999727.
- B known is assumed. Fitting B from an m=0 run with the optimal shot split
  gives 6.98e7 at m1 = 22, a factor 1.74 [VN, T9]. (The earlier critique
  found the same factor.)
- Shot noise only is assumed. That holds for `unif 1` at p=1e-3: across-
  sequence/shot-noise SD ratio 0.8–1.2. It fails for `const 1`: 1.2–2.7 at
  p=1e-3 and up to 5.1 at p=0.01 [VN, `baseline`].
- δ comes from the unweighted fit. The weighted δ = 3.05e-5 lowers the
  offset-regime estimate by a factor 0.8. The offset formula's 3.5e7 uses
  −ln a; with the tabulated 1−a it is 3.1e7.

**C20. Confirmed** [VA]. J = Σ_D π(D) Σ_c δ(c|D)(3P(c|D)−1)/2, and each MAP
term is ≥ 0. J = 0.9428 (MAP) vs 0.7470 (`unif 1`) at p=1e-3, m=2. With 30
sequences at p=0.01, m=2: 0.4228 vs about 0.054 (A2, 8 sequences: 0.414 vs
0.054).

**C21. Needs qualification.**
- "Only when" asserts necessity, which is not proven. For example, a
  finite-memory (hidden-Markov) acceptance process has a well-defined
  asymptotic heralded rate on an enlarged state space [CONJ as to its
  usefulness]. The notes' Sec. 4.6 states the conditions as sufficient.
- As worded the conditions are not sufficient for a single per-step f:
  the per-step accept events must be independent (no hidden syndrome
  state; violated by raw detectors with measurement errors, C15), and the
  per-step channels must be identical or twirled with gate-independent
  noise [VA].
- Condition (c) is right for Pauli noise [VA]. The Levine et al. part is
  confirmed (C25).

**C22. Confirmed** [VA]. |β(δ) − β̃(δ)| ≤ Δ for every rule, and δ̃ is
optimal for the joint π(D)P̃(c|D) at the same π-acceptance by Theorem 1.
Random tables with log-normally perturbed model posteriors (3000 tables):
no violation, max regret/(2Δ) = 0.656 [VN, T10]. The threshold must be
randomised to hit α exactly under the true marginal, as stated.

**C23. Unverifiable** (correctly labelled conjecture), with much stronger
evidence than A2 gives [VN, `chain`].
- Method: cut the circuit after each round's last ancilla measurement. The
  future effect (later raw detectors and L) of all earlier faults spans a
  GF(3) subspace computed from the DEM responses; its dimension is k = 5
  at every cut (4 syndrome trits + 1 logical trit) and 1 after the last
  round. P(D=d, L=l) for any chosen d is then a product of (m+1) gathered
  243×243 matrices built by direct nonnegative convolution. Validated
  against the full exact law at m=2: max error 5e-9 of P(D) over 1177
  histories (p=1e-3), 1.2e-12 (p=0.01).
- Candidates: every single-round flag pattern, every history produced by a
  single fault, and every pair of single-ancilla flags in different rounds
  (the last set omitted at m=22).

| p | m | sequences | candidates | above D=0 | min margin q(0) − max q |
|---|---|---|---|---|---|
| 1e-3 | 4 | 30 | 73,340 | 0 | +3.35e-5 |
| 1e-2 | 4 | 10 | 24,480 | 0 | +9.84e-4 |
| 1e-3 | 6 | 1 | 3,976 | 0 | +4.43e-5 |
| 1e-3 | 10 | 3 | 23,384 | 0 | +4.43e-5 |
| 1e-3 | 22 | 1 | 9,208 | 0 | +4.43e-5 |

- The runner-up is the same kind of history at every depth (an isolated
  round-0 flag on ancilla 6), and the margin does not shrink beyond m=2.
- What remains unproven: histories outside these sets (multi-fault
  patterns spread over several rounds). Their total mass is small, so even
  a failure would change S at the `unif 1` acceptance only by about their
  mass times their q-excess [CONJ].
- Two statements in the notes are too pessimistic or wrong: exact
  verification is not "infeasible beyond m=2" for chosen histories (only
  the exhaustive ranking is), and the proposed sampled frontier with a
  decoder-estimated q cannot resolve margins of 3e-5.

**C24. Confirmed** [VN]. Reproduced: exact minus CSV is +1.2e-5 to +4.6e-5
(≤ 1.6 shot-noise SE); rejections within 1e-4. Extended: at p=0.01, m=2 with
30 sequences the four policies agree with the CSV within 1.5 SE (z = +1.43,
+1.47, −0.35, −0.03), rejections within 1e-4. The four policies use nested
subsets of the same shots, so their deviations are correlated; the 30 m=0
circuits are identical (no Cliffords), so m=0 is 30 replicates of one
circuit.

**C25. Confirmed** [LIT]. arXiv HTML of Levine et al., Fig. 3 caption: the
postselection probability is fit to an exponential decay to 0, giving an
erasure error of 4.38(3)e-3 per Clifford; the postselected survival is fit
with offset 1/2, giving a residual error of 1.01(1)e-4 per Clifford; X90
values are half of these (the abstract quotes 2.19(2)e-3 per gate).
Appendix F reports a mid-circuit-checks-only value of 1.10(2)e-4 per
Clifford, described as valid for short times because the trace does not
decay to a known offset, which supports the notes' point that conditional
rates depend on the rule. Their erasure checks are per Clifford, i.e. a
local rule (condition (a)).

**C26. Confirmed** [LIT]. Meister–Pattison–Preskill: "an estimate of the
probability of a logical failure conditioned on the error syndrome", used
for "flagging those runs that should be discarded". Bombín et al.: the
logical gap is "roughly the unsigned weight difference between inequivalent
logical corrections", and gap-based postselection suppresses the encoding
error of magic states. Caveats [VA]: thresholding the failure probability of
a non-MAP decoder's correction is not Theorem 1 (it needs the argmax
output); for qutrits a two-class gap ignores the third class.

---

## 3. Errors in the notes outside the claim list

1. **Sec. 3.5, "within covariant rules under symmetric noise ... the
   covariant MAP-threshold rule maximises both" is false.** Example A's
   never-correct map (2,1) is covariant and has F_leg = |λ| = 0.5 > 0.4.
   Correct statement [VA]: among covariant rules F_leg = |λ|, so the
   MAP-threshold rule maximises F_leg iff λ*(α) ≥ −λ_min(α), i.e.
   S* + S_min ≥ 2/3; S* ≥ 2/3 suffices. (Same as the earlier critique.)
2. **Sec. 3.2, "special to d=3"**: also true for d=2 (C7).
3. **Two definitions of η** (Sec. 4.5 text vs tables and C16).
4. **Sec. 2.2 paraphrase of Grandvalet et al.**: their statement is for
   binary classification with two thresholds (C3).
5. **Sec. 6, "exact verification is infeasible beyond m=2"**: true for an
   exhaustive ranking only (C23).
6. **Sec. 2.3/2.5 p=0.01, m=2 entries** are 8-sequence values, not
   compared with the CSV in A2 (now compared with 30 sequences, C24).

---

## 4. Missing from the notes

1. **Theorem 1 does not need Pauli noise** [VA]. It needs only the
   classical joint law of (D, K) for the known circuit, whatever produces
   it (coherent errors, leakage, non-Markovian noise). Pauli noise is needed
   to compute that law cheaply and for the decay interpretation of
   Sec. 4.6.
2. **Hardware noise is not conjugation symmetric** [VA]. For qutrit
   amplitude damping with Kraus operators √γ1|0⟩⟨1| and √γ2|1⟩⟨2|, the
   Pauli-twirl weight p(a,b) = Σ_K |Tr((X^aZ^b)†K)|²/9 is nonzero only for
   a ∈ {0, 2}: all X-type weight is on X^{−1}Z^b (lowering), none on
   X^{+1}Z^b. Then P(D,l) ≠ P(−D,−l), detection no longer gives
   F_leg = |λ|, and Im E[ω^R] ≠ 0 is a physical bias, not a convention
   error. C9, C10 and the "survival in disguise" reading are properties of
   the simulation model; on transmon qutrits they should be re-checked.
3. **An exact Bayes-optimal decoder is cheap for this circuit family**
   [VN for m ≤ 22, VA for the structure]. The contraction of C23 gives
   P(L | D) for any history at any depth with a 243-dimensional state
   (k = 5 at every cut), in about 1.5 ms per history at m=6 in numpy. So
   the MAP-threshold rule of Theorem 1 is implementable per shot, not only
   a benchmark. The notes treat it as available only for m ≤ 2.
4. **The repository's metric is a mean of per-circuit ratios** [VA].
   Remark 4 handles fixed per-circuit acceptance and the pooled β. At a
   fixed *mean* acceptance, maximising the mean of per-circuit S gives the
   KKT condition (t*_i − S*_i)/α_i = const across circuits, not a common
   threshold, because d(β*_i/α_i)/dα_i = (t*_i − S*_i)/α_i.
5. **No quantitative model-mismatch number** for C22 (for example, Δ for a
   decoder built at the wrong p, or from a Clifford-averaged DEM). It is
   computable exactly at m ≤ 2 and per history at any m with item 3.
6. **Shot-cost assumptions**: B unknown (×1.74), across-sequence variance
   (dominant for `const` policies), and no error bar on r_c (±8% from the
   weighted fit, 11% shift between fits).
7. **Depth range of the saturation/floor statements** (m ≥ 18, not m ≥ 14).

---

## 5. Sources fetched for this review

1. N. Merhav, "Error Exponents of Erasure/List Decoding Revisited via
   Moments of Distance Enumerators", arXiv:0711.2501.
   https://ar5iv.labs.arxiv.org/html/0711.2501 (Eq. (9), E1/E2,
   equiprobable messages, Neyman–Pearson sentence).
2. C. K. Chow, "On optimum recognition error and reject tradeoff", IEEE
   Trans. Inf. Theory (1970), DOI 10.1109/TIT.1970.1054406.
   https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TIT.1970.1054406
   (abstract only; it does not state the rule's form).
3. Y. Grandvalet, A. Rakotomamonjy, J. Keshet, S. Canu, "Support Vector
   Machines with a Reject Option", NIPS 21 (2008).
   https://papers.nips.cc/paper/3594-support-vector-machines-with-a-reject-option
4. G. D. Forney, "Exponential error bounds for erasure, list, and decision
   feedback schemes", IEEE Trans. Inf. Theory 14(2), 206–220 (1968), DOI
   10.1109/TIT.1968.1054129. https://api.crossref.org/works/10.1109/TIT.1968.1054129
   (metadata only).
5. N. Charoenphakdee, Z. Cui, Y. Zhang, M. Sugiyama, "Classification with
   Rejection Based on Cost-sensitive Classification", arXiv:2010.11748
   (2020). https://ar5iv.labs.arxiv.org/html/2010.11748 (Definition 1,
   Chow's rule, multi-class).
6. H. Levine, A. Haim, J. S. C. Hung, N. Alidoust, M. Kalaee, et al. (113
   authors), "Demonstrating a long-coherence dual-rail erasure qubit using
   tunable transmons", Phys. Rev. X 14, 011051 (2024), arXiv:2307.08737,
   DOI 10.1103/PhysRevX.14.011051. https://arxiv.org/abs/2307.08737 and
   https://arxiv.org/html/2307.08737 (Fig. 3 caption, Appendix F).
7. N. Meister, C. A. Pattison, J. Preskill, "Efficient soft-output decoders
   for the surface code", arXiv:2405.07433 (2024).
   https://arxiv.org/abs/2405.07433
8. H. Bombín, M. Pant, S. Roberts, K. I. Seetharam, "Fault-tolerant
   Post-Selection for Low Overhead Magic State Preparation", PRX Quantum 5,
   010302 (2024), arXiv:2212.00813, DOI 10.1103/PRXQuantum.5.010302.
   https://arxiv.org/abs/2212.00813
9. H. Chen, D. Xu, G. M. Sommers, D. A. Huse, J. D. Thompson,
   S. Gopalakrishnan, "Scalable accuracy gains from postselection in quantum
   error correcting codes", arXiv:2510.05222 (2025).
   https://arxiv.org/abs/2510.05222 (abstract matches A2's paraphrase:
   failures from exponentially unlikely syndromes, p_f → p_f^b with b ≥ 2,
   b = 3.1(1) for the toric code with perfect measurements).
10. E. Magesan, J. M. Gambetta, J. Emerson, "Characterizing Quantum Gates
    via Randomized Benchmarking", arXiv:1109.6887.
    https://ar5iv.labs.arxiv.org/html/1109.6887 (Eq. (35):
    r = (d−1)(1−p)/d).
