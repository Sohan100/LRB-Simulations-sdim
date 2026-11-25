# Critique of A4-nonmarkov-theory.md (adversarial review)

Reviewer: critic agent for A4, correction-decoders / qutrit-512 study. Read with `00-brief.md` and
`A4-nonmarkov-theory.md`.

Labels for my own statements: **[VA]** verified analytically here, **[VN]** verified numerically by
a script named below, **[LIT]** from a paper I fetched in this task, **[CONJ]** conjecture.

## 0. Bottom line

- The core machinery is sound. A4's DEM extraction, character expansion and transfer-matrix
  evaluator are exact: they agree with an independent brute-force character sum to 1e-15, including
  three-round 12-detector cases. a1 = 7.425 and its 4.95 p breakdown are exact, and the 41/k slope
  structure of first-order acceptance is exact [VN].
- **Repository finding that affects every twirl argument.** The 216-string Clifford catalog in this
  worktree (`src/lrb/code_definitions.py`) is not the Clifford group. Nine strings appear twice, and
  `ZZXX` equals `XXZZ`. The catalog therefore contains only 206 distinct logical Cliffords. Ten
  Cliffords are never sampled, and the sampling distribution is not uniform. Its frame potential is
  2.031, so it is not a unitary 2-design. I decoded the 660 forward Cliffords of the baseline run's
  circuits: all of them come from this defective catalog, and 70 of them come only from it. The main
  checkout carries an uncommitted fix (9 Pauli words x 24 symplectic words, file dated 2026-10-28),
  but the baseline predates the fix [VN]. The effect on curve shapes is small. The statements that
  rely on an exact 2-design twirl (sections 2.2 to 2.4, F4) do not strictly apply to the baseline.
- **Overclaims.** Several of the strongest claims overreach their evidence:
  - C20 (correction reduces non-exponentiality) holds robustly only for sparse or terminal-only
    schedules. For unif 1 the eta difference is inside the Clifford-sampling error. For unif 3 eta
    grows, and by absolute misfit the corrected curves are 10 to 100 times less exponential.
  - C14 at p = 1e-3: the exact rise expected for these 30 sequences is 6 times smaller than the
    observed one. The quoted z = 2.9 is mostly a fluctuation.
  - C15: the T5 z-scores ignore the extrapolation error. Corrected values are 2.1, 1.5 and 3.6, not
    2.4, 2.7 and 6.6.
  - C12 and C18: the unif-1 curvature is dominated by gate dependence of the particular Clifford
    draw, not by inverse-block memory. The "40 times more data" figure rests on an unweighted fit;
    with the battery's weighting it becomes several hundred times.
  - C3: the large-m statement fails in non-generic cases (counterexamples below).
  - C6: the cited theorem does not cover mirrored inverse sequences.
  - C9: the formula's intercept is the m = 0 value, not the intercept of the m >= 2 law.

## 1. What I ran

All scripts are in `notes/scripts/`. Run with `module load python/3.11`, `OMP_NUM_THREADS=1` and
`A4_SCRATCH=<dir>`. The total runtime is about 9 minutes on one core, including A4's own exact run.

Re-runs of A4's scripts (all reproduce the quoted numbers):

- `A4_lossy_markov.py`: frame potential 2.000000000000; windowed rates 0.928261, 0.912515 and
  0.984272; r(400) = 0.064302, equal to the Perron prediction 0.064302; twirl error 1.33e-15;
  chi^2/dof 0.31 and 28.88.
- `A4_exact_lrbd.py` (sequences 0 to 5, 150 s), `A4_analyze_exact.py`, `A4_attribution.py`,
  `A4_hmm_toy.py` and `A4_fit_battery.py`: every number I checked in sections 3 to 5 reproduces.
  The CI half-width of the unif-1 d comes out as 8.1e-7 here, against 7.6e-7 in the notes, a
  bootstrap difference.

New scripts:

| script | what it checks |
|---|---|
| `critic_A4-nonmarkov-theory_exact.py` | independent re-extraction of the DEM rows from `lrb.dem_simulation`; brute-force character sums with no round maps, banding or transfer matrices; exact first-order coefficients by single-fault enumeration; c2(m); shape of hybrid against detection for all ten policies; attribution on sequences 3 to 5; exact const 4 for all 30 sequences; eta jackknife |
| `critic_A4-nonmarkov-theory_data.py` | baseline counts: p = 0 sanity check, const 0, the const 4 paired rise against exact, T5 with extrapolation error, saturation, Rice bias, unif-1 power, sawtooth e/p^2 |
| `critic_A4-nonmarkov-theory_toys.py` | catalog 2-design test with sdim matrices; counterexamples to C3; exact enumeration over all 216^m sequences (m = 1, 2) for C5; Monte Carlo with independent sequences per depth; mirrored against standard RB under gate-dependent noise (C6) |
| `critic_A4-nonmarkov-theory_catalog.py` | the catalog's logical action on the [[5,1,2]]_3 code from the repository's own gate sequences and sdim matrices, with convention checks; provenance of the baseline sequences |

Outputs are saved next to the scripts as `critic_A4-nonmarkov-theory_{exact,data,toys,catalog}_output.txt`
and `critic_A4-nonmarkov-theory_A4_reruns_output.txt` (the re-runs of A4's own scripts). The file
`critic_A4-nonmarkov-theory_checks.py` already existed. It comes from an earlier critic run; I did
not use or change it.

## 2. Verdicts

| claim | verdict | one-line reason |
|---|---|---|
| C1 | confirmed | modelling statement; the reset faults are purely transient (checked) |
| C2 | confirmed | check circuits are deterministic on all logical basis states; tokens are code automorphisms |
| C3 | needs_qualification | Jordan-block, periodic and unreachable-Perron counterexamples |
| C4 | confirmed | exact algebra |
| C5 | needs_qualification | exact for a uniform 2-design group; not exact for the repository catalog |
| C6 | needs_qualification | Wallman's Theorem 4 needs one inversion gate and small delta_2; LRB-D is mirrored |
| C7 | confirmed | brute-force agreement 1e-15; validation reproduces |
| C8 | confirmed | exact enumeration: 7.425 = 1.5 x 4.95 everywhere |
| C9 | needs_qualification | slopes exact; intercept wrong (63.2 is the m = 0 value); unif-1 log-convexity marginal |
| C10 | needs_qualification | holds for k <= 3 and c <= 3; large boundary-linear terms; k = 4, 6 and c = 4 deviate |
| C11 | confirmed | a2/m^2 = 2.07 to 2.30; data 1.99 to 2.27e-6 |
| C12 | needs_qualification | unif-1 m^2 is dominated by gate dependence; the 0.05 to 0.17 range conflates two things |
| C13 | confirmed | e/p^2 = 126, 132, 92, 128 across p; matches the exact jump |
| C14 | needs_qualification | the p = 1e-3 data z is mostly noise (exact rise 2.0e-5 against observed 1.16e-4) |
| C15 | needs_qualification | the z-scores omit the extrapolation error (2.1, 1.5, 3.6); the effect is real |
| C16 | confirmed | exact and data agree; crossover confirmed by an exact computation |
| C17 | confirmed | reproduces; no script for it in A4's set |
| C18 | needs_qualification | conclusion holds more strongly; the numbers depend on the fit and on gate dependence |
| C19 | needs_qualification | analytic core right; the quoted numbers are for a hybrid that still postselects |
| C20 | needs_qualification | partly contradicted: policy- and metric-dependent; the unif-1 case is not significant |
| C21 | needs_qualification | toy only (identity gates, i.i.d. noise) |
| C22 | unverifiable | properly labelled conjecture |
| C23 | unverifiable | properly labelled conjecture; needs fault-tolerant extraction at d >= 3 |
| C24 | unverifiable | conjecture; the Wood-Gambetta forms are quoted correctly |
| C25 | needs_qualification | the label VA is too strong; the comparison also changes Markovian noise content |
| C26 | needs_qualification | true for the group; the protocol's catalog is not that group |
| C27 | confirmed | matches the paper (minor wording) |
| C28 | confirmed | matches the abstract; it is an engineered existence result |

## 3. Claim-by-claim review

**C1 (physical Markovianity; memory only from coarse-graining).** [VA] It holds by construction:
independent Weyl channels at fixed locations, ideal preparation and ideal readout. My per-category
enumeration [VN, `exact.py first`] shows that ancilla reset faults (2p X-shift) contribute only
transient detections, 8.000 p per round, with no persistent data component, as E2 states. One
nuance: Figueroa-Romero et al. 2022 remark that gate dependence "can be regarded as a form of
non-Markovianity" [LIT]. A4's Definition M allows gate-dependent maps, so this is a matter of
definition, not an error. Verdict: confirmed.

**C2 (acceptance independent of the logical state).** [VN, `catalog.py`] With sdim's matrices and
the repository's circuits I checked four things:
- the encoded |+>_L has <S> = 1 for all four stabilisers and <X_L> = 1;
- the code space has dimension 3;
- the Z-check and X-check circuits give ancilla outcome 0 with probability 1 on |0_L>, |1_L> and
  |2_L>;
- every logical token preserves the code space (deviation below 5e-15).

The p = 0 baseline data show no rejection for any policy [VN, `data.py sanity`]. Under Weyl
frames, the outcomes are therefore commutation phases plus shifts and do not depend on the logical
state [VA]. The claim needs stochastic Pauli noise. A4 itself notes in section 6 that coherent
errors break it. Verdict: confirmed.

**C3 (large-m limit of the normalised expectation).** The first sentence (a ratio of two exponential
polynomials) is correct [VA]. The large-m sentence fails for non-generic CP trace-non-increasing maps
[VN, `toys.py limits`]:

- (i) Jordan block at the Perron eigenvalue. T = [[0.6, 0.4, 0], [0, 0.6, 0], [0, 0, 0.3]] has column
  sums at most 1, and with rho = |1><1| and O = |1><1| the Perron term enters O. Yet r(m) =
  0.6/(0.4 m + 0.6), and m r(m) tends to 1.4978. This decays like 1/m: neither constant nor
  exponential.
- (ii) Peripheral spectrum 0.9 omega^k (0.9 times a cyclic permutation, with state-independent
  acceptance). r(m) cycles through 0, -1, 1 and has no limit.
- (iii) If rho has no overlap with the left Perron vector, r(m) tends to a constant set by the
  dominant eigenvalue that rho reaches (-0.8 in the example), not by lambda_1.

Fix: add the assumptions that E is diagonalisable on the relevant block, has a unique peripheral
eigenvalue, and that rho overlaps the left Perron vector. Define lambda_1 as the dominant eigenvalue
present in D(m). The numerical example is correct.

**C4 (exact single exponential for E^dag(I) = sI and E^dag(O) = alpha O).** [VA] tr[O E^m rho] =
alpha^m tr[O rho] and tr[E^m rho] = s^m tr[rho]. For the twirled map, Dep^dag(X) = tr(X) I/d, so
E_tw^dag(I) = (alpha + beta) I and E_tw^dag(O) = alpha O for traceless O. Strictly, r(m) =
(alpha/s)^m r(0). The prefactor is 1 for O = X^dag and rho = |+><+|. Verdict: confirmed.

**C5 (ratio of averages exact; mean of ratios exponential only for state-independent acceptance).**
[VN, `toys.py ratio`] I averaged exactly over all 216 and 216^2 sequences of the true Clifford group
with a new random lossy map:
- the ratio of averages equals f^m to 1e-15, with both state-independent and state-dependent
  acceptance;
- the mean of per-sequence ratios has log-curvature ln r0 - 2 ln r1 + ln r2 = 4e-16 for
  state-independent acceptance and -6.2e-3 for state-dependent acceptance. The latter cannot be
  b f^m for any b and f;
- a Monte Carlo with fresh sequences at every depth (A4's run reused the same sequences across m,
  so its chi^2 is not calibrated) gives chi^2/dof = 31.4.

Qualifications:
- (a) The identity requires uniform sampling from a unitary 2-design group with ideal single
  inversion. With the repository's catalog, even state-independent acceptance gives log-curvature
  +4.4e-4 and a 4e-4 deviation of the ratio of averages from f^m for this non-Pauli map.
- (b) The toy's statistic is |mean N/D|. The pipeline averages per-sequence moduli.
- (c) By C2, memoryless LRB-D always has state-independent acceptance. The second case therefore
  matters only through hidden memory.

**C6 (Wallman makes per-sequence normalisation harmless).** The reduction N/D = tr[O E~...E~ rho]
with E~_g = E_g/s_g is right [VA]. I fetched Wallman's Theorem 4 [LIT]. It assumes a unitary
2-design group, gates G_1..G_m drawn independently and uniformly, and a single inversion gate
G_{m+1} = (G_{m:1})^dag. It gives A p^m + B t^m + eps_m with |eps_m| <= delta_1 delta_2^m, which is
informative only when delta_2 is small ("delta_2 <~ 1/2" in the paper). LRB-D satisfies neither the
inversion structure (m mirrored noisy inverses) nor, as run, the group or uniformity assumptions.
So the theorem does not cover the case to which A4 applies it in section 2.4.

My exact mirrored-sequence toy [VN, `toys.py mirror`] suggests the conclusion survives in practice.
With gate-dependent depolarising noise the mirrored average is one exponential to 2e-14. With
gate-dependent coherent over-rotations it is one exponential to 2e-6 for m >= 4, and the catalog
distribution changes almost nothing. However, the mirrored decay constant is not the square of the
standard-RB constant: 0.634 against 0.893^2 = 0.797 for coherent noise. Fix: cite a mirror-RB
argument or state the extension as conjecture, and state the small-delta_2 condition.

**C7 (exact evaluator).** [VN] My independent extraction of fault rows from
`build_compact_detector_error_model` is identical to A4's cache for 246, 2418 and 860 locations.
Every location has probability p x prob_scale, the only channels are d and f, and no detector term is
duplicated. Brute-force sums over all dual vectors match `Prepared.evaluate` to 1.8e-15 (const 0, 1
and 2, all depths, sequences 0 to 2, p from 1e-4 to 2e-2) and to 3e-16 to 7e-16 for unif 1 at m = 2,
unif 2 at m = 4 and unif 3 at m = 6 (12 detectors). The validation z-scores reproduce exactly,
unif-1-only included: -0.22/1.01 and -0.11/1.14.

Caveats:
- the baseline was itself sampled from the same repository DEM, so this validates the evaluator
  against DEM sampling, not the DEM against the circuit;
- "every circuit" means the 54 circuits evaluated (6 of 30 sequences) plus my 60 const-4 circuits;
- cells of different policies share shots.

Verdict: confirmed.

**C8 (a1 = 7.425 = 1.5 x 4.95).** [VN, `exact.py first`] Single-fault enumeration gives a1 =
7.425000 for every sequence, depth and policy. The contributors are:
- N2 after CNOT 7->0: 0.3375;
- 2p idle on wire 0 in the M7 layer: 2.25;
- N2 after CNOT 8->3: 0.3375;
- 2p idle on wire 0 in the M8 layer: 2.25;
- 2p idle on wire 3 in the M8 layer: 2.25.

This is exactly the hand count. For const 0, a1 is zero analytically: every single Z error is caught
by the ideal terminal X check. Verdict: confirmed. The term exists only because the readout is ideal
and the final round is a partial check.

**C9 (-ln A = 63.2 p + 71.1 p m + 41.0 p n_used + O(p^2), log-convex).** [VN] The slopes are exact:
112.0693, 91.5693, 84.7359, 81.3193, 77.9026 and 71.0693, so slope minus 71.0693 = 41.0000/k. The
41 per used round decomposes exactly as:
- 20.0 from measurement shifts;
- 8.0 from reset shifts;
- 3.0 from measured-ancilla depolarisation;
- 9.0 from ancilla idle;
- 0.3 from ancilla H gates;
- 0.7 from CNOTs.

The intercept is wrong, however. 63.15 is c1(m = 0). The m >= 2 law has intercept 69.17 (unif 2 and
const 1) and 61.17 for unif 1, because round 0 has no preceding reset and adds only 33 p. Against the
exact c1 the formula is off by -1.6 to -4.1 for unif 1 and +3.9 to +7.0 for the other policies. The
second-order coefficient c2 is negative at every depth. Log-convexity in m needs the m^2 coefficient
of c2 to be negative:
- strong for const 1 (fit -171 - 105 m - 62.3 m^2) and const 0 (-4.6 m^2);
- marginal for unif 1 (c2 about -180 to -270 at every m; fit -197 + 2.45 m - 0.22 m^2).

Fix: state the intercept as about 69.2 - 8 [round 0 used] for m >= 1, and say that c2 < 0 alone
does not imply log-convexity.

**C10 (interval law).** [VN] beta = 22.74, 45.75, 68.26 and gamma = 26.85, 13.63, 8.80, 5.33
reproduce, and β_k is linear in k with an intercept of about 0. Qualifications:
- the per-interval law has large boundary-linear terms. A4's own no-inverse fit for const 1
  (forward region plus final round) is 13 + 112 L + 21.6 L^2, so the linear part is 20 % of the
  total at L = 22;
- the "k beta_1" law fails for k = 4 and 6: the fitted slopes are 110.7 and 116.6 against 91 and
  136, because the sawtooth and the terminal-only depths m < k dominate;
- gamma_c c is 21.2 for c = 4, not about 26.8.

The mechanism (cancelling pairs inside unchecked intervals) is consistent with the attribution
[VN]. A detectable error that is masked at an intermediate round is caught at the next used round,
so masking contributes only at third order there [VA].

**C11 (const 0 quadratic).** [VN] The exact a2(m)/m^2 for m >= 2 is 2.07, 2.10, 2.11, 2.27, 2.27,
2.30, 2.24 and 2.27, and a1 = 0. The data at p = 1e-3 give (1 - F)/m^2 between 1.99 and 2.27 (x
1e-6), with SE 0.07 to 0.28. Battery d = 2.37e-6 [2.2, 2.5]e-6. "Purely quadratic" is accurate to
about 10 %. Verdict: confirmed.

**C12 (inverse-block attribution).** [VN] The sequence 0 to 2 numbers reproduce. Other numbers do
not support the interpretation:
- on sequences 3 to 5 [`exact.py attr`], the full fit is 170 + 21.2 m + 0.280 m^2, the no-inverse
  fit is 168 + 16.2 m + 0.175 m^2 and the no-forward fit is 112 + 3.5 m + 0.082 m^2. The forward-only
  part, which is Markovian at O(p^2) because every step is checked, carries a larger m^2 term than
  the inverse block;
- the per-sequence unif-1 a2 m^2 coefficients are 0.047, -0.076, 0.187, 0.297, 0.280 and 0.261
  [`exact.py second`].

So the unif-1 m^2 term in any finite draw is dominated by gate-dependent increments of the nested
Clifford prefix, not by memory. "gamma_I about 0.05 to 0.17" conflates the inverse-block term
(0.05 to 0.08) with the total unif-1 coefficient (0.17 for 6 sequences, 0.05 for 3). The linear
split (about 4 m from the inverse block plus final round, about 2 m cross, 16 to 18 m
forward-forward) and the const-1 split (21.6 of 24.4 forward-forward) are fine.

**C13 (sawtooth).** [VN] The battery reproduces: unif 4 chi^2 falls from 75.1 to 1.7 with e =
1.32e-4. My refit gives e/p^2 = 126, 132, 92 and 128 at p = 5e-4, 1e-3, 2e-3 and 5e-3, which scales
as p^2 and agrees with the exact jump of about 135 p^2. Note that unif 3 and unif 6 keep chi^2 = 11.7
and 20.0 on 5 degrees of freedom, so a constant-e sawtooth does not fully describe them. Verdict:
confirmed.

**C14 (fidelity can rise; a rise does not witness non-Markovianity).** The logical point is valid
[VA, LIT]. Figueroa-Romero et al. 2022 derive monotonicity for Markovian ASFs whose depth-n process
extends the depth-m process, and LRB-D's m-dependent schedules violate that premise. The evidence is
weaker than stated [VN, `exact.py const4`, `data.py rise`]:
- the exact const 4 rise F(4) - F(2) over the same 30 sequences is +1.97e-5 at p = 1e-3 (6-sequence
  value +7.9e-6), with per-sequence values from -1.1e-4 to +1.1e-4;
- the data give +1.16e-4 with SE 4.0e-5. That reproduces z = 2.91, but the data exceed the exact
  expectation by 2.4 SE. At 1e-3 the expected z is about 0.5, so the observed significance is mostly
  noise;
- at 5e-3 data (+1.01e-3, z = 3.46) and exact (+9.5e-4) agree, so only 5e-3 supports the claim;
- the paired-z computation is in none of A4's scripts;
- the const-4 rise itself comes from sigma-memory pairs in the m = 2 terminal-only interval. It does
  not exhibit a rise in a memoryless model.

**C15 (m = 0 boundary).** [VN] The effect is real: the exact expected residuals, +4.20e-5 (1e-3) and
+1.70e-4 (2e-3), match the data, +4.25e-5 and +1.72e-4. The significance is overstated. T5 divides by
the m = 0 standard error only. A cluster bootstrap that refits on every replicate gives z = 2.06,
1.46 and 3.62, against the quoted 2.44, 2.66 and 6.62. At 5e-4 the observed residual (3.1e-5) is 3
times the exact one (1.0e-5).

**C16 (saturation).** [VN] The exact A(22) is 0.01235 at p = 1e-2 and 2e-2, and the data give
0.01233 to 0.01237 for p >= 1e-2 (1/81 = 0.012346). The real-part F(22) means lie within ±1.4e-3
with SE about 1e-3. The exact per-window decay rate -d ln F/dm for const 1 rises and then levels off:
0.03, 0.11, 0.27, 0.50, 0.54, 0.49, 0.47, 0.47 at 1e-2 and 0.24, 0.86, 1.06, 0.99, 0.93, 0.88, 0.86,
0.87 at 2e-2. It overshoots slightly before settling. A4 shows no output for the crossover; my check
confirms it. Verdict: confirmed.

**C17 (Rice bias).** [VN, `data.py rice`] The modulus z means are +0.65 and +0.70 (+1.05 and +1.01 in
cells with F < 5 SE). The real-part z means are +0.11 and +0.04. The modulus F(22) of const 1 is
0.0070 to 0.0094, about the Rice floor sqrt(pi/(4 n_acc)) = 0.0080. Unbiasedness of the real part
is exact given P(1|acc) = P(2|acc), which the DEM's e -> -e symmetry guarantees [VA]. A4's scripts
contain no code for these numbers, but they reproduce. Verdict: confirmed.

**C18 (unif-1 curvature undetectable).** [VN, `data.py power`] The conclusion holds and is stronger
than stated. The numbers are not robust:
- the quoted d = 1.9e-7 comes from an unweighted fit to the 6-sequence mean. With the battery's
  weighting the same exact curve gives 5.9e-8, so d/SE = 0.14 and a 3-sigma detection needs about
  460 times more data, not 40;
- per-sequence exact d values at 1e-3 range from -2.2e-7 to +3.3e-7, so the curvature of a 30-sequence
  draw is dominated by gate dependence (see C12);
- at 2e-3, d/SE is about -0.17. Raising p indeed does not help.

**C19 (correction: linear functional; first-order logical error at d = 2).** [VA] Z0 and Z2^-1 have
the same syndrome (both give phase omega against S_X1 and commute with the rest) and differ by Z0 Z2
= Z_L. The quoted numbers reproduce: hybrid rates 0.99796 and 0.99804 against detection 0.999969 and
0.999951; F(22) = 0.4756 against 0.9660 at A = 0.2048. Qualifications:
- the unif-1 numbers are for the hybrid, which still postselects the intermediate rounds
  (A(22) = 0.099), so it still has the normalisation ratio. Only the const-1 hybrid is pure
  correction;
- the decoder is a genie maximum-likelihood decoder that knows the circuit and p;
- the const-1 hybrid's rate still rises with m (eta = 0.44), so accumulation is not purely linear
  at p = 1e-3.

**C20 (correction reduces eta).** The quoted eta values reproduce [VN]. The claim does not hold as a
general statement [VN, `exact.py hybrid, misc`]:
- with a jackknife over the 6 sequences, unif 1 has eta = 0.55 ± 0.45 (detection) against
  -0.04 ± 0.95 (hybrid) at 1e-3, which is not significant;
- unif 3 goes the other way: -0.73 against +4.40 at 1e-3, and -2.34 against +4.51 at 5e-3;
- the reduction is robust only for const 1 (9.13 ± 0.73 against 0.44 ± 0.07) and const 0;
- by absolute shape, the corrected curves are much less exponential for every policy at 1e-3. The
  maximum relative misfit of the best single exponential is 2.2e-3 against 2.6e-5 for unif 1 and
  2.7e-2 to 5.5e-2 against 2e-4 to 6e-4 for unif 3, 4 and 6 and const 4. The hybrid sawtooth policies
  have windowed rates above 1, up to 1.014 per step (unif 4, m = 2 to 4);
- eta is normalised by the decay rate, which correction raises 40 to 70 times, so a smaller eta is
  largely a consequence of that normalisation. Gate dependence also enters the corrected rate at
  first order, so two-window eta values on 6 sequences are noisy.

The toy result (1.6 against 0.09) reproduces. Fix: restrict the claim to terminal-only or sparse
schedules, report absolute measures (misfit, d, N_q), and use all 30 sequences.

**C21 (decoder memory spectrum).** [VN] The toy reproduces: mu = 0.0164 and 0.0823 for correction,
0.0035 and 0.0187 for detection. The toy uses identity logical gates, i.i.d. data noise, independent
syndrome shifts, a lookup decoder with deterministic tie-breaking and a genie final decoder. On the
real circuits a memoryless decoder has to push the residual syndrome through the automorphisms A_g
between rounds. That is exactly the mismatch C22 warns about, and this toy cannot show it.

**C22, C23, C24 (conjectures).** These are correctly labelled and cannot be tested here.
- C22: Ceasura et al. construct an engineered gate- and time-independent noise model that makes
  logical RB oscillate [LIT]. The citation supports possibility, not a generic prediction. A window
  decoder stores at most 81^w syndrome histories, but residual mis-corrections can carry memory
  beyond w rounds [CONJ].
- C23: second-order logical errors at d >= 3 also require fault-tolerant syndrome extraction
  (no single-fault hook logicals).
- C24: Wood and Gambetta's forms are quoted correctly [LIT]: p1(m) = A + B lambda_1^m (Eq. 9),
  lambda_1 = 1 - L1 - L2 (Eq. 14), p0(m) = A0 + B0 lambda_1^m + C0 lambda_2^m (Eq. 15), and coherent
  leakage can cause oscillations. These forms are for unencoded RB.

**C25 (compiled recovery or checks in the inverse block measure the shared memory).** This is a
design argument built on numerical results (C10, C12), so VA is too strong.
- For const 0 the inverse block is not a separate interval; compiling it shortens one 2m interval
  to m + 1.
- The comparison also removes the m - 1 noisy inverse Cliffords' Markovian contributions to
  acceptance, and their pairs with the final round.
- The compiled recovery is not mirrored, so the decay constant changes for non-Pauli noise
  (C6 toy: 0.634 against 0.797).
- The inverse-block m^2 signal (about 0.05 to 0.08 p^2) is comparable to the gate-dependence spread
  of a 30-sequence draw (C12), so many sequences would be needed.

The idea is good; label it as a proposal [CONJ] with these caveats.

**C26 (2-design).** The mathematical statement is true [VN]: the group generated by the Fourier gate
and diag(1, 1, omega) has 216 elements and frame potential 2. The protocol does not sample this
group [VN, `catalog.py`]:
- the logical tokens act exactly as sdim's single-qutrit H, P, X and Z;
- the catalog strings give 206 distinct elements, with multiplicities 2 (eight elements) and 3
  (one element: XXZZ, ZZXX, XXZZ), and 10 elements are missing;
- the catalog is not closed, and its frame potential is 2.031121 (t = 3: 7.32);
- twirling a random Pauli channel (total error 0.05) over the catalog gives eigenvalues 0.94332,
  0.94338, 0.94412 and 0.94418 (each twice) instead of one eightfold eigenvalue, a spread of about
  1.5 % of the error rate;
- the baseline used this catalog: 590 of 660 decoded forward Cliffords occur in both catalogs, 70
  only in the defective one (for example ZZXP eight times) and none only in the corrected one.

**C27 (Figueroa-Romero et al. 2021).** [LIT] The paper confirms that the noise maps are "at least
completely positive (CP) trace non-increasing"; the ASF Eqs. 7 to 9 with composition powers; "will
generally not render an exponential decay", "unless the environment is superfluous"; N_q in Eq. 15;
the Markovianised map in Eq. 16; and the factorisation in Eq. 4. Minor: Eq. 16 uses "an arbitrary
pure state eps_n on E", which may differ at each step; it is not necessarily one fixed state.
Verdict: confirmed.

**C28 (Ceasura et al. 2022).** [LIT] The abstract matches: syndrome qubits take the role of hidden
qubits, coupled to the logical qubits via a decoder, and can make logical RB oscillate under a gate-
and time-independent noise model. The abstract describes a constructed model (an existence result)
and does not use the word "non-Markovian". Verdict: confirmed.

## 4. Missing from the notes

1. **The catalog defect** (C26). It breaks the exact-twirl statements for the baseline. Its practical
   effect is small: about 1.5 % spread in twirled error rates; mirrored-toy shapes are unchanged to
   1e-6. It should be reported, and the worktree should adopt the main checkout's corrected catalog
   before any new runs.
2. **Gate dependence in finite, nested Clifford draws.** S9 treats it as exponentially small, which
   is true only for the average over all sequences. In 6 or 30 nested sequences it dominates small
   curvature terms (C12, C18) and eta (C20). Under correction it enters at first order.
3. **Exact values for all 30 sequences.** They are cheap (the 60 const-4 circuits took 21 s).
   Comparing 6-sequence exact values with 30-sequence data mixes different Clifford draws (C14:
   7.9e-6 against 2.0e-5).
4. **Mirrored inverses.** No cited theorem covers them, and their decay constant differs from
   standard RB for non-Pauli noise (C6).
5. **No real-circuit correction every round.** Every-round conclusions rest on an identity-gate toy;
   the automorphism or mismatch issue is not simulated.
6. **Statistical calibration.**
   - T5 needs the extrapolation error.
   - The toy Monte Carlo reuses sequences across m.
   - The battery mixes a nonlinear point estimate of d with a weighted log-linear bootstrap.
   - Exact-against-data curvature comparisons use different weights.
7. **Numbers without scripts in A4's set:** the const-4 paired z (C14), the modulus against
   real-part z (C17), the unif-1-only z (C7) and the const-1 crossover (C16). All reproduce with my
   scripts.
8. **Dependence on the ideal readout and preparation.** a1 and the m = 0 offset depend on the ideal
   terminal readout and the partial final round. A noisy readout adds an m-independent first-order
   term.
9. **The reference list overstates Figueroa-Romero 2022.** It calls an increasing ASF their
   "strongest" signature. The fetched text calls deviation from exponential decay "a first
   signature", says an increase "points to non-Markovianity", and derives the monotonicity bound in
   Appendix D. I did not find "strongest".

## 5. References fetched in this task

- J. J. Wallman, "Randomized benchmarking with gate-dependent noise", Quantum 2, 47 (2018),
  arXiv:1703.09835, DOI 10.22331/q-2018-01-29-47.
  https://arxiv.org/abs/1703.09835, https://ar5iv.labs.arxiv.org/html/1703.09835
- P. Figueroa-Romero, K. Modi, T. M. Stace, M.-H. Hsieh, "Randomized benchmarking for non-Markovian
  noise", PRX Quantum 2, 040351 (2021), arXiv:2107.05403, DOI 10.1103/PRXQuantum.2.040351.
  https://arxiv.org/abs/2107.05403, https://ar5iv.labs.arxiv.org/html/2107.05403
- P. Figueroa-Romero, K. Modi, M.-H. Hsieh, "Towards a general framework of Randomized Benchmarking
  incorporating non-Markovian Noise", Quantum 6, 868 (2022), arXiv:2202.11338,
  DOI 10.22331/q-2022-12-01-868. https://arxiv.org/abs/2202.11338,
  https://ar5iv.labs.arxiv.org/html/2202.11338
- A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan, "Non-Exponential Behaviour in Logical Randomized
  Benchmarking", arXiv:2212.05488 (2022). https://arxiv.org/abs/2212.05488
- C. J. Wood, J. M. Gambetta, "Quantification and characterization of leakage errors", Phys. Rev. A
  97, 032306 (2018), arXiv:1704.03081. https://ar5iv.labs.arxiv.org/html/1704.03081

## 6. Reproduction

```
module load python/3.11
export OMP_NUM_THREADS=1 A4_SCRATCH=<dir>
S=studies/correction-decoders/qutrit/notes/scripts
python $S/A4_exact_lrbd.py --seqs 0 1 2 3 4 5            # builds the DEM cache (~2.5 min)
python $S/critic_A4-nonmarkov-theory_exact.py            # ~5 min (compiles 48 small circuits)
python $S/critic_A4-nonmarkov-theory_data.py             # ~5 s
python $S/critic_A4-nonmarkov-theory_toys.py             # ~40 s
python $S/critic_A4-nonmarkov-theory_catalog.py          # ~10 s
```
