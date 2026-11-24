# Critique of A1-syndromes ("What distance-2 correction can buy")

Adversarial review of `A1-syndromes.md` and `scripts/A1_syndrome_classification.py`,
2026-10-28. Labels follow the brief: **[VA]** verified analytically, **[VN]**
verified numerically, **[LIT]** from literature fetched in this task,
**[CONJ]** conjecture.

## Bottom line

No claim is refuted. Every number I could recompute independently came out
the same, often to all printed digits: the code algebra, the exact
code-capacity series, the gate maps, the phenomenological coefficients, the
one-round fault table, the memory windows, the 4.95p floor, the LRB
leading-order table and the two-fault coefficients.

Eight claims need qualification (C8, C10, C16, C17, C18, C19, C22, C23).
The substantive issues are:

1. **Sample size (C17, C18, C19).** The circuit-level slopes come from
   sequences 0–9, the m = 2 two-fault coefficients from sequences 0–1, while
   the baseline averages all 30. With all 30 sequences my numbers match the
   baseline almost exactly (detection 111.71p vs 111.72p per depth; const=0
   8.28p vs 8.26p). The 5–10% const=0 gap in the notes' table is a subset
   artefact that the notes do not flag. The hybrid's m = 2 p² coefficient
   ranges from 2.1·10³ to 3.0·10³ over four sequences, not the quoted
   (2.1–2.2)·10³. That makes the C19 crossover estimates uncertain by tens
   of percent.
2. **The threshold family (C8) is only three-regime at leading order, at
   code capacity.** τ = 0.5 actually gives the hybrid. There is a fourth
   operating point between about 1/3 and 1/2, and the 0.99 boundary fails at
   q = 0.05. At circuit level the posteriors of the signatures the strict
   hybrid rejects are broadly distributed. For example, about 29% of the
   m = 22 rejection comes from signatures with majority posterior ≥ 0.95.
3. **Smaller wording errors.** The general tie-pair rule (C10) is wrong
   when one partner has zero rate. The V1 hybrid's first added round costs
   20.575p, not 11.275p (C16). The Bombín et al. "logical gap" is a weight
   gap between inequivalent corrections, not "the general form" of a
   posterior threshold (C22). The 3/4 argument assumes uniform Cliffords,
   but the catalog is not uniform (C23). The fidelity comparison in C18 is
   not performed anywhere in A1's script.

## What I ran

All runs used `module load python/3.11` on the login node with one or two
threads. Nothing in the worktree outside the notes folder was written; the
re-run's JSON went to the session scratchpad. The baseline run folder was
only read.

- **Re-run of A1's script.**
  `A1_syndrome_classification.py --dem-check --out <my scratchpad>` took
  42 s. The output went to the scratchpad so that `A1_results.json` was not
  overwritten. The new log matches the stored `A1_syndrome_classification.log`
  line for line except for timings, so the stored log is current and the
  script is deterministic.
- **Independent scripts.** None imports or copies A1's code. Each `.py` has
  a matching `_output.txt`.

| script (in `scripts/`) | what it does | time |
|---|---|---|
| `critic_A1-syndromes_r2_code.py` | 9-qutrit statevector simulation with the explicit sdim matrices (`sdim/unitary.py`) of the encoder, the noiseless check round and both terminal readouts; own enumeration of all 9⁵ Paulis; own exact series; threshold levels; gate maps; Clifford catalog at the physical and the logical level | 5 s |
| `critic_A1-syndromes_r2_pheno.py` | own single-fault and fault-pair enumeration of the phenomenological model, settings F1–F4 | 2 s |
| `critic_A1-syndromes_r2_circuit.py` | own Pauli-frame propagator over sdim circuit objects, validated against the repository `CompactDetectorErrorModel`; memory windows and one check round built with the repository's `LRBCircuitGenerator`; all 30 sequences × 9 depths for LRB and const=0; two-fault enumeration; comparison with baseline CSVs at p_idx 2–6 | 69 s |
| `critic_A1-syndromes_r2_lopsided.py` | posterior structure of the signatures the strict hybrid rejects (circuit level) | 6 s |
| `critic_A1-syndromes_r2_order2v2.py` | two-fault coefficients of memory windows, V2 view, R = 0..3 | 12 s |

The propagator matched the repository DEM on 5568 unit components with 0
mismatches. The components came from 6 baseline files (four LRB, two
const=0) and 2 repository-generated memory windows **[VN]**.

The scripts directory also contains `critic_A1-syndromes_{code,circuit,pheno}*`
files without `_r2`. They come from an earlier, interrupted run of this
review (11:49–12:08, no critique written). I did not use them as evidence.
One of their outputs suggested the posterior-structure question below, which
I then answered with my own script. Their 30-sequence LRB numbers coincide
with mine.

## Claim by claim

### C1: stabilizers, logicals, ranks, encoder (confirmed)

- **Algebra [VN].** My own symplectic code gives the same commutation
  table and rank(S) = 4, rank(S, X_L, Z_L) = 6.
- **Encoder [VN].** I simulated the encoder as a statevector with the sdim
  gate matrices: X|j⟩ = |j+1⟩, Z = diag(ω^j), H[m,n] = ω^{mn}/√3,
  P = diag(1,1,ω), CNOT|a,b⟩ = |a,a+b⟩. On the encoded state
  ⟨S_Z1⟩ = ⟨S_Z2⟩ = ⟨S_X1⟩ = ⟨S_X2⟩ = ⟨X_L⟩ = 1 and ⟨Z_L⟩ = 0.
- **This is stronger than A1's check.** A1's check is a span equality.
  The statevector also fixes all eigenvalues to +1, so every noiseless
  detector reads 0.

### C2: syndrome functionals, observable, const=0 detectors (confirmed)

- **Method [VN].** I inserted each unit Pauli X_j or Z_j after the encoder.
  I then ran the actual noiseless Z-ALT and X check circuits and the two
  terminal readouts as a statevector, and read the deterministic outcomes.
- **Ancilla shifts.** The shifts are:

  | error | (σ_Z1, σ_Z2, σ_X1, σ_X2) |
  |---|---|
  | X0 | (1,0,0,0) |
  | X1 | (2,1,0,0) |
  | X2 | (0,1,0,0) |
  | X3 | (2,0,0,0) |
  | X4 | (0,2,0,0) |
  | Z0 | (0,0,2,0) |
  | Z1 | (0,0,2,1) |
  | Z2 | (0,0,1,0) |
  | Z3 | (0,0,0,2) |
  | Z4 | (0,0,0,1) |

  These are exactly ⟨S,E⟩ with the stated signs.
- **Readouts.** The observable shift is z0 + z3. The const=0 detectors
  equal −σ_X1 and −σ_X2.
- **Independence.** This check uses quantum mechanics directly, not the
  frame rules A1 used. I also re-read `sdim/program.py`; the frame rules
  quoted in section 1 of the notes are the ones implemented.

### C3: weight-2 logicals and stabilizer weights (confirmed)

- **Enumeration [VN].** It reproduces the 8 weight-2 logicals, the mixed
  cosets with minimum weight 3 and 6 representatives each, and the
  stabilizer weight distribution {0: 1, 3: 8, 4: 24, 5: 48}.
- **CSS argument [VA].** The weight-2 X supports {0,3} and {2,4} meet the
  weight-2 Z supports {0,2} and {3,4} in exactly one qutrit each. The
  argument needs the list of weight-2 X- and Z-type logicals, which the
  enumeration supplies.

### C4: classification of the 81 syndromes (confirmed, wording)

- **Counts [VN].** T 1, U1 24, A1 8 and A2\* 48. The leading posteriors
  are 1/3 (28 syndromes), 2/9 (16) and 1/4 (4).
- **Wording.** "At least 2 cosets tie at weight 2" means that two or more
  cosets attain weight 2. It does not mean the largest count is tied. In 4
  of the 48 weight-2 syndromes (1111, 2211, 1122, 2222) the largest count is
  unique (2 against 1, 1, 1, 1). That is why `hyb_count` accepts more than
  `hyb_strict`.
- **Fix.** Say "at least two cosets have weight-2 representatives".

### C5: 24/40 single errors correctable, 8 tied pairs (confirmed)

- **[VA].** Two distinct weight-1 errors share a syndrome if and only if
  their quotient is a nontrivial normalizer element of weight ≤ 2. Since no
  stabilizer has weight ≤ 2 and the distance is 2, that quotient is one of
  the 8 weight-2 logicals. Each such logical gives exactly one pair.
- **[VN].** The per-qutrit census and the pair list are reproduced.

### C6: exact code-capacity series (confirmed)

- **Series [VN].** My own exact-series code reproduces every coefficient:
  - detection: 5q − 81/8 q² + 657/64 q³, and 1/8 q² + 11/32 q³;
  - MLD: q + 63/16 q² − 141/16 q³;
  - hybrid: 2q − 3/4 q² − 45/16 q³, and 21/8 q² + 3/32 q³.
- **Numeric table.** It is reproduced at q = 10⁻³ to 0.1. For example, at
  q = 0.01: detection 0.95100 / 1.285e-5; hybrid 0.98008 / 2.626e-4;
  MLD 1.0385e-2.
- **MLD decisions.** The finite-q argmax agrees with the leading-order
  choice (A1 log; my numeric MLD uses the exact argmax).

### C7: MLD is no better than a bare qutrit (confirmed, with a remark)

- **[VN].** q + 63/16 q² against q, for the full-class (process) failure.
  This is the relevant metric under the Clifford twirl.
- **Remark [VN].** For the X-basis observable alone, MLD fails at q/2. A
  bare |+⟩ qutrit under the same depolarizing noise fails at 3q/4, the
  probability of a Z component. The comparison therefore depends on the
  figure of merit. Say so where the X-readout-only setting is discussed.

### C8: three-regime posterior-threshold family (needs qualification)

Posterior levels from my run **[VN]**:

| level (class) | q = 10⁻³ | q = 10⁻² | q = 0.05 |
|---|---|---|---|
| T | 1.0000 | 0.99999 | 0.99964 |
| U1 (lowest of the three U1 levels) | 0.99900 | 0.98992 | 0.94785 |
| A1 | 0.49984 | 0.49841 | 0.49148 |
| A2\* (highest level) | 0.33313 | 0.33124 | 0.32259 |

Problems with the stated boundaries:

- **τ = 0.5 does not give MLD.** The A1 posteriors are 1/2 − O(q) < 1/2,
  so τ = 0.5 rejects the A1 syndromes.
- **There is a fourth operating point.** For τ between the top A2\* level
  and the A1 level (0.331 < τ ≤ 0.498 at q = 0.01), the rule accepts T, U1
  and A1 and rejects all 48 weight-2 syndromes. The notes' own q = 0.01
  table shows it (P_acc 0.99938, P(f|a) 9.94e-3). Full MLD needs
  τ ≤ 0.2211.
- **The 0.99 boundary fails at q = 0.05.** The U1 levels there are 0.984,
  0.960 and 0.948, so τ between 0.948 and 0.99 does not give the hybrid.
- **The statement does not carry over to circuits [VN].** See "Missing",
  item 2: in the LRB circuits many rejected signatures have majority
  posterior ≥ 0.95.

**Fix.** "At leading order in q, at code capacity, three regimes appear,
with boundaries at 1 − O(q) and 1/2 − O(q). At finite q there is also an
A1-accepting point, and full MLD needs τ ≤ min posterior."

### C9: observable-only and X-syndrome-only variants (confirmed)

**[VN]** reproduced:

- b only: MLD q/2 + 99/32 q²; hybrid q (P(f|a) = 15/8 q²);
- X-syndrome + b: detection 15/4 q (P(f|a) = 9/16 q²); MLD 3/2 q; hybrid
  3q (P(f|a) = 9/8 q²).

**[VA]** The X-syndrome-only figures follow from simple counting. The
probability of a Z component is 3q/8 per value and per corner. The ties are
Z0/Z2⁻¹ and Z3/Z4⁻¹, and only the centre's 3q/4 is gained.

### C10: biased and restricted noise, general rule (needs qualification)

- **The individual rows are right [VN].** Pure X or pure Z: MLD 2q, hybrid
  4q. Mixed only: MLD 25/4 q². Centre only and single-qutrit-only: nothing
  fails, nothing is rejected.
- **The MLD part of the rule is right [VN].** I added an unequal-pair test
  (X0 at 2q/3, X3⁻¹ at q/3). MLD fails at min = q/3 and the strict hybrid
  rejects q.
- **The hybrid part fails when a partner has zero rate.** As written, the
  rule sums p_e + p_e′ over all 8 pairs. With noise on qutrit 0 only, it
  predicts a rejection of q/2, but the true rejection is 0: the syndrome is
  not ambiguous under that prior. The same claim lists that case as fully
  correctable, so the claim contradicts itself.
- **Fix.** Sum only over pairs with p_e > 0 and p_e′ > 0. The MLD sum of
  minima needs no change.

### C11: logical gates (confirmed)

- **[VN]** My own symplectic maps reproduce the following:
  - the M matrices and Λ;
  - K = 0 for H, H_INV, P and P_INV;
  - the permutation of the coset table;
  - the naive-prior numbers: MLD 9/4 q, hybrid P_rej 5/2 q and
    P(f|a) = q/2 + 7/4 q²;
  - the weight-1 image census. For P: 8 errors go from A1 to A1, 4 from A1
    to U1, 4 from A1 to weight 2 with a U1 syndrome, 12 from U1 to U1, 4
    from U1 to A1, and 8 from U1 to weight 2 with an A2\* syndrome.
- **[VA]** The P_L census follows by hand:
  - X0^a → X0^aZ0^a and X4^a → X4^aZ4^a (become correctable);
  - X2^a → X2^aZ3^a and X3^a → Z2^aX3^a (become weight 2);
  - X0^aZ0^{−a} → X0^a (becomes ambiguous).
- **Caveat.** The naive-prior scenario is a hypothetical decoder mismatch,
  not a property of the circuits. The notes say this.

### C12: phenomenological model (confirmed)

- **Reproduction [VN].** My enumeration, written from the model text
  rather than A1's code, gives identical coefficients for all five settings
  at κ = 1/4, 1 and 4 and R = 1–3:
  - F1: MLD (R+1)q and hybrid 2(R+1)q. The hybrid q² coefficients are
    11/2, 67/8 and 45/4, independent of κ.
  - F2 with class b: MLD R/2, hybrid R + 2κ.
  - F2 with class (a,b): MLD R, hybrid 2R + 4κ.
  - F3: undetected 3/2.
  - F4: detection R(5+4κ) + 15/4, MLD R/2 + 3/2, hybrid R + 3.
- **Gaps in the evidence.**
  - A1's exact forward recursion checks only detection and MLD. The hybrid
    coefficients rest on the enumeration alone; mine now provides a second
    one.
  - In F2 (the setting closest to the as-run readout), the hybrid's q²
    conditional failure grows steeply with κ. At R = 1 it is 23/8, 59/8
    and 383/8 for κ = 1/4, 1 and 4 **[VN]**. SI1000 has a large
    measurement flip (5p), so this matters, but the claim does not mention
    it.

### C13: the 4.95p undetected floor (confirmed)

- **Derivation [VA].**
  - Qutrit 0 idles in the M7 and M8 layers after CNOT 7→0:
    2 × 2p × 6/8 = 3p of Z component.
  - Qutrit 3 idles in the M8 layer after CNOT 8→3: 1.5p.
  - For each of the N2 after CNOT 7→0 and CNOT 8→3, a fault stays
    undetected and flips the observable if the data part has a Z component
    and the ancilla part has none: 3 × 6 = 18 of 80 Paulis, so 0.225p
    each.
  - Total: 4.95p.
- **Numerics [VN].** The DEM-validated breakdown at depth 0 is
  `idle_meas_data` 4.5 and `check_CNOT` 0.45. The undetected weight
  averages exactly 4.950 at every depth over all 30 sequences.
- **Why it is the same in every circuit [VA, using the C14 residual
  table].** No single fault outside the final round has a zero signature
  and flips b:
  - the logical tokens' two-qutrit gates act only on {2,3}, which supports
    no weight-2 logical;
  - no single check-round fault leaves a logical residual (C14 has no such
    category);
  - Z-check hooks in the final round are caught by the X checks that
    follow.

  The undetected set therefore lies entirely in the final round, which is
  identical in all circuits.
- **Minor wording.** "Data components of the N2" should read "N2 faults
  with a Z component on the data and none on the ancilla". The full data-Z
  marginal would be 1.35p.
- **Remark [VA].** No decoder can remove the floor, but the schedule can
  reduce it. If ancillas 7 and 8 were measured in one layer after both
  X-check blocks, the idle part would drop from 4.5p to 3p under the
  brief's 2p-per-measurement-layer rule.

### C14: per-round fault table (confirmed)

- **[VN]** My propagator, on a check round plus reset built with the
  repository's own `apply_si1000_noise_to_circuit`, gives identical
  numbers:
  - total 118.4p: 16.3p no effect, 41.0p measurement only, and data
    residual 30.95p + 30.15p;
  - data residual by type: 23.825p ambiguous, 35.175p correctable, 2.1p
    weight 2;
  - tied-pair rates 1.4p to 1.5125p per Pauli; sum of minima 11.725p, sum
    23.825p.
- **[VA]** I derived the measurement-only flips by hand:
  - S_Z1: 5 (flip) + 0.75 (X part of the p depolarizing) + 0.175 (CNOT
    faults that flip M5 and leave a stabilizer: 6/80 + 2/80 + 6/80)
    = 5.925p.
  - S_Z2: the same plus 1.5 for ancilla 6 idling in the M5 layer:
    7.425p.
  - S_X1: 5 + 0.75 + 3 (ancilla 7 idling in two layers before H) + 0.15
    (H and H_INV noise) + 0.175 = 9.075p.
  - S_X2: 5 + 0.75 + 4.5 + 0.15 + 0.175 = 10.575p.
  - Reset: 2p each into the next round.

### C15: hand counting within about 6% (confirmed; the label is fine)

- **Arithmetic [VA].**
  - λ_corner = 10 + 2 × 0.9 = 11.8p; λ_centre = 13.6p.
  - Ambiguous 23.6p against 23.825p exact.
  - Correctable 37.2p against 35.175p exact.
- **Source of the gap.** The 6% gap comes from the centre qutrit: 13.6p
  estimated against 11.875p exact.
- **Label.** It is a heuristic compared against a computed table. Calling
  it a conjecture is acceptable.

### C16: memory windows (needs qualification, minor)

- **Reproduction [VN].** I built the memory windows with the repository's
  `LRBCircuitGenerator` (with empty Clifford strings), not with A1's
  generator. The numbers are identical for R = 0..4, for example R = 0:
  V1 63.150 / 4.950 / 14.025 / 41.875 and V2 84.100 / 9.425 / 22.525.
- **Increments.**
  - Detection 102.1p.
  - V2: MLD 11.475p, hybrid 28.775p.
  - V1: MLD 5.6p.
- **Ambiguity is data-only [VN].** The V2 ambiguous weight of a bulk round
  is `idle_meas_data` 20, `idle_reset_data` 5 and `check_CNOT` 3.775. No
  measurement, reset or ancilla location appears.
- **The qualification.** The V1 hybrid increments are 20.575, 11.275,
  11.275, 11.275. The first added round costs 20.575p, because faults late
  in that round are seen only by the final round, which has no successor.
  The notes' body says "constant for R ≥ 1", but the claim drops this.
- **Fix.** "Bulk rounds (R ≥ 1) add …; the first added round adds 20.575p
  to the V1 hybrid rejection."

### C17: baseline LRB circuits, per-depth slopes (needs qualification)

- **Reproduction [VN].** My DEM-validated propagation reproduces A1's
  10-sequence table and fits exactly: 112.048, 10.962, 39.281; V2 14.281
  and 48.505; const=0 8.574, 1.763 and 6.972.
- **All 30 sequences [VN].** The same files and code give the following
  slopes per depth:

  | quantity | 10 sequences (A1) | 30 sequences | baseline (30 seq) |
  |---|---|---|---|
  | detection | 112.05 | 111.71 | 111.72 |
  | MLD (V1) | 10.96 | 10.79 | – |
  | hybrid (V1) | 39.28 | 38.69 | – |
  | MLD (V2) | 14.28 | 14.19 | – |
  | hybrid (V2) | 48.51 | 48.22 | – |
  | const=0 detection | 8.57 | 8.28 | 8.26 |
  | const=0 MLD | 1.76 | 1.71 | – |
  | const=0 hybrid | 6.97 | 6.74 | – |

- **The const=0 gap is a subset artefact.** The notes' table puts
  10-sequence values next to 30-sequence baseline values. In the const=0
  columns this produces a 5–10% gap that the notes do not explain: 15.9
  vs 14.4 at m = 2, and 187.7 vs 178.8 at m = 22. With 30 sequences it
  becomes 14.48 vs 14.39 and 179.30 vs 178.84.
- **The undetected 4.95p is flat [VN].**
- **Fix.** Quote the 30-sequence values, or state that the slopes come
  from a 10-sequence subset, which differs from the 30-sequence values by
  0.3–4%.

### C18: two-fault coefficients and fidelity check (needs qualification)

**Reproduced [VN]** with my own pair enumeration (exact base-3 keys,
including depletion terms):

- m = 0, V1: detection 4.95p + 55.789p², hybrid 4.95p + 396.203p².
- Memory, V2, per bulk round: detection 14.969p² for every round. The
  hybrid increments are 998.15, then 1047.34 and 1047.34, so the bulk value
  is about 1.05·10³p².
- m = 2: sequences 0 and 1 reproduce 97.24 / 119.83 and 2172.1 / 2103.1.

The qualifications:

- **The m = 2 hybrid range is too narrow.** Sequences 2 and 3 give
  detection 112.4 / 119.0 but hybrid 2602.7 / 2998.6. The hybrid
  coefficient at m = 2 is therefore at least (2.1–3.0)·10³ p², not
  (2.1–2.2)·10³. The claim omits the notes' "sequences 0, 1" qualifier.
- **The fidelity comparison is not in A1's script.** There is no code
  that reads the baseline at p_idx 3–5. I reproduced it in section 5 of my
  circuit script:
  - **m = 0.** Baseline minus prediction is +2.3e-5, +1.2e-5 and +9.6e-6
    at p = 5e-4, 1e-3 and 2e-3. The standard errors are 1.2e-5, 1.6e-5
    and 2.6e-5. This is a genuine test: without the p² term the
    prediction misses by 7.2e-5 and 3.3e-4 at 1e-3 and 2e-3.
  - **m = 2.** With c₂ between 97 and 120, the baseline lies 0.1–0.9e-5,
    2.8–6.2e-5 and −2.5e-5 to +1.1e-4 below the prediction at the three p.
    The test is weak, and it passes the 5e-5 criterion only for suitable
    c₂. With the four-sequence mean c₂ ≈ 112 it passes.

### C19: implications for MLD and hybrid LRB (needs qualification; the conjecture label is correct)

- **The arithmetic is right [VA].**
  - 1.5 × 10.96 = 16.4p, which is 2.9 × 5.65p.
  - The physical RB value checks out: the fit table gives 1 − f = 5.645p
    at p = 10⁻⁴ and 5.635p at 10⁻³.
  - The acceptances 0.91 and 0.40 (hybrid) and 0.78 and 0.08 (detection)
    at m = 22 follow from exp(−c·p).
  - The crossovers 1.2·10⁻² and 4.3·10⁻³ follow from 1.3·10³p².
- **The hybrid slope is uncertain [VN].** The per-depth p² slope comes
  from (c₂(m=2) − c₂(m=0))/2 over two sequences. Over four sequences it is
  853 to 1301. That gives 1 − f ≈ (1.3–2.0)·10³p² per depth, and moves the
  crossover with physical RB to 2.9–4.4·10⁻³ and the crossover with MLD to
  0.8–1.3·10⁻².
- **Acceptance at the crossover.** The 30-sequence hybrid rejection at
  m = 22 is 898p, so near p ≈ 4·10⁻³ the strict hybrid accepts only
  exp(−898 × 4·10⁻³) ≈ 3% of shots at m = 22. Third-order terms are not
  estimated. "Beats physical RB" there compares a decay rate while
  discarding about 97% of the data.
- **Misquoted baseline.** The notes' table quotes the baseline unif-1
  decay at p = 10⁻³ as 2.7·10⁻⁵ (27p²). The repository fit table gives
  1 − f = 3.05·10⁻⁵ (30.5p²) **[VN]**.

### C20: own generator and DEM cross-check (confirmed)

- **[VN]** The re-run gives 270/270 files reproduced, and 4 circuits with
  2428 components and 0 mismatches. My own propagator also matches the DEM
  on 5568 components.
- **Minor.** `validate_generator` recovers each file's logical-token
  sequence from the file itself. It therefore validates the following:
  - noise insertion;
  - check, reset, encoder and terminal structure;
  - annotations.

  It does not validate the catalog-to-token mapping or the construction of
  the inverse block.

### C21: the Clifford catalog (confirmed, and strengthened)

- **Physical level [VN].** 216 strings, 207 distinct strings, 206 distinct
  unitaries modulo phase (197 once, 8 twice, 1 three times). The group has
  order 216, so 10 elements are never sampled. The reversed reading of the
  strings also gives 206.
- **Logical level [VN], which removes A1's caveat.** I computed the
  logical action of the tokens on the code space by statevector, in the
  logical basis fixed by X_L and Z_L. Up to phase, H_L = H and P_L = P
  exactly, with no extra logical Pauli. The 216 strings therefore give the
  same 206 logical unitaries with the same multiplicities.
- **The 10 missing elements [VN]** are XZHSS, XZSSH, XXZ, XXZHSS, XXZSSH,
  ZZX, ZZXHSS, ZZXSSH, XXZZHSS and XXZZSSH. The list has apparent typos:
  XZHS and XZSH appear twice where XZHSS and XZSSH are expected, "XXZZ"
  and "ZZXX" appear where "XXZ" and "ZZX" are expected, and so on.
- **Non-uniform symplectic parts.** The weights over the 24 SL(2,3)
  elements are 5 (two elements), 9 (19), 10, 12 and 13, where uniform
  sampling would give 9 each.

### C22: literature (needs qualification)

**[LIT]** I fetched all six papers. Titles, authors, years and identifiers
match the notes.

| paper | what I checked |
|---|---|
| Moussa, arXiv:1603.02286 | The abstract has "two families of folded surface codes with transversal Clifford gates" and "presented generally for qudits of any dimension". |
| Gottesman, quant-ph/9802007 | Stabilizer codes generalize directly to d-dimensional systems; universal fault tolerance is proved for prime d. The abstract does not name the symplectic form, and I did not check the body. |
| Gidney et al., arXiv:2108.10457 | Table 2 (with Table 1) gives SI1000 as one-qubit Clifford p/10, two-qubit p, InitZ flip 2p, measurement flip 5p **before** measurement, idle p/10, resonator idle 2p. |
| Dennis et al., quant-ph/0110143 | Sec. IV.1 uses p for qubit errors and q for a faulty syndrome bit, with repeated rounds in space-time. |
| Combes et al., arXiv:1702.03688 | Exists as cited; the arXiv page lists no journal reference. |
| Bombín et al., arXiv:2212.00813 | The "logical gap" is "roughly the unsigned weight difference between inequivalent logical corrections". |

- **The Bombín characterization is the problem.** The logical gap is a
  weight gap between the best corrections in two classes, for a
  minimum-weight-style decoder. It is not a threshold on max_c P(c|s),
  which sums over all errors and all 9 classes. The paper does not call it
  "the general form of the threshold family"; that is the author's
  interpretation.
- **Fix.** Say "a related soft-output postselection rule".

### C23: the 3/4 flip probability (needs qualification)

- **Uniform Cliffords [VA].** The 3/4 is exact for a uniformly random
  Clifford. SL(2,3) acts transitively on the 8 nonzero vectors of Z₃², and
  6 of them have b ≠ 0.
- **The actual catalog [VN].** The catalog is not uniform (C21). A
  residual created in round 0 is conjugated by the inverse of one catalog
  element. It then flips b with probability:

  | residual | P(flip) |
  |---|---|
  | X_L type | 3/4 |
  | Z_L type | 55/72 = 0.764 |
  | X_LZ_L type | 163/216 = 0.755 |
  | X_L type, after two draws | 0.7501 |

- **The ratio check is suggestive only.** The V1/V2 MLD-slope ratio is
  0.7605 with 30 sequences (0.7676 with 10). This is not a derivation:
  the V1 MLD decides on b alone and so merges classes, and its slope need
  not equal 3/4 of the V2 slope even for uniform Cliffords.

## What the notes are missing

1. **No sampled-shot validation of the decoders.** Every MLD and hybrid
   number is perturbative: first order, plus second order for a few
   circuits. Nothing decodes sampled DEM shots at p = 10⁻³ to 5·10⁻³,
   which is where C19 places the crossovers and where higher orders
   matter.
2. **Circuit-level posteriors are lopsided [VN].**
   - Example: LRB sequence 0 at m = 22 (V1), from
     `critic_A1-syndromes_r2_lopsided_output.txt`. Of the strict-hybrid
     rejection of 884p, 260p comes from signatures with majority posterior
     ≥ 0.95, whose minority weight is only 6.25p, and another 85p from the
     band 0.8–0.95.
   - Consequence: a threshold of τ = 0.95 cuts the rejection to 624p
     (−29%) for 11.2p of first-order failure, against 4.95p for the strict
     hybrid. τ = 0.9 gives 556p and 15.8p.
   - Sequence 5 behaves the same way. In the memory windows (V2), by
     contrast, about 80% of the rejected weight is genuine 50/50 ties.
   - So the notes' "tie-breaking buys almost nothing" holds for memory but
     not for the LRB circuits, and the C8 three-regime picture should not
     be carried over to the circuits.
3. **The κ dependence of the final-round hybrid** (C12) is not discussed,
   although SI1000 measurement noise is large.
4. **Sequence sampling** (C17, C18). The notes do not report the
   per-sequence spread, and they compare a subset with the full-set
   baseline without saying so.
5. **No practical decoder specification.** Beyond leading order, the
   hybrid's rule for signatures without a single-fault explanation needs
   an enumeration of two-fault explanations. The notes describe no
   implementable finite-p rule, such as a posterior threshold under the
   DEM prior, and do not evaluate one at circuit level.
6. **The catalog's non-uniformity** (C21) is raised only as a side
   finding. Its effect on the twirl, on the single-exponential decay model
   and on the 3/4 argument (C23) is not assessed.
7. **Schedule changes.** The 4.95p floor is the only first-order failure
   term for detection and for the strict hybrid. It depends on the
   final-round measurement schedule (C13), and this design lever is not
   mentioned.
