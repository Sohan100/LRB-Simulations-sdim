# A1. Syndrome classification and what correction can buy for the folded [[5,1,2]]_3 code

Agent A1, 2026-10-28. Code capacity, phenomenological level, and a
leading-order circuit-level check against the baseline LRB-D circuits.

- Script: `scripts/A1_syndrome_classification.py` (numpy and the Python
  standard library only; 3 min 6 s wall time on a login node with two
  threads after the revision). Run: `module load python/3.11;
  OMP_NUM_THREADS=2 python scripts/A1_syndrome_classification.py
  --dem-check`.
- Outputs: `scripts/A1_results.json` (every table below, the exact
  polynomials, the complete weight-0/1/2 error lists for all 81 syndromes)
  and `scripts/A1_syndrome_classification.log` (stdout of the run).
- Revision script: `scripts/A1_sampled_validation.py` samples shots from the
  exact noise model of the baseline circuits and decodes them (outputs
  `scripts/A1_sampled_validation.json` and `.log`). It reads
  `A1_results.json`, so it runs after the main script.
- The baseline run folder is only read. Nothing outside the notes folder was
  written.
- **Revised after the critique** (`A1-syndromes-critique.md`). The circuit
  tables now use all 30 Clifford sequences. The changes are listed in the
  last section, "Critique resolution". Superseded numbers are kept there
  for reference.

Labels: **[VA]** verified analytically, **[VN]** verified numerically,
**[LIT]** from literature (papers fetched in this task, see section 12),
**[CONJ]** conjecture. The claim list with evidence is in section 11.

---

## 0. Main results

1. **Single-qutrit errors [VA][VN].** The expected classification is
   confirmed. All 8 errors on the centre qutrit 1 and the 4 mixed errors
   X^a Z^b (a, b ≠ 0) on each corner qutrit 0, 2, 3, 4 are uniquely
   decodable at leading order.
   That gives 24 of the 40 single-qutrit errors. The 16 pure errors X^a and
   Z^b on corners are ambiguous. They form 8 tied pairs, and the two members
   of each pair differ by one of the 8 weight-2 logical operators
   X0^a X3^a, X2^a X4^a, Z0^b Z2^b and Z3^b Z4^b. Every weight-2 logical is
   pure X-type or pure Z-type. The mixed cosets X_L^a Z_L^b (a, b ≠ 0) have
   minimum weight 3.
2. **Code capacity, depolarizing noise q per qutrit [VN, exact polynomials].**
   - Detection: P_rej = 5q − (81/8)q², P(fail|acc) = q²/8.
   - Full maximum-likelihood decoding (MLD): P_fail = q + (63/16)q². This is
     no better than an unencoded qutrit, whose error probability is q.
   - Hybrid (correct when one coset is the unique leading explanation,
     reject otherwise): P_rej = 2q − (3/4)q², P(fail|acc) = (21/8)q².
   - At leading order in q the posterior-threshold family has three
     regimes: detection, the hybrid, and MLD. At finite q there is also a
     point that accepts the tied weight-1 syndromes and rejects only the
     weight-2 ones; it differs from MLD only at order q². This picture holds
     at code capacity only. In the LRB circuits the posteriors of the
     signatures that the strict hybrid rejects are spread out (item 5).
3. **Measurement errors with a later round [VN].** Time-like events are
   always identified when a later round exists. In the phenomenological
   model and in the exact SI1000 circuits, no measurement or reset fault in
   a round with a successor makes a signature ambiguous. Such a fault
   therefore never causes a hybrid rejection or an MLD failure at leading
   order.
4. **Final round [VN].** In the ordinary circuits the final round has no
   successor, and only qutrits 0 and 3 are read.
   - Z-check measurement errors there are harmless.
   - X-check measurement errors are confusable with corner Z errors, and one
     member of each such pair flips the observable.
   - Idle and CNOT noise on qutrits 0 and 3 after their last check
     interaction flips the observable undetectably with probability
     **4.95 p** (circuit level). This is a depth-independent floor that no
     decoder can remove. The baseline gives 4.9–5.1 p.
   - The floor is set by the final-round schedule. Measuring ancillas 7
     and 8 in one layer lowers it to 3.45 p under the same noise rules.
5. **Circuit level, per LRB depth [VN, leading order].** The own
   propagation reproduces the repository DEM exactly and matches the
   baseline rejections. Per unit depth (one forward Clifford, one check
   round, one inverse Clifford):
   - detection rejects **112 p**;
   - the hybrid rejects **39 p**;
   - full MLD fails with probability **11.0 p**.

   This implies an MLD-corrected LRB decay 1 − f ≈ 16 p per depth. That is
   about 2.9 times worse than physical RB at p_RB = 5p, which gives 5.65 p.
   The hybrid keeps a quadratic decay, about 1.3·10³ p² per depth, and
   would beat physical RB for p ≲ 4·10⁻³ **[CONJ]**.
6. **Logical gates [VN].** H_L and P_L are code automorphisms. They map the
   syndrome linearly (4×4 matrices below) and map the cosets without a
   syndrome-dependent offset. With the correct prior (the DEM) the amount of
   ambiguity is invariant. A decoder that assumes iid depolarizing noise at
   the check, while the noise actually occurred before a P_L, silently
   accepts logical errors at first order (P(fail|acc) = q/2 + …).

---

## 1. Conventions and checks

A Pauli on the 5 data qutrits is v = (x0..x4, z0..z4) ∈ Z_3^10, meaning
X^x Z^z up to a phase. The symplectic form is ⟨A,B⟩ = z_A·x_B − x_A·z_B, so
that A B = ω^⟨A,B⟩ B A. The qudit stabilizer formalism follows Gottesman
**[LIT]**.

Stabilizers, from the repository code:

| check | operator | functional on (x, z) |
|---|---|---|
| S_Z1 | Z0 Z1⁻¹ Z3⁻¹ | σ_Z1 = x0 − x1 − x3 |
| S_Z2 | Z1 Z2 Z4⁻¹ | σ_Z2 = x1 + x2 − x4 |
| S_X1 | X0 X1 X2⁻¹ | σ_X1 = −z0 − z1 + z2 |
| S_X2 | X1⁻¹ X3 X4⁻¹ | σ_X2 = z1 − z3 + z4 |

The syndrome is σ(E) = (σ_Z1, σ_Z2, σ_X1, σ_X2) = (⟨S,E⟩) and is written
as a digit string, for example `1000`.

The logical label of E is (a, b) = (⟨Z_L,E⟩, −⟨X_L,E⟩) = (x0 + x2, z0 + z3),
with X_L = X0 X3 and Z_L = Z0 Z2. For E in the normalizer,
E ∝ X_L^a Z_L^b × stabilizer. With these representatives the full map
E → (σ, a, b) is linear and onto Z_3^6, and its kernel is the stabilizer
group. b(E) is exactly the shift of the terminal observable x0 + x3. Cosets
are written `XL^a ZL^b`.

Checks:
- **[VA][VN]** All stabilizers commute; X_L and Z_L commute with them; and
  ⟨Z_L, X_L⟩ = 1. The ranks are rank(S) = 4 and rank(S, X_L, Z_L) = 6.
- **[VN]** Propagating Z0..Z4 through the ideal encoder (H_INV 2,3,4;
  CNOT_INV 2→1; CNOT 1→0; CNOT 3→0; CNOT 4→2) gives
  ⟨Z0Z1⁻¹Z3⁻¹, Z1Z2Z4⁻¹, X0⁻¹X1⁻¹X2, X0X3, X2X4⟩. This group equals
  ⟨S, X_L⟩, so the encoder prepares |+⟩_L.
- **[VN]** The sdim frame rules were read from `sdim/program.py`: H (x,z)→(−z,x); H_INV (x,z)→(z,−x);
  P z+=x; CNOT x_t+=x_c, z_c−=z_t; CZ z_b+=x_a, z_a+=x_b; SWAP; M shift = x.
  Propagating every unit data Pauli through a noiseless check round confirms
  three conventions (`G0_convention_check`):
  - each ancilla outcome shifts by exactly ⟨S,E⟩;
  - the terminal readout (H_INV, M on 0 and 3) shifts x0 + x3 by b(E);
  - the const=0 detectors x0+x1−x2 and −x1+x3−x4 equal −σ_X1 and −σ_X2,
    the opposite sign to the ancilla-based X checks.

## 2. Logical operators and minimum-weight coset representatives (task d)

**[VA][VN]** The code is CSS, so any normalizer element factorizes as
L = L_X L_Z with L_X ∈ ker H_Z and L_Z ∈ ker H_X. The weight-2 X-type
logicals are exactly X0^aX3^a and X2^aX4^a. The weight-2 Z-type logicals
are exactly Z0^bZ2^b and Z3^bZ4^b.

A mixed logical needs a nontrivial X part, of weight 2 on {0,3} or {2,4}
or of weight ≥ 3, and a nontrivial Z part, of weight 2 on {0,2} or {3,4}
or of weight ≥ 3. Any X-type support of weight 2 and any Z-type support of
weight 2 share at most one qutrit, so the union has at least 3 elements and
a mixed logical has weight at least 3. Enumeration confirms this:

| coset | min weight | # min-weight reps | representatives |
|---|---|---|---|
| XL | 2 | 2 | X0 X3, X2 X4 |
| XL⁻¹ | 2 | 2 | X0⁻¹ X3⁻¹, X2⁻¹ X4⁻¹ |
| ZL | 2 | 2 | Z0 Z2, Z3 Z4 |
| ZL⁻¹ | 2 | 2 | Z0⁻¹ Z2⁻¹, Z3⁻¹ Z4⁻¹ |
| XL ZL | 3 | 6 | X0 X3Z3 Z4, X0Z0 Z2 X3, Z0 X2Z2 X4, X2 Z3 X4Z4, X1⁻¹Z1 X2Z2 X3Z3, X0Z0 X1Z1⁻¹ X4Z4 |
| XL⁻¹ ZL, XL ZL⁻¹, XL⁻¹ ZL⁻¹ | 3 | 6 each | (inverses / conjugates of the above, listed in the JSON) |

So there are exactly 8 weight-2 logical operators: X0X3, X0⁻¹X3⁻¹, X2X4,
X2⁻¹X4⁻¹, Z0Z2, Z0⁻¹Z2⁻¹, Z3Z4 and Z3⁻¹Z4⁻¹. The stabilizer group (81
elements) has weight distribution {0: 1, 3: 8, 4: 24, 5: 48}. In
particular there is no stabilizer of weight ≤ 2, so distinct single-qutrit
errors are never degenerate.

## 3. Syndrome classification (task a)

For each of the 81 syndromes, all weight-0, 1 and 2 errors (1 + 40 + 640)
were listed and grouped by coset. A syndrome is one of:
- **T**: the trivial syndrome;
- **U_w**: a single coset attains the minimum weight w;
- **A_w**: two or more cosets attain the minimum weight with equal counts;
- **A_w\***: two or more cosets attain the minimum weight with unequal
  counts.

The leading-order posterior of the most likely coset is (max count at
w_min)/(total count at w_min).

**[VN] Result:** T 1, U1 24, A1 8, A2\* 48. No weight-2 syndrome has a
unique coset at weight 2: at least two cosets have weight-2
representatives. In 4 of the 48 (1111, 2211, 1122, 2222) the largest
count is unique (2 against 1, 1, 1, 1). This is why `hyb_count` accepts
more than `hyb_strict`. Their leading posteriors are 1/3 (28 syndromes),
2/9 (16) and 1/4 (4). The 24 U1 syndromes and the trivial one have leading
posterior 1, and the 8 A1 syndromes have 1/2. The full 81-row table is in
the appendix; the complete error lists are in `A1_results.json`
(`B_syndromes`).

Single-qutrit syndromes **[VA]**:
- σ(X0^a) = (a,0,0,0), σ(X3^a) = (−a,0,0,0), σ(X2^a) = (0,a,0,0),
  σ(X4^a) = (0,−a,0,0), σ(X1^a) = (−a,a,0,0);
- σ(Z0^b) = (0,0,−b,0), σ(Z2^b) = (0,0,b,0), σ(Z3^b) = (0,0,0,−b),
  σ(Z4^b) = (0,0,0,b), σ(Z1^b) = (0,0,−b,b).

Single-qutrit errors by qutrit **[VA][VN]**:

| qutrit | correctable at leading order (U1) | ambiguous (A1), with the tied partner |
|---|---|---|
| 0 (corner) | X0Z0, X0Z0⁻¹, X0⁻¹Z0, X0⁻¹Z0⁻¹ | X0 ↔ X3⁻¹, X0⁻¹ ↔ X3, Z0 ↔ Z2⁻¹, Z0⁻¹ ↔ Z2 |
| 1 (centre) | all 8: X1^a, Z1^b, X1^aZ1^b | none |
| 2 (corner) | 4 mixed | X2 ↔ X4⁻¹, X2⁻¹ ↔ X4, Z2 ↔ Z0⁻¹, Z2⁻¹ ↔ Z0 |
| 3 (corner) | 4 mixed | X3 ↔ X0⁻¹, X3⁻¹ ↔ X0, Z3 ↔ Z4⁻¹, Z3⁻¹ ↔ Z4 |
| 4 (corner) | 4 mixed | X4 ↔ X2⁻¹, X4⁻¹ ↔ X2, Z4 ↔ Z3⁻¹, Z4⁻¹ ↔ Z3 |

Total: 24/40 correctable, 16/40 ambiguous. This confirms the expected
classification.

The members of a tied pair differ by a weight-2 logical, for example
X0 (X3⁻¹)⁻¹ = X0X3 = X_L. A mixed corner error X0^aZ0^b cannot tie with
another single-qutrit error, because that would need a mixed weight-2
logical. It is, however, confused at the next order with 8 weight-2 errors
in other cosets, for example X3^{−a}Z0^b, X0^aZ2^{−b} and X3^{−a}Z2^{−b},
which differ from it by X_L^a, Z_L^b and X_L^a Z_L^b. The appendix gives
the counts.

## 4. Code capacity, depolarizing noise (task b)

Model: each qutrit independently suffers each of the 8 non-identity Paulis
with probability q/8. The coset probabilities
P(s, c) = Σ_w N(s,c,w)(1−q)^{5−w}(q/8)^w were computed exactly from all
9^5 = 59049 configurations, as integer polynomials in u = q/8. The total
probability equals 1 identically.

Strategies:
- **detect**: accept iff s = 0.
- **MLD**: accept all; choose the most probable coset. The leading-order
  choice equals the exact argmax for every syndrome at
  q = 10⁻³, 10⁻², 0.05 and 0.1.
- **hybrid (strict)**: accept iff s is T or U1. A2\* syndromes are also
  rejected, because no weight-2 syndrome has a unique coset.
- **hybrid (count)**: accept iff the leading-order maximum is unique.

**[VN] Exact series** (full logical class, 9 cosets; the full polynomials
are in the JSON):

| strategy | P_rej | P(fail \| acc) |
|---|---|---|
| detect | 5q − 81/8 q² + 657/64 q³ | 1/8 q² + 11/32 q³ |
| MLD | 0 | q + 63/16 q² − 141/16 q³ |
| hybrid (strict) | 2q − 3/4 q² − 45/16 q³ | 21/8 q² + 3/32 q³ |
| hybrid (count) | 2q − 9/8 q² − 9/4 q³ | 23/8 q² + 13/32 q³ |

The leading coefficients follow from simple counting **[VA]**:
- detection rejects all 5q single errors and fails on the 8 weight-2
  logicals, 8·(q/8)² = q²/8;
- MLD fails on one member of each of the 8 tied pairs, 8·q/8 = q;
- the hybrid rejects the 16 tied errors, 16·q/8 = 2q.

| q | detect P_acc | detect P(f\|a) | hybrid P_acc | hybrid P(f\|a) | MLD P_fail |
|---|---|---|---|---|---|
| 10⁻³ | 0.99501 | 1.25e-7 | 0.99800 | 2.63e-6 | 1.004e-3 |
| 10⁻² | 0.95100 | 1.29e-5 | 0.98008 | 2.63e-4 | 1.039e-2 |
| 0.05 | 0.77406 | 3.60e-4 | 0.90221 | 6.59e-3 | 5.88e-2 |
| 0.1 | 0.59150 | 1.68e-3 | 0.80999 | 2.65e-2 | 0.131 |

**Remark on the figure of merit [VA].** MLD's failure q is the full-class
(process) failure, which is the relevant one under the Clifford twirl. For
the X-basis observable alone, MLD fails at q/2 (below), while a bare |+⟩
qutrit under the same noise fails at 3q/4, the probability of a Z
component (6 of the 8 Paulis). So the "no better than a bare qutrit"
statement depends on the figure of merit.

**Posterior-threshold family [VN].** Accept s iff max_c P(c|s) ≥ τ. At small
q the posteriors cluster into these levels:
- 1 − O(q²) for s = 0;
- 1 − O(q) for the 24 U1 syndromes, which split into levels 4 + 4 + 16
  because centre and corner errors carry different weight-2 tails;
- 1/2 − O(q) for the 8 A1 syndromes (0.49984, 0.49841 and 0.49148 at
  q = 10⁻³, 10⁻² and 0.05), so they lie just below 1/2;
- ≤ 1/3 for all A2\* syndromes, in five sublevels down to 2/9 − O(q)
  (0.2211 at q = 10⁻²).

Example at q = 0.01:

| τ (level) | # syndromes added | P_acc | P(fail\|acc) |
|---|---|---|---|
| 0.99999 | 1 (s = 0) | 0.95100 | 1.29e-5 |
| 0.9974 | 4 | 0.95582 | 2.60e-5 |
| 0.9924 | 4 | 0.96067 | 6.42e-5 |
| 0.9899 | 16 | 0.98008 | 2.63e-4 (= hybrid) |
| 0.498 | 8 (A1) | 0.99938 | 9.94e-3 |
| ≤ 0.331 | 48 (A2\*) | 1 | 1.04e-2 (= MLD) |

What the levels imply (revised after the critique):
- **At leading order in q there are three regimes.** τ above the U1
  levels (τ > 1 − O(q)) gives detection. τ between the A1 level and the
  lowest U1 level, that is 1/2 − O(q) < τ ≤ 1 − O(q), gives the hybrid.
  τ ≤ 1/2 − O(q) accepts the A1 syndromes and fails at q + O(q²) like MLD.
- **τ = 0.5 gives the hybrid, not MLD**, because the A1 posteriors lie
  just below 1/2. The first version of this note said otherwise.
- **At finite q there is a fourth point.** For τ between the highest A2\*
  level and the A1 level (0.331 < τ ≤ 0.498 at q = 10⁻²) the rule accepts
  T, U1 and A1 and rejects all 48 weight-2 syndromes: P_acc = 0.99938,
  P(f|a) = 9.94·10⁻³ at q = 10⁻² (row 0.498 of the table). It differs
  from MLD only at order q² in the rejection. Full MLD needs τ ≤ the
  lowest posterior (0.2211 at q = 10⁻²).
- **The boundaries move with q.** At q = 0.05 the U1 levels are 0.984,
  0.960 and 0.948, so τ between 0.948 and 0.99 no longer gives the hybrid.
- Fine-tuning τ between the U1 levels and 1 only trades a few U1
  syndromes.
- **Code capacity only.** This three-regime structure is a code-capacity
  statement. With the circuit prior, many signatures that the strict hybrid
  rejects have lopsided posteriors, and the threshold family has a
  continuum of useful operating points (section 9, "Threshold family with
  the circuit prior").

**Observable-only and X-syndrome-only variants [VN].**
- If only b matters, which holds after the last logical gate before an
  X-basis readout:
  - X-type ties become harmless;
  - MLD P_fail = q/2 + 99/32 q²;
  - hybrid P_rej = q + 45/16 q², P(f|a) = 15/8 q².
- If only the perfect X-syndrome and b are available (the const=0 readout):
  - detection P_rej = 15/4 q, P(f|a) = 9/16 q²;
  - MLD P_fail = 3/2 q;
  - hybrid P_rej = 3q, P(f|a) = 9/8 q².

  The hybrid then only gains the 0.75q of centre-qutrit Z components.

## 5. Biased noise and noise restricted to one qutrit type (task c)

Leading terms, exact polynomials in the JSON. Each row has total
single-qutrit error probability q per noisy qutrit.

| noise | detect P_rej, P(f\|a) | MLD P_fail | hybrid P_rej, P(f\|a) | MLD, b only | hybrid P_rej, b only |
|---|---|---|---|---|---|
| depolarizing (8 Paulis, q/8) | 5q, q²/8 | q | 2q, 21/8 q² | q/2 | q |
| pure X (X^a, q/2) | 5q, q² | 2q | 4q, 2q² | 0 | 0 |
| pure Z (Z^b, q/2) | 5q, q² | 2q | 4q, 2q² | 2q | 4q |
| pure X or pure Z (q/4 each) | 5q, q²/2 | 2q | 4q, q² | q | 2q |
| mixed only (X^aZ^b, a,b≠0, q/4) | 5q, q³/8 | 25/4 q² | 15/2 q², 3/2 q² | 5q² | 13/2 q² |
| depolarizing on centre only | q, 0 | 0 | 0, 0 | 0 | 0 |
| depolarizing on corners only | 4q, q²/8 | q | 2q, 11/8 q² | q/2 | q |
| depolarizing on one qutrit j only | q, 0 | 0 | 0, 0 | 0 | 0 |

**General leading-order rule [VA, checked by the rows above and by two
extra cases].** Let each single-qutrit error e have probability p_e. Then:
- MLD fails with Σ_pairs min(p_e, p_e′) over the 8 tied pairs (X0^a, X3^{−a}),
  (X2^a, X4^{−a}), (Z0^b, Z2^{−b}) and (Z3^b, Z4^{−b});
- the strict hybrid rejects Σ (p_e + p_e′) over the tied pairs with
  **both** p_e > 0 and p_e′ > 0 (revised). If one partner has rate zero,
  the syndrome has a single explanation under the prior and is corrected.
  The first version summed over all 8 pairs, which contradicts the
  "noise on one qutrit" row of the table;
- every other single error is corrected.

Extra cases in the script **[VN]**:
- `unequal_pair_X0_X3inv` (X0 at 2q/3, X3⁻¹ at q/3): MLD fails at q/3
  (the minimum) and the strict hybrid rejects q (the sum);
- `single_pure_X0` (X0 at rate q, partner absent): MLD failure 0 and
  hybrid rejection 0. The old rule would have predicted a rejection of q.

**Which qutrits are correctable [VA][VN].**
- The centre qutrit 1 is fully correctable whenever single-qutrit errors
  dominate the prior. No other single-qutrit error shares the syndrome of
  an error on qutrit 1.
- A corner qutrit is correctable for mixed errors and ambiguous for pure
  errors. Pure-X and pure-Z noise are the worst case: only the centre
  errors (1/5 of all single errors) survive.
- Correctability of an error is relative to the prior. If noise is confined
  to one qutrit, even a corner is fully correctable, because the decoder
  knows the partner cannot occur. If the two members of a pair have
  unequal rates, MLD picks the likelier one and fails at the smaller rate.
- For the per-round residual rates of a check round, the two rates in each
  pair are nearly equal (section 8). Tie-breaking between those pairs
  therefore buys almost nothing in memory windows. This does **not** carry
  over to the LRB circuits. There, about 28% of the weight that the strict
  hybrid rejects at m = 22 sits in signatures whose leading-order posterior
  is ≥ 0.95 (section 9, "Threshold family with the circuit prior").

## 6. Logical gates H_L and P_L (task f)

Physical implementations, taken from the code:
- H_L = SWAP(2,3); H on 0, 2, 3, 4; H_INV on 1.
- P_L = P on 0, 4; P_INV on 1; CZ(2,3).
- `_INV` tokens reverse the group order and invert each gate.

Their symplectic matrices were computed from the sdim frame rules. **[VN]**
All four tokens are symplectic and map the stabilizer group onto itself.
In the table, σ′ = σ(U E U†).

| token | U S U† | U X_L U†, U Z_L U† | syndrome map σ′ = M σ | coset map (a,b)→ |
|---|---|---|---|---|
| H | S_Z1→S_X1⁻¹, S_Z2→S_X2⁻¹, S_X1→S_Z1, S_X2→S_Z2 | X_L→Z_L, Z_L→X_L⁻¹ | σ′_Z1=σ_X1, σ′_Z2=σ_X2, σ′_X1=−σ_Z1, σ′_X2=−σ_Z2 | (−b, a) |
| H_INV | inverse of the above | X_L→Z_L⁻¹, Z_L→X_L | σ′_Z1=−σ_X1, σ′_Z2=−σ_X2, σ′_X1=σ_Z1, σ′_X2=σ_Z2 | (b, −a) |
| P | S_Z unchanged, S_X1→S_X1 S_Z1, S_X2→S_X2 S_Z2 | X_L→X_L Z_L, Z_L→Z_L | σ′_Z=σ_Z, σ′_X1=σ_X1−σ_Z1, σ′_X2=σ_X2−σ_Z2 | (a, a+b) |
| P_INV | inverse | X_L→X_L Z_L⁻¹ | σ′_X1=σ_X1+σ_Z1, σ′_X2=σ_X2+σ_Z2 | (a, b−a) |

In every case λ(UEU†) = Λλ(E) with no syndrome-dependent term (K = 0), for
the representatives X0X3 and Z0Z2.

**Does a gate change which errors are ambiguous?**
- **H_L [VN].** H_L is weight-preserving: a permutation (2↔3) composed with
  local Cliffords. It maps the 40 single-qutrit errors bijectively onto
  themselves:
  - centre stays centre;
  - corner pure X ↔ corner pure Z;
  - mixed stays mixed;
  - the 24 U1 and 16 A1 errors are preserved.
- **P_L [VN].** P_L is not weight-preserving, because CZ(2,3) maps
  X2^a → X2^a Z3^a and X3^a → Z2^a X3^a. Of the 16 A1 errors before P_L:
  - 8 stay A1;
  - 4 become correctable mixed errors, for example X0 → X0Z0;
  - 4 become weight-2 errors whose syndrome is U1 but whose coset is not
    the weight-1 one, for example X3⁻¹ → Z2⁻¹X3⁻¹, which shares a syndrome
    with X0Z0.

  Of the 24 U1 errors before P_L:
  - 12 stay U1;
  - 4 become A1, for example X0Z0⁻¹ → X0;
  - 8 become weight-2 errors on A2\* syndromes.
- **True prior versus naive prior [VN].** Let depolarizing noise act before
  the gate and decoding happen after it.
  - With the true prior, the coset-probability table after the gate is a
    permutation of the original, because the gate is a code automorphism.
    Every number of section 4 is unchanged: MLD q, hybrid 2q and 21/8 q².
  - A decoder that assumes iid depolarizing noise at the check gives, after
    P_L or P_INV: MLD P_fail = 9/4 q, and hybrid P_rej = 5/2 q with
    P(fail|acc) = **q/2** + 7/4 q². The hybrid then accepts logical errors
    at first order.
  - After H_L the naive and true priors coincide.

**Consequence [VA].** "Ambiguous" is a property of a fault distribution
relative to the code, not of the observed error pattern. In LRB about
half of the data residual created in check round r (30.95p of 61.1p,
section 8, for example idle noise in the late measurement layers and in
the reset layer) is first detected in round r+1, after passing through the
next random Clifford. A correction decoder must use the circuit's DEM
prior, which handles this automatically. A fixed code-capacity lookup table
would not.

## 7. Phenomenological model (task e)

This is the setting of Dennis et al. **[LIT]**: data errors with probability
q and syndrome-bit errors with probability q_m, with repeated measurement.

Model:
- R rounds. Before each round every data qutrit gets iid depolarizing noise
  q.
- Round r reports o_r = σ(E_r) + m_r. Each digit of m_r is shifted with
  probability q_m, uniformly to 1 or 2. Write κ = q_m/q.
- The decoder sees the detection events d_r = o_r − o_{r−1}.

Final options:
- **F1**: one more data layer, then a perfect final syndrome; the full
  class (a,b) matters.
- **F2**: ordinary LRB. Nothing after round R; only b is read.
- **F3**: F2 plus a data layer after round R.
- **F4**: const=0-like. A late data layer, then a perfect X-syndrome and b.

Every single fault and every pair of faults at distinct locations was
enumerated for R = 1..4 and κ = 1/4, 1, 4. Leading coefficients, per q:

| setting | detection P_rej | MLD P_fail | hybrid P_rej | hybrid P(fail\|acc) |
|---|---|---|---|---|
| F1, class (a,b) | (R+1)·5 + R·4κ | R+1 | 2(R+1) | 11/2, 67/8, 45/4, 113/8 q² for R=1..4 (+23/8 q² per round), independent of κ |
| F2, class b | R(5+4κ) | R/2 | R + 2κ | O(q²) |
| F2, class (a,b) | R(5+4κ) | R | 2R + 4κ | O(q²) |
| F3, class b | R(5+4κ) | R/2 + 3/2 | R + 2κ | 3/2 q (undetected late Z on qutrits 0, 3) |
| F4, class b | R(5+4κ) + 15/4 | R/2 + 3/2 | R + 3 | O(q²) |

**[VN]** An exact forward recursion over the 729 states (σ, a, b) and all
81^R observation histories reproduces these coefficients for R = 1, 2 at
q = 10⁻⁴:
- F1: MLD/q = 2.0007 and 3.0008; detection P_rej/q = 13.99 and 22.97
  (κ = 1);
- F2 with class b: MLD/q = 0.5009 and 1.0014;
- F1, R = 1: detection P(fail|acc)/q² = 0.2501.

**What the table shows [VN][VA]:**
- **Time-like events with a successor round are always correctable.**
  - A measurement error at round r < R produces (v at (r,j), −v at
    (r+1,j)) on a single digit.
  - No single data fault produces a two-round signature, so this is
    unique at leading order. The correct action is "no data correction".
  - Accordingly, in F1 the MLD failure (R+1)q and the hybrid rejection
    2(R+1)q do not depend on κ. In F1 no group containing a measurement
    error is ambiguous.
  - At second order, a measurement-error explanation can hide a logical
    pair split across a round, for example X0 before round r and X3 after
    it. This adds 1/4 q² per round to the hybrid's 21/8 q².
- **Final noisy round without successor (F2).** A measurement error there
  gives a single-digit event, which is identical to a data error just
  before the round.
  - Z-check digits: the explanations {meas, X0^a, X3^{−a}} or
    {meas, X2^a, X4^{−a}} all leave b unchanged. They are harmless for the
    X-basis readout and are accepted.
  - X-check digits: the explanations {meas (b=0), Z2^{−b} (b=0), Z0^b
    (b≠0)} and the analogue on S_X2 are ambiguous. MLD picks b = 0 and
    fails at q/8 per signature, the same as the code-capacity tie. The
    hybrid must reject q + 2q_m in the final round, including all X-check
    measurement errors there.
  - If the class still matters at the final round, as for an intermediate
    round of LRB, every single-digit event becomes a 3-way ambiguity. The
    final-round hybrid rejection is then 2q + 4q_m.
  - **The hybrid's conditional failure in F2 grows quickly with κ**
    (added in the revision). The log gives 23/8, 59/8 and 383/8 q² at
    R = 1 for κ = 1/4, 1 and 4. Section P2 of the script splits the
    pair contributions by type: data-data, data-measurement and
    measurement-measurement. The acceptance set does not depend on
    κ > 0, so the coefficient is exactly quadratic in κ **[VA]**, and
    the enumeration gives **[VN]**
    c(κ) = 15/8 + (7/2)κ + 2κ² at R = 1,
    31/8 + (15/4)κ + 2κ² at R = 2, and 47/8 + (15/4)κ + 2κ² at R = 3.
  - **Mechanism of the 2κ² term [VN].** It comes only from pairs of
    final-round measurement flips on (S_Z1, S_X1) or (S_Z1, S_X2), 4 value
    combinations each. Such a pair has the signature of the correctable
    mixed corner error X0^aZ0^b or X3^aZ3^b, whose b-class is nonzero, so
    the hybrid applies a wrong b-correction. Each pair has probability
    (q_m/2)², which gives 2 × 4 × q_m²/4 = 2κ²q² **[VA]**.
  - **Relevance.** With the SI1000 rates of section 8, κ is about 0.7–1.1
    (measurement-only flips of 5.9–10.6p, plus 2p from the reset, against
    11.8p per corner qutrit). The κ terms of the final-round hybrid
    failure are therefore not small. That they are a main contributor to
    the large circuit coefficients of section 9 is plausible but was not
    isolated **[CONJ]**.
  - **Evidence note.** The exact forward recursion below checks detection
    and MLD only. The hybrid coefficients rest on the enumeration, which
    the critique reproduced with an independent implementation.
- **Data noise after the last syndrome (F3)** is invisible. The Z
  components on qutrits 0 and 3 flip the observable undetected: 3/2 q for
  depolarizing noise. This is the phenomenological analogue of the
  circuit-level 4.95p floor (section 9).
- **const=0 (F4).** The final readout of all five data qutrits gives a
  perfect X-syndrome. X errors are irrelevant to the X-basis observable;
  Z-type ties (Z0/Z2⁻¹, Z3/Z4⁻¹) remain. With only this final syndrome, the
  hybrid rejects 3q and MLD fails at 3/2 q per late layer.

## 8. Per-check-round rates under the SI1000 noise of the brief

SI1000 is defined by Gidney, Newman, Fowler and Broughton **[LIT]** (their
Table 2):
- p/10 after 1-qubit Cliffords;
- p after 2-qubit gates;
- a 2p flip after reset;
- a 5p flip before measurement;
- p/10 idle;
- 2p idle during measurement and reset.

The repository's qutrit generalization differs in two places, read from
the code and the brief. It adds depolarizing p on the measured ancilla,
and it has no idle noise during gate layers.

**Generator and propagator validation [VN].**
- The script's own generator rebuilds all 270 baseline LRB circuit files
  (30 Clifford sequences × 9 depths) op by op, including every noise op
  with its channel and prob_scale and every detector annotation.
- The own batched Pauli-frame propagation reproduces the repository's
  `CompactDetectorErrorModel` exactly: 2428 unit components on 4 circuits
  (depth 0, 2 and 6, and a const=0 circuit), 0 mismatches.

**(i) Code-capacity-style counting [CONJ: approximation that ignores hooks
and correlations].** Per data qutrit and check round:
- idle 4 measurement layers × 2p + reset layer 2p = 10p;
- each check CNOT contributes its data marginal 72/80·p = 0.9p.

This gives λ_corner = 10 + 2·0.9 = 11.8p and λ_centre = 10 + 4·0.9 = 13.6p.
Ambiguous data errors are half of each corner's rate, 4 × 5.9 = 23.6p per
round; correctable ones are 4 × 5.9 + 13.6 = 37.2p per round.
Measurement error per check: 5p flip + 0.75p (the X part of the p
depolarizing before M), plus ancilla idle, H and CNOT contributions.

**(ii) Exact single-fault enumeration of one check round plus reset [VN].**
Each fault was propagated from its location to the end of the round.

| quantity (coefficient of p, per round) | value |
|---|---|
| total fault probability (12 N2, 32 idle N1 in measurement layers, 4×(p + 5p) on measured ancillas, 4×0.1p ancilla H, 5×2p data idle and 4×2p reset flips) | 118.4 |
| no effect | 16.3 |
| measurement-only (data residual is a stabilizer, some outcome flipped) | 41.0 |
| data residual with no flip in this round / with a flip in this round | 30.95 / 30.15 |
| measurement-only flips per check S_Z1, S_Z2, S_X1, S_X2 | 5.925, 7.425, 9.075, 10.575 (+2.0 each into the next round from reset flips) |
| any-cause flips per check in the same round | 7.775, 14.45, 22.625, 29.975 |
| data residual equivalent to a single-qutrit error, ambiguous (A1) | q0 6.05, q2 5.675, q3 6.05, q4 6.05; total **23.825** |
| data residual equivalent to a single-qutrit error, correctable (U1) | q0 5.9, q1 11.875, q2 5.6, q3 5.9, q4 5.9; total **35.175** |
| data residual of minimum weight 2 (hooks / correlated N2) | 2.1 (1.95 on A2\* syndromes, 0.15 on U1 syndromes in the wrong coset) |

Notes on the table:
- **Hooks.** An X-check hook, for example an X on ancilla 7 after
  CNOT 7→0, leaves X1 X2⁻¹ ≡ X0⁻¹ (mod S_X1), which is a pure corner X:
  ambiguous. A Z-check hook after CNOT 0→5 leaves Z1Z3 ≡ Z0 (mod S_Z1),
  also ambiguous. After CNOT 1→6 it leaves Z2⁻¹Z4 ≡ Z1 (mod S_Z2),
  correctable.
- **Measurement noise.** The measurement-only flip rates per check
  (5.9–10.6p, plus 2p from the preceding reset) are comparable to the data
  rates per qutrit (11–12p).
- **Tie-breaking.** The per-Pauli residual rates of the tied pairs are
  almost equal:
  - most pairs: 1.5125p each;
  - X4 vs X2⁻¹: 1.5125 vs 1.4375;
  - Z2 vs Z0⁻¹: 1.4 vs 1.5125.

  The code-capacity-style MLD tie failure is therefore Σ min = 11.725p
  per round, and the tie rejection Σ = 23.825p.

**Logical-gate noise per token [VN].**

| token | total | A1 | U1 | weight 2 |
|---|---|---|---|---|
| H | 1.5p | 0.3 | 0.4 | 0.8 |
| P | 1.3p | 0.2 | 0.3 | 0.8 |
| X | 0.2p | 0.1 | 0.1 | 0 |
| Z | 0.2p | 0.1 | 0.1 | 0 |
| I | 0.1p | 0.05 | 0.05 | 0 |

The catalog averages 1.653 H, 1.463 P, 1.005 X, 1.005 Z and 0.005 I tokens
per Clifford, about 4.8p of gate noise per Clifford.

**(iii) Exact space-time leading order, memory windows [VN].** These are R
noisy check rounds with resets, then the final round, with identity
Cliffords. Single faults were grouped by their full detector signature.
- V1 is the circuit as run: detectors plus the observable.
- V2 adds a perfect final syndrome and uses the full class (a,b). V2 is
  the relevant view for intermediate LRB rounds, where later random
  Cliffords mix X_L and Z_L.

Per-round increments (R → R+1) are exactly constant, with one exception:
the V1 hybrid rejection. Its first added round (R = 0 → 1) costs 20.575p,
and every later round costs 11.275p **[VN]**. The V1 ambiguous weight at
R = 3, by the check round that first follows each fault, is 9.275,
11.275, 14.575 and 49.875p for rounds 0–3. Against the bulk value 11.275p
and the R = 0 final round (41.875p), the extra 9.3p splits as follows:
- +8.0p in the final round. This is the reset layer that now precedes
  it: 4.0p of X-check-ancilla reset flips and 4.0p of data idle. Only the
  final round sees these faults, and it has no successor.
- A net +1.3p from the rounds next to the boundaries. The round before
  the final one carries 14.575p instead of 11.275p. Round 0, which has no
  reset layer before it, carries 9.275p.

| per extra check round | detection P_rej | MLD P_fail | hybrid P_rej |
|---|---|---|---|
| V2 (full class, successor exists) | 102.1p | 11.475p | 28.775p |
| V1 (observable only, memory) | 102.1p | 5.6p | 11.275p for bulk rounds (R ≥ 1); 20.575p for the first added round |

In a bulk round the ambiguous faults are only data faults: idle in the
measurement layers 20p, idle in the reset layer 5p, check CNOTs 3.775p.
Measurement flips, measurement depolarizing, reset flips and ancilla idle
never appear in an ambiguous group when a later round exists.

The final round alone (R = 0, V1) gives:
- detection 63.15p;
- undetected 4.95p;
- MLD 14.025p;
- hybrid rejection 41.875p, of which 10p are the X-check measurement flips
  and 7.5p ancilla idle.

**Schedule variant (added in the revision) [VN, section G7].** The
final round was rebuilt with ancillas 7 and 8 measured in one layer after
both X-check blocks. The noise rules are unchanged: 2p idle on every other
wire per measurement layer, and p and 5p on each measured ancilla. Then:
- undetected 3.45p (instead of 4.95p);
- detection 57.15p;
- MLD 14.025p → 11.025p;
- hybrid rejection 41.875p → 35.875p.

The arithmetic **[VA]**: qutrits 0 and 3 then idle in one layer each, 1.5p
of Z component each, plus the 0.45p from the N2 after CNOT 7→0 and
CNOT 8→3, which gives 3.45p. On hardware, a combined layer may change the
measurement error itself. That was not modelled.

## 9. What correction buys in the actual LRB circuits (leading order) [VN]

Method:
- Own propagation of the baseline circuit files, with single faults
  grouped by the full detector history.
- Class = the observable (V1, exactly what the experiment reads).
- Average over Clifford sequences 0–9.
- Coefficients of p; the "baseline" columns come from the 1M-shot
  simulations at p = 10⁻⁴.

| m | detection rej | baseline −ln(acc)/p | undetected | baseline (1−F)/(1.5p) | MLD fail | hybrid rej | const=0 det rej | baseline const=0 −ln(acc)/p | const=0 MLD | const=0 hybrid rej |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 63.2 | 63.3 | 4.95 | 4.97 | 14.0 | 41.9 | 0 | 0 | 0 | 0 |
| 2 | 285.9 | 284.1 | 4.95 | 4.91 | 36.5 | 128.8 | 15.9 | 14.4 | 3.25 | 13.0 |
| 6 | 733.4 | 729.7 | 4.95 | 4.94 | 80.3 | 283.7 | 49.6 | 46.0 | 10.2 | 40.4 |
| 10 | 1184.0 | 1176.4 | 4.95 | 4.94 | 124.8 | 443.7 | 86.0 | 79.1 | 17.6 | 70.0 |
| 14 | 1632.4 | 1624.2 | 4.95 | 5.06 | 168.6 | 600.0 | 120.4 | 113.5 | 24.7 | 98.0 |
| 22 | 2527.1 | 2518.5 | 4.95 | 5.07 | 255.5 | 915.0 | 187.7 | 178.8 | 38.6 | 152.6 |

Linear fits c0 + c1·m over m ≥ 2 (10 sequences):

| quantity | c0 | c1 (per depth) |
|---|---|---|
| detection rej | 62.1 | **112.0** (baseline, 30 sequences: 60.1 + 111.7 m) |
| MLD fail | 14.9 | **10.96** |
| hybrid rej | 50.1 | **39.3** |
| undetected | 4.95 | 0 |
| V2 MLD | 9.5 | 14.3 |
| V2 hybrid rej | 24.3 | 48.5 |
| const=0 det / MLD / hybrid rej | ≈0 | 8.57 / 1.76 / 6.97 |

The ratio V1/V2 of the MLD slopes is 10.96/14.28 = 0.77. This is close to
3/4 **[VA]**: a uniformly random Clifford maps a fixed nontrivial logical
Pauli to a uniform nontrivial one, and 6 of the 8 have a Z component that
flips the X-basis readout.

**Two-fault (p²) coefficients [VN]** come from an exact pair enumeration,
including the depletion terms of the first-order failures. The routine
reproduces the exact code-capacity 1/8 and 21/8.

| circuit | detection P(fail\|acc) | hybrid P(fail\|acc) | hybrid P_rej |
|---|---|---|---|
| memory, per bulk round (V2) | 15.0 p² | ≈ 1.0·10³ p² (998, 1047) | 28.8 p |
| LRB m = 0 (V1) | 4.95 p + 55.8 p² | 4.95 p + 396 p² | 41.9 p |
| LRB m = 2 (V1), sequences 0, 1 | 4.95 p + 97 / 120 p² | 4.95 p + 2172 / 2103 p² | 109 / 132 p |

The predicted unif-1 fidelities 1 − 1.5(4.95p + c₂p²) agree with the
baseline to within 5·10⁻⁵ at p = 5·10⁻⁴, 10⁻³ and 2·10⁻³ for m = 0 and 2
**[VN]**. For example, at p = 10⁻³ and m = 0 the prediction is 0.992491
against 0.992503 in the baseline.

**Implications.** These use 1 − f ≈ 1.5 × (per-depth observable-error
slope), which is valid at leading order **[VA]**. The crossovers are
extrapolations **[CONJ]**.

| protocol | per-depth 1 − f | acceptance at m = 22, p = 10⁻⁴ / 10⁻³ |
|---|---|---|
| detection, unif 1 | ≈ 1.5 × (20–32) p² (baseline at 10⁻³: 2.7·10⁻⁵ ≈ 27 p²) | 0.78 / 0.08 (baseline 0.777 / 0.081) |
| hybrid (reject only ambiguous signatures) | ≈ 1.5 × (0.85–0.89)·10³ p² ≈ 1.3·10³ p² | ≈ 0.91 / 0.40 |
| full MLD (accept all) | ≈ 1.5 × 10.96 p ≈ 16 p | 1 / 1 |
| physical RB at p_RB = 5p (baseline fit) | 5.65 p | 1 / 1 |

- Full correction makes the LRB decay first order in p and about 2.9 times
  faster than physical RB at p_RB = 5p. This is consistent with the
  code-capacity statement that MLD on this distance-2 code only breaks even
  with a bare qutrit, now at about 128p of fault weight per depth.
- The hybrid keeps a second-order decay while rejecting about 35% as much
  as detection per depth. Its p² coefficient is 30–40 times detection's.
  It beats MLD below p ≈ 10⁻², and physical RB below p ≈ 4·10⁻³.
- None of the three removes the m-independent 4.95p floor of the final
  round. That floor enters the LRB prefactor b, not f.

## 10. Side finding: the Clifford catalog [VN]

This is outside the A1 scope, but relevant to the twirl. The 216 catalog
strings in `code_definitions.py` (with S read as P) contain:
- 207 distinct strings. Nine strings appear twice: XZHS, XZSH, XXZZ, XXZS,
  XXZHS, ZZXS, ZZXHS, XXZZS, XXZZHS. In addition, XXZZ and ZZXX are the
  same unitary up to phase.
- **206 distinct single-qutrit unitaries modulo phase**, with
  H X H† = Z, H Z H† = X⁻¹ and P = diag(1,1,ω) (the sdim conventions). 197
  of them occur once, 8 twice and 1 three times.

H, P, X and Z generate the full group of 216 elements modulo phase
(9 × |SL(2,3)|), so **10 Clifford elements are never sampled**.

Added in the revision:
- **Logical level [VN, section G6].** A 5-qutrit statevector computation
  uses the sdim matrices and the logical basis |j_L⟩ defined by
  Z_L = ω^j on the encoded |+⟩_L. It gives ⟨S⟩ = 1 for all four
  stabilizers, ⟨X_L⟩ = 1 and ⟨Z_L⟩ = 0. The token circuits H, P, X, Z and
  their inverses act on the code space exactly as the physical
  single-qutrit gates, up to a global phase. The logical count is
  therefore also 206 of 216, with the same multiplicities. This removes
  the caveat of the first version.
- **The 10 missing elements [VN].** They are exactly the unitaries of the
  strings XZHSS, XZSSH, XXZ, XXZHSS, XXZSSH, ZZX, ZZXHSS, ZZXSSH, XXZZHSS
  and XXZZSSH, the critique's candidates. The catalog has XZHS twice
  where XZHSS would be expected, XXZZ where XXZ would be expected, and so
  on. This suggests typos in the list **[CONJ]**.
- **Symplectic part [VN, section G5].** Over the 24 elements of SL(2,3)
  the 216 strings have weights 5 (two elements), 9 (19 elements), 10, 12
  and 13. Uniform sampling would give 9 each.
- **Effect on the readout flip [VN, section G5].** Under the nested
  sequences, a logical residual created in the check round after forward
  Clifford j reaches the readout conjugated by (C_j ⋯ C_1)⁻¹. It flips the
  X readout when the b component of the image is nonzero. For uniform
  Cliffords this probability is exactly 3/4 for every nonzero residual
  **[VA]**: SL(2,3) is transitive on the 8 nonzero vectors, and 6 of them
  have b ≠ 0. With the catalog:

  | residual | one draw (j = 1) | two draws | three draws |
  |---|---|---|---|
  | X_L^{±1} | 3/4 | 0.7501 | 0.7500 |
  | Z_L^{±1} | 55/72 = 0.7639 | 0.7498 | 0.7500 |
  | (X_L Z_L)^{±1} | 163/216 = 0.7546 | 0.7501 | 0.7500 |
  | (X_L Z_L⁻¹)^{±1} | 0.7315 | 0.7501 | 0.7500 |
  | largest TV distance of the image from uniform | 0.037 | 0.0015 | 0.0001 |

  The four one-draw values average to exactly 3/4. Only the residual from
  round 0 sees the non-uniformity, at the 2–3% level, and it is gone after
  two draws. The effect on the per-depth decay is therefore negligible
  next to the sequence-to-sequence spread of section 9 **[VN for the
  probabilities, CONJ for "negligible" in the fitted decay]**.
- **Token sequences of the files [VN, section G0].** All 270 baseline
  files pass three checks. Every forward Clifford is a catalog string
  executed right to left. The inverse block is the generator's
  construction from the forward Cliffords. Depth m uses the first m
  Cliffords of the sequence's depth-22 sample.

## 11. Claims

| id | claim | label | evidence |
|---|---|---|---|
| A1-syndromes-C1 | The 4 stabilizers commute; X_L = X0X3 and Z_L = Z0Z2 commute with S and ⟨Z_L,X_L⟩ = 1; rank 4/6; the ideal encoder prepares the stabilizer group ⟨S, X_L⟩ of \|+⟩_L. | VA, VN | `check_code`; log section A |
| A1-syndromes-C2 | Each ancilla outcome shifts by ⟨S,E⟩ (σ_Z1 = x0−x1−x3, σ_Z2 = x1+x2−x4, σ_X1 = −z0−z1+z2, σ_X2 = z1−z3+z4); the terminal observable shifts by b(E) = z0+z3; the const=0 detectors equal −σ_X. | VN | `convention_check`; DEM cross-check |
| A1-syndromes-C3 | The only weight-2 nontrivial logicals are X0^aX3^a, X2^aX4^a (coset X_L^a) and Z0^bZ2^b, Z3^bZ4^b (coset Z_L^b), 8 in total; the mixed cosets X_L^aZ_L^b have minimum weight 3 with 6 representatives each; no stabilizer has weight ≤ 2. | VA, VN | CSS argument; enumeration of 9^5 errors (`E_logicals`) |
| A1-syndromes-C4 | Of the 81 syndromes, 1 is trivial, 24 have a unique weight-1 coset, 8 are tied pairs of weight-1 errors, and 48 have minimum weight 2 with at least 2 cosets tied at weight 2 (leading posteriors 1/3, 2/9, 1/4). | VN | `B_syndromes`, appendix |
| A1-syndromes-C5 | 24/40 single-qutrit errors are correctable at leading order (all 8 on the centre qutrit, the 4 mixed ones on each corner); the 16 pure corner errors form the 8 tied pairs (X0^a,X3^{−a}), (X2^a,X4^{−a}), (Z0^b,Z2^{−b}), (Z3^b,Z4^{−b}). | VA, VN | section 3 |
| A1-syndromes-C6 | Code capacity (depolarizing q): detection P_rej = 5q − 81/8 q², P(f\|a) = q²/8 + …; MLD P_fail = q + 63/16 q² + …; strict hybrid P_rej = 2q − 3/4 q², P(f\|a) = 21/8 q² + …. | VN (exact polynomials), VA (leading terms) | `C_code_capacity.depolarizing` |
| A1-syndromes-C7 | Full MLD on this code is no better than an unencoded qutrit at code capacity (failure q versus q). | VN | C6 |
| A1-syndromes-C8 | The posterior-threshold family has three regimes (τ near 1: detection; 0.5 < τ < 0.99: hybrid; τ ≤ 0.5: MLD), because the posteriors cluster at 1 − O(q), 1/2 and ≤ 1/3. | VN | `threshold_family` |
| A1-syndromes-C9 | If only the observable b matters, MLD fails at q/2 and the hybrid rejects q; with only a perfect X-syndrome (const=0), MLD fails at 3q/2 and the hybrid rejects 3q (versus detection 15q/4). | VN | `obs_only`, `xsyn_obs` |
| A1-syndromes-C10 | Pure X or pure Z noise: MLD 2q, hybrid rejects 4q (only the centre qutrit is correctable); mixed-only noise: every single error is correctable (MLD O(q²)); noise on the centre only or on any single qutrit: everything is correctable. General rule: MLD fails Σ min(p_e,p_e′) and the hybrid rejects Σ(p_e+p_e′) over the 8 tied pairs. | VN, VA | section 5 |
| A1-syndromes-C11 | H_L, P_L (and inverses) are code automorphisms with the syndrome maps M and coset maps of section 6 (K = 0); with the true prior the decoding statistics are invariant (the coset table is permuted); with an iid prior assumed after P_L the hybrid accepts failures at q/2 and MLD fails at 9q/4. | VN | `F_gates` |
| A1-syndromes-C12 | Phenomenological model: measurement errors in rounds with a successor never cause failure or rejection at leading order (MLD (R+1)q and hybrid 2(R+1)q, independent of q_m); in the final round without successor, Z-check measurement errors are harmless for the X readout while X-check measurement errors must be rejected (hybrid rejects q + 2q_m in that round). | VN (enumeration plus exact HMM for R = 1, 2) | section 7, `P_phenomenological`, `P_exact_checks` |
| A1-syndromes-C13 | In the ordinary circuits, data noise on qutrits 0 and 3 after their last check interaction gives an undetectable observable flip of 4.95p (4.5p idle in the M7/M8 layers + 0.45p CNOT 7→0 / 8→3 data components), independent of m; no decoder can remove it. | VN | `G3`, depth-0 breakdown; baseline (1−F)/1.5p = 4.9–5.1 |
| A1-syndromes-C14 | Per SI1000 check round plus reset: 118.4p total fault weight; 41.0p measurement-only; 61.1p data residual, of which 23.8p is equivalent to an ambiguous single-qutrit error, 35.2p to a correctable one, and 2.1p is weight-2-like; measurement-only flips per check 5.9–10.6p (+2p from reset). | VN | `G1_cycle` |
| A1-syndromes-C15 | Code-capacity-style counting (λ_corner = 11.8p, λ_centre = 13.6p, ambiguous 23.6p, correctable 37.2p per round) approximates C14 to within about 6% (ambiguous 23.6 vs 23.8; correctable 37.2 vs 35.2; total 60.8 vs 61.1). | CONJ (approximation) | section 8 (i) versus (ii) |
| A1-syndromes-C16 | Memory windows: per check round with a successor, detection rejects 102.1p, MLD fails 11.475p and the hybrid rejects 28.775p (full class); the ambiguous faults are only data faults. | VN | `G2_memory_windows` |
| A1-syndromes-C17 | Baseline LRB circuits, per unit depth: detection rejects 112.0p (baseline 111.7p), the hybrid rejects 39.3p, MLD fails 10.96p; undetected 4.95p is flat in m. | VN | `G3_lrb_circuits` (10 sequences) and baseline CSVs |
| A1-syndromes-C18 | Second-order coefficients: detection 15 p² and hybrid about 1.0·10³ p² of conditional failure per bulk round (memory, V2); LRB m = 0: detection 4.95p + 55.8p², hybrid 4.95p + 396p²; these reproduce the baseline unif-1 fidelities within 5·10⁻⁵ for p ≤ 2·10⁻³. | VN | `G4_order2` |
| A1-syndromes-C19 | MLD-corrected LRB would decay with 1 − f ≈ 16p per depth (about 2.9 × physical RB at p_RB = 5p); the hybrid with 1 − f ≈ 1.3·10³ p², beating physical RB only for p ≲ 4·10⁻³. | CONJ (extrapolated from the VN leading coefficients) | section 9 |
| A1-syndromes-C20 | The own generator reproduces all 270 baseline LRB circuits op by op, and the own propagation reproduces the repository DEM on 2428 unit components with 0 mismatches. | VN | `G0_generator_validation`, `G_dem_crosscheck` |
| A1-syndromes-C21 | The 216-string Clifford catalog holds only 206 of the 216 single-qutrit Clifford unitaries modulo phase (207 distinct strings; 10 elements never sampled). | VN (sdim conventions; logical phases not checked) | `catalog_check`; group closure check (section 10) |
| A1-syndromes-C22 | Folded surface codes support transversal Clifford gates and are defined for qudits of any dimension; qudit stabilizer codes for prime d use the symplectic formalism used here; SI1000 is the noise model of Gidney et al.; the phenomenological model with repeated noisy syndromes is that of Dennis et al.; logical-gap / soft-output postselection is an established way to trade acceptance for logical error. | LIT | section 12 |

## 12. References (all fetched in this task)

All six pages were fetched again by the reviser on 2026-10-28. Titles,
authors, years and identifiers were checked against the arXiv pages.

1. J. E. Moussa, "Transversal Clifford gates on folded surface codes,"
   Phys. Rev. A 94, 042316 (2016), arXiv:1603.02286,
   DOI 10.1103/PhysRevA.94.042316. Fetched: https://arxiv.org/abs/1603.02286.
   The abstract says "two families of folded surface codes with transversal
   Clifford gates" and "presented generally for qudits of any dimension".
   Used for the folded-surface-code family, its transversal Cliffords and
   its definition for any qudit dimension. The specific [[5,1,2]]_3 instance
   and its gate circuits are taken from the repository code, not from the
   paper.
2. D. Gottesman, "Fault-Tolerant Quantum Computation with Higher-Dimensional
   Systems," Chaos Solitons Fractals 10, 1749–1758 (1999),
   arXiv:quant-ph/9802007, DOI 10.1016/S0960-0779(98)00218-5. Fetched:
   https://arxiv.org/abs/quant-ph/9802007. The abstract says that stabilizer
   codes generalize directly to d-dimensional systems and that universal
   fault-tolerant computation is possible for prime d. Used for the qudit
   stabilizer formalism. The abstract does not name the symplectic form. The
   Z_3 conventions here are taken from sdim and checked by statevector
   (section G6), not from the paper.
3. C. Gidney, M. Newman, A. Fowler, M. Broughton, "A Fault-Tolerant Honeycomb
   Memory," Quantum 5, 605 (2021), arXiv:2108.10457,
   DOI 10.22331/q-2021-12-20-605. Fetched: https://arxiv.org/abs/2108.10457
   and https://ar5iv.labs.arxiv.org/html/2108.10457. Table 2 gives SI1000:
   CZ p, one-qubit Clifford p/10, initialization flip 2p, measurement flip
   5p before measurement, idle p/10, resonator idle 2p. Used for the SI1000
   definition.
4. E. Dennis, A. Kitaev, A. Landahl, J. Preskill, "Topological quantum
   memory," J. Math. Phys. 43, 4452–4505 (2002), arXiv:quant-ph/0110143,
   DOI 10.1063/1.1499754. Fetched: https://arxiv.org/abs/quant-ph/0110143
   and https://ar5iv.labs.arxiv.org/html/quant-ph/0110143. Sec. IV.1 uses
   p for qubit errors and q for a faulty syndrome bit. Sec. IV.2 repeats
   the syndrome measurement and decodes in space-time. Used for the
   phenomenological model.
5. J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized
   Benchmarking," arXiv:1702.03688 (2017). Fetched:
   https://arxiv.org/abs/1702.03688. The arXiv page lists no journal
   reference. Used as LRB context.
6. H. Bombín, M. Pant, S. Roberts, K. I. Seetharam, "Fault-tolerant
   Post-Selection for Low Overhead Magic State Preparation," PRX Quantum 5,
   010302 (2024), arXiv:2212.00813, DOI 10.1103/PRXQuantum.5.010302.
   Fetched: https://arxiv.org/abs/2212.00813. The abstract defines the
   logical gap as "roughly the unsigned weight difference between
   inequivalent logical corrections" and uses soft-information metrics
   built from syndrome and erasure information. **Revised use:** a related
   soft-information postselection rule. It is a weight gap between the best
   corrections in two classes, not a threshold on max_c P(c|s). The first
   version called it "the general form of the posterior-threshold family";
   that was this note's interpretation, not the paper's.

## Appendix. All 81 syndromes

The syndrome digits are (σ_Z1 σ_Z2 σ_X1 σ_X2) ∈ {0,1,2}. The "weight-2
counts" column gives the number of weight-2 errors in each coset; the full
lists are in `A1_results.json` → `B_syndromes[*].w2_errors`. "Leading
posterior" is the iid-depolarizing posterior of the best coset as q → 0.

| # | s (Z1 Z2 X1 X2) | class | w_min | weight-1 errors by coset | weight-2 counts by coset | leading posterior |
|---|---|---|---|---|---|---|
| 1 | 0000 | T | 0 | - | XL:2, XL^-1:2, ZL:2, ZL^-1:2 | 1 |
| 2 | 2100 | U1 | 1 | I: X1 | I:2, ZL:2, ZL^-1:2, XL:1, XL^-1:1 | 1 |
| 3 | 1200 | U1 | 1 | I: X1^-1 | I:2, ZL:2, ZL^-1:2, XL:1, XL^-1:1 | 1 |
| 4 | 1010 | U1 | 1 | XL ZL^-1: X0Z0^-1 | I:2, XL:2, ZL^-1:2, XL ZL:1, XL^-1 ZL^-1:1 | 1 |
| 5 | 2010 | U1 | 1 | XL^-1 ZL^-1: X0^-1Z0^-1 | I:2, XL^-1:2, ZL^-1:2, XL^-1 ZL:1, XL ZL^-1:1 | 1 |
| 6 | 0110 | U1 | 1 | XL: X2Z2 | I:2, ZL^-1:2, XL ZL^-1:2, XL^-1:1, XL ZL:1 | 1 |
| 7 | 0210 | U1 | 1 | XL^-1: X2^-1Z2 | I:2, ZL^-1:2, XL^-1 ZL^-1:2, XL:1, XL^-1 ZL:1 | 1 |
| 8 | 1020 | U1 | 1 | XL ZL: X0Z0 | I:2, XL:2, ZL:2, XL^-1 ZL:1, XL ZL^-1:1 | 1 |
| 9 | 2020 | U1 | 1 | XL^-1 ZL: X0^-1Z0 | I:2, XL^-1:2, ZL:2, XL ZL:1, XL^-1 ZL^-1:1 | 1 |
| 10 | 0120 | U1 | 1 | XL: X2Z2^-1 | I:2, ZL:2, XL ZL:2, XL^-1:1, XL ZL^-1:1 | 1 |
| 11 | 0220 | U1 | 1 | XL^-1: X2^-1Z2^-1 | I:2, ZL:2, XL^-1 ZL:2, XL:1, XL^-1 ZL^-1:1 | 1 |
| 12 | 1001 | U1 | 1 | ZL^-1: X3^-1Z3^-1 | I:2, XL:2, XL ZL^-1:2, ZL:1, XL^-1 ZL^-1:1 | 1 |
| 13 | 2001 | U1 | 1 | ZL^-1: X3Z3^-1 | I:2, XL^-1:2, XL^-1 ZL^-1:2, ZL:1, XL ZL^-1:1 | 1 |
| 14 | 0101 | U1 | 1 | I: X4^-1Z4 | XL:2, ZL^-1:2, XL ZL^-1:2, XL^-1:1, ZL:1 | 1 |
| 15 | 0201 | U1 | 1 | I: X4Z4 | XL^-1:2, ZL^-1:2, XL^-1 ZL^-1:2, XL:1, ZL:1 | 1 |
| 16 | 0021 | U1 | 1 | I: Z1 | I:2, XL:2, XL^-1:2, ZL:1, ZL^-1:1 | 1 |
| 17 | 2121 | U1 | 1 | I: X1Z1 | XL^-1 ZL:1, XL ZL^-1:1 | 1 |
| 18 | 1221 | U1 | 1 | I: X1^-1Z1 | XL ZL:1, XL^-1 ZL^-1:1 | 1 |
| 19 | 1002 | U1 | 1 | ZL: X3^-1Z3 | I:2, XL:2, XL ZL:2, XL^-1 ZL:1, ZL^-1:1 | 1 |
| 20 | 2002 | U1 | 1 | ZL: X3Z3 | I:2, XL^-1:2, XL^-1 ZL:2, XL ZL:1, ZL^-1:1 | 1 |
| 21 | 0102 | U1 | 1 | I: X4^-1Z4^-1 | XL:2, ZL:2, XL ZL:2, XL^-1:1, ZL^-1:1 | 1 |
| 22 | 0202 | U1 | 1 | I: X4Z4^-1 | XL^-1:2, ZL:2, XL^-1 ZL:2, XL:1, ZL^-1:1 | 1 |
| 23 | 0012 | U1 | 1 | I: Z1^-1 | I:2, XL:2, XL^-1:2, ZL:1, ZL^-1:1 | 1 |
| 24 | 2112 | U1 | 1 | I: X1Z1^-1 | XL ZL:1, XL^-1 ZL^-1:1 | 1 |
| 25 | 1212 | U1 | 1 | I: X1^-1Z1^-1 | XL^-1 ZL:1, XL ZL^-1:1 | 1 |
| 26 | 1000 | A1 | 1 | I: X3^-1; XL: X0 | I:1, XL:1, XL^-1:1, ZL:1, XL ZL:1, ZL^-1:1, XL ZL^-1:1 | 1/2 |
| 27 | 2000 | A1 | 1 | I: X3; XL^-1: X0^-1 | I:1, XL:1, XL^-1:1, ZL:1, XL^-1 ZL:1, ZL^-1:1, XL^-1 ZL^-1:1 | 1/2 |
| 28 | 0100 | A1 | 1 | I: X4^-1; XL: X2 | I:1, XL:1, XL^-1:1, ZL:1, XL ZL:1, ZL^-1:1, XL ZL^-1:1 | 1/2 |
| 29 | 0200 | A1 | 1 | I: X4; XL^-1: X2^-1 | I:1, XL:1, XL^-1:1, ZL:1, XL^-1 ZL:1, ZL^-1:1, XL^-1 ZL^-1:1 | 1/2 |
| 30 | 0010 | A1 | 1 | I: Z2; ZL^-1: Z0^-1 | I:1, XL:1, XL^-1:1, ZL:1, ZL^-1:1, XL ZL^-1:1, XL^-1 ZL^-1:1 | 1/2 |
| 31 | 0020 | A1 | 1 | I: Z2^-1; ZL: Z0 | I:1, XL:1, XL^-1:1, ZL:1, XL ZL:1, XL^-1 ZL:1, ZL^-1:1 | 1/2 |
| 32 | 0001 | A1 | 1 | I: Z4; ZL^-1: Z3^-1 | I:1, XL:1, XL^-1:1, ZL:1, ZL^-1:1, XL ZL^-1:1, XL^-1 ZL^-1:1 | 1/2 |
| 33 | 0002 | A1 | 1 | I: Z4^-1; ZL: Z3 | I:1, XL:1, XL^-1:1, ZL:1, XL ZL:1, XL^-1 ZL:1, ZL^-1:1 | 1/2 |
| 34 | 1100 | A2* | 2 | - | I:3, XL^-1:3, XL:2, ZL:1, XL^-1 ZL:1, ZL^-1:1, XL^-1 ZL^-1:1 | 1/4 |
| 35 | 2200 | A2* | 2 | - | I:3, XL:3, XL^-1:2, ZL:1, XL ZL:1, ZL^-1:1, XL ZL^-1:1 | 1/4 |
| 36 | 1110 | A2* | 2 | - | XL^-1:2, XL^-1 ZL^-1:2, I:1, XL:1, XL^-1 ZL:1, ZL^-1:1, XL ZL^-1:1 | 2/9 |
| 37 | 2110 | A2* | 2 | - | I:3, ZL^-1:3, XL:1, ZL:1, XL^-1 ZL^-1:1 | 1/3 |
| 38 | 1210 | A2* | 2 | - | I:3, ZL^-1:3, XL^-1:1, ZL:1, XL ZL^-1:1 | 1/3 |
| 39 | 2210 | A2* | 2 | - | XL:2, XL ZL^-1:2, I:1, XL^-1:1, XL ZL:1, ZL^-1:1, XL^-1 ZL^-1:1 | 2/9 |
| 40 | 1120 | A2* | 2 | - | XL^-1:2, XL^-1 ZL:2, I:1, XL:1, ZL:1, XL ZL:1, XL^-1 ZL^-1:1 | 2/9 |
| 41 | 2120 | A2* | 2 | - | I:3, ZL:3, XL:1, XL^-1 ZL:1, ZL^-1:1 | 1/3 |
| 42 | 1220 | A2* | 2 | - | I:3, ZL:3, XL^-1:1, XL ZL:1, ZL^-1:1 | 1/3 |
| 43 | 2220 | A2* | 2 | - | XL:2, XL ZL:2, I:1, XL^-1:1, ZL:1, XL^-1 ZL:1, XL ZL^-1:1 | 2/9 |
| 44 | 1101 | A2* | 2 | - | I:2, ZL^-1:2, XL:1, XL^-1:1, ZL:1, XL ZL^-1:1, XL^-1 ZL^-1:1 | 2/9 |
| 45 | 2101 | A2* | 2 | - | I:3, ZL^-1:3, XL^-1:1, ZL:1, XL ZL^-1:1 | 1/3 |
| 46 | 1201 | A2* | 2 | - | I:3, ZL^-1:3, XL:1, ZL:1, XL^-1 ZL^-1:1 | 1/3 |
| 47 | 2201 | A2* | 2 | - | I:2, ZL^-1:2, XL:1, XL^-1:1, ZL:1, XL ZL^-1:1, XL^-1 ZL^-1:1 | 2/9 |
| 48 | 0011 | A2* | 2 | - | I:3, ZL:3, ZL^-1:2, XL:1, XL^-1:1, XL ZL:1, XL^-1 ZL:1 | 1/4 |
| 49 | 1011 | A2* | 2 | - | ZL:2, XL ZL:2, I:1, XL:1, XL^-1 ZL:1, ZL^-1:1, XL ZL^-1:1 | 2/9 |
| 50 | 2011 | A2* | 2 | - | ZL:2, XL^-1 ZL:2, I:1, XL^-1:1, XL ZL:1, ZL^-1:1, XL^-1 ZL^-1:1 | 2/9 |
| 51 | 0111 | A2* | 2 | - | I:2, XL:2, XL^-1:1, ZL:1, XL ZL:1, ZL^-1:1, XL ZL^-1:1 | 2/9 |
| 52 | 1111 | A2* | 2 | - | XL ZL^-1:2, I:1, XL^-1:1, ZL:1, XL^-1 ZL:1 | 1/3 |
| 53 | 2111 | A2* | 2 | - | I:2, ZL:2, XL ZL^-1:1, XL^-1 ZL^-1:1 | 1/3 |
| 54 | 0211 | A2* | 2 | - | I:2, XL^-1:2, XL:1, ZL:1, XL^-1 ZL:1, ZL^-1:1, XL^-1 ZL^-1:1 | 2/9 |
| 55 | 1211 | A2* | 2 | - | I:2, ZL:2, XL ZL^-1:1, XL^-1 ZL^-1:1 | 1/3 |
| 56 | 2211 | A2* | 2 | - | XL^-1 ZL^-1:2, I:1, XL:1, ZL:1, XL ZL:1 | 1/3 |
| 57 | 1021 | A2* | 2 | - | I:3, XL:3, XL^-1:1, XL ZL:1, ZL^-1:1 | 1/3 |
| 58 | 2021 | A2* | 2 | - | I:3, XL^-1:3, XL:1, XL^-1 ZL:1, ZL^-1:1 | 1/3 |
| 59 | 0121 | A2* | 2 | - | I:3, XL:3, XL^-1:1, ZL:1, XL ZL^-1:1 | 1/3 |
| 60 | 1121 | A2* | 2 | - | I:2, XL^-1:2, XL ZL:1, XL ZL^-1:1 | 1/3 |
| 61 | 0221 | A2* | 2 | - | I:3, XL^-1:3, XL:1, ZL:1, XL^-1 ZL^-1:1 | 1/3 |
| 62 | 2221 | A2* | 2 | - | I:2, XL:2, XL^-1 ZL:1, XL^-1 ZL^-1:1 | 1/3 |
| 63 | 1102 | A2* | 2 | - | I:2, ZL:2, XL:1, XL^-1:1, XL ZL:1, XL^-1 ZL:1, ZL^-1:1 | 2/9 |
| 64 | 2102 | A2* | 2 | - | I:3, ZL:3, XL^-1:1, XL ZL:1, ZL^-1:1 | 1/3 |
| 65 | 1202 | A2* | 2 | - | I:3, ZL:3, XL:1, XL^-1 ZL:1, ZL^-1:1 | 1/3 |
| 66 | 2202 | A2* | 2 | - | I:2, ZL:2, XL:1, XL^-1:1, XL ZL:1, XL^-1 ZL:1, ZL^-1:1 | 2/9 |
| 67 | 1012 | A2* | 2 | - | I:3, XL:3, XL^-1:1, ZL:1, XL ZL^-1:1 | 1/3 |
| 68 | 2012 | A2* | 2 | - | I:3, XL^-1:3, XL:1, ZL:1, XL^-1 ZL^-1:1 | 1/3 |
| 69 | 0112 | A2* | 2 | - | I:3, XL:3, XL^-1:1, XL ZL:1, ZL^-1:1 | 1/3 |
| 70 | 1112 | A2* | 2 | - | I:2, XL^-1:2, XL ZL:1, XL ZL^-1:1 | 1/3 |
| 71 | 0212 | A2* | 2 | - | I:3, XL^-1:3, XL:1, XL^-1 ZL:1, ZL^-1:1 | 1/3 |
| 72 | 2212 | A2* | 2 | - | I:2, XL:2, XL^-1 ZL:1, XL^-1 ZL^-1:1 | 1/3 |
| 73 | 0022 | A2* | 2 | - | I:3, ZL^-1:3, ZL:2, XL:1, XL^-1:1, XL ZL^-1:1, XL^-1 ZL^-1:1 | 1/4 |
| 74 | 1022 | A2* | 2 | - | ZL^-1:2, XL ZL^-1:2, I:1, XL:1, ZL:1, XL ZL:1, XL^-1 ZL^-1:1 | 2/9 |
| 75 | 2022 | A2* | 2 | - | ZL^-1:2, XL^-1 ZL^-1:2, I:1, XL^-1:1, ZL:1, XL^-1 ZL:1, XL ZL^-1:1 | 2/9 |
| 76 | 0122 | A2* | 2 | - | I:2, XL:2, XL^-1:1, ZL:1, XL ZL:1, ZL^-1:1, XL ZL^-1:1 | 2/9 |
| 77 | 1122 | A2* | 2 | - | XL ZL:2, I:1, XL^-1:1, ZL^-1:1, XL^-1 ZL^-1:1 | 1/3 |
| 78 | 2122 | A2* | 2 | - | I:2, ZL^-1:2, XL ZL:1, XL^-1 ZL:1 | 1/3 |
| 79 | 0222 | A2* | 2 | - | I:2, XL^-1:2, XL:1, ZL:1, XL^-1 ZL:1, ZL^-1:1, XL^-1 ZL^-1:1 | 2/9 |
| 80 | 1222 | A2* | 2 | - | I:2, ZL^-1:2, XL ZL:1, XL^-1 ZL:1 | 1/3 |
| 81 | 2222 | A2* | 2 | - | XL^-1 ZL:2, I:1, XL:1, ZL^-1:1, XL ZL^-1:1 | 1/3 |
