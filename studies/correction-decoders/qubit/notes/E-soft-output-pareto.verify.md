# E verification (adversarial): soft-output Pareto vs LRB-D

Independent verifier for work package E
(`notes/E-soft-output-pareto.md`, `src/lrb_qubit/acceptance_rules.py`,
`studies/correction-decoders/qubit/scripts/pareto.py`, `studies/correction-decoders/qubit/scripts/hybrid_rules.py`,
`results/E/`).

- **Code:** `studies/correction-decoders/qubit/scripts/verify_pareto.py` (new). It never uses E's
  modules to reach a verdict. It imports `acceptance_rules` in two places
  only, to compare E's `tl_d5` rule with mine on identical inputs. It uses
  the lead's `circuits`, `baseline`, `mld` and `exact` modules as trusted
  infrastructure.
- **Outputs:** `results/E-verify/` (`c4.csv`, `c5_c10.csv`, `c12.csv`,
  `c13_audit.csv`, `c14.csv`, `fair.csv`, `calibration_*.csv`, logs).
- **Samples:** fresh, with seed base 31,415,000,000. This is disjoint from
  the grid and from E's 9e9.
- **Labels:** **VA** verified analytically, **VN** verified numerically,
  **LIT** literature (fetched).

## Verdicts

| # | Claim (abridged) | Verdict | Key evidence | Fix needed |
|---|---|---|---|---|
| C1 | Thresholding \|LLR\| minimises accepted error at fixed acceptance | **confirmed** (VA, LIT) | Knapsack argument re-derived (§8). Seip et al. Thm II.1 fetched; equivalent for two cosets | State the scope: per-circuit or pooled error rate, not the repo's sequence-averaged F |
| C2 | Exact-MLD soft output is calibrated | **confirmed** (VA, VN) | Decoder model = sampler model. Direct Bernoulli test on fresh shots, §8 | none |
| C3 | No LRB-D point beats the curve with the same information | **confirmed** (VN; follows from C1) | Exact identical-shot ratios ≥ 1 (§6). E's 10,715-comparison summary re-read | none |
| C4 | unif=1 and unif=1+final keep exactly the gap rule's shots | **confirmed** (VN) | Direct shot-set equality in 12/12 (circuit, information) cases. All-zero syndrome is the global \|LLR\| maximum, exhaustively at m ≤ 4 | E's identity test is weaker than claimed. The "because" is empirical, with margins down to 0.02 (§2) |
| C5 | Data-2 cap \|LLR\| ≤ log((1−u)/u), error ≥ u, u ≈ 4.53p | **confirmed** (VA, VN) | Exhaustive m ≤ 4; u/p = 4.532 (p=1e-4), 4.504 (2e-3), 4.386 (1e-2) | none |
| C6 | Same-information factors: unif=2 0.99–1.17×, unif=5 1.01–1.88×, const=1 1.04–6.9× | **numbers confirmed; framing refuted** | §6: on the frontier built from the rounds each policy reads, the sparse LRB-D points lie exactly on it | Rename "same information"; attribute the factor to the ignored rounds; fix 0.99 and "3–15×" |
| C7 | L0 §7 rows reproduced | **confirmed** (VN) | All nine rows re-read; unif=4 row recomputed with own code (4.698 / 1.398) | Store the unif=4 row in a CSV |
| C8 | Equal-error gains vs const=1 (1.5×, 2.2×, 4.8×) | **confirmed** (VN) | CSV values; exact identical-shot gains in §6 | Same framing fix as C6 |
| C9 | Five-qubit readout lowers error at unif=1's acceptance 248×→9.2× (p=5e-4) | **confirmed** (VN) | CSV values. By C4 this is the mld2/mld5 frontier ratio | Say "ideal (noiseless) readout" |
| C10 | Fixed threshold t ⇒ F(m) ≥ tanh(t/2) | **confirmed** (VA, VN) | Derivation (§3). Exact tables m ≤ 4. Grid 793/793, closest z = +0.416 | none |
| C11 | Plain MLD maximises acceptance × F | **confirmed** (VA, VN) | Triangle-inequality proof (§3). eff_fidelity 0/126, max excess −1.6e-4 | Say "closest" is in sequence-SE units |
| C12 | Qubit-1 syndromes ambiguous via X2X3/Z2Z3 from fold gates | **confirmed** (VA, VN) | Stim Pauli algebra; 918 injected faults at 306 fold-gate locations | Optional wording (§4) |
| C13 | tl_d5 accepts no logical-flipping single fault | **confirmed** (VN) | Own rule from detector definitions. 270 circuits, every mechanism. Full agreement with E's rule | Optional: note the lone-Z clause is redundant |
| C14 | tl_d5 keeps 5.1× unif=1+final's shots; gap rule only 1.26× more at equal error (p=2e-3, m=22) | **confirmed within uncertainty** (VN) | 5.11× [5.05, 5.16]. Gap gain 1.18 [1.14, 1.24] (realised errors) or 1.23 (calibrated q) | Quote as about 1.2–1.3× |
| C15 | tlq1 / lut0 / lut10 at 2.0–2.6× the gap error | **not re-derived** | Consistent with E's tables and with C12 (tlq1 first-order error = fold-gate count × 2p/15) | — |
| C16 | Hybrids lose by acceptance, not correction | **confirmed for tl_d5** (VN) | MLD errors on tl_d5's accepted set = tl_d5's errors exactly (886, 4,372, 1,471 at m = 0, 6, 22) | — |

E's tests pass: `python -m pytest tests/test_qubit512_pareto.py` gives 22
passed in 42.7 s (shared allocation, single-threaded BLAS).

## 2. C4: the LRB-D every-check policies are gap-rule points

### Direct test (VN)

The setting was p = 2e-3, m ∈ {2, 6}, grid sequences 0–2 and 2e5 fresh shots
per circuit. For each shot I computed the lead's exact LLR twice: with all
detectors (mld5), and with the check-round detectors only (mld2, from the
data-2 DEM). Results (`c4.csv`, rows `sampled`):

| m | info | distinct syndromes | \|LLR(0)\| | max \|LLR\| of any other observed syndrome | margin |
|---|---|---|---|---|---|
| 2 | mld5 | 2,215–2,232 | 7.10–7.71 | 6.30–7.27 | 0.45–0.81 |
| 2 | mld2 | 1,416–1,448 | 4.54–4.60 | 4.45–4.58 | 0.025–0.092 |
| 6 | mld5 | 32,340–34,148 | 6.38–6.68 | 5.92–6.50 | 0.18–0.47 |
| 6 | mld2 | 29,539–31,409 | 4.46–4.52 | 4.38–4.50 | 0.023–0.084 |

In all 12 (circuit, information set) cases:

- the all-zero syndrome has one \|LLR\| value and strictly exceeds every
  other observed syndrome;
- the rule \|LLR\| ≥ \|LLR(0)\| − margin/2 accepts exactly the LRB-D shot
  set (bitwise equal), where LRB-D sets come from the lead's
  `raw_syndromes` + `accept_mask`: unif=1 for mld2, unif=1 + all-zero final
  detectors for mld5;
- the top-n shots by \|LLR\| (n = LRB-D accepted count) are that same set,
  with a strict boundary;
- the MLD decision on the all-zero syndrome is "no flip".

### Stronger: unobserved syndromes (VN)

- **Exhaustive (`exhaustive` rows).** I enumerated every syndrome,
  sequences 0–2:
  - m = 2, p = 2e-3: 2^14 (data-5) and 2^12 (data-2) syndromes. The LLRs
    come from my own brute-force P(s, L), independent of the lead's
    decoder, and agree with `TransferMatrixMLD` to 6e-15.
  - m = 4, p ∈ {2e-3, 1e-2}: 2^22 and 2^20 syndromes, using the lead's
    decoder.

  The all-zero syndrome is the global \|LLR\| maximum every time. Its
  margin is 0.26–0.81 (mld5) and 0.023–0.092 (mld2) at p = 2e-3, and
  0.07–0.14 / 0.023–0.059 at p = 1e-2.
- **One- and two-fault syndromes (`one_or_two_mechanisms` rows).** I checked
  every syndrome produced by one or two DEM mechanisms: up to 533,028 pair
  syndromes at m = 22, for p ∈ {2e-3, 1e-2}. The all-zero syndrome wins
  every time.
- **mld2 is the exact check-round posterior.** The data-2 DEM is the exact
  marginal of the data-5 DEM over the final detectors (brute force, m = 2,
  max difference 1.7e-17).

### Two caveats (required fixes)

1. **E's identity test proves less than claimed.** E calls a circuit
   "identical to the gap rule" when the paired difference and its variance
   are both zero (`hybrid_rules.py`, `circuits_identical_to_gap_rule`). That
   tests whether the per-shot *error indicators* coincide, not whether the
   *accepted sets* do. If the two sets differed only by swapping non-error
   shots, z_i = 0 would still hold for every shot. The test is weak exactly
   where few errors are accepted. For unif=1 + final at p = 5e-4, m = 0,
   about 1.6 accepted errors are expected per circuit.
   - The conclusion stands: my direct set comparison and the exhaustive
     enumeration cover that regime.
   - Required fix: describe the 1,414/1,414 count as "identical error
     indicators", or add a direct set comparison.
2. **"Because the all-clean syndrome is the most confident" is empirical,
   not structural.**
   - I know no argument that forces it.
   - The margin shrinks with p·m. The smallest margins are about 0.02 for
     mld2 even at p = 2e-3, and 0.034 for mld5 at p = 1e-2, m = 22.
   - At p = 1e-2, m = 6, sequence 2 (mld5) the runner-up is a
     two-mechanism syndrome.
   - Required fix: label it VN (verified exhaustively at m ≤ 4 and for all
     one- and two-fault syndromes to m = 22), not a general property.

## 3. C10 and C11 analytically

### C10: fixed threshold (VA)

The exact MLD posterior gives, for syndrome s,
P(L ≠ L̂(s) | s) = q(s) = 1 / (1 + e^{\|LLR(s)\|}). This is the definition of
the LLR, and it is calibrated because the decoder's DEM is the sampler's
model.

1. The gap rule accepts s iff \|LLR(s)\| ≥ t. Because q is decreasing in
   \|LLR\|, that is q(s) ≤ q(t).
2. The expected accepted error rate is
   ε(t) = Σ_{acc} P(s) q(s) / Σ_{acc} P(s) ≤ q(t) ≤ 1/2.
3. Hence 1 − 2ε ≥ 1 − 2q(t) = (e^t − 1)/(e^t + 1) = tanh(t/2) ≥ 0.

**The absolute value.** F = \|1 − 2ε\| ≥ 1 − 2ε, so the absolute value can
only raise F. Equality needs every accepted syndrome to sit exactly at
\|LLR\| = t, so the bound is tight only in that degenerate case. Its
direction is a lower bound on F, as E states.

**The repo estimator.** The repo uses the mean over sequences of
\|1 − 2ε̂_seq\|, with ε̂ the empirical accepted error.

- Conditional on n accepted shots, the errors are independent
  Bernoulli(q(s_i)) with every q(s_i) ≤ q(t). So E[ε̂ | n] ≤ q(t).
- Jensen's inequality (\|·\| is convex) gives E[F̂ | n] ≥ tanh(t/2).
- The bound therefore holds in expectation for the repo's F(m).
- Single realisations can dip below it by shot noise; none did on the grid.
- Depths where some sequence accepts nothing are NaN and carry no bound.

**Numerical check (VN).**

- *Exact syndrome tables* (`c5_c10.csv`, own brute-force P(s, L)):
  p ∈ {1e-4, 2e-3, 1e-2} with m ∈ {0, 1, 2} (three sequences), plus
  p = 2e-3, m = 4 (two sequences). Both readouts, t = 0…12 in steps of
  0.25. Every case has ε(t) ≤ q(t) and
  F ≥ tanh(t/2). The smallest slack F − tanh(t/2) is 7e-6.
- *Grid* (`fits_threshold.csv`, recounted independently): 793 cells with
  t > 0 and finite F, 0 below tanh(t/2). The closest is z = +0.416 in
  sequence-SE units (E: +0.42).
- *Quoted fits reproduced.* At p = 2e-3 (mld5),
  f = 0.947727(532), 0.994842(44), 0.998921(14) for t = 0, 2, 4. The
  acceptance at m = 22 is 1, 0.198, 0.049, and F(22) = 0.296, 0.900, 0.979
  against tanh(t/2) = 0, 0.762, 0.964.

### C11: acceptance × F (VA)

For any rule with acceptance probability a(s) and any decision L̂(s):

α·F = \|Σ_s P(s) a(s) [P(L̂|s) − P(L̂ ⊕ 1|s)]\|
    ≤ Σ_s P(s) a(s) \|P(0|s) − P(1|s)\|
    = Σ_s P(s) a(s)(1 − 2q(s))
    ≤ 1 − 2ε_MLD.

So the bound holds with the absolute value, and for uncorrected LRB-D
decisions as well.

**Numerical check (VN).** `eff_fidelity.csv` has 0 of 126 cells with
positive excess; the largest excess is −1.6e-4 (p = 1e-4, m = 0). E's
"closest is 0.9364 vs 0.9385 at p = 5e-4, m = 4" is the closest in units of
the sequence SE (−1.66 SE), not in absolute terms. Fix: say which metric.

## 4. C12: fold-gate faults mimic qubit-1 errors

### Algebra (VA, `stim.PauliString`)

- X1 and X2X3 both have syndrome (S_Z1, S_Z2, S_X1, S_X2) = (1,1,0,0).
- Their product X1X2X3 commutes with all four stabilizers and is not in the
  stabilizer group. It anticommutes with Z_L, so it is X_L up to a
  stabilizer.
- The same holds for Z1 / Z2Z3, with syndrome (0,0,1,1); the product
  anticommutes with X_L.
- It holds for Y1 / Y2Y3 too, with syndrome (1,1,1,1); the product
  anticommutes with both.
- X2X3 = X1·X_L·S_X1 and Z2Z3 = Z1·Z_L·S_Z1 hold exactly, up to sign.

### Circuit level (VN)

Setup:

- noiseless data5 circuits, 5 sequences, m ∈ {2, 6, 10};
- 306 fold-gate locations (every SWAP(2,3) and CZ(2,3));
- at each location I injected each of X2X3, Y2Y3, Z2Z3 and each of X1, Y1,
  Z1 right after the gate.

Results:

- **Twins.** All 918 pair faults have a unique qubit-1 twin with identical
  detection events.
- **Logical flips.** Exactly 2 of the 3 per location flip the observable
  relative to their twin (612/918), for SWAP and CZ alike.
- **Qubit-1 faults.** Faults on qubit 1 never flip the observable.

This matches E's audit: tlq1's first-order error is 2.45p at m = 6 and
8.84p at m = 22. That is (about 1.5 fold gates per Clifford) × 2m ×
(2p/15) = 2.4p and 8.8p.

### Optional wording fix

In the inverse half, `P_INV` applies CZ(2,3) *before* the S on qubit 1, so a
CZ fault X2X3 is twinned by Y1 at that point, not by X1. The statement
"X2X3/Z2Z3/Y2Y3 have the syndromes of X1/Z1/Y1" is exact at the moment the
syndrome is measured. "Same syndrome up to the single-qubit gates still to
act on qubit 1" is more precise.

## 5. C13: own tl_d5 implementation

### Definition used (from E §1.7, implemented without SlotLayout or `LRBCircuit.back`)

1. Parse the Stim circuit. Each detector gets its round label and the set
   of absolute measurement records it XORs.
2. The pattern of a flipped ancilla measurement is the set of detectors
   whose definition contains that record.
3. Accepted cluster patterns are those sets, plus single terminal Z-check
   detectors.
4. A shot is accepted iff every maximal run of consecutive non-empty slots
   is exactly one accepted pattern. No correction is applied.

A set-based per-shot reference and a vectorised version agree on every
input tested.

### Audit (VN, `c13_audit.csv`)

Every DEM mechanism of every circuit was checked: 30 sequences × 9 depths
= 270 circuits at p = 2e-3.

- **No logical-flipping mechanism is accepted.** The count is 0 in all 270
  circuits, with no detector-free flipping mechanisms either.
- **First-order rejected weight** (mean over sequences, /p) is 33.56, 155.2,
  282.5, 409.3, 667.9, 923.0, 1179.5, 1307.4, 1435.9 at m = 0…22. This
  equals E's `hybrid_audit.csv` to 8 digits.
- **Two detailed circuits** (sequence 0 at m = 6, sequence 1 at m = 22): my
  rule and E's `evaluate_cluster_rule` agree on every mechanism, on all
  43,956 / 533,028 two-mechanism syndromes, and on 2e4 shots at p = 0.02.
  They also agree on all 9e6 fresh shots of §7 (0 disagreements). E's rule
  never applies a flip.

### Minor finding

With the final data slot, every lone terminal Z-check event is already the
pattern of a flipped terminal Z-ancilla measurement. E's separate
`terminal_z` clause is therefore redundant in `tl_d5`. It is harmless.

## 6. C6–C9: is the "same information" comparison fair?

### What is fair (VN)

- **Data-2 policies against mld2.** Both use only check-round detection
  events, and mld2 is the exact posterior for them (data-2 DEM = exact
  marginal of the data-5 DEM, §2). Removing the final data readout
  correctly removes the 4.53p floor that inflated L0's 3–15×.
- **`_final` policies against mld5.** Correct.
- **The binned frontier does not distort the factors.**
  - E's own exact identical-shot test (`np_test_exact_lrbd.csv`, per-circuit
    matching, no bins) agrees with the binned grid factors to within about
    2% for every sparse policy.
  - unif=2 / unif=5 / const=1 at p ∈ {5e-4, 1e-3, 2e-3}, m ∈ {6, 14, 22}:
    exact 1.01–1.12 / 1.07–1.87 / 1.21–6.91, against binned 1.02–1.12 /
    1.09–1.86 / 1.23–6.86.
- **The quoted ranges match `matched_points.csv`** over p ≤ 2e-3, m ≥ 6:
  unif=2 0.990–1.172, unif=5 1.012–1.880, const=1 1.040–6.856. So do C8's
  gains (1.490 [1.484, 1.497], 2.242 [2.222, 2.261], 4.761 [4.685, 4.833])
  and C9's factors (247.5 [199, 325] … 9.23 [8.65, 9.85] at p = 5e-4;
  52.9 … 2.95 at p = 2e-3).

### What is not: the label "same information"

The mld2 frontier uses **every** check round. unif=2, unif=5 and const=1
read only some rounds' raw syndromes. That is the same *measurement record*,
not the same *information as the policy*.

To separate the two effects I built a second frontier from **only the rounds
each policy reads**:

- **Detectors.** `policy_info_circuit` keeps the data-5 circuit's noise and
  observable and replaces its detectors with difference detectors between
  consecutive *used* rounds (raw detectors for the first used round).
- **Checks.** Stim accepts them as deterministic. LRB-D acceptance equals
  "all of them zero" bitwise. The lead's transfer-matrix decoder then gives
  the exact posterior given what the policy sees.
- **Comparison.** On identical fresh shots (30 sequences × 1e5) I compared
  LRB-D with both frontiers at equal pooled acceptance. Shots were ranked
  by exact \|LLR\|, with no bins, the boundary tie randomised, and 200
  sequence-bootstrap resamples (`fair.csv`):

FAIR_TABLE_PLACEHOLDER

### Required fix for C6 and C8

- **Wording.** Write "same measurement record (all check rounds)" instead of
  "same information".
- **What the factors measure.** On its own information, each sparse LRB-D
  policy is exactly a gap-rule point, like unif=1. So the 1.0–1.9×
  (unif=2, unif=5) and up to 6.9× (const=1) factors measure the value of
  the check rounds the policy ignores, used by the soft-output decoder. They
  do not measure soft versus hard decisions.
- **The 0.99 lower end for unif=2.** It is a binning artifact; the exact
  per-circuit minimum is 1.01. "Dominated by 0.99×" should read 1.01–1.17×.
- **"vs mld5 by 3–15×".** That is L0's subset (p = 1e-3 and 2e-3, m = 22).
  Over the stated window (p ≤ 2e-3, m ≥ 6) the mld5 factors span 2.75–97×
  (unif=2), 3.9–67× (unif=5) and 7.0–58× (const=1).

## 7. C14 and C16: tl_d5 on identical fresh shots

Setting: p = 2e-3, 30 sequences × 1e5 fresh shots per depth. The rule is my
own tl_d5; the comparison is the pooled exact mld5 gap rule (common
threshold, no bins), with 200 sequence-bootstrap resamples (`c14.csv`).

| m | tl_d5 acc / error | unif=1+final acc / error | tl / unif1+final | gap acceptance gain at tl error [95%] | same, calibrated q | tl error / gap error at tl acceptance | E |
|---|---|---|---|---|---|---|---|
| 0 | 0.9337 / 3.16e-4 | 0.8704 / 1.51e-4 | 1.073 [1.072, 1.073] | 1.000 [1.000, 1.000] | 1.001 | 1.000 | 0.934 / 3.12e-4; 1.1×; 1.00× |
| 6 | 0.4122 / 3.54e-3 | 0.2474 / 1.48e-3 | 1.666 [1.661, 1.671] | 1.028 [1.022, 1.035] | 1.032 | 1.066 [1.049, 1.083] | 0.412 / 3.55e-3; 1.7×; 1.03× [1.03, 1.04] |
| 22 | 0.0439 / 1.12e-2 | 0.0086 / 5.9e-3 (exact: 0.00870 / 4.76e-3) | 5.107 [5.054, 5.163] | 1.183 [1.135, 1.241] | 1.233 | 1.134 [1.095, 1.172] | 0.0442 / 1.12e-2; 5.1×; 1.26× [1.22, 1.31] |

- **At m = 0, tl_d5 is a frontier point.** In 30/30 circuits its accepted
  set is exactly the per-circuit top-n by \|LLR\|, with a strict boundary.
  This confirms E's "identical accepted shots".
- **The 5.1× ratio is confirmed.**
- **The equal-error gain at m = 22 is 1.18 [1.14, 1.24] (realised errors)
  or 1.23 (calibrated q), against E's 1.26 [1.22, 1.31].**
  - The intervals overlap, so the claim holds within uncertainty.
  - The metric is noisy: the frontier's error rate changes slowly with
    acceptance, so a 2.6% error-rate uncertainty becomes about 4% in
    acceptance.
  - Fix: quote "about 1.2–1.3×".
- **My unif=1+final error at m = 22 (5.9e-3) is a 2.6σ shot-noise
  fluctuation.** 152 errors were observed against 123 expected. The lead's
  exact sweep (`exact.exact_all_rounds`) gives 0.00870 / 4.76e-3, matching
  E's grid and hybrid values. The acceptance ratios above do not depend on
  it.
- **C16 for tl_d5.** The MLD decision on tl_d5's accepted shots makes
  exactly as many errors as tl_d5's identity correction: 886, 4,372 and
  1,471 at m = 0, 6, 22.

## 8. Other claims

- **C1 (VA).**
  - The fractional-knapsack proof is correct per circuit. Randomisation is
    needed only on the boundary tie group.
  - It also holds for the pooled problem, where the optimum is one common
    threshold: q is comparable across sequences when each sequence's
    posterior is exact.
  - **Scope caveat:** the theorem is about accepted error mass (or rate).
    It does not cover the repo's figure of merit, the mean over sequences of
    per-sequence F, at matched pooled acceptance. Nor does it cover f fitted
    from that. E's §2.5 matched-acceptance fits are therefore empirical
    comparisons, not consequences of C1.
  - **LIT:** Seip, Colmenarez, Schmitt, Müller, "Machine Learning Optimal
    Quantum Error Correction Thresholds", arXiv:2606.22194 (v1,
    20 June 2026), §II.4, Theorem II.1 (fetched). Discarding when
    max_λ p(λ|s) ≤ c is optimal at a given abort probability. With two
    logical classes, max_λ p(λ|s) = 1 − q(s), so this is the \|LLR\|
    threshold. I did not re-fetch Chow 1970; C1 does not need it.
- **C2 (VA, VN).**
  - The posterior is exact by construction: same DEM for decoder and
    sampler, and Stim's DEM is exact for DEPOLARIZE1/2 and X_ERROR.
  - Direct test (`calibration_c14.csv`, `calibration_fair.csv`): in
    0.5-wide \|LLR\| bins, observed MLD errors against Σ q(\|LLR_i\|),
    Bernoulli variance. CALIB_PLACEHOLDER
- **C3 (VN).**
  - E's `np_test_summary.csv` re-read: 10,715 comparisons, none below the
    optimistic bound by > 3σ (min z = −1.90).
  - The 272 points below the conservative curve all sit at p ≥ 7.5e-3, or
    on the frontier itself, as E explains.
  - My exact identical-shot test (§6) has every LRB-D / mld2 ratio ≥ 1
    within its interval.
  - Only 3,175 of the 10,715 rows are stored (`np_test_grid.csv` keeps
    DETAIL_POLICIES only). The summary is still traceable.
- **C5 (VA, VN).**
  - The derivation is correct. Detector-free flipping mechanisms are
    independent of everything else, so P(L|s) = (1−u)P′(L|s) + uP′(1−L|s).
  - Exhaustive tables: the cap holds and min q(s) ≥ u in every case.
  - The all-zero syndrome sits 0.0025 below the cap at p = 1e-4, m = 0.
  - u/p from the DEM is 4.532 at 1e-4, 4.504 at 2e-3 and 4.386 at 1e-2.
- **C7 (VN).**
  - Eight of the nine L0 rows are in `matched_points.csv` /
    `np_test_grid.csv` and match E's table.
  - The unif=4 row (4.70 vs mld5, 1.40 vs mld2) is in no stored CSV.
    Recomputed from the grid histograms with my own interpolation it gives
    4.698 and 1.398 (optimistic 4.858 / 1.422), so the row is correct.
  - Fix: add lrbd_unif_4 to MATCH_POLICIES or DETAIL_POLICIES.

## 9. Required fixes (summary)

1. **C6/C8 framing.**
   - Replace "same information" with "same measurement record (all check
     rounds)".
   - Add that against the frontier from only the rounds each policy reads,
     sparse LRB-D points are exactly on it (§6). The quoted factors are the
     value of the ignored rounds.
   - Replace unif=2's "0.99" lower end by the exact 1.01.
   - Replace "vs mld5 by 3–15×" by the window's actual 2.75–97×, or
     restrict it to L0's rows.
2. **C4 evidence and wording.**
   - The 1,414/1,414 count tests equal error indicators, not equal accepted
     sets. Add a direct set comparison or reword.
   - The "because" is verified, not structural; margins go down to 0.02.
3. **C14.** Quote the equal-error gain at m = 22 as about 1.2–1.3×. It is
   very sensitive to shot noise in the error rate.
4. **Small items.**
   - C1: state the scope (error rate, not sequence-averaged F or fitted f).
   - C7: store the unif=4 row.
   - C9: say "ideal readout".
   - C11: name the metric behind "closest".
   - C12: the inverse-half twin wording.
   - C13: the redundant lone-Z clause.

No numerical result of E was refuted. The one substantive correction is
interpretive: C6's "same-information domination" is domination by
information the sparse policies discard.

## 10. Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest -q tests/test_qubit512_pareto.py                 # 22 passed
# light parts (~1 min on 16 cores of the shared allocation)
for c in c4 c13 c5c10 c12; do python studies/correction-decoders/qubit/scripts/verify_pareto.py $c --workers 16; done
# identical-shot runs (~2 and ~10 min with 4 workers on the login node)
python studies/correction-decoders/qubit/scripts/verify_pareto.py c14  --workers 4 --c14-shots 100000 --n-boot 200
python studies/correction-decoders/qubit/scripts/verify_pareto.py fair --workers 4 --fair-shots 100000 --n-boot 200
```
