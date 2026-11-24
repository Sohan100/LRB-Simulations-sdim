# WP D verification: causal vs offline decoding, Pauli frame (adversarial review)

Reviewed: `notes/D-causal-offline.md`, `src/lrb_qubit/decoders_causal.py`,
`studies/correction-decoders/qubit/scripts/causal_vs_offline.py`, `results/D/`.
My code: `studies/correction-decoders/qubit/scripts/verify_causal.py` (subcommands `tamper`, `c1`, `c10`, `c3`,
`report`). My outputs: `results/D-verify/`. I did not edit any implementer or lead file.

Labels: **[VN]** verified numerically, **[VA]** verified analytically.

## Verdicts

| claim | verdict | evidence (details in the sections below) |
|---|---|---|
| C1 committed-decoder ratios | **Magnitudes confirmed. The quoted precision at w = 0 is refuted. Fix required.** | On the 30 study sequences (independent shots), the implementer's decoder gives 1.192 ± 0.004 (w = 0), 1.087 ± 0.002 (w = 1) and 1.042 (w = 4) at p = 5e-3; claimed 1.188, 1.086, 1.041. My independent decoder makes the same decisions except at **exact posterior ties in l_t**. These ties make up 1.5% of w = 0 commits, and the implementer breaks them by floating-point round-off. The tie rule moves the w = 0 ratio from 1.172 to 1.220 (paired SE ≤ 0.003). The implementer's value sits 0.020 ± 0.001 above the rule its `argmax` intends, about 7× the quoted SE. At w = 1 the spread is 1.086–1.092. |
| C2 w = 4 equals unlimited lookahead | **Confirmed** at p = 5e-3 and 2e-3 | On identical shots, w = 4 minus w = ∞ errors are +69 ± 109 over 4.6M shot-circuits; ratios 1.0419 vs 1.0418 (study sequences). |
| C3 the soft filter's frame-error rate equals the smoother's | **Confirmed** | Independent method (the lead's MLD on modified DEMs): lag-0/smooth = 0.9965 ± 0.0029, 1.0033 ± 0.0015, 0.9922 ± 0.0041. |
| C5 the offline f is the smoothed per-round rate, equal to the soft real-time rate | **Refuted as stated.** The practical conclusion holds for a simpler, exact reason. Fix required. | −ln f_offline exceeds the smoothed per-slot rate by 8.5%, 6.2% and 4.9% (implementer's own numbers) and by 6.9% and 5.0% (mine). Real-time achievability follows trivially: the offline MLD is itself a forward filter that decides once, at readout. |
| C6 rate changes, shape does not | **Needs more evidence** | κ differences up to 2.4σ, all positive at p = 2e-3 (correlated). At p = 5e-3 the late windows are noise-dominated (F̄ < 0.03), in their data and in mine. |
| C8 commits are exact and causal | **Confirmed** | Tamper test: 0 violations. The power check and negative control behave as expected, and streaming timing is exact. My brute-force decoder reproduces every non-tie decision over 19.6M shot-circuits. |
| C9 analytic Pauli-frame argument | **Confirmed [VA]** | Standard argument; its hypotheses are listed correctly. |
| C10 exact equality over 900 circuits | **Confirmed** | 360 of my own exact comparisons (Stim `CX/CY/CZ rec` feedback against my own frame propagation) give max \|ΔP\| = 0.0, 255 of them non-trivial. Rule A on study sequences matches their CSV to ≤ 5e-11. |
| C11 0 mismatches in 4.5M shots | **Confirmed** | 11.25M shot pairs (non-linear AND rules): 0 output mismatches. Every measurement record also satisfies rec_phys = rec_frame ⊕ frame flip. |
| C12 noisy correction gates break the equivalence | **Confirmed** (true by construction) | ΔP for rule A on new sequences: 6.4e-5 to 8.2e-4 (q = 0.1p), 6.4e-4 to 8.1e-3 (q = p). The size does not depend on the rule: a logically inert rule gets 3.3e-5 to 8.7e-4. |
| C13 non-Clifford gates break the naive frame | **Confirmed [VA]** | Own derivation: naive P(+1) = cos²(π/8) − ε/√2; physical correction = frame plus the S_L† fix-up. |
| C14 leakage breaks it | **Confirmed as a toy model. The hardware statement is unsupported** (the implementer labels it CONJ, correctly) | The difference is ελ by construction of the toy. |
| C15 a naive lookup can be worse than no correction | **Confirmed** | Rule A is worse than no correction in 30/30 sequences at all 15 (p, m) of their exact CSV. My rules A, Y and delayed are worse in all 255 non-trivial configurations. |

Not checked: C4, C7 (CONJ), C16, and the C1 entries at p = 5e-4.

## 1. Tests [VN]

`python -m pytest tests/test_qubit512_causal.py`: **64 passed** (93.8 s, shared allocation).
The lead's `tests/test_lrb_qubit_model.py` now also collects 64 tests, so the note's
"the lead's 64 tests pass" is consistent with the current file (API.md still says 40).

## 2. Does the committed decoder commit? [VN]

`verify_causal.py tamper` (`results/D-verify/tamper.csv`) was run on a new sequence at
p = 5e-3, depths 6 and 12, with 2000 shots. It tests the implementer's
`CausalWindowMLD` (state and logical commit, w = 0, 1, 2, 4) and my own decoder
(w = 0, 1). For every slot t:

- **Future tampering.** The detection events of all slots > t + w were changed in three
  ways: randomised, all flipped, and all set to 1. This gave **0 changed commits**
  (l_t′, b_t′ for t′ ≤ t) in all 20 decoder × depth cases. Changing only slot t + w + 1
  also gave 0 changes.
- **Power check.** Changing only slot t + w, the last slot inside the window, changes the
  slot-t commit in 67–70% (w = 0), 20–25% (w = 1), 10–16% (w = 2) and 8–14% (w = 4) of
  shots on average. The decoder therefore uses its full lookahead.
- **Negative control.** The same harness, applied to a w + 1 decoder against the w
  horizon, flags 6713–14046 (w = 0), 1282–3364 (w = 1), 271–862 (w = 2) and 8–79 (w = 4)
  changed commits. The harness detects a one-slot leak.
- **Streaming.** `CausalStream.push` commits slot t exactly when slot min(t + w, last)
  arrives. A commit never changes afterwards, and streaming equals batch decoding
  (8/8 checks).
- **Independent decoder.** My decoder (§3.1) uses only the committed state and the
  window by construction. It reproduces every non-tie decision of the implementer's
  decoder over 19.6M shot-circuits.

The decoder really commits and is causal.

## 3. Independent reproduction of C1 at p = 5e-3 (w = 0, 1)

### 3.1 My implementation (no code shared with `decoders_causal.py`)

`WindowCommitDecoder` works as follows.

- **Window.** For slot t it builds, by brute-force Z2 convolution, the exact joint
  distribution of the window. The window holds the slot t−1 leftover, slots t..t+w,
  b_t and Δl_t, over the DEM mechanisms whose latest slot lies in [t, t+w+1]. Slot
  t+w+1 is marginalised.
- **Commit.** Mechanisms whose latest slot is ≤ t−1 are summarised by the committed
  (b_{t−1}, l_{t−1}). Their only effect is that the observed slot t−1 bits become
  s_{t−1} ⊕ b_{t−1}. The decoder commits the mode over (l_t = l_{t−1} ⊕ Δl_t, b_t).
- **Inputs.** Detector slots come from the detector coordinates, and the DEM is parsed
  independently. There is no message passing.
- **Reference.** The offline reference is the lead's `TransferMatrixMLD`, on the same
  shots.

### 3.2 Agreement with the implementer: every disagreement is an exact tie [VN]

On identical shots, the two decoders' final predictions differ in 6.1% (w = 0) and
1.1% (w = 1) of shot-circuits. For **every** differing shot, the first differing slot
is a tie in l_t:

- in my posterior, the top-two gap at that slot is ≤ 4.4e-16 (relative) in all runs;
- in the implementer's own stored posterior (a diagnostic on two circuits, depths 3
  and 8 at p = 5e-3, 20,000 shots each), its choice and mine differ by ≤ 2.2e-16.

Such exact ties occur in 1.5% of w = 0 commits and 0.36% of w = 1 commits (p = 5e-3).
They come from symmetric fault classes, for example idle errors on the two members of
an ambiguous pair, whose summed probabilities are exactly equal.

The implementer's `choice = flat.argmax(axis=1)` *intends* first-index tie-breaking,
that is, prefer l_t = 0 (the "l0" rule). But α·β products of mathematically equal
entries differ by one ulp, so ties are in practice broken by floating-point round-off.

### 3.3 Ratios (1 − f)/(1 − f_offline), repo-style fit (offset 0, weights 1/std²)

**Five NEW sequences** (seeds 990000+i), all depths 0..22, 40,000 shots per circuit
(4.6M shot-circuits). Offline f = 0.86383.

| decoder | ratio | SE (paired bootstrap / jackknife) | per-sequence mean ± SE | claimed |
|---|---|---|---|---|
| mine, w = 0 (l0 ties) | 1.227 | 0.039 / 0.040 | 1.176 ± 0.013 | 1.188 ± 0.003 |
| implementer, w = 0 | 1.222 | 0.031 / 0.029 | 1.208 ± 0.015 | |
| mine, w = 1 (l0 ties) | 1.091 | 0.025 / 0.024 | 1.082 ± 0.007 | 1.086 ± 0.002 |
| implementer, w = 1 | 1.091 | 0.022 / 0.025 | 1.084 ± 0.007 | |
| implementer, w = 4 / w = ∞ | 1.048 / 1.047 | 0.023 / 0.015 and 0.024 / 0.012 | | 1.041 / 1.040 |

These agree with C1 within errors. With five sequences, however, the weighted pooled fit
is unstable: its weights come from a 5-sample std, which is why the SEs are 0.02–0.04.
So I repeated the run on the implementer's **30 study sequences**, with independent
shots (seed 555000+…), 20,000 shots per circuit and all depths (13.8M shot-circuits).
Offline f = 0.85749; the claimed value is 0.85722 ± 0.00122.

| decoder | ratio ± paired-bootstrap SE | claimed |
|---|---|---|
| implementer, w = 0 | 1.192 ± 0.004 | 1.188 ± 0.003 |
| implementer, w = 1 | 1.087 ± 0.002 | 1.086 ± 0.002 |
| implementer, w = 4 | 1.042 ± 0.002 | 1.041 ± 0.002 |
| implementer, w = ∞ | 1.042 ± 0.002 | 1.040 ± 0.002 |
| **mine, w = 0, l0 ties (the rule the code intends)** | **1.172 ± 0.004** | |
| mine, w = 1, l0 ties | 1.086 ± 0.002 | |

The implementer's numbers reproduce with fresh shots. The decoder, however, is not
uniquely defined (§3.4).

### 3.4 Tie-breaking sensitivity (main finding) [VN]

All rules below are valid "posterior-mode" commits; they differ only on exactly tied
l_t. `l0` prefers l_t = 0, `l1` prefers l_t = 1, `keep` prefers l_t = l_{t−1}, `flip`
prefers l_t ≠ l_{t−1}, and `random` picks one at random. Ties in b_t go to the smallest
index. All rows use the 30 study sequences at p = 5e-3 on identical shots, so the shifts
are paired.

| w = 0 tie rule | pooled ratio | shift vs l0 | excess error over offline at m = 1 | m = 2 | m = 4 | m = 10 |
|---|---|---|---|---|---|---|
| l0 | 1.172 | 0 | 0.0086 | 0.0142 | 0.0219 | 0.0247 |
| implementer (round-off) | 1.192 | +0.020 ± 0.001 | 0.0123 | 0.0199 | 0.0269 | 0.0269 |
| flip | 1.192 | +0.021 ± 0.002 | 0.0121 | 0.0200 | 0.0284 | 0.0265 |
| random | 1.193 | +0.022 ± 0.002 | 0.0139 | 0.0219 | 0.0293 | 0.0275 |
| keep | 1.197 | +0.025 ± 0.002 | 0.0159 | 0.0230 | 0.0299 | 0.0279 |
| l1 | 1.220 | +0.049 ± 0.003 | 0.0194 | 0.0288 | 0.0364 | 0.0298 |

The excess-error SEs are 0.0003–0.0007 (McNemar).

- **Per sequence.** The implementer's ratio minus l0 is +0.028 ± 0.001, positive in 30/30
  sequences. l1 minus l0 is +0.063 ± 0.003 (30/30).
- **Cost of committing.** The tie rule alone changes the "cost of committing" at m = 1 by
  a factor of 2.3 (0.0086 to 0.0194).
- **w = 1.** The pooled ratios are l0 1.086, implementer 1.087 (+0.0011 ± 0.0004), keep,
  random and flip 1.089, and l1 1.092 (+0.0052 ± 0.0010). The effect is small, about the
  size of the quoted SE.
- **p = 2e-3 spot check** (5 new sequences, 12 depths, 20,000 shots). The implementer's
  w = 0 gives 1.192 ± 0.032 (claimed 1.165 ± 0.005); mine gives 1.183 (l0) and 1.213
  (l1). At w = 1: implementer 1.057, l0 1.054, l1 1.061 (claimed 1.065 ± 0.002). These
  are consistent within the 5-sequence errors, and the same tie ordering appears.

**Consequences for C1.**

- The w = 0 entries describe the implementer's round-off tie-breaking, not a defined
  decoder.
- The systematic uncertainty from the tie rule (±0.025 around the midpoint at
  p = 5e-3) is about 8× the quoted statistical SE.
- The deterministic rule the code intends would lower the p = 5e-3, w = 0 entry from
  1.188 to about 1.17.
- The "cost of committing" is therefore not a property of committing alone. Cheap
  changes to the decision rule move it substantially, which also weakens C2's reading
  that "the residual is the cost of committing hard decisions". For w ≥ 2 the tie
  dependence is untested here: the brute-force window is too large.

## 4. C10/C11 with my own Stim feedback (Pauli frame vs physical correction) [VN]

`verify_causal.py c10`, `results/D-verify/c10_*.csv`.

- **Exact, linear rules.** I tested four rules:
  - **A**, the covariant lookup (X0, X2, Z0, Z3);
  - **"other pair member"** (X3, X4, Z2, Z4);
  - **Y corrections** (S_Z1→Y0, S_X2→Y4, S_Z2→Y2, S_X1→Z1);
  - **delayed**: the corrections after round r are controlled by round r−1 records
    (→Z2, X4, Y3), plus S_X2(r)→Z4.

  The setup was:
  - grid: 3 p × 5 new sequences × m ∈ {0, 1, 2, 6, 10, 22} = 360 circuit pairs;
  - **physical circuit**: Stim `CX/CY/CZ rec[-k] q` inserted after each round's M8;
  - **frame circuit**: no feedback, with the observable set to x0 ⊕ x3 ⊕ the frame's
    parity of the controller's adjusted records;
  - the adjusted records come from my own propagation of each correction Pauli through
    the noiseless circuit (`stim.PauliString.after`, one operation at a time), which
    yields the measurement records it flips.

  Exact P_flip from each DEM: **max |ΔP| = 0.0**.
  - The comparison is non-trivial (the rule changes P_flip) in 255/360 cases: A and Y
    90/90, delayed 75/90.
  - The other-pair rule is logically inert, because ⟨X3, X4, Z2, Z4⟩ is mapped to itself
    by every logical gate and commutes with X0 X3.
- **Cross-check of the implementer's numbers.** For rule A on study sequences 0–2,
  3 p, m ∈ {2, 22}, my P_flip values match `results/D/frame_rule_a_exact.csv` to
  ≤ 5e-11 (its print precision). This covers the physical, frame, no-correction,
  noisy q = 0.1p and noisy q = p values.
- **Non-linear rules, identical noise** (`stim.FlipSimulator`, same seed). Grid: three
  rules × 3 p × 5 new sequences × m ∈ {1, 2, 6, 10, 22} × 50,000 shots, 11.25M shot
  pairs in total.
  - **0 output mismatches**, and the decisions are identical round by round.
  - **Every measurement record** satisfies rec_phys = rec_frame ⊕ (the frame's predicted
    flip). The noise realisations are therefore identical, so the masked Pauli
    injections do not consume randomness. I also checked this separately on a toy
    circuit.
- **The suggested rule is logically inert.** The suggested rule is "(1,1) on the X
  checks → Z1", with X1 for the Z checks. Every logical gate maps qubit 1 to itself
  (H→H1, S→S†1, X and Z do not touch it), so Z1 and X1 never reach the X0 X3 readout.
  Its P_flip equals no correction exactly. It still tests the record-level identity:
  2535 record rows are measured differently.
- **Logically active non-linear rules.** I added two:
  - (1,1) → Z0 / X3 plus (1,0)_X → Z2, which raises P_flip by +0.025 on average;
  - a two-round AND, +0.014.

  Both also show 0 mismatches.

## 5. C5: is the argument sound?

The claim: "the offline f is the per-round accumulation rate of logical-frame
uncertainty given all syndromes (smoothing). In the bulk it equals the soft-filter
rate, so the offline f is achievable in real time."

1. **The identity is false as a number.** f is fitted over circuits of depth m. Each
   depth step adds a forward Clifford, a check round **and one inverse Clifford, whose
   errors only the terminal check round sees**. The per-slot rate measures only
   the forward-slot accumulation in one depth-22 circuit.

   | p | −ln f_offline (implementer) | smoothed per-slot rate (implementer) | ratio | mine (5 new sequences) |
   |---|---|---|---|---|
   | 5e-4 | 0.01229 | 0.01132 | 1.085 | – |
   | 2e-3 | 0.05375 | 0.05060 | 1.062 | 1.069 |
   | 5e-3 | 0.15406 | 0.14690 | 1.049 | 1.050 |

   The gap is several σ and is reproduced independently. The notes themselves invoke
   the inverse half's extra per-step error in §4, where it "dilutes" the ratios. C5
   contradicts that.
2. **The conclusion is true for a simpler, exact reason.** The lead's
   `TransferMatrixMLD` is a forward filter: it processes slots in time order with
   32 numbers per shot, and outputs one decision when the final slot arrives. For
   Clifford-only LRB with Pauli-frame tracking, no decision is needed earlier. The
   real-time soft filter's LRB output is therefore identical to the offline MLD, shot
   by shot, not just equal within 0.1–0.3%.
3. **The per-slot evidence answers a different question.** C3 (confirmed) concerns
   intermediate estimates of l_t. It shows that waiting for future syndromes does not
   change the *accumulation rate* of intermediate frame errors. That is correct and
   useful, but it neither establishes nor is needed for the statement about f.
4. "Not a fixed per-step channel eigenvalue" and "not the rate of hard-decision
   decoders" are fair (the latter is shown by C1).

**Verdict:** refuted as stated (the identity), with the conclusion confirmed by
point 2. See fix 2.

## 6. Other claims

- **C3 [VN, independent method].** For each slot t of depth-22 circuits (5 new
  sequences, 20,000 shots, ground truth from DEM error sampling):
  - filter estimate of l_t: the lead's MLD on the DEM truncated to slots ≤ t + w. The
    mechanisms with latest slot t + w + 1 are cut to their slot t + w bits, and the
    observable counts only mechanisms with latest slot ≤ t.
  - smoother estimate: the full DEM with that observable.

  The bulk rate is fitted over slots 2..20.

  | p | lag 0 / smooth | lag 1 / smooth | claimed lag 0 |
  |---|---|---|---|
  | 5e-4 | 0.9965 ± 0.0029 | 0.9982 ± 0.0005 | 0.998 ± 0.001 |
  | 2e-3 | 1.0033 ± 0.0015 | 1.0005 ± 0.0009 | 0.999 ± 0.001 |
  | 5e-3 | 0.9922 ± 0.0041 | 0.9985 ± 0.0023 | 1.000 ± 0.003 |

  The filter's frame error exceeds the smoother's by a small constant: at slot 10,
  0.0542 vs 0.0531, 0.1958 vs 0.1935, and 0.3850 vs 0.3823. **Confirmed.**
- **C6. Needs more evidence.**
  - At p = 2e-3, the three causal decoders' κ − κ_offline are all positive (+0.036 ± 0.024,
    +0.032 ± 0.014, +0.017 ± 0.011). They are computed on the same shots, so they are
    correlated.
  - At p = 5e-3, F̄(22) ≈ 0.01–0.03, and the 14–22 windows are dominated by noise and
    by the |1 − 2e| bias. This holds in the implementer's data (±0.04–0.07 in κ) and in
    my 5-sequence data, where the offline κ is 1.26 ± 0.08.

  The data support "no shape change detected at the |Δκ| ≲ 0.05–0.1 level", not "the
  shape does not change".
- **C9 [VA]. Confirmed.** The hypotheses (Pauli noise independent of the decisions,
  Clifford gates, Z-basis measurements, corrections only on data qubits that are not
  reset, noiseless corrections) are the right ones. Equal outputs under identical noise
  then follow by induction for any rule acting on the adjusted record. My record-level
  identity in §4 is the direct numerical form of this induction.
- **C12. Confirmed.**
  - On new sequences, rule A gives ΔP = 6.4e-5 to 8.2e-4 (q = 0.1p) and 6.4e-4 to
    8.1e-3 (q = p); ×10 is just linearity.
  - In the always-played model ΔP measures the extra noise layer, not the correction:
    the logically inert rule gets ΔP = 3.3e-5 to 8.7e-4.
  - The conditional-gate FlipSimulator numbers were not rechecked.
- **C13 [VA]. Confirmed** by my own derivation. T_L X0 = X0 T_L†, so the naive frame
  gives (1 − ε)cos²(π/8) + ε sin²(π/8) = cos²(π/8) − ε/√2. Applying S_L† after T_L turns
  X0 T_L†|+_L⟩ into X0 T_L|+_L⟩, which restores the physical value. The fix-up needs the
  frame at the T gate.
- **C14.** The toy is exact and the difference ελ holds by construction. Hardware
  relevance remains a conjecture, as the implementer labels it.
- **C15. Confirmed.**
  - In the implementer's exact CSV, rule A is worse than no correction in 30/30
    sequences at every one of the 15 (p, m) points.
  - Every non-trivial rule I tested is worse than no correction in every configuration.
  - The mechanism holds. With the covariant table, rule A reduces to a terminal lookup
    that flips the readout iff S_X1 ⊕ S_X2 fires in the terminal round. This fixes Z0
    and Z3 but breaks Z2 and Z4 (a coin flip within each pair), and it converts every
    terminal X-ancilla measurement error into a logical error.
  - Note that "no correction" ignores the final data detectors.

## 7. Required fixes

1. **C1, define the decoder.**
   - Make tie-breaking explicit and immune to round-off: compare posterior entries with
     a relative tolerance (for example 1e-12), then apply a stated rule.
   - Re-run the grid and quote the tie-rule spread as a systematic uncertainty.
   - With the intended first-index rule (l0), the p = 5e-3 entries measured here are
     **w = 0: 1.172 ± 0.004** (not 1.188) and **w = 1: 1.086 ± 0.002**. The full spread
     over tie rules is 1.172–1.220 at w = 0 and 1.086–1.092 at w = 1.
   - Re-state C1 for "this commit-and-slide windowed MLD with tie rule X", not
     "committed causal decoders" in general. The decision rule alone changes the
     small-depth cost of committing by a factor of 2.3.
2. **C5, rewrite.**
   - Drop "the offline f is the smoothed per-round rate". The fitted f includes the
     unchecked inverse half, and −ln f exceeds the per-slot rate by 5–9%.
   - Justify real-time achievability by the exact argument: the offline MLD is a causal
     forward filter that decides once, at readout.
   - Keep C3 as a separate statement about intermediate frame estimates.
3. **C6, soften** to a quantified bound on |κ − κ_offline|. Mention the consistently
   positive sign at p = 2e-3. Drop the p = 5e-3 late windows, or gain more statistics
   for them.
4. **C2, interpretation.** "The residual 0.4–4% is the cost of committing hard
   decisions" should note that this residual depends on the tie rule. This is shown at
   w ≤ 1 and untested at w ≥ 2.
5. **C10/C11 and C12, remarks.**
   - State which feedback rules change P_flip. Corrections on qubit 1, or in
     ⟨X3, X4, Z2, Z4⟩, are logically inert for the X_L readout. A and A′ are not, so the
     implementer's test is meaningful.
   - State that the always-played noise penalty does not depend on the rule.

## 8. Reproduce

```bash
cd <repository root>
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest tests/test_qubit512_causal.py -q                     # 64 passed
# on the shared allocation: srun --jobid=<JOBID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c "<env>; <cmd>"
python studies/correction-decoders/qubit/scripts/verify_causal.py tamper --shots 2000                          # ~2 min, 1 process
python studies/correction-decoders/qubit/scripts/verify_causal.py c1 --workers 12 --shots 40000                # 52 s
python studies/correction-decoders/qubit/scripts/verify_causal.py c1 --study --num-seq 30 --shots 20000 --workers 16   # 2 min
python studies/correction-decoders/qubit/scripts/verify_causal.py c1 --p 0.002 --num-seq 5 --shots 20000 \
    --depths 0,1,2,3,4,6,8,10,14,18,20,22 --workers 16                               # 10 s
python studies/correction-decoders/qubit/scripts/verify_causal.py c10 --shots 50000                            # 80 s
python studies/correction-decoders/qubit/scripts/verify_causal.py c3 --workers 2 --shots 20000 --num-seq 5     # < 1 min
python studies/correction-decoders/qubit/scripts/verify_causal.py report --p 0.005 --boot 300
python studies/correction-decoders/qubit/scripts/verify_causal.py report --p 0.002 --boot 300
```

Outputs are in `results/D-verify/`:

| file | contents |
|---|---|
| `tamper.csv` | causality checks |
| `c1_p0.005.json`, `c1_p0.002.json`, `c1_study_p0.005.json` | raw counts and agreement diagnostics |
| `c1*_fits_*.csv` | fits and ratios |
| `c1*_tie_sensitivity_*.csv` | tie-rule shifts, per-sequence ratios, excess errors |
| `c1*_errors_*.csv` | error rates per decoder and depth |
| `c10_exact_linear.csv`, `c10_crosscheck_ruleA_vs_impl_csv.csv`, `c10_flipsim_nonlinear.csv` | Pauli-frame checks |
| `c3_filter_vs_smoother.csv` | C3 check |
| `summary_p*.json` | summaries |
| `logs/` | run logs |

Seeds:

- new sequences: `default_rng(990000 + i)`;
- c1 shots: 555000 + 1000·i + m + p·1e7;
- c10 FlipSimulator: 700000 + 1000·i + 17·m + p·1e6;
- c3: 880000 + 1000·i + p·1e6.
