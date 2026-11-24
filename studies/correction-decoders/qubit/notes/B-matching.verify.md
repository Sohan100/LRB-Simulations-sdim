# B (matching decoders): adversarial verification

Independent verifier of work package B. I tried to refute H1–H6 below.

- My script: `studies/correction-decoders/qubit/scripts/verify_matching.py`. It never imports B's
  `src/lrb_qubit/decoders_matching.py` or `studies/correction-decoders/qubit/scripts/matching_vs_mld.py`.
  It uses Stim 1.15.0, PyMatching 2.3.1 and the lead's `circuits`, `code`,
  `experiment` and `mld` modules, and reads B's CSVs only to compare.
- My data: `studies/correction-decoders/qubit/results/B-verify/`:
  - `verify_report.json`: census (H1), H4, H6 and the m ≤ 2 / PyMatching-only
    sampling;
  - `verify_sweep.json`: sampling at m = 6 and 22 with my min-sum sweep;
  - `verify_validate.json`: sweep against brute force;
  - `census_rows.csv` and `h4_rows.csv`: per-circuit rows.
- I found no bug in B's code. Every quantity I re-derived with independent
  code matches B's CSVs, on B's circuits, to rounding or shot noise. The
  verdicts below concern how the claims are worded and how far they
  generalize.
- New Clifford sequences: `experiment.clifford_sequences(20, 22, 5550000)`
  (none equals one of B's). New sampler seeds:
  31,337,000,000 + 1e8·k + 1000·seq + m (B used 202,610,282 + …).

Labels: **[VA]** verified analytically, **[VN]** verified numerically,
**[CONJ]** conjecture. Units of p unless stated. ± is one standard error.

## Verdicts

| claim | verdict | main evidence |
|---|---|---|
| H1 first-order slopes (MLD 11.79, MWE 11.79, MWPM 24.48, correlated 25.06, none 21.11 p) | **confirmed**; one sentence needs a qualifier | My census reproduces B's `first_order.csv` to 5e-4 p. On 20 new sequences: MLD 11.85 ± 0.18, MWE 11.85 ± 0.18 (an actual hypergraph MWE decoder, either tie rule), MWPM 24.54 ± 0.38, correlated 25.12 ± 0.38, none 21.17 ± 0.20. "Worse than no decoding for every depth ≥ 2" holds for the sequence mean and per step, but not per sequence at m ≤ 4. |
| H2 MWPM/MLD 1.10–1.79 wherever LER_MLD < 0.4; MWE/MLD 1.000–1.022 | **confirmed on B's grid; refuted as worded** | New-seed resamples of 7 B points agree at shot-noise level (\|z\| ≤ 1.8), including both ends of the range and MWE/MLD's maximum. But at p = 1e-4 (LER_MLD ≈ 0.01), on B's own sequences, MWPM/MLD = 1.818 ± 0.006 (m = 6) and 1.939 ± 0.004 (m = 14). As p → 0 it tends to 1.84–2.03. |
| H3 ladder: merge 22–31%, decomposition 58–74%, degeneracy + ties 0.3–14% | **needs-more-evidence** (core split confirmed) | Re-derived independently at 6 B points to ≤ 1.4 pp, and 33/67/0 per step at first order. But the ranges leave out m = 0 (merge ≈ 0, decomposition 84–98%) and carry ±2–11 pp SEs. The tie rule shifts them by up to 4.6 pp, and a new-sequence point gives merge = 37%. |
| H4 L0-only components (2.9p–22.8p); plain drops silently; correlated raises | **confirmed** (fix the wording of the range) | Minimal DEM plus 120 circuits. Plain PyMatching gives 0 warnings, and its graph is identical to the stripped-DEM graph. Correlated PyMatching raises `ValueError` iff such lines exist. 2.9p and 22.8p are the means at m = 2 and m = 22; single circuits range over 0–31p (B's sequences) and 0–50p (new ones). |
| H5 MWPM own gap overconfident (6–8 nats fails 29%); MLD calibrated | **confirmed** | Own gap by brute force (m = 2) and by my own sweep (m = 6, 22): 0.290 failure vs 9.8e-4 predicted at (5e-4, m = 6), the same as B. 18–32% at the other points checked. MLD is calibrated within statistics (worst \|z\| = 2.2 over 98 bins). The A0 factor is 27× and 43× on new sequences (B: 3–20×). |
| H6 physical RB 0.339 p_RB per step | **confirmed** (first order) | 61 tokens over 24 Cliffords gives 2 · (61/24) · 0.1 · (2/3) = 61/180 = 0.33889. Stim's DEM agrees to 1e-13. The exact value is 0.3383 at p_RB = 5e-3 and 0.3332 at 0.05. |

## 1. B's tests [VN]

`python -m pytest tests/test_qubit512_matching.py -q` on the shared allocation:
**33 passed in 21.75 s.**

## 2. H1: first-order error per depth step

**Method (my own code, `census_task`).** For each circuit
`build_lrb_circuit(seq, m, p, terminal="data5")` at p = 1e-5 and p = 1e-6,
every line i of the undecomposed DEM (probability q_i, detectors s_i, flip
l_i) is decoded on its own by:
- `mwpm`: PyMatching built from `detector_error_model(decompose_errors=True)`;
- `mwpm_corr`: correlated PyMatching, built from that DEM with its
  detector-free components deleted (it refuses the raw DEM, see H4);
- `mld`: the lead's `TransferMatrixMLD`;
- `none`: always "no flip";
- `mwe`: exact minimum-weight error on the undecomposed hypergraph, from my
  own min-sum sweep (`MinPlusSweep`, see H5), with ties broken both ways;
- `E1 = Σ_s min(P0(s), P1(s))`, the exact first-order MLD and MWE
  coefficient, read off the DEM directly.

The coefficient is c = Σ_i q_i [D(s_i) ≠ l_i] / p. Slopes are per-sequence
linear fits over m, then mean ± SEM over sequences. 30 sequences × 23 depths
× 2 values of p = 1380 circuits.

**Reproduction of B's numbers.** On B's 10 sequences my coefficients equal
`results/B/first_order.csv` to within 5.0e-4 p for MLD, MWE, MWPM,
correlated MWPM and none. That is the CSV's 6-digit rounding. [VN]

**Error per depth step (slope, units of p):**

| decoder | B's claim (10 seq) | B's 10 seq, my code | 20 new seq, m ∈ {0,2,6,14,22} | 20 new seq, all m = 0–22 | 20 new seq, p = 1e-6 |
|---|---:|---:|---:|---:|---:|
| exact MLD | 11.79 ± 0.18 | 11.79 ± 0.18 | 11.85 ± 0.18 | 11.84 ± 0.19 | 11.85 ± 0.18 |
| MWE (hypergraph decoder; = E1) | 11.79 | 11.79 ± 0.18 | 11.85 ± 0.18 | 11.84 ± 0.19 | 11.85 ± 0.18 |
| PyMatching MWPM | 24.48 ± 0.40 | 24.48 ± 0.40 | 24.54 ± 0.38 | 24.55 ± 0.41 | 24.54 ± 0.38 |
| correlated PyMatching | 25.06 ± 0.38 | 25.06 ± 0.38 | 25.12 ± 0.38 | 25.15 ± 0.43 | 25.11 ± 0.38 |
| no decoding | 21.11 ± 0.21 | 21.11 ± 0.21 | 21.17 ± 0.20 | 21.18 ± 0.22 | 21.17 ± 0.20 |

Every new-sequence value is within 0.3 SE of B's. The slopes do not depend
on the depth set or on p (1e-5 vs 1e-6), so they are converged first-order
quantities. [VN]

**MWE = MLD at first order** [VA, VN]. All 1380 DEMs have exactly one line
per (detectors, L0) symptom and no undetectable logical line. A single line
beats every pair of lines at p ≤ 1e-5 (largest single weight about 14.9 nats,
smallest pair weight about 19.8 nats). So on single-fault syndromes MWE picks
the larger of P0(s) and P1(s), and exact ties cost the same either way. MLD
picks the first-order argmax on every untied symptom (zero exceptions), and
|c_MLD − E1| ≤ 5.3e-6 p, an O(p) effect. The actual hypergraph MWE decoder
gives |c_MWE − E1| ≤ 6e-9 p over all 1380 circuits, with ties broken either
way.

**"So MWPM is worse than no decoding for every depth ≥ 2."** This is true for
the sequence average, and per step, but not per sequence at small depth:

| m | 1 | 2 | 3 | 4 | 6 | 14 | 22 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B's 10 seq: mean c_MWPM − c_none | 1.7 ± 1.9 | 4.8 ± 3.5 | 3.0 ± 2.4 | 6.8 ± 4.4 | 13.8 ± 4.3 | 41.8 ± 5.4 | 73.6 ± 6.7 |
| B's 10 seq: sequences where MWPM beats none | 6/10 | 5/10 | 4/10 | 4/10 | 2/10 | 0/10 | 0/10 |
| 20 new seq: mean c_MWPM − c_none | 2.2 ± 1.2 | 7.1 ± 1.9 | 7.2 ± 2.2 | 12.4 ± 2.3 | 19.6 ± 2.8 | 46.3 ± 4.3 | 74.2 ± 5.8 |
| 20 new seq: sequences where MWPM beats none | 12/20 | 8/20 | 6/20 | 3/20 | 0/20 | 0/20 | 0/20 |

- The mean is positive at every m ≥ 1 in both sets. For B's own 10 sequences
  it is below 2σ at m = 1–4 (1.4σ at m = 2). The 20 new sequences make it
  3.8σ at m = 2.
- Per step it holds for every sequence. The per-sequence slope of
  c_MWPM − c_none is positive for all 30 sequences: 3.37 ± 0.36 (B's 10, min
  1.10) and 3.37 ± 0.29 (new 20, min 1.41) per step.
- At a fixed small depth it does not hold per sequence. At m = 2, MWPM beats
  no decoding on 5 of B's 10 sequences and on 8 of the 20 new ones.

## 3. H2: sampled LER ratios [VN]

**Method.** New sampler seeds, 1e6 shots per circuit and 10 sequences per
point (1e7 shots). On identical shots I decode with:
- PyMatching (plain, `decompose_errors=True`);
- the lead's MLD;
- the hypergraph MWE, by brute force at m ≤ 2 and by my sweep at m = 6
  and 22.

The circuits are B's, with independent shots, so B's value and mine should
agree within shot noise. z is (mine − B) / (√2 × my shot-noise SE).

| p | m | MWPM/MLD, B | MWPM/MLD, mine | z | MWE/MLD, B | MWE/MLD, mine | z |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 2 | 1.614 | 1.615 ± 0.004 | +0.2 | 1.0038 | 1.0016 ± 0.0012 | −1.3 |
| 5e-4 | 6 | 1.729 | 1.733 ± 0.002 | +1.2 | 1.0022 | 1.0039 ± 0.0007 | +1.9 |
| 5e-4 | 14 (B's max) | 1.786 | 1.782 ± 0.002 | −1.5 | 1.0043 | – | – |
| 5e-4 | 22 | 1.742 | 1.743 ± 0.002 | +0.3 | 1.0047 | 1.0043 ± 0.0004 | −0.8 |
| 2e-3 | 6 | 1.509 | 1.510 ± 0.001 | +0.9 | 1.0136 | 1.0140 ± 0.0004 | +0.6 |
| 1e-2 | 0 (B's min) | 1.101 | 1.102 ± 0.001 | +0.2 | 1.0177 | 1.0165 ± 0.0008 | −1.0 |
| 1e-2 | 2 | 1.193 | 1.195 ± 0.001 | +1.8 | 1.0222 (B's max) | 1.0218 ± 0.0004 | −0.7 |

A second, independent resample of (2e-3, m = 6) gives 1.511 (z = +1.8).

**New sequences** (10 of mine, jackknife SE over sequences):
- (5e-4, m = 6): MWPM/MLD 1.793 ± 0.045 against B's 1.729 ± 0.046, and
  MWE/MLD 1.0052 ± 0.0006.
- (2e-3, m = 2): MWPM/MLD 1.590 ± 0.063 against B's 1.500 ± 0.061, and
  MWE/MLD 1.0126 ± 0.0019.

Both are consistent with B within the sequence spread. The first sits just
above 1.79.

**Below B's grid** (B's own 10 sequences, p = 1e-4, 1e7 shots):

| m | LER_MLD | MWPM/MLD | first-order limit (B's `first_order.csv`) |
|---:|---:|---:|---:|
| 6 | 0.0084 | **1.818** ± 0.006 (shot), ± 0.051 (jackknife) | 1.844 |
| 14 | 0.0174 | **1.939** ± 0.004 (shot), ± 0.037 (jackknife) | 1.985 |
| 22 | – | – | 2.032 |

- With LER_MLD around 0.01–0.02 the ratio exceeds 1.79 by about 5σ and 35σ
  (shot noise, same circuits).
- As p → 0 it tends to the first-order ratio: 1.84, 1.99 and 2.03 at m = 6,
  14 and 22 (1.94, 2.01 and 2.03 on the new sequences).
- So "1.10–1.79" describes the sampled grid p ∈ [5e-4, 1e-2], not "wherever
  LER_MLD < 0.4". B's own summary ("about 2× worse") is the better
  statement.
- MWE/MLD stays inside 1.0016–1.0218 in every resample. It tends to exactly 1
  as p → 0 (H1). B's own value at (1e-2, m = 22) is 0.99995, so the lower end
  "1.000" holds only within noise.

## 4. H3: attribution of the MWPM − MLD gap [VN]

**Method.** The ladder is re-derived on identical shots with decoders
independent of B's code:
- MWPM: PyMatching.
- Class-aware graph: exact min-sum on the decomposed components merged per
  (detectors, L0) class. Brute force at m ≤ 2, my sweep at m ≥ 6. Ties go to
  "no flip", as in B.
- MWE: the same methods on the undecomposed DEM.
- MLD: the lead's `TransferMatrixMLD`.

Shares are merge / decomposition / degeneracy + ties, in %, with 1e7
new-seed shots per point:

| point | mine | B (its `per_circuit.csv`) | MWPM − MLD, 1e-3 (mine) |
|---|---|---|---:|
| 5e-4, m = 2 | 25.2 / 74.5 / 0.3 | 26.0 / 73.4 / 0.6 | 11.5 ± 1.7 |
| 5e-4, m = 6 | 25.2 / 74.3 / 0.5 | 25.6 / 74.1 / 0.3 | 30.5 ± 1.8 |
| 5e-4, m = 22 | 29.5 / 69.9 / 0.6 | 29.6 / 69.7 / 0.6 | 90.8 ± 1.9 |
| 2e-3, m = 6 | 24.6 / 72.7 / 2.7 | 25.0 / 72.4 / 2.7 | 79.7 ± 3.9 |
| 1e-2, m = 2 | 30.8 / 58.0 / 11.2 | 30.8 / 57.8 / 11.5 | 60.4 ± 5.6 |
| 1e-2, m = 0 | −1.1 / 84.9 / 16.3 | −1.0 / 83.5 / 17.5 | 10.57 ± 0.06 |
| new seq, 2e-3, m = 2 | **37.3** / 60.6 / 2.1 | – | 41.6 ± 4.3 |
| new seq, 5e-4, m = 6 | 24.8 / 74.6 / 0.7 | – | 31.6 ± 1.7 |

- First order, from the per-step slopes of the H1 census: MWPM 24.48,
  class-aware 20.28, MWE = MLD 11.79, so 33 / 67 / 0. The new sequences give
  the same 33 / 67 / 0.
- The largest deviation from B on identical circuits is the class-aware LER
  at (2e-3, m = 6): z = 2.8 in shot noise. On the same shots no decoding sits
  at z = 2.0, so I read it as a high draw of this shot sample.

**What does not hold up as stated:**

1. **Scope.** B's ranges span 10 of its 20 (p, m) rows. Over all 20 rows of
   B's own `per_circuit.csv`, the shares span −2 to 57% / 51 to 98% / −8 to
   17.5%.
   - At m = 0 the split is entirely different: merge ≈ 0, decomposition
     84–98%, degeneracy up to 17.5% (p = 1e-2, LER 0.10, not saturated).
   - (5e-3, m = 14) has merge 20%, and the saturated rows are erratic.
2. **Precision.**
   - The merge rung's jackknife SE is 1.1–4.6e-3, which is ±2–11 pp of
     share (±6–11 pp at m ≤ 6).
   - Per sequence the merge share runs from −5% to 57%. At (2e-3, m = 2) on
     new sequences it looks bimodal: 4–11% for five sequences and 49–57% for
     the other five.
   - So the pooled 22–31% is not a stable property. 10 new sequences give
     37%.
3. **Tie rule.** The class-aware graph and the hypergraph have exact weight
   ties on 0.6–15% of shots (15% for class-aware at p = 1e-2, m = 0). B's
   graph has none.
   - Breaking ties to "flip" instead of "no flip" moves the shares by
     ≤ 0.3 pp at p = 5e-4 and at m ≥ 6.
   - It moves them by 4.6 pp at (1e-2, m = 2), where merge goes from 30.8% to
     26.2%, and by 4 pp at (1e-2, m = 0). B's upper ends (31% and 14%) come
     from p = 1e-2 points, where ties are frequent.
   - At first order (sequence 0, m = 2) the class-aware coefficient is 76.63p
     (ties to no flip, B's rule), 74.96p (ties to flip) or 75.70p
     (PyMatching's own tie-break in my dummy-node graph). The per-step slopes
     are unaffected (20.28 vs B's 20.26).
4. A sequential ladder through one intermediate decoder is one attribution
   among several. [CONJ: another intermediate would split differently.]

The qualitative split is confirmed: at m ≥ 2 and low p, decomposition is
about 2/3–3/4, the merge rule about 1/4–1/3, and degeneracy is small. The
quoted ranges are not robust.

## 5. H4: detector-free components in Stim's decomposition [VN]

Minimal DEM (my `h4_minimal`): `error(0.1) D0 L0 ^ D1 ^ L0`, `error(0.01) D0`,
`error(0.01) D1`. The first mechanism flips D0 and D1 but not L0.
- Plain PyMatching builds it with 0 warnings. Its graph has L0 on the D0
  boundary edge and nothing on D1, so it decodes the syndrome (D0, D1) as a
  logical flip: the `^ L0` remainder is silently dropped.
- Correlated PyMatching raises `ValueError: Encountered a decomposed error
  instruction with an undetectable component (0 detectors). This is not
  supported.`

B's quoted example exists verbatim: sequence 0, m = 2, p = 2e-3 has
`error(0.0206292) D6 D8 L0 ^ D10 ^ L0`.

Census at p = 2e-3 (`h4_rows.csv`), lines with a detector-free component per
circuit and their mass, mean [min, max] over sequences:

| m | B's 10 seq: lines | B's 10 seq: mass / p | 20 new seq: lines | 20 new seq: mass / p |
|---|---|---|---|---|
| 2 | 0.6 [0, 3] | 2.9 [0, 10.3] | 0.5 [0, 3] | 3.3 [0, 10.3] |
| 6 | 1.8 [0, 4] | 7.1 [0, 18.7] | 1.05 [0, 4] | 5.5 [0, 20.6] |
| 14 | 4.1 [2, 6] | 14.4 [0.3, 20.9] | 3.95 [0, 9] | 14.4 [0, 30.9] |
| 22 | 7.1 [5, 10] | 22.8 [20.8, 31.2] | 6.65 [1, 14] | 20.7 [0.2, 50.0] |

- B's census values are reproduced exactly on B's sequences.
- Every detector-free component is a bare `L0`, never an empty one. So
  dropping it always flips the L0 parity PyMatching attributes to that line:
  379 of 379 such lines.
- Over all 120 circuits, plain PyMatching emits no warning, and its graph is
  identical (edges, fault ids, weights) to the graph of the DEM with those
  components deleted. So it drops them silently.
- Correlated PyMatching raises the `ValueError` above if and only if the
  circuit has at least one such line (120 of 120 circuits). It builds on the
  stripped DEM in every case.
- The mass is linear in p: 20.9, 21.1, 21.2 at p = 1e-5 for three new m = 22
  circuits.

"2.9p–22.8p per circuit" are the means over B's sequences at m = 2 and
m = 22. Individual circuits range from 0 to 31p on B's sequences and 0 to 50p
on new ones.

## 6. H5: MWPM's own complementary gap vs MLD's |LLR| [VN]

**Method.** I computed the exact class weights W(0) and W(1) on PyMatching's
own merged graph (`Matching.edges()`, with its weights and fault ids) in two
ways. Neither uses B's `MinSumTransferMatrix`:
- by brute-force min-sum over all 2^15 (syndrome, class) states at m ≤ 2;
- by my own min-sum time-slot sweep (`MinPlusSweep`) at m = 6 and 22.

The sweep agrees with the brute force to 4e-15 at m = 0, 1, 2 (p = 2e-3 and
0.03, three graphs). Its min over l of W(l) equals PyMatching's returned
weight to within 2.1e-6 at m = 6, 14, 22. There are 0 decision disagreements
with PyMatching in about 8e7 decided sampled shots. The gap is |W1 − W0|, and
its predicted error is 1/(1 + e^gap).

**Calibration** (1e7 new-seed shots per point; empirical error / predicted;
B's value in brackets):

| point | gap bin (nats) | MWPM own gap | MLD \|LLR\| |
|---|---|---|---|
| 5e-4, m = 2 (brute force) | [6, 8) | 0.280 / 9.0e-4 [0.280 / 9.0e-4] | 1.21e-3 / 1.23e-3 |
| 5e-4, m = 2 | [8, 12) | 0.136 / 8.9e-5 [0.136] | 4.03e-5 / 3.91e-5 |
| 5e-4, m = 6 (sweep) | [6, 8) | **0.290 / 9.8e-4** [0.290 / 9.75e-4] | 1.37e-3 / 1.36e-3 |
| 5e-4, m = 6 | [8, 12) | 0.125 / 1.0e-4 [0.124] | 9.15e-5 / 9.02e-5 |
| 5e-4, m = 22 (sweep) | [6, 8) | 0.316 / 1.1e-3 [0.317] | 1.30e-3 / 1.32e-3 |
| 2e-3, m = 6 (sweep) | [6, 8) | 0.185 / 1.2e-3 [0.186] | 1.50e-3 / 1.52e-3 |
| 10 new seq, 5e-4, m = 6 | [6, 8) | 0.322 / 1.0e-3 | 1.45e-3 / 1.40e-3 |

- MLD's |LLR| is calibrated within statistics: the worst |z| over 98 bins
  with at least 1000 shots, on B's and new sequences, is 2.2.
- Shots with an MWPM gap of 6–8 nats fail 18–32% of the time at these
  points, against about 0.1% predicted.
- Across B's whole grid that bin fails 0.6% (1e-2, m = 0) to 32%. So "29%"
  is the (5e-4, m = 6) value, and even the smallest is 12× the prediction.

**Accepted error at fixed acceptance** (gap thresholding; A0 = zero-syndrome
acceptance):

| point | A | \|LLR\| | MWPM own gap | ratio (mine) | ratio (B) |
|---|---|---:|---:|---:|---:|
| 5e-4, m = 2 | A0 = 0.869 | 3.89e-5 | 7.8e-4 ± 3.8e-4 | 20× | 20× |
| 5e-4, m = 2 | 0.9 | 6.07e-5 | 4.4e-3 ± 1.0e-3 | 72× | 74× |
| 5e-4, m = 6 | 0.9 | 1.95e-3 | 4.75e-2 ± 0.17e-2 | 24× | 24× |
| 5e-4, m = 22 | A0 = 0.305 | 2.86e-4 | 4.5e-3 ± 2.1e-3 | 16× | 16× |
| 2e-3, m = 6 | A0 = 0.247 | 1.45e-3 | 7.1e-3 ± 3.9e-3 | 4.9× | 4.9× |
| 1e-2, m = 2 | A0 = 0.065 | 2.16e-2 | 7.1e-2 ± 2.4e-2 | 3.3× | 3.3× |
| 10 new seq, 2e-3, m = 2 | A0 = 0.574 | 6.4e-4 | 1.75e-2 ± 0.81e-2 | **27×** | 8.8× (B's seq) |
| 10 new seq, 5e-4, m = 6 | A0 = 0.706 | 8.7e-5 | 3.7e-3 ± 2.0e-3 | **43×** | 12.6× (B's seq) |

- |LLR| thresholding at A0 reproduces the zero-syndrome error at every point,
  as B says.
- The A0 factor is heavy-tailed across sequence sets. On new sequences it is
  27× and 43×, outside B's "3–20× at A0" for m ≥ 2. So the qualitative claim
  is stronger than stated, but the numeric range belongs to B's 10
  sequences. Its jackknife SEs are about ±50%.

## 7. H6: physical RB error per depth step [VA, VN]

- Catalog: `code.SINGLE_QUBIT_CLIFFORDS` has 24 distinct descriptors with 61
  gate tokens in total. Tokens per Clifford: 1 (I, H, P, X, Z), 2 (×7),
  3 (×7), 4 (×4) and 5 (XZHPH). That is 61/24 = 2.5417 tokens per Clifford,
  and the mirror inverse has the same count for every descriptor.
- Each token is followed by DEPOLARIZE1(0.1 p_RB), and a depolarizing
  channel flips the Z readout with probability 2/3 of its strength. Per
  depth step: 2 × 61/24 × 0.1 × 2/3 = 61/180 = **0.33889 p_RB** [VA].
- Exact sequence-averaged decay f = mean over C of (1 − 0.4 p_RB/3)^(2 n_C).
  Then (1 − f)/2 / p_RB = 0.33888 (p_RB = 1e-5), 0.33877 (1e-3), 0.33832
  (5e-3), 0.33775 (1e-2) and 0.33323 (5e-2) [VA].
- Stim's DEM of `build_physical_rb_circuit` gives the logical flip
  probability (1 − λ^N)/2 with λ = 1 − 0.4 p_RB/3 to 1.4e-13 relative. That
  holds for all 30 checks (5 new sequences, m ∈ {1, 6, 22},
  p_RB ∈ {5e-3, 5e-2}) [VN].

So 0.339 p_RB is exact at first order. At the repo's largest p_RB = 5p = 0.05,
the per-step error is 0.333 p_RB, 1.7% lower. B's ratios (35× and 7.0×) are
unchanged at the quoted precision.

## 8. Required fixes (to `notes/B-matching.md`)

1. **H1 (summary item 1, C9).** Replace "MWPM is worse than no decoding for
   every depth ≥ 2" with: "Averaged over sequences, MWPM is worse than no
   decoding at every depth ≥ 1, and its per-step slope exceeds no decoding's
   for every sequence (by 3.4 ± 0.3p). At a fixed small depth this fails per
   sequence: at m = 2, MWPM beats no decoding on 5 of 10 sequences (8 of 20
   new ones), and the 10-sequence mean difference is only 1.4σ."
2. **H2 (C10, R3).** Restrict "1.10–1.79" to the sampled grid
   p ∈ {5e-4, 2e-3, 5e-3, 1e-2}.
   - Add that below it the ratio rises toward the first-order limit: 1.84,
     1.99 and 2.03 at m = 6, 14 and 22.
   - Sampled at p = 1e-4 it is 1.82 (m = 6) and 1.94 (m = 14) on the same
     sequences.
   - Write MWE/MLD as "1.00–1.022 (one saturated point is 0.9999, which is
     noise)".
3. **H3 (C12, R4, §3).**
   - State the scope: m ≥ 2 and below saturation. Report m = 0 separately:
     merge ≈ 0, decomposition 84–98%, degeneracy + ties 2–17%.
   - Attach uncertainties: ±2–11 pp on the merge share, and a per-sequence
     spread of −5% to 57%.
   - State that the class-aware graph and MWE break exact ties to "no flip",
     and that the other tie rule moves shares by up to 4.6 pp at p = 1e-2.
   - Quote the split as about 1/4–1/3, 2/3–3/4 and small, not as hard ranges.
     On 10 new sequences at (2e-3, m = 2) the merge share is 37%.
4. **H4 (C2, R1).** Replace "2.9p–22.8p per circuit" with "2.9p (m = 2) to
   22.8p (m = 22) on average over sequences; single circuits range from 0 to
   31p (0 to 50p on 20 new sequences)".
5. **H5 (C15, R5).**
   - Say that "29%" is the (5e-4, m = 6) value; the 6–8-nat bin fails
     0.6–32% across the grid.
   - Say that the factor at A0 depends on the sequence set: 10 new sequences
     give 27× at (2e-3, m = 2) and 43× at (5e-4, m = 6), above the "3–20×"
     of B's 10 sequences.
6. **H6 (C17, R7).** Replace "exactly 0.339 p_RB" with "0.339 p_RB at first
   order (61/180). The exact sequence-averaged value is 0.338 at
   p_RB = 5e-3 and 0.333 at p_RB = 0.05."

No change is needed to B's code or CSVs.

## Reproduce

```
module load python/3.11
cd <repository root>
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest tests/test_qubit512_matching.py -q                     # 33 passed
JOB=$(squeue -u sohan100 -n q512shared -h -o %i | head -1)
run() { srun --jobid=$JOB --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c \
  "module load python/3.11; cd $PWD; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; \
   timeout 14m python studies/correction-decoders/qubit/scripts/verify_matching.py $1 --workers 16 --shots 1000000"; }
run all     # census (H1), h4, h6, sampling at m <= 2 and p = 1e-4: verify_report.json, ~35 s
run sweep   # m = 6 and 22 with my min-sum sweep: verify_sweep.json, ~3 min
python studies/correction-decoders/qubit/scripts/verify_matching.py validate --workers 1   # sweep vs brute force, ~2 s
```
