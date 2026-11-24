# B: matching (MWPM) decoders against exact MLD for qubit `[[5,1,2]]` LRB

Work package B. Code: `src/lrb_qubit/decoders_matching.py`,
`studies/correction-decoders/qubit/scripts/matching_vs_mld.py`, `tests/test_qubit512_matching.py` (33
tests, all passing). Data: `studies/correction-decoders/qubit/results/B/`.

Labels: **[VA]** verified analytically, **[VN]** verified numerically,
**[LIT n]** from literature (reference n, fetched by me), **[CONJ]** conjecture.
Uncertainties are one standard error. Pooled quantities carry a
leave-one-sequence-out jackknife SE. That SE includes the sequence-to-sequence
spread, which dominates the shot noise. "Per depth step" means one forward
Clifford, one check round with reset, and one inverse Clifford.

## Summary

1. **At distance 2, PyMatching MWPM is about 2× worse than exact MLD, and, on
   average over sequences, worse than not decoding at all.**
   - First order (p → 0, exact enumeration, 10 sequences), error per depth step:
     MLD 11.79 ± 0.18 p, plain MWPM 24.48 ± 0.40 p, no decoding 21.11 ± 0.21 p.
     The verifier reproduced these on 20 new sequences: 11.85, 24.54 and 21.17.
     [VN]
   - Sampled with 1e7 shots per (p, m) on identical shots, on the sampled grid
     p ∈ {5e-4, 2e-3, 5e-3, 1e-2}: MWPM/MLD error ratio 1.10–1.79 at the points
     with LER_MLD < 0.4. It tends to 1 as every decoder saturates at 0.5.
   - Below the grid the ratio rises toward the first-order limit c_MWPM/c_MLD:
     1.84, 1.99 and 2.03 at m = 6, 14 and 22 (R2). The verifier sampled 1.82
     (m = 6) and 1.94 (m = 14) at p = 1e-4 on the same sequences.
   - MWPM is worse than no decoding on average over sequences at every depth
     m ≥ 1. The first-order mean difference is positive at every m ≥ 1 and
     > 2σ for m ≥ 6 here (m ≥ 2 on the verifier's 20 sequences).
   - Per step it is worse for every sequence: its per-step slope exceeds no
     decoding's by 3.4 ± 0.3p, minimum 1.1p, over 30 sequences (verifier).
   - Sampled, the pooled MWPM − none is positive at all 16 points with m ≥ 2
     (up to +2.7e-2 in LER) and exceeds 2σ at 13 of them. [VN]
   - **This fails per sequence at a fixed small depth.** At m = 2, MWPM beats
     no decoding on 5 of my 10 sequences (8 of the verifier's 20). The
     10-sequence mean first-order difference there is only 1.4σ. [VN, from the
     verifier's census]
2. **Correlated matching exists in PyMatching 2.3.1 (`enable_correlations`),
   but it fails on this protocol and does not help.**
   - Building it directly from Stim's decomposed DEM raises `ValueError` for
     every m ≥ 2 sequence that has an observable-only decomposition component.
     I build it from the DEM with those components stripped.
   - It is then never significantly better than plain MWPM. It is 0–4.8 × 1e-3
     worse; first-order slope 25.06 ± 0.38 p. [VN]
3. **The loss is from projecting onto a matching graph, not from degeneracy.**
   An exact minimum-weight-error decoder on the undecomposed hypergraph (MWE,
   min-sum transfer matrix) gets the following. [VN]
   - Equal to MLD at first order.
   - Within 0–2.2% of MLD's LER sampled.
   - On identical shots, for m ≥ 2 and below saturation, the MWPM − MLD gap
     splits roughly into three steps:
     - PyMatching's "first edge's observable wins" parallel-edge merge: about
       1/4–1/3;
     - decomposing hyperedges into independent graphlike pieces: about
       2/3–3/4;
     - degeneracy plus exact weight ties (MWE → MLD): small (0.3–0.6% at
       p = 5e-4, up to 14% at p = 1e-2).
   - At first order the split is 33% / 67% / 0.
   - The merge share is not a stable number. Its jackknife SE is ±2–11
     percentage points, and per sequence it ranges from −5% to 57%; on 10 new
     sequences at (2e-3, m = 2) the verifier found 37%.
   - Exact weight ties in the class-aware graph and in MWE are broken to "no
     flip". The other rule moves the shares by up to 4.6 percentage points at
     p = 1e-2 (≤ 0.3 at p = 5e-4).
   - At m = 0 the split is different: merge ≈ 0 (−2 to +2%), decomposition
     84–98%, degeneracy + ties 2–17%.
4. **Complementary gap: plain MWPM's own gap is badly overconfident.**
   - The observable-as-detector construction is exact on its own graph. Stim
     decomposes it with every observable-detector component graphlike. Its
     decision is about the same as no decoding.
   - The exact complementary gap of plain MWPM's own graph (min-sum, ≤ 3.8e-6
     from PyMatching's weights) is badly overconfident at d = 2. At
     (p = 5e-4, m = 6), shots with gap 6–8 nats fail 29% of the time, where
     0.1% is predicted. Across the grid that bin fails 0.6% (p = 1e-2, m = 0) to
     32%, 12–310× its prediction.
   - At m ≥ 2 and the zero-syndrome acceptance A0, gap thresholding accepts
     3–20× the error of exact-|LLR| thresholding on my 10 sequences, and up to
     73× at A = 0.9 (p = 5e-4, m = 2).
   - The A0 factor depends heavily on the sequence set: the verifier's 10 new
     sequences give 27× at (2e-3, m = 2) and 43× at (5e-4, m = 6).
   - At m ≥ 2 the observable-detector and class-aware gaps give 1.4–2× the
     error at A0. At A = 0.9 (p = 5e-4, m = 6) that grows to 9× and 13×.
   - At m = 0 every graph gap gives 1.2–1.3×. The MWE gap is within 0–15% of
     |LLR|. [VN]
5. **Against physical RB: correction with check rounds is far worse per depth
   step than a bare qubit.**
   - Physical RB error per depth step is 0.339 p_RB at first order (61/180).
     The exact sequence-averaged value is 0.338 p_RB at p_RB = 5e-3 and
     0.333 p_RB at p_RB = 0.05. That is about 0.339p at p_RB = p and 1.69p at
     the repo's p_RB = 5p. [VA, VN cross-check]
   - Exact-MLD LRB is therefore 35× (p_RB = p) and 7.0× (p_RB = 5p) worse per
     step. Plain MWPM is 72× and 14× worse. [VN]

## 1. Methods

**Circuits.** `circuits.build_lrb_circuit(seq, m, p, terminal="data5")`:
SI1000 noise, all five data qubits read out ideally, observable 0 = x0 ⊕ x3.
Sequences are `experiment.clifford_sequences(10, 22, 20261028)`, the first 10
of the lead's grid. Depths m ∈ {0, 2, 6, 14, 22}; p ∈ {5e-4, 2e-3, 5e-3, 1e-2}.

**Decoders** (`decoders_matching.DecoderSuite`, all decoding the same detection
events; weights are ln((1 − q)/q), so every gap is in nats):

| name | what it is |
|---|---|
| `mld` | the lead's `TransferMatrixMLD` on the undecomposed DEM; gap = \|LLR\| |
| `mwe` | exact minimum-weight error on the undecomposed hypergraph, from `MinSumTransferMatrix` (the min-sum semiring version of the lead's sweep); gap = W(other) − W(chosen) |
| `mwpm` | PyMatching 2.3.1 `Matching.from_detector_error_model(dem(decompose_errors=True))`, `decode_batch` |
| `mwpm_corr` | the same with `enable_correlations=True`, built from the stripped DEM (§5, pitfall P2) |
| `mwpm_gap` | exact class weights W(0), W(1) on PyMatching's own merged graph (min-sum); its argmin is the `mwpm` decision, and its gap is the exact complementary gap of plain MWPM |
| `mwpm_classaware` | min-sum on the decomposed components merged per (detectors, observable) class: PyMatching's graph without the first-observable rule |
| `mwpm_obs` | observable 0 turned into a detector D_L, `decompose_errors=True` on that circuit, PyMatching decode with D_L forced to 0 and to 1; decision argmin, gap \|W1 − W0\| |
| `mwpm_obs_corr` | the same with correlated matching in both forced passes (heuristic; see claim C18) |
| `none` | always predicts "no flip" |

Ties (|W1 − W0| < 1e-9) predict "no flip", like the lead's `llr < 0` rule.

**Why a min-sum transfer matrix for the gap.** The literature construction
[LIT 3] attaches a detector to "all the boundary edges on one side" and toggles
it. That is exact only if every observable-flipping edge is a boundary edge.
- Here 204 two-detector components per m = 22 circuit flip L0.
- At every m ≥ 2, both PyMatching's merged graph and the class-aware graph
  contain bulk cycles of odd logical parity, found by a BFS 2-colouring. So no
  re-choice of the observable representative (L' = L ⊕ detectors) moves all of
  L onto the boundary. [VN]
- Every graphlike component touches at most two consecutive time slots. So the
  exact class weights W(l) of any graph are computable by the lead's slot sweep
  with (+, ×) replaced by (min, +). [VA]

That gives the exact complementary gap. It equals brute-force min-sum to 1e-15
at m ≤ 2, and PyMatching's returned weights to ≤ 3.8e-6, which is
PyMatching's weight discretization. [VN, tests]

**First-order analysis.** For every undecomposed DEM mechanism i (probability
p_i, syndrome s_i, flip l_i), decode s_i alone. The coefficient
c_D = Σ_i p_i [D(s_i) ≠ l_i] / p is exact up to O(p). [VA] Coefficients are
evaluated at p = 1e-5 (and at each grid p in `first_order.csv`).

**Sampling.** One Stim detector sampler per circuit. Seed =
202610282 + 1e6·round(1e5·p) + 1000·seq + m. 1e6 shots per circuit and 1e7 per
(p, m) for every decoder. Total sampling CPU was 16,782 s on the shared
allocation.

**Acceptance curves.** Accept a shot when its gap ≥ τ, with one τ for all
sequences. A threshold that falls inside a 0.02-nat bin accepts a random
fraction of that bin (linear interpolation), and the first bin [0, 1e-9) holds
the exact ties. The jackknife re-derives the threshold for each left-out
sequence.

## 2. Results

### R1. How many mechanisms are not graphlike, and what the decomposition does (`census.csv`)

Mean over the 10 sequences, with [min, max] where it matters. Probability
masses are in units of p (p_ref = 2e-3).

| quantity | m=0 | m=2 | m=6 | m=14 | m=22 |
|---|---:|---:|---:|---:|---:|
| detectors | 6 | 14 | 30 | 62 | 94 |
| undecomposed mechanisms | 27 | 145 | 373 | 824 | 1280 |
| **non-graphlike (≥ 3 detectors)** | **5** | **70.4** | **196** | **446** | **695** |
| their mass / total mass | 1.4 / 69 | 56 / 278 | 154 / 695 | 371 / 1530 | 589 / 2360 |
| largest hyperedge (detectors) | 4 | 4–6 | 6 | 6 | 6 |
| decomposed DEM lines | 35 | 255 | 646 | 1410 | 2180 |
| graphlike, kept as one component | 15 | 53 | 123 | 265 | 406 |
| graphlike (2 detectors), split into two boundary components | 13 | 71 | 173 | 366 | 567 |
| hyperedge, decomposed into graphlike components | 7 | 130 | 348 | 775 | 1200 |
| decomposed with an observable-only remainder (`... ^ L0`) | 0 | 0.6 [0, 3] | 1.8 [0, 4] | 4.1 [2, 6] | 7.1 [5, 10] |
| mass of those lines | 0 | 2.9 [0, 10.3] | 7.1 [0, 18.7] | 14.4 [0.3, 20.9] | 22.8 [20.8, 31.2] |
| two-detector components that flip L0 | 0 | 22.6 | 53.9 | 135 | 204 |
| PyMatching edges / merged from ≥ 2 components | 11 / 7 | 41 / 37 | 97 / 92 | 210 / 204 | 322 / 314 |
| merged edges whose components disagree on L0 | 4 | 11.7 | 25.5 | 54.6 | 83.6 |
| mass of the dropped observable class | 14.4 | 74 | 189 | 418 | 671 |
| conflict edges where the kept class is the less likely one | 1 | 2.4 | 5.5 | 12.9 | 20.9 |
| observable-detector DEM: components touching D_L (all graphlike) | 10 | 73 | 180 | 382 | 589 |

How the decomposition handles them. [VN]
- **Hyperedges.** Stim 1.15 keeps one decomposed line per distinct
  decomposition (more lines than mechanisms). It writes almost every hyperedge
  as graphlike components that already exist elsewhere in the DEM.
- **Graphlike mechanisms are split too.** Two-detector mechanisms are often
  written as two single-detector (boundary) components: 567 decomposed lines
  per m = 22 circuit. From a scratch analysis over 10 sequences at p = 2e-3 (not
  in the CSVs): 59% of these splits are X/Z-mixed (Y-type), 41% are same-type,
  and 40% have an opposite-class partner on the same detector pair.
- **Observable-only remainders.** When no consistent split exists, Stim appends
  an observable-only component. Example (m = 2, sequence 0, q = 10.3p):
  `error(0.0206) D6 D8 L0 ^ D10 ^ L0`. That line is the round-1 S_X1
  measurement error. A logical S in the inverse half maps that stabilizer into
  the reference of both terminal-round detectors, so the time-like error
  touches three detectors.
  - Their mass is 2.9p (m = 2) to 22.8p (m = 22) on average over sequences.
    Single circuits range from 0 to 31p here, and from 0 to 50p on the
    verifier's 20 new sequences.
  - Every such component is a bare `L0`, so dropping it always flips the L0
    parity attributed to the line (verifier: 379 of 379).
- **PyMatching's treatment.** It merges every parallel component into one edge
  and keeps the observable of the first one added (docstring [LIT 4],
  reproduced exactly by `pymatching_merge_model`). At m = 22 that discards an
  observable class carrying 671p, 28% of the DEM's mass.

### R2. Exact first-order coefficients (`first_order.csv`, p = 1e-5)

c_D in units of p (mean ± SEM over 10 sequences). The slope comes from a
per-sequence linear fit over m.

| decoder | m=0 | m=2 | m=6 | m=14 | m=22 | slope per depth step |
|---|---:|---:|---:|---:|---:|---:|
| MLD | 11.1 | 37.3 ± 1.8 | 84.5 ± 2.3 | 176.0 ± 4.1 | 272.3 ± 4.1 | **11.79 ± 0.18** |
| MWE (hypergraph) | 11.1 | 37.3 ± 1.8 | 84.5 ± 2.3 | 176.0 ± 4.1 | 272.3 ± 4.1 | 11.79 ± 0.18 |
| MWPM (PyMatching) | 12.8 | 62.5 ± 4.5 | 155.8 ± 5.1 | 349.4 ± 6.6 | 553.4 ± 7.7 | **24.48 ± 0.40** |
| MWPM correlated | 12.8 | 63.6 ± 4.6 | 161.4 ± 5.6 | 359.0 ± 7.1 | 565.9 ± 7.2 | 25.06 ± 0.38 |
| class-aware graph | 12.8 | 55.6 ± 3.5 | 136.4 ± 4.6 | 294.1 ± 4.4 | 461.1 ± 5.9 | 20.26 ± 0.24 |
| observable-detector | 13.3 | 52.9 ± 2.2 | 132.0 ± 3.2 | 287.9 ± 4.4 | 449.9 ± 5.0 | 19.80 ± 0.21 |
| observable-detector, correlated | 28.7 | 78.2 ± 4.1 | 171.5 ± 9.7 | 347.7 ± 10.4 | 516.6 ± 11.8 | 22.12 ± 0.54 |
| no decoding | 13.3 | 57.7 ± 1.8 | 142.0 ± 2.6 | 307.6 ± 4.7 | 479.8 ± 4.5 | **21.11 ± 0.21** |

MLD's 11.79p agrees with the lead's 11.8p (L0 note). [VN]

Which single faults plain MWPM gets wrong while MLD gets them right, labelled
by how PyMatching sees the mechanism's observable (units of p):

| label | m=2 | m=6 | m=14 | m=22 |
|---|---:|---:|---:|---:|
| a component's observable differs from the one PyMatching kept, 1-detector mechanism | 8.4 ± 1.9 | 15.6 ± 2.5 | 35.8 ± 2.9 | 56.1 ± 3.8 |
| same, 2-detector mechanism | 12.3 ± 0.8 | 35.2 ± 1.3 | 76.3 ± 2.4 | 119.0 ± 2.7 |
| same, hyperedge | 14.1 ± 4.1 | 40.4 ± 4.2 | 106.0 ± 7.5 | 174.2 ± 8.8 |
| observables consistent (matching geometry) | 4.6 ± 1.0 | 8.6 ± 1.1 | 17.5 ± 1.9 | 25.3 ± 2.4 |
| observable-only remainder dropped | 0.0 | 0.1 ± 0.1 | 0.2 ± 0.1 | 0.4 ± 0.1 |
| *offset: MLD wrong, MWPM right* | −14.2 ± 1.5 | −28.6 ± 2.3 | −62.4 ± 2.5 | −93.8 ± 2.9 |

At m = 22 the rows sum to 281 = 553.4 − 272.3. 93% of the gross excess sits on
mechanisms whose component observables conflict with PyMatching's merged
edges. [VN]

### R3. Sampled logical error rate, ratio to MLD and agreement (`ler.csv`; 1e7 identical shots per point)

| p | m | LER MLD | LER MWE | LER MWPM | LER MWPM-corr | LER obs-det | LER none | MWPM/MLD | agree(MWPM, MLD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0 | 0.00550 ± 0.00002 | 0.00553 | 0.00637 | 0.00637 | 0.00662 | 0.00662 | 1.158 ± 0.002 | 0.9945 |
| 5e-4 | 2 | 0.0187 ± 0.0008 | 0.0188 | 0.0302 ± 0.0022 | 0.0308 | 0.0259 | 0.0281 | 1.61 ± 0.08 | 0.974 ± 0.003 |
| 5e-4 | 6 | 0.0417 ± 0.0010 | 0.0418 | 0.0721 ± 0.0021 | 0.0745 | 0.0621 | 0.0661 | 1.73 ± 0.05 | 0.940 ± 0.003 |
| 5e-4 | 14 | 0.0829 ± 0.0017 | 0.0833 | 0.148 ± 0.002 | 0.152 | 0.126 | 0.132 | 1.79 ± 0.03 | 0.871 ± 0.003 |
| 5e-4 | 22 | 0.1225 ± 0.0015 | 0.1231 | 0.213 ± 0.002 | 0.217 | 0.182 | 0.191 | 1.74 ± 0.02 | 0.813 ± 0.002 |
| 2e-3 | 0 | 0.02195 ± 0.00003 | 0.02200 | 0.02506 | 0.02508 | 0.02595 | 0.02595 | 1.142 ± 0.002 | 0.9787 |
| 2e-3 | 2 | 0.074 ± 0.003 | 0.075 | 0.111 ± 0.007 | 0.113 | 0.097 | 0.103 | 1.50 ± 0.06 | 0.906 ± 0.009 |
| 2e-3 | 6 | 0.156 ± 0.003 | 0.158 | 0.236 ± 0.005 | 0.240 | 0.209 | 0.217 | 1.51 ± 0.03 | 0.798 ± 0.008 |
| 2e-3 | 14 | 0.273 ± 0.004 | 0.277 | 0.381 ± 0.003 | 0.384 | 0.347 | 0.354 | 1.40 ± 0.02 | 0.650 ± 0.005 |
| 2e-3 | 22 | 0.353 ± 0.002 | 0.356 | 0.449 ± 0.001 | 0.450 | 0.422 | 0.427 | 1.270 ± 0.007 | 0.574 ± 0.002 |
| 5e-3 | 0 | 0.05381 ± 0.00010 | 0.05429 | 0.06058 | 0.06065 | 0.06275 | 0.06275 | 1.126 ± 0.002 | 0.9495 |
| 5e-3 | 2 | 0.178 ± 0.006 | 0.181 | 0.238 ± 0.011 | 0.239 | 0.214 | 0.220 | 1.34 ± 0.04 | 0.80 ± 0.02 |
| 5e-3 | 6 | 0.326 ± 0.004 | 0.332 | 0.405 ± 0.004 | 0.408 | 0.378 | 0.380 | 1.24 ± 0.01 | 0.639 ± 0.008 |
| 5e-3 | 14 | 0.447 ± 0.002 | 0.452 | 0.488 ± 0.001 | 0.489 | 0.477 | 0.477 | 1.091 ± 0.004 | 0.524 ± 0.002 |
| 5e-3 | 22 | 0.4851 ± 0.0006 | 0.4871 | 0.4986 ± 0.0003 | 0.4986 | 0.4962 | 0.4961 | 1.028 ± 0.001 | 0.5038 ± 0.0003 |
| 1e-2 | 0 | 0.10400 ± 0.00009 | 0.10584 | 0.11454 | 0.11454 | 0.11813 | 0.11813 | 1.101 ± 0.001 | 0.9085 |
| 1e-2 | 2 | 0.310 ± 0.007 | 0.316 | 0.369 ± 0.011 | 0.368 | 0.342 | 0.343 | 1.19 ± 0.02 | 0.69 ± 0.02 |
| 1e-2 | 6 | 0.454 ± 0.002 | 0.459 | 0.485 ± 0.002 | 0.486 | 0.473 | 0.471 | 1.068 ± 0.003 | 0.547 ± 0.007 |
| 1e-2 | 14 | 0.4974 ± 0.0003 | 0.4977 | 0.5000 | 0.4999 | 0.4991 | 0.4991 | 1.005 ± 0.001 | 0.5019 ± 0.0004 |
| 1e-2 | 22 | 0.4998 ± 0.0002 | 0.4997 | 0.5001 | 0.5001 | 0.5000 | 0.4998 | 1.001 ± 0.001 | 0.5000 ± 0.0001 |

The full table, with every decoder, SEs, McNemar SEs and tie statistics, is
`ler.csv`. A few readings:
- MWE/MLD is 1.00–1.022 on the grid. One saturated point (p = 1e-2,
  m = 22) is 0.9999, which is noise.
- MWPM/MLD is 1.10–1.79 at the grid points with LER_MLD < 0.4. That range
  belongs to the sampled grid p ∈ {5e-4, 2e-3, 5e-3, 1e-2}.
  - Below the grid the ratio rises toward the first-order limit c_MWPM/c_MLD
    from R2: 1.84 (m = 6), 1.99 (m = 14), 2.03 (m = 22).
  - The verifier sampled 1.818 ± 0.006 (m = 6) and 1.939 ± 0.004 (m = 14) at
    p = 1e-4 on the same sequences (shot-noise SEs).
- The verifier resampled 7 grid points with new seeds and agreed with this
  table within shot noise (|z| ≤ 1.8).
- The observable-detector decision coincides with "none" at m = 0. They have
  identical LER, identical agreement, and both have first-order coefficient
  13.33. [VN]

### R4. Where MWPM loses: paired ladder on identical shots (`per_circuit.csv`)

Paired LER differences in units of 1e-3, with jackknife SE over sequences:

| p | m | merge rule (MWPM − class-aware) | decomposition (class-aware − MWE) | degeneracy + ties (MWE − MLD) | total (MWPM − MLD) | correlated − plain | MWPM − none |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 2 | +3.0 ± 1.1 | +8.4 ± 1.2 | +0.07 ± 0.02 | +11.5 ± 1.7 | +0.55 ± 0.28 | +2.2 ± 1.7 |
| 5e-4 | 6 | +7.8 ± 2.0 | +22.5 ± 1.5 | +0.09 ± 0.02 | +30.4 ± 1.8 | +2.5 ± 0.5 | +5.9 ± 1.8 |
| 5e-4 | 22 | +26.9 ± 1.9 | +63.4 ± 0.9 | +0.57 ± 0.02 | +90.9 ± 1.9 | +3.5 ± 0.4 | +22.6 ± 1.9 |
| 2e-3 | 2 | +9.6 ± 3.3 | +26.7 ± 3.7 | +0.83 ± 0.07 | +37.2 ± 5.0 | +1.1 ± 0.8 | +8.3 ± 5.1 |
| 2e-3 | 6 | +19.8 ± 4.7 | +57.5 ± 3.8 | +2.1 ± 0.1 | +79.5 ± 4.0 | +4.8 ± 1.0 | +19.0 ± 4.0 |
| 2e-3 | 22 | +21.5 ± 1.6 | +70.6 ± 1.2 | +3.4 ± 0.1 | +95.5 ± 2.0 | +1.7 ± 0.3 | +21.3 ± 1.4 |
| 5e-3 | 2 | +17.1 ± 4.5 | +39.9 ± 5.6 | +3.8 ± 0.3 | +60.8 ± 7.2 | +0.1 ± 1.4 | +18.8 ± 7.6 |
| 5e-3 | 6 | +18.0 ± 4.1 | +54.5 ± 3.2 | +6.7 ± 0.4 | +79.2 ± 3.1 | +3.0 ± 0.9 | +24.7 ± 3.1 |
| 1e-2 | 2 | +18.4 ± 3.7 | +34.6 ± 5.1 | +6.9 ± 0.5 | +59.9 ± 5.5 | −1.4 ± 2.5 | +26.0 ± 6.3 |
| 1e-2 | 6 | +7.2 ± 1.6 | +19.3 ± 1.3 | +4.3 ± 0.3 | +30.8 ± 1.3 | +1.2 ± 0.4 | +13.8 ± 1.2 |

At m = 0 every gap is small: MWPM − MLD is +0.87 ± 0.01 (5e-4) to
+10.6 ± 0.1 (1e-2), and MWPM beats "none" there. At m ≥ 14 and p ≥ 5e-3 every
decoder is near LER = 0.5. All 20 rows can be computed from `per_circuit.csv`;
the table shows the informative ones.

**Scope and precision of the shares.**
- For m ≥ 2 below saturation the split is about 1/4–1/3 (merge), 2/3–3/4
  (decomposition) and small (degeneracy + ties).
  - Pooled values over the 11 rows with m ≥ 2 and LER_MLD < 0.4: 22.5–31%,
    58–74% and 0.3–11.5%.
  - The (1e-2, m = 6) row (LER_MLD = 0.45) gives 23 / 63 / 14%.
- The merge share has jackknife SEs of ±2–11 percentage points (±6–11 at
  m ≤ 6). Per sequence it runs from −5% to 57%, and at (2e-3, m = 2) on new
  sequences it looks bimodal. The verifier's 10 new sequences give 37% there.
- Saturated rows are erratic. At (1e-2, m = 22) the shares are 57 / 51 / −8%
  of a total of only 0.3e-3.
- At m = 0 the split is different: merge −2 to +2%, decomposition 84–98%,
  degeneracy + ties 2–17% (17.5% at p = 1e-2).
- Tie rule: the class-aware graph and MWE break exact weight ties to "no flip".
  Breaking them to "flip" moves the shares by ≤ 0.3 percentage points at
  p = 5e-4 and at m ≥ 6, and by up to 4.6 at p = 1e-2 (verifier). PyMatching's
  graph has no exact ties.
- A sequential ladder through one intermediate decoder is one attribution
  among several. [CONJ: another intermediate would split the gap differently.]

The verifier re-derived the ladder with independent code at 6 grid points
(≤ 1.4 percentage points from these) and got 33 / 67 / 0 at first order. [VN]

### R5. Gap-thresholded acceptance vs error (`acceptance.csv`, `gap_hist.npz`)

Accepted error rate at fixed acceptance A, with jackknife SE. A0 is the
acceptance of the zero-syndrome policy, which keeps shots with no detection
event at all. That policy is the lead's `lrbd_unif_1_final`: a round's raw
syndromes all vanish exactly when its difference detectors all vanish
(triangular linear map). [VA]

| p | m | A0 | zero-syndrome policy | \|LLR\| (MLD) | MWE gap | MWPM own gap | class-aware gap | obs-detector gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 2 | 0.869 | 3.9e-5 ± 0.4e-5 | 3.9e-5 ± 0.4e-5 | 3.9e-5 ± 0.4e-5 | 8.0e-4 ± 3.8e-4 | 5.3e-5 ± 0.6e-5 | 5.9e-5 ± 0.7e-5 |
| 5e-4 | 6 | 0.704 | 8.2e-5 ± 0.5e-5 | 8.2e-5 ± 0.5e-5 | 8.2e-5 ± 0.5e-5 | 1.0e-3 ± 0.9e-3 | 1.4e-4 ± 0.1e-4 | 1.5e-4 ± 0.1e-4 |
| 5e-4 | 22 | 0.304 | 2.81e-4 ± 0.07e-4 | 2.80e-4 ± 0.07e-4 | 2.81e-4 ± 0.07e-4 | 4.5e-3 ± 2.1e-3 | 4.5e-4 ± 0.3e-4 | 5.2e-4 ± 0.4e-4 |
| 2e-3 | 2 | 0.572 | 6.4e-4 ± 0.5e-4 | 6.4e-4 ± 0.5e-4 | 6.5e-4 ± 0.5e-4 | 5.7e-3 ± 2.4e-3 | 1.05e-3 ± 0.17e-3 | 1.07e-3 ± 0.13e-3 |
| 2e-3 | 6 | 0.247 | 1.47e-3 ± 0.06e-3 | 1.47e-3 ± 0.06e-3 | 1.50e-3 ± 0.07e-3 | 7.2e-3 ± 3.8e-3 | 2.7e-3 ± 0.2e-3 | 2.8e-3 ± 0.2e-3 |
| 5e-3 | 2 | 0.250 | 4.5e-3 ± 0.3e-3 | 4.5e-3 ± 0.3e-3 | 4.8e-3 ± 0.4e-3 | 3.8e-2 ± 1.8e-2 | 8.8e-3 ± 1.4e-3 | 9.1e-3 ± 1.6e-3 |
| 1e-2 | 2 | 0.065 | 2.2e-2 ± 0.1e-2 | 2.2e-2 ± 0.1e-2 | 2.5e-2 ± 0.3e-2 | 7.2e-2 ± 2.3e-2 | 4.1e-2 ± 0.6e-2 | 4.1e-2 ± 0.4e-2 |

At a fixed acceptance A = 0.9:

| p | m | \|LLR\| | MWE gap | MWPM own gap | class-aware gap | obs-detector gap | obs-detector corr gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 2 | 6.0e-5 ± 0.5e-5 | 6.1e-5 ± 0.5e-5 | 4.4e-3 ± 0.9e-3 | 9.1e-5 ± 1.5e-5 | 9.0e-5 ± 1.0e-5 | 1.5e-3 ± 0.4e-3 |
| 5e-4 | 6 | 1.95e-3 ± 0.16e-3 | 1.99e-3 ± 0.16e-3 | 4.7e-2 ± 0.2e-2 | 2.5e-2 ± 0.2e-2 | 1.8e-2 ± 0.1e-2 | 1.8e-2 ± 0.2e-2 |
| 2e-3 | 0 | 2.04e-4 ± 0.04e-4 | 2.04e-4 ± 0.04e-4 | 2.22e-4 ± 0.05e-4 | 2.08e-4 ± 0.04e-4 | 2.08e-4 ± 0.04e-4 | 9.4e-4 ± 0.1e-4 |
| 2e-3 | 2 | 2.9e-2 ± 0.3e-2 | 2.9e-2 ± 0.3e-2 | 8.4e-2 ± 0.5e-2 | 6.0e-2 ± 0.6e-2 | 5.4e-2 ± 0.4e-2 | 5.3e-2 ± 0.6e-2 |

The remaining (p, m, A) points are in `acceptance.csv`, for A from 1 down to 0.1.

Readings:
- **|LLR| at A0 reproduces the zero-syndrome error at every point.** So exact
  |LLR| ranks the trivial syndrome first, consistent with Chow-type optimality
  of posterior thresholds. [VN]
- **Thresholding the MWPM own gap is much worse at matched acceptance.** For
  m ≥ 2 on my 10 sequences it is 3–20× worse at A0 and up to 73× at A = 0.9;
  at m = 0 it is 1.2–1.3×.
  - The factor is heavy-tailed across sequence sets. The verifier's 10 new
    sequences give 27× at (2e-3, m = 2) and 43× at (5e-4, m = 6), against my
    8.8× and 12.6×.
  - So the numeric range belongs to these 10 sequences, with jackknife SEs of
    about ±50%. The qualitative conclusion is stronger, not weaker. The large jackknife SEs show that
  a few sequences contribute syndromes to which MWPM gives a large gap but which
  are ambiguous. [VN]
- **The class-aware and observable-detector gaps are 1.4–2.0× worse at A0**
  for m ≥ 2 (1.2–1.3× at m = 0). At A = 0.9, p = 5e-4, m = 6 they are 13× and
  9× worse. [VN]
- **The MWE gap is within about 0–15% of |LLR|.** [VN]

### R6. Calibration (`calibration.csv`)

Empirical error, with binomial SE, against the mean of 1/(1 + e^gap). Example
at p = 5e-4, m = 6:

| gap bin (nats) | MLD | MWE | MWPM own gap | obs-detector gap |
|---|---|---|---|---|
| [1, 2) | 0.181 vs 0.182 | 0.223 vs 0.203 | 0.407 vs 0.180 | 0.263 vs 0.217 |
| [3, 4) | 0.0305 vs 0.0308 | 0.0458 vs 0.0315 | 0.259 vs 0.0296 | 0.248 vs 0.0281 |
| [6, 8) | 1.44e-3 vs 1.36e-3 | 2.26e-3 vs 0.79e-3 | **0.290 vs 0.98e-3** | 3.8e-3 vs 1.4e-3 |
| [8, 12) | 8.5e-5 vs 9.0e-5 | 9.8e-5 vs 1.8e-5 | **0.124 vs 1.0e-4** | 1.6e-4 vs 0.8e-4 |

SEs are ≤ 5% of the empirical values in these bins. Per gap bin:
- **MLD** is calibrated in every bin at every point checked. [VN, VA: exact
  posterior]
- **The MWPM own gap** is grossly overconfident. The [6, 8)-nat bin fails 29%
  at this point (p = 5e-4, m = 6).
  - Across the grid that bin fails 0.56% (p = 1e-2, m = 0) to 32%
    (p = 5e-4, m = 22): 12–310× its prediction (`calibration.csv`).
  - At this point the [8, 12) bin is 1240× off.
  - That is far beyond the 0.9× rescaling reported for large-distance surface
    codes [LIT 3]. [VN]

### R7. Decoded LRB against physical RB at p_RB = p and the repo's p_RB = 5p (`physical_rb_comparison.csv`)

Physical RB (the lead's `build_physical_rb_circuit`) has DEPOLARIZE1(0.1 p_RB)
after each gate token. A depolarizing channel is invariant under Clifford
conjugation, so F_RB = (1 − 0.4 p_RB/3)^N. N counts the gate tokens: 2.5417 per
Clifford on average, forward and inverse. [VA] The formula matches the lead's
sampled `run_physical_rb` to within z = 0.76, −1.10, −0.56 at p_RB = 0.05,
m = 22 (and a unit test at m = 6). [VN]

At first order the per-step error is 61/180 = 0.3389 p_RB. The exact
sequence-averaged values are 0.338 p_RB at p_RB = 5e-3 and 0.333 p_RB at 0.05
(verifier). The F columns below use the exact formula. First-order
comparison:

| per depth step, units of p | value | / physical RB at p_RB = p (0.339p) | / physical RB at p_RB = 5p (1.69p) |
|---|---:|---:|---:|
| exact MLD | 11.79 ± 0.18 | 34.8 | 6.96 |
| plain MWPM | 24.48 ± 0.40 | 72.2 | 14.4 |
| no decoding | 21.11 ± 0.21 | 62.3 | 12.5 |

Fidelity F = |1 − 2 LER|, averaged over the 10 sequences, with SEM. Physical
RB is exact for the same 10 sequences:

| p | m | F MLD | F MWPM | F obs-det | F none | F_RB (p_RB = p) | F_RB (p_RB = 5p) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 2 | 0.963 ± 0.002 | 0.940 ± 0.004 | 0.948 ± 0.002 | 0.944 ± 0.002 | 0.9992 | 0.9962 |
| 5e-4 | 6 | 0.917 ± 0.002 | 0.856 ± 0.004 | 0.876 ± 0.003 | 0.868 ± 0.002 | 0.9979 | 0.9896 |
| 5e-4 | 22 | 0.755 ± 0.003 | 0.573 ± 0.004 | 0.636 ± 0.003 | 0.619 ± 0.003 | 0.9926 | 0.9637 |
| 2e-3 | 2 | 0.852 ± 0.006 | 0.777 ± 0.014 | 0.806 ± 0.007 | 0.794 ± 0.006 | 0.9970 | 0.9849 |
| 2e-3 | 6 | 0.688 ± 0.006 | 0.529 ± 0.010 | 0.581 ± 0.007 | 0.567 ± 0.006 | 0.9917 | 0.9590 |
| 2e-3 | 22 | 0.294 ± 0.005 | 0.103 ± 0.003 | 0.157 ± 0.003 | 0.146 ± 0.003 | 0.9709 | 0.8627 |
| 5e-3 | 6 | 0.349 ± 0.007 | 0.191 ± 0.008 | 0.244 ± 0.006 | 0.240 ± 0.006 | 0.9793 | 0.9008 |
| 1e-2 | 2 | 0.381 ± 0.014 | 0.261 ± 0.021 | 0.315 ± 0.015 | 0.313 ± 0.012 | 0.9849 | 0.9268 |

At m = 0 the encoded F is below 1 (0.989 for MLD at 5e-4) while physical RB is
exactly 1. The LRB depth-0 point carries a full check round plus readout,
whereas physical RB at m = 0 has no gates.

## 3. Why MWPM and MLD differ at distance 2

- **Degeneracy is small.**
  - At first order MWE = MLD. The undecomposed DEM has at most one line per
    (syndrome, class), so both compare P(s,0) with P(s,1) up to O(p). [VA, VN]
  - Sampled, MWE − MLD is a small share of the MWPM gap: 0.3–0.6% at
    p = 5e-4 and up to 14% at p = 1e-2 for m ≥ 2 (17.5% at p = 1e-2, m = 0).
    It grows with p as multi-fault coset sums start to matter. Ties are broken
    to "no flip" (R4). [VN]
- **Hyperedges are the largest loss.**
  - 20–25% of the DEM mass at m ≥ 2 is in mechanisms with 3–6 detectors. Their
    sources are Y-type data faults, fold-gate two-qubit faults (SWAP/CZ on
    2, 3), and time-like faults that logical S/H gates between rounds spread
    over two stabilizer references.
  - Decomposed into independent graphlike pieces, they lose the joint class
    information that identifies the logical. For m ≥ 2 below saturation, this
    rung is about 2/3–3/4 of the gap (67% at first order; 84–98% at m = 0).
    [VN]
- **The merge rule** (the first-added edge's observable wins) is a DEM-order
  tie-break between opposite-class parallel edges. Such edges exist on 26% of
  PyMatching's edges at m = 22 (83.6 / 322). For m ≥ 2 below saturation, this
  rung is about 1/4–1/3 of the gap (33% at first order, ≈ 0 at m = 0). Its SE
  is ±2–11 percentage points and its per-sequence spread is −5% to 57% (R4).
  [VN]
- **Correlated matching cannot help.**
  - It reweights edges after a first matching, but it never changes the kept
    observable of a merged edge.
  - It also needs the observable-only remainders stripped, which reproduces
    the plain graph. [VN for the outcome; CONJ for the mechanism]
- **Weight ties.**
  - Exact ties in the hypergraph: 0.45% of shots (p = 5e-4, m = 0) up to 13%
    (p = 1e-2, m = 22). On those shots MLD's error is 0.48–0.50, so breaking
    them gains almost nothing. [VN]
  - PyMatching's graph has no exact ties (≤ 5e-6 of shots). Its parallel-edge
    merges amount to a hidden, arbitrary tie-break. [VN]
  - Exact MLD ties are ≤ 0.12% of shots. [VN]

## 4. Claims

| # | claim | label | evidence |
|---|---|---|---|
| C1 | PyMatching 2.3.1 exposes correlated matching (`enable_correlations` in `from_detector_error_model` and `decode_batch`), documented as two-pass, citing Fowler 2013 | VN; LIT 2, LIT 4 | installed API; docstring |
| C2 | Stim 1.15 `decompose_errors=True` emits observable-only components for this circuit family: on average 0.6–7.1 lines per circuit for m ≥ 2, carrying 2.9p (m = 2) to 22.8p (m = 22) on average over sequences. Single circuits range from 0 to 31p (0 to 50p on the verifier's 20 new sequences) | VN | `census.csv`, R1; verifier H4 |
| C3 | Plain PyMatching silently drops those components and attributes L0 to the remaining piece. Correlated PyMatching raises `ValueError` ("undetectable component") | VN | tests `test_plain_pymatching_drops_undetectable_component`, `test_correlated_pymatching_rejects_undetectable_component`, `test_correlated_suite_builds_where_plain_correlated_fails` |
| C4 | PyMatching keeps the first-added observable on merged edges. At m = 22, 83.6 of 322 edges carry a class conflict; the dropped class carries 671p (28% of the mass); in 20.9 edges the kept class is the less likely one | VN; LIT 4 | `test_first_added_observable_wins`, `test_merge_model_reproduces_pymatching_graph`, R1 |
| C5 | Non-graphlike mechanisms: 5 of 27 at m = 0 rising to 695 of 1280 at m = 22, with 2–25% of the mass and up to 6 detectors | VN | R1 |
| C6 | The min-sum transfer matrix is exact for slot-local mechanism lists | VA, VN | brute force agrees to 1e-15 (m ≤ 2, three graphs) |
| C7 | Its class weights on PyMatching's graph reproduce PyMatching's solution weights (≤ 3.8e-6) and decisions (≤ 32 near-tie disagreements per 1e7 shots) | VN | `ler.csv` columns `pm_*`; test |
| C8 | The observable-detector construction is exact on its own graph; on PyMatching's plain graph no gauge makes it exact at m ≥ 2 (odd bulk cycles) | VN | `test_observable_detector_matching_is_exact_on_its_graph`; BFS 2-colouring |
| C9 | First-order slopes per depth step: MLD = MWE 11.79 ± 0.18p; MWPM 24.48 ± 0.40p; correlated 25.06 ± 0.38p; class-aware 20.26 ± 0.24p; obs-det 19.80 ± 0.21p; none 21.11 ± 0.21p. On 20 new sequences: 11.85, 24.54, 25.12, 21.17. MWPM's slope exceeds no decoding's for every sequence (3.4 ± 0.3p), but at a fixed small depth MWPM beats no decoding on about half the sequences (m = 2: 5/10, 8/20 new) | VN | R2 (exact enumeration); verifier H1 |
| C10 | On the sampled grid p ∈ {5e-4, 2e-3, 5e-3, 1e-2}, at points with LER_MLD < 0.4, MWPM/MLD is 1.10–1.79. Below the grid it rises toward the first-order limit 1.84–2.03 (1.82 and 1.94 sampled at p = 1e-4 by the verifier). MWE/MLD is 1.00–1.022 (one saturated point is 0.9999, which is noise) | VN | R3, R2, 1e7 identical shots per point; verifier H2 |
| C11 | Pooled over sequences, plain MWPM is worse than no decoding at all 16 points with m ≥ 2 (> 2σ at 13) and better only at m = 0. It is not true per sequence at small m: see C9 | VN | R4 last column, `per_circuit.csv`; verifier H1 |
| C12 | Ladder, for m ≥ 2 below saturation: merge rule about 1/4–1/3 (SE ±2–11 percentage points; per sequence −5% to 57%; 37% on 10 new sequences at (2e-3, m = 2)), hyperedge decomposition about 2/3–3/4, degeneracy + ties small (0.3–14%). At m = 0: merge ≈ 0, decomposition 84–98%, degeneracy 2–17%. At first order: 33/67/0%. Ties are broken to "no flip"; the other rule moves the shares by ≤ 4.6 percentage points | VN; CONJ that another ladder would split differently | R4, R2; verifier H3 |
| C13 | Correlated matching is never significantly better than plain MWPM (0 to +4.8e-3 worse; one point −1.4 ± 2.5e-3) | VN | R4 |
| C14 | The obs-det decision is about equal to no decoding (identical at m = 0) | VN | R2, R3 |
| C15 | Plain MWPM's own complementary gap is overconfident. Its [6, 8)-nat bin fails 29% at (5e-4, m = 6) and 0.6–32% across the grid, 12–310× the prediction. For m ≥ 2 its gap thresholding has 3–20× the accepted error of \|LLR\| thresholding at A0 on my 10 sequences (27× and 43× on the verifier's new ones; heavy-tailed), and up to 73× at A = 0.9. This confirms the decoder-theorist's conjecture 12 (≥ 3×) | VN | R5, R6; verifier H5 |
| C16 | The exact \|LLR\| is calibrated, and at the zero-syndrome acceptance it reproduces the zero-syndrome (= `lrbd_unif_1_final`) error | VN; VA for the equivalence | R5, R6 |
| C17 | Physical RB error per depth step is 0.339 p_RB at first order (61/180). The exact sequence-averaged value is 0.338 at p_RB = 5e-3 and 0.333 at p_RB = 0.05. Exact MLD is 35× (p_RB = p) and 7.0× (p_RB = 5p) worse per step; MWPM 72× and 14× | VA, VN | R7, `test_physical_rb_fidelity_is_exact`; verifier H6 |
| C18 | The correlated observable-detector gap is heuristic. At p = 1e-5, m = 0, a single boundary defect gets W(1) = 8.30 < W(0) = 9.53 (uncorrelated W(1) = 29.2), so c = 28.7 > 13.3 (none). At p ≥ 1e-4 this goes away | VN | R2; scratch check |
| C19 | The complementary gap is the log-likelihood ratio of the minimum and complementary matchings, formed by toggling a boundary detector; it is slightly overconfident (0.9× rescaling) for surface codes | LIT 3 | Yoked surface codes, Sec. 4 |
| C20 | Hyperedges are where MWPM loses the class information (Y and fold-gate correlations, frame-changing time-like faults), and a hypergraph min-weight decoder is enough to recover MLD here | VN (outcome); CONJ (that this generalizes to other d = 2 LRB protocols) | R2–R4 |

## 5. Bugs and pitfalls

No bug found in the lead's model (`code.py`, `circuits.py`, `mld.py`,
`baseline.py`, `experiment.py`). [VN: all lead APIs used as documented; MLD
first-order slope 11.79p agrees with the lead's 11.8p.] Pitfalls found:

- **P1 (Stim + plain PyMatching).** Observable-only decomposition components are
  dropped silently (C2, C3). The first-order effect is small (≤ 0.4p at
  m = 22), but the graph's L0 labels are then wrong for those lines.
- **P2 (correlated PyMatching).** It raises `ValueError` on the same DEMs. The
  fix used here is `strip_undetectable_components`, which exactly reproduces
  plain PyMatching's graph (test).
- **P3 (PyMatching merge rule).** Decisions depend on DEM line order (C4).
- **P4 (operational, for ALLOCATION.md).**
  - `srun --jobid … --overlap -c 24` placed every agent's step on the same
    CPUs 0–11 and 128–139. My workers got 6–30% of a CPU; load average 97–123.
  - Requesting the node cpuset with `-c 256` while running only 24 worker
    processes fixed it: 100% per worker, and the 1e6-shot grid took about
    13 min wall.
- **P5 (operational).** The session scratchpad lives in node-local `/tmp`. Files
  written there by a compute-node step are invisible from the login node and
  disappear with the job. The raw per-task pickles of this run are gone. The
  summaries in `results/B` are complete.

## 6. Open questions

- Would a class-aware decomposition, or belief-matching, close most of the gap?
  Class-aware decomposition would choose components consistent with the
  mechanism's class and keep both classes per edge. [CONJ: the class-aware
  graph only removes the 25–35% merge-rule part; hyperedge information needs a
  hypergraph or BP-type decoder]
- The min-sum transfer matrix (MWE) costs about as much as exact MLD here (32
  states per round). For hardware the exact posterior is just as cheap, so
  MWPM has no role at d = 2 for this protocol. [CONJ for hardware latency]
- Can the MWPM gap be recalibrated? Not by a monotone map: the ranking itself
  is wrong (R5 at A0). [VN for the ranking; CONJ that no monotone fix exists]

## 7. Reproduce

```
module load python/3.11
cd <repository root>
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest tests/test_qubit512_matching.py -q          # 33 tests
JOB=$(squeue -u sohan100 -n q512shared -h -o %i | head -1)
RAW=$SCRATCH/q512_B_raw    # any shared directory
for half in "0.0005,0.002" "0.005,0.01"; do
  srun --jobid=$JOB --overlap -N 1 -n 1 -c 256 bash -c "module load python/3.11; cd $PWD; \
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; \
    timeout 19m python studies/correction-decoders/qubit/scripts/matching_vs_mld.py --raw-dir $RAW --out /tmp/x \
      --num-seq 10 --shots 1000000 --workers 24 --p-list $half --depths 0,2,6,14,22"
done
python studies/correction-decoders/qubit/scripts/matching_vs_mld.py --raw-dir $RAW --summarize-only \
  --p-list 0.0005,0.002,0.005,0.01 --depths 0,2,6,14,22 \
  --out studies/correction-decoders/qubit/results/B
python studies/correction-decoders/qubit/scripts/matching_vs_mld.py --physical-rb-only \
  --out studies/correction-decoders/qubit/results/B
```

Outputs in `results/B/`:
- `ler.csv`: LER, ratios, agreement, ties, PyMatching consistency.
- `per_circuit.csv`: raw counts per (p, m, sequence, decoder).
- `acceptance.csv`: accepted error at 13 acceptance levels and at A0.
- `calibration.csv`.
- `census.csv` (R1) and `first_order.csv` (R2 and the attribution).
- `physical_rb_comparison.csv` (R7).
- `gap_hist.npz`: per-sequence gap histograms in 0.02-nat bins.
- `meta.json`: sequences, seeds, shots.

## 8. Revision log

An adversarial verifier
(`studies/correction-decoders/qubit/notes/B-matching.verify.md`) reproduced
the core numbers with independent code. It used B's sequences and 20 new ones,
and found no code bug. On its required fixes 1–6 I changed the wording of six
claims. No simulation was rerun, and no code or CSV changed.

1. **H1 (summary 1, C9, C11).** "Worse than no decoding" now holds on average
   over sequences and per step for every sequence. It fails at a fixed small
   depth: at m = 2, MWPM beats no decoding on 5 of 10 sequences (8 of 20 new).
2. **H2 (summary 1, R3, C10).** The range 1.10–1.79 is now limited to the
   sampled grid. Below it the ratio rises toward 1.84–2.03 (1.82 and 1.94
   sampled at p = 1e-4). MWE/MLD's lower end reads 1.00 within noise.
3. **H3 (summary 3, R4, §3, C12).** Added:
   - the scope (m ≥ 2, below saturation), with m = 0 given separately;
   - the uncertainties (±2–11 percentage points) and the per-sequence spread
     (−5% to 57%);
   - the tie rule.

   The split is now quoted as about 1/4–1/3 (merge), 2/3–3/4 (decomposition)
   and small (degeneracy).
4. **H4 (R1, C2).** The observable-only mass is 2.9p (m = 2) to 22.8p (m = 22)
   on average. Single circuits range over 0–31p (0–50p on new sequences).
5. **H5 (summary 4, R5, R6, C15).** "29%" is now marked as the (5e-4, m = 6)
   value; across the grid it is 0.6–32%. The A0 factor is now marked as
   depending on the sequence set (27× and 43× on new sequences).
6. **H6 (summary 5, R7, C17).** 0.339 p_RB is now stated as the first-order
   value (61/180). The exact values are 0.338 at p_RB = 5e-3 and 0.333 at
   0.05.

## References (each fetched by me)

1. O. Higgott, C. Gidney, "Sparse Blossom: correcting a million errors per core
   second with minimum-weight matching," Quantum 9, 1600 (2025),
   arXiv:2303.15933. Its abstract describes PyMatching 2.
2. A. G. Fowler, "Optimal complexity correction of correlated errors in the
   surface code," arXiv:1310.0863 (2013). Title, author and year come from the
   arXiv page. I could not extract the PDF text, so the algorithm details are
   UNVERIFIED. The two-pass description comes from ref. 4.
3. C. Gidney, M. Newman, P. Brooks, C. Jones, "Yoked surface codes,"
   arXiv:2312.04522 (2023), Sec. 4. Quotes: "form a detector connecting to all
   the boundary edges on one side of the error graph", "turning this boundary
   detector on/off forces the decoder to match/not match", and the gaps are
   "well-calibrated ... after rescaling the gap by 0.9x".
4. PyMatching 2.3.1 docstrings of `Matching.from_detector_error_model` and
   `Matching.decode_batch` (installed package, read locally). Quotes: "Parallel
   edges are merged ... the `logical_observable` indices associated with the
   first added parallel edge are kept"; `enable_correlations`: "two-pass
   correlated matching ... see https://arxiv.org/abs/1310.0863".
