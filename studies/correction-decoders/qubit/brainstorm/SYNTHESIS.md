# Synthesis: correction versus detection for LRB on the qubit [[5,1,2]] code

Synthesis lead, 2026-10-28. This file merges four inputs:

- the eight literature notes in `lit/` and their eight verification reports (`*.verify.md`);
- the six brainstorm lenses in `brainstorm/` (decoder-theorist, numerics-statistics, rb-theorist,
  skeptic, superconducting-hardware, other-platforms);
- `CONTEXT.md` and `API.md`;
- the lead session's finished main grid (`results/grid_main/`) and first-order ledger
  (`results/first_order/`), read at about 11:35.

I edited none of those files. I wrote only this file, plus throwaway scripts in my session
scratchpad (Appendix A).

## How to read this file

Every claim carries one of the four CONTEXT.md labels, abbreviated as follows.

- **[VA]** verified analytically.
- **[VN]** verified numerically. The bracket names the source.
  - "lead grid" is the lead's production output: 30 sequences from seed 20261028, depths
    0, 2, 4, 6, 10, 14, 18, 20, 22, 1e6 shots per LRB-D circuit and 3e5 decoded shots. "Synthesis
    post-processing" means I computed the number from those files. These numbers are
    preliminary. They use CSV means over sequences and unweighted log-linear fits, with no
    confidence intervals yet.
  - "lead first-order" is `first_order_summary.json`: exact single-fault coefficients R1 (LRB-D
    rejection), E1 (exact-MLD logical error) and A2 (LRB-D second-order accepted error), 30
    sequences, data5 readout.
  - "lens X, lead circuits" means a lens ran its own check on `src/lrb_qubit` circuits. Only the
    decoder-theorist did this; verifiers 03, 05 and 07 also used or read those circuits.
  - "lens X, toy" means the lens's own small model. The skeptic's toy has no random Cliffords,
    and the other lenses used their own Clifford catalogues. Treat toy numbers as orders of
    magnitude.
- **[LIT Key]** from literature. Keys resolve in section 4. A key marked (b) is a brainstorm-only
  citation that no lit verifier checked; I confirmed its metadata myself (section 4.9).
- **[CONJ]** conjecture.

v01 to v08 name the verification reports of lit notes 01 to 08.

Units and names. "Per depth" means per unit of the benchmark depth m: one forward Clifford, one
full check round with reset, and one inverse Clifford. 1−f is the per-depth decay in
F = b f^m. "data2" is the qutrit-style terminal readout of data 0 and 3. "data5" adds the two
final X-stabilizer detectors from an ideal X readout of all five data qubits (the `_final`
columns, `mld5` and `const0`). κ = −d ln A/dm is the per-depth rejection rate. The normalized
curvature of a curve y(m) is κ_c = 2 c2 (m_max − m_min)/|mean slope| from a quadratic fit of
ln y on m ≥ 6; values near 0 mean exponential.

## 0. Where things stand

1. Distance 2 caps correction at first order. Exact ML decoding over all rounds (`mld5`) decays
   at (1−f)/p = 23 to 29 per depth for p ≤ 7.5e-3. That is 7 to 9 times the matched physical
   qubit's 3.3, with no crossing at any p [VN lead grid, synthesis post-processing]. The
   single-fault ledger gives 11.6p of logical error per depth, so 1−f ≈ 23p
   [VN lead first-order]. See H1.
2. About 77% of that first-order error comes from data qubits idling through the four sequential
   measurement layers and the reset layer of each round. Measurement and reset flips contribute
   nothing at first order [VN lead first-order]. See H2.
3. The best acceptance rule is a threshold on the exact posterior. LRB-D `unif=1` with the final
   data detectors lies on that frontier. Every round-skipping policy is dominated: at p = 1e-3,
   m = 22 the frontier at `const=1`'s acceptance has 13 to 15 times lower infidelity
   [VA; LIT Chow1970, CombesFerrie2015, Seip2026; VN lead grid, synthesis post-processing].
   See H8.
4. The winner depends on the metric. Accepted-shot F favours `unif=1` detection. Scoring
   rejected shots as coin flips (A·F) or as sampling cost (A·F²) favours MLD correction over
   every check-every-round LRB-D policy. The no-check circuit `const0` beats everything
   [VN lead grid, synthesis post-processing]. See H10 and H14.
5. The data2 readout leaves an undetectable 68p/15 ≈ 4.53p logical floor. It lowers b, not f.
   data5 removes it [VN lead; v03; v05; lead grid]. See H12.
6. At d=2 the check rounds are the noise. `const0_mld` decays at 1.5p per depth against 3.3p
   for the physical qubit at the repo's matching factor k = 5 (p_RB = 5p). At Gottesman's
   same-hardware matching, k = 1, the physical qubit wins. The break-even factor is k* ≈ 2.3
   [VN lead grid; VA]. See H14.
7. For Q2, the curvature of LRB-D curves is a normalization effect. A·F stays close to
   exponential, while F = (A·F)/A bends because A approaches a floor. For `const=1` that floor is
   1/16, and the grid confirms it [VA; VN lead grid]. MLD-corrected curves are exponential
   within noise (κ_c ≈ −0.02). The null-decoder control is still missing. Until it exists,
   "correction removes non-Markovianity" cannot be told apart from "no postselection"
   (H17 to H19).
8. For Q3 the strongest sustainable proposals are:
   - offline Pauli-frame decoding, with no real-time loop;
   - a check-free, fold-aware protocol on superconducting hardware;
   - erasure heralds, the only route to genuine correction at d=2;
   - trapped ions for a checks-every-round correction-versus-detection comparison.

   Most of these hardware claims are still toy-level (H24 to H30).

### Lead-grid snapshot used as evidence

All entries are [VN lead grid]. Fits and rates are synthesis post-processing.

| quantity | p = 1e-3 | p = 2e-3 | p = 5e-3 |
|---|---|---|---|
| mld5 F at m = 0, 10, 22 | 0.978, 0.757, 0.563 | 0.956, 0.556, 0.296 | 0.893, 0.191, 0.031 |
| mld5 (1−f)/p, fit on m ≥ 2 | 24.4 | 25.6 | 28.1 |
| unif=1 + final: A, F at m = 22 | 0.093, 0.9978 | 0.0087, 0.9905 | 7.5e-6, 0.95 (about 7 accepted shots per sequence, unreliable) |
| unif=1, data2: F at m = 0 and 22 | 0.9907, 0.9884 | 0.9811, 0.9711 | 0.9498, 0.912 (unreliable) |
| const=1 + final: A, F at m = 22 | 0.254, 0.915 | 0.100, 0.588 | 0.053, 0.029 |
| const0 detection: A, F at m = 22 | 0.926, 0.9993 | 0.858, 0.9971 | 0.695, 0.981 |
| const0_mld: F(22), (1−f)/p | 0.968, 1.5 | 0.937, 1.5 | 0.850, 1.5 |
| physical RB at p_RB = 5p: F(22), (1−f)/p | 0.928, 3.3 | 0.862, 3.3 | 0.689, 3.3 |
| unif=1 + final rejection rate κ/p | 104.9 | 104.6 | 104.3 |

First-order ledger, per depth over m = 10 to 22 [VN lead first-order]: R1 = 105.1p, E1 = 11.58p,
A2 = 47.3p², undetectable logical mass 4.533p under data2 and 0 under data5.

### Open disagreements between lenses

| # | Question | One position | Other position | Status and my read |
|---|---|---|---|---|
| D1 | Which terminal readout to headline | ideal data5 (decoder-theorist; lit 03 I8; lit 05) | only partial ideal or full noisy (skeptic Idea 6) | report all three readouts; use noisy data5 for hardware claims (H13) |
| D2 | Does correction mitigate non-Markovianity? | yes: it removes the syndrome-conditioning memory, like a leakage-reduction unit (lit 02 I5, I8; lit 05 3b; lit 08 I15; decoder-theorist §6) | only if MLD is flatter than the null decoder, otherwise it is just "no postselection" (skeptic Idea 11). QEC machinery can create non-exponential decay in constructed cases (Ceasura2022, v06). The unchecked inverse half may keep LRB-C non-exponential (lit 03 I12) | grid: MLD exponential so far; null control missing (H19) |
| D3 | Does d=2 correction beat the physical qubit? | const0 + MLD does (decoder-theorist idea 4) | the verdict is set by the arbitrary k; Gottesman's comparison is k = 1 (skeptic Idea 2) | both numbers are right: it wins at k = 5 and loses at k = 1, with k* ≈ 2.3 (H14) |
| D4 | Can within-pair tie-breaks be learned without labels? | no, syndrome statistics see only the pair sum (skeptic Idea 7) | yes, through Y-signature correlations (SC lens P7) | both hold: identifiable only if p_X = p_Y per qubit, which fails for pure dephasing (H6) |
| D5 | Is ML-merged MWPM adequate? | yes for a comparison mode; default PyMatching is 10% to 93% worse (SC lens P2) | no better than null in LRB circuits with S gates (decoder-theorist §3) | depends on the circuit: memory versus LRB (H4) |
| D6 | What to run on hardware | decode every round, with active reset and a multi-round decoder (lit 02 I8; lit 07 I3) | check-free LRB-C, with checks only in calibration runs (SC lens P1; decoder-theorist idea 4; skeptic Idea 12) | these are different benchmarks: LRB of a QEC cycle versus RB of logical gates (H28) |
| D7 | const=1 two-sector law: does r from F equal r from A? | yes (rb-theorist: qutrit 0.349 vs 0.332; qubit toy 0.310 vs 0.318) | grid: 0.117 vs 0.129 at p = 2e-3, but 0.294 vs 0.385 at 5e-3 | partially; the law probably needs an onset term (H17) |
| D8 | Is there a first-cycle transient? | expected from detection theory (KumarFoxmanDing2026, KumarTsunoda2026; lit 02, lit 08) | the ideal encoding may remove it (lit 04 §3.4); lone readout flips cause rejection, not accepted leakage (v02) | grid: small (≤ 1.3× with data5); the data2 excess is mainly a terminal-round effect (H21) |
| D9 | Leakage under correction | correction suffers unless an LRU runs every round (other-platforms Idea 3; SC lens P9) | check-free fold confinement corrects leakage that stays on {2,3} (SC lens P1) | depends on the protocol; stim cannot test it (H26) |

## 1. Hypothesis registry

Status words: "confirmed" means the lead grid or lead first-order ledger agrees, though
confidence intervals may still be missing. "Supported" means lens numerics or literature agree
but nothing has been checked on the lead grid. "Partial" means part of the prediction failed.
"Open" means the evidence is conjecture only. Priority runs from P1 (needed for a headline answer)
to P4.

### 1.0 Summary table

| ID | Question | Hypothesis in one line | Status | Priority |
|---|---|---|---|---|
| H1 | Q1 | Full MLD correction decays at (1−f)/p ≈ 23-29, 7-9× the physical qubit, with no pseudo-threshold | confirmed | P1 |
| H2 | Q1 | About 77% of MLD error is data idling in measurement and reset layers | confirmed at first order | P2 |
| H3 | Q1, Q2 | MLD's whole gain over no decoding is nonlinear Y inference; the covariant linear decoder equals the raw readout | supported | P2 |
| H4 | Q1 | Matching-family decoders are no better than null at d=2; BP-OSD/LSD are close to MLD | supported | P2 |
| H5 | Q1 | Fold-gate faults alias with weight-1 errors, costing about p/3 per CZ/SWAP(2,3) | supported | P3 |
| H6 | Q1, Q3 | MLD is robust to noise scale, fragile to noise shape; tie-breaks need labels or a p_X = p_Y assumption | supported, contested | P2 |
| H7 | Q1, Q3 | One round of lookahead suffices for a causal decoder | open | P3 |
| H8 | Q1 | The posterior-threshold frontier dominates every LRB-D policy; unif=1 + final lies on it | confirmed | P1 |
| H9 | Q1 | A "correct time-like, reject space-like" rule cuts rejection from 105p to 65p per depth | supported | P3 |
| H10 | Q1 | The ranking flips with the metric; unconditional metrics favour correction and const0 | confirmed (null control missing) | P1 |
| H11 | Q1, Q2 | Postselected decay rates measure the policy; the noise moves into A(m) | confirmed | P2 |
| H12 | Q1 | data2 leaves a 4.53p undetectable floor that lowers b, not f | confirmed | P1 |
| H13 | Q1, Q3 | Ideal data5 flatters everyone; noisy data5 roughly doubles MLD's m = 0 error | supported | P2 |
| H14 | Q1, Q3 | Checks are the noise: const0 + MLD gives 1.5p, beats k = 5 physical, loses to k = 1 | confirmed | P1 |
| H15 | Q3 | Without checks, any error confined to {2,3} is corrected exactly | supported | P2 |
| H16 | Q1, Q3 | One parallel measurement layer per round halves MLD error | supported | P2 |
| H17 | Q2 | const=1 acceptance → 1/16 and F follows a two-sector law | partial | P2 |
| H18 | Q2 | LRB-D curvature is normalization: A·F is exponential, A is log-convex | confirmed | P1 |
| H19 | Q2 | Correction decays exponentially, but so does the null decoder | confirmed for MLD, null missing | P1 |
| H20 | Q2 | Noisy sequential checks carry short-range memory; unif=k bulk rate ≈ k × unif=1 | partial | P2 |
| H21 | Q2 | Start-up and terminal transients bias fits that include m = 0 | confirmed | P3 |
| H22 | Q2 | Unchecked segments (const0) decay as a Gaussian in m | confirmed | P3 |
| H23 | Q2 | The check-free inverse half carries under 15% of MLD failures | open, contested | P4 |
| H24 | Q3 | Offline Pauli-frame decoding is exact; one dataset serves every policy and decoder | supported | P3 |
| H25 | Q3 | Erasure heralds make d=2 correction work: 1−f ∝ (1 − R_e) | supported | P2 |
| H26 | Q2, Q3 | Leakage hurts correction more than detection | open | P3 |
| H27 | Q2, Q3 | Coherent global dephasing makes LRB-D state-dependent; correction makes it Pauli | supported (one round) | P3 |
| H28 | Q3 | On IQM-class SC hardware, check-every-round LRB is shallow; check-free LRB-C is the sustainable protocol | supported, contested | P2 |
| H29 | Q3 | Layout: virtual SWAP, long-range CZ(2,3), and fold-aware qubit assignment | supported | P3 |
| H30 | Q3 | SI1000 overstates rejection 3-5× for ions and atoms | supported | P3 |

### A. Decoders at distance 2 (Q1)

#### H1. Full ML correction is first order in p, about 7 to 9 times the matched physical qubit, with no pseudo-threshold

**Prediction.** For exact MLD over all rounds with data5 (`mld5`), (1−f)/p lies in [21, 30]
for 1e-4 ≤ p ≤ 7.5e-3 and rises slowly with p. The ratio to physical RB at p_RB = 5p stays
between 6 and 10 at every p, so the two curves never cross.

**Evidence.**
- [VA] Every weight-1 X or Z error on qubits 0, 2, 3, 4 shares its syndrome with a partner that
  differs by a weight-2 logical: X0/X3, X2/X4, Z0/Z2, Z3/Z4. v01, v02, v04, v05, v06 and all six
  lenses confirm the CONTEXT syndrome table.
- [VN lead first-order] E1 grows by 11.58p per depth (m = 10 to 22), so 1−f ≈ 23.2p at first
  order.
- [VN lead grid, synthesis post-processing] mld5 (1−f)/p = 23.2, 23.5, 23.8, 24.4, 25.6, 28.1,
  29.4 at p = 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3. Physical RB gives 3.1 to 3.4 at every
  p. mld2 has the same slope.
- [VN decoder-theorist, lead circuits] 10.7p per depth at first order (3 sequences). It
  predicted 21 to 25 for p ≤ 2e-3 rising to about 30 at 5e-3, which the grid matches.
- [VN numerics, own model] ML F(22) = 0.523 at p = 1e-3; the grid gives 0.563.
- [VN skeptic toy, no Cliffords] 17p of polarization loss per round. [VN rb-theorist toy] about
  23p per block.
- [LIT Gottesman2016] a 4-qubit d=2 code "can only improve error rates via post-selection".
  [LIT Andersen2020, Marques2022, Krinner2022] superconducting experiments ran d=2 as detection
  and began correction at d=3. [LIT Kabir2025] a distance-2 syndrome "generally does not provide
  enough information to identify a unique correction" (v08).

**Status.** Confirmed on the lead grid for p ≤ 7.5e-3. Fits at p ≥ 1e-2 are invalid because F
has reached the estimator floor (pitfall 8).

**Test.** Cluster-bootstrap GLS fit of b f^m to `mld5` and `physical_rb` on m ∈ {2, ..., 22}
for every p ≤ 7.5e-3. H1 holds if every (1−f)/p falls in [21, 30] and the 95% CI of the
mld5-to-physical ratio lies inside [6, 10] at every p. At the lowest p, (1−f)/p should equal
2·E1 slope = 23.2 within 0.5.

**Priority.** P1. Nearly settled; it still needs confidence intervals.

#### H2. About three quarters of the d=2 correction error comes from data idling through the measurement and reset layers

**Prediction.** Zeroing data idle noise in the measurement and reset layers
(`SI1000.scaled(meas_idle=0.0, reset_idle=0.0)`) lowers mld5 (1−f)/p at p = 1e-3 from 24 to
about 5 or 6. The first-order value is 2 × (11.58 − 7.12 − 1.78) = 5.4. The same ablation lowers
unif=1's κ/p from 105 to about 47. Zeroing two-qubit noise instead removes only about 17% of
the MLD error.

**Evidence.**
- [VN lead first-order] E1 per depth by source: measurement-layer idle 7.12p, reset-layer idle
  1.78p, two-qubit gates 1.94p, one-qubit gates 0.24p, measurement and reset flips 0. The idle
  share is 77%. R1 per depth by source: measurement flips 20p, reset flips 8p,
  measurement-layer idle 48p, reset idle 10p, measured-ancilla depolarization 2.7p, two-qubit
  14.5p, one-qubit 1.9p.
- [VA] All five DEPOLARIZE1(2p) layers act identically on both members of each ambiguous pair,
  so no prior can break the tie (decoder-theorist §2; SC lens P1(c)).
- [VN decoder-theorist, lead circuits] 9.1p of the 10.7p MLD error comes from pure X or Z coin
  flips. [VN skeptic toy] about 85% of the corrected loss comes from the idle layers.
  [VN numerics] about 50p of the 107p visible fault rate per round is data idle.

**Status.** Confirmed at first order. The finite-p ablation has not been run.

**Test.** Rerun 30 sequences at p = 1e-3 and 5e-3 under each ablation. Pass if (1−f)/p and κ/p
land within 15% of the first-order predictions.

**Priority.** P2.

#### H3. MLD's entire advantage over no decoding is nonlinear X-Z (Y-error) inference; the covariant linear decoder equals the raw readout

**Prediction.**
- (a) The Clifford-covariant linear decoder, which corrects X3, X4 for Z-check syndromes and
  Z2, Z4 for X-check syndromes, returns the raw x0⊕x3 on every shot.
- (b) MLD restricted to the X-check detectors stays within 10% of the null decoder.
- (c) The null decoder has (1−f)/p ≈ 40, so MLD/null ≈ 0.55 to 0.6.

**Evidence.**
- [VA, VN enumeration; rb-theorist §3.3] A decoder linear in the syndrome bits makes F a product
  over fault mechanisms. The covariant one never anticommutes with X0X3, Y0Z2X3 or Z0Z2, so its
  output equals the raw readout. Mixed choices fail 20 of 32 covariance checks.
- [VN decoder-theorist, lead circuits] Y-type faults cost 7.48p per depth without decoding and
  0.07p with MLD. The total falls from 20.2p to 10.7p.
- [VN skeptic toy] null 5.82%, X-only MLD 5.69%, full MLD 3.70% (p = 1e-3, r = 3).
- [VN code capacity, lit 04 Table 3, reproduced by v04] joint decoding resolves Y faults; only
  pure-Z faults on 0, 2, 3, 4 stay ambiguous. [VA, rb-theorist] first-order flips on the X, Y, Z
  readout axes fall from (4, 6, 4)/15 to (2, 4, 2)/15.

**Status.** Supported by four independent checks, none on the lead grid.

**Test.** On the stored decode shots, or 1e5 fresh shots per circuit at p ∈ {1e-3, 5e-3}, 10
sequences and all depths, add the null, covariant-linear and X-only-MLD outputs. Check (a) by
exact per-shot equality, (b) and (c) by fitted rates.

**Priority.** P2. It is cheap, and H10 and H19 need its null control.

#### H4. Matching-family decoders fail at distance 2; BP-based decoders come close to MLD

**Prediction.**
- (a) Default PyMatching on Stim's decomposed DEM is no better than the null decoder, and its
  result moves when the DEM instruction order is shuffled.
- (b) Merging parallel edges by ML class removes the order dependence but leaves MWPM within 10%
  of null in LRB circuits that contain S gates.
- (c) BP-OSD (OSD-CS), BP-LSD and belief-find on the undecomposed DEM decay at MLD's rate, with b
  5% to 15% lower.
- (d) Unweighted union-find is worse than null.

**Evidence.**
- [LIT PyMatchingDocs] parallel edges are merged and keep the observables of the first added
  edge, "since the code has distance 2 if parallel edges have different fault_ids anyway".
  [LIT StimDocs] decomposition splits hyperedges into pieces of at most two detectors.
- [VN lit 03, reproduced by v03] a two-line DEM flips its prediction with line order, with and
  without `enable_correlations=True`. `add_edge` without `weight` stores weight 1.0.
- [VN decoder-theorist, lead circuits, first order] error relative to MLD at m = 6: BP-OSD 1.08,
  belief-find 1.15, LLR-weighted UF 1.29, class-merged MWPM 1.60, null 1.68, default PyMatching
  1.83, unweighted UF 2.31. With S gates present, 26% of detector sets carry both logical
  classes.
- [VN numerics, own model] at p = 1e-3, m = 10: PyMatching F = 0.562, null 0.622, ML 0.742.
  [VN skeptic toy] MWPM 7.7% against null 5.8%.
- [VN SC lens, memory circuits] default PyMatching is 10% worse than ML-merged MWPM with uniform
  noise and 93% worse with weak qubits at {0, 4}.

**Disagreement.** D5. Each lens is right for its own circuits. Memory circuits have no frame
changes, so ML-merged MWPM keeps most of its value there. LRB circuits need the Y correlation,
which matching throws away.

**Test.** Decoder panel on 5 sequences × m ∈ {2, 10, 22} × p ∈ {1e-3, 5e-3}, 1e5 shots, every
decoder on the same shots. Score paired Rao-Blackwell differences against MLD (numerics Idea
4). Run default PyMatching under five shuffled DEM orders.

**Priority.** P2. It decides which decoders the report may use at all.

#### H5. With checks every round, fold-gate faults alias with weight-1 errors and cost about p/3 per CZ(2,3) or SWAP(2,3)

**Prediction.** Per-sequence (1−f) of mld5 regresses on the per-depth count of fold gates with
slope near p/3 per gate (95% CI inside [0.2p, 0.5p]). The LRB-D accepted error has no
first-order slope. Implementing SWAP(2,3) as a relabel lowers mld5's 1−f by about p/3 per H.

**Evidence.**
- [VN lit 06 I4, reproduced by v06] X2X3, Z2Z3, Y2Y3, X2Z3 and Z2X3 share syndromes with X1, Z1,
  Y1, Y4 and Y0 and differ from them by a logical. [VA] 5 × p/15 = p/3. ML makes the same
  choice, because single-qubit errors are far likelier than p/15 (v06).
- [VA, v03] the (1,1) alias exists only in intervals that contain S or H.
- [VN decoder-theorist, lead circuits] fold-pair faults leave 0.78p of the 10.7p MLD error.
  [VN lead first-order] two-qubit gates contribute 1.94p per depth to E1.
- [VN SC lens toy] 7 of 15 CZ(2,3) Paulis become logical errors under the best single-round
  lookup. Skeptic Idea 14 and other-platforms Idea 10 reach the same count.

**Test.** Regress per-sequence mld5 fits (30 sequences, p = 1e-3) on the number of H and S per
sequence. Build the virtual-SWAP variant of H29 for 10 sequences.

**Priority.** P3.

#### H6. MLD is robust to the noise scale but fragile to the noise shape; within-pair tie-breaks are identifiable only with labels or an X-Y rate link

**Prediction.**
- (a) An MLD built at p' ∈ [p/4, 4p] loses less than 1e-3 in F at p = 1e-3, m = 10.
- (b) With Z-biased idle noise, η = p_z/(p_x + p_y) = 10, at equal total rate, MLD's gain over
  null drops below 10%. LRB-C's per-round error rises about 1.5×, while detection's rejection
  does not change.
- (c) With an unmodelled 2× rate asymmetry inside one ambiguous pair, a nominal-prior MLD decays
  faster than null.
- (d) If p_X = p_Y on each qubit, same-round cross-type detector correlations (Y signatures)
  recover per-qubit priors that match oracle-prior performance within 1%.

**Evidence.**
- [VN numerics, own model] decoders built at p' ∈ [2.5e-4, 4e-3] lose under 6e-4 at true
  p = 1e-3, m = 10. The reliability χ² still exposes the mismatch: about 11.5 for the true model,
  906 at p' = p/4 and 997 at 4p.
- [VN skeptic toy] MLD's gain over null is 36% at η = 0.5 and falls to 15%, 9% and 6% at η = 3,
  10 and ∞. A per-qubit rate mismatch makes the nominal-prior MLD worse than null (3.68% vs
  3.52%).
- [VA, VN other-platforms toy] pure dephasing makes 4/5 of single errors ambiguous, against 8/15
  for depolarizing noise; the corrected-error ratio is 1.50.
- [VA, skeptic; SC lens P7] for a pair with rates a and b, the X-syndrome statistics depend only
  on a + b − 2ab. [VN SC lens, memory] Y-edge priors give 0.00547 per round, against 0.00545
  for oracle priors and 0.01096 for uniform priors.
- [LIT Spitz2018] decoder weights can be estimated from detection data without an error model.
  [LIT (b) Wagner2022] syndrome data identify Pauli channels only up to the code's pure
  distance. [LIT (b) Sivak2024] decoder priors can be optimized on data.

**Disagreement.** D4. My read is that both lenses are correct. The tie-break is unidentifiable
in general and identifiable under the extra assumption p_X = p_Y, which pure dephasing breaks.

**Test.** Generate shots in the lead's model with perturbed noise:
- η ∈ {0.5, 3, 10, ∞} on idle and two-qubit channels;
- per-qubit log-normal rate scalings with σ ∈ {0.3, 0.7};
- swapped rates inside one pair.

Decode with nominal, Y-edge-calibrated and oracle priors and report each against null.
Separately, plot the mismatch curve F(p') at true p = 1e-3.

**Priority.** P2. Every hardware conclusion depends on it.

#### H7. A causal decoder needs one round of lookahead to match offline MLD

**Prediction.** A windowed MLD that commits after w rounds has decay rate r(0) ≈ 1.2 to 1.4 ×
r(∞), and r(1) within 5% of r(∞). Curvature stays near 0 for every w.

**Evidence.**
- [VA, rb-theorist Idea 10] for the final outcome of a Clifford-only circuit, global MLD equals
  the final-time output of a forward filter on the 64-state frame HMM, which Pauli-frame
  tracking can run online. Its rate is a smoothed per-block rate. It is not the infidelity of a
  time-local logical channel.
- [CONJ, rb-theorist; skeptic Idea 13] w = 0 loses the Y errors whose Z part shows in round t and
  X part in round t+1.
- [VN numerics, own model] an exact one-round-lookahead finite-state decoder gives F = 0.924,
  0.816, 0.709 and 0.543 at m = 2, 6, 10 and 22 (p = 1e-3). These agree with Monte Carlo within
  1.6σ and trail full ML by 0.0008 to 0.015.
- [LIT Skoric2023] sliding windows with a commit region and a buffer.

**Test.** Commit-after-w on the transfer-matrix forward pass for w = 0, 1, 2, ∞; 10 sequences,
p ∈ {1e-3, 5e-3}, 1e5 shots; paired bootstrap of the fitted rates.

**Priority.** P3. It matters only for real-time hardware claims.

### B. Acceptance rules and figures of merit (Q1)

#### H8. Thresholding the exact posterior is the optimal accept rule; unif=1 with final detectors is one point on that frontier and every round-skipping or data2 policy is dominated

**Prediction.**
- (a) Wherever each sequence has at least 30 accepted shots, the |LLR| frontier evaluated at
  the acceptance of unif=1 + final reproduces unif=1 + final's F within statistical error.
- (b) At matched acceptance the frontier beats `const=k` (k ≥ 1) and `unif=k` (k ≥ 2), and the
  margin grows with m. For `const=1 + final` at p ≤ 2e-3 the infidelity ratio is near 0.6 at
  m = 2 and 0.07 to 0.1 at m = 22.
- (c) Rejecting only shots with |LLR| < 0.25 changes A·F by about 1% or less at p ≤ 1e-3 and
  raises F_acc by more than 0.15 at m ≥ 10.
- (d) Any rule stricter than "accept all" still pays Θ(p) rejection per round at d=2, so A(m)
  decays exponentially. Even rejecting exact coin flips costs about 4p/3 per round at code
  capacity.

**Evidence.**
- [VA; LIT Chow1970, CombesFerrie2015] report the most probable class when its posterior is at
  least 1 − λ, otherwise reject; this minimizes error + λ × rejection. Decoder-theorist §4.1
  and numerics Idea 3 prove the fixed-acceptance form (a fractional knapsack, Neyman-Pearson
  style). [LIT Seip2026, Thm. VI.1] thresholding the maximum coset probability is Pareto-optimal
  under an abort-probability constraint. English2024's equal-cardinality version is wrong as
  stated (Seip2026 Remark G.2; v04).
- [VN code capacity, lit 04 Tables 1 to 3, reproduced by v04] the X-syndromes (1,0) and (0,1)
  have identical class weight enumerators, so their posterior is exactly 1/2 at every p.
- [VN decoder-theorist, lead circuits, exact at m = 3] unif=1 + final has accepted error
  1.85e-4, exactly the frontier value; const=1 is 2.2× above the frontier and other
  round-skipping policies 1.4 to 2.4×.
- [VN numerics, own model] at p = 1e-3, m = 10 the frontier lifts const=1's F from 0.9706 to
  0.9872 at matched acceptance; unif=1 lies on it.
- [VN lead grid, synthesis post-processing, pooled `gap__mld5` histograms]
  - unif=1 + final: infidelity ratio 0.98 to 1.03 at every m for p ≤ 2e-3.
  - const=1 + final at p = 1e-3, m = 22: A = 0.254, F = 0.915, frontier F = 0.994 (ratio 0.074).
  - const=1 at p = 5e-3, m = 22: F = 0.025 against a frontier F of 0.31.
  - coin-flip rejection at p = 1e-3, m = 10: F rises from 0.757 to 0.914 at A = 0.824, and A·F
    moves from 0.757 to 0.753.

**Status.** Confirmed on the lead grid with a pooled, binned frontier. Still missing:
per-sequence frontiers, confidence intervals, the other 11 values of p, and a randomized edge.

**Test.** Build the per-sequence frontier from `gap__mld5` (bins of 0.25 in |LLR|, with linear
interpolation at the edge, which is randomized acceptance). Compare it with every LRB-D
policy's per-sequence (A, F) under the cluster bootstrap. Claim dominance where the 95% CI of
F_frontier − F_policy excludes 0. LRB-D used 1e6 shots and decoding used 3e5, so compare on
matched shot subsets or propagate both errors.

**Priority.** P1. This is the main answer to "which acceptance rule".

#### H9. A "correct time-like, reject space-like" structural rule recovers much of the frontier without a posterior

**Prediction.** Accept a shot iff its detection events form a disjoint union of complete
measurement-flip signatures (under data2 the final round must also be clean), and decode
accepted shots as "no flip". Then:
- A(m) ∝ exp(−κm) with κ/p ≈ 65, against 105 for unif=1;
- the accepted error stays second order, with a per-depth coefficient near
  47p² + 40p × 2.2p ≈ 135p²;
- the accepted error is within 1.3× of the frontier at the rule's own acceptance.

**Evidence.**
- [VN decoder-theorist, lead circuits, one sequence] p = 1e-3, m = 22: A = 0.235 against 0.094
  for unif=1, with F = 0.9915 against 0.9979 (+ final). p = 5e-3, m = 14: A = 0.0091 against
  0.00049, 19× more, with F = 0.852 against 0.956. Exact at m = 3: 7% more error for 15% more
  acceptance. Without the disjointness constraint the rule accepts almost everything (A → 1/4,
  F → 0).
- [VN v03, lead circuits] time-like mass is 40.0p per interior round: 31p from measurement and
  reset, 8p from idle ancillas, 1.3p from gate faults. That is about 40% of first-order
  detections, and no logical-flipping single fault shares a time-like detector pair.
- [VA, decoder-theorist §4.4] each accepted time-like event costs about 2.2p of posterior error
  at d=2, because two complementary data faults in consecutive windows mimic a measurement
  error. At d ≥ 3 this confusion needs three faults.
- [VN rb-theorist toy] a soft rule at threshold t → 0+ keeps 0.87 of shots per round, against
  0.58 for unif=1 at p = 5e-3. [VN skeptic toy] a t = 0.5 hybrid reaches P·μ² = 0.887, against
  0.857 for MLD and 0.685 for unif=1.

**Test.** About 20 lines on the stored detector arrays; 30 sequences at
p ∈ {5e-4, 1e-3, 2e-3, 5e-3}; fit κ and compare F with the frontier at the same A.

**Priority.** P3. The frontier dominates it. Its value is that hardware can apply it without
computing posteriors.

#### H10. The ranking flips with the figure of merit, and the unconditional metrics favour correction and the no-check circuit

**Prediction.** For p ≤ 5e-3 and m ≥ 2:
- (a) by accepted-shot F, unif=1 detection beats MLD correction;
- (b) by P_eff = A·F_acc (rejected shots scored as coin flips) and by the sampling signal
  A·F_acc² against F², MLD correction beats every check-every-round LRB-D policy;
- (c) the null decoder also beats unif=1 on sampling signal;
- (d) `const0`, decoded or postselected, beats every check-every-round scheme on both
  unconditional metrics.

**Evidence.**
- [VA, lit 08 I1 to I3, confirmed by v08] per sequence, F_dec = A·F_acc + (1−A)·F_rej for any
  decoder that leaves trivial syndromes alone. So A·F_acc is detection that outputs a random bit
  on abort. v08 refines the first-order gain: F_dec − A·F_acc = P(qubit-1 fault) + P(ambiguous
  fault whose pair logical commutes with the readout frame) + prior-asymmetry terms.
- [VA, lit 04 §3.2, v04] P_eff = Σ over accepted s of P(s)(2g(s) − 1), so full ML correction
  maximizes it (Chow's rule at λ = 1/2). [VA, skeptic Idea 1] to estimate a ±1 observable to
  fixed precision costs about 1/(P_acc·μ_acc²) shots under postselection and 1/μ² under
  correction.
- [VN lead grid, synthesis post-processing] p = 1e-3, m = 22:

  | method | A·F (or F if A = 1) | A·F² (or F²) |
  |---|---|---|
  | MLD correction | 0.563 | 0.317 |
  | unif=1 + final | 0.093 | 0.092 |
  | const=1 + final | 0.232 | 0.212 |
  | const0 detection | 0.925 | 0.924 |
  | const0_mld | 0.968 | 0.937 |

  At p = 5e-3, m = 10, MLD's F² = 0.036 against A·F² = 0.0037 (unif=1 + final) and 0.015
  (const=1 + final).
- [VN skeptic toy] null P·μ² = 0.781 against 0.685 for unif=1 (p = 1e-3, r = 3).
- [VN numerics, own model] the relative-precision metric Γ = A F²/(1 − F²) favours unif=1 (3.3
  against 0.38 for ML at p = 1e-3, m = 22).

**Disagreement.** Which metric to headline. The skeptic and lit 08 favour unconditional or
shot-cost metrics. Numerics shows Γ favours unif=1. The skeptic adds that accepted F is the
right metric for single-shot tasks. My recommendation is to publish A, F_acc, A·F_acc and
A·F_acc² together for every method, paired on the same shots, and never headline F_acc alone.

**Test.** Add the null decoder. Compute the four metrics per sequence from counts and run a
paired cluster bootstrap. Claims (b) to (d) hold if the paired 95% CIs exclude 0 for all m ≥ 2
and p ≤ 5e-3.

**Priority.** P1.

#### H11. Under postselection the fitted decay of F_acc mostly measures the policy; the noise shows up in A(m)

**Prediction.** A rule with a fixed posterior-error cap ε (accept iff the chosen class has
posterior at least 1 − ε) gives F_acc(m) ≥ 1 − 2ε at every m. Its fitted f then stays within
1e-3 of 1 for p ≤ 2e-3, while κ = −d ln A/dm carries the noise. A global fixed threshold makes F
plateau; per-window thresholds keep it near exponential.

**Evidence.**
- [VA, decoder-theorist §4.1(iv); rb-theorist Idea 9].
- [VN lead grid, synthesis post-processing] unif=1 + final at p = 1e-3: the local decay rates of
  F_acc are 0.6e-4 to 1.2e-4 per depth, while κ/p = 104.3 to 105.0 for every p ≤ 5e-3. That
  equals the first-order R1 slope of 105.1.
- [LIT Levine2024] the dual-rail RB fitted acceptance and conditional survival separately (with
  a fixed 11 checks per circuit, v04). [LIT ChenBaldwin2025] postselected survival is a ratio
  and is single-exponential only in special regimes. [LIT HarperFlammia2019] "there is at
  present no complete theory for fitting an RB decay curve with post-selection".

**Test.** From the gap histograms, compute (κ, F_acc) for ε ∈ {e^-5, e^-7} with global and
per-window thresholds, and fit A(m) and F(m) separately.

**Priority.** P2.

### C. Readout and protocol (Q1, Q3)

#### H12. The qutrit-style data2 readout leaves an undetectable 68p/15 ≈ 4.53p floor that lowers b, not f

**Prediction.** Under data2 every decoder and every acceptance rule has an undetectable,
depth-independent first-order logical error of 4.53p, so F ≤ 1 − 9.07p + O(p²). mld2 and mld5
share f and differ in b. With data5, LRB-D's accepted error at m = 0 is about 37.5p².

**Evidence.**
- [VN lead, API.md] the undetectable faults hit data 0, 2 or 3 during the terminal
  X-ancilla measurement layers, about 4.5p in total. With data2 the circuit-level distance is
  therefore 1. With data5 it is 2 (Stim's `search_for_undetectable_logical_errors`).
- [VA, VN v03] 4p + 8p/15, a single DEM mechanism of 4.52p at depths 0, 1 and 4. [VN v05] the
  same at m = 3 and 6. A reset layer at the end of the final round would add 8p/3 (v03).
- [VN lead first-order] the undetectable mass is 4.533p at every depth under data2 and 0 under
  data5.
- [VN lead grid] at p = 1e-3, F(0) under data2 is 0.99071 against 0.99093 at first order. Under
  data5 it is 0.99992 against 1 − 2 × 37.5p² = 0.99993. mld5 against mld2: (1−f)/p 24.4 vs
  24.5 and b 0.971 vs 0.966 at p = 1e-3; 28.1 vs 28.3 and b 0.871 vs 0.836 at p = 5e-3.
- [VN decoder-theorist, lead circuits; skeptic toy, 4.7p; numerics, F_det(0) 0.950 → 0.998 at
  p = 5e-3; other-platforms toy] all agree.

**Status.** Confirmed.

**Test.** Nothing beyond confidence intervals. Every comparison must state its readout.

**Priority.** P1. Settled, but it must be stated in every comparison.

#### H13. An ideal five-qubit readout flatters both detection and correction; a noisy five-qubit readout is the realistic middle ground

**Prediction.** Put SI1000-style noise on the final five-qubit readout (DEPOLARIZE1(p) plus a
5p flip before MX). Then:
- LRB-D's m = 0 accepted error rises about 4× above its ideal-data5 value;
- MLD's m = 0 error roughly doubles;
- const0_mld's 1−f rises from 1.5p to about 2.0p;
- f of the check-every-round curves does not change.

**Evidence.**
- [VN skeptic toy, r = 0, p = 1e-3] detection 3.7e-5 (ideal full readout) → 1.45e-4 (noisy full
  readout); MLD 1.09% → 2.19%.
- [VN decoder-theorist, lead circuits] const0 MLD with a noisy terminal readout: F = 0.974 at
  m = 1 and 1−f ≈ 2.0p.
- [VA, lit 03 I8; lit 05 §3a] an ideal X readout of all data supplies a noiseless final X
  syndrome, which real hardware does not have.

**Disagreement.** D1.

**Test.** Add a builder option `terminal="data5_noisy"` and rerun LRB-D, mld and const0_mld at
p ∈ {1e-3, 5e-3} for 30 sequences.

**Priority.** P2.

#### H14. At distance 2 the check rounds are the noise: correcting only at the final readout beats every check-every-round scheme, and beats the physical qubit at k = 5 but not at k = 1

**Prediction.**
- (a) `const0_mld` keeps (1−f)/p = 1.5 ± 0.1 for p ≤ 1e-2, against 3.3 for physical RB at k = 5
  and 24 to 28 for mld5.
- (b) The break-even matching factor k* = (1−f)_L/(0.66p) is about 2.3 for const0_mld and 37 to
  43 for mld5, flat in p.
- (c) Idle noise DEPOLARIZE1(q) on data during logical-gate layers erases const0_mld's advantage
  over the k = 5 physical qubit at q ≈ 0.2p to 0.5p per layer.

**Evidence.**
- [VN lead grid] const0_mld (1−f)/p = 1.5 at every p ≤ 1e-2; physical RB 3.3. F(22) at
  p = 1e-3: 0.968 for const0_mld, 0.928 for physical RB, 0.563 for mld5. At p = 5e-3: 0.850,
  0.689 and 0.031. The decoder-theorist had predicted 0.968 and 0.853 against 0.929 and 0.684
  before the grid ran.
- [VN decoder-theorist, lead circuits] for an X-check flip in const0, MLD bets about 4.5:1 on
  the fold pair (2,3) over the quiet partners on 0, 1 and 4.
- [VA, SC lens P1(c)] ambiguous pairs differ by a logical, so intermediate checks cannot
  separate them; only priors can. [VN SC lens, memory toy] perfect checks after every step give
  P_L = 0.0746, the same as a final readout alone; noisy checks make it worse.
- [VA, skeptic Idea 2] physical RB loses 0.267 k L p per depth, with L ≈ 2.5 physical gates per
  Clifford here, i.e. 0.66 k p. Correction is also linear in p, so "beats physical" depends on
  k and not on p. [LIT Gottesman2016] the fair test runs the unencoded circuit on the same
  hardware, which is k = 1.
- [VN skeptic, qutrit production CSVs] at p = 5e-3, m = 22: const0 P·F² = 0.406, against
  2.7e-5 for const=1 and 9.3e-6 for unif=1 (m = 20); physical F² = 0.29.

**Disagreement.** D3 and D6.

**Test.** (a) and (b) come from the existing grid with cluster-bootstrap CIs; report k* for
every p. (c) needs a builder flag for gate-layer idle noise: SI1000's own Idle(p/10), plus
q ∈ {0.1p, 0.2p, 0.5p}.

**Priority.** P1.

#### H15. In the check-free circuit, any error confined to the fold pair {2,3} is corrected exactly by a 2-bit lookup at the final X readout

**Prediction.** With noise only on {2,3}, at any rate and including ZZ or conditional-phase
channels, const0 decoded with the fold lookup (1,0) → Z2, (0,1) → Z3, (1,1) → Z2Z3 gives
F(22) = 1 exactly. Injecting random Paulis on {2,3} after every CZ(2,3) leaves f_corr
unchanged while const0 detection's acceptance falls. The same injection on {0, 1, 4} lowers
f_corr by about (2/3) × the injection rate per injection point.

**Evidence.** [VA, SC lens P1(a)] every logical gate maps the Pauli group on {2,3} to itself, and
the 4-bit syndrome is injective on that group. [VN SC lens toy] F_corr = 1.00000 at m = 22 with
2% or 10% noise per gate on {2,3}; the best min-weight lookup gives 0.574 and 0.501, and const0
detection keeps 0.377 and 0.251 of shots. This agrees with the decoder-theorist's const0 MLD,
which bets on the fold pair.

**Test.** Add a noise-injection switch and the fold lookup to the lead's const0 builder; 10
sequences, m = 22, 1e5 shots.

**Priority.** P2. It is cheap, and the hardware protocol of H28 rests on it.

#### H16. Measuring all four ancillas in one parallel layer roughly halves the d=2 correction error and cuts LRB-D rejection by about a third

**Prediction.** With one measurement layer per round (reset layer kept), the first-order detected
mass falls from 105p to about 67p per depth and the MLD error from about 11p to 6p. So mld5
(1−f)/p ≈ 12 to 14 and unif=1 κ/p ≈ 67. Dropping the reset layer (XOR detectors) lowers the
MLD error further.

**Evidence.**
- [VN decoder-theorist, lead circuits, first order] detected mass 105.4p → 67.4p; MLD error
  10.7p → 6.0p; time-like mass 40p → 32p.
- [VN skeptic toy] MLD 17p → 7.9p to 8.5p; unif=1 rejection 100p → 63p. [VN other-platforms
  toy] detection probability per round 88p → 53p. [VN SC lens, memory] corrected error per round
  1.54% → 0.74% with parallel readout and 0.48% without reset (p = 1e-3).
- [VA, lit 07 I6, v07] k idle layers give total Pauli error (3/4)[1 − (1 − 8p/3)^k]. The repo's
  terminal round has 4 layers, not 5, because it has no reset.
- [LIT Gidney2021honeycomb] SI1000 applies ResonatorIdle(2p) once per measurement or reset step.
  [LIT Google2025] data idling during measurement and reset is about 20% of Willow's 1/Λ
  budget. [LIT Geher2025] no-reset beats reset when reset is slow. [LIT Hothem2025] measured
  mid-circuit-measurement errors exceed calibration predictions, by 4 to 5× in total layer error
  and 26× for the MCM error itself at n = 15 (v07). Treat the layer count as a sensitivity axis,
  not as a known correction.

**Test.** Builder option with all CNOTs first and one M layer; an optional no-reset variant;
p ∈ {1e-3, 5e-3}. Pass within 15% of the predictions.

**Priority.** P2.

### D. Decay shape and non-Markovianity (Q2)

#### H17. const=1 acceptance saturates at 1/16 and F_acc follows a two-sector law, not b f^m

**Prediction.** A_const1(m) → 1/16; the qutrit analogue is 1/81. For 1e-3 ≤ p ≤ 5e-3,
F_acc(m) ≈ F_c / [1 + (e^{rm} − 1)/16] fits at least 10× better than b f^m. The rate fitted to F
matches the rate fitted to A(m) ≈ A_0[e^{−rm} + (1 − e^{−rm})/16] within 15% at p ≤ 2e-3. The law
fails once F reaches the estimator floor.

**Evidence.**
- [VA] in the fully mixed limit four independent stabilizers give 16 equally likely outcomes
  (lit 08 I13 and v08; lit 02 I3 and v02 KF16).
- [VN rb-theorist, qutrit production] the two-sector law gives χ²/dof = 48 against 34,755 for
  b f^m, with r_F = 0.349 and r_A = 0.332 at p = 5e-3.
- [VN lead grid, synthesis post-processing]
  - A(22) = 0.0631 at p = 5e-3 and 0.0625 at every p ≥ 7.5e-3.
  - Sum of squared residuals in ln F, two-sector against b f^m: 0.0021 vs 0.033 at p = 2e-3,
    0.048 vs 0.65 at p = 5e-3.
  - r_F/r_A = 0.117/0.129 at p = 2e-3, but 0.294/0.385 at p = 5e-3.
- [VN rb-theorist toy, written before the qubit grid existed] predicted qubit const=1 F at
  p = 5e-3 on the depth grid: 0.944, 0.911, 0.836, 0.713, 0.401, 0.172, 0.067, 0.042, 0.026.
  The grid gives 0.950, 0.915, 0.839, 0.714, 0.394, 0.168, 0.064, 0.040, 0.025.
- [LIT WoodGambetta2018, Wallman2016, Chasseur2015] leakage RB relaxes to a non-zero steady
  state, and Wallman2016 allows the protected subspace to be a code space. [LIT Kabir2025]
  terminal detection gives curves "with non-Markovian structure".

**Status.** Saturation and the F shape are confirmed. r_F = r_A holds only at p ≤ 2e-3.

**Disagreement.** D7.

**Test.** GLS fits with the cluster bootstrap for b f^m, the two-sector law and the two-sector
law with a quadratic onset term; report a CI on r_F − r_A.

**Priority.** P2.

#### H18. LRB-D's non-exponential decay is a normalization effect: A·F_acc is close to exponential, and the curvature of F_acc comes from dividing by A(m)

**Prediction.** For every LRB-D policy at p ≤ 2e-3, ln(A·F_acc) has normalized curvature
|κ_c| ≤ 0.1 over m ≥ 6 and ln A is convex. For const=1, ln F_acc has κ_c ≤ −1.

**Evidence.**
- [VA, rb-theorist §3.1] for Pauli noise, N = A·F and A are each a sum of 16 exponentials in
  the stabilizer-sum form, while F = N/A has no such form. [VA, numerics Idea 6] N(m) is a linear
  functional of a product of transfer matrices. [LIT Kwiatkowski2025, Prop. 1] a Pauli-stochastic
  QEC cycle acts as a classical Markov chain on syndromes with syndrome-conditioned logical
  channels.
- [VN numerics, own model, p = 1e-3] κ_c = −1.37 for const=1's F_acc, +0.07 for N, +0.25 for A.
- [VN lead grid, synthesis post-processing, const=1 + final] κ_c of F_acc = −1.37, −1.57, −1.70
  and κ_c of A·F = +0.02, +0.04, +0.09 at p = 5e-4, 1e-3, 2e-3. At p = 5e-3 they are −0.63 and
  +0.21, with F already near 0 by m = 14. κ_c of A runs from +0.07 to +3.5. Caveat: I used mean
  F times mean A, not per-sequence products.
- [LIT HarperFlammia2019, ChenBaldwin2025, KumarFoxmanDing2026] postselected decays are ratios,
  or nonlinear maps of the state.

**Status.** Confirmed at p ≤ 2e-3, preliminary.

**Test.** Compute N(m) = (n+ − n−)/N per sequence from the count tables, then c2 and κ_c with
the cluster bootstrap, for every policy at p ≤ 5e-3.

**Priority.** P1. It is the main mechanism behind Q2.

#### H19. Correction at acceptance 1 decays as a single exponential, but so does no decoding; "correction mitigates non-Markovianity" reduces to "removing postselection removes the conditioning curvature" unless MLD beats the null control

**Prediction.**
- (a) mld5, mld2, const0_mld and the null decoder all have |κ_c| ≤ 0.15 over m ≥ 6 for
  p ≤ 5e-3, with flat local rates.
- (b) MLD is not flatter than null within the CIs. If it is, correction does more than remove
  postselection.
- (c) MLD that uses only the terminal check shows mild negative curvature, κ_c ≈ −0.1.

**Evidence.**
- [VN lead grid, synthesis post-processing] mld5 κ_c = −0.02, −0.02, −0.03 at p = 1e-3, 2e-3,
  5e-3, with local rates 0.024 to 0.028 per depth at p = 1e-3. const0_mld κ_c = +0.01 to +0.04;
  physical RB +0.03 to +0.04.
- [VN numerics, own model] all-round ML κ_c = +0.02 at p = 1e-3 (and +0.12 at 5e-4, CI includes
  0); terminal-check-only ML gives −0.14 and −0.08.
- [VN skeptic toy] per-round polarization ratios of null and MLD are constant to about 0.1%,
  while terminal-only detection accelerates.
- [VA, rb-theorist §3.3 and §3.4] linear decoders give exact products over mechanisms. MLD's
  nonlinearity is confined to decoding windows, so its curvature is O(p²).
- [LIT Ziyad2025] Markovian physical noise can give non-Markovian logical dynamics only if the
  data are not always returned to the code space (the paper proves this direction; the converse
  is shown by example, v01, v02). [LIT Kwiatkowski2025] the sufficient condition
  f_1 ≥ 1 − 1/64 fails here above p ≈ 5.2e-4 to 6.3e-4 (v02), so the theorem cannot certify
  most of the grid. [LIT Ceasura2022] constructed models in which QEC machinery makes LRB
  oscillate; v08 calls the construction adversarial and only a loose analogy for decoder memory.

**Disagreement.** D2.

**Test.** Run the null decoder on the same shots. Compute c2, κ_c and local-rate bands with the
cluster bootstrap for null, mld5, mld2 and terminal-only MLD; control the false discovery rate
across p.

**Priority.** P1.

#### H20. Noisy sequential checks carry short-range memory that renormalizes the rate without bending ln F; unif=k bulk rates scale with k

**Prediction.**
- (a) For unif=1, a lumped per-block Markovian surrogate (incoming syndrome 0) underestimates
  the bulk rate by 10% to 15% at p = 5e-3. Δ(m) = ln F − ln F_surrogate drifts linearly in m
  while c2 ≈ 0, so a flat curve does not certify Markovianity.
- (b) unif=k's bulk accepted-F rate is about k times unif=1's, with a period-k sawtooth in the
  windowed rates, and unif=k equals const=1 for m < k.

**Evidence.**
- [VA, rb-theorist §3.1] with ideal used checks and Pauli noise, F_acc factorizes exactly over
  the segments between used checks. Noisy checks pass memory forward with relative amplitude
  O(q·ε). [VN rb-theorist toy] readout masking carries about 13% of unif=1's bulk rate at
  p = 5e-3.
- [VN lead grid, synthesis post-processing, data2, m = 2 to 14] bulk-rate ratio unif=k/unif=1:
  1.65 to 1.95 for k = 2, 2.6 to 3.3 for k = 3, 3.4 to 4.4 for k = 4, over p = 2e-4 to 2e-3.
  Roughly proportional to k, and below k at the lowest p.
- [VN rb-theorist, qutrit] unif=3 and const=1 give the same F(2) = 0.9511.
- [LIT KumarFoxmanDing2026] check imperfections give first-order "accepted leakage". Per v02 its
  sources are late data faults and hook faults, not lone readout flips.

**Test.** Build the exact frame-HMM or transfer-matrix LRB-D evaluator (numerics Idea 2;
rb-theorist Idea 1) for the 30 sequences. Use noiseless MPP "genie" measurements at block
boundaries to measure the lumpability defect, and compare surrogate with exact.

**Priority.** P2. The exact evaluator also removes pitfall 7.

#### H21. Start-up and terminal-boundary transients bias fits that include m = 0

**Prediction.** unif=1's rate over m = 0 → 2 exceeds its 2 → 10 bulk by 1.6 to 1.9× under data2
and by at most 1.3× under data5. Setting the final round's X_ERROR to zero removes most of the
data2 excess. MLD's first window exceeds its bulk by at most 1.2×. Including m = 0 shifts the
fitted f by 5% to 10%.

**Evidence.**
- [VN lead grid, synthesis post-processing] unif=1 (data2) ratios 1.60, 1.78, 1.69, 1.86 at
  p = 5e-4, 1e-3, 2e-3, 5e-3; unif=1 + final 1.29, 1.26, 1.16, 1.20. mld5's 0 → 2 rate is 0.0282
  against a bulk near 0.025 at p = 1e-3.
- [VN rb-theorist] qutrit unif=1: 0 → 2 window 2.3× the bulk. Toy: 2.1×, falling to 1.19× with
  terminal readout flips off. [VN numerics, own model] the curvature of all-round ML is
  significant with m ≥ 2 and not with m ≥ 6.
- [LIT KumarFoxmanDing2026, KumarTsunoda2026] detection cycles show a first-cycle transient.
  v02: a lone readout flip causes rejection, so masking at the terminal check is a second-order
  (p × q) effect.

**Test.** Fit on {m ≥ 2}, on all m, and on all m with a free m = 0 offset; one simulation with
the final round's X_ERROR off.

**Priority.** P3.

#### H22. Unchecked segments decay as a Gaussian in m

**Prediction.** const=0 detection follows −ln F = a m^β with β = 2.0 ± 0.15 for p ≤ 1e-2.
Replacing the mirror inverse with one compiled recovery Clifford cuts the quadratic coefficient
about 4×.

**Evidence.** [VA, rb-theorist §3.2 and Idea 6] in an unchecked segment the first order cancels,
and pairs of cancelling faults grow as (mp)². [VN rb-theorist, qutrit] β = 1.99 to 2.32.
[VN lead grid, synthesis post-processing] β = 1.89 to 1.95 for p = 2e-4 to 1e-2 (m ≥ 2); local
rates rise roughly linearly, from 1e-5 to 6e-5 per depth at p = 1e-3.

**Status.** Confirmed, with β a little below 2.

**Test.** Recovery-Clifford variant on 30 sequences; compare the quadratic coefficients.

**Priority.** P3.

#### H23. The check-free inverse half carries a small share of MLD failures

**Prediction.** At m = 22, p = 1e-3, at least 85% of MLD failure probability comes from the
forward check rounds. Turning off inverse-half noise changes ln f by less than 10% at p ≤ 1e-3.
Adding checks to the inverse half would not visibly change unif=1 shapes with 30 sequences.

**Evidence.** [CONJ, numerics Ideas 14 and 20] about 2p per inverse Clifford against about 107p
per checked round. [VN rb-theorist toy] the inverse half adds c2 ≈ −0.2p² to unif=1. The
opposing conjecture is lit 03 I12: the final detectors alone see inverse-half faults, so that
failure is not a product of per-step factors. v03 confirms only the form,
F = Σ_s |P(s,+) − P(s,−)|, which is not a single exponential in general.

**Test.** Forward-backward attribution on the transfer matrix, or the ablation with inverse-half
noise off.

**Priority.** P4.

### E. Hardware (Q3)

#### H24. Offline Pauli-frame decoding is exact for Clifford LRB; one recorded dataset serves every policy and decoder

**Prediction.** For Pauli-dominated noise, physically applied per-round corrections and offline
Pauli-frame tracking give the same f within 1%. A larger gap points to non-Pauli effects of the
correction gates. One dataset that records all m+1 rounds yields every const ≥ 1 and unif
policy, every decoder and every hybrid.

**Evidence.** [LIT Knill2005, Terhal2015, Chamberland2018, Battistel2023] Pauli corrections can
stay in a software frame until the next non-Clifford gate. [LIT RyanAnderson2021] software-frame
and physically applied corrections gave 92(1)% and 93(2)% on a trapped-ion QCCD. [VA, v07;
other-platforms Idea 6; SC lens P10] every LRB circuit here is Clifford and ends in a Pauli
readout. Caveat from v07: const=0 is a different circuit and needs its own runs. [CONJ,
other-platforms Idea 6] applied corrections add about P_det × 1.5 × (one-qubit error) per round,
under 1% of the corrected error.

**Test.** Simulate decoder-conditioned noisy correction gates and compare with frame tracking.

**Priority.** P3. Confidence is already high.

#### H25. Erasure heralds turn the d=2 code into a correcting code; rejecting on heralds wastes acceptance

**Prediction.** Convert a fraction R_e of each error location into HERALDED_ERASE and decode with
a herald-aware MLD. Then:
- 1−f ≈ (1 − R_e)(1 − f(0)) plus a second-order floor, at acceptance 1;
- a hybrid that rejects only posteriors below 0.9 Pareto-dominates LRB-D at R_e ≥ 0.9;
- LRB-D that rejects on any herald loses acceptance as R_e rises;
- deleting a random fraction q of heralds raises the error linearly in q, and curvature would
  flag unfaithful heralds.

**Evidence.**
- [LIT Kubica2023, Grassl1997] a distance-d code corrects d − 1 erasures but only ⌊(d − 1)/2⌋
  unlocated errors. [LIT Knill2005] "error-detecting codes can always correct an error at a known
  location". [LIT Zhang2026] erasure-aware [[4,2,2]] decoding with no postselection raised
  fidelity from 0.874 to 0.902; postselection on erasure flags is a separate analysis (v05).
  [LIT Wu2022, Gu2025] erasure conversion raises thresholds; checks must be scheduled.
- [VA, VN lit 05 and v05] all 5 single erasures and 6 of the 10 pairs are correctable. At circuit
  level the fault order is 2 when the herald names the gate.
- [VN other-platforms toy, perfect next-round syndrome] corrected error 1.55e-2, 8.1e-3, 2.0e-3
  and 7.3e-4 at R_e = 0, 0.5, 0.9, 0.98. At R_e = 0.98 the hybrid keeps 99.55% of shots per round
  with error 9e-6, against 93.7% and 2.6e-5 for LRB-D. LRB-D that rejects on heralds falls from
  0.912 to 0.889 per round.
- Realistic R_e [LIT]: about 0.38 for the Yb CZ and 0.56 for one-qubit gates (Zhang2026's
  published numbers; lit 05's 0.31 is the v1 value); about 94% of Raman errors in metastable
  ions (Quinn2026); dual-rail transmons with residual errors about 40× below erasures
  (Levine2024).

**Test.** HERALDED_ERASE split in the lead's builder with R_e ∈ {0, 0.3, 0.5, 0.9, 0.98}. Make the
transfer matrix herald-aware by per-shot slot reweighting, cached on herald patterns
(other-platforms Idea 1); 10 sequences at p = 1e-3.

**Priority.** P2. This is the cleanest route to real correction at d=2.

#### H26. Leakage hurts correction more than detection

**Prediction.** With leakage ℓ per two-qubit gate and no leakage-reduction unit (LRU):
- LRB-D's accepted F keeps its leakage-free value and its rejection rises by about the per-round
  data leakage L_r;
- LRB-C's per-round error rises by about (3/4) L_r, which is 50% to 90% above the Pauli-only
  value at Helios-class rates, and F_corr(m) gains an accelerating component;
- a per-round LRU or optical pumping cuts the excess below 20%;
- in the check-free protocol, leakage that stays on {2,3} is corrected (H15).

**Evidence.** [LIT Miao2023, McEwen2021, Battistel2021, Xin2026, Marques2023, Varbanov2020]
leakage creates detection events correlated across rounds, and removing it every cycle makes it
behave like uncorrelated error. [LIT Google2025] leakage-induced time correlations confuse the
decoder. [LIT Ransford2025] on Helios, leakage is 2.4e-4 of the 7.9e-4 two-qubit error, and
transport leakage accounts for nearly all of the linear memory error. [LIT WoodGambetta2018,
Wallman2016, Chasseur2015] leakage RB decays are multi-exponential. [CONJ] other-platforms
Idea 3; SC lens P9; rb-theorist Idea 14; lit 07 I9.

**Test.** Stim cannot model leakage, so use a Pauli-frame Monte Carlo with a leaked flag per
qubit. On hardware, run arms with and without LRUs, plot the detection-event rate against round
index, and fit two exponentials to F_corr(m).

**Priority.** P3.

#### H27. Coherent global dephasing makes LRB-D acceptance depend on the logical state and pulls accepted states; correction turns it into plain dephasing

**Prediction.** Under U = exp(−iθ Σ_i Z_i) per round:
- LRB-D rejects |0>_L at 9θ² and |1>_L at θ² per round, and pulls accepted states toward |1>_L by
  4θ² per round;
- MLD correction gives unital logical dephasing, Bloch vector (1 − 4θ², 0, 0) for |+>_L, with no
  pull;
- the spread of acceptance across 30 sequences exceeds binomial;
- Pauli-frame randomization or dynamical decoupling removes both effects.

**Evidence.** [VN other-platforms, 32-dimensional statevector, one round, θ = 0.01 to 0.05].
[VA] each ambiguous pair's product is a logical on the code space, so the cross terms appear at
first order in θ². [LIT Ransford2025] Helios quotes 3e-5 of coherent error per depth-1 layer from
magnetic-field drift. [LIT FigueroaRomero2024] Pauli twirling reduces RB variance and dynamical
decoupling makes non-Markovian RB close to exponential. [LIT Endo2025] different definitions of
non-Markovianity can disagree.

**Test.** Nine-qubit statevector LRB (512 dimensions) with a coherent Z rotation per round,
against its Pauli twirl, for m ≤ 10.

**Priority.** P3.

#### H28. On IQM-class superconducting hardware, check-every-round LRB is limited to shallow depth, and the sustainable protocol is check-free LRB-C with offline decoding

**Prediction.**
- (a) IQM Garnet medians correspond to SI1000 p ≈ 5e-3 to 6e-3. There the grid gives unif=1 +
  final acceptance 3.9e-3 at m = 10 and 7.5e-6 at m = 22, and checks-every-round MLD F(10) ≈
  0.19. So check-every-round LRB-D is limited to m ≲ 6 to 10, and checks-every-round LRB-C
  reaches F ≲ 0.2 by m = 10.
- (b) Check-free LRB-C (the const0 circuit with fold-aware decoding of a five-qubit X readout)
  keeps F(22) ≈ 0.85 to 0.96 at p = 5e-3, depending on gate compilation.
- (c) The sustainable superconducting protocol is: check-free sequences; a transversal MX of all
  five data qubits with three-state discrimination; offline fold-aware decoding; the best
  qubits on the fold axis {0, 1, 4}; and check rounds only in separate calibration runs.

**Evidence.**
- [LIT Abdurakhimov2024] Garnet medians: one-qubit error 9e-4, CZ 5e-3, simultaneous readout
  3e-2, T1 of order 40 µs. [LIT (b) AWSBraketIQMpage] 20 qubits on a square lattice with tunable
  couplers, median one-qubit fidelity 99.92%, two-qubit 99.51%. [VA, SC lens P11] 99.51% gives
  DEPOLARIZE2 ≈ 6.1e-3, and the one-qubit error is about 2.7× SI1000's p/10 ratio (v07). Caveats
  from v07: Garnet quotes average infidelities, not Pauli strengths, and IQM's round times are
  not published.
- [LIT Vigneau2025] IQM's star-architecture [[4,2,2]] runs kept about 0.67 per cycle.
  [LIT Chen2021] Sycamore's d=2 code discarded 27% of runs per round. [LIT Andersen2020,
  Marques2022] 40% to 55% kept per round.
- [VN lead grid] the numbers in (a); const0_mld F(22) = 0.850 at p = 5e-3 with repo gates and
  ideal readout. [VN SC lens toy] check-free F(22) = 0.887 with repo gates and 0.956 with virtual
  SWAP and virtual-Z gates, at p = 5e-3.
- [LIT IQMdocs, AWSBraket2025] IQM supports mid-circuit measurement. Its reset is a measurement
  plus a classically controlled X and exists only where `cc_prx` does. Dynamic circuits on Garnet
  are experimental and limited to a local group. [LIT IQMHalocene2025] Halocene targets research
  on up to five logical qubits with "a modular decoder architecture".

**Disagreement.** D6. My read: check-free LRB-C benchmarks logical gates, not a QEC cycle. If the
study wants Combes-style LRB of the QEC cycle, it must keep the checks and accept the H1 floor,
or move to erasure-heralded hardware (H25).

**Test.** Simulate (a) and (b) with a Garnet-calibrated noise variant (average infidelities
converted to Pauli rates, T1 and T2 idling per layer) and a virtual-gate compilation option. (c)
is a protocol choice; H15 and H29 test its parts.

**Priority.** P2.

#### H29. Layout: the fold edge does not embed in a bipartite lattice, SWAP(2,3) is free as a relabel, and qubit assignment should follow the fold structure

**Prediction.**
- (a) Treating SWAP(2,3) as a frame relabel lowers check-free LRB-C's 1 − F(22) from 0.113 to
  0.044 at p = 5e-3. It lowers checks-every-round LRB-C error by about 0.35p to 0.4p per Clifford
  and leaves LRB-D almost unchanged.
- (b) Routing CZ(2,3) through the patch breaks fold confinement and adds about (route gates
  touching {0, 1, 4}) × p per logical S.
- (c) Two weak qubits in one ambiguous pair raise the corrected error by about 40%. Weak qubits
  on {2,3} cost about 13% and on {0,4} about 3%. Acceptance does not change.

**Evidence.** [VA, SC lens P5] the 12 check couplings form the 3×3 grid graph, whose square
embedding is unique: q1 sits in the centre and q2, q3 on opposite corners. The fold edge closes
the 5-cycle 2–3–a5–1–a6–2, so no bipartite lattice (square, heavy-hex) hosts it. [VA, lit 06
I11; v05; v06] the same corner geometry, and SWAP(2,3) can be virtual. [VA, other-platforms
Idea 10] H on all five qubits maps the stabilizer supports onto themselves with X and Z swapped.
[VN SC lens toy] a frame-bit test on 200 random depth-12 sequences, the virtualized-gate
numbers and the assignment table. [LIT (b) McKay2017] virtual Z gates. [LIT (b) Renger2025] IQM
Star MOVE-CZ-MOVE gates, qubit-qubit CZ fidelity 0.978. [LIT HarperFlammia2019, BDB2024]
permutations implemented by relabelling.

**Test.** Add virtual-SWAP and per-qubit idle-scaling options to the lead's builder.

**Priority.** P3.

#### H30. SI1000 conclusions do not transfer unchanged to ions and atoms

**Prediction.** At equal p, ion-like and atom-like profiles (one parallel measurement layer,
spectator error per readout of 0.0025 to 0.27 of the two-qubit error instead of SI1000's 2 per
layer) give 3 to 5× lower per-round rejection than the repo's SI1000, with corrected error about
0.14 to 0.19 × the per-round detection probability in every profile. On Helios-class ions
(p ≈ 8e-4), LRB-D keeps about 0.72 of shots at m = 22 but its accepted error, about 2e-6 per
round, cannot be resolved with 1e3 shots per circuit. LRB-C's first-order decay, about 2.6e-3
per round, is resolvable to 10% to 20%.

**Evidence.** [VN other-platforms toy, p = 1e-3, one forward round] detection probability
8.8% (SI1000 sequential), 5.3% (one parallel layer), 1.76% (ION), 2.15% (ATOM). [LIT Moses2023]
H2 measurement and reset crosstalk 4.5e-6 and 3.8e-6; shot times 0.18 to 20.6 s (v07).
[LIT Ransford2025] Helios two-qubit error 7.9e-4. [LIT Bluvstein2023, Muniz2025] atom readout
cost and atom loss per cycle. [LIT Hothem2025] measured MCM errors exceed predictions on IBM and
on H1-1. [CONJ, other-platforms Ideas 7 and 8] platform ranking: trapped ions, then metastable Yb
atoms, dual-rail superconducting qubits and alkali atoms. Cat qubits do not fit, because the
transversal H removes their noise bias [LIT (b) Puri2020, Putterman2025].

**Test.** Run the lead's model with the ION and ATOM `SI1000.scaled` profiles from other-platforms
Idea 2 at each platform's p; subsample to 1e3 shots per circuit and report CI widths on the
per-round error.

**Priority.** P3.

### 1.6 Suggested run order for the lead

1. Add the null decoder and the covariant linear decoder to the stored decode shots (H3, H10,
   H19). Minutes.
2. Statistics layer on the existing grid: signed pooled estimator, accepted counts per cell,
   cluster bootstrap with one resampling draw shared by all depths, policies, decoders and p, fits
   on m ≥ 2 for rates and m ≥ 6 for shape, and per-sequence frontiers (H1, H8, H10, H12, H17,
   H18, H19, H21, H22).
3. Exact zero-shot LRB-D: the transfer-matrix forward pass with all detection events set to zero,
   for every policy at p ≥ 5e-3, where sampling fails (H20; pitfall 7).
4. Readout variant data5_noisy (H13).
5. Ablations: idle off, two-qubit noise off, one parallel measurement layer, gate-layer idle
   (H2, H14(c), H16).
6. Decoder panel on a subset (H4) and windowed MLD (H7).
7. Check-free fold confinement and virtual SWAP (H15, H29).
8. Erasure variant (H25) and noise-shape robustness (H6).
9. Coherent statevector test, leakage Monte Carlo and platform profiles (H26, H27, H30).

## 2. Ranked decoder and acceptance-rule candidates

Ranked by value for the study's three questions, not by novelty. Expected outcomes cite the
hypotheses that carry their evidence.

| Rank | Candidate | Status | Expected outcome | Why |
|---|---|---|---|---|
| 1 | Exact transfer-matrix MLD (`mld.py`) with a posterior-threshold sweep, the Chow / Neyman-Pearson frontier, in global and per-window forms | implemented; run on the grid as mld5, mld2, const0_mld, with \|LLR\| histograms | (1−f)/p ≈ 23-28 at A = 1 for p ≤ 5e-3. The frontier passes through unif=1 + final and has 13-15× lower infidelity than const=1 at its acceptance (p = 1e-3, m = 22). Calibrated LLRs (H1, H8, H11) | exact at 32 states per round, 7-15 µs per shot at m = 22; by H8 no rule beats it at fixed acceptance |
| 2 | Null decoder (raw readout, accept all), identical shot by shot to the covariant linear decoder | trivial; not yet in the grid outputs | (1−f)/p ≈ 40; single exponential (H3, H19) | control that separates "correction" from "no postselection" in H10 and H19 |
| 3 | Fold-aware lookup or MLD on the check-free circuit (`const0_mld`) | run | 1.5p per depth with ideal readout, about 2.0p with noisy readout; immune to faults confined to {2,3} (H14, H15) | no symmetric idle noise from checks; it uses the fold-gate noise asymmetry; basis of the SC protocol (H28) |
| 4 | Herald-aware MLD on an erasure variant, plus a posterior-threshold hybrid | needs a builder option and slot reweighting | 1−f ∝ (1 − R_e); the hybrid dominates LRB-D at R_e ≥ 0.9 (H25) | the only route to genuine correction at d=2 |
| 5 | "Correct time-like, reject space-like" structural rule | about 20 lines | κ ≈ 65p against 105p, accepted error O(p²), near the frontier (H9) | needs no posterior; fits an FPGA lookup |
| 6 | Windowed causal MLD (w = 0, 1, 2) and a Wald-style sequential early abort | needs a commit step in the forward pass | w = 1 ≈ global MLD, w = 0 ≈ 1.3× the rate (H7). Early abort cuts executed forward rounds about 2.5× under unif=1-like thresholds at p = 1e-3, m = 22 [CONJ, decoder-theorist idea 13] | real-time relevance for hardware |
| 7 | Empirical windowed lookup trained on labelled LRB shots and scored out of sample | not built | within 10% of DEM-based MLD at p ≤ 1e-3 with about 1e6 shots [CONJ, decoder-theorist idea 14] | calibration without a noise model; LRB labels come free; must avoid in-sample scoring (pitfall 15) |
| 8 | BP-OSD (OSD-CS), BP-LSD, belief-find (ldpc 2.4.1) on the undecomposed DEM | installed | MLD's rate, b 5-15% lower, erratic per sequence (H4) | standard general-purpose decoders for the comparison table |
| 9 | ML-merged MWPM (explicit weights, SC lens Appendix A) and PyMatching's correlated mode | code exists in the SC lens | ≈ null in LRB circuits with S gates; about 10% better than default PyMatching on memory circuits (H4) | what a typical experiment would run; report it as such |
| 10 | Negative controls, not for results: default PyMatching (order dependent, ≥ null); unweighted union-find (2.3× MLD); "accept iff at most 1 detection event" (87× the frontier error at matched acceptance, p = 1e-3, m = 3); the structural "unique-class" hybrid (2.2-2.5× the frontier error); MWPM complementary-gap postselection (overconfident at d=2, at least 3× the frontier error [CONJ]); every round-skipping const=k and unif=k ≥ 2 policy (dominated, H8) | n/a | n/a | keep them only to show what fails and why |

## 3. Pitfalls and the check for each

| # | Pitfall | Check |
|---|---|---|
| 1 | The data2 readout has single faults that flip the logical with no detection event (68p/15 ≈ 4.53p), so its circuit-level distance is 1 | Compare F(0) with 1 − 9.07p; report data2 and data5 side by side; use the first-order undetectable mass as a regression check |
| 2 | An ideal five-qubit readout hands every method a noiseless final X syndrome | Add the noisy data5 variant (H13) and headline it for hardware claims |
| 3 | PyMatching keeps the first-added parallel edge's observable; `add_edge` without `weight` stores 1.0; correlated mode keeps the order dependence | Shuffle the DEM order and require identical results; pass `weight=log((1−q)/q)` explicitly |
| 4 | Stim's decomposition turns the competing coset into boundary pairs and may add remnant edges; PyMatching silently drops undecomposed hyperedges | Build MLD and BP decoders from `decompose_errors=False`; count hyperedges and decomposition fallbacks |
| 5 | Taking abs(P0 − P1) per sequence hides deterministic sign errors (wrong inverse, missed SWAP frame update, a decoder that always flips) | Unit test: signed X_L = +1 for every circuit at p = 0; inject a frame bug and confirm the signed metric catches it while abs() does not |
| 6 | abs(P0 − P1) is biased up at small accepted counts, by about sqrt(2/(πn)): 0.26 at n = 10. Empty cells become 0 in `extract_statistics` and NaN in `lrb_counts_to_statistics` | Report accepted counts per cell; use the pooled signed estimator with Wilson or Clopper-Pearson intervals; never set 0 or NaN silently |
| 7 | Tiny acceptance: unif=1 at p = 5e-3 has A ≈ 7.5e-6 at m = 22, about 7 accepted shots per sequence, and its F is non-monotone (0.913, 0.870, 0.912 at m = 18, 20, 22) with sequence spread 0.05-0.16. At p ≥ 7.5e-3, A(22) = 0 in the grid | Pre-register a minimum accepted count per sequence (30 or 100) for fits; compute zero-shot exact LRB-D with the transfer matrix (numerics Idea 2) |
| 8 | Once F nears 0, fitted decays become meaningless: mld5 fits at p ≥ 1e-2 return f = 0.74 to 0.99, with (1−f)/p falling to 0.2 | Drop depths with F below 5 standard errors; restrict check-every-round MLD rate tables to p ≤ 7.5e-3 |
| 9 | Fit window and weights decide the answer: on qutrit const=1 at p = 5e-3, 1−f spans 23× across windows; the m = 0 point carries 3200× the weight of m = 10; free-offset fits degenerate (offset −1e2 to −1e5) | Pre-register windows (m ≥ 2 for rates, m ≥ 6 for shape); weight by bootstrap standard errors; report local rates; same model for LRB and physical RB |
| 10 | Nested Clifford prefixes and the same 30 sequences at every p correlate all cells; the repo bootstrap treats depths as independent (about 86% coverage of a nominal 95% interval in a toy) | Cluster bootstrap over sequences, one draw for all depths, policies, decoders and p; check coverage on synthetic data |
| 11 | Mean of per-sequence ratios against pooled ratios; products of CSV means (A × F) against means of per-sequence products | Compute A·F and N(m) per sequence from counts; report both estimators; their gap diagnoses sequence-dependent conditioning |
| 12 | Comparing detection's accepted-shot F with correction's unconditional F | Report A, F_acc, A·F_acc and A·F_acc² for every method, paired on the same shots (H10) |
| 13 | const=k reads a subset of the rounds while decoders read all of them | Also decode with only the policy's rounds; say that the noise of unused rounds is still present |
| 14 | The matching factor k in p_RB = k p alone decides "logical beats physical" | Report k* = (1−f)_L/(0.66p); add k = 1 and an idle-matched physical baseline |
| 15 | Learned or lookup decoders scored on their training shots leak labels (skeptic toy, p = 5e-3, r = 10: 4.6% in sample against 40.6% out of sample) | Disjoint training and evaluation sequences; a label-permutation test; decoder APIs never see observables or error frames |
| 16 | Start-up and terminal-round transients bias f when m = 0 is in the fit | Fit m ≥ 2, and all m with a free m = 0 offset; attribute with the final-round X_ERROR switched off (H21) |
| 17 | Lone readout flips are not "accepted leakage" in LRB-D; they cause rejection (v02) | Do not attribute an O(p) unif=1 transient to readout flips; masking needs a data fault as well |
| 18 | The posterior threshold is only as good as the posterior's calibration | Reliability diagram per p: the wrong fraction in each \|LLR\| bin must equal 1/(1 + e^{\|LLR\|}); on hardware validate on held-out shots |
| 19 | The repo's "generalized SI1000" is not Gidney2021honeycomb's SI1000: sequential single-ancilla measurement layers, an extra DEPOLARIZE1(p) before X_ERROR(5p), no Idle(p/10) in gate layers, and no reset layer in the final round | Call it "generalized SI1000"; run the parallel-layer and gate-idle variants (H14(c), H16) |
| 20 | Kwiatkowski2025's sufficient condition fails above p ≈ 5e-4 to 6.5e-4 here | Do not use it to certify Markovianity; use local rates, curvature and the null control |
| 21 | Combes2017's conditions are assumptions under which LRB reduces to RB, not necessary conditions; the repo does refresh its ancillas | Write that the gap between LRB-D and Combes's LRB is postselection, not a failed "refresh" condition |
| 22 | Ceasura2022's oscillations come from a constructed, adversarial representation | Cite it as a possibility proof, not as the mechanism of matching or MLD memory |
| 23 | Multiplicity: 15 p × 45 policies × several decoders, windows and thresholds | Pre-register a few primary claims; simultaneous bands or Benjamini-Hochberg FDR from the joint bootstrap; confirm on fresh seeds |
| 24 | BLAS oversubscription: a site hook starts 64 OpenBLAS threads per process; threadpoolctl crashes forked workers | Export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 before launching Python (API.md) |
| 25 | Caching unique syndromes speeds MLD about 18× at p = 1e-4 but not at p ≥ 5e-3, where every shot is unique | Plan decode budgets per p; use the Rao-Blackwell estimator in simulation (numerics Idea 4) |
| 26 | SI1000's measurement-layer noise is superconducting-specific, and Garnet's published numbers are average infidelities, not Pauli strengths | Use platform profiles (H30); convert with p = r(d+1)/d before mapping hardware onto the p grid |
| 27 | Stim cannot represent leakage or atom loss | Keep hardware claims about correction conditional until a leakage model exists (H26) |
| 28 | "Correction mitigates non-Markovianity" may only mean "no postselection" | Claim a correction effect only if MLD is flatter than the null decoder (H19) |
| 29 | Process: v02 sent the user's email in a Crossref User-Agent header without being asked | Use a generic User-Agent; never put personal data in request headers |

## 4. Consolidated bibliography

Sections 4.1 to 4.8 contain only citations whose metadata a lit verifier confirmed or
corrected. The bracket after each entry names the verifier, and corrections are stated. Keys
that collided between notes were renamed; old keys appear as aliases. I re-checked every arXiv
ID in this list through the arXiv API on 2026-10-28 (title, author count, first and last author,
first-version date), and the journal DOIs added by verifiers through Crossref (Appendix A).
Nothing failed.

Key collisions resolved here: Gidney2021 (Stim, note 03; honeycomb memory, note 07);
Reichardt2024 (neutral atoms, notes 01 and 05; tesseract code, note 04); Kumar2026 (transition
matrices, note 02; error mitigation and detection, notes 04 and 08); Chen2025 (note 04 and 08)
renamed ChenXu2025 to avoid confusion with Chen2021, Chen2026 and ChenBaldwin2025.

### 4.1 Logical RB, RB theory and non-Markovian RB

- `Combes2017`. J. Combes, C. Granade, C. Ferrie, S. T. Flammia. "Logical Randomized Benchmarking." arXiv:1702.03688 (2017), preprint. [v01, v02, v04]
- `HarperFlammia2019` (aliases HF2019, Harper2018). R. Harper, S. T. Flammia. "Fault-Tolerant Logical Gates in the IBM Quantum Experience." Phys. Rev. Lett. 122, 080504 (2019). arXiv:1806.02359; DOI 10.1103/PhysRevLett.122.080504. [v01, v02, v06, v08]
- `Ceasura2022`. A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan. "Non-Exponential Behaviour in Logical Randomized Benchmarking." arXiv:2212.05488 (2022), preprint. [v01, v02, v05, v08]
- `Ziyad2025`. J. A. Ziyad, R. Blume-Kohout, K. Rudinger. "Emergent Non-Markovianity in Logical Qubit Dynamics." arXiv:2512.08893 (2025; v3 2026), preprint. [v01, v02]
- `Kabir2025`. A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang, B. Sorée. "Sdim: A Qudit Stabilizer Simulator." arXiv:2511.12777 (2025; v2 2026), preprint. In-group source. LRB-D is defined in Sec. VII-A and the non-Markovian warning is in Sec. VIII. [v01, v02, v06, v08]
- `ChenBaldwin2025`. Y.-H. Chen, C. H. Baldwin. "Randomized Benchmarking with Leakage Errors." Phys. Rev. Research 7, 043065 (2025). arXiv:2502.00154; DOI 10.1103/j8sz-6ws9. [v01, v02]
- `Lacroix2025`. N. Lacroix et al. (215 authors). "Scaling and logic in the color code on a superconducting quantum processor." Nature 645, 614-619 (2025); the Nature title spells "colour". arXiv:2412.14256; DOI 10.1038/s41586-025-09061-4. [v01, v06]
- `Mayer2024`. K. Mayer et al. (23 authors). "Benchmarking logical three-qubit quantum Fourier transform encoded in the Steane code on a trapped-ion quantum computer." arXiv:2404.08616 (2024), preprint. [v01]
- `Proctor2022`. T. Proctor, S. Seritan, K. Rudinger, E. Nielsen, R. Blume-Kohout, K. Young. "Scalable randomized benchmarking of quantum computers using mirror circuits." Phys. Rev. Lett. 129, 150502 (2022). arXiv:2112.09853; DOI 10.1103/PhysRevLett.129.150502. [v01]
- `Vigneau2025`. F. Vigneau et al. (16 authors; the arXiv page lists one name twice). "Quantum error detection in qubit-resonator star architecture." PRX Quantum 6, 040369 (2025). arXiv:2503.12869; DOI 10.1103/8m33-wn4g. [v01, corrected author count and venue]
- `Chen2021`. Z. Chen et al. (91 authors on arXiv). arXiv title "Exponential suppression of bit or phase flip errors with repetitive error correction," arXiv:2102.06132 (2021). Published with a Google Quantum AI byline as "Exponential suppression of bit or phase errors with cyclic error correction," Nature 595, 383-387 (2021), DOI 10.1038/s41586-021-03588-y. Cite each title with its own venue. [v01, corrected]
- `Lin2026`. W. Lin et al. (147 authors). "Surface code logical operations on a superconducting quantum processor." arXiv:2607.01473 (2026), preprint. [v01, v06]
- `Paetznick2024`. A. Paetznick et al. (32 authors). "Demonstration of logical qubits and repeated error correction with better-than-physical error rates." arXiv:2404.02280 (2024), preprint. Notes 01 and 06 said 33 authors; v06, v07 and v08 and the arXiv API give 32. Not the same paper as `Paetznick2026Nature`. [v01, v06, v07, v08]
- `Reichardt2024atoms` (alias Reichardt2024 in notes 01 and 05). B. W. Reichardt et al. (72 authors). "Fault-tolerant quantum computation with a neutral atom processor" (v1 title "Logical computation demonstrated with a neutral atom quantum processor"). arXiv:2411.11822 (2024; v3 2025), preprint. [v01, v05]
- `Bluvstein2023` (alias Bluvstein2024 in note 07). D. Bluvstein et al. (21 authors). "Logical quantum processor based on reconfigurable atom arrays." Nature 626, 58-65 (2024; online 2023). arXiv:2312.03982; DOI 10.1038/s41586-023-06927-3. [v01, v04, v07]
- `Smith2024`. S. C. Smith, B. J. Brown, S. D. Bartlett. "Mitigating errors in logical qubits." Commun. Phys. 7, 386 (2024). arXiv:2405.03766; DOI 10.1038/s42005-024-01883-4. [v01, v03, v04, v08]
- `Gottesman2016`. D. Gottesman. "Quantum fault tolerance in small experiments." arXiv:1610.03507 (2016), preprint. [v01, v08]
- `Reinhold2020`. P. Reinhold, S. Rosenblum, W.-L. Ma, L. Frunzio, L. Jiang, R. J. Schoelkopf. "Error-corrected gates on an encoded qubit." Nat. Phys. 16, 822-826 (2020). arXiv:1907.12327; DOI 10.1038/s41567-020-0931-8. [v01]
- `Levine2024` (alias Levine2023 in note 04). H. Levine et al. (113 authors). "Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons." Phys. Rev. X 14, 011051 (2024). arXiv:2307.08737; DOI 10.1103/PhysRevX.14.011051. [v01, v04, v05]
- `Liu2026`. B.-J. Liu, Y.-Y. Wang, Y.-X. Wang, M. Badbaria, S. Puri, C. Wang. "Hardware-Efficient Erasure Qubits With Superconducting Transmon Qutrits." arXiv:2604.08672 (2026), preprint. [v01]
- `Morvan2021`. A. Morvan et al. (10 authors). "Qutrit randomized benchmarking." Phys. Rev. Lett. 126, 210504 (2021). arXiv:2008.09134; DOI 10.1103/PhysRevLett.126.210504. [v01]
- `RyanAnderson2021` (alias RA2021). C. Ryan-Anderson et al. (17 authors). "Realization of real-time fault-tolerant quantum error correction." Phys. Rev. X 11, 041058 (2021). arXiv:2107.07505; DOI 10.1103/PhysRevX.11.041058. [v01, venue added; v03, v06, v07]
- `RyanAnderson2022`. C. Ryan-Anderson et al. (37 authors). "Implementing Fault-tolerant Entangling Gates on the Five-qubit Code and the Color Code." arXiv:2208.01863 (2022), preprint. Contains physical RB only. [v01]
- `Postler2022`. L. Postler et al. (13 authors). "Demonstration of fault-tolerant universal quantum gate operations." Nature 605, 675-680 (2022). arXiv:2111.12654; DOI 10.1038/s41586-022-04721-1. [v01]
- `Wallman2018`. J. J. Wallman. "Randomized benchmarking with gate-dependent noise." Quantum 2, 47 (2018). arXiv:1703.09835; DOI 10.22331/q-2018-01-29-47. [v02]
- `Proctor2017`. T. Proctor, K. Rudinger, K. Young, M. Sarovar, R. Blume-Kohout. "What randomized benchmarking actually measures." Phys. Rev. Lett. 119, 130502 (2017). arXiv:1702.01853; DOI 10.1103/PhysRevLett.119.130502. [v02]
- `Helsen2022`. J. Helsen, I. Roth, E. Onorati, A. H. Werner, J. Eisert. "A general framework for randomized benchmarking" (published as "General Framework for Randomized Benchmarking"). PRX Quantum 3, 020357 (2022). arXiv:2010.07974; DOI 10.1103/PRXQuantum.3.020357. [v02]
- `Epstein2014`. J. M. Epstein, A. W. Cross, E. Magesan, J. M. Gambetta. "Investigating the limits of randomized benchmarking protocols." Phys. Rev. A 89, 062321 (2014). arXiv:1308.2928; DOI 10.1103/PhysRevA.89.062321. [v02]
- `Ball2016`. H. Ball, T. M. Stace, S. T. Flammia, M. J. Biercuk. arXiv title "The effect of noise correlations on randomized benchmarking"; Phys. Rev. A title "Effect of noise correlations on randomized benchmarking." Phys. Rev. A 93, 022303 (2016). arXiv:1504.05307; DOI 10.1103/PhysRevA.93.022303. [v02, title note]
- `Mavadia2018`. S. Mavadia, C. L. Edmunds, C. Hempel, H. Ball, F. Roy, T. M. Stace, M. J. Biercuk. "Experimental quantum verification in the presence of temporally correlated noise." npj Quantum Inf. 4, 7 (2018). arXiv:1706.03787; DOI 10.1038/s41534-017-0052-0. [v02]
- `FongMerkel2017`. B. H. Fong, S. T. Merkel. "Randomized Benchmarking, Correlated Noise, and Ising Models." arXiv:1703.09747 (2017), preprint. [v02]
- `Fogarty2015`. M. A. Fogarty et al. (7 authors). "Non-exponential Fidelity Decay in Randomized Benchmarking with Low-Frequency Noise" (published as "Nonexponential fidelity decay ..."). Phys. Rev. A 92, 022326 (2015). arXiv:1502.05119; DOI 10.1103/PhysRevA.92.022326. [v02]
- `Srivastava2026`. V. Srivastava, A. K. Roy, S. Mahanti, J. Kaur, S. Karuvade, A. Gilchrist. "Blind-spots of Randomized Benchmarking Under Temporal Correlations" (published as "Blind spots of randomized benchmarking under temporal correlations"). Phys. Rev. Research 8, 023258 (2026). arXiv:2510.13051; DOI 10.1103/r1z5-ycqc. [v02]
- `Pollock2018`. F. A. Pollock, C. Rodríguez-Rosario, T. Frauenheim, M. Paternostro, K. Modi. "Non-Markovian quantum processes: complete framework and efficient characterisation." Phys. Rev. A 97, 012127 (2018). arXiv:1512.00589; DOI 10.1103/PhysRevA.97.012127. [v02]
- `FigueroaRomero2021`. P. Figueroa-Romero, K. Modi, R. J. Harris, T. M. Stace, M.-H. Hsieh. "Randomized benchmarking for non-Markovian noise." PRX Quantum 2, 040351 (2021). arXiv:2107.05403; DOI 10.1103/PRXQuantum.2.040351. Five authors in the published version; the arXiv listing shows four. [v02, corrected; note 02's removal of Harris was wrong]
- `FigueroaRomero2022`. P. Figueroa-Romero, K. Modi, M.-H. Hsieh. "Towards a general framework of Randomized Benchmarking incorporating non-Markovian Noise." Quantum 6, 868 (2022). arXiv:2202.11338; DOI 10.22331/q-2022-12-01-868. [v02]
- `FigueroaRomero2024`. P. Figueroa-Romero, M. Papič, A. Auer, M.-H. Hsieh, K. Modi, I. de Vega. "Operational Markovianization in Randomized Benchmarking." Quantum Sci. Technol. 9, 035020 (2024). arXiv:2305.04704; DOI 10.1088/2058-9565/ad3f44. [v02]
- `WoodGambetta2018`. C. J. Wood, J. M. Gambetta. "Quantification and Characterization of Leakage Errors." Phys. Rev. A 97, 032306 (2018). arXiv:1704.03081; DOI 10.1103/PhysRevA.97.032306. [v02, v05]
- `Wallman2016`. J. J. Wallman, M. Barnhill, J. Emerson. arXiv title "Characterization of Leakage Errors via Randomized Benchmarking"; New J. Phys. title "Robust characterization of leakage errors." New J. Phys. 18, 043021 (2016). arXiv:1412.4126; DOI 10.1088/1367-2630/18/4/043021. [v02, v05]
- `Kwiatkowski2025`. A. Kwiatkowski, A. J. Friedman, S. Geller, J. A. Ziyad, S. Glancy, E. Knill. "Constructing an approximate logical Markovian model of consecutive QEC cycles of a stabilizer code." arXiv:2509.16887 (2025), preprint. [v02]
- `KumarFoxmanDing2026` (alias Kumar2026 in note 02). R. S. Kumar, B. Foxman, Y. Ding. "Modeling Quantum Error Detection with Transition Matrices." arXiv:2609.12073 (2026), extended version of an IEEE QCE 2026 paper. [v02]
- `Endo2025`. S. Endo, H. Hakoshima, T. Shitara. "Non-Markovianity in Quantum Information Processing: Interplay with Quantum Error Mitigation." arXiv:2510.20224 (2025), preprint. [v02]
- `Girling2025`. M. Girling, B. Criger, C. Cirstoiu. "Characterization of syndrome-dependent logical noise in detector regions." arXiv:2508.08188 (2025), preprint. [v02]
- `Chasseur2015`. T. Chasseur, F. K. Wilhelm. "A complete randomized benchmarking protocol accounting for leakage errors" (the Phys. Rev. A title drops the leading "A"). Phys. Rev. A 92, 042333 (2015). arXiv:1505.00580; DOI 10.1103/PhysRevA.92.042333. [v05]

### 4.2 Decoders and decoding software

- `Gidney2021stim` (alias Gidney2021 in note 03). C. Gidney. "Stim: a fast stabilizer circuit simulator." Quantum 5, 497 (2021). arXiv:2103.02202; DOI 10.22331/q-2021-07-06-497. [v03]
- `HiggottGidney2025`. O. Higgott, C. Gidney. "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." Quantum 9, 1600 (2025). arXiv:2303.15933; DOI 10.22331/q-2025-01-20-1600. [v03]
- `Fowler2013`. A. G. Fowler. "Optimal complexity correction of correlated errors in the surface code." arXiv:1310.0863 (2013), preprint. [v03]
- `Higgott2023`. O. Higgott, T. C. Bohdanowicz, A. Kubica, S. T. Flammia, E. T. Campbell. "Improved decoding of circuit noise and fragile boundaries of tailored surface codes." Phys. Rev. X 13, 031007 (2023). arXiv:2203.04948; DOI 10.1103/PhysRevX.13.031007. [v03]
- `DelfosseNickerson2021`. N. Delfosse, N. H. Nickerson. "Almost-linear time decoding algorithm for topological codes." Quantum 5, 595 (2021). arXiv:1709.06218; DOI 10.22331/q-2021-12-02-595. [v03]
- `PanteleevKalachev2021`. P. Panteleev, G. Kalachev. "Degenerate Quantum LDPC Codes With Good Finite Length Performance." Quantum 5, 585 (2021). arXiv:1904.02703; DOI 10.22331/q-2021-11-22-585. [v03]
- `Roffe2020`. J. Roffe, D. R. White, S. Burton, E. T. Campbell. "Decoding across the quantum low-density parity-check code landscape" (arXiv: "Decoding Across the Quantum LDPC Code Landscape"). Phys. Rev. Research 2, 043423 (2020). arXiv:2005.07016; DOI 10.1103/PhysRevResearch.2.043423. [v03]
- `Hillmann2025`. T. Hillmann, L. Berent, A. O. Quintavalle, J. Eisert, R. Wille, J. Roffe. "Localized statistics decoding for quantum low-density parity-check codes." Nat. Commun. 16, 8214 (2025). arXiv:2406.18655; DOI 10.1038/s41467-025-63214-7. [v03]
- `IyerPoulin2015` (alias IyerPoulin2013). P. Iyer, D. Poulin. "Hardness of decoding quantum stabilizer codes." IEEE Trans. Inf. Theory 61(9), 5209-5223 (2015). arXiv:1310.3235; DOI 10.1109/TIT.2015.2422294. [v03, venue added]
- `BravyiSucharaVargo2014`. S. Bravyi, M. Suchara, A. Vargo. "Efficient algorithms for maximum likelihood decoding in the surface code." Phys. Rev. A 90, 032326 (2014). arXiv:1405.4883; DOI 10.1103/PhysRevA.90.032326. [v03]
- `Piveteau2024`. C. Piveteau, C. T. Chubb, J. M. Renes. "Tensor Network Decoding Beyond 2D." PRX Quantum 5, 040303 (2024). arXiv:2310.10722; DOI 10.1103/PRXQuantum.5.040303. [v03]
- `Heim2016`. B. Heim, K. M. Svore, M. B. Hastings. "Optimal circuit-level decoding for surface codes." arXiv:1609.06373 (2016), preprint. [v03]
- `Cain2024`. M. Cain, C. Zhao, H. Zhou, N. Meister, J. P. Bonilla Ataides, A. Jaffe, D. Bluvstein, M. D. Lukin. "Correlated decoding of logical algorithms with transversal gates." Phys. Rev. Lett. 133, 240602 (2024). arXiv:2403.03272; DOI 10.1103/PhysRevLett.133.240602. [v03, venue added]
- `SerraPeralta2026` (alias SST2025 in note 06). M. Serra-Peralta, M. H. Shaw, B. M. Terhal. "Decoding across transversal Clifford gates in the surface code." PRX Quantum 7, 010335 (2026). arXiv:2505.13599; DOI 10.1103/sk5y-25b1. [v03, v06]
- `Beni2025`. L. Aghababaie Beni, O. Higgott, N. Shutty. "Tesseract: A search-based decoder for quantum error correction." arXiv:2503.10988 (2025), preprint; code at github.com/quantumlib/tesseract-decoder. [v03]
- `Das2022` (alias Das2021). P. Das, A. Locharla, C. Jones. arXiv title "LILLIPUT: A Lightweight Low-Latency Lookup-Table Based Decoder for Near-term Quantum Error Correction," arXiv:2108.06569 (2021). Published in Proc. ASPLOS '22, pp. 541-553 (2022), DOI 10.1145/3503222.3507707; the proceedings title drops "Based". [v03, venue added]
- `Bausch2024`. J. Bausch et al. (18 authors). arXiv title "Learning to Decode the Surface Code with a Recurrent, Transformer-Based Neural Network"; Nature title "Learning high-accuracy error decoding for quantum processors." Nature 635, 834-840 (2024). arXiv:2310.05900; DOI 10.1038/s41586-024-08148-8. [v03]
- `Skoric2023`. L. Skoric, D. E. Browne, K. M. Barnes, N. I. Gillespie, E. T. Campbell. "Parallel window decoding enables scalable fault tolerant quantum computation." Nat. Commun. 14, 7040 (2023). arXiv:2209.08552; DOI 10.1038/s41467-023-42482-1. [v03]
- `Spitz2018`. S. T. Spitz, B. Tarasinski, C. W. J. Beenakker, T. E. O'Brien. arXiv title "Adaptive weight estimator for quantum error correction," arXiv:1712.02360 (2017). Journal title "Adaptive Weight Estimator for Quantum Error Correction in a Time-Dependent Environment," Adv. Quantum Technol. 1(1), 1800012 (2018), DOI 10.1002/qute.201800012. [v03, title corrected]
- `StimDocs`. Stim developers. "The Detector Error Model File Format (.dem)" and the `stim.Circuit.detector_error_model` docstring, stim 1.15.0, github.com/quantumlib/Stim. [v03]
- `PyMatchingDocs`. PyMatching v2.3.0 (13 Aug 2025) and v2.3.1 (25 Sep 2025) release notes; `Matching.from_detector_error_model` and `add_edge` docstrings. [v03]
- `ldpcDocs`. ldpc (LDPCv2) README and the installed ldpc 2.4.1 docstrings; `SinterLsdDecoder` and `SinterBeliefFindDecoder` live in `ldpc.sinter_decoders`. [v03]
- `Higgott2022`. O. Higgott. "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching." ACM Trans. Quantum Comput. 3(3), 1-16 (2022). arXiv:2105.13082; DOI 10.1145/3505637. [v03]
- `Chubb2021`. C. T. Chubb. "General tensor network decoding of 2D Pauli codes." arXiv:2101.04125 (2021), preprint. [v03]
- `Huang2020`. S. Huang, M. Newman, K. R. Brown. "Fault-tolerant weighted union-find decoding on the toric code." Phys. Rev. A 102, 012419 (2020). arXiv:2004.04693; DOI 10.1103/PhysRevA.102.012419. [v03]
- `TomitaSvore2014`. Y. Tomita, K. M. Svore. "Low-distance surface codes under realistic quantum noise." Phys. Rev. A 90, 062320 (2014). arXiv:1404.3747; DOI 10.1103/PhysRevA.90.062320. [v03]
- `Meister2024`. N. Meister, C. A. Pattison, J. Preskill. "Efficient soft-output decoders for the surface code." arXiv:2405.07433 (2024), preprint. [v03, v04]

### 4.3 Soft output, postselection and decision theory

- `Bombin2022`. H. Bombín, M. Pant, S. Roberts, K. I. Seetharam. "Fault-tolerant Post-Selection for Low Overhead Magic State Preparation" (journal: "Fault-Tolerant Postselection for Low-Overhead Magic State Preparation"). PRX Quantum 5, 010302 (2024). arXiv:2212.00813; DOI 10.1103/PRXQuantum.5.010302. [v04]
- `Gidney2023yoked`. C. Gidney, M. Newman, P. Brooks, C. Jones. "Yoked surface codes." Nat. Commun. 16, 4498 (2025). arXiv:2312.04522; DOI 10.1038/s41467-025-59714-1. [v04, venue added]
- `Gidney2024cult`. C. Gidney, N. Shutty, C. Jones. "Magic state cultivation: growing T states as cheap as CNOT gates." arXiv:2409.17595 (2024), preprint. [v04]
- `English2024`. L. H. English, D. J. Williamson, S. D. Bartlett. "Thresholds for post-selected quantum error correction from statistical mechanics." Phys. Rev. Lett. 135, 120603 (2025). arXiv:2410.07598; DOI 10.1103/nh49-52y2. Its Theorem 1 (equal-cardinality version) is wrong as stated per Seip2026 Remark G.2. [v04, v08]
- `Seip2026`. D. Seip, L. Colmenarez, M. Schmitt, M. Müller. "Machine Learning Optimal Quantum Error Correction Thresholds." arXiv:2606.22194 (2026), preprint. [v04]
- `ChenXu2025` (alias Chen2025 in notes 04 and 08). H. Chen, D. Xu, G. M. Sommers, D. A. Huse, J. D. Thompson, S. Gopalakrishnan. "Scalable accuracy gains from postselection in quantum error correcting codes." Phys. Rev. A 114, 032455 (2026). arXiv:2510.05222; DOI 10.1103/mfyk-5j3x. [v04, venue added; v08]
- `Lee2025`. S.-H. Lee, L. H. English, S. D. Bartlett. "Efficient Post-Selection for General Quantum LDPC Codes." npj Quantum Inf. 12, 96 (2026). arXiv:2510.05795; DOI 10.1038/s41534-026-01242-x. [v04]
- `Prabhu2021` (alias PR2024). P. Prabhu, B. W. Reichardt. "Distance-four quantum codes with combined postselection and error correction." Phys. Rev. A 110, 012419 (2024). arXiv:2112.03785; DOI 10.1103/PhysRevA.110.012419. [v04, v06, v08]
- `ReichardtTess2024` (alias Reichardt2024 in note 04). B. W. Reichardt et al. (18 authors). "Demonstration of quantum computation and error correction with a tesseract code." arXiv:2409.04628 (2024), preprint. [v04, v08]
- `Pattison2021`. C. A. Pattison, M. E. Beverland, M. P. da Silva, N. Delfosse. "Improved quantum error correction using soft information." arXiv:2107.13589 (2021), preprint. [v04]
- `Dinca2025`. M. Dincă, T. Chan, S. C. Benjamin. "Error mitigation for logical circuits using decoder confidence." arXiv:2512.15689 (2025; v4 2026), preprint. [v04]
- `Zhou2025`. Z. Zhou, S. Pexton, A. Kubica, Y. Ding. "Error Mitigation of Fault-Tolerant Quantum Circuits with Soft Information." arXiv:2512.09863 (2025), preprint. [v04]
- `Hanisch2024`. M. D. Hanisch, B. Hetényi, J. R. Wootton. "Soft information decoding with superconducting qubits." APS Open Sci. 1, 000019 (2026). arXiv:2411.16228; DOI 10.1103/y9fh-4x6n. [v04]
- `Xin2026` (alias Xin2025 in note 04). Y. Xin, S. L. M. van der Meer, M. Serra-Peralta, T. H. F. Vroomans, M. Finkel, H. M. Veen, M. W. Beekman, L. DiCarlo. "Improved error correction with leakage reduction units built into qubit measurement in a superconducting quantum processor." PRX Quantum 7, 033062 (2026). arXiv:2511.17460; DOI 10.1103/y66v-988s. [v04, v05]
- `Rosenfeld2025`. E. Rosenfeld et al. (295 authors). "Magic state cultivation on a superconducting quantum processor." arXiv:2512.13908 (2025), preprint. [v04]
- `CombesFerrie2015`. J. Combes, C. Ferrie. "Cost of postselection in decision theory." Phys. Rev. A 92, 022117 (2015). arXiv:1504.03740; DOI 10.1103/PhysRevA.92.022117. [v04]
- `Chow1970`. C. K. Chow (Crossref lists "C. Chow"). "On optimum recognition error and reject tradeoff." IEEE Trans. Inf. Theory 16(1), 41-46 (1970). DOI 10.1109/TIT.1970.1054406. [v04]
- `AGP2007`. P. Aliferis, D. Gottesman, J. Preskill. "Accuracy threshold for postselected quantum computation." Quantum Inf. Comput. 8, 181-244 (2008). arXiv:quant-ph/0703264. [v04, v08]
- `Ali2024`. H. Ali et al. (10 authors). "Reducing the error rate of a superconducting logical qubit using analog readout information." Phys. Rev. Applied 22, 044031 (2024). arXiv:2403.00706; DOI 10.1103/PhysRevApplied.22.044031. [v04]
- `KumarTsunoda2026` (alias Kumar2026 in notes 04 and 08). R. S. Kumar, T. Tsunoda, S. H. Xue, D. Li, R. J. Schoelkopf, Y. Ding. "Co-Designing Error Mitigation and Error Detection for Logical Qubits." arXiv:2604.19871 (2026), preprint. [v04, v08]
- `Shtanko2026`. O. Shtanko, S. Roy, S. Kumar, Z. Minev. "Measure Now, Mitigate Later: Virtual Error Cancellation for Logical Quantum Circuits." arXiv:2610.12400 (2026), preprint. [v04]
- `Tahaab2026`. M. Tahaab, T. Chan, J. Kirkman, S. C. Benjamin. "Stream Decoding with Confidence Scores at Room and Cryogenic Temperatures." arXiv:2608.10576 (2026), preprint. [v04]
- `Forney1968`. G. D. Forney Jr. (Crossref lists "G. Forney"). "Exponential error bounds for erasure, list, and decision feedback schemes." IEEE Trans. Inf. Theory 14(2), 206-220 (1968). DOI 10.1109/TIT.1968.1054129. [v04]
- `Merhav2006`. N. Merhav, M. Feder. "Universal decoding with an erasure option." arXiv:cs/0604069 (2006), preprint. [v04]

### 4.4 Erasure, leakage and flag qubits

- `Wu2022`. Y. Wu, S. Kolkowitz, S. Puri, J. D. Thompson. "Erasure conversion for fault-tolerant quantum computing in alkaline earth Rydberg atom arrays." Nat. Commun. 13, 4657 (2022). arXiv:2201.03540; DOI 10.1038/s41467-022-32094-6. [v05]
- `Kubica2023`. A. Kubica, A. Haim, Y. Vaknin, H. Levine, F. Brandão, A. Retzker. "Erasure qubits: Overcoming the T1 limit in superconducting circuits." Phys. Rev. X 13, 041022 (2023). arXiv:2208.05461; DOI 10.1103/PhysRevX.13.041022. Six authors in PRX; the arXiv v1 listing has five. [v05, corrected]
- `Gu2025`. S. Gu, Y. Vaknin, A. Retzker, A. Kubica. "Optimizing quantum error correction protocols with erasure qubits" (journal: "Optimizing Quantum Error-Correction Protocols with Erasure Qubits"). PRX Quantum 6, 040354 (2025). arXiv:2408.00829; DOI 10.1103/985g-58gd. [v05]
- `Chou2024`. K. S. Chou et al. (34 authors). arXiv title "Demonstrating a superconducting dual-rail cavity qubit with erasure-detected logical measurements"; the journal title drops "Demonstrating". Nat. Phys. 20, 1454-1460 (2024). arXiv:2307.03169; DOI 10.1038/s41567-024-02539-4. [v05, author count corrected from 35]
- `Zhang2026`. B. Zhang, G. Liu, G. Bornet, S. P. Horvath, P. Peng, S. Ma, S. Huang, S. Puri, J. D. Thompson. "Logical qubits with erasure conversion using metastable neutral atoms" (v1: "Leveraging erasure errors in logical qubits with metastable 171Yb atoms"). Nat. Phys. 22, 910-916 (2026). arXiv:2506.13724; DOI 10.1038/s41567-026-03309-0. Use the published and v2 numbers (CZ erasure fraction 38(6)%), not v1's 31%. [v05, corrected]
- `Chow2024`. M. N. H. Chow, V. Buchemmavari, S. Omanakuttan, B. J. Little, S. Pandey, I. H. Deutsch, Y.-Y. Jau. "Circuit-based leakage-to-erasure conversion in a neutral atom quantum processor." PRX Quantum 5, 040343 (2024). arXiv:2405.10434; DOI 10.1103/PRXQuantum.5.040343. [v05]
- `Quinn2026`. A. Quinn, G. J. Gregory, I. D. Moore, S. Brudney, J. Metzner, E. R. Ritchie, J. O'Reilly, D. J. Wineland, D. T. C. Allcock. "High-fidelity entanglement of metastable trapped-ion qubits with integrated erasure conversion." Phys. Rev. A 113, L040601 (2026). arXiv:2411.12727; DOI 10.1103/p3cy-8yjk. [v05]
- `Ransford2025`. A. Ransford et al. (186 authors). "Helios: A 98-qubit trapped-ion quantum computer." arXiv:2511.05465 (2025), preprint. [v05]
- `McEwen2021`. M. McEwen et al. (51 authors). "Removing leakage-induced correlated errors in superconducting quantum error correction." Nat. Commun. 12, 1761 (2021). arXiv:2102.06131; DOI 10.1038/s41467-021-21982-y. [v05, author count corrected from 50]
- `Miao2023`. K. C. Miao et al. (117 authors on arXiv, 118 in the journal). arXiv title "Overcoming leakage in scalable quantum error correction"; journal title "Overcoming leakage in quantum error correction." Nat. Phys. 19, 1780-1786 (2023). arXiv:2211.04728; DOI 10.1038/s41567-023-02226-w. [v05]
- `Battistel2021`. F. Battistel, B. M. Varbanov, B. M. Terhal. "A hardware-efficient leakage-reduction scheme for quantum error correction with superconducting transmon qubits" (the journal title drops the leading "A"). PRX Quantum 2, 030314 (2021). arXiv:2102.08336; DOI 10.1103/PRXQuantum.2.030314. [v05]
- `Varbanov2020`. B. M. Varbanov, F. Battistel, B. M. Tarasinski, V. P. Ostroukh, T. E. O'Brien, L. DiCarlo, B. M. Terhal. "Leakage detection for a transmon-based surface code." npj Quantum Inf. 6, 102 (2020). arXiv:2002.07119; DOI 10.1038/s41534-020-00330-w. [v04, v05]
- `ChaoReichardt2018`. R. Chao, B. W. Reichardt. "Quantum error correction with only two extra qubits." Phys. Rev. Lett. 121, 050502 (2018). arXiv:1705.02329; DOI 10.1103/PhysRevLett.121.050502. [v05]
- `Chamberland2020`. C. Chamberland, G. Zhu, T. J. Yoder, J. B. Hertzberg, A. W. Cross. "Topological and subsystem codes on low-degree graphs with flag qubits." Phys. Rev. X 10, 011022 (2020). arXiv:1907.09528; DOI 10.1103/PhysRevX.10.011022. [v05]
- `ChamberlandBeverland2018`. C. Chamberland, M. E. Beverland. "Flag fault-tolerant error correction with arbitrary distance codes." Quantum 2, 53 (2018). arXiv:1708.02246; DOI 10.22331/q-2018-02-08-53. [v05]
- `Grassl1997`. M. Grassl, T. Beth, T. Pellizzari. "Codes for the quantum erasure channel." Phys. Rev. A 56, 33-38 (1997). arXiv:quant-ph/9610042; DOI 10.1103/PhysRevA.56.33. [v05]
- `Ma2023`. S. Ma, G. Liu, P. Peng, B. Zhang, S. Jandura, J. Claes, A. P. Burgers, G. Pupillo, S. Puri, J. D. Thompson. "High-fidelity gates and mid-circuit erasure conversion in an atomic qubit" (arXiv: "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit"). Nature 622, 279-284 (2023). arXiv:2305.05493; DOI 10.1038/s41586-023-06438-1. [v05]
- `Mehta2025`. N. Mehta et al. (72 named authors plus a team entry). "Bias-preserving and error-detectable entangling operations in a superconducting dual-rail system." arXiv:2503.10935 (2025), preprint. [v05]
- `Hung2026`. J. S.-C. Hung et al. (17 authors). "Fast, high-fidelity erasure detection of dual-rail qubits with symmetrically coupled readout." arXiv:2604.16292 (2026), preprint. [v05]
- `Marques2023`. J. F. Marques et al. (12 authors). "All-microwave leakage reduction units for quantum error correction with superconducting transmon qubits." Phys. Rev. Lett. 130, 250602 (2023). arXiv:2302.09876; DOI 10.1103/PhysRevLett.130.250602. [v05]
- `Stricker2020`. R. Stricker et al. (10 authors). arXiv title "Deterministic correction of qubit loss"; journal title "Experimental deterministic correction of qubit loss." Nature 585, 207-210 (2020). arXiv:2002.09532; DOI 10.1038/s41586-020-2667-0. Its code has a weight-1 X logical, so it is not a d − 1 erasure example. [v05]

### 4.5 Small surface codes and fold-transversal gates

- `Moussa2016`. J. E. Moussa. "Transversal Clifford gates on folded surface codes." Phys. Rev. A 94, 042316 (2016). arXiv:1603.02286; DOI 10.1103/PhysRevA.94.042316. [v06]
- `BB2024`. N. P. Breuckmann, S. Burton. "Fold-Transversal Clifford Gates for Quantum Codes." Quantum 8, 1372 (2024). arXiv:2202.06647; DOI 10.22331/q-2024-06-13-1372. [v06]
- `VK2022`. M. Vasmer, A. Kubica. "Morphing quantum codes." PRX Quantum 3, 030319 (2022). arXiv:2112.01446; DOI 10.1103/PRXQuantum.3.030319. [v06]
- `Chen2026`. Z.-H. Chen, M.-C. Chen, C.-Y. Lu, J.-W. Pan. arXiv title "Transversal Logical Clifford gates on rotated surface codes with reconfigurable neutral atom arrays"; PRL title "Transversal Logical Clifford Gates on the Rotated Surface Code with Reconfigurable Neutral Atom Arrays." Phys. Rev. Lett. 136, 130601 (2026). arXiv:2412.01391; DOI 10.1103/m7tq-9v3g. A proposal with simulations. [v06]
- `KZ2024`. R. Kobayashi, G. Zhu. "Cross-cap defects and fault-tolerant logical gates in the surface code and the honeycomb Floquet code." PRX Quantum 5, 020360 (2024). arXiv:2310.06917; DOI 10.1103/PRXQuantum.5.020360. [v06]
- `Andersen2020`. C. K. Andersen, A. Remm, S. Lazar, S. Krinner, N. Lacroix, G. J. Norris, M. Gabureac, C. Eichler, A. Wallraff. "Repeated quantum error detection in a surface code." Nat. Phys. 16, 875-880 (2020). arXiv:1912.09410; DOI 10.1038/s41567-020-0920-y. [v06]
- `Marques2022`. J. F. Marques et al. (13 authors). "Logical-qubit operations in an error-detecting surface code." Nat. Phys. 18, 80-86 (2022). arXiv:2102.13071; DOI 10.1038/s41567-021-01423-9. [v06]
- `Bultink2020`. C. C. Bultink et al. (12 authors). "Protecting quantum entanglement from leakage and qubit errors via repetitive parity measurements." Sci. Adv. 6, eaay3050 (2020). arXiv:1905.12731; DOI 10.1126/sciadv.aay3050. [v06]
- `Linke2017`. N. M. Linke, M. Gutierrez, K. A. Landsman, C. Figgatt, S. Debnath, K. R. Brown, C. Monroe. "Fault-tolerant quantum error detection." Sci. Adv. 3, e1701074 (2017). arXiv:1611.06946; DOI 10.1126/sciadv.1701074. [v06]
- `Takita2017`. M. Takita, A. W. Cross, A. D. Córcoles, J. M. Chow, J. M. Gambetta. "Experimental demonstration of fault-tolerant state preparation with superconducting qubits." Phys. Rev. Lett. 119, 180501 (2017). arXiv:1705.09259; DOI 10.1103/PhysRevLett.119.180501. [v06]
- `BDB2024`. S. Burton, E. Durso-Sabina, N. C. Brown. "Genons, Double Covers and Fault-tolerant Clifford Gates." arXiv:2406.09951 (2024), preprint. [v06]
- `Krinner2022`. S. Krinner et al. (16 authors). "Realizing repeated quantum error correction in a distance-three surface code." Nature 605, 669-674 (2022). arXiv:2112.03708; DOI 10.1038/s41586-022-04566-8. [v06]
- `KYP2015`. A. Kubica, B. Yoshida, F. Pastawski. "Unfolding the color code." New J. Phys. 17, 083026 (2015). arXiv:1503.02065; DOI 10.1088/1367-2630/17/8/083026. [v06, journal named]
- `ECZ512`. Error Correction Zoo, eds. V. V. Albert and P. Faist. "[[5,1,2]] rotated surface code." https://errorcorrectionzoo.org/c/stab_5_1_2, accessed 2026-10-28, latest edit 2026-09-24. Its "smallest code" sentence rests on a private communication. [v06]
- `MBG2023`. M. McEwen, D. Bacon, C. Gidney. "Relaxing Hardware Requirements for Surface Code Circuits using Time-dynamics." Quantum 7, 1172 (2023). arXiv:2302.02192; DOI 10.22331/q-2023-11-07-1172. [v06]
- `Geher2024`. G. P. Gehér, C. McLauchlan, E. T. Campbell, A. E. Moylett, O. Crawford. "Error-corrected Hadamard gate simulated at the circuit level." Quantum 8, 1394 (2024). arXiv:2312.11605; DOI 10.22331/q-2024-07-02-1394. [v06]
- `Sahay2026`. K. Sahay, P.-K. Tsai, K. Chang, Q. Su, T. B. Smith, S. Singh, S. Puri. "Fold-transversal surface code cultivation." PRX Quantum 7, 033006 (2026). arXiv:2509.05212; DOI 10.1103/gpvl-lg4c. [v06]
- `IQM2025blog`. F. R. F. Pereira (IQM). "How IQM is tackling quantum error correction." Company blog, 14 Aug 2025, https://iqm.tech/blog/how-iqm-is-tackling-quantum-error-correction/. [v06]

### 4.6 Hardware, mid-circuit measurement and real-time decoding

- `Knill2005`. E. Knill. arXiv title "Quantum Computing with Very Noisy Devices" (2004); Nature title "Quantum computing with realistically noisy devices." Nature 434, 39-44 (2005). arXiv:quant-ph/0410199; DOI 10.1038/nature03350. [v07, v08]
- `Terhal2015`. B. M. Terhal. "Quantum Error Correction for Quantum Memories." Rev. Mod. Phys. 87, 307-346 (2015). arXiv:1302.3428; DOI 10.1103/RevModPhys.87.307. [v07]
- `Chamberland2018`. C. Chamberland, P. Iyer, D. Poulin. "Fault-Tolerant Quantum Computing in the Pauli or Clifford Frame with Slow Error Diagnostics." Quantum 2, 43 (2018). arXiv:1704.06662; DOI 10.22331/q-2018-01-04-43. [v07]
- `Battistel2023`. F. Battistel, C. Chamberland, K. Johar, R. W. J. Overwater, F. Sebastiano, L. Skoric, Y. Ueno, M. Usman. "Real-Time Decoding for Fault-Tolerant Quantum Computing: Progress, Challenges and Outlook." Nano Futures 7, 032003 (2023). arXiv:2303.00054; DOI 10.1088/2399-1984/aceba6. [v07]
- `Google2025` (alias Google2024 in note 08). R. Acharya et al., Google Quantum AI and Collaborators (249 authors on arXiv). "Quantum error correction below the surface code threshold." Nature 638, 920-926 (2025). arXiv:2408.13687; DOI 10.1038/s41586-024-08449-y. [v07, v08]
- `Caune2026`. L. Caune et al. (25 authors). "Demonstrating real-time and low-latency quantum error correction with superconducting qubits." Nat. Commun. 17, 7383 (2026). arXiv:2410.05202; DOI 10.1038/s41467-026-73331-6. [v07]
- `Barber2025`. B. Barber et al. (13 authors). arXiv title "A real-time, scalable, fast and highly resource efficient decoder for a quantum computer"; journal title "A real-time, scalable, fast and resource-efficient decoder for a quantum computer." Nat. Electron. 8, 84-91 (2025). arXiv:2309.05558; DOI 10.1038/s41928-024-01319-5. Its results use simulated syndromes, and the ASIC is a pre-tape-out design. [v07]
- `Yang2026`. X. Yang et al. (12 authors). "Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder." arXiv:2605.04892 (2026), preprint. [v07]
- `Corcoles2021`. A. D. Córcoles, M. Takita, K. Inoue, S. Lekuch, Z. K. Minev, J. M. Chow, J. M. Gambetta. "Exploiting dynamic quantum circuits in a quantum algorithm with superconducting qubits." Phys. Rev. Lett. 127, 100501 (2021). arXiv:2102.01682; DOI 10.1103/PhysRevLett.127.100501. [v07]
- `Govia2023`. L. C. G. Govia, P. Jurcevic, C. J. Wood, N. Kanazawa, S. T. Merkel, D. C. McKay. "A randomized benchmarking suite for mid-circuit measurements." New J. Phys. 25, 123016 (2023). arXiv:2207.04836; DOI 10.1088/1367-2630/ad0e19. [v07]
- `Hothem2025`. D. Hothem, J. Hines, C. Baldwin, D. Gresh, R. Blume-Kohout, T. Proctor. "Measuring error rates of mid-circuit measurements." Nat. Commun. 16, 5761 (2025). arXiv:2410.16706; DOI 10.1038/s41467-025-60923-x. [v07]
- `Gidney2021honeycomb` (alias Gidney2021 in note 07). C. Gidney, M. Newman, A. Fowler, M. Broughton. "A Fault-Tolerant Honeycomb Memory." Quantum 5, 605 (2021). arXiv:2108.10457; DOI 10.22331/q-2021-12-20-605. Defines SI1000 in Tables 1 and 2. [v07]
- `Geher2025`. G. P. Gehér, M. Jastrzebski, E. T. Campbell, O. Crawford. "To reset, or not to reset -- that is the question." npj Quantum Inf. 11, 39 (2025). arXiv:2408.00758; DOI 10.1038/s41534-025-00998-y. [v07]
- `Moses2023`. S. A. Moses et al. (96 authors). "A Race Track Trapped-Ion Quantum Processor" (PRX title "A Race-Track Trapped-Ion Quantum Processor"). Phys. Rev. X 13, 041052 (2023). arXiv:2305.03828; DOI 10.1103/PhysRevX.13.041052. [v07]
- `Muniz2025`. J. A. Muniz et al. (62 authors). "Repeated ancilla reuse for logical computation on a neutral atom quantum computer." Phys. Rev. X 15, 041040 (2025). arXiv:2506.09936; DOI 10.1103/v7ny-fg31. [v07]
- `Abdurakhimov2024`. L. Abdurakhimov et al. (114 authors). "Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer." arXiv:2408.12433 (2024), preprint. [v07, author count corrected from 113]
- `IQMdocs`. IQM client 36.0.0 documentation (models API page, "Qiskit on IQM" user guide) and iqm.pulse 15.0.0 `Reset_Conditional`, docs.iqm.tech, last updated 2026-10-06. [v07]
- `AWSBraket2025`. T. Chen, S. Smart, Z. Mohammad. "Experiment with dynamic circuits on IQM Garnet with Amazon Braket." AWS Quantum Technologies blog, 26 Jun 2025. [v07]
- `IQMHalocene2025`. IQM press release. "IQM launches Halocene, a new quantum computer product line for error correction." 13 Nov 2025, https://www.iqm.tech/?p=14348. [v07]

### 4.7 Detection versus correction and break-even

- `Knill2004a`. E. Knill. "Fault-Tolerant Postselected Quantum Computation: Schemes." arXiv:quant-ph/0402171 (2004), preprint. [v08]
- `Knill2004b`. E. Knill. "Fault-Tolerant Postselected Quantum Computation: Threshold Analysis." arXiv:quant-ph/0404104 (2004), preprint. [v08]
- `Reichardt2009`. B. W. Reichardt. "Error-Detection-Based Quantum Fault-Tolerance Threshold." Algorithmica 55(3), 517-556 (2009; online 2007). DOI 10.1007/s00453-007-9069-7. The related arXiv record is the PhD thesis "Error-detection-based quantum fault tolerance against discrete Pauli noise," arXiv:quant-ph/0612004 (2006), with a near-identical abstract. [v08]
- `Self2022`. C. N. Self, M. Benedetti, D. Amaro. "Protecting expressive circuits with a quantum error detection code." Nat. Phys. 20, 219-224 (2024). arXiv:2211.06703; DOI 10.1038/s41567-023-02282-2. [v08]
- `He2024`. Z. He, D. Amaro, R. Shaydulin, M. Pistoia. "Performance of quantum approximate optimization with quantum error detection." Commun. Phys. 8, 217 (2025). arXiv:2409.12104; DOI 10.1038/s42005-025-02136-8. [v08]
- `Jin2025`. Y. Jin, Z. He, T. Hao, S. Omanakuttan, D. Amaro, S. Tannu, R. Shaydulin, M. Pistoia. "Iceberg Beyond the Tip: Co-Compilation of a Quantum Error Detection Code and a Quantum Algorithm." arXiv:2504.21172 (2025), preprint. [v08]
- `Vuillot2017`. C. Vuillot. "Is error detection helpful on IBM 5Q chips ?" Quantum Inf. Comput. 18(11&12), 949-964 (2018). arXiv:1705.08957. [v08]
- `Willsch2018`. D. Willsch, M. Willsch, F. Jin, H. De Raedt, K. Michielsen. "Testing quantum fault tolerance on small systems." Phys. Rev. A 98, 052348 (2018). arXiv:1805.05227; DOI 10.1103/PhysRevA.98.052348. [v08]
- `Gupta2023`. R. S. Gupta et al. (13 authors). "Encoding a magic state with beyond break-even fidelity." Nature 625, 259-263 (2024). arXiv:2305.13581; DOI 10.1038/s41586-023-06846-3. [v08]
- `Paetznick2026Nature`. A. Paetznick, B. W. Reichardt, M. P. da Silva, ..., K. M. Svore (37 authors). "Improved quantum processor logical error rates via correction and detection." Nature 654, 349-355 (2026). DOI 10.1038/s41586-026-10628-y. Related to `Paetznick2024` and `ReichardtTess2024` but not the journal version of either: it cites arXiv:2404.02280 as a preprint and covers the 12-qubit carbon and 16-qubit tesseract codes; the Steane-code Bell result is probably arXiv-only. Take numbers from the arXiv versions. [v06, v08]

### 4.8 Verifier-confirmed side references, not used for claims

A verifier fetched these and confirmed their metadata, but no note relies on them for a claim.
Titles for the v02 group come from my arXiv API check.

- `Brock2025`. B. L. Brock et al. (8 authors). "Quantum Error Correction of Qudits Beyond Break-even." Nature 641, 612-618 (2025). arXiv:2409.15065; DOI 10.1038/s41586-025-08899-y. [v01]
- `Kam2025`. J. F. Kam et al. (5 authors). "Detrimental non-Markovian errors for surface code memory." Quantum Sci. Technol. 10, 035060 (2025). arXiv:2410.23779. v02 recommends it for Q2. [v02]
- `Pollock2018b`. F. A. Pollock, C. Rodríguez-Rosario, T. Frauenheim, M. Paternostro, K. Modi. "Operational Markov condition for quantum processes." Phys. Rev. Lett. 120, 040405 (2018). arXiv:1801.09811. [v02]
- `Brillant2025`. A. Brillant et al. (5 authors). "Randomized benchmarking with non-Markovian noise and realistic finite-time gates." arXiv:2501.06172 (2025). [v02]
- `GandhariGullans2025`. S. Gandhari, M. J. Gullans. "Quantum non-Markovian noise in randomized benchmarking of spin-boson models." arXiv:2502.14702 (2025). [v02]
- `Kobayashi2024`. F. Kobayashi et al. (6 authors). "Tensor-network decoders for process tensor descriptions of non-Markovian noise." arXiv:2412.13739 (2024). [v02]
- `Budini2018`. A. A. Budini. "Quantum non-Markovian processes break conditional past-future independence." Phys. Rev. Lett. 121, 240401 (2018). arXiv:1811.03448. [v02]
- `QiNg2021`. J. Qi, H. K. Ng. "Randomized benchmarking in the presence of time-correlated dephasing noise." Phys. Rev. A 103, 022607 (2021). arXiv:2010.11498. [v02]
- `Magesan2012`. E. Magesan, J. M. Gambetta, J. Emerson. "Characterizing Quantum Gates via Randomized Benchmarking." Phys. Rev. A 85, 042311 (2012). arXiv:1109.6887. [v02]
- `CarignanDugas2018`. A. Carignan-Dugas et al. (4 authors). "From randomized benchmarking experiments to gateset circuit fidelity: how to interpret randomized benchmarking." New J. Phys. 20, 092001 (2018). arXiv:1804.01122. [v02]
- `MilzModi2021`. S. Milz, K. Modi. "Quantum stochastic processes and quantum non-Markovian phenomena." PRX Quantum 2, 030201 (2021). arXiv:2012.01894. [v02]
- `Yang2022nm`. S.-X. Yang et al. (3 authors). "Machine Learning of Average Non-Markovianity from Randomized Benchmarking." arXiv:2207.01542 (2022). [v02]
- `Pryadko2020`. L. P. Pryadko. "On maximum-likelihood decoding with circuit-level errors." Quantum 4, 304 (2020). arXiv:1909.06732. [v03]
- `Cao2025`. H. Cao et al. (11 authors). "Exact Decoding of Repetition Code under Circuit Level Noise." arXiv:2501.03582 (2025). [v03]
- `Varbanov2025nn`. B. M. Varbanov, M. Serra-Peralta, D. Byfield, B. M. Terhal. "Neural network decoder for near-term surface-code experiments." Phys. Rev. Research 7, 013029 (2025). arXiv:2307.03280. [v03]
- `Baireuther2018`. P. Baireuther, T. E. O'Brien, B. Tarasinski, C. W. J. Beenakker. "Machine-learning-assisted correction of correlated qubit errors in a topological code." Quantum 2, 48 (2018). arXiv:1705.07855. [v03]
- `Senior2025`. A. W. Senior et al. (24 authors). "A scalable and real-time neural decoder for topological quantum codes." arXiv:2512.07737 (2025). [v03]
- `Wu2025mwpf`. Y. Wu et al. (5 authors). "Minimum-Weight Parity Factor Decoder for Quantum Error Correction." arXiv:2508.04969 (2025). [v03]
- `CaoYanDuPan2026`. H. Cao, Y. Yan, Y. Du, F. Pan. "Maximum Likelihood Decoding of Quantum Error Correction Codes." arXiv:2605.17230 (2026), review. [v03, v04]
- `Staples2026`. J. W. Staples, Y. Fu, J. D. Thompson. "Scalable Postselection of Quantum Resources." arXiv:2603.08697 (2026). [v04]
- `Mishima2026`. M. Mishima et al. (7 authors). "Adaptive Window Decoding based on Spatiotemporal Complementary Gap." arXiv:2605.14637 (2026). [v04]
- `Zhang2024sc`. J. Zhang et al. (9 authors). "Demonstrating a universal logical gate set in error-detecting surface codes on a superconducting quantum processor." arXiv:2405.09035 (2024). Needed to qualify "no logical H at d=2". [v06]
- `Rudinger2022`. K. Rudinger et al. (9 authors). "Characterizing mid-circuit measurements on a superconducting qubit using gate set tomography." Phys. Rev. Applied 17, 014014 (2022). arXiv:2103.03008. [v07]
- `Gaebler2021`. J. P. Gaebler et al. (8 authors). "Suppression of mid-circuit measurement crosstalk errors with micromotion." Phys. Rev. A 104, 062440 (2021). arXiv:2108.10932. Also cited by other-platforms. [v07]
- `Bluvstein2025`. D. Bluvstein et al. (23 authors). arXiv title "Architectural mechanisms of a universal fault-tolerant quantum computer"; Nature title "A fault-tolerant neutral-atom architecture for universal quantum computation." Nature 649, 39-46 (2026). arXiv:2506.20661; DOI 10.1038/s41586-025-09848-5. Also cited by other-platforms. [v07]
- `Maurer2025`. T. Maurer et al. (7 authors). "Real-time decoding of the gross code memory with FPGAs." arXiv:2510.21600 (2025). [v07]
- `Riesebos2017`. L. Riesebos et al. "Pauli Frames for Quantum Computer Architectures." Proc. DAC 2017, DOI 10.1145/3061639.3062300 (Crossref metadata only). [v07]
- `Baumer2024`. E. Bäumer et al. (8 authors). "Efficient Long-Range Entanglement using Dynamic Circuits." PRX Quantum 5, 030339 (2024). arXiv:2308.13065. Note 07 marked it UNVERIFIED; v07 verified it. [v07]
- `Kasatkin2026`. V. Kasatkin et al. (4 authors). "Quantum Error Correction and Dynamical Decoupling: Better Together or Apart?" arXiv:2602.19042 (2026). [v08]

### 4.9 Brainstorm-only citations: no lit verifier, metadata confirmed by the synthesis lead

These appear only in the lens files, so they are outside the verified bibliography above. I
confirmed title, authors, year and identifier on 2026-10-28 through the arXiv API, Crossref or
the page itself (Appendix A). I did not re-read their content, so the claims the lenses draw
from them are still the lenses' responsibility. In the registry they carry the mark (b).

- `Gidney2022planar`. C. Gidney, M. Newman, M. McEwen. "Benchmarking the Planar Honeycomb Code." Quantum 6, 813 (2022). arXiv:2202.11845. Decoder-theorist cites it for SI1000 but could not extract the table; use `Gidney2021honeycomb` for the definition.
- `Dennis2002`. E. Dennis, A. Kitaev, A. Landahl, J. Preskill. "Topological quantum memory." J. Math. Phys. 43, 4452-4505 (2002). arXiv:quant-ph/0110143; DOI 10.1063/1.1499754.
- `NeymanPearson1933`. J. Neyman, E. S. Pearson. "On the problem of the most efficient tests of statistical hypotheses." Phil. Trans. R. Soc. Lond. A 231, 289-337 (1933). DOI 10.1098/rsta.1933.0009.
- `Wald1945`. A. Wald. "Sequential Tests of Statistical Hypotheses." Ann. Math. Stat. 16, 117-186 (1945). DOI 10.1214/aoms/1177731118.
- `Rabiner1989`. L. R. Rabiner. "A tutorial on hidden Markov models and selected applications in speech recognition." Proc. IEEE 77, 257-286 (1989). DOI 10.1109/5.18626.
- `Efron1979`. B. Efron. "Bootstrap Methods: Another Look at the Jackknife." Ann. Statist. 7, 1-26 (1979). DOI 10.1214/aos/1176344552.
- `DavisonHinkley1997`. A. C. Davison, D. V. Hinkley. "Bootstrap Methods and their Application." Cambridge University Press (1997). DOI 10.1017/CBO9780511802843.
- `FieldWelsh2007`. C. A. Field, A. H. Welsh. "Bootstrapping Clustered Data." J. R. Stat. Soc. B 69, 369-390 (2007). DOI 10.1111/j.1467-9868.2007.00593.x.
- `ClopperPearson1934`. C. J. Clopper, E. S. Pearson. "The use of confidence or fiducial limits illustrated in the case of the binomial." Biometrika 26, 404-413 (1934). DOI 10.1093/biomet/26.4.404.
- `Wilson1927`. E. B. Wilson. "Probable Inference, the Law of Succession, and Statistical Inference." J. Am. Stat. Assoc. 22, 209-212 (1927). DOI 10.1080/01621459.1927.10502953.
- `BrownCaiDasGupta2001`. L. D. Brown, T. T. Cai, A. DasGupta. "Interval Estimation for a Binomial Proportion." Statist. Sci. 16, 101-133 (2001). DOI 10.1214/ss/1009213286.
- `BenjaminiHochberg1995`. Y. Benjamini, Y. Hochberg. "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." J. R. Stat. Soc. B 57, 289-300 (1995). DOI 10.1111/j.2517-6161.1995.tb02031.x.
- `BenjaminiYekutieli2001`. Y. Benjamini, D. Yekutieli. "The control of the false discovery rate in multiple testing under dependency." Ann. Statist. 29, 1165-1188 (2001). DOI 10.1214/aos/1013699998.
- `RomanoWolf2005`. J. P. Romano, M. Wolf. "Stepwise Multiple Testing as Formalized Data Snooping." Econometrica 73, 1237-1282 (2005). DOI 10.1111/j.1468-0262.2005.00615.x.
- `Akaike1974`. H. Akaike. "A new look at the statistical model identification." IEEE Trans. Autom. Control 19, 716-723 (1974). DOI 10.1109/TAC.1974.1100705.
- `Schwarz1978`. G. Schwarz. "Estimating the Dimension of a Model." Ann. Statist. 6, 461-464 (1978). DOI 10.1214/aos/1176344136.
- `HurvichTsai1989`. C. M. Hurvich, C.-L. Tsai. "Regression and time series model selection in small samples." Biometrika 76, 297-307 (1989). DOI 10.1093/biomet/76.2.297.
- `SelfLiang1987`. S. G. Self, K.-Y. Liang. "Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions." J. Am. Stat. Assoc. 82, 605-610 (1987). DOI 10.1080/01621459.1987.10478472.
- `Davies1987`. R. B. Davies. "Hypothesis testing when a nuisance parameter is present only under the alternative." Biometrika 74, 33-43 (1987). DOI 10.1093/biomet/74.1.33.
- `ChubbFlammia2021`. C. T. Chubb, S. T. Flammia. "Statistical mechanical models for quantum codes with correlated noise." Ann. Inst. Henri Poincaré D 8, 269-321 (2021). arXiv:1809.10704; DOI 10.4171/AIHPD/105.
- `WallmanFlammia2014`. J. J. Wallman, S. T. Flammia. "Randomized Benchmarking with Confidence." New J. Phys. 16, 103032 (2014). arXiv:1404.6025.
- `Helsen2019`. J. Helsen, J. J. Wallman, S. T. Flammia, S. Wehner. "Multi-qubit Randomized Benchmarking Using Few Samples." Phys. Rev. A 100, 032304 (2019). arXiv:1701.04299.
- `Harper2019stats`. R. Harper, I. Hincks, C. Ferrie, S. T. Flammia, J. J. Wallman. "Statistical analysis of randomized benchmarking." Phys. Rev. A 99, 052350 (2019). arXiv:1901.00535.
- `Granade2015`. C. Granade, C. Ferrie, D. G. Cory. "Accelerated Randomized Benchmarking." New J. Phys. 17, 013042 (2015). arXiv:1404.5275.
- `BravyiVargo2013`. S. Bravyi, A. Vargo. "Simulation of rare events in quantum error correction." Phys. Rev. A 88, 062308 (2013). arXiv:1308.6270.
- `Heussen2024`. S. Heußen, D. Winter, M. Rispler, M. Müller. "Dynamical subset sampling of quantum error correcting protocols." Phys. Rev. Research 6, 013177 (2024). arXiv:2309.12774.
- `Magesan2011`. E. Magesan, J. M. Gambetta, J. Emerson. arXiv title "Robust randomized benchmarking of quantum processes"; published as "Scalable and Robust Randomized Benchmarking of Quantum Processes," Phys. Rev. Lett. 106, 180504 (2011). arXiv:1009.3639.
- `Renger2025`. M. Renger et al. (65 authors, IQM). "A Superconducting Qubit-Resonator Quantum Processor with Effective All-to-All Connectivity." arXiv:2503.10903 (2025), preprint.
- `McKay2017`. D. C. McKay, C. J. Wood, S. Sheldon, J. M. Chow, J. M. Gambetta. "Efficient Z-Gates for Quantum Computing." Phys. Rev. A 96, 022330 (2017). arXiv:1612.00858.
- `Wagner2022`. T. Wagner, H. Kampermann, D. Bruß, M. Kliesch. "Pauli channels can be estimated from syndrome measurements in quantum error correction." Quantum 6, 809 (2022). arXiv:2107.14252.
- `Sivak2024`. V. Sivak, M. Newman, P. Klimov. "Optimization of decoder priors for accurate quantum error correction." Phys. Rev. Lett. 133, 150603 (2024). arXiv:2406.02700.
- `LacroixLRU2025`. N. Lacroix et al. (17 authors). "Fast Flux-Activated Leakage Reduction for Superconducting Quantum Circuits." Phys. Rev. Lett. 134, 120601 (2025). arXiv:2309.07060.
- `Khezri2023`. M. Khezri et al. (12 authors). "Measurement-Induced State Transitions in a Superconducting Qubit: Within the Rotating Wave Approximation." Phys. Rev. Applied 20, 054008 (2023). arXiv:2212.05097.
- `Heinsoo2018`. J. Heinsoo et al. (11 authors). "Rapid high-fidelity multiplexed readout of superconducting qubits." Phys. Rev. Applied 10, 034040 (2018). arXiv:1801.07904.
- `Klimov2018`. P. V. Klimov et al. (33 authors). "Fluctuations of Energy-Relaxation Times in Superconducting Qubits." Phys. Rev. Lett. 121, 090502 (2018). arXiv:1809.01043.
- `McEwen2022cosmic`. M. McEwen et al. (53 authors). "Resolving catastrophic error bursts from cosmic rays in large arrays of superconducting qubits." Nat. Phys. 18, 107-111 (2022). arXiv:2104.05219.
- `Proctor2020drift`. T. Proctor et al. (8 authors). "Detecting and tracking drift in quantum information processors." Nat. Commun. 11, 5396 (2020). arXiv:1907.13608.
- `WallmanEmerson2016`. J. J. Wallman, J. Emerson. "Noise tailoring for scalable quantum computation via randomized compiling." Phys. Rev. A 94, 052325 (2016). arXiv:1512.01098.
- `AWSBraketIQMpage`. Amazon Braket, IQM device page, https://aws.amazon.com/braket/quantum-computers/iqm/, fetched 2026-10-28. It states 20 transmon qubits "arranged in a square lattice and connected by tunable couplers", median 1-qubit fidelity 99.92% and 2-qubit fidelity 99.51%; it gives no readout fidelity.
- `Hayes2020`. D. Hayes et al. (6 authors). "Eliminating Leakage Errors in Hyperfine Qubits." Phys. Rev. Lett. 124, 170501 (2020). arXiv:1912.13131.
- `Evered2023`. S. J. Evered et al. (15 authors). "High-fidelity parallel entangling gates on a neutral atom quantum computer." Nature 622, 268-272 (2023). arXiv:2304.05420.
- `Scholl2023`. P. Scholl et al. (6 authors). "Erasure conversion in a high-fidelity Rydberg quantum simulator." Nature 622, 273 (2023). arXiv:2305.03406.
- `Teoh2023`. J. D. Teoh et al. (18 authors). "Dual-rail encoding with superconducting cavities." PNAS (2023), DOI 10.1073/pnas.2221736120. arXiv:2212.12077.
- `Gu2025arch`. S. Gu, A. Retzker, A. Kubica. "Fault-tolerant quantum architectures based on erasure qubits." Phys. Rev. Research 7, 013249 (2025). arXiv:2312.14060.
- `Kang2023`. M. Kang, W. C. Campbell, K. R. Brown. "Quantum Error Correction with Metastable States of Trapped Ions Using Erasure Conversion." PRX Quantum 4, 020358 (2023). arXiv:2210.15024.
- `Puri2020`. S. Puri et al. (12 authors). "Bias-preserving gates with stabilized cat qubits." Sci. Adv. (2020), DOI 10.1126/sciadv.aay5901. arXiv:1905.00450.
- `Putterman2025`. H. Putterman et al. (121 authors). "Hardware-efficient quantum error correction via concatenated bosonic qubits." Nature 638, 927-934 (2025). arXiv:2409.13025.

### 4.10 Rejected or unverifiable citations

| Citation or claim | Where | Reason |
|---|---|---|
| S. Ghosh and A. Kabir, APS Global Physics Summit 2026 talk | v01 | The page shows title, authors and date only; content UNVERIFIED. In-group talk |
| M. Serra-Peralta and B. M. Terhal, APS March Meeting 2025 abstract | lit 06 open question 5 | The talk exists (SerraPeralta2026 mentions it), but the abstract content is UNVERIFIED |
| Holm (1979), sequentially rejective multiple testing | numerics | Page could not be fetched; not used |
| `stimbposd` (PyPI), `mdopt` (JOSS) | lit 03 | Search-result-only leads, never fetched |
| Quantinuum Helios real-time GPU decoding with NVIDIA | lit 07 | Vendor press only, no primary paper |
| arXiv:2511.09491, 2606.16288, 2609.13087, 2602.02985, 2601.21616, 2512.24744, 1801.04042 | lit 01, 03 leads | Titles and authors now confirmed by my arXiv API check, but nobody read them. Not usable for claims |
| "b ≈ 2.7 for logical-gap postselection", attributed to arXiv:2603.08697 | lit 04 search summary | Not in the source as read; rejected |
| "A GNN logit beats the MWPM gap" | lit 04 | Seen only on a scirate listing; rejected |
| Exclusive decoders attributed to "Ahmed et al." | lit 04 search summary | Wrong authors; the paper is Smith2024 |
| Chou et al. under arXiv:2307.08737 | lit 05 seed | Wrong ID; that is Levine2024. Chou2024 is arXiv:2307.03169 |
| Note 02's removal of R. J. Harris from FigueroaRomero2021, and the four-author list in numerics, rb-theorist and skeptic | lit 02, brainstorm | Rejected by v02; the published paper has five authors |
| Kubica2023 with five authors | other-platforms | The PRX version has six (v05; Crossref) |
| Chou2024 with 35 authors; McEwen2021 with 50; Abdurakhimov2024 with 113; Paetznick2024 with 33; Vigneau2025 with 17 | lit 01, 05, 06, 07, other-platforms | Correct counts are 34, 51, 114, 32 and 16 |
| Zhang2026's 31% CZ erasure fraction | lit 05 | The v1 value; the published version gives 38(6)% (v05) |
| Paetznick2026Nature as "the journal version" of Paetznick2024 or ReichardtTess2024 | lit 06, lit 08 | It is a related paper that cites the preprint (v06, v08); take numbers from the arXiv versions |
| Stricker2020 as a d − 1 erasure example | lit 05 | Its code has a weight-1 X logical |
| IQM feedback latency figure | SC lens P10 | No fetched IQM source gives it; UNVERIFIED |
| SI1000 attributed to Gidney2022planar | decoder-theorist | Metadata correct, but the lens could not read the table; cite Gidney2021honeycomb Tables 1 and 2 (v07) |
