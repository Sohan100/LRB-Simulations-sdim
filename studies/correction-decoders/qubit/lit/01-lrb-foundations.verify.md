# Verification of lit review 01: LRB foundations

Verified file: `studies/correction-decoders/qubit/lit/01-lrb-foundations.md`.
This is an independent check of its citations and claims, run on 2026-10-28.

How I checked. I fetched the arXiv abstract page of every citation myself and
compared title, authors, version dates and identifiers. For journal DOIs I pulled
the Crossref record, because doi.org redirects to publisher pages that need a
login. For each key finding I opened the cited full text, using arXiv HTML or
ar5iv. For Ceasura2022 I used the same alphaxiv mirror the author used. Quotes
below are short phrases copied from those pages. Statements about papers are
**from literature**. My own checks are labelled **verified numerically** or
**verified analytically**.

## Bottom line

- All 24 arXiv IDs resolve to the papers named in the file. I found no
  hallucinated paper and no wrong ID.
- Metadata is correct for 22 of 24 entries. Chen2021 pairs the arXiv title with
  the Nature venue, but the Nature version has a different title and byline.
  Vigneau2025 has 16 authors, not 17, and is now published in PRX Quantum. Three
  other entries can take a fuller venue.
- Of the 22 key findings in the structured summary, 21 are supported by the
  cited text. One is partially supported: the const=k conjecture, which misreads
  Combes2017's "syndromes refreshed" condition.
- Outside the summary, five sentences in the file are wrong or overstated. Two
  say a paper has "no RB" when it does have physical-gate RB. Implication 11 says
  "d >= 3" where the source talks about a d=4 code. Two findings overstate
  Chen2021 and Bluvstein2023 slightly.

## 1. Citation metadata

| Key | Verdict | What I checked, and corrections |
|---|---|---|
| Combes2017 | ok | arXiv:1702.03688, single version 13 Feb 2017. Combes, Granade, Ferrie, Flammia. No journal ref on arXiv. |
| HarperFlammia2019 | ok | arXiv:1806.02359, v1 6 Jun 2018 to v4 30 Jan 2019. PRL 122, 080504, published 26 Feb 2019. DOI matches Crossref. |
| Ceasura2022 | ok | arXiv:2212.05488, single version 11 Dec 2022. Ceasura, Iyer, Wallman, Pashayan. |
| Ziyad2025 | ok | arXiv:2512.08893, v1 9 Dec 2025, v3 19 Feb 2026. Ziyad, Blume-Kohout, Rudinger. |
| Kabir2025 | ok | arXiv:2511.12777, v1 16 Nov 2025, v2 12 May 2026. Eight authors in the listed order, the last is Bart Sorée. The abstract never mentions LRB. The LRB-D material is in Sec. VII of v2. |
| ChenBaldwin2025 | ok | arXiv:2502.00154, v1 31 Jan 2025, v2 25 Jul 2025. Phys. Rev. Research 7, 043065, published 17 Oct 2025. Crossref confirms DOI 10.1103/j8sz-6ws9. |
| Lacroix2025 | ok | arXiv:2412.14256, single version 18 Dec 2024, 215 authors, Lacroix first, Satzinger last. Nature 645, 614-619, online 26 May 2025. The Nature title spells it "colour code". |
| Mayer2024 | ok | arXiv:2404.08616, single version 12 Apr 2024, 23 authors, Mayer first, Stutz last. No journal ref. |
| Proctor2022 | ok | arXiv:2112.09853, v1 18 Dec 2021, v2 10 Oct 2022. PRL 129, 150502 (2022). |
| Vigneau2025 | fix | arXiv:2503.12869, v1 17 Mar 2025 to v3 21 Aug 2025. The arXiv page lists 17 names, but Hsiang-Sheng Ku appears twice. Crossref has 16 authors. DOI 10.1103/8m33-wn4g resolves to PRX Quantum 6, 040369, published 22 Dec 2025. Write "16 authors" and "PRX Quantum 6, 040369 (2025)". |
| Chen2021 | fix, minor | The arXiv:2102.06132 metadata are right: title "Exponential suppression of bit or phase flip errors with repetitive error correction", 91 authors, Z. Chen first, J. Kelly last. The Nature version, Nature 595, 383-387 (2021), DOI 10.1038/s41586-021-03588-y, is titled "Exponential suppression of bit or phase errors with cyclic error correction", and Crossref gives its byline as Google Quantum AI followed by 96 named authors. Pairing the Nature venue with the arXiv title gives a mismatched reference. |
| Lin2026 | ok | arXiv:2607.01473, single version 1 Jul 2026, 147 authors, W. Lin first, J.-W. Pan last. |
| Paetznick2024 | ok | arXiv:2404.02280, v1 2 Apr 2024, v3 17 Nov 2024, 33 authors, Paetznick first, Svore last. No journal ref on arXiv. |
| Reichardt2024 | ok | arXiv:2411.11822, v1 18 Nov 2024, v3 9 Jun 2025, 72 authors, Reichardt first, Bloom last. |
| Bluvstein2023 | ok | arXiv:2312.03982, single version 7 Dec 2023, 21 authors, Bluvstein first, Lukin last. Crossref gives Nature 626, 58-65, online 6 Dec 2023, in the 1 Feb 2024 issue. Add volume and pages. |
| Smith2024 | ok | arXiv:2405.03766, single version 6 May 2024. Commun. Phys. 7, 386, published 28 Nov 2024. |
| Gottesman2016 | ok | arXiv:1610.03507, v1 11 Oct 2016, v2 14 Oct 2016, sole author. No journal ref. |
| Reinhold2020 | ok | arXiv:1907.12327, single version 29 Jul 2019, six authors as listed. Crossref gives Nature Physics 16, 822-826, online 8 Jun 2020. Add volume and pages. |
| Levine2024 | ok | arXiv:2307.08737, v1 17 Jul 2023 to v3 20 Mar 2024, 113 authors, Levine first, Painter last. PRX 14, 011051 (2024). |
| Liu2026 | ok | arXiv:2604.08672, single version 9 Apr 2026, six authors as listed. |
| Morvan2021 | ok | arXiv:2008.09134, single version 20 Aug 2020, 10 authors, Morvan first, Siddiqi last. PRL 126, 210504 (2021). |
| RyanAnderson2021 | ok, add venue | arXiv:2107.07505, single version 15 Jul 2021, 17 authors, Ryan-Anderson first, Stutz last. The arXiv page has no journal ref, but Crossref shows it appeared as Phys. Rev. X 11, 041058 (2021), DOI 10.1103/PhysRevX.11.041058. |
| RyanAnderson2022 | ok | arXiv:2208.01863, single version 3 Aug 2022, 37 authors, Ryan-Anderson first, Hayes last. No journal ref on arXiv. Its relevance note "no RB" is wrong, see section 3. |
| Postler2022 | ok | arXiv:2111.12654, v1 24 Nov 2021 to v3 2 Nov 2022, 13 authors, Postler first, Monz last. Nature 605, 675-680 (2022). |

## 2. Key findings from the structured summary

Numbering follows the order of the structured summary.

**KF1, Combes2017, fit model and p_L. Supported.** Sec. 3, Eq. 10 gives
p_L = (dim H_L F(Λ_L, 1) - 1)/(dim H_L - 1). The text says "the dimension
dependence of the decay parameter only involves the dimension of the logical
space". The Fig. 1 caption builds the logical channel from "additional noisy
channels" for "the syndrome recovery, measurement, and preparation".

**KF2, Combes2017, assumptions and offline recovery. Supported.** Sec. 1: "we
require the noise to be time-independent, Markovian, and gate-independent (all
at the level of logical Clifford gates)", and "syndrome measurement errors are
uncorrelated between rounds of error correction". Fig. 1 caption: "the syndromes
are refreshed during rounds of error correction". Sec. 4.2: "the recovery can be
done in post-processing for stabilizer codes". Sec. 5: the correction "can be
applied in post-processing rather than concurrently". One reading point matters
for KF21. Protocol step 1(b)ii follows each Clifford with "a round of error
correction (with or without recovery) and a refresh of the ancilla register". So
"refreshed" means fresh ancillas each round. Combes also allows identity
recovery explicitly, to estimate Pr(No).

**KF3, HarperFlammia2019. Supported.** "There is at present no complete theory
for fitting an RB decay curve with post-selection." The infidelities are "0.60(3)%
for the [4,2,2] code and 5.8(2)% for the 2 physical qubits". Postselection uses
only the parity of the final four-qubit measurement. The seed note on Real RB is
also right: "The group R(2) has only 576 elements, in contrast to the 11,520
elements of C(2)".

**KF4, Ceasura2022. Supported.** The abstract says the syndrome qubits "are
strongly coupled to the logical qubits via a decoder", under gate- and
time-independent noise. The full-text details in finding 20 also hold on the
alphaxiv mirror. The main example assumes "line 4 of algorithm 2 completely
fails", the Fig. 4 caption says "the amplitude of the oscillations decrease as
register B probabilistically forgets", and the authors call the examples
"somewhat contrived". The file leaves out Example 3. There a classical repetition
code stored in the syndrome register keeps the oscillations alive when "Each of
the qubits in register B has a 95% chance of resetting" (Fig. 5, 142 copies of
the register). Better reset alone does not guarantee exponential decay. This
matters for implication 3.

**KF5, Ziyad2025. Supported, abstract only.** "non-Markovianity can emerge from
Markovian physical operations if (and only if) the physical qubits are not
necessarily returned to the code subspace after every round of QEC". The source
says "can emerge". The summary's "give ... if and only if" drops the "can".
Finding 21 in the file keeps it.

**KF6, Kabir2025. Supported.** Sec. VII-A: "we only perform a single terminal
syndrome extraction", and "we postselect measurements with the all zeroes
syndrome". Sec. VII-B: "all the LRB-D curves decay much faster than a single
physical, unprotected qutrit". Sec. VIII: terminal detection "can produce decay
curves with non-Markovian structure that are not always well summarized by a
single exponential fit", and "A dedicated future LRB-D study will address this
limitation by examining interval-check postselection." This is the user's group's
own paper, as the file says.

**KF7, ChenBaldwin2025. Supported.** Sec. III, Eq. 19 defines the post-selected
survival as computational survival over retention. In the no-seepage regime,
Sec. IV.3.3, a single exponential gives the ratio r/t. Sec. IV.2.4: "more data is
thrown away with longer sequences making the measured survival probabilities
noisier". Sec. III, on earlier heuristic work: "its applicability is unknown for
general cases".

**KF8, Levine2024. Supported.** Fig. 3 caption: "11 erasure checks evenly spaced
throughout the circuit". The acceptance is fit "to an exponential decay to 0" and
gives an "erasure error per Clifford of 4.38(3)×10^-3". The conditional survival
is fit "with offset 1/2" and gives a residual of 1.01(1)×10^-4 per Clifford. App.
F, Fig. 10(A), for postselection on mid-circuit checks only: it "does not decay to
a known offset, but a fit to A e^{-pN} is valid for short times".

**KF9, Lacroix2025. Supported, arXiv v1 values.** Sec. IV: "with a cycle of
error correction between each Clifford gate", m = 0 to 10 with 25 sequences.
Fig. 3 caption: "each repeated 20 000 times", neural-network decoder. App. G:
eps_C = (1/2)(1 - p_C/p_I), with p from a fit to "A p^m + 1/2", "Using the idling
experiment as reference". Sec. IV gives eps_C = 0.0027(3) against "the logical
error per cycle, ε̄3=0.0171(3)". App. E.1.1: Chromobius "cannot be used to decode
logical randomized benchmarking circuits". App. E.1.3: the network "receives
information about logical gates applied in between cycles". Finding 5's point on
tomography paraphrases App. G fairly. App. G says tomography "is sensitive to
logical SPAM" and that LRB "offers a more reliable method". It does not say in so
many words that SPAM was the reason for the choice.

**KF10, Mayer2024. Supported.** Sec. III.1: "sequence lengths L∈{2,6,10,14}", "10
random circuits", 100 shots per circuit, p(L) = a f^L + 1/4, and "0.9991(2) and
0.9980(8) for H1-1 and H2-1, respectively". Introduction: "we do not employ
fault-tolerant non-Clifford gates or rounds of QEC". Sec. II.1: "at most a few
percent of the shots" are discarded. The RB samples the full two-qubit Clifford
group, with 11,520 elements. It reaches a per-CNOT fidelity by dividing by the
mean number of CNOTs per Clifford.

**KF11, Proctor2022. Supported.** The fits are "S=Ap^d". The paper derives
p^2 ≈ Σ_L Ω(L) γ_{L^-1} γ_L, so each layer and its inverse together contribute
the p^2. The estimate r_Ω = (4^n - 1)(1 - p)/4^n targets the Ω-averaged
entanglement infidelity of Pauli-dressed layers. "We assume that errors are
Markovian", and the Pauli layers twirl the errors "into stochastic Pauli errors".

**KF12, Chen2021 and Vigneau2025. Supported.** Chen2021 Sec. V: "we discard 27%
of runs each round". Its supplement: "the probability of a logical error is about
0.002 each round". Vigneau2025 Sec. V estimates "P_S=η_N/η_{N−1}≈0.67" for the
separable-state runs, against 71% for the Bell state, and reports 0.25(2)% to
0.91(3)% per cycle. The 0.67 is a fitted estimate, and the two experiments use
different codes and check counts.

**KF13, Gottesman2016. Supported.** Sec. III: the 4-qubit code "can only improve
error rates via post-selection". Sec. II: "the encoded circuit must have a lower
error rate for all circuits in the family". App. A.1 says to "compare pL/pS (the
conditional probability of logical error in the circuit) with pU" and gives
"p<6/363≈1.6%" for T=1 and "p<0.12%" for T=100. Finding 14 in the file says "100
gates". In the paper T counts logical time steps, each with a transversal
one-qubit gate, and the bounds are described as rough.

**KF14, Lin2026. Supported.** App. E.1: "the scalable benchmark is instead the
unconditional decoded performance". The conditional fidelity "does not represent
the fidelity of a composable logical operation". The conditional values are
0.9952(10) for H and 0.9998(4) for S, at 1.43(2)% and 1.44(2)% retention. The
unconditional 0.854(1) and 0.846(1) are in the main-text H and S sections, with
0.8544(12) and 0.8461(8) in App. I.2 and J.2. App. E.1 does not set the two side by
side. The file does that, which is fine.

**KF15, Paetznick2024. Supported.** Table 1 gives 0.50% physical, 0.05% with
correction at gain 9.8 with 18,035 of 24,200 runs pre-accepted, about 74.5%, and
0.001% with detection at gain 500 with 17,389 of 24,200 accepted, about 71.9%.
Sec. III.2.1, for the [[12,2,4]] code: "the post-rejection rate is second order in
the physical error rate", "since single faults are corrected instead of
rejected". The 75% in correction mode comes from pre-selection during
resource-state verification, not from rejected syndromes.

**KF16, Bluvstein2023 and Smith2024. Supported.** Bluvstein Methods set "a
threshold weight of the overall correction weight on the decoding hypergraph",
which "enables a continuous trade-off", and "we also apply error correction to
return to the codespace". Smith's abstract: "up to a quadratic improvement in
logical failure rates below threshold", and "the exclusion rate decays with code
distance". The quadratic improvement is Smith's result only.

**KF17, Reinhold2020. Supported.** They "interleave the S(π/2) gate between
randomly selected Clifford operations". Fig. 3 fits "A e^{-γn} + 1/2" and takes
γ_IRB - γ_RB as the gate error. "the error-corrected gate has an error
probability of 2.4%±0.1%", against 4.6%±0.1% without correction. The qubit is a
binomial "kitten" code in a cavity.

**KF18, Reichardt2024. Supported, attribution needs a fix.** Sec. C.2:
"generating a group of size 36", "with error detection between each pair of
elements", and "the encoded circuits outperform the corresponding physical
baselines". The loss trade-off, "Trying to correct more loss generally leads to a
higher acceptance rate, but also a higher error rate", is in the Fig. 6 caption
of the 24-logical-qubit cat-state experiment, not in the random-circuit section.
Neither experiment is RB.

**KF19, Kabir2025 and Morvan2021. Supported as an absence claim.** Morvan2021's
abstract gives "single-qutrit gate infidelity as low as 2.38 × 10^-3". My own two
searches found no other qudit logical RB. They did turn up an in-group APS talk and
a GKP qudit memory experiment, listed in section 4. Neither is a logical RB result.

**KF20, gate action plus a decoder conjecture citing Lacroix2025. Supported.** I
reproduced the tableau result with stim 1.15.0 and by hand. H_L swaps S_Z1 with
S_X1, S_Z2 with S_X2, and X_L with Z_L. S_L fixes S_Z1, S_Z2 and Z_L. It sends
S_X1 to -Y0 Y1 X2 Z3 = +S_Z1·S_X1, S_X2 to -Y1 Z2 X3 Y4 = +S_Z2·S_X2, and X_L to
+Y0 Z2 X3 = Y_L = i X_L Z_L. Lacroix2025 supports building detectors frame by
frame: "We track stabilizer transformations and determine which measurements to
compare in each cycle according to these rules". I also tested the hyperedge
conjecture in a toy model, see section 5. It holds there.

**KF21, const=k conjecture. Partially supported.** Several parts hold up.
Ziyad2025's condition, Ceasura2022's mechanism and Kabir2025's terminal-detection
result all point to non-exponential const=1 curves. The syndrome count, 2^4 = 16
against 3^4 = 81, is right. Combes2017 supports offline Pauli-frame recovery, and
Gottesman2016 supports that a d=2 code cannot correct, so the corrected logical
error stays first order. Three parts do not hold up.

1. "never returns the data to the code space" holds for const=1 only. For
   const=k with k >= 2, accepted shots pass trivial syndromes at the forward
   checks the policy uses. That projects them onto the code space, up to
   measurement errors. The memory only builds up between used checks.
2. "fails the Combes ... conditions" misreads Combes2017. Its "syndromes are
   refreshed" condition is the "refresh of the ancilla register" in step
   1(b)ii, and the repo resets all four ancillas every round. Combes also allows
   identity recovery and scores it against the code projector through Pr(No).
   The real gap is that Combes has no postselection. The same misreading sits in
   section 3, item 1 of the file.
3. With 5p measurement flips, a per-round correction can push the data out of
   the code space by itself. "Near-exponential" is a hypothesis to test. The
   file's open question 2 already says this.

**KF22, discard-rate arithmetic. Supported.** **Verified numerically**: 0.73^22 =
9.84e-4 and 0.67^22 = 1.49e-4. Combes2017 Sec. 5 supports that the correction
can be applied in post-processing. The 27% comes from a 2021 Sycamore d=2
surface code and the 33% from an IQM [[4,2,2]] code. The [[5,1,2]] rate on a given
device needs its own estimate. "Sustainable route" is a judgment, and the file
labels it as a conjecture.

## 3. Problems elsewhere in the file

These sentences are not among the 22 summary findings, but they are in the file
and should change.

1. Section 4 says Bluvstein2023 "uses no RB". That is wrong as written. The
   Methods report physical single-qubit RB, "we perform randomized benchmarking
   using 0, 10, 20, 30, 40, and 50 local Z(π/2) rotations", and use RB again to
   compare two-qubit gates across trap configurations. The intended point holds,
   since there is no logical-level RB. Write "no logical RB, physical-gate RB in
   Methods".
2. Section 1b and the structured summary say RyanAnderson2022 has "no RB". Also
   wrong as written. Its Table 1 caption reports physical error rates "measured
   using randomized benchmarking experiments". Sec. III says a SPAM-free logical
   fidelity "would require randomized benchmarking or gate set tomography", so
   there is no logical RB. The summary also says "no RB" for RyanAnderson2021,
   which uses "parallel randomized benchmarking" for physical gates in App. A.
   The file's own section 4 wording, "no RB at the logical level", is the right
   one.
3. Section 3, item 11 says post-correction rejection is "second order for
   d >= 3 (Paetznick2024)". The source says this for the d=4 [[12,2,4]] code,
   because there "single faults are corrected instead of rejected". A d=3 code
   cannot correct every single fault and also flag weight-2 errors. In the
   Steane code every nonzero X or Z syndrome already belongs to a weight-1 error,
   **verified analytically** from the perfect Hamming code. Change it to
   "d >= 4". KF15 in the summary has it right.
4. Finding 22 says Chen2021's discard rate and logical error are "both matching
   a depolarizing model". Only the discard rate matches, "in good agreement with
   the model prediction". For the logical error the paper says "The model
   slightly underestimates the logical error."
5. Finding 18 calls 72% the "raw" GHZ fidelity. Bluvstein2023 says 72(2)% is "the
   resulting GHZ state fidelity obtained using the fault-tolerant algorithm",
   that is, after correlated decoding with no discards. The 90% and 99.85% figures
   are right.

I also checked these and found them correct. The Postler2022 seed note holds:
78.9(5)% acceptance for fault-tolerant |0>_L and 13.7(3)% for the magic state,
flag-based aborts, and RB only "for single-qubit gates in the 16-ion chain". The
Mayer2024 and HarperFlammia2019 seed notes are right, and Paetznick is first
author of arXiv:2404.02280.

## 4. Related sources the review did not cite

- S. Ghosh and A. Kabir, "Logical Randomized Benchmarking a Qutrit Surface
  Code", APS Global Physics Summit 2026, 18 March 2026,
  https://meetings-archive.aps.org/smt/2026/mar-m09/7/. This is an in-group
  talk. The page I fetched shows title, authors, date and "Planned for
  submission to arxiv and PRX very soon", but no abstract. A search-engine
  summary credited it with interval-postselection results that I could not see
  on the page. Its content is UNVERIFIED. If cited at all, cite it as an in-group
  talk.
- B. L. Brock et al., "Quantum Error Correction of Qudits Beyond Break-even",
  arXiv:2409.15065, Nature 641, 612-618 (2025), DOI 10.1038/s41586-025-08899-y.
  GKP qutrit and ququart memories with gains of 1.82 and 1.87 over the best
  physical qudit. The abstract reports memory only, no RB. It is the nearest
  qudit QEC experiment and fits finding 26's "no experimental qudit logical RB".

## 5. Checks I ran myself

All were light, run on the login node under `nice -n 19` with stim 1.15.0.

- Gate action, **verified numerically** and **verified analytically**. Details
  under KF20.
- CONTEXT.md's single-qubit syndrome table, **verified numerically**. X0 and X3
  give (1,0), X2 and X4 give (0,1), X1 gives (1,1). Z0 and Z2 give (1,0), Z3 and
  Z4 give (0,1), Z1 gives (1,1). X2X4·X_L = S_X1·S_X2 and Z3Z4·Z_L = S_Z1·S_Z2,
  so each ambiguous pair differs by a logical. Implication 8 can drop "still to
  be verified".
- Hyperedge conjecture, **verified numerically in a toy model**. The model is one
  ideal MPP round, a noisy MPP round with 1% outcome flips, logical S, then two
  more noisy rounds. Across S I used the transformed detector
  D_X1(t+1) = m_X1(t+1) xor m_X1(t) xor m_Z1(t), and the same for X2 with Z2.
  stim builds the detector error model without complaint, so these detectors are
  deterministic. One Z1 outcome flip in the round before S then flips D_Z1(t),
  D_Z1(t+1) and D_X1(t+1), three detectors at once. Z2 does the same with X2. I
  did not test the full SI1000 circuit. The claim that matching "may not be
  enough" for the real circuit is still a conjecture, but the three-detector
  mechanism exists.
- Arithmetic, **verified numerically**. 0.73^22 = 9.84e-4, 0.67^22 = 1.49e-4,
  6/363 = 1.65%, 1 - 1/16 = 0.9375. Paetznick's acceptances come to 74.5% and
  71.9%. Liu2026's 5.3e-3/7.6e-4 = 7.0 matches the paper's "seven times".

## 6. Suggested edits to 01-lrb-foundations.md

1. Vigneau2025 row: 16 authors, venue PRX Quantum 6, 040369 (2025).
2. Chen2021 row: give the Nature title "Exponential suppression of bit or phase
   errors with cyclic error correction" and the Google Quantum AI byline, or cite
   the arXiv version under its own title.
3. RyanAnderson2021 row: add Phys. Rev. X 11, 041058 (2021), DOI
   10.1103/PhysRevX.11.041058.
4. Bluvstein2023 row: Nature 626, 58-65 (2024). Reinhold2020 row: Nature Physics
   16, 822-826 (2020).
5. Write "no logical RB" instead of "no RB" for Bluvstein2023, RyanAnderson2021
   and RyanAnderson2022.
6. Implication 11: change "d >= 3" to "d >= 4".
7. Implication 1 and KF21: drop "fails Combes2017's 'syndromes refreshed'
   condition". Say instead that Combes's reduction has no postselection, and
   that under const=k nothing returns the data to the code space between the
   checks the policy uses.
8. Finding 20: add Ceasura2022's Example 3, where a repetition code keeps the
   oscillations at 95% reset.
9. Finding 22: the model matches the discard rate but slightly underestimates
   the logical error.
10. Finding 24: say the loss trade-off comes from the cat-state experiment, Fig. 6.
11. Finding 18: write "72% after decoding, with no discards" instead of "72% raw".
