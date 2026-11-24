# Verification of lit review 06: small surface codes and fold-transversal gates

Independent check of `lit/06-small-surface-codes.md`, written 2026-10-28. I re-fetched every
source myself and did not reuse the author's summaries. Labels follow CONTEXT.md. **From
literature** means I read the statement in the fetched source, and I quote it where it matters.
**Verified numerically** means I re-ran it in stim. **Verified analytically** means I derived
it by hand.

## Bottom line

- All 26 bibliography entries are real. Every arXiv ID, title and author list matches its
  arXiv abstract page. Every journal DOI resolves on Crossref to the stated journal, volume and
  article number. I found no invented paper and no wrong identifier.
- One metadata error: Paetznick2024 has 32 authors on arXiv, not 33. Two entries are
  incomplete. SST2025 is published as PRX Quantum 7, 010335 (2026), and KYP2015 appeared in
  New J. Phys. 17, 083026 (2015).
- Of the 21 key findings, 17 are supported and 4 are partially supported (KF6, KF9, KF13,
  KF19). None is unsupported.
- The body has problems that the key-findings list hides. The worst are a Breuckmann-Burton
  quote applied to the wrong code in section 0, a blanket "no logical H at d=2" that one of
  the author's own fetched papers partly contradicts, and a selective reading of Linke2017.
  Section 3 lists them all.
- Open question 6 can be closed. The Nature 2026 article is a different paper that cites
  arXiv:2404.02280, not its published version.
- The note's stim checks reproduce exactly when I re-run them.

## Method

- arXiv abstract pages for all 24 arXiv IDs, fetched with curl. I parsed title, authors,
  version dates, DOI and journal reference from the page's `citation_*` meta tags and the
  submission history.
- Crossref API records for all 21 journal DOIs, plus a title search for a journal version of
  BDB2024. I found none.
- Full texts. I used arXiv HTML for the 13 papers from 2021 on and ar5iv for the 11 older ones,
  and searched each for every quoted phrase and number in the note.
- Direct fetches of the Error Correction Zoo page, the IQM blog post, the Quantum journal page
  for Gehér et al. and the Nature page for DOI 10.1038/s41586-026-10628-y. The Nature page
  served its abstract and reference list without a login.
- Two adversarial web searches. One looked for any hardware demonstration of a fold-transversal
  surface-code gate, the other for any LRB study of the qubit [[5,1,2]] with correction. Both
  came back empty, which agrees with the note.
- A stim 1.15 script in my scratchpad that takes under a second (section 4).

## 1. Bibliography

| Key | What I confirmed | Verdict | Correction or note |
|---|---|---|---|
| Moussa2016 | arXiv:1603.02286, sole author J. E. Moussa, v1 7 Mar 2016, v3 15 Oct 2016; Crossref PRA 94, 042316 | ok | none |
| BB2024 | arXiv:2202.06647, N. P. Breuckmann and S. Burton; Crossref Quantum 8, 1372, 13 Jun 2024 | ok | none |
| VK2022 | arXiv:2112.01446, M. Vasmer and A. Kubica; Crossref PRX Quantum 3, 030319 | ok | none |
| Kabir2025 | arXiv:2511.12777, the eight listed authors in order, v1 16 Nov 2025, v2 12 May 2026; no journal ref | ok | none |
| Chen2026 | arXiv:2412.01391, four listed authors; Crossref PRL 136, 130601, issue 13, 31 Mar 2026, with the PRL title the note gives | ok | none |
| SST2025 | arXiv:2505.13599, three listed authors, v4 6 Jul 2026, DOI 10.1103/sk5y-25b1 on the arXiv page | ok, incomplete | Crossref resolves the DOI to PRX Quantum 7, 010335, 19 Feb 2026. Name the journal. |
| KZ2024 | arXiv:2310.06917; Crossref PRX Quantum 5, 020360 | ok | none |
| Andersen2020 | arXiv:1912.09410, nine listed authors; Crossref Nat. Phys. 16, 875-880 | ok | volume and pages can be added |
| Marques2022 | arXiv:2102.13071, 13 authors; Crossref Nat. Phys. 18, 80-86 | ok | none |
| Bultink2020 | arXiv:1905.12731, 12 authors; Crossref Sci. Adv. 6, eaay3050 | ok | none |
| Linke2017 | arXiv:1611.06946, seven listed authors; Crossref Sci. Adv. 3, e1701074 | ok | none |
| Takita2017 | arXiv:1705.09259, five listed authors; Crossref PRL 119, 180501 | ok | none |
| HF2019 | arXiv:1806.02359; Crossref PRL 122, 080504 | ok | none |
| BDB2024 | arXiv:2406.09951, three listed authors, v1 only | ok | no journal version on arXiv or Crossref |
| Krinner2022 | arXiv:2112.03708, 16 authors; Crossref Nature 605, 669-674 | ok | none |
| Lacroix2025 | arXiv:2412.14256, 215 authors; Crossref Nature 645, 614-619 | ok | the Nature title spells it "colour" |
| RA2021 | arXiv:2107.07505, 17 authors. The arXiv page lists no DOI; Crossref confirms PRX 11, 041058, Dec 2021 | ok | none |
| Paetznick2024 | arXiv:2404.02280 v3, 17 Nov 2024; no journal ref | count wrong | 32 authors, not 33 |
| KYP2015 | arXiv:1503.02065; Crossref New J. Phys. 17, 083026 | ok | name the journal |
| ECZ512 | page title "[[5,1,2]] rotated surface code"; latest edit 2026-09-24 | ok | Zoo editors are V. V. Albert and P. Faist. The "smallest code" sentence cites "V. V. Albert, private communication, 2026". |
| MBG2023 | arXiv:2302.02192; Crossref Quantum 7, 1172 | ok | none |
| Geher2024 | arXiv:2312.11605; Crossref and journal page agree on Quantum 8, 1394 | ok | none |
| Sahay2026 | arXiv:2509.05212, seven listed authors; Crossref PRX Quantum 7, 033006 | ok | none |
| PR2024 | arXiv:2112.03785; Crossref PRA 110, 012419 | ok | none |
| Lin2026 | arXiv:2607.01473, 147 authors, v1 1 Jul 2026 | ok | none |
| IQM2025 | "How IQM is tackling quantum error correction", dated 14 Aug 2025, by Dr. Francisco Revson Fernandes Pereira | ok | canonical URL is https://iqm.tech/blog/how-iqm-is-tackling-quantum-error-correction/ |

Seed corrections in section 4 of the note. I checked all ten and every one is right. Moussa
works "the specific case of D=5 in detail", and a full-text search finds no D=2 or D=3 case.
Breuckmann and Burton's worked example is Bring's [[30,8,3]] code, with hyperbolic, hypergraph
and balanced product codes in their Appendix B.

## 2. Key findings

| # | Claim, shortened | Verdict | Evidence |
|---|---|---|---|
| 1 | [[5,1,2]] is the D=2 member of Moussa's folded square family | supported | From literature: "The square contains 2D^2-2D+1 qudits, D^2-D vertices and faces". H-bar uses "H and H-dagger on fold qudits and a mirror gate on paired qudits", and S-bar uses "S and S-dagger on fold qudits and CZ and CZ-dagger on paired qudits". The construction is "for qudits of any dimension" (abstract). D=2 gives 5 qubits and 2+2 checks (verified analytically). |
| 2 | Vasmer-Kubica give this [[5,1,2]] with our gate set | supported | From literature: "Morphing the Steane code produces a [[5,1,2]] code with a fault-tolerant implementation of the Clifford group". The Fig. 1 caption reads "apply S^{±1} to the light/dark grey qubits and CZ to the white qubits; H-bar = H^{⊗5} SWAP". Mapping their qubits (1,2,3,1̄,2̄) to ours (0,1,4,2,3) gives our four checks (verified numerically). |
| 3 | BB2024 generalize folding, recover Moussa, call it decoder-agnostic, leave noise to future work | supported | From literature: "trivially fault-tolerant"; "compatible with any decoding scheme, as it can be executed between rounds of error correction"; "We leave to future work establishing the performance of these fold-transversal gates under simulated noise and error correction rounds, as well as their experimental realization". The H-type gate works for any ZX-duality (Thm 6). The S-type gate needs extra conditions (Thm 7). |
| 4 | The Zoo lists the same code and S; Burton et al. call it a genon code | supported | From literature: the Zoo tableau and "S_0 S_2 S_3^† CZ_{1,4} implements the logical gate S-bar". BDB2024: "The surface codes, such as the [[5,1,2]] code are genon codes on a sphere". The relabelling (0,1,2,3,4) to (0,3,4,1,2) maps the Zoo checks and S onto ours (verified numerically). The Zoo's "smallest code" sentence rests on a private communication. |
| 5 | Sdim paper: terminal-only LRB-D on [[5,1,2]]_3, non-Markovian structure | supported | From literature: "we only perform a single terminal syndrome extraction ... and we postselect measurements with the all zeroes syndrome". The code is cited to Moussa (ref. 74). "all the LRB-D curves decay much faster than a single physical, unprotected qutrit"; "terminal detection can produce decay curves with non-Markovian structure". |
| 6 | Rotation duality gives no phase gate; rotated even-d codes have no H-plus-permutation logical H | partially supported | The quote is exact, but BB2024 apply it to "another ZX-duality that rotates the surface code by 90 degrees" for the code of their Fig. 1i, "Moussa's original construction", which is the unrotated code. The parity argument is right (verified analytically). Rotated d=2 has 1 X check and 2 Z checks, so H on every qubit plus a permutation cannot preserve the stabilizer group. "Used in all d=2 hardware experiments" is too broad, since the [[4,2,2]] and BDB [[4,1,2]] runs are also d=2. |
| 7 | Chen: H by patch rotation, S at the half-cycle; 2-3x memory, or the memory baseline with full reweighting | supported | From literature: "Transversal H gates on all data qubits followed by a π/2 patch rotation"; "a rotation is equivalent to two successive reflections" done with 2D-AODs. "the half-cycle state, is simply an unrotated surface code state along with a few unentangled qubits at the boundary". SD6 plus pre-round depolarization, d from 3 to 13. "consistently 2 to 3 times higher than the X memory circuit"; VTB-FR "manages to suppress the logical error rate down to the memory baseline". MBG2023: "The half-cycle state is remarkable for being an unrotated surface code state". |
| 8 | Weight-3 hyperedges; split MWPM has no threshold; ~0.5% with lom; 2-3 rounds worse | supported, with a caveat | From literature: "a measurement error on an ancilla qubit measuring a Z-check immediately following an S gate triggers three detectors". Fig. 11 caption: "no evidence of a threshold". Table 3 under SI1000 gives 0.4963% and 0.5036% for the two-qubit Clifford circuits only. Repeated-gate thresholds are 0.32-0.45%, against 0.44-0.45% for memory. "We also run the same experiments with 2 and 3 QEC rounds ... but we see worse performance". SST blame the longer circuit. |
| 9 | S-gate hook drops minimum failure weight from 3 to 2; fold S prone to correlated noise | partially supported | SST's numbers are right: "\|e_min\|=2,4" for the repeated-S X-basis experiment, "one lower due to a 'hook' error ... from the long-range CZs". But e_min is "the lowest-weight undetectable error pattern ... which leads to a decoding failure ... when decoding with the single-lom decoder", and SST find d_circ = d for d = 3, 5. So the drop belongs to their decoder in one experiment, not to the circuit. The KZ2024 part is supported by Sec. III.6: fold-transversal H and S "can propagate errors by a factor of two in general and can also induce long-range correlated errors". |
| 10 | Color-code LRB: 0.0027(3) per Clifford vs 0.0171(3) per cycle, detectors adjusted for H and S | supported | From literature: "m=0 to 10, with 25 random sequences for each m"; with the neural-network decoder "ε_C = 0.0027(3)", "significantly lower than the logical error per cycle, ε_3 = 0.0171(3)". "a transversal H gate turns an X stabilizer into a Z stabilizer"; "an S-gate transforms X stabilizers into Y stabilizers". Magic-state postselection is "keeping on average 74.8 % of the runs". |
| 11 | Superconducting d=2 keeps ~40-55% per cycle | supported | From literature, Andersen: preparation "success-probability of 25.1%"; "the probability to observe no errors slowly increases with N from about 40% to 50%"; "After N=10 cycles, the success probability ... approaches 10^-4"; logical X_L error "3.1% ± 0.45%" and Z_L error "2.6 ± 1.3%" per cycle. Marques: "γ_pip ∼ 45%" with P(n) = A(1-γ)^n; "We believe that the dominant error source is instead leakage to higher transmon states incurred during CZ gates". |
| 12 | d=3 MWPM ~3% per 1.1 us cycle with ~8% leakage rejection; HMM leakage flags | supported | From literature, Krinner: "error-model-free approach and apply corrections in postprocessing"; "3 % per cycle when rejecting experimental runs in which leakage is detected"; "rejected data fraction per cycle of 8 %"; Table 3 gives 0.032(1) and 0.029(1) with r_c = 0.921(3). Bultink: "hidden Markov models to detect leakage using only the record of parity measurement outcomes". |
| 13 | [[4,2,2]] kept 65-79%; FT qubit 0.1-0.2% vs gauge 1.7-2.4% | partially supported | Takita is right: Table 2 acceptance is 0.757-0.790, and the paper cites the "exponentially decreasing acceptance probability as a function of the number of syndrome measurements". Linke Table 1 yields with a stabilizer run 65.2-81.6%, and 87.0-93.2% without one. The 0.1-0.2% vs 1.7-2.4% pair holds for the three \|00>_L rows only. The \|++>_L rows give 0.2-1.3% on the FT qubit against 3.9-4.5% on the gauge qubit. |
| 14 | Postselected RB decays are non-standard: HF2019 caveat, BDB quadratic | supported | From literature, HF2019: "There is at present no complete theory for fitting an RB decay curve with post-selection"; "5.8(2)% to 0.60(3)%". BDB: "The discard counts were 28/1600, 90/1600, and 56/800" (98.25%, 94.4% and 93.0% kept, verified analytically); "matches a quadratic fit instead of a linear one". |
| 15 | Steane Bell 0.05% vs 0.001%; [[12,2,4]] acceptance 70.9% to 40.3%; RA2021 2.7% without discards | supported | From literature, Paetznick Table 1: 0.50%, 0.05%, 0.001%; "pre-acceptance rate of about 75%", "goes from ≈75% to ≈72%". The Table 3 counts give 70.9/51.7/40.3% pre-acceptance and 98.8/96.9/89.0% post-acceptance. Those are the note's arithmetic, and mine matches. "the fit for each round of error correction points to 0.3% ± 0.1%". RA2021: "approximately 2.70(6)% per QEC cycle". A failed preparation is "reinitialized and the circuit runs again ... up to three iterations"; "Regardless of successful preparation, we proceed". |
| 16 | Correct one fault, reject two: d=4 accumulates ~25x less error than d=5 | supported | From literature, abstract: "if single faults can be corrected and double faults postselected against, logical errors only occur due to at least three faults"; "the distance-four surface code, using postselection, accumulates 25 times less error than its distance-five counterpart". Calling it the analogue of a d=2 hybrid is the author's reading. |
| 17 | Stim: H and S stabilizer maps; X detector after S is m_X(t+1)⊕m_X(t)⊕m_Z(t) | supported | Verified numerically: every map, with sign +1, and S maps X_L to +i·X_L·Z_L = +Y_L. The detector formula follows from S†·S_X·S = S_X·S_Z, and a flipped m_Z(t) then fires three detectors (verified analytically). It matches SST's weight-3 edge in a different frame. Preferring ML or HMM decoding over split-edge MWPM is conjecture. Bultink's HMMs track leakage, not Pauli errors, so the analogy is loose. |
| 18 | Five (2,3) Paulis mimic qubit-1-type errors and differ by a logical; ~p/3 per fold gate | supported | Verified numerically: all 15 non-identity Paulis on (2,3) are detected. The five pairs differ by X_L, Z_L, Y_L, Y_L and Y_L, and the other four have syndromes no single-qubit error gives. 5 × p/15 = p/3 (verified analytically). An ML decoder on the real DEM makes the same choice, because single-qubit errors are far likelier than p/15. So the loss is not limited to a naive decoder (conjecture). |
| 19 | Pure correction loses ~13p per round, clean exponential, likely worse than const=0 | partially supported | The arithmetic holds under its assumptions: 5 × 4p/3 = 20p/3 per qubit per round, and 0.5 × 4 × 20p/3 ≈ 13.3p (verified analytically). The cited papers support only the framing that d=2 hardware was run as detection and correction started at d=3. Krinner builds "on recent distance-two error detection experiments". "Clean exponential" and "worse than const=0" are untested. Ceasura et al. (section 5) show that a decoder can make LRB decays non-exponential. |
| 20 | 3x3 nearest-neighbour layout; SWAP(2,3) can be virtual; CZ(2,3) needs routing on a grid | supported | The geometry is verified analytically. With data on the corners and centre and ancillas on the edge midpoints, every check CNOT is nearest-neighbour. After H on all five qubits the checks become X on {0,1,3},{1,2,4} and Z on {0,1,2},{1,3,4}, so relabelling 2 and 3 restores the code, and S is unchanged by the relabelling. From literature: HF2019 virtual gates "implemented in software by relabeling the physical qubits"; BDB "qubit permutations are implemented via relabelling"; IQM "Star-24 QPU, where all qubits are coupled to a central resonator". A CZ through the Star resonator is conjecture, and the blog gives no fidelities. |
| 21 | ~40-55% kept per round means unif=1 at m=22 keeps ~1e-7; correction applied offline | supported | 0.5^23 ≈ 1.2e-7 (verified analytically). Across 0.40-0.55 per round the range is 7e-10 to 1e-6. From literature: Krinner applies corrections "in postprocessing"; RA2021 makes "software updates to the Pauli frame". That leakage is missing from the stim model is the note's own conjecture. |

## 3. Problems in the body that the key findings do not show

1. **Section 0, item 4 misapplies a quote.** The note says the rotated [[9,1,3]] "has a
   90-degree-rotation duality, which 'fixes a single physical qubit ...' [BB2024]". BB2024 say
   this about the unrotated code of their Fig. 1i ("Moussa's original construction on a surface
   code. The ZX-duality τ is reflecting along a diagonal"). The rotated code does have a
   90-degree ZX-duality that fixes its centre qubit. Chen2026 rely on it ("Transversal H gates
   ... followed by a π/2 patch rotation"). But that is my inference, not BB2024's statement.
   Suggested fix: "BB2024 make this point for the unrotated code; the same argument applies to
   the rotated code's 90-degree duality (inference)." F8 needs the same context.
2. **Section 0, item 3 overgeneralizes.** "The d=2 experiments report no logical H
   [Marques2022]" holds for Marques, whose gates were X_L, Z_L, X_L^{π/2} and T_L, and the word
   "Hadamard" never appears in its text. But Zhang et al., arXiv:2405.09035, which the author
   fetched and listed as unused, report "arbitrary single-qubit rotation gates ... realized
   through gate teleportation" on distance-2 surface codes. Limit the sentence to Andersen and
   Marques, or mention Zhang et al.
3. **F1 misquotes Moussa.** The source reads "both code families admit transversal
   implementations of single-qubit Clifford gates ... 2D^2-2D+1 data qubits per logical qubit".
   The note puts "single-qudit" inside the quotation marks and writes "data qudits per logical
   qudit". Restore the original wording or drop the quotation marks.
4. **F16 needs the decoder qualifier** (see KF9). Add "for their single-lom decoder, in the
   X-basis repeated-S experiment; the circuit distance stays d".
5. **F24 and the experiments table misstate Linke2017.** Give yields with a stabilizer as
   65-82%, and say the 0.1-0.2% vs 1.7-2.4% pair comes from the \|00>_L preparations.
6. **F26 misdescribes HF2019's H.** Their table maps physical "H⊗H⊗H⊗H" to logical
   "SWAP_12∘(H⊗H)". The SWAP is a logical swap of the two encoded qubits that comes with
   transversal H. It is not a physical SWAP applied after H. Physical swaps in HF2019 implement
   logical CNOTs ("SWAP 12 → CNOT 12").
7. **The table row for Marques2022 says "plus leakage rejection".** Marques ran no separate
   leakage-rejection step. Their detection-based postselection removed leaked runs ("the error
   detection scheme successfully post-selects leakage errors"). Drop the phrase.
8. **I6 claims more than it checked.** "The check schedule therefore keeps circuit distance 2"
   rests on ancilla hook errors and single data errors. Nobody ran a full circuit-distance
   search over the noisy round, so call it a conjecture or run one, for example
   `search_for_undetectable_logical_errors` in stim.
9. **I9's conjecture has a known counterpoint.** The claim that correcting or rejecting every
   round returns F(m) to a single exponential conflicts with Ceasura et al. 2022, which builds a
   noise model where "the presence of machinery associated with the implementation of quantum
   error correction can facilitate non-exponential decay". The syndrome qubits act as hidden
   memory through the decoder. This bears directly on research question 2.
10. **I10 overstates the trapped-ion comparison.** "Trapped-ion runs keep far more" holds for
    BDB2024, at 93-98% kept, but that run had detection only at the end. Paetznick's [[12,2,4]]
    pre-acceptance falls to 40.3% after three rounds, about 74% per round, and Linke's one-round
    yields are 65-82%. Neither is far above the 40-55% per round of superconducting d=2.
11. **F31 has a small wording slip.** The IQM blog says "Crystal-20 QPU with nearest neighbor
    coupling" and never calls Crystal a square lattice. The only "square lattice" in the post
    refers to the surface code.
12. **F17 quotes a sentence about a different gate.** The quoted sentence is about the S gate
    "in the non-orientable surface code", which "suffers from the same problem ... as in the
    case of the folded surface code". The point stands, but Sec. III.6 states it directly and is
    the better quote.
13. **Open question 5 can be partly settled.** SST2025's own text says "presenting our work at
    the APS March meeting 2025", so the talk exists. Its abstract content is still unverified.
14. **Open question 6 is resolved.** DOI 10.1038/s41586-026-10628-y is "Improved quantum
    processor logical error rates via correction and detection", Nature 654, 349-355 (2026),
    with 37 authors: all 32 arXiv authors plus B. W. Reichardt, D. Aasen, R. Chao, W. van Dam and
    Z. Wang. Its abstract covers "a 12-qubit code encoding two qubits inspired by Knill and a
    16-qubit tesseract colour code", reports "11× to 800×" gains, and does not mention the
    Steane code. Its reference list cites arXiv:2404.02280 as a separate preprint. It is a
    related paper, not the published version, so the Steane-code Bell numbers in F29 should
    keep citing the arXiv paper.

## 4. Re-run of the numerical claims (verified numerically)

Script: `scratchpad/scripts/check512.py` and `check512b.py`, stim 1.15. The full scratchpad
path is
`<local scratch, not kept>`.

- Single-qubit syndromes match CONTEXT.md. X0 and X3 give (1,0), X2 and X4 give (0,1), X1 gives
  (1,1). Z0 and Z2 give (1,0), Z3 and Z4 give (0,1), Z1 gives (1,1).
- Logical H, which is SWAP(2,3) then H on all five, swaps S_Z1 with S_X1, S_Z2 with S_X2 and
  X_L with Z_L, all with sign +1.
- Logical S fixes S_Z1, S_Z2 and Z_L. It maps S_X1 to S_X1·S_Z1 and S_X2 to S_X2·S_Z2 with sign
  +1, and X_L to +Y_L, where Y_L = i·X_L·Z_L.
- All 15 non-identity Paulis on (2,3) are detected. X2X3, Z2Z3, Y2Y3, X2Z3 and Z2X3 share the
  syndromes of X1, Z1, Y1, Y4 and Y0, and differ from them by X_L, Z_L, Y_L, Y_L and Y_L. X2Y3,
  Y2X3, Y2Z3 and Z2Y3 have syndromes no single-qubit error produces.
- A Z on a Z-check ancilla or an X on an X-check ancilla after the first CNOT spreads to Z1Z3,
  Z2Z4, X1X2 or X1X4. Each equals one single-qubit error times a stabilizer.
- The Zoo relabelling (0,1,2,3,4) to (0,3,4,1,2) maps the Zoo's four checks onto ours and
  S_0 S_2 S_3^† CZ_{1,4} onto S on {0,4}, S-dagger on 1, CZ(2,3). The Vasmer-Kubica relabelling
  (1,2,3,1̄,2̄) to (0,1,4,2,3) maps their four checks onto ours.

## 5. Sources the author may want

- Ceasura, Iyer, Wallman, Pashayan, "Non-Exponential Behaviour in Logical Randomized
  Benchmarking", arXiv:2212.05488 (2022). I fetched the abstract page and confirmed the title,
  authors and ID. It is the counterpoint to I9 and is relevant to research question 2. Lit
  review 02 may already cover it.
- Zhang et al., "Demonstrating a universal logical gate set in error-detecting surface codes on
  a superconducting quantum processor", arXiv:2405.09035. I fetched the abstract page. It is
  needed to qualify section 0, item 3.
- Nature 654, 349-355 (2026), DOI 10.1038/s41586-026-10628-y. I fetched the Nature page and
  the Crossref record. It is a correction-plus-detection result on trapped ions that could be
  cited next to Paetznick2024, not instead of it.
