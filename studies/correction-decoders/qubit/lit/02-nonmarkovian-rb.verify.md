# Verification of lit/02-nonmarkovian-rb.md

Independent citation and claim check, 2026-10-28. Target: `02-nonmarkovian-rb.md`
and its structured summary (25 citations, 18 key findings). I also spot-checked
document items K1-K39, the implications I1-I8, the open questions and the seed table.

## Method

- **Metadata.** For all 25 arXiv IDs I pulled the arXiv API record and the arXiv
  abstract page and compared title, authors, dates, journal reference and DOI. For all
  16 DOIs I pulled the Crossref record. I opened the APS abstract pages for
  PRX Quantum 2, 040351 and PRA 93, 022303 because the document's seed corrections
  depend on them. A Crossref title search checked whether the preprint-only entries
  have journal versions. None turned up.
- **Claims.** I read the full text of all 25 papers, not just abstracts. 21 came from
  the arXiv HTML rendering. Ziyad2025, Kwiatkowski2025 and Girling2025 have no HTML
  version, and the HTML of Ceasura2022 stops after Sec. III, so I read those four from
  their arXiv PDFs with a standard-library text extractor in the session scratchpad.
  I installed no packages. Quotes below come from those texts. Math appears as the
  LaTeX alttext or the PDF glyphs.
- **Disclosure.** My first batch of Crossref DOI lookups put the user's email address
  in the HTTP User-Agent header, following Crossref's "polite pool" convention. Nobody
  asked for that. Later requests did not include it.

Verdicts:

- **supported** means the source says it, in substance and scope.
- **partially supported** means the core is in the source, but a qualifier, number or
  attribution is wrong or overstated.
- **unsupported** means the source does not say it.
- **not checkable** means a conjecture that goes beyond every source. For these I
  checked only the parts attributed to sources, plus the arithmetic.

## Summary

- **No fabricated citations.** All 25 arXiv IDs resolve to the stated papers, and all
  16 DOIs resolve to the stated articles.
- **One wrong author list, introduced by the document itself.** The document
  "corrected" the seed for arXiv:2107.05403 by deleting "Harris". The published paper,
  PRX Quantum 2, 040351 (2021), lists five authors, with Robert J. Harris third. The
  APS page, the Crossref record (ORCID-authenticated) and the arXiv v2 full text all
  include him. Only the arXiv listing metadata shows four names. The seed was right.
- **One half-wrong seed correction.** The published PRA title of Ball2016 is "Effect
  of noise correlations on randomized benchmarking", with no "The". Only the arXiv
  title starts with "The". The seed matched the published version.
- **Key findings.** 12 supported, 4 partially supported (KF7, KF9, KF10, KF17),
  0 unsupported, 2 not checkable (the conjectures KF16 and KF18).
- **Errors outside the summary.**
  - K14 misattributes Epstein's Gaussian fit.
  - K36 adds a priority claim the source does not make.
  - K37 cites the wrong section.
  - I4 counts readout flips as Kumar's "accepted leakage", but in detection they cause
    rejection, not accepted leakage.

## 1. Citation metadata

| Key | arXiv | Record vs. document | metadata_ok | Notes |
|---|---|---|---|---|
| Wallman2018 | 1703.09835 | Title, sole author and v4 (2018-01-25) match. Quantum 2, 47, issued 2018-01-29, doi:10.22331/q-2018-01-29-47. | yes | |
| Proctor2017 | 1702.01853 | 5 authors match. PRL 119, 130502 (2017). | yes | Published title is in title case. |
| Helsen2022 | 2010.07974 | 5 authors match. PRX Quantum 3, 020357 (2022). | yes | Published title is "General Framework for Randomized Benchmarking", without the "A". The arXiv title used here is fine. |
| Epstein2014 | 1308.2928 | Match. PRA 89, 062321 (2014). | yes | |
| Ball2016 | 1504.05307 | Match to the arXiv title. PRA 93, 022303 (2016). | yes | The PRA title has no "The" (APS page, Crossref). See section 4. |
| Mavadia2018 | 1706.03787 | 7 authors match. npj Quantum Inf. 4, 7 (2018). | yes | |
| FongMerkel2017 | 1703.09747 | Match. No journal reference on arXiv. Crossref title search finds no journal version. | yes | |
| Fogarty2015 | 1502.05119 | 7 authors match. PRA 92, 022326 (2015). | yes | Published title: "Nonexponential fidelity decay ...". |
| Srivastava2026 | 2510.13051 | 6 authors and v2 (2026-05-22) match. PRR 8, 023258 (2026), doi:10.1103/r1z5-ycqc. | yes | Published title: "Blind spots of randomized benchmarking under temporal correlations". |
| Pollock2018 | 1512.00589 | 5 authors match. PRA 97, 012127 (2018). | yes | Published spelling: "characterization". |
| FigueroaRomero2021 | 2107.05403 | The arXiv listing has 4 authors. The published paper and the arXiv v2 full text have 5. | **no** | Correct authors: Pedro Figueroa-Romero, Kavan Modi, Robert J. Harris, Thomas M. Stace, Min-Hsiu Hsieh. PRX Quantum 2, 040351 (2021), doi:10.1103/PRXQuantum.2.040351. |
| FigueroaRomero2022 | 2202.11338 | Match. Quantum 6, 868 (2022). | yes | |
| FigueroaRomero2024 | 2305.04704 | 6 authors match. Quantum Sci. Technol. 9, 035020 (2024). | yes | |
| WoodGambetta2018 | 1704.03081 | Match. PRA 97, 032306 (2018). | yes | |
| Wallman2016 | 1412.4126 | The arXiv title is "Characterization of Leakage Errors via Randomized Benchmarking". The NJP 18, 043021 (2016) title is "Robust characterization of leakage errors". | yes | The document gives both titles correctly. |
| ChenBaldwin2025 | 2502.00154 | Match. PRR 7, 043065 (2025), doi:10.1103/j8sz-6ws9. | yes | |
| Combes2017 | 1702.03688 | 4 authors match. Preprint only. | yes | |
| HarperFlammia2019 | 1806.02359 | Match. PRL 122, 080504 (2019). | yes | |
| Kabir2025 | 2511.12777 | 8 authors and v2 (2026-05-12) match. Preprint. | yes | |
| Ceasura2022 | 2212.05488 | 4 authors match. v1 only, preprint. | yes | |
| Ziyad2025 | 2512.08893 | 3 authors and v3 (2026-02-19) match. Preprint. | yes | Sandia OSTI reports with similar titles exist. They are not journal versions. |
| Kwiatkowski2025 | 2509.16887 | 6 authors match. Preprint. | yes | |
| Kumar2026 | 2609.12073 | 3 authors and v1 (2026-09-10) match. | yes | The arXiv comment confirms "Extended version of the paper accepted at IEEE QCE 2026". |
| Endo2025 | 2510.20224 | 3 authors and v2 (2025-12-19) match. | yes | |
| Girling2025 | 2508.08188 | 3 authors match. | yes | |

The structured summary drops some diacritics (Sorée, Rodríguez-Rosario, Papič, Inés).
That is cosmetic.

## 2. Key findings in the structured summary

**KF1 (Ziyad2025): supported.** The abstract says "non-Markovianity can emerge from
Markovian physical operations if (and only if) the physical qubits are not necessarily
returned to the code subspace after every round of QEC. In this situation, the
syndrome qubits can act as a memory". Sec. 3 says the logical dynamics are "(in
general) described by a hidden Markov process". One precision point. The paper proves
the "only if" direction (Thm. C.1). For the "if" direction it says only "can emerge"
and shows examples. The summary's "this happens iff" drops the "can".

**KF2 (Ziyad2025): supported.**

- Sec. 2.1 says "Gate composability always holds if each gate in the gate set is a
  noisy gadget followed by a perfect (error-free) QEC gadget".
- In Fig. 2, an ancilla bit flip "leads to an incorrect readout of the syndrome,
  causing the decoder pick the wrong correction".
- In Sec. 3.2.1, the per-round error-rate changes "die off exponentially".
- The conclusion says "we expect fault-tolerant syndrome extraction to reduce (but not
  eliminate) non-Markovianity", which the authors offer as an expectation.

The paper is not consistent with itself on persistence. The conclusion says
non-Markovianity "disappears rapidly as multiple rounds of error correction are
performed". Sec. 3.2.1 says it "persists indefinitely". K31 quotes the second phrase,
so it is defensible, but readers should know both phrases are in the paper.

**KF3 (Ceasura2022): supported.**

- In Example 2, the survival probability oscillates "between 0 and 1, with a period of
  two".
- The Fig. 4 caption reads "a 2% probability of resetting each of the qubits in
  register B ... One can see the amplitude of the oscillations decrease".
- In Sec. V and Fig. 5, an embedded classical repetition code keeps the oscillations
  alive at 95% reset probability.

The authors read Fig. 4 as showing "some robustness to syndrome reset". "Damps" is
accurate, but the effect does not vanish at a 2% reset rate.

**KF4 (Kwiatkowski2025): supported.**

- Prop. 1 gives a Markov process γ on the syndrome register S and logical Pauli
  channels Λ_{s_out,s_in}.
- The introduction lists condition (ii), syndrome decoding symmetry (Def. 11) or
  syndrome-state randomization (Def. 13). It lists condition (iii) as
  "f_1 ≥ 1 − 1/64" and adds "The first two assumptions can be enforced via Pauli
  twirling".
- The abstract says deviations are "exponentially suppressed in the number of QEC
  cycles".

f_1 is a single-cycle logical entanglement fidelity, averaged over the syndrome
marginal, and the condition is sufficient only. The document says so in I5.

**KF5 (Kumar2026): supported.**

- Sec. 6 says "The denominator depends on the current state, making this a nonlinear
  map".
- Sec. 6, after Corollary 6.3: "QED-check imperfections produce false negatives".
- Same passage: "only QED-check noise contributes to ℓ at first order".
- Sec. 6, "Obstruction to a stationary logical-only model": "no such stationary
  logical-only map reproduces these diagnostics through first order".
- Remark E.5 splits the spectrum into "an (L+1)-eigenvalue cluster near 1 (slow
  accepted-shot decay modes) and an X-eigenvalue cluster near 0 (fast transients
  associated with leakage rejection)".
- The abstract mentions the "first-cycle transient".

**KF6 (HarperFlammia2019, Kabir2025): supported.** Harper and Flammia write in their
Discussion: "There is at present no complete theory for fitting an RB decay curve with
post-selection." Kabir et al. write that "terminal detection can produce decay curves
with non-Markovian structure that are not always well summarized by a single
exponential fit". That sentence is in Sec. VIII, the conclusion. The LRB-D definition
is in Sec. VII-A.

**KF7 (Combes2017): partially supported.** The list of conditions is accurate. Sec. 1
says "we require the noise to be time-independent, Markovian, and gate-independent
(all at the level of logical Clifford gates) ... we must also assume that syndrome
measurement errors are uncorrelated between rounds". Protocol step ii says "a round of
error correction (with or without recovery) and a refresh of the ancilla register".

The "only if" overstates it. These are the assumptions under which Combes et al. prove
the reduction to zeroth-order RB, not necessary conditions. Wallman2018 and Helsen2022
show exponential decays under gate-dependent Markovian noise. "Tracked recovery" is
the document's reading of "without recovery". Combes' step iii also applies one
compiled inverse, unlike the repo's mirror inversion, which the document flags in O2.

**KF8 (Wallman2018): supported.**

- Thm. 4 gives "Ap^{m}+Bt^{m}+ε_m" with "|ε_m| ≤ δ_1 δ_2^m" for any unitary two-design.
- The paper says "we only need δ_2 ⪅ 1/2 in order for the decay to be essentially a
  single exponential for m ≥ 10".
- The theorems are proved "without restricting to trace-preserving noise", with RB on
  encoded qubits in mind.

The limit is one exponential plus a constant only when t = 1. With trace loss it is
A p^m + B t^m, which is two decays.

**KF9 (Helsen2022, Ceasura2022): partially supported.** Ceasura says the output is "a
linear combination of exponential decays provided that the representation is
multiplicity free and that the noise is time and gate independent". Helsen gives
"realistic conditions under which we can rigorously guarantee that the output ... is
well-described by a linear combination of matrix exponential decays". The oscillation
part is supported by Ceasura and by Helsen's "complex eigenvalue pairs, which will
lead to damped-oscillation behaviour".

Two overstatements remain:

- "Only for" turns a sufficient condition into a necessary one. Helsen's Thm. 8 ("RB
  gives exponential decays under broad (Markovian) circumstances") and Wallman2018
  cover gate-dependent noise.
- "In general" is too strong. Helsen's guarantee needs Markovian, time-independent
  noise, and FR2021 says the non-Markovian ASF "is not a simple function of sequence
  length anymore".

**KF10 (FongMerkel2017, Ball2016, Mavadia2018): partially supported.**

- **Exponential to power law: supported.** This is in the FongMerkel abstract.
- **"Factors of 2-6": generalized from one example.** FongMerkel attach the number to
  one quasistatic-noise example: "All curves shown here underestimate the average
  gate error rate by factors of two to six; overestimation is also possible if only
  short RB sequence lengths are used for fitting." The summary applies it to all
  temporally correlated noise.
- **Gamma distribution: not caused by correlations.** Ball derives a gamma
  distribution "for both Markovian and correlated errors". Correlations change the
  shape and scale. Long correlations add "large nonvanishing variance and skew".
- **Mavadia: supported.** The paper reports "approximately Gaussian for rapidly
  varying noise, and a broad, highly skewed distribution for the slowly varying case".

**KF11 (Fogarty2015, FigueroaRomero2021, Ziyad2025, Kumar2026): supported.**

- Fogarty's Table 1 runs an AIC comparison of Ap^m + Aq^m against Ap^m + B.
- Ziyad (Sec. 3.2.1) defines a per-round rate (1 − q_m/q_{m+1})/2 and calls
  "deviation of the decay from a straight line" the signature.
- FR2021 gives the Schatten q-norm measure against a Markovianized ASF (Eq. 15) and
  memory-length estimation with identity gates (Secs. VI and VIII). It also proposes
  identity interleaving as a coherence test (Fig. 7).
- Kumar writes "ℓ = 2p_acc(1) − p_acc(2) − 1 + O(ε²) can be inferred from one- and
  two-cycle acceptance probabilities". The summary's sign convention matches.

**KF12 (Epstein2014, WoodGambetta2018, ChenBaldwin2025, Wallman2016): supported.**

- Epstein (Sec. I) says non-Markovian noise "is history-dependent, so the noise at one
  moment may depend on previous gates".
- Wood-Gambetta gives Eq. 9 (A + Bλ_1^m), Eq. 14 (λ_1 = 1 − L_1 − L_2) and Eq. 15
  (A_0 + B_0λ_1^m + C_0λ_2^m). Sec. III.2 attributes oscillations to an insufficiently
  twirled leakage subspace.
- Chen-Baldwin describe leakage post-selection: "isolate λ = t − r, and by fitting the
  data retention rate ... isolate t".

Strictly, the "single-exponential" computational population is an exponential plus a
constant.

**KF13 (Srivastava2026): supported.** I read the full text, not just the abstract. The
paper identifies "classes of interaction Hamiltonians that render temporal
correlations completely invisible to RB". It adds "while classical correlations may
remain undetectable in the ASF data, they can nonetheless significantly affect
worst-case errors quantified by the diamond norm". It also shows correlations "may
suppress worst-case errors", so the hidden effect can go either way. Open question 6
assumes the effect "raises" error.

**KF14 (FigueroaRomero2024): supported.** I read the full text. The abstract says "fast
DD reduces non-Markovian RB to an exponential decay plus longer-time corrections ...
Pauli-twirling generally does not affect the average, but iii) it always suppresses
the variance". The tool is Pauli twirling, implemented through randomized compiling.
The conclusion softens "always suppresses" to "decrease, or at worst leave unchanged,
the variance".

**KF15 (I1, analytic): supported, verified analytically.**

- Every check in these circuits measures a code stabilizer.
- Logical Cliffords map stabilizers to stabilizers, so each check's noiseless outcome
  is +1 for every code state.
- SI1000 channels are mixtures of Paulis. A Pauli fault flips a deterministic outcome
  exactly when its propagated frame anticommutes with the measured operator. That
  does not depend on the logical state.
- The same argument covers the const=0 X-basis readout.

Kwiatkowski Sec. 4.3 states the general case: "the transformation of the joint state
of L and S ... is independent of ρ_L". Acceptance can still depend on which Clifford
sequence ran, because gate noise differs between gates.

**KF16 (conjecture on const=1 as leakage RB): not checkable.** The attributed parts are
accurate:

- Ziyad's condition is quoted correctly.
- Kumar uses "leakage" for non-code-space sectors: "some population leaks to
  non-code-space sectors while remaining accepted".
- Wallman2016's abstract names leakage out of "the logical code space defined by some
  quantum error-correcting code". That supports the analogy better than the citations
  the document uses.

Two analytic checks I did:

1. The CONTEXT syndrome table and the weight-2 logicals hold.
   - X0 and X3 give (1,0), X2 and X4 give (0,1), X1 gives (1,1).
   - Z0 and Z2 give (1,0), Z3 and Z4 give (0,1), Z1 gives (1,1).
   - X0X3·X2X4 = S_X1·S_X2 and Z0Z2·Z3Z4 = S_Z1·S_Z2.
2. Suppose the accumulated Pauli frame mixes to uniform over the 4^5 Paulis. Then the
   syndrome is uniform over 16 values, so acceptance tends to 1/16. Given acceptance,
   the logical class is uniform over {I, X_L, Y_L, Z_L}, so ⟨X_L⟩ tends to 0. Both
   match the conjecture.

One caveat on the "seepage" picture. An X-type pair (X0 then X3) gives X_L, which does
not flip the X_L readout at that point. "About half the time" holds for producing a
logical operator, not for flipping the readout.

**KF17 (conjecture on unif=1): partially supported.** Kumar does support first-order
accepted leakage from check imperfections, meaning false negatives. Data faults in a
later measurement or reset layer of the same round fit that mechanism. So do CNOT
faults inside a check that spread to data qubits (hook errors), which the document
leaves out.

Readout flips do not fit. In detection, a single readout flip on a clean state gives a
non-zero outcome, and the shot is rejected. That is a false positive. It counts toward
rejection, not toward Kumar's ℓ = ‖(ΔA)_{XL} e_ψ‖, which is accepted leakage. A
readout flip hides a real data error only together with that error, which is second
order. So Kumar does not support "a first-round transient driven by readout flips
(5p)". Readout flips matter for correction (Ziyad's mechanism, KF18), not for unif=1
detection.

**KF18 (conjecture on d=2 correction): not checkable.** The cited facts are accurate.
Ziyad's Fig. 2 shows the readout-error mechanism, and Kwiatkowski's f_1 ≥ 1 − 1/64 is
sufficient only.

The arithmetic mostly holds:

- Each DEPOLARIZE1(2p) layer gives a Z-type error with probability 4p/3. Five layers
  give 6.7p per qubit per round, and the same for X-type.
- Four qubits sit in ambiguous pairs, and the decoder mis-corrects half of their
  errors. That gives about 14p of Z_L and 14p of X_L per round, so 25-30p of logical
  error in total.
- 1/64 = 0.0156, so the condition fails above p ≈ 5.2e-4 (at 30p) to 6.3e-4 (at 25p).

The quoted range "5e-4 to 1e-3" has a loose upper end. The estimate supports about
5e-4 to 6.5e-4. On the repo's p grid, the sufficient condition holds only for
p ≤ 5e-4.

## 3. Document items K1-K39 outside the summary

I checked every K item against full text. These are fine as written: K1-K13, K15,
K16, K18-K25, K27-K35, K38, K39. The rest need edits.

- **K14: wrong parenthetical.** In Epstein's Fig. 5, the Gaussian model
  F = 0.495(0.999)^{m^2} + 0.503 is fitted to the identity-gate sequences, which have
  no randomization. The random-Clifford RB data are fitted with the standard
  exponential F = 0.486(0.998)^m + 0.510. The text says random Cliffords
  "dramatically slowed" the Gaussian decay. "Not completely accurate" refers to the
  RB confidence interval saturating as K grows (Fig. 7). Delete "(they fit a
  Gaussian-in-m form instead)".
- **K17: wording.** Fogarty says "the two fidelity model better explains the data,
  significantly so on all but one of the datasets". So the two-rate model wins on
  every dataset and is significant on all but one.
- **K26: same "only if" problem as KF7.**
- **K36: priority claim.** HarperFlammia2019 never claims to be "the first" logical-RB
  experiment with an error-detecting code. Drop "The first".
- **K37: section.** The definition is in Sec. VII-A. The warning about non-Markovian
  structure is in Sec. VIII.
- **K31: fine, with a note.** The paper's conclusion says "disappears rapidly", next to
  Sec. 3.2.1's "persists indefinitely".

## 4. Seed corrections

- **arXiv:2107.05403: the document's correction is wrong.** Restore Robert J. Harris as
  third author. The published paper and the arXiv v2 full text both list him.
- **arXiv:1504.05307: half-wrong.** "The effect of ..." is the arXiv title. The PRA
  title is "Effect of noise correlations on randomized benchmarking".
- **All other seed rows are correct.** That includes the companion paper arXiv:1801.09811,
  "Operational Markov condition for quantum processes", PRL 120, 040405 (2018).
- **All 12 "found but not cited" IDs exist and match the named authors.** One of them,
  Kam et al. arXiv:2410.23779, "Detrimental non-Markovian errors for surface code
  memory", QST 10, 035060 (2025), looks relevant enough to Q2 to reconsider.

## 5. Implications and open questions

- I1 holds. See KF15.
- The analytic parts of I2 and I3 are consistent. See KF16 for the syndrome-table check.
- The readout-flip attribution in I4 is wrong. See KF17.
- The I5 arithmetic holds, apart from the loose upper bound. See KF18.
- I7.3 is a looser test than Kumar's. Kumar's diagnostic combines one- and two-cycle
  acceptance with p_acc(0) = 1. A constant A(m+2)/A(m) is a weaker stationarity test,
  which is fine for a conjecture.
- I8 is fine. "Syndrome randomization" matches Kwiatkowski's Def. 13.
- O6: Srivastava shows hidden correlations can lower worst-case error as well as raise
  it.

## 6. Suggested edits to 02-nonmarkovian-rb.md

1. Add Robert J. Harris as third author of FigueroaRomero2021, in the bibliography and
   in the seed table.
2. Fix the Ball2016 seed row. The published title has no "The".
3. In K14, delete the Gaussian-fit parenthetical.
4. In K17, say the two-rate model was favored on all datasets and significantly on all
   but one.
5. In K26 and KF7, replace "only if" with "under the assumptions that".
6. In KF9, change "only for" to "provided that". Limit "matrix-exponential decays" to
   Markovian, time-independent noise.
7. In KF10, say "in a quasistatic-noise example, exponential fits underestimated the
   error by factors of 2-6". Note that Ball's gamma form also holds for Markovian
   noise.
8. In K36, drop "The first".
9. In K37, cite Sec. VII-A for the definition and Sec. VIII for the warning.
10. In I4 and KF17, remove readout flips from the accepted-leakage sources. Add CNOT
    hook faults within a check.
11. In I5 and KF18, give the threshold range as about 5e-4 to 6.5e-4.
12. Optionally, in I2, cite Wallman2016's code-space leakage framing.
