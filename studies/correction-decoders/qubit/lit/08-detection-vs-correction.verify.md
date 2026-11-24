# Verification of lit note 08: detection versus correction

Independent check of `08-detection-vs-correction.md`, written 2026-10-28. I ran no
simulations and wrote no other file. Labels follow `../CONTEXT.md`. Where I add my own
reasoning I label it "verified analytically" or "conjecture".

## Method

Metadata. For all 23 cited arXiv IDs I fetched the arXiv abstract page and the arXiv API
record and compared title, authors, first-version date and identifier. I fetched the
Crossref record of every DOI in the bibliography, the Springer page of Reichardt's
Algorithmica paper, and the nature.com page of the 2026 Nature paper.

Claims. I read every cited passage in the full text. Recent papers came from the arXiv HTML
version. Gottesman 2016, Knill's schemes paper, Aliferis-Gottesman-Preskill, Prabhu-Reichardt
and Willsch et al. came from ar5iv. Where ar5iv failed or was truncated I read the arXiv LaTeX
source instead. That was the case for Knill 2005, Knill's threshold analysis, Harper-Flammia,
Vuillot and Ceasura et al. For Reichardt I compared the thesis abstract on arXiv with the
journal abstract on Springer.

## Bottom line

- All 23 citations are real. Titles, authors, years and identifiers are right. I found no
  invented paper, no wrong arXiv ID and no wrong author list. The five extra papers in the
  search log are real and correctly described too.
- 28 of the 29 literature findings in the structured summary are supported as written, and the
  quoted phrases match the sources word for word.
- F3 is only partly supported. It joins two separate remarks of Gottesman with a "so" that
  he does not make, and his stated cause is different.
- I7 mislabels the Z-fault pairs. {0,3} and {2,4} share a syndrome for X faults only. For Z
  faults the pairs are {0,2} and {3,4}, as CONTEXT.md lists.
- I7 and I8 miss a first-order term. An ambiguous fault whose pair logical commutes with the
  final X_L readout in its frame contributes +1 to the decoded fidelity, not 0, whatever the
  decoder guesses. Section 3 below gives the corrected statement.
- Section 0 says correct-plus-reject makes "the rejection rate second order in p". Only the
  post-rejection rate is. Pre-rejection is first order and is the larger part in both
  trapped-ion papers.
- The note's guess that Nature 654, 349-355 (2026) is the journal version of both preprints
  holds up better than the note says. The Steane-code numbers in F20 are probably not in the
  journal version.

## 1. Citation metadata

| Key | Identifiers checked | Result | Note |
|---|---|---|---|
| Gottesman2016 | arXiv:1610.03507 | correct | D. Gottesman, v1 2016-10-11, no journal reference |
| Knill2004a | quant-ph/0402171 | correct | E. Knill, 2004-02-23, no journal reference |
| Knill2004b | quant-ph/0404104 | correct | E. Knill, 2004-04-19, no journal reference |
| Knill2005 | quant-ph/0410199; 10.1038/nature03350 | correct | arXiv title "Quantum Computing with Very Noisy Devices", 2004-10-25. Crossref gives "Quantum computing with realistically noisy devices", Nature 434(7029), 39-44, March 2005 |
| AGP2007 | quant-ph/0703264 | correct | three authors as listed, 2007-03-28, arXiv journal reference Quant. Inf. Comput. 8 (2008) 181-244 |
| Reichardt2009 | quant-ph/0612004; 10.1007/s00453-007-9069-7 | correct, one wording fix | The arXiv record is the 195-page Berkeley PhD thesis of 2006-11-30 under the other title. The journal paper is Algorithmica 55(3), 517-556, online 2007-10-17, print November 2009. An arXiv search on author and title finds no separate preprint of it. The two abstracts are near-identical, not identical: the journal version drops the thesis's first sentence and adds "the distribution of errors in the quantum state has correlations that conceivably could grow out of control" |
| English2024 | 2410.07598; 10.1103/nh49-52y2 | correct | PRL 135, 120603, 2025-09-18. The journal title writes "Postselected" without a hyphen |
| Chen2025 | 2510.05222 | correct | six authors as listed, v1 2025-10-06, v3 2026-05-19, no journal reference |
| Prabhu2021 | 2112.03785; 10.1103/PhysRevA.110.012419 | correct | PRA 110, 012419, 2024-07-08 |
| Smith2024 | 2405.03766; 10.1038/s42005-024-01883-4 | correct | Commun. Phys. 7, 386, 2024-11-28 |
| Paetznick2024 | 2404.02280 | correct | 32 authors in v3, first three as listed. See the Nature note below |
| ReichardtTess2024 | 2409.04628 | correct | 18 authors, first three as listed |
| Self2022 | 2211.06703; 10.1038/s41567-023-02282-2 | correct | Nat. Phys. 20(2), 219-224, online 2024-01-05 |
| He2024 | 2409.12104; 10.1038/s42005-025-02136-8 | correct | Commun. Phys. 8, 217, 2025-05-24 |
| Jin2025 | 2504.21172 | correct | eight authors as listed, v2 2026-04-29, no journal reference |
| Kumar2026 | 2604.19871 | correct | six authors as listed, 2026-04-21 |
| Vuillot2017 | 1705.08957 | correct | The title has a space before "?". QIC 18(11&12), 949-964. The DOI on the arXiv record, 10.26421/QIC18.11-12, is the issue DOI |
| Willsch2018 | 1805.05227; 10.1103/PhysRevA.98.052348 | correct | five authors as listed |
| Harper2018 | 1806.02359; 10.1103/PhysRevLett.122.080504 | correct | PRL 122, 080504, 2019-02-26 |
| Gupta2023 | 2305.13581; 10.1038/s41586-023-06846-3 | correct | 13 authors, last B. J. Brown, Nature 625(7994), 259-263 |
| Google2024 | 2408.13687; 10.1038/s41586-024-08449-y | correct | 249 authors on arXiv, first R. Acharya, Nature 638(8052), 920-926 |
| Ceasura2022 | 2212.05488 | correct | four authors as listed, no journal reference |
| Kabir2025 | 2511.12777 | correct | eight authors as listed, v2 2026-05-12 |

Nature 2026 paper. Crossref and nature.com both give "Improved quantum processor logical error
rates via correction and detection", Nature 654(8118), 349-355, published 2026-06-10, DOI
10.1038/s41586-026-10628-y, received 2025-02-14, 37 authors. Four facts support the note's
inference.

- The 37 names are exactly the union of the 32 Paetznick2024 authors and the 18 tesseract
  authors, who share 13 names.
- The abstract names "a 12-qubit code encoding two qubits inspired by Knill and a 16-qubit
  tesseract colour code encoding four qubits".
- The author contributions say Paetznick, da Silva and Bello-Rivas analysed the carbon code
  experiments and Reichardt the tesseract experiments.
- The paper cites arXiv:2404.02280 as a preprint in its own reference list.

The abstract quotes improvements "ranging from 11x to 800x", the figure list has only
carbon-code and tesseract-code figures, and "Steane" appears only in the reference list. So
the [[7,1,3]] Bell-state result in F20, with gains 9.8 and 500, is probably only in the arXiv
preprint. I could not read the paywalled body to confirm this.

Search-log extras. Combes et al. 1702.03688, Bombin et al. 2212.00813 with PRX Quantum 5,
010302, Bluvstein et al. 2312.03982 with DOI 10.1038/s41586-023-06927-3, Linke et al.
1611.06946 with Sci. Adv. 3, e1701074, and Kasatkin et al. 2602.19042 all exist with the
titles and authors the note gives.

## 2. Literature findings F1 to F36

| ID | Verdict | Evidence from the source |
|---|---|---|
| F1 | supported | Gottesman Sec. II: "the encoded circuit must have a lower error rate for all circuits in the family of circuits of interest" and "it is important to implement the original circuit using the best qubit(s) available in the system". Encodings act on "complete quantum circuits" |
| F2 | supported | Sec. III: "it can only improve error rates via post-selection. Demonstrations that rely on post-selection can be controversial". Supp. A.1: "The unencoded circuit is never rejected, so to make a fair comparison ... we should compare p_L/p_S ... with p_U" |
| F3 | partly supported | Both facts are in the paper, the causal link is not. The quoted sentence is in Supp. A.2 and describes the cost of the repeated-detection Bacon-Shor variant, which "allows much longer circuits with a relatively low logical error rate". The falling threshold, 1.6% at T=1 down to 0.12% at T=100, comes from Supp. A.1, where p_L grows as T^2 p^2 against p_U as T p, and Gottesman says it "reflects the lack of repeated error correction or detection steps to keep the build-up of errors under control". Acceptance is a minor factor there |
| F4 | supported | Google Supp.: gain "over the best physical qubit in the same system". Gupta: "higher than we can achieve with any pair of physical qubits". Vuillot: "we tried out the six different connected pairs". Paetznick Table 1: "'gain' to be the error rate of the physical circuits divided by the error rate of the corresponding logical circuit" |
| F5 | supported | Abstract: "exceeding its best physical qubit's lifetime by a factor of 2.4 +/- 0.3". Supp. VI.2: eps_d = 0.5(1 - (1 - 2 p_L)^(1/t)) |
| F6 | supported | Feed-forward logical infidelity (1.87 +/- 0.16)e-2. Fig. 3 caption: best pair (2.354 +/- 0.271)e-2, average over 28 pairs about 6.2e-2 |
| F7 | supported | "The optimal acceptance rate assuming no noise is 75%, 37.5% and 25%" for feed-forward, post-selection and standard schemes. "14.9 +/- 0.1 %" belongs to the post-selection variant under physical tomography. The break-even statement compares fidelities only |
| F8 | supported | Table of average performance: FTv1 0.0451 at post-selection ratio 0.647, Bare[2-0] 0.0550 at 1.0. "for some circuits the bare version is always better", "only on average over 20 different states". The 4.51e-2 is the FTv1 variant of the logical 00 preparation |
| F9 | supported | "the gate may fail. If it fails, this is known. The probability of success must be non-zero." "Purely postselected computing has little computational power, but we can use it to prepare states needed to enable scalable quantum computing" |
| F10 | supported | Abstract: "it is expected that the encoded computation can be made to be arbitrarily accurate. Although the probability of success of the encoded computation decreases dramatically with accuracy". Knill hedges with "it is expected" |
| F11 | supported | "This works because error-detecting codes can always correct an error at a known location." A level-1 C4 detection marks a pair and the level-2 C6 syndrome corrects it |
| F12 | supported | Accept-if-agree conditional error "4 gamma/15 + O(gamma^2)". "majority decoding of the three measurement outcomes. However, the error probability is now 4 gamma/5 + O(gamma^2)", with e_c = gamma |
| F13 | supported | Knill2004b abstract: "if classical and quantum communication delays are negligible ... errors above 1% per elementary quantum gate". Knill2005: "possible at gamma = 0.03 and perhaps up to gamma = 0.05. Although the postselection overheads are extreme". Precision point: the 3-5% figure is for postselected C4/C6 state preparation feeding teleported correction with a second code C_e. The error-correcting C4/C6 architecture alone is credited with "EPGs above 0.01" |
| F14 | supported | Abstract matches word for word, including "not to strongly-correlated adversarial noise" and 1.04e-3 |
| F15 | supported | Both abstracts: "appear to tolerate as much as 3-6% noise per gate" and "Numerical calculations indicate that the actual noise threshold this method yields is lower-bounded by 0.1% noise per gate" |
| F16 | supported | "four distinct thermodynamic phases". "The existence of an abort threshold is an important necessary condition for post-selected QEC to be scalable." At full post-selection "the decoding strategy is simply to do nothing" |
| F17 | supported | Intro: "if the abort rate per measurement is finite, postselection is not scalable". Abstract: b = 3.1(1) for "the toric code with perfect syndrome measurements". "postselection with finite acceptance probability does not alter the error threshold" |
| F18 | supported | Abstract. The spacetime-volume costing with repetitions is done for the 15-to-1 distillation example. The zero-tolerance decoder's abort probability grows with distance, so the falling exclusion rate needs the modest criterion F18 names |
| F19 | supported | Abstract word for word |
| F20 | supported | Sec. III.1.2 and Table 1 of arXiv v3: pre-acceptance about 75%, detection takes it to about 72%, error 0.50%, 0.05%, 0.001%. Probably arXiv-only, see the Nature note |
| F21 | supported | Table 2: 0.17% to 0.001%, 15,483 to 15,409 accepted, "the post-rejection rate is roughly 0.5%". Sec. III.2.1 has the quoted sentence |
| F22 | supported | "Although post-selection is not scalable in a strict sense, with low rejection rates one can run relatively deep circuits". The unencoded baselines reject nothing |
| F23 | supported | Sec. II: "We cannot reject these first-order events or the rejection rate would be too high", about column faults that flags then let them correct. Two-two row outcomes with no flag are rejected. Table 3 acceptance runs from 50(2)% to 90(1)%, with errors counted among accepted trials |
| F24 | supported | "In the first 10 error correction rounds, the logical error rate increases by 2.1(1)e-4 per round ... and 2.02(2)% of the surviving trials are discarded." This is an emulator run on one block, Fig. 13 |
| F25 | supported | "an overhead in the number of circuit repetitions". "logical quantum volume of 2^8 with a discard rate of 86.8 +/- 0.7%". "adding syndrome measurements increases the HOF" |
| F26 | supported | Fig. 2: "a parameterised physical unitary and its gate-by-gate inverse", survival probability with discard bars. Their rotations have arbitrary angles, so these are not Clifford circuits |
| F27 | supported | Sec. V.2: F_ice = H/(1-D). Fig. 3: red lines where Iceberg fidelity "equals that of unencoded circuits", cyan lines at "a 10% post-selection rate" |
| F28 | supported | "the QAOA logical fidelity stabilizes as the number of syndrome measurements increases, even though the post-selection rate ... decreases". Intro: "While non-scalable, they can still deliver improved algorithmic performance" |
| F29 | supported | arXiv abstract: success probability 44% to 65%, post-selection rate 4% to 33% at 22 algorithmic qubits on H2-1, better than unencoded up to 34 |
| F30 | supported | Eq. 1, (eps_p - eps_l)/p_rej > 1/4. Sec. VI: MSE = bias^2 + SE^2 at 1e5 attempted shots. Kumar reserves "break-even QED" for eps_l < eps_p and calls Eq. 1 an efficiency or sampling-advantage condition, though the abstract also says "break-even" |
| F31 | supported | "the minimum interval (L=1) does not achieve sampling advantage for any code in our evaluation". (L_thresh, L_max) is (4, 345) for [[6,4,2]] and (2, 208) for [[12,10,2]]. "The [[9,1,3]] code does not achieve advantage at any interval" |
| F32 | supported | Sec. IV: "the initial syndrome extraction only injects leakage without removing any, creating an O(p) transient population". Fast modes with eigenvalues O(p) "vanish within O(1) rounds", slow modes have 1 - O(p). Here "leakage" means states outside the codespace that pass the current check |
| F33 | supported | "There is at present no complete theory for fitting an RB decay curve with post-selection." Infidelity 5.8(2)% to 0.60(3)%. SPAM keeps the device from being "below the fault-tolerance threshold" |
| F34 | supported | Abstract. The construction is adversarial: the authors choose a Clifford representation that writes the gate history into syndrome qubits, need n > 3k and a surjective decoder, and either let syndrome reset fail or guard the memory with a repetition code |
| F35 | supported | All three quoted phrases are in the LRB-D section. The distance-2 ambiguity is one of several reasons given for detection, next to hardware size and real-time decoding |
| F36 | supported | Abstract and Secs. III-IV. "the entangling two-qubit gate is the most sensitive gate". Passing needs the detuned gate set. With the first gate set the criterion fails, and with the second it fails for one circuit until measurement error is added |

## 3. Analytic claims and conjectures in section 3 of the note

I1 to I3, verified analytically, I agree. Per sequence the signed outcome splits over
accepted and rejected shots, so F_dec = A F_acc + (1 - A) F_rej exactly. Two cautions. The
bounds in I2 hold only under the stated conditions, so "no decoder beats the accepted-shot
fidelity of detection" needs "if F_rej <= F_acc". The identity is per sequence. The repo
averages |P(0) - P(1)| over 30 sequences, and a mean of products is not a product of means.

I4 and I5, verified analytically, I agree. eps = (1 - f)/2 follows from Google's one-point
formula with 1 - 2 p_L = b f^m. The variance is (1 - F^2)/(A N), and 1/(4e-6) = 2.5e5.

I6, partly. The first-order algebra is right if A(m) is close to a exp(-r m). That is roughly
true for unif policies. It fails for const=k with fixed k, where A(m) levels off, to 1/16 for
const=1 in the fully mixed limit of I13. The rule compares decay rates. At a given m the
prefactors a b and b' also matter, so the summary's "beats decoding iff" should read "decays
more slowly than decoding iff".

I7, factual slip. With S_X1 = X0 X1 X2 and S_X2 = X1 X3 X4, Z0 and Z2 give (1,0) and Z3 and
Z4 give (0,1). The Z pairs are {0,2} and {3,4}, with Z0 Z2 = Z_L and Z3 Z4 equivalent to Z_L.
The pairs {0,3} and {2,4} are the X pairs.

I7 and I8, a missing first-order term. Verified analytically for single data-qubit Pauli
faults with the ideal X0 X3 readout. Within an ambiguous pair {a, b}, E_a E_b = L S with L a
logical and S a stabilizer. Data errors never spread to other data qubits through the check
CNOTs, so each error reaches the readout as a Pauli mapped by the remaining logical gates.
Write s(P) = +1 if that image commutes with X0 X3 and -1 if not. Then s(E_a) s(E_b) = s(L).

- If s(L) = +1, any guess gives the right readout, so the fault contributes +1. Doing nothing
  gives +1 or -1. Every X-type pair in the terminal round is in this case. For uniformly random
  Clifford frames about a third of the ambiguous faults in forward rounds are too.
- If s(L) = -1, ML guessing gives |pi_a - pi_b|/(pi_a + pi_b), which is about 0 for equal priors.

Errors on qubit 1 never flip X0 X3. The listed logical gates act on qubit 1 alone, with H or
S-dagger, and the check CNOTs move its errors only into ancillas. So the gain from qubit-1
faults comes from not rejecting them, not from correcting them. To first order,

F_dec - A F_acc = P(qubit-1 fault) + P(ambiguous fault with s(L) = +1) + prior-asymmetry terms.

The Z-check hook still fits I9: a Z on ancilla 6 after CNOT 1->6 becomes Z2 Z4 = S_Z2 Z1, syndrome
(1,1), the same class as Z1. Measurement faults are the case to watch. In a terminal-only check,
a flipped X-check ancilla looks like a Z fault on {0,2} or {3,4}, so a correcting decoder flips
the readout about half the time where doing nothing would not. The DEM numerics should count
these.

I9, conjecture, plausible. Accepted shots get first-order errors only from unchecked faults,
as in I12. The rule also throws away the harmless s(L) = +1 faults, which the comparison should
report.

I10, conjecture. Chen2025 states the finite-abort-rate result directly. English2024 and
Chen2025 define abort thresholds in the large-distance limit, so "above its abort threshold"
is an analogy for a fixed d=2 code, as open question 2 already says.

I11, the logic is correct.

I12, conjecture, consistent with CONTEXT.md. After CNOT 7->0 qubit 0 still sees the M7 and M8
layers, and after CNOT 8->3 qubit 3 sees M8. Each such layer applies DEPOLARIZE1(2p), and a
Z or Y there flips x0 or x3 with probability 4p/3. Adding the DEPOLARIZE2 after those two
CNOTs gives about 5p of unchecked readout flips, or about 7.7p if the final round ends with a
reset layer. That is my first-order estimate from the noise model in CONTEXT.md, not a check of
the generated circuit.

I13, verified analytically, I agree. Four independent commuting stabilizers on a maximally
mixed state give 16 equally likely outcomes, and ancilla flips keep them uniform.

I14 to I16, conjecture. The citations fit, with one caution. Ceasura's mechanism is an
adversarial representation that stores gate history in syndrome qubits. It is not the memory of
a matching decoder, so I15 should present it as a loose analogy.

I17 to I19, conjecture. I19 leans on Willsch2018, but its failing case is coupling to a
two-level-defect bath during long two-qubit gates, in a protocol with no repeated check rounds.
SI1000's measurement and reset layers apply Markovian depolarizing noise, a different
mechanism. Kumar2026's L=1 result is the better support for "extra check rounds can cost more
than they catch".

I20, agreed.

Section 0, bullet 3, overstated. Both papers say the pre-rejection rate is first order in p.
Paetznick2024 pre-rejects about 25% of Steane runs and 30% of carbon runs. The tesseract Table 3
pre-rejects 6% to 29% depending on the experiment. Only the post-rejection rate is second order.
The same bullet says that in a distance-2 code "the single faults are the ambiguous ones", but
qubit-1 faults are not ambiguous, as I8 itself says.

## 4. Edits I recommend, most important first

1. I7 and the summary key finding: X faults pair as {0,3} and {2,4}, Z faults as {0,2} and {3,4}.
2. I7 and I8: add the s(L) = +1 term to the first-order formula. Say that qubit-1 faults never
   flip X0 X3, so their gain comes from accepting them. Mention terminal measurement faults.
3. F3: split it into two statements and give Gottesman's own cause, error build-up without
   repeated detection.
4. Section 0, bullet 3: write "post-rejection rate" and "most single faults".
5. I6 and the summary: write "decays more slowly", and state the assumption that A(m) is
   exponential in m, which fails for const=k at fixed k.
6. Reichardt2009 and the seed corrections: write "near-identical abstracts".
7. I19: cite Kumar2026's L=1 result for the cost of extra check rounds, and present Willsch2018
   as a different noise mechanism.
8. F20 and the note under section 1: add that the Nature 2026 version seems to drop the
   Steane-code result. This partly answers open question 7.
