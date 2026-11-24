# Detection versus correction: break-even, postselection overhead and fair comparison metrics

Literature notes 08 for the qubit [[5,1,2]] LRB study described in `../CONTEXT.md`.
Written 2026-10-28. I ran no simulations for this file. I fetched every source in
the table myself, from its arXiv abstract page, its arXiv full text, or the Crossref
record of its DOI. Numbers are quoted only from those pages.

Labels follow CONTEXT.md. Section 2 is all "from literature". Section 3 mixes short
derivations I checked by hand ("verified analytically", no numerics) with my own
inferences ("conjecture").

## 0. Short version

- From literature. Papers on postselected experiments report two numbers, the error
  over accepted shots and the acceptance, and almost never fold them into one figure
  of merit. Kumar et al. 2026 is the exception I found. They give an explicit
  break-even condition in terms of error reduction per unit of rejection and compare
  schemes by mean squared error at a fixed shot budget [Kumar2026; He2024; Paetznick2024].
- From literature. Every threshold theorem for postselected computation keeps
  postselection inside fixed-size ancilla preparation, and Knill says purely
  postselected computing "has little computational power" [Knill2005; AGP2007;
  Reichardt2009]. Postselection over a growing circuit stops scaling once the abort
  rate per round is finite [Chen2025; English2024].
- From literature. The experiments with the largest gains over physical baselines
  correct single faults and reject only ambiguous syndromes, which makes the rejection
  rate second order in p [Paetznick2024; ReichardtTess2024; Prabhu2021]. A distance-2
  code cannot copy this, because in such a code the single faults are the ambiguous ones.
- Verified analytically, size of the effect is conjecture. On the same rounds, any
  decoder's fidelity over all shots lies between A·F_acc and F_acc, where A·F_acc is
  detection that outputs a random bit whenever it would reject. For [[5,1,2]] I expect
  decoding to beat A·F_acc only by the rate of uniquely identifiable faults, the ones on
  qubit 1. Numerics should test this.
- From literature, applied as conjecture. Postselected RB has no complete fitting theory
  [Harper2018] and postselected dynamics are multi-exponential [Kumar2026]. A
  non-exponential LRB-D curve is therefore not, by itself, evidence of non-Markovian
  physical noise.

## 1. Bibliography

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched |
|---|---|---|---|---|---|---|
| Gottesman2016 | Quantum fault tolerance in small experiments | D. Gottesman | 2016 | arXiv:1610.03507 | https://arxiv.org/abs/1610.03507 | yes, abstract and ar5iv full text |
| Knill2004a | Fault-Tolerant Postselected Quantum Computation: Schemes | E. Knill | 2004 | arXiv:quant-ph/0402171 | https://arxiv.org/abs/quant-ph/0402171 | yes, abstract |
| Knill2004b | Fault-Tolerant Postselected Quantum Computation: Threshold Analysis | E. Knill | 2004 | arXiv:quant-ph/0404104 | https://arxiv.org/abs/quant-ph/0404104 | yes, abstract |
| Knill2005 | Quantum computing with realistically noisy devices (arXiv title "Quantum Computing with Very Noisy Devices") | E. Knill | 2005, arXiv 2004 | arXiv:quant-ph/0410199; DOI 10.1038/nature03350; Nature 434, 39-44 | https://arxiv.org/abs/quant-ph/0410199 | yes, abstract, arXiv PDF text, Crossref |
| AGP2007 | Accuracy threshold for postselected quantum computation | P. Aliferis, D. Gottesman, J. Preskill | 2007, journal 2008 | arXiv:quant-ph/0703264; Quantum Inf. Comput. 8, 181-244 | https://arxiv.org/abs/quant-ph/0703264 | yes, abstract |
| Reichardt2009 | Error-Detection-Based Quantum Fault-Tolerance Threshold (arXiv title "Error-detection-based quantum fault tolerance against discrete Pauli noise") | B. W. Reichardt | 2009, arXiv 2006 | arXiv:quant-ph/0612004; DOI 10.1007/s00453-007-9069-7; Algorithmica 55, 517-556 | https://arxiv.org/abs/quant-ph/0612004 | yes, arXiv abstract, Crossref, Caltech record |
| English2024 | Thresholds for post-selected quantum error correction from statistical mechanics | L. H. English, D. J. Williamson, S. D. Bartlett | 2024, PRL 2025 | arXiv:2410.07598; DOI 10.1103/nh49-52y2; PRL 135, 120603 | https://arxiv.org/abs/2410.07598 | yes, abstract, HTML v3, Crossref |
| Chen2025 | Scalable accuracy gains from postselection in quantum error correcting codes | H. Chen, D. Xu, G. M. Sommers, D. A. Huse, J. D. Thompson, S. Gopalakrishnan | 2025 | arXiv:2510.05222 | https://arxiv.org/abs/2510.05222 | yes, abstract, HTML v3 |
| Prabhu2021 | Distance-four quantum codes with combined postselection and error correction | P. Prabhu, B. W. Reichardt | 2021, PRA 2024 | arXiv:2112.03785; DOI 10.1103/PhysRevA.110.012419 | https://arxiv.org/abs/2112.03785 | yes, abstract, Crossref |
| Smith2024 | Mitigating errors in logical qubits | S. C. Smith, B. J. Brown, S. D. Bartlett | 2024 | arXiv:2405.03766; DOI 10.1038/s42005-024-01883-4; Commun. Phys. 7, 386 | https://arxiv.org/abs/2405.03766 | yes, abstract, Crossref |
| Paetznick2024 | Demonstration of logical qubits and repeated error correction with better-than-physical error rates | A. Paetznick, M. P. da Silva, C. Ryan-Anderson, et al. | 2024 | arXiv:2404.02280 (see note below) | https://arxiv.org/abs/2404.02280 | yes, abstract, HTML v3 |
| ReichardtTess2024 | Demonstration of quantum computation and error correction with a tesseract code | B. W. Reichardt, D. Aasen, R. Chao, et al. (18 authors) | 2024 | arXiv:2409.04628 (see note below) | https://arxiv.org/abs/2409.04628 | yes, abstract, HTML v2 |
| Self2022 | Protecting expressive circuits with a quantum error detection code | C. N. Self, M. Benedetti, D. Amaro | 2022, Nat. Phys. 2024 | arXiv:2211.06703; DOI 10.1038/s41567-023-02282-2; Nat. Phys. 20, 219-224 | https://arxiv.org/abs/2211.06703 | yes, abstract, HTML v2, Crossref |
| He2024 | Performance of quantum approximate optimization with quantum error detection | Z. He, D. Amaro, R. Shaydulin, M. Pistoia | 2024, Commun. Phys. 2025 | arXiv:2409.12104; DOI 10.1038/s42005-025-02136-8; Commun. Phys. 8, 217 | https://arxiv.org/abs/2409.12104 | yes, abstract, HTML v3, Crossref |
| Jin2025 | Iceberg Beyond the Tip: Co-Compilation of a Quantum Error Detection Code and a Quantum Algorithm | Y. Jin, Z. He, T. Hao, S. Omanakuttan, D. Amaro, S. Tannu, R. Shaydulin, M. Pistoia | 2025 | arXiv:2504.21172 | https://arxiv.org/abs/2504.21172 | yes, abstract |
| Kumar2026 | Co-Designing Error Mitigation and Error Detection for Logical Qubits | R. S. Kumar, T. Tsunoda, S. H. Xue, D. Li, R. J. Schoelkopf, Y. Ding | 2026 | arXiv:2604.19871 | https://arxiv.org/abs/2604.19871 | yes, abstract, HTML v1 |
| Vuillot2017 | Is error detection helpful on IBM 5Q chips ? | C. Vuillot | 2017, journal 2018 | arXiv:1705.08957; Quantum Inf. Comput. 18(11&12), 949-964 | https://arxiv.org/abs/1705.08957 | yes, abstract, arXiv PDF text |
| Willsch2018 | Testing quantum fault tolerance on small systems | D. Willsch, M. Willsch, F. Jin, H. De Raedt, K. Michielsen | 2018 | arXiv:1805.05227; DOI 10.1103/PhysRevA.98.052348 | https://arxiv.org/abs/1805.05227 | yes, abstract, ar5iv full text, Crossref |
| Harper2018 | Fault-Tolerant Logical Gates in the IBM Quantum Experience | R. Harper, S. T. Flammia | 2018, PRL 2019 | arXiv:1806.02359; DOI 10.1103/PhysRevLett.122.080504 | https://arxiv.org/abs/1806.02359 | yes, abstract, ar5iv full text, Crossref |
| Gupta2023 | Encoding a magic state with beyond break-even fidelity | R. S. Gupta, N. Sundaresan, T. Alexander, et al. (13 authors) | 2023, Nature 2024 | arXiv:2305.13581; DOI 10.1038/s41586-023-06846-3; Nature 625, 259-263 | https://arxiv.org/abs/2305.13581 | yes, abstract, HTML v2, Crossref |
| Google2024 | Quantum error correction below the surface code threshold | Google Quantum AI and Collaborators (R. Acharya et al.) | 2024, Nature 2025 | arXiv:2408.13687; DOI 10.1038/s41586-024-08449-y; Nature 638, 920-926 | https://arxiv.org/abs/2408.13687 | yes, HTML v1, Crossref |
| Ceasura2022 | Non-Exponential Behaviour in Logical Randomized Benchmarking | A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan | 2022 | arXiv:2212.05488 | https://arxiv.org/abs/2212.05488 | yes, abstract |
| Kabir2025 | Sdim: A Qudit Stabilizer Simulator | A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang, B. Sorée | 2025 | arXiv:2511.12777 | https://arxiv.org/abs/2511.12777 | yes, abstract, HTML v2 |

Note on Paetznick2024 and ReichardtTess2024. Crossref lists A. Paetznick, B. W. Reichardt,
M. P. da Silva, C. Ryan-Anderson, D. Aasen, et al., 37 authors, "Improved quantum processor
logical error rates via correction and detection", Nature 654, 349-355 (2026), DOI
10.1038/s41586-026-10628-y. I fetched that record. Its relation field is empty, so calling it
the journal version of both preprints is my inference from the title, the author lists and the
numbers in press coverage. All numbers below come from the arXiv versions.

## 2. Key findings (all from literature)

Each line gives an ID, the research question it bears on, the label, and one sentence.

### Fair comparison and break-even

- F1 · Q1 · from literature. Gottesman's criterion for a small fault-tolerance demonstration
  requires the complete encoded circuit, preparation and measurement included, to have a lower
  error rate than the same circuit run unencoded "using the best qubit(s) available in the
  system", for every circuit in a chosen family [Gottesman2016].
- F2 · Q1 · from literature. Gottesman notes that the 4-qubit detecting code "can only improve
  error rates via post-selection", that such demonstrations "can be controversial", and that a
  fair comparison sets the encoded error conditioned on acceptance against the never-rejected
  unencoded error [Gottesman2016].
- F3 · Q1 · from literature. Gottesman warns that in longer circuits "the odds of successfully
  completing a run without detecting an error become much smaller", so the per-gate error
  needed to show fault tolerance drops as the circuit family gets longer [Gottesman2016].
- F4 · Q1 · from literature. Published break-even claims use at least three baselines, namely
  the lifetime of the best physical qubit [Google2024], the best physical qubit pair
  [Gupta2023; Vuillot2017], and the same circuit unencoded with no syndrome extraction or
  postselection, whose error divided by the logical error Paetznick et al. call the "gain"
  [Gottesman2016; Paetznick2024].
- F5 · Q1 · from literature. Google reports its distance-7 memory beyond break-even by a factor
  2.4 ± 0.3 over its best physical qubit's lifetime and converts a logical error probability
  after n cycles into a per-cycle error with ε = ½[1 − (1 − 2 p_L)^(1/n)] [Google2024].
- F6 · Q1 · from literature. Gupta et al. claim beyond-break-even magic-state preparation
  against the best physical qubit pair, with feed-forward logical infidelity
  (1.87 ± 0.16)×10⁻² against (2.35 ± 0.27)×10⁻² for that pair and about 6.2×10⁻² averaged
  over pairs [Gupta2023].
- F7 · Q1 · from literature. Gupta et al. give a noiseless acceptance of 75% with feed-forward
  against 37.5% with pure postselection, measured 14.9(1)% acceptance for the encoded scheme
  under physical-qubit tomography, and do not fold acceptance into the break-even comparison
  [Gupta2023].
- F8 · Q1 · from literature. Vuillot picks the best bare qubit pair by running all six connected
  pairs and finds that the fault-tolerant [[4,2,2]] preparation beats that pair only on
  average, mean statistical distance 4.51×10⁻² at postselection ratio 0.65 against 5.50×10⁻² at
  ratio 1.00, with some circuits always better bare [Vuillot2017].

### Threshold theory for postselected computation

- F9 · Q1 · from literature. Knill defines postselected computing as computing in which a gate
  may fail, the failure is known and success has nonzero probability, says purely postselected
  computing "has little computational power", and uses it only to prepare states for standard
  computation [Knill2005].
- F10 · Q1 · from literature. Knill's schemes paper states that encoded postselected computation
  with no detected errors can be made arbitrarily accurate, but its success probability drops
  sharply as the accuracy rises [Knill2004a].
- F11 · Q1/Q3 · from literature. In Knill's C4/C6 architecture correction comes from
  concatenating detecting codes, because "error-detecting codes can always correct an error at
  a known location", so a detection flag at one level works as an erasure the next level
  corrects [Knill2005].
- F12 · Q1 · from literature. Knill's measurement gadget shows the trade in miniature, since
  accepting a duplicated measurement only when both outcomes agree gives conditional error
  4γ/15 + O(γ²) while a rejection-free majority vote over three outcomes gives 4γ/5 + O(γ²),
  with γ the CNOT error parameter [Knill2005].
- F13 · Q1 · from literature. Heuristic thresholds for Knill's postselected schemes are above 1%
  per gate when communication delays are negligible [Knill2004b] and about 3%, perhaps 5%, for
  the C4/C6 architecture, whose "postselection overheads are extreme" [Knill2005].
- F14 · Q1 · from literature. Aliferis, Gottesman and Preskill prove a threshold of at least
  1.04×10⁻³ for postselected computation in which ancilla preparations protected by a
  concatenated detecting code are discarded on detection, for independent stochastic noise but
  not strongly correlated adversarial noise [AGP2007].
- F15 · Q1 · from literature. Reichardt proves that error-detection-based schemes have a constant
  threshold against discrete Pauli noise, with a numerical lower bound of 0.1% per gate, well
  below the 3 to 6% that heuristics had suggested [Reichardt2009].
- F16 · Q1 · from literature. English, Williamson and Bartlett find that postselected QEC has an
  abort threshold and a conditional logical threshold that together give four phases, call the
  abort threshold "an important necessary condition for post-selected QEC to be scalable", and
  note that when every non-trivial syndrome aborts the decoding strategy "is simply to do
  nothing" [English2024].
- F17 · Q1 · from literature. Chen et al. state that "if the abort rate per measurement is
  finite, postselection is not scalable", show that aborting only on exponentially rare
  syndromes keeps acceptance of order one while suppressing logical failure from p_f to p_f^b
  with b = 3.1(1) for the toric code with perfect measurements, and find that finite-acceptance
  postselection does not move the threshold [Chen2025].
- F18 · Q1 · from literature. Smith, Brown and Bartlett's exclusive decoders abort only on
  instances judged too hard, improve logical failure up to quadratically below threshold, have
  exclusion rates that fall with code distance at low noise for a modest criterion, and are
  costed in total spacetime volume including the repetitions [Smith2024].

### Correction and detection on the same hardware

- F19 · Q1 · from literature. Prabhu and Reichardt correct single faults and postselect against
  double faults so that a logical error needs at least three faults, which lets a postselected
  distance-4 surface code accumulate about 25 times less error than distance 5, at the price of
  restarts that rule out very long storage [Prabhu2021].
- F20 · Q1 · from literature. In the same trapped-ion Steane-code Bell-state experiment,
  correction gave 0.05% logical error at about 75% acceptance after preselection, full
  detection gave 0.001% at about 72% acceptance, and the unencoded baseline was 0.50%
  [Paetznick2024].
- F21 · Q1 · from literature. For the [[12,2,4]] code, adding rejection of ambiguous syndromes
  to correction cut the Bell-state error from 0.17% to 0.001% while rejecting about 0.5% more
  of the preselected runs, because "the post-rejection rate is second order in the physical
  error rate, since single faults are corrected instead of rejected" [Paetznick2024].
- F22 · Q1 · from literature. Paetznick et al. compute logical error over accepted runs against
  unencoded baselines that reject nothing, and argue that postselection is "not scalable in a
  strict sense" but allows relatively deep circuits when rejection is low [Paetznick2024].
- F23 · Q1 · from literature. The tesseract experiments correct single faults because "we cannot
  reject these first-order events or the rejection rate would be too high", reject ambiguous
  unflagged outcomes, and report acceptance between 50% and 90% with error computed on
  accepted trials only [ReichardtTess2024].
- F24 · Q1 · from literature. In the tesseract simulation of one code block the correct-plus-reject
  logical error grows by 2.1(1)×10⁻⁴ per round while 2.02(2)% of surviving trials are discarded
  per round, the per-round pair of numbers a fair comparison needs [ReichardtTess2024].

### Detection codes inside algorithms

- F25 · Q1/Q3 · from literature. The Iceberg [[k+2,k,2]] experiments discard every shot with a
  detected error, accept "an overhead in the number of circuit repetitions", reached an
  86.8(7)% discard rate in the passing 8-logical-qubit quantum volume run, and found that
  adding syndrome measurements raised the heavy-output frequency [Self2022].
- F26 · Q1 · from literature. Self et al. benchmark with mirror circuits that apply a logical
  unitary and then its inverse, the same forward-then-inverse layout as our LRB circuits, and
  report survival probability over postselected shots [Self2022].
- F27 · Q1 · from literature. He et al. model Iceberg logical fidelity as H/(1 − D), with D the
  discard probability, define breakeven as equal accepted-shot fidelity for encoded and
  unencoded circuits, and track the postselection rate separately as a 10% contour instead of
  building sampling cost into the criterion [He2024].
- F28 · Q1/Q3 · from literature. He et al. find that QAOA logical fidelity levels off as syndrome
  measurements are added while the postselection rate keeps falling, and call detection codes
  "non-scalable" but useful at current sizes [He2024].
- F29 · Q3 · from literature. Co-compiling Iceberg gadgets with the QAOA circuit raised the
  postselection rate from 4% to 33% and the success probability from 44% to 65% at 22
  algorithmic qubits on Quantinuum H2-1, and gave better-than-unencoded results up to 34
  algorithmic qubits [Jin2025].
- F30 · Q1/Q3 · from literature. Kumar et al. derive an acceptance-aware break-even condition for
  detection combined with probabilistic error cancellation, (ε_p − ε_ℓ)/p_rej > 1/4, which
  counts error reduction per unit of rejection probability, and they rank schemes by mean
  squared error at a fixed budget of attempted shots [Kumar2026].
- F31 · Q1/Q3 · from literature. Kumar et al. find that one detection cycle per gate reaches
  break-even for none of the codes they evaluate, while optimized intervals on the [[6,4,2]]
  and [[12,10,2]] Iceberg codes do and the [[9,1,3]] code shows no advantage at any interval
  [Kumar2026].

### Postselection and the shape of RB decays

- F32 · Q2 · from literature. Kumar et al. report that the first detection cycle is a transient,
  because the first syndrome extraction only injects detectable errors without removing earlier
  ones, so accepted-state correctness is multi-exponential with fast modes of eigenvalue O(p)
  that die out within O(1) rounds and slow modes of eigenvalue 1 − O(p) [Kumar2026].
- F33 · Q2 · from literature. Harper and Flammia ran RB inside the [[4,2,2]] code space with
  postselection, wrote that "there is at present no complete theory for fitting an RB decay
  curve with post-selection", and measured two-qubit infidelity 0.60(3)% encoded against
  5.8(2)% physical while SPAM noise kept them from a below-threshold claim [Harper2018].
- F34 · Q2 · from literature. Ceasura et al. build a gate- and time-independent noise model in
  which logical RB oscillates instead of decaying, because syndrome qubits coupled to the
  logical qubits through the decoder hold a memory of the gate history [Ceasura2022].
- F35 · Q2 · from literature. The Sdim paper defines LRB-D as LRB with one terminal syndrome
  extraction and postselection on the all-zero syndrome, notes that "terminal detection can
  produce decay curves with non-Markovian structure that are not always well summarized by a
  single exponential fit", and chose detection because a distance-2 syndrome "generally does
  not provide enough information to identify a unique correction" [Kabir2025].
- F36 · Q2/Q3 · from literature. Willsch et al. find that Gottesman's [[4,2,2]] scheme passes his
  criterion when control and measurement errors dominate but fails it when decoherence from a
  two-level-defect environment dominates, with the two-qubit gates the most sensitive
  [Willsch2018].

## 3. Implications for the qubit [[5,1,2]] LRB study

Notation, per random sequence and depth m. A is the acceptance. F_acc is the signed fidelity
P(correct) − P(wrong) over accepted shots, which equals the repo's |P(0) − P(1)| when the sign
is right. F_rej is the same quantity over rejected shots after some decoder has been applied.
F_dec is the decoded signed fidelity over all shots.

### 3.1 Postselection as a decoder that can abort

- I1 · Q1 · verified analytically. If a decoder applies no correction whenever every syndrome
  the policy uses is trivial, and reads only the rounds the policy reads, then
  F_dec = A·F_acc + (1 − A)·F_rej exactly, by the law of total expectation.
- I2 · Q1 · verified analytically. It follows that F_dec ≤ F_acc whenever F_rej ≤ F_acc, so on
  the same information no decoder beats the accepted-shot fidelity of detection, and
  F_dec ≥ A·F_acc whenever F_rej ≥ 0.
- I3 · Q1 · verified analytically. A·F_acc is the fidelity of a scheme that keeps every shot and
  outputs a uniformly random bit whenever detection would reject, so it is detection at
  acceptance 1 and can be set directly against any decoder's F_dec.
- I4 · Q1 · verified analytically. With F(m) = b·f^m and F = 1 − 2 p_L for a qubit observable,
  the Google conversion in F5 gives ε = (1 − f)/2 per unit of m. In our circuits one unit of m
  holds a forward Clifford, one noisy check round and one inverse Clifford, so ε is an error
  per unit of m, not per check round, and the matched physical RB has to be compared per the
  same unit.
- I5 · Q1 · verified analytically. Estimating F from N attempted shots at acceptance A has
  variance (1 − F²)/(A·N), so absolute precision δ needs N ≈ (1 − F²)/(A·δ²) attempted shots.
  For the qutrit unif=1 run at m = 22 and p = 5×10⁻³, A ≈ 4×10⁻⁶ means about 2.5×10⁵ attempted
  shots per accepted shot.
- I6 · Q1 · verified analytically, to first order in the per-unit rates. A·F_acc ≈ b·exp[−m(r + 2ε_acc)]
  and F_dec ≈ b′·exp(−2m·ε_dec), where r is the rejection rate per unit of m, so detection with
  random output on abort decays more slowly than decoding exactly when r + 2ε_acc < 2ε_dec.
  This is the LRB counterpart of the Kumar et al. condition in F30.

### 3.2 What correction can do at distance 2

- I7 · Q1 · conjecture. For [[5,1,2]], X or Z faults on the pairs {0,3} and {2,4} share a
  syndrome and differ by a weight-2 logical, so a maximum-likelihood decoder guesses with
  roughly even odds when the circuit-level priors of the two pair members are close. Each such
  guess contributes about zero signed fidelity, which is exactly what aborting and outputting a
  random bit contributes. The priors are not exactly equal, because qubits 0 and 3 sit at
  different positions in their check circuits, so the DEM should be checked.
- I8 · Q1 · conjecture. Faults on qubit 1 give the (1,1) syndrome and are uniquely correctable,
  so to first order F_dec ≈ A·F_acc + P(a uniquely correctable fault occurred). In the I6
  condition, correction then wins by roughly the per-unit rate of uniquely correctable faults.
  If data faults were uniform over the five qubits that rate would be about one fifth of the
  detectable data faults, but hook errors, measurement faults and the gate layout will move it.
- I9 · Q1 · conjecture. A hybrid that applies the qubit-1 correction for a (1,1) syndrome and
  rejects (1,0) and (0,1) is the distance-2 counterpart of the correct-plus-reject strategy in
  [Paetznick2024; ReichardtTess2024; Prabhu2021]. It should keep accepted-shot logical error at
  second order in p for data faults while cutting the rejection rate by the uniquely
  correctable share. It belongs in the decoder set we test.

### 3.3 Check policies and scalability

- I10 · Q1 · conjecture. unif=1 detection is purely postselected computing in Knill's sense.
  Every detectable single fault in every round rejects, so r is first order in p and A(m) falls
  exponentially in m, which fits the qutrit acceptance of about 4×10⁻⁶ at m = 22. The threshold
  theorems [Knill2005; AGP2007; Reichardt2009] avoid this regime by confining postselection to
  fixed-size gadgets, and in the language of [English2024; Chen2025] our scheme sits above its
  abort threshold at every p > 0.
- I11 · Q1 · conjecture. Under const=k the policy reads fewer rounds than the circuit contains,
  so a decoder that reads all m+1 rounds has more information than the postselection rule and
  can beat F_acc on the policy's accepted shots, which breaks I2. For fair attribution we should
  also run each decoder restricted to the rounds the policy reads.
- I12 · Q1 · conjecture. Data noise on qubits 0 and 3 in the layers after the last X-check CNOT
  that touches them in the final round, from later measurement layers or a final reset layer
  if there is one, is never checked before the ideal X0X3 readout. Z faults there flip the
  logical outcome on accepted shots at first order. This lowers the intercept b rather than f,
  hits detection and correction equally, and matches Gottesman's warning about incomplete
  circuits. The generated circuit's layer order should be checked before relying on this.

### 3.4 Non-Markovianity (Q2)

- I13 · Q2 · verified analytically, in the fully depolarized limit. With const=1 only the
  terminal syndrome is read, and once the data qubits are maximally mixed the four stabilizer
  outcomes are uniform. Acceptance therefore tends to 1/16 and rejection to 15/16, the qubit
  counterpart of the 1 − 1/81 in the qutrit run.
- I14 · Q2 · conjecture. F_acc(m) is a ratio, accepted-and-correct over accepted, and both parts
  are sums of exponentials in m with different weights. A non-exponential const=1 curve can
  therefore come from postselection conditioning alone, with Markovian, gate-independent noise,
  which agrees with [Kumar2026; Harper2018; Kabir2025].
- I15 · Q2 · conjecture. Correction at acceptance 1 removes the conditioning. With a decoder
  that works round by round, the logical process per unit of m is a fixed linear map and LRB
  should return to a single exponential for Markovian, gate-independent noise. A decoder that
  matches across many rounds stores syndrome history and could produce the kind of
  non-exponential behaviour Ceasura et al. construct [Ceasura2022]. "Correction mitigates
  non-Markovianity" is safest as a claim about the conditioning-induced part only.
- I16 · Q2 · conjecture. The first-cycle transient in F32 predicts a fast drop in A(m) and
  F_acc(m) over the first one or two units of m, then slower decay. Fitting only m ≥ 2, or
  fitting two exponentials, should separate the two.

### 3.5 Hardware (Q3)

- I17 · Q3 · conjecture. A sustainable hardware scheme for this code keeps postselection inside
  bounded windows and corrects uniquely identified syndromes. It sets the check interval with an
  acceptance-aware rule such as I6 or F30 instead of checking after every Clifford, which never
  broke even in [Kumar2026]. It also co-compiles the checks with the logical Cliffords, which
  moved Iceberg acceptance from 4% to 33% [Jin2025].
- I18 · Q3 · conjecture. By Knill's known-location principle (F11), any hardware herald of where
  an error happened, such as leakage or erasure detection, turns an ambiguous [[5,1,2]] fault
  into a correctable one. Platforms with heralded erasures should gain the most from moving
  from detection to correction.
- I19 · Q3 · conjecture. Willsch et al. saw detection fail when decoherence dominates (F36), and
  our SI1000 model already places about five DEPOLARIZE1(2p) layers on every data qubit per
  check round, so extra check rounds can cost more than they catch. The per-unit pair
  (r, ε_acc) from I6 shows where that happens as a function of p and policy.

### 3.6 Proposed metric set

Conjecture, as a proposal. No paper I found defines an acceptance-normalized metric for LRB.
This set combines F4, F24, F30 and I1 to I6.

| Quantity | Definition | Role |
|---|---|---|
| ε_acc and r | ε_acc = (1 − f_acc)/2 from the accepted-shot fit; r from the slope of −ln A(m) | report as a pair, like the tesseract per-round numbers (F24) |
| F_abort(m) | A(m)·F_acc(m), detection with a random bit on abort | compare with F_dec(m) at acceptance 1 (I3) |
| ε_dec | (1 − f_dec)/2 from the decoded fit over all shots | headline for correction |
| N_δ | (1 − F²)/(A·δ²) attempted shots for precision δ | sampling cost at each m (I5) |
| MSE at fixed budget | bias² + variance at a fixed number of attempted shots | one-number comparison, following [Kumar2026] |
| break-even | win at every m in the grid against a named baseline | Gottesman's family-wide test (F1) |

- I20 · Q1 · conjecture. Any break-even claim in this study should name its baseline. The repo's
  matched physical RB at p_RB = 5p is a matched-noise baseline, not the best-physical-qubit or
  best-pair baseline used in experiments (F4). Our ideal encoding and ideal readout also fall
  short of Gottesman's complete-circuit condition (F1), which flatters the encoded scheme
  relative to experiments.

## 4. Seed corrections

- Knill, "postselected computation" (quant-ph/0404104). The ID is right. The exact title is
  "Fault-Tolerant Postselected Quantum Computation: Threshold Analysis" (2004, no journal
  reference on arXiv). Its companion is "Fault-Tolerant Postselected Quantum Computation:
  Schemes", quant-ph/0402171. The C4/C6 Nature paper is a third work, quant-ph/0410199, titled
  "Quantum Computing with Very Noisy Devices" on arXiv and "Quantum computing with realistically
  noisy devices" in Nature 434, 39-44 (2005), DOI 10.1038/nature03350.
- Reichardt, error-detection-based fault tolerance (no ID given). Found as quant-ph/0612004,
  "Error-detection-based quantum fault tolerance against discrete Pauli noise" (2006 PhD
  thesis, no journal reference on arXiv). The journal paper is "Error-Detection-Based Quantum
  Fault-Tolerance Threshold", Algorithmica 55(3), 517-556 (2009, online 2007), DOI
  10.1007/s00453-007-9069-7. The two abstracts match. Neither record links the other, so the
  identification rests on the identical abstracts.
- Aliferis, Gottesman, Preskill (quant-ph/0703264). Correct. The arXiv year is 2007 and the
  journal is Quantum Inf. Comput. 8, 181-244 (2008).
- Vuillot (1705.08957). Correct. The exact arXiv title has a space before the question mark,
  "Is error detection helpful on IBM 5Q chips ?". Journal Quantum Inf. Comput. 18(11&12),
  949-964 (2018).
- Willsch et al. (1805.05227). Correct. Authors D. Willsch, M. Willsch, F. Jin, H. De Raedt,
  K. Michielsen; Phys. Rev. A 98, 052348 (2018).
- Self, Benedetti, Amaro (2211.06703). Correct. Published in Nat. Phys. 20, 219-224 (2024).
- Gupta et al. (2305.13581). Correct. Thirteen authors, last author B. J. Brown; Nature 625,
  259-263 (2024).

## 5. Open questions

1. Harper and Flammia say no complete theory exists for fitting RB decays under postselection
   (F33). Is there an LRB-D model, for example a ratio of two exponential sums, that yields an
   interpretable per-round error for each const and unif policy?
2. The abort-threshold language of [English2024; Chen2025] is asymptotic in code distance. A
   fixed-size code at finite depth needs a different notion. One candidate is a depth m*(p)
   beyond which N_δ from I5 exceeds the shot budget.
3. What fraction of detectable circuit-level faults in the [[5,1,2]] SI1000 DEM is uniquely
   identifiable? That number sets the first-order gain of correction over A·F_acc (I8) and
   needs numerics.
4. Does the hybrid correct-unique, reject-ambiguous decoder of I9 beat both pure strategies on
   the metrics in 3.6?
5. Does the first-cycle transient of [Kumar2026] show up in our runs as an initial drop at
   m ≤ 2, and how much of the non-exponential const=1 behaviour does it explain?
6. Which IQM, trapped-ion or neutral-atom platforms provide leakage or erasure heralds that
   could enable known-location correction (I18)? The hardware review should answer this.
7. The Nature 2026 paper in the note under section 1 is paywalled. Someone with access should
   confirm that it combines Paetznick2024 and ReichardtTess2024 and check whether its
   published numbers changed.
8. Kumar et al.'s condition in F30 is derived for detection plus PEC with exponential sampling
   costs. Is the LRB counterpart in I6 the right decision rule when the goal is estimating a
   decay rate, not an expectation value?

## 6. Search log

I ran 19 web searches with these phrasings: Reichardt error-detection-based fault-tolerance
threshold; thresholds for post-selected QEC from statistical mechanics (2024); Gottesman small
experiments with [[4,2,2]] postselection; Iceberg code QAOA postselection overhead (2024);
tesseract code correction versus detection rejection rate; distance-four codes with combined
postselection and correction; logical RB with detection, postselection and leakage;
better-than-physical logical error rates with [[7,1,3]] and [[12,2,4]]; postselection sampling
overhead and acceptance decay with circuit size; soft-output decoders, logical gap and abort
rate; [[k+2,k,2]] codes in 2025 algorithms; sliding-scale detection in neutral-atom logical
processors; break-even definitions; below-threshold surface code logical error per cycle;
RB with postselection and non-exponential decay; Sdim and detection-only LRB; the Nature title
"Improved quantum processor logical error rates via correction and detection"; symmetry
verification sampling overhead; and when detection beats correction for distance-2 codes
(2025 and 2026).

Other hits I fetched and verified but did not use for claims here, listed for the other
reviews: Combes, Granade, Ferrie, Flammia, "Logical Randomized Benchmarking", arXiv:1702.03688
(2017); Bombín, Pant, Roberts, Seetharam, "Fault-tolerant Post-Selection for Low Overhead Magic
State Preparation", arXiv:2212.00813, PRX Quantum 5, 010302 (2024); Bluvstein et al., "Logical
quantum processor based on reconfigurable atom arrays", arXiv:2312.03982, DOI
10.1038/s41586-023-06927-3; Linke et al., "Fault-tolerant quantum error detection",
arXiv:1611.06946, Sci. Adv. 3, e1701074 (2017); Kasatkin, Morford-Oberst, Vezvaee, Lidar,
"Quantum Error Correction and Dynamical Decoupling: Better Together or Apart?",
arXiv:2602.19042 (2026).
