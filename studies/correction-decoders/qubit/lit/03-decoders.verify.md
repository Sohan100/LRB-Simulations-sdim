# Verification of lit review 03: decoders under circuit-level noise

Independent check of `lit/03-decoders.md` (file time 2026-10-28 10:52). I
checked every citation against its arXiv abstract page and, where a DOI
exists, against Crossref. I checked every key finding in the review's
structured summary against the cited text, the installed package
docstrings, or my own light calculations. All fetches were done by me on
2026-10-28.

Labels follow CONTEXT.md: **verified analytically**, **verified
numerically**, **from literature**, **conjecture**.

## Bottom line

- No hallucinated sources and no wrong identifiers. All 29 citations
  resolve to the stated works. Every arXiv ID, DOI, title, author list and
  year matches the arXiv page or Crossref. Four entries need a venue fix.
  Cain2024 is published in Phys. Rev. Lett. 133, 240602 (2024).
  IyerPoulin2013 is published in IEEE Trans. Inf. Theory 61(9), 5209-5223
  (2015). Das2021 is published in the ASPLOS '22 proceedings, pp. 541-553.
  Spitz2018's journal version has a longer title and volume 1. *From
  literature (Crossref).*
- Of the 29 key findings, 23 are supported, 5 are partially supported, none
  is unsupported, and 1 is a conjecture with nothing to check it against.
- One conclusion is wrong as stated. The review says LRB-D has O(p²)
  logical error on accepted shots (summary bullet 2, I1, and the O(p²)
  promise of the abort rule in I9). Under the CONTEXT readout, which
  measures only data 0 and 3, the final noisy round contains first-order
  faults that flip the logical readout and no detector. Their total
  probability is 4p + 8p/15 ≈ 4.53p. *Verified analytically.* The detector
  error model of the lead session's `src/lrb_qubit` circuits with
  `terminal="data2"` has exactly one such mechanism, of probability 4.52p
  at p = 1e-3, at depths 0, 1 and 4. With the all-data X readout of I8
  (`terminal="data5"`) it disappears. *Verified numerically.* So LRB-D has
  a depth-independent O(p) floor in the protocol as written, and no decoder
  or abort rule gets below it without the readout change.
- Smaller corrections. Fowler's "factor of 2" is the change in the ratio of
  d=3 to d=5 logical error rates, not a twofold lower logical error rate.
  LILLIPUT's main text reports 29-42 ns decoding latency, while its abstract
  says "few nanoseconds". The time-like fault rate is about 40p per interior
  round once idle-ancilla and gate terms are counted, not 31p. The (1,1)
  alias in I1 exists only when the interval contains a logical S or H. The
  `add_edge` fix in I4 must pass an explicit weight.

## How I checked

Metadata. I downloaded each arXiv abstract page with curl and read the
`citation_title`, `citation_author`, `citation_date` and DOI meta tags, the
journal-ref field and the submission history. I queried the Crossref API for
every DOI and ran Crossref bibliographic searches for the arXiv-only items.
For the software sources I read the GitHub releases API for PyMatching, the
PyPI JSON for pymatching, ldpc and tesseract-decoder, the Stim DEM format
document from GitHub, and the Tesseract and ldpc READMEs.

Content. I downloaded the arXiv HTML full text of all 26 papers, converted
it to plain text and searched it for each claim. I read the docstrings of
the installed stim 1.15.0, pymatching 2.3.1 and ldpc 2.4.1
(`module load python/3.11`).

Numerics. I wrote my own small scripts and also re-ran the review's two toy
scripts after reading them. Everything ran under `nice -n 19` and took well
under a minute in total. Script paths are in section 3.

## 1. Citation metadata

"ok" means title, authors, year and identifier all match.

| Key | Checked against | Verdict | Correction or note |
|---|---|---|---|
| Gidney2021 | arXiv abs, Crossref | ok | Quantum 5, 497 (2021-07-06). |
| HiggottGidney2025 | arXiv abs, Crossref | ok | Quantum 9, 1600 (2025-01-20). |
| Fowler2013 | arXiv abs, Crossref search | ok | arXiv only. Crossref search found no journal version. |
| Higgott2023 | arXiv abs, Crossref | ok | PRX 13, 031007 (2023). |
| DelfosseNickerson2021 | arXiv abs, Crossref | ok | Quantum 5, 595 (2021). |
| PanteleevKalachev2021 | arXiv abs, Crossref | ok | arXiv capitalizes the title. Quantum 5, 585 (2021). |
| Roffe2020 | arXiv abs, Crossref | ok | The review uses the PRR title. The arXiv title is "Decoding Across the Quantum LDPC Code Landscape". Same work. |
| Hillmann2025 | arXiv abs, Crossref | ok | Six authors confirmed. Nat. Commun. 16, 8214 (2025). |
| IyerPoulin2013 | arXiv abs, Crossref | venue incomplete | Published as IEEE Trans. Inf. Theory 61(9), 5209-5223 (2015), DOI 10.1109/TIT.2015.2422294. The review lists it as an arXiv preprint. |
| BravyiSucharaVargo2014 | arXiv abs, Crossref | ok | PRA 90, 032326 (2014). |
| Piveteau2024 | arXiv abs, Crossref | ok | PRX Quantum 5, 040303 (2024). Crossref spells "Tensor-Network". |
| Heim2016 | arXiv abs, Crossref search | ok | arXiv only. No journal version found. |
| Cain2024 | arXiv abs, Crossref | venue incomplete | Published as Phys. Rev. Lett. 133, 240602 (2024-12-11), DOI 10.1103/PhysRevLett.133.240602, same eight authors. The review lists it as an arXiv preprint. |
| SerraPeralta2026 | arXiv abs, Crossref | ok | PRX Quantum 7, 010335 (2026-02-19). arXiv v4 of 2026-07-06 carries the DOI. |
| Beni2025 | arXiv abs, Crossref search, GitHub | ok | arXiv only, v2 2025-08-06. Code at github.com/quantumlib/tesseract-decoder, as the paper's footnote says. |
| RyanAnderson2021 | arXiv abs, Crossref | ok | 17 authors. PRX 11, 041058 (2021). |
| Das2021 | arXiv abs, Crossref | venue incomplete | Published in Proc. ASPLOS '22, pp. 541-553 (2022), DOI 10.1145/3503222.3507707. The proceedings title drops "Based": "LILLIPUT: A Lightweight Low-Latency Lookup-Table Decoder for Near-term Quantum Error Correction". |
| Bausch2024 | arXiv abs, Crossref | ok | 18 authors. Nature 635, 834-840 (2024) under the journal title the review gives. |
| Skoric2023 | arXiv abs, Crossref | ok | Nat. Commun. 14, 7040 (2023). |
| Smith2024 | arXiv abs, Crossref | ok | Commun. Phys. 7, 386 (2024). |
| Spitz2018 | arXiv abs, Crossref | journal title differs | The DOI resolves to "Adaptive Weight Estimator for Quantum Error Correction in a Time-Dependent Environment", Adv. Quantum Technol. 1(1), 1800012 (2018). The review gives the arXiv title without saying so. |
| StimDocs | GitHub doc, installed docstring | ok | Content matches, see section 2. |
| PyMatchingDocs | GitHub releases API, PyPI, docstrings | ok | v2.3.0 published 2025-08-13 (tag created 2025-08-10), v2.3.1 published 2025-09-25. PyPI upload of 2.3.0 on 2025-08-10. Sid Madhuk, PR #156. PyPI now has 2.4.0; 2.3.1 is the installed version. |
| ldpcDocs | GitHub README, installed package | ok | `SinterLsdDecoder` and `SinterBeliefFindDecoder` live in `ldpc.sinter_decoders`, not top-level `ldpc`. `PyMatchingOverlappingWindowDecoder` lives in `ldpc.ckt_noise.pymatching_overlapping_window`. |
| Higgott2022 | arXiv abs, Crossref | ok | ACM TQC 3(3), 1-16 (2022). |
| Chubb2021 | arXiv abs | ok | arXiv only. |
| Huang2020 | arXiv abs, Crossref | ok | PRA 102, 012419 (2020). |
| TomitaSvore2014 | arXiv abs, Crossref | ok | PRA 90, 062320 (2014). |
| Meister2024 | arXiv abs, Crossref search | ok | arXiv only, v2 2024-06-01. |

Seed corrections (review section 4). All 13 seed IDs are correct and each
stated correction checks out. One nuance: the Stim paper does mention a
`--detector_hypergraph` mode that "groups error mechanisms present in a
circuit based on which annotated detectors and logical observables those
errors flip". That is the precursor of DEMs, but the paper gives no DEM
semantics, so citing StimDocs for DEM claims is right. *From literature.*

## 2. Key findings

Numbering follows the structured summary. Quotes are from the arXiv HTML
version I fetched.

| # | Finding (short) | Cited | Verdict | Evidence and notes |
|---|---|---|---|---|
| 1 | A DEM is a list of independent mechanisms; same detectors with different observables caps distance at 2 | StimDocs | supported | Doc, Semantics: "A detector error model is a list of independent error mechanisms", each with "probability", "symptoms" and "frame changes"; same symptoms with different frame changes "guarantees the error correcting code has distance at most 2". |
| 2 | `decompose_errors` splits into ≤2-detector pieces or throws; `ignore_decomposition_failures` passes them on | StimDocs | supported | Docstring: "Decomposition fails (causing this method to throw) if it's not possible to decompose large errors into simple errors that affect at most two detectors"; with the flag "the undecomposed error is inserted into the output". Not mentioned in the review: by default Stim may introduce "remnant edges", which the docstring warns "can also reduce the effective code distance". |
| 3 | Stim uses frames against one reference sample; d=100 in 15 s, then 1 kHz | Gidney2021 | supported | Abstract: analyses "a distance 100 surface code circuit ... in 15 seconds and then begin sampling full circuit shots at a rate of 1 kHz"; tableau simulation only "to create an initial reference sample", then "batches of Pauli frames". |
| 4 | Sparse blossom: exact MWPM on graphlike models, weights log((1-p)/p), about linear, <1 µs per round at d=17, p=0.1% | HiggottGidney2025 | supported | "we will restrict our attention to graphlike error models"; weight "log((1-p[i])/p[i])"; "less than one microsecond per round" for both bases; running time "roughly linear in the number of nodes". |
| 5 | PyMatching ignores undecomposed >2-detector errors, merges parallel edges as independent, keeps the first-added observables | PyMatchingDocs | supported | `from_detector_error_model` docstring says exactly this. The distance-2 justification is in the `add_edge` docstring ("the code has distance 2 if parallel edges have different fault_ids anyway"). |
| 6 | Correlated matching gives about a factor-2 gain for about twice the work; PyMatching 2.3.0 has it at 2-3x runtime | Fowler2013, PyMatchingDocs | partially supported | Cost is right: "approximately a factor of 2 slower processing due to the need to perform approximately twice as much matching". The factor-2 "improvement" is the ratio of d=3 to d=5 logical error rates (95 to 188 at p=1e-4, 60 to 112 at 2e-4), not a twofold lower logical error rate. The conclusion claims rates "a factor of 2^{d/2} lower". Release notes confirm the 1310.0863 method and "around 2x-3x slower". |
| 7 | Belief-matching: BP on the full DEM, posteriors as weights; threshold 0.82% to 0.94%; about linear expected runtime | Higgott2023 | supported | BP stage "uses knowledge of the full error model, including the hyperedge error mechanisms"; "posterior marginal probabilities ... set the edge weights"; "from 0.82% (for MWPM) to 0.94%"; expected runtime "approximately linear" below threshold. |
| 8 | Union-find O(n α(n)); weight ≤ (d-1)/2 and d-1 erasures; 9.9% and 2.6% | DelfosseNickerson2021 | supported | Abstract states all four. The 2.6% is for phenomenological, not circuit-level, noise. |
| 9 | BP-OSD gains several orders of magnitude; BP+OSD general, 9.9% toric threshold near MWPM | PanteleevKalachev2021, Roffe2020 | supported | "can be improved by several orders of magnitude"; "a general decoder for quantum low density parity check codes constructed from the hypergraph product"; "9.9±0.2%", "approaches ... minimum weight perfect matching" (10.3%). The 9.9% is code-capacity bit-flip noise with OSD-CS; OSD-0 gives 9.2%. |
| 10 | OSD inversion cubic; LSD local; BP+LSD on par with BP+OSD at circuit level; in ldpc v2 | Hillmann2025 | supported | "cubic worst-case time complexity"; "performs on par with BP+OSD"; circuit-level depolarizing noise on surface, HGP and bivariate-bicycle codes; "serial implementation ... available open-source as part of the LDPCv2 package". |
| 11 | Degenerate ML decoding #P-hard; MLE NP-complete | IyerPoulin2013 | supported | Theorem 5.1: DQMLD is #P-complete up to polynomial-time Turing reduction. The NP-completeness of quantum MLE is stated with citations to Hsieh and Le Gall (2011) and Kuo and Lu (2012), not proved there. |
| 12 | Exact ML O(n²) for independent X/Z; MPS O(nχ³) beats matching at χ≥4; TN extends to circuit level | BravyiSucharaVargo2014, Piveteau2024 | supported | BSV abstract states all three points, including the "special noise model with independent bit-flip and phase-flip errors". Piveteau abstract: "tensor network decoding can be applied to circuit-level noise and ... outperforms the matching decoder on the rotated surface code"; "decoding is to be performed offline". |
| 13 | Coset recursion over 4\|S\| cosets; tractable to d=5; +25% (independent X/Z) and 1.7-2x (correlated) thresholds | Heim2016 | supported | "a 4\|S\|×4\|S\| matrix"; resources "limit the simulation ... to a distance five surface code"; p_c 0.28% (MWPM) vs 0.35% (ML). Nuance: the 1.7-2x is the authors' estimate (optimal "between 1.5% and 1.8%" vs about 0.9%), and their correlated-noise ML run tracked separate X and Z marginals, so it was not fully exact. |
| 14 | Transversal gates create order-3 hyperedges not decomposable into existing edges; MLE or belief-HUF; O(d) to O(1) rounds; MWPM needs d rounds | Cain2024 | supported | "this hyperedge has order three ... and cannot be decomposed into existing lower-order edges"; mixed-integer program with a solver (Gurobi cited); "belief-HUF"; "from O(d) to O(1)"; "MWPM can be applied at the expense of d syndrome extraction rounds". The hyperedge comes from measurement errors next to transversal CNOTs. |
| 15 | Fold-transversal S, H and CNOT in the unrotated code give >2-detector measurement errors; logical-observable matching; sublinear windowed failures | SerraPeralta2026 | supported | "single measurement errors can flip more than two detectors ... mwpm ... cannot be directly applied"; lom "applies mwpm only to matchable subgraphs"; windowed errors "scale sublinearly in d"; basic windowed decoder needs "quiescent (slow) resets". |
| 16 | Tesseract: A* MLE on any Stim DEM, exact without cutoffs, about 5x faster than IP, pip and sinter | Beni2025 | supported | Heuristic "is admissible, so Tesseract maintains its correctness guarantee"; at-most-two pruning and beam cutoff are the inexact options; "at p≤0.001 ... nearly identical to that of the integer program decoder ... approximately five times faster". README: `pip install tesseract-decoder`, "Using Tesseract with Sinter". Nuances: the 5x holds at p ≤ 1e-3 on topological codes, and the IP baseline used HiGHS. |
| 17 | [[7,1,3]] ions: real-time two-stage lookup tables, a few QASM if-statements, Pauli-frame or physical corrections | RyanAnderson2021 | supported | "two steps using two different lookup tables"; "a few if-statements in QASM"; corrections "either as software updates to the Pauli frame or as physically applied gates". |
| 18 | FPGA LUT filled offline, <7% logic, few-ns latency, 107x compression without loss | Das2021 | partially supported | Abstract: "<7% logic", "latency of few nanoseconds", "107x (from 148 MB to 1.38 MB) without degrading accuracy", table filled "by running a software decoder offline". Main text: "The decoding latency ... ranges between 29-42 ns". Cite the 29-42 ns figure. |
| 19 | Recurrent transformer pretrained on DEM samples, fine-tuned on device data, beats TN and correlated matching on Sycamore d=3,5; fine-tuning fixes most mismatch; 1-2 orders short of 1 µs | Bausch2024 | supported | Pretraining on "2×10^9 samples drawn from detector error noise models"; LER 2.901% and 2.748% vs TN 3.028% and 2.915%, larger gap to MWPM-Corr; "fine-tuning compensates for most of the mismatch"; throughput "within about one to two orders of magnitude of the target throughput rate of 1µs". The soft-readout and leakage gains (review F23) come from simulated data, not Sycamore. |
| 20 | Sliding window commits only the commit region, buffer about d, sequential; parallel windows lose no fidelity | Skoric2023 | supported | "Only the correction edges in the commit region are taken as final"; "n_buf=d"; "Sliding window decoding is inherently sequential"; Fig. 4 caption: "no numerically significant drop in logical fidelity" (phenomenological noise, 2%). |
| 21 | Exclusive decoders: up to quadratic gain; abort rate can fall with d | Smith2024 | supported | Abstract: "up to a quadratic improvement in logical failure rates below threshold"; "with a modest exclusion criterion ... the exclusion rate decays with code distance". |
| 22 | Data-estimated weights track drift if τ_env > δt/p̄ | Spitz2018 | supported | Abstract states this condition for matching weights estimated "in the absence of an error model". |
| 23 | Every non-trivial [[5,1,2]] syndrome has two first-order explanations differing by a logical; LRB-C linear in p, LRB-D O(p²) | DelfosseNickerson2021 | partially supported | Alias structure: **verified analytically** with Stim Pauli algebra. Z2Z3 gives X-syndrome (1,1), Z1·Z2Z3 ≃ Z_L, X1·X2X3 ≃ X_L, X2X4 ≃ X_L, Z3Z4 ≃ Z_L. Qualification: the (1,1) alias needs a CZ(2,3) or SWAP(2,3), so only intervals with a logical S or H have it. In an all-identity model circuit, interior (1,1) events had no logical-flipping first-order alternative (**verified numerically**). "LRB-C linear in p" fits my first-order numbers (section 3.3). "LRB-D O(p²)" is **wrong** under the CONTEXT readout: there is an undetectable first-order logical floor of 4p + 8p/15 (section 3.3). |
| 24 | Logical S maps S_X1 to S_X1·S_Z1 and S_X2 to S_X2·S_Z2; H swaps sectors; toy gives D0 D4 D6, decomposed as D0 ^ D4 D6 | SerraPeralta2026, Cain2024, StimDocs | supported | Tableau check: S_L(S_X1) = -S_X1·S_Z1, S_L(S_X2) = -S_X2·S_Z2, Z checks fixed, H_L swaps S_X1↔S_Z1 and S_X2↔S_Z2, X_L maps to Y0 Z2 X3 under S (**verified numerically**). Re-running the review's toy reproduces `error(0.00598963) D0 D4 D6` and `D0 ^ D4 D6` (**verified numerically**). SerraPeralta2026 describes the same weight-three edge for a Z-check measurement error next to an S gate. |
| 25 | First-added rule makes naive MWPM order-dependent; minimal DEM flips; toy 6/10/11 sets and 1/1/2 wrong; fix by pre-merging | PyMatchingDocs | supported | My own script reproduces the flip (merged p = 0.26, fault_ids {0} vs {}). Re-run of the review's script reproduces 24/6/1, 25/10/1, 29/11/2 (**verified numerically**). Two caveats on the fix. `add_edge(..., error_probability=q)` without `weight` sets weight 1.0, so pass `weight=log((1-q)/q)` explicitly. `enable_correlations=True` shows the same order dependence on the minimal DEM. |
| 26 | Exact ML affordable: 64 cosets, 92 detectors at m=22; coset recursion or time-ordered elimination; reference decoder | Heim2016, Das2021 | supported | Counts: 2^4 = 16 syndromes × 4 classes = 64 = 4\|S\|; 23 noisy rounds × 4 = 92 detectors (**verified analytically**). Feasibility is still a conjecture in the review, but `src/lrb_qubit/mld.py` already implements an exact time-slot transfer-matrix MLD with 32 numbers per shot, which corroborates it. |
| 27 | Measurement and reset faults are time-like pairs with no observable except in the final round, about 31p per round; posterior ≥ 1-ε abort keeps O(p²) and rejects far fewer shots than LRB-D unif=1 | Smith2024, Meister2024 | partially supported | Signature: in an all-identity depth-4 model circuit no logical-flipping mechanism shares a time-like pair set (**verified numerically**). Rate: 31p is the measurement-plus-reset part. Idle ancillas add about 8p and gate faults about 1.3p, and the DEM gives 40.0p per interior round and 32.1p for the first pair (**verified numerically**, matches the hand count). "Main source of rejection" is about 40% of first-order detections, the largest single class but not most. "Far fewer rejections": at depth 6, ambiguous sets carry 18-41% of first-order detection probability, so roughly 2.4-5.6x fewer rejections than unif=1 (**verified numerically**, first order). "Keeps O(p²)": false with the CONTEXT readout because of the 4.5p undetectable floor; plausible with the all-data readout. |
| 28 | LRB decodable offline with a Pauli frame; LUTs if real time needed; priors from data; d=2 decisions fragile to prior errors | RyanAnderson2021, Das2021, Bausch2024, Spitz2018 | partially supported | Each cited piece checks out (rows 17, 18, 19, 22). Offline decoding follows from the protocol having no feed-forward. The fragility part is half right. Interior ambiguous classes are nearly equal (ratio 1.00-1.06), so tiny prior errors flip decisions, but either choice then fails about half the time, so the cost is small. Final-round classes differ by about 3x and would survive ±10% prior errors (**verified numerically**). |
| 29 | Check-free inverse half seen only by final detectors; failure not a product of per-step factors; LRB-C may decay non-exponentially | Skoric2023 | not checkable | **Conjecture.** Skoric2023 supports only the generic commit-and-buffer idea. The conjecture is plausible: with a final-only syndrome, F(m) = Σ_s \|P(s,+) - P(s,-)\|, a sum of absolute values of exponential mixtures, which is not a single exponential in general (**verified analytically**, form only). |

## 3. My numerical checks

All scripts are in
`<local scratch, not kept>`.
Model circuits come from the lead session's `src/lrb_qubit/circuits.py`
(`build_lrb_circuit`), used read-only, at p = 1e-3. Its final check round
has no ancilla reset, as in the qutrit generator.

### 3.1 PyMatching parallel edges (`pm_parallel.py`)

**Verified numerically** (pymatching 2.3.1):

- `error(0.1) D0 L0`, then `error(0.2) D0`: one boundary edge, p = 0.26,
  fault_ids {0}, prediction 1.
- Same two lines swapped: p = 0.26, fault_ids {}, prediction 0.
- Same pair with `enable_correlations=True`: predictions 1 and 0, so the
  correlated mode does not remove the order dependence here. This answers
  open question 8 for the minimal case only.
- `add_edge(0, 1, fault_ids=0, error_probability=0.1)` stores weight 1.0.
  With `weight=log(0.9/0.1)` it stores 2.197.

### 3.2 Re-run of the review's toy (`toy_dem_check.py`, `parallel_check.py`, `toy_hyper.py`)

I read both scripts, then ran them. **Verified numerically:** 74, 85 and 86
mechanisms with 30, 34 and 37 flipping more than two detectors;
`decompose_errors=True` succeeds for I, H and S; 24/6/1, 25/10/1 and
29/11/2 for the conflict counts; with S the round-1 Z1 measurement error is
`error(0.00598963) D0 D4 D6`, decomposed as `D0 ^ D4 D6`
(p = 0.00573) plus `D4 D6 ^ D0` (p = 0.000267).

### 3.3 Model circuits (`lrbq_checks.py`, `lrbq_classes.py`, `lrbq_split.py`, `alg_checks.py`)

Undetectable logical faults. With `terminal="data2"` (CONTEXT readout of
data 0 and 3) the DEM has one mechanism that flips L0 and no detector, of
probability 4.518e-3 = 4.52p, at depths 0, 1 and 4. With
`terminal="data5"` there is none. **Verified numerically.** The hand count
matches. In the final round, DEPOLARIZE1(2p) hits data 0 during the M7 and
M8 layers and data 3 during the M8 layer, after their last X-check
coupling, and each Z or Y component flips the readout: 3 × 4p/3 = 4p. The
DEPOLARIZE2 after CNOT 7->0 and CNOT 8->3 adds {Z, Y} on the data times {I,
X} on the ancilla, 2 × 4p/15. Total 4p + 8p/15 ≈ 4.53p. **Verified
analytically.** A reset layer at the end of the final round would add
another 8p/3.

Time-like pairs. For an all-identity depth-4 circuit the summed
probability of same-check adjacent-round pairs without L0 is 32.1p for
rounds 0->1 and 40.0p for each later pair. No L0-flipping mechanism has the
same detector set. **Verified numerically.** By hand: measurement and reset
give 4 × (5p + 2p/3 + 2p) = 30.7p, ancillas idling in |0> during earlier
measurement layers give 4p/3 + 8p/3 + 4p = 8p, and gate faults that flip
only the ancilla give about 1.3p, for 40.0p. The first pair lacks the 8p
reset term. **Verified analytically.**

Ambiguous classes. All-identity, depth 4: interior single S_X1 events have
3.66p without a logical flip vs 3.82p with one, and S_X2 events 3.79p vs
3.79p. Final-round single S_X1 events have 15.6p vs 5.5p, S_X2 events 19.6p
vs 6.0p. Interior (1,1) X-syndromes are not ambiguous. With all-S or all-H
words many more detector sets carry both classes, with ratios from 1.0 to
about 100. **Verified numerically.**

First-order split at depth 6, three random words plus all-identity. The
first-order detection probability is 659p to 694p. The share in ambiguous
detector sets is 18-21% for all-identity and 32-41% for random words.
Summing the smaller class over ambiguous sets gives the first-order failure
of an exact ML decoder, 53p to 95p, so LRB-C is linear in p and grows with
depth. **Verified numerically** (first order only, no sampling).

Decomposition. `decompose_errors=True` succeeded on 24 of 24 random model
circuits, depths 2, 6, 10 and 22, both readouts. This partly answers the
review's open question 1. **Verified numerically.**

## 4. Recommended edits to 03-decoders.md

1. Summary bullet 2, I1 and I9: drop "LRB-D has O(p²) error on accepted
   shots". Under the CONTEXT readout both LRB-D and every decoder have a
   depth-independent first-order floor of 4p + 8p/15 from final-round
   faults on data 0 and 3. O(p²) needs the all-data readout of I8, so I8 is
   a precondition for the I9 comparison, not an optional design tweak.
2. I1: qualify "every non-trivial syndrome". The (1,1) alias needs a
   logical S or H in the interval; the (1,0) and (0,1) aliases are always
   present.
3. I9: replace "≈ 31p per round, plus idle-ancilla terms" with "≈ 40p per
   interior round (31p measurement and reset, 8p idle ancillas, 1.3p gate
   faults)". Call time-like faults the largest class of rejections, about
   40%, not the main source.
4. I4: write the fix as `add_edge(u, v, fault_ids=best,
   weight=log((1-q)/q), error_probability=q, merge_strategy=...)`. Note that
   correlated mode keeps the order dependence in the minimal test. Note too
   that the interior Z0/Z2 and Z3/Z4 classes are nearly equal, so order
   effects matter most for the unequal sets around S, H and the final round.
5. F6: say that the factor of 2 is the improvement in the d=3 to d=5 error
   ratio, with logical error rates about 2^{d/2} lower at large d.
6. F22: add the 29-42 ns decoding latency from the main text.
7. Bibliography: add the journal versions of Cain2024, IyerPoulin2013 and
   Das2021, and the journal title and volume of Spitz2018.
8. Minor: F11 is code-capacity noise; F13's NP-completeness is cited, not
   proved, by Iyer and Poulin; F20's 5x holds at p ≤ 1e-3 against a HiGHS
   integer program; F23's soft and leakage gains are from simulated data.

## 5. Other references in the review's search log

The seven "fetched but not used" references all exist with the stated
titles and authors: Pryadko 1909.06732 (Quantum 4, 304), Cao et al.
2501.03582, Varbanov et al. 2307.03280 (PRR 7, 013029 confirmed via
Crossref), Baireuther et al. 1705.07855 (Quantum 2, 48), Senior et al.
2512.07737 (24 authors), Wu et al. 2508.04969, Cao, Yan, Du and Pan
2605.17230. *From literature (arXiv abstract pages, Crossref).* I did not
check the leads the review marked UNVERIFIED.
