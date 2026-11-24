# Lit review 03: decoders under circuit-level noise for small codes

Scope: decoders that could turn detection-only LRB (LRB-D) on the qubit
`[[5,1,2]]` code into LRB with correction (LRB-C). Covered: minimum-weight
perfect matching (PyMatching, sparse blossom), correlated matching and
belief-matching, union-find, BP+OSD and BP+LSD (ldpc package), exact and
tensor-network maximum-likelihood (ML) decoding, most-likely-error (MLE)
search, lookup tables, neural decoders, sliding-window decoding, decoding
across transversal logical gates, abort-capable ("exclusive") decoders, and
Stim detector error models (DEMs). For each decoder: what it needs from the
DEM (graphlike or hyperedges), cost, optimality, and Python availability.

Labels used: **from literature** (with key), **verified analytically**,
**verified numerically**, **conjecture**. Every source below was fetched by me
on 2026-10-28. "abs" means I read the arXiv abstract page (title, authors, date
and IDs confirmed there); "full" means I also read relevant sections of the
arXiv HTML or ar5iv text; "xref" means the journal reference was confirmed via
the Crossref API; "pkg" means I also read the docstrings of the package
installed in this environment (`module load python/3.11`: stim 1.15.0,
pymatching 2.3.1, ldpc 2.4.1). Numbers quoted from full texts are from the
version I read.

---

## 0. Summary

- **Verified analytically.** At distance 2 no decoder can correct all single
  faults. Each ambiguous syndrome has two first-order explanations that differ
  by a logical operator. This also holds for syndrome (1,1): a two-qubit fault
  `Z2 Z3` (or `X2 X3`) from the logical CZ(2,3) or SWAP(2,3) gives the same
  syndrome as `Z1` (`X1`), and the two differ by `Z_L` (`X_L`). Section 3, item I1.
- **Conjecture.** Correction on this code therefore gives a logical error
  linear in p with full acceptance. Postselection gives O(p²) error with
  rejection linear in p. The useful middle ground is a decoder that reports
  its confidence and aborts only on ambiguous syndromes (Smith2024). It would
  accept harmless measurement and reset faults, which LRB-D rejects. Items I1
  and I9.
- **From literature, plus verified numerically on a toy circuit.** PyMatching
  needs graphlike errors. It also keeps the observable of the *first-added*
  parallel edge when merging (PyMatchingDocs). So on exactly the ambiguous
  decisions that matter at d=2, naive MWPM depends on the order of the DEM,
  not on the probabilities. Item I4.
- **Verified analytically and numerically (toy).** The fold-transversal
  logical S turns a Z-check measurement error into a three-detector
  hyperedge, as described for fold-transversal gates in SerraPeralta2026. In a
  two-round toy, Stim still decomposed it into graphlike pieces. Item I3.
- **Conjecture.** Exact ML is cheap for a 5-qubit code (16 syndromes × 4
  logical classes = 64 cosets; Heim2016). It should be the reference decoder.
  It also gives the posterior needed for abort rules. Items I5 and I6.
- **Conjecture.** On hardware, LRB circuits are Clifford-only, so decoding
  can be offline with Pauli-frame bookkeeping, as in the real-time frame
  updates of RyanAnderson2021. No real-time decoder is required. Item I11.

---

## 1. Bibliography

### 1a. Core sources (21)

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched | Read | Relevance |
|---|---|---|---|---|---|---|---|---|
| Gidney2021 | Stim: a fast stabilizer circuit simulator | C. Gidney | 2021 | arXiv:2103.02202; DOI 10.22331/q-2021-07-06-497 (Quantum 5, 497) | https://arxiv.org/abs/2103.02202 | yes | abs | Q1: sampling engine behind DEMs |
| HiggottGidney2025 | Sparse Blossom: correcting a million errors per core second with minimum-weight matching | O. Higgott, C. Gidney | 2023 (arXiv), 2025 (Quantum) | arXiv:2303.15933; DOI 10.22331/q-2025-01-20-1600 (Quantum 9, 1600) | https://arxiv.org/abs/2303.15933 | yes | abs + full (HTML) | Q1: MWPM in PyMatching v2 |
| Fowler2013 | Optimal complexity correction of correlated errors in the surface code | A. G. Fowler | 2013 | arXiv:1310.0863 (no journal ref on arXiv page) | https://arxiv.org/abs/1310.0863 | yes | abs + full (ar5iv) | Q1: correlated (two-pass) matching |
| Higgott2023 | Improved decoding of circuit noise and fragile boundaries of tailored surface codes | O. Higgott, T. C. Bohdanowicz, A. Kubica, S. T. Flammia, E. T. Campbell | 2022 (arXiv), 2023 (PRX) | arXiv:2203.04948; DOI 10.1103/PhysRevX.13.031007 | https://arxiv.org/abs/2203.04948 | yes | abs + full (HTML) | Q1: belief-matching, belief-find |
| DelfosseNickerson2021 | Almost-linear time decoding algorithm for topological codes | N. Delfosse, N. H. Nickerson | 2017 (arXiv), 2021 (Quantum) | arXiv:1709.06218; DOI 10.22331/q-2021-12-02-595 (Quantum 5, 595) | https://arxiv.org/abs/1709.06218 | yes | abs | Q1: union-find |
| PanteleevKalachev2021 | Degenerate quantum LDPC codes with good finite length performance | P. Panteleev, G. Kalachev | 2019 (arXiv), 2021 (Quantum) | arXiv:1904.02703; DOI 10.22331/q-2021-11-22-585 (Quantum 5, 585) | https://arxiv.org/abs/1904.02703 | yes | abs | Q1: BP-OSD |
| Roffe2020 | Decoding across the quantum low-density parity-check code landscape | J. Roffe, D. R. White, S. Burton, E. T. Campbell | 2020 | arXiv:2005.07016; DOI 10.1103/PhysRevResearch.2.043423 (PRR 2, 043423) | https://arxiv.org/abs/2005.07016 | yes | abs | Q1: BP+OSD as a general decoder, ldpc package |
| Hillmann2025 | Localized statistics decoding for quantum low-density parity-check codes | T. Hillmann, L. Berent, A. O. Quintavalle, J. Eisert, R. Wille, J. Roffe | 2024 (arXiv), 2025 (Nat. Commun.) | arXiv:2406.18655; DOI 10.1038/s41467-025-63214-7 (Nat. Commun. 16, 8214) | https://arxiv.org/abs/2406.18655 | yes | abs + full (HTML v2) | Q1: BP+LSD |
| IyerPoulin2013 | Hardness of decoding quantum stabilizer codes | P. Iyer, D. Poulin | 2013 | arXiv:1310.3235 (no journal ref on arXiv page) | https://arxiv.org/abs/1310.3235 | yes | abs | Q1: complexity of ML vs MLE |
| BravyiSucharaVargo2014 | Efficient algorithms for maximum likelihood decoding in the surface code | S. Bravyi, M. Suchara, A. Vargo | 2014 | arXiv:1405.4883; DOI 10.1103/PhysRevA.90.032326 (PRA 90, 032326) | https://arxiv.org/abs/1405.4883 | yes | abs | Q1: exact and MPS-approximate ML |
| Piveteau2024 | Tensor network decoding beyond 2D | C. Piveteau, C. T. Chubb, J. M. Renes | 2023 (arXiv), 2024 (PRX Quantum) | arXiv:2310.10722; DOI 10.1103/PRXQuantum.5.040303 (PRX Quantum 5, 040303) | https://arxiv.org/abs/2310.10722 | yes | abs | Q1: tensor-network ML under circuit-level noise |
| Heim2016 | Optimal circuit-level decoding for surface codes | B. Heim, K. M. Svore, M. B. Hastings | 2016 | arXiv:1609.06373 (no journal ref on arXiv page) | https://arxiv.org/abs/1609.06373 | yes | abs + full (ar5iv) | Q1: exact circuit-level ML for small codes |
| Cain2024 | Correlated decoding of logical algorithms with transversal gates | M. Cain, C. Zhao, H. Zhou, N. Meister, J. P. Bonilla Ataides, A. Jaffe, D. Bluvstein, M. D. Lukin | 2024 | arXiv:2403.03272 (no journal ref on arXiv page) | https://arxiv.org/abs/2403.03272 | yes | abs + full (HTML v2) | Q1, Q3: decoding across transversal gates, MLE |
| SerraPeralta2026 | Decoding across transversal Clifford gates in the surface code | M. Serra-Peralta, M. H. Shaw, B. M. Terhal | 2025 (arXiv), 2026 (PRX Quantum) | arXiv:2505.13599; DOI 10.1103/sk5y-25b1 (PRX Quantum 7, 010335) | https://arxiv.org/abs/2505.13599 | yes | abs + full (HTML, first third) + xref | Q1: fold-transversal S and H, logical-observable matching |
| Beni2025 | Tesseract: A search-based decoder for quantum error correction | L. Aghababaie Beni, O. Higgott, N. Shutty | 2025 | arXiv:2503.10988 (code: github.com/quantumlib/tesseract-decoder) | https://arxiv.org/abs/2503.10988 | yes | abs + full (HTML v2) + GitHub README | Q1: MLE on arbitrary DEMs |
| RyanAnderson2021 | Realization of real-time fault-tolerant quantum error correction | C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, A. Hankin, et al. (17 authors) | 2021 | arXiv:2107.07505; DOI 10.1103/PhysRevX.11.041058 (PRX 11, 041058, via Crossref) | https://arxiv.org/abs/2107.07505 | yes | abs + full (ar5iv) + xref | Q3, Q1: real-time lookup-table decoding on trapped ions |
| Das2021 | LILLIPUT: A lightweight low-latency lookup-table based decoder for near-term quantum error correction | P. Das, A. Locharla, C. Jones | 2021 | arXiv:2108.06569 | https://arxiv.org/abs/2108.06569 | yes | abs | Q3: FPGA lookup tables |
| Bausch2024 | arXiv: Learning to decode the surface code with a recurrent, transformer-based neural network. Nature: Learning high-accuracy error decoding for quantum processors | J. Bausch, A. W. Senior, F. J. H. Heras, T. Edlich, A. Davies, et al. (18 authors) | 2023 (arXiv), 2024 (Nature) | arXiv:2310.05900; DOI 10.1038/s41586-024-08148-8 (Nature 635, 834-840) | https://arxiv.org/abs/2310.05900 | yes | abs + full (HTML, first ~55%) + xref | Q1, Q2, Q3: neural decoding ("AlphaQubit") |
| Skoric2023 | Parallel window decoding enables scalable fault tolerant quantum computation | L. Skoric, D. E. Browne, K. M. Barnes, N. I. Gillespie, E. T. Campbell | 2022 (arXiv), 2023 (Nat. Commun.) | arXiv:2209.08552; DOI 10.1038/s41467-023-42482-1 (Nat. Commun. 14, 7040) | https://arxiv.org/abs/2209.08552 | yes | abs + full (HTML) + xref | Q1, Q3: sliding and parallel windows |
| Smith2024 | Mitigating errors in logical qubits | S. C. Smith, B. J. Brown, S. D. Bartlett | 2024 | arXiv:2405.03766; DOI 10.1038/s42005-024-01883-4 (Commun. Phys. 7, 386) | https://arxiv.org/abs/2405.03766 | yes | abs | Q1: exclusive (abort-capable) decoders |
| Spitz2018 | Adaptive weight estimator for quantum error correction | S. T. Spitz, B. Tarasinski, C. W. J. Beenakker, T. E. O'Brien | 2017 (arXiv), 2018 (Adv. Quantum Technol.) | arXiv:1712.02360; DOI 10.1002/qute.201800012 | https://arxiv.org/abs/1712.02360 | yes | abs | Q2, Q3: decoder weights from data under drift |

### 1b. Software documentation (3)

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched | Read | Relevance |
|---|---|---|---|---|---|---|---|---|
| StimDocs | "The Detector Error Model File Format (.dem)" (doc/file_format_dem_detector_error_model.md) and the `stim.Circuit.detector_error_model` docstring | quantumlib/Stim developers | accessed 2026-10-28 (stim 1.15.0 installed) | none | https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md | yes | docs + pkg | Q1: DEM semantics, decomposition |
| PyMatchingDocs | PyMatching v2.3.0 and v2.3.1 release notes; `Matching.from_detector_error_model` and `Matching.add_edge` docstrings | O. Higgott and contributors (correlations: S. Madhuk, PR #156, per the notes) | 2025 (GitHub releases v2.3.0 13 Aug, v2.3.1 25 Sep; PyPI upload of 2.3.0 on 2025-08-10) | none | https://github.com/oscarhiggott/PyMatching/releases/tag/v2.3.0 | yes | docs + pkg | Q1: correlated matching, parallel-edge rule |
| ldpcDocs | ldpc (LDPCv2) README and the installed ldpc 2.4.1 module tree and docstrings | J. Roffe and contributors | accessed 2026-10-28 | none | https://github.com/quantumgizmos/ldpc | yes | docs + pkg | Q1: BP+OSD, BP+LSD, UF, window decoders in Python |

### 1c. Secondary sources (5; fetched and verified, used only in the decoder table, implications, or seed bookkeeping)

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched | Read | Relevance |
|---|---|---|---|---|---|---|---|---|
| Higgott2022 | PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching | O. Higgott | 2021 (arXiv), 2022 (ACM TQC) | arXiv:2105.13082; DOI 10.1145/3505637 (ACM Trans. Quantum Comput. 3(3), via Crossref) | https://arxiv.org/abs/2105.13082 | yes | abs + xref | seed; describes PyMatching v0.x (local matching), superseded by HiggottGidney2025 for v2 |
| Chubb2021 | General tensor network decoding of 2D Pauli codes | C. T. Chubb | 2021 | arXiv:2101.04125 (no journal ref on arXiv page) | https://arxiv.org/abs/2101.04125 | yes | abs | seed; code-capacity TN decoder, O(n log n + nχ³), SweepContractor.jl (Julia) |
| Huang2020 | Fault-tolerant weighted union-find decoding on the toric code | S. Huang, M. Newman, K. R. Brown | 2020 | arXiv:2004.04693; DOI 10.1103/PhysRevA.102.012419 (PRA 102, 012419) | https://arxiv.org/abs/2004.04693 | yes | abs | Q1: union-find under circuit-level noise |
| TomitaSvore2014 | Low-distance surface codes under realistic quantum noise | Y. Tomita, K. M. Svore | 2014 | arXiv:1404.3747; DOI 10.1103/PhysRevA.90.062320 (PRA 90, 062320) | https://arxiv.org/abs/1404.3747 | yes | abs + full (ar5iv) | Q1: lookup table for d=3 with a 3-round window |
| Meister2024 | Efficient soft-output decoders for the surface code | N. Meister, C. A. Pattison, J. Preskill | 2024 | arXiv:2405.07433 | https://arxiv.org/abs/2405.07433 | yes | abs | Q1: soft output from MWPM and UF, flagging runs to discard |

---

## 2. Key findings (all **from literature**; one sentence each)

DEMs and simulation

- **F1** [Q1] Stim runs one tableau simulation to get a reference sample and
  then propagates batches of Pauli frames against it, and reports analysing a
  distance-100 surface-code circuit in about 15 s and then sampling full shots
  at about 1 kHz. [Gidney2021] *From literature.*
- **F2** [Q1] A Stim DEM is a list of independent error mechanisms, each with
  a probability, the detectors it flips and the observables it flips. Two
  mechanisms with the same detectors but different observables cap the code
  distance at 2. [StimDocs] *From literature (documentation).*
- **F3** [Q1] With `decompose_errors=True` Stim splits composite mechanisms
  (for example a Y error into X and Z parts) into suggested pieces that each
  flip at most two detectors and throws if it cannot. With
  `ignore_decomposition_failures=True` it instead leaves undecomposed
  mechanisms for the downstream decoder to handle. [StimDocs] *From
  literature (documentation).*

Matching family

- **F4** [Q1] Sparse blossom, the algorithm behind PyMatching v2, solves exact
  minimum-weight perfect matching on a graphlike detector graph (every
  mechanism flips at most two detectors) with weights log((1-p)/p). It runs
  in empirically almost-linear time below threshold and handles both bases of
  distance-17 surface-code circuits at 0.1% noise in under 1 µs per round on
  one core. [HiggottGidney2025] *From literature.*
- **F5** [Q1] When loading a DEM, PyMatching ignores mechanisms with more than
  two detectors that have no suggested decomposition, merges parallel edges
  as independent mechanisms, and keeps the observables of the first-added
  parallel edge. Its docs justify this by saying parallel edges with different
  observables already mean distance 2. [PyMatchingDocs] *From literature
  (documentation).*
- **F6** [Q1] Correlated matching matches once, raises the probabilities of
  errors correlated with the matched edges (for example the other half of a Y
  error), and matches again. The paper reports roughly a factor-2 improvement
  in the fault-tolerant surface-code case for about twice the matching work.
  [Fowler2013] *From literature.*
- **F7** [Q1] PyMatching 2.3.0 (2025) implements this two-pass method from
  arXiv:1310.0863 via `enable_correlations=True`, using hyperedges that have
  been decomposed into graphlike parts, at about 2-3× the runtime of plain
  matching. Version 2.3.1, the installed one, fixed bugs in it.
  [PyMatchingDocs, Fowler2013] *From literature (documentation).*
- **F8** [Q1] Belief-matching runs belief propagation on the full circuit-level
  DEM, hyperedges included, and uses the posterior marginals as matching
  weights. This raised the circuit-level surface-code threshold from about
  0.82% (MWPM) to about 0.94%, with roughly linear expected runtime below
  threshold. [Higgott2023] *From literature.*

Union-find

- **F9** [Q1] Union-find runs in O(n α(n)) worst-case time and provably
  corrects every error of weight up to (d-1)/2 and up to d-1 erasures, with
  toric-code thresholds of 9.9% (perfect syndromes) and 2.6% (faulty
  syndromes). [DelfosseNickerson2021] *From literature.*

Belief propagation family

- **F10** [Q1] Adding ordered-statistics post-processing to belief propagation
  (BP-OSD) improves plain BP by several orders of magnitude on many degenerate
  quantum LDPC codes. [PanteleevKalachev2021] *From literature.*
- **F11** [Q1] BP+OSD works as a general decoder across hypergraph-product
  code families and reaches a 9.9 ± 0.2% toric-code threshold, approaching
  MWPM. [Roffe2020] *From literature.*
- **F12** [Q1] OSD's matrix inversion is cubic in the check-matrix size in the
  worst case, while LSD inverts only small local clusters. BP+LSD matches
  BP+OSD accuracy under circuit-level depolarizing noise on surface and
  bivariate-bicycle codes, and a serial implementation ships in the
  open-source ldpc v2 package. [Hillmann2025] *From literature.*

Exact ML, tensor networks and most-likely-error search

- **F13** [Q1] Optimal, degeneracy-aware decoding of stabilizer codes is
  #P-hard in general. That is harder than most-likely-error decoding, which is
  NP-complete. [IyerPoulin2013] *From literature.*
- **F14** [Q1] For the surface code with perfect syndromes, exact ML runs in
  O(n²) time for independent X/Z noise. An MPS contraction gives approximate
  ML in O(nχ³) for general Pauli noise and beats matching already at χ ≥ 4.
  [BravyiSucharaVargo2014] *From literature.*
- **F15** [Q1] Tensor-network ML decoding extends to circuit-level noise,
  outperforms matching on the rotated surface code under such noise, and is
  proposed for near-term experiments that decode offline and put accuracy
  first. [Piveteau2024] *From literature.*
- **F16** [Q1] An optimal circuit-level decoder can update the probabilities of
  all 4·|S| error-correction cosets (logical class × syndrome) cycle by cycle,
  with a transition matrix that includes gate faults and their propagation
  through the syndrome circuit. It was tractable only up to distance 5. It
  raised the threshold over MWPM by about 25% for independent X/Z noise and
  by about 1.7-2× for correlated depolarizing noise. [Heim2016] *From
  literature.*
- **F17** [Q1] Transversal entangling gates turn measurement errors just
  before the gate into order-3 hyperedges that cannot be decomposed into
  existing edges. Joint decoding cuts the noisy syndrome rounds needed between
  transversal Clifford gates from O(d) to O(1), using either MLE by
  mixed-integer programming (Gurobi) or BP followed by hypergraph union-find.
  MWPM needed d rounds. [Cain2024] *From literature.*
- **F18** [Q1] In the unrotated surface code, fold-transversal S and H and
  transversal CNOT map stabilizers to products of generators, so single
  measurement errors can flip more than two detectors and plain MWPM cannot
  decode the whole problem. "Logical-observable matching" instead runs MWPM on
  a matchable subgraph for each reliable logical observable.
  [SerraPeralta2026] *From literature.*
- **F19** [Q1] Windowed versions of logical-observable matching have failure
  patterns whose weight grows sublinearly in d, depending on circuit
  structure, and the basic windowed version needs slow resets.
  [SerraPeralta2026] *From literature.*
- **F20** [Q1] Tesseract finds the most-likely error for an arbitrary Stim DEM
  (an error may flip any number of detectors) by A* search. It stays exact
  when its beam and pruning cutoffs are off, matches integer-programming
  accuracy on topological codes at about 5× the speed, and installs with pip
  (`tesseract-decoder`, with a sinter interface). [Beni2025] *From literature.*

Lookup tables and hardware

- **F21** [Q3] A trapped-ion [[7,1,3]] color-code experiment decoded repeated
  syndrome rounds in real time with two-stage lookup tables, reduced to a few
  if-statements in QASM. Corrections were applied as Pauli-frame updates or
  physical gates. [RyanAnderson2021] *From literature.*
- **F22** [Q3] An FPGA lookup-table decoder, indexed by detection events and
  filled offline by a software decoder, used under 7% of an off-the-shelf
  FPGA's logic at a latency of a few nanoseconds. Storing only probable events
  cut table memory 107× (148 MB to 1.38 MB) without loss of accuracy.
  [Das2021] *From literature.*

Neural decoders

- **F23** [Q1, Q3] A recurrent transformer decoder pretrained on DEM samples
  and fine-tuned on device data beat tensor-network and correlated-matching
  decoders on Sycamore distance-3 and -5 data. Soft readout and leakage flags
  improved it further, but its throughput was still one to two orders of
  magnitude short of 1 µs per round. [Bausch2024] *From literature.*
- **F24** [Q2, Q3] Pretraining that decoder on a standard noise model and then
  fine-tuning it on experimental samples compensated for most of the mismatch
  between model and device. [Bausch2024] *From literature.*

Windows, abort rules, drift

- **F25** [Q1, Q3] Sliding-window decoding makes corrections final only in a
  commit region and keeps a buffer of about d rounds, so it is inherently
  sequential. The parallel window scheme showed no numerically significant
  loss of logical fidelity against a global decoder. [Skoric2023] *From
  literature.*
- **F26** [Q1] Exclusive decoders combine correction with postselection by
  aborting on instances judged too hard. They cut logical failure rates up to
  quadratically below threshold, and under a modest exclusion criterion the
  abort rate falls with code distance. [Smith2024] *From literature.*
- **F27** [Q2, Q3] Decoder weights estimated directly from syndrome data,
  without an error model, track time-dependent noise as long as the
  environment changes more slowly than δt/p̄ (cycle time divided by the typical
  error probability per qubit per cycle). [Spitz2018] *From literature.*

### 2b. Decoder requirements at a glance

"Installed" means present in this environment's `module load python/3.11`
stack, which I checked by importing the packages (**verified numerically**, pkg).
The other cells are **from literature** with the keys shown.

| Decoder | Needs from the DEM | Cost | Optimality | Python | Keys |
|---|---|---|---|---|---|
| MWPM (sparse blossom) | graphlike only (≤2 detectors per mechanism); bigger mechanisms need a `^` decomposition or are ignored; parallel edges merged, first observable kept | worst case O(n q³ + m q²) (q = detection events; authors call it loose); empirically ~linear below threshold | exact minimum-weight solution on the graph; not ML (no degeneracy; decomposition drops correlations) | `pymatching` 2.3.1 installed | HiggottGidney2025, PyMatchingDocs |
| Correlated MWPM (two-pass) | decomposed DEM; hyperedge components become correlations | about 2× (paper), 2-3× (PyMatching) the matching cost | heuristic; better than MWPM | `enable_correlations=True`, installed | Fowler2013, PyMatchingDocs |
| Belief-matching / belief-find | full DEM for BP, decomposed graph for matching or UF | BP linear; worst case set by MWPM; expected ~linear below threshold | heuristic | belief-find: ldpc `BeliefFindDecoder` installed; belief-matching package not checked | Higgott2023, ldpcDocs |
| Union-find (weighted) | matchable syndromes in the original; ldpc's version takes any check matrix (peeling or matrix-solve mode) | O(n α(n)) | guarantee only up to weight (d-1)/2; circuit-level toric threshold 0.62% vs 0.72% for weighted matching | ldpc `UnionFindDecoder` installed | DelfosseNickerson2021, Huang2020, ldpcDocs |
| BP+OSD | any binary check matrix, so hyperedges are fine | BP linear per iteration; OSD cubic worst case | heuristic; OSD-0 picks one consistent solution | ldpc `BpOsdDecoder`, `SinterBpOsdDecoder` installed; DEM to matrices via `ldpc.ckt_noise.dem_matrices.detector_error_model_to_check_matrices` (option `allow_undecomposed_hyperedges`) | PanteleevKalachev2021, Roffe2020, Hillmann2025, ldpcDocs |
| BP+LSD | as BP+OSD | inversion local to clusters | same accuracy as BP+OSD in benchmarks | ldpc `BpLsdDecoder`, `SinterLsdDecoder`, `LsdOverlappingWindowDecoder` installed | Hillmann2025, ldpcDocs |
| MLE by integer programming | any DEM | exponential worst case (MLE is NP-complete) | exact most-likely error | not installed; Gurobi used in Cain2024 | Cain2024, IyerPoulin2013 |
| Tesseract (A* MLE) | any Stim DEM (hyperedges fine) | exponential worst case, fast in practice; about 5× faster than integer programming | exact with cutoffs off | `pip install tesseract-decoder`; not installed | Beni2025 |
| Exact ML (coset recursion or exact contraction) | any | #P-hard in general; small for 64 cosets (I5) | optimal (degeneracy-aware) | none installed; small custom code (conjecture, I5) | Heim2016, BravyiSucharaVargo2014, Piveteau2024, IyerPoulin2013 |
| Approximate TN (MPS, sweep) | any (circuit-level shown) | O(nχ³) (2D, perfect syndromes); O(n log n + nχ³) (general 2D) | near-ML for moderate χ | Piveteau2024 code on GitHub (not checked); SweepContractor.jl is Julia | BravyiSucharaVargo2014, Chubb2021, Piveteau2024 |
| Lookup table | none at run time; the table is filled offline by any decoder or by sampling | O(1) lookup; memory exponential in syndrome bits unless compressed | as good as the decoder that filled it | trivial (dict / numpy) | TomitaSvore2014, RyanAnderson2021, Das2021 |
| Neural (recurrent transformer) | training samples (DEM or device); no DEM at inference | training cost; not yet real-time at 1 µs per round | near-optimal in reported benchmarks | none installed | Bausch2024 |
| Sliding / parallel windows | wrap any inner decoder | commit + buffer of about d rounds | no significant loss vs global in reported tests | ldpc `ckt_noise` overlapping-window decoders (BP+OSD, LSD, PyMatching) installed | Skoric2023, ldpcDocs |
| Logical-observable matching | matchable subgraph per reliable logical observable | MWPM cost per observable | fault tolerant for basic errors below d/2 on reliable observables (paper's Theorem 1) | no package checked | SerraPeralta2026 |
| Soft output / exclusive | a confidence per shot (posterior, gap, cluster geometry) | one extra pass or a by-product | trades acceptance for fidelity | exact ML gives it directly | Smith2024, Meister2024 |

---

## 3. Implications for our qubit [[5,1,2]] LRB study

**I1. Distance 2 limits every decoder.** **Verified analytically.**
Commutation with `S_Z1 = Z0 Z1 Z3`, `S_Z2 = Z1 Z2 Z4`, `S_X1 = X0 X1 X2`,
`S_X2 = X1 X3 X4` reproduces the CONTEXT syndrome table: X0, X3 → (1,0); X2,
X4 → (0,1); X1 → (1,1); Z0, Z2 → (1,0); Z3, Z4 → (0,1); Z1 → (1,1). Also
X0X3 = X_L, X2X4 = X_L·S_X1·S_X2, Z0Z2 = Z_L and Z3Z4 = Z_L·S_Z1·S_Z2. The
claim that qubit-1 errors are uniquely identifiable holds only for
single-qubit Paulis. `Z2 Z3` has syndrome (1,1) and `Z1·Z2Z3·S_Z1 = Z0 Z2 =
Z_L`. Likewise `X2 X3` has syndrome (1,1) and `X1·X2X3·S_X1 = X0 X3 = X_L`.
Both `Z2 Z3` and `X2 X3` are first-order faults of the DEPOLARIZE2 after the
logical CZ(2,3) and SWAP(2,3). So every non-trivial syndrome has two
first-order explanations that differ by a logical operator. The "corrects
weight ≤ (d-1)/2" guarantees of MWPM and UF (DelfosseNickerson2021) give
nothing at d = 2.
**Conjecture.** On each ambiguous syndrome an optimal decoder still fails with
probability min(p_a, p_b)/(p_a + p_b) at first order. So LRB-C keeps every
shot but has a logical error linear in p, while LRB-D has O(p²) error on
accepted shots and rejection linear in p. The comparison with LRB-D is a
trade-off between F and acceptance, not a strict win. Correction should do
best where the two explanations have very different probabilities, for
example `Z1` (many DEPOLARIZE1(2p) layers) against `Z2 Z3` (p/15 per CZ or
SWAP).

**I2. Detector definitions.** **Verified analytically.** Data errors are
never physically corrected in this protocol. A data error therefore persists
and flips every later absolute check outcome it anticommutes with, so
detectors defined as absolute check values (the natural choice for
postselection) give mechanisms with O(m) detectors. Define each detector
instead as the XOR of a check outcome with the outcomes of its preimage
stabilizer in the previous round, propagated through the logical Clifford in
between. Then a data fault between rounds flips at most two Z-type and two
X-type detectors.
**Conjecture.** Give every decoder these round-to-round detectors. The LRB-D
rule ("reject if any used check is non-trivial") is a linear condition on
them, so both baselines can share one DEM.

**I3. Logical gates create hyperedges.** **Verified analytically.** Logical S
(S on 0 and 4, S† on 1, CZ(2,3)) maps `S_X1 → S_X1·S_Z1` and
`S_X2 → S_X2·S_Z2` and fixes the Z checks. Logical H (SWAP(2,3), then H on
all data) swaps `S_X1 ↔ S_Z1` and `S_X2 ↔ S_Z2`. So after an S the X1
detector is `x1(r+1) ⊕ x1(r) ⊕ z1(r)`, and a Z1 measurement error in round r
flips three detectors. After an H, X-type and Z-type detectors are linked
across rounds, so the problem does not split into separate X and Z matching
graphs.
**Verified numerically (toy, Appendix A).** Stim's default determinism check
accepted the propagated detectors and observables for I, H and S, which
confirms these mappings numerically. In a two-round toy circuit the
three-detector mechanism `D0 D4 D6` appears only when the gate between rounds
is S. Stim's `decompose_errors=True` succeeded for I, H and S and wrote it as
`D0 ^ D4 D6`.
**From literature.** This is the mechanism described for fold-transversal S
and H in the unrotated surface code (SerraPeralta2026, F18) and for
transversal CNOT (Cain2024, F17).
**Conjecture.** Check decomposition on the real generator output, including
long words and the final round after the check-free inverse half. If it ever
fails, use `ignore_decomposition_failures=True` only together with a
hyperedge-aware decoder, because PyMatching silently drops undecomposed
mechanisms (F5).

**I4. PyMatching at distance 2.** **From literature.** PyMatching keeps the
observables of the first-added parallel edge (F5). **Verified numerically**
(pymatching 2.3.1, Appendix A). For a minimal DEM with `error(0.1) D0 L0`
listed before `error(0.2) D0`, PyMatching predicts a logical flip on syndrome
D0. With the two lines swapped it predicts no flip. In both cases the merged
edge has p = 0.26. In the two-round toy, 6, 10 and 11 of the 24, 25 and 29
distinct graphlike detector sets (gate I, H, S) carry more than one
observable class. PyMatching's prediction on the matching single-mechanism
syndrome differed from the more probable class on 1, 1 and 2 of them.
**Conjecture.** Unless the DEM is pre-processed, the MWPM column of the
comparison measures DEM ordering rather than the decoder. Before building the
graph, merge mechanisms per detector set and give the merged edge the
observables of the more probable class, for example via
`Matching.add_edge(..., fault_ids=best, error_probability=merged)`.

**I5. Exact ML is affordable here and should be the reference.** **Verified
analytically.** Four stabilizer generators give 16 syndromes. With 4 logical
classes that makes 64 cosets. At m = 22 there are 23 noisy check rounds, so
92 round-to-round detectors (4 per round).
**Conjecture.** Two routes give exact ML and the posterior P(L | syndrome)
cheaply per syndrome: the coset recursion of Heim2016 with a 64-entry vector
per shot, or an exact time-ordered elimination of the DEM whose frontier
spans a few rounds. Memoizing on distinct syndromes turns either into a
lookup table built on the fly, like LILLIPUT's offline-filled table
(Das2021). At p = 0.05 with 10⁶ shots almost every syndrome is distinct, so
that corner needs vectorized code or fewer shots.

**I6. Suggested decoder ladder for Q1.** **Conjecture.**
1. Exact ML: optimal, and it gives soft output.
2. PyMatching with the I4 fix, with and without `enable_correlations=True`:
   the fast standard.
3. BP+OSD and BP+LSD from ldpc on the undecomposed DEM: hyperedges kept,
   already installed.
4. Tesseract (MLE; would need a venv inside the worktree).
5. A lookup table filled from samples: no DEM, the hardware-friendly option.

All of these should agree on unambiguous syndromes and differ mainly on the
ambiguous pairs of I1 and on multi-fault patterns. Neural decoders and
windowing add little for a 5-qubit code decoded offline. AlphaQubit-class
models target large distances and real-time hardware (Bausch2024, F23).

**I7. Check policies under correction.** **Verified analytically** (from the
CONTEXT protocol). Every circuit physically contains all m+1 check rounds
whatever the policy. Restricting the decoder to the rounds a policy uses
therefore throws information away without removing any noise.
**Conjecture.**
- Main LRB-C curve: decode with all rounds.
- Control: decode with only the policy's rounds. This does not faithfully
  emulate hardware that extracts fewer rounds, because the noise of the
  unused rounds is still present.
- `const=0`: correction reduces to choosing, for each of the four final
  X-syndromes, whether to flip the readout. It helps only as far as the
  ambiguous explanations have unequal accumulated probabilities.

**I8. Terminal readout.** **Conjecture** (design suggestion). CONTEXT step 5
rotates and measures data 0 and 3 only. An ideal X-basis readout of all five
data qubits, as `const=0` already does, would add a noiseless final
X-syndrome. That closes the time boundary for the Z-type faults that flip the
X_L readout. A final-round measurement error would then appear as a harmless
time-like pair instead of an ambiguous data-like event. This changes the
protocol relative to the qutrit run, so the LRB-D baseline would have to be
recomputed with the same readout.

**I9. Partial correction: abort only when unsure.** **Verified analytically**
(given I2's detectors). A measurement fault, or an ancilla X error after
reset, flips only the two time-adjacent detectors of one check (three around
a logical S) and no observable. At first order no data fault has the same
signature. The exception is the final noisy round: there a measurement fault
flips one detector and looks like a data fault in the inverse half, unless a
noiseless final readout is added (I8).
**Conjecture** (first-order estimate from the CONTEXT noise model). These
harmless time-like faults occur at roughly 4 × (5p + 2p/3 + 2p) ≈ 31p per
round, plus idle-ancilla terms. They are probably the main source of
rejection in LRB-D with `unif=1`.
**Conjecture.** A rule "correct and accept if the ML posterior of the chosen
class is at least 1-ε, else reject" should keep first-order ambiguous faults
rejected (logical error O(p²)) while accepting the time-like faults.
Rejection would then fall well below LRB-D at similar F. ε → 0 approaches
"reject only ambiguous", and ε = 1/2 is full correction. This is the
exclusive-decoder idea of Smith2024 (F26), with soft outputs as in
Meister2024.

**I10. Q2, non-Markovianity.** **Conjecture.** LRB-D conditions the accepted
ensemble on depth-dependent syndrome histories. That can make F(m)
non-exponential, as in the qutrit `const=1` data quoted in CONTEXT. A decoder
that keeps every shot and makes temporally local decisions (commit region
plus buffer, Skoric2023, F25) should give a roughly time-homogeneous
effective logical channel per step, and so decays closer to exponential.
Correlations from a global decoder should then mostly affect the prefactor b.
Two caveats: the check-free inverse half (I12) breaks this, and decoders
cannot remove slow physical drift unless their weights adapt (Spitz2018, F27)
or are learned from device data (Bausch2024, F24).

**I11. Q3, hardware.** **Conjecture.**
- LRB circuits are Clifford-only with a Pauli readout at the end. Decoding
  can therefore be offline, with corrections kept as a Pauli frame, which
  RyanAnderson2021 did in real time (F21). Storing the 4 syndrome bits per
  round per shot is enough, so no real-time decoder is needed on any
  platform.
- If real-time feedback is wanted later, a tiny-code lookup table fits on an
  FPGA with nanosecond latency (Das2021, F22) or in controller software
  (RyanAnderson2021).
- On hardware the DEM priors must be estimated from data. At d = 2, ML
  decisions on ambiguous syndromes hinge on probability differences, so they
  may be fragile to mis-estimated priors.
- Platforms with transversal or fold-transversal gates (ions, atoms) need
  decoding across those gates (Cain2024, SerraPeralta2026).

**I12. The check-free inverse half limits any decoder.** **Conjecture.** Faults
in the m inverse Cliffords are seen only by the final round's 4 detectors
(2 if only the X sector matters at readout). Multiple faults there alias, and
no decoder can separate them. The inverse half's failure probability grows
with m in a way that is not a product of per-step factors. LRB-C may
therefore still decay non-exponentially unless checks are added to the
inverse half, which would change the protocol.

---

## 4. Seed corrections

All 13 seed arXiv IDs were correct. Corrections and clarifications:

| Seed as given | Status | Correction or note |
|---|---|---|
| Gidney, "Stim: a fast stabilizer circuit simulator" (arXiv:2103.02202) | correct | Published in Quantum 5, 497 (2021). The abstract does not mention detector error models, so DEM claims cite StimDocs instead. |
| Higgott, "PyMatching" (arXiv:2105.13082) | correct ID, short title | Full title "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching"; ACM Trans. Quantum Comput. 3(3) (2022), DOI 10.1145/3505637 (Crossref). It describes v0.x (local matching). The installed v2 is described by HiggottGidney2025, so this seed moved to secondary. |
| Higgott and Gidney, "Sparse Blossom" (arXiv:2303.15933) | correct | Full title adds ": correcting a million errors per core second with minimum-weight matching"; Quantum 9, 1600 (2025). |
| Delfosse and Nickerson (arXiv:1709.06218) | correct | Quantum 5, 595 (2021). |
| Panteleev and Kalachev (arXiv:1904.02703) | correct | Title capitalised "Degenerate Quantum LDPC Codes With Good Finite Length Performance" on arXiv; Quantum 5, 585 (2021). |
| Roffe et al. (arXiv:2005.07016) | correct | Authors J. Roffe, D. R. White, S. Burton, E. T. Campbell; PRR 2, 043423 (2020). |
| Hillmann et al., localized statistics decoding (arXiv:2406.18655) | correct | Title "Localized statistics decoding for quantum low-density parity-check codes"; six authors (Hillmann, Berent, Quintavalle, Eisert, Wille, Roffe); Nat. Commun. 16, 8214 (2025). |
| Bravyi, Suchara, Vargo (arXiv:1405.4883) | correct | PRA 90, 032326 (2014). |
| Chubb, "General tensor network decoding of 2D Pauli codes" (arXiv:2101.04125) | correct | No journal ref on the arXiv page. Code-capacity Pauli noise only, so moved to secondary; Piveteau2024 is the circuit-level successor. Software is Julia (SweepContractor.jl). |
| Piveteau, Chubb, Renes, TN decoding beyond 2D (arXiv:2310.10722) | correct | Exact title "Tensor Network Decoding Beyond 2D"; PRX Quantum 5, 040303 (2024). |
| Fowler, correlated errors in the surface code (arXiv:1310.0863) | correct ID, paraphrased title | Exact title "Optimal complexity correction of correlated errors in the surface code" (2013, arXiv only). PyMatching 2.3.0 cites this paper for its correlated mode. |
| Skoric et al., parallel window decoding (arXiv:2209.08552) | correct | Title "Parallel window decoding enables scalable fault tolerant quantum computation"; five authors; Nat. Commun. 14, 7040 (2023). |
| Bausch et al., AlphaQubit (arXiv:2310.05900) | correct ID, title differs | arXiv title "Learning to Decode the Surface Code with a Recurrent, Transformer-Based Neural Network". Published as "Learning high-accuracy error decoding for quantum processors", Nature 635, 834-840 (2024), DOI 10.1038/s41586-024-08148-8. "AlphaQubit" is the decoder's name, not part of either title. |

---

## 5. Open questions

1. Does Stim's decomposition succeed on the full LRB circuits: all Clifford
   words, m up to 22, and the final round after the check-free inverse half?
   The toy (I3) covers single gates only.
2. How unequal are the two first-order explanations of each ambiguous
   syndrome in the SI1000 circuit? This sets the first-order coefficient of
   LRB-C (I1) and how much ML beats a coin flip.
3. How much does exact ML beat MWPM with the I4 fix, correlated MWPM,
   BP+OSD/LSD and Tesseract on this code, and does any gap survive at second
   order?
4. Is exact ML affordable at the top of the p grid (p = 0.05, 10⁶ shots, nearly
   all syndromes distinct), or should the decoding study use fewer shots
   there (I5)?
5. Does the partial-correction rule (I9) dominate LRB-D on the
   acceptance-versus-F plane for every policy, and what ε is natural?
6. Is LRB-C's F(m) exponential given the check-free inverse half (I12)? Should
   the qubit study add checks to the inverse half, at least as a variant, to
   answer Q2 cleanly?
7. How should "policy" be defined under correction (I7)? Should the LRB-D
   baseline be recomputed with an all-data X readout if I8 is adopted?
8. Does PyMatching's correlated mode change the ambiguous-pair decisions, or
   does the first-added-observable rule still decide them?
9. On hardware (Q3): how robust are d=2 ML decisions to mis-estimated DEM
   priors, and can leakage flags (superconducting) or erasure flags (neutral
   atoms) be fed in as extra detectors? Lit review 05 covers flags.

---

## Appendix A. Light numerical checks (stim 1.15.0, pymatching 2.3.1, ldpc 2.4.1; under a minute in total)

Scratch scripts, not the production generator; they live only in the session
scratchpad
(`<local scratch, not kept>`
and `.../scratchpad/parallel_check.py`).

Toy circuit:
- **Preparation:** RX on the five data qubits (an |+>-type product state
  with X stabilizers +1 and X_L = +1), then a noiseless MPP of the two Z
  stabilizers.
- **Check rounds:** two noisy rounds with the CONTEXT check circuits and the
  CONTEXT SI1000 translation at p = 1e-3, each followed by reset of ancillas
  5-8.
- **Gate between the rounds:** logical I, H (SWAP(2,3) then H on the data) or
  S (S on 0 and 4, S_DAG on 1, CZ(2,3)), with DEPOLARIZE1(0.1p) after
  single-qubit gates and DEPOLARIZE2(p) after CZ and SWAP.
- **Detectors:** round-to-round, propagated through the gate as in I3.
- **Readout:** noiseless MPP of the final X stabilizers and of the
  propagated X_L (`Y0 Z2 X3` after S, `Z0 Z2` after H).

| Gate | DEM mechanisms | mechanisms flipping >2 detectors | `decompose_errors=True` | graphlike detector sets | sets with >1 observable class | PyMatching ≠ more probable class |
|---|---|---|---|---|---|---|
| I | 74 | 30 | succeeded | 24 | 6 | 1 |
| H | 85 | 34 | succeeded | 25 | 10 | 1 |
| S | 86 | 37 | succeeded | 29 | 11 | 2 |

- With S, the Z1-ancilla measurement error of round 1 (p ≈ 6.0e-3) is
  `error(...) D0 D4 D6` (Z1 round 1, Z1 round 2, X1 round 2). With I it is
  `D0 D4`. Stim decomposes the S case as `D0 ^ D4 D6`.
- Parallel-edge test: the DEM `error(0.1) D0 L0`, `error(0.2) D0`,
  `error(0.01) D0 D1` gives one boundary edge with p = 0.26 and fault_ids
  {0}, so a flip is predicted. Swapping the first two lines gives p = 0.26,
  fault_ids {} and no flip. This matches the documented rule (F5).
- Installed ldpc 2.4.1 exposes `BpOsdDecoder`, `BpLsdDecoder`,
  `UnionFindDecoder`, `BeliefFindDecoder`, `SinterBpOsdDecoder`,
  `SinterLsdDecoder`, `SinterBeliefFindDecoder`, and in `ldpc.ckt_noise` the
  window decoders and `detector_error_model_to_check_matrices` (with
  `allow_undecomposed_hyperedges`).

## Appendix B. Search log (WebSearch queries run, 2026-10-28)

1. maximum likelihood decoding detector error model tensor network circuit-level noise exact decoder 2024 2025 arXiv
2. PyMatching correlated matching enable_correlations release 2.3
3. ldpc package BpLsdDecoder BpOsdDecoder circuit-level noise detector error model stim Roffe documentation
4. real-time lookup table decoder small code trapped ion fault-tolerant error correction color code experiment
5. neural network decoder small distance-3 surface code circuit-level noise comparison lookup table optimal decoder
6. soft-output decoder complementary gap postselection logical error mitigation abort decoding surface code 2024
7. correlated decoding logical algorithms transversal gates most likely error integer programming neutral atoms
8. sliding window decoding circuit-level noise causal decoder overlapping recovery commit region buffer region
9. Tesseract search-based decoder quantum error correction A* hyperedges detector error model
10. optimal circuit-level decoding surface codes Heim Svore Hastings
11. stim detector error model graphlike decompose_errors hyperedge "Failed to decompose" matching
12. tensor network decoder Stim detector error model Python package exact contraction maximum likelihood small codes github
13. distance-2 code error detection versus error correction decoding postselection small code logical error rate first order circuit-level
14. decoder for time-correlated non-Markovian noise quantum error correction adaptive weight estimation drifting noise
15. union-find decoder circuit-level noise weighted hypergraph union-find detector error model performance vs MWPM
16. minimum-weight parity factor decoder hypergraph HyperBlossom mwpf Wu Zhong arXiv

Also fetched and verified on abstract pages, but not used for claims here:
- Pryadko, "On maximum-likelihood decoding with circuit-level errors",
  arXiv:1909.06732, Quantum 4, 304 (2020).
- Cao et al., "Exact Decoding of Repetition Code under Circuit Level Noise",
  arXiv:2501.03582 (2025). Exact ML via planar spin glass, tested on SI1000.
- Varbanov, Serra-Peralta, Byfield, Terhal, "Neural network decoder for
  near-term surface-code experiments", arXiv:2307.03280, PRR 7, 013029 (2025).
- Baireuther, O'Brien, Tarasinski, Beenakker, "Machine-learning-assisted
  correction of correlated qubit errors in a topological code",
  arXiv:1705.07855, Quantum 2, 48 (2018).
- Senior et al. (24 authors), "A scalable and real-time neural decoder for
  topological quantum codes", arXiv:2512.07737 (2025).
- Wu, Li, Chang, Puri, Zhong, "Minimum-Weight Parity Factor Decoder for
  Quantum Error Correction", arXiv:2508.04969 (2025).
- Cao, Yan, Du, Pan, "Maximum Likelihood Decoding of Quantum Error Correction
  Codes" (review), arXiv:2605.17230 (2026).

Leads seen only in search results, **UNVERIFIED** (not fetched, not cited):
- arXiv:2511.09491 (adaptive estimation of drifting noise)
- arXiv:2606.16288 (DEM reconstruction)
- arXiv:2609.13087 (composing DEMs)
- arXiv:2602.02985 (accelerating Tesseract)
- arXiv:2410.07598 (post-selected QEC thresholds)
- arXiv:2603.08697 (scalable postselection)
- stimbposd (PyPI)
- mdopt (JOSS)
