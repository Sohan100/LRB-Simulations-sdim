# Citation check: lit-L8-d3codes.md (distance-3 qutrit codes, qudit QEC theory, break-even)

Independent citation checker, written 2026-10-28. Scope: the 22 citations in the L8 citation list,
plus a read-through of `lit-L8-d3codes.md` for statements about these papers that disagree with
what the papers say.

## Method

- Metadata. For every entry I fetched the arXiv abstract page (`https://arxiv.org/abs/<id>`) and
  read its `citation_*` meta tags. I fetched the arXiv Atom API record (one batch query,
  `https://export.arxiv.org/api/query?id_list=<the 22 IDs>`) for title, authors, version dates,
  DOI and journal reference. For every DOI I fetched the Crossref record
  (`https://api.crossref.org/works/<doi>`). Extra lookups are named in the entry remarks.
- Content. For every entry I downloaded the arXiv e-print source (`https://arxiv.org/e-print/<id>`)
  and searched it in memory, without extracting it, for each quote and number that the L8 notes
  attribute to the paper. `lit-L8-citecheck-checks/srcgrep.py` is the search helper.
- Labels. [from literature] means I read it on a page or in a source I fetched in this task.
  [verified numerically] means my own computation, in `lit-L8-citecheck-checks/`.

## Verdicts

Result: 22 confirmed, 0 corrected, 0 not found [from literature]. No arXiv ID, DOI, year or author
list is wrong. The only differences are cosmetic: missing diacritics, and title variants between
the arXiv listing and the journal. The cited titles follow the journal versions.

| Key | Verdict | Title (journal form) | Authors | Year | arXiv | DOI | Venue | Remarks |
|---|---|---|---|---|---|---|---|---|
| chau1997 | confirmed | Five quantum register error correction code for higher spin systems | H. F. Chau | 1997 | quant-ph/9702033 | 10.1103/PhysRevA.56.R1 | Phys. Rev. A 56(1), R1-R4 (July 1997) | The arXiv title is in title case. v3 is dated 1997-05-19. |
| rains1999 | confirmed | Nonbinary quantum codes | Eric M. Rains | 1999 | quant-ph/9703048 | 10.1109/18.782103 | IEEE Trans. Inf. Theory 45(6), 1827-1832 (1999) | arXiv v1 is from 1997-03-25. The arXiv page lists no DOI. I confirmed the DOI through Crossref and the Caltech record. |
| ashikhmin2001 | confirmed | Nonbinary quantum stabilizer codes | Alexei Ashikhmin, Emanuel Knill | 2001 | quant-ph/0005008 | 10.1109/18.959288 | IEEE Trans. Inf. Theory 47(7), 3065-3072 (2001) | arXiv is from 2000-05-01. The arXiv page lists no DOI; Crossref confirms it. |
| ketkar2006 | confirmed | Nonbinary stabilizer codes over finite fields | Avanti Ketkar, Andreas Klappenecker, Santosh Kumar, Pradeep Kiran Sarvepalli | 2006 | quant-ph/0508070 | 10.1109/TIT.2006.883612 | IEEE Trans. Inf. Theory 52(11), 4892-4914 (Nov 2006) | arXiv is from 2005. The arXiv page lists no DOI; Crossref confirms it. |
| gottesman1998 | confirmed | Fault-tolerant quantum computation with higher-dimensional systems | Daniel Gottesman | 1998 (journal 1999) | quant-ph/9802007 | 10.1016/S0960-0779(98)00218-5 | Chaos, Solitons & Fractals 10(10), 1749-1758 (Sept 1999) | The arXiv page lists this DOI. The Crossref record for it has the placeholder title "Full length article" and no authors, but its volume, issue and pages match the arXiv journal reference. Semantic Scholar maps the DOI to this arXiv paper. A second published version exists: Quantum Computing and Quantum Communications (QCQC 1998), Lecture Notes in Computer Science, pp. 302-313, DOI 10.1007/3-540-49208-9_27 (Crossref search). |
| campbell2014 | confirmed | Enhanced fault-tolerant quantum computing in d-level systems | Earl T. Campbell | 2014 | 1406.3055 | 10.1103/PhysRevLett.113.230501 | Phys. Rev. Lett. 113, 230501 (2014-12-03) | none |
| campbell2012 | confirmed | Magic-state distillation in all prime dimensions using quantum Reed-Muller codes | Earl T. Campbell, Hussain Anwar, Dan E. Browne | 2012 | 1205.3104 | 10.1103/PhysRevX.2.041021 | Phys. Rev. X 2, 041021 (2012-12-27) | The arXiv title has no hyphen ("Magic state distillation"). |
| anwar2012 | confirmed | Qutrit magic state distillation | Hussain Anwar, Earl T. Campbell, Dan E. Browne | 2012 | 1202.2326 | 10.1088/1367-2630/14/6/063006 | New J. Phys. 14, 063006 (2012) | none |
| watson2015color | confirmed | Qudit color codes and gauge color codes in all spatial dimensions | Fern H. E. Watson, Earl T. Campbell, Hussain Anwar, Dan E. Browne | 2015 | 1503.08800 | 10.1103/PhysRevA.92.022312 | Phys. Rev. A 92, 022312 (2015) | The arXiv title spells "Colour". |
| moussa2016 | confirmed | Transversal Clifford gates on folded surface codes | Jonathan E. Moussa | 2016 | 1603.02286 | 10.1103/PhysRevA.94.042316 | Phys. Rev. A 94, 042316 (2016) | none |
| breuckmann2024 | confirmed | Fold-transversal Clifford gates for quantum codes | Nikolas P. Breuckmann, Simon Burton | 2024 | 2202.06647 | 10.22331/q-2024-06-13-1372 | Quantum 8, 1372 (2024-06-13) | arXiv v1 is from 2022-02-14 and v3 from 2024-06-10. |
| muralidharan2017 | confirmed | Overcoming erasure errors with multilevel systems | Sreraman Muralidharan, Chang-Ling Zou, Linshu Li, Jianming Wen, Liang Jiang | 2017 | 1504.08054 | 10.1088/1367-2630/aa573a | New J. Phys. 19, 013026 (2017) | arXiv v1 is from 2015-04-30. |
| keppens2025 | confirmed | Qudit vs. qubit: Simulated performance of error-correction codes in higher dimensions | James Keppens, Quinten Eggerickx, Vukan Levajac, George Simion, Bart Sorée | 2025 | 2502.05992 | 10.1103/2w52-qd2j | Phys. Rev. A 112(3), 032435 (2025-09-25) | The last author is "Sorée" with an accent. The arXiv title reads "Qudit vs. Qubit: ... error correction codes ...". |
| kabir2025sdim | confirmed | Sdim: A qudit stabilizer simulator | Adeeb Kabir, Steven Nguyen, Sohan Ghosh, James Keppens, Tijil Kiran, Isaac H. Kim, Yipeng Huang, Bart Sorée | 2025 | 2511.12777 | none (arXiv DOI 10.48550/arXiv.2511.12777) | arXiv preprint: v1 2025-11-16, v2 2026-05-12 | "Sorée" has an accent. arXiv lists no journal reference, and WebSearch found no journal version. |
| ducloscianci2013 | confirmed | Kitaev's Z_d-code threshold estimates | Guillaume Duclos-Cianci, David Poulin | 2013 | 1302.3638 | 10.1103/PhysRevA.87.062338 | Phys. Rev. A 87, 062338 (2013) | The arXiv title reads "Kitaev's Z_d-Codes Threshold Estimates". |
| watson2015decoder | confirmed | Fast fault-tolerant decoder for qubit and qudit surface codes | Fern H. E. Watson, Hussain Anwar, Dan E. Browne | 2015 | 1411.3028 | 10.1103/PhysRevA.92.032309 | Phys. Rev. A 92, 032309 (2015) | The arXiv title starts with "A". arXiv v1 is from 2014. |
| marks2017 | confirmed | Comparison of memory thresholds for planar qudit geometries | Jacob Marks, Tomas Jochym-O'Connor, Vlad Gheorghiu | 2017 | 1701.02335 | 10.1088/1367-2630/aa939a | New J. Phys. 19, 113022 (2017) | v2 is "replaced with the published version". The arXiv listing abstract is stale; see the claim checks below. |
| brock2025 | confirmed | Quantum error correction of qudits beyond break-even | Benjamin L. Brock, Shraddha Singh, Alec Eickbusch, Volodymyr V. Sivak, Andy Z. Ding, Luigi Frunzio, Steven M. Girvin, Michel H. Devoret | 2025 | 2409.15065 | 10.1038/s41586-025-08899-y | Nature 641(8063), 612-618 (2025) | arXiv is from 2024-09-23. |
| iqbal2025 | confirmed | Qutrit toric code and parafermions in trapped ions | Mohsin Iqbal, Anasuya Lyons, Chiu Fan Bowen Lo, et al. (19 authors, full list below) | 2025 | 2411.04185 | 10.1038/s41467-025-61391-z | Nat. Commun. 16, 6301 (2025-07-08) | arXiv is from 2024-11-06. |
| gottesman2016 | confirmed | Quantum fault tolerance in small experiments | Daniel Gottesman | 2016 | 1610.03507 | none | arXiv preprint (v2 2016-10-14) | WebSearch found no journal version. |
| prabhu2024 | confirmed | Distance-four quantum codes with combined postselection and error correction | Prithviraj Prabhu, Ben W. Reichardt | 2024 | 2112.03785 | 10.1103/PhysRevA.110.012419 | Phys. Rev. A 110, 012419 (2024-07-08) | arXiv has only v1, from 2021-12-07. |
| combes2017 | confirmed | Logical randomized benchmarking | Joshua Combes, Christopher Granade, Christopher Ferrie, Steven T. Flammia | 2017 | 1702.03688 | none | arXiv preprint (v1 only) | WebSearch found no journal version. |

The 19 authors of iqbal2025, from the arXiv page and Crossref: Mohsin Iqbal, Anasuya Lyons,
Chiu Fan Bowen Lo, Nathanan Tantivasadakarn, Joan Dreiling, Cameron Foltz, Thomas M. Gatterman,
Dan Gresh, Nathan Hewitt, Craig A. Holliman, Jacob Johansen, Brian Neyenhuis, Yohei Matsuoka,
Michael Mills, Steven A. Moses, Peter Siegfried, Ashvin Vishwanath, Ruben Verresen, Henrik Dreyer.

## Claim checks against lit-L8-d3codes.md

No statement in the notes contradicts the abstract page or the source of the paper it cites. A few
items need a qualification. All items below are [from literature] unless labelled otherwise.

1. Marks 2017 (notes section 3.17). The arXiv listing abstract has not been updated since v1. It
   says "code capacity depolarizing noise" and quotes 9.75%, 11.4%, a 75% increase and a 17.1%
   plateau. The v2 source has its own abstract and body, and both say "generalized bit flip error
   model" and quote 5.6%, 8.0%, a 112% increase and an 11.9% plateau. The v2 table matches the
   notes exactly: MWPM 0.103, HDRG 0.093 and 0.155, DSP 0.080, GCC 0.056 and 0.119, plus GCC 0.084
   for D=3. The notes correctly flag the mismatch and follow the body. One refinement: the
   corrected numbers also appear in the v2 abstract, not only in the body.
2. Keppens et al. 2025 (notes section 3.13).
   - These match the source: the fit table, a, b and thresholds for q = 2, 3, 5 with and without
     a flag; the generator convention, cyclic shifts of I Z X X^dag Z^dag; the per-step
     probability p_s; the error-free ancilla readout in the last cycle; the flag qudit with a
     lookup table; and the q = 7 data point (5000 samples at p = 0.1, standard depolarizing
     model).
   - Not verified: "simulations run 3 cycles". The source says "O(d) error syndrome cycles", and
     I found no explicit count.
   - The fixed-point arithmetic is correct: 1116^(-1/0.870) = 3.14e-4 and 58.7^(-1/0.288) =
     7.2e-7 [verified analytically].
   - The notes accurately quote the paper's claim that T_L = (S_q F_q)^(x5) is transversal
     logical, with T_q "the operation that cyclically shifts between the Heisenberg-Weyl gates".
     The notes then say the claim fails for q = 3. My own check agrees [verified numerically]. For
     generators given by cyclic shifts of I Z X X^-1 Z^-1, the uniform single-qudit symplectic
     maps that preserve the stabilizer space are:
     - q = 2: {I, (0,1,1,1), (1,1,1,0)}, a group of order 3; S F works.
     - q = 3: {+-I, +-(1,1,1,2)}, of order 4. Neither S F (0,2,1,2) nor F S (2,2,1,0), both of
       order 3, preserves the space.
     - q = 5: a group of order 10, without S F.
     - q = 7: a group of order 8, without S F.

     So the claim holds only for q = 2 among these cases. Keppens's S_q reduces to
     diag(w^(n(n-1)/2)), which maps X to XZ, so the convention is the one I used. The notes left
     q = 5 unchecked; it fails there too. The check uses the symplectic part only, which suffices
     for odd q by the argument at the top of the notes. Script and log:
     `lit-L8-citecheck-checks/five_qudit_transversal.py` and `.log`, run with arguments `2 3 5 7`
     in about 2 s on one core.
3. Combes et al. 2017 (notes section 3.22). The notes call the original LRB "a correction-based
   protocol". The protocol step in the source reads "following each with a round of error
   correction (with or without recovery) and a refresh of the ancilla register". Syndrome
   extraction after each Clifford is required, but applying the recovery is optional. This is a
   qualification, not a contradiction. The fit form F_L(m) = A_L p_L^m + B_L and the assumptions
   (time-independent, Markovian and gate-independent noise at the logical level; syndrome
   measurement errors uncorrelated between rounds) are verbatim.
4. Watson, Anwar, Browne 2015 (notes section 3.16). The values 0.0215 for qubits, about 4.2%
   saturation (0.042 at d = 7919) and about 8.2% with four initialization levels all match the
   body. The introduction of the same source says 8.3%, and the abstract says "more than 8%".
   This is a minor internal inconsistency in the paper, not in the notes.
5. Gottesman 1998 (notes section 3.5). The bibliographic data in the notes is correct. The
   Crossref record for the DOI is defective (see the table). Every content claim checked matches
   the source: the footnote "This aspect is actually simpler for odd d than for d=2", P as
   w^(j(j-1)/2), S_a with |j> -> |aj>, and the generalized Shor Toffoli.
6. Prabhu and Reichardt 2024 (notes section 3.21). All quotes appear verbatim in arXiv v1, which is
   the only arXiv version: "Distance-three codes may alternatively be used ...", "wiped and
   reset", the weight-eight flag circuit, the weight-one rule for one fault, and the 25x figure. I
   did not check the published PRA text.
7. Anwar et al. 2012 (notes section 3.8). The 23.3% and 18.3% figures, the generators
   X Z Z^-1 X^-1 I, and "the 7-qutrit code distils not pure, but mixed states" all match. "It
   suppresses noise only linearly" is a paraphrase, not a quote. It is consistent with the
   first-order output errors in the source, for example eps_out about (0.38 + 0.09 eps2) eps1.

Verified with no issues [from literature, source text read]:

- Chau 1997: the optimality claim that no code shorter than five works, and the "Toffoli-like" phase gate.
- Rains 1999: Theorem 2 (quantum Singleton bound with purity), Theorem 13 (((5,a,3))_a and ((6,1,4))_a), the inert/split/ramified classes, and Theorem 14.
- Ashikhmin and Knill 2001: the Bierbrauer-Edel distance-3 families.
- KKKS 2006: the Hamming bound (by linear programming for n >= 5), the cyclic Hamming theorem (m >= 2, gcd(m, q^2-1) = 1), 4 <= n <= q^2+d-2, and the MDS-conjecture corollary.
- Campbell 2014: "For technical reasons, we consider only prime dimensions of five and above", n = d-1, and D = floor((d+1)/3).
- Campbell, Anwar, Browne 2012: 0.20015 and 0.211001, eps' = eps^2[3 + cos 4theta] + ..., P >= 1/9, the 4-ququint code with threshold 0.363, and "We know of only one study ...".
- Watson et al. 2015 (colour codes): the star bipartition, the exceptional d = 3, 6 cases for the T gate, and "We shall not present a decoder".
- Moussa 2016: square and diamond families, 2D^2-2D+1 qudits, the doubly-even quote, the chiral-edges quote, minimal and expanded cones (2D^2-1), depth 6D-5, about 72 D^3, the 8/12/4 split, and eps_T about 0.01 and 0.001.
- Breuckmann and Burton 2024: the 90-degree-rotation quote, the four S_tau conditions, H_tau^2 = I only if tau^2 = id, "trivially fault-tolerant", and the expected qudit generalization.
- Muralidharan et al. 2017: 50% for all code sizes, about 37% for the D=3 surface code, [[3,1,2]]_3 against [[4,2,2]], 87Rb, and the generalized Duan-Kimble CZ.
- Kabir et al. 2025: all seven quoted passages are verbatim in v2; the [[5,1,2]]_3 code is attributed to Moussa.
- Duclos-Cianci and Poulin 2013: "increases as a function of the local dimension d" (from the conclusion), C_2 about 0.110 and C_3 about 0.159.
- Brock et al. 2025: tantalum transmon, generalized sBs with stabilizer length sqrt(pi d), the Fock-qudit baseline, gains 1.82 and 1.87, and a gain of about 1.8 from d = 2 to 4.
- Iqbal et al. 2025: 56 171Yb+ ions on H2, up to 6 x 4 = 24 qutrits, two qubits per qutrit, and 96.5(3)%.
- Gottesman 2016: the "all circuits in the family" criterion, "never rejected", p < (4+2T)/(16T^2+122T+225) with 1.6%, 1.2%, 0.79% and 0.12%, and the SWAP quote.

## URLs fetched in this task

- For each of the 22 IDs: `https://arxiv.org/abs/<id>` and `https://arxiv.org/e-print/<id>`.
  The IDs are quant-ph/9702033, quant-ph/9703048, quant-ph/0005008, quant-ph/0508070,
  quant-ph/9802007, 1406.3055, 1205.3104, 1202.2326, 1503.08800, 1603.02286, 2202.06647,
  1504.08054, 2502.05992, 2511.12777, 1302.3638, 1411.3028, 1701.02335, 2409.15065, 2411.04185,
  1610.03507, 2112.03785 and 1702.03688. Also `https://export.arxiv.org/api/query?id_list=` with
  all 22 IDs.
- `https://api.crossref.org/works/<doi>` for the 19 DOIs in the table.
- Gottesman 1998 extras:
  - `https://doi.org/10.1016/S0960-0779%2898%2900218-5`, which redirects to linkinghub and then
    ScienceDirect (HTTP 403);
  - `https://api.openalex.org/works/doi:10.1016/S0960-0779(98)00218-5`;
  - `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/S0960-0779(98)00218-5`;
  - `https://api.crossref.org/works?query.bibliographic=Fault-tolerant+quantum+computation+with+higher-dimensional+systems+Gottesman`.
- `https://authors.library.caltech.edu/records/js205-g9q18`, the Caltech record for Rains 1999.
- WebSearch for journal versions of 2511.12777, 1702.03688 and 1610.03507. None was found.
