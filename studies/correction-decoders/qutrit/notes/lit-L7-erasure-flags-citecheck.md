# Citation check for lit-L7-erasure-flags.md

Independent check of the 21 citations in the L7 list (topic: erasure and leakage
heralding, flags, error-detection experiments), and of the statements about those
papers in `lit-L7-erasure-flags.md`. Checked 2026-10-28 on the login node. No
computation beyond downloading and text extraction.

Labels. Every verdict and every bibliographic field below is **from literature**
[lit]: it rests on the pages I fetched myself in this task, listed per entry in
Table 2. Arguments of my own are marked [analytic].

## Method

- arXiv. For each of the 20 arXiv IDs I downloaded `https://arxiv.org/abs/<id>` and
  read the `citation_title`, `citation_author`, `citation_date` and `citation_doi`
  meta tags, the journal-ref field and the submission history.
- DOIs. For each of the 19 DOIs I read the Crossref record
  `https://api.crossref.org/works/<doi>` (title, authors, journal, volume, issue,
  pages or article number, dates) and checked that `https://doi.org/<doi>` redirects
  (HTTP 302) to the expected publisher page. All 19 resolve. For Chou 2024, whose
  arXiv page lists no DOI, I also read the citation meta tags of the Nature Physics
  landing page.
- Thesis. The `uwspace.uwaterloo.ca` front end returned a bot-check page, so I read
  the DSpace REST record of the same item on the back-end host
  (`dspacemainprd01.lib.uwaterloo.ca/server/api/core/items/888cc1a3-...`) and the PDF
  bitstream. An arXiv API author search for "Durso Sabina" returned 0 results.
- Journal versions. For Gottesman 2016 (no DOI) a Crossref bibliographic query and a
  web search found no journal version.
- Full text. To spot-check the numbers and quotes in the notes I downloaded the arXiv
  PDF of all 20 arXiv papers and the thesis PDF, and converted them to text with
  `pypdf`. Section 3 lists what I compared.

Result: **21 confirmed, 0 corrected, 0 not found.** No bibliographic field in the
list is wrong. In the notes, two statements conflict with what the papers say
(Section 3.1) and six need a qualifier (Section 3.2). Every other number I
spot-checked matches the full text (Section 3.3).

## 1. Verdicts

| # | Key | Verdict | arXiv ID, versions | DOI | Venue, year (verified) | What differs |
|---|---|---|---|---|---|---|
| 1 | Wu2022 | confirmed | 2201.03540, v1 2022-01-10 | 10.1038/s41467-022-32094-6 | Nat. Commun. 13, 4657 (2022), online 2022-08-09 | None. |
| 2 | Kubica2023 | confirmed | 2208.05461, v1 2022-08-10 (only version) | 10.1103/PhysRevX.13.041022 | Phys. Rev. X 13(4), 041022 (2023), 2023-11-01 | arXiv lists 5 authors without H. Levine; Crossref lists 6, with Levine fourth. The entry already says so. |
| 3 | Levine2024 | confirmed | 2307.08737, v1 2023-07-17, v3 2024-03-20 | 10.1103/PhysRevX.14.011051 | Phys. Rev. X 14(1), 011051 (2024), 2024-03-20 | None. 113 authors on arXiv and in Crossref. |
| 4 | Chou2024 | confirmed | 2307.03169, v1 2023-07-06, v3 2023-10-13 | 10.1038/s41567-024-02539-4 | Nat. Phys. 20(9), 1454-1460 (2024), online 2024-07-02 | arXiv page has no DOI or journal-ref; the DOI is confirmed by Crossref and by the Nature Physics page. 34 authors in all three. The two titles differ, as the entry says. |
| 5 | Ma2023 | confirmed | 2305.05493, v1 2023-05-09 | 10.1038/s41586-023-06438-1 | Nature 622(7982), 279-284 (2023), online 2023-10-11 | None. The two titles differ, as the entry says. |
| 6 | McEwen2021 | confirmed | 2102.06131, v1 2021-02-11 | 10.1038/s41467-021-21982-y | Nat. Commun. 12, 1761 (2021), 2021-03-19 | None. 51 authors in both. |
| 7 | Miao2023 | confirmed | 2211.04728, v1 2022-11-09 | 10.1038/s41567-023-02226-w | Nat. Phys. 19(12), 1780-1786 (2023), online 2023-10-05 | arXiv lists 117 authors; Crossref lists 118 (adds Schurkus). arXiv page gives the DOI but no journal-ref. The two titles differ, as the entry says. |
| 8 | Varbanov2020 | confirmed | 2002.07119, v1 2020-02-17 | 10.1038/s41534-020-00330-w | npj Quantum Inf. 6, 102 (2020), 2020-12-14 | None. |
| 9 | Marques2023 | confirmed | 2302.09876, v1 2023-02-20, v2 2023-03-22 | 10.1103/PhysRevLett.130.250602 | Phys. Rev. Lett. 130(25), 250602 (2023), 2023-06-22 | None. |
| 10 | ChaoReichardt2018 | confirmed | 1705.02329, v1 2017-05-05 | 10.1103/PhysRevLett.121.050502 | Phys. Rev. Lett. 121(5), 050502 (2018), 2018-08-01 | None. |
| 11 | ChamberlandBeverland2018 | confirmed | 1708.02246, v1 2017-08-07, v3 2018-01-31 | 10.22331/q-2018-02-08-53 | Quantum 2, 53 (2018), 2018-02-08 | None. |
| 12 | DursoSabina2022 | confirmed | none (no arXiv version found) | none; handle 10012/17931 | MSc thesis (Physics, Quantum Information), University of Waterloo; repository date issued 2022-01-20 | None. Advisor Crystal Senko; record marks it "Unreviewed"; PDF title page says 2021. |
| 13 | Linke2017 | confirmed | 1611.06946, v1 2016-11-21, v2 2017-05-17 | 10.1126/sciadv.1701074 | Sci. Adv. 3(10), e1701074 (2017), 2017-10-06 | None. |
| 14 | Takita2017 | confirmed | 1705.09259, v1 2017-05-25 | 10.1103/PhysRevLett.119.180501 | Phys. Rev. Lett. 119(18), 180501 (2017), 2017-10-31 | None. |
| 15 | Andersen2020 | confirmed | 1912.09410, v1 2019-12-19 | 10.1038/s41567-020-0920-y | Nat. Phys. 16(8), 875-880 (2020), online 2020-06-08 | None. The arXiv title is in title case. |
| 16 | Marques2022 | confirmed | 2102.13071, v1 2021-02-25 | 10.1038/s41567-021-01423-9 | Nat. Phys. 18(1), 80-86 (2022), online 2021-12-16 | None. |
| 17 | Self2024 | confirmed | 2211.06703, v1 2022-11-12, v2 2024-07-26 | 10.1038/s41567-023-02282-2 | Nat. Phys. 20(2), 219-224 (2024), online 2024-01-05 | None. The arXiv title is in title case. |
| 18 | Bluvstein2024 | confirmed | 2312.03982, v1 2023-12-07 | 10.1038/s41586-023-06927-3 | Nature 626(7997), 58-65 (2024-02-01), online 2023-12-06 | None. The arXiv journal-ref reads only "Nature (2023)". 21 authors in both. |
| 19 | Gottesman2016 | confirmed | 1610.03507, v1 2016-10-11, v2 2016-10-14 | none (arXiv DataCite DOI 10.48550/arXiv.1610.03507) | arXiv preprint | None. No journal version found. |
| 20 | Grassl1997 | confirmed | quant-ph/9610042, v1 1996-10-28, v2 1997-01-17 | 10.1103/PhysRevA.56.33 | Phys. Rev. A 56(1), 33-38 (1997) | None. |
| 21 | Stricker2020 | confirmed | 2002.09532, v1 2020-02-21 | 10.1038/s41586-020-2667-0 | Nature 585(7824), 207-210 (2020), online 2020-09-09 | None. The two titles differ, as the entry says. |

## 2. Verified records and pages fetched

Titles are the arXiv titles, with the journal title after a semicolon where it
differs in more than letter case. Authors are as on the arXiv page. They agree with
Crossref except where Table 1 says otherwise.

| Key | Title | Authors | Pages fetched |
|---|---|---|---|
| Wu2022 | Erasure conversion for fault-tolerant quantum computing in alkaline earth Rydberg atom arrays | Yue Wu, Shimon Kolkowitz, Shruti Puri, Jeff D. Thompson | https://arxiv.org/abs/2201.03540, https://arxiv.org/pdf/2201.03540, https://api.crossref.org/works/10.1038/s41467-022-32094-6, https://doi.org/10.1038/s41467-022-32094-6 |
| Kubica2023 | Erasure qubits: Overcoming the T_1 limit in superconducting circuits | Aleksander Kubica, Arbel Haim, Yotam Vaknin, (journal only: Harry Levine), Fernando Brandão, Alex Retzker | https://arxiv.org/abs/2208.05461, https://arxiv.org/pdf/2208.05461, https://api.crossref.org/works/10.1103/PhysRevX.13.041022, https://doi.org/10.1103/PhysRevX.13.041022 |
| Levine2024 | Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons | Harry Levine, Arbel Haim, Jimmy S. C. Hung, et al. (113), Fernando G. S. L. Brandão, Alex Retzker, Oskar Painter | https://arxiv.org/abs/2307.08737, https://arxiv.org/pdf/2307.08737, https://api.crossref.org/works/10.1103/PhysRevX.14.011051, https://doi.org/10.1103/PhysRevX.14.011051 |
| Chou2024 | Demonstrating a superconducting dual-rail cavity qubit with erasure-detected logical measurements; Nat. Phys.: A superconducting dual-rail cavity qubit with erasure-detected logical measurements | Kevin S. Chou, Tali Shemma, Heather McCarrick, et al. (34), Shruti Puri, Steven M. Girvin, S. Harvey Moseley Jr., Robert J. Schoelkopf | https://arxiv.org/abs/2307.03169, https://arxiv.org/pdf/2307.03169, https://api.crossref.org/works/10.1038/s41567-024-02539-4, https://doi.org/10.1038/s41567-024-02539-4, https://www.nature.com/articles/s41567-024-02539-4 |
| Ma2023 | High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit; Nature: High-fidelity gates and mid-circuit erasure conversion in an atomic qubit | Shuo Ma, Genyue Liu, Pai Peng, Bichen Zhang, Sven Jandura, Jahan Claes, Alex P. Burgers, Guido Pupillo, Shruti Puri, Jeff D. Thompson | https://arxiv.org/abs/2305.05493, https://arxiv.org/pdf/2305.05493, https://api.crossref.org/works/10.1038/s41586-023-06438-1, https://doi.org/10.1038/s41586-023-06438-1 |
| McEwen2021 | Removing leakage-induced correlated errors in superconducting quantum error correction | M. McEwen, D. Kafri, Z. Chen, J. Atalaya, K. J. Satzinger, et al. (51), R. Barends | https://arxiv.org/abs/2102.06131, https://arxiv.org/pdf/2102.06131, https://api.crossref.org/works/10.1038/s41467-021-21982-y, https://doi.org/10.1038/s41467-021-21982-y |
| Miao2023 | Overcoming leakage in scalable quantum error correction; Nat. Phys.: Overcoming leakage in quantum error correction | Kevin C. Miao, Matt McEwen, Juan Atalaya, Dvir Kafri, Leonid P. Pryadko, Andreas Bengtsson, et al. (117 arXiv, 118 journal), Yu Chen | https://arxiv.org/abs/2211.04728, https://arxiv.org/pdf/2211.04728, https://api.crossref.org/works/10.1038/s41567-023-02226-w, https://doi.org/10.1038/s41567-023-02226-w |
| Varbanov2020 | Leakage detection for a transmon-based surface code | B. M. Varbanov, F. Battistel, B. M. Tarasinski, V. P. Ostroukh, T. E. O'Brien, L. DiCarlo, B. M. Terhal | https://arxiv.org/abs/2002.07119, https://arxiv.org/pdf/2002.07119, https://api.crossref.org/works/10.1038/s41534-020-00330-w, https://doi.org/10.1038/s41534-020-00330-w |
| Marques2023 | All-microwave leakage reduction units for quantum error correction with superconducting transmon qubits | J. F. Marques, H. Ali, B. M. Varbanov, M. Finkel, H. M. Veen, S. L. M. van der Meer, S. Valles-Sanclemente, N. Muthusubramanian, M. Beekman, N. Haider, B. M. Terhal, L. DiCarlo | https://arxiv.org/abs/2302.09876, https://arxiv.org/pdf/2302.09876, https://api.crossref.org/works/10.1103/PhysRevLett.130.250602, https://doi.org/10.1103/PhysRevLett.130.250602 |
| ChaoReichardt2018 | Quantum error correction with only two extra qubits | Rui Chao, Ben W. Reichardt | https://arxiv.org/abs/1705.02329, https://arxiv.org/pdf/1705.02329, https://api.crossref.org/works/10.1103/PhysRevLett.121.050502, https://doi.org/10.1103/PhysRevLett.121.050502 |
| ChamberlandBeverland2018 | Flag fault-tolerant error correction with arbitrary distance codes | Christopher Chamberland, Michael E. Beverland | https://arxiv.org/abs/1708.02246, https://arxiv.org/pdf/1708.02246, https://api.crossref.org/works/10.22331/q-2018-02-08-53, https://doi.org/10.22331/q-2018-02-08-53 |
| DursoSabina2022 | Flag Fault-Tolerant Error Correction with Qudits | Elijah Durso-Sabina | https://dspacemainprd01.lib.uwaterloo.ca/server/api/core/items/888cc1a3-bfdd-4b7c-88fa-a1c3ff258632, https://dspacemainprd01.lib.uwaterloo.ca/server/api/core/bitstreams/aad2ba2c-0e69-4384-83bf-fd112ceebc1a/content, https://export.arxiv.org/api/query?search_query=au:Durso_Sabina |
| Linke2017 | Fault-tolerant quantum error detection | Norbert M. Linke, Mauricio Gutierrez, Kevin A. Landsman, Caroline Figgatt, Shantanu Debnath, Kenneth R. Brown, Christopher Monroe | https://arxiv.org/abs/1611.06946, https://arxiv.org/pdf/1611.06946, https://api.crossref.org/works/10.1126/sciadv.1701074, https://doi.org/10.1126/sciadv.1701074 |
| Takita2017 | Experimental demonstration of fault-tolerant state preparation with superconducting qubits | Maika Takita, Andrew W. Cross, A. D. Córcoles, Jerry M. Chow, Jay M. Gambetta | https://arxiv.org/abs/1705.09259, https://arxiv.org/pdf/1705.09259, https://api.crossref.org/works/10.1103/PhysRevLett.119.180501, https://doi.org/10.1103/PhysRevLett.119.180501 |
| Andersen2020 | Repeated quantum error detection in a surface code | Christian Kraglund Andersen, Ants Remm, Stefania Lazar, Sebastian Krinner, Nathan Lacroix, Graham J. Norris, Mihai Gabureac, Christopher Eichler, Andreas Wallraff | https://arxiv.org/abs/1912.09410, https://arxiv.org/pdf/1912.09410, https://api.crossref.org/works/10.1038/s41567-020-0920-y, https://doi.org/10.1038/s41567-020-0920-y |
| Marques2022 | Logical-qubit operations in an error-detecting surface code | J. F. Marques, B. M. Varbanov, M. S. Moreira, H. Ali, N. Muthusubramanian, C. Zachariadis, F. Battistel, M. Beekman, N. Haider, W. Vlothuizen, A. Bruno, B. M. Terhal, L. DiCarlo | https://arxiv.org/abs/2102.13071, https://arxiv.org/pdf/2102.13071, https://api.crossref.org/works/10.1038/s41567-021-01423-9, https://doi.org/10.1038/s41567-021-01423-9 |
| Self2024 | Protecting expressive circuits with a quantum error detection code | Chris N. Self, Marcello Benedetti, David Amaro | https://arxiv.org/abs/2211.06703, https://arxiv.org/pdf/2211.06703, https://api.crossref.org/works/10.1038/s41567-023-02282-2, https://doi.org/10.1038/s41567-023-02282-2 |
| Bluvstein2024 | Logical quantum processor based on reconfigurable atom arrays | Dolev Bluvstein, Simon J. Evered, Alexandra A. Geim, et al. (21), Vladan Vuletić, Mikhail D. Lukin | https://arxiv.org/abs/2312.03982, https://arxiv.org/pdf/2312.03982, https://api.crossref.org/works/10.1038/s41586-023-06927-3, https://doi.org/10.1038/s41586-023-06927-3 |
| Gottesman2016 | Quantum fault tolerance in small experiments | Daniel Gottesman | https://arxiv.org/abs/1610.03507, https://arxiv.org/pdf/1610.03507, Crossref bibliographic query (https://api.crossref.org/works?query.bibliographic=...), web search |
| Grassl1997 | Codes for the Quantum Erasure Channel | Markus Grassl, Thomas Beth, Thomas Pellizzari | https://arxiv.org/abs/quant-ph/9610042, https://arxiv.org/pdf/quant-ph/9610042, https://api.crossref.org/works/10.1103/PhysRevA.56.33, https://doi.org/10.1103/PhysRevA.56.33 |
| Stricker2020 | Deterministic correction of qubit loss; Nature: Experimental deterministic correction of qubit loss | Roman Stricker, Davide Vodola, Alexander Erhard, Lukas Postler, Michael Meth, Martin Ringbauer, Philipp Schindler, Thomas Monz, Markus Müller, Rainer Blatt | https://arxiv.org/abs/2002.09532, https://arxiv.org/pdf/2002.09532, https://api.crossref.org/works/10.1038/s41586-020-2667-0, https://doi.org/10.1038/s41586-020-2667-0 |

## 3. Statements in lit-L7-erasure-flags.md checked

I compared every statement the notes make about each paper with that paper's arXiv
abstract, and spot-checked the numbers and quotes against the full text. Most
numbers in the notes are not in the abstracts, so the full-text check carries most
of the weight.

### 3.1 Statements that conflict with the paper

1. **Bluvstein 2024: the 72% GHZ fidelity is not "correction only".** The notes
   say "72(2)% with correlated decoding (correction, no postselection)" (A6), "72%
   fidelity with correction only" (Section 0, item 1) and "GHZ 72% (correction only)"
   (Table A). The paper's Fig. 3c caption shows three variants: "without
   postselecting on flags (nFT), postselecting on flags (FT), and postselecting on
   flags and stabilizers of the computation logical qubits, corresponding to error
   detection (EDFT)". The 72(2)% is the FT value: "the resulting GHZ state fidelity
   obtained using the fault-tolerant algorithm is 72(2)% (again using correlated
   decoding)". The Methods simulation that reproduces it (79%) includes "the FT
   state preparation with the ancilla logical flag". So the 72% already includes
   postselection on the ancilla-logical flags. Only the computation logicals'
   stabilizers are corrected rather than postselected. The abstract gives no GHZ
   numbers, so this is a full-text finding. The rest of A6 holds: 99.85% with
   flags plus stabilizers, about 90% at 50% acceptance, correction applied whenever
   sliding-scale detection is used, and a threshold on the correlated-matching
   weight.
2. **Marques 2023: "Removal above 99% in 220 ns" (C4) overstates the result.** The
   abstract says the LRU "effectively reduces leakage in the second- and
   third-excited transmon states with up to 99% efficacy in 220 ns". The full text
   gives R_h = 96.(1)% for |h> = |3> (both LRUs on, t_p = 220 ns) and
   R_f = 99.(2)% for the ancilla's |f>. The data-qubit f-LRUs ran at 84.(7)% and
   80.(3)% (Table S1). The authors chose the QEC operating point for R ≳ 80% and
   note that R > 99% for |f> is reached by increasing the pulse length or amplitude. Because the notes put
   the sentence right after the h-LRU, it reads as if |3> removal exceeded 99%. A
   correct version: "up to 99% for |f> in 220 ns; 96% for |h>". This matters for
   the qutrit use proposed in the notes, since |3> is the first qutrit leakage level.

### 3.2 Statements that need a qualifier

3. **Wu 2022 and Kubica 2023: "giving erased edges zero weight" is not in either
   paper.** Section 0 item 5 and B1 label it [lit]. Wu: syndromes "together with
   the locations of the erasure errors, are decoded with weighted Union Find (UF)
   decoder", which is optimal for pure erasures. Its edges are weighted by ln(p')
   "truncated to the nearest integer" (the notes say "rounded"). Kubica: MWPM
   decoder; false positives "affect the edge weights", estimated by replacing e
   with e + q+. The zero-weight rule is the standard way to feed erasures to a
   matching decoder, but neither text states it. It should carry [conj] or cite a
   paper that states it. The random-Pauli model is in Wu verbatim ("a Pauli error
   chosen uniformly at random from {I,X,Y,Z}^{⊗2}"), and Kubica describes a
   "fully-depolarizing two-qubit Pauli channel" at erased locations.
4. **Chou 2024, Table B: the erasure fraction is not the false-positive rate.** The
   "False positive" column lists "erasure fraction 6% (17% with two rounds)". The
   paper says false erasure assignments "are false positives and set a lower bound
   on the erasure fraction". The erasure fraction also contains real preparation
   leakage and errors during readout. It is therefore an upper bound on the
   false-positive rate. The cell should read "at most 6% (17% with two rounds);
   erasure fraction".
5. **Marques 2022, Section 0 item 6: "leakage, not T1/T2, set its detection rate"
   is stronger than the paper.** The paper says models without leakage "clearly fail
   to quantitatively capture the decay of the post-selected fraction" and that
   L1 ≈ 5% "produces a good match with experiment, suggesting that leakage
   significantly impacts the error-detection rate". The wording in A4 is accurate;
   item 6 should match it. Separately, A4 takes the gate fidelities from the main
   text (Z_L 97.9%, X_L 98.1%). The paper's Table I gives them the other way round
   (Z_L 98.1%, X_L 97.9%), so the paper contradicts itself here. A4's measurement
   fidelity X_L 96.4% (95.8% in a second sweep) is from the text; Table I gives a
   weighted average of 96.0%.
6. **Chamberland and Beverland 2018, D2: context of the quote.** The quote "This is
   primarily due to the large circuit depth in flag-EC protocols since idle qubits
   locations significantly outnumber other gate locations" comes from the paper's
   explanation of why the d = 3 flag-EC pseudo-thresholds rise when the idle error
   drops from p to p/10. The notes put it after the [[19,1,5]] versus d = 5
   surface-code comparison. The paper makes the same point for [[19,1,5]] ("flag-EC
   has a larger circuit depth compared to the other FTEC methods and is thus more
   sensitive to idle qubit errors"), so the substance holds. Only the attribution
   should be tidied.
7. **Stricker 2020, B7: one item labelled [lit] is an inference.** "T_X has weight
   1, so this demonstration code is not distance 2 against Pauli errors" sits under
   "Facts [lit]". The code is as stated (S_Z1 = Z1Z2, S_Z2 = Z1Z3,
   S_X1 = X1X2X3X4, T_Z = Z1Z4, T_X = X4), and the inference is correct [analytic]:
   X4 commutes with all three generators, is not in the stabilizer group, and
   anticommutes with T_Z, so it is a weight-1 logical. It should be labelled [ana].
8. **Grassl 1997, B6: optional addition.** The notes read the abstract only. They
   match it: four qubits are necessary and sufficient to encode one qubit and
   correct one erasure. The full text (v2) adds that "the extended code encodes not
   only one, but two qubits", which makes it the [[4,2,2]] code used by Linke,
   Takita and Gottesman.

### 3.3 Spot checks that agree with the full text

Every item below matches the arXiv full text. Abstract-level claims are marked (abs).

- Wu 2022: 98% conversion and threshold 0.937% to 4.15% (abs); 61% BBR decays, 34%
  radiative decay to the ground state, 5% to the qubit subspace; 5.13% at R_e = 1;
  ν = 4.35(2); 3.80(2)% with p_m = 0.005; d rounds plus one perfect round; erased
  pair replaced by atoms in I/2, modelled as a uniform random Pauli.
- Kubica 2023: correctable region about 3.5 times larger; p up to 0.0051 at
  e = 0.01, a 5.2-fold gain; p becomes p + e q-, e becomes e + q+;
  e ≈ 6e-3 (2 × 110 ns gate, 400 ns measurement, T1 = 100 µs); q- ≈ 1e-3 giving
  6e-5; T_meas/T_φ^meas ≈ 1e-4; p ≈ 1.6e-4.
- Levine 2024: erasure 2.19(2)e-3 per gate, residual about 40 times lower, check
  dephasing under 0.1% (abs); residual 5.06(6)e-5 per X90, bias 43(1); 11 erasure
  checks during RB; erasure lifetime 29.3(4) µs; T1 = 906(15) µs; echo T2 from
  543(23) µs to 1.25(8) ms; 540 ns pulse, 880 ns check; false positives 0.58(1)%
  (|1_L>) and about 0.8% (|+_L>); false negatives 1.54(1)%;
  T_check/T_eras ≈ 2.9%; missed erasure 0.04%; postselection quote.
- Chou 2024: SPAM errors at the 0.01% level, over 99% of decays detected, phase
  flips 6 times and bit flips at least 140 times rarer than decays (abs);
  misassignment (1.8 ± 0.3)e-4; erasure fraction (6.03 ± 0.05)e-2;
  leakage-detection error (7.7 ± 0.3)e-3; two rounds (4 ± 2)e-5 and
  (1.2 ± 0.1)e-3 with erasure fraction 0.17; p_φ = 0.023% (Ramsey) and 0.019%
  (echo) per µs; end-of-line detection.
- Ma 2023: gate fidelities 0.9990(1) and 0.980(1), detection disturbance under
  1e-5 (abs); 2.96(12) s lifetime; 56(4)% and about 33% conversion; P_d = 7e-6,
  1.4e-4 at 420 µs; 60% predicted leakage, half of the Rydberg leakage detected;
  p11/p00 > 15(9); atom replacement "after measuring the error syndromes".
- McEwen 2021: 250 ns reset with error about 1e-3; forty-fold (measure) and 2.4-fold
  (data) increase in γ↓; |21>-|03> resonance; Λ_bit 1.98 without reset and 2.80
  with reset over 30 rounds; correlations over more than 10 rounds without reset.
- Miao 2023: ten-fold data-qubit leakage reduction and under 1e-3 average (abs); decay
  constant about 4.4 cycles; phase shift about 0.65π; 18% |30>↔|12> transport;
  "less than 1% cause the logical error probability to rise above 40%" versus
  about 5% for Pauli errors (d = 21, 60 cycles, no reset).
- Varbanov 2020: 47% discarded to restore break-even (abs); L1 = 0.125%; about 9
  (data) and 11 (ancilla) cycles; HMM optimality about 74% (data) and 35% (bulk
  ancillas); F = 88.4%; weight-6 superchecks; distance 3 to 2; MWPM and the
  density-matrix upper bound both above break-even.
- Marques 2023: 71(9) kHz AC-Stark shift; F_avg = 98.(9)% versus 99.(2)% idling;
  detection probability rises about 8% over 50 rounds without LRUs and about 2%
  with them; steady-state leakage about 10% without LRUs, at most about 1% with
  them; 500 and 720 ns cycles; heralded-LRU quote.
- Chao and Reichardt 2018: Section V quote on [[n,n-2,2]] detection; ancilla read
  as a two-qubit Z-error-detecting code; "we do not introduce memory errors on
  resting qubits".
- Chamberland and Beverland 2018: pseudo-thresholds 7.09e-5, 3.39e-5, 3.29e-4,
  6.29e-4, 1.14e-5 and 9.41e-4 with the stated qubit counts and time steps;
  surface-code hook-error footnote.
- Durso-Sabina 2022: abstract (repository and PDF); "SUM ≠ SUM^-1" quote; stabilizer
  XZZ^-1X^-1I with cyclic shifts; SUM^-1 propagation rules (3.18) onward; "a weight
  one error multiplied by a stabilizer"; Chapter 4 ancilla count scaling with
  distance.
- Linke 2017: yields 91.1, 77.8 and 65.2%; Table 1 populations; about 8% and 20%
  discard; 5000-6000 runs; gate fidelities 99.1(5)% and 97(1)%; fitted errors 0.50,
  1.0 and 1.4%; L_a 0.7% versus 1.2% for a physical qubit; postselection quote.
- Takita 2017: Table II acceptance 0.7566-0.7897 and fidelity 0.9619-0.9726. From
  Table II I get protected-qubit errors of 0.59-1.34% and gauge errors of
  2.35-3.18% [analytic], which match the notes' rounded ranges. Quote on repeated
  syndrome measurements; "by construction".
- Andersen 2020: 96.1% average logical fidelity (abs); 25.1% success versus 50%;
  P_L = 0.717; 98.2, 97.3, 94.2 and 94.8%; 40-50% no-error probability per cycle;
  success about 1e-4 after 10 cycles; 3.1% ± 0.45% and 2.6 ± 1.3%; lifetimes 62.7
  and 72.5 µs versus 16.8 and 21.5 µs; 1.92 µs cycle; ambiguity quote.
- Marques 2022: initialization 99.83, 99.97, 97.02/99.78 and 95.54/99.64%; Z_L 99.4%,
  Y_L 87.5%; γ_pip about 45%, ratio about 97%; 840 and 1000 ns cycles; L1 ≈ 5%;
  postselection and fault-tolerance definition quotes.
- Self 2024: k + 2 qubits, non-fault-tolerant logical gates, 4 to 256 layers, QV 2^8
  (abs); discard rate 86.8 ± 0.7%; ABBB...BA ordering; discard rule quote; XX, YY,
  ZZ after MS undetectable; exhaustive check up to k = 16; one syndrome round
  between unitary and inverse for 32-256 layers; conditional-exit and qubit-loss
  remarks.
- Bluvstein 2024: 280 qubits, d = 3 to 7, 48 logical qubits, 228 two-qubit and 48
  CCZ gates (abs); |0_L> 99.91%; GHZ 99.85% (EDFT) and about 90% at 50% acceptance;
  XEB 0.156(2) to 0.616(7); 23,545 of 138,626 shots; [[8,3,2]] distance 2 (Z) and
  4 (X); MWPM fidelity falls with distance, partly from nFT preparation; Gurobi
  mixed-integer program.
- Gottesman 2016: criterion and 5-qubit ring (abs); bound p < (4+2T)/(16T² + 122T
  + 225); 1.6%, 0.79% and 0.12% for T = 1, 10 and 100; postselection, "deceptive"
  and SWAP quotes.
- Stricker 2020: code and logicals as in item 7; detection efficiency 96.5(4)%,
  false positives 3(+1-1)%, false negatives 1(+1-1)%; process fidelity 90(2)% in
  the no-loss case at a loss rate of about 20%; merged Z2Z3 and shrunk X2X3X4 with
  a Pauli-frame update for a -1 outcome.
