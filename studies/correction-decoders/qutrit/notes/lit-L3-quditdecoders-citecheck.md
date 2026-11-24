# Citation check for lit-L3-quditdecoders.md

Independent check of the 29 citations in the L3 list (topic: decoders for qudit and
qutrit codes), and of the statements about those papers in `lit-L3-quditdecoders.md`.
Checked 2026-10-28 on the login node. No computation beyond downloading and text
extraction.

Labels. Every verdict and every bibliographic field below is **from literature**
[lit]: it rests on the pages I fetched myself in this task, listed per entry in
Table 2. The one argument of my own is marked [analytic].

## Method

- arXiv. For each of the 29 IDs I downloaded `https://arxiv.org/abs/<id>` and read
  the `citation_title`, `citation_author`, `citation_date` and `citation_doi` meta
  tags, the journal-ref field and the submission history. One arXiv API query for all
  29 IDs (`https://export.arxiv.org/api/query?id_list=...`) returned the same titles,
  author counts and DOIs.
- DOIs. For each DOI I read the Crossref record `https://api.crossref.org/works/<doi>`
  (title, authors, journal, volume, issue, pages or article number, date) and checked
  that `https://doi.org/<doi>` redirects to the publisher. All 23 DOIs resolve: the 19
  in the list and 4 found here.
- Journal versions. For the 10 entries listed without a DOI I ran a Crossref title
  search (`https://api.crossref.org/works?query.bibliographic=...`) and a Semantic
  Scholar batch query by arXiv ID (`https://api.semanticscholar.org/graph/v1/paper/batch`).
  This found journal versions of Sabo et al. and Iyer-Poulin, and the DOI of
  Duclos-Cianci-Poulin 2014. The other 7 have no journal version in either index.
- Publisher pages. IEEE Xplore returned empty pages and IOPscience redirected to a bot
  check, so for IEEE and IOP items the Crossref record is the publisher-side source.
  The APS, Nature, Quantum, EMS and Rinton DOIs redirect to the expected pages.
- Full text, read only to settle points raised by the notes: Iyer-Poulin PDF
  (abstract wording), Watson et al. v2 PDF Sec. III (noise model), Marks et al. v2 PDF
  (abstract), Gu-Wang-Kubica HTML, Kabir et al. v2 HTML.

Result: 26 confirmed, 3 corrected, 0 not found. No statement in the notes contradicts
an abstract. Three entries in the notes lack a journal version or DOI, and two
statements need a qualifier (Section 3).

## 1. Verdicts

| # | Key | Verdict | arXiv ID, versions | DOI | Venue, year (verified) | What differs |
|---|---|---|---|---|---|---|
| 1 | DuclosCianciPoulin2013 | confirmed | 1302.3638, v1 2013-02-14 | 10.1103/PhysRevA.87.062338 | Phys. Rev. A 87, 062338 (2013) | Journal title is "Kitaev's Z_d-code threshold estimates". The entry gives the arXiv title. |
| 2 | Anwar2014 | confirmed | 1311.4895, v1 2013-11-19 | 10.1088/1367-2630/16/6/063038 | New J. Phys. 16, 063038 (2014) | None. Both titles are right. |
| 3 | Watson2015 | confirmed | 1411.3028, v1 2014-11-12, v2 2015-09-14 | 10.1103/PhysRevA.92.032309 | Phys. Rev. A 92, 032309 (2015) | Journal title drops the leading "A". The notes already say so. |
| 4 | Hutter2015 | confirmed | 1410.4478, v1 2014-10-16, v2 2015-02-26 | 10.1088/1367-2630/17/3/035017 | New J. Phys. 17, 035017 (2015) | None. |
| 5 | Kasai2012 | confirmed | 1007.1778, v1 2010-07-11, v2 2011-07-13 | 10.1109/TIT.2011.2167593 | IEEE Trans. Inf. Theory 58(2), 1223-1230 (Feb. 2012) | None. |
| 6 | BravyiSucharaVargo2014 | confirmed | 1405.4883, v1 2014-05-19 | 10.1103/PhysRevA.90.032326 | Phys. Rev. A 90, 032326 (2014) | None. |
| 7 | Chubb2021 | confirmed | 2101.04125, v1 2021-01-11, v3 2021-10-13 | none | arXiv preprint | No journal version found. |
| 8 | ChubbFlammia2021 | confirmed | 1809.10704, v1 2018-09-27, v2 2019-03-18 | 10.4171/AIHPD/105 | Ann. Inst. Henri Poincaré D 8(2), 269-321 (2021) | None. |
| 9 | Piveteau2024 | confirmed | 2310.10722, v1 2023-10-16, v2 2024-10-08 | 10.1103/PRXQuantum.5.040303 | PRX Quantum 5, 040303 (2024) | Journal title is hyphenated: "Tensor-Network Decoding Beyond 2D". |
| 10 | Moussa2016 | confirmed | 1603.02286, v1 2016-03-07, v3 2016-10-15 | 10.1103/PhysRevA.94.042316 | Phys. Rev. A 94, 042316 (2016) | None. |
| 11 | Keppens2025 | confirmed | 2502.05992, v1 2025-02-09, v2 2025-09-05 | 10.1103/2w52-qd2j | Phys. Rev. A 112, 032435 (2025) | Last author is spelled "Bart Sorée". Journal title: "Qudit vs. qubit: Simulated performance of error-correction codes in higher dimensions". |
| 12 | Kabir2025Sdim | confirmed | 2511.12777, v1 2025-11-16, v2 2026-05-12 | none | arXiv preprint | Last author is spelled "Bart Sorée". No journal version found. |
| 13 | HeimSvoreHastings2016 | confirmed | 1609.06373, v1 2016-09-20 | none | arXiv preprint | No journal version found. |
| 14 | Sabo2021 | **corrected** | 2106.08251, v1 2021-06-15, v2 2022-01-17 | 10.1109/TQE.2024.3401857 (added) | IEEE Trans. Quantum Eng. 5, 1-30 (2024) | Published, but the arXiv page has no journal-ref. Year of the journal version is 2024. |
| 15 | LaiKuo2021 | confirmed | 2104.00304, v1 2021-04-01, v3 2021-10-26 | 10.1109/TQE.2021.3113936 | IEEE Trans. Quantum Eng. 2, 2103615 (2021) | None. |
| 16 | Andriyanova2012 | confirmed | 1202.3338, v1 2012-02-15 | none | arXiv preprint | The same authors published "Quantum LDPC codes obtained by non-binary constructions", ISIT 2012, pp. 343-347, DOI 10.1109/ISIT.2012.6284205. Probably the conference version; I could not read its abstract, so this is not confirmed. |
| 17 | DuclosCianciPoulin2010 | confirmed | 0911.0581, v1 2009-11-03, v2 2010-02-05 | 10.1103/PhysRevLett.104.050504 | Phys. Rev. Lett. 104, 050504 (2010) | None. |
| 18 | DuclosCianciPoulin2014 | **corrected** | 1304.6100, v1 2013-04-22 | 10.26421/QIC14.9-10-1 (added) | Quantum Inf. Comput. 14(9&10), 721-740 (2014) | DOI was missing. All other fields are right. |
| 19 | BravyiHaah2013 | confirmed | 1112.3252, v1 2011-12-14 | 10.1103/PhysRevLett.111.200501 | Phys. Rev. Lett. 111, 200501 (2013) | None. Both titles are right. |
| 20 | DelfosseNickerson2021 | confirmed | 1709.06218, v1 2017-09-19, v3 2021-11-24 | 10.22331/q-2021-12-02-595 | Quantum 5, 595 (2021) | None. |
| 21 | IyerPoulin2013 | **corrected** | 1310.3235, v1 2013-10-11 | 10.1109/TIT.2015.2422294 (added) | IEEE Trans. Inf. Theory 61(9), 5209-5223 (2015) | Published, but the arXiv page has no journal-ref. Year of the journal version is 2015. |
| 22 | GuWangKubica2026 | confirmed | 2606.18145, v1 2026-06-16 | none | arXiv preprint | None. |
| 23 | Cao2025 | confirmed | 2501.03582, v1 2025-01-07 | none | arXiv preprint | 11 authors. The first three are as listed. No journal version found. |
| 24 | SerraPeralta2026 | confirmed | 2505.13599, v1 2025-05-19, v4 2026-07-06 | 10.1103/sk5y-25b1 | PRX Quantum 7, 010335 (2026) | Journal title is in title case. |
| 25 | Iqbal2025 | confirmed | 2411.04185, v1 2024-11-06 | 10.1038/s41467-025-61391-z | Nat. Commun. 16, 6301 (2025) | 19 authors. The first three are as listed. |
| 26 | MaHanksKim2023 | confirmed | 2303.16837, v1 2023-03-29 | 10.1103/PhysRevLett.131.200602 | Phys. Rev. Lett. 131, 200602 (2023) | None. |
| 27 | Marks2017 | confirmed | 1701.02335, v1 2017-01-09, v2 2017-11-16 | 10.1088/1367-2630/aa939a | New J. Phys. 19, 113022 (2017) | None. The arXiv abstract field is out of date (Section 3). |
| 28 | BreuckmannBurton2024 | confirmed | 2202.06647, v1 2022-02-14, v3 2024-06-10 | 10.22331/q-2024-06-13-1372 | Quantum 8, 1372 (2024) | None. |
| 29 | Nourozi2025 | confirmed | 2506.03397, v1 2025-06-03 | none | arXiv preprint | Exact arXiv title ends "over $\mathbb{F}_q$". No journal version found. |

## 2. Verified records and pages fetched

Titles are the arXiv titles, with the journal title after a semicolon where it differs
in more than letter case. Authors are as on the arXiv page, which agrees with Crossref
for every entry that has a DOI.

| Key | Title | Authors | Pages fetched |
|---|---|---|---|
| DuclosCianciPoulin2013 | Kitaev's Z_d-Codes Threshold Estimates; PRA: Kitaev's Z_d-code threshold estimates | Guillaume Duclos-Cianci, David Poulin | https://arxiv.org/abs/1302.3638, https://api.crossref.org/works/10.1103/PhysRevA.87.062338, https://doi.org/10.1103/PhysRevA.87.062338 |
| Anwar2014 | Efficient Decoders for Qudit Topological Codes; NJP: Fast decoders for qudit topological codes | Hussain Anwar, Benjamin J. Brown, Earl T. Campbell, Dan E. Browne | https://arxiv.org/abs/1311.4895, https://api.crossref.org/works/10.1088/1367-2630/16/6/063038, https://doi.org/10.1088/1367-2630/16/6/063038 |
| Watson2015 | A fast fault-tolerant decoder for qubit and qudit surface codes; PRA: Fast fault-tolerant decoder for qubit and qudit surface codes | Fern H. E. Watson, Hussain Anwar, Dan E. Browne | https://arxiv.org/abs/1411.3028, https://arxiv.org/pdf/1411.3028, https://api.crossref.org/works/10.1103/PhysRevA.92.032309, https://doi.org/10.1103/PhysRevA.92.032309 |
| Hutter2015 | Improved HDRG decoders for qudit and non-Abelian quantum error correction | Adrian Hutter, Daniel Loss, James R. Wootton | https://arxiv.org/abs/1410.4478, https://api.crossref.org/works/10.1088/1367-2630/17/3/035017, https://doi.org/10.1088/1367-2630/17/3/035017 |
| Kasai2012 | Quantum Error Correction beyond the Bounded Distance Decoding Limit | Kenta Kasai, Manabu Hagiwara, Hideki Imai, Kohichi Sakaniwa | https://arxiv.org/abs/1007.1778, https://api.crossref.org/works/10.1109/TIT.2011.2167593, https://doi.org/10.1109/TIT.2011.2167593 |
| BravyiSucharaVargo2014 | Efficient Algorithms for Maximum Likelihood Decoding in the Surface Code | Sergey Bravyi, Martin Suchara, Alexander Vargo | https://arxiv.org/abs/1405.4883, https://api.crossref.org/works/10.1103/PhysRevA.90.032326, https://doi.org/10.1103/PhysRevA.90.032326 |
| Chubb2021 | General tensor network decoding of 2D Pauli codes | Christopher T. Chubb | https://arxiv.org/abs/2101.04125, Crossref title search, Semantic Scholar batch query |
| ChubbFlammia2021 | Statistical mechanical models for quantum codes with correlated noise | Christopher T. Chubb, Steven T. Flammia | https://arxiv.org/abs/1809.10704, https://api.crossref.org/works/10.4171/AIHPD/105, https://doi.org/10.4171/AIHPD/105 |
| Piveteau2024 | Tensor Network Decoding Beyond 2D; PRXQ: Tensor-Network Decoding Beyond 2D | Christophe Piveteau, Christopher T. Chubb, Joseph M. Renes | https://arxiv.org/abs/2310.10722, https://api.crossref.org/works/10.1103/PRXQuantum.5.040303, https://doi.org/10.1103/PRXQuantum.5.040303 |
| Moussa2016 | Transversal Clifford gates on folded surface codes | Jonathan E. Moussa | https://arxiv.org/abs/1603.02286, https://api.crossref.org/works/10.1103/PhysRevA.94.042316, https://doi.org/10.1103/PhysRevA.94.042316 |
| Keppens2025 | Qudit vs. Qubit: Simulated performance of error correction codes in higher dimensions; PRA: Qudit vs. qubit: Simulated performance of error-correction codes in higher dimensions | James Keppens, Quinten Eggerickx, Vukan Levajac, George Simion, Bart Sorée | https://arxiv.org/abs/2502.05992, https://api.crossref.org/works/10.1103/2w52-qd2j, https://doi.org/10.1103/2w52-qd2j |
| Kabir2025Sdim | Sdim: A Qudit Stabilizer Simulator | Adeeb Kabir, Steven Nguyen, Sohan Ghosh, James Keppens, Tijil Kiran, Isaac H. Kim, Yipeng Huang, Bart Sorée | https://arxiv.org/abs/2511.12777, https://arxiv.org/html/2511.12777v2, Crossref title search, Semantic Scholar batch query |
| HeimSvoreHastings2016 | Optimal Circuit-Level Decoding for Surface Codes | Bettina Heim, Krysta M. Svore, Matthew B. Hastings | https://arxiv.org/abs/1609.06373, Crossref title search, Semantic Scholar batch query |
| Sabo2021 | Trellis Decoding For Qudit Stabilizer Codes And Its Application To Qubit Topological Codes | Eric Sabo, Arun B. Aloshious, Kenneth R. Brown | https://arxiv.org/abs/2106.08251, Crossref title search, https://api.crossref.org/works/10.1109/TQE.2024.3401857, https://doi.org/10.1109/TQE.2024.3401857, Semantic Scholar batch query |
| LaiKuo2021 | Log-domain decoding of quantum LDPC codes over binary finite fields | Ching-Yi Lai, Kao-Yueh Kuo | https://arxiv.org/abs/2104.00304, https://api.crossref.org/works/10.1109/TQE.2021.3113936, https://doi.org/10.1109/TQE.2021.3113936 |
| Andriyanova2012 | New constructions of CSS codes obtained by moving to higher alphabets | Iryna Andriyanova, Denise Maurice, Jean-Pierre Tillich | https://arxiv.org/abs/1202.3338, Crossref title search, https://api.crossref.org/works/10.1109/ISIT.2012.6284205, https://doi.org/10.1109/ISIT.2012.6284205, Semantic Scholar batch query, https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ISIT.2012.6284205 |
| DuclosCianciPoulin2010 | Fast Decoders for Topological Quantum Codes | Guillaume Duclos-Cianci, David Poulin | https://arxiv.org/abs/0911.0581, https://api.crossref.org/works/10.1103/PhysRevLett.104.050504, https://doi.org/10.1103/PhysRevLett.104.050504 |
| DuclosCianciPoulin2014 | Fault-Tolerant Renormalization Group Decoder for Abelian Topological Codes | Guillaume Duclos-Cianci, David Poulin | https://arxiv.org/abs/1304.6100, Crossref title search, https://api.crossref.org/works/10.26421/QIC14.9-10-1, https://doi.org/10.26421/QIC14.9-10-1, Semantic Scholar batch query |
| BravyiHaah2013 | Analytic and numerical demonstration of quantum self-correction in the 3D Cubic Code; PRL: Quantum Self-Correction in the 3D Cubic Code Model | Sergey Bravyi, Jeongwan Haah | https://arxiv.org/abs/1112.3252, https://api.crossref.org/works/10.1103/PhysRevLett.111.200501, https://doi.org/10.1103/PhysRevLett.111.200501 |
| DelfosseNickerson2021 | Almost-linear time decoding algorithm for topological codes | Nicolas Delfosse, Naomi H. Nickerson | https://arxiv.org/abs/1709.06218, https://api.crossref.org/works/10.22331/q-2021-12-02-595, https://doi.org/10.22331/q-2021-12-02-595 |
| IyerPoulin2013 | Hardness of decoding quantum stabilizer codes | Pavithran Iyer, David Poulin | https://arxiv.org/abs/1310.3235, https://arxiv.org/pdf/1310.3235, Crossref title search, https://api.crossref.org/works/10.1109/TIT.2015.2422294, https://doi.org/10.1109/TIT.2015.2422294, Semantic Scholar batch query |
| GuWangKubica2026 | A polynomial-time approximation scheme for minimum-weight decoding of topological codes | Shouzhen Gu, Lily Wang, Aleksander Kubica | https://arxiv.org/abs/2606.18145, https://arxiv.org/html/2606.18145, Crossref title search, Semantic Scholar batch query |
| Cao2025 | Exact Decoding of Repetition Code under Circuit Level Noise | Hanyan Cao, Shoukuan Zhao, Dongyang Feng, Zisong Shen, Haisheng Yan, Tang Su, Weijie Sun, Huikai Xu, Feng Pan, Haifeng Yu, Pan Zhang | https://arxiv.org/abs/2501.03582, Crossref title search, Semantic Scholar batch query |
| SerraPeralta2026 | Decoding across transversal Clifford gates in the surface code | Marc Serra-Peralta, Mackenzie H. Shaw, Barbara M. Terhal | https://arxiv.org/abs/2505.13599, https://api.crossref.org/works/10.1103/sk5y-25b1, https://doi.org/10.1103/sk5y-25b1 |
| Iqbal2025 | Qutrit Toric Code and Parafermions in Trapped Ions | Mohsin Iqbal, Anasuya Lyons, Chiu Fan Bowen Lo, Nathanan Tantivasadakarn, Joan Dreiling, Cameron Foltz, Thomas M. Gatterman, Dan Gresh, Nathan Hewitt, Craig A. Holliman, Jacob Johansen, Brian Neyenhuis, Yohei Matsuoka, Michael Mills, Steven A. Moses, Peter Siegfried, Ashvin Vishwanath, Ruben Verresen, Henrik Dreyer | https://arxiv.org/abs/2411.04185, https://api.crossref.org/works/10.1038/s41467-025-61391-z, https://doi.org/10.1038/s41467-025-61391-z |
| MaHanksKim2023 | Non-Pauli errors can be efficiently sampled in qudit surface codes | Yue Ma, Michael Hanks, M. S. Kim | https://arxiv.org/abs/2303.16837, https://api.crossref.org/works/10.1103/PhysRevLett.131.200602, https://doi.org/10.1103/PhysRevLett.131.200602 |
| Marks2017 | Comparison of memory thresholds for planar qudit geometries | Jacob Marks, Tomas Jochym-O'Connor, Vlad Gheorghiu | https://arxiv.org/abs/1701.02335, https://arxiv.org/pdf/1701.02335v2, https://api.crossref.org/works/10.1088/1367-2630/aa939a, https://doi.org/10.1088/1367-2630/aa939a |
| BreuckmannBurton2024 | Fold-Transversal Clifford Gates for Quantum Codes | Nikolas P. Breuckmann, Simon Burton | https://arxiv.org/abs/2202.06647, https://api.crossref.org/works/10.22331/q-2024-06-13-1372, https://doi.org/10.22331/q-2024-06-13-1372 |
| Nourozi2025 | Reinforcement Learning Enhanced Greedy Decoding for Quantum Stabilizer Codes over $\mathbb{F}_q$ | Vahid Nourozi | https://arxiv.org/abs/2506.03397, Crossref title search, Semantic Scholar batch query |

## 3. Statements in lit-L3-quditdecoders.md checked against the abstracts

I compared every statement the notes make about each paper with that paper's arXiv
abstract. None contradicts an abstract. Most numbers in the notes (thresholds and
complexities taken from the body or from figures) do not appear in the abstracts, so
this check does not cover them, apart from the full-text spot checks listed in the
Method section.

### Needs a qualifier or an added reference

1. Gu, Wang, Kubica (notes Sec. 2.4). The notes say circuit-level noise "is claimed
   'analogous' but only phenomenological noise is verified". The abstract claims more:
   the scheme covers "certain higher-dimensional topological codes and quantum memories,
   including the toric code with phenomenological or circuit-level noise". The body
   supports the notes. It says "for concreteness, we verify this condition for the
   quantum memory setting with the toric code and phenomenological noise" and "the
   scenario with circuit-level noise is analogous, albeit with a more complex decoding
   graph". The notes should quote both, so that a reader who checks the abstract does
   not think the notes misreport it. The notes' other details hold in the HTML:
   Theorem 1 gives a randomized algorithm with time L^2 (log L)^{O(1/eps)}, charges
   lie in Z_2^{2 alpha}, and the text never mentions qudits.
2. Sabo, Aloshious, Brown (notes Sec. 4.2) and Iyer, Poulin (notes Sec. 2.4). The
   notes cite only the arXiv versions ("no journal reference on arXiv" is literally
   true for Sabo et al.). Both are published: IEEE Trans. Quantum Eng. 5, 1-30 (2024),
   DOI 10.1109/TQE.2024.3401857, and IEEE Trans. Inf. Theory 61(9), 5209-5223 (2015),
   DOI 10.1109/TIT.2015.2422294.
3. Duclos-Cianci, Poulin 2014 (notes Sec. 1.1). Add DOI 10.26421/QIC14.9-10-1.
4. Kabir et al. (notes Sec. 5.3). Every quoted sentence is in the v2 HTML verbatim:
   "especially delicate in the small, distance-two systems", "must distinguish between
   different nontrivial Pauli powers", "contrary to the original paper's claims, both
   MWPM and BM decoders report strictly lower logical error curves as q increases",
   "reproduced here exactly", "interval-check postselection", and "[[5,1,2]]_3
   detection surface code" with Moussa as its Ref. [74]. The Fig. 11 caption confirms
   belief-matching decoding and 1.5 million shots per point. The value p_s = 1.56e-5
   does not occur in the HTML text, so it was presumably read off Fig. 11 and should
   carry the [lit, fig] label.

### Checked and correct

- Iyer, Poulin. The arXiv abstract field ends "it is #P."; the PDF abstract reads "it
  is #P-Complete." The notes quote the PDF correctly.
- Marks et al. The notes say the arXiv abstract and the v2 PDF disagree, and they do.
  The arXiv abstract field, which the arXiv API also returns, still carries the v1
  text: code-capacity depolarizing noise, 9.75% against 11.4%, up to 75% higher,
  plateau 17.1%. The v2 PDF, which the arXiv comment calls the published version,
  says generalized bit flip, 5.6% against 8.0%, up to 112% higher, plateau 11.9%.
  The v2 numbers are the ones to use. IOPscience blocked me, so I did not see the
  journal's own abstract.
- Watson et al. The notes' noise model matches Sec. III of the v2 PDF: X^k, then Z^k,
  each non-zero power with probability p/(d-1), and each syndrome outcome shifted by
  each non-zero k with probability p/(d-1). The abstract's "more than 8%" with
  enhancements at high d agrees with the notes' 8.2-8.3%; the PDF introduction says
  8.3%.
- Duclos-Cianci, Poulin 2010. The abstract says the algorithm "runs in time log L"; the
  notes call this parallel run time. That reading is right, since merely reading the
  order-L^2 syndrome values takes order L^2 sequential steps [analytic].
- Iqbal et al. The journal abstract in the Crossref record adds that each qutrit is
  encoded in two qubits, as the notes say, and gives a per-qutrit fidelity exceeding
  96.5(3)%. The notes' statements about Quantinuum H2 and the 11% leakage
  postselection are not in either abstract and I did not check them.
- Keppens et al. The abstract reports thresholds "of the order of 10^-4" for q = 2, 3,
  5 once a flag qudit is added. The notes' 4.95e-4, 3.24e-4 and 2.32e-4 agree.
- Chubb 2021. The abstract speaks of "a code consisting of n qubits", which supports
  the notes' correction that the paper does not itself treat qudits.
- Hutter et al. The abstract confirms Theta(L^{2/3}) to Omega(L^{1-eps}), poly(log L)
  parallel run time with perfect measurements, and O(1) average run time in the
  continuous Abelian case.
- Duclos-Cianci, Poulin 2014. The abstract confirms 1.9(4)% against 2.9% for matching
  under a 3D bit-flip channel, run time linear in space-time volume, logarithmic time
  in parallel.
- Anwar et al., Bravyi-Haah, Bravyi-Suchara-Vargo, Chubb-Flammia, Piveteau et al.,
  Moussa, Kasai et al., Lai-Kuo, Andriyanova et al., Heim et al., Sabo et al., Cao et
  al., Serra-Peralta et al., Delfosse-Nickerson, Ma-Hanks-Kim, Breuckmann-Burton,
  Nourozi. Each abstract matches what the notes say about the paper's setting, method
  and headline result. The Bravyi-Haah sentence the notes quote is verbatim.

## 4. Cosmetic fixes

- Spell the last author of Keppens2025 and Kabir2025Sdim "Sorée".
- When citing a journal version, use its title where it differs from the arXiv title:
  DuclosCianciPoulin2013, Watson2015, Piveteau2024, Keppens2025, BravyiHaah2013 and
  Anwar2014 (Table 2). The other journal titles differ from the arXiv titles only in
  letter case.
