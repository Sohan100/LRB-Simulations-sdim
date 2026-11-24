# Citation check: L4 (soft-output decoding, complementary gap, optimal reject rules)

This is an independent check of the 23-entry citation list behind
`lit-L4-softoutput.md`, done on 2026-10-28. I also read that notes file for
statements that contradict the papers' abstract pages, and spot-checked its
main full-text quotes and numbers.

**Label.** Every verdict and flag below is **from literature [LIT]**. Each
one rests on pages I fetched in this task (fetch log at the end). Where I say
that a paraphrase in the notes mislabels or drops something, the source text
is quoted next to it.

## Method

- **arXiv records (19 entries).** I fetched all 19 IDs from the arXiv API
  (`export.arxiv.org/api/query?id_list=...`) and from
  `https://arxiv.org/abs/<id>`. All abstract pages returned HTTP 200. From the
  raw HTML I read the `citation_*` meta tags, the Journal-ref and DOI rows, the
  submission history and the abstract.
- **DOIs (14 cited, 3 found).** For each DOI I fetched the Crossref record
  (`api.crossref.org/works/<doi>`): title, authors, container, volume, issue,
  pages or article number, and dates. For 11 of them I also checked that
  `https://doi.org/<doi>` redirects to the expected publisher page. All did.
- **Publisher and other pages.**
  - JMLR abstract page and PDF for Franc et al.
  - The IBM Research page for Chow 1970.
  - The Technion preprint of Hof, Sason and Shamai.
  - The APS pages for the new Chen et al. and Staples et al. DOIs.
  - science.org for Rosenfeld et al. returned HTTP 403, so that entry rests
    on Crossref and the doi.org redirect.
- **arXiv-only entries (8).** I ran a Crossref bibliographic search and a
  title search for all eight, plus a web search for Meister et al. and for
  Gidney, Shutty and Jones. This found journal versions for three of them:
  Chen et al., Staples et al. and Rosenfeld et al.
- **Full-text spot checks.**
  - arXiv HTML for 2410.07598v3, 2409.17595v1, 2510.05222v3, 2405.07433v2,
    2405.03766v1, 2512.13908v1, 2312.04522v1, 2603.08697v1 and 2510.05795v4.
  - arXiv PDFs for 1302.2669v2, 2212.00813v1, 2107.13589v1, 2112.03785v1,
    0711.2501v1, 1504.03740v2 and quant-ph/0410199v2.
  - I used these only to check quotes and numbers in the notes that go
    beyond the abstracts.

## Summary

- **18 confirmed, 5 corrected, 0 not found.**
  - All 19 arXiv IDs exist and point to the cited paper.
  - All 14 cited DOIs resolve and match the cited work.
  - Author lists match the arXiv and Crossref records everywhere. Bombín
    carries accents in both. The one exception is the Science version of
    Rosenfeld et al., which lists one more author than arXiv v1.
- **The five corrections.**
  - **Chen25.** Now published: Phys. Rev. A 114, 032455 (2026), DOI
    10.1103/mfyk-5j3x. The arXiv page does not list it yet.
  - **SFT26.** Accepted in PRX Quantum on 2026-10-13, DOI 10.1103/vp11-s9sl.
    No volume or article number has been assigned yet.
  - **Rosenfeld25.** Now in Science, article eaee7961, published online
    2026-10-15, DOI 10.1126/science.aee7961. Crossref lists 296 personal
    authors, against 295 on arXiv v1. The first three are unchanged.
  - **Knill04a.** The cited DOI 10.1038/nature03350 is the Nature version,
    which has a different title ("Quantum computing with realistically noisy
    devices") and year: Nature 434(7029):39-44, March 2005.
  - **Merhav08.** The DOI is the journal article: IEEE Trans. Inf. Theory
    54(10):4439-4447, October 2008. The entry gave 2007 and an arXiv venue.
- **Flags in the notes file.** No statement in the notes reverses a paper's
  abstract. Six items need attention:
  1. **Knill (Sec. 3.6).** The notes attribute the γ > 0.06 threshold to the
     "heuristic analysis"; in the paper, the 0.06 comes from direct
     simulation (Flag 1).
  2. **Combes-Ferrie.** The equation numbers in the notes are ar5iv-only
     (Flag 2).
  3. **Franc et al. (Sec. 1.3).** The notes use the arXiv-v1 model name
     "bounded-coverage", not the JMLR name "bounded-abstention" (Flag 3).
  4. **Publication status.** The arXiv-only statements for Chen, Staples and
     Rosenfeld are out of date (Flag 4).
  5. **Rosenfeld (Sec. 4.2).** The "portion fetched" hedge can be dropped
     (Flag 5).
  6. **Meister et al.** One caveat is internal to the paper (Flag 6).

## Verdict table

| # | key | verdict | confirmed title; authors; year | arXiv | DOI | venue | note |
|---|---|---|---|---|---|---|---|
| 1 | HWL14 | confirmed | Efficient Markov chain Monte Carlo algorithm for the surface code (arXiv title: "An efficient Markov chain Monte Carlo algorithm for the surface code"); Adrian Hutter, James R. Wootton, Daniel Loss; 2014 (arXiv v1 2013-02-11, v2 2013-08-08) | 1302.2669 | 10.1103/PhysRevA.89.022326 | Phys. Rev. A 89, 022326 (2014), published 2014-02-18 | No conflict with the abstract. Matches in the v2 PDF: threshold between 15% and 16%, the 18.9% bound, "421 to 61", the O(2^((L-1)/2)) low-p factor, O(L) distinguishability, "breadcrumbs". The words "complementary" and "confidence" do not occur. Yoked [HWL14], Meister [3] and Smith [66] are this paper. |
| 2 | BPRS24 | confirmed | Fault-Tolerant Postselection for Low-Overhead Magic State Preparation (arXiv v1 title: "Fault-tolerant Post-Selection for Low Overhead Magic State Preparation"); Héctor Bombín, Mihir Pant, Sam Roberts, Karthik I. Seetharam; 2024 (arXiv v1 2022-12-01) | 2212.00813 | 10.1103/PRXQuantum.5.010302 | PRX Quantum 5, 010302 (2024), published 2024-01-04 | No conflict. The abstract has "factor of 15" at "relative overhead factor of <2 (approximately 23 times less than ... syndrome-counting rules)". The v1 PDF has 1.78, "~23 times lower than the best annular syndrome rule" and "~17" at the bulk threshold. The notes' "i.i.d. bit-flip and erasure" is consistent: Pauli errors are fusion-outcome bit flips (App. B). |
| 3 | GNBJ25 | confirmed | Yoked surface codes; Craig Gidney, Michael Newman, Peter Brooks, Cody Jones; 2025 (arXiv v1 2023-12-07 only; the arXiv page lists no journal ref) | 2312.04522 | 10.1038/s41467-025-59714-1 | Nature Communications 16, 4498 (2025), published 2025-05-14 | No conflict. These match in the v1 HTML: the complementary-gap definition, "rescaling the gap by 0.9x", "tends towards overconfidence", SI1000 at 1e-3, integer-dB bins, "slightly optimistic", the min-of-m-samples extrapolation. "Postselect" appears only in one reference title, consistent with "does not postselect on gaps". |
| 4 | SBB24 | confirmed | Mitigating errors in logical qubits; Samuel C. Smith, Benjamin J. Brown, Stephen D. Bartlett; 2024 (arXiv v1 2024-05-06) | 2405.03766 | 10.1038/s42005-024-01883-4 | Communications Physics 7, 386 (2024), published 2024-11-28 | No conflict. The abstract has 50%, 32(1)%, the quadratic gain and 75%/60%. The v1 HTML has thresholds 50/42/38/33/15% for c = 0, 2/5, 1/2, 2/3, 1; abort thresholds 1.3(1)/2.1(1)/4.5(1)%; the rule 1 - Delta/d > c; p_m = 2p/3, t = d. |
| 5 | MPP24 | confirmed | Efficient soft-output decoders for the surface code; Nadine Meister, Christopher A. Pattison, John Preskill; 2024 (v1 2024-05-13, v2 2024-06-01) | 2405.07433 | none | arXiv preprint; Crossref and a web search found no journal version | No conflict. The v2 HTML has n = 12, p = 0.05, ~0.2% rejected (phi <= 15), ~1e-5 to 4e-10; for d = 9, ~5e-4 discarded, 3.0(5)e-5 to at most 2e-6 (95%). Also: "nearly a factor of 4", the [[1054,140]] QCLP with a ~3x pseudothreshold, and the theorem/definition labels. See Flag 6. |
| 6 | GSJ24 | confirmed | Magic state cultivation: growing T states as cheap as CNOT gates; Craig Gidney, Noah Shutty, Cody Jones; 2024 (v1 2024-09-26 only) | 2409.17595 | none | arXiv preprint; Crossref and a web search found no journal version | No conflict: the abstract gives 2e-9 at 1e-3 and 4e-11 at 5e-4. Every quote in notes Sec. 4.1 is verbatim in the v1 HTML, and the four Fig. 2 "This Paper" rows match exactly. The "1e-4 retry per round, 1e-9 per round" statement is made at d2 = 15. |
| 7 | PBDD21 | confirmed | Improved quantum error correction using soft information; Christopher A. Pattison, Michael E. Beverland, Marcus P. da Silva, Nicolas Delfosse; 2021 (v1 2021-07-28 only) | 2107.13589 | none | arXiv preprint; Crossref found no journal version | No conflict: the abstract has the 25% higher threshold and the d = 19 thousand-fold effect. PDF table: 3.665(2)% vs 2.637(1)% (best hard 2.93(2)%), 0.727(1)% vs 0.702(1)%, 0.5824(6)% vs 0.4991(5)%. "Postselect" and "confidence" do not occur, as the notes say. |
| 8 | EWB25 | confirmed | Thresholds for Postselected Quantum Error Correction from Statistical Mechanics (arXiv title: "Thresholds for post-selected quantum error correction from statistical mechanics"); Lucas H. English, Dominic J. Williamson, Stephen D. Bartlett; 2025 (v1 2024-10-10, v2 2024-10-28, v3 2025-09-18) | 2410.07598 | 10.1103/nh49-52y2 | Phys. Rev. Lett. 135, 120603 (2025), published 2025-09-18 | No conflict. The v3 HTML has Theorem 1 verbatim and the S7 sentence on Markovian channels; S7 is titled "Effectiveness of heuristic post-selection rule". Also matching: 1/(2 + sqrt 2) ~ 0.2929; the depolarizing gamma = 1 threshold 0.5 (App. B, "consistent with ... Ref. [10]"); "abort threshold is fixed at the code's non-post-selected logical threshold". |
| 9 | Chen25 | corrected | Scalable accuracy gains from postselection in quantum error correcting codes; Hongkun Chen, Daohong Xu, Grace M. Sommers, David A. Huse, Jeff D. Thompson, Sarang Gopalakrishnan; 2026 (journal; arXiv v1 2025-10-06, v3 2026-05-19) | 2510.05222 | 10.1103/mfyk-5j3x (add) | Phys. Rev. A 114, 032455 (2026); received 2025-10-18, published 2026-09-25 | Correction: add the journal version, which the arXiv page does not yet list. Title and authors unchanged. No conflict with the abstract. The v3 HTML has b = 3.1(1), s_bar = 0.80(1), I(0) = 0.26(1) at p = 0.06; b ~ 2.87 (circuit-level, p = 0.005); b ~ 3.44 (perfect measurement, p = 0.06); 2.91/2.92 (bond-diluted); and the overhead sentence. |
| 10 | SFT26 | corrected | Scalable Postselection of Quantum Resources; J. Wilson Staples, Winston Fu, Jeff D. Thompson; 2026 (arXiv v1 2026-03-09) | 2603.08697 | 10.1103/vp11-s9sl (add) | PRX Quantum, accepted 2026-10-13 (accepted-manuscript page; no volume or article number yet) | Correction: add the venue and DOI. No conflict with the abstract (partial gap; 4x lower overhead per logical gate). The v1 HTML has "sensitive to this guess", string splitting, "r ~ 0.01", dephasing concurrent with CZ, and Stim. |
| 11 | LEB26 | confirmed | Efficient Post-Selection for General Quantum LDPC Codes (Crossref capitalizes it as "Efficient post-selection for general quantum LDPC Codes"); Seok-Hyung Lee, Lucas H. English, Stephen D. Bartlett; 2026 (arXiv v1 2025-10-07 to v4 2026-01-25) | 2510.05795 | 10.1038/s41534-026-01242-x | npj Quantum Information 12, 96 (2026), published 2026-04-24 | No conflict. The abstract gives about three orders of magnitude at 1% (19%) abort for p = 0.1% (0.3%), and "matches or even surpasses". The v4 HTML has "remains superior" for surface codes. Its "p_log/T almost constant, log(1 - p_abort) nearly linear in T" supports the notes' per-round abort statement. |
| 12 | Kishi26 | confirmed | Even More Efficient Soft-Output Decoding with Extra-Cluster Growth and Early Stopping; Kaito Kishi, Riki Toshio, Jun Fujisaki, Hirotaka Oshima, Shintaro Sato, Keisuke Fujii; 2026 (v1 2026-02-03) | 2602.03336 | none | arXiv preprint; Crossref found no journal version | No conflict. The notes' three statements are close paraphrases of the abstract. |
| 13 | PR24 | confirmed | Distance-four quantum codes with combined postselection and error correction; Prithviraj Prabhu, Ben W. Reichardt; 2024 (arXiv v1 2021-12-07) | 2112.03785 | 10.1103/PhysRevA.110.012419 | Phys. Rev. A 110, 012419 (2024), published 2024-07-08 | No conflict. The v1 PDF has "we stop the computation instead of attempting to correct, ensuring logical flips only occur at rate O(p^{t+2})", the union-find rule "three or more edges, we immediately reject", and "distance-two codes can detect single errors". |
| 14 | Knill04a | corrected | Quantum computing with realistically noisy devices (arXiv title: "Quantum Computing with Very Noisy Devices"); E. Knill; 2005 (journal; arXiv v1 2004-10-25, v2 2004-11-02) | quant-ph/0410199 | 10.1038/nature03350 | Nature 434(7029), 39-44 (March 2005) | Correction: the DOI's record has a different title and year. If the 47-page arXiv preprint is the source, as in the notes, cite it as arXiv:quant-ph/0410199 (2004) and give the DOI as the published version. See Flag 1. |
| 15 | Knill04b | confirmed | Fault-Tolerant Postselected Quantum Computation: Schemes; E. Knill; 2004 (v1 2004-02-23) | quant-ph/0402171 | none | arXiv preprint; Crossref found no journal version | No conflict. The notes' paraphrase (nonzero acceptance; success probability "decreases dramatically with accuracy") matches the abstract. |
| 16 | Rosenfeld25 | corrected | Magic state cultivation on a superconducting quantum processor; Emma Rosenfeld, Craig Gidney, Gabrielle Roberts, et al. (295 authors on arXiv v1; Crossref lists 296 for the Science version); 2026 (journal; arXiv v1 2025-12-15) | 2512.13908 | 10.1126/science.aee7961 (add) | Science, article eaee7961, published online 2026-10-15 (Crossref has no volume or issue yet) | Correction: venue, DOI, year. science.org returned 403, so this rests on Crossref and the doi.org redirect. The v1 HTML has "Filtering for any detection events in the entire circuit ... 8%", 76(3)% injection retention, "between 1e-4 and 2e-4", Tesseract, and flux transients or calibration drift. See Flag 5. |
| 17 | Forney68 | confirmed | Exponential error bounds for erasure, list, and decision feedback schemes; G. D. Forney, Jr. (Crossref: "G. Forney"); 1968 | none | 10.1109/TIT.1968.1054129 | IEEE Trans. Inf. Theory 14(2), 206-220 (March 1968) | The DOI resolves to IEEE Xplore document 1054129. Content was not read; the notes say they rely on HSS10 and Merhav for it, which is accurate. |
| 18 | Chow70 | confirmed | On optimum recognition error and reject tradeoff; C. K. Chow; 1970 | none | 10.1109/TIT.1970.1054406 | IEEE Trans. Inf. Theory 16(1), 41-46 (Jan. 1970) | The IBM Research page abstract matches the notes' paraphrase: an optimum rejection rule, the error-reject relation, "The error rate can be directly evaluated from the reject function", and normal and uniform examples. |
| 19 | Chow57 | confirmed | An optimum character recognition system using decision functions; C. K. Chow; 1957 | none | 10.1109/TEC.1957.5222035 | IRE Trans. Electronic Computers EC-6(4), 247-254 (Dec. 1957) | Metadata only (Crossref; the DOI resolves to IEEE Xplore document 5222035). The notes use it for metadata only. |
| 20 | HSS10 | confirmed | Performance Bounds for Erasure, List, and Decision Feedback Schemes With Linear Block Codes; Eran Hof, Igal Sason, Shlomo Shamai; 2010 | none | 10.1109/TIT.2010.2050797 | IEEE Trans. Inf. Theory 56(8), 3754-3778 (Aug. 2010) | The DOI resolves to IEEE Xplore document 5508626. The Technion preprint (dated May 6, 2010) matches the notes on Definition 2, Remark 2, Remark 3, Proposition 1 and Definition 3. Definition 2 uses joint probabilities Pr(y, x_m); Remark 2 says T = 0 agrees with MAP; Remark 3 says T > 0 gives disjoint regions. No conflict. |
| 21 | Merhav08 | corrected | Error Exponents of Erasure/List Decoding Revisited via Moments of Distance Enumerators (the journal title capitalizes "Via"); Neri Merhav; 2008 (journal; arXiv v1 2007-11-15) | 0711.2501 | 10.1109/TIT.2008.929004 | IEEE Trans. Inf. Theory 54(10), 4439-4447 (Oct. 2008) | Correction: year and venue. The entry gave 2007 and an arXiv venue, but the DOI is the October 2008 journal article, which also fits the key. No conflict with the abstract. The v1 PDF has the Neyman-Pearson sentence, E1(R,T) as a max over 0 <= s <= rho <= 1, and E2 = E1 + T. |
| 22 | FPV23 | confirmed | Optimal Strategies for Reject Option Classifiers (sentence case on arXiv); Vojtech Franc, Daniel Prusa, Vaclav Voracek (arXiv gives initials only); 2023 (arXiv v1 2021-01-29) | 2101.12523 | none (JMLR issues no DOI) | Journal of Machine Learning Research 24(11):1-49 (2023); submitted 1/21, revised 12/21, published 1/23 | No conflict with either abstract. The notes' numbering of Problems 1, 2 and 4, Theorems 1, 3, 4 and 6, and Definition 1 matches the JMLR PDF. See Flag 3. |
| 23 | CF15 | confirmed | Cost of postselection in decision theory; Joshua Combes, Christopher Ferrie; 2015 (v1 2015-04-14, v2 2015-08-30, which changed the title) | 1504.03740 | 10.1103/PhysRevA.92.022117 | Phys. Rev. A 92, 022117 (2015), published 2015-08-19 | No conflict with the abstract. The content in the notes is correct, but the equation numbers are ar5iv-specific; see Flag 2. |

## Flags in the notes file (`lit-L4-softoutput.md`)

None of these reverses an abstract. Numbers 1 to 3 are about attribution or
locators; 4 and 5 are out of date; 6 is a caveat inside the source paper.

1. **Knill 2004/2005, Sec. 3.6.**
   - **The γ > 0.06 method.** The notes say "Heuristic analysis of
     postselected encoded CNOTs suggests a threshold above gamma = 0.06". In
     the arXiv v2 PDF, the "computer-assisted heuristic analysis of the
     conditional errors in prepared encoded Bell pairs" gives "thresholds of
     above γ = 0.03". The 0.06 comes from the second method, "direct
     simulation of the error behavior of postselected encoded CNOTs", which
     "suggest a threshold of above γ = 0.06 by extrapolation". Relabel the
     method.
   - **A dropped hedge.** The abstract says "we give evidence that accurate
     quantum computing is possible with error probabilities above 3% per
     gate". The notes drop "give evidence".
2. **Combes and Ferrie 2015 equation numbers (Secs. 0.1, 1.1, 1.4).** The
   notes cite Eqs. (14), (35a), (35b), (36), (37) and (47). Those are the
   numbers in the ar5iv rendering (confirmed on ar5iv). The arXiv v2 PDF
   numbers the same results differently:

   | result | notes / ar5iv | arXiv v2 PDF |
   |---|---|---|
   | posterior threshold rule, 1 - λ | (14) | (10) |
   | Chow risk identity | (35a), (35b) | (25a), (25b) |
   | Stieltjes integral for Pr(E) | (36) | (26) |
   | slope relation | (37) | (27) |
   | general-cost threshold 1 - λR/λE | (47) | (31) |
   | loss ordering | (4) | (4) |

   The content is correct. The Fig. 7 caption ("the derivative of Pr(E) with
   respect to Pr(R) is λ") is verbatim. Cite the PDF or journal numbers, or
   say that the numbers are ar5iv's.
3. **Franc, Prusa and Voracek, Sec. 1.3.** "Problem 4 is the bounded-coverage
   model" uses the arXiv v1 name. The cited JMLR version calls it the
   "bounded-abstention model". JMLR Sec. 4 also says the bounded-improvement
   and bounded-abstention models "were coined in Pietraszek (2005)". The
   theorem, problem and definition numbers in the notes agree with the JMLR
   PDF.
4. **Out-of-date publication status.** These parts of the notes say
   "arXiv preprint" or "no journal reference listed":
   - Sec. 3.3 (Chen et al.): now Phys. Rev. A 114, 032455 (2026).
   - Sec. 3.4 (Staples, Fu and Thompson): now accepted in PRX Quantum,
     DOI 10.1103/vp11-s9sl.
   - Sec. 4.2 (Rosenfeld et al.): now Science eaee7961 (2026).
   - Appendix B rows for the three papers above.

   Also, Sec. 1.5 cites Merhav only as arXiv plus a "related DOI". The
   journal version is IEEE TIT 54(10):4439-4447 (2008).
5. **Rosenfeld et al., Sec. 4.2.** The notes hedge: "No complementary-gap
   cutoff is described in the portion fetched." The full arXiv v1 HTML,
   supplement included (it runs to Fig. S19), does not contain the word "gap"
   at all. The hedge can be dropped for v1. I did not check the Science
   version's text.
6. **Meister, Pattison and Preskill (a caveat in the paper itself).**
   Theorem 10 (the exact LLR for union-find on the repetition code) is stated
   for "an odd number n of data bits". The paper's own n = 12 example (Fig. 7)
   is nonetheless "computed using Theorem 10". The notes quote both without
   the odd-n condition. This matters only if Theorem 10 is applied to an
   even-distance case, such as our d = 2 code, where weight ties occur.

## Fetch log

- **arXiv API (one call, all 19 IDs).** `https://export.arxiv.org/api/query?id_list=1302.2669,2212.00813,2312.04522,2405.03766,2405.07433,2409.17595,2107.13589,2410.07598,2510.05222,2603.08697,2510.05795,2602.03336,2112.03785,quant-ph/0410199,quant-ph/0402171,2512.13908,0711.2501,2101.12523,1504.03740&max_results=50`
- **arXiv abstract pages.** `https://arxiv.org/abs/<id>` for the same 19 IDs.
- **Crossref works.** `https://api.crossref.org/works/<doi>` for:
  - the 14 cited DOIs: 10.1103/PhysRevA.89.022326, 10.1103/PRXQuantum.5.010302, 10.1038/s41467-025-59714-1, 10.1038/s42005-024-01883-4, 10.1103/nh49-52y2, 10.1038/s41534-026-01242-x, 10.1103/PhysRevA.110.012419, 10.1038/nature03350, 10.1109/TIT.1968.1054129, 10.1109/TIT.1970.1054406, 10.1109/TEC.1957.5222035, 10.1109/TIT.2010.2050797, 10.1109/TIT.2008.929004, 10.1103/PhysRevA.92.022117;
  - the 3 DOIs found in this check: 10.1103/mfyk-5j3x, 10.1103/vp11-s9sl, 10.1126/science.aee7961.
- **doi.org redirects.** All returned HTTP 302 to the expected host:
  - IEEE Xplore documents 1054129, 1054406, 5222035, 5508626 and 4626073;
  - nature.com/articles/nature03350 and nature.com/articles/s41534-026-01242-x;
  - link.aps.org for nh49-52y2, mfyk-5j3x and vp11-s9sl;
  - science.org for science.aee7961.
- **Crossref searches.** `https://api.crossref.org/works?query.bibliographic=...` and `...?query.title=...` for MPP24, GSJ24, PBDD21, Chen25, SFT26, Kishi26, Knill04b and Rosenfeld25.
- **Web searches.** One each for MPP24 and GSJ24 (journal versions). Neither found one.
- **Publisher and other pages.**
  - `https://www.jmlr.org/papers/v24/21-0048.html` and `https://www.jmlr.org/papers/volume24/21-0048/21-0048.pdf`
  - `https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff`
  - `https://webee.technion.ac.il/people/sason/bounds_list_erasure_feedback.pdf`
  - `https://journals.aps.org/pra/abstract/10.1103/mfyk-5j3x`
  - `https://journals.aps.org/prxquantum/accepted/10.1103/vp11-s9sl`
  - `https://www.science.org/doi/10.1126/science.aee7961` (HTTP 403)
  - `https://ar5iv.labs.arxiv.org/html/1504.03740` (WebFetch; curl from the login node timed out)
- **Full texts for spot checks.**
  - arXiv HTML, `https://arxiv.org/html/<id>`: 2410.07598v3, 2409.17595v1, 2510.05222v3, 2405.07433v2, 2405.03766v1, 2512.13908v1, 2312.04522v1, 2603.08697v1, 2510.05795v4.
  - arXiv PDF, `https://arxiv.org/pdf/<id>`: 1302.2669v2, 2212.00813v1, 2107.13589v1, 2112.03785v1, 0711.2501v1, 1504.03740v2, quant-ph/0410199v2.
