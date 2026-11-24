# Citation check: L2 (non-Markovian RB, gate-dependent noise, logical-noise structure under QEC)

Independent check of the 31-entry citation list behind `lit-L2-nonmarkov.md`,
done on 2026-10-28. It also skims that notes file for statements that
contradict the papers' abstract pages.

**Label.** Every verdict and flag below is **from literature [LIT]**. Each one
rests on pages fetched in this task (see the fetch log at the end). Where I say
a paraphrase in the notes "overstates" an abstract, that is my reading of the
quoted abstract text, which is given next to it.

## Method

- **arXiv abstract pages.** All 31 were fetched with `curl` from
  `https://arxiv.org/abs/<id>`; every one returned HTTP 200. From the raw HTML
  I read the `citation_title` and `citation_author` meta tags, the visible
  author line, the Journal-ref and DOI rows, the submission history and the
  abstract.
- **DOIs.** For each of the 23 DOIs I fetched the Crossref record
  (`https://api.crossref.org/works/<doi>`), with title, authors, container,
  volume, article number and dates. I also checked that `https://doi.org/<doi>`
  resolves (HTTP 302) to the expected publisher page; all 23 do.
- **Publisher pages.** For the four entries where arXiv and the journal
  disagree, I fetched the APS pages: Ball 2016, Helsen 2022,
  Figueroa-Romero 2021 and Pollock 2018 PRA.
- **arXiv-only entries.** I ran a Crossref bibliographic search for all eight
  and a web search for four of them (Ceasura, Gravier, Kobayashi,
  Kwiatkowski). I found no journal version of any of them.
- **TeX sources.** I fetched the arXiv TeX for 2307.08737 (Levine), 1805.08802
  (Beale) and 2107.05403 (Figueroa-Romero 2021). I used them only to check
  statements in the notes that go beyond the abstract and differ from it or
  look surprising.

## Summary

- **28 confirmed, 3 corrected, 0 not found.**
  - All 31 arXiv IDs exist and point to the cited paper.
  - All 23 DOIs resolve and match the cited paper.
  - All years and venue strings are correct.
- **The three corrections are titles.** For Ball 2016, Pollock 2018 (PRA) and
  Helsen 2022, the cited title is the arXiv title. The journal version, which
  the DOI and venue point to, has a slightly different title:
  - Ball 2016: the journal drops "The".
  - Pollock 2018 (PRA): the journal spells "characterization".
  - Helsen 2022: the journal drops "A".
- **Flags in the notes file.** No statement in the notes reverses a paper's
  abstract. Five paraphrases overstate their source (Ziyad, Kam, Epstein,
  Kobayashi, Morales-Hashemi). One attribution should be relabelled (Beale,
  Sec. 8.5 item 7). One provenance remark is wrong: the missing Harris
  authorship is in arXiv's own metadata, not an artifact of the fetch tool.
  Details are in the table and in the list after it.

## Verdict table

| # | key | verdict | confirmed title; authors; year | arXiv | DOI | venue | note |
|---|---|---|---|---|---|---|---|
| 1 | wallman2018 | confirmed | Randomized benchmarking with gate-dependent noise; Joel J. Wallman; 2018 (arXiv v1 2017-03-28) | 1703.09835 | 10.22331/q-2018-01-29-47 | Quantum 2, 47 (2018) | No conflict with the abstract. |
| 2 | proctor2017 | confirmed | What randomized benchmarking actually measures (PRL uses title case); Timothy Proctor, Kenneth Rudinger, Kevin Young, Mohan Sarovar, Robin Blume-Kohout; 2017 | 1702.01853 | 10.1103/PhysRevLett.119.130502 | Phys. Rev. Lett. 119, 130502 (2017) | No conflict. |
| 3 | merkel2021 | confirmed | Randomized Benchmarking as Convolution: Fourier Analysis of Gate Dependent Errors; Seth T. Merkel, Emily J. Pritchett, Bryan H. Fong; 2021 (arXiv v1 2018) | 1804.05951 | 10.22331/q-2021-11-16-581 | Quantum 5, 581 (2021) | No conflict. |
| 4 | figueroaromero2021 | confirmed | Randomized Benchmarking for Non-Markovian Noise; Pedro Figueroa-Romero, Kavan Modi, Robert J. Harris, Thomas M. Stace, Min-Hsiu Hsieh; 2021 | 2107.05403 | 10.1103/PRXQuantum.2.040351 | PRX Quantum 2, 040351 (2021) | The five-author list is right for the journal version (APS page, Crossref, arXiv TeX v2). The arXiv abstract page itself (meta tags and author line) lists only four authors, without Harris. Notes Sec. 3.7 and App. B blame the omission on the fetch tool; it is in arXiv's metadata. |
| 5 | figueroaromero2022 | confirmed | Towards a general framework of Randomized Benchmarking incorporating non-Markovian Noise; Pedro Figueroa-Romero, Kavan Modi, Min-Hsiu Hsieh; 2022 | 2202.11338 | 10.22331/q-2022-12-01-868 | Quantum 6, 868 (2022) | No conflict. |
| 6 | fongmerkel2017 | confirmed | Randomized Benchmarking, Correlated Noise, and Ising Models; Bryan H. Fong, Seth T. Merkel; 2017 | 1703.09747 | none | arXiv preprint (no journal ref; none found in Crossref) | No conflict. |
| 7 | ball2016 | **corrected** | Journal title: **Effect of noise correlations on randomized benchmarking** (the arXiv title, as cited, starts with "The"); Harrison Ball, Thomas M. Stace, Steven T. Flammia, Michael J. Biercuk; 2016 | 1504.05307 | 10.1103/PhysRevA.93.022303 | Phys. Rev. A 93, 022303 (2016) | Authors, year, IDs and venue confirmed. Notes App. B lists Ball et al. among title differences but does not give the journal title. |
| 8 | mavadia2018 | confirmed | Experimental quantum verification in the presence of temporally correlated noise; S. Mavadia, C. L. Edmunds, C. Hempel, H. Ball, F. Roy, T. M. Stace, M. J. Biercuk; 2018 (arXiv 2017) | 1706.03787 | 10.1038/s41534-017-0052-0 | npj Quantum Information 4, 7 (2018) | No conflict. |
| 9 | pollock2018pra | **corrected** | Journal title: **Non-Markovian quantum processes: Complete framework and efficient characterization** (z spelling; the arXiv title, as cited, has "characterisation"); Felix A. Pollock, César Rodríguez-Rosario, Thomas Frauenheim, Mauro Paternostro, Kavan Modi; 2018 | 1512.00589 | 10.1103/PhysRevA.97.012127 | Phys. Rev. A 97, 012127 (2018) | Notes App. B already records the arXiv spelling. |
| 10 | pollock2018prl | confirmed | Operational Markov condition for quantum processes (PRL uses title case); same five authors as row 9; 2018 | 1801.09811 | 10.1103/PhysRevLett.120.040405 | Phys. Rev. Lett. 120, 040405 (2018) | No conflict. |
| 11 | beale2018 | confirmed | Quantum Error Correction Decoheres Noise (PRL title, as cited; arXiv title "Coherence in quantum error-correcting codes"); Stefanie J. Beale, Joel J. Wallman, Mauricio Gutiérrez, Kenneth R. Brown, Raymond Laflamme; 2018 | 1805.08802 | 10.1103/PhysRevLett.121.190501 | Phys. Rev. Lett. 121, 190501 (2018) | Sec. 5.1 is consistent; its "h_crit ≈ 1/r, independent of d" is in the TeX. Flag on Sec. 8.5 item 7, see the list below. |
| 12 | iverson2020 | confirmed | Coherence in logical quantum channels; Joseph K. Iverson, John Preskill; 2020 | 1912.04319 | 10.1088/1367-2630/ab8e5c | New J. Phys. 22, 073066 (2020) | No conflict. |
| 13 | huang2019 | confirmed | Performance of quantum error correction with coherent errors; Eric Huang, Andrew C. Doherty, Steven Flammia; 2019 | 1805.08227 | 10.1103/PhysRevA.99.022313 | Phys. Rev. A 99, 022313 (2019) | No conflict. |
| 14 | chamberland2017 | confirmed | Hard decoding algorithm for optimizing thresholds under general Markovian noise; Christopher Chamberland, Joel J. Wallman, Stefanie Beale, Raymond Laflamme; 2017 | 1612.02830 | 10.1103/PhysRevA.95.042332 | Phys. Rev. A 95, 042332 (2017) | No conflict. |
| 15 | combes2017 | confirmed | Logical Randomized Benchmarking; Joshua Combes, Christopher Granade, Christopher Ferrie, Steven T. Flammia; 2017 | 1702.03688 | none | arXiv preprint (no journal ref; none found) | No conflict. |
| 16 | ceasura2022 | confirmed | Non-Exponential Behaviour in Logical Randomized Benchmarking; Athena Ceasura, Pavithran Iyer, Joel J. Wallman, Hakop Pashayan; 2022 | 2212.05488 | none | arXiv preprint (no journal ref; none found) | No conflict. |
| 17 | ziyad2025 | confirmed | Emergent Non-Markovianity in Logical Qubit Dynamics; Jalan A. Ziyad, Robin Blume-Kohout, Kenneth Rudinger; 2025 (v1 2025-12-09, v3 2026-02-19) | 2512.08893 | none | arXiv preprint (no journal ref; Crossref has only Sandia report records with the same title) | Flag on Sec. 0 item 4, see below. |
| 18 | kwiatkowski2025 | confirmed | Constructing an approximate logical Markovian model of consecutive QEC cycles of a stabilizer code; Alex Kwiatkowski, Aaron J. Friedman, Shawn Geller, Jalan A. Ziyad, Scott Glancy, Emanuel Knill; 2025 | 2509.16887 | none | arXiv preprint (no journal ref; none found) | No conflict. |
| 19 | wallman2016leak | confirmed | Robust characterization of leakage errors (NJP title, as cited; arXiv listing title "Characterization of Leakage Errors via Randomized Benchmarking"); Joel J. Wallman, Marie Barnhill, Joseph Emerson; 2016 | 1412.4126 | 10.1088/1367-2630/18/4/043021 | New J. Phys. 18, 043021 (2016) | No conflict. |
| 20 | harper2019 | confirmed | Fault-Tolerant Logical Gates in the IBM Quantum Experience; Robin Harper, Steven T. Flammia; 2019 | 1806.02359 | 10.1103/PhysRevLett.122.080504 | Phys. Rev. Lett. 122, 080504 (2019) | No conflict. The 5.8(2)% to 0.60(3)% figures match the abstract. |
| 21 | levine2024 | confirmed | Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons; Harry Levine, Arbel Haim, Jimmy S. C. Hung, et al. (113 authors on arXiv and in Crossref); 2024 | 2307.08737 | 10.1103/PhysRevX.14.011051 | Phys. Rev. X 14, 011051 (2024) | No conflict. The abstract gives 2.19(2)e-3 erasure per X90 gate; the notes' 4.38(3)e-3 and 1.01(1)e-4 are per Clifford (two X90s each) and appear verbatim in the TeX, as do "11 erasure checks ... regardless of depth" and about 0.8% loss per check. |
| 22 | helsen2022 | **corrected** | Journal title: **General Framework for Randomized Benchmarking** (the arXiv title, as cited, is "A general framework for randomized benchmarking"); Jonas Helsen, Ingo Roth, Emilio Onorati, Albert H. Werner, Jens Eisert; 2022 | 2010.07974 | 10.1103/PRXQuantum.3.020357 | PRX Quantum 3, 020357 (2022) | Authors, year, IDs and venue confirmed. |
| 23 | wallmanflammia2014 | confirmed | Randomized benchmarking with confidence (NJP title; arXiv uses title case); Joel J. Wallman, Steven T. Flammia; 2014 | 1404.6025 | 10.1088/1367-2630/16/10/103032 | New J. Phys. 16, 103032 (2014) | No conflict. |
| 24 | epstein2014 | confirmed | Investigating the limits of randomized benchmarking protocols; Jeffrey M. Epstein, Andrew W. Cross, Easwar Magesan, Jay M. Gambetta; 2014 (arXiv 2013) | 1308.2928 | 10.1103/PhysRevA.89.062321 | Phys. Rev. A 89, 062321 (2014) | Minor flag on Sec. 3.4, see below. |
| 25 | figueroaromero2024 | confirmed | Operational Markovianization in Randomized Benchmarking (journal uses lower case "randomized benchmarking"); Pedro Figueroa-Romero, Miha Papič, Adrian Auer, Min-Hsiu Hsieh, Kavan Modi, Inés de Vega; 2024 | 2305.04704 | 10.1088/2058-9565/ad3f44 | Quantum Sci. Technol. 9, 035020 (2024) | No conflict. |
| 26 | proctor2022 | confirmed | Scalable randomized benchmarking of quantum computers using mirror circuits (PRL uses title case); Timothy Proctor, Stefan Seritan, Kenneth Rudinger, Erik Nielsen, Robin Blume-Kohout, Kevin Young; 2022 (arXiv 2021) | 2112.09853 | 10.1103/PhysRevLett.129.150502 | Phys. Rev. Lett. 129, 150502 (2022) | No conflict. |
| 27 | gravier2025 | confirmed | Simulated non-Markovian Noise Resilience of Silicon-Based Spin Qubits with Surface Code Error Correction; Oscar Gravier, Thomas Ayral, Benoît Vermersch, Tristan Meunier, Valentin Savin; 2025 | 2507.08713 | none | arXiv preprint (no journal ref; none found) | No conflict. The abstract says the results "suggest" QEC Markovianizes the noise, and the notes keep that hedge. |
| 28 | kam2025 | confirmed | Detrimental non-Markovian errors for surface code memory; John F. Kam, Spiro Gicev, Kavan Modi, Angus Southwell, Muhammad Usman; 2025 (arXiv 2024) | 2410.23779 | 10.1088/2058-9565/adebab | Quantum Sci. Technol. 10, 035060 (2025) | Flag on Sec. 8.5 item 7, see below. Sec. 6.5 is fine. |
| 29 | kobayashi2024 | confirmed | Tensor-network decoders for process tensor descriptions of non-Markovian noise; Fumiyoshi Kobayashi, Hidetaka Manabe, Gregory A. L. White, Terry Farrelly, Kavan Modi, Thomas M. Stace; 2024 | 2412.13739 | none | arXiv preprint (no journal ref; none found) | Minor flag on Sec. 6.6, see below. |
| 30 | moraleshashemi2026 | confirmed | What Must a Quantum-Memory Decoder Know About Temporally Correlated Noise?; Danesh Morales-Hashemi, Graeme Smith, Ningping Cao; 2026 (v1 2026-10-02) | 2610.03545 | none | arXiv preprint | Minor flag on Sec. 6.7, see below. |
| 31 | gross2007 | confirmed | Evenly distributed unitaries: on the structure of unitary designs; D. Gross, K. Audenaert, J. Eisert; 2007 (arXiv 2006) | quant-ph/0611002 | 10.1063/1.2716992 | J. Math. Phys. 48, 052104 (2007) | No conflict. |

## Flags in `lit-L2-nonmarkov.md` (statement versus abstract page)

1. **Ziyad et al. 2025, Sec. 0 item 4.** The notes say logical noise under QEC
   "is generically non-Markovian" and "becomes so whenever the data are not
   returned to the code space every round."
   - The abstract says only that non-Markovianity "can emerge from Markovian
     physical operations if (and only if) the physical qubits are not
     necessarily returned to the code subspace after every round of QEC."
   - So the condition is necessary, and it makes non-Markovianity possible.
     The abstract does not say that non-Markovianity always follows, or that
     it is generic.
   - Sec. 6.2 quotes the abstract correctly. Suggested wording for item 4:
     "can become non-Markovian, and only when ...".
2. **Kam et al. 2025, Sec. 8.5 item 7.** The notes say "Correlated ancilla
   and readout ('streaky') noise is the most damaging class for correction
   (Kam et al.)."
   - The abstract names "multi-time 'streaky' correlations affecting syndrome
     qubits and two-qubit gates" and says they "can severely degrade logical
     error rate scaling."
   - It does not mention readout, it includes two-qubit gates, and it does
     not call these correlations the most damaging class.
3. **Beale et al. 2018, Sec. 8.5 item 7.** The notes say "Coherent errors make
   d = 2 postselected logical noise nearly unitary (Beale et al. scaling)."
   - The abstract's result points the other way: syndrome measurement "causes
     coherent errors to converge toward probabilistic Pauli errors."
   - The near-unitary d = 2 statement is the notes' own extrapolation. Sec. 5.1
     labels it correctly as [VA], "not a theorem of the paper." Item 7 should
     carry the same label rather than cite the paper.
4. **Epstein et al. 2014, Sec. 3.4 (minor).** The notes say RB "estimated the
   average error rate to within a factor of two." The abstract says it does so
   "in almost all cases."
5. **Kobayashi et al. 2024, Sec. 6.6 (minor).** The notes say "Correlated noise
   can significantly degrade performance." The abstract says the results
   "highlight the possible detrimental effects of correlated noise."
6. **Morales-Hashemi et al. 2026, Sec. 6.7 (minor).** The notes say the twirled
   Pauli description "suffices for decoding only if the twirl record is
   discarded."
   - The abstract says discarding the record "makes the SPP sufficient for
     optimal recovery." It adds that keeping the record "can improve the
     attainable fidelity, but reaching that optimum may require additional
     noise information." "Only if" is stronger than "may require".
   - The result is also proved for the specific model pair studied: weak
     coherent X rotations on a class of CSS codes.
7. **Figueroa-Romero et al. 2021, Sec. 3.7 and App. B (provenance).** The
   notes say the arXiv listing "as returned by the fetch tool" omitted Robert
   J. Harris. In fact the raw arXiv abstract page omits him: both the
   `citation_author` meta tags and the author line list four names. The TeX
   (v2) and the PRX Quantum page list five. The citation itself is correct.

I checked three statements in the notes against the TeX and found them
correct:
- Beale: h_crit ≈ 1/r, independent of the code distance.
- Levine: the per-Clifford rates, the 11 checks spread "regardless of depth",
  and about 0.8% loss per check.
- Figueroa-Romero 2021: the TeX lists Harris.

All other summaries in the notes that I compared with the abstracts agree with
them. Many technical details in the notes come from the TeX sources and cannot
be checked against an abstract page. I did not check them, apart from the three
above.

## Related papers seen during the check (not in the list; pages fetched)

These are optional pointers for the L2 topic. I fetched their arXiv abstract
pages and Crossref records in this task.

- **Rohan S. Kumar, Ben Foxman, Yongshan Ding (2026), "Modeling Quantum Error
  Detection with Transition Matrices".** arXiv:2609.12073; extended version of
  a paper accepted at IEEE QCE 2026. The abstract describes:
  - an exact transition matrix for post-selected quantum error detection, with
    rejection as an absorbing state;
  - "a leading-order obstruction to a stationary logical-only accepted-map
    description" caused by "accepted leakage injection from QED-check
    imperfections";
  - a proof that "less frequent checks improve efficiency to leading order";
  - a "first-cycle transient".

  This bears directly on the LRB-D check policies.
- **Antoine Brillant, Peter Groszkowski, Alireza Seif, Jens Koch, Aashish A.
  Clerk (2025), "Randomized Benchmarking with Non-Markovian Noise and
  Realistic Finite-Time Gates".** Phys. Rev. Lett. 135, 070601 (2025);
  arXiv:2501.06172; DOI 10.1103/tmfd-qc5q. Under non-Markovian classical
  noise, the RB decay can be exponential or a power law, depending on how the
  gates are implemented.
- **Srilekha Gandhari, Michael J. Gullans (2026), "Quantum non-Markovian noise
  in randomized benchmarking of spin-boson models".** Phys. Rev. Research 8,
  023075 (2026); arXiv:2502.14702 (published 2025 on arXiv); DOI
  10.1103/z8zh-n1hl. A non-Markovian bosonic bath gives an "almost
  polynomial" RB decay.

## Fetch log

- arXiv abstract pages (curl, HTTP 200 for all):
  `https://arxiv.org/abs/<id>` for the 31 IDs in the table, plus 2609.12073,
  2501.06172 and 2502.14702.
- Crossref: `https://api.crossref.org/works/<doi>` for the 23 DOIs in the
  table, plus 10.1103/tmfd-qc5q and 10.1103/z8zh-n1hl.
- Crossref bibliographic search:
  `https://api.crossref.org/works?query.bibliographic=...` for the 8 arXiv-only
  entries.
- doi.org resolution: `https://doi.org/<doi>` for the 23 DOIs. All returned
  302 to the publisher: quantum-journal.org, link.aps.org, nature.com,
  iopscience.iop.org or pubs.aip.org.
- APS pages:
  - https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.022303
  - https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020357
  - https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040351
  - https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.012127
- arXiv TeX sources: `https://arxiv.org/e-print/2307.08737`, `.../1805.08802`
  and `.../2107.05403`.
- Web searches for journal versions of Kobayashi 2024, Gravier 2025,
  Ceasura 2022 and Kwiatkowski 2025 found none.
