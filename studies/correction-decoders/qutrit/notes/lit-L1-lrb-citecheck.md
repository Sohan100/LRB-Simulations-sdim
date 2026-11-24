# Citation check: L1 (logical RB, encoded-gate RB, qudit RB, leakage RB)

This is an independent check of the 33-entry citation list behind `lit-L1-lrb.md`, done on
2026-10-28. I also read that notes file for statements that contradict the papers' abstract
pages. Where the notes' headline findings rest on statements that go beyond an abstract, I
checked eight of them against the full text.

**Label.** Every verdict and flag below is **from literature**: it rests on pages fetched in
this task (fetch log at the end). One explanation is marked **conjecture**. I ran no
simulations, so nothing here is verified numerically.

## Method

- **arXiv abstract pages.** I fetched `https://arxiv.org/abs/<id>` with WebFetch for all 32
  arXiv entries. From each page I read the title, author line, submission history,
  Journal-ref, DOI and related-DOI rows, comments and abstract. `curl` from the login node
  timed out, so every fetch went through WebFetch.
- **Journal DOIs.** For each journal DOI I read either the publisher page or the Crossref
  record (`https://api.crossref.org/works/<doi>`, either one DOI at a time or in two batch
  filter queries). The publisher pages are the APS pages for Magesan 2011, Ryan-Anderson
  2021 and Helsen 2022. nature.com pages redirect to a login flow and the IOP page redirects
  to a bot check, so Crossref is the source for those DOIs.
- **Entries with only an arXiv DOI.** A web search for a journal version found none for
  Combes 2017, Mayer 2024, Paetznick 2024, Ryan-Anderson 2022, Ceasura 2022, Graydon 2021 or
  Mai 2026. Berthusen 2026 (posted 2 Sep 2026) was checked only on its arXiv page, which has
  no Journal-ref.
- **Full text.** I used three kinds of full text:
  - arXiv LaTeX source, fetched from `arxiv.org/e-print/<id>`, unpacked in the session
    scratchpad (not in the repo) and grepped: Combes 2017 (v1), Harper-Flammia 2019 (archive
    dated 31 Jan 2019, i.e. v4) and Kabir 2025 (archive dated 14 May 2026, i.e. v2).
  - arXiv HTML: Mayer 2024, Bluvstein 2025, and the Paetznick 2024 author block.
  - ar5iv renderings: Morvan 2021, Marques 2022 and Kononenko 2021.

## Summary

- **27 confirmed, 6 corrected, 0 not found.** All 32 arXiv IDs exist and point to the cited
  paper. Every listed DOI resolves to the cited work. The arXiv DataCite DOIs are as shown
  on the abstract pages, and the journal DOIs match Crossref or the publisher page.
- **The six corrections:**
  - Nigg2014: Science published it as "Quantum computations on a topologically encoded
    qubit" (without "Experimental"), Science 345(6194), 302-305 (2014).
  - Bluvstein2025: published as "A fault-tolerant neutral-atom architecture for universal
    quantum computation", Nature 649, 39-46 (2026), online 10 Nov 2025. The listed DOI is
    right; the venue "arXiv preprint" is out of date.
  - Helsen2022: the PRX Quantum title is "General Framework for Randomized Benchmarking".
  - RyanAnderson2021: published as Phys. Rev. X 11, 041058 (2021), DOI
    10.1103/PhysRevX.11.041058. The arXiv page shows no Journal-ref.
  - Paetznick2024: 32 authors, not 33.
  - Lacroix2025: 215 authors, not "about 214". The Nature title spells "colour code".
- **Notes file.**
  - One statement contradicts an abstract page: a quotation of the Proctor et al. 2017
    abstract that is not verbatim. The meaning is preserved.
  - The other issues are the six bibliographic corrections above, plus an incomplete venue
    for Bluvstein2023.
  - All eight full-text spot checks support the notes.

## Verdict table

| Key | Status | Verified title | Authors | Year | arXiv | DOI | Venue (verified) | Note |
|---|---|---|---|---|---|---|---|---|
| Combes2017 | confirmed | Logical Randomized Benchmarking | Joshua Combes, Christopher Granade, Christopher Ferrie, Steven T. Flammia | 2017 | 1702.03688 (v1 13 Feb 2017, only version) | 10.48550/arXiv.1702.03688 | arXiv preprint; no Journal-ref; no journal version found | - |
| HarperFlammia2019 | confirmed | Fault-Tolerant Logical Gates in the IBM Quantum Experience | Robin Harper, Steven T. Flammia | 2019 | 1806.02359 (v1 6 Jun 2018 to v4 30 Jan 2019) | 10.1103/PhysRevLett.122.080504 | Phys. Rev. Lett. 122, 080504 (2019); Crossref issue date 26 Feb 2019 | The current arXiv title equals the PRL title. |
| Heeres2017 | confirmed | Implementing a Universal Gate Set on a Logical Qubit Encoded in an Oscillator | Reinier W. Heeres, Philip Reinhold, Nissim Ofek, Luigi Frunzio, Liang Jiang, Michel H. Devoret, Robert J. Schoelkopf | 2017 (arXiv 8 Aug 2016) | 1608.02430 | 10.1038/s41467-017-00045-1 | Nat. Commun. 8, 94 (2017), online 21 Jul 2017 | The venue can be completed with the volume and article number. |
| Reinhold2020 | confirmed | Error-corrected gates on an encoded qubit | Philip Reinhold, Serge Rosenblum, Wen-Long Ma, Luigi Frunzio, Liang Jiang, Robert J. Schoelkopf | 2020 (arXiv 29 Jul 2019) | 1907.12327 | 10.1038/s41567-020-0931-8 | Nat. Phys. 16, 822-826 (2020), online 8 Jun 2020 | - |
| Nigg2014 | **corrected** | arXiv: Experimental Quantum Computations on a Topologically Encoded Qubit. Science: **Quantum computations on a topologically encoded qubit** | Daniel Nigg, Markus Mueller (Science: M. Müller), Esteban A. Martinez, Philipp Schindler, Markus Hennrich, Thomas Monz, Miguel A. Martin-Delgado, Rainer Blatt | 2014 | 1403.5426 (v1 21 Mar 2014) | 10.1126/science.1253742 | **Science 345(6194), 302-305 (2014)** | The listed title is the arXiv title. The Science version drops "Experimental". |
| Marques2022 | confirmed | Logical-qubit operations in an error-detecting surface code | J. F. Marques, B. M. Varbanov, M. S. Moreira, H. Ali, N. Muthusubramanian, C. Zachariadis, F. Battistel, M. Beekman, N. Haider, W. Vlothuizen, A. Bruno, B. M. Terhal, L. DiCarlo (13) | 2022 (arXiv 25 Feb 2021; online 16 Dec 2021) | 2102.13071 | 10.1038/s41567-021-01423-9 | Nat. Phys. 18, 80-86 (2022) | - |
| Mayer2024 | confirmed | Benchmarking logical three-qubit quantum Fourier transform encoded in the Steane code on a trapped-ion quantum computer | Karl Mayer, Ciarán Ryan-Anderson, Natalie Brown, et al. (23) | 2024 | 2404.08616 (v1 12 Apr 2024, only version) | 10.48550/arXiv.2404.08616 | arXiv preprint; no journal version found | arXiv spells "Ciarán". |
| Paetznick2024 | **corrected** | Demonstration of logical qubits and repeated error correction with better-than-physical error rates | A. Paetznick, M. P. da Silva, C. Ryan-Anderson, et al. (**32 authors**) | 2024 | 2404.02280 (v1 2 Apr 2024; v2 4 Apr 2024; v3 17 Nov 2024) | 10.48550/arXiv.2404.02280 | arXiv preprint; no journal version found | The v3 HTML author block and the arXiv API both list 32 names. Only the abstract page's link text ("and 32 other authors") implies 33. |
| Lacroix2025 | **corrected** | Scaling and logic in the color code on a superconducting quantum processor (Nature: "colour code") | Nathan Lacroix, Alexandre Bourassa, Francisco J. H. Heras, et al. (**215 authors**, Google Quantum AI) | 2025 (arXiv 18 Dec 2024; online 26 May 2025) | 2412.14256 | 10.1038/s41586-025-09061-4 | Nature 645, 614-619 (2025) | The arXiv link text reads "Nathan Lacroix and 214 other authors", and Crossref lists 215 author entries. |
| Berthusen2026 | confirmed | Experimental validation of a compact fault-tolerant architecture for trapped ions | Noah Berthusen, Ali Lavasani, Asmae Benhemou, M. S. Allman, Joan Dreiling, Brian Estey, Cameron Foltz, Trent Jacobs, Michael Mills, Annie Jihyun Park, Adam P. Reed, David Hayes, Tzvetan S. Metodi, Andrew C. Potter (14) | 2026 | 2609.03194 (v1 2 Sep 2026) | 10.48550/arXiv.2609.03194 | arXiv preprint | - |
| Mai2026 | confirmed | A biased-erasure cavity qubit with hardware-efficient quantum error detection | Jiasheng Mai, Qiyu Liu, Xiaowei Deng, Yanyan Cai, Zhongchu Ni, Libo Zhang, Ling Hu, Pan Zheng, Song Liu, Yuan Xu, Dapeng Yu | 2026 | 2601.21616 (v1 29 Jan 2026) | 10.48550/arXiv.2601.21616 | arXiv preprint; no journal version found | - |
| Bluvstein2023 | confirmed | Logical quantum processor based on reconfigurable atom arrays | Dolev Bluvstein, Simon J. Evered, Alexandra A. Geim, et al. (21) | 2023 (online 6 Dec 2023) | 2312.03982 (v1 7 Dec 2023) | 10.1038/s41586-023-06927-3 | Nature 626, 58-65 (2024) | The arXiv Journal-ref says "Nature (2023)"; the print issue is Feb 2024. Add the volume and pages. |
| Bluvstein2025 | **corrected** | arXiv: Architectural mechanisms of a universal fault-tolerant quantum computer. Nature: **A fault-tolerant neutral-atom architecture for universal quantum computation** | Dolev Bluvstein, Alexandra A. Geim, Sophie H. Li, et al. (23) | 2025 (arXiv 25 Jun 2025; online 10 Nov 2025) | 2506.20661 | 10.1038/s41586-025-09848-5 | **Nature 649, 39-46 (2026)** | From the Crossref record of the listed DOI: same 23 authors, same first five. |
| Kabir2025 | confirmed | Sdim: A Qudit Stabilizer Simulator | Adeeb Kabir, Steven Nguyen, Sohan Ghosh, James Keppens, Tijil Kiran, Isaac H. Kim, Yipeng Huang, Bart Sorée | 2025 | 2511.12777 (v1 16 Nov 2025; v2 12 May 2026) | 10.48550/arXiv.2511.12777 | arXiv preprint | arXiv spells "Sorée". The abstract does not mention LRB; the LRB-D material is in v2, Sec. VII (checked in the source). |
| GhoshKabirAPS2026 | confirmed | Logical Randomized Benchmarking a Qutrit Surface Code | Sohan Ghosh (UC Davis, presenter), Adeeb Kabir (Rutgers University) | 2026 | none | none | APS Global Physics Summit 2026, 18 Mar 2026, 1:12-1:24 PM | The page does not show the session code; the URL path is mar-m09/7. |
| RyanAnderson2021 | **corrected** | Realization of real-time fault-tolerant quantum error correction (PRX: Realization of Real-Time Fault-Tolerant Quantum Error Correction) | C. Ryan-Anderson, J. G. Bohnet, K. Lee, et al. (17) | 2021 | 2107.07505 (v1 15 Jul 2021) | **10.1103/PhysRevX.11.041058** (arXiv DOI 10.48550/arXiv.2107.07505 is also valid) | **Phys. Rev. X 11, 041058 (2021)**, published 23 Dec 2021 | The arXiv page has no Journal-ref. The PRX page has the same title and the same first nine of 17 authors. |
| RyanAnderson2022 | confirmed | Implementing Fault-tolerant Entangling Gates on the Five-qubit Code and the Color Code | C. Ryan-Anderson, N. C. Brown, M. S. Allman, et al. (37) | 2022 | 2208.01863 (v1 3 Aug 2022) | 10.48550/arXiv.2208.01863 | arXiv preprint; no journal version found | - |
| Postler2022 | confirmed | Demonstration of fault-tolerant universal quantum gate operations | Lukas Postler, Sascha Heußen, Ivan Pogorelov, et al. (13) | 2022 | 2111.12654 (v1 24 Nov 2021; v3 2 Nov 2022) | 10.1038/s41586-022-04721-1 | Nature 605, 675-680 (2022) | arXiv spells "Heußen". |
| Magesan2011 | confirmed | arXiv: Robust randomized benchmarking of quantum processes. PRL: Scalable and Robust Randomized Benchmarking of Quantum Processes | Easwar Magesan, J. M. Gambetta, Joseph Emerson | 2011 (arXiv 19 Sep 2010) | 1009.3639 | 10.1103/PhysRevLett.106.180504 | Phys. Rev. Lett. 106, 180504 (2011), published 6 May 2011 | Both titles are as listed. |
| Proctor2017 | confirmed | What randomized benchmarking actually measures | Timothy Proctor, Kenneth Rudinger, Kevin Young, Mohan Sarovar, Robin Blume-Kohout | 2017 | 1702.01853 (v1 7 Feb 2017 to v4 29 Sep 2017) | 10.1103/PhysRevLett.119.130502 | Phys. Rev. Lett. 119, 130502 (2017) | The notes misquote the abstract; see "Contradicts an abstract page" below. |
| Wallman2018 | confirmed | Randomized benchmarking with gate-dependent noise | Joel J. Wallman | 2018 | 1703.09835 (v1 28 Mar 2017 to v4 25 Jan 2018) | 10.22331/q-2018-01-29-47 | Quantum 2, 47 (2018) | - |
| Helsen2022 | **corrected** | arXiv: A general framework for randomized benchmarking. PRX Quantum: **General Framework for Randomized Benchmarking** | Jonas Helsen, Ingo Roth, Emilio Onorati, Albert H. Werner, Jens Eisert | 2022 | 2010.07974 (v1 15 Oct 2020; v2 5 Apr 2022) | 10.1103/PRXQuantum.3.020357 | PRX Quantum 3, 020357 (2022), published 16 Jun 2022 | The journal title drops the leading "A"; the APS page and Crossref agree. |
| Ceasura2022 | confirmed | Non-Exponential Behaviour in Logical Randomized Benchmarking | Athena Ceasura, Pavithran Iyer, Joel J. Wallman, Hakop Pashayan | 2022 | 2212.05488 (v1 11 Dec 2022) | 10.48550/arXiv.2212.05488 | arXiv preprint; no journal version found | - |
| GAE2007 | confirmed | Evenly distributed unitaries: on the structure of unitary designs | D. Gross, K. Audenaert, J. Eisert | 2007 (arXiv 31 Oct 2006) | quant-ph/0611002 | 10.1063/1.2716992 | J. Math. Phys. 48, 052104 (2007) | - |
| Jafarzadeh2020 | confirmed | Randomized benchmarking for qudit Clifford gates | Mahnaz Jafarzadeh, Ya-Dong Wu, Yuval R. Sanders, Barry C. Sanders | 2020 | 1911.08162 (v1 19 Nov 2019; v2 10 Jun 2020) | 10.1088/1367-2630/ab8ab1 | New J. Phys. 22, 063014 (2020) | - |
| Graydon2021 | confirmed | Clifford groups are not always 2-designs | Matthew A. Graydon, Joshua Skanes-Norman, Joel J. Wallman | 2021 | 2108.04200 (v1 9 Aug 2021) | 10.48550/arXiv.2108.04200 | arXiv preprint; no journal version found | - |
| Morvan2021 | confirmed | Qutrit randomized benchmarking | A. Morvan, V. V. Ramasesh, M. S. Blok, J. M. Kreikebaum, K. O'Brien, L. Chen, B. K. Mitchell, R. K. Naik, D. I. Santiago, I. Siddiqi | 2021 (arXiv 20 Aug 2020) | 2008.09134 | 10.1103/PhysRevLett.126.210504 | Phys. Rev. Lett. 126, 210504 (2021) | - |
| Kononenko2021 | confirmed | Characterization of Control in a Superconducting Qutrit Using Randomized Benchmarking | M. Kononenko, M. A. Yurtalan, S. Ren, J. Shi, S. Ashhab, A. Lupascu | 2021 | 2009.00599 (v1 1 Sep 2020; v2 22 Jul 2021) | 10.1103/PhysRevResearch.3.L042007 | Phys. Rev. Research 3, L042007 (2021) | - |
| WBE2016 | confirmed | NJP: Robust characterization of leakage errors. arXiv: Characterization of Leakage Errors via Randomized Benchmarking | Joel J. Wallman, Marie Barnhill, Joseph Emerson | 2016 (arXiv 12 Dec 2014) | 1412.4126 | 10.1088/1367-2630/18/4/043021 | New J. Phys. 18, 043021 (2016), online 15 Apr 2016 | Both titles are as listed. The iopscience URL redirected me to a bot check, so the NJP title is from Crossref. |
| WoodGambetta2018 | confirmed | Quantification and Characterization of Leakage Errors | Christopher J. Wood, Jay M. Gambetta | 2018 (arXiv 10 Apr 2017) | 1704.03081 | 10.1103/PhysRevA.97.032306 | Phys. Rev. A 97, 032306 (2018) | - |
| ProctorCap2022 | confirmed | Measuring the Capabilities of Quantum Computers | Timothy Proctor, Kenneth Rudinger, Kevin Young, Erik Nielsen, Robin Blume-Kohout | 2022 (online 20 Dec 2021) | 2008.11294 | 10.1038/s41567-021-01409-7 | Nat. Phys. 18, 75-79 (2022) | - |
| ProctorMRB2022 | confirmed | Scalable randomized benchmarking of quantum computers using mirror circuits | Timothy Proctor, Stefan Seritan, Kenneth Rudinger, Erik Nielsen, Robin Blume-Kohout, Kevin Young | 2022 | 2112.09853 (v1 18 Dec 2021; v2 10 Oct 2022) | 10.1103/PhysRevLett.129.150502 | Phys. Rev. Lett. 129, 150502 (2022) | - |
| Moussa2016 | confirmed | Transversal Clifford gates on folded surface codes | Jonathan E. Moussa | 2016 | 1603.02286 (v1 7 Mar 2016 to v3 15 Oct 2016) | 10.1103/PhysRevA.94.042316 | Phys. Rev. A 94, 042316 (2016) | - |

## Statements in lit-L1-lrb.md checked against the sources

### Contradicts an abstract page

1. **Proctor2017, notes Sec. 3.2 (lines 603-604).**
   - The notes say r "does not appear to correspond to the infidelity of any physically
     valid, completely positive representation" and attribute the words to the abstract.
   - The abstract (v4) reads: "These theories allow explicit computation of the error rate
     that RB measures (r), but as far as we can tell it does not correspond to the
     infidelity of a physically allowed (completely positive) representation of the
     imperfect gates."
   - The meaning is the same, but the quoted words are not the abstract's. Replace them with
     the verbatim sentence, or drop the quotation marks.

No other statement in the notes contradicts an abstract page.

### Bibliographic details in the notes to update

- Sec. 2.4, Nigg2014: Science title "Quantum computations on a topologically encoded qubit";
  Science 345, 302-305 (2014).
- Sec. 2.7, Paetznick2024: 32 authors, not 33.
- Sec. 2.8, Lacroix2025: 215 authors, not "about 214".
- Sec. 2.11, Bluvstein2023: "Nature (2023, per the arXiv journal reference)" should become
  Nature 626, 58-65 (2024), online 6 Dec 2023.
- Sec. 2.11, Bluvstein2025: published as "A fault-tolerant neutral-atom architecture for
  universal quantum computation", Nature 649, 39-46 (2026).
- Sec. 2.13, RyanAnderson2021: Phys. Rev. X 11, 041058 (2021), DOI
  10.1103/PhysRevX.11.041058.
- Sec. 3.4, Helsen2022: the journal title is "General Framework for Randomized Benchmarking".

### Full-text spot checks (all consistent with the notes)

None of these statements is on the abstract pages. I checked them because the notes'
headline findings (Sec. 0) rest on them.

- **Combes2017** (LaTeX `logical_rb.tex`).
  - None of "postselect", "post-select", "reject", "discard", "herald" or "detect" occurs
    anywhere in the source. This supports the notes' claim that the protocol has no
    postselection.
  - The quoted sentences are all present:
    - line 223: syndrome measurement errors must be uncorrelated between rounds;
    - line 753: the decay parameter depends only on the dimension of the logical space;
    - line 797: "we need not actually perform the recovery operation, as it is always a
      Pauli operation that we can commute forward";
    - line 945: leakage benchmarking;
    - lines 979-982: Heeres p_L ≈ 0.982, "we cannot directly conclude Pr(No) ≈ 0.982";
    - line 1657: "no correlations between ancilla in successive rounds".
- **HarperFlammia2019** (LaTeX `IBMCArXiv.tex`, v4). Every quoted passage is present:
  - "There is at present no complete theory for fitting an RB decay curve with
    post-selection" (line 278);
  - "Odd parity heralds an error and the run should be discarded" (130);
  - Pr(S|m,c) = Pr(S|m)/Pr(c|m) (1016-1019);
  - 1 - Pr(c|m) is "relatively flat ... starting off at 0.3 (0.4) ... slowly rising to its
    asymptote of 0.5" and is "approximately modeled ... as constant" (1023-1025);
  - "leaky subspace" (1031);
  - IBM-QX5 (824), R(2) with 576 elements (191, 950), 5.8(2)% to 0.60(3)% (60, 103, 238),
    and the thanks to T. Yoder (287).

  ar5iv's rendering of this paper did not show the passage at lines 1016-1025, but the
  e-print source contains it.
- **Mayer2024** (arXiv HTML v1). Logical RB of the transversal CNOT ran on both H1-1 and
  H2-1.
  - Fit p(L) = a f^L + 1/4.
  - 0.9991(2) and 0.9980(8) per logical CNOT, against physical 0.99912(3) and 0.99817(5).
  - L in {2, 6, 10, 14}, 10 circuits per length, 100 shots each, 16 qubits.
  - The only discards are failed repeat-until-success preparations ("at most a few
    percent").

  The abstract names only H2-1, for the QFT circuits, so it does not contradict the notes.
- **Morvan2021** (ar5iv). The value 2.38 × 10^-3 appears in Table 1, whose caption defines
  the entries as the average error per Clifford. The notes' "error per Clifford" is
  therefore right; the abstract calls the same number a "single-qutrit gate infidelity".
  Also confirmed: 3.325 native gates per Clifford, the fit <Z>(m) = A p^m + b, and the
  statement that the imaginary part stays zero under depolarization.
- **Marques2022** (ar5iv). γ_pip ∼ 45% per cycle; the schemes are compared for up to
  n = 15 cycles.
- **Bluvstein2025** (arXiv HTML v1). All three quoted statements are present:
  - "using postselection on decoding confidence (see Methods), observe that approximately
    three CNOTs per round is optimal at low error rates (Fig. 3d)";
  - 1 - F_L ∝ (p_det + N Δp_det)^((d+1)/2) / N;
  - "logical gate performance depends both on the decoding success probability ... and on
    how much that entropy increases per logical gate".
- **Kononenko2021** (ar5iv).
  - The leakage estimate comes from simulation: the Wood-Gambetta model was fitted to
    simulated populations, giving negligible leakage (L1 ≈ 6.5 × 10^-7 per Clifford). The
    abstract does not mention leakage. The notes are correct; adding "in simulation" would
    make them precise.
  - The fit P_n = (P_in - P_fn) p_n^l + P_fn and the fidelity 98.89 ± 0.05% are confirmed.
  - The device is described as a "capacitively-shunted flux-biased superconducting
    circuit", which the notes' "flux-qubit-based qutrit" describes fairly.
- **Kabir2025** (LaTeX v2, `6.1_background.tex` and `6.2_LRB-D.tex`). All quotes are
  present:
  - "[[5, 1, 2]]_3 detection surface code \cite{Moussa_2016}" (6.1, line 65);
  - in 6.2, line 8:
    - "full LRB is a demanding target for early qudit hardware";
    - "the syndrome can reveal that an error has occurred, but it generally does not
      provide enough information to identify a unique correction";
    - "a practical qudit decoder must distinguish between different nontrivial Pauli
      powers";
    - "we only perform a single terminal syndrome extraction prior to measuring Q̄";
    - "we postselect measurements with the all zeroes syndrome";
  - the exponential-decay sentence cites Wallman2018 (6.1, line 62);
  - f(D) = B α^D (6.1, line 53).

### Minor observations

- **GhoshKabirAPS2026.** The notes say the page "shows no abstract text". The page has no
  abstract section. However, its HTML carries a description that begins "In recent years,
  engineers have worked towards creating qutrit computers ...", and the page has a field
  "Publication: Planned for submission to arxiv and PRX very soon".
- **Paetznick2024 author count.** (Conjecture) The abstract page's "and 32 other authors"
  probably comes from splitting the author string "J. P. Campora III"; the API returns that
  name as "J. P. Campora".
- **Kabir2025 source typos.** These are outside the citation check; I note them for the
  authors.
  - `6.1_background.tex` line 60 writes the fidelity as
    `f_{D_,k} = |sum_{j=0}^{d} P_{D,k}(j) ω^j|`. The outcomes are j in {0, ..., d-1}, so
    the upper limit should be d - 1, and the subscript has a stray underscore.
  - Line 62 has "shappe" for "shape".
- **Diacritics dropped in the citation list.** Ciarán Ryan-Anderson (Mayer2024), Sascha
  Heußen (Postler2022), Bart Sorée (Kabir2025). Science spells Markus Müller (Nigg2014);
  arXiv spells "Mueller".

## Fetch log

| Key | URLs fetched in this task |
|---|---|
| Combes2017 | https://arxiv.org/abs/1702.03688 ; https://ar5iv.labs.arxiv.org/html/1702.03688 ; https://arxiv.org/e-print/1702.03688 |
| HarperFlammia2019 | https://arxiv.org/abs/1806.02359 ; https://ar5iv.labs.arxiv.org/html/1806.02359 ; https://arxiv.org/e-print/1806.02359 ; Crossref batch A |
| Heeres2017 | https://arxiv.org/abs/1608.02430 ; https://api.crossref.org/works/10.1038/s41467-017-00045-1 |
| Reinhold2020 | https://arxiv.org/abs/1907.12327 ; https://api.crossref.org/works/10.1038/s41567-020-0931-8 |
| Nigg2014 | https://arxiv.org/abs/1403.5426 ; https://api.crossref.org/works/10.1126/science.1253742 |
| Marques2022 | https://arxiv.org/abs/2102.13071 ; https://ar5iv.labs.arxiv.org/html/2102.13071 ; Crossref batch B |
| Mayer2024 | https://arxiv.org/abs/2404.08616 ; https://arxiv.org/html/2404.08616v1 ; web search for a journal version |
| Paetznick2024 | https://arxiv.org/abs/2404.02280 ; https://arxiv.org/abs/2404.02280v3 ; https://export.arxiv.org/api/query?id_list=2404.02280 ; https://arxiv.org/html/2404.02280v3 ; web search |
| Lacroix2025 | https://arxiv.org/abs/2412.14256 ; https://api.crossref.org/works/10.1038/s41586-025-09061-4 |
| Berthusen2026 | https://arxiv.org/abs/2609.03194 |
| Mai2026 | https://arxiv.org/abs/2601.21616 ; web search |
| Bluvstein2023 | https://arxiv.org/abs/2312.03982 ; https://api.crossref.org/works/10.1038/s41586-023-06927-3 |
| Bluvstein2025 | https://arxiv.org/abs/2506.20661 ; https://arxiv.org/html/2506.20661v1 ; https://api.crossref.org/works/10.1038/s41586-025-09848-5 |
| Kabir2025 | https://arxiv.org/abs/2511.12777 ; https://arxiv.org/html/2511.12777v2 ; https://arxiv.org/e-print/2511.12777 |
| GhoshKabirAPS2026 | https://meetings-archive.aps.org/smt/2026/mar-m09/7/ |
| RyanAnderson2021 | https://arxiv.org/abs/2107.07505 ; https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.041058 |
| RyanAnderson2022 | https://arxiv.org/abs/2208.01863 ; web search |
| Postler2022 | https://arxiv.org/abs/2111.12654 ; Crossref batch B |
| Magesan2011 | https://arxiv.org/abs/1009.3639 ; https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504 |
| Proctor2017 | https://arxiv.org/abs/1702.01853 ; Crossref batch A |
| Wallman2018 | https://arxiv.org/abs/1703.09835 ; Crossref batch A |
| Helsen2022 | https://arxiv.org/abs/2010.07974 ; https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020357 ; Crossref batch A |
| Ceasura2022 | https://arxiv.org/abs/2212.05488 ; web search |
| GAE2007 | https://arxiv.org/abs/quant-ph/0611002 ; Crossref batch A |
| Jafarzadeh2020 | https://arxiv.org/abs/1911.08162 ; Crossref batch A |
| Graydon2021 | https://arxiv.org/abs/2108.04200 ; web search |
| Morvan2021 | https://arxiv.org/abs/2008.09134 ; https://ar5iv.labs.arxiv.org/html/2008.09134 ; Crossref batch A |
| Kononenko2021 | https://arxiv.org/abs/2009.00599 ; https://ar5iv.labs.arxiv.org/html/2009.00599 ; Crossref batch B |
| WBE2016 | https://arxiv.org/abs/1412.4126 ; https://api.crossref.org/works/10.1088/1367-2630/18/4/043021 (https://iopscience.iop.org/article/10.1088/1367-2630/18/4/043021 redirected to a bot check) |
| WoodGambetta2018 | https://arxiv.org/abs/1704.03081 ; Crossref batch B |
| ProctorCap2022 | https://arxiv.org/abs/2008.11294 ; Crossref batch B |
| ProctorMRB2022 | https://arxiv.org/abs/2112.09853 ; Crossref batch B |
| Moussa2016 | https://arxiv.org/abs/1603.02286 ; Crossref batch B |

- Crossref batch A:
  `https://api.crossref.org/works?filter=doi:10.1103/PhysRevLett.122.080504,doi:10.1103/PhysRevLett.119.130502,doi:10.22331/q-2018-01-29-47,doi:10.1103/PRXQuantum.3.020357,doi:10.1063/1.2716992,doi:10.1088/1367-2630/ab8ab1,doi:10.1103/PhysRevLett.126.210504&select=DOI,title,container-title,volume,page,article-number,issued&rows=20`
- Crossref batch B:
  `https://api.crossref.org/works?filter=doi:10.1103/PhysRevResearch.3.L042007,doi:10.1103/PhysRevA.97.032306,doi:10.1038/s41567-021-01409-7,doi:10.1103/PhysRevLett.129.150502,doi:10.1103/PhysRevA.94.042316,doi:10.1038/s41567-021-01423-9,doi:10.1038/s41586-022-04721-1&select=DOI,title,container-title,volume,page,article-number,issued&rows=20`
