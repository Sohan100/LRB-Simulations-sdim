# L5 citation check: Pauli frames, real-time vs offline decoding, windows, latency

Independent check of the 22 citations used in `lit-L5-realtime.md`, done 2026-10-28.

**Label.** Every statement in this file is **from literature**: I read it myself, in this
task, on the pages listed in the fetch log at the end. "[full text]" means the arXiv PDF;
everything else comes from the arXiv abstract page, the Crossref record or the publisher's
landing page. I make no analytic or numerical claims here, and no conjectures.

**Method.**

- arXiv abstract pages (`https://arxiv.org/abs/<id>`) for the 21 entries that have an arXiv
  ID. From each page I read the title, the authors, the version history, the journal-ref,
  the DOI link and the abstract.
- Crossref records (`https://api.crossref.org/works/<doi>`) for all 21 DOIs.
- Publisher landing pages: nature.com (8 articles), journals.aps.org (6), quantum-journal.org
  (1) and the TU Delft research portal (1). Three publishers returned HTTP 403: ACM DL, AIP
  and ScienceDirect.
- arXiv PDFs for 19 entries. I used these only to spot-check numbers and quotations that the
  notes attribute to the full text.

## 1. Verdicts

Verdicts: 21 confirmed, 1 corrected (author list of the Nature version of GoogleQAI2025),
0 not found.

| Key | Verdict | Verified title | Authors | Year | arXiv | DOI; venue |
|---|---|---|---|---|---|---|
| Knill2005 | confirmed | Nature: "Quantum computing with realistically noisy devices". arXiv: "Quantum Computing with Very Noisy Devices" | E. Knill | 2005 (arXiv v1 25 Oct 2004, v2 2 Nov 2004) | quant-ph/0410199 | 10.1038/nature03350; Nature 434(7029), 39-44 (Mar 2005) |
| DiVincenzoAliferis2007 | confirmed | "Effective fault-tolerant quantum computation with slow measurements" | David P. DiVincenzo, Panos Aliferis | 2007 (published 9 Jan 2007; arXiv v1 6 Jul 2006, v2 3 Aug 2006) | quant-ph/0607047 | 10.1103/PhysRevLett.98.020501; Phys. Rev. Lett. 98(2), 020501. The arXiv journal-ref reads "98 (2007) 220501"; APS and Crossref give 020501, so the misprint is confirmed |
| Riesebos2017 | confirmed | "Pauli Frames for Quantum Computer Architectures" | L. Riesebos, X. Fu, S. Varsamopoulos, C. G. Almudever, K. Bertels (TU Delft portal: Leon Riesebos, Xiang Fu, Savvas Varsamopoulos, Carmina García Almudever, Koen Bertels) | 2017 (Crossref 18 Jun 2017) | none | 10.1145/3061639.3062300; Proc. 54th Annual Design Automation Conference (DAC 2017), Austin TX, Article 76, pp. 1-6, ACM, ISBN 978-1-4503-4927-7. The ACM page returned 403 |
| DKLP2002 | confirmed | "Topological quantum memory" | Eric Dennis, Alexei Kitaev, Andrew Landahl, John Preskill | 2002 (arXiv v1 24 Oct 2001) | quant-ph/0110143 | 10.1063/1.1499754; J. Math. Phys. 43(9), 4452-4505 (Sep 2002). The AIP page returned 403 |
| Terhal2015 | confirmed | "Quantum error correction for quantum memories" | Barbara M. Terhal | 2015 (published 7 Apr 2015; arXiv v1 14 Feb 2013, v7 10 Apr 2015) | 1302.3428 | 10.1103/RevModPhys.87.307; Rev. Mod. Phys. 87(2), 307-346 |
| Fowler2012SC | confirmed | "Surface codes: Towards practical large-scale quantum computation" | Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland | 2012 (published 18 Sep 2012; arXiv v1 4 Aug 2012, v2 27 Oct 2012) | 1208.0928 | 10.1103/PhysRevA.86.032324; Phys. Rev. A 86(3), 032324 |
| Fowler2012TO | confirmed | "Time-optimal quantum computation" | Austin G. Fowler | 2012 (v1 17 Oct 2012, v2 19 Dec 2012, v3 2 Feb 2013) | 1210.4626 | No DOI and no journal-ref on arXiv. A Crossref bibliographic search found no journal version. Preprint only |
| Chamberland2018 | confirmed | "Fault-tolerant quantum computing in the Pauli or Clifford frame with slow error diagnostics" | Christopher Chamberland, Pavithran Iyer, David Poulin | 2018 (published 4 Jan 2018; arXiv v1 21 Apr 2017, v2 25 Dec 2017) | 1704.06662 | 10.22331/q-2018-01-04-43; Quantum 2, 43 |
| Skoric2023 | confirmed | "Parallel window decoding enables scalable fault tolerant quantum computation" | Luka Skoric, Dan E. Browne, Kenton M. Barnes, Neil I. Gillespie, Earl T. Campbell | 2023 (published 3 Nov 2023; arXiv v1 18 Sep 2022, v2 4 Feb 2023) | 2209.08552 | 10.1038/s41467-023-42482-1; Nat. Commun. 14, 7040 |
| Tan2023 | confirmed | PRX Quantum: "Scalable Surface-Code Decoders with Parallelization in Time". arXiv: "Scalable surface code decoders with parallelization in time" | Xinyu Tan, Fang Zhang, Rui Chao, Yaoyun Shi, Jianxin Chen | 2023 (published 15 Dec 2023; arXiv v1 19 Sep 2022, v2 29 Sep 2022) | 2209.09219 | 10.1103/PRXQuantum.4.040344; PRX Quantum 4(4), 040344. The arXiv page shows no DOI |
| Battistel2023 | confirmed | "Real-time decoding for fault-tolerant quantum computing: progress, challenges and outlook" | F. Battistel, C. Chamberland, K. Johar, R. W. J. Overwater, F. Sebastiano, L. Skoric, Y. Ueno, M. Usman | 2023 (online 9 Aug 2023; arXiv v1 28 Feb 2023, v2 22 May 2023) | 2303.00054 | 10.1088/2399-1984/aceba6; Nano Futures 7(3), 032003 |
| GoogleQAI2025 | **corrected** (authors) | "Quantum error correction below the surface code threshold" | Nature: Google Quantum AI and Collaborators, 255 named authors starting R. Acharya, **D. A. Abanin**, L. Aghababaie-Beni, I. Aleiner, T. I. Andersen, M. Ansmann, et al. The entry's list is the arXiv v1 order (249 names, no Abanin) | 2025 (online 9 Dec 2024, issue Feb 2025; arXiv v1 24 Aug 2024) | 2408.13687 | 10.1038/s41586-024-08449-y; Nature 638(8052), 920-926 |
| Barber2025 | confirmed | Nat. Electron.: "A real-time, scalable, fast and resource-efficient decoder for a quantum computer". arXiv: "A real-time, scalable, fast and highly resource efficient decoder for a quantum computer" | Ben Barber, Kenton M. Barnes, Tomasz Bialas, Okan Buğdaycı, Earl T. Campbell, Neil I. Gillespie, Kauser Johar, Ram Rajan, Adam W. Richardson, Luka Skoric, Canberk Topal, Mark L. Turner, Abbas B. Ziad | 2025 (online 7 Jan 2025; arXiv v1 11 Sep 2023, v2 24 Sep 2024) | 2309.05558 | 10.1038/s41928-024-01319-5; Nat. Electron. 8(1), 84-91 |
| Caune2026 | confirmed | "Demonstrating real-time and low-latency quantum error correction with superconducting qubits" | Laura Caune, Luka Skoric, Nick S. Blunt, Archibald Ruban, et al. 25 authors, same list and order on arXiv v1 and in Nat. Commun. Affiliations: Riverlane (UK and USA), Rigetti Computing, Rigetti UK, University of Sheffield | 2026 (published 1 Jun 2026; arXiv v1 7 Oct 2024 is the only version) | 2410.05202 | 10.1038/s41467-026-73331-6; Nat. Commun. 17, 7383. The arXiv page shows no DOI |
| Liyanage2023 | confirmed | "Scalable Quantum Error Correction for Surface Codes Using FPGA" ("using" on arXiv) | Namitha Liyanage, Yue Wu, Alexander Deters, Lin Zhong | 2023 (arXiv v1 20 Jan 2023, v2 15 May 2023) | 2301.08419 | 10.1109/QCE57702.2023.00106; 2023 IEEE QCE, Bellevue WA, 17-22 Sep 2023, pp. 916-927. The FCCM 2023 one-page version mentioned in the notes is also confirmed: 10.1109/FCCM57271.2023.00045, p. 217, same four authors |
| Liyanage2024 | confirmed | "FPGA-Based Distributed Union-Find Decoder for Surface Codes" | Namitha Liyanage, Yue Wu, Siona Tagare, Lin Zhong | 2024 (arXiv v1 dated 20 Mar 2024, v2 2 Oct 2024) | 2406.08491 | 10.1109/TQE.2024.3467271; IEEE Trans. Quantum Eng. 5, 1-18 |
| RyanAnderson2021 | confirmed | "Realization of Real-Time Fault-Tolerant Quantum Error Correction" | C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, A. Hankin, J. P. Gaebler, et al. (17 authors; arXiv has "J. Gerber" where PRX has "J. A. Gerber") | 2021 (published 23 Dec 2021; arXiv v1 15 Jul 2021) | 2107.07505 | 10.1103/PhysRevX.11.041058; Phys. Rev. X 11(4), 041058. The arXiv page shows no DOI |
| Geher2025 | confirmed | "To reset, or not to reset—that is the question" (arXiv: "-- that is") | György P. Gehér, Marcin Jastrzebski, Earl T. Campbell, Ophelia Crawford (the entry drops the diacritics) | 2025 (published 7 Mar 2025; arXiv v1 1 Aug 2024, v2 5 Mar 2025) | 2408.00758 | 10.1038/s41534-025-00998-y; npj Quantum Inf. 11, 39 |
| Sivak2023 | confirmed | "Real-time quantum error correction beyond break-even" | V. V. Sivak, A. Eickbusch, B. Royer, S. Singh, I. Tsioutsios, S. Ganjam, A. Miano, B. L. Brock, A. Z. Ding, L. Frunzio, S. M. Girvin, R. J. Schoelkopf, M. H. Devoret | 2023 (online 22 Mar 2023; arXiv v1 16 Nov 2022) | 2211.09116 | 10.1038/s41586-023-05782-6; Nature 616(7955), 50-55 |
| Brock2025 | confirmed | "Quantum error correction of qudits beyond break-even" | Benjamin L. Brock, Shraddha Singh, Alec Eickbusch, Volodymyr V. Sivak, Andy Z. Ding, Luigi Frunzio, Steven M. Girvin, Michel H. Devoret | 2025 (online 14 May 2025; arXiv v1 23 Sep 2024, v2 8 Oct 2024) | 2409.15065 | 10.1038/s41586-025-08899-y; Nature 641(8063), 612-618 |
| Gottesman1998 | confirmed | "Fault-Tolerant Quantum Computation with Higher-Dimensional Systems" | Daniel Gottesman | 1998 for arXiv (2 Feb 1998); the journal issue is Sep 1999 | quant-ph/9802007 | 10.1016/S0960-0779(98)00218-5; Chaos Solitons Fractals 10(10), 1749-1758 (1999), per the arXiv journal-ref and Crossref. Crossref's title field for this DOI is the placeholder "Full length article" and lists no authors, and ScienceDirect returned 403. I therefore did not see the journal-side title directly. A second version appeared in LNCS 1509, pp. 302-313 (1999) |
| HowardVala2012 | confirmed | PRA: "Qudit versions of the qubit π/8 gate". arXiv: 'Qudit versions of the qubit "pi-over-eight" gate' (the entry gives the arXiv title) | Mark Howard, Jiri Vala | 2012 (published 15 Aug 2012; arXiv v1 7 Jun 2012, v2 16 Aug 2012) | 1206.1598 | 10.1103/PhysRevA.86.022316; Phys. Rev. A 86(2), 022316 |

## 2. Statements in `lit-L5-realtime.md` that need changes

No statement in the notes contradicts an abstract page outright. The items below are a
misquote of an abstract (F3), a version issue in an author list (F5) and characterizations
that the fetched full texts or other fetched papers do not support (F1, F2, F6). F4, F7 and
F8 are minor.

- **F1. Tan2023 and DKLP2002 (notes Section 1.2, row C6).** The notes say: "The fetched
  studies find no difference within error bars when buffer >= d [lit: Skoric2023, Tan2023,
  DKLP2002]". This holds only for Skoric2023, which uses n_buf = n_com = d and reports "no
  noticeable increase in logical error rate".
  - Tan2023 benchmarks s = (d+1)/2 and w = 3s with w = s + 2b. Its buffer is therefore
    b = (d+1)/2 < d [full text].
  - Tan2023 reports thresholds that are "almost identical", not equal within error bars:
    0.68% vs 0.70% for MWPM and 0.55% vs 0.554% for UF [abstract; Fig. S8].
  - DKLP2002 gives an analytic condition, T ≫ L, and no numerical comparison [full text].
- **F2. Caune2026 (notes Section 4.14, Bearing).** The notes say: "Among the papers fetched
  here, this is the only measurement of a full feedback loop (decode then act) on
  superconducting hardware". This is overstated. Sivak2023, also fetched, runs a real-time
  measure-decide-act loop on superconducting hardware. Its QEC cycle "2 × 4.924 µs ...
  includes ... real-time processing and decision making by the controller", with an FPGA
  virtual rotation of 448 ns [full text]. Section 4.18 of the notes already quotes this.
  The claim is correct if restricted to a stabilizer-code decoder in the feedback loop.
- **F3. HowardVala2012 (notes Section 4.19).** Misquote. The notes quote "the simplest
  example from the third level of the Clifford hierarchy". The abstract reads "the simplest
  example of an operation from the third level of the Clifford hierarchy".
- **F4. Tan2023 (notes Section 4.10).** The notes say the PDF thanks Xiaotong Ni "for the
  insight". The PDF says "for the insightful discussion"; the word is hyphenated across a
  line break.
- **F5. GoogleQAI2025 (notes Section 4.12, Citation).** The author list "R. Acharya,
  L. Aghababaie-Beni, I. Aleiner, T. I. Andersen, M. Ansmann et al." is the arXiv v1 order.
  The Nature article, which is the version the notes cite, lists D. A. Abanin second. See
  the corrected entry in Section 1.
- **F6. RyanAnderson2021 (notes Section 2 table; Section 0 item 3(c) and row C5).**
  - The label "Honeywell (Quantinuum) H1" does not appear in the paper. The paper describes
    a ten-qubit QCCD trapped-ion computer from Honeywell Quantum Solutions [abstract; full
    text].
  - Section 0 cites this paper, with others, for "real-time physical correction is strictly
    worse than the frame". The paper motivates software correction ("can potentially induce
    more errors"), but its own measurement found 93(2)% vs 92(1)%, "not significantly
    different" [full text]. Section 4.16 of the notes states this correctly. Section 0
    should cite the paper for the motivation only.
- **F7. Barber2025 (notes Sections 2 and 4.13).** arXiv v2 is internally inconsistent for
  the d = 23 ASIC: one passage gives 210 ns per round, and two others give 240 ns
  [full text]. The notes use 240 ns. No change is needed, but the number should be cited
  with this caveat.
- **F8. Caune2026 (notes Section 2 table).** The Nat. Commun. abstract weakens the arXiv
  claim on the backlog. The arXiv abstract says "showing that we avoid the backlog problem".
  The journal abstract says "a sub-microsecond mean decoding time per round, providing
  strong evidence that the backlog problem will be avoided when the decoder is operated as
  a streaming decoder". The headline numbers are the same in both. The notes label their
  numbers "arXiv v1 numbers", which is correct.

## 3. Spot checks that passed (full text unless marked)

- **Knill2005.** All five quotes are present:
  - "differs from that expected according to the Pauli frame";
  - "modulating the Pauli frame";
  - "no error and no speed constraint on classical computations";
  - "known only to the simulation";
  - "none of the Pauli corrections actually have to be implemented explicitly".
- **DiVincenzoAliferis2007.**
  - The abstract quote matches; arXiv writes "1,000" and APS writes "1000".
  - In the full text: "known for some time"; Pauli "reference frame"; S "only if the
    measurement outcome is −1"; C = 34 with t_meas = 1000a; "cannot be used in
    post-selected computation".
- **Riesebos2017.** All three abstract quotes match the TU Delft abstract word for word.
- **DKLP2002.**
  - The abstract assumptions quote matches, and so does the four-dimensional robust
    recovery.
  - In the full text: lower bound p_c ≥ .0114 for p = q; "very nearly as well as if an
    indefinite amount of information were stored"; T ≫ L; "overreact"; "instantaneously and
    flawlessly".
- **Terhal2015.** All present: "never necessary to physically correct Pauli errors"; "simply
  alters the interpretation of the final measurement outcome"; "'retarded' by only a
  constant amount of time"; "constant slow-down"; "processed at leisure"; "weakly anharmonic
  multi-level systems"; Bullock and Brennen.
- **Fowler2012SC.** Present: "a hardware-based solution will always have a higher error rate
  than one implemented in the control software"; "corrective phase- and bit-flips to the
  qubit measurements".
- **Fowler2012TO.**
  - The abstract matches: a Clifford circuit, then a layer of independent T gates, then the
    feedforward S corrections, all within one physical measurement time, assuming fast
    classical processing and communication.
  - In the full text: 100 ns measurement, 200 ns cycle, 76.5 µs vs 100 ns; S "does not
    gracefully commute".
- **Chamberland2018.** Present: "10 to 1000 times slower than gates times"; "well above
  1µs"; O(n²) frame update; "without ever exiting the Pauli group"; "no need to actively
  error-correct".
- **Skoric2023.**
  - "Offline decoding is sufficient for computations consisting solely of Clifford gates
    (e.g. CNOT and Hadamard gates)".
  - "not compatible with the real-time feedback required for non-Clifford gates".
  - n_buf = n_com = d gives "no noticeable increase".
  - The caveat on homology-based inner decoders.
  - N_par = 16 gives "greater than an order-of-magnitude increase in decoding" frequency.
  - Round times of 1.1 µs (Krinner) and 921 ns (Google).
  - LILLIPUT processes a round every 300 ns for d ≤ 5.
- **Tan2023.** Thresholds 0.68%/0.70% (MWPM) and 0.55%/0.554% (UF); s = (d+1)/2 and w = 3s;
  detectors are the XOR of consecutive syndrome rounds; ansatz 1 − 2p_L(i) = [1 − 2p_L(1)]^i;
  "further theoretical justification or numerical validation is needed"; equal contribution
  of Tan and Zhang; Alibaba Quantum Laboratory.
- **Battistel2023.**
  - "needed only for the next-in-line logical non-Clifford gate".
  - "≲ O(1) µs ... ≲ O(1) ms ... beyond O(100) ms".
  - "relaxed to a few µs".
  - Throughput is "even more important than latency".
  - "a few tens of Mbps".
  - "at most a couple hundreds of nanoseconds"; "10s of nanoseconds".
  - "in excess of many milliseconds, leaving ample time for real-time decoding".
  - 1 ms for ions vs the 200 ms cycle.
  - Real-time demonstrations so far only on ion traps with look-up tables; transmon
    demonstrations decoded offline.
- **GoogleQAI2025.**
  - The Nature abstract names "Willow" for both memories.
  - 72-qubit processor for the real-time d = 5 run; 105-qubit processor for d = 7.
  - Latency 63 ± 17 µs.
  - Real time: ε5 = 0.35% ± 0.01%, Λ = 2.0 ± 0.1. Offline NN: 0.269% ± 0.008%,
    Λ = 2.18 ± 0.09.
  - NN decoding takes 24 µs per cycle.
  - T_react = T_decode + T_control (their eq. (1)); T_input < 10 µs.
  - Blocks of M = 10 and M = 90 cycles; a 128-block buffer; bitwise XOR of block
    predictions.
  - NN pre-training on SI1000 at p = 0.4%.
  - "within an order of magnitude" of the assumed reaction time; "does not yet include
    feedback".
- **Barber2025.**
  - The Table I d = 3 to 23 values checked (0.07 ... 1.18 µs) match the notes. The d = 13
    and d = 15 entries were not checked.
  - 4.5% of logic LUTs and 10 KB; threshold 0.78%; < 13 KB at d = 23.
  - ASIC: 10 ns and 240 ns per round at 2.75 mW and 7.85 mW; 0.06 mm²; 8 mW.
  - Helios: 900k LUTs and 240k registers.
  - Noise model with p/10 for single-qubit gates, measurement and reset.
  - "cannot support logic branching"; streamed decoding left for future work.
- **Caune2026.**
  - 0.44 µs to 0.79 µs per round.
  - 9.6 µs = 6.5 µs + 3.1 µs.
  - 1.4 µs, plus 1 to 1.5 µs (250 to 370 FPGA cycles).
  - About 1.7 µs per round.
  - T1 used as a clock.
  - No ancilla reset in the main text; "conditional resets cannot improve the performance";
    DDROP.
  - Google comparison: 63 µs plus 10 µs.
  - Gidney-Ekerå target of 10 µs.
  - Definition of logical branching.
- **Liyanage2023.** The abstract gives 11.5 ns at d up to 21 and 0.1% phenomenological noise.
  The full text gives 90% within 250 ns and 99.99% within 370 ns at d = 13, with a
  right-skewed distribution.
- **Liyanage2024.**
  - The abstract gives 11.5 ns, 23.7 ns (d = 17, circuit-level) and 544 ns (d = 51).
  - Full text, d = 13: 97% within 250 ns and 99.99% within 510 ns; mean 194 ns.
  - Longest latency 920 ns, against a theoretical 857 µs.
  - Sliding window: 194 → 230 ns, 371K vs 166K LUTs, 56× faster than the measurement rate;
    "prevalent method"; window 2d, commit d.
- **RyanAnderson2021.** Present: "software updates to the Pauli frame or as physically
  applied gates" (abstract); 93(2)% vs 92(1)%; 1.75(4)×10⁻² per cycle; < 60 ms SPAM;
  < 200 ms per cycle; < 70 ms transport; < 250 µs distribution latency; two lookup tables;
  a two-bit Pauli frame; "essentially perfect software correction"; "Pauli frame updates
  cannot always be used".
- **Geher2025.** In the abstract: no benefit from reset in memory experiments; up to a
  factor of two for logical operations; about 100 ns and about 10^-2.5 ≈ 0.003. In the full
  text: "a correlated pair of errors" and "electronics latency".
- **Sivak2023.** t_c = 2 × 4.924 µs; virtual rotation in 448 ns.
- **Brock2025.** Gains 1.82 ± 0.03 and 1.87 ± 0.03 (abstract). Full text: for odd d the
  displacement "brings us outside the code space"; for even d it is "tracked in software";
  7 µs per sBs round; "autonomous, only requiring reset of the ancilla".
- **Gottesman1998.** The "prime d" quote matches the abstract.

## 4. Fetch log

- arXiv abstract pages, `https://arxiv.org/abs/<id>`: quant-ph/0410199, quant-ph/0607047,
  quant-ph/0110143, 1302.3428, 1208.0928, 1210.4626, 1704.06662, 2209.08552, 2209.09219,
  2303.00054, 2408.13687, 2309.05558, 2410.05202, 2301.08419, 2406.08491, 2107.07505,
  2408.00758, 2211.09116, 2409.15065, quant-ph/9802007, 1206.1598.
- arXiv PDFs, `https://arxiv.org/pdf/<id>`, for full-text spot checks: the same IDs except
  1206.1598 and quant-ph/9802007.
- Crossref, `https://api.crossref.org/works/<doi>`, for all 21 DOIs in Section 1 and for
  10.1109/FCCM57271.2023.00045. Also the bibliographic query
  `https://api.crossref.org/works?query.bibliographic=Time-optimal+quantum+computation+Fowler&rows=5`.
- nature.com, `https://www.nature.com/articles/<id>`: nature03350, s41467-023-42482-1,
  s41586-024-08449-y, s41928-024-01319-5, s41467-026-73331-6, s41534-025-00998-y,
  s41586-023-05782-6, s41586-025-08899-y.
- journals.aps.org abstract pages: PhysRevLett.98.020501, RevModPhys.87.307,
  PhysRevA.86.032324, PRXQuantum.4.040344, PhysRevX.11.041058, PhysRevA.86.022316.
- https://quantum-journal.org/papers/q-2018-01-04-43/
- https://research.tudelft.nl/en/publications/pauli-frames-for-quantum-computer-architectures/
- These returned HTTP 403:
  - https://dl.acm.org/doi/10.1145/3061639.3062300
  - https://pubs.aip.org/aip/jmp/article/43/9/4452/231779/Topological-quantum-memory
  - https://www.sciencedirect.com/science/article/pii/S0960077998002185
- Gottesman, for the journal metadata:
  - https://api.openalex.org/works/doi:10.1016/S0960-0779(98)00218-5, which has the same
    placeholder title as Crossref;
  - https://doi.org/10.1016/S0960-0779(98)00218-5, which redirects to the Elsevier
    linking hub;
  - https://ftp.math.utah.edu/pub/tex/bib/idx/lncs1999a/1509/0/302-313.html, which gives
    the LNCS 1509, pp. 302-313 version.
