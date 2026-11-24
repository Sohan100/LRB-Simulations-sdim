# Lit 05: Erasure conversion, leakage, and flag qubits

Scope: how heralded erasures change what a distance-2 code can correct; erasure
qubits on superconducting, neutral-atom and trapped-ion hardware; leakage as a
non-Markovian error source and how to mitigate it (LRUs, leakage detection with
three-outcome readout); flag qubits for hook errors. Prepared 2026-10-28 for the
qubit `[[5,1,2]]` LRB study (see `../CONTEXT.md`).

Claim labels used below: **from literature** (with citation keys),
**verified analytically**, **verified numerically** (my own light stim check,
Appendix A), **conjecture** (my inference).

Name collision: in the leakage literature "LRB" means *leakage* randomized
benchmarking (WoodGambetta2018 and others). In this repo LRB means *logical*
randomized benchmarking. These notes always spell out "leakage RB".

## 0. Summary

- A distance-d code corrects d-1 erasures but only floor((d-1)/2) unlocated
  errors. Heralding therefore turns the `[[5,1,2]]` code from detect-only into
  a code that corrects one located error. Two recent `[[4,2,2]]` experiments
  do exactly this with a hybrid "correct heralded erasures, postselect the
  rest" policy (Zhang2026, Reichardt2024). This is the most direct template for
  turning LRB-D into a correcting protocol.
- For our code: all 5 single erasures are correctable, 6 of the 10 erasure
  pairs are correctable, and no erasure triple is correctable
  (**verified numerically**).
- Hardware erasure fractions are well below the 98% projection: in Yb they are
  about 0.31 for CZ gates and 0.56 for one-qubit gates (Zhang2026 error
  model). Superconducting
  dual-rail qubits (Levine2024, Chou2024) and metastable ions (Quinn2026) herald
  most of their dominant error.
- Leakage is the standard example of a non-Markovian error in QEC. Leaked
  qubits produce detection events correlated across rounds, and removing
  leakage every cycle (DQLR, LRUs) makes it behave like an uncorrelated error
  (McEwen2021, Miao2023, Battistel2021, Xin2026). Leakage RB theory gives
  multi-exponential decays with a nonzero steady state (WoodGambetta2018,
  Wallman2016). Wallman2016 explicitly allows the "subspace" to be a code
  space. That is the analogy I propose for LRB-D rejection (conjecture).
- Flags are not needed for single-fault detection with our weight-3 checks and
  the given CNOT orders: no single gate fault yields an undetectable logical
  (**verified numerically**). This differs from `[[n,n-2,2]]` codes, whose single
  weight-n check does need a flag (ChaoReichardt2018).

## 1. Bibliography

Core sources (20). I fetched each one myself (arXiv abstract page, plus full
text where noted). Journal metadata was confirmed on the arXiv page or through
Crossref (`api.crossref.org/works/<DOI>`).

| Key | Title | Authors | Year | arXiv | DOI (venue) | URL | Fetched | Bears on |
|---|---|---|---|---|---|---|---|---|
| Wu2022 | Erasure conversion for fault-tolerant quantum computing in alkaline earth Rydberg atom arrays | Y. Wu, S. Kolkowitz, S. Puri, J. D. Thompson | 2022 | 2201.03540 | 10.1038/s41467-022-32094-6 (Nat. Commun. 13, 4657) | https://arxiv.org/abs/2201.03540 | yes (abstract + full text) | Q1, Q3 |
| Kubica2023 | Erasure qubits: Overcoming the T1 limit in superconducting circuits | A. Kubica, A. Haim, Y. Vaknin, F. Brandão, A. Retzker | 2023 (arXiv 2022) | 2208.05461 | 10.1103/PhysRevX.13.041022 (PRX 13, 041022) | https://arxiv.org/abs/2208.05461 | yes (abstract + full text) | Q1, Q3 |
| Gu2025 | Optimizing quantum error correction protocols with erasure qubits | S. Gu, Y. Vaknin, A. Retzker, A. Kubica | 2025 (arXiv 2024) | 2408.00829 | 10.1103/985g-58gd (PRX Quantum 6, 040354) | https://arxiv.org/abs/2408.00829 | yes (abstract + intro; Crossref) | Q1, Q3 |
| Levine2024 | Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons | H. Levine, A. Haim, J. S. C. Hung, N. Alidoust, et al. (113 authors), O. Painter | 2024 (arXiv 2023) | 2307.08737 | 10.1103/PhysRevX.14.011051 (PRX 14, 011051) | https://arxiv.org/abs/2307.08737 | yes (abstract) | Q3 |
| Chou2024 | A superconducting dual-rail cavity qubit with erasure-detected logical measurements (arXiv title starts "Demonstrating a ...") | K. S. Chou, T. Shemma, H. McCarrick, T.-C. Chien, J. D. Teoh, P. Winkel, et al. (35 authors), R. J. Schoelkopf | 2024 (arXiv 2023) | 2307.03169 | 10.1038/s41567-024-02539-4 (Nat. Phys. 20, 1454-1460) | https://arxiv.org/abs/2307.03169 | yes (abstract; Crossref) | Q3 |
| Zhang2026 | Logical qubits with erasure conversion using metastable neutral atoms (arXiv v1: "Leveraging erasure errors in logical qubits with metastable 171Yb atoms") | B. Zhang, G. Liu, G. Bornet, S. P. Horvath, P. Peng, S. Ma, S. Huang, S. Puri, J. D. Thompson | 2026 (arXiv 2025) | 2506.13724 | 10.1038/s41567-026-03309-0 (Nat. Phys. 22, 910-916) | https://arxiv.org/abs/2506.13724 | yes (abstract + full text v1; Crossref) | Q1, Q3 |
| Reichardt2024 | Fault-tolerant quantum computation with a neutral atom processor (v1: "Logical computation demonstrated with a neutral atom quantum processor") | B. W. Reichardt, A. Paetznick, D. Aasen, I. Basov, J. M. Bello-Rivas, et al. (72 authors) | 2024 | 2411.11822 | none listed (arXiv DOI 10.48550/arXiv.2411.11822) | https://arxiv.org/abs/2411.11822 | yes (abstract + full text v3) | Q1, Q3 |
| Chow2024 | Circuit-based leakage-to-erasure conversion in a neutral atom quantum processor | M. N. H. Chow, V. Buchemmavari, S. Omanakuttan, B. J. Little, S. Pandey, I. H. Deutsch, Y.-Y. Jau | 2024 | 2405.10434 | 10.1103/PRXQuantum.5.040343 (PRX Quantum 5, 040343) | https://arxiv.org/abs/2405.10434 | yes (abstract) | Q3 |
| Quinn2026 | High-fidelity entanglement of metastable trapped-ion qubits with integrated erasure conversion | A. Quinn, G. J. Gregory, I. D. Moore, S. Brudney, J. Metzner, E. R. Ritchie, J. O'Reilly, D. J. Wineland, D. T. C. Allcock | 2026 (arXiv 2024) | 2411.12727 | 10.1103/p3cy-8yjk (Phys. Rev. A 113, L040601) | https://arxiv.org/abs/2411.12727 | yes (abstract; Crossref) | Q3 |
| Ransford2025 | Helios: A 98-qubit trapped-ion quantum computer | A. Ransford, M. S. Allman, J. Arkinstall, J. P. Campora III, S. F. Cooper, R. D. Delaney, et al. (186 authors) | 2025 | 2511.05465 | none listed | https://arxiv.org/abs/2511.05465 | yes (abstract + full text) | Q2, Q3 |
| McEwen2021 | Removing leakage-induced correlated errors in superconducting quantum error correction | M. McEwen, D. Kafri, Z. Chen, J. Atalaya, K. J. Satzinger, et al. (50 authors), R. Barends | 2021 | 2102.06131 | 10.1038/s41467-021-21982-y (Nat. Commun. 12, 1761) | https://arxiv.org/abs/2102.06131 | yes (abstract only; PDF text not extractable) | Q2, Q3 |
| Miao2023 | Overcoming leakage in quantum error correction (arXiv title: "Overcoming leakage in scalable quantum error correction") | K. C. Miao, M. McEwen, J. Atalaya, D. Kafri, L. P. Pryadko, et al. (117 authors) | 2023 (arXiv 2022) | 2211.04728 | 10.1038/s41567-023-02226-w (Nat. Phys. 19, 1780-1786) | https://arxiv.org/abs/2211.04728 | yes (abstract + full text; Crossref) | Q2, Q3 |
| Battistel2021 | A hardware-efficient leakage-reduction scheme for quantum error correction with superconducting transmon qubits | F. Battistel, B. M. Varbanov, B. M. Terhal | 2021 | 2102.08336 | 10.1103/PRXQuantum.2.030314 (PRX Quantum 2, 030314) | https://arxiv.org/abs/2102.08336 | yes (abstract) | Q2, Q3 |
| Xin2026 | Improved error correction with leakage reduction units built into qubit measurement in a superconducting quantum processor | Y. Xin, S. L. M. van der Meer, M. Serra-Peralta, T. H. F. Vroomans, M. Finkel, et al. (8 authors), L. DiCarlo | 2026 (arXiv 2025) | 2511.17460 | 10.1103/y66v-988s (PRX Quantum 7, 033062) | https://arxiv.org/abs/2511.17460 | yes (abstract; Crossref) | Q2, Q3 |
| Varbanov2020 | Leakage detection for a transmon-based surface code | B. M. Varbanov, F. Battistel, B. M. Tarasinski, V. P. Ostroukh, T. E. O'Brien, L. DiCarlo, B. M. Terhal | 2020 | 2002.07119 | 10.1038/s41534-020-00330-w (npj Quantum Inf. 6, 102) | https://arxiv.org/abs/2002.07119 | yes (abstract; full-text HTML conversion failed) | Q1, Q2 |
| WoodGambetta2018 | Quantification and characterization of leakage errors | C. J. Wood, J. M. Gambetta | 2018 (arXiv 2017) | 1704.03081 | 10.1103/PhysRevA.97.032306 (PRA 97, 032306) | https://arxiv.org/abs/1704.03081 | yes (abstract + full text) | Q2 |
| Wallman2016 | Robust characterization of leakage errors (arXiv title: "Characterization of leakage errors via randomized benchmarking") | J. J. Wallman, M. Barnhill, J. Emerson | 2016 (arXiv 2014) | 1412.4126 | 10.1088/1367-2630/18/4/043021 (New J. Phys. 18, 043021) | https://arxiv.org/abs/1412.4126 | yes (abstract + full text; Crossref) | Q2 |
| Ceasura2022 | Non-exponential behaviour in logical randomized benchmarking | A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan | 2022 | 2212.05488 | none listed | https://arxiv.org/abs/2212.05488 | yes (abstract) | Q2 |
| ChaoReichardt2018 | Quantum error correction with only two extra qubits | R. Chao, B. W. Reichardt | 2018 (arXiv 2017) | 1705.02329 | 10.1103/PhysRevLett.121.050502 (PRL 121, 050502) | https://arxiv.org/abs/1705.02329 | yes (abstract + full text) | Q1 |
| Chamberland2020 | Topological and subsystem codes on low-degree graphs with flag qubits | C. Chamberland, G. Zhu, T. J. Yoder, J. B. Hertzberg, A. W. Cross | 2020 (arXiv 2019) | 1907.09528 | 10.1103/PhysRevX.10.011022 (PRX 10, 011022) | https://arxiv.org/abs/1907.09528 | yes (abstract) | Q1, Q3 |

Secondary sources (also fetched and verified; each is cited once or for
context only):

| Key | Title | Authors | Year | arXiv | DOI (venue) | URL | Fetched | Bears on |
|---|---|---|---|---|---|---|---|---|
| ChamberlandBeverland2018 | Flag fault-tolerant error correction with arbitrary distance codes | C. Chamberland, M. E. Beverland | 2018 (arXiv 2017) | 1708.02246 | 10.22331/q-2018-02-08-53 (Quantum 2, 53) | https://arxiv.org/abs/1708.02246 | yes (abstract) | Q1 |
| Grassl1997 | Codes for the quantum erasure channel | M. Grassl, T. Beth, T. Pellizzari | 1997 (arXiv 1996) | quant-ph/9610042 | 10.1103/PhysRevA.56.33 (PRA 56, 33-38) | https://arxiv.org/abs/quant-ph/9610042 | yes (abstract) | Q1 |
| Ma2023 | High-fidelity gates and mid-circuit erasure conversion in an atomic qubit (arXiv: "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit") | S. Ma, G. Liu, P. Peng, B. Zhang, S. Jandura, J. Claes, A. P. Burgers, G. Pupillo, S. Puri, J. D. Thompson | 2023 | 2305.05493 | 10.1038/s41586-023-06438-1 (Nature 622, 279-284) | https://arxiv.org/abs/2305.05493 | yes (abstract; Crossref) | Q3 |
| Mehta2025 | Bias-preserving and error-detectable entangling operations in a superconducting dual-rail system | N. Mehta, J. D. Teoh, T. Noh, A. Agrawal, A. Anderson, et al. (73 authors), R. J. Schoelkopf | 2025 | 2503.10935 | none listed | https://arxiv.org/abs/2503.10935 | yes (abstract) | Q3 |
| Hung2026 | Fast, high-fidelity erasure detection of dual-rail qubits with symmetrically coupled readout | J. S.-C. Hung, A. Haim, M. Raha, G. Kim, Z. Huang, et al. (17 authors), H. Levine, O. Painter | 2026 | 2604.16292 | none listed | https://arxiv.org/abs/2604.16292 | yes (abstract) | Q3 |
| Marques2023 | All-microwave leakage reduction units for quantum error correction with superconducting transmon qubits | J. F. Marques, H. Ali, B. M. Varbanov, et al. (12 authors), L. DiCarlo | 2023 | 2302.09876 | 10.1103/PhysRevLett.130.250602 (PRL 130, 250602) | https://arxiv.org/abs/2302.09876 | yes (abstract) | Q2, Q3 |
| Chasseur2015 | A complete randomized benchmarking protocol accounting for leakage errors | T. Chasseur, F. K. Wilhelm | 2015 | 1505.00580 | 10.1103/PhysRevA.92.042333 (PRA 92, 042333) | https://arxiv.org/abs/1505.00580 | yes (abstract + full text) | Q2 |
| Stricker2020 | Experimental deterministic correction of qubit loss (arXiv: "Deterministic correction of qubit loss") | R. Stricker, D. Vodola, A. Erhard, L. Postler, M. Meth, M. Ringbauer, P. Schindler, T. Monz, M. Müller, R. Blatt | 2020 | 2002.09532 | 10.1038/s41586-020-2667-0 (Nature 585, 207-210) | https://arxiv.org/abs/2002.09532 | yes (abstract + full text) | Q3 |

## 2. Key findings (all **from literature**)

### A. What heralded erasures buy a distance-2 code

1. A distance-d code can correct any d-1 erasures but is guaranteed to correct
   only floor((d-1)/2) arbitrary unlocated single-qubit errors. A d=2 code
   therefore goes from correcting no unlocated error to correcting one
   heralded erasure. [Kubica2023] (Q1)
2. Four qubits are necessary and sufficient to encode one qubit against one
   erasure, compared with five against one unlocated error. [Grassl1997] (Q1)
3. Wu et al. estimate that 98% of errors in metastable 171Yb can be converted
   to erasures. In their model this raises the circuit-level surface-code
   threshold from 0.937% to 4.15%. The logical-error exponent is nu_e = d for
   erasures against nu_p = (d+1)/2 for Pauli errors, with erasure detection
   assumed after each gate. [Wu2022] (Q1, Q3)
4. Zhang et al. encode in a `[[4,2,2]]` code that is "too small to correct any
   Pauli errors" and still correct single heralded erasures. Their rule: if
   the parity is odd and exactly one erasure is present, flip the erased
   qubit. Erasure information raises unconditional state-preparation fidelity
   from 0.874(4) to 0.902(3) and makes postselected error growth 3.6(1) times
   slower. [Zhang2026] (Q1, Q3)
5. Erasure fractions in that Yb platform's error model (their Table 1) are
   about 0.56 for one-qubit Cliffords, 0.31 for the amplitude-robust CZ, 0.72
   for idling and 0.5 for transport. The paper states that roughly 31% of CZ
   gate error is detected mid-circuit as erasure. [Zhang2026] (Q3)
6. Reichardt et al. use the `[[4,2,2]]` code to "detect errors and correct for
   lost atoms". For a 24-qubit logical cat state the results are:
   - p_X + p_Z = 42.0% unencoded;
   - 26.6% with error detection plus loss correction;
   - 29.5% before conditioning on no loss, falling to 10.2% after it, but only
     858 of 29,760 trials had no loss.
   [Reichardt2024] (Q1, Q3)
7. Once a qubit is lost, the distance-2 code gives no protection against
   errors on the remaining qubits. [Reichardt2024] (Q1)
8. In erasure-qubit decoding, false-negative erasure detections act as
   ordinary Pauli errors. False positives only change the edge weights used by
   the matching decoder. [Kubica2023] (Q1)
9. Erasure checks cost time and add noise, so check schedules have to be
   optimized. Applying a reset pulse only to qubits flagged as erased cuts the
   impact of measurement errors about in half. Erasure-qubit protocols can beat
   state-of-the-art transmons "even in the absence of precise information
   about the locations of erasure errors." [Gu2025] (Q1, Q3)

### B. Erasure qubits on hardware

10. Superconducting, transmon dual-rail: T1 errors become erasures, with
    erasure probability 2.19(2)e-3 per single-qubit gate and residual errors
    about 40 times lower. Mid-circuit erasure detection costs less than 0.1%
    dephasing per check. [Levine2024] (Q3)
11. Superconducting, transmon dual-rail, later readout design: a 384 ns
    erasure check has residual error 6.0(2)e-4 per check and induced dephasing
    8(3)e-5. [Hung2026] (Q3)
12. Superconducting, cavity dual-rail: logical SPAM error is at the 0.01%
    level and more than 99% of cavity decays are detected as erasures. Decay
    occurs at about 0.2% per microsecond, phase flips are about 6 times rarer
    and bit flips at least 140 times rarer. [Chou2024] (Q3)
13. Superconducting, cavity dual-rail two-qubit gate: about 500 ns, residual
    infidelity below 0.1% after error detection, and about 0.5% erasure per
    gate. Bit flips are bounded at parts per million. [Mehta2025] (Q3)
14. Superconducting, without dual-rail hardware: Kubica et al. also propose a
    g-f qubit in the lowest three levels of one transmon. Population in |e>
    heralds the erasure and is detected by mapping it onto an extra transmon.
    [Kubica2023] (Q3)
15. Neutral atoms, Yb: metastable 171Yb nuclear-spin qubits reach 0.9990(1)
    one-qubit and 0.980(1) two-qubit gate fidelity, with mid-circuit erasure
    detection. The detection step adds less than 1e-5 error to atoms still in
    the qubit subspace. [Ma2023] (Q3)
16. Neutral atoms, Cs: leakage detection units map atom presence onto an
    ancilla without disturbing the stored qubit. They convert all major leakage
    pathways to erasures and detect atom loss with about 93.4% accuracy.
    [Chow2024] (Q3)
17. Trapped ions, metastable qubits: about 94% of spontaneous Raman scattering
    gate errors are detected as erasures. Bell-state fidelity is 97.73% raw and
    99.16% with erasures subtracted. [Quinn2026] (Q3)
18. Trapped ions, Quantinuum Helios: the native ternary measurement returns 0,
    1 or L. It measures leakage of 2.4(1)e-4 per two-qubit gate (RB) against
    7.9(2)e-4 average two-qubit infidelity, and 1.12(6)e-5 per one-qubit
    Clifford against 2.5(1)e-5 infidelity. Transport leakage accounts for
    nearly all of the linear memory error. [Ransford2025] (Q2, Q3)

### C. Leakage as a non-Markovian error source

19. Leakage populates long-lived, mobile higher levels in transmons, which
    "opens a path to errors that are correlated in space and time". A reset
    that returns all relevant higher levels to the ground state lowers logical
    error and stabilizes error suppression with size. [McEwen2021] (Q2, Q3)
20. Without leakage removal, data-qubit leakage builds to about 5%, and
    detection-event correlations more than one round apart exceed 1%. With
    data-qubit leakage removal (DQLR) every cycle, leakage stays near 1e-3,
    those correlations fall to about 2e-3 or less, and injected leakage acts
    roughly like an uncorrelated error. [Miao2023] (Q2, Q3)
21. Miao et al. characterize leakage with readout that resolves |2> (their
    pulses resolve the four lowest levels). They note that "identifying and
    post-selecting out leakage events" enabled earlier surface-code
    experiments, but they mitigate by removal, not postselection. [Miao2023]
    (Q2, Q3)
22. Two leakage-reduction units with no extra hardware or cycle time are
    proposed: a microwave drive moves data-qubit leakage to the readout
    resonator, and a measurement-conditioned 1-2 pi pulse handles ancillas.
    They cut the average leakage lifetime to almost one QEC cycle in d=3
    surface-code simulations. [Battistel2021] (Q2, Q3)
23. An LRU that runs during transmon measurement removes 98.4% of leakage with
    no time overhead and 99.2% computational-state assignment fidelity. It
    suppresses logical error in memory and stability experiments without
    postselection. [Xin2026] (Q2, Q3)
24. All-microwave LRUs keep leakage buildup below 1% over 50 cycles of a
    weight-2 parity check and lower the error-detection rate. [Marques2023]
    (Q2, Q3)
25. Leakage can be located in time and space with hidden Markov models on the
    syndrome record plus analog ancilla readout. Postselecting detected leakage
    brings the logical error below memory break-even but discards about 47% of
    the data. [Varbanov2020] (Q1, Q2)
26. With leakage, RB population in the computational subspace decays as
    A + B lambda_1^m with lambda_1 = 1 - L1 - L2 toward a nonzero steady state.
    Survival is bi-exponential for one qubit and multi-exponential for several,
    and poorly twirled coherent leakage can produce non-Markovian oscillations.
    [WoodGambetta2018] (Q2)
27. Leakage RB applies to any subspace of interest, including "the logical
    code space defined by some quantum error-correcting code". Coherent leakage
    is "fundamentally non-Markovian" because population returns.
    [Wallman2016] (Q2)
28. Standard RB does not account for leakage, "not least because it is a
    non-Markovian process". The leakage-aware fit is a sum of exponentials with
    at least one eigenvalue equal to 1. [Chasseur2015] (Q2)
29. In logical RB, syndrome qubits coupled to the logical qubit through a
    decoder can act as hidden memory of the gate history. That memory produces
    oscillating, non-exponential decays even under gate- and time-independent
    noise. [Ceasura2022] (Q2)

### D. Flags and hook errors

30. A single ancilla fault while measuring a weight-4 `[[5,1,3]]` stabilizer
    can spread into a weight-2 data error. Adding a flag lets the code be
    error-corrected fault-tolerantly with 7 qubits in total.
    [ChaoReichardt2018] (Q1)
31. For `[[n,n-2,2]]` error-detecting codes, measuring the single weight-n check
    with one unflagged ancilla is not fault tolerant: a single fault can cause
    an undetectable logical error. One flag makes every single fault
    detectable or harmless. [ChaoReichardt2018] (Q1)
32. Flag error correction extends beyond distance 3 for code families that
    satisfy stated conditions. [ChamberlandBeverland2018] (Q1)
33. On low-degree superconducting layouts (heavy-hex, heavy-square), flag
    qubits plus flag-aware MWPM decoding recover the full code distance.
    [Chamberland2020] (Q1, Q3)

## 3. Implications for the qubit `[[5,1,2]]` LRB study

### 3a. Facts about our code (my checks, Appendix A)

- **Verified numerically.** The code has distance 2, and its only weight-2
  logicals are X0X3, X2X4, Z0Z2 and Z3Z4. This confirms the "to be verified"
  representatives in CONTEXT (`X_L ~ X2X4`, `Z_L ~ Z3Z4`). There are 48
  nontrivial logical Paulis up to sign. Y-type logicals have weight at least 3.
- **Verified numerically.** The single-qubit syndrome table in CONTEXT is
  correct. For every data qubit, X, Y and Z give three distinct nonzero
  syndromes. So one heralded erasure at a known qubit is always corrected:
  read the syndrome and apply the matching Pauli on that qubit.
- **Verified numerically.** Erasure patterns:
  - all 5 single erasures are correctable;
  - 6 of the 10 pairs are correctable: {0,1}, {0,4}, {1,2}, {1,3}, {1,4},
    {2,3};
  - the pairs {0,2}, {0,3}, {2,4} and {3,4} carry a logical and are not
    correctable;
  - none of the 10 triples is correctable.

  This is the d-1 = 1 guarantee of finding 1, plus extra pairs that the
  guarantee does not promise.
- **Verified numerically.** Hook errors and single faults in one check round
  (CONTEXT CNOT orders; every non-identity 1- or 2-qubit Pauli after every CX
  and H):
  - No single fault leaves an undetectable nontrivial logical.
  - Pure ancilla hook errors reduce, up to a stabilizer, to weight 1:
    Z1Z3 ~ Z0, Z2Z4 ~ Z1, X1X2 ~ X0, X1X4 ~ X3.
  - Correlated data-plus-ancilla faults on a CNOT can leave weight-2 mixed
    errors that do not reduce: X1Z3, Y1Z3, X2Z4, Y2Z4, Y1X2, Z1X2, Y1X4, Z1X4.
    All have nonzero syndrome.
  - So, unlike the `[[n,n-2,2]]` case (finding 31), flags are not needed to keep
    LRB-D fault tolerant at first order.
- **Verified numerically.** The installed stim 1.15.0 has `HERALDED_ERASE` and
  `HERALDED_PAULI_CHANNEL_1`. Both write a herald bit to the measurement
  record, so erasure conversion can be simulated in the existing pipeline and
  the herald passed to a decoder.
- **Verified analytically.** Without heralds, one fault is enough to make
  correction fail on this code: X0/X3, X2/X4, Z0/Z2 and Z3/Z4 share syndromes
  but differ by a logical. Correction-only logical error per round is
  therefore first order in p.
- **Verified analytically.** For LRB-D, every single fault that a used check
  later sees is detected, so its accepted error is second order in p, with
  acceptance 1 - O(p). The exception is the tail of the terminal check round:
  a Z0 or Z3 data error after the X-check CNOTs flips X0X3 and no later check
  sees it.
  - With the `data2` terminal (data 0 and 3 only, as in the qutrit pipeline),
    this tail gives a first-order, depth-independent error floor.
  - The `data5` terminal option already in `src/lrb_qubit/circuits.py` adds
    final X-stabilizer detectors that catch it.
- **Verified analytically.** If every fault were a heralded erasure, failure
  would need two erasures on one of the four bad pairs. The fault orders are
  therefore nu = 1 for correction-only, 2 for LRB-D and 2 for perfectly
  heralded erasures: the d=2 version of finding 3.
- **Verified analytically.** X0X3 and X2X4 are disjoint X_L representatives.
  Reading all five data qubits in the X basis at the end (`const=0`, or the
  `data5` terminal) gives two independent X_L estimates plus both
  X-stabilizer parities.

### 3b. Conjectures (my inference, to be tested)

- **Q1, hybrid decoder.** The natural correcting policy for a d=2 LRB is the
  experimental `[[4,2,2]]` one (findings 4 and 6), and it may be the only one
  that keeps second-order accepted error:
  1. Within each policy window, if the heralded erasures form a correctable
     set (one qubit, or one of the 6 good pairs) and the syndrome is
     consistent with Paulis on those qubits, apply the correction.
  2. Otherwise reject, as LRB-D does.

  Acceptance should rise roughly by the fraction of rejected shots whose
  faults were all heralded, about R_e at first order, while accepted F stays
  close to LRB-D. Expected gains at realistic R_e (0.3-0.7, finding 5) are much
  smaller than at the 0.98 projection (finding 3).
- **Q1, decoding implementation.** For 5 data qubits, an exact maximum-likelihood
  lookup conditioned on (syndrome history, herald pattern) is tiny. Heralds
  can also enter a DEM-based MWPM decoder by setting erased-location
  probabilities to 1/2 per shot (Kubica2023 handles false positives through
  edge weights, finding 8). Use `HERALDED_ERASE` in place of a fraction R_e of
  each SI1000 depolarizing channel, sweep R_e over {0, 0.3, 0.5, 0.98}, and
  compare against LRB-D on the same circuits.
- **Q1, circuit correlations.** The mixed weight-2 single-fault errors (3a)
  correlate the X and Z syndromes, so a circuit-level ML decoder may break some
  ambiguous-pair ties that a code-capacity decoder treats as 50/50. One
  example: X1Z3 makes Z3 more likely than Z4 when Z-syndrome (1,1) and
  X-syndrome (0,1) appear together. The symmetric ambiguity (e.g. X0 versus X3
  from independent data noise) cannot be removed without heralds.
- **Q2, rejection as code-space leakage.** LRB-D rejection behaves like leakage
  out of the code space, in the sense of Wallman2016 (finding 27):
  - "leakage" is a transition to a nontrivial syndrome sector;
  - "seepage" is a later fault that restores a trivial syndrome, possibly
    applying a logical.

  With `const=1` (terminal check only), acceptance should approach a
  fully-mixed steady state of 1/16 for qubits (4 binary checks). This mirrors
  the qutrit saturation at rejection 1 - 1/81 in CONTEXT and the
  L1/(L1+L2) leakage steady state of WoodGambetta2018. Accepted F would then
  mix never-left and left-and-returned shots, giving multi-exponential or
  non-exponential decays (findings 26-29).
- **Q2, correction as an LRU for the syndrome sector.** Per-round correction
  returns every shot to the trivial sector, like an LRU or DQLR returning
  leaked population every cycle. That limits the hidden syndrome memory to
  about one round (Battistel2021, Miao2023: lifetime about one cycle,
  long-range time correlations suppressed). The memory is not removed for
  free: it becomes a Markovian logical error whenever the d=2 decoder guesses
  wrong.
  - Prediction: correction-based LRB decays should be close to single
    exponentials, with a larger per-round rate than LRB-D at small m and no
    acceptance collapse.
  - Caveat (Ceasura2022): a decoder that carries syndrome history across rounds
    can re-create memory. Per-round decoding with Pauli-frame updates is the
    clean test.
- **Q2, physical leakage.** Real leakage (transmon |2>, ion leakage) adds a
  genuine non-Markovian source that SI1000 lacks:
  - under LRB-D it would appear as bursts of rejection over consecutive rounds
    (findings 20 and 25);
  - under correction it would appear as repeated wrong corrections;
  - mitigation is an LRU every round (findings 22-24) plus heralding through
    readout that resolves |2> or a ternary measurement (findings 18 and 21),
    which turns leakage into erasures the code can correct (finding 4).
- **Q3, superconducting (IQM-like transmons).**
  - The least intrusive path is an LRU every round, ideally built into
    readout (finding 23), with |2>-resolving ancilla readout as a herald.
  - Native erasure needs dual-rail pairs (findings 10-13), which doubles the
    transmon count (18 instead of 9 for this code), or the g-f encoding
    (finding 14).
  - The 9-qubit layout is a 3x3 nearest-neighbour square patch: data at the
    corners and centre, ancillas at edge midpoints (**verified analytically**).
    The fold gates, however, need a SWAP/CZ between data 2 and 3 on opposite
    corners, which a square lattice does not provide. That is a question for
    the hardware note.
- **Q3, ions and neutral atoms.** Heralding is available now:
  - Helios ternary readout. About 30% of the average two-qubit infidelity is
    leakage (2.4e-4 / 7.9e-4, ratio computed by me from finding 18). Leakage
    also dominates transport memory error.
  - Metastable-ion erasure conversion (finding 17).
  - Yb erasure conversion and atom-loss imaging (findings 4-6, 15).

  All-to-all or movable connectivity also makes the SWAP(2,3)/CZ(2,3) fold
  gates native. For the hybrid policy these platforms look like the most
  sustainable near-term targets.
- **Q1/Q3, reporting.** Erasure-aware LRB should report three numbers per
  depth: herald rate, rejection of unexplained syndromes, and accepted logical
  fidelity. This matches how dual-rail papers separate erasure from residual
  error by conditioning on erasure checks (findings 10-12).
- **Q3, readout.** In hardware (not in our ideal-readout simulation), a
  ternary or erasure-detecting `data5`-style terminal readout would do two
  things. It would let the LRB outcome fall back to X2X4 when qubit 0 or 3 is
  heralded, and to X0X3 when qubit 2 or 4 is. It would also catch the
  terminal-round tail errors described in 3a.

## 4. Seed corrections

- **Chou et al.** The seed ID arXiv:2307.08737 is wrong. Chou et al. is
  **arXiv:2307.03169**, "Demonstrating a superconducting dual-rail cavity qubit
  with erasure-detected logical measurements". The published title drops
  "Demonstrating": Nat. Phys. 20, 1454-1460 (2024),
  DOI 10.1038/s41567-024-02539-4.
- **Levine et al.** This is the paper at **arXiv:2307.08737**, "Demonstrating a
  long-coherence dual-rail erasure qubit using tunable transmons", PRX 14,
  011051 (2024). The seed's guess for Chou is Levine's ID.
- **Miao et al.** The arXiv title is "Overcoming leakage in *scalable* quantum
  error correction" (2211.04728). The journal title matches the seed:
  "Overcoming leakage in quantum error correction", Nat. Phys. 19, 1780-1786
  (2023).
- **Battistel et al.** The full title is "A hardware-efficient
  leakage-reduction scheme for quantum error correction with superconducting
  transmon qubits", PRX Quantum 2, 030314 (2021). The ID 2102.08336 is correct.
- **Kubica et al.** Confirmed: authors Kubica, Haim, Vaknin, Brandão, Retzker,
  PRX 13, 041022 (2023). The capital "O" in "Overcoming" follows the arXiv
  listing.
- **Confirmed unchanged:**
  - Wu et al. 2201.03540, Nat. Commun. 13, 4657 (2022);
  - McEwen et al. 2102.06131, Nat. Commun. 12, 1761 (2021);
  - Chao and Reichardt 1705.02329, PRL 121, 050502 (2018);
  - Chamberland and Beverland 1708.02246, "Flag fault-tolerant error correction
    with arbitrary distance codes", Quantum 2, 53 (2018).
- **Title changes in non-seed sources**, worth knowing when cross-citing:
  - Reichardt2024 v1 was "Logical computation demonstrated with a neutral atom
    quantum processor";
  - Zhang2026 v1 was "Leveraging erasure errors in logical qubits with
    metastable 171Yb atoms";
  - Wallman2016's arXiv title differs from the NJP title;
  - Stricker2020's arXiv title omits "Experimental".
- **Stricker2020 caveat.** Its 4-qubit "minimal surface code" has a weight-1
  logical X (X4). It is a loss-specific construction, not a distance-2 code,
  so it should not be cited as a d-1 erasure example.

## 5. Open questions

1. At realistic R_e (0.3-0.7) and SI1000 strength, how much LRB-D acceptance
   does the hybrid "correct heralded, reject unexplained" policy recover?
   Does accepted F stay at the LRB-D level?
2. Does per-round correction make the qubit LRB decay single-exponential, as
   the LRU analogy predicts? Or does a history-aware decoder bring back the
   memory effects of Ceasura2022?
3. Does the `const=1` acceptance really saturate at 1/16 for the qubit code?
   Can the decay be fit by a two-eigenvalue leakage-RB model (Wallman2016,
   WoodGambetta2018) with "leakage" and "seepage" rates read off the
   syndrome-sector dynamics?
4. How should physical leakage be modelled in a stim-based qubit simulation?
   Options include heralded erasure, unheralded erasure, or a multi-round
   "sticky" error. Which approximation keeps the time-correlated signature of
   finding 20?
5. Erasure-check timing: after every gate (Wu2022) or once per round (Gu2025)?
   For LRB policies, should herald windows follow the `const`/`unif` check
   windows or every round?
6. Does timing separate the mixed X/Z single-fault errors in 3a enough for a
   DEM decoder to break ambiguous-pair ties? (Hand off to the decoders
   agent.)
7. Do the qutrit `[[5,1,2]]_3` check circuits also avoid single-fault
   undetectable logicals? The GF(3) hook equivalences differ and have not been
   checked here.
8. False heralds: a false positive costs nothing when the syndrome is trivial,
   but how often does a false negative plus another fault give an accepted
   logical error at second order?

## Appendix A. Light numerical check (stim 1.15.0, Python 3.11, under a second)

The script enumerates all 4^5 data Paulis to find the logical operators,
tests every erasure set, and pushes every single Pauli fault (after each CX
and H of one check round, CONTEXT ordering, identical to
`src/lrb_qubit/code.py`) to the end of the round with
`stim.PauliString.after`. Key output:

```
code distance (min logical weight): 2
weight-2 logicals: X0X3, Z0Z2, X2X4, Z3Z4
1-erasure patterns: correctable 5/5
2-erasure patterns: correctable 6/10; uncorrectable: (0,2) (0,3) (2,4) (3,4)
3-erasure patterns: correctable 0/10
single faults giving an UNDETECTABLE nontrivial logical: []
pure ancilla hooks: Z1Z3~Z0, Z2Z4~Z1, X1X2~X0, X1X4~X3
irreducible weight-2 single-fault data errors: X1Z3 Y1Z3 X2Z4 Y2Z4 Y1X2 Z1X2 Y1X4 Z1X4 (all nonzero syndrome)
HERALDED_ERASE / HERALDED_PAULI_CHANNEL_1 available in stim 1.15.0 (each emits a herald record bit)
```

Script (re-run with `module load python/3.11; python <file>`):

```python
import itertools, stim
N = 5
STABS = [stim.PauliString(s) for s in ("ZZ_Z_", "_ZZ_Z", "XXX__", "_X_XX")]

def in_stab_group(p):
    for bits in itertools.product([0, 1], repeat=4):
        q = stim.PauliString(N)
        for b, s in zip(bits, STABS):
            if b: q = q * s
        if all(q[i] == p[i] for i in range(N)): return True
    return False
syndrome = lambda p: tuple(0 if p.commutes(s) else 1 for s in STABS)
support = lambda p: frozenset(i for i in range(N) if p[i] != 0)

logicals = []
for v in itertools.product(range(4), repeat=N):
    p = stim.PauliString(N)
    for i, x in enumerate(v): p[i] = x
    if all(p.commutes(s) for s in STABS) and not in_stab_group(p): logicals.append(p)
for k in (1, 2, 3):
    bad = [S for S in itertools.combinations(range(N), k)
           if any(support(p) <= set(S) for p in logicals)]
    print(k, "erasures: uncorrectable", bad)

ops = [("CX",(0,5)),("CX",(1,5)),("CX",(3,5)),("M",(5,)),("CX",(1,6)),("CX",(2,6)),
       ("CX",(4,6)),("M",(6,)),("H",(7,)),("CX",(7,0)),("CX",(7,1)),("CX",(7,2)),
       ("H",(7,)),("M",(7,)),("H",(8,)),("CX",(8,3)),("CX",(8,1)),("CX",(8,4)),
       ("H",(8,)),("M",(8,))]
for i, (g, t) in enumerate(ops):
    if g not in ("CX", "H"): continue
    for v in itertools.product(range(4), repeat=len(t)):
        if not any(v): continue
        f = stim.PauliString(9)
        for q, x in zip(t, v): f[q] = x
        c = stim.Circuit(); c.append("I", [8])
        for g2, t2 in ops[i + 1:]:
            if g2 in ("CX", "H"): c.append(g2, list(t2))
        out = f.after(c)
        d = stim.PauliString(N)
        for q in range(N): d[q] = out[q]
        if support(d) and syndrome(d) == (0,0,0,0) and not in_stab_group(d):
            print("UNDETECTABLE LOGICAL from fault", f, "after op", i)
```
