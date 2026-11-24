# Lit review 01: logical randomized benchmarking (LRB), incl. postselected / error-detected LRB

Scope: what LRB measures, how decay parameters map to logical error per round,
LRB with error-detecting codes and postselection, qudit LRB, interleaved LRB,
and hardware experiments (superconducting, trapped ions, neutral atoms). The
emphasis is on work that analyses how postselection or decoding changes the RB
decay model.

Labels used: **from literature** (with key), **verified analytically**,
**verified numerically**, **conjecture**. Every source below was fetched by me
on 2026-10-28. "abs" means I read the arXiv abstract page (title, authors, date
and IDs confirmed there). "full" means I also read the relevant sections of the
arXiv HTML or ar5iv full text. Numbers quoted from arXiv full texts are from the
version I read, and journal versions may differ slightly.

Note on Kabir2025: this is the user's group's own paper (Sohan Ghosh is a
co-author). I cite it for what it states, not as independent confirmation.

---

## 1. Bibliography

### 1a. Core sources (20)

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched | Read | Relevance |
|---|---|---|---|---|---|---|---|---|
| Combes2017 | Logical Randomized Benchmarking | J. Combes, C. Granade, C. Ferrie, S. T. Flammia | 2017 | arXiv:1702.03688 (no journal ref on arXiv page) | https://arxiv.org/abs/1702.03688 | yes | abs + full (ar5iv, main text) | Q1, Q2: defines LRB |
| HarperFlammia2019 | Fault-Tolerant Logical Gates in the IBM Quantum Experience | R. Harper, S. T. Flammia | 2018 (arXiv), 2019 (PRL) | arXiv:1806.02359; DOI 10.1103/PhysRevLett.122.080504 (PRL 122, 080504) | https://arxiv.org/abs/1806.02359 | yes | abs + full (ar5iv) | Q1, Q3: postselected LRB on a [[4,2,2]] code |
| Ceasura2022 | Non-Exponential Behaviour in Logical Randomized Benchmarking | A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan | 2022 | arXiv:2212.05488 | https://arxiv.org/abs/2212.05488 | yes | abs (arXiv). Full text through the alphaxiv mirror (https://www.alphaxiv.org/abs/2212.05488.md) because the ar5iv render was truncated | Q2: non-exponential LRB from syndrome memory |
| Ziyad2025 | Emergent Non-Markovianity in Logical Qubit Dynamics | J. A. Ziyad, R. Blume-Kohout, K. Rudinger | 2025 (v3 Feb 2026) | arXiv:2512.08893 | https://arxiv.org/abs/2512.08893 | yes | abs | Q2: condition for logical non-Markovianity |
| Kabir2025 | Sdim: A Qudit Stabilizer Simulator | A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang, B. Sorée | 2025 (v2 May 2026) | arXiv:2511.12777 | https://arxiv.org/abs/2511.12777 | yes | abs + full (arXiv HTML v2, Secs. VII to VIII) | Q2: qutrit LRB-D (in-group source) |
| ChenBaldwin2025 | Randomized Benchmarking with Leakage Errors | Y.-H. Chen, C. H. Baldwin | 2025 | arXiv:2502.00154; DOI 10.1103/j8sz-6ws9 (Phys. Rev. Research 7, 043065) | https://arxiv.org/abs/2502.00154 | yes | abs + full (arXiv HTML, first ~70%) | Q1, Q2: theory of postselected RB decays |
| Lacroix2025 | Scaling and logic in the color code on a superconducting quantum processor | N. Lacroix, A. Bourassa, F. J. H. Heras, L. M. Zhang, J. Bausch, ..., K. J. Satzinger (215 authors) | 2024 (arXiv), 2025 (Nature) | arXiv:2412.14256; DOI 10.1038/s41586-025-09061-4 (Nature 645, 614-619) | https://arxiv.org/abs/2412.14256 | yes | abs + full (HTML v1: Sec. IV, Fig. 3, App. E, App. G) | Q1, Q3: superconducting LRB with decoding, interleaved vs idle reference |
| Mayer2024 | Benchmarking logical three-qubit quantum Fourier transform encoded in the Steane code on a trapped-ion quantum computer | K. Mayer, C. Ryan-Anderson, N. Brown, E. Durso-Sabina, C. H. Baldwin, ..., R. Stutz (23 authors) | 2024 | arXiv:2404.08616 | https://arxiv.org/abs/2404.08616 | yes | abs + full (arXiv HTML) | Q3: trapped-ion logical RB |
| Proctor2022 | Scalable randomized benchmarking of quantum computers using mirror circuits | T. Proctor, S. Seritan, K. Rudinger, E. Nielsen, R. Blume-Kohout, K. Young | 2021 (arXiv), 2022 (PRL) | arXiv:2112.09853; DOI 10.1103/PhysRevLett.129.150502 | https://arxiv.org/abs/2112.09853 | yes | abs + full (ar5iv) | Q1, Q2: decay model of mirror-style (forward then inverse) circuits |
| Vigneau2025 | Quantum error detection in qubit-resonator star architecture | F. Vigneau, S. Majumder, A. Rath, P. Parrado-Rodríguez, F. R. F. Pereira, ..., J. Heinsoo (17 authors) | 2025 | arXiv:2503.12869; related DOI 10.1103/8m33-wn4g (journal not named on arXiv page) | https://arxiv.org/abs/2503.12869 | yes | abs + full (arXiv HTML) | Q3: IQM hardware, postselected [[4,2,2]] cycles |
| Chen2021 | Exponential suppression of bit or phase flip errors with repetitive error correction | Z. Chen, K. J. Satzinger, J. Atalaya, A. N. Korotkov, A. Dunsworth, ..., J. Kelly | 2021 | arXiv:2102.06132; DOI 10.1038/s41586-021-03588-y (Nature 595, 383-387) | https://arxiv.org/abs/2102.06132 | yes | abs + full (ar5iv, d=2 surface code section) | Q3: d=2 detection-only on superconducting hardware |
| Lin2026 | Surface code logical operations on a superconducting quantum processor | W. Lin, S. Guo, Y. Ma, Z. Yi, K. Zhang, ..., J.-W. Pan (147 authors) | 2026 | arXiv:2607.01473 | https://arxiv.org/abs/2607.01473 | yes | abs + full (HTML incl. App. E.1) | Q1, Q3: case for unconditional, decoded benchmarks |
| Paetznick2024 | Demonstration of logical qubits and repeated error correction with better-than-physical error rates | A. Paetznick, M. P. da Silva, C. Ryan-Anderson, J. M. Bello-Rivas, J. P. Campora III, ..., K. M. Svore (33 authors) | 2024 | arXiv:2404.02280 | https://arxiv.org/abs/2404.02280 | yes | abs + full (HTML v3, Tables 1 to 3) | Q1, Q3: correction vs detection on the same trapped-ion hardware |
| Reichardt2024 | Fault-tolerant quantum computation with a neutral atom processor | B. W. Reichardt, A. Paetznick, D. Aasen, I. Basov, J. M. Bello-Rivas, ..., B. J. Bloom (72 authors) | 2024 (v3 2025) | arXiv:2411.11822 | https://arxiv.org/abs/2411.11822 | yes | abs + full (HTML v3) | Q3: neutral-atom random logical circuits with detection between steps |
| Bluvstein2023 | Logical quantum processor based on reconfigurable atom arrays | D. Bluvstein, S. J. Evered, A. A. Geim, S. H. Li, H. Zhou, ..., M. D. Lukin | 2023 | arXiv:2312.03982; DOI 10.1038/s41586-023-06927-3 | https://arxiv.org/abs/2312.03982 | yes | abs + full (HTML, first ~57%) | Q1, Q3: sliding-scale detection/correction |
| Smith2024 | Mitigating errors in logical qubits | S. C. Smith, B. J. Brown, S. D. Bartlett | 2024 | arXiv:2405.03766; DOI 10.1038/s42005-024-01883-4 (Commun. Phys. 7, 386) | https://arxiv.org/abs/2405.03766 | yes | abs | Q1: exclusive (abort-capable) decoders |
| Gottesman2016 | Quantum fault tolerance in small experiments | D. Gottesman | 2016 | arXiv:1610.03507 | https://arxiv.org/abs/1610.03507 | yes | abs + full (ar5iv) | Q1, Q3: fair accounting for postselected d=2 codes |
| Reinhold2020 | Error-corrected gates on an encoded qubit | P. Reinhold, S. Rosenblum, W.-L. Ma, L. Frunzio, L. Jiang, R. J. Schoelkopf | 2019 (arXiv), 2020 (Nat. Phys.) | arXiv:1907.12327; DOI 10.1038/s41567-020-0931-8 | https://arxiv.org/abs/1907.12327 | yes | abs + full (ar5iv) | Q1: interleaved logical RB with vs without correction |
| Levine2024 | Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons | H. Levine, A. Haim, J. S. C. Hung, N. Alidoust, M. Kalaee, ..., O. Painter | 2023 (arXiv), 2024 (PRX) | arXiv:2307.08737; DOI 10.1103/PhysRevX.14.011051 | https://arxiv.org/abs/2307.08737 | yes | abs + full (ar5iv, RB section and App. F) | Q1, Q2: two-decay analysis of postselected RB |
| Liu2026 | Hardware-Efficient Erasure Qubits With Superconducting Transmon Qutrits | B.-J. Liu, Y.-Y. Wang, Y.-X. Wang, M. Badbaria, S. Puri, C. Wang | 2026 | arXiv:2604.08672 | https://arxiv.org/abs/2604.08672 | yes | abs + full (arXiv HTML, RB section) | Q2, Q3: postselected RB with periodic checks |

### 1b. Context and low-relevance sources (verified seeds and qudit RB baseline)

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched | Read | Relevance |
|---|---|---|---|---|---|---|---|---|
| Morvan2021 | Qutrit randomized benchmarking | A. Morvan, V. V. Ramasesh, M. S. Blok, J. M. Kreikebaum, K. O'Brien, ..., I. Siddiqi | 2020 (arXiv), 2021 (PRL) | arXiv:2008.09134; DOI 10.1103/PhysRevLett.126.210504 | https://arxiv.org/abs/2008.09134 | yes | abs | qudit RB baseline (physical only) |
| RyanAnderson2021 | Realization of real-time fault-tolerant quantum error correction | C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, A. Hankin, ..., R. P. Stutz | 2021 | arXiv:2107.07505 (no journal ref on arXiv page) | https://arxiv.org/abs/2107.07505 | yes | abs | Q3: real-time decoding, Pauli-frame correction. No RB |
| RyanAnderson2022 | Implementing Fault-tolerant Entangling Gates on the Five-qubit Code and the Color Code | C. Ryan-Anderson, N. C. Brown, M. S. Allman, B. Arkin, G. Asa-Attuah, ..., D. Hayes | 2022 | arXiv:2208.01863 | https://arxiv.org/abs/2208.01863 | yes | abs | low: logical CNOTs, no RB |
| Postler2022 | Demonstration of fault-tolerant universal quantum gate operations | L. Postler, S. Heußen, I. Pogorelov, M. Rispler, T. Feldker, ..., T. Monz | 2021 (arXiv), 2022 (Nature) | arXiv:2111.12654; DOI 10.1038/s41586-022-04721-1 (Nature 605, 675-680) | https://arxiv.org/abs/2111.12654 | yes | abs + full (ar5iv) | low: flag-based postselected preparation. RB on physical gates only |

---

## 2. Key findings (all **from literature**; one sentence each)

### A. What LRB measures, and how the decay maps to error per round

1. **[from literature]** (Q1, Q2) LRB applies random logical Cliffords, each followed by a round of error correction (with or without recovery) and an ancilla refresh, and then fits F_L(m) = A_L p_L^m + B_L, where p_L = (d_L F - 1)/(d_L - 1) depends only on the logical dimension d_L and on the average fidelity F of the effective logical channel per Clifford, which includes noisy syndrome measurement, recovery and ancilla preparation. [Combes2017]
2. **[from literature]** (Q1, Q2) The reduction of LRB to standard RB assumes time-independent, Markovian, gate-independent noise at the logical-Clifford level, syndrome-measurement errors that are uncorrelated between rounds, and syndromes that are "refreshed during rounds of error correction". [Combes2017]
3. **[from literature]** (Q1, Q3) For stabilizer codes the Pauli recovery "can be done in post-processing", so no recovery map has to be fixed at experiment time, and comparing runs with identity recovery (which estimate the no-error probability Pr(No)) against runs with recovery separates correctable from uncorrectable error probabilities. [Combes2017]
4. **[from literature]** (Q1, Q3) In a distance-3 superconducting color code, LRB with one QEC cycle between transversal Cliffords (m = 0 to 10, 25 sequences of 20,000 repetitions each, neural-network decoding) was fitted to A p^m + 1/2 and compared with an idling reference via eps_C = (1/2)(1 - p_C/p_I), giving 0.0027(3) added error per Clifford against 0.0171(3) for an idle d=3 cycle (arXiv v1 values). [Lacroix2025]
5. **[from literature]** (Q1) Because transversal H and S change which stabilizer measurements must be compared across cycles, the LRB decoder had to be given the applied logical gates (the authors' Chromobius decoder "cannot be used to decode" their LRB circuits), and the authors preferred LRB to logical process tomography because tomography is sensitive to logical SPAM. [Lacroix2025]
6. **[from literature]** (Q3) On Quantinuum H1-1 and H2-1, two-qubit logical RB of the transversal Steane-code CNOT (L in {2, 6, 10, 14}, 10 circuits of 100 shots each, fit p(L) = a f^L + 1/4) gave logical CNOT fidelities of 0.9991(2) and 0.9980(8), with no QEC rounds between logical Cliffords described and postselection only on repeat-until-success initialization (at most a few percent of shots). [Mayer2024]
7. **[from literature]** (Q1, Q2) In mirror circuits (random Clifford layers followed later by their inverses, dressed with random Pauli layers), the effective polarization decays as A p^d with each forward/inverse pair contributing about p^2, so r = (4^n - 1)(1 - p)/4^n estimates the mean infidelity of a Pauli-dressed layer, assuming Markovian errors that the Pauli layers twirl into stochastic Pauli noise. [Proctor2022]
8. **[from literature]** (Q3) IQM's star-architecture [[4,2,2]] experiment defined the logical error per cycle through <X_L>_{N+1} = (1 - 2 eps) <X_L>_N, i.e. eps = (1 - e^{-b})/2 from a fit a e^{-bN} weighted by the surviving fraction, and reported 0.25% to 0.91% per cycle. [Vigneau2025]

### B. Postselection and the RB decay model

9. **[from literature]** (Q1, Q2) Logical RB of the [[4,2,2]] code on IBM hardware, with terminal parity postselection and "Real RB" over a 576-element group, gave 0.60(3)% two-qubit logical infidelity against 5.8(2)% physical, and the authors stated that "there is at present no complete theory for fitting an RB decay curve with post-selection". [HarperFlammia2019]
10. **[from literature]** (Q2) The detection-only LRB-D variant (one terminal syndrome extraction, postselect all-zero syndromes) on the qutrit [[5,1,2]]_3 code decayed much faster than an unprotected physical qutrit, and the authors warn that terminal detection can yield decay curves with non-Markovian structure that one exponential does not summarise well, proposing interval-check postselection as future work. [Kabir2025]
11. **[from literature]** (Q1, Q2) In leakage RB, the postselected survival is the ratio of computational-subspace survival to the retention (acceptance) rate and reduces to a single exponential plus offset only in particular regimes (for example no seepage), it gets noisy at long sequences as retention falls, and earlier heuristic postselection analyses have an "applicability [that] is unknown for general cases". [ChenBaldwin2025]
12. **[from literature]** (Q1, Q2) Dual-rail erasure RB with 11 evenly spaced erasure checks fitted the postselection probability to an exponential decaying to 0 (4.38(3)e-3 erasure per Clifford) separately from the postselected survival, fitted to an exponential with offset 1/2 (1.01(1)e-4 residual per Clifford), and postselecting only on mid-circuit checks left a trace that "does not decay to a known offset". [Levine2024]
13. **[from literature]** (Q2, Q3) Transmon-qutrit erasure qubits ran RB with an erasure check every 29 Cliffords, and postselecting on no erasure lowered the error per Clifford from 5.3e-3 to 7.6e-4. [Liu2026]
14. **[from literature]** (Q1, Q3) A 4-qubit distance-2 code "can only improve error rates via post-selection", and a fair test must compare the conditional logical error with the unencoded error for every circuit in the family; on that accounting the encoded advantage appears only at low physical error (below about 1.6% for one gate and about 0.12% for 100 gates in Gottesman's estimate). [Gottesman2016]
15. **[from literature]** (Q1, Q3) Lin et al. argue that for algorithmic logical operations "the scalable benchmark is instead the unconditional decoded performance", because conditioning on zero detection events makes acceptance fall with spacetime volume and yields fidelities that are not composable, and in their d=3 surface code such conditioning gave H and S fidelities of 0.9952 and 0.9998 at 1.43% and 1.44% retention, against unconditional values of 0.854 and 0.846. [Lin2026]

### C. Correction, detection, and hybrids

16. **[from literature]** (Q1, Q3) On trapped ions, the Steane-code Bell-state error fell from 0.50% physical to 0.05% with correction (gain 9.8, about 75% acceptance) and to 0.001% with detection (gain 500, about 72% acceptance), and for the distance-4 [[12,2,4]] code the authors note that "the post-rejection rate is second order in the physical error rate". [Paetznick2024]
17. **[from literature]** (Q1) Interleaved logical RB on a bosonic encoded qubit (fit A e^{-gamma n} + 1/2, gate error gamma_IRB - gamma_RB) gave 2.4(1)% error for the error-corrected S gate against 4.6(1)% uncorrected, so interleaved LRB can isolate what correction does to the per-gate logical error. [Reinhold2020]
18. **[from literature]** (Q1, Q3) "Sliding-scale" error detection discards shots by thresholding a decoder-derived weight, giving "a continuous trade-off" between fidelity and acceptance (GHZ fidelity 72% raw, about 90% after discarding about half the data, about 99.85% with full postselection), and corrections are applied to accepted shots "to return to the codespace". [Bluvstein2023]
19. **[from literature]** (Q1) "Exclusive" decoders that abort on hard decoding instances improve the logical failure rate by up to a quadratic factor below threshold, with an exclusion rate that decreases with code distance at low physical error. [Smith2024]

### D. Non-Markovianity in LRB

20. **[from literature]** (Q2) Gate- and time-independent noise can make LRB oscillate instead of decay when syndrome qubits act as a hidden memory of the gate history, coupled to the logical qubits through the decoder; the full text (read through the alphaxiv mirror) builds its main example on a failed syndrome reset, shows the oscillation amplitude decaying as reset improves, and calls the examples "somewhat contrived". [Ceasura2022]
21. **[from literature]** (Q2) Markovian physical operations can give non-Markovian logical dynamics if and only if the physical qubits are "not necessarily returned to the code subspace after every round of QEC", with the syndrome qubits carrying the memory, and the paper proposes sufficient conditions for trusting gate-based characterization such as GST. [Ziyad2025]

### E. Hardware

22. **[from literature]** (Q3) On Sycamore, a distance-2 surface code that discarded any run with a detection event lost 27% of runs per round and kept about 2e-3 logical error per round from undetectable double faults, both matching a depolarizing model. [Chen2021]
23. **[from literature]** (Q3) The IQM [[4,2,2]] experiment postselected on no stabilizer detection with an estimated per-cycle survival P_S of about 0.67 (about 33% of runs discarded per cycle), and reported logical state fidelities above 96% for every cardinal state. [Vigneau2025]
24. **[from literature]** (Q3) On a Yb neutral-atom processor, [[4,2,2]] random circuits drawn from a 36-element logical group, with error detection between consecutive elements, outperformed physical baselines, and correcting more lost atoms raised acceptance but also raised logical error. [Reichardt2024]
25. **[from literature]** (Q3) Trapped-ion hardware has run repeated QEC rounds with a real-time decoder, applying corrections either as software Pauli-frame updates or as physical gates. [RyanAnderson2021]

### F. Qudit LRB

26. **[from literature]** (Q3) Apart from the simulated qutrit LRB-D in Kabir2025, my searches found no qudit logical RB, while physical qutrit RB has been demonstrated on a superconducting processor with single-qutrit infidelity as low as 2.38e-3. [Kabir2025, Morvan2021]

---

## 3. Implications for our qubit [[5,1,2]] LRB study

1. **[conjecture]** (Q2) In LRB-D with const=k, most forward syndromes are ignored and nothing returns the data to the code space. Accumulated data errors then play the role of the hidden syndrome memory in Ceasura2022 and Ziyad2025, which fails Combes2017's "syndromes refreshed" condition. I expect qubit LRB-D(const=1) to be non-exponential, like the qutrit run (Kabir2025, and CONTEXT.md's production CSVs).
2. **[verified analytically + conjecture]** (Q2) The qubit code has four independent checks, so 2^4 = 16 syndrome values (the qutrit code has 3^4 = 81, matching the observed rejection plateau 1 - 1/81). If accumulated errors become uniform at large m, acceptance under const=1 should flatten near 1/16 (rejection about 0.9375) while F tends to 0. The counting is verified analytically; whether the errors actually become uniform is conjecture.
3. **[conjecture]** (Q1, Q2) A decoder that returns the data to the code space every round meets the Combes2017 and Ziyad2025 conditions except for misfires caused by measurement errors, so it should give near-single-exponential decays. For a d=2 code, though, the logical error per round stays first order in p because pair-ambiguous single faults are guessed. Detection-only instead gives second-order conditional error at the price of first-order rejection per round (Gottesman2016, Chen2021). All gates are Clifford, so that correction can be a Pauli frame applied offline (Combes2017), with no real-time feed-forward.
4. **[conjecture]** (Q1) SI1000 puts X_ERROR(5p) before every measurement. A per-round "syndrome = error" decoder would then inject data errors on measurement flips, so a detector-based (space-time) decoder over the whole circuit record is the natural choice. With 5 data qubits and 4 checks, an exact maximum-likelihood or lookup decoder over the detector error model should be cheap.
5. **[verified numerically]** (Q1) A stim 1.15 tableau check on the CONTEXT.md gate implementations shows the following. Logical H = SWAP(2,3) then H^{x5} maps S_Z1<->S_X1, S_Z2<->S_X2 and X_L<->Z_L. Logical S = S{0,4}, S_dag{1}, CZ(2,3) fixes S_Z1, S_Z2 and Z_L, maps S_X1 -> +S_Z1*S_X1 and S_X2 -> +S_Z2*S_X2, and maps X_L -> Y_L. **[conjecture]** As in Lacroix2025, detectors that span a logical gate must be built from these transformed checks. After S, a detector compares an X-check with the product of an X-check and a Z-check, which may create hyperedges in the DEM, so plain matching may not be enough and BP-OSD or exact ML may be needed.
6. **[conjecture]** (Q1, Q2) Because F = |<X_L>| has no offset, f = 1 - 2 eps_block, the same convention as Vigneau2025 and p_L = 2F - 1 in Combes2017. In our mirror-style circuit, each unit of m holds a forward Clifford, a full check round and an inverse Clifford with no check (Proctor2022's forward/inverse pairing), so eps_block is not the error of the check round alone. Isolating the check-round cost needs a reference, for example the same circuits with identity Cliffords, combined as eps = (1/2)(1 - f_C/f_I) (Lacroix2025).
7. **[conjecture]** (Q1, Q3) Report acceptance A(m) and conditional F(m) together, plus one unconditional score. A natural one is A(m) * F_cond(m), which equals <X_L> with rejected shots scored 0 (a random guess); it can be compared directly with a correction-based F. This follows Gottesman2016's accounting and Lin2026's argument. For unif=1, the erasure-RB template of fitting the acceptance decay and the conditional decay separately (Levine2024) is a usable starting point, but ChenBaldwin2025 warns that the ratio is single-exponential only in special regimes.
8. **[conjecture]** (Q1) A hybrid policy is the d=2 analogue of Bluvstein2023's sliding-scale detection and Smith2024's exclusive decoders. It would correct syndromes that have a unique weight-1 explanation (qubit-1 errors in the CONTEXT.md table, which is still to be verified) and reject ambiguous ones. It should interpolate between LRB-D and full correction in both acceptance and F.
9. **[conjecture]** (Q2) The unif=k policies resemble periodic-check postselected RB (Liu2026 checks every 29 Cliffords). The difference is that heralded erasures are caught at the next check, whereas in our d=2 code two errors can cancel each other's syndrome between checks. Larger k should therefore give more room for Ceasura-type memory effects and less exponential decays.
10. **[conjecture, arithmetic verified]** (Q3) With per-round discard of 27% (Chen2021) or about 33% per cycle (Vigneau2025), acceptance after 22 checked rounds is about 0.73^22 = 9.8e-4 or 0.67^22 = 1.5e-4. That makes unif=1 out to m=22 impractical on current superconducting hardware, IQM included. Offline-decoded correction, or sparse-check or hybrid policies, look like the sustainable route.
11. **[conjecture]** (Q3) Trapped ions already combine real-time Pauli-frame decoding (RyanAnderson2021) with selective postselection whose post-correction rejection is second order for d >= 3 (Paetznick2024). Neutral atoms add loss/erasure flags that can be corrected or postselected (Reichardt2024). These platforms suit a correction-vs-detection LRB comparison, but for a d=2 code the detection-only acceptance still decays exponentially in the number of checked rounds.

---

## 4. Seed corrections

- All six seed arXiv IDs resolve to the claimed papers. Titles and authors were confirmed on the arXiv abstract pages:
  - arXiv:1702.03688: "Logical Randomized Benchmarking", Combes, Granade, Ferrie, Flammia (2017). Correct. No journal reference is listed on arXiv.
  - arXiv:1806.02359: "Fault-Tolerant Logical Gates in the IBM Quantum Experience", Harper and Flammia. Correct. Published as PRL 122, 080504 (2019), DOI 10.1103/PhysRevLett.122.080504. The RB uses Real RB over a 576-element subgroup, not the full two-qubit Clifford group.
  - arXiv:2111.12654: "Demonstration of fault-tolerant universal quantum gate operations", Postler et al. Correct. Nature 605, 675-680 (2022). This is not an LRB paper: RB appears only for physical single-qubit gates, and its postselection is on flags during FT preparation (78.9% acceptance for FT |0>_L, 13.7% for the FT magic state). Moved to context.
  - arXiv:2312.03982: "Logical quantum processor based on reconfigurable atom arrays", Bluvstein et al. Correct. DOI 10.1038/s41586-023-06927-3. It uses no RB; the relevant content is error detection and sliding-scale postselection with XEB and GHZ figures of merit.
  - arXiv:2107.07505: title is "Realization of real-time fault-tolerant quantum error correction" (Ryan-Anderson et al., 2021). Correct ID. It contains no RB at the logical level. Moved to context.
  - arXiv:2208.01863: title is "Implementing Fault-tolerant Entangling Gates on the Five-qubit Code and the Color Code" (Ryan-Anderson et al., 2022). Correct ID. It contains no RB. Low relevance, moved to context.
  - arXiv:2404.08616: "Benchmarking logical three-qubit quantum Fourier transform encoded in the Steane code on a trapped-ion quantum computer", Mayer et al. (2024). Correct. Its logical RB covers the transversal CNOT only, with no QEC rounds between logical Cliffords described.
- Non-seed correction to my own recollection: the trapped-ion "logical qubits with better-than-physical error rates" paper (arXiv:2404.02280) has Paetznick, not da Silva, as first author.

---

## 5. Open questions

1. Is there now a complete theory for fitting RB decays under postselection? HarperFlammia2019 said there was none. ChenBaldwin2025 covers leakage-type postselection and Levine2024 uses two separate fits, but I found no treatment of postselection on delayed (terminal or periodic) stabilizer checks. Can LRB-D(unif=k) be written as a ratio of two sums of exponentials over the joint (syndrome x logical) Pauli space and fitted stably?
2. Does per-round correction with noisy syndromes (measurement flips at 5p) leave residual non-Markovianity under Ziyad2025's iff-condition, and how large is it at m <= 22?
3. Does the decoder type (per-round, global space-time, exact ML) change only the decay rate, or also the shape of the decay?
4. Which unconditional score should headline a d=2 comparison: A * F_cond, F with rejected shots counted as failures, or Lin2026's unconditional decoded fidelity?
5. What is the right reference sequence for isolating the per-round check cost when the inverse half has no checks (Lacroix2025's idle reference has checks throughout)?
6. Does realistic LRB-D (data errors never reset, only ancillas reset) meet the representation-with-multiplicity condition that Ceasura2022 identifies, and does that explain the qutrit const=1 curve quantitatively?
7. Interleaved LRB of the individual logical gates of this d=2 code (fold-transversal H against S with CZ(2,3)) under correction vs detection appears in none of the sources I found.
8. Are there experimental qudit LRB demonstrations? I found none as of 2026-10-28; only the simulation in Kabir2025.

---

## Appendix: search log (WebSearch queries run)

1. logical randomized benchmarking error detecting code postselection decay
2. "logical randomized benchmarking" arXiv 2024 OR 2025
3. randomized benchmarking transversal Clifford gates color code superconducting logical error per Clifford
4. interleaved logical randomized benchmarking encoded qubit experiment
5. "Logical randomized benchmarking" "logical-qubit-level reference" interleaved 2025
6. qudit logical randomized benchmarking qutrit error correcting code randomized benchmarking encoded
7. color code logical randomized benchmarking neural network decoder interleaved superconducting processor 2025 Zuchongzhi lattice surgery
8. neutral atom logical qubit randomized benchmarking logical gates error detection [[4,2,2]] OR [[8,3,2]] 2024 2025
9. trapped ion logical randomized benchmarking Steane code H2 Quantinuum logical two-qubit gate RB error correction rounds between gates
10. erasure-conditioned randomized benchmarking postselection dual-rail qubit decay of erasure probability and residual error
11. theory of randomized benchmarking with post-selection heralded errors non-trace-preserving conditional channel fidelity decay
12. distance-2 code postselected error-detecting regime accepted runs quadratic suppression logical error rate randomized benchmarking
13. Ceasura Iyer Wallman Pashayan "Non-exponential behaviour in logical randomized benchmarking" syndrome qubits decoder
14. emergent non-Markovianity logical qubit dynamics quantum error correction Markovian physical noise arXiv 2025
15. superconducting error-detecting code logical gates randomized benchmarking postselection surface code distance-2 logical Clifford fidelity 2023 2024 2025
16. tesseract code trapped ion error detection versus error correction rounds logical error discard rate Reichardt 2024
17. "randomized benchmarking" logical qubit "error detection" postselected acceptance rate decays exponentially sequence length encoded Clifford
18. logical randomized benchmarking neutral atom reconfigurable array transversal Clifford gates QEC rounds 2025 2026
19. "Improved quantum processor logical error rates via correction and detection" arXiv
20. "A fault-tolerant neutral-atom architecture for universal quantum computation" Bluvstein arXiv logical randomized benchmarking
21. qutrit encoded logical qubit randomized benchmarking experiment error detection qudit code postselection superconducting transmon 2024 2025
22. randomized benchmarking heralded erasure errors theory protocol separate erasure rate and residual Pauli error postselected decay 2024 2025 arXiv

Leads I saw but did not fetch, so they are not cited: arXiv:2409.04628 (tesseract code, abstract fetched but no detection-vs-correction detail found); arXiv:2512.24744 (interleaved RB under coherent errors); arXiv:1801.04042 (RB with restricted gate sets); arXiv:2604.16292 (dual-rail checks every 5 Cliffords); arXiv:2601.21616 (biased-erasure cavity qubit RB).
