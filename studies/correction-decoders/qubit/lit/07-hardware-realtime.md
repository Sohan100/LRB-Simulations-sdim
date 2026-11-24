# Lit review 07: hardware (mid-circuit measurement, reset, Pauli frames, real-time decoding latency)

Study: correction vs detection for LRB on the qubit `[[5,1,2]]` code (see `../CONTEXT.md`).
Author: literature subagent 07. Date: 2026-10-28. No code was run except a
two-line arithmetic check, and nothing outside this file was written.

Labels used (per CONTEXT rules): **from literature** (with citation key),
**verified analytically**, **verified numerically**, **conjecture**.
Every source in the table below was fetched in this session (arXiv abstract
page, arXiv HTML/ar5iv full text, PDF text, Crossref record, or vendor page).

## 0. Bottom line

- **Conjecture (built on Knill2005, Terhal2015, Chamberland2018, Battistel2023):**
  LRB with correction (LRB-C) needs no real-time decoder, no feed-forward and no
  physical correction. Every LRB circuit is Clifford and ends in a Pauli readout,
  so the decoder output is a Pauli frame that only flips the recorded logical
  bit, and it can be computed offline. Decoder latency is irrelevant and only
  accuracy matters.
- **Conjecture:** the only hardware requirement is repeated mid-circuit measurement
  (MCM) of the four ancillas, plus either a fast unconditional reset or no reset
  with XOR bookkeeping. IQM supports MCM. Its reset is a measurement followed by a
  conditional pi pulse, available only where classically controlled gates are
  enabled (IQMdocs, AWSBraket2025).
- **Conjecture:** the repo noise model gives every ancilla measurement its own layer.
  That puts 5 x DEPOLARIZE1(2p) per round on each data qubit, where
  simultaneous multiplexed readout (SI1000 as defined in Gidney2021) would put 1
  to 2 layers. Measured MCM spectator errors on IBM, however, exceed
  calibration-based predictions by 4 to 26x (Hothem2025), so this should be
  swept as a sensitivity axis rather than "fixed".
- **Conjecture:** shot budgets on real hardware (seconds per shot on ions) make deep
  LRB-D with acceptance near 1e-6 unmeasurable. LRB-C uses every shot.
- **Conjecture (Q2):** hardware adds non-Markovian ingredients that SI1000 lacks:
  leakage, rare error bursts, coherent measurement-induced phases, atom loss and
  transport dephasing. These could produce non-exponential LRB decays on their
  own, independent of the postselection mechanism.

## 1. Bibliography

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched |
|---|---|---|---|---|---|---|
| Knill2005 | Quantum computing with realistically noisy devices (arXiv title: "Quantum Computing with Very Noisy Devices") | E. Knill | 2005 (arXiv 2004) | arXiv:quant-ph/0410199; DOI 10.1038/nature03350 (Nature 434, 39-44) | https://arxiv.org/abs/quant-ph/0410199 | yes (arXiv abs + full PDF text; Nature metadata via Crossref) |
| Terhal2015 | Quantum Error Correction for Quantum Memories | B. M. Terhal | 2015 (arXiv 2013) | arXiv:1302.3428; DOI 10.1103/RevModPhys.87.307 (RMP 87, 307) | https://arxiv.org/abs/1302.3428 | yes (abs + ar5iv full text, Sec. II.7.3) |
| Chamberland2018 | Fault-Tolerant Quantum Computing in the Pauli or Clifford Frame with Slow Error Diagnostics | C. Chamberland, P. Iyer, D. Poulin | 2018 (arXiv 2017) | arXiv:1704.06662; DOI 10.22331/q-2018-01-04-43 (Quantum 2, 43) | https://arxiv.org/abs/1704.06662 | yes (abs + ar5iv full text) |
| Battistel2023 | Real-Time Decoding for Fault-Tolerant Quantum Computing: Progress, Challenges and Outlook | F. Battistel, C. Chamberland, K. Johar, R. W. J. Overwater, F. Sebastiano, L. Skoric, Y. Ueno, M. Usman | 2023 | arXiv:2303.00054; DOI 10.1088/2399-1984/aceba6 (Nano Futures 7, 032003) | https://arxiv.org/abs/2303.00054 | yes (abs + ar5iv full text) |
| Google2025 | Quantum error correction below the surface code threshold | R. Acharya, L. Aghababaie-Beni, I. Aleiner, T. I. Andersen, M. Ansmann et al. (Google Quantum AI and Collaborators) | 2025 (arXiv 2024) | arXiv:2408.13687; DOI 10.1038/s41586-024-08449-y (Nature 638, 920-926) | https://arxiv.org/abs/2408.13687 | yes (abs + arXiv HTML incl. supplement) |
| Caune2026 | Demonstrating real-time and low-latency quantum error correction with superconducting qubits | L. Caune, L. Skoric, N. S. Blunt, A. Ruban, J. McDaniel et al. (25 authors; Riverlane + Rigetti) | 2026 (arXiv 2024) | arXiv:2410.05202; DOI 10.1038/s41467-026-73331-6 (Nat. Commun. 17, 7383) | https://arxiv.org/abs/2410.05202 | yes (abs + arXiv HTML; journal via Crossref) |
| Barber2025 | A real-time, scalable, fast and resource-efficient decoder for a quantum computer (arXiv title: "...fast and highly resource efficient decoder...") | B. Barber, K. M. Barnes, T. Bialas, O. Bugdayci, E. T. Campbell, N. I. Gillespie, K. Johar, R. Rajan, A. W. Richardson, L. Skoric, C. Topal, M. L. Turner, A. B. Ziad | 2025 (arXiv 2023) | arXiv:2309.05558; DOI 10.1038/s41928-024-01319-5 (Nat. Electron. 8, 84-91) | https://arxiv.org/abs/2309.05558 | yes (abs; journal via Crossref) |
| Yang2026 | Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder | X. Yang, X. Sun, Z. Wu, J. Zhang, J. Jiang, X. Linpeng, Y. Zhou, J. Chu, J. Niu, Y. Zhong, S. Liu, D. Yu | 2026 | arXiv:2605.04892 (preprint) | https://arxiv.org/abs/2605.04892 | yes (abs only) |
| Corcoles2021 | Exploiting dynamic quantum circuits in a quantum algorithm with superconducting qubits | A. D. Corcoles, M. Takita, K. Inoue, S. Lekuch, Z. K. Minev, J. M. Chow, J. M. Gambetta | 2021 | arXiv:2102.01682; DOI 10.1103/PhysRevLett.127.100501 (PRL 127, 100501) | https://arxiv.org/abs/2102.01682 | yes (abs + ar5iv full text) |
| Govia2023 | A randomized benchmarking suite for mid-circuit measurements | L. C. G. Govia, P. Jurcevic, C. J. Wood, N. Kanazawa, S. T. Merkel, D. C. McKay | 2023 (arXiv 2022) | arXiv:2207.04836; DOI 10.1088/1367-2630/ad0e19 (NJP 25, 123016) | https://arxiv.org/abs/2207.04836 | yes (abs + arXiv HTML v2) |
| Hothem2025 | Measuring error rates of mid-circuit measurements | D. Hothem, J. Hines, C. Baldwin, D. Gresh, R. Blume-Kohout, T. Proctor | 2025 (arXiv 2024) | arXiv:2410.16706; DOI 10.1038/s41467-025-60923-x (Nat. Commun. 16, 5761) | https://arxiv.org/abs/2410.16706 | yes (abs + arXiv HTML; journal via Crossref) |
| Gidney2021 | A Fault-Tolerant Honeycomb Memory | C. Gidney, M. Newman, A. Fowler, M. Broughton | 2021 | arXiv:2108.10457; DOI 10.22331/q-2021-12-20-605 (Quantum 5, 605) | https://arxiv.org/abs/2108.10457 | yes (abs + ar5iv Tables 1-2) |
| Geher2025 | To reset, or not to reset -- that is the question | G. P. Geher, M. Jastrzebski, E. T. Campbell, O. Crawford | 2025 (arXiv 2024) | arXiv:2408.00758; DOI 10.1038/s41534-025-00998-y (npj Quantum Inf. 11, 39) | https://arxiv.org/abs/2408.00758 | yes (abs + arXiv HTML v2; journal via Crossref) |
| RyanAnderson2021 | Realization of Real-Time Fault-Tolerant Quantum Error Correction | C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, A. Hankin, J. P. Gaebler et al. (17 authors; Honeywell) | 2021 | arXiv:2107.07505; DOI 10.1103/PhysRevX.11.041058 (PRX 11, 041058) | https://arxiv.org/abs/2107.07505 | yes (abs + ar5iv full text; journal via Crossref) |
| Moses2023 | A Race Track Trapped-Ion Quantum Processor | S. A. Moses, C. H. Baldwin, M. S. Allman, R. Ancona, L. Ascarrunz et al. (96 authors; Quantinuum) | 2023 | arXiv:2305.03828; DOI 10.1103/PhysRevX.13.041052 (PRX 13, 041052) | https://arxiv.org/abs/2305.03828 | yes (abs + ar5iv full text) |
| Paetznick2024 | Demonstration of logical qubits and repeated error correction with better-than-physical error rates | A. Paetznick, M. P. da Silva, C. Ryan-Anderson, J. M. Bello-Rivas, J. P. Campora III et al. (Microsoft + Quantinuum) | 2024 | arXiv:2404.02280 (no journal ref found) | https://arxiv.org/abs/2404.02280 | yes (abs + arXiv HTML v3) |
| Bluvstein2024 | Logical quantum processor based on reconfigurable atom arrays | D. Bluvstein, S. J. Evered, A. A. Geim, S. H. Li, H. Zhou, T. Manovitz et al. (21 authors) | 2024 (online 2023) | arXiv:2312.03982; DOI 10.1038/s41586-023-06927-3 (Nature 626, 58-65) | https://arxiv.org/abs/2312.03982 | yes (abs + ar5iv full text; journal via Crossref) |
| Muniz2025 | Repeated ancilla reuse for logical computation on a neutral atom quantum computer | J. A. Muniz, D. Crow, H. Kim, J. M. Kindem, W. B. Cairncross, A. Ryou et al. (62 authors; Atom Computing + Microsoft) | 2025 | arXiv:2506.09936; DOI 10.1103/v7ny-fg31 (PRX 15, 041040) | https://arxiv.org/abs/2506.09936 | yes (abs + arXiv HTML; journal via Crossref) |
| Abdurakhimov2024 | Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer | L. Abdurakhimov, J. Adam, H. Ahmad, O. Ahonen, M. Algaba, G. Alonso et al. (113 authors; IQM) | 2024 | arXiv:2408.12433 (no journal ref) | https://arxiv.org/abs/2408.12433 | yes (abs + full PDF text) |
| IQMdocs | IQM client 36.0.0 documentation: "models" API page and "Qiskit on IQM" user guide; iqm.pulse 15.0.0 "Reset_Conditional" | IQM Quantum Computers (vendor documentation) | accessed 2026-10-28 | n/a | https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.html ; https://docs.iqm.tech/iqm-client/user_guide_qiskit.html ; https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.reset.Reset_Conditional.html | yes |
| AWSBraket2025 | Experiment with dynamic circuits on IQM Garnet with Amazon Braket | T. Chen, S. Smart, Z. Mohammad (AWS Quantum Technologies blog) | 2025 (26 Jun) | n/a | https://aws.amazon.com/blogs/quantum-computing/experiment-with-dynamic-circuits-on-iqm-garnet-with-amazon-braket | yes |
| IQMHalocene2025 | IQM launches Halocene, a new quantum computer product line for error correction | IQM Quantum Computers (press release) | 2025 (13 Nov) | n/a | https://www.iqm.tech/?p=14348 | yes |

Fetched but not cited (superseded or off-scope): Rudinger et al., "Characterizing
mid-circuit measurements on a superconducting qubit using gate set tomography",
arXiv:2103.03008, PRApplied 17, 014014 (2022). Gaebler et al., "Suppression of
mid-circuit measurement crosstalk errors with micromotion", arXiv:2108.10932, PRA
104, 062440 (2021); Moses2023 and Hothem2025 already carry its result. Bluvstein
et al., arXiv:2506.20661, published as "A fault-tolerant neutral-atom architecture
for universal quantum computation", Nature 649, 39-46 (2026); only the abstract
was read. Maurer et al., "Real-time decoding of the gross code memory with FPGAs",
arXiv:2510.21600 (IBM Relay-BP, 24 ns per BP iteration). Riesebos et al., "Pauli
Frames for Quantum Computer Architectures", DAC 2017, DOI 10.1145/3061639.3062300:
only the Crossref metadata could be fetched (the ACM page returned 403), so it is
not cited.

## 2. Key findings (all **from literature**)

Format: claim [keys] (question).

### 2.1 Pauli frames: when is feedback really needed?

1. Knill defines each encoded state only relative to a current Pauli frame (a Pauli product that restores the block to the syndrome-0 subspace), which "makes it possible to avoid explicitly applying Pauli products for correction or teleportation compensation" [Knill2005] (Q1, Q3).
2. Knill: Pauli-product corrections "can be absorbed into the Pauli frame and do not need to be known immediately", but a teleported non-Clifford gate may need a non-Pauli compensation, and "in this case, we must wait for the measurement outcomes" [Knill2005] (Q3).
3. Knill: an error is detected when the syndrome differs from what the Pauli frame predicts. Postselected computing then rejects and restarts, and standard computing uses the syndrome to update the frame; he adds that "purely postselected computing has little computational power" [Knill2005] (Q1).
4. Knill's C4/C6 architecture builds correction out of error-detecting codes by concatenation, because "error-detecting codes can always correct an error at a known location" [Knill2005] (Q1, Q3).
5. Terhal: "in principle it is never necessary to physically correct Pauli errors". In an all-Clifford circuit the frame only changes how the final measurement is read, and after a T gate one must wait for the logical frame before the classically controlled S X (or S) correction [Terhal2015, Sec. II.7.3] (Q1, Q3).
6. Terhal: decoding "should never lead to a increasing backlog of syndrome data". If r_gen/r_proc > 1 the delay grows at every T gate (exponential slowdown), whereas a Clifford-only quantum memory never waits for classical processing [Terhal2015, Sec. II.7.3] (Q3).
7. Chamberland et al.: in Clifford-only circuits "all Pauli recovery operators can be tracked in classical software" and "there is no need to actively error-correct". Waiting for the frame before T gates costs only a constant factor if decoding keeps pace, and they generalise the idea to a Clifford frame [Chamberland2018] (Q1, Q3).
8. Battistel et al.: the decoder's correction "is needed only for the next-in-line logical non-Clifford gate" (for example the S correction in magic-state injection); otherwise it is tracked in the Pauli (or leakage) frame "until needed" [Battistel2023] (Q3).
9. On a trapped-ion QCCD device, software Pauli-frame tracking and physically applied corrections before a logical S gave statistically indistinguishable logical fidelities, 92(1)% vs 93(2)% [RyanAnderson2021] (Q3).
10. A 2026 preprint demonstrates mid-circuit feedback in non-Clifford logical circuits on a superconducting processor, presented as the case "where Pauli-frame updates alone are insufficient" [Yang2026] (Q3).

### 2.2 Decoder latency budgets and demonstrated latencies

11. Battistel et al. take ~1 us per QEC round as the benchmark for transmons ("the tightest constraints") and ~1 ms or looser for trapped ions, where 200 ms cycles have been reported. For ions and neutral atoms the QEC cycles are "usually in excess of many milliseconds", which leaves "ample time for real-time decoding" [Battistel2023] (Q3).
12. Battistel et al. rank throughput (decoder compute time) above latency (the time to ship outcomes to the decoder), name the failure to keep up "the backlog problem", and require the readout-to-decoder link to take "at most a couple hundreds of nanoseconds" [Battistel2023] (Q3).
13. Google decoded a d=5 surface-code memory in real time on a workstation (specialised Sparse Blossom correlated matching, fed over low-latency Ethernet). It kept up with the 1.1 us cycle for up to 1e6 cycles with 63 +- 17 us average latency and stayed below threshold: eps_5 = 0.35%, Lambda = 2.0, vs 0.269% and 2.18 offline with a neural decoder. No feedback went into the logical circuit [Google2025] (Q1, Q3).
14. Caune et al. (Riverlane + Rigetti Ankaa-2) ran an FPGA Collision Clustering decoder inside the control system. Mean decoding time per round was 0.44 to 0.79 us for 5 to 25 rounds, below the 1 us data-generation rate. The response from last round to conditional logical gate was 9.6 us for 9 rounds: 6.5 us decoding plus 3.1 us communication and control [Caune2026] (Q3).
15. Barber et al. implement Collision Clustering on FPGA (surface codes up to 881 qubits) and on ASIC (1057 qubits, 0.06 mm^2, 8 mW), at "MHz decoding speed", for surface-code memory experiments [Barber2025] (Q1, Q3).
16. Yang et al. report a 124 ns FPGA neural-network decoder and a 550 ns deterministic closed-loop latency inside a 1.25 us distance-3 surface-code cycle, with real-time decoding comparable to offline [Yang2026] (Q1, Q3).
17. IBM dynamic circuits on FPGA control used a 300 ns readout pulse. Conditional reset (measure, then conditional bit flip), done twice, took 1.4 us in total and left a 0.01 reset error [Corcoles2021] (Q3).

### 2.3 Superconducting MCM, reset and spectator errors; the SI1000 model

18. Google's 1/Lambda error budget attributes ~20% to data-qubit idling during measurement and reset (idle error 0.9e-2), ~11% to readout (0.8e-2) and ~2% to reset (1.5e-3), with CZ errors largest. Dynamical decoupling runs during measurement and reset [Google2025, Suppl. IV.1.3, V.2] (Q3).
19. Google: leakage that survives several cycles creates "time correlations in error detection events that confuse the error decoder" (data-qubit leakage removal raised Lambda by 35% at d=5). Rare correlated bursts about once an hour (~400 us decay, ~30-qubit regions) give an apparent 1e-10 error floor for d >= 15 repetition codes [Google2025] (Q2, Q3).
20. MCM randomized benchmarking on an IBM Falcon device found measurement-induced errors on most unmeasured qubits, for example 1.7e-3 +- 1.0e-3 per MCM for one pair. The authors attribute them to measurement-induced Stark shift or cross-dephasing, find no consistent device-wide pattern, and note measurement times approaching 1 us [Govia2023] (Q2, Q3).
21. On a 27-qubit IBM processor, MCM-induced crosstalk was 4 to 5x above calibration-based predictions (~26x at 15 qubits). Dynamical decoupling removed ~22% (up to 38%), and "the majority of MCM-induced crosstalk errors are not mitigatable in this way, and remain unexplained" [Hothem2025] (Q2, Q3).
22. Caune et al. ran without mid-circuit resets: ancillas were "not reset between rounds" and "resets are not available". Readout was a 964 ns pulse plus 336 ns ring-down, about 1.7 us per QEC round [Caune2026] (Q3).
23. Geher et al.: reset gives no benefit for memory experiments (no-reset is slightly better because data qubits idle less). Unconditional reset can halve the duration of logical operations, no-reset wins when reset takes more than ~100 ns and p > ~10^-2.5, and conditional reset "should not be used" in any QEC experiment they consider [Geher2025] (Q1, Q3).
24. Without reset, each round's syndrome is the XOR of consecutive outcomes and detectors compare outcomes two rounds apart. A classification error then makes a length-2 time-like edge and halves the number of misclassifications the code tolerates [Geher2025] (Q1).
25. SI1000 is defined as CZ(p), 1-qubit Clifford(p/10), InitZ(2p) (bit flip after reset), MZ(5p) (bit flip before measurement), Idle(p/10) on every unused qubit at every time step, and ResonatorIdle(2p) on qubits not measured or reset during a step in which others are. It was chosen "to be representative of ... Google's superconducting quantum chip, where measurement and reset take an order of magnitude longer than other gates" (~1000 ns cycle) [Gidney2021] (Q3).

### 2.4 IQM: what the hardware supports

26. IQM's API allows mid-circuit measurements ("each qubit may be measured multiple times"). `cc_prx` applies a PRX if the referenced result is 1 and "an identity gate of similar time duration" otherwise, and the measurement must precede the conditional gate [IQMdocs] (Q3).
27. IQM's reset is "a (projective) measurement followed by a classically controlled X gate" and exists only where classically controlled gates do. Only x, y, rx, ry and r can be conditioned, on one classical bit, and availability "depends on the instrumentation". `Reset_Conditional` inserts waits for the "feedback result propagation delay" and works only on components with both readout and drive [IQMdocs] (Q3).
28. On Amazon Braket (June 2025), dynamic circuits on IQM Garnet (`measure_ff` + `cc_prx`) are experimental and need verbatim compilation, and "conditional operations can only be applied to qubits in a local group". Active reset is a pi rotation conditioned on the measured bit [AWSBraket2025] (Q3).
29. IQM Garnet (20 qubits) uses frequency-multiplexed dispersive readout with per-qubit Purcell filters on three probe lines. Reported medians: single-qubit error 9e-4, CZ error 5e-3, two-qubit Clifford error 2e-2, simultaneous readout error 3e-2. Relaxation time is of order 40 us (target above 100 us) and pulse control runs on FPGAs. The paper reports no MCM, reset or feedback benchmarks: a text search of the PDF finds no "reset" or "mid-circuit" [Abdurakhimov2024] (Q3).
30. IQM Halocene (announced 13 Nov 2025: 150 qubits, commercially available by end of 2026, 99.7% two-qubit fidelity target) is aimed at "research on up to five logical qubits and the implementation of Clifford gates". It has a modular decoder architecture and NVIDIA NVQLink support [IQMHalocene2025] (Q3).

### 2.5 Trapped ions (Honeywell, now Quantinuum)

31. Repeated [[7,1,3]] color-code QEC used a real-time decoder built from two lookup tables, with corrections applied either as Pauli-frame updates or as physical gates. QEC cycles took under 200 ms, of which transport was under 70 ms and cooling most of the rest, and conditional information took under 250 us to distribute in the worst case [RyanAnderson2021] (Q3).
32. In that experiment the logical-cycle error budget was ~49% unitary gates, ~45% dephasing (memory) and ~6% SPAM/MCMR, even though the per-operation dephasing error (~2.2e-4) was an order of magnitude below the two-qubit rate (3.1e-3) [RyanAnderson2021] (Q2, Q3).
33. On H2, mid-circuit measurement crosstalk is 4.5e-6 and reset crosstalk 3.8e-6 average infidelity on neighbouring ions (Table 2). Stray light causes it, and micromotion hiding mitigates it for unmeasured ions in gate zones but not in conveyor regions. Transport takes ~60% of circuit time, and real-time classical logic (OpenQASM 2.0++, Wasm) is available. The repetition-code benchmark "does not perform mid-circuit real-time decoding" [Moses2023] (Q3).
34. On Quantinuum H1-1, previously unmodelled long-range MCM crosstalk (MCM error 0.63 +- 0.09% vs 0.23 +- 0.09% from the emulator) was traced to stray light and removed by micromotion hiding of auxiliary-zone ions, bringing it to 0.29 +- 0.06% [Hothem2025] (Q2, Q3).

### 2.6 Neutral atoms

35. Harvard/QuEra perform mid-circuit readout by moving atoms ~100 um to a readout zone: 500 us imaging at ~99.8% fidelity, and at most ~0.1% data decoherence per 500 us of imaging. The full cycle of readout, FPGA decoding and conditional Raman feed-forward takes under 1 ms, with ~100 us expected later [Bluvstein2024] (Q3).
36. The same work combines correlated (hypergraph, maximum-likelihood) decoding with software Pauli-frame correction and sliding-scale postselection. Teleportation fidelity was 77(2)% with feed-forward, 75(2)% when postselecting on the middle logical qubit, and 92(2)% after adding stabilizer postselection [Bluvstein2024] (Q1, Q3).
37. Atom Computing ran 41 rounds of repetition-code syndrome extraction with reused ancillas. Each MCM cycle (two ~7 ms images, 4 ms cooling, 0.3 ms optical pumping) cost the unmeasured register 1.06(7)% atom loss and 0.49(7)% contrast loss. Decoding used PyMatching on matching graphs edited per shot for loss, and conditional moves took under 10 ms [Muniz2025] (Q2, Q3).

### 2.7 Detection vs correction on hardware

38. On H2, the [[7,1,3]] Bell-state logical error was 0.05% with correction (pre-selection, ~75% pre-acceptance) and 0.001% with detection (17,389 of 24,200 runs accepted), against a 0.50% physical baseline. For the [[12,2,4]] code the corresponding errors were 0.17% and 0.001%, with ~0.5% post-rejection [Paetznick2024] (Q1, Q3).
39. In repeated [[12,2,4]] rounds the rejection rates grow linearly with the number of repetitions to leading order, and memory and transport errors dominate. The [[7,1,3]] decoder ran on the device during the program, the [[12,2,4]] decoder by table lookup after data collection, and MCM crosstalk was at most 2e-5 [Paetznick2024] (Q1, Q2, Q3).

### 2.8 Platform summary (from literature)

| Platform | Round / cycle time | MCM | Reset | Real-time decoding or feed-forward shown |
|---|---|---|---|---|
| Google (SC) | 1.1 us cycle [Google2025] | readout error 0.8e-2; data idling during M/R ~20% of budget; DD during M/R [Google2025] | multi-level reset on measure qubits (1.5e-3), leakage removal on data [Google2025] | software matching, 63 us latency at d=5, keeps up with 1e6 cycles, no feedback [Google2025] |
| Rigetti Ankaa-2 (SC) | ~1.7 us per round [Caune2026] | 964 ns pulse + 336 ns ring-down [Caune2026] | none ("resets are not available") [Caune2026] | FPGA, 0.44-0.79 us per round; 9.6 us to conditional gate [Caune2026] |
| IBM (SC) | not in sources | ~1 us measurement; spectator error ~1.7e-3 per MCM [Govia2023]; crosstalk 4-26x above prediction [Hothem2025] | conditional reset, 1.4 us for 2 cycles, 0.01 error [Corcoles2021] | FPGA dynamic circuits [Corcoles2021] |
| Unnamed SC (Yang et al.) | 1.25 us | n/a | n/a | 550 ns closed loop, 124 ns NN decode [Yang2026] |
| IQM Garnet (SC) | not reported | multiplexed readout, median simultaneous error 3e-2; T1 ~40 us [Abdurakhimov2024] | measure + conditional X, needs cc_prx [IQMdocs] | single-bit cc_prx; experimental, local group only on Braket [AWSBraket2025]; decoders promised for Halocene (2026) [IQMHalocene2025] |
| Quantinuum H1/H2 (ions) | QEC cycle < 200 ms (H1, 2021) [RyanAnderson2021] | crosstalk 4.5e-6 (H2) [Moses2023]; <= 2e-5 [Paetznick2024] | in-zone reset, crosstalk 3.8e-6 [Moses2023] | lookup-table decoder, < 250 us latency [RyanAnderson2021]; Wasm [Moses2023] |
| Harvard/QuEra (atoms) | moves ~200 us; gates 270 ns [Bluvstein2024] | 500 us imaging in separate zone, <= 0.1% data error [Bluvstein2024] | n/a | readout + decode + feed-forward < 1 ms [Bluvstein2024] |
| Atom Computing (atoms) | MCM cycle ~ 2x7 ms + 4.3 ms [Muniz2025] | 0.49% contrast + 1.06% atom loss per cycle on register [Muniz2025] | cooling + optical pumping [Muniz2025] | conditional moves < 10 ms [Muniz2025] |

## 3. Implications for our [[5,1,2]] qubit LRB study

I1. **Conjecture** (applies findings 1-8) (Q1, Q3). LRB-C needs no
real-time decoding, no feed-forward and no physically applied correction. The
logical H/S/X/Z words, their inverses and the check circuits are all Clifford,
and the run ends in a Pauli (X_L) readout. The decoder's output is therefore a
Pauli frame whose only effect is to flip the recorded logical bit x0 + x3, and it
can be computed after the run from the stored syndromes. Decoder latency places
no constraint, so slow but accurate decoders (exact maximum likelihood over the
whole syndrome history) are acceptable.

I2. **Conjecture** (Q1, Q3). Every circuit physically contains all m + 1 check
rounds whatever the policy (CONTEXT). LRB-D under every `const`/`unif` policy and
LRB-C under every decoder are then just different classical post-processing of the
same recorded outcomes, so one hardware dataset per (sequence, depth) gives all
of them. The experiment must store every raw mid-circuit outcome: 4(m + 1) per
shot, which is 92 at m = 22. On IQM every measurement key must be unique
[IQMdocs].

I3. **Conjecture** (Q3). The minimal hardware requirement is repeated (ideally
simultaneous) MCM of the four ancillas plus either a fast unconditional reset or
no reset. No classically controlled gate is needed. Every platform reviewed
provides this: Google, Rigetti (without reset), IBM, IQM, Quantinuum and both
neutral-atom groups (Section 2.8).

I4. Reset or no reset (Q3).
- **Verified analytically** (two-line argument): without reset, the raw outcome of
  an ancilla in round j is n_j = n_{j-1} XOR s_j for both check types in our
  circuits, where s_j is that round's stabilizer bit. For a Z check the CNOTs
  data->ancilla add the parity to the stored bit. For an X check, H|a> =
  (|0> + (-1)^a |1>)/sqrt2, phase kickback multiplies the |1> term by (-1)^s, and
  the final H gives |a XOR s>. Each round's syndrome bit is therefore recovered by
  XOR, as in [Geher2025].
- **Conjecture:** on IQM the reset is itself a measurement plus a conditional pi pulse
  that needs cc_prx support and a feedback wait [IQMdocs, AWSBraket2025], so
  no-reset is the simpler choice and probably the less noisy one.
- In the repo model, no-reset would remove the reset layer (DEPOLARIZE1(2p) on
  data, X_ERROR(2p) on ancillas). It would also let X-type errors that an ancilla
  picks up between its measurement and its next use persist into its next syndrome
  bit, including errors from the later ancillas' measurement layers in the same
  round. Simulation is needed to see the net effect.

I5. **Verified analytically + conjecture** (Q1). In the repo model every measurement
flip is a pre-measurement X_ERROR(5p), plus DEPOLARIZE1(p), on the measured
ancilla. That is a "quantum" flip: without reset it persists in the ancilla and
corrupts exactly one XOR-recovered syndrome bit, the same as an ordinary
measurement error with reset (verified analytically). Real readout also has
classical misassignments, which corrupt two consecutive bits without reset
[Geher2025]. A faithful no-reset study should therefore split measurement noise
into quantum and classical parts; Geher2025 use 4p and p. LRB-D rejects either
way, but an LRB-C decoder must model the difference (conjecture).

I6. Measurement layering (Q3).
- **Verified by reading the code:** `src/lrb_qubit/code.py` interleaves each `M`
  with the CNOTs of the next check, so `emit_operations` makes every ancilla
  measurement its own layer. Each data qubit then gets five DEPOLARIZE1(2p) per
  round (4 measurement layers + 1 reset layer).
- **Verified analytically:** k such layers give a total Pauli-error probability of
  (3/4)[1 - (1 - 8p/3)^k]. At p = 5e-3 that is 0.0487 for k = 5, 0.0199 for k = 2
  and 0.0100 for k = 1. The chance that at least one of the five data qubits is
  hit by these layers alone is then 0.22, 0.096 and 0.049.
- **Conjecture:** simultaneous multiplexed readout (IQM Garnet [Abdurakhimov2024];
  SI1000 as defined in [Gidney2021]) would give one ResonatorIdle(2p) per
  measurement step and one per reset step. On this term the repo model would
  then be ~2.5x (with reset) to ~5x (no reset) pessimistic.
- Measured MCM spectator errors run 4 to 26x above calibration predictions on IBM
  [Hothem2025], so treat this as a sensitivity sweep rather than a correction.
  The `SI1000.scaled()` ablation hook in `src/lrb_qubit/circuits.py` can
  already do it, for example `meas_idle` x 0.25. The original SI1000 also has
  Idle(p/10) on unused qubits at every step, which the repo omits, and MZ(5p) is
  a pure bit flip with no extra DEPOLARIZE1(p) [Gidney2021]. Both differences
  are verified by reading Gidney2021 Table 1 against CONTEXT.

I7. **Conjecture** (Q3). IQM Garnet's medians (CZ 5e-3, single-qubit 9e-4,
simultaneous readout 3e-2, T1 ~40 us [Abdurakhimov2024]) are close to SI1000 at
p ~ 5e-3 (CZ p, single-qubit 0.1p = 5e-4, readout flip 5p = 2.5e-2). Current
IQM hardware therefore sits near the p = 5e-3 grid point, where the qutrit LRB-D
run already rejected ~27% of shots at depth 0. With 1.1 to 1.7 us per round
[Google2025, Caune2026], m = 22 needs 23 rounds, roughly 25 to 40 us. That is
comparable to Garnet's T1, so idling during readout will likely dominate deep
sequences. IQM's readout duration is not given in the sources read.

I8. **Conjecture** (Q1, Q3). Shot budgets favour correction. On H2, circuits
with tens to hundreds of measurements take 0.18 to 5.4 s per shot [Moses2023,
Table 1], so the grid's 1e6 shots per circuit is out of reach on ions. LRB-D
policies whose acceptance falls to ~1e-6 at depth (as `unif=1` did in the qutrit
run) would return almost no accepted shots on any platform, while LRB-C keeps
every shot. Rejection on hardware also grows about linearly per round
[Paetznick2024].

I9. **Conjecture** (Q2). Hardware brings non-Markovian ingredients that SI1000
lacks:
- time-correlated detection events from leakage [Google2025];
- rare correlated bursts [Google2025];
- coherent measurement-induced phases on spectator qubits [Govia2023];
- unexplained MCM crosstalk [Hothem2025];
- atom loss that accumulates with each MCM [Muniz2025];
- dephasing that scales with transport time [RyanAnderson2021, Moses2023].

Postselection discards shots hit by these events. That raises rejection and may
tilt the accepted ensemble toward quiet periods. Correction keeps those shots and
turns the events into correlated logical errors. A hardware LRB-C vs LRB-D
comparison could therefore show non-exponential decays from these sources,
separate from the protocol-level mechanism studied in simulation. A hardware plan
should include a leakage- or erasure-aware decoder variant and dynamical
decoupling during readout [Google2025, Hothem2025].

I10. **Conjecture** (Q1, Q3). Heralded erasures could make d = 2 correctable.
Knill notes that error-detecting codes correct errors at known locations
[Knill2005]. With heralded atom loss [Muniz2025] or leakage detection, the
[[5,1,2]] code could *correct* a single heralded erasure per round instead of
guessing inside an ambiguous syndrome pair. This is a possible route to useful
LRB-C at distance 2 and is worth a simulation.

I11. **Conjecture** (Q3). SI1000's ResonatorIdle(2p) is a superconducting
term. On trapped ions measurement crosstalk is ~1e-5 or less [Moses2023,
Paetznick2024] and memory and transport dephasing dominate [RyanAnderson2021]. On
neutral atoms each MCM cycle costs the register ~0.5% contrast and ~1% of its
atoms [Muniz2025], or at most ~0.1% with zoned readout [Bluvstein2024].
Conclusions drawn from the SI1000 grid need platform-specific noise variants
before they are applied to other platforms.

I12. **Conjecture** (Q3). If real-time decoding is wanted later, the budgets can be
met. Two possible uses are aborting rejected LRB-D shots early and extending to
non-Clifford gates. FPGA decoders already run under 1 us per round on
superconducting hardware [Caune2026, Barber2025, Yang2026], and ions and atoms
have millisecond budgets [Battistel2023]. IQM's documented conditional control
(single bit, single-qubit rotations, local group) supports active reset and
simple feed-forward. No documented IQM interface puts a decoder in the loop;
Halocene promises one from end of 2026 [IQMdocs, AWSBraket2025,
IQMHalocene2025].

I13. **Conjecture** (Q1). On hardware, detection buys one to two orders of
magnitude in logical error for shallow circuits at a modest acceptance cost
[Paetznick2024, Bluvstein2024]. That is consistent with LRB-D being attractive at
small m and LRB-C at large m. A hybrid policy is a natural candidate: correct the
correctable patterns and reject only the patterns flagged as most likely to be
logical errors. It mirrors Paetznick et al.'s pre-selection plus correction
strategy.

## 4. Seed corrections

| Seed (as given) | Status | Correction |
|---|---|---|
| Knill, "Quantum computing with realistically noisy devices", arXiv:quant-ph/0410199 | Title mismatch | The arXiv title is "Quantum Computing with Very Noisy Devices" (2004). The seed title is that of the Nature version: Nature 434, 39-44 (2005), DOI 10.1038/nature03350. |
| Chamberland, Iyer, Poulin, arXiv:1704.06662 | Correct | Exact title "Fault-Tolerant Quantum Computing in the Pauli or Clifford Frame with Slow Error Diagnostics"; Quantum 2, 43 (2018). |
| Terhal, arXiv:1302.3428 | Correct | Rev. Mod. Phys. 87, 307 (2015). |
| Google Quantum AI, arXiv:2408.13687 | Correct | First author R. Acharya; Nature 638, 920-926 (2025). |
| Caune et al., arXiv:2410.05202 | Correct | Riverlane + Rigetti on Ankaa-2. Published in Nat. Commun. 17, 7383 (2026), DOI 10.1038/s41467-026-73331-6. |
| Barber et al. (ID unknown) | Found | arXiv:2309.05558, "A real-time, scalable, fast and highly resource efficient decoder for a quantum computer". Published as "A real-time, scalable, fast and resource-efficient decoder for a quantum computer", Nature Electronics 8, 84-91 (2025). |
| Corcoles et al., arXiv:2102.01682 | Correct | "Exploiting dynamic quantum circuits in a quantum algorithm with superconducting qubits"; PRL 127, 100501 (2021). |
| Ryan-Anderson et al., arXiv:2107.07505 | Correct | PRX 11, 041058 (2021). The arXiv page lists no journal reference. Hardware was Honeywell's 10-qubit QCCD. |
| Moses et al., arXiv:2305.03828 | Correct | Exact title "A Race Track Trapped-Ion Quantum Processor"; PRX 13, 041052 (2023). |
| IQM documentation or papers | Found | Primary IQM sources: IQM client 36.0.0 docs, iqm.pulse 15.0.0 docs, arXiv:2408.12433 and the Halocene press release. AWS's Braket blog documents Garnet's dynamic circuits. No IQM paper characterising MCM-induced dephasing, reset fidelity or feedback latency was found. |

## 5. Open questions

- What are IQM's readout duration, MCM-induced dephasing on data qubits, reset
  fidelity and cc_prx feedback latency? None appears in the sources found. One
  way to get them is to run MCM RB [Govia2023, Hothem2025] on Garnet or Emerald.
- On which IQM QPUs (Garnet, Emerald, Crystal/Halocene) are cc_prx and reset
  enabled through IQM Resonance, as opposed to Braket? What is the "local group"
  restriction concretely, and is there a limit on MCMs per circuit (we need 92)?
- How much of the hardware MCM spectator error is coherent (Stark-shift phase,
  [Govia2023]) and how much stochastic? Do LRB-C and LRB-D respond differently to
  coherent phases that accumulate over rounds (Q2)?
- In our model, does no-reset with XOR bookkeeping help or hurt LRB-D acceptance
  and LRB-C fidelity, once ancilla errors that persist between rounds and a
  quantum/classical split of readout errors are included (I4, I5)?
- How large are the LRB-C vs LRB-D differences under a parallel-readout noise
  variant (I6), and under a trapped-ion variant without ResonatorIdle but with
  time-proportional dephasing (I11)?
- Can heralded loss or leakage be folded into a d = 2 decoder to give real
  correction (I10)? How do loss-adapted matching graphs [Muniz2025] map onto our
  4-stabilizer syndrome history?
- With simultaneous readout on IQM, are the four ancilla readouts correlated
  (readout crosstalk)? Abdurakhimov2024 says the architecture has low readout
  crosstalk but gives no MCM-level numbers.

## 6. Search log (web searches run in this session)

1. IQM mid-circuit measurement conditional reset feedback cc_prx documentation
2. real-time decoding latency superconducting quantum error correction FPGA 2025 arXiv
3. Barber Riverlane "real-time, scalable, fast and highly resource efficient decoder" arXiv
4. randomized benchmarking mid-circuit measurements crosstalk spectator qubits superconducting arXiv
5. IQM "Technology and performance benchmarks" 20-qubit quantum computer arXiv
6. Pauli frame tracking hardware unit quantum computer architecture Riesebos
7. "Real-time decoding for fault-tolerant quantum computing" progress challenges outlook Battistel
8. Quantinuum real-time decoding logical qubits error detection versus correction 2024 2025 arXiv mid-circuit measurement
9. "Demonstration of logical qubits and repeated error correction with better-than-physical error rates" arXiv
10. neutral atom array mid-circuit measurement qubit reuse real-time feedback logical qubits 2025 arXiv
11. IQM quantum error correction experiment real-time feedback latency Garnet surface code 2025
12. SI1000 noise model superconducting inspired Gidney honeycomb memory measurement idle error
13. qiskit-iqm user guide mid-circuit measurement c_if conditional x reset feedback IQM Resonance
14. IQM Halocene error correction quantum computer 2025 mid-circuit measurement real-time decoding
15. arXiv IQM superconducting qubits stabilizer measurement repeated rounds error detection experiment conditional reset 2025
16. Pauli noise learning mid-circuit measurements superconducting spectator crosstalk Hines Proctor 2024 2025
17. IBM dynamic circuits feed-forward latency mid-circuit measurement idle error dynamical decoupling long-range entanglement Baumer
18. Quantinuum Helios trapped-ion real-time decoding GPU NVIDIA logical qubits arXiv 2025
19. Google Quantum AI real-time decoding feedback lattice surgery logical operation superconducting 2025 2026 arXiv
20. docs.iqm.tech qiskit user guide "if_test" OR "c_if" mid-circuit measurement reset IQM backend
21. IQM Garnet Emerald "classically controlled" gates mid-circuit measurement supported Resonance release notes

Searched but not used: Quantinuum Helios and NVIDIA GPU real-time decoding,
for which only vendor press coverage was found and no primary paper (not
fetched, **UNVERIFIED**). Baumer et al., arXiv:2308.13065, IBM long-range
entanglement with dynamic circuits (search result only, not fetched,
**UNVERIFIED**).
