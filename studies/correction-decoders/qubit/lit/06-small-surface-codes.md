# Lit review 06: distance-2 surface codes, fold-transversal Clifford gates, small-code experiments

Assignment 06 of the qubit-512 study, written 2026-10-28. I fetched every source in the
bibliography myself (arXiv abstract page, plus the full text through arXiv HTML or ar5iv
where a finding needed it). Labels follow CONTEXT.md: **from literature** (with key),
**verified analytically**, **verified numerically**, **conjecture**.

Naming caveat. The Error Correction Zoo calls our code the "[[5,1,2]] rotated surface code"
[ECZ512], while CONTEXT.md and Moussa's family call it the d=2 unrotated (square, planar)
surface code. It is the same code (relabelling checked in section 3, I1).

## 0. Short answers

1. **Is the qubit [[5,1,2]] with fold-transversal H and S known?** Yes.
   - It is the D=2 member of Moussa's folded "square" family [Moussa2016]. The paper says
     "The square contains 2D^2-2D+1 qudits, D^2-D vertices and faces", which gives 5 qubits
     and 2+2 checks at D=2 (substitution verified analytically). Moussa works D=5 in detail
     and never singles out D=2.
   - Breuckmann and Burton give the general fold-transversal construction and recover
     Moussa's surface-code H and S as a special case [BB2024].
   - Vasmer and Kubica draw this [[5,1,2]] code explicitly, as a morphed Steane code, with
     S on three qubits (S or S-dagger by shading), CZ on the remaining pair, and logical H =
     H on all five qubits plus one SWAP [VK2022]. That is our gate set.
   - The qutrit [[5,1,2]]_3 version was already used for detection-only LRB in the Sdim
     paper, which cites Moussa for the code [Kabir2025].
2. **Hardware.** My searches (through October 2026) found no hardware demonstration of a
   fold-transversal gate on any surface code. I also found no LRB study, experiment or
   simulation, on the qubit [[5,1,2]] that uses correction.
3. **Hardware d=2 codes are not ours.** Every d=2 surface-code experiment I found uses the
   rotated four-data-qubit layout ([[4,1,2]], 7 qubits with ancillas) or the [[4,2,2]] code.
   A rotated code of even distance has d^2-1 independent checks, an odd number, so it cannot
   have equally many X and Z checks. A symmetry of the form "H on every qubit, then a qubit
   permutation" needs equal numbers, so the rotated [[4,1,2]] has no such logical H
   (verified analytically). The d=2 experiments report no logical H [Marques2022].
4. **d=3 versions.** The unrotated [[13,1,3]] (D=3 in Moussa's count) keeps the diagonal
   fold and both fold-transversal gates. Serra-Peralta et al. decode exactly this family at
   d=3, 5, 7 [SST2025]. The rotated [[9,1,3]] has a 90-degree-rotation duality, which "fixes
   a single physical qubit and so we cannot construct a phase-type encoded logical gate"
   [BB2024]. Its H needs a patch rotation or a deformation [Chen2026; Geher2024], and its S
   needs the half-cycle trick [Chen2026; MBG2023].

## 1. Bibliography

Tier A sources are core to this assignment. Tier B sources support a single point.
"Fetched" means I opened the page and confirmed the title, authors, year and IDs.

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched | Tier |
|---|---|---|---|---|---|---|---|
| Moussa2016 | Transversal Clifford gates on folded surface codes | J. E. Moussa | 2016 | arXiv:1603.02286; 10.1103/PhysRevA.94.042316 (PRA 94, 042316) | https://arxiv.org/abs/1603.02286 | yes, abs + full text (ar5iv) | A |
| BB2024 | Fold-Transversal Clifford Gates for Quantum Codes | N. P. Breuckmann, S. Burton | 2022 arXiv; 2024 Quantum 8, 1372 | arXiv:2202.06647; 10.22331/q-2024-06-13-1372 | https://arxiv.org/abs/2202.06647 | yes, abs + HTML v3 | A |
| VK2022 | Morphing quantum codes | M. Vasmer, A. Kubica | 2022 (PRX Quantum 3, 030319) | arXiv:2112.01446; 10.1103/PRXQuantum.3.030319 | https://arxiv.org/abs/2112.01446 | yes, abs + HTML v2 (Fig. 1) | A |
| Kabir2025 | Sdim: A Qudit Stabilizer Simulator | A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang, B. Sorée | 2025 | arXiv:2511.12777 | https://arxiv.org/abs/2511.12777 | yes, abs + HTML v2 | A |
| Chen2026 | Transversal Logical Clifford gates on rotated surface codes with reconfigurable neutral atom arrays (PRL title: Transversal Logical Clifford Gates on the Rotated Surface Code with Reconfigurable Neutral Atom Arrays) | Z.-H. Chen, M.-C. Chen, C.-Y. Lu, J.-W. Pan | 2024 arXiv; 2026 PRL 136, 130601 | arXiv:2412.01391; 10.1103/m7tq-9v3g | https://arxiv.org/abs/2412.01391 | yes, abs + HTML + PRL page | A |
| SST2025 | Decoding across transversal Clifford gates in the surface code | M. Serra-Peralta, M. H. Shaw, B. M. Terhal | 2025 | arXiv:2505.13599; journal DOI 10.1103/sk5y-25b1 listed on arXiv (journal not named there) | https://arxiv.org/abs/2505.13599 | yes, abs + HTML | A |
| KZ2024 | Cross-cap defects and fault-tolerant logical gates in the surface code and the honeycomb Floquet code | R. Kobayashi, G. Zhu | 2024 (PRX Quantum 5, 020360) | arXiv:2310.06917; 10.1103/PRXQuantum.5.020360 | https://arxiv.org/abs/2310.06917 | yes, abs + HTML | A |
| Andersen2020 | Repeated quantum error detection in a surface code | C. K. Andersen, A. Remm, S. Lazar, S. Krinner, N. Lacroix, G. J. Norris, M. Gabureac, C. Eichler, A. Wallraff | 2020 (Nat. Phys.) | arXiv:1912.09410; 10.1038/s41567-020-0920-y | https://arxiv.org/abs/1912.09410 | yes, abs + ar5iv | A |
| Marques2022 | Logical-qubit operations in an error-detecting surface code | J. F. Marques, B. M. Varbanov, M. S. Moreira, H. Ali, N. Muthusubramanian, et al., B. M. Terhal, L. DiCarlo (13 authors) | 2022 (Nat. Phys. 18, 80) | arXiv:2102.13071; 10.1038/s41567-021-01423-9 | https://arxiv.org/abs/2102.13071 | yes, abs + ar5iv | A |
| Bultink2020 | Protecting quantum entanglement from leakage and qubit errors via repetitive parity measurements | C. C. Bultink, T. E. O'Brien, R. Vollmer, N. Muthusubramanian, M. W. Beekman, M. A. Rol, et al., L. DiCarlo (12 authors) | 2020 (Sci. Adv. 6, eaay3050) | arXiv:1905.12731; 10.1126/sciadv.aay3050 | https://arxiv.org/abs/1905.12731 | yes, abs | A |
| Linke2017 | Fault-tolerant quantum error detection | N. M. Linke, M. Gutierrez, K. A. Landsman, C. Figgatt, S. Debnath, K. R. Brown, C. Monroe | 2017 (Sci. Adv. 3, e1701074) | arXiv:1611.06946; 10.1126/sciadv.1701074 | https://arxiv.org/abs/1611.06946 | yes, abs + ar5iv | A |
| Takita2017 | Experimental demonstration of fault-tolerant state preparation with superconducting qubits | M. Takita, A. W. Cross, A. D. Córcoles, J. M. Chow, J. M. Gambetta | 2017 (PRL 119, 180501) | arXiv:1705.09259; 10.1103/PhysRevLett.119.180501 | https://arxiv.org/abs/1705.09259 | yes, abs + ar5iv | A |
| HF2019 | Fault-Tolerant Logical Gates in the IBM Quantum Experience | R. Harper, S. T. Flammia | 2019 (PRL 122, 080504) | arXiv:1806.02359; 10.1103/PhysRevLett.122.080504 | https://arxiv.org/abs/1806.02359 | yes, abs + ar5iv | A |
| BDB2024 | Genons, Double Covers and Fault-tolerant Clifford Gates | S. Burton, E. Durso-Sabina, N. C. Brown | 2024 | arXiv:2406.09951 | https://arxiv.org/abs/2406.09951 | yes, abs + HTML | A |
| Krinner2022 | Realizing repeated quantum error correction in a distance-three surface code | S. Krinner, N. Lacroix, A. Remm, A. Di Paolo, E. Genois, C. Leroux, et al., A. Wallraff (16 authors) | 2022 (Nature 605, 669) | arXiv:2112.03708; 10.1038/s41586-022-04566-8 | https://arxiv.org/abs/2112.03708 | yes, abs + ar5iv | A |
| Lacroix2025 | Scaling and logic in the color code on a superconducting quantum processor | N. Lacroix, A. Bourassa, F. J. H. Heras, L. M. Zhang, J. Bausch, A. W. Senior, et al. (215 authors) | 2025 (Nature 645, 614) | arXiv:2412.14256; 10.1038/s41586-025-09061-4 | https://arxiv.org/abs/2412.14256 | yes, abs + HTML | A |
| RA2021 | Realization of real-time fault-tolerant quantum error correction | C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, A. Hankin, J. P. Gaebler, et al. (17 authors) | 2021 (PRX 11, 041058) | arXiv:2107.07505; 10.1103/PhysRevX.11.041058 | https://arxiv.org/abs/2107.07505 | yes, abs + ar5iv + PRX page | A |
| Paetznick2024 | Demonstration of logical qubits and repeated error correction with better-than-physical error rates | A. Paetznick, M. P. da Silva, C. Ryan-Anderson, J. M. Bello-Rivas, J. P. Campora III, A. Chernoguzov, et al. (33 authors) | 2024 | arXiv:2404.02280 | https://arxiv.org/abs/2404.02280 | yes, abs + HTML v3 | A |
| KYP2015 | Unfolding the color code | A. Kubica, B. Yoshida, F. Pastawski | 2015 | arXiv:1503.02065; 10.1088/1367-2630/17/8/083026 | https://arxiv.org/abs/1503.02065 | yes, abs | B |
| ECZ512 | [[5,1,2]] rotated surface code (Error Correction Zoo entry) | Error Correction Zoo (community wiki) | accessed 2026-10-28 | none | https://errorcorrectionzoo.org/c/stab_5_1_2 | yes | B |
| MBG2023 | Relaxing Hardware Requirements for Surface Code Circuits using Time-dynamics | M. McEwen, D. Bacon, C. Gidney | 2023 (Quantum 7, 1172) | arXiv:2302.02192; 10.22331/q-2023-11-07-1172 | https://arxiv.org/abs/2302.02192 | yes, abs + ar5iv | B |
| Geher2024 | Error-corrected Hadamard gate simulated at the circuit level | G. P. Gehér, C. McLauchlan, E. T. Campbell, A. E. Moylett, O. Crawford | 2024 (Quantum 8, 1394) | arXiv:2312.11605; 10.22331/q-2024-07-02-1394 | https://quantum-journal.org/papers/q-2024-07-02-1394/ | yes, journal page | B |
| Sahay2026 | Fold-transversal surface code cultivation | K. Sahay, P.-K. Tsai, K. Chang, Q. Su, T. B. Smith, S. Singh, S. Puri | 2025 arXiv; 2026 PRX Quantum 7, 033006 | arXiv:2509.05212; 10.1103/gpvl-lg4c | https://arxiv.org/abs/2509.05212 | yes, abs + HTML v2 | B |
| PR2024 | Distance-four quantum codes with combined postselection and error correction | P. Prabhu, B. W. Reichardt | 2024 (PRA 110, 012419) | arXiv:2112.03785; 10.1103/PhysRevA.110.012419 | https://arxiv.org/abs/2112.03785 | yes, abs + APS page | B |
| Lin2026 | Surface code logical operations on a superconducting quantum processor | W. Lin, S. Guo, Y. Ma, Z. Yi, K. Zhang, J. Bei, et al. (147 authors) | 2026 | arXiv:2607.01473 | https://arxiv.org/abs/2607.01473 | yes, abs only | B |
| IQM2025 | How IQM is tackling quantum error correction (company blog) | F. R. F. Pereira (IQM) | 2025 | none | https://iqm.tech/?p=13678 | yes | B |

Fetched and confirmed but not used below: Zhang et al., arXiv:2405.09035 (d=2 codes on the
Wukong processor, ancilla-free, postselection on final readout only); Postler et al.,
arXiv:2111.12654 (Nature 605, 675, flag-FT Steane code, 78.9% acceptance for FT |0>_L);
Postler et al., arXiv:2312.09745 (PRX Quantum 5, 030326, Steane-type QEC); Nigg et al.,
arXiv:1403.5426 (7-qubit color code, code-space overlap only 34%); Bluvstein et al.,
arXiv:2312.03982 and arXiv:2506.20661 (neutral atoms); Moylett and Jonnadula,
arXiv:2512.17999 (fold-transversal gates on Floquet codes); Combes et al., arXiv:1702.03688
(logical RB); Roffe, arXiv:1907.11157 (QEC review).

## 2. Key findings (all from literature)

Each line gives the research question it bears on (Q1 decoders, Q2 non-Markovianity,
Q3 hardware) and the citation keys.

### A. The code and its gates

- F1. Q1, Q3. From literature [Moussa2016]. Moussa builds folded surface codes for qudits of any dimension. In them, H is H or H-dagger on the fold qudits plus a "mirror gate" (H combined with a swap of the two layers) on mirror pairs, and S is S or S-dagger on the fold qudits plus CZ or CZ-dagger on mirror pairs. He states that "both code families admit transversal implementations of single-qudit Clifford gates" at 2D^2-2D+1 data qudits per logical qudit.
- F2. Q1, Q3. From literature [BB2024]. Breuckmann and Burton extend folding to any CSS code with a ZX-duality. The Hadamard-type gate is transversal H followed by SWAPs of mirror pairs. The phase-type gate puts S on one set of fixed qubits and S-dagger on the other (chosen so each X check meets both sets equally) and CZ on mirror pairs, which reproduces Moussa's surface-code H and S.
- F3. Q1. From literature [BB2024]. They call the scheme "trivially fault-tolerant" and say it "is compatible with any decoding scheme, as it can be executed between rounds of error correction". They leave "the performance of these fold-transversal gates under simulated noise", and experiments, to future work.
- F4. Q1. From literature [VK2022]. Vasmer and Kubica state that "Morphing the Steane code produces a [[5,1,2]] code with a fault-tolerant implementation of the Clifford group". Their Fig. 1 caption puts S or S-dagger (by shading) on the grey qubits, CZ on the two white qubits, and gives logical H as H on all five qubits followed by one SWAP.
- F5. Q1. From literature [ECZ512; BDB2024]. The Error Correction Zoo lists our CSS code (four weight-3 checks around one central qubit) as the "[[5,1,2]] rotated surface code", gives logical S = S0 S2 S3-dagger CZ(1,4) in its own labelling, and calls it the smallest code with a fault-tolerant logical S from a diagonal depth-one circuit. Burton et al. write that "surface codes, such as the [[5,1,2]] code are genon codes on a sphere".
- F6. Q2. From literature [Kabir2025]. The Sdim paper ran detection-only LRB (LRB-D) on the five-qutrit [[5,1,2]]_3 detection surface code with "a single terminal syndrome extraction", postselecting on the all-zero syndrome under depolarizing gate noise. It reports that the curves decay faster than one unprotected qutrit and that "terminal detection can produce decay curves with non-Markovian structure".
- F7. Background. From literature [KYP2015]. The 2D triangular color code with boundaries is equivalent, up to local unitaries and ancillas, to a folded toric code with boundaries. The folding picture starts here.

### B. Rotated versus unrotated codes, and d=3

- F8. Q3. From literature [BB2024]. Breuckmann and Burton note that a second ZX-duality of the surface code is a 90-degree rotation. They write that "this rotation fixes a single physical qubit and so we cannot construct a phase-type encoded logical gate".
- F9. Q3. From literature [MBG2023]. In the standard rotated-surface-code cycle, "The half-cycle state is remarkable for being an unrotated surface code state". It is "a surface code state with the same distance as the end-cycle state", and all qubits take part in it.
- F10. Q3. From literature [Chen2026]. Chen et al. do logical H on the rotated code as transversal H plus a pi/2 patch rotation, made of two reflections by 2D acousto-optic deflector moves. They do logical S as the fold-transversal S at the half-cycle of one syndrome-extraction round. Under SD6 noise at d = 3 to 13, a plain decoder with the S rounds kept away from the time boundaries gave "2 to 3 times higher" logical error than an X-memory circuit, while S rounds near the time boundaries with their full-reweighting decoder reached "the memory baseline" (the abstract calls the S gate "on par with the quantum memory").
- F11. Q3. From literature [Geher2024]. On square-grid hardware the rotated code's logical H can use a transversal H (a domain wall in time) or a domain wall in space, followed by a SWAP-based QEC round that moves the patch back. Its circuit-level failure probabilities are "comparable to those of a quantum memory experiment".
- F12. Q3. From literature [Lin2026]. On a 107-qubit superconducting processor, Lin et al. implemented logical Hadamard and phase gates on d=3 rotated patches by code deformation and lattice surgery. They used multi-round syndrome extraction, neural-network decoding and no postselection.
- F13. Q1, Q3. From literature [Sahay2026]. Sahay et al. measure the fold-transversal H_XY of the unrotated surface code (gates on the diagonal, CZ on mirror pairs) to cultivate magic states. They say the scheme "is most naturally implemented with non-local gates", postselect with a complementary-gap decoder, and test noise models where single-qubit gates fail at p/10 and non-local gates at p or 5p.

### C. Decoding across fold-transversal gates

- F14. Q1. From literature [SST2025]. On the unrotated code with fold-transversal H, S and transversal CNOT, each followed by one QEC round, S and CNOT map stabilizers to products of stabilizers. A measurement error just after such a gate then fires two detectors of one type and one of the other (a weight-3 hyperedge), and MWPM with split hyperedges showed "no evidence of a threshold".
- F15. Q1. From literature [SST2025]. Their logical-observable matching decoder reached thresholds near 0.5% under SI1000 noise for two-qubit Clifford circuits at d = 3, 5, 7. Repeated-gate runs came close to the memory runs, and 2 or 3 QEC rounds between gates did worse than one.
- F16. Q1. From literature [SST2025]. Under SI1000 noise they found a hook error from "the long-range CZs in the physical implementation of the S gate". It lowers the minimum weight of a logical failure by one, from 3 to 2 at d=3.
- F17. Q1, Q3. From literature [KZ2024]. Kobayashi and Zhu write that the fold-transversal S "suffers from the same problem of error-propagation and long-range correlated noise as in the case of the folded surface code", because its gates couple mirror points that are far apart in the unfolded layout.
- F18. Q1, Q2. From literature [Lacroix2025]. Logical RB on a d=3 color code used m = 0 to 10, 25 sequences per m, one QEC cycle between transversal Cliffords and a neural-network decoder. It gave an added error of 0.0027(3) per logical Clifford against 0.0171(3) per QEC cycle. Syndrome comparisons were adjusted because H maps X stabilizers to Z stabilizers and S maps X stabilizers to Y-type ones.
- F19. Q1, Q3. From literature [Krinner2022]. The 17-qubit d=3 surface code decoded with MWPM, using weights inferred from measured syndrome correlations ("error-model-free"), reached about 3% logical error per 1.1 us cycle. This figure holds after rejecting runs with detected leakage, about 8% of runs per cycle.
- F20. Q1. From literature [PR2024]. Prabhu and Reichardt simulate a scheme that corrects single faults and postselects on double faults, so that a logical failure needs three faults. A postselected distance-four planar code then accumulates about 25 times less error than a distance-five surface code.

### D. d=2 surface-code and [[4,2,2]] experiments

- F21. Q2, Q3. From literature [Andersen2020]. In the 7-qubit d=2 rotated code (4 data, 3 ancillas, 1.92 us cycle, up to 10 cycles), about 25% of runs passed the preparation round. The chance of seeing no error in a cycle rose from about 40% to 50%, and the total success probability after 10 cycles approached 1e-4. The postselected logical error was 3.1% (X_L) and 2.6% (Z_L) per cycle.
- F22. Q2, Q3. From literature [Marques2022]. In the Delft Surface-7 d=2 code (up to 15 cycles), the postselected fraction decayed as A(1-gamma)^n with an error-detection rate gamma of about 45% per cycle. Fault-tolerant logical initialization reached 99.64% to 99.97%, no logical H was reported, and the authors "believe that the dominant error source is instead leakage to higher transmon states incurred during CZ gates".
- F23. Q3. From literature [Bultink2020]. Bultink et al. detect transmon leakage with hidden Markov models that read only the parity-measurement record. They keep Bell states over up to 26 parity measurements, removing leakage by postselection and correcting qubit errors with Pauli-frame updates.
- F24. Q3. From literature [Linke2017]. On five trapped ions the [[4,2,2]] code kept 65% to 78% of runs. After postselection, the fault-tolerantly encoded qubit's error was 0.1% to 0.2%, against 1.7% to 2.4% for the non-fault-tolerant gauge qubit.
- F25. Q2, Q3. From literature [Takita2017]. On five transmons the [[4,2,2]] code accepted about 76% to 79% of runs. The authors give the "exponentially decreasing acceptance probability as a function of the number of syndrome measurements" as a reason repeated detection is impractical.
- F26. Q2. From literature [HF2019]. Harper and Flammia ran RB inside the [[4,2,2]] code space on IBM QX5, postselecting only on the final measurement, with logical H as H on four qubits then SWAP12 and logical CNOTs by relabelling. Two-qubit infidelity fell from 5.8(2)% to 0.60(3)%, and they warn that "There is at present no complete theory for fitting an RB decay curve with post-selection".
- F27. Q2. From literature [BDB2024]. Burton et al. ran single-qubit Clifford RB in a [[4,1,2]] error-detecting code on Quantinuum H1-1, with H and S done by local Cliffords and qubit relabelling and detection only at the end. They discarded 28/1600, 90/1600 and 56/800 shots at depths 16, 256 and 512, and found that the curve "matches a quadratic fit instead of a linear one".

### E. Steane-code and other trapped-ion experiments

- F28. Q3. From literature [RA2021]. The [[7,1,3]] color (Steane) code on a ten-qubit trapped-ion QCCD machine, with real-time decoding and Pauli-frame tracking, showed "approximately 2.70(6)% per QEC cycle" of logical error. Its logical SPAM error was 1.7e-3 against 2.4e-3 physical, and its encoding check retries a failed preparation instead of discarding the shot.
- F29. Q1, Q3. From literature [Paetznick2024]. Measured side by side on a trapped-ion QCCD machine, [[7,1,3]] Bell states had 0.05% logical error with correction, with about 75% of shots kept by the preparation check. With detection they had 0.001%, with about 72% kept overall, against a 0.50% physical baseline.
- F30. Q2, Q3. From literature [Paetznick2024]. For 1, 2 and 3 rounds of [[12,2,4]] error correction, preparation acceptance fell from 70.9% to 51.7% to 40.3%. Final-detection acceptance fell from 98.8% to 96.9% to 89.0%, and the logical error grew by about 0.3% per round.
- F31. Q3. From literature [IQM2025]. IQM describes a square-lattice Crystal QPU with nearest-neighbour coupling and a Star QPU in which "all qubits are coupled to a central resonator".

### Table: acceptance and logical error in the experiments (from literature)

| Key | Code and platform | Rounds | Kept fraction | Logical error | Decoding |
|---|---|---|---|---|---|
| Andersen2020 | rotated d=2, 7 transmons | up to 10, 1.92 us | ~25% at preparation; 40-50% per cycle; ~1e-4 after 10 cycles | 3.1% (X_L), 2.6% (Z_L) per cycle | detection |
| Marques2022 | rotated d=2, Surface-7 transmons | up to 15, 840 ns pipelined | ~55% per cycle (gamma ~45%), plus leakage rejection | FT init 99.64-99.97%; logical X, Z gate fidelity 97.9%, 98.1% | detection |
| Krinner2022 | rotated d=3, 17 transmons | up to 16, 1.1 us | ~92% per cycle (leakage rejection only) | 3.2% (Z_L), 2.9% (X_L) per cycle | MWPM, data-derived weights |
| Linke2017 | [[4,2,2]], 5 ions | 1 | 65-78% | 0.1-0.2% (FT qubit), 1.7-2.4% (gauge) | detection |
| Takita2017 | [[4,2,2]], 5 transmons | 1 | ~76-79% | e.g. 0.67% (protected) vs 2.57% (gauge) preparation error | detection |
| HF2019 | [[4,2,2]], IBM QX5 | final only | not given in the text | RB infidelity 0.60(3)% vs 5.8(2)% physical | detection |
| BDB2024 | [[4,1,2]] genon code, H1-1 | final only | 98.3% (depth 16), 94.4% (256), 93.0% (512) | quadratic decay, no rate extracted | detection |
| Lacroix2025 | d=3 color code, superconducting | 1 per Clifford, m up to 10 | no postselection reported for LRB | 0.0027(3) per Clifford, 0.0171(3) per cycle | neural network |
| RA2021 | [[7,1,3]], 10-qubit QCCD | repeated | no discards (retry on failed prep) | ~2.7% per cycle; SPAM 1.7e-3 | real-time, Pauli frame |
| Paetznick2024 | [[7,1,3]] and [[12,2,4]], QCCD | 1-3 ([[12,2,4]]) | Bell: ~75% (correct), ~72% (detect); rounds: 70.9/51.7/40.3% prep, 98.8/96.9/89.0% final | Bell: 0.05% (correct), 0.001% (detect); ~0.3% per round | correct vs detect |

Kept fractions for BDB2024 are my arithmetic from the discard counts in F27.

## 3. Implications for our [[5,1,2]] qubit LRB study

- I1. Verified analytically, from literature. Our code and gates are the D=2 case of [Moussa2016] and the explicit [[5,1,2]] of [VK2022], so we can cite them for H and S instead of deriving them. The Zoo's S formula [ECZ512] is ours after relabelling Zoo qubits (0,1,2,3,4) to our (0,3,4,1,2): S0 S2 S3-dagger CZ(1,4) becomes S on {0,4}, S-dagger on 1, CZ(3,2).
- I2. Verified numerically. I ran a stim 1.15 tableau check for this note; the scripts are in my scratchpad, not in the repo, and take under a second. The single-qubit syndromes in CONTEXT.md are correct. Logical H maps S_Z1 to S_X1, S_Z2 to S_X2 (and back) and swaps X_L and Z_L. Logical S fixes S_Z1, S_Z2 and Z_L, maps X_L to Y_L, and maps S_X1 to S_X1·S_Z1 and S_X2 to S_X2·S_Z2 with relative sign +1.
- I3. Conjecture, from I2 with [SST2025; Lacroix2025]. In a space-time decoder, the detector for an X check in the round after an S has to be m_X(t+1) xor m_X(t) xor m_Z(t). A measurement error on a Z-check ancilla in the round before an S then fires three detectors, the d=2 version of the weight-3 hyperedge in F14, so split-edge MWPM is the wrong default. With four checks per round, an exact maximum-likelihood decoder should be cheap. One way is a forward pass over 16 syndrome states times 4 logical cosets per round, close to the hidden-Markov-model approach of [Bultink2020].
- I4. Verified numerically, same check. All 15 non-identity Paulis on qubits (2,3) have a non-zero syndrome. So one fault in CZ(2,3) or SWAP(2,3) is always detected, which supports these gates in LRB-D. But five of the nine weight-2 Paulis have the syndrome of a single-qubit error and differ from it by a logical. The other four have syndromes that no single-qubit error produces:

  | Pauli on (2,3) | Syndrome (Z checks; X checks) | Same syndrome as | Difference |
  |---|---|---|---|
  | X2 X3 | (1,1); (0,0) | X1 | X_L |
  | Z2 Z3 | (0,0); (1,1) | Z1 | Z_L |
  | Y2 Y3 | (1,1); (1,1) | Y1 | Y_L |
  | X2 Z3 | (0,1); (0,1) | Y4 | Y_L |
  | Z2 X3 | (1,0); (1,0) | Y0 | Y_L |
  | X2 Y3, Y2 X3, Y2 Z3, Z2 Y3 | (1,1);(0,1), (1,1);(1,0), (0,1);(1,1), (1,0);(1,1) | none | weight-2 only |

- I5. Conjecture. A correcting decoder that trusts the qubit-1 syndromes as "uniquely identifiable" (CONTEXT.md) turns the first five rows of I4 into logical errors. If DEPOLARIZE2(p) gives each two-qubit Pauli probability p/15, that costs about p/3 per S gate, and per H gate while the SWAP is physical. This is the d=2 version of the S-gate hook error in [SST2025] and of the correlated noise in [KZ2024]. It is small next to the check-round noise in I7, so it matters for decoder design more than for the headline numbers.
- I6. Verified numerically, same check. Hook errors inside our check circuits stay benign. A Z error on a Z-check ancilla, or an X error on an X-check ancilla, between its CNOTs always leaves a data error equivalent to one single-qubit error. For example, ancilla 6 after CNOT 1->6 leaves Z2 Z4, which equals Z1 up to a stabilizer. The check schedule therefore keeps circuit distance 2.
- I7. Conjecture, a rough count from the CONTEXT.md noise budget. Distance 2 lets a correcting decoder only guess inside the pairs {0,3} and {2,4} for X errors and {0,2} and {3,4} for Z errors. With about 5 x DEPOLARIZE1(2p) per data qubit per round, each qubit takes an X-type error with probability about 5 x 4p/3, roughly 6.7p, per round. A decoder that guesses right half the time then loses about 0.5 x 4 x 6.7p, roughly 13p, per round to X_L-type failures, and about as much to Z_L-type. Matched physical RB, DEPOLARIZE1(0.5p) per gate, has infidelity about p/3 per gate. I expect pure correction on [[5,1,2]] to give a clean exponential with acceptance 1 but a large decay rate, likely worse than `const=0`, which has no check noise at all. This matches the experimental groups' view that d=2 detects but does not correct [Andersen2020; Marques2022] and that correction needs d=3 [Krinner2022].
- I8. Conjecture. The decoders worth testing at d=2 are hybrids. One kind corrects where the posterior is lopsided (qubit-1 syndromes and hook-equivalent patterns) and rejects where it is near 50/50. Another rejects on a soft-output gap, as in the complementary-gap postselection of [Sahay2026]. [PR2024] is the distance-four version of the same idea: correct one fault, reject on two. We should compare the whole acceptance-versus-fidelity curve with LRB-D, not one point. [Paetznick2024] shows the scale: for the Steane code, detection cut the logical error 50-fold at the cost of a few percent more rejection.
- I9. From literature and conjecture, Q2. Detection that sees few syndromes gives non-exponential decays in an experiment [BDB2024, quadratic], in a fitting caveat [HF2019] and in the Sdim qutrit LRB-D case study [Kabir2025]. This agrees with the qutrit `const=1` curve in CONTEXT.md. Conjecture: correcting, or rejecting, every round makes each round's logical error close to independent of the others, so F(m) should return to a single exponential. An infidelity that grows like m^2 at small m under `const=1` would be the signature of "single faults caught, fault pairs not caught".
- I10. From literature and conjecture, Q3. Superconducting d=2 codes keep only about 40% to 55% of runs per round [Andersen2020; Marques2022]. LRB-D with `unif=1` at m=22 would then keep about 0.5^23, roughly 1e-7, of shots, so on such hardware LRB-D stops being usable near m = 10 and only correction, or hybrids, reach our full depth grid. Trapped-ion runs keep far more [Linke2017; BDB2024; Paetznick2024].
- I11. Geometry verified analytically; hardware claims are conjecture, Q3. The [[5,1,2]] with one ancilla per check fits a 3x3 nearest-neighbour patch, with data on the corners and centre and ancillas on the edge midpoints, so all 12 check CNOTs are nearest-neighbour.
  - The SWAP(2,3) inside logical H can be virtual. Apply H to the five data qubits and relabel 2 and 3. After that, ancillas 5 and 6 (wired to {0,1,3} and {1,2,4}) measure X checks and ancillas 7 and 8 (wired to {0,1,2} and {1,3,4}) measure Z checks, so only the check circuit type changes. This is how [HF2019] and [BDB2024] handle permutations. Logical S is unchanged by the relabelling, while logical X, logical Z and the terminal readout pair follow the frame.
  - Our model's DEPOLARIZE2(p) after SWAP(2,3) is therefore pessimistic for hardware, and a variant without it would be fairer.
  - CZ(2,3) joins opposite corners. It is native on trapped ions and on movable neutral atoms [Chen2026], and may be available through IQM's Star resonator [IQM2025]. A nearest-neighbour Crystal-type grid needs routing.
- I12. Conjecture, Q1 and Q3. All our logical gates are Clifford, so corrections can be applied after the run as a Pauli frame. [Krinner2022] did this in postprocessing and [RA2021] in real time, so LRB with correction needs no feed-forward hardware. Leakage will still force some rejection on superconducting devices [Marques2022; Krinner2022; Bultink2020], and our leakage-free stim model cannot show that.
- I13. Conjecture, Q1 and Q2. The Clifford error in [Lacroix2025] is about six times smaller than the QEC-cycle error, and extra rounds between gates hurt in [SST2025]. Both fit the CONTEXT.md estimate that check rounds are our main data-noise source. Every circuit in our grid physically contains all m+1 rounds, so a fair correction study should also include variants that physically drop rounds, not only policies that ignore their outcomes.

## 4. Seed corrections

No seed had a wrong arXiv ID. The fixes are titles, venues and scope.

- Moussa (arXiv:1603.02286) is correct; sole author J. E. Moussa, PRA 94, 042316 (2016). It treats qudits of any dimension and works D=5 in detail; it gives the family size formula but does not discuss D=2 or D=3 explicitly.
- Breuckmann and Burton (arXiv:2202.06647) is correct; exact title "Fold-Transversal Clifford Gates for Quantum Codes", published in Quantum 8, 1372 (2024). Its worked examples are LDPC codes (Bring's [[30,8,3]] code and others), not small surface codes.
- Chen et al.: the guessed ID arXiv:2412.01391 is right. Authors Z.-H. Chen, M.-C. Chen, C.-Y. Lu, J.-W. Pan. Published as PRL 136, 130601 (2026), DOI 10.1103/m7tq-9v3g, with the title changed to "...on the Rotated Surface Code...". It is a proposal with simulations, not an experiment.
- Kubica, Yoshida, Pastawski (arXiv:1503.02065) is correct; title "Unfolding the color code", DOI 10.1088/1367-2630/17/8/083026 as listed on arXiv. It is about color-code and toric-code equivalence, not about gates on folded surface codes.
- Andersen et al. (arXiv:1912.09410) is correct; Nat. Phys. (2020), DOI 10.1038/s41567-020-0920-y. The code is the rotated 4-data-qubit d=2 code, not [[5,1,2]].
- Marques et al. (arXiv:2102.13071) is correct; Nat. Phys. 18, 80 (2022). It uses the same rotated d=2 layout and reports no logical H.
- Bultink et al. (arXiv:1905.12731) is correct; exact title "Protecting quantum entanglement from leakage and qubit errors via repetitive parity measurements", Sci. Adv. 6, eaay3050 (2020). It stabilizes two-qubit Bell states, not a surface code.
- Krinner et al. (arXiv:2112.03708) is correct; exact title "Realizing repeated quantum error correction in a distance-three surface code", Nature 605, 669 (2022).
- Linke et al. (arXiv:1611.06946) is correct; Sci. Adv. 3, e1701074 (2017). The code is [[4,2,2]], not a surface code.
- Takita et al. (arXiv:1705.09259) is correct; exact title "Experimental demonstration of fault-tolerant state preparation with superconducting qubits", PRL 119, 180501 (2017). The code is [[4,2,2]] on five transmons.

## 5. Open questions

1. I found no hardware demonstration of a fold-transversal gate on a surface code. Bluvstein et al. 2025 (arXiv:2506.20661) mentions only transversal gates in general in its abstract. I did not check its full text.
2. Is the half-cycle state of the 7-qubit rotated d=2 code (Surface-7) a d=2 unrotated code on which our fold-transversal S could act? [MBG2023] says all qubits take part in the half-cycle state, so it may not be exactly our 5-qubit code. If it is, the Chen trick would let 7-qubit superconducting devices run our S without a corner-to-corner CZ. Untested.
3. Burton et al.'s [[4,1,2]] genon code gets H and S from local Cliffords plus relabelling [BDB2024]. Is it local-Clifford equivalent to the rotated d=2 surface code? Would it be a cheaper LRB test bed than [[5,1,2]]? Not checked.
4. How should we fit postselected LRB decays? [HF2019] says no complete theory exists, and [BDB2024] saw quadratic behaviour. A model with one parameter for caught single faults and one for uncaught fault pairs could be tried on the qutrit `const=1` data.
5. UNVERIFIED: an APS March Meeting 2025 abstract by Serra-Peralta and Terhal ("Decoding across transversal logical Clifford gates: the S gate for surface code computation"). A search snippet says MWPM performs poorly for the fold-transversal S and a low-complexity decoder matches BP-OSD, but the abstract page was truncated, so I do not use it.
6. UNVERIFIED: a Nature 2026 article, "Improved quantum processor logical error rates via correction and detection" (DOI 10.1038/s41586-026-10628-y), reports 11x to 800x gains that resemble [Paetznick2024]. The page redirected to a login, so I could not confirm that it is the published version.
7. How large is the X2X3-type failure (I5) compared with check-round noise once the decoder uses the real detector error model? The decoder agents could settle this quickly with stim.
8. IQM's blog gives no gate fidelities for a CZ through the Star resonator, so whether CZ(2,3) through it is competitive with a Crystal CZ is open.

## 6. Search log

Queries run (WebSearch), besides direct fetches of every seed and of every cited page:
`fold-transversal Hadamard S gate unrotated surface code distance 2 five qubit [[5,1,2]]`;
`"[[5,1,2]]" surface code`; `fold-transversal logical gate experiment surface code demonstration 2025`;
`logical randomized benchmarking transversal Clifford gates colour code superconducting processor Lacroix`;
`distance-two surface code repeated error detection experiment postselection fraction logical error per cycle`;
`Steane code trapped ion experiment repeated error correction postselection acceptance rate logical error rate 2024`;
`fold-transversal gate hardware demonstration trapped ion OR neutral atom OR superconducting "fold-transversal" experiment`;
`logical randomized benchmarking error detecting code experiment postselection [[4,2,2]] surface code`;
`transversal Hadamard rotated surface code d=3 [[9,1,3]] fold symmetry reflection unrotated [[13,1,3]] logical S gate circuit-level simulation`;
`IQM superconducting quantum computer surface code error detection logical qubit experiment 2025`;
`logical Hadamard gate distance-two surface code experiment superconducting qubits 2024 2025 error detection postselection`;
`"surface code" "fold-transversal" "S gate" logical error rate decoder "unrotated" 2026`;
`"Demonstrating a universal logical gate set in error-detecting surface codes on a superconducting quantum processor" npj Quantum Information 2025`;
`neutral atom experiment surface code transversal logical S gate fold diagonal CZ demonstration Bluvstein 2025 "architectural mechanisms"`.
