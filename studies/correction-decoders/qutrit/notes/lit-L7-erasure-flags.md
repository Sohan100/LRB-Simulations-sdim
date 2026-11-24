# L7: erasure conversion, leakage heralding, flags, and what distance-2 experiments did with their syndromes

Agent topic L7 of the qutrit LRB correction study (see `00-brief.md`).

## Labels and method

Every claim carries one label:

- **[lit]**: from literature. The source is the paper of the section it appears in, fetched in this task.
- **[ana]**: verified analytically (an argument written out here).
- **[num]**: verified numerically (a computation run in this task; Appendix A gives the method and the paths).
- **[conj]**: conjecture.

How the sources were read. For every paper I fetched the arXiv abstract page, or the repository page for the thesis, to confirm title, authors and identifiers. I confirmed the journal metadata (title, volume, pages, DOI) through the Crossref API record of the DOI. For the extracted facts I read the full text: arXiv HTML (`arxiv.org/html/<id>`), ar5iv HTML (`ar5iv.org/abs/<id>`), or the arXiv PDF converted to text. All numbers quoted below come from the arXiv full text. Journal versions may differ in small details. Paper titles on arXiv and in the journal sometimes differ; both are given where they do.

All computations ran on the login node in under a few minutes (Appendix A). Read-only inputs were the baseline run folder `Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000` and the code in `src/lrb/`.

---

## 0. Main findings for this study

1. **Distance-2 experiments only postselected.** Linke 2017, Takita 2017, Andersen 2020, Marques 2022 and Self 2024 rejected a shot whenever any ancilla outcome, or a final stabilizer value recomputed from the data readout, was nontrivial [lit]. None of them corrected any syndrome. None treated measurement-only events differently from data errors [lit]. The one partial-correction precedent is Bluvstein 2024. It applied correction (correlated decoding) to every shot, then rejected only shots whose decoder correction weight exceeded a threshold ("sliding-scale error detection"). This traded acceptance for fidelity continuously: for the color-code GHZ state, 72% fidelity with correction only, about 90% at 50% acceptance, and 99.85% with full postselection [lit].
2. **The d=2 experiments all ended with a stabilizer check computed from the data readout. Our ordinary LRB circuits do not.** Each experiment measured all data qubits at the end and recomputed the stabilizers relevant to the readout basis (Linke, Takita, Andersen, Marques), or added a flagged final check (Self) [lit]. In our ordinary circuits only data qutrits 0 and 3 are read out. As a result there is a first-order hole: a single fault that puts a Z-type error on qutrit 0 or 3 after its last X-check in the final round flips the logical outcome and leaves no syndrome. Exact single-fault enumeration on the compact DEM gives an undetected logical-flip probability of 4.95 p. This is the same for m = 0, 2 and 4, so it does not depend on depth [num]. It predicts 1 - F(m) = 1.5 × 4.95 p + O(p^2) = 7.425 p + O(p^2) for every m: verified numerically for `unif 1`. For the other policies the same holds analytically, because every other single-fault data error stays detectable until the terminal round, which is always used [ana]. The baseline LRB-D data give (1 - F(m=0))/p = 7.46, 7.47 and 7.42 at p = 1e-4, 2e-4 and 5e-4 [num]. So at small p this hole is the entire m = 0 infidelity of LRB-D. No decoder can correct it, because it leaves no syndrome [ana]. Reading out all five data qutrits in the X basis, as the `const 0` circuits already do, removes it at first order [ana].
3. **Flags are not needed for hook errors in this circuit.** All checks have weight 3. Every hook residual of the single-ancilla circuits is equivalent, modulo stabilizers, to a weight-1 error, and every weight-1 error is detectable [num]. Chao and Reichardt show that flags are needed for detection with weight-n checks of [[n,n-2,2]] codes [lit]; that is not our situation. Every nontrivial error on the only data-data gate pair {2,3} (SWAP(2,3), CZ(2,3)) is detectable [num]. A flag scheme for prime-dimension qudits exists (Durso-Sabina 2022, SUM/SUM^-1 pairs) [lit]. Adding flags under SI1000, where every measurement layer costs 2p of idle noise on every other wire, would most likely hurt [conj; Chamberland and Beverland report low pseudo-thresholds for flag EC, which they attribute to depth and idle noise, lit].
4. **Partial correction is possible in the [[5,1,2]]_3 code, but only with joint X/Z decoding.** Of the 40 weight-1 data errors, 24 lie in syndrome classes whose weight-1 members are all logically equivalent, so they are correctable at first order. The other 16 (pure X or pure Z errors on the corner qutrits 0, 2, 3, 4) form 8 ambiguous pairs that differ by a logical [num]. All 8 errors on the center qutrit 1, and all mixed X^a Z^b (a, b ≠ 0) errors on corner qutrits, are unambiguous [num]. A decoder that decodes the X and Z parts separately loses this, because the X and Z parts of a mixed error are each ambiguous on their own [ana].
5. **Heralded erasures turn this d=2 code into a correcting code for heralded errors.** Any single erased qutrit is correctable. Six of the ten erased pairs are also correctable: {0,1}, {0,4}, {1,2}, {1,3}, {1,4}, {2,3}. The pairs {0,2}, {0,3}, {2,4}, {3,4} are not [num]. The literature gives realistic herald quality: false positives 0.6-0.8% and false negatives 1.5% per check for dual-rail transmons (Levine 2024); leakage-detection error 7.7e-3 (1.2e-3 with two rounds) for dual-rail cavities (Chou 2024); 56% (single-qubit) and 33% (CZ) erasure fractions with under 1e-5 disturbance (Ma 2023) [lit]. Decoders use heralds by resetting erased locations to a uniformly random Pauli and giving erased edges zero weight (Union-Find in Wu 2022, MWPM in Kubica 2023) [lit].
6. **Leakage is the dominant non-Pauli, time-correlated error in superconducting QEC. Detection hides it; correction exposes it.** Marques 2022 found that leakage, not T1/T2, set its detection rate, and that postselection removed the leaked runs [lit]. McEwen 2021 and Miao 2023 show that without leakage removal, leakage produces detection events correlated over more than 10 rounds. Under correction, under 1% injected leakage pushed the logical error above 40% [lit]. For study question 2 (non-Markovianity), the Pauli SI1000 model contains no such memory. On hardware, moving from detection to correction would make the experiment more exposed to leakage-induced non-Markovian errors unless leakage-reduction units or leakage heralding are added [conj]. The h-LRU of Marques 2023 acts on the transmon level |3>, the first leakage level of a transmon qutrit [lit]. Whether it preserves coherence within {0,1,2} was not tested [conj].
7. **In the terminal round, Z-check outcomes do not matter at first order for the X-basis logical readout.** No single fault that fires only terminal-round Z-check detectors (ancillas 5, 6) flips the logical (m = 0, 2, 4 checked) [num]. Such events have total first-order probability 15.6 p (m=0) to 26-29 p (m=2, 4) [num]. Ignoring the terminal-round Z-checks raises acceptance by that amount with no first-order fidelity cost [num]. The second-order cost is not known [conj]. This mirrors the experiments, which recomputed only the stabilizers relevant to the measured basis from the final data readout (Takita: S_z only; Marques 2022: XXXX only for X_L readout) [lit].
8. **Scale of the acceptance problem.** The superconducting d=2 experiments rejected 45-60% of shots per cycle (Andersen, Marques); Andersen's success probability reached about 1e-4 after 10 cycles [lit]. The trapped-ion iceberg QV test discarded 86.8% [lit]. In the SI1000 qutrit model, the first-order rejection of one check round is 63.15 p [num], matching the baseline's 62-63 p at m = 0 [num]. The baseline rejects 46% at m = 0 for p = 0.01. By per-round rejection, those experiments therefore sat roughly where this model sits at p ≈ 0.01 [conj: the codes and circuits differ, so this is only an order-of-magnitude calibration].

---

## Part A. Distance-2 error-detection experiments

### A1. Linke et al. 2017: fault-tolerant [[4,2,2]] detection with trapped ions

**Citation.** "Fault-tolerant quantum error detection". N. M. Linke, M. Gutierrez, K. A. Landsman, C. Figgatt, S. Debnath, K. R. Brown, C. Monroe. Sci. Adv. 3, e1701074 (2017). arXiv:1611.06946. DOI 10.1126/sciadv.1701074. Fetched: https://arxiv.org/abs/1611.06946, full text https://ar5iv.org/abs/1611.06946, Crossref https://api.crossref.org/works/10.1126/sciadv.1701074.

Facts [lit]:
- Five 171Yb+ ions: four data qubits and one bare ancilla. The [[4,2,2]] code has S_x = XXXX and S_z = ZZZZ. Of the two encoded qubits, L_a is fault-tolerant; L_b is a gauge qubit with an undetectable single-fault channel. From a fault-tolerance viewpoint this is a [[4,1,2]] subsystem code. Gate orders were chosen so that no single fault causes an undetectable error on L_a; L_b absorbs the hook errors.
- With one ancilla, the two stabilizers were measured **in separate experiments** (one round each).
- **Syndrome use: pure postselection.** The run is discarded if the ancilla reads 1 or if the final data parity is odd. Quote: "We must therefore rely on post-selection to find and discard cases where an error occurred". Detected errors "cannot be uniquely identified and corrected".
- Yields (acceptance) for |00>_L: 91.1% after preparation (data-parity check only), 77.8% with S_z, 65.2% with S_x. In general, about 8% of runs are rejected by the data parity and a further about 20% by the ancilla. 5000-6000 shots per circuit, about 0.1% statistical uncertainty.
- Logical populations after postselection (Table 1, |L_a L_b>):
  - Preparation of |00>_L: 98.0% |00>, 1.7% |01> (L_b error), 0.1% |10> (L_a only), 0.2% |11>.
  - With S_z: 97.8 / 1.7 / 0.2 / 0.3%.
  - With S_x: 97.1 / 2.4 / 0.2 / 0.3%.
  - The text summarizes the L_a errors after a stabilizer as "0.3%"; the table gives 0.2% L_a-only plus 0.3% both.
  - After a transversal logical X_aX_b plus S_x, the L_a error is 0.7%, against 1.2% for a physical qubit after an X gate.
  - With added Pauli errors, L_a stays below the physical qubit over the whole range, and L_a and L_b converge above about 20% added error.
- Hardware: two-qubit gates about 97% fidelity, single-qubit about 99.1%. Fitted error model: 0.5% single-qubit, 1.0% two-qubit, 1.4% Stark phase.

Bearing on this study:
- The reference pattern: one round, reject on any nontrivial ancilla outcome, and check the readout-basis stabilizer from the data readout [lit]. Our LRB circuits lack that last check (finding 2) [num].
- Fault tolerance came from gate ordering: hooks were made harmless for L_a. In our code, hooks from weight-3 checks are already weight-1 equivalent (Part E1) [num].

### A2. Takita et al. 2017: fault-tolerant state preparation with superconducting qubits

**Citation.** "Experimental demonstration of fault-tolerant state preparation with superconducting qubits". M. Takita, A. W. Cross, A. D. Córcoles, J. M. Chow, J. M. Gambetta. Phys. Rev. Lett. 119, 180501 (2017). arXiv:1705.09259. DOI 10.1103/PhysRevLett.119.180501. Fetched: https://arxiv.org/abs/1705.09259, https://arxiv.org/html/1705.09259, Crossref record.

Facts [lit]:
- Five fixed-frequency transmons (IBM Quantum Experience device): four data qubits and one syndrome qubit. [[4,2,2]] code with a protected qubit p and a gauge qubit g. Logical states are prepared by measuring S_x with the syndrome qubit; S_z is computed in software from the final data readout. No flag qubit is used. Fault tolerance holds "by construction", because two-qubit gates never act directly between data qubits.
- **Syndrome use: pure postselection.** Postselect on the syndrome-qubit outcome, then on even data parity c1⊕c2⊕c3⊕c4 = 0.
- Acceptance 0.757-0.790 across the eight prepared cardinal states (Table 2). Conditional fidelity 0.962-0.973. Protected-qubit error 0.6-1.3% and gauge error 2.3-3.2% across the eight states (computed from the Table 2 populations). Example |1p1g>: acceptance 0.7853, correct population 0.9709, protected-qubit error 0.0067, gauge error 0.0257 (matching Table S3). With two-pulse CNOTs, acceptance for |++> fell to 0.277 because of coherent ZZ phases.
- Two-qubit error per gate 2.8-4.5%.
- On repeated detection: "Repeated syndrome measurements are not feasible in this device due to both technical limitations of measurement durations as well as exponentially decreasing acceptance probability as a function of the number of syndrome measurements. These limitations could be overcome by implementing a quantum error-correcting, rather than error-detecting, code".

Bearing on this study:
- The authors name exponentially decreasing acceptance as the reason to move to correction. This is the same motivation as this study's [lit].
- "Two-qubit gates never act directly on pairs of data qubits": our logical H and P do act on the data pair (2,3) through SWAP and CZ. Part E2 shows that any error on {2,3} is still detectable, and that the 81 operators on {2,3} have distinct syndromes [num].

### A3. Andersen et al. 2020: repeated error detection in a d=2 surface code

**Citation.** "Repeated quantum error detection in a surface code". C. K. Andersen, A. Remm, S. Lazar, S. Krinner, N. Lacroix, G. J. Norris, M. Gabureac, C. Eichler, A. Wallraff. Nat. Phys. 16, 875-880 (2020). arXiv:1912.09410. DOI 10.1038/s41567-020-0920-y. Fetched: https://arxiv.org/abs/1912.09410, https://arxiv.org/html/1912.09410, Crossref record.

Facts [lit]:
- Seven transmons (D1-D4, A1-A3) implement the d=2 surface code with XXXX, Z1Z3, Z2Z4. "Once an error is detected, the error can not be unambiguously identified, e.g. one would obtain the same syndrome outcome for an X-error on D1 and on D3".
- Initialization is heralded by a pre-measurement. Logical states are prepared probabilistically: product state, one cycle of checks, postselect on all ancillas reading 0. All ancillas read 0 with probability 25.1%, against 50% ideally. The probability of being in the code space is P_L = 0.717, and the logical fidelity in the code space averages 96.1% over |0_L>, |1_L>, |+_L>, |-_L> (98.2, 97.3, 94.2, 94.8%).
- **Syndrome use: pure postselection**, conditioned "on having detected no error in any repetition of the stabilizer measurement and on having the final measurement of the data qubits satisfy Z_D1 Z_D3 = Z_D2 Z_D4 = 1 (X_D1X_D2X_D3X_D4 = 1)".
- The probability of no error in a cycle, given none in earlier cycles, rises from about 40% to 50% with N. After N = 10 cycles (1.92 µs per cycle), total success approaches 1e-4.
- Postselected logical error per cycle: X_L 3.1% ± 0.45%, Z_L 2.6 ± 1.3%. Postselected logical lifetime 62.7 ± 9.4 µs and coherence time 72.5 ± 32.9 µs, against 16.8 µs and 21.5 µs for the best physical qubit.

Bearing on this study:
- This is the cleanest experimental analogue of our `unif 1` policy: reject on any raw ancilla flip in any round, plus a final data-derived check [lit]. Our baseline at p = 1e-3 rejects 91.9% at m = 22 (23 rounds, brief [data]). Andersen's device rejected about 99.99% after 10 rounds [lit]. A sustainable hardware LRB therefore cannot rely on full postselection over tens of rounds at current error rates [conj].

### A4. Marques et al. 2022: logical operations in an error-detecting surface code

**Citation.** "Logical-qubit operations in an error-detecting surface code". J. F. Marques, B. M. Varbanov, M. S. Moreira, H. Ali, N. Muthusubramanian, C. Zachariadis, F. Battistel, M. Beekman, N. Haider, W. Vlothuizen, A. Bruno, B. M. Terhal, L. DiCarlo. Nat. Phys. 18, 80-86 (2022). arXiv:2102.13071. DOI 10.1038/s41567-021-01423-9. Fetched: https://arxiv.org/abs/2102.13071, https://arxiv.org/html/2102.13071, Crossref record.

Facts [lit]:
- Surface-7 (d=2) on transmons: "no error syndrome combination is unique to a single error ... Consequently, this code cannot be used to correct such errors. We thus perform state stabilization by post-selecting runs in which no error is detected by the stabilizer measurements in any cycle."
- Definition used: "an operation is fault-tolerant if any single-fault produces a non-trivial syndrome and can therefore be post-selected out" (citing Gottesman 2016). The CZ order in the XXXX check "prevents the propagation of ancilla errors into logical qubit errors".
- Logical measurement: all data qubits are measured in the X or Z basis, and the outcomes are used both for the logical value and to recompute the stabilizers of that basis, "enabling a final step of error detection". Logical assignment fidelities: Z_L 99.4%, X_L 96.4% (95.8% in a second sweep), Y_L 87.5% (not fault-tolerant).
- Logical initialization fidelities (postselected): 99.83% (|0_L>), 99.97% (|1_L>), 97.02% (|+_L>), 95.54% (|-_L>). With fault-tolerant product-state variants, |±_L> reach 99.78% and 99.64%.
- Gate-by-measurement T_L and X^{π/2}_L are also implemented by postselection on the ancilla outcome. Logical gate fidelities: 97.3% (T_L), 95.6% (X^{π/2}_L), 97.9% (Z_L), 98.1% (X_L).
- Repetitive detection up to 15 cycles. The postselected fraction follows P(n) = A(1-γ)^n with error-detection rate γ_pip ≈ 45% per cycle for the pipelined scheme, and γ_pip/γ_par ≈ 97%. Cycle times: 840 ns (pipelined), 1000 ns (parallel).
- Leakage: conventional errors (T1, T2, readout) do not explain the detection rate. Simulations need about 5% leakage per CZ (L1 a free parameter) to reproduce P(n). Readout histograms show leakage accumulating with n. "The error detection scheme successfully post-selects leakage errors in both the ancilla and data qubits".

Bearing on this study:
- Pure postselection again. The per-cycle rejection was about 45% [lit].
- Detection automatically removes leaked runs, so pure-detection LRB on transmons is partially protected from leakage-induced, non-Markovian memory. Correction would not be, without leakage reduction units [conj; supported by C1, C2].
- Recomputing from the final data readout only the stabilizers of the measured basis is standard practice [lit]. In our circuits the analogous check would use the two X-stabilizers from an X-basis readout of all five qutrits [ana].

### A5. Self, Benedetti, Amaro 2024: the iceberg [[k+2,k,2]] code on trapped ions

**Citation.** "Protecting expressive circuits with a quantum error detection code". C. N. Self, M. Benedetti, D. Amaro. Nat. Phys. 20, 219-224 (2024). arXiv:2211.06703. DOI 10.1038/s41567-023-02282-2. Fetched: https://arxiv.org/abs/2211.06703, https://arxiv.org/html/2211.06703, Crossref record.

Facts [lit]:
- [[k+2,k,2]] code, k = 8 logical qubits on 10 data qubits plus 2 ancillas (Quantinuum H1-2, 12 qubits). Stabilizers are S_X = X^{⊗n} and S_Z = Z^{⊗n}.
- Initialization, syndrome measurement and final measurement are fault-tolerant flag circuits (after Chao and Reichardt). When both stabilizers are extracted, "the ancillas act as flag qubits to each other" (an ABBB...BA CNOT ordering). Fault tolerance (no single fault gives an undetectable logical error) was verified exhaustively for k up to 16.
- The logical gates (one MS gate each) are **not** fault-tolerant: XX, YY or ZZ errors after an MS gate are undetectable logical errors.
- **Syndrome use: pure postselection.** "If at any point in the circuit the readout of a1, a2, or the post-processed value of S_Z is -1 the circuit is discarded". S_X is extracted with a flagged circuit before the final readout, because it cannot be reconstructed from Z-basis data.
- Mirror circuits (a random unitary followed by its gate-by-gate inverse), 4 to 256 layers. For 32-256 layers, one syndrome round sits between the unitary and its inverse. In simulation, encoded circuits sustain about 4 times more layers at fixed survival probability.
- Logical quantum volume 2^8 passed with **86.8 ± 0.7% discard rate** (2 mid-circuit syndrome rounds, 100 circuits). Adding syndrome rounds raised the heavy-output frequency and the discard rate.
- "All circuits run to the end and error detection is implemented as a post-selection. However, it is possible to implement a conditional exit during runtime as soon as an error is detected. We expect this to halve the experimental runtime of a rejected circuit on average."
- The code "can correct for one qubit loss" using the protocol of Stricker 2020 (B7).

Bearing on this study:
- Its mirror-circuit layout (forward block, one check round, inverse block, final check) is structurally the same as our LRB circuits, where the inverse block has no checks [lit].
- An S_X check flagged before a Z readout is the analogue of the missing final data-derived check in our circuits (finding 2) [ana].
- Early abort on the first detection is a cheap hardware saving for LRB-D [lit/conj].

### A6. Bluvstein et al. 2024: logical processor with correlated decoding and sliding-scale postselection

**Citation.** "Logical quantum processor based on reconfigurable atom arrays". D. Bluvstein, S. J. Evered, A. A. Geim, S. H. Li, H. Zhou, T. Manovitz, S. Ebadi, M. Cain, M. Kalinowski, D. Hangleiter, J. P. Bonilla Ataides, N. Maskara, I. Cong, X. Gao, P. Sales Rodriguez, T. Karolyshyn, G. Semeghini, M. J. Gullans, M. Greiner, V. Vuletić, M. D. Lukin. Nature 626, 58-65 (2024; online Dec 2023). arXiv:2312.03982. DOI 10.1038/s41586-023-06927-3. Fetched: https://arxiv.org/abs/2312.03982, full text https://ar5iv.org/abs/2312.03982, Crossref record.

Facts [lit]:
- Surface codes d = 3, 5, 7 with a transversal CNOT. Correlated decoding builds a joint hypergraph with edges and hyperedges between the two blocks' stabilizers, because errors propagate deterministically through the transversal CNOT. It is solved as a mixed-integer program (Gurobi). With correlated decoding, the Bell-pair error decreases with d. With independent MWPM it increases with d, partly because of non-fault-tolerant state preparation.
- d = 3 color codes: fault-tolerant preparation by postselecting on an ancilla logical flag gives |0_L> fidelity 99.91%. GHZ state on 4 logical qubits: **72(2)% with correlated decoding (correction, no postselection)** and **99.85% with full postselection** on all stabilizers. With **partial postselection**, "syndrome events most likely to have caused algorithmic failure are discarded, given by the weight of the correlated matching in the whole algorithm"; discarding 50% of the data gives about 90%.
- Methods, "Sliding-scale error detection": a confidence threshold on the correction weight decides whether a shot is accepted. "When sliding-scale error detection is applied, in all applicable cases we also apply error correction to return to the codespace." For the [[8,3,2]] runs, the sliding scale was simply the number of detected stabilizer errors (3, 6, 24, 48 logical qubits). For 12 logical qubits, the 16 stabilizer outcomes were ranked by their measured XEB.
- [[8,3,2]] blocks have distance 2 in the Z basis and distance 4 in the X basis: "error detection (correction) capabilities for Z (X) errors". In the two-copy measurements, "Z basis measurements are corrected with an [[8,3,2]] decoder (when full error detection is not applied)".
- 12-logical-qubit IQP sampling: XEB 0.156(2) raw and 0.616(7) with full error detection. The detected data set is 23,545 of 138,626 shots (17.0% acceptance). XEB is about 0.1 for 48 logical qubits (228 logical two-qubit gates, 48 CCZ).

Bearing on this study:
- This is the direct precedent for a decoder that returns both a Pauli-frame correction and a reject flag (the interface in the brief): correct every shot and reject on a likelihood or weight threshold [lit].
- In the [[5,1,2]]_3 code, a natural threshold rule is: correct unambiguous weight-1 classes and reject ambiguous ones (Part E4) [conj on performance].
- The [[8,3,2]] runs used correction only for the error type with the larger distance [lit]. Our code has distance 2 for both types, so the analogous split is by syndrome class, not by error type [ana].

### A7. Gottesman 2016: criterion for fault tolerance in small experiments (supporting)

**Citation.** "Quantum fault tolerance in small experiments". D. Gottesman. arXiv:1610.03507 (2016); no journal reference listed. Fetched: https://arxiv.org/abs/1610.03507, full text https://ar5iv.org/abs/1610.03507.

Facts [lit]:
- Criterion: the encoded circuit must have a lower error rate than the unencoded circuit for all circuits in the family, comparing **complete** circuits that include preparation and measurement. "Leaving them out of our fault-tolerant protocol can lead to a highly deceptive impression of the robustness of the system against errors."
- The [[4,2,2]] code "can only improve error rates via post-selection", and "demonstrations that rely on post-selection can be controversial". The fair comparison is the postselected conditional error p_L/p_S against the unencoded error p_U.
- Without repeated detection, the encoded circuit wins only if p < (4+2T)/(16T² + 122T + 225): about 1.6% for T = 1, 0.79% for T = 10 and 0.12% for T = 100 time steps.
- "The physical SWAP gate would not be fault-tolerant, since an error during it could introduce errors to two qubits, which might not be detected."

Bearing on this study:
- Our LRB-D uses ideal encoding and ideal terminal readout. Gottesman's criterion would require noisy preparation and readout before claiming an encoded advantage over physical RB [lit/conj].
- The SWAP warning applies to [[4,2,2]]. In [[5,1,2]]_3, every error from SWAP(2,3) or CZ(2,3) is detectable (E2) [num].
- The long inverse block without checks is the regime in which Gottesman's bound tightens with T [conj].

### Table A: what d=2 experiments did with their syndromes [lit]

| Experiment | Code / platform | Rounds | Syndrome use | Acceptance | Reported logical error (postselected) |
|---|---|---|---|---|---|
| Linke 2017 | [[4,2,2]], 5 ions | 1 (S_x or S_z, separate runs) | reject on ancilla = 1 or odd data parity | 77.8% (S_z), 65.2% (S_x), 91.1% (prep only) | L_a-only 0.1-0.2%, both 0.2-0.3%, L_b 1.7-2.4%; L_a 0.7% after a logical X |
| Takita 2017 | [[4,2,2]], 5 transmons | 1 | reject on syndrome qubit or software S_z | 0.757-0.790 | protected 0.6-1.3%, gauge 2.3-3.2% |
| Andersen 2020 | d=2 surface code, 7 transmons | up to 10 | reject on any ancilla = 1 in any cycle, or a failed final data stabilizer | about 40-50% per cycle; about 1e-4 after 10 cycles | 3.1% (X_L), 2.6% (Z_L) per cycle |
| Marques 2022 | Surface-7, transmons | up to 15 | reject on any detected error in any cycle, plus a final data-derived check | γ ≈ 45% rejection per cycle | init 95.5-99.97%; gates 95.6-98.1% |
| Self 2024 | [[10,8,2]] iceberg, 12 ions | 0-3 mid-circuit | reject on any ancilla = -1 or final S_Z = -1 | QV test: 13.2% accepted | QV 2^8 passed |
| Bluvstein 2024 | [[8,3,2]], color codes, surface codes, atoms | 1 (transversal circuits) | correlated decoding, then optional full or sliding-scale rejection | GHZ: 50% accepted gives about 90%; 12-qubit IQP: 17% accepted at full detection | GHZ 72% (correction only) to 99.85% (full detection) |

---

## Part B. Erasure conversion and erasure qubits

### B1. Wu, Kolkowitz, Puri, Thompson 2022: erasure conversion with 171Yb Rydberg arrays

**Citation.** "Erasure conversion for fault-tolerant quantum computing in alkaline earth Rydberg atom arrays". Y. Wu, S. Kolkowitz, S. Puri, J. D. Thompson. Nat. Commun. 13, 4657 (2022). arXiv:2201.03540. DOI 10.1038/s41467-022-32094-6. Fetched: https://arxiv.org/abs/2201.03540, https://ar5iv.org/abs/2201.03540, Crossref record.

Facts [lit]:
- Qubits are encoded in the metastable 3P0 nuclear-spin levels of 171Yb. Most Rydberg-gate errors (BBR transitions, about 61%; radiative decay to the ground state) leave the qubit subspace. These are detected by fluorescence on cycling transitions (1S0-1P1 for ground-state atoms; Yb+ fluorescence after autoionization for Rydberg population). Only about 5% of decays return to the qubit subspace.
- Projected erasure fraction R_e = 0.98 of two-qubit gate errors.
- Simulation model: each two-qubit gate has a Pauli error with probability p(1-R_e), uniform over 15, or an erasure with probability pR_e. After an erasure, **both atoms are replaced by fresh atoms in the maximally mixed state**, modeled as a uniformly random Pauli from {I,X,Y,Z}^{⊗2}. Erasure detection follows every gate (XZZX surface code, d rounds plus one perfect round).
- Decoder: "The syndrome measurement results, together with the locations of the erasure errors, are decoded with weighted Union Find (UF) decoder"; UF is optimal for pure erasures (Delfosse-Zémor). Edges are weighted by ln(p') rounded to the nearest integer.
- Threshold rises from 0.937(4)% (R_e = 0) to 4.15(2)% (R_e = 0.98), and to 5.13% at R_e = 1. The logical-error exponent rises from (d+1)/2 toward d (ν = 4.35 at R_e = 0.98, d = 5). Adding p_m = 0.005 ancilla preparation and measurement error lowers the threshold to 3.80%.
- Background: "a four-qubit quantum code is sufficient to correct a single erasure error" (Grassl et al.); the surface-code erasure threshold approaches 50%.

Bearing on this study:
- This is the template for using heralds in a decoder: an erased location becomes a uniformly random Pauli at a known place, and the decoder gives it zero weight [lit].
- For the [[5,1,2]]_3 code, every single erasure is correctable (E5) [num]. A d=2 code goes from detection only (Pauli errors) to correction (heralded errors), so the gain for d=2 is categorical, not just a threshold shift [ana].
- In the qutrit DEM, an erasure would be a location with probability e, uniform over all 9 Weyl operators including the identity (or 81 for a two-qutrit gate), plus one herald detector that always fires when the location is active [conj: this is an implementation suggestion; the current DEM samples only non-identity vectors and has no herald detectors].

### B2. Kubica et al. 2023: erasure qubits in superconducting circuits

**Citation.** "Erasure qubits: Overcoming the T1 limit in superconducting circuits". A. Kubica, A. Haim, Y. Vaknin, H. Levine, F. Brandão, A. Retzker. Phys. Rev. X 13, 041022 (2023). arXiv:2208.05461 (the arXiv v1 author list omits H. Levine). DOI 10.1103/PhysRevX.13.041022. Fetched: https://arxiv.org/abs/2208.05461, https://ar5iv.org/abs/2208.05461, Crossref record.

Facts [lit]:
- Proposal: engineer qubits whose T1 events leave the computational subspace detectably. Two realizations: a dual-rail qubit (two resonant transmons; a decay to |gg> is detected through a dispersively coupled ancilla) and a g-f transmon qubit (computational states |g>, |f>; decay lands in |e>).
- Heralded-erasure-plus-Pauli (HEP) noise model: Pauli error p after preparation, idling and CNOTs; measurement error p_M; and with probability e, a **known** fully depolarizing two-qubit channel after a CNOT. Decoder: MWPM. Simulation: d rounds plus a noiseless round.
- The correctable region is about 3.5 times larger than when erasure locations are unknown. At e = 0.01 (p_M = 2p/3), Pauli rates up to p = 0.0051 are tolerated, 5.2 times the standard value.
- Imperfect heralds: false negatives (probability q-) act like Pauli errors (p becomes p + e q-). False positives (probability q+) affect the matching weights (e becomes e + q+).
- Dual-rail estimates: e ≈ 6e-3 (2 × 110 ns gates and a 400 ns |gg> measurement at T1 = 100 µs); p ≈ 1.6e-4; q- ≈ 1e-3, contributing 6e-5.

Bearing on this study:
- The false-positive and false-negative bookkeeping (q+ adds to erasure weight, q- adds to Pauli rate) carries over directly to a qutrit DEM with herald detectors [ana].
- A g-f-type construction uses up a transmon level. A transmon qutrit has no spare level below |3>, so the natural qutrit erasure herald is leakage to |3> or above detected at readout. Conjecture: such a herald would be weaker and slower than dual-rail checks [conj].

### B3. Levine et al. 2024: dual-rail erasure qubit with tunable transmons

**Citation.** "Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons". H. Levine, A. Haim, J. S. C. Hung, ..., F. G. S. L. Brandão, A. Retzker, O. Painter (113 authors). Phys. Rev. X 14, 011051 (2024). arXiv:2307.08737. DOI 10.1103/PhysRevX.14.011051. Fetched: https://arxiv.org/abs/2307.08737, https://arxiv.org/html/2307.08737, Crossref record.

Facts [lit]:
- Logical subspace (|01> ± |10>)/√2 of two resonantly coupled transmons. A transmon T1 decay goes to |00>, the erasure. Erasure lifetime 29.3(4) µs. Within the subspace (postselected), T1 = 906(15) µs and T2 (CPMG) = 543 µs to 1.25 ms.
- Single-qubit RB with 11 interleaved erasure checks: erasure probability 2.19(2)e-3 per X90 and residual (non-erasure) error 5.06(6)e-5 per X90, an erasure bias of 43(1).
- The erasure check is a 540 ns conditional π pulse on an ancilla that is resonant only if the dual rail is in |00>, followed by ancilla readout; 880 ns in total. False positives: 0.58(1)% for |1_L> and about 0.8% for |+_L>. False negatives: 1.54(1)%, limited by ancilla T1 during readout. Probability of a missed erasure ≈ 2.9% × 1.54% ≈ 0.04% per check. Check-induced dephasing: under 0.1% per check. Benchmark targets were 1% erasure and 0.1% Pauli error per gate.
- "Postselection is used here only to characterize the non-erasure error rates and would not be needed in a concatenated surface code architecture."

Bearing on this study:
- These are realistic herald numbers for a superconducting erasure check [lit]. For a correction-based qutrit LRB with heralds, a sensitivity scan should include q+ ≈ 0.5-1%, q- ≈ 1-2%, and a per-check dephasing of up to 0.1% [conj].
- A false-positive rate near 1% per check is comparable to the per-round rejection of LRB-D at p ≈ 1.5e-4 (63 p per round). With heralds used for correction rather than rejection, false positives would cost fidelity, not acceptance [conj].

### B4. Chou et al. 2024: dual-rail cavity qubit with erasure-detected logical measurement

**Citation.** "A superconducting dual-rail cavity qubit with erasure-detected logical measurements" (arXiv title: "Demonstrating a superconducting dual-rail cavity qubit with erasure-detected logical measurements"). K. S. Chou, T. Shemma, H. McCarrick, et al. (34 authors in the journal record), S. M. Girvin, S. H. Moseley Jr., R. J. Schoelkopf. Nat. Phys. 20, 1454-1460 (2024). arXiv:2307.03169. DOI 10.1038/s41567-024-02539-4. Fetched: https://arxiv.org/abs/2307.03169, PDF https://arxiv.org/pdf/2307.03169, Crossref bibliographic search result.

Facts [lit]:
- A single photon shared by two 3D cavities: |0_L> = |01>, |1_L> = |10>. Outcomes "00" and "11" are erasures. This is **end-of-line** erasure detection (built into the final measurement), not a mid-circuit check.
- Single-round logical measurement: misassignment (1.8 ± 0.3)e-4; erasure fraction (6.03 ± 0.05)e-2; **leakage-detection error (false negatives) (7.7 ± 0.3)e-3**, so over 99% of decays are converted into erasures. Two rounds that must agree: misassignment (4 ± 2)e-5 and leakage-detection error (1.2 ± 0.1)e-3, at the cost of an erasure fraction of 0.17.
- Idling errors: decay about 0.2% per µs, phase flips about 6 times less often (p_φ ≈ 0.019-0.023% per µs), bit flips at least 140 times less often.
- Logical outcomes are computed after discarding the assigned erasures.

Bearing on this study:
- The two-round agreement rule is a measurement-level analogue of LRB-D: it trades misassignment against erasures (rejections) [lit]. It shows that a herald-plus-rejection layer can be tuned independently of the code-level decoder [conj].
- For bosonic qudits (one of the brief's hardware options), a cavity-based herald of this type exists for dual-rail qubits. It is not demonstrated for qutrit encodings in this paper [lit/conj].

### B5. Ma et al. 2023: high-fidelity gates and mid-circuit erasure conversion in 171Yb

**Citation.** "High-fidelity gates and mid-circuit erasure conversion in an atomic qubit" (arXiv title: "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit"). S. Ma, G. Liu, P. Peng, B. Zhang, S. Jandura, J. Claes, A. P. Burgers, G. Pupillo, S. Puri, J. D. Thompson. Nature 622, 279-284 (2023). arXiv:2305.05493. DOI 10.1038/s41586-023-06438-1. Fetched: https://arxiv.org/abs/2305.05493, https://arxiv.org/html/2305.05493, Crossref record.

Facts [lit]:
- Metastable 171Yb nuclear-spin qubit: single-qubit gates F = 0.9990(1), CZ F = 0.980(1). Metastable lifetime 2.96(12) s.
- Fast mid-circuit imaging on 1S0-1P1 detects atoms that decayed to the ground state. Single-qubit RB with an image every 50 gates: error 1.0(1)e-3 overall and 4.5(3)e-4 conditioned on no detection, so **56(4)% of errors are converted into erasures**.
- Detection disturbance on undecayed qubits: P_d = 7e-6 from extra decay during imaging; scattering below 1e-6 per imaging time. Waiting for Rydberg population to decay extends detection to 420 µs, with P_d = 1.4e-4.
- CZ with imaging after every two gates: 2.0(1)e-2 overall and 1.3(1)e-2 conditioned on no detection, so **about 33% of errors are converted**. The model predicts 60% of errors are leakage; only half of the Rydberg leakage is detected. Erasures are biased by state: p11/p00 > 15(9).
- In a code, erased atoms are to be replaced from a reservoir and the code state restored "after measuring the error syndromes".

Bearing on this study:
- Ma used heralds only to condition (excise) in benchmarking, which is the erasure analogue of LRB-D [lit]. A mixed scheme (correct heralded errors, reject unheralded syndromes) is the natural next step for a d=2 code [conj].
- An erasure fraction of 33-56% is far from the R_e = 0.98 assumed in Wu 2022 [lit]. A qutrit study should scan R_e from 0.3 to 1 [conj].

### B6. Grassl, Beth, Pellizzari 1997: codes for the quantum erasure channel (supporting)

**Citation.** "Codes for the Quantum Erasure Channel". M. Grassl, T. Beth, T. Pellizzari. Phys. Rev. A 56, 33-38 (1997). arXiv:quant-ph/9610042. DOI 10.1103/PhysRevA.56.33. Fetched: https://arxiv.org/abs/quant-ph/9610042 (abstract only).

Facts [lit]:
- A four-qubit code that encodes one qubit and corrects one erasure. Four qubits are necessary and sufficient, whereas five are needed for errors at unknown positions.

Bearing on this study:
- This is the founding statement that a d=2 code corrects one erasure [lit]. The [[5,1,2]]_3 version is checked in E5: all single erasures, and 6 of 10 pairs, are correctable [num].

### B7. Stricker et al. 2020: deterministic correction of qubit loss (supporting)

**Citation.** "Experimental deterministic correction of qubit loss" (arXiv title: "Deterministic correction of qubit loss"). R. Stricker, D. Vodola, A. Erhard, L. Postler, M. Meth, M. Ringbauer, P. Schindler, T. Monz, M. Müller, R. Blatt. Nature 585, 207-210 (2020). arXiv:2002.09532. DOI 10.1038/s41586-020-2667-0. Fetched: https://arxiv.org/abs/2002.09532, PDF https://arxiv.org/pdf/2002.09532.

Facts [lit]:
- 40Ca+ ions, a 1 + 4-qubit "minimal instance of the surface code": stabilizers Z1Z2, Z1Z3, X1X2X3X4, with logical T_Z = Z1Z4 and T_X = X4. T_X has weight 1, so this demonstration code is not distance 2 against Pauli errors.
- A QND detection unit maps loss of qubit 1 onto an ancilla: detection efficiency 96.5(4)%, false positives 3(1)%, false negatives 1(1)%. With no loss, the detection unit has process fidelity 90(2)% at about 20% loss rate.
- On a detected loss, real-time feed-forward measures the shrunk stabilizer X2X3X4 and applies a Pauli-frame update for a -1 outcome. The code is rebuilt on the remaining qubits, with the merged Z stabilizer Z2Z3.

Bearing on this study:
- This is the experimental precedent for herald, then code deformation (merged and shrunk stabilizers), then Pauli-frame update [lit]. For a lost or erased qutrit in [[5,1,2]]_3, the same super-stabilizer idea applies: products of checks that avoid the erased qutrit stay valid [ana: products of stabilizers that act trivially on the erased qutrit commute with any error there].

### Table B: heralded-error metrics [lit]

| Paper | Herald | Conversion or bias | False positive | False negative / missed | Disturbance from the herald | Use of the herald |
|---|---|---|---|---|---|---|
| Wu 2022 (proposal) | fluorescence, Yb atoms and ions | R_e = 0.98 (projected) | not modeled (main text) | not modeled | not modeled | weighted UF decoder; erased pair reset to a random Pauli |
| Kubica 2023 (proposal) | dispersive |gg> check | e ≈ 6e-3, p ≈ 1.6e-4 | q+ adds to e | q- ≈ 1e-3 adds e·q- to p | measurement dephasing about 1e-4 | MWPM with erasure-aware weights |
| Levine 2024 | conditional π pulse + readout, 880 ns | bias 43 (2.19e-3 vs 5.06e-5 per X90) | 0.58% (|1_L>), about 0.8% (|+_L>) | 1.54%; missed about 0.04% | under 0.1% per check | postselection (characterization only) |
| Chou 2024 | end-of-line cavity measurement | decay 6 times phase flips, at least 140 times bit flips | erasure fraction 6% (17% with two rounds) | 7.7e-3 (1.2e-3 with two rounds) | n/a (end-of-line) | discard erasures |
| Ma 2023 | fast 399 nm imaging | 56% (1q), 33% (CZ) | small at the chosen threshold | not quantified separately | P_d = 7e-6 (20 µs); 1.4e-4 (420 µs) | conditioning (excision) |
| Stricker 2020 | QND loss unit | n/a | 3% | 1% | process fidelity 90% (no loss) | code reconstruction + Pauli frame |

---

## Part C. Leakage in superconducting QEC

### C1. McEwen et al. 2021: removing leakage-induced correlated errors

**Citation.** "Removing leakage-induced correlated errors in superconducting quantum error correction". M. McEwen, D. Kafri, Z. Chen, J. Atalaya, K. J. Satzinger, et al. (51 authors), R. Barends. Nat. Commun. 12, 1761 (2021). arXiv:2102.06131. DOI 10.1038/s41467-021-21982-y. Fetched: https://arxiv.org/abs/2102.06131, PDF https://arxiv.org/pdf/2102.06131, Crossref record.

Facts [lit]:
- Multi-level reset (MLR): adiabatic swap of the qubit's excitations into a lossy readout resonator. It resets |1>, |2> and |3> to |0> in about 250 ns with error about 1e-3, insensitive to parameters. Because it removes all quantum data, it is applied only to measure qubits, right after readout.
- 21-qubit bit-flip (repetition) code on Sycamore. Without reset, measure qubits accumulate more |2> population than data qubits, so readout is a significant source of leakage. With MLR, γ↓ (leakage decay) of measure qubits rises 40-fold, and that of data qubits 2.4-fold, which is evidence of leakage transport from data to measure qubits through |21>-|03> resonances of the CZ.
- Injected leakage produces a pair of detection events, then a long tail of correlated detection events. MLR removes the tail. The p_ij matrices (pairwise detection-event correlations, used to set matching weights) show long-range correlations over more than 10 rounds without reset; these are mostly removed with reset.
- Error suppression factor Λ_bit = 1.98 without reset and 2.80 with reset (30 rounds, 5-21 qubits); Λ_bit is more stable over time with reset.

Bearing on this study:
- MLR resets every level |1>, |2>, |3> to |0>, so it applies unchanged to transmon-qutrit ancillas, matching the brief's ancilla reset layer [lit/ana].
- The p_ij estimator, which recovers edge probabilities from detection-event correlations, is a hardware-sustainable way to calibrate a decoder from data. For Z_3 detectors it would need a q-ary generalization [conj].
- Leakage creates exactly the time-correlated, non-Markovian errors relevant to study question 2. The SI1000 Pauli model cannot show them [ana: the model has independent locations].

### C2. Miao et al. 2023: overcoming leakage in QEC

**Citation.** "Overcoming leakage in quantum error correction" (arXiv title: "Overcoming leakage in scalable quantum error correction"). K. C. Miao, M. McEwen, J. Atalaya, D. Kafri, L. P. Pryadko, A. Bengtsson, et al. (118 authors in the journal record), Y. Chen. Nat. Phys. 19, 1780-1786 (2023). arXiv:2211.04728. DOI 10.1038/s41567-023-02226-w. Fetched: https://arxiv.org/abs/2211.04728, https://arxiv.org/html/2211.04728, Crossref record.

Facts [lit]:
- Leakage injected on the central data qubit of a d=3 surface code decays with a constant of about 4.4 cycles and spreads to neighbors. A leaked higher-frequency qubit imparts a spurious phase of about 0.65π on its CZ partner. Leakage transport through the CZ: about 18% between |30> and |12>.
- Data-qubit leakage removal (DQLR): after MLR on the measure qubits, a LeakageISWAP (an iSWAP in the |11>-|20> subspace) moves data-qubit leakage to the reset measure qubit, which is then reset again.
- Result: **ten-fold reduction** in steady-state data-qubit leakage, and average leakage **below 1e-3** over the device. Leakage dynamics are confined to one cycle.
- d=21 bit-flip code, 60 cycles: with no reset, "even small amounts of injected leakage population less than 1% cause the logical error probability to rise above 40%". Pauli errors can be injected up to about 5% before similar logical error. d=3 surface code: DQLR gives the lowest logical error at all injected leakage levels, despite the longer cycle.

Bearing on this study:
- This quantifies how much worse leakage is than Pauli error under **correction** [lit]. A correction-based LRB on transmon qutrits would need a qutrit DQLR analogue: moving |3> population of a data qutrit to an ancilla without disturbing |0>, |1>, |2> [conj]. No such gate is reported here.
- Leakage removal turns leakage into ordinary errors that the decoder sees. It is not a herald [lit].

### C3. Varbanov et al. 2020: leakage detection for a transmon surface code

**Citation.** "Leakage detection for a transmon-based surface code". B. M. Varbanov, F. Battistel, B. M. Tarasinski, V. P. Ostroukh, T. E. O'Brien, L. DiCarlo, B. M. Terhal. npj Quantum Inf. 6, 102 (2020). arXiv:2002.07119. DOI 10.1038/s41534-020-00330-w. Fetched: https://arxiv.org/abs/2002.07119, PDF https://arxiv.org/pdf/2002.07119, Crossref record.

Facts [lit]:
- Full density-matrix simulation of Surface-17 with a CZ leakage model (L1 = 0.125% per CZ by default, with leakage-conditional phases). Leakage pushes the logical error rate above memory break-even, for both the MWPM decoder and the density-matrix decoding upper bound.
- Signatures: a leaked data qubit makes its neighboring stabilizers anticommute and randomizes them. Data leakage lasts about 9 cycles and ancilla leakage about 11 cycles on average. A leaked ancilla effectively disables its check and spreads phase errors. A leaked data qubit "effectively" reduces the code distance from 3 to 2 (weight-6 super-checks can be used for decoding).
- Detection: one two-state hidden Markov model (HMM) per qubit, fed with defects on neighboring checks and, for ancillas, the analog readout signal. Average HMM optimality (area under the precision-recall curve relative to the ideal) is about 74% for data qubits and about 35% for bulk ancillas from defects alone; boundary ancillas have no ability from defects alone. Adding analog readout helps; the experimental |1>/|2> discrimination fidelity was 88.4%.
- **Postselecting on HMM-detected leakage restores sub-break-even logical error while discarding about 47% of the data.** The authors propose leakage-aware decoding and targeted LRUs.

Bearing on this study:
- For a qutrit transmon code, leakage to |3> would show up the same way, as persistent randomized checks near the leaked qutrit [conj].
- In LRB-D, leaked shots are rejected automatically because they produce nontrivial syndromes [ana for the syndrome effect, conj for the hardware magnitude].
- In correction mode, a d=2 code with one leaked data qutrit has its distance effectively reduced to 1 for errors in that region unless the leak is heralded. If it is heralded, the leaked qutrit becomes an erasure, which is correctable (E5) [conj/num].

### C4. Marques et al. 2023: all-microwave leakage reduction units

**Citation.** "All-microwave leakage reduction units for quantum error correction with superconducting transmon qubits". J. F. Marques, H. Ali, B. M. Varbanov, M. Finkel, H. M. Veen, S. L. M. van der Meer, S. Valles-Sanclemente, N. Muthusubramanian, M. Beekman, N. Haider, B. M. Terhal, L. DiCarlo. Phys. Rev. Lett. 130, 250602 (2023). arXiv:2302.09876. DOI 10.1103/PhysRevLett.130.250602. Fetched: https://arxiv.org/abs/2302.09876, https://ar5iv.org/abs/2302.09876, Crossref record.

Facts [lit]:
- An LRU (after Battistel et al.) drives |f> → |g> through the readout and Purcell resonators. It is extended with an **h-LRU** for the third excited state |h> = |3>, which runs simultaneously with the f-LRU at no extra time and without affecting the |f> removal fraction. Removal above 99% in 220 ns. The only effect on the qubit subspace is an AC-Stark shift (71 kHz), corrected by a virtual Z. LRU average gate fidelity 98.(9)%, against 99.(2)% for idling the same 220 ns.
- In a weight-2 parity check repeated for 50 rounds (500 ns cycle without LRU, 720 ns with), without LRUs the detection probability rises about 8% over 50 rounds and steady-state leakage of D1 and the ancilla reaches about 10%. With LRUs, the rise is limited to 2% within about 10 rounds and leakage is at most about 1% for all transmons. Leakage above |f> in the ancilla is caused by readout.
- "A more powerful operation called 'heralded leakage reduction' would both reduce and herald leakage, leading to a so-called erasure error". This LRU itself does not herald.

Bearing on this study:
- The h-LRU acts on |3>, the first leakage level of a transmon **qutrit** [lit]. Using it alone, without the f-LRU, as a qutrit LRU has not been tested: the paper does not report whether the h-drive preserves coherence of superpositions involving |2> [conj].
- Readout-induced leakage to higher levels matters for the four sequential ancilla measurements per round in this circuit [conj].

---

## Part D. Flags

### D1. Chao and Reichardt 2018: quantum error correction with two extra qubits

**Citation.** "Quantum error correction with only two extra qubits". R. Chao, B. W. Reichardt. Phys. Rev. Lett. 121, 050502 (2018). arXiv:1705.02329. DOI 10.1103/PhysRevLett.121.050502. Fetched: https://arxiv.org/abs/1705.02329, https://arxiv.org/html/1705.02329, Crossref record.

Facts [lit]:
- A flag ancilla in |+>, coupled to the syndrome ancilla by CNOTs around the middle data couplings, catches syndrome-ancilla faults that would spread to data errors of weight 2 or more. Flagged data errors must have distinct syndromes so that they can be corrected. This gives [[5,1,3]] fault-tolerant EC with 7 qubits, and [[7,1,3]] and [[15,7,3]] with 2 extra qubits (7 logical qubits on 17 physical).
- Section V, [[n,n-2,2]] detection: "extracting a syndrome with a single ancilla qubit is not fault tolerant because, for example, a Z fault at either location ... results in an undetectable logical error. With a flag qubit, the circuit is fault tolerant". Interpretation: the ancilla is encoded in a two-qubit Z-error-detecting code.
- Overlapping flags, one per CNOT, can localize faults closely; shared flags can serve several syndromes.
- Simulations (depolarizing noise, no idle errors): two-qubit EC is comparable to Shor and Steane EC for [[5,1,3]] and [[7,1,3]]; Steane is better for [[15,7,3]].

Bearing on this study:
- The flag requirement in Section V arises from **weight-n** checks, where a mid-check fault leaves a weight-2 or larger residual that can be a logical [lit]. Our checks have weight 3 and distance 2. Every hook residual is weight-1 equivalent and therefore detectable (E1) [num], so flags add nothing at first order [ana].
- The remaining first-order hole is at the end of the circuit (E3), and a final data-derived check fixes it, not a flag [num/ana].

### D2. Chamberland and Beverland 2018: flag fault tolerance for arbitrary distance

**Citation.** "Flag fault-tolerant error correction with arbitrary distance codes". C. Chamberland, M. E. Beverland. Quantum 2, 53 (2018). arXiv:1708.02246. DOI 10.22331/q-2018-02-08-53. Fetched: https://arxiv.org/abs/1708.02246, https://arxiv.org/html/1708.02246, Crossref record.

Facts [lit]:
- Defines t-flag circuits and the flag t-FTEC conditions. Flag 1-FTEC condition: any two data errors from a single fault that raises a given flag must be distinguishable or logically equivalent. The protocol repeats syndrome extraction with flag circuits until a syndrome repeats with no flags, and switches to non-flag circuits after a flag or a disagreement. Generalized to d = 5 and arbitrary d, with infinite code families satisfying the conditions.
- They note that the surface code satisfies only the first FTEC condition because of hook errors.
- Pseudo-thresholds (idle error = p): flag-EC [[5,1,3]] 7.09e-5 (7 qubits, 64-88 time steps); [[7,1,3]] 3.39e-5 (9 qubits); d=3 surface code 3.29e-4 (17 qubits); Steane-EC [[7,1,3]] 6.29e-4. Flag-EC [[19,1,5]] 1.14e-5 (22 qubits, 504-960 time steps), against 9.41e-4 for the d=5 surface code. "This is primarily due to the large circuit depth in flag-EC protocols since idle qubits locations significantly outnumber other gate locations."

Bearing on this study:
- Flag schemes save qubits at the cost of depth, so they lose most when idle noise is large [lit]. SI1000 charges 2p idle depolarizing per measurement layer on every other wire, so extra flag measurement layers would add substantial idle noise [ana for the noise accounting, conj for the net effect].

### D3. Durso-Sabina 2022: flag fault-tolerant error correction with qudits (thesis)

**Citation.** "Flag Fault-Tolerant Error Correction with Qudits". E. Durso-Sabina. MSc thesis, University of Waterloo (supervisor C. Senko), 2022 (title page dated 2021). Handle http://hdl.handle.net/10012/17931. Fetched: https://uwspace.uwaterloo.ca/items/888cc1a3-bfdd-4b7c-88fa-a1c3ff258632/full and the PDF https://dspacemainprd01.lib.uwaterloo.ca/server/api/core/bitstreams/aad2ba2c-0e69-4384-83bf-fd112ceebc1a/content. A search for a peer-reviewed or arXiv version found none; the thesis is the qudit flag scheme cited here.

Facts [lit]:
- Extends the Chao-Reichardt flag procedure for the five-qubit code to the five-qudit code [[5,1,3]]_d for any prime d. Stabilizers: X Z Z^-1 X^-1 I and its cyclic shifts.
- Circuit translation: H → F or F^-1 (Fourier) and CNOT → SUM or SUM^-1. Gates are inverted where the stabilizer has inverted powers. "The flag must be 'turned off' by a SUM^-1 gate since, for qudits of d > 2, SUM ≠ SUM^-1." For qubits, a Z fault copied twice onto the flag cancels; for qudits, the second coupling must be the inverse so that the power cancels.
- Uses qudit propagation rules, for example SUM^-1(I ⊗ Z) = (Z ⊗ Z)SUM^-1 and SUM^-1(X ⊗ I) = (X ⊗ X^-1)SUM^-1. A worked example shows that a residual X^j-type hook becomes "a weight one error multiplied by a stabilizer".
- Chapter 4 generalizes flag FTEC for any stabilizer code to any qudit stabilizer code, with the number of ancillas scaling with distance rather than stabilizer weight.

Bearing on this study:
- If a future qutrit code with higher-weight checks needs flags, this is the construction: SUM/SUM^-1 pairs on the flag, with every X^a power (a = 1, 2) caught [lit].
- For [[5,1,2]]_3 with weight-3 checks, flags are unnecessary (E1) [num].

---

## Part E. Checks on the [[5,1,2]]_3 LRB circuit (this task)

### E1. Hook errors of the single-ancilla checks are weight-1 equivalent

- In the X-check S_X1 (H 7, CNOT 7→0, CNOT 7→1, CNOT_INV 7→2, H_INV 7, M 7), an X^c fault on ancilla 7 after CNOT 7→0 propagates to X1^c X2^-c = S_X1^c · X0^-c [ana].
- Likewise for S_X2 (residual X1^-c X4^-c after CNOT 8→3), S_Z1 (Z-type residual on {1,3} after CNOT 0→5) and S_Z2 (residual on {2,4} after CNOT 1→6). By enumeration, each residual has minimum weight 1 modulo stabilizers and a nonzero syndrome [num].
- Since d = 2, every weight-1 error is detectable [num: the minimum nontrivial logical weight is 2].
- Hence no single hook fault causes an undetectable logical error, and flags are not needed [ana].

### E2. The data-data gate pair {2,3}

- All 80 nontrivial Weyl operators supported on {2,3} have nonzero syndrome, and the 81 operators on {2,3} have 81 distinct syndromes [num].
- The logical gates' only two-qutrit gates act on the pair (2,3) (SWAP, CZ) [code, from the brief]. Logical gates map stabilizers to stabilizers. Therefore a single fault in any logical-gate block (single-qutrit or {2,3} error) remains detectable through all later gates, including the unchecked inverse block [ana].
- This contrasts with Gottesman's [[4,2,2]] SWAP warning and with the iceberg MS gates (Self 2024) [lit].

### E3. The final-round hole explains the m = 0 infidelity of LRB-D

- Exact single-fault enumeration on the compact DEM (Appendix A) of the m = 0 circuit. The undetected logical-flip probability is **4.95 p**, from five locations: N2 after CNOT 7→0 restricted to no Z-part on ancilla 7 (18/80 p = 0.225 p); idle 2p on qutrit 0 in the M7 layer (1.5 p); N2 after CNOT 8→3 (0.225 p); idle 2p on qutrits 0 and 3 in the M8 layer (1.5 p each) [num]. The same value 4.95 p holds for m = 2 and m = 4, for two Clifford sequences each [num]. With every round used, no other first-order undetectable logical faults exist anywhere in the circuit [num, for the circuits checked].
- These faults put a Z-type error on readout qutrit 0 or 3 after its last X-check. Nothing measures them before the ideal X-basis readout of wires 0 and 3 [ana; the final round is followed directly by the terminal readout, with no reset layer, `src/lrb/circuit_generator.py`].
- A logical shift of ±1 with total probability ε gives fidelity |Σ_k ω^k P(k)| = 1 - 3ε/2. The prediction is therefore **1 - F = 7.425 p** at first order [ana].
- Baseline LRB-D data (`results/LRB/<idx>/unif_check_data/1.csv`, m = 0) [num]:

| p | 1 - F(m=0) | (1 - F)/p | rejection(m=0) | rejection/p |
|---|---|---|---|---|
| 1e-4 | 7.462e-4 | 7.46 | 0.0063 | 63 |
| 2e-4 | 1.495e-3 | 7.47 | 0.0125 | 62.5 |
| 5e-4 | 3.710e-3 | 7.42 | 0.0310 | 62.0 |
| 1e-3 | 7.497e-3 | 7.50 | 0.0611 | 61.1 |
| 1e-2 | 8.289e-2 | 8.29 | 0.4615 | 46 |

  The first-order detection probability from the enumeration is 63.15 p, matching the rejection column at small p [num].
- Consequences:
  - No decoder can remove this term, because the events leave no syndrome [ana].
  - A final X-basis readout of all five data qutrits, compared with the round-m X-checks (the d=2 experiments' standard, and what `const 0` circuits do), detects all five fault types at first order [ana]. With that readout, single Z errors on 0 or 3 change the data-derived S_X1 or S_X2 value.
  - Whether that readout is noisy or ideal changes only second-order terms [ana: a single readout shift on wire 0 or 3 also changes the data-derived stabilizer].
  - Prediction: m = 0 infidelity falls from 7.4 p to O(p^2), which raises the LRB intercept b without changing f at first order [conj until simulated].

### E4. Syndrome structure of weight-1 errors and partial correction

- Of the 40 weight-1 errors (5 qutrits × 8), **24 lie in unambiguous syndrome classes** (all weight-1 members logically equivalent) and **16 in 8 ambiguous pairs** that differ by a logical [num]:
  - X0^a with X3^-a (X0X3 = logical X)
  - X2^a with X4^-a (X2X4 ≡ logical X)
  - Z0^b with Z2^-b (logical Z)
  - Z3^b with Z4^-b
- Unambiguous: all 8 errors on the center qutrit 1, and the 16 mixed errors X^aZ^b (a, b ≠ 0) on corner qutrits [num].
- Under uniform single-qutrit depolarizing noise, 60% of single-qutrit data errors are first-order correctable from a perfect syndrome. A joint X/Z decoder is required, because separate X and Z decoding sees each part of a mixed corner error as ambiguous [ana].
- Circuit-level timing (errors between the Z-block and X-block of a round) splits a mixed error's signature across rounds. Decoders therefore have to work on the space-time detector graph [ana]. The resulting correctable fraction in the circuit has not been computed here [conj].

### E5. Erasure capacity of [[5,1,2]]_3

- Every single erased qutrit is correctable: the 9 operators on any one qutrit have distinct syndromes [num; also ana, because d = 2 means no weight-1 logical].
- Erased pairs: {0,1}, {0,4}, {1,2}, {1,3}, {1,4}, {2,3} are correctable. {0,2}, {0,3}, {2,4}, {3,4} are not, since each supports a weight-2 logical [num].
- In particular, a heralded erasure on SWAP(2,3) or CZ(2,3) that erases both qutrits (the Wu 2022 model) is correctable [num].

### E6. Terminal-round Z-checks do not affect the X-basis logical at first order

- No single fault whose only signature is terminal-round Z-check detectors (ancillas 5, 6) flips the logical. Checked for m = 0, and for m = 2 and m = 4 with two sequences each [num]. Such events have first-order probability 15.6 p (m = 0) and 25.7-28.8 p (m = 2, 4) [num].
- Reason: Z-checks precede X-checks in each round. A terminal-round fault that fires only Z-checks has a final data error with no Z-component on qutrits 0 and 3 modulo stabilizers, and X-type errors do not change the X-basis readout [ana].
- A policy that ignores terminal-round Z-check outcomes accepts these events at no first-order fidelity cost [num]. Its second-order cost and its interaction with intermediate rounds are not known [conj].

---

## Appendix A. Computations (reproducibility)

All scripts are in the session scratchpad
`<local scratch, not kept>`
(this location may be cleaned later; the logic is summarized below). Runtime is seconds, after `module load python/3.11`.

1. `code_checks.py`: symplectic Z_3 representation of the code from the brief (S_Z1 = Z0 Z1^-1 Z3^-1, S_Z2 = Z1 Z2 Z4^-1, S_X1 = X0 X1 X2^-1, S_X2 = X1^-1 X3 X4^-1, X_L = X0X3, Z_L = Z0Z2). Commutation exponent: Σ_i (x_i z'_i - z_i x'_i) mod 3. It enumerates:
   - the 81-element stabilizer group and the minimum logical weight (2);
   - the syndrome classes of weight-1 errors and their logical-equivalence status (E4);
   - errors on {2,3} (E2);
   - hook residuals (E1).
2. `erasure_pairs.py`: for each erased set Q, checks that operators supported on Q with equal syndrome differ by a stabilizer (E5).
3. `first_order_m0.py <worktree>/src <circuit.chp> <p>`: builds `CompactDetectorErrorModel` (`src/lrb/dem_simulation.py`) for one baseline circuit (`experiments/LRB/<cliff>/4/<depth_idx>.chp`, p = 1e-3, read-only). Core logic:

   ```
   for loc in dem.locations:
       for v, cp in fault_vectors(loc):   # N1 'd': 8 nonzero (a,b) uniform; 'f': (a,0), a=1,2;
                                          # 'p': (0,b); N2: 80 nonzero 4-vectors uniform or prob_dist
           det = sum_c v[c] * loc.component_terms[c].detector   (mod 3)
           log = sum_c v[c] * loc.component_terms[c].logical    (mod 3)
           w = loc.probability * cp / p   # first-order coefficient
           accumulate: detected (any det != 0), undetected_logical (det == 0 and log != 0),
                       only terminal-round w5/w6 detectors fired (and whether log != 0)
   ```

   Output for m = 0: detected 63.15 p, undetected_logical 4.95 p, only-terminal-Z-check 15.575 p (none with a logical flip). For m = 2 and 4 (sequences 0 and 7): undetected_logical 4.95 p in all four circuits.
4. Baseline comparison: read `Fidelity averages` and `Rejected Runs` (first depth, m = 0) from `results/LRB/<p_idx>/unif_check_data/1.csv` in the read-only run folder.
