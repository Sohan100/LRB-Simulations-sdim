# Verification report: lit/07-hardware-realtime.md

Reviewed file: `studies/correction-decoders/qubit/lit/07-hardware-realtime.md`
(topic: mid-circuit measurement, reset, Pauli frames, real-time decoding latency).
Verifier: independent citation and claim verifier. Date: 2026-10-28.

How it was checked. In this session I downloaded every arXiv abstract page
(19 of 19), the Crossref record of every journal DOI (16 of 16), the full text
of every arXiv paper (arXiv HTML, ar5iv, or the PDF converted to text), and the
five vendor pages (3 IQM docs pages, the AWS blog, the IQM press release). I
then searched each source for the quoted phrases and numbers. I also read
`src/lrb_qubit/code.py` and `src/lrb_qubit/circuits.py` (read only) and
recomputed the document's arithmetic. Nothing outside this report was written,
and no git command was run.

Labels: a source confirmation is **from literature** (the source is named).
Re-derivations are **verified analytically**, recomputed numbers are
**verified numerically**, and my own judgements are marked **conjecture**.

## 1. Verdict summary

- **Citations.** All 22 cited works exist. All 19 arXiv IDs resolve to the
  stated titles and first authors, and all 16 DOIs resolve to the stated
  journal, volume and pages. No hallucinated paper and no wrong identifier was
  found. One metadata slip: Abdurakhimov2024 has 114 authors on arXiv, not 113.
  Two cosmetic notes: the PRX title of Moses2023 is hyphenated ("A Race-Track
  Trapped-Ion Quantum Processor"), and the Nature author list of Google2025 adds
  D. A. Abanin as second author.
- **Claims.** I checked the 31 key findings of the structured summary: 26 are
  **supported**, 5 are **partially supported** and none is unsupported. I also
  checked 7 statements that appear only in the document body: 2 are supported
  and 5 partially supported.
- **Corrections needed, most important first.**
  1. *Hothem2025 "4-26x".* The 4-5x factor is the total QIRB layer error rate
     r_Omega against calibration-based simulation. The MCM error itself,
     epsilon_MCM, is 26x the prediction at n = 15. The range "MCM crosstalk 4-26x"
     mixes these two quantities (findings 14 and 27, I6, platform table).
  2. *Muniz2025 MCM cycle.* The cycle is not "in-place": ancillas are moved from
     the register to a measurement zone (MZ) and imaged with a global beam. It
     is also not "about 18 ms". Each of the two 7 ms images is followed by 4 ms
     of cooling plus 300 us of optical pumping (the paper books "0.006 loss per
     MCM cycle" for two cooling steps). That is at least about 22.6 ms before
     moves and hand-offs (finding 22, platform table).
  3. *Moses2023 shot times.* Table I spans 0.18 s to 20.6 s per shot, the top
     value being HoloQUADS with 129 measurements. The range "0.18-5.4 s" leaves
     that circuit out, and the document's own citation note says "0.18-20 s".
     The conclusion survives and is even stronger (finding 29, I8).
  4. *const = 0 is a different circuit.* It has no check rounds (CONTEXT). So
     "LRB-D under every const/unif policy" is not post-processing of the same
     data. That holds for every const >= 1 and unif policy, but const = 0 needs
     its own runs (finding 25, I2).
  5. Smaller wording issues:
     - Moses2023 says conveyor-belt ions see "a similar level of crosstalk" as
       gate-zone ions, even though hiding is not applied to them.
     - The Yang2026 quotation is not verbatim.
     - Barber2025 results come from simulated syndromes and a signed-off,
       untaped ASIC design, not from qubit hardware (I12).
     - The Halocene release promises "a modular decoder architecture", not
       explicit in-loop real-time decoding.
     - RyanAnderson2021 calls its device "a ten qubit QCCD", not "H1".
     - In `build_lrb_circuit` the terminal check round has no reset, so it has
       4 layers, not 5.

## 2. Citation metadata (all fetched in this session)

| Key | Exists / ID ok | Metadata ok | Notes and corrections |
|---|---|---|---|
| Knill2005 | yes: arXiv quant-ph/0410199, DOI 10.1038/nature03350 | yes | arXiv title "Quantum Computing with Very Noisy Devices" (v1 25 Oct 2004). Crossref: "Quantum computing with realistically noisy devices", Nature 434(7029), 39-44 (2005). The document's dual title is correct. |
| Terhal2015 | yes: 1302.3428, 10.1103/RevModPhys.87.307 | yes | RMP 87(2), 307-346 (2015); B. M. Terhal is the only author. |
| Chamberland2018 | yes: 1704.06662, 10.22331/q-2018-01-04-43 | yes | Quantum 2, 43 (2018); authors Chamberland, Iyer, Poulin. |
| Battistel2023 | yes: 2303.00054, 10.1088/2399-1984/aceba6 | yes | Nano Futures 7(3), 032003 (2023); all 8 authors match. |
| Google2025 | yes: 2408.13687, 10.1038/s41586-024-08449-y | yes | Nature 638, 920-926 (2025). arXiv lists 249 authors beginning Acharya, Aghababaie-Beni, Aleiner, Andersen, Ansmann. The Crossref (Nature) list puts D. A. Abanin second (cosmetic). |
| Caune2026 | yes: 2410.05202, 10.1038/s41467-026-73331-6 | yes | Nat. Commun. 17, 7383 (2026), confirmed by Crossref; 25 authors; arXiv v1 7 Oct 2024. |
| Barber2025 | yes: 2309.05558, 10.1038/s41928-024-01319-5 | yes | arXiv title "...fast and highly resource efficient decoder..."; journal title "...fast and resource-efficient decoder..."; Nat. Electron. 8(1), 84-91 (2025); all 13 authors match. |
| Yang2026 | yes: 2605.04892 | yes | arXiv v1 6 May 2026; all 12 authors match; Crossref shows no journal version. |
| Corcoles2021 | yes: 2102.01682, 10.1103/PhysRevLett.127.100501 | yes | PRL 127, 100501 (2021); 7 authors. |
| Govia2023 | yes: 2207.04836, 10.1088/1367-2630/ad0e19 | yes | NJP 25, 123016 (2023); 6 authors. |
| Hothem2025 | yes: 2410.16706, 10.1038/s41467-025-60923-x | yes | Nat. Commun. 16, 5761 (2025); 6 authors. |
| Gidney2021 | yes: 2108.10457, 10.22331/q-2021-12-20-605 | yes | Quantum 5, 605 (2021); 4 authors. |
| Geher2025 | yes: 2408.00758, 10.1038/s41534-025-00998-y | yes | npj Quantum Inf. 11, 39 (2025); 4 authors (G. P. Geher). |
| RyanAnderson2021 | yes: 2107.07505, 10.1103/PhysRevX.11.041058 | yes | PRX 11, 041058 (2021); 17 authors; Honeywell Quantum Solutions. The arXiv page has no journal ref, as the document says. |
| Moses2023 | yes: 2305.03828, 10.1103/PhysRevX.13.041052 | yes | 96 authors. The PRX title is "A Race-Track Trapped-Ion Quantum Processor" (hyphenated); the arXiv title has no hyphen. |
| Paetznick2024 | yes: 2404.02280 | yes | 32 authors; v3 17 Nov 2024; Crossref title search finds no journal version. |
| Bluvstein2024 | yes: 2312.03982, 10.1038/s41586-023-06927-3 | yes | Nature 626, 58-65 (online 6 Dec 2023, issue Feb 2024); 21 authors. |
| Muniz2025 | yes: 2506.09936, 10.1103/v7ny-fg31 | yes | PRX 15, 041040 (4 Dec 2025); 62 authors; Atom Computing plus Microsoft Quantum co-authors. |
| Abdurakhimov2024 | yes: 2408.12433 | **no (minor)** | arXiv lists **114** authors (all distinct), not 113. Title, year, ID and first six authors are correct. No journal version found. |
| IQMdocs | yes (3 pages) | yes | "IQM client 36.0.0" (models; Qiskit user guide) and "iqm.pulse 15.0.0" (Reset_Conditional); pages say "Last updated on 2026-10-06". |
| AWSBraket2025 | yes | yes | "by Tim Chen, Scott Smart, and Zia Mohammad on 26 JUN 2025". |
| IQMHalocene2025 | yes | yes | Dated "13 Nov 2025" / "Espoo, Finland - 13th November 2025". |

## 3. Key findings (structured summary), claim by claim

S = supported, P = partially supported. Evidence is quoted from the fetched
source (**from literature**).

| # | Claim (short) | Verdict | Evidence / problem |
|---|---|---|---|
| 1 | Knill: state defined relative to a Pauli frame; Pauli corrections need not be applied; wait only for non-Pauli compensation | S | "makes it possible to avoid explicitly applying Pauli products for correction or teleportation compensation"; "For teleportations used to implement non-Clifford gates, a non-Pauli compensation may be required. In this case, we must wait for the measurement outcomes." |
| 2 | Knill: postselected vs standard computing; detecting codes correct known-location errors | S | "In 'postselected' quantum computing, the state is then rejected and the computation restarted. In standard quantum computing, the syndrome information must be used to update the Pauli frame."; "error-detecting codes can always correct an error at a known location"; "Purely postselected computing has little computational power". |
| 3 | Never necessary to correct Pauli errors physically; in all-Clifford circuits the frame only reinterprets the final measurement; wait only before the T-gate correction | S | Terhal Sec. II.7.3: "in principle it is never necessary to physically correct Pauli errors"; "The Pauli frame simply alters the interpretation of the final measurement outcome"; "required to wait before doing the classically-controlled SX gate". Chamberland: "for circuits containing only Clifford gates, all Pauli recovery operators can be tracked in classical software without ever exiting the Pauli group". |
| 4 | Slow decoding: backlog and exponential slowdown at T gates; a Clifford-only memory never waits | S | Terhal: "should never lead to a increasing backlog of syndrome data"; "r_gen/r_proc=f>1 ... exponential slow down"; "quantum memory ... never has to wait for the classical processing". Chamberland: "an exponential slow down would occur during the computation". Battistel: "known as the backlog problem". |
| 5 | Correction needed only before the next non-Clifford gate | S | Battistel: "needed only for the next-in-line logical non-Clifford gate ... otherwise it can be tracked by the decoder in the Pauli frame (or leakage frame) until needed". |
| 6 | Ion QCCD: software frame vs physical correction, 92(1)% vs 93(2)% | S | RyanAnderson: "achieving a logical fidelity of 93(2)%" (physical correction) and "software-correction version ... 92(1)%"; "not significantly different". |
| 7 | Budgets about 1 us (transmons) and 1 ms or looser (ions, atoms); throughput over latency; link at most a few hundred ns | S | Battistel: "transmons pose the tightest constraints, the value of 1 us is often used as the benchmark"; "For trapped ions we take 1000 times that (i.e., 1 ms) ... 200 ms ... might be even much looser"; "QEC-cycle times are usually in excess of many milliseconds, leaving ample time"; "Throughput ... is even more important than latency"; "taking at most a couple hundreds of nanoseconds". |
| 8 | Google: real-time d=5, Sparse Blossom, 63 +- 17 us, 1.1 us cycle, up to 1e6 cycles, no feedback | S | "net average of 63 +- 17 us independent of the length of the experiment (up to 1.1 seconds)"; "specialized version of the Sparse Blossom algorithm"; "low-latency Ethernet"; "does not yet include feedback into the logical circuit"; eps_5 = 0.35%, Lambda = 2.0 vs 0.269%, 2.18 (neural network, offline). |
| 9 | Caune: FPGA CC decoder at 0.44-0.79 us per round; 9.6 us = 6.5 + 3.1 us; Ankaa-2 without resets | S | "ranges from 0.44 us when decoding 5 rounds to 0.79 us when decoding 25 rounds"; "9.6 us, including 6.5 us decoding time and 3.1 us communication and control latency"; "ancilla qubits are not reset between rounds"; "While resets are not available". |
| 10 | MHz decoders (CC ASIC 0.06 mm2, 8 mW); Yang 550 ns loop, 124 ns NN, 1.25 us d=3 cycle, non-Clifford feedback | S | Barber abstract: "MHz decoding speed ... up to an 881 and 1057 qubits surface code with the FPGA and ASIC ... 0.06 mm2 ... 8 mW". Caveat: the syndromes are simulated, and the ASIC is "signed off to a standard that is ready to be taped out" (not fabricated). Yang abstract: "closed-loop latency of 550 ns, including 124 ns for NN decoding ... within a 1.25 us QEC cycle"; "where Pauli-frame updating alone becomes insufficient". |
| 11 | IBM: 300 ns readout pulse; conditional reset 1.4 us for two cycles; 0.01 reset error | S | "the 300 ns long readout pulse"; "This cycle is repeated twice ... total qubit reset latency of 1.4 us"; "The reset error after two cycles of conditional reset is 0.01". (The 0.03 reset error in their simulations is a separate assumed value.) |
| 12 | Google 1/Lambda budget: data idle during M/R about 20%, readout 11%, reset 2%; DD during M/R | S | Suppl. Table S4: Data qubit idle 0.9e-2 gives 0.09 (20%), Readout 0.8e-2 gives 0.048 (11%), Reset 1.5e-3 gives 0.009 (2%), and CZ rows 41% + 11% + 5%. Fig. 2c: "Data idle: Data qubit idle error during measurement and reset". DD appears in the cycle drawings (Fig. 2a, Fig. S12) rather than in a sentence. |
| 13 | Leakage time correlations; hourly bursts, about 400 us decay, 1e-10 floor | S | "time correlations in error detection events that confuse the error decoder"; DQLR gives "a 35% increase in Lambda"; bursts "occur approximately once an hour ... exponential time constant around 400 us"; "roughly 30 qubits"; "apparent logical error floor of 10^-10" at d >= 15. |
| 14 | IBM MCM spectator errors (1.7e-3, Stark shift or cross-dephasing); crosstalk "4-26x" above prediction; DD removes at most 38% | **P** | Govia: "epsilon_IRB = 1.7e-3 +- 1.0e-3"; "measurement induced Stark shift, or weak cross-dephasing"; "Most control qubits showed some measurement induced control error". Hothem: "experimental r_Omega that is 4-5x larger than simulations using the calibration data"; "for n=15, epsilon_MCM is 26x larger than predicted"; "epsilon_MCM decreased by as much as 38%". **Problem:** 4-5x applies to the total QIRB layer error r_Omega, not to MCM crosstalk alone (Hothem adds that "most of this discrepancy is due to measurement-induced crosstalk"). |
| 15 | Geher: reset gives no memory benefit; XOR syndromes; length-2 edge; conditional reset not recommended | S | "For memory experiments, we confirm numerically that resetting provides no benefit"; Eq. (1) "m_{g,j}=n_{g,j-1} xor n_{g,j}"; "time-like edge with length 2"; conditional reset "should not be used in any of the QEC experiments considered here"; "reset duration is greater than approximately 100 ns and the physical error probability is greater than approximately 10^-2.5". |
| 16 | Original SI1000 definition and its rationale | S | Gidney Table 2: CZ(p), AnyClifford_1(p/10), Init_Z(2p), M_Z(5p), Idle(p/10), ResonatorIdle(2p), "7 time steps (about 1000 ns)". Table 1: M_Z is "Precede with a bitflip channel of strength p"; ResonatorIdle applies when "the qubit is not measured or reset in a time step during which other qubits are being measured or reset". "measurement and reset take an order of magnitude longer than other gates". |
| 17 | IQM: MCM, cc_prx semantics, reset = measure + conditional X, only x/y/rx/ry/r on one bit | S | Models page: "Each qubit may be measured multiple times, i.e. mid-circuit measurements are allowed"; "If it is 0, an identity gate of similar time duration gate is applied instead"; Reset "is only available when cc_prx is". Qiskit guide: "a (projective) measurement followed by a classically controlled X gate"; "Only the x, y, rx, ry and r gates can be classically controlled ... conditioned on one classical bit"; "depends on the instrumentation". Reset_Conditional: "feedback result propagation delay"; "only locus components that have readout and drive". |
| 18 | Braket: Garnet dynamic circuits experimental, verbatim-only, local group | S | "As an experimental capability ... conditional operations can only be applied to qubits in a local group"; "only supported as in verbatim compilation"; "pi rotation conditioned on the measured bit". |
| 19 | Garnet medians and no MCM/reset benchmarks; Halocene targets | S | Abdurakhimov: "median single qubit gate error of 9 x 10^-4, CZ error of 5 x 10^-3 and two qubit Clifford error of 2x10^-2"; "median readout error for simultaneous QPU readout to be 3x10^-2"; "T1 from the order of 40us to above 100us"; "frequency-multiplexed dispersive readout"; "3 probe lines". My own text search also finds 0 hits for "reset" and "mid-circuit". Halocene: "research on up to five logical qubits and the implementation of Clifford gates"; "a modular decoder architecture, and support for NVIDIA NVQLink". |
| 20 | H2: MCMR crosstalk 4.5e-6 / 3.8e-6 (<= 2e-5); transport about 60%; memory errors dominate; Wasm; decoded after final measurement | S | Moses Table II: "Measurement crosstalk 0.045(6)", "Reset crosstalk 0.038(6)" (units 1e-4); "60% on average (see Table I)"; "Web Assembly (Wasm)"; "does not perform mid-circuit real-time decoding". Paetznick (H2): "crosstalk errors <= 2x10^-5"; "memory and transport errors are the dominating source of error". Caveat: the RyanAnderson 45% dephasing share comes from the earlier ten-qubit Honeywell QCCD, not H2. |
| 21 | H1-1 long-range MCM crosstalk 0.63% vs 0.23%, fixed to 0.29% | S | "epsilon_MCM dropped from 0.63 +- 0.09% to 0.29 +- 0.06% ... consistent with the 0.23 +- 0.09% MCM error rate obtained from the emulator". |
| 22 | Neutral atoms: zoned readout (500 us, <= 0.1%), < 1 ms loop; alternative "in-place" cycle of "about 18 ms" costs 0.5% contrast and 1% loss; 41 rounds with loss-aware matching | **P** | Bluvstein is correct: "~100 um away to a readout zone"; "500 us imaging time, readout fidelity about 99.8%"; "<= 0.1% per 500 us"; "slightly less than 1 ms". Muniz numbers are correct: "atom loss of 0.0106(7) per cycle, and a site-averaged contrast loss of 0.0049(7)"; "41 rounds"; "matching graph was modified on a per-shot basis to account for atom loss". **Problems:** (a) Muniz's cycle is not in place: "moving subsets of ancilla qubits from the register to the MZ tweezers", with "a global beam that illuminates all atoms". (b) The cycle runs two 7 ms images, each followed by 4 ms gray-molasses cooling and "a 300 us optical pumping pulse" ("0.006 loss per MCM cycle" from two cooling steps). That is at least about 22.6 ms, not about 18 ms. |
| 23 | H2: [[7,1,3]] 0.05% (correction) vs 0.001% (detection) at about 72% acceptance; rejection roughly linear in rounds | S | Table 1: 24,200 runs, 18,035 pre-accepted, 17,389 post-accepted; 0.05% vs 0.001% (0 errors); baseline 0.50%. "we expect the pre- and post-rejection rates to be linear in the number of repetitions (to leading order), and the experimental data is highly consistent". I computed 17,389/24,200 = 0.719 (**verified numerically**). |
| 24 | Conjecture: LRB-C needs no real-time decoder, feed-forward or physical correction | S | The premises are those of findings 1, 3, 4 and 5 (Terhal: frame "does not necessarily need to be available during the execution of the circuit" for all-Clifford circuits). The LRB circuits are Clifford with a terminal X_L readout (CONTEXT). The inference is sound (**conjecture** is the right label). |
| 25 | Conjecture: LRB-D (all const/unif) and LRB-C are post-processings of the same data; store 4(m+1) outcomes | **P** | True for unif and const >= 1, and 4 x 23 = 92 (**verified numerically**). IQM: "All the measurement keys and feedback keys used in a circuit must be unique". **Problem:** CONTEXT says "const=0 is a different circuit: no check rounds at all", so it cannot be obtained from the same data. |
| 26 | Verified analytically + conjecture: no-reset XOR identity; IQM reset = measure + pi pulse, so no-reset is simpler | S | I re-derived the identity (**verified analytically**). Z check: CNOTs add the parity to the stored bit, so n_j = n_{j-1} xor s_j. X check: H|a>, phase kickback (-1)^s on the \|1> branch, then H gives \|a xor s>. This matches Geher Eq. (1). IQM premise confirmed (finding 17); Caune ran with "resets ... not available". "Less noisy" remains a conjecture. |
| 27 | Code reading + arithmetic: 5 x DEPOLARIZE1(2p) per round; 0.0487 vs 0.0199; parallel readout gives 2.5-5x | S | I read the code: `x_stabilizer_check_operations`/`z_stabilizer_check_operations` interleave each `M` with the next check's gates, so `emit_operations` makes every M its own layer, and `SI1000.scaled()` exists with a `meas_idle` field. Arithmetic (**verified numerically**): 0.0487 / 0.0199 / 0.0100 per qubit; any-of-5 = 0.221 / 0.0955 / 0.049; ratios 2.45 and 4.87. Nuances: the terminal round is built with `reset=False` (4 layers), and the "4-26x" caveat of finding 14 applies here too. |
| 28 | Conjecture: Garnet maps to SI1000 p about 5e-3; 23 rounds at 1.1-1.7 us is 25-40 us, close to T1 = 40 us | **P** | The cited numbers are quoted correctly, and 23 x 1.1-1.7 us = 25.3-39.1 us (**verified numerically**). **Caveats:** Garnet quotes average infidelities while SI1000 parameters are Pauli-channel strengths. At p = 5e-3 SI1000 gives CZ infidelity 4e-3 and 1q infidelity 3.3e-4, so Garnet's 1q error (9e-4) is about 2.7x SI1000's. The 1.1-1.7 us round times are Google and Rigetti values with simultaneous readout, not IQM ones (IQM readout duration not given in any source; I confirmed it is absent from Abdurakhimov2024). The repo's sequential single-ancilla readout would need 4 readouts per round. |
| 29 | Conjecture: shot times of 0.18-5.4 s on H2 make deep LRB-D unmeasurable | **P** | Moses Table I shot times run from 0.18 s (GHZ) to **20.6 s** (HoloQUADS t=24, 129 measurements); the document's own citation note says "0.18-20 s". The conclusion is unaffected. The qutrit unif=1 acceptance of about 4e-6 at m=22 is from CONTEXT. |
| 30 | Conjecture: hardware non-Markovian ingredients could cause non-exponential LRB decays | S | Premises confirmed: Google (leakage time correlations, bursts); Govia (tomography shows "coherent Z-phase error induced by Stark shift"); Hothem ("remain unexplained"); Muniz (loss per cycle); RyanAnderson (dephasing angle "dephasing rate x duration", including transport). The effect on LRB decays is untested (**conjecture**). |
| 31 | Conjecture: heralded erasure makes d=2 correction useful | S | Knill: "error-detecting codes can always correct an error at a known location". Muniz heralds loss per shot. A distance-2 code corrects one erasure (standard). Remains a **conjecture** for LRB. |

## 4. Statements found only in the document body

| Location | Statement | Verdict | Evidence |
|---|---|---|---|
| Sec. 2.5, item 33 | Micromotion hiding mitigates H2 MCMR crosstalk "for unmeasured ions in gate zones but not in conveyor regions" | P | Moses: "Ions in the conveyor belt regions suffer from a similar level of crosstalk errors as ions in the gate zones, although we do not attempt to apply the micromotion hiding technique to them." Hiding is indeed not applied there, but the crosstalk level is similar, not worse. |
| Sec. 2.1, item 10 (and 16) | Yang: case "where Pauli-frame updates alone are insufficient" (in quotation marks) | S (quote not verbatim) | The actual wording is "where Pauli-frame updating alone becomes insufficient". The meaning is unchanged. |
| I12 | "FPGA decoders already run under 1 us per round on superconducting hardware [Caune2026, Barber2025, Yang2026]"; Halocene "promises" a decoder in the loop | P | Caune and Yang ran on hardware. Barber evaluated simulated memory experiments ("We simulate logical memory experiments"), and its ASIC is a pre-tape-out design. The Halocene release says "a modular decoder architecture, and support for NVIDIA NVQLink" but makes no explicit real-time in-loop claim. |
| Sec. 2.8 table | Atom Computing: "MCM cycle ~ 2x7 ms + 4.3 ms"; Quantinuum: "QEC cycle < 200 ms (H1, 2021)" | P | The Atom Computing cycle omits the second cooling and pumping block (see finding 22). RyanAnderson's text names "a ten qubit QCCD trapped-ion quantum computer" (Honeywell), not H1. The "H1" label is an inference. The other rows are correct, including Bluvstein's "free-space movement time between gates is roughly 200 us" and "Entangling gates (270-ns duration)". |
| I13 | Detection buys 1-2 orders of magnitude "at a modest acceptance cost" [Paetznick2024, Bluvstein2024] | P | Paetznick confirms it: a 50x gain at 72% overall acceptance ([[7,1,3]]), and 0.17% to 0.001% with about 0.5% post-rejection ([[12,2,4]]). Bluvstein's GHZ error detection (72% to 99.85%) comes "at the cost of postselection overhead", and "discarding just 50% of the data improves GHZ fidelity to about 90%". The cost is not modest there. Bluvstein's sliding-scale scheme is a published precedent for the proposed hybrid policy. |
| I11 | Neutral atoms: Muniz cycle cost "or at most ~0.1% with zoned readout [Bluvstein2024]" | P | Muniz also reads out in a separate zone (MZ). The difference is a global vs a local imaging beam (and the atom species), not zoned vs non-zoned. |
| Sec. 1 / Sec. 6 | Side references (Rudinger, Gaebler, Bluvstein arXiv:2506.20661, Maurer, Riesebos) and Baumer arXiv:2308.13065 marked UNVERIFIED | S | All fetched: Rudinger, PRApplied 17, 014014 (2022); Gaebler, PRA 104, 062440 (2021). arXiv:2506.20661 (arXiv title "Architectural mechanisms of a universal fault-tolerant quantum computer") is "A fault-tolerant neutral-atom architecture for universal quantum computation", Nature 649, 39-46 (2026). Maurer: "belief propagation iteration time of 24ns". Riesebos: DAC 2017, DOI 10.1145/3061639.3062300. Baumer is now **verified**: "Efficient Long-Range Entanglement using Dynamic Circuits", PRX Quantum 5, 030339 (2024). |

## 5. Analytic and code checks redone here

- **Verified analytically:** without reset, n_j = n_{j-1} xor s_j for both check
  types. The derivation is in finding 26.
- **Verified analytically:** in the repo model a pre-measurement X_ERROR on an
  ancilla is a real flip. Without reset it corrupts exactly one XOR-recovered bit,
  m_j, and m_{j+1} is unaffected. This matches Geher2025, where "a quantum
  measurement error corresponds to a time-like edge with length 1".
  Misclassification (classical) errors give length-2 edges.
- **Verified numerically:** the k-layer formula (3/4)[1-(1-8p/3)^k] at p = 5e-3
  gives the numbers stated in I6. 23 rounds x 1.1-1.7 us gives 25.3-39.1 us, and
  17,389/24,200 = 0.719.
- **Code reading:** `src/lrb_qubit/circuits.py:emit_operations` groups only
  *consecutive* `M` operations into one layer. `code.py` places CNOTs between
  every pair of `M`s, so each measurement is its own layer. `build_lrb_circuit`
  calls `run_check_round(depth, reset=False)` for the terminal round, which
  therefore has 4 measurement layers and no reset layer.

## 6. Sources fetched (all retrieved over HTTP in this session)

- arXiv abstract pages:
  - quant-ph/0410199, 1302.3428, 1704.06662, 2303.00054, 2408.13687
  - 2410.05202, 2309.05558, 2605.04892, 2102.01682, 2207.04836
  - 2410.16706, 2108.10457, 2408.00758, 2107.07505, 2305.03828
  - 2404.02280, 2312.03982, 2506.09936, 2408.12433
  - side references: 2103.03008, 2108.10932, 2506.20661, 2510.21600, 2308.13065
- Crossref works records for all 16 journal DOIs, plus 10.1145/3061639.3062300,
  10.1038/s41586-025-09848-5 and 10.1103/PhysRevApplied.17.014014. I also ran
  Crossref title searches for Paetznick2024, Abdurakhimov2024 and Yang2026; none
  has a journal version.
- Full texts:
  - arXiv HTML: 2408.13687, 2410.05202, 2309.05558, 2605.04892, 2207.04836,
    2410.16706, 2408.00758, 2404.02280, 2312.03982, 2506.09936
  - ar5iv: 1302.3428, 2102.01682, 2108.10457, 2305.03828
  - arXiv PDF converted to text: quant-ph/0410199, 1704.06662, 2303.00054,
    2107.07505, 2305.03828, 2408.12433
- Vendor pages:
  - docs.iqm.tech: the models page, the Qiskit user guide and Reset_Conditional
  - aws.amazon.com: the Braket blog post of 26 Jun 2025
  - iqm.tech: the press release at ?p=14348
