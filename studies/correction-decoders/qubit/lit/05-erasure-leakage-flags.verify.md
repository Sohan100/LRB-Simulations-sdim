# Verification of Lit 05 (erasure conversion, leakage, flag qubits)

Target: `lit/05-erasure-leakage-flags.md` (I did not edit it). I am an
independent verifier, and this report was prepared on 2026-10-28.

Labels follow `../CONTEXT.md`:
- **from literature**: I read the quoted text myself.
- **verified numerically**: my own scripts, run with stim 1.15.0 and Python
  3.11 (seconds each, `nice -n 19`).
- **verified analytically**: my own derivation.

## Method

- **arXiv abstract pages.** I fetched the abstract page for all 28 IDs and
  read title, authors, dates, journal reference and DOI from the page meta
  tags. I also fetched the v1 pages of 2411.11822, 2506.13724, 2307.03169 and
  2208.05461.
- **Crossref.** I fetched `api.crossref.org/works/<DOI>` for all 23 DOIs. I
  also ran a Crossref title search for the five arXiv-only items. None has a
  journal version; Helios has only a Research Square preprint DOI,
  10.21203/rs.3.rs-8130124/v1.
- **Publisher pages.**
  - APS page for PRX 13, 041022 (Kubica2023).
  - Nature page for Nat. Phys. 22, 910 (Zhang2026). The abstract is visible;
    the full text is paywalled.
  - Open-access Nat. Commun. HTML for McEwen2021, whose arXiv entry is a PDF
    only.
- **Full text.** I downloaded the arXiv LaTeX sources of the other 27 papers
  (current version, plus v1 of 2506.13724). Every number and quote below was
  searched for in those sources.
- **Own numerics.**
  - An independent stim re-derivation of every Section 3a code fact.
  - A detector-error-model (DEM) check of the terminal-round tail using
    `src/lrb_qubit/circuits.py` (import only, no edits).
  - A check of how much a known herald location helps the decoder.

## Bottom line

1. **No hallucinated sources.** All 28 arXiv IDs resolve to the stated papers.
   All 23 DOIs resolve to the stated journal, volume and article number.
2. **Metadata errors (3).**
   - **Kubica2023 (substantive).** The published PRX version has six
     authors: Kubica, Haim, Vaknin, **H. Levine**, Brandão, Retzker (APS page
     and Crossref). The review lists the five arXiv-v1 authors and, in Sec. 4,
     calls that list "Confirmed" for PRX 13, 041022. That is wrong.
   - **Chou2024** has 34 authors (arXiv, Crossref and LaTeX source), not
     "35".
   - **McEwen2021** has 51 authors (arXiv and Crossref), not "50".
3. **One wrong number (substantive).** The Zhang2026 CZ erasure fraction is
   wrong for the version cited.
   - The review cites the Nat. Phys. paper but quotes v1: "Roughly 31% of the
     AR gate error is detected mid-circuit as erasure errors" and Table
     "r_e,CZ ~ 0.31".
   - arXiv v2 (posted 2026-06-23, after the 2026-06-12 publication) has an
     abstract identical to the Nature abstract. It says "Moreover, 38(6)\% of
     the AR gate error is detected mid-circuit as erasure errors" and Table
     "AR CZ erasure fraction r_e,CZ ~ 0.38". v2 adds: "Even with the current
     erasure fraction of our CZ gates (about 38\%), erasure conversion is
     predicted to raise the surface-code threshold from 0.94\% to 1.45\%".
   - The error affects Summary bullet 3, Finding 5, key finding 3 and the 3b
     wording.
4. **Overstatements in the structured key findings (3).**
   - Key finding 3 calls 94% (ions) and 93.4% (Cs) "far below" 98%. Also,
     93.4% is a loss-detection accuracy, not an erasure fraction of gate
     errors.
   - Key finding 11 calls the `[[5,1,3]]` hook an "undetectable logical
     error". Chao and Reichardt say the weight-2 error is detected but
     miscorrected into a logical.
   - Key finding 2 says Zhang2026 "already run[s] the hybrid policy". That
     paper reports erasure-corrected decoding with no postselection and
     postselection on erasure flags as separate analyses.
5. **Own analysis holds.** Every "verified numerically" and "verified
   analytically" code claim in Sec. 3a reproduces independently. I extended
   two of them to circuit level (Sec. 4 below).
   - Reproducibility nit: the Appendix A script as printed is abridged. Run
     verbatim, it prints only the three erasure lines; the distance, logical
     list, hooks, irreducible list and stim lines of its "Key output" come
     from code that is not shown.

## 1. Citation metadata

`ok` means title, authors, year and ID/DOI are as stated, up to arXiv-vs-journal
title capitalization.

| Key | ID / DOI checked | ok? | Correction or note |
|---|---|---|---|
| Wu2022 | 2201.03540; 10.1038/s41467-022-32094-6 | yes | Nat. Commun. 13, 4657 (2022); 4 authors as stated. |
| Kubica2023 | 2208.05461; 10.1103/PhysRevX.13.041022 | **no** | Published authors: A. Kubica, A. Haim, Y. Vaknin, **H. Levine**, F. Brandão, A. Retzker (6). arXiv v1 lists 5. PRX 13, 041022, 1 Nov 2023. |
| Gu2025 | 2408.00829; 10.1103/985g-58gd | yes | PRX Quantum 6, 040354 (issued 2025-12-04). Journal title: "Optimizing Quantum Error-Correction Protocols with Erasure Qubits". |
| Levine2024 | 2307.08737; 10.1103/PhysRevX.14.011051 | yes | 113 authors, PRX 14, 011051 (2024). |
| Chou2024 | 2307.03169; 10.1038/s41567-024-02539-4 | **no** (count) | **34** authors, not 35. Nat. Phys. 20, 1454-1460 (2024). Journal title drops "Demonstrating" (as stated). No DOI on the arXiv page; Crossref confirms it. |
| Zhang2026 | 2506.13724; 10.1038/s41567-026-03309-0 | yes | Nat. Phys. 22, 910-916 (2026). v1 title confirmed. Numbers quoted from v1 differ from v2/published (see Bottom line 3). |
| Reichardt2024 | 2411.11822 | yes | v3 title as stated; v1 "Logical computation demonstrated with a neutral atom quantum processor" confirmed (v1 had 71 authors, v3 72). No journal reference or DOI. |
| Chow2024 | 2405.10434; 10.1103/PRXQuantum.5.040343 | yes | PRX Quantum 5, 040343 (2024). |
| Quinn2026 | 2411.12727; 10.1103/p3cy-8yjk | yes | Phys. Rev. A 113, L040601 (2026-04-02). |
| Ransford2025 | 2511.05465 | yes | 186 authors; arXiv only (Research Square preprint exists). |
| McEwen2021 | 2102.06131; 10.1038/s41467-021-21982-y | **no** (count) | **51** authors, not 50. Nat. Commun. 12, 1761 (2021). |
| Miao2023 | 2211.04728; 10.1038/s41567-023-02226-w | yes | arXiv title as stated; Nat. Phys. 19, 1780-1786 (2023). arXiv has 117 authors; the journal version lists 118. |
| Battistel2021 | 2102.08336; 10.1103/PRXQuantum.2.030314 | yes | The PRX Quantum title drops the leading "A" ("Hardware-Efficient Leakage-Reduction Scheme ..."). Sec. 4 calls the arXiv title "the full title" of the journal paper. |
| Xin2026 | 2511.17460; 10.1103/y66v-988s | yes | PRX Quantum 7, 033062 (2026-09-21); 8 authors. |
| Varbanov2020 | 2002.07119; 10.1038/s41534-020-00330-w | yes | npj Quantum Inf. 6, 102 (2020). |
| WoodGambetta2018 | 1704.03081; 10.1103/PhysRevA.97.032306 | yes | PRA 97, 032306 (2018). |
| Wallman2016 | 1412.4126; 10.1088/1367-2630/18/4/043021 | yes | NJP title "Robust characterization of leakage errors" confirmed. |
| Ceasura2022 | 2212.05488 | yes | arXiv only (no journal version found). |
| ChaoReichardt2018 | 1705.02329; 10.1103/PhysRevLett.121.050502 | yes | PRL 121, 050502 (2018). |
| Chamberland2020 | 1907.09528; 10.1103/PhysRevX.10.011022 | yes | PRX 10, 011022 (2020). |
| ChamberlandBeverland2018 | 1708.02246; 10.22331/q-2018-02-08-53 | yes | Quantum 2, 53 (2018). |
| Grassl1997 | quant-ph/9610042; 10.1103/PhysRevA.56.33 | yes | PRA 56, 33-38 (1997). |
| Ma2023 | 2305.05493; 10.1038/s41586-023-06438-1 | yes | Nature 622, 279-284 (2023); journal title as stated. |
| Mehta2025 | 2503.10935 | yes | arXiv metadata lists 72 named authors; the LaTeX source also has a "Quantum Circuits, Inc. Team" author entry, so "73" is defensible. arXiv only. |
| Hung2026 | 2604.16292 | yes | 17 authors, 2026-04-17, arXiv only. |
| Marques2023 | 2302.09876; 10.1103/PhysRevLett.130.250602 | yes | PRL 130, 250602 (2023), 12 authors. |
| Chasseur2015 | 1505.00580; 10.1103/PhysRevA.92.042333 | yes | The PRA title drops the leading "A" ("Complete randomized benchmarking protocol ..."). |
| Stricker2020 | 2002.09532; 10.1038/s41586-020-2667-0 | yes | Nature 585, 207-210 (2020); journal title "Experimental deterministic correction of qubit loss". |

## 2. Structured key findings

| # | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| 1 | Corrects d-1 erasures but only floor((d-1)/2) unlocated errors | supported | Kubica2023: "the distance-d code allows to successfully correct any d-1 (or fewer) erasure errors ... guaranteed to successfully correct only up to floor((d-1)/2) arbitrary single-qubit errors". Grassl1997 abstract: "four qubits are necessary and sufficient to encode one qubit and correct one erasure, in contrast to five qubits for unknown positions". |
| 2 | `[[4,2,2]]` experiments run the hybrid policy (Yb 0.874 to 0.902, 3.6x; 26.6% vs 42.0%; 10.2%, 858 of 29,760) | partially supported | All numbers exact (Zhang2026 v1 and v2; Reichardt2024 Sec. on the 24-qubit cat). Reichardt does run the hybrid ("using error detection and loss correction, p_X+p_Z = 26.6%"). Zhang2026 reports separate analyses: 0.874 to 0.902 is *unconditional* decoding with the rule "if the parity is odd and exactly one erasure is present, we flip the erased qubit". The 3.6(1) comes from "additionally post-selecting on the mid-circuit erasure information", i.e. discarding erasures, not correcting them. Reichardt2024 is Microsoft plus Atom Computing, not Atom Computing alone. |
| 3 | Erasure fractions far below 98%: Yb 0.31 CZ / 0.56 1Q, ions ~94%, Cs ~93.4% | partially supported | 0.56, 0.72 and 0.5 are correct. **CZ is 0.38 (38(6)%) in v2/published**; 0.31 is v1. Quinn2026 "~94% of spontaneous Raman scattering errors" and Chow2024 "~93.4% accuracy" are real, but they are not "far below" 98%. The 93.4% is LDU atom-loss detection accuracy, not an erasure fraction. |
| 4 | Superconducting erasure qubits (2.19e-3, ~40x, <0.1%; >99%, 0.01%; g-f) | supported | Levine2024 abstract: "p_erasure = 2.19(2)x10^-3 per gate while the residual errors are ~40 times lower ... < 0.1% dephasing error per check" (per X90). Chou2024 abstract: "0.01%-level ... over 99% of cavity decay events as erasures". Kubica2023: g-f qubit "encoded in the first three levels of a single transmon"; detection "by mapping the \|e> state to an extra transmon". |
| 5 | Leakage non-Markovian; ~5%, >1%; DQLR ~1e-3, ~2e-3, uncorrelated | supported | Miao2023: "nearly 5% average leakage population for data qubits ... over 30 cycles" (still growing). "No reset never has p̄_t,t' < 1% for any of the correlation distances". DQLR "around 10^-3 for data qubits"; non-local correlations "do not exceed 2x10^-3" (text: above 2e-3 only at distance 2). "Injected leakage impacts logical error rates in the same fashion as uncorrelated Pauli errors" (extrapolated simulations). Note: McEwen2021 and Miao2023 never say "non-Markovian"; they say "correlated in space and time". The label comes from Wallman2016 and Chasseur2015. |
| 6 | LRUs cut leakage lifetime to ~1 cycle; readout LRU 98.4%, no overhead, no postselection | supported | Battistel2021 abstract: "average leakage lifetime is reduced to almost 1 QEC cycle" (density-matrix simulations, d=3), with no hardware or cycle-time overhead. Xin2026: "98.4% leakage removal fraction ... (99.2%) ... without any post-selection". Marques2023: "suppress leakage buildup within 1% ... over 50 cycles". |
| 7 | Leakage heralded by multi-outcome readout; HMM; ~47% discarded at d=3 | supported | Ransford2025: "ternary (three outcome) measurement ... result of 0, 1, or L". Miao2023: "measurement that can resolve \|2>" and "a readout pulse capable of distinguishing all of the four lowest qubit energy levels". Varbanov2020: "discarding about 47% of the data" (Surface-17 simulation). |
| 8 | Helios leakage ~30% of 2Q infidelity; nearly all linear transport memory error | supported | Ransford2025: "7.9(2)x10^-4, which includes a leakage rate of r_L,2Q = 2.4(1)x10^-4" (ratio 0.304); leakage "accounts for nearly all of the linear memory error". |
| 9 | Leakage RB: multi-exponential to a nonzero steady state; oscillations; code space; "fundamentally non-Markovian" | supported | WoodGambetta2018: λ1 = 1 - L1 - L2 with steady state L1/(L1+L2); "insufficient twirling ... oscillations"; "non-Markovian effects may appear". Wallman2016: "the logical code space defined by some quantum error-correcting code"; "make coherent leakage a fundamentally non-Markovian process". Chasseur2015: Φ_y = Σ a_i λ_i^y, "At least a single λ_i is unity". Nuance: WoodGambetta's own leakage-RB population fit is a single exponential; their "bi-exponential ... multiple-exponential" sentence describes the Epstein2014 and Chasseur2015 models. |
| 10 | Ceasura2022: syndrome qubits as hidden memory give oscillating, non-exponential LRB | supported | Ceasura2022 abstract: "gate and time-independent noise model ... oscillating rather than decaying exponentially ... the role of the hidden qubits is assigned to the syndrome qubits ... coupled to the logical qubits via a decoder". This is a constructed example ("can"). The "as well as remove them" clause is the reviewer's inference, not Ceasura's. |
| 11 | Flags needed when an ancilla fault becomes an *undetectable logical* (`[[n,n-2,2]]`, weight-4 `[[5,1,3]]`); flag-MWPM gives full distance | partially supported | `[[n,n-2,2]]` is right (Chao and Reichardt section "Two-qubit fault-tolerant error detection for [[n,n-2,2]] codes": "a Z fault ... results in an undetectable logical error. With a flag qubit, the circuit is fault tolerant"). For `[[5,1,3]]` the weight-2 error is *detectable but miscorrected*: "even a perfect error-correction procedure will correct this error in the wrong direction, creating the logical error". Chamberland2020: "correct up to the full code distance while respecting the limited connectivity" is right. |
| 12 | Our code: d=2; weight-2 logicals X0X3, X2X4, Z0Z2, Z3Z4; erasures 5/5, 6/10 (bad {0,2},{0,3},{2,4},{3,4}), 0/10 | supported (verified numerically) | My independent script gives 48 logicals, d=2, exactly those weight-2 logicals, X_L/Z_L min weight 2 and Y_L min weight 3. Erasures: 5/5, 6/10 with the same 4 bad pairs, 0/10 triples. |
| 13 | No single check-round fault gives an undetectable logical; hooks reduce to weight 1; irreducible mixed weight-2 errors are detectable | supported (verified numerically) | Reproduced: undetectable list empty. Hooks Z1Z3~Z0, Z2Z4~Z1, X1X2~X0, X1X4~X3. Irreducible: X1Z3, Y1Z3, X2Z4, Y2Z4, Y1X2, Z1X2, Y1X4, Z1X4, all with nonzero syndrome. Logical-gate faults on {2,3} (SWAP/CZ) are also detectable because {2,3} is a correctable pair. |
| 14 | Correction fails at first order; LRB-D second order except a first-order terminal Z0/Z3 tail; `data5` catches it | supported (verified numerically) | DEM of `build_lrb_circuit`, p=1e-3: the undetectable first-order logical mass is **4.52p** with `data2` and **0** with `data5`. This holds at m=0, and at m=3 and m=6 with random Cliffords, so it is depth-independent. Analytic budget (terminal round has no reset): 2·(4/15)p + 3·(4/3)p = 4.53p. `circuits.py` documents `data5` as "enables final X detectors". |
| 15 | stim 1.15.0 has HERALDED_ERASE and HERALDED_PAULI_CHANNEL_1 with a herald bit | supported (verified numerically) | `stim.gate_data(...).produces_measurements` is True for both. A circuit with 3 heralds and 3 M has 6 measurements. |
| 16 | LRB-D rejection behaves like code-space leakage; const=1 acceptance about 1/16; correction acts as an LRU | not checkable (conjecture) | The premises are cited correctly: code-space leakage (Wallman2016), steady state (WoodGambetta2018), lifetime of about one cycle (Battistel2021), uncorrelated-like after removal (Miao2023), decoder memory (Ceasura2022). 1/16 = 1/2^4 mirrors the CONTEXT qutrit 1/81 = 1/3^4. |
| 17 | Hardware path "herald and correct one, reject the rest" | not checkable (conjecture) | The premises check out: Helios ternary readout, Quinn ~94%, Reichardt "Full connectivity is enabled by atom movement", Xin LRU, Levine dual-rail, Kubica g-f. **Verified analytically:** in the 3x3 patch the corners must be cyclically 0-3-4-2 (ancilla 5 on edge 0-3, 8 on 3-4, 6 on 4-2, 7 on 2-0), so data 2 and 3 sit on opposite corners. "Ions can do this now" is a stretch, because the ion erasure conversion (Quinn2026) is a separate two-ion demonstration, not part of Helios. |

## 3. Numbered findings in Sec. 2 (1-33)

All are **from literature**. Each was checked against the source text; the
quote is given where useful.

| # | Verdict | Note |
|---|---|---|
| 1 | supported | Kubica2023 quote above. |
| 2 | supported | Grassl1997 abstract. |
| 3 | supported | Wu2022: "98% of errors can be converted into erasures"; "threshold increase from 0.937% to 4.15%"; "ν_p = (d+1)/2 ... ν_e = d"; Fig. 1d "Erasure detection is applied after each gate". |
| 4 | supported | Zhang2026 v1 and v2, identical text. |
| 5 | **partially supported** | 0.56, 0.72 and 0.5 are correct. **CZ is ~0.38 / "38(6)%" in v2/published**; 0.31 and "Roughly 31%" are v1 only. |
| 6 | supported | 42.0, 26.6, 29.5(3.0) to 10.2, "858 trials with no loss at all" of 29,760. |
| 7 | supported | "once a qubit is lost, the code does not protect against errors on the remaining qubits". |
| 8 | supported | Kubica2023, false negative and false positive passage. ("only" is the reviewer's word; the paper calls the treatment "crude".) |
| 9 | supported | Gu2025 abstract plus "reduces the impact of measurement errors by approximately a factor of 2". |
| 10 | supported | Levine2024 abstract. |
| 11 | supported | Hung2026 abstract (384 ns, 6.0(2)e-4, 8(3)e-5). |
| 12 | supported | Chou2024 abstract. |
| 13 | supported | Mehta2025 abstract ("bounded at the few parts per million level"). |
| 14 | supported | Kubica2023, g-f section. |
| 15 | supported | Ma2023 abstract. |
| 16 | supported | Chow2024 abstract. |
| 17 | partially supported | The ~94% and 97.73% raw figures are right. The 99.16% is the *SPAM-corrected* fidelity (98.61%) with erasures subtracted, not the raw fidelity with erasures subtracted. |
| 18 | supported | Ransford2025, Secs. on 1Q RB, 2Q RB and transport. |
| 19 | supported | McEwen2021 abstract (bit-flip code). |
| 20 | supported | See key finding 5. |
| 21 | supported | Miao2023 quotes as given. |
| 22 | supported | Battistel2021 abstract. |
| 23 | supported | Xin2026 abstract. |
| 24 | supported | Marques2023 abstract. |
| 25 | supported | Varbanov2020 abstract. |
| 26 | supported with attribution nuance | See key finding 9. The bi- and multi-exponential statement is WoodGambetta's description of Epstein2014 and Chasseur2015. |
| 27 | supported | Wallman2016 quotes exact. |
| 28 | supported | Chasseur2015 quotes exact. |
| 29 | supported | Ceasura2022 abstract. |
| 30 | supported | Chao and Reichardt `[[5,1,3]]` section; "only seven qubits total". |
| 31 | supported | Section "Two-qubit fault-tolerant error detection for [[n,n-2,2]] codes" (Sec. V in the arXiv numbering). |
| 32 | supported | ChamberlandBeverland2018 abstract. |
| 33 | supported | Chamberland2020 abstract. |

## 4. Section 3a own-analysis claims

All of these were re-derived independently with my own scripts, not the
review's appendix.

- **Distance, logicals, syndrome table, erasure patterns, hooks, irreducible
  errors, stim heralds.** All reproduced (**verified numerically**). The
  single-qubit syndrome table in CONTEXT is right: for every data qubit, X, Y
  and Z give three distinct nonzero syndromes.
- **First-order failure of unheralded correction.** Confirmed: X0/X3, X2/X4,
  Z0/Z2 and Z3/Z4 share syndromes, and each pair differs by a weight-2
  logical (**verified numerically**).
- **LRB-D is second order except for the terminal tail; `data5` removes the
  tail.** Confirmed with the repo's own circuit builder; numbers are in key
  finding 14 above (**verified numerically**). This assumes LRB-D rejects on
  the absolute syndromes of the used checks, as CONTEXT defines it. The
  qubit builder emits round-difference detectors, so a policy that drops
  rounds must use cumulative parities to keep this property.
- **"nu = 2 for perfectly heralded erasures."** This also holds at circuit
  level when the herald names the erased gate (**verified numerically**). For
  every check-round CX, I gave the erased pair each of its 16 Paulis and
  propagated to the end of the round. No two outcomes with equal data
  syndrome differ by a logical, even if the erased ancilla's readout is
  ignored (0 ambiguous pairs at all 12 CXs). I did not test heralds that only
  give the round, not the gate.
- **Appendix A reproducibility.** Run verbatim, the script prints only the
  1-, 2- and 3-erasure lines (which match). The other "Key output" lines need
  code that is not shown.

## 5. Seed corrections (Sec. 4 of the target)

| Item | Verdict |
|---|---|
| Chou: arXiv 2307.03169, titles, Nat. Phys. 20, 1454-1460, DOI | correct |
| Levine: 2307.08737, PRX 14, 011051 | correct |
| Miao: arXiv vs journal title | correct |
| Battistel: ID and venue | correct, but the PRX Quantum title has no leading "A" |
| **Kubica: "Confirmed: authors Kubica, Haim, Vaknin, Brandão, Retzker, PRX 13, 041022"** | **wrong**: the PRX version adds H. Levine (6 authors) |
| Wu, McEwen, Chao-Reichardt, Chamberland-Beverland | correct |
| Reichardt v1 and Zhang v1 titles; Wallman and Stricker titles | correct |
| Stricker caveat | correct: stabilizers {Z1Z2, Z1Z3, X1X2X3X4}, logicals "T^Z = Z1Z4, T^X = X4", so a weight-1 X logical |

## 6. Edits I recommend to the target

1. Replace the Zhang2026 CZ erasure fraction with ~0.38 ("38(6)% of the AR
   gate error"; v2/published), or cite v1 explicitly. This touches Summary
   bullet 3, Finding 5 and key finding 3.
2. Fix the Kubica2023 authors to add H. Levine, and correct the Sec. 4 line
   that calls the five-author list "Confirmed".
3. Change the Chou2024 author count to 34 and the McEwen2021 count to 51.
4. Key finding 3: drop "far below" for the ion and Cs figures, and call
   93.4% a loss-detection accuracy.
5. Key finding 11: say that for `[[5,1,3]]` the single fault gives a
   *miscorrected* weight-2 error. Only `[[n,n-2,2]]` gives an undetectable
   logical.
6. Key finding 2 and Sec. 0: say that Reichardt2024 runs detect-plus-correct.
   Zhang2026 shows erasure-corrected decoding with no postselection, and,
   separately, postselection on erasure flags.
7. Finding 17: give 99.16% as SPAM-corrected with erasures subtracted.
8. Appendix A: include the code that prints the full "Key output", or trim
   the output block to what the script prints.
