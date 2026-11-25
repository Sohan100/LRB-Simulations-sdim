# R4-completeness: review of REPORT.md against the user's request and the candidate list

Reviewer: R4-completeness, 2026-10-28. Lens: completeness against the request and the task's candidate list, plus readability for a physicist who did not follow the study. Sections 13.1 and 13.2 were ignored, as instructed.

Every number below can be regenerated with
`python notes/report-review/R4-completeness/r4_evidence.py` (run from `studies/correction-decoders/qutrit` after `module load python/3.11`; a few seconds, committed result files only). The output is saved in `notes/report-review/R4-completeness/r4_evidence_output.txt`, and evidence blocks are cited as E1 to E13. Two helper scripts sit in the same folder: `check_bib.py` (Section 14 against `bibliography_raw.csv`) and `m0_effect.py`. I edited nothing in REPORT.md, `src/`, `scripts/` or `tests/`.

## Verdict

The report covers the core of Q1 well: exact circuit-level MLD, the Bayes frontier, BFD, causal vs offline decoding, the Pauli-frame argument and the distance-3 question all have evidence. The gaps are elsewhere:

- **Q2 is the weakest part.**
  - There is no Markovian surrogate test.
  - The m = 0 boundary transient, which both A4 and N1 flagged, is dropped from the tables, so the summary's "single exponential" verdict for `unif 1` and BFD is wrong as stated.
  - The definition calls an m-dependent check schedule a memory carrier, although A4 and N1 classify it as protocol time-inhomogeneity.
  - The controlled test of "does correction itself mitigate?" (MLD with terminal-round information only) exists in the data but is not shown, and it shows strong acceleration.
  - The leakage counter-effect, that correction exposes the memory which detection hides, is missing.
- **Q3 is incomplete against the request.**
  - Of the "other platforms", only trapped ions are covered.
  - Mid-circuit measurement and reset *requirements* are not stated, only availability.
  - The latency claim cites no verified source.
  - No decoder numbers are given at the p where hardware actually is.
- **Candidates from the task list.**
  - Complementary-gap postselection is never named or compared.
  - The circuit-level "correct time-like, reject ambiguous space-like" hybrid is never evaluated as a rule.
  - Heralded-erasure decoding is absent.
- **Wrong numbers found along the way.**
  - The p range quoted for full MLD is wrong.
  - The Section 4.3 MLD row is wrong at p = 1e-2.
  - The hybrid's p^2 coefficient is quoted in the wrong unit.

## Coverage checklist

| Item (user request / task list) | Status | Where / finding |
|---|---|---|
| Circuit-level MLD on the q-ary DEM | covered | §2.1–2.4, tests collect (11 tests), `forward.py` |
| Correct time-like faults, reject ambiguous space-like ones | partial | BFD (§2.2), code-capacity hybrid (§3.1), leading-order hybrid (§3.3); no evaluated circuit-level hybrid (R4-11) |
| Soft-output / complementary-gap postselection, Pareto curve | partial | soft MLD and frontier covered (§4.2); complementary gap absent (R4-10); frontier coverage overstated (R4-21) |
| Z_3 matching, union-find, GF(3) BP, tensor networks, lookup MLD | partial | one-line cost argument (§4.4, §9); GF(3) and lookup never named (R4-19) |
| Erasure / leakage heralding, flags | partial | flags (§3.2) covered; erasure decoding absent (R4-12); leakage under correction absent (R4-07) |
| Is distance 3 required | covered | §8 |
| Pauli frame | covered | §7.1 |
| Causal vs offline | covered | §4.5 |
| Precise definition of non-Markovianity | partial | §5.1 (R4-05) |
| Depth-windowed rates | covered, selectively | §5.3–5.4; first window omitted (R4-01) |
| Check-spacing dependence | covered | §5.3; sparse-policy r values not meaningful (R4-08) |
| Markovian surrogate | **missing** | R4-04 |
| Literature frameworks | covered | §5.1, §5.6 |
| Signatures in the existing LRB-D fits | partial | `const c` yes; m = 0 transient and rising-fidelity windows missing (R4-01, R4-08) |
| Offline frame vs real-time feedback | covered | §7.1 |
| Mid-circuit measurement and reset requirements | partial | availability only (R4-14) |
| Leakage (hardware) | partial | §7.4 (R4-07, R4-12) |
| Latency | partial, unsourced | §7.1 (R4-15) |
| IQM | covered | §7.5 |
| Other platforms | **mostly missing** | ions only (R4-13) |
| Performance at hardware p | missing | R4-16 |
| Refuted ideas | covered, incomplete | §9 (R4-28) |
| Verified bibliography | covered, gaps | §14 (R4-15, R4-27) |
| Open questions | covered, incomplete | §11 (R4-29) |
| Reproducibility | mostly covered | §12 (R4-17, R4-26) |
| Definitions of BFD, slot, h, r per depth step | partial | BFD ok; slot vague; h implicit; r mixed with 1 − f and ε (R4-22, R4-23) |
| Figures | all six paths exist and match captions | minor issues (R4-25) |
| Executive summary self-contained | partial | R4-24 |

---

## Findings (most severe first)

### R4-01 [critical] `unif 1` and BFD are called single exponentials, but the m = 0 to 2 window is about twice every later window

- **Quote (exec summary, lines 84–86):** "LRB-D with a check every round, and BFD, decay as a single exponential within statistics on the 9 production depths." **§5.3 table:** "| LRB-D `unif 1` | 2.6e-5, 2.2e-5, 2.7e-5, 2.8e-5, 2.9e-5, 2.6e-5, 2.9e-5 | … | consistent with one exponential |"
- **Problem.** The §5.3 windows column is labelled "m = 2 to 22". For `unif 1`, BFD and soft MLD it silently drops the m = 0 to 2 window. For `const 1`, `const 0` and full MLD that window is kept, and it produces the "18x", "26x" and "−15%" headline numbers. The dropped window is the largest one for `unif 1` and BFD.
  - The exact data resolve it at every p.
  - It biases the fitted r, which includes m = 0.
  - It survives the "checks inside the inverse half" fix.
  - Only full terminal readout removes it.
  - Both source notes flagged it: A4 ("the m = 0 point sits 2.4 to 6.6 sigma above the single-exponential extrapolation from m ≥ 2 (a boundary effect)") and N1 (z_m0 = 6.9 and 5.7 at p = 5e-3 and 7.5e-3; "biases the pipeline r upward by 50 to 65 %").

  The user asked whether the existing LRB-D fits already show signatures, and this is the main one for `unif 1`. The report never mentions it. It is visible in the report's own figure `windows_baseline_p0.001.png` (first blue point at 5.1e-5).
- **Evidence.**
  - E3: `lrbd:all`, first window 5.12e-5 (CI 4.7e-5 to 5.5e-5) against a later median of 2.71e-5 at p = 1e-3, a ratio of 1.79 to 1.89 for p = 5e-4 to 5e-3. BFD: 1.65 to 1.67. Soft MLD t = 0.9: 4.39 at p = 5e-3.
  - E4: an m ≥ 2 refit lowers r by 8 to 11% for LRB-D and BFD, and by 71% for soft MLD at p = 5e-3.
  - E5 (dense exact study): the window spread including [0,1] is 6.4 (baseline), 2.2 (checked inverse half) and 1.56 (parallel + terminal) at p = 1e-3, against the quoted 3.8, 1.16 and 1.56 without it.
  - N1 §4.2 and §5 item 4; A4 summary item 5 and §5.3 (T5 z0 = +2.7 for `unif 1` at p = 1e-3).
- **Fix.** Report the first window for every rule. Restate the verdict: "`unif 1` and BFD are single exponentials for m ≥ 2. The m = 0 to 2 window decays 1.7 to 1.9 times faster (a boundary effect: the m = 0 circuit has no forward round, and N1 conjectures a two-fault mechanism that exists only for m ≥ 1), and this inflates the all-depth fit by about 10% (by 50 to 65% in the pipeline's fit at p ≥ 5e-3)." Recommend excluding m = 0 from rate fits, or a free amplitude on m ≥ 2. Note in §5.4 that checking the inverse half does not remove the first-window excess, while full terminal readout does.

### R4-02 [critical] Full MLD was not simulated up to p = 2e-2

- **Quote (exec summary, lines 44–45):** "and it is worse than physical RB at every p simulated (1e-4 to 2e-2) **[VN]**."
- **Problem.** All sampled rules (full MLD, soft MLD, causal decoders, raw) were run only at 7 values, up to p = 1e-2. Only the exact rules (LRB-D, BFD, `const 0` MLD) go to 5e-2.
- **Evidence.** E1: `mld:all` n = 7, max p = 0.01. `run_all_on_node.sh` uses `--sampled-probs 1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2`. §2.3 states the same grid.
- **Fix.** "(1e-4 to 1e-2)".

### R4-03 [critical] §4.3 MLD row is wrong at p = 1e-2

- **Quote (§4.3):** "| Full MLD | 11 to 15 p, 3 to 4 times above physical RB at every p | 0 |"
- **Problem.** At p = 1e-2, full MLD has r = 0.174 = 17.4 p, which is 4.74 times physical RB. At p = 1e-4 the ratio is 2.92.
- **Evidence.** E2: MLD/RB(5p) = 2.92, 3.15, 4.09 and 4.74 at p = 1e-4, 1e-3, 5e-3 and 1e-2. r/p = 11.0, 11.8, 15.2 and 17.4.
- **Fix.** "11 to 17 p, 2.9 to 4.7 times above physical RB (ratio growing with p)".

### R4-04 [major] No Markovian-surrogate test (a requested item), although A4 designed and ran one

- **Quote (§2.5):** "Non-exponential shape is measured two ways: depth-windowed decay factors between consecutive depths, and the quadratic coefficient `c2` of `log lambda` against m (bootstrap CI)."
- **Problem.** The task asks for a Markovian surrogate. The report's tests (windows, c2) detect non-exponentiality, not memory. A4 §5.2 defines two surrogate tests:
  - T4: B f^m fitted on m = 2, 4, 6 and extrapolated to m = 22, giving Δ_M.
  - T9: the exact `unif 1`-like surrogate exp(−a1 p − (α + β1 m) p^2), which resets the syndrome by an ideal check every step. Its deviation for the other policies is the "memory penalty", (k−1) β1 m p^2 for `unif k` and β1 (m^2/c − m) p^2 for `const c`.

  A4 also explains why Figueroa-Romero's Markovianisation (trace out E each step) is the wrong surrogate here. A4 §5.3 already has Δ_M at p = 1e-3. None of this is in the report.
- **Evidence.** A4 §5.2 table (T4, T9) and the text under it. A4 §5.3 Δ_M: `unif 1` −1.5e-4 [−5.1e-4, +1.8e-4]; `unif 4` −2.9e-4 [−7.1e-4, +9.5e-5]; `const 1` −1.9e-2 [−2.0, −1.9]e-2; `const 2` −6.6e-3; `const 0` −8.1e-4. `grep -i surrogate REPORT.md` finds no match.
- **Fix.** Add a §5.3 paragraph: define the surrogate (T9, ideal reset of σ each step), give the memory-penalty formula, and tabulate Δ_M by policy (A4 §5.3). Say why tracing out E is not a valid Markovianisation here. Label it [VA] for the construction and [VN] for Δ_M.

### R4-05 [major] The definition treats m-dependent check schedules as a memory carrier, contrary to its own sources, and omits the effects that mimic non-Markovianity

- **Quote (§5.1):** "3. the check schedule, when the policy makes it depend on m." **Exec summary:** "Non-Markovianity appears at the logical level, through the code block's syndrome register (…) and through check schedules that depend on m."
- **Problem.** The user asked for a precise definition. A4 §1.3 lists three effects that are *not* non-Markovianity but mimic it in LRB-D:
  - gate dependence (Markovian, Wallman);
  - protocol time-inhomogeneity: the depth-m circuits are not prefixes of one process, so each circuit can be Markovian while the family over m is not a semigroup;
  - estimator effects (the |·| metric, mean of ratios).

  N1 §5 item 3 says the sparse-policy deviations "are a property of the policy definition, not evidence about the noise". The report merges the second effect into "memory", drops the list, and never states the logic that connects the observable to the property: non-exponential decay, minus the mimics, implies memory.
- **Evidence.** A4 §1.2 (Definitions M and NM), A4 §1.3, N1 §5 item 3.
- **Fix.** Keep Definition M. Add Definition NM (no factorised process tensor; Figueroa-Romero's N_q). Add the three mimicking effects. Move the check schedule from "memory carriers" to "protocol time-inhomogeneity (mimics non-Markovianity; removed by a fixed schedule, not by correction)". Change the exec-summary Q2 sentence to match.

### R4-06 [major] "Correction mitigates non-exponentiality" is not tested with matched information; the controlled comparison exists and shows the opposite for terminal-only correction

- **Quote (exec summary, lines 91–94):** "**Correction mitigates non-exponentiality in shape only.** It turns the quadratic pair mechanism into first-order logical errors … MLD's windowed rate drifts by 15% (downward) where `const 1` grows 18 times."
- **Problem.** The comparison sets MLD on *all* rounds against detection on the *terminal round only*. The controlled pair is MLD restricted to the terminal round (`mld:terminal`, already in `fits_baseline.csv.gz`) against `const 1`. That decoder still accelerates.
  - Its curvature c2 = −3.0e-4 [−3.4e-4, −2.7e-4] is 4.4 times larger in magnitude than `const 1`'s −6.8e-5.
  - Its windowed rate rises 59% (2.29e-2 to 3.64e-2).

  So the pair mechanism is removed by *using the intermediate rounds*, which `unif 1` detection also does, not by correcting. N1 states this as a testable prediction ("the same correction applied only at the end should show [the m^2 term]"), and it was not reported. The report gives only `mld:terminal`'s r (2.01e-2, §4.1).
- **Evidence.** E6: `mld:all` c2 = +5.4e-5, last/first = 0.85; `mld:terminal` c2 = −3.0e-4, last/first = 1.59; `lrbd:const1` c2 = −6.8e-5, last/first = 18.0. `key_summary.md` lists `mld:terminal` c2 = −1.07e-4 at p = 5e-4. N1 §5, "Testable predictions".
- **Fix.** Add `mld:terminal` (and `mld:unif4`) to the §5.3 table. Rephrase: "With the same syndrome information, correction keeps (and enlarges) the pair curvature. What removes it is using every round, by detection or by correction. Correction then makes the relative drift small because its first-order rate dominates."

### R4-07 [major] The leakage counter-effect is missing from the Q2 answer, and §7.4's conjecture points the other way from the literature note

- **Quote (§7.4):** "Leakage is persistent memory, the most non-Markovian error there is; detection without leakage reduction lets it poison many rounds **[CONJ]**." **§5.6 item 3:** "**Correction removes a different, more dangerous non-Markovianity:** postselection's selection of the quiet part of temporally correlated noise".
- **Problem.** lit-L7 finding 6 says leakage is "the dominant non-Pauli, time-correlated error in superconducting QEC. Detection hides it; correction exposes it." Postselection discards leaked runs (Marques 2022). Under correction, leakage produces detection events correlated over more than 10 rounds, and "under 1% injected leakage pushed the logical error above 40%" (McEwen 2021, Miao 2023). For the user's Q2 this is a non-Markovian effect that *correction makes worse* on hardware. The verdict in §5.6 presents only the effect that favours correction.
- **Evidence.** `notes/lit-L7-erasure-flags.md` §0 item 6 and Part C. McEwen2021 and Miao2023 are already in §14.
- **Fix.** Add to §5.6: "On hardware, moving from detection to correction exposes leakage-induced memory that postselection hides [LIT: McEwen 2021; Miao 2023; Marques 2022]. Correction-based LRB therefore needs leakage heralding (four-level readout) or leakage reduction." Rewrite the §7.4 [CONJ] to say what detection does to leakage (rejection, lower acceptance) and what correction does (accepted, mis-corrected shots).

### R4-08 [major] The check-policy table quotes "decay rates" fitted to non-monotone curves

- **Quote (§4.7):** "| `const 22` | 4.5e-5 | 0.916 | 9.8e-5 | 0.932 |" and "**LRB-D trades acceptance for error through the policy. BFD barely does:**"
- **Problem.** For c > m, `const c` is `const 1` (terminal only). So `const 22` is the accelerating `const 1` curve up to m = 20, and at m = 22 it uses 22 of the 23 rounds. The polarization *rises* by 2.0% from m = 20 to 22 (0.97197 to 0.99166; windowed 1 − f = −1.0e-2 per step). Its "r = 4.5e-5" is not a decay rate. `const 4` (window [2,4]) and `unif 4` (window [18,20]) also have rising-fidelity windows with CIs below zero. `results/baseline_nonmarkov/rising_windows.csv` lists 156 significantly rising windows across 30 policies in the production data (N1's text says 158). N1 wrote: "'the LRB-D decay' is not one experiment repeated at different m … A single f fitted across m mixes protocols." A4 says rising fidelity "does not witness non-Markovianity in postselected RB with m-dependent schedules". This is a signature in the existing fits, which the user asked about, and the report does not mention it.
- **Evidence.** E8: `lrbd:const22` window [20–22] = −1.0e-2; `lrbd:const4` [2–4] = −1.0e-5; `lrbd:unif4` [18–20] = −3.2e-5; `bfd:const22` [20–22] = −9.5e-3. `curves_baseline.csv.gz`, signed mean of `lrbd:const22` at m = 20 and 22. `rising_windows.csv` (156 rows). N1 §1 items 3 and 6; A4 summary item 5.
- **Fix.** Mark `const c ≥ 2` and `unif k ≥ 3` rows "not a decay rate: the policy changes with m" (or drop the r column for them). Quote acceptance and conditional error at fixed m instead. Add the rising-fidelity observation to §5.3 as a protocol artefact.

### R4-09 [major] Headline comparisons with "physical RB" depend on an unexplained `p_RB = 5p` convention; the caveat is given only for `const 0`

- **Quote (preamble):** "Physical RB at the repository's matched noise `p_RB = 5p` has `r = 3.76 p` per depth step." **Exec summary:** "a 0.9 threshold gives 3.1% rejection per step and `r = 9.4e-4` (4 times below physical RB …)".
- **Problem.**
  - The report never says what the physical RB circuit is: one qutrit, SI1000 single-qutrit gate noise 0.1 p_RB per gate, so 0.5 p per gate. It also never says why 5p is "matched". The pipeline treats 5p as a display-axis scale ("LRB lambda = 5 * raw_p", `scripts/run_physical_rb_comparison.py`).
  - Break-even claims flip under the other natural choice. With p_RB = p at p = 1e-3:
    - soft MLD t = 0.9 is 1.27 times *worse* than physical RB, not 4 times better;
    - `const 0` MLD is 2.25 times worse, not 0.45 times;
    - only LRB-D, BFD and soft MLD t ≥ 0.97 stay below.

  §4.6 gives this caveat for `const 0` only, and §8's break-even estimates have the same dependence.
- **Evidence.** E2 (columns /RB5p and /RBp). `rb_reference.csv` (rb_r = 7.41e-4 at p_RB = 1e-3). Brief, "Physical RB comparison uses p_RB = 5p".
- **Fix.** In §1, define physical RB (circuit, noise per gate, depth step) and the rationale for 5p. In §4.3, add a column or sentence giving the ratios at p_RB = p. Attach "(at the p_RB = 5p convention)" to every "below or above physical RB" statement in the summary, §4 and §8.

### R4-10 [major] Complementary-gap postselection (task candidate) is never named or compared

- **Quote (§2.2 table):** "| Soft MLD (threshold t) | accept iff `max_L P(L given s) >= t`, apply the argmax | …"
- **Problem.** The task lists "soft-output/complementary-gap postselection". The report implements the max-posterior (Chow/Forney) rule but never says how it relates to the complementary gap used in the literature. lit-L4 has the qutrit-specific analysis:
  - with three logical classes, the two-class complementary gap g1 is Forney's *suboptimal* likelihood-ratio rule;
  - the optimal rule thresholds −log(e^−g1 + e^−g2);
  - the two statistics differ by at most log 2 and coincide for qubits.

  The verified bibliography holds the gap literature (BPRS24, GNBJ25, SBB24, MPP24) and the hardware precedent of "sliding-scale" postselection on decoder output (Bluvstein 2024). None is cited.
- **Evidence.** `notes/lit-L4-softoutput.md` §0.2 item 5, §5.3, line 104. `bibliography_raw.csv` rows BPRS24, GNBJ25, SBB24, MPP24 and Bluvstein2024 (all confirmed). `grep -ci "complementary" REPORT.md` returns 0.
- **Fix.** Add a §4.2 paragraph labelled [VA, LIT]: "The complementary gap … is Forney's suboptimal rule for q = 3, within log 2 of the optimal statistic; we use the optimal one." Optionally compute the gap rule's frontier from the stored posteriors. Cite BPRS24, GNBJ25, SBB24, MPP24 and Bluvstein 2024.

### R4-11 [major] "Correct time-like faults, reject ambiguous space-like ones" is never evaluated as a circuit-level rule; the three-regime statement does not carry over

- **Quote (§3.1):** "The posterior-threshold family has only these three regimes at code capacity." **§3.3:** "| Hybrid rejection | about 39 p |"
- **Problem.** This task candidate is A1's hybrid ("correct when one coset is the unique leading explanation, reject otherwise"). It appears only as leading-order coefficients (§3.3), with no curve, fit or acceptance. BFD is a different rule: it forgives time-like blips but rejects even unambiguous space-like errors. At circuit level the posterior-threshold family is continuous, not three-regime: at p = 1e-4, rejection per step is 13 p (t = 0.5), 25 p (t = 0.9) and 41 p (t = 0.99). A1's 39 p hybrid therefore corresponds to t ≈ 0.99, not to the t = 0.9 used as the report's example. A reader cannot tell which evaluated rule implements the requested idea.
- **Evidence.** E10. A1 §0 items 2 and 5, A1 §9 (hybrid rejection slope 39.3 p). `grep -i "space-like" REPORT.md` finds no match.
- **Fix.** In §3.3 or §4.1, state that the circuit-level hybrid is the soft-MLD family near t = 0.99, and give its measured r and acceptance. Say explicitly that "time-like faults are always corrected (§3.2), ambiguous space-like ones rejected" is what soft MLD at high t does. Note that the three-regime statement holds at code capacity only.

### R4-12 [major] Heralded-erasure decoding is missing: the d = 2 code corrects any single heralded erasure

- **Quote (§7.4):** "The decoder treats a flagged shot as rejected (hybrid rule) or marginalizes the flagged qutrit."
- **Problem.** The task asks about erasure and leakage heralding. lit-L7 finding 5 [num] shows that heralded erasures turn this d = 2 code into a *correcting* code for heralded errors:
  - any single erased qutrit is correctable;
  - 6 of 10 erased pairs are correctable, all except {0,2}, {0,3}, {2,4} and {3,4}, which support weight-2 logicals.

  It also gives realistic herald quality: dual-rail false positives 0.6 to 0.8% and false negatives 1.5% (Levine 2024); leakage-detection error 7.7e-3 (Chou 2024); erasure fractions 56% and 33% (Ma 2023). This is the main reason a four-level readout flag helps a d = 2 code, and the report omits it. "Hybrid rule" is also used here in a different sense from §3.1.
- **Evidence.** `notes/lit-L7-erasure-flags.md` §0 item 5 and Part B.
- **Fix.** Add to §7.4: "With |3> outcomes treated as erasures, a single erased qutrit (and 6 of 10 pairs) is corrected, not rejected [VN: notes/lit-L7]." Give the herald-quality numbers [LIT]. Rename "(hybrid rule)" to "rejected".

### R4-13 [major] "Other platforms" covers only trapped ions; H2's platform assessment and conclusion are dropped

- **Quote (§7.3):** "Ion chains and movable atom arrays need no routing." **Exec summary:** "…and trapped-ion qutrit gates to about 0.014."
- **Problem.** The user asked for "superconducting qutrits incl. IQM, plus other platforms", and the brief names photonics, Rydberg and bosonic qudits. H2 §2 assesses each, and the report mentions none.
  - **Neutral atoms:** no demonstrated two-qudit gate (only proposals).
  - **Photonics:** gates succeed with probability 1/d, so a depth-22 shot succeeds with about 10^−197.
  - **Bosonic GKP qutrits:** beyond break-even but single-mode, with no inter-mode gates (Brock 2025, verified in `bibliography_raw.csv`).
  - **Spin and molecular qudits:** single objects only.
  - **Qubit-emulated qutrits on Quantinuum H2:** feasible, but about 20% error per round.

  H2's conclusion, that trapped ions are the most sustainable platform for offline-decoded correction at 1e3 to 1e4 shots per circuit (18 h to 7.6 days for 270 circuits), conflicts with the report's IQM-centred staging, and this is not discussed. "Movable atom arrays need no routing" implies viability that H2 rejects.
- **Evidence.** `notes/H2-other-platforms.md` §0 items 3, 4 and 8; §2.2 to 2.7, including the comparison table. `grep -ci "photon\|bosonic\|GKP\|Quantinuum" REPORT.md` returns 0.
- **Fix.** Add a §7.x "Other platforms" with H2's §2.7 table (condensed), labelled [LIT] for the platform facts and [CONJ] for the mappings. State H2's ion-vs-transmon conclusion and reconcile it with Stage A to C. Cite Brock 2025 and the verified neutral-atom sources.

### R4-14 [major] Mid-circuit measurement and reset *requirements* are not stated, only what exists

- **Quote (§7.1):** "The ancilla reset can be dropped. With QND readout an un-reset ancilla returns `m + s` (mod 3), so detectors become differences of consecutive outcomes".
- **Problem.** The user asked for mid-circuit measurement and reset requirements. The report gives "best published" values (§7.2) and the condition "with QND readout", but no requirement. Available but omitted:
  - **Gate error:** H1, "Usable postselected depths (m ≈ 10 at ≥10% acceptance) need two-qutrit errors of about 0.5% … or a decoder that keeps the rejected shots" [VN].
  - **Calibrations (H1 §7.0):** QND-ness by repeated measurement, spectator error per readout layer (qutrit mid-circuit-measurement RB), multi-level reset target 1e-3, and |3> leakage RB per gate.
  - **SI1000 vs hardware:** H1 §0 item 3 shows readout error of 2 to 3% against 13.5% assigned by SI1000 at p = 0.027, and idle error per readout layer of 0.5 to 1.1% against 5.4%. These mismatches set which SI1000 conclusions transfer.
- **Evidence.** `notes/H1-sc-hardware.md` lines 423–425, §0 item 3, §7.0. `grep -ci "spectator\|requirement" REPORT.md` returns 0.
- **Fix.** Add a "Requirements" table to §7: two-qutrit error, ternary readout fidelity and duration, spectator error per layer, QND-ness if reset is dropped, reset residual and leakage herald. Give the target values from H1 with labels, and the SI1000-vs-hardware ratios.

### R4-15 [major] The latency claim carries a bare "[LIT]" with no citation; verified latency sources exist but are not used

- **Quote (§7.1):** "It would then need sub-µs latency: a waiting transmon qutrit loses 1.4 to 9% per µs, and current FPGA loops take 0.2 to 0.6 µs [notes/H1, notes/H2, LIT]."
- **Problem.**
  - No paper is named. The H2 sources behind "0.2 to 0.6 µs" (St21, X23, ZI-FB, ZI-PQSC, QB, Cao26, …) were never citation-checked (there is no H2 citecheck) and are not in `bibliography_raw.csv`, which breaks the report's own [LIT] definition.
  - "1.4 to 9% per µs" is [VN] under a coherence model in H1, not [LIT].
  - The verified lit-L5 latency sources are unused: Battistel 2023, Google QAI 2025 (63 µs decoder), Barber 2025, Caune 2026, Liyanage 2023/2024 (FPGA), Ryan-Anderson 2021, Sivak 2023, and Chamberland 2018 (Pauli/Clifford frame with slow diagnostics, directly on the Pauli-frame point).
  - H1's latency table (readout 0.14 to 0.5 µs, discrimination 110 ns, FPGA decoder under 1 µs per round, software decoder 63 µs) and its point that a w-round lookup has 81^w entries are also omitted.
- **Evidence.** `bibliography_raw.csv` topic lit-L5-realtime (22 entries). Of these only DiVincenzoAliferis2007, Riesebos2017 and Geher2025 appear in §14, with Knill under an L4 key; Battistel2023, GoogleQAI2025, Barber2025, Caune2026, Liyanage2023/2024, RyanAnderson2021, Sivak2023 and Chamberland2018 do not. H1 §5 table. H2 §4.1 and §0 item 7.
- **Fix.** Replace with H1's latency table. Cite verified sources (Liyanage 2023/2024, Caune 2026, Google QAI 2025, Battistel 2023, Chamberland 2018). Tag "1.4 to 9% per µs" as [VN, notes/H1]. Drop or re-verify the H2-only sources.

### R4-16 [major] No decoder numbers at the p where hardware is, although exact results exist up to p = 0.05

- **Quote (§7.2):** "At these p the baseline keeps less than 0.1% of shots by m = 2 (transmons) or m = 4 (ions) **[VN, notes/H2]**." **§4.3 table:** "| Soft MLD, t = 0.9 | between first and second order; 4 to 7 times below physical RB from p = 2e-4 to 1e-2 | |" (rejection cell empty).
- **Problem.** The report places today's hardware at p ≈ 0.014 to 0.027, but its decoder comparison stops at 1e-2, and only acceptance is given at hardware p. The committed exact fits answer the question directly:
  - BFD's conditional error equals physical RB at p = 0.02 (0.98x) and is 2.5 times worse at 0.03.
  - LRB-D is 0.32x and 0.85x at 0.02 and 0.03.
  - `const 0` MLD stays at 0.46x.
  - Soft MLD's acceptance advantage over BFD disappears by p = 1e-2 (rejection 0.588 vs 0.544 per step).
  - With the baseline protocol, t = 0.97 rejects more than LRB-D at p = 5e-3 (0.754 vs 0.427), and accepts nothing at any depth at 1e-2.
  - The sampled rules recommended for Stage C (soft and full MLD) were never simulated above 1e-2.
  - Stage C's fixed "t in 0.9 to 0.97" also conflicts with §4.8's advice to set t by an acceptance quantile.
- **Evidence.** E2 (rows p = 0.01, 0.02, 0.03). E9 (rejection per step against p). E1 (sampled rules stop at 1e-2).
- **Fix.**
  - Fill the empty §4.3 cell (soft t = 0.9: 24 to 59 p, i.e. 0.0025 to 0.59 per step).
  - Add a "hardware-p" row block (p = 0.0125 to 0.03) for the exact rules.
  - State that the soft-MLD rules are untested there.
  - In Stage C, specify t by acceptance quantile.

### R4-17 [major] Two committed result files are not produced by any listed command, and one contradicts the report

- **Quote (§12 table):** "| `results/` | `decoder_tables.md`, `variant_summary.md`, `key_summary.md`, `curves_*.csv.gz`, … |"
- **Problem.**
  - `key_summary.md` has a column "physical RB r at 5p" equal to 0.00492 at p = 1e-3 and 0.00226 at 5e-4. These are the free-offset values (`rb_free_r`) that §2.5 and §10.3 call ill-conditioned; the report uses 3.76e-3 and 1.88e-3. The file also labels the LRB r "per Clifford", while the report calls the same numbers "per depth step".
  - No script in `studies/correction-decoders/qutrit/scripts/` writes `key_summary.md` or `variant_summary.md`. Only `decoder_tables.md` is written (by `make_tables.py`).
  - `variant_summary.md` is the cited source for §6 and the §2.4 validation row.
- **Evidence.** E12. `grep -n "variant_summary\|key_summary" studies/correction-decoders/qutrit/scripts/*.py` finds no match. `key_summary.md` header line 3 and its table.
- **Fix.** Regenerate `key_summary.md` from `rb_reference.csv` (offset 0) with "per depth step", or delete it. Add the generator of `variant_summary.md` (and of `key_summary.md` if kept) to `make_tables.py` and the §12 commands.

### R4-18 [major] The hybrid's p^2 coefficient is quoted as r but is 1 − f

- **Quote (§3.3):** "Accepted-shot errors stay second order for LRB-D (`r` about 20 p^2), BFD (about 53 p^2) and the hybrid (about 1.3e3 p^2)"
- **Problem.** A1 §9 gives the hybrid as "1 − f ≈ 1.5 × (0.85–0.89)·10³ p² ≈ 1.3·10³ p²". In the report's r it is about 0.87e3 p^2. The LRB-D and BFD numbers in the same sentence are r. §8 then uses a third convention, "about 27 p^2 per step" for LRB-D: A5's ε = (4/3) r, a depolarizing-equivalent probability. So the same LRB-D quantity appears as r ≈ 20 p^2 (§3.3, §4.3) and ε ≈ 27 p^2 (§8), and the §5.3 windows use yet another unit, 1 − f = 1.5 r.
- **Evidence.** E13. A1 §9 "Implications" table. A5 §3.3 ("eps = (4/3) r = 27.1 p^2").
- **Fix.** "hybrid (r about 0.87e3 p^2)". Define ε in §8 or convert to r (LRB-D r ≈ 20 p^2; the d = 3 estimate becomes about 68 to 1,400 p^2 in r).

### R4-19 [minor] The decoder families are dismissed in one unlabelled line; GF(3) BP and lookup MLD are never named

- **Quote (exec summary):** "Approximate decoders (matching, union-find, BP, tensor networks) are therefore unnecessary." **§4.4:** "| Not recommended | | Matching, union-find, BP, tensor networks | Unnecessary when exact MLD costs microseconds per round. |"
- **Problem.** The task asks for each family to be implemented or argued away. Cost alone is a valid argument, but lit-L3 §0.1 to 0.3 has the specific reasons:
  - binary-node matching on Z_3 is non-linear and mis-models two faults on one detector;
  - union-find and HDRG collapse to one cluster at n = 5;
  - non-binary BP over GF(3) is approximate on a small loopy DEM, with no qutrit benchmarks;
  - tensor-network contraction at n = 5 is exact and equals MLD;
  - code-capacity lookup MLD is an 81-entry table, and a w-round history lookup has 81^w entries.

  None of this is in the report. The claim lines carry no label. Keppens 2025, the source of binary-node belief-matching for qudits, is in §14 but never cited.
- **Fix.** Add a five-row table to §4.4 or §9 with the lit-L3 reasons, labelled [VA] or [LIT], and cite Keppens 2025 in the matching row.

### R4-20 [minor] BFD-X and LRB-D-X are defined but their results never appear

- **Quote (§2.2):** "| BFD-X, LRB-D-X | as above, but in the final round only the X-check digits must be trivial (an X error left at the end cannot change the X readout) | exact, allowed-set observation |"
- **Problem.** Results exist (`lrbd:xtriv`, `bfd:xtriv` in every fits file) but are absent from all report tables, `decoder_tables.md` and `variant_summary.md`. They answer lit-L7's open question (finding 7: "the second-order cost is not known"): relaxing the final round costs 2.5 to 2.8 times the error rate for +1.6 points of acceptance at m = 0 and +4 to 6% relative at m = 22.
- **Evidence.** E7: r = 5.69e-5 vs 2.00e-5 (LRB-D) and 1.31e-4 vs 5.30e-5 (BFD); acceptance at m = 22 is 0.0842 vs 0.0807 and 0.2025 vs 0.1914.
- **Fix.** Add the two rows to §4.1 and an entry to §9 ("ignore final-round Z checks: rejected [VN]").

### R4-21 [minor] Pareto claims go beyond what the sampled frontier shows

- **Quote (§4.2):** "and holds within the precision of the sampled frontier at m = 6 and 22." and "BFD sits on it".
- **Problem.**
  - The sampled frontier is traced only down to acceptance 0.535 (m = 6) and 0.093 (m = 22), while `unif 1` sits at 0.483 and 0.081, left of where the frontier begins (top histogram bin). The figure cannot show `unif 1` "on" it, only that no sampled threshold dominates it.
  - BFD lies 1.5% (m = 6) and 4.1% (m = 22) above the interpolated frontier.
- **Evidence.** E11. `results/figures/pareto_p0.001.png`.
- **Fix.** "At m = 6 and 22 no sampled threshold dominates `unif 1`. The frontier is resolved only down to acceptance 0.54 and 0.09. BFD lies within 2 to 4% of it."

### R4-22 [minor] Key definitions for a reader who did not follow the study are missing or inconsistent

- **Quote (preamble):** "\"Per depth step\" means one forward Clifford plus one inverse Clifford, because a depth-m circuit has 2m noisy Cliffords." **§1:** "The `const c` and `unif k` policies only choose which rounds' outcomes reject a shot".
- **Problem.**
  - `unif k` and `const c` are never defined (which rounds, i.e. j·k − 1 and j·(m//c) − 1, plus the terminal round), although §4.7 and §5.3 depend on them; for example, why `unif 22` equals `const 1` for m < 22.
  - "Per depth step" omits that each depth step also adds a full check round (118 p, the dominant noise). The `rate_vs_p.png` title repeats the incomplete definition.
  - "Acceptance per step" (a fitted per-step factor with a free intercept, `acc_per_step`/`acc_c`) is never defined.
  - The symbol a is used for the syndrome shift (§2.1), acceptance per step (§2.4, §5.4, §6) and amplitude A (§4.6, §10.4).
  - Three per-step error measures are mixed (R4-18).
  - "slot" ("the first round it touches") and h/s_r in the §2.1 recursion are only implicit.
  - The BFD definitions in the summary ("forgives a syndrome seen in one round only") and in §2.2 ("final round trivial and no two consecutive rounds non-trivial") are not obviously the same rule.
- **Fix.**
  - Add a short "Protocol and notation" box to §1 with the round sets of `unif k` and `const c`, depth step = forward Clifford + check round + reset + inverse Clifford, acceptance per step = a from acc(m) ≈ c·a^m, and r vs 1 − f vs ε.
  - Rename one of the "a" symbols.
  - Define s_r and h_r before the recursion.

### R4-23 [minor] Unexpanded acronyms and unexplained jargon

- **Quote (exec summary):** "removes a depth-independent `4.95 p` SPAM floor".
- **Problem.**
  - Never expanded: LRB-D (detection-only logical RB), LRB/RB, DEM (detector error model), MAP.
  - "SPAM floor" is a misnomer: preparation and readout are ideal here, and the floor comes from idle and CNOT faults after the last check.
  - Unexplained: "Rice-biased", "Weyl vectors/twirling", "code capacity", "corner/centre qutrit" (the layout is never shown), α and β in §5.2 ("`(alpha/(alpha + beta))^m`").
  - "The same 729-coset construction" sits next to a 243-state filter without saying that 729 = 81 × 9 logical classes, while 243 = 81 × 3 because only the read-out logical component is tracked.
- **Fix.** Expand each acronym at first use. Replace "SPAM floor" with "final-round floor (faults after the last check)". Add one sentence each for Rice bias, Weyl operators, code capacity and α/β. Add a small figure or table of the 5 data qutrits (corners 0, 2, 3, 4; centre 1) and the 243/729 remark.

### R4-24 [minor] The executive summary is not self-contained

- **Quote (exec summary):** "LRB-D with every round (`unif 1`) already sits on that frontier … Sparse-check LRB-D (`const c`, `unif k >= 2`) is strictly dominated **[VN, VA]**."
- **Problem.**
  - The summary relies on `unif k`, `const c`, LRB-D, DEM, "corner qutrit", "SPAM floor", "lookahead" and "physical RB at 5p" without definitions (see R4-22, R4-23).
  - Its Q2 bullet repeats the R4-01 error.
  - Its Q3 section does not mention platforms other than transmons and ions (R4-13).
- **Fix.** Add a three-line glossary at the top of the summary covering the policies, LRB-D, DEM, physical RB and the depth step, and apply the fixes from R4-01, R4-09 and R4-13 to the summary text.

### R4-25 [minor] Figures: all six exist and match their captions, with small issues

- **Quote (§5.4):** "![Exact windowed rates at every depth, p = 2e-3](results/figures/dense_windows_p0.002.png)"
- **Problem.**
  - The §5.4 table is at p = 1e-3, but the figure shown is p = 2e-3 (`dense_windows_p0.001.png` exists).
  - Its log y axes carry one tick label (1e-4) on the left and none on the right panel.
  - In `rate_vs_p.png` the two panel titles overlap ("…inverse Cliffordn) per depth step").
  - `windows_baseline_p0.001.png` shows the first-window excess that the text omits (R4-01).
- **Fix.** Use the p = 1e-3 dense figure, or say why 2e-3 was chosen. Add y tick labels. Shorten the left title of `rate_vs_p.png` and correct its depth-step definition.

### R4-26 [minor] Reproduction commands are incomplete

- **Quote (§2.5):** "with 95% intervals from 100 bootstrap resamples of the 30 sequences." **§12:** the command block.
- **Problem.**
  - `studies/correction-decoders/qutrit/scripts/mismatch_check.py` (§4.8) is not in the command list.
  - `results/baseline_nonmarkov/` (18 files, written by `notes/scripts/N1_baseline_fits.py`) is neither listed nor generated by a listed command.
  - The §5.3 χ² claim (5,461 vs 68) comes from `notes/scripts/A4_fit_battery.py`, which is not listed.
  - `analyze_study.py` defaults to `--boot 200`, not 100.
  - `run_all_on_node.sh`'s usage line lists "baseline | variants | all", and "all" does not run the dense stage.
- **Fix.** Add the three scripts to §12 and the `baseline_nonmarkov/` directory to the file table. Make the bootstrap count consistent (state 200, or pass `--boot 100`). Fix the usage comment.

### R4-27 [minor] Bibliography gaps

- **Quote (§14 intro):** "or, for the two marked entries, fetched by the lead."
- **Problem.**
  - No entry is marked. The two entries not in `bibliography_raw.csv` are Hrmo2023 and Ringbauer2022, which were not double-fetched and so do not meet the report's [LIT] definition.
  - Keppens2025 is listed but never cited.
  - §7.2 has [LIT] tags without any reference: "readout errors biased toward decay, not uniform shifts [LIT]" and "leakage to |3> [LIT]".
  - Verified sources that support requested items are unused (R4-10, R4-13, R4-15). §14 has 54 entries: 52 from the 214 verified rows plus the two unverified ones.
- **Evidence.** `check_bib.py` output: "Sec14 keys not in CSV: ['Hrmo2023', 'Ringbauer2022']".
- **Fix.** Mark the two entries (for example with a dagger), cite Keppens2025 (R4-19), and add references to the bare [LIT] tags.

### R4-28 [minor] §9 (refuted ideas) misses ideas that have evidence

- **Quote:** "## 9. Ideas tested and refuted or rejected"
- **Problem.** Missing entries:
  - "Ignoring final-round Z checks is free": costs 2.5 to 2.8 times the error, E7.
  - "Correction alone removes the pair mechanism": terminal-only MLD accelerates, E6.
  - "Tracing out the environment each step gives the Markovian reference": wrong surrogate, A4 §5.2.
  - "Rising postselected fidelity witnesses non-Markovianity": a schedule artefact, A4 and N1.
  - "The complementary gap is the optimal confidence for qutrits": Forney's suboptimal rule, lit-L4.
  - "Flags are needed": not for this code, §3.2.
- **Fix.** Add these rows with their labels.

### R4-29 [minor] §11 (open questions) misses items the report itself leaves open

- **Quote:** "## 11. Open questions"
- **Problem.** Missing:
  - leakage and erasure in the simulation (no leakage model; erasure-aware MLD);
  - the Markovian-surrogate test (R4-04);
  - the mechanism of the m = 0 boundary effect (N1 [conj]: an escape after the last forward ancilla combined with a terminal measurement fault);
  - soft and full MLD at p ≥ 1e-2;
  - calibrating the soft threshold by acceptance quantile on hardware;
  - finite (2 to 7%) terminal readout error, now only a [CONJ] in §4.6;
  - IQM ternary readout and feedback (§7.5).
- **Fix.** Add these as bullets.
