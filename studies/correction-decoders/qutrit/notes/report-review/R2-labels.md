# Report review R2-labels: claim labels and overclaiming

Reviewer: R2-labels, 2026-10-28. Lens: does each claim's label match the evidence in the
worktree ([VN] needs a script or result file, [VA] a derivation, [LIT] a paper in the
verified bibliography that says it, [CONJ] otherwise)? Which claims carry no label, which
are stronger than the evidence, and which generalize beyond the simulated model (SI1000,
stochastic Pauli noise, noiseless terminal readout, ideal encoding, `p_RB = 5p`)?

`REPORT.md` was not edited. Sections 13.1 and 13.2 were ignored.

## What I ran

All in `notes/report-review/R2-labels/` (run after `module load python/3.11`):

| script | output | what it checks | time |
|---|---|---|---|
| `label_checks.py` | `label_checks_output.txt` | B1 Section 14 against `bibliography_raw.csv`; B2/B3 full and soft MLD against offset-0 physical RB at every simulated p; B4 rule points against the sampled Chow frontier (`pareto_frontier.csv` against `curves_baseline.csv.gz`); B5 curvature of MLD restricted to fewer rounds; B6 curvature of the quasi-static mixture curves; B7 `const 22` windows; B8 where the reproduction z-tail sits; B9 single-exponential fits of `const c` at p = 1e-4 | seconds |
| `mismatch_vs_frontier.py` | `mismatch_vs_frontier_output.txt` | true-model frontier for the Section 4.8 mismatch shots (same seeds as `studies/correction-decoders/qutrit/scripts/mismatch_check.py`) and the excess infidelity of rate-mis-scaled soft-MLD points | 2 min |

I also read `decoder_tables.md`, `variant_summary.md`, `key_summary.md`, `first_order_check.json`,
`mismatch_check.csv`, `mixture_*.csv`, `fits_*.csv.gz`, the `baseline_nonmarkov/` tables,
`studies/correction-decoders/qutrit/scripts/*.py`, the tests, the `.cache_correction` logs, and the notes the report
cites. Three cited notes were revised after the report (H2 at 21:07, H1 at 21:12; A1, A2 and
A4 critiques at 20:56 to 21:07). Where I cite them, I say so, and I rely on result files or
logs where possible.

## Summary

The numbers in the main tables (4.1, 4.7, 5.3, 5.4, 6) match the result files. The labelling
problems sit in the interpretive sentences: the executive summary, the verdicts of Section 5,
and the hardware Section 7.

- **Critical (3).**
  - The hardware acceptance claim ("< 0.1% of shots by m = 2 to 4") is SI1000 at a
    gate-matched p, labelled [LIT]. The cited H2 and H1 models give 18% at m = 4 for ions and
    0.5-7% at m = 2 for transmons.
  - "Correction removes ... non-Markovianity" is contradicted by the study's own mixture
    file: the corrected mixture curve has significant curvature.
  - The 4e7-shots-per-circuit budget is a single-depth total at m = 17, retracted in H2. The
    per-circuit figure is 1.75e6.
- **Major (20).**
  - BFD's "never accepts a first-order logical error" ignores the 4.95 p floor.
  - The Q2 shape claim holds only for full-information MLD on a relative measure.
  - [LIT: Wallman; Merkel] is cited for postselected, mirrored, non-2-design LRB, which those
    theorems do not cover.
  - Non-stationarity and gate dependence are not separated from memory in the definition and
    the tests.
  - Pareto claims: BFD is 1-17% above the frontier, not on it.
  - Ranges are stated beyond the simulated p.
  - Several [LIT] tags rest on sources outside the verified bibliography, or on papers that do
    not say the claim.
  - Hardware feasibility statements are unlabelled.
  - Protocol conclusions are specific to SI1000 and noiseless readout but stated as general.
- **Minor (7).** Wording, units, p-specific numbers quoted as general, and tag hygiene.

## Findings

Severity: critical = wrong number or conclusion; major = unsupported or mislabelled claim,
missing item, misleading statement; minor = clarity.

### R2-01 (critical): hardware acceptance claim is a conjecture contradicted by the cited models

- **Quote (exec. Q3):** "Today's transmon two-qutrit gates correspond to SI1000 `p = 0.027` at best (typically 0.05 to 0.18), and trapped-ion qutrit gates to about 0.014. At those values the baseline protocol keeps less than 0.1% of shots by `m = 2` to 4 **[LIT, VN]**." Also 7.2: "At these p the baseline keeps less than 0.1% of shots by m = 2 (transmons) or m = 4 (ions) **[VN, notes/H2]**."
- **Problem:** Only the identity "SI1000 two-qutrit p = process infidelity" is exact. Mapping a whole device to one SI1000 p is a modelling conjecture, not literature. The acceptance figures are SI1000 results at that p, and the platform-resolved models in the cited notes disagree with them.
- **Evidence:**
  - `notes/H2-other-platforms.md`, Summary item 2 (current): `ion_now` "behaves like SI1000 at p ≈ 3e-3 to 5e-3, not at the p ≈ 0.014 that a two-qutrit-gate-only mapping suggests".
  - H2 Section 8, item 3: the claim "rejects more than 99.9% of shots by m = 4 on ions does not hold for the ion model, which keeps 18% at m = 4 and 8% at m = 6".
  - `notes/H1-scripts/sc_dem_hw.log`: under HW-now, seq9 unif-1 acceptance is 0.0285 at m = 2 (par9 0.0674).
  - Current H1, Summary item 4: "keeps 0.5-7% of shots at m = 2".
  - H1, Summary item 3: SI1000 at p = 0.027 assigns 13.5% readout flips against measured 2-7%, and 5.4% idle per readout layer against 0.7%. It has no idle noise during gate layers.
  - "Typically 0.05 to 0.18" mixes a CZ (95.2%), CSUM process fidelities (0.82/0.85, 0.889) and a CPhase measured by QPT including SPAM (89.3%). The "others 82 to 89%" row of 7.2 has no citation.
- **Fix:**
  - "In SI1000 at p = 0.027 (0.014), the baseline keeps < 0.1% of shots by m = 2 (4) [VN]."
  - "Mapping a device to one SI1000 p is [CONJ]. Platform-resolved models give 0.5-7% at m = 2 for today's transmon components [VN, H1 sc_dem_hw] and 18% at m = 4 for native ion qutrits [VN, H2 ion_now]."
  - Keep the transmon conclusion (checked LRB-D is impractical today). Drop it for ions.

### R2-02 (critical): correction does not remove the non-Markovianity of correlated noise

- **Quote (5.6 item 3):** "**Correction removes a different, more dangerous non-Markovianity:** postselection's selection of the quiet part of temporally correlated noise (Section 5.5)." Exec.: "MLD correction decays normally from 0.974 to 0.617 **[VN]**."
- **Problem:** The 50/50 quasi-static mixture is a hidden-classical-memory process, which is non-Markovian by the report's own definition in 5.1. Correction removes the depth-dependent reweighting toward quiet shots (the selection bias). It does not remove the memory: the MLD mixture curve is significantly non-exponential, more curved than either component.
  - The mixture MLD is also an oracle decoder: each shot is decoded with its own p (`extra_analyses.py` combines each component's own decoded points). This assumption is unstated.
- **Evidence:**
  - `results/mixture_nonmarkov.csv` (`label_checks_output.txt`, B6): mld:all mix c2 = +1.68e-4 [+1.28e-4, +1.99e-4], against low +1.91e-5 and high +1.28e-4. raw mix c2 = +4.57e-4 [3.88e-4, 5.16e-4].
  - The report's own 5.5 says "Its decay is the usual mixture of exponentials".
  - `lit-L2-nonmarkov.md`, item 3: correlated classical noise moves the decay toward a power law.
  - `notes/review/V3/v3_mixture_oracle_output.txt`: the oracle and the Bayes-mixture decoder differ by < 1e-3 for MLD, so the numbers stand. Only the assumption is unstated.
- **Fix:** "Correction removes the selection bias (no depth-dependent reweighting toward quiet shots) [VN]. The corrected curve keeps the noise's memory: it is a mixture of exponentials with c2 = +1.7e-4 [1.3e-4, 2.0e-4], and its fitted rate lies between the components' rates [VN]." Replace "decays normally" with "decays monotonically, non-exponentially". State that the decoder uses the per-shot p.

### R2-03 (critical): the detection shot budget is the wrong number

- **Quote (7.6 item 4):** "Detection needs about 4e7 raw shots per circuit to resolve `unif 1` at m = 22 to 10%; correction removes the `1/acceptance` factor [notes/H2, notes/A2]."
- **Problem:** The number, the depth and the hidden p are all wrong.
  - A2's 4e7 is a single-depth Cramér-Rao count of submitted shots at the optimal depth m* = 17, not m = 22.
  - It assumes the offset B is known.
  - It is computed at SI1000 p = 1e-3, which the report does not state; 7.2 places hardware at p ≥ 0.014.
  - H2 retracted "4e7 per circuit" as a total over sequences and gives 1.75e6 per circuit.
  - The "8e6 shots per full dataset" in the same item cannot cover detection.
- **Evidence:**
  - `notes/A2-optimal-postselection.md`: item 8 ("N* ≈ 4×10^7 submitted shots for 10% relative precision", p = 1e-3) and table row "4.0e7 (17)".
  - A2 critique C19: "Assumes B known (×1.74 if fitted)".
  - H2 Section 8, item 2: "used the standard error of the 30-sequence mean, so it was a total over sequences ... per-circuit numbers: 1.75e6 at SI1000 p = 1e-3".
  - H2 Section 4.2 table: unif 1 needs 1.75e6, const 1 8.2e3 and MLD 39 shots per circuit.
- **Fix:** "At SI1000 p = 1e-3, ±10% (one standard error) on r with 30 sequences and two depths needs about 1.75e6 raw shots per circuit for LRB-D unif 1, 8e3 for const 1 and about 40 for full MLD [VN, H2 T3]." Reconcile or drop the 8e6 figure; the current H1 gives 2.7e6.

### R2-04 (major): BFD accepts the first-order floor

- **Quote (exec.):** "It never accepts a first-order logical error **[VN]**." The same wording appears in 4.4, rank 2.
- **Problem:** As worded, this is false. BFD, like LRB-D, accepts the undetectable 4.95 p final-round floor at first order.
  - The cited check excludes undetected faults by construction, and it covers 8 circuits at one p.
  - "Which at first order is always a measurement or reset fault" is loose: BFD-forgiven faults also include ancilla idle and ancilla-only components of data-ancilla gate faults.
- **Evidence:**
  - `studies/correction-decoders/qutrit/scripts/verify_first_order.py`: `if not nontrivial[v].any(): ... continue` runs before the BFD branch, so `bfd_accepted_logical` counts detected faults only.
  - `results/first_order_check.json`: undetected_logical = 4.95e-4 at p = 1e-4, and exact_bfd_logical = 4.92e-4 at m = 0. Only Cliffords 0 and 7, depths 0, 4, 14 and 22 were checked.
  - Table 4.1: BFD accepted P(correct) at m = 0 is 0.99499, an error of 5p.
  - `notes/review/V4/check_first_order_bfd.out`: forgiven faults include ancilla idle (72 p) and data-ancilla gate faults (4.2 p) as well as measurement flips (160 p). The logical shift of all of them is 0.
- **Fix:** "Apart from the 4.95 p undetectable floor, which every rule accepts, BFD accepts no detected single fault that shifts the logical [VN, 8 circuits at p = 1e-4]." Use "fault confined to an ancilla".

### R2-05 (major): "Correction mitigates non-exponentiality in shape" holds only for full-information MLD on a relative measure

- **Quote (exec.):** "**Correction mitigates non-exponentiality in shape only.** It turns the quadratic pair mechanism into first-order logical errors, so the decay is dominated by memoryless failures." 5.6 item 2 says the same.
- **Problem:** MLD with sparse information accelerates: MLD on the terminal round only is more curved than const 1 detection. In absolute curvature, full MLD is about as curved as const 1. "Dominated by memoryless failures" has no attribution computation behind it and is [CONJ].
- **Evidence:**
  - `label_checks_output.txt` B5 (p = 1e-3):

    | rule | c2 [95% CI] | windowed 1 - f, first to last window |
    |---|---|---|
    | mld:terminal | −3.04e-4 [−3.37e-4, −2.74e-4] | 0.0229 → 0.0364 (+58%) |
    | mld:unif22 | −7.6e-5 [−1.1e-4, −4.6e-5] | |
    | mld:all | +5.36e-5 | |
    | lrbd:const1 | −6.75e-5 | |
    | lrbd:unif1 | −1.06e-7 | |

  - A4 critique C20: the η reduction is significant only for const 1 and const 0, and "by absolute shape, the corrected curves are much less exponential for every policy".
- **Fix:** "Full-information MLD flattens the windowed rate (−15% against 18× for const 1) [VN]. MLD on the terminal round alone accelerates (c2 = −3.0e-4) [VN]. The memoryless-failure mechanism is [CONJ]." Name the metric (relative windowed rate) and also give the absolute c2.

### R2-06 (major): [LIT] for a single exponential under postselection, mirrored inverses and a non-2-design catalog

- **Quote (5.1):** "Under that assumption, with Clifford twirling (the qutrit Clifford group is a unitary 2-design, prime d [LIT: Graydon et al. 2021; Jafarzadeh et al. 2020]), the decay is a single exponential up to small gate-dependent corrections [LIT: Wallman 2018; Merkel et al. 2021]."
- **Problem:** The assumption includes trace-decreasing maps and a postselected ratio, which the cited theorems do not cover. The protocol also uses m mirrored noisy inverses and a catalog that is not a 2-design (Section 10). The report's own Section 1 quotes "no complete theory" (Harper and Flammia).
  - 5.2's "postselection by itself does not make the decay non-exponential" needs stochastic Pauli noise (for state-independent acceptance) and an exact twirl. Neither condition is stated.
- **Evidence:**
  - `lit-L1-lrb.md` lines 539-543: Wallman's theorem "does not cover a postselected ratio ... For LRB-D the single exponential is therefore an empirical model".
  - A4 critique C6: Wallman's Theorem 4 needs a single inversion gate and a uniform 2-design.
  - A4 critique C26: the catalog's frame potential is 2.031, and twirled Pauli eigenvalues spread by 1.5%.
  - A4 critique C2: state independence needs Pauli noise.
  - A2 critique C15: "'Exactly' needs i.i.d. per-step increments ...; LRB-D violates this."
- **Fix:**
  - "[LIT] for CPTP, gate-dependent Markovian noise with a single inversion gate."
  - "For trace-decreasing steps, the postselected ratio is a single exponential under an exact 2-design twirl and Pauli noise [VA, L1 9.2 / A4]."
  - "For mirrored inverses and this catalog: [CONJ], supported by the A4-critique toy."
  - Add the same conditions to 5.2.

### R2-07 (major): the "precise definition" and the tests do not match

- **Quote (5.1 item 3):** "the check schedule, when the policy makes it depend on m."
- **Problem:** Non-Markovianity is defined as history dependence through hidden memory (the process-tensor sense), yet an m-dependent schedule is counted as a memory carrier. That is non-stationarity: each depth runs a different experiment. The tests (windowed rates, c2) detect non-exponentiality, which also comes from m-dependent schedules, from gate dependence of the finite draw and from the non-uniform catalog. No test probes the stated definition.
- **Evidence:**
  - `label_checks_output.txt` B7: `const 22` has windowed f = 1.010 for m = 20 to 22. It runs terminal-only below m = 22 (00-brief, policy definition).
  - `lit-L2-nonmarkov.md` item 3: an uptick is a witness of non-Markovianity only "provided the protocol is the same at every m".
- **Fix:** Separate three causes:
  1. memory (syndrome register, decoder state);
  2. m-dependent protocols (non-stationarity);
  3. gate dependence and non-uniform twirling.

  Call windowed rates and c2 non-exponentiality tests. Add a memory-specific contrast, for example checked against unchecked inverse on the same sequences. Label which cause each result supports.

### R2-08 (major): inverse-block attribution is labelled [VN] but contested

- **Quote (5.3):** "`unif 1`: only a small `m^2` term from the m-step unchecked inverse block." 5.4 says: "Each depth has its own unchecked inverse block, whose undetected-pair probability depends on its gates."
- **Problem:** The critique of the cited note finds that the forward part, which is checked every step, carries a larger m² term than the inverse block, and that per-sequence coefficients change sign. This points to gate dependence of the particular draw as the main cause. The dense checked-inverse result (c2 → 0) fits the report's attribution but does not exclude this alternative.
- **Evidence:**
  - A4 critique C12, on sequences 3 to 5: full fit 170 + 21.2 m + 0.280 m², no-inverse 168 + 16.2 m + 0.175 m², no-forward 112 + 3.5 m + 0.082 m². Per-sequence a2 values are 0.047, −0.076, 0.187, 0.297, 0.280 and 0.261.
  - `fits_dense.csv.gz`: base c2 = −1.1e-7 [−2.1e-7, −3.0e-8]; chk c2 = +6e-10 [−8.7e-8, +7.8e-8].
- **Fix:** Label the attribution [CONJ] ("consistent with [VN]"), name gate dependence as the competing explanation, and quantify both with more sequences.

### R2-09 (major): an unsimulated fix is presented as "decisive"

- **Quote (5.6 item 4):** "**The decisive fix is protocol-level:** leave no long unchecked interval. Check inside the inverse block or compile a single recovery Clifford, and never use sparse-check policies to estimate a per-step rate."
- **Problem:** The sentence is unlabelled. The compiled recovery Clifford was never simulated (it is [CONJ] in 5.4 and "Not simulated" in Section 11), yet it is called decisive and recommended in 7.6, Stage C.
  - The checked-inverse fix makes every correction rule's rate worse.
  - A compiled recovery is not mirrored, so the decay constant changes under non-Pauli noise.
- **Evidence:**
  - `variant_summary.md`, p = 1e-3: with checks inside the inverse half, soft:0.9 r = 1.52e-3 and mld:all r = 2.00e-2, against 9.42e-4 and 1.19e-2 for the baseline.
  - A4 critique C25: the VA label is too strong; in the toy, the non-mirrored constant is 0.634 against 0.797.
- **Fix:** "Candidate fixes: checks inside the inverse block [VN for LRB-D and BFD; costs rate for correction] or a compiled recovery Clifford [CONJ]." Drop "decisive".

### R2-10 (major): Pareto-frontier statements

- **Quote (4.2):** "BFD sits on it, `const 1` is dominated, and full MLD and the causal decoders are its acceptance-1 endpoint."
- **Problem:**
  - BFD lies strictly below the frontier.
  - The causal decoders lie below MLD at acceptance 1.
  - The [VN] for unif-1 LRB-D "within the precision of the sampled frontier at m = 6 and 22" (line 368) is not a test, because the sampled frontier never reaches LRB-D's acceptance.
  - The [VA] on "strictly dominated" in the executive summary has no proof behind it.
- **Evidence:**
  - `label_checks_output.txt` B4: BFD's accepted infidelity exceeds the frontier at equal acceptance by +0.7% and +2.1% (p = 5e-4, m = 6 and 22), +1.5% and +4.1% (1e-3), +2.9% and +7.4% (2e-3), +7.2% and +17.1% (5e-3). At p = 1e-3, m = 22 the gap is 2.4e-4 against a sequence standard error of about 1e-5.
  - Causal decoders with lookahead 0, 1 and 2 sit 21.4%, 4.8% and 1.6% above MLD's infidelity at acceptance 1 (p = 1e-3, m = 6).
  - unif-1 acceptance is 0.0807, below the lowest sampled frontier point (0.0933) at p = 1e-3, m = 22, and likewise at every (p, m).
  - A2 critique C23: on-frontier at m ≥ 4 is unverifiable [CONJ]; 0 of about 130,000 candidate histories outrank D = 0.
- **Fix:**
  - "BFD lies 1-17% above the frontier [VN]."
  - "The causal decoders lie 2-24% below MLD at acceptance 1 [VN]."
  - "unif 1 is on the frontier at m = 0 and 2 [VN, exact] and is not dominated by any sampled point at m = 6 and 22 [VN]; on-frontier there is [CONJ]."
  - Drop [VA] from "strictly dominated".

### R2-11 (major): the soft-MLD row omits acceptance, against the report's own rule

- **Quote (4.3):** "| Soft MLD, t = 0.9 | between first and second order; 4 to 7 times below physical RB from p = 2e-4 to 1e-2 | |"
- **Problem:** 4.2 states that "The decay rate of a postselected rule must therefore always be quoted with its acceptance per step." Here the rejection cell is empty, and "7 times below" at p = 1e-2 comes with 59% rejection per step. The actual range is 4.0 to 7.7.
- **Evidence:** `label_checks_output.txt` B3. Acceptance per step is 0.9951, 0.9868, 0.9687, 0.9221, 0.7241 and 0.4121; the corresponding ratios to physical RB are 1/5.1, 1/4.1, 1/4.0, 1/4.3, 1/6.4 and 1/7.7.
- **Fix:** Fill in the rejection, from 0.5% at p = 2e-4 to 59% at p = 1e-2. Write "4 to 8 times below physical RB, with acceptance per step falling from 0.995 to 0.41".

### R2-12 (major): the full-MLD range goes beyond the simulated p

- **Quote (exec.):** "and it is worse than physical RB at every p simulated (1e-4 to 2e-2) **[VN]**". Table 4.3 says: "11 to 15 p, 3 to 4 times above physical RB at every p".
- **Problem:** Sampled rules were run only for p ≤ 1e-2 (Section 2.3), and the table's ranges miss the end points.
- **Evidence:** `label_checks_output.txt` B2. mld:all fits exist for p = 1e-4 to 1e-2 only. r/p = 11.0, 11.1, 11.4, 11.9, 12.8, 15.2 and 17.4; r/r_RB = 2.92, 2.99, 3.03, 3.15, 3.41, 4.09 and 4.74.
- **Fix:** "(1e-4 to 1e-2)". In the table: "11 to 17 p, 2.9 to 4.7 times above physical RB".

### R2-13 (major): offline-frame and no-reset claims drop their conditions

- **Quote (exec. Q3):** "For Clifford-only LRB, correction is a Pauli-frame relabelling of the terminal outcome, computed offline. No feedback, no decoder latency budget, and no ancilla reset are needed **[VA, VN, LIT]**."
- **Problem:**
  - Exactness needs stochastic Pauli noise (the 7.1 statement). On hardware, coherent errors and damping break it at first order, so randomized compiling is a precondition.
  - "No ancilla reset" needs QND readout, and the no-reset variant was never simulated with noise.
  - Geher et al., which the report cites, find that the reset choice changes logical-operation performance by up to a factor of two.
  - Section 9's "Real-time decoding is needed ... **Refuted [VA, VN]**" is equally unconditional.
- **Evidence:**
  - A3 C5: P(correct) 0.98508 against 0.94134 (coherent error) and 0.950 against 0.9025 (damping).
  - `src/lrb/correction/variants.py` has no no-reset flag.
  - H2 C6 checks only the noiseless mapping.
  - `bibliography_raw.csv`, Geher2025 note: "up to a factor of two for logical operations".
- **Fix:** "... under stochastic Pauli or Weyl-twirled noise [VA, VN]. Dropping reset needs QND ancilla readout and was not simulated with noise [CONJ; LIT Geher 2025: up to 2× effects]." Apply the same condition in Section 9.

### R2-14 (major): [LIT] tags whose sources are not in the verified bibliography, or do not say the claim

- **Quote (7.2):** "| Trapped-ion qutrit two-qudit gate | 98.7%, readout about 99.7%, all-to-all [Ringbauer et al. 2022; Hrmo et al. 2023, via notes/H2] | p about 0.014 |"
- **Problem:** The report defines [LIT] as fetched and independently re-fetched (Section 14). These claims fail that definition:
  - Ringbauer2022 and Hrmo2023 are absent from `bibliography_raw.csv`. Section 14 refers to "the two marked entries", but no entry is marked.
  - 7.1: "current FPGA loops take 0.2 to 0.6 µs [... LIT]". H2's sources (QICK, QubiC 2.0, vendor pages, Cao26) are not in the bibliography.
  - 7.1: "a waiting transmon qutrit loses 1.4 to 9% per µs" is an H1 model result [VN, assumed T2], not literature. The current H1 gives 1.4-2.9% (or 1.4-4%) per µs.
  - 7.2: "readout errors biased toward decay ... [LIT]" and "leakage to |3> [LIT]" carry no citation.
  - 7.2: "0.5-1.5 µs" has no citation. L6 attributed it to Blok 2021, which the checker found "gives no readout duration".
  - 7.5: "Emerald 54" is cited to Abdurakhimov 2024 (the 20-qubit paper). The actual source is AWSEmerald2025, which is in the CSV but not in Section 14.
- **Evidence:** `label_checks_output.txt` B1 (Hrmo2023 and Ringbauer2022 missing); H2 Section 9 references; the Blok2021 note in `bibliography_raw.csv`; `lit-L6-scqutrit.md` lines 88-90; current H1 Summary items 3 and 7.
- **Fix:**
  - Re-check Ringbauer and Hrmo and add them to the CSV, or tag them "[LIT, single fetch, notes/H2]". Mark the lead-fetched entries.
  - Cite the latency sources, or tag them that way too.
  - Tag the idle loss as [VN under assumed T2, notes/H1] and use the current number.
  - Cite McEwen2021/Miao2023 for leakage and Chen2023 for readout decay.
  - Cite AWSEmerald2025.

### R2-15 (major): Mavadia and Ball do not establish drift or two-level fluctuators in devices

- **Quote (5.5):** "Real devices drift and have two-level fluctuators [LIT: Mavadia et al. 2018; Ball et al. 2016], so the noise strength varies between shots."
- **Problem:** Mavadia et al. inject engineered σ_z detuning noise into an ion; Ball et al. is a theory paper. The 50/50, 4× per-shot mixture is an assumed model. So the exec. headline "a postselection bias that matters on hardware" is [VA] for the direction and [CONJ] for the size.
- **Evidence:** `lit-L2-nonmarkov.md` 3.3, line 417: "injected σ_z detuning noise".
- **Fix:** Cite a device-drift or TLS paper (verified), or tag the sentence [CONJ]. Reword the exec. bullet as "matters whenever the per-shot error rate varies [VA direction; VN for a 4× toy; size on hardware CONJ]".

### R2-16 (major): "The only hardware LRB with a decoder" is contradicted by the study's literature notes

- **Quote (Section 1):** "The only hardware LRB with a decoder found is on a distance-3 colour code, with one QEC cycle between transversal gates, a single recovery Clifford and no postselection [Lacroix et al. 2025]."
- **Problem:** The literature notes list two more hardware LRB experiments with decoding, plus a postselected LRB with a check after every Clifford, which is directly relevant to Q2.
- **Evidence:**
  - `lit-L1-lrb.md` lines 49-53 lists Steane-code LRB with terminal decoding [Mayer2024], C4-Helix LRB with adaptive extraction [Berthusen2026], and a bosonic erasure qubit with a check after every Clifford and separately fitted acceptance [Mai2026].
  - Table rows 978-981 of the same note.
  - `bibliography_raw.csv`: Mayer2024 and Berthusen2026 are confirmed.
- **Fix:** List all four, and add them to Section 14.

### R2-17 (major): Kwiatkowski et al. cited without their conditions; the [CONJ] names the wrong obstacle

- **Quote (5.6 item 2):** "Decoded repeated QEC cycles are expected to be approximately Markovian with exponentially suppressed deviations [LIT: Kwiatkowski et al. 2025]. That suppression needs more distance than this code has **[CONJ]**."
- **Problem:** The suppression is exponential in the number of cycles, and it holds under four conditions:
  1. Pauli-stochastic noise;
  2. syndrome marginal independence (SMIP), which follows from a decoding symmetry such as odd distance with minimum-weight decoding, or can be enforced by syndrome randomization;
  3. single-cycle logical entanglement fidelity ≥ 1 − 1/64;
  4. qubits.

  The obstacles here are the decoding ties (SMIP) and the qutrits, not distance as such. Condition 3 fails for full MLD at p ≥ 1e-3.
- **Evidence:**
  - `lit-L2-nonmarkov.md` lines 1029-1051.
  - Kwiatkowski note in `bibliography_raw.csv`.
  - `fits_baseline`: with ε = (4/3) r, full MLD gives ε ≈ 1.6e-2 at p = 1e-3 and 3.4e-2 at p = 2e-3, both above 1/64.
- **Fix:** State the four conditions. Replace the [CONJ] with: "Ties break SMIP unless the syndrome frame is randomized, and ε > 1/64 for p ≥ 1e-3 [VA]; the extension to qutrits is [CONJ]."

### R2-18 (major): Stage A hardware feasibility is unlabelled and assumes ideal encoding and readout

- **Quote (exec. Q3):** "`const 0` LRB on the five data qutrits with terminal decoding. It needs no mid-circuit measurement and runs on IQM-class square lattices today."
- **Problem:**
  - Section 7.5 itself finds no qutrit gates, multi-state discrimination or ternary feedback in IQM's circuit API. The only IQM-fabricated qutrit run used non-IQM electronics.
  - All simulations, including H1's hardware models, assume an ideal |+>_L encoder. That encoder is a 4-CNOT unitary on data couplings whose noise is never modelled. They also assume ideal readout.
  - Under H1's HW-now model, const-0 LRB-D is worse than a physical qutrit. The report quotes only SI1000's "0.45 times physical RB".
  - H1's Stage C condition (two-qutrit error ≤ about 1%) is missing from 7.6.
- **Evidence:**
  - `src/lrb/code_definitions.py`, `logical_plus_initial_state`: CNOT_INV 2→1, CNOT 1→0, CNOT 3→0, CNOT 4→2.
  - H1 lines 572-573: "Terminal readout and encoding stay ideal".
  - H1 line 621: "const-0 LRB-D at HW-now. r = 0.017 per step with 13.5% rejection per step: worse than physical qutrits".
  - H1 Summary item 4 (accepted fidelity 0.996 / 0.94 / 0.81 at m = 2 / 6 / 10) and item 9(C).
- **Fix:** Tag it [CONJ]. Add: "needs a native qutrit CZ and 4-level readout (pulse level on IQM, or non-IQM electronics as in Goss 2026); encoding and readout are idealized". Quote the HW-now const-0 numbers, and add the ≤ 1% precondition to Stage C.

### R2-19 (major): protocol conclusions specific to SI1000 and noiseless readout are stated as general

- **Quote (Section 6):** "- **Full terminal readout** is free, adds no noise, and removes the 4.95 p floor."
- **Problem:**
  - "Free, adds no noise" holds only because the terminal readout is noiseless by construction.
  - The parallel-readout gain ("cuts every strategy's error rate by 1.6 to 3 times") follows SI1000's rule of 2p idle per measurement layer, whatever the layer's duration.
  - The const-0 "0.45 × physical RB" caveats (exec. and 4.6) name `p_RB = 5p` and noiseless readout, but omit SI1000's missing gate-layer idle and the ideal encoder.
- **Evidence:**
  - `variants.py` docstring: full terminal readout "(noiselessly, as the terminal readout already is)".
  - `notes/review/V5/out/check6.txt`: with SI1000 readout noise, the m = 0 MLD first-order failure under full terminal readout rises from 11.9p to 23.55p, and the base floor rises from 4.95p to 16.6p.
  - H1 Summary item 3.
- **Fix:** Prefix each statement with "In SI1000 with noiseless terminal readout and ideal encoding [VN]". Note that with noisy readout, full terminal readout trades the floor for extra rejections and miscorrections, and that the size of the parallel-readout gain depends on the per-layer idle rule (H1's hardware model confirms the direction).

### R2-20 (major): flag rule contradicted by the study's own A5 and Section 8

- **Quote (3.2):** "Flags become necessary for distance-3 codes with weight-4 or higher checks, such as the `[[7,1,3]]_3` colour code (Section 8)."
- **Problem:** This is an unlabelled general rule, and it is false: weight-4 surface-code checks need no flags if the CNOT order is chosen well.
- **Evidence:** `notes/A5-distance3.md` C9: [[13,1,3]]_3 has 24 of 24 safe orders per weight-4 check and [[9,1,3]]_3 has 16 of 24; flags are needed only for [[7,1,3]]_3 and [[5,1,3]]_3. Report Section 8: "One-ancilla-per-check extraction tolerates any single fault."
- **Fix:** "The [[7,1,3]]_3 colour code and [[5,1,3]]_3 need flags (no safe CNOT order exists); the [[13,1,3]]_3 and [[9,1,3]]_3 surface codes do not [VN, A5]."

### R2-21 (major): the "three regimes" claim holds only at leading order

- **Quote (3.1):** "The posterior-threshold family has only these three regimes at code capacity."
- **Problem:** At finite q:
  - τ = 0.5 gives the hybrid, not MLD;
  - a fourth operating point exists (accept T, U1 and A1);
  - the 0.99 boundary fails at q = 0.05.
- **Evidence:** A1 critique C8 (A1 posterior 0.4998 / 0.4984 / 0.4915; fourth regime for 0.331 < τ ≤ 0.498 at q = 0.01).
- **Fix:** "At leading order in q, three regimes, with boundaries 1 − O(q) and 1/2 − O(q); at finite q there is also an A1-accepting regime [VN, A1 critique]."

### R2-22 (major): the `const 22` rate is an artefact

- **Quote (4.7):** "| `const 22` | 4.5e-5 | 0.916 | 9.8e-5 | 0.932 |"
- **Problem:** `const 22` is terminal-only at every depth below 22 and nearly every-round at m = 22. Its fitted r, which comes out lower than const 4 and unif 4, reflects that jump, not a per-step rate. It appears without comment, against 5.6's own advice not to use sparse-check policies for rates.
- **Evidence:** `label_checks_output.txt` B7: windowed f is ≤ 0.99985 for m ≤ 20 and 1.0101 for m = 20 to 22.
- **Fix:** Remove the row, or mark it "r not meaningful: the protocol changes at m = 22". The same applies to any const c with c above some simulated depths.

### R2-23 (major): soft-MLD optimality labels

- **Quote (exec.):** "It traces the Bayes-optimal acceptance-fidelity frontier **[VA, LIT]**." 4.4 says: "Exactly the optimal frontier."
- **Problem:**
  - The theorem is per depth and assumes the decoder's posterior is the true one.
  - The [LIT] support is thin: the fetched Chow abstract does not state the max-posterior form, Forney's rule assumes equal priors (here P(L = 0) ≈ 0.99), and Forney was read only second-hand.
  - In a table of (r, acceptance per step), "Exactly the optimal frontier" implies an optimal rate trade-off, which is not shown.
  - With a miscalibrated DEM the rule is off the frontier.
- **Evidence:** A2 critique C3; Forney68 note in `bibliography_raw.csv` ("Full text not read"); `mismatch_vs_frontier_output.txt` (+3% to +9% excess accepted infidelity at 4× rate errors).
- **Fix:** "[VA, A2] at each depth, given the true DEM; a Chow-type reject rule [LIT: Chow 1970, abstract]." In 4.4: "on the per-depth frontier when the DEM is exact".

### R2-24 (minor): the condition for the Rice bias is misstated

- **Quote (10.2):** "**The legacy fidelity metric** is biased upward by about `0.89/sqrt(N_acc)` when few shots are accepted. This explains every large deviation between the exact LRB-D and the 1M-shot CSVs **[VN]**." Section 2.4 has the same wording.
- **Problem:** 0.89/√N is the bias when the true polarization is ≈ 0 (the Rayleigh floor). Near λ = 1 the bias is ~(1 − λ²)/(4λN), for example ~2e-6 at λ = 0.99 and N = 1000.
- **Evidence:** `label_checks_output.txt` B8: 473 of the 474 rows with |z| > 3 have true fidelity < 0.05 (median N_acc 12,343). The remaining row is an artefact of zero variance (fid_z = −7.7e7, p = 1e-4, const 0, m = 2).
- **Fix:** "... when the true polarization is near zero (fully decayed curves)". Drop the zero-variance row from the z statistics.

### R2-25 (minor): validation wording

- **Quote (exec.):** "It is a forward filter over 243 states per check round, validated to machine precision."
- **Problem:**
  - The tests use rtol 1e-9 (1e-8 for the variants) against brute force, on depth-1 circuits only.
  - "Within 4 sigma" in 2.4 is actually 4σ + 1e-4.
  - Section 12 says the tests take "about 3.5 min"; the log says 35 s.
  - "Realized counts ... agree" (2.3) has no committed comparison. A reviewer probe finds z = +3.5 and +3.2 on two production circuits at p = 0.01, fine on fresh seeds.
- **Evidence:** `tests/test_correction_chain.py` lines 95, 125 and 129; `tests/test_correction_variants.py` lines 66 and 107; `.cache_correction/pytest_final.log`; `notes/review/V2/v2_realized_probe.log`.
- **Fix:** Correct the wording, or add a results table that compares realized and expected counts.

### R2-26 (minor): hybrid numbers and the "27 p²" misquote

- **Quote (3.3):** "Accepted-shot errors stay second order for LRB-D (`r` about 20 p^2), BFD (about 53 p^2) and the hybrid (about 1.3e3 p^2)"
- **Problem:**
  - The hybrid is not in the production fits, so "the fits agree" does not cover it.
  - Its p² slope comes from 2 sequences and spans 853-1301 across 4.
  - The sentence mixes r (for LRB-D and BFD) with 1 − f (for the hybrid).
  - Section 8's "about 27 p^2 per step here" misquotes the baseline: 1 − f is 30.5 p², and r is 20 p².
- **Evidence:** A1 critique C18-C19; the strategy list of `fits_baseline.csv.gz`.
- **Fix:** "hybrid r ≈ (0.9-1.3)e3 p² from a 2-4-sequence enumeration [VN, A1]". Use one unit throughout.

### R2-27 (minor): mismatch claims

- **Quote (4.8):** "The mismatched points stay close to the true frontier."
- **Problem:**
  - No script computes the true frontier for these shots. My check supports the claim and quantifies it.
  - "Structural mismatch ... can admit first-order logical errors [VN, notes/A1]" is a code-capacity result for the hybrid rule, not for soft MLD on circuits.
  - 4.8 advises setting thresholds by target acceptance, but 7.6 then uses fixed "t in 0.9 to 0.97".
- **Evidence:** `mismatch_vs_frontier_output.txt`: +3.3% to +8.7% excess accepted infidelity at equal acceptance for 4× rate errors. At p_dec = 4p with t = 0.97 (m = 10, p = 2e-3) nothing is accepted. A1 lines 351-357.
- **Fix:** "stay within 9% of the true frontier [VN, R2-labels script]"; give the scope of the structural-mismatch result; use acceptance quantiles in 7.6.

### R2-28 (minor): p-specific or first-order statements quoted as general

- **Quote (exec.):** "Hard-decision causal decoders with lookahead 2, 1 and 0 are 4%, 9% and 31% worse than MLD **[VA, VN]**."
- **Problem:** Each of these holds only at one p, at first order, or for a subset, but is quoted without the scope:
  - The causal penalties are for p = 1e-3. Over the simulated p they range 22-36% (lookahead 0), 2-16% (1) and 1-9% (2).
  - 4.4 credits unif 1 with "exponential decay", but 5.4 reports a significant curvature. The exec. "resolvable" curvature does not hold for BFD at p = 1e-3 (dense c2 [−4.0e-7, +2.9e-8]).
  - "Reject a single exponential for every `const c`" fails at p = 1e-4 (B9: const 4, 5, 6, 7, 11 have p_gof 0.07-0.58).
  - The scalings "22.7 k p²" and "m²/c" hold for k, c ≤ 3 only (A4 critique C10).
  - "Time-like faults are always correctable" holds at first order only (A1 lines 408-418).
  - Section 8's "3 times worse" is 2.9-4.7 times over p and depends on `p_RB = 5p`.
- **Fix:** Add the p, the order and the scope to each statement.

### R2-29 (minor): Stage B learning claim ignores identifiability

- **Quote (7.6):** "They learn the DEM: per-location rates from detector correlations, readout confusion matrices, and leakage flags."
- **Problem:** Detector correlations identify mechanisms by their detector signature, not per-location rates. The relative rates within the 8 tied pairs, which decide MLD's choice, cannot be separated from detector statistics alone.
- **Fix:** "Learn rates per detector-signature class; the logical split within tied classes needs memory runs with logical readout in both bases [CONJ]."

### R2-30 (minor): unlabelled claims and tag hygiene

- **Quote (exec.):** "The decision rule matters, not the decoder."
- **Problem:** The label section says "Every claim carries one tag", but these claims have no tag, or a tag outside the four:
  - "Approximate decoders ... are therefore unnecessary" and the quote above;
  - `[code: lrb_simulation.py]`;
  - "1e5 shots of a depth-22 circuit decode in 6 to 11 s on one core", with no timing log;
  - the Section 6 header and table;
  - "Ion chains and movable atom arrays need no routing" (atom arrays need shuttling);
  - "This is the strongest practical argument for correction";
  - "about 118 p ... against about 4.7 p per logical Clifford": only 62.0 p of the 118.4 p touches data (A3 line 826), and Section 1 gives 3.3-4.9 p per Clifford.
- **Also:** Keppens2025 is listed in Section 14 but never cited; "Vora et al. 2026" has a corrected year of 2024.
- **Fix:** Tag code readings [VA] and opinions and recommendations [CONJ]. Add [VN] to Section 6. Fix the bibliography.
