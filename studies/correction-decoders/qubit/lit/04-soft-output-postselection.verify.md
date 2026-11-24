# Verification of lit review 04: soft-output decoding, complementary gap, decoder-informed postselection

Target file: `04-soft-output-postselection.md` (same folder), as it stood at 10:49 on 2026-10-28.
Verifier: independent citation and claim check. Accessed 2026-10-28.

## Bottom line

- **Citations.** I checked all 29 bibliography entries. All 29 have correct title, authors, year and
  identifier. I found no hallucinated paper and no wrong arXiv ID or DOI. Two entries are missing
  journal versions that now exist: Gidney2023yoked and Chen2025 (details in section 1).
- **Key findings.** The structured summary lists 21 key findings, not 22. Results:
  - 17 supported;
  - 2 partially supported (#9 cultivation hardware, #21 the d = 2 sustainability conjecture);
  - 2 not checkable, because they are labelled conjectures (#19, #20);
  - none unsupported.
- **Section 3.1 numbers.** I re-ran the code-capacity enumeration independently. Every number in
  Tables 1, 2 and 3 reproduces to the printed digits.
- **Corrections needed.** None of the problems I found change a conclusion. The most useful fixes
  are:
  - the 23x overhead figure belongs to the radial-gap variant (Bombín);
  - Rosenfeld et al. do not say that decoder-based selection is future work;
  - the "about 2% per round" tesseract discard comes from simulation, not from the experiment;
  - at d = 2, even "reject only g = 1/2" already pays about 4p/3 rejection per round.

  Section 3 lists all of them.

## Method

- **Metadata.**
  - All 27 arXiv IDs: the arXiv API record (export.arxiv.org/api/query, the same metadata as the
    abstract page) for title, full author list, first-version date, journal ref and DOI.
  - DOIs: the Crossref works API for 15 of them. These are both IEEE papers, every journal DOI in the
    bibliography, and the two newly found journal versions.
- **Claims.**
  - Full text (arXiv HTML or ar5iv) for 18 papers, searched for the exact numbers and statements.
    These are Bombin2022, Gidney2023yoked, Gidney2024cult, Meister2024, Smith2024, English2024 (v3),
    Seip2026, Chen2025 (v3), Reichardt2024 (v2), Dinca2025 (v4), Zhou2025, Bluvstein2023,
    Hanisch2024 (v2), Xin2025, Rosenfeld2025, Levine2023 (v3), CombesFerrie2015 and Combes2017.
  - Abstracts only for Lee2025, Prabhu2021, Pattison2021, Ali2024, AGP2007, Merhav2006, Kumar2026,
    Shtanko2026 and Tahaab2026. The note cites these at abstract level too.
  - Chow1970 and Forney1968: Crossref metadata, with content through CombesFerrie2015 and
    Merhav2006, as the note does.
- **Enumeration.** I re-ran the section 3.1 enumeration with my own code. Both scripts run in under a
  second: `v04_enum512.py` and `v04_gap512.py` in the session scratchpad `v04scripts/`.

## 1. Citation metadata

"OK" means title, authors, year and identifier match the arXiv record or the Crossref record.

| key | identifier checked | result | notes |
|---|---|---|---|
| Bombin2022 | arXiv:2212.00813; DOI 10.1103/PRXQuantum.5.010302 | OK | Crossref: PRX Quantum 5, 010302 (2024-01-04). The journal title is "Fault-Tolerant Postselection for Low-Overhead Magic State Preparation"; the arXiv title as cited is fine. |
| Gidney2023yoked | arXiv:2312.04522 | OK | Add the journal version: Nature Communications 16, 4498 (2025), DOI 10.1038/s41467-025-59714-1 (Crossref, same four authors). |
| Gidney2024cult | arXiv:2409.17595 | OK | The full title as cited is correct. |
| Meister2024 | arXiv:2405.07433 | OK | Nadine Meister, Christopher A. Pattison, John Preskill; v2 2024-06-01. |
| Smith2024 | arXiv:2405.03766; DOI 10.1038/s42005-024-01883-4 | OK | Crossref: Commun. Phys. 7, 386 (2024-11-28). |
| English2024 | arXiv:2410.07598; DOI 10.1103/nh49-52y2 | OK | Crossref: PRL 135, 120603 (2025-09-18). |
| Seip2026 | arXiv:2606.22194 | OK | Dominik Seip, Luis Colmenarez, Markus Schmitt, Markus Müller; v1 2026-06-20. |
| Chen2025 | arXiv:2510.05222 | OK | Add the journal version: Phys. Rev. A 114, 032455 (2026), DOI 10.1103/mfyk-5j3x (Crossref, same six authors). |
| Lee2025 | arXiv:2510.05795; DOI 10.1038/s41534-026-01242-x | OK | Crossref: npj Quantum Inf. 12, 96 (2026-04-24). |
| Prabhu2021 | arXiv:2112.03785; DOI 10.1103/PhysRevA.110.012419 | OK | Crossref: PRA 110, 012419 (2024-07-08). |
| Reichardt2024 | arXiv:2409.04628 | OK | 18 authors, first Ben W. Reichardt, last Matt Zanner. |
| Pattison2021 | arXiv:2107.13589 | OK | Four authors as cited. |
| Dinca2025 | arXiv:2512.15689 | OK | Maria Dincă, Tim Chan, Simon C. Benjamin; v4 2026-09-30. |
| Zhou2025 | arXiv:2512.09863 | OK | Zeyuan Zhou, Shaun Pexton, Aleksander Kubica, Yongshan Ding. |
| Bluvstein2023 | arXiv:2312.03982; DOI 10.1038/s41586-023-06927-3 | OK | Crossref: Nature 626, 58-65 (print 2024-02-01, online 2023-12-06); 21 authors, last Mikhail D. Lukin. |
| Hanisch2024 | arXiv:2411.16228; DOI 10.1103/y9fh-4x6n | OK | Crossref: APS Open Science 1, 000019 (2026-05-08). |
| Xin2025 | arXiv:2511.17460; DOI 10.1103/y66v-988s | OK | Crossref: PRX Quantum 7, 033062 (2026-09-21). arXiv spells "Mark W. Beekman" and Crossref "Marc W. Beekman"; "M. W. Beekman" fits both. |
| Rosenfeld2025 | arXiv:2512.13908 | OK | 295 authors; the first four are as cited. |
| CombesFerrie2015 | arXiv:1504.03740; DOI 10.1103/PhysRevA.92.022117 | OK | Crossref: PRA 92, 022117 (2015-08-19). |
| Chow1970 | DOI 10.1109/TIT.1970.1054406 | OK | Crossref: "C. Chow", IEEE TIT 16(1), 41-46, January 1970. The CombesFerrie2015 reference list gives "C. K. Chow". |
| AGP2007 | arXiv:quant-ph/0703264 | OK | arXiv journal ref: Quant. Inf. Comput. 8 (2008) 181-244. |
| Ali2024 | arXiv:2403.00706; DOI 10.1103/PhysRevApplied.22.044031 | OK | Crossref: PR Applied 22, 044031 (2024-10-11); 10 authors, last Earl T. Campbell. |
| Levine2023 | arXiv:2307.08737; DOI 10.1103/PhysRevX.14.011051 | OK | Crossref: PRX 14, 011051 (2024-03-20); 113 authors, last Oskar Painter. |
| Combes2017 | arXiv:1702.03688 | OK | arXiv only. |
| Kumar2026 | arXiv:2604.19871 | OK | Six authors as cited. |
| Shtanko2026 | arXiv:2610.12400 | OK | Oles Shtanko, Saswata Roy, Shashwat Kumar, Zlatko Minev; v1 2026-10-08. |
| Tahaab2026 | arXiv:2608.10576 | OK | Maries Tahaab, Tim Chan, James Kirkman, Simon C. Benjamin. |
| Forney1968 | DOI 10.1109/TIT.1968.1054129 | OK | Crossref: "G. Forney" (G. D. Forney Jr.), IEEE TIT 14(2), 206-220, March 1968. |
| Merhav2006 | arXiv:cs/0604069 | OK | Neri Merhav, Meir Feder. |

The five screened, unused sources in section 1c also check out:

- Staples, Fu, Thompson, arXiv:2603.08697;
- Mishima et al., arXiv:2605.14637 (7 authors);
- Cao, Yan, Du, Pan, arXiv:2605.17230;
- Varbanov et al., arXiv:2002.07119, npj Quantum Inf. 6, 102 (2020);
- Knill, arXiv:quant-ph/0410199, which carries DOI 10.1038/nature03350.

The seed corrections in section 4 of the note are correct as stated. In particular:

- English2024's Theorem 1 really is stated for abort sets of equal cardinality.
- Yoked surface codes really uses the gaps as outer-decoder edge weights and does not discard on them.

## 2. Key findings (the 21 in the structured summary)

Verdict scale: supported, partially supported, unsupported, not checkable.

1. **Max-coset-probability thresholding is Pareto-optimal under an abort-probability constraint;
   English et al.'s cardinality version is wrong (Seip2026, English2024).** Supported.
   - Seip Theorem VI.1: if p_abort(c', f') <= p_abort(c, f), then p_L(c', f') >= p_L(c, f).
   - Remark G.2: "In Ref. [36] a similar result is stated. However, we find that their theorem is
     wrong." Ref. [36] is English et al.
   - English Theorem 1 (v3) reads "abort set ... of some cardinality |E_abort| = r ... for any abort
     set with equivalent cardinality r".
2. **Chow's rule; slope equals threshold; conditional loss means zero rejection cost (Chow1970,
   CombesFerrie2015).** Supported. CombesFerrie Sec. II says "find the largest posterior probability;
   if it is greater than or equal to the threshold 1-λ, report it". Sec. IV says "the slope of the
   error-reject tradeoff curve ... is exactly value of the rejection threshold". The discussion
   (Sec. VI) says that a loss "conditional on some event being successful" amounts to assigning a
   (zero) cost to rejection. Sec. III adds that at λ = 0 the risk is also minimised by "reporting
   'reject' for any outcome".
   - Wording nit: the slope magnitude is λ, which Chow calls the rejection threshold. It is not the
     posterior cutoff 1 - λ. The note's L08 places "threshold" right after "1 - λ", which invites the
     wrong reading.
3. **Complementary gap definition; the 0.9 rescaling; 23x less overhead than syndrome counting
   (Gidney2023yoked, Bombin2022).** Supported, with one precision fix.
   - Yoked Sec. 4: "we can form a detector connecting to all the boundary edges on one side ...
     turning this boundary detector on/off forces the decoder to match/not match". It calls the gap
     "the log-ratio of the probabilities of the minimum-weight matching and the complementary
     matching", bins gaps "in dB", and reports gaps "well-calibrated ... after rescaling the gap by
     0.9x" while it "tends towards overconfidence".
   - Bombín's abstract credits "post-selection strategies based on the logical gap" with "approximately
     23 times less" overhead than syndrome-counting rules. The body pins the number down: "the radial
     gap rule has relative overhead of only 1.78, which is ~23 times lower than the best annular
     syndrome rule and 1.17 times lower than the gap rule", at 0.6 of the bulk threshold. So the 23x
     belongs to the radial variant at α = 0.1, and the plain gap rule is about 20x. The overhead
     measured is the one needed to reach breakeven.
4. **Cluster soft output tracks the failure log-likelihood; swim distance and gap track the window
   LEP; d = 9 example (Meister2024, Dinca2025).** Supported.
   - Meister: "the soft output accurately approximates the log-likelihood". It is exactly the
     log-likelihood "when applied to UFD on the classical repetition code".
   - Meister's d = 9 example: "we remove a fraction ≈5×10^-4 of the data and achieve an improvement
     in the logical error rate from 3.0(5)×10^-5 to at most 2×10^-6 (95% confidence)". The setup is
     a rotated code, phenomenological noise with p = 0.005, T = 9 and UFD.
   - Dinca: both scores "reliably estimate the logical error probability (LEP) in a decoding window".
     They "find the complementary gap to be somewhat more accurate".
5. **Exclusive decoders (Smith2024).** Supported. Sec. II.1 gives the abort rule "1 - Δ/d > c", with
   c = 0 defined as aborting on any non-trivial syndrome and c = 1 as standard MWPM.
   - The threshold is "50% at zero tolerance, and decreases smoothly down to the standard MWPM decoder
     value of 15%". With phenomenological measurement errors (p_m = 2p/3) it is 32(1)%.
   - "For the case of zero-tolerance, the exclusive decoder does not have an abort threshold". Its
     abort probability "increases quadratically with the code distance" (App. C).
   - For c > 0, the abort probability "decreases exponentially in the distance" below the abort
     threshold.
6. **Free-energy postselection, p_f to p_f^b (Chen2025).** Supported.
   - b = 3.1(1) for the toric code with perfect measurements (abstract).
   - "decoding under circuit-level noise with p=0.005 gives b ≈ 2.87" with MWPM (Supplement, Fig. S5).
   - The abort rate is exp(-I(s*) d).
   - "for circuits of depth scaling exponentially in d ... the gain from postselection is entirely
     lost". Also "postselection with finite acceptance probability does not alter the error threshold"
     (supports L16).
7. **Correct-and-reject works: d = 4 beats d = 5 by about 25x; tesseract 82% kept, 0.11% vs 2.7%,
   about 2% per round in simulation (Prabhu2021, Reichardt2024).** Supported.
   - Prabhu abstract: "accumulates 25 times less error than its distance-five counterpart".
   - Reichardt Table 3, "5 rounds of 4 qubit error correction": 2500 trials, acceptance 82(+1/-2)%,
     error 0.11(+0.21/-0.09)%, against a 2.7(4)% teleportation baseline.
   - Reichardt Sec. on simulations (Fig. 13): "2.02(2)% of the surviving trials are discarded" per
     round.
   - Context the note omits: 297 of the 2500 trials were pre-rejected at state preparation and 148
     post-rejected.
8. **Neutral-atom sliding threshold: 72% to about 90% at 50% discard, 99.85% with full detection; raw
   counts suboptimal (Bluvstein2023).** Supported. The text says the GHZ fidelity "is 72(2)% (again
   using correlated decoding)", "discarding just 50% of the data improves GHZ fidelity to ≈ 90%", and
   error detection gives "99.85(+0.1/-1.0)%". The threshold is "a threshold weight of the overall
   correction weight on the decoding hypergraph". Methods: counting detected stabilizer errors "is
   suboptimal".
   - Context: the 72% baseline already postselects on the fault-tolerant-preparation flags (d = 3 color
     codes). Only stabilizer postselection is absent from it.
9. **Cultivation: detection-only early stages and a gap cutoff in escape; the hardware run used
   detection-only postselection (8%, 0.9999) and left decoder-based selection for future work
   (Gidney2024cult, Rosenfeld2025).** Partially supported.
   - Supported parts:
     - Cultivation Sec. 2.5 (Decoding): "During the injection stage and the cultivation stage, we use
       full postselection", and keep-or-discard is decided "based on the complementary gap of the
       escape stage". The same section notes that the escape stage also postselects the red X and
       green Z color-code stabilizers, a detail the note leaves out.
     - Rosenfeld abstract: "state fidelity of 0.9999(1) (retaining 8% of attempts)". The main text
       postselects "for no detection events".
   - Not in the source: Rosenfeld nowhere says that decoder-based or gap-based selection is left for
     future work. The closest statements are:
     - "one could instead decode in a trade-off of performance for speed, beyond the scope of this
       work" (SI, about kickback tomography);
     - "further study into decoders for the grafted code" (outlook).
   - Also, the grafting experiments did use a decoder, Tesseract, after postselecting the
     state-preparation detectors. "Detection-only" describes the headline cultivation data, not every
     experiment in the paper.
10. **NN-confidence postselection beats leakage postselection, but the kept fraction decays
    exponentially (Xin2025).** Supported. Suppl. Sec. V says "Post-selection on NN confidence ...
    performs better than simple leakage post-selection" and "maintains a nearly flat error rate". It
    also says "The fraction of shots kept, f, decays exponentially with R ... This is why
    post-selection is not extendable to large-sca[le experiments]".
11. **Soft readout: about 25% threshold gain in theory, IBM d up to 51 with up to 30x, one byte, Delft
    d = 3 up to 6.8% (Pattison2021, Hanisch2024, Ali2024).** Supported.
    - Pattison abstract: "threshold 25% higher than any hard decoder for phenomenological noise with
      Gaussian soft measurement outcomes".
    - Hanisch: IBM Sherbrooke, "d=3 to d=51", "up to 30 times lower error rates", "a single byte of
      information per measurement suffices". Leaked outcomes get a soft flip probability of 0.5
      instead of being postselected, which supports L32.
    - Ali abstract: "reduction of up to 6.8% in the extracted logical error rate" for a d = 3 bit-flip
      surface code.
12. **Using soft output without discarding: MLE, logical PEC and ZNE, in-situ channel estimates, VEC
    (Dinca2025, Zhou2025, Shtanko2026).** Supported.
    - Dinca:
      - "use it as a basis for maximum likelihood estimation";
      - "the amount of achievable reduction is very sensitive to DCS accuracy";
      - "decoding windows are not independent".
    - Zhou abstract: PEC, ZNE and "in situ characterization of logical channels".
    - Shtanko abstract: virtual error cancellation, "entirely in classical post-processing".
13. **Heralded RB fits two rates (Levine2023).** Supported. The Fig. 3 caption fits the "postselection
    probability ... to an exponential decay to 0 to extract the erasure error per Clifford of
    4.38(3)×10^-3". It fits the postselected survival "with offset 1/2" and gets a "residual error rate
    of 1.01(1)×10^-4 per Clifford". "Natural template" is the author's judgment.
    - Context: Levine used 11 erasure checks per circuit, "regardless of depth", plus the final
      readout. That is not per-round heralding (see section 3, item 8).
14. **[[5,1,2]] phase flips: (1,0) and (0,1) have identical enumerators (1,2,3,4), so g = 1/2 exactly;
    (1,1) has gap 1 and g about 1 - 2p; trivial syndrome has gap 2.** Supported (I re-enumerated it).
    Coset weights, P(s) and g reproduce exactly at p = 1e-3, 1e-2 and 5e-2. For example, g(1,1) =
    0.998004 at p = 1e-3.
15. **Per-round rejection and error: detect 5p and 2p^2; gap >= 1 4p and 4p^2; ML 0 and 2p; equal A*F
    = (1-2p)^2.** Supported (re-enumerated). At p = 1e-3 the three rules give:
    - detect: A = 0.995012, conditional error 2.006e-6, A*F = 0.995008;
    - gap >= 1: 0.996012, 4.008e-6, 0.996004;
    - full ML: 1, 1.998e-3, 0.996004.

    (1-2p)^2 = 0.996004. The gap >= 1 rule and full ML have equal A*F at every p checked, as expected
    because the rejected syndromes have g = 1/2 exactly.
16. **Depolarizing, joint decoding: only pure-Z faults on 0, 2, 3, 4 stay ambiguous; posterior >= 0.6
    rejects about 4p/3 vs 5p, keeps O(p^2) error, matches full ML in A*F.** Supported
    (re-enumerated).
    - Every Table 3 entry reproduces at p = 1e-3 and p = 1e-2. For example, post >= 0.6 gives A =
      0.998669, error 3.55e-6, A*F = 0.998662, and full ML gives A*F = 0.998662.
    - The rejected set is exactly the four syndromes with g = 0.5. Two are single pure-Z faults, about
      6.6e-4 each; two are weight-2 cases, about 9e-7 each. Their total is 1.331e-3 against 4p/3 =
      1.333e-3.
    - The Y0 posterior is 0.998667.
17. **Coin-flip scoring: A*F = sum over accepted s of P(s)(2g - 1); full ML maximises it; rejecting g =
    1/2 is free; this is Chow at λ = 1/2.** Supported. The algebra is correct. CombesFerrie Sec. V
    gives the same condition directly: "if λ_R > λ_E/2 we should never reject". The enumeration
    confirms that A*F is equal to all printed digits.
18. **At d = 2, Smith's family collapses to three policies.** Supported. I computed Smith's Δ for all
    16 joint syndromes, with Y counted as weight 1.
    - Δ = 2 on the trivial syndrome and Δ ∈ {0, 1} on every non-trivial one.
    - This holds both with Smith's four-class gap and with the X_L-only gap. So c < 1/2 is pure
      detection, 1/2 <= c < 1 rejects only Δ = 0, and c = 1 is MWPM.
    - Caveat: under Smith's actual four-class Δ, the pure-X ambiguous syndromes (X0 vs X3, X2 vs X4)
      also have Δ = 0. They do not affect the X_L readout, so use the readout-relevant gap.
19. **Under SI1000, extra acceptance should come mostly from time-like measurement flips.** Not
    checkable (labelled conjecture). Its premise that X_ERROR(5p) is the largest single term matches
    the noise model in CONTEXT.md.
20. **LRB-D memory explains the non-exponential const=1 decay; per-round correction restores
    near-exponential decay.** Not checkable (labelled conjecture). The cited side facts hold: Levine's
    two-rate fit, and Dinca's statement that decoding windows are not independent.
21. **At d = 2, any rule stricter than rejecting only g = 1/2 pays Θ(p) per round; the sustainable mode
    is always-correct plus soft weighting or flags.** Partially supported.
    - The literature parts are represented correctly:
      - scalable postselection needs growing d (Chen, Smith);
      - the kept fraction decays exponentially on hardware (Xin);
      - FPGA confidence scores cost negligible latency (Tahaab);
      - per-shot MLE, PEC and ZNE (Dinca, Zhou);
      - soft edge weights (Hanisch).
    - Problem 1: the framing is imprecise. By the note's own Table 3, which I re-verified, the "reject
      only g = 1/2" rule already rejects about 4p/3 per round, so it is also Θ(p). The accurate
      statement is the one in section 3.5 of the note: any non-trivial rejection rule at d = 2 makes
      A(m) decay exponentially.
    - Problem 2: section 3.3 item 7 credits the "about 2% per round discard" to "the tesseract
      experiment". In fact it comes from emulator simulations of one code block over the first 10
      rounds.

## 3. Other issues in the note (L items and section 3)

1. **L01.** Same as finding 3. The 23x applies to the radial logical gap at α = 0.1 compared with the
   best annular rule, for reaching breakeven. The plain gap rule needs 1.17x more overhead than the
   radial rule.
2. **L05.** Meister proposes the union bound for assigning one soft output to a whole circuit made of
   windows (Sec. V). The d = 9 example is a single memory block, and no union bound was used there.
3. **L14 and open question 7.** Seip reports "abrupt jumps in the abort probability and sudden drops in
   the logical error rate" as p varies at a fixed cutoff (App. H, Fig. 19).
   - The "staircase trade-off" and the "randomised acceptance" fix are the author's own inference. Seip
     mentions neither.
   - A direct source for step-like behaviour is Bombín Fig. 3: "step-like due to the discrete logical
     gap sectors".
4. **L17.** "improves the failure exponent up to quadratically" should read "up to a quadratic
   improvement in logical failure rates". That is up to a doubling of the exponent, not a squaring.
5. **L35.** Same as finding 9. Drop "It left decoding-based selection to future work", or quote the SI
   sentence about decoding being beyond scope. Add that the grafting data were decoded with the
   Tesseract decoder.
6. **Section 3.3 item 1.** The note cites English2024 together with Seip2026 for Pareto optimality.
   Cite Seip Theorem VI.1 alone. English's Theorem 1 is the cardinality version that L13 says is wrong.
7. **Section 3.3 item 7.** Fix the two points from finding 21. Rejecting only the g = 1/2 class
   already costs Θ(p), and the 2% per round figure is simulated.
8. **Section 3.4 item 3.** Levine's single-exponential fits come from a fixed 11 checks per circuit plus
   the final readout, not from per-round heralding. That the fits extend to per-round heralding is the
   note's own conjecture.
9. **Section 3.3 item 2.** Add the caveat from finding 18 about Smith's four-class Δ.
10. **Section 3.5 item 1.** The note offers "a cluster proxy" as a cheap confidence signal. Dinca2025
    (Discussion) reports that Lee2025's cluster-statistic score "does not satisfy condition (1), as its
    correlation with the true gap is weak", citing Lee's Suppl. Note 1. Condition (1) is calibration to
    the true log odds. So a cluster score is fine for abort decisions but risky for per-shot MLE
    weighting.
11. **Bibliography.** Add the journal versions of Gidney2023yoked and Chen2025 from section 1.

Everything else in section 2 checks out against the text named above. That covers L02-L04,
L06-L13, L15, L16, L18-L34 and L36. Some are abstract-only by design: L24 (Lee), L27 (Shtanko), L30
(Kumar), L31 (Pattison), L33 (Ali) and L36 (Tahaab). For those, every number quoted in the note
appears in the abstract.

## 4. Independent re-run of section 3.1

My code is independent of the appendix, but it uses the same stabilizers, logicals and X_L class bit.
- Stabilizer and logical commutation checks pass. The single-qubit syndrome table in CONTEXT.md is
  confirmed. The alternative logical representatives X2X4 and Z3Z4 are confirmed.
- Table 1: every coset weight list, P(s) and g matches.
- Table 2: every A, conditional error and A*F matches. The ratios rejection/p and error/p^2 come out
  4.99 and 2.01 for detect, and 3.99 and 4.01 for gap >= 1, at p = 1e-3.
- Table 3: every entry matches at both p values. The distinct posteriors at p = 1e-3 start at 0.5,
  0.74975, 0.998667 and so on. So the 0.6 cutoff removes exactly the g = 1/2 class at code capacity.
- Smith Δ: computed for all 16 joint syndromes (see finding 18).
