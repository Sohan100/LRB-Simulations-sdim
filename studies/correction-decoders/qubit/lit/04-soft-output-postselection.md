# Lit review 04: soft-output decoding, complementary gap, decoder-informed postselection

Scope: using a decoder's soft output (complementary gap, posterior or confidence
score) to postselect or flag low-confidence shots; acceptance vs logical-error
trade-offs (Pareto curves); combining postselection with correction; thresholds
for postselected QEC; optimality of likelihood-ratio postselection
(Neyman-Pearson or Chow style). Written for the qubit `[[5,1,2]]` LRB study
(see `../CONTEXT.md`).

How to read the labels:

- Every finding in section 2 is **from literature**. The cited source was
  fetched and read at the level noted in the bibliography.
- Statements in section 3 carry their own label: **verified analytically**,
  **verified numerically** (small enumerations done for this note, code
  appended), or **conjecture**.
- Nothing here is cited from memory alone.

Method: 19 web searches (section 6). For each source I fetched the arXiv
abstract page, arXiv HTML or ar5iv full text where a claim needed it, and
Crossref records for the two IEEE papers. Accessed 2026-10-28.

## 0. Summary

1. The soft output that matters is the posterior probability, given the
   syndrome, of the logical outcome actually read out. Practical proxies are:
   - the complementary or logical gap (MWPM run once per logical class);
   - cluster-based "swim distance" (the shortest path between the decoder's
     boundaries after zeroing edges inside clusters);
   - neural-network decoder confidence.
2. Thresholding the maximum-likelihood coset probability is the Pareto-optimal
   accept rule, provided the constraint is the abort probability. Seip et al.
   (2026) show that the earlier English et al. theorem, stated in terms of the
   size (cardinality) of the abort set, is wrong as stated.
3. This is Chow's classical reject rule. For a binary outcome, rejecting pays
   only when a rejection costs less than a coin flip.
4. Combined "correct what you can, reject what you cannot" rules already run on
   hardware: the tesseract code on trapped ions, and sliding thresholds on
   correlated-decoding weight on neutral atoms. Analog readout sharpens the
   confidence on superconducting devices.
5. "Scalable postselection" (abort rate falling with distance, error
   p_f -> p_f^b) needs growing distance and does not exist at d = 2.
6. For the `[[5,1,2]]` code at code capacity, my own enumeration (section 3)
   shows:
   - With independent X/Z decoding, 4 of 5 single phase flips are exactly
     ambiguous: posterior 1/2 for every p.
   - Joint decoding under depolarizing noise resolves Y faults. A rule that
     accepts when the posterior is at least 0.6 rejects about 4p/3 per round
     instead of about 5p, and the conditional error stays O(p^2).
   - The rejection-charged figure A*F (acceptance times fidelity) of that rule
     is exactly that of full ML correction.
7. Conjecture: under SI1000, most of the extra acceptance should come from
   keeping shots whose syndrome is explained by a measurement flip.

## 1. Bibliography

All URLs were fetched. "Fetched" says what was read: abs = arXiv abstract page;
full = arXiv HTML or ar5iv full text (targeted sections); Crossref = DOI
registry record.

### 1a. Core sources (20)

| key | title | authors | year | arXiv / DOI (venue) | URL | fetched |
|---|---|---|---|---|---|---|
| Bombin2022 | Fault-tolerant Post-Selection for Low Overhead Magic State Preparation | H. Bombín, M. Pant, S. Roberts, K. I. Seetharam | 2022 (journal 2024) | arXiv:2212.00813; DOI 10.1103/PRXQuantum.5.010302 (PRX Quantum 5, 010302) | https://arxiv.org/abs/2212.00813 | yes (abs + full) |
| Gidney2023yoked | Yoked surface codes | C. Gidney, M. Newman, P. Brooks, C. Jones | 2023 | arXiv:2312.04522 | https://arxiv.org/abs/2312.04522 | yes (abs + full) |
| Gidney2024cult | Magic state cultivation: growing T states as cheap as CNOT gates | C. Gidney, N. Shutty, C. Jones | 2024 | arXiv:2409.17595 | https://arxiv.org/abs/2409.17595 | yes (abs + full) |
| Meister2024 | Efficient soft-output decoders for the surface code | N. Meister, C. A. Pattison, J. Preskill | 2024 | arXiv:2405.07433 | https://arxiv.org/abs/2405.07433 | yes (abs + full) |
| Smith2024 | Mitigating errors in logical qubits | S. C. Smith, B. J. Brown, S. D. Bartlett | 2024 | arXiv:2405.03766; DOI 10.1038/s42005-024-01883-4 (Commun. Phys. 7, 386) | https://arxiv.org/abs/2405.03766 | yes (abs + full) |
| English2024 | Thresholds for post-selected quantum error correction from statistical mechanics | L. H. English, D. J. Williamson, S. D. Bartlett | 2024 (journal 2025) | arXiv:2410.07598; DOI 10.1103/nh49-52y2 (PRL 135, 120603) | https://arxiv.org/abs/2410.07598 | yes (abs + full v3) |
| Seip2026 | Machine Learning Optimal Quantum Error Correction Thresholds | D. Seip, L. Colmenarez, M. Schmitt, M. Müller | 2026 | arXiv:2606.22194 | https://arxiv.org/abs/2606.22194 | yes (abs + full incl. App. G) |
| Chen2025 | Scalable accuracy gains from postselection in quantum error correcting codes | H. Chen, D. Xu, G. M. Sommers, D. A. Huse, J. D. Thompson, S. Gopalakrishnan | 2025 | arXiv:2510.05222 | https://arxiv.org/abs/2510.05222 | yes (abs + full v3) |
| Lee2025 | Efficient Post-Selection for General Quantum LDPC Codes | S.-H. Lee, L. H. English, S. D. Bartlett | 2025 (journal 2026) | arXiv:2510.05795; DOI 10.1038/s41534-026-01242-x (npj Quantum Inf. 12, 96) | https://arxiv.org/abs/2510.05795 | yes (abs) |
| Prabhu2021 | Distance-four quantum codes with combined postselection and error correction | P. Prabhu, B. W. Reichardt | 2021 (journal 2024) | arXiv:2112.03785; DOI 10.1103/PhysRevA.110.012419 (PRA 110, 012419) | https://arxiv.org/abs/2112.03785 | yes (abs) |
| Reichardt2024 | Demonstration of quantum computation and error correction with a tesseract code | B. W. Reichardt et al. (18 authors, last M. Zanner) | 2024 | arXiv:2409.04628 | https://arxiv.org/abs/2409.04628 | yes (abs + full v2) |
| Pattison2021 | Improved quantum error correction using soft information | C. A. Pattison, M. E. Beverland, M. P. da Silva, N. Delfosse | 2021 | arXiv:2107.13589 | https://arxiv.org/abs/2107.13589 | yes (abs) |
| Dinca2025 | Error mitigation for logical circuits using decoder confidence | M. Dincă, T. Chan, S. C. Benjamin | 2025 (v4 2026) | arXiv:2512.15689 | https://arxiv.org/abs/2512.15689 | yes (abs + full) |
| Zhou2025 | Error Mitigation of Fault-Tolerant Quantum Circuits with Soft Information | Z. Zhou, S. Pexton, A. Kubica, Y. Ding | 2025 | arXiv:2512.09863 | https://arxiv.org/abs/2512.09863 | yes (abs + full) |
| Bluvstein2023 | Logical quantum processor based on reconfigurable atom arrays | D. Bluvstein, S. J. Evered, A. A. Geim, et al. (last M. D. Lukin) | 2023 (print 2024) | arXiv:2312.03982; DOI 10.1038/s41586-023-06927-3 (Nature 626, 58-65) | https://arxiv.org/abs/2312.03982 | yes (abs + full + Crossref) |
| Hanisch2024 | Soft information decoding with superconducting qubits | M. D. Hanisch, B. Hetényi, J. R. Wootton | 2024 (journal 2026) | arXiv:2411.16228; DOI 10.1103/y9fh-4x6n (APS Open Sci. 1, 000019) | https://arxiv.org/abs/2411.16228 | yes (abs + full v2) |
| Xin2025 | Improved error correction with leakage reduction units built into qubit measurement in a superconducting quantum processor | Y. Xin, S. L. M. van der Meer, M. Serra-Peralta, T. H. F. Vroomans, M. Finkel, H. M. Veen, M. W. Beekman, L. DiCarlo | 2025 (journal 2026) | arXiv:2511.17460; DOI 10.1103/y66v-988s (PRX Quantum 7, 033062) | https://arxiv.org/abs/2511.17460 | yes (abs + full + Crossref) |
| Rosenfeld2025 | Magic state cultivation on a superconducting quantum processor | E. Rosenfeld, C. Gidney, G. Roberts, A. Morvan, et al. | 2025 | arXiv:2512.13908 | https://arxiv.org/abs/2512.13908 | yes (abs + full) |
| CombesFerrie2015 | Cost of postselection in decision theory | J. Combes, C. Ferrie | 2015 | arXiv:1504.03740; DOI 10.1103/PhysRevA.92.022117 (PRA 92, 022117) | https://arxiv.org/abs/1504.03740 | yes (abs + full) |
| Chow1970 | On optimum recognition error and reject tradeoff | C. K. Chow (Crossref lists "C. Chow") | 1970 | DOI 10.1109/TIT.1970.1054406 (IEEE Trans. Inf. Theory 16(1), 41-46) | https://doi.org/10.1109/TIT.1970.1054406 | metadata yes (Crossref); content via CombesFerrie2015 |

### 1b. Supporting sources (each cited for one point)

| key | title | authors | year | arXiv / DOI (venue) | URL | fetched |
|---|---|---|---|---|---|---|
| AGP2007 | Accuracy threshold for postselected quantum computation | P. Aliferis, D. Gottesman, J. Preskill | 2007 (journal 2008) | arXiv:quant-ph/0703264 (Quantum Inf. Comput. 8, 181-244) | https://arxiv.org/abs/quant-ph/0703264 | yes (abs) |
| Ali2024 | Reducing the error rate of a superconducting logical qubit using analog readout information | H. Ali, J. Marques, O. Crawford, et al. (last E. T. Campbell) | 2024 | arXiv:2403.00706; DOI 10.1103/PhysRevApplied.22.044031 (PR Applied 22, 044031) | https://arxiv.org/abs/2403.00706 | yes (abs) |
| Levine2023 | Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons | H. Levine, A. Haim, J. S. C. Hung, N. Alidoust, et al. (last O. Painter) | 2023 (journal 2024) | arXiv:2307.08737; DOI 10.1103/PhysRevX.14.011051 (PRX 14, 011051) | https://arxiv.org/abs/2307.08737 | yes (abs + full, Fig. 3) |
| Combes2017 | Logical Randomized Benchmarking | J. Combes, C. Granade, C. Ferrie, S. T. Flammia | 2017 | arXiv:1702.03688 | https://arxiv.org/abs/1702.03688 | yes (abs + part of full) |
| Kumar2026 | Co-Designing Error Mitigation and Error Detection for Logical Qubits | R. S. Kumar, T. Tsunoda, S. H. Xue, D. Li, R. J. Schoelkopf, Y. Ding | 2026 | arXiv:2604.19871 | https://arxiv.org/abs/2604.19871 | yes (abs) |
| Shtanko2026 | Measure Now, Mitigate Later: Virtual Error Cancellation for Logical Quantum Circuits | O. Shtanko, S. Roy, S. Kumar, Z. Minev | 2026 | arXiv:2610.12400 | https://arxiv.org/abs/2610.12400 | yes (abs) |
| Tahaab2026 | Stream Decoding with Confidence Scores at Room and Cryogenic Temperatures | M. Tahaab, T. Chan, J. Kirkman, S. C. Benjamin | 2026 | arXiv:2608.10576 | https://arxiv.org/abs/2608.10576 | yes (abs) |
| Forney1968 | Exponential error bounds for erasure, list, and decision feedback schemes | G. Forney (as listed by Crossref) | 1968 | DOI 10.1109/TIT.1968.1054129 (IEEE Trans. Inf. Theory 14(2), 206-220) | https://doi.org/10.1109/TIT.1968.1054129 | metadata yes (Crossref); content via Merhav2006 |
| Merhav2006 | Universal decoding with an erasure option | N. Merhav, M. Feder | 2006 | arXiv:cs/0604069 | https://arxiv.org/abs/cs/0604069 | yes (abs) |

### 1c. Screened, not used for any claim

These pages were fetched while searching. They are not used because they are
off the main line or could not be read beyond the abstract:

- Staples, Fu, Thompson, "Scalable Postselection of Quantum Resources",
  arXiv:2603.08697 (2026). Partial-gap postselection of resource states.
- Mishima et al., "Adaptive Window Decoding based on Spatiotemporal
  Complementary Gap", arXiv:2605.14637 (2026). Uses the gap to decide when to
  re-decode with a larger buffer, not to discard.
- Cao, Yan, Du, Pan, "Maximum Likelihood Decoding of Quantum Error Correction
  Codes" (review), arXiv:2605.17230 (2026).
- Varbanov et al., "Leakage detection for a transmon-based surface code",
  arXiv:2002.07119, npj Quantum Inf. 6, 102 (2020).
- Knill, arXiv:quant-ph/0410199 (arXiv title "Quantum Computing with Very Noisy
  Devices"), published as "Quantum computing with realistically noisy devices",
  Nature 434, 39-44 (2005), DOI 10.1038/nature03350. The PDF text could not be
  extracted here, so it is cited only through AGP2007.

## 2. Key findings (all **from literature**)

Format: `Lxx` [keys] (question) one sentence. Q1 = decoders, Q2 =
non-Markovianity, Q3 = hardware.

### 2.1 What the soft output is and how it is computed

- **L01** [Bombin2022] (Q1) The logical gap is the difference, in
  log-likelihood edge weights w_e = ln((1-p_e)/p_e), between the minimum-weight
  corrections of the two inequivalent logical classes. The decoder fails when
  the signed gap is negative. Ranking runs by gap needed about 23x less overhead
  than the best syndrome-counting ("annular") rule at 0.6 of threshold.
- **L02** [Gidney2023yoked] (Q1) The complementary gap is the log-likelihood
  ratio between the minimum-weight matching and the best matching constrained to
  the opposite logical outcome. It is obtained by toggling an extra boundary
  detector and quoted in dB. It is slightly overconfident for MWPM: an empirical
  0.9 rescaling matched the observed failure rates.
- **L03** [Gidney2023yoked] (Q1) In yoked surface codes the gaps are not used to
  discard shots. They become edge weights in the outer (yoke) decoder, so the
  soft output tells the outer code which inner patch most likely failed.
- **L04** [Meister2024] (Q1) A soft-output decoder estimates P(logical failure |
  syndrome). Their cluster-geometry proxy is the shortest boundary-to-boundary
  path after zeroing the in-cluster edges. It tracks the log-likelihood of
  failure, equals it exactly for the repetition code under Union-Find, and
  avoids re-running MWPM on atypical syndromes as the complementary gap does.
- **L05** [Meister2024] (Q1) In a d = 9 phenomenological-noise sampling example,
  discarding about 5x10^-4 of the samples by soft output lowered the logical
  error from 3.0(5)x10^-5 to at most 2x10^-6. Per-window soft outputs were
  combined by a union bound.
- **L06** [Dinca2025] (Q1) Swim distance and complementary gap both track the
  window's logical error probability against tensor-network ground truth under
  phenomenological noise, through an empirical calibration curve. The gap was
  somewhat more accurate.
- **L07** [Zhou2025] (Q1) The most informative soft output is the posterior
  over logical classes: exact by tensor-network contraction, approximate by
  boundary-enforced MWPM. Averaging it over shots gives an in-situ estimate of
  the logical channel with no extra circuits. This was shown with perfect
  syndrome extraction; circuit-level noise is left to future work.

### 2.2 Optimal accept rules and thresholds

- **L08** [Chow1970, CombesFerrie2015] (Q1) Chow's rule: report the most probable
  hypothesis if its posterior is at least 1 - λ, otherwise reject, where λ is
  the cost of a rejection relative to an error. This minimises Pr(error) +
  λ Pr(reject). The slope of the optimal error-reject curve equals the
  threshold, so the first rejections remove the most error.
- **L09** [CombesFerrie2015] (Q1) Reporting a loss conditioned on acceptance is
  equivalent to giving rejection zero cost, and then rejecting everything is
  optimal. Comparing postselected with non-postselected schemes therefore needs
  an explicit rejection cost.
- **L10** [Forney1968, Merhav2006] (Q1) In classical channel coding, Forney's
  erasure/list decoder is the reference rule that optimally trades the
  probability of erasure against the probability of undetected error.
- **L11** [English2024] (Q1) Optimal postselection aborts when the
  maximum-likelihood (ML) coset probability falls below a cutoff. Post-selected
  QEC has both a conditional logical threshold and an abort threshold, giving
  four phases. A decoder-free rule on the density of violated stabilizers works
  as a heuristic. The analysis is for code-capacity bit flips; circuit level is
  left open.
- **L12** [Seip2026] (Q1) Thresholding the maximum coset probability is
  Pareto-optimal: any rule whose abort probability is no larger has a logical
  error rate no smaller. The code's threshold is preserved under this
  postselection.
- **L13** [Seip2026, English2024] (Q1) English et al.'s optimality theorem is
  stated for abort sets of equal cardinality, and Seip et al. (Remark G.2) give
  a counterexample showing it is wrong as stated. The correct constraint is the
  abort probability.
- **L14** [Seip2026] (Q1) At finite size, the abort probability and the logical
  error rate jump abruptly as individual syndromes cross the cutoff. The
  achievable trade-off is a staircase, not a smooth curve.
- **L15** [Chen2025] (Q1) In the coding phase, most logical failures come from
  exponentially unlikely syndromes. Discarding trials with small |ΔF| (the
  free-energy difference, i.e. log-likelihood ratio, between logical sectors)
  lowers p_f to p_f^b with b >= 2, at an abort rate exponentially small in d.
  Measured values: b = 3.1(1) for the toric code with perfect measurements, and
  about 2.87 with MWPM under circuit noise.
- **L16** [Chen2025] (Q1) The gain is lost for circuits whose depth grows
  exponentially in d, and postselection at finite acceptance does not move the
  threshold.
- **L17** [Smith2024] (Q1) Exclusive decoders abort when 1 - Δ/d > c, where Δ is
  the gap. One tolerance c interpolates between aborting on any non-trivial
  syndrome (c = 0) and standard MWPM (c = 1). Zero tolerance raises the
  code-capacity depolarizing threshold from 15% to 50% (32(1)% with
  phenomenological measurement errors) and improves the failure exponent up to
  quadratically.
- **L18** [Smith2024] (Q1, Q2) For c > 0 there is an abort threshold (1.3%,
  2.1%, 4.5% for c = 2/5, 1/2, 2/3), below which the abort probability falls
  exponentially with distance. Zero tolerance has no abort threshold, and its
  abort probability grows with distance.
- **L19** [AGP2007] (Q1) A rigorous threshold (>= 1.04x10^-3 for independent
  stochastic noise) exists for postselected computation in which ancilla
  preparation is protected by a concatenated error-detecting code and discarded
  whenever an error is flagged (Knill's scheme).

### 2.3 Combined correction and postselection

- **L20** [Prabhu2021] (Q1) Correcting single faults while postselecting against
  detected double faults makes a logical failure need at least three faults. A
  distance-four surface code run this way accumulates about 25x less error than
  its distance-five counterpart.
- **L21** [Reichardt2024] (Q1, Q3) On Quantinuum trapped-ion hardware, the
  [[16,4,4]] tesseract code corrected flagged single faults and rejected
  uncorrectable patterns. Over five rounds on four logical qubits it kept 82% of
  trials, with logical error 0.11% against a 2.7% baseline. Simulations put the
  per-round discard at about 2% of surviving trials.
- **L22** [Bluvstein2023] (Q1, Q3) On neutral atoms, a sliding confidence
  threshold on the correlated-decoding correction weight raised a 4-logical-qubit
  GHZ fidelity from 72(2)% (no stabilizer postselection) to about 90% at 50%
  discard, and to 99.85% with full detection. The authors note that raw counts
  of detected errors are a suboptimal score.
- **L23** [Gidney2024cult] (Q1) Magic state cultivation discards on any
  detection event in its injection and cultivation stages and on a
  complementary-gap cutoff in its escape stage. Varying the cutoff trades
  expected attempts against logical error.
- **L24** [Lee2025] (Q1, Q3) Cluster statistics from clustering decoders
  (aggregated cluster sizes, log-likelihood ratios) replace the gap for general
  QLDPC codes and allow early mid-circuit abort with sliding-window decoding. For
  the [[144,12,12]] bivariate bicycle code at p = 0.1%, this gave about three
  orders of magnitude lower logical error at 1% abort.

### 2.4 Using the soft output without (only) discarding; benchmarking

- **L25** [Dinca2025] (Q1, Q2) Aborting when any decoding window's confidence
  falls below threshold is very effective for shallow circuits. A
  maximum-likelihood estimator that gives each shot its own logical error
  probability helps further at no extra quantum cost. Caveats: windows are not
  independent, and the gain is very sensitive to calibration.
- **L26** [Zhou2025] (Q1) In their simulations, posterior-based postselection
  with runtime abort lowered the logical error more than 100x while discarding
  under 0.1% of shots. The same posteriors drive logical-level PEC and ZNE.
- **L27** [Shtanko2026] (Q1) Virtual error cancellation pairs the online decoder
  with an offline high-complexity decoder in classical post-processing. It
  claims a sampling overhead below that of syndrome-based postselection, and
  more than three orders of magnitude suppression in d = 7 simulations
  (abstract only).
- **L28** [Combes2017] (Q1) Logical RB estimates:
  - the uncorrectable error as one minus the fidelity with recovery;
  - the correctable error as the fidelity with recovery minus the fidelity
    without recovery.

  It assumes ancillas are refreshed each round.
- **L29** [Levine2023] (Q1, Q2) Clifford RB with mid-circuit erasure checks fit
  two rates for one heralded protocol. The postselection probability was fit to
  an exponential decay to zero (erasure 4.38(3)x10^-3 per Clifford), and the
  conditional survival separately (residual 1.01(1)x10^-4 per Clifford).
- **L30** [Kumar2026] (Q1, Q2) One detection cycle per gate did not break even
  in any code they evaluated. A transient error profile in the first detection
  cycle biased the PEC noise model until removed by "steady-state extraction"
  (abstract only).

### 2.5 Hardware evidence

- **L31** [Pattison2021] (Q3) Soft (analog) measurement outcomes raise the
  surface-code threshold about 25% over any hard decoder for Gaussian soft
  noise. The measurement time that minimises physical error differs from the one
  that minimises logical error.
- **L32** [Hanisch2024] (Q3) On IBM Sherbrooke, repetition codes up to d = 51
  were decoded with per-shot soft edge weights from IQ data. This gave a 25%
  higher threshold and up to 30x lower logical error, and one byte per
  measurement was enough. Suspected leakage was given maximally ambiguous
  (p = 0.5) edges instead of being postselected.
- **L33** [Ali2024] (Q3) Using analog readout in decoding reduced the logical
  error of a distance-3 bit-flip transmon surface code by up to 6.8%.
- **L34** [Xin2025] (Q2, Q3) Postselecting on a neural-network decoder's
  confidence beat postselecting on detected leakage: the error stayed nearly
  flat in the leakage rate. But the kept fraction decayed exponentially with the
  number of rounds, and the authors conclude postselection cannot be extended to
  large experiments.
- **L35** [Rosenfeld2025] (Q3) The superconducting cultivation experiment kept
  only shots with no detection events (8% of attempts, fidelity 0.9999(1)). It
  left decoding-based selection to future work.
- **L36** [Tahaab2026] (Q3) Decoder confidence scores were added to an FPGA
  streaming decoder (Snowflake) at negligible latency and resource cost, and
  tested at room and cryogenic temperature (abstract only).

## 3. Implications for the qubit `[[5,1,2]]` LRB study

### 3.1 Code-capacity gap structure of `[[5,1,2]]`

**Verified numerically** by exhaustive enumeration of the 32 Z-type patterns and
the 1024 Paulis on the 5 data qubits (code in the appendix). Scope: one round,
code capacity (perfect checks), terminal X_L = X0 X3 readout. Class "+" commutes
with X_L and class "-" anticommutes. Coset weights list the weights of the four
members of each coset.

Table 1. Independent phase flips (probability p per data qubit), equivalent to
decoding the X checks alone:

| X-check syndrome (S_X1, S_X2) | coset weights, class + | coset weights, class - | gap (weight units) | P(syndrome) at p = 1e-3 | ML posterior |
|---|---|---|---|---|---|
| (0,0) | 0, 3, 3, 4 | 2, 2, 3, 3 (Z0Z2, Z3Z4, ...) | 2 | 0.995 | 0.999998, about 1 - 2p^2 |
| (1,0) | 1, 2, 3, 4 (Z2, ...) | 1, 2, 3, 4 (Z0, ...) | 0 | 2.0e-3 | 0.5 exactly, for every p |
| (0,1) | 1, 2, 3, 4 (Z4, ...) | 1, 2, 3, 4 (Z3, ...) | 0 | 2.0e-3 | 0.5 exactly, for every p |
| (1,1) | 1, 2, 2, 5 (Z1, ...) | 2, 2, 3, 3 (Z0Z4, Z2Z3, ...) | 1 | 1.0e-3 | 0.998004, about 1 - 2p |

The two classes of the (1,0) and (0,1) syndromes have identical weight
enumerators. Under any i.i.d. noise these syndromes therefore carry no
information about the logical outcome. This is the exact form of the "can only
guess within each pair" remark in CONTEXT.md.

Table 2. Accept policies under independent phase flips (one round). "rej" is
the rejected fraction 1 - A; "cond. err" is the logical error among accepted
shots.

| policy | accepts | A at p = 1e-3 | cond. err at 1e-3 | A*F at 1e-3 | leading order (rej, cond. err) |
|---|---|---|---|---|---|
| detect | (0,0) | 0.995012 | 2.0e-6 | 0.995008 | 5p, 2p^2 |
| gap >= 1 | (0,0), (1,1); correct (1,1) | 0.996012 | 4.0e-6 | 0.996004 | 4p, 4p^2 |
| full ML correction | all | 1 | 2.0e-3 | 0.996004 | 0, 2p |

Here A*F = (1 - 2p)^2 for full correction.

Table 3. Single-qubit depolarizing noise (X, Y, Z each p/3), joint decoding of
all four checks, terminal X_L readout:

| policy | A (p = 1e-3) | cond. err | A*F | A (p = 1e-2) | cond. err | A*F |
|---|---|---|---|---|---|---|
| detect-all (any check fires -> reject) | 0.995010 | 2.2e-7 | 0.995010 | 0.951034 | 2.3e-5 | 0.950990 |
| detect-X (X checks only; the const=0 rule) | 0.996672 | 8.9e-7 | 0.996670 | 0.967196 | 9.1e-5 | 0.967021 |
| posterior >= 0.6 on the X_L class | 0.998669 | 3.6e-6 | 0.998662 | 0.986931 | 3.5e-4 | 0.986233 |
| full ML correction | 1 | 6.7e-4 | 0.998662 | 1 | 6.9e-3 | 0.986233 |

Reading of Table 3 (**verified numerically**, code capacity only):

- Joint decoding resolves Y faults. For example, Y0 fires Z-check (1,0) and
  X-check (1,0), and the single-fault explanation beats every two-fault one, so
  the posterior is 0.9987.
- Pure X faults do not affect the X_L readout at all.
- Only pure-Z faults on qubits 0, 2, 3, 4 stay ambiguous. Their total
  probability is 1.33e-3 at p = 1e-3, about 4p/3.
- So the posterior rule rejects about 4p/3 per round instead of about 5p
  (3.75x fewer), and its conditional error is still O(p^2).

### 3.2 A figure of merit that charges for rejection

**Verified analytically.** Count rejected shots as 0, i.e. as a coin flip on a
binary outcome with known ideal value. The acceptance-weighted polarization is
then

    P_eff = A * F = (N_correct - N_wrong) / N_total = sum over accepted s of P(s) * (2 g(s) - 1),

where g(s) >= 1/2 is the posterior of the class the ML decoder picks. Every
accepted syndrome adds a non-negative term. Hence:

- full ML correction maximises P_eff;
- rejecting syndromes with g(s) = 1/2 exactly costs nothing in P_eff and
  strictly raises the conditional F.

This is Chow's rule at λ = 1/2 [Chow1970, CombesFerrie2015]: with two classes,
max posterior >= 1/2 >= 1 - λ, so it never pays to reject. Tables 2 and 3 show
the identity numerically (the gap >= 1 or posterior >= 0.6 rules match full
correction to all printed digits).

Consequences for the study (**conjecture**, following L09):

- LRB-D "wins" only on the conditional F, which corresponds to λ = 0.
- Any comparison of LRB-D with correction should report A, F and P_eff
  together.
- The repo's |P(0) - P(1)| averaged over sequences is nonlinear, but for each
  circuit the identity holds for the signed polarization.

### 3.3 Q1: which decoders and policies (conjecture unless labelled)

1. **Policy family.** Threshold the posterior of the logical observable
   actually read out (X_L at the end, with errors propagated through the
   Clifford sequence) at t in [1/2, 1]:
   - t = 1/2 is full ML correction;
   - t near 1 is detection-like, but still keeps shots whose syndrome is
     confidently explained;
   - for each circuit, this family is the Pareto-optimal accept rule
     [English2024, Seip2026].

   The optimality is **from literature**; applying it here is a conjecture.
2. **Exclusive decoders at d = 2.** With integer weights (code capacity), Smith
   et al.'s tolerance family [Smith2024] collapses to three policies (**verified
   analytically** from 1 - Δ/2 > c with Δ in {0, 1, 2}):
   - c < 1/2: pure detection;
   - 1/2 <= c < 1: reject gap 0 only;
   - c = 1: MWPM.

   With circuit-level log-likelihood weights the gap is continuous, and so is
   the family.
3. **Exact posteriors are affordable here.** The code has one logical qubit,
   4 detectors per round and at most 23 rounds. Exact coset probabilities
   should be computable by contracting the detector error model round by round,
   or by brute force per distinct syndrome with caching, since most shots share
   a few syndromes. The true Pareto frontier can then be computed and used to
   benchmark the PyMatching complementary gap. Per [Gidney2023yoked], the gap is
   obtained by decoding once with each logical value forced, and MWPM gaps run
   about 10% overconfident.
4. **Where the extra acceptance should come from under SI1000:**
   - Measurement flips in forward rounds. X_ERROR(5p) is the largest single
     SI1000 term. A flip produces a time-like pair of detection events that a
     decoder attributes to measurement with a large gap. LRB-D with unif=1
     rejects these shots; the posterior rule keeps them.
   - Y-type faults on qubits 0, 2, 3, 4 and any fault on qubit 1, as in
     Table 3.
   - Faults whose propagated Pauli commutes with the final X_L, if the policy
     rejects on final Z-check outcomes. These shots are rejected for nothing.

   Ambiguous Z-like faults in the last check round and in the check-free
   inverse half cannot be rescued, because no later information exists.
5. **Hook errors and the CNOT order** break the code-capacity equal-enumerator
   tie. Some (1,0)/(0,1)-type histories may then have posteriors away from 1/2
   under the actual detector error model. Measure the posterior distribution
   numerically before assuming "coin flip".
6. **Reporting.** For each (p, m, policy), report A(m), the conditional F(m) and
   P_eff(m):
   - fit A(m) and F(m) separately, as heralded RB does [Levine2023];
   - compare with full correction through P_eff [CombesFerrie2015];
   - from the same shots, estimate correctable vs uncorrectable error by
     decoding with and without recovery [Combes2017].
7. **No scalable postselection at d = 2.** The results that make postselection
   asymptotically cheap (abort rate falling with distance, b >= 2 gains;
   [Chen2025, Seip2026, Smith2024]) need growing d. At d = 2, any rule stricter
   than "reject only g = 1/2" pays Θ(p) rejection per round, so A(m) ~
   exp(-Θ(p) m). This matches the exponential kept-fraction decay on hardware
   [Xin2025] and the about 2% per round discard in the tesseract experiment
   [Reichardt2024].

### 3.4 Q2: non-Markovianity (conjecture)

1. **Why LRB-D has memory.** In LRB-D nothing is corrected, so a data fault
   persists in the Pauli frame of the code space. The accept decision at a later
   selected round therefore depends on the whole accumulated fault history. The
   accepted-shot logical process is a renormalised, non-trace-preserving map
   that does not factor into per-round channels. Plausibly this produces the
   non-exponential const=1 decay: pairs of faults that combine into a weight-2
   logical, or into a trivial syndrome, grow like (m p)^2 before saturating.
2. **What correction changes.** Per-round correction (a Pauli-frame update)
   erases that memory each round and leaves an approximately i.i.d. per-round
   logical error, so the decay should be close to exponential. A posterior rule
   that decides window by window should stay close to Markovian. A single global
   accept decision over the whole history can bring the history dependence back,
   since windows are correlated [Dinca2025].
3. **Use the heralded-RB template.** With per-round heralding, A(m) and F(m) are
   each a single exponential [Levine2023]. Sparse const=k checks mix rounds and
   should not be expected to give clean exponentials.
4. **First-cycle transient.** Watch for one; [Kumar2026] reports one in
   detection cycles. Here the first check round follows an ideal encoding, so it
   is not obvious one will appear.

### 3.5 Q3: hardware (conjecture)

1. **The confidence signal is cheap.** For this code, a posterior lookup table,
   two MWPM decodes, or a cluster proxy can all run in real time. FPGA
   confidence scores cost negligible latency [Tahaab2026]; sliding-window early
   abort exists for general codes [Lee2025].
2. **Superconducting (IQM-like, measurement-dominated as in SI1000).** Feeding
   analog readout into the edge weights [Pattison2021, Ali2024, Hanisch2024]
   should sharpen exactly the time-like posteriors the policy relies on. Leakage
   flags can enter the same posterior [Hanisch2024, Xin2025].
3. **Trapped ions.** "Correct what is flagged, reject what is uncorrectable" is
   already demonstrated at small distance [Reichardt2024]. It is the closest
   hardware analogue of the posterior-threshold policy.
4. **Neutral atoms.** The sliding threshold on correlated-decoding weight
   [Bluvstein2023] is a gap-threshold policy. Atom-loss or erasure flags would
   enter the same posterior.
5. **Sustainable default.** Any non-trivial rejection rule at d = 2 makes A(m)
   decay exponentially. The sustainable default is therefore to always correct
   and use the soft output to weight or flag shots: per-shot MLE weighting
   [Dinca2025], posterior-based PEC or ZNE [Zhou2025], or VEC [Shtanko2026].
   Light postselection fits only short sequences or the final check-free
   segment.

## 4. Seed corrections

- All eight seed arXiv IDs resolve to the stated papers; no ID was wrong.
- Gidney, Shutty, Jones (arXiv:2409.17595): the full title is "Magic state
  cultivation: growing T states as cheap as CNOT gates".
- Bombín et al. (arXiv:2212.00813): the exact title is "Fault-tolerant
  Post-Selection for Low Overhead Magic State Preparation". It appeared in PRX
  Quantum 5, 010302 (2024); the arXiv version is 2022.
- Meister et al. (arXiv:2405.07433): the full author list is Nadine Meister,
  Christopher A. Pattison, John Preskill.
- English et al. (arXiv:2410.07598): the authors are Lucas H. English, Dominic
  J. Williamson, Stephen D. Bartlett. It appeared in PRL 135, 120603 (2025),
  DOI 10.1103/nh49-52y2.
- Its Theorem 1 (optimal postselection), stated for abort sets of equal
  cardinality, is called wrong by Seip2026 (Remark G.2). Use the abort
  probability as the constraint.
- Prabhu and Reichardt (arXiv:2112.03785): arXiv 2021, published in PRA 110,
  012419 (2024).
- Smith, Brown, Bartlett (arXiv:2405.03766): published in Commun. Phys. 7, 386
  (2024).
- One web-search summary attributed exclusive decoders to "Ahmed et al.". That
  is wrong; the authors are Smith, Brown and Bartlett.
- Yoked surface codes (arXiv:2312.04522) does not use the complementary gap to
  postselect. It passes the gaps as edge weights to the outer yoke decoder. The
  postselection uses are in Bombin2022 and in cultivation's escape stage.
- The hardware cultivation experiment (Rosenfeld2025) postselected on "no
  detection event at all", not on the complementary gap.
- Not seeds, but search summaries made claims I could not verify, so they are
  not used: a "b ~ 2.7 for logical-gap postselection" attributed to
  arXiv:2603.08697, and a "GNN logit beats MWPM gap" claim seen only on a
  scirate listing.

## 5. Open questions

1. Under SI1000 circuit noise, what fraction of LRB-D rejections are time-like
   (measurement flips in forward rounds) and could be kept at negligible cost by
   a posterior rule?
2. At d = 2, ties and near-ties are common. Does an exact ML posterior give a
   materially better Pareto frontier than the PyMatching complementary gap, and
   how badly does MWPM tie-breaking bias the "gap = 0" class?
3. For postselected LRB, which decay is the meaningful "logical error per
   Clifford": the A(m) rate, the conditional F(m) rate, or the P_eff(m) rate?
   Which one is robust to state-preparation-and-measurement errors and well
   described by a Markovian model?
4. Does per-round correction (or window-wise posterior postselection) remove the
   non-exponential const=1 behaviour? Does a single global accept decision over
   the whole history bring it back?
5. The inverse half has no checks. Would one extra check round before the
   terminal round (or splitting the inverse half) turn its code-capacity-like
   ambiguity into time-like information?
6. Calibration: MWPM gaps are about 10% overconfident [Gidney2023yoked], and MLE
   weighting is calibration-sensitive [Dinca2025]. Can LRB data, where the ideal
   outcome is known, calibrate the posterior itself?
7. Randomised acceptance at the threshold (Neyman-Pearson style) would fill in
   the staircase frontier [Seip2026]. Is it needed for clean Pareto curves at
   d = 2, where few distinct syndromes per round exist?
8. Hardware: does analog-readout soft information cut the dominant SI1000
   measurement-flip term enough to change the ranking of policies?

## 6. Search log (19 queries)

1. complementary gap postselection surface code soft output decoder
2. optimal post-selection quantum error correction likelihood ratio logical failure probability
3. soft-output decoder logical error rate estimate per shot 2025 arXiv
4. decoder confidence error mitigation logical qubits discard low-confidence shots
5. Forney erasure decoding reject option likelihood ratio threshold optimal tradeoff error versus erasure
6. sliding scale error detection correction postselection fraction of shots logical qubits neutral atom experiment
7. virtual error cancellation logical circuits decoder disagreement signed weights Google arXiv
8. abort threshold postselected surface code acceptance rate versus logical error Pareto frontier
9. "virtual error cancellation" arXiv quantum error correction preprint
10. analog readout information decoder superconducting logical qubit reduces logical error rate Ali Marques
11. Chow optimum recognition error and reject tradeoff IEEE Transactions on Information Theory 1970
12. error detection versus error correction small distance-2 code postselection logical randomized benchmarking partial postselection decoder
13. (arxiv.org only) "error cancellation" logical circuits decoder agreement real-time decoder offline decoder signed weights surface code distance-7 2025
14. randomized benchmarking with postselection error detection non-exponential decay non-trace-preserving subnormalized channel logical qubit
15. experimental demonstration magic state cultivation superconducting processor complementary gap postselection 2025 2026
16. neural network decoder calibrated probability output confidence postselection surface code experiment AlphaQubit soft output
17. randomized benchmarking erasure detection postselected fidelity and erasure rate reported separately dual-rail qubit heralded errors
18. Pareto frontier acceptance rate logical error rate postselection policy small code d=2 surface code repeated syndrome measurement measurement errors time-like
19. tesseract code trapped ion demonstration error correction and detection combined postselection Reichardt Quantinuum Microsoft 2024

## Appendix: enumeration used in section 3.1

Plain Python 3; runs in well under a second. This is the depolarizing version
behind Table 3. Tables 1 and 2 use the same loop restricted to Z-type errors.

```python
import itertools
def table(p):
    tab = {}
    for paulis in itertools.product("IXYZ", repeat=5):
        x = [1 if c in "XY" else 0 for c in paulis]
        z = [1 if c in "ZY" else 0 for c in paulis]
        w = sum(c != "I" for c in paulis)
        pr = (p/3)**w * (1-p)**(5-w)
        s = ((x[0]+x[1]+x[3]) % 2, (x[1]+x[2]+x[4]) % 2,   # Z checks S_Z1, S_Z2
             (z[0]+z[1]+z[2]) % 2, (z[1]+z[3]+z[4]) % 2)   # X checks S_X1, S_X2
        cz = (z[0] + z[3]) % 2          # Z part anticommutes with X_L = X0 X3
        tab.setdefault(s, [0.0, 0.0])[cz] += pr
    return tab
def evaluate(tab, accept):
    A = good = 0.0
    for s, (p0, p1) in tab.items():
        if accept(s, p0, p1):
            A += p0 + p1; good += max(p0, p1)
    err = 1 - good / A
    return A, err, A * (1 - 2 * err)    # acceptance, conditional error, A*F
tab = table(1e-3)
for name, rule in [("detect-all", lambda s, a, b: s == (0, 0, 0, 0)),
                   ("detect-X",   lambda s, a, b: s[2:] == (0, 0)),
                   ("post>=0.6",  lambda s, a, b: max(a, b) / (a + b) >= 0.6),
                   ("correct",    lambda s, a, b: True)]:
    print(name, evaluate(tab, rule))
```
