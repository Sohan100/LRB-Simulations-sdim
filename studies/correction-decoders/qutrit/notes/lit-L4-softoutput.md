# L4 literature notes: soft-output decoding, postselection on decoder confidence, optimal reject/erasure rules

Agent: L4 (literature). Written 2026-10-28. Read together with `00-brief.md`.

Label legend. Every claim below carries one of these tags:

- **[lit]** from literature: stated in a paper fetched in this task (URL in the paper's citation line). For the two IEEE papers whose full text could not be rendered (Forney 1968, Chow 1970), the statement is attributed to the secondary source that was read, and that source is named.
- **[ana]** verified analytically: derived in these notes (derivation given or sketched).
- **[num]** verified numerically: computed in this task by first-order (single-fault) enumeration of the compact DEM of actual baseline circuits. Script in Appendix A; runtime under 2 minutes on one login-node core.
- **[conj]** conjecture.

---

## 0. Summary

### 0.1 The optimality statement

1. **[lit]** Chow's rule. For a classifier with a reject option, under a loss of 0 for a correct decision, 1 for an error and lambda for a rejection, the Bayes-optimal rule outputs the class with the largest posterior and rejects when that largest posterior is below 1 - lambda. Sources: Combes and Ferrie 2015, Eq. (14) and Sec. II, citing Chow 1957/1970; Franc, Prusa and Voracek 2023, Problem 1.
2. **[lit]** The same rule solves the "bounded-coverage" problem: minimize the error rate among accepted inputs subject to an acceptance rate of at least omega. The optimal strategy is the Bayes classifier plus a selection function that accepts when the conditional risk is below a threshold, rejects above it, and randomizes exactly at the threshold (Franc, Prusa and Voracek 2023, Theorem 6). This is the precise form of "accept iff posterior confidence is above a threshold maximizes accuracy at a fixed reject rate".
3. **[lit]** Chow 1970 relation, as restated by Combes and Ferrie (Eqs. 35 and 36). Along the optimal error/reject curve, P(error) = -integral_0^lambda lambda' dP(reject | lambda'). Equivalently, the marginal exchange rate between error and rejection equals the rejection threshold. This is how "the error rate can be directly evaluated from the reject function" (Chow 1970 abstract, IBM Research page).
4. **[lit]** Forney's optimum erasure/list rule (as stated by Hof, Sason and Shamai 2010, Definition 2 and Remark 2). Decide message m iff Pr(y; x_m) / sum_{m' != m} Pr(y; x_m') >= e^{nT}. Equivalently, decide m iff its posterior Pr(x_m | y) >= e^{nT}/(1 + e^{nT}), and erase otherwise. With T > 0 the decision regions are disjoint (erasure mode); with T < 0 the rule outputs variable-size lists. No other rule has both a lower total error probability (erasure plus undetected error) and a lower undetected-error probability (Proposition 1). Forney proved this with the Neyman-Pearson lemma (Merhav 2008). Forney also proposed a suboptimal "likelihood ratio" rule that compares the best codeword only with the second best (HSS10, Definition 3). That rule is the classical analogue of the complementary gap.
5. **[lit]** Quantum version (English, Williamson and Bartlett, PRL 2025, Theorem 1). Aborting whenever maximum-likelihood decoding returns a maximum coset probability below gamma is optimal for the post-selected success probability. Their Supplement S7 adds that the theorem holds "as long as the error channels are Markovian", because the coset probabilities must be computable.

### 0.2 What the complementary or logical gap is and how it is computed

1. **[lit]** Definition (Gidney et al. 2023). The complementary gap is the log-likelihood ratio between the decoder's best hypothesis and its best hypothesis conditioned on the complementary logical outcome.
2. **[lit]** Bombin et al. (2024) define the signed logical gap as Delta = w(l_wrong) - w(l_correct), with edge weights ln((1 - p_e)/p_e) and erased edges set to weight 0. Only the unsigned |Delta| is observable, and the encoding error rate is the integral of P(Delta) over Delta < 0.
3. **[lit]** Computation with a matching decoder:
   - Hutter, Wootton and Loss 2014 add a virtual anyon joined by zero-weight edges to the boundary anyons, which forces a parity change and returns the minimum-weight matching of the other class.
   - Gidney et al. add a detector joined to all boundary edges of one side and decode with it on and off.
   - Cultivation (Gidney, Shutty and Jones 2024): "adding a detector equivalent to an observable ... decoding once with it forced on and once with it forced off".
4. **[lit]** Cluster-based alternative (Meister, Pattison and Preskill 2024). The soft output is the shortest path between inequivalent boundaries after every edge inside a decoder cluster is set to weight 0 (Dijkstra, O(|E| + |V| log |V|)).
   - For union-find on the repetition code the soft output equals the log-likelihood ratio exactly (Theorem 10).
   - For MWPM on the surface code it lower-bounds the log ratio of the most likely errors in the two classes (Theorem 13).
5. **[ana]** For the Z_3 logical of the `[[5,1,2]]_3` code there are three classes. Let g1 <= g2 be the log-posterior gaps from the best class to the other two. The confidence is q* = 1/(1 + e^-g1 + e^-g2). Forney's optimal rule thresholds -log(e^-g1 + e^-g2). The two-class complementary gap g1 is Forney's suboptimal rule. The two statistics differ by at most log 2, so the rules can disagree only on syndromes whose g1 lies within log 2 of the cutoff (Sec. 5.3).

### 0.3 Reported acceptance versus logical-error trade-offs [lit]

- **Bombin et al.** At 60% of the bulk threshold, logical-gap postselection lowers the encoding error rate by about 15x at a relative overhead below 2. The radial gap rule reaches breakeven at overhead 1.78, about 23 times lower than syndrome counting. At the bulk threshold, breakeven costs about 17x overhead.
- **Smith, Brown and Bartlett.** Depolarizing code-capacity thresholds rise from 15% (MWPM) to 50% (zero tolerance), and reach 32(1)% with measurement errors. Below threshold the logical failure rate improves up to quadratically, from p^(d/2) to p^d. Abort probabilities have their own threshold for tolerance c > 0. For 15-to-1 distillation they report 75% fewer qubits and 60% less spacetime volume.
- **Meister, Pattison and Preskill.**
  - Repetition code, n = 12, p = 0.05: rejecting 0.2% of runs lowers the logical error from about 1e-5 to 4e-10.
  - d = 9 surface code, phenomenological noise, p = 0.005: discarding about 5e-4 of runs lowers it from 3.0(5)e-5 to at most 2e-6.
  - Corollary 15: discarding at most half the samples cuts the repetition-code length needed by nearly a factor of 4.
- **Cultivation (Gidney, Shutty and Jones).** At p = 1e-3, end to end: 3e-6 with 80% discard at d1 = 3, and 2e-9 with 99% discard at d1 = 5. At p = 5e-4, d1 = 5: 4e-11 with 90% discard. The early stages discard on any detection event; only the escape stage uses a gap cutoff.
- **Chen et al.** Postselecting on |DeltaF| >= s* d gives an abort rate of exp(-I(s*) d) and a conditional failure rate of exp(-I(-s*) d). The maximal effective-distance gain b is at least 2 under a large-deviation principle. Fitted values: b = 3.1(1) for the toric code with optimal decoding, b of about 2.87 for MWPM with circuit-level noise at p = 0.005, and b of about 3.44 for MWPM with perfect measurements. Postselecting a whole circuit gives an accuracy gain but no overhead reduction.
- **Lee, English and Bartlett.** On the [[144,12,12]] BB code, cluster-statistics postselection cuts the logical error about 1000-fold at 1% abort for p = 0.1%, and at 19% abort for p = 0.3%.
- **Staples, Fu and Thompson.** Partial-gap postselection of resource states gives a 4x lower overhead per teleported gate.
- **Prabhu and Reichardt.** Correcting one fault and rejecting two makes logical errors O(p^3) and rejections O(p^2). Their d = 4 code accumulates 25x less error than d = 5.
- **Rosenfeld et al. (hardware).** Full detector postselection during cultivation keeps 8% of attempts and cuts the error 40-fold, to a fidelity of 0.9999(1).

### 0.4 Consequences for this study (details in Sec. 5)

1. **[ana]** For one LRB circuit, the decoder that outputs the MAP logical shift and accepts iff q*(D) >= t maximizes the accepted-shot logical accuracy at every fixed acceptance rate. It may use all executed check rounds, so it dominates every `unif k` and `const c` LRB-D policy at that policy's own acceptance rate.
2. **[ana]** Every channel in the generalized SI1000 model as implemented is symmetric under v -> -v, and the DEM is linear, so P(D, L) = P(-D, -L). For any symmetric rule (LRB-D, MAP with symmetric tie-breaking, the Chow rule), the accepted residual satisfies Q(1) = Q(2). The LRB fidelity is then exactly F = (3 Q(0) - 1)/2, an affine function of the accepted accuracy. Chow optimality therefore transfers exactly to the per-circuit LRB fidelity.
3. **[ana]** Suppose rejected shots are counted as fully depolarized outcomes, so the figure of merit is the unconditional fidelity kappa * F_acc. Maximizing it is Chow's problem with rejection cost lambda = 2/3, relative to an error cost of 1. Because q* >= 1/3 always holds for three classes, the optimal rule then never rejects. Whether postselection "helps" therefore depends entirely on how rejections are costed. LRB-D reports a conditional fidelity, so any comparison with physical RB must state that cost (Combes and Ferrie make the same point).
4. **[num]** First-order numbers for one Clifford sequence, scaled per unit p:
   - The LRB-D accepted infidelity floor comes from syndrome-free faults in the last check round. Their first-order logical rate is u = 4.95p, the same at every depth m checked (0, 2, 4, 10, 22). This reproduces F = 0.99257 at m = 0, p = 1e-3; the baseline is 0.992503.
   - The Chow rule that accepts only first-order-unambiguous patterns (q* = 1 at first order) keeps the same first-order accepted error. For m >= 2 it rejects only 35-41% of the mass that `unif 1` rejects (66% at m = 0): 884p versus 2516p at m = 22. That means about 5x more accepted shots at m = 22, p = 1e-3 in a first-order extrapolation, which is not reliable there (Sec. 5.5).
   - MAP correction without rejection has a first-order error of about 10.7p per unit depth. The physical-qutrit comparison in the baseline is about 3.8p per unit depth.
5. **[ana]** In code capacity, the distance-2 ambiguity affects only pure X-type or pure Z-type single-qutrit errors on qutrits {0, 2, 3, 4}: 16 of the 40 single-qutrit Weyl errors. Mixed X^a Z^b errors and all errors on qutrit 1 have unique weight-1 syndromes. This is the qutrit analogue of the "breadcrumbs" effect in Hutter, Wootton and Loss.

---

## 1. Classical decision theory with a reject (erasure) option

### 1.1 Chow 1970 (with Chow 1957)

Citations:
- C. K. Chow, "On optimum recognition error and reject tradeoff", IEEE Transactions on Information Theory, vol. 16, no. 1, pp. 41-46, January 1970. DOI 10.1109/TIT.1970.1054406. Fetched: https://api.crossref.org/works/10.1109/TIT.1970.1054406 (metadata) and https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff (abstract). IEEE Xplore (https://ieeexplore.ieee.org/document/1054406) returned an empty page.
- C. K. Chow, "An optimum character recognition system using decision functions", IRE Transactions on Electronic Computers, vol. EC-6, no. 4, pp. 247-254, 1957. DOI 10.1109/TEC.1957.5222035. Fetched (metadata only): https://api.crossref.org/works?query.bibliographic=Chow+An+optimum+character+recognition+system+using+decision+functions+1957&rows=3

What could be read [lit]:
- Chow 1970 abstract (IBM Research page): "The performance of a pattern recognition system is characterized by its error and reject tradeoff." In paraphrase, the paper derives an optimum rejection rule and a general relation between error and reject probabilities, gives properties of the trade-off, and shows the error rate can be evaluated directly from the reject function. The examples use normal and uniform distributions. The research.ibm.com page lists C. K. Chow, IEEE Trans. Inf. Theory, 1970; the volume, issue, pages and DOI above come from Crossref.
- The exact statements are taken from the two secondary sources in Secs. 1.3 and 1.4, which cite Chow 1957/1970:
  - The rule: report the class with the largest posterior if that posterior is at least 1 - lambda, otherwise reject (Combes and Ferrie, Eq. 14).
  - With general costs, the threshold is 1 - lambda_R/lambda_E (Combes and Ferrie, Eq. 47).
  - The minimum risk is R* = P(E) + lambda P(R) = integral_0^lambda P(R | lambda') dlambda' (Eqs. 35a, 35b).
  - The error probability is P(E) = -integral_0^lambda lambda' dP(R | lambda') (Eq. 36). Their Fig. 7 caption states that the slope of the error-reject curve equals the threshold, dP(E)/dP(R) = lambda in their sign convention.
  - Franc et al. add that Chow proved monotonicity of the error and reject rates in the reject cost.

How it bears on this study:
- [ana] Map Chow's "pattern x" to the full detector history D of one shot, "class" to the logical shift L in Z_3, and the posterior to P(L | D) from the DEM. The rule is then exactly the soft-output postselection decoder of Sec. 5.1.
- [ana] Chow's slope relation gives a check on any implementation. Near the operating threshold, the shots additionally accepted when the threshold is loosened by d(lambda) must have conditional error approximately lambda. The first-order numbers of Sec. 5.5 satisfy this: between eps = 0 and eps = 0.05 at m = 10, the slope dE/dR is about -0.025.

### 1.2 Forney 1968

Citation: G. D. Forney (Crossref lists "G. Forney"), "Exponential error bounds for erasure, list, and decision feedback schemes", IEEE Transactions on Information Theory, vol. 14, no. 2, pp. 206-220, March 1968. DOI 10.1109/TIT.1968.1054129. Fetched (metadata only): https://api.crossref.org/works/10.1109/TIT.1968.1054129. IEEE Xplore (https://ieeexplore.ieee.org/document/1054129) returned an empty page, so the content below comes from Sec. 1.5.

Extracted facts [lit] (via Hof, Sason and Shamai 2010 and Merhav 2008):
- Two error events (Merhav, Sec. 2, Eqs. 6-8):
  - E1, "not making the right decision", is the disjoint union of erasure and undetected error.
  - E2 is the undetected error event.
- Optimum generalized decoding rule (HSS10, Definition 2, Eq. 5): Lambda_m = { y : Pr(y; x_m) / sum_{m' != m} Pr(y; x_m') >= e^{nT} }.
  - Remark 2 rewrites this as Pr(x_m | y) >= e^{nT}/(1 + e^{nT}). At T = 0 the decision agrees with MAP.
  - Remark 3: T controls the trade-off between erasures and undetected errors (or list size and list error). T > 0 makes the decision regions disjoint, i.e. erasure decoding. The variable-size list case is T < 0 (HSS10, text after Definition 3 and Sec. 3).
- Optimality: HSS10 Proposition 1 states that "there is no other decoding rule that simultaneously gives a lower error probability and a lower undetected error probability". Merhav: "Forney shows, using the Neyman-Pearson theorem, that the best tradeoff between" Pr(E1) and Pr(E2) is attained by this rule.
- Random-coding exponents (Merhav, Eqs. 1-2): E1(R, T) = max_{0 <= s <= rho <= 1} [E0(s, rho) - rho R - s T] and E2(R, T) = E1(R, T) + T.
- Suboptimal rule suggested by Forney (HSS10, Definition 3, "LR decoding"): decide m iff Pr(y; x_m)/Pr(y; x_m2) >= e^{nT}, where m2 is the second most probable codeword.
- HSS10 notes that erasure options are useful with feedback (ARQ) and in concatenation.

How it bears on this study:
- [ana] Forney's rule is Chow's rule written as posterior odds, P(best)/P(all others) >= e^T. For our Z_3 logical, "all others" means two classes.
- [ana] The complementary gap that matching decoders compute, the log-ratio of best to second best class, is Forney's suboptimal LR rule. For a qubit observable (two classes) the two rules coincide; for a qutrit they differ (Sec. 5.3).
- [ana] Forney's "erasure" is what LRB-D calls rejection. His exponent relation E2 = E1 + T means Pr(E2)/Pr(E1) ~ e^{-nT}: each nat of the threshold exponent nT buys a factor e^-1 in undetected error relative to the total failure probability. In our setting, raising the confidence threshold lowers accepted error at the cost of acceptance, as in Sec. 5.5.
- [conj] Forney's motivating use, erasures passed to an outer code or a repeat request, is the sustainable alternative to discarding shots (Secs. 2.4 and 3.1).

### 1.3 Franc, Prusa and Voracek 2023 (equivalence of reject-option formulations)

Citation: Vojtech Franc, Daniel Prusa, Vaclav Voracek, "Optimal Strategies for Reject Option Classifiers", Journal of Machine Learning Research 24(11):1-49, 2023. arXiv:2101.12523 (v1 29 Jan 2021). Fetched: https://arxiv.org/abs/2101.12523, https://ar5iv.labs.arxiv.org/html/2101.12523, https://www.jmlr.org/beta/papers/v24/21-0048.html

Extracted facts [lit]:
- Problem 1 is Chow's cost-based model: minimize the expected loss, which is the classification loss when predicting and the reject cost epsilon when abstaining. The optimal strategy is the Bayes classifier h_B(x) in argmin_yhat sum_y p(y|x) l(y, yhat), with selection c_B(x) = 1 if r*(x) < epsilon, tau if r*(x) = epsilon, and 0 if r*(x) > epsilon. Here r*(x) is the minimal conditional risk and tau in [0, 1] is arbitrary.
- Problem 4 is the bounded-coverage model: minimize the selective risk R_S(h, c) subject to coverage phi(c) >= omega. Theorem 4: the Bayes classifier is optimal. Theorem 6: the optimal selection function is 1 if r(x) < beta, kappa if r(x) = beta, and 0 if r(x) > beta, so it randomizes only at the threshold.
- Problem 2 is the bounded-improvement model (maximize coverage subject to selective risk at most lambda). Theorems 1 and 3 give the same form.
- Definition 1: any score that preserves the ordering of the conditional risk r(x) is a "proper uncertainty score". Thresholding it gives the same optimal selection.

How it bears on this study:
- [ana] Theorem 6 is the exact optimality statement asked for: at fixed acceptance (coverage), the error among accepted shots is minimized by accepting the shots with the smallest conditional risk 1 - q*(D), randomizing at the boundary. With a discrete syndrome space, a single acceptance target generally needs a randomized boundary class.
- [ana] Definition 1 means any monotone function of q*(D) is an equally good statistic: the gap -log(e^-g1 + e^-g2), the posterior odds, or dB values. The two-class gap g1 is not a monotone function of q* for three classes, so it is not proper in general (Sec. 5.3).

### 1.4 Combes and Ferrie 2015 (cost of postselection in quantum decision theory)

Citation: Joshua Combes, Christopher Ferrie, "Cost of postselection in decision theory", Physical Review A 92, 022117 (2015). arXiv:1504.03740 (v1 14 Apr 2015, v2 30 Aug 2015). DOI 10.1103/PhysRevA.92.022117. Fetched: https://arxiv.org/abs/1504.03740, https://ar5iv.labs.arxiv.org/html/1504.03740

Extracted facts [lit]:
- Abstract: postselection is "the process of discarding outcomes from statistical trials that are not the event one desires". Without an exact cost for discarding, one may wrongly conclude that discarding all data is optimal.
- With a prespecified cost of discarding data, the optimal measurement interpolates between unambiguous state discrimination (zero cost) and the minimum-error (Helstrom) measurement (maximal cost).
- Loss ordering (Eq. 4): correct < reject < wrong. Threshold rule (Eq. 14): report the most probable hypothesis if its posterior is at least 1 - lambda, otherwise reject.
- Risk identities (Eqs. 35-37) as quoted in Sec. 1.1. These hold only when the optimal rule and measurement are used.
- Postselected strategies other than unambiguous discrimination are optimal for intermediate costs.

How it bears on this study:
- [ana] The LRB-D figure of merit, fidelity of accepted shots only, corresponds to an unspecified cost of rejection. Sec. 5.2 makes this explicit for the LRB fidelity: if a rejected shot counts as a fully depolarized outcome, the cost is lambda = 2/3 and the optimal rule never rejects.
- [conj] Recommendation for the study: always report (F_acc, kappa) together. If one scalar is required, report the risk curve R*(lambda) or kappa F_acc and state lambda.

### 1.5 Secondary sources used for Forney's statements

- Eran Hof, Igal Sason, Shlomo Shamai, "Performance Bounds for Erasure, List, and Decision Feedback Schemes With Linear Block Codes", IEEE Transactions on Information Theory 56(8):3754-3778, 2010. DOI 10.1109/TIT.2010.2050797. Fetched: https://webee.technion.ac.il/people/sason/bounds_list_erasure_feedback.pdf (preprint "to appear in IEEE Trans. IT, August 2010"). The DOI was confirmed via https://api.crossref.org/works?query.bibliographic=Performance+Bounds+for+Erasure+List+and+Decision+Feedback+Schemes+with+Linear+Block+Codes+Hof+Sason+Shamai&rows=3. Used for Definitions 2 and 3, Remarks 2 and 3, and Proposition 1 quoted above. [lit]
- Neri Merhav, "Error Exponents of Erasure/List Decoding Revisited via Moments of Distance Enumerators", arXiv:0711.2501 (v1 15 Nov 2007). Related DOI 10.1109/TIT.2008.929004. Fetched: https://arxiv.org/abs/0711.2501, https://ar5iv.labs.arxiv.org/html/0711.2501. Used for the event definitions, the Neyman-Pearson optimality statement and the exponents E1, E2. Merhav's own result is a bound at least as tight as Forney's that needs one optimization parameter under symmetry conditions. [lit]

---

## 2. Soft information and soft-output decoders

### 2.1 Hutter, Wootton and Loss 2014: per-class matching, the origin of the complementary-gap construction

Citation: Adrian Hutter, James R. Wootton, Daniel Loss, "Efficient Markov chain Monte Carlo algorithm for the surface code", Physical Review A 89, 022326 (2014), published 18 Feb 2014. The arXiv title is "An efficient Markov chain Monte Carlo algorithm for the surface code", arXiv:1302.2669 (v1 11 Feb 2013, v2 8 Aug 2013). DOI 10.1103/PhysRevA.89.022326. Fetched: https://arxiv.org/abs/1302.2669, https://ar5iv.labs.arxiv.org/html/1302.2669, https://arxiv.org/pdf/1302.2669v2 (text extracted locally), https://api.crossref.org/works/10.1103/PhysRevA.89.022326

Extracted facts [lit]:
- Sec. III, "Enhanced MWPM". Standard MWPM "outputs only a single matching for a single equivalence class". To fix this, "we will force the decoder to output the minimum weight matching for each equivalence class, which requires us to enforce changes in the parities of error numbers along lines across the code". This is done "by adding one further virtual anyon on the top and bottom or the left and right boundary ... connecting it with zero-weight edges to all virtual anyons already present at this boundary". Combining the 2 x 2 chains gives four hypotheses. The decoder picks the most likely one under the true correlated (depolarizing) model.
- Sec. IV. The relative probability of class E given anyon configuration A is Z_E(beta) = sum over error chains in E compatible with A of e^{-beta n}, where n is the number of single-qubit errors and beta = -log[(p/3)/(1 - p)] for depolarizing noise. "The goal is to find the equivalence class E with maximal Z_E". The Metropolis chains stay inside one class, starting from "the minimum weight error configuration from each equivalence class". The single-temperature algorithm compares the average error count <n> between classes.
- Below threshold, the "distinguishability" (difference of <n> between the correct and the other classes) grows linearly with L. "The probability of all equivalence classes but the most likely one decreases exponentially with L".
- Results:
  - Depolarizing threshold of the single-temperature algorithm between 15% and 16%; theoretical bound 18.9%.
  - Runtime O(L^4) serial and O(L^2) parallel, against O(L^2) for MWPM.
  - At p = 13%, L >= 6 suffices instead of L >= 15 (61 versus 421 data qubits).
  - As p -> 0, decoding with the bit/phase correlation gives a logical error lower by O(2^{(L-1)/2}). The reason given is that Y errors leave "a trail of breadcrumbs" showing that the larger-weight matching is correct.
- Terminology: the paper does not use the words "complementary gap" or "confidence". Gidney et al. 2023 ("complementary gaps [HWL14; Bom+22]"), Meister et al. 2024 ("the complementary gap method, has been formulated previously [3-5]", with [3] = HWL14) and Smith et al. 2024 (Ref. [66]) all cite it for the per-class minimum-weight construction.

How it bears on this study:
- [ana] The per-class computation generalizes to the qutrit code: compute the best explanation (or the full class probability) for each of the three values of the logical shift. Our DEM is small enough that exact class probabilities are available, which includes the entropic (degeneracy) terms HWL14 stress. Approximating a class by its single most likely error would drop multiplicity factors.
- [ana] The breadcrumbs mechanism has a direct counterpart here (Sec. 5.4). Under depolarizing noise, a mixed X^a Z^b error on any data qutrit has a unique weight-1 syndrome even though the code has distance 2. Pure X or pure Z errors on qutrits {0, 2, 3, 4} are ambiguous.

### 2.2 Pattison, Beverland, da Silva and Delfosse 2021: soft measurement information

Citation: Christopher A. Pattison, Michael E. Beverland, Marcus P. da Silva, Nicolas Delfosse, "Improved quantum error correction using soft information", arXiv:2107.13589 (v1 28 Jul 2021). No journal reference listed on arXiv. Fetched: https://arxiv.org/abs/2107.13589, https://ar5iv.labs.arxiv.org/html/2107.13589, https://arxiv.org/pdf/2107.13589v1 (text extracted locally to verify the numbers)

Extracted facts [lit]:
- The usual model flips binary measurement outcomes, but experiments record richer analog data.
- Gaussian soft noise: f0 = N(+1, sigma^2), f1 = N(-1, sigma^2). The hard outcome is the maximum-likelihood assignment, and a "soft flip" occurs when that assignment disagrees with the ideal outcome.
- In the soft decoders (MWPM and union-find), soft edges get weight w = -log L(mu), where L is the likelihood ratio of the alternative to the inferred outcome given the analog value mu (Eq. 7). Hard edges keep -log(p_e/(1 - p_e)). The soft MWPM returns a most likely fault set (Theorem 4.1).
- Thresholds (PDF text, Sec. 5):
  - Phenomenological noise: soft union-find 3.665(2)% against hard union-find 2.637(1)% and the best known hard threshold 2.93(2)%. This is the "25% higher" in the abstract.
  - Circuit noise: 0.727(1)% against 0.702(1)%, and 0.5824(6)% against 0.4991(5)% when the hardened measurement error is 10p.
- With an amplitude-damping measurement model, a five-fold increase in measurement time can raise the d = 19 logical error rate a thousand-fold (abstract). The measurement time that minimizes physical error is not the one that minimizes logical error.
- The paper does not output a decoder confidence and does not postselect.

How it bears on this study:
- [ana] For symmetric Gaussian readout, L = exp(-2|mu|/sigma^2), so a soft measurement edge has weight 2|mu|/sigma^2.
- [conj] Superconducting-qutrit dispersive readout produces three IQ clusters, so each ancilla shot yields a likelihood vector over {0, 1, 2}. Feeding these into the DEM posterior replaces the fixed 5p measurement-flip prior with a per-shot value.
- [conj] This matters most where Sec. 5.5 locates the remaining ambiguity, in final-round patterns, where a measurement flip and a data error in the unchecked inverse block look the same. A low-contrast analog outcome is evidence for a measurement error; a crisp one is evidence for a data error.

### 2.3 Bombin, Pant, Roberts and Seetharam 2024: logical gap and fault-tolerant postselection

Citation: Hector Bombin, Mihir Pant, Sam Roberts, Karthik I. Seetharam, "Fault-Tolerant Postselection for Low-Overhead Magic State Preparation", PRX Quantum 5, 010302 (2024), published 4 Jan 2024. DOI 10.1103/PRXQuantum.5.010302. arXiv:2212.00813 (v1 1 Dec 2022), titled there "Fault-tolerant Post-Selection for Low Overhead Magic State Preparation". Fetched: https://arxiv.org/abs/2212.00813, https://arxiv.org/pdf/2212.00813 (text extracted locally), https://ar5iv.labs.arxiv.org/html/2212.00813, https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010302, https://api.crossref.org/works/10.1103/PRXQuantum.5.010302

Extracted facts [lit]:
- Framework (Sec. III):
  - A postselection rule is a pair: a map Q from visible syndrome and erasure information to soft information, and a policy P that accepts or rejects.
  - It is usually implemented with a score S and a cutoff s*, P = Theta(s* - S(q_E)).
  - Keeping a fraction kappa costs overhead 1/kappa. The aim is that the logical error on the accepted kappa fraction is much lower than on all blocks.
- Rules: annular syndrome (radially power-law-weighted count of lit syndromes), logical gap, nested logical gap (gap first, annular syndrome to break ties), radial logical gap (gap computed with radially reweighted edges), and surviving distance (App. VI).
- Logical gap (Sec. III.A.2):
  - Signed gap: Delta = w(l_wrong) - w(l_correct) (Eq. 5), with w_e = ln((1 - p_e)/p_e), erased edges w_e = 0, and a correction's weight the sum over its edges.
  - "In reality ... we only have access to the unsigned logical gap".
  - With MWPM, Delta < 0 means the decoder fails, Delta > 0 that it succeeds, and Delta = 0 that it succeeds half the time. Hence p_enc = sum_i integral_{-inf}^{0} P(Delta_i) dDelta_i (Eq. 6).
  - The combined score is S_G = sum_i a_i e^{-|Delta_i|}.
  - Computation: each boundary gets a pseudosyndrome vertex, and lighting pairs of these flips the logical sector. "Any decoder can be used to compute a logical gap and biased noise can be accommodated by modifying the weights".
- Results (Sec. IV; i.i.d. bit-flip and erasure noise on the FBQC 6-ring fusion network; MWPM):
  - The logical and radial gap scores "have a strong exponential correlation with EER". The annular syndrome score correlates poorly.
  - EER suppression is "super exponential (in kappa)" for the gap rules. The annular rule's breakeven overhead "increases exponentially", and that rule "performs increasingly poorly" as the block size L grows.
  - At 0.6 p*, the radial gap rule reaches breakeven (p_enc = p_init = p_error) at relative overhead 1.78, "~23 times lower than the best annular syndrome rule". At the bulk threshold, breakeven costs about 17.
  - Abstract: an EER reduction by a factor of 15 at relative overhead < 2 when operating at 60% of the bulk threshold.
- Degeneracy: near the injection point the block has constant fault distance 2. The gap per membrane takes only the normalized values {0, 1, 2}, and the combined score takes five values. The EER-versus-kappa curves are therefore step-like, and the nested and radial variants exist to break the degeneracy.

How it bears on this study:
- [ana] Our code has distance 2 everywhere, so Bombin et al.'s degeneracy problem is the generic situation here. A gap based on a single most likely error per class takes few distinct values. The exact DEM posterior breaks ties automatically, because it sums all fault paths and so includes the entropic terms their nested and radial heuristics approximate.
- [ana] Their Eq. (6) is the two-class special case of Chow's identity: the error rate is a functional of the gap distribution alone. For our three classes the analogue is P(error | accepted) = E[1 - q*(D) | accepted].
- [lit] "Fault-tolerant postselection" here means rules built on decoder soft information, whose overhead stays benign below threshold. Their syndrome-count (annular) rule is the counterexample: its breakeven overhead increases exponentially.

### 2.4 Gidney, Newman, Brooks and Jones 2023/2025: complementary gap, calibration, soft information for an outer code

Citation: Craig Gidney, Michael Newman, Peter Brooks, Cody Jones, "Yoked surface codes", Nature Communications 16, 4498 (2025), published 14 May 2025. DOI 10.1038/s41467-025-59714-1. arXiv:2312.04522 (v1 7 Dec 2023). Fetched: https://arxiv.org/abs/2312.04522, https://arxiv.org/html/2312.04522v1, https://arxiv.org/pdf/2312.04522v1 (text extracted locally), https://pmc.ncbi.nlm.nih.gov/articles/PMC12078507/

Extracted facts [lit]:
- Definition (Sec. 4):
  - The paper compares "the probability of a set of errors obtained from minimum-weight matching against the probability of a set of errors obtained from a minimum-weight matching conditioned on the complementary logical outcome".
  - "We call the log-likelihood ratio of these two hypotheses the complementary gap - the log-ratio of the probabilities of the minimum-weight matching and the complementary matching."
  - "A complementary gap close to zero indicates that the decoder is not confident in its decision, while a high complementary gap indicates the decoder is highly confident."
- Computation: "we can form a detector connecting to all the boundary edges on one side of the error graph, similar to [HWL14] ... turning this boundary detector on/off forces the decoder to match/not match to the corresponding boundary". The decoder is a two-pass correlated matcher [Fow13]: the X-type graph reweights the Z-type graph before the gap is computed.
- Units: gaps are reported "in terms of their ratio in dB" and binned to integer dB. "A negative gap indicates that the more likely outcome was incorrect".
- Calibration: "the likelihood of success predicted by the gap is close to the true empirical likelihood of success after rescaling the gap by 0.9x". The decoder "tends towards overconfidence". The calibration holds across distances and rounds.
- Extrapolation: the gap distribution over mn rounds is approximated well by the minimum of m draws from the n-round distribution. This is "slightly optimistic".
- Use: gaps serve as edge weights of the outer (yoke) parity-check error graph, decoded by MWPM, and enable "gap simulation" of large outer codes. The paper does not postselect on gaps.
- Noise model: SI1000 at 1e-3 (Appendix A), the same family as this study.

How it bears on this study:
- [ana] For a calibrated gap g in dB between two classes, the predicted failure probability is 1/(1 + 10^{g/10}). Sec. 0.2 gives the three-class form.
- [conj] Their calibration plot (empirical success per gap bin against prediction) is the right validation for an LRB decoder. In LRB the ideal logical outcome is known shot by shot, so calibration can be measured on hardware without extra circuits. Systematic overconfidence, like their 0.9x, would indicate DEM mismatch.
- [conj] Their use of gaps (soft-weighted erasure-like inputs to an outer code instead of discards) is the sustainable alternative to rejection when the `[[5,1,2]]_3` block is an inner code.

### 2.5 Meister, Pattison and Preskill 2024: efficient soft-output decoders

Citation: Nadine Meister, Christopher A. Pattison, John Preskill, "Efficient soft-output decoders for the surface code", arXiv:2405.07433 (v1 13 May 2024, v2 1 Jun 2024). No journal reference listed. Fetched: https://arxiv.org/abs/2405.07433, https://arxiv.org/html/2405.07433, https://arxiv.org/pdf/2405.07433v2 (text extracted locally). The task brief's author guess "Meister, Pattison, Preskill" is correct.

Extracted facts [lit]:
- Soft-output decoders give "an estimate of the probability of a logical failure conditioned on the error syndrome" (abstract).
- Definition 9 and Algorithm 2: take the fully grown clusters from union-find or MWPM (dual radii), set the weights of edges inside clusters to 0, and run Dijkstra between the two inequivalent boundary nodes. The soft output phi is the shortest path length, computed in O(|E_D| + |V_D| log |V_D|).
- The authors contrast this with "the complementary gap method", which needs a MWPM call on atypical syndromes and so loses the benefit of decoder optimizations.
- Theorem 10 (union-find, repetition code, i.i.d. bit flips, w = log((1 - p)/p)): log[Pr(E = F | sigma)/Pr(E != F | sigma)] = phi(sigma). This is exact.
- Theorem 13 (MWPM, surface code, bit flips): log[Pr(E = F | sigma)/Pr(E = M | sigma)] >= phi(sigma), where M is the minimum-weight valid edge set in the opposite class. They could not prove an upper bound; numerics suggest the bound is tight up to a multiplicative factor.
- Numerics: below the MWPM threshold, phi tracks the log-likelihood of logical failure linearly over about six orders of magnitude, for d = 5 to 13, planar and rotated.
- Postselection (Sec. V):
  - Repetition code, n = 12, p = 0.05: rejecting 0.2% of runs (phi <= 15) lowers the logical error from about 1e-5 to 4e-10 (Fig. 7).
  - Theorem 14 and Corollary 15: when discarding up to half the samples, "postselection guided by soft-output decoding reduces the required code length n by nearly a factor of 4".
  - d = 9 rotated surface code, phenomenological noise, p = 0.005, T = 9, union-find: discarding about 5e-4 of the data improves the logical error from 3.0(5)e-5 to at most 2e-6 (95% confidence, Fig. 8). They caution that this is a single-gate, simplified model.
- Whole-circuit soft output: compute phi_i per decoding window, then q = sum_i 1/(1 + e^{phi_i}) estimates the probability that any window failed (union bound), and the circuit's soft output is log((1 - q)/q).
- Hierarchical codes: soft information from d = 5 or 7 inner surface codes raises the pseudothreshold of a [[1054,140]] lifted-product outer code (BP decoding) by about 3x.

How it bears on this study:
- [ana] The windowed union bound gives a cheap rule for composing per-round confidences over an LRB sequence: accept iff sum_j (1 - q*_j) <= t. It is conservative but monotone. For our small DEM the exact whole-history posterior is computable, so the union bound is only needed for a real-time, round-by-round implementation.
- [conj] Their Theorem 14 is a repetition-code result. Its lesson (a cutoff on soft output reaches a target total-variation distance with a smaller code) applies to circuit sampling, i.e. to the final LRB outcome distribution.

### 2.6 Kishi et al. 2026: hardware-friendly soft outputs

Citation: Kaito Kishi, Riki Toshio, Jun Fujisaki, Hirotaka Oshima, Shintaro Sato, Keisuke Fujii, "Even More Efficient Soft-Output Decoding with Extra-Cluster Growth and Early Stopping", arXiv:2602.03336 (v1 3 Feb 2026). Fetched: https://arxiv.org/abs/2602.03336

Extracted facts [lit] (abstract):
- Meister et al.'s cluster soft output can cost as much as union-find itself in parallel settings and needs global graph information, which is hard on FPGAs.
- Because "the precise value of a large soft output is often unnecessary in practice", two early-stopping soft outputs are proposed:
  - the "bounded cluster gap", which truncates Meister's computation and scales better with distance;
  - the "extra-cluster gap", which grows the clusters slightly further and fits existing FPGA union-find architectures.

How it bears on this study:
- [ana] A postselection decision only needs to know whether q* is above the cutoff, not its exact value. Early stopping is therefore lossless for the accept/reject decision, though not for calibration studies.
- [conj] For the tiny `[[5,1,2]]_3` DEM, a lookup table or a forward recursion over a few hundred states is cheap, so FPGA latency is not the bottleneck for this code.

### 2.7 Lee, English and Bartlett 2025/2026: cluster-statistics confidence for QLDPC codes and real-time abort

Citation: Seok-Hyung Lee, Lucas H. English, Stephen D. Bartlett, "Efficient Post-Selection for General Quantum LDPC Codes", npj Quantum Information 12, 96 (2026). DOI 10.1038/s41534-026-01242-x. arXiv:2510.05795 (v1 7 Oct 2025 to v4 25 Jan 2026). Fetched: https://arxiv.org/abs/2510.05795, https://arxiv.org/html/2510.05795v4

Extracted facts [lit]:
- The logical gap with MWPM needs one decoding per logical class. That is 2^k decodings for k logical qubits in their Z-only setting (up to 4^k in general), and it does not carry over to other decoders.
- Their metrics come from one run of a clustering decoder (BP+LSD in the numerics):
  - cluster-size alpha-norm fraction Q_size = (1/|E|)(sum_i |C_i|^alpha)^(1/alpha);
  - cluster-LLR alpha-norm fraction Q_LLR = [sum_i (sum_{e in C_i} w_e)^alpha]^(1/alpha) / sum_e w_e, with w_e = log((1 - p_e)/p_e).
  - Accept iff Q <= c. The cutoff c is swept, not derived.
- Results:
  - [[144,12,12]] BB code: about 1000x lower logical error, with 1% abort at p = 0.1% and 19% at p = 0.3% (abstract).
  - For surface codes, the MWPM logical gap "remains superior".
  - Sliding-window real-time abort, with a lookback over committed clusters, "matches or even surpasses the original strategy for global decoding" (abstract).
  - The per-round abort rate stays constant as the number of rounds T grows.
- The metrics are heuristic "inverse confidence" metrics. The paper makes no Bayes or Chow optimality claim.

How it bears on this study:
- [conj] The real-time (windowed) abort is the operational form a hardware LRB experiment would use: abort the sequence at the first round whose committed confidence falls below the cutoff, which saves the remaining gates.
- [ana] In LRB the expected cost per accepted shot is (circuit time) / kappa. Early abort reduces the numerator but not the 1/kappa factor.

---

## 3. Postselection on decoder confidence: scaling, thresholds, limits

### 3.1 Smith, Brown and Bartlett 2024: exclusive decoders

Citation: Samuel C. Smith, Benjamin J. Brown, Stephen D. Bartlett, "Mitigating errors in logical qubits", Communications Physics 7, 386 (2024), published 28 Nov 2024. DOI 10.1038/s42005-024-01883-4. arXiv:2405.03766 (v1 6 May 2024). Fetched: https://arxiv.org/abs/2405.03766, https://arxiv.org/html/2405.03766v1, https://arxiv.org/pdf/2405.03766v1 (text extracted locally), https://www.nature.com/articles/s42005-024-01883-4 (publisher page, via the cookie redirect)

Extracted facts [lit]:
- Exclusive MWPM:
  - The matching graph is modified so that it "returns a minimum-weight correction for each logical equivalence class" (App. A, following HWL14).
  - "Let Delta be the unsigned weight difference between the global least-weight correction C and the next-least-weight alternate correction, C', that is logically inequivalent to C."
  - The decoder "will abort if 1 - Delta/d > c" (Eq. 1), with tolerance 0 < c <= 1. Here c = 1 is standard MWPM, and c = 0 is defined as aborting on any non-trivial syndrome.
  - The rule is "similar to post-selection rules that have been studied previously [15, 20]" ([15] Bombin et al., [20] Gidney et al.).
- Code-capacity depolarizing thresholds: 50%, 42%, 38%, 33% and 15% for c = 0, 2/5, 1/2, 2/3 and 1.
- Fault-tolerant zero-tolerance threshold: 32(1)% with measurement error 2p/3 and t = d rounds (phenomenological).
- Failure scaling (Eqs. 2-3): f = C (A p)^{k d} with k = 1 - c/2, which runs from p^{d/2} (MWPM) to p^d (zero tolerance). This is the "up to a quadratic improvement".
- Aborts (Eqs. 4-5): g = C' p (A' p)^{k' d} with k' = c/2 below an abort threshold. The abort thresholds are 1.3(1)%, 2.1(1)% and 4.5(1)% for c = 2/5, 1/2 and 2/3. The zero-tolerance decoder has no abort threshold; its abort probability grows roughly as p d^2.
- Applications:
  - 15-to-1 distillation: 75% fewer physical qubits and 60% less spacetime volume, including repetitions.
  - Error mitigation by repeat-until-success.
  - Concatenation, where an abort heralds a logical erasure for an outer code.

How it bears on this study:
- [ana] Their abort cutoff is a fixed fraction (1 - c) of the distance in edge units. For uniform weights w = log((1 - p)/p) this is a cutoff on the posterior odds of about e^{(1-c) d w}, so the confidence demanded grows with d. For d = 2, Delta/d takes only a few values, so the tolerance knob is coarse. The continuous q* threshold of Sec. 5.1 is the appropriate knob for this study.
- [conj] Their observation that aborts herald erasures for an outer code is the hardware-sustainable reading of rejection. If the `[[5,1,2]]_3` block were concatenated, a rejected block would be an erasure, and a distance-2 outer code can correct one erasure (Knill, Sec. 3.6).

### 3.2 English, Williamson and Bartlett 2025: thresholds for post-selected QEC and the optimal-postselection theorem

Citation: Lucas H. English, Dominic J. Williamson, Stephen D. Bartlett, "Thresholds for Postselected Quantum Error Correction from Statistical Mechanics", Physical Review Letters 135, 120603 (2025), published 18 Sep 2025. DOI 10.1103/nh49-52y2. arXiv:2410.07598 (v1 10 Oct 2024, v2 28 Oct 2024, v3 18 Sep 2025). Fetched: https://arxiv.org/abs/2410.07598, https://arxiv.org/html/2410.07598v3, https://arxiv.org/pdf/2410.07598v3 (text extracted locally), https://api.crossref.org/works/10.1103/nh49-52y2. This is the "English, Williamson, Bartlett" paper the brief asked about. It concerns limits of postselection in general, not error-detection codes specifically.

Extracted facts [lit]:
- Theorem 1: "Post-selection that aborts if MLD returns a maximum coset probability less than some gamma in [0,1] is optimal in the following sense." It partitions the errors into an abort set of some cardinality r, and this partition "strictly upper bounds P_succ for any abort set with equivalent cardinality r". The proof is in Supplement S2.
- Supplement S7: "as long as the error channels are Markovian - enabling the calculation of logical error coset probabilities - Theorem 1 remains valid."
- Decoder-free heuristic: R_heuristic(gamma) = { E : m(E) < -1 + 2 gamma }, where m is the non-equilibrium magnetization (mean stabilizer spin). gamma = 0 aborts nothing; gamma = 1 aborts every non-trivial syndrome.
- Definitions: the conditional logical threshold is computed over the accepted set only; the abort threshold is the error rate above which the decoder aborts with certainty.
- Four regimes, set by being below or above the abort threshold and below or above the logical threshold. Only "below both" is scalable and useful.
- Analytic values:
  - Bit flip, gamma = 1: p_th = 1/(2 + sqrt 2), about 0.2929.
  - Depolarizing, gamma = 1: 0.5, matching Smith et al.
  - The conditional threshold rises with gamma.
  - "For the optimal post-selection, the abort threshold is fixed at the code's non-post-selected logical threshold".

How it bears on this study:
- [ana] Theorem 1 is stated at fixed cardinality of the abort set. The probability-weighted statement relevant to LRB (fixed abort probability) is the fractional-knapsack argument of Sec. 5.1, i.e. Chow/Franc Theorem 6, and it needs randomization at the boundary when the syndrome space is discrete.
- [lit, then conj] S7 is the key caveat for the non-Markovianity question. The optimality of posterior-threshold postselection rests on a correct (Markovian) model of coset probabilities. [conj] Under temporally correlated noise, a decoder whose DEM assumes independent locations is no longer guaranteed optimal, and its confidence will be miscalibrated (Sec. 5.6).

### 3.3 Chen et al. 2025: large deviations and scalable accuracy gains

Citation: Hongkun Chen, Daohong Xu, Grace M. Sommers, David A. Huse, Jeff D. Thompson, Sarang Gopalakrishnan, "Scalable accuracy gains from postselection in quantum error correcting codes", arXiv:2510.05222 (v1 6 Oct 2025, v2 20 Oct 2025, v3 19 May 2026). No journal reference listed. Fetched: https://arxiv.org/abs/2510.05222, https://arxiv.org/html/2510.05222, https://arxiv.org/pdf/2510.05222v3 (text extracted locally). This is the "Chen et al." paper the brief asked about.

Extracted facts [lit]:
- Free-energy difference between classes: beta DeltaF = log(Z_PBC/Z_APBC) (Eq. 2).
- The Nishimori condition implies the symmetry P(DeltaF) = P(-DeltaF) e^{beta DeltaF} (Eq. 3).
- The failure probability is p_f = integral_{-inf}^0 P(DeltaF) dDeltaF = integral P(DeltaF)/(1 + e^{|DeltaF|}) dDeltaF (Eq. 4). Decoders see only |DeltaF|, which they "can use ... as a confidence measure".
- Large-deviation principle: P(DeltaF = s d) ~ exp(-I(s) d), with I convex, I decreasing for s <= s_bar, and I(s) + s = I(-s) (Eq. 5).
  - Failures are dominated by atypical syndromes near DeltaF = 0, with p_f ~ e^{-I(0) d}.
  - Postselecting |DeltaF| >= s* d gives an abort rate of exp(-I(s*) d) and a conditional failure rate of exp(-I(-s*) d).
  - The maximal gain is b = I(-s_bar)/I(0) >= 2, by convexity.
- Fitted gains:
  - Toric code, optimal decoding (TEBD), p = 0.06: s_bar = 0.80(1), I(0) = 0.26(1), b = 3.1(1).
  - MWPM complementary gap: b of about 2.87 for the rotated surface code with circuit-level noise at p = 0.005, and about 3.44 with perfect measurements at p = 0.06.
  - Bond-diluted (partially heralded) RBIM: about 2.91 to 2.92.
- Finite-acceptance postselection "does not alter the error threshold, but effectively increases the code distance from d to I(-s*) d/I(0)".
- With N operations, keeping O(1) acceptance needs N exp(-I(s*) d) <= 1. "Thus, postselecting on an entire circuit cannot be used to reduce overhead, but rather provides a scalable gain in accuracy". Postselecting subroutines such as Bell pairs or resource states can reduce overhead.
- Code splitting: two distance-d copies with keep-the-better postselection match one distance-2d code asymptotically, by Eq. 7.

How it bears on this study:
- [ana] Eqs. 3-4 are the quantum, two-class form of Chow's statement that the error rate follows from the reject (confidence) distribution. For our three-class exact posterior, the analogue is E[accepted error] = E[1 - q*(D) | accepted]. On data this is a calibration identity (Sec. 5.6).
- [ana] The scaling argument explains the LRB-D baseline. In LRB the number of operations grows with m, and for a distance-2 code every rule that keeps the leading-order accepted error from growing with m must reject an O(p) mass per round, because the ambiguous single-fault events occur at rate O(p) per round (about 39p per unit depth for sequence 0 in Sec. 5.5 [num]). Acceptance then decays exponentially in m; the rate constant depends on the rule. The baseline `unif 1` keeps F_acc flat (0.9925 to 0.9919 at p = 1e-3) while rejection climbs to 91.9%, which is this regime. The Chow rule lowers the rate constant (Sec. 5.5) but cannot remove the exponential decay at d = 2.

### 3.4 Staples, Fu and Thompson 2026: partial gap for resource states

Citation: J. Wilson Staples, Winston Fu, Jeff D. Thompson, "Scalable Postselection of Quantum Resources", arXiv:2603.08697 (v1 9 Mar 2026). Fetched: https://arxiv.org/abs/2603.08697, https://arxiv.org/html/2603.08697v1

Extracted facts [lit]:
- Logical gap: G(sigma) = exp(-|w(sigma, l0) - w(sigma, l1)|) (Eq. 1). This is the relative probability of the most likely errors with distinct logical outcomes.
- When a resource state is to be consumed later, its boundary stabilizers are not yet measured, so the ordinary gap "is sensitive to this guess".
- The "partial gap" (Eq. 3) is the likelihood-weighted average of the gap over hidden boundary syndromes sigma_h, with weights exp(-w(sigma, l0)) + exp(-w(sigma, l1)). It is approximated greedily (repetition code) or by "string splitting" (surface code).
- Resource states (cluster states of d rounds plus boundary layers) are accepted or rejected before data is teleported through them, at a fixed rejection fraction such as r = 1/2. Postselection reaches the same logical error per gate with about 4x less overhead.
- At large d, "nearly the full benefit of postselection is realized for very small rejection rates, i.e. r = 0.01".
- Noise: circuit-level dephasing concurrent with CZ gates, simulated in Stim. Caveat: idle coherence must greatly exceed the prepare, measure and postselect time.
- No paper using the exact phrase "partial postselection" was found. This "partial gap" paper and Prabhu and Reichardt (Sec. 3.5) are the closest matches to the brief's "partial postselection".

How it bears on this study:
- [ana] The LRB inverse block has no checks, so at the moment of a mid-sequence decision the final round is a hidden boundary. A mid-circuit (real-time) accept decision should therefore use a partial-gap-type expectation over the not-yet-measured final round, P(L | D_{0..j}) marginalized over future detectors. The exact forward recursion provides this; a gap computed as if the future were trivial does not.
- [conj] Postselecting a resource that is small and repeatable is where postselection reduces overhead. For LRB this suggests postselecting the encoded |+>_L preparation, which is currently ideal, rather than the whole sequence.

### 3.5 Prabhu and Reichardt 2024: combined postselection and correction

Citation: Prithviraj Prabhu, Ben W. Reichardt, "Distance-four quantum codes with combined postselection and error correction", Physical Review A 110, 012419 (2024). DOI 10.1103/PhysRevA.110.012419. arXiv:2112.03785 (v1 7 Dec 2021). Fetched: https://arxiv.org/abs/2112.03785, https://ar5iv.labs.arxiv.org/html/2112.03785

Extracted facts [lit]:
- "if single faults can be corrected and double faults postselected against", logical errors need at least three faults.
- For even distance, "we stop the computation instead of attempting to correct, ensuring logical flips only occur at rate O(p^{t+2})". For d = 4 (t = 1) the logical error is O(p^3) and the rejection O(p^2).
- Flag-pattern tables map each raised-flag set to a correction or a rejection.
- For the surface code the decoder is union-find, rejecting at three or more edges.
- The distance-four surface code with postselection accumulates 25x less error than distance five. For six logical qubits, 25 qubits at d = 4 protect about as well as 246 at d = 5.
- Acceptance falls over long storage or many patches.
- "distance-two codes can detect single errors and distance-three codes can correct them".

How it bears on this study:
- [ana] For d = 2 (t = 0) their rule reduces to pure detection. Our circuit-level setting allows more: repeated rounds and Weyl-error correlations make many single faults uniquely identifiable even at code distance 2 (Sec. 5.4), so the leading-order Chow rule is "correct the unambiguous single faults, reject the ambiguous ones". This is a circuit-level "partial postselection" for d = 2.
- [num] The terminal structure of the LRB-D circuit (noisy final round, then noiseless readout of wires 0 and 3 only) has fault distance 1 for the final readout. A first-order rate u = 4.95p of syndrome-free logical faults sets an O(p) floor that no decoder can remove (Sec. 5.5).

### 3.6 Knill 2004/2005: postselected computation with distance-2 codes

Citations:
- E. Knill, "Quantum Computing with Very Noisy Devices", arXiv:quant-ph/0410199 (v1 25 Oct 2004, v2 2 Nov 2004). The related DOI listed on the arXiv page is 10.1038/nature03350. Fetched: https://arxiv.org/abs/quant-ph/0410199, https://arxiv.org/pdf/quant-ph/0410199v2 (text extracted locally)
- E. Knill, "Fault-Tolerant Postselected Quantum Computation: Schemes", arXiv:quant-ph/0402171 (v1 23 Feb 2004). Fetched: https://arxiv.org/abs/quant-ph/0402171

Extracted facts [lit]:
- The C4/C6 architecture concatenates the "simplest possible error-detecting codes" (C4, a 4-qubit code, and C6, a code on three qubit pairs), both distance 2.
- "In 'postselected' quantum computing, the state is then rejected and the computation restarted. In standard quantum computing, the syndrome information must be used to update the Pauli frame."
- At level 2 and above, a level-1 block where an error was detected is marked; if exactly one of three level-1 pairs is marked, the C6 syndrome corrects it. "This works because error-detecting codes can always correct an error at a known location."
- Thresholds: accurate computation is possible with error probabilities "above 3% per gate". Heuristic analysis of postselected encoded CNOTs suggests a threshold above gamma = 0.06. With resources comparable to today's digital computers, about 1% (abstract).
- Postselected computing must accept with non-zero probability; the success probability falls sharply as accuracy rises (quant-ph/0402171 abstract).

How it bears on this study:
- [ana] The `[[5,1,2]]_3` code can likewise correct one heralded (located) error.
- [conj] Two hardware routes turn LRB rejections into corrections: heralded leakage or erasure detection on qutrits, and concatenation, where a rejected inner block becomes a located error for an outer code.

---

## 4. Postselection in magic-state preparation (where gap postselection is used in practice)

### 4.1 Gidney, Shutty and Jones 2024: magic state cultivation

Citation: Craig Gidney, Noah Shutty, Cody Jones, "Magic state cultivation: growing T states as cheap as CNOT gates", arXiv:2409.17595 (v1 26 Sep 2024). No journal reference listed. Fetched: https://arxiv.org/abs/2409.17595, https://arxiv.org/html/2409.17595v1, https://arxiv.org/pdf/2409.17595v1 (text extracted locally)

Extracted facts [lit]:
- Stages: injection, cultivation, escape.
- "During the injection stage and the cultivation stage, we use full postselection. If any detector from these stages fires, we discard the system and start over. We did some exploratory experiments where we allowed correcting simple detection event patterns, to attempt to reduce retry rates, but ultimately decided to stick with full postselection of the early stages." A secondary benefit is that postselection needs no just-in-time decoding.
- Escape stage: its circuit "is far too large to postselect every possible detection event. Fine grained measures of whether or not to discard a shot are needed".
  - Detection events on red X and green Z color-code stabilizers are postselected so that matching applies.
  - Doublet nodes expose the color-code region to PyMatching, "allowing a confidence to be estimated via the complementary gap".
- Final step: "decide whether or not to keep the state, based on the complementary gap of the escape stage". This waits for the decoder, assumed to take 10 microseconds, and "By varying the cutoff for an acceptable gap, we can achieve a variety of trade-offs between expected cost and fidelity" (Fig. 14).
- Gap computation for Chromobius (Fig. 30): "adding a detector equivalent to an observable to the boundary of the patch then decoding once with it forced on and once with it forced off in order to take a weight difference".
- Calibration (Fig. 26): "the complementary gap is strongly predictive of the logical error rate" when idling in both the surface code and the grafted code.
- Fig. 2 ("This Paper" rows):

  | noise | fault distance | ungrown error rate (discard) | end-to-end error rate (discard) |
  |---|---|---|---|
  | 1e-3 | 3 | 6e-7 (35%) | 3e-6 (80%) |
  | 1e-3 | 5 | 6e-10 (85%) | 2e-9 (99%) |
  | 5e-4 | 5 | 2e-11 (65%) | 4e-11 (90%) |
  | 1e-4 | 5 | 6e-15 (20%) | N/A |

- Light postselection during idling, about a 1e-4 chance of retry per round, maintains an error rate of 1e-9 per round.

How it bears on this study:
- [lit] For small-distance stages, a leading group chose full detection over partial correction. Their stated reasons: feedback is simpler; correcting during cultivation "would require complex just-in-time decoding" because "Fixing such an X error too late ruins the shot"; and a shot with a detected X error "was going to be discarded anyways".
- [ana] The latency argument is specific to non-Clifford state preparation. In Clifford-only LRB, Pauli faults propagate to Pauli faults, so a correction can be deferred to the terminal readout as a Pauli-frame (logical-shift) update. That argument for full postselection does not carry over.
- [conj] For LRB with a d = 2 code, the choice then depends on the reported metric (Sec. 5.2). If the metric is conditional fidelity at a fixed acceptance, the Chow rule of Sec. 5.1 dominates full detection by Claim A. Cultivation's "full postselection" is the corner of that curve that rejects every detection event.

### 4.2 Rosenfeld et al. 2025: cultivation on hardware

Citation: Emma Rosenfeld, Craig Gidney, Gabrielle Roberts, et al. (295 authors), "Magic state cultivation on a superconducting quantum processor", arXiv:2512.13908 (v1 15 Dec 2025). Fetched: https://arxiv.org/abs/2512.13908v1, https://arxiv.org/html/2512.13908v1

Extracted facts [lit]:
- Injection and cultivation keep only shots "in which no events were observed", including the phase-kickback measurement.
- The grafted code is decoded with Tesseract, a most-likely-error decoder.
- Abstract: error reduced by a factor of 40, state fidelity 0.9999(1), "retaining 8% of attempts".
- Injection alone retains 76(3)%. The full-circuit logical error probability lies between 1e-4 and 2e-4.
- A 10% difference in retention between configurations is attributed to possible flux transients or calibration drift.
- No complementary-gap cutoff is described in the portion fetched. The last 72k characters of the HTML page, which include later supplement sections, were not read.

How it bears on this study:
- [lit] Heavy detection-based postselection on superconducting hardware is operationally routine at 8% retention.
- [conj] The retention drift they report is a hardware non-stationarity that a postselected LRB experiment would also see. Acceptance itself should be monitored as an observable.

---

## 5. Synthesis for the `[[5,1,2]]_3` LRB-D study

### 5.1 The optimal accept/correct rule for one LRB circuit [ana]

Setup:
- Fix one circuit (Clifford sequence, depth m, p). A shot yields the full detector history D in Z_3^{4(m+1)}; all rounds are executed in every circuit (brief). The logical shift is L in Z_3, and the ideal outcome is 0.
- A rule is a pair (lhat(D), a(D)): the predicted shift and an acceptance probability in [0, 1], so randomized rules are allowed.
- Acceptance is kappa = sum_D pi(D) a(D).
- The accepted residual distribution is Q(r) = (1/kappa) sum_D pi(D) a(D) P(L = lhat(D) + r | D).
- q*(D) = max_l P(L = l | D).

Claim A. For every kappa in (0, 1], Q(0) is maximized by lhat = argmax_l P(l | D) and a(D) = 1 if q*(D) > t, a(D) = tau if q*(D) = t, a(D) = 0 if q*(D) < t, with (t, tau) fixed by the acceptance constraint.

Proof:
1. kappa Q(0) = sum_D pi(D) a(D) P(lhat(D) | D) <= sum_D pi(D) a(D) q*(D), with equality for MAP.
2. Maximizing sum_D [pi(D) a(D)] q*(D) subject to sum_D pi(D) a(D) = kappa and 0 <= a <= 1 is a fractional knapsack: weights pi(D), value density q*(D).
3. By exchange, the optimum fills the budget in decreasing order of q*.

The Lagrangian form, minimize P(error and accept) + lambda P(reject), gives "accept iff 1 - q* < lambda", which is Chow's rule. Along the optimal curve, d P(error and accept) = -lambda d P(reject).

Consequences:
- Every LRB-D policy (`unif k`, `const c`) is a feasible rule: lhat = 0, accept iff the used detectors are zero. Hence at that policy's own acceptance rate, the Chow rule's accepted accuracy is at least LRB-D's, for every circuit and every p, and for all policies simultaneously.
- Caveat: "optimal" means optimal with respect to the posterior used. With the DEM posterior this is optimal for the simulated model. On hardware it is optimal only to the extent that the DEM is correct (English et al. S7; Sec. 5.6).

### 5.2 The LRB fidelity is affine in accepted accuracy [ana]

1. As implemented in `src/lrb/dem_simulation.py`, every channel is invariant under v -> -v:
   - N1 `d` is uniform over the 8 nonzero (a, b);
   - N1 `f` and `p` are uniform over a in {1, 2};
   - N2 is uniform over the 80 nonzero 4-vectors.
2. The response is linear over Z_3, so the shot distribution satisfies P(D, L) = P(-D, -L).
3. Let a rule be symmetric: a(-D) = a(D) and lhat(-D) = -lhat(D). LRB-D is symmetric, and so is MAP with the Chow acceptance, given P(l | -D) = P(-l | D) and symmetric tie-breaking. Then the residual r = L - lhat(D) has the same accepted distribution as -r, so Q(1) = Q(2).
4. With omega = e^{2 pi i/3} and omega + omega^2 = -1, F = |Q(0) + omega Q(1) + omega^2 Q(2)| = Q(0) - (1 - Q(0))/2 = (3 Q(0) - 1)/2 exactly, for Q(0) >= 1/3.

Consequences:
- Claim A therefore maximizes the per-circuit LRB fidelity at fixed per-circuit acceptance.
- Check against the baseline: F = 0.992503 at m = 0, p = 1e-3 means Q(0) = 0.995002, an accepted error of 4.998e-3. The first-order prediction is 4.95e-3 [num].
- Caveat (circuit average): the pipeline averages per-circuit accepted fidelities over 30 sequences with equal weight. A threshold common to all circuits maximizes pooled accuracy, not the equal-weight mean of per-circuit ratios, so the exact optimum of the reported average would use sequence-dependent thresholds. The difference is small when acceptance is similar across sequences. In the first-order numbers below, at m = 10 across three sequences, the Chow q*=1 rejected mass varies by about 7% and the `unif 1` rejected mass by about 1% [num].

Cost of rejection:
- If rejected shots are counted as fully depolarized (uniform outcome), then sum_k omega^k P_total(k) = kappa sum_k omega^k Q(k), so F_uncond = kappa F_acc.
- Per shot, a correct outcome then contributes 1, an error -1/2 and a rejection 0. That is Chow's loss with lambda = 1/1.5 = 2/3.
- Since q* >= 1/3, we have 1 - q* <= 2/3 = lambda, so the optimal rule never rejects.
- Statement: postselection improves F_uncond for no rule. It improves only the conditional F_acc, at an acceptance cost that must be reported alongside. This parallels Combes and Ferrie's warning in Sec. 1.4.

### 5.3 Gap versus posterior for a Z_3 logical [ana]

- With class log-posteriors ordered so that g1 = log P(lhat)/P(l2) <= g2 = log P(lhat)/P(l3), the confidence is q* = 1/(1 + e^-g1 + e^-g2).
- Forney's optimal rule accepts iff -log(e^-g1 + e^-g2) >= T. The two-class complementary gap rule accepts iff g1 >= T (Forney's LR rule).
- Since e^-g1 <= e^-g1 + e^-g2 <= 2 e^-g1, the two statistics differ by at most log 2. The rules disagree only on syndromes whose g1 lies within log 2 of T.
- Example: posteriors (0.6, 0.2, 0.2) and (0.75, 0.25, 0) have the same g1 = log 3, but error probabilities 0.4 and 0.25.
- Practical rule for this study: report and threshold q*(D), or the odds q*/(1 - q*) in dB. If a "complementary gap" is quoted for comparison with the literature, define it from exact class probabilities as g1 (nats; dB = 4.343 g1) and state that it is not the optimal statistic for three classes.
- Computing class probabilities with a matching-type decoder would need three constrained decodings (logical shift forced to 0, 1, 2), the Z_3 analogue of forcing the observable detector on and off. With the exact DEM posterior no extra decoding is needed.

### 5.4 What distance 2 does and does not prevent [ana]

Single-qutrit Weyl error X^a Z^b on data qutrit q, perfect syndrome, stabilizers as in the brief:
- The Z-check syndrome of X_q^a is a times the exponent column of qutrit q in (S_Z1, S_Z2): q0 (1, 0), q1 (-1, 1), q2 (0, 1), q3 (-1, 0), q4 (0, -1).
- The X-check syndrome of Z_q^b uses (S_X1, S_X2): q0 (1, 0), q1 (1, -1), q2 (-1, 0), q3 (0, 1), q4 (0, -1). The overall sign convention does not affect collisions.

Weight-1 collisions (same syndrome, different class):
- X0^a and X3^-a (differ by (X0 X3)^a = X_L^a);
- X2^a and X4^-a (differ by X2^a X4^a, which is X_L^a times a stabilizer: X0 X3 (X2 X4)^-1 = S_X1 S_X2);
- Z0^b and Z2^-b (differ by Z_L^b);
- Z3^b and Z4^-b (differ by Z3^b Z4^b, which is Z_L^b times a stabilizer: Z0 Z2 (Z3 Z4)^-1 = S_Z1 S_Z2).

Every mixed error (a != 0 and b != 0) and every error on qutrit 1 has a unique weight-1 syndrome, because its X-type and Z-type parts would need partners on different qutrits. Under uniform single-qutrit depolarizing noise, 16 of the 40 (qutrit, Weyl) single errors are ambiguous (40%) and 24 are uniquely identifiable (60%).

Whether an ambiguous pair matters depends on when the fault happens:
- The two explanations differ by a logical Pauli at the fault time.
- For a fault after forward Clifford j, the rest of the circuit composes to U = (C_j ... C_1)^-1. The pair changes the x0 + x3 readout iff U X_L^a U^dagger (or U Z_L^b U^dagger) has a Z_L component.
- For uniformly random Clifford prefixes (the 216-element catalog is the single-qutrit Clifford group modulo phases, and Sp(2, Z_3) acts transitively on nonzero vectors), this happens with probability 6/8 = 3/4.
- At the final round (U = I), X-type ambiguities are harmless and Z-type ones are harmful.

Measurement and reset flips in rounds j < m are identifiable because detectors are raw outcomes and data errors persist into later rounds. Flips in the terminal round m are confounded with data errors in the unchecked inverse block.

### 5.5 First-order (p -> 0) trade-off curves on baseline circuits [num]

Method:
- For a circuit, build the compact DEM (`build_compact_detector_error_model`).
- Enumerate every location and every nonzero Weyl vector its channel allows, and record the detector pattern D and logical shift L.
- Accumulate first-order rates per (D, L). Every SI1000 location probability is proportional to p (0.1p, p, 2p, 5p), so the first-order posterior of each pattern, and hence the classification below, is independent of p [ana].
- All masses are reported per unit p.
- Circuits: `experiments/LRB/<seq>/4/<depth_idx>.chp` of the baseline run (p_idx 4, p = 1e-3, used only to read the circuit structure). Script in Appendix A.

Definitions:
- u: syndrome-free logical mass (D = 0, L != 0).
- R: rejected mass.
- E: accepted residual-error mass, excluding u.
- "Chow q*=1": accept patterns whose single-fault explanations all agree on L, and decode to that L.
- "Chow eps": accept patterns with first-order max posterior >= 1 - eps.
- `const 1`: reject iff a final-round detector fires; no correction.
- MAP: accept everything.

Clifford sequence 0, masses per unit p:

| m | u | R `unif 1` | R `const 1` (E) | R Chow q*=1 (E) | R Chow eps=0.05 (E) | R Chow eps=0.1 (E) | MAP, no rejection: E |
|---|---|---|---|---|---|---|---|
| 0 | 4.95 | 63.15 | 63.15 (0) | 41.88 (0) | 39.03 (0.10) | 38.70 (0.13) | 9.08 |
| 2 | 4.95 | 280.55 | 206.55 (0) | 109.33 (0) | 86.33 (0.78) | 79.23 (1.28) | 25.75 |
| 4 | 4.95 | 506.15 | 350.15 (0) | 205.48 (0) | 164.90 (1.00) | 142.98 (2.43) | 53.03 |
| 10 | 4.95 | 1187.75 | 785.75 (0) | 450.95 (0) | 333.33 (2.90) | 300.78 (5.23) | 119.93 |
| 22 | 4.95 | 2515.95 | 1621.95 (0) | 884.00 (0) | 624.33 (6.25) | 556.25 (10.88) | 239.75 |

Sequences 1 and 2 at m = 10:

| sequence | u | R `unif 1` | R `const 1` | R Chow q*=1 | R Chow eps=0.05 (E) | MAP: E |
|---|---|---|---|---|---|---|
| 1 | 4.95 | 1173.95 | 771.95 | 449.83 | 314.45 (3.18) | 118.73 |
| 2 | 4.95 | 1187.35 | 785.35 | 421.08 | 330.20 (2.18) | 115.25 |

Consistency with the baseline at p = 1e-3, using the first-order Poisson estimate P(accept) = exp(-R p):

| policy, depth | predicted rejection | baseline rejection |
|---|---|---|
| `unif 1`, m = 2 | 24.5% | 24.7% |
| `unif 1`, m = 4 | 39.7% | 39.8% |
| `unif 1`, m = 10 | 69.5% | 69.1% |
| `unif 1`, m = 22 | 91.9% | 91.9% |
| `const 1`, m = 2 | 18.7% | 18.9% |
| `const 1`, m = 4 | 29.5% | 29.5% |
| `const 1`, m = 10 | 54.4% | 53.5% |
| `const 1`, m = 22 | 80.2% | 79.4% |

The m = 0 accepted fidelity is 1 - 1.5 x 4.95e-3 = 0.99257 against 0.992503 in the baseline.

Findings:
1. **[num]** The syndrome-free logical mass u = 4.95 is identical at every depth checked: the first-order floor of LRB-D accepted infidelity does not grow with m. It matches a hand count [ana] of Z-type faults on wires 0 and 3 after their last X-check in the final round:
   - idle depolarizing during the M7 and M8 layers on wire 0 and during the M8 layer on wire 3: 3 layers x (6/8) x 2p = 4.5p;
   - the two-qutrit depolarizing after CNOT 7->0 and after CNOT 8->3, counting components with a Z part on the data qutrit and no flip of the ancilla outcome: 2 x (18/80) p = 0.45p;
   - total 4.95p. The m = 0 circuit was listed operation by operation to confirm that the final round has four measurement layers and no reset layer.
   - No decoder can remove this floor; changing the circuit can.
   - [conj] For example, add a terminal data readout that yields the X-stabilizers, as the `const 0` circuits do, or end the round with the X-checks after the idle.
2. **[num]** At first order every logical-affecting single fault also flips a final-round detector, so `const 1` has E = 0 and accepted error u, like `unif 1`, while rejecting less. Its worse baseline fidelity at large m (0.9666 at m = 22) is therefore a higher-order effect: multi-fault patterns that cancel in the final round.
3. **[num]** The Chow rule that accepts only first-order-unambiguous patterns has the same first-order accepted error u as LRB-D. For m >= 2 its rejected mass is 35-41% of `unif 1`'s: ratios 0.390, 0.406, 0.380, 0.351 for m = 2, 4, 10, 22 (sequence 0), and 0.383 and 0.355 for sequences 1 and 2 at m = 10. At m = 0 the ratio is 0.663.
   - The accepted-but-detected mass splits into patterns decoded with no correction (MAP class 0) and patterns needing a logical correction: 150.2 versus 21.1 at m = 2, and 618.7 versus 118.1 at m = 10.
   - Isolated single-detector patterns account for 99.6 (m = 2) and 428.7 (m = 10) of the accepted mass.
   - All first-order-ambiguous mass touches the final round.
4. **[num]** Per unit depth (sequence 0, m = 2 to 22):
   - `unif 1` rejected mass grows by about 112p and the Chow q*=1 rule by about 39p.
   - With no rejection, MAP accumulates about 10.7p of accepted error per unit depth.
   - The baseline physical qutrit (p_RB = 5p) decays from 0.9891 at m = 2 to 0.8832 at m = 22, about 0.0057 per unit depth in log-fidelity, or 3.8p per unit depth as an error rate under F = 1 - 1.5 e.
   - So for this code and check schedule, full correction is about 2.8x worse per unit depth than the unencoded qutrit at first order, and partial postselection is needed for any encoded advantage.
5. **[num, then conj]** At p = 1e-3 the first-order extrapolation gives acceptance exp(-0.884), about 41%, for Chow q*=1 at m = 22, against 8.1% for `unif 1`. [conj] This is not reliable at m = 22: the mean number of faults per shot is about 2.9 there, so multi-fault patterns dominate and the exact posterior is needed. The package docstring and function list of the worktree's `src/lrb/correction/` (written by another agent in this study) describe an exact forward recursion (`forward.py`) and soft-threshold and blip-forgiving rules (`decoders.py`). Only the docstring and function names were read; the implementations were neither used nor checked here.
6. **[num]** Chow's slope relation holds on these curves. From eps = 0 to 0.05 at m = 10, Delta E/Delta R = 2.90/(-117.6) = -0.025, below eps as it must be. The d = 2 ambiguous pairs enter in two large jumps: between eps = 0.2 and 0.34 (first-order posteriors near 2/3) and at eps = 0.5 (posteriors near 1/2).

### 5.6 Calibration as a diagnostic for model mismatch and non-Markovianity

- [lit] Confidence calibration is standard practice: the gap-versus-empirical-error curves in Yoked (0.9x rescaling, overconfident at high confidence) and in cultivation (Fig. 26). Chen et al.'s Eq. 3 is the identity that a correct posterior must satisfy.
- [ana] For an exact posterior, P(correct | q* = q) = q for every q, and E[accepted error] = E[1 - q* | accepted] (three-class Chow/Bombin Eq. 6). In LRB the ideal outcome is known per shot, so both sides are measurable.
- [conj] Run the DEM-posterior decoder on hardware (or on simulated non-Markovian noise) and plot the empirical error against 1 - q* in bins. Deviations from the diagonal quantify how wrong the Markovian DEM is, separately from the LRB decay fit.
- [conj] Correction is model-dependent where detection is not. LRB-D's accept set (no detection on the used rounds) needs no model. The Chow rule's accept set and its corrections depend on the model's relative rates, for instance measurement flips at 5p versus inverse-block data errors in the final round.
- [conj] Under temporally correlated noise, confident corrections can be confidently wrong, so correction may raise sensitivity to non-Markovian noise relative to detection. English et al. S7 states the Markovian assumption explicitly. This should be tested numerically by whoever handles study question 2.

### 5.7 Hardware implications

- [lit] Latency budgets in the literature:
  - Cultivation assumes about 10 microseconds for the final gap decision.
  - Real-time windowed abort exists for QLDPC codes (Lee et al.).
  - FPGA-friendly early-stopping soft outputs exist (Kishi et al.).
- [conj] For this 5-qutrit code, the exact posterior update per round is a small table lookup, so latency is not limiting.
- [conj] The dominant hardware question is calibration of the DEM rates, especially the measurement-flip rate against idle rates. These set which final-round patterns the Chow rule accepts.
- [lit] Soft readout (Pattison et al.) improves thresholds when analog data are fed to the decoder.
- [conj] For superconducting qutrits, the three-outcome IQ likelihoods are the natural input, and they target exactly the final-round ambiguity.
- [lit] Heralded location information makes distance-2 codes correcting (Knill).
- [conj] Leakage heralding or erasure-type qutrit encodings would convert part of the ambiguous mass into correctable events.
- [lit] Hardware postselection at low retention (8%) is routine for state preparation (Rosenfeld et al.). Retention drifts with calibration.
- [conj] Treat acceptance as a monitored observable in any hardware LRB run.

### 5.8 Caveats and open items

- [ana] The first-order tables use the circuits' structure only. They are exact as p -> 0 and become unreliable once R p is of order 1.
- [conj] At p >= 5e-3 or m >= 10 the higher-order (multi-fault) posterior changes the picture, and the exact forward recursion should be used for the trade-off curves.
- [ana] The LRB fit (offset fixed at 0) assumes a single exponential in m for the accepted fidelity. Under the Chow rule both F_acc and kappa depend on m. Reporting (F_acc(m), kappa(m)) or the risk R*(lambda, m) avoids hiding the decay in the acceptance.
- Literature checked and found not relevant: Google Quantum AI, "Quantum error correction below the surface code threshold", arXiv:2408.13687 v1 (first authors Rajeev Acharya, Laleh Aghababaie-Beni, Igor Aleiner, et al.), fetched at https://arxiv.org/html/2408.13687v1. It contains no confidence-based postselection result; the only related text is a decoder heralding failure in the real-time decoding section. It is not used as support here.

---

## Appendix A. First-order enumeration script

Run on the login node with `module load python/3.11`, from a scratch folder, reading circuits from the read-only baseline folder. Runtime: m = 22 build about 60 s; everything else a few seconds.

```python
"""First-order (p -> 0) error/reject trade-off of decision rules on one LRB circuit DEM.
Usage: python first_order_tradeoff.py <worktree_src> <p> <circuit.chp> [...]
"""
import sys, itertools, time, re
from collections import defaultdict
src, p = sys.argv[1], float(sys.argv[2]); paths = sys.argv[3:]
sys.path.insert(0, src)
import numpy as np
from lrb.correction.compact_dem import read_run_circuit as read_circuit
from lrb.correction.compact_dem import build_compact_detector_error_model

def items_for(loc, base):
    if loc.gate == "N1":
        if loc.channel == "d":
            vs = [v for v in itertools.product(range(base), repeat=2) if any(v)]
        elif loc.channel == "f":
            vs = [(a, 0) for a in range(1, base)]
        elif loc.channel == "p":
            vs = [(0, b) for b in range(1, base)]
        else:
            raise ValueError(loc.channel)
        return [(v, loc.probability / len(vs)) for v in vs]
    dist = np.asarray(loc.params["prob_dist"], dtype=float)   # N2: dense 81-entry distribution
    out = []
    for idx, pr in enumerate(dist):
        x, digs = idx, []
        for _ in range(4):
            digs.append(x % base); x //= base
        digs = tuple(reversed(digs))                           # same digit order as the sampler
        if any(digs) and pr > 0:
            out.append((digs, pr))
    return out

for path in paths:
    model = build_compact_detector_error_model(read_circuit(path))
    base, nd = model.base, model.num_detectors
    rounds = np.array([int(re.search(r"_r(\d+)_", lab).group(1)) for lab in model.detector_labels])
    m = rounds.max(); rates = defaultdict(lambda: np.zeros(base)); u = 0.0
    for loc in model.locations:
        for v, w in items_for(loc, base):
            D = np.zeros(nd, dtype=np.int64); L = 0
            for ci, terms in enumerate(loc.component_terms):
                c = v[ci] % base
                if c == 0: continue
                for di, coef in terms.detector: D[di] = (D[di] + c * coef) % base
                for li, coef in terms.logical:  L = (L + c * coef) % base
            nz = np.flatnonzero(D)
            if len(nz) == 0:
                u += (w / p) if L != 0 else 0.0
                continue
            rates[(tuple(nz.tolist()), tuple(D[nz].tolist()))][L] += w / p
    det = sum(v.sum() for v in rates.values())
    pats = []
    for k, vec in rates.items():
        tot = vec.sum(); q = vec.max() / tot
        pats.append((q, tot, tot - vec.max(), any(rounds[i] == m for i in k[0]), tot - vec[0]))
    print(path, "m", m, "u", u, "unif1 R", det)
    print("const1 R", sum(t[1] for t in pats if t[3]), "E", sum(t[4] for t in pats if not t[3]))
    print("MAP E", sum(t[2] for t in pats))
    for eps in [0.0, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.34, 0.5]:
        acc = [t for t in pats if t[0] >= 1 - eps - 1e-12]
        print("Chow eps", eps, "R", det - sum(t[1] for t in acc), "E", sum(t[2] for t in acc))
```

## Appendix B. Status of the papers named in the task

| Named in task | Status |
|---|---|
| Hutter, Wootton, Loss 2014 PRA | Confirmed (Sec. 2.1). It introduces per-class MWPM but not the term "complementary gap"; later papers attribute the construction to it. |
| Bombin, Pant, Roberts, Seetharam 2024 PRX Quantum | Confirmed (Sec. 2.3). |
| Gidney, Newman, Brooks, Jones 2023 "Yoked surface codes" | Confirmed (Sec. 2.4); published in Nature Communications 2025. |
| Smith, Brown, Bartlett 2024 Commun. Phys. | Confirmed (Sec. 3.1). |
| "Meister, Pattison, Preskill" 2024 | Confirmed with exactly these authors (Sec. 2.5); arXiv only. |
| Gidney, Shutty, Jones 2024 "Magic state cultivation" | Confirmed (Sec. 4.1); arXiv only. |
| Pattison, Beverland, da Silva, Delfosse 2021 | Confirmed (Sec. 2.2); arXiv only. |
| "Chen et al." on postselection with error correction | Hongkun Chen et al. 2025 (Sec. 3.3). |
| "Partial postselection" | No paper with that phrase found. Closest: Staples, Fu, Thompson 2026 "partial gap" (Sec. 3.4) and Prabhu, Reichardt 2024 (Sec. 3.5). |
| "English, Williamson, Bartlett" on postselection limits | Confirmed: PRL 135, 120603 (2025) (Sec. 3.2). It is about thresholds of post-selected QEC in general, not error-detection codes specifically. Knill's distance-2 work (Sec. 3.6) covers error-detection codes. |
| Forney 1968 IEEE TIT | Metadata confirmed via Crossref; content via HSS10 and Merhav (Sec. 1.2). |
| Chow 1970 | Metadata confirmed via Crossref; abstract via IBM Research; content via Combes-Ferrie and Franc et al. (Sec. 1.1). Chow 1957 metadata also confirmed. |
| Extra (not named) | Franc-Prusa-Voracek 2023, Combes-Ferrie 2015, Hof-Sason-Shamai 2010, Merhav 2008, Lee-English-Bartlett 2026, Kishi et al. 2026, Knill 2004/2005, Rosenfeld et al. 2025. |
