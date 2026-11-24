# E: soft-output (|LLR| gap) postselection versus LRB-D

Work package E. Qubit `[[5,1,2]]` LRB model of `src/lrb_qubit/` (lead), main
grid `results/grid_main/` (30 sequences, depths 0-22, 1e6 LRB-D shots and 3e5
exactly decoded shots per circuit, physical RB at p_RB = 5p), plus fresh
identical-shot samples generated here. Labels: **VA** verified analytically,
**VN** verified numerically, **LIT** from literature (fetched), **CONJ**
conjecture. Unless stated otherwise, error rates are logical errors among
accepted shots, pooled over the 30 sequences, and brackets are 95% sequence-
bootstrap intervals (500 resamples of the 30 sequences, paired across methods).

## Summary

1. **The exact-MLD soft output is calibrated** (VN). In every |LLR| bin the
   fraction of wrong MLD decisions lies inside the exact-posterior band
   `[q(upper edge), q(lower edge)]`, `q(g) = 1/(1+e^g)`: 2 of 4,979 per-depth
   bins sit outside it at more than 3 sigma and none at more than 4 sigma
   (p = 1e-4 to 5e-2, mld2, mld5 and the no-check decoder).
2. **Thresholding |LLR| is the optimal acceptance rule at fixed acceptance**
   (VA: fractional-knapsack / Neyman-Pearson argument, section 1.2; LIT:
   Seip et al. 2026, Theorem II.1). No LRB-D point beats the frontier built
   from the same detection events: on the grid none of 10,715 comparisons
   lies below the optimistic frontier bound by more than 3 sigma (min
   z = -1.90), and on fresh identical shots none of 7,025 paired comparisons
   has z < -2 (min z = -1.41) (VN).
3. **LRB-D unif=1 is itself a frontier point** (VN, identical shots). The
   all-clean check syndrome is the most confident syndrome of every circuit,
   so the gap rule tuned to unif=1's acceptance accepts exactly the same shots
   (1,414 of 1,414 circuits with the data-2 information, 1,414 of 1,414 for
   unif=1 + final with the five-qubit readout). The soft output therefore
   cannot lower unif=1's error at unif=1's acceptance. This independently
   confirms the lead's L0 section 7 (unif=1 + final on the mld5 frontier,
   ratios 0.97-1.03; all nine L0 rows reproduced within our intervals).
4. **Where the soft output does help.**
   (a) Every sparser LRB-D policy is dominated. Against the frontier with the
   *same* information (check rounds only, mld2) the factors are modest:
   unif=2 0.99-1.17x, unif=5 1.01-1.88x, const=1 1.04-6.9x in accepted error
   at equal acceptance (m >= 6, p <= 2e-3). Against the full-information mld5
   frontier they are the 3-15x of L0, because most of that gap is the final
   data readout, not the decision rule.
   (b) At equal error rate the gap rule accepts more shots, e.g. 1.5x, 2.2x and
   4.8x more than const=1 at m = 22 for p = 5e-4, 1e-3, 2e-3.
   (c) The data-2 readout caps every soft output: an undetectable flip of
   probability u (exact DEM value: 4.53p at small p, 4.50p at p = 2e-3) is
   independent of the syndrome, so `|LLR| <= log((1-u)/u)` and the error of
   every data-2 rule is at least u (VA; VN at all 14 p). Reading all five data
   qubits removes the cap and lowers the error at unif=1's acceptance by 248x
   [199, 325] (m = 0) to 9.2x [8.7, 9.9] (m = 22) at p = 5e-4.
5. **Fixed-threshold LRB curves cannot decay to zero** (VA, VN): a threshold
   t bounds every accepted shot's error by q(t), so F(m) >= tanh(t/2) at
   every depth (793/793 grid cells). The repo's offset-0 fit then reports f
   close to 1 while the decay moves into the acceptance; e.g. at p = 2e-3,
   f = 0.94773(53), 0.99484(4), 0.99892(1) for t = 0, 2, 4 (mld5), with
   acceptance at m = 22 of 1, 0.20 and 0.049. f is a property of the
   postselection, not of the logical gates. For comparison, physical RB gives
   f = 0.99322(13) at p_RB = 5p and 0.99864(3) at p_RB = p (offset 0).
6. **Charging rejection as a random guess, plain MLD wins** (VA, VN):
   acceptance x F is maximised by accepting everything; no LRB-D policy or
   gap threshold beats it at any of 126 (p, m) points.
7. **Rule-based hybrids** (fresh identical shots, section 2.7). The best is
   the decoder-free time-like-pair rule with the final data check (`tl_d5`:
   tolerate isolated flipped ancilla measurements and lone terminal Z-check
   events, reject everything else, never correct). It accepts no single fault
   with a wrong output (all 1,620 fresh circuits audited). At p = 2e-3 it
   accepts 0.93, 0.41 and 0.044 of the shots at m = 0, 6 and 22 (unif=1 +
   final: 0.87, 0.25, 0.0087) with errors 3.1e-4, 3.6e-3 and 1.1e-2; at those
   error rates the gap rule accepts only 1.00x, 1.03x [1.03, 1.04] and 1.26x
   [1.22, 1.31] as many shots, so `tl_d5` sits close to the frontier.
   Adding qubit-1 syndromes (`tlq1`, first-order unsafe, section 4) or a DEM
   single-fault lookup (`lut0`, `lut10`) accepts more shots but ends at
   2.0-2.6x the frontier error at equal acceptance by m = 22 (paired z > 60).
   No hybrid beats the gap rule on identical shots (VN).

## 1. Methods

### 1.1 Soft output and the gap rule

`mld.TransferMatrixMLD.log_likelihood_ratio` returns the exact
`LLR = log P(L=0|s) - log P(L=1|s)` for the detection events `s` it is given.
Two information sets are used: **mld5** (all check rounds plus the two final
X-stabilizer detectors of the five-qubit terminal readout) and **mld2** (check
rounds only, the qutrit pipeline's data-2 readout; built from the `data2`
circuit's DEM, the exact marginal). The *gap rule* with threshold t accepts a
shot iff |LLR| >= t and outputs the MLD decision; its conditional error is
`q(|LLR|) = 1/(1+e^|LLR|)`.

### 1.2 Optimality and its corollaries (VA)

For one circuit, any (randomised) acceptance rule that looks at `s` is a pair
`a(s) in [0,1]`, `L_hat(s)`. Its acceptance is `alpha = sum_s P(s) a(s)` and
its accepted error mass is
`E = sum_s P(s) a(s) P(L != L_hat(s) | s) >= sum_s P(s) a(s) q(s)`, with
equality for the MLD decision. Minimising `sum P(s) a(s) q(s)` at fixed
`alpha` is a fractional knapsack: accept syndromes in increasing order of
`q(s)`, i.e. decreasing |LLR|, randomising only on the boundary tie. Hence the
optimal `E*(alpha)` is convex, piecewise linear with slopes `q`, and every
rule's `(alpha, E)` lies on or above it (this is Chow's error-reject rule;
LIT: C. Chow, IEEE Trans. Inf. Theory 16(1), 41-46 (1970), metadata fetched
via Crossref; Seip, Colmenarez, Schmitt, Muller, arXiv:2606.22194 (2026),
Theorem II.1, fetched). Corollaries used below:

- *Information ordering.* Rules on the check rounds are rules on all
  detectors, so mld5-frontier <= mld2-frontier <= every data-2 rule (LRB-D
  data-2 policies, `_d2` hybrids). `_final` policies and `_d5` hybrids are
  compared with mld5 only.
- *Pooling.* With the sequence known and every posterior exact, the optimum
  for the pooled problem is one common |LLR| threshold over sequences; per-
  circuit thresholds are also valid rules (used in the identical-shot test).
- *Per-shot bound.* Every shot accepted at threshold t has error <= q(t), so
  in every circuit `F = |1 - 2 eps| >= tanh(t/2)` (expectation).
- *Data-2 cap.* The data-2 DEM contains detector-free mechanisms that flip the
  logical with total probability u, independent of everything else. Then
  `P(L|s) = (1-u) P'(L|s) + u P'(1-L|s)`, so `q(s) >= u` and
  `|LLR(s)| <= log((1-u)/u)` for every syndrome.
- *Rejection as a guess.* Scoring a rejected shot as a random bit gives
  `alpha F = sum_s P(s) a(s) (1 - 2 q(s)) <= 1 - 2 eps_MLD`: accept all.
- The theorem is about expected counts. Every numerical test below allows for
  statistical error.

### 1.3 Pareto curves from the grid histograms

`gap_hist` stores, per sequence and depth, counts of correct and wrong MLD
decisions in 0.25-wide |LLR| bins (`GAP_EDGES`). `acceptance_rules.
cumulative_counts` gives every bin-edge threshold rule. Between edges,
`interpolate_counts` accepts a fraction `lam` of the boundary bin at random:
an achievable rule, but slightly worse than the exact threshold inside the bin
(the conservative curve). `ParetoCurve.error_at_acceptance` also returns an
optimistic bound that charges the boundary-bin shots the smallest error
`q(upper edge)` their |LLR| allows. An LRB-D point violates optimality only if
it lies below the *optimistic* bound by more than 3 sigma, with
`sigma = max(paired sequence bootstrap, binomial, 1/N_accepted)`. Note that
LRB-D uses all 1e6 shots and the histograms the first 3e5 of them, so the grid
comparison is on overlapping, not identical, shots; section 1.6 fixes this.

### 1.4 Repo-style fits

`acceptance_rules.fit_fixed_offset` reproduces `LRBResultsPlotter.
_fit_lrb_from_stats` (`_mask_invalid`, ceiling cut at 1.0, weights
`1/std^2` via `_safe_weights`, coarse-to-fine scan over f in [1e-8, 1] with
2001 points and 5 refinements, offset a = 0) and `fit_free_offset` the
physical-RB fit (free a). Both are tested against the repo functions
(`test_fixed_offset_fit_matches_repo`, `..._free_offset_...`: identical f).
Means and population std are over the 30 sequences as in the repo CSVs;
a depth where some sequence accepts nothing gets NaN and is dropped, as in the
repo. Uncertainties: 500 sequence resamples; each resample recomputes the
per-depth means and stds and refits (vectorised scan, agreeing with the scalar
scan to the final grid step). Physical RB is fitted at p_RB = 5p (grid) and
p_RB = p (generated here, same 30 sequences, 1e6 shots, seed 4242), with both
the repo's free offset and offset 0.

### 1.5 Matched operating points

For an LRB-D policy at (p, m): *matched acceptance* finds the (randomised)
gap rule with the same pooled acceptance and reports
`eps_LRB-D / eps_gap` (error reduction); *matched error* finds the
largest-acceptance gap rule with `eps_gap <= eps_LRB-D` and reports
`acc_gap / acc_LRB-D` (acceptance gain). Both are recomputed inside every
bootstrap resample. Applying the per-depth matched rule to every sequence
gives F(m) curves that are fitted like LRB-D (`matched_fits.csv`).

### 1.6 Fresh identical shots (`studies/correction-decoders/qubit/scripts/hybrid_rules.py`)

For p in {5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 1e-2}, the 30 grid sequences and all
9 depths, 2e5 fresh shots per circuit (seed base 9e9, disjoint from the grid)
are sampled once and scored by every method: LRB-D const/unif 1-22 (data-2)
and every `_final` variant, the hybrids, and the exact LLR with both
information sets. `MatchedGapScorer` ranks the same shots by |LLR| (rounded to
1e-9, boundary ties randomised) and accepts as many as the rule did, per
circuit. The paired difference `D = errors_rule - errors_gap` has shot-noise
variance `sum z_i^2 - (sum z_i)^2/n` with
`z_i = 1[i accepted] e_rule(i) - w_gap(i) e_MLD(i)`; per-circuit D and variances
are summed over sequences. Optimality predicts `D >= 0`.

### 1.7 Hybrid rules

Detection events are grouped into time slots (check rounds, plus the final
data-readout slot for `_d5` rules). A *cluster* is a run of consecutive
non-empty slots. A cluster longer than two slots, or any cluster not in the
rule's table, rejects the shot; otherwise the logical correction is the XOR
of the clusters' table entries (`evaluate_cluster_rule`).

| rule | table | correction | needs |
|---|---|---|---|
| `tl` | isolated time-like pairs: one flipped ancilla measurement in round r plus the detectors of round r+1 that compare with it (from the logical gates' stabilizer maps `LRBCircuit.back`); lone terminal Z-check events | none | code structure only |
| `tlq1` | `tl` plus qubit-1 syndromes in one round (both Z checks, both X checks, or all four; both final X detectors) | none | code structure only |
| `lut0` | exactly one DEM mechanism's signature per cluster, and all mechanisms with that signature agree on the logical flip | that flip | a lookup table from the DEM |
| `lut10` | as `lut0`, accepting signatures whose first-order posterior odds are >= 10 | majority flip | a lookup table from the DEM |

`single_fault_audit` applies a rule to every DEM mechanism on its own and
returns the first-order error budget (accepted-and-wrong probability).

## 2. Results

### 2.1 Calibration (VN)

`calibration.csv`, `calibration_summary.csv`, `figures/E/calibration.png`.
Pooled over sequences and depths, the largest deviation outside the band is
|z| = 3.08 (no-check decoder, p = 2e-4) and 2.31 for mld5; per depth, 2 of
4,979 bins exceed 3 sigma and none 4 sigma. The data-2 cap is visible: the
highest occupied mld2 bin always sits just below `log((1-u)/u)` (e.g. 5.25-5.50
vs 5.40 at p = 1e-3, 3.50-3.75 vs 3.78 at p = 5e-3), while mld5 reaches
|LLR| ~ 10 at p = 1e-3.

### 2.2 Neyman-Pearson tests

Grid, binned (`np_test_summary.csv`, `np_test_grid.csv`): 10,715 comparisons
(every LRB-D policy at every depth and all 14 p > 0; data-2 policies against
mld2 and mld5, `_final` policies against mld5, const0 against its own decoder).
None lies below the optimistic frontier bound by more than 3 sigma (smallest z
= -1.90). 272 lie more than 3 sigma below the conservative bin-randomised
curve; these are points that sit exactly on the frontier (unif=1, unif=1 +
final, const0, whose accepted syndrome shares its 0.25-wide bin with worse
ones) or high-p points where almost all |LLR| fall into the first two bins. The
identical-shot test resolves them.

Identical fresh shots, exact (`np_test_exact.csv`, `np_test_exact_lrbd.csv`):
7,025 paired comparisons (every LRB-D policy and `_final` variant, every
hybrid, every depth, six p; data-2 rules against mld2 and mld5, the others
against mld5). None has z < -2 (if every rule sat exactly on the frontier,
about 160 would by chance); the smallest is -1.41 and the median per (p,
family) is 12-160. In every circuit with accepted shots, unif=1 accepts exactly
the same shots as the mld2 gap rule (1,414 of 1,414 circuits, paired difference
identically zero) and unif=1 + final exactly the same shots as the mld5 gap
rule (1,414 of 1,414).

### 2.3 LRB-D against the frontier (grid)

LRB-D operating points (pooled acceptance / error rate):

| p | m | unif=1 | unif=5 | const=1 | unif=1+final | const=1+final |
|---|---|---|---|---|---|---|
| 0.0005 | 0 | 0.971 / 0.00228 | 0.971 / 0.00228 | 0.971 / 0.00228 | 0.966 / 8.14e-6 | 0.966 / 8.14e-6 |
| 0.0005 | 6 | 0.709 / 0.00237 | 0.781 / 0.00263 | 0.798 / 0.00304 | 0.705 / 8.52e-5 | 0.794 / 6.25e-4 |
| 0.0005 | 22 | 0.306 / 0.00261 | 0.44 / 0.00369 | 0.488 / 0.0113 | 0.304 / 2.81e-4 | 0.485 / 0.00845 |
| 0.001 | 0 | 0.943 / 0.00464 | 0.943 / 0.00464 | 0.943 / 0.00464 | 0.933 / 3.83e-5 | 0.933 / 3.83e-5 |
| 0.001 | 6 | 0.503 / 0.00502 | 0.612 / 0.00608 | 0.64 / 0.00784 | 0.497 / 3.53e-4 | 0.633 / 0.00266 |
| 0.001 | 22 | 0.0938 / 0.00579 | 0.195 / 0.0105 | 0.258 / 0.0495 | 0.0928 / 0.00111 | 0.254 / 0.0426 |
| 0.002 | 0 | 0.89 / 0.00946 | 0.89 / 0.00946 | 0.89 / 0.00946 | 0.87 / 1.54e-4 | 0.87 / 1.54e-4 |
| 0.002 | 6 | 0.253 / 0.0111 | 0.378 / 0.0158 | 0.42 / 0.0241 | 0.248 / 0.00146 | 0.409 / 0.0123 |
| 0.002 | 22 | 0.0089 / 0.0144 | 0.0398 / 0.0355 | 0.105 / 0.22 | 0.0087 / 0.00475 | 0.0998 / 0.206 |
| 0.005 | 0 | 0.75 / 0.0251 | 0.75 / 0.0251 | 0.75 / 0.0251 | 0.708 / 0.00102 | 0.708 / 0.00102 |
| 0.005 | 6 | 0.0333 / 0.0364 | 0.0981 / 0.0739 | 0.15 / 0.143 | 0.0313 / 0.0104 | 0.135 / 0.106 |
| 0.005 | 22 | 8.20e-6 / 0.0488 | 5.17e-4 / 0.218 | 0.0631 / 0.488 | 7.50e-6 / 0.0222 | 0.053 / 0.485 |

Matched acceptance, `eps_LRB-D / eps_gap` (> 1: the gap rule is better):

| p | m | unif=1 / mld2 | unif=5 / mld2 | const=1 / mld2 | unif=1 / mld5 | unif=1+final / mld5 | const=1+final / mld5 |
|---|---|---|---|---|---|---|---|
| 0.0005 | 0 | 0.993 [0.983, 1.01] | 0.993 [0.983, 1.01] | 0.993 [0.983, 1.01] | 248 [199, 325] | 1.05 [0.866, 1.4] | 1.05 [0.866, 1.4] |
| 0.0005 | 6 | 1.01 [0.997, 1.02] | 1.09 [1.07, 1.1] | 1.23 [1.22, 1.25] | 27.6 [25.2, 30.5] | 0.99 [0.921, 1.06] | 3.53 [3.3, 3.82] |
| 0.0005 | 22 | 0.988 [0.973, 1] | 1.29 [1.26, 1.31] | 3.7 [3.62, 3.77] | 9.23 [8.65, 9.85] | 0.994 [0.946, 1.05] | 12.2 [11.8, 12.6] |
| 0.001 | 0 | 1 [0.992, 1.01] | 1 [0.992, 1.01] | 1 [0.992, 1.01] | 104 [94.5, 115] | 0.998 [0.92, 1.08] | 0.998 [0.92, 1.08] |
| 0.001 | 6 | 1.01 [0.997, 1.02] | 1.15 [1.14, 1.16] | 1.43 [1.42, 1.45] | 14.4 [13.8, 15.2] | 1.03 [0.992, 1.06] | 3.54 [3.41, 3.66] |
| 0.001 | 22 | 0.97 [0.951, 0.99] | 1.53 [1.5, 1.55] | 6.22 [6.13, 6.33] | 5.24 [4.96, 5.56] | 1 [0.954, 1.06] | 13.5 [13.1, 13.9] |
| 0.002 | 0 | 0.998 [0.994, 1] | 0.998 [0.994, 1] | 0.998 [0.994, 1] | 52.9 [50.9, 55.5] | 1 [0.966, 1.04] | 1 [0.966, 1.04] |
| 0.002 | 6 | 0.975 [0.966, 0.985] | 1.27 [1.25, 1.28] | 1.82 [1.8, 1.84] | 7.36 [7.01, 7.7] | 0.971 [0.949, 0.992] | 3.77 [3.68, 3.87] |
| 0.002 | 22 | 0.925 [0.89, 0.963] | 1.86 [1.82, 1.9] | 6.86 [6.68, 6.99] | 2.95 [2.69, 3.22] | 0.972 [0.891, 1.06] | 10.1 [9.86, 10.4] |
| 0.005 | 0 | 0.999 [0.995, 1] | 0.999 [0.995, 1] | 0.999 [0.995, 1] | 21.5 [21, 22.1] | 1.02 [0.997, 1.04] | 1.02 [0.997, 1.04] |
| 0.005 | 6 | 0.984 [0.966, 1.01] | 1.59 [1.57, 1.61] | 2.41 [2.37, 2.46] | 3.36 [3.24, 3.49] | 0.97 [0.939, 0.999] | 3.89 [3.8, 3.99] |
| 0.005 | 22 | 1.92 [0.867, 7.06] | 2.33 [2.12, 2.56] | 1.37 [1.35, 1.38] | 1.4 [0.536, 5.81] | 0.641 [0.148, 2.99] | 1.46 [1.44, 1.48] |

Matched error rate, `acc_gap / acc_LRB-D`:

| p | m | unif=1 / mld2 | unif=5 / mld2 | const=1 / mld2 | unif=1 / mld5 | unif=1+final / mld5 | const=1+final / mld5 |
|---|---|---|---|---|---|---|---|
| 0.0005 | 0 | 0 [0, 1.01] | 0 [0, 1.01] | 0 [0, 1.01] | 1.02 [1.02, 1.02] | 1 [0, 1.01] | 1 [0, 1.01] |
| 0.0005 | 6 | 1.06 [0, 1.09] | 1.07 [1.07, 1.08] | 1.09 [1.08, 1.09] | 1.28 [1.28, 1.28] | 0.961 [0.463, 1.03] | 1.09 [1.09, 1.1] |
| 0.0005 | 22 | 0 [0, 1.17] | 1.29 [1.28, 1.3] | 1.49 [1.48, 1.5] | 2.08 [2.07, 2.09] | 0.891 [0.247, 1.12] | 1.51 [1.5, 1.51] |
| 0.001 | 0 | 1.01 [0, 1.02] | 1.01 [0, 1.02] | 1.01 [0, 1.02] | 1.05 [1.05, 1.05] | 0 [0, 1.01] | 0 [0, 1.01] |
| 0.001 | 6 | 1.04 [0, 1.1] | 1.14 [1.14, 1.15] | 1.18 [1.17, 1.18] | 1.58 [1.58, 1.59] | 1.03 [0.92, 1.06] | 1.2 [1.19, 1.2] |
| 0.001 | 22 | 0 [0, 0] | 1.68 [1.66, 1.7] | 2.24 [2.22, 2.26] | 3.57 [3.52, 3.62] | 1.03 [0, 1.23] | 2.3 [2.27, 2.32] |
| 0.002 | 0 | 0 [0, 1.01] | 0 [0, 1.01] | 0 [0, 1.01] | 1.09 [1.09, 1.09] | 1 [0, 1.01] | 1 [0, 1.01] |
| 0.002 | 6 | 0.412 [0, 0.623] | 1.3 [1.28, 1.31] | 1.4 [1.39, 1.41] | 2.27 [2.24, 2.29] | 0.806 [0.652, 0.943] | 1.43 [1.42, 1.44] |
| 0.002 | 22 | 0.135 [0, 0.409] | 2.92 [2.85, 2.98] | 4.76 [4.69, 4.83] | 8.17 [7.74, 8.56] | 0 [0, 1.26] | 4.86 [4.79, 4.94] |
| 0.005 | 0 | 0 [0, 1.02] | 0 [0, 1.02] | 0 [0, 1.02] | 1.25 [1.25, 1.25] | 1.01 [0, 1.02] | 1.01 [0, 1.02] |
| 0.005 | 6 | 0.894 [0, 1.05] | 1.98 [1.95, 2.03] | 2.37 [2.32, 2.41] | 5.08 [4.94, 5.22] | 0.812 [0.201, 0.992] | 2.58 [2.54, 2.63] |
| 0.005 | 22 | 1.87 [0.887, 14.2] | 20.6 [18.5, 22.6] | 15.8 [15.8, 15.9] | 18.6 [0, 81.5] | 0.0784 [0, 7.56] | 18.9 [18.4, 18.9] |

Reading notes. A ratio of 1 means the LRB-D point is on the frontier. Values
slightly below 1 for unif=1 / mld2 and unif=1 + final / mld5 (down to 0.93)
come from the conservative in-bin interpolation: the exact identical-shot test
shows these policies *are* frontier points. For the same reason the matched-
error gain is 0 (no binned rule reaches their error) in several cells; read
those as "on the frontier". At p = 5e-3, m = 22 unif=1 accepts 8e-6 of the
shots and the intervals are wide.

Cross-check of L0 section 7 (independent code; ratio vs the mld5 frontier):

| p | m | policy | L0 | this work [95% CI] |
|---|---|---|---|---|
| 1e-3 | 6 | unif=1 + final | 1.03 | 1.03 [0.99, 1.06] |
| 1e-3 | 22 | unif=1 + final | 1.00 | 1.00 [0.95, 1.06] |
| 1e-3 | 22 | unif=1 (data-2) | 5.2 | 5.24 [4.96, 5.56] |
| 1e-3 | 22 | const=1 | 15 | 15.3 [14.8, 15.7] |
| 1e-3 | 22 | unif=4 | 4.7 | 4.70 [4.53, 4.88]; vs mld2: 1.40 [1.37, 1.42] |
| 2e-3 | 14 | unif=1 + final | 1.00 | 1.00 [0.96, 1.05] |
| 2e-3 | 22 | const=1 | 10 | 10.2 [9.8, 10.5] |
| 5e-3 | 6 | unif=1 + final | 0.97 | 0.97 [0.94, 1.00] |
| 5e-3 | 14 | const=1 | 2.6 | 2.58 [2.53, 2.64] |

### 2.4 Fixed thresholds: F(m), acceptance and fitted decays

`fits_threshold.csv`, `fits_lrbd.csv`, `fits_rb.csv`,
`figures/E/F_and_acceptance_vs_m.png`, `figures/E/fits_vs_p.png`. Fitted f
(offset 0) +- bootstrap std, acceptance at m = 22 in brackets; physical RB with
offset 0 and with the repo's free offset:

| p | MLD (t=0) | gap t=2 | gap t=4 | gap t=2, mld2 | unif=1 | unif=1+final | const=1 | unif=5 | RB 5p (a=0) | RB p (a=0) | RB 5p (free a) | RB p (free a) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | 0.987811 +- 1.4e-4 (1) | 0.999104 +- 6.1e-6 (0.74) | 0.999798 +- 9.8e-7 (0.62) | 0.99911 +- 6.6e-6 (0.73) | 0.999972 +- 1.8e-6 (0.31) | 0.999975 +- 4.5e-7 (0.3) | 0.999452 +- 2.2e-5 (0.49) | 0.999872 +- 2.4e-6 (0.44) | 0.998301 +- 3.4e-5 | 0.999659 +- 6.7e-6 | 0.994225 +- 0.0031 | 0.995944 +- 0.0028 |
| 0.001 | 0.974989 +- 2.8e-4 (1) | 0.997775 +- 1.1e-5 (0.52) | 0.99954 +- 3.1e-6 (0.32) | 0.997828 +- 1.2e-5 (0.5) | 0.999889 +- 2.9e-6 (0.094) | 0.9999 +- 1.7e-6 (0.093) | 0.998087 +- 1.4e-4 (0.26) | 0.999453 +- 7.7e-6 (0.2) | 0.996605 +- 6.7e-5 | 0.999319 +- 1.3e-5 | 0.992626 +- 0.0031 | 0.995059 +- 0.003 |
| 0.002 | 0.947727 +- 5.3e-4 (1) | 0.994842 +- 4.4e-5 (0.2) | 0.998921 +- 1.4e-5 (0.049) | 0.995108 +- 5.5e-5 (0.18) | 0.999493 +- 1.2e-5 (0.0089) | 0.999568 +- 8.5e-6 (0.0087) | 0.991105 +- 7.8e-4 (0.11) | 0.997489 +- 3.9e-5 (0.04) | 0.993223 +- 1.3e-4 | 0.99864 +- 2.7e-5 | 0.988986 +- 0.0032 | 0.994693 +- 0.003 |
| 0.005 | 0.857563 +- 0.0011 (1) | 0.986496 +- 3.0e-4 (0.0013) | 0.996299 +- 5.8e-5 (0) | 0.988143 +- 2.0e-4 (6.7e-4) | 0.995976 +- 1.0e-4 (8.2e-6) | 0.996807 +- 2.2e-4 (7.5e-6) | 0.904218 +- 0.0039 (0.063) | 0.97795 +- 4.7e-4 (5.2e-4) | 0.983137 +- 3.3e-4 | 0.996605 +- 6.7e-5 | 0.979101 +- 0.0033 | 0.992565 +- 0.0031 |
| 0.01 | 0.702338 +- 0.0028 (1) | 0.967142 +- 6.6e-4 (0) | - +- - (0) | 0.970252 +- 9.9e-4 (0) | 0.976875 +- 5.9e-4 (0) | 0.982966 +- 4.0e-4 (0) | 0.780036 +- 0.0053 (0.063) | 0.880072 +- 0.0028 (5.4e-6) | 0.966555 +- 6.7e-4 | 0.993216 +- 1.3e-4 | 0.962916 +- 0.0034 | 0.98935 +- 0.0032 |

- MLD without rejection (t = 0) decays fast: at p = 2e-3, F(22) = 0.296 (mld5).
  Its f is the only one in the table that measures the logical channel of the
  whole sequence without selection; it is far below physical RB at either
  scale, because every check round adds about 5 x DEPOLARIZE1(2p) per data
  qubit (L0).
- Every postselected curve (gap t > 0, LRB-D) sits near f = 1 with an
  acceptance that falls exponentially in m; the f values are not comparable
  without the acceptance. The physical-RB free-offset fits are poorly
  conditioned here (a = 0.07-0.74 trades against f; bootstrap std ~3e-3); the
  offset-0 fits are stable.

### 2.5 Matched-acceptance fits

`matched_fits.csv`, `figures/E/matched_operating_points.png`. Gap rule with the
same acceptance as the policy at every depth, fitted like LRB-D:

| p | policy / curve | f_LRB-D | f_gap (same acceptance per m) | difference | F(22) LRB-D -> gap |
|---|---|---|---|---|---|
| 0.001 | const_1 / mld2 | 0.998087 +- 1.4e-4 | 0.999707 +- 5.5e-6 | 0.00162 [0.00136, 0.00189] | 0.901 -> 0.984 |
| 0.001 | unif_5 / mld2 | 0.999453 +- 7.7e-6 | 0.999803 +- 5.6e-6 | 3.50e-4 [3.35e-4, 3.65e-4] | 0.979 -> 0.986 |
| 0.001 | unif_1 / mld2 | 0.999889 +- 2.9e-6 | 0.999881 +- 3.9e-6 | -8.07e-6 [-1.46e-5, -1.45e-6] | 0.988 -> 0.988 |
| 0.001 | const_1_final / mld5 | 0.998759 +- 1.1e-4 | 0.999738 +- 4.7e-6 | 9.80e-4 [7.66e-4, 0.00122] | 0.915 -> 0.994 |
| 0.001 | unif_1_final / mld5 | 0.9999 +- 1.7e-6 | 0.999899 +- 2.2e-6 | -2.12e-7 [-2.80e-6, 2.47e-6] | 0.998 -> 0.998 |
| 0.002 | const_1 / mld2 | 0.991105 +- 7.8e-4 | 0.998274 +- 6.0e-5 | 0.00717 [0.00572, 0.00871] | 0.56 -> 0.936 |
| 0.002 | unif_5 / mld2 | 0.997489 +- 3.9e-5 | 0.999105 +- 2.3e-5 | 0.00162 [0.00156, 0.00169] | 0.929 -> 0.962 |
| 0.002 | unif_1 / mld2 | 0.999493 +- 1.2e-5 | 0.999445 +- 1.8e-5 | -4.80e-5 [-7.65e-5, -2.62e-5] | 0.971 -> 0.969 |
| 0.002 | const_1_final / mld5 | 0.993985 +- 6.6e-4 | 0.998651 +- 3.6e-5 | 0.00467 [0.00351, 0.00604] | 0.588 -> 0.959 |
| 0.002 | unif_1_final / mld5 | 0.999568 +- 8.5e-6 | 0.999558 +- 9.7e-6 | -1.01e-5 [-1.81e-5, 1.34e-6] | 0.99 -> - |
| 0.005 | const_1 / mld2 | 0.904218 +- 0.0039 | 0.969763 +- 0.0014 | 0.0655 [0.0576, 0.0736] | 0.0248 -> 0.286 |
| 0.005 | unif_5 / mld2 | 0.97795 +- 4.7e-4 | 0.991752 +- 2.4e-4 | 0.0138 [0.0131, 0.0147] | 0.563 -> 0.81 |
| 0.005 | unif_1 / mld2 | 0.995976 +- 1.0e-4 | 0.995553 +- 9.6e-5 | -4.24e-4 [-6.53e-4, -2.28e-4] | 0.912 -> 0.9 |
| 0.005 | const_1_final / mld5 | 0.914867 +- 0.004 | 0.979598 +- 0.0011 | 0.0647 [0.0574, 0.073] | 0.0293 -> 0.336 |
| 0.005 | unif_1_final / mld5 | 0.996807 +- 2.2e-4 | 0.996641 +- 1.7e-4 | -1.66e-4 [-3.11e-4, -9.04e-5] | 0.95 -> - |

The gap rule raises f and F(22) for every sparser policy (largest for
const=1, whose late F collapses) but not for unif=1 or unif=1 + final, which
are already on the frontier (the small negative differences are the binning
artifact of section 2.3; F(22) is blank where a common threshold leaves a
sequence with no accepted shot).

### 2.6 Acceptance x F (VN)

`eff_fidelity.csv`: for each (p, m) the best acceptance x F over all 44 LRB-D
policies and all gap thresholds never exceeds plain MLD's F (0 of 126; the
closest is 0.9364 vs 0.9385 at p = 5e-4, m = 4, gap t = 0.5).

### 2.7 Hybrid rules (fresh identical shots)

`hybrid_points.csv`, `hybrid_fits.csv`, `hybrid_audit.csv`,
`figures/E/hybrids_pareto.png`, `figures/E/hybrids_F_and_acceptance.png`.
2e5 fresh shots per circuit, 30 sequences, 9 depths, p in {5e-4, 1e-3, 2e-3,
5e-3, 7.5e-3, 1e-2}; every method scored on the same shots.

**First-order budget** (`single_fault_audit`, mean over sequences, p = 2e-3,
units of p). "wrong": probability of single faults the rule accepts with a
wrong logical output; "rejected": single-fault probability the rule rejects
(first-order acceptance ~ exp(-rejected)). For reference unif=1 + final rejects
every detectable single fault.

| rule | m = 0: wrong / rejected | m = 6 | m = 22 |
|---|---|---|---|
| tl_d5 | 0p / 33.56p | 0p / 409.3p | 0p / 1436p |
| tlq1_d5 | 0p / 29.1p | 2.45p / 354.3p | 8.84p / 1246p |
| lut0_d5 | 0p / 25.24p | 0p / 240.1p | 0p / 792.1p |
| lut10_d5 | 0.133p / 23.24p | 2.09p / 185p | 7.65p / 579.8p |
| tl_d2 | 4.5p / 42.15p | 4.5p / 421.4p | 4.5p / 1448p |
| tlq1_d2 | 4.5p / 39.29p | 6.96p / 368p | 13.3p / 1260p |
| lut0_d2 | 4.5p / 37.29p | 4.5p / 255.6p | 4.5p / 807.5p |
| lut10_d2 | 4.64p / 35.29p | 6.59p / 200.6p | 12.2p / 595.2p |

`tl` and `lut0` accept no single fault with a wrong output when the final
data detectors are used; with the data-2 readout they inherit only the
undetectable floor u = 4.5p. `tlq1` and `lut10` accept first-order errors that
grow with depth (the fold-gate faults of section 4 for `tlq1`).

**Identical-shot comparison** (eps +- binomial SE; "gap, same acceptance" is
the matched gap rule on the same shots with the same information; paired z
from the per-circuit paired differences; acceptance gain of the pooled gap
rule at the rule's error rate, with 95% sequence-bootstrap interval):

| p | m | rule | acceptance | eps (rule) | eps (MLD on same set) | eps gap, same acceptance | paired z | acc gain at same eps [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 0 | lrbd_unif_1 | 0.943 | 0.00465 +- 2.9e-5 | 0.00465 | 0.00465 | - | 0 [0, 1] |
| 0.001 | 0 | tl_d2 | 0.958 | 0.00467 +- 2.8e-5 | 0.00467 | 0.00467 | - | 1 [1, 1] |
| 0.001 | 0 | lut0_d2 | 0.964 | 0.005 +- 2.9e-5 | 0.005 | 0.0049 | 14.6 | 1 [1, 1] |
| 0.001 | 0 | lrbd_unif_1_final | 0.933 | 3.64e-5 +- 2.6e-6 | 3.64e-5 | 3.64e-5 | - | 1 [0, 1] |
| 0.001 | 0 | tl_d5 | 0.967 | 7.62e-5 +- 3.6e-6 | 7.62e-5 | 7.62e-5 | - | 1 [1, 1] |
| 0.001 | 0 | tlq1_d5 | 0.971 | 1.10e-4 +- 4.3e-6 | 1.10e-4 | 1.07e-4 | 1.75 | 1 [1, 1] |
| 0.001 | 0 | lut0_d5 | 0.975 | 2.01e-4 +- 5.9e-6 | 2.01e-4 | 1.74e-4 | 11.1 | 1 [1, 1] |
| 0.001 | 0 | lut10_d5 | 0.977 | 4.13e-4 +- 8.4e-6 | 4.13e-4 | 3.65e-4 | 16.8 | 1 [1, 1] |
| 0.001 | 6 | lrbd_unif_1 | 0.503 | 0.005 +- 4.1e-5 | 0.005 | 0.005 | - | 0.877 [0, 1.02] |
| 0.001 | 6 | tl_d2 | 0.645 | 0.00565 +- 3.8e-5 | 0.00565 | 0.00551 | 9.88 | 1.02 [1.02, 1.03] |
| 0.001 | 6 | lut0_d2 | 0.754 | 0.0114 +- 5.0e-5 | 0.0114 | 0.00786 | 104 | 1.06 [1.06, 1.06] |
| 0.001 | 6 | lrbd_unif_1_final | 0.497 | 3.36e-4 +- 1.1e-5 | 3.36e-4 | 3.36e-4 | - | 1 [1, 1.01] |
| 0.001 | 6 | tl_d5 | 0.651 | 8.84e-4 +- 1.5e-5 | 8.84e-4 | 8.38e-4 | 6.29 | 1.01 [1.01, 1.02] |
| 0.001 | 6 | tlq1_d5 | 0.685 | 0.00366 +- 3.0e-5 | 0.00366 | 0.00107 | 95.9 | 1.13 [1.12, 1.14] |
| 0.001 | 6 | lut0_d5 | 0.763 | 0.00609 +- 3.6e-5 | 0.00609 | 0.00297 | 101 | 1.06 [1.06, 1.06] |
| 0.001 | 6 | lut10_d5 | 0.799 | 0.00867 +- 4.2e-5 | 0.00866 | 0.00525 | 108 | 1.04 [1.03, 1.04] |
| 0.001 | 22 | lrbd_unif_1 | 0.094 | 0.00587 +- 1.0e-4 | 0.00587 | 0.00587 | - | 0 [0, 1.2] |
| 0.001 | 22 | tl_d2 | 0.219 | 0.00786 +- 7.7e-5 | 0.00786 | 0.00728 | 13.3 | 1.15 [1.13, 1.17] |
| 0.001 | 22 | lut0_d2 | 0.395 | 0.0269 +- 1.1e-4 | 0.0269 | 0.0146 | 137 | 1.26 [1.26, 1.27] |
| 0.001 | 22 | lrbd_unif_1_final | 0.093 | 0.00121 +- 4.7e-5 | 0.00121 | 0.00121 | - | 1.07 [0.18, 1.18] |
| 0.001 | 22 | tl_d5 | 0.221 | 0.00297 +- 4.7e-5 | 0.00297 | 0.00257 | 12.5 | 1.1 [1.09, 1.11] |
| 0.001 | 22 | tlq1_d5 | 0.262 | 0.0125 +- 8.8e-5 | 0.0125 | 0.00334 | 108 | 1.64 [1.62, 1.66] |
| 0.001 | 22 | lut0_d5 | 0.4 | 0.0216 +- 9.4e-5 | 0.0216 | 0.00976 | 140 | 1.26 [1.26, 1.27] |
| 0.001 | 22 | lut10_d5 | 0.477 | 0.0305 +- 1.0e-4 | 0.0305 | 0.0177 | 153 | 1.15 [1.15, 1.15] |
| 0.002 | 0 | lrbd_unif_1 | 0.89 | 0.00946 +- 4.2e-5 | 0.00946 | 0.00946 | - | 1 [0, 1] |
| 0.002 | 0 | tl_d2 | 0.919 | 0.00953 +- 4.1e-5 | 0.00953 | 0.00953 | - | 1 [1, 1] |
| 0.002 | 0 | lut0_d2 | 0.93 | 0.0108 +- 4.4e-5 | 0.0108 | 0.0103 | 35 | 1 [1, 1] |
| 0.002 | 0 | lrbd_unif_1_final | 0.871 | 1.53e-4 +- 5.4e-6 | 1.53e-4 | 1.53e-4 | - | 1 [0, 1] |
| 0.002 | 0 | tl_d5 | 0.934 | 3.12e-4 +- 7.5e-6 | 3.12e-4 | 3.12e-4 | - | 1 [1, 1] |
| 0.002 | 0 | tlq1_d5 | 0.942 | 4.51e-4 +- 8.9e-6 | 4.51e-4 | 4.28e-4 | 6.79 | 1 [1, 1] |
| 0.002 | 0 | lut0_d5 | 0.95 | 7.83e-4 +- 1.2e-5 | 7.83e-4 | 6.33e-4 | 24.7 | 1 [1, 1] |
| 0.002 | 0 | lut10_d5 | 0.954 | 0.00136 +- 1.5e-5 | 0.00136 | 0.00112 | 30.3 | 1 [1, 1] |
| 0.002 | 6 | lrbd_unif_1 | 0.254 | 0.0112 +- 8.5e-5 | 0.0112 | 0.0112 | - | 1 [0.859, 1.11] |
| 0.002 | 6 | tl_d2 | 0.406 | 0.0135 +- 7.4e-5 | 0.0135 | 0.013 | 14.9 | 1.06 [1.05, 1.07] |
| 0.002 | 6 | lut0_d2 | 0.549 | 0.0332 +- 9.9e-5 | 0.0332 | 0.0199 | 167 | 1.17 [1.17, 1.18] |
| 0.002 | 6 | lrbd_unif_1_final | 0.248 | 0.0015 +- 3.2e-5 | 0.0015 | 0.0015 | - | 0.948 [0.856, 1.05] |
| 0.002 | 6 | tl_d5 | 0.412 | 0.00355 +- 3.8e-5 | 0.00355 | 0.00335 | 10.1 | 1.03 [1.03, 1.04] |
| 0.002 | 6 | tlq1_d5 | 0.452 | 0.00953 +- 5.9e-5 | 0.00953 | 0.00415 | 105 | 1.23 [1.22, 1.25] |
| 0.002 | 6 | lut0_d5 | 0.557 | 0.0214 +- 7.9e-5 | 0.0214 | 0.0095 | 166 | 1.17 [1.17, 1.18] |
| 0.002 | 6 | lut10_d5 | 0.606 | 0.027 +- 8.5e-5 | 0.027 | 0.0145 | 173 | 1.12 [1.12, 1.13] |
| 0.002 | 22 | lrbd_unif_1 | 0.00893 | 0.0142 +- 5.1e-4 | 0.0142 | 0.0142 | - | 0.315 [0, 1.63] |
| 0.002 | 22 | tl_d2 | 0.0436 | 0.0215 +- 2.8e-4 | 0.0215 | 0.0193 | 10.7 | 1.37 [1.31, 1.44] |
| 0.002 | 22 | lut0_d2 | 0.133 | 0.0827 +- 3.1e-4 | 0.0825 | 0.0397 | 140 | 1.88 [1.86, 1.9] |
| 0.002 | 22 | lrbd_unif_1_final | 0.00873 | 0.0047 +- 3.0e-4 | 0.0047 | 0.0047 | - | 0.479 [0.248, 1.13] |
| 0.002 | 22 | tl_d5 | 0.0442 | 0.0112 +- 2.0e-4 | 0.0112 | 0.0097 | 9.47 | 1.26 [1.22, 1.31] |
| 0.002 | 22 | tlq1_d5 | 0.0605 | 0.0316 +- 2.9e-4 | 0.0315 | 0.0121 | 68.8 | 2.34 [2.29, 2.4] |
| 0.002 | 22 | lut0_d5 | 0.135 | 0.0718 +- 2.9e-4 | 0.0717 | 0.0295 | 146 | 1.88 [1.87, 1.9] |
| 0.002 | 22 | lut10_d5 | 0.184 | 0.089 +- 2.7e-4 | 0.0889 | 0.0455 | 164 | 1.58 [1.57, 1.58] |
| 0.005 | 6 | lrbd_unif_1 | 0.0332 | 0.0361 +- 4.2e-4 | 0.0361 | 0.0361 | - | 0.993 [0.917, 1.05] |
| 0.005 | 6 | tl_d2 | 0.0939 | 0.0495 +- 2.9e-4 | 0.0495 | 0.0457 | 19 | 1.19 [1.16, 1.21] |
| 0.005 | 6 | lut0_d2 | 0.193 | 0.13 +- 3.1e-4 | 0.129 | 0.0737 | 180 | 1.71 [1.69, 1.72] |
| 0.005 | 6 | lrbd_unif_1_final | 0.0312 | 0.00989 +- 2.3e-4 | 0.00989 | 0.00989 | - | 0.933 [0.794, 1.09] |
| 0.005 | 6 | tl_d5 | 0.0948 | 0.0217 +- 1.9e-4 | 0.0217 | 0.02 | 14.1 | 1.11 [1.09, 1.13] |
| 0.005 | 6 | tlq1_d5 | 0.116 | 0.0402 +- 2.4e-4 | 0.0402 | 0.0238 | 79 | 1.56 [1.52, 1.61] |
| 0.005 | 6 | lut0_d5 | 0.192 | 0.0977 +- 2.8e-4 | 0.0973 | 0.0438 | 189 | 1.72 [1.71, 1.73] |
| 0.005 | 6 | lut10_d5 | 0.225 | 0.112 +- 2.7e-4 | 0.111 | 0.0556 | 203 | 1.6 [1.59, 1.61] |

**Fits** (offset 0, +- bootstrap std):

| p | method | f +- boot std | F(0) | F(22) | acc(0) | acc(22) |
|---|---|---|---|---|---|---|
| 0.001 | lrbd_unif_1_final | 0.999898 +- 1.7e-6 | 0.9999 | 0.9976 | 0.933 | 0.093 |
| 0.001 | tl_d5 | 0.999733 +- 4.3e-6 | 0.9998 | 0.9941 | 0.967 | 0.221 |
| 0.001 | tlq1_d5 | 0.998841 +- 2.0e-5 | 0.9998 | 0.9751 | 0.971 | 0.262 |
| 0.001 | lut0_d5 | 0.998015 +- 1.6e-5 | 0.9996 | 0.9567 | 0.975 | 0.4 |
| 0.001 | lut10_d5 | 0.997187 +- 2.1e-5 | 0.9992 | 0.939 | 0.977 | 0.477 |
| 0.001 | gap_t2_mld5 | 0.997773 +- 1.1e-5 | 0.9992 | 0.9524 | 0.977 | 0.517 |
| 0.001 | gap_t4_mld5 | 0.999541 +- 2.5e-6 | 0.9997 | 0.9901 | 0.974 | 0.318 |
| 0.001 | gap_t0_mld5 | 0.974951 +- 2.7e-4 | 0.9782 | 0.5627 | 1 | 1 |
| 0.001 | lrbd_unif_1 | 0.999897 +- 6.4e-6 | 0.9907 | 0.9883 | 0.943 | 0.094 |
| 0.001 | tl_d2 | 0.999717 +- 7.1e-6 | 0.9907 | 0.9843 | 0.958 | 0.219 |
| 0.001 | lut0_d2 | 0.997931 +- 1.8e-5 | 0.99 | 0.9462 | 0.964 | 0.395 |
| 0.002 | lrbd_unif_1_final | 0.999563 +- 9.1e-6 | 0.9997 | 0.9906 | 0.871 | 0.00873 |
| 0.002 | tl_d5 | 0.998973 +- 1.5e-5 | 0.9994 | 0.9776 | 0.934 | 0.0442 |
| 0.002 | tlq1_d5 | 0.997027 +- 4.1e-5 | 0.9991 | 0.9368 | 0.942 | 0.0605 |
| 0.002 | lut0_d5 | 0.993053 +- 6.0e-5 | 0.9984 | 0.8563 | 0.95 | 0.135 |
| 0.002 | lut10_d5 | 0.991248 +- 6.5e-5 | 0.9973 | 0.8217 | 0.954 | 0.184 |
| 0.002 | gap_t2_mld5 | 0.994836 +- 4.0e-5 | 0.9984 | 0.8991 | 0.952 | 0.199 |
| 0.002 | gap_t4_mld5 | 0.998872 +- 2.0e-5 | 0.999 | 0.9793 | 0.945 | 0.0496 |
| 0.002 | gap_t0_mld5 | 0.947756 +- 5.2e-4 | 0.9562 | 0.2967 | 1 | 1 |
| 0.002 | lrbd_unif_1 | 0.999497 +- 2.1e-5 | 0.9811 | 0.9715 | 0.89 | 0.00893 |
| 0.002 | tl_d2 | 0.998831 +- 2.3e-5 | 0.9809 | 0.957 | 0.919 | 0.0436 |
| 0.002 | lut0_d2 | 0.992686 +- 6.1e-5 | 0.9784 | 0.8344 | 0.93 | 0.133 |

Reading notes:

- A "-" in the paired-z column means the rule and the matched gap rule
  accepted identical shots in every circuit (difference and variance exactly
  zero). For unif=1 and unif=1 + final the pooled acceptance gain is computed
  from 0.05-wide pooled bins with one common threshold and is not reliable
  (values below 1, intervals reaching 1); the per-circuit identity is exact.
- `tl_d5` is on the frontier at m = 0 (identical accepted shots, so no z) and
  stays close to it: at p = 2e-3 the gap rule accepts 1.03x [1.03, 1.04] and
  1.26x [1.22, 1.31] as many shots at its error for m = 6 and 22, while
  `tl_d5` itself accepts 1.1x, 1.7x and 5.1x as many shots as unif=1 + final
  (m = 0, 6, 22).
- The MLD decision on each hybrid's accepted shots reproduces the hybrid's own
  correction to three digits. The hybrids lose to the gap rule by accepting
  the wrong shots, not by correcting them wrongly.
- `tlq1` adds little acceptance to `tl` (0.061 vs 0.044 at p = 2e-3, m = 22)
  at 2.8x its error: the fold-gate faults of section 4 make the qubit-1
  extension a bad trade.
- `lut0` and `lut10` accept 3-4x as many shots as `tl` at 6-8x its error; at
  their error the gap rule accepts 1.6-1.9x more (p = 2e-3, m = 22).
- The `_d2` versions inherit the data-2 floor u: e.g. `tl_d2` at p = 2e-3,
  m = 0 has error 9.5e-3 against 3.1e-4 for `tl_d5`.
- Fitted f must again be read with the acceptance: at p = 2e-3, `tl_d5`
  gives f = 0.998973 at acceptance 0.044 (m = 22), between unif=1 + final
  (0.999563 at 0.0087) and the gap rule t = 2 (0.994836 at 0.20); physical RB
  gives 0.99322(13) at p_RB = 5p and 0.99864(3) at p_RB = p.

## 3. Claims

| # | Claim | Label | Evidence |
|---|---|---|---|
| C1 | At fixed acceptance, accepting by decreasing \|LLR\| with the MLD decision minimises the accepted logical error among all (randomised) rules that use the same detection events. | VA; LIT | Section 1.2 (fractional knapsack); Seip et al. 2026 Thm II.1; Chow 1970. Exhaustive enumeration at depths 0-2: 300 random rules and the LRB-D policies per depth all lie on or above the exact frontier (`test_gap_threshold_is_neyman_pearson_optimal`). |
| C2 | The exact-MLD \|LLR\| is calibrated against `1/(1+e^\|LLR\|)`. | VN | 2 of 4,979 per-depth bins outside the band at > 3 sigma, 0 at > 4 sigma, p = 1e-4 to 5e-2 (`calibration_summary.csv`). |
| C3 | No LRB-D policy beats the gap frontier built from the same information. | VN | Grid: 0 of 10,715 comparisons below the optimistic bound by > 3 sigma (min z = -1.90). Identical fresh shots: 0 of 7,025 paired comparisons with z < -2 (min z = -1.41). |
| C4 | unif=1 (data-2) is exactly the mld2 gap rule at its acceptance, and unif=1 + final exactly the mld5 gap rule, in every circuit. | VN | Same accepted shots in 1,414 of 1,414 circuits for each pair (six p) (`np_test_exact.csv`, rows `identity:`). Reason: the all-clean syndrome has the largest \|LLR\| of every circuit. |
| C5 | With the data-2 readout every syndrome has \|LLR\| <= log((1-u)/u) and every rule has error >= u; u = 4.53p at small p (exact DEM value). | VA, VN | Section 1.2; the highest occupied mld2 bin lies at the cap for all 14 p (section 2.1). |
| C6 | Sparser LRB-D policies are dominated at equal acceptance: vs the information-matched mld2 frontier by 0.99-1.17x (unif=2), 1.01-1.88x (unif=5), 1.04-6.9x (const=1) for m >= 6, p <= 2e-3; vs mld5 by 3-15x. | VN | `matched_points.csv`, `np_test_grid.csv`; bootstrap intervals in section 2.3. |
| C7 | The lead's L0 section 7 numbers (all nine rows) are reproduced by independent code. | VN | Section 2.3 cross-check table. |
| C8 | At equal error the gap rule accepts 1.5x [1.48, 1.50], 2.2x [2.22, 2.26] and 4.8x [4.69, 4.83] more shots than const=1 at m = 22 for p = 5e-4, 1e-3, 2e-3 (mld2). | VN | `matched_points.csv`. |
| C9 | Reading all five data qubits lowers the error at unif=1's acceptance by 248x [199, 325] (m = 0) to 9.2x [8.7, 9.9] (m = 22) at p = 5e-4 and 53x to 2.95x at p = 2e-3. | VN | `matched_points.csv` (unif=1 / mld5). |
| C10 | A fixed threshold t gives F(m) >= tanh(t/2) at every depth, so the offset-0 fit gives f -> 1 as t grows; f of a postselected curve is not a gate-error estimate. | VA, VN | Section 1.2; 793/793 grid cells satisfy the bound (closest z = +0.42). |
| C11 | Charging rejection as a random guess (acceptance x F), plain MLD without rejection is optimal. | VA, VN | Section 1.2; 0 of 126 (p, m) points beaten (`eff_fidelity.csv`). |
| C12 | Qubit-1 syndromes are first-order ambiguous at circuit level: DEPOLARIZE2 after SWAP(2,3)/CZ(2,3) gives X2X3/Z2Z3/Y2Y3 = X1/Z1/Y1 times a logical (~2p/15 per gate). | VA, VN | Pauli algebra; `single_fault_audit`; stim `explain_detector_error_model_errors` (test). |
| C13 | The decoder-free time-like-pair rule `tl_d5` and the pure lookup `lut0_d5` accept no single fault with a wrong logical output; `tlq1` and `lut10` do. | VN | `single_fault_audit` on every DEM mechanism of all 1,620 fresh-run circuits (`hybrid_audit.csv`) and in tests. |
| C14 | `tl_d5` is on the frontier at m = 0 and within 1.26x [1.22, 1.31] of the frontier's acceptance at its error up to m = 22 (p = 2e-3), while accepting 1.1-5.1x as many shots as unif=1 + final. | VN | `hybrid_points.csv`, paired identical-shot tests. |
| C15 | `tlq1`, `lut0`, `lut10` accept more shots but have 2.0-2.6x the gap rule's error at the same acceptance by m = 22 (p = 2e-3, paired z > 60). | VN | `hybrid_points.csv`. |
| C16 | The hybrids' excess error comes from which shots they accept, not from their correction (MLD on the same accepted shots gives the same error to 3 digits). | VN | `hybrid_points.csv` column `error_rate_mld_on_accepted`. |
| C17 | Physical RB (offset 0): f = 0.99830(3), 0.99661(7), 0.99322(13), 0.98314(33), 0.96656(67) at p_RB = 5p and 0.999659(7), 0.999319(13), 0.99864(3), 0.99661(7), 0.99322(13) at p_RB = p, for p = 5e-4, 1e-3, 2e-3, 5e-3, 1e-2. The two independent runs at p_RB = 0.01 agree: 0.993223(133) and 0.993216(135). | VN | `fits_rb.csv` (free-offset fits also listed; poorly conditioned). |
| C18 | The repo's LRB and RB fits are reproduced exactly. | VN | `test_fixed_offset_fit_matches_repo`, `test_free_offset_fit_matches_repo` (identical f). |
| C19 | A two-fault lookup or a small cluster decoder would close most of the hybrids' gap to the frontier at m <= 6. | CONJ | Gap is second order (C13, C16). |

## 4. Bugs and model issues

- No bug found in the lead's model. Calibration (2.1), the exact NP tests
  (2.2) and the LRB-D/MLD identities (2.2, 2.7) are all consistent with
  `TransferMatrixMLD` returning the exact posterior and with
  `raw_syndromes` / `accept_mask` reproducing the policies on the same
  detection events.
- **CONTEXT.md caveat (not a code bug).** "Errors on qubit 1 are uniquely
  identifiable" holds for single-qubit data errors but not at circuit level.
  `DEPOLARIZE2(p)` after the fold gates' `SWAP(2,3)` (logical H) and
  `CZ(2,3)` (logical S) produces X2X3, Z2Z3, Y2Y3, whose syndromes equal those
  of X1, Z1, Y1, while `X2X3 = X1 X_L S_X1` and `Z2Z3 = Z1 Z_L S_Z1` differ from
  them by a logical (VA). Two of the three flip the measured logical in any
  frame, about 2p/15 per (2,3) gate (VN: `single_fault_audit`, and
  `explain_detector_error_model_errors` traces every such signature to
  `DEPOLARIZE2 2 3`; test
  `test_qubit1_rule_first_order_errors_are_fold_gate_pair_faults`).
- **Grid design limits paired tests.** LRB-D counts use all 1e6 shots but the
  gap histograms only the first 3e5, and the 0.25-wide bins cannot isolate
  the all-clean syndrome, so unif=1 cannot be placed exactly on the binned
  frontier. Storing per-circuit counts of unique |LLR| values (or 0.01 bins)
  for the decoded shots, and the LRB-D counts restricted to them, would make
  the grid's own comparisons exact.
- `GAP_EDGES` puts NaN and infinite |LLR| in the last bin; harmless here
  (no shot at p > 0 reaches |LLR| >= 15, and p = 0 gives +inf only).

## 5. Open questions

- The hybrids' remaining gap to the frontier is second order (multi-fault
  clusters). Would a two-fault lookup or a matching-based cluster decoder close
  it at hardware-friendly cost? (CONJ: yes for m <= 6.)
- A figure of merit that charges rejection (acceptance x F, or the
  Combes-Ferrie cost; LIT: J. Combes and C. Ferrie, Phys. Rev. A 92, 022117
  (2015), arXiv:1504.03740, fetched) makes plain MLD optimal. Which
  rejection cost corresponds to the experimental budget (shots per hour) the
  user cares about?
- The gap rule's f depends on t; a decoder-agnostic LRB estimator would fit
  acceptance(m) and F(m) jointly. Not attempted here.

## 6. Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$PWD/src
python -m pytest tests/test_qubit512_pareto.py            # 22 tests, ~1 min
# grid analysis (needs results/grid_main); ~10 min on 15 cores:
python studies/correction-decoders/qubit/scripts/pareto.py --n-boot 500 --workers 15
# fresh identical-shot run (~15 min on 24 cores) and its analysis:
python studies/correction-decoders/qubit/scripts/hybrid_rules.py run --raw <dir> \
    --p-indices 3,4,5,6,7,8 --shots 200000 --workers 24
python studies/correction-decoders/qubit/scripts/hybrid_rules.py analyse --raw <dir> --n-boot 500
```

On the shared allocation use `--cpu-bind=none -c 256` and at most 24 workers,
optionally pinned to idle cores (this work used `taskset -c 32-55` and
`taskset -c 32-46`):
`srun --jobid=<id> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c "...; taskset -c 32-55 python ..."`.
Plain `--overlap -c N` steps from different agents all land on the same few
CPUs and crawl. The `--raw` directory must be on a shared filesystem
(`/tmp` of the login node is not visible on the compute node).

## Files

- `src/lrb_qubit/acceptance_rules.py`: gap-rule curves and matching
  (`cumulative_counts`, `ParetoCurve`, `batch_*`), repo-identical fits and the
  sequence bootstrap, `MatchedGapScorer` (exact paired test), `SlotLayout`,
  cluster rules (`structural_tables`, `dem_lookup_tables`,
  `evaluate_cluster_rule`) and `single_fault_audit`.
- `studies/correction-decoders/qubit/scripts/pareto.py` (grid analysis, physical RB at p_RB = p),
  `studies/correction-decoders/qubit/scripts/hybrid_rules.py` (fresh identical-shot run and analysis).
- `tests/test_qubit512_pareto.py` (22 tests).
- `results/E/`: `calibration*.csv`, `pareto_curves.csv`, `np_test_grid.csv`,
  `np_test_summary.csv`, `fits_threshold.csv`, `fits_lrbd.csv`,
  `fits_rb.csv`, `matched_points.csv`, `matched_fits.csv`,
  `eff_fidelity.csv`, `hybrid_points.csv`, `hybrid_fits.csv`,
  `hybrid_audit.csv`, `np_test_exact.csv`, `np_test_exact_lrbd.csv`. The raw
  fresh-sample arrays (~7 MB of npz) are not kept; `hybrid_rules.py run`
  regenerates them from fixed seeds (Stim's seeded sampling repeats exactly
  for the same Stim version and CPU architecture).
- `figures/E/`: `pareto_all.pdf` (every p, all depths), `pareto_p{4,5,6,8,11}.png`,
  `calibration.png`, `F_and_acceptance_vs_m.png`, `fits_vs_p.png`,
  `matched_operating_points.png`, `hybrids_pareto.png`,
  `hybrids_F_and_acceptance.png`.
