# N1: Non-exponential and history-dependent behaviour in the LRB-D baseline

Agent N1, correction-decoders / qutrit-512 study. Read `00-brief.md` first.

Inputs: only the result CSVs of the baseline run, read-only:
`/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000/results/`
plus the run metadata files (depths, probabilities, shot and sequence counts) and
the pipeline's own fit table in `results/plots/`. No DEM, no simulation, no new
shots. p index 0 (p = 0) is skipped.

Script: `notes/scripts/N1_baseline_fits.py` (one core, 45 to 70 s on the login
node). Outputs: `results/baseline_nonmarkov/` (section 9). An earlier N1 run from
11:27 today is kept for provenance in `notes/_superseded/N1-2026-10-28T1127/` and
`results/baseline_nonmarkov/_superseded_2026-10-28T1127/`. This note replaces it.

Labels: [VA] verified analytically, [VN] verified numerically (on the baseline
CSVs with the script), [LIT] from literature fetched in this task (section 8),
[CONJ] conjecture. [code] marks a fact read from the source in this worktree,
[data] a fact read from the run folder.

Units. p is the SI1000 parameter. m is the depth (number of random forward
Cliffords). One unit of depth adds one forward Clifford, its check round and
reset, and one inverse Clifford. "Per Clifford" means per unit of m, as in the
pipeline: r = 1 - F = 2(1 - f)/3. Quadratic coefficients are quoted in units of
p^2 because every LRB-D logical failure needs at least two faults.

## 1. Short answer

1. Physical RB, the Markovian reference, is a single exponential within errors
   at all 28 noise values (RB with p_RB = p and RB_comparison with p_RB = 5p).
   No cell deviates. Goodness-of-fit p ranges from 0.33 to 0.999, the curvature
   z from -2.1 to +0.2, and the windowed rates are flat. [VN]
2. 434 of the 492 testable LRB cells (p, policy) deviate significantly from
   y = b f^m (alpha = 1e-3 per test; 428 also pass a Bonferroni level of
   3.4e-5). The exceptions are unif 2 at every p, unif 1 at p <= 2e-3, 32 of
   45 policies at p = 1e-4 and 13 of 45 at p = 2e-4. Table in section 4.1. [VN]
3. unif 1 (every round used) shows the predicted near-flat fidelity with fast,
   exponential acceptance loss. The error per Clifford is about 19 p^2
   (16.5 to 19.8 p^2 for m > 0, 5e-4 <= p <= 5e-3). Acceptance falls by a
   factor exp(-111.6 p) per unit depth, with no trend in m. At p = 1e-3 that is
   10.6 % of shots lost per step against a logical error of 2.0e-5 per
   Clifford. The decay is a single exponential for p <= 2e-3 (smallest test
   p-value 0.009). At p = 5e-3 and 7.5e-3 the only deviation is the m = 0
   point, 6.9 and 5.7 standard errors above the extrapolation of the m > 0
   exponential. [VN]
4. const 1 (terminal round only) shows the predicted quadratic-in-m infidelity at
   all 11 testable p. Curvature z rises from 6.4 (p = 1e-4) to 415 (5e-3).
   1 - y = b0 + b1 m + b2 m^2 fits at p <= 5e-4 with b2 = 35 +- 6, 31 +- 2 and
   38 +- 1 p^2, and b2 scales as p^(2.17 +- 0.06). [VN]
5. The predicted cause of that term, pairs of undetected errors inside the
   unchecked inverse half, is not what the data show. Such pairs enter unif 1
   in exactly the same way at order p^2, and unif 1 has no m^2 term:
   b2 = (-0.04 +- 0.22) p^2, below 0.33 p^2 at 95 %. So inverse-half pairs are
   at most about 1 % of the const 1 term. [VA for the logic, VN for the
   numbers] The term comes from faults in the m forward steps whose check
   outcomes const 1 ignores. unif k data give about 25 k^2 p^2 of infidelity per
   closed forward segment of k steps; one unchecked span of m steps then gives
   about 25 m^2 p^2, most of const 1's term. A cross-policy model puts about
   3 p^2 (10 %) in pairs of one forward-tail fault and one inverse-half fault.
   [VN for the coefficients, CONJ for reading them as fault pairs]
6. const 0 (no ancilla checks) follows a pure m^2 law at p <= 2e-3:
   b2 = (2.26 +- 0.04) p^2, scaling as p^(2.01 +- 0.02), with no positive
   linear term. It deviates at all 14 p (curvature z 13 to 80). [VN]
7. Every other deviation is tied to the check schedule, not to depth as such.
   The infidelity added per unit depth grows linearly with the check spacing of
   unif k, (25 +- 3) k p^2 for k = 1 to 6 and 10. Forward steps left
   unchecked before the inverse half add of order 100 to 200 p^2 each (at
   m = 2, 135 p^2 for one such step and 215 p^2 for two). An 8-parameter
   model in the segment lengths of the schedule describes all 192 distinct
   (policy, m > 0) points at p = 2e-4 to 0.47 standard errors rms, and when
   fitted to unif k alone it predicts every const c point to 0.57 standard
   errors rms. All 156 windows in which the fidelity rises significantly with
   m have a shrinking unchecked tail. [VN]
8. Acceptance has no history dependence at first order. Over all 44 ordinary
   policies, -ln(acceptance) = p (63.6 + 71.0 m + 40.5 n_used) fits 193
   distinct points at p = 1e-4 to 1.04 standard errors rms, where n_used is the
   number of used intermediate rounds. A persistent detectable fault is
   rejected once, whatever the schedule. A transient readout or ancilla fault is
   rejected only if it hits a used round. Acceptance becomes non-exponential
   only through n_used(m) and, for p >= 1e-3, through saturation (const 1 tends
   to 1/81, const 0 to 1/9). [VN; VA for the first-order argument]

For the study's question 2. The physical noise is i.i.d. Pauli noise, so it is
Markovian. The postselected logical process is not: its memory is the set of
detectable errors that sit on the data until a used check sees them. The memory
length equals the time since the last used check. Where the schedule renews
that memory at a fixed rate in m (unif 1, and unif 2 on this even depth grid),
the decay is a single exponential within the errors. Everywhere else, fitting a
single f mixes different effective protocols. [VN for the data statements, CONJ
for the memory interpretation]

## 2. Data and the pipeline fit

### 2.1 What the CSVs contain

- Five rows per file: probability, mean fidelity per depth, its standard
  deviation over the 30 Clifford sequences, mean rejected fraction, and its
  standard deviation. The per-sequence fidelity is |sum_k omega^k n_k| / N_acc,
  and the stored standard deviation is `np.std` with ddof = 0 [code:
  `LRBSimulationPipeline.lrb_counts_to_statistics`]. A sequence with no
  accepted shot makes the mean NaN. 233 of the 630 LRB files contain such NaN
  depths, all at large p [data].
- Shots: 1e6 per LRB circuit [data: `shots.txt`], 1e4 per RB circuit [code:
  `NORMAL_RB_SHOTS`; the RB_comparison timing log also shows 10 000].
- Nested sequences, reused at every p [code: `circuit_generator.generate_tests`].
  Each sequence index draws one 22-Clifford string, and the depth-m circuit
  uses its first m Cliffords. LRB and LRB_const0 share the string. RB draws its
  own string the same way. Circuits are built once per sequence and only the
  noise parameter is rewritten per p. The 30-sequence means are therefore
  positively correlated across depths and across p.
- Policies only choose which rounds are used, so all ordinary policies at one p
  come from the same shots. The script checks the consequences exactly [VN]:
  unif k at m < k and const c at m < c equal const 1 in all four rows (2828
  comparisons, largest difference 0), and all policies share the m = 0 point.
- The 30 depth-0 circuits are identical (byte-identical files for sequences 0,
  1, 7 and 29 at p index 4) [data], so the spread at m = 0 is pure shot noise.

### 2.2 The pipeline fit, reproduced

`src/lrb/lrb_plotting.py` fits LRB data as follows [code]:
- `_mask_invalid` drops depths whose mean or std is NaN;
- `_apply_ceiling_cut` keeps y <= 1.0 (LRB only; never active here because
  |sum omega^k P(k)| <= 1);
- `_safe_weights` uses w = 1/std^2, replaces std <= 0 by
  max(1e-3 x median(std), 1e-12), and gives NaN stds zero weight;
- `_fit_decay_parameter_fixed_a` scans f in [1e-8, 1] on 2001 points, solves b
  in closed form, and refines five times around the best point;
- F = (1 + (d - 1) f)/d with d = 3.

The notebook sets `a_fixed = 0`, `rb_a_fixed = 0`, `y_ceiling = 1`,
`min_fit_points_lrb = 2` [code: `Visualize LRB DEM Single Ancilla Stab Check
Results.ipynb`]. The script mirrors all of this. Against the pipeline's table
`unif_lrb_vs_rb_table_all_mixed_fits.csv` it reproduces all 308 unif LRB rows
(p > 0) and the RB_comparison columns with zero difference in f, b, chi2 and
number of points [VN]. `pipeline_fits.csv` gives f, b, F and r for all 658
cells (630 LRB, 14 RB, 14 RB_comparison).

Pipeline F at a few points [VN]:

| policy | p = 1e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 |
|---|---|---|---|---|---|
| unif 1 | 0.9999995 | 0.999995 | 0.999980 | 0.999914 | 0.99926 |
| unif 2 | 0.9999994 | 0.999992 | 0.999966 | 0.999855 | 0.99890 |
| const 1 | 0.999995 | 0.999881 | 0.999549 | 0.998248 | 0.97568 |
| const 0 | 1 (see below) | 0.999997 | 0.999990 | 0.999964 | 0.99979 |
| RB, p_RB = 5p | 0.999624 | 0.998124 | 0.996243 | 0.992483 | 0.98141 |

Two pipeline behaviours matter when the pipeline F is read [VN]:
- const 0 at p = 1e-4 gets F = 1 exactly. Its m = 0 and m = 2 points have
  std = 0 (y = 1 in all 30 sequences), receive the floor weight, and pin f at
  the bound f <= 1, although y(22) = 0.999994.
- The pipeline fits every non-NaN depth, including depths with a handful of
  accepted shots per sequence, where |.| sits near its noise floor. unif 1 at
  p = 0.01 is fitted on m = 0 to 10, and at m = 10 the sequences keep 8 shots
  on average.

### 2.3 Why the standard error, not the std

The pipeline weights each depth by 1/std^2, where std is the spread of one
sequence's fidelity. The plotted quantity is the mean over 30 sequences, whose
standard error is SEM = std/sqrt(30). Multiplying every sigma by the same
constant c multiplies the weighted sum of squares by 1/c^2. The minimiser does
not move, and the pipeline's zero-std floor (1e-3 x median) scales with the
data, so the statement holds exactly. f and b are therefore identical under
both weightings, and chi2 grows by exactly 30. [VA] The script confirms it
for all 658 fits: f is unchanged and the chi2 ratio is 30.000 [VN].

The pipeline chi2 is thus 30 times too small to be compared with a chi2
distribution. With std weights a single exponential would almost never be
rejected, even for const 1. The SEM-based chi2 tests whether the
population-mean curve, the average over random Clifford sequences, is a single
exponential. That is the quantity RB theory makes predictions about (section
3.4). Three approximations remain:
- the stored std has ddof = 0, so the SEM is about 1.7 % low and chi2 about
  3.4 % high (kept, as the task specifies std/sqrt(30));
- each variance is estimated from 30 values, so standardised residuals are
  closer to Student t with 29 dof than to a normal; at 3.3 standard errors the
  two-sided tail is about 2.5 times the normal one;
- depths are correlated through the nested sequences; section 4.8 tests this.

### 2.4 Usable points

The alternative fits and the windowed rates use only depths with finite mean
and std, at least 1000 accepted shots per sequence on average, and
y >= 10 x sqrt(pi/(4 N_acc)). The last term is the mean of |.| for a fully
mixed logical state, the Rayleigh floor; at the cut, the bias of |.| is about
0.2 standard errors. sigma = max(SEM, 1.5/(30 N_acc)); the floor only acts where
std = 0 (const 0 at m = 0, and at m = 2 for p = 1e-4; RB at m = 0), all of
which are exact points (y = 1). A cell needs 3 usable depths for model A,
4 for B, D and E, and 5 for C. 520 of the 658 cells are testable.

### 2.5 Shot noise at m = 0

At m = 0 the 30 circuits are identical, so the spread should be multinomial.
For Re(sum omega^k n_k / N) the per-shot variance is exactly (1 - F)(F + 1/2),
for any split of errors between k = 1 and 2 [VA]. Observed std divided by the
prediction, averaged over the 14 p values [VN, `m0_shot_noise_check.csv`]:

| quantity | observed ratio | expected for ddof = 0, n = 30 | pooled dispersion test |
|---|---|---|---|
| fidelity | 0.88 +- 0.04 | 0.975 | P(chi2 <= D) = 0.003 |
| rejected fraction | 1.06 +- 0.03 | 0.975 | P(chi2 >= D) = 0.007 |

The fidelity spread is about 10 % below the multinomial value and the rejection
spread about 9 % above the binomial value. I have no explanation; the DEM
sampler draws independent Bernoulli faults per shot [code:
`dem_simulation.sample_location_fault_vectors`]. Two 2.7-sigma effects with
opposite signs may be chance. In practice the SEM of shot-noise-dominated
points (unif 1, unif 2, RB at small p) carries a 10 to 15 % uncertainty, and so
do their z values. No conclusion below depends on it.

## 3. Methods

### 3.1 Models

All fitted to usable points with sigma = SEM, by weighted least squares
(profiled over the linear coefficients, refined with Levenberg-Marquardt):

- A: b f^m, the pipeline model. A0: A on m > 0 only.
- B: a + b f^m, free offset. It contains the linear model (f -> 1) and a
  "first point free" model (f -> 0, because 0^0 = 1).
- C: b1 f1^m + b2 f2^m, double exponential, fitted by a 2-D profile over
  (f1, f2) in [0, 1.3] plus local refinement. It contains A (b2 = 0) and
  B (f2 = 1), and the script enforces chi2_C <= min(chi2_A, chi2_B).
- D: b exp(-c m - d m^2). d > 0 means the decay accelerates with m.
- E: 1 - y = b0 + b1 m + b2 m^2, a quadratic infidelity, linear in its
  parameters. E0: E on m > 0 only.

Each fit reports chi2, dof, the goodness-of-fit p, AIC = chi2 + 2k,
AICc and BIC = chi2 + k ln n. Likelihood-ratio tests: A vs D (curvature,
1 dof), A vs B (offset, 1 dof) and A vs C (2 dof, nominal only: under b2 = 0
the parameter f2 is not identified).

### 3.2 Significance and form

A cell deviates if the smallest of three p-values (goodness of fit of A, A vs
D, A vs B) is below alpha = 1e-3. Over 520 cells and 1452 tests the Bonferroni
level is 3.4e-5; both flags are in `deviation_summary.csv`. The cells are far
from independent (shared shots, shared sequences), so these are per-cell
thresholds, not a calibrated family-wise error rate.

The form of a deviating cell:
- boundary (m = 0): A0 fits (p >= 1e-3) and y(0) lies more than 3.29 standard
  errors off the A0 extrapolation;
- accelerating or decelerating: the A vs D test fails, with d > 0 or d < 0;
- irregular: neither (for example 3 usable depths only, or a sawtooth);
- "+misfit" is added when even model D fails (p < 1e-3).

Size relative to the statistical error is given by chi2_A/dof, the signed
curvature z = sign(d) sqrt(chi2_A - chi2_D), the largest standardised
residual, and the miss of model A at the last usable depth, in absolute terms
and in standard errors.

### 3.3 Windowed rates and trend tests

For consecutive usable depths, f_i = (y_{i+1}/y_i)^(1/dm) and
r_i = 2(1 - f_i)/3; for acceptance acc = 1 - rejected,
a_i = (acc_{i+1}/acc_i)^(1/dm), with per-step hazard h_i = -ln(a_i). The error
bars assume independent depths. With the positive depth correlations of nested
sequences the true error of each window is smaller, so the bars are
conservative.

Adjacent windows share a depth, so the ln f_i are correlated. The trend test is
a GLS fit of ln f_i (or ln a_i) on the window midpoint with the exact
covariance D S D^T of the differences, plus a test of a constant rate. If
ln y = b0 + b1 m + b2 m^2, the window values are ln f_i = b1 + 2 b2 m_mid,
exactly linear in the midpoint. With the REML identity
S^-1 - S^-1 1 (1^T S^-1 1)^-1 1^T S^-1 = D^T (D S D^T)^-1 D,
the GLS window slope equals 2 b2 of the log-quadratic fit, and the
constant-rate chi2 equals the chi2 of the log-linear fit. [VA] The script
confirms both to a relative 1e-11 in all 466 cells with at least 4 depths [VN].
The windowed trend test is therefore the curvature test written on the rates;
the windows add where in m the change happens. Tests are run with and without
the first window (m = 0 to 2).

### 3.4 What "single exponential" should mean here

For the metric y = |sum_k omega^k P(k)|, a logical channel that is depolarising
after the Clifford twirl, rho -> lambda rho + (1 - lambda) I/3, gives
P(k) = lambda^m delta_k0 + (1 - lambda^m)/3 and therefore y = lambda^m exactly,
because sum_k omega^k = 0. With state preparation and measurement errors,
y = b f^m with a zero asymptote. [VA] This needs acceptance that does not depend
on the logical state. That holds for Pauli faults, whose syndromes depend only
on the fault pattern and not on the encoded state. [VA]

For mirrored single-qutrit RB with depolarising noise after each gate, the
channel commutes with every gate, so sequence s gives y_s = lambda^(N_s) with
N_s its total gate count. If n is the gate count of one random Clifford (its
inverse has the same count), i.i.d. draws give E[y] = (E[lambda^(2n)])^m: an
exact exponential with b = 1 and a = 0 [VA]. Fixing the RB offset to 0, as the
notebook does, is correct for this metric. The code-default free-offset fit
(`_fit_decay_parameter_free_a`) is degenerate when the decay over 22 steps is
small: it returns offsets from 0.97 (p_RB = 1e-4) down to 0.003 and inflates r
by up to 36 times [VN, `pipeline_fits.csv`, columns free_a_*].

Literature. RB estimates an average error rate under stated conditions that
include time- and gate-dependent noise (Magesan et al. [LIT 1]).
Gate-dependent Markovian noise adds a perturbation that itself decays
exponentially, so the decay converges to an exact exponential (Wallman
[LIT 2]). Deviations of the average sequence fidelity from the Markovian form
signal non-Markovian noise (Figueroa-Romero et al. [LIT 3]). In LRB, a hidden
register that records gate history can make the output non-exponential under
gate- and time-independent noise, and syndrome qubits coupled through a decoder
can play that role (Ceasura et al. [LIT 4]). Here the hidden register is the
undetected part of the data's Pauli frame, and postselection, not a decoder,
couples it to the output [CONJ]. LRB itself is defined in Combes et al.
[LIT 5]; the forward-plus-inverse structure is that of mirror circuits
(Proctor et al. [LIT 6]).

## 4. Results

### 4.1 Which cells deviate

| policy group | testable cells | deviating (1e-3) | Bonferroni | forms of the deviating cells | p range of deviation |
|---|---|---|---|---|---|
| unif 1 | 9 | 4 | 4 | boundary (m = 0) 2, irregular 2 (3 usable depths) | 5e-3 to 1.25e-2 |
| unif 2 | 10 | 0 | 0 | none | none |
| unif 3 | 10 | 7 | 7 | irregular 4, decelerating 2, boundary 1 | 1e-3 to 1.5e-2 |
| unif 4 to 10 | 77 | 63 | 62 | accelerating 36, irregular 25, decelerating 2 | from 5e-4 |
| unif 11 to 22 | 132 | 122 | 118 | accelerating 94, irregular 27, decelerating 1 | from 2e-4 (unif 15, 16 from 1e-4) |
| const 0 | 14 | 14 | 14 | accelerating 14 | all |
| const 1 | 11 | 11 | 11 | accelerating 10, irregular 1 (3 depths) | all |
| const 2 | 11 | 11 | 10 | accelerating 10, irregular 1 | all |
| const 3 to 6 | 42 | 34 | 34 | accelerating 16, irregular 11, decelerating 6, boundary 1 | from 2e-4 or 5e-4 |
| const 7 to 14 | 88 | 80 | 80 | accelerating 40, irregular 21, decelerating 19 | from 2e-4 |
| const 15 to 22 | 88 | 88 | 88 | accelerating 41, decelerating 37, irregular 10 | all |
| RB, p_RB = p | 14 | 0 | 0 | none | none |
| RB, p_RB = 5p | 14 | 0 | 0 | none | none |

[VN, `deviation_by_policy.csv`, `deviation_summary.csv`; "testable" means at
least 3 usable depths.] Untestable cells are high-p cells whose acceptance has
collapsed, for example unif 1 at p >= 0.015.

Size of the deviation: signed curvature z / chi2_A per dof, * = deviating [VN]:

| policy | p = 1e-4 | 2e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 |
|---|---|---|---|---|---|---|
| unif 1 | +0.7 / 1.13 | +0.5 / 0.21 | -1.8 / 2.19 | -1.4 / 2.07 | -2.5 / 2.45 | -5.5 / 16.3* |
| unif 2 | +0.8 / 1.04 | +0.4 / 0.28 | -1.5 / 1.93 | -1.2 / 1.27 | +0.2 / 0.65 | -0.6 / 1.92 |
| unif 3 | +1.0 / 1.06 | 0.0 / 0.52 | -1.4 / 2.37 | -1.0 / 13.1* | -1.4 / 24.3* | -3.9 / 108* |
| unif 4 | +0.7 / 1.08 | +0.2 / 0.50 | -2.9 / 4.83* | -2.2 / 11.5* | -2.2 / 16.0* | +6.1 / 73.9* |
| unif 6 | +1.1 / 1.23 | -0.2 / 1.13 | -0.4 / 5.96* | -1.5 / 17.3* | +0.2 / 47.4* | +12.3 / 187* |
| unif 11 | +0.5 / 0.65 | -1.9 / 6.53* | -1.9 / 53.1* | +10.9 / 123* | +37.5 / 556* | +126 / 3210* |
| unif 22 | +3.3 / 3.30 | +10.2 / 29.8* | +39.6 / 313* | +90 / 1290* | +175 / 4650* | +415 / 29600* |
| const 0 | +13.0 / 24.5* | +24.1 / 83.4* | +38.1 / 208* | +43.5 / 271* | +47.1 / 318* | +46.6 / 312* |
| const 1 | +6.4 / 7.08* | +17.0 / 41.4* | +52.8 / 401* | +104 / 1580* | +187 / 5370* | +415 / 29600* |
| const 2 | +3.4 / 2.76* | +8.1 / 9.74* | +24.9 / 90* | +52.9 / 404* | +103 / 1550* | +139 / 3480* |
| const 4 | +2.0 / 1.85 | +3.0 / 2.03 | +7.8 / 14.7* | +19.8 / 78.3* | +35.5 / 223* | +35.2 / 511* |
| const 10 | -0.3 / 2.03 | -0.4 / 11.3* | -7.3 / 111* | -9.3 / 360* | -6.9 / 857* | -41.2 / 1680* |
| const 22 | -4.8 / 28.5* | -7.1 / 180* | -30.2 / 1780* | -29.4 / 3280* | +3.2 / 5410* | +415 / 29600* |
| RB, p_RB = p | -0.5 / 0.66 | -0.4 / 0.36 | -0.7 / 0.17 | -0.9 / 0.67 | -2.1 / 1.15 | -1.2 / 0.44 |
| RB, p_RB = 5p | -0.9 / 0.61 | -0.7 / 0.48 | -0.2 / 0.33 | -0.7 / 0.65 | -0.9 / 0.69 | -0.9 / 0.35 |

In absolute terms at p = 1e-3, the single exponential overestimates y(22) by
1.2e-2 for const 1 (49 standard errors), 3.1e-3 for const 2 (25), 7.6e-4 for
const 0 (23), and underestimates it by 6.9e-3 for const 22 (90), whose checks
switch on at m = 22. For unif 1 and unif 2 the miss is +1.1 and +0.6 standard
errors. [VN] `curvature_map.png` shows the signed curvature of every cell, and
`residuals_p1e-3.png` the residual patterns.

### 4.2 Physical RB: the Markovian reference

- Goodness of fit of b f^m: p = 0.33 to 0.99 (RB) and 0.59 to 0.999
  (RB_comparison). Curvature z from -2.07 to -0.44 and from -0.95 to +0.17.
  AICc selects b f^m in 28 of 28 cells, BIC in 26 of 28. [VN]
- Windowed rates are flat (`windowed_rates.png`, lower right). At p_RB = 1e-3
  the per-window error is 7.9, 7.0, 7.9, 8.3, 7.7, 7.2, 5.8 and 8.9 x 1e-4
  (errors 0.4 to 2.2 x 1e-4). Windowed-trend z ranges over 0.44 to 2.07 (RB)
  and -0.19 to 0.95 (RB_comparison), and over -0.59 to 1.58 without the first
  window. [VN]
- All 14 RB trend values have the same sign. That is expected, not a signal:
  every p uses the same 30 RB sequences [code], so the 14 values are one
  realisation of the sequence sample.
- The pipeline error per Clifford is 0.77 p_RB (RB) and 0.75 p_RB
  (RB_comparison), linear in p_RB. [VN]

The windowed analysis passes the Markovian reference cleanly, so the LRB
deviations below are not artefacts of the method.

### 4.3 unif 1 and unif 2

Prediction tested: near-flat fidelity with fast-decaying acceptance.

| p | r per Clifford, m > 0 (p^2) | pipeline r (p^2) | acceptance hazard (p) | rejected per step / r | m = 0 offset (p^2), z |
|---|---|---|---|---|---|
| 2e-4 | 19.5 +- 8.3 | 15.1 | 111.64 +- 0.11 | 3.6e4 | -228 +- 236, -1.0 |
| 5e-4 | 16.5 +- 2.6 | 19.2 | 111.68 +- 0.11 | 1.1e4 | 124 +- 67, 1.9 |
| 1e-3 | 18.9 +- 1.4 | 20.3 | 111.68 +- 0.11 | 5.2e3 | 43 +- 25, 1.7 |
| 2e-3 | 18.6 +- 1.2 | 21.5 | 111.62 +- 0.11 | 2.3e3 | 43 +- 13, 3.3 |
| 5e-3 | 19.8 +- 1.7 | 29.8 | 111.45 +- 0.25 | 574 | 60 +- 9, 6.9 |
| 7.5e-3 | 20.5 +- 2.7 | 33.9 | 111.4 +- 0.3 | 297 | 64 +- 11, 5.7 |

[VN, `signatures_unif1_const1_const0.csv`]

- The fidelity is nearly flat, and the acceptance decays quickly and
  exponentially. Over m = 0 to 22 at p = 1e-3, y goes from 0.99250 to 0.99190
  while the acceptance goes from 0.939 to 0.081. The per-step rejection hazard
  is 111.6 p, the same from p = 1e-4 to 5e-3, with no trend in m (windowed
  trend |z| <= 0.7). The per-Clifford logical error, about 19 p^2, scales as
  p^2. The two rates differ by 3 to 4 orders of magnitude at p <= 1e-3. [VN]
- The decay is one exponential for p <= 2e-3. The largest residual at
  p = 1e-3 is 2.4 standard errors, and the smallest test p-value at
  p <= 2e-3 is 0.009. Windowed trends without m = 0 have |z| <= 0.9 at every
  p. [VN]
- At p >= 5e-3 the m = 0 point lies above the curve. The m > 0 points alone fit
  b f^m (p = 0.67 at 5e-3), and y(0) sits 60 p^2 above its extrapolation. The
  offset is constant in units of p^2 (43, 60, 64 at 2e-3, 5e-3, 7.5e-3), as
  for a two-fault effect. It inflates the pipeline rate by 50 to 65 % at
  p >= 5e-3 (29.8 against 19.8 p^2). At p = 0.01 and 0.0125 only m = 0, 2, 4
  are usable and the deviation cannot be separated from curvature. [VN] A
  plausible cause is that every circuit with m >= 1 contains the junction
  between the last forward round and the inverse half (late-round and reset
  faults that only the terminal round can see), while the m = 0 circuit does
  not. [CONJ]
- No m^2 term. E0 gives b2/p^2 = -0.50 +- 0.77, -0.09 +- 0.42, -0.03 +- 0.29
  and +0.81 +- 0.92 at p = 5e-4, 1e-3, 2e-3, 5e-3; the weighted mean is
  -0.04 +- 0.22, so b2 < 0.33 p^2 at 95 % (one-sided). [VN] unif 2 gives
  0.16 +- 0.19 p^2 (bound 0.46).
- unif 2 is a single exponential at every testable p (0 of 10 cells; smallest
  p-value 0.019). On this all-even depth grid unif 2 always uses round m - 1,
  so it never leaves forward steps unchecked before the inverse half. Its
  per-Clifford error is about 33 p^2, about 1.7 times unif 1's (section 4.6).
  [VN]

### 4.4 const 1, and where its m^2 term comes from

Prediction tested: quadratic-in-m infidelity from pairs of undetected errors in
the unchecked inverse half.

- The decay accelerates at every p. At p = 1e-3 the windowed error per
  Clifford rises monotonically through 1.18, 1.52, 3.04, 4.69, 7.81, 11.6, 15.5
  and 18.7 x 1e-4, a factor 16. [VN]
- The infidelity is quadratic in m at small p. Model E fits at p = 1e-4, 2e-4
  and 5e-4 (goodness-of-fit p = 0.17, 0.93, 0.004) with b2 = 35.1 +- 5.5,
  30.9 +- 1.8 and 37.8 +- 0.7 p^2. Its linear term is within 1.6 standard errors
  of zero. b2 scales as p^(2.17 +- 0.06) over 1e-4 to 5e-4. From p = 1e-3 on no
  3-parameter model fits (E: p = 1e-36), b2/p^2 grows to 52 and 86 at 1e-3 and
  2e-3, and the fitted exponent over 2e-4 to 2e-3 is 2.62: higher orders enter
  once the expected number of detectable faults per shot passes about one.
  [VN]
- The inverse half is not the source. Every policy runs the same circuit, and
  no check is used between the start of the inverse half and the terminal
  round. A pair of faults that both sit in the inverse half therefore produces
  the same accepted logical error in unif 1 and in const 1. In unif 1 it is
  the only order-p^2 source of an m^2 term. A forward fault is caught by the
  next used round unless a second fault in that round masks it, and such local
  pairs grow linearly in m. Faults after the interactions of round m - 1 can
  pair with inverse-half faults, but they sit at a fixed number of locations,
  so these pairs also grow linearly in m. Conditioning on acceptance adds
  U1 R1(m) p^2 to the error
  rate, with U1 independent of m and R1 linear in m, so it adds no m^2 term
  either. [VA, to order p^2] unif 1's m^2 coefficient, below 0.33 p^2, thus
  bounds the inverse-half pairs at 0.33/31 = 1.1 % of const 1's coefficient.
  At equal p the ratio is (-0.2 +- 0.8) % at 1e-3 and (-0.03 +- 0.34) % at
  2e-3. [VN]
- The source is the forward span. const 1 ignores the outcomes of the m
  forward check rounds, so all m forward steps form one unchecked segment
  ending at the terminal round. unif k at depths with no unchecked tail
  consists of m/k closed segments of k steps. It gains (25 +- 3) k p^2 of
  infidelity per unit depth (section 4.6), i.e. about 25 k^2 p^2 per segment,
  the pair count of one segment. One segment of m steps gives about
  25 m^2 p^2: 65 to 80 % of const 1's 31 to 38 m^2 p^2 at p <= 5e-4. The
  cross-policy model of section 4.6 assigns 2.6 +- 1.4 p^2 (p = 2e-4) and
  3.5 +- 0.5 p^2 (5e-4) to pairs made of one forward-tail fault and one
  inverse-half fault (term r m). [VN for the coefficients; CONJ for the pair
  reading]
- So the predicted form is confirmed and the predicted cause is not. About
  90 % of the m^2 term involves two forward-half faults, about 10 % one forward
  and one inverse fault, and at most 1 % two inverse faults. [VN, with the
  split model-dependent] A4's exact attribution on the real circuits, read
  but not re-derived here, gives the same picture: 21.6 of 24.4 from
  forward-forward pairs, about 3 from forward-inverse pairs, about 0.05 from
  inverse-inverse pairs (`A4-nonmarkov-theory.md`, section 5.3).
- Acceptance. The per-step hazard is 70.5 p at p = 1e-4 and constant in m up
  to 5e-4. From 1e-3 on it falls with m (trend z = 4.2 at 1e-3, 19 at 2e-3,
  m = 0 excluded) as the syndrome randomises, and at p >= 2e-2, m >= 14 the
  acceptance is 0.012343 (sd 0.000015 over 16 points), against
  1/81 = 0.012346. The fidelity there is 0.0081, the Rayleigh floor of |.|
  (0.0080). [VN] Each step maps the
  syndrome group Z_3^4 bijectively (the logical gates permute the stabilisers)
  and adds an increment independent of the current syndrome (Pauli faults), so
  the transition matrix is doubly stochastic and the uniform distribution,
  1/81 trivial, is its fixed point. [VA, under those two conditions]

### 4.5 const 0

- Deviates at all 14 p, always accelerating, curvature z = 13 (1e-4) to 80
  (3e-2). At p = 1e-3 the windowed rate rises 26-fold from 2.7e-6 to
  6.9e-5. [VN]
- Pure m^2 law. y(0) = 1 exactly, because the m = 0 const 0 circuit has no
  noisy operation. For p <= 2e-3 model E fits (p = 0.31 to 0.79) with
  b2 = 2.17 +- 0.17, 2.22 +- 0.09, 2.27 +- 0.06, 2.27 +- 0.05 and 2.44 +- 0.05
  p^2; the weighted mean over 1e-4 to 1e-3 is 2.26 +- 0.04 (chi2 0.6 on 3 dof)
  and b2 scales as p^(2.01 +- 0.02). The linear term is slightly negative,
  between -0.8 and -2.5 p^2 (at most 3.6 standard errors from zero). [VN] If
  every pair of distinct Clifford blocks contributes equally, the pair count
  C(2m, 2) = 2m^2 - m predicts b1 = -b2/2 = -1.1 p^2, close to what is seen.
  [CONJ]
- Acceptance: hazard 8.2 p per unit depth for p <= 2e-3 (7.7 p at 5e-3),
  constant in m (trend |z| <= 1.1 for p <= 5e-3). It saturates toward 1/9
  (0.1112 to 0.1126 at p >= 0.04, m >= 18), the trivial fraction of the two
  X-stabiliser outcomes. [VN]

### 4.6 Check spacing: the decay follows the schedule, not m

The schedule variables used below, per (policy, m), are in `points.csv`: the
used intermediate rounds, the closed forward segments between them, and the
unchecked forward tail r (forward Cliffords after the last used intermediate
round, followed by the inverse half).

1. Per-step infidelity grows linearly with the spacing. For unif k, depths
   that are multiples of k have r = 0. On those depths the infidelity grows
   per unit depth by [VN, `check_spacing_unif.csv`]

   | k | 1 | 2 | 3 | 4 | 5 | 6 | 10 |
   |---|---|---|---|---|---|---|---|
   | p = 5e-4, rate/(k p^2) | 24.7 | 23.8 | 26.3 | 25.1 | 22.9 | 26.9 | 24.7 |
   | p = 1e-3, rate/(k p^2) | 28.1 | 24.3 | 24.9 | 25.8 | 24.1 | 25.3 | 28.7 |
   | p = 2e-3, rate/(k p^2) | 27.5 | 26.5 | 26.5 | 28.0 | 29.6 | 30.6 | 36.9 |

   with errors of 0.7 to 3.9 on each entry. A process that forgot everything after
   each step would show a rate independent of k. The observed rate is what
   pairs inside each segment of k unchecked steps give: about 25 k^2 p^2 per
   segment, 25 k p^2 per unit depth. [VN; CONJ for the pair reading]
   `spacing_and_quadratic.png`, left.
2. Unchecked steps before the inverse half cost extra infidelity. At m = 2 the
   policies differ only in whether rounds 0 and 1 are used. Relative to unif 1
   (both used), using only round 0 (const 2, r = 1) adds 135 +- 36 p^2, and
   using neither (const 1, r = 2) adds 215 +- 37 p^2 at p = 1e-3; the values at
   5e-4 and 2e-3 agree within errors, so the cost scales as p^2.
   [VN, `tail_cost_m2.csv`]
3. This makes unif k >= 3 a sawtooth in m mod k. On m >= k, adding a tail term,
   1 - y = alpha + beta m + tau r, lowers chi2 at p = 1e-3 from 59/5 to 1.7/4
   (k = 4), 146/4 to 6.9/3 (k = 5), 99/4 to 8.9/3 (k = 6) and 216/3 to 5.2/2
   (k = 8), with tau = 120 to 215 p^2 per unchecked step. For k = 7, 10 and 11
   the tails reach 6 to 8 steps and a linear term is not enough. For m < k,
   unif k is const 1 and contributes const 1's curvature. [VN]
4. A model in the schedule geometry describes every policy. Over the 192
   distinct (used rounds, m > 0) points of the 44 ordinary policies, the
   descriptive model
   1 - y = a0 + a1 m + a_n n_used + k_seg S2 + tau r + k_t r^2 + k_FI r m + k_II m^2,
   with S2 the sum of squared closed-segment lengths, gives [VN,
   `segment_model.csv`]:

   | p | rms residual (SEM) | chi2 / chi2 without schedule terms | unif-only fit, const c rms (max) | k_seg | k_t | k_FI | k_II (p^2) |
   |---|---|---|---|---|---|---|---|
   | 1e-4 | 1.07 | 219 / 2710 | 1.13 (2.7) | 23.9 +- 0.9 | 22 +- 2 | 4.7 +- 4.0 | 4.2 +- 1.8 |
   | 2e-4 | 0.47 | 42.5 / 13 900 | 0.57 (1.7) | 24.1 +- 0.4 | 25.4 +- 0.9 | 2.6 +- 1.4 | -1.0 +- 0.7 |
   | 5e-4 | 1.56 | 468 / 118 000 | 2.08 (5.1) | 27.7 +- 0.1 | 31.7 +- 0.4 | 3.5 +- 0.5 | -0.7 +- 0.2 |
   | 1e-3 | 3.33 | 2130 / 221 000 | 4.35 (11.4) | 34.1 +- 0.1 | 43.1 +- 0.3 | 2.6 +- 0.3 | 0.1 +- 0.1 |
   | 2e-3 | 8.75 | 14 700 / 432 000 | 9.21 (35) | 50.5 +- 0.1 | 71.9 +- 0.4 | 0.7 +- 0.3 | 1.3 +- 0.1 |

   At p <= 2e-4 the schedule explains the policy and depth dependence to
   within the per-point standard error, and the const c data are predicted
   from unif k data alone. k_seg is about equal to k_t: pairs inside a closed
   segment and inside the open tail have the same density, about 25 p^2. The
   fit degrades from 5e-4 on as higher orders enter. Two warnings apply. The
   chi2 is not calibrated, because points share shots (chi2 = 42.5 on 184 dof
   at 2e-4). And the coefficients of r, r^2 and r m trade off: the m = 2
   comparison in item 2 shows the tail cost is concave at small r, which a
   quadratic in r cannot capture. k_II flips sign across p at the 1 p^2 level,
   which marks the resolution of this model; the unif 1 bound of section 4.3,
   0.33 p^2, is the cleaner number for inverse-half pairs.
5. const c. On m >= c the quadratic coefficient falls with c. For c = 1, 2, 3,
   c x b2/p^2 = 29.2 +- 2.5, 24.6 +- 4.6, 18 +- 9 at p = 2e-4 and
   40.1 +- 1.0, 31.6 +- 1.6, 24.2 +- 3.0 at p = 5e-4. Pair counting with
   25 p^2 per closed-segment pair and 3 p^2 per forward-inverse pair would keep
   c x b2 near 28; the data fall somewhat faster than 1/c. For c >= 4 the
   quadratic model fails (p < 1e-5) because m//c changes irregularly over the
   depth grid. [VN for the values; CONJ for the pair-counting expectation]
6. The fidelity can rise with m, and only when the tail shrinks. 156 windows
   (all p) show a significant rise (z > 3.29). All 156 have a shrinking
   unchecked tail, and 128 of them switch on more used rounds. Example: const 22
   is const 1 for m < 22 and uses 21 intermediate rounds at m = 22 (tail 20 ->
   1), so its last window rises by 109 standard errors at p = 1e-3.
   [VN, `rising_windows.csv`]
   Over the 192 distinct windows per p, the windowed error grows with the tail
   growth per step g = d(tail)/dm, with z = 23 (p = 1e-4) to 271 (2e-3)
   (`tail_growth_regression.csv`).

Taken together [VN for the statements, CONJ for the mechanism]: the logical
error per step is set by how long errors have been able to accumulate since
the last used check. unif 1 and unif 2 renew the memory at a fixed rate on this
grid and are single exponentials. unif k >= 3 and const c >= 2 change their
segment structure with m (sawtooth, switch-on at m = c, irregular m//c), so a
single f fitted across m mixes different protocols. These deviations are
large and real, but they describe the policy definitions, not a change of the
noise with time.

### 4.7 Acceptance: first-order additivity

Over all distinct (used rounds, m) points of the 44 ordinary policies, m = 0
included, a three-parameter law fits the acceptance [VN,
`acceptance_additivity.csv`]:

| p | H0 (p) | H_p per unit depth (p) | H_t per used round (p) | rms residual (SEM) | points |
|---|---|---|---|---|---|
| 1e-4 | 63.6 | 71.0 +- 0.02 | 40.5 +- 0.04 | 1.04 | 193 |
| 2e-4 | 63.3 | 70.9 | 40.5 | 1.12 | 193 |
| 5e-4 | 63.1 | 70.7 | 40.8 | 1.43 | 193 |
| 1e-3 | 63.1 | 70.3 | 41.3 | 2.28 | 193 |
| 2e-3 | 62.9 | 69.1 | 42.9 | 6.0 | 193 |

with -ln(acc) = H0 + H_p m + H_t n_used. It reproduces the per-step hazards of
section 4.3: unif 1, H_p + H_t = 111.5 (measured 111.7); unif 2,
H_p + H_t/2 = 91.2 (measured 91.2); const 1, H_p = 71.0 (measured 70.5).

The reason is first order [VA]. A single fault that leaves an error on the data
leaves a detectable error: every weight-1 error is detectable at distance 2,
and the p^2 scaling of every LRB-D error rate shows that no single fault in the
bulk produces an undetectable logical error [VN]. The error persists and is
caught by the next used round. The terminal round is always used, so each
persistent fault is rejected exactly once, whatever the schedule. A transient
fault (a readout error, or an ancilla error that does not
reach the data) flips one round's record and is rejected only if that round is
used. A rough count of transient flips per round [VA, approximate]: readout
shifts 4 x 5p = 20p; reset shifts before the round 4 x 2p = 8p; depolarising p
on each measured ancilla, X part 6/8, 3p; idle depolarising on ancillas
waiting for their turn (1, 2 and 3 layers of 2p for ancillas 6, 7, 8, X part
6/8; Z parts are harmless, they either propagate to a Z-stabiliser or act on
|0>) 9p. That is 40p, against H_t = 40.5p fitted. Most of H_p comes from the
10p of idle depolarising on each of the five data qutrits per round (50p), the
data parts of the 12 check CNOTs (about 11p), and the two logical Cliffords
per unit depth (about 10p, consistent with const 0's 8.2p, which sees only the
Z parts through its X checks). [CONJ for the split of H_p]

Acceptance therefore carries no memory at first order: it is exponential in m
for any policy whose n_used grows linearly in m. Its non-exponential features
come from the schedule (n_used(m) of const c and unif k) and, from p of about
1e-3, from second-order cancellations that end in saturation at 1/81 (const 1)
and 1/9 (const 0). The fidelity carries the memory instead, at order p^2.
`acceptance_hazard.png` shows the windowed hazards.

### 4.8 Sensitivity to depth correlations

The diagonal SEM ignores the nested-sequence correlations. As a check, the main
log-space fits were repeated with a covariance that keeps the same diagonal but
splits it into multinomial shot noise (independent) and a sequence part
correlated as sqrt(min(m, m')/max(m, m')) across depths (the small-noise limit
for additive per-Clifford contributions; an assumption for LRB) [VN,
`covariance_sensitivity.csv`]:

- curvature z changes by factors 0.56 to 1.57 over the 31 cells with
  |z| > 3, for example const 0 at 1e-3 from 43.5 to 32.2, const 1 at 2e-3
  from 180 to 128, const 1 at 1e-4 from 6.4 to 6.0, const 2 at 1e-4 from 3.4
  to 3.25;
- RB: goodness-of-fit p from 0.025 to 0.98 and |z| <= 2.0; RB_comparison
  0.09 to 0.91, |z| <= 1.2; unif 1 at p <= 2e-3, p >= 0.018 and |z| <= 2.5;
  unif 2, p >= 0.06.

No conclusion changes. Cells near the threshold (|z| of 3 to 4) are
model-dependent and should not be over-read. Sequence variation dominates
const 0 from p = 5e-4 and const 1 from p = 1e-3 (median shot fraction of the
variance at m > 0 below 0.3), while unif 1 and unif 2 stay shot-noise dominated
(0.76 to 1.0) for p <= 2e-3.

## 5. Interpretation

1. Physical noise is Markovian here; the accepted logical process is not.
   The hidden state is the undetected part of the data's Pauli frame. A
   detectable error waits on the data until a used round sees it; a second
   fault can turn it into an undetectable logical error before that. The
   number of such pairs grows with the square of the time since the last used
   check, which produces the m^2 terms of const 1 and const 0, the k-linear
   per-step rate of unif k, and the tail cost. [VN for the scalings, CONJ for
   the mechanism] This is the hidden-memory mechanism of Ceasura et al.
   [LIT 4], driven here by postselection rather than decoding [CONJ].
2. Checking every forward step renews the process. unif 1 is a single
   exponential within errors. Its one shared unchecked interval, the inverse
   half, leaves no measurable m^2 term at this sample size, below 0.33 p^2
   [VN]. A4 reports 0.05 to 0.2 p^2 from an exact evaluation, below this
   sensitivity, so the absence of the term in the data is not evidence that it
   is zero.
3. The LRB-D "decay" of unif k >= 3 and const c >= 2 is not one experiment
   repeated at several m. Its segment structure changes with m, which produces
   sawtooths, switch-on jumps and rising windows. Reporting one F per such
   policy hides this. [VN]
4. The m = 0 point is structurally special in every ordinary policy (no
   forward round, no junction with the inverse half). From p of about 2e-3 it
   biases the unif 1 pipeline rate upward. Fitting b f^m to m >= 2 only
   removes the bias. [VN]
5. Implication for the correction study [CONJ]. A decoder that uses only the
   terminal round (const 1 data usage) cannot remove the forward-span pairs:
   at distance 2 those pairs are undetectable, hence uncorrectable. A decoder
   that acts on every round resets the frame as unif 1 postselection does and
   should give a single exponential, but with an order-p rather than order-p^2
   rate, because at distance 2 correction turns single faults with ambiguous
   syndromes into logical errors. The test is the same as here: m^2 terms,
   windowed trends and the spacing scaling of section 4.6.

## 6. Caveats

- Diagonal-SEM chi2 ignores depth correlations (section 4.8), and the cells
  share shots and sequences. The p-values are per-cell indicators, not a
  calibrated family-wise test.
- For const 1 at p >= 1e-3 and const 0 at p >= 0.01 the three-parameter
  models B, D and E all fail. The curvature z then measures how badly A fails,
  not a physical parameter.
- The segment model of section 4.6 is descriptive. Its fit quality is a
  numerical fact; the reading of its coefficients as fault-pair densities is a
  conjecture, and its r, r^2 and r m coefficients are not cleanly separable.
- The m = 0 dispersion check (section 2.5) is off by about 10 % in opposite
  directions for fidelity and rejection, with no identified cause.
- The run folder also holds per-sequence outcome counts under `progress/`. This
  analysis follows the task and uses only the result CSVs. Per-sequence data
  would allow a full depth covariance or a sequence bootstrap in place of the
  model of section 4.8.
- Only even depths exist. Odd depths would resolve the unif 2 and unif 3
  sawtooth and the const c switch-on.

## 7. Claims

| id | claim | label | evidence |
|---|---|---|---|
| N1-baseline-nonmarkov-C1 | The script reproduces the pipeline LRB fit (a = 0, ceiling 1, 1/std^2 weights with the zero-std floor, 5-pass grid) exactly: 308 unif rows and the RB_comparison columns of the pipeline table, zero difference in f, b, chi2 and point count. F = (1 + 2f)/3 for all 658 cells is in `pipeline_fits.csv`. | verified numerically | `pipeline_table_check.csv`; section 2.2 |
| N1-baseline-nonmarkov-C2 | Weighting by SEM = std/sqrt(30) instead of std leaves f and b unchanged and multiplies chi2 by exactly 30, so only the SEM chi2 is a goodness-of-fit statistic for the 30-sequence mean. | verified analytically | section 2.3; numerically f unchanged and chi2 ratio 30.000 in all 658 fits |
| N1-baseline-nonmarkov-C3 | For a Clifford-twirled (depolarising), time-homogeneous logical channel with state-independent acceptance the metric gives y = b f^m with zero asymptote; mirrored single-qutrit RB with depolarising gate noise and i.i.d. Cliffords gives E[y] = g^m exactly (b = 1, a = 0). | verified analytically | section 3.4 |
| N1-baseline-nonmarkov-C4 | Physical RB (28 cells, p_RB = p and 5p) shows no significant deviation from b f^m: gof p 0.33 to 0.999, curvature z -2.07 to +0.17, flat windowed rates, AICc picks b f^m in 28/28, also under a correlated depth covariance (p >= 0.025). The code-default free-offset RB fit is degenerate (offset up to 0.97, r inflated up to 36x). | verified numerically | section 4.2; `deviation_summary.csv`, `covariance_sensitivity.csv`, `pipeline_fits.csv` |
| N1-baseline-nonmarkov-C5 | 434 of 492 testable LRB cells deviate from b f^m at alpha = 1e-3 per test (428 at Bonferroni 3.4e-5); the non-deviating cells are all of unif 2 (0/10), unif 1 for p <= 2e-3, and otherwise mostly at small p (32 of 45 policies at p = 1e-4, 13 of 45 at 2e-4). Forms by policy group as in section 4.1. | verified numerically | `deviation_by_policy.csv`; section 4.1 |
| N1-baseline-nonmarkov-C6 | unif 1 has near-flat fidelity and fast exponential acceptance: error per Clifford about 19 p^2 (16.5 to 19.8 p^2 for m > 0, 5e-4 to 5e-3), per-step acceptance exp(-111.6 p) with no m trend (|z| <= 0.7) and the same coefficient from 1e-4 to 5e-3; at 1e-3, 10.6 % rejected per step against 2.0e-5 error per Clifford. | verified numerically | section 4.3 table |
| N1-baseline-nonmarkov-C7 | unif 1 is a single exponential for p <= 2e-3 (smallest test p 0.009); at 5e-3 and 7.5e-3 its only deviation is y(0), 6.9 and 5.7 standard errors above the m > 0 exponential, an offset of 60 +- 9 and 64 +- 11 p^2 that inflates the pipeline rate by 50 to 65 %. | verified numerically | section 4.3 (cause, the forward-to-inverse junction missing at m = 0, is a conjecture) |
| N1-baseline-nonmarkov-C8 | unif 1 has no detectable m^2 term: b2 = (-0.04 +- 0.22) p^2 over 5e-4 <= p <= 5e-3, below 0.33 p^2 at 95 %. | verified numerically | section 4.3; `attribution_and_scaling.csv` |
| N1-baseline-nonmarkov-C9 | To order p^2, pairs of faults inside the unchecked inverse half are the only source of an m^2 term in unif 1 and contribute identically to const 1; with C8 they are at most 1.1 % of const 1's m^2 coefficient (about 31 p^2), so at least 99 % of it involves a fault in the forward span. The predicted attribution of const 1's quadratic term to the unchecked inverse half fails. | verified analytically | section 4.4 argument plus C8 and C10 numbers; per-p ratio (-0.2 +- 0.8) % at 1e-3 |
| N1-baseline-nonmarkov-C10 | const 1 shows the predicted quadratic-in-m infidelity at all 11 testable p: curvature z 6.4 (1e-4) to 415 (5e-3); b2 = 35 +- 6, 31 +- 2, 38 +- 1 p^2 at 1e-4, 2e-4, 5e-4, scaling as p^(2.17 +- 0.06); at 1e-3 the single exponential misses y(22) by 1.2e-2 (49 standard errors) and the windowed rate rises 16-fold. | verified numerically | section 4.4; `signatures_unif1_const1_const0.csv` |
| N1-baseline-nonmarkov-C11 | About 90 % of const 1's m^2 term comes from pairs of two forward-span faults (unif k gives about 25 k^2 p^2 per closed segment, i.e. about 25 m^2 p^2 for one m-step span) and about 10 % (3 p^2) from one forward-tail and one inverse-half fault. | conjecture | coefficients from `check_spacing_unif.csv` and `segment_model.csv` (numerical); the split is model-dependent; A4's exact attribution (21.6, about 3, about 0.05 of 24.4) agrees |
| N1-baseline-nonmarkov-C12 | const 0 deviates at all 14 p (curvature z 13 to 80) and follows a pure m^2 law at p <= 2e-3: b2 = (2.26 +- 0.04) p^2 over 1e-4 to 1e-3, scaling as p^(2.01 +- 0.02), linear term -0.8 to -2.5 p^2; its acceptance hazard is 8.2 p per unit depth, exponential for p <= 5e-3. | verified numerically | section 4.5 |
| N1-baseline-nonmarkov-C13 | For unif k, the infidelity added per unit depth at tail-free depths is (25 +- 3) k p^2 for k = 1 to 6 and 10 (5e-4 to 1e-3): the per-step logical error grows linearly with the check spacing. | verified numerically | section 4.6 item 1; `check_spacing_unif.csv` |
| N1-baseline-nonmarkov-C14 | Leaving forward steps unchecked before the inverse half costs infidelity: at m = 2, +135 +- 36 p^2 for one step and +215 +- 37 p^2 for two (p = 1e-3, p^2 scaling from 5e-4 to 2e-3); this produces the sawtooth of unif k >= 3 (chi2 59/5 -> 1.7/4 with a tail term, unif 4 at 1e-3). | verified numerically | section 4.6 items 2 and 3; `tail_cost_m2.csv` |
| N1-baseline-nonmarkov-C15 | An 8-parameter descriptive model in the schedule geometry fits all 192 distinct (policy, m > 0) LRB points at p = 2e-4 to 0.47 SEM rms (chi2 42.5 against 13 900 without schedule terms); fitted on unif k only it predicts all const c points to 0.57 SEM rms (max 1.7); it degrades at higher p (1.6, 3.3, 8.8 SEM rms at 5e-4, 1e-3, 2e-3). | verified numerically | section 4.6 item 4; `segment_model.csv` |
| N1-baseline-nonmarkov-C16 | All 156 windows with significantly rising fidelity (z > 3.29) have a shrinking unchecked forward tail (128 also switch on used rounds): rising fidelity in this data set is a schedule effect. | verified numerically | `rising_windows.csv` |
| N1-baseline-nonmarkov-C17 | Acceptance obeys -ln(acc) = p (63.6 + 71.0 m + 40.5 n_used) over 193 distinct points of all 44 ordinary policies at p = 1e-4 (rms 1.04 SEM), 1.4 SEM at 5e-4, failing from 2e-3; it reproduces the unif 1, unif 2 and const 1 hazards (111.5, 91.2, 71.0 against 111.7, 91.2, 70.5). | verified numerically | section 4.7; `acceptance_additivity.csv`; first-order reason: persistent faults rejected once, transient flips only in used rounds |
| N1-baseline-nonmarkov-C18 | An approximate first-order count of transient record flips per used round (readout 20p, reset 8p, measured-ancilla depolarising 3p, idle on waiting ancillas 9p; CNOT terms neglected) gives about 40p, matching the fitted 40.5p. | verified analytically | section 4.7 |
| N1-baseline-nonmarkov-C19 | Saturation: const 1 acceptance tends to 0.012343 (sd 0.000015 over 16 points; 1/81 = 0.012346) with fidelity at the Rayleigh floor (0.0081 against 0.0080); const 0 acceptance tends to 1/9 (0.1112 to 0.1126); both follow from a uniform syndrome distribution over Z_3^4 and Z_3^2. | verified numerically | sections 4.4 and 4.5; limits derived analytically under bijective syndrome maps and syndrome-independent increments |
| N1-baseline-nonmarkov-C20 | The GLS window-trend slope equals 2 x the log-quadratic coefficient and the constant-rate chi2 equals the log-linear chi2 (REML identity), so the windowed trend test is the curvature test. | verified analytically | section 3.3; numerical check to 1e-11 relative in 466 cells (`identity_checks.csv`) |
| N1-baseline-nonmarkov-C21 | A correlated (Brownian) depth covariance changes curvature z by factors 0.56 to 1.57 and changes no conclusion; borderline cells (|z| of 3 to 4) are model-dependent. | verified numerically | section 4.8; covariance model assumed |
| N1-baseline-nonmarkov-C22 | At m = 0 the observed std is 0.88 +- 0.04 of the multinomial prediction for fidelity and 1.06 +- 0.03 of the binomial one for rejection (0.975 expected), a 10 to 15 % uncertainty on SEM-based z of shot-noise-dominated points; cause unidentified. | verified numerically | section 2.5; `m0_shot_noise_check.csv` |
| N1-baseline-nonmarkov-C23 | Pipeline caveats: const 0 at p = 1e-4 is reported with F = 1 exactly (f pinned at the bound by zero-std points); the pipeline fits depths with as few as about 8 accepted shots per sequence (unif 1, p = 0.01). | verified numerically | section 2.2 |
| N1-baseline-nonmarkov-C24 | A decoder that acts on every round should give a single exponential with an order-p rate at distance 2; terminal-only decoding should keep the forward-span m^2 term. | conjecture | section 5 |

## 8. Literature fetched in this task

1. E. Magesan, J. M. Gambetta, J. Emerson, "Robust randomized benchmarking of
   quantum processes", Phys. Rev. Lett. 106, 180504 (2011),
   doi:10.1103/PhysRevLett.106.180504, arXiv:1009.3639. Fetched
   https://arxiv.org/abs/1009.3639. Used for: RB gives a reliable average
   error-rate estimate under stated conditions, including time- and
   gate-dependent noise.
2. J. J. Wallman, "Randomized benchmarking with gate-dependent noise",
   Quantum 2, 47 (2018), doi:10.22331/q-2018-01-29-47, arXiv:1703.09835.
   Fetched https://arxiv.org/abs/1703.09835. Used for: gate-dependent noise
   adds a perturbation that decays exponentially, so RB converges to an exact
   exponential of the gate-independent form.
3. P. Figueroa-Romero, K. Modi, T. M. Stace, M.-H. Hsieh, "Randomized
   benchmarking for non-Markovian noise", PRX Quantum 2, 040351 (2021),
   doi:10.1103/PRXQuantum.2.040351, arXiv:2107.05403. Fetched
   https://arxiv.org/abs/2107.05403. Used for: non-Markovian noise appears as
   deviations of the average sequence fidelity from the Markovian form.
4. A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan, "Non-Exponential Behaviour
   in Logical Randomized Benchmarking", arXiv:2212.05488 (2022),
   doi:10.48550/arXiv.2212.05488 (no journal reference listed). Fetched
   https://arxiv.org/abs/2212.05488. Used for: a hidden register that records
   gate history makes LRB output non-exponential under gate- and
   time-independent noise; syndrome qubits coupled through a decoder can play
   that role.
5. J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized
   Benchmarking", arXiv:1702.03688 (2017). Fetched
   https://arxiv.org/abs/1702.03688. Used for: definition of LRB.
6. T. Proctor, S. Seritan, K. Rudinger, E. Nielsen, R. Blume-Kohout, K. Young,
   "Scalable randomized benchmarking of quantum computers using mirror
   circuits", Phys. Rev. Lett. 129, 150502 (2022),
   doi:10.1103/PhysRevLett.129.150502, arXiv:2112.09853. Fetched
   https://arxiv.org/abs/2112.09853. Used for: context only; the RB and LRB
   circuits here are mirrored (forward Cliffords followed by their inverses).

Sibling notes cited for comparison only, not re-derived: `A4-nonmarkov-theory.md`
(exact second-order attribution, section 5.3).

## 9. Files

Script: `notes/scripts/N1_baseline_fits.py`
(`module load python/3.11 && python N1_baseline_fits.py [--no-plots]`).
Outputs in `results/baseline_nonmarkov/`:

| file | contents |
|---|---|
| `pipeline_fits.csv` | pipeline f, b, F, r per cell; chi2 with std and with SEM weights; free-offset RB fit |
| `pipeline_table_check.csv` | comparison with the pipeline's own table |
| `points.csv` | every depth: y, std, SEM, sigma used, rejection, N_acc, Rayleigh floor, usable flag, used rounds, segments, tail, residual of A |
| `model_fits.csv` | models A, B, C, D, E, A0, E0: parameters, errors, chi2, dof, p, AIC, AICc, BIC |
| `deviation_summary.csv`, `deviation_by_policy.csv` | tests, significance (nominal and Bonferroni), form and size per cell; per-policy summary |
| `windowed_rates.csv` | f_i, r_i, a_i, hazard per window, with errors and schedule changes |
| `trend_tests.csv` | GLS slope and constant-rate tests for ln f_i and ln a_i, with and without m = 0 |
| `signatures_unif1_const1_const0.csv` | per-p coefficients for unif 1, unif 2, const 1, const 0 (E, E0, D, A0, m = 0 offset, hazard) |
| `attribution_and_scaling.csv` | weighted means of b2/p^2, log-log exponents, unif 1 / const 1 ratio, pipeline-rate exponents |
| `check_spacing_unif.csv`, `check_spacing_const.csv` | unif k rate at tail-free depths and sawtooth fits; const c quadratic coefficients |
| `tail_cost_m2.csv`, `tail_growth_regression.csv`, `rising_windows.csv` | unchecked-tail cost at m = 2; windowed rate against tail growth; rising windows |
| `segment_model.csv`, `acceptance_additivity.csv` | cross-policy schedule model for infidelity; first-order acceptance law |
| `covariance_sensitivity.csv`, `m0_shot_noise_check.csv`, `identity_checks.csv` | checks of the statistics (depth covariance, m = 0 shot noise, REML and policy identities, SEM-vs-std) |

Figures: `windowed_rates.png` (r_i/p^2 for unif 1, const 1, const 0 and r_i/p_RB
for RB at three p), `acceptance_hazard.png` (-ln a_i / p), `residuals_p1e-3.png`
(residuals of b f^m in SEM units at p = 1e-3), `curvature_map.png` (signed
curvature z of every cell), `spacing_and_quadratic.png` (unif k rate against k;
b2/p^2 of const 1, unif 1 and const 0 against p).
