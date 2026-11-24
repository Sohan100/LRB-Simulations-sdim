# L2: exact (shot-noise-free) stationarity tests (lead session)

## Tools (VA + VN)

`src/lrb_qubit/exact.py`:

- `exact_condition(decoder, slots)` is the transfer-matrix sweep of `mld.py`.
  Each time slot is either conditioned on an all-zero syndrome or
  marginalized. Conditioning on nothing gives the exact null-decoder flip
  probability. Conditioning on every slot gives LRB-D unif=1 exactly.
- `RawPolicyEvaluator` extends the sweep by carrying the previous round's raw
  syndrome (4 bits). It gives the exact acceptance and accepted flip
  probability of any raw-syndrome policy, which covers the repo's
  const=k / unif=k, with or without the final data check.

Both are tested against brute-force enumeration over every syndrome pattern at
depths 0–3, for 6 policies (24 tests in `tests/test_lrb_qubit_model.py`, 1e-10
relative). They also agree with the sampled main grid: 378 comparisons,
max |z| = 2.97.

## Null decoder: no correction, no postselection (VN, exact)

Settings: 200 sequences (seed 20261028), every depth 0–40, F = mean over
sequences of |1 − 2 P_flip|. Results: `results/exact_null/`.

The curvature witness c2 is the quadratic coefficient of a fit of log F(m)
over m ∈ [2, 40]. The decay rates come from log-linear fits over the windows
[2, 12] and [20, 40]. Intervals are 95%, from a 300-sample sequence bootstrap.

| p | circuit | f early | f late [95% CI] | c2 [95% CI] |
|---|---|---|---|---|
| 5e-4 | all check rounds | 0.97931 | 0.97934 [0.97922, 0.97947] | +1.1e-6 [−3.8e-6, +5.9e-6] |
| 5e-4 | no checks | 0.99783 | 0.99784 [0.99780, 0.99788] | +1.8e-7 [−1.2e-6, +1.6e-6] |
| 1e-3 | all check rounds | 0.95905 | 0.95910 [0.95887, 0.95935] | +2.0e-6 [−8.6e-6, +1.3e-5] |
| 1e-3 | no checks | 0.99567 | 0.99568 [0.99560, 0.99576] | +3.5e-7 [−2.4e-6, +3.4e-6] |
| 2e-3 | all check rounds | 0.91974 | 0.91984 [0.91925, 0.92030] | +3.2e-6 [−1.8e-5, +2.2e-5] |
| 2e-3 | no checks | 0.99136 | 0.99139 [0.99122, 0.99154] | +6.6e-7 [−5.2e-6, +5.9e-6] |
| 5e-3 | no checks | 0.97854 | 0.97861 [0.97826, 0.97895] | +1.5e-6 [−1.3e-5, +1.6e-5] |

- The null decoder decays as a single exponential to five digits. The early
  and late rates agree, and c2 is consistent with 0 everywhere [VN].
- What theory says [VA, structural]: with no postselection and no decoding,
  every block applies the same Pauli channel to the data frame, conjugated by
  the block's Clifford. The sequence-averaged F(m) is then a fixed linear
  combination of powers of the eigenvalues of one averaged transfer operator.
  That is a finite sum of exponentials with constant coefficients: stationary
  (Markovian in the RB sense), though not necessarily a single exponential.
  The data show that a single exponential dominates.
- Error per depth step, (1 − f)/2, at p = 1e-3 is 20.5p with checks. Work
  package B finds 21.1p for no decoding at first order, and the lead finds
  12.5p for exact MLD. Correction therefore lowers the error by a factor of
  about 1.6, but it does not change the shape of the decay.

**Consequence for Q2.** The decoded curves are stationary, but so are the
undecoded, unpostselected ones. The non-Markovian features of LRB-D (L0 §6)
therefore come from postselection, not from the absence of correction.
Correction removes them only in the sense that it removes postselection.

## Exact LRB-D policies (VN, exact; `results/exact_lrbd/stationarity.csv`)

Settings: 200 sequences, every depth 0–40, the repo-SI1000 model, and the
data5 circuit. The `_final` variants also require the two final X-stabilizer
detectors. F is the repo estimator, the mean over sequences of
|1 − 2 P(flip | accept)|.

- c2 is the quadratic coefficient of log F over m ∈ [2, 40].
- f early and f late are log-linear fits over [2, 12] and [20, 40].
- a early and a late are the same fits applied to the acceptance.
- Intervals are 95%, from a 1000-sample sequence bootstrap.

| p | policy | c2 [95% CI] | f early → late (late − early [95% CI]) | a early → late |
|---|---|---|---|---|
| 1e-3 | unif1 | −1.1e-7 [−1.5e-7, −5.8e-8] | 0.99990 → 0.99989 (−4.8e-6 [−7.2e-6, −2.4e-6]) | 0.9005 → 0.9005 |
| 1e-3 | unif2 | +4.0e-8 [−4.6e-8, +1.2e-7] | 0.99980 → 0.99980 (−5.1e-6 [−9.5e-6, −1.0e-6]) | 0.9189 → 0.9189 |
| 1e-3 | unif4 | −7.9e-8 [−2.6e-7, +1.1e-7] | 0.99959 → 0.99957 | 0.9280 → 0.9287 |
| 1e-3 | unif8 | −1.2e-6 [−1.6e-6, −8.9e-7] | 0.99916 → 0.99905 | 0.9334 → 0.9340 |
| 1e-3 | const8 | −1.6e-5 [−1.6e-5, −1.5e-5] | 0.99973 → 0.99900 | 0.9033 → 0.9388 |
| 1e-3 | const4 | −3.8e-5 [−3.8e-5, −3.7e-5] | 0.99949 → 0.99781 | 0.9284 → 0.9405 |
| 1e-3 | const1 | −3.4e-4 [−3.4e-4, −3.4e-4] | 0.99770 → 0.98343 (−1.4e-2) | 0.9407 → 0.9581 |
| 5e-3 | unif1 | −5.6e-6 [−7.0e-6, −4.2e-6] | 0.99690 → 0.99665 (−2.5e-4 [−3.2e-4, −1.9e-4]) | 0.5947 → 0.5949 |
| 5e-3 | unif4 | −9.2e-6 [−1.5e-5, −3.6e-6] | 0.98345 → 0.98234 | 0.7025 → 0.7061 |
| 5e-3 | const1 | −1.1e-3 | 0.88276 → 0.79868 (−8.4e-2) | 0.8471 → 0.9994 |

The full table covers every policy at p = 5e-4, 1e-3, 2e-3 and 5e-3.

- **const=k policies are strongly non-stationary at every p** [VN]. The
  accepted decay accelerates (c2 < 0, f late < f early). Because the check
  positions scale with depth, so do the effective checks per Clifford, and the
  acceptance per step drifts upward towards the random-syndrome floor
  (1/16 for const=1). This is the qutrit production run's const=1 shape.
- **unif=1 (every check) is nearly stationary** [VN]. Its acceptance per step
  is exactly constant. Its accepted per-step error grows by 5–8% from early to
  late depths, and c2 ≈ −0.11p² at 1e-3 and −0.22p² at 5e-3. The rb-theorist
  brainstorm predicted about −0.2p² from error pairs in the unchecked inverse
  half; that mechanism is tested directly below.
- **Stationarity degrades as checks get sparser.** unif2–4 have c2
  consistent with 0, unif8 is clearly curved, and the const policies are
  strongly curved. Sparse checks let undetected weight-2 logical pairs
  accumulate between used checks, with probability ∝ (interval)².

## Inverse-half checks: the cause of unif=1's residual curvature (VN, exact)

`studies/correction-decoders/qubit/scripts/exact_inverse_checks.py`; `results/exact_inverse_checks/`.

Settings: 100 sequences, depths 0–40. The policy is LRB-D with every check
round plus the final data check, on the repo's circuit (inverse half
unchecked) and on a variant with a check round after every inverse Clifford
(`inverse_check_every=1`). Intervals are 95%, from a 500-sample bootstrap.

| p | inverse half | c2 [95% CI] | f early → late | acceptance per step |
|---|---|---|---|---|
| 1e-3 | unchecked (repo) | −9.5e-8 [−1.6e-7, −2.8e-8] | 0.999902 → 0.999898 | 0.9005 |
| 1e-3 | checked | +2.2e-8 [−8.8e-8, +1.4e-7] | 0.999825 → 0.999825 | 0.8146 |
| 5e-3 | unchecked (repo) | −4.6e-6 [−6.6e-6, −2.9e-6] | 0.997063 → 0.996854 | 0.5946 |
| 5e-3 | checked | +5.8e-7 [−2.7e-6, +3.8e-6] | 0.994806 → 0.994807 | 0.3614 |

- **Checking the inverse half removes the residual curvature.** c2 becomes
  consistent with 0, and the early and late rates agree to 1e-6. Every-round
  LRB-D is then stationary to the precision of 100 sequences.
- **The mechanism.** The residual non-Markovianity of unif=1 is caused by
  the repo's unchecked mirror-inverse half [VN]. In that half, pairs of faults
  forming an undetectable logical accumulate with probability ∝ m². Checking
  every inverse Clifford makes every segment between checks the same length.
- **The price.** The checked variant doubles the check rounds per depth step,
  so its acceptance per step and its per-step accepted fidelity are both
  lower. Stationarity is not free.
