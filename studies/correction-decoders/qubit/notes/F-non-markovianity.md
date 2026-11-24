# WP-F: non-Markovianity of LRB-D versus LRB with correction (qubit [[5,1,2]])

Work package F. Author: WP-F agent, 2026-10-28. Code: `studies/correction-decoders/qubit/scripts/nonmarkov_runs.py`
(simulations), `studies/correction-decoders/qubit/scripts/nonmarkov_analysis.py` (tests, tables, figures).
Results: `studies/correction-decoders/qubit/results/F/`. Figures:
`studies/correction-decoders/qubit/figures/F/`.

Labels: **[A]** verified analytically, **[N]** verified numerically, **[L]** from
literature (fetched this session), **[C]** conjecture. Every number carries a 95%
sequence-bootstrap interval unless it is exact.

## Summary and verdict

**Verdict: yes, correction removes the non-Markovianity that matters for LRB-D, because
the decoded ensemble is not conditioned on anything. The price is a decay rate of first
order in p instead of second order, and the same Markovian shape is already present with
no decoding at all.**

1. **Correction (exact MLD, every round, no rejection) gives a single exponential at
   all three p** **[N]**. The relative rate drift `rho = (r_late - r_early)/r` compares
   the last and first thirds of m = 2..30; p_GOF is from the null-bootstrap
   goodness-of-fit test of `A f^m`.

   | p | rho | p_GOF |
   |---|---|---|
   | 5e-4 | 0.021 [-0.029, 0.074] | 0.55 |
   | 2e-3 | 0.014 [-0.032, 0.061] | 0.79 |
   | 5e-3 | 0.050 [0.004, 0.096] (marginal) | 0.51 |

   - No two-exponential, stretched or quadratic-log model fits better (LR p >= 0.3).
   - **Why** **[N]**: in the exact first-order expansion over 150 sequences, the decoded
     failure weight is affine in m: slope 11.6 p per block, quadratic coefficient
     -0.0035 [-0.011, 0.004] p per block^2.
   - The only depth-dependent first-order piece is global decoding of faults that only
     the terminal check sees (the unchecked inverse half). It saturates instead of
     growing, and it is 5% of the slope.
2. **LRB-D is non-Markovian.**
   - *Strongly*, for every policy that leaves long unchecked segments or moves its checks
     with m **[N]**:
     - `const=1`: rho = 2.9 to 4.0;
     - `const=4`: rho = 1.8 to 2.7;
     - `const=0`: rho = 1.1 to 1.4.

     These are exact, shot-noise-free curves. The undetected-logical weight grows like
     m^2.
   - *Weakly but resolved, even for the densest policy* `unif=1` **[N]**: exact
     rho = 0.073 [0.005, 0.14] at p = 2e-3 and 0.085 [0.021, 0.15] at 5e-3. The single
     exponential is rejected (p_GOF <= 0.007) and the quadratic-log form is preferred.
   - The mechanism is pairs of faults inside the unchecked inverse half, which only the
     terminal check sees **[N]**. In the exact second-order expansion (150 sequences):
     - pairs whose shared syndrome lives only in the terminal round carry a quadratic
       term of 0.040 [0.021, 0.058] p^2 per block^2;
     - forward-round pairs are affine: 0.012 [-0.023, 0.047];
     - checking the inverse half too removes the term (0.016 [-0.051, 0.083]), and the
       exact curves then pass the single-exponential test (p_GOF = 0.13 to 0.32).
3. **The LRB-D per-block rate belongs to the checked segment, not to the block**
   **[N]**. On exact curves, `r(unif=k)/r(unif=1)` for k = 2 / 4 is:

   | p | LRB-D | MLD on the same rounds |
   |---|---|---|
   | 5e-4 | 1.98 / 4.04 | 1.06 / 1.11 |
   | 2e-3 | 2.04 / 4.47 | 1.09 / 1.20 |
   | 5e-3 | 2.19 / 5.52 | 1.11 / 1.21 |

   Faults in different blocks combine through the syndrome register into undetected
   logicals. That is memory. Under correction each fault is close to a block-local
   event.
4. **Caveats that bound the verdict.**
   - The null decoder (no decoding, no rejection) is just as Markovian: rho =
     0.021 [-0.013, 0.056] (exact) and p_GOF = 0.47 to 0.54. The Markovian shape comes
     from dropping postselection; correction keeps it and lowers the rate by 28 to 43%.
     Rates per block at p = 5e-4 / 2e-3 / 5e-3:
     - MLD: 0.0120 / 0.0528 / 0.152;
     - null decoder: 0.0210 / 0.084 / 0.210;
     - LRB-D `unif=1`: 2.42e-5 / 4.15e-4 / 3.00e-3.

     So the decoded rates are 50 to 500 times the LRB-D `unif=1` rates.
   - Markovianity is lost again when the decoder postselects on soft output
     (`|LLR| >= t`: `F >= tanh(t/2)`, rho = -0.13 to -0.91), when its information depends
     on m (`const=k` rounds: rho = 0.13 to 0.25), or when extra check noise sits in an
     unchecked segment.
   - Against dense `unif=1` the gap in drift is small and not significant at any single
     p: |rho_MLD| - |rho_LRBD| = -0.045 / -0.059 / -0.036, each with a 95% CI that
     includes 0. What clearly separates the two is the resolved m^2 mechanism in LRB-D
     (exact curves and exact expansion).
   - Correction removes non-Markovianity in shape, not in absolute size: the MLD data
     allow absolute log-F deviations of up to about 0.02 to 0.2 at m = 30 (CI bounds of
     Delta(30)). LRB-D `unif=1` deviates by less than 0.01, because its whole decay is
     small.

---

## 1. Definition

### 1.1 Setting and estimator

A depth-m LRB sequence applies logical Cliffords C_1 ... C_m, drawn i.i.d. uniformly from
the 24-element single-qubit Clifford group, each followed by one noisy check round, then
the inverses C_m^-1 ... C_1^-1 with no checks, a final check round, and the ideal X_L
readout (CONTEXT.md). The figure of merit used here is the signed polarization

    F(m) = (1/30) sum_s F_s(m),   F_s(m) = 1 - 2 e_s(m) / a_s(m),

on the reported ensemble: accepted shots for LRB-D (`a_s` accepted shots, `e_s` logical
errors among them), all shots for correction. The repo averages `|P0 - P1|`; the sign is
kept here because the ideal outcome is known and the absolute value inflates small
signals. Both versions are in `analysis.json` (`F` and `F_abs`).

### 1.2 Definition (logical Markovianity)

A protocol is **Markovian at the logical level** if there are fixed, trace-preserving
logical channels `Lambda_C` (forward block: logical C, its check round, and what the
protocol does with that round's outcomes) and `Lambda'_C` (inverse block), and fixed SPAM
(rho, E), all independent of the depth m, of the position j and of the outcome history,
such that for every m and every sequence the reported outcome distribution is

    p(m, C) = Tr[ E Lambda'_{C_1^-1} ... Lambda'_{C_m^-1} Lambda_{C_m} ... Lambda_{C_1}(rho) ].

"Trace preserving" is the normalization requirement: for postselection, `Lambda_C` must be
the block's conditional channel divided by its own acceptance, and the acceptance of the
whole sequence must factorize into per-block acceptances. Otherwise the conditional
ensemble at depth m is not a composition of fixed block maps.

Consequence **[A]**: the uniform Clifford twirl of any single-qubit channel is
depolarizing. The 24 Cliffords act on the traceless Pauli space span{X, Y, Z} as the
rotation group of the octahedron, which is irreducible. By Schur's lemma the twirled
Pauli-transfer matrix is f on that space, and the only invariant vector is 0, so the
twirled channel is unital. For gate-independent blocks the sequence average is therefore
`F(m) = A f^m` with **B = 0** for the signed polarization: log F is affine in m. For
gate-dependent but fixed blocks, Wallman [L1] shows that the decay is the same
exponential plus one extra term that decays exponentially fast in m, i.e. a short
transient at small m.

### 1.3 Tests derived from the definition

If the protocol is Markovian, then:

- **T1** the local rates `r(m) = -d log F / dm` are constant (after a short transient);
- **T2** log F has no curvature; `A f^m` fits, and two-exponential, stretched-exponential
  (`A exp(-l m^b)`) and quadratic-log (`A exp(-a m - g m^2)`) forms do not fit better;
- **T3** f does not depend on the fit window;
- **T4** a Markov model calibrated at short depth (`A f^m` fitted on m = 1..6) predicts
  the long-depth data;
- **T5** the per-block rate does not depend on how blocks are grouped into checked
  segments, beyond the information effect that exists for every decoder. Under a
  memoryless block model a k-block segment has k times the 1-block logical error.
- **T6** a leading-order failure weight that is affine in m (exact expansion, Section 3.7).

### 1.4 What breaks it for LRB-D

1. **Postselection conditioning.** `F = N(m)/A(m)` with `N = P(acc, +) - P(acc, -)` and
   `A = P(acc)`. Even when the unconditioned process is a Markov chain on a hidden register
   (logical class, 4-bit syndrome, ancilla flags), the accepted process is a Doob
   h-transform of a killed chain. The conditional channel of block j depends on which
   checks come later, i.e. on m and j. Fixed normalized block channels exist only if
   acceptance factorizes per block, which fails as soon as syndrome information crosses
   a block boundary **[A]**.
2. **Policy-dependent check positions.** `const=k` uses forward checks at
   `j*floor(m/k) - 1`, so the check pattern, and with it every block's conditional
   channel, changes with m. The circuit family is not nested and no fixed block channel
   exists even in principle. `unif=k` is periodic: it can be Markovian only on the
   lattice m = 0 mod k, with a period-k sawtooth in between **[A]**.
3. **The inverse half has no checks.** The m inverse blocks form one unchecked segment
   that only the terminal check sees. Two faults inside it with equal syndromes and
   different logical classes pass. Their number grows like m^2, so log F gets curvature
   **[A]**, measured in Section 3.7.
4. **Persistent residual errors.** With sequential single-ancilla rounds, a data fault
   after its stabilizer's readout (or one masked by a readout flip) enters the next block
   as a non-zero syndrome. The next block's logical gates act on the joint (syndrome,
   logical) frame, and the next check rejects it or, if masked again, passes it. This
   couples consecutive blocks **[A]**.
5. **Ancilla faults.** Readout flips (DEPOLARIZE1(p) + X_ERROR(5p)) and reset flips
   (X_ERROR(2p)) reject shots with intact data, mask real errors (feeding 4), and a reset
   flip correlates rejections in adjacent rounds **[A]**.

### 1.5 What breaks it for correction

1. **Global decoders use future syndromes.** Exact MLD decides about a fault in block j
   using every later syndrome. A fault seen only by the terminal check (any inverse-half
   fault) is decoded with a posterior spread over all m inverse blocks, so the decoded
   failure probability of a given fault depends on m and j **[A]**. This is a
   smoothed per-block rate, legitimate for a Clifford-only circuit with Pauli-frame
   tracking, not a time-local logical channel.
2. **Correlated logical errors.** One fault can produce logical errors attributed to two
   blocks (a misdecoded hook or a measurement error), pairs of faults combine, and the
   decoder posterior is itself a hidden state carried across blocks **[A]**. Section 1.6
   shows when this still gives a single exponential.
3. **m-dependent information.** A decoder that sees only the `const=k` rounds inherits
   the policy's m-dependence (item 2 of 1.4) **[A]**.
4. **Soft-output postselection.** Accepting only `|LLR| >= t` conditions again. For the
   exact posterior, `P(wrong | |LLR| = g) = 1/(1 + e^g)`, so every accepted ensemble has
   `F >= tanh(t/2)` at every m **[A]**. The decay must flatten into a floor while the
   acceptance keeps falling, which is non-exponential by construction.

### 1.6 Clifford twirl with correlated Pauli logical errors

Let `U_j = C_j ... C_1` (`U_0 = I`).

- **Lemma 1 [A].** If the C_j are i.i.d. uniform on the Clifford group, so are
  U_1 ... U_m: `(C_1..C_m) -> (U_1..U_m)` is a bijection of the group^m that preserves
  the uniform measure.
- **Lemma 2 (mirror pairing) [A].** A logical Pauli E occurring right after forward
  C_j, or right after inverse C_{j+1}^-1, reaches the readout as `U_j^dag E U_j`. The
  rest of the circuit is U_j^-1 in both cases. So twirl block j = (forward block j,
  inverse block j+1). Forward block m has no inverse partner, and faults after C_1^-1
  (including the terminal round) are untwirled SPAM.
- **Theorem [A].** Suppose each shot ends with the logical frame
  `R = prod_{j=0..m} U_j^dag E_j U_j` (after decoding, if any), with `E_j` in {I,X,Y,Z}
  the net logical fault of twirl block j. Suppose also that the joint law of
  (E_0, ..., E_m) does not depend on (U_1, ..., U_m) (assumption T: gate-independent noise
  and a twirl-covariant decoder). Then

      F(m) = E[ c_0 (-1/3)^K ],   K = #{ j >= 1 : E_j != I },

  where c_0 = +-1 is the commutation sign of the untwirled E_0 with X.
  *Proof.* `<X_L> = prod_j s_j`, with `s_j = +1` if `U_j^dag E_j U_j` commutes with X
  and -1 otherwise. Conditioned on the E's, the s_j (j >= 1) are independent by Lemma 1.
  For `E_j != I`, `U_j^dag E_j U_j` is uniform over the six signed non-identity Paulis
  (the group acts transitively on them, with stabilizers of size 4). So s_j = +1 with
  probability 1/3 and -1 with probability 2/3, and `E[s_j | E] = -1/3`. QED.
- **Independent blocks [A].** If the E_j are i.i.d. with `P(E_j != I) = eps`, then
  `F = E[c_0] (1 - 4 eps/3)^m`: Markovian, with f = 1 - 4 eps/3.
- **Generating function [A].** Let `a_j = 1[E_j != I]` be emitted by a stationary
  finite hidden Markov chain: hidden state h = syndrome frame, ancilla flags and decoder
  memory; kernel `T = T_0 + T_1`, where T_a are the transitions that emit a. Then
  `G_m(z) = E[z^K] = u^T (T_0 + z T_1)^m v` and

      F(m) = sum_k alpha_k lambda_k^m,   lambda_k = eigenvalues of M = T_0 - T_1/3.

  **When is it asymptotically single-exponential?** When M has a simple eigenvalue
  lambda_1 of strictly largest modulus. This holds if T is irreducible and aperiodic with
  spectral gap 1 - rho and the error weight is small next to the gap: by perturbation
  theory `lambda_1 = 1 - 4 eps_bar/3 + O(eps^2/(1 - rho))`, while the other eigenvalues
  stay within `rho + O(eps)`. Then
  `F(m) = alpha_1 lambda_1^m [1 + O((|lambda_2|/lambda_1)^m)]`, a single exponential after
  a transient of about `1/ln(lambda_1/|lambda_2|)` blocks. Correlations do not bend the
  curve asymptotically; they renormalize the rate. A cumulant expansion of
  `log E[prod_j (1 - 4 a_j/3)]` gives

      ln lambda_1 = ln(1 - 4 eps_bar/3) + (16/9) sum_{d>=1} Cov(a_0, a_d) / (1 - 4 eps_bar/3)^2 + O(kappa_3),

  so bursts (positive correlations) slow the decay and anti-bunching speeds it up.
  **When it fails:**
  1. non-mixing hidden variables: a frozen per-shot or per-sequence rate, or an absorbing
     sector. This gives a mixture `sum_theta w_theta f_theta^m`, which is log-convex
     and reaches its asymptote only for `m >> 1/ln(f_1/f_2)`;
  2. non-stationary kernels: `const=k`, the unchecked inverse half whose memory spans all
     m inverse blocks, or decoders whose memory grows with m;
  3. postselection: `F = G^acc_m(-1/3) / G^acc_m(1)` is a ratio of two sums of
     exponentials. It becomes single-exponential only when both leading eigenvalues are
     isolated; the clean and randomized sectors of `const=1` are a near-degeneracy and
     give a crossover instead;
  4. acceptance or decoder decisions that depend on the twirl (gate-dependent syndrome
     propagation), which breaks assumption T.
- **Numerical check [N].** `twirl_toy` in the analysis script draws 4e5 trajectories
  with uniformly random Clifford sequences and maximally biased (Z-only) logical errors
  from a two-state burst HMM (eps = 0.01 / 0.15, stationary eps_bar = 0.03). The Monte
  Carlo `<X_L>` matches `pi T(-1/3)^m 1` at all m = 1..40 (max |z| = 0.86). The decay is
  single-exponential with `lambda_1 = 0.96646`, slower than the independent-block
  `0.96000` because the bursts are positively correlated. The second-order cumulant
  estimate gives `ln lambda_1 = -0.0322` (exact -0.0341; higher cumulants matter at
  eps = 0.15), and `|lambda_2/lambda_1| = 0.549`. A frozen per-trajectory rate (0.01 or
  0.12) gives the predicted curved mixture (max |z| = 1.65 over 40 depths).
  Figure: `figures/F/twirl_toy.png`.

---

## 2. Methods

- **Model.** The lead's qubit model (`src/lrb_qubit/`, not modified): qubit [[5,1,2]],
  SI1000 noise, `data5` circuits. LRB-D policies use the repo's data-2 semantics (raw
  check-round syndromes only), and `_final` adds the two final data X-stabilizer
  detectors.
- **Sequences.** 30 sequences from `experiment.clifford_sequences(30, 30, 20261028)`;
  depths 0..30 in steps of 1 (the inverse-check variant uses steps of 2).
- **Sampled runs** (`nonmarkov_runs.py`, the same bookkeeping as `run_point`).
  - One Stim sample per (sequence, depth) cell feeds every method, so all methods are
    compared on identical shots:
    - LRB-D `unif=1,2,4` and `const=1,4`, with and without `_final`;
    - `mld5`, `mld2`;
    - information-restricted MLD (`mld5_<policy>`: the decoder sees only the raw
      syndromes of the rounds that policy uses, plus the final data detectors);
    - `|LLR|` histograms for the thresholds;
    - the null decoder (`raw`).
  - `const0` and `const0_mld` use their own circuit, as in `run_point`.
  - Physical RB at p_RB = 5p uses `experiment.run_physical_rb` (2e5 shots per cell).
  - Shot budgets: 3e5 shots per cell for every standard cell (all p, all 30 sequences);
    2e5 for the inverse-check variant. About 1.0e5 core-seconds in total, inflated by
    the chunks that ran on pinned CPUs (Section 6).
  - Runs were split into chunks of 10 or 15 sequences on the shared allocation, with at
    most 24 workers and each chunk under 20 minutes. Seeds depend only on (p, sequence,
    depth).
- **Exact curves.** The lead's `lrb_qubit.exact` was used for the same 30 sequences:
  `RawPolicyEvaluator` for every LRB-D policy, and `exact_condition(dec, [])` for the
  null decoder (`--exact`). They carry no shot noise, so they cover all depths even at
  acceptances of 1e-7. The analysis also includes the lead's 200-sequence exact null
  curves (`results/exact_null/`).
- **Exact low-order expansion** (`--low-order`). From the DEM of every circuit:
  - the first-order failure weight of each decoder, `S1 = sum_e p_e [MLD(D_e) != L_e]`;
  - for LRB-D, the second-order undetected-logical weight
    `S2u = sum_sig q0(sig) q1(sig)` over raw-syndrome signatures on the used rounds;
  - a split of each into pairs or faults seen only by the terminal round (the inverse
    half) and the rest.

  This was computed for 30 sequences at every p, and for 150 sequences at p = 5e-4.
- **Statistics** (`nonmarkov_analysis.py`).
  - Sequence-level bootstrap with B = 2000, sharing the resampling indices across
    methods so comparisons are paired. The sampled F_s carry their own shot noise.
  - Fits are weighted (sigma from the bootstrap).
  - Models are compared with profiled grid fits. The chi^2 and likelihood-ratio
    statistics are calibrated by a null residual bootstrap (300 replicates of
    `A f^m`-hat plus resampled sequence fluctuations), which respects the strong
    correlation between nested depths. p = 0.0033 is the smallest attainable value.
  - "Usable" depths: every sequence has at least 100 accepted shots, and the mean
    exceeds 4 bootstrap SE.

## 3. Results

The complete tables are auto-generated in `results/F/note_tables.md` (shape tests for
every method at every p, paired comparisons, spacing, low order, inverse variant) and
`results/F/summary_<variant>_<ptag>.md` (every sampled method, with chi^2, f and
acceptance). The numbers below are copied from them.

### 3.1 Controls

| p | method | r = -ln f (per block) | rho | Delta(30) vs short-depth Markov | p_GOF |
|---|---|---|---|---|---|
| 5e-4 | physical RB at 5p | 0.00167 [0.00163, 0.00172] | -0.019 [-0.10, 0.063] | 6e-4 [-0.003, 0.004] | 0.91 |
| 2e-3 | physical RB at 5p | 0.0067 [0.0065, 0.0069] | -0.006 [-0.087, 0.072] | 0.003 [-0.012, 0.017] | 0.99 |
| 5e-3 | physical RB at 5p | 0.0167 [0.0162, 0.0172] | -0.007 [-0.083, 0.067] | 0.009 [-0.028, 0.044] | 0.99 |
| 5e-4 | null decoder (exact) | 0.0210 [0.0207, 0.0213] | 0.021 [-0.012, 0.055] | 0.001 [-0.016, 0.020] | 0.49 |
| 2e-3 | null decoder (exact) | 0.084 [0.083, 0.085] | 0.021 [-0.012, 0.055] | 0.006 [-0.066, 0.079] | 0.47 |
| 5e-3 | null decoder (exact) | 0.210 [0.208, 0.213] | 0.021 [-0.013, 0.056] | 0.016 [-0.16, 0.20] | 0.54 |
| 5e-3 | null decoder (exact, 200 seq, m = 0..40) | 0.209 [0.208, 0.210] | 0.000 [-0.011, 0.013] | 0.009 [-0.087, 0.10] (at m = 40) | 1.0 |

- Physical RB at 5p is exactly Markovian by construction: depolarizing noise after each
  gate, and the average of i.i.d. products is a product **[A]**. The tests do not flag
  it **[N]**, so they have no false-positive problem at this precision.
- The null decoder is Markovian too **[N]**. Over 150 sequences its exact first-order
  weight has quadratic coefficient 0.0012 [-0.0068, 0.0097] p per block^2.

### 3.2 Correction (exact MLD), shape tests

| p | method | r (per block) | c2 (per block^2) | rho | Delta(30) | p_GOF | LR p (2exp / stretched / quadlog) |
|---|---|---|---|---|---|---|---|
| 5e-4 | MLD, all rounds + final data | 0.0120 [0.0118, 0.0123] | -9.2e-6 [-2.6e-5, 6.9e-6] | 0.021 [-0.029, 0.074] | 0.003 [-0.014, 0.020] | 0.55 | 0.58 / 0.59 / 0.57 |
| 2e-3 | MLD, all rounds + final data | 0.0528 [0.0518, 0.0537] | -3.1e-5 [-9.8e-5, 3.4e-5] | 0.014 [-0.032, 0.061] | 0.009 [-0.054, 0.074] | 0.79 | 0.72 / 0.61 / 0.59 |
| 5e-3 | MLD, all rounds + final data | 0.152 [0.150, 0.154] | -1.3e-4 [-2.9e-4, 4.0e-5] | 0.050 [0.004, 0.096] | -0.05 [-0.21, 0.11] | 0.51 | 0.32 / 0.71 / 0.70 |
| 5e-4 / 2e-3 / 5e-3 | MLD, data-2 readout | 0.0121 / 0.053 / 0.153 | consistent with 0 | 0.027 / 0.019 / 0.042 (all CIs include 0) | consistent with 0 | 0.50 / 0.75 / 0.42 | >= 0.43 |
| 5e-4 / 2e-3 / 5e-3 | MLD on `unif=2` rounds, m = 0 mod 2 | 0.0127 / 0.0575 / 0.168 | consistent with 0 | 0.014 / 0.010 / -0.024 (CIs include 0) | consistent with 0 | 0.67 / 0.91 / 0.96 | >= 0.61 |
| 5e-4 / 2e-3 / 5e-3 | `const0` decoded (no checks) | 7.3e-4 / 2.9e-3 / 7.3e-3 | consistent with 0 | -0.026 / -0.017 / -0.019 (CIs include 0) | consistent with 0 | 0.67 / 0.96 / 0.98 | >= 0.23 |

- **f does not depend on the fit window** **[N]**. For full MLD, fits on [m0, 30] with
  m0 = 0, 2, 10, 20 give:
  - p = 5e-4: 0.0122, 0.0121, 0.0122, 0.0123 (CI half-widths about 0.0003);
  - p = 2e-3: 0.0535, 0.0528, 0.0532, 0.0537;
  - p = 5e-3: 0.154, 0.152, 0.154, 0.158 [0.154, 0.163].

  The +4% at the largest m0 and p = 5e-3 is the marginal drift seen in rho. Figures:
  `figures/F/rate_vs_window.png`, `local_rates_main.png`, `local_rates_exact.png`.
- **The Markov surrogate predicts the long depths** **[N]**. `A f^m` fitted on m = 1..6
  predicts log F(30) within 0.003 [-0.014, 0.020], 0.009 [-0.054, 0.074] and
  -0.05 [-0.21, 0.11] at the three p (`surrogate_exact.png`).

### 3.3 LRB-D, shape tests (exact curves, 30 sequences, no shot noise)

| p | policy | r (per block) | c2 (per block^2) | rho | p_GOF | LR p (2exp / stretched / quadlog) | acceptance at m = 30 |
|---|---|---|---|---|---|---|---|
| 5e-4 | `unif=1` + final | 2.42e-5 [2.36e-5, 2.48e-5] | -5.8e-8 [-9.7e-8, -2.2e-8] | 0.066 [-0.003, 0.14] | 0.04 | 0.37 / 0.017 / 0.013 | 0.20 |
| 2e-3 | `unif=1` + final | 4.15e-4 [4.05e-4, 4.25e-4] | -1.1e-6 [-1.7e-6, -4.5e-7] | 0.073 [0.005, 0.14] | 0.007 | 0.35 / 0.007 / 0.003 | 1.6e-3 |
| 5e-3 | `unif=1` + final | 3.00e-3 [2.93e-3, 3.07e-3] | -8.4e-6 [-1.3e-5, -4.4e-6] | 0.085 [0.021, 0.15] | 0.003 | 0.003 / 0.003 / 0.003 | 1.2e-7 |
| 2e-3 | `unif=1` (data-2) | 4.43e-4 [4.32e-4, 4.54e-4] | -1.1e-6 [-1.8e-6, -5.3e-7] | 0.055 [-0.019, 0.13] | 0.047 | 0.26 / 0.007 / 0.003 | 1.7e-3 |
| 2e-3 | `unif=2` + final, m = 0 mod 2 | 8.19e-4 [8.01e-4, 8.36e-4] | -1.3e-6 [-2.6e-6, -1.7e-8] | 0.042 [-0.026, 0.11] | 0.11 | 0.62 / 0.10 / 0.55 | |
| 2e-3 | `unif=2` + final, all m | 8.47e-4 | -8.5e-8 | -0.071 [-0.13, -0.010] | 0.003 (sawtooth of period 2) | | |
| 2e-3 | `unif=4` + final, m = 0 mod 4 | 1.81e-3 [1.77e-3, 1.84e-3] | -3.8e-6 [-6.7e-6, -1.0e-6] | n/a | 0.047 | 0.79 / 0.047 / 0.60 | |
| 5e-4 / 2e-3 / 5e-3 | `const=4` + final | 1.6e-4 / 3.4e-3 / 3.3e-2 | strongly negative | 1.8 / 2.2 / 2.7 (CIs +-0.1) | 0.003 | 0.003 (stretched, quadlog) | |
| 5e-4 / 2e-3 / 5e-3 | `const=1` + final | 6.1e-4 / 1.6e-2 / 0.128 | strongly negative | 2.9 / 4.0 / 1.3 (CIs +-0.1) | 0.003 | 0.003 | 0.38 / 0.073 / 0.052 |
| 5e-4 / 2e-3 / 5e-3 | `const=0` (sampled) | 1.1e-5 / 1.8e-4 / 1.0e-3 | strongly negative | 1.1 / 1.2 / 1.4 | 0.003 | 0.003 | 0.95 / 0.82 / 0.62 |

- A negative c2 means the decay accelerates with m.
- **f depends on the fit window** **[N]**. For `unif=1` + final at p = 5e-3, fits on
  [m0, 30] give r = 3.05e-3, 3.00e-3, 3.10e-3, 3.20e-3 [3.07, 3.31]e-3 for m0 = 0, 2, 10,
  20. For `const=1` + final the rate quadruples: 0.059 -> 0.128 -> 0.232 -> 0.235.
- **The sampled LRB-D curves agree with the exact ones** **[N]**. At p = 5e-3, acceptance
  z-scores have sd 1.00 to 1.01 and accepted-flip z-scores sd 0.91 to 0.98 over all
  usable cells. But sampling resolves `unif=1` only up to m = 14 at p = 5e-3
  (acceptance 4.8e-4 at 3e5 shots), which hides the curvature: sampled rho =
  -0.19 [-0.70, 0.39].
- **The two-sector law for `const=1` is only qualitative** **[N]**. The acceptance-fitted
  and F-fitted rates disagree, and the F fit is poor (D = 16, exact curves):

  | p | r_A | r_F | chi2_F/dof |
  |---|---|---|---|
  | 5e-4 | 0.0342 | 0.0082 | 701 |
  | 2e-3 | 0.128 | 0.109 | 1290 |
  | 5e-3 | 0.349 | 0.292 | 92 |

  The law misses the quadratic onset at small m (first-order errors cancel in a checked
  segment).

### 3.4 Check spacing

| p | family (window 2..30) | r_1 | r_2 / r_1 | r_4 / r_1 |
|---|---|---|---|---|
| 5e-4 | LRB-D `unif=k` + final (exact) | 2.42e-5 | 1.98 [1.96, 2.00] | 4.04 [4.00, 4.08] |
| 2e-3 | LRB-D `unif=k` + final (exact) | 4.15e-4 | 2.04 [2.02, 2.06] | 4.47 [4.42, 4.51] |
| 5e-3 | LRB-D `unif=k` + final (exact) | 3.00e-3 | 2.19 [2.17, 2.21] | 5.52 [5.46, 5.59] |
| 5e-4 | MLD on the `unif=k` rounds (sampled, same shots) | 0.0120 | 1.06 [1.05, 1.06] | 1.11 [1.11, 1.12] |
| 2e-3 | MLD on the `unif=k` rounds | 0.0528 | 1.09 [1.09, 1.10] | 1.20 [1.19, 1.20] |
| 5e-3 | MLD on the `unif=k` rounds | 0.152 | 1.11 [1.10, 1.11] | 1.21 [1.20, 1.21] |
| 5e-4 | exact second-order slope of S2u (LRB-D + final, 150 seq) | 47.3 p^2 | 1.91 | 3.92 |
| 5e-4 | exact first-order slope of S1 (MLD, 150 seq) | 11.6 p | n/a | 1.06 (`unif=4`) |

- With the same noise and the same rounds, the per-block LRB-D rate scales with the
  segment length k, while the decoded rate barely moves **[N]**.
- Under a memoryless block model (each block's undetected logical error independent of
  the others) the LRB-D ratio would be 1. The factor k is the signature of pairs of
  faults in different blocks of one segment cancelling each other's syndrome: memory
  carried by the syndrome register **[A]** (the pair count per segment is about k^2).
- For MLD, sparser information costs only 6 to 21%, because at d = 2 the dominant
  first-order failures are pair ambiguities that timing information cannot resolve
  **[C]**. Figure: `local_rates_spacing_exact.png`.

### 3.5 The inverse half (variant `inverse_check_every=1`, `inv1`)

Exact LRB-D curves, sampled MLD (30 sequences, 2e5 shots, depths 0..30 step 2), and the
exact low-order expansion (150 sequences at p = 5e-4):

| p | method | r (per block) | rho | p_GOF |
|---|---|---|---|---|
| 5e-4 / 2e-3 / 5e-3 | LRB-D all rounds incl. inverse checks + final (exact) | 4.3e-5 / 7.3e-4 / 5.2e-3 | 0.043 [-0.032, 0.12] / 0.042 [-0.031, 0.11] / 0.059 [-0.017, 0.14] | 0.17 / 0.25 / 0.30 |
| 5e-4 / 2e-3 / 5e-3 | LRB-D `unif=1` + final, inverse checks executed but unused (exact) | 5.7e-4 / 1.5e-2 / 0.118 | 2.9 / 4.0 / 1.5 | 0.003 |
| 5e-4 / 2e-3 | MLD, all rounds incl. inverse checks | 0.0228 [0.0224, 0.0233] / 0.0999 [0.098, 0.102] | 0.040 [-0.019, 0.10] / 0.035 [-0.016, 0.088] | 0.27 / 0.47 |
| 5e-3 | MLD, all rounds incl. inverse checks (usable to m = 22) | 0.285 [0.278, 0.291] | 0.003 [-0.14, 0.17] | 0.033 |

Exact low order, 150 sequences, p = 5e-4:

| circuit | quantity | slope | quadratic coefficient |
|---|---|---|---|
| standard | LRB-D `unif=1` + final, S2u | 47.3 p^2 | 0.051 [0.015, 0.088] |
| standard | its terminal-only part | 5.39 p^2 | 0.040 [0.021, 0.058] |
| standard | its forward part | 41.9 p^2 | 0.012 [-0.023, 0.047] |
| inv1 | LRB-D all rounds + final, S2u | 83.9 p^2 | 0.016 [-0.051, 0.083] |
| standard | MLD, S1 | 11.6 p | -0.0035 [-0.011, 0.004] |
| inv1 | MLD, S1 | 22.1 p | 0.0036 [-0.011, 0.019] |

- In inv1 the terminal-only (late) component of MLD becomes exactly constant in m.
- **Checking and using the inverse half removes the one resolved non-Markovian term of
  dense LRB-D** **[N]**: the LRB-D exact curves pass, and the m^2 pair term vanishes.
- **The price is noise**: the extra check rounds double the per-block rate, both for
  LRB-D (x1.7 to 1.8) and for MLD (x1.9).
- **Executing the inverse checks without using them is the worst option** **[N]**.
  Their noise lands in an unchecked segment, and the m^2 pair term grows 900-fold
  (quadratic coefficient 46 p^2).
- **Figures**: `F_inv1.png`, `local_rates_inv1.png`.

### 3.6 Soft-output postselection and m-dependent information

| p | MLD + accept `|LLR| >= t` | min over m of F (95% lower) vs floor tanh(t/2) | rho |
|---|---|---|---|
| 5e-4 | t = 1 / 2 | 0.962 / 0.974 vs 0.462 / 0.762 | -0.13 [-0.19, -0.07] / -0.036 [-0.08, 0.007] |
| 2e-3 | t = 1 / 2 | 0.762 (0.760) / 0.880 (0.879) vs 0.462 / 0.762 | -0.33 [-0.36, -0.31] / -0.60 [-0.63, -0.57] |
| 5e-3 | t = 0.5 / 1 / 2 / 4 | 0.369 / 0.558 / 0.813 / 0.969 vs 0.245 / 0.462 / 0.762 / 0.964 | -0.91 [-0.98, -0.85] (t = 1) |

- **The floor holds** at every p, t and m **[N]**, as derived **[A]**.
- **The decay flattens** (rho < 0, p_GOF = 0.003 at p >= 2e-3) while the acceptance
  falls (5e-3: 1.4e-3 at m = 30 for t = 1). Thresholded correction is non-Markovian by
  construction.
- **m-dependent information is non-Markovian too** **[N]**. MLD restricted to the
  `const=4` / `const=1` rounds gives rho = 0.17 / 0.25 (5e-4), 0.24 / 0.16 (2e-3) and
  0.21 / 0.13 (5e-3), all with p_GOF <= 0.03. That is about 10x less than LRB-D on the
  same rounds (1.8 to 4.0, paired, same shots), but clearly non-zero.
- **Figures**: `F_thresholds.png`, `A_postselected.png`, `F_mld_info.png`.

### 3.7 Exact low-order expansion (30 sequences at each p; 150 at 5e-4)

| p | quantity | slope per block | quadratic coeff. | late/early slope |
|---|---|---|---|---|
| 5e-4 (150 seq) | MLD S1/p | 11.6 [11.5, 11.7] | -0.0035 [-0.011, 0.004] | 0.99 [0.97, 1.01] |
| 5e-4 (150 seq) | MLD S1_late/p (terminal-only faults) | 0.578 [0.563, 0.594] | -0.0055 [-0.0072, -0.0039] | 0.68 [0.59, 0.80] |
| 5e-4 (150 seq) | null decoder S1/p | 21.0 [20.9, 21.1] | 0.0012 [-0.0068, 0.010] | 1.00 [0.99, 1.02] |
| 5e-4 (150 seq) | `const0` decoded S1/p | 0.734 [0.723, 0.746] | -3.5e-4 [-0.001, 4e-4] | 0.99 [0.95, 1.03] |
| 5e-4 (150 seq) | LRB-D `unif=1` + final S2u/p^2 | 47.3 [46.8, 47.9] | 0.051 [0.015, 0.088] | 1.03 [1.00, 1.07] |
| 5e-4 (150 seq) | LRB-D `unif=4` + final S2u/p^2 | 186 [184, 188] | 0.33 [0.20, 0.46] (forward part 0.001 [-0.13, 0.13]) | 1.03 [1.00, 1.06] |
| 5e-4 (150 seq) | LRB-D `const=1` + final S2u/p^2 | 1660 [1650, 1680] | 50 [50, 51] | 4.04 [3.98, 4.11] |
| 5e-4 (150 seq) | LRB-D `const=0` S2u/p^2 | 22.2 [21.5, 22.8] | 0.67 [0.64, 0.70] | 4.12 [3.89, 4.35] |
| 2e-3 / 5e-3 (30 seq) | MLD S1/p | 11.7 / 11.6 | 0.012 / 0.012 (CIs include 0) | 1.02 / 1.02 |
| 2e-3 / 5e-3 (30 seq) | MLD on `const=1` rounds S1/p | 16.2 / 20.3 | 0.084 [0.049, 0.12] / 0.32 [0.28, 0.37] | 1.19 / 1.78 |

- The decoder decisions are those of the exact MLD at that p, so S1/p changes with p only
  where those decisions change. That happens strongly for terminal-only information,
  where the multi-fault posterior matters.
- With 30 sequences the null-decoder and LRB-D quadratic terms came out about 2x larger
  than with 150 (for example 0.12 vs 0.051 for `unif=1`). Curvature estimates from 30
  nested sequences are noisy, and the 150-sequence values are the reliable ones **[N]**.
- **Figure**: `low_order.png`.

### 3.8 Cross-checks

- `nonmarkov_runs.py --selftest` **[N]** checks that:
  - the fast raw-syndrome map equals `LRBCircuit.raw_syndromes`;
  - the restricted detectors equal those of a circuit rebuilt with only the selected
    rounds (via `compile_m2d_converter`);
  - the restricted decoder on all rounds equals `mld5` (LLRs within 1e-8);
  - `run_cell` reproduces `experiment.run_point` counts and `|LLR|` histograms exactly on
    identical seeds;
  - noiseless circuits give no errors and no rejections.
- My exact null curves equal the lead's 200-sequence file on the shared 30 sequences
  (maximum difference 0.0), so both use the same sequences **[N]**.
- The sampled null decoder against the exact one at p = 5e-3 gives z-score mean 0.02
  and sd 0.96 over 930 cells **[N]**.

## 4. Why correction mitigates it

1. **No conditioning.** The decoded outcome is a fixed function of the syndrome and the
   readout, averaged over all shots. There is no normalization by a depth-dependent
   acceptance, so the h-transform mechanism of 1.4 (item 1) is absent by construction
   **[A]**.
2. **Leading-order failures are local and affine in m.** At d = 2, MLD fails at first
   order on about 11.6p per block of ambiguous single faults. Each such failure is
   decided by syndromes within about one round, so the weight is a sum of per-block
   terms **[N]** (affine S1, 150 sequences). In the language of 1.6, the decoded
   occurrence process has short memory, so `F = sum_k alpha_k lambda_k^m` is dominated by
   one eigenvalue.
3. **The residual decoder memory saturates.** Faults seen only by the terminal check (the
   inverse half) are decoded with a posterior spread over all m inverse blocks. Their
   failure weight is concave in m and small (0.58p per block against 11.6p) **[N]**.
   This kind of memory makes the per-block channel approach a limit as m grows, which is
   exactly the case of 1.6 where the decay becomes single-exponential **[A]**.
4. **LRB-D's leading term is second order and lives in segments.** Undetected logicals
   need two faults that cancel each other's syndrome inside one unchecked or unused
   stretch. The pair count grows like (segment length)^2. That gives:
   - the factor k in the rate for `unif=k`;
   - curvature for every `const` policy;
   - the m^2 term from the inverse half **[A, N]**.

   Correction turns each of those pairs into, at most, a first-order event in each
   block. The O(p) single-fault failures then dominate, and they carry no memory.
5. **Correction does not remove the physical mechanisms.** Pairs in the inverse half
   still exist, but their contribution to the decoded F is O(p^2 m^2) and is swamped by
   the O(p m) Markovian part. Postselecting on the decoder output brings the conditioning
   back (Section 3.6).

## 5. Claims

| # | claim | label | evidence |
|---|---|---|---|
| 1 | The Clifford twirl of any logical channel is depolarizing, so B = 0 for the signed polarization and Markovian means log F affine in m | A | Section 1.2 (Schur's lemma, octahedral group irreducible on span{X,Y,Z}) |
| 2 | U_j = C_j...C_1 are i.i.d. uniform; forward block j and inverse block j+1 share the frame U_j | A | Section 1.6, Lemmas 1-2 |
| 3 | Under assumption T, F(m) = E[c_0 (-1/3)^K] | A, N | proof in 1.6; toy Monte Carlo against the generating function, max \|z\| = 0.86 over m = 1..40 (`twirl_toy.json`) |
| 4 | A stationary, mixing finite-memory occurrence process gives an asymptotically single exponential with rate ln lambda_1 shifted by (16/9) sum_d Cov(a_0, a_d); a frozen rate gives a curved mixture | A, N | 1.6; toy: lambda_1 = 0.96646 vs 0.96000 independent; second-order estimate -0.0322 vs exact -0.0341; frozen-rate mixture matched (max \|z\| = 1.65) |
| 5 | For exact MLD with acceptance \|LLR\| >= t, F >= tanh(t/2) at every m | A, N | 1.5 item 4; Section 3.6, all p and t |
| 6 | Physical RB at 5p and the null decoder pass every Markovianity test | N | Section 3.1: rho CIs include 0; p_GOF 0.47-1.0 |
| 7 | Exact MLD correction decays as a single exponential (rho within +-5%, p_GOF >= 0.5, no better alternative) at p = 5e-4, 2e-3, 5e-3 | N | Section 3.2; at 5e-3 rho = 0.050 [0.004, 0.096] is marginal |
| 8 | MLD's first-order failure weight is affine in m; its only depth-dependent part (terminal-only faults) saturates and is 5% of the slope | N | Section 3.7: quadratic -0.0035 [-0.011, 0.004]; late part quadratic -0.0055 [-0.0072, -0.0039] (150 seq) |
| 9 | LRB-D `const=1`, `const=4` and `const=0` are strongly non-Markovian (rho = 1.1-4.0); their undetected weight grows like m^2 | N | Section 3.3 (exact curves); Section 3.7 (quadratic 50 p^2 and 0.67 p^2) |
| 10 | Dense LRB-D (`unif=1`) has a small, resolved acceleration (rho = 0.07-0.09) at p >= 2e-3; single exponential rejected | N | Section 3.3, exact curves: p_GOF 0.007 / 0.003; quadratic-log LR p <= 0.003 |
| 11 | That acceleration comes from fault pairs in the unchecked inverse half | N | Section 3.5: terminal-only pair term 0.040 [0.021, 0.058] p^2 vs forward 0.012 [-0.023, 0.047]; the term vanishes with inverse checks (0.016 [-0.051, 0.083]); exact inv1 LRB-D passes (p_GOF 0.17-0.30) |
| 12 | LRB-D's per-block rate scales with check spacing (r_k/r_1 about k); MLD on the same rounds scales by 1.06-1.21 | N | Section 3.4 (exact LRB-D, sampled MLD on identical shots; exact second order 1 : 1.91 : 3.92) |
| 13 | That k-scaling is memory: errors in different blocks combine through the syndrome register | A | Section 3.4 (pair counting per segment) |
| 14 | Correction keeps the Markovian shape of the no-postselection protocol and lowers its rate by 28-43%; it does not create the Markovianity | N | Section 3.1 vs 3.2; paired \|rho_MLD\| - \|rho_null\| = 0.000 [-0.008, 0.022] (5e-4), -0.007 [-0.013, 0.021] (2e-3), 0.028 [-0.009, 0.063] (5e-3) |
| 15 | The decoded rate is 50-500x the LRB-D `unif=1` rate on the same circuits | N | r = 0.0120 / 0.0528 / 0.152 vs 2.42e-5 / 4.15e-4 / 3.00e-3 |
| 16 | MLD with m-dependent information (`const=k` rounds) is non-Markovian (rho = 0.13-0.25), about 10x less than LRB-D on the same rounds | N | Section 3.6, paired on the same shots |
| 17 | Inverse checks used by the protocol remove the m^2 term; executed but unused, they make LRB-D strongly non-Markovian | N | Section 3.5 |
| 18 | The two-sector law with D = 16 fits `const=1` only qualitatively | N | Section 3.3: r_A and r_F differ by 15-76%, chi2_F/dof 92-1290 |
| 19 | For d = 2 MLD, timing information barely matters at first order because the failures are pair ambiguities | C | Section 3.4: S1 slope 11.6p (all rounds) vs 12.3p (`unif=4` rounds), 12.7p (`const=1` rounds), p = 5e-4 |
| 20 | Gate-dependent noise only adds a transient that decays exponentially | L | Wallman [L1] |
| 21 | Non-Markovian (memory) noise shows up as deviations from the Markovian sequence-fidelity decay | L | Figueroa-Romero et al. [L3]; temporally correlated noise also changes the distribution of sequence fidelities [L2] |

## 6. Bugs and caveats

- **No bug found in the lead's model.** `run_point`, `RawPolicyEvaluator`,
  `exact_condition` and `TransferMatrixMLD` agree with my independent runner and with each
  other (Section 3.8).
- **Seed reuse (minor, not a bug).** `run_point` seeds the `const0` sampler of cell
  (s, d) with `seed + 7 + 1000 s + d`. That is the main-circuit seed of cell (s, d + 7).
  The circuits differ, so no shots are duplicated, but the two streams share random
  numbers. My runner keeps the convention so that its counts match `run_point`
  exactly. It affects only cross-method covariances, not means.
- **Exact acceptances can be tiny.** The exact `unif=1` curves at p = 5e-3 reach
  acceptance 1.2e-7 at m = 30. They describe the infinite-shot limit. A real experiment
  could not see the curvature that the exact curve resolves (the sampled curve stops at
  m = 14 with 3e5 shots).
- **Statistics are sequence-limited.** With 30 nested sequences the curvature
  estimators are noisy: the 30-sequence low-order quadratic terms came out about 2x the
  150-sequence values. The verdict therefore rests on the 150-sequence expansion and on
  exact curves where possible.
- **The repo's `|P0 - P1|` estimator** is reported (`F_abs` in `analysis.json`) but not
  used. For postselected cells with few accepted shots it biases F upward.
- **Operational.** Several early chunks ran on CPUs pinned by `srun --overlap -c 24`
  (about 14% CPU each). One timed out and was rerun; later runs used `--cpu-bind=none
  -c 256` with 24 workers, as the lead advised.

## 7. Open questions

1. **The marginal +5% MLD drift at p = 5e-3** (rho = 0.050 [0.004, 0.096]; window rate
   0.158 [0.154, 0.163] at m0 = 20 against 0.152). It is not explained by first-order
   terms, and second-order inverse-half pairs give about 1%. It may be a multiple-
   comparison fluctuation **[C]**. 150+ sequences at 5e-3, or an exact second-order
   decoder expansion, would settle it.
2. **Per-block attribution for the decoder.** A genie instrumentation with noiseless MPP
   of the logical at block boundaries would give the actual K process. That would test
   the generating-function prediction F = E[(-1/3)^K] directly on the circuit, instead
   of on a toy.
3. **Causal and windowed decoders.** Global MLD uses future syndromes. A sliding-window
   or hard-decision decoder (WP on causal decoders) should keep the bulk Markovian with
   a transient of about one window; this was not tested here **[C]**.
4. **Hardware memory** (leakage, drift, 1/f noise). SI1000 has none. Leakage would add a
   slow mode in both A(m) and F(m). Correction would turn it into correlated logical
   bursts, which the generating-function argument predicts as a renormalized rate while
   the leakage lifetime is short **[C]**.
5. **A compiled recovery Clifford instead of the mirror inverse** would halve the
   unchecked segment, and should cut the LRB-D m^2 term by about 4x **[C]** (pair
   count).

## 8. Reproduce

From the repository root, on the shared allocation (see ALLOCATION.md; 24 workers,
`--cpu-bind=none -c 256`, each chunk under 20 minutes):

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --selftest
# sampled runs (one chunk per command), then merge (merge also runs physical RB at 5p)
for p in 5e-4 2e-3 5e-3; do for r in 0:10 10:20 20:30; do
  srun --jobid=<JOBID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 \
    python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --variant std --p $p --shots 300000 \
      --seq-range $r --workers 24; done
  python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --merge --variant std --p $p --rb-shots 200000
done
for p in 5e-4 2e-3 5e-3; do for r in 0:15 15:30; do
  srun ... python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --variant inv1 --p $p \
    --shots 200000 --depths 0:30:2 --seq-range $r --workers 24; done
  python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --merge --variant inv1 --p $p
done
# exact LRB-D / null curves and the exact low-order expansion (login node, 4 workers)
nice -n 19 python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --exact --p 5e-4,2e-3,5e-3 --workers 4
nice -n 19 python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --exact --variant inv1 --p 5e-4,2e-3,5e-3 --depths 0:30:2 --workers 4
nice -n 19 python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --low-order --p 5e-4,2e-3,5e-3 --workers 4
nice -n 19 python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --low-order --variant inv1 --p 5e-4,2e-3 --depths 0:30:2 --workers 4
nice -n 19 python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --low-order --p 5e-4 --num-seq 150 --depths 0:30:2 --workers 4
nice -n 19 python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --low-order --variant inv1 --p 5e-4 --num-seq 150 --depths 0:30:2 --workers 4
# analysis: tables, JSON, figures (about 2 minutes)
python studies/correction-decoders/qubit/scripts/nonmarkov_analysis.py
```

Outputs:

- `results/F/`:
  - `std_<ptag>.npz`, `inv1_<ptag>.npz` (`PointResult` layout) and `rb_<ptag>.npz`;
  - `exact_*.npz`, `loworder_*.npz`;
  - `analysis.json`, `note_tables.md`, `summary_*.md`, `twirl_toy.json`;
  - `chunks/` and `logs/`.
- `figures/F/*.png`.

## References (each fetched from its arXiv abstract page this session)

- [L1] J. J. Wallman, "Randomized benchmarking with gate-dependent noise", Quantum 2, 47
  (2018), arXiv:1703.09835.
- [L2] H. Ball, T. M. Stace, S. T. Flammia, M. J. Biercuk, "The effect of noise
  correlations on randomized benchmarking", Phys. Rev. A 93, 022303 (2016),
  arXiv:1504.05307.
- [L3] P. Figueroa-Romero, K. Modi, T. M. Stace, M.-H. Hsieh, "Randomized benchmarking
  for non-Markovian noise", PRX Quantum 2, 040351 (2021), arXiv:2107.05403.
- [L4] J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized
  Benchmarking", arXiv:1702.03688 (2017). This is the background protocol.
