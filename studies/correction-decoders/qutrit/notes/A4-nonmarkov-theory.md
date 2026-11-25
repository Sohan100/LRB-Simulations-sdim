# A4: Non-Markovianity in qutrit LRB-D: definitions, exact structure, predictions and tests

Agent A4, correction-decoders / qutrit-512 study. Read `00-brief.md` first.

Labels used for every claim:

- **[VA]** verified analytically (derivation given here),
- **[VN]** verified numerically (script and output quoted),
- **[LIT n]** from literature (reference n in section 8; every paper was fetched in this task),
- **[CONJ]** conjecture.

Scripts (all under `notes/scripts/`, run with `module load python/3.11`; one core, each 1 to 8 minutes):

| script | what it does |
|---|---|
| `A4_lossy_markov.py` | one-qutrit toy for section 2: Clifford 2-design check, fixed lossy map, twirled map, random-sequence RB with per-sequence normalisation |
| `A4_exact_lrbd.py` | **exact** (no sampling) acceptance and accepted fidelity of the real baseline circuits for any check policy and any p, from the repository DEM (section 3.1); terminal ML decoding variant; validation against the baseline shot counts |
| `A4_analyze_exact.py` | validation z-scores, small-p expansion coefficients, exact curve shapes |
| `A4_attribution.py` | removes the faults of one circuit region and recomputes the second-order coefficient (which pairs of faults cause the decay) |
| `A4_hmm_toy.py` | 243-state hidden-Markov toy of detection vs correction (section 4) |
| `A4_fit_battery.py` | the test battery of section 5 run on the per-sequence baseline counts |

Numbers below quoted as "exact" come from `A4_exact_lrbd.py` on Clifford sequences 0 to 5 (of 30), all 9 depths, policies unif 1, 2, 3, 4, 6, const 0, 1, 2, 3, 4.

---

## 0. Summary

1. In this simulation the physics is Markovian. Non-Markovianity appears only at the logical level, after the
   code block's Pauli frame is coarse-grained to the logical qutrit. The memory carrier is the out-of-codespace part of the
   frame, the syndrome register sigma in Z_3^4. With correction, the decoder state is a second carrier. Acceptance never
   depends on the encoded logical state (F1, [VA]). So every non-exponential feature comes from one of four sources:
   memory in sigma; a check schedule that depends on m; the boundary at m = 0; or estimator effects.
2. A Markovian but lossy map E gives a postselected normalised expectation tr[O E^m(rho)]/tr[E^m(rho)] that is a
   **ratio of two exponential polynomials**. It is exactly a single exponential if O and the identity are both
   eigen-operators of the adjoint map E^dag, for example E^dag(I) = s I together with a twirled form alpha Id + beta Dep.
   Then f = alpha/s. In random-sequence RB, the ratio of sequence averages is always exactly (alpha/(alpha+beta))^m.
   The mean of per-sequence ratios, which is the statistic the pipeline uses, is exactly exponential only if acceptance is
   state independent ([VA], [VN]).
3. **The exact character expansion** (section 3.1, [VA]) writes LRB-D acceptance and accepted fidelity, for every circuit
   and policy, as a ratio of two sums of products of factors (1 - kappa p). After a change to difference detectors the
   sums become products of 81 x 81 transfer matrices. The evaluator reproduces the 1e6-shot baseline counts per sequence
   (z-scores: mean -0.1, sd 1.0; [VN]).
4. **Leading-order structure** ([VN] from the exact small-p expansion, [VA] for the mechanism):
   -ln F(m) = a1 p + a2(m) p^2 + O(p^3) for every policy that has a noisy terminal check.
   - a1 = 7.425 is m-independent. It equals 3/2 times the 4.95 p rate of single faults after the last relevant check in
     the final round.
   - a2(m) is set by **pairs of faults inside unchecked intervals**: a persisting detectable error is cancelled by a
     second error into a logical error. An interval of L forward steps contributes about 22 L^2.
   - Hence unif k has a per-step rate of about 22.7 k p^2 (22.7, 45.8, 68.3 for k = 1, 2, 3), const c has an m^2/c term
     (gamma = 26.8, 13.6, 8.8 for c = 1, 2, 3), and const 0 is almost purely quadratic: -ln F = 2.24 m^2 p^2.
   - The m-step unchecked inverse block adds a small m^2 term to unif 1 (gamma about 0.05 to 0.17) and terms linear in m.
5. **Data signatures** ([VN], `A4_fit_battery.py`, 30 sequences x 1e6 shots):
   - const 0 and const c >= 1 strongly reject a single exponential. At p = 1e-3, chi^2 of b f^m is 632 for const 0 and
     5461 for const 1; the log-quadratic model gives 1.5 and 68. The curvature d is non-zero with CI excluding 0.
   - unif k (k = 3, 4, 6) show a **sawtooth** in m mod k: chi^2 drops from 75 to 1.7 for unif 4 at 1e-3 when the
     remainder term is added.
   - unif 1 is statistically indistinguishable from a single exponential at this sample size. Its exact curvature is
     about 0.5 of the bootstrap standard error.
   - For unif 1 the m = 0 point sits 2.4 to 6.6 sigma above the single-exponential extrapolation from m >= 2 (a
     boundary effect).
   - The fidelity can **increase** with m when more checks switch on (const 4, m = 2 to 4, paired z = 2.9 and 3.5).
     A rising average sequence fidelity therefore does not witness non-Markovianity in postselected RB with m-dependent
     schedules.
6. **Correction** ([VN] on the real circuits for terminal ML decoding; [VN] in the toy for decoding every round; [CONJ]
   for general decoders).
   - Correction removes the normalisation. It turns the quadratic pair mechanism of unchecked intervals into a linear,
     single-fault mechanism, so the relative drift of the decay rate falls from about 0.6 to 20 under detection to about
     0 to 0.45 under correction.
   - At distance 2 this costs a first-order logical error rate: about 2e-3 per step against about 3e-5 at p = 1e-3.
   - A memoryless decoder that runs every round keeps the hidden-state memory short. In the toy the second-to-first
     eigenvalue ratio is 0.016 against 0.0035 for detection.
   - Correction therefore mitigates non-exponential behaviour in shape, but not the logical error.
7. **Battery and design.** Use the real-part fidelity estimator, because the pipeline's |.| estimator is Rice-biased
   (+1 sigma) once F < 5 SE. Use a cluster bootstrap over sequences, the curvature and sawtooth parameters, windowed
   rates, a surrogate fitted on m = 2, 4, 6, and the m = 0 test. Replace the m mirrored inverse Cliffords by one
   compiled recovery Clifford, or check inside the inverse block, to remove the only memory interval that every policy
   shares.

---

## 1. Definitions

### 1.1 The process

A depth-m LRB-D circuit (brief, "Circuit structure") is a sequence of steps. Step t (t = 1..m) is the forward Clifford
g_t followed by check round t-1 and a reset. Then come the m inverse Cliffords, all unchecked, then the final round m and
the ideal readout of x0 + x3. A policy pi selects the used rounds U_pi(m), a subset of {0..m} that always contains m.
Every element of the physical circuit is a fixed CPTP map or instrument on 9 qutrits, with Weyl noise channels, Clifford
unitaries, measurements and resets. **At the physical level the process is Markovian** [VA]: by construction of the
SI1000 model there is no external bath and no correlated noise.

Split the code block into the logical qutrit S and an environment E. E holds everything that carries information from one
step to the next:

- (E1) the data-block Pauli frame modulo its logical part. For Pauli noise only its syndrome sigma in Z_3^4 matters, since
  the stabiliser part acts trivially. This is the out-of-codespace state;
- (E2) ancilla states between rounds. They are reset every round, so their only effect is reset faults, which are
  transient;
- (E3) leakage populations (not modelled here);
- (E4) for correction, the decoder's internal state (Pauli frame, stored syndromes).

Postselection makes each step's accepted branch a CP trace-non-increasing map Lambda_t(g_t) on S (x) E.

### 1.2 Markovian logical noise

**Definition M.** Logical noise is *Markovian* (possibly gate-dependent, possibly trace-decreasing) if there are CP
trace-non-increasing maps Phi_t(g) on S alone, a logical input rho_0 and an effect Q such that for every sequence and m the
accepted unnormalised logical output is Phi_T(g_T) o ... o Phi_1(g_1)(rho_0). The maps may depend on the gate and on t. They
may not depend on earlier gates or outcomes, so there is no memory beyond the logical state.

Equivalently, the process tensor factorises into independent single-step maps with an uncorrelated initial state. This is
Eq. (4) of Figueroa-Romero et al. [LIT 1]. Pollock et al. derive a necessary and sufficient operational condition for a
quantum process to be Markovian, together with operational measures of non-Markovianity [LIT 4; abstract only]. In
Figueroa-Romero et al. the noise maps act on S (x) E and are "at least completely positive and trace non-increasing"; the
average sequence fidelity (ASF) is F_m = tr{M tr_E Lambda_{m+1} (A_m + B_m)(rho)}, where A_m and B_m are built from two
environment maps $_Lambda and Theta_Lambda. For time-independent noise these enter as matrix powers ($ - Theta)^m and
Theta^m, and the paper says this "generally does not give an exponential decay" unless the environment is superfluous
[LIT 1].

**Definition NM.** Logical noise is *non-Markovian* when no factorised description exists. The process tensor
Upsilon_m(S; E) then carries correlations between steps through E. Figueroa-Romero et al. quantify the deviation from
Markovianity as N_q = ||F_m - F_m^(M)||_q, a q-norm over sequence lengths. Here F^(M) is the ASF of a Markovianised
process with Lambda^(M)(sigma) = tr_E Lambda(eps (x) sigma) for a fixed environment state eps [LIT 1]. For a finite group
gate set, their 2022 framework writes the ASF as a sum over irreducible representations of quality maps that are
contracted through the environment and do not factorise in time [LIT 2].

### 1.3 Three effects that are *not* non-Markovianity but mimic it in LRB-D

- **Gate dependence (Markovian).** Physical gate counts depend on the Clifford: a logical Z is two single-qutrit gates,
  a logical H is a SWAP plus five single-qutrit gates. Wallman proves that the RB average survival probability with
  arbitrary gate-dependent noise is A p^m + B t^m + eps_m with |eps_m| <= delta_1 delta_2^m. Here t is the
  largest-magnitude eigenvalue of E_G(G~_G); it describes loss of trace and equals 1 for trace-preserving noise. The
  proof needs only Hermiticity-preserving maps, not trace preservation, and the paper says this covers loss from the
  code space [LIT 3].
- **Protocol time-inhomogeneity.** The set of used rounds depends on m: const c uses rounds j(m//c) - 1, and unif k leaves
  m mod k unchecked forward steps before the inverse block. The depth-m circuits are therefore not prefixes of one fixed
  process. Each circuit can be Markovian while the family over m is not a semigroup.
- **Estimator effects.** The pipeline averages per-sequence ratios and uses the modulus |sum_k omega^k P(k)| (sections
  2.4 and 5.1).

### 1.4 Structural facts used throughout

- **F1 [VA]. Acceptance is independent of the logical state.** For Pauli frames, stabiliser outcomes equal the commutation
  phases of the frame with the stabilisers plus measurement shifts. Every code state is a +1 eigenstate of every
  stabiliser, so no syndrome outcome depends on the encoded state. For every hidden state, the accepted map therefore
  satisfies Phi^dag(I_L) proportional to I_L.
- **F2 [VA]. The process is a classical hidden Markov model.** With Weyl noise and Clifford gates the joint (S, E) dynamics
  is a random walk on the frame group. Gates act linearly; a logical Clifford is a code automorphism, so it maps sigma to
  A_g sigma. Each fault adds a uniformly random non-zero vector with fixed probability. The readout shift b is linear in
  the frame. The accepted fidelity is |E[omega^b | acc]|, and E[omega^b | acc] itself is real because the noise is symmetric under
  e -> -e (P(1) = P(2)).
- **F3 [VA]. A detectable error persists.** A code automorphism maps a non-zero syndrome to a non-zero syndrome, and check
  circuits never alter an existing data error. A single detectable data error therefore stays visible in every later used
  round until a second error cancels it. Weight-1 errors are always detectable (d = 2, stabilisers of weight 3), and the
  pair {2,3} carries no non-trivial normaliser element, so single faults on CZ(2,3) or SWAP(2,3) are detectable too.
- **F4 [VN].** The single-qutrit Clifford group has 216 elements modulo phase and is a unitary 2-design: its frame
  potential is 2.000000000000 (`A4_lossy_markov.py` (a)).

---

## 2. Markovian but lossy: is the normalised postselected curve one exponential?

### 2.1 General form [VA]

Let E be CP and trace non-increasing on a d-dimensional system, with Liouville matrix R. Take the Jordan form
R = sum_k (lambda_k P_k + N_k) with N_k nilpotent. Then R^m = sum_k sum_{j < size_k} C(m, j) lambda_k^{m-j} P_k N_k^j, so

    N(m) = tr[O E^m(rho)] = sum_k q_{O,k}(m) lambda_k^m ,     D(m) = tr[E^m(rho)] = sum_k q_{I,k}(m) lambda_k^m ,

where q_{X,k} are polynomials of degree below the Jordan block size (constants if R is diagonalisable). The normalised
postselected expectation r(m) = N(m)/D(m) is a **ratio of two exponential polynomials**. It equals C f^m for all m only if
the exponential-polynomial identity N(m) = C f^m D(m) holds term by term, which is non-generic.

**Large m.** For a CP map the spectral radius lambda_1 is an eigenvalue with a positive semidefinite eigenvector v_1. If
q_{O,1} is non-zero, r(m) tends to the constant tr[O v_1]/tr[v_1]: the postselected state converges to a quasi-stationary
state. Otherwise r(m) ~ (lambda_O/lambda_1)^m, where lambda_O is the largest eigenvalue whose term appears in N. The
asymptotic decay rate is a *ratio* of eigenvalues, and r(m) approaches it at rate |lambda_2/lambda_1|^m.

[VN] `A4_lossy_markov.py` (b) uses a random untwirled lossy qutrit map with acceptance operator diag(1, 0.9, 0.75). Its
windowed rate goes 0.928 (m 0 to 1), 0.913 (10 to 11), 0.984 (39 to 40), which is non-monotone. r(200) = 0.064305 and
r(400) = 0.064302 match the Perron-vector prediction 0.064302.

### 2.2 When it is exactly one exponential [VA]

Sufficient condition: the identity and O are eigen-operators of the adjoint map,

    E^dag(I) = s I   and   E^dag(O) = alpha O   =>   r(m) = (alpha/s)^m tr[O rho]/tr[rho]   exactly.

The first equation says acceptance does not depend on the state. A twirled map E_tw = alpha Id + beta Dep, with
Dep(rho) = tr(rho) I/d, satisfies both for every traceless O, with s = alpha + beta, so f = alpha/(alpha+beta).
If E^dag(I) = sI, then E/s is CPTP and r(m) is the ordinary RB expectation of the channel E/s. The sum of exponentials
then sits only in the numerator and collapses to one term after twirling.

[VN] `A4_lossy_markov.py` (c): the twirl of the toy map is a Id + b Dep with a = 0.825985 and b = 0.057349 (residual
1e-15), and max over m of |r(m) - (a/(a+b))^m| is 1.3e-15.

### 2.3 Random-sequence RB with per-sequence normalisation [VA, VN]

Take gate-independent noise E after every uniformly random Clifford, with ideal inversion.

- **Ratio of sequence averages.** E[N]/E[D] = (alpha/(alpha+beta))^m exactly, for any trace-decreasing E. This follows
  from the standard twirl identity: the averaged superoperator is E_tw^m. [VN] the exact relative deviation is 1e-14 for
  both a state-independent and a state-dependent acceptance.
- **Mean of per-sequence ratios**, which is what the pipeline reports. It equals (alpha/s)^m exactly when E^dag(I) = sI,
  because D is then the same deterministic s^m for every sequence. [VN] 40000 sequences: the fitted f is 0.957830 against
  0.957829, chi^2/dof = 0.31. With state-dependent acceptance (E^dag(I) eigenvalues 0.25, 0.55, 1) it is *not*
  exponential: chi^2/dof = 28.9 for a single exponential, the fitted f is 0.868 against 0.882, and the deviation at m = 30
  is -1.1e-2 with SE 1.2e-3. The direct Monte Carlo estimate of E[D] is heavy-tailed in that case (effective sample size
  11112 of 40000), which is another reason not to work with unnormalised averages.
- **Gate-dependent noise with state-independent acceptance per gate.** If E_g^dag(I) = s_g I, each per-sequence ratio is
  tr[O E~_{g_m}...E~_{g_1}(rho)] with CPTP maps E~_g = E_g/s_g. Wallman's theorem [LIT 3] then applies to the normalised
  maps and gives A p^m + eps_m for traceless O (B = 0). Normalising per sequence costs nothing in the Markovian case.

### 2.4 Consequence for LRB-D [VA]

By F1 the per-step acceptance never depends on the logical state. If the environment E were reset to a fixed state every
step (Markovian), the LRB-D curve would therefore be one exponential up to Wallman's exponentially small eps_m. All
non-exponential behaviour in this simulation must come from:

1. memory in E (section 3);
2. protocol time-inhomogeneity and the m = 0 boundary;
3. estimator effects.

On item 3: the modulus |Z^| of the complex average has a Rice bias when the true F is a few standard errors or less.
[VN] For const 1 at p = 1e-2 and 2e-2, the modulus estimator gives mean z = +0.65 and +0.70 against the exact values
(n = 54 independent circuits; +1.0 in cells with F < 5 SE). The real-part estimator
(n0 - (n1+n2)/2)/n_acc gives +0.11 and +0.04. Under the symmetric noise of F2 the real-part estimator is unbiased; the
modulus is not. Without this correction the modulus produces a spurious fidelity floor of order n_acc^(-1/2), which looks
like a slow second exponential.

---

## 3. Mapping the mechanisms onto the LRB-D circuit

### 3.1 Exact character expansion and transfer matrices [VA, VN]

Write the detector record as D = sum_j A_j e_j mod 3 and the logical shift as L = sum_j l_j . e_j. The independent fault
vector e_j is uniformly non-zero in Z_3^{k_j} with probability p_j: k = 1 for the 'f' channel, 2 for N1 depolarising,
4 for N2. Using 1{D_U = 0} = 3^{-|U|} sum_u omega^{u.D_U},

    Z_w := E[ omega^{w L} 1{D_U = 0} ] = 3^{-|U|} sum_{u in Z_3^{|U|}} prod_j phi_j( A_j^T u + w l_j ),
    phi_j(v) = 1 if v = 0,   1 - kappa_j p_j otherwise,   kappa_j = 3^{k_j}/(3^{k_j} - 1)  (9/8, 3/2, 81/80).

Acceptance is A = Z_0, and the accepted fidelity is F = Z_1/Z_0, a ratio of two sums of products of factors (1 - kappa p)
in [0, 1]. Over a time-homogeneous stretch each product behaves like an exponential in m, so the general form of
section 2.1 holds exactly with an enormous number of terms.

The sum over u is exponential in the number of used detectors, but it factorises. Define difference detectors between
consecutive used rounds, delta_i = D_{r_i} - A_i D_{r_{i-1}}, where A_i (4 x 4 over GF(3)) is the syndrome automorphism of
the Clifford block in between. Change the dual variables triangularly, v_i = sum_{i' >= i} (A_{i'}...A_{i+1})^T u_{i'}, which
is a bijection. Every fault then touches at most two consecutive used rounds: a data fault is seen fully from the next round
on, a fault partly through a check is seen partially in that round and fully in the next, and a transient appears as t in
round i and -A t in round i+1. Hence

    Z_w = G_w * 1^T diag(s_1) T_1 diag(s_2) ... T_{R-1} diag(s_R) 1 / 81^R ,   T_i, s_i of size 81.

`A4_exact_lrbd.py` fits A_i from data-only faults born before round r_{i-1}, asserts the banding for every circuit (no
violation occurred), stores integer counts per fault type and evaluates any p in milliseconds. It also inverts the last
vector by an 81-point Fourier transform, which gives the joint law of (L, final syndrome) and hence exact terminal ML
decoding (section 4).

**Validation [VN].** Cells with at least 1000 accepted shots, compared with the per-sequence baseline counts:

| p | acceptance z mean / sd | fidelity z mean / sd | cells |
|---|---|---|---|
| 1e-3 | -0.16 / 0.98 | -0.10 / 1.07 | 534 |
| 5e-3 | -0.02 / 1.07 | -0.02 / 0.95 | 402 |
| 1e-2 | -0.04 / 0.98 | real-part estimator: +0.11 / 1.03 (const 1) | 288 |
| 2e-2 | -0.14 / 0.84 | real-part estimator: +0.04 / 0.85 (const 1) | 215 |

Independent cells (unif 1 only, one per circuit) at 1e-3 give z mean -0.22 / -0.11 with sd 1.01 / 1.14. Runtime is about
80 s per sequence on one core, including DEM compilation.

### 3.2 Catalogue of sources of non-exponential or history-dependent behaviour

| # | mechanism | memory carrier | order | m-scaling | policies |
|---|---|---|---|---|---|
| S1 | postselection normalisation (ratio of sums) | none by itself (F1) | - | harmless without memory (2.4) | all |
| S2 | detectable data error persists until the next used round and is cancelled by a second error with the opposite syndrome into a weight-2 logical | sigma (E1) | p^2 | about L^2 per unchecked interval of length L | unif k >= 2, const c, const 0 |
| S3 | m unchecked inverse Cliffords and one final check: pairs inside the block (quadratic), pairs with the final round and the last forward tail, and masking of inverse-block errors by final-round measurement faults (linear) | sigma | p^2 | gamma_I m^2 + O(m) | **every** policy |
| S4 | measurement and reset faults: transient detections; masking of a data error | ancilla (E2) for one round; sigma if masked | p (acceptance), p^2 (fidelity, final round) | adds r_M per *used* round | used rounds |
| S5 | partial final check: faults after the last relevant check in round m are silent | none | p | m-independent factor b | all except const 0 |
| S6 | m = 0 boundary: no preceding round tail or inverse block; second-order offset jumps between m = 0 and m >= 1 | - | p^2 | step at m = 0 | all |
| S7 | schedule depends on m: unif k sawtooth in m mod k; const c spacing m//c grows; F can rise with m when checks switch on | - (protocol) | p^2 | sawtooth, non-monotone | unif k, const c |
| S8 | leakage (not modelled) | E3 | - | extra exponential (leakage/seepage) | all, on hardware [CONJ] |
| S9 | gate dependence: Clifford-dependent gate counts | none | p | exponentially small correction [LIT 3]; sequence variance | all |

### 3.3 Per-step rates and leading-order formulas

Definitions (probabilities per unit p, first order):

- r_0 = **4.95** p: single faults in the final round that occur after the relevant check and flip the readout
  [VA]. From the brief's circuit: Z-components of 2p idle depolarising on wire 0 during the M7 and M8 layers give
  2 x 2 x 6/8 = 3; on wire 3 during M8, 1.5; and N2 after CNOT 7->0 and CNOT 8->3 contribute 18/80 each. There is no
  reset after round m.
- c_0 = **63.2** p: detected faults in the final round.
- r_step = **71.1** p: persistent detectable data faults per unit depth (one forward Clifford, its check round and reset,
  and one inverse Clifford). Most of it is the 2p idle noise on 5 data wires in 5 layers per round.
- r_M = **41.0** p: transient detection probability added by each *used* intermediate round [VA]. It comprises
  4 x 5p measurement shifts, 4 x 0.75p measured-ancilla depolarising, 4 x 2p reset shifts, and 6 x 2p x 0.75 ancilla idle
  before measurement, plus about 1p from CNOTs.
- r_Z = **8.6** p: Z-type gate faults per unit depth, the only faults const 0 detects.
- beta_1 = **about 23** (units of p^2 per step, in -ln F): 3/2 times the probability per step that two faults within
  one check interval combine into a silent readout-flipping logical.

**Acceptance [VN] (exact first-order coefficients, 6 sequences):**

    -ln A(m) = c_0 p + r_step m p + r_M n_used(m) p + c2(m) p^2 + ...,
    n_used = m (unif 1), floor(m/k) (unif k), c - 1 (const c <= m), 0 (const 1); const 0: -ln A = r_Z m p.

The linear fits of c1(m) give these slopes: 112.07 (unif 1), 91.57 (unif 2), 84.74 (unif 3), 81.32 (unif 4),
77.90 (unif 6) and 71.07 (const 1). These equal 71.07 + 41.0/k to the quoted precision. The second-order coefficient c2 is negative and
grows like L^2, because syndrome cancellations raise acceptance: c2(22) = -269 (unif 1) and -32629 (const 1). Acceptance
is therefore log-convex.

**Accepted fidelity [VN] (exact small-p expansion, sequences 0 to 5):**

    -ln F(m) = a1 p + a2(m) p^2 + O(p^3),     a1 = 7.425 = (3/2) r_0  for every policy and m (const 0: a1 = 0).

| policy | fit a2(m) = alpha + beta m + gamma m^2 (m >= 2) | reading |
|---|---|---|
| unif 1 | 165 + 22.7 m + 0.17 m^2 | forward intervals of length 1; small inverse-block m^2 |
| unif 2 | 135 + 45.8 m + 0.14 m^2 | beta = 2.0 beta_1 |
| unif 3 | 209 + 68.3 m + 0.31 m^2 | beta = 3.0 beta_1; sawtooth |
| unif 4, 6 | 129 + 111 m - 0.43 m^2; 273 + 117 m + 1.3 m^2 | sawtooth dominates the fit |
| const 1 | 133 + 84 m + 26.8 m^2 | one interval of length m |
| const 2 | 207 + 37 m + 13.6 m^2 | gamma halves |
| const 3 | 253 + 36 m + 8.8 m^2 | gamma / 3 |
| const 4 | 220 + 59 m + 5.3 m^2 | about gamma / 4 to 5 (floor effects) |
| const 0 | -5 + 0.7 m + 2.24 m^2 (a1 = 0) | purely quadratic |

**Region attribution [VN]** (`A4_attribution.py`, sequences 0 to 2). Removing a region removes every pair that touches it.

- unif 1: a2 = 160 + 24.3 m + 0.05 m^2. The parts are forward-forward pairs (18.3 m), pairs inside the inverse block or
  with the final round (4.3 m + 0.05 m^2), forward-inverse cross pairs (about 2 m), and the final round alone (111).
- const 1: a2 = -9 + 122 m + 24.4 m^2, of which forward-forward pairs give 21.6 m^2 and forward-inverse cross pairs about
  3 m^2.

The inverse block alone, with gate noise only, has a detectable rate r_I of about 3.4 p per Clifford [VA estimate from
gamma_I = (3/2) phi r_I^2 / 2].

**Interval law [VA mechanism, VN coefficients].** Let phi be the fraction of fault pairs in one interval that are
complementary and readout-flipping, about 0.6 to 0.7 % here (from beta_1 = (3/2) phi r_step^2 / 2). An unchecked interval
of L forward steps that ends at a used round contributes about pi_2 L^2 + (boundary-linear) to a2, with pi_2 of order
beta_1, about 22. Hence:

    unif k :  -ln F ~ a1 p + [alpha + k beta_1 m + e (m mod k) + gamma_I m^2] p^2
    const c:  -ln F ~ a1 p + [alpha' + beta' m + (gamma_1/c) m^2] p^2,    gamma_1 ~ 24-27
    const 0:  -ln F ~ gamma_0 m^2 p^2,                                     gamma_0 = 2.24
    unif 1 :  -ln F ~ 7.425 p + (160..165 + 23..24 m + 0.05..0.17 m^2) p^2

[VN] The data agree with these forms. const 0 at p = 1e-3 has (1 - F)/m^2 = 2.0 to 2.3 e-6 for m = 2..22, against
gamma_0 p^2 = 2.24e-6. The sawtooth coefficient fitted to the data, e of about 1.3e-4 at p = 1e-3 (about 130 p^2,
section 5), agrees with the exact a2 jump of unif 4 between m = 18, 20, 22: about 135 p^2 per unchecked remainder step.

**Validity.** The expansion needs the expected number of detectable faults in the longest unchecked interval to be well
below 1, that is r_step L p << 1. For unif 1 at 1e-3, r_step p = 0.07 per interval and r_I m p = 0.075 for the inverse
block at m = 22, so the expansion is fine. For const 1 at 1e-3, r_step m p = 1.6 at m = 22. That curve has left the
quadratic regime: the second-order term explains 0.015 of the observed 0.026 in -ln F(22)/F(0).

### 3.4 Beyond leading order: saturation [VA, VN]

With terminal-only checks the frame walk mixes. The syndrome becomes uniform, so A tends to 1/81, and the logical class
becomes uniform, so F tends to 0. In the character expansion, the u = 0 term of Z_0 equals 1/81 exactly (all its
factors are 1) and every other term decays. F(m) is then governed by the slowest-decaying character carrying the logical, set by the rate at which
errors fail to commute with the minimum-weight dressed logical (weight 2).

[VN] For const 1, the exact A(22) is 0.0123 at p = 1e-2 and 2e-2, and the data give 0.0123 to 0.0124 for p >= 2e-2,
against 1/81 = 0.012346. With the real-part estimator, F fluctuates around 0 (+-1e-3).

The ln F of const 1 therefore crosses over from quadratic at small m p to linear at large m p. Its windowed rate rises and
then plateaus.

### 3.5 Qualitative predictions for unif k and const c [VA; VN where marked]

- **unif 1.** Nearly exponential. The per-step rate is about 23 p^2, plus a weak concave term from the inverse block
  (gamma_I m^2) and a second-order step between m = 0 and m >= 2 (S6). [VN] The exact windowed rate per step goes from
  0.999969 to 0.999951 (m 2->4 to 20->22) at p = 1e-3, and from 0.999089 to 0.998503 at 5e-3.
- **unif k, k >= 2.** The per-step rate is about k beta_1 p^2: longer check spacing means longer-lived detectable errors
  and more cancelling pairs. A **sawtooth** follows the number m mod k of unchecked forward steps that precede the inverse
  block. When the depth grid crosses a multiple of k the curve can rise: [VN] exact unif 3 windowed rate 20->22 is 1.0015
  at 5e-3.
- **const c.** Gaussian onset, ln F ~ -beta_1 m^2 p^2/c. The windowed rate grows linearly with m (drift about
  2 gamma p^2/c per step), then crosses over to the asymptotic logical rate. When m < c the policy is terminal only, so the
  first depths behave like const 1 until checks switch on. F can increase there: [VN] exact const 4 windowed rate 2->4 is
  1.000004 (1e-3) and 1.00036 (5e-3); data paired z = 2.9 and 3.5.
- **const 0.** Pure Gaussian onset with no offset, then the same crossover.
- **Single vs double exponential.** A sum of two exponentials approximates a Gaussian onset poorly. [VN] For const 1 at
  1e-3: chi^2 = 128 (double exponential) against 68 (log-quadratic). The natural leading-order models are log-quadratic
  (const c, const 0) and log-linear plus sawtooth (unif k).

---

## 4. Correction: a hidden Markov model with a decoder

### 4.1 Formulation [VA]

With a decoder, step t maps the hidden state X_t = (sigma_t, l_t, dec_t) to X_{t+1}. Here sigma_t is the residual syndrome
of the frame after the applied correction, l_t its logical class, and dec_t the decoder memory: for example the last
syndrome, or a window of records. The map is stochastic because there is no rejection. The readout fidelity is a **linear**
functional of the state distribution:

    F(m) = | sum_x  w(x) [T_m ... T_1 pi_0](x) |,       w(x) = omega^{b(x)}.

For a periodic block T this is sum_i c_i lambda_i^m: a sum of exponentials **without** the ratio of section 2. The
stationary eigenvalue 1 carries no readout weight, because the logical class is uniform there. The logical mode lambda_L
sets the decay.

**Memory measures:**

- mu = |lambda_next| / |lambda_L|, the largest other eigenvalue that carries readout weight relative to the logical
  mode. The memory time is tau = -1/ln mu.
- The rate-drift index eta = ln f_late / ln f_early - 1, from depth-windowed rates f.
- For detection, the analogue of mu is the ratio of sub-leading to leading eigenvalues of the sub-stochastic forward
  operator in the readout sector. Its acceptance eigenvalue lambda_1 divides out.

Ceasura et al. show that syndrome qubits coupled to the logical qubits through a decoder can act as a hidden memory that
produces non-exponential, even oscillating, logical RB output under gate- and time-independent noise [LIT 5].

### 4.2 Predictions for this distance-2 code

1. **The logical error becomes first order [VA].** The decoder cannot separate errors that differ by a readout-flipping
   logical. Z0 and Z2^-1 share a syndrome and differ by Z_L. Every such detectable weight-1 error is therefore
   mis-corrected with probability about 1/2.
   - [VN] Exact terminal ML decoding of the real circuits (hybrid: intermediate rounds of unif 1 postselected, final round
     decoded) at p = 1e-3 has a per-step rate of about 2.0e-3, against 3 to 5e-5 for detection, and acceptance at m = 22 of
     0.099 against 0.080.
   - Terminal-only correction (const 1 rounds, nothing rejected) gives F(22) = 0.476, against 0.966 with detection at
     acceptance 0.205.
   - At m = 0 decoding gives 0.9789 and detection 0.9925.
2. **Per-round memory stays short for a memoryless decoder [VN toy].** In `A4_hmm_toy.py` (q_check = 12 p, q_meas = 7 p),
   with one check per step:

   | p | operator | eigenvalues | mu |
   |---|---|---|---|
   | 1e-3 | detection | acceptance 0.915356, logical 0.915343, next readout mode 0.0032 | 3.5e-3 |
   | 1e-3 | correction | stationary 1, logical 0.989038, next 0.0162 | 0.016 |
   | 5e-3 | detection | | 0.019 |
   | 5e-3 | correction | | 0.082 |

   Correction has the somewhat longer per-round memory, because every measurement fault leaves a residual syndrome for
   one round. Both are far from long memory.
3. **Unchecked intervals change character [VA].** Under detection, a silent logical needs two faults in the interval, so
   it grows like L^2 at leading order. Under correction a single fault suffices, so it grows like L. In the inverse block,
   correction therefore turns the quadratic onset into a linear one that is compatible with an exponential at leading
   order. Curvature appears only at order (r L p)^2, where pairs and mixing matter.
4. **Normalisation disappears**, and so do the covariance effects between acceptance and logical error (section 2.3).
5. **Decoders with memory [CONJ, supported by LIT 5].** A decoder that uses a syndrome history over w rounds (sliding-window
   ML, matching over time, syndrome differences across Clifford-dependent automorphisms A_g) carries an environment of
   size about 81^w. Its memory time is about w rounds. Mismatched syndrome maps (a lookup that does not commute with A_g)
   can create gate-history-dependent logical errors.

### 4.3 Does correction reduce non-exponential behaviour? [VN; interpretation VA]

Rate-drift index eta, exact on the real circuits (mean of 6 sequences), detection against the hybrid with terminal ML
decoding:

| policy | p = 1e-3 det | p = 1e-3 hybrid | p = 5e-3 det | p = 5e-3 hybrid |
|---|---|---|---|---|
| unif 1 | +0.58 | -0.04 | +0.64 | -0.03 |
| const 1 | +9.1 | +0.44 | +20.4 | +0.15 |
| const 0 | +7.7 | +0.09 | +11.0 | +0.09 |

Toy, correction applied every round against detection every round: eta = 0.09 against 1.6 at p = 1e-3, and 0.10 against
4.1 at 5e-3.

**Answer.** For this d = 2 code, correction makes the logical decay markedly more exponential relative to its own decay
rate: no normalisation, linear accumulation in unchecked intervals, short memory with a memoryless decoder. It does this by
making the logical error first order, one to two orders of magnitude faster than the postselected decay. In absolute
fidelity units a corrected curve can deviate from an exponential by more than a detected one.

[VN toy] The maximum relative deviation from the best single exponential is 2.3e-3 for correction every round and 9e-5 for
detection every round at p = 1e-3. As fractions of the total decay over the depth range these are 0.83 % and 7.8 %.

"Correction mitigates non-Markovianity" is true for the shape and memory time of the logical decay. It is not a claim
about logical performance. At d = 2, the detection protocol's apparently Markovian unif-1 decay is a slow O(p^2) process
riding on an acceptance that falls by about e^{-112 p} per step.

[CONJ] The interval law (quadratic for detection, linear for correction) is generic for distance-2 codes. For d >= 3,
correction regains a second-order logical error per interval and the corrected curve should then show the quadratic
interval law that detection shows at d = 2.

---

## 5. Test battery

### 5.1 Data model and statistics for 30 sequences x 1e6 shots

- **Unit of replication: the Clifford sequence.** Depth-m circuits of one sequence share the first m Cliffords of a single
  22-Clifford draw, so per-sequence fidelities are correlated across depths. Inference should resample whole sequences
  (cluster bootstrap, B >= 1000).
- **Per-sequence estimator.** Use F^_c(m) = (n0 - (n1+n2)/2)/n_acc. Its shot variance is
  [sum_k P_k Re(omega^k)^2 - F^2]/n_acc. Report acceptance A^_c = n_acc/1e6. Do not use |.| when F is below about 5 SE
  (section 2.4).
- **Variance decomposition [VN].** Var_total(m) = Var_seq(m) + E[Var_shot]. At p = 1e-3 the shot-noise fraction at
  m = 2 and 22 is 0.81 and 0.92 (unif 1), 0.64 and 0.50 (unif 4), 0.64 and 0.13 (const 1), 0.33 and 0.06 (const 0).
  For unif policies more shots per sequence help most; for const policies more sequences do.
- **Weights and model comparison.** Fit the sequence-mean curve with weights 1/SEM^2, SEM = sd_c/sqrt(30). Use AIC =
  chi^2 + 2k, BIC = chi^2 + k ln n and nested F-tests only as indicators: the errors are correlated across depths and
  n = 8 depths. Base decisions on bootstrap intervals of interpretable parameters. Apply a Bonferroni factor over the
  (p, policy) cells tested.
- **Exclude m = 0 from shape fits** and test it separately (S6).

### 5.2 Tests

| id | test | null (Markovian, fixed schedule) | statistic |
|---|---|---|---|
| T0 | variance decomposition | - | shot vs sequence variance per depth |
| T1 | single vs log-quadratic vs double exponential on m >= 2 | M1 = b f^m adequate | chi^2, AIC, BIC, F-test M1 vs M4 |
| T2 | curvature of M4 = b exp(-c m - d m^2) | d = 0 | cluster-bootstrap 95 % CI of d |
| T3 | depth-windowed rates f_i = (F_{i+1}/F_i)^{1/dm} | constant | drift ln f_last - ln f_first, bootstrap CI |
| T4 | Markovian surrogate F_M = B f_M^m fitted on m = 2, 4, 6 | F(22) = F_M(22) | signed Delta_M = F(22) - F_M(22), surrogate refitted per bootstrap replicate. A max-|.| statistic is positive by construction and needs a calibrated null. |
| T5 | first point | m = 0 on the extrapolated curve | z of the m = 0 residual against M1 fitted on m >= 2 |
| T6 | sawtooth (unif k) | e = 0 in ln F = ln b - c m - e (m mod k) | Delta chi^2 and bootstrap CI of e |
| T7 | check-spacing scaling | per-step rate independent of k; curvature independent of c | beta_k / beta_1 against k; gamma_c against 1/c (memory predicts linear in k, 1/c) |
| T8 | p-scaling | - | curvature and sawtooth coefficients scale as p^2 at small p (pairs); first-order terms as p |
| T9 | exact Markovian comparison | - | the exact unif-1-like surrogate exp(-a1 p - (alpha + beta_1 m) p^2) against the exact policy curve; N_q of Figueroa-Romero et al. [LIT 1] |
| T10 | acceptance shape | ln A linear | curvature of ln A (memory predicts convex, c2 < 0) |

On T9: the Figueroa-Romero Markovianisation, which traces out E after each step, is not a good surrogate here. Forgetting
sigma turns undetected single errors into first-order logical errors. The relevant surrogate resets sigma by an ideal
check after every step: this is the unif-1 forward channel, with per-step factor exp(-beta_1 p^2). Its deviations from the
other policies are, at second order, (k - 1) beta_1 m p^2 for unif k and beta_1 (m^2/c - m) p^2 for const c, which is the
**memory penalty** [VN coefficients in 3.3].

### 5.3 Results on the baseline data [VN] (`A4_fit_battery.py`, 1000 bootstrap replicates)

At p = 1e-3:

| policy | chi^2 M1 / M4 / M3 (n = 8) | d (95 % CI) | drift (95 % CI) | Delta_M (95 % CI) | T5 z0 | T6 e (CI); chi^2 drop |
|---|---|---|---|---|---|---|
| unif 1 | 11.3 / 11.3 / 7.1 | -0.9e-7 [-8.7e-7, +6.5e-7] | -6.7e-5 [-2.0e-4, +5.9e-5] | -1.5e-4 [-5.1e-4, +1.8e-4] | +2.7 | - |
| unif 4 | 75.1 / 73.2 / 56.9 | -6.0e-7 [-1.6e-6, +3.3e-7] | -2.9e-4 [-4.0e-4, -1.7e-4] | -2.9e-4 [-7.1e-4, +9.5e-5] | +2.8 | 1.32e-4 [0.98, 1.6]e-4; 75.1 -> 1.7 |
| unif 3 | 81.6 / 79.2 / 47.7 | | | | +4.5 | 1.34e-4 [1.1, 1.6]e-4; 81.6 -> 11.7 |
| unif 6 | 112.2 / 112.1 / 109.5 | | | | +3.9 | 0.94e-4 [0.75, 1.1]e-4; 112.2 -> 20.0 |
| const 1 | 5461 / 68 / 128 | 5.97e-5 [5.8, 6.2]e-5 | -2.6e-3 [-2.7, -2.4]e-3 | -1.9e-2 [-2.0, -1.9]e-2 | - | - |
| const 2 | 1436 / 5.2 / 8.0 | 2.04e-5 [1.9, 2.2]e-5 | -8.4e-4 [-9.5, -7.3]e-4 | -6.6e-3 [-7.0, -6.1]e-3 | - | - |
| const 0 | 632 / 1.5 / 1.8 | 2.37e-6 [2.2, 2.5]e-6 | -9.1e-5 | -8.1e-4 [-8.5, -7.6]e-4 | n/a | - |

Notes:

- The const 0 curvature 2.37e-6 matches the exact 2.52e-6 (log-quadratic fit) and gamma_0 p^2 = 2.24e-6.
- The const 1 curvature (5.97e-5) is close to the exact 6.94e-5. The latter is an unweighted fit on the mean of 6
  sequences, which also contains the higher-order terms discussed in 3.3.
- The sawtooth e grows roughly as p^2 (by factors of 3 to 4 per doubling of p, plus higher orders): 1.9 to 3.2e-5
  (5e-4), 0.9 to 1.3e-4 (1e-3), 3.7 to 4.7e-4 (2e-3), 3.2 to 4.2e-3 (5e-3), for k = 3, 4, 6.
- The T5 boundary effect appears at 5e-4, 1e-3 and 2e-3 for unif policies: z0 = +1.2 to +14, the one exception being
  unif 6 at 2e-3 (-7.3, where the sawtooth spoils the extrapolation).

**Cross-check with N1 (`N1-baseline-nonmarkov.md`, CSV-level fits in this study).** N1 reports the same signatures from
the result CSVs alone:

- unif 1 is consistent with one exponential apart from the m = 0 point;
- const 1 has a quadratic infidelity of 30 to 40 p^2 at small p;
- const 0 follows a pure m^2 law with b2 = 2.2 to 2.6 p^2;
- inverse-block pairs are below sensitivity.

These agree with the exact values here (gamma_0 = 2.24; gamma_1 about 24 to 27). N1 labels the split of the quadratic term
between forward and inverse halves a conjecture; `A4_attribution.py` verifies it numerically (C12): for const 1,
forward-forward pairs give 21.6 of 24.4 in the m^2 coefficient, forward-inverse cross pairs about 3, and inverse-inverse
pairs about 0.05.

**Power for unif 1 [VN].** The exact curvature at 1e-3 is d = 1.9e-7, and the bootstrap SE is about 3.9e-7. A 3-sigma
detection needs about 40 times more data. Raising p does not help, because acceptance at m = 22 collapses (0.6 % at
p = 2e-3; the CI widens to +-2.2e-6). With the present design the inverse-block memory of unif 1 is invisible. Its absence
from the fits is not evidence of Markovianity.

### 5.4 Design recommendations [VA unless marked]

1. **Remove the shared memory interval.** Compile the inverse of the whole forward sequence into one logical Clifford, as
   in standard RB. Alternatively apply the policy's checks inside the inverse block. With either change, unif 1 becomes
   Markovian at O(p^2) apart from one-round masking. Comparing the mirrored and compiled versions at equal m then measures
   the inverse-block memory directly.
2. **Add odd depths and several depths below 6.** This resolves the curvature d, the sawtooth for every k, and the
   switch-on of const-c checks.
3. **Probe memory length the way Figueroa-Romero et al. suggest [LIT 1]:** fix some Cliffords to the identity and vary the
   check spacing at fixed m. With memory, the decay depends on spacing; without memory it does not (T7).
4. **Analyse acceptance as well as fidelity (T10).** It is the sensitive memory witness. Its curvature c2 is large and
   negative for every unchecked interval.
5. **Correction experiments.** Report the hidden-state spectrum (mu) of the decoder model and the rate-drift index eta
   next to the fit. Decoders with temporal windows should be tested with T3 and T6 at window-size resolution [CONJ].

---

## 6. Leakage, hardware and other unmodelled effects [CONJ unless marked]

- Leakage, for example of a transmon-based qutrit to |3>, is a non-Pauli environment state with a lifetime of
  1/(leakage + seepage rate), possibly many rounds. In RB it adds an exponential set by leakage and seepage rates. Wood and
  Gambetta fit the computational-subspace population as A + B lambda_1^m with lambda_1 = 1 - L1 - L2, and a fidelity-type
  curve as A0 + B0 lambda_1^m + C0 lambda_2^m [LIT 6]. They also note that coherent leakage that is not depolarised by the
  twirl can produce oscillations [LIT 6].
- For LRB-D the conjecture is: a leaked data qutrit randomises the checks it touches. Leakage events are then mostly
  rejected under unif 1, which adds to the acceptance decay. Under sparse checks they spread errors through CNOTs, which
  gives a slow extra exponential with weight set by seepage.
- Under correction, leakage produces repeated mis-corrections, error bursts with memory equal to the leakage lifetime.
  This is the most serious non-Markovian risk on hardware unless leakage reduction is run every round.
- Coherent errors would break F1, because acceptance could then depend on the logical state through coherent
  interference with the stabiliser measurement. They would break F2 as well (no classical frame), and the Clifford twirl
  only partly restores the depolarising form.
- [LIT 9] Temporally correlated physical noise (not present here) broadens and skews the distribution of per-sequence
  outcomes (gamma-distributed with correlation-dependent parameters), so per-sequence spread is itself a diagnostic.

---

## 7. Claims

Each item gives the claim, its label and the evidence.

- **A4-nonmarkov-theory-C1.** Physical SI1000 dynamics is Markovian. Logical non-Markovianity in LRB-D arises only from
  coarse-graining away the syndrome register (plus ancilla, leakage and decoder states).
  [VA] Section 1.1; the noise model is a product of fixed channels (brief).
- **A4-nonmarkov-theory-C2.** Acceptance is independent of the encoded logical state, so every accepted step map
  satisfies Phi^dag(I_L) proportional to I_L for each hidden state.
  [VA] Section 1.4 F1: stabiliser outcomes are commutation phases of Pauli frames.
- **A4-nonmarkov-theory-C3.** For a fixed CP trace-non-increasing E, tr[O E^m rho]/tr[E^m rho] is a ratio of two
  exponential polynomials. At large m it tends to a constant if the Perron term enters O; otherwise it decays as
  (lambda_O/lambda_1)^m.
  [VA] Section 2.1. [VN] Toy (b): r(400) = 0.064302 equals the Perron prediction 0.064302.
- **A4-nonmarkov-theory-C4.** It is exactly (alpha/s)^m if E^dag(I) = sI and E^dag(O) = alpha O, for example for twirled
  E = alpha Id + beta Dep with s = alpha + beta.
  [VA] Section 2.2. [VN] Max error 1.3e-15.
- **A4-nonmarkov-theory-C5.** In random-sequence RB the ratio of sequence averages is exactly (alpha/(alpha+beta))^m for
  any lossy E. The mean of per-sequence ratios is exponential if acceptance is state independent and deviates otherwise.
  [VA] + [VN] Toy (d): chi^2/dof 0.31 against 28.9; exact ratio of averages to 1e-14.
- **A4-nonmarkov-theory-C6.** With gate-dependent but state-independent acceptance, per-sequence normalisation reduces
  LRB to RB of CPTP maps. Wallman's bound then gives A p^m + eps_m with eps_m exponentially small.
  [VA] given [LIT 3] Theorem 4.
- **A4-nonmarkov-theory-C7.** The exact character expansion and 81 x 81 transfer-matrix factorisation hold for every LRB-D
  circuit and policy, and reproduce the baseline shot counts.
  [VA] Section 3.1. [VN] z mean -0.1, sd 1.0 (534 cells at 1e-3); banding asserted for all circuits.
- **A4-nonmarkov-theory-C8.** The first-order accepted infidelity is m- and policy-independent: -ln F = 7.425 p + O(p^2),
  equal to (3/2) x 4.95 p silent final-round faults. const 0 has none.
  [VA] Hand count, section 3.3. [VN] a1 = 7.425 at every depth and policy.
- **A4-nonmarkov-theory-C9.** Acceptance: -ln A = 63.2 p + 71.1 p m + 41.0 p n_used(m) + O(p^2), with log-convex
  curvature at second order.
  [VN] Exact c1 slopes 112.07, 91.57, 84.74, 81.32, 77.90, 71.07 = 71.07 + 41/k; c2 < 0. [VA] r_M breakdown.
- **A4-nonmarkov-theory-C10.** The second-order accepted infidelity comes from pairs of faults inside unchecked intervals
  (persisting detectable error plus cancelling error). An interval of L forward steps contributes about 22 L^2 p^2.
  Hence the unif-k per-step rate is about k beta_1 p^2 (beta_1 about 23) and the const-c curvature is gamma_1/c.
  [VA] Mechanism (F3). [VN] beta = 22.7, 45.8, 68.3 for k = 1, 2, 3; gamma = 26.8, 13.6, 8.8, 5.3 for c = 1..4.
- **A4-nonmarkov-theory-C11.** const 0 is purely quadratic at second order: -ln F = 2.24 m^2 p^2 and a1 = 0.
  [VN] Exact a2 fit; data (1-F)/m^2 = 2.0 to 2.3e-6 at 1e-3; battery d = 2.37e-6 [2.2, 2.5]e-6.
- **A4-nonmarkov-theory-C12.** The m-step unchecked inverse block is a memory interval shared by all policies. In unif 1 it
  gives a small m^2 term (gamma_I about 0.05 to 0.17) and terms linear in m (about 4 to 6 m). Forward-forward pairs give
  about 18 m.
  [VN] `A4_attribution.py`.
- **A4-nonmarkov-theory-C13.** unif k (k not dividing the depths) shows a sawtooth in m mod k with positive coefficient,
  scaling as p^2.
  [VN] T6: chi^2 75 -> 1.7 (unif 4, 1e-3); e = 1.3e-4 [0.98, 1.6]e-4; e grows from about 2e-5 to 4e-3 between 5e-4 and
  5e-3. Exact a2 jump about 135 p^2.
- **A4-nonmarkov-theory-C14.** The accepted fidelity can increase with m (non-monotone) when the m-dependent schedule
  switches on checks. A rising ASF is therefore not a non-Markovianity witness in postselected RB with m-dependent
  schedules (compare [LIT 2]).
  [VN] Exact const 4 windowed rate 1.000004 (1e-3); data paired z = 2.9 (1e-3) and 3.5 (5e-3).
- **A4-nonmarkov-theory-C15.** The m = 0 point lies above the single-exponential extrapolation from m >= 2: a boundary
  effect of the second-order offset.
  [VN] T5 z0 = +2.4 to +6.6 for unif 1 at 5e-4 to 2e-3; exact a2(0) = 111 against a2(2) = 201 with slope 23.
- **A4-nonmarkov-theory-C16.** Terminal-only checks saturate: acceptance tends to 1/81 and F tends to 0. ln F crosses over
  from quadratic to linear in m.
  [VA] + [VN] Exact and data A = 0.0123 for p >= 2e-2.
- **A4-nonmarkov-theory-C17.** The pipeline's modulus estimator is Rice-biased at low SNR (about +1 sigma when F < 5 SE).
  The real-part estimator is unbiased under the symmetric Weyl noise.
  [VN] z = +0.65/+0.70 (|.|) against +0.11/+0.04 (Re) for const 1 at 1e-2 and 2e-2.
- **A4-nonmarkov-theory-C18.** With the present design the unif-1 curvature is undetectable (exact d about 0.5 of the
  bootstrap SE); about 40 times more data would be needed.
  [VN] Exact d = 1.9e-7, CI half-width 7.6e-7.
- **A4-nonmarkov-theory-C19.** Correction turns the logical decay into a linear functional (no normalisation). For
  d = 2 it makes the logical error first order and per-step error accumulation in unchecked intervals linear instead of
  quadratic.
  [VA] Section 4.1 and 4.2. [VN] Hybrid per-step rate 2.0e-3 against 3 to 5e-5 at 1e-3.
- **A4-nonmarkov-theory-C20.** Correction strongly reduces the relative non-exponentiality (rate-drift index eta) compared
  with detection on the same circuits, but increases the decay rate by one to two orders of magnitude.
  [VN] eta: unif 1 0.58 -> -0.04, const 1 9.1 -> 0.44, const 0 7.7 -> 0.09 (1e-3); toy 1.6 -> 0.09.
- **A4-nonmarkov-theory-C21.** A memoryless decoder run every round has short hidden-state memory (mu about 0.016 at
  1e-3, 0.08 at 5e-3), somewhat longer than detection's (0.0035, 0.019) but far from long memory.
  [VN] `A4_hmm_toy.py` spectra.
- **A4-nonmarkov-theory-C22.** Decoders with temporal windows or Clifford-mismatched syndrome maps carry an environment of
  size about 81^w and memory of about w rounds; they can produce non-exponential and even oscillating LRB output.
  [CONJ] for this code; mechanism [LIT 5].
- **A4-nonmarkov-theory-C23.** For d >= 3, correction regains second-order logical errors per interval, and the
  quadratic interval law should then appear in corrected curves.
  [CONJ]
- **A4-nonmarkov-theory-C24.** Leakage, unmodelled here, would add a long-lived memory: an extra exponential in acceptance
  and fidelity under detection, error bursts under correction.
  [CONJ] with [LIT 6] for the RB-with-leakage decay form.
- **A4-nonmarkov-theory-C25.** Replacing the m mirrored inverse Cliffords by one compiled recovery Clifford, or checking
  inside the inverse block, removes the only memory interval common to all policies. Comparing it with the mirrored
  version measures that memory.
  [VA] From C10 and C12.
- **A4-nonmarkov-theory-C26.** The single-qutrit Clifford group (216 elements mod phase) is a unitary 2-design.
  [VN] Frame potential 2.000000000000.
- **A4-nonmarkov-theory-C27.** In the system-environment model with CP trace-non-increasing noise maps, the RB average
  sequence fidelity contains matrix powers of environment maps, ($ - Theta)^m and Theta^m. It is generally not
  exponential unless the environment is superfluous. Its deviation from a Markovianised counterpart is measured by
  N_q = ||F_m - F_m^(M)||_q.
  [LIT 1] Figueroa-Romero et al. 2021, Eqs. 7 to 16.
- **A4-nonmarkov-theory-C28.** Syndrome qubits coupled to the logical qubit through a decoder can act as hidden memory
  and make logical RB non-exponential, even oscillating, under gate- and time-independent noise.
  [LIT 5] Ceasura et al. 2022, abstract.

---

## 8. References (all fetched in this task)

1. P. Figueroa-Romero, K. Modi, T. M. Stace, M.-H. Hsieh, "Randomized benchmarking for non-Markovian noise", PRX Quantum 2,
   040351 (2021); arXiv:2107.05403; DOI 10.1103/PRXQuantum.2.040351. Fetched: https://arxiv.org/abs/2107.05403 and
   https://ar5iv.labs.arxiv.org/html/2107.05403 (Eqs. 2, 4, 7 to 16 and the identity-interleaving memory test used above).
2. P. Figueroa-Romero, K. Modi, M.-H. Hsieh, "Towards a general framework of Randomized Benchmarking incorporating
   non-Markovian Noise", Quantum 6, 868 (2022); arXiv:2202.11338; DOI 10.22331/q-2022-12-01-868. Fetched:
   https://arxiv.org/abs/2202.11338 and https://ar5iv.labs.arxiv.org/html/2202.11338. This is the source for the
   quality-map ASF, for an increasing ASF as their strongest non-Markovian signature, and for the stability under
   gate-dependent perturbations.
3. J. J. Wallman, "Randomized benchmarking with gate-dependent noise", Quantum 2, 47 (2018); arXiv:1703.09835; DOI
   10.22331/q-2018-01-29-47. Fetched: https://arxiv.org/abs/1703.09835 and https://ar5iv.labs.arxiv.org/html/1703.09835
   (Theorem 4: A p^m + B t^m + eps_m, |eps_m| <= delta_1 delta_2^m; no trace preservation needed).
4. F. A. Pollock, C. Rodríguez-Rosario, T. Frauenheim, M. Paternostro, K. Modi, "Operational Markov condition for quantum
   processes", Phys. Rev. Lett. 120, 040405 (2018); arXiv:1801.09811; DOI 10.1103/PhysRevLett.120.040405. Fetched:
   https://arxiv.org/abs/1801.09811 (abstract only; the factorisation statement is used as given in ref. 1, Eq. 4).
5. A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan, "Non-Exponential Behaviour in Logical Randomized Benchmarking",
   arXiv:2212.05488 (2022). Fetched: https://arxiv.org/abs/2212.05488 (abstract).
6. C. J. Wood, J. M. Gambetta, "Quantification and characterization of leakage errors", Phys. Rev. A 97, 032306 (2018);
   arXiv:1704.03081; DOI 10.1103/PhysRevA.97.032306. Fetched: https://arxiv.org/abs/1704.03081 and
   https://ar5iv.labs.arxiv.org/html/1704.03081.
7. J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized Benchmarking", arXiv:1702.03688 (2017). Fetched:
   https://arxiv.org/abs/1702.03688 (the LRB protocol that LRB-D modifies).
8. J. Helsen, I. Roth, E. Onorati, A. H. Werner, J. Eisert, "A general framework for randomized benchmarking", PRX Quantum
   3, 020357 (2022); arXiv:2010.07974. Fetched: https://arxiv.org/abs/2010.07974 (RB output as linear combinations of
   matrix exponential decays).
9. H. Ball, T. M. Stace, S. T. Flammia, M. J. Biercuk, "Effect of noise correlations on randomized benchmarking", Phys.
   Rev. A 93, 022303 (2016); arXiv:1504.05307. Fetched: https://arxiv.org/abs/1504.05307.
10. A. Kabir, S. Nguyen, S. Ghosh, et al., "Sdim: A Qudit Stabilizer Simulator", arXiv:2511.12777 (2025). Fetched:
    https://arxiv.org/abs/2511.12777 and https://arxiv.org/html/2511.12777v2. Section VII introduces LRB-D on
    [[5,1,2]]_3 and notes that terminal detection can give non-Markovian decay structure.

Not cited because they could not be confirmed from fetched pages: the AIC and BIC original papers (landing pages
empty), and a claimed 12.5 % acceptance saturation in arXiv:2602.22211 (absent from the abstract). AIC and BIC are used
above only as standard formulas.

---

## 9. Reproduction

```
module load python/3.11
export OMP_NUM_THREADS=1 A4_SCRATCH=<a scratch dir>
S=studies/correction-decoders/qutrit/notes/scripts
python $S/A4_lossy_markov.py                       # section 2   (~20 s)
python $S/A4_exact_lrbd.py --seqs 0 1 2 3 4 5      # section 3   (~8 min, caches DEMs in $A4_SCRATCH)
python $S/A4_analyze_exact.py $A4_SCRATCH/A4_exact.npz
python $S/A4_attribution.py                        # section 3.3 (~1 min, needs the cache)
python $S/A4_hmm_toy.py                            # section 4   (~2 min)
python $S/A4_fit_battery.py                        # section 5   (~1.5 min)
```

All scripts only read the baseline run folder. Set `OMP_NUM_THREADS=1`: without it, sdim and numpy spawn about 128 threads
on the login node.
