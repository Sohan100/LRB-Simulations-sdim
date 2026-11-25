# L1 literature notes: logical RB, RB of encoded gates, qudit and qutrit RB, leakage RB, mirror RB

Agent: L1 (literature). Date: 2026-10-28. Scope: the topic list in the task, read against
`00-brief.md`.

## How to read these notes

Every claim carries one label:

- **(from literature [Key])**: I fetched the arXiv abstract page (and, where given, the
  publisher or DOI page) during this task, and, unless I say "abstract only", I read the
  passage in the arXiv LaTeX source (downloaded with `arxiv.org/e-print/<id>` and searched
  with grep). Formulas are transcribed from those sources. Quotations are short and in quotes.
- **(verified analytically)**: a derivation I did here; it is written out or sketched.
- **(verified numerically)**: a check I ran; scripts are in `lit-L1-lrb-checks/`, outputs
  are in the appendix.
- **(conjecture)**: my expectation, not checked.

Notation follows the brief: m = depth = number of forward logical Cliffords; a depth-m
circuit has 2m noisy logical Clifford blocks and m+1 check rounds; the per-circuit metric is
`|sum_k w^k P(k)|` over accepted shots (w = exp(2 pi i/3)), averaged over 30 sequences, and
fitted to `b f^m` with zero offset; F = (1+2f)/3.

---

## 0. Findings that matter for this study

1. **The original LRB protocol is LRB with correction (or with no recovery), not with
   postselection.** Combes et al. trace out the syndrome register after every round, so the
   logical channel is trace preserving, and the decay is Magesan's zeroth-order model
   `A_L p_L^m + B_L` with `p_L = (d_L F - 1)/(d_L - 1)`. A full-text search of the source finds
   no postselection, rejection or discard anywhere in the paper (from literature
   [Combes2017]). For stabilizer codes they note that the recovery can be applied entirely in
   post-processing by propagating it as a Pauli frame to the final measurement. That is exactly
   what a DEM decoder in this repo would do (from literature [Combes2017]; the match with the
   repo's DEM backend is verified analytically from the brief).
2. **No complete theory of RB with postselection exists in the sources I read.** Harper and
   Flammia: "There is at present no complete theory for fitting an RB decay curve with
   post-selection." They write the postselected survival as `Pr(S|m,c) = Pr(S|m)/Pr(c|m)` and
   approximate the rejected fraction as constant in m (from literature [HarperFlammia2019]).
   Wallman proves that for non-trace-preserving (lossy, for example encoded) noise the
   unnormalized survival is `A p^m + B t^m + eps_m`, two exponentials (from literature
   [Wallman2018]). If the accepted branch of each step were a fixed completely positive map, a
   twirl argument makes the conditional Weyl expectation `(a/t)^m`, a single exponential with
   zero offset (verified analytically, section 9.2). Syndrome memory breaks that assumption.
   It is present in this repo for unchecked rounds, for the unchecked inverse block, and for
   faults that land after the data interacts with the ancillas.
3. **LRB on hardware.**
   - With decoding and no postselection: Google color code [Lacroix2025], Quantinuum Steane
     code [Mayer2024], Quantinuum C4-Helix [Berthusen2026].
   - With postselection: IBM [[4,2,2]], terminal parity check only [HarperFlammia2019]; a
     bosonic erasure qubit with a check after every Clifford, where the acceptance decay and
     the conditional decay are fitted separately [Mai2026].
   - Repeated d = 2 detection with a postselected fraction `P(n) = A(1-gamma)^n`, gamma about
     45% per cycle [Marques2022].
   - I found no neutral-atom logical RB and no ETH logical RB. The Harvard/QuEra work uses
     sliding-scale, decoder-confidence detection and shows that the logical error per gate
     depends on how many gates sit between QEC rounds [Bluvstein2023, Bluvstein2025]. (All from
     literature.)
4. **d = 3 is safe for Clifford twirling.** The single-qutrit Clifford group has 216 elements
   mod phase and is a unitary 2-design (from literature [GAE2007, Graydon2021]). My check gives
   order 216 and frame potentials 1.000000, 2.000000, 7.000000 for t = 1, 2, 3. It is therefore
   a 2-design but not a 3-design, since the Haar value at t = 3 is 6 (verified numerically).
   For composite d the Clifford group is not a 2-design [Graydon2021]. Graydon et al. point
   out that [Jafarzadeh2020] assumed it was (from literature).
5. **The repo's metric is Morvan's qutrit RB observable.** `sum_k w^k P(k)` is the qutrit
   "<Z>" of [Morvan2021]. They fit `A p^m + b` to its real part and check that the imaginary
   part stays near 0; the repo takes the modulus per circuit (from literature [Morvan2021]).
   Under the brief's noise model every fault distribution is symmetric under negation and
   acceptance is a linear condition, so the accepted logical-shift distribution satisfies
   P(1) = P(2) and `sum_k w^k P(k) = (3P(0)-1)/2` is real (verified analytically). Two
   baseline circuits confirm this (|Im| < 5e-5 with 4e4 accepted shots; verified numerically).
   The modulus therefore changes results only if P(0) < 1/3 or the accepted-shot count is
   small, where `|.|` has a positive noise floor (verified analytically).
6. **The repo's inversion is a mirror circuit without Pauli randomization.** Inverting
   Clifford by Clifford in reverse order is "simple circuit mirroring" in the sense of
   [ProctorCap2022, ProctorMRB2022]. Two consequences:
   - Under gate-independent Weyl noise, the per-depth decay factor is the polarization of the
     composed (forward error of C_i) and (inverse error of C_{i+1}^-1). Both sit in the same
     random frame. This equals the product of the two polarizations for depolarizing noise but
     differs at second order for sparse noise. This is verified analytically and verified
     numerically by exact enumeration; it is the qutrit analog of Proctor's eta term.
   - The fitted f is therefore per pair (forward Clifford + check round + inverse Clifford).
     F = (1+2f)/3 (code: `lrb_plotting.py`, line 894) is a per-pair fidelity, and the
     per-Clifford value is roughly (1+2 f^(1/2))/3 (verified analytically). On hardware,
     coherent errors in C and C^-1 can cancel unless random Pauli layers are inserted (from
     literature [ProctorCap2022]).
7. **Leakage-RB view of the check policies.** Treat the code space as the computational
   subspace (d1 = 3) and the 80 nontrivial syndrome sectors as the leakage subspace
   (d2 = 240).
   - `unif 1` is like erasure-heralded RB. Rejected population never returns (seepage
     L2 = 0), so acceptance should be close to a single exponential (conjecture; compare
     [Mai2026, Marques2022, WoodGambetta2018]).
   - `const 1` is like leakage RB with seepage. Syndrome-cancelling error pairs return
     population to the code space, sometimes with a logical error. If the syndrome random walk
     mixes, acceptance tends to a floor of 1/81, not to 0 (verified analytically). The decay is
     a ratio of sums of exponentials (conjecture).
   - Leakage that can coherently return is "fundamentally non-Markovian" (from literature
     [WBE2016]). The single-exponential leakage model needs the twirl to depolarize the leakage
     space (from literature [WoodGambetta2018]).
   - Ceasura et al. built gate- and time-independent noise models in which the syndrome
     register, coupled to the logical qubit through the decoder, acts as a hidden memory and
     makes LRB output oscillate (from literature [Ceasura2022]).

---

## 1. Core theory: Combes, Granade, Ferrie, Flammia 2017, "Logical randomized benchmarking"

**Citation.** J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized
Benchmarking", arXiv:1702.03688 (v1, 13 Feb 2017; no journal reference listed). Fetched:
https://arxiv.org/abs/1702.03688 and e-print source `logical_rb.tex`.

**Protocol** (from literature [Combes2017]):
- Prepare and encode. Apply m random logical Cliffords, "following each with a round of error
  correction (with or without recovery) and a refresh of the ancilla register". Then apply
  "the Clifford operation `U^{-1} = prod U^dagger`", a single inverting Clifford. Decode and
  measure.
- The circuit in their Fig. 1(a) is `E, U1, R, U2, R, ..., Um, R, U^-1, R, D, measure`, so a
  recovery round also follows the inverse.
- Each length uses L random sequences repeated T times. With T = 1 the data are binomial.

**Decay model and what it measures** (from literature [Combes2017]):
- `F_L(m) = E[q(m)] = A_L p_L^m + B_L`, with `p_L = (dim H_L F(Lambda_L, id) - 1)/(dim H_L - 1)`.
  "The dimension dependence of the decay parameter only involves the dimension of the logical
  space."
- Lambda_L is the logical channel
  `Tr_synd[U^{-1} D R~ U~ E (rho_L (x) |0><0|_synd)]`, with the syndrome traced out.
- The proof reduces LRB to Magesan's zeroth-order RB model: noisy encoding and decoding
  become logical SPAM, absorbed in A_L and B_L.
- Running with R = id gives F_L(Lambda) (the averaged no-error probability); running with
  recovery gives F_L(R Lambda). From the two they estimate averaged code properties:
  Pr(Un) = 1 - F_L(R Lambda), Pr(Co) = F_L(R Lambda) - F_L(Lambda), Pr(No) = F_L(Lambda).

**Assumptions under which the decay is a single exponential** (from literature [Combes2017]):
- "we require the noise to be time-independent, Markovian, and gate-independent (all at the
  level of logical Clifford gates)".
- "Unique to the situation of logical qubits, we must also assume that syndrome measurement
  errors are uncorrelated between rounds of error correction."
- In the appendix: "It is vital to our protocol that there is no correlations between ancilla
  in successive rounds of error correction", which is why the ancillas are refreshed (measured
  and re-prepared). They also assume the SPAM channels factor into logical and syndrome parts.
- They cite robustness to small violations of gate independence (Wallman 2017, later
  [Wallman2018]) and Proctor et al.'s counterexamples (later [Proctor2017]).

**Rejected shots.** None. The paper contains no postselection; I verified this by full-text
search for postselect, reject, discard, herald and detect (from literature [Combes2017]).
"Without recovery" means keep every shot and apply no correction; it does not mean discard
flagged shots.

**Recovery in post-processing** (from literature [Combes2017]): for stabilizer codes with
Pauli preparation and measurement, "we need not actually perform the recovery operation, as it
is always a Pauli operation that we can commute forward". The recovery R(s) for each round's
syndrome is propagated through the later Cliffords by Gottesman-Knill simulation and flips (or
does not flip) the survival outcome. "No particular recovery map needs to be chosen at
experiment time." So several decoders can be compared on the same data.

**Other statements** (from literature [Combes2017]):
- In a [[3,1,1]] bit-flip example, physical RB can overestimate logical fidelity under
  correlated noise (to leading order the difference is about (4/3)q) or underestimate it.
- LRB can be combined with leakage RB "to test leakage of the encoded information to outside
  the code space", and with unitarity, crosstalk, and dihedral or CTPT RB.
- They point out that [Heeres2017] performed logical RB without recovery or syndrome reset,
  with p_L ≈ 0.982, "thus we cannot directly conclude Pr(No) ≈ 0.982".

**Bearing on this study.**
- LRB-D differs from Combes's LRB in five ways (verified analytically, comparing the brief
  with the protocol above):
  - it postselects instead of correcting, so the logical map is trace decreasing;
  - it inverts with m inverse Cliffords instead of one recovery Clifford;
  - it has no check rounds between the inverse Cliffords;
  - its metric is `|sum w^k P(k)|` rather than a survival probability;
  - it uses check-round subsets (unif k, const c).
- An "LRB-C" variant in this repo is literally Combes's LRB with recovery done in
  post-processing. The decoder maps a shot's detector history to a logical shift, which is
  subtracted from the terminal outcome, and no shot is rejected.
- With d_L = 3, Combes's p_L formula gives F = (1+2p)/3, matching the repo's conversion
  (verified analytically).
- Combes's assumption that the syndrome register is traced out and refreshed each round is the
  assumption that fails when a d = 2 code is used with postselection and partial check
  schedules (conjecture; see section 9.2).

---

## 2. Experiments that benchmark encoded gates with RB-like decays

### 2.1 Harper and Flammia 2019, "Fault-tolerant logical gates in the IBM Quantum Experience" (key paper for postselection)

**Citation.** R. Harper, S. T. Flammia, Phys. Rev. Lett. 122, 080504 (2019),
DOI 10.1103/PhysRevLett.122.080504, arXiv:1806.02359 (v1 6 Jun 2018; v4 30 Jan 2019).
Fetched: https://arxiv.org/abs/1806.02359 and e-print `IBMCArXiv.tex`.

**Setup** (from literature [HarperFlammia2019]):
- The [[4,2,2]] code (stabilizers XXXX and ZZZZ) on IBM QX5.
- RB in the logical space ("Using RB in the logical code space [Combes2017]").
- They use real RB over the "Realizable Group" R(2), with 576 elements and an orthogonal
  2-design, because the code has no fault-tolerant phase gate. The full two-qubit Clifford
  group has 11,520 elements; an R(2) element costs about 4 elementary gates against about 7.
- The decay model is `q(m,E,rho) = A + b^m B + c^m C`, with `F = (9b + 6c + 5)/20` for two
  qubits. A is fixed at 0.25 by compiling in random Pauli flips, and B or C is zeroed by the
  choice of input state, so each run fits a single exponential (`B b^m + 0.25` or `C c^m`).

**Handling of rejected shots** (from literature [HarperFlammia2019]):
- Detection happens only at the end. "To measure the logical qubits in the computational
  basis, one simply measures all four qubits. Odd parity heralds an error and the run should
  be discarded." The device had no mid-circuit measurement or reset.
- In the supplement: "the probability of survival `Pr(S)` ... conditioned on number of gates
  (m) and return to the code space (c) ... `Pr(S|m,c) = Pr(S|m)/Pr(c|m)`".
- "The numerator ... can be modeled in the logical code space using the RB formula, however
  the denominator requires an examination of the physical 4 qubit system and the effective
  twirl of the physical realization of the logical gates in the larger Hilbert space. Failure
  to take into account `Pr(c|m)` might lead to inaccurate fits."
- The rejected fraction `1 - Pr(c|m)` was "relatively flat", from 0.3 (0.4 for phased runs)
  rising to its asymptote 0.5. "We have therefore approximately modeled it as constant and fit
  a standard RB curve to `Pr(S|m,c)`."
- Error bars come from a nonparametric bootstrap over 36 sequences per length, with binomial
  resampling using "the number of error-free runs for that sequence".

**What the decay is claimed to measure, with and without postselection** (from literature
[HarperFlammia2019]):
- Two-qubit average infidelity is 0.60(3)% for the code with postselection against 5.8(2)% for
  two physical qubits.
- Fitting all runs without postselection gave an infidelity "less than 5.8%, but still
  significantly higher". That separates the benefit of virtual CNOTs from the benefit of
  detection.
- For the run without postselection they warn: "a fit to the survival probability in the 4
  qubit Hilbert space is not justified as the physical implementation of the logical gates in
  the 4 qubits gates do not form an orthogonal 2-design in the 4 qubit space ... we have a
  leaky subspace."
- "There is at present no complete theory for fitting an RB decay curve with post-selection."
  They thank T. Yoder "for highlighting the impact of post-selection on the RB decay curve."
- Their simulation shows that correlated ZZ noise can make encoded gates worse than bare ones.

**Bearing on this study.**
- This is the closest precedent for the repo's `const 1` policy, where only the terminal check
  is used (from literature).
- Their constant-denominator approximation does not hold here. In the brief's baseline at
  p = 1e-3, `const 1` rejection rises to 79% at m = 22, so `Pr(c|m)` depends strongly on m.
  The conditional curve is then not a plain RB curve (verified analytically: a ratio of an RB
  numerator to an m-dependent denominator).
- Their per-sequence bootstrap with binomial resampling of accepted shots transfers directly
  to this repo's error bars (conjecture: suitable).

### 2.2 Heeres et al. 2017, "Implementing a universal gate set on a logical qubit encoded in an oscillator"

**Citation.** R. W. Heeres, P. Reinhold, N. Ofek, L. Frunzio, L. Jiang, M. H. Devoret,
R. J. Schoelkopf, Nature Communications (DOI 10.1038/s41467-017-00045-1), arXiv:1608.02430
(8 Aug 2016). Fetched: https://arxiv.org/abs/1608.02430 and e-print `main.tex`.

**Facts** (from literature [Heeres2017]):
- RB and interleaved RB of numerically optimized gates on a four-component cat-code qubit in
  a cavity: encode, apply n Cliffords and the inverting Clifford, decode, read out the transmon.
- No error correction and no postselection. "a new random sequence for every measurement".
- Fit `p_correct = 0.5 + A e^{-n/tau}`, error per gate `r = (1 - e^{-1/tau(RB)})/2`, and
  interleaved `r(X) = (1 - e^{1/tau(X) - 1/tau(RB)})/2`. Inferred average gate fidelity 0.991.

**Bearing.** This is the R = id case of [Combes2017], since there is no syndrome extraction.
It shows LRB without QEC is just RB on the encoded subspace (from literature).

### 2.3 Reinhold et al. 2020, "Error-corrected gates on an encoded qubit"

**Citation.** P. Reinhold, S. Rosenblum, W.-L. Ma, L. Frunzio, L. Jiang, R. J. Schoelkopf,
Nature Physics (DOI 10.1038/s41567-020-0931-8), arXiv:1907.12327 (29 Jul 2019). Fetched:
https://arxiv.org/abs/1907.12327 and e-print `main.tex`, `supplement.tex`.

**Facts** (from literature [Reinhold2020]):
- A path-independent SNAP gate on a cavity logical qubit, using a three-level ancilla. Ancilla
  dephasing and relaxation are made detectable. The error-corrected gate S_C "includes the
  error-transparency drive, ancilla measurement, reset, and conditional repetition", with a
  software phase update after a relaxation outcome. So detected ancilla errors are corrected
  or retried, not discarded.
- Benchmarked by interleaved RB over logical Cliffords. Fit `A e^{-gamma n} + 1/2`; the gate
  error is `gamma_IRB - gamma_RB`: 2.4% ± 0.1% corrected against 4.6% ± 0.1% uncorrected,
  with gamma_RB = 2.5% ± 0.1%.
- Under injected noise, the suppression is about sixfold for relaxation and fourfold for
  dephasing (abstract).

**Bearing.**
- This is the hardware template for converting heralded errors into corrections or retries
  instead of rejections. The logical map stays trace preserving and ordinary RB fitting applies
  (from literature).
- It needs real-time feed-forward; the Pauli-frame post-processing of [Combes2017] does not
  (verified analytically: a conditional repetition cannot be done in post-processing).

### 2.4 Nigg et al. 2014, "Experimental quantum computations on a topologically encoded qubit" (Innsbruck)

**Citation.** D. Nigg, M. Mueller, E. A. Martinez, P. Schindler, M. Hennrich, T. Monz,
M. A. Martin-Delgado, R. Blatt, arXiv:1403.5426 (21 Mar 2014); related DOI
10.1126/science.1253742. Fetched: https://arxiv.org/abs/1403.5426 and e-print `arxiv.tex`.

**Facts** (from literature [Nigg2014]):
- A 7-qubit color code. Transversal logical Cliffords X_L, Z_L, H_L, K_L, and up to 10 logical
  X_L gates applied to |-_y>_L.
- Not RB. They define a "quantum state fidelity within the code space",
  `F(rho_CS, rho_t) = F(rho, rho_t)/p_CS`, with `rho_CS = P_CS rho P_CS / p_CS`. This is the
  fidelity renormalized by the code-space population.
- A weighted fit `A exp(-n_gate/B)` to <Y_L> gives "a decay rate of the average expectation
  value of 3.8(5)% per gate".

**Bearing.** The code-space renormalization is the same ratio estimator as postselection,
`Pr(S and c)/Pr(c)`, and as the repo's accepted-shot fidelity (verified analytically). This
early encoded-gate decay used zero offset, like the repo's LRB fit.

### 2.5 Marques et al. 2022, "Logical-qubit operations in an error-detecting surface code" (Delft)

**Citation.** J. F. Marques, B. M. Varbanov, M. S. Moreira, et al. (13 authors), Nature
Physics 18, 80 (2022), DOI 10.1038/s41567-021-01423-9, arXiv:2102.13071. Fetched:
https://arxiv.org/abs/2102.13071 and e-print sections.

**Facts** (from literature [Marques2022]):
- A distance-2 surface code (Surface-7) with repeated error detection for up to n = 15
  cycles, comparing pipelined and parallel stabilizer schedules.
- Logical operations (preparation, measurement, single-qubit gates) are characterized by state
  and process tomography (a logical Pauli transfer matrix), not by RB. [Lacroix2025] makes the
  same point.
- "The fraction of post-selected data P in the repetitive error detection experiment decays
  exponentially with the number of cycles n. This is consistent with a constant
  error-detection rate per cycle gamma", fitted as `P(n) = A(1-gamma)^n`, with gamma_pip
  about 45% per cycle.
- `<Z_L>` is averaged over postselected runs. "the error detection scheme successfully
  post-selects leakage errors in both the ancilla and data qubits".

**Bearing.** This is the nearest hardware analog of `unif 1` on a d = 2 code. It reports the
acceptance decay rate gamma as its own figure of merit, separate from the conditional logical
observable (from literature).

### 2.6 Mayer et al. 2024, "Benchmarking logical three-qubit quantum Fourier transform encoded in the Steane code on a trapped-ion quantum computer" (Quantinuum)

**Citation.** K. Mayer, C. Ryan-Anderson, N. Brown, et al. (23 authors), arXiv:2404.08616
(12 Apr 2024). Fetched: https://arxiv.org/abs/2404.08616 and e-print `main.tex`.

**Facts** (from literature [Mayer2024]):
- Two-qubit logical RB of the transversal CNOT in the [[7,1,3]] code on H1-1 and H2-1.
  "Logical RB was previously considered in Ref. [Combes2017]."
- Random two-qubit Cliffords (11,520 elements) "followed by `C_inv = C_1^{-1}...C_L^{-1}`",
  a single inverse element. L in {2, 6, 10, 14}; 10 circuits per length; 100 shots each.
- Fit `p(L) = a f^L + 1/4`, `F_avg = 1 - 3(1-f)/4`, divided by the average number of CNOTs per
  Clifford. Result: 0.9991(2) (H1-1) and 0.9980(8) (H2-1) per logical CNOT, against physical
  0.99912(3) and 0.99817(5).
- Logical measurement decodes the final destructive readout ("a final correction can be
  applied to the raw logical outcome").
- The only discards are failed fault-tolerant preparations (repeat-until-success up to three
  times; "at most a few percent").
- The circuits use 16 qubits: 7 per block plus one preparation ancilla per block. Reading
  this, I infer that the RB sequences contained no mid-circuit syndrome extraction (from
  literature, with my inference).

**Bearing.** LRB with correction at the terminal readout only. It is the correction analog of
`const 1` (verified analytically, by comparing the protocols).

### 2.7 Paetznick, da Silva, Ryan-Anderson et al. 2024, "Demonstration of logical qubits and repeated error correction with better-than-physical error rates" (Quantinuum/Microsoft)

**Citation.** A. Paetznick, M. P. da Silva, C. Ryan-Anderson, et al. (33 authors),
arXiv:2404.02280 (v1 2 Apr 2024; v3 17 Nov 2024). Fetched: https://arxiv.org/abs/2404.02280
and e-print sections.

**Facts** (from literature [Paetznick2024]):
- Not RB. Repeated QEC with the [[12,2,4]] Carbon code, "using a combination of error
  correction and detection".
- Rejection: "pre-selection rejections are expected to be first-order in the physical error
  rates ... When error correction and error rejection are combined ... the post-selection
  rejections rates [are] second-order in the physical error rates, and therefore much
  smaller". For small circuits, rejection rates grow "roughly linearly with the logical
  volume".
- Error per QEC round from maximum-likelihood linear-trend fits: 0.3% ± 0.1%.

**Bearing.**
- A d = 4 code can correct single faults and reject only the rarer detectable ones, so
  rejection becomes second order.
- The [[5,1,2]]_3 code has d = 2 and corrects ⌊(d-1)/2⌋ = 0 errors. X0 and X3^-1 share a
  syndrome and differ by logical X (brief). Any policy that accepts a shot after a weight-1
  data syndrome therefore risks a first-order logical error, while rejecting it makes
  rejection first order (verified analytically).
- Faults that only flip ancilla outcomes (measurement shifts, reset shifts) are a different
  case, discussed in section 9.3.

### 2.8 Lacroix et al. 2025, "Scaling and logic in the color code on a superconducting quantum processor" (Google)

**Citation.** N. Lacroix, A. Bourassa, F. J. H. Heras, et al. (about 214 authors), Nature 645,
614–619 (2025), DOI 10.1038/s41586-025-09061-4, arXiv:2412.14256. Fetched:
https://arxiv.org/abs/2412.14256 and e-print `main_arxiv.tex`, `supplemental.tex`,
`preamble.tex`.

**Protocol and fit** (from literature [Lacroix2025]):
- LRB [Combes2017] of transversal single-qubit Cliffords (all 24) on a d = 3 color code.
- m from 0 to 10 random Cliffords, then "a final Clifford gate C^{-1} that inverts the
  sequence", "with a cycle of error correction between each Clifford gate". 25 sequences per
  m, each repeated 20,000 times. Decoded with a neural-network decoder (AlphaQubit) that
  "receives information about logical gates applied in between cycles".
- Reference: a state-preservation (idle) experiment.
- Both curves are fitted to `A p^m + 1/2`, and the error per Clifford is
  `eps_C = (1/2)(1 - p_Clifford/p_idle)` = 0.0027(3), against a logical error per cycle of
  0.0171(3).
- "We observe an exponentially decaying randomized benchmarking fidelity."

**Decoder and detector construction** (from literature [Lacroix2025]):
- "Applying a logical gate between error correction cycles transforms the stabilizers,
  changing how syndrome elements are constructed". After an H, the X auxiliary result in
  cycle n is compared with the Z result in cycle n+1; after S, a phase flip flips both X and Z
  auxiliaries.
- "Chromobius ... cannot be used to decode logical randomized benchmarking circuits", because
  it does not support gates that map X stabilizers to Y.
- The logical gate error also reflects "frequent basis changes on the logical qubit, including
  time spent in the Y-basis", which makes the logical noise gate dependent.

**Rejected shots.** None in LRB. Postselection is used only for magic-state injection, where
they argue it is "acceptable in this context because state injection protocols can be repeated
until a high-fidelity resource state is achieved" (from literature [Lacroix2025]).

**Bearing.**
- This is the hardware template for LRB-C.
- It shows the decoder must track how each logical Clifford maps the stabilizer group between
  rounds. For qutrits this is a Z_3-linear map on syndromes (conjecture that this carries over
  to the qutrit code without obstruction; see 2.9).
- Their idle-referenced, interleaved-style extraction is one way to separate the cost of gates
  from the cost of check rounds (from literature).

### 2.9 Berthusen et al. 2026, "Experimental validation of a compact fault-tolerant architecture for trapped ions" (Quantinuum Helios)

**Citation.** N. Berthusen, A. Lavasani, A. Benhemou, et al. (14 authors), arXiv:2609.03194
(2 Sep 2026). Fetched: https://arxiv.org/abs/2609.03194 and e-print `main.tex`,
`appendix.tex`.

**Protocol and fit** (from literature [Berthusen2026]):
- Two-qubit logical RB of the complete two-qubit Clifford group inside one [[20,2,6]]
  C4-Helix block, "under active error correction". Each S̄0S̄1 gate is followed by a round of
  adaptive syndrome extraction.
- "an independent random sequence for every shot", generated at runtime. The final inverse is
  computed by tracking "accumulated symplectic action".
- Decoded over "the complete circuit volume" with the Frontier decoder on a Stim DEM. "count a
  shot as a logical error if the expected initial logical state is not recovered."
- m in {5, 7, 13}, including the final corrective Clifford, with up to 27 rounds.
- Fit `p_L(m) = (1/2)[1 - (1-2 eps_C)^m]`, giving eps_C = 2.8 (+1.0/−1.6) × 10^-4, against a
  physical two-qubit RB error per Clifford of about 1.2 × 10^-3, "achieved without
  postselection".
- Simulations fit `(eps_0 - 1/2)(1-2 eps_C)^m + 1/2` with a SPAM offset eps_0.

**Detectors under logical gates** (from literature [Berthusen2026]):
- "Because the intervening logical gates transform both the stabilizer generators and logical
  observables, the corresponding detector structure depends on the particular Clifford
  sequence executed". Detectors are
  `D_i^(t) = sigma_i^(t) XOR (T^{-1} sigma^(t-1))_i`, where T maps the stabilizer generators
  before the gate to those after.
- Remark: "the sign tracking can be entirely dropped if all stabilizers are initialized in
  their +1 eigenspace ... In that case each syndrome can be a detector by itself. Those
  detectors however are not local in space-time, so decoders can still benefit from sign
  tracking".

**Hybrid correction plus postselection** (from literature [Berthusen2026]):
- Forced-gap postselection rejects a shot when the decoder's confidence gap Γ(s) between the
  best and second-best logical classes falls below a threshold γ.
- For γ > 0 in memory simulations, "the rejection rate per round exceeds the logical error
  rate per round".

**Bearing.**
- The brief's raw-ancilla-outcome detectors are the "each syndrome is its own detector" case.
  The encoding is ideal, so every stabilizer starts at eigenvalue 1. They are valid for
  postselection but not space-time local (from literature plus brief).
- A decoder should form time differences `sigma^(t) - T sigma^(t-1)` (mod 3, with the
  qutrit-code T induced by each logical Clifford's action on the stabilizer group, and the
  product of T's across the unchecked inverse block). This is the qutrit analog of their
  construction (conjecture: direct generalization).
- Forced-gap postselection gives a tunable path between LRB-D and LRB-C (from literature).

### 2.10 Mai et al. 2026, "A biased-erasure cavity qubit with hardware-efficient quantum error detection"

**Citation.** J. Mai, Q. Liu, X. Deng, Y. Cai, Z. Ni, L. Zhang, L. Hu, P. Zheng, S. Liu, Y. Xu,
D. Yu, arXiv:2601.21616 (29 Jan 2026). Fetched: https://arxiv.org/abs/2601.21616 and e-print
`Biased_erasure-v3.tex`, `Supplementary-v3.tex`.

**Protocol and fit** (from literature [Mai2026]):
- A logical qubit in the cavity Fock states {|0>, |2>}, with a mid-circuit erasure check after
  every Clifford: "(i) apply M random single-qubit logical Clifford gates ...; (ii) insert a
  mid-circuit erasure detection after each Clifford gate; (iii) append a recovery Clifford gate
  and a final erasure check".
- "The postselection probability ... decays exponentially with the number of Clifford gates
  M, revealing an erasure error probability of 9.77 × 10^-2 per Clifford."
- "The measured state probability remaining in |0_L> after postselection is fitted to an
  exponential function `F(M) = A p^M + B`, yielding a residual error of
  `r_Clif = (1-p)/2 = 6.21 × 10^-3` per Clifford." Per elementary X_L/2 gate: erasure
  4.50 × 10^-2 and residual 2.86 × 10^-3.

**Bearing.**
- Same structure as `unif 1`: a check after every Clifford, then postselection. They report
  two decays, the acceptance (erasure) rate and the conditional residual error (from
  literature).
- Unlike this repo, the recovery is one Clifford (no mirror) and there is a check after the
  recovery too (verified analytically, by comparing the protocols).

### 2.11 Bluvstein et al. 2023 and 2025 (Harvard/QuEra): no logical RB, but two relevant mechanisms

**Citations.**
- D. Bluvstein, S. J. Evered, A. A. Geim, et al. (21 authors), "Logical quantum processor based
  on reconfigurable atom arrays", Nature (2023, per the arXiv journal reference), DOI
  10.1038/s41586-023-06927-3, arXiv:2312.03982. Fetched: https://arxiv.org/abs/2312.03982 and
  e-print.
- D. Bluvstein, A. A. Geim, S. H. Li, et al. (23 authors), "Architectural mechanisms of a
  universal fault-tolerant quantum computer", arXiv:2506.20661 (25 Jun 2025), related DOI
  10.1038/s41586-025-09848-5. Fetched: https://arxiv.org/abs/2506.20661 and e-print.

**Facts** (from literature [Bluvstein2023, Bluvstein2025]):
- RB in these sources is physical only. A grep of both sources finds only physical RB.
- Sliding-scale error detection [Bluvstein2023]: "one can set a desired 'confidence
  threshold', where based on the syndrome outcomes one determines whether to accept a given
  measurement". The threshold is the weight of the correlated-decoding correction. Fidelity
  and XEB are reported against the accepted fraction.
- Gates per QEC round [Bluvstein2025]: "the error per logical gate depends on the number of
  gates applied per QEC round". With postselection on decoding confidence, "approximately
  three CNOTs per round is optimal".
- Their model: `1 - F_L ∝ (p_det + N Δp_det)^{(d+1)/2} / N`. "while physical gate performance
  can be well-characterized by a single fidelity value F, logical gate performance depends both
  on the decoding success probability ... and on how much that entropy increases per logical
  gate".

**Bearing.**
- The "logical error per Clifford" in this repo is a property of the check schedule as well
  as the gates. Different `unif k` give different per-gate numbers by construction, and there
  is no schedule-independent single number to recover (from literature, for transversal CNOT
  on surface codes; conjecture that it carries over to this repo).
- Sliding-scale detection is another detection-to-correction interpolation (from literature).

### 2.12 The repo's own prior work: Kabir et al. (sdim), and an APS 2026 talk

**Citation.** A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang,
B. Sorée, "Sdim: A Qudit Stabilizer Simulator", arXiv:2511.12777 (v1 16 Nov 2025, v2 12 May
2026). Fetched: https://arxiv.org/abs/2511.12777 and e-print sections `6_LRB.tex` to
`6.3_results.tex`.

**Facts** (from literature [Kabir2025]):
- They define LRB-D for the [[5,1,2]]_3 "detection surface code (Moussa 2016)": "we only
  perform a single terminal syndrome extraction prior to measuring Q̄, and we postselect
  measurements with the all zeroes syndrome".
- Fidelity `f_{D,k} = |sum_j P_{D,k}(j) w^j|`, averaged over circuits, fitted to `B alpha^D`.
- Full LRB with correction is "a demanding target for early qudit hardware". For d = 2 "the
  syndrome can reveal that an error has occurred, but it generally does not provide enough
  information to identify a unique correction"; "a practical qudit decoder must distinguish
  between different nontrivial Pauli powers".
- They state the averaged fidelity "takes the shape of an exponential decay [Wallman2018]".

**My comment.** Wallman's theorem covers the sequence-averaged survival probability for a
unitary 2-design under Markovian noise. For trace-decreasing maps it gives two exponentials
for the unnormalized quantity. It does not cover a postselected ratio or a modulus taken per
circuit. For LRB-D the single exponential is therefore an empirical model, not a consequence
of the theorem (verified analytically, by reading the theorem statement in 3.3).

**APS talk.** S. Ghosh (UC Davis) and A. Kabir (Rutgers), "Logical Randomized Benchmarking a
Qutrit Surface Code", APS Global Physics Summit 2026, 18 March 2026. Fetched:
https://meetings-archive.aps.org/smt/2026/mar-m09/7/ (the page shows no abstract text; I can
confirm only title, authors and date).

### 2.13 Searched, with no logical RB found

These sources mention RB only for physical gates (from literature, by grep of each e-print
source):
- C. Ryan-Anderson, J. G. Bohnet, K. Lee, et al. (17), "Realization of real-time
  fault-tolerant quantum error correction", arXiv:2107.07505 (fetched
  https://arxiv.org/abs/2107.07505).
- C. Ryan-Anderson, N. C. Brown, M. S. Allman, et al. (37), "Implementing fault-tolerant
  entangling gates on the five-qubit code and the color code", arXiv:2208.01863 (fetched
  https://arxiv.org/abs/2208.01863).
- L. Postler, S. Heußen, I. Pogorelov, et al. (13), "Demonstration of fault-tolerant universal
  quantum gate operations", Nature 605, 675–680 (2022), arXiv:2111.12654 (fetched
  https://arxiv.org/abs/2111.12654). [Lacroix2025] states that Postler et al. used partial
  process tomography.

I found no ETH Zurich logical RB experiment and do not cite one.

---

## 3. RB theory that governs the shape of the decay

### 3.1 Magesan, Gambetta, Emerson 2011

**Citation.** E. Magesan, J. M. Gambetta, J. Emerson, "Robust randomized benchmarking of
quantum processes" (arXiv title), Phys. Rev. Lett. 106, 180504 (2011), DOI
10.1103/PhysRevLett.106.180504, arXiv:1009.3639. Fetched: https://arxiv.org/abs/1009.3639 and
e-print `RandomBench.tex`.

**Facts** (from literature [Magesan2011]):
- Zeroth-order model `F_seq = A_0 p^m + B_0` for gate- and time-independent errors. A_0 and
  B_0 absorb SPAM and "an edge effect from the error on the final gate".
- First-order model `A_1 p^m + B_1 + C_1 (m-1)(q - p^2) p^{m-2}`, where "q - p^2 is a measure
  of the degree of gate-dependence".
- `r = 1 - p - (1-p)/d`.
- This requires the set to be "a unitary 2-design as well as a subgroup of U(d)".

**Bearing.**
- For d = 3, `r = 2(1-p)/3`, the repo's conversion (verified analytically).
- The first-order term is the oldest statement that gate dependence produces a
  non-exponential correction. Logical noise here is strongly gate dependent: H and P tokens
  contain two-qutrit gates, Pauli tokens only single-qutrit gates (brief).

### 3.2 Proctor et al. 2017, "What randomized benchmarking actually measures"

**Citation.** T. Proctor, K. Rudinger, K. Young, M. Sarovar, R. Blume-Kohout, Phys. Rev.
Lett. 119, 130502 (2017), arXiv:1702.01853. Fetched: https://arxiv.org/abs/1702.01853 and
e-print.

**Facts** (from literature [Proctor2017]):
- r and the average gate-set infidelity "can differ by orders of magnitude". The average
  gate-set infidelity is not gauge invariant.
- For small Markovian errors the curve is `P_m = A + B p^m` up to exponentially decreasing
  terms.
- r "does not appear to correspond to the infidelity of any physically valid, completely
  positive representation" (abstract).

**Bearing.** Compare LRB f with simulations of the same protocol, not with an average of
per-gate logical channel infidelities computed from a model (conjecture: recommendation).

### 3.3 Wallman 2018, "Randomized benchmarking with gate-dependent noise" (key for trace-decreasing logical maps)

**Citation.** J. J. Wallman, Quantum 2, 47 (2018), DOI 10.22331/q-2018-01-29-47,
arXiv:1703.09835. Fetched: https://arxiv.org/abs/1703.09835 and e-print
`RB_gate_dependence.tex`.

**Facts** (from literature [Wallman2018]):
- Theorem: for any unitary 2-design and any Hermiticity-preserving gate implementations, the
  average survival probability is `A p^m + B t^m + eps_m` with `|eps_m| <= delta_1 delta_2^m`.
  "p and t give the fidelity and loss of trace of the average noise between ideal gates".
  For completely positive trace-preserving noise this reduces to the single exponential plus
  the perturbation.
- "We prove all theorems without restricting to trace-preserving noise, as we anticipate that
  near-term implementations of RB on encoded qubits [Combes2017] will involve significant loss
  from the encoded space due to imperfect error-correction procedures."
- delta_2 ≲ 1/2 suffices for an essentially single exponential at m ≥ 10. Fluctuations at
  small m are possible.
- "any 'revivals' in RB experiments ... can be attributed directly to the noise being
  non-Markovian." The decay parameter is gauge invariant.

**Bearing.**
- This is the only rigorous model I found for logical RB with loss from the code space. It
  applies to the unnormalized quantity Pr(accepted and correct), not to the ratio (verified
  analytically).
- Fitting acceptance, conditional fidelity and their product separately lets the two-
  exponential structure be checked (conjecture: recommendation).

### 3.4 Helsen, Roth, Onorati, Werner, Eisert 2022, "A general framework for randomized benchmarking"

**Citation.** J. Helsen, I. Roth, E. Onorati, A. H. Werner, J. Eisert, PRX Quantum 3, 020357
(2022), DOI 10.1103/PRXQuantum.3.020357, arXiv:2010.07974. Fetched:
https://arxiv.org/abs/2010.07974 and e-print `theRBframework.tex`.

**Facts** (from literature [Helsen2022]):
- For a reference representation with multiplicities, RB data take the form
  `sum_lambda tr(A_lambda M_lambda^m)`, matrix exponentials. They "can appear to be strikingly
  non-exponential (at least if m is fairly small)".
- Under Markovian, time-independent noise the expected form holds up to an exponentially small
  correction. "any deviation from the prescribed functional form can in fact be taken as
  evidence of non-Markovian or time-dependent noise".
- "we recommend as a rule of thumb that RB experiments should not include very short (m ≤ 5)
  sequence lengths, especially when strong gate-dependent (but Markovian) noise is suspected".
- Logical RB [Combes2017], leakage RB [WBE2016, WoodGambetta2018] and real RB are in their list
  of covered protocols.

**Bearing.**
- The baseline depths include 0, 2 and 4, and the logical noise is strongly gate dependent.
  Fits that use m ≤ 4 may bias f. Refit with m ≥ 6 and compare (from literature for the rule;
  conjecture for its size here).
- On the full 5-qutrit data space the logical Cliffords permute 81 syndrome sectors that each
  carry a copy of the logical qutrit. That representation has multiplicities, so RB without
  postselection or correction on that space gives matrix-exponential decays, consistent with
  Harper and Flammia's "leaky subspace" remark (verified analytically for the multiplicity;
  conjecture for the size of the effect).

### 3.5 Ceasura, Iyer, Wallman, Pashayan 2022, "Non-exponential behaviour in logical randomized benchmarking"

**Citation.** A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan, arXiv:2212.05488 (11 Dec 2022;
no journal reference). Fetched: https://arxiv.org/abs/2212.05488 and e-print `main.tex`.

**Facts** (from literature [Ceasura2022]):
- They "construct a gate and time-independent noise model that results in the output of a
  logical randomized benchmarking protocol oscillating rather than decaying exponentially".
- Mechanism: "the role of the hidden qubits is assigned to the syndrome qubits used in error
  correction and these are strongly coupled to the logical qubits via a decoder".
- A "surjective" decoder (any logical Pauli is D(s) for some syndrome) can be driven by a
  stored syndrome pattern to apply logical Paulis. The minimum-weight decoder of a
  non-degenerate CSS code with d ≥ 3 is surjective.
- They first assume the syndrome reset fails, then show oscillations survive a 2% per-qubit
  reset probability. A classical repetition code embedded in the syndromes makes the memory
  robust to reset.
- "The output of RB is a linear combination of exponential decays provided that the
  representation is multiplicity free and that the noise is time and gate independent".
- Their examples are "somewhat contrived". They "suspect that there exist realistic noise
  models which result in less severe failure".

**Bearing.**
- In LRB-D the syndrome memory is classical: an uncaught fault leaves the data in a nontrivial
  syndrome sector until a later used round. Under correction, the decoder couples that memory
  back into the logical frame. Both are the mechanism class Ceasura et al. identify (from
  literature).
- I expect much milder, non-oscillatory deviations for SI1000-type Pauli noise, because
  nothing in it stores and replays logical Paulis adversarially (conjecture).

---

## 4. Qudit RB theory

### 4.1 Gross, Audenaert, Eisert 2007, "Evenly distributed unitaries: on the structure of unitary designs"

**Citation.** D. Gross, K. Audenaert, J. Eisert, J. Math. Phys. 48, 052104 (2007), DOI
10.1063/1.2716992, arXiv:quant-ph/0611002. Fetched: https://arxiv.org/abs/quant-ph/0611002
and e-print `design15.tex`.

**Facts** (from literature [GAE2007]):
- In the Clifford-design section (phase space `V = F_p^{2n}`, p prime): "the symplectic group
  acts transitively on `V^# := {a in V | a != 0}`". The Clifford twirl therefore equals the
  U⊗U twirl, so the Clifford group is a unitary 2-design.
- Any subgroup of Sp acting transitively on the nonzero vectors, together with the Weyl
  operators, also gives a 2-design.
- Criterion: a set is a 2-design if and only if its frame potential (the average of
  `|Tr U^dagger V|^4`) equals 2 (stated as Lemma 1 in [Graydon2021]).

### 4.2 Jafarzadeh, Wu, Sanders, Sanders 2020, "Randomized benchmarking for qudit Clifford gates"

**Citation.** M. Jafarzadeh, Y.-D. Wu, Y. R. Sanders, B. C. Sanders, New J. Phys. 22, 063014
(2020), DOI 10.1088/1367-2630/ab8ab1, arXiv:1911.08162 (v1 19 Nov 2019, v2 10 Jun 2020).
Fetched: https://arxiv.org/abs/1911.08162 and e-print `main2.tex`.

**Facts** (from literature [Jafarzadeh2020]):
- Qudit unitary-gate RB with pseudocode. For prime d the single-qudit Clifford group has
  `d^3(d^2-1)` elements; `|C_3| = 216`, generated by F_3 and S_3.
- Fit `F_seq(m) = A_0 p^{m-1} + B_0` (their sequence length counts the inverse), with
  `r = (1-p)(1 - 1/d^n)`.
- They assume gate-independent noise and note that gate-dependent noise "decays in the same
  form ... plus a perturbation [Wallman2018]".
- They state, without restricting to prime d: "the multi-qudit Clifford group `C_d^n`
  evidently forms a U2D based on Webb's analysis".

### 4.3 Graydon, Skanes-Norman, Wallman 2021, "Clifford groups are not always 2-designs"

**Citation.** M. A. Graydon, J. Skanes-Norman, J. J. Wallman, arXiv:2108.04200 (9 Aug 2021).
Fetched: https://arxiv.org/abs/2108.04200 and e-print `Final.tex`.

**Facts** (from literature [Graydon2021]):
- Theorem III.1: for composite d the Clifford group is not a unitary 2-design.
- Theorem III.2: the multipartite Clifford group is a 2-design only when every local dimension
  is the same prime.
- "The Clifford group is a unitary 2-design for prime dimensions". They note the result is
  folklore "in certain (but not all [Jafarzadeh2020, ...]) circles".

**Numerical check** (verified numerically; `lit-L1-lrb-checks/qutrit_checks.py`):
- The group generated by Morvan's H (Fourier) and S = diag(1, 1, w) has 216 elements mod
  phase and contains X and Z.
- Frame potentials `mean |Tr g|^{2t}` are 1.000000, 2.000000 and 7.000000 for t = 1, 2, 3.
  The Haar values are 1, 2 and 6, so the group is a 2-design and not a 3-design.

**Bearing.**
- The logical qutrit has d_L = 3 (prime), so uniform sampling from the 216-element catalog
  twirls any Markovian logical channel into a qutrit depolarizing channel. The standard
  single-exponential theory applies at the logical level whenever its other assumptions hold
  (from literature plus verified numerically).
- Qudit LRB with composite logical dimension (for example ququarts) would need a different
  twirling group or the multi-exponential framework of [Helsen2022] (from literature).

---

## 5. Qutrit RB experiments

### 5.1 Morvan et al. 2021, "Qutrit randomized benchmarking" (key paper for the metric)

**Citation.** A. Morvan, V. V. Ramasesh, M. S. Blok, J. M. Kreikebaum, K. O'Brien, L. Chen,
B. K. Mitchell, R. K. Naik, D. I. Santiago, I. Siddiqi, Phys. Rev. Lett. 126, 210504 (2021),
DOI 10.1103/PhysRevLett.126.210504, arXiv:2008.09134. Fetched: https://arxiv.org/abs/2008.09134
and e-print `main.tex`, `supplementary.tex`.

**Facts** (from literature [Morvan2021]):
- Five transmon qutrits.
- "Qubit-like" RB in each two-level subspace needs renormalized populations, and "the qubit RB
  sequences do not depolarize the full qutrit state", which motivates genuine qutrit RB.
- Qutrit Cliffords (216, generated by H and S) are compiled to 3.325 native gates on average.
- "To extract an RB number, we can fit this decay of <Z> to a single exponential
  `<Z>(m) = A p^m + b`", with `<Z> = P(|0>) + w P(|1>) + w^2 P(|2>)`.
- "Though the Pauli Z operator is in general non-Hermitian, its phase does not change under
  depolarization; thus the imaginary component, which by preparation was zero initially,
  remains zero throughout." They plot the imaginary part to confirm it.
- `r = (d-1)(1-p)/d`; the best single-qutrit error per Clifford is 2.38 × 10^-3. Also
  interleaved RB of native pulses and of the qutrit Hadamard (`r_H` = 4.6 × 10^-3),
  simultaneous RB for crosstalk, and cycle benchmarking of CSUM (process fidelity 0.82).

**Bearing.**
- The repo's metric is the same observable in the logical readout basis (verified
  analytically). Morvan take the real part and fit a free offset b. The repo takes the modulus
  per circuit and fixes b = 0.
- Under a qutrit depolarizing twirl the expectation of this observable has zero offset:
  `(1+2f^n)/3 + (w + w^2)(1-f^n)/3 = f^n` (verified analytically). So b = 0 is correct for this
  observable. A free-offset fit would test for non-depolarized residual structure (conjecture:
  useful diagnostic).
- On qutrit hardware, Morvan's qutrit-twirl requirement and the leakage to |3> they report are
  the physical-layer precedents for any real-device LRB (from literature).

### 5.2 Kononenko et al. 2021, "Characterization of control in a superconducting qutrit using randomized benchmarking"

**Citation.** M. Kononenko, M. A. Yurtalan, S. Ren, J. Shi, S. Ashhab, A. Lupascu, Phys. Rev.
Research 3, L042007 (2021), DOI 10.1103/PhysRevResearch.3.L042007, arXiv:2009.00599.
Fetched: https://arxiv.org/abs/2009.00599 and e-print.

**Facts** (from literature [Kononenko2021]):
- A flux-qubit-based qutrit, with RB over the qutrit Clifford group. Each basis-state
  population is fitted as `P_n = (P_i - P_f) p_n^l + P_f`.
- Average fidelity 98.89 ± 0.05%. Each gate is built from two-level Givens rotations, with
  leakage to higher levels estimated using the [WoodGambetta2018] model.

---

## 6. Leakage RB, the closest theory for "leaving the code space between checks"

### 6.1 Wallman, Barnhill, Emerson 2016, "Robust characterization of leakage errors"

**Citation.** J. J. Wallman, M. Barnhill, J. Emerson, New J. Phys. 18, 043021 (2016), DOI
10.1088/1367-2630/18/4/043021; arXiv title "Characterization of Leakage Errors via Randomized
Benchmarking", arXiv:1412.4126. Fetched: https://arxiv.org/abs/1412.4126,
https://iopscience.iop.org/article/10.1088/1367-2630/18/4/043021 (published title confirmed)
and e-print `characterizing_leakage_resubmit.tex`.

**Facts** (from literature [WBE2016]):
- Leakage is defined relative to "the logical code space defined by some quantum
  error-correcting code" as well as qubit subspaces. "our protocol can be applied to both
  physical and logical qudit systems".
- Survival rate `s(rho|E, H1) = tr[P_{H1} E(rho)]/tr rho`, with incoherent and coherent
  survival and leakage rates `l_inc = 1 - s_inc` and `l_coh = s_inc - s_coh`.
- Protocol: random gates from `G = {v ⊕ (±w)}` with unitary 1-designs on H1 and H2, no inverse
  gate, and a measurement of the population in H1. The decay is `A s_inc^{m-1}` for H1 = H, and
  `B lambda_+^{m-1} + C lambda_-^{m-1} + O(m eps)` for H1 a proper subspace, with
  `s_coh = lambda_+ + lambda_-`.
- "These transitions back to the Hilbert space make coherent leakage a fundamentally
  non-Markovian process."

### 6.2 Wood and Gambetta 2018, "Quantification and characterization of leakage errors"

**Citation.** C. J. Wood, J. M. Gambetta, Phys. Rev. A 97, 032306 (2018), DOI
10.1103/PhysRevA.97.032306, arXiv:1704.03081. Fetched: https://arxiv.org/abs/1704.03081 and
e-print `cjwood-leakage-arxiv1.tex`.

**Definitions and fits** (from literature [WoodGambetta2018]):
- Leakage rate `L1 = L(E(I1/d1))` and seepage rate `L2 = 1 - L(E(I2/d2))`.
- Their leakage RB: standard Clifford RB with the inverse, plus a measurement of the total
  computational population.
- Fit `p_{I1}(m) = A + B lambda_1^m` with `lambda_1 = 1 - L1 - L2`, `L1 = (1-A)(1-lambda_1)`
  and `L2 = A(1-lambda_1)`. Then fit `p_0(m) = A_0 + B_0 lambda_1^m + C_0 lambda_2^m` and use
  `F = [(d1-1) lambda_2 + 1 - L1]/d1`. "If leakage is weak ... fitting directly to a RB model
  `p_0(m) = A_0 + C_0 lambda_2^m`" is more robust.

**Assumptions** (from literature [WoodGambetta2018]):
- Twirling the computational subspace must induce at least a 1-design on the leakage subspace.
  "If the leakage subspace is not sufficiently depolarized then non-Markovian effects may
  appear"; the signature is "oscillations in the leakage decay model".
- Erasure is leakage with `L1 = p_l` and `L2 = 0`, giving `p_l(m) = 1 - (1 - p_l)^m`. A
  "Logical Leakage Errors" section treats this.
- They summarize earlier leakage-RB models (Epstein et al.) as bi- or multi-exponential.

**Bearing: mapping LRB-D onto leakage RB.**
- Data register: 3^5 = 243 = 81 syndrome sectors × 3 logical states, with four independent
  stabilizer generators. The code space is the "computational subspace" (d1 = 3); the other
  80 sectors form the "leakage subspace" (d2 = 240) (verified analytically).
- Every used check round projects onto syndrome 0. A rejected shot never comes back, so with
  respect to acceptance this is erasure with L2 = 0.
- Between used rounds, Pauli faults move the data among sectors in both directions. Seepage
  happens when a later fault cancels the syndrome, and the logical frame may shift.
- Consequences:
  - For `unif 1`, acceptance should be close to `A t^m`, the same model Marques and Mai fit
    (conjecture). Two effects modify it: faults that land after a round's data-ancilla
    interactions, and the unchecked inverse block.
  - For `const 1`, if the syndrome random walk mixes over Z_3^4, acceptance tends to the floor
    1/81 ≈ 1.2% instead of 0. This holds because single-qutrit X and Z faults generate all
    syndromes and the walk has an identity step (verified analytically). The brief's 79%
    rejection at m = 22, p = 1e-3 is still far from 98.8%.
  - Whether acceptance follows Wood and Gambetta's single exponential `A + B lambda^m` depends
    on whether the logical Cliffords depolarize the 80 sectors. They permute sectors but need
    not mix them uniformly, so a multi-exponential acceptance and a non-exponential
    conditional fidelity are expected (conjecture).

---

## 7. Mirror circuits and mirror RB, which explain the repo's Clifford-by-Clifford inverse

### 7.1 Proctor, Rudinger, Young, Nielsen, Blume-Kohout 2022, "Measuring the capabilities of quantum computers"

**Citation.** Nature Physics 18, 75–79 (2022), DOI 10.1038/s41567-021-01409-7,
arXiv:2008.11294. Fetched: https://arxiv.org/abs/2008.11294 and e-print.

**Facts** (from literature [ProctorCap2022]):
- Subroutine Clifford circuit (SCC) mirroring builds `R L0^{-1} C̃^{-1} Q0 C L0 I`. L0 is a
  random layer of single-qubit Cliffords and Q0 a random Pauli layer. Each quasi-inversion
  layer `L̃_i^{-1}` satisfies `U(L_i L̃_i^{-1}) = U(Q_i)` for a random Pauli Q_i, "or ... the
  identity layer, so that each quasi-inversion layer is simply the inverse layer".
- "Simple circuit mirroring is essentially a type of Loschmidt echo".
- "Systematic error cancellation ... it is possible for systematic (coherent) errors in the
  implementation of C to exactly cancel with systematic errors in the implementation of
  C^{-1}. For example, if `Λ(C) = V` and `Λ(C^{-1}) = V^{-1}` ... then S = 1". The central
  random Pauli layer and the alternating random Pauli layers are introduced to prevent this.
- Ordered (periodic) circuits fail sooner than disordered ones, which they read as a signature
  of coherent errors.

### 7.2 Proctor, Seritan, Rudinger, Nielsen, Blume-Kohout, Young 2022, "Scalable randomized benchmarking of quantum computers using mirror circuits"

**Citation.** Phys. Rev. Lett. 129, 150502 (2022), arXiv:2112.09853. Fetched:
https://arxiv.org/abs/2112.09853 and e-print.

**Facts** (from literature [ProctorMRB2022]):
- Circuit: `C = F0^{-1} P_d L1^{-1} ... P_{1+d/2} L_{d/2}^{-1} P_{d/2} L_{d/2} ... P1 L1 P0 F0`.
  Benchmark depth d counts the Pauli-dressed layers in both halves (d/2 forward and d/2
  inverse).
- Effective polarization `S = 4^n/(4^n-1) [sum_k (-1/2)^k h_k] - 1/(4^n-1)`. Fit
  `S̄_d = A p^d` with no offset, and `r = (4^n-1)(1-p)/4^n`.
- Theory, assuming Markovian but not necessarily gate-independent noise: the Pauli layers
  twirl errors into stochastic Pauli channels. `γ(E_A E_B) = γ(E_A)γ(E_B) + η`, where η, the
  rate at which errors cancel, "is negligible unless" the Pauli error distributions are
  sparse.
- `p^2 ≈ sum_L Ω(L) γ_{L^{-1}} γ_L`. A covariance between the error rates of L and L^{-1}
  gives `ε + O(ε^2) ≳ r ≳ ε/2 + O(ε^2)`.

**Bearing: frame analysis of the repo's inversion** (verified analytically; checked by exact
enumeration in `lit-L1-lrb-checks/mirror_exact.py`, appendix A2):

Single logical qutrit, gate-independent Weyl noise E_f after each forward Clifford (including
its check round) and E_b after each inverse Clifford, fixed readout of one Weyl observable.

1. Let `S_i = C_i ... C_1`. The forward error after C_i sits in frame S_i. The inverse error
   after `C_k^{-1}` sits in frame S_{k-1}. Frame S_i (1 ≤ i ≤ m-1) therefore holds the forward
   error of C_i and the inverse error of C_{i+1}^{-1}. S_m holds only the forward error of
   C_m. The inverse error of C_1^{-1} (and the terminal check round) sits untwirled in the lab
   frame S_0 = I.
2. S_1, ..., S_m are i.i.d. uniform, because `(C_1..C_m) -> (S_1..S_m)` is a bijection of
   G^m. SL(2, Z_3) acts transitively on the 8 nonzero Weyl vectors. So each frame turns its
   composite error w into a depolarized error with polarization `γ(w) = (9 Pr(w=0) - 1)/8`.
3. Hence `E[sum_k w^k P(k)] = λ_lab · γ(E_f) · γ(E_b ∘ E_f)^{m-1}`. With the brief's fit
   `y = b f^m`, the decay is `f = γ(E_b ∘ E_f)` and the edge terms go into b.
   - For depolarizing noise, `γ(E_b ∘ E_f) = γ(E_f)γ(E_b)`, which is the RB product.
   - For sparse noise it is not. At ε = 0.2 and m = 4, symmetric Z dephasing gives 0.182479
     exactly against 0.167924 from the product of polarizations, and forward Z with inverse X
     gives 0.114275 against 0.117547. The frame prediction matched the exact enumeration to
     six digits in every case.
4. The pairing in item 1 assumes after-gate, gate-independent errors. If C^{-1} is implemented
   as the exact time reverse of C, so that its error is the inverse of C's error, then the
   error of C_i^{-1} pairs with C_i in frame S_i. Coherent errors then cancel exactly, the
   Loschmidt echo of [ProctorCap2022]. SI1000 Weyl noise cannot echo out coherently, but on
   hardware it could. One random Pauli (or random Weyl) layer between every Clifford and its
   inverse, as in MRB, would prevent this.

**Consequences for the repo** (verified analytically):
- The fitted f is a decay per depth step: one forward Clifford, one check round, and one
  inverse Clifford of a different random Clifford.
- F = (1+2f)/3 is a per-pair average fidelity, and the per-logical-Clifford value is about
  `(1 + 2 sqrt(f))/3`, up to the η term.
- The repo's physical RB comparison uses the same mirrored 2m-Clifford structure (brief), so
  LRB against RB is compared per pair on both sides.

---

## 8. Code background: Moussa 2016, "Transversal Clifford gates on folded surface codes"

**Citation.** J. E. Moussa, Phys. Rev. A 94, 042316 (2016), arXiv:1603.02286 (v3 15 Oct 2016).
Fetched: https://arxiv.org/abs/1603.02286 (abstract only).

**Facts** (from literature [Moussa2016], abstract only):
- Two families of folded surface codes with transversal Clifford gates, "presented generally
  for qudits of any dimension".
- [Kabir2025] cites this as the source of the [[5,1,2]]_3 code.

**Bearing.** The logical H, P, X and Z of the brief are short-depth gates on this folded code.
Their unequal costs (SWAP or CZ against single-qutrit Paulis) are the source of the strong gate
dependence noted in 3.1 and 3.4 (from brief; the link to Moussa's construction is from
literature, abstract level).

---

## 9. Synthesis for this study

### 9.1 How the literature handles postselection and correction in RB-like logical decays

| Source | Platform / code | Between gates | Rejected shots | Fit model | Decay claimed to measure |
|---|---|---|---|---|---|
| [Combes2017] | theory | QEC round, recovery or not, ancilla refresh | none (syndrome traced out) | `A_L p_L^m + B_L` | logical average fidelity, Pr(No/Co/Un) |
| [HarperFlammia2019] | IBM QX5, [[4,2,2]] | nothing; parity checked at the end | discarded at the end; conditional survival = Pr(survive)/Pr(in code space), rejection taken as constant | real RB `A + b^m B + c^m C`, A fixed | logical two-qubit infidelity, with and without postselection |
| [Heeres2017] | Yale cat code | nothing | none | `0.5 + A e^{-n/τ}` | encoded gate error (R = id) |
| [Reinhold2020] | Yale cavity, path-independent gate | ancilla check, reset, conditional repetition | none (retry) | `A e^{-γn} + 1/2`, IRB | error-corrected gate error |
| [Nigg2014] | Innsbruck 7-qubit color code | nothing | renormalized by code-space population | `A exp(-n/B)` | decay per logical gate |
| [Marques2022] | Delft d=2 surface code | repeated detection | postselected; fraction `A(1-γ)^n` | (tomography for gates) | detection rate per cycle |
| [Mayer2024] | Quantinuum Steane code | nothing (transversal gates) | only failed preparation | `a f^L + 1/4` | logical CNOT fidelity |
| [Lacroix2025] | Google d=3 color code | one QEC cycle | none; NN decoder | `A p^m + 1/2`, idle reference | added error per logical Clifford |
| [Berthusen2026] | Quantinuum C4-Helix | adaptive syndrome extraction | none (forced-gap optional) | `(1/2)[1-(1-2ε_C)^m]` | error per two-qubit logical Clifford |
| [Mai2026] | cavity erasure qubit | erasure check | postselected; acceptance fitted separately | `A p^M + B` conditional; exponential acceptance | residual error and erasure rate per Clifford |
| [Kabir2025] | sdim, [[5,1,2]]_3 | nothing | postselected on the terminal syndrome | `B α^D` | LRB-D fidelity decay |

### 9.2 When is the logical decay single-exponential?

From literature:
- Markovian, time-independent noise with a multiplicity-free twirling representation gives a
  sum of exponentials, and one exponential for the Clifford 2-design [Magesan2011, Helsen2022].
- Gate dependence adds only an exponentially small perturbation for m ≳ 5–10 [Wallman2018,
  Helsen2022].
- Trace-decreasing noise gives two exponentials, `A p^m + B t^m` [Wallman2018].
- Multiplicities give matrix exponentials [Helsen2022].
- Leakage with seepage gives two or more exponentials, and oscillations if the leakage space
  is not depolarized [WBE2016, WoodGambetta2018].
- Syndrome memory coupled through a decoder can produce oscillations even for gate- and
  time-independent noise [Ceasura2022].
- Revivals imply non-Markovian noise [Wallman2018].

Conditional (postselected) decay under an ideal twirl (verified analytically):
- Model the accepted branch of one logical step as a CP trace-decreasing map Λ, and twirl it
  with the qutrit Clifford 2-design: `T(ρ) = a ρ + b Tr(ρ) I/3`, with trace factor `t = a + b`.
- After m steps the unnormalized state is `a^m ρ_0 + (t^m - a^m) I/3`. So acceptance is
  `t^m`, the accepted Weyl expectation is `a^m <X̄>_0`, and the conditional Weyl expectation
  is `(a/t)^m <X̄>_0`.
- This is a single exponential with zero offset; the conditional survival is
  `1/3 + (a/t)^m (...)`.
- The repo's zero-offset fit is therefore the correct model when each step's accepted branch
  is a fixed CP map on the logical qutrit (Markovian, no syndrome memory) and the twirl is
  effective.

Where this assumption fails in LRB-D (conjecture unless marked):
1. A fault after a round's data-ancilla interactions is not seen in that round; it carries over
   and is rejected one round later. This is short memory and should give a small edge
   correction.
2. With `unif k > 1` and `const c`, the unused rounds let faults accumulate. A pair of faults
   whose syndromes cancel passes undetected, possibly as a logical error. The conditional
   infidelity then gains a term growing like `(number of fault pairs) ~ (k p)^2` per block.
3. The m inverse Cliffords have no checks and are filtered only by the terminal round. Two
   detectable faults in the inverse block whose propagated syndromes cancel give an
   undetected logical error with probability `O((m p)^2)`. That term is not exponential in m.
   It is strongest for `const 1`, where both halves are filtered only once.
4. `const 1` acceptance goes to a floor of 1/81, not to 0 (verified analytically, section 6.2).

Under correction, a decoder that resolves each round's syndrome returns the data to the code
space, with a logical Pauli if it guessed wrong. That restores Combes's "trace out the
syndrome" structure, gives a trace-preserving logical map per step, and so makes the logical
process closer to Markovian (conjecture). For a d = 2 code the guess is wrong at first order
in p, so the logical error rate becomes O(p) (verified analytically: d = 2 corrects no
single-qutrit data error). Whole-circuit (space-time) decoding as in [Berthusen2026] couples
steps through the decoder. Ceasura et al. show that this coupling can in principle produce
non-exponential LRB. At low p I expect a near-exponential bulk with edge effects at both time
boundaries (conjecture).

### 9.3 Consequences and suggestions for the repo's analysis

1. Report three curves per (p, policy): acceptance a(m), conditional `<X̄>` c(m), and the
   joint `u(m) = a(m) c(m)`. Wallman's two-exponential form applies to u. Mai et al. and
   Marques et al. report acceptance decay rates as figures of merit (from literature for the
   precedent; conjecture that the split clarifies which effects drive the decay).
2. Fit windows: refit with m ≥ 6 and compare with the full-range fit [Helsen2022]. A
   free-offset fit of `<X̄>` tests for non-depolarized structure, since the expected offset is
   0 (from literature plus verified analytically).
3. Interpret f per pair (section 7.2) when quoting per-Clifford logical error rates (verified
   analytically).
4. The modulus `|.|` is harmless under this noise model, since the expectation is real, except
   at very low acceptance. With N_acc accepted shots, `E|X̂|` has a positive floor of order
   `N_acc^{-1/2}`. At the highest p in the sweep, `unif` policies keep almost no shots at large
   m. One accepted shot gives fidelity 1 if it is correct, so drop or flag points with tiny
   N_acc (verified analytically: elementary bias of an absolute value; conjecture about which
   baseline points are affected). In my DEM checks at p = 0.01, m = 6 (circuit
   `LRB/0/8/3.chp`), `unif 1` kept 266 of 4 × 10^5 shots.
5. LRB-C needs no real-time decoding. For stabilizer codes the recovery is a Pauli frame
   applied to the terminal outcome in post-processing [Combes2017, Berthusen2026]. Real-time
   feed-forward is needed only for conditional operations such as repeat-until-success
   [Reinhold2020] or adaptive extraction [Berthusen2026] (from literature).
6. Detectors for decoders. Raw ancilla outcomes are valid detectors because the encoding is
   ideal and starts in the +1 eigenspace. They are not space-time local
   [Berthusen2026, Lacroix2025]. Time differences `σ^(t) - T σ^(t-1) mod 3`, with the
   stabilizer map T of each logical Clifford, make measurement and reset faults appear as
   pairs of nearby detection events (conjecture: direct qutrit generalization).
   - Under detection-only these faults are counted as rejections. In SI1000 they are a large
     share: 4 × 5p measurement-result shifts plus 4 × 2p reset shifts per round, 28p in total.
   - Neither kind touches the data (verified analytically from the brief's noise model). A
     reset shift X^a on a Z-check ancilla is an X on the SUM target, which commutes with SUM.
     On an X-check ancilla it becomes Z^{±a} after H; that sits on the SUM control, commutes
     with SUM and SUM^-1, and is mapped back by H_INV. Either way exactly one outcome of that
     ancilla shifts: the next round's, because the reset layer comes after the measurements. A
     measurement-result shift changes only the recorded value.
   - At p = 1e-3, 28p = 0.028 against a measured first-round rejection of 0.061 (brief, `unif 1`
     at m = 0). Roughly half of the per-round rejection is therefore ancilla-only faults (a
     first-order estimate, not measured).
   - A policy that accepts shots whose only events are isolated in time, a time-aware
     detection, could cut rejection without increasing first-order logical error (conjecture;
     other agents should test it).
7. Hardware (Q3).
   - Use genuine qutrit Clifford twirling [Morvan2021] and measure leakage out of the qutrit
     [Morvan2021, Kononenko2021, WoodGambetta2018].
   - Insert random Weyl layers into the mirrored inverse, or use one recovery Clifford, to
     avoid echo cancellation [ProctorCap2022, ProctorMRB2022].
   - Pass logical-gate information to the decoder [Lacroix2025, Berthusen2026]. (All from
     literature.)
   - Forced-gap or sliding-scale postselection [Berthusen2026, Bluvstein2023] is a hardware-
     ready dial between LRB-D and LRB-C (from literature).

### 9.4 Open questions raised by the literature

- Is there a closed-form fit model for a postselected LRB ratio with syndrome memory? Harper
  and Flammia call this missing. The leakage-RB machinery of [WoodGambetta2018] and [WBE2016],
  with 80 sectors and the per-round projection, looks like the right starting point
  (conjecture).
- How large is the second-order mirror term η for the actual logical channels of H-type
  against P-type Cliffords after a check round? It can be computed exactly from the DEM by
  collecting each Clifford's logical Weyl distribution (conjecture: small at p ≤ 1e-3).
- Does the Harvard finding that logical error per gate depends on gates per round also hold
  for `unif k` in LRB-C with this d = 2 code? The prediction is that the optimal k balances
  undetected pairs against check-round noise (conjecture).

---

## Appendix A: numerical checks (all on a login node, under a minute each)

Scripts are in `studies/correction-decoders/qutrit/notes/lit-L1-lrb-checks/`. Run them with
`module load python/3.11`. The DEM check needs the user's site-packages for sdim, so do not
run it with `-I`.

**A1. `qutrit_checks.py`** (Clifford design, plus Monte Carlo illustrations):

```
single-qutrit Clifford group order (mod phase): 216  expected d^3(d^2-1) = 216
X in group: True  Z in group: True
frame potential t=1: 1.000000
frame potential t=2: 2.000000
frame potential t=3: 7.000000
[m=4 mirrored] symmetric Z-dephasing: max |Im<Z>| per sequence = 8.88e-16; mean Re = 0.18284; mean |.| = 0.18284
[m=4 mirrored] asymmetric Z only:     max |Im<Z>| per sequence = 2.49e-01; mean Re = 0.16165; mean |.| = 0.20553
```

The last two lines show that the modulus equals the real part when the noise is symmetric
under inversion, and is larger than it otherwise. The script's Monte Carlo lines at
ε = 0.2, m = 8 were within 2σ of the exact predictions and are superseded by A2.

**A2. `mirror_exact.py`** (exact enumeration over SL(2,3)^m, m ≤ 4, ε = 0.2):

```
depolarizing   m=4: mirrored exact=0.130141 frame-pred=0.130141 naive=0.130141 | standard exact=0.279582 pred=0.279582
Z-dephasing    m=4: mirrored exact=0.182479 frame-pred=0.182479 naive=0.167924 | standard exact=0.360750 pred=0.360750
fwd Z, bwd X   m=4: mirrored exact=0.114275 frame-pred=0.114275 naive=0.117547 | standard exact=0.252525 pred=0.252525
```

The frame prediction `λ γ_f γ(E_b∘E_f)^{m-1}` is exact for m = 1 to 4 in all cases. "naive"
is the product of polarizations.

**A3. `dem_symmetry_check.py`** (DEM sampling of baseline circuits from the read-only run
folder; acceptance as in `unif 1` and `const 1`):

```
LRB/3/6/2.chp (p=5e-3, m=4), 5e5 shots:
  unif 1:  accepted 39619 (0.0792); P(shift)=[0.97173 0.01411 0.01416]; sum w^k P = 0.95760-0.00004i
  const 1: accepted 91307 (0.1826); P(shift)=[0.95034 0.02483 0.02483]; sum w^k P = 0.92552+0.00000i
LRB/0/8/3.chp (p=1e-2, m=6), 4e5 shots:
  unif 1:  accepted 266; P(1)-P(2) = +0.015 (s.e. of the difference about 0.018)
  const 1: accepted 8421; P(1)-P(2) = +0.015 (s.e. of the difference about 0.0068, a 2.2σ fluctuation)
```

This is consistent with the exact symmetry P(1) = P(2) derived from the negation-symmetric
fault distributions in `src/lrb/dem_simulation.py` (`sample_nonzero_pauli_powers`, the `f`
and `p` channels, and the uniform N2 tail).

## Appendix B: fetched sources

All of these were fetched during this task. The arXiv abstract pages were fetched with
WebFetch, and the LaTeX source with `arxiv.org/e-print`, unless marked otherwise.

| Key | arXiv | URL fetched |
|---|---|---|
| Combes2017 | 1702.03688 | https://arxiv.org/abs/1702.03688 |
| HarperFlammia2019 | 1806.02359 | https://arxiv.org/abs/1806.02359 |
| Heeres2017 | 1608.02430 | https://arxiv.org/abs/1608.02430 |
| Reinhold2020 | 1907.12327 | https://arxiv.org/abs/1907.12327 |
| Nigg2014 | 1403.5426 | https://arxiv.org/abs/1403.5426 |
| Marques2022 | 2102.13071 | https://arxiv.org/abs/2102.13071 |
| Mayer2024 | 2404.08616 | https://arxiv.org/abs/2404.08616 |
| Paetznick2024 | 2404.02280 | https://arxiv.org/abs/2404.02280 |
| Lacroix2025 | 2412.14256 | https://arxiv.org/abs/2412.14256 |
| Berthusen2026 | 2609.03194 | https://arxiv.org/abs/2609.03194 |
| Mai2026 | 2601.21616 | https://arxiv.org/abs/2601.21616 |
| Bluvstein2023 | 2312.03982 | https://arxiv.org/abs/2312.03982 |
| Bluvstein2025 | 2506.20661 | https://arxiv.org/abs/2506.20661 |
| Kabir2025 | 2511.12777 | https://arxiv.org/abs/2511.12777 |
| GhoshKabirAPS2026 | none | https://meetings-archive.aps.org/smt/2026/mar-m09/7/ (title, authors and date only) |
| RyanAnderson2021 | 2107.07505 | https://arxiv.org/abs/2107.07505 |
| RyanAnderson2022 | 2208.01863 | https://arxiv.org/abs/2208.01863 |
| Postler2022 | 2111.12654 | https://arxiv.org/abs/2111.12654 |
| Magesan2011 | 1009.3639 | https://arxiv.org/abs/1009.3639 |
| Proctor2017 | 1702.01853 | https://arxiv.org/abs/1702.01853 |
| Wallman2018 | 1703.09835 | https://arxiv.org/abs/1703.09835 |
| Helsen2022 | 2010.07974 | https://arxiv.org/abs/2010.07974 |
| Ceasura2022 | 2212.05488 | https://arxiv.org/abs/2212.05488 |
| GAE2007 | quant-ph/0611002 | https://arxiv.org/abs/quant-ph/0611002 |
| Jafarzadeh2020 | 1911.08162 | https://arxiv.org/abs/1911.08162 |
| Graydon2021 | 2108.04200 | https://arxiv.org/abs/2108.04200 |
| Morvan2021 | 2008.09134 | https://arxiv.org/abs/2008.09134 |
| Kononenko2021 | 2009.00599 | https://arxiv.org/abs/2009.00599 |
| WBE2016 | 1412.4126 | https://arxiv.org/abs/1412.4126 and https://iopscience.iop.org/article/10.1088/1367-2630/18/4/043021 |
| WoodGambetta2018 | 1704.03081 | https://arxiv.org/abs/1704.03081 |
| ProctorCap2022 | 2008.11294 | https://arxiv.org/abs/2008.11294 |
| ProctorMRB2022 | 2112.09853 | https://arxiv.org/abs/2112.09853 |
| Moussa2016 | 1603.02286 | https://arxiv.org/abs/1603.02286 (abstract only) |

A note on IDs: the qudit-RB paper by Jafarzadeh et al. is arXiv:1911.08162. The ID
1911.08651 is an unrelated computer-vision paper, which I checked and discarded.
