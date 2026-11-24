# L2 literature notes: non-Markovian noise in RB, and the structure of logical noise under error correction and detection

Agent topic L2. Written 2026-10-28 for the qutrit `[[5,1,2]]_3` LRB-D study
(shared brief: `00-brief.md`). All papers below were fetched in this task
(arXiv abstract page, and for most of them the arXiv TeX source via
`https://arxiv.org/e-print/<id>`, which is where the theorem statements and
short quotations come from; some also via publisher pages). Quotations are
copied from the TeX sources.

## Claim labels

Every claim carries one label:

- **[LIT]** from literature: stated in the paper of the section it appears in
  (or in the paper named in the bracket), fetched in this task.
- **[VA]** verified analytically: derived by me in this task; the argument is
  given in place.
- **[VN]** verified numerically: computed by me in this task from the read-only
  baseline run folder (scripts in Appendix A; runtime under one second).
- **[CONJ]** conjecture: plausible, not verified here.

Baseline run folder (read-only):
`/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000/`.

---

## 0. Summary of what matters for this study

1. **What "Markovian" means in RB.** Every rigorous single-exponential theorem
   assumes that each gate label is always implemented by the same linear (CP,
   possibly trace-decreasing) map acting on the benchmarked system alone,
   independent of time and of the gates before and after it. Wallman 2018,
   Proctor et al. 2017, Merkel et al. 2021, Helsen et al. 2022 all state this
   assumption explicitly; Ziyad et al. 2025 call it "button-theoretic
   Markovianity" and, for logical qubits, "logical gate composability"
   (Ω[G_b∘G_a] = Ω[G_b]∘Ω[G_a]). [LIT]
2. **Theorems under that assumption.** Gate-dependent, Markovian,
   time-independent noise gives `A p^m + B t^m + ε_m` with
   `|ε_m| ≤ δ1 δ2^m` (Wallman 2018). Trace-decreasing maps are allowed, and
   Wallman says this was done because RB on encoded qubits will lose
   population from the code space. In Fourier language the average sequence
   is the m-th power of a fixed matrix with two eigenvalues near 1 (Merkel et
   al. 2021). When the reference representation has multiplicities, the
   output is a sum of matrix exponentials `Σ tr(A_λ M_λ^m)`, which can look
   "strikingly non-exponential" at small m (Helsen et al. 2022). [LIT]
3. **What non-Markovian noise does.** The average sequence fidelity becomes
   a contraction with a process tensor. It is a matrix power on the
   environment ("quality maps") and generally not one exponential
   (Figueroa-Romero et al. 2021, 2022). Correlated classical noise moves the
   decay toward a power law, at fixed mean, and broadens and skews the
   sequence-to-sequence distribution (Fong and Merkel 2017; Ball et al.
   2016; Mavadia et al. 2018). Under Markovian noise, increases of the
   average fidelity with m ("revivals" or "upticks") are impossible.
   Observing one is a witness of non-Markovianity, provided the protocol is
   the same at every m (Wallman 2018; Wallman and Flammia 2014;
   Figueroa-Romero et al. 2022). [LIT]
4. **Logical noise under QEC is generically non-Markovian.** It becomes so
   whenever the data are not returned to the code space every round. The
   syndrome register is then a classical (for Pauli noise) memory coupled to
   the logical qubit (Ziyad et al. 2025, with an "if and only if"
   statement). LRB can then oscillate even under gate- and time-independent
   physical noise (Ceasura et al. 2022). Combes et al. 2017 required that
   "syndromes are refreshed during rounds of error correction". Under Pauli
   noise and high-quality decoded QEC cycles, an approximate logical
   Markovian model exists, with deviations exponentially suppressed in the
   number of cycles (Kwiatkowski et al. 2025). That proof covers qubits and
   needs a decoder symmetry or syndrome randomization. [LIT]
5. **In this study's data, detection with unchecked windows is strongly
   non-Markovian; every-round detection is not measurably so at
   p ≤ 1e-3.**
   - `const 0`, `const 1` and `const 2` show accelerating decays: the
     per-interval log-slope grows 16-fold (`const 1`) and 26-fold
     (`const 0`) between m = 0–2 and m = 20–22 at p = 1e-3. For `const 0`,
     ln F ≈ −κ p² m² with κ ≈ 2.3. [VN]
   - The `const 1` acceptance saturates at exactly 1/81, the 3^4 = 81 states
     of the syndrome register, and the `const 0` acceptance at 1/9. [VN]
   - `unif 1` and `unif 2` fit a single exponential within statistical error
     at p ≤ 1e-3, and `unif 2` also at 2e-3. `unif 1` shows a marginal
     (2.5σ) positive curvature at 2e-3 and a significant one at 5e-3, driven
     by a steeper first interval (m 0→2). [VN]
   - Policies whose used-round set changes non-uniformly with m (`unif 5`,
     `const 5`) show upticks. These are protocol artifacts that track the
     length of the unchecked tail; they are not, by themselves, a
     non-Markovianity witness. [VN] [VA]
6. **Does correction mitigate non-Markovianity?** The literature answer is
   yes, insofar as decoding every round removes the "unused round" memory.
   It does not remove the memory created by noisy syndrome extraction, which
   is smaller and decays after a few rounds (Ziyad et al.; Kwiatkowski et
   al.). [LIT]
   For this circuit there are three further points:
   - Correction cannot touch the memory window of the unchecked inverse
     block; checks would have to be added there. [VA]
   - The distance-2 code cannot correct all single faults, so decoding trades
     rejection for a first-order logical error rate. [VA]
   - A space-time decoder must define detection events as
     `s_{j+1} − M_{C_{j+1}} s_j`, because the logical Cliffords permute the
     checks. [VA]

---

## 1. Definitions collected (with sources)

- **Markovian gate noise (RB sense).** "If each operation is independent of
  all external contexts -- e.g., time, external fields, ancillary qubits --
  then each gate can be represented by a process matrix" (Proctor et al.
  2017). Merkel et al. 2021 write the device as a map φ from group elements
  to CP trace-non-increasing maps and add that this "is consistent with
  Markovian error processes (otherwise we might parameterize our maps by
  some side-channel information, i.e., φ=φ(u,α))". Helsen et al. 2022 assume
  that "the implementation of a gate g ∈ G is always the same, independently
  of when it is performed in the RB protocol and independently of its
  context". [LIT]
- **Operational (process-tensor) Markov condition** (Pollock et al. 2018
  PRL). Perform a causal break at step k: measure, then re-prepare a
  randomly chosen known state. The process is Markovian iff the conditional
  state at every later step l depends only on the re-prepared input, for all
  measurements, inputs and earlier controls. Markov processes are
  K-divisible; the converse fails. [LIT]
- **Non-Markovianity measures** (Pollock et al. 2018 PRL;
  Figueroa-Romero et al. 2021):
  - `N = min_{Υ^Markov} D(Υ ‖ Υ^Markov)` for any CP-contractive
    quasi-distance D on generalized Choi states.
  - With relative entropy, the closest Markov process is the product of
    marginals, and `exp(−nN)` is the probability of confusing the process
    with that Markovian hypothesis after n samples of the Choi state.
  - The bond dimension of the process tensor's matrix-product form is the
    memory size; it is 1 iff the process is Markovian.
  - In RB, Figueroa-Romero et al. 2021 define the "RB non-Markovianity"
    `N_q = ‖F_m − F_m^M‖_q`. This is the q-norm, over sequence lengths, of
    the deviation of the average sequence fidelity (ASF) from a
    "Markovianized" counterpart in which the environment is discarded after
    every step. [LIT]
- **Logical gate composability** (Ziyad et al. 2025). The gadget retraction
  is `Ω[G] = D∘G∘E`, with D the ideal recovery followed by unencoding. Gates
  are logically Markovian iff `Ω[G_b∘G_a] = Ω[G_b]∘Ω[G_a]` for all pairs.
  [LIT]
- **Trace-decreasing / heralded maps in RB.** These are allowed in Wallman
  2018 (Hermiticity-preserving linear maps, with two decay constants p and
  t), Merkel et al. 2021 (CP trace-non-increasing, `p ≤ t ≤ 1`),
  Figueroa-Romero et al. 2021 and 2022 (CP noise on S⊗E), and Wallman,
  Barnhill and Emerson 2016 (leakage). Harper and Flammia 2019 still wrote:
  "There is at present no complete theory for fitting an RB decay curve with
  post-selection." [LIT]

---

## 2. RB theory under Markovian (possibly gate-dependent, trace-decreasing) noise

### 2.1 Wallman 2018, "Randomized benchmarking with gate-dependent noise"

Joel J. Wallman. Quantum 2, 47 (2018). arXiv:1703.09835;
DOI 10.22331/q-2018-01-29-47. Fetched: https://arxiv.org/abs/1703.09835,
https://quantum-journal.org/papers/q-2018-01-29-47/, TeX via
https://arxiv.org/e-print/1703.09835.

Extracted facts:
- Setting: a unitary 2-design 𝔾. Noisy implementations G̃ are arbitrary
  Hermiticity-preserving linear maps (not necessarily trace-preserving). The
  noise is Markovian and time-independent. [LIT]
- Twirl lemma: a 2-design twirls any linear map into
  `D_{p,t}(ρ) = (t/d) I + p (ρ − I/d)`, with `t(C) = tr[C(I)]/d` and
  `p(C) = (tr C − t)/(d²−1) = (d f(C) − t)/(d−1)`. [LIT]
- Theorem 3 (existence of L, R). There are linear maps L and R such that
  `E_G(G̃ L G†) = L D_{p,t}`, `E_G(G† R G̃) = D_{p,t} R` and
  `E_G(G R L G†) = D_{p,t}`. Here p and t are the largest eigenvalues of
  `E_G(G_u ⊗ G̃)` and `E_G(G̃)`. The "noise between ideal gates" R L is
  gauge invariant. [LIT]
- Theorem 6 (main). `E_{G⃗}(Q_{G⃗}) = A p^m + B t^m + ε_m`, with
  `|ε_m| ≤ δ1 δ2^m`, where `δ1 = max_G ‖Δ_G‖_{1→1} ‖ρ‖_1 ‖Q‖_1`,
  `δ2 = E_G ‖Δ_G‖` and `Δ_G = G̃ − L G R`. For CPTP noise, t = 1 and the
  standard `A p^m + B` is recovered. Numerically the perturbation was
  negligible for m ≥ 3 with strongly gate-dependent unitary noise. [LIT]
- Theorem 5 bounds `‖G̃ − G R‖` in the gauge L = I, so `δ2` is small when
  the gates are close to `G D_{p,t}`. [LIT]
- Quote on trace-decreasing maps: "We prove all theorems without
  restricting to trace-preserving noise, as we anticipate that near-term
  implementations of RB on encoded qubits [Combes2017] will involve
  significant loss from the encoded space due to imperfect error-correction
  procedures." [LIT]
- Experimental test. "the only effect that time-dependent noise can have is
  to alter the rate at which the survival probability decays [...].
  Consequently, any ``revivals'' in RB experiments, that is, statistically
  significant increases in the average survival probability as m increases
  [...], can be attributed directly to the noise being non-Markovian."
  [LIT]
- Short sequences: m ≈ 1–2 are the most sensitive to gate-dependent
  fluctuations, which may reach 0.1 at m = 1 and 0.01 at m = 2. These
  lengths "should be omitted when fitting". [LIT]
- Open problem: L and R need not be CPTP. [LIT]

Bearing on this study:
- For Pauli noise on a stabilizer code, the probability of a trivial
  syndrome does not depend on the logical input state. So the accepted
  branch of one step is `t_G · N_G`, with N_G a CPTP logical Pauli channel.
  The per-sequence acceptance is then a scalar. When every step's accepted
  branch returns to the code space, the study's estimator (the average over
  sequences of per-circuit conditional fidelities) is an RB-type average of
  a Markovian process with CPTP, gate-dependent maps N_G. Wallman's
  single-exponential-plus-perturbation form then applies to the forward
  half (Sec. 8.2 gives the conditions). The mirror structure is not
  covered verbatim by the theorem; see Sec. 3.10 and Sec. 8.1. [VA]
- If instead the unnormalized character and the acceptance are averaged
  separately, the theorem gives `χ(m) ≈ A p^m` (the `B t^m` term vanishes for
  the traceless Z_3 character under unital noise) and `a(m) ≈ C t^m`. Their
  ratio decays at rate p/t ≤ 1. [VA]
- The m = 0 and m = 2 points are where gate-dependence transients live. The
  baseline `unif 1` shows a steeper first interval, significant at
  p = 5e-3 (Sec. 8.4). This supports fitting from m ≥ 2, or with a free
  offset. [VN] for the observation; [CONJ] for attributing it to
  gate-dependence.

### 2.2 Proctor, Rudinger, Young, Sarovar, Blume-Kohout 2017, "What randomized benchmarking actually measures"

PRL 119, 130502 (2017). arXiv:1702.01853; DOI 10.1103/PhysRevLett.119.130502.
Fetched: https://arxiv.org/abs/1702.01853, TeX via https://arxiv.org/e-print/1702.01853.

Extracted facts:
- The average gateset infidelity ε (the mean AGI of the Cliffords against a
  fixed ideal representation) is not gauge invariant. It can differ from the
  RB number r by orders of magnitude. In Example 1, `r ∝ θ⁴` while
  `ε ∝ θ²`, for a detuning that is to first order a gauge transformation.
  [LIT]
- Exact theory. `P_m = Σ_i α_i λ_i^{m+1}`, where the λ_i are the eigenvalues
  of a `|C|·4^n`-dimensional block matrix 𝓡 built from
  `C̃_{j→k} = C_j^{-1} C_k`. Approximate theory: the "superduperoperator"
  `𝓛(E) = avg_i[C_i^{-1} E C̃_i]` gives
  `P_m = Σ ω_i γ_i^m + δ̃_m`, with
  `|δ̃_m| ≤ (1/2) avg_i ‖Λ_i − Λ̄‖_⋄` minimized over CPTP representations.
  For small errors the eigenvalues are 1, γ (near 1), and the rest near 0.
  So "the RB decay curve is guaranteed to always be exponential for small
  errors that can be described by CPTP maps -- including gate-dependent
  errors". Fitting a `C m p^m` term is unnecessary. [LIT]
- Diagnostic: "significant non-exponential decay is a clear symptom of
  non-Markovianity (e.g., time dependence)." [LIT]

Bearing on this study:
- The physical implementation of a logical gate acts on the syndrome
  register (Sec. 8.1). The "context" in Proctor's definition then includes
  the hidden syndrome state, and whenever that state is not reset the
  logical gate is not a fixed process matrix. [VA]
- r in this study should be read as a gauge-invariant decay rate, not as
  the average infidelity of a fixed per-Clifford error map. [LIT]

### 2.3 Merkel, Pritchett, Fong 2021, "Randomized Benchmarking as Convolution: Fourier Analysis of Gate Dependent Errors"

Seth T. Merkel, Emily J. Pritchett, Bryan H. Fong. Quantum 5, 581 (2021).
arXiv:1804.05951; DOI 10.22331/q-2021-11-16-581. Fetched:
https://arxiv.org/abs/1804.05951, https://quantum-journal.org/papers/q-2021-11-16-581/,
TeX via https://arxiv.org/e-print/1804.05951.

Extracted facts:
- RB is a nested convolution, `Φ_m(C) = E[φ(C C_{m-1}^{-1}) ··· φ(C_2 C_1^{-1}) φ(C_1)]`.
  Under the Fourier transform of matrix-valued functions on groups
  (Gowers–Hatami), `Φ̃_m(σ) = φ̃(σ)^m`. "In the limit of Markovian noise, the
  exponential decay of an RB sequence [...] is a direct consequence of it
  being a convolution." [LIT]
- In the trivial and adjoint irreps, `t = ⟨ψ_I|φ̃(σ_I)|ψ_I⟩` and
  `p = ⟨ψ_P|φ̃(σ_P)|ψ_P⟩`. Complete positivity gives p ≤ t and trace
  non-increase gives t ≤ 1 (their Appendix). If the average gate fidelity is
  1 − δ, the next singular value q satisfies `q ≤ sqrt(4^n(2δ − δ²)/d_σ)`.
  Hence `S_m = A t̄^m + B p̄^m + O(δ^{m/2})`. [LIT]
- Embedding a qubit Clifford group into a qutrit (leakage) gives a map that
  is not a representation: global phases become relative phases, and five
  unit Fourier eigenvalues appear instead of two. [LIT]

Bearing on this study:
- The decay of postselected data is governed by both the fidelity eigenvalue
  p and the trace eigenvalue t (acceptance). Acceptance is the t sector,
  which is not independent information. [VA]
- The physical implementation of the logical qutrit Cliffords on the
  five-qutrit space is a representation on the code space only (Sec. 8.1).
  This is the "not a representation / extra unit eigenvalues" situation of
  the leakage example. [CONJ] that the Fourier picture with multiplicities
  (Helsen et al.) is the right generalization.

### 2.4 Helsen, Roth, Onorati, Werner, Eisert 2022, "A general framework for randomized benchmarking"

PRX Quantum 3, 020357 (2022). arXiv:2010.07974; DOI 10.1103/PRXQuantum.3.020357.
Fetched: https://arxiv.org/abs/2010.07974, TeX via https://arxiv.org/e-print/2010.07974.

Extracted facts:
- For a reference representation `ω = ⊕_λ σ_λ^{⊕ n_λ}`,
  `p(i,m) ≈ Σ_λ tr(A_λ M_λ^m)`, with `M_λ` an `n_λ × n_λ` matrix that may be
  non-normal or even non-diagonalizable. So "p(i,m) can appear to be
  strikingly non-exponential (at least if m is fairly small) unless ω is
  known to be multiplicity-free." [LIT]
- Main theorem: under Markovian, time-independent noise with
  `(1/|G|) Σ_g ‖φ_r(g) − φ(g)‖_⋄ ≤ δ` small, the output is a linear
  combination of (matrix) exponentials up to an exponentially small error.
  "Any deviation from the prescribed functional form can in fact be taken as
  evidence of non-Markovian or time-dependent noise processes." Gate
  correlations and quasi-static noise break this form. [LIT]

Bearing on this study:
- Multiplicities on the full physical space arise naturally here. Each
  syndrome sector carries a copy of the logical action, and the logical
  gates permute sectors. Unchecked windows can therefore produce
  matrix-exponential (multi-rate) behaviour even with Markovian physical
  noise. Ceasura et al. 2022 exploit exactly this mechanism. [CONJ] for the
  precise multiplicity structure; [LIT] for the mechanism.

### 2.5 Wallman and Flammia 2014, "Randomized benchmarking with confidence"

Joel J. Wallman, Steven T. Flammia. New J. Phys. 16, 103032 (2014). arXiv:1404.6025;
DOI 10.1088/1367-2630/16/10/103032. Fetched: https://arxiv.org/abs/1404.6025, TeX
via https://arxiv.org/e-print/1404.6025.

Extracted facts:
- Time-dependent Markovian noise gives a product of per-step decay factors.
  Products over intervals can be estimated by ratios. "if there are no
  temporal correlations in the noise, then all of the parameters r_t [...]
  are lower-bounded by zero. Therefore any negative values (or average
  values) of r_t are an indicator of temporal correlations in the noise,
  that is, of non-Markovian behavior." [LIT]
- Sequence-sampling variance for qubits with mr ≪ 1 is bounded by
  `σ_m² ≲ m² r² + (7/4) m r²`. For example, 145 sequences suffice for
  m = 100, r = 1e-4 and their accuracy and confidence parameters
  ε = δ = 1%. For unitary and non-unital noise, the variance can stay
  constant with m. [LIT]

Bearing on this study:
- The "negative r_t" and "revival" tests require the same protocol at every
  m. The `unif k > 1` and `const c` policies change the used-round set with m
  and produce upticks without any temporal correlation in the physical noise
  (Sec. 8.4). [VA] [VN]
- 30 sequences are adequate in the Markovian regime. The Ball et al. DC
  example below shows how many more are needed under correlated
  accumulation. [LIT]

### 2.6 Gross, Audenaert, Eisert 2007, "Evenly distributed unitaries: on the structure of unitary designs" (brief)

D. Gross, K. Audenaert, J. Eisert. J. Math. Phys. 48, 052104 (2007). arXiv:quant-ph/0611002;
DOI 10.1063/1.2716992. Fetched: https://arxiv.org/abs/quant-ph/0611002, TeX via e-print.

- The paper reviews that "the full Clifford group C_d constitutes a
  2-design" in prime-power dimensions, via the Weyl twirl followed by a
  symplectic twirl. [LIT]
- So the 216-element single-qutrit Clifford group (216 = 3³(3²−1)) is a
  unitary 2-design. The logical twirl in this study therefore depolarizes the
  logical noise. This underlies the `y = b f^m` model with the Z_3 character.
  [VA] for 216 = 3³·8 and for the character argument; [LIT] for the
  2-design property.

---

## 3. Correlated and non-Markovian noise in RB

### 3.1 Ball, Stace, Flammia, Biercuk 2016, "The effect of noise correlations on randomized benchmarking"

Harrison Ball, Thomas M. Stace, Steven T. Flammia, Michael J. Biercuk. PRA 93,
022303 (2016). arXiv:1504.05307; DOI 10.1103/PhysRevA.93.022303. Fetched:
https://arxiv.org/abs/1504.05307, TeX via https://arxiv.org/e-print/1504.05307.

Extracted facts:
- Error accumulation in single-qubit Clifford RB is a 3D random walk in
  "Pauli space". The noise-averaged fidelity per sequence is
  gamma-distributed. In the Markovian limit the norm-squared walk length is
  Γ(3/2, 2Jσ²/3) per noise realization. In the DC (quasi-static) limit the
  walk is fixed by the sequence and only scaled by the noise. [LIT]
- The means agree between the Markovian and DC regimes for weak noise, but
  the higher moments do not. DC noise gives a broad distribution skewed
  toward high fidelity, whose variance and skew do not shrink with more noise
  averaging. [LIT]
- Some sequences contain error-suppressing (decoupling-like) subsequences.
  This is visible in filter functions that vary by orders of magnitude
  between sequences. [LIT]
- Sampling requirement: for ±10% accuracy on the mean (σ = 0.015), more than
  9 sequences are needed in the Markovian case and more than 443 in the DC
  case. The authors recommend publishing per-sequence fidelities. [LIT]

Bearing on this study:
- The sequence-to-sequence spread, after subtracting finite-shot noise, is a
  cheap diagnostic of sequence-dependent error accumulation. In the baseline
  it is negligible for `unif 1` but 16–62% of the infidelity for `const 0`
  (Sec. 8.4). [VN]
- In LRB-D the "correlation" comes from the stored syndrome. Error pairs
  whose propagated syndromes cancel play the role of correlated phases in
  Ball's walk. [CONJ]

### 3.2 Fong and Merkel 2017, "Randomized Benchmarking, Correlated Noise, and Ising Models"

Bryan H. Fong, Seth T. Merkel. arXiv:1703.09747 (2017). No journal reference was
listed on the arXiv page as fetched. Fetched: https://arxiv.org/abs/1703.09747,
TeX via https://arxiv.org/e-print/1703.09747.

Extracted facts:
- For Gaussian time-correlated dephasing with covariance matrix χ, the
  expected single-qubit sequence fidelity is `P_0 = 1/2 + Z/2`, where Z is
  the partition function of a long-range spin-1 Ising model,
  `Z = 3^{-N} Σ_{g ∈ {−1,0,1}^N} exp(−g·χ·g/2) exp(i g·μ)`. For d-state
  systems the site variables are the weights of the adjoint representation
  of SU(d). [LIT]
- Leading order is `Z_0 = |I + (2/3)χ|^{-1/2}(1 − (1/12) Σ_n Σ_nn² + …)`.
  For uncorrelated noise this gives `(1 + 4β/3)^{-N/2}` (exponential); for
  quasistatic noise, `(1 + 4Nβ/3)^{-1/2}` (power law). Exponential fits to
  quasistatic decays underestimated the error rate by factors 2–6 in their
  example. [LIT]
- Proposed test and fix: the initial decay `Z_0 ≈ 1 − (2/3) N β` does not
  depend on the off-diagonal correlations. So fit the early part linearly,
  `A + B(1 − 2Nε)`, with A from the asymptote; this recovered β to 7%.
  [LIT]

Bearing on this study:
- "Fit only the initial slope" is a model-agnostic estimator of the per-step
  error rate when the decay is non-exponential. For the `const` policies the
  baseline log-slope grows monotonically with m, so the initial interval
  gives a lower error rate than a global exponential fit. [VN] for the
  monotone growth; [CONJ] that the initial slope is the right
  "per-Clifford" figure of merit for LRB-D.

### 3.3 Mavadia, Edmunds, Hempel, Ball, Roy, Stace, Biercuk 2018, "Experimental quantum verification in the presence of temporally correlated noise"

npj Quantum Information 4, 7 (2018). arXiv:1706.03787; DOI 10.1038/s41534-017-0052-0.
Fetched: https://arxiv.org/abs/1706.03787, TeX via https://arxiv.org/e-print/1706.03787.

Extracted facts:
- The experiment used a trapped 171Yb+ ion with injected σ_z detuning noise,
  either slowly or rapidly varying, with 25 ≤ J ≤ 200 and 200 noise
  realizations per sequence. Distributions of survival over sequences
  matched gamma distributions with no free parameters: near-Gaussian for
  rapid noise, broad and skewed with a long low-fidelity tail for slow
  noise. [LIT]
- Proposed test: select "long-walk" sequences (large computed
  `‖V_2D‖²`). Under slow noise their error rate is 2–5× the standard
  `p_RB`; under rapid noise the two agree. The authors advocate reporting
  "the difference between the extracted p_RB of long and short walk
  sequences" as a quantitative measure of violation of the uncorrelated
  model. [LIT]
- For GST, the chosen gate set strongly affects estimated diamond distances
  under correlated σ_z errors. [LIT]

Bearing on this study:
- A logical-level analogue exists. One could classify sequences by how the
  logical Cliffords transform the syndromes, for example by how often the
  propagated single-fault syndromes collide within unchecked windows. Then
  compare decay rates between classes. [CONJ] (proposal).

### 3.4 Epstein, Cross, Magesan, Gambetta 2014, "Investigating the limits of randomized benchmarking protocols" (brief)

PRA 89, 062321 (2014). arXiv:1308.2928; DOI 10.1103/PhysRevA.89.062321. Fetched:
https://arxiv.org/abs/1308.2928.

- In simulations with systematic rotations, amplitude damping, leakage and
  1/f noise, RB estimated the average error rate to within a factor of two.
  The paper derives new decay models for 1/f noise and leakage. (Abstract
  only.) [LIT]

### 3.5 Pollock, Rodríguez-Rosario, Frauenheim, Paternostro, Modi 2018 (PRA), "Non-Markovian quantum processes: complete framework and efficient characterisation"

Felix A. Pollock, César Rodríguez-Rosario, Thomas Frauenheim, Mauro Paternostro,
Kavan Modi. PRA 97, 012127 (2018). arXiv:1512.00589; DOI 10.1103/PhysRevA.97.012127.
Fetched: https://arxiv.org/abs/1512.00589, TeX via https://arxiv.org/e-print/1512.00589.

Extracted facts:
- The process tensor `T_{k:0}` maps a sequence of control operations
  (CP maps, i.e. instruments) to the output state. It contains all
  uncontrollable dynamics, including system–environment interaction and
  correlated initial states. [LIT]
- Representation theorem: any k-step process is a generalized Choi state of
  a (2k+1)-body system. Temporal correlations become spatial ones. The state
  has a matrix-product-operator form with bond dimension at most the
  effective environment dimension. [LIT]
- Full tomography needs O(d^{4k}) basis operations. The one structural
  assumption is "that experimental control operations act solely on S and do
  not directly influence E (in practice, this could be used as a definition
  for S)". [LIT]

Bearing on this study:
- For LRB with Pauli noise, a natural environment is the syndrome register
  (81 states for the four Z_3 checks) together with the ancilla states. The
  logical Clifford implementations do influence E (they permute syndrome
  sectors, Sec. 8.1). In process-tensor language, part of the
  "control-dependent environment dynamics" must therefore be folded into the
  process. [VA]

### 3.6 Pollock, Rodríguez-Rosario, Frauenheim, Paternostro, Modi 2018 (PRL), "Operational Markov condition for quantum processes"

PRL 120, 040405 (2018). arXiv:1801.09811; DOI 10.1103/PhysRevLett.120.040405.
Fetched: https://arxiv.org/abs/1801.09811, TeX via https://arxiv.org/e-print/1801.09811.

Extracted facts:
- Definition: a process is Markovian when the system state after a causal
  break at step k depends only on the re-prepared input,
  `ρ_l(P_k | Π_k; A_{k-1:0}) = ρ_l(P_k)` for all inputs, measurements and
  earlier controls, and all l > k. [LIT]
- Theorem: a process is non-Markovian iff two control choices exist that
  give different conditional states after a causal break. Markovianity can
  be certified with finitely many linearly independent controls. [LIT]
- Corollary: for a quantum Markov process, fixing instruments yields
  classical distributions satisfying the classical Markov condition. Lemma:
  Markov implies K-divisible; the converse fails. [LIT]
- Measures:
  - `N = min D[Υ ‖ Υ^Markov]` for any CP-contractive quasi-distance.
  - With relative entropy, `Prob_confusion = exp(−nN)`.
  - The MPO bond dimension is the memory size.
  - The paper also remarks that "all open quantum evolutions generated by a
    time-independent system-environment Hamiltonian are non-Markovian [...]
    when considering more than two time steps". [LIT]

Bearing on this study (test design):
- A logical causal break can be simulated in the Pauli-frame / DEM
  framework: at step k, reset the logical frame component (an ideal logical
  measure-and-prepare) while keeping the syndrome frame. Then measure
  whether later logical statistics depend on the earlier Cliffords or
  faults. In Pauli-frame form, this is the dependence of the logical error
  after k on the hidden syndrome at k. [CONJ] (proposal; not implemented
  here.)

### 3.7 Figueroa-Romero, Modi, Harris, Stace, Hsieh 2021, "Randomized Benchmarking for Non-Markovian Noise"

Pedro Figueroa-Romero, Kavan Modi, Robert J. Harris, Thomas M. Stace, Min-Hsiu
Hsieh. PRX Quantum 2, 040351 (2021). arXiv:2107.05403; DOI 10.1103/PRXQuantum.2.040351.
Fetched: https://arxiv.org/abs/2107.05403,
https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040351, TeX via
https://arxiv.org/e-print/2107.05403.

The author list was checked on the PRX Quantum page and in the TeX source
(Robert J. Harris is an author). The arXiv listing as returned by the fetch
tool omitted him. [LIT]

Extracted facts:
- Model: noisy gates are `Ĝ_i = Λ_i∘(I_E ⊗ G_i)`, with `Λ_i` CP and
  trace-non-increasing on S⊗E (gate-independent, possibly time-dependent),
  and Clifford (2-design) gates acting on S only. [LIT]
- Markov iff the process tensor factorizes,
  `Υ_m^M = ρ_S ⊗ [⊗_i (Λ_i^M ⊗ I_{B_i})] ψ^{⊗ m+1}`. [LIT]
- Main theorem (ASF):
  - `F_m = tr[M tr_E ∘ Λ_{m+1} ∘ (𝒜_m + ℬ_m) ρ]`.
  - `𝒜_m = [∘_n (\$_{Λ_n} − Θ_{Λ_n}) ⊗ I_S]/(d_S² − 1)^m (ρ − ρ_E ⊗ I/d_S)`.
  - `ℬ_m = (∘_n Θ_{Λ_n}) ρ_E ⊗ I/d_S`.
  - The maps on E are
    `\$_Λ(ε) = Σ_{s,s'} ⟨s|Λ(ε ⊗ |s⟩⟨s'|)|s'⟩` and
    `Θ_Λ(ε) = tr_S[Λ(ε ⊗ I/d_S)]`.
  - The Markovian limit gives `F_m^M = p_1 ··· p_m A + B`.
  - For time-independent Λ, `(\$_Λ − Θ_Λ)^{∘m}` "will generally not render
    an exponential decay". [LIT]
- Caveat: "while non-Markovianity generally leads to non-exponential decays,
  there can also be other contextual factors [...], such as gate-dependence
  or other rather arbitrary external time-dependence leading to such
  behavior." [LIT]
- Classical correlations, as in Ball et al., reduce to a Markovian
  time-dependent ASF with a random noise parameter. Some non-Markovian
  processes are "RB-blind": an XX spin chain showed practically exponential
  decay. [LIT]
- Finite memory. Initial non-Markovian noise up to length ℓ shows up after ℓ
  only as SPAM-like constants. There are corollaries for late and for
  blockwise non-Markovian noise. [LIT]
- Proposed experimental tests:
  - Fix the Cliffords to identity on chosen steps, giving `F_{m/{i,…,j}}`,
    and match decay rates to the exponential tail. This estimates the memory
    length ℓ and builds a Markovianized ASF.
  - Interleave identities. If the deviations from an exponential vanish or
    fade, there is dissipation; if they persist, the non-Markovian noise is
    coherent.
  - Deviations are most efficiently detected at short m (m ≲ 50 in their
    example). [LIT]

Bearing on this study:
- The LRB-D analogue of Λ on S⊗E is the noisy physical step: one logical
  Clifford, plus a check round where present. E is the syndrome register and
  ancillas. Two of the paper's assumptions fail here:
  - The gates act on E (the logical H and P circuits transform the syndrome
    register; Sec. 8.1). The "gate-independent noise, gates on S only"
    assumption therefore does not hold verbatim, and the 2022 paper is the
    relevant generalization. [VA]
  - Postselection is an instrument on E. In the accepted branch of a used
    check with ideal measurement, it projects E onto the trivial syndrome:
    a "reset" of E that kills the memory, as in a causal break of the
    environment. [VA]
- The "fixed-identity" test translates directly to this study. Replace
  logical Cliffords on chosen steps by the identity (keeping the check
  rounds) and see whether the non-exponential `const`-policy decays become
  exponential. This separates check-round-induced memory from gate-mediated
  memory. [CONJ] (proposal.)

### 3.8 Figueroa-Romero, Modi, Hsieh 2022, "Towards a general framework of Randomized Benchmarking incorporating non-Markovian Noise"

Pedro Figueroa-Romero, Kavan Modi, Min-Hsiu Hsieh. Quantum 6, 868 (2022).
arXiv:2202.11338; DOI 10.22331/q-2022-12-01-868. Fetched:
https://arxiv.org/abs/2202.11338, https://quantum-journal.org/papers/q-2022-12-01-868/,
TeX via https://arxiv.org/e-print/2202.11338.

Extracted facts:
- Setting: any finite group with a multiplicity-free representation,
  `Ĝ = ⊕_π φ_π(g)`. The subspace twirl of a CP map Λ on S⊗E gives
  `(I_E ⊗ T_φ) Λ̂ = Σ_π Q̂_π ⊗ P̂_π`, where the Q_π are "quality maps" on E.
  Length-n quality maps are contractions of 1-point quality maps through E.
  [LIT]
- Theorem (ASF): `F_m = Σ_π ⟨⟨M| t̂r_E Λ̂_{m+1} (Q̂_{m,π} ⊗ P̂_π) |ρ⟩⟩`. In the
  Markov limit, the quality maps become scalars (quality factors) and the
  Markovian ASF is a sum of exponentials. [LIT]
- Corollary (finite non-Markovianity). For `Λ = qΓ + (1−q)Φ^M`, the ASF is a
  binomial mixture of Markovian and length-ℓ quality-map terms. [LIT]
- Signature: "having F_n > F_m for any n > m must imply a non-Markovian
  effect". A footnote adds that gate-dependence "can be regarded as a form of
  non-Markovianity". [LIT]
- Lemma (conditions for a non-monotone ASF). A necessary condition is
  `tr(Λ̂_{n+1}) Π_{i=m+1}^n ‖Λ̂_i‖ > d_S tr(Λ̂_{m+1})`. For time-independent
  noise and n = m+1 this reads `‖Λ̂‖ > d_S`: the noise must raise the purity
  of the S⊗E state by more than d_S², which "already rules out, for example,
  coherent (unitary) noise". It also requires `d_E > d_S^{2/(n−m) − 1}`.
  [LIT]
- Gate dependence. The Wallman/Proctor result, a single exponentially
  decaying perturbation, does not carry over to non-Markovian noise, because
  E carries the gate-dependence through the quality maps. The ASF and its
  variance are stable for small gate-dependent perturbations,
  `ε ≲ δ_F/((m+1) d_E d_S)`. [LIT]
- Proposed test: average the ASF over initial states and measurements
  ("average process fidelity") to remove the environment dependence of
  correlated SPAM. [LIT]

Bearing on this study:
- The lemma's mechanism for genuine upticks needs purity-increasing noise on
  S⊗E. Postselection, a non-trace-preserving projection that raises purity,
  is such a mechanism. So in postselected LRB, upticks are not excluded even
  for Markovian physical noise, once the protocol (the used-check set)
  changes with m. This is consistent with the baseline upticks in `unif 5`
  and `const 5`, which occur exactly when an extra used round shortens the
  unchecked tail (Sec. 8.4). [VA] for the observation that projective
  acceptance raises purity; [VN] for the uptick pattern; [CONJ] for the
  identification of mechanisms.
- With gates that act on E (Sec. 8.1), the 2022 "gate-dependence carried by
  E" case is the applicable theory for LRB with unchecked windows. [VA]

### 3.9 Figueroa-Romero, Papič, Auer, Hsieh, Modi, de Vega 2024, "Operational Markovianization in Randomized Benchmarking" (brief)

Quantum Sci. Technol. 9, 035020 (2024). arXiv:2305.04704; DOI 10.1088/2058-9565/ad3f44.
Fetched: https://arxiv.org/abs/2305.04704, TeX via https://arxiv.org/e-print/2305.04704.

- Ideal-pulse universal dynamical decoupling between RB gates reduces a broad
  class of non-Markovian ASFs to the Markovian exponential form, to first
  order in the decoupling interval. [LIT]
- Pauli twirling (randomized compiling) "does not Markovianize the average
  sequence fidelity", but it reduces the variance of sequence fidelities.
  [LIT]
- Bearing: randomizing the logical frame (the twirl) does not remove
  syndrome-register memory. Only actions on the environment, the analogue of
  decoupling or reset (in QEC: syndrome reset by decoding and recovery, or
  more checks), remove it. [CONJ] by analogy.

### 3.10 Proctor, Seritan, Rudinger, Nielsen, Blume-Kohout, Young 2022, "Scalable randomized benchmarking of quantum computers using mirror circuits" (brief; relevant to the mirror structure here)

PRL 129, 150502 (2022). arXiv:2112.09853; DOI 10.1103/PhysRevLett.129.150502. Fetched:
https://arxiv.org/abs/2112.09853, TeX via https://arxiv.org/e-print/2112.09853.

- Mirror circuits are made of layers followed by their inverses (with random
  Pauli layers). The theory assumes Markovian errors that are not
  necessarily gate-independent: "many, but not all, non-Markovian errors
  appear Markovian within random circuits". [LIT]
- `γ(E_d) ≈ Π_i γ_{i^{-1}} γ_i`, and
  `p² ≈ (1 − 4^n/(4^n−1) ε_Ω)² + 4^n/(4^n−1) Cov_Ω`. Here `Cov_Ω` is the
  covariance between the error rate of a layer and that of its inverse.
  [LIT]
- Bearing: this study's circuits are mirror-like (m forward Cliffords, then
  their exact inverses). The decay per m is that of a (forward, check,
  inverse) composite, and the forward and inverse error rates of the same
  Clifford are correlated. See Sec. 8.1 for the fault-propagation identity.
  [VA]

---

## 4. Trace-decreasing, postselected and heralded RB

### 4.1 Wallman, Barnhill, Emerson 2016, "Robust characterization of leakage errors"

Joel J. Wallman, Marie Barnhill, Joseph Emerson. New J. Phys. 18, 043021 (2016)
(journal reference from the arXiv page). arXiv:1412.4126; journal DOI
10.1088/1367-2630/18/4/043021. The arXiv listing title is "Characterization of
Leakage Errors via Randomized Benchmarking"; the TeX of arXiv v2 is titled
"Robust Characterization of Leakage Errors". Fetched: https://arxiv.org/abs/1412.4126,
TeX via https://arxiv.org/e-print/1412.4126.

Extracted facts:
- Leakage is defined relative to any subspace, including "the logical code
  space defined by some quantum error-correcting code". Survival rate:
  `s(ρ|E, H_1) = tr[P_{H_1} E(ρ)]/tr ρ`. For a proper subspace the survival
  rate can increase under composition, through coherences between H_1 and
  H_2. [LIT]
- Protocol: random gates from a 1-design `v ⊕ (±w)`, no inversion, measure
  the population of H_1. This gives `A s_inc^{m−1}` for H_1 = H and
  `B λ_+^{m−1} + C λ_−^{m−1}` for a proper subspace, with
  `λ_+ + λ_− = s_coh` (leakage and seepage). Gate-dependent corrections are
  O(mε), with ε the diamond-norm variation over the gate set. [LIT]
- Test: "If the experimental data deviates significantly from our decay
  curves, then the experimental noise is either strongly gate-dependent or
  non-Markovian." [LIT]

Bearing on this study:
- The `const 1` and `const 0` acceptance curves behave like leakage
  survival with many leakage sectors (80, resp. 8, non-trivial syndrome
  sectors). Their asymptote, 1/81 and 1/9, is the "seepage" equilibrium of
  a uniformly mixed syndrome register (Sec. 8.4). [VN] for the asymptotes;
  [CONJ] that a few-exponential leakage-type model (relaxation to 1/81)
  describes the full acceptance curve.

### 4.2 Harper and Flammia 2019, "Fault-Tolerant Logical Gates in the IBM Quantum Experience"

Robin Harper, Steven T. Flammia. PRL 122, 080504 (2019). arXiv:1806.02359;
DOI 10.1103/PhysRevLett.122.080504. Fetched: https://arxiv.org/abs/1806.02359, TeX via
https://arxiv.org/e-print/1806.02359.

Extracted facts:
- This is an RB experiment on [[4,2,2]] error-detected logical qubits with
  postselection. Infidelity fell from 5.8(2)% (two physical qubits) to
  0.60(3)% (encoded, postselected). [LIT]
- "There is at present no complete theory for fitting an RB decay curve with
  post-selection." The detected-error rate `1 − Pr(c|m)` rose slowly toward
  its asymptote of 0.5. The authors modelled it as constant and fitted
  standard RB to the conditional survival. [LIT]
- Without postselection, a fit in the 4-qubit space is "not justified as the
  physical implementation of the logical gates [...] do not form an
  orthogonal 2-design in the 4 qubit space (they only form such a design in
  the code subspace)"; "we have a leaky subspace". [LIT]

Bearing on this study:
- This is the closest experimental precedent for LRB-D. Its acceptance
  asymptote, 1/2 for one measured stabilizer, is the analogue of the 1/9
  (`const 0`) and 1/81 (`const 1`) asymptotes seen here. [LIT] for theirs;
  [VN] for ours.
- The "physical gates are a design only on the code space" remark is the
  same structural fact as in Sec. 8.1. [VA]

### 4.3 Levine et al. 2024, "Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons" (brief; hardware precedent)

Harry Levine, Arbel Haim, Jimmy S. C. Hung, et al. (113 authors listed). PRX 14,
011051 (2024). arXiv:2307.08737; DOI 10.1103/PhysRevX.14.011051. Fetched:
https://arxiv.org/abs/2307.08737, TeX via https://arxiv.org/e-print/2307.08737.

- Postselected single-qubit RB with 11 erasure checks "evenly spaced
  throughout the circuit, regardless of depth". The postselection probability
  was fitted to an exponential decaying to 0 (erasure error per Clifford
  4.38(3)e-3). The postselected survival was fitted to an exponential with
  offset 1/2 (residual error 1.01(1)e-4 per Clifford). [LIT]
- Mid-circuit checks catch decay-then-reheat events. "Coherence appears to
  grow with a small number of checks", then saturates, with about 0.8% false
  positives per check. [LIT]
- Bearing: this is a superconducting-hardware instance of a fixed number of
  checks spread over the depth. It is structurally like `const c`: the
  check density changes with depth, and events that leave and return between
  checks contaminate the postselected data. [VA] for the structural analogy;
  [CONJ] that the same sawtooth effect would appear on hardware.

---

## 5. Structure of logical noise under error correction and detection

### 5.1 Beale, Wallman, Gutiérrez, Brown, Laflamme 2018, "Quantum Error Correction Decoheres Noise"

Stefanie J. Beale, Joel J. Wallman, Mauricio Gutiérrez, Kenneth R. Brown, Raymond
Laflamme. PRL 121, 190501 (2018). arXiv:1805.08802 (arXiv listing title "Coherence in
quantum error-correcting codes"); DOI 10.1103/PhysRevLett.121.190501. Fetched:
https://arxiv.org/abs/1805.08802, https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.190501,
TeX via https://arxiv.org/e-print/1805.08802.

Extracted facts:
- Per-syndrome effective logical noise:
  `N̄(s)_{L,L'} = ⟨⟨L Π_s| N |L' Π_0⟩⟩ / (p(s) 2^k)`. No recovery is applied:
  "syndrome measurements alone decohere the noise". [LIT]
- Lemma 1: for single-qubit noise with infidelity r, the diagonal error
  entries are ≤ 3r and the off-diagonal entries are ≤ sqrt(6r). [LIT]
- Theorem 2: for an [[n,k,d]] stabilizer code and local noise, the
  syndrome-averaged off-diagonal (coherent) logical elements are
  `O(r^{d/2})`. For typical noise the logical infidelity scales as
  `r'^{⌈d/2⌉}`, so with correction the off-diagonals are at worst
  proportional to the diagonals. Coherent terms dominate the accumulated
  logical error after about 1/r rounds, independent of d. [LIT]
- Consequences: logical error rates are well quantified by logical
  infidelity (what LRB measures). Recovery maps can be optimized per syndrome
  by logical fidelity, as in Chamberland et al. [LIT]

Bearing on this study:
- The simulation uses Pauli noise, so per-syndrome logical channels are
  exactly Pauli and coherence plays no role. [VA]
- For hardware (question 3), detection with d = 2 is the worst case for
  coherence. The accepted (s = 0) branch keeps off-diagonal logical terms
  of order r^{d/2} = r (two first-order coherent amplitudes that combine
  into a logical operator). Its Pauli infidelity is of order r^d = r². So
  postselected logical noise of a d = 2 code can be close to unitary:
  diamond distance ~ sqrt(infidelity). [VA] (order-of-magnitude scaling
  from Lemma 1 and the expansion of N̄(0); not a theorem of the paper.)
- The random logical Cliffords of LRB twirl this coherence on average, but
  sequence-to-sequence variance grows (Wallman and Flammia 2014). [LIT]
  for the variance statement; [CONJ] for its size here.

### 5.2 Iverson and Preskill 2020, "Coherence in logical quantum channels"

Joseph K. Iverson, John Preskill. New J. Phys. 22, 073066 (2020). arXiv:1912.04319;
DOI 10.1088/1367-2630/ab8e5c. Fetched: https://arxiv.org/abs/1912.04319, TeX via
https://arxiv.org/e-print/1912.04319.

Extracted facts:
- Coherence measures:
  - unitarity `u = tr(N_u^T N_u)/(d²−1)`;
  - coherence angle `Θ = arccos(p/sqrt(u))`, with Θ² ≈ the normalized sum of
    squared off-diagonal Pauli-transfer elements;
  - for repeated unital channels,
    `r_m ≤ m r + ((d−1)/(2d)) m(m−1) Θ² + …` (Carignan-Dugas et al.),
    i.e. quadratic growth controlled by coherence. [LIT]
- Main theorem (toric code, minimal-weight decoding, perfect syndrome
  extraction, single-qubit rotations with |sin θ| < 1/L):
  `Σ_{i≠j} |χ̃_ij|² < (2/sin²θ)(Σ_{l≠0} χ̃_ll)² (1+E)`, `D_⋄ ≤ c r`, and
  `r_m = m r (1 + d_L (m−1) r (1+E)/((d_L+1) sin²θ))`. The quadratic term is
  exponentially small in L relative to the linear term. The constant-θ case
  remains open. [LIT]
- On syndrome-conditioned channels and postselection: "properties of
  logical channels conditioned on the syndrome are potentially of interest
  for near-term experiments using relatively small codes, particularly
  because it might be feasible to postselect by retaining favorable
  syndromes and rejecting unfavorable ones". For large codes, syndrome
  histories are too complex and syndrome-averaged channels are the useful
  object. [LIT]
- "We expect, for example, that randomized benchmarking applied to logical
  gates will accurately characterize logical noise even when the physical
  noise is highly coherent, at least for large code blocks." [LIT]

Bearing on this study:
- LRB-D on a 5-qutrit d = 2 code is the "small code, postselected,
  syndrome-conditioned" regime that Iverson and Preskill single out. Their
  large-code coherence suppression does not apply. [LIT]
- The quadratic coherent term in r_m is a second, hardware-only source of
  m² curvature in unrandomized sequences. In this study it would be removed
  on average by the logical Clifford twirl. [VA] for the twirl removing
  the average quadratic term (2-design property).

### 5.3 Huang, Doherty, Flammia 2019, "Performance of quantum error correction with coherent errors"

Eric Huang, Andrew C. Doherty, Steven Flammia. PRA 99, 022313 (2019).
arXiv:1805.08227; DOI 10.1103/PhysRevA.99.022313. Fetched: https://arxiv.org/abs/1805.08227,
TeX via https://arxiv.org/e-print/1805.08227.

Extracted facts:
- Noiseless error correction, effective logical channels computed per
  syndrome. Conditional channels are CP but not trace-preserving,
  `N_s(ρ) = x̄ρ + xZρZ − iy(Zρ − ρZ)`. [LIT]
- Theorem 1: for any [[n,k,d]] code with minimum-weight decoding and
  independent unitary errors (|θ_i| ≤ π/4), `D'_⋄ ≤ c_{n,k} D_⋄^d`. For
  dephasing at odd d the scaling is `D_⋄^{(d+1)/2}`. In terms of r:
  `D'_⋄ ≤ c r^{t+1/2}` for unitary noise vs `r^{t+1}` for dephasing. [LIT]
- Even-distance codes: "the effective channel conditioned on the syndrome
  measurement corresponds to a (very) weak measurement of logical Z. As a
  result the syndrome measurement outcomes do depend weakly on the logical
  state". Syndrome probabilities depend on y (the coherent part) and on
  `⟨Z̄⟩`. Averaged over syndromes, the unitary part cancels after one round
  for even-distance repetition codes. [LIT]

Bearing on this study:
- This is a hardware caveat for d = 2 detection. Under coherent errors, the
  acceptance probability can depend on the logical state. The accepted
  branch is then a weak logical measurement: a non-unital, state-dependent
  trace-decreasing map. Wallman's t-sector then couples to the logical
  signal, and the clean `(p/t)^m` interpretation of conditional decays needs
  to be revisited. Under the simulated Pauli noise this cannot happen.
  [VA] for the Pauli case (syndrome probabilities are independent of the
  logical state for Pauli errors on stabilizer codes); [LIT] for the
  even-distance coherent case; [CONJ] for its size on qutrit hardware.

### 5.4 Chamberland, Wallman, Beale, Laflamme 2017, "Hard decoding algorithm for optimizing thresholds under general Markovian noise"

Christopher Chamberland, Joel J. Wallman, Stefanie Beale, Raymond Laflamme. PRA 95,
042332 (2017). arXiv:1612.02830; DOI 10.1103/PhysRevA.95.042332. Fetched:
https://arxiv.org/abs/1612.02830, TeX via https://arxiv.org/e-print/1612.02830.

Extracted facts:
- "Markovian" here means "general completely positive and trace-preserving
  (i.e., Markovian) noise", acting independently on the physical qubits.
  Encoding, syndrome measurement and decoding are perfect. [LIT]
- Per-syndrome effective process matrix:
  `G_στ(N, R_l) = (1/p(l)) ⟨⟨E_σ| R_l 𝒫_l N(E_τ) 𝒫_l† R_l†⟩⟩`. The
  syndrome-averaged channel is the coding map. Hard decoding chooses
  recovery level by level from the current syndrome only: "Syndrome
  information from previous levels are not used to update the recovery
  maps." [LIT]
- The algorithm minimizes the logical infidelity per distinct per-syndrome
  process matrix by applying a transversal logical correction
  `T(L_g†) R`, including non-Pauli transversal gates. Gains over symmetric
  decoders reach orders of magnitude for coherent noise. Pauli twirling can
  raise or lower thresholds depending on the code. [LIT]
- "The next step in our work will be to generalize our results to include
  measurement and state-preparation errors." [LIT]

Bearing on this study:
- For decoders of this LRB, the per-syndrome effective-channel viewpoint
  says that, given a round's syndrome, the best Pauli-frame correction is
  the most likely logical coset. This is the right object only if each
  round's syndrome is reliable and the register is reset. With noisy
  measurements, a per-round effective channel is not well defined without
  the syndrome history. That is precisely the non-Markovian situation of
  Sec. 6. [VA]

### 5.5 Combes, Granade, Ferrie, Flammia 2017, "Logical Randomized Benchmarking"

Joshua Combes, Christopher Granade, Christopher Ferrie, Steven T. Flammia.
arXiv:1702.03688 (2017); no journal reference listed on the arXiv page as fetched.
Fetched: https://arxiv.org/abs/1702.03688, TeX via https://arxiv.org/e-print/1702.03688.

Extracted facts:
- Protocol: random logical Cliffords, each followed by a round of error
  correction ("with or without recovery") and "a refresh of the ancilla
  register", then the inverting Clifford, decoding, and measurement.
  `F_L(m) = A_L p_L^m + B_L`, where
  `Λ_L[ρ_L] = Tr_synd[U^{-1} D R̃ Ũ E(ρ_L ⊗ |0⟩⟨0|_synd)]`. [LIT]
- Assumptions: "we require the noise to be time-independent, Markovian, and
  gate-independent (all at the level of logical Clifford gates). Unique to
  the situation of logical qubits, we must also assume that syndrome
  measurement errors are uncorrelated between rounds of error correction".
  The figure caption adds: "It is important to our protocol that the
  syndromes are refreshed during rounds of error correction." [LIT]
- For stabilizer codes the recovery can be applied as a propagated Pauli
  frame in post-processing, so "no particular recovery map needs to be
  chosen at experiment time". The trivial recovery R = I gives the
  detection-free fidelity. Comparing `F_L(R̃Λ)` with `F_L(Λ)` estimates the
  probabilities of correctable and uncorrectable errors. [LIT]

Bearing on this study:
- The study's correction-vs-detection comparison through a Pauli-frame
  decoder on the DEM is exactly Combes's post-processing route. [LIT]
- Two differences break Combes's reduction to standard RB: there is no
  check after the m inverse Cliffords, and with detection-only there is no
  syndrome refresh in the unused rounds. The literature then predicts
  deviations from `A p^m + B` (Ceasura et al.; Ziyad et al.). [VA] for the
  structural comparison.

---

## 6. Logical noise with syndrome memory: emergent logical non-Markovianity

### 6.1 Ceasura, Iyer, Wallman, Pashayan 2022, "Non-Exponential Behaviour in Logical Randomized Benchmarking"

Athena Ceasura, Pavithran Iyer, Joel J. Wallman, Hakop Pashayan. arXiv:2212.05488 (2022);
no journal reference listed on the arXiv page as fetched. Fetched:
https://arxiv.org/abs/2212.05488, TeX via https://arxiv.org/e-print/2212.05488.

Extracted facts:
- They construct a gate- and time-independent noise model for which RB and
  LRB oscillate rather than decay. In RB the mechanism uses "hidden" qubits,
  i.e. a Clifford representation with multiplicities, to store a memory of
  the gate history that evades the exponential-decay theorems. [LIT]
- "In the setting of LRB, the target system is the set of logical qubits
  and thus is not a physical system that can be isolated from the
  environment". The syndrome qubits play the hidden memory, and "the decoder
  itself can function as a noise process that strongly couples the target
  system to the environment". With a surjective decoder (for example,
  minimum weight for CSS codes with d ≥ 3), the syndrome can make the
  decoder apply a stored destabilizer. [LIT]
- The construction first assumes that the syndrome reset fails. With a 2%
  reset probability the oscillations persist but decay. With 95% reset
  probability, a classical repetition code embedded in the syndromes
  preserves them, using `c ≤ log_{1/r}(sqrt(2km)/(1−A))` copies. [LIT]
- "We suspect that there exist realistic noise models which result in less
  severe failure of RB." [LIT]

Bearing on this study:
- In LRB-D the syndrome register is never reset physically. In detection,
  it is reset in the accepted branch only at used checks with correct
  readout. In correction via a Pauli frame it is not reset at all, only
  "accounted for". The unchecked inverse block and the unused rounds of the
  `const` policies are the "failed reset" regime of this paper. [VA] for
  the mapping.
- The baseline does not oscillate. The `const` policies do show strongly
  non-exponential decays (Sec. 8.4), which is the "less severe failure" the
  authors anticipated. [VN] for the observation; [CONJ] for the
  identification.

### 6.2 Ziyad, Blume-Kohout, Rudinger 2025, "Emergent Non-Markovianity in Logical Qubit Dynamics"

Jalan A. Ziyad, Robin Blume-Kohout, Kenneth Rudinger. arXiv:2512.08893 (v1 Dec 2025,
v3 Feb 2026); no journal reference listed on the arXiv page as fetched. Fetched:
https://arxiv.org/abs/2512.08893, TeX via https://arxiv.org/e-print/2512.08893.

Extracted facts:
- Definitions: button-theoretic Markovianity (every press of a button
  applies the same CPTP map) and logical gate composability (Sec. 1).
  Proposition: a device that is button-theoretic Markovian on the data
  qubits is button-theoretic Markovian for the logical gate set iff the
  gates are logically gate-composable. [LIT]
- Sufficient condition (appendix theorem). If every gate's image lies in
  the image of the encoding, the gate set is composable. Corollary: if every
  gate is followed by a perfect QEC round, the logical gates are Markovian.
  [LIT]
- Abstract: "non-Markovianity can emerge from Markovian physical operations
  if (and only if) the physical qubits are not necessarily returned to the
  code subspace after every round of QEC. In this situation, the syndrome
  qubits can act as a memory". [LIT]
- Stabilizer theorem (Sec. 3). For d ≥ 3 and a noisy syndrome-extraction
  round `R̃(ρ) = Σ_{s,s'} C_{s,s'} R(s') Π_s ρ Π_s R(s')` with all
  `C_{s,s'} > 0`, `Ω[R̃∘R̃] ≠ Ω[R̃]∘Ω[R̃]`. The mechanism: two correctable
  corrections applied in consecutive rounds combine into an uncorrectable
  error. [LIT]
- Examples (3-qubit repetition and [[5,1,3]], syndrome-readout errors only).
  The logical error rate is 0 in round 1, jumps, then relaxes quickly to a
  constant: "non-Markovianity is strongest early on, and disappears rapidly
  as multiple rounds of error correction are performed". [LIT]
- Open-system picture. The encoding unitary `U_E` factors the data space as
  `H_Syn ⊗ H_L`, and an ideal QEC round resets the syndrome qubits to |0…0⟩.
  The logical dynamics is "a hidden Markov process", which for Pauli noise
  is an explicit classical Markov chain on data basis states. [LIT]
- Outlook: "we expect fault-tolerant syndrome extraction to reduce (but not
  eliminate) non-Markovianity". They also note that reducing syndrome errors
  may make standard QCVV (logical RB and GST) more reliable. [LIT]

Bearing on this study:
- Under `unif 1`, with ideal syndrome readout and no data noise after the
  measured stabilizers, the accepted branch of each forward step has image
  in the code space. The sufficient condition then gives a Markovian forward
  half: `Π_0∘G̃` is a CP map with image in the code space. [VA]
- The baseline circuit violates this condition in three ways:
  - (a) ancilla readout and reset errors can mask a data error, leaving a
    hidden syndrome in the accepted branch;
  - (b) data idle and reset noise after the last measurement of a check
    round is carried undetected into the next step;
  - (c) the m inverse Cliffords have no checks.
  (a) and (b) give memory of about one round at second order in p;
  (c) gives an m-step memory window. [VA] (from the circuit and noise
  description in the brief).

### 6.3 Kwiatkowski, Friedman, Geller, Ziyad, Glancy, Knill 2025, "Constructing an approximate logical Markovian model of consecutive QEC cycles of a stabilizer code"

Alex Kwiatkowski, Aaron J. Friedman, Shawn Geller, Jalan A. Ziyad, Scott Glancy,
Emanuel Knill. arXiv:2509.16887 (2025); no journal reference listed on the arXiv page
as fetched. Fetched: https://arxiv.org/abs/2509.16887, TeX via https://arxiv.org/e-print/2509.16887.

Extracted facts:
- Total logical error channel of a QEC experiment with Pauli-stochastic
  noise:
  `Λ_tot^{(K)} = Σ_{s_0…s_K} γ_{s_K,s_{K−1}} ··· γ_{s_1,s_0} γ^{prep}_{s_0} Λ^{meas}_{s_K} ∘ Λ_{s_K,s_{K−1}} ∘ ··· ∘ Λ_{s_1,s_0} ∘ Λ^{prep}_{s_0}`.
  That is, a hidden Markov model whose per-cycle logical channel depends on
  the syndrome-register transition. [LIT]
- Main theorem. Assume (i) Pauli-stochastic noise; (ii) the syndrome
  marginal independence property (SMIP: transition probabilities
  `γ_{s_out,s_in}` independent of `s_in`), implied by a syndrome-decoding
  symmetry `D(E(s) + θ') = s + D(θ')` or enforced by randomizing the
  syndrome subsystem; (iii) single-cycle logical entanglement fidelity
  `f_1 ≥ 1 − 1/64`. Then a (G, ε)-approximate logical Markovian model exists
  with `ε = 2(1+√2) sqrt(ε_1)`, and its deviations from the true outcome
  probabilities are exponentially suppressed in the number of cycles. The
  construction is explicit. [LIT]
- Odd-distance classical codes with minimum-weight decoding satisfy the
  decoding symmetry. [LIT]
- "This justifies the common technique of fitting exponential decays to the
  results of QEC memory experiments." For logical Clifford RB with a QEC
  cycle after each gate and gate-independent noise, they expect
  exponentially decaying success probabilities. Gate-dependent errors, and
  "stabilizer codes with qudits", are listed as future work. [LIT]

Bearing on this study:
- The theorem applies to decoded QEC cycles (correction), not to
  postselection, and only to qubits. Its symmetry condition is stated for
  odd distance. This code has d = 2 and qutrits, and minimum-weight decoding
  has ties (for example X0 vs X3^{-1}), so the decoding symmetry is unlikely
  to hold. SMIP would have to be enforced by syndrome randomization, i.e.
  random stabilizer-sector Pauli frames. [VA] for the tie; [CONJ] that SMIP
  fails without randomization and that the theorem extends to Z_3.
- Practical implication [CONJ]: a decoded `unif 1`-type experiment (a
  decoder over every round's syndromes) should decay exponentially to good
  approximation after a short transient, provided the single-step decoded
  logical error is ≤ 1/64. For d = 2 at the baseline p this condition is
  borderline, because single faults are not all correctable.

### 6.4 Gravier, Ayral, Vermersch, Meunier, Savin 2025, "Simulated non-Markovian Noise Resilience of Silicon-Based Spin Qubits with Surface Code Error Correction" (brief)

Oscar Gravier, Thomas Ayral, Benoît Vermersch, Tristan Meunier, Valentin Savin.
arXiv:2507.08713 (2025). Fetched: https://arxiv.org/abs/2507.08713, TeX via
https://arxiv.org/e-print/2507.08713.

- Setting: distance-3 rotated surface code and its XZZX variant, under
  non-Markovian (1/f-type) noise on Larmor frequency and exchange.
  Measurement errors are excluded. Physical Ramsey decay is Gaussian; the
  logical fidelity decays exponentially. "Such an exponential decay suggests
  that quantum error correction effectively Markovianizes the noise". This
  gives `T*_{2,L} ∝ (T*_2)^4`. [LIT]
- Bearing: this is evidence that active correction, with every round
  decoded and no syndrome-readout errors, converts physical temporal
  correlations into an exponential logical decay. It does not test the
  syndrome-memory mechanism, since measurement errors are excluded. [LIT]

### 6.5 Kam, Gicev, Modi, Southwell, Usman 2025, "Detrimental non-Markovian errors for surface code memory" (brief; abstract only)

John F. Kam, Spiro Gicev, Kavan Modi, Angus Southwell, Muhammad Usman. Quantum Sci.
Technol. 10, 035060 (2025). arXiv:2410.23779; DOI 10.1088/2058-9565/adebab. Fetched:
https://arxiv.org/abs/2410.23779.

- Multi-time "streaky" correlations affecting syndrome qubits and two-qubit
  gates can severely degrade logical error-rate scaling. Other temporal
  correlations are less harmful. [LIT]
- Bearing: for hardware, correlated ancilla and readout noise is the class
  most likely to defeat a correction-based LRB. [CONJ]

### 6.6 Kobayashi, Manabe, White, Farrelly, Modi, Stace 2024, "Tensor-network decoders for process tensor descriptions of non-Markovian noise" (brief; abstract only)

arXiv:2412.13739 (2024). Fetched: https://arxiv.org/abs/2412.13739.

- Builds a maximum-likelihood decoder from a process-tensor noise model
  (the "strategic code" framework), with a tensor-network approximation.
  Correlated noise can significantly degrade performance. [LIT]
- Bearing: if the hardware noise has memory, a decoder for this study's
  correction variant could be conditioned on a process-tensor model rather
  than on the independent-location DEM. [CONJ]

### 6.7 Morales-Hashemi, Smith, Cao 2026, "What Must a Quantum-Memory Decoder Know About Temporally Correlated Noise?" (brief; abstract only)

Danesh Morales-Hashemi, Graeme Smith, Ningping Cao. arXiv:2610.03545 (2026). Fetched:
https://arxiv.org/abs/2610.03545.

- Two noise models with identical syndrome statistics and the same
  spatiotemporal Pauli process can require different logical corrections.
  Without model information the optimal final recovery can lose fidelity
  approaching 1/2. Calibration cost scales with inverse rotation angle to the
  power of the X distance. Discarding the Pauli-twirl record makes the Pauli
  process sufficient. [LIT]
- Bearing on sustainable hardware use (question 3): the twirled Pauli
  description, which is what the DEM is, suffices for decoding only if the
  twirl record is discarded. Coherent temporally correlated noise otherwise
  needs extra calibration. [LIT]

---

## 7. Experimental tests for non-Markovianity proposed in the literature

| Test | Source | Requirement | Applicability here |
|---|---|---|---|
| Revival or uptick, `F_n > F_m` for n > m | Wallman 2018; Figueroa-Romero et al. 2022 | Same protocol at every m | Valid for `unif 1`, and `unif k` at m ∈ kℕ; not for `const c` or `unif k` at other m [VA] |
| Negative interval infidelity `r_t` | Wallman and Flammia 2014 | Same as above | Same caveat [VA] |
| Non-exponential shape | Proctor et al. 2017; Helsen et al. 2022; Figueroa-Romero et al. 2021 | Separate gate-dependence transients (small m) and multiplicities | Used in Sec. 8.4 [VN] |
| Fix chosen Cliffords to identity; interleave identities | Figueroa-Romero et al. 2021 | Ability to replace gates | Direct in simulation [CONJ proposal] |
| Sequence distribution (gamma shape); long- vs short-walk rates | Ball et al. 2016; Mavadia et al. 2018 | Per-sequence data, shot noise removed | Used in Sec. 8.4 [VN] |
| Initial-slope fit, independent of correlations | Fong and Merkel 2017; Ball et al. 2016 | Small m | Recommended for `const` [CONJ] |
| Causal break | Pollock et al. 2018 PRL | Reset the system while keeping the environment | Logical-frame reset in the Pauli-frame simulator [CONJ proposal] |
| Logical gate composability, `Ω[G2∘G1]` vs `Ω[G2]∘Ω[G1]` | Ziyad et al. 2025 | Estimate one- and two-step logical channels | Direct with the DEM sampler (Pauli channels) [CONJ proposal] |
| Average over input states and measurements | Figueroa-Romero et al. 2022 | Several logical preparations and readouts | Partly available (only \|+⟩_L and X readout are used now) [VA] |
| Leakage-type acceptance decay | Wallman, Barnhill, Emerson 2016 | Acceptance vs m | Used for the saturation check [VN] |

---

## 8. Bearing on this study (synthesis)

### 8.1 Structural facts about the circuits

1. **Exact mirror structure.** The brief says the inverse block applies
   each forward Clifford's reversed and inverted token sequence, in reverse
   order. So the noiseless physical circuit acts as the identity on all five
   data qutrits, not only on the code space. Let `W_k = U_{C_k}···U_{C_1}` be
   the physical forward prefix. A Pauli fault after forward block k reaches
   the end conjugated by `W_k^{-1}`. A fault after inverse block k+1 (that
   is, after `U_{C_{k+1}}^{-1}`) is also conjugated by `W_k^{-1}`. Each
   (forward step k+1, inverse step k+1) pair is therefore "twirled" by the
   same random prefix, whose logical part is a uniformly random logical
   Clifford for k ≥ 1. [VA] This is the mirror-RB pairing of Proctor et al.
   2022. [LIT]
2. **The logical gates act on the syndrome register.** Using the brief's
   gate definitions and the qudit Fourier convention `F X F† = Z`,
   `F Z F† = X^{-1}`:
   - Logical H = SWAP(2,3), then H on {0,2,3,4}, then H_INV on 1. It maps
     `S_Z1 = Z0 Z1^{-1} Z3^{-1} ↦ X0^{-1} X1^{-1} X2 = S_X1^{-1}` and
     `S_X1 ↦ Z0 Z1^{-1} Z3^{-1} = S_Z1`. Similarly `S_Z2 ↦ S_X2^{-1}`.
   - Logical H therefore exchanges the Z-type and X-type syndrome
     components. The exchange does not depend on the sign convention (only
     the signs do).
   - Logical P puts Z-components on qutrits 0, 1 and 3 into the image of
     `S_X1`. If, as intended, the circuit preserves the stabilizer group, it
     maps `S_X1` to `S_X1` times a non-trivial Z-type stabilizer.
   [VA]
   So, in the logical ⊗ syndrome factorization of Ziyad et al. and Ceasura
   et al., each physical logical gate is a syndrome-sector permutation
   `s ↦ M_U s`, possibly with syndrome-dependent logical Paulis that depend
   on the pure-error convention. It is not `G ⊗ I_syndrome`. [VA]
   Consequences:
   - (a) Figueroa-Romero et al. 2021's "gates act on S only" assumption
     fails in unchecked windows, and the 2022 "gate-dependence carried by E"
     case applies. [VA]
   - (b) In the accepted branch of a used check, the register is in sector
     0 and `M_U 0 = 0`, so the gate action on E is irrelevant there. [VA]
   - (c) A space-time decoder should define detection events as
     `s_{j+1} − M_{C_{j+1}} s_j (mod 3)` between consecutive used rounds.
     Across the unchecked inverse block it should use
     `s_m − M_{W_m^{-1}} s_{m−1}`. Raw-outcome differences are wrong
     whenever a logical H or P sits between the rounds. [VA]
3. **Size of the environment.** Four Z_3 checks give 3^4 = 81 syndrome
   states. The `const 0` terminal X-check gives 3^2 = 9. [VA]
4. **Where the faults are.** From the brief's SI1000 description, the
   summed fault probabilities per check round are:
   - data idle: 4 measurement layers × 2p on each of the 5 data qutrits,
     plus 2p reset idle on each data qutrit: 50p;
   - idle on the 3 non-measured ancillas in each of the 4 measurement
     layers: 24p;
   - 12 two-qutrit CNOT-type gates: 12p;
   - measured-ancilla depolarizing (4 × p), measurement shifts (4 × 5p)
     and reset shifts (4 × 2p): 32p;
   - 2 ancilla H and 2 ancilla H_INV gates: 0.4p;
   - total ≈ 118p, of which 50p is data idle.
   Each logical H token carries p (SWAP) + 5 × 0.1p and each P token carries
   p (CZ) + 3 × 0.1p; X and Z tokens carry 0.2p. Check rounds therefore
   dominate the fault budget by roughly an order of magnitude per Clifford.
   [VA] (arithmetic on the brief's model; token counts per catalog Clifford
   not checked).

### 8.2 When is the accepted logical process Markovian?

- With ideal syndrome readout and checks after every step, the accepted
  branch returns to the code space every step. Under Pauli noise the
  acceptance is independent of the logical state, so the normalized per-step
  maps are CPTP logical Pauli channels `N_G`. The process is then Markovian
  (Ziyad et al. sufficient condition) and the average of per-sequence
  conditional fidelities is gate-dependent Markovian RB. Wallman's theorem
  applies to the forward half. [VA]
- In the actual circuit, three memory sources exist (Sec. 6.2): readout and
  reset masking, post-measurement data noise (about one round, second
  order), and the unchecked inverse block (m steps). [VA] Because the
  inverse block contains only gate faults (Sec. 8.1, item 4), its pair
  contribution is suppressed relative to check-round pairs. This is
  consistent with `unif 1` being exponential within error at p ≤ 1e-3
  (Sec. 8.4). [CONJ] for the explanation; [VN] for the observation.
- For `const c` and `unif k > 1`, unused rounds leave the register unreset.
  For `const c`, the unchecked windows grow as m/c, so the memory window
  grows with m. A non-exponential, accelerating conditional decay is
  expected: it combines pair (∝ m²) contributions with an acceptance that
  saturates at 1/81. [VA] for the window lengths; [CONJ] for the
  functional form.

### 8.3 Correction instead of detection: what changes for non-Markovianity

- **[VA]** A Pauli-frame decoder (Combes-style post-processing) does not
  reset the physical syndrome register; it changes only the classical
  post-processing. Whether the decoded LRB decay is exponential depends on
  how quickly the decoded (logical, residual-syndrome) chain forgets.
  Kwiatkowski et al. give such conditions for qubits. [LIT]
- **[VA]** With ideal syndrome readout, decoding every round's syndrome
  returns the decoded residual frame to the trivial sector every round. This
  removes the "unused round" memory that dominates the `const` policies. It
  cannot remove the inverse-block window, which has no syndrome information
  until the terminal round. Matching LRB theory (Combes et al.)
  would require check rounds after the inverse Cliffords as well, or the
  standard single-recovery-Clifford structure with a check after it.
- **[VA]** For d = 2, at least some single faults are ambiguous: X0 and
  X3^{-1} have the same syndrome and differ by logical X. Any decoder
  therefore turns a first-order fraction of faults into logical errors. The
  decoded per-step logical error is O(p), against O(p²) for accepted shots
  under detection. Correction trades rejection for a first-order logical
  error rate. [CONJ] for the size of the ambiguous fraction under the
  circuit-level DEM (other agents can compute it exactly).
- **[LIT]** Imperfect syndrome extraction leaves emergent logical
  non-Markovianity that is strongest in the first rounds and relaxes quickly
  (Ziyad et al.). Under Pauli noise and high-quality cycles it is
  exponentially suppressed in the number of cycles (Kwiatkowski et al.). QEC
  can turn physically non-Markovian noise into an exponential logical decay
  (Gravier et al., d = 3, no readout errors).
- **[CONJ]** Net expectation for this study: decoding every round will make
  the decoded `unif 1`-type LRB decay closer to a single exponential than
  any `const` detection policy. It will not beat `unif 1` detection on
  conditional fidelity at low p, because of the d = 2 first-order term. The
  residual non-exponentiality will be dominated by the inverse block and by
  readout errors.

### 8.4 Baseline data checks (read-only; scripts in Appendix A)

Fit of `ln F(m) = a + b m + c m²`, weighted by
`std over sequences / sqrt(30)`, using points with F > 0.05 at depths
{0, 2, 4, 6, 10, 14, 18, 20, 22}:

| p | policy | c (m² coeff.) | χ² quadratic | χ² linear | Reading |
|---|---|---|---|---|---|
| 5e-4 | unif 1 | +2.5e-7 ± 1.4e-7 | 12.3 | 15.3 | exponential within error |
| 5e-4 | const 1 | −9.52e-6 ± 1.8e-7 | 19.2 | 2808 | strongly non-exponential |
| 5e-4 | const 0 | −5.68e-7 ± 1.5e-8 | 7.1 | 1438 | ln F ∝ −m² |
| 1e-3 | unif 1 | +3.9e-7 ± 2.8e-7 | 12.6 | 14.5 | exponential within error |
| 1e-3 | unif 2 | +3.2e-7 ± 2.7e-7 | 7.5 | 8.9 | exponential within error |
| 1e-3 | const 1 | −5.27e-5 ± 5.1e-7 | 197 | 11000 | accelerating decay |
| 1e-3 | const 2 | −1.83e-5 ± 3.5e-7 | 31 | 2829 | accelerating decay |
| 1e-3 | const 0 | −2.27e-6 ± 5.2e-8 | 3.2 | 1892 | ln F ∝ −m² |
| 2e-3 | unif 1 | +2.0e-6 ± 7.9e-7 | 11.0 | 17.2 | marginal positive curvature |
| 2e-3 | unif 2 | −1.0e-7 ± 6.6e-7 | 4.6 | 4.6 | exponential |
| 2e-3 | const 1 | −3.65e-4 ± 2.0e-6 | 2464 | 34820 | accelerating decay |
| 2e-3 | const 0 | −9.77e-6 ± 2.1e-7 | 5.5 | 2223 | ln F ∝ −m² |
| 5e-3 | unif 1 | +5.4e-5 ± 1.1e-5 (n = 8) | 24.4 | 50.0 | significant positive curvature |
| 5e-3 | const 0 | −7.27e-5 ± 1.6e-6 | 15.9 | 2149 | ln F ∝ −m² |

[VN] for every entry. Readings:

- **`const 0` is the cleanest emergent-memory example.**
  - Physically it has no checks at all: Markovian SI1000 noise on the
    Clifford gates only, ideal X-basis readout of all data qutrits, and
    postselection on the 2 terminal X-checks. [VA]
  - Yet `1 − F ≈ κ p² m²`, with `c/p²` = 2.27, 2.27, 2.44 and 2.91 at
    p = 5e-4, 1e-3, 2e-3 and 5e-3. [VN]
  - A Markovian per-step logical channel with O(p²) per-step error would
    give 1 − F ∝ m. The m² scaling shows that the dominant logical failures
    are pairs of faults in different steps, correlated through the stored
    syndrome. This is non-Markovian in the gate-composability sense. [VA]
    for "Markovian implies linear at small m"; [CONJ] for the pair-mechanism
    attribution (strongly suggested by the p² m² scaling).
- **`const 1` at p = 1e-3.** The per-interval log-slope grows from −1.8e-4
  (m 0→2) to −2.8e-3 (m 20→22), a 16-fold increase. **`const 0`** goes from
  −4.1e-6 to −1.04e-4, a 26-fold increase. **`unif 1`** stays near −3.7e-5,
  fluctuating with the statistical noise. [VN]
- **Acceptance saturation.**
  - `const 1` at m = 22 and p ∈ {0.01, …, 0.05} accepts 0.01233–0.01237,
    against 1/81 = 0.012346. `const 0` at p = 0.05 accepts 0.1113, against
    1/9 = 0.1111. [VN]
  - Explanation: when the syndrome register is uniformly randomized,
    `P(s = 0) = 3^{-#checks}`, whatever the readout noise (convolution with
    a uniform distribution stays uniform). [VA]
- **Estimator floor.**
  - The per-circuit metric `|Σ_k ω^k P(k)|` is biased upward when the true
    value is near 0. For N accepted shots with a uniform logical outcome,
    the sample mean of ω^k is complex Gaussian with covariance
    (1/(2N)) I₂, so `E|·| = sqrt(π/(4N)) ≈ 0.886/sqrt(N)`. [VA]
  - Observed `const 1` fidelities at m = 22, p ≥ 0.01: 0.0070–0.0094,
    against 0.0080 predicted for N = 10⁶/81. [VN]
  - Zero-offset fits must therefore exclude points near the floor, or
    subtract it. [VA]
- **Sequence-to-sequence spread versus shot noise.**
  - The finite-shot std of the per-circuit estimate is about
    `sqrt(1.5 (1−F)/N_acc)` (variance of Re ω^k when P(1) = P(2)). [VA]
  - Excess spread over shot noise:
    - `unif 1`, `unif 2`: ≤ 0.03 (1−F) at p = 1e-3 and 2e-3;
    - `const 1`, `const 2`: 0.01–0.05 (1−F);
    - `const 0`: 0.60 (m = 2) falling to 0.16 (m = 22) of (1−F) at
      p = 1e-3, similar at 2e-3. [VN]
  - So `const 0` failures depend strongly on the Clifford sequence. This is
    the Ball/Mavadia signature of sequence-dependent error accumulation,
    here arising from syndrome-register memory, not from temporally
    correlated physical noise. [CONJ] for the mechanism.
- **Upticks are protocol artifacts.**
  - `unif 5` and `const 5` show increases of F with m at the same intervals
    for every p examined: m 4→6 and 18→20 (at p = 1e-3: `unif 5` +2.5e-5
    and +9.8e-5; `const 5` +1.35e-4 and +1.65e-4; larger at 2e-3 and 5e-3).
    [VN]
  - From the brief's policy definitions [VA]:
    - `unif 5` gains a used round at m = 6 and m = 20;
    - `const 5` at m = 6 uses rounds 0–3, with only rounds 4–5 unused;
    - `const 5` at m = 18 uses rounds 2, 5, 8 and 11, leaving rounds 12–17
      (six) unchecked; at m = 20 it uses 3, 7, 11 and 15, leaving 16–19
      (four) unchecked;
    - the largest decays occur where the unchecked tail lengthens (m 10→14:
      rounds 8–13 unused).
  - These upticks track the length of the unchecked window. They are not a
    Wallman/Figueroa-Romero witness, because the protocol differs between
    m values. [VA]
- **`unif 1` at p = 5e-3.**
  - Positive curvature, driven by a steeper first interval: −1.55e-3 for
    m 0→2, against about −0.8e-3 afterwards. [VN]
  - Candidate causes: a gate-dependence or initial-memory transient (Wallman
    2018; Ziyad et al.), or selection effects at very high rejection. [CONJ]

### 8.5 Recommendations for the other agents (from this literature)

1. Use **stationary check patterns** for any RB-style fit: `unif 1`, or
   `unif k` only at m ∈ kℕ. `const c` curves should be analysed as
   finite-memory processes (Figueroa-Romero et al. corollaries), not fitted
   with `b f^m`. [VA]
2. Add **pooled linear estimators** alongside the per-circuit average:
   - the complex character `χ(m) = (1/30) Σ_seq Σ_shots ω^k / N_shots`
     (unnormalized, linear in the maps);
   - the acceptance `a(m)`.
   Fit each to (matrix) exponentials, so that Wallman/Merkel theory applies
   directly. The conditional rate is the ratio of the leading eigenvalues,
   `p/t`. [VA]
3. **Drop m = 0** from decay fits, or fit with a free offset (Wallman
   2018's advice on the shortest lengths). Exclude points near the
   0.886/sqrt(N_acc) floor. [VA]
4. **Report per-sequence distributions** after removing shot noise
   (Ball et al.; Mavadia et al.). The excess spread is a cheap memory
   diagnostic (Sec. 8.4). [LIT] [VN]
5. Non-Markovianity witnesses that are cheap in the DEM simulator [CONJ]:
   - compare the one-step and two-step logical Pauli channels for fixed
     Cliffords (Ziyad composability);
   - a logical-frame causal break (Pollock et al.);
   - fixed-identity Cliffords (Figueroa-Romero et al.).
6. For correction decoders: define detection events through the
   logical-Clifford syndrome maps `M_U` (Sec. 8.1, item 2c), and treat the
   inverse block as a single unchecked window. Expect a first-order decoded
   logical error from d = 2 ties. To restore Markovian LRB, add checks after
   the inverse Cliffords (Combes et al. protocol). [VA] [CONJ]
7. Hardware (question 3) [LIT] [CONJ]:
   - Coherent errors make d = 2 postselected logical noise nearly unitary
     (Beale et al. scaling) and can make acceptance logical-state dependent
     (Huang et al., even distance).
   - Correlated ancilla and readout ("streaky") noise is the most damaging
     class for correction (Kam et al.).
   - Syndrome randomization can enforce SMIP-type conditions (Kwiatkowski
     et al.).
   - Pauli twirling reduces variance but does not Markovianize the ASF
     (Figueroa-Romero et al. 2024).

---

## Appendix A. Scripts used for the [VN] claims

Run with `module load python/3.11; python3 -I <script> <run_folder>`. The
scripts only read CSVs from the run folder. Session copies:
`<local scratch, not kept>`
and `.../scratchpad/scripts/seqvar_check.py`. Core logic:

```python
# shape_check.py (core): weighted fit of ln F = a + b m + c m^2
rows = dict((r[0], [float(x) for x in r[1:]]) for r in csv.reader(open(f"{run}/results/LRB/{pidx}/{kind}_check_data/{k}.csv")))
F, sF, rej = map(np.array, (rows["Fidelity averages"], rows["Fidelity Standard Deviations"], rows["Rejected Runs"]))
sel = np.isfinite(F) & (F > 0.05)
y, sy = np.log(F[sel]), sF[sel] / np.sqrt(30) / F[sel]
A = np.vstack([np.ones_like(m), m, m**2]).T[sel] / sy[:, None]
coef, *_ = np.linalg.lstsq(A, y / sy, rcond=None); cov = np.linalg.inv(A.T @ A)
# acceptance saturation: 1 - rej[-1] vs 1/81 (const 1) and 1/9 (const 0)
# estimator floor: 0.5*sqrt(pi)/sqrt(1e6*(1-rej[-1]))

# seqvar_check.py (core): excess spread over finite-shot noise
sshot = np.sqrt(1.5 * (1 - F) / (1e6 * (1 - rej)))
excess = np.sqrt(np.maximum(0, sF**2 - sshot**2)) / (1 - F)
```

## Appendix B. Fetch log and discrepancies

- All arXiv abstract pages were fetched with WebFetch, and TeX sources with
  `curl https://arxiv.org/e-print/<id>` into isolated scratch folders.
  Publisher pages were fetched for Wallman 2018, Merkel et al. 2021 and
  Figueroa-Romero et al. 2022 (quantum-journal.org), and for
  Figueroa-Romero et al. 2021 and Beale et al. 2018 (journals.aps.org).
- Title differences between arXiv listings and journals:
  - Beale et al.: arXiv "Coherence in quantum error-correcting codes",
    PRL "Quantum Error Correction Decoheres Noise".
  - Wallman, Barnhill, Emerson: arXiv listing "Characterization of Leakage
    Errors via Randomized Benchmarking", TeX v2 "Robust Characterization of
    Leakage Errors".
  - Ball et al.: "The effect of noise correlations on randomized
    benchmarking".
  - Pollock et al. PRA: arXiv spelling "characterisation".
- Figueroa-Romero et al. 2021: the WebFetch summary of the arXiv page
  omitted Robert J. Harris, but the PRX Quantum page and the TeX list him.
- No journal reference appeared on the arXiv pages of Fong and Merkel 2017,
  Combes et al. 2017, Ceasura et al. 2022, Ziyad et al. 2025, Kwiatkowski
  et al. 2025, Gravier et al. 2025, Kobayashi et al. 2024 or
  Morales-Hashemi et al. 2026.
- For Kam et al. 2025, Kobayashi et al. 2024, Morales-Hashemi et al. 2026
  and Epstein et al. 2014, only the abstract pages were read.
