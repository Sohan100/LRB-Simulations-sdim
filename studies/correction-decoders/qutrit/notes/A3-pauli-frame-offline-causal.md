# A3: Pauli-frame (software) correction, and offline versus causal decoding, for qutrit LRB-D

Agent A3. Scope: (1) when applying the decoder's correction in software (a
Pauli-frame update of the terminal readout) is exactly equivalent to applying it
physically in real time; (2) whether offline (global) decoding, which uses
future syndromes, changes what the logical RB decay measures, and which
boundary effects of this particular circuit enter the fit parameters; whether a
decoder that knows the Clifford sequence is "fair"; (3) the latency budget real-time
correction would need, and why Clifford-only LRB does not need it.

Labels used for every claim: **[analytic]** = verified analytically (derivation
given here); **[numeric]** = verified numerically (script and output quoted);
**[literature]** = from a paper fetched in this task (Section 7); **[conjecture]**.
Claim IDs `A3-...-Cn` are collected in Section 6.

Scripts (all in `notes/scripts/`, run with `module load python/3.11`, single core,
read-only access to the baseline run folder):

- `A3_pauli_frame_statevector_check.py`: dense 3^9 state-vector simulation of the
  actual baseline circuits; frame-update table; physical vs frame correction under
  SI1000 Pauli noise and under coherent noise; exact one-qutrit counterexamples.
- `A3_offline_vs_causal_toy.py`: Z_3 repetition-chain toy model; offline MAP vs
  causal committing decoders with lookahead w; three boundary schedules; exact
  transfer matrix for the w = 0 decoder.
- `A3_fault_budget.py`: per-segment sum of fault-location probabilities in the
  baseline circuits (forward Cliffords, check rounds, inverse block, final round).

---------------------------------------------------------------------------

## 0. Summary

1. Under the assumptions of the simulated model (Clifford gates only, stochastic
   Weyl/Pauli noise, Z-basis measurements after Clifford basis changes,
   unconditional resets, no feed-forward), keeping the correction in a Pauli frame
   and relabelling the terminal readout gives exactly the same joint distribution
   of all recorded outcomes as applying the same decoder's corrections physically,
   ideally and instantly, at the time they are decided. The proof is an induction
   over circuit operations (Section 1.3) **[analytic]**. On the actual baseline
   circuits (depth 2, three Clifford sequences, SI1000 noise rescaled to p = 0.02, real-time
   lookup decoder after every round), 180 of 180 shots had non-trivial syndromes,
   and the corrected records of the two protocols agreed shot by shot in all 180.
   At depth 6 they agreed in 40 of 40 **[numeric]**.
2. The equivalence fails, by amounts that are first order in the
   non-covariant part of the noise, for coherent errors (one-qutrit example:
   P(correct) 0.985 physical vs 0.941 frame), amplitude damping (0.950 vs
   0.9025 = (1-gamma)^2 at gamma = 0.05). Weyl-twirling the same channels restores exact
   equality **[numeric]**. It also fails for leakage, non-Clifford gates, and
   feed-forward that depends non-linearly on outcomes **[analytic]/[literature]**.
   For a realistic, noisy physical correction the frame protocol coincides with
   the *ideal* physical one: the frame removes the correction gates and the
   decoder wait **[analytic]/[literature]**.
3. For Clifford-only LRB with a terminal readout, the offline MAP decoder is the
   causal Bayesian filter that never commits: it needs only the posterior of the
   final state. "Use of future syndromes" only distinguishes it from decoders that
   commit irrevocably before the end **[analytic]**. For every depth, the
   offline optimum is at least as good as any causal decoder **[analytic]**. A
   finite-state causal decoder gives F(T) = sum_k c_k lambda_k^T exactly: a
   per-step rate (the leading eigenvalue) and boundary-dependent amplitudes
   **[analytic]**, confirmed by the exact transfer matrix of the toy model
   **[numeric]**. The offline decay is bracketed by the windowed rates and is
   exponential in the bulk in the toy **[numeric]**. That it is exponential in
   general is **[conjecture]** (supported by DKLP [literature]).
4. Boundary structure of this circuit **[numeric, from the circuit files]**:
   every check round carries 118.4 p of fault probability (100.4 p for the
   final round, which has no reset). Each logical Clifford carries only 3.3 to 4.9 p.
   The decay is therefore driven by the check rounds, not by the gates. The
   inverse block (m Cliffords, ~4.7 p each, no checks) is decoded from a single
   syndrome (round m); for a fixed final-round rule its corrected fidelity is a sum
   of up to 81 exponentials, and a single exponential only without correction
   **[analytic]**. The final round has no successor, and faults on wires 0 and 3 after
   their last check interaction are invisible. These set an m-independent ceiling
   on A for every decoder: at p = 1e-3 the baseline data imply F_corr(m=0) <= 0.9930
   **[numeric + analytic bound]**. Depth-dependent decision rules of a per-depth
   DEM decoder add curvature **[analytic]/[conjecture for magnitude]**.
5. Knowing the gate sequence is intrinsic to RB (to build the inverse) and to any
   correction decoder (frame propagation). It does not violate RB's
   gate-independence assumption, which concerns the noise. Gate-dependent
   decoded logical noise is covered by gate-dependent RB theory [literature]. A
   gate-agnostic local decoder is a useful lower baseline but is not required for
   fairness **[analytic argument; recommendation]**.
6. Real-time correction would need a decoder latency below about one reset layer
   (~1 us, w = 0) or about w round durations (~15 us per round for transmon
   qutrits, lookahead w). It is unnecessary here: Clifford-only, no feed-forward,
   and the frame is exact. Syndromes (4(m+1) trits per shot) can be stored and
   decoded after the run **[analytic + literature numbers]**.

---------------------------------------------------------------------------

## 1. Software (Pauli-frame) correction equals ideal real-time correction

### 1.1 Notation and the frame-update rule for this circuit

Qutrit Weyl operators: W(a,b) = X^a Z^b with X|j> = |j+1>, Z|j> = w^j |j>,
w = e^{2 pi i/3}; on n qutrits W(v) = prod_q X_q^{a_q} Z_q^{b_q},
v = (a_1, b_1, ..., a_n, b_n) in Z_3^{2n}. Global phases never matter for a frame
(it acts by conjugation), so a frame is a vector f in Z_3^{2n}. A Clifford U maps
W(v) to a phase times W(S_U v) with S_U a symplectic matrix mod 3. The qudit
stabilizer formalism for prime d is standard [literature: Gottesman 1998].

The sdim gate conventions (read from `sdim/unitary.py` and the tableau rules in
`sdim/tableau/tableau_prime.py`) were rebuilt as dense matrices. The symplectic
images below were extracted numerically by conjugating the generators
(`A3_pauli_frame_statevector_check.py --mode gates`) **[numeric]**. As maps
(a, b) of one qutrit, or (a_c, b_c, a_t, b_t) of control/target:

| gate | frame update |
|---|---|
| X, X_INV, Z, Z_INV, I | unchanged |
| H | (a, b) -> (-b, a) |
| H_INV | (a, b) -> (b, -a) |
| P | (a, b) -> (a, a + b) |
| P_INV | (a, b) -> (a, b - a) |
| CNOT c->t | (a_c, b_c, a_t, b_t) -> (a_c, b_c - b_t, a_t + a_c, b_t) |
| CNOT_INV c->t | -> (a_c, b_c + b_t, a_t - a_c, b_t) |
| CZ | -> (a_c, b_c + a_t, a_t, b_t + a_c) |
| CZ_INV | -> (a_c, b_c - a_t, a_t, b_t - a_c) |
| SWAP | exchange the two qutrits' (a, b) |
| M (Z basis) on q | frame unchanged; recorded outcome = raw + a_q |
| RESET on q | (a_q, b_q) -> (0, 0) |
| decoder correction C = W(c) | f -> f + c |

(Check: sdim's SWAP, built from its CNOT/H tableau sequence, equals the standard
SWAP to 2e-15 **[numeric]**.) The terminal readout is `H_INV 0, H_INV 3, M 0, M 3`
with logical outcome x0 + x3. If (a_q, b_q) is the frame on wire q just before
the terminal H_INV, the X-power after H_INV is b_q. The corrected logical outcome is
therefore x0 + x3 + b_0 + b_3 (mod 3). Only the Z-powers of the frame on wires 0 and 3
at the end matter **[analytic, from the table]**. A decoder that estimates an error ê
uses the frame f = -ê, so its correction is "subtract the predicted logical
shift". That is exactly how the DEM backend's logical response is used: the
compiled response of each fault on `lrb_logical` *is* the linear frame propagation
of that fault to the readout [code: `src/lrb/dem_simulation.py`].

### 1.2 Theorem

Consider a circuit of operations, each of which is (G) a Clifford unitary, (N) a
noise channel, (M) a Z-basis measurement of one qutrit whose recorded outcome may
receive an independent classical shift, (R) a reset to |0>, or (D) a decoder step
that outputs a Weyl correction C = W(c), c = D_j(h, zeta), as a (possibly
randomized, seed zeta) function of the recorded history h so far.
*Physical protocol*: C is applied as an ideal, instantaneous unitary.
*Frame protocol*: nothing is applied; a classical frame f (initially 0) is
updated by the table of Section 1.1. Every Z-measurement on qutrit q is
recorded as raw + a_q(f), and these corrected records are what the decoder sees.

**Theorem [analytic].** Suppose assumptions A1 to A7 below hold for all operations after
the first non-trivial decoder output. Then the joint distribution of the recorded
history (all corrected measurement records, decoder outputs and the final
corrected logical outcome) is the same in the two protocols. Equivalently, the
fidelity |sum_k w^k P(k)| of any decoder is the same whether its corrections
are applied in software at the end or physically in real time.

Assumptions:

- **A1 (Clifford closure).** Every unitary after the first correction is Clifford,
  so the frame stays a Weyl operator.
- **A2 (Weyl covariance of the noise).** Every noise channel N satisfies
  N(W rho W^dag) = W N(rho) W^dag for every Weyl W that can occur in the frame.
  Stochastic Weyl channels N(rho) = sum_v p(v) W(v) rho W(v)^dag satisfy this,
  including correlated multi-qutrit Weyl noise and classically time-correlated
  Weyl noise, provided the error draws do not depend on the quantum state or on
  outcomes. Conversely, a channel covariant under the full Weyl group equals its
  own Weyl twirl (each term W^dag N(W . W^dag) W of the twirl equals N), and a Weyl
  twirl is a Weyl channel. So full covariance is equivalent to being a stochastic
  Weyl channel **[analytic]**.
- **A3 (Weyl-basis measurements).** Measurements are of Weyl observables (Z after
  Clifford basis changes), so W(f)^dag Pi_k W(f) = Pi_{k - a_q}. Classical readout
  errors are state-independent additive shifts, or Weyl operators before the
  measurement, as SI1000's `N1 f` is.
- **A4 (Weyl-eigenstate preparation and reset).** Resets prepare |0>; any reset
  noise is Weyl (SI1000: X^a after RESET).
- **A5 (ideal physical correction).** The physical correction adds no gate noise
  and no waiting time. Without A5 the frame protocol still equals the ideal physical
  protocol, and the comparison with a realistic one differs by exactly the
  correction's own noise and delay.
- **A6 (no other feed-forward).** No operation other than the decoder's Weyl
  corrections is conditioned on outcomes. Allowed exceptions are a Weyl operation
  whose exponent is a linear function of outcomes (absorbed into the frame) and
  a reset-type operation whose result does not depend on the input state (for
  example, active reset X^{-k} after reading k).
- **A7 (no leakage, no quantum memory).** The dynamics stay in the qutrit space on
  which W(v) is defined, and the noise has no environment memory that couples to
  the system state. Classical correlations in the Weyl error process are allowed
  (A2).

### 1.3 Proof [analytic]

Index the unnormalized conditional state by the recorded history h: rho_h
(physical) and sigma_h (frame protocol, conditioned on the *corrected* record h;
the raw-to-corrected map is a bijection because the frame is a function of h).
The claim is the invariant

    rho_h = W(f_h) sigma_h W(f_h)^dag      for every prefix and every h,

where f_h is the frame after history h. Since W(f) is unitary, tr rho_h = tr sigma_h,
so P_phys(h) = P_frame(h). That is the theorem. Induction over operations:

- (G) Clifford U: rho' = U rho U^dag = U W(f) sigma W(f)^dag U^dag
  = W(S_U f) (U sigma U^dag) W(S_U f)^dag, because U W(f) U^dag = phase x W(S_U f)
  and the phase cancels. The frame update is f' = S_U f.
- (N) channel N: rho' = N(W sigma W^dag) = W N(sigma) W^dag by A2.
- (M) Z-measurement of qutrit q with outcome k:
  Pi_k rho Pi_k = W (W^dag Pi_k W) sigma (W^dag Pi_k W) W^dag
  = W Pi_{k-a_q} sigma Pi_{k-a_q} W^dag. The physical branch "outcome k" is the
  frame branch "raw outcome k - a_q", which is recorded as corrected k. An
  independent classical shift xi is added identically in both protocols.
- (R) reset of q: R(W_rest W_q tau W_q^dag W_rest^dag) = W_rest R(tau) W_rest^dag, since the
  output of a reset does not depend on its input. So the frame on q is set to 0.
- (D) correction C = W(c): rho' = C W(f) sigma W(f)^dag C^dag = W(f + c) sigma W(f + c)^dag
  up to phase. The value c = D_j(h, zeta) is the same in both protocols because the
  histories are equal by the induction hypothesis (same seed zeta).

The final logical readout is a sequence of (G) and (M) steps. QED.

The proof shows more than equality in distribution. With common random numbers
(same Weyl error draws; same uniform variate per measurement used to sample the
*corrected* outcome), the two protocols produce identical records shot by shot.
This is the version that was checked numerically.

### 1.4 Numerical verification on the actual circuits [numeric]

`A3_pauli_frame_statevector_check.py --mode all --shots 60 --coh-shots 150 --theta 0.1 --p 0.02`
(57 s on one core). It parses `experiments/LRB/{0,1,2}/8/1.chp` (depth m = 2) and
`.../0/8/0.chp` (m = 0) from the baseline run, read-only, and simulates them with
a dense 19683-dimensional state vector. It does not use sdim's tableau or the DEM.

- Noiseless circuits: every ancilla and reset outcome is deterministic 0, and
  P(x0 + x3 = 0) = 1 in all four circuits. This validates the gate conventions
  and the check circuits.
- Pauli mode: the circuit's own SI1000 locations, with probabilities rescaled to
  p = 0.02, give 180 shots over 3 circuits. After every check round a lookup decoder
  (weight-1 data error per 4-trit round syndrome, built by frame propagation
  through the check circuit) produces a correction. Result: **180/180 shots had a
  non-trivial syndrome; 0 shots had any difference between the physical and the
  frame-corrected records** (14 measurement records plus the logical outcome per shot).
  The logical fidelity with this crude decoder at p = 0.02 is 0.075, so the
  frames were far from trivial.
- Depth m = 6 (`--mode pauli --depth-idx 3 --shots 20 --cliffs 0,1 --p 0.01`,
  15 s): 40/40 shots with non-trivial syndromes, 0 shots with any difference.

### 1.5 What breaks the equivalence

**B1. Coherent (non-Weyl) errors [analytic + numeric].** If N(rho) = V rho V^dag
with V not proportional to a Weyl operator, A2 fails. In the frame picture, the
physical protocol has noise W(f)^dag V W(f) after a correction, while the frame
protocol has V. Coherent errors therefore add up differently. Exact one-qutrit
example (`--mode minimal`): input |+>, error E = X^{-1}, then V, then correction
C = X (physical) or frame, then V, then X-basis readout. With
V = diag(1, e^{0.15 i}, e^{0.30 i}), P(corrected outcome = 0) = 0.98508
(physical) vs 0.94134 (frame). The physical correction acts like an echo here. For a
generic random unitary near identity (same angle scale) it is 0.99596 vs 0.98590.
Two controls give exact equality (difference below 1e-16): replacing each channel by its Weyl twirl, and the
commuting case (diagonal V with Z-type frame, E = Z^{-1}, C = Z). So covariance is needed
only for the frames that actually occur. In this circuit the logical H swaps X- and
Z-type frames, so no such restriction applies.
*Circuit level*: replacing every data-qutrit `N1 d` location of the m = 2 circuits
by V = diag(1, e^{0.1 i}, e^{0.2 i}) and keeping the measurement and reset flips
stochastic, the per-shot exact logical distributions of the two protocols differ in 67% of
150 shots. The Clifford- and shot-averaged difference of Re E[w^k] was
-0.005 +- 0.024, which is **not resolved**. **[conjecture]**: the random logical Cliffords and the
randomness of the frame partially twirl the coherent error, so the averaged LRB
fidelity difference is small compared with the per-shot difference. Knill notes
the same mechanism from the other side: consecutive coherent errors can add
coherently, and "applying known but random Pauli products, thus modulating the
Pauli frame" reduces this **[literature: Knill 2005]**.

**B2. Amplitude damping (T1) [analytic + numeric].** The ladder channel |2>->|1>->|0>
with Kraus operators diag(1, sqrt(1-g), sqrt(1-2g)), sqrt(g)|0><1|, sqrt(2g)|1><2| is
not Weyl-covariant. Example: a known X error takes |0> to |1>, then damping, then the
correction, then damping, then Z readout. P(correct) = 1 - g physically (the
corrected qutrit sits in |0>, which does not decay) vs (1-g)^2 with the frame (the
uncorrected qutrit stays excited), i.e. 0.950 vs 0.9025 at g = 0.05. The Weyl-twirled
damping channel gives identical results (0.9025 both) **[numeric]**. Which
protocol does better is example-dependent, since frames permute which levels are
populated. In an encoded state many levels are populated anyway.

**B3. Leakage [analytic].** Weyl operators and the frame-update table are defined
only on the 3-level space. Population in |3> (transmon) is not tracked by any
frame. Leaked qutrits produce non-Weyl, correlated errors on their neighbours
through two-qutrit gates. Heralded leakage handling, such as leakage-reduction units
or resets conditioned on a 4-outcome readout, is an operation conditioned on
outcomes and is not Weyl (A6, A7). Knill flags undetected leakage as
problematic and notes that teleportation-based correction controls it at each
step **[literature: Knill 2005]**.

**B4. Non-Clifford gates [analytic + literature].** If U is not Clifford,
U W U^dag is not a Weyl operator, so the frame cannot be propagated through it. DiVincenzo and
Aliferis: "all information determining the current Pauli frame must be known
before the application of a non-Clifford gate", and for state-injected T gates
the conditional S correction requires the measured value including "all the
preceding Pauli-frame information that determines its meaning"
**[literature: DiVincenzo & Aliferis 2007]**. Terhal makes the same distinction:
for circuits that are entirely Clifford, "the Pauli frame simply alters the
interpretation of the final measurement outcome", unlike circuits with
non-Clifford gates **[literature: Terhal 2015]**. Chamberland, Iyer and Poulin
handle slow diagnostics before T gates with a buffer of error-correction rounds and a
Clifford frame **[literature: Chamberland et al. 2018]**.

**B5. Operations conditioned on outcomes [analytic].** A Weyl operation whose
exponent is linear in recorded outcomes is absorbed into the frame, as Knill does
for teleportation corrections ("they can be absorbed into the Pauli frame and do not need to be known immediately")
**[literature: Knill 2005]**. Two other kinds need the frame in real time: a
non-Weyl conditioned operation (e.g. conditional S after T teleportation), and
one whose condition is a non-linear function of decoded information (adaptive
circuits, repeat-until-success, real-time early abort). Active ancilla reset
conditioned on the *raw* ancilla outcome is harmless because the reset erases the
frame on that ancilla. This was checked algebraically in Section 1.3 and the
simulator uses it.

**B6. Noisy or delayed physical corrections [analytic + literature].** Real
physical corrections are extra gates (SI1000 would add 0.1 p depolarizing per
single-qutrit gate) and may require waiting for the decoder (idle noise). The
frame protocol is equivalent to the *ideal* physical protocol, so it avoids both.
Chamberland et al.: tracking the correction in software "saves us from performing
additional gates on the system, thus removing potential sources of errors"
**[literature]**. Riesebos et al. report, in simulations of a surface-code-17
(SC17, distance-3) logical qubit, up to 70% lower logical error rate with a Pauli
frame than without one when decoding time equals error-correction time and
parallelism is maximal **[literature, abstract only; the abstract does not state
the mechanism]**.

**B7. State-dependent or quantum-memory noise [analytic].** Examples are a bath that couples
coherently to the system and retains memory, and crosstalk whose phase
depends on the computational state (e.g. ZZ-type). Both violate A2 or A7. Classical, time-correlated
Weyl noise (bursts, drifts in a stochastic Weyl rate) does not. This matters for
the study's question 2: frame equivalence survives classical non-Markovianity
of Weyl type, but not quantum non-Markovianity in general.

### 1.6 Consequences for this repository

- The simulated model satisfies A1 to A7 exactly. SI1000 consists of Weyl channels,
  `N1 f` is an X^a before M, resets are unconditional, there is no feed-forward,
  and all gates are qutrit Cliffords. A DEM decoder that outputs a
  logical shift for the terminal readout therefore gives *exactly* the same output
  distribution as an ideal real-time correcting implementation of the same
  decoder **[analytic + numeric]**.
- On hardware (transmon qutrits: T1 decay, coherent miscalibration, leakage) the
  two differ at first order in the non-Weyl part of the noise. A
  sustainable approach is Weyl twirling (randomized compiling) of each layer: the inserted random
  Weyl operators are themselves tracked in the frame, so they cost no extra
  physical undo gates, and the effective noise becomes a Weyl channel, for which the
  frame is exact **[analytic]**. Leakage needs separate handling (B3).

---------------------------------------------------------------------------

## 2. Offline (global) versus causal decoding

### 2.1 Definitions and a basic observation

A shot produces a record s (all detectors) and a logical shift L (the error's
effect on x0 + x3). A decoder outputs L_hat(s), and the residual is R = L - L_hat.
The brief's per-circuit metric is F = |E[w^R]|.

- *Offline (global) decoder*: any function of the complete record. The
  fidelity-optimal one maximizes |E w^{L - L_hat}| over all such functions.
- *Causal committing decoder with lookahead w*: the correction attributed to
  step t is fixed irrevocably using records up to round t + w. Later decisions
  take earlier commits as certain, as in a real-time decoder that discards data or
  acts on its commits.

**Observation [analytic].** For a Clifford-only circuit with a terminal readout,
only the posterior of the final logical class, P(L = k | s), matters. In a
hidden-Markov description (hidden state = current error class, observations =
rounds), this posterior is the *forward filter* at the final time. The offline
MAP decoder is therefore the causal Bayesian filter that never commits. "Using
future syndromes" means only this: the decision about an error at step t is
deferred until later rounds have been seen. The Pauli frame (Section 1) makes
deferral free in a Clifford-only circuit. The offline and causal decoders differ
only when a decision must be committed before the end: feed-forward,
non-Clifford gates, or bounded decoder memory. The toy code implements "offline" literally
as the forward filter.

### 2.2 The metric for qutrits [analytic]

Every SI1000 location has an error distribution invariant under v -> -v: `N1 d`
is uniform on the 8 non-zero (a,b), `N1 f` uniform on a in {1,2}, `N2` uniform on
the 80 non-zero vectors. Detector and logical responses are linear in v, so the
joint law of (s, L) is invariant under (s, L) -> (-s, -L). For any decoder with
L_hat(-s) = -L_hat(s) (MAP with symmetric tie-breaking), P(R=1) = P(R=2), and then

    E[w^R] = P(R=0) - (P(R=1)+P(R=2))/2 = (3 P_succ - 1)/2   (real).

Pointwise, max_k Re(w^{-k} E[w^L | s]) = max_k (3 P(L=k|s) - 1)/2, so for d = 3
the fidelity-optimal decision is the MAP class. The brief's metric is then a monotone
function of the success probability, and the offline MAP decoder maximizes it.

### 2.3 Offline dominates causal at every depth [analytic]

A causal decoder's final output is a (randomized) function of the complete
record, so it is one of the functions the offline optimum maximizes over. Hence
F_off(m) >= F_causal(m) for every m, every lookahead w and every commit rule, and
the same holds for any per-step decay rate extracted from them. Here F_off is
the fidelity-optimal offline decoder. Under 2.2 it is the MAP decoder with
symmetric tie-breaking. P_succ(MAP) >= P_succ(causal) holds without any
symmetry assumption. One caveat: a decoder with *asymmetric* tie-breaking can
have |E w^R| above (3 P_succ - 1)/2 through an imaginary part. The toy therefore
breaks ties at random.

### 2.4 Causal finite-state decoders: exact sum of exponentials [analytic]

Suppose the decoder has finite memory: its state after round t (pending window,
committed frame modulo what affects the future) is a function of its previous
state and the new round. Suppose also that the faults in different steps are
independent, with the same law in the bulk. Then (residual error class, decoder
state) is a finite Markov chain with transition matrix M. Because neither the
noise law nor the decoder depends on the logical class (it is invisible to the
syndrome), M commutes with logical shifts. Restricted to the sector that carries
the character w^R it gives a matrix M_1, and

    F(T) = u^T M_1^T v = sum_k c_k lambda_k^T,     c_k = (u . r_k)(l_k . v),

where r_k and l_k are right and left eigenvectors. The leading eigenvalue lambda_1 is the per-step
decay. For small noise it is real, positive and isolated, and the per-step logical
failure probability is (2/3)(1 - lambda_1) under 2.2. The boundaries enter *only*
through u (the initial state, e.g. ideal encoding) and v (the end: last round,
readout), and through segments with different matrices (e.g. the unchecked
inverse block). They set the amplitudes c_k, not lambda_1. The subleading
|lambda_k| < lambda_1 give transients ("curvature") that decay on the decoder's
correlation time. In the toy model with w = 0 the subleading eigenvalues are
-0.0075 and 0.0022, so the transients vanish after one step (Section 2.8)
**[numeric]**.

### 2.5 The offline decoder [analytic bounds, conjecture for the limit]

The offline MAP decision is a non-linear function of the whole record, so no
finite transfer matrix exists. Three statements:

1. Lower bound **[analytic]**: F_off(T) >= max_w F_causal,w(T) = max_w (c_w lambda_w^T (1+o(1))).
2. Upper bound **[analytic]**: with a_T(s) = E[w^L ; record = s] =
   u^T A(s_T)...A(s_1) v (tilted transfer matrices),
   F_off(T) <= sum_s |a_T(s)| <= ||u|| ||v|| N_T, where
   N_T = sum_{s_1..s_T} ||A(s_T)...A(s_1)|| is submultiplicative for
   time-homogeneous bulk steps. So lim (1/T) log N_T exists (Fekete) and
   limsup (1/T) log F_off(T) is at most that limit. The
   matching lower bound F_off(T) >= (1/2) sum_s |a_T(s)| also holds: the MAP choice achieves
   max_k Re(w^{-k} a(s)) >= |a(s)|/2 for each s.
3. **[conjecture]**, supported by the toy model and by DKLP: windowed decoders
   converge to the offline optimum as the window grows (lambda_w -> lambda_off).
   Consequently F_off(T) = A_off lambda_off^T (1 + o(1)): exponential in the bulk,
   with all boundary effects in A_off. DKLP's overlapping-recovery decoder, which commits only
   long-lived defects and keeps recent ones for the next step, "will perform very
   nearly as well as if an indefinite amount of information were stored" when the
   window is long compared with the correlation time **[literature: Dennis et al.
   2002]**. Parallel/sliding window decoders show no noticeable fidelity loss for the
   surface code **[literature: Skoric et al., abstract]**.

### 2.6 Does offline decoding change what the RB decay measures?

- In the bulk, the offline-decoded LRB decay rate is the per-step logical error
  rate of the best decoder with unlimited lookahead. That is also the rate of a
  real-time sliding-window decoder with lookahead longer than the correlation time
  (a few rounds) **[analytic (2.3, 2.4) + conjecture (2.5.3)]**. Offline decoding
  thus measures a *real-time-achievable* figure of merit for Clifford-only
  operation. It does not use any information that a real-time decoder with
  memory would lack. It only defers decisions.
- It is *not* the right number for a computation that must commit decisions at
  feed-forward points (T gates). There the relevant rate is that of a committing
  decoder with the available latency. In the toy (distance 3) committing with w = 0
  costs a factor 1.5 in the per-step logical error, w = 1 a factor 1.13 and w = 2
  a factor 1.07 relative to offline (Section 2.8) **[numeric, toy]**.
- The residual logical errors of any decoder are correlated in time over the
  decoder's correlation length. An example is a measurement flip that is first
  mis-committed and later undone. In RB language this is a mild, decoder-induced
  non-Markovianity of the logical noise. By 2.4 it produces transients and
  boundary amplitudes, not a change of the asymptotic rate **[analytic]**. Standard
  RB theory already tolerates weak gate and time dependence
  **[literature: Magesan et al. 2012]**. For gate-dependent noise the deviation
  from a single exponential is one perturbation term that itself decays
  exponentially with sequence length **[literature: Wallman 2018]**.

### 2.7 Boundary effects specific to this circuit

**Fault budget [numeric]** (`A3_fault_budget.py`, Clifford sequence 0, p = 0.01;
expected number of activated fault locations, in units of p):

| segment | total | data-qutrit | ancilla-only | meas. flips | reset flips |
|---|---|---|---|---|---|
| check round 0..m-1 (with reset layer) | 118.4 | 62.0 | 28.4 | 20.0 | 8.0 |
| final round m (no reset) | 100.4 | 52.0 | 28.4 | 20.0 | 0 |
| one logical Clifford (forward or inverse) | 3.3 to 4.9 | all | 0 | 0 | 0 |

At m = 22: forward Cliffords 103 p, rounds 0..21 2605 p, inverse block 103 p,
final round 100 p. Sequence 5 gives 107 p for the Cliffords. "Data-qutrit" counts every location that
touches a data wire (including the 12 data-ancilla CNOTs). Two consequences
**[analytic, from the table]**. (i) The per-step decay is dominated by the noise
*of the check rounds themselves* (about 30 times the Clifford noise). (ii) The
unchecked inverse block carries about 1/25 of the per-step fault mass of the forward
block. Its decoding handicap (no time resolution) therefore matters less for
the rate than its structure suggests.

Walking through the circuit:

(a) **Round 0 after ideal encoding.** The initial state is a perfect codeword, so
the first step has no earlier measurement noise to confuse. In 2.4 this is u =
(clean state) instead of the bulk's stationary mix. It raises A slightly above
the bulk extrapolation, by O(p) **[analytic]**.

(b) **The forward bulk.** Each step is one Clifford (3 to 5 p) plus one round (118 p).
Measurement flips (5 p `N1 f` plus the X-part of the p depolarizing on the measured
ancilla, about 5.75 p per ancilla) are confusable with data faults unless the next
round is used. That is where lookahead pays: a flip changes one round, while a data
fault persists, transformed by the next Clifford, into later rounds. For this
distance-2 code, *pure* X-type or *pure* Z-type single-qutrit faults on wires 0,
2, 3, 4 have weight-1 partners in another logical class with the same syndrome:
X0 ~ X3^{-1}, X2 ~ X4^{-1}, Z0 ~ Z2^{-1}, Z3 ~ Z4^{-1}. Two of these come from
X2 X4 = S_X1^{-1} S_X2^{-1} X_L and Z3 Z4 = (S_Z1 S_Z2)^{-1} Z_L. Mixed faults X^a Z^b
(a, b != 0) and all faults on wire 1 have only weight >= 2 partners
**[analytic]**. For ambiguous faults the MAP decision is right only as far as the
priors differ, and the priors are nearly equal because idle noise dominates (2.7
table). The offline-vs-causal gap therefore mostly concerns measurement flips.
A first-order logical error per round from ambiguous data faults remains for every
decoder **[analytic; magnitude conjecture, Section 5]**.

(c) **The unchecked inverse block (m Cliffords, ~4.7 p each).** Its faults are
seen only through the single round-m syndrome, so no decoder, offline or causal,
has time resolution there. Write the accumulated block error as E (independent
faults E_j, propagated to round m), the round-m syndrome as s(E) =
(<S_i, E>)_i, and the readout shift as l(E) = <X_L, E> (symplectic products). Take a
final-round rule g(s), and Fourier-expand h(s) = w^{-g(s)} = sum_t h^(t) w^{t.s}
with h^(t) = 3^{-4} sum_s w^{-g(s) - t.s}. Then t.s(E) = <S^t, E> with
S^t = prod_i S_i^{t_i}, and

    E[w^{l(E) - g(s(E))}] = sum_{t in Z_3^4} h^(t) prod_{j=1}^m lambda_j(X_L S^t),

where lambda_j(Q) = E[w^{<Q, E_j>}] is the Weyl-channel eigenvalue of step j on Q **[analytic]**.
Consequences: no correction (g = 0) gives a single product, i.e. one exponential
for time-homogeneous steps. Postselection on s = 0 gives the ratio
[sum_t prod_j lambda_j(X_L S^t)] / [sum_t prod_j lambda_j(S^t)]. A correcting rule
gives a mixture of up to 81 exponentials, one per "dressed" logical representative X_L S^t, with
weights h^(t) set by the decoder. The block contributes to the decay *rate*,
because its fault mass grows with m. With correction it generally also contributes
curvature. A DEM/MAP decoder rebuilt for each depth also changes g with m,
because the prior of "inverse-block fault" vs "round-m measurement flip" grows with m.
This makes the m-dependence non-exponential even beyond the mixture
**[analytic; size conjecture]**. How strong the curvature is depends on the distance.
Decoded from one round, a distance-d code fails at order (fault count)^{ceil(d/2)}.
Its block contribution starts linear in m for d = 2 (here) and quadratic for d = 3.
In the d = 3 toy with an unchecked block as noisy as a checked step, the per-m
decay factor of every correcting decoder falls steadily from ~0.997 to 0.989
between m = 1 and m = 22, while without correction it stays at the exact
0.9702 (Table B in 2.8) **[numeric]**.

(d) **Round m-1 has a degraded successor.** The round that confirms or refutes a
round-(m-1) anomaly is round m, which also sees the m-Clifford inverse block and
its own 100 p of faults. Its confirming power decreases with m. This gives an
m-dependent end factor and hence curvature at second order **[analytic]**.

(e) **The final round m has no successor and the readout gives no syndrome**
(only wires 0 and 3 are read). A single flipped detector in round m has three
explanations with different logical classes: a measurement flip (most likely,
~5.75 p per ancilla), a data fault in the inverse block or in round m, or its
weight-1 partner. Faults on wires 0 and 3 *after their last check interaction* in
round m are invisible to every decoder: Z-type errors on wire 0 after `CNOT 7->0`
(including the 2p idle of the M7 and M8 layers) and on wire 3 after `CNOT 8->3`
(including the M8-layer idle).
**Bound from the baseline data [numeric + analytic]:** at m = 0
(the circuit is: ideal encoding, round m = 0, readout) the LRB-D run accepts 93.89%
of shots with F_acc = 0.992503 at p = 1e-3 (unif 1 = const 1 at m = 0). Every
sensible decoder applies no correction to an all-zero syndrome. Then
F_any(0) <= P_acc F_acc + (1 - P_acc) x 1 = 0.938916 x 0.992503 + 0.061084 = **0.99296**,
even if every detected shot were decoded perfectly. At p = 1e-2: 0.538474 x 0.917113 +
0.461526 = **0.9554**. This m-independent loss sits in A for correction just as the
same faults sit in A for detection.

(f) **How this enters the fit.** With a fixed decision rule, the structure is a
product of transfer matrices. For time-homogeneous segments

    F(m) = sum_{k,k'} c_{k k'} (lambda_k mu_{k'})^m

where the lambda_k come from the forward step (Clifford plus round) and the mu_k' from the
inverse step. The fit parameters therefore read as f = lambda_1 mu_1 and A ~ c_11
(times the invisible-fault factor), and B = 0 for this metric because the residual
tends to uniform **[analytic]**. Curvature at small m comes from three sources:
subleading terms (short-lived), the inverse-block mixture (c), and depth-dependent
decision rules (c, d) **[analytic]**. Practical consequences **[recommendation]**:
(1) report F(0) separately and fit from m >= 2, comparing with the fit including m = 0;
(2) for a pure-exponential reference, also decode with the final-round rule frozen
across depths (e.g. the rule optimal at the largest m);
(3) for the symmetric SI1000 noise use the signed Re E[w^R], which equals the
brief's |.| when positive, so that sampling noise near zero does not bias
the large-m points upward.

### 2.8 Toy-model results [numeric]

`A3_offline_vs_causal_toy.py` contains the model, the decoders and the
schedules. Three runs, on one core:
A: `--N 50000 --ws 0 1 2 --Ts 1 2 4 6 10 15 20 30` (126 s);
B: `--N 50000 --ws 0 1 2 --qu 0.01 --qpost 0.0 --schedules lrb` (58 s);
C: `--N 100000 --n 2 --ws 0 1 2 --Ts 1 2 4 6 10 15 20 30` (91 s).
The default rates mimic the SI1000 ratios of 2.7. Per checked step each trit has
data error q = 0.01 (dominated by the check round). Each check has a measurement
flip q_m = 0.006. An unchecked (Clifford-only) step has q_u = 0.0005, about 1/25
of a checked step. The readout trit has an invisible post-check error
q_post = 0.001. Ties in MAP decisions are broken uniformly at random, so 2.2
applies. Fits use T >= 6 (memory) or m >= 4 (lrb). Uncertainties are 1 sigma from
the shot count.

**Table A, distance 3 (n = 3)**: fitted per-step factor f, per-step logical
failure eps = (2/3)(1 - f), and amplitude A.

| schedule | quantity | none | offline | causal w=0 | w=1 | w=2 |
|---|---|---|---|---|---|---|
| memory-perfect | f | 0.98476 | 0.99931(2) | 0.99891(3) | 0.99921(3) | 0.99924(3) |
| | eps | 1.02e-2 | 4.61e-4 | 7.24e-4 | 5.24e-4 | 5.06e-4 |
| | A | 1.003 | 1.001 | 1.001 | 1.001 | 1.001 |
| memory-noisy | f | 0.98496 | 0.99930(3) | 0.99894(4) | 0.99922(4) | 0.99925(4) |
| | eps | 1.00e-2 | 4.64e-4 | 7.03e-4 | 5.22e-4 | 4.97e-4 |
| | A | 1.000 | 0.991 | 0.991 | 0.991 | 0.991 |
| lrb (per m) | f | 0.98418 | 0.99921(4) | 0.99882(5) | 0.99912(4) | 0.99917(4) |
| | A | 0.984 | 0.989 | 0.990 | 0.990 | 0.990 |

Exact transfer matrix for causal w = 0 on memory-noisy: lambda_1 = 0.998956 with
weight 0.99174. The next eigenvalues are -0.0075 (weight 0.0089) and 0.0022
(weight -0.0007), so F(T) = 0.99174 x 0.998956^T to better than 1e-4 for T >= 2. The
Monte Carlo fit gives f = 0.99894(4), A = 0.9913, in agreement. For 'none' the exact
f = 1 - 3q/2 = 0.985.

Observations **[numeric]**:
1. Every decoder's bulk decay is exponential: the local per-step factors are
   constant within noise for T >= 6.
2. Per-step error rates order as offline < w=2 < w=1 < w=0 << none. Relative to
   offline, w = 0 costs a factor 1.5, w = 1 a factor 1.13 and w = 2 a factor 1.07.
   One or two rounds of lookahead recover most of the offline advantage, as
   conjectured in 2.5.3.
3. Replacing the perfect final readout by a noisy last round with no successor
   lowers A from 1.001 to 0.991 for *every* decoder, offline included. That is about
   (3/2) q_m, the cost of a last-round flip that cannot be told apart from a data
   fault. The per-step factor does not change (offline 0.99931 vs 0.99930). This
   is the mechanism of 2.7(e).
4. LRB schedule with q_u << q: the per-m decay is the forward-step decay times a
   small unchecked-step factor. The points m <= 2 lie within 0.1% of the bulk line,
   except F(0) = 0.9880 vs A = 0.9892. Post-selection keeps F ~ 0.999 while
   acceptance falls to 37% at m = 22.

**Table B, unchecked block as noisy as a checked step** (q_u = q = 0.01,
q_post = 0, lrb schedule, distance 3). Local per-m decay factor
[F(m')/F(m)]^{1/(m'-m)}:

| decoder | 0->1 | 1->2 | 2->4 | 4->6 | 6->10 | 10->14 | 14->18 | 18->22 |
|---|---|---|---|---|---|---|---|---|
| none (exact 0.97023) | 0.9704 | 0.9671 | 0.9728 | 0.9710 | 0.9703 | 0.9676 | 0.9731 | 0.9686 |
| offline | 0.9967 | 0.9990 | 0.9963 | 0.9956 | 0.9947 | 0.9923 | 0.9908 | 0.9889 |
| causal w=0 | 0.9963 | 0.9993 | 0.9954 | 0.9953 | 0.9941 | 0.9919 | 0.9905 | 0.9880 |

Without correction the decay is the single exponential (1 - 3q/2)^{2m+1}. With
any correcting decoder the per-step decay *accelerates* with m. The naive fit gives
A = 1.012 > 1, which is a curvature artifact. The unchecked block is decoded from one round,
and for distance 3 its failure starts at second order in the block's fault count.
It therefore adds curvature instead of a constant rate. This is 2.7(c) in the toy's
language **[numeric]**.

**Table C, distance 2 (n = 2)**: per-step eps = (2/3)(1 - f).

| schedule | none | offline | w=0 | w=1 | w=2 | post-selection F (acceptance), last point |
|---|---|---|---|---|---|---|
| memory-perfect | 1.01e-2 | 9.40e-3 | 1.29e-2 | 1.01e-2 | 1.01e-2 | 0.997 (0.46) at T=30 |
| memory-noisy | 9.97e-3 | 9.32e-3 | 1.30e-2 | 9.89e-3 | 9.96e-3 | 0.998 (0.46) at T=30 |
| lrb | 1.05e-2 | 1.00e-2 | 1.36e-2 | 1.06e-2 | 1.05e-2 | 0.996 (0.54) at m=22 |

For distance 2 with symmetric priors, the offline decoder gains only about 7% over
no correction. The w = 0 decoder is *worse* than no correction, because it commits on
measurement flips and resolves 50/50 data ambiguities by coin flip. With w >= 1 it
equals no correction. Post-selection keeps F >= 0.996 at the cost of acceptance.
For the exact w = 0 chain the single eigenvalue is lambda = 0.980567 with weight 1.000, and the Monte Carlo
gives f = 0.98054. This is the toy version of 2.7(b) and Section 5 **[numeric]**.

### 2.9 Gate knowledge and the gate-independence of RB

- *RB needs the sequence anyway.* The experimenter computes the inverse from the
  sampled Cliffords. Combes et al.'s LRB protocol propagates the recovery through
  "each subsequent logical Clifford operation" and states that this reasoning "is
  a function of the syndromes s_j collected after the jth encoded gate and of the
  sequence i performed" **[literature: Combes et al. 2017]**. Their protocol
  performs a round of error correction after each encoded Clifford and then a
  single inverting Clifford. The present circuit instead applies m separate
  inverse Cliffords without checks, which is the origin of 2.7(c).
- *A correcting decoder needs the gates for bookkeeping.* Frame propagation
  (Section 1.1) uses S_U for every later gate. By the theorem, this bookkeeping is
  equivalent to ideal physical correction, which needs no knowledge of later
  gates. Comparing rounds across a logical gate also needs the gate: logical H
  exchanges the X- and Z-check sectors, so "syndrome change" between rounds r-1
  and r is defined only after mapping round r-1 through the gate. DKLP make the
  same point for a transversal CNOT between blocks, where the error chain of one block has a
  non-trivial boundary at the gate **[literature: Dennis et al. 2002]**.
- *What RB's gate-independence assumption is about.* It concerns the noise
  channel per gate, not the classical post-processing. A gate-aware decoder makes
  the decoded logical noise gate-dependent (Cliffords with more physical gates or
  different fault propagation decode differently). Detection does too, since compiled
  Cliffords differ in length (3.3 to 4.9 p in 2.7). Gate-dependent noise still gives an
  exponential decay up to an exponentially decaying perturbation, and the rate keeps its
  meaning as an average fidelity between ideal gates **[literature: Wallman
  2018]**. So the DEM decoder does not violate the protocol **[analytic argument]**.
- *What would be unfair.* Three things would be: a decoder that uses the measured
  logical outcomes or the ideal answer; per-sequence tuning on the benchmark's own
  outcomes; and post-selection on the logical value. The DEM decoder does none of
  these. It uses syndromes, the circuit and a fixed noise model
  [code: dem_simulation.py]. It does use the *exact* noise model, which is a genie
  advantage over hardware, where the model is estimated. A mis-specified-prior
  robustness check is recommended **[recommendation]**.
- *Is a gate-agnostic decoder needed?* Not for validity. LRB-D's rule (reject if
  any raw outcome is non-zero) is gate-agnostic because raw outcomes are
  deterministic 0 for every logical gate (verified for the noiseless circuits,
  Section 1.4). The correction analogue is a per-round lookup on the
  frame-adjusted round syndrome (equivalently, on the raw syndrome with physical
  correction). Its decision rule is gate-independent, and only the bookkeeping uses
  the gates. Reporting it next to the DEM decoder separates the
  benefit that comes from sequence-specific modelling from the benefit of correction itself.
  It is also the cheapest real-time option on hardware **[recommendation]**.

---------------------------------------------------------------------------

## 3. Latency budget and why Clifford-only LRB does not need real-time decoding

**Durations [literature + assumptions].** The compiled round is sequential: 12
ancilla-data two-qutrit gates, 4 ancilla H/H_INV, 4 separate measurement layers,
1 reset layer. Measured transmon-qutrit CZ and CZ^dag times are 783 ns and 580 ns,
and the 1<->2 transition has T1 = 28 to 63 us
**[literature: Goss et al. 2022]**. Assuming ~1 to 2 us per qutrit readout, ~1 us
reset and ~50 ns single-qutrit gates (assumptions, not from a fetched source),
T_round ~ 12 x 0.7 + 4 x 1.5 + 1 ~ 15 us and one logical Clifford ~ 1 to 3 us. A
depth-22 shot then lasts ~0.4 ms, roughly 10 T1(1->2). That is a separate warning that the SI1000
idle model may be optimistic for transmon qutrits **[conjecture]**.

**If corrections had to be physical.**
- w = 0 (correct before the next logical Clifford): the budget is the gap between
  the last measurement of a round and the next gate, about one reset layer (~1 us),
  for readout digitization, transport and decoding. A lookup over 3^4 = 81 round
  syndromes is trivially fast in an FPGA. But w = 0 commits measurement flips as
  data errors (Section 2), and every physical correction adds gate noise.
- Lookahead w: the correction for round r is applied after round r + w. The budget
  is about w x T_round (~15 us per unit of w), and the correction must be propagated
  through the intervening Cliffords (gate knowledge). Throughput: 4 trits per
  T_round.
- Reference point: Google's distance-5 surface-code memory ran a real-time decoder with
  63 us average latency at a 1.1 us cycle time, i.e. more than 50 cycles of lag
  **[literature: Acharya et al. 2025]**. Lag of that size is harmless in a memory or
  Clifford-only experiment because nothing waits for it.

**Why none of this is needed here [analytic + literature].** (1) The circuit is
Clifford-only with Weyl noise and no feed-forward, so the frame is exact
(Section 1). (2) The offline decoder is at least as good as any causal one
(2.3). (3) Storage is trivial: 4(m+1) trits per shot, 92 at m = 22, about 1e8 trits
per 1M shots. (4) Physical correction would add gate noise, and waiting would add
idle noise. Terhal: in "a quantum memory (including Clifford gates) ... the
computation never has to wait for the classical processing of the logical Pauli
frame" **[literature: Terhal 2015]**. LRB-D's post-selection needs no real-time
processing either: Knill notes that post-selected computing can proceed
"optimistically, anticipating but not waiting for measurement outcomes"
**[literature: Knill 2005]**.

**When latency becomes binding [literature].** Interleaving a non-Clifford logical
gate (magic-state injection) requires the logical frame before the conditional
Clifford correction (B4). Leakage handling with conditional resets also needs the
frame in real time. Throughput must then match syndrome generation. Otherwise a backlog
grows and the computation slows exponentially in the number of T gates (Terhal's
argument with f = r_gen/r_proc > 1). Parallel window decoders remove the
throughput bottleneck at the price of a polynomial logical-clock slowdown
**[literature: Terhal 2015; Skoric et al.]**. Chamberland et al. distinguish
measurement latency t_l from repetition time t_r. Long t_l is tolerable,
long t_r is not **[literature]**.

---------------------------------------------------------------------------

## 4. Answers in one paragraph each

- **(1)** For this experiment's model, software correction = ideal real-time
  correction, exactly and shot by shot (theorem, 180/180 numerically). The rule
  is: compose the decoder's Weyl correction into the frame, conjugate the frame by
  every later Clifford with the table in 1.1, zero it on reset, and add its
  X-power to every later Z-measurement. At the end, add b_0 + b_3 to x0 + x3. It fails for
  coherent errors, amplitude damping, leakage, non-Clifford gates and non-linear or
  non-Weyl feed-forward. Weyl twirling restores it for the first two.
- **(2)** Offline decoding defers decisions but does not see anything a
  non-committing real-time decoder would not. The bulk decay is exponential with
  the per-step rate of the decoder (exact for finite-state causal decoders,
  bracketed and conjectured for offline). Offline is better at every depth. This
  circuit's boundaries are the clean start, the unchecked inverse block decoded from one
  syndrome, the degraded successor of round m-1, and the final round without successor
  plus invisible post-check faults. They enter A, add curvature through m-dependent
  decision rules and the 81-term mixture, and leave B = 0.
  Gate knowledge is legitimate. A gate-agnostic lookup decoder is a useful extra
  baseline but is not required.
- **(3)** Physical real-time correction would need ~1 us (w = 0) or ~w x 15 us
  latency per round on transmon qutrits. For Clifford-only LRB it is unnecessary,
  and it is counter-productive because of the added gate and idle noise.

---------------------------------------------------------------------------

## 5. Implications for the other study questions [conjecture unless stated]

- For a distance-2 code, correction cannot remove first-order logical errors from
  ambiguous single faults (2.7 b). Check rounds carry about 30 times the Clifford noise (2.7,
  [numeric]). A correcting decoder that uses every round is therefore expected to have a
  per-step logical error rate of order 10 p. A rough count: 62 p data-fault mass
  per round x 4/5 (wires 0, 2, 3, 4) x 1/2 (pure X or Z type) x 1/2 (coin-flip MAP)
  x 3/4 (logical classes that shift the X_L readout) gives about 9 p per round. That is far above the unif-1 LRB-D decay and
  plausibly above the check-free `const 0` circuits (F = 0.9989 at m = 22,
  p = 1e-3, 16% rejection; baseline data). A decoder comparison should include
  "decode using only the final round" and "const 0 + correction from the ideal
  final X-check" **[conjecture; to be checked by the decoder agents]**.
- Correction does not remove decoder-induced temporal correlations of the logical
  noise. They are short-ranged (2.4) and are not a source of long-memory
  non-Markovianity **[analytic]**.

---------------------------------------------------------------------------

## 6. Claims

| ID | Label | Claim | Evidence |
|---|---|---|---|
| A3-pauli-frame-C1 | verified analytically | Under A1 to A7, Pauli-frame correction (compose correction into frame, conjugate by later Cliffords, clear on reset, add the frame's X-power to Z-measurements) gives the same joint distribution of all recorded outcomes as ideal, instantaneous physical correction by the same decoder. With common random numbers it gives the same records shot by shot. | Theorem and induction proof, 1.2 and 1.3 |
| A3-pauli-frame-C2 | verified numerically | Frame-update table for sdim's qutrit gates: H (a,b)->(-b,a), H_INV (a,b)->(b,-a), P (a,b)->(a,a+b), CNOT, CZ, SWAP as in 1.1. sdim's SWAP decomposition equals the standard SWAP. The corrected logical outcome is x0 + x3 + b_0 + b_3. | `A3_pauli_frame_statevector_check.py --mode gates`; dense matrices built from sdim conventions |
| A3-pauli-frame-C3 | verified numerically | On the baseline depth-2 circuits (3 Clifford sequences, SI1000 rescaled to p = 0.02, lookup decoder after every round), physical and frame protocols gave identical records in 180/180 shots, all with non-trivial syndromes. On depth-6 circuits (2 sequences, p = 0.01) they agreed in 40/40. Noiseless circuits have deterministic zero syndromes and logical outcome. | `--mode all`, Section 1.4 |
| A3-pauli-frame-C4 | verified analytically | A channel covariant under the full Weyl group is exactly a stochastic Weyl (Pauli) channel. SI1000 satisfies A1 to A7, so the repository's DEM logical-shift correction is exactly equivalent to ideal real-time correction. | Twirl argument (A2), noise model in the brief and circuit files |
| A3-pauli-frame-C5 | verified numerically | Coherent errors and amplitude damping break the equivalence at first order: P(correct) 0.98508 vs 0.94134 (Z-rotation, theta = 0.15), 0.99596 vs 0.98590 (generic unitary), 0.950 vs 0.9025 (damping, gamma = 0.05). Weyl twirling restores exact equality (difference < 1e-15), and so does a frame that commutes with the coherent error. | `--mode minimal` (exact density matrices) |
| A3-pauli-frame-C6 | conjecture | For the Clifford-averaged LRB fidelity, the frame-vs-physical difference under coherent noise is much smaller than the per-shot difference (random Cliffords and random frames partially twirl). Circuit-level test: per-shot logical distributions differ in 67% of 150 shots, while the averaged difference -0.005 +- 0.024 is not resolved. | `--mode coherent`; Knill 2005 remark on random Pauli frames |
| A3-pauli-frame-C7 | from literature | Clifford-only circuits can defer all Pauli corrections to the interpretation of the final measurement. Non-Clifford gates require the frame before the gate. Frames avoid extra correction gates; Riesebos et al. report up to 70% lower logical error for SC17 when decoding time equals the EC time. | DiVincenzo & Aliferis 2007; Terhal 2015; Chamberland et al. 2018; Riesebos et al. 2017 (abstract); Knill 2005 |
| A3-pauli-frame-C8 | verified analytically | Leakage, non-Weyl or non-linear feed-forward, and quantum-memory or state-dependent noise violate the assumptions. Linear Weyl feed-forward and raw-outcome active reset are compatible. | Section 1.5, B3, B5, B7 |
| A3-offline-causal-C1 | verified analytically | With a terminal readout in a Clifford-only circuit, the offline MAP decoder equals the causal Bayesian forward filter that never commits. Offline and causal differ only when decisions must be committed before the end. | Section 2.1 |
| A3-offline-causal-C2 | verified analytically | SI1000 is inversion-symmetric, so for symmetric decoders E[w^R] = (3 P_succ - 1)/2 is real, and MAP maximizes the brief's fidelity metric. | Section 2.2 |
| A3-offline-causal-C3 | verified analytically | F_offline(m) >= F_causal(m) at every depth, for every lookahead and commit rule. | Section 2.3 |
| A3-offline-causal-C4 | verified analytically + numerically | For finite-state causal decoders with i.i.d. bulk faults, F(T) = sum_k c_k lambda_k^T exactly. The boundaries enter only the amplitudes. In the toy (w = 0), lambda_1 = 0.998956 with weight 0.99174 and subleading abs(lambda) < 0.008. | Section 2.4; exact transfer matrix in run A |
| A3-offline-causal-C5 | verified analytically | F_off(T) lies in [Z_T/2, Z_T] with Z_T = sum_s abs(a_T(s)) <= C N_T and N_T submultiplicative (bulk). So limsup (1/T) log F_off is bounded by the norm growth rate, and F_off(T) is bounded below by every windowed causal rate. | Section 2.5, items 1 and 2 |
| A3-offline-causal-C8 | conjecture | Windowed (sliding, committing) decoders converge to the offline rate as the window grows, so the offline-decoded decay is F_off = A lambda_off^T (1 + o(1)) in the bulk, with all boundary effects in A. | Section 2.5 item 3; DKLP overlapping-recovery statement; toy Table A (w = 2 within 7% of offline) |
| A3-offline-causal-C6 | verified numerically (toy) | Distance-3 toy: per-step error offline 4.6e-4 < w=2 5.0e-4 < w=1 5.2e-4 < w=0 7.0e-4 << none 1.0e-2. The decay is exponential in the bulk. A noisy last round without successor lowers A from 1.001 to 0.991 for every decoder and leaves f unchanged. | `A3_offline_vs_causal_toy.py` run A, Table A |
| A3-offline-causal-C7 | verified analytically | Offline decoding measures the per-step rate of the best non-committing (equivalently long-lookahead real-time) decoder. Decoder-induced temporal correlations of the residual logical noise produce transients and amplitudes, not a different asymptotic rate. | Sections 2.4 and 2.6 |
| A3-boundary-C1 | verified numerically | Fault budget per segment: 118.4 p per check round (62.0 p on data-touching locations, 20 p measurement flips, 8 p reset flips, 28.4 p ancilla-only), 100.4 p for the final round, 3.3 to 4.9 p per logical Clifford. The decay is driven by the check rounds. | `A3_fault_budget.py` on baseline circuits (seq 0 and 5, m = 2 and 22) |
| A3-boundary-C2 | verified analytically | For a fixed final-round rule g, the corrected fidelity of the unchecked inverse block is sum_t h^(t) prod_j lambda_j(X_L S^t), a mixture of up to 81 exponentials. It is a single exponential only without correction, and a ratio of such sums with post-selection. | Section 2.7(c) |
| A3-boundary-C3 | verified numerically | A correcting decoder decoded from one final round adds curvature to the inverse-block contribution. In the d = 3 toy with q_u = q, the per-m decay factor falls from ~0.997 to 0.989 for every correcting decoder, while it stays constant at the exact 0.9702 without correction. | Run B, Table B |
| A3-boundary-C4 | verified numerically + analytic bound | Every decoder that leaves the all-zero syndrome uncorrected has F(m=0) <= 0.99296 at p = 1e-3 and <= 0.9554 at p = 1e-2. These follow from the LRB-D acceptance and accepted fidelity at m = 0 and reflect invisible post-check and undetectable final-round faults. | Baseline CSV `results/LRB/{4,8}/unif_check_data/1.csv`; Section 2.7(e) |
| A3-boundary-C5 | verified analytically | Pure X- or pure Z-type single-qutrit faults on wires 0, 2, 3, 4 have weight-1 partners in another logical class with the same syndrome (X0~X3^-1, X2~X4^-1, Z0~Z2^-1, Z3~Z4^-1). Mixed faults and faults on wire 1 do not. | Stabilizer algebra, Section 2.7(b) |
| A3-boundary-C6 | conjecture | Correction using every round of this d = 2 circuit gives a per-step logical error of order 10 p (rough count 9 p per round). That is worse than unif-1 detection and plausibly worse than the check-free const-0 circuits. In the d = 2 toy, offline gains only ~7% over no correction and w = 0 is worse than none. | Fault budget + C5; toy run C, Table C |
| A3-boundary-C7 | verified analytically | Boundary effects enter A (clean start; final round without successor; invisible faults), the rate (unchecked block), and curvature (subleading terms, the inverse-block mixture, depth-dependent decision rules, degraded successor of round m-1). B = 0 for this metric. | Section 2.7(f) |
| A3-gate-knowledge-C1 | from literature + analytic argument | Knowledge of the Clifford sequence is intrinsic to RB inversion, to LRB recovery propagation (Combes et al.) and to frame bookkeeping. RB's gate-independence assumption concerns the noise. Gate-dependent decoded noise still gives an exponential decay with a meaningful rate (Wallman). A gate-agnostic per-round lookup decoder is a useful lower baseline but is not required for fairness. | Section 2.9; Combes et al. 2017; Wallman 2018; DKLP 2002 |
| A3-latency-C1 | verified analytically | Real-time (physical) correction is unnecessary for this Clifford-only LRB with Weyl noise and no feed-forward. The frame is exact (C1), offline decoding is at least as good as any causal decoder (offline-causal-C3), and the 4(m+1) syndrome trits per shot can be stored and decoded after the run. Physical correction would only add gate and idle noise. | Sections 1 and 3; consistent with Terhal 2015 and Knill 2005 |
| A3-latency-C3 | conjecture | Latency budget if corrections had to be physical on transmon qutrits: about 1 us (w = 0, within the reset gap) or about w x 15 us (lookahead w). Estimated from measured CZ/CZ^dag times 783/580 ns and T1(1-2) = 28 to 63 us (Goss et al.) plus *assumed* readout (1 to 2 us) and reset (~1 us) durations. Google's d = 5 decoder ran with 63 us latency at a 1.1 us cycle. | Section 3; Goss et al. 2022; Acharya et al. 2025 |
| A3-latency-C2 | from literature | Latency and throughput become binding only with feed-forward (non-Clifford gates, conditional leakage handling). Throughput must then match syndrome generation to avoid exponential backlog; parallel window decoding fixes throughput. Post-selected computing can proceed optimistically. | Terhal 2015; Skoric et al.; Chamberland et al. 2018; Knill 2005 |

---------------------------------------------------------------------------

## 7. References (all fetched in this task)

1. E. Knill, "Quantum computing with realistically noisy devices", Nature 434,
   39-44 (2005), DOI 10.1038/nature03350; preprint arXiv:quant-ph/0410199 (title
   "Quantum Computing with Very Noisy Devices"). Fetched
   https://arxiv.org/abs/quant-ph/0410199 and the full PDF
   https://arxiv.org/pdf/quant-ph/0410199 (Pauli-frame passages read). The Nature
   DOI landing page redirected to a login. The journal reference is from the arXiv
   listing's related DOI and a web-search index entry.
2. D. P. DiVincenzo and P. Aliferis, "Effective Fault-Tolerant Quantum
   Computation with Slow Measurements", Phys. Rev. Lett. 98, 020501 (2007),
   DOI 10.1103/PhysRevLett.98.020501; arXiv:quant-ph/0607047. Fetched
   https://arxiv.org/abs/quant-ph/0607047, the full PDF, and
   https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.020501 (which gives
   article number 020501 and publication date 9 Jan 2007. The arXiv listing's
   journal-ref field reads 220501, which disagrees with APS).
3. L. Riesebos, X. Fu, S. Varsamopoulos, C. G. Almudever, K. Bertels, "Pauli
   Frames for Quantum Computer Architectures", Proc. 54th Design Automation
   Conference (DAC '17), ACM, pp. 1-6 (2017), DOI 10.1145/3061639.3062300.
   Fetched https://research.tudelft.nl/en/publications/pauli-frames-for-quantum-computer-architectures
   (abstract only; the ACM page returned 403).
4. C. Chamberland, P. Iyer, D. Poulin, "Fault-tolerant quantum computing in the
   Pauli or Clifford frame with slow error diagnostics", Quantum 2, 43 (2018),
   DOI 10.22331/q-2018-01-04-43; arXiv:1704.06662. Fetched abstract page and full
   PDF (Sections 1-2 read).
5. B. M. Terhal, "Quantum error correction for quantum memories", Rev. Mod. Phys.
   87, 307 (2015), DOI 10.1103/RevModPhys.87.307; arXiv:1302.3428. Fetched
   abstract page and full PDF (section "The (logical) Pauli frame" and the backlog
   argument read).
6. J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized
   Benchmarking", arXiv:1702.03688 (2017). Fetched abstract page and full PDF.
7. E. Dennis, A. Kitaev, A. Landahl, J. Preskill, "Topological quantum memory",
   J. Math. Phys. 43, 4452-4505 (2002), DOI 10.1063/1.1499754;
   arXiv:quant-ph/0110143. Fetched abstract page and full PDF (overlapping
   recovery and transversal-CNOT sections read).
8. L. Skoric, D. E. Browne, K. M. Barnes, N. I. Gillespie, E. T. Campbell,
   "Parallel window decoding enables scalable fault tolerant quantum
   computation", arXiv:2209.08552 (2022); related journal DOI
   10.1038/s41467-023-42482-1 (listed on the arXiv page). Fetched abstract page.
9. R. Acharya, L. Aghababaie-Beni, I. Aleiner, et al. (Google Quantum AI),
   "Quantum error correction below the surface code threshold", Nature 638,
   920-926 (2025), DOI 10.1038/s41586-024-08449-y; arXiv:2408.13687. Fetched
   abstract page (63 us average decoder latency at distance 5, 1.1 us cycle).
10. J. J. Wallman, "Randomized benchmarking with gate-dependent noise", Quantum 2,
    47 (2018), DOI 10.22331/q-2018-01-29-47; arXiv:1703.09835. Fetched abstract
    page.
11. E. Magesan, J. M. Gambetta, J. Emerson, "Characterizing quantum gates via
    randomized benchmarking", Phys. Rev. A 85, 042311 (2012),
    DOI 10.1103/PhysRevA.85.042311; arXiv:1109.6887. Fetched abstract page.
12. N. Goss, A. Morvan, B. Marinelli, et al., "High-Fidelity Qutrit Entangling
    Gates for Superconducting Circuits", arXiv:2206.07216 (v3, 2023). Fetched
    https://arxiv.org/html/2206.07216v3 (CZ 783 ns, CZ^dag 580 ns; T1 of the 1<->2
    transitions 28-63 us in the supplementary tables). Journal reference not
    confirmed on a fetched page.
13. D. Gottesman, "Fault-tolerant quantum computation with higher-dimensional
    systems", Chaos Solitons Fractals 10, 1749-1758 (1999),
    DOI 10.1016/S0960-0779(98)00218-5; arXiv:quant-ph/9802007. Fetched abstract
    page.
