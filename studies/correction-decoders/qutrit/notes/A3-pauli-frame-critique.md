# Critique of A3: Pauli frame, offline vs causal decoding, boundaries, latency

Target: `notes/A3-pauli-frame-offline-causal.md` and its scripts
`A3_pauli_frame_statevector_check.py`, `A3_offline_vs_causal_toy.py`,
`A3_fault_budget.py`. Role: adversarial critic. I tried to refute each claim
by re-deriving it, re-running the A3 scripts, writing independent checks, reading
the repository and sdim code, and fetching every cited paper.

Labels for my own statements: **[VA]** verified analytically (derivation given
here), **[VN]** verified numerically (script and output file named),
**[LIT]** from a paper fetched in this review (Section 6), **[CONJ]** conjecture.

---------------------------------------------------------------------------

## 0. Summary

- **Section 1 holds up within the simulated model.** The frame theorem (C1) and
  the gate table (C2) are correct. An independent exact check of the full joint
  record distribution on random three-qutrit circuits gives total-variation
  distance (TVD) <= 7e-16 between physical and frame correction. This holds with
  non-stabilizer inputs and non-symmetric Weyl noise **[VN]**. sdim's own
  matrices and frame rules match the A3 table **[VN]**. The SWAP-decomposition
  sub-claim is correct, but the cited A3 script never checks it.
- **The list of what breaks the frame (C8) mixes two different requirements.**
  Some items break the equivalence; others only need the frame in real time.
  Abort and Clifford feed-forward conditioned on corrected records keep the
  equivalence exactly (TVD ~1e-16) **[VN]**. Two hardware-relevant items break
  it and are missing from the list:
  - level-dependent readout errors (TVD 0.07 to 0.14);
  - active reset whose conditional gate is noisy only when it fires (TVD ~1e-2).

  A3 lists the second as compatible **[VN]**.
- **Coherent-noise conjecture (C6).** The circuit-level evidence was
  underpowered. With 600 shots the Clifford-averaged frame-vs-physical
  difference is resolved: +0.034 +- 0.012. The per-sequence difference is
  as large as +0.090 +- 0.026 **[VN]**.
- **Offline vs causal.** C1, C3 and C5 are correct. C2 overclaims: MAP
  maximizes Re E[w^R] and P_succ, but not the brief's |E w^R|. An exact
  code-capacity counterexample is given, with an O(q^2) effect **[VN]**.
  C7 and C8 do not hold for the study's distance-2 code. There, every
  ambiguous fault leaves a 50/50 logical posterior that later rounds never
  resolve. The offline fidelity is then a two-exponential mixture,
  F ~ (3/4) e^{-mu/2} + (1/4) e^{-3 mu/2}, whose second term decays on the
  same time scale as the signal **[VA]**. Exact enumeration of A3's d = 2 toy
  shows the local decay factor drifting over the whole depth range **[VN]**.
  A3's own Table C already shows committing decoders not approaching offline
  for d = 2.
- **Boundary and fault-budget claims, checked with the repository's own DEM.**
  - The per-round budget is correct (118.4 p and 100.4 p).
  - "3.3 to 4.9 p per Clifford" is wrong. Individual Cliffords carry 0.4 to
    7.9 p; 3.3 to 4.9 p is the range of per-sequence means.
  - The m = 0 bound (0.99296 at p = 1e-3) is valid but loose. The exact
    optimum over all decoders is 0.97888 at p = 1e-3 and 0.79281 at
    p = 1e-2 **[VN]**.
  - First-order slopes, in units of p of 1 - Re E[w^R] per step:
    - MAP: 15.7 to 16.1 p, i.e. a per-step logical error of about 10.5 p,
      which confirms "order 10 p";
    - no correction: 42 to 43 p, so correction does give a first-order gain;
    - MAP using only the final round: 18.4 to 19.0 p.

---------------------------------------------------------------------------

## 1. What I ran and read

Re-runs of A3 scripts (outputs saved in `notes/scripts/`):

- `A3_pauli_frame_statevector_check.py --mode gates|minimal|noiseless`: the
  gate table, the one-qutrit numbers and the noiseless determinism all reproduce.
- `--mode pauli --depth-idx 3 --shots 20 --cliffs 0,1 --p 0.01`: 40/40 shots,
  0 mismatches (`critic_A3-pauli-frame_v2_rerun_depth6_output.txt`).
- `A3_offline_vs_causal_toy.py --schedules memory-noisy --N 2000 --ws 0`: the
  exact w = 0 transfer-matrix spectrum reproduces
  (`critic_A3-pauli-frame_v2_toy_exactw0_output.txt`).
- Run B with a new seed, `--N 50000 --ws 0 --qu 0.01 --qpost 0.0 --schedules lrb --seed 99`
  (`critic_A3-pauli-frame_v2_toy_runB_seed99_output.txt`).
- `A3_fault_budget.py --cliff 0 --depth-idx 1 8`, whose own output shows
  individual Clifford blocks from 0.40 p to 7.90 p.

Independent scripts (all `notes/scripts/critic_A3-pauli-frame_v2_*.py`, outputs
in the matching `_output.txt`):

| script | purpose | runtime |
|---|---|---|
| `_v2_gates.py` | symplectic action of sdim's own matrices (`sdim.unitary`) vs the A3 table; both sdim tableau SWAP decompositions vs the standard SWAP | 5 s |
| `_v2_exact_frame.py` | exact joint record distributions, physical vs frame, on 5 random 3-qutrit circuits and 13 variants | 14 s |
| `_v2_metric.py` | weight-1 ambiguity table; MAP vs the exact optimum of |E w^R| at code capacity; first-order none vs MAP | 15 s |
| `_v2_dem.py` | repository DEM (`src/lrb/dem_simulation.py`): fault budget; exact P(syndrome, logical) for m = 0, 2; first-order budgets for m = 2..22, three sequences | 5 min |
| `_v2_rerun_a3.py` | A3 Pauli mode with a correction counter; A3 coherent mode with |diff| statistics and a 600-shot independent sample | 61 s |
| `_v2_toy_exact.py` | exact offline fidelity of A3's d = 3 toy, T <= 6 | 1 s |
| `_v2_toy_d2_exact.py` | exact offline fidelity of A3's d = 2 toy, T <= 13, vs the two-exponential formula | 3 s |

Code read:

- **Repository DEM** (`src/lrb/dem_simulation.py`): `N1 d` is uniform over the
  8 non-zero (a,b), drawn by rejection sampling; `N1 f` is X^a with a in {1,2};
  `N2` is uniform over the 80 non-zero vectors (`two_qudit_depol_prob_dist`).
- **sdim 1.3.4** (user site-packages):
  - `unitary.py`: X|j> = |j+1>, H = Fourier, P = diag(w^{j(j-1)/2}),
    CNOT|i,j> = |i,i+j>, CZ = (I x H) CNOT (I x H^dag) = diag(w^{ij});
  - `program.py`: frame rules on lines 124-154, identical to the A3 table;
    `ExtendedTableau` for prime d (lines 268-271);
  - `tableau_gates.py::apply_SWAP`.
- **Circuit files** (`experiments/LRB/0/8/{0,1}.chp`): each `N1 f 5p` sits
  immediately before its ancilla `M`, and each reset flip `N1 f 2p` directly
  after `RESET`. Encoding and terminal readout are noiseless.

Note: four files named `critic_A3-pauli-frame_{first_order,gates_dem_check,metric_codecap,toy_exact}.py`
already existed in `notes/scripts/` before this review. I did not write them and
did not use them as evidence.

---------------------------------------------------------------------------

## 2. Pauli-frame claims

### A3-pauli-frame-C1 (theorem, A1-A7): **confirmed**

- The induction in notes 1.3 is correct **[VA]**. I checked each step:
  - Clifford: conjugation of the frame;
  - noise: covariance;
  - measurement: W^dag Pi_k W = Pi_{k - a_q};
  - reset: the output does not depend on the input;
  - correction: frames add.
- Independent exact test **[VN]** (`_v2_exact_frame_output.txt`). Five random
  circuits, each with:
  - a generic non-Clifford unitary on every qutrit before the first correction;
  - stochastic Weyl noise with random, non-symmetric distributions;
  - an asymmetric but state-independent classical readout shift (0.93, 0.05, 0.02);
  - three decoder steps with hash-like Weyl corrections;
  - a mid-circuit reset;
  - six recorded measurements.

  The TVD between the exact joint distributions of all records is at most
  6.4e-16.
- The assumption list is sufficient but stronger than necessary:
  - A4 does not need Weyl-eigenstate resets: a reset to a fixed non-stabilizer
    state also gives TVD <= 6e-16;
  - A6 excludes harmless cases (Section C8).

### A3-pauli-frame-C2 (gate table, SWAP, readout rule): **confirmed**, with a labelling defect

- Built from sdim's own generators, every gate maps all 9 (or 81) Weyl vectors
  exactly as the A3 table states, with 0 mismatches **[VN]** (`_v2_gates_output.txt`).
- sdim's frame simulator (`program.py` 124-154) uses the same rules **[VA, code]**.
- Both sdim tableau SWAP sequences equal the standard SWAP exactly, not just up
  to a Weyl factor. The deviation is 2.5e-15 for the prime-dimension branch,
  which d = 3 uses, and 7.9e-16 for the Weyl branch **[VN]**. The same follows
  algebraically: CNOT, H x H, CNOT, H x H, CNOT, H_b^2 maps |i,j> to |j,i> **[VA]**.
- The X-power after H_INV of a frame (a,b) is b, so the corrected outcome is
  x0 + x3 + b0 + b3 **[VN]**.
- **Defect:** `A3_pauli_frame_statevector_check.py` contains no
  SWAP-decomposition check; it hard-codes the standard SWAP matrix. The
  "verified numerically" label for that sub-claim was unsupported by the cited
  script. It is now supported by `_v2_gates.py`.

### A3-pauli-frame-C3 (180/180, 40/40, noiseless determinism): **confirmed**, with a caveat

- Reproduced: noiseless circuits are deterministic **[VN]**; depth 6 gives
  40/40 with 0 mismatches **[VN]**; the first 20 shots per sequence at depth 2
  (same seed) give 60/60 with 0 mismatches **[VN]**.
- **Caveat:** "every shot had a non-trivial syndrome" does not mean every shot
  had a correction. The weight-1 lookup covers only 32 of the 80 non-zero round
  syndromes:
  - a correction fired in 89 of 180 rounds;
  - 51 of 60 shots (85%) had at least one correction;
  - 15% of shots therefore had a trivial frame **[VN]** (`_v2_rerun_a3_output.txt`).

  The notes' inference "logical fidelity 0.075, so the frames were far from
  trivial" uses the wrong statistic: low fidelity reflects the noise, not the
  frame. The test is still meaningful.

### A3-pauli-frame-C4 (twirl argument; SI1000 satisfies the assumptions; DEM correction = ideal real-time correction): **needs_qualification**

- The twirl argument is correct **[VA]**. Full Weyl covariance gives
  W^dag N(W . W^dag) W = N for every W, so N equals its own twirl, and a Weyl
  twirl is a stochastic Weyl channel.
- SI1000 as compiled satisfies A1-A7 **[VA, code]**:
  - all noise locations are uniform Weyl channels (`dem_simulation.py`);
  - measurement flips are X^a immediately before M, and reset flips X^a after
    an unconditional RESET;
  - the encoding is ideal, there is no feed-forward, and all gates are qutrit
    Cliffords.
- The conclusion is mis-stated. The theorem is per decoder: frame and physical
  application of *the same* decision function agree. The DEM decoder is offline
  on raw detectors, so applying its logical correction "physically" means
  applying it just before readout, which is trivially the same as relabelling.
  The useful statement is different: every ideal real-time decoder can be
  emulated exactly by an offline function of the raw DEM record (corrected
  record = raw + frame propagation, a bijection given the history). Hence the
  best DEM decoder is at least as good as any ideal real-time decoder. This
  holds only for the simulated Weyl model; under coherent noise, physical
  real-time correction can be better (C5, C6).

### A3-pauli-frame-C5 (coherent and damping break equivalence; twirling restores it): **confirmed**

- All seven one-qutrit numbers reproduce exactly **[VN]**.
- My closed forms agree **[VA]**:
  - physical |1 + e^{2i th} + e^{i th}|^2/9 = 0.985084;
  - frame |1 + e^{2i th} + e^{4i th}|^2/9 = 0.941335 at th = 0.15;
  - damping 1 - g vs (1 - g)^2.
- In random multi-qutrit circuits, one coherent or damping location gives a TVD
  of 5e-3 to 0.18; its Weyl twirl gives <= 7e-16 **[VN]**.
- Two remarks. In the coherent example, E = X^-1 acts trivially on |+>, so the
  difference is an echo effect of the physically applied X between two copies
  of V, not the correction of a real error. "First order" means first order in
  the error probability (difference ~2 th^2), not in th.

### A3-pauli-frame-C6 (Clifford-averaged coherent difference much smaller than per shot; conjecture): **needs_qualification**

- A3's run reproduces: -0.0052 +- 0.0235 over 150 shots, 66.7% of shots differ
  **[VN]**.
- An independent 600-shot sample (seed 777, same three sequences, th = 0.1)
  gives **[VN]** (`_v2_rerun_a3_output.txt`):
  - Re E[w^k] physical 0.2048 +- 0.0171, frame 0.1710 +- 0.0169;
  - paired difference **+0.0338 +- 0.0119 (2.8 sigma)**;
  - per sequence +0.033 +- 0.018, -0.021 +- 0.016, **+0.090 +- 0.026**;
  - mean per-shot |difference| 0.166.
- The averaged difference is therefore resolved and non-zero. It is about 20%
  of the per-shot scale and 17% of the fidelity itself. The two runs are
  consistent with each other (1.4 sigma apart). A3's unresolved number was
  underpowered, not evidence of cancellation.
- The test also has two undisclosed limitations. It switches off all `N2` and
  ancilla `N1 d` noise (the code does this, the notes do not say so). It runs at
  F ~ 0.2, far from the regime of an LRB fit.
- Knill's remark [LIT] concerns *deliberately applied* known random Paulis
  ("frequently applying known but random Pauli products, thus modulating the
  Pauli frame"), i.e. twirling, not decoder-induced frames.

### A3-pauli-frame-C7 (literature: deferral, non-Clifford gates, 70% SC17): **confirmed**

All quotes were found in the fetched texts [LIT]:

- Terhal: "The Pauli frame simply alters the interpretation of the final
  measurement outcome", in the all-Clifford case.
- DiVincenzo and Aliferis: "all information determining the current Pauli frame
  must be known before the application of a non-Clifford gate".
- Chamberland, Iyer and Poulin: "saves us from performing additional gates on
  the system".
- Riesebos et al. (abstract): up to 70% "when the decoding time equals the error
  correction time, and maximum parallelism can be obtained".

The C7 summary drops the parallelism condition. Terhal also refines DA: one
"only needs to know the logical Pauli frame which influences the outcome of the
measurement", before the gate that follows the T gate.

### A3-pauli-frame-C8 (what violates / what is compatible): **needs_qualification**

"Violates the assumptions" and "breaks the equivalence" are different. A6 is
sufficient, not necessary. Exact tests **[VN]** (`_v2_exact_frame_output.txt`,
TVD over 5 circuits):

| variant | TVD | A3's classification | correct classification |
|---|---|---|---|
| abort when a corrected record is non-zero | <= 3.3e-16 | violation ("real-time abort") | compatible; needs records in real time only to save time |
| Clifford (H) applied conditionally on a corrected record, frame conjugated in real time | <= 5.1e-16 | violation ("adaptive circuits") | compatible if the frame is tracked in real time |
| active reset X^{-r} with outcome-independent noise | <= 5.4e-16 | compatible | compatible |
| active reset whose conditional gate is noisy only when r != 0 | 6.7e-3 to 2.0e-2 | compatible | **breaks**: the frame changes the raw r, hence which gates fire |
| level-dependent readout confusion (decay-like) | 0.071 to 0.14 | not listed | **breaks** (the transmon readout mechanism) |
| H, qutrit T, H with the frame not updated through T | 6.6e-3 to 0.25 | violation | violation |
| noisy physical correction | 0.025 to 0.087 | (A5) | frame equals the *ideal* physical protocol only |

The correct criterion is:

- a Weyl operation linear in outcomes needs no real-time frame;
- a Clifford operation, or an abort, conditioned on corrected records keeps the
  equivalence but needs the frame in real time;
- a non-Clifford gate requires resolving the frame first.

The leakage item violates A7 by definition. No example shows that it breaks the
equivalence. A concrete mechanism is that frames change which levels are
populated, and |2> -> |3> leakage depends on population.

---------------------------------------------------------------------------

## 3. Offline vs causal claims

### A3-offline-causal-C1 (offline MAP = never-committing forward filter): **confirmed**

P(L_T | s_1..s_T) is the filtering distribution at the final time **[VA]**. The
hidden state must be the error coset (3^6 = 729 classes for this code), with
in-round correlated faults folded into the transition and observation kernel.
This is tractable here. My exact d = 3 enumeration (`_v2_toy_exact.py`)
implements offline decoding as this forward recursion and reproduces A3's
offline rate.

### A3-offline-causal-C2 (SI1000 symmetric; E[w^R] real; MAP maximizes |sum w^k P(k)|): **needs_qualification**

- The symmetry holds **[VA, code]**. E[w^R] = (3 P_succ - 1)/2 for decoders with
  L_hat(-s) = -L_hat(s) **[VA]**.
- "MAP maximizes the brief's metric" is false over all decoders. The optimum of
  |E w^R| is max_phi sum_s max_k Re(e^{-i phi} w^{-k} a(s)), and it can exceed
  MAP by aligning the phases of tied posteriors.
- Exact code-capacity [[5,1,2]]_3 example **[VN]** (`_v2_metric_output.txt`),
  |.|-optimum vs MAP:

  | q | |.|-optimum | MAP |
  |---|---|---|
  | 0.01 | 0.99205 | 0.99204 |
  | 0.05 | 0.95213 | 0.95193 |
  | 0.15 | 0.81060 | 0.80931 |
  | 0.30 | 0.55074 | 0.54767 |

  The gain is O(q^2) **[VA]**.
- |3 P_succ - 1|/2 is also not monotone below P_succ = 1/3.
- In the actual m = 0 and m = 2 circuits at p = 1e-3 and 1e-2 the excess is
  below 1e-5 (`_v2_dem_output.txt`), so it is negligible in practice. The
  correct statement is that MAP maximizes Re E[w^R] and P_succ over all decoders.
- Deterministic argmax tie-breaking (lowest index) can be asymmetric (1|2 ties)
  and then makes E[w^R] complex.

### A3-offline-causal-C3 (fidelity-optimal offline >= any causal decoder; MAP maximizes P_succ): **confirmed**

The optimum is over a superset of functions. Randomized decoders are dominated by
deterministic ones by convexity of |.| **[VA]**. Two notes:

- A *physical* real-time decoder is a function of the raw record only by the
  frame theorem, i.e. under Weyl noise. With coherent noise, real-time physical
  correction beat the frame by 0.034 +- 0.012 in the circuit test above.
- The notes' added sentence in 2.3 ("the same holds for any per-step decay rate
  extracted from them") holds only asymptotically, not for fits over a finite
  depth window.

### A3-offline-causal-C4 (finite-state causal decoder: exact sum of exponentials; boundaries only in amplitudes): **needs_qualification**

- The spectrum reproduces **[VN]**: lambda_1 = 0.998956 (weight 0.99174),
  -0.007505 (0.00892) and 0.002167 (-0.00070).
- The formula needs a diagonalizable M_1; otherwise there are T^j lambda^T terms.
- "Boundaries (..., unchecked segments) enter only the amplitudes" is wrong for
  segments whose length grows with T. The LRB inverse block has m steps and
  multiplies the decay rate. The notes themselves say so in 2.7(c) and 2.7(f)
  ("f = lambda_1 mu_1") and in A3-boundary-C7 ("decay rate: the unchecked
  inverse block"). C4 contradicts them.

### A3-offline-causal-C5 (Z_T/2 <= F_off <= Z_T; Fekete; lower bound by windowed rates): **confirmed**

Each step checks **[VA]**:

- max_k Re(w^{-k} a) >= |a| cos(pi/3);
- |u^T M v| <= ||u|| ||M|| ||v||;
- N_{T+T'} <= N_T N_{T'} for time-homogeneous steps, and also for i.i.d.
  random steps after averaging, so Fekete's lemma applies;
- F_off >= F_causal pointwise.

### A3-offline-causal-C6 (d = 3 toy numbers): **confirmed**, minor statistical caveats

- Exact enumeration of the memory-noisy chain **[VN]** (`_v2_toy_exact_output.txt`)
  gives an offline local factor of 0.9993064, constant to 1e-7 from T = 3 to 6.
  That is eps = 4.62e-4 (A3: 4.64e-4) and A ~ 0.9915 (A3: 0.991).
- The exact w = 0 value is eps = 6.96e-4 (A3: 7.03e-4), and "none" is exactly
  1.0e-2.
- The "last round lowers A" observation is consistent.
- Caveat 1: the toy's standard error, sqrt((1 - F^2)/(2N)), assigns half of
  Var(w^R) to the real part. For a symmetric residual, Var(Re w^R) =
  (1 - F)(1 + 2F)/2, so the quoted errors are about 22% too small near F = 1
  **[VA]**.
- Caveat 2: decoders are compared with unpaired errors on common samples. The
  w = 1 vs w = 2 ordering (5.22e-4 vs 4.97e-4) is not established.

### A3-offline-causal-C7 (offline LRB measures the rate of the best non-committing = long-lookahead real-time decoder; correlations give only transients): **needs_qualification**

- "Equivalently long-lookahead real-time" rests on the conjecture C8, so the
  claim cannot be "verified analytically". The notes' own section 2.6 labels it
  analytic + conjecture.
- For the study's distance-2 code the offline decoder has long memory, not
  short-range correlations. A pure-X or pure-Z fault on wires 0, 2, 3 or 4 has a
  same-syndrome partner in another logical class (C5 below). With nearly equal
  priors, it leaves a near-50/50 logical posterior that later rounds never
  resolve. Take the idealized case: exact 50/50 ties, Poisson(mu) such faults,
  and readout shifts b_i = +-1 with equal probability. Then
  a = prod_i (1 + w^{b_i})/2 and |a| = 2^{-k}. MAP achieves 2^{-k} (k even) or
  2^{-k-1} (k odd), so **[VA]**:

      F_off = (3/4) e^{-mu/2} + (1/4) e^{-3 mu/2},   none: e^{-3 mu/4}

  The "transient" carries weight 1/4 and decays only three times faster than the
  main term, i.e. on the same time scale as the signal.
- Exact enumeration of A3's d = 2 toy **[VN]** (`_v2_toy_d2_exact_output.txt`)
  shows the offline local factor drifting monotonically, 0.98507 at T = 2 to
  0.98584 at T = 13 (q_m = 0), toward the asymptote e^{-q} = 0.99005 reached only
  at T ~ 200. The formula matches the exact values to 0.2%. A3's Table C value
  "offline 9.32e-3" is therefore a fit over the transition, not a rate.
- In the full code most single faults are unambiguous (C5 below), so the effect
  is diluted but present.
- Citing Wallman 2018 [LIT] does not cover this case. Wallman treats Markovian
  gate-dependent noise, whereas decoded logical noise depends on the history.
- The inverse block is decoded once from round m. It is not a "per-step rate of
  a decoder" and enters the LRB decay separately.

### A3-offline-causal-C8 (windowed decoders converge to offline; F_off = A lambda^T (1 + o(1)), boundaries in A; conjecture): **needs_qualification**

- This is plausible for d >= 3: in A3's d = 3 toy, w = 2 is within about 7%
  of the exact offline value. DKLP's statement [LIT] presupposes a correlation
  time ("If we choose T to be large compared to the characteristic correlation
  time").
- For d = 2 it fails as stated:
  - With A3's commit rule (random tie-breaking) every committed ambiguous fault
    costs a factor 1/4, exactly like "none", for any finite window. A3's Table C
    shows exactly this: w = 1 and w = 2 equal "none" (9.89e-3 and 9.96e-3 vs
    9.97e-3), with no trend toward offline (9.32e-3).
  - The offline curve is asymptotically A lambda^T with A = 3/4. The o(1) term
    decays like e^{-mu}, so in the fitting range it is not a boundary effect.
  - **[CONJ]** A committing decoder that alternates tie directions, remembering
    the phase of its own past commits, could recover the offline parity
    advantage. That would make convergence depend on the commit rule, not only
    on the window length.
- "All boundary effects in A" is also contradicted for LRB by A3's own
  boundary-C3 (inverse-block curvature).

---------------------------------------------------------------------------

## 4. Boundary claims (checked with the repository DEM)

### A3-boundary-C1 (fault budget): **needs_qualification**

- Per round **[VN]**, from my own segmentation and A3's script:
  - 118.4 p in total: 62.0 p data-touching, 28.4 p ancilla-only, 20 p
    measurement flips, 8 p reset flips;
  - 100.4 p for the final round;
  - the inverse block carries 4.70 p per inverse Clifford (seq 0, m = 22).
- **Wrong:** "each logical Clifford: 3.3 to 4.9 p". Individual Clifford blocks
  carry 0.4 to 7.9 p (seq 0, m = 22; mean 4.70) and 1.5 to 7.7 p (seq 5). A3's
  own `A3_fault_budget.py` prints 0.40 and 7.90. The values 3.30, 4.70 and 4.86
  are per-sequence means.
- The round/Clifford fault-mass ratio is 25 (m = 22) and 36 (m = 2, seq 0).
- "The LRB decay is driven by the check rounds" is a statement about logical
  failure. Fault mass alone does not establish it.
  - For MAP correction it holds **[VN]**: first-order per-step attribution is
    check rounds 12.8 to 13.3 p, forward Cliffords 1.3 to 1.4 p, inverse block
    1.3 to 1.4 p. The ratio is about 10, not 30.
  - For LRB-D the first-order accepted-shot slope is exactly 0 at every depth
    (7.42 p constant, from invisible final-round faults). The LRB-D decay is
    second order, and the fault budget says nothing about which segment drives it.

### A3-boundary-C2 (81-term formula for the inverse block): **confirmed**

The Fourier expansion is correct **[VA]**. The steps are:

- h(s) = sum_t h^(t) w^{t.s};
- t.s(E) = <S^t, E>;
- E = sum_j E_j with independent E_j, so the expectation factorizes into
  prod_j lambda_j(X_L S^t);
- post-selection gives the stated ratio.

"Single exponential" requires lambda_j independent of j, which holds only after
Clifford averaging. The formula treats the block in isolation, with a clean input.

### A3-boundary-C3 (toy curvature with q_u = q): **confirmed**

With seed 99 **[VN]**:

- offline local factors 0.9981, 0.9973, ..., 0.9896;
- fitted A = 1.0116 (A3: 1.012), causal-0 A = 1.0113;
- "none" stays near 0.970.

This is a stress test (q_u = q). The real block is about 1/25 of a step, and the
real code has d = 2.

### A3-boundary-C4 (F(m=0) <= 0.99296 at 1e-3, <= 0.9554 at 1e-2): **needs_qualification**

- The exact joint P(syndrome, L) at m = 0 from the repository DEM reproduces the
  baseline CSV **[VN]**:
  - p = 1e-3: P_acc 0.938907 and F_acc 0.992491, vs 0.938916 and 0.992503;
  - p = 1e-2: P_acc 0.538529 and F_acc 0.917159, vs 0.538474 and 0.917113.
- The bound is arithmetically correct (0.99295, 0.95539) but loose. The exact
  optimum over **all** decoders at m = 0 is **0.97888** (p = 1e-3) and
  **0.79281** (p = 1e-2). MAP equals that optimum; no correction gives 0.97718
  and 0.79220.
- The invisible-fault factor alone is 0.99259 and 0.92771, already tighter than
  A3's bound.
- Most of the m = 0 correction loss comes from *detected but undecodable*
  final-round faults, not from invisible ones. A3's assumption that every
  detected shot is decoded perfectly is far from attainable, since the last
  round has no successor.
- Calling F(0) a ceiling on the fitted A is unjustified: run B gives A = 1.012
  > F(0).
- At m = 2 (seq 0, p = 1e-3), MAP gives 0.95073 vs LRB-D accepted 0.99243
  (24% rejected).

### A3-boundary-C5 (weight-1 ambiguities): **confirmed**

Exact enumeration **[VN]** finds the 8 ordered pairs:

- X0 ~ X3^-1 and X2 ~ X4^-1;
- Z0 ~ Z2^-1 and Z3 ~ Z4^-1;
- each of these with its inverse.

Mixed faults and wire-1 faults have minimum partner weight 2. The stabilizers
commute, and X_L, Z_L lie in the normalizer.

### A3-boundary-C6 (correction ~10 p per step; worse than detection; no first-order gain; conjecture): **needs_qualification**

First-order single-fault enumeration on the actual circuits (three sequences,
slopes between m = 2 and 22, units of p of 1 - Re E[w^R] per step) **[VN]**:

| decoder | slope |
|---|---|
| MAP | 16.05, 15.70, 15.74 (eps = (2/3) x slope ~ 10.5 to 10.7 p) |
| MAP using only round-m detectors | 19.0, 18.4, 18.4 |
| no correction | 43.2, 42.2, 43.0 |
| LRB-D unif-1 / const-1 (accepted) | 0 |

- "Order 10 p" and "far worse than unif-1 detection" are confirmed.
- **"No first-order gain over no correction" is refuted for this circuit**:
  correction is 2.7 times better at first order. Most single faults (mixed
  X^aZ^b, wire 1, ancilla and measurement faults) are decodable. A3's d = 2 toy,
  where every fault is ambiguous, overstates the ambiguity.
- At code capacity for the X_L readout the gain is 3: 0.75 q vs 2.25 q **[VN]**.
- The toy's "offline gains only ~7%" depends on the fit range (C7).

### A3-boundary-C7 (how boundaries enter the fit): **needs_qualification**

- (i) F(m) = sum c_{kk'} (lambda_k mu_k')^m treats the forward and inverse
  segments as independent. In mirrored LRB, inverse Clifford j undoes forward
  Clifford j, and forward step j and inverse step j+1 are both conjugated by the
  same prefix G_j = C_j...C_1. The Clifford average is u^T Phi^m(.) v with a
  paired operator Phi(X) = E_C[M(C) X N(C^{-1})]. The per-m factor is then
  E_Q[lambda_F(Q) lambda_I(Q)], not E[lambda_F] E[lambda_I]; the two agree only
  to first order or for twirl-invariant noise **[VA]**.
- (ii) The d = 2 offline-memory curvature is missing (C7).
- (iii) "B = 0" is correct in expectation. The recommended signed Re E[w^R] is
  the right fix for the finite-shot bias of |.|.

---------------------------------------------------------------------------

## 5. Gate knowledge and latency claims

### A3-gate-knowledge-C1 (sequence-aware decoder is fair; from literature): **needs_qualification**

- The fairness argument is sound. Combes et al. [LIT] write "this reasoning is a
  function of the syndromes s_j collected after the jth encoded gate and of the
  sequence i performed". They also note that for stabilizer codes "we need not
  actually perform the recovery operation". That directly supports the
  frame-based reading of LRB.
- DKLP [LIT] confirms the transversal-CNOT boundary point.
- Overreach: Wallman 2018 [LIT] proves near-exponential decay for Markovian
  gate-dependent noise. Decoded logical noise is history-dependent, both
  committed and offline, so "gate-dependent decoded logical noise still gives
  exponential decay with a meaningful rate" does not follow. For offline d = 2
  decoding the ambiguous-fault part is a two-exponential mixture (C7).

### A3-latency-C1 (real-time correction unnecessary here): **confirmed**

This holds within the stated scope: Clifford-only gates, Weyl noise, no
feed-forward. Storage of 4(m+1) = 92 trits at m = 22, about 9.2e7 trits per 1M
shots, is correct. "Physical correction would only add noise" is true only for
Weyl noise; with coherent errors physical correction can help (C5, C6).

### A3-latency-C2 (backlog; parallel windows): **confirmed**

Each source states what is attributed to it [LIT]:

- Terhal: "if rgen/rproc = f > 1, a small initial backlog ... [leads] to an
  exponential slow down", with D_k = f^k D_1 at the k-th T gate; and "a quantum
  memory (including Clifford gates) ... never has to wait".
- Skoric et al. (abstract): "the slow-down is now only polynomial in code size".
- Chamberland et al.: they distinguish measurement latency t_l from repetition
  time t_r; long t_l is tolerable, while increasing t_r lowers the threshold.

### A3-latency-C3 (1 us / w x 15 us budget; 0.4 ms ~ 10 T1; conjecture): **confirmed**, as an order-of-magnitude estimate

- The Goss et al. numbers check [LIT]: CZ dagger 580 ns and CZ 783 ns (Fig. 3
  caption); T1 of 1-2 is 63, 47, 33 and 28 us (Supplementary Tables 1-2).
- The Acharya et al. numbers check [LIT]: 63 us latency at distance 5 with a
  1.1 us cycle.
- The arithmetic is consistent: 23 rounds of ~15 us plus 44 Cliffords of 1 to
  3 us gives ~0.4 ms, i.e. 6 to 14 T1.
- Caveats:
  - the circuit uses CNOT, which costs a CZ plus two single-qutrit gates;
  - the four sequential readouts are a compile choice;
  - the readout and reset times are assumptions, as the notes state.

---------------------------------------------------------------------------

## 6. References fetched in this review

1. D. P. DiVincenzo, P. Aliferis, "Effective fault-tolerant quantum computation
   with slow measurements", Phys. Rev. Lett. 98, 020501 (2007),
   DOI 10.1103/PhysRevLett.98.020501; arXiv:quant-ph/0607047. Fetched
   https://arxiv.org/abs/quant-ph/0607047 (journal-ref field says 220501) and
   https://arxiv.org/pdf/quant-ph/0607047 (text extracted, page 3 quotes).
2. B. M. Terhal, "Quantum error correction for quantum memories", Rev. Mod.
   Phys. 87, 307 (2015), DOI 10.1103/RevModPhys.87.307; arXiv:1302.3428. Fetched
   https://arxiv.org/abs/1302.3428 and https://arxiv.org/pdf/1302.3428
   (Section II.G.3, "The (Logical) Pauli frame", and the backlog argument).
3. E. Knill, "Quantum Computing with Very Noisy Devices", arXiv:quant-ph/0410199
   (2004); related journal DOI 10.1038/nature03350 listed on the arXiv page.
   Fetched https://arxiv.org/abs/quant-ph/0410199 and the PDF (page 5 passages).
4. C. Chamberland, P. Iyer, D. Poulin, "Fault-tolerant quantum computing in the
   Pauli or Clifford frame with slow error diagnostics", Quantum 2, 43 (2018),
   DOI 10.22331/q-2018-01-04-43; arXiv:1704.06662. Fetched the abstract page and
   the PDF (introduction).
5. L. Riesebos, X. Fu, S. Varsamopoulos, C. G. Almudever, K. Bertels, "Pauli
   Frames for Quantum Computer Architectures", Proc. 54th DAC (2017),
   DOI 10.1145/3061639.3062300. Fetched
   https://research.tudelft.nl/en/publications/pauli-frames-for-quantum-computer-architectures
   (abstract only).
6. J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized
   Benchmarking", arXiv:1702.03688 (2017); no journal reference on the arXiv page.
   Fetched the abstract page and https://arxiv.org/pdf/1702.03688.
7. E. Dennis, A. Kitaev, A. Landahl, J. Preskill, "Topological quantum memory",
   J. Math. Phys. 43, 4452-4505 (2002), DOI 10.1063/1.1499754;
   arXiv:quant-ph/0110143. Fetched the abstract page and the PDF (overlapping
   recovery; transversal CNOT).
8. J. J. Wallman, "Randomized benchmarking with gate-dependent noise", Quantum 2,
   47 (2018), DOI 10.22331/q-2018-01-29-47; arXiv:1703.09835. Fetched the
   abstract page.
9. L. Skoric, D. E. Browne, K. M. Barnes, N. I. Gillespie, E. T. Campbell,
   "Parallel window decoding enables scalable fault tolerant quantum
   computation", arXiv:2209.08552; related DOI 10.1038/s41467-023-42482-1.
   Fetched the abstract page.
10. R. Acharya, L. Aghababaie-Beni, I. Aleiner, et al., "Quantum error correction
    below the surface code threshold", Nature 638, 920-926 (2025),
    DOI 10.1038/s41586-024-08449-y; arXiv:2408.13687. Fetched the abstract page.
11. N. Goss, A. Morvan, B. Marinelli, et al., "High-Fidelity Qutrit Entangling
    Gates for Superconducting Circuits", arXiv:2206.07216 (v3); related DOI
    10.1038/s41467-022-34851-z. Fetched https://arxiv.org/abs/2206.07216 and
    https://arxiv.org/html/2206.07216v3.
12. E. Magesan, J. M. Gambetta, J. Emerson, "Characterizing quantum gates via
    randomized benchmarking", Phys. Rev. A 85, 042311 (2012),
    DOI 10.1103/PhysRevA.85.042311; arXiv:1109.6887. Fetched the abstract page.
13. D. Gottesman, "Fault-Tolerant Quantum Computation with Higher-Dimensional
    Systems", Chaos Solitons Fractals 10, 1749-1758 (1999),
    DOI 10.1016/S0960-0779(98)00218-5; arXiv:quant-ph/9802007. Fetched the
    abstract page.
