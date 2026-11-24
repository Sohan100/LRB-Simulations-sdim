# WP D: causal (real-time) versus offline decoding, and the Pauli-frame claim

Qubit `[[5,1,2]]` LRB (lead's model in `src/lrb_qubit/`), generalized SI1000 noise,
`data5` terminal readout, 30 Clifford sequences from
`experiment.clifford_sequences(30, 22, 20261028)`.

Labels: **[VA]** verified analytically, **[VN]** verified numerically, **[LIT]** from
literature, **[CONJ]** conjecture. Every number has a statistical uncertainty
(± one standard error) unless it is an exact computation.

## 1. Summary

1. **Causal decoders that must commit a Pauli-frame correction every round cost
   12–19% in per-step logical error at lookahead w = 0, a few percent at w = 1–2,
   and almost nothing beyond w = 2.** The cost is measured as the fitted per-step
   rate (1−f) relative to offline exact MLD, on identical shots
   (paired-bootstrap SE ≤ 0.006) **[VN]**.

   | lookahead w (state commit) | p = 5e-4 | p = 2e-3 | p = 5e-3 |
   |---|---|---|---|
   | 0 | 1.115 ± 0.004 | 1.165 ± 0.005 | 1.188 ± 0.003 |
   | 1 | 1.018 ± 0.002 | 1.065 ± 0.002 | 1.086 ± 0.002 |
   | 2 | 1.005 ± 0.002 | 1.033 ± 0.002 | 1.049 ± 0.002 |
   | 4 | 1.004 ± 0.002 | 1.028 ± 0.002 | 1.041 ± 0.002 |
   | ∞ (commit, all slots visible) | 1.004 ± 0.002 | 1.028 ± 0.002 | 1.040 ± 0.002 |

   The offline values are f = 0.98779 ± 0.00014, 0.94767 ± 0.00061 and
   0.85722 ± 0.00122. Within errors, w = 4 equals unbounded lookahead. So the
   residual 0.4–4% is the cost of *committing hard decisions*, not of missing
   future syndromes.
2. **Future syndromes barely improve the estimate of the logical frame if the
   decoder keeps soft information.** Ground truth from DEM error sampling
   (15 sequences, depth 22) gives the per-round accumulation rate of
   logical-frame error. For an uncommitted forward filter with w = 0 (all past
   syndromes, none from the future), this rate relative to offline smoothing is
   0.998 ± 0.001, 0.999 ± 0.001 and 1.000 ± 0.003 at the three p values **[VN]**.
   Future syndromes do fix the *attribution* of detection events (measurement
   versus data fault, 11× fewer attribution errors at p = 5e-4). That matters
   only once the decoder hardens its decisions.
3. **What offline decoding does to the decay rate.** For this Clifford-only
   protocol the offline f is the per-round accumulation rate of
   logical-frame uncertainty given all syndromes. In the bulk it equals the rate
   of a real-time *soft* filter (point 2), so future syndromes change only a
   boundary term at the decision point, not f **[VN]**. The offline f is therefore
   achievable in real time with Pauli-frame tracking and a 32-number forward
   filter per shot. It is *not* the rate of a hard-decision real-time decoder:
   one forced to commit every round (for example before non-Clifford gates)
   decays faster, by the factors in point 1 **[VN]**.
4. **Causal decoding changes the rate, not the shape, of F(m).** Local decay
   rates over depth windows are flat within errors for every decoder. The
   curvature index κ = r(14–22)/r(2–10) of each causal decoder agrees with
   offline (|κ − κ_offline| ≤ 2.4σ, mostly < 1.5σ). The amplitude b changes only
   for w = 0 **[VN]**.
5. **Pauli-frame claim, verified.** With noiseless corrections, a correction
   applied physically mid-circuit and a Pauli-frame flip of the readout give the
   same logical output distribution, for linear and non-linear feedback rules.
   - Exact DEM comparison of Stim classically controlled Paulis (`CX rec[-k] q`)
     with the frame observable, 900 circuits: max |ΔP| = 0.0 **[VN]**.
   - Shot by shot with identical noise realisations: 4.5M shots, 0 mismatches,
     although the measured syndromes differ in up to 1.8M round-rows **[VN]**.
   - Analytic argument in §5.4 **[VA]**.
6. **Where the two differ.**
   - Noisy correction gates: ΔP_flip = +6e-5 to +8e-4 for an always-played
     DEPOLARIZE1(0.1p) gate, and about 10× that at q = p (exact) **[VN]**.
   - Non-Clifford gates: across a logical T the naive frame is wrong by ε/√2 in
     an exact model; it is fixed by a frame-conditioned S† that needs the
     decision at the T gate **[VN, VA]**.
   - Leakage: in an exact single-transmon toy, physical and frame differ by
     ε·λ, with the sign set by the return channel **[VN toy; CONJ for hardware]**.
7. **Side result: the natural lookup decoder is logically inert in forward
   rounds.** The corrections {X0, Z0, X2, Z3} form a group that the
   fold-transversal logical gates map to itself. Forward-round corrections from
   this table therefore never change the logical output; only the terminal
   decision matters **[VA, VN]**. A non-covariant table ({X3, X2, Z0, Z4}) makes
   forward corrections harmful: P_flip 0.333 versus 0.194 at p = 5e-4, m = 22.

No bug was found in the lead's model (§7). One operational problem affects every
agent sharing the allocation (§7).

## 2. Definitions and methods

### 2.1 Causal decoders (`src/lrb_qubit/decoders_causal.py`)

The lead's exact MLD (`mld.TransferMatrixMLD`) is a forward filter over time slots
(slot t = check round t, slot m = terminal round, slot m+1 = final data detectors).
Its state after slot t is (b_t, l_t):

- b_t: the slot-t detector bits produced by fault mechanisms whose latest slot is t;
- l_t: the cumulative observable flip of all mechanisms whose latest slot is ≤ t.

Mechanisms with latest slot t+1 (measurement errors, faults between a round's Z and X
readouts, hooks) may also flip slot-t detectors. The leftover s_t ⊕ b_t is explained
at the next slot.

`CausalWindowMLD(dem, slots, lookahead=w, commit=...)` commits, for every slot t in
time order, using only slots ≤ t + w:

- `commit="state"` (primary; windowed MLD with commit-and-slide): commit the
  posterior mode of (b_t, l_t), which is a complete Pauli-frame decision (syndrome
  attribution plus logical frame). The decoder then restarts its filter from the
  committed state and forgets the past. Because the DEM is Markov in this state,
  this equals conditioning on "the committed state is true" **[VA]**.
- `commit="logical"` (diagnostic): commit only l_t and keep the soft distribution
  over b_t. This models a decoder that must resolve the logical frame every round
  but may keep 16 soft syndrome numbers.
- Posterior: the exact forward message from the committed state times an exact
  backward message over slots t..t+w. When the window ends before the last slot,
  the unobserved future is marginalised exactly ("open boundary"). The prediction
  is the committed l̂ of the last slot. `lookahead=None` lets every commit see all
  slots (a non-causal diagnostic).
- `lagged_marginals(dets, w)`: the *uncommitted* fixed-lag estimate
  P(l_t, b_t | s_0..s_{t+w}). `None` gives offline smoothing.
- `true_boundary_states(dem, slots, errors)`: ground-truth (b_t, l_t) from
  `dem.compile_sampler().sample(..., return_errors=True)`.
- `CausalStream`: round-by-round API; slot t is committed as soon as slot t+w
  arrives.

Correctness checks (tests in `tests/test_qubit512_causal.py`; 64 pass):

- Every commit posterior, in both modes, with committed history and open and
  closed window ends, equals a brute-force enumeration of the DEM to 1e-9
  (depth 1–2, p = 0.03) **[VN]**.
- The uncommitted lagged marginals equal brute force **[VN]**.
- The extracted kernels reproduce the lead's LLR exactly (max |Δ| = 0.0) **[VN]**.
- Causality: randomising detection events in slots > t + w never changes the
  commits of slots ≤ t **[VN]**.
- Streaming equals batch; full lookahead equals a window wider than the circuit
  **[VN]**.
- My offline MLD agrees with the lead's `grid_main` `mld5` (independent shots) at
  all 27 shared (p, m) points; largest deviation −0.0017 ± 0.0008 (2.2σ) **[VN]**.

### 2.2 Grid (`studies/correction-decoders/qubit/scripts/causal_vs_offline.py grid`)

`data5` circuits; one detector sample per circuit, decoded by every method (identical
shots).

| p | shots per circuit | depths | sequences |
|---|---|---|---|
| 5e-3 | 20,000 | all 0..22 | 30 |
| 2e-3 | 10,000 | 0,1,2,3,4,6,8,10,14,18,20,22 | 30 |
| 5e-4 | 10,000 | 0,1,2,3,4,6,8,10,14,18,20,22 | 30 |

Contention on the shared allocation (§7) forced the scale-down for p = 2e-3 and 5e-4.
The second depth set is a superset of the repo's nine depths.

### 2.3 Statistics

- e(m) and F(m) = |1 − 2e(m)| are computed per sequence and averaged over sequences.
  The quoted SE is std/√30, which includes shot noise and sequence variation.
- Paired differences against offline on identical shots: McNemar
  Δe = (n10 − n01)/N with SE √(n10 + n01)/N. The sequence-level SE is in the CSV
  and agrees.
- Fit: the repo's fixed-offset weighted grid scan F = b f^m (offset 0, weights
  1/std², std over sequences, points with F > 1 dropped), on all depths.
  `fit_fixed_offset` is tested to equal `LRBResultsPlotter._fit_decay_parameter_fixed_a`
  to 1e-12. SEs come from 400 bootstrap resamples of the 30 sequences. Ratios
  against offline use a paired bootstrap (same resampled sequences). The
  convention r = (1−f)/2 is in the CSV.
- Per-slot study: bulk rate = slope of −ln(1 − 2 e_t) over forward slots 2..20,
  with a sequence bootstrap paired across estimators.

### 2.4 Pauli-frame experiments

- `build_segmented_lrb` cuts the lead's circuit after every check round. The
  concatenation equals `build_lrb_circuit` instruction by instruction (tested).
- Feedback rules:
  - **A**: memoryless linear lookup every round, S_Z1→X0, S_Z2→X2, S_X1→Z0,
    S_X2→Z3.
  - **A'**: the same with the non-covariant table S_Z1→X3, S_Z2→X2, S_X1→Z0,
    S_X2→Z4.
  - **B1**: non-linear two-round confirmation; correct only stabilizer flips
    present in this and the propagated previous round.
  - **B2_w**: this WP's causal MLD (w = 0, 1, state commit) in streaming mode; its
    logical decisions are applied as the propagated Z_L mid-circuit.
- Exact check: rule A/A' as Stim classically controlled Paulis (`CX rec[-4] 0`,
  `CX rec[-3] 2`, `CZ rec[-2] 0`, `CZ rec[-1] 3` after each round) versus the
  no-feedback circuit whose observable includes the frame's parity of ancilla
  records. P(flip) is computed exactly from each DEM.
- Identical-noise check: `stim.FlipSimulator` with the same seed in both modes. I
  checked that masked Pauli injections do not consume random numbers, so both runs
  see the same noise realisation. The physical mode injects the correction Paulis
  into the frame simulator (an explicit conditional gate). The frame mode XORs the
  frame's syndrome into the controller's view and flips the readout.
- Noisy corrections:
  - always-played gate: DEPOLARIZE1(q) on the corrected qubits every round, exact
    by DEM;
  - conditional: a random Pauli with probability q only when a correction fires,
    by FlipSimulator.

## 3. Results: causal versus offline

### 3.1 Fits F = b f^m (offset 0, all depths)

f ± bootstrap SE; ratio = (1−f_decoder)/(1−f_offline) ± paired-bootstrap SE.

| decoder | f (p=5e-4) | ratio | f (p=2e-3) | ratio | f (p=5e-3) | ratio |
|---|---|---|---|---|---|---|
| offline | 0.98779 ± 0.00014 | 1 | 0.94767 ± 0.00061 | 1 | 0.85722 ± 0.00122 | 1 |
| state_w0 | 0.98638 ± 0.00014 | 1.115 ± 0.004 | 0.93902 ± 0.00054 | 1.165 ± 0.005 | 0.83041 ± 0.00111 | 1.188 ± 0.003 |
| state_w1 | 0.98757 ± 0.00013 | 1.018 ± 0.002 | 0.94427 ± 0.00062 | 1.065 ± 0.002 | 0.84491 ± 0.00128 | 1.086 ± 0.002 |
| state_w2 | 0.98772 ± 0.00013 | 1.005 ± 0.002 | 0.94594 ± 0.00063 | 1.033 ± 0.002 | 0.85016 ± 0.00125 | 1.049 ± 0.002 |
| state_w4 | 0.98774 ± 0.00013 | 1.004 ± 0.002 | 0.94623 ± 0.00062 | 1.028 ± 0.002 | 0.85141 ± 0.00120 | 1.041 ± 0.002 |
| state_winf | 0.98774 ± 0.00013 | 1.004 ± 0.002 | 0.94623 ± 0.00062 | 1.028 ± 0.002 | 0.85152 ± 0.00119 | 1.040 ± 0.002 |
| logical_w0 | 0.98605 ± 0.00014 | 1.142 ± 0.006 | 0.93902 ± 0.00057 | 1.165 ± 0.005 | 0.83315 ± 0.00138 | 1.169 ± 0.003 |
| logical_w1 | 0.98761 ± 0.00014 | 1.015 ± 0.004 | 0.94544 ± 0.00060 | 1.043 ± 0.003 | 0.84718 ± 0.00132 | 1.070 ± 0.002 |
| logical_winf | 0.98767 ± 0.00015 | 1.010 ± 0.003 | 0.94631 ± 0.00060 | 1.026 ± 0.003 | 0.85002 ± 0.00136 | 1.050 ± 0.002 |

The amplitude b equals offline within 0.001 for w ≥ 1. It drops only for w = 0, for
example at p = 2e-3: 0.9495 ± 0.0007 (state_w0) and 0.9490 ± 0.0005 (logical_w0)
versus 0.9563 ± 0.0006 offline. Fits on the repo's nine depths agree with these
within 1σ (`fits_data5.csv`, depths = repo9).

### 3.2 Logical error rate e(m) and F(m) (mean over 30 sequences ± SE)

p = 5e-4

| decoder | e(2) | e(10) | e(22) | F(2) | F(10) | F(22) |
|---|---|---|---|---|---|---|
| offline | 0.0194 ± 0.0005 | 0.0631 ± 0.0011 | 0.1222 ± 0.0012 | 0.9612 ± 0.0010 | 0.8738 ± 0.0022 | 0.7556 ± 0.0024 |
| state_w0 | 0.0215 ± 0.0004 | 0.0696 ± 0.0012 | 0.1330 ± 0.0012 | 0.9571 ± 0.0008 | 0.8609 ± 0.0024 | 0.7341 ± 0.0024 |
| state_w1 | 0.0195 ± 0.0005 | 0.0639 ± 0.0012 | 0.1240 ± 0.0012 | 0.9609 ± 0.0009 | 0.8721 ± 0.0024 | 0.7520 ± 0.0023 |
| state_w2 | 0.0195 ± 0.0005 | 0.0634 ± 0.0012 | 0.1229 ± 0.0012 | 0.9610 ± 0.0009 | 0.8732 ± 0.0024 | 0.7543 ± 0.0024 |
| state_w4 | 0.0195 ± 0.0005 | 0.0633 ± 0.0012 | 0.1227 ± 0.0012 | 0.9611 ± 0.0009 | 0.8734 ± 0.0024 | 0.7546 ± 0.0023 |

p = 2e-3

| decoder | e(2) | e(10) | e(22) | F(2) | F(10) | F(22) |
|---|---|---|---|---|---|---|
| offline | 0.0756 ± 0.0016 | 0.2222 ± 0.0027 | 0.3514 ± 0.0019 | 0.8487 ± 0.0032 | 0.5557 ± 0.0053 | 0.2971 ± 0.0037 |
| state_w0 | 0.0844 ± 0.0014 | 0.2483 ± 0.0023 | 0.3813 ± 0.0015 | 0.8312 ± 0.0028 | 0.5034 ± 0.0045 | 0.2375 ± 0.0030 |
| state_w1 | 0.0776 ± 0.0016 | 0.2321 ± 0.0026 | 0.3638 ± 0.0020 | 0.8448 ± 0.0032 | 0.5359 ± 0.0051 | 0.2724 ± 0.0041 |
| state_w2 | 0.0766 ± 0.0016 | 0.2272 ± 0.0026 | 0.3577 ± 0.0020 | 0.8467 ± 0.0032 | 0.5456 ± 0.0052 | 0.2846 ± 0.0040 |
| state_w4 | 0.0766 ± 0.0016 | 0.2264 ± 0.0026 | 0.3566 ± 0.0019 | 0.8468 ± 0.0032 | 0.5472 ± 0.0053 | 0.2868 ± 0.0039 |

p = 5e-3

| decoder | e(2) | e(10) | e(22) | F(2) | F(10) | F(22) |
|---|---|---|---|---|---|---|
| offline | 0.1788 ± 0.0030 | 0.4053 ± 0.0022 | 0.4851 ± 0.0009 | 0.6424 ± 0.0060 | 0.1895 ± 0.0043 | 0.0299 ± 0.0018 |
| state_w0 | 0.1986 ± 0.0025 | 0.4305 ± 0.0015 | 0.4935 ± 0.0007 | 0.6027 ± 0.0050 | 0.1391 ± 0.0030 | 0.0130 ± 0.0015 |
| state_w1 | 0.1847 ± 0.0028 | 0.4177 ± 0.0021 | 0.4892 ± 0.0008 | 0.6306 ± 0.0055 | 0.1646 ± 0.0042 | 0.0215 ± 0.0015 |
| state_w2 | 0.1822 ± 0.0028 | 0.4128 ± 0.0021 | 0.4867 ± 0.0008 | 0.6356 ± 0.0056 | 0.1744 ± 0.0042 | 0.0267 ± 0.0016 |
| state_w4 | 0.1821 ± 0.0028 | 0.4119 ± 0.0021 | 0.4858 ± 0.0008 | 0.6358 ± 0.0056 | 0.1763 ± 0.0043 | 0.0284 ± 0.0015 |

All depths and all nine decoders are in `results/D/summary_data5.csv`.

### 3.3 Paired excess error on identical shots, Δe = e_causal − e_offline (McNemar ± SE)

| decoder | p | Δe(2) | Δe(6) | Δe(10) | Δe(22) |
|---|---|---|---|---|---|
| state_w0 | 5e-4 | +0.00206 ± 0.00023 | +0.00458 ± 0.00033 | +0.00645 ± 0.00040 | +0.01075 ± 0.00055 |
| state_w1 | 5e-4 | +0.00014 ± 0.00009 | +0.00064 ± 0.00015 | +0.00083 ± 0.00020 | +0.00180 ± 0.00028 |
| state_w2 | 5e-4 | +0.00008 ± 0.00009 | +0.00018 ± 0.00015 | +0.00028 ± 0.00019 | +0.00065 ± 0.00027 |
| state_w4 | 5e-4 | +0.00007 ± 0.00009 | +0.00015 ± 0.00015 | +0.00022 ± 0.00019 | +0.00048 ± 0.00026 |
| logical_w0 | 5e-4 | +0.00175 ± 0.00029 | +0.00495 ± 0.00042 | +0.00735 ± 0.00051 | +0.01513 ± 0.00070 |
| logical_w1 | 5e-4 | +0.00008 ± 0.00018 | −0.00002 ± 0.00030 | +0.00050 ± 0.00037 | +0.00122 ± 0.00054 |
| state_w0 | 2e-3 | +0.00876 ± 0.00050 | +0.01756 ± 0.00067 | +0.02614 ± 0.00078 | +0.02982 ± 0.00098 |
| state_w1 | 2e-3 | +0.00193 ± 0.00027 | +0.00593 ± 0.00044 | +0.00988 ± 0.00054 | +0.01238 ± 0.00074 |
| state_w2 | 2e-3 | +0.00100 ± 0.00026 | +0.00268 ± 0.00041 | +0.00503 ± 0.00050 | +0.00625 ± 0.00069 |
| state_w4 | 2e-3 | +0.00094 ± 0.00026 | +0.00236 ± 0.00040 | +0.00424 ± 0.00050 | +0.00515 ± 0.00068 |
| logical_w0 | 2e-3 | +0.00860 ± 0.00056 | +0.01849 ± 0.00076 | +0.02593 ± 0.00089 | +0.02846 ± 0.00109 |
| logical_w1 | 2e-3 | +0.00114 ± 0.00040 | +0.00382 ± 0.00062 | +0.00583 ± 0.00074 | +0.00889 ± 0.00096 |
| state_w0 | 5e-3 | +0.01985 ± 0.00049 | +0.02981 ± 0.00065 | +0.02522 ± 0.00074 | +0.00847 ± 0.00086 |
| state_w1 | 5e-3 | +0.00594 ± 0.00032 | +0.01335 ± 0.00052 | +0.01244 ± 0.00064 | +0.00417 ± 0.00080 |
| state_w2 | 5e-3 | +0.00341 ± 0.00030 | +0.00786 ± 0.00049 | +0.00755 ± 0.00060 | +0.00166 ± 0.00076 |
| state_w4 | 5e-3 | +0.00330 ± 0.00030 | +0.00677 ± 0.00048 | +0.00661 ± 0.00059 | +0.00074 ± 0.00075 |
| logical_w0 | 5e-3 | +0.02128 ± 0.00056 | +0.02959 ± 0.00070 | +0.02531 ± 0.00078 | +0.00685 ± 0.00088 |
| logical_w1 | 5e-3 | +0.00636 ± 0.00044 | +0.01255 ± 0.00062 | +0.01073 ± 0.00070 | +0.00323 ± 0.00083 |

At p = 5e-3 the differences shrink at large m only because every decoder approaches
e = 1/2.

### 3.4 Shape of F(m): local rates r = −Δln F̄/Δm (bootstrap SE)

| p | decoder | r(2–6) | r(6–10) | r(10–14) | r(14–18) | r(18–22) | κ = r(14–22)/r(2–10) | κ − κ_offline |
|---|---|---|---|---|---|---|---|---|
| 5e-4 | offline | 0.0117 ± 0.0004 | 0.0122 ± 0.0005 | 0.0118 ± 0.0005 | 0.0118 ± 0.0005 | 0.0128 ± 0.0006 | 1.031 ± 0.043 | — |
| 5e-4 | state_w0 | 0.0131 ± 0.0004 | 0.0134 ± 0.0005 | 0.0131 ± 0.0006 | 0.0137 ± 0.0006 | 0.0130 ± 0.0006 | 1.010 ± 0.037 | −0.021 ± 0.023 |
| 5e-4 | state_w1 | 0.0120 ± 0.0004 | 0.0123 ± 0.0006 | 0.0123 ± 0.0005 | 0.0120 ± 0.0006 | 0.0128 ± 0.0006 | 1.022 ± 0.046 | −0.009 ± 0.012 |
| 5e-4 | state_w4 | 0.0117 ± 0.0004 | 0.0122 ± 0.0006 | 0.0121 ± 0.0005 | 0.0117 ± 0.0006 | 0.0127 ± 0.0006 | 1.019 ± 0.046 | −0.012 ± 0.011 |
| 2e-3 | offline | 0.0526 ± 0.0012 | 0.0533 ± 0.0016 | 0.0516 ± 0.0016 | 0.0507 ± 0.0023 | 0.0542 ± 0.0024 | 0.991 ± 0.028 | — |
| 2e-3 | state_w0 | 0.0605 ± 0.0011 | 0.0649 ± 0.0018 | 0.0591 ± 0.0020 | 0.0637 ± 0.0024 | 0.0651 ± 0.0036 | 1.027 ± 0.027 | +0.036 ± 0.024 |
| 2e-3 | state_w1 | 0.0558 ± 0.0012 | 0.0580 ± 0.0017 | 0.0528 ± 0.0017 | 0.0573 ± 0.0024 | 0.0591 ± 0.0031 | 1.023 ± 0.028 | +0.032 ± 0.014 |
| 2e-3 | state_w4 | 0.0537 ± 0.0011 | 0.0554 ± 0.0017 | 0.0515 ± 0.0017 | 0.0538 ± 0.0024 | 0.0562 ± 0.0028 | 1.007 ± 0.027 | +0.017 ± 0.011 |
| 5e-3 | offline | 0.1497 ± 0.0030 | 0.1556 ± 0.0039 | 0.1494 ± 0.0048 | 0.1565 ± 0.0061 | 0.1558 ± 0.0130 | 1.023 ± 0.041 | — |
| 5e-3 | state_w0 | 0.1800 ± 0.0026 | 0.1867 ± 0.0046 | 0.1866 ± 0.0070 | 0.1888 ± 0.0105 | 0.2164 ± 0.0287 | 1.105 ± 0.077 | +0.082 ± 0.074 |
| 5e-3 | state_w1 | 0.1647 ± 0.0029 | 0.1711 ± 0.0051 | 0.1589 ± 0.0052 | 0.1902 ± 0.0079 | 0.1592 ± 0.0174 | 1.041 ± 0.048 | +0.018 ± 0.048 |
| 5e-3 | state_w4 | 0.1569 ± 0.0029 | 0.1639 ± 0.0048 | 0.1552 ± 0.0048 | 0.1695 ± 0.0072 | 0.1315 ± 0.0130 | 0.939 ± 0.039 | −0.084 ± 0.042 |

- Every decoder's local rate is flat from m = 2 to 22 within errors.
- The 0–2 window is higher for all decoders alike (for example 0.0600 ± 0.0019
  offline at 2e-3; in `local_rates_data5.csv`).
- The κ differences are within 2.4σ. The two largest (+0.032 ± 0.014 at 2e-3,
  w = 1; −0.084 ± 0.042 at 5e-3, w = 4) are not consistent in sign. At
  p = 5e-3 the late windows have F̄ ≈ 0.03, where ln F is noisy and |1−2e| is
  biased.

### 3.5 Per-round ground truth: smoothing, uncommitted lag, committed (depth 22, 15 sequences × 20,000 shots)

Bulk rate of logical-frame error accumulation, −d ln(1 − 2 e_t)/dt over forward
slots 2..20, ± bootstrap SE. The ratio to offline smoothing uses a paired
bootstrap.

| estimator | rate 5e-4 | ratio | rate 2e-3 | ratio | rate 5e-3 | ratio |
|---|---|---|---|---|---|---|
| smoothing (offline) | 0.01132 ± 0.00019 | 1 | 0.05060 ± 0.00073 | 1 | 0.14690 ± 0.00179 | 1 |
| lag_w0 (uncommitted filter) | 0.01131 ± 0.00019 | 0.9984 ± 0.0014 | 0.05056 ± 0.00073 | 0.9993 ± 0.0012 | 0.14691 ± 0.00193 | 1.0001 ± 0.0029 |
| lag_w1 | 0.01132 ± 0.00019 | 0.9995 ± 0.0005 | 0.05059 ± 0.00073 | 0.9999 ± 0.0004 | 0.14731 ± 0.00184 | 1.0028 ± 0.0014 |
| lag_w2 | 0.01132 ± 0.00019 | 1.0000 ± 0.0001 | 0.05059 ± 0.00073 | 1.0000 ± 0.0003 | 0.14704 ± 0.00180 | 1.0010 ± 0.0006 |
| state_w0 | 0.01267 ± 0.00022 | 1.1190 ± 0.0073 | 0.06001 ± 0.00066 | 1.1861 ± 0.0071 | 0.18361 ± 0.00424 | 1.2499 ± 0.0243 |
| state_w1 | 0.01156 ± 0.00019 | 1.0212 ± 0.0035 | 0.05408 ± 0.00067 | 1.0688 ± 0.0043 | 0.15889 ± 0.00310 | 1.0817 ± 0.0108 |
| state_w2 | 0.01141 ± 0.00019 | 1.0076 ± 0.0024 | 0.05198 ± 0.00068 | 1.0274 ± 0.0030 | 0.15538 ± 0.00260 | 1.0577 ± 0.0096 |
| state_w4 | 0.01138 ± 0.00019 | 1.0049 ± 0.0022 | 0.05168 ± 0.00064 | 1.0214 ± 0.0032 | 0.15556 ± 0.00219 | 1.0590 ± 0.0095 |
| state_winf | 0.01138 ± 0.00019 | 1.0051 ± 0.0022 | 0.05167 ± 0.00064 | 1.0212 ± 0.0033 | 0.15583 ± 0.00222 | 1.0608 ± 0.0102 |
| logical_w0 | 0.01307 ± 0.00022 | 1.1544 ± 0.0071 | 0.06064 ± 0.00086 | 1.1986 ± 0.0081 | 0.17645 ± 0.00339 | 1.2012 ± 0.0169 |
| logical_w1 | 0.01149 ± 0.00021 | 1.0147 ± 0.0080 | 0.05284 ± 0.00083 | 1.0444 ± 0.0049 | 0.16253 ± 0.00340 | 1.1064 ± 0.0139 |
| logical_winf | 0.01138 ± 0.00022 | 1.0054 ± 0.0063 | 0.05179 ± 0.00087 | 1.0237 ± 0.0064 | 0.15902 ± 0.00293 | 1.0825 ± 0.0102 |

- Attribution error at slot 10 (b̂_t ≠ b_t; `perslot_depth22.csv`) at
  p = 5e-4, 2e-3 and 5e-3:
  - smoothing: 0.0014, 0.0211, 0.111;
  - uncommitted w = 0: 0.0154, 0.0677, 0.190.
- The uncommitted filter's frame error exceeds smoothing's by a constant that
  does not grow with t. At p = 2e-3, slot 10: 0.2073 versus 0.2048, standard
  errors about 0.0026 but the two are strongly correlated.

### 3.6 Cost

Python decode time per 10^5 shot-circuits, averaged over the depth grid, login
node, single thread:

| offline MLD | state w0 | state w1 | state w2 | state w4 | state w∞ | logical w0 / w1 / w∞ |
|---|---|---|---|---|---|---|
| 0.43–0.48 s | 0.30 s | 0.5 s | 0.7–0.8 s | 1.0–1.1 s | 0.5 s | 1.1–1.4 s |

These are not hardware latencies. A committed w = 0 step is a table gather of 32
numbers.

## 4. Interpretation (task item 3)

**What future syndromes do.** The offline MLD outputs one bit from the whole syndrome
record. In the HMM picture its decay rate is the per-round accumulation rate of
uncertainty about the logical frame, each round being judged with all syndromes
before and after it (smoothing). Table 3.5 shows this rate equals the rate of the
causal *filter* that keeps soft information, at every w including w = 0, to
0.1–0.3% **[VN]**.

The reason is that the posterior of the logical frame of this d = 2 code is limited by
the intrinsic pair ambiguity (Z0 vs Z2, X0 vs X3, ...: errors that differ by a
logical). Future syndromes cannot resolve that ambiguity. They resolve time-like
ambiguity (measurement versus data fault, faults split between the Z and X readouts
of a round), which the filter keeps as soft weights and revises one round later. The
filter's extra error is a boundary term at the decision time. It does not accumulate
**[VN]**.

So, in Clifford-only LRB:

- (a) offline decoding does not inflate f relative to the best real-time decoder
  with soft memory **[VN]**;
- (b) the offline f is the decoded-logical rate of *this protocol with this
  decoder*, not the eigenvalue of a fixed per-step logical channel. The decision
  about step j depends on later syndromes, and LRB-D's retroactive rejection is
  acausal in the same sense **[VA]**;
- (c) f describes Clifford-only computation with Pauli-frame tracking. Where the
  frame must be *committed* (non-Clifford gates, feed-forward), the relevant rate
  is the committed one, larger by the factors in Tables 3.1 and 3.5 **[VN]**.

**Why committing costs.** A committing decoder forgets that its past decisions could
be wrong. Each round's boundary term then becomes permanent and is paid every round.
Its size shrinks with lookahead w. Even with all slots visible, a residual cost of
hard decisions remains (state_winf: +0.4%, +2.8% and +4.0% in 1−f at the three p).
That residual grows with p, as expected for a higher-order effect (it needs several
faults in a decision window) **[VN]**; the mechanism is **[CONJ]**.

Quantitatively, the committed excess rate equals the full-lookahead commit cost plus
the filtering boundary term B_w = ⟨−ln(1−2e^lag_w) + ln(1−2e^smooth)⟩ within a factor
of about 1.5. At p = 2e-3, w = 0: excess 0.0094 ± 0.0003 versus B_0 = 0.0095 ± 0.0003.
At p = 5e-4 the excess is 0.66 B_0; at p = 5e-3 it is 1.4 B_0 after subtracting the
commit cost **[VN numbers, CONJ model]**.

Committing only the logical frame (`logical_*`, soft attribution kept) does not avoid
the w = 0 penalty (1.14–1.17) **[VN]**. The penalty comes from fixing the logical
increment of the latest round before the next round can disambiguate it. Hardening
the attribution adds a further cost at w ≥ 1 for p ≥ 2e-3 (state_w1 1.065 versus
logical_w1 1.043 at 2e-3) **[VN]**.

**Shape of F(m).** Causal decoding multiplies the per-round rate but keeps
the decay exponential.

- Local rates are flat within errors and the curvature index agrees with offline
  (Table 3.4) **[VN]**.
- Structurally, every bulk round is committed by the same rule, so the per-round
  cost is constant **[VA]**.
- The decoder's behaviour differs only in the first round (no history) and in the
  last w slots before the end, where the window reaches the end. Those terms are
  m-independent for m > w and land in b; for m < w the w-decoder equals the
  full-lookahead commit decoder **[VA]**.
- The predicted transient at m ≈ w is smaller than our errors for w ≥ 2 (Table 3.2:
  state_w2 versus state_w4 differ by ≤ 0.001 in e at m ≤ 4) **[VN]**.
- The inverse half has no checks. It is decoded identically by all methods from the
  terminal syndrome, so it adds the same rate to all decoders and dilutes their
  ratios: fitted ratios in Table 3.1 are 1.115/1.165/1.188 at w = 0 versus per-round
  1.119/1.186/1.250 in Table 3.5 **[VN]**.

## 5. Results: physical correction versus Pauli frame (task item 4)

### 5.1 Exact comparison, rule A/A' with Stim classically controlled Paulis

900 circuits: rules A and A' × 3 p × 30 sequences × m ∈ {0,2,6,10,22}. For every one,
P_flip(physical feedback circuit) equals P_flip(frame circuit) exactly (max |ΔP| = 0.0)
**[VN]**.

| rule | p | m | P_flip feedback = frame | P_flip no correction | ΔP always-played gate q=0.1p | ΔP q=p |
|---|---|---|---|---|---|---|
| A | 5e-4 | 2 | 0.0395 | 0.0282 | +1.5e-4 (1.2e-4..1.8e-4) | +1.5e-3 |
| A | 5e-4 | 22 | 0.1942 | 0.1898 | +6.5e-4 (5.8e-4..6.9e-4) | +6.4e-3 |
| A | 2e-3 | 2 | 0.1405 | 0.1039 | +4.7e-4 (3.9e-4..5.6e-4) | +4.7e-3 |
| A | 2e-3 | 22 | 0.4302 | 0.4261 | +5.9e-4 (5.6e-4..6.1e-4) | +5.6e-3 |
| A | 5e-3 | 2 | 0.2816 | 0.2210 | +7.2e-4 (6.1e-4..8.3e-4) | +7.1e-3 |
| A' | 5e-4 | 2 | 0.0548 | 0.0282 | +2.0e-4 (1.8e-4..2.4e-4) | +2.0e-3 |
| A' | 5e-4 | 22 | 0.3326 | 0.1898 | +6.7e-4 (6.1e-4..7.2e-4) | +6.6e-3 |
| A' | 2e-3 | 2 | 0.1855 | 0.1039 | +5.6e-4 (4.8e-4..6.8e-4) | +5.6e-3 |

The table shows means over 30 sequences; ranges over sequences are in brackets. All
are exact DEM values (`results/D/frame_rule_a_exact.csv`).

### 5.2 Identical noise realisations (FlipSimulator), 15 circuits per (rule, p), 20,000 shots each

| rule | p | output mismatches | shots with ≥1 correction | measured-syndrome round-rows that differ | err (physical = frame) | Δerr, conditional noisy gate q=0.1p | Δerr q=p |
|---|---|---|---|---|---|---|---|
| A | 5e-4 | 0 / 300,000 | 0.314 | 514,223 | 0.0888 | +1.7e-5 ± 7.5e-6 | +7.0e-5 ± 1.7e-5 |
| A | 2e-3 | 0 / 300,000 | 0.630 | 1,280,779 | 0.2397 | +2.0e-5 ± 2.2e-5 | +3.2e-4 ± 7.2e-5 |
| A | 5e-3 | 0 / 300,000 | 0.790 | 1,779,144 | 0.3584 | +3.0e-5 ± 5.4e-5 | +9.1e-4 ± 1.7e-4 |
| A' | 5e-3 | 0 / 300,000 | 0.790 | 1,779,144 | 0.3845 | −1.7e-4 ± 7.1e-5 | +5.3e-5 ± 2.2e-4 |
| B1 | 5e-3 | 0 / 300,000 | 0.746 | 1,478,131 | 0.3584 | +2.7e-5 ± 4.4e-5 | +4.1e-4 ± 1.4e-4 |
| B2_w0 | 5e-3 | 0 / 300,000 | 0.403 | 0 | 0.3086 | +4.3e-5 ± 3.0e-5 | +1.4e-4 ± 9.7e-5 |
| B2_w1 | 5e-3 | 0 / 300,000 | 0.399 | 0 | 0.2986 | 0.0 ± 2.9e-5 | −1.2e-4 ± 9.1e-5 |

All 15 (rule, p) rows are in `frame_flipsim.csv`. In total: 4,500,000 paired shots,
0 output mismatches, 0 mismatches of the controller's adjusted syndromes **[VN]**.

- For the syndrome-changing rules (A, A', B1), the physically corrected run measures
  different raw syndromes in up to 1.8M round-rows. The test is therefore not
  trivial.
- B2 applies logical Paulis (trivial syndrome), so its measured syndromes are
  identical. Its decisions are this WP's non-linear causal MLD.
- With conditional noisy correction gates the physical error rises by up to
  +9.1e-4 ± 1.7e-4 at q = p. A' has one −2.4σ entry: its corrections are themselves
  harmful, so noise on them can partly cancel. With 30 noisy comparisons one such
  deviation is expected.

### 5.3 Covariant versus non-covariant lookup **[VA + VN]**

The table {X0, Z0, X2, Z3} generates a group G. The physical logical gates
(SWAP(2,3)+H, S/S†/CZ, X, Z) map G to itself (closure checked in
`test_covariant_lookup_forward_corrections_are_logically_inert`). The syndrome map
G → F2^4 is a bijection, so g·lookup(syn g) = I for every g ∈ G. A frame built from
forward corrections therefore reduces at the terminal round to lookup(raw_m), which
gives three consequences:

- rule A and rule B1 give *identical* logical outputs, shot by shot, equal to a
  terminal-only lookup (tested; also seen in 5.2: err 0.0888/0.2397/0.3584 for both);
- a memoryless real-time lookup decoder gains nothing in forward rounds;
- with the non-covariant table (X3, Z4), logical H maps X3 to Z2, which is
  re-corrected by Z0 and leaves Z_L. Forward corrections then hurt badly: P_flip
  0.3326 versus 0.1942 at p = 5e-4, m = 22, both worse than no correction (0.1898).

### 5.4 Why the two agree, and where they would not **[VA]**

The argument: all noise is Pauli and all gates are Clifford, measurements are Z-basis
(Pauli frames act on them as outcome flips), and corrections act on data qubits,
which are never reset mid-circuit. Then:

1. a Pauli channel commutes with any Pauli (PEP† = ±E);
2. a Clifford maps the correction Pauli to another Pauli;
3. a Z measurement after a Pauli frame F gives the frame-free outcome XOR
   [F anticommutes with Z].

By induction over rounds, the frame run's measured record XOR the frame's syndrome
equals the physical run's record, *for the same noise realisation*. The controller,
applying the same decision rule to that adjusted record, makes the same decisions, and
the readout flip from the frame reproduces the physical output. This holds for any
decision rule, linear or not. Each hypothesis can fail:

| broken assumption | consequence | evidence |
|---|---|---|
| noiseless correction gates | physical run gets extra errors; frame does not | always-played DEPOLARIZE1(0.1p): +6e-5..+8e-4 in P_flip (exact, 5.1); about 10× at q = p **[VN]** |
| all gates Clifford | across T_L = exp(−iπ/8 Z_L) a Pauli frame becomes non-Pauli (T X T† ∝ (X+Y)/√2) | exact 5-qubit model: X0 error (prob ε), T_L, Y_L readout. Physical correction and frame + S_L† fix-up both give P(+1) = cos²(π/8) = 0.85355; naive frame gives 0.85355 − ε/√2 (0.78284 at ε = 0.1) **[VN]**. The fix-up needs the frame at the T gate, so it must be a committed (causal) decision before the next non-commuting gate **[VA]** |
| Pauli action on the qubit | a physical X/Z on a leaked qubit (level 2) does nothing; the frame flip still applies | exact single-transmon toy, error ε = 0.1 corrected, leakage λ: incoherent leak with return to 1 gives P_err physical = ελ, frame 0; return to 0 gives the reverse; coherent leak and X readout give physical ελ, frame 0 **[VN toy]**. Which approach is better depends on the leakage return channel **[CONJ for hardware]** |

## 6. Claims

| # | claim | label | evidence |
|---|---|---|---|
| C1 | Committed (hard-decision) causal decoding raises the fitted 1−f over offline MLD by 11.5/16.5/18.8% (w=0), 1.8/6.5/8.6% (w=1), 0.5/3.3/4.9% (w=2), 0.4/2.8/4.1% (w=4) at p = 5e-4/2e-3/5e-3, SE ≤ 0.6% | VN | Table 3.1, `fits_data5.csv` |
| C2 | w = 4 equals unbounded lookahead within errors; the remaining excess is the cost of committing | VN | Tables 3.1, 3.3, 3.5 |
| C3 | An uncommitted causal filter (w = 0) accumulates logical-frame error at the same per-round rate as offline smoothing, within 0.1–0.3% | VN | Table 3.5, `perslot_rates_depth22.csv` |
| C4 | Future syndromes resolve time-like attribution (3–11× fewer attribution errors), not the d=2 logical pair ambiguity | VN (numbers), CONJ (reason) | §3.5, `perslot_depth22.csv` |
| C5 | Offline f is the smoothed per-round rate of Clifford-only LRB with Pauli-frame tracking, equal to the real-time soft-filter rate. It is not a fixed per-step channel eigenvalue and not the rate of hard-decision real-time decoders | VA (structure) + VN (equality) | §4 |
| C6 | Causal decoding changes the rate, not the shape, of F(m): flat local rates, curvature index equal to offline within 2.4σ | VN | Table 3.4 |
| C7 | Committed excess ≈ full-lookahead commit cost + filtering boundary term B_w, within a factor of about 1.5 | CONJ (with VN numbers) | §4 |
| C8 | The committed posteriors are exact (brute force) and causal (future-slot tampering test) | VN | `tests/test_qubit512_causal.py` |
| C9 | Physical mid-circuit correction ≡ readout frame flip (noiseless corrections, Clifford + Pauli noise), for any decision rule | VA | §5.4 |
| C10 | Same, exact for linear rules A/A' as Stim `CX rec[-k] q` feedback: max |ΔP| = 0.0 over 900 circuits | VN | §5.1 |
| C11 | Same, shot by shot under identical noise for A, A', B1 (non-linear) and B2 (non-linear causal MLD): 0 mismatches in 4.5M shots | VN | §5.2 |
| C12 | Noisy correction gates break the equivalence: +6e-5..+8e-4 (always-played, q = 0.1p) and about 10× at q = p | VN | §5.1, §5.2 |
| C13 | Across a non-Clifford gate the naive frame fails (by ε/√2 in the model); a frame-conditioned Clifford fix-up restores it, which requires a committed decision at the gate | VN (model) + VA | §5.4, `toys.csv` |
| C14 | Leakage breaks the equivalence; the sign of the difference depends on the return channel | VN (toy) / CONJ (hardware) | §5.4, `toys.csv` |
| C15 | Lookup table {X0,Z0,X2,Z3} is covariant under the fold-transversal gates, so a memoryless real-time lookup's forward corrections are logically inert; a non-covariant table makes them harmful | VA + VN | §5.3 |
| C16 | My offline MLD reproduces the lead's `grid_main` mld5 within errors at 27 points | VN | §2.1 |

Literature context: sliding-window or overlapping-recovery decoding, and the Pauli
frame (Knill 2005, arXiv:quant-ph/0410199; Skoric et al., arXiv:2209.08552). These are
**UNVERIFIED by WP D** (not fetched by me; `brainstorm/other-platforms.md` and
`brainstorm/skeptic.md` list them). No claim above depends on them.

## 7. Bugs and issues

- **No bug found in the lead's model.** Checks:
  - kernel extraction from `TransferMatrixMLD.transfer` reproduces its LLR exactly;
  - brute-force posteriors match;
  - the rebuilt segmented circuit equals `build_lrb_circuit` instruction by
    instruction;
  - raw-syndrome/detector reconstruction from `lc.back` equals Stim's detectors (the
    B2 rule reproduces batch decoding of the same FlipSimulator samples);
  - my offline numbers match `grid_main`;
  - the lead's 64 tests pass.
- **Operational issue (affects all agents): CPU-set collision on the shared
  allocation.** Every `srun --jobid=<id> --overlap -c 24` step gets the *same* cgroup
  CPU set, `Cpus_allowed_list: 0-11,128-139` (12 physical cores). I saw this on both
  nid200047 and nid200033. With five or six agents running at once (about 150
  processes), each process got 15–20% of a hardware thread while most of the 256
  CPUs were idle. `--cpu-bind=none` does not help, because the cgroup restricts the
  step. My p = 5e-3 tasks took about 1000 s each on the allocation versus about
  40 s on the login node. To finish, I ran the p = 2e-3 and p = 5e-4 grids, and the
  last 5 sequences at p = 5e-3, on the login node: 4 processes, nice 19, 75–90 s
  each.
  - The lead has since confirmed the problem and set the rule: run every step
    as `srun --overlap --cpu-bind=none -c 256` and keep at most 24 workers. That
    fix covers my observation that `--cpu-bind=none` alone does not help with
    `-c 24`. My runs had already finished and were not repeated with it.
- A memoryless lookup is worse than no correction at all for the X_L readout (for
  example 0.1405 versus 0.1039 at p = 2e-3, m = 2). This is because a terminal Z
  correction acts on measurement errors. This is not a model bug; it is a caution
  for anyone using lookup baselines.

## 8. Open questions

1. A decoder that stays soft and commits only where the circuit forces it
   (non-Clifford points) should recover the smoothing rate between those points.
   How does the committed rate depend on T-gate density in Clifford+T benchmarks?
   **[CONJ]**
2. The fixed-offset fit has χ² (SE-weighted) of 30–37 for 10 dof at p ≤ 2e-3 for
   every decoder alike. This mild excess is common to offline and causal, so it is
   not caused by causality. Its origin (initial-window transient at m ≤ 2, visible
   in the 0–2 local rates) is not resolved here.
3. Only the `data5` terminal was run. The qutrit pipeline's `data2` readout adds a
   first-order undetectable terminal floor (lead's finding). It should affect b,
   not the causal ratios **[CONJ]**.
4. Circuit-level leakage needs a leakage-aware simulator (beyond Stim); the toy here
   only shows the sign ambiguity.
5. Hardware latency of the committed w = 0/1 filter: a 32-entry gather per round,
   trivially FPGA-sized **[CONJ]**. Real timing was not measured.

## 9. Reproduce

```bash
cd <repository root>
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
# tests (about 40 s)
python -m pytest tests/test_qubit512_causal.py -q
# grids (identical shots for all decoders); --resume skips completed sequences
python studies/correction-decoders/qubit/scripts/causal_vs_offline.py grid --p 0.005:20000 --workers 4 --resume
python studies/correction-decoders/qubit/scripts/causal_vs_offline.py grid --p 0.002:10000,0.0005:10000 \
    --depths 0,1,2,3,4,6,8,10,14,18,20,22 --workers 4 --resume
python studies/correction-decoders/qubit/scripts/causal_vs_offline.py report --boot 400
# per-round ground truth
python studies/correction-decoders/qubit/scripts/causal_vs_offline.py perslot --p 0.0005,0.002,0.005 \
    --depth 22 --num-seq 15 --shots 20000 --workers 4
# Pauli frame and toys
python studies/correction-decoders/qubit/scripts/causal_vs_offline.py frame --p 0.0005,0.002,0.005 \
    --depths 0,2,6,10,22 --num-seq 30 --flip-seqs 3 --shots 20000
python studies/correction-decoders/qubit/scripts/causal_vs_offline.py toys
```

On the shared allocation, use the lead's form (at most 24 workers):

```bash
srun --jobid=<JOBID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c \
  "module load python/3.11; cd <WORKTREE>; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; \
   python studies/correction-decoders/qubit/scripts/causal_vs_offline.py grid --p 0.005:20000 --workers 24 --resume"
```

Outputs in `studies/correction-decoders/qubit/results/D/`:

| file | contents |
|---|---|
| `summary_data5.csv` | e(m) and F(m) for every decoder, p and depth |
| `fits_data5.csv` | f, b, bootstrap SE, ratios |
| `paired_vs_offline_data5.csv` | Δe with McNemar and sequence SE |
| `local_rates_data5.csv` | windowed rates and curvature |
| `perslot_depth22.csv` | per-slot frame and attribution errors |
| `perslot_rates_depth22.csv` | per-slot bulk rates |
| `frame_rule_a_exact.csv` | exact rule A/A' comparison |
| `frame_flipsim.csv` | identical-noise physical versus frame runs |
| `toys.csv` | non-Clifford and leakage models |
| `raw/*.npz` | counts |
| `logs/` | run logs |

Seeds:

- grid: `--seed 4242` plus a per-p offset, sampler seed per (sequence, depth);
- perslot: 91000 + 1000·seq + 1e6·p;
- frame: 1000·seq + 17·m + 1e6·p.
