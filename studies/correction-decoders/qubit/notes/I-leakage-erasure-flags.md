# WP I: erasure heralding, leakage and flags

Qubit `[[5,1,2]]` LRB under SI1000 noise. Code, protocol and noise follow
`../CONTEXT.md` and `../API.md`. Every number below comes from code in this
worktree; the reproduce commands are at the end.

Claim labels: **VA** verified analytically, **VN** verified numerically,
**LIT** from literature, **C** conjecture. Uncertainties are one standard
error (SE). Unless stated otherwise, it is the SE over Clifford sequences of
the per-sequence quantity, which includes shot noise. Comparisons between
methods use identical shots wherever both methods can be computed on them.
Erasure methods share one circuit per configuration, leakage methods share
one simulated circuit per configuration, and the first-order analysis is
exact.

## Summary

**Headline.**

- **Erasures help correction only once the dominant data-idle errors are
  heralded.**
  - Heralding every two-qubit-gate error (r = 1) cuts the exact MLD's
    first-order error per depth step by only 21%. Sampled decay: f = 0.9480(7)
    → 0.9586(7) at p = 2e-3.
  - Also heralding the measurement-idle errors cuts it by 76–83%:
    f → 0.9873(2), and F(22) goes from 0.299 to 0.757.
  - Every single heralded erasure is decoded correctly at circuit level
    (0 of 14052 locations).
- **Erasure postselection.**
  - The hybrid rule "correct heralded, reject unexplained" raises LRB-D
    acceptance 1.4–4.2 times, at slightly lower accepted F: 0.983–0.992
    against 0.995 at m = 10, p = 2e-3.
  - Herald-aware |LLR| postselection at the same acceptance as LRB-D beats
    LRB-D's F: 0.9998(1) against 0.9951(5).
- **Leakage.**
  - Leakage persistence leaves clear fingerprints: a linearly growing
    leaked-data population and long-range detection correlations.
  - The decoded-LRB decay it adds is exponential, and an LRU only reduces
    that rate by about 20%.
  - Rejecting shots with three-state-readout or LRU leakage flags restores
    the leakage-free decoded fidelity, at a cost in acceptance.
  - LRB-D unif=1 keeps its fidelity and pays in acceptance.
  - Sparse-check LRB-D (const=1) loses fidelity non-exponentially, and
    three-state readout repairs it.
- **Flags.** At least 77% of the exact MLD's first-order failures are data
  errors on the ambiguous pairs, which no flag can resolve. The best layout
  (`wrap`, a flag CNOT before the first and after the second data CNOT)
  removes every hook miscorrection but cuts the first-order error only 5–7%.
  With SI1000-noisy flags it is neutral at p = 2e-3 and harmful at p = 5e-3.

Statistics are modest:

- erasure: 12 sequences × 2e4 shots per circuit;
- leakage: 12 sequences × 4e4 shots;
- flags, sampled: 6 sequences × 1e4 shots;
- exact first-order analysis: 6 sequences.

Sequences are `experiment.clifford_sequences(n, 22, 20261028)`; depths are
0, 2, 4, 6, 10, 14, 18, 20 and 22; p ∈ {5e-4, 2e-3, 5e-3}.

## 1. Methods

### 1.1 Erasure conversion that keeps the Pauli channel exactly (VA, VN)

`DEPOLARIZE2(q)` is the same channel as replacing both qubits by the
maximally mixed state with probability λ = 16q/15. In that picture, a fraction
r of the replacement events is heralded:

- heralded event, probability 16rq/15: both qubits replaced, herald bit 1;
- unheralded event, probability 16(1−r)q/15: both qubits replaced, no herald.

The two events are mutually exclusive. The marginal Pauli channel is
`DEPOLARIZE2(q)` for every r (VA: the heralded events put rq/15 on each of
the 15 non-identity Paulis). One-qubit channels work the same way:
`DEPOLARIZE1(q)` is replacement with probability 4q/3.

- **Converted locations.** `ErasureSpec(two_qubit=r)` converts every
  two-qubit gate: the 12 check CNOTs per round and the logical SWAP(2,3) and
  CZ(2,3). `ErasureSpec(meas_idle=r)` also converts the `DEPOLARIZE1(2p)` on
  every idle qubit in each single-ancilla measurement layer. Other channels
  stay unheralded: data idle during reset, gate depolarizing, measurement
  and reset flips.
- **Sampling.**
  - Two-qubit locations use one exact `CORRELATED_ERROR` /
    `ELSE_CORRELATED_ERROR` chain with 31 outcomes: 16 heralded outcomes,
    one per two-qubit Pauli (identity included), each also flipping a
    dedicated herald qubit read by `MR`; and the 15 unheralded non-identity
    Paulis.
  - One-qubit locations use Stim's `HERALDED_ERASE(4rq/3)` followed by
    `DEPOLARIZE1((1−r)q/(1−4rq/3))`. This is the same channel exactly (VA):
    given a herald the Pauli is uniform; given none it is depolarizing; the
    total is `DEPOLARIZE1(q)`.
  - Herald records become detectors, appended after the lead's detector
    layout. Each herald detector carries its time slot: the index of the
    first check round that can see the event, stored as
    `NoiseLocation.slot` and as a detector coordinate.
- **Checks.**
  - With no conversion, `build_instrumented_lrb` reproduces
    `circuits.build_lrb_circuit` instruction for instruction, for both data2
    and data5 terminals (VN, test).
  - With conversion, detector and observable statistics equal the
    unconverted circuit's (VN: max |z| < 4.5 over all detectors, and the
    herald rate matches its expected value).
  - `HERALDED_ERASE` heralds with probability e and then applies a uniform
    Pauli (VN, Bell-pair test).

### 1.2 Exact herald-conditioned MLD (VN)

Stim's DEM cannot represent heralded channels exactly. `detector_error_model`
refuses `HERALDED_ERASE` and `ELSE_CORRELATED_ERROR` unless
`approximate_disjoint_errors=True`, which treats mutually exclusive outcomes
as independent mechanisms (VN).

The lead's `TransferMatrixMLD` also cannot take herald detectors as slot
state. A round slot would need 4 + 12 or more detectors, above its limit of 8.

`erasure.HeraldedMLD` therefore conditions on the heralds instead.

- **Prior.** The prior is the DEM of the *no-herald conditional* circuit
  (`mode="decode"`). Every converted channel is replaced by its exact
  conditional given "no herald": `DEPOLARIZE2((1−r)q/(1−16rq/15))`, or
  `DEPOLARIZE1((1−r)q/(1−4rq/3))`.
- **A heralded location.** Its error is a uniform Pauli on its qubits. That
  is a product of independent X and Z components on each qubit, each with
  probability 1/2. Convolving these components into the slot kernels absorbs
  the location's base channel exactly, because a uniform distribution
  convolved with anything stays uniform.
- **Heralds as kernel modifiers.** Herald bits are not part of the
  transfer-matrix state. Each herald modifies the kernels of the slots that
  hold its components. The component signatures come from a noiseless *probe*
  circuit: a `stim.FlipSimulator` with one lane per (location, qubit, X or Z).
  Shots are grouped per slot by their pattern of active heralds, and the
  modified transfer matrices are cached.
- **Validation (VN).**
  - Without heralds it equals the lead's MLD to 4e-15 in the LLR.
  - With heralds it equals brute-force conditional inference to 4e-15
    (depths 0–2, with herald patterns from the true circuit).
  - It is calibrated on the true sampled channel. χ² of predicted against
    empirical P(L | syndrome, heralds) over frequent patterns gives p-values
    0.31, 0.32 and 0.92. The overall error rate is 0.23880 against 0.23895
    predicted, with ±0.00049 shot noise.
  - Probe signatures reproduce the lead's DEM mechanism by mechanism: same
    set, first-order probabilities within 2e-5 relative at p = 1e-6.

### 1.3 Policies scored on identical shots (erasure study)

| name | rule |
|---|---|
| `lrbd_unif_1`, `lrbd_const_1` | repo LRB-D on raw check syndromes; heralds ignored |
| `lrbd_unif_1_final` (u1f) | unif=1 plus the two final X detectors from the data5 readout (the fair comparator for MLD) |
| `lrbd_unif_1_final_noherald` | u1f and additionally reject any shot with a herald |
| `hybrid` | accept iff the full syndrome is explained by heralded erasures alone with an unambiguous logical class, and apply that class ("correct heralded, reject unexplained", the `[[4,2,2]]` rule of Zhang2026/Reichardt2024 as cited in `../lit/05`) |
| `mld` | the lead's exact MLD (heralds ignored, i.e. exactly `mld5`) |
| `mld_herald` | exact herald-conditioned MLD (full correction) |
| gap postselection | accept iff \|LLR\| of `mld_herald` ≥ g; reported at g = 2, 4, 6 and at the g that matches u1f's acceptance |

`hybrid` is computed exactly with a second `HeraldedMLD` that has no base
noise. At r = 0 it reduces to u1f, and `mld_herald` reduces to `mld`. Both
reductions hold shot for shot (test).

### 1.4 Leakage model and simulator (VN)

The model is `leakage.LeakageModel`.

- **Leak events.** After every noisy two-qubit gate (check CNOTs, logical
  SWAP/CZ; the ideal encoder never leaks), each qubit leaks with probability
  κp.
- **Effect of a leaked qubit.** It gives every two-qubit-gate partner a
  uniformly random Pauli, including at the gate where the leak happens. It
  reads out as the fixed value 1.
- **Return.**
  - The ancilla reset returns leaked ancillas.
  - With `lru=True`, a leakage-reduction unit returns leaked data qubits at
    the start of every forward round's reset layer. It returns them in a
    maximally mixed state and, as a leakage-detection unit, reports which
    qubits were leaked.
  - Without LRU, a leaked data qubit stays leaked until the terminal readout.
    Seepage is set to 0.
- **Flags.**
  - Three-state readout flags a leaked ancilla when it is measured.
  - Three-state terminal readout flags leaked data.
  - The LRU flags the data qubits it returns.

`leakage.simulate_leakage` is a numpy Pauli-frame simulator that steps
through the lead's own circuit and keeps per-shot leakage masks. It follows
Stim's conventions: frames are relative to `reference_sample`; Z frames are
randomized at initialization, reset and measurement; noise positions are
exact Bernoulli draws via geometric gaps. A leaked qubit's readout is forced
by setting its X frame to `value ⊕ reference`.

`stim.FlipSimulator` was tried first. Its `broadcast_pauli_errors` costs
about 10 ms per call at 5e4 shots, which made per-gate bookkeeping 20 to 30
times slower.

Validation (VN, tests):

- **Leakage switched off (κ = 0).** The simulator reproduces Stim's sampler:
  detector rates within |z| < 3 over 94 detectors, observable within 2.3σ,
  pair correlations, and LRB-D acceptance.
- **Data qubit 1 leaked from the start.** Every round syndrome is uniform.
- **The same with an LRU.** Rounds 0 and 1 are random and later rounds are
  clean.
- **Ancilla 5 that never returns.** It reads 1 in every round.
- **Data qubit 0 leaked.** The logical readout becomes uniformly random.
- **Leak rate.** The number of leak events per shot matches 2 × (number of
  noisy two-qubit gates) × κp to within 5%.

Configurations (identical shots for every method within a configuration):

- `none` (κ = 0);
- `k0.1` and `k0.5` (no LRU);
- `k0.1_lru` and `k0.5_lru` (heralded LRU every round).

Methods:

- LRB-D `unif_1`, `unif_1_final` and `const_1`;
- `*_flag` versions, which also reject any shot with a leakage flag;
- `mld`: the lead's exact MLD, which knows nothing about leakage;
- `mld_flag`: `mld` on the shots without a leakage flag.

### 1.5 Flags and single-fault enumeration (VA, VN)

Each weight-3 check gets one flag qubit (9–12), coupled by two flag CNOTs and
measured in the same layer as its ancilla. That adds no measurement layer, so
data idle noise is unchanged.

- **Layouts.**
  - `standard`, d1 F d2 F d3, is the usual "after the first, before the last"
    placement.
  - `wrap`, F d1 d2 F d3, flags every ancilla fault that spreads to data.
- **Noise.** `noisy` puts SI1000 noise on the flag gates, measurements, resets
  and idles. `ideal` adds pure information.
- **Decoding.** Flag detectors sit in their round's slot, which makes 8
  detectors per slot, so the lead's `TransferMatrixMLD` decodes flagged
  circuits exactly (VN: equal to brute force to 4e-15 at depth 1, 18
  detectors).
- **Single-fault enumeration.** For every noise location and every Pauli in
  its channel, the signature is taken from the probe and the exact MLD's
  decision on that syndrome is recorded. A fault is *miscorrected* when the
  decision differs from its logical flip.
  - The miscorrected probability mass is the exact first-order MLD error c1·p.
    It equals the minority mass per syndrome class: VN (test), and identical
    to the lead's E1 of `L0` §4, 11.07p at depth 0 against 11.1p there.
  - A *hook* is a check-CNOT fault whose spreading ancilla component (Z/Y on
    a Z check, X/Y on an X check) occurs before at least two data CNOTs.

## 2. Erasure results

### 2.1 One erasure is correctable, at code capacity and at circuit level

- **Code capacity (VA, VN).** For every data qubit, X, Y and Z have three
  distinct non-zero syndromes. So one located error is always corrected
  (test `test_code_capacity_single_erasures_are_correctable`; cf. the
  distance-2 erasure bound in `../lit/05` finding 1).
- **Circuit level (VN).** I took every single heralded erasure location in
  the circuits: check CNOTs, logical SWAP/CZ and measurement-idle qubits, at
  depths 0–22 on 6 sequences, 14052 location instances in the `all_*`
  configurations. For each I enumerated every Pauli on the erased qubits,
  16 or 4 of them.
  - The herald-aware MLD returns the correct logical class every time, with
    no other fault present.
  - The heralds-only explanation used by `hybrid` is never ambiguous.

  A two-qubit-gate erasure covers a data qubit and an ancilla, or data
  qubits 2 and 3. Ancilla hooks reduce to the erased data qubit's own
  Paulis up to stabilizers, and {2,3} is one of the six correctable erasure
  pairs, so this is expected (VA, consistent with `../lit/05` §3a).

So every heralded error contributes zero to the first-order MLD error. What
remains is the unheralded part.

#### Table E1. Exact first-order error of the herald-aware MLD (c1, units of p; mean [SD] over 6 sequences), first-order rejection of the hybrid rule (unheralded detected fault mass R1, units of p) and the single-erasure check

| config | c1 m=0 | c1 m=10 | c1 m=22 | bulk slope (c1(22)−c1(10))/12 | Δc1/c1 m=22 (paired) | R1 hybrid m=10 | heralds per shot / p, m=10 | single heralded erasures miscorrected (all depths and sequences) |
|---|---|---|---|---|---|---|---|---|
| r0 | 11.07 [0.00] | 134.88 [16.54] | 272.30 [10.61] | 11.45 ± 0.28 | +0.00% ± 0.00% | 1125 | 0 | 0 of 0 (0 ambiguous) |
| 2q_r0.5 | 10.20 [0.00] | 120.67 [15.96] | 244.55 [10.20] | 10.32 ± 0.26 | -10.20% ± 0.19% | 1046 | 88 | 0 of 4452 (0 ambiguous) |
| 2q_r0.9 | 9.51 [0.00] | 108.94 [15.44] | 220.73 [9.73] | 9.32 ± 0.24 | -18.95% ± 0.28% | 983 | 159 | 0 of 4452 (0 ambiguous) |
| 2q_r1 | 9.33 [0.00] | 105.99 [15.31] | 214.69 [9.61] | 9.06 ± 0.24 | -21.17% ± 0.31% | 967 | 176 | 0 of 4452 (0 ambiguous) |
| all_r0.5 | 5.53 [0.00] | 78.42 [9.84] | 159.81 [6.33] | 6.78 ± 0.17 | -41.31% ± 0.07% | 787 | 558 | 0 of 14052 (0 ambiguous) |
| all_r0.9 | 1.11 [0.00] | 32.87 [4.40] | 68.38 [2.79] | 2.96 ± 0.08 | -74.89% ± 0.06% | 518 | 1004 | 0 of 14052 (0 ambiguous) |
| all_r1 | 0.00 [0.00] | 21.32 [3.06] | 45.24 [1.94] | 1.99 ± 0.06 | -83.39% ± 0.06% | 450 | 1115 | 0 of 14052 (0 ambiguous) |

Residual c1 for 2q_r1 at m = 22, by kind: gate1_logical 6.02, meas_idle(data) 168.89, reset_idle(data) 39.78.
Residual c1 for all_r1 at m = 22, by kind: gate1_logical 5.46, reset_idle(data) 39.78.


- **Reading E1 (VN, exact).**
  - Heralding all two-qubit gates removes their whole contribution
    (check CNOTs, hooks and logical SWAP/CZ: 21% of c1).
  - What remains at r = 1 is data idling in the measurement layers
    (168.9p of 214.7p at m = 22), data idling in the reset layer (39.8p) and
    one-qubit gates (6.0p).
  - Also heralding the measurement-idle errors leaves data idling during
    reset (39.8p) and one-qubit gates (5.5p). That is 45.2p at m = 22, or
    1.99(6)p per step instead of 11.45(28)p.
  - The hybrid rule's first-order rejection R1 is the unheralded detected
    fault mass. It falls from 1125p (LRB-D, r = 0) to 450p at m = 10 for
    `all_r1`.

### 2.2 Full correction and paired comparisons (VN)

- **Same shots for both decoders.** `mld` and `mld_herald` decode the same
  shots. At r = 0 they agree shot for shot (test).
- **The marginal channel does not change with r.** `mld`'s F and fitted f
  are statistically identical across configurations, a built-in consistency
  check: f = 0.9879 at 5e-4 (all configurations), 0.9479–0.9482 at 2e-3 and 0.8566–0.8580
  at 5e-3.
- **The gains match first order.** The relative gain in error per step
  ε = (1 − f)/2 at p = 5e-4 is 21.5(1.8)% for `2q_r1` and 81.0(0.9)% for
  `all_r1`. The first-order predictions from Table E1 are 21% and 83%.

#### Table E2. Full correction: F of `mld` (heralds ignored) and `mld_herald`, paired ΔF = F(mld_herald) − F(mld) on identical shots, fitted decay f (F = b f^m)

| p | config | n_seq | m=0: F mld / F mld_herald / ΔF | m=4: F mld / F mld_herald / ΔF | m=10: F mld / F mld_herald / ΔF | m=22: F mld / F mld_herald / ΔF | f mld | f mld_herald |
|---|---|---|---|---|---|---|---|---|
| 0.0005 | r0 | 12 | 0.989(0) / 0.989(0) / 0.000(0) | 0.941(2) / 0.941(2) / 0.000(0) | 0.873(4) / 0.873(4) / 0.000(0) | 0.758(4) / 0.758(4) / 0.000(0) | 0.9879(2) | 0.9879(2) |
| 0.0005 | 2q_r0.5 | 12 | 0.989(0) / 0.990(0) / 0.001(0) | 0.940(2) / 0.946(2) / 0.006(0) | 0.874(2) / 0.887(3) / 0.013(1) | 0.756(3) / 0.778(3) / 0.021(1) | 0.9879(2) | 0.9891(2) |
| 0.0005 | 2q_r0.9 | 12 | 0.989(0) / 0.990(0) / 0.002(0) | 0.939(2) / 0.950(2) / 0.010(0) | 0.872(3) / 0.894(3) / 0.022(1) | 0.757(3) / 0.799(3) / 0.041(1) | 0.9879(2) | 0.9902(2) |
| 0.0005 | 2q_r1 | 12 | 0.989(0) / 0.991(0) / 0.002(0) | 0.940(2) / 0.952(2) / 0.012(0) | 0.874(3) / 0.900(3) / 0.026(1) | 0.757(2) / 0.804(3) / 0.047(1) | 0.9879(1) | 0.9905(2) |
| 0.0005 | all_r0.5 | 12 | 0.988(0) / 0.994(0) / 0.006(0) | 0.941(2) / 0.966(1) / 0.025(1) | 0.873(3) / 0.923(2) / 0.050(1) | 0.756(3) / 0.848(2) / 0.092(2) | 0.9879(2) | 0.9928(1) |
| 0.0005 | all_r0.9 | 12 | 0.989(0) / 0.999(0) / 0.010(0) | 0.941(2) / 0.986(1) / 0.045(2) | 0.875(3) / 0.966(1) / 0.091(2) | 0.758(4) / 0.929(1) / 0.170(3) | 0.9879(2) | 0.9967(1) |
| 0.0005 | all_r1 | 12 | 0.989(0) / 1.000(0) / 0.011(0) | 0.940(2) / 0.990(0) / 0.050(2) | 0.874(3) / 0.978(1) / 0.103(2) | 0.758(4) / 0.952(1) / 0.194(3) | 0.9879(2) | 0.9977(1) |
| 0.002 | r0 | 12 | 0.956(1) / 0.956(1) / 0.000(0) | 0.766(6) / 0.766(6) / 0.000(0) | 0.554(7) / 0.554(7) / 0.000(0) | 0.297(5) / 0.297(5) / 0.000(0) | 0.9479(7) | 0.9479(7) |
| 0.002 | 2q_r0.5 | 12 | 0.956(1) / 0.960(1) / 0.003(0) | 0.766(7) / 0.787(7) / 0.021(1) | 0.555(6) / 0.588(7) / 0.033(1) | 0.297(5) / 0.333(5) / 0.037(1) | 0.9482(8) | 0.9531(7) |
| 0.002 | 2q_r0.9 | 12 | 0.957(0) / 0.963(0) / 0.006(0) | 0.768(6) / 0.805(6) / 0.037(1) | 0.556(8) / 0.617(7) / 0.061(2) | 0.297(5) / 0.365(5) / 0.068(2) | 0.9480(8) | 0.9571(7) |
| 0.002 | 2q_r1 | 12 | 0.955(1) / 0.962(1) / 0.007(0) | 0.767(6) / 0.806(6) / 0.039(1) | 0.556(8) / 0.625(6) / 0.069(2) | 0.296(4) / 0.380(5) / 0.085(2) | 0.9480(7) | 0.9586(7) |
| 0.002 | all_r0.5 | 12 | 0.956(1) / 0.977(0) / 0.021(0) | 0.765(6) / 0.852(4) / 0.087(2) | 0.554(7) / 0.697(6) / 0.143(1) | 0.299(5) / 0.476(4) / 0.178(2) | 0.9480(9) | 0.9676(5) |
| 0.002 | all_r0.9 | 12 | 0.955(1) / 0.995(0) / 0.039(1) | 0.767(7) / 0.928(3) / 0.161(4) | 0.557(6) / 0.839(3) / 0.282(4) | 0.298(5) / 0.687(3) / 0.389(2) | 0.9482(8) | 0.9834(3) |
| 0.002 | all_r1 | 12 | 0.956(1) / 0.999(0) / 0.043(1) | 0.767(7) / 0.947(2) / 0.180(5) | 0.557(8) / 0.877(3) / 0.320(5) | 0.299(5) / 0.757(3) / 0.458(3) | 0.9481(8) | 0.9873(2) |
| 0.005 | r0 | 12 | 0.891(1) / 0.891(1) / 0.000(0) | 0.478(11) / 0.478(11) / 0.000(0) | 0.188(6) / 0.188(6) / 0.000(0) | 0.032(2) / 0.032(2) / 0.000(0) | 0.8579(21) | 0.8579(21) |
| 0.005 | 2q_r0.5 | 12 | 0.893(1) / 0.901(1) / 0.008(0) | 0.481(10) / 0.514(10) / 0.033(1) | 0.191(7) / 0.221(7) / 0.030(1) | 0.031(2) / 0.039(2) / 0.008(1) | 0.8575(21) | 0.8687(17) |
| 0.005 | 2q_r0.9 | 12 | 0.892(1) / 0.906(1) / 0.014(0) | 0.481(10) / 0.540(11) / 0.059(2) | 0.188(6) / 0.244(7) / 0.056(3) | 0.030(2) / 0.053(2) / 0.023(2) | 0.8566(20) | 0.8790(15) |
| 0.005 | 2q_r1 | 12 | 0.893(1) / 0.908(1) / 0.015(1) | 0.477(11) / 0.544(11) / 0.067(1) | 0.188(6) / 0.249(8) / 0.061(4) | 0.034(2) / 0.059(4) / 0.025(3) | 0.8576(19) | 0.8818(18) |
| 0.005 | all_r0.5 | 12 | 0.892(1) / 0.942(0) / 0.050(1) | 0.475(10) / 0.623(8) / 0.148(2) | 0.192(6) / 0.339(8) / 0.147(2) | 0.031(2) / 0.099(3) / 0.068(3) | 0.8574(20) | 0.9033(13) |
| 0.005 | all_r0.9 | 12 | 0.892(1) / 0.983(0) / 0.092(1) | 0.481(11) / 0.768(7) / 0.287(4) | 0.192(6) / 0.533(7) / 0.341(3) | 0.030(2) / 0.255(5) / 0.225(5) | 0.8575(19) | 0.9410(8) |
| 0.005 | all_r1 | 12 | 0.893(1) / 0.994(0) / 0.101(1) | 0.481(11) / 0.809(7) / 0.328(5) | 0.191(6) / 0.597(7) / 0.405(3) | 0.031(1) / 0.329(6) / 0.299(6) | 0.8580(16) | 0.9508(7) |


### 2.3 Acceptance and postselection (VN)

#### Table E3. Postselected methods: acceptance (fraction of all shots) and accepted F

| p | config | m | acc u1f | F u1f | acc u1f+no-herald | acc hybrid | F hybrid | gap g* (= u1f acc) | F gap@g* | acc gap≥4 | F gap≥4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | r0 | 0 | 0.9657(4) | 1.0000(0) | 0.9657(4) | 0.9657(4) | 1.0000(0) | 11.5 | 1.0000(0) | 0.9871 | 0.9999(0) |
| 0.0005 | r0 | 4 | 0.7837(9) | 0.9999(0) | 0.7837(9) | 0.7837(9) | 0.9999(0) | 9.25 | 0.9999(0) | 0.9070 | 0.9992(1) |
| 0.0005 | r0 | 10 | 0.5721(15) | 0.9997(1) | 0.5721(15) | 0.5721(15) | 0.9997(1) | 8.5 | 0.9997(1) | 0.8008 | 0.9980(1) |
| 0.0005 | 2q_r1 | 0 | 0.9657(3) | 1.0000(0) | 0.9643(3) | 0.9706(3) | 1.0000(0) | 11.75 | 1.0000(0) | 0.9894 | 0.9999(0) |
| 0.0005 | 2q_r1 | 4 | 0.7823(8) | 0.9999(0) | 0.7787(8) | 0.8101(6) | 0.9996(1) | 9.75 | 0.9999(0) | 0.9403 | 0.9989(1) |
| 0.0005 | 2q_r1 | 10 | 0.5706(15) | 0.9997(1) | 0.5654(15) | 0.6183(14) | 0.9989(1) | 8.75 | 0.9998(0) | 0.8706 | 0.9974(2) |
| 0.0005 | all_r0.5 | 0 | 0.9651(3) | 1.0000(0) | 0.9528(4) | 0.9767(2) | 1.0000(0) | 9.75 | 1.0000(0) | 0.9930 | 1.0000(0) |
| 0.0005 | all_r0.5 | 4 | 0.7828(8) | 0.9998(0) | 0.7435(7) | 0.8439(6) | 0.9996(1) | 9.75 | 0.9999(0) | 0.9462 | 0.9993(1) |
| 0.0005 | all_r0.5 | 10 | 0.5709(8) | 0.9998(1) | 0.5105(9) | 0.6761(7) | 0.9992(1) | 9.25 | 0.9999(0) | 0.8812 | 0.9983(1) |
| 0.0005 | all_r1 | 0 | 0.9654(4) | 1.0000(0) | 0.9408(6) | 0.9884(2) | 1.0000(0) | 39.75 | 1.0000(0) | 0.9999 | 1.0000(0) |
| 0.0005 | all_r1 | 4 | 0.7813(8) | 0.9999(0) | 0.7045(11) | 0.9066(6) | 0.9998(0) | 13 | 1.0000(0) | 0.9879 | 0.9998(0) |
| 0.0005 | all_r1 | 10 | 0.5684(15) | 0.9998(0) | 0.4567(15) | 0.7976(11) | 0.9996(0) | 12 | 1.0000(0) | 0.9706 | 0.9995(1) |
| 0.002 | r0 | 0 | 0.8690(4) | 0.9998(1) | 0.8690(4) | 0.8690(4) | 0.9998(1) | 8.75 | 0.9998(1) | 0.9443 | 0.9992(1) |
| 0.002 | r0 | 4 | 0.3745(14) | 0.9978(2) | 0.3745(14) | 0.3745(14) | 0.9978(2) | 6.5 | 0.9978(2) | 0.5946 | 0.9931(3) |
| 0.002 | r0 | 10 | 0.1071(6) | 0.9954(7) | 0.1071(6) | 0.1071(6) | 0.9954(7) | 5.75 | 0.9954(7) | 0.2789 | 0.9869(6) |
| 0.002 | 2q_r1 | 0 | 0.8694(7) | 0.9997(1) | 0.8641(7) | 0.8876(6) | 0.9994(1) | 9 | 0.9998(1) | 0.9546 | 0.9989(1) |
| 0.002 | 2q_r1 | 4 | 0.3766(17) | 0.9979(2) | 0.3690(18) | 0.4351(12) | 0.9929(4) | 7 | 0.9984(2) | 0.6385 | 0.9934(3) |
| 0.002 | 2q_r1 | 10 | 0.1085(9) | 0.9959(5) | 0.1042(8) | 0.1512(7) | 0.9830(8) | 6.25 | 0.9972(4) | 0.3292 | 0.9880(4) |
| 0.002 | all_r0.5 | 0 | 0.8704(5) | 0.9996(1) | 0.8260(5) | 0.9122(5) | 0.9995(1) | 8.25 | 0.9999(0) | 0.9724 | 0.9992(1) |
| 0.002 | all_r0.5 | 4 | 0.3753(13) | 0.9980(2) | 0.3035(11) | 0.5093(11) | 0.9949(3) | 7.5 | 0.9994(1) | 0.7215 | 0.9945(4) |
| 0.002 | all_r0.5 | 10 | 0.1072(7) | 0.9936(6) | 0.0686(5) | 0.2122(10) | 0.9852(6) | 6.75 | 0.9975(3) | 0.4369 | 0.9880(4) |
| 0.002 | all_r1 | 0 | 0.8693(6) | 0.9997(0) | 0.7832(8) | 0.9537(4) | 1.0000(0) | 39.75 | 1.0000(0) | 0.9991 | 1.0000(0) |
| 0.002 | all_r1 | 4 | 0.3781(12) | 0.9974(2) | 0.2478(9) | 0.6771(15) | 0.9965(2) | 10 | 0.9999(0) | 0.9082 | 0.9971(2) |
| 0.002 | all_r1 | 10 | 0.1072(7) | 0.9951(5) | 0.0437(5) | 0.4028(13) | 0.9919(5) | 9.25 | 0.9998(1) | 0.7785 | 0.9927(4) |
| 0.005 | r0 | 0 | 0.7077(7) | 0.9977(1) | 0.7077(7) | 0.7077(7) | 0.9977(1) | 5.75 | 0.9977(2) | 0.8399 | 0.9962(2) |
| 0.005 | r0 | 4 | 0.0887(8) | 0.9830(15) | 0.0887(8) | 0.0887(8) | 0.9830(15) | 4.5 | 0.9826(13) | 0.1330 | 0.9802(13) |
| 0.005 | r0 | 10 | 0.0038(1) | 0.9554(83) | 0.0038(1) | 0.0038(1) | 0.9554(83) | 3.75 | 0.9597(85) | 0.0035 | 0.9563(77) |
| 0.005 | 2q_r1 | 0 | 0.7084(12) | 0.9980(1) | 0.6976(13) | 0.7478(8) | 0.9960(2) | 7 | 0.9987(1) | 0.8407 | 0.9969(1) |
| 0.005 | 2q_r1 | 4 | 0.0878(8) | 0.9834(18) | 0.0831(8) | 0.1316(8) | 0.9484(31) | 5 | 0.9871(14) | 0.1606 | 0.9846(8) |
| 0.005 | 2q_r1 | 10 | 0.0039(1) | 0.9753(65) | 0.0035(1) | 0.0100(2) | 0.8778±0.0109 | 4.25 | 0.9800(66) | 0.0081 | 0.9807(40) |
| 0.005 | all_r0.5 | 0 | 0.7074(9) | 0.9981(2) | 0.6182(10) | 0.7969(10) | 0.9966(2) | 7 | 0.9995(1) | 0.9097 | 0.9967(2) |
| 0.005 | all_r0.5 | 4 | 0.0884(7) | 0.9849(11) | 0.0513(7) | 0.1926(12) | 0.9585(21) | 5.5 | 0.9942(7) | 0.2283 | 0.9856(6) |
| 0.005 | all_r0.5 | 10 | 0.0039(2) | 0.9705(65) | 0.0013(1) | 0.0229(4) | 0.9140(67) | 5 | 0.9901(37) | 0.0204 | 0.9827(28) |
| 0.005 | all_r1 | 0 | 0.7066(8) | 0.9980(1) | 0.5418(7) | 0.8890(6) | 0.9998(0) | 39.75 | 1.0000(0) | 0.9910 | 1.0000(0) |
| 0.005 | all_r1 | 4 | 0.0881(7) | 0.9844(11) | 0.0300(4) | 0.3747(19) | 0.9751(15) | 8.25 | 0.9998(1) | 0.5794 | 0.9891(5) |
| 0.005 | all_r1 | 10 | 0.0040(1) | 0.9671(75) | 0.0004(0) | 0.1020(8) | 0.9352(28) | 7.5 | 1.0000(0) | 0.1880 | 0.9799(11) |


- **The hybrid rule** ("correct heralded, reject unexplained", the
  `[[4,2,2]]` rule of `../lit/05` findings 4 and 6) accepts more shots than
  LRB-D on the same shots.
  - At p = 2e-3, m = 10 the acceptance is ×1.41 for `2q_r1`, ×1.98 for
    `all_r0.5` and ×3.76 for `all_r1`. At p = 5e-3, m = 4 it is ×4.2 for
    `all_r1`.
  - Its accepted F is somewhat lower: 0.9830(8), 0.9852(6) and 0.9919(5)
    against LRB-D's 0.9954(7). It accepts heralded shots, and a heralded
    erasure combined with an unheralded fault whose syndrome falls in the
    erasure's span is miscorrected at second order.
- **Rejecting heralds as well as syndromes wastes shots.** LRB-D acceptance
  drops by up to 2.5 times (0.044 against 0.107 for `all_r1`) with no gain in
  F.
- **Herald-aware gap postselection dominates both.**
  - Threshold the exact herald-conditioned |LLR| where its acceptance equals
    LRB-D's.
  - At r = 0 this reproduces LRB-D exactly: 0.9954(7) against 0.9954(7),
    the frontier result of `L0` §7.
  - With heralds it is strictly better at the same acceptance: 0.9972(4)
    against 0.9959(5) for `2q_r1`, 0.9975(3) against 0.9936(6) for
    `all_r0.5`, and 0.9998(1) against 0.9951(5) for `all_r1` (p = 2e-3,
    m = 10).
  - At |LLR| ≥ 4 the acceptance is 0.28 for r0, 0.33 for `2q_r1`, 0.44 for
    `all_r0.5` and 0.78 for `all_r1`, at accepted F between 0.987 and 0.993.

## 3. Leakage results (VN; 12 sequences × 4e4 shots per circuit)

#### Table L1. Decoded LRB (`mld`, leakage-unaware exact MLD) under leakage

| p | config | n_seq | F m=0 | F m=4 | F m=10 | F m=22 | F/F_none m=10 | F/F_none m=22 | f all | f (m≤10) | f (m≥10) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | none | 12 | 0.989(0) | 0.940(2) | 0.874(3) | 0.759(3) | 1.000(4) | 1.000(5) | 0.9879(2) | 0.9876(4) | 0.9882(4) |
| 0.0005 | k0.1 | 12 | 0.988(0) | 0.936(2) | 0.865(3) | 0.740(3) | 0.990(4) | 0.975(5) | 0.9869(2) | 0.9867(3) | 0.9871(3) |
| 0.0005 | k0.1_lru | 12 | 0.989(0) | 0.937(2) | 0.865(3) | 0.744(3) | 0.990(5) | 0.980(5) | 0.9870(2) | 0.9867(4) | 0.9873(3) |
| 0.0005 | k0.5 | 12 | 0.987(0) | 0.918(2) | 0.827(3) | 0.671(3) | 0.947(5) | 0.884(5) | 0.9826(2) | 0.9824(5) | 0.9826(4) |
| 0.0005 | k0.5_lru | 12 | 0.987(0) | 0.923(2) | 0.835(4) | 0.691(3) | 0.955(5) | 0.910(5) | 0.9838(2) | 0.9833(5) | 0.9843(5) |
| 0.002 | none | 12 | 0.956(0) | 0.769(7) | 0.557(7) | 0.296(4) | 1.000(17) | 1.000(21) | 0.9480(8) | 0.9471(14) | 0.9489(14) |
| 0.002 | k0.1 | 12 | 0.954(0) | 0.753(7) | 0.530(7) | 0.269(4) | 0.952(18) | 0.909(20) | 0.9440(7) | 0.9430(14) | 0.9452(13) |
| 0.002 | k0.1_lru | 12 | 0.955(1) | 0.757(7) | 0.535(7) | 0.275(4) | 0.962(17) | 0.927(20) | 0.9446(7) | 0.9435(14) | 0.9455(13) |
| 0.002 | k0.5 | 12 | 0.948(0) | 0.697(7) | 0.440(6) | 0.180(3) | 0.790(15) | 0.608(13) | 0.9270(8) | 0.9261(16) | 0.9281(15) |
| 0.002 | k0.5_lru | 12 | 0.947(0) | 0.708(8) | 0.462(6) | 0.199(3) | 0.830(15) | 0.673(15) | 0.9314(8) | 0.9300(15) | 0.9329(14) |
| 0.005 | none | 12 | 0.892(0) | 0.479(11) | 0.190(6) | 0.032(1) | 1.000(43) | 1.000(49) | 0.8584(18) | 0.8560(29) | 0.8615(20) |
| 0.005 | k0.1 | 12 | 0.888(1) | 0.455(10) | 0.166(6) | 0.024(2) | 0.875(40) | 0.768(57) | 0.8481(16) | 0.8460(31) | 0.8515(40) |
| 0.005 | k0.1_lru | 12 | 0.887(0) | 0.458(11) | 0.169(6) | 0.024(2) | 0.891(41) | 0.757(78) | 0.8500(21) | 0.8475(31) | 0.8529(45) |
| 0.005 | k0.5 | 12 | 0.873(1) | 0.371(9) | 0.104(4) | 0.008(2) | 0.549(27) | 0.243(52) | 0.8109(27) | 0.8087(35) | 0.8283(83) |
| 0.005 | k0.5_lru | 12 | 0.872(1) | 0.382(11) | 0.115(4) | 0.010(1) | 0.607(28) | 0.305(47) | 0.8165(26) | 0.8152(34) | 0.8229(79) |


#### Table L2. Postselection under leakage (m = 10 and 22): acceptance and accepted F

| p | config | m | acc u1f | F u1f | acc u1f+flag | F const1 | acc const1 | F const1+flag | acc const1+flag | F mld+flag | acc mld+flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | none | 10 | 0.5713(12) | 0.9997(0) | 0.5713(12) | 0.992(0) | 0.7031(12) | 0.992(0) | 0.7031(12) | 0.874(3) | 1.0000(0) |
| 0.0005 | none | 22 | 0.3047(6) | 0.9995(1) | 0.3047(6) | 0.978(0) | 0.4892(10) | 0.978(0) | 0.4892(10) | 0.759(3) | 1.0000(0) |
| 0.0005 | k0.1 | 10 | 0.5618(11) | 0.9997(1) | 0.5617(11) | 0.990(0) | 0.6944(12) | 0.992(0) | 0.6928(12) | 0.874(2) | 0.9839(3) |
| 0.0005 | k0.1 | 22 | 0.2953(9) | 0.9995(1) | 0.2952(9) | 0.973(0) | 0.4755(9) | 0.977(0) | 0.4724(9) | 0.758(3) | 0.9662(5) |
| 0.0005 | k0.1_lru | 10 | 0.5612(9) | 0.9997(1) | 0.5609(9) | 0.991(0) | 0.6935(11) | 0.992(0) | 0.6915(11) | 0.873(3) | 0.9839(3) |
| 0.0005 | k0.1_lru | 22 | 0.2941(6) | 0.9991(1) | 0.2937(6) | 0.974(0) | 0.4741(8) | 0.977(0) | 0.4707(9) | 0.758(3) | 0.9661(3) |
| 0.0005 | k0.5 | 10 | 0.5279(16) | 0.9996(0) | 0.5275(16) | 0.985(0) | 0.6567(18) | 0.992(0) | 0.6488(18) | 0.874(3) | 0.9216(9) |
| 0.0005 | k0.5 | 22 | 0.2576(9) | 0.9989(1) | 0.2573(9) | 0.956(1) | 0.4257(13) | 0.977(0) | 0.4119(15) | 0.756(3) | 0.8422(12) |
| 0.0005 | k0.5_lru | 10 | 0.5282(15) | 0.9994(1) | 0.5268(15) | 0.988(0) | 0.6583(14) | 0.992(0) | 0.6493(14) | 0.873(3) | 0.9222(8) |
| 0.0005 | k0.5_lru | 22 | 0.2574(8) | 0.9986(2) | 0.2558(8) | 0.963(1) | 0.4250(13) | 0.977(0) | 0.4094(14) | 0.759(3) | 0.8411(10) |
| 0.002 | none | 10 | 0.1071(9) | 0.9961(4) | 0.1071(9) | 0.899(2) | 0.2721(16) | 0.899(2) | 0.2721(16) | 0.557(7) | 1.0000(0) |
| 0.002 | none | 22 | 0.0087(1) | 0.9902(28) | 0.0087(1) | 0.558(5) | 0.1053(8) | 0.558(5) | 0.1053(8) | 0.296(4) | 1.0000(0) |
| 0.002 | k0.1 | 10 | 0.1003(5) | 0.9951(6) | 0.1003(5) | 0.885(2) | 0.2595(12) | 0.896(2) | 0.2548(12) | 0.555(7) | 0.9368(6) |
| 0.002 | k0.1 | 22 | 0.0075(1) | 0.9867(16) | 0.0075(1) | 0.525(6) | 0.1003(4) | 0.564(6) | 0.0919(4) | 0.298(5) | 0.8701(10) |
| 0.002 | k0.1_lru | 10 | 0.1007(7) | 0.9947(4) | 0.1005(7) | 0.887(1) | 0.2585(15) | 0.897(1) | 0.2532(15) | 0.556(7) | 0.9361(8) |
| 0.002 | k0.1_lru | 22 | 0.0075(2) | 0.9897(21) | 0.0075(2) | 0.521(5) | 0.1006(5) | 0.554(5) | 0.0923(5) | 0.296(4) | 0.8721(10) |
| 0.002 | k0.5 | 10 | 0.0776(9) | 0.9947(7) | 0.0774(9) | 0.836(2) | 0.2163(16) | 0.901(2) | 0.1957(16) | 0.554(7) | 0.7220(26) |
| 0.002 | k0.5 | 22 | 0.0043(1) | 0.9861(35) | 0.0042(1) | 0.377(4) | 0.0847(4) | 0.555(4) | 0.0528(5) | 0.301(4) | 0.5014(25) |
| 0.002 | k0.5_lru | 10 | 0.0776(7) | 0.9940(3) | 0.0766(7) | 0.849(3) | 0.2167(15) | 0.900(2) | 0.1948(15) | 0.557(6) | 0.7201(25) |
| 0.002 | k0.5_lru | 22 | 0.0046(1) | 0.9866(35) | 0.0045(1) | 0.388(5) | 0.0850(4) | 0.561(6) | 0.0529(5) | 0.298(4) | 0.5013(25) |
| 0.005 | none | 10 | 0.0038(1) | 0.9785(52) | 0.0038(1) | 0.398(6) | 0.0864(4) | 0.398(6) | 0.0864(4) | 0.190(6) | 1.0000(0) |
| 0.005 | none | 22 | 0.0000(0) | 1.0000(0) | 0.0000(0) | 0.039(5) | 0.0634(3) | 0.039(5) | 0.0634(3) | 0.032(1) | 1.0000(0) |
| 0.005 | k0.1 | 10 | 0.0033(1) | 0.9707(53) | 0.0033(1) | 0.350(6) | 0.0821(5) | 0.387(7) | 0.0729(5) | 0.188(6) | 0.8499(14) |
| 0.005 | k0.1 | 22 | 0.0000(0) | 1.0000(0) | 0.0000(0) | 0.024(3) | 0.0626(4) | 0.029(5) | 0.0446(4) | 0.031(2) | 0.7084(17) |
| 0.005 | k0.1_lru | 10 | 0.0033(1) | 0.9656(66) | 0.0033(1) | 0.360(7) | 0.0832(6) | 0.392(7) | 0.0737(6) | 0.188(7) | 0.8497(15) |
| 0.005 | k0.1_lru | 22 | 0.0000(0) | n/a | 0.0000(0) | 0.018(4) | 0.0626(4) | 0.027(4) | 0.0444(3) | 0.031(3) | 0.7080(19) |
| 0.005 | k0.5 | 10 | 0.0018(1) | 0.9540(92) | 0.0018(1) | 0.232(7) | 0.0721(3) | 0.393(10) | 0.0382(6) | 0.189(6) | 0.4419(43) |
| 0.005 | k0.5 | 22 | 0.0000(0) | 1.0000(0) | 0.0000(0) | 0.018(4) | 0.0612(2) | 0.038(7) | 0.0112(2) | 0.032(4) | 0.1787(22) |
| 0.005 | k0.5_lru | 10 | 0.0019(0) | 0.9586(56) | 0.0018(0) | 0.245(7) | 0.0725(4) | 0.405(8) | 0.0384(5) | 0.191(6) | 0.4427(45) |
| 0.005 | k0.5_lru | 22 | 0.0000(0) | 1.0000(0) | 0.0000(0) | 0.015(4) | 0.0609(4) | 0.053(11) | 0.0113(2) | 0.040(4) | 0.1790(21) |


#### Table L3. Persistence diagnostics at m = 22

| p | config | leaked-data population after round 1 / 5 / 10 / 21 / at readout | detection autocorrelation lag 1 / 2 / 4 / 6 | shots flagged (m=22) |
|---|---|---|---|---|
| 0.0005 | none | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0037 / -0.0000 / 0.0000 / -0.0000 | 0.0000 |
| 0.0005 | k0.1 | 0.0017 / 0.0047 / 0.0083 / 0.0166 / 0.0206 | 0.0053 / 0.0015 / 0.0013 / 0.0012 | 0.0338 |
| 0.0005 | k0.1_lru | 0.0007 / 0.0008 / 0.0007 / 0.0007 / 0.0041 | 0.0039 / 0.0000 / 0.0000 / -0.0000 | 0.0339 |
| 0.0005 | k0.5 | 0.0081 / 0.0230 / 0.0410 / 0.0801 / 0.0980 | 0.0111 / 0.0069 / 0.0063 / 0.0056 | 0.1578 |
| 0.0005 | k0.5_lru | 0.0038 / 0.0038 / 0.0036 / 0.0040 / 0.0198 | 0.0044 / -0.0000 / -0.0000 / -0.0000 | 0.1589 |
| 0.002 | none | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0111 / 0.0000 / 0.0000 / 0.0000 | 0.0000 |
| 0.002 | k0.1 | 0.0065 / 0.0184 / 0.0329 / 0.0650 / 0.0798 | 0.0151 / 0.0041 / 0.0037 / 0.0033 | 0.1299 |
| 0.002 | k0.1_lru | 0.0031 / 0.0032 / 0.0029 / 0.0030 / 0.0163 | 0.0113 / 0.0000 / 0.0000 / 0.0000 | 0.1278 |
| 0.002 | k0.5 | 0.0305 / 0.0871 / 0.1524 / 0.2823 / 0.3378 | 0.0262 / 0.0155 / 0.0140 / 0.0126 | 0.4986 |
| 0.002 | k0.5_lru | 0.0151 / 0.0155 / 0.0146 / 0.0151 / 0.0766 | 0.0122 / -0.0000 / -0.0000 / -0.0001 | 0.4987 |
| 0.005 | none | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0152 / -0.0000 / 0.0000 / 0.0000 | 0.0000 |
| 0.005 | k0.1 | 0.0154 / 0.0449 / 0.0796 / 0.1530 / 0.1863 | 0.0191 / 0.0049 / 0.0044 / 0.0040 | 0.2916 |
| 0.005 | k0.1_lru | 0.0076 / 0.0077 / 0.0073 / 0.0077 / 0.0397 | 0.0153 / -0.0000 / -0.0000 / -0.0000 | 0.2920 |
| 0.005 | k0.5 | 0.0752 / 0.2047 / 0.3397 / 0.5648 / 0.6439 | 0.0241 / 0.0130 / 0.0117 / 0.0104 | 0.8213 |
| 0.005 | k0.5_lru | 0.0377 / 0.0380 / 0.0359 / 0.0370 / 0.1819 | 0.0151 / -0.0001 / -0.0000 / -0.0000 | 0.8210 |


#### Table L4. Is the leakage-induced decay exponential? Paired ratio R(m) = F_leak(m)/F_none(m) on the same 12 sequences (independent shots) and its local decay rate (R(m2)/R(m1))^(1/(m2−m1)); bootstrap SE over sequences (300 resamples). A constant local rate means an exponential leakage contribution

| p | config | method | R(4) | R(10) | R(22) | rate 0→4 | rate 4→10 | rate 10→22 |
|---|---|---|---|---|---|---|---|---|
| 0.0005 | k0.1 | mld | 0.9954(6) | 0.9897(9) | 0.9745(13) | 0.9990(2) | 0.9990(2) | 0.9987(1) |
| 0.0005 | k0.1_lru | mld | 0.9963(9) | 0.9903(12) | 0.9798(11) | 0.9990(2) | 0.9990(3) | 0.9991(1) |
| 0.0005 | k0.5 | mld | 0.9767(9) | 0.9469(16) | 0.8844(17) | 0.9946(2) | 0.9949(3) | 0.9943(2) |
| 0.0005 | k0.5_lru | mld | 0.9816(8) | 0.9554(17) | 0.9102(21) | 0.9959(2) | 0.9955(4) | 0.9960(3) |
| 0.0005 | k0.1 | lrbd_const_1 | 0.9996(3) | 0.9988(3) | 0.9954(6) | 0.9999(1) | 0.9999(1) | 0.9997(1) |
| 0.0005 | k0.1_lru | lrbd_const_1 | 0.9997(2) | 0.9993(2) | 0.9966(5) | 0.9999(1) | 0.9999(1) | 0.9998(0) |
| 0.0005 | k0.5 | lrbd_const_1 | 0.9976(3) | 0.9938(4) | 0.9775(6) | 0.9995(1) | 0.9994(1) | 0.9986(0) |
| 0.0005 | k0.5_lru | lrbd_const_1 | 0.9984(3) | 0.9960(3) | 0.9844(7) | 0.9997(1) | 0.9996(1) | 0.9990(1) |
| 0.002 | k0.1 | mld | 0.9795(11) | 0.9523(30) | 0.9093(39) | 0.9953(3) | 0.9953(5) | 0.9962(3) |
| 0.002 | k0.1_lru | mld | 0.9844(17) | 0.9619(20) | 0.9273(40) | 0.9963(5) | 0.9962(3) | 0.9970(5) |
| 0.002 | k0.5 | mld | 0.9071(18) | 0.7900(37) | 0.6079(24) | 0.9780(5) | 0.9772(7) | 0.9784(5) |
| 0.002 | k0.5_lru | mld | 0.9213(25) | 0.8296(27) | 0.6729(65) | 0.9819(7) | 0.9827(6) | 0.9827(9) |
| 0.002 | k0.1 | lrbd_const_1 | 0.9975(8) | 0.9839(13) | 0.9400(95) | 0.9995(2) | 0.9977(3) | 0.9962(8) |
| 0.002 | k0.1_lru | lrbd_const_1 | 0.9992(7) | 0.9862(13) | 0.9323(94) | 1.0000(3) | 0.9978(3) | 0.9953(8) |
| 0.002 | k0.5 | lrbd_const_1 | 0.9846(7) | 0.9295(20) | 0.6756(69) | 0.9966(2) | 0.9905(4) | 0.9738(9) |
| 0.002 | k0.5_lru | lrbd_const_1 | 0.9897(10) | 0.9437(20) | 0.6949(96) | 0.9980(3) | 0.9921(4) | 0.9748(11) |
| 0.005 | k0.1 | mld | 0.9499(39) | 0.8747±0.0102 | 0.7675±0.0605 | 0.9884(11) | 0.9863(18) | 0.9892(73) |
| 0.005 | k0.1_lru | mld | 0.9549(52) | 0.8912±0.0129 | 0.7574±0.0718 | 0.9901(14) | 0.9886(28) | 0.9865(79) |
| 0.005 | k0.5 | mld | 0.7747(46) | 0.5493(92) | 0.2430±0.0511 | 0.9433(16) | 0.9443(31) | 0.9343±0.0162 |
| 0.005 | k0.5_lru | mld | 0.7967(67) | 0.6069(90) | 0.3049±0.0444 | 0.9502(21) | 0.9557(29) | 0.9442±0.0115 |
| 0.005 | k0.1 | lrbd_const_1 | 0.9829(18) | 0.8799±0.0111 | 0.6076±0.1134 | 0.9960(4) | 0.9817(20) | 0.9696±0.0143 |
| 0.005 | k0.1_lru | lrbd_const_1 | 0.9857(29) | 0.9043±0.0147 | 0.4493±0.1484 | 0.9965(7) | 0.9857(29) | 0.9434±0.0251 |
| 0.005 | k0.5 | lrbd_const_1 | 0.9093(37) | 0.5845±0.0169 | 0.4721±0.1166 | 0.9777(9) | 0.9290(45) | 0.9824±0.0209 |
| 0.005 | k0.5_lru | lrbd_const_1 | 0.9240(30) | 0.6152±0.0190 | 0.3850±0.1189 | 0.9813(8) | 0.9345(52) | 0.9617±0.0254 |


### 3.1 What leakage does to each method

- **Decoded LRB loses fidelity, and the added decay is exponential.**
  - At κ = 0.5 and p = 2e-3, F_mld(22) is 0.180(3) against 0.296(4)
    without leakage (ratio 0.608(13)). The fitted f falls from 0.9480(8) to
    0.9270(8).
  - Table L4 gives the local decay rate of the paired ratio
    R(m) = F_leak/F_none. It is the same over 0→4, 4→10 and 10→22 within
    errors: 0.9780(5), 0.9772(7), 0.9784(5). It is equally flat at κ = 0.1
    and at p = 5e-4. The early (m ≤ 10) and late (m ≥ 10) fits of F_mld
    agree as well as they do without leakage (Table L1).
  - So persistence across rounds does **not** make the decoded decay
    non-exponential at these depths. A leaked data qubit gives random Paulis
    to the partners of all its later CNOTs, which scrambles the logical
    within a round or two. Longer persistence then cannot lose more, and
    each leak event costs a roughly fixed amount of fidelity.
  - Rough estimate (C): there are about 28κp leak opportunities per depth
    step, and −ln 0.978 ≈ 0.022 ≈ 0.8 × 28κp at κp = 1e-3. A leak costs about
    0.8 of the shot's F.
- **The LRU helps decoding only a little.**
  - The heralded LRU bounds the leaked-data population at 0.015, while
    without it the population grows linearly to 0.28 by round 21 (Table L3).
  - Yet F(22) is 0.199(3) with the LRU against 0.180(3) without, and the
    leakage-induced rate drops by only 20% (−ln 0.9825 = 0.018 against
    0.022).
  - The returned qubit is maximally mixed. For a d = 2 code that is an
    unheralded weight-1 error, which the exact MLD miscorrects at a rate
    comparable to a persistent leak. The LRU's herald is what helps, through
    the next point.
- **Rejecting shots with leakage flags restores the leakage-free decoded
  fidelity**, at the price of acceptance. The flags come from three-state
  readout of the ancillas and the final data, plus LRU heralds. `mld_flag`
  at κ = 0.5, p = 2e-3, m = 22 gives F = 0.301(4) against 0.296(4) without
  leakage, at acceptance 0.501(3). At κ = 0.1 it gives 0.298(5) at
  acceptance 0.870(1).
- **LRB-D unif=1 (+ final check) keeps its fidelity.** Leaked shots show
  random syndromes in every later round and are rejected.
  - The accepted F is unchanged within errors: 0.9947(7) against
    0.9961(4) at m = 10 (Table L2).
  - The acceptance ratio to the leakage-free run falls from 0.98 at
    m = 0 to 0.49 at m = 22 for κ = 0.5, and to 0.86 for κ = 0.1.
  - Flag rejection adds almost nothing on top, because the syndromes
    already reject almost every leaked shot.
- **Sparse-check LRB-D (const=1, terminal check only) loses fidelity.** A
  leaked-and-scrambled shot is accepted whenever its terminal syndrome is
  trivial.
  - At m = 22, p = 2e-3, κ = 0.5, F falls from 0.558(5) to 0.377(4).
  - Rejecting flagged shots restores it to 0.555(4), at acceptance 0.053
    against 0.085 (0.105 without leakage).
  - This is the one place where the decay is clearly non-exponential: the
    local rate of R(m) for const=1 is 0.9966(2), 0.9905(4) and 0.9738(9)
    over 0→4, 4→10 and 10→22 (VN).
  - With an LRU the acceleration is just as strong: 0.9980(3), 0.9921(4),
    0.9748(11). It therefore comes from const=1's syndrome-sector mixing
    (the drifting const=1 rates of `L0` §6), amplified by the extra error
    rate, and not from leakage persistence (C, supported by the LRU
    control).

### 3.2 Signatures of persistence (VN)

- **Leaked-data population (Table L3).** Without an LRU it grows linearly
  with the number of rounds: 0.031, 0.087, 0.152 and 0.282 after rounds 1, 5,
  10 and 21 (p = 2e-3, κ = 0.5), and 0.338 at the readout. With the LRU it is
  flat at 0.015. It jumps to 0.077 at the readout because the inverse half
  has no LRU.
- **Detection-event autocorrelation at lag ≥ 2.**
  - Without leakage it is 0.0000.
  - With leakage and no LRU it is 0.0155, 0.0140 and 0.0126 at lags 2, 4
    and 6 (κ = 0.5), and 0.0041, 0.0037 and 0.0033 at κ = 0.1.
  - With the LRU it returns to 0.0000.

  This is the depth-resolved fingerprint of persistent leakage. Lag-1
  correlations appear in every configuration, because a measurement error
  fires two consecutive difference detectors. Miao et al. use the same
  diagnostic on hardware (LIT via `../lit/05` finding 20; I did not re-fetch
  it).

## 4. Flag results

#### Table F1. Exact first-order MLD error c1 = E1/p: mean over 6 sequences [SD over sequences]. Decisions are those of the exact MLD with prior p = 1e-4; the last columns give the paired relative change versus no flags (mean ± SE over sequences)

| config | c1 m=0 | c1 m=2 | c1 m=4 | c1 m=6 | c1 m=10 | c1 m=22 | Δc1/c1 m=10 | Δc1/c1 m=22 | hook mass miscorrected /p, m=22 |
|---|---|---|---|---|---|---|---|---|---|
| none | 11.07 [0.00] | 36.84 [5.84] | 58.41 [10.03] | 83.97 [8.48] | 134.88 [16.54] | 272.30 [10.61] | +0.00% ± 0.00% | +0.00% ± 0.00% | 3.47 |
| ideal_standard | 10.80 [0.00] | 35.59 [5.40] | 56.13 [9.70] | 80.64 [8.14] | 129.21 [15.98] | 260.77 [10.46] | -4.22% ± 0.08% | -4.24% ± 0.09% | 4.03 |
| ideal_wrap | 10.53 [0.00] | 34.66 [5.30] | 54.73 [9.28] | 78.55 [7.98] | 125.79 [15.33] | 254.09 [10.03] | -6.73% ± 0.07% | -6.69% ± 0.08% | 0.00 |
| noisy_standard | 10.93 [0.00] | 36.32 [5.66] | 57.36 [9.96] | 82.47 [8.41] | 132.41 [16.48] | 267.21 [10.63] | -1.85% ± 0.08% | -1.87% ± 0.05% | 5.64 |
| noisy_wrap | 10.67 [0.00] | 35.17 [5.40] | 55.58 [9.43] | 79.77 [8.13] | 127.81 [15.64] | 258.29 [10.22] | -5.24% ± 0.06% | -5.15% ± 0.07% | 0.00 |

With prior p = 5e-3 instead of 1e-4, c1(m=10, no flags) changes from 134.88 to 135.58, because a few decision boundaries move. Bulk slope with no flags: (c1(22) − c1(10))/12 = 11.45 ± 0.28 per depth step.


#### Table F1b. c1 split by fault kind (m = 22, mean of 6 sequences, units of p)

| config | meas_idle(data) | reset_idle(data) | cx_check | cx_check(hook) | cx_flag | cx_flag(hook) | gate2_logical | gate1_logical | total c1 |
|---|---|---|---|---|---|---|---|---|---|
| none | 170.22 | 39.89 | 31.18 | 3.47 | 0.00 | 0.00 | 18.94 | 8.60 | 272.30 |
| ideal_wrap | 170.22 | 39.89 | 17.39 | 0.00 | 0.00 | 0.00 | 17.01 | 9.58 | 254.09 |
| noisy_wrap | 170.22 | 39.89 | 17.39 | 0.00 | 4.20 | 0.00 | 17.01 | 9.58 | 258.29 |
| ideal_standard | 170.22 | 39.89 | 20.37 | 4.03 | 0.00 | 0.00 | 17.54 | 8.71 | 260.77 |
| noisy_standard | 170.22 | 39.89 | 20.71 | 3.49 | 4.20 | 2.16 | 17.94 | 8.60 | 267.21 |

Total single-fault mass by kind at m = 22 (units of p), with the miscorrected fraction in brackets: meas_idle(data) 920.0 [19%], meas_idle(anc) 552.0 [0%], meas_flip 460.0 [0%], cx_check 226.9 [14%], reset_idle(data) 220.0 [18%], reset_flip 176.0 [0%], meas_depol(anc) 92.0 [0%], gate2_logical 70.0 [27%], cx_check(hook) 49.1 [7%], gate1_logical 37.5 [23%], gate1_check 9.2 [0%].


### 4.1 Which single faults the exact MLD miscorrects (VN, exact first order)

The first-order error is c1·p with c1 = 272.3p at m = 22 and 11.45(28)p per
depth step. This agrees with the lead's independent DEM computation in `L0`
§4: 270p and 11.8p over 30 sequences. The miscorrected faults are:

- **Data errors on the ambiguous pairs: 77%.**
  - Data idling in the four single-ancilla measurement layers: 170.2p, which
    is 19% of that fault mass.
  - Data idling in the reset layer: 39.9p, 18% of its mass.
  - These are X/Y/Z on data 0, 2, 3 and 4 whose equivalents form the pairs
    X0/X3, X2/X4, Z0/Z2 and Z3/Z4 with identical syndrome histories.
  - Timing resolves part of each class (an error between two check
    couplings is seen one round earlier or later), and the rest is
    fundamental to d = 2.
  - The largest single classes are idle Z2, X3, Z4 and Z3. MLD picks the
    heavier coset, and hook-equivalent faults make, for example, Z0's class
    the heavier one, so it is the bare Z2 that gets miscorrected.
- **Check-CNOT faults: 34.7p (13%).** 3.5p of it is hooks; 7% of all hook
  mass is miscorrected.
- **Logical SWAP/CZ(2,3) faults: 18.9p** (27% of their mass).
- **One-qubit logical gate faults: 8.6p** (23%).
- **Never miscorrected at first order (0%):** measurement flips (460p),
  ancilla idling (552p), reset flips (176p), measured-ancilla depolarizing
  (92p) and ancilla H gates. They give time-like detection pairs and are
  uniquely identified, as in `L0` §4.

The decisions barely depend on the prior: switching from p = 1e-4 to 5e-3
moves c1(10) from 134.88 to 135.58.

### 4.2 Would flag qubits on the weight-3 checks disambiguate them? (VN)

Only the CNOT part, which is small.

- **The `wrap` layout flags every ancilla fault that spreads to data.**
  - It removes all hook miscorrections (3.47p → 0 at m = 22).
  - It cuts the non-hook check-CNOT term by 44% (31.2p → 17.4p). The flag
    tells ancilla-spread errors apart from the data errors they mimic.
  - It leaves data idling untouched (210.1p).
  - Net change in c1 at m = 22, paired over sequences: −6.69(8)% with ideal
    flags, and −5.15(7)% with SI1000-noisy flags, whose own flag-CNOT faults
    add 4.2p.
- **The textbook `standard` layout** (d1 F d2 F d3) gives −4.24(9)% ideal and
  −1.87(5)% noisy.
  - It misses the dominant hook: an ancilla fault right after the first
    data CNOT occurs before the first flag CNOT.
  - Its miscorrected hook mass rises (3.47p → 4.03p ideal; 5.64p noisy,
    including 2.16p of flag-CNOT hooks) as decision boundaries move.
- **At finite p (Table F2).** These are pooled over m = 2..10 relative to no
  flags. The shots are independent, because the flagged circuit is a
  different physical circuit; the 6 sequences are the same.

  | p | ideal wrap | noisy wrap | ideal standard | noisy standard |
  |---|---|---|---|---|
  | 5e-4 | −7.1(1.4)% | −4.8(1.4)% | −3.7(1.4)% | −1.9(1.4)% |
  | 2e-3 | −5.5(0.7)% | −1.2(0.7)% | −2.9(0.7)% | +0.3(0.7)% |
  | 5e-3 | −3.4(0.4)% | +1.1(0.5)% | −1.9(0.4)% | +2.0(0.5)% |

  The flag gates add second-order fault pairs, which outweigh the
  first-order gain once p ≳ 2e-3.

#### Table F2. Sampled exact-MLD logical error rate with flag qubits (binomial SE, pooled over sequences)

| p | config | n_seq | m=0 | m=2 | m=4 | m=6 | m=10 |
|---|---|---|---|---|---|---|---|
| 0.0005 | none | 6 | 0.0055(3) | 0.0187(6) | 0.0286(7) | 0.0417(8) | 0.0652(10) |
| 0.0005 | ideal_standard | 6 | 0.0056(3) | 0.0182(5) | 0.0280(7) | 0.0397(8) | 0.0625(10) |
| 0.0005 | ideal_wrap | 6 | 0.0055(3) | 0.0184(5) | 0.0283(7) | 0.0374(8) | 0.0591(10) |
| 0.0005 | noisy_standard | 6 | 0.0057(3) | 0.0176(5) | 0.0285(7) | 0.0405(8) | 0.0647(10) |
| 0.0005 | noisy_wrap | 6 | 0.0053(3) | 0.0176(5) | 0.0281(7) | 0.0402(8) | 0.0609(10) |
| 0.002 | none | 6 | 0.0225(6) | 0.0735(11) | 0.1160(13) | 0.1556(15) | 0.2250(17) |
| 0.002 | ideal_standard | 6 | 0.0206(6) | 0.0718(11) | 0.1100(13) | 0.1523(15) | 0.2196(17) |
| 0.002 | ideal_wrap | 6 | 0.0214(6) | 0.0684(10) | 0.1083(13) | 0.1487(15) | 0.2130(17) |
| 0.002 | noisy_standard | 6 | 0.0213(6) | 0.0738(11) | 0.1113(13) | 0.1574(15) | 0.2294(17) |
| 0.002 | noisy_wrap | 6 | 0.0222(6) | 0.0736(11) | 0.1119(13) | 0.1547(15) | 0.2230(17) |
| 0.005 | none | 6 | 0.0552(9) | 0.1772(16) | 0.2558(18) | 0.3229(19) | 0.4057(20) |
| 0.005 | ideal_standard | 6 | 0.0524(9) | 0.1695(15) | 0.2496(18) | 0.3175(19) | 0.4027(20) |
| 0.005 | ideal_wrap | 6 | 0.0522(9) | 0.1671(15) | 0.2495(18) | 0.3129(19) | 0.3921(20) |
| 0.005 | noisy_standard | 6 | 0.0548(9) | 0.1778(16) | 0.2634(18) | 0.3302(19) | 0.4137(20) |
| 0.005 | noisy_wrap | 6 | 0.0528(9) | 0.1785(16) | 0.2589(18) | 0.3285(19) | 0.4086(20) |


**Conclusion.** Flags cannot rescue distance-2 correction. At least 77% of
the first-order MLD failure mass is irreducible data ambiguity. Flags are not
needed for LRB-D either: no single fault is an undetected logical with the
data5 readout (column `undetected` = 0 in `firstorder.json`; `L0`).

## Claims

Labels: **VA** verified analytically, **VN** verified numerically, **LIT**
from literature, **C** conjecture. SE in parentheses.

| # | claim | label | evidence |
|---|---|---|---|
| 1 | Stim's DEM of `HERALDED_ERASE` / `ELSE_CORRELATED_ERROR` needs `approximate_disjoint_errors=True`, which treats mutually exclusive outcomes as independent. A DEM-based MLD on heralded circuits is therefore not exact. | VN | Stim 1.15 raises without the flag; §1.2 |
| 2 | The conversion keeps the SI1000 Pauli channel exactly for every r: `DEPOLARIZE2(q)` = replacement with probability 16q/15, and `HERALDED_ERASE(4rq/3)` followed by `DEPOLARIZE1((1−r)q/(1−4rq/3))` = `DEPOLARIZE1(q)`. | VA, VN | §1.1 derivation; detector/observable statistics test (max \|z\| < 4.5); herald rate matches its expectation |
| 3 | `HeraldedMLD` is the exact MLD given the heralds. | VN | equals brute force to ≤4e-15; equals `TransferMatrixMLD` without heralds; calibrated on the true sampled channel (χ² p = 0.31, 0.32, 0.92) |
| 4 | One heralded erasure is always corrected, at code capacity and at circuit level. | VA, VN | X/Y/Z syndromes distinct on every qubit; 0 of 14052 circuit-level erasure locations miscorrected or ambiguous (Table E1) |
| 5 | The exact MLD's first-order error is 11.07p at m = 0 and 272.3p at m = 22 (11.45(28)p per step). 77% of it is data errors on the ambiguous pairs; measurement, reset and ancilla faults are never miscorrected. | VN | exact single-fault enumeration, 6 sequences; agrees with `L0` §4 (270p, 11.8p) |
| 6 | Heralding fraction r of all two-qubit-gate errors lowers the first-order MLD error by 10.2%, 19.0% and 21.2% (r = 0.5, 0.9, 1). Also heralding measurement-idle errors lowers it by 41.3%, 74.9% and 83.4%. The residual is data idling during reset plus one-qubit gates. | VN | Table E1 (paired over 6 sequences, SE ≤ 0.3%) |
| 7 | Sampled full correction at p = 2e-3: f = 0.9480(7) without heralds, 0.9586(7) with all two-qubit erasures heralded and 0.9873(2) with measurement-idle ones heralded too. ΔF(22) = +0.085(2) and +0.458(3) on identical shots. At p = 5e-4 the error per step falls by 21.5% and 81%, matching first order. | VN | Table E2, 12 sequences × 2e4 shots |
| 8 | "Correct heralded, reject unexplained" multiplies LRB-D acceptance by 1.41, 1.98 and 3.76 (2q_r1, all_r0.5, all_r1; p = 2e-3, m = 10). Accepted F is 0.983–0.992 against LRB-D's 0.995. | VN | Table E3, identical shots |
| 9 | At LRB-D's acceptance, herald-aware \|LLR\| postselection beats LRB-D: F = 0.9998(1) against 0.9951(5) (all_r1), and 0.9972(4) against 0.9959(5) (2q_r1). Without heralds the two coincide (0.9954). | VN | Table E3; `L0` §7 for r = 0 |
| 10 | Rejecting heralded shots on top of LRB-D only lowers acceptance (0.044 against 0.107 for all_r1) and does not improve F. | VN | Table E3 |
| 11 | The numpy leakage simulator reproduces Stim at κ = 0, and the forced-leak tests behave as designed. | VN | tests: detector rates, observable, pair correlations; leaked ancilla reads 1; LRU clears; leaked data randomizes the readout |
| 12 | Leakage adds an *exponential* decay to decoded LRB: the local rate of F_leak/F_none is constant at 0.9780(5), 0.9772(7), 0.9784(5) for κ = 0.5, p = 2e-3. A heralded LRU lowers the leakage-induced rate by only about 20%. | VN | Tables L1, L4 |
| 13 | Rejecting shots with leakage flags restores the leakage-free decoded F: 0.301(4) against 0.296(4) at acceptance 0.50 (κ = 0.5), or 0.87 (κ = 0.1). | VN | Table L2 |
| 14 | LRB-D unif=1 (+ final check) keeps its accepted F under leakage (0.9947(7) against 0.9961(4)) and pays in acceptance (×0.49 at m = 22, κ = 0.5). | VN | Table L2 |
| 15 | LRB-D const=1 loses F under leakage (0.558 → 0.377 at m = 22) with an accelerating, non-exponential local rate (0.9966 → 0.9905 → 0.9738). Three-state readout restores 0.555. | VN | Tables L2, L4 |
| 16 | That acceleration comes from const=1's syndrome-sector mixing amplified by extra errors, not from leakage persistence. | C | the LRU control shows the same acceleration (Table L4) |
| 17 | Persistence fingerprints: the leaked-data population grows linearly without an LRU (0.031 → 0.282 over 21 rounds) and stays flat at 0.015 with one. Lag ≥ 2 detection autocorrelation is 0.013–0.016 against 0.000. | VN | Table L3 |
| 18 | The `standard` flag layout (d1 F d2 F d3) cannot flag the hook from the first data CNOT: that fault precedes both flag CNOTs, so its spread to the flag cancels. | VA | Heisenberg propagation of Z_a through CX(f,a)·CX(d2,a)·CX(f,a) (§4.2); confirmed by the hook column of Table F1 |
| 19 | The `wrap` flag layout removes all hook miscorrections, but lowers the first-order MLD error by only 6.69(8)% (ideal) or 5.15(7)% (noisy). | VN | Tables F1 and F1b, paired over 6 sequences |
| 20 | At finite p, SI1000-noisy flags are neutral at p = 2e-3 (wrap −1.2(0.7)%) and harmful at p = 5e-3 (wrap +1.1(0.5)%, standard +2.0(0.5)%). Ideal flags give −3.4% to −7.1%. | VN | Table F2, 6 sequences × 1e4 shots |
| 21 | Each leak event costs about 0.8 of a shot's F (rate counting). | C | §3.1 estimate |
| 22 | Heralding the reset-layer idle errors as well would leave only one-qubit-gate faults at first order (about 5.5p at m = 22), so correction would become nearly second order. | C | extrapolated from the Table E1 residuals |
| 23 | The hybrid rule is the experimental `[[4,2,2]]` erasure rule; detection-correlation diagnostics of leakage are used on hardware. | LIT, UNVERIFIED by me | via `../lit/05` (findings 4, 6, 20), fetched by the literature agent, not re-fetched here |

## 5. Bugs and pitfalls

I found no bug in the lead's model. I checked it independently three ways:

- My re-implementation of the circuit builder reproduces
  `build_lrb_circuit` instruction for instruction (data2 and data5, depths 0,
  1, 4 and 22).
- My probe-based single-fault enumeration, which does not use Stim's DEM,
  reproduces the DEM's mechanism set. Its first-order E1 equals `L0` §4: 11.07p
  at depth 0, and 85.1p against 84.3p at depth 6.
- `HeraldedMLD` without heralds equals `TransferMatrixMLD` to 4e-15.

Pitfalls and latent issues, none of which affects the lead's results:

1. **Stim DEMs of heralded channels are approximate.**
   `detector_error_model(approximate_disjoint_errors=True)` turns the
   mutually exclusive outcomes of `HERALDED_ERASE`,
   `HERALDED_PAULI_CHANNEL_1` and `ELSE_CORRELATED_ERROR` into independent
   mechanisms. Two heralded mechanisms can then cancel each other's herald
   bit, which is O(q²) wrong. Feeding such a DEM to `TransferMatrixMLD` would
   not give an exact MLD. `HeraldedMLD` avoids it by conditioning on the
   heralds.
2. **NaN LLRs.** `TransferMatrixMLD` returns NaN when the syndrome has zero
   probability under its DEM: the normalization `psi /= psi.max()` divides
   0/0. `experiment._histogram_gap` then puts NaN into the *top* (most
   confident) gap bin, and `llr < 0` is False. This cannot happen for
   SI1000 samples, because every mechanism has p > 0 and the mechanism span
   covers all syndromes (VA). It does matter for zero-noise priors such as my
   `hybrid` explainer, which handles −inf explicitly. My `_histogram_gap`
   maps NaN to gap 0.
3. **FlipSimulator stepping.** `stim.FlipSimulator.do` stepped instruction by
   instruction samples the right distribution, but with the same seed it is
   not bit-identical to `do(circuit)`. `broadcast_pauli_errors` costs about
   10 ms per call at 5e4 shots whatever the mask size, which is why I wrote a
   numpy frame simulator for the leakage study.
4. **Flag placement.** The textbook placement (`standard`: flag CNOTs after
   the first and before the last data CNOT) misses the ancilla fault right
   after the first data CNOT, because it occurs before the first flag CNOT.
   For weight-3 checks that is the hook. Use the `wrap` layout to flag it.

## 6. Open questions

1. **Exact leakage-aware decoding.** A leakage flag says *that* a qubit leaked,
   but not when. The exact posterior is a mixture over leak times, which
   `HeraldedMLD.log_joint` can evaluate (it returns exact joint
   log-likelihoods). How much of the `mld_flag` acceptance loss could
   correction recover? Not run.
2. **Imperfect heralds.** Herald false negatives act like unheralded errors;
   false positives only widen the posterior. A stated rate of either is easy
   to add to the `ELSE` chain. Not run.
3. **Converting reset-idle noise too.** With every two-qubit and
   measurement-idle event heralded, data idling during the reset layer
   (11.1p of 12.4p at depth 6) is the remaining first-order term. A platform
   that heralds idle errors (metastable or dual-rail qubits) should approach
   second-order correction (C).
4. **Seepage and partial LRU efficiency** (`seepage`, `lru_efficiency` are
   implemented): with finite leakage lifetime the decoded decay should cross
   over from the no-LRU to the LRU curve. Not swept.
5. **Hook-aware flagged decoding at d=3.** Flags buy only about 5% here
   because d=2 ambiguity dominates. Would the same `wrap` layout matter more
   for the distance-3 codes of WP G (C)?

## 7. Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd <worktree>
python -m pytest -q tests/test_qubit512_leak.py          # 44 tests, ~10 s
# heavy runs on the shared allocation; --cpu-bind=none -c 256 avoids the
# shared-CPU pinning of overlapping steps:
#   srun --jobid=<ID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c "<env>; python ..."
python studies/correction-decoders/qubit/scripts/erasure_leakage.py erasure    --workers 24 --num-seq 12 --shots 20000
python studies/correction-decoders/qubit/scripts/erasure_leakage.py leakage    --workers 24 --num-seq 12 --shots 40000
python studies/correction-decoders/qubit/scripts/erasure_leakage.py flags      --workers 24 --num-seq 6  --shots 10000
python studies/correction-decoders/qubit/scripts/erasure_leakage.py firstorder --workers 24 --num-seq 6  --depths 0,2,4,6,10,22
python studies/correction-decoders/qubit/scripts/erasure_leakage.py report     --num-seq 12
```

- Seeds depend only on (study, configuration, p, sequence). Finished parts in
  `results/I/<study>/parts/` are skipped, so interrupted runs resume and give
  the same numbers.
- On a fixed allocation the full set takes about 6 minutes on 24 cores. Under
  the old pinned binding it took more than 30 minutes.
- Outputs:
  - `results/I/tables.md`: every method, depth and configuration, as mean
    (SE) plus the fitted f;
  - `results/I/highlights.md` and `highlights.json`: the curated
    comparisons, including gap thresholds at matched acceptance;
  - `results/I/summary.csv`;
  - `results/I/firstorder/firstorder.json` (6 sequences) and
    `firstorder_3seq.json` (the first, 3-sequence pass);
  - `results/I/logs/`.
- The tables in this note were generated from `highlights.json` and
  `firstorder.json` by the formatting snippets used in the session; every
  number appears in `tables.md` or `highlights.md`.
