# L5: Pauli-frame tracking, offline vs real-time decoding, sliding windows, decoder latency

Agent L5, 2026-10-28. Scope: when can the decoder's correction for the qutrit LRB
circuits be applied offline as a Pauli (Weyl) frame update, when does real-time
decoding with feedforward become necessary, what sliding/parallel window decoders
do, and what decoder latencies have been measured on hardware.

Label convention used on every claim:

- **[analytic]**: verified analytically (argument given here).
- **[numeric]**: verified numerically in this task (scripts in `L5-scripts/`, outputs quoted).
- **[lit: Key]**: from a paper I fetched in this task (Key refers to Section 4).
  Numbers quoted secondhand inside a fetched paper are marked "(quoted by Key)".
- **[conjecture]**: my inference, not verified.

"Weyl operator" means a qutrit Pauli `X^a Z^b`. sdim conventions apply: `X|j> = |j+1>`,
`Z = diag(w^j)`, `w = exp(2 pi i/3)`.

---

## 0. Summary for the study

1. **Offline correction is exact for this experiment.** All gates in the LRB circuits are
   qutrit Cliffords. The noise is stochastic Weyl noise. The terminal readout is a
   computational-basis measurement after `H_INV`, and the circuits contain no classically
   controlled operations. Under these conditions, applying a decoder's output offline as a
   Z_3 shift of the terminal logical outcome gives the same joint distribution of records
   and corrected logical outcome as applying the same corrections physically, in real time,
   with perfect and instantaneous Weyl gates. **[analytic]** (Section 1.1), **[numeric]**
   (exact toy model, total-variation distance ~1e-17, Section 1.3), **[lit: Terhal2015,
   DiVincenzoAliferis2007, Knill2005, Chamberland2018, Skoric2023]**.
2. **Offline processing can do anything a real-time decoder can, and more.** Any causal
   (streaming or windowed) correction protocol can be replayed offline exactly. An offline
   decoder can also use the whole record, for example a global MLD over the compact DEM.
   For this Clifford-only experiment, real-time decoding therefore gives no accuracy gain.
   It is only a demonstration of throughput and latency. **[analytic]**
3. **When the equivalence breaks.** (a) A non-Clifford gate (for example a qutrit T gate
   via magic-state injection) or any classically controlled operation needs the frame
   before the gate. This is feedforward, and the reaction time is
   `T_react = T_decode + T_control` **[lit: GoogleQAI2025, Skoric2023, Terhal2015,
   DiVincenzoAliferis2007]**. (b) Non-Weyl noise breaks it: amplitude damping, leakage,
   coherent or state-dependent errors. The frame-tracked state then evolves differently
   from the corrected state. The discrepancy is second order in the damping strength in my
   toy model: TV distance 1.5e-3 to 3.8e-3 at damping 0.05, and about 4x smaller each time
   the damping is halved **[numeric]**. (c) It also breaks if the
   correction gates or the waiting time for the decoder add noise. Then real-time physical
   correction is strictly worse than the frame **[analytic]**, **[lit: Fowler2012SC,
   RyanAnderson2021, Chamberland2018]**.
4. **The repository's raw-outcome detectors are not local in time.** In the baseline
   depth-22 circuit, 60% of detected unit faults shift the raw detector of every round
   from their first appearance to the terminal round **[numeric]**. Two consequences
   follow. A sliding-window or streaming decoder, and any hybrid policy that corrects some
   rounds and detects on others, must work with frame-relative detectors (syndrome changes
   relative to the Clifford-propagated previous syndrome), not raw outcomes. Knill's rule
   is the same: "An error is detected when the syndrome differs from that expected
   according to the Pauli frame" **[lit: Knill2005]**.
5. **Five unit faults per circuit are invisible and flip the logical.** They are
   independent of m, sit at the very end of the circuit (Z-type parts of the X-check CNOT
   noise and of the measurement-layer idles on wires 0 and 3 in the final round), shift
   the logical outcome, and trigger no detector **[numeric]**. No decoder, offline or real-time, can remove them. They also
   survive LRB-D postselection.
6. **Hardware.** LRB with correction needs no real-time decoding and no feedback for the
   decoder. The control system only has to store about 4(m+1) trits per shot (92 trits,
   about 146 bits, at m = 22) **[analytic]**. The remaining real-time elements are
   ancilla reset, leakage handling and mid-circuit-measurement timing. Reset can be made
   feedback-free with unconditional reset or with "no reset" plus software tracking
   **[lit: Geher2025, Caune2026]**. Measured decoder latencies are summarised in
   Section 2. Window sizes are summarised in Section 3.
7. **Non-Markovianity [conjecture].** A decoder that commits corrections window by window
   gives a logical shift that is a sum of nearly independent per-window shifts. Its
   characteristic function `E[w^S]` then factorizes, and the decay is approximately
   exponential. A global decoder correlates each round's correction with future data. It
   does not, however, condition on acceptance the way postselection does. See Section 5.

---

## 1. Offline frame correction versus real-time correction

### 1.1 Statement and proof sketch [analytic]

**Setting.** n qutrits. A circuit is a sequence of

- (i) Clifford unitaries from the repository's gate set {H, H_INV, P, P_INV, X, X_INV, Z,
  Z_INV, CNOT, CNOT_INV, CZ, CZ_INV, SWAP};
- (ii) stochastic Weyl channels `N(rho) = sum_w p(w) W_w rho W_w^dag`, with p independent
  of the state and of any classical data;
- (iii) computational-basis measurements, with mid-circuit ancilla measurements followed
  by an unconditional reset to |0>;
- (iv) a final computational-basis readout, which for the logical X readout comes after
  `H_INV`.

Protocol **R** (real time): at chosen times t, apply a Weyl correction
`C_t = c_t(record so far)` physically, perfectly and instantly.

Protocol **F** (frame): apply nothing physically. Keep a classical frame F:

- `F <- U F U^dag` after each Clifford U;
- `F <- C_t F` at each correction time;
- after resetting qutrit q, delete F's component on q;
- report each computational-basis outcome on q shifted by a_q(F), the X-exponent of F on q
  at that moment.

**Lemmas.**

- (L1) A Clifford maps a Weyl operator to a phase times a Weyl operator. This is the
  definition of the qudit Clifford group **[lit: Gottesman1998]**. It is verified for the
  exact sdim matrices in Section 1.3 **[numeric]**.
- (L2) Weyl operators commute up to a phase, `W V = w^{<w,v>} V W`. Hence every stochastic
  Weyl channel is covariant: `N(V rho V^dag) = V N(rho) V^dag`.
- (L3) For `F = X^a Z^b` on qutrit q: `|k><k| X^a = X^a |k-a><k-a|`, and `Z^b` commutes
  with the projector. So outcome k on `F rho F^dag` has the probability of outcome k-a on
  rho, and the post-measurement states are related by F. A reset maps both states to |0>
  on q, which clears F's q-component up to a global phase.

**Claim.** For every fixed realization of the noise (a sequence of Weyl errors), induction
over the circuit gives `rho_R(t) = F_t rho_F(t) F_t^dag`. In addition, the raw records of R
equal the shifted records of F. The two protocols draw the noise from the same distribution,
so averaging over it makes the full joint distributions of (records, corrected logical
outcome) identical. The decoder in R reads raw outcomes. The decoder in F reads shifted
outcomes, which are the "syndrome relative to the frame". It therefore receives identical
inputs and produces identical corrections.

**Consequences.**

- (a) Any causal real-time protocol can be emulated exactly offline.
- (b) The best offline protocol may use the full record, so it is at least as good.
- (c) When noisy correction gates or decoder-waiting idles are added to R, R gets worse
  and F does not.

A concrete case of (c) under the repository's SI1000 model: the model has no explicit
"wait for decoder" step. If such a wait were modelled like its measurement and reset
idles, each waiting layer would cost 2p depolarizing per data qutrit. F has none of these
extra fault locations. Take the special case where an extra independent Weyl shift E adds
directly to the logical shift S. Then `|E[w^(S+E)]| = |E[w^S]| |E[w^E]| <= |E[w^S]|`, so
the fidelity cannot rise **[analytic]**. In general I expect, but have not proved, that
extra fault locations lower the decoded fidelity **[conjecture]**.

**Twirling remark [analytic].** Averaging any channel over conjugation by the qutrit Weyl
group, `(1/9) sum_W W^dag N(W rho W^dag) W`, gives a Weyl channel. Randomized frames
therefore restore (L2) on average. Knill proposed exactly this for coherent errors:
"frequently applying known but random Pauli products, thus modulating the Pauli frame"
**[lit: Knill2005]**.

### 1.2 Conditions for the equivalence and what breaks them

| # | Condition | Status in the repository's LRB circuits | What breaks it on hardware |
|---|---|---|---|
| C1 | Only Clifford operations between a correction point and the readout | Holds: every gate in the logical H, P, X, Z tokens, the checks and the readout is in the gate set of 1.1, and the Weyl-group closure is checked numerically **[numeric]** | Non-Clifford gates, e.g. a qutrit T gate. Its conjugate `T X T^dag` is Clifford but not Weyl **[numeric]**, so the frame cannot be pushed through it **[lit: DiVincenzoAliferis2007, Terhal2015, Chamberland2018, Fowler2012TO]** |
| C2 | Readout in a Weyl eigenbasis | Holds: `H_INV` then M on wires 0 and 3; logical outcome x0 + x3 | n/a |
| C3 | Stochastic Weyl noise, independent of state and frame | Holds by construction (`N1 d`, uniform N2, `N1 f` shifts) | Amplitude damping, leakage (qutrit transmon to the 4th level), coherent rotations, state-dependent readout errors **[numeric toy; lit: Knill2005, Terhal2015]** |
| C4 | No classical control based on decoded or raw data (no feedforward) | Holds: sdim's gate list (`sdim/gatedata.py`) has no classically controlled gate, and the circuits contain none (read from code) | Conditional gates, repeat-until-success, magic-state injection, conditional reset, adaptive stopping that changes later gates |
| C5 | For the real-time comparison only: corrections are perfect and instantaneous | Model assumption | Correction gates and decoder latency add noise. The frame avoids both **[lit: Fowler2012SC, RyanAnderson2021, Chamberland2018]** |
| C6 | Decoder choice | Offline, any decoder (global, MLD, windowed) is allowed | A real-time decoder that must commit early, such as a sliding window, can be slightly worse than a global one. The fetched studies find no difference within error bars when buffer >= d **[lit: Skoric2023, Tan2023, DKLP2002]** |

Two qudit-specific caveats:

- **Bosonic qudits.** In GKP qudits, for odd d the extra half-displacement left by echoed
  conditional displacements "brings us outside the code space, an effect we have to
  reverse". This is in contrast to even d, where it is a logical Pauli "tracked in
  software" **[lit: Brock2025]**. So for odd-d bosonic qudits, part of the correction
  cannot be absorbed into a Weyl frame.
- **Transmon qutrits [conjecture].** Diagonal (Z-type) Weyl corrections can be applied as
  virtual phase updates in the control software. X-type corrections need pulses. Even a
  "physical" real-time correction is therefore free for Z-type errors.

### 1.3 Numerical checks [numeric]

Script: `L5-scripts/frame_equivalence_toy.py` (numpy only; about 10 s; single thread with
`OMP_NUM_THREADS=1`). Its output:

```
(1) Clifford conjugation maps Weyl -> phase * Weyl
   H, H_INV, P, P_INV, X, Z : closure on all 9 one-qutrit Weyl operators: True
   CNOT, CNOT_INV, CZ, CZ_INV, SWAP : closure on all 81 two-qutrit Weyl operators: True
(2) qutrit T = diag(1, z, z^8), z = exp(2 pi i/9): T X T^dag is Weyl? False
    ... but T X T^dag is Clifford (maps all Weyl to Weyl): True
(3) covariance || N(V rho V^dag) - V N(rho) V^dag ||_1, V = X, random rho
   random Weyl channel             : 1.6e-16
   ladder amplitude damping g=0.05 : 1.3e-01
(4) total-variation distance between R (physical correction) and F (frame relabelling)
   case                                          | uniform depol p=0.02 | random biased Weyl | amp. damping g=0.05
   |+>_L, U = H0 CNOT02 P1 H1^dag CZ12 (Clifford)  |   5.0e-18            |   3.8e-17          |   1.5e-03
   |psi>_L, U = decode, P0, H0 (Clifford)          |   1.4e-18            |   2.3e-17          |   3.8e-03
   |psi>_L, U = decode, T0, H0 (naive frame thru T)|   1.7e-18 (*)        |   3.1e-03          |   3.8e-03
```

Model for (4): an exact density-matrix calculation on 3 data qutrits (repetition code) and
2 ancillas.

1. Encode.
2. Noise on the data.
3. Measure `Z0 Z1^-1` and `Z1 Z2^-1` with CNOT / CNOT_INV, then reset the ancillas.
4. Lookup correction (perfect gate in R; frame update in F).
5. Second noise layer.
6. Apply U.
7. Read out the data in the computational basis.
8. Compare the exact joint distributions P(s1, s2, k0, k1, k2).

The sdim matrices are used for H, P and CNOT. CZ and SWAP use the standard definitions.

Section (5) of the same script rescales the amplitude-damping strength in the two
Clifford cases:

```
   case            g = 0.05   g = 0.025   g = 0.0125
   |+>_L, U ...    1.50e-03   3.96e-04    1.02e-04
   |psi>_L, U ...  3.81e-03   1.01e-03    2.59e-04
```

Each halving of g cuts the TV distance by 3.8 to 3.9. That is consistent with second
order in g. A frame differs from the identity only after a damping jump (probability about
g), and the covariance violation of the next damping layer is again about g **[numeric]**.
The explanation is **[conjecture]**.

(*) Uniform depolarizing makes the naive offline treatment of T look correct here, but
only by accident. In the branches where an X error is detected, the Z part of the
uniform depolarizing error is uniformly random. That dephases qutrit 0 completely, and T
then acts trivially on the outcome distribution. With biased Weyl noise the naive frame
fails (TV 3.1e-3). The correct protocol for a non-Clifford gate is R: apply the correction,
or the Clifford fix-up after T, before continuing. That is feedforward.

### 1.4 Raw-outcome detectors in the baseline circuits [numeric]

Script: `L5-scripts/raw_detector_persistence.py`. It compiles the repository's compact DEM
(`lrb.dem_simulation.build_compact_detector_error_model`) for baseline circuits. The
circuits are read read-only from
`.../Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000/experiments/LRB/<cliff>/4/<depth_idx>.chp`.
For every unit Weyl component of every noise location, it records which check rounds' raw
detectors (`lrb_stab_r{round}_w{wire}`) are shifted.

| circuit (cliff, m) | detectors | active noise locations | unit components | detected | shift 1 round | shift every round from first appearance to m (>= 2 rounds) | reach round m | undetected but flip logical |
|---|---|---|---|---|---|---|---|---|
| (0, 4)  | 20 | 464  | 1092 | 862  | 418  | 444 (51.5%)  | 84.7% | 5 |
| (0, 10) | 44 | 1130 | 2660 | 2202 | 944  | 1258 (57.1%) | 84.7% | 5 |
| (7, 22) | 92 | 2308 | 5416 | 4502 | 1806 | 2696 (59.9%) | 83.5% | 5 |

The time-span histogram for m = 22 runs from 1 to 23 rounds, with mass at every value.
By the circuit structure, the single-round signatures should be measurement-type faults,
faults in round m itself, and faults in the unchecked inverse block, which are visible only
at round m. I did not count these categories separately **[analytic]**.

Interpretation **[analytic]**. Raw detector = raw ancilla outcome. A data error persists,
and the logical Cliffords map the stabilizer group onto itself, so a persistent error keeps
a non-trivial syndrome in every later round. Time-local detection events are
`D_r = s_r - A_r s_{r-1}` (mod 3), where `A_r` is the Z_3-linear map by which the r-th
forward Clifford permutes and combines the four check values. For the inverse block,
`A_m` is the composite map. This is the input a windowed decoder needs. Global MLD or
table decoders over the DEM can use raw outcomes directly, because the DEM is linear.

Script: `L5-scripts/undetected_logical.py`, run on circuit (0, m = 4). The five faults
that flip the logical and trigger no detector are, at operations 696 to 727 of 760:

- op 696: N2 component 3 after `CNOT 7->0`;
- op 717: N2 component 3 after `CNOT 8->3`;
- ops 703, 724, 727: N1 component 1 (2p idle) on data 0 during ancilla 7's measurement
  layer, and on data 0 and data 3 during ancilla 8's measurement layer.

All five are Z-type faults on wires 0 or 3 that occur after the last check able to see
them. The circuit ends with `M 8`, then the ideal `H_INV 0, H_INV 3, M 0, M 3`. There is
no reset layer after the final round **[numeric, read from the circuit]**.

---

## 2. Latency, throughput and cycle-time numbers

| System / paper | Platform, decoder | QEC round time | Decoder number | Definition used |
|---|---|---|---|---|
| GoogleQAI2025 | Superconducting 72-qubit processor (the real-time experiment; the d=7 memory ran on a 105-qubit processor; the Nature abstract calls both the "Willow" generation), d=5 surface code, software "Correlated Parallel Blossom" on a workstation | 1.1 µs | 63 ± 17 µs average latency, roughly constant up to 10^6 cycles (1.1 s); throughput < 1.1 µs/cycle; `T_input` < 10 µs | Latency = time from the software receiving the final cycle to returning the correction. Excludes feedback into the logical circuit |
| GoogleQAI2025 | Offline neural-network decoder (comparison) | n/a | 24 µs per cycle | Offline |
| Caune2026 (arXiv v1 numbers) | Rigetti Ankaa-2, 8-qubit stability experiment, Collision Clustering on FPGA inside the control system | ≈1.7 µs | Mean decoding time per round 0.44 µs (5 rounds) to 0.79 µs (25 rounds). Full decoding response 9.6 µs for 9 rounds = 6.5 µs decoding + 3.1 µs communication and control. 1.4 µs for the final readout to reach the decoder; 1 to 1.5 µs (250 to 370 FPGA cycles) of control logic | Response time = final extraction round to application of the conditional gate (an X gate was applied) |
| Barber2025 | Collision Clustering on FPGA (Xilinx XCVU3P, about 400 MHz), circuit-level noise p = 0.1% | n/a (simulated memory) | Per-round execution time 0.06 to 1.18 µs for d = 3 to 23 (0.81 µs at d = 21); threshold 0.78%; < 13 KB memory at d = 23 | Execution time averaged over 1e5 shots, divided by d rounds; not streaming |
| Barber2025 | Same decoder on a 12 nm ASIC (signed off, not taped out) | n/a | d = 7: 10 ns/round at 2.75 mW. d = 23: 240 ns/round at 7.85 mW, 0.06 mm² | Same |
| Liyanage2023 | Helios distributed union-find on FPGA (VCU129), phenomenological noise 0.1% | n/a | 11.5 ns per round at d = 21. At d = 13, 90% of trials within 250 ns and 99.99% within 370 ns | Average decoding time divided by rounds |
| Liyanage2024 | Helios | n/a | 23.7 ns/round at d = 17 under circuit-level noise. d = 51 at 544 ns/round (resource-efficient mode). Sliding window at d = 13: latency 194 → 230 ns. Worst observed 920 ns, against a theoretical worst case of 857 µs | Same |
| RyanAnderson2021 | Honeywell (Quantinuum) H1 trapped ions, [[7,1,3]] color code, two-stage lookup table, real-time | < 200 ms per QEC cycle | Conditional information distribution adds < 250 µs in the worst case | Real-time Pauli-frame updates and conditional gates |
| Battistel2023 | Review | about 1 µs (transmons), ≲ O(1) ms (spin qubits), ≳ O(100) ms (ion traps) | Requirement: decode at the acquisition rate. FPGA bus latency "at most a couple hundreds of nanoseconds". Ideal integration "in the order of 10s of nanoseconds" | |
| Skoric2023 | Review and simulation | 1.1 µs (Krinner et al.) and 921 ns (Google 2023), as quoted by Skoric2023 | LILLIPUT lookup table: 300 ns per round for d <= 5 (quoted by Skoric2023) | |
| Sivak2023 | GKP qubit (bosonic), FPGA controller | QEC cycle 2 × 4.924 µs, including real-time processing and decision | Virtual oscillator rotations implemented "dynamically on the FPGA in 448 ns" | |
| Brock2025 | GKP qutrit and ququart | 7 µs per sBs round | Autonomous protocol; only ancilla reset and phase updates between rounds | |
| Fowler2012TO | Hypothetical superconducting architecture | 100 ns measurement, 200 ns cycle | Canonical T gate 76.5 µs vs 100 ns time-optimal, assuming fast classical processing | |
| Caune2026, GoogleQAI2025 | Algorithm-scale target | | Full decoding response within 10 µs, as assumed for RSA-2048 factoring (Gidney and Ekerå, quoted by Caune2026); Google: "within an order of magnitude" of that assumption | |

**For this study [analytic].** The code has four checks per round, so a per-round lookup
table has 3^4 = 81 entries. A table indexed by the full history is out of reach for an
explicit table (3^92 at m = 22). The DEM, however, has only 464 to 2308 active locations,
so DEM-based offline decoders are cheap.

**[conjecture]** If real-time decoding were ever needed, a per-round table or a small
windowed decoder on the control FPGA would decode in nanoseconds. The reaction time would
then be dominated by readout and control latency, which is microseconds on present
superconducting stacks (Caune2026).

---

## 3. Window sizes and commit rules

| Scheme | Window | Commit / step | Size rule quoted | Accuracy vs global | Key |
|---|---|---|---|---|---|
| Overlapping recovery | 2T rounds | Commit E'_old (chains with both endpoints in the older T rounds). Carry the boundary of E'_keep forward. Step T | T ≫ L for comparable qubit and measurement error rates (their eq. (71) gives the general condition) | "very nearly as well as if an indefinite amount of information were stored" | DKLP2002 |
| Sliding window | n_W = n_com + n_buf | Commit n_com, slide by n_com. Artificial defects at the commit/buffer boundary | n_buf = n_com = d | No noticeable increase in logical error rate | Skoric2023 |
| Parallel window | Layer A: buffer + commit + buffer = 3w. Layer B: 3d, no buffers | A windows decoded in parallel, then B windows fill the gaps | w = d | Within numerical error of global MWPM and UF (2% phenomenological noise) | Skoric2023 |
| Sandwich (Alibaba) | w = s + 2b | Core region s, buffers b on both sides; seams decoded afterwards | s = (d+1)/2, w = 3s | Circuit-level thresholds 0.68% vs 0.70% (MWPM sandwich vs batch), 0.55% vs 0.554% (UF) | Tan2023 |
| Forward (sequential) | w = s + b | Step s | b fixed; s = b, w = 2b keeps a constant overhead relative to batch decoding | | Tan2023 |
| Google streaming | Blocks of M cycles, two fusion layers | Heralds failure if regions remain matched to block boundaries | M = 10 (d = 3, 5 surface code), M = 90 (d = 29 repetition code); buffer of 128 blocks | Real-time ε5 = 0.35% vs offline NN 0.269%. Blocks were chosen so that heralded block-boundary failures are negligible, and Google attributes the gap to the faster, less accurate real-time decoder | GoogleQAI2025 |
| Helios sliding window | 2d rounds | Commit the oldest d, advance by d | | | Liyanage2024 |
| LILLIPUT lookup table (quoted) | ≤ 3 rounds, step 1 | | d ≤ 5 | | quoted by Tan2023 and Skoric2023 |

**Mapping to this study [conjecture].** For d = 2 with ≤ 23 rounds, the global problem is
tiny, so windowing is never needed for compute reasons. A d-scaled window
(n_com = n_buf = 2 rounds) is the natural model of a streaming decoder if the study wants
to compare the decay shape under local commits with global decoding (Section 5, Q2).

The unchecked inverse block breaks time-translation invariance. A window cannot straddle
it usefully: everything in the m inverse Cliffords is seen only by round m.

---

## 4. Paper-by-paper notes

### 4.1 Knill 2005 [Knill2005]

**Citation.** E. Knill, "Quantum computing with realistically noisy devices", *Nature* 434,
39–44 (2005), DOI 10.1038/nature03350. arXiv:quant-ph/0410199 under the preprint title
"Quantum Computing with Very Noisy Devices" (v1 25 Oct 2004, v2 2 Nov 2004). Fetched:
https://arxiv.org/abs/quant-ph/0410199, https://arxiv.org/pdf/quant-ph/0410199 (full text
v2; quotes below are from it), https://www.nature.com/articles/nature03350 (title, volume,
pages).

**Extracted facts [lit].**

- Definition: "the state is defined only with respect to a current 'Pauli frame' and an
  implicit recovery scheme. The Pauli frame is defined by a Pauli product that restores the
  error-free state of the block to the syndrome 0 subspace." This "makes it possible to
  avoid explicitly applying Pauli products for correction or teleportation compensation."
- Detection versus correction in one sentence: "An error is detected when the syndrome
  differs from that expected according to the Pauli frame. In 'postselected' quantum
  computing, the state is then rejected and the computation restarted. In standard quantum
  computing, the syndrome information must be used to update the Pauli frame."
- Postselected computing can run "optimistically", without waiting for outcomes: "The
  output of the computation is accepted only if all measurement outcomes are as
  anticipated."
- When waiting is needed: "If the correction gates are simple Pauli products, they can be
  absorbed into the Pauli frame and do not need to be known immediately. For teleportations
  used to implement non-Clifford gates, a non-Pauli compensation may be required. In this
  case, we must wait for the measurement outcomes."
- Prepared states must be checked before use: "until the state is ready to be used for a
  logical gate. At this point it is necessary to wait for measurement outcomes".
- "none of the Pauli corrections actually have to be implemented explicitly. One can just
  update the Pauli frame as needed."
- Error model: errors are "applications of Pauli products with probabilities determined by
  the gate" (depolarizing).
- On coherent errors: random known Pauli products "modulating the Pauli frame" reduce
  coherent addition. On leakage: error-correcting teleportation controls leakage, and a
  leaked Bell-measurement outcome "can be filled in arbitrarily ... the effect is the same
  as if a Pauli error occurred".
- Assumption: "no error and no speed constraint on classical computations required to
  interpret measurement outcomes and control future gates."
- Knill's simulator tracks an "error vector", "known only to the simulation, not to the
  user", together with the Pauli frame. This is the same split as the repository's DEM
  sampler (simulation-side Weyl error) versus a decoder's frame.

**Bearing.**

- LRB-D is Knill's postselection rule with a trivial frame. With no corrections the
  expected syndrome is 0, so "syndrome differs from that expected" reduces to "a used raw
  outcome is non-zero" **[analytic]**.
- LRB with correction is Knill's "standard" mode **[analytic]**.
- For Clifford-only circuits both modes can be evaluated offline; postselection is decided
  at the end **[analytic; lit: Knill2005]**.
- Hybrid policies must compare syndromes with the frame-expected syndrome (Summary item 4)
  **[analytic; lit: Knill2005]**.

### 4.2 DiVincenzo and Aliferis 2007 [DiVincenzoAliferis2007]

**Citation.** D. P. DiVincenzo, P. Aliferis, "Effective fault-tolerant quantum computation
with slow measurements", *Phys. Rev. Lett.* 98, 020501 (2007), published 9 Jan 2007, DOI
10.1103/PhysRevLett.98.020501. arXiv:quant-ph/0607047 (v2 3 Aug 2006). The arXiv
journal-ref field misprints the article number as "220501"; the APS page confirms 020501.
Fetched: https://arxiv.org/abs/quant-ph/0607047, https://arxiv.org/pdf/quant-ph/0607047,
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.020501.

**Extracted facts [lit].**

- Abstract: "measurement times as long as even 1,000 gate times or more have a very
  minimal effect on the quantum accuracy threshold."
- Pauli frames: "it is not necessary to directly apply these recovery operations on the
  data. Instead, it is sufficient to merely record and keep track of them in a classical
  memory as a reference frame defined by a Pauli rotation. This is so because the Pauli
  group is closed under the action of the Clifford group ... the application of the
  recovery operations specified by the syndrome can usually be delayed a long time."
- Feedforward requirement: "Non-Clifford operations require a different treatment since a
  Pauli frame cannot be simply propagated through them ... all information determining the
  current Pauli frame must be known before the application of a non-Clifford gate".
- Example (their Fig. 3): T via the ancilla `|a_pi/8>` with S applied "only if the
  measurement outcome is −1".
- Fixes:
  - (i) replace ancilla verification, which forces waiting, by "ancilla decoding", whose
    output is again a Pauli-frame update;
  - (ii) inject `|a_pi/8>` by teleportation only at a concatenation level k_min where
    `t_gate^(k) = a C^k` exceeds the measurement time (k ≥ 2 for `t_meas = 1000 a`,
    C = 34).
- Limitation: the scheme "cannot be used in post-selected computation".

**Bearing.**

- This is an early explicit statement that the Clifford part of an FT circuit can defer
  all corrections. The authors say it "has been known for some time" and cite Knill and
  Aliferis–Gottesman–Preskill **[lit]**.
- For a qutrit LRB the same closure holds for the Weyl group (Section 1.3), so slow
  readout of qutrit transmons does not by itself force real-time decoding
  **[analytic]**.
- Measurement latency must be distinguished from the syndrome acquisition rate. Only the
  latter lowers the threshold **[lit: Terhal2015, Chamberland2018]**.

### 4.3 Riesebos et al. 2017 [Riesebos2017]

**Citation.** L. Riesebos, X. Fu, S. Varsamopoulos, C. G. Almudever, K. Bertels, "Pauli
Frames for Quantum Computer Architectures", in *Proc. 54th Annual Design Automation
Conference (DAC 2017)*, Austin, 18–22 June 2017, Article 76, pp. 1–6, ACM, DOI
10.1145/3061639.3062300 (authors confirmed by the TU Delft portal and Crossref). Fetched:
https://research.tudelft.nl/en/publications/pauli-frames-for-quantum-computer-architectures/
(full abstract), https://api.crossref.org/works/10.1145/3061639.3062300. The ACM full text
returned 403 and the TU Delft repository was unreachable, so **only the abstract was
read**.

**Extracted facts [lit, abstract only].**

- "The Pauli frame mechanism allows Pauli gates to be tracked in classical electronics and
  can relax the timing constraints for error syndrome measurement and error decoding."
- They "implemented and simulated the Pauli frame module which, in principle, can be
  directly mapped into a hardware implementation."
- "Simulation of a surface code 17 logical qubit has shown that a Pauli frame can reduce
  the error rate of a logical qubit up to 70% compared to the same logical qubit without
  Pauli frame when the decoding time equals the error correction time, and maximum
  parallelism can be obtained."

**Bearing.**

- A hardware Pauli frame module in the control electronics is the architectural form of
  protocol F in Section 1.1 **[lit: Riesebos2017 abstract]**.
- My reading of "when the decoding time equals the error correction time": the 70% gain
  comes from avoiding the waiting and extra gates that physical correction would need.
  That is condition C5 **[conjecture]**.
- I could not check the paper's noise model, so the 70% figure should not be carried over
  to SI1000 qutrits **[conjecture]**.
- Battistel2023 cites this paper for the claim that the correction is "needed only for
  the next-in-line logical non-Clifford gate" **[lit: Battistel2023]**.

### 4.4 Dennis, Kitaev, Landahl, Preskill 2002 [DKLP2002]

**Citation.** E. Dennis, A. Kitaev, A. Landahl, J. Preskill, "Topological quantum memory",
*J. Math. Phys.* 43, 4452–4505 (2002), DOI 10.1063/1.1499754. arXiv:quant-ph/0110143
(v1 24 Oct 2001). Fetched: https://arxiv.org/abs/quant-ph/0110143,
https://arxiv.org/pdf/quant-ph/0110143, https://api.crossref.org/works/10.1063/1.1499754.

**Extracted facts [lit].**

- Assumptions (abstract): "qubits can be measured rapidly, and that polynomial-size
  classical computations can be executed instantaneously". Of the minimum-weight chain:
  "We assume this classical computation can be performed instantaneously and flawlessly."
  Section X gives a recovery "that does not require measurement or fast classical
  processing", but it is local only in ≥ 4 spatial dimensions.
- Sec. VI: in a computation with interacting blocks, reconstructing histories over the
  causal past is "far too complex", so "we will need to intervene and perform recovery
  repeatedly".
- **Overlapping recovery** (Sec. VI.B, Fig. 13):
  - measure 2T rounds and build the minimum-weight chain E';
  - correct only E'_old, the chains with both endpoints in 0 ≤ t < T;
  - keep the relative boundary of E'_keep;
  - measure the next T rounds and repeat.
  - Choosing "T to be large compared to the characteristic correlation time", the method
    performs "very nearly as well as if an indefinite amount of information were stored".
    For comparable qubit and measurement error rates "it suffices to choose T ≫ L".
  - Principle: "we should not overreact to syndrome information that is potentially
    faulty ... we take action to remove only these long-lived defects, leaving those of
    more recent vintage to be dealt with in the next recovery step."
  - "to construct E', we do not need to store all of S ... only the relative boundary of
    S is needed."
- Crude storage threshold: bit-flip, phase and syndrome error probabilities each below
  1.14% per round.

**Bearing.**

- Skoric2023 and Tan2023 both trace their windows to this method **[lit]**. Liyanage2024
  calls the 2d-window/d-commit scheme "the prevalent method" for decoding indefinitely
  **[lit]**.
- For the qutrit code, "T ≫ L" becomes "buffer of a few rounds". It is irrelevant for
  compute cost at ≤ 23 rounds but relevant as a model of a streaming decoder
  **[conjecture]**.

### 4.5 Terhal 2015 [Terhal2015]

**Citation.** B. M. Terhal, "Quantum error correction for quantum memories", *Rev. Mod.
Phys.* 87, 307 (2015), published 7 Apr 2015, DOI 10.1103/RevModPhys.87.307.
arXiv:1302.3428 (v7, 10 Apr 2015, final). Fetched: https://arxiv.org/abs/1302.3428,
https://arxiv.org/pdf/1302.3428,
https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.87.307.

**Extracted facts [lit]** (Sec. II.G.3 "The (Logical) Pauli frame" and Sec. III).

- "in principle it is never necessary to physically correct Pauli errors to map back to
  the +1 eigenspace of the stabilizer S. This is because any syndrome eigenspace of S is a
  good code and we simply need to know which code space we are using."
- **The explicit Clifford statement:** "if the quantum circuit entirely consists of
  Clifford gates, both at the logical as well as at the physical level, the classical
  information of the logical and physical Pauli frame does not necessarily need to be
  available during the execution of the circuit as the entire Pauli frame can be commuted
  through the circuit (as Clifford gates map Paulis onto Paulis). The Pauli frame simply
  alters the interpretation of the final measurement outcome of the computation. This is
  different if the circuit consists of non-Clifford gates".
- A Pauli X before a teleported T becomes a Clifford error, so "one does not need to
  physically implement any logical or elementary Pauli operation, but one does need to
  know the logical Pauli before one can proceed further with the computation."
- Only the part of the frame that affects `M_Z` must be known in time. Other parity checks
  "can be processed later".
- **Backlog argument.** If `r_gen/r_proc = f > 1`, the record to process before the k-th T
  gate grows as `D_k = f^k D_1`, an exponential slowdown. The frame must be "'retarded' by
  only a constant amount of time".
- **Memory contrast:** "a quantum memory (including Clifford gates) in which the
  computation never has to wait for the classical processing of the logical Pauli frame
  ... it would just mean that the computation, including the processing of syndrome data
  is finished in time f T_store which is just a constant slow-down."
- Sec. III: "As one never needs to physically do any correction ... the measurement record
  can be stored and the data record can be processed at leisure and used to interpret a
  final M_X or M_Z measurement".
- Caveats: "qubit coding does not directly deal with leakage errors". Leakage reduction
  units such as teleportation convert leakage into regular errors. Transmons are "weakly
  anharmonic multi-level systems".
- Mentions qudit stabilizer surface codes (Bullock and Brennen) but analyses qubits.

**Bearing.**

- This is the most explicit statement of Summary items 1 and 2 among the fetched papers
  **[lit: Terhal2015]**.
- LRB, even with randomly chosen logical Cliffords, is the "quantum memory including
  Clifford gates" case. Offline processing costs at most a constant slowdown and no
  fidelity **[analytic; lit: Terhal2015]**.

### 4.6 Fowler, Mariantoni, Martinis, Cleland 2012 [Fowler2012SC]

**Citation.** A. G. Fowler, M. Mariantoni, J. M. Martinis, A. N. Cleland, "Surface codes:
Towards practical large-scale quantum computation", *Phys. Rev. A* 86, 032324 (2012), DOI
10.1103/PhysRevA.86.032324. arXiv:1208.0928 (v2 27 Oct 2012). Fetched:
https://arxiv.org/abs/1208.0928, https://arxiv.org/pdf/1208.0928.

**Extracted facts [lit].**

- Sec. IX "Software-implemented X_L and Z_L": "The logical operators X_L and Z_L ... are
  not actually implemented in the surface code hardware! These operations are handled
  entirely by the classical control software". The reason: "a hardware-based solution will
  always have a higher error rate than one implemented in the control software." The
  operator is "commuted through each subsequent logical operation in the algorithm until a
  second identical operation is called for ... or until a measurement of the logical qubit
  is performed, in which case the operator is applied to the measurement outcome".
- Byproduct operators of logical moves are also "track[ed] ... in the control software" and
  only flip measurement signs.
- Errors are corrected "in the classical control software, by applying corrective phase-
  and bit-flips to the qubit measurements".
- For T via `|A_L>`: when M_Z = −1, apply S_L (feedforward). Byproducts commute through
  S_L as `S_L X_L → Z_L X_L S_L`; the Clifford S_L itself must be applied.

**Bearing.**

- In the repository the logical X and Z tokens are noisy physical Paulis on {0,3} and
  {0,2}, each with 0.1p single-qutrit noise per gate (per `00-brief.md`, read from the
  code).
- On hardware they could be virtual (frame-only), which would remove that noise
  **[analytic]**.
- Doing so changes what the LRB measures: logical Pauli gates become perfect. Whether to
  do it is a protocol choice for the study **[conjecture]**.

### 4.7 Fowler 2012, "Time-optimal quantum computation" [Fowler2012TO]

**Citation.** A. G. Fowler, "Time-optimal quantum computation", arXiv:1210.4626 (v1 17 Oct
2012, v3 2 Feb 2013; no journal reference shown on arXiv). Fetched:
https://arxiv.org/abs/1210.4626, https://arxiv.org/pdf/1210.4626.

**Extracted facts [lit].**

- "All known forms of universal error corrected quantum computation use intermediate
  measurement results to determine future unitary operations."
- With `TX = XT†`, "one must know after applying T what byproduct operators were present
  to know what the result is and whether a corrective S gate is required. The application
  of S, if required, cannot reasonably be delayed as S does not gracefully commute through
  H or the target qubit of CX, and changes the probability distribution of an X basis
  measurement."
- With selective-destination and selective-source teleportation, Clifford layers plus
  layers of independent T gates run in about one physical measurement time per T layer.
  This assumes "fast classical processing and classical communication".
- Example: 100 ns measurement, 200 ns cycle; canonical T at 76.5 µs vs 100 ns.

**Bearing.** Reaction time is the clock of non-Clifford computation. LRB with Clifford
twirls has no T layers, so its clock is set by the QEC cycle alone **[analytic]**.

### 4.8 Chamberland, Iyer, Poulin 2018 [Chamberland2018]

**Citation.** C. Chamberland, P. Iyer, D. Poulin, "Fault-tolerant quantum computing in the
Pauli or Clifford frame with slow error diagnostics", *Quantum* 2, 43 (2018), DOI
10.22331/q-2018-01-04-43. arXiv:1704.06662 (v2 25 Dec 2017). Fetched:
https://arxiv.org/abs/1704.06662, https://arxiv.org/pdf/1704.06662,
https://api.crossref.org/works/10.22331/q-2018-01-04-43.

**Extracted facts [lit].**

- Keeping the correction in software "saves us from performing additional gates on the
  system, thus removing potential sources of errors."
- "for circuits containing only Clifford gates, all Pauli recovery operators can be
  tracked in classical software without ever exiting the Pauli group". The frame update
  `P' = U P U†` costs O(n²). "as long as we only apply Clifford gates, there is no need to
  actively error-correct".
- Through T, the output correction is a logical Clifford times a Pauli (their Appendix A).
  A buffer of EC rounds is inserted before the next T or two-qubit gate, the Clifford C is
  learned, and then C† is applied.
- Measurement latency t_l and repetition time t_r are distinguished. Schemes tolerate
  large t_l but need small t_r.
- Measurements in some solid-state and ion architectures are "10 to 1000 times slower
  than gates times". A decoding cycle "can take well above 1 µs" on a standard processor.
- Slow decoding can be parallelized to match the syndrome rate.
- They also define a Clifford-frame variant.

**Bearing.**

- For a qutrit version, the T-gate buffer construction carries over with qutrit Cliffords,
  because the qutrit T is in the third level of the hierarchy (Section 1.3)
  **[numeric for the hierarchy property; conjecture for the protocol]**.
- Not needed for LRB, which has no non-Clifford gate **[analytic]**.

### 4.9 Skoric, Browne, Barnes, Gillespie, Campbell 2023 [Skoric2023]

**Citation.** L. Skoric, D. E. Browne, K. M. Barnes, N. I. Gillespie, E. T. Campbell,
"Parallel window decoding enables scalable fault tolerant quantum computation", *Nat.
Commun.* 14, 7040 (2023), published 3 Nov 2023, DOI 10.1038/s41467-023-42482-1.
arXiv:2209.08552 (v2 4 Feb 2023). Fetched: https://arxiv.org/abs/2209.08552,
https://arxiv.org/pdf/2209.08552, https://www.nature.com/articles/s41467-023-42482-1.

**Extracted facts [lit].**

- **The explicit offline statement:** "An offline decoder collects and stores all the
  syndrome data generated during a hardware run (often called a shot) and then performs
  decoding as a post-processing step. Offline decoding is sufficient for computations
  consisting solely of Clifford gates (e.g. CNOT and Hadamard gates)." For T by
  teleportation, the S correction "can only be reliably made after we decode the syndrome
  data", so "online, or real-time, decoding is necessary for useful quantum computation".
  Online decoders "need only react fast enough to enable feed-forward and Clifford
  correction". Global decoding "is not compatible with the real-time feedback required for
  non-Clifford gates".
- Committing means "making a final correction decision for potential error locations, with
  all corrections performed in software."
- **Sliding window.**
  - Commit region of n_com rounds plus buffer of n_buf rounds. The top boundary is rough,
    and artificial defects are created where committed chains cross into the buffer.
  - Avoiding backlog needs `τ_W < n_com τ_rd`. For the surface code `τ_W = Ω(n_com d²)`,
    so "we are restricted to `d² ≤ O(τ_rd)`".
  - With `n_buf = n_com = d` there is "no noticeable increase in logical error rate".
  - The inner decoder must return an approximately minimum-weight correction (MWPM, UF).
    Homology-class decoders (tensor-network or MLD-type) can return a high-weight
    representative. That creates extra artificial defects and logical errors in the next
    window: "additional modifications beyond those discussed in this work would be needed
    to use homology-based inner decoders."
  - **[conjecture]** This matters for the qutrit study: MLD over the DEM is the natural
    decoder for a d = 2 code. Inside a committed-window scheme it would need a
    minimum-weight representative or a modified commit rule.
- **Parallel window.**
  - Layer A windows are buffer–commit–buffer with `n_buf = n_com = w = d` (n_W = 3d) and
    are decoded concurrently. Layer B windows (3d, smooth boundaries) fill in afterwards.
  - Throughput condition `N_par (n_com + n_W) τ_rd ≥ 2 τ_W`. Response time after each T
    is `τ = N_par (n_com + n_W) τ_rd ≈ 2 τ_W`. Logical clock `τ_clock = d τ_rd + τ`.
  - With auto-corrected T teleportation the delay becomes about `⌈τ/(d τ_rd)⌉` extra
    logical qubits. Example: 100 logical qubits and τ_clock = 10 d τ_rd give 10× time
    without auto-correction against 9 extra logical qubits (1.09× qubits) with it.
- Numerics (phenomenological noise 2%): parallel-window MWPM is within error of global
  MWPM. N_par = 16 gives more than 10× decoding frequency.
- Hardware context (quoted): round times 1.1 µs (Krinner et al.) and 921 ns (Google 2023).
  "both aforementioned teams instead performed offline decoding". LILLIPUT processes a
  round in 300 ns for d ≤ 5.
- Windows smaller than d may be adequate for stochastic noise (O(polylog d) clusters), but
  this is open.

**Bearing.**

- Supports C6 and Section 3 **[lit: Skoric2023]**.
- The study's LRB is in the "solely Clifford" class, so the paper's own criterion says
  offline decoding suffices **[analytic; lit: Skoric2023]**.

### 4.10 Tan, Zhang, Chao, Shi, Chen 2023 [Tan2023]

**Author check.** The task listed "Tan, Niu, Gidney (check authors)". That is wrong. The
arXiv page, the PDF and Crossref all give **Xinyu Tan, Fang Zhang, Rui Chao, Yaoyun Shi,
Jianxin Chen** (Alibaba Quantum Laboratory). Tan and Zhang contributed equally. The PDF
thanks Xiaotong Ni "for the insight".

**Citation.** X. Tan, F. Zhang, R. Chao, Y. Shi, J. Chen, "Scalable surface-code decoders
with parallelization in time", *PRX Quantum* 4, 040344 (2023), published 15 Dec 2023, DOI
10.1103/PRXQuantum.4.040344. arXiv:2209.09219 (v2 29 Sep 2022). Fetched:
https://arxiv.org/abs/2209.09219, https://arxiv.org/pdf/2209.09219, Crossref title query.

**Extracted facts [lit].**

- Most architectures' threshold and overhead estimates "mostly assume instantaneous
  classical computations".
- A non-Clifford gate by magic state "typically involves a classically controlled Clifford
  correction".
- Batch decoders: "decoding is usually an offline process ... decoding will not start until
  the entire quantum circuit finishes". They need O(p d² n) storage. "it is unclear how to
  perform (especially non-Clifford) logical operations using batch decoders in the context
  of magic state injections".
- Detectors are the XOR of consecutive syndrome rounds (their eq. (1)). This is the qubit
  analogue of the frame-relative detectors recommended in Section 1.4.
- **Forward decoder** (DKLP's overlapping recovery reformulated): w = s + b, sequential.
  Latency accumulates if a window takes longer than s cycles. Step size s = b, w = 2b keeps
  a constant overhead relative to batch decoding.
- **Sandwich decoder**: type-1 windows have buffers on both sides (w = s + 2b) and are
  independent, hence parallel. Seams are decoded after neighbours.
- Benchmarks: s = (d+1)/2, w = 3s, circuit-level depolarizing noise, d = 3…17.
  Thresholds 0.68% (MWPM sandwich) vs 0.70% (MWPM batch), and 0.55% vs 0.554% (UF).
- Logical-error ansatz: `1 − 2 p_L(i) = [1 − 2 p_L(1)]^i`, i.e. independent per-cycle
  flips.
- LILLIPUT (quoted): lookup tables built offline with MWPM, limited to d ≤ 5 and window
  ≤ 3.
- "further theoretical justification or numerical validation is needed to fully evaluate
  our sandwich decoder in real-time decoding for fault-tolerant computation."

**Bearing.**

- The independence ansatz is the qubit form of the factorization argument in Section 5
  (Q2) **[conjecture for qutrits]**.
- Their detector definition (XOR of consecutive rounds) is the qubit form of the
  frame-relative detectors recommended in Section 1.4 **[lit: Tan2023; analytic]**.

### 4.11 Battistel et al. 2023 [Battistel2023]

**Citation.** F. Battistel, C. Chamberland, K. Johar, R. W. J. Overwater, F. Sebastiano,
L. Skoric, Y. Ueno, M. Usman, "Real-time decoding for fault-tolerant quantum computing:
progress, challenges and outlook", *Nano Futures* 7, 032003 (2023), DOI
10.1088/2399-1984/aceba6. arXiv:2303.00054 (v2 22 May 2023). Fetched:
https://arxiv.org/abs/2303.00054, https://arxiv.org/pdf/2303.00054,
https://api.crossref.org/works/10.1088/2399-1984/aceba6. The IOP page was blocked by a bot
check.

**Extracted facts [lit].**

- "The correction established by the decoder is needed only for the next-in-line logical
  non-Clifford gate (for example to determine whether a logical S correction has to be
  applied during magic-state injection), otherwise it can be tracked by the decoder in the
  Pauli frame (or leakage frame) until needed." This cites Riesebos2017 and, for the
  leakage frame, Suchara, Cross and Gambetta 2015, which I did not fetch.
- Their framework figure: the decoder "updates its estimate of the Pauli frame ... (as well
  as the leakage frame possibly). Physical correction pulses are sent to the QPU when
  requested because of an impending logical non-Clifford gate".
- Throughput requirements: "≲ O(1) µs for a surface code using superconducting transmon
  qubits", "≲ O(1) ms for silicon spin-qubits", "beyond O(100) ms for ion traps". 1 µs is
  the usual benchmark. This can be relaxed "to a few µs" with parallelization or Auto-T
  gadgets.
- Their Fig. 2 takes 1 ms for ions (1000× transmons) while noting the 200 ms QEC cycle of
  RyanAnderson2021.
- Status at the time: real-time decoding was demonstrated only on ions, with precomputed
  lookup tables. Superconducting demonstrations decoded offline.
- Their definitions differ from the usual ones. Throughput is the decode time after the
  outcomes arrive; latency is the transport time. Throughput is "even more important than
  latency".
- About 1 µs rounds give "a few tens of Mbps of syndrome data per logical qubit".
- FPGA bus protocol: "at most a couple hundreds of nanoseconds". Ideal placement is
  "alongside the control electronics, integrated with very low latency in the order of
  10s of nanoseconds".
- MLD is ♯P-complete in general. Lookup tables are fast but exponential in memory.

**Bearing.** Summary items 3 and 6. A "leakage frame" is the qutrit-transmon analogue to
watch: leakage to the 4th level is outside the Weyl model **[conjecture]**.

### 4.12 Google Quantum AI 2025 [GoogleQAI2025]

**Citation.** Google Quantum AI and Collaborators (R. Acharya, L. Aghababaie-Beni,
I. Aleiner, T. I. Andersen, M. Ansmann et al.), "Quantum error correction below the surface
code threshold", *Nature* 638, 920–926 (2025), published online 9 Dec 2024, DOI
10.1038/s41586-024-08449-y. arXiv:2408.13687 (v1 24 Aug 2024). Fetched:
https://arxiv.org/abs/2408.13687, https://arxiv.org/pdf/2408.13687 (main text and
supplement), https://www.nature.com/articles/s41586-024-08449-y (journal abstract with the
same numbers).

**Extracted facts [lit].**

- Abstract: below threshold "when decoding in real time, achieving an average decoder
  latency of 63 microseconds at distance 5 up to a million cycles, with a cycle time of 1.1
  microseconds." Also Λ = 2.14 ± 0.02 and 0.143% error per cycle at d = 7 (offline
  decoders).
- Why real time: "some logical operations are non-deterministic; they depend on logical
  measurement outcomes that must be correctly interpreted on the fly". A backlog causes "an
  exponential blow-up in computation time".
- Pipeline:
  - control electronics classify readouts into bits;
  - low-latency Ethernet to a workstation;
  - detections are formed there and streamed through shared memory to a specialized Sparse
    Blossom, parallelized like Fusion Blossom;
  - the graph buffer is constant-size;
  - greedy reweighting accounts for Y correlations.
- Latency definition: from "receiving the final cycle of syndrome measurements" to "the
  decoder returns its correction". 63 ± 17 µs "independent of the length of the experiment
  (up to 1.1 seconds)". This "will eventually determine the reaction time ... but does not
  yet include feedback into the logical circuit."
- Accuracy cost of real time on the 72-qubit chip: ε5 = 0.35% ± 0.01%, Λ = 2.0 ± 0.1
  (real time) vs ε5 = 0.269% ± 0.008%, Λ = 2.18 ± 0.09 (offline NN decoding of the same
  data). Real time needs < 1.1 µs per cycle against the NN's 24 µs per cycle.
- Supplement:
  - "offline mode ... is insufficient for experiments involving fault-tolerant
    constructions, such as gate teleportation that use feed-forward".
  - `T_react = T_decode + T_control` (their eq. (1)). "Inverse reaction time plays the role
    of clock speed".
  - `T_decode = T_input + T_software + T_output`, with `T_input` < 10 µs.
  - Correlated Parallel Blossom: blocks of M cycles (M = 10 for d = 3, 5; M = 90 for the
    d = 29 repetition code), two fusion layers, a heralded failure if regions remain
    matched to block boundaries, and a 128-block buffer.
  - "The overall prediction of the decoder is then the bitwise XOR of the predicted logical
    observable bitmask for each block."
- Their simulated pre-training data include SI1000 noise, the same family as the brief's
  noise model.

**Bearing.**

- Real-time decoding costs accuracy relative to the best offline decoder. For a
  Clifford-only LRB there is no reason to pay it **[analytic + lit]**.
- The per-block XOR of logical predictions becomes, for qutrits, a sum mod 3 of per-window
  logical shifts. This is the structure behind the factorization conjecture in Section 5
  **[analytic for the additivity; conjecture for independence]**.

### 4.13 Barber et al. 2025, Riverlane [Barber2025]

**Citation.** B. Barber, K. M. Barnes, T. Bialas, O. Buğdaycı, E. T. Campbell,
N. I. Gillespie, K. Johar, R. Rajan, A. W. Richardson, L. Skoric, C. Topal, M. L. Turner,
A. B. Ziad, "A real-time, scalable, fast and resource-efficient decoder for a quantum
computer", *Nat. Electron.* 8, 84–91 (2025), published 7 Jan 2025, DOI
10.1038/s41928-024-01319-5. The arXiv title is "...fast and highly resource efficient
decoder...": arXiv:2309.05558 (v2 24 Sep 2024). Fetched: https://arxiv.org/abs/2309.05558,
https://arxiv.org/pdf/2309.05558, https://www.nature.com/articles/s41928-024-01319-5.

**Extracted facts [lit].**

- Most experiments decode offline, and "This type of decoding cannot support logic
  branching which is required to implement certain non-Clifford gates".
- Collision Clustering (CC) is a union-find variant with the same clusters, and hence the
  same accuracy.
- Noise model: two-qubit depolarizing p, one-qubit, measurement and reset depolarizing p/10,
  measurement flip p, at p = 0.1%. This is an SI1000-like weighting.
- FPGA (Xilinx Ultrascale+ XCVU3P, about 400 MHz):
  - Table I per-round time (averaged over 1e5 shots, divided by d): 0.07, 0.06, 0.06,
    0.07, 0.11, 0.16, 0.25, 0.37, 0.55, 0.81, 1.18 µs for d = 3…23;
  - 881 qubits (d = 21) in 810 ns using 4.5% of the logic LUTs and 10 KB of memory;
  - threshold 0.78%;
  - < 13 KB of memory at d = 23.
- ASIC (12 nm, signed off): d = 7 at 10 ns/round and 2.75 mW; d = 23 at 240 ns/round and
  7.85 mW; 0.06 mm²; about 8 mW total.
- "To preserve a logical state indefinitely, the decoder must be able to handle data being
  streamed in. We save this investigation for future work." The numbers are per-round
  times for whole-block decoding.
- Comparison: Helios is faster per round (11.5 ns at d = 21) but needs about 900k LUTs and
  240k registers.

**Bearing.** Hardware decoders are already far faster than this study would ever need. Its
decoding problem is about 4 checks per round **[analytic]**.

### 4.14 Caune et al. 2024 / 2026, Riverlane and Rigetti [Caune2026]

**Citation.** L. Caune, L. Skoric, N. S. Blunt, A. Ruban, J. McDaniel, J. A. Valery,
A. D. Patterson, A. V. Gramolin, J. Majaniemi, K. M. Barnes, T. Bialas, O. Buğdaycı,
O. Crawford, G. P. Gehér, H. Krovi, E. Matekole, C. Topal, S. Poletto, M. Bryant,
K. Snyder, N. I. Gillespie, G. Jones, K. Johar, E. T. Campbell, A. D. Hill,
"Demonstrating real-time and low-latency quantum error correction with superconducting
qubits", arXiv:2410.05202 (v1 7 Oct 2024); *Nat. Commun.* 17, 7383 (2026), published 1 Jun
2026, DOI 10.1038/s41467-026-73331-6. Fetched: https://arxiv.org/abs/2410.05202,
https://arxiv.org/pdf/2410.05202 (numbers below are from v1),
https://www.nature.com/articles/s41467-026-73331-6 (journal abstract with the same headline
numbers).

**Extracted facts [lit].**

- Logical branching is "a logical operation conditional on a corrected observable, which
  is computed by combining the measured value of the observable and the logical correction
  returned by the decoder". In the qutrit setting this would be `x0 + x3` plus the
  decoder's shift.
- Real-time decoding means data are "processed as soon as it becomes available. This
  contrasts with offline decoding, where the decoding can be performed at any time after
  the data has been collected."
- Setup: an 8-qubit stability experiment on Rigetti Ankaa-2, with CC on an FPGA integrated
  in the control system. One QEC round takes about 1.7 µs.
- Mean decoding time per round: 0.44 µs (5 rounds) to 0.79 µs (25 rounds). It grows with
  rounds because later rounds are noisier (heating, leakage). The few runs above 1 µs per
  round had worse logical error.
- Full decoding response, from the final extraction round to the conditional X gate:
  9.6 µs for 9 rounds = 6.5 µs decoding + 3.1 µs communication and control. Of the latter,
  1.4 µs is readout reaching the decoder and 1–1.5 µs (250–370 FPGA cycles) is control
  logic.
- The timing was cross-checked using the qubit's T1 decay as a clock.
- Comparison with GoogleQAI2025: "an average of 63 µs to process the final 10 QEC cycle
  block" plus "an additional 10 µs communication latency", "significantly slower", with a
  caution against direct comparison.
- Main-text experiments did "not involve resetting ancilla qubits between QEC rounds".
  "Conditional (or measurement-based) resets ... cannot improve the performance of QEC
  circuits." Unconditional resets double the number of tolerable measurement errors in a
  stability experiment. DDROP unconditional reset was demonstrated with a software
  decoder.

**Bearing.**

- Among the papers fetched here, this is the only measurement of a full feedback loop
  (decode then act) on superconducting hardware: about 10 µs **[lit: Caune2026]**.
- For a qutrit LRB that would ever need branching, this sets the scale of `T_react`
  **[conjecture for transfer to qutrits]**.

### 4.15 Liyanage et al. 2023 and 2024 [Liyanage2023, Liyanage2024]

**Citations.**

- N. Liyanage, Y. Wu, A. Deters, L. Zhong, "Scalable Quantum Error Correction for Surface
  Codes Using FPGA", *2023 IEEE Int. Conf. on Quantum Computing and Engineering (QCE)*,
  pp. 916–927, DOI 10.1109/QCE57702.2023.00106. arXiv:2301.08419 (v2 15 May 2023). A
  1-page abstract also appeared at FCCM 2023, DOI 10.1109/FCCM57271.2023.00045.
- N. Liyanage, Y. Wu, S. Tagare, L. Zhong, "FPGA-Based Distributed Union-Find Decoder for
  Surface Codes", *IEEE Trans. Quantum Eng.* 5, 1–18 (2024), DOI 10.1109/TQE.2024.3467271.
  arXiv:2406.08491 (v1 20 Mar 2024, v2 2 Oct 2024).

Fetched: https://arxiv.org/abs/2301.08419, https://arxiv.org/pdf/2301.08419,
https://arxiv.org/abs/2406.08491, https://arxiv.org/pdf/2406.08491, plus Crossref records
for both DOIs.

**Extracted facts [lit].**

- Helios: a distributed union-find with one processing element per decoding-graph vertex,
  arranged as a hybrid tree-grid on a Xilinx VCU129.
- Average time per round decreases with d, "a first time for a quantum error decoder", so
  there is no growing backlog.
- 2023: d up to 21. "average decoding time is 11.5 ns per measurement round under
  phenomenological noise of 0.1%". At d = 13, 90% of trials finish within 250 ns and
  99.99% within 370 ns. The latency distribution is right-skewed.
- 2024: adds circuit-level noise. 23.7 ns per round at d = 17 (circuit-level). d = 51 at
  544 ns per round (resource-efficient mapping).
- 2024, d = 13 latency distribution: 97% of trials within 250 ns, 99.99% within 510 ns,
  mean about 194 ns. The longest observed latency was 920 ns, against a theoretical worst
  case of 857 µs.
- 2024 sliding window: "2d rounds are decoded simultaneously, but corrections are only
  committed for the oldest d rounds. Subsequently, the decoding window advances by d
  rounds". At d = 13 the latency goes 194 → 230 ns with 371K vs 166K LUTs, "56 times faster
  than the rate of measurement".
- Caveat: phenomenological results decode only X errors, and the simulation generates
  syndromes on-chip.

**Bearing.** Commit-plus-buffer windows (Section 3) cost little latency in hardware
**[lit: Liyanage2024]**. Decoder speed has no bearing on the accuracy of an offline LRB
analysis **[analytic]**.

### 4.16 Ryan-Anderson et al. 2021, Honeywell / Quantinuum [RyanAnderson2021]

**Citation.** C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, A. Hankin, J. P. Gaebler,
D. Francois, A. Chernoguzov, D. Lucchetti, N. C. Brown, T. M. Gatterman, S. K. Halit,
K. Gilmore, J. Gerber, B. Neyenhuis, D. Hayes, R. P. Stutz, "Realization of Real-Time
Fault-Tolerant Quantum Error Correction", *Phys. Rev. X* 11, 041058 (2021), DOI
10.1103/PhysRevX.11.041058. arXiv:2107.07505 (v1 15 Jul 2021). Fetched:
https://arxiv.org/abs/2107.07505, https://arxiv.org/pdf/2107.07505, Crossref title query.

**Extracted facts [lit].**

- [[7,1,3]] color code on a 10-qubit QCCD trap.
- "a real-time decoder to determine any necessary corrections that are done either as
  software updates to the Pauli frame or as physically applied gates".
- Decoder: two lookup tables (flag-based, FT to hook and measurement errors).
- "Applying corrections physically via noisy gates can potentially induce more errors ...
  instead done with an essentially perfect software correction." The Pauli frame is 2 bits
  per logical qubit, updated at the end of each QEC cycle.
- **Direct experimental comparison (the only one fetched):**
  - |+>_L, one QEC cycle, then the correction applied physically according to the Pauli
    frame, then logical S, a further cycle and a Y-basis measurement: logical fidelity
    93(2)%;
  - the same with the frame instead rotated through S in software: 92(1)%;
  - "not significantly different". S stands in for T.
- "Pauli frame updates cannot always be used ... since Pauli operators have non-trivial
  transformations under conjugation by non-Clifford gates".
- Timing: logical SPAM < 60 ms, each QEC cycle < 200 ms (transport < 70 ms, the rest mostly
  cooling). Conditional information is distributed by a microprocessor, adding < 250 µs in
  the worst case.
- Average logical error per QEC cycle: 1.75(4)e-2.

**Bearing.** This experiment is consistent with the Section 1.1 equivalence within its 1–2%
error bars, on hardware whose noise is not exactly Pauli **[lit: RyanAnderson2021]**.
Ion-trap cycle times make real-time decoding of a small code easy; a table suffices
**[lit: RyanAnderson2021, Battistel2023]**.

### 4.17 Gehér, Jastrzebski, Campbell, Crawford 2025 [Geher2025]

**Citation.** G. P. Gehér, M. Jastrzebski, E. T. Campbell, O. Crawford, "To reset, or not
to reset—that is the question", *npj Quantum Inf.* 11, 39 (2025), DOI
10.1038/s41534-025-00998-y. arXiv:2408.00758 (v2 5 Mar 2025). Fetched:
https://arxiv.org/abs/2408.00758, https://arxiv.org/pdf/2408.00758, Crossref bibliographic
query.

**Extracted facts [lit].**

- Three reset classes:
  - unconditional (a non-unitary map to |0>);
  - conditional (a bit flip if the outcome is 1, whose duration "depends on both the
    bit-flip gate duration and the classical electronics latency");
  - no reset, where "we may simply track the effect of the conditional bit-flip in
    software".
- With conditional reset or no reset, a misclassification both records a wrong outcome
  and applies a wrong flip, which is "a correlated pair of errors".
- "For memory experiments, we confirm numerically that resetting provides no benefit."
- For logical operations (lattice surgery, stability experiments), unconditional reset
  tolerates twice as many measurement errors. Without it, operations must last up to twice
  as long.
- No reset wins "when the reset duration is greater than approximately 100 ns and the
  physical error probability is greater than approximately 10^−2.5 ≈ 0.003" in their noise
  model.

**Bearing.**

- The ancilla reset in the LRB circuits is unconditional, with its own 2p shift and 2p
  data idle (per `00-brief.md`, read from the code). The final round has no reset layer
  **[numeric, Section 1.4]**.
- On qutrit hardware, a conditional reset would be a feedback loop (`X^−m` conditioned on
  m ∈ {1, 2}) **[analytic]**.
- The feedback-free alternatives are unconditional reset or no reset plus software
  tracking. For no reset, the outcome of an unreset ancilla is `m_r = s_r + m_{r−1}`
  (mod 3) for both check types. For the X-check this follows from
  `H_INV · Z^s · H |m> = |m + s>`, up to the sign convention of H **[analytic]**.
- LRB is a memory-type experiment (no logical measurement mid-circuit). Their memory
  result suggests no reset costs nothing here **[conjecture for qutrits]**.

### 4.18 Bosonic qudits: Sivak et al. 2023 and Brock et al. 2025 [Sivak2023, Brock2025]

**Citations.**

- V. V. Sivak, A. Eickbusch, B. Royer, S. Singh, I. Tsioutsios, S. Ganjam, A. Miano,
  B. L. Brock, A. Z. Ding, L. Frunzio, S. M. Girvin, R. J. Schoelkopf, M. H. Devoret,
  "Real-time quantum error correction beyond break-even", *Nature* 616, 50–55 (2023), DOI
  10.1038/s41586-023-05782-6. arXiv:2211.09116.
- B. L. Brock, S. Singh, A. Eickbusch, V. V. Sivak, A. Z. Ding, L. Frunzio, S. M. Girvin,
  M. H. Devoret, "Quantum error correction of qudits beyond break-even", *Nature* 641,
  612–618 (2025), DOI 10.1038/s41586-025-08899-y. arXiv:2409.15065.

Fetched: https://arxiv.org/pdf/2211.09116, https://www.nature.com/articles/s41586-023-05782-6,
https://arxiv.org/pdf/2409.15065, https://www.nature.com/articles/s41586-025-08899-y.

**Extracted facts [lit].**

- Sivak: the GKP qubit QEC cycle is "2 × 4.924 µs, which includes execution of unitary
  gates, ancilla measurements, and real-time processing and decision making by the
  controller". The conditional unitary U_b is "realized as only a virtual rotation",
  implemented "dynamically on the field-programmable gate array (FPGA) in 448 ns".
- Brock: the first error-corrected logical qutrit (d = 3) and ququart (d = 4), with GKP
  gains of 1.82 ± 0.03 (qutrit) and 1.87 ± 0.03 (ququart). The sBs protocol is
  "autonomous, only requiring reset of the ancilla between rounds", plus cavity phase
  updates; 7 µs per round.
- Brock, the qudit subtlety: for even d the extra displacement `D(−√(πd)/2)` "is the ideal
  Pauli operator X_d^{d/2}, the effect of which can be tracked in software". For odd d it
  "brings us outside the code space, an effect we have to reverse before moving on to
  measure S_Z", which is done with a different phase-update schedule.

**Bearing.**

- For GKP qutrits, correction is autonomous. The classical side between rounds is an
  ancilla reset and a scheduled phase (frame) update, not a decoder
  **[lit: Brock2025; Sivak2023 for the qubit case]**.
- Odd d removes one software-tracking option. That is a qutrit-specific constraint for
  Q3 **[lit]**.

### 4.19 Qudit background: Gottesman 1998; Howard and Vala 2012 [Gottesman1998, HowardVala2012]

**Citations.**

- D. Gottesman, "Fault-Tolerant Quantum Computation with Higher-Dimensional Systems",
  arXiv:quant-ph/9802007 (2 Feb 1998); *Chaos Solitons Fractals* 10, 1749–1758 (1999), DOI
  10.1016/S0960-0779(98)00218-5.
- M. Howard, J. Vala, "Qudit versions of the qubit 'pi-over-eight' gate", *Phys. Rev. A*
  86, 022316 (2012), DOI 10.1103/PhysRevA.86.022316. arXiv:1206.1598.

Fetched: https://arxiv.org/abs/quant-ph/9802007, https://arxiv.org/abs/1206.1598 (abstract
pages only).

**Extracted facts [lit].**

- Gottesman extends stabilizer codes to d-dimensional systems and proves that "universal
  fault-tolerant computation is possible with any higher-dimensional stabilizer code for
  prime d."
- Howard and Vala: the qubit π/8 gate is "the simplest example from the third level of the
  Clifford hierarchy". They derive explicit qudit versions for prime d; these are diagonal,
  and the group they generate differs for d = 2, 3 and > 3. They are proposed for qudit
  magic-state distillation.

**Bearing.**

- These are the basis for treating qutrit frames exactly like qubit frames: the Weyl group
  replaces the Pauli group, and the qudit Clifford group normalizes it
  **[lit: Gottesman1998; analytic; numeric, Section 1.3]**.
- My numerical check that `diag(1, z, z^8)` maps X outside the Weyl group, but into the
  Clifford group, is consistent with the third-level property **[numeric]**. That
  matrix is my choice of representative; I did not check it against their explicit
  formula.

---

## 5. Implications for the study's questions

**Q1 (which decoders, how they compare).**

- **[analytic]** The repository's DEM sampling is already a frame simulation. Any decoder
  that maps a shot's detector record to a Z_3 logical shift, optionally with a reject
  flag, gives exactly the statistics a real-time hardware implementation with perfect
  instantaneous corrections would give.
- No real-time or windowed variant can beat the best offline decoder for this experiment
  **[analytic]**.
- Comparisons between decoders should therefore be made offline. "Real-time" only adds the
  constraint of committing early (Section 3) and the cost of latency idles
  **[analytic]**.
- Five unit faults per circuit at the end of the circuit are undetectable and flip the
  logical; their number does not depend on m **[numeric]**. They set an m-independent
  floor for every decoder, including LRB-D **[analytic]**.

**Q2 (non-Markovianity) [conjecture].**

- With correction and no rejection, the corrected shift is `S = sum_r S_r (mod 3)`. The
  fidelity is `|E[w^S]|`.
- If a decoder commits per window and the S_r are independent, then
  `E[w^S] = prod_r E[w^{S_r}]`, which decays exponentially in the number of windows. This
  is the qutrit form of Tan2023's ansatz `1 − 2p_L(i) = (1 − 2p_L(1))^i`.
- Departures from a single exponential then come from three sources:
  - correlations across window boundaries;
  - the unchecked inverse block, whose faults are all seen only at round m and grow with m
    (two-segment, not translation-invariant);
  - boundary rounds.
- Postselection conditions on the whole history, so its acceptance-weighted average need
  not factorize. Correction removes that conditioning, which is one mechanism by which it
  could mitigate the non-Markovian shape.
- A global offline decoder also uses future data. It does not reweight shots, but the
  per-round corrections are correlated with later syndromes.
- A test: compare the decay shape from a global decoder with that from a d-scaled
  committed-window decoder (n_com = n_buf = 2 rounds) on the same samples.

**Q3 (sustainable hardware).**

- **[analytic + lit]** Store raw records, about 4(m+1) trits per shot. Decode offline on a
  CPU and apply the frame to the logical readout. This needs no decoder in the feedback
  loop **[lit: Terhal2015, Skoric2023]**.
- Use unconditional reset or no reset with software tracking to avoid conditional-reset
  feedback **[lit: Geher2025, Caune2026]**.
- Logical X and Z can be made virtual (frame-only) if the protocol allows
  **[lit: Fowler2012SC]**.
- Leakage on transmon qutrits is outside the Weyl model. It needs either a leakage-aware
  decoder input (a "leakage frame", Battistel2023) or physical leakage reduction, which is
  a real-time element **[lit: Terhal2015, Battistel2023]**.
- If qutrit non-Clifford gates are ever added (magic-state T with Clifford fix-ups), the
  frame must be known before the gate. `T_react = T_decode + T_control`, with
  `T_decode` ~ ns for this code size on an FPGA, and `T_control` ~ µs on present
  superconducting stacks (Caune2026: 3.1 µs communication and control)
  **[conjecture for transfer to qutrits]**.
- Ion-trap and neutral-atom QEC cycles are "in excess of many milliseconds, leaving ample
  time for real-time decoding" **[lit: Battistel2023]**. A lookup-table decoder sufficed
  for the H1 color-code experiment **[lit: RyanAnderson2021]**.
- GKP qutrits correct autonomously, with the odd-d frame caveat **[lit: Brock2025]**.

## 6. Confirmations, corrections, and items I could not verify

- **Confirmed and fetched:** every paper named in the task. Knill's Nature title is
  "realistically noisy"; the arXiv title is "very noisy".
- "Tan, Niu, Gidney" is not the author list. It is Tan, Zhang, Chao, Shi, Chen (Section
  4.10).
- Riesebos2017: abstract only (ACM full text 403; TU Delft repository unreachable).
- Google's 63 µs is decoder latency without feedback. Caune's 9.6 µs includes feedback.
  The setups differ (Caune2026 cautions against direct comparison).
- Secondhand numbers: LILLIPUT 300 ns and d ≤ 5 (quoted by Skoric2023, Tan2023); Krinner
  1.1 µs and Google 921 ns rounds (quoted by Skoric2023); the Gidney–Ekerå 10 µs reaction
  time (quoted by Caune2026, GoogleQAI2025). I did not fetch these sources.
- Not fetched and not cited: Litinski 2019 (Auto-T gadgets), Suchara, Cross and Gambetta
  2015 (leakage frame), Sparse Blossom and Fusion Blossom. They are mentioned only as cited
  inside the fetched papers.
- Scripts: `L5-scripts/frame_equivalence_toy.py`, `L5-scripts/raw_detector_persistence.py`,
  `L5-scripts/undetected_logical.py`.
  - Run the latter two from the worktree with `module load python/3.11` and
    `PYTHONPATH=src`. They read baseline circuits read-only and generate nothing.
  - Use `OMP_NUM_THREADS=1`. Without it, numpy's threading made an earlier version of the
    toy script far too slow on the login node.
