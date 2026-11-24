# Verification of WP J (hardware protocol and DEM learning)

Adversarial verifier, 2026-10-28. Under review: `notes/J-hardware-protocol.md`,
`studies/correction-decoders/qubit/scripts/hardware_model.py`, `studies/correction-decoders/qubit/scripts/dem_learning.py`,
`results/J/`. My code is `studies/correction-decoders/qubit/scripts/verify_hw.py`. My outputs are in
`results/J_verify/` (one JSON per subcommand, logs in `logs/`).

Labels: **VA** verified analytically, **VN** verified numerically (by me),
**LIT** checked against a source I fetched in this session, **CONJ**
conjecture.

## Verdicts

| # | claim | verdict | one-line reason |
|---|---|---|---|
| 1 | The checks force a 3x3 square-lattice patch, unique up to symmetry | **confirmed** | My own search finds 200 embeddings in a 7x7 grid (25 placements x 8 symmetries), every one a 3x3 block |
| 2 | (2,3) closes a 5-cycle, so it is not native; d(2,3) = 4 in the patch and on all 4 Garnet patches | **confirmed** | Odd cycle 2-7-0-5-3-2; 0 embeddings with (2,3); IQM's own coupler list gives 4 patches x 8 labellings, all with d = 4 |
| 3 | SWAP(2,3) can be a relabelling | **confirmed** | (2 3)(5 7)(6 8) is a Tanner-graph automorphism that maps every check to its image under H_L; noiseless circuits are deterministic in both frames |
| 4 | A routed CZ(2,3) costs 8 CZ | **refuted as a cost of the logical gate** | J's 8-CZ construction is correct, but 4 CZ suffice on the code space. J's gadget also breaks the circuit distance (d = 1) |
| 6 | A check round costs 12 CZ and 1.8 µs | **12 CZ confirmed; 1.8 µs needs more evidence** | 1.3 µs of the 1.8 µs is the readout pulse of a different IQM device, plus an assumed integration time, plus a reset latency that is unverified |
| 8 | Readout-window idle = 4.7p in SI1000 terms | **formula and arithmetic confirmed; inputs need more evidence** | The twirl matches a Kraus calculation to 1e-16. The inputs come from mixed sources: a consistent T1/T2 pair gives 5.1p, and the plausible range is 3.8p to 9.9p |
| 9 | Garnet-model f: 0.7705 / 0.9033 / 0.9497 / 0.99701 / 0.99904 (+ ion, atom) | **confirmed (reproduced), with caveats** | Independent exact re-implementation and a fresh-seed re-run agree within error. The values depend on the fit range, and the routed values include first-order undetectable faults |
| 10 | Sparser checks are always better for correction | **refuted as a general statement; confirmed in the tested range** | With noiseless checks, MLD with a check every round beats check-free MLD (0.9264 vs 0.9034; native 0.98936 vs 0.98487) |
| 11 | Reset latency ≈ 0.07 f per µs; no reset ≥ zero-latency reset | **confirmed with corrections** | The slope is 0.068/µs only for 0-1 µs (0.036/µs at 2-4 µs). The no-reset advantage comes entirely from the assumed 3 % reset error and is a tie when e_reset = 0 |
| 13 | Offline Pauli-frame decoding is exact; no feed-forward needed | **confirmed (scoped)** | VA for Clifford LRB. J's main IQM model still uses feed-forward in its conditional reset |
| 14 | Helios supports early exit; IBM and IQM document none | **confirmed** | Quotes fetched from the Helios paper, the Guppy docs, IBM's dynamic-circuits page and OpenQASM 3 feature table, and IQM client 36.0.0 |
| 15 | DEM learning: ≤0.32 %; pairs-only 24-104 %; p x2 ≤1.5 %; split invisible | **confirmed as computed; ≤0.32 % needs a scope fix** | Two rows re-run bit-identically. ≤0.32 % holds only when the nominal model is the truth (2-77 % for heterogeneous truths). The pairs-only figure depends on the estimator |

## 1. Re-run of J's tests (VN)

`python3 -m pytest -q tests/test_qubit512_hw.py` on the shared allocation:
**26 passed** (9.4 s).

## 2. Claims 1-3: layout (VN, VA, LIT)

`verify_hw.py graph` builds the graph given in the brief (edges 0-5, 1-5, 3-5,
1-6, 2-6, 4-6, 0-7, 1-7, 2-7, 1-8, 3-8, 4-8). It matches the lead's
`STABILIZER_PAULIS`. Its degrees are (1: 4; 5-8: 3; 0, 2, 3, 4: 2), and it is
isomorphic to the 3x3 grid graph.

- **Embeddings.** networkx subgraph monomorphisms into a 7x7 grid: **200 =
  25 placements x 8 symmetries**. Every one has a 3x3 bounding box with qubit
  1 at the centre and 2, 3 at opposite corners. Manhattan distance and grid
  graph distance d(2,3) are both **4** in every embedding.
- **Why (VA).** The square lattice's only 4-cycles are unit squares. Each
  pair of neighbouring midpoint ancillas shares one corner, which forces the
  layout.
- **Adding (2,3).** The graph becomes non-bipartite, with odd cycle
  **2-7-0-5-3-2 (length 5)**, and has **0** embeddings in the grid.
- **Garnet.** I parsed the coupler list from IQM's own
  `qiskit-on-iqm/src/iqm/qiskit_iqm/fake_backends/fake_garnet.py`, downloaded
  as text on 2026-10-28.
  - 20 qubits and 30 couplers, identical to J's `GARNET_COUPLERS`; the graph
    is bipartite and embeds in a square lattice.
  - The Tanner graph has 32 embeddings = **4 patches x 8 labellings**, centred
    on QB10 / QB9 / QB11 / QB15.
  - **d(q2, q3) = 4 on the full chip for all 32 labellings.** Tanner + (2,3)
    has 0 embeddings.
- **Claim 3.** σ = (2 3)(5 7)(6 8) is an automorphism of the Tanner graph. In
  frame 1 the physical ancilla σ(a) has exactly the support of the image
  stabilizer (S_Z1 on a7, S_Z2 on a8, S_X1 on a5, S_X2 on a6), as in J's
  `ANC_OF_STAB`. My independent builder also uses the virtual SWAP and is
  noiselessly deterministic.
- **Scope note.** Uniqueness assumes one ancilla per check, directly coupled
  to its three data qubits (no bridge or SWAP routing in the checks), which is
  the protocol's design.

## 3. Claim 4: the routed CZ(2,3) (VN, VA)

**J's construction is correct.** My tableau tests reproduce it: 8 CZ, patch
couplers only, ancillas restored. As a cost of the logical S it is not
minimal, and it breaks fault tolerance.

**A 4-CZ gadget suffices (VA, VN).** On the code space z2 = z1 ⊕ z4 (from
S_Z2), so (-1)^{z2 z3} = (-1)^{z3 z1}(-1)^{z3 z4}. In other words,
CZ(2,3) = CZ(1,3)CZ(3,4) whenever S_Z2 = +1. Qubits 1, 3 and 4 are all
neighbours of a8, so the gate is

    CNOT(3 -> a8), CZ(a8, 1), CZ(a8, 4), CNOT(3 -> a8)

using a6 and physical qubit 2 in frame 1. That is **4 CZ, 4 pulses and
240 ns**, against 8 CZ, 12 pulses and 380 ns.

`verify_hw.py gadget` checks:

- equality with CZ(2,3) on 60 random encoded logical states in each frame,
  with the ancilla returned to |0>;
- inequality off the code space (38 of 40 random stabilizer states);
- that the data unitary is exactly CZ(1,3)CZ(3,4);
- noiseless determinism of J's own builder with the gadget patched in at
  runtime, for both terminals, reset and no-reset, depths 0-22.

J's code-level ops are patched consistently, so detector back-propagation
uses the true tableau. A parity variant uses 5 CZ: copy z1 ⊕ z4 into a8, then
CZ(a8, 3), then uncopy.

| CZ(2,3) implementation (J's Garnet model) | CZ | check-free MLD f (exact) | check-free detection f (exact) | MLD every step f (2e4 shots) | LRB-D unif=1+final f |
|---|---|---|---|---|---|
| J routed (bridge on q0) | 8 | 0.90332 | 0.94960 | 0.7751 +- 0.0034 | 0.9863 +- 0.0008 |
| code-space 4-CZ (via a8) | 4 | **0.94615** | **0.97899** | **0.8030 +- 0.0029** | 0.9913 +- 0.0004 |
| code-space parity 5-CZ | 5 | 0.93586 | 0.97318 | 0.7985 +- 0.0029 | 0.9909 +- 0.0005 |
| native coupler (J's hypothetical) | 1 | 0.98489 | 0.99906 | (J: 0.8410) | (J: 0.99526) |

At T1 = T2 = 100 µs, check-free MLD is 0.9347 for the routed gadget and
0.9666 for the 4-CZ one; detection is 0.9717 and 0.9894.

**J's routed gadget breaks the circuit distance (VN).** Stim's
`search_for_undetectable_logical_errors` (the lead's distance test) finds
single faults that flip the logical with no detection event:

- a `Z7 X0` (or `Y7 X0`) fault on the second bridge CZ(a7, q0);
- X0 or Z0 one-qubit faults on the bridge qubit q0.

Each propagates to **Z0 Z2 = Z_L** after the uncopy. The circuit-level
distance of J's routed IQM circuits is **1** at every depth tested (m = 1, 2,
4, 6, with checks and check-free). The lead's SI1000 circuit has distance 2.
The first-order undetectable logical probability U1 (sum of DEM mechanisms
that flip L0 with no detector) is:

| variant | distance with checks / check-free | U1 per depth step, with checks | U1 per step, check-free | U1 at m = 22, check-free |
|---|---|---|---|---|
| J routed 8-CZ | 1 / 1 | 0.0015 | 0.0038 | 0.086 |
| code-space 4-CZ | 1 / 1 | 0.0004 | 0.0036 | 0.080 |
| parity 5-CZ | 1 (2 at m = 1) / 1 | 0.00013 | 0.0042 | 0.093 |
| native CZ(2,3) | **2 / 2** | **0** | **0** | **0** |

The check-free U1 values were computed with both J's builder and mine and
agree to 4 digits.

**Mechanism (VN, VA).** The code-space gadgets equal CZ(2,3) times a
Z3-controlled S_Z2. A single pre-existing X4 therefore becomes Y4 Z3 after
S_L†, whose Z part is Z3Z4 = Z_L. Under the check-free X-only readout the X
part is invisible. A noiseless reset of the mediating ancilla after every
gadget does not restore d = 2 (fresh-ancilla diagnostic in `distance.json`).
Only the native coupler is distance-preserving.

**Consequences for J's text.**

- On the routed patch, LRB-D unif=1+final has a first-order accepted error
  floor of about 0.0015 per step, roughly 22 % of its r = 0.0068.
- Check-free detection has about 0.0038 per step, roughly 15 % of its
  r = 0.025.
- J attributes the routed-vs-native gap to CZ count alone. Part of it is the
  lost distance.
- J's final recommendation (native (2,3) layout for check-free runs) is
  strengthened by this: it is the only option with d = 2.

## 4. Claim 6: one check round (VN, LIT)

- **12 CZ: confirmed.** There are 12 data-ancilla couplings and each needs one
  CZ, so 12 is minimal for one ancilla per check. Pulses: 2 + 9 + 7 = 18.
  Gate time: 3 x 20 + 6 x 40 = 300 ns.
- **1.8 µs: needs more evidence for Garnet.** The other 1.5 µs is made of:
  - an 800 ns readout pulse from Renger et al. (arXiv:2503.10903, Table S4,
    fetched). That is IQM's 6-qubit qubit-resonator (star) processor, not
    Garnet, and its 1q gates are 40 ns;
  - 200 ns of assumed integration and ring-down;
  - a 0.5 µs conditional-reset latency. IQM's `Reset_Conditional` docs
    (iqm-pulse 15.0.0, fetched) say only that the implementation "handles any
    necessary waits". No number is given, so J's UNVERIFIED label is correct.
- **Unmodelled parallel-CZ penalty.** The Garnet paper (fetched) measures the
  medians 9e-4 (1q) and 5e-3 (CZ) in *distance-two* parallel groups, and
  states "Benchmarking the two qubit gates in distance-one groups results in
  a reduction of mean fidelity by 0.6 for CZ gates and 3 percentage points
  for two-qubit Clifford gates". Most check-round CZ layers are distance-one
  pairs (e.g. CZ(1,a5) with CZ(2,a6)). The model uses 5e-3 throughout, so
  check-round numbers are optimistic (CONJ for the size).

## 5. Claim 8: the 4.7p idle mapping (VA, VN, LIT)

- **Formula.** My own Kraus construction (amplitude damping γ = 1 - e^{-t/T1}
  plus pure dephasing to coherence e^{-t/T2}) followed by a Pauli twirl
  (diagonal of the Pauli transfer matrix) reproduces p_X = p_Y = γ/4 and
  p_Z = (1 - e^{-t/T2})/2 - γ/4. The agreement is to 8e-17 at 7 (t, T1, T2)
  points, and J's `pauli_twirl_idle` agrees exactly. Analytically, the twirled
  channel has λ_x = λ_y = e^{-t/T2} and λ_z = e^{-t/T1}.
- **Arithmetic.** t = 1.0 µs, T1 = 40 µs and T2 = 21.05 µs give 2.94 % per data
  qubit. With p = 1.25 x 5e-3 = 6.25e-3 that is **4.70p**. The reset window
  gives 2.37p and readout plus reset gives 4.36 %, all matching J.
- **SI1000 reading.** Gidney et al. 2021 (arXiv:2108.10457, fetched),
  Table 2: CZ(p), AnyClifford1(p/10), InitZ(2p), MZ(5p), Idle(p/10),
  ResonatorIdle(2p). ResonatorIdle is "a 1-qubit depolarizing channel" per
  measurement time step, so comparing total Pauli probability with 2p is the
  right comparison.
- **Input provenance (needs more evidence).**
  - T1 = 40 µs is the whitepaper's "improve relaxation time T1 from the order
    of 40µs to above 100µs" (fetched). That is not a median.
  - T2e = 21.05 µs is Sharma (arXiv:2512.23009, Table 3, fetched, no snapshot
    date). Its own T1 median is **27.42 µs**.
  - The 1.0 µs window is borrowed from another device, as above.

  Sensitivity (`idle.json`):

  | case | total | in units of p |
  |---|---|---|
  | J main | 2.94 % | 4.70p |
  | 800 ns readout pulse only | 2.36 % | 3.78p |
  | consistent snapshot (T1 27.42, T2e 21.05) | 3.22 % | 5.14p |
  | no dynamical decoupling (T2 Ramsey 8.43) | 6.21 % | 9.94p |
  | fake-Garnet sample-profile medians (T1 37.6, T2 9.6) | 5.60 % | 8.96p |

  The qualitative conclusion (well above 2p) is robust.

## 6. Claim 9: f values (VN)

**Independent exact reproduction (check-free).** `verify_hw.py checkfree` uses
no `hardware_model.py` code:

- my own layer compiler and noise insertion, written from J's documented
  model (virtual SWAP, S, S†, Z and Pauli frame; H = one pulse layer; routed
  gadget; per-layer T1/T2 twirl on active qubits; ions get 5e-4 extra per 2q
  layer);
- **exact** logical error rates from the DEM, by Walsh-Hadamard over the 3
  outcome bits. No sampling; the enumeration matches 4e5 samples to 9e-4.

On the same 30 sequences with the same weighted fit:

| model | J const0_mld | mine | J const0 | mine | rejection at m = 22, J / mine |
|---|---|---|---|---|---|
| Garnet routed | 0.90329 +- 0.00237 | 0.90332 +- 0.00236 | 0.94969 +- 0.00206 | 0.94960 +- 0.00212 | 0.72 / 0.722 |
| Garnet native (2,3) | 0.98489 +- 0.00032 | 0.98489 +- 0.00032 | 0.99904 +- 0.00005 | 0.99906 +- 0.00005 | 0.31 / 0.307 |
| T1 = T2 = 100 µs | 0.93473 +- 0.00144 | 0.93474 +- 0.00146 | 0.97157 +- 0.00111 | 0.97165 +- 0.00113 | 0.67 / 0.674 |
| ion (Helios-class) | 0.99770 +- 0.00006 | 0.99771 +- 0.00006 | 0.999973 | 0.999972 | 0.056 / 0.056 |
| atom (Rb) | 0.99089 +- 0.00023 | 0.99092 +- 0.00023 | 0.99940 +- 0.00003 | 0.99940 +- 0.00003 | 0.26 / 0.263 |

**MLD every step.** I re-ran with J's builder and the lead's exact MLD (fresh
seed 99001, 30 sequences, 2e4 shots): **f = 0.7751 +- 0.0034**, against J's
0.7705 +- 0.0029.

- The per-depth F agree through m = 10: 0.8757 / 0.4935 / 0.2956 / 0.1744 /
  0.0619 against J's 0.8756 / 0.4916 / 0.2958 / 0.1742 / 0.0618.
- The difference comes from m ≥ 14, where F ≈ 0.004-0.02 sits at the |F| noise
  floor of about 1/√shots.
- LRB-D unif=1+final: 0.9863 against 0.9865.

**Compiled physical RB.** Analytically there are 5/3 H pulses per depth step,
so f = (1 - 4p1/3)^{5/3} = **0.997002** with p1 = 1.35e-3 (J: 0.99701).

**Caveats** (`fitcheck.json`, J's own data refitted):

- The fitted f depends on the depth window, because the decays are not
  exponential:

  | method | all depths | m ≤ 10 | m ≤ 6 |
  |---|---|---|---|
  | MLD every step | 0.7705 | 0.7637 | 0.7612 |
  | check-free detection | 0.9497 | 0.9669 | 0.9736 |

- The routed values include first-order undetectable faults (Section 3) and a
  sub-optimal 8-CZ routing.
- The check rounds ignore the distance-one CZ penalty (Section 4).
- The native-layout claim (detection 0.99904 above compiled physical
  0.99701) holds for every depth window.

## 7. Claim 10: "sparser checks are always better for correction" (VN, VA)

**Within J's tested regimes this holds.** I read J's `sweeps/*/summary.json`.
MLD f increases from k = 1 to k = 2 to k = 5 to check-free at every
coherence time (Garnet up to 1 ms), and on ion, atom, native and T100. For
example, at 1 ms: 0.911 / 0.925 / 0.937 / 0.945.

**As a general or structural statement it is false.** I ran J's builder with
the check rounds made noiseless and instantaneous (patched at runtime), so the
data see exactly the check-free noise plus syndrome information. Every-round
MLD then beats check-free MLD on the same run:

- routed Garnet: **0.9264 +- 0.0018** against 0.9034 +- 0.0024;
- native Garnet: **0.98936 +- 0.00025** against 0.98487 +- 0.00032.

This follows from the data-processing inequality (VA): more observations
cannot raise the Bayes error. The routed comparison has a small confound,
because resets also clean the gadget ancillas; the native one does not.

So "fewer checks is better" is a noise trade-off in J's parameter range, not a
consequence of the ambiguous-pair argument.

## 8. Claim 11: reset latency and no-reset (VN)

**Slope.** From J's latency sweep, f = 0.7986, 0.7789, 0.7635, 0.7302, 0.6752
and 0.6027 at τ_R = 0, 0.25, 0.5, 1, 2 and 4 µs. That is 0.068/µs over 0-1 µs,
0.055/µs over 1-2 µs and 0.036/µs over 2-4 µs. "About 0.07 per µs" holds only
for the first microsecond.

**No reset ≥ zero-latency reset.**

- A paired bootstrap over sequences of J's sweep gives f(no reset) −
  f(τ_R = 0) = **0.0097 +- 0.0019**.
- My re-run (10 sequences, m ≤ 14, 2e4 shots):
  - zero latency with J's e_reset = 3 %: 0.7992 +- 0.0073;
  - **zero latency with e_reset = 0: 0.8079 +- 0.0067**;
  - no reset: 0.8084 +- 0.0054.
- The advantage therefore comes entirely from J's *assumed* 3 % residual
  excitation after a conditional reset (e_reset = e_meas, "assumed" in
  `DEVICES`). With a clean reset the two tie.

## 9. Claim 13: offline Pauli-frame decoding (VA)

The claim is correct for this protocol. Every gate is Clifford, all
corrections and byproducts are Paulis, and the run ends with a Pauli-basis
readout, so a Pauli correction applied after the run is equivalent to one
applied in real time.

- I checked J's no-reset byproduct handling (classically controlled Z =
  frame update), and my 4-CZ and 5-CZ variants of it, for noiseless
  determinism.
- **Scope.** J's main IQM model still uses feed-forward in the conditional
  reset (measure + `cc_prx`). "No feed-forward" holds for the check-free and
  no-reset variants, as J's own table states.

## 10. Claim 14: early exit (LIT)

- **Helios.** Ransford et al., "Helios: A 98-qubit trapped-ion quantum
  computer", arXiv:2511.05465 (v1, 7 Nov 2025, 186 authors). Sec. II.4: "This
  runtime enables state-of-the-art user programming constructs ... early
  termination of programs based on mid-circuit measurement or arbitrary
  classical logic." Confirmed.
- **Guppy.** "Postselection: exit and panic" (Guppy v1.2.0 docs): "`exit`
  will end the current shot and carry on with subsequent ones", used for
  postselection. The example runs on the emulator, as J notes.
- **IBM.** "Execute dynamic circuits" (qiskit-ibm-runtime ~0.49): "IBM Quantum
  Compute Service currently only supports the conditional `if` statement."
  `for`, `while` and `switch` are unsupported, nested conditionals are not
  allowed, and reset or measurement inside conditionals is unsupported. The
  page says nothing about end, exit or early termination. IBM's OpenQASM 3
  feature table has no `end` row and marks for, while, break and continue
  unsupported.
- **IQM.** IQM client 36.0.0 user guide: "Only the `x`, `y`, `rx`, `ry` and
  `r` gates can be classically controlled", "conditioned on one classical
  bit", "an identity gate of corresponding duration is applied instead",
  `if_test` blocks cannot be nested, and reset is "a (projective) measurement
  followed by a classically controlled X gate". There is no termination
  primitive.

Absence from the documentation is not proof of absence. J's wording ("not
supported as documented") is appropriate.

## 11. Claim 15: DEM learning (VN, VA)

**Exact re-run.** `dem_learning.study_circuit` with J's seeds and arguments:

- `iqm_garnet` m = 2, sequence 1: all 19 decoders' error counts identical to
  J's JSON (oracle 26023, learned_1e5 26098, graph_1e6 37074, scaled_2 26254).
- `iqm_garnet_het0.5` m = 0, sequence 0: identical (oracle 4486, learned_1e5
  7983, nominal 7810).

**Independent estimator.** My own Eq. S11 p_ij at 1e6 shots gives a median
relative error of 0.9 % over 46 pairs with p > 1e-3 (J reports 0.8-1.3 %).

**JSON ranges** (from `per_depth`; all match):

- learned DEM with N ≥ 1e5 on homogeneous truths: max **+0.32 +- 0.11 %**
  (ion m = 2);
- pairs-only for m ≥ 1: **+24.1 % to +103.5 %**;
- p x2: max **+1.52 +- 0.13 %** (heterogeneous Garnet m = 2);
- p x10: up to +31.4 %.

**Split invisibility (VA).** Mechanisms with the same detector footprint
enter the syndrome distribution only through their XOR-combined probability.
Only the total is identifiable, so this is correct.

**Required qualifications.**

- **(a) Scope of ≤0.32 %.** It holds only for *homogeneous* truths. There the
  nominal model that supplies the candidate footprints and the observable
  split is the truth itself, so only statistical noise is tested. With
  heterogeneous truths, MLD on the learned DEM loses **+2.1 % to +77 %**
  (e.g. +77.0 % het Garnet m = 0, +12.3 % m = 2, +2.7 % het T100 m = 4) unless
  the true split is supplied.
- **(b) The pairs-only cost depends on the construction.** On Garnet m = 2,
  sequence 0 (2e5 test shots, oracle ε = 0.259):
  - J's pairs-only estimator: +24.6 %;
  - stim's graphlike decomposition of the true DEM: +28.4 %;
  - the true DEM with the footprints of size ≥ 3 simply dropped: only
    **+6.7 %**.

  Footprints of size ≥ 3 carry 18 % of the total mechanism weight.
- **(c) No unmodelled mechanisms.** Every candidate footprint comes from the
  nominal circuit structure, so unmodelled mechanisms (crosstalk, leakage)
  are never tested.

## 12. Other observations

- **Helios timing.** The Helios paper's depth-1 time is 55 ms per 98-qubit
  layer (Sec. II.3). J's ion wall-clock uses about 70 µs per 2q layer, a lower
  bound (already CONJ in J).
- **Helios crosstalk.** J uses only the global MCMR crosstalk (4.8e-5). The
  paper also gives a local 2.1e-4.
- **Garnet sample profile.** IQM's fake Garnet profile is labelled
  "sample-chip": T1 median 37.6 µs, T2 median 9.6 µs (type unstated), CZ
  depolarizing median 5.8e-3. It is not a calibration.

## Required fixes

1. **Claim 4 and Section 1 of J's notes.**
   - State that 8 CZ is the cost of J's literal CZ(2,3).
   - State that the logical S needs only 4 CZ on the patch (CZ(1,3)CZ(3,4)
     through the S_X2 ancilla, equal to CZ(2,3) on the code space), and report
     the gain.
   - Add a circuit-distance test (stim `search_for_undetectable_logical_errors`)
     to `tests/test_qubit512_hw.py`.
   - Report that every patch-routed variant, J's included, has circuit
     distance 1 (U1 table in Section 3), so routed LRB-D and check-free
     detection are not second order.
   - Re-attribute part of the routed-vs-native gap to the lost distance.
2. **Claims 6 and 8.**
   - Mark the 1.0 µs window (800 ns from IQM's star device + 200 ns assumed)
     and the 0.5 µs reset as non-Garnet or assumed.
   - Pair T1 and T2 from one source (5.1p), and give the 3.8p-9.9p range.
   - Note IQM's distance-one CZ penalty (−0.6 points), which the check rounds
     ignore.
3. **Claim 9.** State the fit-window sensitivity (MLD 0.761-0.771; check-free
   detection 0.950-0.974) and that m ≥ 14 MLD points sit at the |F| noise
   floor.
4. **Claim 10.** Scope it to the tested parameter range. Add the clean-check
   counterexample. Drop the implication that it follows from the
   ambiguous-pair argument.
5. **Claim 11.** Give the slope as 0.07/µs for 0-1 µs, decreasing to
   0.04/µs. State that no reset ≥ zero-latency reset only because e_reset is
   assumed to be 3 %, and that the two tie with a clean reset.
6. **Claim 13.** Note that the main IQM model's conditional reset is
   feed-forward.
7. **Claim 15.** Restrict "≤0.32 %" to homogeneous truths where the nominal
   model equals the truth; give 2-77 % for heterogeneous truths; call the
   24-104 % estimator-specific (an exact graph truncation costs 6.7 % on the
   same circuit).

## Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
V=studies/correction-decoders/qubit/scripts/verify_hw.py
python3 -m pytest -q tests/test_qubit512_hw.py
python3 $V graph --garnet-file <qiskit-on-iqm fake_garnet.py saved as text>
python3 $V idle; python3 $V checkfree; python3 $V gadget; python3 $V distance; python3 $V fitcheck
# on the shared allocation (srun --jobid=<id> --overlap --cpu-bind=none -c 256 ...):
python3 $V mld --num-seq 30 --shots 20000 --workers 16      # ~20 s
python3 $V reset --num-seq 10 --shots 20000 --workers 16
python3 $V dem
```

## Sources fetched in this session

- L. Abdurakhimov et al. (113 authors), "Technology and Performance Benchmarks
  of IQM's 20-Qubit Quantum Computer", arXiv:2408.12433 (v1, 22 Aug 2024).
  PDF read; quotes in Sections 4-5.
- IQM, qiskit-on-iqm `fake_garnet.py` (main branch, raw file, 2026-10-28).
- M. Renger et al. (65 authors, IQM), "A Superconducting Qubit-Resonator
  Quantum Processor with Effective All-to-All Connectivity", arXiv:2503.10903
  (v1 13 Mar 2025, v3 8 Dec 2025). Table S4: readout pulse 800 ns, 1q 40 ns;
  T1 41.3 µs, T2e 45.8 µs.
- R. Sharma, "Symmetry-Preserving Variational Quantum Simulation of the
  Heisenberg Spin Chain on Noisy Quantum Hardware", arXiv:2512.23009
  (28 Dec 2025). Table 3: T1 27.42, T2 Ramsey 8.43, T2 echo 21.05 µs (medians).
- C. Gidney, M. Newman, A. Fowler, M. Broughton, "A Fault-Tolerant Honeycomb
  Memory", arXiv:2108.10457. Tables 1-2.
- A. Ransford et al. (186 authors), "Helios: A 98-qubit trapped-ion quantum
  computer", arXiv:2511.05465 (7 Nov 2025). Abstract, Secs. II.3, II.4,
  III.2.4, III.2.5.
- Quantinuum, Guppy v1.2.0 docs, "Postselection: exit and panic".
- IBM Quantum docs, "Execute dynamic circuits" and "OpenQASM 3 feature table".
- IQM client 36.0.0 docs, "Qiskit on IQM" user guide; iqm-pulse 15.0.0,
  `Reset_Conditional`.
