# WP J: a concrete hardware protocol for qubit [[5,1,2]] LRB with correction

Superconducting (IQM-style square-lattice transmons) first, trapped ions and
neutral atoms second. Code: `studies/correction-decoders/qubit/scripts/hardware_model.py` (layout,
compilation, device model, hardware circuit builder, runner, fits) and
`studies/correction-decoders/qubit/scripts/dem_learning.py` (detector-error-model learning). Tests:
`tests/test_qubit512_hw.py` (26 tests, all passing). Results:
`studies/correction-decoders/qubit/results/J/`.

Labels: **[VA]** verified analytically, **[VN]** verified numerically,
**[LIT n]** from literature (reference n in Section 10, every one fetched in
this session), **[CONJ]** conjecture. Uncertainties are one standard error.
Decay fits are F(m) = b f^m on the mean over Clifford sequences, with
bootstrap errors over sequences (30 sequences, seed 20261028, the same as
`results/grid_main`). "Depth step" = one forward Clifford, its inverse, and
one check round where the protocol has one; r = (1 - f)/2 is the logical
error per depth step.

## 0. Summary

1. **Layout.** The 12 check couplings are exactly the 3x3 grid graph, so on a
   square lattice the code occupies a 3x3 patch, unique up to the 8 lattice
   symmetries [VN]. Data 2 and 3 land on opposite corners (distance 4), and
   the logical coupling (2,3) closes 5-cycles, so no bipartite lattice hosts it
   natively [VA, VN]. IQM Garnet contains 4 such patches, each with
   d(q2, q3) = 4 [VN, LIT 1, 2].
2. **Compilation.** SWAP(2,3) inside logical H is a relabelling (frame bit),
   so H_L is one layer of 5 pi/2 pulses; S, S-dagger, Z are virtual; logical X
   and Z are Pauli-frame updates. CZ(2,3) costs 8 CZ, routed through ancillas
   a7 and a5 (in |0> between rounds) and a bridge on data 0 [VN]. One check
   round costs 12 CZ, 18 pulses and one multiplexed readout: 1.8 us at
   IQM-class timing. A random logical Clifford costs 5.33 CZ, 12.2 pulses and
   270 ns on average [VN].
3. **IQM Garnet-class model (published medians).** The data idle during the
   1.0 us readout plus 0.5 us reset window is 4.4 % Pauli error per data qubit
   per round. Mapped onto SI1000 at p = 6.25e-3, the measurement-idle term is
   4.7p instead of 2p [VA]. Checks every round are then a net loss. Exact MLD
   with checks every round gives f = 0.7705 +- 0.0029; the check-free circuit
   decoded by MLD gives 0.9033 +- 0.0024; check-free detection gives
   0.9497 +- 0.0021 with rejection rising from 22 % (m = 2) to 72 % (m = 22)
   [VN]. Matched physical RB gives 0.99701 +- 0.00007 (compiled pulses),
   0.99583 +- 0.00008 (p_RB = p) and 0.9793 +- 0.0004 (p_RB = 5p) [VN].
4. **The routed CZ(2,3) dominates the check-free protocol, and the
   check-free protocol does not need the patch.** It uses no ancillas, and
   its couplings (encoder edges plus (2,3)) form a bipartite graph. A 2x2
   plaquette plus one site of Garnet therefore makes (2,3) a native coupler
   [VA]. Check-free detection then reaches f = 0.99904 +- 0.00005, above all
   three physical baselines, at 31 % rejection by m = 22; check-free MLD
   reaches 0.98489 +- 0.00032 [VN, run `hw_iqm_native23`].
5. **Ions (Helios-class).** Check-free detection f = 0.999973 +- 0.000002 at 5.6 %
   rejection (m = 22), above compiled physical RB (0.99992). MLD with checks
   every round gives 0.99061 +- 0.00009 [VN].
6. **Deployment.** Clifford-only LRB needs neither feed-forward nor real-time
   decoding: offline Pauli-frame decoding of the stored records is exact [VA,
   LIT 17]. Mid-circuit measurement is needed only by the check-round
   protocols. Reset is optional: no-reset XOR bookkeeping is better on IQM
   (MLD f = 0.809 vs 0.771) [VN]. Real-time abort of LRB-D is supported on
   Quantinuum Helios [LIT 21, 25]. It is not documented for IBM (only `if`)
   [LIT 27] or IQM (single-bit conditional rotations) [LIT 6].
7. **DEM learning.** A hyperedge generalisation of the p_ij estimator
   [LIT 12, 15] recovers the DEM.
   - Exact MLD on the learned DEM loses at most 0.32 +- 0.11 % (relative)
     logical error against the true DEM with >= 1e5 training shots, across
     Garnet-class, T1 = T2 = 100 us and ion truths [VN].
   - A pairs-only p_ij DEM loses 24-104 % [VN].
   - A decoder prior with p wrong by x2 costs <= 1.5 %; x10 costs up to
     31 % [VN].
   - The d = 2 tie-break (which qubit of an ambiguous pair failed) is
     invisible in syndromes [VA, LIT 14]. With heterogeneous qubits a
     homogeneous split prior costs 2-74 %, and learning the rates does not
     help. Learned rates with the per-qubit split recover oracle performance
     (<= 0.11 %) [VN].

**Headline.** On IQM-class superconducting hardware, checks every round make
d = 2 correction worse than not checking. The idle error during readout and
reset is about 2.4x the SI1000 2p term, and the routed fold coupling costs
8 CZ. In the coherence sweep (T1 = T2 up to 1 ms), fewer checks were always
better for correction, and the same ordering held for ions and atoms.

The sustainable protocol is check-free mirror LRB with a transversal final
X readout of all five data qubits, decoded offline twice on the same shots
(detection and MLD). The (2,3) pair sits on a native coupler, which a 2x2
plaquette of the existing lattice provides. On Garnet-class gates this
check-free detection beats the physical qubit (f = 0.99904 against 0.99701).
Check rounds are best used as separate DEM-calibration runs, or for
detection at shallow depth.

## 1. Layout on a square lattice (IQM crystal topology)

IQM "qubit crystals" place qubits on a square lattice (drawn rotated by
45 degrees), connected by tunable couplers. Native operations are PRX
rotations, virtual RZ and CZ [LIT 1; LIT 5]. Garnet has 20 qubits and 30
couplers [LIT 1, 2].

```
 code patch (frame 0)                  IQM Garnet lattice, one of 4 patches
                                       (row, column) from the coupler list
  q0 ---- a7 ---- q2                         QB1  QB2
  |       |       |                    QB3  [QB4=q0] [QB5=a5] [QB6=q3]  QB7
  a5 ---- q1 ---- a6                   QB8  [QB9=a7] [QB10=q1][QB11=a8] QB12
  |       |       |                    QB13 [QB14=q2][QB15=a6][QB16=q4] QB17
  q3 ---- a8 ---- q4                         QB18 QB19 QB20

  a5: Z0 Z1 Z3   a6: Z1 Z2 Z4   a7: X0 X1 X2   a8: X1 X3 X4
  fold axis q0, q1, q4; logical coupling q2..q3 at Manhattan distance 4
```

- The Tanner graph (data-ancilla couplings) is the 3x3 grid. Its embeddings
  in a 3x3 grid number 8 (the grid's symmetries), and in a 5x5 grid 72
  (9 positions x 8), so the patch is unique up to symmetry
  [VN: `test_tanner_graph_embeds_uniquely_as_3x3_patch`].
- Adding the (2,3) edge creates a cycle of length 5 (2-3-a5-1-a6-2). A
  square lattice is bipartite, so (2,3) cannot be a native coupler in any
  embedding [VA; VN: `test_logical_coupling_is_not_native`].
- Garnet: the 30 couplers of IQM's own Garnet backend description [LIT 2]
  form a 5x5 square grid with 5 sites missing near the corners. It contains 4 3x3
  patches, centred on QB10, QB9, QB11 and QB15, each with
  d(q2, q3) = 4 on the full chip, so the extra qubits give no shortcut
  [VN: `test_garnet_coupling_map_and_patches`, `results/J/layout.json`].
- Frames: after an odd number of logical H, code qubits 2 and 3 have swapped
  physical sites. Each ancilla keeps its physical neighbours but changes type:
  a7 and a8 measure the Z checks, a5 and a6 the X checks
  [VN: `test_frame_ancilla_roles_match_patch_neighbours`, and noiseless
  circuits give zero detection events in both frames].
- The check-free (const0) protocol needs only the encoder couplings
  {0-1, 1-2, 2-4, 0-3} plus (2,3). That graph is bipartite (4-cycle 0-1-2-3
  plus a pendant 4), so a 2x2 plaquette plus one site hosts it with **no**
  routing [VA]. The patch layout is needed only for check rounds.

### SWAP overheads for the logical coupling

| logical op | implementation | CZ count | notes |
|---|---|---|---|
| SWAP(2,3) in H_L | relabelling (frame bit) | 0 | [VN] determinism in both frames |
| SWAP(2,3) in H_L, physical | 3 CNOT from routed CZ | 24 | avoided; +20 CZ per Clifford on average |
| CZ(2,3) in S_L | copy z2 -> a7, z3 -> a5; CZ(a7, a5) via bridge q0; uncopy | 8 | 2 + 4 + 2; data q0 touched by 4 gates |
| CZ(2,3), native coupler | 1 CZ | 1 | ions, atoms, or a hypothetical long coupler |

The 8-CZ construction uses only patch couplers and returns ancillas to |0>
[VN: `test_routed_cz23_equals_cz`, 25 random 5-qubit stabilizer input states,
both bridges q0 and q4]. A light-cone argument gives at least 4 CZ for a
distance-4 interaction [VA]. Whether 8 is optimal is [CONJ]. SWAP-based
routing costs at least 6 SWAPs = 18 CNOT + 1 CZ [VA].

## 2. Compilation, gate counts and timing

CNOT = H(t) CZ(c, t) H(t) [VN: `test_cnot_from_cz`]. Z rotations (S,
S-dagger, Z) are virtual: zero-duration frame changes [LIT 11], also native
on IQM [LIT 1]. One check round (`check_round_layers`):

- H on the Z ancillas;
- 3 CZ layers for the Z checks (data 1 in different layers for the two
  checks);
- H on all data, on the Z ancillas (closing) and on the X ancillas (opening);
- 3 CZ layers for the X checks;
- H on all data and on the X ancillas;
- one simultaneous readout of all four ancillas;
- conditional reset (forward rounds only).

IQM Garnet's readout is frequency-multiplexed, with per-qubit Purcell filters
on 3 probe lines and a readout module for "simultaneous readout of up to ten
qubits" [LIT 1].

| item (IQM-class timing) | CZ | 1q pulses | layers | duration |
|---|---|---|---|---|
| check round (forward) | 12 | 18 | 3 x 1q, 6 x CZ, M, R | 1.80 us (0.30 gates + 1.0 readout + 0.5 reset) |
| terminal check round | 12 | 18 | no reset | 1.30 us |
| H_L | 0 | 5 | 1 | 20 ns |
| S_L (routed CZ(2,3)) | 8 | 12 (+3 virtual) | 7 x 1q, 6 x CZ | 380 ns |
| X_L, Z_L | 0 | 0 | Pauli frame | 0 |
| random logical Clifford, routed (mean of 24) | 5.33 | 12.17 | | 270 ns (max 420) |
| random logical Clifford, native (2,3) | 0.67 | 4.17 | | 43 ns |
| random logical Clifford, physical SWAP, routed | 25.33 | 45.5 | | 1220 ns |
| depth step (Clifford + inverse + round), routed | 22.7 | 42.3 | | 2.34 us |
| lead's gate model (reference) | 1.5 per Clifford, 12 per round | | | |

Source: `results/J/gate_counts.json` [VN by construction and
`test_gate_counts`]. Durations use t_1q = 20 ns, t_CZ = 40 ns (IQM's Garnet
backend description [LIT 2]; the whitepaper gives 20-40 ns for CZ [LIT 1]),
readout window 1.0 us (800 ns pulse [LIT 3] plus 200 ns assumed integration
and ring-down [CONJ]) and conditional-reset latency 0.5 us
(**UNVERIFIED**; IQM does not publish it [LIT 6]; swept in Section 4.3).

Whole-circuit wall clock, mean +- std over the 30 sequences
(`results/J/timing.json`):

| platform | m = 2 | m = 10 | m = 22 | check-free (const0) at m = 22 |
|---|---|---|---|---|
| IQM-class, routed (2,3) | 6.2 +- 0.4 us | 24.8 +- 0.9 us | 52.7 +- 1.5 us | 11.8 us |
| IQM-class, native (2,3) | 5.1 us | 20.2 us | 42.8 us | 1.9 us |
| ion (Helios-class) | 1.9 ms | 7.2 ms | 15.1 ms | 2.4 ms |
| atom (Rb, zoned) | 5.8 ms | 21.5 ms | 45.1 ms | 6.0 ms |

The m = 22 circuit with checks lasts longer than Garnet's T1 (about 40 us
[LIT 1]). Ion and atom durations assume 10 us (ion) and 1 us (atom)
single-qubit gates and a 100 us ion readout [CONJ]. The 70 us ion two-qubit
gate [LIT 21], the 200 us atom moves and 500 us imaging [LIT 23] are
published; ion and atom noise does not depend on these durations in the
model (Section 3).

## 3. Device model and SI1000 mapping

Conversions [VA; VN: `test_pauli_twirl_matches_kraus`,
`test_depolarizing_from_average_infidelity`]:

- An average gate infidelity r maps to depolarizing strength
  p = r (d+1)/d: 3r/2 for one qubit, 5r/4 for two [LIT 10].
- Idling for time t is the Pauli twirl of amplitude plus phase damping:
  p_X = p_Y = (1 - e^{-t/T1})/4 and
  p_Z = (1 - e^{-t/T2})/2 - (1 - e^{-t/T1})/4 [LIT 9, Eqs. 9-10]. The test
  checks this against the chi matrix of the Kraus channel.
- Every layer applies its gate error to acted qubits and the twirled idle
  channel for the layer duration to every other active qubit. Ancillas in a
  known |0> are inactive.

Parameter sets (`DEVICES` in `hardware_model.py`):

| parameter | IQM Garnet-class (main) | source |
|---|---|---|
| topology | square lattice, tunable couplers, 30 couplers | [LIT 1, 2, 5] |
| native gates | PRX, virtual RZ, CZ; measure; reset = measure + classically controlled X | [LIT 1, 2, 6] |
| 1q error | 9e-4 (median, RB per native gate) -> p1 = 1.35e-3 | [LIT 1] (Braket: 99.92 % [LIT 5]) |
| CZ error | 5e-3 (median, interleaved RB) -> p2 = 6.25e-3 | [LIT 1] (Braket: 99.51 % [LIT 5]) |
| readout error | 3e-2 (median, simultaneous readout) | [LIT 1] |
| T1 | 40 us ("of the order of 40 us"; target > 100 us) | [LIT 1] |
| T2 (echo, used with DD) | 21.05 us (median, Garnet calibration snapshot) | [LIT 4, Table 3] |
| T2* (no DD, variant) | 8.43 us | [LIT 4] |
| t_1q, t_CZ | 20 ns, 40 ns | [LIT 2]; CZ 20-40 ns [LIT 1] |
| readout window | 1.0 us = 800 ns pulse + 200 ns | [LIT 3, Table S4] + [CONJ] |
| reset | conditional (measure + cc_prx); 0.5 us; residual = readout error | latency **UNVERIFIED** [LIT 6, 7] |

Variants: `iqm_garnet_snapshot` (all medians from [LIT 4]: T1 27.42 us,
PRX 99.86 %, CZ 99.40 %, readout 97.29 %), `iqm_garnet_ramsey` (T2* idles,
no DD), `iqm_t100` (T1 = T2 = 100 us, IQM's T1 target [LIT 1] with T2 = T1
assumed), `iqm_garnet_native23` (native CZ(2,3)).

- **Trapped ions** (`ion_helios`, Quantinuum Helios [LIT 21]; the older H2
  reports 1q 2.5e-5, 2q 1.84e-3 and SPAM 1.6e-3 [LIT 22]):
  - errors: 1q 2.5e-5, 2q 7.9e-4, SPAM 4.8e-4;
  - memory: linear memory error 5e-4 per depth-1 layer, applied per CZ
    layer to idle qubits;
  - measurement crosstalk: global MCMR crosstalk 4.8e-5 per measured ion
    (x4 per readout layer);
  - connectivity: all-to-all, so CZ(2,3) is native.
- **Neutral atoms** (`atom_rb`):
  - errors: CZ 99.5 % [LIT 24], local 1q 99.912 %, readout about 99.8 %
    [LIT 23];
  - readout and coherence: imaging costs data at most about 0.1 % per
    500 us, T2 effectively 1-2 s, moves about 200 us per gate layer
    [LIT 23];
  - connectivity: CZ(2,3) is native after a move.

**SI1000 mapping** (`si1000_equivalent`; `results/J/mapping.json`). SI1000
[LIT 8, Table 2] is CZ (p), AnyClifford1 (p/10), InitZ (2p), MZ (5p),
Idle (p/10) and ResonatorIdle (2p). It was chosen to represent a chip "where
measurement and reset take an order of magnitude longer than other gates".
Every term is expressed as a multiple of p = the CZ depolarizing strength:

| SI1000 term (repo value) | physical origin | IQM Garnet-class | IQM T1=T2=100us | ion | atom |
|---|---|---|---|---|---|
| p (CZ, DEPOLARIZE2) | 5/4 x CZ infidelity | 6.25e-3 | 6.25e-3 | 9.9e-4 | 6.25e-3 |
| one_qubit (0.1 p) | 3/2 x pulse infidelity | 0.216 p | 0.216 p | 0.038 p | 0.211 p |
| **meas_idle (2 p, ResonatorIdle)** | **twirled T1/T2 decay of each data qubit during one readout window (plus measurement crosstalk)** | **4.70 p** (2.94 %) | 1.19 p | 0.70 p | 0.19 p |
| meas_flip (5 p) | readout assignment error | 4.8 p | 4.8 p | 0.49 p | 0.32 p |
| meas_depol (p, repo only) | none: dephasing of the measured ancilla is irrelevant; its T1 decay is in the assignment error | 0 | 0 | 0 | 0 |
| reset_idle (2 p) | decay during the reset or feedback window | 2.37 p | 0.60 p | 0 | 0 |
| reset_flip (2 p) | residual excitation after conditional reset | 4.8 p | 4.8 p | 0 | 0.32 p |
| Idle (0.1 p in SI1000, omitted by the repo) | decay during one CZ layer | 0.19 p | 0.05 p | 0.51 p | 0.01 p |

The 2p measurement-idle term is the decoherence of the data qubits during
the ancilla readout window:
p_idle(tau_M) ~ tau_M (1/(4 T1) + 1/(2 T2)) for tau_M << T1, T2 [VA].
On IQM-class parameters it is 2.4x SI1000's value, and the repo's four
sequential readouts per round apply it four times.

## 4. Simulation results (LRB-D, exact MLD, gap postselection)

The circuits come from `build_hw_lrb_circuit`. They follow the repo protocol:

- ideal encoding;
- each forward Clifford followed by a check round;
- mirror inverses without checks;
- a terminal check;
- an ideal transversal X readout of all five data.

They are compiled and noised as in Sections 2-3 and returned as `LRBCircuit`
objects. The lead's `TransferMatrixMLD` and LRB-D policies run on them
unchanged. Exact MLD equals brute-force inference on the hardware DEM
[VN: `test_hw_dem_works_with_exact_mld_and_brute_force`].

Methods:

- `mld5`: exact MLD on all detectors;
- `const0` / `const0_mld`: the check-free circuit, postselected on the final
  X checks or decoded;
- `lrbd_*`: LRB-D policies; `_final` adds the two final X detectors.

Each run uses 30 sequences x 9 depths (0-22) x 1e5 shots, with all shots
decoded unless noted.

### 4.1 IQM Garnet-class (main run, `results/J/hw_iqm_garnet`)

| method | f | b | r per step | F(2) | F(6) | F(22) | rejection m = 2 / 6 / 22 |
|---|---|---|---|---|---|---|---|
| MLD, checks every round | 0.7705 +- 0.0029 | 0.8756 | 0.115 | 0.492 | 0.174 | 0.004 | 0 |
| MLD, data 0/3 readout only (`mld2`) | 0.7606 +- 0.0031 | 0.8691 | 0.120 | 0.458 | 0.161 | 0.003 | 0 |
| check-free + MLD (`const0_mld`) | 0.9033 +- 0.0024 | 1.012 | 0.048 | 0.817 | 0.558 | 0.106 | 0 |
| check-free detection (`const0`) | 0.9497 +- 0.0021 | 1.091 | 0.025 | 0.967 | 0.863 | 0.254 | 0.22 / 0.44 / 0.72 |
| LRB-D unif=1 + final | 0.9865 +- 0.0006 | 0.997 | 0.0068 | 0.968 | 0.920 | none accepted | 0.843 / 0.991 / 1 (0 of 3e6 accepted) |
| LRB-D const=1 + final | 0.897 +- 0.006 | 0.997 | 0.052 | 0.924 | 0.493 | 0.008 | 0.76 / 0.93 / 0.96 |
| physical RB, compiled pulses | 0.99701 +- 0.00007 | | 0.0015 | | | | |
| physical RB, p_RB = p | 0.99583 +- 0.00008 | | 0.0021 | | | | |
| physical RB, p_RB = 5p | 0.9793 +- 0.0004 | | 0.010 | | | | |

The F standard errors over sequences are <= 0.004 for MLD and 0.004-0.018
for `const0` (full per-depth tables in `summary.json`). The fits for
postselected methods use depths with >= 1000 accepted shots. `const0` decays
non-exponentially (b = 1.09), so its f is a summary only.

Readings [VN]:

- Correction with checks every round loses 11.5 % per depth step, against
  0.15-1.0 % for physical RB.
- The check-free MLD loses 4.8 % per step.
- LRB-D unif=1+final has a small accepted decay but keeps 16 % of shots at
  m = 2, 0.9 % at m = 6 and almost none from m = 14.
- The `const=1` policy decays slowly, then fast (0.997, 0.924, 0.743,
  0.493, 0.133 at m = 0-10). This is the same non-exponential shape as the
  qutrit production run (CONTEXT.md).

### 4.2 Variants (same sequences and shots)

| variant | MLD f | const0_mld f | const0 f (rej. m = 22) | LRB-D unif=1+final f | gain over main (MLD) |
|---|---|---|---|---|---|
| main (reset 0.5 us, routed) | 0.7705 +- 0.0029 | 0.9033 +- 0.0024 | 0.9497 (0.72) | 0.9865 +- 0.0006 | |
| no reset (XOR bookkeeping) | 0.8086 +- 0.0025 | same | same | 0.9898 +- 0.0005 | +0.038 |
| native CZ(2,3) | 0.8410 +- 0.0017 | 0.98489 +- 0.00032 | 0.99904 +- 0.00005 (0.31) | 0.99526 +- 0.00016 | +0.071 |
| T1 = T2 = 100 us | 0.8792 +- 0.0017 | 0.9347 +- 0.0014 | 0.9716 +- 0.0011 (0.67) | 0.9948 +- 0.0003 | +0.109 |
| lead's circuit (`build_lrb_circuit`), IQM-mapped SI1000 factors (p = 6.25e-3) | 0.709 +- 0.006 | 0.9804 +- 0.0004 | 0.9977 +- 0.0001 (0.44) | 0.9735 +- 0.0009 | -0.061 |

The last row runs the lead's own builder with the Section 3 scale factors
(`--kind repo`; 5e4 decoded shots). It reads out the four ancillas
sequentially, so the 4.7p idle term is applied four times per round, and
LRB-D rejects 42 % already at m = 0. Its gate model has a native (2,3) and
no gate-layer idling, so its check-free numbers are close to the
native-(2,3) row. Running the repo protocol unchanged on IQM-class noise
would therefore be pessimistic for the check rounds by 0.06 in f [VN].

No-reset mode uses the XOR of consecutive outcomes of the same physical
ancilla as the syndrome. The routed CZ(2,3) then picks up known Z byproducts
from unreset ancillas, and these are applied as Pauli-frame updates
(`feedback` layer) [VN: deterministic noiseless circuits].

### 4.3 Sweeps: when do checks start to pay? (10 sequences, m <= 14, 5e4 shots)

**Coherence x check interval** (`results/J/sweeps/coherence/summary.json`).
IQM Garnet-class gates and readout with routed (2,3) and T1 = T2 = T. A
check round is executed after every k-th forward Clifford (the rounds
`unif=k` uses), and the terminal round is always executed. MLD uses every
executed round. LRB-D (`lrbd_all_final`) postselects on all executed rounds
and the final X checks.

| T1 = T2 | MLD, k = 1 | MLD, k = 2 | MLD, k = 5 | check-free MLD | check-free detection (acc. m = 14) | LRB-D k = 5 + final (acc. m = 14) |
|---|---|---|---|---|---|---|
| Garnet (40 / 21 us) | 0.764 +- 0.009 | 0.827 +- 0.007 | 0.878 +- 0.005 | 0.905 +- 0.005 | 0.960 +- 0.003 (0.34) | 0.9886 +- 0.0008 (0.029) |
| 50 us | 0.841 +- 0.005 | 0.878 +- 0.005 | 0.907 +- 0.004 | 0.924 +- 0.004 | 0.971 +- 0.002 (0.39) | 0.9924 +- 0.0006 (0.056) |
| 100 us | 0.877 +- 0.005 | 0.902 +- 0.005 | 0.923 +- 0.004 | 0.935 +- 0.004 | 0.977 +- 0.002 (0.43) | 0.9936 +- 0.0005 (0.082) |
| 200 us | 0.896 +- 0.004 | 0.915 +- 0.004 | 0.931 +- 0.003 | 0.941 +- 0.003 | 0.979 +- 0.001 (0.45) | 0.9941 +- 0.0005 (0.099) |
| 500 us | 0.908 +- 0.004 | 0.922 +- 0.004 | 0.936 +- 0.003 | 0.944 +- 0.003 | 0.980 +- 0.001 (0.46) | 0.9945 +- 0.0004 (0.11) |
| 1 ms | 0.911 +- 0.004 | 0.925 +- 0.004 | 0.937 +- 0.003 | 0.945 +- 0.003 | 0.981 +- 0.001 (0.46) | 0.9948 +- 0.0004 (0.11) |

Readings [VN]:

- At every coherence time, correction gets better with fewer checks
  (k = 1 < 2 < 5 < none).
- As T grows, MLD saturates near f = 0.91-0.94. The remaining error is the
  checks' own CZ and readout faults plus the routed CZ(2,3), not idling.
- So within the tested range, check rounds never pay for d = 2 correction
  with a final transversal readout. This is consistent with the analytic
  point that no syndrome history separates the members of an ambiguous
  pair [VA]. Any gain from checks would have to come from correlations
  such as Y-error signatures, and here it is smaller than the checks' own
  noise [CONJ for the mechanism].
- Checks do pay for *detection*: LRB-D with a check every 5th Clifford
  reaches f = 0.989-0.995, against 0.960-0.981 for check-free detection,
  but keeps only 3-11 % of shots at m = 14, against 34-46 %.

**Reset / feedback latency** (`sweeps/latency`; Garnet-class, checks every
round). IQM's reset is a measurement plus a classically controlled pi pulse
[LIT 6], and its feedback latency is not published. The table sweeps the
wait tau_R spent on the data per round.

| tau_R | 0 | 250 ns | 500 ns (main) | 1 us | 2 us | 4 us | no reset (XOR) |
|---|---|---|---|---|---|---|---|
| MLD f | 0.799 +- 0.008 | 0.779 +- 0.007 | 0.764 +- 0.009 | 0.730 +- 0.008 | 0.675 +- 0.011 | 0.603 +- 0.015 | 0.808 +- 0.007 |
| LRB-D unif=1+final: acceptance at m = 2 | 0.179 | 0.168 | 0.157 | 0.137 | 0.104 | 0.062 | 0.229 |
| LRB-D: F(2) | 0.976 | 0.972 | 0.969 | 0.959 | 0.940 | 0.887 | 0.977 |

Readings [VN]:

- Each microsecond of feedback wait costs about 0.07 in f [VN]. This is
  consistent in size with 3 %/us per data qubit (Section 5 [VA]): five data
  qubits pick up about 15 % per us, and if about half ends as a logical
  error, r rises by 0.035/us and f falls by 0.07/us [CONJ for the factor
  1/2].
- A 10 % relative increase in the logical error per step,
  r = (1 - f)/2, is reached at about 250 ns. That is the latency budget if
  a conditional reset is used.
- No reset (XOR bookkeeping) is at least as good as a zero-latency reset,
  consistent with Gehér et al.'s conclusion for slow resets [LIT 20].
  IQM's own [[4,2,2]] error-detection experiment did not reset its
  ancillas between cycles [LIT 7].

**Readout window** (`sweeps/readout`; checks every round, 0.5 us reset):

| tau_M | 250 ns | 500 ns | 1 us (main) | 2 us |
|---|---|---|---|---|
| MLD f | 0.816 +- 0.006 | 0.797 +- 0.007 | 0.764 +- 0.009 | 0.707 +- 0.010 |
| LRB-D unif=1 f | 0.9859 +- 0.0013 | 0.9844 +- 0.0014 | 0.9806 +- 0.0023 | 0.9709 +- 0.0029 |

Even a 250 ns readout leaves MLD with checks (f = 0.816) far below
check-free MLD (0.905) [VN]. The check-free numbers do not depend on
tau_M or tau_R.

**Sparse checks on the other platforms** (`sweeps/platforms_sparse`; the
k = 1 column is from the main 30-sequence runs):

| platform | MLD k = 1 | MLD k = 2 | MLD k = 5 | check-free MLD | LRB-D k = 5 + final (acc. m = 14) | check-free detection (acc. m = 14) |
|---|---|---|---|---|---|---|
| ion | 0.99061 +- 0.00009 | 0.9942 +- 0.0002 | 0.9972 +- 0.0001 | 0.9976 +- 0.0001 | 1.0000 (0.88) | 1.0000 (0.96) |
| atom | 0.9628 +- 0.0004 | 0.9749 +- 0.0010 | 0.9851 +- 0.0008 | 0.9912 +- 0.0009 | 0.9998 (0.55) | 0.9996 +- 0.0001 (0.82) |
| IQM, native (2,3) | 0.8410 +- 0.0017 | 0.907 +- 0.004 | 0.977 +- 0.003 | 0.985 +- 0.001 | 0.9994 +- 0.0001 (0.16) | 0.9994 +- 0.0001 (0.79) |
| IQM, T1 = T2 = 100 us | 0.8792 +- 0.0017 | 0.902 +- 0.005 | 0.923 +- 0.004 | 0.935 +- 0.004 | 0.9936 +- 0.0005 (0.08) | 0.977 +- 0.002 (0.43) |

On every platform correction improves monotonically as checks get sparser,
and the check-free circuit is best (ion k = 5 equals it within 2 sigma)
[VN]. For detection, sparse checks buy accepted fidelity only where
check-free detection is still far from 1 (IQM T100). On ions, atoms and
native-(2,3) IQM, check-free detection comes within 2e-4 of the k = 5
fidelity at 1.1-4.8x the acceptance.

### 4.4 Ions and atoms

| platform | MLD f (checks every round) | const0_mld f | const0 f (rej. m = 22) | LRB-D unif=1+final f (rej. m = 22) | physical RB: compiled / p / 5p |
|---|---|---|---|---|---|
| ion (Helios-class) | 0.99061 +- 0.00009 | 0.99770 +- 0.00006 | 0.999973 +- 0.000002 (0.056) | 0.999983 +- 0.000001 (0.52) | 0.99992 / 0.99934 / 0.99670 |
| atom (Rb, zoned) | 0.96282 +- 0.00036 | 0.99089 +- 0.00023 | 0.99940 +- 0.00003 (0.26) | 0.99978 +- 0.00001 (0.95) | 0.99708 / 0.99583 / 0.97927 |

On ions, checks are cheap: rejection is 2.3 % at m = 0. LRB-D unif=1 is
therefore viable to m = 22 (48 % acceptance), and correction is first order
(r = 4.7e-3 per step, 110x physical).

The atom model has IQM-like gate errors (p = 6.25e-3) but almost no idle
error during its 500 us readout (0.19p against 4.7p). Its MLD with checks
(f = 0.963) is far better than IQM's (0.771) but still below physical. LRB-D
unif=1 keeps 70 % of shots at m = 2 and 5 % at m = 22.

On both platforms the detection-based protocols (check-free `const0`, and
LRB-D) beat compiled physical RB; correction does not [VN].

### 4.5 Gap postselection (MLD soft output)

Accepting the shots with |LLR| above a threshold, at LRB-D's own acceptance,
reproduces LRB-D's fidelity (`gap` entries in `summary.json`) [VN]:

- IQM main, m = 2: MLD gap F = 0.966 at 17.6 % acceptance, against
  unif=1+final F = 0.968 +- 0.002 at 15.8 %.
- IQM main, m = 6: 0.922 at 1.2 %, against 0.920 +- 0.004 at 0.9 %.
- Check-free: `const0_mld` with a gap threshold equals `const0` detection at
  matched acceptance at every depth (identical F to 3 decimals), because the
  final syndrome is the only information.

At d = 2 the soft output adds a continuous acceptance knob but no fidelity
beyond detection at equal acceptance. On ions it gives 0.9996 at 48.5 %
acceptance at m = 22, matching unif=1+final.

## 5. Decoder deployment

- **Offline Pauli-frame decoding is exact for this protocol** [VA]. Every
  gate in the circuit is Clifford: logical words, inverses and checks. The
  run ends with an X-basis readout. Any Pauli correction commutes through the
  remaining Cliffords to a classical flip of the recorded parity x0 + x3, so
  a decoder applied after the run equals one applied in real time, minus the
  noise of physical corrections. This is Knill's Pauli frame [LIT 17]. One
  stored dataset (all 4(m+1) ancilla bits plus 5 data bits per shot) serves
  every LRB-D policy, MLD, gap postselection and DEM learning.
- **What needs what.**

  | protocol | mid-circuit measurement | reset | feed-forward | real-time decoding |
  |---|---|---|---|---|
  | check-free (const0, recommended on IQM-class) | no | no | no | no |
  | LRB-D / MLD with checks | yes, 4 per round | conditional reset (IQM: measure + `cc_prx`) or none (XOR bookkeeping) | only the conditional pi pulse of the reset | no |
  | real-time abort of LRB-D | yes | as above | shot termination | a 4-bit "syndrome = 0?" test per round |
  | non-Clifford extension (outside this study) | yes | yes | yes, byproduct before the next non-Clifford gate | yes [LIT 18] |

- **Latency budget for d = 2 lookup tables.**
  - The exact decoder is a 32-state transfer matrix per round (one 32 x 32
    matrix-vector product). A per-round lookup table has 2^4 entries.
  - In numpy this costs 3.2 s per 1e5 shots at depth 22 on one loaded
    login-node core, about 1.3 us per shot per round [VN: timing of
    `TransferMatrixMLD.log_likelihood_ratio` on the `iqm_garnet` depth-22
    circuit in this session].
  - FPGA decoders already reach < 1 us per round and a 9.6 us response time
    on transmons [LIT 19]; the review [LIT 18] cites about 1 us per round as
    the transmon requirement.
  - For Clifford LRB the latency budget is unbounded (offline). If feedback
    were needed, each microsecond of waiting costs every data qubit
    1/(4 T1) + 1/(2 T2) = 3.0 %/us of Pauli error at Garnet-class coherence
    [VA]. Section 4.3 measures the cost of the reset-feedback wait directly.
- **Early termination (real-time abort of LRB-D unif=1).** The lead finds
  this would execute 42 % (p = 1e-3) or 12 % (p = 5e-3) of the check rounds
  at m = 22.
  - Quantinuum Helios: supported. The Helios runtime "enables ... early
    termination of programs based on mid-circuit measurement or arbitrary
    classical logic" [LIT 21, Sec. II.4], and Guppy's `exit` "will end the
    current shot and carry on with subsequent ones", used for postselection
    [LIT 25]. The Guppy notebook ran on the Selene emulator.
  - IBM: not available as documented. The IBM service "currently only
    supports the conditional `if` statement". `for`, `while` and `switch`
    are unsupported, nested conditionals are not allowed, and measurement or
    reset inside conditionals is not supported [LIT 27]. OpenQASM 3 defines
    `end;` ("immediately terminates the program") [LIT 26], but IBM does not
    list it. A conditional skip of the remaining check rounds is also
    impossible, because they contain measurements and resets: **unknown /
    not supported as documented**.
  - IQM: only x, y, rx, ry, r can be classically controlled, on one bit,
    with an identity of equal duration otherwise [LIT 6]. No shot-termination
    primitive is documented: **unknown / not supported as documented**.
  - Where it exists (ions, millisecond rounds), early abort saves
    proportional QPU time. On IQM-class timing a full m = 22 shot is about
    53 us, so the saving is in rounds, not wall clock, until repetition
    overheads are known [CONJ].

## 6. Detector-error-model calibration from syndrome data

`dem_learning.py` learns the DEM of one circuit from N_train detection-event
samples of the "true" circuit, then decodes 1e5 independent test shots. The
oracle decoder uses the true DEM; every other decoder uses a substitute.
Scores are paired logical-error differences on the same shots
(mean over 4 sequences; SE from per-circuit paired variances).

- **Estimator.** For a detector set T of size k,
  sum over mechanisms e containing T of log(1 - 2 p_e) =
  (-1)^(k+1) 2^(1-k) sum over S subset T of (-1)^(k-|S|) log c_S,
  with c_S = E[(-1)^(sum of x_i over S)] [VA]. For k = 2 this is the p_ij
  formula of Chen et al., Eq. S11 [LIT 12], which the paper says the authors
  generalised to 3- and 4-point correlators. Takou and Brown use
  multi-point correlators for hyperedges [LIT 15].
  - Exact footprint probabilities follow by subtracting candidate supersets.
    With exact moments the recovery error is 4e-16
    [VN: `test_hyperedge_estimator_is_exact_with_exact_moments`].
  - The two forms of p_ij agree to 1e-9 on data
    [VN: `test_pij_formula_forms_agree_and_match_truth`].
  - Candidate footprints come from the nominal model's structure; no true
    footprint was missing in any circuit.
- **Observable split.** The probability that a footprint flips X_L is not
  identifiable from syndromes: ambiguous pairs such as Z0/Z2 give identical
  detection events, and only their sum is learnable [VA; VN:
  `test_ambiguous_footprints_exist_and_split_is_invisible`; consistent with
  the pure-distance limit, LIT 14]. The learned DEM therefore takes the split
  from the nominal (homogeneous) model. `*_oracle_split` uses the true split
  to separate the two error sources.

p_ij recovery (median relative error for true p_ij > 1e-3; the fraction of
pairs within 3 batch SE was 95-100 % in every case):

| N_train | IQM Garnet-class, m = 0-2 | T1=T2=100us, m = 1-4 | ion, m = 2-14 |
|---|---|---|---|
| 1e4 | 7-11 % | 6.5-7.5 % | 12-14 % |
| 1e5 | 2.1-3.8 % | 2.1-2.6 % | 3.7-4.7 % |
| 1e6 | 0.8-1.3 % | 0.6-0.8 % | 1.4-1.6 % |

Relative excess logical error of exact MLD, (eps - eps_oracle)/eps_oracle,
in the largest circuit of each set, with paired SE [VN,
`results/J/dem_learning/*.json`, field `rel_diff`]:

| decoder prior | Garnet-class m = 2 (eps_oracle = 0.257 +- 0.019) | T1=T2=100us m = 4 (0.215 +- 0.024) | ion m = 14 (0.065 +- 0.003) | heterogeneous Garnet m = 2 (0.238 +- 0.022) | heterogeneous T100 m = 4 (0.213 +- 0.024) |
|---|---|---|---|---|---|
| learned hyperedge DEM, N = 1e4 | +0.75 +- 0.12 % | +2.01 +- 0.15 % | +1.83 +- 0.19 % | +12.7 +- 0.3 % | +5.0 +- 0.2 % |
| learned, N = 1e5 | +0.26 +- 0.09 % | +0.20 +- 0.09 % | +0.03 +- 0.11 % | +12.3 +- 0.3 % | +2.7 +- 0.2 % |
| learned, N = 1e6 | +0.07 +- 0.06 % | +0.07 +- 0.06 % | -0.02 +- 0.06 % | +11.8 +- 0.3 % | +2.1 +- 0.2 % |
| learned N = 1e6 + per-qubit (true) split | same | same | same | +0.10 +- 0.05 % | +0.07 +- 0.06 % |
| pairs-only p_ij (graph) DEM, N = 1e6 | +40.6 +- 0.3 % | +51.3 +- 0.4 % | +86.3 +- 0.8 % | +55.5 +- 0.4 % | +56.9 +- 0.4 % |
| true DEM, all p x 0.5 / x 2 | +0.14 / +0.40 % (SE 0.1) | +0.25 / +0.40 % (0.1) | 0.00 / 0.00 % (0.01) | +0.42 / +1.52 % (0.1) | +0.26 / +0.79 % (0.1) |
| true DEM, all p x 0.1 / x 10 | +0.66 / +10.0 % (0.1, 0.3) | +1.29 / +21.7 % (0.1, 0.3) | 0.00 / +0.15 % (0.01, 0.07) | +1.56 / +11.4 % (0.1, 0.2) | +1.37 / +24.7 % (0.2, 0.3) |
| SI1000-shaped prior on the same circuit | +0.23 +- 0.10 % | +0.84 +- 0.13 % | +4.4 +- 0.5 % | +11.9 +- 0.3 % | +2.7 +- 0.2 % |
| homogeneous nominal model | 0 (= truth) | 0 (= truth) | 0 (= truth) | +12.7 +- 0.3 % | +2.2 +- 0.2 % |
| uniform (all classes equal) | +12.9 +- 0.3 % | +25.6 +- 0.3 % | +33.0 +- 0.7 % | +22.6 +- 0.3 % | +27.7 +- 0.3 % |

Notes on the table:

- eps_oracle is the mean over 4 sequences +- SEM.
- The heterogeneous truths draw a per-qubit log-normal spread
  (sigma = 0.5) of T1, T2, readout, 1q and CZ errors (`heterogeneous()`,
  seed 11).
- Over all depths, learned DEMs with N >= 1e5 cost at most +0.32 +- 0.11 %
  (ion, m = 2).
- The pairs-only DEM costs +24 % to +104 % for m >= 1, but only 1.2 % on the
  6-detector m = 0 circuit.
- The homogeneous split costs up to +74 % on the heterogeneous Garnet m = 0
  circuit (terminal round only), while the per-qubit split costs <= 0.11 %.
- Gap-postselected error at 50 % and 80 % acceptance follows the same
  ranking (`post_err_a0.5`, `post_err_a0.8`).

Two sources fit this picture.

- Data-driven weights also track slow drift, provided it is slow compared
  with t_cycle/p [LIT 13]. Learning per block of shots therefore applies to
  drifting hardware [CONJ].
- On Willow, DEMs estimated from syndromes fit unseen syndromes best, while
  DEMs trained for logical performance make better decoder priors [LIT 16].
  This matches what we find: syndrome learning recovers everything
  identifiable, and the residual loss sits in the split, which only labelled
  logical data or per-qubit calibration can fix. LRB supplies labelled
  logical outcomes for free (the ideal result is known), so the split could
  be fitted on training sequences and validated on held-out ones [CONJ].

## 7. Recommended protocol

**IQM-class square-lattice transmons (now).**

1. **Layout.**
   - Check-free runs: data on a 2x2 plaquette plus one site, with (2,3) on
     a native coupler (Section 1). This removes the 8-CZ routing, which is
     the largest error in the check-free protocol
     (f = 0.903 -> 0.985 for MLD, 0.950 -> 0.999 for detection).
   - Calibration runs with checks: the 3x3 patch. Put the best qubits on
     the fold axis q0, q1, q4 [CONJ].
2. **Compilation.**
   - Virtual SWAP(2,3) (frame bit), virtual S, S-dagger and Z, logical
     Paulis in the Pauli frame, CNOT = H CZ H.
   - Dynamical decoupling on idle data (the model uses echo T2: 21 us; the
     no-DD variant is 2x worse in the measurement-idle term).
3. **Benchmark circuits.** Check-free mirror LRB:
   - encoding;
   - m Cliffords and their mirror inverses;
   - transversal X readout of all 5 data;
   - no mid-circuit measurement, no reset, no feed-forward.
4. **Decoding.** Offline on the same shots:
   - detection: postselect on the 2 final X checks;
   - correction: exact MLD of the final syndrome;
   - optionally gap postselection as an acceptance knob.

   Report both against physical RB at p_RB = p and 5p and against compiled
   physical RB.
5. **Calibration.**
   - Interleave short check-round circuits (m <= 2, simultaneous readout,
     no-reset XOR bookkeeping), at >= 1e5 shots per circuit.
   - Learn the DEM with the hyperedge estimator.
   - Take the tie-break split from per-qubit calibration data (T1, T2,
     readout and gate errors per qubit) rather than a homogeneous model
     [CONJ that vendor calibration captures the per-location rates; VN that
     the per-qubit split recovers oracle MLD].
6. **When to add checks.**
   - Not for correction: with routed (2,3) and these gate and readout
     errors, no coherence time up to 1 ms made them pay (Section 4.3).
   - For detection, a check every 5th Clifford raises the accepted f from
     0.960 to 0.989 at Garnet-class coherence, but keeps only 3 % of shots
     at m = 14. Use it at m <= 6, or in a separate comparison dataset for
     the study's detection-vs-correction question.

**Trapped ions.**

- Checks are cheap (rejection 2.3 % at m = 0). The full repo protocol, with
  checks every round on all-to-all connectivity (native CZ(2,3)), is
  therefore affordable to m = 22 (48 % acceptance for unif=1) as the
  detection-vs-correction comparison dataset.
- Decode that dataset offline with LRB-D, MLD and gap postselection.
- Use the runtime's early exit for LRB-D [LIT 21, 25].
- For the benchmark number itself, the check-free circuit is better for
  correction (MLD f = 0.9977 against 0.9906) and almost as good for
  detection at twice the acceptance (f = 0.999973 at 94 % against 0.999983
  at 48 %) [VN].
- Correction is first order and loses to detection on these noise levels.

**Neutral atoms.**

- Readout is slow (500 us) but almost free for the data (<~0.1 %
  [LIT 23]).
- Checks cost mostly gate errors, so LRB-D keeps 70 % of shots at m = 2,
  but only 5 % at m = 22 at p = 6.25e-3.
- Run the check-free protocol (f = 0.99940 at 26 % rejection by m = 22)
  plus LRB-D at m <= 6, with native CZ(2,3) after a move.
- Atom loss and leakage are not modelled here; they need loss-aware
  decoding [CONJ].

## 8. Claims

| # | claim | label | evidence |
|---|---|---|---|
| 1 | The Tanner graph embeds in a square lattice only as a 3x3 patch (8 symmetric labellings) | VN | `test_tanner_graph_embeds_uniquely_as_3x3_patch` |
| 2 | The (2,3) coupling closes a 5-cycle, so it is not native on any bipartite lattice; in the patch d(q2, q3) = 4 | VA, VN | `test_logical_coupling_is_not_native` |
| 3 | Garnet has 4 patches, each with d(q2, q3) = 4 | VN, LIT 2 | `test_garnet_coupling_map_and_patches`, `layout.json` |
| 4 | SWAP(2,3) can be a relabelling; ancilla roles swap between frames | VN | `test_frame_ancilla_roles_match_patch_neighbours`, deterministic noiseless circuits in both frames |
| 5 | CZ(2,3) on the patch in 8 CZ with ancillas restored | VN | `test_routed_cz23_equals_cz` |
| 6 | Round 12 CZ / 18 pulses; Clifford 5.33 CZ routed, 0.67 native; physical SWAP 25.3 | VN | `test_gate_counts`, `gate_counts.json` |
| 7 | Twirled T1/T2 formula and r -> p conversions | VA, VN, LIT 9, 10 | `test_pauli_twirl_matches_kraus`, `test_depolarizing_from_average_infidelity` |
| 8 | Garnet-class idle during the readout window = 2.94 % = 4.7p (SI1000: 2p) | VA, LIT 1, 3, 4 | `mapping.json` |
| 9 | Hardware circuits are deterministic and exact MLD = brute force on them | VN | `test_noiseless_hw_circuits_are_deterministic`, `test_hw_dem_works_with_exact_mld_and_brute_force` |
| 10 | IQM Garnet-class: MLD with checks f = 0.7705 +- 0.0029 < check-free MLD 0.9033 +- 0.0024 < check-free detection 0.9497 +- 0.0021 < physical (0.979-0.997) | VN | `hw_iqm_garnet/summary.json` |
| 11 | No reset beats conditional reset on IQM-class timing (MLD f 0.809 vs 0.771) | VN | `hw_iqm_garnet_noreset` |
| 12 | Native CZ(2,3): check-free detection f = 0.99904 +- 0.00005 > compiled physical 0.99701 +- 0.00007 | VN | `hw_iqm_native23` |
| 13 | Ions: check-free detection f = 0.999973 at 5.6 % rejection beats compiled physical 0.99992; MLD with checks 0.99061 | VN, LIT 21 | `hw_ion_helios` |
| 14 | At matched acceptance, gap postselection equals LRB-D (d = 2) | VN | `summary.json` `gap` |
| 15 | Offline Pauli-frame decoding is exact for Clifford LRB; no feed-forward is needed | VA, LIT 17 | Section 5 |
| 16 | Helios can terminate a shot early; IBM (only `if`) and IQM (single-bit rotations) document no such primitive | LIT 21, 25, 26, 27, 6 | Section 5 |
| 17 | Generalised p_ij (hyperedge) estimator is exact for independent mechanisms | VA, VN | `test_hyperedge_estimator_is_exact_with_exact_moments` |
| 18 | Learned hyperedge DEM (>= 1e5 shots) costs <= 0.32 +- 0.11 % relative logical error; pairs-only p_ij DEM costs 24-104 % (m >= 1) | VN | `dem_learning/*.json` |
| 19 | MLD is robust to a factor-2 error in p (<= 1.5 %) but not to x10 (up to 31 %) | VN | `dem_learning/*.json` |
| 20 | The d = 2 split is not learnable from syndromes; with heterogeneous qubits a homogeneous split costs 2-74 %, and the per-qubit split recovers the oracle (<= 0.11 %) | VA, VN, LIT 14 | `test_ambiguous_footprints_exist_and_split_is_invisible`, `*_het0.5.json` |
| 21 | Whether 8 CZ is optimal for the routed CZ(2,3) | CONJ | lower bound 4 [VA] |
| 22 | IQM conditional-reset latency 0.5 us | UNVERIFIED | swept in Section 4.3 |
| 23 | Atoms (Rb, zoned): check-free detection f = 0.99940 +- 0.00003 beats compiled physical 0.99708 +- 0.00007; MLD with checks 0.96282 +- 0.00036 | VN, LIT 23, 24 | `hw_atom_rb` |
| 24 | The lead's builder with IQM-mapped SI1000 factors (sequential readout) gives MLD f = 0.709 +- 0.006, 0.06 below the parallel-readout compilation | VN | `repo_iqm_mapped` |
| 25 | Physical RB (lead's convention) at p_RB = p: 0.99583 +- 0.00008 (p = 6.25e-3) and 0.99934 +- 0.00001 (ion p); at 5p: 0.9793 +- 0.0004 and 0.99670 +- 0.00006 | VN | `physical_rb.json` |
| 26 | The check-free protocol embeds with native (2,3) on a 2x2 plaquette plus one site | VA | encoder and logical couplings form a bipartite 4-cycle plus a pendant |
| 27 | For d = 2 correction, sparser checks are always better and check-free is best: IQM-class at T1 = T2 = 21 us to 1 ms, ions, atoms, native (2,3) | VN | `sweeps/coherence`, `sweeps/platforms_sparse` |
| 28 | Feedback (reset) latency costs about 0.07 in f per us on IQM-class coherence; a 10 % budget is about 250 ns; no-reset >= zero-latency reset | VN, VA | `sweeps/latency` |
| 29 | A shorter readout helps but does not rescue checks: MLD f = 0.816 at 250 ns against check-free 0.905 | VN | `sweeps/readout` |

## 9. Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
H=studies/correction-decoders/qubit/scripts/hardware_model.py; D=studies/correction-decoders/qubit/scripts/dem_learning.py
python3 -m pytest -q tests/test_qubit512_hw.py                  # 26 tests
python3 $H layout; python3 $H counts; python3 $H mapping; python3 $H physrb
# heavy runs: srun --jobid=<id> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c "..."
# with --workers <= 24 (unbound steps are otherwise pinned to the same CPUs); ~3-7 min each
python3 $H simulate --device iqm_garnet --tag hw_iqm_garnet --workers 24 --shots 100000
python3 $H simulate --device iqm_garnet --no-reset --tag hw_iqm_garnet_noreset --workers 24 --shots 100000
python3 $H simulate --device iqm_garnet_native23 --tag hw_iqm_native23 --workers 24 --shots 100000
python3 $H simulate --device iqm_t100 --tag hw_iqm_t100 --workers 24 --shots 100000
python3 $H simulate --device ion_helios --tag hw_ion_helios --workers 12 --shots 100000 --decode-shots 50000 --reduced --no-mld2
python3 $H simulate --device atom_rb --tag hw_atom_rb --workers 12 --shots 100000 --decode-shots 50000 --reduced --no-mld2
python3 $H simulate --device iqm_garnet --kind repo --tag repo_iqm_mapped --workers 12 --shots 100000 --decode-shots 50000 --reduced
python3 $H sweep --which coherence --workers 12        # also: latency, readout, platforms_sparse
python3 $D study --truth iqm_garnet --depths 0,1,2 --num-seq 4 --n-test 100000 --workers 12
python3 $D study --truth iqm_garnet --hetero-sigma 0.5 --depths 0,1,2 --num-seq 4 --n-test 100000 --workers 12
python3 $D study --truth iqm_t100 --depths 1,2,4 --num-seq 4 --n-test 100000 --workers 12
python3 $D study --truth iqm_t100 --hetero-sigma 0.5 --depths 1,2,4 --num-seq 4 --n-test 100000 --workers 12
python3 $D study --truth ion_helios --depths 2,6,14 --num-seq 4 --n-test 100000 --workers 12
```

## 10. References (all fetched in this session; title, authors, year, ID)

1. L. Abdurakhimov et al. (113 authors, IQM), "Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer", 2024, arXiv:2408.12433.
2. IQM, qiskit-on-iqm fake Garnet backend, `src/iqm/qiskit_iqm/fake_backends/fake_garnet.py` (coupler list, `prx`/`cz`/`measure`, sample-profile durations prx 20 ns, cz 40 ns), github.com/iqm-finland/qiskit-on-iqm, fetched 2026-10-28.
3. M. Renger et al. (65 authors, IQM), "A Superconducting Qubit-Resonator Quantum Processor with Effective All-to-All Connectivity", 2025, arXiv:2503.10903 (Table S4: readout pulse 800 ns, DRAG 40 ns; Table S2: T1, T2e).
4. R. Sharma, "Symmetry-Preserving Variational Quantum Simulation of the Heisenberg Spin Chain on Noisy Quantum Hardware", 2025, arXiv:2512.23009 (Table 3: IQM Garnet calibration snapshot).
5. Amazon Braket, "IQM" device page, aws.amazon.com/braket/quantum-computers/iqm/, fetched 2026-10-28.
6. IQM client 36.0.0 documentation, "Qiskit on IQM" user guide (docs.iqm.tech/iqm-client/user_guide_qiskit.html), and iqm-pulse 15.0.0 `Reset_Conditional` (docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.reset.Reset_Conditional.html), fetched 2026-10-28.
7. F. Vigneau et al. (16 authors, IQM), "Quantum error detection in qubit-resonator star architecture", 2025, arXiv:2503.12869 (ancillas not reset between cycles).
8. C. Gidney, M. Newman, A. Fowler, M. Broughton, "A Fault-Tolerant Honeycomb Memory", 2021, arXiv:2108.10457, Quantum 5, 605 (SI1000, Table 2).
9. Y. Tomita, K. M. Svore, "Low-distance Surface Codes under Realistic Quantum Noise", 2014, arXiv:1404.3747, Phys. Rev. A 90, 062320 (Eqs. 9-10).
10. M. A. Nielsen, "A simple formula for the average gate fidelity of a quantum dynamical operation", 2002, arXiv:quant-ph/0205035, Phys. Lett. A 303, 249.
11. D. C. McKay, C. J. Wood, S. Sheldon, J. M. Chow, J. M. Gambetta, "Efficient Z-Gates for Quantum Computing", 2017, arXiv:1612.00858, Phys. Rev. A 96, 022330.
12. Z. Chen et al. (91 authors, Google Quantum AI), "Exponential suppression of bit or phase flip errors with repetitive error correction", 2021, arXiv:2102.06132, Nature 595, 383, DOI 10.1038/s41586-021-03588-y (Eq. 2; Supplement Eq. S11).
13. S. T. Spitz, B. Tarasinski, C. W. J. Beenakker, T. E. O'Brien, "Adaptive weight estimator for quantum error correction", 2017, arXiv:1712.02360, Adv. Quantum Technol. 1800012 (2018).
14. T. Wagner, H. Kampermann, D. Bruss, M. Kliesch, "Pauli channels can be estimated from syndrome measurements in quantum error correction", 2021, arXiv:2107.14252, Quantum 6, 809 (2022).
15. E. Takou, K. R. Brown, "Estimating decoding graphs and hypergraphs of memory QEC experiments", 2025, arXiv:2504.20212.
16. K. E. Arms, M. J. McHugh, J. E. Nyhan, W. F. Reus, J. L. Ulrich, "Estimating Detector Error Models on Google's Willow", 2025, arXiv:2512.10814.
17. E. Knill, "Quantum Computing with Very Noisy Devices", 2004, arXiv:quant-ph/0410199 (related DOI 10.1038/nature03350).
18. F. Battistel, C. Chamberland, K. Johar, R. W. J. Overwater, F. Sebastiano, L. Skoric, Y. Ueno, M. Usman, "Real-Time Decoding for Fault-Tolerant Quantum Computing: Progress, Challenges and Outlook", 2023, arXiv:2303.00054, Nano Futures 7, 032003.
19. L. Caune et al. (25 authors), "Demonstrating real-time and low-latency quantum error correction with superconducting qubits", 2024, arXiv:2410.05202.
20. G. P. Geher, M. Jastrzebski, E. T. Campbell, O. Crawford, "To reset, or not to reset -- that is the question", 2024, arXiv:2408.00758.
21. A. Ransford et al. (186 authors, Quantinuum), "Helios: A 98-qubit trapped-ion quantum computer", 2025, arXiv:2511.05465.
22. S. A. Moses et al. (96 authors, Quantinuum), "A Race Track Trapped-Ion Quantum Processor", 2023, arXiv:2305.03828, Phys. Rev. X 13, 041052.
23. D. Bluvstein et al. (21 authors), "Logical quantum processor based on reconfigurable atom arrays", 2023, arXiv:2312.03982, Nature, DOI 10.1038/s41586-023-06927-3.
24. S. J. Evered et al. (15 authors), "High-fidelity parallel entangling gates on a neutral atom quantum computer", 2023, arXiv:2304.05420, Nature 622, 268.
25. Quantinuum, Guppy documentation, notebook "Postselection: exit and panic", docs.quantinuum.com/guppy, fetched 2026-10-28.
26. OpenQASM Live Specification, "Classical instructions" (`end;` statement), openqasm.com/language/classical.html, fetched 2026-10-28.
27. IBM Quantum documentation, "Execute dynamic circuits", quantum.cloud.ibm.com/docs/guides/execute-dynamic-circuits, fetched 2026-10-28.
