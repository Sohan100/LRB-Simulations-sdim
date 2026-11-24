# H1: Hardware protocol for correction-based (hybrid) LRB of the [[5,1,2]]_3 folded code on superconducting transmon qutrits

Agent H1, 2026-10-28 (evening run; this file replaces an earlier draft of the
same note, see section 9). Read with `00-brief.md`.

Labels on every claim:

- **[VA]** verified analytically (argument given in place);
- **[VN: script]** verified numerically (scripts in
  `notes/H1-scripts/`, logs next to them as `<script>.log`);
- **[LIT n]** from literature, reference n at the end, fetched by me in
  this task (IQM documentation pages and vendor blogs count as literature);
- **[CONJ]** conjecture or assumption.

Scripts written for this note (all run on the login node with
`module load python/3.11`, one core, `OMP_NUM_THREADS=1`):

| script | what it does | runtime |
|---|---|---|
| `sc_graphs.py` | coupling graphs parsed from `src/lrb/code_definitions.py`; bipartiteness, all generator choices, relay parity, embeddings into grids and IQM Garnet, exhaustive minimum relay hops, planarity, Clifford token statistics | 2 min |
| `sc_idle.py` | Markovian qutrit idle channel (closed form plus matrix-exponential check), Weyl-twirl split, coherence sets from the fetched papers | seconds |
| `sc_gate_identities.py` | CNOT/CNOT_INV from CZ or CZ^dagger, diagonal P, logical H without SWAP, F_pro of SI1000 N2, qutrit reset level bookkeeping | seconds |
| `sc_dem_hw.py` | three check schedules (repo sequential, parallel readout, square-lattice relay layout) and const-0, under SI1000 and duration-aware hardware models, on the repo generator and compact DEM; noiseless determinism; first-order fault classes; sequence spread; empirical lookup decoder | 4.6 min |
| `sc_budget.py` | per-step decay estimates, variance components and shot/time budgets parsed from `sc_dem_hw.log` | instant |

The files `h1_*.py` in the same folder belong to the earlier draft and are
superseded. A pre-existing worktree venv `.venv-h1/` (pypdf only) was used to
extract text from three downloaded PDFs (refs 2, 17, 24); downloaded files were
kept in my session scratchpad, not in the repository.

## 0. Summary

1. **Layout.** The 12 check couplings form the unrotated d=2 surface-code
   patch on a 3x3 grid, unique up to symmetry **[VN: sc_graphs]**. The fold
   gate (SWAP/CZ between data 2 and 3) joins opposite corners. Adding the
   fold coupling creates an odd 5-cycle, so no square lattice (IQM's rotated
   "crystal" lattice included) can make every coupling nearest-neighbour
   **[VA + VN]**. No choice of stabilizer generators avoids this
   (exhaustive over all 63,180 generator bases) **[VA + VN]**. With the fold
   native, the frustrated couplings are those of S_Z1 and S_X2 to data 3 (or
   of S_Z2 and S_X1 to data 2). The repo's split-ancilla variant relays S_Z1
   to data 3 but S_X2 to data 4; its coupling graph has an odd 5-cycle even
   without the fold, so it is not a square-lattice layout either **[VN]**.
   The exhaustive minimum on a square lattice is 6 relay hops per round
   (24 instead of 12 two-qutrit gates) on 11 qutrits and 13 couplers. This
   layout fits IQM Garnet's coupling map in 24 ways **[VN]**.
2. **Gates.** The best published transmon two-qutrit gate is a CZ^dagger
   with a 2.7% Weyl (process) error in 580 ns **[LIT 1]**. This equals SI1000
   p = 0.027, because SI1000's two-qutrit p is exactly a process infidelity
   **[VA + VN]**. Every repo CNOT and CNOT_INV can be compiled to that
   CZ^dagger plus target basis changes **[VN: sc_gate_identities]**.
   Single-qutrit Clifford errors are 2.4e-3 to 5.4e-3 **[LIT 2, 4]**. P and
   Z are virtual **[LIT 2; VA]**, and the logical SWAP can be a classical
   relabelling **[VA + VN]**.
3. **Readout, reset, idle.**
   - Three-state readout errors are 2-7% in 140-750 ns **[LIT 3, 11, 12, 14, 8]**,
     against SI1000's 5p = 13.5% at p = 0.027.
   - An echo-limited qutrit idles at 0.7% per 500 ns readout layer
     **[VN: sc_idle]**, against SI1000's 2p = 5.4%.
   - SI1000 has no idle noise during gate layers. With 580 ns CZs, an LRB
     step (Clifford, check round, reset) lasts 6-12 µs, during which a data
     qutrit idles at 1.4-2.9% per µs **[VN]**.
   - Reading the four ancillas in one layer instead of four cuts the
     measurement-layer idle on the data 4x (8p to 2p per round) **[VA]**.
     It raises unif-1 acceptance at SI1000 p = 1e-3, m = 10 from 0.31 to 0.47
     **[VN: sc_dem_hw]**.
   - IQM's documented composite reset maps a qutrit's |2> to |1>; a qutrit
     needs f0g1, prx_12(pi), f0g1 **[LIT 31; VN]**.
4. **What the hardware models say** **[VN: sc_dem_hw]** (duration-aware DEM,
   section 3.6):
   - At today's best components (HW-now), a shot already contains 3-6
     expected faults at m = 2. LRB-D with a check after every Clifford keeps
     0.5-7% of shots at m = 2 and essentially none at m = 6, and the raw
     logical fidelity is 0.08-0.24 at m = 2.
   - Only the check-free const-0 protocol is usable. It reaches accepted
     fidelity 0.996 / 0.94 / 0.81 at m = 2 / 6 / 10, with acceptance
     0.76 / 0.39 / 0.24.
   - With 1%-error, 200 ns CZs (HW-target), parallel readout on the 9-qutrit
     coupling graph keeps 47% at m = 2. The square-lattice relay layout keeps
     only 24%, worse than the repo's sequential schedule (38%), because 12
     extra CNOTs per round cost more than three readout layers of idling.
5. **Correction on a distance-2 code** **[VN: sc_dem_hw]**:
   - At circuit level, 40-56% of all single faults fall in syndrome-history
     classes that also contain faults with a different logical effect.
     Maximum-likelihood decoding fails on 8-16% of single faults.
   - An empirical lookup decoder at m = 2 under HW-target (parallel readout)
     raises the all-shot fidelity from 0.66 (raw) to 0.77. A hybrid that
     rejects low-confidence syndromes reaches 0.85 at 88% acceptance.
     LRB-D reaches 0.97 at 46%.

   Correction keeps every shot but cannot match postselected fidelity on
   this code.
6. **Leakage.**
   - No |3>-leakage rate per qutrit gate was found in the fetched sources.
   - Qubit analogues: 0.09-0.11% per round into |2> **[LIT 17]**, and a
     three-state-readout leakage rejection of 0.17% per data qubit per cycle
     and 0.94% per auxiliary qubit per cycle **[LIT 12]**.
   - Four-state readout (91% |3> assignment) **[LIT 4]** can herald leakage
     as an erasure. Leakage handling therefore belongs in the hybrid decoder
     as a reject (or erasure) flag.
7. **Feedback.** For Clifford-only LRB every correction is a classical
   relabelling of the terminal outcome, so decoding is offline with no
   latency limit **[VA]**. Reset can be unconditional **[VA; LIT 17, 18, 31]**.
   - Real-time loops would cost 1.4-4% per µs of data idling
     **[VN: sc_idle]**.
   - Ternary feed-forward has been shown once, on QubiC with 40 ns FPGA state
     discrimination **[LIT 14]**.
   - IQM documents only one-bit-conditioned PRX gates **[LIT 29, 30]**.
8. **IQM** (sourced facts only; section 6):
   - Square-lattice crystal QPUs with 20-40 ns coupler CZs, mid-circuit
     measurement and experimental one-bit feed-forward within a local group
     **[LIT 24, 29, 30, 36, 37]**.
   - Pulse-level access through Pulla on Resonance **[LIT 32, 33]**.
   - A prx_12 operation and f0g1 leakage/reset units in the pulse library
     **[LIT 31]**.
   - An IQM-designed and -fabricated 20-transmon chip has been operated as a
     15-qutrit chain with non-IQM electronics: 180 ns SU(3) gates, coupler
     cross-Kerr interactions, 93% qutrit readout in 500 ns, and no mid-circuit
     measurement **[LIT 8]**.
   - Not found: qutrit gates in the circuit API, multi-state discrimination,
     ternary feed-forward.
9. **Protocol** (section 7). A sustainable plan uses only documented
   capabilities, runs every decoder offline on one dataset, and spends shots
   on many sequences rather than many shots per sequence. It has three stages:
   - **(A)** const-0 LRB on 5 data qutrits, with no mid-circuit measurement
     (feasible today).
   - **(B)** memory and stability runs that learn a Z_3 detector error model.
   - **(C)** LRB with checks, on parallel readout, once the two-qutrit error
     is at or below about 1%. A custom planar coupler layout should be
     preferred to the relay layout.

   Shot budget: the sequence-to-sequence spread of the raw fidelity (sd
   0.009-0.036) makes extra shots beyond 150-6,500 per sequence useless
   **[VN: sc_budget]**. Use about 100 sequences of 3,000 shots per depth:
   2.7e6 shots for 9 depths, 0.07-0.75 h at 0.1-1 ms per shot.

## 1. Layout of 5 data + 4 ancilla qutrits on a square lattice

### 1.1 Coupling graphs [VN: sc_graphs]

Couplings parsed from the repo circuits:

- Checks (z_ALT and X blocks): 5-{0,1,3}, 6-{1,2,4}, 7-{0,1,2}, 8-{1,3,4}.
- Logical H (SWAP) and logical P (CZ): the fold coupling 2-3.
- Encoder: 0-1, 0-3, 1-2, 2-4.
- Split variant: 5-{0,1,9}, 9-3, 6-{1,2,4}, 7-{0,1,2}, 8-{1,3,10}, 10-4.

| graph | bipartite | shortest odd cycle |
|---|---|---|
| 12 check couplings | yes | none |
| checks + fold | no | 0-5-3-2-7 |
| repo split variant | no | 5-1-8-3-9 |
| split + fold | no | 5-1-8-3-9 |
| Stage A: encoder + fold | yes | none |
| checks + fold + encoder | no | 0-5-1 |

A nearest-neighbour square lattice is bipartite, and so is any subgraph of it,
including IQM's crystal lattice (a square lattice rotated by 45 degrees
**[LIT 24]**) and Garnet's coupling map **[VA]**. A graph with an odd cycle
therefore cannot be realized with every coupling native.

The check graph embeds into a 3x3 grid in exactly 8 ways, the symmetry group
of the square **[VN]**: data 1 sits at the centre, data 0, 2, 3, 4 at the
corners, and the ancillas on the edges. Data 0 and 4 are opposite corners on
the fold axis; data 2 and 3 are the mirror-image corners. The "folded" code
is this patch folded along the 0-1-4 diagonal, and the fold gate acts on the
mirror pair (2, 3) **[VA]**. In the planar patch, 2 and 3 are at lattice
distance 4, and every shortest path between them crosses a data site. The
shortest path avoiding data sites has length 8 (7 relay hops) **[VA]**.

### 1.2 No generator choice removes the obstruction [VA + VN: sc_graphs G2]

Argument:

- In a bipartite graph containing the edge 2-3, data 2 and 3 have opposite
  colours. An ancilla adjacent to 3 has 2's colour, an ancilla adjacent to 2
  has 3's colour, and no ancilla can be adjacent to both.
- Every element of the stabilizer group that acts on exactly one of data 2
  and 3 also acts on data 1. There are 8 such one-dimensional subgroups,
  with supports {0,1,2}, {0,1,3}, {1,2,4}, {1,3,4}, {0,1,2,4} and {0,1,3,4};
  no element avoids both 2 and 3 **[VN]**.
- Any generating set needs an element acting on 3 (S_Z1 does) and an element
  acting on 2 (S_Z2 does). By the first bullet neither may act on both, so
  by the second bullet both act on data 1, and data 1 would need both
  colours.

An exhaustive check over all 63,180 bases of the group (one ancilla per
generator, direct couplings, plus the fold edge) finds no bipartite graph
**[VN]**.

### 1.3 Which stabilizers are geometrically nonlocal; the repo split variant [VN: sc_graphs G3]

- **Planar patch.** All four stabilizers are local, and only the fold gate
  is nonlocal: distance 4 for CZ(2,3), which appears 2.93 times per LRB step
  **[VN, G7]**. Implementing it costs a 7-hop relay chain per CZ, about 41
  extra two-qutrit gates per step **[VA]**.
- **Fold made native (2 and 3 adjacent).** Parity then forces an odd number
  of extra hops on some check couplings. The minimal single-relay sets
  restoring bipartiteness have two elements: {S_Z1-data 3, S_X2-data 3} or
  {S_Z2-data 2, S_X1-data 2} **[VN]**. So the frustrated stabilizers are the
  two that share one fold qutrit. The repo's choice of S_Z1 and S_X2 as the
  "boundary" stabilizers is one of the two consistent choices **[VN]**.
- **The repo split variant**
  (`FoldedQutritSplitAncillaUniformDetectionCode`):
  - It relays S_Z1's coupling to data 3 through ancilla 9, and S_X2's
    coupling to data 4 through ancilla 10.
  - The second relay is on the wrong edge. With 8-3 direct, the cycle
    8-3-2-7-1 is odd once the fold is present. Even without the fold, the
    split graph has the odd 5-cycle 5-1-8-3-9 **[VN]**.
  - The variant therefore presupposes a coupler graph with odd cycles: not a
    square lattice, not Garnet, not heavy-hex (all bipartite **[VA]**).
  - Its cost (+4 two-qutrit gates per round, 6 measured ancillas)
    underestimates the square-lattice cost (+12, section 1.4).
- **Relay consistency.** The variant measures relays 9 and 10 and demands
  they read 0 (`split_ancilla_stabilizer_check`). Section 1.5 keeps this.

### 1.4 Cheapest square-lattice layouts [VN: sc_graphs G5]

Cost model:

- A check coupling at data-avoiding grid distance L costs L-1 relay hops.
- Each hop adds 2 two-qutrit gates: compute, couple, uncompute, as in the
  repo's split circuits.
- Relays may pass through free sites or through other ancillas' sites, but
  never through data sites. Ancillas are assigned optimally (Hungarian
  algorithm).

The search fixes data 2 at the origin and data 3 at the fold offset. It
enumerates data 0, 1 and 4 within L1 radius R = 5 of data 2 or 3, and prunes
with the lower bound Σ_stabilizers (x-range + y-range of the support - 3).

The search is exhaustive for totals of 5 hops or less. A data qutrit
q ∈ {0,1,4} farther than R from both 2 and 3 forces at least 2(R-2) = 6 hops,
because two (data 0, 4) or four (data 1) supports pair q with 2 or 3 **[VA]**.

| fold placement | minimum check relay hops per round | extra two-qutrit gates per LRB step |
|---|---|---|
| native (distance 1) | 6 | 12 |
| distance 2, straight (one relay per CZ) | 5 | 10 + 2 x 2.93 = 15.9 |
| distance 2, diagonal | 6 | 12 + 5.9 = 17.9 |
| planar patch (fold by a 7-hop chain) | 0 | about 41 |

**Proposed layout (native fold).** Dn = data qutrit n, An = ancilla n,
R1 and R2 = relay-only qutrits (wires 9 and 10 in the scripts):

```
        c3    c4    c5    c6
  r2    D0
  r3    A5    R1    R2
  r4    A7    D2    D3
  r5    D1    A6    A8    D4
```

- The layout uses 11 qutrits and 13 couplers: A5-D0, A6-D1, A6-D2, A7-D1,
  A7-D2, A8-D3, A8-D4, D2-D3 (fold), A5-A7, A5-R1, R1-R2, R2-D3, A6-A8. All
  are grid-adjacent and the total is 6 relay hops **[VN]**.
- Relayed couplings:
  - A5-D1 through A7 (1 hop); A5-D3 through R1 and R2 (2 hops);
  - A6-D4 through A8 (1 hop); A7-D0 through A5 (1 hop); A8-D1 through A6
    (1 hop).
- Garnet: 24 placements of this coupler graph exist on the 30-edge coupling
  map **[VN; map from LIT 34]**. One is:

  | qutrit | D0 | D1 | D2 | D3 | D4 | A5 | A6 | A7 | A8 | R1 | R2 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Garnet qubit | QB1 | QB6 | QB10 | QB15 | QB17 | QB4 | QB11 | QB5 | QB16 | QB9 | QB14 |

### 1.5 Round schedule for the relay layout [VN: sc_dem_hw]

A relay that holds a stale, misread value behaves differently depending on
its type.

- **Z-type (collector) relay** that starts in |s>. It adds the known s to
  the ancilla's syndrome. A misread s only shifts a recorded syndrome, which
  is a measurement-type error **[VA]**.
- **X-type (fan-out) relay** that starts in |s>. It applies X^s to the data.
  A misread s leaves an unrecorded X on the data, which is a data error
  **[VA]**. The earlier draft's schedule used just-measured ancillas as
  fan-out relays, which converts their readout errors into data errors
  (section 9).

Schedule used in `sc_dem_hw.py` (relay11, standard frame):

1. **X half.** H on A7 and A8. Then the X-type checks:
   - S_X1 on A7 directly to D1 and D2, and to D0 through the freshly reset
     A5: CSUM(A7→A5), CSUM(A5→D0), CSUM^-1(A7→A5);
   - S_X2 on A8 directly to D3 and D4, and to D1 through the freshly reset
     A6.

   Then H_INV on A7 and A8. A7 and A8 now hold their X-syndromes s7 and s8
   in basis states and are **not measured yet**.
2. **Z half.**
   - S_Z1 on A5: direct to D0; to D1 with A7 as collector relay; to D3 with
     the chain R2, R1.
   - S_Z2 on A6: direct to D1 and D2; to D4 with A8 as collector relay.

   A5 then holds S_Z1 + s7, and A6 holds S_Z2 + s8. In the analysis, use the
   detectors m5 - m7 and m6 - m8; this is an invertible relabelling, and
   postselection on "all zero" is unchanged **[VA]**.
3. **One readout layer** of A5, A6, A7, A8, R1 and R2. R1 and R2 are relay
   cleanliness detectors, as in the repo's split variant. Then
   unconditional reset of all six.

Properties:

- 24 two-qutrit gates per round; noiseless sdim runs give deterministic zero
  for all six detectors and the logical (0 of 18 shots bad; the same holds
  for the parallel 9-qutrit schedule) **[VN]**.
- In the frame swapped by an odd number of logical H (section 1.6), A5 and
  A6 carry the X-type checks and A7 and A8 the Z-type checks. The same rule
  still applies: X-type checks first, with fresh fan-out relays.

### 1.6 Logical H needs no physical SWAP [VA + VN]

- **Circuit identity.** Deleting SWAP(2,3) and relabelling wires 2↔3 in
  every later operation (and in the readout) gives an identical circuit
  **[VA]**.
- **Implementability.** The single-qutrit part of logical H (H on
  {0,2,3,4}, H_INV on 1) maps the stabilizer group onto its 2↔3-relabelled
  version, and X_L, Z_L onto the relabelled Z_L, X_L^-1
  **[VN: sc_gate_identities I4]**. The four ancilla supports are invariant
  under the relabelling, so every ancilla keeps its couplings and only
  switches between X- and Z-type extraction (5↔7, 6↔8) **[VA]**.
- **Frame at the end.** Forward and inverse Cliffords have equal H-token
  parity for all 216 catalogue elements, so the frame is standard at the
  terminal readout **[VN: sc_graphs G7]**.
- **Independent check.** The critic script `scripts/critic_H1-sc-hardware_faults.py`,
  run by me in this task, finds 0 of 48 noiseless shots bad with relabelling
  and 48 bad without **[VN]**.
- **Gate counts.** Catalogue Cliffords average 1.653 H, 1.463 P, 1.005 X and
  1.005 Z tokens. Per LRB step (forward plus inverse), 2.93 CZ(2,3) remain
  physical, and the 3.31 SWAPs become free **[VN: G7]**.

### 1.7 Custom chip option [VN + LIT]

The checks-plus-fold graph is planar with maximum degree 4 (data 1 has degree
4; data 2, 3 and the ancillas have 3) **[VN: G6]**. A dedicated 9-qutrit,
13-coupler chip would make every coupling native, with no relays.
IQM's open-source KQCircuits (GPL-3.0, KLayout-based) generates multi-layer
chip geometry, mask layouts and simulation exports **[LIT 35]**. Whether a
foundry would fabricate a non-square layout for this purpose is unknown
**[CONJ]**. Section 3.6 shows that this matters: at HW-target, the parallel
9-qutrit schedule (which this chip would allow) keeps twice as many shots as
the relay layout at m = 2.

## 2. Native gate set for transmon qutrits

### 2.1 Single-qutrit gates

| source | gate set and durations | error |
|---|---|---|
| Morvan 2021 **[LIT 2]** | pi and pi/2 rotations in {0,1} and {1,2}; phase gates virtual (software); 3.325 native gates per Clifford; qutrit H 128 ns | qutrit RB r = 2.38e-3 to 3.94e-3 per Clifford; r_H = 4.6e-3; errors largest in the {1,2} subspace |
| Blok 2021 **[LIT 3]** | single-qutrit gates within 30 ns | subspace RB f01 = 0.9997, f12 = 0.9994 |
| Liu 2023 **[LIT 4]** | pi/2 cosine pulses 45 ns (0-1), 35 ns (1-2), 85 ns (2-3), each with a 10 ns buffer | SU(3) r = 5.4e-3 per Clifford (1.02e-3 per pi/2 pulse); SU(4) 5.8e-2 |
| Yu 2026 **[LIT 5]** | 35 ns qutrit H and X by SU(3) coherent control | 99.5% average fidelity (RB) |
| Wang 2025 **[LIT 9]** | high E_J/E_C transmons, up to 12 levels | process infidelity < 3e-3 for qubit-like operations in each adjacent-level subspace of the lowest 10 levels |
| Goss 2026 **[LIT 8]** | IQM-fabricated chip, arbitrary SU(3) gate 180 ns | RB in figure only |

Notes:

- P = diag(1, 1, ω) and Z are diagonal in the repo's conventions
  **[VN: sc_gate_identities I3]**, so they are virtual frame updates on
  transmons **[LIT 2]**.
- SI1000's 0.1p per single-qutrit gate at the gate-matched p = 0.027 is
  2.7e-3. That is consistent with measured physical gates (process infidelity
  e = 4r/3 = 3.2e-3 to 7.2e-3). SI1000 is pessimistic for P and Z, which cost
  nothing **[VA]**.

### 2.2 Two-qutrit gates

| source | mechanism | duration | fidelity |
|---|---|---|---|
| Goss 2022 **[LIT 1]** | microwave-activated differential AC-Stark cross-Kerr between fixed-frequency transmons | CZ^dagger 580 ns, CZ 783 ns | process fidelity 97.3(1)% and 95.2(3)% (cycle benchmarking; Weyl error rates 2.7% and 4.8%); "a factor of 4 reduction in infidelity" over the previous best of 11.1% |
| Blok 2021 **[LIT 3]** | cross-Kerr CSUM | about 1.5 µs | 0.889 (process tomography) |
| Morvan 2021 **[LIT 2]** | CSUM | not stated | cycle-benchmarking process fidelity 0.82 (gate fidelity 0.85) |
| Luo 2023 **[LIT 6]** | tunable coupler | not in abstract | conditional phase 89.3%; qutrit EPR state 95.5% |
| Goss 2026 **[LIT 8]** | fast flux pulses on IQM couplers give a strong dispersive (cross-Kerr) interaction | depends on pulse amplitude and length | no gate fidelity reported |
| Subramanian 2023 **[LIT 7]** | parametric coupler, two partial swaps (analysis) | not in abstract | not in abstract |

Coherence of the Goss 2022 qutrits **[LIT 1]**:

- T1(1→0) = 45-125 µs and T1(2→1) = 28-63 µs.
- Echo T2: 01 transition 63-190 µs, 12 transition 28-61 µs.
- Ramsey T2 for the 12 transition: 10-18 µs.

### 2.3 Compiling the repo gates [VA + VN: sc_gate_identities]

- **CNOT and CNOT_INV.** With sdim's H (the Fourier gate):
  - CNOT = (1 ⊗ H^-1) CZ (1 ⊗ H) = (1 ⊗ H) CZ^dagger (1 ⊗ H^-1);
  - CNOT_INV = (1 ⊗ H^-1) CZ^dagger (1 ⊗ H) = (1 ⊗ H) CZ (1 ⊗ H^-1)
    **[VN, I1-I2]**.

  Every repo two-qutrit gate can therefore use the faster and better
  CZ^dagger (2.7%, 580 ns) **[LIT 1]**, plus basis changes on the target.
  Consecutive basis changes on a Z-check ancilla merge. My hardware models
  use 1.33 single-qutrit gates per CZ on average **[CONJ, modelling choice]**.
- **Logical gate costs.** P and Z are free. X needs a physical shift (or a
  Pauli-frame update, which would exclude it from what is benchmarked).
  SWAP is free (section 1.6). Logical H costs five Fourier gates. CZ(2,3) is
  one native gate.
- **SI1000 mapping.** Uniform two-qutrit depolarizing with probability p
  has process fidelity 1 - p **[VA; VN: I5]**. Cycle benchmarking's Weyl
  error rate is that same quantity, so today's best gate sits at SI1000
  p ≈ 0.027 (CZ^dagger) to 0.048 (CZ) **[VA + LIT 1]**.
- **Baseline at that p.** Read from the baseline CSVs (p index 11, 12):
  - p = 0.02: unif-1 LRB-D rejects 99.60% of shots at m = 2. const-0 keeps
    fidelity 0.9965 at m = 2 with 24.4% rejection, and 0.436 at m = 22.
  - p = 0.03: unif-1 rejects 99.96% at m = 2. const-0 gives 0.9915 at m = 2
    (33.7% rejection).

  **[VA]**

### 2.4 Flux-activated qutrit CZ and leakage

- IQM's qubit CZ is a 20-40 ns flux pulse on a transmon coupler. The coupler
  gives a tunable ZZ interaction up to 50 MHz that is fully off when idle
  **[LIT 24]**.
- For Google's flux CZ, the gate condition also puts |21> and |03> on
  resonance, which transports leakage between qubits **[LIT 17]**. For
  qutrits, |21> is a computational state. A flux-activated qutrit CZ
  therefore has a direct |21> → |03> leakage channel that must be detuned or
  calibrated away **[VA for the level argument; CONJ for its size]**.
- A qutrit CZ also needs four conditional phases (on |11>, |12>, |21>, |22>)
  rather than one **[VA]**.
- Static cross-Kerr between idle qutrits is tens to hundreds of kHz. The
  values in the Blok 2021 table run up to 748 kHz in magnitude **[LIT 3]**,
  and higher levels couple strongly **[LIT 9]**. On a 5-12 µs round this
  gives coherent phase errors unless the couplers null it **[CONJ]**.

## 3. Mid-circuit readout, reset and idle versus SI1000

### 3.1 Three- and four-state readout [LIT]

| source | levels | assignment | duration |
|---|---|---|---|
| Chen 2023 **[LIT 11]** | 3 | 96.9% (two-state 99.5%); shelving, two-tone, ML classifier, no quantum-limited amplifier, frequency multiplexed | 140 ns |
| Krinner 2022 **[LIT 12]** | 3 (qubits plus leakage) | mean assignment error 0.022(14) | 200/300 ns pulses, 400 ns integration |
| Blok 2021 **[LIT 3]** | 3 | average 0.96 ± 0.02; state 2: 0.92-0.96 | not stated |
| Vora 2026 **[LIT 14]** | 3 | averages 94.5-97.3%; state 2: 88.0-95.0% | 500-750 ns |
| Goss 2026 **[LIT 8]** | 3 | 93% average (IQM-fabricated chip) | 500 ns |
| Vigneau 2025 **[LIT 26]** | 3 (IQM star, qubits plus leakage) | > 99% / > 94% / > 93% for states 0 / 1 / 2 (one qubit lower) | readout 1 µs |
| Liu 2023 **[LIT 4]** | 4 | 99.10 / 94.46 / 94.50 / 91.11% | not stated |
| Wang 2025 **[LIT 9]** | 10 | 93.8% | not stated |

Errors are dominated by decay during readout. They shift the outcome down
by one (|2> to |1>, |1> to |0>), which is biased, not the uniform shift that
SI1000 assumes **[CONJ, consistent with the per-state numbers above]**.

### 3.2 Multiplexing and the repo's sequential readout

Multiplexing:

- Five qubits have been read on one line (80 ns pulse, 97% average) with
  per-resonator Purcell filters against off-resonant driving **[LIT 13]**.
- Garnet has 3 probe lines with 7, 7 and 6 readout structures; its control
  modules read up to ten qubits simultaneously **[LIT 24]**.
- IQM Spark reads all qubits in a multiplexed readout within 100 ns
  **[LIT 27]**.

Simultaneous ancilla readout is therefore the hardware default.

Effect of the repo's sequential schedule:

- The repo measures one ancilla per layer. Each data qutrit receives
  4 x 2p of measurement-layer idle plus 2p in the reset layer, 10p per round.
  One parallel layer gives 2p + 2p = 4p, so measurement-layer idle drops 4x
  and total idle 2.5x **[VA]**.
- Not-yet-used X-check ancillas also idle in |0> through earlier layers;
  SI1000 charges them 2p each time **[VA, read from
  `apply_si1000_noise_to_circuit`]**.

Results **[VN: sc_dem_hw]**:

- Sanity check: the sequential SI1000 numbers reproduce the production
  baseline (unif-1 acceptance 0.753 / 0.483 / 0.310 at p = 1e-3,
  m = 2 / 6 / 10, against 0.753 / 0.482 / 0.309; at p = 5e-3, m = 2:
  0.244 against 0.243).
- Parallel readout raises unif-1 acceptance:
  - p = 1e-3, m = 10: 0.31 → 0.47;
  - p = 5e-3, m = 2: 0.24 → 0.41.

  The raw (no rejection) fidelity at p = 5e-3, m = 2 rises from 0.56 to 0.72.

### 3.3 Spectator effects of mid-circuit readout

- On an IBM Falcon R8, a mid-circuit measurement caused a spectator error of
  1.7e-3 ± 1.0e-3. That is more than 3x an equal-length delay, with readout
  approaching 1 µs **[LIT 15]**.
- Readout photons can drive transitions out of the computational subspace at
  specific photon numbers **[LIT 16]**.
- In a Sycamore bit-flip code, measure qubits built up more |2> population
  than data qubits, which marks readout as a significant leakage source
  **[LIT 17]**.

For qutrit spectators, the |2> level has a different dispersive shift and a
shorter T2. I expect measurement-induced dephasing of qutrit superpositions
to exceed these qubit numbers **[CONJ]**. It must be measured (section 7.0).

### 3.4 Reset

- **Swap-based multi-level reset.** 250 ns, ground-state fidelity above 99%
  from any of the first three excited states (so it also removes |3>),
  error about 1e-3 **[LIT 17]**.
- **All-microwave f0g1 reset.** Unconditional, under 500 ns, 0.2% residual
  excitation **[LIT 18]**.
- **IQM pulse library** **[LIT 31]**:
  - Reset_Wait (relaxation), Reset_F0G1, Reset_F0G1_Composite ("a prx_12(pi)
    gate followed by the lru.f0g1 operation") and Reset_Conditional;
  - LRU_F0G1 drives |f0> ↔ |g1> to remove |2> population.
- **The composite on a qutrit.** It maps |0>, |1>, |2> to |0>, |0>, |1>: the
  |2> population ends in |1>. A qutrit reset needs f0g1, then prx_12(pi), then
  f0g1, which maps all three levels to |0> **[VN: sc_gate_identities I6,
  level bookkeeping]**.
- **Client-level reset.** IQM's client reset is "a (projective) measurement
  followed by a classically controlled X gate" **[LIT 29]**. A binary result
  plus a conditional 0-1 pi pulse cannot return |2> to |0> **[VA]**.
- **No reset.** Krinner 2022 and Vigneau 2025 (on an IQM star QPU) leave
  ancillas unreset and build detectors from consecutive outcomes
  **[LIT 12, 26]**. For qutrits the detector is m_t - m_(t-1) mod 3 **[VA]**.
  An unreset ancilla in |s ≠ 0> can decay before the next readout, with
  probability about t_round/T1 (6-26% for 5-12 µs rounds and T1 of 47-78 µs).
  Unconditional reset at 0.1-0.2% is preferable whenever syndromes are often
  nonzero **[VA, estimate]**.

### 3.5 Idle model and SI1000 per location [VN: sc_idle]

Model:

- Amplitude damping 1→0 and 2→1 (and 2→0 if measured), with pure dephasing
  fitted to the measured T2 of the 01, 12 and 02 coherences.
- With no Hamiltonian part, the process fidelity has the closed form
  F_pro(t) = [1 + e^(-γ10 t) + e^(-γ2 t) + 2e^(-t/T2_01) + 2e^(-t/T2_12) + 2e^(-t/T2_02)] / 9.
- The closed form matches the matrix exponential to 1e-10, and the SI1000
  'd' location with equal process infidelity has probability e(t) = 1 - F_pro
  **[VN]**.
- Where T2_02 was not measured, I assume independent 01 and 12 frequency
  noise. For the Goss sets, where it was measured, this rule gives
  0.0060/µs against 0.0055/µs measured **[VA]**.

| coherence set | e(100 ns) | e(500 ns) | e(1 µs) | X-type share at 500 ns |
|---|---|---|---|---|
| Goss 2022 Q4, echo T2 **[LIT 1]** | 1.4e-3 | 6.9e-3 | 1.38e-2 | 0.82 |
| Goss 2022 Q4, Ramsey T2 | 3.0e-3 | 1.47e-2 | 2.90e-2 | 0.39 |
| Goss 2022 Q5, echo / Ramsey | 2.3e-3 / 5.4e-3 | 1.16e-2 / 2.67e-2 | 2.30e-2 / 5.23e-2 | 0.75 / 0.33 |
| Blok 2021 average, echo / Ramsey **[LIT 3]** | 2.4e-3 / 4.4e-3 | 1.21e-2 / 2.18e-2 | 2.40e-2 / 4.30e-2 | 0.64 / 0.35 |
| Liu 2023, Ramsey **[LIT 4]** | 1.8e-3 | 9.1e-3 | 1.81e-2 | 0.28 |
| IQM-like: T1 from IQM Spark **[LIT 27]**, T2_01 = 35 µs from the IQM star model **[LIT 26]**, T2_12 = 17 µs assumed **[CONJ]** | 4.1e-3 | 2.03e-2 | 4.01e-2 | 0.53 |

Comparison at the gate-matched p = 0.027 **[VA]**:

- **Readout.** SI1000's 5p = 13.5% is 2-6x the measured three-state error.
- **Readout-layer idle.** SI1000's 2p = 5.4% is 2-8x the 0.7-2.7% of a
  500 ns layer at Berkeley-class coherence, and 2.7x the IQM-like 2.0%.
- **Gate-layer idle.** SI1000 has none. In the duration-aware model, data
  qutrits spend most of each 6-12 µs LRB step waiting for other qutrits' CZs.
  At 1.4%/µs (echo) to 2.9%/µs (Ramsey), this is the largest idle term, and
  it is missing from SI1000 **[VN]**.
- **Bias.** SI1000's uniform Weyl noise is unbiased. The twirled idle channel
  is dominated by X-type (decay) errors for echo-limited qutrits and by Z-type
  errors for Ramsey-limited ones **[VN]**.

### 3.6 Duration-aware hardware models on the repo DEM [VN: sc_dem_hw]

Method:

- The repo generator builds each circuit with SI1000 placeholders. Every noise
  location is relabelled by context, and an ASAP schedule assigns times:
  - measurement and reset layers block all qutrits;
  - an idle 'd' location is inserted for every gap of an active qutrit
    (data always; ancillas from their first gate after reset until measured);
  - SWAP is virtual, P and Z are free.
- Terminal readout and encoding stay ideal, as in the repo, so the readout
  error at the end only changes the SPAM prefactor (see below).

Models:

- **HW-now.** CZ^dagger 2.7% in 580 ns **[LIT 1]**. CNOT 3.48% in 713 ns,
  including 1.33 basis changes. H and X gates 6e-3 in 100 ns **[LIT 2, 5]**.
  Readout flip 3% in 500 ns **[LIT 8, 12, 14]**. Reset 0.2% in 300 ns
  **[LIT 17, 18]**. Coherence Goss Q4 echo. HW-now-R uses Ramsey instead.
- **HW-target [CONJ, what-if].** CZ 1% in 200 ns; 1q 1e-3 in 50 ns;
  readout 1% in 300 ns; reset 0.1% in 250 ns; coherence 2x Goss Q4 echo.
- **HW-qubit [CONJ, what-if].** As HW-target but CZ 0.5%, 1q 5e-4,
  readout 0.5%. The CZ error matches IQM's qubit median of 5e-3 **[LIT 24]**.

LRB step duration (forward Clifford, check round and reset):

| schedule | HW-now | HW-target |
|---|---|---|
| seq9 (repo sequential) | 12.0 µs | 5.1 µs |
| par9 (parallel readout) | 5.9 µs | 2.5 µs |
| relay11 (square lattice) | 10.8 µs | 4.3 µs |

Results: mean over 6 random sequences, 15,000 shots each. Entries are
unif-1 acceptance / accepted fidelity, and raw is the exact no-rejection
fidelity. const-0 uses its terminal X-checks only.

| model | m | seq9 unif-1 | par9 unif-1 | relay11 unif-1 | raw seq9 / par9 / relay11 | const-0 acceptance / fidelity (raw) |
|---|---|---|---|---|---|---|
| SI1000 1e-3 | 2 | 0.753 / 0.9928 | 0.839 / 0.9947 | 0.778 / 0.9936 | 0.890 / 0.938 / 0.925 | 0.987 / 1.0000 (0.986) |
| SI1000 1e-3 | 10 | 0.310 / 0.9921 | 0.467 / 0.9949 | 0.347 / 0.9945 | 0.626 / 0.756 / 0.714 | 0.923 / 0.9998 (0.920) |
| SI1000 5e-3 | 2 | 0.244 / 0.960 | 0.413 / 0.972 | 0.282 / 0.968 | 0.559 / 0.725 / 0.676 | 0.934 / 0.9998 (0.933) |
| HW-now | 2 | 0.029 / 0.82 | 0.067 / 0.87 | 0.005 / 0.73 | 0.121 / 0.244 / 0.083 | 0.759 / 0.996 (0.757) |
| HW-now | 6 | 1e-4 / n.a. | 9e-4 / n.a. | 0 | 0.005 / 0.023 / 0.002 | 0.395 / 0.939 (0.370) |
| HW-now | 10 | 0 | 0 | 0 | ≤ 0.002 | 0.239 / 0.811 (0.183) |
| HW-now-R | 2 | 0.005 / 0.66 | 0.029 / 0.78 | 6e-4 / 0.77 | 0.035 / 0.153 / 0.025 | 0.757 / 0.996 (0.753) |
| HW-target | 2 | 0.382 / 0.966 | 0.467 / 0.974 | 0.236 / 0.949 | 0.575 / 0.671 / 0.519 | 0.929 / 0.9999 (0.926) |
| HW-target | 6 | 0.081 / 0.967 | 0.132 / 0.959 | 0.025 / 0.951 | 0.244 / 0.351 / 0.197 | 0.771 / 0.9976 (0.763) |
| HW-target | 10 | 0.018 / 0.966 | 0.037 / 0.953 | 0.003 / 0.93 | 0.105 / 0.184 / 0.077 | 0.644 / 0.9935 (0.624) |
| HW-qubit | 10 | 0.058 / 0.969 | 0.127 / 0.954 | 0.022 / 0.96 | 0.202 / 0.355 / 0.182 | 0.794 / 0.9983 (0.788) |

Per-step decay estimates, f = (y10/y2)^(1/8) and r = 2(1-f)/3
**[VN: sc_budget]**:

- **Raw.** f = 0.85 (par9), 0.81 (seq9) and 0.79 (relay11) at HW-target;
  0.95 for const-0. The raw logical r per step, 0.03 (const-0) to 0.14
  (relay11), is one to two orders of magnitude above a physical qutrit's
  r ≈ 2 r_C per step (2 Cliffords). That is 0.002 at the HW-target
  single-qutrit error and 0.005-0.011 with today's r_C **[LIT 2, 4]**. One
  logical Clifford uses about ten physical gates, including a CZ **[VA]**.
- **const-0 LRB-D at HW-now.** r = 0.017 per step with 13.5% rejection per
  step: worse than physical qutrits.
- **const-0 LRB-D at HW-target.** r = 5e-4 per step with 4.5% rejection per
  step: better than a physical qutrit with r_C = 1e-3 **[CONJ for the
  physical reference]**.

Conclusions:

- **HW-now.** Mid-circuit checks after every Clifford destroy the signal.
  Three to six faults are expected per shot already at m = 2 (first-order
  table, section 7.4). Neither postselection nor any decoder can recover them
  **[VN]**. The baseline at p = 0.02-0.03 shows the same (section 2.3).
- **HW-target.** Parallel readout on the 9-qutrit coupling graph is the best
  schedule. The square-lattice relay layout is the worst: at HW-target its
  12 extra CNOTs per round cost more than the sequential schedule's three
  extra readout layers **[VN]**. This favours a custom planar chip
  (section 1.7) over Garnet-type layouts for Stage C **[CONJ, recommendation]**.
- **const-0** has the highest acceptance and fidelity in every model
  **[VN]**. The repo's policies only *select* rounds from circuits that execute
  all of them, so in simulation unused rounds still add noise. On hardware,
  unexecuted rounds cost nothing **[VA]**, so I expect const-c with small c
  to perform between const-0 and the all-rounds numbers above **[CONJ]**.
- **Terminal readout on hardware.** It is noisy. For const-0, every single
  readout flip changes at least one terminal X-check, because each data
  qutrit is in some X stabilizer. At first order this lowers acceptance by
  (1 - e)^5 (0.86 at e = 3%) without lowering the accepted fidelity
  **[VA]**. For ordinary circuits that read only data 0 and 3, it multiplies
  the fidelity by (1 - 3e/2)^2 (0.91 at 3%). Both are SPAM-like (independent
  of m), so they change the LRB prefactor b, not f **[VA]**.

## 4. Leakage to |3> and above

**Mechanisms.**

- Drives on the 1-2 transition are detuned from 2-3 by roughly one
  anharmonicity, the analogue of qubit 0-1 drives leaking to |2>. Morvan
  2021 observed such off-resonant leakage for qubit-like RB **[LIT 2]**.
  IQM shapes its pulses against leakage to higher states **[LIT 24]**;
  HD DRAG reaches leakage below 3e-5 for a 6.25 ns qubit pi/2 gate
  **[LIT 28]**.
- Flux-CZ resonances, such as |21> ↔ |03>, which involve a computational
  state for qutrits (section 2.4) **[LIT 17; VA]**.
- Readout-induced transitions **[LIT 16, 17]**.

**Rates found.**

- No |3>-leakage rate per qutrit gate appears in the fetched sources. The
  two-qutrit gate papers restrict their analysis to {0,1,2} or state only
  that ramps "limit leakage" **[LIT 1]**.
- Qubit analogues:
  - Sycamore bit-flip code: γ↑ = 0.09% (data) and 0.11% (measure) per round;
    steady-state |2> population 0.97% and 1.30%. Multi-level reset of the
    measure qubits lowers these to 0.50% and 0.03%, and speeds up data-qubit
    leakage decay 2.4x by transport **[LIT 17]**.
  - ETH, three-state readout: rejected fractions 0.0017(2) per data qubit per
    cycle and 0.0094(4) per auxiliary qubit per cycle, 8% of runs per cycle
    for 17 qubits **[LIT 12]**.
- Higher levels:
  - |3> lifetime 73 µs (best single qudit) **[LIT 4]**;
  - decay is mainly sequential, above 20 µs per transition **[LIT 10]**;
  - high-E_J/E_C devices show 12 levels **[LIT 9]**.

  A leaked qutrit therefore stays leaked for several rounds **[CONJ]**.

**Detection as erasure.**

- **Ancillas.** Read four states in every mid-circuit readout: |3> assignment
  91.1% in Liu 2023, and 10-state readout 93.8% in Wang 2025 **[LIT 4, 9]**.
  A |3> outcome flags the round. Then reset with the swap-based reset, which
  clears |3> **[LIT 17]**. The f0g1 units clear only |2>, and do so
  unconditionally **[LIT 31; VA]**.
- **Data.** Four-state readout at the end flags data leakage. ETH did the
  three-state equivalent for qubits **[LIT 12]**.
- **Related work.** Repeated mid-circuit erasure checks on transmon qutrit
  levels, with postselection, extend a g-f logical qubit's T1 beyond 500 µs
  **[LIT 22]**. This shows that per-round heralding of the dominant qutrit
  error is practical.

**Policy.** Feed the flags to the decoder as erasures, or reject flagged
shots (hybrid decoder, section 7.4).

Acceptance estimate:

- With ETH-like flag rates (0.17% per data qutrit and 0.94% per ancilla per
  round), the parallel 9-qutrit schedule loses about 4.6% per round, and the
  11-qutrit relay layout (6 measured qutrits) about 6.5% **[VA arithmetic;
  CONJ that qutrit |3> rates are similar]**.
- At m = 10 (11 rounds), flag-only rejection keeps about 0.6 and 0.48 of
  shots. That is far more than LRB-D keeps at HW-target (0.04 and 0.003)
  **[VA + VN]**.

## 5. Offline Pauli frame versus real-time feedback

**Offline is exact for Clifford LRB [VA].**

- Every gate is a qutrit Clifford. A Weyl correction inferred at any time
  propagates through the rest of the circuit to a known Weyl operator at the
  terminal X-basis readout.
- Only its commutation with X_L matters, so the decoder outputs a shift Δ in
  Z_3, and the outcome k becomes k - Δ before the fidelity |Σ_k ω^k P(k)| is
  computed.
- Known relay states (section 1.5), unreset-ancilla offsets (section 3.4)
  and virtual SWAPs (section 1.6) are also classical relabellings.
- No feed-forward is needed and decoding latency is irrelevant. ETH applied
  its distance-3 surface-code corrections in postprocessing in the same way
  **[LIT 12]**.

**What needs feedback.**

- **Conditional (active) reset.** Not needed: unconditional swap or f0g1
  resets exist **[LIT 17, 18, 31]**. A conditional reset also makes the data
  wait for the decision and the conditional pulses, so it saves little or no
  time against a 250-500 ns unconditional reset **[VA, from the latency
  table below]**.
- **Non-Clifford logical gates** by magic-state injection need a conditional
  Clifford correction before the next non-Clifford gate. This is outside
  Clifford LRB **[VA]**.
- **Leakage-conditioned operations.** Not needed: apply the LRU or reset
  every round unconditionally, as McEwen 2021 did for its measure qubits
  **[LIT 17]**.

**Latency budget if feedback were used.**

| element | value |
|---|---|
| three-state readout | 140-750 ns **[LIT 8, 11, 14]** |
| FPGA state discrimination | 40 ns (66 ns with softmax), qutrit-capable **[LIT 14]**; 110 ns trigger latency **[LIT 19]** |
| decoding | mean below 1 µs per round on FPGA, 9.6 µs response for 9 rounds **[LIT 20]**; 63 µs average real-time decoder latency at d = 5, 1.1 µs cycle **[LIT 21]** |
| this code | a lookup table over w rounds has 3^(4w) entries (6,561 for w = 2; 3^(6w) with relays) **[VA]** |

A data qutrit waiting for the loop loses 1.4%/µs (echo), 2.9%/µs (Ramsey) or
4.0%/µs (IQM-like assumption) **[VN: sc_idle]**. A 1 µs loop therefore costs
about as much as today's best two-qutrit gate (2.7%) **[VA]**.

Qutrit feed-forward has been demonstrated once **[LIT 14]**:

- mid-circuit measurement of a control qutrit followed by X^m on a target
  for m ∈ {0,1,2};
- raw branch fidelities 98.9-99.0% for m = 0 and 83.6-88.1% for the m = 2
  branches.

IQM's client conditions only x, y, rx, ry and r gates, on one classical bit:
cc_prx applies PRX if the result is 1 and a duration-matched identity
otherwise **[LIT 29, 30]**. Ternary-conditioned qutrit operations would need
custom firmware **[CONJ]**.

## 6. What IQM hardware supports (sourced facts only)

- **Garnet** **[LIT 24]**:
  - 20 qubits in a square lattice rotated by 45 degrees, connected by
    tunable couplers; qubits and couplers are flux-tunable transmons.
  - QPUs with 5, 20, 54 and 150 qubits use this topology.
  - CZ by a coupler flux pulse, typically 20-40 ns.
  - Median errors: single-qubit 9e-4, CZ 5e-3, two-qubit Clifford 2e-2,
    simultaneous readout 3e-2.
  - 3 probe lines with 7, 7 and 6 readout structures; a TWPA; FPGA real-time
    control; readout modules for simultaneous readout of up to ten qubits.
  - T1 of order 40 µs, with more than 100 µs shown in test devices.
- **Fake Garnet backend** **[LIT 34]**: the 30-pair coupling map used in
  section 1; PRX 20 and CZ 40 (no units in the file); T1 11555-60725 and
  T2 8270-11930 (ns if read with the paper's T1 scale); readout errors
  2.2-2.9%.
- **Best IQM qubit device**: CZ 99.93% (40 h average), single-qubit 99.98%
  and readout above 99.94% in one device **[LIT 25]**.
- **Emerald (54 qubits, Crystal 54)** on Amazon Braket: square lattice with
  tunable couplers; native X/Y rotations and CZ; medians 99.93%
  (single-qubit) and 99.5% (two-qubit); supports Braket's experimental
  dynamic circuits **[LIT 37]**.
- **Dynamic circuits on Garnet** **[LIT 36]**:
  - mid-circuit measurement and feed-forward, experimental;
  - "conditional operations can only be applied to qubits in a local group";
  - verbatim compilation only; no latency numbers given.
- **IQM client 36.0.0** **[LIT 29, 30]**:
  - mid-circuit measurement with feedback keys;
  - cc_prx semantics as in section 5;
  - reset as measurement plus classically controlled X, inheriting cc_prx's
    calibration;
  - if_test cannot be nested;
  - heralding mode ZEROS;
  - an example backend with native set ['id', 'r', 'cz', 'measure'];
  - MOVE gates only on the Star (computational-resonator) architecture.
- **iqm-pulse 15.0.0** **[LIT 31]**:
  - acquisitions: ComplexIntegration (weighted integration, "resulting in a
    complex number"), TimeTrace (raw IQ) and ThresholdStateDiscrimination
    (boolean);
  - measure implementations with an `acquisition_type` setting (default
    'threshold') and integration weights and threshold;
  - Shelved_Measure_Constant (a measure "surrounded by two prx_12 gates"),
    LRU_F0G1, the reset family of section 3.4, and a ConditionalInstruction
    ("Choice between multiple Instructions, depending on a condition", with
    the condition type unspecified).
- **Pulla 15.0.0** **[LIT 32, 33]**: custom gates, composite gates and custom
  waveforms (examples use qubit pulses only); available through IQM Resonance
  and on-premises (blog of February 2025, updated April 2026).
- **Higher levels on IQM hardware.**
  - IQM Spark prototype (5-qubit star) **[LIT 27]**: preparation and readout
    of the second excited state through direct hardware access, with
    Γ10^-1 = 44.4 µs, Γ21^-1 = 35.0 µs, Γ20^-1 = 69.2 µs, and multiplexed
    readout within 100 ns.
  - IQM-designed, -fabricated and -packaged 20-transmon, 30-coupler chip
    **[LIT 8]**: operated as up to 15 qutrits with LBNL's QubiC electronics.
    It used 180 ns SU(3) gates, coupler flux-pulse cross-Kerr, 93% qutrit
    readout in 500 ns, and no mid-circuit measurement.
- **IQM quantum error detection** **[LIT 26]**:
  - [[4,2,2]] on a 6-qubit star QPU, with repeated mid-circuit measurement
    and no ancilla reset;
  - a three-state classifier used to flag leakage;
  - postselection kept about 67-71% of runs per cycle; cycle 2.00 µs,
    readout 1 µs, CZ 80 ns; qubit T1 40 µs, T2* 35 µs.
- **Not found in any fetched IQM source**: qutrit gates in the circuit-level
  API, multi-state discrimination in the threshold acquisition, ternary
  classical control, qutrit CZ calibrations, or an IQM-run qutrit
  computation.
- **Implications [CONJ]**:
  - A qutrit LRB on IQM hardware needs Pulla (or on-premises access) to
    calibrate prx_12-type 1-2 pulses into full single-qutrit gates and to
    calibrate four-phase qutrit CZs on the couplers.
  - Three- or four-state classification should be done offline from
    ComplexIntegration data. Offline decoding makes this sufficient, if
    ComplexIntegration is exposed for mid-circuit measurements. The pages
    I fetched do not say whether it is.
- **Repo history [VA, read with `git show 1bd97c40`]**:
  - The old pipeline, in `external/iqm_qpu_circuits/[[5,1,2]]_3 Qutrit
    Surface Code/X Stabilizer Terminal Check Circuits/`, generated only
    noiseless const-0 circuits: 30 sequences, depths 0-22, seed 12345, and a
    terminal X readout of all five data qutrits (S1 = x0 + x1 - x2,
    S2 = -x1 + x3 - x4, logical x0 + x3).
  - It converted them to Cirq on `LineQid(dimension=3)` with sdim's
    generalized gate matrices, N1/N2 as identities and `cirq.reset`.
  - It kept the unitary encoder and the physical SWAP(2,3). It had no
    mid-circuit checks and no decomposition into IQM-native operations.

## 7. Step-by-step experimental protocol

### 7.0 Calibrations (per qutrit and per used coupler)

1. **Single-qutrit gates.**
   - Calibrate 0-1 and 1-2 pi and pi/2 pulses, with DRAG-type shaping
     against 2-3 leakage **[LIT 28]**, plus two virtual phase frames.
   - Run qutrit RB and simultaneous RB **[LIT 2]**.
   - Use qutrit noise tailoring so that the Weyl-channel assumption behind any
     DEM holds on average. Tailoring and mitigating Markovian noise on a
     transmon qutrit processor improved results up to 3x **[LIT 23]**.
2. **Readout.**
   - Build four-state classifiers on integrated IQ data and measure 4x4
     assignment matrices under simultaneous readout.
   - Check QND-ness by repeated readout.
   - Run a qutrit version of the mid-circuit-measurement RB suite **[LIT 15]**
     to measure the spectator error per readout layer.
3. **Reset.** Implement f0g1, prx_12(pi), f0g1, or a swap-based multi-level
   reset. Verify the residual |1>, |2>, |3> populations with four-state
   readout. Targets are 1e-3 **[LIT 17]** and 0.2% **[LIT 18]**.
4. **Two-qutrit gates.**
   - Calibrate CZ^dagger (and CZ) with four conditional phases on each used
     coupler; verify with cycle benchmarking **[LIT 1]**.
   - Measure |3> leakage per gate with four-state readout.
   - Map and null idle cross-Kerr between all used pairs **[LIT 3, 9, 24]**.
5. **Coherence.** Measure T1 of |1>, |2>, |3> and the echo and Ramsey T2 of
   the 01, 12 and 02 coherences. Feed them to `sc_idle.py` and decide on
   dynamical decoupling for idling data: echo-limited idle costs half as much
   as Ramsey-limited idle **[VN]**.

### 7.1 Stage A: const-0 LRB on five data qutrits (feasible now)

- **Circuits.** Use the repo's const-0 circuits with the SWAP made virtual.
  Encoding is the unitary encoder: couplings 0-1, 0-3, 1-2, 2-4 plus the fold
  2-3. This graph is bipartite and has 128 placements on Garnet's coupling
  map, for example data 0, 1, 3, 2, 4 on QB1, QB2, QB4, QB5, QB10
  **[VN: sc_graphs G4]**.
- No mid-circuit measurement, reset or feedback is needed. The terminal
  readout is of all five data qutrits in the X basis. This is what the old
  IQM export prepared **[VA, repo history]**. It is also the operating mode
  (terminal readout only, no mid-circuit measurement) already used for a
  different experiment on the IQM-fabricated qutrit chip **[LIT 8]**.
- **Expected numbers [VN: sc_dem_hw].**
  - HW-now: accepted fidelity 0.996 / 0.94 / 0.81 at m = 2 / 6 / 10, with
    acceptance 0.76 / 0.39 / 0.24 (before the readout factor (1 - e)^5).
  - HW-target: 0.9999 / 0.9976 / 0.9935, with acceptance 0.93 / 0.77 / 0.64.
- **Decoders.** Run LRB-D on the two terminal X-checks, plus MLD on the
  9-valued terminal syndrome. Here 79-83% of single faults are in ambiguous
  classes **[VN: first-order table]**, so correction is weak and the hybrid
  equals LRB-D for most syndromes.
- **Reference.** Run single-qutrit RB on each data qutrit (the p_RB
  reference).

### 7.2 Stage B: check rounds and learning the detector error model

- **Encoding by measurement.**
  - Prepare |+>^5 (H on each data qutrit). This is a +1 eigenstate of both
    X stabilizers and of X_L.
  - Run one check round. The random Z-syndrome is fixed by a known X-type
    frame correction, which commutes with X_L **[VA]**.
  - The unitary encoder (and its triangle with S_Z1, section 1.1) is then
    not needed.
- **Circuits.** Memory runs with r = 1 to 10 rounds and no logical gates,
  plus runs with only logical Paulis (frame updates). Use the chosen
  schedule (par9 on a custom chip, relay11 on a square lattice).
- **Learning.**
  - For independent Weyl mechanisms with uniform nonzero vectors,
    log E[ω^(a·d)] = Σ_mechanisms log(1 - q_e (n_e + 1)/n_e) over the
    mechanisms whose response to the detector combination a is nonzero
    **[VA; this is the product formula in `exact_raw`]**.
  - Measuring these characteristic functions for low-weight a gives linear
    equations for the mechanism parameters. This is a Z_3 generalization of
    the pair-correlation (p_ij) analysis **[LIT 39]** **[CONJ, method sketch;
    identifiability not checked]**.
  - Related methods: decoding weights informed by measured syndrome
    correlations **[LIT 38]**; logical-noise learning from syndrome data
    **[LIT 40]**; optimizing decoder priors directly on the logical error
    rate **[LIT 41]**.
- **Biased noise.** Learn asymmetric (biased) shift distributions for
  readout and idle. Decay makes outcome shifts of -1 more likely than +1
  (section 3.1), and SI1000 has no such term **[CONJ]**.
- **Shots.** A mechanism of probability q needs about 100/q shots for 10%
  relative precision (1e4 at q = 1e-2, 1e5 at 1e-3) **[VA, Poisson
  counting]**.

### 7.3 Stage C: LRB with checks

- **Gate requirement.** Run Stage C only when the two-qutrit error is at or
  below about 1% **[CONJ, recommendation]**. At HW-now every check schedule
  leaves less than 7% of shots at m = 2 and a raw fidelity below 0.25, while
  HW-target leaves 24-47% **[VN]**.
- **Dataset.** Execute the unif-1 circuits (a check after every forward
  Clifford). Every repo policy (unif k, const c ≥ 1) is a selection of rounds
  from these circuits **[VA, brief]**. One dataset therefore serves all
  policies and all decoders, and matches the repo simulations.
- **Sparse policies on hardware.** Additionally execute circuits for the
  best one or two sparse policies with only their rounds. On hardware the
  unexecuted rounds add no noise (section 3.6).
- **Terminal readout.** Read all five data qutrits in the X basis rather than
  only 0 and 3.
  - In the repo circuits, data faults after the last check are invisible.
    They make up 1.5-4% of all single faults and set the m = 0 floor; for
    example, 0.017 x 0.29 = 0.5% logical error per shot at SI1000 p = 1e-3
    gives the baseline's 0.9925 **[VN: first-order table; baseline]**.
  - With all five data read, every single such fault flips a terminal
    X-check, so it is detected **[VA]**.

### 7.4 Offline decoders to run on the same data

1. **LRB-D**: every unif k and const c policy, as in the repo.
2. **Maximum-likelihood correction** on (a) the SI1000 DEM, (b) the
   duration-aware hardware DEM (`sc_dem_hw.py` style, with measured
   parameters) and (c) the learned DEM of Stage B. The output is Δ; all
   shots are kept. This is the variant closest to logical randomized
   benchmarking as proposed for error-correction implementations, which
   benchmarks the decoded logical channel **[LIT 42]**.
3. **Hybrid**: correct, then reject only
   - leakage-flagged shots (|3> on any ancilla or data qutrit), and
   - shots whose posterior maximum is below a threshold.

   Sweep the threshold to trace the acceptance-fidelity curve.
4. **Empirical lookup decoder** (cross-validated). In LRB the ideal outcome
   of every shot is 0, so each shot's logical shift is observed. A
   syndrome → most-frequent-shift table learned on half the shots and applied
   to the other half is an honest decoder for small m **[VA]**.
   - At m = 2 (2e5 training and 1e5 test shots, 3 sequences), 83-98% of
     test syndromes were seen in training **[VN: sc_dem_hw]**.
   - If this decoder beats DEM-based MLD, the DEM is miscalibrated.

Empirical lookup decoder at m = 2 **[VN: sc_dem_hw]** (fidelity, with
acceptance where shots are rejected):

| schedule, model | raw | corrected (all shots) | hybrid (confidence ≥ 0.6) | LRB-D unif-1 |
|---|---|---|---|---|
| par9, HW-target | 0.664 | 0.770 | 0.849 at 0.881 | 0.972 at 0.459 |
| par9, HW-qubit | 0.778 | 0.861 | 0.903 at 0.939 | 0.977 at 0.620 |
| seq9, SI1000 5e-3 | 0.556 | 0.638 | 0.766 at 0.768 | 0.958 at 0.243 |
| relay11, HW-target | 0.513 | 0.591 | 0.760 at 0.702 | 0.950 at 0.229 |

First-order fault classes at m = 2 **[VN: sc_dem_hw]**, sequence 0. "Faults"
here are single faults that are detected or change the logical outcome
(harmless undetected faults are excluded), weighted by probability. Fractions
are of these faults, with ranges over the six noise models. 'Ambiguous' means
the fault's full syndrome history is shared by faults with a different
logical shift.

| schedule | expected faults per shot (SI1000 1e-3 / HW-target / HW-now) | silent logical | ambiguous | MLD first-order failure |
|---|---|---|---|---|
| seq9 | 0.29 / 1.04 / 3.8 | 0.017-0.029 | 0.45-0.56 | 0.11-0.16 |
| par9 | 0.18 / 0.84 / 3.0 | 0.019-0.041 | 0.40-0.50 | 0.09-0.12 |
| relay11 | 0.26 / 1.55 / 5.7 | 0.015-0.031 | 0.41-0.44 | 0.08-0.12 |
| const-0 | 0.017 / 0.11 / 0.38 | 0 | 0.79-0.83 | 0.11-0.21 |

These numbers explain why correction is weak on this code **[VA from the
table]**. At low noise, correction leaves a logical error of about (faults
per shot) x (silent + MLD-failure fraction). LRB-D removes the detected
faults at the cost of acceptance.

### 7.5 Validation checks

- **V1 Stationarity.** The detection-event rate per detector is stationary in
  the round index and matches the DEM.
- **V2 Z_3 symmetry.** Under twirled noise, P(det = 1) = P(det = 2).
  Asymmetry reveals decay-driven bias, which a biased DEM should absorb
  **[CONJ]**.
- **V3 Hold-out.** Learn the DEM on half the shots. Predict detector
  correlations and decoded logical errors on the other half, and compare
  with the empirical lookup decoder.
- **V4 SPAM floor.** Check the m = 0 floor against the first-order prediction
  (silent faults plus terminal readout factor). Compare the LRB decay with
  physical qutrit RB at matching error.
- **V5 Gate dependence.** Non-exponential decay, or a sequence spread above
  the DEM's prediction (raw sd 0.009-0.036), indicates gate-dependent or
  non-Markovian noise (drift, two-level-system fluctuations, leakage).
  Correlate multi-round detection bursts with leakage flags; ETH saw
  correlations over up to eight cycles **[LIT 38]**.
- **V6 Drift.** Interleave sequences and depths in time, and repeat one
  reference circuit in every block.
- **V7 Relay cleanliness.** R1 and R2 must read 0 at the DEM-predicted rate.

### 7.6 Shot and time budget [VN: sc_budget]

**Variance components.** A single shot contributes Re ω^k with variance
v(F) = (1 + F)/2 - F^2 **[VA]**. With K sequences and N accepted shots each,
the SE of the sequence average is sqrt(s_seq^2/K + v/(K N)).

- The exact raw fidelity varies across sequences with sd s_seq = 0.009-0.036
  (6 sequences, HW-target and HW-qubit) **[VN]**. Different Cliffords have
  different numbers of physical gates (each logical H is five Fourier gates,
  each P one CZ), so this is gate-dependent noise inherent to the logical
  gate set **[VA]**.
- Shot noise equals the sequence spread at N_eq = v/s_seq^2 = 150-6,500
  shots per sequence. Beyond a few N_eq, extra shots per sequence are wasted
  **[VN]**.
- For example, for relay11 at HW-target with K·N = 3e5 per depth, the SE of
  the raw-fidelity mean is 0.0048 (K = 30, N = 1e4), 0.0028 (K = 100,
  N = 3,000) and 0.0019 (K = 300, N = 1,000).
- Postselected fidelities at low acceptance are shot-noise limited. par9 at
  HW-target, m = 10, has a = 3.7% and SE 0.0025 for 3e5 shots.

**Recommendation.** K = 100 sequences x N = 3,000 shots per depth, 9 depths:
2.7e6 shots, or 0.07 / 0.22 / 0.75 h at 0.1 / 0.3 / 1 ms per shot.

- An m = 22 circuit lasts at most 0.11-0.47 ms (2 x 22 steps of 2.5-10.8 µs),
  so the repetition time is set by circuit length plus reset **[VN]**.
- Cloud overheads for 900 distinct circuits are unknown **[CONJ]**.
- Stage A with N = 1,000 needs 9e5 shots. Stage B needs 1e4-1e5 shots per
  memory configuration.

## 8. Open issues and limitations

- **What the hardware models omit.** Leakage, crosstalk, measurement-induced
  dephasing, coherent errors and drift. The terminal readout and encoding stay
  ideal, as in the repo. The models use uniform Weyl noise per location, with
  process infidelity matched to measured or assumed rates **[CONJ: optimistic
  for omitted terms; possibly pessimistic for bias]**.
- **Schedules.** ASAP in the given gate order, not optimized. A better CZ
  ordering could shorten par9 rounds towards 4 CZ layers **[CONJ]**.
- **Statistics.** The decoder and first-order results use 3 sequences and one
  sequence respectively. They are indicative, not converged.
- **Upstream decoder study.** The quantitative decoder comparison under SI1000
  belongs to the other agents' work. The numbers here only test how the
  conclusions shift with hardware-like noise.

## 9. Changes relative to the earlier draft of this note

The earlier draft (same file name, written by an H1 agent in the morning run)
was checked by a critic. I ran the four critic scripts
`notes/scripts/critic_H1-sc-hardware_{noise,faults,layout,dem}.py` in this
task and changed the note accordingly:

- **X-type relays.** Fan-out relays in a just-measured state turn a readout
  error into a data error. The schedule now uses fresh fan-out relays and
  one readout layer (section 1.5) **[VA]**.
- **Hybrid ambiguity.** The earlier code-capacity count (8 of 32 syndromes
  ambiguous) understated circuit-level ambiguity. At circuit level, 40-56%
  of single faults are ambiguous (section 7.4) **[VN]**.
- **Shot budget.** The earlier shot-noise-only budget (K = 30, N = 3e4)
  ignored a sequence spread that dominates the raw fidelity **[VN]** and, I
  expect, the corrected fidelity **[CONJ]**. It is replaced by K = 100,
  N = 3,000 (section 7.6).
- **Minimum-hop search.** It now uses radius 5, with a bound that makes it
  exhaustive (section 1.4) **[VA + VN]**. The minima (6 hops native, 5 at
  distance 2) are unchanged.
- **Hardware models.** They are now duration-aware, with gate-layer idle,
  basis-change gates, a virtual SWAP and fresh-ancilla handling, and they
  report echo versus Ramsey sensitivity. The earlier "HW-A" table is
  replaced by section 3.6.
- **Reset.** IQM's composite reset leaves a qutrit's |2> in |1> (new,
  section 3.4).

## References (all fetched in this task, 2026-10-28)

1. N. Goss, A. Morvan, B. Marinelli, B. K. Mitchell, L. B. Nguyen, R. K. Naik, L. Chen, C. Jünger, J. M. Kreikebaum, D. I. Santiago, J. J. Wallman, I. Siddiqi, "High-Fidelity Qutrit Entangling Gates for Superconducting Circuits", arXiv:2206.07216 (2022), DOI 10.1038/s41467-022-34851-z. https://arxiv.org/abs/2206.07216, https://arxiv.org/html/2206.07216v3
2. A. Morvan, V. V. Ramasesh, M. S. Blok, J. M. Kreikebaum, K. O'Brien, L. Chen, B. K. Mitchell, R. K. Naik, D. I. Santiago, I. Siddiqi, "Qutrit randomized benchmarking", Phys. Rev. Lett. 126, 210504 (2021), DOI 10.1103/PhysRevLett.126.210504, arXiv:2008.09134. https://arxiv.org/abs/2008.09134, https://ar5iv.labs.arxiv.org/html/2008.09134, https://arxiv.org/pdf/2008.09134 (text extracted locally)
3. M. S. Blok, V. V. Ramasesh, T. Schuster, K. O'Brien, J. M. Kreikebaum, D. Dahlen, A. Morvan, B. Yoshida, N. Y. Yao, I. Siddiqi, "Quantum Information Scrambling in a Superconducting Qutrit Processor", Phys. Rev. X 11, 021010 (2021), DOI 10.1103/PhysRevX.11.021010, arXiv:2003.03307. https://arxiv.org/abs/2003.03307, https://arxiv.org/html/2003.03307
4. P. Liu, R. Wang, J.-N. Zhang, Y. Zhang, X. Cai, H. Xu, Z. Li, J. Han, X. Li, G. Xue, W. Liu, L. You, Y. Jin, H. Yu, "Performing SU(d) operations and rudimentary algorithms in a superconducting transmon qudit for d=3 and d=4", Phys. Rev. X 13, 021028 (2023), DOI 10.1103/PhysRevX.13.021028, arXiv:2304.11841. https://arxiv.org/abs/2304.11841, https://ar5iv.labs.arxiv.org/html/2304.11841
5. X.-M. Yu, X. Deng, W. Zheng, et al. (18 authors), "Efficient Implementation of a Single-Qutrit Gate Set via Coherent Control", Phys. Rev. Lett. 136, 230803 (2026), DOI 10.1103/vwzp-szjp, arXiv:2507.06860. https://arxiv.org/abs/2507.06860
6. K. Luo, W. Huang, Z. Tao, et al. (15 authors), "Experimental Realization of Two Qutrits Gate with Tunable Coupling in Superconducting Circuits", arXiv:2206.11199 (2022; page lists DOI 10.1103/PhysRevLett.130.030603). https://arxiv.org/abs/2206.11199
7. M. Subramanian, A. Lupascu, "Efficient two-qutrit gates in superconducting circuits using parametric coupling", Phys. Rev. A 108, 062616 (2023), DOI 10.1103/PhysRevA.108.062616, arXiv:2309.05766. https://arxiv.org/abs/2309.05766
8. N. Goss, N. Suri, B. Marinelli, L. Chen, A. Hashim, S. Anand, A. Morvan, R. K. Naik, E. Rrapaj, D. I. Santiago, W. de Jong, N. Y. Yao, J. E. Moore, I. Siddiqi, "A Qutrit Time Crystal Stabilized with Native Chiral Interactions", arXiv:2605.14293 (2026). https://arxiv.org/abs/2605.14293, https://arxiv.org/html/2605.14293
9. Z. Wang, R. W. Parker, E. Champion, M. S. Blok, "Systematic study of High E_J/E_C transmon qudits up to d = 12", Phys. Rev. Applied 23, 034046 (2025), DOI 10.1103/PhysRevApplied.23.034046, arXiv:2407.17407. https://arxiv.org/abs/2407.17407
10. M. J. Peterer, S. J. Bader, X. Jin, F. Yan, A. Kamal, T. Gudmundsen, P. J. Leek, T. P. Orlando, W. D. Oliver, S. Gustavsson, "Coherence and Decay of Higher Energy Levels of a Superconducting Transmon Qubit", Phys. Rev. Lett. 114, 010501 (2015), DOI 10.1103/PhysRevLett.114.010501, arXiv:1409.6031. https://arxiv.org/abs/1409.6031
11. L. Chen, H.-X. Li, Y. Lu, et al. (20 authors), "Transmon qubit readout fidelity at the threshold for quantum error correction without a quantum-limited amplifier", npj Quantum Inf. 9, 26 (2023), DOI 10.1038/s41534-023-00689-6, arXiv:2208.05879. https://arxiv.org/abs/2208.05879
12. S. Krinner, N. Lacroix, A. Remm, et al. (16 authors), "Realizing Repeated Quantum Error Correction in a Distance-Three Surface Code", Nature 605, 669 (2022), DOI 10.1038/s41586-022-04566-8, arXiv:2112.03708. https://arxiv.org/abs/2112.03708, https://ar5iv.labs.arxiv.org/html/2112.03708
13. J. Heinsoo, C. K. Andersen, A. Remm, S. Krinner, T. Walter, Y. Salathé, S. Gasparinetti, J.-C. Besse, A. Potočnik, C. Eichler, A. Wallraff, "Rapid high-fidelity multiplexed readout of superconducting qubits", Phys. Rev. Applied 10, 034040 (2018), DOI 10.1103/PhysRevApplied.10.034040, arXiv:1801.07904. https://arxiv.org/abs/1801.07904
14. N. R. Vora, Y. Xu, A. Hashim, N. Fruitwala, H. N. Nguyen, N. Goss, H. Liao, J. Balewski, A. Rajagopala, K. Nowrouzi, Q. Ji, K. B. Whaley, I. Siddiqi, P. Nguyen, G. Huang, "ML-Powered FPGA-based Real-Time Quantum State Discrimination Enabling Mid-circuit Measurements", arXiv:2406.18807 (v1 2024, v5 2026). https://arxiv.org/abs/2406.18807, https://arxiv.org/html/2406.18807
15. L. C. G. Govia, P. Jurcevic, C. J. Wood, N. Kanazawa, S. T. Merkel, D. C. McKay, "A randomized benchmarking suite for mid-circuit measurements", New J. Phys. 25, 123016 (2023), DOI 10.1088/1367-2630/ad0e19, arXiv:2207.04836. https://arxiv.org/abs/2207.04836, https://arxiv.org/html/2207.04836v2
16. D. Sank, Z. Chen, M. Khezri, et al. (24 authors), "Measurement-induced state transitions in a superconducting qubit: Beyond the rotating wave approximation", Phys. Rev. Lett. 117, 190503 (2016), DOI 10.1103/PhysRevLett.117.190503, arXiv:1606.05721. https://arxiv.org/abs/1606.05721
17. M. McEwen, D. Kafri, Z. Chen, et al. (50 authors), "Removing leakage-induced correlated errors in superconducting quantum error correction", Nat. Commun. 12, 1761 (2021), DOI 10.1038/s41467-021-21982-y, arXiv:2102.06131. https://arxiv.org/abs/2102.06131, https://arxiv.org/pdf/2102.06131 (text extracted locally)
18. P. Magnard, P. Kurpiers, B. Royer, T. Walter, J.-C. Besse, S. Gasparinetti, M. Pechal, J. Heinsoo, S. Storz, A. Blais, A. Wallraff, "Fast and Unconditional All-Microwave Reset of a Superconducting Qubit", Phys. Rev. Lett. 121, 060502 (2018), DOI 10.1103/PhysRevLett.121.060502, arXiv:1801.07689. https://arxiv.org/abs/1801.07689
19. Y. Salathé, P. Kurpiers, T. Karg, C. Lang, C. K. Andersen, A. Akin, C. Eichler, A. Wallraff, "Low-Latency Digital Signal Processing for Feedback and Feedforward in Quantum Computing and Communication", Phys. Rev. Applied 9, 034011 (2018), DOI 10.1103/PhysRevApplied.9.034011, arXiv:1709.01030. https://arxiv.org/abs/1709.01030
20. L. Caune, L. Skoric, N. S. Blunt, et al. (25 authors), "Demonstrating real-time and low-latency quantum error correction with superconducting qubits", arXiv:2410.05202 (2024). https://arxiv.org/abs/2410.05202
21. R. Acharya, L. Aghababaie-Beni, I. Aleiner, et al. (249 authors), "Quantum error correction below the surface code threshold", Nature 638, 920 (2025), DOI 10.1038/s41586-024-08449-y, arXiv:2408.13687. https://arxiv.org/abs/2408.13687
22. B.-J. Liu, Y.-Y. Wang, Y.-X. Wang, M. Badbaria, S. Puri, C. Wang, "Hardware-Efficient Erasure Qubits With Superconducting Transmon Qutrits", arXiv:2604.08672 (2026). https://arxiv.org/abs/2604.08672
23. N. Goss, S. Ferracin, A. Hashim, A. Carignan-Dugas, J. M. Kreikebaum, R. K. Naik, D. I. Santiago, I. Siddiqi, "Extending the Computational Reach of a Superconducting Qutrit Processor", arXiv:2305.16507 (2023). https://arxiv.org/abs/2305.16507
24. L. Abdurakhimov, J. Adam, H. Ahmad, O. Ahonen, M. Algaba, et al. (113 authors, IQM), "Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer", arXiv:2408.12433 (2024). https://arxiv.org/abs/2408.12433, https://arxiv.org/pdf/2408.12433 (text extracted locally)
25. F. Marxer, J. Mrożek, J. Andersson, et al. (IQM), "Above 99.9% Fidelity Single-Qubit Gates, Two-Qubit Gates, and Readout in a Single Superconducting Quantum Device", arXiv:2508.16437 (2025). https://arxiv.org/abs/2508.16437
26. F. Vigneau, S. Majumder, A. Rath, et al. (IQM), "Quantum error detection in qubit-resonator star architecture", arXiv:2503.12869 (v3, 2025; page lists DOI 10.1103/8m33-wn4g). https://arxiv.org/abs/2503.12869, https://arxiv.org/html/2503.12869
27. J. Rönkkö, O. Ahonen, V. Bergholm, et al. (16 authors, IQM), "On-Premises Superconducting Quantum Computer for Education and Research", arXiv:2402.07315 (2024; page lists DOI 10.1140/epjqt/s40507-024-00243-z). https://arxiv.org/abs/2402.07315, https://arxiv.org/html/2402.07315v2
28. E. Hyyppä, A. Vepsäläinen, M. Papič, C. F. Chan, S. Inel, A. Landra, W. Liu, J. Luus, F. Marxer, C. Ockeloen-Korppi, S. Orbell, B. Tarasinski, J. Heinsoo, "Reducing leakage of single-qubit gates for superconducting quantum processors using analytical control pulse envelopes", PRX Quantum 5, 030353 (2024), DOI 10.1103/PRXQuantum.5.030353, arXiv:2402.17757. https://arxiv.org/abs/2402.17757
29. IQM, "Qiskit on IQM user guide", IQM client 36.0.0 documentation. https://docs.iqm.tech/iqm-client/user_guide_qiskit.html
30. IQM, `iqm.iqm_client.models`, IQM client 36.0.0 API documentation. https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.html
31. IQM, iqm-pulse 15.0.0 API documentation: https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.html, .../iqm.pulse.gates.lru.html, .../iqm.pulse.gates.prx.html, .../iqm.pulse.gates.measure.html, .../iqm.pulse.gates.measure.Shelved_Measure_Constant.html, .../iqm.pulse.gates.measure.Measure_Constant.html, .../iqm.pulse.gates.reset.html, .../iqm.pulse.playlist.instructions.html (all under https://docs.iqm.tech/iqm-pulse/api/)
32. IQM, "Custom Gates and Gate Implementations", IQM Pulla 15.0.0 documentation. https://docs.iqm.tech/iqm-pulla/Custom%20Gates%20and%20Implementations.html
33. M. Bruce (IQM), "How IQM's pulse-level access is fuelling value for end-users", IQM blog, 28 Feb 2025, updated April 2026. https://iqm.tech/blog/how-iqms-pulse-level-access-is-fuelling-value-for-end-users/
34. IQM, qiskit-on-iqm fake Garnet backend source. https://raw.githubusercontent.com/iqm-finland/qiskit-on-iqm/main/src/iqm/qiskit_iqm/fake_backends/fake_garnet.py
35. IQM, KQCircuits repository README (GPL-3.0). https://github.com/iqm-finland/KQCircuits
36. T. Chen, S. Smart, Z. Mohammad, "Experiment with dynamic circuits on IQM Garnet with Amazon Braket", AWS Quantum Technologies Blog, 26 Jun 2025. https://aws.amazon.com/blogs/quantum-computing/experiment-with-dynamic-circuits-on-iqm-garnet-with-amazon-braket
37. Z. Mohammad, C. Panchalam Govindarajan, P. Komar, S. Seegerer, "Amazon Braket launches new 54-qubit superconducting quantum processor from IQM", AWS Quantum Technologies Blog, 21 Jul 2025. https://aws.amazon.com/blogs/quantum-computing/amazon-braket-launches-new-54-qubit-superconducting-quantum-processor-from-iqm/
38. A. Remm, N. Lacroix, L. Bödeker, et al. (12 authors), "Experimentally Informed Decoding of Stabilizer Codes Based on Syndrome Correlations", arXiv:2502.17722 (2025). https://arxiv.org/abs/2502.17722
39. Z. Chen, K. J. Satzinger, J. Atalaya, et al. (91 authors), "Exponential suppression of bit or phase flip errors with repetitive error correction", Nature 595, 383 (2021), DOI 10.1038/s41586-021-03588-y, arXiv:2102.06132. https://arxiv.org/abs/2102.06132
40. T. Wagner, H. Kampermann, D. Bruß, M. Kliesch, "Learning logical Pauli noise in quantum error correction", Phys. Rev. Lett. 130, 200601 (2023), DOI 10.1103/PhysRevLett.130.200601, arXiv:2209.09267. https://arxiv.org/abs/2209.09267
41. V. Sivak, M. Newman, P. Klimov, "Optimization of decoder priors for accurate quantum error correction", Phys. Rev. Lett. 133, 150603 (2024), DOI 10.1103/physrevlett.133.150603, arXiv:2406.02700. https://arxiv.org/abs/2406.02700
42. J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical Randomized Benchmarking", arXiv:1702.03688 (2017). https://arxiv.org/abs/1702.03688

Searched but not cited: no |3>-leakage rate for qutrit gates was found
(searches for qutrit leakage to the fourth level returned simulations and a
CCZ paper only). A qubit-style reset reference (arXiv:2506.09145) was fetched,
but its abstract does not cover reset of leaked states, so it is not used.
