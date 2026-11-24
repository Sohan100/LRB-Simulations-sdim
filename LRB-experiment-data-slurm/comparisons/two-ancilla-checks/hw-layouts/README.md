# Hardware layouts: one stabilizer at a time, two ancillas per check (2026-11-16)

The question was whether the [[5,1,2]]_3 checks can be measured one stabilizer at a time,
with at most two ancillas per check and only the couplers drawn in the user's figure.

**Layouts.**
- Left: a touches q0, q1, q2, q3; b touches q2, q4; data coupler q2-q3.
- Right: a touches q0, q1, q2; b touches q0, q3; c touches q1, q2, q4; data coupler q2-q3.

**Chosen protocols (user, 2026-11-16).** The user said the q2-q3 coupler "has to be
there", because q2 and q3 take part in the two-qutrit gates of the logical Cliffords. The
protocols are therefore the two bold rows in the Results table:
- left: hybrid, where b reads G3 G4^-1;
- right: c reads G4 through q2-q3.

**Stabilizers.** These are the code's own: G1 = Z0 Z1^-1 Z3^-1, G2 = Z1 Z2 Z4^-1,
G3 = X0 X1 X2^-1, G4 = X1^-1 X3 X4^-1.

**Scripts.**
- `hw_layouts_check.py` runs three tests on every variant:
  - zero noise, 5 rounds, with and without a logical H;
  - each of the 40 single data Paulis gives a deterministic readout that matches its
    syndrome and repeats between rounds;
  - exhaustive single faults, using the same fault model as `../fault_check.py`.
- `hw_layouts_bench.py` is a noisy 3-round memory benchmark with production SI1000/SD6
  noise, using the same harness as `../noise_bench.py`.
- `hw_layouts_statevector.py` runs the chosen rounds noiselessly on a random
  (non-stabilizer) logical state, as an exact 9-qutrit statevector. Over 5 states and
  3 rounds, the data fidelity stays 1, every readout gives 0 with probability 1, and
  all four stabilizers stay +1. It also traces the left G2 relay gate by gate and runs
  the naive split of G2 (fidelity 1/3, ⟨G4⟩ = 0). Output: `hw_layouts_statevector.out`.

**Outputs.**
- `hw_layouts_check.out` covers the first variants and the negative control.
- `hw_layouts_check_v2.out` and `hw_layouts_check_v3.out` cover the hook-safe and hybrid
  variants.
- `bench-*.txt` holds the noisy benchmarks.

## Why two separate readouts are not enough

Splitting a check over two ancillas and adding the two readouts measures each half on its
own, which disturbs the code (see `../CONV-SITES-FACTS.md`). The negative control here
splits G4 over b (x3) and c (-x1 - x4) on the right layout. With all checks required to be
0, 3000/3000 shots pass after 1 round, 974 after 2 and 43 after 5.

A split check therefore uses a data relay:
1. the lender ancilla collects its part;
2. it adds that part to a data qutrit that the reader also touches;
3. the reader reads;
4. the lender takes its part back and clears itself;
5. the lender is read and must give 0.

## Results

Every variant below passes the noiseless tests. The silent counts are single faults that
cause an undetected logical error, out of about 1600-2000 faults. Z flips are found with
the logical-H sandwich; X flips without it. The one-ancilla reference gives 0 / 0.

| layout | X check through | gates | silent (X flip / Z flip) |
|---|---|---|---|
| left | b lends x4 via q2, a reads G3G4 | 19 | 0 / 6 |
| left | a lends x0 x3 via q2, b reads G3G4 | 20 | 0 / 8 |
| left | b lends via q2, a reads G3G4^-1 or G3^-1G4 | 20 | 0 / 4 |
| left | difference trick, a reads G4 | 20 | 0 / 6 |
| **left** | **hybrid with q2-q3: b reads G3G4^-1, a lends x0 x1 via q2, data CNOT moves x3 into q2** | **22** | **0 / 0** |
| left | same hybrid, G4 | 21 | 0 / 2 |
| right | 3 ancillas: b via q0, c via q2, a reads G3G4 | 22 | 0 / 20 |
| right | 3 ancillas, a reads G3G4^-1 (c via q2 or q1) | 23 | 0 / 8 or 4 |
| **right** | **c reads G4; q2-q3 data CNOT and its inverse, c reads q2 before and after** | **18** | **0 / 0** |

The Z relays are always fine:
- left G2: a reads, b lends -z4 via q2 (or the reverse);
- right G1: a reads, b lends -z3 via q0 (or the reverse).

## Noisy cost (3 rounds, 50k shots, `bench-*.txt`)

The table gives acceptance / logical error among accepted shots, X basis. Error counts are
small (3-29 events), so read the error column as an order of magnitude. Every
one-at-a-time round has 8 collapse moments: a and c are measured and reset after each
check.

| scheme | moments | SI1000 0.005 | SI1000 0.01 | SD6 0.01 |
|---|---|---|---|---|
| one ancilla (production) | 7g+2c | 0.394 / 3.0e-4 | 0.155 / 1.0e-3 | 0.150 / 2.8e-3 |
| conventional sites, relay once | 11g+2c | 0.345 / 1.7e-4 | 0.119 / 8.4e-4 | 0.064 / 6.9e-3 |
| left, ancillas only (G3G4^-1) | 19g+8c | 0.105 / 3.2e-3 | 0.011 / 9.2e-3 | 0.010 / 2.4e-2 |
| left, hybrid with q2-q3 | 23g+8c | 0.097 / 3.5e-3 | 0.010 / 2.0e-2 | 0.005 / 3.5e-2 |
| right, 3 ancillas (G3G4^-1, c via q1) | 16g+8c | 0.081 / 1.7e-3 | 0.006 / 1.3e-2 | 0.010 / 6.3e-3 |
| right, c with q2-q3 | 14g+8c | 0.107 / 5.4e-3 | 0.012 / 1.9e-2 | 0.019 / 2.6e-2 |

At these p, the cost of the extra collapse moments and gates hides the difference between
0 and about 4 silent single faults. That difference is a term linear in p, so it matters
as p goes down.

## Why the ancilla-only X relays leak

Every X-type stabilizer that is not a power of G3 contains both q3 and q4.

**Left layout.** q4 reaches the rest of the chip only through b, and b only through q2. A
fault on a gate between q2 and the relay leaves the data with X2^x times the relayed part
raised to y, for any x and y. Modulo the measured stabilizer, this equals X2^x' X4^y'. The
family includes X2 X4, which is a weight-2 logical that no check sees.

**Right layout.**
- q3's only ancilla is b and q4's only ancilla is c. Since b and c share no data qutrit, no
  two-ancilla relay exists. With only ancilla-data gates, the two readouts would be
  independent measurements on {q0, q3} and on {q1, q2, q4}.
- With a third ancilla, b's part has to pass through q0. The same family then includes
  X0 X3, which is the logical X.

The weight-2 X logicals sit on {0,3} and {2,4}, and the weight-2 Z logicals on {0,2} and
{3,4}. A relay is single-fault safe only if no gate holds a pair of parts that can combine
into one of these. The q2-q3 coupler gets around this by moving x3 into q2. The pair {2,3}
carries no logical, and on the left a's part {0,1} combines with q2 only into G3.

## Best superconducting layout for the paper's code (2026-11-16)

The question was which chip layout best fits the paper's [[5,1,2]]_3 code with
nearest-neighbour logical gates. The logical H and S use a single data-data pair, q2-q3
(SWAP_{2,3} in H, CZ_{2->3} in S), so the q2-q3 coupler is fixed.

**Scripts.**
- `layout_candidates.py` covers six layouts: graph facts, statevector checks, the
  noiseless tests, `--faults` and `--bench`. A `--noreset` variant drops every ancilla
  reset; with all syndromes 0 every readout stays 0, so the same postselection applies.
- `grid_embedding.py` searches exhaustively for a crossing-free layout on a square grid
  with nearest-neighbour couplers (horizontal, vertical, one-cell diagonal). It also
  reports the number of frequency groups (chromatic number).

**Outputs.** `layout_candidates.out`, `layout_candidates_faults.out`,
`layout-bench-*.txt`, `layout-bench-noreset-*.txt`, `grid_embedding.out`.

**Recommended layout ("2a").**
- 5 data and 2 ancillas, 9 couplers.
- Ancilla a couples to q0, q1, q2, q3 and reads G1 and G3. Ancilla b couples to q1, q2,
  q3, q4 and reads G2 and G4. Every check is a direct read: no relays, and no data-data
  gate in the check round.
- G1 and G2 run in parallel, then G3 and G4.
- It fits a 3x3 grid patch, all couplers nearest-neighbour and none crossing:

```
  .   q3   .
  a   q2   b
  q0  q1   q4
```

| layout | qutrits | couplers | max degree data/anc | NN grid, no crossing | 2q gates | readout layers | silent single faults |
|---|---|---|---|---|---|---|---|
| **2a** | 7 | 9 | 3 / 4 | yes | 12 | 2 | 0 / 0 |
| 2b: a{0134}, b{0124} | 7 | 9 | 2 / 4 | no | 12 | 2 | 0 / 0 |
| production, 4 ancillas | 9 | 13 | 4 / 3 | no | 12 | 1 | 0 / 0 |
| 1 ancilla on all five | 6 | 6 | 2 / 5 | - | 12 | 4 | 0 / 0 |
| user LEFT, hybrid relay | 7 | 7 | 3 / 4 | yes | 22 | 4 | 0 / 0 |
| user RIGHT, q2-q3 difference | 8 | 9 | 3 / 3 | yes | 18 | 4 | 0 / 0 |

Every layout with the q2-q3 coupler needs 3 frequency groups. The proof: an ancilla
reading a check that contains q2, then q1, then an ancilla reading a check that contains
q3, then q3, then q2, closes an odd cycle. Since every check contains q1, this always
exists. So no such layout fits a plain square lattice (see also
`paper-rework/garnet-connectivity/FINDINGS.md`). 2a has two triangles, a-q2-q3 and b-q2-q3.

**Noisy cost.** 3 rounds, acceptance / logical error among accepted shots, X basis. The
production row is the LRB generator's own round. Event counts are small (3-50).

| layout | SI1000 0.002 | SI1000 0.005 | SD6 0.01 |
|---|---|---|---|
| production, with reset (as in the LRB runs) | 0.690 / 2.9e-5 | 0.394 / 2.5e-4 | 0.156 / 1.3e-3 |
| production, no reset | 0.758 / 0 (0 events) | 0.500 / 2.0e-4 | 0.183 / 1.4e-3 |
| 2a, with reset | 0.627 / 1.9e-4 | 0.309 / 1.2e-3 | 0.138 / 7.3e-3 |
| **2a, no reset** | **0.718 / 5.6e-5** | **0.434 / 4.2e-4** | **0.186 / 3.4e-3** |
| user LEFT hybrid | 0.395 / 5.7e-4 | 0.098 / 3.1e-3 | 0.005 / 1.2e-2 |
| user RIGHT q2-q3 | 0.406 / 5.1e-4 | 0.107 / 3.2e-3 | 0.018 / 1.8e-2 |

**What the comparison shows.**
- Readout layers dominate the cost, because data idle during every measure/reset moment.
- Production reads all four checks at once, but cannot be drawn with nearest-neighbour
  couplers only: q2 and q3 meet only on the outer ring, so the coupler runs around the
  patch.
- 2a without reset matches production with reset in acceptance, with about 2x the
  logical error (8 vs 4 events at p = 0.002).
- 2a is 3-4x better than the relay layouts in both columns.

**Extras (statevector, `layout_candidates.out`).**
- The layout is symmetric under q2 <-> q3. H-bar's SWAP can therefore be applied by
  relabelling: the single-qutrit layer alone maps the code to the relabelled code, which
  the same ancillas read. Two relabelled rounds read 0 with probability 1, and the
  fidelity with H-bar|psi> is 1. CZ is symmetric, so S-bar is unchanged.
  Result: CZ_{2,3} in S-bar is the only data-data gate.
- |+_L> can be prepared without the encoder's data couplers. Start from |+>^5, read G1
  and G2 once, and apply X3^{G1} X4^{G2}. The fidelity with |+_L> is 1.

**Not modelled.** Crosstalk and frequency crowding (the qutrit 1-2 transitions double
the collisions), measurement-induced dephasing of neighbours during mid-circuit readout,
qutrit readout T1, and leakage. Ancilla degree 4 is assumed to be acceptable.

## Square lattice only, no diagonal couplers (2026-11-16)

The user's constraint is "keep out non local connections". 2a uses four one-cell diagonal
couplers, which do not exist on a square-lattice chip.

**The obstruction.** Without diagonals the coupling graph must be bipartite. Any
direct-read layout with q2-q3 contains an odd cycle (previous section). So two of the
four checks have to use the q2-q3 data CNOT inside the check, which is the difference
trick of the right layout's G4.

**Layout.**
- Ancilla a couples to q0, q1, q2. Ancilla c couples to q1, q2, q4. Plus q2-q3.
- 7 qutrits, 7 couplers, max degree 3, bipartite, so 2 frequency groups.
- It is the user's right layout without ancilla b.

```
  q3  q2  c   q4
  q0  a   q1
```

**Checks.**
- a reads G3 directly.
- a reads G1 as follows: read Z0 Z1^-1 Z2, apply CNOT 3->2, read Z2^-1, then apply
  CNOT_INV 3->2. This works because CNOT_INV(3->2) Z2^-1 CNOT(3->2) = Z2^-1 Z3^-1.
- c reads G2 directly.
- c reads G4 with the same trick through CNOT 2->3, as in `R_G4_dd`.
- G1 and G2 run in parallel, then G3 and G4. Per round: 18 two-qutrit gates and
  4 readouts, in 13 gate + 4 collapse moments (13 + 2 without reset).

**Checks run.** `layout_square.out`, `layout_square_faults.out`,
`layout-bench-square-*.txt`, `garnet_square_layout.out`.
- Noiseless tests pass.
- 0 of 1576 single faults are silent, with or without the logical-H sandwich.
- It fits IQM Garnet's coupler graph in 276 placements, e.g. a=QB4, c=QB8, q0=QB5,
  q1=QB3, q2=QB9, q3=QB10, q4=QB13. 2a fits in 0.
- This revises `paper-rework/garnet-connectivity/FINDINGS.md`, which found that only
  terminal postselection fits Garnet. That search did not allow data-data gates inside
  the checks.

**Noisy cost.** 3 rounds, acceptance / logical error among accepted shots, X basis.

| layout | SI1000 0.002 | SI1000 0.005 | SD6 0.01 |
|---|---|---|---|
| production, with reset (LRB runs; not local) | 0.690 / 2.9e-5 | 0.394 / 2.5e-4 | 0.156 / 1.3e-3 |
| 2a, no reset (diagonals) | 0.718 / 5.6e-5 | 0.434 / 4.2e-4 | 0.186 / 3.4e-3 |
| square lattice, with reset | 0.598 / 2.8e-4 | 0.278 / 1.4e-3 | 0.065 / 1.1e-2 |
| **square lattice, no reset** | **0.686 / 8.0e-5** | **0.383 / 6.8e-4** | **0.085 / 8.7e-3** |
| user RIGHT (needs diagonals) | 0.406 / 5.1e-4 | 0.107 / 3.2e-3 | 0.018 / 1.8e-2 |
| user LEFT (needs diagonals) | 0.395 / 5.7e-4 | 0.098 / 3.1e-3 | 0.005 / 1.2e-2 |

At SI1000 p = 0.002, the square layout without reset matches the production round's
acceptance, with about 3x its logical error (11 vs 2-5 events). Under SD6, where every
moment costs full idle noise, its 13 gate moments cost more.

**Does the square layout run both LRB-D versions?** Yes. Script
`square_protocol_compat.py`, output `square_protocol_compat.out`.
- **Gates.** Every two-qutrit gate of the check round, logical H, H_INV, P and P_INV
  sits on a coupler: the logical gates use only q2-q3. The terminal X readout has no
  two-qutrit gates.
- **The paper's encoder does not fit.** It uses data CNOTs on (0,1), (0,3), (1,2) and
  (2,4), none of which is a coupler.
- **Routed encoder.** Same unitary, with ancillas returned to |0>, 14 two-qutrit gates,
  all on couplers. It matches the paper's encoder at fidelity 1 (statevector). CNOT 3->0
  runs first, while q2 is still |0>, through q2 and a.
- **Preparation by measurement** also works: |+>^5, then the G1/G2 half of the round.
  With postselection on 0 about 1/9 of shots are kept; Pauli-frame correction keeps all.
- **Noiseless sdim runs.** 20 random mirror sequences of 6 logical Cliffords, with the
  terminal check only and with check rounds every k = 1, 2, 3 Cliffords. Each of the
  three preparations passes every kept shot: all ancilla readouts 0, terminal G3 = G4 = 0,
  logical x0 + x3 = 0.

## Square lattice + the paper's encoder run natively (2026-11-16)

The user added a constraint: the layout must run the paper's encoder as written.

**What the encoder forces.**
- The encoder's data couplers are q0-q1, q1-q2, q0-q3 and q2-q4. With q2-q3 they close
  the 4-cycle q0-q1-q2-q3.
- On a square lattice that 4-cycle is a single plaquette, with q4 attached to q2.
- No empty site then touches more than two data qutrits, so no check can be read
  directly.
- Inside a check, data-data gates are allowed only on q0-q1, q1-q2 and q2-q3. A single
  fault on a q0-q3 or q2-q4 gate can produce X0X3 or X2X4, which are weight-2 logicals,
  so those two couplers stay encoder-only.

**Search.**
- `encoder_layout_search.py` finds the cheapest single-ancilla check circuits for each
  set of touched data qutrits: reads interleaved with data CNOTs on the safe edges,
  undone at the end.
- `encoder_layout_rounds.py` fault-checks every combination.
- Minimum two-qutrit gates per check, by touched set:
  - {q1, q4}: G2 4, G3 5, G4 8, G1 10.
  - {q2}: G3 5, G1 8.
  - {q1}: G3 5, G1 10.
  - {q3, q4}: G2 7, G4 8.

**Best layout "A1": 7 qutrits, 8 couplers, 2 frequency groups.**
- E sits next to q1 and q4 and reads G2 and G4.
- N sits above q2 and reads G1 and G3.
- Grid, with dashed links encoder-only:

```
       N
  q3  q2  q4
  q0  q1  E
```

**Results** (`encoder_rounds_A1.out`, `encoder_A1_grouped_check.out`,
`encoder_A1_protocol_compat.out`, `enc-bench-A1-*.txt`):
- 20 of 50 A1 combinations have 0 silent single faults. Safety depends only on which G4
  circuit is used.
- With the readouts grouped into one Z layer and one X layer, combo (0,0,0,1) has 0 of
  2136 silent faults, with or without the H sandwich.
- Per round: 25 two-qutrit gates, 4 readouts, 22 gate + 4 collapse moments (2 without
  reset). The circuits are in `encoder_A1_circuits.txt`.
- The encoder, logical H/S, checks and terminal readout all sit on couplers.
- Noiseless random mirror sequences pass every shot: terminal check only, and k = 1, 2, 3.
- Fits Garnet in 48 placements.
- A2 (second ancilla next to q1 instead of q2): 26 gates, 16 of 100 combinations safe.

**Noisy cost.** 3 rounds, acceptance / logical error among accepted shots, X basis.

| layout | SI1000 0.002 | SI1000 0.005 | SD6 0.01 |
|---|---|---|---|
| A1, encoder native, no reset | 0.640 / 1.2e-4 | 0.327 / 9.8e-4 | 0.022 / 3.6e-2 |
| A1, with reset | 0.561 / 3.0e-4 | 0.234 / 2.6e-3 | 0.016 / 3.6e-2 |
| square layout (routed encoder), no reset | 0.686 / 8.0e-5 | 0.383 / 6.8e-4 | 0.085 / 8.7e-3 |
| user RIGHT (needs diagonals) | 0.406 / 5.1e-4 | 0.107 / 3.2e-3 | 0.018 / 1.8e-2 |
| production, with reset (not local) | 0.690 / 2.9e-5 | 0.394 / 2.5e-4 | 0.156 / 1.3e-3 |

**What the comparison shows.**
- Every A1 check routes through the q0-q1-q2-q3 data path, so the checks serialize: 22
  gate moments.
- Under SI1000 at p = 0.002, A1 still beats the user's layouts.
- Under SD6, where every moment costs full idle noise, A1 is the weakest.
- The square layout keeps better check rounds but encodes through the ancillas: 14
  two-qutrit gates instead of 4, which is SPAM and enters only the fit's A.

### A1 constraint audit (`a1_constraint_audit.py` → `a1_constraint_audit.out`)

This audit checks every constraint in one run, on the composed circuits as they execute:

- **Couplers.** The 8 couplers are exactly the grid-adjacent pairs of the 7 placed qutrits. Each one has grid distance 1, with no diagonals.
- **Encoder (state prep).** The paper's encoder (`logical_plus_initial_state`) uses q1-q2, q0-q1, q0-q3 and q2-q4. All four are nearest-neighbour couplers.
- **Logical gates.** The logical Cliffords use only q2-q3.
- **Terminal readout.** It has no two-qutrit gates.
- **Check round.** It uses q0-q1, q1-q2, q2-q3, q1-E, q4-E and q2-N.
- **Composed runs.** The terminal-only and interval k = 1, 2, 3 runs were scanned operation by operation. Every two-qutrit operation sits on a coupler, and no operation acts on 3 or more qutrits.
- **Ancillas and readouts.** There are 2 ancillas, and each readout measures exactly one stabilizer (N: G1, G3; E: G2, G4).
- **Frequency groups.** The coupler graph is bipartite, so it needs 2 groups.
- **Noiseless runs.** They pass 160/160 shots for each version.
- **Single faults.** 0 of 2136 are silent (see `encoder_A1_grouped_check.out`).

### A1 checks are non-destructive (`a1_statevector.py` → `a1_statevector.out`)

This is an exact 9-qutrit statevector run with projective ancilla readouts.

- **Result.** Over 10 random (non-stabilizer) logical states × 3 rounds, the data fidelity with the input state is 1.000000000000. The largest P(readout ≠ 0) is 6e-16.
- **Mid-check behaviour.** The gate-by-gate traces of G1 and G2 show the ancilla entangled with the data partway through a check (P(N=0) = 1/3 after G1's first read). The data's CNOTs also take it temporarily out of the code space (overlap 0.012).
- **After the check.** Both return to normal once the check's last gate is done.

## Optimal encoders and the final layout (2026-11-17, folder `encoders/`)

The user's request: the layout and its optimized encoding circuit must be final before the LRB-D campaigns start. The best encoder has the fewest CNOT gates, with every two-qudit gate nearest-neighbour. Encoders were also requested for the paper's one-ancilla layout (Overleaf) and for IQM Garnet.

**Minimal encoder of |+_L> (any layout).** Five entangled qutrits need at least 4 two-qutrit gates, and exactly 4 form a spanning tree of the data.
- The state has entropy 1 across every single-qutrit cut and across {q0,q3} and {q2,q4}, and 2 across every other cut (`encoder_trees.out`).
- So only 21 of the 125 trees pass the cut test.
- 17 of them have a 4-CNOT encoder (subspace search over F_3^5).
- An exhaustive Clifford search with the exact LC test covered the 4 star trees (`clifford_tree_search.out`): the stars at q2, q3 and q4 have a 4-gate encoder (CZ-type gates); the star at q1 has none.

**Co-design** (`codesign.py`, `codesign_rounds.py`, `codesign_finalists.py`). For every encoder tree T, data couplers = T + q2-q3 on the square lattice, plus two ancillas on lattice sites.
- Data-data gates inside checks use only pairs without a weight-2 logical; 02, 03, 24 and 34 are encoder-only.
- The search finds the minimal difference-trick protocols per check. Rounds are built with no mid-round reset (reset only at round start), fault-screened, fully fault-checked, scheduled, and benchmarked (3-round memory, 200k shots).
- The 23-gate layouts all have silent single faults. Fault-safe finalists (X-run acceptance / logical error):

| layout (encoder tree) | 2q gates | gate moments | SI1000 0.002 | SI1000 0.005 | SD6 0.01 |
|---|---|---|---|---|---|
| **B: 01 03 12 14** | 24 | 19 | 0.601 / 1.6e-4 | 0.281 / 1.0e-3 | 0.029 / 2.1e-2 |
| mirror of B: 01 13 14 24 | 24 | 19 | 0.602 / 1.6e-4 | 0.282 / 9.9e-4 | 0.030 / 2.0e-2 |
| star at q3 | 24 | 20 | 0.599 / 1.1e-4 | 0.277 / 1.0e-3 | 0.024 / 2.4e-2 |
| star at q2 | 24 | 21 | 0.596 / 1.8e-4 | 0.274 / 1.2e-3 | 0.020 / 2.7e-2 |
| A1 (paper encoder) | 25 | 22 | 0.594 / 2.1e-4 | 0.271 / 1.4e-3 | 0.018 / 4.1e-2 |

**Layout B** (recommended):

```
q4  A
q1  q2  B
q0  q3
```

- Couplers: q0-q1, q0-q3, q1-q2, q2-q3, q1-q4, A-q4, A-q2, B-q2 (8, all lattice-adjacent; q0-q3 is encoder-only).
- Ancillas: A reads G2, G4; B reads G1, G3.
- Encoder: H on q2, q3, q4; CNOT 3->0; CNOT 4->1; CNOT_INV 2->1; CNOT 1->0. That is 4 gates and depth 4: the paper's encoder with CNOT 4->2 replaced by CNOT 4->1.
- Verification (`layoutB_verify.out`, `layoutB_readout_check.out`):
  - the encoder state equals the paper's |+_L> (d = 3 and 2);
  - every gate is on a coupler;
  - noiseless terminal and k = 1, 2, 3 runs pass 160/160 shots for d = 3 and 2;
  - each readout is one stabilizer;
  - the statevector fidelity is 1;
  - 0 of 2056 (d = 3) and 0 of 411 (d = 2) single faults are silent;
  - it fits Garnet in 48 placements.
- Round ops: `layoutB_round_ops.txt`.

**One-ancilla layout (paper).** Any encoder needs at least 6 gates, by a Steiner tree through two ancillas.
- The minimal CNOT-network encoder has 8 gates: the per-pair search proves this through a1 and a2, and the meet-in-the-middle search shows nothing at 7 gates or fewer with all four ancillas (`ancilla_encoder_pairs.out`, `ancilla_encoder_mitm.out`).
- Best depth 5, valid for d = 3 and 2, ancillas back in |0> (`one_ancilla_encoder_pick.out`): H on q3, a1, a2; CNOT_INV a2->q4; CNOT_INV a1->q1; CNOT q1->a2; CNOT_INV a2->q2; CNOT q2->a2; CNOT_INV q3->a1; CNOT_INV a1->q0; CNOT q0->a1.

**Garnet.** Layout B, and the data part alone for the terminal-check version, embeds in Garnet, so the 4-CNOT encoder is native there. Example placement: q0 QB3, q1 QB4, q2 QB9, q3 QB8, q4 QB5, A QB10, B QB14. The encoder is then:
- H on QB9, QB8, QB5;
- CNOT QB8->QB3; CNOT QB5->QB4; CNOT_INV QB9->QB4; CNOT QB4->QB3.

The logical gates use QB9-QB8.

## One ancilla per check on a square lattice (2026-11-17, folder `one-ancilla/`)

The user asked: "is there a better one ancilla layout that can get us higher pseudo thresholds and will be easier to fabricate?"

The setting is the paper's one-ancilla round: ancilla a_j reads only G_j, and all four checks share one reset layer and one readout layer. "Easier to fabricate" means a square lattice with nearest-neighbour couplers only, so no diagonal, long-range or crossing couplers. q2-q3 stays a coupler because the logical gates use it.

**Answer.**
- **Easier to fabricate: yes.** Layout SQ14 fills a 3x3 square patch with 10 nearest-neighbour couplers. The current layout has 13, and one of them, q2-q3, is long range. SQ14 has maximum degree 2 on data and 3 on ancillas, against 4 and 3 now. It needs 2 frequency groups and fits IQM Garnet in 32 placements. The current layout fits in none.
- **Higher p\*: no.** No square-lattice one-ancilla round reaches the current round's 12 two-qutrit gates. Every one needs data-data gates inside the round, and those gates set the logical error. SQ14 has the fewest data-data gates (4) of any round found, with 19 two-qutrit gates in 9 gate moments, against 12 in 7 now.
- **Noise near p\*.** SQ14's logical error per kept round is 0.96x and 1.06x the current round's at SD6 p = 0.02 and SI1000 p = 0.015, and 1.15-1.18x at SI1000 0.03 and SD6 0.04. It keeps 11-37% fewer shots. The other square-lattice rounds do worse, up to 1.7x.
- **What that means for p\*.** Logical error is second order in p and physical RB error first order. So a logical error 1.15x larger lowers p* by at most about 13%, and less, because the logical Cliffords' own error is unchanged. I expect SQ14's p* close to today's values or a little below. Only an LRB-D run can give the number.
- **Where the p\* gain is.** The current layout's own round packs into 5 gate moments instead of 7, with the same 12 gates and 0 silent single faults. Under SD6 this halves the logical error per round (0.51x at p = 0.02 and 0.04) and keeps 26-70% more shots. Under SI1000 the logical error drops to 0.78x at p = 0.015 and 0.94x at 0.03.

So the choice is between SQ14, which is easy to build and gives about today's p*, and the current layout with the 5-moment round, which gives higher p* but keeps the long-range coupler.

### Why the square lattice costs gates

**1. No round without data-data gates.** This holds even when the two ancillas of one type may add partial sums into each other ("transfers": CNOTs a1-a2 or a3-a4).
- A square lattice is bipartite, so all neighbours of a site have one colour. q2 and q3 are coupled, so they have different colours.
- Every data qutrit is in a Z check and in an X check, since G1 and G2 cover all five data and so do G3 and G4.
- Without data-data gates, a Z ancilla learns z_q only by touching q or through a transfer from the other Z ancilla. So every data qutrit borders a Z ancilla and an X ancilla.
- A Z ancilla cannot touch both colours, so one Z ancilla borders every black data qutrit, and so does one X ancilla. Two lattice sites share at most two neighbours, so at most 2 data qutrits are black, and by the same argument at most 2 are white. That leaves room for 4 data qutrits, not 5.

**2. Data-data gates sit in windows, and windows on 02, 03, 24 or 34 fail.** A window is a data CNOT applied and undone inside the round. While it is open, an ancilla on the target reads the control's Z through it, and an ancilla on the control reads the target's X. A single fault on the pair inside the window is a two-qutrit error in the code frame, and on 02, 03, 24 and 34 that can be a weight-2 logical.
- With all pairs allowed, the search below finds layouts with 15-18 two-qutrit gates and two data CNOTs. All 213 of them use a window on one of these pairs.
- For each one, `oa_finalists.py` sampled 6000 schedules. All 185,724 schedules that measure the checks correctly have silent single faults.

**3. Search.** `oa_transfer_search.py` searches exhaustively within this model:
- data CNOT^±1 windows on the safe pairs 01, 04, 12, 13, 14, 23, each opened and closed inside the round, with reads allowed in every intermediate frame;
- each ancilla reads its lattice-neighbour data with CNOTs of its own type, and the two ancillas of a type may be coupled for transfers;
- 4 distinct ancilla sites, a bipartite colouring, and an exact embedding on Z^2.

Per type, a breadth-first search over the two ancillas' accumulated parities gives the fewest reads plus transfers for every pair of touched sets. `oa_lower_bound.py` adds a relaxed bound and a pruned exact pass for 3 and 4 data CNOTs. Fault safety comes from `oa_finalists.py` (random schedules, tableau screen) and the sdim checker.

| data CNOTs per round (data-data gates) | fewest two-qutrit gates | fault-safe layouts at that count |
|---|---|---|
| 0 | none exists (part 1) | - |
| 1 (2) | none with <= 24 | - |
| 2 (4) | 19, all with transfers | 12 distinct layouts, 8 of them with 10 couplers; best schedule 9 gate moments |
| 3 (6) | 15, no transfers | 8 distinct layouts, 11 couplers; 9 gate moments |
| 4 (8) | 15, none at 14 | 72 (layout, frame) pairs at 15; not fault-screened, since each carries 8 data-data gates |

With 5 or more data CNOTs the windows alone cost at least 10 data-data gates, and every ancilla needs at least one gate, so such a round has at least 14. So no square-lattice one-ancilla round in this model reaches the current round's 12.

Fewer gates did not mean less noise. The 15-gate rounds have 6 data-data gates and the highest logical error of all, see the benchmarks.

### Ranked candidates

All rounds pass the noiseless tests and have 0 silent single faults in the sdim checker, with and without H-bar. "2q gates" splits into data-data / ancilla-ancilla / ancilla-data. "Logical error near p*" is the ratio to the current round at SI1000 0.015, SI1000 0.03, SD6 0.02 and SD6 0.04.

| rank | layout | couplers | max degree data / anc | 2q gates | moments | Garnet | logical error near p* |
|---|---|---|---|---|---|---|---|
| 1 | **SQ14** | 10 | 2 / 3 | 19 (4 / 5 / 10) | 9 gate + reset + readout | 32 | 1.06, 1.15, 0.96, 1.18 |
| 2 | SQ01 | 10 | 2 / 3 | 19 (4 / 5 / 10) | 9 + 2 | 32 | 1.10, 1.24, 1.30, 1.41 |
| 3 | SQ15A, SQ15B | 11 | 4 / 3 | 15 (6 / 0 / 9) | 9 + 2 | 54 | 1.39-1.43, 1.41, 1.38-1.39, 1.57-1.68 |
| ref | current one-ancilla layout | 13, q2-q3 long range | 4 / 3 | 12 (0 / 0 / 12) | 7 + 2 | 0 | 1 |
| ref | current layout, 5-moment round | same | 4 / 3 | 12 | 5 + 2 | 0 | 0.78, 0.94, 0.51, 0.51 |

The ratios have about 4-7% statistical error. The other fault-safe 19-gate layouts were not benchmarked: 6 more with 10 couplers and the same degrees as SQ14, and 4 with 11 couplers and degree 3 / 3. All of them reach 9 gate moments. Layout B, the two-ancilla layout in the LRB-D campaign, has 8 couplers and 24 gates in 19 gate moments plus 3 collapse moments.

SQ01 and SQ15A, as lattice patches. SQ15B is drawn in `rounds.py`.

```
SQ01:  a4  q0  q1        SQ15A:   .  a1  q3   .
       a3  q4  a1                 .  q1  q2  a2
       q2  q3  a2                a3  q0  a4  q4
```

- SQ01: windows CNOT_INV 0->1 and CNOT_INV 2->3; a1 on q1, q4; a2 on q3; a3 on q2, q4; a4 on q0; couplers a1-a2 and a3-a4.
- SQ15A: windows [CNOT 1->2, then CNOT_INV 0->1 inside it] and CNOT 2->3; a1 on q1, q3; a2 on q2, q4; a3 on q0; a4 on q0, q2, q4; no ancilla-ancilla coupler. q2 has degree 4.

### Layout SQ14 and its round

```
a3  a4  q1
q3  q0  q4
q2  a2  a1
```

- Couplers: q1-q4, q2-q3, a1-q4, a2-q0, a2-q2, a3-q3, a4-q0, a4-q1, a1-a2, a3-a4. The lattice neighbours q0-q3 and q0-q4 are not couplers.
- Two windows: CNOT_INV 1->4 and CNOT 3->2. Through the first, a1 at q4 reads z1 and a4 at q1 reads x4. Through the second, a2 at q2 reads z3 and a3 at q3 reads x2.
- Transfers carry the rest. a1 and a2 exchange Z partial sums, and a3 and a4 exchange X partial sums.
- Op list: `rounds.py` (`SQ14`); wires a1..a4 = 5..8.

Schedule (ASAP, the generator's scheduler gives the same 9 gate moments):

| moment | gates |
|---|---|
| 1 | H a3, H a4 |
| 2 | CNOT_INV a3->q3, CNOT a4->q1 |
| 3 | CNOT_INV q1->q4 (opens window), CNOT a4->q0, CNOT q3->q2 (opens window) |
| 4 | CNOT_INV a3->a4, CNOT_INV q4->a1, CNOT q2->a2 |
| 5 | CNOT a4->a3, CNOT_INV q0->a2 |
| 6 | CNOT_INV a4->q1, CNOT a3->q3, CNOT_INV a2->a1 |
| 7 | CNOT q1->q4 (closes), CNOT_INV q3->q2 (closes), CNOT a1->a2, H_INV a3, H_INV a4 |
| 8 | CNOT_INV q4->a1, CNOT q2->a2 |
| 9 | CNOT a2->a1 |

Checks run (`verify_sq.out`, `protocol_compat_sq.out`, `layout_facts_sq.out`):
- noiseless: 5 rounds with and without H-bar, and the 40 single data Paulis, each read deterministic, QND and with the right syndrome;
- single faults: 0 of 1656 silent, with and without H-bar;
- every two-qutrit gate of the logical Cliffords is on q2-q3, and the terminal readout has none;
- noiseless LRB-D mirror sequences (20 random sequences of 6 Cliffords plus inverse) pass 160/160 shots for the terminal-check version and for k = 1, 2, 3. With preparation by measurement (|+>^5, one round, readouts postselected on 0) every kept shot passes.

**Encoder.** None of the paper's encoder pairs 01, 03, 12, 24 is a coupler. The lattice-adjacent data pairs, 03, 04, 14 and 23, contain no 4-gate encoder tree either, so no encoder-only coupler would give a 4-gate encoder. A unitary encoder has to run through the ancillas, and the sparse coupler graph makes it long. Any encoder needs at least 6 two-qutrit gates, since joining the five data qutrits takes two ancillas. The CNOT-network search of `../encoders/ancilla_encoder_search.py`, run on SQ14's own couplers, finds none with 10 gates or fewer through any pair of helper ancillas (`oa_encoder.py`, `encoder_sq14_pairs.out`). The search through three or four helpers did not finish. Preparation by measurement needs no extra couplers. Preparation is ideal in every LRB simulation, so the encoder does not change p*.

### Error propagation (`oa_fault_spread.py` → `fault_spread.out`)

The extra gates spread errors further, but no single fault spreads into an undetected logical. What grows is the number of fault pairs that combine into one. Tableau propagation over the sdim checker's fault locations:

| round | single faults | silent single faults | read faults leaving a data error on >= 2 qutrits | silent fault pairs, both on two-qutrit gates |
|---|---|---|---|---|
| current round | 1064 | 0 | 42% | 506 |
| current, 5 moments | 1064 | 0 | 42% | 410 |
| SQ14 | 1624 | 0 | 58% | 690 |
| SQ15A | 1304 | 0 | 65% | 1682 |

- In SQ14, faults on the data-data CNOTs are always detected, but 80% of them leave errors on two data qutrits.
- SQ14's 690 silent pairs split into read + read 252, read + data CNOT 210, read + transfer 176, transfer + data CNOT 28, data CNOT + data CNOT 24.
- Under two-qutrit depolarizing noise the leading logical error of a round is about N_pairs (p/80)^2, so SQ14's two-qutrit gates give about 1.4x the current round's and SQ15A's about 3.3x. Idle and readout noise, which this count leaves out, dilute the gap in the benchmarks below.

### Noisy benchmarks

Harness: `../../noise_bench.memory`, with production SI1000 and SD6 noise applied by the LRB generator, |+_L>, and a noiseless terminal readout. A shot is kept if every readout and the terminal G3, G4 are 0. Cells give acceptance / logical error among kept shots, pooled over the X and Z (H-bar sandwich) runs, with the event count. Every chunk of 20k shots has its own seed. Outputs: `bench_baselines.out`, `bench_sq*.out`; tables from `bench_summary.py`.

1 round, 400k shots per run, near p*:

| round | moments | SI1000 0.015 (R=1) | SI1000 0.03 (R=1) | SD6 0.02 (R=1) | SD6 0.04 (R=1) |
|---|---|---|---|---|---|
| current round | 7g+2c | 0.404 / 1.3e-03 (423) | 0.162 / 6.3e-03 (815) | 0.307 / 5.4e-03 (1321) | 0.098 / 2.5e-02 (1967) |
| current, 5 moments | 5g+2c | 0.414 / 1.0e-03 (338) | 0.169 / 5.9e-03 (797) | 0.406 / 2.7e-03 (888) | 0.167 / 1.3e-02 (1724) |
| SQ14 | 9g+2c | 0.361 / 1.4e-03 (402) | 0.129 / 7.2e-03 (748) | 0.245 / 5.2e-03 (1015) | 0.062 / 2.9e-02 (1467) |
| SQ01 | 9g+2c | 0.363 / 1.4e-03 (416) | 0.129 / 7.8e-03 (804) | 0.253 / 7.0e-03 (1410) | 0.067 / 3.5e-02 (1902) |
| SQ15A | 9g+2c | 0.384 / 1.8e-03 (557) | 0.147 / 8.8e-03 (1037) | 0.249 / 7.4e-03 (1477) | 0.068 / 4.2e-02 (2299) |
| SQ15B | 9g+2c | 0.386 / 1.9e-03 (579) | 0.148 / 8.8e-03 (1050) | 0.248 / 7.5e-03 (1484) | 0.068 / 3.9e-02 (2121) |

3 rounds, 200k shots per run (the points used in the earlier sections):

| round | moments | SI1000 0.002 (R=3) | SI1000 0.005 (R=3) | SD6 0.01 (R=3) | SD6 0.005 (R=3) |
|---|---|---|---|---|---|
| current round | 7g+2c | 0.689 / 3.6e-05 (10) | 0.391 / 2.2e-04 (35) | 0.151 / 2.4e-03 (144) | 0.387 / 5.3e-04 (82) |
| current, 5 moments | 5g+2c | 0.693 / 1.8e-05 (5) | 0.402 / 1.7e-04 (28) | 0.239 / 1.2e-03 (113) | 0.488 / 2.7e-04 (52) |
| SQ14 | 9g+2c | 0.660 / 3.8e-05 (10) | 0.352 / 2.6e-04 (37) | 0.107 / 2.1e-03 (88) | 0.328 / 5.4e-04 (71) |
| SQ01 | 9g+2c | 0.660 / 2.3e-05 (6) | 0.352 / 2.6e-04 (37) | 0.110 / 2.6e-03 (114) | 0.330 / 5.3e-04 (70) |
| SQ15A | 9g+2c | 0.673 / 4.5e-05 (12) | 0.370 / 3.7e-04 (55) | 0.106 / 3.3e-03 (140) | 0.324 / 8.3e-04 (107) |
| SQ15B | 9g+2c | 0.673 / 4.5e-05 (12) | 0.372 / 3.1e-04 (46) | 0.107 / 3.2e-03 (138) | 0.326 / 6.4e-04 (84) |

What the tables show:
- Near p*, SQ14 is the only square-lattice round within 20% of the current round's logical error. SQ01 is up to 1.4x and the 15-gate rounds up to 1.7x.
- The square-lattice rounds keep fewer shots. That costs sampling time but does not enter p*.
- The 5-moment round beats the current one at every point, by a factor 2 under SD6.

### Current layout, 5-moment round (user decision 2026-11-19: keep the current layout)

The user asked: "Lets stick to the current layout, does moment thing also work for si1000?" Yes, the gain is smaller under SI1000 than under SD6.

`rank_prod5.py` ranks all 52 fault-safe 5-moment schedules by silent fault pairs on two-qutrit gates. The best ones have 346, against 410 for PROD5 and 506 for the generator's 7-moment round. PROD5B, one of the best, is in `rounds.py`. It passes the sdim checks: noiseless tests, and 0 of 1096 single faults silent with and without H-bar.

| moment | PROD5B gates |
|---|---|
| 1 | a1 reads q1; H on a3, a4 |
| 2 | a1 reads q0; a3 reads q2; a4 reads q1 |
| 3 | a1 reads q3; a2 reads q2; a3 reads q0; a4 reads q4 |
| 4 | a2 reads q4; a3 reads q1; a4 reads q3 |
| 5 | a2 reads q1; H^-1 on a3, a4 |

Benchmark, 1 round, 1M shots per run (`bench_prod5_R1_1M.out`). Logical error among kept shots and acceptance, as ratios to the generator's round:

| noise | p | generator round: acceptance / logical error | PROD5 | PROD5B |
|---|---|---|---|---|
| SI1000 | 0.015 | 0.405 / 1.26e-3 | 0.90 +- 0.04, acceptance x1.02 | 0.85 +- 0.04, x1.02 |
| SI1000 | 0.02 | 0.299 / 2.48e-3 | 0.84 +- 0.03, x1.03 | 0.86 +- 0.03, x1.02 |
| SI1000 | 0.03 | 0.163 / 6.36e-3 | 0.86 +- 0.03, x1.04 | 0.81 +- 0.03, x1.03 |
| SD6 | 0.02 | 0.307 / 5.26e-3 | 0.53 +- 0.01, x1.32 | 0.48 +- 0.01, x1.31 |
| SD6 | 0.04 | 0.099 / 2.60e-2 | 0.50 +- 0.01, x1.70 | 0.44 +- 0.01, x1.67 |

- Under SI1000, PROD5B lowers the logical error per round by 14-19%, and acceptance barely moves. Idle noise is only p/10 per moment there, so most of the gain comes from the CNOT order, which leaves fewer silent fault pairs.
- Under SD6 it more than halves the logical error, because every saved moment removes full-strength idle noise on all nine qutrits.
- The LRB-D rerun with this round in the generator is below, under "The five-moment round in the generator".

### Best schedule of the current round: exhaustive search (2026-11-19)

The user asked: "we want a schedule that allows the least amount of errors to propagate". Answer: PROD5B. No ordering of the current round's 12 CNOTs lets fewer errors through, under SI1000 or SD6.

**Method.** `oa_pair_score.py` computes the exact leading-order logical error of the memory benchmark (`noise_bench.memory`) from an op list.
- Moments and noise copy the generator: `moments()` is `schedule_moments`, and the noise locations and rates are those of `apply_circuit_level_noise`.
- A fault pair counts when the two faults sit at different locations, their readout flips and final G3/G4 cancel, and their combined data error shifts the logical result. Each pair is weighted by its probability.
- Logical error per round ≈ L2·p², where L2 = L(3 rounds) − L(2 rounds), summed over the X and Z runs. Faults pair only within one round or across two adjacent rounds, so L(K) is linear for K ≥ 2 (checked up to K = 4).
- Acceptance per round ≈ 1 − A1·p.
- Check against the 1M-shot benchmark (R = 1):
  - Ratio of PROD5B to the generator round: predicted 0.85 (SI1000) and 0.47 (SD6); measured 0.81–0.86 and 0.44–0.48.
  - Absolute value, generator round, X run: predicted 4.1e-3 at SD6 p = 0.02, measured 4.8e-3. The difference is higher order.

**Search.** `oa_schedule_search.py` enumerates every gate order on every wire and packs each one ASAP, as the generator does. That is 6^4 orders of the data on the ancillas times 384 orders of the ancillas on the data.
- 283,152 orders have no cycle.
- 31,104 give a correct round (24 of the 384 data orders).
- All 31,104 have 0 silent single faults.
- Depth ranges from 5 to 14 gate moments; 52 have 5.

`rank_schedules.py` → `schedule_search.out`:

| round | gate moments | L2 per round, SI1000 | L2 per round, SD6 | A1 per round, SI1000 / SD6 |
|---|---|---|---|---|
| generator round | 7 | 3.73 | 8.14 | 63.3 / 65.1 |
| PROD5 | 5 | 3.36 | 4.33 | 61.7 / 48.8 |
| **PROD5B** | 5 | **3.20 (x0.86)** | **4.15 (x0.51)** | 61.7 / 48.8 |
| median of the 31,104 | | 3.85 | 12.2 | |
| worst | 14 | 4.67 | 27.7 | |

- PROD5B has the lowest L2 of all 31,104 schedules under both SI1000 and SD6, so the SI1000/SD6 Pareto front is a single point.
- Three other orders tie with it exactly (listed in `breakdown.out`). The next best scores 3.28 / 4.24.
- PROD5B is already in `rounds.py`, passes the sdim checks and has the 1M-shot benchmark above.

**What still gets through** (`pair_breakdown.py` → `breakdown.out`). L2 per round, split by the two faults' noise types:

| fault pair | SI1000 generator | SI1000 PROD5B | SD6 generator | SD6 PROD5B |
|---|---|---|---|---|
| data idle in a reset/readout moment + the same | 2.00 | 2.00 | 0.50 | 0.50 |
| CNOT + data idle in a reset/readout moment | 0.94 | 0.75 | 0.47 | 0.38 |
| gate-moment idle + reset/readout-moment idle | 0.51 | 0.30 | 2.56 | 1.50 |
| CNOT + CNOT | 0.13 | 0.08 | 0.13 | 0.08 |
| CNOT + gate-moment idle | 0.12 | 0.06 | 1.20 | 0.56 |
| gate-moment idle + gate-moment idle | 0.03 | 0.01 | 3.28 | 1.13 |
| total | 3.73 | 3.20 | 8.14 | 4.15 |

- The first row is 2.00 (SI1000) and 0.50 (SD6) for every order; this was checked on random schedules and on the deepest one. It comes from two data qutrits idling through a reset or readout moment (2p each under SI1000) and picking up a weight-2 logical between them. The code has distance 2, so such a pair is never detected.
- Under SI1000, 2.75 of PROD5B's 3.20 involve that idle. The gate order can only act on the rest: from 1.73 down to 1.20 under SI1000, and from 7.64 down to 3.65 under SD6.
- These counts are for back-to-back rounds. In LRB-D a logical Clifford sits between rounds, which adds noise on the data but does not change which round is best here.

### The five-moment round in the generator: LRB-D pseudo-thresholds (2026-11-19)

PROD5B is called "the five-moment round" in everything user-facing; the published round is "the seven-moment round" (user rule, 2026-11-19). The user said "Yes, do it" to rerunning the one-ancilla SI1000 and SD6 LRB-D campaigns with it, and later chose it for the paper. Each new run copies a current run's Clifford sequences, grids, shots and noise, and its RB circuits byte for byte. Only the check round differs. Full write-up: `../../sdim140-prod5b/README.md`.

| noise | p* seven-moment, k = 1/2/3/4 | p* five-moment | ratio (paired 95%) |
|---|---|---|---|
| SI1000 | 0.0292 / 0.0239 / 0.0166 / 0.0132 | 0.0313 / 0.0258 / 0.0178 / 0.0144 | 1.07-1.09, every interval above 1.04 |
| SD6 | 0.0399 / 0.0269 / 0.0181 / 0.0155 | 0.0501 / 0.0326 / 0.0214 / 0.0177 | 1.25 / 1.21 / 1.18 / 1.14, every interval above 1.10 |

- The LRB error per Clifford drops by the same factor the round-only score predicted under SI1000 (about 0.85). Under SD6 it drops to 0.65-0.84 of the current value, less than the round-only 0.51.
- SD6 k = 2 has a dip in the five-moment curve at p = 0.045, which inflates its bootstrap SE to 0.0059. Without it the SE is 0.0008. Details are in the write-up.

### Files

All in `one-ancilla/`.
- `oa_frame_search.py`: data windows without transfers.
- `oa_transfer_search.py`: windows plus transfers, numba BFS; `--safe` keeps windows on the safe pairs.
- `oa_lower_bound.py`: relaxed bound and pruned exact pass for 3 and 4 data CNOTs.
- `oa_build.py`, `oa_finalists.py`: event sequences, random interleavings, scheduling, tableau fault screen.
- `oa_symbolic.py`: tableau screen of round correctness and single faults. On six earlier rounds it reproduces the sdim silent counts exactly: 0, 6, 20, 4, 0 and 2.
- `oa_verify_bench.py`: sdim noiseless and single-fault checks, and benchmarks.
- `oa_protocol_compat.py`, `oa_layout_facts.py`, `oa_encoder.py`: LRB-D compatibility, hardware facts, encoder search.
- `oa_fault_spread.py`: single-fault spread and exact count of silent fault pairs.
- `figure/`: review diagrams (`sq14-layout.tex`, `circuits.tex` from `draw_circuits.py`), not paper figures.
  - `three-layouts.pdf` (2026-11-19) is the newest one, from `draw_three_layouts.py` and `three-layouts-panels.tex`. It shows the paper's three layouts, then each layout's encoder and check circuit:
    - (a) the terminal check on IQM Garnet;
    - (b) the one-ancilla layout, with the five-moment round and the seven-moment round;
    - (c) the two-ancilla layout.
  - The panels are copied read-only from the figure session's `ancilla-layouts-B.tex`. Every circuit comes from the production code and is scheduled by the generator.
  - The script checks couplers, gate and moment counts, and noiseless sdim runs. Results are in `three-layouts-verify.out`: all pass.
- `prod_reschedule.py`: 5-moment schedules of the current round; 52 of 1620 have 0 silent faults. `rank_prod5.py` ranks them by silent fault pairs (`rank_prod5.out`).
- `rounds.py`: op lists of PROD5, PROD5B, SQ14, SQ01, SQ15A, SQ15B.
- `oa_pair_score.py`: exact leading-order acceptance and logical error of a round under the generator's SI1000/SD6 noise, from fault pairs; `breakdown()` splits it by noise type.
- `oa_schedule_search.py` → `schedule_search.pkl`, `schedule_search.log`: all 31,104 correct schedules of the current round, scored. `rank_schedules.py` → `schedule_search.out`. `pair_breakdown.py` → `breakdown.out`.
- Main outputs: `transfer_v1.out` (all pairs), `transfer_safe_L12.out`, `lower_bound_L2_le19.out`, `lower_bound_L3_le18.out`, `lower_bound_L4_le15.out`, `finalists_*.out`, `prod_reschedule_T5.out`, `verify_*.out`, `bench_*.out`.

**Not covered.** Windows left open across rounds, other generator sets, more than 4 ancillas, and ancilla-data gates other than the CNOTs of the ancilla's own type. Frame sets with 5 or more data CNOTs were bounded by counting only. The crosstalk caveats of the earlier sections apply here too.
