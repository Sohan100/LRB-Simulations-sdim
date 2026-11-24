# WP G: is distance 3 needed before correction beats detection?

Work package G of the qubit-512 study. Code: `src/lrb_qubit/d3_code.py`,
`src/lrb_qubit/d3_circuits.py`, `studies/correction-decoders/qubit/scripts/d3_compare.py`, tests in
`tests/test_qubit512_d3.py`, data in `studies/correction-decoders/qubit/results/G/`.
Labels: **[VA]** verified analytically, **[VN]** verified numerically,
**[LIT]** from literature (fetched), **[C]** conjecture.

## Summary

**Answer.** Yes: distance 3 is necessary, but it is not sufficient. With
correction (no rejection) the d=2 code never beats physical RB, while the d=3
code beats it only with a decoder that handles the fold-transversal S
hyperedges, parallel ancilla measurement, and p below about 1e-3 (p_RB = 5p)
or 2e-4 (p_RB = p). Detection keeps a far lower error per accepted round at
every p and both distances; what correction buys is zero rejection.

- **d=2, exact MLD (the best possible d=2 decoder):** the error per round is
  first order: r = (6.35 ± 0.06)p with parallel measurement and
  (11.9 ± 0.1)p with the repo's sequential layers (fitted slope
  k = 1.02 ± 0.01). That is 3.7-8.5 times physical RB at p_RB = 5p
  (r_phys = 0.339 p_RB = 1.69p) and 19-42 times physical RB at p_RB = p, at
  every p in {1e-4, ..., 5e-3}. **[VN]**
- **d=3, BP-OSD, parallel measurement:** second order, r = A p^k with
  k = 1.88 ± 0.09 and A = (1.52 ± 0.08)e3. It crosses physical RB at
  **p* = 1.17e-3 (16-84% bootstrap 1.11e-3 - 1.22e-3) for p_RB = 5p** and
  **p* = 1.94e-4 (1.72e-4 - 2.17e-4) for p_RB = p**. A fit on depths m >= 10
  only gives the lower values 0.89e-3 and 1.76e-4, because the decay steepens
  with depth. **[VN]**
- **d=3, BP-OSD, the repo's sequential single-ancilla layers:** each round
  puts 13 `DEPOLARIZE1(2p)` idle layers on every data qubit (about 26p) instead
  of 2. This gives A = (1.48 ± 0.02)e4, ten times larger, and r is already
  1.13 ± 0.04 times physical (5p) at p = 1e-4: **no pseudo-threshold on the
  grid** (extrapolated p* ≈ 0.9e-4 for 5p and ≈ 1.4e-5 for p_RB = p, **[C]**).
  **[VN]**
- **PyMatching (the requested primary decoder) is not distance-preserving
  here.** Stim's graphlike decomposition of the hyperedges created by the
  fold-transversal S needs observable-only pieces, so MWPM misdecodes single
  faults: about 8-25p of first-order failure per circuit with parallel
  measurement and 25-100p with sequential. The d=3 PyMatching error per round
  is first order (k = 1.14 ± 0.02 parallel, 1.28 ± 0.01 sequential), is
  2.7-30 times physical RB (5p) and never crosses it. **[VN]** This agrees
  with Serra-Peralta, Shaw, Terhal 2025, who need "logical observable"
  matching for these gates **[LIT]**.
- **Detection vs correction at the same p (p = 1e-3, parallel, m = 22):**
  corrected d=3 has F = 0.932 ± 0.008 with no rejection
  (r = (1.43 ± 0.10)e-3). LRB-D unif=1 has accepted F = 0.993 ± 0.0002 at d=2
  (77.9% rejected) and 0.990 ± 0.002 at d=3 (98.8% rejected); its accepted
  r ≤ 2.2e-5 is about 60 times lower. The d=3 checks make LRB-D reject more,
  not less. **[VN]**
- **The cheapest "correction" here uses no checks at all.** In the repo's
  const=0 circuit (no check rounds, ideal final X readout) with exact MLD on
  the six final X syndromes, d=3 gives r = (232 ± 8) p^2 (k = 1.87 ± 0.07).
  That is below physical RB (5p) on the whole grid, and below physical RB
  (p_RB = p) for p < 1.60e-3 (1.55e-3 - 1.70e-3). In this noise model every
  check round costs more than it corrects. At d=2 the same no-check decoder
  gives r = 0.73p: 0.43 times physical (5p), 2.2 times physical (p). **[VN]**
  [C] This relies on the protocol's ideal terminal readout and on SI1000 having
  no idle noise during gate layers.
- **Construction checks [VN]:** the [[13,1,3]] code, its fold symmetry and its
  fold-transversal H and S are checked with Stim tableaus. The circuit-level
  distance is 3 for both schedules with the repo-derived CNOT order, but 32 of
  the 576 uniform CNOT orders drop it to 2. The generic builder reproduces the
  lead's d=2 circuits exactly, and my d=2 numbers agree with
  `results/grid_main` within errors.
- **Lead's model:** no bug found. `lrb_qubit` circuits, policies and the MLD
  are reproduced exactly by the generic code (tests). The lead's 64 tests
  still pass.

## 1. The distance-3 member of the family

The repo's qubit [[5,1,2]] is the d=2 unrotated planar surface code that is
symmetric under the diagonal fold. `d3_code.UnrotatedSurfaceCode(d)` builds any
member from its geometry: sites (r, c) with 0 <= r, c <= 2d-2, data at r+c
even, Z checks ("plaquettes") at (even, odd), X checks ("stars") at
(odd, even), each check on its nearest data neighbours. `X_L` is X on row 0
and `Z_L` is Z on column 0. `CODE13 = UnrotatedSurfaceCode(3)` is the
[[13,1,3]] code: 13 data qubits, 6 + 6 checks (weights 3 and 4), one ancilla
per check (wires 13-24), `X_L = X0 X1 X2`, `Z_L = Z0 Z5 Z10` (row-major
labels, qubit q at site `code.site_of[q]`).

Fold-transversal gates (Moussa 2016 [LIT]; Breuckmann and Burton 2024 [LIT]):

- logical H = SWAP on the four mirror pairs (q, q^T) = (1,5), (2,10), (4,8),
  (7,11), then H on all 13 data qubits;
- logical S = S on the diagonal sites (0,0), (2,2), (4,4) (qubits 0, 6, 12),
  S-dagger on (1,1), (3,3) (qubits 3, 9), and CZ on the four mirror pairs.

Why the S/S-dagger pattern alternates [VA]: under CZ on mirror pairs and
S^(+-1) on the fold line, an X check A_v maps to A_v B_{v^T} times
i^(#S) (-i)^(#S-dagger) over the diagonal qubits in its support. Every X check of
the unrotated code touches either no diagonal site or two consecutive ones,
(c, c) and (c+1, c+1), so alternating S and S-dagger cancels the phase. S on
the corner (0,0) makes `X_L -> +Y_L`.

Checks with Stim tableaus (`tests/test_qubit512_d3.py`) [VN]:

- the 12 checks commute, are independent (rank 12) and commute with `X_L`,
  `Z_L`, which anticommute; no Pauli of weight 1 or 2 is a nontrivial
  logical (exhaustive), and `X_L` has weight 3, so d = 3;
- the transpose maps data to data, Z-check supports exactly onto X-check
  supports, `X_L` onto `Z_L`; it fixes the 5 diagonal qubits and pairs the
  other 8;
- every stabilizer maps to a stabilizer with a + sign under H, S, X, Z, I and
  the inverse tokens; H: `X_L -> +Z_L`, `Z_L -> +X_L`; S: `X_L -> +Y_L`,
  `Z_L -> +Z_L`; S-dagger: `X_L -> -Y_L` (also checked at d = 4);
- all 24 Clifford descriptors act on (X_L, Z_L) like the single-qubit
  Clifford and their mirror inverses invert them exactly;
- the Stim-built ideal encoder prepares |+>_L;
- `UnrotatedSurfaceCode(2, labels=FIVE_QUBIT_LABELS)` reproduces the lead's
  [[5,1,2]] exactly: stabilizer strings, logicals, check CNOT lists, and the
  S, X, Z gate lists; its H applies the same gates with the H gates in a
  different order (same tableau).

The CNOT order inside each check is the compass order that reproduces the
lead's [[5,1,2]] checks exactly (Z checks N, W, S, E; X checks N, W, E, S).

Fallback (rotated [[9,1,3]]) was not needed.

## 2. Circuits

`d3_circuits.build_lrb_circuit(code, cliffords, depth, p, schedule=...)`
generalizes the lead's builder to any code object:

- ideal encoding of |+>_L; for each forward Clifford the logical gates, a full
  noisy check round and an ancilla reset; the mirror inverses
  C_m^-1 ... C_1^-1 with no checks; a final check round (no reset); an ideal
  X readout of every data qubit;
- SI1000 noise via the lead's `circuits.emit_operations` and `SI1000`
  (`DEPOLARIZE1(0.1p)` after 1-qubit gates, `DEPOLARIZE2(p)` after CX, CZ and
  SWAP, per measurement layer `DEPOLARIZE1(2p)` on all other wires plus
  `DEPOLARIZE1(p)`, `X_ERROR(5p)` on the measured wire, reset layer
  `DEPOLARIZE1(2p)` on all other wires and `X_ERROR(2p)`);
- the lead's difference detectors (stabilizer s in round r against the
  previous round's generators equal to s after back-propagation through the
  logical gates) plus final X-check detectors from the data readout;
  observable = parity of the `X_L` support;
- `schedule="sequential"`: the repo layout, each check entangled and measured
  on its own: 12 single-ancilla measurement layers per round (13 idle
  `DEPOLARIZE1(2p)` layers per data qubit per round with the reset layer);
- `schedule="parallel"`: identical gates and order, all 12 ancillas measured
  in one layer (2 idle layers per round).

[VN] With `LeadFiveQubitCode` (an adapter over `lrb_qubit.code`) the
sequential builder produces exactly the lead's `build_lrb_circuit(...,
terminal="data5")` (identical Stim text at depths 0, 1, 3, 22). [VN]
Noiseless d=3 circuits give no detection events and no observable flips at
depths 0-22 for both schedules; the idle-layer count per round is 13
(sequential) and 2 (parallel).

LRB-D uses the raw per-round syndromes (`raw_syndromes`, checked against the
measurement records) with `baseline.used_rounds` (const 1-22, unif 1-22),
`_final` variants additionally require clean final data detectors, and
`const0` is the separate no-check circuit postselected on the ideal final X
checks or decoded exactly (`const0_mld`, brute force over the 2^6 final
syndromes at d=3).

## 3. Decoders

- `pm`: PyMatching 2.3.1 on Stim's graphlike decomposition (the requested
  primary decoder).
- `pmf`: PyMatching on an observable-frame subgraph: in each round keep only
  the detectors that see error components able to flip the final `X_L` (a
  single-observable simplification of logical-observable matching,
  Serra-Peralta, Shaw, Terhal 2025 [LIT]). Diagnostic only.
- `bposd`: `ldpc` 2.4.1 `BpOsdDecoder` (product-sum BP, 30 iterations,
  OSD-CS order 10) on the full hypergraph DEM, with one fix described in
  Section 4.2.
- `mld` (d=2 only): the lead's exact transfer-matrix MLD.
- physical RB: exact, `F = (1 - 4q/3)^N` with q = 0.1 p_RB and N the gate
  count (depolarizing noise commutes with the gates) [VA]; agrees with
  sampling the lead's `build_physical_rb_circuit` [VN]. Ensemble decay per
  depth step `f = mean_C (1 - 4q/3)^(2 len C)` over the 24 Cliffords (61 gates
  in total), so `r_phys = (1 - f)/2 = 0.339 p_RB` to first order [VA].

## 4. Structural results

### 4.1 Circuit-level distance and CNOT order

`d3_compare.py checks` (`results/G/checks.json`, `results/G/order_scan_bad.json`):

| check | sequential | parallel |
|---|---|---|
| `search_for_undetectable_logical_errors` (hyperedges allowed; detection-event sets <= 4), grid sequences 0 and 1 and an S/H-rich sequence at depths 1, 3, 6 | 3 in all 9 cases | 3 in all 9 cases |
| `shortest_graphlike_error(ignore_ungraphlike_errors=True)` | 3 in all 9 cases | 3 in all 9 cases |
| `shortest_graphlike_error` on Stim's decomposed DEM | 1 in all 9 cases | 1 in all 9 cases |
| uniform CNOT orders (same order for all checks of a type) with circuit distance 3 | 544 / 576 | 544 / 576 |

- [VN] With the repo-derived order (Z checks N, W, S, E; X checks N, W, E, S)
  the circuit-level distance is 3 for both schedules.
- [VN] 32 of the 576 uniform orders drop the circuit distance to 2, the same
  32 for both schedules. In all of them both check types start with two
  opposite neighbours (NS, SN, WE or EW first), so the two-qubit hooks are
  straight pairs. For the order NSWE/NSWE (parallel schedule) the distance
  is still 3 with I-only, H-only or S-only sequences and drops to 2 only with
  mixed S/H words.
  [C] The failing pairs combine a straight X hook and a straight Z hook whose
  images under the logical frame changes line up with a weight-3 logical. My
  first, analytic argument (no hook is contained in a minimum-weight row or
  column, so every order is safe) ignored such cross-type combinations and is
  wrong.
- [VN] The graphlike distance after Stim's decomposition is 1: across a
  fold-transversal S, Stim decomposes some hyperedges into graphlike pieces
  plus an observable-only piece (for example
  `error(2.7e-4) D12 D18 L0 ^ D1 ^ L0`). PyMatching ignores the `L0` piece
  in plain mode and rejects it with `enable_correlations=True` ("decomposed
  error instruction with an undetectable component").

### 4.2 Which decoders correct every single fault?

`single_fault_report` decodes the syndrome of every DEM mechanism and lists
those decoded wrongly; their total probability is the first-order coefficient
of the logical error. Values at p = 1e-3 (`checks.json`) [VN]:

| schedule | sequence | depth | mechanisms | PyMatching: failing (prob/p) | frame matching | BP-OSD (with fallback) |
|---|---|---|---|---|---|---|
| sequential | grid0 | 2 | 568 | 45 (24.64p) | 32 (5.80p) | 4 (0.40p) |
| sequential | grid0 | 6 | 1474 | 112 (75.92p) | 79 (12.27p) | 7 (0.80p) |
| sequential | grid1 | 6 | 1544 | 136 (98.60p) | 83 (13.27p) | 7 (0.80p) |
| sequential | only Paulis | 6 | 1295 | 0 | 0 | 1 (0.07p) |
| sequential | only H | 6 | 1536 | 16 (1.60p) | 16 (1.60p) | 4 (0.27p) |
| sequential | only S | 6 | 1546 | 175 (142.96p) | 108 (15.93p) | 16 (1.07p) |
| parallel | grid0 | 2 | 568 | 23 (8.40p) | 12 (4.47p) | 0 |
| parallel | grid0 | 6 | 1474 | 78 (22.86p) | 26 (9.00p) | 0 |
| parallel | grid1 | 6 | 1544 | 98 (24.36p) | 31 (9.33p) | 0 |
| parallel | only Paulis | 6 | 1295 | 0 | 0 | 0 |
| parallel | only H | 6 | 1536 | 0 | 0 | 0 |
| parallel | only S | 6 | 1546 | 155 (42.16p) | 42 (12.80p) | 0 |

- [VN] PyMatching on Stim's decomposition corrects every single fault for
  Pauli-only sequences, and for H-only sequences with parallel measurement.
  It fails across every fold-transversal S, and across H with the sequential
  schedule (mid-round errors). The first-order term grows with the number of
  S gates: about 8-25p per circuit for the grid sequences with parallel
  measurement, 25-100p with sequential measurement, against
  `r_phys = 1.69p` per round for physical RB at 5p. [LIT] This is the failure
  mode described by Serra-Peralta, Shaw and Terhal (MWPM cannot decode the
  weight-3 hyperedges after S and CNOT; they introduce logical-observable
  matching).
- [VN] The frame-restricted matcher (`pmf`) removes most but not all of them.
  The remaining conflicts are genuine ambiguities of the restricted graph: for
  example a hook of the `S_X32` check before an S gate and a Y error on (0,4)
  give the same X-type syndrome after the S but differ by `Z_L`; only the
  dropped Z-type detectors separate them. A correct logical-observable matcher
  needs the full construction of Serra-Peralta et al. (open item).
- [VN] Plain `ldpc` BP-OSD (30 iterations) misdecodes some single faults even
  at p = 1e-5: `ldpc` skips OSD whenever BP's hard decision satisfies the
  syndrome, and BP sometimes "converges" to a valid but absurd solution
  (5-13 mechanisms, prior about 1e-28 to 1e-65). `BpOsdDEMDecoder` therefore
  also tries a 2-iteration BP-OSD (OSD then nearly always runs) and the most
  likely single mechanism with that syndrome whenever BP converged, and keeps
  the most likely candidate. With this fix BP-OSD corrects every single fault
  for both schedules at p = 1e-4 (depth 3: 839 mechanisms; depth 10, grid
  sequence 0: 2378 mechanisms) and at p = 1e-5 (depth 3). The few
  failures at p = 1e-3 with the sequential schedule are rare mechanisms (CZ
  or CX components of probability p/15) whose syndromes have likelier
  two-fault explanations once the sequential idle noise (about 26p per data
  qubit per round) is included; [C] exact MLD would decode those the same way.
- [VN] On sampled shots the fallback costs nothing in accuracy (p = 1e-3,
  depth 10 parallel: 0.0088 ± 0.0024 with plain 30-iteration BP-OSD,
  0.0118 ± 0.0028 with 2 iterations, 0.0094 ± 0.0025 with the more likely of
  the two; 1500 shots, binomial SE).

## 5. LRB results

Setup: the main grid's 30 Clifford sequences (seed 20261028), depths
0, 2, 4, 6, 10, 14, 18, 20, 22, p in {1e-4, 5e-4, 1e-3, 2e-3, 5e-3}.

- LRB-D, PyMatching, frame matching and const0 use all 30 sequences, with
  3e4 shots per circuit at d=3 and 4e4 at d=2 (exact MLD on all 4e4).
- BP-OSD uses sequences 0-9 with time-budgeted decoding (Section 9).
  Decoded shots summed over sequences at m = 0 / 10 / 22:
  - parallel: 30000 / 217400 / 65325 (p = 1e-4), 12000 / 15450 / 9725
    (5e-4), 6000 / 8325 / 4150 (1e-3), 3000 / 4175 / 3475 (2e-3),
    1500 / 800 / - (5e-3, depths <= 10 only);
  - sequential: 30000 / 260000 / 72425, 12000 / 13725 / 5000,
    6000 / 7125 / 2750, 3000 / 3000 / 350, 1500 / 1500 / -.

Fits: F = b f^m on the sequence-mean F with the repo's weighted grid scan
(offset 0, points with F >= 1 dropped); physical RB is exact. Error per round
r = (1 - f)/2. Uncertainties are bootstrap standard errors over sequences
(300 replicates), which include shot noise. All numbers come from
`results/G/summary.json`; more methods are in `results/G/tables.md`, and
repo-format CSVs are in `results/G/csv/`.

### 5.1 Error per round r (fit on all depths)

| method | p = 1e-4 | p = 5e-4 | p = 1e-3 | p = 2e-3 | p = 5e-3 |
|---|---|---|---|---|---|
| physical RB, p_RB = 5p (exact) | 1.69e-04 | 8.47e-04 | 1.69e-03 | 3.38e-03 | 8.40e-03 |
| physical RB, p_RB = p (exact) | 3.39e-05 | 1.69e-04 | 3.39e-04 | 6.77e-04 | 1.69e-03 |
| d=3 parallel, BP-OSD | 1.76e-05 ± 3.5e-06 | 4.33e-04 ± 3.9e-05 | 1.43e-03 ± 9.9e-05 | 5.91e-03 ± 3.0e-04 | 2.77e-02 ± 1.3e-03 |
| d=3 sequential, BP-OSD | 1.92e-04 ± 6.4e-06 | 3.97e-03 ± 1.2e-04 | 1.45e-02 ± 2.2e-04 | 5.37e-02 ± 2.1e-03 | 1.89e-01 ± 8.9e-03 |
| d=3 parallel, PyMatching | 4.62e-04 ± 2.0e-05 | 2.71e-03 ± 9.2e-05 | 6.40e-03 ± 1.6e-04 | 1.64e-02 ± 3.0e-04 | 6.19e-02 ± 7.2e-04 |
| d=3 sequential, PyMatching | 1.79e-03 ± 5.4e-05 | 1.28e-02 ± 2.3e-04 | 3.31e-02 ± 3.7e-04 | 8.79e-02 ± 8.7e-04 | 2.56e-01 ± 4.4e-03 |
| d=3 parallel, frame matching | 1.38e-04 ± 5.6e-06 | 9.60e-04 ± 2.3e-05 | 2.79e-03 ± 5.2e-05 | 8.05e-03 ± 1.2e-04 | 4.16e-02 ± 5.4e-04 |
| d=3 sequential, frame matching | 5.72e-04 ± 1.3e-05 | 6.40e-03 ± 9.2e-05 | 2.17e-02 ± 3.1e-04 | 6.96e-02 ± 7.9e-04 | 2.37e-01 ± 4.5e-03 |
| d=2 parallel, exact MLD | 6.35e-04 ± 5.8e-06 | 3.25e-03 ± 3.6e-05 | 6.61e-03 ± 6.6e-05 | 1.36e-02 ± 1.3e-04 | 3.69e-02 ± 3.4e-04 |
| d=2 sequential, exact MLD | 1.19e-03 ± 1.3e-05 | 6.11e-03 ± 7.3e-05 | 1.25e-02 ± 1.3e-04 | 2.61e-02 ± 2.6e-04 | 7.14e-02 ± 6.5e-04 |
| d=2 sequential, PyMatching | 2.45e-03 ± 2.6e-05 | 1.22e-02 ± 1.4e-04 | 2.41e-02 ± 2.5e-04 | 4.76e-02 ± 4.9e-04 | 1.13e-01 ± 1.3e-03 |
| d=3 const0 (no checks), exact MLD | 3.16e-06 ± 5.3e-07 | 5.97e-05 ± 3.0e-06 | 2.26e-04 ± 1.1e-05 | 8.24e-04 ± 4.2e-05 | 4.27e-03 ± 1.7e-04 |
| d=2 const0 (no checks), exact MLD | 7.34e-05 ± 1.8e-06 | 3.65e-04 ± 7.1e-06 | 7.30e-04 ± 1.4e-05 | 1.47e-03 ± 2.9e-05 | 3.66e-03 ± 6.9e-05 |
| d=3 parallel LRB-D unif=1 (accepted) | 1.5e-06 ± 1.2e-06 | 2e-07 ± 2.6e-06 | 0 ± 4.4e-06 | 2.5e-05 ± 3.3e-05 | 5.3e-04 ± 2.3e-04 |
| d=2 parallel LRB-D unif=1 (accepted) | 0 ± 4.3e-07 | 2.5e-06 ± 1.4e-06 | 2.18e-05 ± 3.2e-06 | 6.5e-05 ± 1.1e-05 | 5.7e-04 ± 6.1e-05 |
| d=2 sequential LRB-D unif=1 (accepted) | 7e-07 ± 9e-07 | 1.54e-05 ± 3.0e-06 | 5.92e-05 ± 6.7e-06 | 2.77e-04 ± 1.9e-05 | 2.19e-03 ± 1.5e-04 |
| d=3 parallel LRB-D const=1 (accepted) | 1.5e-06 ± 1.0e-06 | 1.63e-05 ± 3.2e-06 | 1.01e-04 ± 7.4e-06 | 4.56e-04 ± 3.3e-05 | 3.04e-03 ± 1.9e-04 |
| d=2 parallel LRB-D const=1 (accepted) | 3.2e-06 ± 6.9e-07 | 9.14e-05 ± 2.7e-06 | 3.99e-04 ± 1.4e-05 | 1.60e-03 ± 7.4e-05 | 1.07e-02 ± 6.7e-04 |
| grid_main d=2 sequential mld5 (lead) | 1.19e-03 ± 1.3e-05 | 6.09e-03 ± 7.0e-05 | 1.25e-02 ± 1.3e-04 | 2.61e-02 ± 2.7e-04 | 7.12e-02 ± 5.6e-04 |
| grid_main d=2 sequential unif=1 (lead) | 4.5e-07 ± 1.5e-07 | 1.39e-05 ± 9.6e-07 | 5.56e-05 ± 1.5e-06 | 2.53e-04 ± 6.1e-06 | 2.01e-03 ± 5.4e-05 |

Fits on m >= 10 only (`tables.md`) agree with the table for the first-order
decoders. For BP-OSD the late fits are larger, which shows the decay
steepening with depth: d=3 parallel gives 1.83e-5, 5.27e-4, 1.86e-3 and
6.65e-3 at p = 1e-4 ... 2e-3. A likely cause is the 2m check-free inverse
Cliffords accumulating errors before the final round [C].

### 5.2 Scaling and pseudo-thresholds

Power-law fits r = A p^k over p <= 1e-3 (weighted by the bootstrap SE); A is
taken from the p = 5e-4 and 1e-3 points with k = 2:

| code, schedule, decoder | slope k | A (r = A p^2) |
|---|---|---|
| d=3 parallel BP-OSD | 1.88 ± 0.09 | (1.52 ± 0.08)e3 |
| d=3 sequential BP-OSD | 1.88 ± 0.02 | (1.48 ± 0.02)e4 |
| d=3 parallel const0 exact MLD | 1.87 ± 0.07 | (2.32 ± 0.08)e2 |
| d=3 parallel PyMatching | 1.14 ± 0.02 | first order |
| d=3 sequential PyMatching | 1.28 ± 0.01 | first order |
| d=3 parallel frame matching | 1.33 ± 0.02 | mixed |
| d=2 parallel exact MLD | 1.02 ± 0.01 | first order, r = 6.35p |
| d=2 sequential exact MLD | 1.02 ± 0.01 | first order, r = 11.9p |
| d=2 const0 exact MLD | 1.00 ± 0.01 | first order, r = 0.73p |

Pseudo-threshold p*: where the corrected r equals the exact physical r, by
log-log interpolation between grid points. The bootstrap resamples the
per-p fits; all p* intervals below are 16-84% bootstrap ranges.

| code, schedule, decoder | p* (p_RB = 5p) | p* (p_RB = p) | r / r_phys(5p) at 1e-4, 5e-4, 1e-3, 2e-3, 5e-3 |
|---|---|---|---|
| d=3 parallel BP-OSD | **1.17e-3** (1.11e-3 - 1.22e-3) | **1.94e-4** (1.72e-4 - 2.17e-4) | 0.10, 0.51, 0.85, 1.7, 3.3 |
| d=3 sequential BP-OSD | none on grid (ratio 1.13 at 1e-4) | none on grid (5.7 at 1e-4) | 1.1, 4.7, 8.6, 16, 23 |
| d=3 parallel PyMatching | none (never below) | none | 2.7, 3.2, 3.8, 4.9, 7.4 |
| d=3 sequential PyMatching | none | none | 11, 15, 20, 26, 30 |
| d=3 parallel frame matching | 2.72e-4 (2.49e-4 - 3.06e-4) | none | 0.81, 1.1, 1.6, 2.4, 5.0 |
| d=3 parallel const0 exact MLD | below on whole grid | 1.60e-3 (1.55e-3 - 1.70e-3) | 0.019, 0.071, 0.13, 0.24, 0.51 |
| d=3 sequential const0 exact MLD | below on whole grid | 1.66e-3 (1.60e-3 - 1.76e-3) | 0.015, 0.068, 0.13, 0.24, 0.51 |
| d=2 parallel exact MLD | none | none | 3.7, 3.8, 3.9, 4.0, 4.4 |
| d=2 sequential exact MLD | none | none | 7.0, 7.2, 7.4, 7.7, 8.5 |
| d=2 const0 exact MLD | below on whole grid (ratio 0.43) | none (ratio 2.2) | 0.43 at every p |

With p* = A^-1 c (c = 0.339 * 5 or 0.339) the k = 2 model gives
1.12e-3 ± 0.06e-3 and 2.23e-4 ± 0.12e-4 for d=3 parallel BP-OSD, consistent
with the interpolation. The frame-matching crossing at 2.7e-4 is an artifact:
its first-order coefficient (r/p about 1.2-1.4 at low p) happens to sit below
1.69, and it never beats physical RB at p_RB = p.

### 5.3 F(22) and rejection (± SE over sequences)

| method | p = 1e-4 | p = 5e-4 | p = 1e-3 | p = 2e-3 | p = 5e-3 |
|---|---|---|---|---|---|
| physical RB 5p, F(22) (exact) | 0.9926 | 0.9635 | 0.9283 | 0.8617 | 0.6892 |
| physical RB p, F(22) (exact) | 0.9985 | 0.9926 | 0.9852 | 0.9707 | 0.9283 |
| d=3 parallel BP-OSD, F(22) | 0.9990 ± 0.0002 | 0.978 ± 0.002 | 0.932 ± 0.008 | 0.768 ± 0.013 | - |
| d=3 sequential BP-OSD, F(22) | 0.9910 ± 0.0004 | 0.837 ± 0.007 | 0.491 ± 0.021 | 0.12 ± 0.04 | - |
| d=3 parallel PyMatching, F(22) | 0.9803 ± 0.0008 | 0.890 ± 0.003 | 0.755 ± 0.005 | 0.481 ± 0.007 | 0.054 ± 0.002 |
| d=2 parallel exact MLD, F(22) | 0.9715 ± 0.0004 | 0.865 ± 0.001 | 0.742 ± 0.002 | 0.541 ± 0.003 | 0.182 ± 0.003 |
| d=2 sequential exact MLD, F(22) | 0.9466 ± 0.0006 | 0.757 ± 0.002 | 0.563 ± 0.003 | 0.296 ± 0.004 | 0.032 ± 0.001 |
| d=3 const0 exact MLD, F(22) | 0.99990 ± 0.00002 | 0.9966 ± 0.0001 | 0.9876 ± 0.0005 | 0.957 ± 0.002 | 0.812 ± 0.005 |
| d=3 parallel unif=1: F(22) / rejection(22) | 0.9991 / 0.3604 | 0.9950 / 0.8931 | 0.990 / 0.9884 | 0.99 / 0.9999 | - / 1.0000 |
| d=3 sequential unif=1: F(22) / rejection(22) | 0.9958 / 0.7271 | 0.972 / 0.9984 | - / 1.0000 | - / 1.0000 | - / 1.0000 |
| d=2 parallel unif=1: F(22) / rejection(22) | 0.9994 / 0.1396 | 0.9967 / 0.5286 | 0.9926 / 0.7787 | 0.985 / 0.9506 | 0.97 / 0.9994 |
| d=2 sequential unif=1: F(22) / rejection(22) | 0.9991 / 0.2120 | 0.9947 / 0.6939 | 0.9885 / 0.9060 | 0.969 / 0.9910 | - / 1.0000 |
| d=3 parallel const=1: F(22) / rejection(22) | 0.9990 / 0.2097 | 0.9945 / 0.6898 | 0.9852 / 0.9011 | 0.932 / 0.9888 | 0.36 / 0.9997 |
| d=2 parallel const=1: F(22) / rejection(22) | 0.9993 / 0.0768 | 0.9922 / 0.3260 | 0.9723 / 0.5388 | 0.881 / 0.7648 | 0.300 / 0.9228 |

LRB-D standard errors are in `tables.md` (F: 4e-5 to 0.05; rejection:
< 1e-3). "-" means no accepted shot (or no BP-OSD decoding at m > 10 for
p = 5e-3).

### 5.4 Comparison with the d=2 main grid

My d=2 sequential runs use the lead's circuits (identical Stim text) on new
shots. They agree with `results/grid_main` within errors:

- MLD r = 1.19e-3 ± 1.3e-5 here vs 1.19e-3 ± 1.3e-5 for mld5 at p = 1e-4;
  1.25e-2 here vs 1.25e-2 at 1e-3;
- unif=1 accepted r = (5.92 ± 0.67)e-5 here vs (5.56 ± 0.15)e-5 at 1e-3;
- const0_mld r = (7.42 ± 0.17)e-5 vs (7.33 ± 0.14)e-5 at 1e-4. **[VN]**

The parallel-measurement d=2 variant halves the d=2 MLD error
(6.35p vs 11.9p) but stays first order.

## 6. Interpretation

1. At d=2 the single-qubit errors X0/X3, X2/X4, Z0/Z2 and Z3/Z4 share
   syndromes pairwise and each pair differs by a weight-2 logical, so any
   decoder must guess within a pair and correction is first order [VA,
   CONTEXT.md]. The measured slopes
   (1.02 ± 0.01) confirm it [VN]. With 4-13 idle layers per round the
   coefficient (6-12p per round) is far above the physical 0.34 p_RB per
   step.
2. At d=3 the code corrects every single fault in principle (circuit
   distance 3 [VN]). Whether that shows up depends on the decoder: MWPM on
   Stim's decomposition loses it across the fold-transversal S, BP-OSD keeps
   it [VN].
3. The idle noise of measurement layers dominates the check cost. Per round,
   each data qubit gets 13 x 2p (sequential) or 2 x 2p (parallel) of
   depolarizing, plus about 3p from its CNOTs. That makes A ten times larger
   for the sequential layout [VN], and in the sequential layout no pseudo-
   threshold exists on the grid even for the best decoder.
4. Detection still gives the lowest error per accepted round at both
   distances, because rejecting any nontrivial syndrome removes all
   single-fault and most double-fault events. Its acceptance decays
   exponentially, faster at d=3 (12 checks per round) than at d=2 [VN].
   Correction at d=3 trades a 10^1.5-10^2 higher error per round for full
   acceptance.
5. With an ideal final readout, the no-check const0 circuit decoded once at
   the end is the best corrected strategy at d=3 (A = 232 vs 1.5e3). The
   check rounds inject more noise than they remove within one logical step
   [VN]. [C] This advantage would shrink with idle noise during gate layers or
   a noisy terminal readout, neither of which the repo's SI1000 includes.

## 7. Claims

| # | claim | label | evidence |
|---|---|---|---|
| 1 | The [[13,1,3]] unrotated planar code (13 data, 12 checks, X_L row 0, Z_L column 0) has distance 3. | VN | `test_parameters_and_commutation`, `test_code_distance_is_three` (exhaustive weight <= 2) |
| 2 | The code is symmetric under the diagonal fold (Z checks map onto X checks, X_L onto Z_L; 5 fixed and 4 paired qubits). | VN | `test_diagonal_fold_symmetry` |
| 3 | Logical H = SWAP on mirror pairs then transversal H; logical S = S on (0,0),(2,2),(4,4), S-dagger on (1,1),(3,3), CZ on mirror pairs; all stabilizers map with + signs. | VN, VA | `test_fold_transversal_logical_gates` (also d=4), `test_fold_gate_structure`; phase argument in Section 1 |
| 4 | Folded surface codes have transversal (fold-transversal) Clifford gates, and folding generalizes to CSS codes with dualities; our H and S are the d=3 instance (gate forms verified by me numerically, claim 3). | LIT | J. E. Moussa, "Transversal Clifford gates on folded surface codes", PRA 94, 042316 (2016), arXiv:1603.02286; N. P. Breuckmann, S. Burton, "Fold-Transversal Clifford Gates for Quantum Codes", Quantum 8, 1372 (2024), arXiv:2202.06647 (both abstracts fetched) |
| 5 | The d=2 member with the lead's labels is the lead's [[5,1,2]], and the generic builder reproduces `build_lrb_circuit` exactly. | VN | `test_distance_two_member_is_the_lead_code`, `test_generic_builder_reproduces_lead_circuit` |
| 6 | With the repo-derived CNOT order the circuit-level distance is 3 for sequential and parallel measurement. | VN | `test_circuit_level_distance_is_three`, `checks.json` (18 cases) |
| 7 | 32 of 576 uniform CNOT orders (both check types starting with two opposite neighbours) give circuit distance 2 once S and H are mixed; the mechanism is two straight hooks of opposite type. | VN (count, pattern); C (mechanism) | `order_scan_bad.json`, Stim explanation in Section 4.1 |
| 8 | PyMatching on Stim's decomposition misdecodes single faults across fold-transversal S (8-25p per circuit parallel, 25-100p sequential) and across H in the sequential schedule; it is exact for Pauli-only sequences. | VN | `checks.json`, `test_matching_fails_single_faults_across_fold_transversal_s`, `test_matching_corrects_single_faults_without_s_gates` |
| 9 | This is the known failure of MWPM across transversal S/CNOT in the unrotated surface code. | LIT | Serra-Peralta, Shaw, Terhal, "Decoding across transversal Clifford gates in the surface code", arXiv:2505.13599 (abstract and HTML fetched) |
| 10 | Plain ldpc BP-OSD skips OSD after a false BP convergence and then misdecodes single faults; with the fallbacks in `BpOsdDEMDecoder` all single faults are corrected at p <= 1e-4. | VN | `test_plain_bposd_can_converge_to_absurd_solutions`, `test_bposd_corrects_every_single_fault`, Section 4.2 |
| 11 | d=2 correction (exact MLD) is first order, r = (6.35 ± 0.06)p parallel and (11.9 ± 0.1)p sequential, and never beats physical RB at p_RB = 5p or p on the grid. | VN | Section 5.1-5.2 |
| 12 | d=3 correction with BP-OSD and parallel measurement is second order (k = 1.88 ± 0.09) and beats physical RB below p* = 1.17e-3 (1.11e-3 - 1.22e-3) for p_RB = 5p and below p* = 1.94e-4 (1.72e-4 - 2.17e-4) for p_RB = p. | VN | Section 5.2 |
| 13 | With the repo's sequential single-ancilla measurement layers, d=3 BP-OSD has A ten times larger and no pseudo-threshold on the grid (ratio 1.13 ± 0.04 at p = 1e-4 vs 5p). | VN | Section 5.1-5.2 |
| 14 | Extrapolated sequential pseudo-thresholds: about 0.9e-4 (5p) and 1.4e-5 (p). | C | extrapolation with k = 1.88 |
| 15 | d=3 correction with PyMatching never beats physical RB on the grid (first order, k = 1.14-1.28). | VN | Section 5.2 |
| 16 | LRB-D keeps a lower error per accepted round than any corrected scheme at both distances, and d=3 LRB-D rejects more than d=2 (98.8% vs 77.9% at m=22, p=1e-3, parallel). | VN | Section 5.3 |
| 17 | The no-check const0 circuit with exact MLD on the final syndrome is second order at d=3 (A = 232 ± 8) and beats physical RB (5p) on the whole grid and (p) below 1.60e-3. | VN | Section 5.2 |
| 18 | The const0 advantage depends on the ideal terminal readout and on SI1000 having no idle noise during gate layers. | C | reasoning, not simulated |
| 19 | The steeper late-depth decay of corrected d=3 LRB comes from error accumulation in the check-free inverse half. | C | late-depth fits only; `inverse_check_every` variant not run |
| 20 | My d=2 numbers agree with the lead's `grid_main` within errors. | VN | Section 5.4 |
| 21 | Physical RB at p_RB: F = (1 - 4q/3)^N, q = 0.1 p_RB, r = 0.339 p_RB to first order. | VA, VN | Section 3; `test_exact_physical_rb_matches_sampling`, `test_physical_rb_scales` |
| 22 | No bug found in the lead's `lrb_qubit` modules. | VN | exact reproduction tests; lead's 64 tests pass |

Literature note: I fetched all three sources in this session.

- arXiv:2505.13599: abstract and HTML; title, authors and v1 date
  19 May 2025 confirmed. Its Appendix F.1 on distance-reducing hooks for its
  decoder was beyond the part of the HTML I read.
- arXiv:1603.02286 (v1 7 Mar 2016, PRA 94, 042316): abstract.
- arXiv:2202.06647 (v1 14 Feb 2022, Quantum 8, 1372): abstract.


## 8. Open questions

1. A matching decoder that is distance-preserving across fold-transversal S
   (the full logical-observable matching of Serra-Peralta et al., including
   their fixes for hook errors) would be about 1000x faster than BP-OSD here.
   My frame-restricted simplification is not enough (Section 4.2).
2. Exact MLD for d = 3 (12 detectors per round, 2^25-entry transfer kernels)
   was not attempted; BP-OSD with fallbacks corrects all single faults at
   p <= 1e-4 but is not proven optimal, so the d = 3 numbers are upper bounds
   on what an optimal decoder achieves [C].
3. The 32 distance-2 CNOT orders: the exact two-fault mechanism (straight X
   and Z hooks combined through the S/H frame changes) is only inferred from
   one Stim explanation; a full classification is open [C].
4. The inverse half has no checks, so at d = 3 errors from up to 22 inverse
   Cliffords (about 6 two-qubit gates each) accumulate before the final round.
   A variant with checks in the inverse half (`inverse_check_every`) would
   separate this contribution from the forward rounds.
5. Statistics: BP-OSD uses 10 of the 30 sequences with 35 to 26000 decoded
   shots per circuit (time-budgeted; fewest at p = 2e-3, m = 22), so its r
   has 1.5-20% relative uncertainty (largest at p = 1e-4). The p* intervals
   include this, but not the sequence-sampling bias of using 10 sequences.
6. A noise model with less measurement-layer idle noise (or a hardware
   schedule closer to the parallel variant) is the main lever; how the
   pseudo-threshold moves with `SI1000.scaled(meas_idle=...)` is not run.

## 9. Reproduce

From the repository root (`module load python/3.11`, `export
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`); heavy runs on
the shared allocation with `srun --jobid=<JOB> --overlap --cpu-bind=none -N 1
-n 1 -c 256 bash -c "..."` and at most 24 workers.

```
PYTHONPATH=$PWD/src python -m pytest -q tests/test_qubit512_d3.py
G=studies/correction-decoders/qubit/results/G
python studies/correction-decoders/qubit/scripts/d3_compare.py checks --out $G
# d=2 (LRB-D, exact MLD, PyMatching, const0), 30 sequences
python studies/correction-decoders/qubit/scripts/d3_compare.py run --kind main --out $G --codes d2 \
  --schedules parallel,sequential --p-values 0.0001,0.0005,0.001,0.002,0.005 \
  --num-seq 30 --shots 40000 --decoders mld,pm --mld-shots 40000 --workers 4
# d=3 (LRB-D, PyMatching, frame matching, const0), 30 sequences
python studies/correction-decoders/qubit/scripts/d3_compare.py run --kind main --out $G --codes d3 \
  --schedules parallel,sequential --p-values 0.0001,0.0005,0.001,0.002,0.005 \
  --num-seq 30 --shots 30000 --decoders pm,pmf --workers 8
# d=3 BP-OSD, one time-budgeted task per (schedule, p, sequence, depth)
python studies/correction-decoders/qubit/scripts/d3_compare.py run --kind decoder --decoder bposd \
  --out $G --codes d3 --schedules parallel,sequential \
  --p-values 0.0001,0.0005,0.001,0.002,0.005 --num-seq 10 --bposd-shots 600 \
  --bposd-shots-by-p 0.0001:3000,0.0005:1200,0.001:600,0.002:300,0.005:150 \
  --max-depth-by-p 0.005:10 --budget 40 --workers 6
# top-up passes (new seeds, summed by analyze): see results/G/runs/*.json
python studies/correction-decoders/qubit/scripts/d3_compare.py analyze --out $G
```

The exact commands of every pass are stored in `results/G/runs/*.json`; logs
are `results/G/run*.log`. `runA.log` is a first attempt that saved nothing:
its `srun -c 24` step was pinned to a few shared CPUs and every bundled task
ran past its timeout. All later passes used `--cpu-bind=none -c 256` with at
most 24 workers. Time budgets make the number of decoded BP-OSD
shots depend on node load, so a rerun reproduces the estimates within their
statistical errors, not bit for bit.
