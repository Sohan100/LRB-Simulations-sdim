# K: Steane [[7,1,3]] LRB with correction vs detection (work package K)

Labels: **VA** verified analytically, **VN** verified numerically, **LIT** from
literature (fetched by me, or verified in `lit/`), **C** conjecture. Every
number has a statistical uncertainty (sequence bootstrap, 200 replicates,
unless stated) or is exact (DEM computations).

## Summary

ε below is the fitted logical error per depth step, (1−f)/2. "Physical RB"
is single-qubit RB with DEPOLARIZE1(0.1 p_RB) per gate, at p_RB = p and at
p_RB = 5p. Its ε is 0.336 ± 0.016 p and 1.68 ± 0.08 p on these 10 sequences.

- **Headline.** The best variant is flagged extraction with all ancillas
  measured in one layer (parallel). With full exact MLD correction and no
  rejection it gives ε = (1030 ± 10)·p² per step; the low-p log-log slope
  is 2.04. At p = 1e-4 that is ε = 0.096 ± 0.003 p:
  - 3.5× below physical RB at p_RB = p;
  - 124× below the d=2 [[5,1,2]] exact MLD of notes/L0 (11.9 ± 0.1 p).

  It beats physical RB below these pseudo-thresholds (bootstrap 68%):
  - **p* = 3.3e-4 [3.2e-4, 3.5e-4]** against p_RB = p;
  - **p* = 1.60e-3 [1.53e-3, 1.68e-3]** against p_RB = 5p.

  (VN)
- **Schedule.** The repo's sequential schedule measures six ancillas one
  layer at a time, and each layer idles every data qubit at 2p. That raises
  the second-order coefficient 5.2× (c2 = 5390 ± 65) and lowers the
  pseudo-thresholds by the same factor:
  - p* = 3.1e-4 [2.9e-4, 3.2e-4] against p_RB = 5p;
  - p* = 6.1e-5 [5.8e-5, 6.4e-5] against p_RB = p, bracketed by an extra
    run at p = 2.5e-5 and 5e-5 with 1e6 decoded shots.

  (VN)
- **Naive extraction (no flags).** The circuit-level distance is 2. Hook
  errors leave an exact first-order correction floor of **0.753 ± 0.016 p
  per step** in both schedules. That is above physical RB at p_RB = p
  (0.339p), so naive extraction never beats it, at any p. It beats
  p_RB = 5p only below 1.0e-3 [0.89e-3, 1.09e-3] (parallel) or 1.5e-4
  [1.4e-4, 1.7e-4] (sequential). (VN, exact DEM; VA)
- **Flags restore distance 3.** The exact DEM pair check, Stim's
  hypergraph search and `shortest_graphlike_error` all give d = 3. The
  first-order MLD error and the second-order accepted LRB-D error are
  exactly 0. (VN)
- **Correction vs detection.**
  - *On accepted-shot fidelity, detection still wins at d = 3.* LRB-D with
    every check, the flags and the final data check has a third-order
    accepted error.
    - Parallel: 0–27 logical errors in 1.5e6–1.5e7 accepted shots per p.
    - Sequential: 0–159 errors in 7.9e5–1.4e7.

    Its fitted decay is consistent with 0 (crude 95% bound ε ≲ 1.4e-3 p for
    p ≤ 1e-3).
  - *Detection pays more rejection than at d = 2.* LRB-D's acceptance per
    step is 1 − 158p (parallel) or 1 − 268p (sequential), against 1 − 105p
    at d=2. At p = 1e-3 and m = 22 only 2.8% (parallel) or 0.23%
    (sequential) of shots survive.
  - *Correction is now useful.* At d=2 it was first order (12p) and never
    reached physical RB. At d = 3, with flags and the parallel schedule, it
    beats physical RB without discarding any shot below p*.
  - *The soft-output middle ground is the practical optimum.* MLD
    postselected at |LLR| ≥ 4 gives ε = 0.178 ± 0.002 p at p = 1e-3 and
    keeps 99.25% of shots per step, against LRB-D's 85.4%. That is half of
    physical RB at p_RB = p.

  So correction beats detection whenever rejection counts as a cost, and
  only with flags and the parallel schedule. (VN)

## 1. Methods

### 1.1 Code and logical gates (`src/lrb_qubit/steane_code.py`)

- Data qubits 0–6. Qubit q is column q+1 of the [7,4,3] Hamming parity-check
  matrix, so the three supports are (3,4,5,6), (1,2,5,6) and (0,2,4,6).
  Stabilizers S_Z1, S_Z2, S_Z3 (index 0–2) and S_X1, S_X2, S_X3 (index 3–5)
  use these supports. The ancilla of stabilizer s is wire 7+s, and with flags
  its flag is wire 13+s. Naive extraction uses 13 qubits, flagged uses 19.
- X_L = X0X1X2 and Z_L = Z0Z1Z2 (weight 3), with Y_L = iX_LZ_L = −Y0Y1Y2.
- Logical H = H on all 7 data qubits. **Logical S = S_DAG on all 7.**
  Transversal S maps X_L → −Y_L, so it is logical S†. That follows from
  7 ≡ 3 mod 4. Logical X and Z act on {0,1,2} (default, weight 3); the
  option `pauli_gates="transversal"` uses all seven qubits. The identity
  Clifford is an `I` on data 0, as in `lrb_qubit.code`.
- The 24-descriptor catalog and the token-by-token inverses come from
  `lrb_qubit.code`. Sequences are `experiment.clifford_sequences(10, 22,
  20261028)`, the first 10 of the main grid's 30.

### 1.2 Circuits (`src/lrb_qubit/steane_circuits.py`)

The structure is `circuits.build_lrb_circuit`, data5 semantics:

1. ideal encoding of |+>_L (Stim elimination circuit, noiseless);
2. after each forward Clifford, a full noisy check round (Z checks, then X
   checks), then a reset of every ancilla and flag in one reset layer;
3. mirror-style inverses with no checks;
4. a final check round with no reset;
5. an ideal X readout of all 7 data qubits. The observable is the x0+x1+x2
   parity. Three final X-stabilizer detectors come from the readout.

Noise is the repo's SI1000 through `circuits.emit_operations`:
DEPOLARIZE1(0.1p) after 1-qubit gates, DEPOLARIZE2(p) after CNOTs; per
measurement layer DEPOLARIZE1(2p) on every other wire, plus DEPOLARIZE1(p)
and X_ERROR(5p) on the measured wires; per reset layer DEPOLARIZE1(2p) on
every other wire and X_ERROR(2p) after each reset. There is no idle noise in
gate layers. The repo's variant follows the SI1000 table of Gidney et al.
(LIT, arXiv:2108.10457, Table 2, fetched: AnyClifford1(p/10), CZ(p),
InitZ(2p), MZ(5p), ResonatorIdle(2p)). It adds DEPOLARIZE1(p) on measured
qubits and drops the gate-layer Idle(p/10).

Syndrome extraction (weight-4 checks, CNOT order = support order):

- **naive**: one ancilla per stabilizer. Z check: CX d→a ×4. X check: H a,
  CX a→d ×4, H a.
- **flag** (Chao–Reichardt, LIT arXiv:1705.02329, PRL 121, 050502 (2018)):
  one flag per stabilizer, coupled after the first and before the last data
  CNOT.
  - X check: `H a; CX a d1; CX a f; CX a d2; CX a d3; CX a f; CX a d4; H a`,
    with the flag measured in the Z basis.
  - Z check: `H f; CX d1 a; CX f a; CX d2 a; CX d3 a; CX f a; CX d4 a; H f`,
    with the flag measured in the X basis.

  Each flag reads 0 without faults. One flag per weight-4 check is the
  textbook distance-3 construction, and Reichardt applies it to the Steane
  code (LIT arXiv:1804.06995, Quantum Sci. Technol. 6, 015007).

Measurement schedules:

- **sequential** (repo layout): each check's ancilla and flag are measured
  in their own layer right after its CNOTs. That gives 6 measurement layers
  per round, each putting DEPOLARIZE1(2p) on every other wire.
- **parallel**: the same gates in the same order, with every ancilla and
  flag measured in one layer at the end of the round.

Detectors are the lead's difference detectors: round r against round r−1's
generator product after back-propagation through the transversal gates.
Flag detectors are single measurements.

### 1.3 Decoders

- **Exact MLD.** `SlotMLD` is the lead's transfer-matrix decoder
  (`mld.TransferMatrixMLD`: time slots = check rounds plus the final readout,
  6 and 3 detectors) extended with *conditioning detectors*. Every mechanism
  that flips a flag of round r touches only the syndromes of rounds r and
  r+1, and at most one flag. So the flag mechanisms are folded into the r→r+1
  transfer kernel with one extra parity bit per flag, and the kernel is
  restricted to the shot's observed flags. That gives P(syndrome, flags, L)
  exactly with 128×128 transfer matrices. Kernels are built lazily per
  observed flag pattern and cached by flag prefix. Validation (VN):
  - equals brute-force inference to < 4e-15 in the LLR (flag depth 0,
    15 detectors; naive depth 1);
  - equals it on a depth-1 flagged instance where the round-0 flags condition
    the step into round 1 (18 detectors);
  - equals the lead's `TransferMatrixMLD` exactly (difference 0.0) without
    flags.
- **LRB-D.** The `baseline.used_rounds` const/unif semantics. A shot is
  accepted when every used round has a trivial raw syndrome and, with flags,
  no raised flag. The `_final` variants also require the three final data
  detectors to be clean. `_noflag` ignores the flags.
- **Soft output.** Accept a shot when the exact |LLR| ≥ τ, τ ∈ {2,4,6,8}, as
  in the lead's gap histograms. The exact |LLR| is the exact counterpart of
  the complementary gap.
- **Physical RB.** Exact: F = (1−4q/3)^N with q = 0.1 p_RB, where N is the
  physical gate count of the forward and inverse halves. It matches Stim
  sampling of `circuits.build_physical_rb_circuit` (test). It is evaluated
  on the same 10 sequences at **p_RB = p** and **p_RB = 5p**. Ensemble value:
  (1−f)/2 = 0.3389 p_RB.

### 1.4 Statistics

- Grid: p ∈ {1e-4, 5e-4, 1e-3, 2e-3, 5e-3}; depths 0,2,4,6,10,14,18,20,22;
  10 sequences; 2e5 shots per circuit for LRB-D; 1e5 of them decoded by
  exact MLD on the same shots.
- Fit: the repo's offset-0 fit `fits.fit_from_counts` (F = b f^m, weighted
  grid scan; points with no accepted shots dropped). The reported number is
  the error per depth step ε = (1−f)/2.
- Uncertainty: std over 200 sequence-bootstrap replicates, which also covers
  shot noise because each replicate refits the resampled per-sequence counts.
  The same resampled indices are used at every p and for physical RB, so the
  ratios and pseudo-thresholds are bootstrapped consistently.

## 2. Structural results (exact)

### 2.1 Circuit-level distance (`results/K/distance.json`)

Checked on sequence 0 at depths 0, 1, 2 and 4, both schedules. "Exact" is
the DEM pair check:
- d ≥ 2 if no mechanism flips L without a detection event;
- d ≥ 3 if no two mechanisms share their detection events but differ in L.

Stim's `search_for_undetectable_logical_errors` (size 6, degree 6, no
monotonicity pruning) and `shortest_graphlike_error` (hyperedges ignored)
give upper bounds.

| extraction | schedule | exact lower bound | ambiguous symptom classes (depth 4) | Stim hypergraph search | shortest graphlike | ⇒ d |
|---|---|---|---|---|---|---|
| naive | sequential | 2 | 32 | 2 | 2 | **2** |
| naive | parallel | 2 | 32 | 2 | 2 | **2** |
| flag | sequential | 3 | 0 | 3 | 3 | **3** |
| flag | parallel | 3 | 0 | 3 | 3 | **3** |

The naive witness at depth 0 has two faults:
- a Y fault on ancilla 9 (S_Z3, CNOT order 0, 2, 4, 6) right after
  CX(2→9). This is the hook: the Z part spreads to Z4 Z6, and the X part
  flips that check's outcome;
- a Y2·Z8 fault on CX(2→8).

Together they leave no detection event and flip X_L.

Every weight-2 error of the Steane code equals a weight-1 error times a
logical operator, because the Hamming code is perfect: columns c+1 ⊕ d+1 =
e+1 (VA; the test checks all 21 pairs). So the hook is miscorrected for every
CNOT order. That gives the task's "distance 2" (VN, VA).

### 2.2 First-order DEM structure (`results/K/first_order.json`)

The construction is the lead's (notes/L0 §4), with p = 1e-6, 10 sequences and
all depths. The slopes per depth step are linear fits per sequence; ± is the
SEM over sequences.

| variant | R1/p per step (LRB-D rejection) | E1/p per step (MLD error, 1st order) | A2/p² per step (LRB-D accepted, 2nd order) | E1/p at depth 0 |
|---|---|---|---|---|
| naive, sequential | 190.3 ± 0.1 | **0.753 ± 0.016** | 3.35 ± 0.08 | 0.80 |
| naive, parallel | 100.3 ± 0.1 | **0.753 ± 0.016** | 1.17 ± 0.02 | 0.80 |
| flag, sequential | 267.9 ± 0.1 | **0 (exact)** | **0 (exact)** | 0 |
| flag, parallel | 157.9 ± 0.1 | **0 (exact)** | **0 (exact)** | 0 |
| [[5,1,2]] (L0, sequential) | ≈105 | ≈11.8 | ≈48 | 11.1 |

Reading the table:
- E1 is the same for both naive schedules, so it comes only from CNOT hooks.
- With flags, every single fault is corrected; this is also checked
  directly by decoding every DEM mechanism (test).
- Flags cost rejection: 58p (parallel) to 78p (sequential) more per step.
  The cost comes from the twelve extra CNOTs, the flag measurements and
  resets, and the flags' own idling.

## 3. Main grid: full exact-MLD correction (`results/K/tables.md`)

Error per step ε/p, ± sequence-bootstrap std (VN):

| method | p = 1e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 |
|---|---|---|---|---|---|
| physical RB, p_RB = p | 0.336 ± 0.016 | 0.336 ± 0.016 | 0.336 ± 0.016 | 0.336 ± 0.016 | 0.336 ± 0.016 |
| physical RB, p_RB = 5p | 1.68 ± 0.08 | 1.68 ± 0.08 | 1.68 ± 0.08 | 1.68 ± 0.08 | 1.67 ± 0.08 |
| Steane flag, parallel, MLD | **0.096 ± 0.003** | **0.516 ± 0.006** | 1.04 ± 0.01 | 2.09 ± 0.02 | 4.93 ± 0.04 |
| Steane flag, sequential, MLD | 0.553 ± 0.010 | 2.74 ± 0.04 | 5.28 ± 0.06 | 9.64 ± 0.09 | 17.7 ± 0.2 |
| Steane naive, parallel, MLD | 0.852 ± 0.017 | 1.22 ± 0.02 | 1.68 ± 0.02 | 2.58 ± 0.03 | 4.87 ± 0.04 |
| Steane naive, sequential, MLD | 1.31 ± 0.03 | 3.41 ± 0.05 | 5.77 ± 0.06 | 9.72 ± 0.09 | 16.9 ± 0.1 |
| [[5,1,2]] mld5, 30 seq (L0 grid, refit here) | 11.9 ± 0.1 | 12.2 ± 0.1 | 12.5 ± 0.1 | 13.1 ± 0.1 | 14.2 ± 0.1 |
| [[5,1,2]] mld5, same 10 seq | 11.9 ± 0.2 | 12.2 ± 0.2 | 12.5 ± 0.2 | 13.1 ± 0.2 | 14.3 ± 0.2 |
| Steane no-check circuit (const0), decoded | 0 (no errors) | 0.008 ± 0.004 | 0.010 ± 0.002 | 0.021 ± 0.006 | 0.050 ± 0.010 |
| [[5,1,2]] const0, decoded (30 seq) | 0.744 ± 0.015 | 0.744 ± 0.015 | 0.744 ± 0.015 | 0.743 ± 0.015 | 0.742 ± 0.015 |

The physical-RB ensemble value is exactly 0.3389 p_RB per step. The fit on
these 10 sequences, 0.336 ± 0.016, agrees.

Low-p model (VN): ε = e1·p + c2·p². e1 is the exact first-order slope of
§2.2; c2 is the inverse-variance mean of (ε − e1 p)/p² at p ≤ 1e-3.

| variant | e1 | c2 (bootstrap std) | c2 at p = 1e-4 / 5e-4 / 1e-3 | log-log slope, p ≤ 1e-3 |
|---|---|---|---|---|
| flag, parallel | 0 | 1030 ± 10 | 962 / 1032 / 1037 | 2.04 |
| flag, sequential | 0 | 5390 ± 65 | 5530 / 5488 / 5280 | 1.98 |
| naive, parallel | 0.753 | 931 ± 29 | 986 / 941 / 926 | 1.28 (→ 1 as p → 0) |
| naive, sequential | 0.753 | 5115 ± 78 | 5555 / 5316 / 5017 | 1.64 (→ 1) |

The per-point c2 values agree within about 10%, so the two-term model
describes the p ≤ 1e-3 data. The naive c2 is close to the flagged one: the
second-order budget is set by data idling and CNOT noise, not by the hooks.

**Low-p extension** (`results/K/tables_lowp.md`). Flag, sequential, 10
sequences, 1e6 shots per circuit, all decoded (VN):

| p | 2.5e-5 | 5e-5 | 1e-4 |
|---|---|---|---|
| MLD ε/p | 0.137 ± 0.003 | 0.276 ± 0.003 | 0.553 ± 0.010 |
| physical RB (p_RB = p) ε/p | 0.337 ± 0.016 | 0.336 ± 0.016 | 0.336 ± 0.016 |
| MLD, \|LLR\| ≥ 4: ε/p; acceptance per step | 0.040 ± 0.002; 1.0000 | 0.074 ± 0.002; 0.9999 | 0.132 ± 0.005; 0.9997 |
| LRB-D unif=1 + final: acceptance per step | 0.9933 | 0.9867 | 0.9735 |

ε/p doubles with p, so the p² law holds down to 2.5e-5. The fit over all
p ≤ 1e-3 gives c2 = 5435 ± 63 and a log-log slope of 1.99.

## 4. Detection and soft-output postselection

ε/p (± bootstrap std) and acceptance per step a (log-linear fit of the mean
acceptance). LRB-D uses `used_rounds`. With flags, a raised flag in a used
round also rejects the shot. "+ final" also requires the three final data
detectors to be clean. (VN)

| method | p = 1e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 |
|---|---|---|---|---|---|
| flag par, LRB-D unif=1 + final | 0 errors / 1.5e7 acc.; a = 0.9843 | 1 / 8.7e6; 0.924 | 5 / 5.6e6; 0.854 | 11 / 3.3e6; 0.729 | 27 / 1.5e6; 0.462 |
| flag seq, LRB-D unif=1 + final | 0 / 1.4e7; 0.9735 | 9 / 6.2e6; 0.875 | 18 / 3.7e6; 0.766 | 73 / 2.0e6; 0.590 | 159 / 7.9e5; 0.288 |
| flag par, unif=1 + final, flags ignored | 2 err.; 0.989 | 0.0008 ± 0.0003; 0.947 | 0.0017 ± 0.0004; 0.896 | 0.0039 ± 0.0016; 0.803 | –; 0.574 |
| naive par, LRB-D unif=1 + final | 2 err.; 0.990 | 27 err.; 0.951 | 0.0015 ± 0.0004; 0.905 | 0.0025 ± 0.0009; 0.818 | –; 0.604 |
| flag par, LRB-D const=1 + final | 0.000 (86 err.); 0.994 | 0.018 ± 0.005; 0.972 | 0.049 ± 0.019; 0.945 | 0.11 ± 0.03; 0.899 | 0.49 ± 0.11; 0.837 |
| flag par, MLD with \|LLR\| ≥ 4 | 0.031 ± 0.002; 1.0000 | 0.114 ± 0.002; 0.9984 | **0.178 ± 0.002; 0.9925** | 0.170 ± 0.004; 0.958 | 0.110 ± 0.002; 0.803 |
| flag par, MLD with \|LLR\| ≥ 8 | 0 err. fit; 0.997 | 0.0013 ± 0.0006; 0.980 | 0.0031 ± 0.0005; 0.957 | 0.0044 ± 0.0004; 0.888 | –; 0.503 |
| flag seq, MLD with \|LLR\| ≥ 4 | 0.132 ± 0.005; 0.9997 | 0.444 ± 0.007; 0.989 | 0.329 ± 0.008; 0.945 | 0.205 ± 0.006; 0.835 | – |
| [[5,1,2]] LRB-D unif=1 + final (30 seq) | 0.0045 ± 0.0002; 0.990 | 0.0248 ± 0.0005; 0.949 | 0.0502 ± 0.0008; 0.900 | 0.108 ± 0.002; 0.811 | 0.32 ± 0.04; 0.594 |

Notes on the table:
- Where the fit returns f = 1 I give the raw error count instead of "0 ± 0".
  For flagged LRB-D with every check, a crude 95% bound from the best depth,
  3/(N_acc·m), gives ε ≲ 1e-3 p at p ≤ 1e-3 and ≲ 1e-3 p (parallel) or
  1.6e-2 p (sequential) at 2e-3.
- At p = 5e-3 the postselected fits are unreliable. The acceptance at
  m ≥ 14 is below 1e-3, and F(m) is no longer exponential.
- Ignoring the flags turns the flagged LRB-D accepted error second order:
  hook plus data pairs give 0.0017 ± 0.0004 p at 1e-3, the size of the naive
  A2·p = 0.0012p. The flags are what make detection third order (VN).

## 5. Pseudo-thresholds (`results/K/summary.json`)

p* is where the corrected ε (full MLD, no rejection) equals physical RB.
"Interpolated" is log-log interpolation between grid points of
ε/ε_RB, with the same bootstrap replicates for both (VN); the model is §3's
ε = e1 p + c2 p².

| variant | vs p_RB = p | vs p_RB = 5p |
|---|---|---|
| flag, parallel | **3.32e-4 [3.17e-4, 3.47e-4]**; model 3.27e-4 | **1.60e-3 [1.53e-3, 1.68e-3]**; model 1.63e-3 |
| flag, sequential | **6.09e-5 [5.80e-5, 6.42e-5]** (low-p run); model 6.19e-5 | 3.06e-4 [2.91e-4, 3.21e-4]; model 3.11e-4 |
| naive, parallel | **none**: ε → 0.753p > 0.339p as p → 0 (VA from §2.2) | 1.00e-3 [0.89e-3, 1.09e-3]; model 0.99e-3 |
| naive, sequential | **none** (same reason) | 1.52e-4 [1.39e-4, 1.67e-4]; model 1.81e-4 |

Below the grid, tables.md's pure power-law extrapolation for the naive
variants (1.0e-5 and 1.6e-6) is not physical. Their low-p exponent tends
to 1, and the exact first-order coefficient rules out any crossing. Read
those rows as "none".

For the naive sequential variant the interpolated and model values differ
(1.5e-4 against 1.8e-4). The grid there is coarse (1e-4 to 5e-4), so read
1.5–1.8e-4 as the range.

## 6. Answer: does correction beat detection at d = 3, and with which schedule?

- **Accepted-shot error per step: no.** LRB-D with every check plus flags
  stays third order: A2 = 0 exactly, and essentially no accepted errors at
  any p (§4). Exact MLD without rejection is second order (c2 ≈ 1030 or
  5390). Detection beats correction in accepted fidelity by orders of
  magnitude at every p of the grid. (VN)
- **Against the physical qubit, without rejection: yes, below p*.** At d=2,
  correction was first order (≈12p) and was 35× worse than physical RB at
  p_RB = p at every p. At d = 3 with flags and parallel measurement it is
  below physical RB for p < 3.3e-4 (p_RB = p) or p < 1.6e-3 (p_RB = 5p). It
  needs no postselection, so F(m) has no acceptance decay. LRB-D's
  acceptance per step falls as 1 − 158p, which is 0.854 at p = 1e-3, so 2.8%
  of shots survive at m = 22. (VN)
- **Schedule.** Parallel ancilla measurement is required to make correction
  competitive. The sequential layout raises c2 5.2× and lowers both
  pseudo-thresholds 5×. Detection prefers parallel too: the rejection per
  step drops from 268p to 158p. (VN)
- **Flags are required for correction.** Without them the 0.753p hook floor
  rules out beating physical RB at p_RB = p. (VN, VA)
- **The practical optimum is in between.** MLD with an |LLR| threshold keeps
  most shots and reaches sub-physical error per step up to p ≈ 2e-3. At
  p = 1e-3, parallel: ε = 0.178p with 0.75% rejection per step
  (|LLR| ≥ 4), or 0.0031p with 4.3% (|LLR| ≥ 8). LRB-D gives ≈0 with 14.6%.
  This is the d=3 version of the "correct the confident, reject the
  ambiguous" policy. At d=3 it dominates LRB-D whenever rejection has a cost.
  (VN for the numbers; the dominance statement is C, since the frontier at
  matched acceptance was not computed. See open questions.)

## 7. Claims

| # | claim | label | evidence |
|---|---|---|---|
| K1 | Transversal S implements logical S†; logical S = S_DAG on all 7 qubits; H^⊗7 = logical H | VN, VA | `test_transversal_gate_conventions` (Stim tableaus); 7 ≡ 3 mod 4 |
| K2 | All 24 descriptors give the same logical Clifford as in the lead's [[5,1,2]] model, and their token-wise inverses undo them, for both X/Z conventions | VN | `test_clifford_catalog_matches_lead_code` |
| K3 | Every weight-2 Pauli error of the Steane code is a weight-1 error times a logical | VA, VN | Hamming columns; `test_code_distance_three_and_weight_two_equivalence` |
| K4 | Naive one-ancilla extraction has circuit-level distance 2 (both schedules, depths 0–4) | VN | exact DEM pair check, Stim hypergraph search = 2, `shortest_graphlike_error` = 2 (`distance.json`, `test_circuit_distance`) |
| K5 | One flag per weight-4 check (Chao–Reichardt) gives circuit-level distance 3 | VN | exact pair check ≥ 3, Stim search = 3, graphlike = 3; exact MLD corrects every single DEM mechanism (`test_flagged_mld_corrects_every_single_fault`) |
| K6 | The flag-conditioned transfer-matrix MLD is exact | VN | brute force agrees to < 1e-9 (4e-15 observed) at 15 and 18 detectors, including a non-final flag transition; equals `mld.TransferMatrixMLD` without flags (difference 0) |
| K7 | Naive extraction has a first-order MLD error of 0.753 ± 0.016 p per step (both schedules) | VN | exact DEM, 10 sequences × 9 depths (`first_order.json`); sampled ε/p → 0.85 at p = 1e-4 matches 0.753 + c2 p |
| K8 | Naive extraction cannot beat physical RB at p_RB = p at any p | VA + VN | ε ≥ 0.753p − O(p²) > 0.339p; the first-order coefficient is exact |
| K9 | With flags, the MLD error is second order: c2 = 1030 ± 10 (parallel), 5390 ± 65 (sequential) p² per step | VN | E1 = 0 exactly; sampled log-log slope 2.04 / 1.98; per-point c2 within 10% |
| K10 | Flagged parallel MLD: ε = 0.096 ± 0.003 p at p = 1e-4, 124× below [[5,1,2]] MLD | VN | §3 table |
| K11 | Pseudo-threshold (flag, parallel): 3.3e-4 [3.2, 3.5]e-4 vs p_RB = p, 1.60e-3 [1.53, 1.68]e-3 vs p_RB = 5p | VN | §5; interpolation and model agree within 2% |
| K12 | Sequential schedule (flags): pseudo-thresholds 5× lower (3.1e-4 [2.9, 3.2]e-4 vs 5p; 6.1e-5 [5.8, 6.4]e-5 vs p) | VN | §5; low-p run brackets the crossing, model agrees within 2% |
| K13 | Flagged LRB-D (all checks + final) has third-order accepted error; ignoring flags makes it second order | VN | A2 = 0 exactly; 0–27 errors / ≥ 1.5e6 accepted shots; no-flag variant 0.0017 ± 0.0004 p at 1e-3 |
| K14 | LRB-D rejection per step is 158p (flag par), 268p (flag seq), 100p (naive par), 190p (naive seq), against ≈105p at d=2 | VN | exact R1 slopes; sampled acceptance per step agrees (e.g. 0.9843 = 1 − 157p at 1e-4) |
| K15 | Detection beats correction on accepted-shot fidelity at every p of the grid; correction beats physical RB without rejection only with flags, below p* | VN | §3–§5 |
| K16 | Soft-output |LLR| ≥ 4 gives sub-physical error per step up to p ≈ 2e-3 with ≤ 4.2% rejection per step (flag parallel) | VN | §4 |
| K17 | At matched acceptance, the MLD soft-output frontier dominates LRB-D at d=3 | C | not computed; at LRB-D's own acceptance the frontier should coincide with it (trivial syndromes have the largest \|LLR\|, about log 1/p³), as at d=2 (L0 §7) |
| K18 | Without check rounds the Steane LRB (transversal gates only, decoded) has ε ≈ 0.01p at 1e-3, against 0.74p at d=2: the gates are not the bottleneck, the check rounds are | VN | §3 const0 rows |
| K19 | Flag extraction for the Steane code needs 2 ancillas per check here (19 qubits). Reichardt shows shared-flag schemes that extract several syndromes at once with fewer qubits | LIT | arXiv:1804.06995 (fetched) |
| K20 | Trapped-ion experiments have run flag-FT Steane-code preparation and repeated Steane-code QEC with real-time decoding, and compared correction with detection on the [[7,1,3]] code | LIT | lit/06 and lit/01 entries Postler2022 (arXiv:2111.12654), RA2021 (arXiv:2107.07505), Paetznick2024 (arXiv:2404.02280), verified in those files |
| K21 | Platforms whose idle error during readout is small relative to gate error are closer to our parallel schedule than to the sequential one, so they should see pseudo-thresholds nearer 3e-4 (p_RB = p) than 6e-5 | C | follows from K9 and K12: the sequential penalty comes only from per-layer data idling; not simulated with an ion or atom noise model |

## 8. Open questions

1. Compute the full |LLR| frontier at matched acceptance against each LRB-D
   policy, as in L0 §7 (K17). The gap histograms are stored in the parts.
2. Inverse-half accumulation. With no checks in the inverse half, two
   uncorrected data faults there fail at second order, roughly ∝ (m·1.35p)².
   That is a depth-dependent (non-exponential) term at d=3, which d=2 did
   not have at leading order. Local decay rates should be checked (work
   package F tools).
3. Shared-flag and parallel flag schemes (Reichardt arXiv:1804.06995), and
   Steane- or Knill-style extraction, use fewer qubits and fewer measurement
   layers. Under SI1000 the measurement-layer idling dominates, so these
   could change the sequential picture.
4. Hardware-specific noise for ions and atoms: no ResonatorIdle, long but
   low-error readout, transport or SWAP costs. The Steane advantage over the
   fold-transversal codes (no SWAP or CZ in logical gates) is invisible here:
   const0 shows the gates contribute only about 0.01p per step. It would
   matter only for platforms with expensive two-qubit gates.
5. Comparison with WP G (d=3 surface code, fold-transversal gates with
   SWAP/CZ) on the same sequences and noise, once G's numbers are final.
6. p = 5e-3 postselected fits are unreliable (acceptance < 1e-3 at m ≥ 14).
   More shots or the lead's local-rate analysis are needed there.

## Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest -q tests/test_qubit512_steane.py                 # 27 tests, ~3 s
python studies/correction-decoders/qubit/scripts/steane_compare.py checks --out studies/correction-decoders/qubit/results/K --num-seq 10 --workers 4
# main grid (about 10 min on 24 cores; pin to idle cores on the shared node, see the note below)
srun --jobid=<JOBID> --overlap -N 1 -n 1 -c 256 bash -c "module load python/3.11; cd $PWD; \
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src; \
  exec taskset -c 100-123 python studies/correction-decoders/qubit/scripts/steane_compare.py run \
  --out studies/correction-decoders/qubit/results/K --num-seq 10 --shots 200000 \
  --decode-shots 100000 --workers 24 --max-minutes 14"
# low-p extension (flag, sequential; about 10 min on 20 cores)
srun --jobid=<JOBID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c "module load python/3.11; cd $PWD; \
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src; \
  python studies/correction-decoders/qubit/scripts/steane_compare.py run --out studies/correction-decoders/qubit/results/K \
  --num-seq 10 --p-values 2.5e-05,5e-05 --variants flag:sequential --shots 1000000 \
  --decode-shots 1000000 --workers 20 --max-minutes 14 --seed 2"
python studies/correction-decoders/qubit/scripts/steane_compare.py analyze --out studies/correction-decoders/qubit/results/K --num-seq 10 --num-boot 200 \
  --p-values 0.0001,0.0005,0.001,0.002,0.005 --variants naive:sequential,naive:parallel,flag:sequential,flag:parallel
python studies/correction-decoders/qubit/scripts/steane_compare.py analyze --out studies/correction-decoders/qubit/results/K --num-seq 10 --num-boot 200 \
  --p-values 2.5e-05,5e-05,0.0001,0.0005,0.001,0.002,0.005 --variants flag:sequential --tag lowp
```

Compute note: overlapping `srun --overlap -c 24` steps are all bound to the
lowest CPUs of the node (0–11, 128–139 here). With other agents running there,
my first attempt got 8% of a core per worker. Pinning the 24 workers with
`taskset` to idle physical cores (100–123) fixed this. The run still used
24 cores. The lead's recommended form, `--cpu-bind=none -c 256` with ≤ 24
workers, was used for the low-p run and works equally well.
