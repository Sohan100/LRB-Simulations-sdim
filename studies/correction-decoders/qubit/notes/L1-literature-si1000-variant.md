# L1: robustness under literature-exact SI1000 with ASAP scheduling (lead session)

## Why this variant exists

The brief asked for the noise of the August 2026 qutrit production run, which
is the repo's "generalized SI1000". The main grid (notes L0) uses it.

On 2026-10-28 the user decided that new qutrit production runs use a
different model. The decision is recorded in the user's memory note "sdim
1.4.0 SI1000/SD6 runs", and the code is uncommitted in the main checkout's
`src/lrb/circuit_generator.py`:

- literature-exact Gidney et al. 2021 SI1000, with ASAP moment scheduling and
  gate-layer idle noise;
- ancilla resets placed before each check round;
- plots that use p, the two-qudit gate error, as the axis, with physical RB
  compared at the same p.

This note reruns the core analyses under the qubit translation of that model,
`src/lrb_qubit/circuits_lit.py`.

## The model (mirrors `CircuitLevelNoiseRates` / `apply_circuit_level_noise`)

**Segments.** Each logical Clifford, and each check round, is one
barrier-separated segment packed into ASAP moments. A moment holds either
gates or collapse operations, never both.

**Check-round segment.** The segment is, in order:

1. R on ancillas 5–8;
2. the Z blocks;
3. the X blocks.

Ancilla measurements whose wire is not reused are deferred to one final
measurement moment, so all four ancillas read out together.

**Noise per moment** (multiples of p):

| location | channel | rate |
|---|---|---|
| one-qubit gate, including `I` | DEPOLARIZE1 after | 0.1 |
| two-qubit gate | DEPOLARIZE2 after | 1 |
| reset | X_ERROR after | 2 |
| measurement | X_ERROR before | 5 |
| measurement | DEPOLARIZE1 after; never reaches a detector, because ancillas are reset before reuse | 1 |
| untouched wire, gate moment | DEPOLARIZE1 | 0.1 |
| untouched wire, collapse moment | DEPOLARIZE1 | 2 |

**Ideal parts.** The encoding and the terminal readout stay ideal.

**Validation (VN).**

- Noiseless circuits are deterministic at depths 0, 1, 3 and 22, with both
  readouts.
- Stim builds the DEM.
- The transfer-matrix MLD matches brute force, with |ΔLLR| ≤ 4e-15 at depths
  0, 1 and 3.
- The circuit-level distance is 2 with the five-qubit readout.
- The first check round, printed for `HP`, shows one reset moment, interleaved
  CNOT moments, and a single shared measurement moment.

## Exact low-order census (VA + VN; `results/lit_first_order/`, 30 sequences)

The definitions are as in L0 §4.

| depth | R1/p | R1/p, data2 | E1/p | data2 floor U1/p | A2/p² |
|---|---|---|---|---|---|
| 0 | 57.5 | 48.7 | 6.03 | 3.73 | 9.9 |
| 6 | 468.8 | 460.0 | 49.3 | 3.73 | 101 |
| 22 | 1564 | 1556 | 159.6 | 3.73 | 370 |

Per depth step, the literature model compares with the August model as
follows:

| quantity per depth step | literature SI1000 | August SI1000 (L0) |
|---|---|---|
| LRB-D first-order rejection | 68.5p | 105p |
| full-MLD first-order error | 7.0p | 11.8p |
| LRB-D accepted error | 16.4p² | 48p² |
| data2 floor | 3.73p | 4.53p |

Components at depth 22:

| component | R1/p | E1/p |
|---|---|---|
| idle during reset and measurement moments (2p) | 457 | 83.3 |
| measurement flip (5p) | 460 | **0** |
| two-qubit gate depolarizing (p) | 329 | 47.1 |
| reset flip (2p) | 184 | **0** |
| gate-moment idle (0.1p) | 93 | 11.6 |
| one-qubit gate depolarizing (0.1p) | 42 | 5.1 |
| post-measurement depolarizing (p) | 0 | 0 |

What follows:

- **The qualitative conclusions are unchanged.** Time-like faults contribute
  nothing to the MLD error. Collapse-moment idling is still the largest
  residual (52%). Two-qubit gates are second (30%).
- **The data2 floor persists**, at 3.7p, from data idling through the shared
  measurement moment after its last X coupling.
- **Full correction remains far worse than a bare qubit.** Under the user's
  newer same-p comparison, physical RB has an error per depth step of about
  0.34p, against about 7p for full MLD correction. That is 20 times worse.

## Sampled grid (VN; `results/lit_grid/`, `lit_fits.json`)

Settings: 30 sequences, depths 0–22 on the main grid, 2e5 shots for LRB-D and
1e5 decoded shots on the same samples. Physical RB uses 5e4 shots at
p_RB = p and at p_RB = 5p.

The table gives the error per depth step, (1 − f)/2, divided by p, from the
repo LRB fit with the offset fixed at 0 and a 200-sample sequence bootstrap
(68% intervals are in the JSON). A(22) is the acceptance at m = 22.

| method | p = 5e-4 | 1e-3 | 2e-3 | 5e-3 | A(22) at 1e-3 |
|---|---|---|---|---|---|
| LRB-D unif=1 (data2) | 0.0105 | 0.018 | 0.043 | 0.118 | 0.21 |
| LRB-D unif=1 + final check | 0.0081 | 0.016 | 0.033 | 0.090 | 0.21 |
| LRB-D const=1 | 0.23 | 0.46 | 0.79 | 2.7 | 0.43 |
| LRB-D unif=4 | 0.036 | 0.070 | 0.16 | 0.47 | 0.36 |
| full exact MLD (mld5) | 7.17 | 7.30 | 7.53 | 8.16 | 1 |
| MLD, accept \|LLR\| ≥ 4 | 0.075 | 0.109 | 0.146 | 0.198 | 0.54 |
| MLD, accept \|LLR\| ≥ 2 | 0.54 | 0.68 | 0.74 | 0.71 | 0.69 |
| const0, postselect on ideal X readout | 0.0081 | 0.012 | 0.021 | 0.052 | 0.91 |
| const0, decoded | 1.29 | 1.28 | 1.28 | 1.28 | 1 |
| physical RB, p_RB = p | 0.35 | 0.33 | 0.34 | 0.34 | 1 |
| physical RB, p_RB = 5p | 1.72 | 1.70 | 1.69 | 1.70 | 1 |

What this shows:

- **Full correction loses badly.** Under the user's current conventions
  (literature SI1000, physical RB at the same p), full correction is 21–24
  times worse per depth step than a bare qubit.
- **No-check correction loses too.** The no-check decoded protocol is 3.8
  times worse. It does better under the August model (0.74p) because that
  model has no idle noise during the logical Cliffords.
- **Only postselection beats the bare qubit here.** Every-round LRB-D, the
  soft-output protocols with |LLR| ≥ 4, and the no-check postselected
  protocol all beat the bare qubit per step at every p tested. The cost is
  acceptance: at p = 5e-3 and m = 22, every-round LRB-D accepts 4e-4, the
  |LLR| ≥ 4 rule accepts 2.5e-3, and no-check LRB-D accepts 0.65.
- **The ordering of methods is the same** as under the August model (L0 §3).
