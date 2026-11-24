# WP H: check schedules and noise ablations: do checks help?

Labels: **VA** verified analytically, **VN** verified numerically, **LIT** from
literature, **C** conjecture.

Conventions:

- Rates are per depth step (one forward Clifford, one inverse Clifford and the
  check rounds in between), r = (1 − f)/2, in units of p.
- `x(y)` means x ± y in the last digits. Sampled errors are sequence-bootstrap
  standard deviations; exact first-order errors are s.e.m. over the 10
  sequences.
- "MLD5" is exact maximum-likelihood decoding with the final five-qubit X
  readout. "MLD2" uses only the check rounds (the qutrit pipeline's
  two-qubit readout).
- "LRB-D" postselects on every executed round (data2 semantics); "+final" also
  requires a clean final readout.

## Summary

**Headline.** Under the repo's SI1000 model, mid-circuit checks on the d=2
code never help decoded LRB. Sparse checks help postselected LRB (LRB-D).

The best hardware schedule depends on the method:

- **Decoded LRB:** no check rounds. Read all five data qubits out in the X
  basis at the end and decode the two final parities (`const0`): r = 0.74(3)p
  at p = 5e-4, 2e-3 and 5e-3. That is a factor 6 to 19 below any
  every-round-checked schedule.
- **LRB-D:** read all four ancillas in one parallel measurement layer, skip
  the reset, place a few sparse checks split between the forward and inverse
  halves (about every 8 to 11 Cliffords in each half), and postselect on the
  final readout as well.
  - The verifier's exact evaluation at p = 5e-3, m = 22 gives 0.45 to 0.46%
    accepted infidelity at A(22) ≈ 0.20 to 0.25. Example: forward unif=8 plus
    inverse every 8 gives 0.464% at A = 0.203.
  - My own search covered only forward-half placements. There the plateau is
    every 3 to 11 Cliffords, at 0.57(3) to 0.63(3)% with acceptance 0.10 to
    0.32.
  - For comparison: a check every round gives 0.88(6)% at A = 0.0037, and no
    checks give 1.89(14)% at A = 0.69.

Findings:

1. **The checks are the noise.** A d=2 check round adds more logical error
   than its syndrome removes (VN, exact DEM, 10 sequences).
   - Decoded error falls monotonically as checks are removed: serial
     11.76(19)p → unif=11 1.96(7)p → terminal only 0.95(3)p → none
     0.73(3)p.
   - Noiseless check rounds every round give 0.72(3)p, a paired gain of only
     0.010(1)p over no checks.
   - With gate noise fixed, every-round checks beat no checks only if
     check-round noise is scaled below about 6e-4 (serial) or 1.7e-3
     (parallel, no reset) of SI1000. That is checks about 500 to 1700 times
     quieter, at first order and with about ±25% from sequence sampling (the
     verifier's new sequences give 7.5e-4 and 2.2e-3).
   - Sampled results agree at all three p (sections 2.2 and 2.3).
2. **One parallel measurement layer is the most effective schedule change.**
   It cuts the data qubits' idle layers per round from 5 to 2 (VA; 1 without
   reset).
   - Decoded r falls from 12.24(20), 13.08(21), 14.26(22)p (serial) to
     6.52(10), 6.79(12), 7.36(12)p (parallel) and 4.62(8), 4.79(9),
     5.11(8)p (parallel, no reset) at p = 5e-4, 2e-3, 5e-3 (VN).
   - LRB-D unif=1 acceptance at m = 22 grows ×1.5, ×5.5 and ×67 (parallel)
     and ×1.9, ×12 and ×466 (no reset) (VN).
3. **Literature-SI1000 analogue.** The `parallel` variant is the closest
   analogue of the user's literature-exact SI1000 with ASAP scheduling and
   resets before each round: one measurement layer and one reset layer per
   round.
   - Define the **every-round check penalty (decoded)** as r(MLD5, every round
     checked) − r(no-check decoding, `const0` MLD5). No-check decoding also
     corrects, so this is the cost of a noisy check round every step, not of
     correction as such.
   - At first order the penalty is 11.0p for the serial schedule. `parallel`
     leaves 5.5p and parallel without reset leaves 3.7p, removing 49.7% and
     66.3% (11.76, 6.28, 4.45 against 0.73p).
   - Sampled at p = 5e-4, 2e-3, 5e-3, the serial penalty is 11.5, 12.3 and
     13.5p. `parallel` removes 49.8(3)%, 51.0(2)% and 51.0(2)% of it and
     leaves 5.8 to 6.6p, still 8 to 9 times the no-check error. Dropping the
     reset as well removes 66.3(2)%, 67.2(2)% and 67.7(2)% (VN).
   - Gate-layer idle noise is not in this model; the lead's literature
     variant covers it.
4. **Dominant term: data idle during the four serial measurement layers.**
   - It is 7.31(14)p of the serial 11.76(19)p first-order decoded error
     (62%). The other first-order terms are reset-layer idle 1.82(4)p, CNOTs
     1.59(3)p and logical gates 1.36(5)p.
   - Measurement flips, reset flips, measured-qubit depolarizing, ancilla
     idle and ancilla H contribute exactly 0 at first order: MLD corrects
     them (VN, exact).
   - After parallelisation the four remaining terms are comparable: 1.83,
     1.83, 1.61 and 1.30p.
   - For LRB-D rejection, measurement flips (20p of 67p per step) then
     dominate (VN, exact; sampled ablations in 2.5).
5. **Executing only the rounds a policy uses** is a large gain over the repo
   protocol, where every round runs and the policy only selects (VN).
   - Example, `const=1` at p = 5e-3, serial: F(22) 0.024(2) → 0.931(1) and
     A(22) 0.063 → 0.462. This matches the verifier's exact values. The fitted
     f of the repo-protocol curve (0.901 here) depends on the fit weights: the
     exact data give 0.886, and the curve is not exponential.
   - Decoding the executed circuit gives the same monotone picture as in
     item 1.
6. **Inverse-half checks: neutral for decoding, a real gain for LRB-D.**
   - Decoding: moving about half of the check rounds into the inverse half
     changes the first-order decoded error by +0.11(22)p, −0.25(9)p,
     +0.11(10)p and −0.12(3)p per step (21 vs 22 and 10 vs 11 rounds at
     m = 22). At exactly equal round count the verifier finds a change under
     1% of E1. Adding inverse checks on top of the forward ones roughly
     doubles the cost (VN, exact).
   - LRB-D: at equal round count and unchanged acceptance, the split
     placement lowers the accepted infidelity by 15 to 28%. The verifier finds
     this exactly, in 10/10 sequences at 2e-3 and 5e-3, with and without
     reset.
   - My sampled data agree: with 3 rounds plus the terminal at m = 22
     (parallel, with reset), unif=11 plus inverse every 11 gives 0.096(6)%
     against 0.115(7)% for forward-only unif=6 at 2e-3, and 0.665(42)%
     against 0.859(46)% at 5e-3, at the same A(22) (VN).
7. **Validation.**
   - The serial builder reproduces the lead's circuits instruction for
     instruction.
   - The serial and `const0` counts agree with the lead's `grid_main` within
     binomial noise (24 χ² comparisons).
   - The data2 undetectable floor is 68p/15 (serial) and 48p/15 (zx and
     parallel with the repo's CNOT order). With the X-check CNOTs first it is
     54p/15, per the verifier (VA + VN).

## 1. Methods

### 1.1 Circuit variants (`src/lrb_qubit/variants.py`)

`build_variant_circuit` re-implements `circuits.build_lrb_circuit` with four
extra options. With the defaults it emits the same Stim circuit as the lead's
builder, instruction for instruction (56 cases in
`test_serial_reproduces_lead_builder`). All noise goes through
`circuits.emit_operations`. Bookkeeping is unchanged (`rounds`,
`detector_keys`, `back`, `segments`), so
`mld.TransferMatrixMLD(dem, mld.lrb_detector_slots(lc))` and
`lc.raw_syndromes` work unchanged. Every variant builds a valid transfer-matrix
decoder at depths 0, 1, 6 and 22: no DEM mechanism spans non-adjacent slots.

| option | values |
|---|---|
| `schedule` | `serial` (repo: M5, M6, M7, M8 each its own layer); `zx` (M5 M6 after both Z checks, then M7 M8 after both X checks); `parallel` (all 12 CNOTs, then M5 M6 M7 M8 in one layer). The CNOT order is always the repo's. |
| `reset=False` | Measure without reset after forward and inverse rounds. The raw stabilizer value is the ancilla's current record XOR its previous record, and the detectors use these XORs. |
| `forward_check_rounds`, `inverse_check_every` | As in the lead's builder. `VariantSpec(forward=("unif", k))` executes only the rounds `unif=k` uses, `("const", k)` likewise, and `("terminal", 0)` executes only the terminal round. |
| `NoiseConfig` | Separate SI1000 factors (`SI1000.scaled`) for check rounds and for logical Clifford gates, plus multipliers that split measurement-layer idle between data qubits and idle ancillas. |

Variant names in the result files:

- `sched-{serial,zx,parallel,parallel-nr}`: every forward round executed (the
  repo protocol). `-nr` means no reset.
- `int-<sched>-unif<k>`, `int-<sched>-const<k>`, `int-<sched>-term`: only the
  forward rounds used by that policy are executed.
- `const0`: no check rounds, ideal X readout of all data. This is the lead's
  circuit.
- `inv-<sched>-k<k>`: forward rounds of `unif=k`, plus a check after every k-th
  inverse Clifford. k = 1 means every forward and every inverse round.
- `abl-<sched>-<term>`: every forward round, with one noise term removed.

### 1.2 Per-round noise budget (VA, counting; `test_data_idle_layers_per_round`)

Each entry is the sum of channel probabilities in one forward check round, in
units of p.

| term | serial | zx | parallel | parallel, no reset |
|---|---|---|---|---|
| data idle in measurement layers (DEPOLARIZE1 2p) | 40 (4 layers) | 20 (2) | 10 (1) | 10 (1) |
| ancilla idle in measurement layers | 24 | 8 | 0 | 0 |
| data idle in the reset layer (2p) | 10 | 10 | 10 | 0 |
| reset flips X_ERROR(2p) | 8 | 8 | 8 | 0 |
| measurement flips X_ERROR(5p) / measured-qubit DEPOLARIZE1(p) | 20 / 4 | 20 / 4 | 20 / 4 | 20 / 4 |
| CNOT DEPOLARIZE2(p) / ancilla H (0.1p) | 12 / 0.4 | 12 / 0.4 | 12 / 0.4 | 12 / 0.4 |
| **data-qubit idle layers per round** | **5** | **3** | **2** | **1** |

In the serial schedule, half of the ancilla idle lands after the ancilla's own
measurement. With reset that half is erased.

**Measure-without-reset (VA + VN).** In the parallel schedule no noise acts on
an ancilla between its measurement and its next CNOT. SI1000's measurement
error is an `X_ERROR` before `M`, which is a physical flip, so it flips exactly
one raw syndrome with or without reset. Dropping the reset is therefore
exactly equivalent to deleting the reset layer's noise.
`test_parallel_no_reset_equals_reset_without_reset_noise` checks this: the
exact joint distribution of (detectors, logical) is identical at depths 1
and 2. This does not hold for the serial schedule: there, idle noise after an
ancilla's measurement persists to the next round
(`test_serial_no_reset_differs_from_reset_ablation`).

### 1.3 Methods scored on the same shots (`variants.run_variant`)

- `lrbd`: the qutrit pipeline's rule. Accept when every *executed* round's raw
  syndrome is trivial (data2 semantics). For `sched-*` all 44 const/unif
  policies are scored on the same shots.
- `lrbd_final`: `lrbd` plus a clean final five-qubit X readout. For `const0`
  the repo's rule (final X parities clean) is `lrbd`.
- `mld5` / `mld2`: exact transfer-matrix MLD with or without the final data
  detectors, on the first 1e5 shots of the same sample.
- `raw`: no decoding and no rejection.

### 1.4 Statistics

- Sequences: `experiment.clifford_sequences(10, 22, 20261028)`. These are the
  first 10 of the main grid's 30.
- Depths: 0, 2, 4, 6, 10, 14, 18, 20, 22. Noise: p ∈ {5e-4, 2e-3, 5e-3}.
- Shots per circuit:
  - `sched-*`: 1e6 for LRB-D.
  - `int-*`, `inv-*`, `const0`: 3e5.
  - Ablations: 1e5.
  - The first 1e5 shots of every circuit are decoded.
- F per sequence is |1 − 2·errors/accepted|. F(m) and A(m) are the mean over
  sequences ± s.e.m.
- Fit: F = b f^m with the offset fixed at 0. This copies the repo's
  `LRBResultsPlotter._fit_decay_parameter_fixed_a`, vectorised: weights
  1/std² over sequences, points with F > 1 dropped, and a 2001-point × 5-round
  grid scan. `test_fit_matches_repo_implementation` calls the repo method
  itself and agrees to 2e-6 in f.
- Errors on f are the standard deviation over 1000 sequence-bootstrap
  resamples (each refitted with 401 points × 6 rounds, the same final grid
  spacing).
- At each p every variant uses the same resample indices, so all differences
  and ratios between variants are paired.
- LRB-D sampling cost: g is fitted to sqrt(A)·F per sequence (shots needed
  ∝ g^(−2m)), and α is fitted to A(m).
- First-order coefficients use the lead's definitions (`first_order.py`),
  re-implemented as `variants.first_order_coefficients`. They reproduce his
  per-sequence values at depths 0 and 22 to all printed digits.
  - E1 = Σ_s min(P0, P1) + undetectable mass: the exact first-order MLD error.
    `mld2` restricts symptoms to the round detectors.
  - R1 = Σ_s (P0 + P1): the first-order LRB-D rejection on all executed rounds
    plus the final check.
  - A2 = Σ_s P0·P1: the second-order accepted LRB-D error.
  - Slopes are least-squares lines over the nine depths, one per sequence,
    reported as mean ± s.e.m. Differences are paired per sequence.

## 2. Results

Full machine-readable output is in `results/H/`:

- `summary.json`: every variant × p × method, with F(m), A(m), fits,
  bootstrap errors, paired differences and penalty fractions.
- `fm_long.csv` and `fits_long.csv`.
- `tables.md`: all sampled tables, including every policy and the
  executed-vs-full comparison.
- `first_order_summary.json`, `first_order_rows.csv`, `first_order_tables.md`,
  and `first_order_checkscan_*`.
- `crosscheck_grid_main.csv`.
- `tasks/`: raw counts per (variant, p, sequence).

### 2.1 Validation (VN)

- 154 tests pass (`tests/test_qubit512_variants.py`, 15 to 30 s). They cover:
  - byte-identical serial circuits;
  - deterministic noiseless circuits for every variant;
  - transfer-matrix MLD equal to brute-force inference (|ΔLLR| < 1e-9) for
    zx, parallel, no-reset, inverse-check and sparse-round circuits;
  - circuit distance 2 with the five-qubit readout for every schedule;
  - raw syndromes equal to the measurement records, with and without reset;
  - the no-reset equivalence (1.2);
  - the fit against the repo's own code.
- Cross-check with the lead's `grid_main` (same circuits, independent shots),
  per (sequence, depth) two-proportion χ² on error and acceptance fractions.
  - 23 of 24 comparisons (mld5, mld2, unif=1, unif=1+final, const=1, unif=4,
    const0 and const0_mld at three p) have p-values 0.10 to 0.90.
  - `const0` LRB-D at 5e-4 has p = 0.025. Its accepted-error counts are 0 to
    15 per cell, where the normal approximation fails. Exact Fisher tests
    give 3 of 155 cells below 0.01, against 1.6 expected (P(≥3) ≈ 0.2).
- The sampled serial rates match the lead's table: 13.08(21)p against 13.1p
  for mld5 at 2e-3, and 0.13p (0.11p with the final check) for LRB-D unif=1.

### 2.2 Schedules, every forward round checked

Sampled, 10 sequences. "First order" is the exact E1 or R1 slope.

| schedule | MLD5 r/p at 5e-4 | at 2e-3 | at 5e-3 | MLD5 first order | LRB-D rejection per step, first order | rejection per step / p, sampled at 5e-4 | data2 floor / p |
|---|---|---|---|---|---|---|---|
| serial (repo) | 12.24(20) | 13.08(21) | 14.26(22) | 11.76(19) | 105.01(18) | 102.2(2) | 68/15 = 4.53 |
| zx | 8.37(15) | 8.83(14) | 9.63(14) | 8.08(13) | 79.67(18) | 78.2(2) | 48/15 = 3.20 |
| parallel | 6.52(10) | 6.79(12) | 7.36(12) | 6.28(10) | 67.01(18) | 66.0(2) | 3.20 |
| parallel, no reset | 4.62(8) | 4.79(9) | 5.11(8) | 4.45(7) | 49.01(18) | 48.4(2) | 3.20 |
| no checks (`const0`) | 0.74(3) | 0.74(3) | 0.74(3) | 0.73(3) | 3.54(14) | – | – |

MLD2 is 0.06 to 0.26p worse than MLD5 throughout (tables.md). Undecoded (`raw`) r/p:

| schedule | 5e-4 | 2e-3 | 5e-3 |
|---|---|---|---|
| serial | 20.98(24) | 20.34(23) | 19.17(24) |
| parallel | 11.70(14) | 11.50(14) | 11.12(13) |
| parallel, no reset | 8.57(12) | 8.46(12) | 8.27(11) |

The fitted MLD5 decays f, with paired differences vs serial:

| schedule | p = 5e-4 | p = 2e-3 | p = 5e-3 |
|---|---|---|---|
| serial | 0.9878(2) | 0.9477(9) | 0.8574(22) |
| zx | 0.9916(1) | 0.9647(6) | 0.9037(14) |
| parallel | 0.9935(1) | 0.9728(5) | 0.9264(12) |
| parallel, no reset | 0.9954(1) | 0.9809(3) | 0.9489(8) |
| fraction of the every-round check penalty (decoded) removed by `parallel` | 49.8(3)% | 51.0(2)% | 51.0(2)% |
| fraction removed by parallel, no reset | 66.3(2)% | 67.2(2)% | 67.7(2)% |
| fraction removed by `zx` | 33.7(3)% | 34.4(2)% | 34.2(3)% |

The every-round check penalty (decoded) is r(MLD5, every round checked,
serial) − r(no-check decoding, `const0` MLD5). The fraction is
(f_sched − f_serial)/(f_const0 − f_serial), with a paired bootstrap.

- Absolute sizes, first order: 11.0p (serial) → 5.5p left (parallel) → 3.7p
  left (parallel, no reset). First-order fractions 49.7% and 66.3%.
- Absolute sizes, sampled at 5e-4, 2e-3, 5e-3: 11.5, 12.3, 13.5p (serial).
- The MLD2 fractions in `tables.md` are within 0.01 of these. They use the
  same `const0` MLD5 reference, so they mix two decoders: `const0` has no
  data2 analogue.
- The name: no-check decoding also corrects (it decodes the final X
  parities), and perfect checks bring E1 to 0.01p below `const0`. So the
  whole penalty is check-round noise, not a cost of correction.

LRB-D unif=1, every round, data2 semantics:

| schedule | A(22) at 5e-4 | A(22) at 2e-3 | A(22) at 5e-3 | rejection per round at 2e-3 / at 5e-3 | 1 − F(22), +final, at 5e-3 |
|---|---|---|---|---|---|
| serial | 0.306(1) | 8.92(7)e-3 | 8.2(8)e-6 | 0.1888(3) / 0.4052(8) | 5.8(39)% |
| zx | 0.408(1) | 0.028 | 1.32(4)e-4 | 0.1471(4) / 0.3267(7) | – |
| parallel | 0.471(1) | 0.049 | 5.49(9)e-4 | 0.1254(4) / 0.2839(7) | 1.80(21)% |
| parallel, no reset | 0.574(1) | 0.108(1) | 3.82(8)e-3 | 0.0936(4) / 0.2179(8) | 0.88(6)% |

The data2 floor (VA + VN) is the first-order undetectable logical mass.

- In the serial schedule it is 68p/15: data 0 idles in the M7 and M8 layers
  after its X coupling, data 3 in M8, and the Z/Y components on data 0 or 3 of
  DEPOLARIZE2 after CNOT 7→0 and 8→3 add 8p/15.
- What counts is the number of measurement layers after the X-check CNOTs on
  data 0 and 3. zx (two layers) and parallel (one layer) both give
  2·(4p/3) + 8p/15 = 48p/15 with the repo's CNOT order. The verifier's
  one-layer variant with the X-check CNOTs first gives 54p/15.
- Stim's `explain_detector_error_model_errors` lists exactly these mechanisms.
- It sets the LRB-D (data2) F(0): 0.981 against 0.987 at 2e-3.

### 2.3 Check interval: physically executing only the policy's rounds

Decoded r/p (MLD5, sampled). The first-order slopes for the same circuits are
in `first_order_tables.md`; for example, parallel without reset gives 4.45,
2.86, 2.14, 1.94, 1.57, 1.26 and 0.79 for every round, unif = 2, 3, 4, 6, 11
and terminal only.

| forward checks | serial 5e-4 | serial 2e-3 | serial 5e-3 | parallel 5e-4 | parallel 2e-3 | parallel 5e-3 | par-nr 5e-4 | par-nr 2e-3 | par-nr 5e-3 |
|---|---|---|---|---|---|---|---|---|---|
| every round | 12.24(20) | 13.08(21) | 14.26(22) | 6.52(10) | 6.79(12) | 7.36(12) | 4.62(8) | 4.79(9) | 5.11(8) |
| unif=2 | 6.95(16) | 7.50(15) | 8.28(15) | 3.94(8) | 4.15(9) | 4.51(8) | 2.96(6) | 3.07(6) | 3.29(6) |
| unif=3 | 4.18(20) | 4.46(22) | 5.08(17) | 2.69(7) | 2.78(8) | 3.05(8) | 2.16(6) | 2.24(6) | 2.39(6) |
| unif=4 | 3.76(10) | 4.05(11) | 4.48(12) | 2.43(4) | 2.53(6) | 2.75(5) | 1.98(3) | 2.02(4) | 2.17(4) |
| unif=6 | 2.60(9) | 2.75(12) | 3.07(12) | 1.82(8) | 1.89(6) | 2.06(7) | 1.57(4) | 1.63(5) | 1.72(5) |
| unif=11 | 1.69(7) | 1.75(8) | 1.94(10) | 1.33(6) | 1.39(6) | 1.45(6) | 1.23(4) | 1.27(5) | 1.32(5) |
| const=2 | 1.94(7) | 2.04(7) | 2.25(9) | 1.46(6) | 1.50(5) | 1.58(5) | 1.26(6) | 1.31(5) | 1.36(6) |
| terminal only | 1.04(5) | 1.09(4) | 1.12(4) | 0.87(4) | 0.91(4) | 0.91(3) | 0.90(4) | 0.90(4) | 0.92(3) |
| **none (`const0`)** | **0.74(3)** | **0.74(3)** | **0.74(3)** | | | | | | |

**Optimal interval for decoded LRB: no mid-circuit checks (VN).**

- r falls monotonically as checks are removed, at every tested p and for
  every schedule, along the nested unif sequence (every round ⊃ unif=2 ⊃ …
  ⊃ terminal ⊃ none). The verifier also finds that every one of 2,640 random
  single-check additions at depth 22 raised E1 (first order).
- The const rows are not nested with the unif rows and are not comparable
  per step: const=2 jumps from 0 to 1 forward round between m = 0 and m = 2.
  So const=2 (2.25p) lying above unif=11 (1.94p) at 5e-3 is not a
  violation.
- With the final readout available, even the terminal round costs 0.13 to
  0.38p per step (sampled): paired first-order Δ +0.065(2)p (parallel, no
  reset) and +0.221(4)p (serial) per step relative to `const0`.
- The terminal round's own noise also lowers F(0), to between 0.893 (serial,
  5e-3) and 0.996 (parallel, 5e-4), against 1 for `const0`.
- `const0` is 2.3 times better than physical RB at 5p: 0.74p against 1.69p,
  from the lead's L0 note.

LRB-D on the same executed circuits, with the final-readout check: accepted
infidelity 1 − F(22) in %, acceptance A(22), and cost-adjusted decay g of
sqrt(A)·F. Values are mean(s.e.m.) or fit(bootstrap).

| p | variant | 100(1 − F(22)) | A(22) | g |
|---|---|---|---|---|
| 2e-3 | serial, every round | 0.963(58) | 0.00871 | 0.9002(2) |
| 2e-3 | parallel, no reset, every round | 0.129(5) | 0.106 | 0.9520(2) |
| 2e-3 | parallel, no reset, unif=4 | 0.093(3) | 0.483 | 0.9856(4) |
| 2e-3 | parallel, no reset, unif=11 | **0.072(6)** | 0.631 | 0.9929(3) |
| 2e-3 | parallel, no reset, terminal only | 0.107(7) | 0.754 | 0.9953(2) |
| 2e-3 | none (`const0`) | 0.289(24) | 0.858 | **0.9964(2)** |
| 5e-3 | serial, every round | 5.8(39) | 7.7e-6 | 0.7686(5) |
| 5e-3 | parallel, every round | 1.80(21) | 5.3e-4 | 0.8455(4) |
| 5e-3 | parallel, no reset, every round | 0.880(61) | 0.00366 | 0.8840(5) |
| 5e-3 | parallel, no reset, unif=4 | 0.632(27) | 0.163 | 0.9643(6) |
| 5e-3 | parallel, no reset, unif=6 | 0.614(34) | 0.254 | 0.9735(7) |
| 5e-3 | parallel, no reset, unif=11 | **0.568(32)** | 0.317 | 0.9821(7) |
| 5e-3 | parallel, no reset, terminal only | 0.785(59) | 0.498 | 0.9884(5) |
| 5e-3 | serial, unif=11 | 1.96(12) | 0.160 | 0.9738(11) |
| 5e-3 | none (`const0`) | 1.89(14) | 0.694 | **0.9909(4)** |

At 5e-4 every LRB-D variant has 1 − F(22) ≤ 0.053%; the lowest values are
0.005% for parallel without reset with unif = 2 to 11.

**Optimal interval for LRB-D (VN).** It depends on the objective.

- Lowest accepted infidelity among *forward-half* placements: a plateau of
  sparse checks every 3 to 11 Cliffords in the parallel no-reset schedule.
  - They beat checking every round (1.5 to 1.8 times lower 1 − F(22) at 2e-3
    and 5e-3) and no checks (3.3 to 4 times lower).
  - The second-order coefficient A2 explains why: 4.2 to 4.5p² per step for
    unif = 3 to 11 against 6.65p² every round, while `const0` grows like m².
  - Order inside the plateau depends on where the checks fall relative to m.
    unif=11 wins at m = 22 because it checks right after Clifford 22. At
    m = 20 unif=4 is better (0.526% against 0.590%, verifier, exact).
- **Splitting the checks between the forward and inverse halves does better**
  (verifier, exact, p = 5e-3, m = 22, parallel without reset):
  - forward unif=8 plus inverse every 8 (5 rounds): 0.464% at A = 0.203;
  - forward unif=11 plus inverse every 11 (4 rounds): 0.448% at A = 0.253,
    against 0.607% at A = 0.254 for forward-only unif=6 (also 4 rounds).
  - The best placement found is therefore about 0.45 to 0.46%. See 2.4.
- Lowest sampling cost: no checks (`const0`), then terminal only.
- Fitted f is not a good LRB-D summary here. The curves without mid-circuit
  checks are not exponential: `const0` LRB-D at 5e-3 falls from 1.000 to 0.981
  roughly quadratically. The repo fit is pinned at F(0) = 1, where the
  sequence std is 0, so it returns f = 0.99976(6). All LRB-D f lie in
  0.9984 to 1.0000 for the executed circuits, so F(22) and A(22) carry the
  comparison.

**Repo protocol vs executed circuits (item 2, LRB-D data2 semantics).**
Running only the rounds a policy uses improves both accepted fidelity and
acceptance. F(22) and A(22) are sampled mean(s.e.m.), and agree with the
verifier's exact values within 0.004. Every pair is significant.

| p | schedule | policy | all rounds run: F(22), A(22) | only used rounds run: F(22), A(22) | fitted f, all rounds run → used rounds (fit-dependent) |
|---|---|---|---|---|---|
| 2e-3 | serial | unif=4 | 0.939(1), 0.0363 | 0.976(0), 0.268 | 0.9980 → 0.9998 |
| 2e-3 | serial | const=1 | 0.560(3), 0.105 | 0.978(0), 0.728 | 0.9900(21) → 0.9999 |
| 5e-3 | serial | unif=4 | 0.642(6), 3.7e-4 | 0.917(2), 0.0384 | 0.9832(4) → 0.9984 |
| 5e-3 | serial | unif=11 | 0.190(11), 3.0e-3 | 0.925(2), 0.171 | 0.9436(42) → 0.9990 |
| 5e-3 | serial | const=1 | 0.024(2), 0.063 | 0.931(1), 0.462 | 0.9014(67) → 0.9992 |
| 5e-3 | parallel, no reset | unif=4 | 0.928(1), 0.032 | 0.959(0), 0.170 | 0.9982(1) → 0.9996 |
| 5e-3 | parallel, no reset | const=1 | 0.573(5), 0.108 | 0.956(1), 0.521 | 0.9933(14) → 0.9995 |

The repo-protocol curves are far from exponential; for unif=11, F(m) is not
even monotone. Their fitted f depends on the fit weights.

- For `const=1` at 5e-3, the repo fit gives 0.9014(67) with the sampled
  stds, 0.886 with exact data, and 0.899 unweighted (verifier).
- The bootstrap errors understate this dependence.
- So F(22) and A(22), not f, carry the comparison.

In the repo protocol the unused rounds still inject their noise. The used
rounds then see that noise but cannot locate it. This is consistent with the
fast, non-exponential `const=1` decay in the qutrit production data (C).

### 2.4 Inverse-half checks

Decoded r/p (MLD5) and LRB-D with the final check, forward unif=k plus an
inverse check every k. k = 1 means every forward and every inverse round.

| schedule | k | 5e-4 | 2e-3 | 5e-3 | first-order Δ vs forward only with nearly the same round count (rounds at m = 22, split vs forward only) |
|---|---|---|---|---|---|
| serial | 1 | 22.15(44) | 23.31(39) | 23.74(38) | (2 rounds per step; no forward-only match) |
| serial | 2 | 11.54(39) | 12.36(36) | 13.51(34) | +0.11(22) vs every forward round (21 vs 22) |
| serial | 4 | 5.03(43) | 5.78(31) | 6.79(23) | −0.25(9) vs unif=2 (10 vs 11) |
| serial | 11 | 2.00(11) | 2.12(16) | 2.36(15) | – |
| parallel | 1 | 11.28(22) | 11.76(20) | 12.55(18) | – |
| parallel | 2 | 6.20(17) | 6.53(17) | 7.09(16) | +0.11(10) vs every forward round (21 vs 22) |
| parallel | 4 | 3.47(9) | 3.59(10) | 3.96(8) | −0.12(3) vs unif=2 (10 vs 11) |
| parallel | 11 | 1.53(6) | 1.53(7) | 1.66(8) | – |

LRB-D with the final check, 100(1 − F(22)) and A(22). Rounds are counted at
m = 22 without the terminal round.

| comparison | p | split (forward + inverse) | forward only |
|---|---|---|---|
| parallel: unif=11 + inverse every 11 (3 rounds) vs unif=6 (3 rounds) | 2e-3 | 0.096(6), A 0.519 | 0.115(7), A 0.518 |
| same | 5e-3 | 0.665(42), A 0.194 | 0.859(46), A 0.196 |
| serial: same placements | 2e-3 | 0.274(20), A 0.389 | 0.300(14), A 0.390 |
| same | 5e-3 | 2.00(12), A 0.097 | 2.29(10), A 0.097 |
| parallel: k = 4 (10 rounds) vs unif=2 (11 rounds) | 2e-3 | 0.143(9), A 0.217 | 0.163(14), A 0.191 |
| same | 5e-3 | 1.07(9), A 0.022 | 1.15(9), A 0.016 |

- **Decoding:** inverse-half placement is neutral. Under 1% of E1 at exactly
  equal round count, with mixed sign (verifier). Adding inverse checks on top
  of the forward ones costs like any other round.
- **LRB-D:** at equal round count, the split placement lowers the accepted
  infidelity at unchanged acceptance. The sampled gain is 9 to 23% from
  unpaired means; with 10 vs 11 rounds, unif=2 also has lower acceptance.
- The verifier's exact, paired evaluation at 10 vs 10 rounds gives −15.5(10)%
  and −17.7(10)% (parallel) and −24.2(11)% and −26.7(12)% (parallel, no
  reset) at 2e-3 and 5e-3. That holds in 10/10 sequences, and acceptance
  changes by at most 2e-4.
- The reason (C): an inverse-half check catches errors from the inverse
  Cliffords before they pair up with forward-half errors. The A2 ratio
  split/forward-only is 0.78 to 0.94 (verifier).
- **Conclusion for LRB-D:** put some checks in the inverse half. My earlier
  statement that the missing inverse-half checks were "not a limitation worth
  fixing" was wrong (VN, verifier; see Revision log).

### 2.5 Noise-component ablations: which term dominates

**Exact first order** (`first_order_tables.md`). These are leave-one-out
changes of the per-step slopes; ΔE1 is the decoded error and ΔR1 is the LRB-D
rejection, both in units of p.

| term removed | serial ΔE1 | serial ΔR1 | parallel ΔE1 | parallel ΔR1 |
|---|---|---|---|---|
| idle in measurement layers (all wires) | **−7.31(14)** | **−48.00** | −1.83(4) | −10.00 |
|   … data qubits only | −7.31(14) | −40.00 | −1.83(4) | −10.00 |
|   … idle ancillas only | 0.00 | −8.00 | 0.00 | 0 |
| measurement flips X_ERROR(5p) | 0.00 | −20.00 | 0.00 | **−20.00** |
| measured-qubit DEPOLARIZE1(p) | 0.00 | −2.67 | 0.00 | −2.67 |
| CNOT DEPOLARIZE2(p) | −1.59(3) | −11.47 | −1.61(3) | −11.47 |
| ancilla H (0.1p) | 0.00 | −0.27 | 0.00 | −0.27 |
| data idle in the reset layer | −1.82(4) | −10.00 | −1.83(4) | −10.00 |
| reset flips X_ERROR(2p) | 0.00 | −8.00 | 0.00 | −8.00 |
| logical-gate noise (0.1p and SWAP/CZ p) | −1.36(5) | −4.61(18) | −1.30(5) | −4.61(18) |
| all check-round noise ("perfect checks") | −11.05(19) | −100.40 | −5.56(10) | −62.40 |
| full model (E1 / R1) | 11.76(19) | 105.01(18) | 6.28(10) | 67.01(18) |

- R1 is exactly additive.
- The E1 parts sum to 12.09p (serial) and 6.57p (parallel), against the
  totals 11.76p and 6.28p. That is 3 to 5% non-additivity from cross-term
  ambiguity.
- Terms with ΔE1 = 0 change only the rejection: MLD corrects every single
  such fault, because their symptoms carry no logical ambiguity.

**Sampled at finite p** (MLD5 r/p, paired change vs the full model, 10
sequences, 1e5 shots per circuit):

The first row is r/p of the full model. The other rows are paired Δ(r/p)
when the term is removed (VN):

| term removed | serial 5e-4 | serial 2e-3 | serial 5e-3 | parallel 5e-4 | parallel 2e-3 | parallel 5e-3 |
|---|---|---|---|---|---|---|
| full model r/p | 12.24(20) | 13.08(21) | 14.26(22) | 6.52(10) | 6.79(12) | 7.36(12) |
| idle in measurement layers (all) | **−7.64(13)** | **−8.25(14)** | **−9.08(15)** | −1.90(4) | −1.97(4) | −2.18(4) |
|   … data qubits only | −7.62(14) | −8.25(14) | −9.01(16) | −1.91(3) | −1.98(4) | −2.18(5) |
|   … idle ancillas only | −0.03(4) | −0.06(3) | −0.13(7) | −0.03(3) | +0.01(2) | −0.01(1) |
| measurement flips | −0.01(5) | −0.16(3) | −0.50(5) | −0.06(3) | −0.12(3) | −0.32(2) |
| measured-qubit depolarizing | −0.03(5) | −0.03(3) | −0.11(8) | −0.03(2) | −0.01(2) | −0.05(2) |
| CNOT depolarizing | −1.70(5) | −1.97(4) | −2.25(8) | −1.78(4) | −1.93(4) | **−2.22(3)** |
| ancilla H | +0.00(5) | +0.02(3) | −0.03(8) | −0.06(3) | +0.02(2) | −0.01(1) |
| reset-layer data idle | −1.92(4) | −2.08(5) | −2.14(7) | −1.91(5) | −1.99(5) | −2.18(5) |
| reset flips | −0.02(5) | −0.07(3) | −0.20(6) | −0.04(2) | −0.05(3) | −0.13(2) |
| logical-gate noise | −1.41(8) | −1.52(7) | −1.50(9) | −1.43(5) | −1.44(7) | −1.49(7) |
| all check-round noise (perfect checks) | −11.51(20) | −12.37(21) | −13.55(22) | −5.79(11) | −6.08(11) | −6.64(11) |

Reading the table:

- **Serial.** Idle in the measurement layers dominates at every p: 62 to 64%
  of r, almost entirely on data qubits. Then come the reset-layer idle, the
  CNOTs and the logical gates, about 1.4 to 2.3p each.
- **Parallel.** The four remaining terms are nearly equal, so no single fix
  beyond parallelisation gains more than a third.
- **Second-order terms.** Terms that are zero at first order (measurement
  flips, reset flips, ancilla idle and depolarizing) appear at finite p and
  grow roughly linearly in p. At 5e-3 measurement flips cost 0.50(5)p
  (serial) and 0.32(2)p (parallel).
- **Perfect checks.** Removing all check-round noise leaves r = 0.73, 0.71
  and 0.71p (serial, by subtraction). That is the no-check value 0.74(3)p:
  even noiseless checks do not help correction at finite p (VN).
- **Additivity.** The disjoint parts sum to 104%, 108% and 111% of r for
  serial, and 111%, 110% and 116% for parallel, at 5e-4, 2e-3 and 5e-3.

LRB-D unif=1 rejection per step (1 − α), divided by p (sampled; the paired
change is the difference from the first row):

| term removed | serial 2e-3 | parallel 2e-3 | serial 5e-3 | parallel 5e-3 |
|---|---|---|---|---|
| full model | 94.4(2) | 62.7(2) | 81.0(2) | 56.8(1) |
| idle in measurement layers (all) | **53.9(2)** | 53.8(2) | **49.5(2)** | 49.6(2) |
|   … data qubits only | 60.8(2) | 53.8(2) | 55.2(2) | 49.5(2) |
| measurement flips | 78.0(2) | **44.9(2)** | 68.7(2) | **41.8(2)** |
| CNOT depolarizing | 85.0(2) | 52.6(2) | 74.2(2) | 48.5(2) |
| reset-layer data idle | 86.3(2) | 53.9(2) | 75.1(2) | 49.5(2) |
| reset flips | 88.0(2) | 55.7(2) | 76.5(2) | 51.1(2) |
| logical-gate noise | 90.6(1) | 58.6(1) | 78.1(2) | 53.6(1) |
| perfect checks | 4.7(2) | 4.6(2) | 4.6(2) | 4.6(2) |

For LRB-D acceptance the largest single term is measurement-layer idle in the
serial schedule, and measurement flips (X_ERROR 5p) in the parallel one. The
sampled per-step rejection lies below the first-order R1 (94 against 105p for
serial) because the acceptance saturates.

### 2.6 When would checks help decoded LRB? Check-noise scan (exact first order)

All check-round SI1000 factors are scaled by C, with logical-gate noise fixed.
Entries are E1 per step in units of p; `const0` is 0.73(3).

| C | serial every round | serial unif=4 | serial unif=11 | par-nr every round | par-nr unif=4 | par-nr unif=11 | par-nr terminal |
|---|---|---|---|---|---|---|---|
| 0 | 0.72(3) | 0.73(3) | 0.73(3) | 0.72(3) | 0.73(3) | 0.73(3) | 0.73(3) |
| 0.01 | 0.89(3) | 0.78(3) | 0.75(3) | 0.78(3) | 0.74(3) | 0.73(3) | 0.73(3) |
| 0.05 | 1.44(4) | 0.96(3) | 0.81(3) | 1.01(3) | 0.81(3) | 0.76(3) | 0.73(3) |
| 0.1 | 2.02(4) | 1.16(3) | 0.89(3) | 1.23(4) | 0.89(3) | 0.79(3) | 0.73(3) |
| 0.2 | 3.15(5) | 1.52(3) | 1.05(3) | 1.61(4) | 1.05(3) | 0.84(3) | 0.74(3) |
| 0.5 | 6.42(10) | 2.43(4) | 1.43(4) | 2.70(5) | 1.41(3) | 1.01(3) | 0.76(3) |
| 1 | 11.76(19) | 3.86(9) | 1.96(7) | 4.45(7) | 1.94(3) | 1.26(3) | 0.79(3) |

Paired against `const0`:

- At C = 0, checking every round gains −0.010(1)p.
- At C = 0.01 it loses +0.16(0)p (serial) and +0.05(0)p (parallel, no reset).
- The crossovers are C* ≈ 6e-4 and 1.7e-3 (VN, first order). The verifier
  confirms E1(C) − E1(const0) is linear in C on a 1e-4 grid, so the
  interpolation is exact.
- The small perfect-check gain sets C*, and it has a 14 to 20% s.e.m. over
  10 sequences. So the factor is about 500 to 1700× at first order, about
  ±25% from sequence sampling: the verifier's new sequences give
  1/C* = 466 to 1341× across schedules.
- At finite p the perfect-check gain stays at −0.011(2)p for p = 2e-3 to
  1e-2, on identical shots (verifier).

At d=2 a check can only *detect* a single fault; the correction stays a guess
between two equally likely cosets. Knowing *when* the fault happened changes
almost nothing (C for the explanation; VN: the C = 0 row). Under this model
check rounds would need to be about 500 to 1700 times quieter than SI1000,
with logical-gate noise kept at SI1000, before they help correction (VN,
first order, ±25% from sequence sampling). The second-order benefit is
real but only matters for postselection (2.3).

### 2.7 F(m) and A(m) of key variants (mean ± s.e.m. over 10 sequences)

Decoded F(m), MLD5:

| p | variant | m=0 | 2 | 4 | 6 | 10 | 14 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5e-4 | serial | 0.989 | 0.962(2) | 0.940(2) | 0.917(2) | 0.872(3) | 0.834(3) | 0.793(4) | 0.774(3) | 0.756(3) |
| 5e-4 | parallel | 0.996 | 0.981(1) | 0.969(1) | 0.956(1) | 0.932(2) | 0.910(2) | 0.887(2) | 0.874(2) | 0.863(2) |
| 5e-4 | parallel, no reset | 0.996 | 0.985(1) | 0.976(1) | 0.967(1) | 0.950(1) | 0.934(1) | 0.917(2) | 0.909(1) | 0.900(1) |
| 5e-4 | `const0` | 1.000 | 0.998 | 0.997 | 0.995 | 0.993(1) | 0.990(1) | 0.987(1) | 0.985(1) | 0.984(1) |
| 2e-3 | serial | 0.956 | 0.851(6) | 0.766(8) | 0.689(6) | 0.556(8) | 0.453(7) | 0.366(7) | 0.328(5) | 0.294(5) |
| 2e-3 | parallel | 0.983 | 0.925(3) | 0.876(5) | 0.828(3) | 0.744(6) | 0.670(6) | 0.600(6) | 0.569(4) | 0.539(4) |
| 2e-3 | parallel, no reset | 0.983 | 0.941(2) | 0.906(3) | 0.869(3) | 0.807(4) | 0.752(5) | 0.698(4) | 0.670(4) | 0.646(4) |
| 2e-3 | parallel, terminal only | 0.983 | 0.970(1) | 0.963(1) | 0.956(2) | 0.945(2) | 0.934(2) | 0.923(2) | 0.917(2) | 0.912(2) |
| 2e-3 | `const0` | 1.000 | 0.993(1) | 0.988(1) | 0.981(2) | 0.970(2) | 0.960(2) | 0.949(2) | 0.943(2) | 0.937(2) |
| 5e-3 | serial | 0.893 | 0.645(11) | 0.476(13) | 0.350(7) | 0.189(7) | 0.105(4) | 0.057(2) | 0.042(2) | 0.030(2) |
| 5e-3 | parallel | 0.957 | 0.811(7) | 0.699(10) | 0.596(7) | 0.442(9) | 0.332(7) | 0.245(6) | 0.209(4) | 0.179(4) |
| 5e-3 | parallel, no reset | 0.957 | 0.852(5) | 0.767(8) | 0.688(5) | 0.560(8) | 0.461(7) | 0.378(7) | 0.338(5) | 0.305(4) |
| 5e-3 | `const0` | 1.000 | 0.983(2) | 0.970(2) | 0.955(3) | 0.929(5) | 0.901(5) | 0.877(5) | 0.863(5) | 0.850(5) |

LRB-D unif=1 acceptance A(m), data2 semantics:

| p | variant | m=0 | 2 | 4 | 6 | 10 | 14 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2e-3 | serial | 0.890 | 0.585 | 0.385 | 0.253 | 0.110 | 0.048 | 0.021 | 0.014 | 0.0089 |
| 2e-3 | parallel | 0.939 | 0.717 | 0.549 | 0.419 | 0.246 | 0.144 | 0.084 | 0.064 | 0.049 |
| 2e-3 | parallel, no reset | 0.939 | 0.770 | 0.633 | 0.520 | 0.351 | 0.238 | 0.161 | 0.132 | 0.108 |
| 5e-3 | serial | 0.750 | 0.265 | 0.094 | 0.033 | 4.2e-3 | 5.3e-4 | 6.6e-5 | 2.1e-5 | 8.2e-6 |
| 5e-3 | parallel | 0.854 | 0.436 | 0.224 | 0.115 | 0.030 | 7.9e-3 | 2.1e-3 | 1.1e-3 | 5.5e-4 |
| 5e-3 | parallel, no reset | 0.854 | 0.520 | 0.319 | 0.194 | 0.073 | 0.027 | 0.010 | 6.3e-3 | 3.8e-3 |

All A(m) s.e.m. are below 1% relative; see `tables.md`. The LRB-D accepted
F(m) for every variant, p and policy is in `fm_long.csv`.

## 3. Best hardware schedule

Within the repo's SI1000 model:

1. **For correction (decoded LRB): run no mid-circuit check rounds.**
   - End with a transversal X readout of all five data qubits and decode the
     two X parities exactly (`const0` + MLD): r = 0.74(3)p per step at every
     p tested.
   - Physical RB at 5p gives 1.69p (lead).
   - The best schedule with a check every round gives 4.6 to 5.1p (VN).
2. **For detection (LRB-D)**:
   - Measure all four ancillas in one layer, and skip the reset if the
     platform's readout errors are mostly physical flips.
   - Split a few sparse checks between the forward and inverse halves, about
     every 8 to 11 Cliffords in each. That gives about 0.45% at
     A(22) ≈ 0.20 to 0.25 at p = 5e-3 (verifier, exact). Forward-only
     placement every 3 to 11 Cliffords gives 0.54 to 0.63%.
   - Postselect on the final transversal readout too.
   - If shots are the bottleneck rather than accepted fidelity, use the
     terminal round only, or `const0`.
3. **If check rounds must run every round** (for example a QEC-cycle
   demonstration): use one parallel measurement layer, and drop the reset
   where the readout allows it. Each step reduces data idle.
   - The every-round check penalty (decoded), r(MLD5, every round) −
     r(`const0` MLD5), is 11.0p at first order for serial.
   - Parallel readout cuts it to 5.5p (50%), and dropping the reset to 3.7p
     (66 to 68% removed; 51% and 67 to 68% sampled).

**Relation to the user's literature-SI1000 choice.** In the literature-exact
Gidney et al. 2021 SI1000 model with ASAP moments, all ancillas are measured
in one layer, and the reset sits before each check round instead of after it.
Per round that is one measurement layer and one reset layer on the data, the
same as my `parallel` variant.

- A reset *before* round 1 adds one depth-independent reset layer. It moves
  only F(0), not f (VA).
- So the parallel numbers above are the right expectation for the
  every-round check penalty (decoded) under that model, *before* gate-layer
  idle noise: about half of the serial penalty removed (11.0p → 5.5p at first
  order).
- Gate-layer idle (Idle(p/10) per gate layer) is absent here. The lead's
  literature variant covers it.

Hardware notes (C):

- The 2p idle per measurement layer is the model's stand-in for slow readout.
  On platforms with fast or mid-circuit-isolated readout (ions, atoms) the
  serial-vs-parallel gap shrinks, but the ranking of check intervals should
  not change, because d=2 correction gains essentially nothing from checks.
- Measure-without-reset is optimistic here because SI1000 errors are physical
  flips. Classical assignment errors would flip two consecutive raw syndromes
  (2.5 and open questions).

## 4. Claims

| # | claim | label | evidence |
|---|---|---|---|
| 1 | `build_variant_circuit(schedule="serial")` reproduces the lead's `build_lrb_circuit` instruction for instruction (all forward subsets and inverse intervals tested) | VN | `test_serial_reproduces_lead_builder` (56 cases) |
| 2 | Data-qubit idle layers per round: serial 5, zx 3, parallel 2; without reset 4, 2, 1 | VA, VN | counting; `test_data_idle_layers_per_round` |
| 3 | Data2 first-order undetectable floor: serial 68p/15 (data 0 in M7 and M8, data 3 in M8, Z/Y data parts of DEPOLARIZE2 after CNOT 7→0 and 8→3); 48p/15 for zx and parallel with the repo's CNOT order (X-check CNOTs first: 54p/15, verifier) | VA, VN | derivation in 2.2; `test_parallel_two_qubit_readout_floor_is_smaller`; Stim `explain_detector_error_model_errors` |
| 4 | With a parallel measurement layer, measure-without-reset equals deleting the reset layer's noise (identical joint distribution); not so for the serial schedule | VA, VN | 1.2; exact brute-force joint distributions at depths 1 and 2 |
| 5 | Every variant keeps circuit distance 2 with the five-qubit readout and keeps all DEM mechanisms within two adjacent slots (exact transfer-matrix MLD applies) | VN | `test_circuit_distance_two_with_full_readout`; `test_noiseless_variants_are_deterministic`; MLD equals brute force |
| 6 | First-order MLD error per step: serial 11.76(19)p, zx 8.08(13)p, parallel 6.28(10)p, parallel no reset 4.45(7)p, no checks 0.73(3)p; sampled values at 5e-4 to 5e-3 lie 0 to 2.5p above (second-order terms), rising with p | VN | `first_order_tables.md`; 2.2 |
| 7 | Every-round check penalty (decoded) = r(MLD5, every round) − r(`const0` MLD5): 11.0p at first order (serial). One parallel measurement layer leaves 5.5p (removes 49.8(3)%, 51.0(2)%, 51.0(2)% sampled at p = 5e-4, 2e-3, 5e-3); also dropping the reset leaves 3.7p (66.3(2)%, 67.2(2)%, 67.7(2)%). MLD2 fractions use the same `const0` MLD5 reference | VN | paired bootstrap, `summary.json` `penalty_removed`; first order 49.7% and 66.3%; verifier: 49.6% and 66.2% on new sequences |
| 8 | LRB-D unif=1 rejection per step falls from 105p (serial) to 67p (parallel) and 49p (no reset); acceptance at m = 22 grows up to 466 times at 5e-3 | VN | exact R1; sampled α and A(22) in 2.2 |
| 9 | Data idle during the serial measurement layers is the dominant term: 7.31(14)p of 11.76p first-order decoded error. Measurement flips, reset flips, measured-qubit depolarizing, ancilla idle and ancilla H contribute 0 at first order | VN | exact leave-one-out ablations, 2.5 |
| 10 | Decoded LRB gets monotonically worse with more executed checks at all tested p (sampled 5e-4 to 5e-3; first order; every round against none up to 5e-2 in the lead's grid) and schedules; no checks (`const0`) is optimal (0.74(3)p). const rows are not nested with unif rows | VN | 2.3 sampled table and exact first-order table; verifier: 2,640 random single-check additions all raise E1 |
| 11 | Noiseless checks every round improve first-order decoded error by only 0.010(1)p over no checks (0.011(2)p at finite p, verifier). At first order, checks must be about 500 to 1700 times quieter than SI1000 to break even (±25% from sequence sampling) | VN (first order) | check-noise scan 2.6, paired per sequence; verifier: linear in C, 466 to 1341× on new sequences |
| 12 | Among forward-half placements, sparse checks every 3 to 11 Cliffords (parallel, no reset) give the lowest LRB-D accepted infidelity: 0.57(3)% to 0.63(3)% at p = 5e-3 and m = 22, against 0.88(6)% every round and 1.89(14)% with no checks. Forward+inverse placements do better: about 0.45 to 0.46%, e.g. forward unif=8 + inverse every 8 gives 0.464% at A = 0.203 (verifier, exact). The order inside the plateau depends on alignment with m | VN | 2.3; exact A2 slopes in `first_order_tables.md`; verifier's exact LRB-D |
| 13 | Executing only a policy's rounds strictly improves LRB-D accepted fidelity and acceptance over the repo protocol (all rounds run): `const=1` at 5e-3 goes from F(22) 0.024(2), A(22) 0.063 to F(22) 0.931(1), A(22) 0.462. The repo-protocol f is fit-dependent (0.901 with sampled weights, 0.886 with exact data) | VN | 2.3 last table; `tables.md`; verifier's exact values agree within 0.004 |
| 14 | Inverse-half placement is neutral for decoding (first-order Δ within ±0.25p at nearly equal round count; under 1% of E1 at exactly equal count, verifier) but lowers LRB-D accepted infidelity by 15 to 28% at equal round count and unchanged acceptance (verifier, exact, 10/10 sequences; sampled here 9 to 23%). Added on top of forward checks, inverse checks cost like any round | VN | 2.4; `first_order_tables.md`; verifier's paired exact evaluation |
| 15 | Sampled serial and `const0` counts agree with the lead's `grid_main` within binomial noise | VN | `crosscheck_grid_main.csv` (2.1) |
| 16 | The `parallel` variant is the closest analogue of literature SI1000 with ASAP scheduling and resets before rounds, up to gate-layer idle noise and a depth-independent first reset | VA | 3 |
| 17 | The ranking of check intervals carries over to platforms with faster readout | C | not simulated |
| 18 | Measure-without-reset gains are optimistic if readout errors are mostly classical assignment errors | C | 1.2 argument; not simulated |

## Revision log

- 2026-10-28, after the adversarial verification
  (`notes/H-schedules-ablation.verify.md`). The verifier reproduced every
  number to all printed digits (first order) or within errors (sampled), found
  no code bug, and refuted two claims as stated.
  - Fix 1, claim 3: the data2 floor of 48p/15 holds for zx and parallel with
    the repo's CNOT order. With the X-check CNOTs first it is 54p/15. What
    counts is the number of measurement layers after the X-check CNOTs on
    data 0 and 3.
  - Fix 2, claim 7 (verifier claim 6), Summary item 3, sections 2.2 and 3:
    "d=2 correction penalty" is renamed "every-round check penalty
    (decoded)". It is defined as r(MLD5, every round checked) − r(`const0`
    MLD5), with absolute sizes 11.0p (serial), 5.5p left (parallel) and 3.7p
    left (no reset) at first order. The MLD2 fractions use the same `const0`
    MLD5 reference.
  - Fix 3, claim 10 (verifier 9): "at all p" becomes "at all tested p", with
    the scope stated. The const rows are not comparable per step with the
    unif rows.
  - Fix 4, claim 11 (verifier 10): the break-even factor is first order,
    about 500 to 1700×, ±25% from sequence sampling. The finite-p
    perfect-check gain of 0.011(2)p is added.
  - Fix 5, claim 12 (verifier 11): "lowest" is restricted to forward-half
    placements (plateau every 3 to 11 Cliffords). Forward+inverse placements
    do better, about 0.45 to 0.46%; for example unif=8 + inverse every 8
    gives 0.464% at A = 0.203. The Summary recipe and section 3 item 2 are
    updated.
  - Fix 6, claim 13 (verifier 12): the repo-protocol vs executed table now
    leads with F(22) and A(22). Its f is marked fit-dependent (0.886 with
    exact data against 0.901 sampled-weight fit).
  - Fix 7, claim 14 (verifier 13): the LRB-D conclusion on inverse-half checks
    is reversed. Splitting the checks between the halves lowers the accepted
    infidelity by 15 to 28% at unchanged acceptance, while the decoded error
    changes by less than 1%. Section 2.4's "same round count" is corrected:
    unif=2 has 11 forward rounds against 10 for k = 4 at m = 22. My own
    sampled data at equal round count (3 + terminal) agree in sign and size.
    Section 2.4 and section 3 item 2 are rewritten.

## 5. Bugs and issues found

- **No bug in the lead's model code.** My serial builder is byte-identical,
  the DEM-level first-order numbers match his to all digits, and sampled
  statistics match `grid_main`.
- **Minor doc inaccuracy** in `API.md`: it says the data2 floor comes from
  idle on "data 0, 2 or 3". Z/Y errors on data 2 commute with the X0 X3
  readout and never contribute. The mechanisms are data 0 (two layers),
  data 3 (one layer) and the CNOT 7→0 and 8→3 faults (claim 3). The lead's
  L0 note has it right.
- **Fit caveat (not a bug).** In the repo's fixed-offset fit with
  std-over-sequence weights, a depth whose std is 0 gets the floor weight.
  Depth 0 is sequence-independent, so this happens at m = 0. For curves that
  are not exponential (LRB-D without mid-circuit checks, where accepted
  errors grow like m²) the fit is pinned at F(0) and reports a too-small
  decay, for example f = 0.99976(6) for `const0` LRB-D at 5e-3 while F falls
  to 0.981 by m = 22. Report F(22) next to f for LRB-D.
- **Infrastructure (important for every agent).** On the shared allocation,
  `srun --overlap -c N` gives each step a CPU set that starts at CPU 0, and
  the steps of all agents overlap. I saw about 76 processes from three
  agents stacked on 23 logical CPUs (CPU 136 had 14), so my tasks ran 6 times
  slower. `--cpu-bind=none` does not help, because the cpuset is fixed per
  step. What worked: `srun ... -c 256` (still `--overlap`, so it reserves
  nothing), then `taskset -c <free range>`, and one CPU per pool worker
  (`_pin_worker` in `schedule_variants.py`). Unpinned workers inside a
  20-CPU taskset also left 5 CPUs idle.

## 6. Open questions

- Gate-layer idle noise and ASAP moments (literature SI1000), in the lead's
  separate variant. The parallel schedule makes Z ancillas wait during the
  X-check CNOTs; with Idle(p/10) per gate layer that adds about 0.1p per
  ancilla per layer, small next to the 2p measurement-layer idle removed
  (C).
- Measure-without-reset with *classical* readout errors produces mechanisms
  spanning three slots, which the transfer-matrix MLD cannot handle. LRB-D
  on executed policies is unaffected, because both errors reject the shot
  (VA). Decoding them needs a two-round slot grouping (2^17-entry kernels)
  or a different decoder.
- How far beyond m = 22 does the advantage of sparse checks over no checks
  grow for LRB-D, given A2 ∝ m² without checks? This is relevant to WP F
  (non-Markovianity).
- For d = 3 codes single faults become correctable, so correction should gain
  from checks; the break-even interval there is open.
- Statistics: 10 sequences. Bootstrap errors from 10 sequences are uncertain
  by about 25% relative, and the first-order slopes have s.e.m. of 1 to 4%.

## Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest -q tests/test_qubit512_variants.py          # 154 tests, ~30 s
OUT=studies/correction-decoders/qubit/results/H
# exact first-order coefficients (no sampling; a few minutes on 4 cores)
python studies/correction-decoders/qubit/scripts/schedule_variants.py first-order --out $OUT --num-seq 10 --workers 4
python studies/correction-decoders/qubit/scripts/schedule_variants.py first-order --groups checkscan --tag _checkscan --out $OUT --num-seq 10 --workers 4
python studies/correction-decoders/qubit/scripts/schedule_variants.py first-order-tables --out $OUT
# sampling: resumable, about 1 core-hour per 100 tasks; 1,860 tasks in total
srun --jobid=<JOBID> --overlap -N 1 -n 1 -c 256 bash -c "module load python/3.11; cd $PWD; \
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src; \
  taskset -c 64-87 python studies/correction-decoders/qubit/scripts/schedule_variants.py run --out $OUT --num-seq 10 --workers 24 --max-minutes 17"
# repeat the srun line until the run reports 0 tasks to run, then:
python studies/correction-decoders/qubit/scripts/schedule_variants.py analyze --out $OUT --num-seq 10 --workers 20
python studies/correction-decoders/qubit/scripts/schedule_variants.py crosscheck --out $OUT --grid studies/correction-decoders/qubit/results/grid_main
```

The run logs are in `results/H/logs/`.
