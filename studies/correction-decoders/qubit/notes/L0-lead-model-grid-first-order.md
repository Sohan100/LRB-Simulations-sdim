# L0: model, main grid, exact first-order structure (lead session)

All numbers come from code in this worktree; commands are at the end. Labels:
**VA** verified analytically, **VN** verified numerically, **LIT** from
literature, **C** conjecture.

## 1. Model validation (VN)

`tests/test_lrb_qubit_model.py`: 40 tests pass.

- The 24 descriptors form the single-qubit Clifford group.
- Every inverse sequence inverts its Clifford.
- The encoder prepares |+>_L.
- Logical H, S, X and Z preserve the stabilizer group with + signs, and act as
  H, S, X and Z on (X_L, Z_L).
- Noiseless circuits are deterministic at every depth.
- Raw per-round syndromes reconstructed from difference detectors equal the
  sampled ancilla records bit for bit.
- `used_rounds` reproduces the repo's `postselect_constant_number` and
  `postselect_uniform_interval` on random records.
- The transfer-matrix MLD equals brute-force exact inference
  (|ΔLLR| < 1e-9; 4e-15 observed).
- With the five-qubit terminal readout the circuit-level distance is 2.

## 2. The repo's two-qubit terminal readout has a first-order undetectable floor (VN, VA)

With `data2`, the readout measures data 0 and 3 only, as in the qutrit
pipeline's ordinary readout. In the terminal check round, the X ancillas are
measured one at a time. During each measurement layer every other wire gets
DEPOLARIZE1(2p). A Z (or Y) error on data 0 or 3 that lands after that
qubit's last X-check coupling is seen by no detector and flips x0 + x3; two
CNOT fault locations do the same (corrected after work package H listed the
faults).

The exact DEM gives a total undetectable logical probability of 4.53p, the
same at every depth and for every sequence. The brainstorm derives it
analytically as 68p/15. It bounds the accepted fidelity of every acceptance
rule from above by 1 − 2 × 4.53p.

Reading all five data qubits in the X basis at the end removes this floor. The
extra readout is ideal and comes after everything else, so it does not change
the physical noise. The `*_final` LRB-D variants and `mld5` use it.

## 3. Main grid (VN)

Settings: 30 sequences (seed 20261028), depths 0,2,4,6,10,14,18,20,22, and
the 15 probabilities of the qutrit run. LRB-D uses 1e6 shots per circuit;
decoding uses 3e5 shots per circuit, on the same shots. Outputs are
`results/grid_main/` and `results/summary/{fits,curves}.csv`.

Error per depth step, (1 − f)/2, divided by p. The fit is the repo's LRB fit
with the offset fixed at 0. Physical RB is shown both with the repo's
free-offset fit and with the offset fixed at 0.

| method | p=1e-4 | 1e-3 | 2e-3 | 5e-3 | 1e-2 |
|---|---|---|---|---|---|
| LRB-D unif=1 (data2) | 0.0045 | 0.056 | 0.13 | 0.40 | 1.16 |
| LRB-D unif=1 + final data check | 0.0045 | 0.050 | 0.11 | 0.32 | 0.85 |
| LRB-D const=1 | 0.13 | 0.96 | 2.2 | 9.6 | 11 |
| full exact MLD (mld5) | 11.9 | 12.5 | 13.1 | 14.2 | 14.9 |
| MLD, accept \|LLR\| ≥ 4 | 0.092 | 0.23 | 0.27 | 0.37 | – |
| const0 (no checks, ideal X readout, postselect) | 0.00075 | 0.0061 | 0.011 | 0.029 | 0.058 |
| const0, decoded (no rejection) | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 |
| physical RB at 5p, offset 0 | 1.70 | 1.70 | 1.69 | 1.69 | 1.67 |
| physical RB at 5p, repo free-offset fit | 20.2 | 3.7 | 2.8 | 2.1 | 1.9 |

How to read the table:

- Full correction is first order, at about 12–15p per step. That is 7 times
  worse than physical RB at 5p, and about 35 times worse than physical RB at the
  same p, where the error per step is about 0.34p.
- LRB-D with every check is second order (≈48p²), at the price of an
  acceptance that decays as about (1 − 105p)^m.
- The repo-style free-offset RB fit is badly biased at low p. F(m) is nearly
  flat there, so the offset and the decay trade off against each other. Fix the
  offset at 0, since |<Z>| decays to 0.

## 4. Exact first- and second-order structure of the DEM (VA + VN)

Use the data5 circuit with all rounds, and group the single DEM mechanisms by
symptom s. Let P0(s) be the total weight in class s that leaves the logical
unchanged, and P1(s) the weight that flips it. Then:

- R1 = Σ_s (P0 + P1): first-order rejection of LRB-D with all checks and the
  final data check.
- E1 = Σ_s min(P0, P1): first-order error of exact MLD.
- A2 = Σ_s P0·P1: second-order accepted error of that LRB-D. Two faults with
  the same symptom cancel it and differ in their logical flip.

`studies/correction-decoders/qubit/scripts/first_order.py` computes these over 30 sequences at
p = 1e-6, where the DEM is linear:

| depth | R1/p | E1/p | A2/p² |
|---|---|---|---|
| 0 | 69.5 | 11.1 | 37.5 |
| 6 | 700 | 84.3 | 336 |
| 22 | 2381 | 270 | 1096 |

The slopes are about 105p rejection, 11.8p MLD error and 48p² accepted LRB-D
error per depth step. These agree with the sampled grid: 12p for mld5 at low p
in §3, and the LRB-D unif=1 + final values.

**Component breakdown at depth 22.** Rebuild the DEM with one SI1000 term on at
a time. `figures/main/fig4_first_order_by_component.png` shows the result.

| SI1000 term | R1/p | E1/p |
|---|---|---|
| data idle during the 4 sequential ancilla-measurement layers (2p) | 1093 | 168 |
| ancilla measurement flip (5p) | 460 | **0** |
| two-qubit gate depolarizing (p) | 329 | 47 |
| data idle during reset layer (2p) | 220 | 40 |
| reset flip (2p) | 176 | **0** |
| measured-ancilla depolarizing (p) | 61 | **0** |
| one-qubit gate depolarizing (0.1p) | 42 | 5 |

The single faults that cause 29% of LRB-D's rejections are measurement,
reset and ancilla faults. Exact MLD corrects every one of them at first order.

**How much rejection is free to remove (depth 22, mean of 30 sequences).**

- 66.6% of the first-order rejected weight is in classes with min(P0, P1) = 0.
  Correcting those costs nothing at first order.
- 24.5% is in genuinely ambiguous classes, where the two cosets are within a
  factor of 10.
- Rejecting only the ambiguous classes keeps 75.5% of the weight LRB-D
  rejects, and leaves a first-order error of 7.7p instead of 270p.

The first-order Pareto frontier for sequence 0 at depth 22 is ordered by each
class's conditional error. Rejecting the worst 30% of R1 brings E1 from 283p
to 1.7p, and rejecting the worst 40% brings it to 0.

## 5. Noisy terminal readout (VN)

`results/terminal_noise/`, 30 sequences, 1e5 shots. Here the final data
measurement also gets SI1000 measurement noise. The fitted decay rates do not
change, and only the depth-0 intercept moves, as RB's SPAM robustness
predicts. At p = 1e-3:

| method | f | F(0) |
|---|---|---|
| mld5, noisy readout | 0.97500 | 0.956 |
| no-check decoded (`const0_mld`), ideal readout | 0.9985 | – |
| no-check decoded (`const0_mld`), noisy readout | 0.99787 | 0.977 |
| LRB-D unif=1 (data2), noisy readout | 0.99991 | 0.968 |
| LRB-D unif=1 + final noisy data check | 0.99990 | 0.9997 |

## 6. Local decay rates from the main grid (VN, a first look at non-Markovianity)

Local rates are (F(m2)/F(m1))^{1/(m2−m1)} of the sequence-averaged F.
`figures/main/fig3_local_decay_rates.png` shows them at p = 2e-3.

Constant within noise:

- LRB-D unif=1, with and without the final check;
- mld5;
- const0_mld;
- physical RB.

Strongly drifting:

- LRB-D const=1: 0.9995 → 0.9905 at p = 1e-3, and 0.98 → 0.78 at p = 5e-3.
  Its acceptance per step drifts too, from 0.934 to 0.950 at p = 1e-3.
- LRB-D const=4 and unif=4 have a sawtooth, from the check positions.
- MLD + a fixed |LLR| ≥ 4 threshold drifts mildly. Its acceptance per step
  falls from 0.890 to 0.853 at p = 2e-3, and its local error falls with depth.

Work package F tests all of this properly.

## Reproduce

```
module load python/3.11
python -m pytest -q tests/test_lrb_qubit_model.py
python studies/correction-decoders/qubit/scripts/run_grid.py --out studies/correction-decoders/qubit/results/grid_main \
    --shots 1000000 --decode-shots 300000 --workers 120 --rb-shots 1000000   # ~10 min on one CPU node
python studies/correction-decoders/qubit/scripts/summarize_grid.py --grid studies/correction-decoders/qubit/results/grid_main \
    --out studies/correction-decoders/qubit/results/summary
python studies/correction-decoders/qubit/scripts/first_order.py --num-seq 30 --out studies/correction-decoders/qubit/results/first_order
python studies/correction-decoders/qubit/scripts/terminal_noise.py --out studies/correction-decoders/qubit/results/terminal_noise --workers 64
python studies/correction-decoders/qubit/scripts/plot_main.py
```

## 7. LRB-D points against the exact-MLD |LLR| frontier (VN)

Data: the main-grid gap histograms of `mld5`, pooled over the 30 sequences.
Accept shots in decreasing order of |LLR| until the acceptance equals that of
an LRB-D policy, with a fractional final bin, and compare the logical error
rates. Numbers are logical error among accepted shots (not F):

| p | m | policy | acceptance | LRB-D error | frontier error | ratio |
|---|---|---|---|---|---|---|
| 1e-3 | 6 | unif=1 + final | 0.497 | 3.5e-4 | 3.4e-4 | 1.03 |
| 1e-3 | 22 | unif=1 + final | 0.093 | 1.11e-3 | 1.10e-3 | 1.00 |
| 1e-3 | 22 | unif=1 (data2) | 0.094 | 5.8e-3 | 1.1e-3 | 5.2 |
| 1e-3 | 22 | const=1 | 0.258 | 4.95e-2 | 3.2e-3 | 15 |
| 1e-3 | 22 | unif=4 | 0.187 | 9.4e-3 | 2.0e-3 | 4.7 |
| 2e-3 | 14 | unif=1 + final | 0.046 | 3.10e-3 | 3.09e-3 | 1.00 |
| 2e-3 | 22 | const=1 | 0.105 | 0.22 | 2.2e-2 | 10 |
| 5e-3 | 6 | unif=1 + final | 0.031 | 1.04e-2 | 1.07e-2 | 0.97 |
| 5e-3 | 14 | const=1 | 0.069 | 0.42 | 0.16 | 2.6 |

- LRB-D with every check plus the final data check lies on the optimal
  frontier, with ratio 0.97–1.03 everywhere it has statistics. It is the most
  conservative point of the soft-output family.
- Every sparser LRB-D policy is strictly dominated, by 3–15× in accepted
  error at equal acceptance.
- The data-2 policies are dominated partly because of the 4.53p floor. The
  fair comparison for them is the `mld2` frontier (work package E).

## 8. Real-time abort of LRB-D (VN, exact)

Here a shot is stopped at its first check round with a non-zero syndrome,
which needs only a trivial real-time rule. "Round k executes" means that
rounds 0..k−1 were all clean, which the exact sweep gives as
`exact_condition(decoder, range(k))`. The table is the mean over 30 sequences
of the fraction of check rounds actually executed, using every-check LRB-D
(unif=1).

| p | m = 10 | m = 22 | acceptance at m = 22 |
|---|---|---|---|
| 1e-3 | 0.654 | 0.419 | 0.094 |
| 2e-3 | 0.470 | 0.249 | 0.0089 |
| 5e-3 | 0.259 | 0.125 | 8.2e-6 |

On hardware where check rounds dominate the wall-clock time per depth step, as
on superconducting qubits, early abort saves 35–88% of QPU time. It requires
the control stack to end a shot early. Work package J found that Quantinuum
Helios documents this, and that IBM and IQM document no such feature.
