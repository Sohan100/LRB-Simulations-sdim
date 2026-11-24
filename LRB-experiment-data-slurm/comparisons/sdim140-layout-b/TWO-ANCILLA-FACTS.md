# The two-ancilla layout: facts for replacing the conventional-site two-ancilla version in the paper

Naming (user, 2026-11-18): call it "the two-ancilla layout", not "layout B". Internal identifiers
keep `layout_b` / `layout-b` (code names, run folders, script and folder names), because renaming
them would break the run data and checkpoints.

User decisions, 2026-11-17:
- "use layout B".
- "We will just replace the 2 ancilla version with this".
- The terminal-postselection layout has exactly IQM Garnet's connectivity ("no more no less").

Code: `src/lrb/code_definitions.py` and `src/lrb/code_simulation_profiles.py` (written in the worktree `a1-square-layout`, merged into the main code on 2026-11-24).
Profiles: `folded_qutrit_layout_b`, `folded_qubit_layout_b` (internal names).

## Layout and circuits

**Lattice.** Square lattice, x to the right and y up:

```
q4(0,2)  N(1,2)
q1(0,1)  q2(1,1)  E(2,1)
q0(0,0)  q3(1,0)
```

**Couplers.** The 8 couplers are exactly the lattice-adjacent pairs: q0-q1, q0-q3, q1-q2, q2-q3,
q1-q4, N-q4, N-q2, E-q2.
- q2-q3 carries SWAP_{2,3} in H-bar and CZ_{2->3} in S-bar, and is also used by the checks.
- q0-q3 is used only by the encoder, because X0 X3 is a logical operator.

**Checks.**
- N measures G2 and then G4; E measures G1 and then G3.
- Neither ancilla neighbours three data qudits of a stabilizer. Each check therefore moves parity
  through data SUM gates on q0-q1, q1-q2, q1-q4 and q2-q3, and undoes them before the readout.
- Per round: 24 two-qudit and 4 one-qudit gates; 19 gate moments plus 1 reset and 2 readout
  moments (ASAP scheduler).
- Ancillas are reset once per round, at its start, and read twice (N, E, N, E). A shot passes only
  if every readout is 0.
- For comparison: one-ancilla round 12 two-qutrit gates and 7 gate moments; conventional-site
  round 18 and 11.

**Single faults.** None is silent: 0 of 2056 fault locations (qutrit) and 0 of 411 (qubit), with
and without an H-bar sandwich.

**Encoder (two-ancilla layout).** H on q2, q3, q4; then CNOT 3->0, CNOT 4->1, CNOT_INV 2->1, CNOT 1->0.
- 4 two-qudit gates, which is the minimum for any connectivity.
- It is the paper's encoder with CNOT 4->2 replaced by CNOT 4->1, and it gives exactly the same
  |+_L>.
- Preparation is ideal in all simulations, as before, so the encoder has no effect on the results.

**Encoders of the other setups.**
- **Garnet terminal postselection.** q0..q4 on QB1, QB4, QB5, QB2, QB6. H on q0, q2, q3; then
  [CNOT q0->q1, CNOT q2->q4], then [CNOT q3->q0, CNOT_INV q1->q2]: 4 gates in 2 layers. The q2-q3
  logical gates use QB5-QB2.
- **One-ancilla layout.** 8 CNOTs through a1 and a2, depth 5; this is the CNOT-network optimum.
  Details are in `comparisons/two-ancilla-checks/hw-layouts/README.md`, section "Optimal encoders
  and the final layout".

## Runs (same production grids as every sdim 1.4.0 campaign)

| run | role |
|---|---|
| `Run-2026-11-17-14-38-05-folded_qutrit_layout_b-layout-b-si1000-sdim140` | qutrit SI1000 production |
| `Run-2026-11-17-14-39-18-folded_qutrit_layout_b-layout-b-sd6-sdim140` | qutrit SD6 production |
| `Run-2026-11-17-14-40-29-folded_qubit_layout_b-layout-b-si1000-sdim140` | qubit SI1000 production |
| `Run-2026-11-17-14-41-29-folded_qubit_layout_b-layout-b-sd6-sdim140` | qubit SD6 production |
| `Run-2026-11-17-15-25-58-...-si1000-sdim140-highp-ext` | qutrit SI1000 extension, p 0.0125-0.03 |
| `Run-2026-11-17-15-12-56-...-sd6-sdim140-highp-ext` | qutrit SD6 extension, p 0.0125-0.03 |
| `Run-2026-11-17-15-26-16-...-si1000-sdim140-threshold-refine` | qubit SI1000 refinement, p 0.001-0.007, step 0.00025 |
| `Run-2026-11-17-15-42-03-...-sd6-sdim140-threshold-refine` | qubit SD6 refinement, p 0.0015-0.009, step 0.00025 |

**Production runs.** 30 Cliffords, depths 0..22 with m = 0 fitted, 1M shots, and the 15
p values 0 to 0.05.

**Extensions.** Depths 0,1,2,3,4,6 and 20M shots. Only even depths are fitted for k >= 2. They
take over for p >= p_split = 0.0125.
- This p_split is lower than the other layouts' (SD6 0.025, SI1000 0.015) because the two-ancilla
  layout's crossings sit lower.
- Every two-ancilla qutrit crossing for k >= 3 lies below p_split and comes from production rows,
  as in the paper.

**Qubit refinements.** Production settings on a fine p grid inside their own window, with the
scale-free crossing rule, as for the other qubit runs.

All runs are complete and validated (`validation.txt` in each folder).

## Pseudo-thresholds (final; p* +- one SE from the sequence bootstrap)

Hand-off files: `comparisons/pseudo-thresholds-layout-b-current.{md,csv}` and
`comparisons/pseudo-thresholds-qubit-layout-b-current.{md,csv}`.

| code | noise | k | one ancilla | previous: conventional sites (current paper) | two ancillas | two / one ancilla |
|---|---|---|---|---|---|---|
| qutrit | SI1000 | 1 | 0.0292 +- 0.0006 | 0.0266 +- 0.0004 | 0.0191 +- 0.0004 | 0.66 |
| qutrit | SI1000 | 2 | 0.0239 +- 0.0005 | 0.0209 +- 0.0003 | 0.0139 +- 0.0002 | 0.58 |
| qutrit | SI1000 | 3 | 0.0166 +- 0.0004 | 0.0136 +- 0.0003 | 0.0085 +- 0.0002 | 0.51 |
| qutrit | SI1000 | 4 | 0.0132 +- 0.0002 | 0.0108 +- 0.0001 | 0.0064 +- 0.0001 | 0.49 |
| qutrit | SD6 | 1 | 0.0399 +- 0.0015 | 0.0283 +- 0.0007 | 0.0206 +- 0.0004 | 0.52 |
| qutrit | SD6 | 2 | 0.0269 +- 0.0007 | 0.0212 +- 0.0003 | 0.0147 +- 0.0003 | 0.55 |
| qutrit | SD6 | 3 | 0.0181 +- 0.0006 | 0.0122 +- 0.0003 | 0.0088 +- 0.0002 | 0.49 |
| qutrit | SD6 | 4 | 0.0155 +- 0.0004 | 0.0103 +- 0.0001 | 0.0074 +- 0.0002 | 0.48 |
| qubit | SI1000 | 1 | 0.0112 +- 0.0002 | 0.0093 +- 0.0001 | 0.0051 +- 0.0002 | 0.46 |
| qubit | SI1000 | 2 | 0.0078 +- 0.0002 | 0.0064 +- 0.0001 | 0.0031 +- 0.0001 | 0.40 |
| qubit | SI1000 | 3 | 0.0049 +- 0.0001 | 0.0040 +- 0.0001 | 0.0020 +- 0.0000 | 0.40 |
| qubit | SI1000 | 4 | 0.0040 +- 0.0001 | 0.0031 +- 0.0001 | 0.0015 +- 0.0000 | 0.38 |
| qubit | SD6 | 1 | 0.0210 +- 0.0008 | 0.0142 +- 0.0004 | 0.0085 +- 0.0002 | 0.41 |
| qubit | SD6 | 2 | 0.0151 +- 0.0005 | 0.0100 +- 0.0003 | 0.0058 +- 0.0001 | 0.38 |
| qubit | SD6 | 3 | 0.0094 +- 0.0003 | 0.0061 +- 0.0002 | 0.0035 +- 0.0001 | 0.37 |
| qubit | SD6 | 4 | 0.0072 +- 0.0002 | 0.0047 +- 0.0001 | 0.0027 +- 0.0000 | 0.37 |

**Significance.**
- One ancilla vs two ancillas differ by 12.7 to 30.9 sigma, with the SEs added in quadrature.
- Adjacent-k gaps within the two-ancilla layout are 8.8 to 27 sigma (paired replicates; the ordering holds in
  all 1000 replicates). See `comparisons/sequence-bootstrap/interval_gap_significance-layout-b.csv`.

**Qubit / qutrit p* for the two-ancilla layout** (`comparisons/qubit-vs-qutrit-layout-b/`): SI1000 0.23-0.27,
SD6 0.37-0.41. The one-ancilla values are 0.30-0.38 and 0.47-0.56.

## Figures and tables

- **Per-run fit plots and threshold grids.** In each run's `results/plots/`, made by
  `notebooks/07-two-ancilla-layout-si1000-sd6.ipynb` in the worktree (internal file name). These are the
  counterparts of the paper's `figures/sdim140/*-conv-*.pdf`. The merged threshold grids are in
  `comparisons/sdim140-layout-b-{sd6,si1000}-highp-merged/` and
  `comparisons/sdim140-qubit-layout-b-*-refine-merged/`.
- **Layout comparisons.** `comparisons/sdim140-layout-b/pseudo-thresholds-by-layout.{md,csv,pdf}`
  and `lrb-vs-rb-unif-1-{qutrit,qubit}-layouts.{pdf,csv}`.
- **Layout and circuit figures.** TikZ, verified, in `paper-rework/a1-layout-figure/`:
  `ancilla-layouts-B.pdf` with (a) on IQM Garnet, (b) one-ancilla, (c) two-ancilla layout; `B-layout.pdf`;
  `B-checks.pdf`; captions; `HANDOFF-B.md` with the sentences that change.
- **Data figures.** The paper's pgfplots data figures (p* vs k, k = 1 error rates, qubit vs
  qutrit) need regenerating with the two-ancilla CSVs. Under the user's figure rule, that is done in
  a figure session.
