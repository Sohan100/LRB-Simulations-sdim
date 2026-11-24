# Two-ancilla checks on the conventional sites: ground truth

**Paper decision (user, 2026-11-02): "remove any coherent relay results we just want the
conventional sites".**
- The paper's only two-ancilla layout is the conventional sites, compared with one ancilla only.
- The coherent-relay columns, ratios and sigma gaps below are run-side context and must not
  appear in the paper.

Every statement about the new two-ancilla layout should follow from this file. Scripts and
outputs are in this folder unless stated; code is in the worktree
`/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/conventional-sites`
(merged into the main code since): `src/lrb/code_definitions.py`
(`FoldedQutritConventionalSitesDetectionCode`, `FoldedQubitConventionalSitesDetectionCode`),
`src/lrb/code_simulation_profiles.py` (profiles `folded_qutrit_conv_sites`,
`folded_qubit_conv_sites`), `tests/test_conventional_sites_code.py`.

## Layout (user's intent, 2026-11-01)

The data sit as in the paper's patch, q2 at the centre: q0 top left, q1 top right, q3 bottom
left, q4 bottom right. Ancillas sit at the four conventional sites between the data:

| site | wire | data it touches |
|---|---|---|
| N | 5 | q0, q1, q2 |
| W | 6 | q0, q2, q3 |
| E | 7 | q1, q2, q4 |
| S | 8 | q2, q3, q4 |

Only site-data gates between neighbours; no two sites are coupled; no long-range gates.
Stabilizers: G1 = Z0 Z1^-1 Z3^-1, G2 = Z1 Z2 Z4^-1, G3 = X0 X1 X2^-1, G4 = X1^-1 X3 X4^-1.
G3 fits N, G2 fits E. G1 and G4 both contain the opposite corners q1 and q3, so no site
covers them, and any generating set has the same problem (G1 G2 and G3 G4 have weight 4).
Four sites for four stabilizers means N and E are used twice per round.

## What does not work: separate readout of the halves

Splitting G1 or G4 over two sites, reading each site, and adding the readouts does not
measure the stabilizer. Reading a site measures the operator it collected on the data. W's
half of G1 is just z3 and S's half of G4 is x3, so this measures q3 in the Z and then the X
basis; each randomizes the other. Zero noise, 3000 shots:
- the user's schedule (G2 and G3, then G4, then G1): 101/3000 shots have all checks 0 after
  2 rounds, 0/3000 after 5 (`order_g4_then_g1.py`);
- G1 and G4 are random in every round after the first; the logical survives only as
  X0 X1 X4, and a single Z4 error after round 2 is never detected (`backaction_check.py`).
Reset does not help: the disturbance comes from the coupling, reading or not.

## What works: data relay, every site read once (the simulated protocol)

The partial value is lent to a data qutrit next to both sites and taken back, so the reader
site holds the whole stabilizer before anything is read.
- G1: N collects -z1 and lends it to q0 (q0 -> z0 - z1); W reads q0 and q3 (W reads G1);
  N takes the value back from q0 and clears itself against q1.
- G4: the same in the X basis: E lends its q1 part to q4, S reads q4 and q3 (S reads G4),
  E takes it back.
- N measures G3 before lending and keeps that value while lending; E lends before measuring
  G2. Each site is read once per round.
- Readouts, with the syndrome s(G) defined by G|psi> = w^s |psi> and sdim's X|j> = |j+1>,
  Z|j> = w^j |j>: N = s(G3), W = s(G1) + s(G3), E = s(G2), S = -s(G4), so S reads the
  syndrome of G4^-1.
  - These were checked for every single-qutrit X and Z error, noiselessly.
  - The sign is a convention; in the paper, write "each up to its sign convention", not
    sign-exact equalities.
- All four readouts must vanish, which is equivalent to all four stabilizers vanishing.
- The class docstring in `code_definitions.py` still says `m8 = G4`. It is left unedited
  because that file is part of the checkpoint source fingerprint.

One round (moments of the production ASAP scheduler; preceded by one reset moment of all
four sites, followed by one measurement moment of all four):

| moment | gates |
|---|---|
| 1 | H E, H S, H N |
| 2 | CNOT E->q1, CNOT_INV S->q3, CNOT N->q0 |
| 3 | CNOT q4->E, CNOT_INV q3->W, CNOT_INV N->q2 |
| 4 | CNOT S->q4, CNOT N->q1 |
| 5 | H_INV S, CNOT_INV q4->E, H_INV N |
| 6 | CNOT_INV E->q1 |
| 7 | CNOT_INV q1->N, H_INV E |
| 8 | CNOT N->q0, CNOT q2->E |
| 9 | CNOT_INV q4->E, CNOT q0->W |
| 10 | CNOT_INV N->q0, CNOT q1->E |
| 11 | CNOT q1->N |

Couplers used: N-q0, N-q1, N-q2, W-q0, W-q3, E-q1, E-q2, E-q4, S-q3, S-q4, i.e. 10 of the 12
site-data couplers of the conventional patch (W and S never touch q2). No ancilla-ancilla gate,
no long-range gate.

18 two-qudit gates per round, 11 gate moments, 2 collapse moments, 9 qudits. For comparison:
one ancilla 12 gates, 7 gate moments; coherent relay (previous two-ancilla runs, six
dedicated ancillas, ancilla-ancilla gates CNOT 9->5 and CNOT_INV 8->10) 16 gates, 9 gate
moments; both 2 collapse moments.

## How it was chosen (`protocol_search.py`, `fault_check.py`, `noise_bench.py`)

Two families, each searched over gate orders (random topological orders scheduled with the
production scheduler, shallowest correct ones kept):
- "mid": N and E read G1/G4, W and S lend; N and E are measured and reset mid-round.
  Best: 11 gate moments, 4 collapse moments.
- "once" (chosen): N and E lend, W and S read; every site read once. Best: 11 gate moments,
  2 collapse moments.
Single faults: every one-qudit Pauli at every location and every two-qudit Pauli after every
two-qudit gate (1592-1608 faults) gives 0 undetected logical errors for the top 3 schedules of
both families, as for the one-ancilla and coherent-relay rounds.
Noisy memory benchmark (3 rounds, production noise injection, 200k shots), acceptance /
logical error among accepted (X basis):

| | one ancilla | coherent relay | data relay, mid | data relay, once |
|---|---|---|---|---|
| SI1000 p=0.005 | 0.393 / 1.9e-4 | 0.289 / 2.8e-4 | 0.188 / 9.3e-4 | 0.345 / 3.0e-4 |
| SI1000 p=0.01 | 0.154 / 9.4e-4 | 0.083 / 6.0e-4 | 0.035 / 2.9e-3 | 0.119 / 1.1e-3 |
| SD6 p=0.01 | 0.151 / 1.9e-3 | 0.061 / 3.3e-3 | 0.038 / 3.8e-3 | 0.063 / 4.7e-3 |

(Z-basis numbers in `bench/`.) Mid-round readout costs two extra global collapse moments,
which under SI1000 put 2p idle noise on every waiting qudit; "once" avoids them.

## Runs (generated 2026-11-01 09:22-09:31, same grids as the earlier two-ancilla campaigns)

- qutrit production: `Run-2026-11-01-09-22-32-folded_qutrit_conv_sites-conv-sites-si1000-sdim140`,
  `Run-2026-11-01-09-24-10-folded_qutrit_conv_sites-conv-sites-sd6-sdim140`
- qutrit high-p extensions: `...09-25-50-...-si1000-sdim140-highp-ext`, `...09-26-20-...-sd6-sdim140-highp-ext`
- qubit production: `Run-2026-11-01-09-26-44-folded_qubit_conv_sites-conv-sites-si1000-sdim140`,
  `Run-2026-11-01-09-28-00-folded_qubit_conv_sites-conv-sites-sd6-sdim140`
- qubit near-threshold refinements: `...09-29-15-...-si1000-sdim140-threshold-refine` (SI1000,
  0.002-0.013 in steps of 0.0005), and for SD6
  `Run-2026-11-02-01-15-57-folded_qubit_conv_sites-conv-sites-sd6-sdim140-threshold-refine-wide`
  (0.003-0.0055 in steps of 0.0005, 0.006-0.026 in steps of 0.001). The first SD6 refinement,
  `...09-31-12-...-sd6-sdim140-threshold-refine` (0.006-0.026, the coherent-relay window), is
  superseded: the k=4 crossing fell below it (0.0040 when interpolated over the coarse
  production grid; 0.0047 on the fine grid). Do not use it.

All nine runs are complete and validated (`validation.txt` in each run folder, status 0).
Merging and error bars follow the existing campaigns exactly: production plus the high-p
extension (qutrit, p_split 0.015 SI1000 / 0.025 SD6, extension k >= 2 at even depths only) or
plus the near-threshold refinement (qubit, scale-free threshold rule), every fit including m=0,
error = sequence-bootstrap SE (1000 replicates, 68%).

## Results (final, 2026-11-02)

All 16 pseudo-thresholds are real crossings (`permanent_regime_interp`, status `final`).
p* +- sequence-bootstrap SE; ratio = conventional sites / coherent relay; gap = (coherent relay -
conventional sites) / SE with the two SEs added in quadrature (independent runs, the paper's rule
for cross-run comparisons).

| code | noise | k | one ancilla | coherent relay | conventional sites | ratio | gap |
|---|---|---|---|---|---|---|---|
| qutrit | SI1000 | 1 | 0.0292 +- 0.0006 | 0.0288 +- 0.0006 | 0.0266 +- 0.0004 | 0.92 | 2.8 sigma |
| qutrit | SI1000 | 2 | 0.0239 +- 0.0005 | 0.0228 +- 0.0005 | 0.0209 +- 0.0003 | 0.92 | 3.2 sigma |
| qutrit | SI1000 | 3 | 0.0166 +- 0.0004 | 0.0150 +- 0.0003 | 0.0136 +- 0.0003 | 0.91 | 3.1 sigma |
| qutrit | SI1000 | 4 | 0.0132 +- 0.0002 | 0.0119 +- 0.0003 | 0.0108 +- 0.0001 | 0.91 | 3.7 sigma |
| qutrit | SD6 | 1 | 0.0399 +- 0.0015 | 0.0360 +- 0.0013 | 0.0283 +- 0.0007 | 0.79 | 5.2 sigma |
| qutrit | SD6 | 2 | 0.0269 +- 0.0007 | 0.0251 +- 0.0015 | 0.0212 +- 0.0003 | 0.84 | 2.5 sigma |
| qutrit | SD6 | 3 | 0.0181 +- 0.0006 | 0.0145 +- 0.0004 | 0.0122 +- 0.0003 | 0.84 | 4.3 sigma |
| qutrit | SD6 | 4 | 0.0155 +- 0.0004 | 0.0125 +- 0.0003 | 0.0103 +- 0.0001 | 0.83 | 6.8 sigma |
| qubit | SI1000 | 1 | 0.0112 +- 0.0002 | 0.0102 +- 0.0002 | 0.0093 +- 0.0001 | 0.91 | 3.3 sigma |
| qubit | SI1000 | 2 | 0.0078 +- 0.0002 | 0.0068 +- 0.0001 | 0.0064 +- 0.0001 | 0.94 | 2.2 sigma |
| qubit | SI1000 | 3 | 0.0049 +- 0.0001 | 0.0043 +- 0.0001 | 0.0040 +- 0.0001 | 0.91 | 3.5 sigma |
| qubit | SI1000 | 4 | 0.0040 +- 0.0001 | 0.0033 +- 0.0001 | 0.0031 +- 0.0001 | 0.94 | 2.7 sigma |
| qubit | SD6 | 1 | 0.0210 +- 0.0008 | 0.0184 +- 0.0004 | 0.0142 +- 0.0004 | 0.77 | 7.0 sigma |
| qubit | SD6 | 2 | 0.0151 +- 0.0005 | 0.0132 +- 0.0004 | 0.0100 +- 0.0003 | 0.76 | 6.8 sigma |
| qubit | SD6 | 3 | 0.0094 +- 0.0003 | 0.0082 +- 0.0002 | 0.0061 +- 0.0002 | 0.75 | 8.3 sigma |
| qubit | SD6 | 4 | 0.0072 +- 0.0002 | 0.0063 +- 0.0001 | 0.0047 +- 0.0001 | 0.75 | 8.5 sigma |

What the numbers say:
- The order is the same in all 16 cases: one ancilla > coherent relay > conventional sites.
- Under SI1000 the conventional sites lose 6-9% against the coherent relay (ratio 0.91-0.94,
  qutrit and qubit alike; 2.2-3.7 sigma). Under SD6 they lose 16-21% for the qutrit (0.79-0.84;
  2.5-6.8 sigma) and 23-25% for the qubit (0.75-0.77; 6.8-8.5 sigma).
- p* falls with k for the conventional sites as for the other layouts: every adjacent-k gap is
  at least 6.6 sigma and positive in all 1000 replicates
  (`sequence-bootstrap/interval_gap_significance-conv-sites.csv`, same paired rule as
  `interval_gap_significance.csv`, which it reproduces to machine precision on the existing
  rows).
- Why: the conventional-site round has 18 two-qudit gates against 16, and 11 gate moments
  against 9, and all 18 gates touch a data qudit, against 14 of the 16 for the coherent relay
  (each lend and un-lend goes through q0, q1 or q4). SI1000 mostly charges the extra gates
  (idle noise 0.1p). SD6 charges full p on every idle qudit in every moment, so the two extra
  moments over nine qudits cost more.
- The extra cost is the price of keeping every gate between a site and its own neighbouring
  data: no ancilla-ancilla gates, no long-range gates.

Files (all in `LRB-experiment-data-slurm/comparisons/`):
- hand-off tables, same format as the existing ones: `pseudo-thresholds-conv-sites-current.{md,csv}`
  (qutrit) and `pseudo-thresholds-qubit-conv-sites-current.{md,csv}` (qubit);
- the layout comparison above: `sdim140-conv-sites/pseudo-thresholds-by-layout.{md,csv,pdf,png}`
  (p* against k, one panel per code and noise model, three layouts);
- LRB error against p at k=1 for the three layouts: `sdim140-conv-sites/lrb-vs-rb-unif-1-{qutrit,qubit}-layouts.{pdf,png}`;
- merged per-configuration outputs (paper-grid error-vs-p figures, summary CSVs):
  `sdim140-conv-sites-{si1000,sd6}-highp-merged/`, `sdim140-qubit-conv-sites-si1000-refine-merged/`,
  `sdim140-qubit-conv-sites-sd6-refine-merged/`;
- sequence bootstraps: `sequence-bootstrap/{conv-two-ancilla,qubit-conv-two-ancilla}-{si1000,sd6}/`;
- qubit vs qutrit with the conventional sites as the two-ancilla layout (no relay rows):
  `qubit-vs-qutrit-conv-sites/threshold-comparison.{md,csv}`,
  `thresholds-vs-interval-qubit-vs-qutrit.{pdf,png}`, `lrb-vs-rb-unif-1-qubit-vs-qutrit.{pdf,png}`
  (`scripts/compare_qubit_qutrit_thresholds.py --two-ancilla conv-sites` in the worktree; the
  older `qubit-vs-qutrit/` folder has coherent-relay rows);
- the four-layout figure `two-ancilla-checks/two-ancilla-checks.{tex,pdf,png}`: row (d) shows the
  simulated roles (W reads G1 while N lends -z1 through q0; S reads G4 while E lends its q1 part
  through q4; N reads G3 before lending, E reads G2 after lending); row (a) is the earlier
  coherent-relay layout.
