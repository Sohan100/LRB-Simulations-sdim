# Conventional-site two-ancilla layout (previous)

The two-ancilla version the paper used from 2026-11-01 to 2026-11-17, before the two-ancilla
layout replaced it. The data sit as in the paper's patch with `q2` at the centre, and four ancillas
sit at the conventional surface-code sites between them:

| site | data it touches |
|---|---|
| N | q0, q1, q2 |
| W | q0, q2, q3 |
| E | q1, q2, q4 |
| S | q2, q3, q4 |

Only site-data gates between neighbours, no coupled sites and no long-range gates. Two of the
generators do not fit one site, so a data relay lends part of them between sites: N lends `-z1`
through `q0` and W reads `G1`; E lends its `x1` part through `q4` and S reads `G4`. N measures `G3`
before lending and E measures `G2` after, so each site is read once per round. The round has 11 gate
moments and 2 collapse moments.

- **Qutrits:** 5 data, 4 ancillas, 9 in all.
- **Code:** profiles `folded_qutrit_conv_sites`, `folded_qubit_conv_sites`; classes
  `FoldedQutritConventionalSitesDetectionCode`, `FoldedQubitConventionalSitesDetectionCode`.
- **Runs:** `LRB-experiment-data-slurm/Run-2026-11-0*-conv_sites-*`, analysed in
  `notebooks/06-conventional-site-two-ancilla-si1000-sd6.ipynb`.
- **Design record:** `LRB-experiment-data-slurm/comparisons/two-ancilla-checks/CONV-SITES-FACTS.md`
  and the scripts beside it.

| code | noise | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|
| qutrit | SI1000 | 0.0266(4) | 0.0209(3) | 0.0136(3) | 0.01084(14) |
| qutrit | SD6 | 0.0283(7) | 0.0212(3) | 0.0122(3) | 0.01034(14) |
| qubit | SI1000 | 0.00932(14) | 0.00642(12) | 0.00395(8) | 0.00313(5) |
| qubit | SD6 | 0.0142(4) | 0.0100(3) | 0.00608(15) | 0.00469(12) |

(`pseudo-thresholds-conv-sites-current.md` and `pseudo-thresholds-qubit-conv-sites-current.md`.)
