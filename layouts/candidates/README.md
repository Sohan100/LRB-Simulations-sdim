# Layout candidates

Layouts we designed and benchmarked but did not take into the paper. The search code, the
benchmarks and the full tables are in
`LRB-experiment-data-slurm/comparisons/two-ancilla-checks/hw-layouts/` (start with its README).

| candidate | qutrits | couplers | what it is | why not |
|---|---|---|---|---|
| A1 (`a1-layout.pdf`, `a1-check-round.pdf`) | 7 | 8 | square lattice that also runs the paper's four-gate encoder natively; 22 gate moments per round | weak under SD6; the two-ancilla layout has a nearest-neighbour encoder of its own and half its logical error |
| 2a | 7 | 9 | two ancillas on a 3 x 3 grid, reading G1, G3 and G2, G4 directly | needs diagonal couplers |
| square lattice only | 7 | 7 | a reads G3 and G1, c reads G2 and G4, using a data CNOT across `q2-q3` | superseded by the two-ancilla layout |
| SQ14 | 9 | 10 | one ancilla per check on a square lattice | about the same pseudo-thresholds as the one-ancilla layout, which the five-moment round then improved |

A1 has a code profile (`folded_qutrit_a1`, `folded_qubit_a1`) and tests
(`tests/test_a1_square_layout_code.py`), but no production runs.
