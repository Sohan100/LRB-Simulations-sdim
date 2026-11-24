# One-ancilla layout

Panel (b) of `../figures/three-layouts.pdf`. Each stabilizer generator `G_j` gets its own ancilla
`a_j`, which reads it every round.

- **Qutrits:** 5 data, 4 ancillas, 9 in all.
- **Couplings:** 13. Twelve join an ancilla to its data qutrits; the thirteenth is the long-range
  `q2-q3` coupling that the logical S and H need.
- **Check round (the five-moment round):** 12 two-qutrit and 4 one-qutrit gates in 5 gate moments,
  7 moments with reset and readout. Of all 31,104 correct gate orders of this round (qutrit code),
  it has the lowest leading-order undetected logical error per round under both SI1000 and SD6, and
  no single fault goes undetected. The paper's earlier round needed 7 gate
  moments (the seven-moment round); the five-moment round replaces it.
- **Encoder:** the layout's own encoder has 8 two-qutrit gates through `a1` and `a2` in 5 gate
  moments. The simulations prepare `|+_L>` with the paper's earlier four-gate circuit instead.
  Encoding is noiseless in the simulations, so no result depends on this choice.
- **Hardware:** the one-ancilla checks alone fit IQM Garnet's square lattice, but not together with
  the `q2-q3` coupler (a 5-cycle; see `../garnet-terminal-check/connectivity/FINDINGS.md`).

Circuits: `../figures/encoders.pdf` and `../figures/check-rounds.pdf`, panels (a) to (d), and pages
1 and 3 of `../figures/layouts-and-circuits-overview.pdf`.

## Code and runs

| round | code profile (qutrit / qubit) | code class |
|---|---|---|
| five-moment | `folded_qutrit_prod5b` / `folded_qubit_prod5b` | `FoldedQutritProd5bDetectionCode`, `FoldedQubitProd5bDetectionCode` |
| seven-moment | `folded_qutrit` / `folded_qubit` | `FoldedQutritDetectionCode`, `FoldedQubitDetectionCode` |

Code classes are in `src/lrb/code_definitions.py`, profiles in `src/lrb/code_simulation_profiles.py`.
Run folders in `LRB-experiment-data-slurm/` contain `prod5b-single-ancilla` (five-moment round) or
`folded_qutrit-single-ancilla` / `folded_qubit-single-ancilla` (seven-moment round) in their names.
Notebooks: `notebooks/08-five-moment-round-qutrit-si1000-sd6.ipynb`,
`notebooks/09-five-moment-round-qubit-si1000-sd6.ipynb`,
`notebooks/04-qutrit-one-and-relay-two-ancilla-si1000-sd6.ipynb`.

## Pseudo-thresholds

`p*` for check interval `k`, with one standard error from the sequence bootstrap
(`LRB-experiment-data-slurm/comparisons/pseudo-thresholds-prod5b-current.md` and
`pseudo-thresholds-current.md`, plus their qubit versions):

| round | code | noise | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|---|
| five-moment | qutrit | SI1000 | 0.0313(6) | 0.0258(5) | 0.0178(3) | 0.0144(3) |
| five-moment | qutrit | SD6 | 0.0501(26) | 0.033(6) | 0.0214(3) | 0.0177(5) |
| five-moment | qubit | SI1000 | 0.0123(4) | 0.00866(16) | 0.00564(9) | 0.00450(8) |
| five-moment | qubit | SD6 | 0.0290(10) | 0.0203(7) | 0.0123(4) | 0.0095(3) |
| seven-moment | qutrit | SI1000 | 0.0292(6) | 0.0239(5) | 0.0166(4) | 0.0132(2) |
| seven-moment | qutrit | SD6 | 0.0399(15) | 0.0269(7) | 0.0181(6) | 0.0155(4) |
| seven-moment | qubit | SI1000 | 0.0112(2) | 0.00778(15) | 0.00495(9) | 0.00395(8) |
| seven-moment | qubit | SD6 | 0.0210(8) | 0.0151(5) | 0.0094(3) | 0.0072(2) |

The search for the round and the square-lattice alternatives (none beats this layout) are in
`LRB-experiment-data-slurm/comparisons/two-ancilla-checks/hw-layouts/one-ancilla/` and
`LRB-experiment-data-slurm/comparisons/sdim140-prod5b/README.md`.
