# Two-ancilla layout

Panel (c) of `../figures/three-layouts.pdf`, drawn alone in `two-ancilla-layout.pdf`. Two
ancillas read all four generators, one after the other, and every coupler is a nearest-neighbour
pair on a square lattice. It replaced the conventional-site two-ancilla layout in the paper.

```
q4  N
q1  q2  E
q0  q3
```

- **Qutrits:** 5 data, 2 ancillas (`N`, `E`), 7 in all.
- **Couplings:** 8, exactly the lattice-adjacent pairs. `q0-q3` is used only by the encoder; the
  `q2-q3` coupling carries the two-qutrit gates of the logical S and H.
- **Check round:** `N` reads `G2` then `G4`, `E` reads `G1` then `G3`. 24 two-qutrit gates (16 of them
  between data qutrits) and 4 one-qutrit gates in 19 gate moments, 22 moments with reset and
  readout (`two-ancilla-check-round.pdf`, and `../figures/check-rounds.pdf` panels (e) to (h)).
  No single fault goes undetected.
- **Encoder:** 4 two-qutrit gates in 4 gate moments, all nearest-neighbour. The simulations use it.
- **Hardware:** fits IQM Garnet in 48 placements.

## Code and runs

- Profiles `folded_qutrit_layout_b` and `folded_qubit_layout_b` (internal names), classes
  `FoldedQutritLayoutBDetectionCode` and `FoldedQubitLayoutBDetectionCode` in
  `src/lrb/code_definitions.py`.
- Run folders `LRB-experiment-data-slurm/Run-2026-11-17-*-layout-b-*`, analysed in
  `notebooks/07-two-ancilla-layout-si1000-sd6.ipynb`.
- Write-up and comparisons: `LRB-experiment-data-slurm/comparisons/sdim140-layout-b/` (start with
  `TWO-ANCILLA-FACTS.md`); the co-design search for this layout and its encoder:
  `LRB-experiment-data-slurm/comparisons/two-ancilla-checks/hw-layouts/README.md`.

## Pseudo-thresholds

One standard error from the sequence bootstrap (`pseudo-thresholds-layout-b-current.md` and
`pseudo-thresholds-qubit-layout-b-current.md` in `LRB-experiment-data-slurm/comparisons/`):

| code | noise | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|
| qutrit | SI1000 | 0.0191(4) | 0.0139(2) | 0.00849(17) | 0.00643(13) |
| qutrit | SD6 | 0.0206(4) | 0.0147(3) | 0.0088(2) | 0.00740(20) |
| qubit | SI1000 | 0.00511(15) | 0.00313(7) | 0.00197(4) | 0.00152(3) |
| qubit | SD6 | 0.00854(18) | 0.00578(12) | 0.00347(6) | 0.00271(4) |

The round is almost four times as deep as the one-ancilla layout's five-moment round (19 gate
moments against 5) and the pseudo-thresholds are lower: the two ancillas buy a square lattice and
two fewer qutrits at the cost of depth.
