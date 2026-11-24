# Relay two-ancilla layout (superseded)

Our first two-ancilla version (June to October 2026). It keeps the one-ancilla layout's checks but
splits the two geometrically nonlocal boundary generators: a second, local relay ancilla carries
half of each, so no ancilla needs a long-range coupler. The code and the stabilizers are unchanged;
only the check circuits and the syndrome arithmetic differ.

- **Qutrits:** 5 data, 6 ancillas, 11 in all.
- **Code:** profiles `folded_qutrit_split_unif`, `folded_qubit_split_unif`; classes
  `FoldedQutritSplitAncillaUniformDetectionCode`, `FoldedQubitSplitAncillaUniformDetectionCode`.
- **Runs:** folder names containing `split_unif` or `split-ancilla`, analysed in
  `notebooks/02-si1000-relay-two-ancilla-2026-06.ipynb`,
  `notebooks/04-qutrit-one-and-relay-two-ancilla-si1000-sd6.ipynb` and
  `notebooks/05-qubit-one-and-relay-two-ancilla-si1000-sd6.ipynb`.
- **Status:** left out of the paper on 2026-11-02 in favour of the conventional sites, and later
  the two-ancilla layout. On IQM Garnet no relay layout fits together with the `q2-q3` coupler
  (`../garnet-terminal-check/connectivity/FINDINGS.md`).

| code | noise | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|
| qutrit | SI1000 | 0.0288(6) | 0.0228(5) | 0.0150(3) | 0.0119(3) |
| qutrit | SD6 | 0.0360(13) | 0.0251(15) | 0.0145(4) | 0.0125(3) |
| qubit | SI1000 | 0.0102(2) | 0.00679(12) | 0.00433(8) | 0.00333(5) |
| qubit | SD6 | 0.0184(4) | 0.0132(4) | 0.00816(20) | 0.00629(14) |

(rows `two-ancilla-*` of `pseudo-thresholds-current.md` and `pseudo-thresholds-qubit-current.md`.)
