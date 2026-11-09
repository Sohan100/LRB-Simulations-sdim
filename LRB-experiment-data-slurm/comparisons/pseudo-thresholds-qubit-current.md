# Current LRB-D pseudo-thresholds (sdim 1.4.0, SI1000/SD6, qubit [[5,1,2]] folded code)

Generated 2026-11-01 03:26 by `scripts/merge_highp_threshold_extension.py`; rerun it to refresh. Machine-readable copy: `pseudo-thresholds-qubit-current.csv` next to this file.

p* is the two-qudit gate error p where the LRB error per Clifford equals physical RB at the same p, same noise model. Intervals are 68% bootstrap. `source=merged` means production data outside the refinement window plus the near-threshold refinement (production depths, 1M shots, 30 new Clifford sequences, fine p grid) inside it: SD6 0.006 <= p <= 0.026 in steps of 0.001, SI1000 0.002 <= p <= 0.013 in steps of 0.0005. p* is where the LRB error equals the RB error, linearly interpolated (tol 0, no error floor; the qutrit settings tol 5e-4 and floor 2e-3 are on the scale of the qubit errors themselves). On the qutrit tables this rule gives the published qutrit p* (15 of 16 identical, the 16th within 0.0002). `method` ending in `no_interp` means the crossing sits on a grid point; `no_permanent_regime` means no real crossing (not quotable).

Two uncertainty columns. `68% interval (per point)` is the notebook's interval: each depth and each p resampled independently. `error of the mean` is the sequence bootstrap (`scripts/sequence_bootstrap_thresholds.py`, 1000 replicates): whole Clifford sequences are resampled with one draw shared by every depth and every p, every curve is refit and the crossing recomputed, so the correlations from reusing the same 30 sequences are included. It is the standard error over the random-sequence ensemble, given as SE and as its 68% percentile interval. Quote the error of the mean.

| configuration | interval | p* | error of the mean (SE) | error-of-mean 68% interval | 68% interval (per point) | method | source | status |
|---|---|---|---|---|---|---|---|---|
| qubit-one-ancilla-sd6 | 1 | 0.0210 | 0.0008 | [0.0201, 0.0217] | [0.0206, 0.0214] | permanent_regime_interp | merged | final |
| qubit-one-ancilla-sd6 | 2 | 0.0151 | 0.0005 | [0.0145, 0.0155] | [0.0148, 0.0153] | permanent_regime_interp | merged | final |
| qubit-one-ancilla-sd6 | 3 | 0.0094 | 0.0003 | [0.0090, 0.0096] | [0.0093, 0.0095] | permanent_regime_interp | merged | final |
| qubit-one-ancilla-sd6 | 4 | 0.0072 | 0.0002 | [0.0070, 0.0075] | [0.0071, 0.0073] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-sd6 | 1 | 0.0184 | 0.0004 | [0.0180, 0.0189] | [0.0180, 0.0187] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-sd6 | 2 | 0.0132 | 0.0004 | [0.0128, 0.0135] | [0.0130, 0.0133] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-sd6 | 3 | 0.0082 | 0.0002 | [0.0080, 0.0083] | [0.0081, 0.0082] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-sd6 | 4 | 0.0063 | 0.0001 | [0.0062, 0.0064] | [0.0062, 0.0063] | permanent_regime_interp | merged | final |
| qubit-one-ancilla-si1000 | 1 | 0.0112 | 0.0002 | [0.0111, 0.0114] | [0.0111, 0.0113] | permanent_regime_interp | merged | final |
| qubit-one-ancilla-si1000 | 2 | 0.0078 | 0.0002 | [0.0076, 0.0079] | [0.0077, 0.0079] | permanent_regime_interp | merged | final |
| qubit-one-ancilla-si1000 | 3 | 0.0049 | 0.0001 | [0.0049, 0.0050] | [0.0049, 0.0050] | permanent_regime_interp | merged | final |
| qubit-one-ancilla-si1000 | 4 | 0.0040 | 0.0001 | [0.0039, 0.0040] | [0.0039, 0.0040] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 1 | 0.0102 | 0.0002 | [0.0100, 0.0105] | [0.0101, 0.0104] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 2 | 0.0068 | 0.0001 | [0.0067, 0.0069] | [0.0067, 0.0069] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 3 | 0.0043 | 0.0001 | [0.0043, 0.0044] | [0.0043, 0.0044] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 4 | 0.0033 | 0.0001 | [0.0033, 0.0034] | [0.0033, 0.0034] | permanent_regime_interp | merged | final |

Per-configuration summary CSVs:

- `LRB-experiment-data-slurm/comparisons/sdim140-qubit-sd6-refine-merged/qubit-one-ancilla-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-qubit-sd6-refine-merged/qubit-two-ancilla-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-qubit-si1000-refine-merged/qubit-one-ancilla-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-qubit-si1000-refine-merged/qubit-two-ancilla-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
