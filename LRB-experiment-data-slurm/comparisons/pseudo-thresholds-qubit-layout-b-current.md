# Current LRB-D pseudo-thresholds (sdim 1.4.0, SI1000/SD6, qubit [[5,1,2]] folded code, two-ancilla layout)

Generated 2026-11-18 16:04 by `scripts/merge_highp_threshold_extension.py`; rerun it to refresh. Machine-readable copy: `pseudo-thresholds-qubit-layout-b-current.csv` next to this file.

p* is the two-qudit gate error p where the LRB error per Clifford equals physical RB at the same p, same noise model. Intervals are 68% bootstrap. `source=merged` means production data outside the refinement window plus the near-threshold refinement (production depths, 1M shots, 30 new Clifford sequences, fine p grid) inside it: SD6 0.0015 <= p <= 0.009 in steps of 0.00025, SI1000 0.001 <= p <= 0.007 in steps of 0.00025. p* is where the LRB error equals the RB error, linearly interpolated (tol 0, no error floor; the qutrit settings tol 5e-4 and floor 2e-3 are on the scale of the qubit errors themselves). On the qutrit tables this rule gives every published one-ancilla and conventional-site p* exactly (16 of 16); of the eight coherent-relay p* it gives six exactly and two within 0.0002 (SD6 k=4 +0.00001, SI1000 k=3 +0.00015). `method` ending in `no_interp` means the crossing sits on a grid point; `no_permanent_regime` means no real crossing (not quotable).

Two uncertainty columns. `68% interval (per point)` is the notebook's interval: each depth and each p resampled independently. `error of the mean` is the sequence bootstrap (`scripts/sequence_bootstrap_thresholds.py`, 1000 replicates): whole Clifford sequences are resampled with one draw shared by every depth and every p, every curve is refit and the crossing recomputed, so the correlations from reusing the same 30 sequences are included. It is the standard error over the random-sequence ensemble, given as SE and as its 68% percentile interval. Quote the error of the mean.

| configuration | interval | p* | error of the mean (SE) | error-of-mean 68% interval | 68% interval (per point) | method | source | status |
|---|---|---|---|---|---|---|---|---|
| qubit-two-ancilla-sd6 | 1 | 0.0085 | 0.0002 | [0.0083, 0.0087] | [0.0084, 0.0086] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-sd6 | 2 | 0.0058 | 0.0001 | [0.0057, 0.0059] | [0.0057, 0.0058] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-sd6 | 3 | 0.0035 | 0.0001 | [0.0034, 0.0035] | [0.0034, 0.0035] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-sd6 | 4 | 0.0027 | 0.0000 | [0.0027, 0.0027] | [0.0027, 0.0027] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 1 | 0.0051 | 0.0002 | [0.0050, 0.0052] | [0.0051, 0.0051] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 2 | 0.0031 | 0.0001 | [0.0031, 0.0032] | [0.0031, 0.0032] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 3 | 0.0020 | 0.0000 | [0.0019, 0.0020] | [0.0020, 0.0020] | permanent_regime_interp | merged | final |
| qubit-two-ancilla-si1000 | 4 | 0.0015 | 0.0000 | [0.0015, 0.0015] | [0.0015, 0.0015] | permanent_regime_interp | merged | final |

Per-configuration summary CSVs:

- `LRB-experiment-data-slurm/comparisons/sdim140-qubit-layout-b-sd6-refine-merged/qubit-layout-b-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-qubit-layout-b-si1000-refine-merged/qubit-layout-b-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
