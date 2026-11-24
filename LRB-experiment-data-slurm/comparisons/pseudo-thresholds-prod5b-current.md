# Current LRB-D pseudo-thresholds (sdim 1.4.0, SI1000/SD6, one-ancilla round in 5 gate moments, PROD5B)

Generated 2026-11-19 14:39 by `scripts/merge_highp_threshold_extension.py`; rerun it to refresh. Machine-readable copy: `pseudo-thresholds-prod5b-current.csv` next to this file.

p* is the two-qudit gate error p where the LRB error per Clifford equals physical RB at the same p, same noise model. Intervals are 68% bootstrap. `source=merged` means production data for p < p_split plus the high-p extension (depths 0,1,2,3,4,6, 20M shots, new Clifford sequences; for k >= 2 only its even depths are fitted, matching production) for p >= p_split: SD6 p_split = 0.025, SI1000 p_split = 0.015. `method` ending in `no_interp` means the crossing sits on a grid point; `no_permanent_regime` means no real crossing (not quotable).

Two uncertainty columns. `68% interval (per point)` is the notebook's interval: each depth and each p resampled independently. `error of the mean` is the sequence bootstrap (`scripts/sequence_bootstrap_thresholds.py`, 1000 replicates): whole Clifford sequences are resampled with one draw shared by every depth and every p, every curve is refit and the crossing recomputed, so the correlations from reusing the same 30 sequences are included. It is the standard error over the random-sequence ensemble, given as SE and as its 68% percentile interval. Quote the error of the mean.

| configuration | interval | p* | error of the mean (SE) | error-of-mean 68% interval | 68% interval (per point) | method | source | status |
|---|---|---|---|---|---|---|---|---|
| prod5b-one-ancilla-sd6 | 1 | 0.0501 | 0.0026 | [0.0477, 0.0530] | [0.0488, 0.0515] | permanent_regime_interp | merged | final |
| prod5b-one-ancilla-sd6 | 2 | 0.0326 | 0.0059 | [0.0318, 0.0453] | [0.0323, 0.0451] | permanent_regime_interp | merged | final |
| prod5b-one-ancilla-sd6 | 3 | 0.0214 | 0.0003 | [0.0210, 0.0217] | [0.0211, 0.0216] | permanent_regime_interp | merged | final |
| prod5b-one-ancilla-sd6 | 4 | 0.0177 | 0.0005 | [0.0173, 0.0182] | [0.0175, 0.0178] | permanent_regime_interp | merged | final |
| prod5b-one-ancilla-si1000 | 1 | 0.0313 | 0.0006 | [0.0307, 0.0320] | [0.0311, 0.0316] | permanent_regime_interp | merged | final |
| prod5b-one-ancilla-si1000 | 2 | 0.0258 | 0.0005 | [0.0250, 0.0262] | [0.0255, 0.0260] | permanent_regime_interp | merged | final |
| prod5b-one-ancilla-si1000 | 3 | 0.0178 | 0.0003 | [0.0175, 0.0181] | [0.0175, 0.0179] | permanent_regime_interp | merged | final |
| prod5b-one-ancilla-si1000 | 4 | 0.0144 | 0.0003 | [0.0141, 0.0148] | [0.0142, 0.0146] | permanent_regime_interp | merged | final |

Per-configuration summary CSVs:

- `LRB-experiment-data-slurm/comparisons/sdim140-prod5b-sd6-highp-merged/prod5b-one-ancilla-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-prod5b-si1000-highp-merged/prod5b-one-ancilla-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
