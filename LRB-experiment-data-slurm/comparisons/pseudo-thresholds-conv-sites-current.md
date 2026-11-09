# Current LRB-D pseudo-thresholds (sdim 1.4.0, SI1000/SD6, two ancillas on the conventional sites)

Generated 2026-11-02 03:03 by `scripts/merge_highp_threshold_extension.py`; rerun it to refresh. Machine-readable copy: `pseudo-thresholds-conv-sites-current.csv` next to this file.

p* is the two-qudit gate error p where the LRB error per Clifford equals physical RB at the same p, same noise model. Intervals are 68% bootstrap. `source=merged` means production data for p < p_split plus the high-p extension (depths 0,1,2,3,4,6, 20M shots, new Clifford sequences; for k >= 2 only its even depths are fitted, matching production) for p >= p_split: SD6 p_split = 0.025, SI1000 p_split = 0.015. `method` ending in `no_interp` means the crossing sits on a grid point; `no_permanent_regime` means no real crossing (not quotable).

Two uncertainty columns. `68% interval (per point)` is the notebook's interval: each depth and each p resampled independently. `error of the mean` is the sequence bootstrap (`scripts/sequence_bootstrap_thresholds.py`, 1000 replicates): whole Clifford sequences are resampled with one draw shared by every depth and every p, every curve is refit and the crossing recomputed, so the correlations from reusing the same 30 sequences are included. It is the standard error over the random-sequence ensemble, given as SE and as its 68% percentile interval. Quote the error of the mean.

| configuration | interval | p* | error of the mean (SE) | error-of-mean 68% interval | 68% interval (per point) | method | source | status |
|---|---|---|---|---|---|---|---|---|
| conv-two-ancilla-sd6 | 1 | 0.0283 | 0.0007 | [0.0277, 0.0291] | [0.0279, 0.0288] | permanent_regime_interp | merged | final |
| conv-two-ancilla-sd6 | 2 | 0.0212 | 0.0003 | [0.0209, 0.0214] | [0.0209, 0.0215] | permanent_regime_interp | merged | final |
| conv-two-ancilla-sd6 | 3 | 0.0122 | 0.0003 | [0.0119, 0.0126] | [0.0120, 0.0125] | permanent_regime_interp | merged | final |
| conv-two-ancilla-sd6 | 4 | 0.0103 | 0.0001 | [0.0102, 0.0105] | [0.0103, 0.0104] | permanent_regime_interp | merged | final |
| conv-two-ancilla-si1000 | 1 | 0.0266 | 0.0004 | [0.0262, 0.0271] | [0.0264, 0.0268] | permanent_regime_interp | merged | final |
| conv-two-ancilla-si1000 | 2 | 0.0209 | 0.0003 | [0.0206, 0.0212] | [0.0207, 0.0210] | permanent_regime_interp | merged | final |
| conv-two-ancilla-si1000 | 3 | 0.0136 | 0.0003 | [0.0134, 0.0139] | [0.0135, 0.0138] | permanent_regime_interp | merged | final |
| conv-two-ancilla-si1000 | 4 | 0.0108 | 0.0001 | [0.0107, 0.0110] | [0.0108, 0.0109] | permanent_regime_interp | merged | final |

Per-configuration summary CSVs:

- `LRB-experiment-data-slurm/comparisons/sdim140-conv-sites-sd6-highp-merged/conv-two-ancilla-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-conv-sites-si1000-highp-merged/conv-two-ancilla-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
