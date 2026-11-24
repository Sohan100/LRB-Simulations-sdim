# Current LRB-D pseudo-thresholds (sdim 1.4.0, SI1000/SD6, two-ancilla layout)

Generated 2026-11-18 15:58 by `scripts/merge_highp_threshold_extension.py`; rerun it to refresh. Machine-readable copy: `pseudo-thresholds-layout-b-current.csv` next to this file.

p* is the two-qudit gate error p where the LRB error per Clifford equals physical RB at the same p, same noise model. Intervals are 68% bootstrap. `source=merged` means production data for p < p_split plus the high-p extension (depths 0,1,2,3,4,6, 20M shots, new Clifford sequences; for k >= 2 only its even depths are fitted, matching production) for p >= p_split: SD6 p_split = 0.0125, SI1000 p_split = 0.0125. `method` ending in `no_interp` means the crossing sits on a grid point; `no_permanent_regime` means no real crossing (not quotable).

Two uncertainty columns. `68% interval (per point)` is the notebook's interval: each depth and each p resampled independently. `error of the mean` is the sequence bootstrap (`scripts/sequence_bootstrap_thresholds.py`, 1000 replicates): whole Clifford sequences are resampled with one draw shared by every depth and every p, every curve is refit and the crossing recomputed, so the correlations from reusing the same 30 sequences are included. It is the standard error over the random-sequence ensemble, given as SE and as its 68% percentile interval. Quote the error of the mean.

| configuration | interval | p* | error of the mean (SE) | error-of-mean 68% interval | 68% interval (per point) | method | source | status |
|---|---|---|---|---|---|---|---|---|
| two-ancilla-sd6 | 1 | 0.0206 | 0.0004 | [0.0203, 0.0210] | [0.0204, 0.0208] | permanent_regime_interp | merged | final |
| two-ancilla-sd6 | 2 | 0.0147 | 0.0003 | [0.0144, 0.0151] | [0.0145, 0.0150] | permanent_regime_interp | merged | final |
| two-ancilla-sd6 | 3 | 0.0088 | 0.0002 | [0.0086, 0.0091] | [0.0087, 0.0089] | permanent_regime_interp | merged | final |
| two-ancilla-sd6 | 4 | 0.0074 | 0.0002 | [0.0071, 0.0076] | [0.0073, 0.0075] | permanent_regime_interp | merged | final |
| two-ancilla-si1000 | 1 | 0.0191 | 0.0004 | [0.0187, 0.0197] | [0.0189, 0.0193] | permanent_regime_interp | merged | final |
| two-ancilla-si1000 | 2 | 0.0139 | 0.0002 | [0.0136, 0.0141] | [0.0138, 0.0140] | permanent_regime_interp | merged | final |
| two-ancilla-si1000 | 3 | 0.0085 | 0.0002 | [0.0083, 0.0087] | [0.0084, 0.0085] | permanent_regime_interp | merged | final |
| two-ancilla-si1000 | 4 | 0.0064 | 0.0001 | [0.0063, 0.0066] | [0.0064, 0.0065] | permanent_regime_interp | merged | final |

Per-configuration summary CSVs:

- `LRB-experiment-data-slurm/comparisons/sdim140-layout-b-sd6-highp-merged/layout-b-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-layout-b-si1000-highp-merged/layout-b-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
