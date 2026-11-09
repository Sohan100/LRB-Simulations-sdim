# Current LRB-D pseudo-thresholds (sdim 1.4.0, SI1000/SD6)

Generated 2026-10-31 04:55 by `scripts/merge_highp_threshold_extension.py`; rerun it to refresh. Machine-readable copy: `pseudo-thresholds-current.csv` next to this file.

p* is the two-qudit gate error p where the LRB error per Clifford equals physical RB at the same p, same noise model. Intervals are 68% bootstrap. `source=merged` means production data for p < p_split plus the high-p extension (depths 0,1,2,3,4,6, 20M shots, new Clifford sequences; for k >= 2 only its even depths are fitted, matching production) for p >= p_split: SD6 p_split = 0.025, SI1000 p_split = 0.015. `method` ending in `no_interp` means the crossing sits on a grid point; `no_permanent_regime` means no real crossing (not quotable).

Two uncertainty columns. `68% interval (per point)` is the notebook's interval: each depth and each p resampled independently. `error of the mean` is the sequence bootstrap (`scripts/sequence_bootstrap_thresholds.py`, 1000 replicates): whole Clifford sequences are resampled with one draw shared by every depth and every p, every curve is refit and the crossing recomputed, so the correlations from reusing the same 30 sequences are included. It is the standard error over the random-sequence ensemble, given as SE and as its 68% percentile interval. Quote the error of the mean.

| configuration | interval | p* | error of the mean (SE) | error-of-mean 68% interval | 68% interval (per point) | method | source | status |
|---|---|---|---|---|---|---|---|---|
| one-ancilla-sd6 | 1 | 0.0399 | 0.0015 | [0.0381, 0.0411] | [0.0389, 0.0406] | permanent_regime_interp | merged | final |
| one-ancilla-sd6 | 2 | 0.0269 | 0.0007 | [0.0261, 0.0275] | [0.0265, 0.0272] | permanent_regime_interp | merged | final |
| one-ancilla-sd6 | 3 | 0.0181 | 0.0006 | [0.0176, 0.0186] | [0.0178, 0.0184] | permanent_regime_interp | merged | final |
| one-ancilla-sd6 | 4 | 0.0155 | 0.0004 | [0.0151, 0.0159] | [0.0154, 0.0157] | permanent_regime_interp | merged | final |
| two-ancilla-sd6 | 1 | 0.0360 | 0.0013 | [0.0349, 0.0377] | [0.0352, 0.0367] | permanent_regime_interp | merged | final |
| two-ancilla-sd6 | 2 | 0.0251 | 0.0015 | [0.0245, 0.0266] | [0.0245, 0.0259] | permanent_regime_interp | merged | final |
| two-ancilla-sd6 | 3 | 0.0145 | 0.0004 | [0.0141, 0.0150] | [0.0143, 0.0149] | permanent_regime_interp | merged | final |
| two-ancilla-sd6 | 4 | 0.0125 | 0.0003 | [0.0121, 0.0127] | [0.0123, 0.0126] | permanent_regime_no_interp | merged | final |
| one-ancilla-si1000 | 1 | 0.0292 | 0.0006 | [0.0286, 0.0298] | [0.0289, 0.0294] | permanent_regime_interp | merged | final |
| one-ancilla-si1000 | 2 | 0.0239 | 0.0005 | [0.0234, 0.0244] | [0.0237, 0.0241] | permanent_regime_interp | merged | final |
| one-ancilla-si1000 | 3 | 0.0166 | 0.0004 | [0.0162, 0.0171] | [0.0165, 0.0168] | permanent_regime_interp | merged | final |
| one-ancilla-si1000 | 4 | 0.0132 | 0.0002 | [0.0130, 0.0134] | [0.0131, 0.0133] | permanent_regime_interp | merged | final |
| two-ancilla-si1000 | 1 | 0.0288 | 0.0006 | [0.0282, 0.0294] | [0.0284, 0.0291] | permanent_regime_interp | merged | final |
| two-ancilla-si1000 | 2 | 0.0228 | 0.0005 | [0.0223, 0.0233] | [0.0225, 0.0230] | permanent_regime_interp | merged | final |
| two-ancilla-si1000 | 3 | 0.0150 | 0.0003 | [0.0148, 0.0154] | [0.0150, 0.0153] | permanent_regime_no_interp | merged | final |
| two-ancilla-si1000 | 4 | 0.0119 | 0.0003 | [0.0117, 0.0122] | [0.0118, 0.0120] | permanent_regime_interp | merged | final |

Per-configuration summary CSVs:

- `LRB-experiment-data-slurm/comparisons/sdim140-sd6-highp-merged/one-ancilla-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-sd6-highp-merged/two-ancilla-sd6/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-si1000-highp-merged/one-ancilla-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
- `LRB-experiment-data-slurm/comparisons/sdim140-si1000-highp-merged/two-ancilla-si1000/unif_thresholds_summary_monotone_trim_zoom_pwindow.csv`
