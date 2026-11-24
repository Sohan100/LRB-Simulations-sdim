# $[[5,1,2]]_2$ (qubit), one-ancilla layout: the five-moment round vs the seven-moment round, paired pseudo-thresholds

Generated 2026-11-20 01:19 by `scripts/compare_round_variant_thresholds.py` (1000 paired sequence-bootstrap replicates, seed 20261119).

Same Clifford sequences, grids, noise and estimator; only the check round differs. The seven-moment round is the one of the published runs. `paired` p* use one common RB curve (the seven-moment run's); `published` p* use each run's own RB. diff = five-moment - seven-moment, with its paired bootstrap SE and 95% interval.

| noise | k | p* seven-moment | p* five-moment | diff (SE) | diff 95% | ratio (95%) | published seven -> five |
|---|---|---|---|---|---|---|---|
| SI1000 | 1 | 0.0112 | 0.0123 | +0.0011 (0.0003) | [+0.0008, +0.0016] | 1.096 [1.069, 1.137] | 0.0112 -> 0.0123 |
| SI1000 | 2 | 0.0078 | 0.0087 | +0.0009 (0.0001) | [+0.0007, +0.0010] | 1.113 [1.089, 1.130] | 0.0078 -> 0.0087 |
| SI1000 | 3 | 0.0049 | 0.0057 | +0.0007 (0.0000) | [+0.0006, +0.0008] | 1.143 [1.119, 1.161] | 0.0049 -> 0.0056 |
| SI1000 | 4 | 0.0040 | 0.0045 | +0.0005 (0.0000) | [+0.0005, +0.0006] | 1.139 [1.117, 1.159] | 0.0040 -> 0.0045 |
| SD6 | 1 | 0.0210 | 0.0290 | +0.0080 (0.0012) | [+0.0064, +0.0111] | 1.381 [1.289, 1.563] | 0.0210 -> 0.0290 |
| SD6 | 2 | 0.0151 | 0.0202 | +0.0052 (0.0004) | [+0.0045, +0.0060] | 1.342 [1.306, 1.397] | 0.0151 -> 0.0203 |
| SD6 | 3 | 0.0094 | 0.0123 | +0.0030 (0.0002) | [+0.0025, +0.0034] | 1.316 [1.273, 1.355] | 0.0094 -> 0.0123 |
| SD6 | 4 | 0.0072 | 0.0095 | +0.0023 (0.0001) | [+0.0021, +0.0025] | 1.317 [1.285, 1.345] | 0.0072 -> 0.0095 |
