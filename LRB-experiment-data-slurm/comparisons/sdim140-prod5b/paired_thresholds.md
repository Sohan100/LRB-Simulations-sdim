# $[[5,1,2]]_3$ (qutrit), one-ancilla layout: the five-moment round vs the seven-moment round, paired pseudo-thresholds

Generated 2026-11-20 00:09 by `scripts/compare_round_variant_thresholds.py` (1000 paired sequence-bootstrap replicates, seed 20261119).

Same Clifford sequences, grids, noise and estimator; only the check round differs. The seven-moment round is the one of the published runs. `paired` p* use one common RB curve (the seven-moment run's); `published` p* use each run's own RB. diff = five-moment - seven-moment, with its paired bootstrap SE and 95% interval.

| noise | k | p* seven-moment | p* five-moment | diff (SE) | diff 95% | ratio (95%) | published seven -> five |
|---|---|---|---|---|---|---|---|
| SI1000 | 1 | 0.0292 | 0.0313 | +0.0021 (0.0002) | [+0.0017, +0.0025] | 1.073 [1.061, 1.085] | 0.0292 -> 0.0313 |
| SI1000 | 2 | 0.0239 | 0.0258 | +0.0019 (0.0002) | [+0.0015, +0.0022] | 1.078 [1.063, 1.091] | 0.0239 -> 0.0258 |
| SI1000 | 3 | 0.0166 | 0.0178 | +0.0012 (0.0002) | [+0.0008, +0.0015] | 1.074 [1.043, 1.094] | 0.0166 -> 0.0178 |
| SI1000 | 4 | 0.0132 | 0.0144 | +0.0012 (0.0003) | [+0.0008, +0.0019] | 1.094 [1.061, 1.145] | 0.0132 -> 0.0144 |
| SD6 | 1 | 0.0399 | 0.0500 | +0.0101 (0.0014) | [+0.0087, +0.0136] | 1.254 [1.222, 1.331] | 0.0399 -> 0.0501 |
| SD6 | 2 | 0.0269 | 0.0326 | +0.0057 (0.0056) | [+0.0051, +0.0198] | 1.213 [1.192, 1.743] | 0.0269 -> 0.0326 |
| SD6 | 3 | 0.0181 | 0.0213 | +0.0033 (0.0004) | [+0.0023, +0.0039] | 1.180 [1.119, 1.222] | 0.0181 -> 0.0214 |
| SD6 | 4 | 0.0155 | 0.0176 | +0.0021 (0.0003) | [+0.0017, +0.0028] | 1.136 [1.109, 1.184] | 0.0155 -> 0.0177 |
