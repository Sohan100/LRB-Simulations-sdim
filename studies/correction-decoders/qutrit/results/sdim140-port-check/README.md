# sdim 1.4.0 port check

On 2026-11-24 we ran the baseline stage of the ported study on the sdim 1.4.0 SI1000 one-ancilla run
`Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140`: 30 Clifford sequences,
9 depths, 14 values of p, 45 LRB-D policies. The study computes LRB-D exactly from the chain DEM;
production sampled 1M shots per circuit with sdim 1.4.0's own DEM. `baseline_reproduction.csv.gz`
holds the 5,670 (p, policy, depth) rows; its columns are those of the report's
`results/baseline_reproduction.csv.gz`.

| | this port, sdim140 run | original study, Aug run |
|---|---|---|
| max abs rejection difference | 3.1e-4 | 2.8e-4 |
| fidelity rows (production kept at least 1,000 shots per circuit) | 4,403 | 4,210 |
| median abs fidelity z | 0.34 | 0.43 |
| 95th percentile abs fidelity z | 3.45 | 3.71 |

Rejection z-scores (sequence standard error) have mean −0.04 and standard deviation 0.34 over the
4,389 rows where the rejection is below 0.999, and none exceeds 3. Fidelity z-scores stay below 3
up to p = 1.25%. From p = 1.5% on they drift negative (mean −0.3 at 1.5%, −1.9 at 5%), as in the
original study: at low acceptance the production estimator `abs(E[omega^k])` is biased up by about
`0.89/sqrt(N_acc)` (REPORT.md, Section 2, validation table).

Commands, from the repository root:

```bash
RUN=LRB-experiment-data-slurm/Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140
S=studies/correction-decoders/qutrit/scripts
python $S/run_decoder_study.py prepare --run-folder $RUN --cache .cache_correction/structures --workers 24
python $S/run_decoder_study.py run --run-folder $RUN --cache .cache_correction/structures \
    --out .cache_correction/results_baseline --workers 24
python $S/analyze_study.py --baseline-results .cache_correction/results_baseline \
    --run-folder $RUN --out .cache_correction/analysis
```

The three stages took 19 minutes with 24 workers on a Perlmutter login node.
