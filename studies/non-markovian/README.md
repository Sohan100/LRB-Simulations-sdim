# Non-Markovian decay in LRB-D

**Question.** Why do some LRB-D decay curves bend away from a single exponential, and how can we still quote a logical gate fidelity from them?

**Answer.** The full write-up is [REPORT.md](REPORT.md). In short:

- Under SI1000, one-ancilla unif1 LRB-D fits a single exponential within statistics. Under SD6 it does not: the decay speeds up with depth.
- The cause is the unchecked mirror inverse. Every policy ends with m Clifford steps and no checks, and pairs of data faults in that block become logical errors at a rate that grows as m².
- Three effects of the protocol and the estimator also bend the curves: the depth-0 boundary, the staircase for k ≥ 2, and the estimator floor at low acceptance. None of them is physical memory.
- One compiled recovery Clifford, in place of the m mirror inverses, fixes the SD6 shape at low p. It does not settle the high-p thresholds.
- The quoted pseudo-threshold intervals cover statistics only. Under SD6 and for k ≥ 2, the threshold moves by 10 to 30% with the choice of estimator and depth set.

[COMPILED_RECOVERY_PLAN.md](COMPILED_RECOVERY_PLAN.md) describes the opt-in `--inverse-mode compiled` option and its checks. That option now lives in `src/lrb/`.

## Contents

| Path | What it holds |
|---|---|
| `REPORT.md` | the study, with the reproduction table in Section 6 |
| `COMPILED_RECOVERY_PLAN.md` | design and verification of the compiled-recovery mode |
| `nmlib.py` | loaders, estimators and a copy of the production fit |
| `diagnose.py`, `estimators.py`, `variant_sim.py`, `variant_summary.py`, `compiled_vs_mirror.py`, `check_gate_conventions.py`, `figures.py` | analysis scripts |
| `run_variants.sh` | the three protocol-variant runs used in the report |
| `compiled-runs/` | scripts that generated and drove the eight compiled-recovery runs, their validation output and pooled fits |
| `results/` | tables (CSV), sampled variant counts (NPZ) and logs |
| `figures/` | Figs. 1 to 5 of the report |

## How to run

We run every script from this folder with the repository's sdim 1.4.0 environment:

```bash
cd studies/non-markovian
PYTHONNOUSERSITE=1 ../../.venv-sdim140/bin/python diagnose.py
```

The scripts import `lrb` from the repository's `src/`. Most of them also read the per-sequence tallies in the `progress/` folders of the production runs under `LRB-experiment-data-slurm/`. Those folders are not in git, so these scripts only run where the runs exist. They write only to `results/` and `figures/`. The two shell scripts in `compiled-runs/` change to the repository root before they start. After the move we checked that every script imports and finds `src/` and `LRB-experiment-data-slurm/` at the repository root. We did not rerun the analyses.

This folder was `non-markovian-brainstorm/` at the repository root before the reorganization.
