# Correction instead of detection: the qutrit [[5,1,2]]_3 code

**Questions.** Which decoders let LRB correct errors instead of rejecting shots, and how do they compare with LRB-D? Does correction mitigate non-Markovian decay? What is a sustainable way to run this on hardware?

**Answers.** The full write-up is [REPORT.md](REPORT.md). In short:

- Exact maximum-likelihood decoding (MLD) of the circuit-level DEM is cheap for this code: a forward filter over 243 states per check round. The decision rule matters more than the decoder.
- Full correction does not work at distance 2. At p = 1e-3 it is 3.1 times worse than physical RB, because 16 of the 40 single-qutrit errors are ambiguous.
- Selective rules trade accepted error for acceptance. Blip-forgiving detection more than doubles the acceptance at m = 22 and keeps the accepted error 70 times below physical RB. Soft-output MLD traces the optimal acceptance-fidelity frontier. LRB-D with a check every round sits on that frontier; sparser LRB-D policies fall below it.
- Correction brings the decay closer to a single exponential only by making the logical error first order, and some curvature remains. It does remove the bias that postselection picks up when the noise strength drifts from shot to shot.
- The protocol matters more than the decoder: read all five data qutrits at the end and read the ancillas in parallel. Genuine correction needs distance 3.

Scope: the study used the pre-sdim-1.4.0 pipeline. The baseline is the one-ancilla SI1000 run `Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000`, and physical RB is compared at p_RB = 5p. The newer runs use same-p RB and a different SI1000 schedule. Section 5 of the [non-Markovian study](../../non-markovian/REPORT.md) rechecks some of these claims and says which ratios we expect to change.

## Status: runs on sdim 1.4.0

The code is part of the package and runs on sdim 1.4.0:

- `src/lrb/correction/`: the chain DEM, the exact forward filter, the decision rules, the sampler, the per-circuit study and the protocol variants. `compact_dem.py` there is the per-fault DEM builder the chain model reads. It was the repository's DEM backend before sdim 1.4.0; the production backend now wraps `sdim.dem.DetectorErrorModel`, whose merged mechanisms do not keep per-location data.
- `scripts/` in this folder: the drivers, aggregation, tables, figures and the first-order check. They read the baseline run from `LRB_BASELINE_RUN` (default: the August run the report used).
- `tests/test_correction_chain.py`, `tests/test_correction_variants.py` and `tests/test_correction_io.py` run in CI.

The port changed three things:

- **Circuit files.** sdim 1.4.0 run files do not store each noise gate's `prob_scale`, which the chain DEM needs to rescale the noise to other values of p, and sdim 1.4.0's reader rejects the field in older files. `compact_dem.read_run_circuit` restores it: from `prob / p` for new files, from the stored field for old ones. The note scripts under `notes/` read through it too.
- **Reference shot.** sdim 1.4.0 applies N1/N2 noise in the tableau reference shot unless told not to. The DEM builder now switches it off, as `Program.simulate` does.
- **Variant circuits.** `variants.build_variant_circuit` builds the generator's current schedule: each Clifford is one noisy, moment-scheduled segment and the ancilla reset sits inside each check round. With every flag off it equals `generate_lrb_clifford_sequence` gate for gate (tested). The generator already reads all ancillas in one moment, so `parallel_measurement` now only reorders the entangling gates.

We checked the port end to end on 2026-11-24: the baseline stage on the sdim 1.4.0 SI1000 one-ancilla run reproduces the production LRB-D rejection and fidelity as closely as the original study reproduced its own run ([results/sdim140-port-check](results/sdim140-port-check/README.md)).

The numbers in REPORT.md still come from the earlier pipeline: sdim 1.3.x, the one-ancilla SI1000 run `Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000` (old schedule, its circuit files read through `read_run_circuit`), and physical RB at p_RB = 5p. Pointing `LRB_BASELINE_RUN` at a sdim 1.4.0 run gives the current-schedule numbers. We have not redone the report with them.

## Contents

| Path | What it holds |
|---|---|
| `REPORT.md` | the study; Section 12 gives the commands |
| `bibliography_raw.csv` | every citation and its independent verification status |
| `notes/` | analysis notes A1 to A5, hardware notes H1 and H2, the baseline analysis N1, literature notes L1 to L8 with citation checks, the critiques, the code-review summaries, and the small scripts these notes cite (kept as they ran, so `notes/ruff.toml` turns off the unused-name checks) |
| `results/` | decoder and variant tables (Markdown), curves and fits (`*.csv.gz`), RB reference, Pareto frontier, the first-order check, the baseline non-Markovianity tables, the figures the report shows, and the sdim 1.4.0 port check |
| `scripts/` | drivers, aggregation, tables, figures, the first-order check |

The package code is in `src/lrb/correction/` and the tests in `tests/test_correction_*.py`.

## How to run it

From the repository root, with the package installed (`pip install -e ".[dev]"`):

```bash
python -m pytest -q tests/test_correction_chain.py tests/test_correction_variants.py tests/test_correction_io.py
export LRB_BASELINE_RUN=$PWD/LRB-experiment-data-slurm/Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140
salloc -N 1 -C cpu -q debug -t 00:30:00 srun -n 1 -c 256 \
    bash studies/correction-decoders/qutrit/scripts/run_all_on_node.sh baseline
```

The tests take about 2 minutes. Section 12 of the report lists the other stages and the analysis. They need the baseline run's circuit files, which are not in git. `run_all_on_node.sh` takes the interpreter from `LRB_PYTHON`.

## What we left out

The original folder was 22 MB. We left out:

- the raw per-circuit outputs (about 1,800 NPZ files in `.cache_correction/`, never in git);
- `notes/_superseded/`, `results/baseline_nonmarkov/_superseded_*/` and an earlier attempt of the A5 scripts;
- the reviewers' working files under `notes/review/` and `notes/report-review/`. We kept their Markdown summaries.
