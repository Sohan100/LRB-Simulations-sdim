# Correction instead of detection: the qutrit [[5,1,2]]_3 code

**Questions.** Which decoders let LRB correct errors instead of rejecting shots, and how do they compare with LRB-D? Does correction mitigate non-Markovian decay? What is a sustainable way to run this on hardware?

**Answers.** The full write-up is [REPORT.md](REPORT.md). In short:

- Exact maximum-likelihood decoding (MLD) of the circuit-level DEM is cheap for this code: a forward filter over 243 states per check round. The decision rule matters more than the decoder.
- Full correction does not work at distance 2. At p = 1e-3 it is 3.1 times worse than physical RB, because 16 of the 40 single-qutrit errors are ambiguous.
- Selective rules trade accepted error for acceptance. Blip-forgiving detection more than doubles the acceptance at m = 22 and keeps the accepted error 70 times below physical RB. Soft-output MLD traces the optimal acceptance-fidelity frontier. LRB-D with a check every round sits on that frontier; sparser LRB-D policies fall below it.
- Correction brings the decay closer to a single exponential only by making the logical error first order, and some curvature remains. It does remove the bias that postselection picks up when the noise strength drifts from shot to shot.
- The protocol matters more than the decoder: read all five data qutrits at the end and read the ancillas in parallel. Genuine correction needs distance 3.

Scope: the study used the pre-sdim-1.4.0 pipeline. The baseline is the one-ancilla SI1000 run `Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000`, and physical RB is compared at p_RB = 5p. The newer runs use same-p RB and a different SI1000 schedule. Section 5 of the [non-Markovian study](../../non-markovian/REPORT.md) rechecks some of these claims and says which ratios we expect to change.

## Status: archived

This code targets commit `c21386f6` and does not run against the current `src/lrb`. We tried it:

- `lrb.dem_simulation.build_compact_detector_error_model` no longer exists. The DEM backend now wraps `sdim.dem.DetectorErrorModel` from sdim 1.4.0. Its mechanisms have a different form (each fires with some probability and then adds a uniform combination of its generators, zero included), and they do not carry the per-location gate data that `chain_dem.extract_mechanisms` reads.
- `LRBCircuitGenerator.apply_si1000_noise_to_circuit` no longer exists. More importantly, `variants.py` rebuilds the old circuit schedule (noise per logical gate, a separate ancilla reset round after each check). The current generator schedules each Clifford as one noisy segment and resets the ancillas inside each check round.

With a private copy of the old DEM builder, the five chain-model tests pass on the current package. The six variant tests still fail, because the rebuilt circuits differ from the generator's (the first difference is the reset placement). Making them pass would mean writing a new variant builder, and its circuits would no longer be the ones behind the report. So we keep the code as an archive.

## Contents

| Path | What it holds |
|---|---|
| `REPORT.md` | the study; Section 12 gives the commands |
| `bibliography_raw.csv` | every citation and its independent verification status |
| `notes/` | analysis notes A1 to A5, hardware notes H1 and H2, the baseline analysis N1, literature notes L1 to L8 with citation checks, the critiques, the code-review summaries, and the small scripts these notes cite |
| `results/` | decoder and variant tables (Markdown), curves and fits (`*.csv.gz`), RB reference, Pareto frontier, the first-order check, the baseline non-Markovianity tables, and the figures the report shows |
| `code/src/lrb/correction/` | the chain DEM, the exact forward filter, the decision rules, the sampler, the per-circuit study, the protocol variants |
| `code/scripts/correction/` | drivers, aggregation, tables, figures, the first-order check |
| `code/tests/` | `test_correction_chain.py`, `test_correction_variants.py` |
| `ruff.toml` | turns off the unused-name lint checks for this archive, so the code stays as it ran |

Paths in the report such as `src/lrb/correction/` and `scripts/correction/` refer to the original worktree. Here they sit under `code/`.

## How to run it

We rebuild the original layout at `c21386f6` and use Python 3.11 with sdim 1.3.4 (on Perlmutter, `module load python/3.11` provides it in the user site):

```bash
git worktree add ../lrb-c21386f6 c21386f6
cd ../lrb-c21386f6
STUDY=<this repository>/studies/correction-decoders/qutrit
cp -r "$STUDY/code/src/lrb/correction" src/lrb/
mkdir -p scripts && cp -r "$STUDY/code/scripts/correction" scripts/
cp "$STUDY"/code/tests/*.py tests/
module load python/3.11
python -m pytest -q tests/test_correction_chain.py tests/test_correction_variants.py
```

We checked this recipe on 2026-11-24: all 11 tests pass in about 45 s. The full studies then follow Section 12 of the report. They need the baseline run's circuit files, which are not in git. `run_all_on_node.sh` reads the run folder from `LRB_BASELINE_RUN`.

## What we left out

The original folder was 22 MB. We left out:

- the raw per-circuit outputs (about 1,800 NPZ files in `.cache_correction/`, never in git);
- `notes/_superseded/`, `results/baseline_nonmarkov/_superseded_*/` and an earlier attempt of the A5 scripts;
- the reviewers' working files under `notes/review/` and `notes/report-review/`. We kept their Markdown summaries.
