# Correction instead of detection: the qubit [[5,1,2]] code

**Questions.** The same three questions as the [qutrit study](../qutrit/README.md), for the qubit counterpart of the folded code (the distance-2 planar surface code on five data qubits). Which decoders let LRB correct instead of reject, and how do they compare with LRB-D? Does correction mitigate non-Markovian decay? What is a sustainable hardware protocol?

**Answers.** The full write-up is [REPORT.md](REPORT.md). In short:

- Full correction does not pay off at distance 2. Exact MLD over every check round leaves a first-order logical error of about 12 to 15p per depth step, 20 to 40 times worse than a bare qubit at the same p.
- The right decoder is exact MLD with soft-output acceptance (accept when |LLR| ≥ t). LRB-D with every check is the most conservative point of that family. Every sparser LRB-D policy is dominated.
- Removing checks beats correcting with them. A protocol with no mid-circuit checks that decodes only the final five-qubit readout reaches 0.74p per step under the repository's earlier SI1000 convention. That beats physical RB at 5p but not at p.
- Corrected LRB decays as a single exponential, because it does not postselect, not because it decodes. Sparse LRB-D policies are strongly non-Markovian.
- On hardware, read all five data qubits at the end, record every syndrome and decode offline in the Pauli frame. Genuine correction needs distance 3 with fault-tolerant extraction, parallel readout and p below about 2 to 3e-4.

REPORT.md is marked as a round-2 draft. Work packages F (non-Markovianity at dense depths), G (distance 3), I (erasure, leakage, flags) and K (Steane code) were pending when it was written. Their results are in `notes/` and `results/`.

## Status: integrated

The study's code runs against the consolidated package:

- `src/lrb_qubit/` is a separate package next to `src/lrb/`. It builds the qubit circuits in stim with its own SI1000 noise insertion. Work package A cross-checks them against sdim.
- `tests/test_lrb_qubit_model.py` and `tests/test_qubit512_*.py` hold its 538 tests. A few of them compare with `lrb` itself, for example the postselection policies and the plotter's fit.
- `scripts/` in this folder holds the study's drivers and verifiers. They find the repository root from their own location and put `src/` on `sys.path`.

The code needs three packages that are not in the base dependencies: stim (1.15), PyMatching (2.3) and ldpc (2.4), the `decoders` extra. Each test module skips cleanly when one is missing. With all three installed next to sdim 1.4.0, the 538 tests pass in about 2 minutes. The CI job installs the extra, so none of them skip there.

```bash
pip install -e ".[dev,decoders]"
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_lrb_qubit_model.py tests/test_qubit512_*.py
```

### sdim 1.4.0

`lrb_qubit` builds its circuits in stim and does not call sdim. It reproduces the repository's SI1000 schedule from before the sdim 1.4.0 port: noise after every logical gate, a check round, then a separate ancilla reset. The current generator schedules each segment into moments, with idle noise on every untouched wire, and resets the ancillas inside each check round. The study's numbers are therefore for the earlier schedule.

Work package A compares `lrb_qubit` with the sdim pipeline. We ported its two scripts to sdim 1.4.0 on 2026-11-24 (the frame sampler's noiseless reference shot, N2 gates without `prob_dist`, the renamed noise insertion) and reran the quick checks:

- `crosscheck_sdim.py selftest`: 9 of 9 pass, including the equality of the script's fast frame path with `Program.simulate`.
- `verify_crosscheck.py semantics`: all pass. `hazards` runs. Two sdim hazards it recorded are gone in sdim 1.4.0: the generator rejects gate aliases such as `S` or `CX`, and `DETECTOR` record placeholders no longer collide.
- `crosscheck_sdim.py dem`: only 6 of 108 detector error models now match exactly, the noiseless depth-0 ones. The mechanisms agree and the probabilities differ, which is the schedule change above. The results in `results/A/` are from the earlier pipeline, where all matched.

Moving `lrb_qubit` to the current schedule would change every number in the study, so we left it as it is.

## How to run the study

We run the scripts from the repository root. For example, the main grid and its summary (Section 13 of the report lists the rest):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python studies/correction-decoders/qubit/scripts/run_grid.py \
    --out studies/correction-decoders/qubit/results/grid_main \
    --shots 1000000 --decode-shots 300000 --workers 120 --rb-shots 1000000
python studies/correction-decoders/qubit/scripts/summarize_grid.py \
    --grid studies/correction-decoders/qubit/results/grid_main \
    --out studies/correction-decoders/qubit/results/summary
```

Each `notes/<X>-*.md` has a "Reproduce" section with the commands of its work package. We rewrote the paths in the report, the notes and the scripts from `docs/correction-decoders/qubit-512/` and `scripts/qubit512/` to this folder. `plot_main.py` and `plot_summary.py` take no arguments: they run on import and redraw `figures/main/` from `results/summary/` and `results/first_order/`.

## Contents

| Path | What it holds |
|---|---|
| `REPORT.md` | the study |
| `API.md`, `CONTEXT.md` | the briefs the study's agents worked from: the `lrb_qubit` API, the code, the noise model and the conventions |
| `notes/` | one note per work package (A to K, L0 to L2), with the independent verifier reports (`*.verify.md`) |
| `lit/` | the eight literature reviews and their citation checks |
| `brainstorm/SYNTHESIS.md` | the hypotheses and the verified bibliography from the brainstorm round |
| `results/` | tables per work package: CSV, JSON and Markdown up to 200 KB, and nine larger summary tables as `*.json.gz` or `*.csv.gz` |
| `figures/` | `main/` (figures 0 to 4 of the report and note L0), `E/` and `F/` (the figures of notes E and F) |
| `scripts/` | drivers, analyses, plots and verifiers |

## What we left out

The original folder was 135 MB in 6,847 files. We kept the documents, the figures and the tables. We left out:

- All NPZ files: about 76 MB of sampled shots, exact curves and per-point outputs.
- Per-point table folders (`csv/`, `parts*/`, `runs/`, `raw/`, `chunks/`), other compressed raw outputs, logs and `.done` markers.
- Other tables over 200 KB, mostly row-level dumps such as `H-verify/census_rows.csv` (5.5 MB).
- `F/analysis.json` and `H/summary.json`. Their numbers are in `F/note_tables.md` and `H/tables.md`.
- The six brainstorm lens files. `SYNTHESIS.md` consolidates them.
- `figures/E/pareto_all.pdf`, the Pareto plots at every p, written by `scripts/pareto.py`. The files `pareto_p{4,5,6,8,11}.png` keep five of them.

The scripts write uncompressed tables. To reuse a compressed one as input, run `gunzip -k` on it first.
