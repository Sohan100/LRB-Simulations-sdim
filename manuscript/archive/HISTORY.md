# Overleaf history

The Overleaf project has 24 commits. This table lists all of them, oldest first. Dates are as the
Overleaf log shows them; the two "Update on Overleaf." commits carry UTC times, the others Pacific
time. The last column says where in `manuscript/` that state is kept.

| Commit | Date | Message | What changed | Kept in |
|---|---|---|---|---|
| `fa43b79` | 2026-05-28 | Update on Overleaf. | First version of the paper on Overleaf. File names use colons (`bg: RB.tex`). | `2026-05-28-first-overleaf-draft` (as `eda26bb`) |
| `eda26bb` | 2026-08-11 | Update results.tex via Overleaf MCP | Renames the files from `bg: X.tex` to `bg - X.tex` and updates the `\input` lines. No text change. | `2026-05-28-first-overleaf-draft` |
| `9bcd49c` | 2026-08-11 | Update main.tex via Overleaf MCP | Adds the SI1000 section, noise models, sdim DEM and Clifford group appendices, SI1000 figures. | not kept |
| `3cdeb51` | 2026-08-11 | Update si1000 sdim simulation.tex via Overleaf MCP | Rewords four sentences of the SI1000 section. | not kept |
| `ab22def` | 2026-08-11 | Update main.tex via Overleaf MCP | Moves the code figures into TikZ files and adds `tikz preamble.tex`. | not kept |
| `9f4d0d9` | 2026-08-11 | Update main.tex via Overleaf MCP | Changes the colors of the surface-code figure. | not kept |
| `8cdcbd6` | 2026-08-12 | Update main.tex via Overleaf MCP | Adds Chuan-Hong Liu as an author. | not kept |
| `a6cfdb0` | 2026-08-12 | Update method - LRB-D.tex via Overleaf MCP | Fixes two gate labels in the LRB-D circuit. | not kept |
| `f6d2510` | 2026-08-12 | Update fig - 5-qutrit code.tex via Overleaf MCP | Fixes one label in the code figure. | not kept |
| `97f086b` | 2026-09-12 | Update on Overleaf. | Adds the appendix on baseband two-qubit and two-qutrit gates. | `2026-09-12-pre-rework-original` |
| `b3429b5` | 2026-10-30 | Rework simulation results onto the sdim 1.4.0 SI1000/SD6 runs | New results section on circuit-level noise (SI1000, SD6), one- and two-ancilla layouts, new figures. | not kept |
| `5b87c6d` | 2026-10-31 | Fix missing citations and the irrep multiplicity m_triv | Adds the missing bibliography entries and fixes the irrep multiplicity. | not kept |
| `b82b201` | 2026-10-31 | Remove alternative-estimator analyses from the results | Removes the alternative-estimator analyses from the results. | not kept |
| `b8c0c5f` | 2026-10-31 | Describe SI1000 decays as single-exponential for k = 1 to 4 | Rewords a few sentences. | not kept |
| `1f333b6` | 2026-10-31 | Quote pseudo-thresholds with one standard error from a sequence bootstrap | Adds bootstrap standard errors to the pseudo-thresholds, updates one figure. | `2026-10-31-si1000-sd6-rework` |
| `00f46fa` | 2026-11-02 | Two-ancilla layout on the conventional surface-code sites | Moves the two-ancilla layout to the conventional surface-code sites, adds the layout figure. | not kept |
| `c169070` | 2026-11-02 | Add the qubit [[5,1,2]] comparison | Adds the qubit comparison and two figures. | not kept |
| `438f6f1` | 2026-11-08 | Remove the abstract until the experimental results are added | Comments out the abstract. | `2026-11-08-before-restructure` |
| `9929f74` | 2026-11-23 | Update archive/manuscript-2026-11-23/introduction.tex via Overleaf MCP | Step of the 2026-11-23 upload. | not kept |
| `3d4a785` | 2026-11-23 | Update archive/manuscript-2026-11-23/main.tex via Overleaf MCP | Step of the 2026-11-23 upload. | not kept |
| `5ec3cce` | 2026-11-23 | Update main.tex via Overleaf MCP | Step of the 2026-11-23 upload. | not kept |
| `ae1c498` | 2026-11-23 | Update introduction.tex via Overleaf MCP | Step of the 2026-11-23 upload. | not kept |
| `e819e40` | 2026-11-23 | Update references.bib via Overleaf MCP | Lists all eight authors of the sdim paper. | not kept |
| `53713f5` | 2026-11-23 | Update archive/manuscript-2026-11-23/README.md via Overleaf MCP | Current state. | `../overleaf/` |

## Why we keep these four states

- `2026-05-28-first-overleaf-draft` (`eda26bb`, same text as `fa43b79`). The paper as first put on
  Overleaf: Background, the code, LRB-D and the first sdim simulation under depolarizing noise. It
  predates the SI1000 section, the TikZ figures and the appendices on noise models and the sdim
  detector error model. We store `eda26bb` and not `fa43b79` because the two differ only in file
  names, and colons in file names break checkouts on some systems.
- `2026-09-12-pre-rework-original` (`97f086b`). The paper before any of the rework of October and
  November. Every later folder is measured against it.
- `2026-10-31-si1000-sd6-rework` (`1f333b6`). The paper after the results moved to the sdim 1.4.0
  SI1000 and SD6 runs, with the one-ancilla layout and the first two-ancilla layout (a relay ancilla
  for each boundary stabilizer). It is the last state before the conventional-site two-ancilla
  layout and the qubit comparison.
- `2026-11-08-before-restructure` (`438f6f1`). The last state before the restructure into the
  simulation and hardware versions. It contains the conventional-site two-ancilla layout and the
  qubit comparison.

## Why we drop the others

- `9bcd49c` to `f6d2510` (2026-08-11 and 2026-08-12) are working steps between the first draft and
  `97f086b`. `97f086b` contains all of them plus the baseband-gate appendix.
- `b3429b5` to `b8c0c5f` (2026-10-30 and 2026-10-31) are the first rework draft and the citation
  fixes, cut and rewording that led to `1f333b6`. Only `b3429b5` and `5b87c6d` still hold the
  alternative-estimator analyses that `b82b201` removed.
- `00f46fa` and `c169070` are the two steps between `1f333b6` and `438f6f1`. `c169070` differs from
  `438f6f1` only in that its abstract is not commented out.
- The six commits of 2026-11-23 upload the simulation version's front matter in steps. Their end
  state is `53713f5`, kept in `../overleaf/`. There the previous manuscript still compiles from
  `archive/manuscript-2026-11-23/main.tex`. It equals `438f6f1` except for the author list of the
  sdim entry in `references.bib`.

Any dropped state can be restored from the Overleaf history (Menu, History) or from a git-bridge
clone with `git archive <commit>`.

## What each archive folder contains

Each folder holds the files its `main.tex` reads, byte for byte as in that Overleaf commit: the
`.tex` files, `references.bib` and the figures the text includes. We left out the files the build
does not read (`extra.tex`, unused figures, old section files). Each of them is in `../overleaf/`
unchanged. To compile a folder, see `../README.md`.
