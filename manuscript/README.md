# Manuscript

This folder holds the paper "Logical Randomized Benchmarking on a Qutrit Surface Code": the copy on
Overleaf, the two versions we are writing, and the earlier states of the Overleaf project.

| Folder | Contents | Date |
|---|---|---|
| `simulation/` | The simulation paper. This is the current version. | 2026-11-23 |
| `hardware/` | The version that waits for the hardware results. | 2026-11-13 |
| `overleaf/` | Exact copy of the Overleaf project at commit `53713f5`. | 2026-11-23 |
| `archive/` | Four earlier Overleaf states, and `HISTORY.md` with all 24 Overleaf commits. | 2026-05-28 to 2026-11-08 |

## Which version is current

On 2026-11-13 we split the paper into two versions for PRX Quantum. They are meant to share the
Methods, the Results and Appendices A to G word for word.

- `simulation/` is the paper we write if it reports simulations only. It is the newest version and
  uses REVTeX (`prxquantum`). Its title, abstract and Introduction went to Overleaf on 2026-11-23.
  The sections after the Introduction exist only here. The section "Circuit-Level Noise Simulations"
  is interim. It still describes the seven-moment round and the previous conventional-site
  two-ancilla layout, until we rewrite it on the runs with the five-moment round and the two-ancilla
  layout.
- `hardware/` is the paper we write if the hardware results arrive. It uses the `article` class. It
  adds a section "Hardware Implementation" with placeholders and the appendix on baseband two-qubit
  and two-qutrit gates. It has no abstract. Its Introduction and Results are still those of
  2026-11-08, until the simulation version hands over the new ones.
- `overleaf/` is the shared copy that we edit with our co-authors. It is read-only from this
  repository: we never edit files in it here. We change the paper on Overleaf and then refresh this
  copy. Its root `main.tex` builds the simulation version up to the end of the Introduction (2
  pages). `overleaf/archive/manuscript-2026-11-23/main.tex` builds the previous manuscript (58 pages)
  from the old files at the project root. That `archive/` folder belongs to the Overleaf project and
  is not the same as `manuscript/archive/`. Only `main.pdf` is ours; it is our build of the root
  `main.tex` and is not on Overleaf.

To refresh `overleaf/` from a git-bridge clone of the Overleaf project, we empty the folder, run
`git -C <clone> archive HEAD | tar -x -C manuscript/overleaf`, rebuild `main.pdf` and note the new
commit here.

## Old versions

`archive/HISTORY.md` lists every Overleaf commit and says which folder holds each state:

- `2026-05-28-first-overleaf-draft`: the first version on Overleaf.
- `2026-09-12-pre-rework-original`: the paper before the rework of October and November.
- `2026-10-31-si1000-sd6-rework`: the results on the SI1000 and SD6 runs of sdim 1.4.0.
- `2026-11-08-before-restructure`: the last state before the split into two versions.

`hardware/` and each archive folder hold only the files their `main.tex` reads: the `.tex` files,
`references.bib` and the included figures. `simulation/` holds those plus three drafts its build does
not read yet (`method - LRB-D.tex`, `method - conventions.tex`, `results.tex`). The files no build
reads, such as `extra.tex` and unused figures, are in `overleaf/`.

## How to build

Every version builds with TeX Live 2024 on Perlmutter. The BibTeX styles (`apsrev4-2`, `unsrt`) come
with TeX Live. File names contain spaces, as on Overleaf. We build in a scratch copy so that no
build files land in the repository:

```bash
module load texlive/2024
tmp=$(mktemp -d) && cp -r manuscript/simulation/. "$tmp" && cd "$tmp"
latexmk -pdf -interaction=nonstopmode main.tex
```

For the previous manuscript inside `overleaf/`, run the same `latexmk` line with
`archive/manuscript-2026-11-23/main.tex` from the top of the copy.

## Build status (2026-11-24)

| Version | Result |
|---|---|
| `overleaf/main.tex` | Compiles, 2 pages, no errors. |
| `overleaf/archive/manuscript-2026-11-23/main.tex` | Compiles, 58 pages, no errors. |
| `simulation/` | Compiles, 49 pages, no errors, no undefined references. LaTeX warns that some floats are stuck and deferred (in `sec - lrbd.tex` and the appendices). No float is lost. |
| `hardware/` | Compiles, 54 pages, no errors, no undefined references. |
| `archive/2026-05-28-first-overleaf-draft` | Compiles, 27 pages. The label `tab:clifford-conjugation` is defined twice. |
| `archive/2026-09-12-pre-rework-original` | Compiles, 42 pages. The citations `Marxer2023` and `Luo2023` are undefined; their entries reached `references.bib` on 2026-10-31. |
| `archive/2026-10-31-si1000-sd6-rework` | Compiles, 53 pages, no errors, no undefined references. |
| `archive/2026-11-08-before-restructure` | Compiles, 58 pages, no errors, no undefined references. |
