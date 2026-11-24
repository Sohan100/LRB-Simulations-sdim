# Archived manuscript (2026-11-23)

On 2026-11-23 the project's main.tex and introduction.tex were replaced by the simulation version of the
manuscript, uploaded up to the end of the Introduction (title, authors, abstract, Introduction,
references). The remaining sections follow.

This folder keeps the two files of the previous manuscript that were replaced:

- `main.tex`: the previous main file, unchanged except that it now inputs the archived Introduction
  (`\input{archive/manuscript-2026-11-23/introduction}`) and starts with a four-line comment.
- `introduction.tex`: the previous Introduction, byte for byte.

Every other file of the previous manuscript (background, method, results, appendices, tikz preamble,
figures/, references.bib) is unchanged at its old path in the project root; the new main.tex does not
input them. To compile the previous manuscript, set `archive/manuscript-2026-11-23/main.tex` as the main
document (Menu -> Main document), and set `main.tex` back afterwards.

The one change to a shared file: in `references.bib` the entry `sdim` now lists the eight authors of
arXiv:2511.12777v2 (J. Keppens and B. Sorée added). The full previous state is also in the project's
history (Overleaf History, or the git commit before 2026-11-23).
