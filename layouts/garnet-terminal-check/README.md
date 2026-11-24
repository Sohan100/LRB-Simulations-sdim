# Terminal check on IQM Garnet

Panel (a) of `../figures/three-layouts.pdf`. The five data qutrits sit on five of Garnet's 20
transmons and there are no ancillas. The only check is the final one: every data qutrit is read
in the X basis, which checks `G3` and `G4` and gives the logical outcome. In the LRB-D runs this is
the `const-0` family (no mid-circuit checks), present in every run folder.

- **Couplings used:** 5 of Garnet's 30. Four carry the encoder (`q0-q1`, `q1-q2`, `q0-q3`, `q2-q4`)
  and `q2-q3` carries the two-qutrit gates of the logical S and H.
- **Encoder:** 4 two-qutrit gates. The paper's earlier four-gate circuit, which the one-ancilla runs
  use, needs exactly these four couplings, so it runs natively here.
- **Why the one-ancilla layout does not fit:** Garnet's coupler graph is a square lattice and
  therefore bipartite. A one-ancilla check puts its three data qutrits on one sublattice, and two
  checks together then put `q2` and `q3` on the same sublattice, so the `q2-q3` coupler cannot exist
  alongside them. The relay layouts fail too. `connectivity/FINDINGS.md` (2026-10-31) has the
  argument and the placement counts. The two-ancilla layout came later and was built for a square
  lattice: it reads checks through data gates and fits Garnet in 48 placements.

## Files

| file | what |
|---|---|
| `connectivity/FINDINGS.md` | which LRB-D circuits fit Garnet without SWAP routing |
| `connectivity/embed_check.py`, `relay_chains.py`, `relay_subsets.py`, `sanity.py` | the subgraph-embedding checks behind it (networkx) |
| `connectivity/garnet-connectivity.pdf`, `.tex`, `caption.tex` | the figure |

The coupler list comes from IQM's public `fake_garnet` backend in `qiskit-on-iqm`. The encoder and
terminal readout circuits are drawn on page 2 of `../figures/layouts-and-circuits-overview.pdf`.
