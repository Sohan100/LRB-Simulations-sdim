# Which LRB-D circuits fit IQM Garnet without SWAP routing (2026-10-31)

Garnet coupler graph: 20 qubits, 30 CZ pairs, from IQM's qiskit-on-iqm fake backend
(`src/iqm/qiskit_iqm/fake_backends/fake_garnet.py`, `qubit_connectivity`, branch main).
It is a square-lattice graph (arXiv:2408.12433), so it is bipartite (no odd cycles), and the
maximum degree is 4.

Rule: every two-qutrit gate of the simulated circuits (`src/lrb/code_definitions.py`, used by
`src/lrb/code_simulation_profiles.py`) must act on a coupled pair. The logical gates need
q2-q3 (CZ_{2->3} in S, SWAP_{2,3} in H). Test: subgraph monomorphism with networkx
(`embed_check.py`, `relay_chains.py`, `sanity.py`).

| circuit | fits Garnet? | reason |
|---|---|---|
| terminal check (const-0): encoder + q2-q3 | yes, 128 placements | 4-cycle q0-q1-q2-q3 plus q4 on q2 |
| one-ancilla checks + q2-q3 | no | 5-cycle aG1-q0-aG3-q2-q3 |
| one-ancilla checks alone | yes, 32 placements | standard d=2 patch, but q2 and q3 end 4 couplers apart |
| two-ancilla checks as simulated (alone or + q2-q3) | no | 5-cycle aG1-q1-aG4-q3-rG1 inside the check round |
| any relay layout (1-3 relays per stabilizer, any relayed qutrit) + q2-q3 | no (0 of 286 on a 13x13 lattice; 0 on Garnet) | |
| same relay layouts without q2-q3 | 283 fit Garnet | (sanity check of the search) |

Parity argument: on a bipartite graph a one-ancilla stabilizer puts its three data qutrits on
one sublattice. G1 and G3 (or any Z/X pair not both containing q2) then put q2 and q3 on the
same sublattice, but the q2-q3 coupler needs them on opposite ones. In the simulated relay
layout the G4 syndrome ancilla couples to q1 and q3, which G1's relay chain puts on opposite
sublattices.

Relays carrying one or two data qutrits (single relay per stabilizer): 0 of 72 bipartite
layouts fit Garnet (`relay_subsets.py`).

Authors' intended two-ancilla layout (user, 2026-11-01): the conventional surface-code grid
with q2 at the centre (q2=QB9, q0=QB3, q1=QB5, q3=QB13, q4=QB15; ancilla sites N=QB4, W=QB8,
E=QB10, S=QB14). G3 sits on N and G2 on E; G1 and G4 each need two sites (e.g. N+W, E+S).
No two sites share a coupler, so the direct ancilla-ancilla gates of the simulated relay
(CNOT 9->5, CNOT_INV 8->10) have no coupler; the simulated G4 split pairs q1 with q3, which no
site touches; the simulation uses six dedicated ancillas instead of four sites; q2-q3 is two
couplers apart. With q1 at the centre instead, every stabilizer is one site (the one-ancilla
layout, which fits), but q2-q3 is four couplers apart.

Not covered: time-multiplexed ancillas, SWAP routing of the q2-q3 gate, passing a partial
syndrome between two sites through a shared data qutrit (possible in principle with data-ancilla
gates only, not simulated).

Figure: `figure/garnet-connectivity.{tex,pdf}` with `figure/caption.tex`, reviewed in six
verifier/corrector rounds (`figure/reviews/round-1..6.md`, last round no findings).
