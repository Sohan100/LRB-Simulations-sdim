# What is the paper's main message?

Opened 2026-11-24 by Sohan, from the meeting with Chuanhong
(`../meetings/2026-11-24-chuanhong.md`). Status: open. This is the most urgent question before we
aim at a high-impact journal.

## The question

The analysis alone is not enough; we need one strong, concrete message. Three candidates came up:

1. **Application.** A concrete task where qutrit error correction beats qubit error correction even
   with worse physical gates.
2. **Layout.** Our hardware layouts are more practical and newer than earlier proposals.
3. **Noise framework.** The first systematic treatment of circuit-level noise for qutrit error
   correction.

## What the repo already shows for each

**1. Qutrit against qubit.** Pseudo-thresholds `p*` (two-qudit gate error at which LRB-D matches
physical RB, check after every Clifford, one standard error), from
`LRB-experiment-data-slurm/comparisons/pseudo-thresholds-*-current.md`:

| layout | noise | qutrit | qubit | ratio |
|---|---|---|---|---|
| one-ancilla, five-moment round | SI1000 | 0.0313(6) | 0.0123(4) | 2.6 |
| one-ancilla, five-moment round | SD6 | 0.0501(26) | 0.0290(10) | 1.7 |
| two-ancilla | SI1000 | 0.0191(4) | 0.00511(15) | 3.7 |
| two-ancilla | SD6 | 0.0206(4) | 0.00854(18) | 2.4 |

This is the break-even framing from the meeting: about 3% for the qutrit code against about 1% for
the qubit code. The reviewer's answer is that qubit gates are already far below 1%, so both codes
work and the easier gate wins. A clean application has not come up yet; Chuanhong's gate-drift idea
(`gate-drift-and-uneven-fidelity.md`) is one way to look for a regime where the qutrit code keeps
working and the qubit code does not.

**2. Layout.** `layouts/` has the three placements of the paper (terminal check on IQM Garnet,
one-ancilla, two-ancilla) and the folded surface code at any distance (the Dorito chip). On
Garnet's square lattice the one-ancilla and relay layouts do not fit, because the logical gates need
the `q2-q3` coupler (`layouts/garnet-terminal-check/connectivity/FINDINGS.md`); the two-ancilla layout
was designed for a square lattice and fits Garnet in 48 placements. The Dorito chip keeps
every coupler nearest or next-nearest neighbour for every distance, where Moussa's folding needs
extra swaps or long-range couplers. As a memory it matches the rotated surface code's threshold
under SI1000 and SD6 and needs 10 to 30% more qubits, in exchange for a free logical H and a logical S
that costs about half a round (`layouts/folded-surface-code/memory/README.md`). That memory study is
for qubits; the qutrit version is not run yet.

**3. Noise framework.** SI1000 and SD6 generalized to qudits, with an unbiased depolarizing channel
over all two-qutrit Paulis and moment-by-moment idle noise (`docs/simulation-model.md`). With no
qutrit device to fit a bias to, the unbiased model is the defensible baseline. Chuanhong thinks this
can stand as the message if we frame it as a methodological contribution.

## Replies
