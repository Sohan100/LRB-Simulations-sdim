# Shared brief: correction instead of detection for qutrit LRB-D

Read this whole file before starting. It is the shared context for every
agent working on this study. Facts marked [code] were read from the source in
this worktree; facts marked [data] were read from the baseline run folder.

## Goal

The repository simulates detection-only logical randomized benchmarking
(LRB-D) of the `[[5,1,2]]_3` folded qutrit surface code with `sdim`. Any shot
whose used stabilizer checks show a non-trivial syndrome is rejected. The
study asks:

1. Which decoders let this LRB experiment use error *correction* instead of
   detection/postselection, and how do they compare with the LRB-D baseline
   (logical fidelity, decay rate, acceptance vs p and check policy)?
2. Does correction mitigate non-Markovianity in LRB, or not, and why?
3. What is a sustainable way to do this on real hardware (superconducting
   qutrits are the likely target; IQM export existed once in git history;
   also trapped ions, photonics, Rydberg, bosonic qudits)?

Every claim must be labelled: **verified analytically**, **verified
numerically**, **from literature** (with a citation you actually fetched), or
**conjecture**.

## Worktree and rules

- Worktree root: `/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/correction-decoders-qutrit-2c455b`
- Notes folder (write your output here): `studies/correction-decoders/qutrit/notes/`
- Python: run `module load python/3.11` first (Python 3.11.7 with sdim, stim,
  pymatching, ldpc, sinter, numpy, scipy, networkx). Plain `python3` is 3.6
  and has none of them. Never `pip install` into the user's site-packages; if
  you need a package, make a venv inside the worktree under `.venv-<name>/`.
- This is a NERSC Perlmutter **login node**: keep any computation to a few
  minutes of one or a few cores. No `sbatch`.
- Read-only: the main checkout `/pscratch/sd/s/sohan100/LRB-Simulations-sdim/`
  and everything under it, including its run folders and its
  `LRB-experiment-data-slurm/working-folder*.txt`. Never run
  `scripts/generate_circuits_folded.py` (it overwrites pointer files).
- Do not commit, push, or edit files outside the notes folder unless your task
  explicitly says so.
- Citations: every paper you cite must have been fetched by you (arXiv abstract
  page, publisher page, or DOI landing page) in this task. Record title, full
  author list (or first three + "et al."), year, arXiv ID and/or DOI, and the
  URL you fetched. Never cite from memory alone. If you cannot confirm a paper
  exists, say so and drop it.

## The code [code: src/lrb/code_definitions.py, FoldedQutritDetectionCode]

- Qutrits, d = 3. Data wires 0-4, ancillas 5-8 (one ancilla per stabilizer).
- Stabilizers (exponent signs follow the circuit's CNOT/CNOT_INV):
  - `S_Z1 = Z0 Z1^-1 Z3^-1` (ancilla 5), `S_Z2 = Z1 Z2 Z4^-1` (ancilla 6)
  - `S_X1 = X0 X1 X2^-1` (ancilla 7), `S_X2 = X1^-1 X3 X4^-1` (ancilla 8)
  - Terminal X-form: `x0 + x1 - x2`, `-x1 + x3 - x4` (mod 3).
- Logical X = X on {0,3} (X0 X3). Logical Z = Z on {0,2} (Z0 Z2).
- Logical H = SWAP(2,3), then H on {0,2,3,4}, then H_INV on 1.
- Logical P = P on {0,4}, P_INV on 1, CZ(2,3). X/Z logicals are transversal
  Paulis on {0,3}/{0,2}. `_INV` tokens reverse and invert the sequence.
- Distance 2: e.g. X0 and X3^-1 have the same syndrome and differ by logical X.
- Z-checks (used profile: `z_ALT_stabilizer_measurement_ancillae_circuit`):
  `CNOT 0->5, CNOT_INV 1->5, CNOT_INV 3->5, M 5` then
  `CNOT 1->6, CNOT 2->6, CNOT_INV 4->6, M 6`.
- X-checks: `H 7, CNOT 7->0, CNOT 7->1, CNOT_INV 7->2, H_INV 7, M 7` then
  `H 8, CNOT 8->3, CNOT_INV 8->1, CNOT_INV 8->4, H_INV 8, M 8`.
  Ancillas are measured one at a time (four separate measurement layers),
  then all four are reset in one reset layer.
- Encoded |+>_L preparation is ideal (noiseless).
- Terminal readout (ordinary circuits): `H_INV` + `M` on data wires 0 and 3
  only, **noiseless**; logical outcome `x0 + x3 mod 3`.
- `const=0` circuits: no ancilla checks at all; terminal noiseless X-basis
  readout of all five data wires, giving the two X-stabilizer values as
  detectors plus the logical outcome.

## Circuit structure [code: src/lrb/circuit_generator.py, generate_lrb_clifford_sequence]

For depth m (m random single-qutrit Cliffords from the 216-element catalog,
each a string of logical H, P, X, Z tokens):

1. ideal encoding of |+>_L;
2. for each of the m forward Cliffords: its logical gates (noisy), then a full
   noisy stabilizer-check round (Z-block then X-block), then a noisy ancilla
   reset layer. **Every forward Clifford is followed by a physical check round
   in every circuit, whatever the postselection policy.**
3. the m inverse Cliffords (each forward Clifford inverted, in reverse order),
   noisy, **with no checks between them**;
4. one final noisy check round (round index m);
5. ideal terminal readout of wires 0 and 3.

So a depth-m circuit has 2m noisy Clifford blocks, m+1 check rounds (rounds
0..m-1 after forward Cliffords, round m after the inverse block), and
4(m+1) detectors labelled `lrb_stab_r{round}_w{wire}`. Each detector is the
raw ancilla outcome of that round (not a difference between rounds). One
logical observable `lrb_logical` = shift of x0 + x3.

## Noise model: generalized SI1000 [code: apply_si1000_noise_to_circuit]

Swept parameter p. Locations:
- after every 1-qutrit gate: single-qutrit depolarizing with prob 0.1 p
  (`N1 d`, uniformly one of the 8 non-identity Weyl operators X^a Z^b);
- after every 2-qutrit gate (CNOT, CZ, SWAP and inverses): two-qutrit
  depolarizing with prob p (uniform over the 80 non-identity pairs);
- each measurement layer (here one ancilla per layer): idle depolarizing 2p on
  every other wire, depolarizing p on the measured ancilla, plus a
  measurement-result shift (`N1 f`, X^a with a in {1,2}) with prob 5p;
- reset layer (all four ancillas): idle depolarizing 2p on the data wires,
  then an X^a shift with prob 2p on each reset ancilla.
- No idle noise during gate layers. State prep and terminal readout ideal.

Sequential measurement layers mean each data qutrit picks up 4 x 2p idle
depolarizing per check round plus 2p in the reset layer.

## Postselection policies [code: lrb_simulation.py postselect_*]

The policies only choose which already-executed check rounds are used:
- The terminal round m is always used.
- `unif k`: also rounds j*k - 1 for j = 1..floor(m/k) (rounds after forward
  Cliffords k, 2k, ...). `unif 1` uses every round.
- `const c` (c >= 1): if c <= m, also c-1 intermediate rounds at
  j*(m//c) - 1, j = 1..c-1; if c > m, terminal only. `const 1` = terminal only.
- `const 0` uses the separate const0 circuits (no ancilla checks; ideal
  final X-stabilizer check from data).

## Fidelity metric and fit [code: lrb_simulation.py, lrb_plotting.py]

- Per circuit: fidelity = |sum_k omega^k P(k)| over accepted shots, where k is
  the logical outcome and omega = e^{2 pi i/3}. Averaged over 30 random
  Clifford sequences.
- LRB fit: y = b * f^m with the offset fixed at 0, weighted grid scan.
  RB fit: y = a + b f^m, free offset. F = (1 + 2f)/3, r = 1 - F.
- Physical RB comparison uses p_RB = 5p (one qutrit, SI1000 single-qutrit gate
  noise, same mirrored forward/inverse structure, 2m Cliffords).

## Simulation backend [code: src/lrb/dem_simulation.py]

`CompactDetectorErrorModel` compiles an annotated sdim circuit into a list of
independent noise locations. Each location has an activation probability and
a uniform distribution over non-zero Weyl vectors (2 components for N1, 4 for
N2), plus, for each unit component, the sparse Z_3 shift it causes on every
detector and on the logical observable. Sampling is exact for this Pauli
noise model (linear Pauli-frame propagation). This is the natural input for
decoders: a decoder maps the detector-shift history of one shot to a
predicted logical shift (Pauli-frame correction of the terminal readout) and
optionally a reject flag.

## Baseline LRB-D data [data]

Run folder (read-only):
`/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000/`
- 30 Clifford sequences, depths 0,2,4,6,10,14,18,20,22, 1M shots each,
  15 probabilities p in {0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 0.01,
  0.0125, 0.015, 0.02, 0.03, 0.04, 0.05}.
- `results/LRB/<p_idx>/unif_check_data/<k>.csv`,
  `results/LRB/<p_idx>/const_check_data/<c>.csv`: 5 rows (Probability;
  Fidelity averages per depth; Fidelity std over Cliffords; Rejected Runs per
  depth; Rejected std).
- `results/RB/<p_idx>.csv`, `results/RB_comparison/<p_idx>.csv` (p_RB = 5p).
- `results/plots/unif_lrb_vs_rb_table_all_mixed_fits.csv` (fit table).
- Circuits: `experiments/LRB/<cliff>/<p_idx>/<depth_idx>.chp`,
  `experiments/LRB_const0/...`, `experiments/RB/...` (readable with
  `sdim.circuit_io.read_circuit`).
- Example, p = 1e-3: `unif 1` fidelity 0.9925 at m=0 to 0.9919 at m=22 with
  rejection 6.1% to 91.9%; `const 1` (terminal only) 0.9925 to 0.9666 with
  rejection 79% at m=22; `const 0` 0.99999 to 0.9989 with rejection 16%.
  Physical RB at p_RB = 5e-3: 0.989 at m=2 to 0.883 at m=22.
