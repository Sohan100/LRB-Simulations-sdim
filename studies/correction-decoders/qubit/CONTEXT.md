# Shared context: correction vs detection for LRB on the qubit [[5,1,2]] code

This file briefs every agent working on this study. Read it fully before
starting. Paths are relative to the repository root.

## The question

The repo simulates detection-only logical randomized benchmarking (LRB-D) on
the qutrit `[[5,1,2]]_3` folded surface code. Any shot with a non-trivial
syndrome in the checks that the policy uses is rejected. We study the qubit
counterpart, the qubit `[[5,1,2]]` code, and ask:

1. Which decoders let LRB use error *correction* instead of detection, and how
   well do they do against the LRB-D postselection baseline (logical fidelity,
   fitted decay rate, acceptance, as functions of p and check policy)?
2. Does correction mitigate non-Markovianity in the LRB protocol, and why?
3. What is a sustainable way to do this on real hardware (superconducting,
   e.g. IQM, plus trapped ions, neutral atoms, others)?

"The qubit code" is interpreted as the qubit counterpart of the qutrit code the
repo already simulates (the d=2 unrotated/planar surface code, which is
symmetric under a diagonal fold).

## The code (qubits)

- Data qubits 0-4; ancillas 5, 6 (Z checks) and 7, 8 (X checks), one ancilla
  per stabilizer.
- Stabilizers: `S_Z1 = Z0 Z1 Z3`, `S_Z2 = Z1 Z2 Z4`, `S_X1 = X0 X1 X2`,
  `S_X2 = X1 X3 X4`.
- Logical operators: `X_L = X0 X3`, `Z_L = Z0 Z2`.
  Other weight-2 representatives (derived, to be verified):
  `X_L ~ X2 X4`, `Z_L ~ Z3 Z4`.
- Fold-transversal logical H = `SWAP(2,3)` then H on all five data qubits.
- Logical S = S on {0,4}, S-dagger on 1, CZ(2,3).
- Logical X = X on {0,3}; logical Z = Z on {0,2}.
- Ideal (noiseless) preparation of |+>_L. Terminal logical-X readout.
- Single-qubit error syndromes (derived analytically, to be verified):
  X errors (seen by Z checks): X0 and X3 -> (1,0), X2 and X4 -> (0,1),
  X1 -> (1,1). Z errors (seen by X checks): Z0 and Z2 -> (1,0),
  Z3 and Z4 -> (0,1), Z1 -> (1,1). Each ambiguous pair differs by a weight-2
  logical, so distance 2 means correction can only guess within each pair.
  Among single-qubit errors, those on qubit 1 have unique syndromes. At the
  circuit level they do not: two-qubit noise after the fold gates SWAP(2,3)
  and CZ(2,3) produces X2X3, Z2Z3 and Y2Y3, which have the same syndromes as
  X1, Z1 and Y1 but differ from them by a logical (correction found by work
  package E).

## The protocol being mirrored (qutrit production run, read-only)

Run folder (read-only, never modify):
`/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000/`
Grid: SI1000 noise, 30 random Clifford sequences, depths m = 0,2,4,6,10,14,18,20,22,
1,000,000 shots per circuit, 15 probabilities
p = 0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 1e-2, 1.25e-2, 1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2,
check policies const 0..22 and unif 1..22.

Circuit structure for depth m (code in `src/lrb/circuit_generator.py`,
`generate_lrb_clifford_sequence`):

1. Ideal encoding of |+>_L.
2. For j = 1..m: logical Clifford C_j (a short word in H, S, X, Z), then a FULL
   noisy stabilizer-check round, then reset of all four ancillas.
3. The inverses C_m^-1 ... C_1^-1, each Clifford inverted individually (mirror
   style, not one compiled recovery Clifford), with NO checks in between.
4. One final noisy check round.
5. Ideal terminal readout: H on data 0 and 3, measure, logical outcome is the
   parity x0 + x3.

Important facts that are easy to miss:

- Every circuit physically contains all m+1 check rounds, whatever the policy.
  The `const`/`unif` policies only choose which rounds' outcomes are used for
  postselection (`postselect_constant_number`, `postselect_uniform_interval` in
  `src/lrb/lrb_simulation.py`). So all check noise is always present.
  - `unif=k`: use the terminal check plus forward checks at indices
    j*k - 1 for j = 1..floor(m/k) (index < m).
  - `const=k` (k >= 1): use the terminal check; if k <= m, also forward checks at
    (j*floor(m/k)) - 1 for j = 1..k-1. If k > m only the terminal check is used.
  - `const=0` is a different circuit: no check rounds at all, and an ideal
    terminal X-basis readout of all five data qubits. It rejects if either X
    stabilizer reads non-trivially; the logical outcome is still x0 + x3.
- The inverse half has no intermediate checks, so errors there are only seen
  by the final check.
- Check circuits (qubit version of the `folded_qutrit` profile; Z checks use the
  CNOT data->ancilla variant, X checks use H on ancilla, CNOT ancilla->data, H):
  - Z checks: CNOT 0->5, CNOT 1->5, CNOT 3->5, M 5; CNOT 1->6, CNOT 2->6, CNOT 4->6, M 6.
  - X checks: H 7, CNOT 7->0, CNOT 7->1, CNOT 7->2, H 7, M 7;
    H 8, CNOT 8->3, CNOT 8->1, CNOT 8->4, H 8, M 8.
  - Then reset 5, 6, 7, 8.
- Generalized SI1000 noise as implemented in the repo
  (`apply_si1000_noise_to_circuit`), translated to qubits:
  - after every 1-qubit gate (including the identity Clifford, which is an `I`
    on data 0): DEPOLARIZE1(0.1 p);
  - after every 2-qubit gate (CNOT, CZ, SWAP): DEPOLARIZE2(p);
  - every measurement is its own layer here: DEPOLARIZE1(2p) on every other
    qubit (data and ancillas), and on the measured qubit DEPOLARIZE1(p) then
    X_ERROR(5p) before measuring;
  - reset layer (all four ancillas at once): DEPOLARIZE1(2p) on every other
    qubit, X_ERROR(2p) after each reset;
  - no idle noise during gate layers; the encoding and terminal readout are ideal.
  Consequence: each check round puts about 5 x DEPOLARIZE1(2p) on every data qubit
  (four single-ancilla measurement layers plus one reset layer), which is
  probably the dominant data-noise source.
- Figure of merit in the repo: for qubits, F = |P(0) - P(1)| = |<X_L>| over
  accepted shots, averaged over the 30 sequences. Rejection = fraction of shots
  rejected. LRB fit: F(m) = b f^m with the offset fixed at 0 (weighted grid
  scan, points at F >= 1 dropped); physical RB fit: a + b f^m with a free offset.
- Matched physical RB: a single physical qubit, each physical gate (H, S, X, Z, I)
  followed by DEPOLARIZE1(0.1 p_RB), ideal measurement, with p_RB = 5 p.
- Qutrit LRB-D observations (from the production CSVs): at p = 5e-3 the depth-0
  rejection with one check round is about 27%. With `const=1` (terminal check
  only) F decays non-exponentially (0.96, 0.95, 0.93, 0.88, 0.67, 0.35, 0.13,
  0.08, 0.05 at the depth grid) and the rejection saturates at 1 - 1/81. With
  `unif=1` F stays near 0.95, but acceptance falls to about 4e-6 by m = 22.
  `const=0` (no checks) keeps F = 0.83 at m = 22, p = 1e-2.

## Tools and environment

- Run `module load python/3.11` first (Python 3.11.7 with stim 1.15,
  pymatching 2.3.1, ldpc 2.4.1, sinter, sdim, numpy, scipy, networkx).
  Plain `python3` is 3.6 and has none of them. Install extra packages only into
  a venv inside the worktree, never into the user's site-packages.
- NERSC Perlmutter login node, shared with about 50 users: keep computations
  light (minutes, a few cores, `nice -n 19`). Heavy sweeps are coordinated by
  the lead session.
- The qubit model lives in `src/lrb_qubit/` (built by the lead session). Check
  its docstrings for the API.

## Rules for every agent

- Write only the files you are assigned. Do not edit `src/lrb/`, `scripts/`,
  `slurm/`, notebooks, or anything outside `studies/correction-decoders/qubit/`
  unless your task says otherwise.
- Never touch the main checkout `/pscratch/sd/s/sohan100/LRB-Simulations-sdim`
  except to read the run folder above. Do not run `scripts/generate_circuits_folded.py`.
- No git commits, no pushes, no sbatch or salloc.
- Label every claim as one of: **verified analytically**, **verified
  numerically**, **from literature** (with citation), or **conjecture**.
- Every citation must be fetched by you (title, authors, year, arXiv ID or DOI
  confirmed from the actual page). Never cite from memory alone. If you cannot
  fetch it, mark it UNVERIFIED.
