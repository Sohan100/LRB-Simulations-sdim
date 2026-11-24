# Memory of the folded surface code under circuit-level noise

We wanted to know whether the folded surface code, laid out as in `../layout-note.pdf`, keeps up
with the rotated surface code as a memory, since its selling point is elsewhere: every
single-qubit Clifford is a constant-depth, fold-transversal gate. Short answer: the thresholds are
the same, the folded code needs 10 to 30% more qubits for the same memory, and a logical H costs
nothing while a logical S costs about half a round of memory.

## What we simulated

- **Codes.** The folded code is the unrotated surface code `[[d^2 + (d-1)^2, 1, d]]` with
  `(2d-1)^2` physical qubits. Folding moves qubits and keeps every ancilla-data coupling, so its
  check round is the unrotated code's. The baseline is the rotated code `[[d^2, 1, d]]` with
  `2d^2 - 1` qubits. Both are built by `memory_circuits.py` from the same three ingredients (data
  qubits, checks with a four-step schedule, logical operators).
- **Schedule.** One round is: reset ancillas, basis change, four CNOT layers, basis change, measure.
  Folded checks visit north, west, east, south, so the hook left halfway is a diagonal pair. Rotated
  X checks visit NW, NE, SW, SE and Z checks NW, SW, NE, SE.
- **Noise.** SD6 and SI1000 exactly as in Tables 1 and 2 of Gidney, Newman, Fowler and Broughton,
  "A fault-tolerant honeycomb memory", Quantum 5, 605 (2021). Under SI1000 every CNOT is compiled to
  H CZ H and neighbouring Hadamards cancel. Measurement and reset are separate time steps, as in our
  LRB-D runs, and idle qubits in those steps also get the resonator-idle noise. Unlike the LRB-D runs
  (`docs/simulation-model.md`), a measured qubit gets no depolarizing channel after the measurement,
  because Gidney et al.'s table has none; the ancilla is reset in the next time step anyway.
- **Experiment.** Prepare a logical X or Z eigenstate, run d rounds, measure every data qubit.
  Error per round: `eps = (1 - (1 - 2 P_L)^(1/d)) / 2`, averaged over the X and Z bases.
- **Decoder.** Minimum-weight perfect matching (pymatching 2.4 through sinter 1.16, stim 1.16),
  the same for both codes. Each task ran to 1,000 logical errors or 10^7 shots; the points with
  p <= 0.15% at d = 7 to 11 ran to 200 errors or 2 x 10^9 shots.
- **Checks** (`verify_memory.py`, run for d = 3 to 9): every detector and the observable are
  deterministic without noise; the shortest logical error of every noisy circuit has weight d; and
  every two-qubit gate of the folded circuit acts on a coupler of `../data/layout-d<d>.json`.

## Results

| | | folded | rotated |
|---|---|---|---|
| threshold per round (d = 9 and 11 cross) | SD6 | 0.70% | 0.73% |
| | SI1000 | 0.51% | 0.54% |
| threshold per d-round block (d = 9 and 11) | SD6 | 0.53% | 0.52% |
| | SI1000 | 0.42% | 0.39% |
| Lambda at p = 0.1% (fit over d = 5 to 11) | SD6 | 11.3 | 6.7 |
| | SI1000 | 9.3 | 5.7 |
| qubits for a teraquop memory at p = 0.1% | SD6 | 2,025 (d = 23) | 1,681 (d = 29) |
| | SI1000 | 2,401 (d = 25) | 2,177 (d = 33) |

Our rotated baseline lands inside the per-block thresholds that the honeycomb paper reports for
the rotated surface code (0.5 to 0.7% for SD6, 0.3 to 0.5% for SI1000), so the comparison is
against the standard numbers.

Error per round at p = 0.1% (`results/summary.csv` has every point with its interval):

| d | folded qubits | SD6 | SI1000 | rotated qubits | SD6 | SI1000 |
|---|---|---|---|---|---|---|
| 3 | 25 | 1.07e-3 | 1.54e-3 | 17 | 9.9e-4 | 1.46e-3 |
| 5 | 81 | 1.01e-4 | 1.63e-4 | 49 | 1.53e-4 | 2.50e-4 |
| 7 | 169 | 9.1e-6 | 1.76e-5 | 97 | 2.34e-5 | 4.65e-5 |
| 9 | 289 | 7.7e-7 | 1.86e-6 | 161 | 3.46e-6 | 8.3e-6 |
| 11 | 441 | 7.1e-8 | 2.07e-7 | 241 | 5.0e-7 | 1.37e-6 |

What this says:

1. **The thresholds agree.** The crossings differ by less than the step of our p grid.
2. **At equal distance the folded code is better.** Its Lambda is about 1.6 times the rotated
   code's. The unrotated lattice has fewer logical strings of minimum weight.
3. **At equal qubit count the rotated code is better,** because it reaches distance d with about
   half the qubits. At 441 qubits and p = 0.1% (SD6) the fits give 7e-8 per round for the folded
   code and 1.3e-8 for the rotated one. Because the error falls exponentially in the distance, this
   costs the folded code only 10 to 30% more qubits for a teraquop memory (`results/teraquop.csv`
   has p = 0.05 to 0.2%).

## Logical gates between rounds

The folded code's advantage is its gates, so we also ran memories with a fold-transversal gate
between every two rounds (`--gate H` and `--gate S` in `run_memory.py`, folded code, d = 3 to 9,
p = 0.1, 0.2 and 0.3%, up to 2,000 errors each). `results/gates.csv` has every ratio.

- **H** is a Hadamard on every data qubit in the time step where the X ancillas change basis, which
  they do anyway. The SWAPs of the fold pairs are a relabelling of the two layers, so instead every
  ancilla measures the other check type from then on. Under SD6 the noise is unchanged and so is
  the error per round (ratio 1.00 +- 0.05 in both bases). Under SI1000 the new Hadamards cancel
  some of the compiled ones and the ratio sits slightly below 1.
- **S** adds one time step of CZ gates on the `(d-1)^2` fold pairs; S and S^dag on the fold
  qubits share the basis-change time step. It raises the error per round by a factor 1.2 to 1.7,
  about half a round of memory per gate, and keeps the circuit distance at d.
- On the rotated code neither gate is transversal on fixed hardware. H needs a patch rotation, and
  Moussa estimates a logical S at about 72 d^3 qubit-rounds (Phys. Rev. A 94, 042316, 2016).

The S memories run in the Z basis only. A CZ never turns a Z error into an X error, so matching the
Z checks decodes Z_L. In the X basis an X error between two gates becomes a Z error on its mirror
qubit at the next CZ and flips X_L, and only the Z checks see it. Decoding that needs both sectors at
once (correlated decoding of transversal gates), which we have not done. With `detectors="sector"`
only the checks of the memory's basis get detectors; for plain and H memories this gives the same
predictions as all detectors, which we checked (all differences within 1.8 standard errors).

## Files

| file | what |
|---|---|
| `memory_circuits.py` | both codes, the SD6/SI1000 noise, the circuits, the logical gates |
| `verify_memory.py` | the checks listed above |
| `run_memory.py` | runs a sweep with sinter; the output CSV is also its resume file |
| `run_memory.slurm`, `run_gates.slurm` | the SLURM jobs we used (debug queue, resubmit to resume) |
| `analyze_memory.py` | error per round, thresholds, Lambda, teraquop sizes, gate ratios |
| `plot_memory.py` | `figures/memory-{thresholds,footprint,lambda,gates}.pdf` |
| `results/stats.csv` | shots and logical errors of every task: everything above follows from it |
| `results/*.csv` | the derived tables |

To reproduce:

```bash
pip install -e ".[memory]"                    # from the repository root
python layouts/folded-surface-code/memory/verify_memory.py 9
python layouts/folded-surface-code/memory/run_memory.py --noise SD6 --distances 3 5 7 \
    --ps 0.001 0.002 0.003 --out layouts/folded-surface-code/memory/results/raw/memory-SD6.csv
python layouts/folded-surface-code/memory/analyze_memory.py --extras
python layouts/folded-surface-code/memory/plot_memory.py
```

`analyze_memory.py` reads the sinter files in `results/raw/` when they exist (they are not in git)
and `results/stats.csv` otherwise; both give identical tables.
