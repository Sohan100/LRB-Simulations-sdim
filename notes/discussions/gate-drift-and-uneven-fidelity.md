# Gate drift and uneven fidelity across a chip

Opened 2026-11-24 by Sohan, from Chuanhong's proposal in the meeting
(`../meetings/2026-11-24-chuanhong.md`). Status: open.

## The question

Real chips do not have one gate error. If the gate errors follow a Gaussian spread across the QPU,
or drift over time, how do the LRB-D pseudo-thresholds of the qutrit and the qubit code move? A
related question from the meeting: if 10 of 100 qubits fall below threshold, does error correction
break? The meeting's consensus was that every site has to meet the threshold and that one site past
the crossover already degrades the logical error, but how much depends on the code and on where the
bad site sits.

## What it would take

- The generator applies one `p` to every gate of a type (`src/lrb/circuit_generator.py`,
  `CircuitLevelNoiseRates`). A per-site or per-coupler `p` drawn from a distribution needs a noise
  hook that takes a map from qudits and pairs to rates.
- The rest of the pipeline (runs, postselection, fits, thresholds) would not change. A run would sweep
  the mean `p` at a fixed spread, or the spread at a fixed mean.
- The `[[5,1,2]]` code is small enough that we can also put one bad gate on each location in turn
  and see which locations matter.

## Replies
