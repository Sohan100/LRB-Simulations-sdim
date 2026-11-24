# Simulation model

What one LRB-D run simulates, in the order the generator builds it. The paper's methods section
says the same in prose; this page points at the code.

## Protocol

A benchmark sequence of depth `m` is: prepare the logical `|+_L>`, apply `m` random logical
Cliffords and then their inverses in reverse order (the mirror inverse, `2m` noisy Cliffords), with
a noisy check round after every forward Clifford, and read every data qudit in the X basis. The
interval `k` only decides which rounds' syndromes are used for postselection: `unif-k` keeps a shot
if the checks after every `k`-th Clifford pass, and `const-c` uses `c` evenly spaced checks;
`const-0` is the terminal check alone. Physical RB on one qudit runs at the same `p` with the same
noise model.

Fits use the zero-offset model `F(m) = A f^m` over every depth including `m = 0`, so `f` is per
forward-plus-inverse pair. For qudit dimension `q` the average fidelity is
`F = (1 + (q - 1) f) / q`. The pseudo-threshold `p*(k)` is where LRB-D with interval `k` and physical
RB have the same error per Clifford.

## Clifford group

Random Cliffords are drawn uniformly from the full single-qutrit Clifford group (216 elements),
each written as one of the 9 Paulis `X^a Z^b` times one of 24 words in `H` and `S`
(`tests/test_qutrit_clifford_group.py` checks the group against sdim's gates). The qubit code uses
the 24-element single-qubit group the same way.

## Noise

`si1000` and `sd6` are the circuit-level models of Gidney, Newman, Fowler and Broughton (2021),
generalized to qudits. The swept `p` is the two-qudit gate error. Every logical Clifford and every
check round is scheduled into as-soon-as-possible moments, and every idle qudit gets idle noise.

| location | SI1000 | SD6 |
|---|---|---|
| two-qudit gate (DEP2 after) | p | p |
| one-qudit gate (DEP1 after) | p/10 | p |
| idle qudit in a gate moment (DEP1) | p/10 | p |
| idle qudit in a measure or reset moment (DEP1) | 2p | p |
| measurement (result flip before, DEP1 after) | 5p, p | p, p |
| reset (flip after) | 2p | p |

The encoded preparation and the terminal logical readout are noiseless. Ancillas are reset right
before each check round, so they do not idle through the logical gates. The historical
`depolarizing` model keeps its original placement.

## Sampling

The `dem` backend compiles each circuit into an `sdim.dem.DetectorErrorModel` and samples detector
and observable flips from it; `tests/test_dem_backend.py` checks it against sdim's Pauli-frame
sampler. Postselection is vectorized over shots. Each probability index checkpoints its tallies, and
the checkpoint carries a fingerprint of `scripts/run_lrb_experiment.py` and of `src/lrb/`
`lrb_simulation.py`, `dem_simulation.py`, `detector_events.py`, `code_definitions.py` and
`code_simulation_profiles.py`, so a run only resumes with the code that started it.

## Result files

`LRBSimulationPipeline.write_stats()` writes five rows per depth list, and
`LRBSimulationPipeline.read_stats()` reads the same layout:

1. `Probability,<p>`
2. `Fidelity averages,<d0>,<d1>,...`
3. `Fidelity Standard Deviations,<d0>,<d1>,...`
4. `Rejected Runs,<d0>,<d1>,...`
5. `Rejected Standard Deviations,<d0>,<d1>,...`

## Uncertainties

Pseudo-thresholds are quoted with one standard error (68% confidence) from a bootstrap over whole
Clifford sequences (`scripts/sequence_bootstrap_thresholds.py`), one draw shared by all depths and
all `p`.
