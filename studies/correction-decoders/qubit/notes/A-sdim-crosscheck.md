# A: sdim cross-check of the qubit Stim model

Work package A. This is an independent check of the lead's Stim model of qubit
`[[5,1,2]]` LRB (`src/lrb_qubit/`) against the repository's own qudit pipeline
(`src/lrb/` plus sdim), run at local dimension d = 2.

Script: `studies/correction-decoders/qubit/scripts/crosscheck_sdim.py`. Data: `studies/correction-decoders/qubit/results/A/`.

## Summary

**Headline: no discrepancy.** The repo's qudit pipeline, run at d = 2 with a
qubit code definition that mirrors `folded_qutrit`, reproduces the lead's Stim
model exactly at the circuit level and within statistical error at the
sampling level. No bug was found in the lead's model.

- **Exact check, no sampling.** All 108 repo-generated circuits were
  translated gate for gate into Stim: 3 grid sequences x p in {0.005, 0.02} x
  {data2 LRB, const0} x all 9 grid depths 0-22. Every one has a detector error
  model identical to the lead's circuit, down to the last floating-point bit
  of every mechanism probability (maximum relative difference 0). All 6
  negative controls are flagged. The smallest of them is a 0.1% change in one
  SI1000 component, which gives a relative difference of 8.8e-4. Two caveats:
  - "Last bit" holds for the DEM. The translated circuit's DEPOLARIZE2
    arguments are 1 - (1 - p), a few ulp from p.
  - DEM identity cannot see substitutions that act identically on Pauli
    frames, such as S <-> S^dagger (verifier, V1).
- **Sampling check.** sdim's Pauli-frame simulator (5e5 shots per circuit,
  2e5 at m = 22) was compared with Stim (5e6 per circuit, 2e6 at m = 22) on
  40 circuits. This is not the repo's production DEM-backend sampler, which
  the verifier tested separately and found in agreement (V4c). The comparison
  used 1130 informative two-sample tests:
  - 104 headline tests: rejection for const=1, unif=1 and const=0, and the
    logical flip rate among accepted shots;
  - 958 per-detector marginals;
  - 68 joint-distribution tables.

  None is significant after Holm (FWER 0.05) or Benjamini-Hochberg
  (FDR 0.05), within families or globally. The smallest global Holm-adjusted p
  is 0.080, and the largest headline |z| is 2.96 (Holm p = 0.32). Combined
  over circuits, every quantity type has Stouffer |z| <= 1.53. For example,
  the mean sdim - Stim difference is -3.9e-5 +- 1.1e-4 for const=1 rejection
  and -4.4e-5 +- 1.3e-4 for the const=1 flip rate.
- **Absolute check.** Both samplers match exact values computed from the
  DEM. For the goodness of fit of the terminal joint law (last-round syndrome
  x logical, 34 circuits), sdim gives chi2 = 692.7 on 718 dof (p = 0.75) and
  Stim gives 665.4 on 718 dof (p = 0.92).
- **One flag, not reproduced.** In the main run, one sdim marginal differed
  from its exact value at the Holm level: m = 22, z = -4.21, Holm p = 0.025.
  It did not replicate on fresh seeds:
  - that detector gave z = +0.72 at 4e5 shots;
  - at 1e6 shots, the largest |z| over all 185 marginals of that circuit was
    2.95, which a 40-replicate Stim null puts at p = 0.46.

  It was a fluctuation.
- **The data2 floor belongs to the protocol.** The lead's first-order
  undetectable-logical floor with the data-0/3 readout is a property of the
  repo protocol, not of the Stim re-implementation. In the repo-generated
  circuits, as in the lead's, the merged DEM mechanism that flips the logical
  and fires no detector has probability
  [1 - (1 - 8p/3)^3 (1 - 16p/15)] / 2 at every depth. Its small-p limit is
  68p/15 = 4.533p, the lead's value. At p = 0.005 and p = 0.02 it is 4.459p
  and 4.243p.
- **Hazards in `src/lrb`, not production bugs.** The generator mis-slices
  pre-sampled Clifford lists longer than max(depths), and the unpack specs
  default to dimension = 3.

## Methods

### 1. A qubit `LRBCodeDefinition` written in sdim's d = 2 gate names

`FoldedQubitCode` in the script is a gate-for-gate copy of
`FoldedQutritDetectionCode` (`src/lrb/code_definitions.py`). It uses only sdim's
d = 2 names H, P, P_INV, CNOT, CZ, SWAP, X, Z, I, M and RESET. At d = 2 every
qutrit inverse gate is its own inverse, so the translation is H_INV -> H,
CNOT_INV -> CNOT, CZ_INV -> CZ, X_INV -> X, Z_INV -> Z. P and P_INV stay
distinct. Self-test (a) compares every template, instruction by instruction,
against the qutrit template with those names mapped.

What P and CNOT mean in sdim at d = 2:

- **P = S = diag(1, i), P_INV = S^dagger.** sdim's prime-dimension tableau
  maps X -> omega^{1/2} X Z for even d (`ExtendedTableau.phase` docstring,
  `sdim/tableau/tableau_prime.py`). At d = 2 that is i·XZ = Y, and
  `sdim.unitary.generate_p_matrix(2)` is diag(1, omega^{1/2}) = diag(1, i).
  Self-test (b) confirms this numerically: sdim's cirq export of the circuit
  "H; P" gives (|0> + i|1>)/sqrt 2, and "H; P_INV" gives (|0> - i|1>)/sqrt 2.
  The qubit catalog's `P` is therefore Stim's `S`, which is what the lead's
  model uses. Only this unitary check (self-test b) fixes the convention.
  Every other gate here (H, X, Z, CNOT, CZ, SWAP), the |0> preparation and the
  Z measurement are real. Complex conjugation therefore swaps S and S^dagger
  and leaves every outcome distribution unchanged. Pauli-frame propagation is
  also the same for S and S^dagger. So self-test (d), the noiseless tests and
  every statistic in this study are blind to the convention. A P <-> P_INV
  swap would change none of them (verifier, V2). The convention is correct,
  but no result here depends on it.
- **`add_gate("CNOT", a, b)` has control a and target b.** The frame rule is
  `x[b] += x[a]; z[a] -= z[b]` (`sdim/program.py`, `simulate_frame`).
  Self-test (c) checks this numerically: X0; CNOT 0 1 -> (1, 1) and
  X1; CNOT 0 1 -> (0, 1). So the z_ALT block `CNOT data -> ancilla` and the
  X block `CNOT ancilla -> data` are the CONTEXT.md check circuits.

The profile mirror is a `_UnpackSpec` with `dimension=2`, `stabilizer_wires=(5,6,7,8)`,
logical terms `x0 + x3`, and terminal X-stabilizer terms
`x0+x1+x2` and `x1+x3+x4`. These are the folded_qutrit terms
`(x0+x1-x2, x3-x1-x4)` reduced mod 2. The spec feeds a `CodeSimulationProfile`
whose unpackers come from `CodeSimulationProfileRegistry._make_unpacker` and
`_make_const0_unpacker`, so the repo's detector-backed unpack code is used
unchanged. Descriptors come straight from `lrb_qubit.code.SINGLE_QUBIT_CLIFFORDS`,
which uses P for S. Their inverse lists follow
`QutritCliffordLibrary.build_inverse_lookup` (token -> token_INV, in
descriptor order).

### 2. Circuit generation through the repo generator

`LRBCircuitGenerator(code_definition=<qubit definition>, noise_model="si1000")`
runs `generate_lrb_clifford_sequence` (terminal readout of data 0 and 3) and
`generate_lrb_const0_clifford_sequence` (no checks, ideal X readout of all
five data qubits). After generation, `update_noise_param(circuit, p)` and
`NoiseModelUtils.ensure_noise_params` are applied, exactly as
`LRBSimulationPipeline.LRB` does. The last step expands every `N2` into the
uniform 16-entry `prob_dist` [1-p, p/15 x 15], which is DEPOLARIZE2(p). sdim's
`N1` channel 'd' draws X, Z or XZ uniformly (DEPOLARIZE1), and 'f' draws X
(X_ERROR). Self-test (g) lists the inventory: 'd' at 0.1p, p and 2p, 'f' at 5p
and 2p.

Sequences are the lead's grid sequences 0, 1, 2
(`lrb_qubit.experiment.clifford_sequences(3, 22, 20261028)`, matching
`results/grid_main/meta.json`): `HP I Z PH P ZHP`, `PH XPH XZHPH XZHPH ZP XZHPH`
and `XZ XHPH H ZP PH PH` for the first six Cliffords. Every circuit is built
with `clifford_strings` cut to exactly max(depths) descriptors. See "Bugs and
hazards" for why.

### 3. Simulation with sdim

`sdim_frame_sample` calls `Program._simulate_tableau(shots=1)` for the
noiseless reference shot, then `_build_ir`, `simulate_frame` and
`_combine_detector_results`. These are the calls `Program.simulate` makes, in
the same order. The only call left out is `_combine_results`, which wraps
every frame shot in a Python object.

This is sdim's Pauli-frame sampler. It is not the repo's production sampler.
The mirrored production run used `--simulation-backend dem`, which is
`lrb.dem_simulation.simulate_circuit_with_dem`. That backend uses these sdim
calls only to compile unit-fault responses. It then samples faults with its
own code (`dem_simulation.py:428-611`), which this work package did not run.
The verifier ran that backend separately, on 8 new circuits at 1e6 shots
each, and found that it reproduces the exact DEM law (verify notes, V4c).
Self-test (f) shows that with the same numpy and Python seeds, every detector
array, the reference shot and every frame measurement are identical to
`Program(circuit).simulate(shots=401)`.

The output `(reference results, detector payload)` goes to the repo's
`unpack_func` / `const0_unpack_func` with `shots = extra + 1`. As in
`LRBSimulationPipeline.LRB`, index 0, the noiseless reference shot, is then
dropped. Policies are applied with the repo's `postselect_constant_number`
(const=1, and const=0 on the const0 circuit), `postselect_uniform_interval`
(unif=1), `postselect_record` and `extract_lrb_counts(dimension=2)`. Every
chunk is checked internally:

- reference ancilla values are 0;
- the reference logical parity x0 + x3 is 0;
- the unpacker's pass/fail per round equals "no raw flip in that round" from
  the detector arrays;
- the detector arrays equal the legacy raw-measurement tensor (stride 2: M then
  the RESET slot);
- `extract_lrb_counts` agrees with the per-shot arrays.

### 4. The Stim model

`build_lrb_circuit(seq, m, p, terminal="data2")` and `build_lrb_const0_circuit`
are sampled with `compile_detector_sampler`. Raw per-round syndromes come from
`LRBCircuit.raw_syndromes`, and policies from `baseline.used_rounds` plus
`accept_mask`, which is the lead's own path. const=0 accepts when both final
X detectors are clean.

### 5. Exact circuit-level comparison (no sampling)

`sdim_to_stim` translates each repo-generated d = 2 circuit gate for gate:
P -> S, P_INV -> S_DAG, RESET -> R, N1 -> DEPOLARIZE1 / X_ERROR, uniform N2 ->
DEPOLARIZE2, and sdim `DETECTOR` / `LOGICAL_OBSERVABLE` expressions parsed
mod 2. The lead's circuit gets the same raw detectors: one per ancilla
measurement, in the generator's (round, wire 5..8) order, or the two direct
X parities for const0. Both Stim detector error models are reduced to
{(detector set, observable set): probability} with equal symptom sets merged,
and then compared. Negative controls check that the comparison can fail:
perturbing one SI1000 component of the lead circuit (meas_idle x1.001,
reset_idle x0, one_qubit x1.01) or giving the lead circuit a different
sequence must be flagged.

Limits of this comparison, found by the verifier:

- **Frame-equivalent substitutions are invisible.** A DEM cannot see a
  substitution that acts identically on Pauli frames. The verifier's
  "P <-> P_INV everywhere" control leaves every DEM bit-identical (V1). Only
  its per-qubit timeline comparison catches it.
- **"Bit-identical" refers to the DEMs.** The translator writes N2 as
  `DEPOLARIZE2(1 - prob_dist[0])`, which is a few ulp away from p (for
  example 0.0050000000000000044). Stim's DEM conversion rounds this away, so
  the DEMs are bit-identical but the translated circuit text is not.
- **The translator parses the intended expression.** It reads the detector
  expression string, not the lambda sdim actually evaluates. sdim's
  placeholder substitution can mis-evaluate some index orders, for example
  `rec[1]+rec[0]` (verifier, V7 N3). The expressions in these circuits are
  single records, the const0 parities `rec[0]+rec[1]+rec[2]` and
  `rec[3]+rec[1]+rec[4]`, and logicals with increasing indices. Tracing
  sdim's substitution shows they evaluate as intended (verified
  analytically). A mis-evaluation would also show up in the sampling
  comparison (R4).

### 6. Statistics

Shots per circuit: 5e5 for sdim (25 chunks of 2e4, each with its own
reference shot) and 5e6 for Stim on the 18 + 18 main circuits; 2e5 for sdim
(20 chunks of 1e4) and 2e6 for Stim on the four extra deep circuits. The
main circuits are (3 sequences) x (m = 0, 2, 6) x (p = 0.005, 0.02) x
(data2 LRB, const0). The deep circuits are sequence 0 at m = 22 for both p
and both kinds. Seeds are fixed through `numpy.random.SeedSequence` over
(namespace, kind, sequence, depth, p index, chunk), with namespace 0 for sdim
and 1 for Stim.

- Two-sided pooled two-proportion z-tests compare sdim and Stim for rejection
  (const=1, unif=1, const=0), the logical flip rate among accepted shots, the
  unconditional logical flip rate, every raw ancilla-flip marginal per
  (round, stabilizer), every difference-detector marginal, and the const0
  X-parity marginals. Where a 2x2 table has an expected count below 10, a
  Fisher exact p-value is also reported (column `p_fisher`).
- Two chi-square homogeneity tests per circuit check joint structure: the
  last-round 4-bit syndrome x logical bit (32 cells), and the number of
  detection events x logical bit. Cells with expected count below 5 are
  pooled.
- Multiple comparisons use Holm-Bonferroni (FWER 0.05) within each family
  (headline, marginal, joint) and over all tests together. Benjamini-Hochberg
  (FDR 0.05) is a secondary, more powerful screen. Degenerate comparisons
  (both samples all 0 or all 1, or no accepted shots) carry no information and
  are excluded from the corrections. Examples are the noiseless m = 0 const0
  circuit and unif=1 at m = 22 with p = 0.02.
- To look for small systematic biases, the z-scores of each quantity type are
  combined over the independent circuits with a Stouffer statistic. For
  detector types, the z-scores are first averaged within each circuit. That
  average's null variance is at most 1, so the combination is conservative.
- As an absolute reference, every marginal is also compared with its exact
  value computed from the lead circuit's DEM:
  P(fire) = (1 - prod(1 - 2 p_i)) / 2.
- Headline quantities are compared with exact values wherever the joint law
  is small enough to compute by convolving the DEM mechanisms over the needed
  bits: const=1 at every depth (4 terminal bits + the logical), const=0
  (2 X parities + the logical) and unif=1 up to m = 2 (13 bits).
- Within one circuit, the headline quantities come from the same shots and
  are strongly correlated. For example, unif=1 rejections contain the const=1
  rejections, with correlation about 0.75 at m = 2. So a dispersion statistic
  such as sum z^2 over them is not chi-square distributed. The
  correlation-free test is one multinomial goodness of fit per circuit and
  simulator: the full terminal joint law (last-round syndrome x logical, 32 or
  8 cells) against its exact value. These tests are independent across
  circuits, so their chi2 values add up.
- Follow-ups were fixed before resampling with fresh seeds. `confirm` takes
  the two largest sdim-vs-exact marginal deviations of the main run,
  resamples them at 4e5 shots each, and applies Bonferroni over 2. `recheck`
  takes all 185 marginals of the deep circuit at 1e6 sdim shots. Its max|z|
  is calibrated against 40 independent Stim replicates of the same size,
  which accounts for the correlations between marginals exactly.

## Results

### R1. Self-tests (`results/A/selftest.csv`): 9/9 pass

| # | check | result |
|---|---|---|
| a | d = 2 templates equal the folded_qutrit templates with H_INV, CNOT_INV, CZ_INV, X_INV, Z_INV mapped to H, CNOT, CZ, X, Z (encoder, z_ALT, X checks, reset, both terminal readouts, 9 logical tokens) | pass |
| b | sdim P at d = 2 is S: `generate_p_matrix(2)` = diag(1, i); statevectors of "H;P" and "H;P_INV" are (\|0> +- i\|1>)/sqrt 2 | pass |
| c | `add_gate("CNOT", a, b)`: control a, target b (X0;CNOT 0 1 -> (1,1); X1;CNOT 0 1 -> (0,1)) | pass |
| d | sdim d = 2 tableau vs Stim on 300 random 4-qubit Clifford circuits (25 gates from H, P, P_INV, X, Z, CNOT, CZ, SWAP), Z readout: 0 mismatches in determinism or deterministic value. This test cannot tell S from S^dagger (complex conjugation symmetry; see Methods 1). | pass |
| e | Noiseless circuits (24 descriptors at m = 1; 3 sequences at m = 0, 2, 6, 22; LRB and const0): reference ancillas 0 and deterministic, reference x0 + x3 = 0, no detection events or logical shifts | pass |
| f | Fast frame path is bit-identical to `Program(circuit).simulate(shots=401)` under the same seeds | pass |
| g | Noise inventory after `update_noise_param` + `ensure_noise_params`: N1 'd' at 0.1p, p, 2p; 'f' at 5p, 2p; every N2 = [1-p] + 15 x [p/15] | pass |
| h | INFO: generator with a pre-sampled list longer than max(depths) builds a wrong circuit (see Bugs) | hazard shown |
| i | INFO: unpack spec left at default dimension 3 returns logical value 2 at d = 2 (see Bugs) | hazard shown |

### R2. Exact detector-error-model comparison (`dem_exact.csv`, `dem_negative_controls.csv`)

| kind | p | circuits | DEM mechanisms per circuit | identical | max rel. difference | undetectable logical prob. (both) |
|---|---|---|---|---|---|---|
| LRB data2 | 0.005 | 27 | 20 - 1274 | 27/27 | 0 | 4.4592 p (all depths) |
| LRB data2 | 0.02 | 27 | 20 - 1274 | 27/27 | 0 | 4.2429 p (all depths) |
| const0 | 0.005 | 27 | 0 - 6 | 27/27 | 0 | 0 |
| const0 | 0.02 | 27 | 0 - 6 | 27/27 | 0 | 0 |

Negative controls (sequence 0, m = 6, p = 0.005; lead circuit perturbed):

| control | kind | flagged | max rel. difference |
|---|---|---|---|
| meas_idle x 1.001 | LRB | yes | 8.8e-4 |
| reset_idle x 0 | LRB | yes | 0.45 |
| one_qubit x 1.01 | LRB | yes | 5.0e-3 |
| one_qubit x 1.01 | const0 | yes | 9.9e-3 |
| lead uses sequence 1 | LRB | yes (different symptom sets) | 1 |
| lead uses sequence 1 | const0 | yes | 0.54 |

Caveats:

- The identity is exact for the DEMs, but a DEM cannot see S <-> S^dagger: a
  P <-> P_INV control stays bit-identical (verifier, V1).
- The translated circuits' DEPOLARIZE2 arguments are 1 - (1 - p), a few ulp
  from p, which Stim's DEM conversion rounds away.

**The data2 undetectable-logical floor.** The last column of the first table
is the probability of the merged DEM mechanism that flips the logical and
fires no detector. At every depth it is

    P_silent(p) = [1 - (1 - 8p/3)^3 (1 - 16p/15)] / 2.

Its small-p limit is 68p/15 = 4.533p, the lead's value. The tabulated
4.4592p and 4.2429p are its values at p = 0.005 and p = 0.02. The verifier
derived the closed form and matched it to 3e-15 relative (V5). The values
stored here match it to their printed 7 digits.

At first order the floor has five sources, all in the terminal round
(verifier, V5):

- DEPOLARIZE1(2p) idle noise on data 0 in the M7 layer, data 0 in the M8
  layer and data 3 in the M8 layer, contributing 4p/3 each;
- DEPOLARIZE2(p) after CX 7->0 and after CX 8->3, contributing 4p/15 each.

Data 2 contributes nothing. In the DEM, each idle location contributes a
factor (1 - 8p/3) and the two two-qubit channels together contribute
(1 - 16p/15). That is where the closed form comes from.

### R3. Headline comparisons per circuit (`tests_headline.csv`)

Notation: 0.24979(61) means 0.24979 +- 0.00061 (one binomial SE). k/n is
shown where a count is below 10. "exact" is computed from the lead circuit's
DEM ("-" means unif=1 at m >= 6, where it is not tractable). z is the pooled
two-proportion z-score, sdim minus Stim. At m = 0, unif=1 uses the same
single round as const=1, so those columns repeat.

**LRB data2 circuits: rejection**

| p | seq | m | rej const=1 exact | sdim | Stim | z | rej unif=1 exact | sdim | Stim | z |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0 | 0 | 0.24982 | 0.24979(61) | 0.24968(19) | +0.16 | 0.24982 | 0.24979(61) | 0.24968(19) | +0.16 |
| 0.005 | 1 | 0 | 0.24982 | 0.24993(61) | 0.24971(19) | +0.34 | 0.24982 | 0.24993(61) | 0.24971(19) | +0.34 |
| 0.005 | 2 | 0 | 0.24982 | 0.24990(61) | 0.24961(19) | +0.46 | 0.24982 | 0.24990(61) | 0.24961(19) | +0.46 |
| 0.005 | 0 | 2 | 0.5966 | 0.59697(69) | 0.59695(22) | +0.03 | 0.72799 | 0.72835(63) | 0.72834(20) | +0.02 |
| 0.005 | 1 | 2 | 0.60826 | 0.60976(69) | 0.60788(22) | +2.60 | 0.73542 | 0.73702(62) | 0.73514(20) | +2.87 |
| 0.005 | 2 | 2 | 0.60439 | 0.60434(69) | 0.60440(22) | -0.09 | 0.734 | 0.73443(62) | 0.73405(20) | +0.59 |
| 0.005 | 0 | 6 | 0.84522 | 0.84513(51) | 0.84514(16) | -0.02 | - | 0.96544(26) | 0.965292(82) | +0.54 |
| 0.005 | 1 | 6 | 0.85637 | 0.85622(50) | 0.85629(16) | -0.12 | - | 0.96894(25) | 0.968921(78) | +0.08 |
| 0.005 | 2 | 6 | 0.84948 | 0.85055(50) | 0.84941(16) | +2.14 | - | 0.96675(25) | 0.966422(81) | +1.22 |
| 0.005 | 0 | 22 | 0.93689 | 0.93683(54) | 0.93680(17) | +0.05 | - | 200000/200000 | 0.9999935(18) | +1.14 |
| 0.02 | 0 | 0 | 0.65104 | 0.65109(67) | 0.65106(21) | +0.03 | 0.65104 | 0.65109(67) | 0.65106(21) | +0.03 |
| 0.02 | 1 | 0 | 0.65104 | 0.65111(67) | 0.65134(21) | -0.31 | 0.65104 | 0.65111(67) | 0.65134(21) | -0.31 |
| 0.02 | 2 | 0 | 0.65104 | 0.65021(67) | 0.65118(21) | -1.38 | 0.65104 | 0.65021(67) | 0.65118(21) | -1.38 |
| 0.02 | 0 | 2 | 0.91703 | 0.91631(39) | 0.91713(12) | -2.00 | 0.99083 | 0.99060(14) | 0.990839(43) | -1.66 |
| 0.02 | 1 | 2 | 0.92074 | 0.92036(38) | 0.92083(12) | -1.19 | 0.99152 | 0.99169(13) | 0.991551(41) | +1.01 |
| 0.02 | 2 | 2 | 0.91951 | 0.91860(39) | 0.91953(12) | -2.30 | 0.99179 | 0.99145(13) | 0.991802(40) | -2.66 |
| 0.02 | 0 | 6 | 0.93732 | 0.93747(34) | 0.93714(11) | +0.91 | - | 500000/500000 | 0.99999720(75) | +1.18 |
| 0.02 | 1 | 6 | 0.93741 | 0.93742(34) | 0.93735(11) | +0.19 | - | 499997/500000 | 0.99999700(77) | -1.12 |
| 0.02 | 2 | 6 | 0.93738 | 0.93756(34) | 0.93739(11) | +0.47 | - | 499999/500000 | 0.99999740(72) | +0.25 |
| 0.02 | 0 | 22 | 0.9375 | 0.93725(54) | 0.93735(17) | -0.18 | - | 200000/200000 | 2000000/2000000 | +0.00 |

**LRB data2 circuits: logical flip rate among accepted shots**

| p | seq | m | flip const=1 exact | sdim | Stim | z | flip unif=1 exact | sdim | Stim | z |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0 | 0 | 0.025112 | 0.02471(25) | 0.025062(81) | -1.30 | 0.025112 | 0.02471(25) | 0.025062(81) | -1.30 |
| 0.005 | 1 | 0 | 0.025112 | 0.02518(26) | 0.025128(81) | +0.18 | 0.025112 | 0.02518(26) | 0.025128(81) | +0.18 |
| 0.005 | 2 | 0 | 0.025112 | 0.02472(25) | 0.025031(81) | -1.17 | 0.025112 | 0.02472(25) | 0.025031(81) | -1.17 |
| 0.005 | 0 | 2 | 0.044535 | 0.04470(46) | 0.04442(15) | +0.57 | 0.032307 | 0.03257(48) | 0.03228(15) | +0.58 |
| 0.005 | 1 | 2 | 0.042038 | 0.04193(45) | 0.04199(14) | -0.11 | 0.032403 | 0.03198(49) | 0.03220(15) | -0.44 |
| 0.005 | 2 | 2 | 0.039375 | 0.04018(44) | 0.03934(14) | +1.83 | 0.028517 | 0.02942(46) | 0.02838(14) | +2.18 |
| 0.005 | 0 | 6 | 0.1482 | 0.1498(13) | 0.14826(40) | +1.14 | - | 0.0380(15) | 0.03827(46) | -0.16 |
| 0.005 | 1 | 6 | 0.15076 | 0.1523(13) | 0.15087(42) | +1.05 | - | 0.0346(15) | 0.03694(48) | -1.45 |
| 0.005 | 2 | 6 | 0.13466 | 0.1357(13) | 0.13514(39) | +0.45 | - | 0.0344(14) | 0.03381(44) | +0.41 |
| 0.005 | 0 | 22 | 0.48824 | 0.4808(44) | 0.4898(14) | -1.94 | - | no acc. | 3/13 | n/a |
| 0.02 | 0 | 0 | 0.12557 | 0.12543(79) | 0.12495(25) | +0.58 | 0.12557 | 0.12543(79) | 0.12495(25) | +0.58 |
| 0.02 | 1 | 0 | 0.12557 | 0.12558(79) | 0.12545(25) | +0.16 | 0.12557 | 0.12558(79) | 0.12545(25) | +0.16 |
| 0.02 | 2 | 0 | 0.12557 | 0.12550(79) | 0.12593(25) | -0.52 | 0.12557 | 0.12550(79) | 0.12593(25) | -0.52 |
| 0.02 | 0 | 2 | 0.37681 | 0.3757(24) | 0.37573(75) | -0.02 | 0.2548 | 0.2646(64) | 0.2535(20) | +1.65 |
| 0.02 | 1 | 2 | 0.37299 | 0.3724(24) | 0.37239(77) | +0.01 | 0.25803 | 0.2608(68) | 0.2576(21) | +0.45 |
| 0.02 | 2 | 2 | 0.3516 | 0.3471(24) | 0.35156(75) | -1.81 | 0.21413 | 0.2165(63) | 0.2159(20) | +0.09 |
| 0.02 | 0 | 6 | 0.4977 | 0.4967(28) | 0.49892(89) | -0.74 | - | no acc. | 4/14 | n/a |
| 0.02 | 1 | 6 | 0.49758 | 0.4998(28) | 0.49737(89) | +0.83 | - | 1/3 | 8/15 | -0.63 |
| 0.02 | 2 | 6 | 0.49582 | 0.4941(28) | 0.49724(89) | -1.07 | - | 0/1 | 6/13 | -0.90 |
| 0.02 | 0 | 22 | 0.5 | 0.5044(45) | 0.4998(14) | +0.99 | - | no acc. | no acc. | n/a |

**const=0 circuits: rejection and flip rate among accepted shots**

| p | seq | m | rej const=0 exact | sdim | Stim | z | flip const=0 exact | sdim | Stim | z |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0-2 | 0 | 0 | 0/500000 | 0/5000000 | (degenerate) | 0 | 0/500000 | 0/5000000 | (degenerate) |
| 0.005 | 0 | 2 | 0.022371 | 0.02215(21) | 0.022333(66) | -0.85 | 2.8248e-05 | 0.0000225(68) | 0.0000272(24) | -0.61 |
| 0.005 | 1 | 2 | 0.043132 | 0.04354(29) | 0.043095(91) | +1.48 | 0.00010485 | 0.000100(14) | 0.0001058(47) | -0.35 |
| 0.005 | 2 | 2 | 0.035846 | 0.03562(26) | 0.035908(83) | -1.05 | 7.826e-05 | 0.000054(11) | 0.0000743(39) | -1.58 |
| 0.005 | 0 | 6 | 0.075122 | 0.07516(37) | 0.07520(12) | -0.10 | 0.00034398 | 0.000314(26) | 0.0003428(86) | -1.03 |
| 0.005 | 1 | 6 | 0.14727 | 0.14878(50) | 0.14722(16) | +2.96 | 0.0015242 | 0.001490(59) | 0.001525(19) | -0.57 |
| 0.005 | 2 | 6 | 0.096118 | 0.09633(42) | 0.09597(13) | +0.82 | 0.00058342 | 0.000494(33) | 0.000588(11) | -2.51 |
| 0.005 | 0 | 22 | 0.26411 | 0.26488(99) | 0.26435(31) | +0.51 | 0.0061259 | 0.00614(20) | 0.006243(65) | -0.50 |
| 0.02 | 0-2 | 0 | 0 | 0/500000 | 0/5000000 | (degenerate) | 0 | 0/500000 | 0/5000000 | (degenerate) |
| 0.02 | 0 | 2 | 0.086044 | 0.08527(39) | 0.08615(13) | -2.12 | 0.00046386 | 0.000514(34) | 0.000477(10) | +1.08 |
| 0.02 | 1 | 2 | 0.15906 | 0.15821(52) | 0.15947(16) | -2.32 | 0.0017482 | 0.001829(66) | 0.001759(20) | +1.03 |
| 0.02 | 2 | 2 | 0.13407 | 0.13396(48) | 0.13412(15) | -0.32 | 0.0012955 | 0.001277(54) | 0.001290(17) | -0.22 |
| 0.02 | 0 | 6 | 0.25952 | 0.25892(62) | 0.25943(20) | -0.79 | 0.005851 | 0.00586(13) | 0.005803(39) | +0.40 |
| 0.02 | 1 | 6 | 0.43861 | 0.43756(70) | 0.43859(22) | -1.40 | 0.026516 | 0.02645(30) | 0.026585(96) | -0.44 |
| 0.02 | 2 | 6 | 0.31803 | 0.31768(66) | 0.31804(21) | -0.52 | 0.010022 | 0.01009(17) | 0.010073(54) | +0.12 |
| 0.02 | 0 | 22 | 0.61908 | 0.6217(11) | 0.61985(34) | +1.66 | 0.1011 | 0.1002(11) | 0.10084(35) | -0.57 |

The const=0 circuit at m = 0 is noiseless, because encoding and readout are
ideal and there are no gates, so both simulators give exactly 0. Per-detector
marginals, raw and difference, are in `tests_marginals.csv`, with the exact
DEM value of every marginal in column `exact`.

### R4. Two-sample tests, sdim vs Stim, with multiple-comparison corrections

| family | informative tests | raw p < 0.05 (expected) | Holm sig. (family) | BH sig. (family) | mean z | sd z | max \|z\| (Holm p) |
|---|---|---|---|---|---|---|---|
| headline (rejection, flip among accepted) | 104 | 11 (5.2) | 0 | 0 | -0.06 | 1.13 | 2.96 (0.32) |
| marginal (raw, difference, X-parity, logical) | 958 | 41 (47.9) | 0 | 0 | +0.03 | 1.01 | 3.97 (0.067) |
| joint (chi2 homogeneity) | 68 | 4 (3.4) | 0 | 0 | - | - | min p 0.0081 (0.55) |
| all together | 1130 | 56 | 0 (global Holm) | 0 (global BH) | | | min global Holm p 0.080 |

The 11 headline excursions are not independent. Several come in pairs from
one circuit, because const=1 and unif=1 rejection share shots (seq 1 and 2 at
m = 2 in the table), and their signs are mixed. Fisher exact tests on the six
small-count headline cells give min p = 0.22.

Combined over circuits (Stouffer, one z per circuit; `tests_combined.csv`):

| quantity type | circuits | Stouffer z | mean diff sdim - Stim (+- SE) |
|---|---|---|---|
| LRB rej const=1 | 20 | -0.05 | -3.9e-5 +- 1.1e-4 |
| LRB rej unif=1 | 19 | +0.63 | +7e-7 +- 1.4e-6 |
| LRB flip\|acc const=1 | 20 | -0.20 | -4.4e-5 +- 1.3e-4 |
| LRB flip\|acc unif=1 | 17 | -0.07 | -6.5e-5 +- 1.3e-4 |
| LRB logical flip (all shots) | 20 | +0.63 | +5.9e-5 +- 1.3e-4 |
| LRB raw ancilla-flip marginals | 20 | +0.33 | +3.5e-5 +- 3.3e-5 |
| LRB difference-detector marginals | 20 | +0.13 | +3.4e-5 +- 3.2e-5 |
| const0 rej | 14 | -0.54 | -9.1e-5 +- 1.1e-4 |
| const0 flip\|acc | 14 | -1.53 | -9.6e-6 +- 5.7e-6 |
| const0 X-parity marginals | 14 | -0.13 | -3.6e-5 +- 6.5e-5 |
| const0 logical flip (all shots) | 14 | -0.45 | -7.1e-5 +- 8.9e-5 |

All Holm-adjusted Stouffer p-values are 1.00. The mean differences are
inverse-variance weighted with the pooled SE, so circuits with rates near 0
or 1 dominate them. The Stouffer z is the scale-free bias test.

### R5. Each simulator against exact DEM values

| comparison | tests | mean z | sd z | max \|z\| | Holm sig. (min Holm p) | KS p vs N(0,1) |
|---|---|---|---|---|---|---|
| sdim headline vs exact | 92 | -0.08 | 1.16 | 3.01 | 0 (0.24) | 0.34 |
| Stim headline vs exact | 92 | -0.02 | 0.97 | 2.51 | 0 (1.00) | 0.77 |
| sdim marginals vs exact | 958 | +0.03 | 1.00 | 4.21 | 1 (0.025) | 0.77 |
| Stim marginals vs exact | 958 | -0.01 | 0.97 | 3.50 | 0 (0.44) | 0.27 |
| sdim terminal joint law, GOF (34 circuits) | sum chi2 692.7 / 718 dof | | | | p = 0.75 (min per-circuit Holm p 0.39) | |
| Stim terminal joint law, GOF (34 circuits) | sum chi2 665.4 / 718 dof | | | | p = 0.92 | |

sdim's headline-vs-exact z-scores have sd 1.16, but these quantities are
correlated within each circuit. The correlation-free GOF on the same
information, the full terminal joint law per circuit, shows no excess for
sdim (p = 0.75).

### R6. Follow-up of the one Holm-level flag (`confirm.csv`, `recheck_*.csv`)

The flag: `lrb`, seq 0, m = 22, p = 0.005, difference detector round 19 /
S_Z1. In the main run sdim gave 0.14669 against exact 0.15005 (n = 2e5,
z = -4.21, Holm p = 0.025 over 958); Stim gave 0.15002 (z = -0.14).

| fresh-seed follow-up | result |
|---|---|
| `confirm`, flagged detector, 4e5 shots | 0.15046 +- 0.00056 vs exact 0.15005: z = +0.72, Bonferroni (x2) p = 0.94 |
| `confirm`, second-largest cell (const0 seq 1, m 6, p 0.005, X parity 0), 4e5 shots | 0.09881 +- 0.00047 vs exact 0.09838: z = +0.93, Bonferroni p = 0.71 |
| `recheck`, all 185 marginals of the m = 22 circuit, 1e6 shots | max\|z\| = 2.95 (raw r5 s1), min Holm p = 0.59, KS p = 0.19, mean z -0.09, sd z 0.97 |
| Stim null for that max\|z\| (40 replicates x 1e6 shots) | median 2.88, 95th percentile 3.81: p = 0.46 |

The deviation does not replicate, so the main-run flag was a fluctuation. By
the Stim-calibrated null, a max|z| of 4.21 over this circuit's correlated
marginals is roughly a 2-3% event.

### R7. Observations on the protocol, consistent in both simulators

- At m = 0 the qubit model rejects 24.98% of shots at p = 0.005 (exact
  0.24982). The qutrit production run rejects about 27% at p = 5e-3.
- Raw (non-difference) syndromes accumulate data errors across rounds, and
  the terminal raw syndrome tends to 4 uniformly random bits. The const=1
  rejection therefore approaches 1 - 1/16 = 0.9375 and the flip rate among
  accepted shots approaches 0.5. Exact values are 0.93732 and 0.4977 at
  m = 6, p = 0.02; 0.9375 and 0.5 at m = 22, p = 0.02; and 0.93689 and
  0.48824 at m = 22, p = 0.005. This is the qubit analogue of the qutrit
  1 - 1/81 saturation. unif=1 accepts essentially nothing at m >= 6 for
  p = 0.02, or at m = 22 for p = 0.005 (Stim: 13 of 2e6 shots).
- With the data-0/3 readout, the m = 0 flip rate among accepted shots is
  already 0.0251 at p = 0.005 and 0.126 at p = 0.02 (exact). In the DEM,
  mechanisms with no detector symptoms are independent of acceptance. So the
  depth-independent undetectable-logical probability accounts for about 89%
  and 68% of these rates. That probability is
  [1 - (1 - 8p/3)^3 (1 - 16p/15)] / 2, which is 4.46p and 4.24p here and
  tends to 68p/15 = 4.533p at small p (R2). This is the lead's
  "SPAM-like floor".

## Claims

| # | Claim | Label | Evidence |
|---|---|---|---|
| 1 | sdim's `P` at d = 2 is S = diag(1, i) and `P_INV` is S^dagger, so the catalog's `P` is Stim's `S` | verified analytically and numerically | sdim even-d phase rule X -> omega^{1/2} XZ (`ExtendedTableau.phase`), `generate_p_matrix(2)`; self-test b. Only self-test b fixes this convention. By complex conjugation symmetry, no statistic in this study depends on it (Methods 1; verifier V2). |
| 2 | sdim `add_gate("CNOT", a, b)` has control a and target b | verified analytically and numerically | frame rule `x[b] += x[a]` in `simulate_frame`; self-test c |
| 3 | sdim's d = 2 tableau (reference-shot) simulator agrees with Stim | verified numerically | self-test d: 0 mismatches in 300 random Clifford circuits. This test cannot tell S from S^dagger. The verifier extended it to mid-circuit M and RESET (V2). |
| 4 | `FoldedQubitCode` is the folded_qutrit profile with the inverse gate names mapped to their d = 2 values | verified numerically | self-test a |
| 5 | The repo's SI1000 insertion at d = 2 uses the same channels as the lead's model: DEPOLARIZE1(0.1p) after 1q gates (incl. identity); uniform 15-Pauli DEPOLARIZE2(p) after 2q gates; DEPOLARIZE1(2p) on idle wires plus DEPOLARIZE1(p) and X_ERROR(5p) per measurement; DEPOLARIZE1(2p) plus X_ERROR(2p) per reset layer; ideal encoding and readout; no reset after the terminal round | verified analytically (source) and numerically | self-test g; claim 6 |
| 6 | For 3 grid sequences x p in {0.005, 0.02} x all 9 grid depths, the repo-generated data2 LRB and const0 circuits have detector error models identical to `build_lrb_circuit(terminal="data2")` and `build_lrb_const0_circuit` | verified numerically (exact) | `dem_exact.csv`: 108/108 identical, max relative difference 0; `dem_negative_controls.csv`: 6/6 perturbations flagged. Caveats: (i) DEM identity cannot see frame-equivalent substitutions (S <-> S^dagger). The verifier's per-qubit timeline check does see them. It finds the lead's circuits equal to the mechanically translated repo circuits on new sequences, p and depths (V1), and `FoldedQubitCode` equals that mechanical translation instruction for instruction (self-test a). (ii) "To the last bit" refers to the DEM probabilities; the translated DEPOLARIZE2 arguments are 1 - (1 - p), a few ulp from p. |
| 7 | sdim frame-sampler sampling of the repo circuits and Stim sampling of the lead circuits agree on rejection for const=1, unif=1 and const=0, and on the logical flip rate among accepted shots | verified numerically | R3, R4: 104 tests, 0 significant after Holm or BH, max \|z\| = 2.96 (Holm p 0.32); Stouffer \|z\| <= 1.53 per quantity type |
| 8 | Per-detector marginal flip probabilities agree, both raw per ancilla and round and as difference detectors | verified numerically | R4: 958 tests, 0 Holm or BH, max \|z\| = 3.97 (Holm p 0.067), sd(z) = 1.01 |
| 9 | Joint distributions agree (terminal syndrome x logical; detection-event count x logical) | verified numerically | R4: 68 chi-square tests, 0 Holm (min p 0.0081, Holm 0.55) |
| 10 | Both samplers reproduce the exact DEM values; the single Holm-level sdim deviation was a fluctuation | verified numerically | R5: GOF sdim chi2 692.7/718 (p 0.75), Stim 665.4/718 (p 0.92); R6: flag did not replicate (z = +0.72; 185-marginal recheck p = 0.46) |
| 11 | sdim's frame sampling, used through the same internal calls as `Program.simulate`, is a faithful sampler of these circuits | verified numerically | self-test f (bit-identical to `Program.simulate`); claims 7-10 |
| 11b | The repo's production sampler (`--simulation-backend dem`, `lrb.dem_simulation`) also reproduces the exact DEM law at d = 2 | verified numerically by the verifier, not by this work package | verify notes V4c: 8 new circuits x 1e6 shots; 132 tests against exact values, max \|z\| 3.39, min Holm p 0.091; terminal joint law chi2 174.8 / 152 dof (p = 0.10). The one extreme cell gave z = +0.70 in a pre-registered replication. |
| 12 | No bug in the lead's model along the paths exercised: `code.py`, `circuits.build_lrb_circuit` / `build_lrb_const0_circuit` / `LRBCircuit.raw_syndromes`, and `baseline.used_rounds` / `accept_mask` for const=1 and unif=1 | verified numerically within the tested scope: 3 sequences, p in {0.005, 0.02}, exact at m = 0-22, sampled at m = 0, 2, 6, 22 | claims 6-10 |
| 13 | The equivalence also holds for other sequences and p (13a) and for the other LRB-D policies const=k and unif=k (13b) | (a) verified numerically by the verifier; (b) verified analytically | (a) verify notes V1: 3 new sequences, one hand-made to contain every token type, with p in {0.01, 0.0037} and m in {0, 3, 7}, plus all 24 descriptors at m = 1. Per-qubit timelines and DEMs are identical in both detector bases. (b) Every policy is a function of the raw flips, and their joint law with the logical is identical by the raw-basis DEM identity (claim 6). The lead's `used_rounds` is tested against the repo's `postselect_*` functions for k = 1..22 (`test_policies_match_qutrit_pipeline`). |
| 14 | The data2 undetectable-logical floor is a property of the repo protocol itself (terminal check without reset, then data-0/3 readout), not of the Stim re-implementation. In both circuit families the merged silent-logical DEM mechanism has probability [1 - (1 - 8p/3)^3 (1 - 16p/15)] / 2 at every depth: small-p limit 68p/15 = 4.533p (the lead's value); 4.459p at p = 0.005 and 4.243p at p = 0.02 | verified numerically (identical in the repo-generated and lead circuits); closed form verified analytically and numerically by the verifier | R2; `dem_exact.csv` matches the closed form to its 7 printed digits; verify notes V5 |
| 15 | Under raw-syndrome LRB-D, const=1 rejection tends to 1 - 1/16 and the flip rate among accepted shots tends to 1/2 at large depth | verified numerically (exact DEM values in R3); limit verified analytically (4 uniform terminal bits) | R3, R7 |
| 16 | `LRBCircuitGenerator` builds wrong circuits when given a pre-sampled list longer than max(depths); production `generate_tests` is unaffected | verified numerically and analytically | self-test h; source lines under "Bugs and hazards found" |
| 17 | The unpack specs' default `dimension = 3` corrupts d = 2 logical values when the reference shot has x0 = x3 = 1 | verified numerically | self-test i |

Scope of claims 7-11: they exercise sdim's Pauli-frame sampler
(`simulate_frame`), not the production `--simulation-backend dem` sampler.
Claim 11b covers the production sampler, from the verifier's run.

## Bugs and hazards found

### Lead's model (`src/lrb_qubit/`)

No bug found. The DEM identity (R2) covers every non-annotation instruction of
`build_lrb_circuit(terminal="data2")` and `build_lrb_const0_circuit`. The raw
marginals of the Stim samples come through `LRBCircuit.raw_syndromes` and
match both sdim's direct ancilla records and the exact values (R4, R5). That
validates the `back` bookkeeping of the difference detectors. The
`used_rounds` / `accept_mask` acceptance for const=1 and unif=1 matches the
repo's `postselect_constant_number` / `postselect_uniform_interval` applied to
sdim output (R3, R4).

### Repo pipeline (`src/lrb/`): latent hazards, not production bugs

1. **Pre-sampled Clifford lists longer than max(depths) give wrong
   circuits.** This affects `generate_lrb_clifford_sequence` and
   `generate_lrb_const0_clifford_sequence` in `src/lrb/circuit_generator.py`.
   - What happens: the inverse list is built from the whole
     `clifford_strings` list and reversed (lines 919-923 and 1102-1106), then
     sliced with `[(max_depth - depth):]` (lines 948-949 and 1120-1121). The
     only guard rejects lists that are too short (lines 912 and 1095). With a
     longer list, every circuit's inverse half silently gets extra inverses.
   - Failing check: self-test h. `depths=[2]` with 6 descriptors gives
     4 SWAPs instead of 2, and the noiseless logical readout flips in 48% of
     shots.
   - Production is safe: `generate_tests` always passes exactly `max(depths)`
     freshly sampled descriptors (lines 1245-1250).
   - Suggested fix: `clifford_strings = list(clifford_strings[:max_depth])`,
     or raise on a longer list.
   - This script avoids the problem by cutting every sequence to
     max(depths).
2. **Unpack specs default to `dimension = 3`.** The defaults are in
   `_UnpackSpec` (`code_simulation_profiles.py:89`), `LRBUnpackSpec`
   (`lrb_simulation.py:1086`) and `extract_lrb_counts(dimension=3)`
   (`lrb_simulation.py:2427`).
   - A d = 2 profile that leaves out `dimension=2` reconstructs logical
     values mod 3 from the reference shot. In about half of the reference
     shots x0 = x3 = 1, and then every shot reads logical value 2 (self-test
     i).
   - `default_stabilizer_pass_check` hard-codes `% 3`
     (`lrb_simulation.py:1377`). That is harmless for qubit values in {0, 1}.

### sdim 1.3.4

No incorrect behaviour found at d = 2 for H, P, P_INV, X, Z, CNOT, CZ, SWAP,
M, RESET, N1 ('d', 'f') or N2 (self-tests b-g; R2-R6). The only problem is
practical: the frame simulator loops over shots in Python for every M and
RESET, and `Program.simulate` also builds one Python object per (qudit,
measurement, shot). 2e4 shots at m = 6 took about 6 s on the login node and
up to about 25 s on the shared, oversubscribed compute node.

## Reproduce

```bash
cd <repository root>
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
S=studies/correction-decoders/qubit/scripts/crosscheck_sdim.py

# Light: login node, about 1 minute each.
nice -n 19 python $S selftest   # selftest.csv (9 checks)
nice -n 19 python $S dem        # dem_exact.csv, dem_negative_controls.csv

# Heavy: shared allocation (never sbatch/salloc).
JOB=$(squeue -u sohan100 -n q512shared -h -o %i | head -1)
run() { srun --jobid=$JOB --overlap -N 1 -n 1 -c 24 bash -c \
  "module load python/3.11; cd $PWD; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src; python $S $*"; }
run sample --workers 24 --sdim-shots 500000 --stim-shots 5000000 \
           --deep-sdim-shots 200000 --deep-stim-shots 2000000   # 9.4 min wall
nice -n 19 python $S analyze    # tests_*.csv (about 1 min)
run confirm --workers 24 --confirm-shots 400000                 # 5.7 min wall
run recheck --workers 24 --recheck-shots 1000000 --null-replicates 40
nice -n 19 python $S table      # markdown tables of R3
```

For a quick, statistically weaker rerun, use for example
`sample --sdim-shots 20000 --stim-shots 200000 --deep-sdim-shots 10000 --deep-stim-shots 100000`.
Seeds are fixed, so sdim reruns are bit-identical for the same chunking. Stim
reruns are bit-identical only with the same Stim version (1.15.0) and CPU
architecture.

Cost of the reported run: `sample` took 564 s wall on 24 workers of a node
shared with the lead's grid, 56 CPU-minutes of user time. The summed task
durations recorded in `sample_meta.csv` (`sdim_cpu_seconds` 10796,
`stim_cpu_seconds` 2225) are wall-clock times on that oversubscribed node,
not CPU time.

### Files

Code: `studies/correction-decoders/qubit/scripts/crosscheck_sdim.py` (subcommands `selftest`, `dem`,
`sample`, `analyze`, `confirm`, `recheck`, `table`).

Data in `studies/correction-decoders/qubit/results/A/`:

| file | content |
|---|---|
| `selftest.csv` | 9 semantic and plumbing checks |
| `dem_exact.csv` | 108 exact DEM comparisons (mechanism counts, max abs/rel difference, undetectable-logical probability) |
| `dem_negative_controls.csv` | 6 perturbed-lead comparisons that must fail |
| `counts.csv` | raw counts k/n per circuit, quantity and simulator (the input of every test) |
| `hist.csv` | joint histograms: terminal pattern x logical, detection-event count x logical |
| `sample_meta.csv` | sequences, shots, seeds scheme, timing, versions |
| `tests_headline.csv` | two-sample tests and exact values for the headline quantities |
| `tests_marginals.csv` | two-sample tests and exact values for every marginal |
| `tests_joint.csv` | chi-square homogeneity tests |
| `tests_combined.csv` | Stouffer combinations and weighted mean differences |
| `tests_gof_exact.csv` | per-circuit GOF of the terminal joint law against the exact law |
| `tests_vs_exact_summary.csv` | per-simulator summaries against exact values |
| `confirm.csv` | fresh-seed follow-up of the two most extreme cells |
| `recheck_marginals.csv`, `recheck_summary.csv` | 1e6-shot recheck of all 185 marginals of the m = 22 circuit and its Stim null |

## Revision log

- **2026-10-28, initial version.**
- **2026-10-28, fix round after the adversarial verification**
  (`A-sdim-crosscheck.verify.md`). It confirmed every claim. No simulations
  were rerun and no reported number changed. Fixes:
  - **F1, the S vs S^dagger convention.** Removed the statement that the
    noiseless tests check it. Only the unitary export (self-test b) fixes it.
    Every other gate, the preparation and the measurement are real, so
    complex conjugation hides it from self-test d and from every statistic.
    Changed: Methods 1, R1 row d, claims 1 and 3.
  - **F2, limits of the DEM identity.** It cannot see frame-equivalent
    substitutions (S <-> S^dagger). "To the last bit" refers to the DEM, not
    to the translated circuit's DEPOLARIZE2 arguments 1 - (1 - p). Also
    recorded the verifier's sdim detector-parser caveat (V7 N3) for the
    translator, with the analytic check that this study's expressions are
    unaffected. Changed: Summary, Methods 5, R2, claim 6.
  - **F3, sampler scope.** Claims 7-11 test sdim's Pauli-frame sampler, not
    the production `--simulation-backend dem` sampler. Corrected the
    statement that "the repo's DEM backend uses the same calls": it uses them
    only to compile fault responses. Added claim 11b for the verifier's
    backend check (V4c). Changed: Summary, Methods 3, claim 7, claims-scope
    note.
  - **F4, the data2 floor.** Added the closed form
    [1 - (1 - 8p/3)^3 (1 - 16p/15)] / 2, its small-p limit
    68p/15 = 4.533p (the lead's value), and the first-order breakdown from
    V5. The reported 4.459p and 4.243p are its values at p = 0.005 and 0.02.
    Changed: Summary, R2, R7, claim 14.
  - **F5, claim 13 upgraded from conjecture.** Other sequences and p are now
    verified numerically by the verifier (V1). Other const=k / unif=k
    policies are verified analytically, from the raw-basis DEM identity plus
    the lead's policy test.
