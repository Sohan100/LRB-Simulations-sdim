# A: sdim cross-check, adversarial verification

This is an independent verification of work package A (`notes/A-sdim-crosscheck.md`,
`studies/correction-decoders/qubit/scripts/crosscheck_sdim.py`).

- Script: `studies/correction-decoders/qubit/scripts/verify_crosscheck.py`. It imports nothing from the
  implementer's script.
- Data: `studies/correction-decoders/qubit/results/A-verify/`.
- Scope: `src/lrb` at c21386f6 (identical to `main`, unmodified), sdim 1.3.4,
  Stim 1.15.0. This is the code that generated the mirrored August 2026
  production run. The project memory records a later sdim 1.4.0 port with a
  different SI1000 variant. It is not on `main`, and this verification does
  not cover it.

Labels follow CONTEXT.md: **VA** verified analytically, **VN** verified
numerically.

## Verdicts

| # | Claim (implementer's claim numbers) | Verdict | Required fix |
|---|---|---|---|
| 1 | sdim's P at d = 2 is S and P_INV is S†; `CNOT(a, b)` has control a (1, 2) | **confirmed** (VA, VN) | Wording only. The S/S† convention cannot be observed in any statistic here, and the noiseless tests cannot check it (F1). |
| 2 | sdim's d = 2 tableau agrees with Stim on 300 random circuits (3) | **confirmed**, and extended to mid-circuit M and RESET (VN) | State that this test cannot tell S from S† (F1). |
| 3 | The repo's SI1000 at d = 2 uses the lead's channels; no reset after the terminal round (5) | **confirmed** (VA, VN) | none |
| 4 | 108 repo circuits translated to Stim have DEMs identical to the lead's to the last bit; 6 negative controls flagged (6) | **confirmed**: reproduced byte for byte, and re-derived independently on new sequences, a new p and new depths (VN) | Add two caveats: DEM identity is blind to S↔S†, and "last bit" holds for the DEM, not for the translated circuit (F2). |
| 5, 6 | Sampling tests (rejection, logical flip, marginals, joint) show nothing significant after Holm/BH; 1130 tests (7–11) | **confirmed as stated** (VN). The scope is narrower than the notes imply. | The tests exercise sdim's frame sampler, not the repo's production sampler (`--simulation-backend dem`). This verification now covers that sampler (F3). |
| 8 | The data2 undetectable-logical floor (about 4.5p) also exists in the repo-generated circuits (14) | **confirmed**. Small-p value 68/15 p = 4.533p, identical to the lead's (VA, VN). | Give the small-p value and the closed form. 4.459p and 4.243p are its finite-p values (F4). |
| H1 | A pre-sampled Clifford list longer than max(depths) builds wrong inverses; production `generate_tests` is unaffected (16) | **confirmed** (VA, VN) | none (repo fix R1 suggested) |
| H2 | The unpack specs default to dimension = 3; production is unaffected (17) | **confirmed** (VA, VN) | none (repo fix R2 suggested) |

No claim is refuted, and no bug was found in the lead's model. Four further
latent hazards in `src/lrb` and sdim are listed in V7. None of them touches
the production run.

## V1. Claim 4: independent DEM re-derivation

**Reproduction.** Rerunning the implementer's `crosscheck_sdim.py dem` into a
scratch directory gives `dem_exact.csv` and `dem_negative_controls.csv`
byte-identical to theirs (108/108 identical, 6/6 controls flagged, 19 s).

**Independent construction.** My construction differs from the implementer's
at every step.

- **Code definition.** The qubit `LRBCodeDefinition` is derived
  mechanically from the production registry entry
  (`CodeSimulationProfileRegistry._build_code_definition("folded_qutrit")`).
  Every qutrit template is rebuilt at d = 2, renaming H_INV, CNOT_INV, CZ_INV,
  X_INV and Z_INV to their d = 2 equals. The check blocks, reset block,
  readouts, detector wires and the qutrit terminal X terms (with their −1
  coefficients) are used unchanged. Nothing is written by hand except the
  qubit Clifford catalog, whose inverses use the repo's own
  `SINGLE_QUTRIT_INVERSES`.
- **Generation.** Circuits come from `LRBCircuitGenerator(noise_model="si1000")`
  with depths {0, 3, 7} in one call and exactly max(depths) descriptors, the
  production slicing path. Then `update_noise_param` and
  `ensure_noise_params` are applied.
- **Translation.** It goes by gate id. The detectors come from sdim's own
  parser (`Program._build_ir`): every detector lambda is probed with unit
  vectors and checked for linearity, so the translated detector is what sdim
  evaluates, not what the expression string intends (V7, N3). N2 must equal
  the exact uniform expansion of its `prob` and becomes `DEPOLARIZE2(prob)`.
- **Three comparisons** against `build_lrb_circuit(terminal="data2")` and
  `build_lrb_const0_circuit`:
  1. Per-qubit timelines: every gate and noise channel with its exact float
     probability, equal up to reordering of operations on disjoint qubits.
     This is stronger than DEM identity: it also sees noise that no detector
     sees, and it sees S vs S_DAG.
  2. The DEM in the repo's raw-detector basis (lead circuit body with the
     repo's detectors) and in the lead's difference-detector basis (repo body
     with the lead's detectors). Equal symptoms are XOR-merged, and equality
     is exact.
  3. The production route: `write_circuit` → `read_circuit` →
     `ensure_noise_params`, which is what the runtime actually simulates.

**New sequences**, none equal to the implementer's grid sequences 0–2:

- `new_A` = `ZHP XHPH XZ XHPH XHPH X ZH` (`numpy.default_rng(9170331)`);
- `new_B` = `XHPH XZPH XZH I XP P XHPH`;
- `hand_C` = `XZHPH I ZPH XHP PH HPH XZP`, hand-made to contain every token
  type and a five-token descriptor.

| set | circuits | timelines equal | DEM bit-identical (both bases) | `.chp` route identical | sdim parse = intended |
|---|---|---|---|---|---|
| 3 sequences × p ∈ {0.01, 0.0037} × m ∈ {0, 3, 7} × {data2 LRB, const0} | 36 | 36 | 36 | 36 | 36 |
| all 24 descriptors, m = 1, p = 0.01, {data2, const0} | 48 | 48 | 48 | – | 48 |

The maximum relative difference is exactly 0 everywhere. LRB circuits at
m = 7 have 410–420 merged mechanisms; const0 circuits have 6 (0 at m = 0).
Source: `dem_new_sequences.csv`.

**Negative controls** (`new_A`, m = 7, p = 0.01; `dem_negative_controls.csv`):

| control | kind | flagged by DEM | flagged by timeline | max rel. difference |
|---|---|---|---|---|
| lead meas_flip × 1.001 | data2 | yes | yes | 8.3e-4 |
| lead reset_flip × 0.999 | data2 | yes | yes | 2.3e-4 |
| lead two_qubit × 1.001 | data2 | yes | yes | 1.0e-3 |
| lead meas_depol × 1.001 | data2 | yes | yes | 1.0e-4 |
| lead two_qubit × 1.001 | const0 | yes | yes | 8.5e-4 |
| lead one_qubit × 1.001 | const0 | yes | yes | 9.8e-4 |
| lead terminal = data5 | data2 | yes (measurement sequence differs) | yes | – |
| lead descriptor 3 → H | data2 / const0 | yes / yes | yes / yes | 1 / 0.11 |
| repo: first measurement X_ERROR × 1.001 | data2 | yes | yes | 8.3e-4 |
| repo: P ↔ P_INV everywhere | data2 / const0 | **no** (bit-identical) | yes | 0 |

All 10 rate and structure perturbations are flagged. The last control
demonstrates a blind spot that the implementer's DEM-only comparison shares:
a substitution that acts identically on Pauli frames (S ↔ S†) leaves every
DEM bit-identical. Only the timeline check catches it.

**Bit-level detail.** The implementer translates N2 to
`DEPOLARIZE2(1 − dist[0])`, which is a few ulp away from p (for example
1 − (1 − 0.005) = 0.0050000000000000044). Stim's DEPOLARIZE2 conversion
rounds the difference away, and I checked that the DEM arguments are
bit-identical at all four p used. So "identical to the last bit" is true of
the DEMs but not of the translated circuits. My translation is exact at the
circuit level.

## V2. Claims 1–2: d = 2 gate semantics (`semantics.csv`)

- **Unitaries** (VN). sdim's own gate classes (cirq export) equal Stim's
  unitaries up to phase for all 12 names in that export. SWAP is not in the
  export; its frame rule is a plain swap, and the random replay below covers
  it. P = S and not S_DAG. P_INV = S_DAG. H_INV = H, X_INV = X, Z_INV = Z, CNOT_INV = CNOT,
  CZ_INV = CZ. CNOT(0, 1) has control 0. The tableau convention agrees:
  X → ω^{1/2} XZ = iXZ = Y (`tableau_prime.py`, `phase`).
- **CNOT direction** (VN). In sdim's tableau, `X0; CNOT 0 1` gives (1, 1)
  and `X1; CNOT 0 1` gives (0, 1). The frame rule is `x[t] += x[c]` with
  c = first argument (`program.py:136-138`).
- **Random replay** (VN). I generated 300 random circuits on 2–5 qubits, each
  with 40 random operations from all 13 unitary names (including the `*_INV`
  names), mid-circuit M and RESET, and a final M. The sdim reference shot was
  replayed in Stim's TableauSimulator, post-selecting each random outcome on
  sdim's value. 2580 deterministic and 879 random outcomes agree, with 0
  mismatches.
- **Limit of such tests** (VA, VN). H, X, Z, CNOT, CZ, SWAP, |0⟩ preparation
  and Z measurement are all real. Complex conjugation therefore swaps S and S†
  and leaves every outcome distribution unchanged. Accordingly, the same
  replay with P → S_DAG and P_INV → S also gives 0 mismatches, and the
  P ↔ P_INV control leaves the DEM bit-identical. The convention "P = S" is
  correct, but nothing in this study depends on it.

## V3. Claim 3: noise channels and the terminal round (VA, VN)

**From source:**

- In sdim, N1 'd' draws X^a Z^b uniformly over the d² − 1 nonzero pairs with
  probability `prob`, which is DEPOLARIZE1 at d = 2 (`program.py:638-657`).
  'f' is X^a, which is X_ERROR at d = 2.
- N2 samples from `prob_dist`. `ensure_noise_params` rebuilds that as
  [1 − p, p/15 × 15], which is DEPOLARIZE2.
- The generator inserts noise in `circuit_generator.py:409-514`.
- After the inverse half, the terminal round is the two check blocks followed
  by `terminal_measurement`, with no reset block
  (`circuit_generator.py:1030-1052`). The lead does the same:
  `run_check_round(depth, reset=False)`.

**Numerically**, the per-qubit timelines of all 84 circuits are equal (V1),
including every noise location and its exact probability.

## V4. Claims 5–6: sampling

**(a) Recomputed from the implementer's `counts.csv`.**

- There are 1130 informative tests (104 headline, 958 marginal, 68 joint).
- Every z recomputes to within 5e-7.
- The global Holm and BH minima are both 0.0796, as reported.
- 56 tests have raw p < 0.05, against 56.5 expected.

The statistics machinery is correct. The power is about a 1.5–3% relative
bias per marginal at |z| = 4. That is enough, because the exact DEM identity
already settles model equality, so the sampling tests only test sdim's
sampler.

**(b) sdim's public `Program.simulate`, on new circuits.** I sampled
`new_A` and `new_B` × {data2, const0} × m ∈ {3, 7} at p = 0.01, 5e5 shots
each, against exact DEM values and 4e6 Stim shots (`sample_tests.csv`,
`sample_gof.csv`):

- Against exact values: 132 tests, mean z +0.05, sd 1.05, max |z| 3.02,
  minimum Holm p 0.33.
- Against Stim: max |z| 2.66.
- Terminal joint law: χ² 146.5 on 152 dof (p = 0.61).
- Headline values agree. For example, data2 `new_A` m = 3 terminal acceptance
  is 0.12621 (sdim), 0.12639 (Stim) and 0.12640 (exact).

**(c) The production sampler** (VN, new). The mirrored run used
`run_lrb_experiment.py --simulation-backend dem`
(`run_dem_array_worker.sh` in the run folder), which is
`lrb.dem_simulation.simulate_circuit_with_dem`. That backend uses sdim only
to compile unit-fault responses. It samples faults with its own code
(`dem_simulation.py:428-611`). A's sampling tests never ran it, and the
notes' "the repo's DEM backend uses the same calls" is true only of the
compilation step.

I ran it on the same 8 circuits with 1e6 shots each (`*_repo_dem.csv`):

- Against exact values: 132 tests, mean z +0.03, sd 1.02, max |z| 3.39,
  minimum Holm p 0.091.
- Against Stim: max |z| 3.70.
- Terminal joint law: χ² 174.8 on 152 dof (p = 0.10).

The one extreme cell (const0 `new_A` m = 7, X parity det0, z = −3.39) was
replicated by a pre-registered test with fresh seeds and 4e6 shots: z = +0.70
(p = 0.48), and that circuit's joint law gives χ² 2.78 on 7 dof (p = 0.91).
See `followup_repo_dem.txt`. It was a fluctuation. The production sampler
reproduces the exact DEM law at d = 2.

## V5. Claim 8: the data2 floor (`floor.csv`, `floor_locations.csv`)

The table below covers 9 circuits per p (3 new sequences × m ∈ {0, 3, 7}),
built both by the repo and by the lead. Each p gives exactly one distinct
value, and the repo's value equals the lead's.

| p | silent-logical probability / p | closed form / p |
|---|---|---|
| 1e-7 | 4.533332 | 4.533332 |
| 1e-4 | 4.531840 | 4.531840 |
| 2e-3 | 4.503550 | 4.503550 |
| 5e-3 | 4.459187 | 4.459187 |
| 1e-2 | 4.386076 | 4.386076 |
| 2e-2 | 4.242929 | 4.242929 |

The closed form is [1 − (1 − 8p/3)³ (1 − 16p/15)]/2. It matches to within
3e-15 relative for p ≥ 2e-3. At smaller p, cancellation in 1 − (…) limits
the match to 6.5e-13 (p = 1e-4) and 6.9e-10 (p = 1e-7). Its small-p limit is 68p/15 = 4.5333p, which is the lead's 4.53p. The
implementer's 4.4592p and 4.2429p reproduce to all printed digits; they are
the finite-p values.

At first order the floor comes from 14 circuit components (Stim
`explain_detector_error_model_errors`), summing exactly to 68/15 p:

| location (terminal round) | components (flip the logical, fire no detector) | weight |
|---|---|---|
| DEPOLARIZE1(2p) on data 0, M7 layer | Z0, Y0 | 4p/3 |
| DEPOLARIZE1(2p) on data 0, M8 layer | Z0, Y0 | 4p/3 |
| DEPOLARIZE1(2p) on data 3, M8 layer | Z3, Y3 | 4p/3 |
| DEPOLARIZE2(p) after CX 7→0 | Z0, Y0, X7·Z0, X7·Y0 | 4p/15 |
| DEPOLARIZE2(p) after CX 8→3 | Z3, Y3, X8·Z3, X8·Y3 | 4p/15 |

Data 2 contributes nothing, and 8p/15 (12% of the floor) is gate noise after
the coupling CNOTs, not idle noise. API.md's "idle depolarizing on data 0, 2
or 3" is therefore loose; the L0 note's wording is right. See L1.

## V6. The two latent hazards (VA from source, VN demonstrated)

**H1. A pre-sampled list longer than max(depths)** (`hazards.csv`).

- **Mechanism.** The inverse list is built from the whole list and reversed
  (`circuit_generator.py:919-923`, `1102-1106`), then sliced with
  `[(max_depth - depth):]` (`948-949`, `1120-1121`). The only guard rejects
  lists that are too short (`912`, `1095`).
- **Demonstration.** `depths=[3]` with 7 descriptors builds 11 SWAPs instead
  of 6, for both LRB and const0. The noiseless circuit's detectors or logical
  become non-deterministic: Stim refuses to build a DEM.
- **Why production cannot hit it.** The only callers are in `generate_tests`
  (`1245-1258`), which passes `generate_random_clifford_strings(max(depths))`.
  That is exactly `sorted(depths)[-1]` fresh descriptors.
  `generate_tests` is called only from `experiment_setup.py:453`. A grep of
  `src/`, `scripts/`, `tests/`, `slurm/` and the notebooks finds no other
  caller. The run folder's `depths.txt` is `0,2,4,6,10,14,18,20,22`.

**H2. Unpack dimension default of 3.**

- **Mechanism.** The defaults are `_UnpackSpec.dimension = 3`
  (`code_simulation_profiles.py:89`), `LRBUnpackSpec.dimension = 3`
  (`lrb_simulation.py:1086`) and `extract_lrb_counts(dimension=3)`
  (`lrb_simulation.py:2427`). The dimension is used for the reference
  reconstruction at `lrb_simulation.py:1538-1561` and `1619-1664`.
- **Demonstration.** At d = 2, a reference shot with (x0, x3) = (1, 1) gives
  logical value 2 under the default and 0 with `dimension=2`. With (0, 0),
  both give 0.
- **Why production cannot hit it.** All three registered profiles are qutrit
  (`code_simulation_profiles.py:138-365`), and the run's `code_name.txt` is
  `folded_qutrit`. The production call passes `dimension=dimension` from
  `logical_dimension` (`lrb_simulation.py:2734-2736`).
  `default_stabilizer_pass_check`'s `% 3` is harmless for qubit values.

## V7. Further latent hazards found here (none affects the production run)

- **N1. Depth order.** The generator iterates over `sorted(depths)` and
  `generate_tests` names files by enumeration index
  (`circuit_generator.py:907, 1283`). The runtime reads file k as `depths[k]`
  and takes `max_depth = depths[-1]` (`lrb_simulation.py:1954, 2004, 2038`).
  An unsorted `depths` configuration would mislabel every circuit.
  Production depths are sorted.
- **N2. Aliases bypass the SI1000 insertion.** `apply_si1000_noise_to_circuit`
  matches on the alias-sensitive `gate_name`. A code definition written with
  sdim aliases gets no noise for `S` (alias of P), `MZ` (of M, 0 vs 3 noise
  operations) or `MR` (of RESET, 0 vs 2). `CX` is caught by the two-qudit
  fallback. sdim also maps `R` to the Hadamard, not to reset. Stim-style names
  in a future code definition would therefore fail silently.
- **N3. sdim's detector parser** (`program.py:704-714`). It replaces `[k]`
  placeholders with sequential `str.replace`, so `rec[1]+rec[0]` is evaluated
  as the constant 0 (demonstrated). My translator compares sdim's evaluation
  with the intended expression for every annotation. They agree for all 120
  circuits here, and the production qutrit expressions have increasing or
  large indices, so they are safe. The implementer's translation parses the
  intended expression and would not notice this failure mode.
- **N4. `.chp` round trip.** `read_circuit` returns every N2 with sdim's
  default `prob_dist`, a uniform 1/16 ndarray: a 15/16 two-qubit depolarizing
  channel (demonstrated). `ensure_noise_params` and `ensure_dem_noise_params`
  replace it only because they test `isinstance(..., (list, tuple))`
  (`lrb_simulation.py:1021-1024`, `dem_simulation.py:642-644`).
  Production calls one of them, and the round trip is exact (V1). Any other
  consumer of the `.chp` files would simulate p₂ = 15/16 without warning.

## Required fixes

**In A's notes:**

- **F1** (claims 1, 2, 3; Methods §1). Replace "It would only change the
  logical action, and that is checked by the noiseless tests" with this:
  every other gate, the preparation and the measurement are real, so complex
  conjugation swaps S and S† and leaves every statistic unchanged. Only the
  unitary export (self-test b) fixes the convention. Self-test d and the
  noiseless tests cannot see it.
- **F2** (claim 6, R2). Add two caveats:
  - DEM identity is insensitive to substitutions that act the same on Pauli
    frames (S ↔ S†).
  - "To the last bit" refers to the DEM probabilities. The translated
    DEPOLARIZE2 arguments are 1 − (1 − p).
- **F3** (claims 7–11, Methods §3). State that the sampling tests exercise
  sdim's frame sampler, not the production `--simulation-backend dem`
  sampler, and correct "the repo's DEM backend uses the same calls". Cite V4c
  for the backend check.
- **F4** (claim 14, R2, Summary). Give the small-p floor 68p/15 = 4.533p,
  equal to the lead's value, and the closed form above. State that 4.459p and
  4.243p are its values at p = 0.005 and 0.02.
- **F5** (claim 13, optional). The "other sequences and p" part is now
  verified numerically (V1). "Other policies" follows analytically: the
  policies are functions of the raw flips, whose joint law with the logical
  is identical (raw-basis DEM identity), and the lead's `used_rounds` is
  tested against the repo's functions.

**For the lead (report only):**

- **L1.** In API.md's floor bullet, replace "data 0, 2 or 3" with data 0 and
  3, and note that 8p/15 of the 68p/15 floor is DEPOLARIZE2 after CX 7→0 and
  CX 8→3.

**For the repo (report only; `src/lrb` was not edited):**

- **R1.** Truncate or reject `clifford_strings` longer than max(depths).
- **R2.** Make the unpack `dimension` field required.
- **R3.** Assert that `depths` is sorted, or name circuit files by depth.
- **R4.** Match the SI1000 rules on the canonical `op.name` or `gate_id`.
- **R5.** Always rebuild N2's `prob_dist` from `prob` when `prob` is present.
- **R6** (upstream sdim). Use single-pass placeholder substitution.

## Reproduce

```bash
cd <repository root>
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
V=studies/correction-decoders/qubit/scripts/verify_crosscheck.py
nice -n 19 python $V dem         # 12 s: V1
nice -n 19 python $V floor       # V5
nice -n 19 python $V semantics   # V2
nice -n 19 python $V hazards     # V6, V7
nice -n 19 python $V followup    # 14 s: V4c replication
JOB=$(squeue -u sohan100 -n q512shared -h -o %i | head -1)
srun --jobid=$JOB --overlap -N 1 -n 1 -c 16 bash -c "module load python/3.11; cd $PWD; \
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src; \
  python $V sample --workers 16 --chunks 25 --chunk-shots 20000 --stim-shots 4000000; \
  python $V sample --backend repo_dem --workers 16 --chunks 25 --chunk-shots 40000 --stim-shots 4000000"
# The implementer's DEM step, into a scratch directory (byte-identical CSVs):
python studies/correction-decoders/qubit/scripts/crosscheck_sdim.py dem --out <scratch-dir>
```

The two `sample` runs took 137 s and 77 s on 16 cores of allocation 60478392.
