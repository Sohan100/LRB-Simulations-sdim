# Qubit [[5,1,2]] model: API and verified facts (for agents)

Owned by the lead session. Do not edit these modules; if you find a bug, report
it in your notes and structured output, and work around it in your own module.

```
module load python/3.11
export PYTHONPATH=$PWD/src   # from the repository root
```

## Modules (`src/lrb_qubit/`)

- `code.py`: wires, stabilizers (`STABILIZER_PAULIS`, order S_Z1, S_Z2, S_X1, S_X2 on
  ancillas 5, 6, 7, 8), `LOGICAL_X/Z`, `SINGLE_QUBIT_CLIFFORDS` (24 descriptors, the
  rightmost symbol acts first, `P` = S), `logical_clifford_operations(desc, inverse)`,
  `check_round_operations()`, `encoding_operations()`, `data_tableau(ops)`,
  `decompose_in_stabilizers(pauli)`. An operation is `(stim_gate_name, targets)`.
- `circuits.py`:
  - `SI1000` scale factors (`one_qubit=0.1, two_qubit=1, meas_idle=2, meas_depol=1,
    meas_flip=5, reset_idle=2, reset_flip=2`; `.scaled(meas_idle=0.0)` for ablations).
  - `emit_operations(circuit, ops, p, noise, num_qubits, noisy_measurements)`: the
    SI1000 noise insertion (consecutive M = one layer, consecutive R = one layer).
    Reuse it for circuit variants.
  - `build_lrb_circuit(cliffords, depth, p, terminal="data5"|"data2",
    terminal_noise=False, forward_check_rounds=None, terminal_check=True,
    noise=SI1000(), inverse_check_every=0) -> LRBCircuit`.
  - `LRBCircuit`: `.circuit` (stim, with detectors and observable 0 = X_L flip),
    `.rounds` (round labels in time order: 0..m-1 forward, m terminal),
    `.detector_keys` [(round, stabilizer)], with final data detectors labelled
    m+1 and stabilizers 2, 3, `.back` (back-propagated generator subsets),
    `.raw_syndromes(dets) -> (shots, rounds, 4)` raw stabilizer flips.
    Detectors are *difference* detectors: (round r, stabilizer s) compares with
    the previous physical round's product of generators equal to s after
    propagation through the logical gates in between.
  - `build_lrb_const0_circuit`, `build_physical_rb_circuit`,
    `random_clifford_sequence(n, rng)`.
- `baseline.py`: `used_rounds(policy, k, depth)` (identical to the qutrit
  `postselect_*` functions, tested), `accept_mask(raw, rounds, used)`,
  `CountTable` (per sequence x depth counts, `.summary()` mean/std over sequences
  as in the repo CSVs), `fidelity_from_counts`, `write_repo_stats`.
- `mld.py`: `TransferMatrixMLD(dem, lrb_detector_slots(lc))` exact maximum-likelihood
  decoder; `.log_likelihood_ratio(dets)` = log P(L=0|s) - log P(L=1|s) (soft output,
  exact), `.decode(dets) -> (predicted_flip, gap)`; `brute_force_joint(dem)` for small
  DEMs. About 1.2 s per 1e5 shots at depth 22 on one core.
- `experiment.py`: `clifford_sequences(n, max_depth, base_seed)`,
  `run_point(p, sequences, depths, shots, decode_shots=...) -> PointResult`
  (all LRB-D policies, `_final` variants, `mld5`, `mld2`, `const0`, `const0_mld`,
  and gap histograms with `GAP_EDGES`), `run_physical_rb`. `PointResult.from_npz`.
- `studies/correction-decoders/qubit/scripts/run_grid.py`: process-pool grid runner (see its docstring).
- `tests/test_lrb_qubit_model.py`: 40 tests, all passing.

## Main grid results

`studies/correction-decoders/qubit/results/grid_main/` (being produced now):
`p<idx>.npz` (load with `PointResult.from_npz`), `rb<idx>.npz` (physical RB at
p_RB = 5p), `csv/<idx>/<method>.csv` in the repo's five-row format, `meta.json`
(30 sequences from seed 20261028, depths 0,2,4,6,10,14,18,20,22, 1e6 shots for
LRB-D, 3e5 decoded shots).

## Verified facts so far (lead session)

- Verified numerically: the encoder prepares |+>_L; the 24 descriptors are the
  full single-qubit Clifford group; logical H, S, X, Z map stabilizers to
  stabilizers with + signs and act as H, S, X, Z on (X_L, Z_L); every
  Clifford's inverse sequence inverts it exactly.
- Verified numerically: noiseless circuits give zero detection events and zero
  observable flips at every depth, and Stim accepts every detector as
  deterministic.
- Verified numerically (Stim `search_for_undetectable_logical_errors`): with the
  five-qubit terminal readout the circuit-level distance is 2 (no single fault
  is an undetected logical error).
- Verified numerically: with the qutrit pipeline's two-qubit terminal readout
  (data 0 and 3 only) there ARE single faults that flip the logical outcome with
  no detection event: Z-type faults that reach data 0 or 3 after their last
  X-check coupling in the terminal round (idle depolarizing during the X-ancilla
  measurement layers, plus two CNOT fault locations). Total probability is
  68p/15 = 4.53p (9.0e-3 at p = 2e-3); 48p/15 with one parallel measurement
  layer (work package H). This is a
  first-order, depth-independent SPAM-like floor on LRB-D accepted fidelity.
- Verified numerically: the transfer-matrix MLD equals brute-force exact
  inference to about 1e-14 in the LLR at depths 0-4.
- Verified numerically: the repo's raw-syndrome postselection is reproduced
  exactly (`test_policies_match_qutrit_pipeline`, `test_raw_syndromes_match_measurement_records`).
- First look (4 sequences, 2e4 shots, p = 2e-3): LRB-D unif=1 accepted F about 0.98
  (0.996 with the five-qubit readout) but rejection 11% at m=0 rising to 99% at
  m=22; full MLD correction F = 0.956 at m=0 falling to 0.30 at m=22; physical RB
  at 5p F = 0.865 at m=22; no-check `const0` decoded F = 0.94 at m=22.

## IMPORTANT: BLAS threads

A site hook in the user's Python environment imports numpy at interpreter
startup, and numpy's OpenBLAS then starts 64 threads per process. Setting
thread variables inside a Python script is too late. Always launch Python with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` exported in the
shell (forked multiprocessing workers inherit it), or re-exec the interpreter
with them set as `studies/correction-decoders/qubit/scripts/run_grid.py` does. Do NOT call
`threadpoolctl.threadpool_limits` inside forked workers: the startup hook also
loads numba, and threadpoolctl aborts the worker. Without single-threaded BLAS
the small matrix products in the MLD decoder run about 7x slower and overload
the node.
