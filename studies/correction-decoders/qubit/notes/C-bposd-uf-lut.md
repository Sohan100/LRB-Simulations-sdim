# WP-C: BP+OSD, BP+LSD, union-find and lookup-table decoders vs exact MLD

Work package C of the qubit `[[5,1,2]]` LRB decoder study. Every number below
comes from `studies/correction-decoders/qubit/results/C/` (CSV files named in
each section; file paths in this note are relative to that directory). Units: LER and excess LER in percentage points (pp) unless
stated; "±" is one standard error.

## Summary

Setup: 200 `data5` LRB circuits (10 Clifford sequences x depths 0, 2, 6, 14,
22 x p = 5e-4, 2e-3, 5e-3, 1e-2). Each decoder is scored against exact MLD
(`TransferMatrixMLD`) on identical shots, by logical error rate, agreement
and runtime. A sampling-free first-order (single-fault) analysis supports the
sampled results. All results below are **verified numerically** unless
labelled otherwise.

**Headline.** Nothing beats exact MLD, and the decoder families split
sharply:

1. **BP+OSD and BP+LSD (ldpc 2.4.1)** come close to MLD only with two
   non-default choices:
   - check-matrix columns that keep each mechanism's logical class;
   - a BP stage that cannot short-circuit OSD/LSD (serial schedule or
     `always_run_lsd`).

   All BP runs here use product-sum BP. That is ldpc 2.4.1's actual default,
   although its docstring says `minimum_sum` (found by the verifier).

   With both choices, BP+OSD-CS7 stays within about 0.7-0.8 pp of MLD
   across the grid, and BP+LSD-CS7 within about 1.4 pp; both are
   first-order exact to <= 2 p.
   - BP+OSD-CS7 worst cell: 0.68 ± 0.05 pp (between-sequence SE); 0.83 ±
     0.07 pp on the verifier's 10 new sequences.
   - BP+LSD-CS7 worst cell: 1.35 ± 0.12 pp; new sequences 1.46 ± 0.09 pp.

   Where the defaults or other choices fall short:
   - Product-sum flooding (parallel) BP converges on 5-6% of unambiguous
     single-fault syndromes to a wrong-class solution, and OSD/LSD never
     run. At p = 5e-4, m = 22 this costs 4.75 ± 0.39 pp (MLD LER 12.3%), and
     208 ± 14 p of first-order LER against MLD's 272 p floor.
   - Min-sum flooding BP fails the same way on 1.3-4.1% of those syndromes.
     It costs less at p = 5e-4 (0.71 pp at m = 22 for scaling 1) but is
     1.2-1.4 pp worse than product-sum at p >= 2e-3 (verifier).
   - With ldpc's full defaults (OSD-0, max_iter = n), 10-13% of unambiguous
     single-fault syndromes get the wrong class (verifier).
   - The DEM conversion used by `SinterBpOsdDecoder` keeps the last-listed
     mechanism's logical class for merged detector sets. That costs
     206 ± 5 p at first order and 8.8 ± 0.6 pp sampled at the worst cell,
     of which about 8.0 pp is the overwrite itself: correctly labelled
     merged ("dominant") columns give 0.77 ± 0.09 pp there.
2. **Union-find is far from MLD at d = 2.** This holds for textbook
   Delfosse-Nickerson UF on the decomposed graph, ldpc's matrix-solve UF on
   graph or hypergraph, and BP+UF: 1.2-9.5 pp excess at the informative
   points, and 221-575 p first-order excess at m = 22. MWPM on the same graph
   (control) is also 0.9-8.5 pp off. ldpc's peeling UF hung on a valid
   syndrome and could not be used.
3. **A sliding-window LUT computed from exact window marginals** works at
   every depth. Its tables have a 4w-bit address and 5-bit words; it does one
   lookup and two XORs per round and has w - 1 rounds of latency.
   - It equals MLD when the window covers the circuit.
   - Every w >= 2 is first-order exact as p -> 0. Checked at p = 1e-5
     (excess <= 1e-6 p, the same near-tie residual as MLD). At p = 5e-4,
     w = 2 shows a 0.10 p finite-p effect at m = 22.
   - Its excess over MLD is about 0.84 pp for w = 3 (52.6 KiB of tables per
     depth-22 circuit) and about 0.56 pp for w = 4 (802 KiB). Worst grid
     cells: 0.82 ± 0.03 and 0.54 ± 0.03 pp (between-sequence SEs); the
     verifier's new sequences give 0.87 ± 0.02 and 0.58 ± 0.02 pp.
   - Batch throughput on one core: LUT 0.03-0.7 us/shot, vectorized MLD
     1.4-14 us/shot, BP+OSD-CS7 with syndrome de-duplication
     18 us-7 ms/shot.
   - Single-shot latency is a different measure. In Python the LUT takes
     0.13-1.1 ms, MLD 0.23-1.2 ms and BP+OSD-CS7 0.3-8.6 ms per call on
     random shots. That is only 2.4-7.6x in the LUT's favour (verifier:
     3.5-11x).

   It is the natural FPGA-class real-time decoder here, but nanosecond
   hardware lookups are a **conjecture**. For offline LRB decoding, exact
   MLD is cheap enough and should stay the reference.
4. **No bug was found in the lead's model.** An independent exact-inference
   implementation (the LUT with w >= T) matches `TransferMatrixMLD` to
   3e-15. The MLD posterior is calibrated against the circuit sampler at all
   20 points (chi-square 17.0 for 20 degrees of freedom). Five ldpc 2.4.1
   issues are documented under Bugs.

## Methods

### Circuits, shots and reference

- Circuits: `circuits.build_lrb_circuit(seq, m, p, terminal="data5")` (SI1000,
  ideal five-qubit X readout, observable = `x0 + x3`), sequences
  `experiment.clifford_sequences(10, 22, 20261028)` (indices 0-9),
  depths m in {0, 2, 6, 14, 22}, p in {5e-4, 2e-3, 5e-3, 1e-2}; 200 circuits.
- Reference: `mld.TransferMatrixMLD` on the full DEM (all `4(m+1)+2`
  detectors; the lead's `mld5` setting). All decoders see the same detection
  events, sampled from the circuit with seed
  `31415 + 100000 i_p + 1000 i_seq + i_m` (indices into the lists above).
- MLD and the LUTs decode all 50,000 shots of each circuit. The ldpc-family
  decoders (and the controls) decode the longest shot prefix that contains at
  most U distinct non-zero syndromes, because they run shot by shot from
  Python. The prefix depends only on the syndromes, never on any decoder's
  output. It is still a syndrome-dependent stopping time: it excludes the
  next shot, which is always a new and usually heavier syndrome. That bias is
  O(1/N) and negligible; the verifier's fixed-prefix runs reproduce the
  headline cells. Compute limits forced four
  stages (`meta.json` field `stages`):

  | stage | circuits | U |
  |---|---|---|
  | 1 | 24 (p = 1e-2 at m = 14, 22; p = 5e-3 at m = 22, sequences 0-3) | 2000 |
  | 2a | 20 (p = 1e-2 at m = 6; p = 5e-3 at m = 14) | 800 |
  | 2b | the other 156 | 300 |
  | 3 | all 200 | 300 |

  Stage 3 refilled every (circuit, decoder) pair missing from earlier
  stages, including the ablation decoders, with the same sampler seeds; when
  a pair exists twice the earlier row wins. Every cell of the tables
  therefore has all 10 sequences, but statistical precision varies by stage.
  Every comparison with MLD uses MLD's decisions on exactly the decoder's
  shots.
- Each decoder's settings were chosen on held-out sequences 10 and 11 (not in
  the evaluation set); `tuning.csv`.
- The product-sum versus min-sum choice comes from an earlier scan on
  evaluation sequence 2 (p = 2e-3, m = 6; rows
  `depth6_p2e-3_evalseq2_first_scan` in `tuning.csv`). There product-sum
  OSD-CS7 beat min-sum with scaling 0.625: 0.0053 ± 0.0008 vs
  0.0113 ± 0.0014 excess. The verifier found that this holds at p >= 2e-3.
  At p = 5e-4 flooding min-sum is better than flooding product-sum
  (0.71 vs 4.81 pp at m = 22), though both lose to serial product-sum
  (0.09 pp).

### Decoders

All decoders output a predicted flip of observable 0 from the full syndrome.

| name | family | details |
|---|---|---|
| `mld` | exact | `TransferMatrixMLD` (reference) |
| `bposd_cs7` | BP+OSD | ldpc `BpOsdDecoder`, product-sum BP (ldpc 2.4.1's actual default; its docstring says `minimum_sum`), parallel (flooding) schedule, 5 iterations, OSD-CS order 7; columns = one per DEM mechanism (`"mechanism"`) |
| `bposd_cs7_serial` | BP+OSD | same, serial BP schedule |
| `bposd_0` | BP+OSD | parallel BP, OSD-0 |
| `bposd_cs7_dominant` | BP+OSD | `bposd_cs7` on `"dominant"` columns: one column per detector set, prior = XOR-sum, observable of the more probable class |
| `bposd_cs7_ldpcmatrices` | BP+OSD | `bposd_cs7` on ldpc's own `detector_error_model_to_check_matrices` (the `SinterBpOsdDecoder` path) |
| `bplsd_cs7` | BP+LSD | ldpc `BpLsdDecoder`, parallel BP (5 it.), LSD-CS order 7, `bits_per_step=1` |
| `bplsd_cs7_always` | BP+LSD | same with `always_run_lsd=True` |
| `belief_find` | BP+UF | ldpc `BeliefFindDecoder`, parallel BP (5 it.), matrix-solve UF |
| `uf_dn` | UF | `decoders_ldpc.DelfosseNickersonUF`: textbook union-find (half-edge growth, boundary, peeling on a most-probable-first spanning forest) on the decomposed matching graph |
| `uf_graph` | UF | ldpc `UnionFindDecoder`, matrix-solve mode, weighted growth (lowest LLR first, one column per step) on the decomposed graph |
| `uf_hyper` | UF | same on the undecomposed hypergraph (`"mechanism"` columns) |
| `mwpm_graph` | control | PyMatching on the same decomposed graph (not a WP-C decoder; separates graph losses from UF losses) |
| `lut_w{w}_{joint,carry}` | LUT | `decoders_lut.SlidingWindowLUT`, window w = 1..4 (below) |

Check matrices (`decoders_ldpc.dem_check_matrices`): H has one row per
detector; L is the observable row; priors are the mechanism probabilities
(exact duplicates XOR-merged). The decomposed graph
(`graph_check_matrices`) uses Stim's `decompose_errors=True` DEM; each
graphlike component becomes an edge carrying its mechanism's probability,
parallel edges are XOR-merged and keep the observable of the more probable
class. Stim's decomposition succeeded for every circuit in the grid.

### Sliding-window LUT

Detectors are grouped into time slots (one per check round, plus the final
data slot; T = m + 2 slots). Every DEM mechanism touches one slot or two
adjacent slots. Grouping mechanisms by their *earliest* slot u gives exact
kernels `A_u[x, y, l]` = P(group u flips slot-u bits x, slot-(u+1) bits y,
observable l), by XOR-convolution.

Running state: a 4-bit carry kappa (slot-t bits already explained by
committed mechanisms) and a 1-bit logical frame f.

- Interior step t = 0 .. T-w-1. Inputs: sigma = s_t XOR kappa and the
  look-ahead syndromes s_{t+1} .. s_{t+w-1}. Output:
  (kappa', lambda) = argmax A_t[sigma, kappa', lambda] B_{t+1}(kappa'), where
  B_{t+1} is the exact probability of the look-ahead syndromes given kappa'.
  B_{t+1} sums over the later groups in the window, their observable bits and
  the unseen slot t+w. Then f ^= lambda and kappa = kappa'. This is the
  `joint` rule; `carry` first maximizes over kappa' with lambda summed out.
- Final step t_f = max(T-w, 0): flip f if the window posterior of the parity
  of all remaining observable bits favours 1. This is exact marginal MAP on
  the last window.

Every step is one table lookup. An interior table has 2^(4w) entries
(address = sigma and the look-ahead bits) of 5 bits (kappa', lambda). The
final table has 2^(4(w-1)+2) one-bit entries. Tables are computed offline
from the DEM by an exact backward recursion. For w >= T there is a single
step over the whole syndrome and the LUT *is* the exact MLD: its final-step
LLR equals `TransferMatrixMLD`'s to 1e-15, tested for m = 0, 1, 2, both
terminals and p = 2e-3 and 3e-2. In this grid that means w = 2 at m = 0 and
w = 4 at m <= 2. For w < T the hard commits and the finite look-ahead are
the approximations being measured.

### Statistics

For each circuit and decoder the runner stores paired counts on identical
shots: n10 (decoder wrong, MLD right), n01 (decoder right, MLD wrong), and
disagreements. It also stores two posterior-weighted (Rao-Blackwell) sums
that use MLD's exact posterior P(L | s):

- `ler_rb` = mean over shots of P(L != decision | s);
- `delta_ler_rb` = mean over shots of [decision != MLD] tanh(|LLR|/2). This
  is the expected excess error of the decoder over MLD on the sampled
  syndromes.

Both are unbiased because the DEM is the circuit's exact error model
(**verified analytically**: tower property; **verified numerically**: the
MLD calibration table below).
They have much lower variance than counting flips, because they remove the
noise of the logical outcome given the syndrome. Per (p, m) the quantities
are averaged over the 10 sequences with equal weight, as the repo does for F.
Quoted SEs are shot-noise SEs conditional on the 10 sequences, combined over
sequences. `summary.csv` also lists the raw paired difference
`delta_ler = (n10 - n01)/N` with its paired SE, the between-sequence SE of
the LER, and agreement with MLD.

Runtime: the numbers quoted come from a separate controlled benchmark
(`timing.csv`, section 4), not from the grid. Stages 1-2 ran with 24 worker
processes on the shared node, where `srun --overlap` steps from different
agents were pinned to the same few CPUs, so their `us_per_call`/`us_per_shot`
columns are contention-dominated. Stage 3 used `--cpu-bind=none`. In the
benchmark, MLD, the LUTs and PyMatching are timed per shot on vectorized
50,000-shot batches; the ldpc decoders and `uf_dn` are timed per call on
distinct non-zero syndromes. All decoders here are deterministic functions of
the syndrome, so de-duplicating syndromes lowers the effective cost per shot
(`summary.csv` column `us_per_shot`).

## Results

### 1. Sampled comparison on identical shots (`summary.csv`)

Columns: MLD's LER (mean over sequences ± between-sequence SE) and, for each decoder, the posterior-weighted excess LER over MLD (`delta_ler_rb`, in pp). The ± on each excess is the shot-noise SE *conditional on these 10 sequences*. Between-sequence SEs, which also contain sequence-to-sequence variation and are the right error bar for "any random sequence", are in `between_sequence_se.csv`; Table 1d lists them for the worst cells. All cells average 10 sequences. MLD and LUT columns use 50,000 shots per circuit (500,000 per point). The ldpc-family columns use the syndrome-capped prefix: 3,021 to 500,000 shots per point for `bposd_cs7`.

**Table 1a. BP+OSD, BP+LSD, union-find and the matching control: excess LER over MLD (pp).**

| p | m | MLD LER (%) | `bposd_cs7` | `bposd_cs7_serial` | `bplsd_cs7` | `bplsd_cs7_always` | `uf_dn` | `uf_hyper` | `mwpm_graph` |
|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | 0 | 0.53 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.09 ± 0.00 | 0.08 ± 0.00 | 0.08 ± 0.00 |
| 0.0005 | 2 | 1.83 ± 0.09 | 0.71 ± 0.01 | 0.01 ± 0.00 | 0.72 ± 0.01 | 0.02 ± 0.00 | 1.24 ± 0.02 | 1.27 ± 0.02 | 0.91 ± 0.01 |
| 0.0005 | 6 | 4.15 ± 0.11 | 2.01 ± 0.08 | 0.03 ± 0.00 | 2.06 ± 0.08 | 0.08 ± 0.01 | 2.89 ± 0.09 | 2.71 ± 0.08 | 2.31 ± 0.08 |
| 0.0005 | 14 | 8.31 ± 0.15 | 3.87 ± 0.22 | 0.08 ± 0.01 | 3.96 ± 0.22 | 0.14 ± 0.03 | 5.09 ± 0.23 | 5.13 ± 0.23 | 4.56 ± 0.22 |
| 0.0005 | 22 | 12.27 ± 0.16 | 4.75 ± 0.30 | 0.10 ± 0.02 | 4.96 ± 0.30 | 0.33 ± 0.06 | 7.27 ± 0.35 | 8.76 ± 0.37 | 6.10 ± 0.32 |
| 0.002 | 0 | 2.21 ± 0.02 | 0.02 ± 0.00 | 0.01 ± 0.00 | 0.02 ± 0.00 | 0.01 ± 0.00 | 0.32 ± 0.01 | 0.30 ± 0.01 | 0.30 ± 0.01 |
| 0.002 | 2 | 7.39 ± 0.31 | 0.13 ± 0.01 | 0.10 ± 0.01 | 0.25 ± 0.02 | 0.21 ± 0.01 | 4.02 ± 0.09 | 3.92 ± 0.09 | 3.00 ± 0.08 |
| 0.002 | 6 | 15.66 ± 0.28 | 0.28 ± 0.04 | 0.24 ± 0.03 | 0.57 ± 0.06 | 0.51 ± 0.05 | 7.39 ± 0.30 | 6.64 ± 0.27 | 6.12 ± 0.27 |
| 0.002 | 14 | 27.29 ± 0.36 | 0.40 ± 0.06 | 0.30 ± 0.03 | 1.03 ± 0.10 | 0.96 ± 0.10 | 9.50 ± 0.42 | 8.30 ± 0.38 | 8.53 ± 0.40 |
| 0.002 | 22 | 35.22 ± 0.25 | 0.54 ± 0.08 | 0.42 ± 0.04 | 1.07 ± 0.11 | 0.92 ± 0.09 | 7.98 ± 0.38 | 8.18 ± 0.37 | 7.51 ± 0.36 |
| 0.005 | 0 | 5.37 ± 0.02 | 0.06 ± 0.00 | 0.04 ± 0.00 | 0.04 ± 0.00 | 0.04 ± 0.00 | 0.81 ± 0.01 | 0.69 ± 0.01 | 0.70 ± 0.01 |
| 0.005 | 2 | 17.67 ± 0.59 | 0.39 ± 0.03 | 0.39 ± 0.03 | 0.83 ± 0.05 | 0.87 ± 0.05 | 6.80 ± 0.21 | 6.19 ± 0.19 | 5.13 ± 0.18 |
| 0.005 | 6 | 32.61 ± 0.35 | 0.70 ± 0.05 | 0.68 ± 0.05 | 1.27 ± 0.08 | 1.34 ± 0.09 | 7.95 ± 0.33 | 6.16 ± 0.28 | 6.76 ± 0.30 |
| 0.005 | 14 | 44.64 ± 0.18 | 0.50 ± 0.02 | 0.52 ± 0.02 | 0.85 ± 0.04 | 0.85 ± 0.04 | 3.68 ± 0.12 | 2.82 ± 0.10 | 3.32 ± 0.11 |
| 0.005 | 22 | 48.35 ± 0.10 | 0.23 ± 0.02 | 0.22 ± 0.02 | 0.39 ± 0.03 | 0.39 ± 0.03 | 1.21 ± 0.08 | 1.24 ± 0.08 | 1.27 ± 0.08 |
| 0.01 | 0 | 10.41 ± 0.05 | 0.11 ± 0.00 | 0.12 ± 0.00 | 0.06 ± 0.00 | 0.06 ± 0.00 | 1.42 ± 0.01 | 1.03 ± 0.01 | 1.07 ± 0.01 |
| 0.01 | 2 | 30.91 ± 0.70 | 0.71 ± 0.04 | 0.68 ± 0.04 | 1.34 ± 0.07 | 1.35 ± 0.07 | 6.38 ± 0.24 | 5.35 ± 0.20 | 5.44 ± 0.21 |
| 0.01 | 6 | 45.44 ± 0.26 | 0.52 ± 0.02 | 0.50 ± 0.02 | 0.88 ± 0.03 | 0.88 ± 0.03 | 2.80 ± 0.08 | 1.96 ± 0.06 | 2.86 ± 0.08 |
| 0.01 | 14 | 49.76 ± 0.09 | 0.09 ± 0.00 | 0.08 ± 0.00 | 0.13 ± 0.01 | 0.12 ± 0.01 | 0.27 ± 0.01 | 0.18 ± 0.01 | 0.26 ± 0.01 |
| 0.01 | 22 | 49.90 ± 0.05 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.02 ± 0.00 |

**Table 1b. Sliding-window LUTs: excess LER over MLD (pp).** Cells with w >= T (w >= 2 at m = 0, w = 4 at m = 2) are exactly 0, because the LUT is then the exact MLD. Other 0.00 entries are rounding.

| p | m | `lut_w1_joint` | `lut_w2_joint` | `lut_w3_joint` | `lut_w4_joint` | `lut_w2_carry` | `lut_w3_carry` |
|---|---|---|---|---|---|---|---|
| 0.0005 | 0 | 0.05 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0.0005 | 2 | 0.38 ± 0.01 | 0.03 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.03 ± 0.00 | 0.00 ± 0.00 |
| 0.0005 | 6 | 0.67 ± 0.01 | 0.09 ± 0.00 | 0.02 ± 0.00 | 0.01 ± 0.00 | 0.10 ± 0.00 | 0.03 ± 0.00 |
| 0.0005 | 14 | 1.24 ± 0.01 | 0.22 ± 0.01 | 0.06 ± 0.00 | 0.04 ± 0.00 | 0.24 ± 0.01 | 0.08 ± 0.00 |
| 0.0005 | 22 | 1.64 ± 0.02 | 0.31 ± 0.01 | 0.09 ± 0.00 | 0.06 ± 0.00 | 0.34 ± 0.01 | 0.12 ± 0.00 |
| 0.002 | 0 | 0.32 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0.002 | 2 | 1.24 ± 0.01 | 0.28 ± 0.01 | 0.05 ± 0.00 | 0.00 ± 0.00 | 0.33 ± 0.01 | 0.06 ± 0.00 |
| 0.002 | 6 | 2.41 ± 0.02 | 0.86 ± 0.01 | 0.31 ± 0.00 | 0.20 ± 0.00 | 0.99 ± 0.01 | 0.38 ± 0.00 |
| 0.002 | 14 | 3.53 ± 0.02 | 1.42 ± 0.01 | 0.56 ± 0.01 | 0.40 ± 0.00 | 1.60 ± 0.01 | 0.71 ± 0.01 |
| 0.002 | 22 | 3.39 ± 0.02 | 1.47 ± 0.01 | 0.60 ± 0.01 | 0.44 ± 0.00 | 1.65 ± 0.01 | 0.77 ± 0.01 |
| 0.005 | 0 | 0.86 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0.005 | 2 | 2.72 ± 0.02 | 0.88 ± 0.01 | 0.20 ± 0.00 | 0.00 ± 0.00 | 1.08 ± 0.01 | 0.24 ± 0.00 |
| 0.005 | 6 | 3.55 ± 0.02 | 1.74 ± 0.01 | 0.82 ± 0.01 | 0.54 ± 0.00 | 2.09 ± 0.01 | 1.02 ± 0.01 |
| 0.005 | 14 | 2.17 ± 0.01 | 1.20 ± 0.01 | 0.64 ± 0.00 | 0.49 ± 0.00 | 1.44 ± 0.01 | 0.82 ± 0.00 |
| 0.005 | 22 | 0.85 ± 0.01 | 0.51 ± 0.00 | 0.29 ± 0.00 | 0.23 ± 0.00 | 0.61 ± 0.00 | 0.37 ± 0.00 |
| 0.01 | 0 | 1.47 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 0.01 | 2 | 3.46 ± 0.01 | 1.28 ± 0.01 | 0.33 ± 0.00 | 0.00 ± 0.00 | 1.78 ± 0.01 | 0.45 ± 0.00 |
| 0.01 | 6 | 1.85 ± 0.01 | 1.07 ± 0.00 | 0.61 ± 0.00 | 0.43 ± 0.00 | 1.38 ± 0.01 | 0.81 ± 0.00 |
| 0.01 | 14 | 0.20 ± 0.00 | 0.13 ± 0.00 | 0.09 ± 0.00 | 0.07 ± 0.00 | 0.17 ± 0.00 | 0.12 ± 0.00 |
| 0.01 | 22 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 |

**Table 1c. Ablations: excess LER over MLD (pp).** Mostly from stage 3, with U = 300.

| p | m | `bposd_cs7_dominant` | `bposd_cs7_ldpcmatrices` | `bposd_0` | `belief_find` | `uf_graph` |
|---|---|---|---|---|---|---|
| 0.0005 | 0 | 0.00 ± 0.00 | 0.08 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.45 ± 0.01 |
| 0.0005 | 2 | 0.02 ± 0.00 | 1.01 ± 0.01 | 0.73 ± 0.01 | 0.73 ± 0.01 | 2.55 ± 0.02 |
| 0.0005 | 6 | 0.12 ± 0.02 | 2.35 ± 0.08 | 2.07 ± 0.08 | 2.07 ± 0.08 | 6.98 ± 0.14 |
| 0.0005 | 14 | 0.26 ± 0.05 | 4.58 ± 0.21 | 3.98 ± 0.22 | 3.99 ± 0.22 | 12.08 ± 0.36 |
| 0.0005 | 22 | 0.38 ± 0.07 | 6.58 ± 0.32 | 4.99 ± 0.30 | 4.99 ± 0.30 | 16.65 ± 0.51 |
| 0.002 | 0 | 0.04 ± 0.00 | 0.30 ± 0.01 | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.30 ± 0.01 |
| 0.002 | 2 | 0.26 ± 0.02 | 3.23 ± 0.08 | 0.29 ± 0.02 | 0.30 ± 0.02 | 8.13 ± 0.13 |
| 0.002 | 6 | 0.52 ± 0.05 | 6.46 ± 0.27 | 0.72 ± 0.07 | 0.73 ± 0.07 | 15.51 ± 0.44 |
| 0.002 | 14 | 0.77 ± 0.08 | 8.80 ± 0.40 | 1.17 ± 0.11 | 1.18 ± 0.11 | 17.03 ± 0.56 |
| 0.002 | 22 | 0.74 ± 0.08 | 7.66 ± 0.36 | 1.26 ± 0.12 | 1.26 ± 0.12 | 13.14 ± 0.50 |
| 0.005 | 0 | 0.14 ± 0.00 | 0.69 ± 0.01 | 0.07 ± 0.00 | 0.07 ± 0.00 | 0.69 ± 0.01 |
| 0.005 | 2 | 0.89 ± 0.05 | 5.29 ± 0.18 | 0.96 ± 0.05 | 0.96 ± 0.05 | 12.89 ± 0.30 |
| 0.005 | 6 | 1.06 ± 0.07 | 5.91 ± 0.27 | 1.44 ± 0.09 | 1.41 ± 0.09 | 14.00 ± 0.47 |
| 0.005 | 14 | 0.78 ± 0.03 | 3.42 ± 0.11 | 1.00 ± 0.06 | 1.00 ± 0.06 | 5.01 ± 0.25 |
| 0.005 | 22 | 0.35 ± 0.02 | 1.21 ± 0.08 | 0.44 ± 0.03 | 0.44 ± 0.03 | 1.34 ± 0.08 |
| 0.01 | 0 | 0.43 ± 0.01 | 1.03 ± 0.01 | 0.17 ± 0.00 | 0.19 ± 0.00 | 1.03 ± 0.01 |
| 0.01 | 2 | 1.40 ± 0.07 | 5.13 ± 0.20 | 1.64 ± 0.08 | 1.59 ± 0.08 | 11.05 ± 0.33 |
| 0.01 | 6 | 0.89 ± 0.03 | 2.53 ± 0.08 | 1.08 ± 0.06 | 1.05 ± 0.05 | 3.92 ± 0.18 |
| 0.01 | 14 | 0.13 ± 0.01 | 0.26 ± 0.02 | 0.14 ± 0.01 | 0.13 ± 0.01 | 0.31 ± 0.01 |
| 0.01 | 22 | 0.01 ± 0.00 | 0.02 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.02 ± 0.00 |

**Table 1d. Worst cells with between-sequence SEs** (`between_sequence_se.csv`).
"New seqs" are the verifier's 10 new sequences (seed 777001) at the same
point (`results/C-verify/`, verified numerically by the verifier).

| decoder | worst cell (p, m) | excess (pp) ± shot SE | ± between-seq SE | per-sequence range (pp) | new seqs (pp) | approximate bound |
|---|---|---|---|---|---|---|
| `bposd_cs7_serial` | 5e-3, 6 | 0.68 ± 0.05 | ± 0.048 | 0.47-0.89 | 0.75 ± 0.04 (here); 0.83 ± 0.07 at (1e-2, 2) | about 0.7-0.8 |
| `bplsd_cs7_always` | 1e-2, 2 | 1.35 ± 0.07 | ± 0.12 | 0.62-1.90 | 1.39 ± 0.08 (here); 1.46 ± 0.09 at (5e-3, 6) | about 1.4 |
| `lut_w2_joint` | 5e-3, 6 | 1.74 ± 0.01 | ± 0.073 | 1.22-2.11 | 1.81 ± 0.05 | about 1.8 |
| `lut_w3_joint` | 5e-3, 6 | 0.82 ± 0.01 | ± 0.033 | 0.65-0.96 | 0.87 ± 0.02 | about 0.84 (pooled) |
| `lut_w4_joint` | 5e-3, 6 | 0.54 ± <0.01 | ± 0.025 | 0.42-0.66 | 0.58 ± 0.02 | about 0.56 (pooled) |
| `bposd_cs7` (parallel) | 5e-4, 22 | 4.75 ± 0.30 | ± 0.39 | 3.16-6.83 | 4.81 ± 0.28 | about 4.8 |

**Table 2. Agreement with MLD (% of shots with the same decision).**

| p | m | `bposd_cs7` | `bposd_cs7_serial` | `bplsd_cs7_always` | `uf_dn` | `uf_hyper` | `lut_w2_joint` | `lut_w3_joint` | `lut_w4_joint` |
|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | 0 | 99.6 ± 0.0 | 99.8 ± 0.0 | 99.6 ± 0.0 | 99.1 ± 0.0 | 99.9 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| 0.0005 | 2 | 98.8 ± 0.0 | 99.5 ± 0.0 | 99.5 ± 0.0 | 96.9 ± 0.0 | 97.9 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 | 100.0 ± 0.0 |
| 0.0005 | 6 | 97.1 ± 0.1 | 98.8 ± 0.1 | 99.1 ± 0.1 | 93.1 ± 0.1 | 95.0 ± 0.1 | 99.4 ± 0.0 | 99.5 ± 0.0 | 99.6 ± 0.0 |
| 0.0005 | 14 | 94.4 ± 0.3 | 97.3 ± 0.2 | 97.9 ± 0.2 | 87.0 ± 0.4 | 88.5 ± 0.4 | 98.5 ± 0.0 | 98.8 ± 0.0 | 98.9 ± 0.0 |
| 0.0005 | 22 | 92.9 ± 0.4 | 96.7 ± 0.3 | 97.3 ± 0.2 | 83.2 ± 0.5 | 79.8 ± 0.6 | 97.7 ± 0.0 | 98.2 ± 0.0 | 98.3 ± 0.0 |
| 0.002 | 0 | 98.2 ± 0.0 | 99.3 ± 0.0 | 98.3 ± 0.0 | 97.8 ± 0.0 | 99.6 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| 0.002 | 2 | 97.8 ± 0.1 | 98.1 ± 0.1 | 97.7 ± 0.1 | 88.6 ± 0.2 | 92.1 ± 0.1 | 98.7 ± 0.0 | 99.6 ± 0.0 | 100.0 ± 0.0 |
| 0.002 | 6 | 94.9 ± 0.3 | 95.0 ± 0.3 | 93.9 ± 0.3 | 77.8 ± 0.6 | 83.1 ± 0.5 | 94.8 ± 0.0 | 96.4 ± 0.0 | 97.3 ± 0.0 |
| 0.002 | 14 | 90.7 ± 0.5 | 90.2 ± 0.5 | 88.1 ± 0.6 | 65.3 ± 0.8 | 68.8 ± 0.8 | 88.4 ± 0.0 | 91.5 ± 0.0 | 92.5 ± 0.0 |
| 0.002 | 22 | 86.6 ± 0.6 | 85.8 ± 0.6 | 84.5 ± 0.7 | 59.8 ± 0.9 | 58.4 ± 0.9 | 82.7 ± 0.1 | 86.7 ± 0.0 | 87.9 ± 0.0 |
| 0.005 | 0 | 96.0 ± 0.0 | 98.4 ± 0.0 | 96.2 ± 0.0 | 94.8 ± 0.0 | 98.6 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| 0.005 | 2 | 93.5 ± 0.3 | 94.9 ± 0.2 | 92.2 ± 0.3 | 77.4 ± 0.5 | 82.7 ± 0.4 | 94.7 ± 0.0 | 97.8 ± 0.0 | 100.0 ± 0.0 |
| 0.005 | 6 | 84.0 ± 0.7 | 85.4 ± 0.6 | 81.0 ± 0.7 | 60.8 ± 0.9 | 69.2 ± 0.8 | 82.9 ± 0.1 | 87.3 ± 0.0 | 90.1 ± 0.0 |
| 0.005 | 14 | 73.0 ± 0.5 | 73.3 ± 0.5 | 68.3 ± 0.5 | 53.0 ± 0.6 | 57.8 ± 0.6 | 68.4 ± 0.1 | 73.8 ± 0.1 | 76.3 ± 0.1 |
| 0.005 | 22 | 64.2 ± 0.7 | 64.2 ± 0.9 | 61.2 ± 0.7 | 52.2 ± 0.8 | 51.3 ± 0.8 | 60.1 ± 0.1 | 64.9 ± 0.1 | 67.1 ± 0.1 |
| 0.01 | 0 | 96.5 ± 0.0 | 96.8 ± 0.0 | 97.0 ± 0.0 | 90.2 ± 0.0 | 96.9 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| 0.01 | 2 | 86.4 ± 0.5 | 87.4 ± 0.5 | 83.4 ± 0.6 | 66.5 ± 0.7 | 73.2 ± 0.7 | 87.4 ± 0.0 | 94.7 ± 0.0 | 100.0 ± 0.0 |
| 0.01 | 6 | 75.3 ± 0.5 | 75.6 ± 0.5 | 70.5 ± 0.5 | 54.8 ± 0.6 | 64.8 ± 0.5 | 69.8 ± 0.1 | 76.1 ± 0.1 | 80.4 ± 0.1 |
| 0.01 | 14 | 59.2 ± 0.3 | 59.6 ± 0.3 | 56.7 ± 0.4 | 50.0 ± 0.4 | 56.1 ± 0.4 | 56.0 ± 0.1 | 59.8 ± 0.1 | 62.2 ± 0.1 |
| 0.01 | 22 | 54.0 ± 0.4 | 53.1 ± 0.9 | 52.8 ± 0.4 | 50.2 ± 0.5 | 50.9 ± 0.5 | 51.7 ± 0.1 | 53.8 ± 0.1 | 55.1 ± 0.1 |

**Table 3. Raw paired excess (n10 - n01)/N in pp, a check on the posterior-weighted estimator.** Over all 368 non-MLD cells of `summary.csv`, (raw - posterior-weighted)/raw SE has chi-square per degree of freedom 1.01 (6.5% of cells beyond 2 SE, one at 3.1 SE). No raw excess is below -2 SE.

| p | m | `bposd_cs7` | `bposd_cs7_serial` | `bplsd_cs7_always` | `uf_dn` | `lut_w3_joint` |
|---|---|---|---|---|---|---|
| 0.0005 | 0 | 0.03 ± 0.01 | 0.00 ± 0.01 | 0.03 ± 0.01 | 0.11 ± 0.01 | 0.00 ± 0.00 |
| 0.0005 | 2 | 0.71 ± 0.02 | 0.01 ± 0.01 | 0.03 ± 0.01 | 1.24 ± 0.03 | 0.00 ± 0.00 |
| 0.0005 | 6 | 2.04 ± 0.09 | 0.04 ± 0.06 | 0.11 ± 0.05 | 3.10 ± 0.15 | 0.05 ± 0.01 |
| 0.0005 | 14 | 4.04 ± 0.27 | 0.20 ± 0.19 | 0.19 ± 0.16 | 4.96 ± 0.41 | 0.06 ± 0.02 |
| 0.0005 | 22 | 5.10 ± 0.37 | -0.45 ± 0.26 | 0.53 ± 0.24 | 6.81 ± 0.58 | 0.05 ± 0.02 |
| 0.002 | 0 | 0.01 ± 0.02 | 0.02 ± 0.01 | -0.00 ± 0.02 | 0.30 ± 0.02 | 0.00 ± 0.00 |
| 0.002 | 2 | 0.05 ± 0.08 | 0.02 ± 0.08 | 0.14 ± 0.08 | 4.01 ± 0.18 | 0.05 ± 0.01 |
| 0.002 | 6 | 0.25 ± 0.31 | -0.17 ± 0.30 | 0.30 ± 0.34 | 6.98 ± 0.63 | 0.31 ± 0.03 |
| 0.002 | 14 | 0.31 ± 0.54 | 0.37 ± 0.55 | 1.19 ± 0.61 | 10.36 ± 1.03 | 0.59 ± 0.04 |
| 0.002 | 22 | 0.99 ± 0.66 | 0.90 ± 0.68 | 1.16 ± 0.72 | 8.74 ± 1.14 | 0.63 ± 0.05 |
| 0.005 | 0 | 0.07 ± 0.03 | 0.06 ± 0.02 | 0.04 ± 0.03 | 0.81 ± 0.03 | 0.00 ± 0.00 |
| 0.005 | 2 | 0.13 ± 0.28 | 0.11 ± 0.24 | 0.57 ± 0.30 | 6.59 ± 0.51 | 0.18 ± 0.02 |
| 0.005 | 6 | 1.65 ± 0.71 | 1.40 ± 0.68 | 1.39 ± 0.77 | 8.44 ± 1.10 | 0.82 ± 0.05 |
| 0.005 | 14 | -0.65 ± 0.58 | -0.65 ± 0.58 | -0.56 ± 0.63 | 3.66 ± 0.76 | 0.60 ± 0.07 |
| 0.005 | 22 | 0.50 ± 0.89 | 0.37 ± 1.09 | 1.21 ± 0.92 | 1.52 ± 1.08 | 0.40 ± 0.08 |
| 0.01 | 0 | 0.12 ± 0.03 | 0.11 ± 0.03 | 0.06 ± 0.02 | 1.45 ± 0.04 | 0.00 ± 0.00 |
| 0.01 | 2 | 0.18 ± 0.57 | 0.02 ± 0.55 | 0.62 ± 0.63 | 5.40 ± 0.89 | 0.34 ± 0.03 |
| 0.01 | 6 | 0.67 ± 0.56 | 1.00 ± 0.55 | 1.05 ± 0.61 | 3.28 ± 0.75 | 0.49 ± 0.07 |
| 0.01 | 14 | 0.55 ± 0.45 | 0.78 ± 0.45 | 0.22 ± 0.59 | 0.11 ± 0.50 | 0.09 ± 0.09 |
| 0.01 | 22 | 0.36 ± 0.48 | -0.77 ± 1.25 | 0.11 ± 0.49 | 1.08 ± 0.73 | 0.17 ± 0.10 |

### 2. Exact first-order analysis (`first_order.csv`, p = 5e-4, sequences 0-5)

This analysis needs no sampling. For each distinct detector set D of the DEM,
P_D(c) is the XOR-combined probability of the mechanisms with detectors D and
logical effect c. Each decoder decodes the syndrome D once, *with its own
p-dependent decision rule at p = 5e-4*. The table is therefore the
single-fault part of the LER at p = 5e-4, not the strict p -> 0 limit.
The two differ only where a decoder's decision depends on p; see the LUT
bullet below. A single fault is
then decoded wrongly with probability P_D(1 - decision). Summed over D and
divided by p, this gives each decoder's first-order LER coefficient. MLD's
floor is sum_D min_c P_D(c) / p. The table lists the *excess* over that
floor (mean ± SE over sequences; at m = 0 all sequences give the same
circuit). Single-fault mass / p is 69, 280, 701, 1541 and 2381 at
m = 0, 2, 6, 14, 22. About 24-29% of detector sets carry both logical classes
(`logs/C_decomposition_check.log`).

Excess first-order LER / p, LUTs and BP+OSD variants:

| m | MLD floor / p | `lut_w1_joint` | `lut_w2_joint` | `lut_w3_joint` | `bposd_cs7` | `bposd_cs7_serial` | `bposd_0` | `bposd_cs7_dominant` | `bposd_cs7_ldpcmatrices` |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 11.1 | 0.86 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.73 ± 0.00 |
| 2 | 36.8 | 7.80 ± 1.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 16.84 ± 6.33 | 0.13 ± 0.06 | 17.00 ± 6.34 | 0.36 ± 0.17 | 21.25 ± 2.15 |
| 6 | 83.9 | 12.91 ± 1.12 | 0.00 ± 0.00 | 0.00 ± 0.00 | 62.56 ± 7.20 | 0.27 ± 0.05 | 63.19 ± 7.29 | 1.78 ± 0.33 | 58.17 ± 3.70 |
| 14 | 178.4 | 25.48 ± 0.87 | 0.00 ± 0.00 | 0.00 ± 0.00 | 130.35 ± 13.61 | 0.43 ± 0.10 | 131.89 ± 13.71 | 5.61 ± 0.91 | 132.12 ± 5.38 |
| 22 | 272.0 | 33.67 ± 1.57 | 0.10 ± 0.04 | 0.00 ± 0.00 | 207.64 ± 13.81 | 0.70 ± 0.13 | 210.20 ± 13.90 | 6.62 ± 1.70 | 205.75 ± 4.76 |

Excess first-order LER / p, BP+LSD, belief-find, union-find and the matching control:

| m | MLD floor / p | `bplsd_cs7` | `bplsd_cs7_always` | `belief_find` | `uf_dn` | `uf_graph` | `uf_hyper` | `mwpm_graph` |
|---|---|---|---|---|---|---|---|---|
| 0 | 11.1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.73 ± 0.00 | 9.19 ± 0.00 | 1.73 ± 0.00 | 1.73 ± 0.00 |
| 2 | 36.8 | 16.96 ± 6.35 | 0.12 ± 0.04 | 17.00 ± 6.34 | 28.84 ± 2.33 | 59.96 ± 4.51 | 29.68 ± 5.05 | 19.67 ± 3.15 |
| 6 | 83.9 | 63.08 ± 7.27 | 0.52 ± 0.13 | 63.19 ± 7.29 | 69.46 ± 2.46 | 168.29 ± 12.59 | 70.90 ± 10.44 | 53.39 ± 4.14 |
| 14 | 178.4 | 131.56 ± 13.69 | 1.21 ± 0.12 | 131.89 ± 13.71 | 145.44 ± 8.84 | 375.55 ± 15.77 | 145.41 ± 13.85 | 122.92 ± 3.98 |
| 22 | 272.0 | 209.65 ± 13.87 | 2.01 ± 0.10 | 210.20 ± 13.90 | 221.06 ± 7.34 | 575.30 ± 23.08 | 290.96 ± 19.93 | 191.82 ± 3.77 |

Reading (all **verified numerically**, this exact enumeration):

- **LUT.** At p = 5e-4 the excess is zero for w >= 3 at every depth, and
  for w = 2 up to m = 14. At m = 22, w = 2 has 0.10 ± 0.04 p. That is a
  finite-p effect of the window, not a first-order failure: at p = 5e-4,
  order-p^2 competitors whose follow-up fault would fire the unseen slot
  can outweigh a weak single fault.
- At p = 1e-5 (`first_order_lut_p1e-5.csv`, sequences 0-5,
  m = 2, 6, 14, 22), w = 2, 3 and 4 all have excess <= 1e-6 p. That is the
  same near-tie residual as MLD itself (MLD excess <= 1.04e-6 p).
- So **every w >= 2 is first-order exact as p -> 0**. The verifier gives
  the analytic argument (a w >= 2 window sees both slots a single fault
  can touch) and numerics at 1e-5 and 1e-6.
- w = 1 (no look-ahead) is not: 0.9-34 p at p = 5e-4 and 7.8-34 p at
  p = 1e-5. It cannot tell a measurement error from a data error.
- Product-sum flooding-BP decoders (`bposd_cs7`, `bposd_0`, `bplsd_cs7`,
  `belief_find`) have an excess that grows with depth: 17 p at m = 2 and
  208 ± 14 p at m = 22, where MLD's floor is 272 p.
- The verifier's independent re-implementation reproduces these numbers to
  all printed digits.
- In the verifier's runs, min-sum flooding BP shows the same failure on
  1.3-1.5% (scaling 1) or 3.3-4.1% (scaling 0.625) of one-class sets:
  14-17 p and 81-99 p at m = 22.
- With ldpc's full defaults (OSD-0, max_iter = n), 10-13% of one-class sets
  are decoded to the wrong class; 5.5-5.9 points of that come from
  converged BP and the rest from OSD-0. They are first-order
  exact only at m = 0. The serial schedule (<= 0.7 ± 0.1 p) and
  `always_run_lsd` (<= 2.0 ± 0.1 p) remove almost all of it, and so do
  dominant columns (6.6 ± 1.7 p at m = 22).
- The failures are *not* on the ambiguous (two-class) detector sets.
  Sequences 0-2, m = 6 and 14, p = 5e-4
  (`logs/C_duplicate_columns.log`): parallel BP+OSD-CS7 decodes 5-6% of the
  *one-class* sets (a single first-order explanation) to the wrong class,
  always with BP "converged". Summed over the three sequences, these sets
  carry 172 of 176 p (m = 6) and 319 of 326 p (m = 14) of the excess.
  Serial BP makes no error on one-class sets; its errors are on near-tied
  two-class sets (0.7 and 0.9 p summed over the three sequences).
- ldpc's own conversion (`bposd_cs7_ldpcmatrices`) costs 206 ± 5 p at
  m = 22 even though its columns are already merged. That is the
  last-listed-observable overwrite.
- Union-find adds excess at every depth: 1.7 p at m = 0 (9.2 p for
  `uf_graph`). At m = 22 the excess is 221 ± 7 p for `uf_dn`, 291 ± 20 p for
  `uf_hyper` and 575 ± 23 p for `uf_graph`. MWPM on the same graph has
  192 ± 4 p.

### 3. LUT footprint (`lut_footprint.csv`; **verified analytically** and numerically)

Per circuit of depth m (T = m + 2 slots), with w' = min(w, T) and w' >= 2,
the memory is (m + 2 - w') * 2^(4w') * 5 bits for the interior tables plus
2^(4w'-2) bits for the final table.

- When w >= T the decoder stores only the final table, over all slots. For
  example, m = 0 with w = 3 or 4 stores 64 bits, not the negative number
  that w in place of w' would give.
- For w = 1, the last interior table has 3-bit words: m * 80 + 48 + 4 bits.

With these rules the formula matches `table_bits` for all 160 rows of
`lut_footprint.csv`.

| w | interior table (per round) | address bits | total, m = 14 | total, m = 22 | build time, m = 22 (single core) |
|---|---|---|---|---|---|
| 1 | 16 x 5 bit | 4 | 0.14 KiB | 0.22 KiB | (not timed separately; < 30 ms) |
| 2 | 256 x 5 bit | 8 | 2.2 KiB | 3.45 KiB | 28 ms |
| 3 | 4096 x 5 bit (20 kbit) | 12 | 32.6 KiB | 52.6 KiB | 50 ms |
| 4 | 65536 x 5 bit (320 kbit) | 16 | 482 KiB | 802 KiB | 409 ms |

At run time each check round costs a 4-bit XOR (sigma = s XOR kappa), one
table read addressed by sigma and the look-ahead bits, and a 1-bit XOR into
the frame. A decision is final w - 1 rounds after its round. Tables depend
on the Clifford sequence, which LRB fixes before the run, so they can be
precomputed and loaded per sequence.
**Conjecture:** w = 3 fits comfortably in FPGA block RAM (one 20 kbit table
per round in use at a time).

### 4. Runtime (`timing.csv`; controlled single-process benchmark)

Two different measures appear below; do not compare them across columns.
*Batch throughput* is µs per shot for numpy-vectorized decoding of a 50,000-
shot batch (MLD, LUTs, PyMatching). *Per-call latency* is µs for one
Python-to-C++ call on one syndrome (ldpc decoders, `uf_dn`).

**Table 4a (`timing.csv`).** Login node `login15` (AMD EPYC 7713, load
about 30 on 256 hardware threads), p = 5e-3, sequence 0, best of 3 runs.

- Per-call decoders were timed on the first 300 distinct non-zero syndromes
  *in byte-lexicographic order*. That sample is biased towards light
  syndromes and understates per-call cost by up to 25-43% at p = 5e-4,
  m = 22 (verifier); Table 4b uses random samples instead.
- **Verified numerically**, for this machine and load only.

| decoder | unit | m = 2 | m = 6 | m = 14 | m = 22 | build at m = 22 (ms) |
|---|---|---|---|---|---|---|
| `mld` | µs / shot | 1.4 | 2.87 | 6.21 | 8.98 | 24.6 |
| `lut_w2_joint` | µs / shot | 0.0407 | 0.0949 | 0.23 | 0.405 | 27.9 |
| `lut_w3_joint` | µs / shot | 0.0362 | 0.093 | 0.208 | 0.404 | 49.8 |
| `lut_w4_joint` | µs / shot | 0.0305 | 0.0897 | 0.24 | 0.501 | 409 |
| `bposd_cs7` | µs / syndrome | 182 | 611 | 1.88e+03 | 4.82e+03 | 14.4 |
| `bposd_cs7_serial` | µs / syndrome | 974 | 3.31e+03 | 7.94e+03 | 1.58e+04 | 13.4 |
| `bposd_0` | µs / syndrome | 117 | 311 | 670 | 1.21e+03 | 13.3 |
| `bplsd_cs7` | µs / syndrome | 140 | 313 | 636 | 1.2e+03 | 12.9 |
| `bplsd_cs7_always` | µs / syndrome | 144 | 317 | 644 | 1.21e+03 | 12.8 |
| `bposd_cs7_dominant` | µs / syndrome | 118 | 403 | 1.08e+03 | 3.15e+03 | 14.1 |
| `bposd_cs7_ldpcmatrices` | µs / syndrome | 119 | 402 | 1.08e+03 | 3.14e+03 | 14.4 |
| `uf_dn` | µs / syndrome | 227 | 318 | 500 | 943 | 30.5 |
| `uf_graph` | µs / syndrome | 16 | 24.8 | 42.4 | 74.6 | 30 |
| `uf_hyper` | µs / syndrome | 54 | 260 | 1.11e+03 | 3.49e+03 | 12.5 |
| `belief_find` | µs / syndrome | 124 | 297 | 611 | 1.15e+03 | 13.1 |
| `mwpm_graph` | µs / shot | 0.372 | 1.47 | 3.26 | 5.36 | 30.1 |

**Table 4b (`timing_random.csv`, `logs/C_timing_random.log`).** Same node,
heavier load (about 77), sequence 0, 50,000 shots per point.

- Per-call decoders were timed on 200 random shots with non-zero syndromes
  (frequency-weighted, i.e. what a real-time decoder sees) and on 200
  random distinct syndromes.
- Throughput with de-duplication = (mean per-call time on distinct
  syndromes) x (number of distinct non-zero syndromes) / 50,000.
- "Single-shot" is the median time to decode one shot with the
  numpy/Python MLD or LUT.
- Absolute numbers are about 1.5-2x the Table 4a values because of the load.

| p, m | MLD batch µs/shot | LUT w=3 batch µs/shot | MLD single-shot µs | LUT w=3 single-shot µs | `bposd_cs7` µs/call (random shots / random distinct) | `bposd_cs7` throughput µs/shot (de-dup) | `bposd_cs7_serial` µs/call (random shots) | `bplsd_cs7_always` µs/call (random shots) | `uf_dn` µs/call (random shots) |
|---|---|---|---|---|---|---|---|---|---|
| 0.005, 6 | 5.5 | 0.175 | 533 | 205 | 1.07e+03 / 1.25e+03 | 841 | 5.68e+03 | 600 | 804 |
| 0.005, 22 | 14.2 | 0.719 | 1.23e+03 | 1.14e+03 | 8.64e+03 / 7.06e+03 | 7.06e+03 | 2.04e+04 | 1.93e+03 | 1.37e+03 |
| 0.0005, 6 | 3.2 | 0.107 | 230 | 133 | 318 / 474 | 18.5 | 2.49e+03 | 207 | 271 |
| 0.0005, 22 | 11.8 | 0.45 | 660 | 464 | 2.48e+03 / 3.71e+03 | 1.18e+03 | 8.02e+03 | 686 | 629 |

Reading (**verified numerically**, this machine and load):

- **Throughput.** The vectorized LUT handles 0.1-0.7 µs/shot, against
  3-14 µs/shot for MLD and 18 µs-7 ms/shot for BP+OSD-CS7 even with
  de-duplication.
- **Single-shot latency in Python.** The LUT (0.13-1.1 ms) is only 1.1-2.6x
  faster than MLD and 2.4-7.6x faster than BP+OSD-CS7 per call on random
  shots (verifier: 3.5-11x at lower load). Python call overhead dominates.
- The four-orders-of-magnitude gap between a table lookup and BP+OSD
  exists only for a hardware LUT, and that is a **conjecture**.

Grid-time measurements (`us_per_call`, `us_per_shot` in `summary.csv`) from
stages 1-2 were taken while `srun --overlap` steps were pinned to shared CPUs
(node load about 241). They are 5-20x slower than this benchmark; use them
only for relative comparisons within a stage.

### 5. MLD calibration (`calibration.csv`)

| p | m | shots | observed MLD LER (%) | posterior-predicted LER (%) | z |
|---|---|---|---|---|---|
| 0.0005 | 0 | 500,000 | 0.53 ± 0.01 | 0.551 ± 0.007 | -1.38 |
| 0.0005 | 2 | 500,000 | 1.83 ± 0.02 | 1.856 ± 0.012 | -0.97 |
| 0.0005 | 6 | 500,000 | 4.15 ± 0.03 | 4.140 ± 0.018 | 0.42 |
| 0.0005 | 14 | 500,000 | 8.31 ± 0.04 | 8.300 ± 0.024 | 0.17 |
| 0.0005 | 22 | 500,000 | 12.27 ± 0.05 | 12.228 ± 0.028 | 0.81 |
| 0.002 | 0 | 500,000 | 2.21 ± 0.02 | 2.202 ± 0.014 | 0.22 |
| 0.002 | 2 | 500,000 | 7.39 ± 0.04 | 7.441 ± 0.022 | -1.17 |
| 0.002 | 6 | 500,000 | 15.66 ± 0.05 | 15.626 ± 0.028 | 0.54 |
| 0.002 | 14 | 500,000 | 27.29 ± 0.06 | 27.308 ± 0.029 | -0.25 |
| 0.002 | 22 | 500,000 | 35.22 ± 0.07 | 35.294 ± 0.025 | -1.02 |
| 0.005 | 0 | 500,000 | 5.37 ± 0.03 | 5.367 ± 0.020 | -0.07 |
| 0.005 | 2 | 500,000 | 17.67 ± 0.05 | 17.724 ± 0.027 | -0.84 |
| 0.005 | 6 | 500,000 | 32.61 ± 0.07 | 32.560 ± 0.025 | 0.73 |
| 0.005 | 14 | 500,000 | 44.64 ± 0.07 | 44.746 ± 0.013 | -1.54 |
| 0.005 | 22 | 500,000 | 48.35 ± 0.07 | 48.494 ± 0.006 | -2.08 |
| 0.01 | 0 | 500,000 | 10.41 ± 0.04 | 10.427 ± 0.025 | -0.44 |
| 0.01 | 2 | 500,000 | 30.91 ± 0.07 | 30.955 ± 0.024 | -0.66 |
| 0.01 | 6 | 500,000 | 45.44 ± 0.07 | 45.407 ± 0.011 | 0.47 |
| 0.01 | 14 | 500,000 | 49.76 ± 0.07 | 49.717 ± 0.002 | 0.64 |
| 0.01 | 22 | 500,000 | 49.90 ± 0.07 | 49.984 ± 0.000 | -1.15 |

Chi-square over the 20 points: 17.0 for 20 degrees of freedom; the largest |z| is 2.08. The z denominator sqrt(SE_obs^2 + SE_post^2) is conservative, because the two estimates are positively correlated on the same shots.

## Claims

| # | Claim | Label | Evidence |
|---|---|---|---|
| 1 | No decoder beats exact MLD in expectation. Every posterior-weighted excess is >= 0 by construction, and every raw paired excess is consistent with >= 0. | verified analytically (MLD is Bayes-optimal for the exact DEM) and verified numerically | Tables 1-3 |
| 2 | ldpc 2.4.1's `detector_error_model_to_check_matrices` (the `SinterBpOsdDecoder` path) merges equal detector sets and keeps the *last listed* observable. 24-29% of detector sets carry both logical classes as per-depth means; per circuit the range is 20-34%. | verified numerically (verifier census: 19.5-33.6% per circuit) | `test_ldpc_conversion_keeps_last_listed_observable`; `logs/C_decomposition_check.log` |
| 3 | That overwrite costs BP+OSD-CS7 206 ± 5 p of first-order LER at m = 22 (MLD floor 272 p); the verifier's label-only cost is 202.9 ± 3.9 p. Sampled, the excess is 1.0-8.8 pp at the informative points. At the worst cell (p = 2e-3, m = 14) the total of 8.8 ± 0.6 pp includes about 8.0 pp from the overwrite: the same BP+OSD on correctly labelled merged columns gives 0.77 ± 0.09 pp. | verified numerically | `first_order.csv`; Table 1c; `between_sequence_se.csv` |
| 4 | With product-sum flooding (parallel) BP, BP+OSD, BP+LSD and belief-find return a syndrome-valid solution of the wrong logical class on 5-6% of *unambiguous* single-fault syndromes. BP has converged, so OSD/LSD never run. This adds 17 ± 6 p (m = 2) to 208 ± 14 p (m = 22) of first-order LER. Product-sum is ldpc 2.4.1's actual default, although its docstring says `minimum_sum`. | verified numerically (reproduced by the verifier) | `first_order.csv`; `logs/C_duplicate_columns.log`; Bugs item 2 |
| 5 | The serial BP schedule (<= 0.7 ± 0.1 p) or `always_run_lsd=True` (<= 2.0 ± 0.1 p) removes the failure of claim 4. Merging duplicate columns ("dominant") reduces it to <= 6.6 ± 1.7 p. | verified numerically | `first_order.csv` |
| 6 | Duplicate opposite-class columns trigger claim 4 by destabilizing flooding BP *globally*: merging them ("dominant") mostly cures it, yet the failing syndromes are mostly *not* the duplicated sets (5-6% of one-class sets fail versus near-ties among two-class sets). | conjecture (correlation verified numerically; mechanism not shown) | `first_order.csv`; `logs/C_duplicate_columns.log` |
| 7 | The product-sum parallel-BP failure is a low-p effect. Parallel and serial BP+OSD-CS7 differ by <= 0.03 pp at p >= 5e-3 and by 0.04-0.12 pp at p = 2e-3, but by 0.70-4.65 pp at p = 5e-4 (m = 2-22). At p = 5e-4, m = 22 parallel BP+OSD-CS7 reaches 4.75 ± 0.39 pp (between-sequence SE; MLD LER 12.3%). | verified numerically | Table 1a; Table 1d |
| 8 | With the serial schedule, BP+OSD-CS7 is the closest general-purpose decoder. Its excess is about 0.7-0.8 pp: worst grid cell 0.68 ± 0.05 pp (between-sequence SE), 0.83 ± 0.07 pp on the verifier's new sequences. BP+LSD-CS7 with `always_run_lsd` reaches about 1.4 pp: 1.35 ± 0.12, new sequences 1.46 ± 0.09. | verified numerically | Table 1a; Table 1d |
| 9 | Union-find is far from MLD on this d = 2 circuit: 1.2-9.5 pp (2.6-17 pp for ldpc's matrix UF on the graph) sampled excess at the informative points (m >= 2, MLD LER < 40%) for the textbook Delfosse-Nickerson UF and ldpc's matrix-solve UF, and 221-575 p first-order excess at m = 22. MWPM on the same decomposed graph is also far off (0.9-8.5 pp; 192 ± 4 p). | verified numerically | Table 1a; `first_order.csv` |
| 10 | Cause of claim 9: at d = 2, decoding means picking the more probable of explanations in different logical classes. Peeling and minimum-weight matching on a decomposed graph do not compare class probabilities. | conjecture | claim 9; Open question 5 |
| 11 | The sliding-window LUT with w >= T is the exact MLD; its final LLR equals `TransferMatrixMLD`'s to <= 3e-15 (verifier: 3.6e-15 on 48 new circuits). | verified numerically | `test_lut_with_full_window_is_exact_mld` |
| 12 | Every LUT table entry equals the brute-force window MAP, and vectorized `predict` equals a brute-force run of the sliding-window algorithm, on random multi-slot DEMs. | verified numerically | `test_lut_tables_match_brute_force`, `test_lut_predict_matches_reference_stepping` |
| 13 | Every LUT with w >= 2 is first-order exact as p -> 0: at p = 1e-5 the excess is <= 1e-6 p for w = 2, 3, 4, the same as MLD's near-tie residual. At p = 5e-4, w = 2 shows a 0.10 ± 0.04 p finite-p effect at m = 22; w = 1 is 7.8-34 p off. Sampled excess bounds are approximate (between-sequence SEs, verifier's new sequences): w = 2 about 1.8 pp, w = 3 about 0.84 pp, w = 4 about 0.56 pp. The `joint` commit rule is never worse than `carry`. | verified numerically (analytic argument in the verifier's report) | `first_order.csv`; `first_order_lut_p1e-5.csv`; Table 1b; Table 1d |
| 14 | LUT memory per circuit is (m+2-w') 2^(4w') x 5 + 2^(4w'-2) bits with w' = min(w, T) >= 2. For w = 1 it is m x 80 + 52 bits; for w >= T only one final table is stored (64 bits at m = 0). This gives 3.45 / 52.6 / 802 KiB at m = 22 for w = 2 / 3 / 4, and each round costs one lookup with a 4w-bit address. | verified analytically; verified numerically (all 160 rows of `lut_footprint.csv`) | `lut_footprint.csv` |
| 15 | Batch throughput (numpy-vectorized, one core): MLD 1.4-14 us/shot and LUT w = 3 0.03-0.7 us/shot; BP+OSD-CS7 with syndrome de-duplication 18 us-7 ms/shot. Per-call latency: BP+OSD-CS7 0.18-8.6 ms, serial 1.0-20 ms, BP+LSD 0.14-1.9 ms, ldpc UF on the graph 16-75 us, DN UF 0.23-1.4 ms. Single-shot latency in Python: LUT 0.13-1.1 ms and MLD 0.23-1.2 ms, so the Python LUT beats BP+OSD by only 2.4-7.6x per shot. | verified numerically (this machine and load; Table 4a uses a lexicographic, light-biased syndrome sample, Table 4b a random one) | `timing.csv`; `timing_random.csv` |
| 16 | The MLD posterior is calibrated against the circuit sampler at all 20 grid points (chi-square 17.0 for 20 degrees of freedom; max |z| = 2.08). | verified numerically | `calibration.csv` |
| 17 | An independent exact-inference implementation reproduces `TransferMatrixMLD`. No bug was found in the lead's model. | verified numerically (checks listed under Bugs) | tests; `calibration.csv` |
| 18 | ldpc 2.4.1's peeling UF hangs on a valid 26-defect syndrome (p = 1e-2, m = 22). `UnionFindDecoder(H)` raises `TypeError`, and `uf_method="peeling"` selects matrix-solve mode. | verified numerically | Bugs items 3-4 |
| 19 | The C++ "Iterator index out of bounds" abort in the first launch came from ldpc's peeling UF. | conjecture | Bugs item 5 |
| 20 | Stim's `decompose_errors=True` succeeds for all 200 grid circuits, and the resulting graph has only weight-1/2 columns. | verified numerically | `logs/C_decomposition_check.log` |
| 21 | A w = 3 LUT (20 kbit per round) fits FPGA block RAM at nanosecond-scale lookup latency, which would make it orders of magnitude faster than BP+OSD per shot. | conjecture (not tested on hardware) | claim 14 |
| 22 | Min-sum flooding BP shows the claim-4 failure on 1.3-4.1% of one-class single-fault syndromes (14-17 p at m = 22 for scaling 1; 81-99 p for scaling 0.625). It beats product-sum flooding BP at p = 5e-4 (0.71 vs 4.81 pp at m = 22) but is 1.2-1.4 pp worse at p >= 2e-3. ldpc's full defaults (OSD-0, max_iter = n) decode 10-13% of one-class single-fault syndromes to the wrong class. | verified numerically by the WP-C verifier (not re-run by me) | `results/C-verify/`; `notes/C-bposd-uf-lut.verify.md` sec. 3 |

"Informative points" are those with m >= 2 and MLD LER below 40%: all of
p = 5e-4 and 2e-3, p = 5e-3 at m <= 6, and p = 1e-2 at m = 2. Near LER 50%
every decoder is close to a coin and excesses shrink trivially. Citation:
Delfosse and Nickerson, arXiv:1709.06218, as fetched and checked in
`lit/03-decoders.md`; I did not re-fetch it, so it is **UNVERIFIED by me**.

## Bugs and pitfalls found

### In the lead's model: none found

What I checked:

1. My LUT's exact final step is an independent exact-inference
   implementation: earliest-slot grouping and a backward recursion, against
   the lead's latest-slot forward transfer matrices. With w >= T it
   reproduces `TransferMatrixMLD.log_likelihood_ratio` to <= 3e-15 for
   m = 0, 1, 2, both terminals (`data5`, `data2`, the latter with
   undetectable logical mechanisms) and p = 2e-3 and 3e-2
   (`test_lut_with_full_window_is_exact_mld`). **Verified numerically.**
2. MLD's observed LER matches its own posterior prediction at every grid
   point (calibration table). This checks the DEM against the circuit
   sampler at all depths. **Verified numerically.**
3. Every DEM mechanism of all 200 grid circuits touches at most two adjacent
   slots: `earliest_slot_kernels` raises otherwise, and it built kernels for
   every circuit. **Verified numerically.**

One latent pitfall, not a bug in current use: `mld.dem_mechanisms` lists the
detectors of every `^` component without cancelling repeats. The kernels XOR
bit vectors, so repeats cancel there. Only code that reads
`Mechanism.detectors` as a set on a *decomposed* DEM would be misled.

### ldpc 2.4.1 (external library)

1. **Observable overwrite in the DEM conversion.**
   `ldpc.ckt_noise.detector_error_model_to_check_matrices`, which
   `SinterBpOsdDecoder` uses, keys columns by detector set only. Mechanisms
   with equal detector sets are XOR-merged into one column, and that column
   keeps the observable of the *last listed* mechanism, whatever its
   probability. In these DEMs every multi-mechanism detector set is a pair
   with opposite logical effect: for sequence 0, 6 of 21 sets at m = 0, 34
   of 113 at m = 2 and 244 of 1033 at m = 22. So the conversion decides the
   logical class of those columns by DEM order. Test:
   `test_ldpc_conversion_keeps_last_listed_observable`. Impact: the
   `bposd_cs7_ldpcmatrices` column of the results. **Verified numerically.**
2. **BP convergence skips OSD/LSD.** `BpOsdDecoder`, `BpLsdDecoder` and
   `BeliefFindDecoder` post-process only when BP fails to satisfy the
   syndrome. With parallel (flooding) product-sum BP, BP often converges to
   a syndrome-valid but improbable solution.
   - Product-sum is ldpc 2.4.1's *actual* default: the base class reads
     `kwargs.get("bp_method", 0)`, and 0 maps to product-sum, although the
     docstring says `minimum_sum`. Found by the verifier.
   - `SinterBpOsdDecoder` instead defaults to min-sum with scaling 0.625
     and OSD-0.
   - Min-sum flooding BP fails the same way, less often (claim 22).

   Diagnostic: held-out sequence 10, m = 14, p = 5e-4, 4000 shots, 1045
   distinct non-zero syndromes.

   | | BP converged | disagreements with MLD | of which BP had converged |
   |---|---|---|---|
   | parallel schedule | 619 | 109 | 70 |
   | serial schedule | 671 | 54 | 27 |

   Example: parallel BP returned three mechanisms with total log-prior
   -18.5 (to -20.4) where one mechanism with log-prior -5.9 explains the
   syndrome and gives MLD's class. The serial schedule and
   `always_run_lsd=True` mostly remove this. ldpc 2.4.1 has no
   "always run OSD" switch. **Verified numerically.**
3. **`UnionFindDecoder` API.** `UnionFindDecoder(H)` raises `TypeError`,
   because `uf_method` is typed `str` with default `False`.
   `uf_method="peeling"` silently selects matrix-solve mode, since any
   non-empty string is truthy. Peeling needs `uf_method=""`. Lower `llrs`
   grow first. **Verified numerically** (probe; behaviour reproduced in
   `make_union_find`).
4. **Peeling UF aborts or hangs.** On a column of weight > 2, peeling mode
   throws a C++ `runtime_error` that aborts the interpreter instead of
   raising a Python exception; `make_union_find` checks this first. Peeling
   mode also *hangs* (no result after 600 s) on a valid syndrome. Case:
   p = 1e-2, m = 22, sequence 0, sampler seed 31415 + 300000 + 4, 2000
   shots, distinct syndrome no. 1177. It has 26 detection events,
   rank [H|s] = rank H = 94, and one connected graph component with
   boundary; matrix-solve mode solves it at once. **Verified numerically.**
   So the standard set uses ldpc's matrix-solve UF and my pure-Python
   `DelfosseNickersonUF` for textbook peeling.
5. **Unattributed abort.** The first full launch lost a worker to a C++
   `std::invalid_argument: Iterator index out of bounds` in a p = 1e-2,
   m = 14/22 task; the decoder was not identified. **Conjecture:** the
   weighted peeling UF, which ran in that launch and misbehaves as in item 4.
   Since then every task runs in its own subprocess, and a decoder that
   crashes or times out is dropped for that circuit and listed in
   `failed_decoders.csv`.

### In this work package's own runs

- Stage 1 used a 600 s per-task timeout, too short under the CPU pinning
  described below: 39 (circuit, decoder) pairs were dropped
  (`failed_decoders.csv`). The retry timings point to timeouts rather than
  aborts, but the stage-1 log was lost with the node-local scratch
  directory. Stage 3 refilled all 39 pairs, at U = 300.
- Stages 1-2 shared the CPUs to which `srun --overlap` pins steps from
  different agents. They ran 5-20x slower than an idle core (MLD 87 us/shot
  in the grid vs about 10 us/shot alone). The fix is
  `--cpu-bind=none -c 256` (lead's note), used in stage 3. The runtime table
  therefore uses the separate single-process benchmark (`timing.csv`).

## Open questions

1. **Clifford-covariant LUTs.** Can one fixed table set serve every
   sequence? **Conjecture:** an interior table depends only on the Cliffords
   inside its window, the current logical frame (whether a fault flips the
   final X_L depends only on the image of X_L at that time) and the gate
   count of those Clifford words. That would be about 24^w x 3 contexts.
   The terminal step depends on the whole check-free inverse half.
2. **Better hardware decoders at the same cost.** A max-product (Viterbi)
   pass over the 32-state (carry, frame) trellis, or a two-hypothesis frame,
   might close the remaining w = 3-4 gap. **Conjecture.**
3. **Robustness to mis-estimated priors.** On hardware the DEM is estimated
   from data. At d = 2 the decisions hinge on ratios of class probabilities,
   so LUT and MLD decisions may flip under small prior errors. Not tested.
4. **"Always OSD."** Would BP+OSD with OSD always run (not exposed in ldpc
   2.4.1) match `bposd_cs7_serial`, and does the parallel-BP failure grow as
   p falls? The first-order table covers p = 5e-4 only.
5. **Probability-aware union-find.** Can union-find made probability-aware
   (weighted growth that also compares the two logical classes inside a
   cluster) approach MLD at d = 2? Every non-trivial syndrome has two
   first-order explanations that differ by a logical, so plain cluster
   growth has no basis for choosing between them.

## Reproduce

```bash
module load python/3.11
cd <repository root>
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest tests/test_qubit512_ldpc.py          # 40 tests
OUT=studies/correction-decoders/qubit/results/C
J=$(squeue -u sohan100 -n q512shared -h -o %i)
RUN="srun --jobid=$J --overlap --cpu-bind=none -N 1 -n 1 -c 256 python studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py --out $OUT --workers 24 --shots 50000"
# The script is resumable (raw/tasks.jsonl), runs every task in its own subprocess, and stops
# submitting after --max-minutes; rerun the same command until no task is left.
# Stage 1: default decoder set, U = 2000 (completed the 24 most expensive circuits)
$RUN --max-minutes 13 --task-timeout 600 --ldpc-unique 2000
# Stage 2a/2b: remaining circuits, core set, U = 800 (2a) then U = 300 (2b)
$RUN --max-minutes 12 --task-timeout 900 --ldpc-unique 300 \
     --decoders bposd_cs7,bposd_cs7_serial,bplsd_cs7,bplsd_cs7_always,uf_dn,uf_hyper,mwpm_graph
# Stage 3: same seeds, new keys; fills every missing (circuit, decoder) pair, adds ablations
$RUN --max-minutes 10 --task-timeout 900 --ldpc-unique 300 --windows '' --tag _supp \
     --decoders bposd_cs7_serial,bplsd_cs7,bplsd_cs7_always,uf_dn,uf_hyper,bposd_cs7_dominant,bposd_cs7_ldpcmatrices,bposd_0,belief_find,uf_graph
# Exact first-order (single-fault) table, login node
python studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py --out /tmp/fo --first-order --p-list 5e-4 \
    --depths 0,2,6,14,22 --num-seq 6 --workers 4 && cp /tmp/fo/first_order.csv $OUT/
# Controlled timing benchmark, single process
python studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py --out /tmp/tm --timing --p-list 5e-3 \
    --depths 2,6,14,22 --num-seq 1 --windows 2,3,4 && cp /tmp/tm/timing.csv $OUT/
# Re-aggregate summary.csv, calibration.csv, lut_footprint.csv, per_circuit.csv.gz,
# failed_decoders.csv from raw/tasks.jsonl(.gz)
python studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py --out $OUT --aggregate-only
# Revision round (see Revision log)
# LUT first-order check at p = 1e-5 (MLD and LUTs only)
python studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py --out /tmp/fo5 --first-order --p-list 1e-5 \
    --depths 2,6,14,22 --num-seq 6 --windows 1,2,3,4 --decoders "" --workers 4 \
    && cp /tmp/fo5/first_order.csv $OUT/first_order_lut_p1e-5.csv
# Random-sample timing and single-shot latency (script kept with the logs)
python $OUT/logs/C_timing_random_script.py $OUT/timing_random.csv
# between_sequence_se.csv: for each (p, m, decoder), per-sequence
# rb_diff/shots and (n10-n01)/shots from per_circuit.csv.gz; SE = SD over
# the 10 sequences / sqrt(10)
```

The markdown tables in this note were generated from these CSVs. The
held-out tuning scan (`tuning.csv`) was run with a scratch script lost with
the node-local `/tmp`. Its numbers are transcribed from that run's printed
output, and every configuration is named in the CSV.

Files of this work package:

- `src/lrb_qubit/decoders_ldpc.py`: DEM to (H, L, priors) in three column
  conventions, the decomposed graph, ldpc BP+OSD / BP+LSD / UF / belief-find
  wrappers, the pure-Python Delfosse-Nickerson UF and the PyMatching control.
- `src/lrb_qubit/decoders_lut.py`: earliest-slot kernels and
  `SlidingWindowLUT`.
- `tests/test_qubit512_ldpc.py`: 40 tests.
- `studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py`: the grid runner and the
  `--first-order`, `--timing` and `--aggregate-only` modes.
- `results/C/`:
  - CSVs: `summary.csv`, `between_sequence_se.csv`, `calibration.csv`,
    `lut_footprint.csv`, `first_order.csv`, `first_order_lut_p1e-5.csv`,
    `timing.csv`, `timing_random.csv`, `tuning.csv`, `failed_decoders.csv`,
    `per_circuit.csv.gz`;
  - `meta.json`, with the sequences and stages;
  - `raw/tasks.jsonl.gz`;
  - `logs/`.

## Revision log

**Revision 1 (2026-10-28), after the adversarial verification**
(`notes/C-bposd-uf-lut.verify.md`). All main claims were confirmed. Fixes
1-6 were applied as follows; no code was changed.

1. **BP rule named.** All BP results now say product-sum (flooding or
   serial).
   - Noted that product-sum is ldpc 2.4.1's actual default although its
     docstring says `minimum_sum`.
   - Added the verifier's min-sum numbers: 1.3-4.1% failure rate; 0.71 vs
     4.81 pp at (5e-4, 22); 1.2-1.4 pp worse at p >= 2e-3.
   - Added the full-defaults number (10-13% wrong with OSD-0).
   - Added the min-sum rows from my first scan to `tuning.csv`. They were
     on evaluation sequence 2, not held out, and are labelled so. They
     support "product-sum beat min-sum" only at p = 2e-3, m = 6, which the
     Methods now state.
   - Locations: Summary item 1, Methods, Bugs item 2, claims 4, 7 and the
     new 22.
2. **Between-sequence SEs.** Added `between_sequence_se.csv` and Table 1d,
   and stated the bounds as approximate:
   - serial BP+OSD about 0.7-0.8 pp;
   - BP+LSD-always about 1.4 pp;
   - LUT w = 3 about 0.84 pp and w = 4 about 0.56 pp (pooled with the
     verifier's new sequences).

   The table captions now say that the quoted ± is the shot-noise SE
   conditional on the 10 sequences. Locations: Summary items 1 and 3,
   claims 7, 8 and 13.
3. **First-order LUT statement corrected.** Every w >= 2 is first-order
   exact as p -> 0.
   - The 0.10 p of w = 2 at m = 22 is a finite-p effect of decisions made
     at p = 5e-4.
   - My own check at p = 1e-5 (`first_order_lut_p1e-5.csv`) gives excess
     <= 1e-6 p for w = 2-4, equal to MLD's near-tie residual.
   - The first-order section now says its decisions are p-dependent (made
     at p = 5e-4). Location: claim 13.
4. **Memory formula.** It now uses w' = min(w, T): for w >= T a single
   final table is stored (64 bits at m = 0, w = 3, 4). With the w = 1 rule
   it matches all 160 rows; "every (m, w)" was dropped. Location: claim 14.
5. **Timing labels.**
   - Batch throughput (numpy-vectorized MLD and LUT) and per-call latency
     (ldpc, `uf_dn`) are now labelled as different measures.
   - Added `timing_random.csv` (Table 4b): random frequency-weighted and
     random distinct syndromes, throughput with de-duplication, and
     single-shot latency.
   - Single-shot latency of the Python LUT is 0.13-1.1 ms, only 2.4-7.6x
     below BP+OSD-CS7 per call (verifier: 3.5-11x). The large ratios hold
     for batch throughput, and for a hardware LUT only as a conjecture.
   - Flagged that `timing.csv` used a lexicographic, light-biased syndrome
     sample.
   - Locations: Summary item 3, claims 15 and 21.
6. **Minor.**
   - Claim 2: 24-29% are per-depth means; per circuit the range is 20-34%.
   - Claim 3: about 8.0 of the 8.8 pp is the overwrite (dominant columns
     give 0.77 ± 0.09 pp there).
   - Methods: "no selection bias" became "negligible (O(1/N)) bias of a
     syndrome-dependent stopping time".
   - Claim 11: added the verifier's 3.6e-15 on new circuits.

