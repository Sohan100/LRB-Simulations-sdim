# Verification of WP-C (BP+OSD, BP+LSD, union-find, sliding-window LUT)

Independent adversarial review of `notes/C-bposd-uf-lut.md`, its code
(`src/lrb_qubit/decoders_ldpc.py`, `src/lrb_qubit/decoders_lut.py`,
`studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py`) and `results/C/`. I edited none of
those files.

My script is `studies/correction-decoders/qubit/scripts/verify_ldpc.py`. My outputs are in
`results/C-verify/` (`*.json`, `logs/`).

To stay independent of the implementer's helpers, the script does the
following itself:

- It parses every DEM (one check-matrix column per Stim error instruction).
- It computes the single-fault classes P_D(c).
- It builds the ldpc 2.4.1 decoders directly, with explicit settings.
- It decodes per-syndrome decoders on fixed-length shot prefixes.
- It uses 10 new Clifford sequences (`clifford_sequences(n, 22, 777001)`)
  and new sampler seeds.

The reference is the lead's exact `TransferMatrixMLD`. The implementer's
`SlidingWindowLUT`, `DelfosseNickersonUF` and `MatchingControl` are the
objects under test, so they are imported. Labels: **VN** verified
numerically, **VA** verified analytically.

## Verdicts

| # | Claim (short) | Verdict | Required fix |
|---|---|---|---|
| 1 | No decoder beats exact MLD | **confirmed** | – |
| 2 | ldpc's DEM conversion merges equal detector sets and keeps the last-listed class; 24–29% of sets are two-class | **confirmed** | minor: 24–29% are per-depth means; per circuit it is 20–34% |
| 3 | The overwrite costs 206 ± 5 p (m = 22) and up to 8.8 pp | **confirmed** | minor: 8.8 pp is the total excess; about 8.0 pp is due to the overwrite |
| 4 | Parallel BP converges to the wrong class on 5–6% of unambiguous single-fault syndromes and OSD never runs; 4.75 pp at p = 5e-4, m = 22 | **confirmed**, needs rewording | name the BP rule (product-sum); report the min-sum and true-default results (§3) |
| 5 | Serial ≤ 0.7 p, `always_run_lsd` ≤ 2.0 p, dominant ≤ 6.6 p | **confirmed** | – |
| 8 | Serial BP+OSD-CS7 within 0.68 ± 0.05 pp; BP+LSD-always within 1.35 ± 0.07 pp | ranking **confirmed**; the bounds **need more evidence** | give between-sequence SEs; new sequences give 0.83 ± 0.07 and 1.46 ± 0.09 pp |
| 9 | Union-find is 1.2–9.5 pp off MLD | **confirmed** | – |
| 11 | LUT with w ≥ T is exact MLD (LLR to ≤ 3e-15) | **confirmed** | trivial: 3.6e-15 on new circuits |
| 12 | LUT tables and `predict` match brute force | **confirmed** (tests pass) | – |
| 13 | LUT first-order exact for w ≥ 3; w = 2 off by 0.10 p at m = 22; sampled excess ≤ 0.82 pp (w = 3), ≤ 0.54 pp (w = 4) | w ≥ 3 first-order: **confirmed**. "w = 2 is not first-order exact": **refuted** (finite-p artifact). Sampled bounds: **need more evidence** | say every w ≥ 2 is first-order exact as p → 0; quote new-sequence values 0.87 ± 0.02 and 0.58 ± 0.02 pp with between-sequence SEs |
| 14 | Memory = (m+2−w)·2^(4w)·5 + 2^(4w−2) bits, "matches every (m, w) in the grid" | formula **confirmed** for 2 ≤ w ≤ T; "every (m, w)" **refuted** for m = 0, w = 3, 4 | use w' = min(w, T) |
| 15 | LUT 0.03–0.5 µs/shot, MLD 1.4–9 µs/shot, BP+OSD 0.2–16 ms/syndrome | numbers **confirmed** (within 1–16%); the comparison is **not like-for-like** | caveat batch vs single-call timing; add single-shot latency or per-shot throughput; time a random syndrome sample |
| 16 | MLD is calibrated | **confirmed** | – |

## 1. Tests

`python -m pytest tests/test_qubit512_ldpc.py`: **40 passed** in 61 s
(shared allocation 60478392).

## 2. Claims 2–3: ldpc's DEM conversion (VN)

**Minimal toy.** `claim2.json` runs two mechanisms on detectors D0, D1. It
uses ldpc 2.4.1 `detector_error_model_to_check_matrices` and also the full
`SinterBpOsdDecoder.decode_via_files` path with sinter's defaults
(min-sum, scaling 0.625, OSD-0).

| DEM (listing order) | merged column | exact P(L=1 \| D0 D1) | MLD | SinterBpOsd |
|---|---|---|---|---|
| `0.3 D0 D1`; `0.1 D0 D1 L0` | L0 = 1, prior 0.34 | 0.206 | 0 | **1 (wrong)** |
| `0.1 D0 D1 L0`; `0.3 D0 D1` | L0 = 0, prior 0.34 | 0.206 | 0 | 0 |
| `0.1 D0 D1`; `0.3 D0 D1 L0` | L0 = 1, prior 0.34 | 0.794 | 1 | 1 |

The prior is XOR-combined (0.3 + 0.1 − 2·0.03 = 0.34). The observable is
whatever the last-listed mechanism has. Source:
`ldpc/ckt_noise/dem_matrices.py`, line `hyperedge_obs_map[hid] = hyperedge_obs`,
which runs for every instruction.

**Census.** 100 grid DEMs: sequences 0–9, m ∈ {0, 2, 6, 14, 22},
p ∈ {5e-4, 1e-2}.

- Every multi-mechanism detector set is an opposite-class pair.
- ldpc's label equals the last-listed class in all 100 DEMs.
- The two-class fraction, as per-depth means, is 0.286, 0.287, 0.256,
  0.239 and 0.236. Per circuit it ranges from 0.195 to 0.336.
- At m = 22 an average of 115 of about 245 two-class sets end up with the
  less probable label.

**Cost of the labels alone (no decoder).** Sum over the mislabelled sets
of |P0 − P1|, divided by p. Sequences 0–5, p = 5e-4:

| m | 0 | 2 | 6 | 14 | 22 |
|---|---|---|---|---|---|
| label-only cost | 1.7 | 21.1 ± 2.2 | 58.2 ± 3.1 | 130.9 ± 5.0 | 202.9 ± 3.9 |
| implementer's `bposd_cs7_ldpcmatrices` excess | 1.73 | 21.25 | 58.17 | 132.12 | 205.75 |

So the first-order cost is essentially the labelling alone. The sampled
"8.8 pp" (p = 2e-3, m = 14) is the total excess. Correctly labelled
merged columns (`dominant`) already have 0.77 pp there, so the overwrite
costs about 8.0 pp. I did not re-run the sampled ldpc-matrices cells.

## 3. Claims 4, 5, 8: BP failure on single faults (VN)

**Method.** For one circuit (and 11 more), I decoded the syndrome of every
distinct single-mechanism detector set with `BpOsdDecoder` /
`BpLsdDecoder`. The check matrix has one column per DEM mechanism. I
compared the output class with exact MLD and with the first-order argmax,
and recorded ldpc's `converge` flag (`claim4_m6.json`, `claim4_m22*.json`).

**ldpc defaults are not what the docstring says.** In ldpc 2.4.1,
`BpOsdDecoder(H, error_channel=...)` resolves to:

- `bp_method = product_sum`, although the docstring says `minimum_sum`.
  The base class reads `kwargs.get("bp_method", 0)`, and "0" maps to
  product-sum.
- `max_iter = n` (number of columns), parallel schedule, OSD-0.

`SinterBpOsdDecoder` instead defaults to min-sum with scaling 0.625.

**The requested circuit:** grid sequence 0, m = 6, p = 5e-4. It has 375
columns, 297 sets (219 one-class, 78 two-class) and an MLD floor of
94.3 p. MLD itself differs from the first-order argmax only on 11
near-tied two-class sets.

| configuration | one-class sets decoded to the wrong class (vs MLD) | of which BP had converged | first-order excess / p |
|---|---|---|---|
| ldpc defaults (product-sum, max_iter = 375, parallel, OSD-0) | **19 / 219** | 16 | 39.0 |
| same, `schedule="serial"` | **9 / 219** | 0 | 0.90 |
| implementer's `bposd_cs7` (product-sum, 5 iterations, parallel, OSD-CS7) | 16 / 219 | 16 | 38.6 |
| implementer's `bposd_cs7_serial` | 0 / 219 | 0 | 0.10 |

The 9 serial errors with OSD-0 are OSD-0 errors on low-mass sets, not BP
convergence.

**Pooled over 6 sequences.** One-class wrong counts are sets with the
wrong class over all one-class sets; "BP-conv" counts those where BP had
converged.

| configuration | m = 6, grid seqs 0–5 | m = 6, new seqs 0–5 | m = 22, grid seqs 0–5 | m = 22, new seqs 0–2 |
|---|---|---|---|---|
| `bposd_cs7` (product-sum, parallel) | 62.56 ± 7.20 p; 79/1331 = 5.9%, all BP-conv | 52.9 ± 12.2 p; 5.7% | 207.64 ± 13.81 p; 279/4729 = 5.9%, all BP-conv | 191.8 ± 8.6 p; 5.5% |
| `bposd_cs7_serial` | 0.27 ± 0.05 p; 0 | 0.21 ± 0.09 p; 0 | 0.70 ± 0.13 p; 0 | 0.50 ± 0.10 p; 0 |
| `bplsd_cs7_always` | 0.52 ± 0.13 p | 0.48 ± 0.09 p | 2.01 ± 0.10 p | 2.04 ± 0.14 p |
| dominant columns, parallel | – | – | 6.62 ± 1.70 p | 7.69 ± 1.05 p |
| ldpc defaults (OSD-0) | 65.3 ± 8.0 p; 11.0% wrong (5.9% BP-conv) | 55.9 ± 11.7 p; 10.2% | 219.4 ± 15.7 p; 12.5% (5.9% BP-conv) | 205.8 ± 6.6 p; 13.4% |
| min-sum α = 1, parallel, CS7 | 4.5 ± 0.3 p; 1.5% | 2.4 ± 0.9 p; 1.4% | 16.6 ± 2.5 p; 1.3% | 14.3 ± 0.6 p; 1.3% |
| min-sum α = 0.625, parallel, CS7 | 22.8 ± 2.5 p; 3.3% | 26.3 ± 10.3 p; 3.6% | 99.1 ± 10.2 p; 4.1% | 80.7 ± 8.9 p; 3.3% |

Every grid-sequence number for the implementer's configurations matches
`results/C/first_order.csv` to all printed digits. My check matrix and
first-order bookkeeping were written independently. Raising product-sum to
`max_iter = n` changes nothing (62.56 p at m = 6). BP converges early on
the wrong solution.

**Sampled, new sequences, fixed prefixes** (`sampled_bp_*.json`,
`sampled_minsum.json`). Excess in pp, ± between-sequence SE. The
implementer's value is in brackets.

| p, m | `bposd_cs7` | `bposd_cs7_serial` | `bplsd_cs7_always` | min-sum α=1 parallel CS7 |
|---|---|---|---|---|
| 5e-4, 22 | **4.81 ± 0.28** [4.75] | 0.09 ± 0.01 [0.10] | 0.24 ± 0.02 [0.33] | 0.71 ± 0.08 |
| 5e-4, 6 | 1.99 ± 0.29 [2.01] | 0.03 ± 0.00 [0.03] | – | 0.20 ± 0.04 |
| 2e-3, 14 | 0.52 ± 0.03 [0.40] | 0.44 ± 0.02 [0.30] | – | 1.88 ± 0.21 |
| 5e-3, 6 | 0.76 ± 0.04 [0.70] | **0.75 ± 0.04** [0.68] | **1.46 ± 0.09** [1.34] | 1.98 ± 0.25 |
| 1e-2, 2 | 0.86 ± 0.08 [0.71] | **0.83 ± 0.07** [0.68] | 1.39 ± 0.08 [1.35] | 1.99 ± 0.25 |

**Reading.**

- **Claim 4 is confirmed.** The 5–6% rate, "BP converged, so OSD/LSD never
  ran", the first-order costs, and 4.75 pp at (5e-4, 22) all reproduce
  exactly or within noise, on new sequences too.
- **The failure belongs to the BP update rule as well as the schedule.**
  Min-sum flooding BP shows the same failure, but on 1.3–1.5% (α = 1) or
  3.3–4.1% (α = 0.625) of one-class sets. Paired on the same shots, min-sum
  is better at p = 5e-4 (0.71 vs 4.81 pp at m = 22) but 1.16–1.36 pp worse
  at p ≥ 2e-3.
- Product-sum is therefore a defensible choice at high p. But
  `tuning.csv` has no min-sum rows to support the code comment "product-sum
  BP beat min-sum".
- With ldpc's true defaults (OSD-0), 10–13% of unambiguous single-fault
  syndromes get the wrong class: 5.5–5.9% from converged BP and the rest
  from OSD-0.
- **Claim 5 is confirmed**, including dominant columns at 6.62 ± 1.70 p.
- **Claim 7 is consistent**: parallel and serial differ by ≤ 0.03 pp at
  p ≥ 5e-3 and by 0.08 pp at (2e-3, 14).
- **Claim 8's ranking holds.** Serial BP+OSD-CS7 is the closest
  general-purpose decoder everywhere I tested. Its bounds do not carry
  over to new sequences: 0.83 ± 0.07 pp (serial) and 1.46 ± 0.09 pp
  (LSD-always), about 1–2 between-sequence SEs above the quoted maxima.

The quoted "±" values are shot-noise SEs conditional on the 10 sequences.
The between-sequence SEs from `per_circuit.csv.gz` are larger: 0.39 pp for
`bposd_cs7` at (5e-4, 22) against the quoted 0.30, and 0.048 pp for the
serial maximum.

## 4. Claims 11–14: sliding-window LUT

**What a w-round window can see (VA).** Every DEM mechanism touches one
time slot or two adjacent slots (`earliest_slot_kernels` raises
otherwise). Take one fault with earliest slot u and syndrome (x on u, y on
u+1).

1. **Steps t < u.** The residual is σ = 0. Committing (κ', λ) ≠ (0, 0)
   needs at least two group-t faults, order p², against about 1 × O(p) for
   committing nothing. So nothing is committed.
2. **Step u.** The window covers slots u … u+w−1, which for w ≥ 2 includes
   slot u+1.
   - The true commitment (y, l) scores P_D(l) × (1 − O(p)).
   - Its partner (y, 1−l) scores P_D(1−l) × the same factor, so l versus
     1−l is decided by the first-order argmax.
   - Any (y' ≠ y, ·) also needs a later fault to explain y ⊕ y' on slot
     u+1, so it is O(p²).
3. **Later steps.** The residuals are 0, so nothing is committed. The final
   step on an all-zero window returns parity 0, since the data5 terminal
   has no undetectable logical error.
4. **Faults in the final window** are decided by the final step's exact
   marginal MAP.

Hence **every w ≥ 2 is first-order exact as p → 0**. w = 1 is not: at step
u it cannot see y, so it cannot separate (x, y, l) from a more probable
(x, y', l'), for example a data error from a measurement error.

At finite p, w = 2 also counts competitors (y', l') whose follow-up fault
fires the unseen slot u+2. Full MLD knows that slot is silent. Such
order-p² terms can outweigh a weak single fault at p = 5e-4. With w = 3
the follow-up fault must leave slot u+2 clean, as in full MLD, so the
first difference from MLD moves to O(p³).

**First-order numerics, 10 new sequences, m = 2, 6, 14, 22** (VN,
`lutfo.json`):

- **p = 5e-4.**
  - w = 3 and w = 4 have excess ≤ 7.8e-5 p per circuit. MLD itself reaches
    the same value. Every set off the first-order argmax is a near-exact
    tie (relative gap ≤ 5.8e-5).
  - w = 2 has 0.02 ± 0.02 p at m = 22: one sequence at 0.20 p, a one-class
    set. It is zero at the other depths.
  - w = 1 has 6.3–31.4 p.
- **p = 1e-5 and 1e-6.** w = 2, 3 and 4 are all ≤ 1.6e-6 p and ≤ 1.6e-7 p
  respectively. This is the same near-tie residual as MLD's, and it shrinks
  in proportion to p.

So "first-order exact for w ≥ 3" is confirmed. The reported 0.10 ± 0.04 p
for w = 2 at m = 22 is a finite-p effect of the window, not a first-order
property. The first-order table is computed with p-dependent decisions at
p = 5e-4 and should say so.

**Claim 11.** With w = T the LUT LLR matches `TransferMatrixMLD` to
**3.6e-15**, with identical decisions and identical infinite LLRs. This
covers 48 new-sequence circuits: m = 0–3, data5 and data2 terminals,
p = 2e-3 and 3e-2.

**Sampled spot-check, 10 new sequences × 50,000 shots, joint rule** (VN,
`sampled_lut.json`). Excess in pp, ± between-sequence SE; implementer in
brackets.

| p, m | w = 2 | w = 3 | w = 4 |
|---|---|---|---|
| 5e-4, 22 | 0.309 [0.314] | 0.082 [0.087] | 0.059 [0.060] |
| 2e-3, 6 | 0.897 [0.859] | 0.327 [0.315] | 0.193 [0.204] |
| 2e-3, 22 | 1.483 [1.470] | 0.605 [0.604] | 0.450 [0.445] |
| 5e-3, 6 | 1.814 ± 0.053 [1.739] | **0.869 ± 0.022** [0.819] | **0.575 ± 0.015** [0.541] |
| 5e-3, 14 | 1.264 [1.199] | 0.671 [0.640] | 0.524 [0.494] |
| 1e-2, 2 | 1.533 ± 0.112 [1.279] | 0.378 ± 0.037 [0.331] | 0 (w = T) [0] |
| 1e-2, 6 | 1.189 [1.070] | 0.695 ± 0.028 [0.613] | 0.497 ± 0.021 [0.433] |

All raw paired excesses are ≥ 0 within noise.

- The new maxima exceed the claimed "≤ 0.82 ± 0.01" and "≤ 0.54". The
  quoted ±0.01 is a shot-noise SE.
- The implementer's own between-sequence SE at (5e-3, 6) is 0.033 pp, with
  per-sequence values from 0.65 to 0.96.
- The difference is 1.3 combined SEs, so it is ordinary sequence-to-
  sequence variation. Pooling both sequence sets gives about 0.84 ± 0.02
  pp (w = 3) and 0.56 ± 0.02 pp (w = 4) at the worst point.

**Claim 12.** The brute-force tests pass. I did not write a second brute
force.

**Claim 14 (VA).**

- For 2 ≤ w ≤ T, each interior window holds only 4-bit check slots, which
  gives T − w tables of 2^(4w) 5-bit entries.
- The final window holds 4(w−1)+2 bits. The formula follows and matches
  `lut_footprint.csv` (e.g. 52.625 and 802 KiB at m = 22).
- For w > T, which occurs at m = 0 with w = 3 and 4 in the grid, the
  decoder stores one 64-bit final table. The formula gives −19,456 and
  −638,976 bits.
- "Matches table_bits exactly for every (m, w) in the grid" is therefore
  false for those two cells. Replace w by w' = min(w, T).

## 5. Claim 9: union-find (VN)

New sequences, 3000 shots each (`sampled_uf.json`):

| p, m | `uf_dn` | `mwpm_graph` |
|---|---|---|
| 2e-3, 6 | 7.50 ± 0.29 pp [7.39] | 6.38 ± 0.22 pp [6.12] |
| 2e-3, 14 | 9.54 ± 0.26 pp [9.50] | 8.26 ± 0.23 pp [8.53] |

Confirmed.

## 6. Claim 15: runtime (VN)

`timing.json`: login node (load about 40), single process, sequence 0,
and the implementer's configurations.

| p, m | MLD batch µs/shot | LUT w=3 batch µs/shot | MLD single shot µs | LUT w=3 single shot µs | BP+OSD-CS7 µs/syndrome (lexicographic 300 / random 300) | BP+OSD-CS7 µs/shot with de-dup | serial: lexicographic / random / per shot |
|---|---|---|---|---|---|---|---|
| 5e-3, 6 | 2.89 [2.87] | 0.094 [0.093] | 234 | 136 | 610 [611] / 702 | 577 | 3417 [3310] / 3735 / 2959 |
| 5e-3, 22 | 9.19 [8.98] | 0.389 [0.404] | 663 | 469 | 5580 [4816] / 5356 | 5387 | 15801 [15751] / 16734 / 17203 |
| 5e-4, 6 | 2.97 | 0.097 | 231 | 136 | 465 / 473 | 38 | 2543 / 2833 / 213 |
| 5e-4, 22 | 9.39 | 0.407 | 651 | 463 | 2857 / 3794 | 1230 | 10909 / 19010 / 8702 |

The implementer's numbers reproduce. They do not compare like with like:

- **Different units.** The LUT and MLD figures are numpy-vectorized batch
  throughput over 50,000 shots. The BP+OSD figures are per-call latency.
  - Per shot with de-duplication (a fair throughput comparison), BP+OSD
    costs 38 µs–5.4 ms, so the large ratio stands at p = 5e-3.
  - In single-shot latency, the same Python LUT takes 136–469 µs. That is
    only 3.5–11 times below parallel BP+OSD-CS7 (21–41 times below serial),
    not 10^4.
  - Ns-scale hardware lookups remain a conjecture.
- **Biased syndrome sample.** The per-syndrome sample is the first 300
  distinct syndromes in byte-lexicographic order (`unique_rows` sorts).
  These are light syndromes: mean weight 3.3–10 against 3.9–18.5 for a
  random sample. At p = 5e-4, m = 22 this understates the per-syndrome cost
  by 25% (parallel) and 43% (serial). At the p = 5e-3 benchmark point it
  ranges from 13% too low to 4% too high.

The note's Methods disclose the units. The summary's sentence ("0.03–0.5
us/shot … against … 0.2–16 ms per syndrome") does not.

## 7. Claims 1 and 16 (VN)

**Claim 1.** MLD is Bayes-optimal for the exact DEM (VA). The
posterior-weighted excess is ≥ 0 by construction, so only the raw paired
excess can test the claim.

- In `results/C/summary.csv` no raw excess is below −2 SE (most negative
  z = −1.72).
- In every one of my new-seed runs (LUT, UF, MWPM, BP variants), each raw
  paired excess is ≥ 0 within 2 SE.

**Claim 16.** I tested calibration on 16 new-seed points (3.79 M shots)
with the exact conditional variance Σ q(1 − q) of the MLD error count.
This is sharper than the note's conservative denominator.

- Points: χ² = 11.9 for 16 dof (p = 0.75), max |z| = 1.73.
- 160 circuits: z has mean −0.09 and SD 0.98, χ²/dof = 0.96.

Confirmed.

## Minor methodological notes (no verdict change)

- **Stopping rule.** The ldpc runs score the longest prefix with at most U
  distinct syndromes. That is a syndrome-dependent stopping time: the next
  shot, always a new and usually heavier syndrome, is excluded. The bias is
  O(1/N) and negligible. My fixed-prefix runs reproduce the headline cells.
  "No selection bias" should read "negligible bias".
- **Tuning is unverifiable.** `tuning.csv` was transcribed from a lost
  script's printed output, and it has no min-sum rows.

## Required fixes (summary)

1. **Claims 4 and 7, Summary item 1.** Say product-sum flooding BP. Note
   that it is ldpc 2.4.1's actual default although the docstring says
   `minimum_sum`.
   - Min-sum flooding BP fails the same way on 1.3–4.1% of one-class
     syndromes. It costs 0.71 pp at (5e-4, 22) but is 1.2–1.4 pp worse at
     p ≥ 2e-3.
   - With true ldpc defaults (OSD-0), 10–13% of unambiguous single-fault
     syndromes get the wrong class.
   - Add min-sum rows to `tuning.csv`, or drop "product-sum beat min-sum".
2. **Claims 8, 13 and Summary items 1, 3.** Quote between-sequence SEs.
   State the maxima as approximate:
   - serial BP+OSD about 0.7–0.8 pp (new: 0.83 ± 0.07);
   - BP+LSD-always about 1.4 pp (new: 1.46 ± 0.09);
   - LUT w = 3 about 0.84 pp (new: 0.87 ± 0.02);
   - LUT w = 4 about 0.56 pp (new: 0.58 ± 0.02).
3. **Claim 13.** Every w ≥ 2 is first-order exact as p → 0 (argument in §4;
   numerics at 1e-5 and 1e-6). The w = 2 value of 0.10 p at m = 22 is a
   finite-p (p = 5e-4) effect. The first-order table uses p-dependent
   decisions at p = 5e-4.
4. **Claim 14.** Use w' = min(w, T). Drop "every (m, w) in the grid":
   m = 0, w = 3, 4 store 64 bits.
5. **Claim 15, Summary item 3.** Label the LUT and MLD numbers as batch
   throughput and the BP+OSD numbers as per-call latency.
   - Add single-shot latency (LUT 136–469 µs in Python, 3.5–11 times below
     parallel BP+OSD-CS7) or per-shot throughput with de-duplication for
     every decoder.
   - Time a random or frequency-weighted sample of syndromes instead of the
     lexicographically first 300.
6. **Minor.**
   - Claim 2: 24–29% is per-depth means (per circuit 20–34%).
   - Claim 3: about 8.0 of the 8.8 pp is due to the overwrite.
   - Claim 11: 3.6e-15 on new circuits.

## Reproduce

```bash
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
OUT=studies/correction-decoders/qubit/results/C-verify
V=studies/correction-decoders/qubit/scripts/verify_ldpc.py
python -m pytest tests/test_qubit512_ldpc.py                       # 40 passed
python $V claim2 --out $OUT                                         # login node, 9 s
python $V claim4 --out $OUT --num-grid 6 --num-new 6 --depths 6 --p 5e-4 --workers 12 --tag _m6
python $V claim4 --out $OUT --num-grid 6 --num-new 3 --depths 22 --p 5e-4 --workers 9 --tag _m22 \
  --configs osd_cs7_ps_it5_parallel,osd_cs7_ps_it5_serial,osd_cs7_ms1_it5_parallel,osd_cs7_ms0625_it5_parallel,lsd_cs7_ps_it5_always,osd_ldpc_defaults
python $V claim4 --out $OUT --num-grid 6 --num-new 3 --depths 22 --p 5e-4 --workers 9 --tag _m22_dominant \
  --columns dominant --configs osd_cs7_ps_it5_parallel,osd_cs7_ps_it5_serial
python $V lutfo --out $OUT --num-new 10 --depths 2,6,14,22 --p-list 5e-4,1e-5,1e-6 --workers 4
python $V sampled --out $OUT --num-new 10 --points 5e-4:22,2e-3:6,2e-3:22,5e-3:6,5e-3:14,1e-2:2,1e-2:6 \
  --shots 50000 --windows 2,3,4 --decoders '' --workers 7 --tag _lut
python $V sampled --out $OUT --num-new 10 --points 2e-3:14,2e-3:6 --shots 3000 --prefix 3000 --windows '' \
  --decoders uf_dn,mwpm_graph --workers 7 --sample-seed 515151 --tag _uf
python $V sampled --out $OUT --num-new 10 --points 5e-3:6,1e-2:2 --shots 3000 --prefix 3000 --windows '' \
  --decoders osd_cs7_ps_it5_serial,lsd_cs7_ps_it5_always,osd_cs7_ps_it5_parallel --workers 7 --sample-seed 525252 --tag _bp_highp
python $V sampled --out $OUT --num-new 10 --points 5e-4:22 --shots 5000 --prefix 5000 --windows '' \
  --decoders osd_cs7_ps_it5_parallel,osd_cs7_ps_it5_serial,lsd_cs7_ps_it5_always,osd_cs7_ms1_it5_parallel --workers 7 --sample-seed 535353 --tag _bp_lowp
python $V sampled --out $OUT --num-new 10 --points 5e-3:6,1e-2:2,2e-3:14,5e-4:6 --shots 3000 --prefix 3000 --windows '' \
  --decoders osd_cs7_ms1_it5_parallel,osd_cs7_ps_it5_parallel,osd_cs7_ps_it5_serial --workers 16 --sample-seed 545454 --tag _minsum
python $V timing --out $OUT --p-list 5e-3,5e-4 --depths 6,22 --configs osd_cs7_ps_it5_parallel,osd_cs7_ps_it5_serial
```

The heavy runs used `srun --jobid=60478392 --overlap --cpu-bind=none -c 256`
with ≤ 16 workers and ≤ 15 min per run. `claim2`, `lutfo` and `timing` ran
on the login node under `nice`. The LLR check of claim 11 was an inline
snippet: `SlidingWindowLUT(dem, slots, T).predict(..., return_llr=True)`
against `TransferMatrixMLD.log_likelihood_ratio` on 3000 shots per circuit.
