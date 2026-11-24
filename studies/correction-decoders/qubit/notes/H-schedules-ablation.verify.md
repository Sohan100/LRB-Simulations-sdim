# Verification of WP H: check schedules and noise ablation

Independent adversarial verification, 2026-10-28. Script:
`studies/correction-decoders/qubit/scripts/verify_variants.py`. Outputs: `results/H-verify/` (JSON,
CSV and logs). Labels: **VN** verified numerically, **VA** verified
analytically. Claim numbers follow my brief; the notes' own claim-table
numbers are given in brackets where they differ.

## Verdicts

| # | claim | verdict | evidence in one line |
|---|---|---|---|
| 1 | serial variant reproduces the lead's circuits | confirmed | 154 H tests pass; variants.py, the lead's builder and my builder give identical DEMs |
| 2 | data idle layers per round 5/3/2, no reset 4/2/1 | confirmed | counted in my own circuits, per round and over whole circuits |
| 3 | data2 floor 68p/15 serial, 48p/15 with one measurement layer | confirmed, wording fix | exact at every depth for 20 sequences; zx (two layers) also gives 48p/15, and X-check CNOTs first give 54p/15 |
| 5 [6] | first-order E1 per step 11.76/8.08/6.28/4.45/0.73 p | confirmed | identical to machine precision on their sequences; new sequences within 0.45σ |
| 6 [7] | parallel removes about 50% of the "correction penalty", no reset 66-68% | confirmed, wording fix | 49.6-49.7% and 66.2-66.3% at first order, 50.6-50.8% and 67.0-67.4% sampled on new sequences; the name is misleading |
| 9 [10] | decoded error rises with every added check; no checks optimal at all p | confirmed, scope fix | none of 2,640 single-check additions lowered E1; "all p" means all tested p |
| 10 [11] | noiseless checks gain 0.010p; break-even needs 600-1700× quieter checks | confirmed, uncertainty fix | gain 0.0103(14)p at first order and 0.011(2)p at p = 2e-3 to 1e-2 on identical shots; new sequences give 470-1340× |
| 11 [12] | checks every 4-11 Cliffords give the lowest LRB-D infidelity, 0.57-0.63% vs 0.88% | refuted as stated; numbers confirmed | exact values match the sampled ones, but forward+inverse placements reach 0.45-0.46% at equal or higher acceptance |
| 12 [13] | executing only the used rounds beats the repo protocol | confirmed, caveat on f | exact A(22) 0.0631 → 0.4614 and F(22) 0.024 → 0.931; the quoted f = 0.901 depends on the fit weights (0.886 with exact data) |
| 13 [14] | inverse-half checks give no benefit | refuted for LRB-D; confirmed for decoding | at equal round count, inverse placement lowers accepted infidelity by 15-28% in 10/10 sequences at unchanged acceptance |

No code bug found. The implementer's circuits are DEM-identical to an
independent construction, and every number I recomputed matches their
first-order tables to all printed digits.

## What I did

1. Re-ran the tests. `tests/test_qubit512_variants.py` gives 154 passed in
   17 s. The lead's `tests/test_lrb_qubit_model.py` gives 64 passed.
2. Wrote my own circuit builder. It does not import `variants.py`. It
   implements the SI1000 channels from CONTEXT.md and the check circuits. It
   has four schedules: serial, zx, parallel (one measurement layer of all four
   ancillas) and parallel with the X-check CNOTs first. For measure-without-
   reset it XORs consecutive ancilla records. It builds two detector bases:
   raw syndromes and difference detectors. From the lead's package I used
   only `code.py` (gate decompositions, encoder), the sequence generator,
   `mld.TransferMatrixMLD`, `exact.RawPolicyEvaluator` / `exact_all_rounds` and
   `fits.fit_fixed_offset`. Stabilizer back-propagation uses stim's
   `PauliString.before` and asserts a + sign.
3. Sequences: 10 new ones (`clifford_sequences(10, 22, 777000001)`), plus the
   implementer's 10 (seed 20261028) to test exact reproduction.
4. First-order census at p = 1e-6, with single DEM mechanisms grouped by
   symptom: E1 = Σ min(P0, P1) plus the undetectable P1, R1 = Σ (P0 + P1),
   A2 = Σ P0·P1. Slopes are least-squares fits over the nine depths, one per
   sequence.
5. Exact LRB-D at p = 5e-3 and 2e-3 with the lead's evaluators, all depths,
   both sequence sets.
6. Two finite-p checks. The first decodes perfect-check shots twice, on
   identical shots. The second samples decoded rates with 1e5 shots per
   circuit.
7. Last, compared my circuits with `variants.py`.

### Builder validation (`selftest.json`, `compare.json`)

- 210 cases cover all eight schedule/reset variants, const0 and inverse-check
  variants, at depths 0, 1, 3, 6 and 22, with both readouts and both detector
  bases. In every case the noiseless circuit gives no detection events and no
  observable flips, the noisy DEM builds, and the transfer-matrix MLD accepts
  every difference-detector DEM.
- My serial builder gives the lead's `build_lrb_circuit` DEM symptom by
  symptom. I checked 24 cases, including forward subsets and inverse checks;
  the largest probability difference is 0.0.
- Raw and difference detectors give the same E1, R1 and A2 (4e-16 relative).
  This must hold: the two bases differ by an invertible GF(2) map, which only
  relabels symptom classes.
- `RawPolicyEvaluator` equals brute-force inference on my reset and no-reset
  circuits at depths 1 and 2 (5e-16). It had not been tested on
  measure-without-reset circuits before.
- Exact acceptance and accepted-flip probability agree with 4e5-shot Monte
  Carlo computed straight from measurement records, with no detectors
  involved. Three circuits at p = 5e-3 give |z| ≤ 1.5.
- At the end, `variants.py` gave DEMs identical to mine for serial, zx and
  parallel, each with and without reset. I checked 4 sequences, depths 0, 2,
  6 and 22, every round and unif=4; the largest difference is 0.0.

## Evidence per claim

### 1. Serial variant equals the lead's builder: confirmed (VN)

`test_serial_reproduces_lead_builder` (56 cases, string equality) passes.
Independently, both `variants.py` and my builder reproduce the lead's DEM
exactly.

### 2. Data idle layers per round: confirmed (VA, VN)

In one forward round of my construction, each data qubit gets
DEPOLARIZE1(2p) in 5, 3 and 2 layers (serial, zx, parallel), and 4, 2 and 1
without reset. A whole depth-6 circuit gives 34, 20, 13 and 28, 14, 7 layers.
That is 6·L plus the terminal round's measurement layers, since the terminal
round has no reset. The channel mass per round matches the notes' table 1.2
term by term: data idle 40/20/10p, ancilla idle 24/8/0p, reset idle 10p,
reset flips 8p, measurement flips 20p, measured-qubit depolarizing 4p, CNOTs
12p, ancilla H 0.4p.

### 3. Data2 floor: confirmed for the repo CNOT order (VA, VN); wording fix

I built real two-qubit-readout circuits instead of masking a data5 DEM. Each
variant ran on 20 sequences at all 9 depths. Undetectable logical mass / p:

| schedule | with reset | no reset |
|---|---|---|
| serial | 4.5333 = 68/15 | 4.5333 |
| zx | 3.2000 = 48/15 | 3.2000 |
| parallel | 3.2000 | 3.2000 |
| parallel, X-check CNOTs first (my robustness variant) | 3.6000 = 54/15 | 3.6000 |

The minimum equals the maximum in every row. The analytic count holds: 8p/3
from data 0 idling in the M7 and M8 layers, 4p/3 from data 3 in M8, and
2 × 4p/15 from CNOT 7→0 and 8→3 faults with a Z or Y data part and an I or X
ancilla part.

Fix: "48p/15 with one measurement layer" is imprecise. What counts is the
number of measurement layers after the X-check CNOTs on data 0 and 3. zx has
two layers and also gives 48p/15. A one-layer schedule with the X-check
CNOTs first gives 54p/15. Write "48p/15 for zx and parallel with the repo's
CNOT order".

### 5 [notes 6]. First-order decoded error per step: confirmed (VN)

E1 slope per depth step / p, mean (s.e.m.) over 10 sequences:

| variant | claimed | mine, their sequences | mine, new sequences | z, new vs claimed |
|---|---|---|---|---|
| serial | 11.76(19) | 11.76(19) | 11.60(31) | −0.45 |
| zx | 8.08(13) | 8.08(13) | 7.99(21) | −0.38 |
| parallel (one measurement layer) | 6.28(10) | 6.28(10) | 6.22(15) | −0.32 |
| parallel, no reset | 4.45(7) | 4.45(7) | 4.42(11) | −0.27 |
| no checks (const0) | 0.73(3) | 0.73(3) | 0.74(3) | +0.36 |

On their sequences the agreement is to machine precision. The same holds for
R1 and A2, and for every row of `first_order_tables.md` I recomputed: unif=2
to 11, const=2 to 4 and terminal-only for all three schedules, inverse
k = 1, 2, 4, 11, and the check-noise scan at C = 0, 0.01, 0.1 and 1. Moving
the X-check CNOTs first changes the parallel value by less than 0.1p
(6.22(10) on their sequences).

### 6 [notes 7]. Penalty fractions: numbers confirmed (VN); rename and restate

| fraction of the penalty removed | parallel | parallel, no reset |
|---|---|---|
| claimed, first order | 49.7% | 66.3% |
| mine, first order, their sequences | 49.70(12)% | 66.28(13)% |
| mine, first order, new sequences | 49.56(20)% | 66.18(23)% |
| claimed, sampled at 2e-3 / 5e-3 | 51.0(2)% / 51.0(2)% | 67.2(2)% / 67.7(2)% |
| mine, sampled, new sequences and seeds, 2e-3 / 5e-3 | 50.8(2)% / 50.6(3)% | 67.0(3)% / 67.4(3)% |

My sampled r/p at 2e-3 and 5e-3 match the notes' table 2.2: serial
12.96(28) and 14.06(24), parallel 6.76(13) and 7.34(12), no reset 4.79(8)
and 5.11(8), const0 0.767(19) and 0.769(19).

The definition (part 4 of my brief):

- The notes define it twice, in Summary item 3 and in section 2.2: the
  penalty is r(MLD5, every round checked, serial) − r(no-check decoding), and
  the fraction is (f_sched − f_serial)/(f_const0 − f_serial). That equals the
  ratio of the r differences. The quantity is well defined and I could
  reproduce it.
- The name is wrong. No-check decoding also corrects, because it decodes the
  final X parities. So the quantity is not a cost of correction. It is the
  extra decoded error from a noisy check round every step: perfect checks
  bring E1 to 0.01p below const0, so all of the penalty is check-round noise.
- The headline and section 3 item 3 quote "50%" and "66 to 68%" without the
  definition. Without it, a reader will take "correction penalty" to mean
  correction against detection, or decoded LRB against physical RB.
- The MLD2 fractions use const0 decoded with the five-qubit readout as the
  reference. That mixes two decoders, and the notes should say so.

Required fix: rename it, for example to "every-round check penalty
(decoded)". Wherever a fraction appears, restate the definition with the
absolute values: 11.0p penalty, 5.5p left with parallel, 3.7p left with
parallel and no reset.

### 9 [notes 10]. Monotone in added checks: confirmed at first order (VN); scope fix

- Random nested chains at depth 22. I used two random orders for each of the
  10 new sequences and each of serial, parallel and parallel without reset.
  Each chain starts at const0, adds the terminal round, then adds the 22
  forward and 21 inverse positions one at a time. All 2,640 additions raised
  E1(22). The smallest increases were +1.37p (parallel, no reset), +2.70p
  (parallel) and +5.93p (serial).
- Even noiseless checks gain only 0.011p per step at finite p (claim 10). A
  perfect terminal check changes no decoding decision at all.
- In the lead's grid, const0 decoding beats every-round MLD at all 15 p up to
  5e-2.

Scope fix: "at all p" is supported for every-round against none up to 5e-2,
for the tested densities at 5e-4 to 5e-3, and at first order. Intermediate
densities above 5e-3 were never sampled. Write "at all tested p". Also, the
sampled table in 2.3 lists const=2 (2.25p at 5e-3) above unif=11 (1.94p).
That is not a violation, because their check sets are not nested and const=2
jumps from 0 to 1 forward round between m = 0 and m = 2. Say that const rows
are not comparable per step.

### 10 [notes 11]. Noiseless checks and break-even: confirmed at first order (VN); state the uncertainty

| | their sequences | new sequences |
|---|---|---|
| ΔE1 per step, perfect checks every round − const0 | −0.0103(14)p | −0.0126(25)p |
| C* and 1/C*, serial | 5.97e-4, 1674× | 7.46e-4, 1341× |
| C* and 1/C*, zx | 8.46e-4, 1181× | 1.06e-3, 946× |
| C* and 1/C*, parallel | 1.07e-3, 931× | 1.34e-3, 746× |
| C* and 1/C*, parallel, no reset | 1.72e-3, 582× | 2.15e-3, 466× |

- E1(C) − E1(const0) is linear in C up to at least 1.5e-3, with equal steps
  on a 1e-4 grid, so the notes' two-point interpolation is exact.
- The small perfect-check gain sets C*. That gain has a 14-20% s.e.m. over 10
  sequences, so the factor should read "about 500 to 1700× (first order,
  about ±25% from sequence sampling)".
- New at finite p: a perfect-check circuit has the same noise as const0. So I
  decoded the same shots with all detectors and with only the final X
  parities. Δr = −0.0110(19)p, −0.0113(20)p and −0.0108(18)p at 2e-3, 5e-3
  and 1e-2 (3e5 shots per circuit, 10 new sequences). The gain does not grow
  with p, which supports the claim.
- Sparse checks gain nothing even when perfect. ΔE1(C = 0) is −0.0008(8)p for
  unif=4 and exactly 0 for unif=11 and terminal-only.

### 11 [notes 12]. Best LRB-D schedule: numbers confirmed, "lowest" refuted (VN, exact)

Exact LRB-D with the final data check, p = 5e-3, m = 22, parallel without
reset, only the policy's forward rounds executed. Values are the mean (s.e.m.
of the sequence spread) over 10 sequences and carry no shot noise.

| forward checks | claimed 100(1−F22) | exact, their sequences | exact, new sequences | claimed A(22) | exact A(22) |
|---|---|---|---|---|---|
| every round | 0.880(61) | 0.895(24) | 0.886(31) | 0.00366 | 0.00368 |
| unif=3 | 0.628 (summary.json) | 0.617(28) | 0.634(31) | 0.104 | 0.104 |
| unif=4 | 0.632(27) | 0.617(24) | 0.625(30) | 0.163 | 0.163 |
| unif=6 | 0.614(34) | 0.607(28) | 0.615(37) | 0.254 | 0.254 |
| unif=11 | 0.568(32) | 0.541(30) | 0.532(33) | 0.317 | 0.317 |
| terminal only | 0.785(59) | 0.774(55) | 0.797(59) | 0.498 | 0.499 |
| none (const0) | 1.89(14) | 1.892(143) | 1.955(156) | 0.694 | 0.694 |

The sampled numbers are right. The ranking claim is not, because the
implementer only searched forward-half placements. Splitting the same number
of checks, or fewer, between the forward and inverse halves wins on both
axes. Their sequences, parallel without reset, p = 5e-3, m = 22:

| check placement | rounds incl. terminal | 100(1−F22) | A(22) |
|---|---|---|---|
| forward unif=4 (inside the claimed best range) | 6 | 0.617 | 0.163 |
| forward unif=8 + inverse every 8 | 5 | 0.464 | 0.203 |
| forward unif=6 | 4 | 0.607 | 0.254 |
| forward unif=11 + inverse every 11 | 4 | 0.448 | 0.253 |
| forward unif=11 | 3 | 0.541 | 0.317 |
| one forward check after Clifford 11 + one after inverse Clifford 11 | 3 | 0.500 | 0.317 |

Paired per sequence, unif=11 + inverse every 11 against forward unif=6 (both
4 rounds) gives −26(2)% accepted infidelity for −0.0009 acceptance.
unif=8 + inverse every 8 against forward unif=4 gives −25(2)% infidelity and
+0.040 acceptance. Both hold in 10/10 sequences, for both sequence sets, at
2e-3 and 5e-3, with and without reset.

Required fix: either restrict the claim to forward-half placements, where the
plateau is unif=3 to 11, or extend the search. The best placement I found at
5e-3 gives about 0.45% at 25% acceptance. Inside the plateau the order also
depends on where the checks fall relative to m. unif=11 wins at m = 22
because it checks right after Clifford 22. At m = 20 that check is missing,
and unif=4 is better (0.526% against 0.590%).

### 12 [notes 13]. Executing only the used rounds: confirmed (VN, exact); f is fit-dependent

Exact values, data2 semantics, p = 5e-3, their 10 sequences. "Repo" runs every
round and the policy selects; "executed" runs only the policy's rounds.

| schedule, policy | A(22) repo → executed | F(22) repo → executed | f repo: claimed / exact data | f executed: claimed / exact |
|---|---|---|---|---|
| serial, const=1 | 0.0631 → 0.4614 | 0.024 → 0.931 | 0.9014(67) / 0.8862 | 0.9992 / 0.9993 |
| serial, unif=4 | 3.63e-4 → 0.0384 | 0.645 → 0.918 | 0.9832(4) / 0.9833 | 0.9984 / 0.9985 |
| serial, unif=11 | 2.95e-3 → 0.1709 | 0.194 → 0.925 | 0.9436(42) / 0.9379 | 0.9990 / 0.9991 |
| parallel no reset, const=1 | 0.1085 → 0.5213 | 0.572 → 0.957 | 0.9933(14) / 0.9938 | 0.9995 / 0.9996 |
| parallel no reset, unif=4 | 0.0321 → 0.1698 | 0.928 → 0.960 | 0.9982(1) / 0.9982 | 0.9996 / 0.9996 |

My serial builder gives the same numbers as the lead's builder. Exact and
sampled F(m) agree to about 0.001 at every depth. The two f gaps (0.886
against 0.901, and 0.938 against 0.944) come only from the fit weights:

- my exact means with their sampled per-depth stds give 0.9015 and 0.9439;
- their sampled means with my exact stds give 0.886 and 0.937;
- an unweighted fit gives 0.899 and 0.928.

The repo-protocol curves are far from exponential. For unif=11, F(m) is not
even monotone: 0.39 at m = 10 and 0.41 at m = 14. So f has no unique value
there, and the bootstrap errors (0.0067, 0.0042) understate the dependence on
the fit method. The conclusion stands. Quote F(22) and A(22) for these rows
and mark their f as fit-dependent.

### 13 [notes 14]. Inverse-half checks: refuted for LRB-D, confirmed for decoding (VN)

I compared two placements at depth 22 with the same number of rounds:

- forward only: 10 forward rounds, after Cliffords 2, 4, ..., 20;
- split: 5 forward rounds after Cliffords 4, 8, ..., 20, plus 5 inverse
  rounds after inverse Cliffords 4, 8, ..., 20.

Both also have the terminal round. First order:

| | serial | parallel | parallel, no reset |
|---|---|---|---|
| ΔE1(22), split − forward only, their sequences | −0.90(28)p of 148p | −0.38(24)p of 84p | +0.46(16)p of 64p |
| ΔR1 | 0 | 0 | 0 |
| A2(22) ratio split/forward only, their / new sequences | 0.944(16) / 0.926(17) | 0.862(9) / 0.851(7) | 0.776(10) / 0.765(8) |

Exact accepted infidelity with the final check, split relative to forward
only, paired per sequence (their sequences; new sequences in brackets):

| | p = 2e-3 | p = 5e-3 |
|---|---|---|
| parallel | −15.5(10)% [−16.5(8)%], 10/10 | −17.7(10)% [−18.8(9)%], 10/10 |
| parallel, no reset | −24.2(11)% [−25.2(9)%], 10/10 | −26.7(12)% [−27.7(10)%], 10/10 |

Acceptance changes by 2e-4 or less.

For decoding, the placement is neutral: under 1% of E1, with mixed sign.
Adding inverse checks on top costs, as the notes say, and their per-step
inverse table reproduces exactly. For LRB-D the split placement is a real
gain at no acceptance cost, and it gives the best schedules found under
claim 11.

The notes' own section 2.4 pointed this way. Inverse k = 4 gave 0.143(9)% at
A = 0.217 against forward unif=2 at 0.163(14)% with A = 0.191; the exact
values are 0.145% and 0.176%. Two things in 2.4 need correcting. First,
unif=2 does not have the "same round count": it has 11 forward rounds
against 10 at m = 22, which explains its lower acceptance. Second, the
conclusion "buy no fidelity ... not a limitation worth fixing" must be
reversed for LRB-D, and so must section 3's advice "Do not add inverse-half
checks".

## Required fixes

1. Claim 3: write "48p/15 for zx and parallel with the repo's CNOT order";
   with the X-check CNOTs first it is 54p/15.
2. Notes claim 7, Summary item 3, section 2.2 and section 3 item 3: rename the
   "correction penalty" to something like "every-round check penalty
   (decoded)". Restate its definition and absolute size wherever a fraction is
   quoted, and say that the MLD2 fractions use the const0 MLD5 reference.
3. Notes claim 10: say "at all tested p", with the scope given above.
4. Notes claim 11 and section 2.6: label 600-1700× as first order with about
   ±25% sequence uncertainty ("about 500 to 1700×"). Optionally add the
   finite-p paired gain of 0.011p.
5. Notes claim 12, the Summary's LRB-D recipe and section 3 item 2: restrict
   "lowest" to forward-half placements ("every 3 to 11 Cliffords"), or add
   forward+inverse placements. The best found at 5e-3 is about 0.45% at
   A ≈ 0.25. Note that the order inside the plateau depends on alignment
   with m.
6. Notes claim 13: mark f for the repo-protocol const and unif rows as
   dependent on the fit weights (0.886 with exact data), and lead with F(22)
   and A(22).
7. Notes claim 14, section 2.4 and section 3 item 2: reverse the inverse-half
   conclusion for LRB-D. At equal round count, the split placement lowers the
   accepted infidelity by 15-28% at unchanged acceptance. Also fix "same
   round count" in 2.4. Keep the decoding statement.

## Reproduce

```
module load python/3.11
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
python -m pytest -q tests/test_qubit512_variants.py tests/test_lrb_qubit_model.py
V=studies/correction-decoders/qubit/scripts/verify_variants.py
python $V selftest                                    # ~20 s, login node
# on the shared allocation (srun --jobid=<id> --overlap --cpu-bind=none -c 256,
# then taskset to 16 free CPUs):
python $V census --workers 16                          # ~2 min
python $V exact --workers 16 --probs 5e-3,2e-3         # ~1 min
python $V paired --workers 16 --shots 300000 --probs 2e-3,5e-3,1e-2
python $V sample --workers 16 --shots 100000 --probs 2e-3,5e-3
python $V compare                                      # last; imports variants.py
```

Files in `results/H-verify/`: `selftest.json`, `census_rows.csv`,
`census_summary.json`, `monotone_chains.json`, `equal_count_inverse.json`,
`exact_rows.csv`, `exact_summary.json`, `exact_inverse_rows.json`,
`paired_perfect_checks_rows.json`, `paired_perfect_checks_summary.json`,
`sample_rows.json`, `sample_summary.json`, `compare.json`, `logs/`.
