# A5: is distance 3 needed before correction beats detection in qutrit LRB, and which code next?

Agent A5, 2026-10-28. Scripts (all in `scripts/`): `A5_code_capacity_compare.py`
(codes, logical gates, fault-tolerance checks, code capacity, phenomenological
noise), `A5_circuit_first_order.py` (exact first-order coefficients of the
baseline LRB circuits), `A5_catalog_check.py` (side check of the Clifford
catalog). Outputs: `scripts/A5_out/` (`A5_results.json`, `A5_run_log.txt`,
`A5_eps_vs_q.png`, `A5_circuit_first_order.json`).

Labels: **[VA]** verified analytically, **[VN]** verified numerically,
**[LIT]** from literature fetched in this task (Section 8), **[CONJ]**
conjecture. **[data]** marks numbers read from the baseline run folder.

## Answer in brief

1. **With [[5,1,2]]_3, correction cannot beat an unencoded qutrit, so a
   corrected LRB on this code cannot show any benefit of encoding.**
   - Code capacity: the maximum-likelihood decoder (MLD) gives
     eps = q + (63/16) q^2 + O(q^3), and eps > q for every q in (1e-3, 0.45)
     [VN].
   - Phenomenological noise with q_m = q: eps/q -> 1.00 for the memoryless
     lookup decoder (exact). The optimal space-time decoder gives eps = q
     within Monte Carlo error [VN].
   - At circuit level, on the actual LRB circuits at p = 1e-3, to first order
     in p: optimal correction gives 1 - f = 16.6 p per depth step (twirled
     eps = 14.7 p). The physical-qutrit RB comparison gives 5.6 p
     (eps = 5.0 p). Correction is about 3x worse than the bare qutrit [VN].
2. **Distance 3 is required for correction to reach O(p^2) per round and to
   beat the physical qutrit.** Under the SI1000 model this also needs
   simultaneous ancilla readout. The repository measures one ancilla per
   layer, so idle noise grows with the number of ancillas. With that
   schedule, d = 3 codes break even only for p below about 1e-4 to 6e-4.
   With simultaneous readout the estimate is p below about 1.3e-3 to 3.4e-3
   [CONJ, Section 3.3].
3. **Distance 3 is not enough to beat detection on postselected fidelity.**
   [[5,1,2]]_3 detection reaches about 27 p^2 accepted logical error per depth
   step [data]. A corrected d = 3 code is estimated at A kappa^2 p^2, with
   A kappa^2 of at least about 90 even with optimistic bounds, and about 460
   to 1900 for realistic values [CONJ]. What correction buys is
   acceptance 1 instead of about exp(-112 p m) [VN], an LRB decay that is
   measurable at any p, and an error rate that applies without
   postselection.
4. **Best next code: [[13,1,3]]_3, the unrotated qutrit surface code folded
   along its diagonal.** It is the distance-3 member of the family that
   contains the repository's [[5,1,2]]_3 (Moussa's "square" family [LIT]).
   - The repository's sign rule extends to fold-transversal H_L and P_L,
     which generate the full logical Clifford group [VN].
   - One-ancilla-per-check extraction tolerates any single fault. Every
     coupling order is safe check by check, a jointly safe choice exists, and
     faults of the fold-pair gates are correctable [VN].
   - It has the best code-capacity coefficient of the candidates, 89/16 [VN].

   The runner-up is the [[7,1,3]]_3 colour code. Its H_L and P_L are
   transversal single-qutrit layers [VN], but one-ancilla extraction is not
   fault tolerant, so it needs flag qutrits [VN].

   Not recommended for full-Clifford LRB:
   - [[9,1,3]]_3 rotated surface code: no fold-type P_L in a fixed layout [VN].
   - [[5,1,3]]_3: no transversal P_L, since its automorphisms give a logical
     group of order 8 [VN]; it also needs flags [VN].

## 0. Definitions used throughout

- **Code-capacity noise.** Each data qutrit independently suffers, with
  probability q per round, one of the 8 non-identity Weyl operators X^a Z^b,
  chosen uniformly. Syndromes are perfect.
- **Phenomenological noise.** The same data noise, then each check outcome is
  shifted by +1 or -1 with probability q_m/2 each. The final round is
  perfect.
- **eps (per-round logical error).** The total probability of a non-identity
  logical Pauli per round.
  - For a twirled logical channel, 1 - f = (9/8) eps and r = 1 - F = (3/4) eps.
  - The unencoded comparator has eps = q.
  - The circuit LRB comparator is the physical RB fit: r = 3.757e-3 at
    p_RB = 5p = 5e-3, which gives eps = 5.0 p per depth step [data, fit table].
- **Per-round decay in the exact phenomenological computations.** Each
  decoder's process is a transfer matrix. For each non-trivial logical
  character chi of Z_3^2 the leading eigenvalue mu_chi is computed;
  f = mean over chi of mu_chi / mu_1, and mu_1 is the acceptance per round.
  The transfer matrices were checked against direct Monte Carlo
  character by character [VN; `--validate` flag, for [[5,1,2]]_3 and
  [[9,1,3]]_3 at q = q_m = 0.05: exact and Monte Carlo per-round eigenvalues
  agree to within 0.002, e.g. 0.91977 vs 0.92008 and 0.84545 vs 0.84706].
- **Decoders.**
  - *MLD:* exact per-q maximum-likelihood decoding of a single round.
  - *Memoryless lookup:* a fixed minimum-weight table, applied every round
    with no memory. Ties go to the lowest class index, and the table is
    symmetrised so that lstar(-s) = -lstar(s).
  - *Space-time decoder:* an exact Bayesian forward filter over the state
    (syndrome, logical class) in Z_3^(n+1). It is "joint" when it is
    correlation-aware, and "sep" when the X and Z parts of a CSS code are
    filtered separately.
  - *Detection:* reject if any raw check outcome is non-zero.
  - *Hybrid ("strict"):* reject a syndrome if its minimum-weight explanations
    span more than one logical class, otherwise correct it. This is the same
    rule as `hyb_strict` in A1's script. For [[5,1,2]]_3 it rejects 56 of the
    80 non-zero syndromes and corrects the 24 unambiguous weight-1 syndromes.
- **Three meanings of "correction beats detection".**
  1. Lower per-round error than detection's accepted error.
  2. Lower per-round error than the physical qutrit at full acceptance. This
     is the notion that makes a corrected LRB meaningful.
  3. Lower effective error when rejected shots are counted as failures.

  For [[5,1,2]]_3: meaning 1 never holds (q^2/8 vs q) [VN]; meaning 2 fails
  [VN]; meaning 3 already holds at d = 2 (14.7 p vs 112 p per step at circuit
  level) [VN].

## 1. Candidate codes

All constructions below are verified by `A5_code_capacity_compare.py`,
Sections 2, 3 and 3b [VN]. The checks are:
- pairwise commutation of the generators and rank over GF(3);
- k = 1, with logicals that commute with the stabilizer and are independent;
- the distance, by brute force over all Paulis of weight <= 3;
- the logical action of each gate, as a 2x2 symplectic matrix, and the order
  of the group the gates generate.

Single-qutrit gate conventions are sdim's frame rules:
- H: (x,z) -> (-z,x);
- P: z += x;
- CNOT(c,t): x_t += x_c, z_c -= z_t;
- CZ(a,b): z_b += x_a, z_a += x_b.

**[[5,1,2]]_3 (repository).** The repository's H_L and P_L act on the logical
operators as [[0,2],[1,0]] (H-type) and [[1,0],[1,1]] (P-type). Together they
generate all of SL(2,3) (order 24) [VN].

**[[5,1,3]]_3 (cyclic, "perfect" for qubits).** The generators are four
cyclic shifts of X Z Z^-1 X^-1 I:
- g0 = X0 Z1 Z2^-1 X3^-1
- g1 = X1 Z2 Z3^-1 X4^-1
- g2 = X2 Z3 Z4^-1 X0^-1
- g3 = X3 Z4 Z0^-1 X1^-1

The logicals are X_L = X^(x5) and Z_L = Z^(x5), with <X_L, Z_L> = 1 mod 3.
The code has distance 3 and 80 weight-3 logical operators [VN]. This is the
code of Chau (1997) and Rains (1997) [LIT]. For qutrits it is not perfect:
the 41 errors of weight <= 1 use 41 of the 81 syndromes [VA].

**[[7,1,3]]_3 colour code (Steane-like).**
- Faces: {0,2,4,6}, {1,2,5,6}, {3,4,5,6}.
- Star bipartition: unstarred {0,1,3,6} (exponent +1), starred {2,4,5}
  (exponent -1).
- X checks: X on each face. Z checks: Z^eps on each face, with eps = +1 on
  unstarred and -1 on starred qutrits.
- Logicals: X_L = X^(x7), Z_L = prod_i Z_i^eps_i.
- Distance 3, with 48 weight-3 logicals [VN].

The weight-4 faces force signs: with all-+1 exponents, X_f and Z_f overlap on
4 sites and 4 = 1 mod 3. The bipartition makes every face sum and every
two-site face overlap vanish mod 3 [VA].
- H_L is F on unstarred and F^dagger on starred qutrits; P_L is P on
  unstarred and P^-1 on starred qutrits. Both are H-type and P-type, and they
  generate SL(2,3) [VN].
- Uniform F^(x7) or P^(x7) does not preserve the code [VN].

This is the qudit colour-code rule "U on unstarred, U* on starred" of Watson
et al. (2015) [LIT].

**[[9,1,3]]_3 rotated surface code.** Data q(r,c) = 3r + c.
- X checks, all exponents +1: plaquettes {0,1,3,4} and {4,5,7,8}, plus the
  boundaries {1,2} (top) and {6,7} (bottom).
- Z checks: Z1 Z2^-1 Z4^-1 Z5, Z3 Z4^-1 Z6^-1 Z7, Z0 Z3^-1 (left) and
  Z5 Z8^-1 (right).
- Logicals: X_L = X0 X3 X6 and Z_L = Z0 Z1^-1 Z2.
- Distance 3, with 64 weight-3 logicals [VN].

Logical gate searches (in the script):
- A transversal F^s with s in {F, F^dagger}^9 composed with each of the 8
  symmetries of the square preserves the code only for the 90-degree and
  270-degree rotations (2 sign patterns each). These give an H-type logical
  [VN].
- Plain transversal F maps the code onto the code with X and Z checks
  exchanged at the same plaquettes [VA]. On fixed hardware, H_L is therefore
  transversal F followed by a swap of check roles (or a patch rotation). The
  hook-safe coupling order must be swapped as well, since the logical
  directions exchange [VA].
- No fold-type phase gate exists. The search covered P^t on the fixed sites
  and CZ^u on the swapped pairs of each of the four reflections, with all
  3^6 exponent patterns per reflection. No pattern preserves the code [VN].

This matches Moussa's statement that the rotated ("diamond") code cannot be
folded because of its chiral edges [LIT]. For qubits, Chen et al. (2024)
obtain S on the rotated code by embedding a fold-transversal step inside a
syndrome round and rotating patches by atom movement [LIT]. That route needs
reconfigurable connectivity.

**[[13,1,3]]_3 unrotated surface code folded along the diagonal.**
- Layout: a 5x5 grid, with data at (i,j) for i+j even (13 sites).
- X checks at (odd, even): X on the grid neighbours, exponents +1.
- Z checks at (even, odd): Z^((-1)^i) on the neighbours, where i is the
  neighbour's row.
- Logicals: X_L = X(0,0) X(0,2)^-1 X(0,4) and Z_L = Z(0,0) Z(2,0)^-1 Z(4,0).
- Distance 3, with only 12 weight-3 logicals [VN].
- The fold tau: (i,j) <-> (j,i) has 5 diagonal sites and 4 mirror pairs.

Fold-transversal gates:
- H_L: SWAP each mirror pair, then F on even rows and F^dagger on odd rows.
- P_L: P^((-1)^i) on the diagonal sites (i,i), and CZ^((-1)^i) on each mirror
  pair ((i,j),(j,i)).

Both are H-type and P-type, and they generate SL(2,3) [VN]. The same row-parity
rule applied to the 3x3 grid reproduces exactly the repository's H_L and P_L
for [[5,1,2]]_3 [VA]:
- the qutrits are 0 = (0,0), 1 = (1,1), 2 = (0,2), 3 = (2,0), 4 = (2,2);
- the repository's X-check signs differ only by the local relabelling
  C = F^2 on the pair {2,3}, which commutes with F, P, CZ and SWAP.

So [[13,1,3]]_3 is the d = 3 member of the same construction. This is the
folded "square" family of Moussa (2016), defined for qudits of any dimension
with 2D^2 - 2D + 1 data qudits (5 at D = 2, 13 at D = 3) [LIT].

### Fault tolerance of plain one-ancilla-per-check extraction [VN]

The test is whether the following set of single-fault data errors contains
two errors with the same syndrome but different logical class:
- every single-qutrit Pauli;
- every Pauli on a fold pair, for the folded codes (faults of the SWAP and CZ
  gates in H_L and P_L);
- for each check measured with coupling order q_1..q_w, every error
  P_{q_k} (x) (check restricted to q_{k+1}..q_w)^a. This is a two-qutrit
  fault on (ancilla, q_k) followed by propagation of the ancilla error.

| code | coupling orders that are safe for each check | jointly safe order | fold-pair faults |
|---|---|---|---|
| [[5,1,3]]_3 | 0 of 24 for every check | no, so flags are needed | n/a |
| [[7,1,3]]_3 | 0 of 24 for every check | no, so flags are needed | n/a |
| [[9,1,3]]_3 | 16 of 24 (weight 4), 2 of 2 (weight 2) | yes, e.g. X plaquette {0,1,3,4} in order 0,1,3,4 | n/a |
| [[13,1,3]]_3 | 24 of 24 (weight 4), 6 of 6 (weight 3) | yes (the first combination tested is safe; not all combinations were tested) | correctable; none is an undetectable logical |
| [[5,1,2]]_3 (d = 2) | not applicable, single errors already collide (8 syndromes) | | 24 colliding syndromes; no pair fault is an undetectable logical, so detection catches them |

For qubits, Chao and Reichardt (2018) use flags to make [[5,1,3]] and
[[7,1,3]] fault tolerant [LIT]. For the 17-qubit layout, Tomita and Svore
(2014) give a CNOT order that keeps a single ancilla fault from causing a
logical error [LIT]. The qutrit results above are this task's own checks.

### Summary table (part 1)

| code | data + ancillas (1 per check) | check weights | 2-qutrit gates per round | H_L | P_L | full Clifford fault tolerant | Tanner graph |
|---|---|---|---|---|---|---|---|
| [[5,1,2]]_3 (repo) | 5 + 4 | 4 x 3 | 12 | fold-transversal: 1 SWAP + 5 single | fold-transversal: 1 CZ + 3 single | full Clifford group yes; d = 2, so faults are detected, not corrected | planar, also with the fold pair |
| [[5,1,3]]_3 | 5 + 4 (+ flags) | 4 x 4 (mixed X/Z) | 16 (controlled-Weyl) | transversal only with a reflection relabelling | none; logical automorphism group is order 8 | no | non-planar; each ancilla touches 4 of 5 data |
| [[7,1,3]]_3 colour | 7 + 6 (or 3, one per face) (+ flags) | 6 x 4 | 24 | 7 single (F / F^dagger) | 7 single (P / P^-1) | yes, if flags are added | non-planar with 6 ancillas; planar with 3 face ancillas [VN] |
| [[9,1,3]]_3 rotated | 9 + 8 | 4 x 4 + 4 x 2 | 24 | 9 single + 90-degree relabel or check-role swap | none in the fold class | no (P_L missing) | planar square grid |
| [[13,1,3]]_3 folded | 13 + 12 | 4 x 4 + 8 x 3 | 40 | 4 SWAP + 13 single | 4 CZ + 5 single | yes | planar; non-planar once the 4 fold pairs are added (needs a bilayer layout or long-range couplers) |

[[5,1,3]]_3 automorphism search [VN]:
- All 80 local-Clifford x permutation automorphisms were found by a
  meet-in-the-middle search over 24^5 x 120 candidates.
- Their logical actions are the identity, C = H^2, two H-type elements and
  four order-4 elements. These form a group of order 8 with no element of
  order 3, so there is no P-type element.
- Without permutations the group has order 4.

## 2. Code-capacity and phenomenological comparison (part 2)

Script: `scripts/A5_code_capacity_compare.py` [VN]. Figure:
`scripts/A5_out/A5_eps_vs_q.png`.

### 2.1 Code capacity (exact)

The coset distribution P(syndrome, logical) is computed exactly as one inverse
DFT over Z_3^(n+1), using E[w^(k.c)] = (1 - 9q/8)^(number of sites touched by
k). Leading coefficients are exact integer counts over errors of weight 1
and 2.

| code / decoder | eps leading order | eps at q = 1e-3 | q = 1e-2 | q = 5e-2 | break-even q* (eps = q) |
|---|---|---|---|---|---|
| [[5,1,2]]_3 MLD | q + (63/16) q^2 | 1.004e-3 | 1.039e-2 | 5.88e-2 | none, eps > q up to 0.45 |
| [[5,1,2]]_3 detection, accepted | q^2/8 (acceptance 1 - 5q) | 1.25e-7 (acc. 0.9950) | 1.29e-5 (0.9510) | 3.60e-4 (0.7741) | none, eps < q |
| [[5,1,2]]_3 hybrid, accepted | about 2.63 q^2 (acceptance 1 - 2q) | 2.63e-6 (acc. 0.9980) | 2.63e-4 (0.9801) | 6.59e-3 (0.9022) | 0.380 |
| [[5,1,3]]_3 MLD | (35/4) q^2 | 8.73e-6 | 8.58e-4 | 1.98e-2 | 0.157 |
| [[7,1,3]]_3 MLD | (51/4) q^2 | 1.27e-5 | 1.24e-3 | 2.78e-2 | 0.105 |
| [[9,1,3]]_3 MLD (3^8 syndromes) | (93/8) q^2 | 1.16e-5 | 1.14e-3 | 2.61e-2 | 0.110 |
| [[13,1,3]]_3 MLD | (89/16) q^2 | 5.58e-6 | 5.73e-4 | 1.56e-2 | 0.152 |

Why the d = 2 MLD only breaks even at first order [VA]:
- Pure X^a errors on qutrits {0,3} or {2,4}, and pure Z^b errors on {0,2} or
  {3,4}, come in 8 pairs. The two errors of a pair have the same syndrome,
  equal probability q/8, and differ by a logical.
- MLD therefore fails on 8 x q/8 = q. That is 1/5 of the total error mass 5q.
- The q^2 coefficient, +63/16, is positive [VN, exact enumeration of 9^5
  errors].

### 2.2 Phenomenological noise, q_m = q

All values below use q_m = q.

| code | memoryless lookup: eps/q^2 as q -> 0 (q_m = q, q_m = q/2, q_m = 0) | lookup q* | space-time decoder: eps per round at q = 0.01 and 0.02 | space-time q* |
|---|---|---|---|---|
| [[5,1,2]]_3 | eps/q = 1.006, 1.003, 1.001 | none | joint: 9.96e-3, 2.16e-2 (eps = q within error) | none |
| [[5,1,3]]_3 | 46.6, 25.2, 8.75 | 0.027 | joint: 1.07e-3 (10.7 q^2), 5.48e-3 (13.7 q^2) | about 0.082 |
| [[7,1,3]]_3 | 83.2, 43.5, 12.7 | 0.0145 | sep: 2.70e-3 (27 q^2), 1.02e-2 (26 q^2); joint at 0.02: 5.33e-3 (13.3 q^2) | about 0.041 (sep) |
| [[9,1,3]]_3 | 66.0, 35.3, 11.6 | 0.0176 | sep: 3.15e-3 (32 q^2), 1.14e-2 (28 q^2); joint at 0.02: 5.88e-3 (14.7 q^2) | about 0.039 (sep) |
| [[13,1,3]]_3 | per-q MLD table: 5.95e-3 at q = 0.01 | 0.0172 | sep: 2.13e-3 (21 q^2), 1.02e-2 (26 q^2) | about 0.042 (sep) |

How the space-time numbers were obtained:
- The per-round eps comes from P_L after 6 rounds (5 noisy and 1 perfect),
  divided by the exact one-round value. The 1-sigma bands are in the log.
- The joint calibration at q = q_m = 0.02 used 4000 shots for [[7,1,3]]_3 and
  1500 for [[9,1,3]]_3 (`A5_run_log.txt`, Section 5c):
  - [[7,1,3]]_3: 5.33e-3, 1-sigma band [4.76e-3, 5.89e-3], or 13.3 q^2;
  - [[9,1,3]]_3: 5.88e-3, band [4.93e-3, 6.84e-3], or 14.7 q^2.

  An independent development run with another seed gave 12.8 and 16.4 q^2.

[[5,1,2]]_3 detection and hybrid under phenomenological noise, at
q = q_m = 0.01 (exact transfer matrices) [VN]:
- detection: acceptance per round 0.9135 (about 1 - 9q) and accepted eps
  1.30e-5 (0.13 q^2);
- hybrid: acceptance per round 0.9419 (about 1 - 6q) and accepted eps
  2.84e-4 (2.8 q^2). Its accepted-error break-even is 0.151.

Readings:
- For d = 2, measurement errors do not enter at leading order. A single
  measurement error leads the memoryless decoder to apply a weight-1 Pauli,
  and the symmetric table undoes it in the next round [VA].
- For d = 3, a memoryless lookup decoder loses a factor of 4 to 7 in the q^2
  coefficient at q_m = q. A joint space-time decoder brings it back to about
  1.0 to 1.6 times the code-capacity coefficient [VN].
- Separate X/Z filtering of the CSS codes costs about a factor of 2 [VN].
- With joint space-time decoders the three d = 3 codes tested (all except
  [[13,1,3]]_3) are within a factor of about 1.5 of each other at q_m = q
  (about 11 to 16 q^2). Under phenomenological noise the code choice is
  therefore not decided by its error-correction coefficient.

## 3. Circuit-level picture (part 3)

### 3.1 Exact first-order coefficients of the baseline [[5,1,2]]_3 LRB circuits [VN]

`A5_circuit_first_order.py` reads the stored circuits (p = 1e-3, Clifford
sequences 0 to 9, depths m = 0, 2, 4, 6, 10) from the read-only run folder. It
compiles each with the repository's `CompactDetectorErrorModel`. It then
enumerates every single fault (location x non-zero Weyl vector) and groups
the faults by their full detector signature over all rounds. The decoder
"optimal correction" picks, for each signature, the most probable observable
shift. This is the p -> 0 limit of MLD on the whole detector history.

| units of p | slope per depth step | intercept |
|---|---|---|
| detection (unif 1) rejection | 112.1 | 62.0 |
| detection, undetected logical | 0.000 | 4.95 |
| optimal correction, observable-flip probability | 11.06 | 14.25 |
| hybrid rejection (ambiguous signatures) | 39.9 | 45.7 |
| terminal-only (const 1) rejection | 71.7 | 65.8 |
| terminal-only correction | 13.5 | 14.5 |
| total fault mass | 128.4 | 99.3 |

The fidelity metric |sum_k w^k P(k)| equals 1 - 1.5 P(shift != 0) at first
order. Optimal correction therefore gives 1 - f = 16.6 p per depth step,
which is twirled eps = 14.7 p. The physical RB comparator is 1 - f = 5.63 p
(eps = 5.0 p) [data]. The corrected [[5,1,2]]_3 logical qutrit is about 3x
worse than the bare qutrit.

Validation against the baseline [data]:
- At m = 0 the predicted fidelity is 1 - 1.5 x 4.95e-3 = 0.99258, against a
  measured 0.99250.
- Rejection at m = 0 is 6.3% at first order, against 6.1% measured.
- Extrapolating the fit to m = 22 gives 92.0% rejection for unif 1 (measured
  91.9%) and 80.7% for const 1 (measured 79.4%).

Where the 11.06 p per step of correction failures comes from:

| fault source | failure per step | total mass in the source |
|---|---|---|
| data idle in the measurement and reset layers | 7.65 p | 50 p, i.e. 10 p per data qutrit, all detected |
| check CNOT faults | 1.52 p | 12 p |
| two-qutrit logical gates (SWAP and CZ), forward + inverse | 1.38 p | 6.5 p |
| single-qutrit logical gates | 0.51 p | 3.4 p |
| measurement flips | 0 | 20 p |
| reset errors | 0 | 8 p |
| pre-measurement and idle ancilla depolarizing | 0 | 28 p, of which 12 p detected |
| ancilla H gates | 0 | 0.4 p |

The multi-round decoder removes all ancilla-side faults (measurement flips,
reset errors, ancilla depolarizing) completely at first order [VN].

The hybrid cuts the rejection exponent from 112 p to 40 p per step and has no
first-order accepted error per step [VN]. It is a useful middle ground for
d = 2, but it is still a postselection scheme.

### 3.2 The code-capacity picture predicts the d = 2 circuit coefficient

Define q_eff as the effective depolarizing per data qutrit per depth step. It
has three parts:
- idle noise, 4 x 2p + 2p = 10p;
- CNOT faults, 0.9p per CNOT (72 of the 80 two-qutrit faults touch the data
  qutrit) x 12/5 CNOTs per qutrit, which is 2.16p;
- logical gates, which carry 7.27p of fault mass per Clifford over the five
  qutrits. That is 1.45p per qutrit per Clifford, or 2.9p for the two
  Cliffords of a step. The per-Clifford mass uses the catalog averages of
  1.65 H, 1.46 P, 1.00 X and 1.00 Z per Clifford [VN].

The total is q_eff = 15.1p. Code capacity says eps = q_eff, which gives
15.1p against the measured 14.7p [VA + VN]. The [[5,1,2]]_3 check round puts
about 10p of idle noise plus about 5p of gate noise on each data qutrit per
step. With correction, all of it becomes logical error with coefficient 1.

### 3.3 When does distance 3 (O(p^2)) win? [CONJ]

This section assumes a correction-based LRB round of one logical Clifford
followed by one check round, with checks after inverse Cliffords too (see
3.4). The estimate is

eps_round ~ A x (kappa_1 p)^2, where:
- **kappa_1** is the SI1000 effective depolarizing per data qutrit per round,
  built from the same terms as 3.2:
  - 2p per measurement layer;
  - 2p for the reset layer;
  - 0.9p x (CNOTs per data qutrit);
  - the logical-gate mass per data qutrit per Clifford: 1.45 for
    [[5,1,2]]_3, 0.40 for [[7,1,3]]_3, 0.38 for [[9,1,3]]_3 (this assumes a
    hypothetical transversal-cost P), and 1.99 for [[13,1,3]]_3 (4 SWAP or
    4 CZ per H_L or P_L).
- **A** is the phenomenological space-time coefficient at q_m = q from
  Section 2.2: about 13 for [[7,1,3]]_3 and 16 for [[9,1,3]]_3 (joint
  decoders; the measured 14.7 to 16.4 is rounded up for [[9,1,3]]_3). For [[13,1,3]]_3 the range is 10 to 25: 25 is the separate X/Z
  value, and about 10 is expected from the joint/separate ratio seen at code
  capacity.

The SI1000 measurement error is about q_m = 5p + 0.75p + 2p + 0.68p per
coupled CNOT. With sequential readout, waiting ancillas add about
0.75p (n_a - 1). This puts q_m within a factor of about 1.5 of kappa_1 p, so
the q_m = q coefficients apply [VA, approximate].

| code | kappa_1, sequential readout (repo style) | kappa_1, simultaneous readout | break-even vs physical (2.5p per Clifford): p* = 2.5/(A kappa_1^2), sequential / simultaneous | beats d = 2 correction for p below, sequential / simultaneous |
|---|---|---|---|---|
| [[5,1,2]]_3 (d = 2, eps = kappa_1 p) | 13.6 | 7.6 | never (13.6p or 7.6p vs 2.5p) | n/a |
| [[7,1,3]]_3 (flag overhead not included) | 17.5 | 7.5 | 6.3e-4 / 3.4e-3 | 3.4e-3 / 1.0e-2 |
| [[9,1,3]]_3 (no fault-tolerant P_L) | 20.8 | 6.8 | 3.6e-4 / 3.4e-3 | 2.0e-3 / 1.0e-2 |
| [[13,1,3]]_3 | 30.8 | 8.8 | 1.1e-4 to 2.6e-4 / 1.3e-3 to 3.3e-3 | 5.8e-4 to 1.4e-3 / 4.0e-3 to 9.9e-3 |

The conditions for d = 3 to win are therefore the following.

1. **The decay must be measured at full acceptance.** Against the
   postselected accepted error of d = 2 detection, distance 3 never wins
   under SI1000. Detection reaches about 27 p^2 per depth step [data: fit
   table, r = 2.03e-5 at p = 1e-3, so eps = (4/3) r = 27.1 p^2], about
   13.5 p^2 per Clifford. Correction gives A kappa_1^2 p^2. Even with the
   most optimistic bounds, A kappa_1^2 >= 5.6 x 4^2 = 90: kappa_1 >= 4 from
   one measurement layer plus reset alone, and A >= 5.6, the smallest
   code-capacity coefficient. That is already 6.6 times detection. For the
   realistic values in the table, A kappa_1^2 is about 460 to 1900, or 30 to
   140 times detection [CONJ].
2. **p must lie below the break-even line**, about 1e-3 to 3e-3 with
   simultaneous ancilla readout. It falls to about 1e-4 to 6e-4 with the
   repository's one-ancilla-per-layer readout, because idle noise of 2p per
   measurement layer grows linearly with the number of ancillas: 8 for
   [[9,1,3]]_3 and 12 for [[13,1,3]]_3 [CONJ]. At the baseline's
   p = 1e-3, [[13,1,3]]_3 with simultaneous readout gives an estimated
   eps of about 0.8e-3 to 1.9e-3 per Clifford, against 2.5e-3 for the
   physical qutrit and about 7e-3 for d = 2 correction. With sequential
   readout it gives about 9e-3 to 2.4e-2.
3. **Single faults must stay correctable.** This needs a hook-safe coupling
   order (available for both surface codes) or flags (needed for the colour
   code and [[5,1,3]]_3), together with logical gates whose single faults are
   correctable (verified for the [[13,1,3]]_3 fold pairs) [VN]. If any one
   fault is uncorrectable, eps returns to O(p).
4. **A space-time decoder is needed.** At q_m = q a memoryless lookup costs a
   factor of 4 to 7 in A [VN].
5. **The circuit coefficient must stay close to the phenomenological one.**
   Correlated two-qutrit faults and hooks raise the second-order coefficient
   above the phenomenological mapping. For d = 2 the first-order mapping held
   to 3%; for d = 3 a factor of 2 to 3 is plausible and would lower every
   break-even p by that factor [CONJ]. The circuit-level number needs a DEM
   simulation of the d = 3 circuit.

### 3.4 Structural requirements of the LRB circuit for correction

- **Checks after inverse Cliffords.** In the current generator the m inverse
  Cliffords have no checks between them; only the terminal round m sees them
  [code, brief].
  - For d = 2 detection this is harmless at first order.
  - For d = 3 correction the inverse block behaves as one unprotected round
    carrying m x lambda p of fault mass. lambda is the logical-gate fault mass
    per Clifford: 25.9 for [[13,1,3]]_3 and 2.8 for [[7,1,3]]_3 [VA].
  - Two faults in that block are uncorrectable with probability of order
    (m lambda p)^2 / 2. For example, 0.13 x (malignant fraction) for
    [[13,1,3]]_3 at m = 20 and p = 1e-3. This grows quadratically in m and
    breaks the single-exponential LRB model [VA].
  - A corrected LRB needs a check round after every Clifford, or the inverse
    compiled into one logical Clifford.
- **Sampling cost of detection vs correction [VA, order of magnitude].**
  - With detection, the useful depth is m of about 1/(112 p), about 9 at
    p = 1e-3. Over that depth the accepted decay m x 27 p^2 is about 2.4e-4.
    That is far below the depth-independent offset of about 7.5 p from the
    final round.
  - Resolving it to 10% needs about sigma^2 / (0.1 x 2.4e-4)^2, which is
    about 2e7 accepted shots per depth with sigma^2 of about 1e-2. The cost
    grows as 1/p.
  - With d = 3 correction the decay over m of about 1/eps rounds is O(1) at
    acceptance 1. About 10^2 to 10^3 shots per depth then suffice, at any p.

## 4. Decision

- **Is distance 3 required?** Yes, if "correction beats detection" means
  either of the following:
  - a corrected (unpostselected) LRB whose logical error per round is below
    the physical qutrit's;
  - a corrected LRB that scales as O(p^2).

  [[5,1,2]]_3 with optimal correction is never better than the bare qutrit:
  q + (63/16) q^2 at code capacity, q at phenomenological level, and 3x worse
  at circuit level [VN].
- **Is distance 3 sufficient?** Not for beating detection's postselected
  fidelity per round. That would need A kappa^2 < 27, which no d = 3 code
  reaches under SI1000-like noise [CONJ]. It would take d >= 5, or idle and
  gate noise far below SI1000.
- **Best next step: [[13,1,3]]_3 folded surface code with simultaneous ancilla
  readout.** It is the direct continuation of the repository's code
  (same fold-transversal H_L/P_L rule, same CSS check structure, same X-basis
  logical readout), and it is fault tolerant with one ancilla per check.
  Every coupling order is safe check by check. It has the best code-capacity coefficient (89/16) and the
  highest code-capacity break-even after [[5,1,3]]_3 (0.152 vs 0.157). Its
  costs are 25 qutrits, 40 two-qutrit gates per check round, and 8
  two-qutrit gates in each H_L or P_L (4 SWAP or 4 CZ). It also needs either
  long-range couplers for the 4 mirror pairs or a folded bilayer layout.
- **Second candidate: [[7,1,3]]_3 colour code with flags.** It has the cheapest
  logical gates (7 single-qutrit gates per H_L or P_L), and its Tanner graph
  is planar with 3 face ancillas. It needs flag qutrits and flag-aware
  decoding; flag gadgets for qutrits were not found in the literature search
  (Section 8).
- **Not recommended for full-Clifford LRB:**
  - [[9,1,3]]_3, which has no fault-tolerant P_L on fixed couplers;
  - [[5,1,3]]_3, which has no transversal P_L and needs flags.

## 5. Side finding: the Clifford catalog is not uniform [VN]

`A5_catalog_check.py` checks the 216 strings of
`single_qutrit_cliffords_exhaustive_strings`:
- Only 207 strings are distinct, and they give 206 distinct Clifford elements
  modulo phase out of 216.
- 10 elements are missing, for example the Paulis ZXX and ZZX, PXPH and HPXP.
  9 elements appear 2 or 3 times (e.g. XXZZ, ZZXX, XXZZ).
- The 24 symplectic parts appear 9 times each for 19 of them, and 5, 5, 10, 12
  and 13 times for the other 5.

Pauli noise is twirled only through the symplectic part. The sampled twirl is
therefore not uniform over SL(2,3). The logical H_L and P_L have the same
symplectic action as the physical gates, so the LRB twirl inherits this
non-uniformity. This could bias the fitted decay or make it non-exponential.
It is relevant to the study's non-Markovianity question (outside A5's task;
not quantified here).

## 6. Claims

| id | claim | label | evidence |
|---|---|---|---|
| A5-distance3-C1 | [[5,1,2]]_3 MLD at code capacity: eps = q + (63/16) q^2 + O(q^3), above q for all q in (1e-3, 0.45); no break-even | VN | exact enumeration of 9^5 errors; exact DFT distribution; break-even search (`A5_code_capacity_compare.py` Section 4) |
| A5-distance3-C2 | [[5,1,2]]_3 detection: acceptance 1 - 5q, accepted eps = q^2/8; strict hybrid: acceptance 1 - 2q, accepted eps of about 2.63 q^2, break-even 0.38 | VN | Section 4 table (q = 1e-3, 1e-2, 5e-2), leading-order counts |
| A5-distance3-C3 | Phenomenological q_m = q, [[5,1,2]]_3: correction eps/q -> 1.00 (memoryless, exact); space-time MLD consistent with eps = q; detection acceptance about 1 - 9q, accepted eps 0.13 q^2; hybrid acceptance about 1 - 6q, accepted eps 2.8 q^2 | VN | exact transfer matrices (MC cross-checked), forward-filter MC (Section 5, 5b) |
| A5-distance3-C4 | Generators X Z Z^-1 X^-1 I and cyclic shifts define [[5,1,3]]_3 with d = 3 (80 weight-3 logicals); the 7-qutrit colour code with star bipartition is [[7,1,3]]_3; the rotated [[9,1,3]]_3 and the unrotated [[13,1,3]]_3 are valid d = 3 qutrit codes | VN | brute-force commutation, rank and distance (Section 2) |
| A5-distance3-C5 | Fold-transversal H_L (SWAP on mirror pairs, F^((-1)^row)) and P_L (P^((-1)^i) on the diagonal, CZ^((-1)^i) on pairs) on [[13,1,3]]_3 generate the full logical SL(2,3); the same rule reproduces the repository's [[5,1,2]]_3 gates | VN | logical action matrices and group closure (Section 3) |
| A5-distance3-C6 | [[7,1,3]]_3: F/F^dagger and P/P^-1 on the two bipartition classes are transversal H_L and P_L generating SL(2,3); uniform F or P does not preserve the code | VN | Section 3 |
| A5-distance3-C7 | [[9,1,3]]_3: transversal F^s preserves the code only together with a 90-degree or 270-degree rotation; no fold-type P^t/CZ^u pattern along any of the 4 reflections preserves it | VN | exhaustive searches (Section 3) |
| A5-distance3-C8 | [[5,1,3]]_3 has exactly 80 local-Clifford x permutation automorphisms; their logical actions form a group of order 8 with no order-3 (P-type) element (order 4 without permutations) | VN | meet-in-the-middle search over 24^5 x 120 (Section 3) |
| A5-distance3-C9 | One-ancilla-per-check extraction tolerates any single fault for a suitable coupling order: [[9,1,3]]_3 has 16 of 24 safe orders per weight-4 check; for [[13,1,3]]_3 every order of each check is safe individually and a jointly safe choice was verified; fold-pair gate faults of [[13,1,3]]_3 are correctable. For [[7,1,3]]_3 and [[5,1,3]]_3 no order of any check works, so flags are needed | VN | collision test on the single-fault error sets (Section 3b) |
| A5-distance3-C10 | Code-capacity MLD coefficients: 35/4 ([[5,1,3]]), 51/4 ([[7,1,3]]), 93/8 ([[9,1,3]]), 89/16 ([[13,1,3]]); break-even q* = 0.157, 0.105, 0.110, 0.152 | VN | exact weight-1/2 counts and DFT distributions (Section 4) |
| A5-distance3-C11 | Phenomenological q_m = q: memoryless lookup costs a factor of 4 to 7 in the d = 3 q^2 coefficient (47, 83, 66 for [[5,1,3]], [[7,1,3]], [[9,1,3]]); space-time decoders give about 11 to 14 ([[5,1,3]] joint), about 13 ([[7,1,3]] joint) and about 15 to 16 ([[9,1,3]] joint) at q = 0.02; separate X/Z filters cost about 2x | VN | Section 5 (exact) and 5b/5c (MC) |
| A5-distance3-C12 | Baseline [[5,1,2]]_3 LRB circuits, first order in p: optimal correction 1 - f = 16.6 p per depth step (eps = 14.7 p), versus 5.6 p (eps = 5.0 p) for physical RB; detection rejection 112 p per step with zero first-order accepted error per step (offset 4.95 p); hybrid rejection 40 p per step | VN | `A5_circuit_first_order.py`, 10 sequences x 5 depths; matches baseline m = 0 fidelity 0.9925 and rejection 6.1% [data] |
| A5-distance3-C13 | Of the 11.06 p per step of d = 2 correction failures, 7.65 p come from data idle noise in the measurement and reset layers; measurement flips and reset errors contribute nothing at first order | VN | category breakdown in `A5_circuit_first_order.py` |
| A5-distance3-C14 | The d = 2 circuit coefficient is reproduced by eps = q_eff with q_eff = 10p (idle) + 2.16p (CNOT) + 2.9p (logical gates) = 15.1p (measured 14.7p) | VA + VN | Section 3.2 arithmetic plus C12 |
| A5-distance3-C15 | d = 3 correction beats the physical qutrit for p below about 1.3e-3 to 3.4e-3 with simultaneous ancilla readout, but only below about 1e-4 to 6e-4 with the repository's one-ancilla-per-layer readout; it beats d = 2 correction for p below about 4e-3 to 1e-2 (simultaneous) | CONJ | phenomenological coefficients (C11) times SI1000 kappa_1 (Section 3.3); no d = 3 circuit simulated |
| A5-distance3-C16 | No d = 3 code with correction reaches the postselected per-round error of [[5,1,2]]_3 detection (about 27 p^2 per step) under SI1000-like noise; A kappa^2 is at least several hundred | CONJ | Section 3.3 item 1; fit table r = 2.03e-5 at p = 1e-3 [data] |
| A5-distance3-C17 | For d = 3 correction the current LRB structure (no checks between inverse Cliffords) gives failures of order (m lambda p)^2 from the unchecked inverse block, so a corrected LRB needs a check after every Clifford or a single compiled inverse | VA | circuit structure [code] plus counting (Section 3.4) |
| A5-distance3-C18 | With detection the accepted decay is resolvable only with about 1e7 or more shots per depth at p = 1e-3, and the cost grows as 1/p; with d = 3 correction about 10^2 to 10^3 shots suffice independent of p | VA (order of magnitude) | Section 3.4 scaling argument using C12 and the baseline numbers |
| A5-distance3-C19 | Recommended next code: [[13,1,3]]_3 folded surface code with simultaneous readout; runner-up [[7,1,3]]_3 colour code with flags; [[9,1,3]]_3 and [[5,1,3]]_3 are unsuited to full-Clifford LRB | CONJ (decision built on C5 to C11 and C15) | Sections 1 to 4 |
| A5-distance3-C20 | The repository's 216-string Clifford catalog covers only 206 Clifford elements, with non-uniform symplectic multiplicities (19 x 9, 2 x 5, 10, 12, 13), so the LRB twirl is not uniform | VN | `A5_catalog_check.py` |
| A5-distance3-C21 | Folded square surface codes with transversal Clifford gates exist for qudits of any dimension, with 2D^2 - 2D + 1 data qudits; the rotated (diamond) code cannot be folded because of chiral edges | LIT | Moussa 2016 |
| A5-distance3-C22 | Qudit colour codes support transversal Clifford gates applied as U on unstarred and U* on starred vertices of a star bipartition | LIT | Watson et al. 2015 |
| A5-distance3-C23 | Qubit [[5,1,3]] and [[7,1,3]] codes need flags for fault-tolerant single-ancilla syndrome extraction; two extra qubits suffice | LIT | Chao and Reichardt 2018 |
| A5-distance3-C24 | Rotated surface code: logical H by patch rotation and S by a fold-transversal step inside a syndrome round (qubits, neutral atoms) | LIT | Chen et al. 2024 |

## 7. Caveats

- The phenomenological model uses i.i.d. depolarizing data noise and
  symmetric outcome shifts. The circuit-level first-order analysis covers
  only d = 2. No d = 3 circuit was simulated, so every d = 3 circuit
  statement is a conjecture.
- The space-time Monte Carlo values have 5 to 20% statistical errors (bands
  in the log). The per-round extraction compares 6 rounds with 1 round and
  ignores boundary effects of order 1/5.
- The [[13,1,3]]_3 phenomenological coefficient was obtained only with the
  separate X/Z filter, which is an upper bound. A joint filter on
  3^14 = 4.8M states per shot was too slow for a login node.
- The kappa_1 estimates count the SI1000 noise terms that dominate at d = 2:
  idle, CNOT and logical gates. For [[7,1,3]]_3 they exclude flag overhead.
  For [[9,1,3]]_3 they assume a P_L that the code does not have.

## 8. Literature fetched in this task

1. J. E. Moussa, "Transversal Clifford gates on folded surface codes",
   Phys. Rev. A 94, 042316 (2016), arXiv:1603.02286,
   DOI 10.1103/PhysRevA.94.042316. Fetched: https://arxiv.org/abs/1603.02286
   and https://ar5iv.arxiv.org/html/1603.02286.
2. N. P. Breuckmann, S. Burton, "Fold-Transversal Clifford Gates for Quantum
   Codes", Quantum 8, 1372 (2024), arXiv:2202.06647,
   DOI 10.22331/q-2024-06-13-1372. Fetched: https://arxiv.org/abs/2202.06647
   (used only for the term "fold-transversal").
3. F. H. E. Watson, E. T. Campbell, H. Anwar, D. E. Browne, "Qudit color codes
   and gauge color codes in all spatial dimensions", Phys. Rev. A 92, 022312
   (2015), arXiv:1503.08800, DOI 10.1103/PhysRevA.92.022312. Fetched:
   https://arxiv.org/abs/1503.08800 and https://ar5iv.arxiv.org/html/1503.08800.
4. H. F. Chau, "Five quantum register error correction code for higher spin
   systems", Phys. Rev. A 56, R1 (1997), arXiv:quant-ph/9702033,
   DOI 10.1103/PhysRevA.56.R1. Fetched: https://arxiv.org/abs/quant-ph/9702033.
5. E. M. Rains, "Nonbinary quantum codes" (1997), arXiv:quant-ph/9703048.
   Fetched: https://arxiv.org/abs/quant-ph/9703048.
6. Error Correction Zoo, "[[5,1,3]]_q modular-qudit code" (web page; generators
   X Z Z^dagger X^dagger I). Fetched: https://errorcorrectionzoo.org/c/qudit_5_1_3.
7. R. Chao, B. W. Reichardt, "Quantum error correction with only two extra
   qubits", Phys. Rev. Lett. 121, 050502 (2018), arXiv:1705.02329,
   DOI 10.1103/PhysRevLett.121.050502. Fetched: https://arxiv.org/abs/1705.02329.
8. Y. Tomita, K. M. Svore, "Low-distance surface codes under realistic quantum
   noise", Phys. Rev. A 90, 062320 (2014), arXiv:1404.3747,
   DOI 10.1103/PhysRevA.90.062320. Fetched: https://arxiv.org/abs/1404.3747 and
   https://ar5iv.arxiv.org/html/1404.3747. The paper gives Surface-17 as 9
   data + 8 syndrome qubits and Surface-25 as 13 + 12, with a reordered CNOT
   sequence so that a single syndrome-qubit fault does not cause a logical
   error.
9. Z.-H. Chen, M.-C. Chen, C.-Y. Lu, J.-W. Pan, "Transversal Logical Clifford
   gates on rotated surface codes with reconfigurable neutral atom arrays"
   (2024), arXiv:2412.01391. Fetched: https://arxiv.org/abs/2412.01391.
10. F. H. E. Watson, H. Anwar, D. E. Browne, "A fast fault-tolerant decoder for
    qubit and qudit surface codes", Phys. Rev. A 92, 032309 (2015),
    arXiv:1411.3028, DOI 10.1103/PhysRevA.92.032309. Fetched:
    https://arxiv.org/abs/1411.3028 (context: qudit surface codes with noisy
    syndromes have been decoded before; no qutrit numbers used here).

A search for flag gadgets designed for qudits found no qudit-specific paper.
This is reported as "not found", not as "does not exist".

## 9. Reproduction

```bash
module load python/3.11
cd docs/correction-decoders/qutrit-512/notes/scripts
OMP_NUM_THREADS=2 python A5_code_capacity_compare.py                 # about 2 min
OMP_NUM_THREADS=2 python A5_code_capacity_compare.py --phen1313 --joint-check  # about 5.5 min, used for A5_out/
OMP_NUM_THREADS=2 python A5_code_capacity_compare.py --validate      # adds the transfer-matrix MC check (~20 s)
python A5_circuit_first_order.py --cliffs 10 --depth-idx 0 1 2 3 4   # about 1 min
python A5_catalog_check.py                                           # seconds
```

`A5_circuit_first_order.py` only reads the baseline run folder (read-only) and
writes to `scripts/A5_out/`. Nothing is written outside the notes folder.
