# L8: distance-3 qutrit codes for correction-based LRB, and qudit QEC theory on correction versus detection

Agent L8, written 2026-10-28. Scope set by the brief (`00-brief.md`): find distance-3 qutrit codes
that could replace the `[[5,1,2]]_3` folded surface code in a correction-based LRB, extract their
transversal or fold-transversal logical Cliffords, syndrome-extraction cost and known
pseudo-thresholds, and collect statements on when correction beats detection or an unencoded qudit.

Labels used on every claim:

- [verified analytically]: I derived it here, or it follows by short algebra from cited facts.
- [verified numerically]: computed by `L8-scripts/d3codes_check.py` in this folder. The run log is
  `L8-scripts/d3codes_check.log`.
- [from literature: X]: stated in paper X, which I fetched in this task. Section 3 lists the URLs.
- [conjecture]: my expectation, not checked.

Reproduction: `module load python/3.11; cd L8-scripts; python -I d3codes_check.py` runs every
check in 81 s on one login-node core. The script sets the OpenMP, OpenBLAS and MKL thread counts
to 1 before it imports numpy. Section names can be passed as arguments:
`512 chau 513 513p 713 a7 913 sq3 css5 css6 design`. The script uses numpy only and does
symplectic linear algebra mod 3.

Conventions. A qutrit Pauli `X^a Z^b` is the vector `(a, b)` mod 3. Gate names follow sdim, whose
docstrings I read in the installed package: `F` is sdim's `H` with `X -> Z`, `Z -> X^-1`; `P` maps
`X -> XZ`, `Z -> Z`; `SUM` is sdim's `CNOT`; `CZ` maps `X_i -> X_i Z_j`. A single-qutrit Clifford
modulo Paulis is a matrix in SL(2,3). I write it row-major as `(a, b, c, d)` with columns equal to
the images of `X` and `Z`, so `F = (0,2,1,0)` and `P = (1,0,1,1)`. SL(2,3) has 24 elements; with
the 9 Paulis it gives the 216-element Clifford catalog used by the LRB code [verified numerically:
log section 8 generates 216 elements from F, P, X, Z].

One fact used throughout [verified analytically]: for odd prime d, write the symmetric Weyl
operators `W(v)`, which satisfy `W(u)W(v) = w^([u,v]/2) W(u+v)`. A Clifford U acts as
`U W(v) U^dag = w^f(v) W(Mv)` with M symplectic. Applying U to the product rule gives
`f(u+v) = f(u) + f(v)`, so f is linear, f(v) = [a,v] for some a. Up to a global phase, U is
then the Pauli W(a) times a unitary that maps `W(v)` to `W(Mv)` with no phase. So a circuit
whose symplectic map sends the stabilizer space onto itself maps the code space to a
Pauli-shifted copy of itself. One known transversal Pauli, which can live in the Pauli frame,
undoes the shift. The qubit phase conditions behind "doubly even" codes and Breuckmann-Burton's
parity conditions on fixed points therefore do not constrain qutrit codes. All gate checks below
are mod-3 symplectic checks.

---

## 1. Bottom line

1. The direct distance-3 analogue of the baseline is the D=3 folded "square" (unrotated) surface
   code `[[13,1,3]]_3`. Moussa built this family for qudits of any dimension
   [from literature: Moussa 2016]. The baseline `[[5,1,2]]_3` is exactly its D=2 member, with the
   same edge orientations and the same gate rule [verified numerically]. With that rule:
   - logical F is SWAP on the 4 mirror pairs, F on the 9 "even-even" qutrits and F^-1 on the 4
     "odd-odd" qutrits;
   - logical P is P on the 3 even-even fold qutrits, P^-1 on the 2 odd-odd fold qutrits, CZ on the
     3 even-even pairs and CZ^-1 on the odd-odd pair.

   These two generate SL(2,3), so all 216 logical Cliffords compile from the same H/P/X/Z tokens
   the current generator uses [verified numerically]. Every single fault on a fold gate, meaning an
   arbitrary two-qutrit Pauli on one mirror pair, is correctable. Every single fault in a
   one-ancilla-per-check syndrome circuit is correctable for every CNOT order of every check, and
   both statements hold jointly [verified numerically]. Cost: 12 checks (8 of weight 3, 4 of
   weight 4), 40 two-qutrit gates per round, 25 qutrits with one ancilla per check.
2. The smallest distance-3 qutrit code is `[[5,1,3]]_3` [from literature: Chau 1997, Rains 1999].
   Its transversal logical group is only Z4 = {±I, ±K}, with K = P F P^-1 on every qutrit.
   Adding qudit permutations (the 20 affine maps i -> a i + b mod 5) enlarges it to the quaternion
   group Q8. Q8 contains logical F: F on all five qutrits combined with the qutrit permutation
   i -> 2i mod 5. There is no transversal or permutation-transversal logical P
   [verified numerically]. One-ancilla extraction is not
   hook-safe for any of the 24^4 CNOT orders [verified numerically], which agrees with the
   flag-qudit result of Keppens et al. 2025 [from literature].
3. No CSS `[[5,1,3]]_3` or `[[6,1,3]]_3` exists. The exhaustive search covers all CSS pairs,
   degenerate ones included [verified numerically]. The qutrit Steane/2D colour code `[[7,1,3]]_3`
   therefore is the smallest CSS candidate. Its whole logical Clifford group is transversal:
   F on the 4 "unstarred" qutrits with F^-1 on the 3 "starred" ones, and the same split for P
   [verified numerically; from literature: Watson et al. 2015, Moussa 2016]. Logical gates need
   no two-qutrit gates at all. The catch is that one-ancilla extraction is not hook-safe for any
   order [verified numerically], so it needs flags or Steane/Knill-type extraction.
4. The rotated `[[9,1,3]]_3` code has hook-safe one-ancilla orders. Among the eight symmetries
   of its 3x3 lattice, only the two 90-degree rotations act as ZX-dualities, and they are not
   involutions. Logical F needs F/F^-1 on all qutrits plus a 4-cycle permutation of the lattice,
   and no phase-type fold gate exists on the orbits of any involutive lattice symmetry
   [verified numerically]. I did not search permutations outside the lattice symmetries. Moussa states that the rotated ("diamond") code cannot be folded and
   builds 13- and 17-qudit "cone" codes instead [from literature: Moussa 2016].
5. Correction versus detection. Detection on a distance-2 code and minimum-weight correction on a
   distance-3 code both fail at second order in the fault rate. Correction does not lower the
   order of the accepted-shot error. It removes rejection and pays with a larger second-order
   prefactor, because the circuit is bigger [verified analytically]. Detection on a distance-3
   code fails at third order, with first-order rejection, if single faults never pair into a
   logical. That holds for the folded square but not for the unflagged 5- and 7-qutrit codes
   (section 5). A distance-4 code that corrects one fault and
   rejects two gets O(p^3) error with O(p^2) rejection [from literature: Prabhu and Reichardt 2024,
   qubit codes; the counting is dimension-independent]. The only circuit-level pseudo-threshold I
   found for a distance-3 qutrit code is about 3x10^-4 per time step for `[[5,1,3]]_3` with flags,
   and about 7x10^-7 without them [from literature: Keppens et al. 2025]. I expect a distance-3
   correction LRB at the brief's p = 1e-3 to show lower per-shot fidelity than the baseline
   LRB-D, in exchange for no rejection [conjecture].
6. Experiments. The one beyond-break-even qudit QEC result is a single-mode GKP qutrit and
   ququart memory, with a qutrit gain of 1.82 +/- 0.03 [from literature: Brock et al. 2025]. A Z3
   toric-code ground state has been prepared on 24 qutrits, each encoded in two trapped-ion qubits
   [from literature: Iqbal et al. 2025]. I found no experiment that runs repeated syndrome
   extraction on a multi-qutrit stabilizer code with native qutrits. That is a negative search
   result, not proof of absence.

## 2. Candidate comparison

All rows are [verified numerically] except the cone-code row, which is [from literature: Moussa 2016].
"Hook-safe" means every single fault in a one-ancilla-per-check circuit, including a hook, gives
a data error that a syndrome lookup corrects. Measurement errors are excluded and handled by
repetition. Gate counts for logical F and P are physical gates per logical token.

| Code | Data qutrits | Checks | 2-qutrit gates per round (1 ancilla/check) | One-ancilla hook-safe? | Logical Cliffords without ancillas | Logical F / logical P |
|---|---|---|---|---|---|---|
| `[[5,1,2]]_3` folded square D=2, the baseline | 5 | 2 X + 2 Z, weight 3 | 12 | distance 2; every single fault is detectable with the brief's order | all of SL(2,3), fold-transversal | 1 SWAP + 4 F + 1 F^-1 / 1 CZ + 2 P + 1 P^-1 |
| `[[5,1,3]]_3` five-qutrit code | 5 | 4 mixed, weight 4 | 16 | no, 0 of 24^4 orders | Z4 transversal; Q8 with permutations; no logical P | 5 x F + permutation i -> 2i mod 5 / needs a non-transversal gadget |
| `[[7,1,3]]_3` qutrit Steane / colour | 7 | 3 X + 3 Z, weight 4 | 24 | no, 0 of 24^6 | all of SL(2,3), transversal with a 4+3 star split | 4 F + 3 F^-1 / 4 P + 3 P^-1, no two-qutrit gates |
| `[[9,1,3]]_3` rotated surface | 9 | 4 X + 4 Z, weights 4,4,2,2 | 24 | yes; 2/3 of weight-4 orders work alone; at least 5000 global assignments | F only, with a 90-degree lattice rotation; no fold P | 9 single-qutrit gates + 4-cycle permutations / none |
| `[[13,1,3]]_3` folded square D=3 (Moussa) | 13 | 6 X + 6 Z, 8 of weight 3, 4 of weight 4 | 40 | yes, every order of every check; fold-pair faults also correctable | all of SL(2,3), fold-transversal | 4 SWAP + 9 F + 4 F^-1 / 3 CZ + 1 CZ^-1 + 3 P + 2 P^-1 |
| cone code, minimal | 13 | n/a | n/a | n/a | all of SL(2,3), fold-transversal | distorted edge, stabilizers incompatible with the rotated code |
| cone code, expanded | 17 | n/a | n/a | n/a | all of SL(2,3), fold-transversal | converts to and from the rotated code by lattice surgery |

Two remarks on cost under the brief's SI1000 model [verified analytically from the brief]:

- Logical gate noise. A logical F or P costs one two-qutrit depolarizing location (probability p)
  in the baseline, four in `[[13,1,3]]_3`, and none in `[[7,1,3]]_3`. The colour code's logical
  gates carry only the 0.1p single-qutrit locations.
- Idle noise from sequential readout. The brief measures one ancilla per layer, and each
  measurement layer adds 2p idle depolarizing to every other wire. To first order a data qutrit
  collects 2p per check per round, plus 2p from the reset layer: 8p + 2p in the baseline,
  24p + 2p with the 12 checks of `[[13,1,3]]_3`, and 12p + 2p with the 6 checks of
  `[[7,1,3]]_3` before any flags. A correction study with these codes needs parallel ancilla
  readout, or the idle term dominates.

## 3. Paper notes

### 3.1 Chau 1997

H. F. Chau, "Five quantum register error correction code for higher spin systems,"
Phys. Rev. A 56, R1-R4 (1997). arXiv:quant-ph/9702033. DOI 10.1103/PhysRevA.56.R1.
Fetched: https://arxiv.org/abs/quant-ph/9702033 (abstract page),
https://arxiv.org/e-print/quant-ph/9702033 (LaTeX source, read in full),
https://api.crossref.org/works/10.1103/PhysRevA.56.R1 (metadata).

Extracted [from literature: Chau 1997]:

- For any register dimension N, prime or not, the encoding
  `|k> -> N^(-3/2) sum_{p,q,r} w^(k(p+q+r)+pr) |p+q+k, p+r, q+r, p, q>` corrects an arbitrary
  error on any one of five registers. The proof checks the Knill-Laflamme conditions pair by pair
  with the character sum rule.
- Optimality: no code of length 4 or less corrects an arbitrary single-register error. The proof
  is a reduced-density-matrix argument after Knill and Laflamme. For N=2 the code is equivalent to
  the Laflamme et al. / Bennett et al. 5-qubit code up to a register permutation and phases.
- An encoding circuit is given: SUMs, Fourier transforms on three registers, the phase
  `w^(pr)`, more SUMs. The paper calls the phase "Toffoli-like". For qudits it is a two-register
  controlled-phase, which is Clifford [verified analytically].
- The paper does not treat fault tolerance, logical gates or syndrome circuits.

My checks [verified numerically]: for N=3, exactly 81 Weyl operators act as scalars on Chau's code
space, so it is a stabilizer code with 4 generators, k=1 and d=3. The generators found are listed in
the log. Under qudit permutation times local Clifford it is equivalent to the cyclic code
`X Z Z^-1 X^-1 I` (80 equivalence maps over 20 permutations).

Bearing on this study: this paper establishes `[[5,1,3]]_3` as the smallest qutrit code that
corrects one error. Its gate set is the problem (section 4.2).

### 3.2 Rains 1999

E. M. Rains, "Nonbinary quantum codes," IEEE Trans. Inf. Theory 45(6), 1827-1832 (1999).
arXiv:quant-ph/9703048. DOI 10.1109/18.782103.
Fetched: https://arxiv.org/abs/quant-ph/9703048, https://arxiv.org/e-print/quant-ph/9703048
(source, read in full), https://authors.library.caltech.edu/records/js205-g9q18 (journal record).

Extracted [from literature: Rains 1999]:

- Symplectic codes `[[n,k,d]]_p` are self-orthogonal subspaces of (GF(p) x GF(p))^n. The group
  G_n = S_n x| Sp_2(p)^n of permutations times local symplectic maps acts on codes and defines
  equivalence and automorphisms. My "transversal + permutation" search in section 4 searches
  exactly this group.
- Quantum Singleton bound `K <= alpha^(n-2d+2)`. Equality forces purity to weight n-d+2.
- Theorem 13: `((5, alpha, 3))_alpha` and `((6, 1, 4))_alpha` exist for every alphabet size
  alpha > 1. They come from quantum quadratic-residue codes over Z[(1+sqrt 5)/2].
- Codes invariant under a 2-dimensional algebra A fall into three classes: split
  (A = GF(p) x GF(p), the CSS analogue), inert (A = GF(p^2), the analogue of GF(4)-linear codes)
  and ramified.
- Theorem 14 and its examples: the operations that act transversally on every A-linear code over m
  blocks are the symplectic elements of Mat_m(A). For m = 1 and inert A this is the norm-1
  subgroup of GF(p^2)*, which has order p+1. For CSS codes the cross-block operations form
  GL_m(GF(p)), for example SUM between blocks.

Analysis [verified analytically]: 3 is inert in Q(sqrt 5) because 5 is a non-residue mod 3, so the
five-qutrit QR code is GF(9)-linear. Multiplication by a norm-1 element of GF(9) is then a
transversal symplectic map on every such code, and these elements form a cyclic group of order
p+1 = 4. For p = 2 the same count gives order 3, the familiar transversal "SH" of the 5-qubit
code. Numerically I found exactly this Z4 and no other transversal Clifford, for two different
generator conventions (section 4.2). In particular the qutrit code has no transversal Clifford of
order 3 [verified numerically].

Bearing: the minimal code has only a cyclic transversal group of order 4. A full Clifford set for
LRB has to come from somewhere else.

### 3.3 Ashikhmin and Knill 2001

A. Ashikhmin and E. Knill, "Nonbinary quantum stabilizer codes," IEEE Trans. Inf. Theory 47(7),
3065-3072 (2001). arXiv:quant-ph/0005008. DOI 10.1109/18.959288.
Fetched: https://arxiv.org/abs/quant-ph/0005008, https://arxiv.org/e-print/quant-ph/0005008
(source), https://api.crossref.org/works/10.1109/18.959288 (metadata).

Extracted [from literature: Ashikhmin and Knill 2001]:

- q-ary stabilizer codes (q a prime power) are defined through nonbinary error bases. They
  correspond to self-orthogonal codes over GF(q^2) under a trace-symplectic form, which
  generalizes the GF(4) picture.
- Distance-3 families taken from Bierbrauer and Edel: `[[q^r, q^r-(r+2), 3]]_q`,
  `[[q^2+1, q^2-3, 3]]_q`, `[[(q^(r+2)-1)/(q^2-1), (q^(r+2)-1)/(q^2-1)-(r+2), 3]]_q` for even r,
  and others.

For q = 3 these give `[[9,5,3]]_3` and `[[10,6,3]]_3`. The second meets the quantum Hamming bound
of section 3.4 with equality, since 3^6 x 81 = 3^10, so it is perfect [verified analytically].

Bearing: high-rate distance-3 qutrit codes exist, but they add nothing for single-qutrit LRB and
the paper gives no gate constructions.

### 3.4 Ketkar, Klappenecker, Kumar and Sarvepalli 2006

A. Ketkar, A. Klappenecker, S. Kumar, P. K. Sarvepalli, "Nonbinary stabilizer codes over finite
fields," IEEE Trans. Inf. Theory 52(11), 4892-4914 (2006). arXiv:quant-ph/0508070.
DOI 10.1109/TIT.2006.883612.
Fetched: https://arxiv.org/abs/quant-ph/0508070, https://arxiv.org/e-print/quant-ph/0508070
(source, relevant sections read), https://api.crossref.org/works/10.1109/TIT.2006.883612.

Extracted [from literature: KKKS 2006]:

- Quantum Hamming bound for any `((n,K,3))_q` stabilizer code with K > 1, impure codes included:
  `K <= q^n / (n(q^2-1)+1)`. Linear programming gives it for n >= 5 and the Singleton bound below
  that.
- Cyclic quantum Hamming codes: a pure `[[n, n-2m, 3]]_q` code with n = (q^(2m)-1)/(q^2-1)
  exists when gcd(m, q^2-1) = 1.
- Maximal length of MDS codes: a nontrivial `[[n,k,d]]_q` MDS stabilizer code has
  n <= q^2 + d - 2. Under the classical MDS conjecture, n <= q^2 + 1 apart from two even-q
  exceptions. All quantum MDS codes are pure.

Consequences for qutrits [verified analytically]:

- With k = 1 the Hamming bound needs 3^(n-1) >= 8n + 1, so n >= 5.
- Distance-3 qutrit MDS codes `[[n, n-4, 3]]_3` can only exist for 5 <= n <= 10.
- The cyclic theorem does not cover `[[10,6,3]]_3`, because gcd(2, 8) = 2. Ashikhmin and Knill's
  list does cover it.
- Purity of `[[5,1,3]]_3` means every nontrivial stabilizer has weight 4 or more. I found weights
  4 and 5 [verified numerically].

Bearing: this fixes the smallest possible size (n = 5) and the maximal MDS length. The paper says
nothing about gates or circuits.

### 3.5 Gottesman 1998

D. Gottesman, "Fault-tolerant quantum computation with higher-dimensional systems,"
arXiv:quant-ph/9802007 (1998); Chaos, Solitons & Fractals 10(10), 1749-1758 (1999).
DOI 10.1016/S0960-0779(98)00218-5.
Fetched: https://arxiv.org/abs/quant-ph/9802007, https://arxiv.org/e-print/quant-ph/9802007
(source, read in full), https://api.crossref.org/works/10.1016/S0960-0779(98)00218-5.

Extracted [from literature: Gottesman 1998]:

- Qudit Pauli group from X_d and Z_d. For odd d, Pauli eigenvalues are powers of w, which the
  paper calls "actually simpler for odd d than for d=2".
- Clifford generators:
  - R, the DFT: X -> Z, Z -> X^-1;
  - P: |j> -> w^(j(j-1)/2)|j>, so X -> XZ;
  - SUM;
  - multiplication gates S_a: |j> -> |aj>.

  For prime d, P, R and S give the whole one-qudit Clifford group.
- Fault-tolerant measurement of Z-type Paulis uses transversal SUMs into a cat-like ancilla, the
  uniform superposition of strings that sum to 0 mod d. A general Pauli is measured after a
  transversal Clifford conjugation. This makes FT syndrome extraction possible for any prime-d
  stabilizer code.
- For any stabilizer code with prime d, logical P, R and S_a, and SUM between any two encoded
  qudits, come from measurement gadgets. Each gadget uses one logical ancilla and FT measurements
  of logical Pauli operators, with no transversality needed. A generalized Shor Toffoli completes
  a universal set.

Bearing: on `[[5,1,3]]_3` the missing logical P can in principle come from these gadgets. Each one
needs several FT logical-Pauli measurements with verified cat states, which is far heavier than
one transversal layer. That makes `[[5,1,3]]_3` a poor fit for an LRB that needs all 216 Cliffords
[conjecture on the relative cost].

### 3.6 Campbell 2014

E. T. Campbell, "Enhanced fault-tolerant quantum computing in d-level systems,"
Phys. Rev. Lett. 113, 230501 (2014). arXiv:1406.3055. DOI 10.1103/PhysRevLett.113.230501.
Fetched: https://arxiv.org/abs/1406.3055, https://arxiv.org/e-print/1406.3055 (source),
https://api.crossref.org/works/10.1103/PhysRevLett.113.230501.

Extracted [from literature: Campbell 2014]:

- Quantum Reed-Muller codes on n = d-1 qudits have transversal non-Clifford gates
  M_mu = exp of a cubic phase, in the third level of the Clifford hierarchy. The distance is
  floor((d+1)/3), so about d/3 errors are detected.
- The paper treats only primes d >= 5 ("For technical reasons, we consider only prime dimensions
  of five and above").
- Magic-state-distillation efficiency and thresholds improve with d.

Bearing: low. The "enhanced with d" claim is about distillation of non-Clifford resources. The
paper gives no Clifford-transversal distance-3 qutrit code and excludes d = 3.

### 3.7 Campbell, Anwar and Browne 2012

E. T. Campbell, H. Anwar, D. E. Browne, "Magic-state distillation in all prime dimensions using
quantum Reed-Muller codes," Phys. Rev. X 2, 041021 (2012). arXiv:1205.3104.
DOI 10.1103/PhysRevX.2.041021.
Fetched: https://arxiv.org/abs/1205.3104, https://arxiv.org/e-print/1205.3104 (source),
https://api.crossref.org/works/10.1103/PhysRevX.2.041021.

Extracted [from literature: Campbell, Anwar, Browne 2012]:

- QRM_3(2) is an 8-qutrit CSS code with 2 X-type and 5 Z-type generators and a transversal
  non-Clifford M_3 gate.
- Distilling with it suppresses error quadratically: eps' = eps^2 (3 + cos 4 theta) + O(eps^3).
  Thresholds are 0.20015 for general noise and 0.211 for depolarizing noise. Success probability
  is at least 1/9.
- A 4-ququint code is, in their words, the smallest stabilizer code with a transversal
  non-Clifford gate. Its depolarizing threshold is 0.363.
- On the state of the field in 2012: "We know of only one study of qudit fault tolerance
  thresholds". The authors suggest that qutrit and ququint fault-tolerance schemes may do better.

Bearing: relevant only if the LRB were extended beyond Cliffords.

### 3.8 Anwar, Campbell and Browne 2012

H. Anwar, E. T. Campbell, D. E. Browne, "Qutrit magic state distillation," New J. Phys. 14,
063006 (2012). arXiv:1202.2326. DOI 10.1088/1367-2630/14/6/063006.
Fetched: https://arxiv.org/abs/1202.2326, https://arxiv.org/e-print/1202.2326 (source),
https://api.crossref.org/works/10.1088/1367-2630/14/6/063006.

Extracted [from literature: Anwar et al. 2012]:

- The five-qudit code `[[5,1,3]]_d` has cyclic generators `X Z Z^-1 X^-1 I` "of the same form in
  all dimensions".
- It distils qutrit Hadamard eigenstates up to 23.3% depolarizing noise, against 18.3% for the
  5-qubit protocol, but suppresses noise only linearly.
- A seven-qutrit code comes from the Steane generators with -1 powers inserted. It distils mixed
  rather than pure states.

My checks [verified numerically]: their 7-qutrit code is `[[7,1,3]]_3` and has the full SL(2,3)
as local-Clifford logical gates. Logical F is F on qutrits {0,1,3,6} and F^-1 on {2,4,5}. Logical
P uses P and P^-1 on the same split.

Bearing: an independent, explicit qutrit Steane-type code with transversal logical Cliffords. Its
generator table is a ready-made alternative to the colour-code labeling of section 3.9.

### 3.9 Watson, Campbell, Anwar and Browne 2015

F. H. E. Watson, E. T. Campbell, H. Anwar, D. E. Browne, "Qudit color codes and gauge color codes
in all spatial dimensions," Phys. Rev. A 92, 022312 (2015). arXiv:1503.08800.
DOI 10.1103/PhysRevA.92.022312.
Fetched: https://arxiv.org/abs/1503.08800, https://arxiv.org/e-print/1503.08800 (source),
https://api.crossref.org/works/10.1103/PhysRevA.92.022312.

Extracted [from literature: Watson et al. 2015]:

- Any qubit colour-code lattice (a D-colex) defines a qudit colour code for every d through a
  "star bipartition" of vertices into starred and unstarred sets. Every cell holds equally many of
  each. X stabilizers are `X[v_unstarred] (x) X*[v_starred]`, which is X on all vertices; Z
  stabilizers are `Z[v_unstarred] (x) Z*[v_starred]`, which puts Z^-1 on starred vertices.
- The table of "star-conjugate transversal" gates:
  - H on unstarred and H* = H^-1 on starred, in 2D;
  - S on unstarred and S* on starred, in 2D and higher;
  - SUM applied blockwise;
  - non-Clifford gates in 3D and higher, with a separate construction for d = 3 and 6.
- No decoder is given. The authors note that for large d, neighbouring errors rarely cancel each
  other's syndromes, so decoders must use more of the syndrome.

My checks [verified numerically]: I labeled the 7-qutrit triangle with corners 0,1,2, centre 3
(unstarred) and edge midpoints 4,5,6 (starred). Then:

- the code is `[[7,1,3]]_3`;
- star-transversal F acts as logical F and star-transversal P as logical P, and together they
  generate SL(2,3);
- uniform F on all 7 qutrits and uniform P on all 7 are not logical gates;
- the code has exactly 24 local-Clifford automorphisms, one per logical Clifford modulo Paulis.

Bearing: the 2D qutrit colour code has the cheapest logical Cliffords of all candidates. A logical
gate is one layer of single-qutrit gates.

### 3.10 Moussa 2016 (main source for the replacement code)

J. E. Moussa, "Transversal Clifford gates on folded surface codes," Phys. Rev. A 94, 042316
(2016). arXiv:1603.02286. DOI 10.1103/PhysRevA.94.042316.
Fetched: https://arxiv.org/abs/1603.02286, https://arxiv.org/pdf/1603.02286,
https://arxiv.org/e-print/1603.02286 (source, sections 2-3 and the conclusion read in full).

Extracted [from literature: Moussa 2016]:

- Qudit conventions: X, Z, H as the DFT, CX. S is chosen nonstandard so that the conjugation rules
  do not depend on d, with `S X S^dag = -w^(-1/2) X Z`. Surface codes come from a directed planar
  graph. Vertices put X or X^dag on outgoing or incoming edges, and faces put Z or Z^dag by
  orientation.
- There are two families of distance D:
  - "square" (unrotated), with 2D^2 - 2D + 1 qudits and D^2 - D vertices and faces;
  - "diamond" (rotated), with D^2 qudits.
- Qudit Steane example: star-transversal H and S. With X <-> Z substitution on every qudit,
  stabilizer commutators would pick up w^4 or w^2. The paper also states that "Doubly even
  self-dual qubit codes have strongly transversal S gates since w^2 = 1, but this does not
  generalize to qudits".
- The folded square code is folded along a diagonal and has mirror-dual symmetry.
  - Logical H is H/H^dag on fold qudits and a "mirror" gate (H on both plus a swap) on each pair.
  - Logical S is S/S^dag on fold qudits and CZ/CZ^dag on pairs.
  - The cross product of the overlapping edge directions splits qudits into two types.
- "Because the diamond surface code has chiral edges, it is not possible to impose mirror-dual
  symmetry by folding." Instead the paper builds cone codes:
  - a minimal `[[2D^2-2D+1, 1, D]]` cone with a distorted edge, whose stabilizers do not commute
    with the diamond's;
  - an expanded `[[2D^2-1, 1, D]]` cone that converts to and from the diamond by lattice surgery.
- Fault tolerance: the gates "can spread errors between layers of the folded surface code within
  2-qudit transversal subsystems. This spreading of errors must be considered when studying the
  fault tolerance". The paper suggests treating the folded code as an unfolded code on qudit pairs
  and gives no numbers.
- Syndrome extraction:
  - on two physical layers it is the same as for an ordinary surface code;
  - on a planar layout it needs one global pairwise swap per round;
  - near the fold, an extra local CX^dag between neighbouring ancillas fixes a misordered CX;
  - it produces hook errors and spreading between layers within pairs.

  Folding an unfolded code by swaps takes depth 6D - 5.
- The paper quotes epsilon_L ~ epsilon_0 (epsilon_P/epsilon_T)^floor((D+1)/2) with epsilon_T ~ 0.01
  for qubit surface codes and ~0.001 for colour codes under per-gate depolarizing noise.
- On a standard qubit surface code, 8 of the 24 single-qubit Cliffords are transversal up to a
  rotation, 12 need one S and 4 need two. A logical S costs about 72 D^3 qubit-rounds. Folded
  codes make all 24 cheap.

My checks [verified numerically, sections 4.1 and 4.5]:

- The brief's `[[5,1,2]]_3` is exactly the D=2 folded square code. The qutrit positions (X,Y) with
  X+Y even in [0,2]^2 are 0=(0,0), 1=(1,1), 2=(2,0), 3=(0,2), 4=(2,2). Horizontal edges point in
  -X and vertical edges in +Y.
- The brief's H and P circuits act as logical F and logical P.
- The D=3 code built with the same rule has the fold gates, fault properties and hook safety given
  in section 1, item 1.

Bearing: this is the natural replacement. It keeps the same compiler, the same gate types and the
same style of check circuit.

### 3.11 Breuckmann and Burton 2024

N. P. Breuckmann, S. Burton, "Fold-transversal Clifford gates for quantum codes," Quantum 8, 1372
(2024). arXiv:2202.06647. DOI 10.22331/q-2024-06-13-1372.
Fetched: https://arxiv.org/abs/2202.06647, https://arxiv.org/e-print/2202.06647 (source, sections
on fold-transversal gates and the conclusion), https://api.crossref.org/works/10.22331/q-2024-06-13-1372.

Extracted [from literature: Breuckmann and Burton 2024]:

- A ZX-duality tau is a qubit permutation that maps X checks to Z checks. A fold-transversal
  circuit uses single-qubit gates plus two-qubit gates on the orbits {i, tau(i)}.
- H_tau, all SWAPs on orbits followed by H on every qubit, is a logical gate for any ZX-duality.
  H_tau^2 = I only if tau^2 = id.
- The phase-type S_tau, with S or S^dag on fixed points and CZ on 2-orbits, is a logical gate
  under four qubit conditions:
  - tau is an involution;
  - the fixed set has even size;
  - every X check has even overlap with the fixed set;
  - every X check contains an even number of 2-orbits.

  The conditions come from cancelling i^|s cap A| (-i)^|s cap B| phases.
- With the diagonal fold of the surface code, H_tau and S_tau reproduce Moussa's gates. "There is
  another ZX-duality that rotates the surface code by 90 degrees. However, this rotation fixes a
  single physical qubit and so we cannot construct a phase-type encoded logical gate from this
  ZX-duality."
- The scheme is called "trivially fault-tolerant", with no time overhead and no ancillas.
  Performance under noise is left to future work, and the authors expect the construction to
  extend to qudit codes. The worked example is the `[[30,8,3]]` Bring's code.

My checks [verified numerically]:

- On the qutrit `[[9,1,3]]_3` rotated code, only the 90-degree rotations with an F/F^-1 pattern
  give an H-type logical gate. No involutive lattice symmetry gives one, and no phase-type gate
  exists on the orbits of any involutive symmetry.
- On qutrits the qubit parity conditions become sublattice sign choices: P versus P^-1 and CZ
  versus CZ^-1. The D=3 fold has 5 fixed qutrits, an odd number, which would break the qubit
  condition but works for d = 3, as the phase argument at the top predicts.
- "Trivially fault-tolerant" needs a qualification at d = 3. A fault on an orbit gate is a
  weight-2 error inside one block, and whether that error is correctable depends on the code. It
  is correctable for the D=3 folded square [verified numerically].

### 3.12 Muralidharan, Zou, Li, Wen and Jiang 2017

S. Muralidharan, C.-L. Zou, L. Li, J. Wen, L. Jiang, "Overcoming erasure errors with multilevel
systems," New J. Phys. 19, 013026 (2017). arXiv:1504.08054. DOI 10.1088/1367-2630/aa573a.
Fetched: https://arxiv.org/abs/1504.08054, https://arxiv.org/e-print/1504.08054 (source),
https://api.crossref.org/works/10.1088/1367-2630/aa573a.

Extracted [from literature: Muralidharan et al. 2017]:

- Quantum polynomial codes `[[2k+1, 1, k+1]]_d` (d prime, CSS) correct k erasures. The qutrit case
  `[[3,1,2]]_3` stores one logical qutrit in three time-bin qutrits and is compared with the
  `[[4,2,2]]` qubit code.
- The erasure threshold is 50% for every code size, at the no-cloning limit, against about 37% for
  a D=3 surface code.
- Teleportation-based error correction is generalized to qudits for erasures plus operation
  errors, with an application to one-way repeaters. Polynomial codes beat qubit parity codes in
  some ranges of the operation-error model.
- Physical proposal: time-bin qutrits stored in 87Rb, and photonic CZ via a generalized
  Duan-Kimble scheme.

Bearing:

- A distance-d code corrects d-1 erasures [verified analytically, standard]. If the hardware
  heralds an error, for example by detecting leakage, the baseline `[[5,1,2]]_3` could correct
  that one located error instead of rejecting the shot [conjecture on how often that applies,
  which depends on the hardware].
- Larger qudit polynomial codes such as `[[5,1,3]]_5` and `[[7,1,4]]_7` need d >= n, so for
  qutrits this family stops at n = 3 [verified analytically].

### 3.13 Keppens, Eggerickx, Levajac, Simion and Soree 2025

J. Keppens, Q. Eggerickx, V. Levajac, G. Simion, B. Soree, "Qudit vs. qubit: simulated performance
of error-correction codes in higher dimensions," Phys. Rev. A 112, 032435 (2025). arXiv:2502.05992.
DOI 10.1103/2w52-qd2j.
Fetched: https://arxiv.org/abs/2502.05992, https://arxiv.org/html/2502.05992,
https://arxiv.org/e-print/2502.05992 (source; the threshold table was checked there),
https://api.crossref.org/works/10.1103/2w52-qd2j.

Extracted [from literature: Keppens et al. 2025]:

- Code and circuit:
  - the code is `[[5,1,3]]_q` for q = 2, 3, 5, with q = 7 as a spot check;
  - the generators are cyclic shifts of `I Z X X^dag Z^dag`;
  - extraction uses 4 ancillas, F_q/F_q^dag basis changes and SUM/SUM^dag gates;
  - simulations run 3 cycles.
- Noise models:
  - a per-cycle depolarizing model;
  - a circuit-level model at rate p_s per time step: single-qudit depolarizing on idles and
    single-qudit gates, two-qudit depolarizing on two-qudit gates, and ancilla shifts before
    readout, with the last cycle error-free.
- Decoders: MWPM via PyMatching, and belief-matching on a graph with q-1 nodes per ancilla. One
  flag qudit, reused across checks, catches hook errors, which a lookup table then corrects.
- Fits P_L = a p_s^b and level-by-level concatenation give "thresholds" (a, b, threshold):

  | q | flag | a | b | threshold |
  |---|---|---|---|---|
  | 2 | no | 36.7 | 1.264 | 1.21e-6 |
  | 2 | yes | 766 | 1.873 | 4.95e-4 |
  | 3 | no | 58.7 | 1.288 | 7.22e-7 |
  | 3 | yes | 1116 | 1.870 | 3.24e-4 |
  | 5 | no | 35.3 | 1.149 | 4.36e-11 |
  | 5 | yes | 792 | 1.798 | 2.32e-4 |

  The abstract reads "comparable error thresholds of the order of 10^-4 ... for qudits of
  dimensions 2, 3 and 5".
- The paper states that the only single-qudit transversal operations are X^(x5), Z^(x5) and
  T^(x5), with T_q = S_q F_q "the operation that cyclically shifts between the Heisenberg-Weyl
  gates".

My checks:

- The concatenation fixed point equals the level-1 pseudo-threshold p* = a^(-1/(b-1)). The rounded
  fit parameters give 3.1x10^-4 with a flag and 7.2x10^-7 without [verified analytically].
- The T_L claim fails for q = 3. With this generator convention and the sdim gate conventions,
  P F on every qutrit is not a logical gate. The only uniform transversal Cliffords are +/-I and
  +/-(P^-1 F^-1 P), both of order 4, and no transversal element has order 3 or 6
  [verified numerically], as Rains's count predicts (section 3.2). I did not check q = 5.
- No CNOT order makes one-ancilla extraction hook-safe [verified numerically]. This agrees with
  their need for flags.

Bearing: this is the only circuit-level pseudo-threshold I found for a distance-3 qutrit code,
about 3x10^-4 per time step with flags. The noise model is not SI1000, so it sets a scale only.

### 3.14 Kabir et al. 2025, the sdim paper

A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang, B. Soree, "Sdim: A qudit
stabilizer simulator," arXiv:2511.12777 (v1 Nov 2025, v2 May 2026). No journal reference listed.
Fetched: https://arxiv.org/abs/2511.12777, https://arxiv.org/html/2511.12777v2,
https://arxiv.org/e-print/2511.12777 (source; the quotes below were checked there).

Extracted [from literature: Kabir et al. 2025]:

- sdim is an open-source qudit stabilizer tableau and Pauli-frame sampler for all dimensions. It is
  the backend of this repository.
- The paper replicates Keppens et al.'s `[[5,1,3]]_q` study and extends it to q = 7 with
  belief-matching: "seemingly the first threshold graph of a high-dimensional (q>5) code under
  circuit-level noise". The value appears only in a figure. The conclusion says the more
  optimistic logical error rates "appear to challenge the conclusions presented in" Keppens
  et al.
- LRB-D case study on the `[[5,1,2]]_3` detection code. The protocol has "a single terminal
  syndrome extraction" with postselection on the all-zero syndrome. Results: "all the LRB-D curves
  decay much faster than a single physical, unprotected qutrit"; "The single round of
  postselection cannot adequately protect the logical state once physical errors build into
  logical errors during increasingly deeper error trajectories prior to postselection."
- Motivation: "full LRB is a demanding target for early qudit hardware". In distance-two codes
  "the syndrome can reveal that an error has occurred, but it generally does not provide enough
  information to identify a unique correction". The target is "a near-term experiment on a
  superconducting qutrit QPU", with no vendor named.

Bearing: an earlier form of this study's baseline. It used only a terminal check, while the
brief's runs add a check round after every forward Clifford. The paper also frames the
encoded-versus-unencoded comparison that Gottesman 2016 formalizes (section 3.20).

### 3.15 Duclos-Cianci and Poulin 2013

G. Duclos-Cianci, D. Poulin, "Kitaev's Z_d-code threshold estimates," Phys. Rev. A 87, 062338
(2013). arXiv:1302.3638. DOI 10.1103/PhysRevA.87.062338.
Fetched: https://arxiv.org/abs/1302.3638, https://arxiv.org/e-print/1302.3638 (source),
https://api.crossref.org/works/10.1103/PhysRevA.87.062338.

Extracted [from literature: Duclos-Cianci and Poulin 2013]:

- A renormalization-group decoder for the Z_d toric code under a generalized bit-flip model,
  without measurement errors.
- Thresholds for 2 <= d <= 6 "increase as a function of the local dimension d". They track a
  rescaled hashing bound, with C_3 ~ 0.159 against C_2 ~ 0.110.
- Measurement errors and qudit syndrome circuits are left open.

Bearing: code-capacity evidence that qutrit topological codes tolerate more noise per qudit than
qubit ones. It does not transfer to circuit noise.

### 3.16 Watson, Anwar and Browne 2015

F. H. E. Watson, H. Anwar, D. E. Browne, "Fast fault-tolerant decoder for qubit and qudit surface
codes," Phys. Rev. A 92, 032309 (2015). arXiv:1411.3028. DOI 10.1103/PhysRevA.92.032309.
Fetched: https://arxiv.org/abs/1411.3028, https://arxiv.org/e-print/1411.3028 (source),
https://api.crossref.org/works/10.1103/PhysRevA.92.032309.

Extracted [from literature: Watson, Anwar, Browne 2015]:

- A hard-decision RG decoder for qudit surface codes with noisy syndromes, under independent
  bit-flip errors on qudits and on syndrome bits.
- The qubit threshold is 2.15%. Thresholds rise with d and saturate near 4.2% because of syndrome
  percolation.
- An "initialization step" raises the large-d value to about 8.2% (d = 7919) but lowers it for
  small d.

Bearing: the phenomenological threshold is higher for qutrits than for qubits, but no
circuit-level data comes with it.

### 3.17 Marks, Jochym-O'Connor and Gheorghiu 2017

J. Marks, T. Jochym-O'Connor, V. Gheorghiu, "Comparison of memory thresholds for planar qudit
geometries," New J. Phys. 19, 113022 (2017). arXiv:1701.02335. DOI 10.1088/1367-2630/aa939a.
Fetched: https://arxiv.org/abs/1701.02335, https://arxiv.org/abs/1701.02335v2,
https://arxiv.org/e-print/1701.02335 (source), https://api.crossref.org/works/10.1088/1367-2630/aa939a.

Extracted [from literature: Marks et al. 2017]:

- A clustering (GCC) decoder for qudit 6.6.6 colour codes under code-capacity generalized
  bit-flip noise.
- The table in the v2 source gives, where the paper's D is the qudit dimension and not a
  distance:
  - surface code: MWPM 0.103 and HDRG 0.093 for qubits, HDRG 0.155 for large dimension;
  - colour code: DSP 0.080 and GCC 0.056 for qubits, GCC 0.119 for large dimension;
  - a figure caption gives GCC 0.084 for qutrits.
- The arXiv listing's abstract quotes different numbers: 9.75%, 11.4% and a 17.1% plateau. The
  body numbers above are the ones I trust.

Bearing: qudit colour-code thresholds also rise with d, but only code-capacity numbers exist.

### 3.18 Brock et al. 2025 (experiment)

B. L. Brock, S. Singh, A. Eickbusch, V. V. Sivak, A. Z. Ding, L. Frunzio, S. M. Girvin,
M. H. Devoret, "Quantum error correction of qudits beyond break-even," Nature 641, 612-618 (2025).
arXiv:2409.15065. DOI 10.1038/s41586-025-08899-y.
Fetched: https://arxiv.org/abs/2409.15065, https://arxiv.org/e-print/2409.15065 (source),
https://api.crossref.org/works/10.1038/s41586-025-08899-y.

Extracted [from literature: Brock et al. 2025]:

- Setup: a single-mode square GKP qutrit and ququart in a 3D superconducting cavity with a
  tantalum-transmon ancilla. A generalized small-Big-small protocol stabilizes the code with
  stabilizer length sqrt(pi d), and a reinforcement-learning agent tunes it as a memory.
- QEC gain: 1.82 +/- 0.03 for the qutrit and 1.87 +/- 0.03 for the ququart, relative to the best
  physical qudit, the cavity Fock qudit. The gain stays near 1.8 from d = 2 to 4.
- The experiment is a memory only, with no logical gates.

Bearing: qudit QEC beyond break-even exists, but in a bosonic memory, not a multi-transmon
stabilizer code. It supports the "bosonic qudits" option in the brief's hardware question.

### 3.19 Iqbal et al. 2025 (experiment)

M. Iqbal, A. Lyons, C. F. B. Lo, N. Tantivasadakarn, J. Dreiling, C. Foltz, T. M. Gatterman,
D. Gresh, N. Hewitt, C. A. Holliman, J. Johansen, B. Neyenhuis, Y. Matsuoka, M. Mills, S. A. Moses,
P. Siegfried, A. Vishwanath, R. Verresen, H. Dreyer, "Qutrit toric code and parafermions in trapped
ions," Nat. Commun. 16, 6301 (2025). arXiv:2411.04185. DOI 10.1038/s41467-025-61391-z.
Fetched: https://arxiv.org/abs/2411.04185, https://arxiv.org/e-print/2411.04185 (source),
https://api.crossref.org/works/10.1038/s41467-025-61391-z.

Extracted [from literature: Iqbal et al. 2025]:

- A Z3 toric-code ground state on up to 6 x 4 = 24 qutrits with fidelity per qutrit above 96.5(3)%.
- Hardware: Quantinuum H2, with 56 171Yb+ ions and pairs of qubits algorithmically encoded into
  qutrits.
- They make parafermion and charge-conjugation defects, fuse defects, and transfer entanglement
  between anyons and defects.

Bearing: a qutrit stabilizer state at scale, but without repeated syndrome extraction or
correction, and with qubit-encoded qutrits.

### 3.20 Gottesman 2016

D. Gottesman, "Quantum fault tolerance in small experiments," arXiv:1610.03507 (2016). No journal
reference listed.
Fetched: https://arxiv.org/abs/1610.03507, https://arxiv.org/e-print/1610.03507 (source, read in full).

Extracted [from literature: Gottesman 2016]:

- Criterion: an experiment demonstrates fault tolerance if the encoded circuit has a lower error
  rate than the unencoded one on the same hardware for every circuit in the chosen family, with
  preparation and measurement included.
- For a postselected code the fair comparison is the conditional error p_L/p_S, measured on
  accepted runs, against p_U, because "The unencoded circuit is never rejected".
- For the 4-qubit detection code without repeated detection, the bound is
  p < (4+2T)/(16T^2+122T+225) for T logical steps:

  | T | bound |
  |---|---|
  | 1 | 1.6% |
  | 4 | 1.2% |
  | 10 | 0.79% |
  | 100 | 0.12% |

  This "reflects the lack of repeated error correction or detection steps".
- On SWAPs: "The physical SWAP gate would not be fault-tolerant, since an error during it could
  introduce errors to two qubits, which might not be detected". A conceptual SWAP by relabeling is
  fine.

Bearing:

1. The right comparison for LRB-D is accepted-shot logical error against an unencoded qutrit, at
   every depth.
2. The brief's inverse half has no checks. Two faults there can combine into an undetected logical
   error with probability growing like (m lambda_C)^2, where lambda_C is the per-Clifford fault
   rate [verified analytically by counting]. That is Gottesman's "no repeated detection" regime
   and gives a non-exponential term in m.
3. The baseline's SWAP(2,3) does not have Gottesman's problem. None of the 81 two-qutrit Paulis on
   the pair {2,3} is an undetectable nontrivial logical, so a SWAP or CZ fault there is either
   detected or harmless [verified numerically].

### 3.21 Prabhu and Reichardt 2024

P. Prabhu, B. W. Reichardt, "Distance-four quantum codes with combined postselection and error
correction," Phys. Rev. A 110, 012419 (2024). arXiv:2112.03785. DOI 10.1103/PhysRevA.110.012419.
Fetched: https://arxiv.org/abs/2112.03785, https://arxiv.org/e-print/2112.03785 (source),
https://api.crossref.org/works/10.1103/PhysRevA.110.012419.

Extracted [from literature: Prabhu and Reichardt 2024]:

- "Distance-three codes may alternatively be used to detect one or two errors, but then they lose
  the ability to correct and computations are very short-lived."
- A distance-4 code that corrects one fault and rejects detected two-fault events has O(p^3)
  logical error and O(p^2) rejection. A postselected distance-4 surface code "accumulates 25 times
  less error than its distance-five counterpart".
- Rules for distance-4 fault tolerance: one fault must leave an error of X- and Z-weight at most
  one; two faults must be corrected or detected. Flag circuits can measure weight-8 stabilizers
  in this way.
- Take-home: postselection suits low-depth work, but "when logical information is stored for too
  long, it is likely to be wiped and reset".

Bearing: this is the hybrid between the baseline (detect only) and full correction. The qutrit
analogue would be the D=4 folded square `[[25,1,4]]_3` with correct-one/reject-two decoding
[conjecture: I did not check its fold-gate fault distance].

### 3.22 Combes, Granade, Ferrie and Flammia 2017

J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical randomized benchmarking,"
arXiv:1702.03688 (2017). No journal reference listed.
Fetched: https://arxiv.org/abs/1702.03688, https://arxiv.org/e-print/1702.03688 (source, protocol
section read).

Extracted [from literature: Combes et al. 2017]:

- The protocol applies each encoded Clifford, then "a round of error correction (with or without
  recovery)" and an ancilla refresh. A single Clifford at the end inverts the sequence. The fit is
  F_L(m) = A_L p_L^m + B_L.
- Assumptions: noise that is time-independent, Markovian and gate-independent at the logical
  level, and syndrome measurement errors that are uncorrelated between rounds.

Bearing: the original LRB is a correction-based protocol. The brief's LRB-D departs from it in two
ways: it uses m inverse Cliffords without checks instead of one inverting Clifford, and it
postselects.

## 4. My checks in detail

All results below are [verified numerically] by `L8-scripts/d3codes_check.py` unless marked.

### 4.1 Baseline `[[5,1,2]]_3`

- Basic parameters: k=1, d=2, and all four checks have weight 3. Weight-2 nontrivial logicals live
  only on the qutrit pairs {0,2}, {0,3}, {2,4} and {3,4}.
- Logical gates: the brief's H circuit (SWAP(2,3), H on {0,2,3,4}, H_INV on 1) acts as logical F.
  The brief's P circuit acts as logical P.
- Fold-pair faults: none of the 81 Paulis on the fold pair {2,3} is an undetectable nontrivial
  logical. A SWAP(2,3) or CZ(2,3) fault is always detected or harmless.
- Check-circuit faults: none of the 228 single faults in the four check circuits gives an
  undetectable logical, counting any single-qutrit error at a CNOT times any power of the hook
  suffix, with the brief's CNOT order. The baseline circuits are single-fault detecting, so
  accepted-shot failures need at least two faults.
- Relation to Moussa's family: after relabeling qutrits as 0=(0,0), 1=(1,1), 2=(2,0), 3=(0,2),
  4=(2,2), the brief's stabilizer space equals that of the D=2 member of the folded-square rule
  in section 4.5, and the brief's X-bar and Z-bar map to that construction's X-bar and Z-bar.

### 4.2 `[[5,1,3]]_3`

- Basic parameters: the cyclic code `X Z Z^-1 X^-1 I` has k=1 and d=3. Nontrivial stabilizer
  weights are 4 and 5.
- Uniform transversal Cliffords: exactly 4, namely {I, -I, K, -K} with K = (1,2,2,2) =
  P F P^-1 as an operator. The local-Clifford automorphisms with no permutation are the same 4.
  The group is Z4. It is not transitive on the 8 nonzero phase-space points, so the 36-element
  group it generates with the Paulis is not a 2-design.
- With permutations: there are 80 monomial automorphisms (permutation times local SL(2,3)) over
  20 permutations. These are exactly the affine maps i -> a i + b mod 5.
  - a in {1,4} pairs with uniform {±I, ±K};
  - a in {2,3} pairs with uniform {±F, ±K'}, where K' = (1,1,1,2) = P^-1 F^-1 P.

  In every case the logical action equals the local map, because X-bar = X on all five qutrits and
  Z-bar = Z on all five are invariant under permutations. So F on all five qutrits after the
  permutation i -> 2i is logical F. The induced logical group has order 8 with element orders
  1, 2, 4, 4, 4, 4, 4, 4. That is Q8.
- 2-design check: Weyl x| Q8 has 72 elements and frame potential exactly 2.000000, so it is a
  unitary 2-design. Weyl x| Z4 gives 3.000000. The full Clifford group gives 2.000000.
- Logical P: it has order 3, so it is not in Q8. No permutation-transversal logical P exists.
- The sdim/Keppens generator convention gives the same picture, with transversal +/-(P^-1 F^-1 P).
- Hooks: one-ancilla extraction is not hook-safe for any of the 24^4 = 331776 CNOT-order
  assignments. Every order of every check fails on its own.

Consequence [verified analytically from the above]: a `[[5,1,3]]_3` LRB is possible with all
transversal gates only over the 72-element group Weyl x| Q8. Even then the permutation i -> 2i,
a 4-cycle on qutrits 1, 2, 4, 3, must be a relabeling. Done with physical SWAPs, one fault leaves a
weight-2 error in a distance-3 block, which Gottesman 2016 also flags. Using the 216-element group
needs a non-transversal logical P.

### 4.3 `[[7,1,3]]_3` and CSS minimality

- Exhaustive search over all CSS codes with k=1:
  - n=5: 20570 (H_X, H_Z) pairs, no code with d >= 3;
  - n=6: 484484 pairs, none.

  (k_X, k_Z) and (k_Z, k_X) are related by an X <-> Z swap. A positive control on the 7-qutrit
  colour code passes and a weakened negative control fails. So `[[7,1,3]]_3` is the smallest CSS
  qutrit code with k=1 and d=3.
- Colour code with the star split: F on {0,1,2,3} and F^-1 on {4,5,6} is logical F, and the same
  split for P gives logical P. These generate SL(2,3), and there are exactly 24 local
  automorphisms.
- Hooks: no one-ancilla order assignment out of 24^6 is hook-safe, and every check fails for every
  order on its own. Example: a hook X_3 X_6 from face {0,4,3,6} has the same syndrome as X_1, and
  X_3 X_6 X_1^-1 is a weight-3 logical [verified analytically by hand, consistent with the search].

### 4.4 `[[9,1,3]]_3` rotated surface code

- Construction: X faces carry +1 exponents and Z faces carry the checkerboard signs (-1)^(r+c).
  Result: k=1, d=3; two weight-4 and two weight-2 checks of each type; 24 two-qutrit gates per
  round.
- ZX-dualities among the 8 lattice symmetries with an F/F^-1 pattern: only rot90 and rot270 work,
  2 sign patterns each, acting as logical F or F^-1.
- Phase-type gates on the orbits of the involutive symmetries (diag, anti, horiz, vert, rot180):
  none gives a nontrivial logical.
- Hooks: weight-4 checks are individually safe for 2/3 of their orders, and the backtracking search
  found at least 5000 global safe assignments before its cap. Example: X checks in row order
  (0,1,3,4) and Z checks in column order (1,4,2,5).

### 4.5 `[[13,1,3]]_3` folded square

- Construction: qutrits sit at (X,Y) in [0,4]^2 with X+Y even. X checks sit at (odd, even) with
  exponents +1 on left/below and -1 on right/above. Z checks sit at (even, odd) with +1 on
  top/left and -1 on bottom/right. X-bar is X on the column X=0 and Z-bar is Z on the row Y=0.
  This rule reproduces the baseline exactly at D=2.
- Parameters: k=1, d=3, 40 two-qutrit gates per round. The fold has 5 fixed qutrits and 4 mirror
  pairs.
- Fold gates: logical F is (0,2,1,0) and logical P is (1,0,1,1). Together they generate a group of
  order 24.
- Fault tolerance:
  - the error set formed by single-qutrit errors and all two-qutrit Paulis on any one mirror pair
    is correctable by syndrome lookup;
  - every one of the 9072 pairs of (check, order) choices is mutually consistent, so every
    order assignment, about 5.6x10^11 of them, is hook-safe;
  - the fold-pair error set is consistent with every (check, order) choice.

  All single faults (data, hooks, fold gates) are therefore correctable together, measurement
  errors aside.
- Why the hooks are harmless [verified analytically]: minimum-weight logicals are straight lines
  of one edge class. A hook from a vertex check is a horizontal pair or its equivalent vertical
  pair, and never two edges of one same-type line. The face checks behave the same way.

## 5. Correction versus detection, applied to this study

1. Order of failure [verified analytically]. With independent faults at rate about p and circuits
   that are fault-tolerant to the stated distance:

   | Strategy | Accepted-shot logical error | Rejection |
   |---|---|---|
   | distance 2, reject on any syndrome (the baseline) | O(p^2) | about 1 - exp(-lambda), lambda = total rate of detectable faults |
   | distance 3, minimum-weight correction | O(p^2) | none |
   | distance 3, reject on any syndrome | O(p^3) | O(p) |
   | distance 4, correct one / reject two | O(p^3) | O(p^2) |

   The baseline meets its premise: every single fault is detectable (section 4.1). The
   distance-3 rows need single faults that never combine in pairs into a logical. The fault sets
   are closed under inverses, so this is the same as the hook-safety condition. It holds for the
   folded square with any CNOT order (section 4.5). It fails for the 5- and 7-qutrit codes without
   flags, where one hook plus one more fault already makes a logical, so their detection error
   stays O(p^2) [verified analytically from the numerical hook results]. Moving from the baseline
   to distance-3 correction keeps the order. The prefactor, the number of malignant
   fault pairs, grows with circuit size: 40 against 12 two-qutrit gates per round, and three times
   as many measurement layers if readout stays sequential. Rejection goes to zero. At the brief's
   p = 1e-3 and depth 22, the baseline unif-1 run keeps 8.1% of shots at 0.9919 fidelity. I expect
   a distance-3 correction run to keep 100% of shots at lower fidelity [conjecture].
2. Measurement errors [verified analytically]. My hook analysis covers data errors only. Under
   SI1000 the largest single term is the 5p measurement shift. Detection turns a single
   measurement error into a rejection, never into a logical error. Correction with a per-round
   syndrome lookup turns it into a miscorrection, an O(p) logical error. A correction LRB
   therefore has to decode round-to-round syndrome differences over the whole check history,
   while the brief's detectors are raw per-round outcomes. The final check round has no later
   round to confirm it. The brief's ordinary circuits read only wires 0 and 3 at the end. A
   terminal X-basis readout of all data qutrits, as in the brief's const-0 circuits, gives
   error-free values of the X-type checks. Those are the checks that see the Z-type errors which
   flip the X-bar readout.
3. Break-even. The only circuit-level qutrit distance-3 pseudo-threshold I found is about 3x10^-4
   per time step, with flags, for `[[5,1,3]]_3` [from literature: Keppens et al. 2025]. Without
   flags it drops to about 7x10^-7, which shows how much hooks cost. SI1000 at p puts 5p on
   measurement and 2p per measurement layer on idles. I expect a distance-3 qutrit code with
   correction to beat an unencoded qutrit memory only for p around 1e-4 or below under SI1000
   [conjecture]. The folded square's hook safety for every order should help it relative to
   `[[5,1,3]]_3`, but no circuit-level number exists for it.
4. Gottesman's criterion [from literature: Gottesman 2016]. The fair test compares accepted-shot
   error to the unencoded circuit at every depth. The earlier sdim LRB-D, with one terminal check,
   failed it [from literature: Kabir et al. 2025].
5. Heralded errors [verified analytically, standard]. A distance-2 code corrects one erasure, so
   leakage or other heralded events need not be rejected [from literature: Muralidharan et al. 2017
   for qudit erasure codes].
6. Group choice for LRB. Weyl x| Q8, with 72 elements, is a unitary 2-design [verified numerically].
   Any code whose cheap logical gates cover Q8 can run a 2-design LRB without logical P. Of the
   candidates here this helps only `[[5,1,3]]_3`, and only if permutations are relabelings.
7. Recommendations [conjecture, based on the verified properties above]:
   - For a correction-based LRB that reuses the current compiler and token set, use the D=3 folded
     square `[[13,1,3]]_3`. Plan parallel ancilla readout, or the brief's sequential readout adds
     24p of idle noise per data qutrit per round.
   - For the fewest qutrits with fully transversal logical Cliffords, use the qutrit colour code
     `[[7,1,3]]_3` with flags or Steane/Knill extraction. Its circuit-level performance is unknown,
     and I found no literature on it.
   - Avoid `[[5,1,3]]_3` for full-Clifford LRB, because logical P is missing and flags are
     required. Avoid the rotated `[[9,1,3]]_3`, because it has no fold-type logical P.
   - For a middle ground that keeps O(p^3) per-shot error with much less rejection than the
     baseline, consider the D=4 folded square `[[25,1,4]]_3` with correct-one/reject-two decoding.

## 6. Gaps, negative results and items not confirmed

- I found no circuit-level threshold or pseudo-threshold for qutrit `[[7,1,3]]`, `[[9,1,3]]`,
  `[[13,1,3]]` or any qutrit colour or surface code under circuit noise. Only code-capacity and
  phenomenological qudit numbers exist (sections 3.15 to 3.17), plus the `[[5,1,3]]_q` circuit
  study (3.13, 3.14). Searches were WebSearch queries on qutrit/qudit surface and colour codes
  with circuit-level noise.
- I found no experiment with repeated syndrome extraction on a native-qutrit multi-qudit
  stabilizer code. The experiments found are 3.18, a bosonic memory, and 3.19, a state
  preparation with qubit-encoded qutrits.
- Keppens et al.'s transversal T_L claim is contradicted for q = 3 (section 3.13).
- Marks et al.'s arXiv abstract numbers differ from the numbers in their v2 source (section 3.17).
- Not fetched, so not cited as sources: Kubica-Yoshida-Pastawski "Unfolding the color code",
  Gross-Audenaert-Eisert on unitary designs, Chao-Reichardt flag qubits. The 2-design statement
  rests on my frame-potential computation. The flag statements rest on Keppens et al. 2025.
- Checks not done: the D=4 folded square, measurement-error tolerance of any candidate, decoder
  performance, and q = 5 versions of the five-qudit transversal check.
