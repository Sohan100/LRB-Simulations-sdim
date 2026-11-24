# L3 literature notes: decoders for qudit (Z_3) stabilizer codes

Agent L3, 2026-10-28. Scope: which decoders accept Z_3 syndromes natively, what they
cost, what qutrit thresholds have been reported, and what changes for a code as small
as the `[[5,1,2]]_3` folded surface code used in this repository.

Every paper cited below was fetched in this task (arXiv abstract page and, for all
key papers, the full PDF read locally; journal metadata confirmed through the DOI
landing page or the Crossref API). Fetched URLs are given per paper.

Labels used on claims:

- **[lit]** from literature (the cited paper, fetched here). "[lit, fig]" means the
  number was read off a figure (no number in the text), so it is approximate.
- **[analytic]** verified analytically (argument given here).
- **[numeric]** verified numerically (script in the appendix; run on the login node,
  about 3 s on one core).
- **[conj]** conjecture or estimate, not verified.

---

## 0. Summary

### 0.1 Which decoders handle Z_3 syndromes natively

| Decoder family | Z_3-native? | Cost | Qutrit (d=3) thresholds reported | Fit for `[[5,1,2]]_3` LRB |
|---|---|---|---|---|
| Exact maximum likelihood (lookup, forward recursion on cosets, trellis/BCJR, exact tensor contraction) | Yes, any prime d [lit: Chubb-Flammia; Sabo et al.] | Exponential in code size, linear in number of rounds. For this code: 729 hidden states per round [analytic, Sec. 3] | Not applicable (fixed small code) | Best fit: exact and cheap (Sec. 3) |
| Soft-decision RG (SDRG), Duclos-Cianci and Poulin | Yes [lit] | d^7 per 2x2 cell; O(L^2) per RG step, log L parallel depth [lit] | Code capacity 13.0% [lit]; small-cell variant about 10.9% [lit, fig] | Built for large lattices; nothing to coarse-grain at n=5 |
| Hard-decision RG (HDRG) cluster decoders: Bravyi-Haah, Anwar et al., Watson et al. | Yes: clusters are annihilated when their total charge is 0 mod d [lit] | O(L^2) (perfect syndrome), typical O(L^3), worst O(L^6) (noisy syndrome); run time independent of d [lit] | Code capacity about 11.9% [lit, fig]; phenomenological about 2.7-2.8% [lit, fig] | Trivial at n=5 (every cluster is the whole code) |
| MWM-HDRG, Hutter-Loss-Wootton (matching used to propose pairs, neutrality checked mod d) | Yes [lit] | Parallel poly(log L) (perfect syndrome); O(1) average per round (noisy, Abelian) [lit] | Code capacity 12.3%; phenomenological 3.2% [lit] | Same as HDRG |
| MWPM / belief-matching with one binary node per non-zero syndrome value (Keppens et al.) | No: the Z_3 to binary map is not linear [analytic, Sec. 2.5] | Sparse blossom, near-linear in nodes [lit] | Circuit-level, `[[5,1,3]]_3` only: pseudo-threshold 3.24e-4 with a flag qudit, 7.22e-7 without, by concatenation extrapolation [lit] | Poor: mis-models any two faults on one detector |
| Non-binary belief propagation over GF(q) (Kasai et al.; Lai-Kuo) | In principle, for GF(3) = Z_3 [analytic]; published work is GF(2^m) on qubits [lit] | O(q log q) per check per iteration with FFT; Lai-Kuo claim O(p log p) per edge for GF(p^l) [lit] | None found | Approximate on a tiny, loopy DEM; exact ML is cheaper |
| Tensor-network approximate ML (Bravyi-Suchara-Vargo; Chubb-Flammia; Chubb; Piveteau-Chubb-Renes) | The stat-mech/TN construction of Chubb-Flammia is written for Z_d [lit]; the decoders benchmarked are qubit-only [lit] | O(n chi^3) (2D), O(n log n + n chi^3) (Chubb) [lit] | None found | At n=5 the contraction is exact without truncation [analytic] |
| Union-find | No qudit paper found | Qubit: O(n alpha(n)) [lit] | None | The grow-until-neutral rule is the HDRG rule; peeling generalizes [analytic, Sec. 2.6] |

### 0.2 Reported thresholds, qutrit surface/toric codes, X sector, all non-zero powers equally likely

| Noise model | Value | Source |
|---|---|---|
| Code capacity, hashing bound | 15.9% (C_3 = 0.159 in the paper; 0.15946 by root-finding here) | [lit] Duclos-Cianci-Poulin 2013; Hutter et al. 2015 Eq. (27); [numeric] |
| Code capacity, SDRG with 2x2 cells and BP | 13.0% | [lit] Duclos-Cianci-Poulin 2013 |
| Code capacity, MWM-HDRG | 12.3% | [lit] Hutter et al. 2015 |
| Code capacity, HDRG | about 11.9% | [lit, fig] Anwar et al. 2014, Fig. 2 |
| Code capacity, small-cell SDRG | about 10.9% (crossing in 10.8-11.0%) | [lit, fig] Anwar et al. 2014, Figs. 2 and 13 |
| Phenomenological (data and measurement shifts, both rate p) | 3.2% | [lit] Hutter et al. 2015 |
| Phenomenological, HDRG | about 2.7-2.8% | [lit, fig] Watson et al. 2015, Fig. 6 |
| Circuit level, qutrit surface code | **none found** | search log, Sec. 7 |
| Circuit level, `[[5,1,3]]_3` perfect code, belief-matching + flag | 3.24e-4 (concatenation extrapolation, not a code-family threshold) | [lit] Keppens et al. 2025 |

Qubit reference values from the same papers: MWPM 10.3% and optimal about 10.9% (code
capacity); MWPM 2.9%, optimal 3.3%, SDRG 1.9% (phenomenological) [lit: Anwar et al.;
Watson et al.; Duclos-Cianci-Poulin 2014]. Circuit level (qubit surface code, Heim et
al. model): MWPM about 0.9%, exact ML 1.5-1.8% [lit].

### 0.3 What is special about a distance-2 code (main points for this study)

1. Exact ML decoding is cheap. The code-capacity decoder is a lookup over
   3^4 = 81 syndromes and 3^2 = 9 logical classes [analytic]. For the LRB circuits,
   exact ML over the full syndrome history is a forward recursion over the data Pauli
   frame modulo the stabilizer group, 3^10 / 3^4 = 729 states, about 5.3e5
   multiply-adds per check round per shot [analytic]. This is the qutrit version of
   the coset-tracking ML decoder of Heim, Svore and Hastings [lit], and is the same
   computation as forward-backward (BCJR) decoding on a time trellis, the circuit
   analog of Sabo et al.'s syndrome trellis, or an exact time-ordered contraction of
   the Chubb-Flammia tensor network [analytic].
2. Distance 2 means pure correction cannot suppress logical failure below first
   order. At code capacity with single-qutrit depolarizing noise p on each data
   qutrit, exact ML gives logical-frame failure 1.00 p and terminal X-readout failure
   0.50 p, while detection gives rejection 4.99 p and failure among accepted shots
   0.125 p^2 (0.063 p^2 for the X readout) [numeric, Sec. 3.1].
3. A hybrid decoder that corrects when the ML posterior is unambiguous and rejects
   otherwise sits between the two: rejection 2.0 p and accepted failure 2.6 p^2
   (full frame), or rejection 1.0 p and accepted failure 1.9-2.4 p^2 (X readout only)
   [numeric]. The exact ML posterior gives this confidence for free. In circuits the
   ambiguous patterns differ (measurement shifts at 5p dominate in SI1000), so the
   numbers do not transfer, but the structure should [conj].
4. None of the scalable qudit decoders (RG, HDRG, matching variants, union-find) has
   anything to work with at n = 5: their clustering and coarse-graining steps
   collapse to "the whole code is one cluster" [analytic]. Their value for this
   study is as background, and as the route to larger qutrit codes later.

---

## 1. Native Z_d decoders for topological codes

### 1.1 Duclos-Cianci and Poulin 2013, "Kitaev's Z_d-codes threshold estimates"

- G. Duclos-Cianci, D. Poulin. Phys. Rev. A 87, 062338 (2013).
  arXiv:1302.3638; DOI 10.1103/PhysRevA.87.062338.
  Fetched: https://arxiv.org/abs/1302.3638, https://arxiv.org/pdf/1302.3638,
  Crossref record for the DOI.

Extracted facts [lit]:

- Studies the Z_d toric code (Kitaev's construction over Z_d) for 2 <= d <= 6 under a
  "generalized bit-flip" channel: with probability p_phys apply one of X, X^2, ...,
  X^(d-1) uniformly. Perfect syndrome measurement (code capacity). Footnote: this is
  the qudit depolarizing channel rho -> (1-q) rho + q I/d with X and Z treated
  independently, p_phys = q(1 - 1/d).
- States why matching fails: the Z_2 code "can be decoded by a binary perfect matching
  algorithm, since every particle is its own anti-particle", and "this is not the case
  for d > 2".
- Decoder: their soft-decision renormalization-group (RG) decoder generalized to Z_d.
  The lattice is cut into 2x2 unit cells (10 qudits). Each cell is decoded exactly:
  the marginal distribution of the charge flow through its north and west boundaries
  is computed by summing over the cell's error configurations, with one plaquette left
  unmeasured so the cell charge can fluctuate (a heuristic to evade the hard
  charge-conservation constraint). A few rounds (e.g. three) of belief propagation
  between neighbouring cells precede each RG step.
- Complexity: d^7 operations per unit cell (sum over |<L0,L1>| x |<E0,E1>| x
  |<S0,S1,S2>| = d^2 d^2 d^3); one RG step costs d^c (L/k)^2 for cell size k; cells
  are independent, so the work parallelizes.
- Result for d = 3: threshold p_th = 0.130 (Fig. 4; lattice sizes L = 32, 64, 128;
  about 1e4 samples per point; fit p_dec = (p - p_th) L^(1/nu)).
- Thresholds rise monotonically with d and track a rescaled hashing bound,
  p_th(d) close to 0.81 C_d, with C_2 = 0.110, C_3 = 0.159, C_4 = 0.189. (The text
  prints p_th(2) as "0.89(6)", evidently 0.089(6), since alpha = 0.81 and
  0.81 x 0.110 = 0.089.)
- Open problem stated at the end: thresholds "in the presence of measurement error and
  detailed syndrome measurement circuits on qudits".

Bearing on this study:

- This is the reference code-capacity number for Z_3: 13.0% with an approximate
  soft-decision decoder, against a hashing bound of 15.95% [lit; numeric for the
  bound].
- The cell decoder itself is exact ML on a small patch: summing error probabilities
  over a 10-qudit cell with charge-flow boundary conditions. That primitive, applied
  to the whole 5-qutrit code with no coarse-graining, is exactly the lookup ML decoder
  of Sec. 3 [analytic].
- The noise model is X-only and i.i.d.; SI1000 circuit noise is neither [analytic].

Context papers by the same authors (abstracts fetched):

- G. Duclos-Cianci, D. Poulin, "Fast Decoders for Topological Quantum Codes",
  Phys. Rev. Lett. 104, 050504 (2010). arXiv:0911.0581; DOI
  10.1103/PhysRevLett.104.050504. Fetched https://arxiv.org/abs/0911.0581. [lit]
  Origin of the RG + BP decoder: parallel run time log L against L^6 for basic MWPM,
  and a higher depolarizing threshold than MWPM (qubits).
- G. Duclos-Cianci, D. Poulin, "Fault-Tolerant Renormalization Group Decoder for
  Abelian Topological Codes", Quant. Inf. Comp. 14 (9&10), 0721-0740 (2014).
  arXiv:1304.6100. Fetched https://arxiv.org/abs/1304.6100. [lit] A 3D (space-time)
  version for codes with Abelian anyons, so Z_d is in scope by construction, but the
  only number reported is for the qubit toric code with noisy syndromes: 1.9(4)%,
  against 2.9% for matching. Run time linear in space-time volume, log-depth in parallel.

### 1.2 Anwar, Brown, Campbell, Browne 2014, "Fast decoders for qudit topological codes"

- H. Anwar, B. J. Brown, E. T. Campbell, D. E. Browne. New J. Phys. 16, 063038 (2014).
  arXiv:1311.4895 (arXiv title: "Efficient Decoders for Qudit Topological Codes");
  DOI 10.1088/1367-2630/16/6/063038.
  Fetched: https://arxiv.org/abs/1311.4895, https://arxiv.org/pdf/1311.4895,
  https://iopscience.iop.org/article/10.1088/1367-2630/16/6/063038 (title confirmed
  there), Crossref record.

Extracted facts [lit]:

- Noise: independent X-type and Z-type channels, each applying X^j (resp. Z^j) with
  probability p/(d-1) for each j != 0. Perfect syndrome. They stress these are code
  thresholds, not fault-tolerance thresholds.
- Why not MWPM (Sec. II): charges take values in {1, ..., d-1}, so "any neutral set
  (not just a pairing)" must be considered. Lowest-weight correction then becomes
  minimum-weight matching on a hypergraph whose hyperedges are all charge-neutral
  subsets, which they note is NP-hard in general (citing general hypergraph matching,
  not a proof specific to Z_d toric codes). They did not pursue it.
- Also notes that for d > 2 a string of unequal errors produces non-trivial syndromes
  all along the string, not only at its ends, so the syndrome carries more information
  about the error path; this is part of why qudit thresholds are higher.
- HDRG decoder (generalizing Bravyi-Haah): at level l, group non-trivial syndromes into
  clusters using a metric region R_{r,s} (intersection of a Manhattan ball and a
  max-norm ball); annihilate every cluster whose total charge is 0 mod d; pass charged
  clusters to the next level. Run time O(L^2), with negligible d dependence in
  practice. Qubit threshold improved from 6.7% (original Bravyi-Haah) to 8.4%.
- HDRG thresholds rise with d and saturate near 18% because of a "syndrome
  percolation" effect: above a d-dependent error rate the non-trivial syndromes form a
  spanning cluster at the first level.
- Enhanced HDRG adds an initialization step that searches for and removes small
  neutral sub-clusters along shortest paths (overhead alpha L^2 with
  alpha = (r+s)!/(r! s!)). It reaches nearly 30% for very large d, but for d = 2 and
  d = 3 it lowers the threshold ("finding a neutral sub-cluster is very probable, and
  hence the initialization step is in fact too destructive").
- Small-cell SDRG (2x1 overlapping cells, five BP rounds per RG step): O(L^2 log L),
  d dependence O(d^4) (versus O(d^7) for Duclos-Cianci-Poulin). Qutrit crossing with
  L = 512, 1024, 2048 lies in 10.80-11.00% (Fig. 13 inset).
- Fig. 2 (no table in the text) [lit, fig]: at d = 3, HDRG about 11.9%, SDRG about 10.9%;
  at d = 2, HDRG about 8.4-8.5%, SDRG about 7.6%. Both decoders track about 0.69 of
  the hashing bound.
- They flag that depolarizing-type noise correlates X and Z errors and that a decoder
  using this correlation could do better; their decoders treat X and Z separately.

Bearing on this study:

- HDRG is the simplest decoder that is native to Z_d: it only needs "sum of charges
  mod d" tests. It is the right mental model for a qutrit union-find (Sec. 2.6).
- The two warnings carry over to SI1000: X/Z correlations (two-qutrit depolarizing
  noise produces correlated X and Z parts) are ignored by all decoders in this
  paper [lit], and the d = 3 enhanced-HDRG result shows that aggressive local
  neutralization hurts at small d [lit].

Context (abstract fetched): S. Bravyi, J. Haah, "Quantum Self-Correction in the 3D
Cubic Code Model", Phys. Rev. Lett. 111, 200501 (2013), arXiv:1112.3252 (arXiv title
"Analytic and numerical demonstration of quantum self-correction in the 3D Cubic
Code"), DOI 10.1103/PhysRevLett.111.200501, fetched https://arxiv.org/abs/1112.3252.
[lit] Introduces the HDRG decoder, which "can be implemented efficiently for any
topological stabilizer code and has a constant error threshold under random
uncorrelated errors".

### 1.3 Watson, Anwar, Browne 2015, "A fast fault-tolerant decoder for qubit and qudit surface codes"

- F. H. E. Watson, H. Anwar, D. E. Browne. Phys. Rev. A 92, 032309 (2015) (published
  title without the leading "A"). arXiv:1411.3028; DOI 10.1103/PhysRevA.92.032309.
  Fetched: https://arxiv.org/abs/1411.3028, https://arxiv.org/pdf/1411.3028 (v2),
  Crossref record.

Extracted facts [lit]:

- Planar qudit surface code, distance L, L noisy rounds then one perfect round.
  Per round: X^k with probability p/(d-1) for each k, then Z^k likewise, and each
  syndrome outcome j -> j + k with probability p/(d-1) per k (all at rate p).
- The decoder input is the 3D history of syndrome changes s'_t = s_t - s_{t-1} mod d.
  The correction found on the 3D lattice is projected onto the last time slice by
  summing exponents mod d.
- HDRG with the 3D Manhattan metric, search distance 2^l at level l; neutral clusters
  and "boundary-neutral" clusters (touching a smooth or time boundary) are fused.
  Typical run time O(L^3), worst case O(L^6), and independent of d because the
  arithmetic is mod d.
- Qubit threshold 2.15 +- 0.06%. Qudit thresholds rise with d and saturate at about
  4.2% (d = 7919), bounded by the syndrome percolation threshold (about 4.5%).
  With four initialization levels the large-d threshold reaches about 8.2-8.3%; for
  small d initialization lowers the threshold.
- d = 3 is shown only in Fig. 6 [lit, fig]: about 2.7-2.8% without initialization;
  lower (about 1.6-2.0%) with any initialization level.
- They note that with high d under isotropic depolarizing noise a decoder exploiting
  X/Z correlation "might achieve significantly higher thresholds".

Bearing on this study:

- This is the template for using noisy repeated rounds with Z_d syndromes: decode the
  difference history, not raw outcomes [lit]. The LRB circuits in this repository record raw
  ancilla outcomes per round, and logical Cliffords between rounds permute and mix the
  stabilizer generators. The correct generalization is
  D_t = raw_t - M_t raw_{t-1} (mod 3), where M_t is the Z_3-linear map that the logical
  Clifford induces on the generator syndromes. M_t is invertible because the Clifford
  maps the stabilizer group onto itself, so no information is lost [analytic]. With
  this basis every fault touches detectors of at most two consecutive rounds (Sec. 3.2)
  [analytic]. The DEM in `src/lrb/dem_simulation.py` already gives the exact detector
  shifts in the raw basis; the change of basis is a fixed linear map [analytic].
- For d = 3 the phenomenological HDRG threshold is about 2.7-2.8% [lit, fig], below
  Hutter et al.'s 3.2% [lit].

### 1.4 Hutter, Loss, Wootton 2015, "Improved HDRG decoders for qudit and non-Abelian quantum error correction"

- A. Hutter, D. Loss, J. R. Wootton. New J. Phys. 17, 035017 (2015).
  arXiv:1410.4478; DOI 10.1088/1367-2630/17/3/035017.
  Fetched: https://arxiv.org/abs/1410.4478, https://arxiv.org/pdf/1410.4478 (v2),
  Crossref record.

Extracted facts [lit]:

- Defines HDRG broadly: repeatedly merge clusters, test each new cluster for
  neutrality (sum of charges 0 mod d for D(Z_d)), remove neutral clusters with a local
  correction. Clusters never split, so the scheme also works for non-Abelian anyons.
- Two improvements. (i) "Shortcuts": after a neutral cluster is removed, distances
  between remaining clusters may route through it (Eq. 5). This raises the minimum
  number of errors causing a logical failure from Theta(L^(2/3)) to Omega(L^(1-eps)).
  (ii) Clustering by minimum-weight matching: each step pairs clusters with a
  minimum-weight (not necessarily perfect) matching, built from MWPM on a doubled
  graph so that vertices may stay unpaired at a "vertex weight".
- Weights (Eqs. 12-17): beta = -log[(p/(d-1))/(1-p)]; pairing weight
  W_jk = d_jk - log(mu_jk)/beta with path-and-type multiplicity
  mu_jk = (d-1) binom(d_jk, d^x_jk). Vertex weights interpolate (lambda = 0.3) between
  a "tag-along" overestimate and an "abstaining" underestimate. The text notes that
  the true error model also produces more general error nets, "such as a triskelion
  with an anyon at each foot", which pairings cannot represent.
- Noise: X^g with probability p/(d-1) per g != 0, per qudit; for noisy syndromes the
  outcome is shifted by a uniformly random non-zero value with probability p. Final
  round perfect.
- d = 3 results: perfect syndrome threshold 12.3% (L = 10..60); noisy syndrome 3.2%
  (L = 8..32). d = 7919: 21.9% and 6.1%. They compare: qubit MWPM 2.9%, qubit SDRG
  1.9% under the analogous noisy-syndrome model.
- Small codes: L*(p) (smallest L with p_L < p) is 3 for all p/p_c < 0.4, for both
  d = 3 and d = 7919, with perfect and with imperfect measurements (Figs. 5 and 8).
- Run time: graph decomposes into O(1)-size components at low p; parallelized run
  time poly(log L) for perfect measurement; average O(1) per round in the continuous
  Abelian case.

Bearing on this study:

- This is the published "matching on Z_d charges" decoder: matching is used only to
  propose cluster merges, and neutrality is checked mod d. It is the strongest
  scalable native-Z_3 decoder found (12.3% / 3.2%) [lit].
- At n = 5 the method has nothing to cluster, but its weight formula is a reminder that
  for Z_3 the multiplicity of error types (d-1 = 2 powers) enters the likelihood [lit].
  An exact ML decoder accounts for this automatically [analytic].

---

## 2. Matching, belief propagation, tensor networks and union-find for Z_d

### 2.1 Kasai, Hagiwara, Imai, Sakaniwa 2012, "Quantum error correction beyond the bounded distance decoding limit"

- K. Kasai, M. Hagiwara, H. Imai, K. Sakaniwa. IEEE Trans. Inf. Theory 58(2),
  1223-1230 (2012). arXiv:1007.1778; DOI 10.1109/TIT.2011.2167593.
  Fetched: https://arxiv.org/abs/1007.1778, https://arxiv.org/pdf/1007.1778,
  Crossref record (volume, issue, pages).

Extracted facts [lit]:

- Codes: CSS codes on qubits whose parity-check matrices are also sparse matrices over
  GF(2^p) with column weight 2 (quasi-cyclic construction; each GF(2^p) symbol is a
  block of p qubits). Channel: depolarizing, decoded as two independent X and Z
  problems ("the correlations between X errors and Z errors are neglected").
- Decoder: non-binary sum-product (BP) over GF(2^p); check-node convolutions computed
  by FFT; the horizontal step costs O(N q log q) per iteration for q = 2^p.
- Claims: beats the best known quantum LDPC codes of the time and the
  bounded-distance-decoding limit; error floors improve with field size.

Bearing on this study:

- The decoder is non-binary BP, but the physical systems are qubits; it is not a qudit
  decoder [lit]. The machinery transfers to qutrits
  directly: for prime d, Z_3 = GF(3), the check constraint is a Z_3-linear sum, and
  the convolution over the additive group is a 3-point DFT [analytic].

### 2.2 Non-binary BP: two follow-ups

- I. Andriyanova, D. Maurice, J.-P. Tillich, "New constructions of CSS codes obtained
  by moving to higher alphabets", arXiv:1202.3338 (2012). Fetched
  https://arxiv.org/abs/1202.3338 and the PDF. [lit] Generalizes Kasai et al.'s
  construction to F_{2^m} versions of toric codes; reports dramatically better
  iterative decoding than for the binary toric code, with distance growing as the
  square root of length. Duclos-Cianci and Poulin 2013 cite this as evidence that "the
  performance of BP for Z_d-KTC ... is greatly increased as d grows"; the paper itself
  concerns F_{2^m} alphabets on qubits, not Z_d qudits [lit, both papers read].
- C.-Y. Lai, K.-Y. Kuo, "Log-domain decoding of quantum LDPC codes over binary finite
  fields", IEEE Trans. Quantum Eng. 2, 2103615 (2021). arXiv:2104.00304; DOI
  10.1109/TQE.2021.3113936. Fetched https://arxiv.org/abs/2104.00304,
  https://arxiv.org/html/2104.00304 (sentence verified in the downloaded HTML),
  Crossref record. [lit] Shows that because a check only reveals the symplectic
  product of the error with the check (an element of the prime field), scalar messages
  suffice instead of vectors over GF(q^2). States: "The proposed algorithm could be
  extended to quantum codes over GF(p^l) for a prime p. In this case, the check-node
  complexity would reduce from O(p^{2l} log p^{2l}) to O(p log p) per edge."
  Simulations are for GF(2^l) only.

Bearing on this study: on the LRB detector error model the natural BP has one
variable per fault location (alphabet 9 for N1, 81 for N2) and one Z_3 parity check per
detector; check-to-variable messages need only be distributions over Z_3, the
p = 3, l = 1 case of Lai-Kuo's remark [analytic]. The DEM of a 5-qutrit circuit is
small and densely looped, so BP would be approximate and needs a post-processor; exact
ML costs less than one BP-OSD run here [conj].

### 2.3 Tensor-network maximum-likelihood decoders

**Bravyi, Suchara, Vargo 2014.** S. Bravyi, M. Suchara, A. Vargo, "Efficient
algorithms for maximum likelihood decoding in the surface code", Phys. Rev. A 90,
032326 (2014). arXiv:1405.4883; DOI 10.1103/PhysRevA.90.032326. Fetched
https://arxiv.org/abs/1405.4883, https://arxiv.org/pdf/1405.4883, Crossref record.

- [lit] Noiseless syndrome. Exact ML in O(n^2) for independent bit- and phase-flip
  noise, by reduction to matchgate (free-fermion) circuits. Approximate ML for
  depolarizing noise by contracting the 2D network with an MPS of bond dimension chi,
  O(n chi^3); chi = 6 is already "virtually indistinguishable" from exact ML for
  bit-flip noise. Thresholds: bit-flip 10.9-11%; depolarizing 17-18.5% with the MPS
  decoder against about 18.9% optimal and 15-16% for matching.
- The exact O(n^2) algorithm uses the Ising (Z_2) structure through a matchgate
  (free-fermion) reduction [lit]. For Z_3 the corresponding stat-mech models are
  Potts/clock-type gauge glasses (Anwar et al., App. B) [lit], and I know of no
  free-fermion reduction for them [conj]; the MPS contraction route needs no such
  structure and carries over with larger tensor indices [analytic]. The same remark
  applies to Cao et al. 2025 (Sec. 4.3).

**Chubb and Flammia 2021.** C. T. Chubb, S. T. Flammia, "Statistical mechanical models
for quantum codes with correlated noise", Ann. Inst. Henri Poincaré D 8(2), 269-321
(2021). arXiv:1809.10704; DOI 10.4171/AIHPD/105. Fetched
https://arxiv.org/abs/1809.10704, https://arxiv.org/pdf/1809.10704, Crossref record.

- [lit] Generalizes the Dennis-Kitaev-Landahl-Preskill mapping to arbitrary stabilizer
  and subsystem codes under correlated Pauli noise, including fault-tolerant
  measurement circuits ("history code"). The formalism is written for qudits: one
  classical spin c_k in Z_d per stabilizer generator S_k, with a caveat for composite d
  when generators are not of maximal order ("as is always the case for prime d").
- [lit] Shows the partition-function form gives (approximately) optimal decoding, and
  gives a linear-time TN decoder for 2D codes with locally correlated noise that
  extends the Bravyi-Suchara-Vargo MPS decoder.
- Bearing: this is the paper that makes "TN decoding of qudit codes, including
  circuit-level noise" a published construction. For the 5-qutrit LRB circuits the
  network is a chain in time of width at most 729 (Sec. 3), so it can be contracted
  exactly [analytic].

**Chubb 2021.** C. T. Chubb, "General tensor network decoding of 2D Pauli codes",
arXiv:2101.04125 (v3, 2021). Fetched https://arxiv.org/abs/2101.04125 and the PDF.

- [lit] Approximate ML for arbitrary 2D stabilizer and subsystem codes under Pauli
  noise, with a new approximate contraction scheme for irregular 2D networks
  (SweepContractor.jl). Run time O(n log n + n chi^3). Benchmarks: regular, irregular
  and subsystem surface codes and color codes under bit-flip, phase-flip and
  depolarizing noise; thresholds consistent with optimal values where known.
- **Correction to the task premise:** the paper does not itself treat qudits. Its text
  is written for "n qubits" with tensor indices over {I, X, Y, Z}; "qudit" appears only
  in the titles of two cited papers [lit, full text searched]. The TN it contracts is
  taken from Chubb-Flammia (its Ref. [57]), which is written for Z_d, so qudit support
  holds by inheritance [lit].

**Piveteau, Chubb, Renes 2024.** C. Piveteau, C. T. Chubb, J. M. Renes, "Tensor
Network Decoding Beyond 2D", PRX Quantum 5, 040303 (2024). arXiv:2310.10722; DOI
10.1103/PRXQuantum.5.040303. Fetched https://arxiv.org/abs/2310.10722, the PDF,
Crossref record.

- [lit] Extends TN decoding to 3D codes and to 2D codes with phenomenological and
  circuit-level noise; on the rotated surface code under circuit-level noise it beats
  matching. Qubit only (no qudit mention in the text). Code released.
- Bearing: shows TN contraction over a circuit-level DEM is practical for offline
  decoding when accuracy matters, which is the LRB use case [lit]; for this tiny code
  no approximation is needed [analytic].

### 2.4 Hardness context

- P. Iyer, D. Poulin, "Hardness of decoding quantum stabilizer codes",
  arXiv:1310.3235 (2013). Fetched https://arxiv.org/abs/1310.3235 and PDF. [lit]
  Degenerate (optimal) decoding of stabilizer codes is #P-complete ("it is
  #P-Complete" in the PDF abstract), harder than the NP-complete most-likely-error
  problem. Written for qubits. [analytic] Hardness is
  asymptotic in code size; it does not bear on a fixed 5-qutrit code, where the
  number of cosets is 729.
- S. Gu, L. Wang, A. Kubica, "A polynomial-time approximation scheme for
  minimum-weight decoding of topological codes", arXiv:2606.18145 (2026). Fetched
  https://arxiv.org/abs/2606.18145 and https://arxiv.org/html/2606.18145. [lit]
  Minimum-weight decoding of 2D translation-invariant codes is NP-hard in basic
  settings, yet admits a (1+eps)-approximation in L^2 (log L)^{O(1/eps)} time
  (randomized). Qubit-only: syndromes are binary and charges lie in Z_2^{2 alpha};
  circuit-level noise is claimed "analogous" but only phenomenological noise is
  verified. No Z_d statement.

### 2.5 Matching on Z_d syndromes: what exists

Findings:

1. Plain MWPM is not a Z_d decoder. For d = 3 a neutral set of charges can be a pair
   {a, -a} or a triple {1, 1, 1} or {2, 2, 2} [analytic: 1+1+1 = 0 mod 3]; minimum
   weight correction is then a hypergraph problem (Anwar et al., Sec. 1.2) [lit].
2. Matching as a subroutine with mod-d neutrality checks (Hutter et al., Sec. 1.4) is
   the published native-Z_d use of matching [lit].
3. For d = 4 with independent X, X^2, Z, Z^2 channels, MWPM works by factoring into
   two binary problems (Moussa 2016, Sec. 5.1) [lit]. This needs composite d and that
   specific channel; it does not apply to d = 3 [analytic].
4. One-hot embedding (Keppens et al. 2025, Sec. 4.1): each detector becomes d-1 binary
   nodes, one per non-zero value, and the graph is fed to PyMatching or belief
   matching [lit]. **This map is not linear** [analytic]: with
   phi(1) = (1,0), phi(2) = (0,1), phi(1+1) = phi(2) = (0,1) but
   phi(1) xor phi(1) = (0,0); and phi(1+2) = phi(0) = (0,0) but
   phi(1) xor phi(2) = (1,1). So the binary model mispredicts the detector pattern
   whenever two faults touch the same detector with values that do not cancel in both
   pictures. With the raw (non-differenced) detectors of the LRB circuits, any two
   faults that affect the same check, however far apart in time, collide on every
   later raw detector of that check; with difference detectors only faults touching
   the same check in the same or adjacent rounds collide [analytic]. Keppens et al.
   use difference detectors [lit].

### 2.6 Union-find for qudits

- Not found. Searches for "union-find" with qudit, Z_d and qutrit returned only qubit
  papers, including the original: N. Delfosse, N. H. Nickerson, "Almost-linear time
  decoding algorithm for topological codes", Quantum 5, 595 (2021), arXiv:1709.06218,
  DOI 10.22331/q-2021-12-02-595, fetched https://arxiv.org/abs/1709.06218 and
  Crossref. [lit] O(n alpha(n)) worst case; 9.9% (perfect syndrome) and 2.6% (faulty)
  for the qubit toric code; optimal up to weight (d-1)/2 errors.
- [analytic] The two steps generalize to Z_d for graph-like codes (each elementary
  fault flips at most two detectors with coefficients +-1). Growth: grow a cluster until
  its total charge is 0 mod d or it touches a boundary (this is the HDRG neutrality
  rule). Peeling: on a spanning tree of the cluster, process leaves; a leaf v with
  residual charge c_v sets the value on its tree edge to cancel c_v (possible because
  +-1 is a unit mod d) and passes the opposite charge to its parent; the root ends with
  the cluster's total charge, which is 0 or absorbed by the boundary.
- [conj] Performance would sit near HDRG; no qutrit numbers exist to check this.

---

## 3. Exact ML for the `[[5,1,2]]_3` code (analysis for this study)

### 3.1 Code capacity: ML correction versus detection versus hybrid [numeric]

Setup (stabilizers and logicals from the brief and `src/lrb/code_definitions.py`):
S_Z1 = Z0 Z1^-1 Z3^-1, S_Z2 = Z1 Z2 Z4^-1, S_X1 = X0 X1 X2^-1, S_X2 = X1^-1 X3 X4^-1;
logical X = X0 X3, logical Z = Z0 Z2. The script checks the commutation relations.
Noise: each data qutrit independently suffers one of the 8 non-identity Weyl operators
X^a Z^b with total probability p; perfect syndrome. Enumerate all 3^10 = 59049 errors
and tabulate P(syndrome, logical class). "Full" scores both logical components;
"X readout" scores only the shift of the terminal X-basis readout x0 + x3, which is
what LRB measures at the end. Script in the appendix.

| Policy | Target | Rejected fraction | Failure among accepted |
|---|---|---|---|
| Detection (accept trivial syndrome only) | full | 4.99 p | 0.125 p^2 |
| Detection | X readout | 4.99 p | 0.063 p^2 |
| Exact ML correction (accept all) | full | 0 | 1.00 p (1.04 p at p = 0.01) |
| Exact ML correction | X readout | 0 | 0.50 p (0.53 p at p = 0.01) |
| Hybrid: correct if max posterior >= 0.6-0.9, else reject | full | 2.0 p | 2.6 p^2 |
| Hybrid | X readout | 1.0 p | 1.9-2.4 p^2 |
| Bare qutrit (no code), X-basis readout | - | 0 | 0.75 p [analytic: 6 of 8 Weyl errors have a Z part] |

Weight-one structure [numeric, and checked by hand [analytic]]: of the 40 weight-one
Weyl errors, 16 lie in 8 ambiguous pairs with equal syndrome and different logical
class: X0^a ~ X3^-a and X2^a ~ X4^-a (differ by logical X^a), Z0^b ~ Z2^-b and
Z3^b ~ Z4^-b (differ by logical Z^b). All 8 errors on qutrit 1, and all mixed
X^a Z^b (a, b != 0) errors on qutrits 0, 2, 3, 4, are uniquely decodable. Only the
four Z-type pairs affect the terminal X readout, which is why the X-readout numbers
are half the full-frame numbers.

Interpretation [analytic from the table]: at distance 2, correction trades a
first-order rejection (about 5p) for a first-order logical error (0.5p on the
readout). The hybrid rejects only the ambiguous weight-one patterns (16 of 40 for the
full frame, the 8 Z-type ones for the X readout), keeps the error second order, and
rejects 2.5 to 5 times less than detection. Most-likely-error versus ML decoding for
the same enumeration is compared in Sec. 4.2. These numbers are for code capacity only; in the LRB circuits, check-round
faults, SI1000 measurement shifts at 5p, and four idle layers per round change the
ambiguity structure [conj].

### 3.2 Circuit level: exact ML by forward recursion [analytic unless marked]

- Hidden state: the data Pauli frame modulo the stabilizer group,
  Z_3^10 / S = 3^6 = 729 states, equivalently (syndrome in Z_3^4) x (logical class in
  Z_3^2). The terminal readout shift <E, X0 X3> is invariant under multiplication by
  stabilizers, so this state determines it.
- Linearity: for Pauli noise and Clifford circuits, the outcomes of round t are
  B_t f_{t-1} + (fault contribution) and the outgoing frame is A_t f_{t-1} + (fault
  contribution), with A_t, B_t fixed Z_3-linear maps (A_t includes the logical
  Clifford's action). The fault contributions of one segment (logical gates plus one
  check round, or the whole inverse block plus the final round) form one distribution
  on Z_3^6 x Z_3^4 = 3^10 points, obtained by convolving the independent DEM
  locations (or multiplying their 3-ary Fourier transforms).
- Ancilla reset shifts at the end of round t only shift round t+1 outcomes: with the
  CNOT pattern in the brief, an X^a on a Z-check ancilla (CNOT target) stays on the
  target, and on an X-check ancilla it becomes Z-type on the CNOT control, which does
  not spread. They therefore belong to round t+1's segment kernel.
- Recursion: alpha_t(f') = sum_f alpha_{t-1}(f) K_t(f' - A_t f, obs_t - B_t f).
  Cost: 729^2 = 5.3e5 multiply-adds per segment per shot; about 1.2e7 for m = 22.
  The final posterior over the 3 terminal shifts gives both the ML correction and a
  confidence for a correct-or-reject rule.
- Rounds a policy does not "use" can be marginalized (sum over their outcomes), so
  correction can be compared with the baseline at identical used-round sets.
- Equivalent formulations: Heim-Svore-Hastings coset tracking (Sec. 4.1) [lit]; exact
  contraction of the Chubb-Flammia network along time; and BCJR (forward-backward) on
  a time trellis whose sections are the 729 frame classes. At code capacity the
  analogous object is Sabo et al.'s syndrome trellis, which for this code has at most
  3^(n+k) = 3^6 = 729 vertices per section (Sec. 4.2).
- Alternative without frames: in the difference-detector basis of Sec. 1.3 every fault
  touches detectors in at most two consecutive rounds plus the single observable, so
  variable elimination over the DEM in time order needs a frontier of about 4-8
  detector trits plus 1 observable trit (at most about 3^9 states) [analytic for the
  locality; the frontier size is an estimate [conj]].
- Throughput estimate [conj]: 1e6 shots x 30 sequences x 9 depths x 15 p values in pure
  Python is too slow on a login node. Caching repeated syndrome histories (most shots
  at low p) and batched BLAS on a compute node should make full re-decoding of the
  baseline configuration feasible.

---

## 4. Optimal decoding under circuit-level noise for small codes

### 4.1 Heim, Svore, Hastings 2016, "Optimal circuit-level decoding for surface codes"

- B. Heim, K. M. Svore, M. B. Hastings. arXiv:1609.06373 (2016), no journal reference
  on arXiv. Fetched https://arxiv.org/abs/1609.06373 and the PDF.

Extracted facts [lit]:

- Exact ML for circuit-level noise ("MLCLN"): an "EC-coset" is fixed by the ideal
  syndrome, the measured syndrome and the logical class, giving 4 |S|^2 cosets for a
  qubit code with |S| = 2^{n_a}. A one-round transition matrix between cosets is
  computed from the gate-level noise model; given the measured syndrome, the update is
  a 4|S| x 4|S| matrix. Corrections are never applied, only a Pauli frame is tracked.
- Cost grows exponentially with code size; the next size up from distance 5 would need
  a vector of 2^35 entries (about 140 GB), even tracking only bit flips.
- Thresholds (qubit surface code): correlated CNOT model, MWPM about 0.9% versus an
  optimal 1.5-1.8%; independent bit/phase-flip gate noise with an 8-step circuit,
  0.28% versus 0.35%.

Bearing on this study: the 5-qutrit LRB decoder of Sec. 3.2 is this algorithm with
qubit cosets replaced by Z_3 cosets; the state is 9 x 81 = 729, which is tiny [analytic].
Heim et al. report an even larger relative advantage of exact ML over MWPM for
strongly correlated noise [lit]; their correlated model is two-qubit depolarizing
noise on CNOTs, the qubit analog of SI1000's two-qutrit depolarizing channel
[analytic].

### 4.2 Sabo, Aloshious, Brown 2021, "Trellis decoding for qudit stabilizer codes and its application to qubit topological codes"

- E. Sabo, A. B. Aloshious, K. R. Brown. arXiv:2106.08251 (v1 2021, v2 2022), no
  journal reference on arXiv. Fetched https://arxiv.org/abs/2106.08251 and the v2 PDF.

Extracted facts [lit]:

- Syndrome-trellis decoding (after Ollivier and Tillich) for any stabilizer code in any
  prime dimension; offline construction of a compact trellis for the normalizer S-perp,
  online Viterbi queries in Theta(|E|), parallelizable.
- Vertex count per section: |V_i| = p^(dim S-perp - dim S-perp_{p_i} - dim S-perp_{f_i}),
  with dim S-perp = n + k. Edge count grows exponentially with distance (numerically).
- Viterbi gives the optimal minimum-weight (most likely error) decoder, "not a maximal
  likelihood decoder"; a degenerate trellis (Pelchat and Poulin) can do degenerate ML.
- "Although qudit codes are not simulated in this work", the theory is for any prime
  dimension. Reports a 20% higher code-capacity threshold for color codes with
  boundaries than decoders that map color codes to surface codes.

Bearing on this study: for `[[5,1,2]]_3`, dim S-perp = 6, so every trellis section has
at most 3^6 = 729 vertices [analytic]. At code capacity, most-likely-error decoding
(Viterbi) picks a strictly worse class than ML for 24 of the 81 syndromes, but the
success probabilities are equal to 6 digits at p = 1e-3 and differ by 5e-5 at
p = 0.01, 1.1e-3 at p = 0.05 and 3.7e-3 at p = 0.1 [numeric; most-likely-error ties
broken by enumeration order]. The weight-one ambiguities are exact ties at first
order; ML gains only through higher-order terms of the coset sums. For LRB the target is the
marginal posterior of the terminal readout shift, a sum over all consistent fault
histories, so the sum-product (BCJR) form is the natural one [analytic]; whether the
difference matters at the large per-round error rates of SI1000 near p = 0.05 is
untested [conj].

### 4.3 Cao et al. 2025, "Exact decoding of repetition code under circuit level noise"

- H. Cao, S. Zhao, D. Feng, et al. (11 authors; last author P. Zhang).
  arXiv:2501.03582 (2025). Fetched https://arxiv.org/abs/2501.03582.
- [lit] Exact ML for the repetition code under circuit-level noise, including SI1000,
  via exact planar spin-glass partition functions in polynomial time; finds part of an
  experimental error floor came from the decoder used, and beats matching on hardware
  data.
- The method rests on "the exact solution of the spin glass partition function on
  planar graphs" [lit], an Ising (Z_2) property; I know of no Z_3 analog [conj]. For a
  code as small as ours exactness comes from brute-force dynamic programming instead
  (Sec. 3.2) [analytic].

### 4.4 Serra-Peralta, Shaw, Terhal 2026, "Decoding across transversal Clifford gates in the surface code"

- M. Serra-Peralta, M. H. Shaw, B. M. Terhal. PRX Quantum 7, 010335 (2026).
  arXiv:2505.13599; DOI 10.1103/sk5y-25b1. Fetched https://arxiv.org/abs/2505.13599,
  Crossref record.
- [lit] Qubit unrotated surface code, arbitrary sequences of transversal gates with a
  single check round between gates (the LRB pattern). Decoding must cross the gates; a
  "logical observable" MWPM decoder is given, plus windowed variants with identified
  sublinear-in-d failure mechanisms.
- Bearing: our LRB circuits are this situation for a qutrit code: transversal logical
  H and P turn X-checks into Z-checks and mix them. Decoders that assume a fixed
  matching graph per check type need this bookkeeping; the exact recursion of Sec. 3.2
  handles it through A_t and B_t [analytic].

---

## 5. Qudit codes under circuit-level noise (2020-2026), and the folded code

### 5.1 Moussa 2016, "Transversal Clifford gates on folded surface codes" (source of the code family)

- J. E. Moussa. Phys. Rev. A 94, 042316 (2016). arXiv:1603.02286; DOI
  10.1103/PhysRevA.94.042316. Fetched https://arxiv.org/abs/1603.02286,
  https://arxiv.org/pdf/1603.02286 (v3), Crossref record.

Extracted facts [lit]:

- Constructs folded versions of the square and diamond surface-code families "for
  qudits of any dimension", with transversal single-qudit logical Clifford gates
  (H and S), at the cost of 2D^2 - 2D + 1 data qudits for distance D.
- The square family has 2D^2 - 2D + 1 qudits and D^2 - D vertices and faces (X- and
  Z-type generators). Folding along a diagonal keeps the stabilizer structure; it only
  changes the layout.
- Logical H: H and H-dagger on fold qudits, a "mirror gate" on paired qudits. Logical S:
  S and S-dagger on fold qudits, CZ and CZ-dagger on paired qudits. These gates spread
  errors only within the 2-qudit transversal subsystems.
- Decoding remarks: "To simplify decoding, we can consider a folded surface code of
  qudits to be an unfolded surface code of qudit pairs. When an error spreads between
  layers, we can interpret it as a change of error type rather than a change in error
  location." Mentions MWPM for qubits and "renormalization group decoders for qudits".
  For d = 4 with independent X, X^2, Z, Z^2 channels, decoding factors into binary
  problems, with code-capacity and phenomenological thresholds equal to the qubit ones
  (about 0.1 and 0.03).

Bearing on this study:

- [analytic] D = 2 gives 2(4) - 4 + 1 = 5 qudits, 2 X-type and 2 Z-type generators:
  the `[[5,1,2]]_d` member of the square family. The repository's logical gates match
  Moussa's pattern: qudits 0, 1, 4 act as fold qudits (H or H_INV; P or P_INV) and
  (2, 3) as the mirror pair (SWAP then H; CZ). I checked the pattern, not a full
  re-derivation of Moussa's Fig. 5.
- [lit] Moussa never discusses D = 2 explicitly; the identification of this repo's code
  as Moussa's family is made in Kabir et al. (Sec. 5.3), which cites Moussa for the
  "[[5,1,2]]_3 detection surface code".
- No paper specifically on the "[[5,1,2]] folded surface code" or "smallest folded
  surface code" was found apart from Kabir et al. Related qubit work, abstract
  fetched: N. P. Breuckmann, S. Burton, "Fold-Transversal Clifford Gates for Quantum
  Codes", Quantum 8, 1372 (2024), arXiv:2202.06647, DOI 10.22331/q-2024-06-13-1372,
  fetched https://arxiv.org/abs/2202.06647 [lit]: generalizes folding to CSS codes via
  code dualities; examples are qubit codes (Bring's code).

### 5.2 Keppens, Eggerickx, Levajac, Simion, Soree 2025, "Qudit vs. qubit: simulated performance of error-correction codes in higher dimensions"

- J. Keppens, Q. Eggerickx, V. Levajac, G. Simion, B. Soree. Phys. Rev. A 112, 032435
  (2025). arXiv:2502.05992; DOI 10.1103/2w52-qd2j. Fetched
  https://arxiv.org/abs/2502.05992, https://arxiv.org/pdf/2502.05992v2, Crossref
  record (journal, volume, article number).

Extracted facts [lit]:

- Code: the `[[5,1,3]]_q` perfect code, q = 2, 3, 5 (one q = 7 point), simulated in Cirq
  with custom qudit gates; detection events are changes m_c - m_{c-1} != 0 mod q.
- Decoders: PyMatching MWPM and belief matching (BP + MWPM), adapted by creating q - 1
  detector nodes per ancilla, one per non-zero eigenvalue (Appendix A gives the
  construction). They note: q^N syndromes for N ancillas "makes the use of a simple
  lookup table impractical"; Y-type (hyperedge) errors are decomposed into 2 equally
  weighted edges, with 3 decompositions for qubits but 2 for qutrits, so MWPM does
  relatively better at q = 3; for q >= 5 the graph splits into components; BP priors
  become harder to set since error configurations scale as q^2.
- Code capacity (depolarizing on data, perfect checks): adapted BM corrects all single
  qudit errors (stays below P_M1, the probability of more than one error); adapted MWPM
  does not.
- Circuit level: idle depolarizing on data, single- and two-qudit depolarizing at
  rate p_s, ancilla bit flips before readout, last round error free. Hook errors limit
  performance; a flag qudit plus a brute-force lookup for flagged hooks fixes this.
- "Thresholds" from fitting P_L = a p_s^b and iterating it as a level-by-level
  concatenation model: q = 3: 7.22e-7 without flag, 3.24e-4 with flag (Table I; the
  figure label shows 3.29e-4); q = 2: 1.21e-6 / 4.95e-4; q = 5: 4.36e-11 / 2.32e-4.
- Outlook: "maximum likelihood decoders and decoders that incorporate flag qudit
  information".

Bearing on this study: this is the only published circuit-level qutrit code simulation
with a named decoder that I found [lit, search log]. Its decoders are binary decoders
on a one-hot graph, which is non-linear (Sec. 2.5) [analytic]; its thresholds are
pseudo-thresholds from a concatenation model, not code-family thresholds [lit]. For a
5-qudit code the authors themselves point to ML decoding as the next step [lit], and for
our 4 checks per round ML is cheap (Sec. 3) [analytic].

### 5.3 Kabir et al. 2025, "Sdim: A Qudit Stabilizer Simulator" (contains the `[[5,1,2]]_3` LRB-D study)

- A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang, B. Soree.
  arXiv:2511.12777 (v1 Nov 2025, v2 May 2026); no journal reference on arXiv.
  Fetched https://arxiv.org/abs/2511.12777, https://arxiv.org/html/2511.12777v2,
  https://arxiv.org/pdf/2511.12777v2.

Extracted facts [lit]:

- Sdim is the simulator this repository uses. Section VI reproduces Keppens et al.'s
  adapted MWPM and belief-matching decoders ("introducing new nodes for the non-binary
  syndrome flips ... reproduced here exactly") on `[[5,1,3]]_q` for q = 3, 5, 7, using a
  logical-operator success criterion instead of state-vector overlap. It reports that,
  "contrary to the original paper's claims, both MWPM and BM decoders report strictly
  lower logical error curves as q increases", and calls for "more careful analysis
  into edge weights".
- A `[[5,1,3]]_7` circuit-level threshold with belief matching, from the same power-law
  concatenation method: p_s = 1.56e-5 (Fig. 11; 1.5e6 shots per point).
- Section VII: LRB-D on the "[[5,1,2]]_3 detection surface code" (citing Moussa), with
  the same stabilizers as this repository (written 1-indexed). Only one terminal check
  round; postselection on the all-zero syndrome. The text states that full LRB with
  correction is hard here: "The decoding problem is especially delicate in the small,
  distance-two systems considered here: the syndrome can reveal that an error has
  occurred, but it generally does not provide enough information to identify a unique
  correction", and a practical qudit decoder "must distinguish between different
  nontrivial Pauli powers". Future work named: "interval-check postselection", which is
  the policy family (`unif k`, `const c`) in this repository.

Bearing on this study: it is the direct predecessor of this study. Sec. 3.1 quantifies
its "delicate" remark: at code capacity 16 of the 40 weight-one errors are ambiguous,
the other 24 are uniquely correctable [numeric].

### 5.4 Other qudit items found (brief)

- M. Iqbal, A. Lyons, C. F. B. Lo, N. Tantivasadakarn, et al. (19 authors),
  "Qutrit toric code and parafermions in trapped ions", Nat. Commun. 16, 6301 (2025).
  arXiv:2411.04185; DOI 10.1038/s41467-025-61391-z. Fetched
  https://arxiv.org/abs/2411.04185, https://arxiv.org/html/2411.04185v1, Crossref.
  [lit] Z_3 toric-code ground state on 24 qutrits (two qubits per qutrit) on
  Quantinuum H2, per-qutrit fidelity >= 96.5(3)%. No syndrome decoding: the only
  postselection discards heralded leakage into the unused two-qubit state (about 11% of
  shots); syndrome measurement is listed as future work.
- Y. Ma, M. Hanks, M. S. Kim, "Non-Pauli errors can be efficiently sampled in qudit
  surface codes", Phys. Rev. Lett. 131, 200602 (2023). arXiv:2303.16837; DOI
  10.1103/PhysRevLett.131.200602. Fetched https://arxiv.org/abs/2303.16837. [lit]
  Uses BP and percolation arguments to show residual correlations after syndrome
  measurement are sparse below threshold; not a decoder.
- J. Marks, T. Jochym-O'Connor, V. Gheorghiu, "Comparison of memory thresholds for
  planar qudit geometries", New J. Phys. 19, 113022 (2017). arXiv:1701.02335; DOI
  10.1088/1367-2630/aa939a. Fetched https://arxiv.org/abs/1701.02335 and the v2 PDF.
  [lit] "General color clustering" RG decoder for qudit 6.6.6 color codes, plus an HDRG
  for qudit surface codes (qubit 9.3%, d = 5 12.55%, plateau about 15.5% in the PDF).
  The arXiv abstract and the v2 PDF disagree (abstract: depolarizing noise, 9.75% /
  11.4% / plateau 17.1%; PDF: generalized bit flip, 5.6% / 8.0% / plateau 11.9%, qutrit
  color code 8.4%). Not used further.
- V. Nourozi, "Reinforcement learning enhanced greedy decoding for quantum stabilizer
  codes over F_q", arXiv:2506.03397 (2025). Fetched https://arxiv.org/abs/2506.03397.
  [lit] A `[[27,13,4]]_3` code decoded by a greedy syndrome decoder followed by a Deep
  Q-Network; F_3-native but unrelated to topological or circuit-level settings.

---

## 6. Answers to the questions asked

1. **Decoders handling Z_3 natively.** Exact ML in any form (lookup, coset recursion,
   trellis, exact TN) [lit + analytic]; soft-decision RG [lit]; hard-decision RG and
   the MWM-HDRG of Hutter et al. [lit]; the Chubb-Flammia stat-mech/TN construction
   [lit]; non-binary BP over GF(3) in principle [analytic; only GF(2^m) published, with
   the GF(p^l) extension stated by Lai-Kuo [lit]]. Not native: plain MWPM and
   union-find (binary parity), and the one-hot MWPM/belief-matching adaptation
   (non-linear embedding) [analytic].
2. **Complexity.** See Table 0.1. HDRG run time is independent of d [lit]; SDRG grows as
   d^7 (2x2 cells) or d^4 (small cells) [lit]; exact ML for this code is about 5e5
   operations per check round per shot [analytic].
3. **Thresholds.** See Table 0.2: code capacity 10.9-13.0% for native decoders against
   a hashing bound of 15.9%; phenomenological 2.7-3.2%; no circuit-level qutrit
   surface-code threshold has been published that I could find.
4. **Tiny distance-2 codes.** Exact ML is cheap, and is the only decoder worth
   implementing at n = 5 [analytic]. Pure correction leaves O(p) logical error;
   detection leaves O(p) rejection; the ML posterior allows a tunable hybrid that keeps
   second-order error with less rejection [numeric, code capacity; conj for circuits].
   Hutter et al.'s small-L results (L* = 3) concern distance 3 and do not cover d = 2
   [lit].

## 7. Search log and items not confirmed

Queries run (WebSearch, standard or extended): "qutrit surface code circuit-level noise
decoder threshold"; "qudit surface code threshold circuit noise simulation arXiv";
"union-find decoder qudit codes Z_d"; "Z3 toric code minimum weight matching decoder
qutrit"; "belief propagation decoder qudit stabilizer codes prime dimension GF(p)
quantum LDPC"; "qudit union-find decoder OR union find qudit surface code decoding
arXiv"; "qutrit toric code trapped ions parafermions error detection decoding";
"neural network decoder qudit surface code qutrit decoding"; "'qutrit' 'surface code'
circuit-level noise simulation decoder 2024 OR 2025 OR 2026"; "qudit stabilizer codes
detector error model decoding Z_d syndromes hypergraph matching 2025"; "'minimum
weight' decoding qudit toric code NP-hard OR 'Potts gauge glass' optimal threshold Z_3
toric code"; "'folded surface code' [[5,1,2]] OR 'smallest folded surface code'";
Ma-Hanks-Kim title; Marks-Jochym-O'Connor-Gheorghiu title.

Not found, so not cited: a union-find decoder for Z_d codes; a circuit-level threshold
for a qutrit surface code; any paper on the `[[5,1,2]]` folded code other than Moussa
(family) and Kabir et al. (LRB-D); a neural decoder for qutrit surface codes; an
optimal-threshold (Potts gauge glass) study of the Z_3 toric code that I fetched myself
(Anwar et al., App. B, state that the Nishimori-point duality gives the hashing bound
for the independent model, citing Nishimori and collaborators; I did not fetch those
papers).

Premise corrections: Chubb 2021 does not itself treat qudits (Sec. 2.3).
Duclos-Cianci-Poulin's description of Andriyanova et al. as BP for Z_d codes is loose:
that paper is about F_{2^m} alphabets on qubits (Sec. 2.2).

## Appendix: enumeration script (Sec. 3.1)

Run with `module load python/3.11`; numpy only; about 3 s. The most-likely-error
comparison of Sec. 4.2 uses the same enumeration and records, for each syndrome, the
logical class of the single most probable error (ties broken by enumeration order).

```python
import itertools, numpy as np
d = 3
zchk = np.array([[1,-1,0,-1,0],[0,1,1,0,-1]]) % d   # S_Z1, S_Z2 (Z exponents)
xchk = np.array([[1,1,-1,0,0],[0,-1,0,1,-1]]) % d   # S_X1, S_X2 (X exponents)
LX = np.array([1,0,0,1,0]); LZ = np.array([1,0,1,0,0])   # X0 X3, Z0 Z2
assert all((zchk @ LX) % d == 0) and all((xchk @ LZ) % d == 0) and (LX @ LZ) % d
V = np.array(list(itertools.product(range(d), repeat=5)))
sx, sz = (V @ zchk.T) % d, (V @ xchk.T) % d          # Z-checks see x; X-checks see z
lx, lz = (V @ LZ) % d, (V @ LX) % d                  # lz = shift of X readout x0+x3
nz = V != 0
def table(p):                                        # P[syndrome (81), logical (9)]
    P = np.zeros((81, 9))
    for i in range(243):
        for j in range(243):
            pr = np.prod(np.where(nz[i] | nz[j], p/8, 1-p))
            s = ((sx[i,0]*3 + sx[i,1])*3 + sz[j,0])*3 + sz[j,1]
            P[s, lx[i]*3 + lz[j]] += pr
    return P
for p in [1e-3, 1e-2]:
    P = table(p); Px = P.reshape(81, 3, 3).sum(1)    # Px: marginal on X-readout shift
    for name, Q in [("full", P), ("xread", Px)]:
        tot, best = Q.sum(1), Q.max(1)
        print(name, p, "ML fail/p", (1 - best.sum())/p,
              "detect reject/p", (1 - Q[0].sum())/p,
              "detect fail|acc/p^2", (1 - Q[0,0]/Q[0].sum())/p**2)
        acc = best/np.where(tot > 0, tot, 1) >= 0.9         # hybrid policy
        print("   hybrid reject/p", (1 - tot[acc].sum())/p,
              "fail|acc/p^2", (tot - best)[acc].sum()/tot[acc].sum()/p**2)
```
