# R5-citations: citation-statement audit of REPORT.md

Reviewer R5, 2026-10-28. Lens: does each [LIT] statement say what the cited work says, does every
cited work appear in Section 14, and is Section 14 free of misleading padding. Sections 13.1 and
13.2 were ignored as instructed. Nothing under `src/`, `scripts/`, `tests/` or `REPORT.md` was
edited.

## Method

- Read `REPORT.md` in full and extracted every bracketed citation, including those that wrap
  across lines (50 author-year citations plus 13 bare `[LIT]` tags).
- Matched each cited work to `bibliography_raw.csv` and to the literature notes and citation
  checks (`notes/lit-L1` to `lit-L8`, `*-citecheck.md`), plus `notes/A2-optimal-postselection-critique.md`,
  `notes/H1` and `notes/H2` where the report points to them.
- Fetched the arXiv abstract page of all 14 required papers (table below). Where a statement goes
  beyond the abstract I also read full text: Ceasura (ar5iv), Kwiatkowski (PDF text, extracted
  in an isolated scratch directory), Vigneau (arXiv HTML), Hrmo and Ringbauer (ar5iv). I also
  fetched Berthusen 2026, Mayer 2024, Mai 2026, Keppens 2025 and Kumar et al. 2026 to test claims
  of absence or uniqueness.
- `notes/report-review/R5-citations/check_sec14.py` compares Section 14 with the CSV by key,
  arXiv ID, DOI and year. It lists Section 14 keys never cited in the body and in-text years
  that differ from Section 14. Run it from the `qutrit-512` folder after
  `module load python/3.11`. It is read-only and takes under 1 s.

## Spot checks of the 14 required papers

| Paper | Metadata vs Section 14 | Report's use | Verdict |
|---|---|---|---|
| Ceasura et al. 2022, 2212.05488 | match (v1 11 Dec 2022, no journal-ref) | Sec. 5.1 item 1; Sec. 5.6 item 1 | **overstated** (R5-04) |
| Ziyad, Blume-Kohout, Rudinger 2025, 2512.08893 | match (v1 9 Dec 2025, v3 19 Feb 2026) | Sec. 5.1, 5.6: memory when data are not returned to the code space | supported. The abstract says "if (and only if) the physical qubits are not necessarily returned to the code subspace after every round". Composability is in the body (notes/lit-L2 Sec. 6.2). |
| Kwiatkowski et al. 2025, 2509.16887 | match (v1 21 Sep 2025) | Sec. 5.6: decoded cycles approximately Markovian | **out of scope as applied** (R5-05) |
| Combes et al. 2017, 1702.03688 | match (v1 only, no journal-ref) | Sec. 1: recovery or none, syndrome traced out | supported by full text (notes/lit-L1 Sec. 1); not credited for offline frame recovery (R5-14) |
| Harper and Flammia 2019, 1806.02359 | match (PRL 122, 080504) | Sec. 1: postselected d = 2 LRB; no complete theory | supported (abstract: [4,2,2], 5.8(2)% to 0.60(3)%; quote in TeX line 278) |
| Lacroix et al. 2025, 2412.14256 | match (Nature 645, 614-619; 215 authors) | Sec. 1: "the only hardware LRB with a decoder" | description right, **uniqueness claim wrong** (R5-01) |
| English, Williamson, Bartlett 2025, 2410.07598 | match (PRL 135, 120603; DOI 10.1103/nh49-52y2) | Sec. 4.2 theorem | **misattributed** (R5-02) |
| Heim, Svore, Hastings 2016, 1609.06373 | match (v1 20 Sep 2016) | Sec. 2.2: coset tracking | supported. The abstract says the ML decoder "tracks the success probabilities for all possible corrections over time". |
| Goss et al. 2022, 2206.07216 | match (Nat. Commun. 13, 7481) | Sec. 7.2: CZ† 97.3% in 580 ns | supported. 97.3(1)% is in the abstract; 580 ns is in the full text (notes/lit-L6). |
| Vora et al., 2406.18807 | Section 14 has 2024 (v1); body says 2026 | Sec. 7.2: qutrit MCM with feed-forward, 40 ns | numbers supported ("within 40 ns", "conditional qutrit logic protocol"); year and uniqueness wording (R5-17) |
| Graydon et al. 2021, 2108.04200 | match (v1 9 Aug 2021) | Sec. 5.1: Clifford 2-design for prime d | supported by the body (notes/lit-L1 Sec. 4.3); the abstract states only the composite-d negative; co-citation weak (R5-13) |
| Geher et al. 2025, 2408.00758 | match (npj QI 11, 39) | Sec. 7.1: reset can be dropped | **selective** (R5-06) |
| Vigneau et al. 2025, 2503.12869 | match (PRX Quantum 6, 040369) | Sec. 7.1: IQM detection without ancilla reset | fact supported (App. F); **omits App. G leakage result** (R5-06); also R5-19 |
| Moussa 2016, 1603.02286 | match (PRA 94, 042316) | Sec. 8: `[[13,1,3]]_3` is the distance-3 member | supported. The square family has 2D^2 - 2D + 1 qudits, so D = 3 gives 13 (notes/lit-L3 Sec. 5.1). |

`check_sec14.py` result: 54 Section 14 entries. 52 match a CSV row with identical DOI, arXiv ID
and year. Two are missing from the CSV (R5-07). Two keys are never cited in the body: Keppens2025
and IQMclientDocs (R5-18). Two in-text years differ from Section 14: Sabo (2021 vs 2024) and Vora
(2026 vs 2024).

## Findings

Severity follows the brief. **Critical** means a wrong statement of fact or conclusion. **Major**
means an unsupported or mislabelled claim, a misleading statement or a missing requested item.
**Minor** means clarity.

### R5-01 (critical): "only hardware LRB with a decoder" is false

- **Quote (Sec. 1).** "The only hardware LRB with a decoder found is on a distance-3 colour
  code, with one QEC cycle between transversal gates, a single recovery Clifford and no
  postselection [Lacroix et al. 2025]."
- **Problem.** The study's own literature search found other hardware LRB experiments with
  decoding. The sentence frames the report's novelty, and it is wrong.
- **Evidence.**
  - notes/lit-L1 Sec. 0 item 3 lists three: "With decoding and no postselection: Google color
    code [Lacroix2025], Quantinuum Steane code [Mayer2024], Quantinuum C4-Helix
    [Berthusen2026]."
  - Berthusen et al. 2026 (arXiv:2609.03194, fetched) benchmarked the full two-qubit Clifford
    group "under active error correction", "without relying on postselection", at 2.8(+1.0/-1.6)e-4
    per logical Clifford. Per notes/lit-L1 Sec. 2.9, which used the TeX, there is a syndrome
    round after each gate, the circuit is decoded over the full volume with a Stim DEM, and a
    "forced-gap" decoder-confidence postselection variant is included. That variant is the
    closest precedent for the report's soft-MLD rule.
  - Mayer et al. 2024 (arXiv:2404.08616, fetched) ran RB of the transversal logical two-qubit
    gate, with the final readout decoded (notes/lit-L1 Sec. 2.6).
  - Mai et al. 2026 (arXiv:2601.21616, fetched) ran RB "with interleaved erasure detections",
    postselected, with a check after every Clifford. Acceptance and conditional decays were
    fitted separately (notes/lit-L1 Sec. 2.10). This is the closest hardware analogue of LRB-D
    `unif 1` and is not mentioned either.
  - All three are confirmed rows in `bibliography_raw.csv` (lit-L1) but are absent from
    Section 14.
- **Fix.** Replace the sentence with: "Hardware LRB with decoding and no postselection has been
  reported on a d = 3 colour code with one QEC cycle between transversal gates [Lacroix et al.
  2025], on the Steane code with a decoded terminal readout [Mayer et al. 2024], and on the
  [[20,2,6]] C4-Helix code with a syndrome round after every logical Clifford, which also tested
  decoder-confidence postselection [Berthusen et al. 2026]. A postselected LRB with a check after
  every Clifford, with acceptance and conditional decays fitted separately, was run on a bosonic
  erasure qubit [Mai et al. 2026]." Add the three works to Section 14.

### R5-02 (major): the Sec. 4.2 theorem is attributed to sources that do not state it

- **Quote.** "**Theorem (notes/A2; [LIT: Chow 1970; Forney 1968; English, Williamson and
  Bartlett 2025]).** Among all rules, randomized ones included, that map a shot's full detector
  history to "reject" or "accept and apply correction c", the rule that maximizes the accepted
  P(correct) at a fixed acceptance accepts the histories with the largest `max_L P(L | D)` and
  applies the argmax."
- **Problem.** None of the three cited works states this fixed-acceptance, randomized,
  multi-class theorem.
  - **EWB 2025.** Theorem 1 is optimal "for any abort set with equivalent cardinality r", a
    fixed number of error configurations, not a fixed probability (notes/lit-L4 Sec. 3.2). The
    abstract (fetched) describes only a decoder-free heuristic, conditional and abort thresholds,
    and four phases.
  - **Forney 1968.** The content was never read. The citation checker says "Content was not
    read; the notes ... rely on HSS10 and Merhav". Forney's rule assumes equiprobable messages,
    and LRB violates that (P(L = 0) is about 0.99).
  - **Chow 1970.** Only the abstract was read. It "does not state the rule's form".
  - The A2 critic already flagged all of this as C3 "Needs qualification"
    (notes/A2-optimal-postselection-critique.md lines 137-160), with the fix: "cite a
    multi-class statement; say that Chow's and Forney's rules are the cost (Lagrangian) forms
    and that Theorem 1 adds the acceptance-constrained randomisation; mark Forney/NP as via
    Merhav". The report did not apply it.
  - Under the report's own [LIT] definition ("fetched and re-fetched and confirmed"), Forney
    1968 does not qualify for content.
- **Evidence.** notes/lit-L4 Sec. 0.1 item 2 locates the exact statement in Franc, Prusa and
  Voracek 2023, Theorem 6 ("bounded-coverage" problem, with randomization at the threshold).
  FPV23 is a confirmed CSV row but is not in Section 14.
- **Fix.** Label the theorem "[VA, notes/A2]". Then add: "it is the acceptance-constrained form of
  Chow's reject rule [LIT: Chow 1970, via Combes and Ferrie 2015; Franc, Prusa and Voracek 2023,
  Thm 6]; English et al. 2025 prove the fixed-cardinality analogue for QEC (Thm 1) [LIT];
  Forney's erasure rule (via Merhav 2008) is the equal-prior special case." Add FPV23 (and CF15
  or Merhav08 if cited) to Section 14.

### R5-03 (major): Wallman 2018 and Merkel 2021 give two exponentials for trace-decreasing maps

- **Quote (Sec. 5.1).** "...where each `Lambda_t` is a fixed (possibly gate-dependent, possibly
  trace-decreasing) map that does not depend on history. Under that assumption, with Clifford
  twirling (...), the decay is a single exponential up to small gate-dependent corrections [LIT:
  Wallman 2018; Merkel et al. 2021]."
- **Problem.** For trace-decreasing maps, both cited results give two decay constants, not one.
  - Wallman 2018, Theorem 6: `A p^m + B t^m + eps_m` (notes/lit-L2 Sec. 2.1).
  - Merkel et al. 2021: `S_m = A t^m + B p^m + O(delta^{m/2})` (notes/lit-L2 Sec. 2.3).
  - Getting a single exponential for the report's statistic takes two more steps that are the
    study's own derivation [VA]: the traceless Weyl observable removes `B t^m`, and dividing by
    acceptance (about `C t^m`) gives `(p/t)^m`.
  - notes/lit-L1 Sec. 2.12 says it outright: "For LRB-D the single exponential is therefore an
    empirical model, not a consequence of the theorem."
  - The report's own Sec. 5.2 adds a further condition: the mean of per-sequence ratios "is
    exactly exponential only when acceptance is state independent".
- **Fix.** "...the unnormalized survival is `A p^m + B t^m` up to small gate-dependent
  corrections [LIT: Wallman 2018; Merkel et al. 2021]; for the traceless Weyl observable the
  `t^m` term vanishes, and the accepted polarization decays as `(p/t)^m` [VA, notes/A4,
  notes/lit-L2 Sec. 2.1]."

### R5-04 (major): Ceasura et al. 2022 is cited as matching a mechanism it does not describe

- **Quotes.** Sec. 5.1: "the code block's syndrome register, an out-of-codespace error
  persisting between checks [LIT: Ceasura et al. 2022; Ziyad, Blume-Kohout and Rudinger 2025];".
  Sec. 5.6 item 1: "Detectable errors persist in the syndrome register until a check, and pairs
  inside unchecked intervals become logical errors. This matches the theory of syndrome
  registers as hidden memory [LIT: Ceasura et al. 2022]". The exec summary carries the same
  claim as "[VN, VA, LIT]".
- **Problem.** The two mechanisms differ in ways that matter.
  - In Ceasura et al. the hidden memory is "the syndrome qubits used in error correction ...
    strongly coupled to the logical qubits via a decoder" (abstract, fetched). The construction
    assumes failed syndrome reset and a surjective decoder (notes/lit-L1 Sec. 3.5; notes/lit-L2
    Sec. 6.1). The output oscillates, and the authors call the example "somewhat contrived"
    (ar5iv, fetched).
  - LRB-D has no decoder and shows monotone acceleration, not oscillation.
  - notes/lit-L2 Sec. 6.1 labels the identification "[CONJ] for the identification" and the
    mapping "[VA]". The report upgrades both to [LIT].
- **Fix.** Use Ziyad et al. for the necessary-and-sufficient condition [LIT]. Say that Ceasura
  et al. showed a stored syndrome coupled through a decoder can make LRB non-exponential, in
  contrived models [LIT]. Mark the mapping of LRB-D's unchecked intervals onto the "failed reset"
  regime as [VA/CONJ, notes/lit-L2 Sec. 6.1].

### R5-05 (major): Kwiatkowski et al. 2025 is applied outside its stated scope

- **Quote (Sec. 5.6).** "Decoded repeated QEC cycles are expected to be approximately Markovian
  with exponentially suppressed deviations [LIT: Kwiatkowski et al. 2025]. That suppression needs
  more distance than this code has **[CONJ]**."
- **Problem.** The paper's own conditions exclude this study's decoders and code, and the [CONJ]
  points at the wrong gap. From the full text (PDF extracted):
  - It requires "(i) Pauli-stochastic noise ...; (ii) a particular property of the
    syndrome-extraction procedure (see Def. 11) or the ability to randomize the syndrome
    subsystem prior to the QEC cycle (see Def. 13); and (iii) that the single-cycle logical
    entanglement fidelity ... is sufficiently high (f1 >= 1 - 1/64)". Def. 11 is the syndrome
    decoding symmetry `D(E(s) + θ') = s + D(θ')`.
  - The decoding symmetry holds, for example, for "a minimum-weight decoder for a classical
    linear code with odd code distance".
  - It excludes multi-round decoders: schemes that "use the measurement outcomes from previous
    QEC cycles in order to determine correction operations" are "not compatible" with its
    definition and are left to future work. The study's MLD and lookahead decoders are exactly
    such decoders.
  - "Stabilizer codes with qudits" and "logical Clifford randomized-benchmarking experiments
    with gate-dependent errors" are listed as future directions.
  - The binding gaps are therefore multi-round decoding, qutrits, the symmetry broken by d = 2
    ties (which syndrome randomization could restore at any distance), and f1 at the baseline p.
    Distance is not the gap. notes/lit-L2 Sec. 6.3 already lists these.
- **Fix.** "For qubit stabilizer codes with Pauli-stochastic noise, per-cycle decoding with a
  syndrome-decoding symmetry (or syndrome randomization) and f1 >= 1 - 1/64, consecutive QEC
  cycles admit a logical Markov model whose deviations decay exponentially in the number of
  cycles [LIT: Kwiatkowski et al. 2025]. Multi-round decoders like ours, qudits and gate-dependent
  LRB are outside that theorem [LIT]; whether it extends here is open [CONJ]."

### R5-06 (major): the reset recommendation cites Geher and Vigneau without their caveats

- **Quotes.** Exec summary: "No feedback, no decoder latency budget, and no ancilla reset are
  needed **[VA, VN, LIT]**." Sec. 7.1: "The ancilla reset can be dropped. With QND readout an
  un-reset ancilla returns `m + s` (mod 3), so detectors become differences of consecutive
  outcomes [notes/L6; LIT: Geher et al. 2025 on reset choices; Vigneau et al. 2025 ran IQM error
  detection without ancilla reset]."
- **Problem.** Reset and leakage are explicitly requested items. Both cited works, and McEwen
  2021 (cited elsewhere in the report), qualify the recommendation in ways the report omits.
  - **Vigneau 2025.** The same IQM no-reset experiment reports "a steady increase of leakage
    fraction in ancilla qubits with the cycle index", most probably from readout-induced
    transitions (App. G, arXiv HTML, fetched). notes/lit-L6 line 974 records the same: no-reset
    "also lets ancilla leakage persist across rounds [Vigneau2025, McEwen2021]".
  - **McEwen 2021.** Multi-level reset of the measure qubits is what removes leakage-induced
    correlated detection events lasting more than 10 rounds. It speeds measure-qubit leakage
    decay 40-fold (notes/lit-L7 C1).
  - **Geher 2025** (abstract, fetched):
    - resetting gives no benefit for memory experiments;
    - for logical operations, unconditional reset halves the duration and doubles the tolerable
      measurement error;
    - no-reset wins only when reset takes about 100 ns or more and p is about 3e-3 or more;
    - with no reset, a misread becomes "a correlated pair of errors" (notes/lit-L5 Sec. 4.17).
  - Every simulated LRB circuit includes the reset layer, so nothing in the report tests the
    no-reset DEM. The revised notes/H2 Sec. 1.4 now has a statevector check [VN, A6] that an
    un-reset ancilla returns `k + s(E)`, which could back the `m + s` statement as [VN]. That
    check is noiseless, though, and has no leakage caveat.
  - The pointer "notes/L6" does not exist; it should be notes/lit-L6 (lines 113, 967), with the
    derivation also in notes/lit-L5 Sec. 4.17.
- **Fix.**
  - Exec summary: "No feedback or decoder latency budget is needed [VA, VN, LIT]. Ancilla reset
    is not needed for the frame logic [VA]; whether to drop it is a hardware trade-off."
  - Sec. 7.1: "By Geher et al.'s criterion (reset of about 100 ns or more, p of about 3e-3 or
    more) no-reset is favoured at today's qutrit error rates [LIT]. But without reset, ancilla
    leakage accumulates, as seen on IQM hardware [LIT: Vigneau 2025, App. G], and the
    multi-level reset that removes leakage-induced correlations [LIT: McEwen 2021] is lost.
    No-reset turns each misread into a correlated detector pair; that DEM was not simulated
    [CONJ]."

### R5-07 (major): two Section 14 entries skipped the citation check, and the "two marked entries" are not marked

- **Quotes.** Sec. 14: "Every entry below was fetched by a literature agent and independently
  re-fetched by a citation checker (...), or, for the two marked entries, fetched by the lead."
  Sec. 7.2: "Trapped-ion qutrit two-qudit gate | 98.7%, readout about 99.7%, all-to-all
  [Ringbauer et al. 2022; Hrmo et al. 2023, via notes/H2] | p about 0.014". The [LIT]
  definition in the label table says "a second agent re-fetched and confirmed".
- **Problem.**
  - Ringbauer2022 and Hrmo2023 have no row in `bibliography_raw.csv` (`check_sec14.py`:
    "MISSING FROM CSV"). No entry in Section 14 carries a mark.
  - The source is the H2 agent's fetch (notes/H2 Sec. 9, [R22] and [H23], lines 966-968 of the
    revised file), not "the lead".
  - The table row also merges results that were not shown together:
    - Hrmo's 98.7(2)% is a SPAM-corrected average gate fidelity from repeated applications on
      **two** ions with global beams. The single-application state fidelity is 0.978(12). The
      authors say the repeated sequences are "sensitive to the presence of non-Markovian noise"
      (ar5iv, fetched).
    - Arbitrary-pair addressing in Ringbauer 2022 is shown for qubit entangling gates (0.993 in
      a 3-ion string, 0.983 in a 10-ion string), and their MS-based qudit gates are 97.5(2)%
      (Cex) and 93.8(2)% (Cinc).
    - The revised notes/H2 itself now qualifies the gate as "LS 98.7(2)%, 2 ions, global beams"
      (Sec. 2.8 table) and says "I found no register of 9 qutrits with addressed native qutrit
      entangling gates [CONJ, absence]" (Sec. 2.1).
  - The numbers themselves check out. 98.7(2)% for d = 3 is in Hrmo. "readout error of ∼99.7%"
    is in Ringbauer and means 99.7% fidelity (optimum 3e-3).
- **Fix.**
  - Mark the two entries in Section 14, or add CSV rows. Suggested status: "verified by
    R5-citations from ar5iv full text, 2026-10-28".
  - Rewrite the Section 14 preamble to say which agent fetched them.
  - In 7.2, write: "native light-shift gate 98.7(2)% (SPAM-corrected, 2 ions, global beams)
    [Hrmo 2023]; qutrit readout about 99.7% [Ringbauer 2022]; addressed arbitrary-pair gates
    shown for qubits and for MS-based qudit gates at 93.8 to 97.5% [Ringbauer 2022]."

### R5-08 (major): several [LIT] statements have no verified source in Section 14

The report defines [LIT] as "from a paper that an agent fetched and a second agent re-fetched
and confirmed (Section 14)". These statements fail that definition.

| Report text | Actual source (per notes) | Status |
|---|---|---|
| "current FPGA loops take 0.2 to 0.6 µs [notes/H1, notes/H2, LIT]" (Sec. 7.1) | revised notes/H2 Sec. 5.1: QICK [St21], QubiC 2.0 [X23], Zurich Instruments and Qblox vendor pages, Cao et al. 2026; H2 notes "vendor pages are vendor claims" and now quotes "0.2 to 0.55 µs" (Sec. 0) | none in CSV or Section 14; upper bound differs from the revised note |
| "readout errors biased toward decay, not uniform shifts [LIT]" (Sec. 7.2) | Blok2021, Liu2023, Wang2025 (notes/lit-L6 Sec. 0 item 4) | uncited; Wang2025 not in Section 14 |
| "leakage to \|3> [LIT]" (Sec. 7.2) | Marques2023 h-LRU acts on \|3>; McEwen2021 (notes/lit-L7 Sec. C) | uncited; Marques2023 not in Section 14 |
| "square-lattice Crystal QPUs (Garnet 20, Emerald 54) ... [Abdurakhimov et al. 2024]" (Sec. 7.5) | Abdurakhimov 2024 is the 20-qubit Garnet paper; Emerald 54 is from AWSEmerald2025 (notes/lit-L6 line 778) | misattributed; AWSEmerald2025 not in Section 14 |
| "mid-circuit measurement and one-bit classically controlled PRX as an experimental feature;" (Sec. 7.5) | AWSGarnetDyn2025, AWSEmerald2025, IQMclientDocs | uncited at the bullet |
| "others 82 to 89%" (Sec. 7.2, two-qutrit gates) | Blok2021 (CSUM 0.889), Morvan2021 (0.82-0.85), Luo2023 (89.3%) | uncited; Luo2023 not in Section 14 |
| "to 0.5-1.5 µs" (Sec. 7.2, readout) | Goss2026 500 ns, Vora 500-750 ns, Hyyppa2024 1.5 µs; L6 checker flag 7 found no duration in Blok2021 or Goss2024 and did not check Hyyppa | uncited; Hyyppa2024 not in Section 14 |
| "Sparse-check LRB-D is strongly non-exponential ... [VN, VA, LIT]" (exec summary) | no literature citation for this claim anywhere in Sec. 5.3 | [LIT] has no referent |

- **Fix.** Cite the named sources and add them to Section 14, or relabel. Vendor specifications
  could be labelled "[vendor claim]". The exec summary [LIT] should either cite Ziyad 2025 or be
  dropped.

### R5-09 (major): a qudit flag scheme does not exist "only in a thesis"

- **Quote (Sec. 8 table).** "`[[7,1,3]]_3` colour code | Runner-up. Transversal H_L and P_L, but
  needs flag qutrits; a qudit flag scheme exists only in a thesis [LIT: Durso-Sabina 2022]."
- **Problem.** The claim is false, and the counterexample is already in Section 14 as Keppens2025.
  - Keppens et al. 2025, PRA 112, 032435 (abstract fetched): "After introducing a flag qudit to
    protect the code from hook errors". This is a circuit-level simulation of the
    `[[5,1,3]]_q` code for q = 2, 3, 5, with thresholds of order 1e-4.
  - notes/lit-L3 Sec. 5.2 says "a flag qudit plus a brute-force lookup for flagged hooks fixes
    this". notes/lit-L8 line 531 says "One flag qudit, reused across checks, catches hook
    errors". Their q = 3 pseudo-threshold is 3.24e-4 with the flag and 7.22e-7 without.
- **Fix.** "needs flag qutrits; flag qudits have been simulated at circuit level for
  `[[5,1,3]]_q`, q = 2, 3, 5 [Keppens et al. 2025], and a general prime-d flag construction is in
  a thesis [Durso-Sabina 2022]."

### R5-10 (major): a directly relevant 2026 paper flagged by the L2 checker is missing

- **Quote (Sec. 9).** "More checks always help | **Refuted in this noise model [VN]** |
  Sequential checks cost 118 p per round." Sec. 5.1 and 5.6 discuss memory in postselected LRB
  without it.
- **Problem.** The L2 citation checker listed Kumar, Foxman and Ding 2026 and said "This bears
  directly on the LRB-D check policies" (notes/lit-L2-citecheck lines 155-166). The report cites
  it nowhere, and it is not in the CSV.
- **Evidence.** arXiv:2609.12073, abstract fetched. It is an extended version of an IEEE QCE 2026
  paper. It gives:
  - an exact transition matrix for postselected QED with "rejected outcomes [in] an absorbing
    state";
  - "a leading-order obstruction to a stationary logical-only accepted-map description" from
    "accepted leakage injection from QED-check imperfections", the same mechanism as the
    report's hidden syndrome in the accepted branch;
  - a "first-cycle transient", compare the `unif 1` first window, 5.1e-5 against about 2.6e-5
    later, in `fits_baseline.csv.gz`;
  - a proof that "less frequent checks improve efficiency to leading order", compare Sec. 4.6
    and Sec. 9 here.
- **Fix.** Cite it in Sec. 5.1 (definition and mechanism) and in Sec. 4.6 or 9 ("measuring
  less"). State how the result relates: their statement is about executed checks, while the
  report's "dominated" result is about choosing which outcomes of executed rounds to use. Add it
  to Section 14 after a citation check.

### R5-11 (minor): McEwen 2021 and Miao 2023 cited as "erasure-qubit practice"; leakage persistence mislabelled [CONJ]

- **Quotes (Sec. 7.4).** "treat a |3> outcome as an erasure flag, as in erasure-qubit practice
  [LIT: Wu et al. 2022; Kubica et al. 2023; Levine et al. 2024; McEwen et al. 2021; Miao et al.
  2023]." and "Leakage is persistent memory, the most non-Markovian error there is; detection
  without leakage reduction lets it poison many rounds **[CONJ]**."
- **Problem.**
  - McEwen and Miao are about leakage removal, not heralding. notes/lit-L7 C2 says "Leakage
    removal turns leakage into ordinary errors that the decoder sees. It is not a herald [lit]."
  - Conversely, the "poison many rounds" clause is [LIT]. McEwen shows correlated detection
    events over more than 10 rounds without reset. Miao shows, for a d = 21 bit-flip code with
    no reset, that "even small amounts of injected leakage population less than 1% cause the
    logical error probability to rise above 40%"
    (notes/lit-L7 Sec. 0 item 6, C1, C2). Only "the most non-Markovian error there is" is
    opinion.
- **Fix.** Keep Wu, Kubica and Levine on the erasure-flag sentence. Move McEwen and Miao to the
  persistence sentence and relabel it [LIT], dropping the superlative. Consider citing the
  Marques 2023 h-LRU, which acts on |3> (notes/lit-L7 C4), next to IQM's f0g1 LRU.

### R5-12 (minor): Mavadia 2018 and Ball 2016 do not show that real devices drift

- **Quote (Sec. 5.5).** "Real devices drift and have two-level fluctuators [LIT: Mavadia et al.
  2018; Ball et al. 2016], so the noise strength varies between shots."
- **Problem.** Neither paper documents device drift.
  - Mavadia et al. used injected detuning noise on a trapped ion (notes/lit-L2 Sec. 3.3).
  - Ball et al. is theory. It supports the quasi-static ("DC") modelling choice, not the
    existence of drift or two-level fluctuators on devices (notes/lit-L2 Sec. 3.1).
- **Fix.** Cite Ball 2016 for the quasi-static model. For device evidence, cite works already in
  Section 14: Hrmo 2023 ("slow frequency noise", non-Markovian), Vigneau 2025 (suspected
  "device parameter drift"), and Goss 2022 (drift between batches, notes/lit-L6).

### R5-13 (minor): Jafarzadeh 2020 is a weak source for the prime-d 2-design property

- **Quote (Sec. 5.1).** "(the qutrit Clifford group is a unitary 2-design, prime d [LIT: Graydon
  et al. 2021; Jafarzadeh et al. 2020])"
- **Problem.** Jafarzadeh et al. state, without restricting to prime d, that the Clifford group
  "evidently forms a U2D". Graydon et al. cite them as an example of that assumption, which
  fails for composite d (notes/lit-L1 Sec. 4.2-4.3). The primary source, Gross, Audenaert and
  Eisert 2007, is a confirmed CSV row in L1 and L2 but is not cited.
- **Fix.** Cite "[LIT: Gross, Audenaert and Eisert 2007; Graydon et al. 2021]". If Jafarzadeh is
  kept, cite it only for qudit RB.

### R5-14 (minor): Combes 2017 is not credited for offline Pauli-frame recovery in LRB

- **Quotes.** Exec summary: "correction is a Pauli-frame relabelling of the terminal outcome,
  computed offline ... [VA, VN, LIT]". Sec. 7.1 cites "[LIT: Knill 2005; Riesebos et al. 2017;
  DiVincenzo and Aliferis 2007]". Sec. 5.4: "a single compiled recovery Clifford instead of m
  mirrored inverses (standard RB) **[CONJ]**".
- **Problem.**
  - The LRB-specific precedent is Combes et al. (TeX, notes/lit-L1 Sec. 1): "we need not
    actually perform the recovery operation, as it is always a Pauli operation that we can
    commute forward", and "No particular recovery map needs to be chosen at experiment time".
  - Combes, Lacroix, Berthusen and Mai all use a single recovery Clifford, so the protocol
    itself is [LIT]. Only its effect on inverse-block memory is [CONJ].
  - Riesebos 2017 was read as an abstract only (notes/lit-L5 Sec. 4.3).
- **Fix.** Add Combes et al. 2017 to the Sec. 7.1 [LIT] list. In Sec. 5.4, write "a single
  recovery Clifford, as in standard LRB [LIT: Combes 2017; Lacroix 2025]; that it removes the
  inverse-block memory is [CONJ]."

### R5-15 (minor): the Sec. 8 decoder citations are not "windowed or matching-style"

- **Quote.** "windowed or matching-style decoders become relevant there [LIT: Duclos-Cianci and
  Poulin 2013; Anwar et al. 2014; Watson et al. 2015; Hutter et al. 2015]."
- **Problem.** Three of the four are renormalization-group or hard-decision RG (HDRG) cluster
  decoders. Hutter et al. use matching only to propose pairs inside HDRG. None is a windowed
  decoder (notes/lit-L3 Sec. 0.1).
- **Fix.** "renormalization-group and cluster (HDRG) decoders native to Z_d, including
  matching-assisted HDRG [LIT: ...]; windowed decoding for transversal-gate circuits [LIT:
  Serra-Peralta et al. 2026, notes/lit-L3 Sec. 4.4]." Add that work to Section 14 if cited.

### R5-16 (minor): the Sabo analogy, the in-text year and "same 729-coset construction"

- **Quote (Sec. 2.2).** "It is the qutrit, LRB version of coset tracking [LIT: Heim, Svore and
  Hastings 2016] and of trellis decoding [LIT: Sabo et al. 2021]. The same 729-coset
  construction was found independently by the literature agent [notes/lit-L3]."
- **Problem.**
  - Sabo et al. build a spatial syndrome trellis at code capacity, and their Viterbi decoder is
    "not a maximal likelihood decoder" (notes/lit-L3 Sec. 4.2). The forward filter is a
    sum-product (BCJR) recursion on a time trellis.
  - Section 14 dates Sabo 2024 (IEEE TQE), but the body says 2021.
  - The report's filter has 243 states (Z_3^4 x Z_3, Sec. 2.2), while L3's construction has 729
    (Z_3^4 x Z_3^2). The two are related by marginalizing the irrelevant logical component, so
    "same" is inaccurate.
- **Fix.** "...and a time-trellis (BCJR) analogue of the syndrome-trellis decoding of Sabo et al.
  [2024]. Literature agent L3 independently derived the full 729-coset version; the filter here
  tracks only the 243 states that affect the X readout."

### R5-17 (minor): Vora year, and "shown once" stated as fact

- **Quote (Sec. 7.2).** "Qutrit mid-circuit readout with feed-forward | shown once, 40 ns FPGA
  discriminator [Vora et al. 2026]".
- **Problem.**
  - Section 14 and the L6 checker correct the year to 2024 (v1 27 Jun 2024; v5 13 Sep 2026).
  - "Shown once" is an absence claim. notes/lit-L6 Sec. 0 item 1 adds "Absence in my searches is
    not proof of absence", and that hedge was dropped.
- **Fix.** "[Vora et al. 2024, v5 2026]" and "the only demonstration found [CONJ: search-based]".

### R5-18 (minor): Section 14 keys never cited in the body

- **Problem.** Keppens2025 and IQMclientDocs are in Section 14 but never cited
  (`check_sec14.py`). This is not misleading padding: both should be cited.
  - Keppens: see R5-09. Sec. 9's "Matching on Z_3 through binary nodes also mis-models two faults
    on one detector [notes/lit-L3]" is a critique of Keppens et al.'s one-hot construction. It
    should name them and note that they use difference detectors, where the collision is
    limited to adjacent rounds (notes/lit-L3 Sec. 2.5).
  - IQMclientDocs: see R5-08, Sec. 7.5 bullet.
- **Fix.** Add the two in-text citations.

### R5-19 (minor): IQM "Not found: multi-state discrimination" needs qualifying

- **Quote (Sec. 7.5).** "Not found: native qutrit gates in the circuit API, multi-state
  discrimination, or ternary feedback."
- **Problem.** IQM's own experiments used three-state classifiers.
  - Vigneau 2025, App. G (fetched): >99% for |0>, >94% for |1>, >93% for |2>.
  - Hyyppa 2024 used three-state readout on an IQM test device (notes/lit-L6 Sec. 3.5, Sec. 4.4).
  - What was not found is user-facing documentation (notes/lit-L6 Sec. 6).
- **Fix.** "Not documented for users: ... multi-state discrimination (IQM's own experiments used
  three-state classifiers [Vigneau 2025])...".

### R5-20 (minor): the definition cites process tensors but lists a non-memory carrier

- **Quote (Sec. 5.1).** "Non-Markovian means history dependence through hidden memory, in the
  process-tensor sense [LIT: Pollock et al. 2018; Figueroa-Romero et al. 2021, 2022]. Here the
  memory carriers are: ... 3. the check schedule, when the policy makes it depend on m."
- **Problem.** An m-dependent check schedule is not a hidden memory in the process-tensor sense.
  It is a time-inhomogeneous protocol. The RB literature classes it as "non-Markovian or
  time-dependent": Helsen 2022 and Proctor 2017 (notes/lit-L2 Sec. 2.2, 2.4), and notes/lit-L2
  Sec. 7 says revival tests are "not [valid] for `const c`".
- **Fix.** Split it out: "3'. Separately, `const c` and `unif k` change the protocol with m (time
  dependence, not memory)."

### R5-21 (minor): bibliography bookkeeping

- **Problems.**
  - Sec. 13 says "Of 214 citations". These are 214 CSV rows for 182 distinct works (several
    papers were checked by two to four agents).
  - The Section 14 preamble promises "corrected fields are shown"; that holds for all
    CSV-backed entries checked by the script.
- **Fix.** Write "214 citation records (182 distinct works)".

## Statements checked and found supported

- **Sec. 1.**
  - Combes 2017: recovery or none, syndrome traced out (TeX, via notes/lit-L1 Sec. 1).
  - Harper and Flammia 2019: postselected [[4,2,2]]; "There is at present no complete theory for
    fitting an RB decay curve with post-selection".
  - Marques 2022: acceptance decay reported separately.
  - Kabir 2025: terminal-postselection LRB-D in v2 Sec. VII.
- **Sec. 2.2.** Heim, Svore and Hastings 2016 coset tracking (abstract).
- **Sec. 5.1 and 5.6.** Pollock 2018 and Figueroa-Romero 2021/2022 as definition sources. Ziyad
  2025 iff condition (abstract).
- **Sec. 7.1.**
  - Knill 2005 and DiVincenzo and Aliferis 2007 on Pauli frames (full text via notes/lit-L5).
  - Riesebos 2017 (abstract only).
  - Vigneau 2025 no-reset rule (App. F).
- **Sec. 7.2.**
  - Goss 2022: 97.3(1)%, 580 ns.
  - Morvan 2021 and Liu 2023: 2.38e-3 to 5.4e-3 per Clifford.
  - Chen 2023: 96.9% three-state readout at 140 ns.
  - Blok 2021 and Liu 2023: T1(2 to 1) about 35 µs; 101 µs.
  - Magnard 2018 and McEwen 2021: 0.2% in under 500 ns; about 250 ns.
  - Vora: 40 ns, conditional qutrit protocol.
  - Hrmo 2023 and Ringbauer 2022 numbers (see R5-07 for context).
- **Sec. 7.4.** Wu 2022, Kubica 2023 and Levine 2024 for erasure flags.
- **Sec. 7.5.**
  - Abdurakhimov 2024: Garnet, 20 to 40 ns CZ.
  - IQM pulse docs: `prx_12`, `LRU_F0G1`.
  - Goss 2026: IQM-fabricated chip, 15 qutrits, QubiC, 93%, terminal readout only.
- **Sec. 8.**
  - Moussa 2016: square family, 2D^2 - 2D + 1 qudits, so D = 3 gives 13.
  - Breuckmann and Burton 2024: fold-transversal gates, qubit examples only, which is adequate
    as general support.
  - Durso-Sabina 2022 exists as described.
- **Section 14 metadata.** DOIs, arXiv IDs and years of all 52 CSV-backed entries match the CSV,
  and Sec. 13's 188 confirmed plus 26 corrected matches it.
