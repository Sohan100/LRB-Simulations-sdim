# L6: Superconducting transmon-qutrit hardware for a 9-qutrit [[5,1,2]]_3 LRB experiment

Agent L6, 2026-10-28. Topic: what transmon qutrits (and IQM hardware in particular)
can do today, in numbers, and what that means for running this repository's LRB-D
circuits on hardware and for replacing postselection by correction.

## 0. Conventions and method

Labels used on every claim:

- **[lit:Key]**: from a paper or documentation page I fetched in this task. The key
  refers to the per-source sections below, which list the fetched URLs.
- **[analytic]**: verified analytically. The argument is given in place.
- **[numeric]**: verified numerically. The scripts are in `notes/L6-scripts/` (run with
  `module load python/3.11`; each runs in seconds).
- **[code]**: read from this worktree's source.
- **[conjecture]**: my inference, not checked.

Definitions. Process (entanglement) fidelity F_pro and average gate fidelity F_avg
satisfy F_avg = (D F_pro + 1)/(D + 1) in dimension D. So the process infidelity
e = 1 - F_pro and the average infidelity r = 1 - F_avg satisfy e = (D+1)/D * r. For a
single qutrit e = 4r/3; for two qutrits e = 10r/9 [analytic; standard identity]. An
SI1000 Weyl-depolarizing location with total probability q has F_pro = 1 - q. This is
because the identity Kraus weight is 1 - q and no non-identity Weyl operator has
nonzero trace. I also checked it numerically for q = 0.05 [analytic + numeric,
`L6_si1000_map.py`]. "Equivalent SI1000 p" below means the p for which the SI1000
location has the same process infidelity as the hardware operation after Weyl
twirling.

Method. I fetched arXiv abstract pages (title, full author list, DOI and journal
reference, read from the page's citation meta tags) and the full texts (arXiv HTML,
ar5iv, or PDF with pypdf in a worktree venv `.venv-pdf/`). I also fetched publisher or
repository landing pages where the published version differs, IQM documentation pages,
and two AWS announcements. I grepped the numbers below from the full texts, not from
abstracts or search snippets. A number that is in a figure only is marked "figure only,
not extracted".

## 1. Summary for this study

1. **No transmon-qutrit stabilizer-code experiment (QED or QEC) was found.** The closest
   hardware precedents are:
   - repeated mid-circuit erasure checks on transmon qutrit levels, with postselection
     [lit:LiuBJ2026];
   - a qutrit mid-circuit measurement with real-time conditional qutrit-X feed-forward
     [lit:Vora2026];
   - repeated [[4,2,2]] error-detection cycles on an IQM qubit star QPU, with
     postselection and no ancilla reset [lit:Vigneau2025];
   - GKP qutrit/ququart QEC beyond break-even in a cavity [lit:Brock2025].

   Absence in my searches is not proof of absence.
2. **Two-qutrit gates are the bottleneck.** The best documented transmon two-qutrit gate
   is a qutrit CZ† with 97.3(1)% process fidelity in 580 ns (CZ: 95.2(3)% in 783 ns)
   [lit:Goss2022]. Other demonstrated gates sit at about 82–89%: CSUM 0.889 (QPT) in
   about 1.5 μs [lit:Blok2021]; CSUM 0.82–0.85 (cycle benchmarking) [lit:Morvan2021];
   tunable-coupler CPhase 89.3% (QPT including SPAM) [lit:Luo2023]. In SI1000 terms the
   two-qutrit location has p = e, so current hardware is at **p ≈ 0.027 at best, and
   0.05–0.18 typically** [analytic mapping].
3. **Single-qutrit Clifford error is 2.4×10⁻³ to 5.4×10⁻³ per Clifford.** Sources: RB r,
   [lit:Morvan2021, Liu2023]; published average process infidelity 3.8×10⁻³
   [lit:Morvan2021]. With SI1000's 0.1p rule this corresponds to p ≈ 0.03–0.07
   [numeric]. Hardware therefore has a single-qutrit to two-qutrit error ratio of about
   1:4 to 1:10, close to SI1000's 1:10 assumption [analytic arithmetic]. The diagonal
   gates Z and P are frame updates (virtual Z) on transmon qutrits, at essentially zero
   cost [lit:Blok2021, Goss2026; analytic: a Clifford that fixes Z must be diagonal
   because Z has non-degenerate spectrum]. SI1000 charges them 0.1p.
4. **Three-state single-shot readout reaches 93–97%.** Durations range from 140 ns (with
   shelving through |3>) [lit:Chen2023] to 0.5–1.5 μs on most devices [lit:Blok2021,
   Goss2024, Goss2026, Vora2026, Hyyppa2024]. As an SI1000 measurement flip (5p) this is
   **p ≈ 0.006–0.014** [numeric]. Errors are dominated by decay during readout,
   P(i-1|i), so they are biased rather than uniform shifts [lit:Blok2021, Liu2023,
   Wang2025].
5. **The |2> level has typical T1(2→1) ≈ 30–40 μs and T2echo(1-2) ≈ 30 μs** on
   multi-qutrit chips. The best single qudit has T1(|2>) = 101 μs [lit:Blok2021,
   Goss2022, Goss2024, Liu2023]. Decay is sequential, with Γ(i→i-1) ≈ i·Γ(1→0), and
   direct 2→0 decay is about 100× slower [lit:Peterer2015].
6. **Unconditional all-level reset exists.** All-microwave f0g1 reset of all three
   transmon levels reaches 0.2% residual excitation in under 500 ns [lit:Magnard2018].
   Flux-swap multi-level reset brings |1>, |2>, |3> to |0> in 250 ns, down to a 0.2%
   readout floor [lit:McEwen2021]. A qubit-style active reset (binary measurement plus a
   conditional X01) does not remove |2> population [lit:Haupt2025; analytic].
7. **Qutrit mid-circuit measurement with real-time feed-forward has been shown once.** It
   ran on Berkeley/LBNL QubiC electronics: a 40 ns FPGA neural-network discriminator,
   500–750 ns readouts, and conditional X^m with branch fidelities of 84–99% (raw)
   [lit:Vora2026]. A ternary readout can also be built from two binary mid-circuit
   measurements with an X01 pulse in between [lit:Kanazawa2023 (IBM); Brock2025
   (logical GKP qutrit)].
8. **What IQM hardware supports:**
   - Connectivity: square-lattice "Crystal" QPUs (Garnet 20 qubits, Emerald 54) built
     from flux-tunable transmons and tunable couplers, with 20–40 ns CZ gates
     [lit:Abdurakhimov2024, AWSEmerald2025]. There is also a 6-qubit star QPU with a
     computational resonator and MOVE gates [lit:Renger2025].
   - Mid-circuit measurement, feed-forward, and active reset: offered as an experimental
     capability on Garnet and Emerald. Feed-forward is a single-bit-controlled PRX
     (`cc_prx`), conditional ops are limited to a "local group", and verbatim
     compilation is required [lit:AWSGarnetDyn2025, AWSEmerald2025, IQMclientDocs].
   - Higher levels: the circuit API has no qutrit gate. The pulse library contains
     `prx_12` (used for shelved readout) and an f0g1 leakage-reduction unit
     [lit:IQMpulseDocs].
   - Qutrit operation of an **IQM-fabricated** 20-transmon square-lattice chip has been
     shown (a 15-qutrit chain). It used non-IQM control electronics, 500 ns qutrit
     readout with 93% average fidelity, and no mid-circuit measurement [lit:Goss2026].
   - I found no IQM-authored qutrit computation paper.
9. **Connectivity of the [[5,1,2]]_3 circuits.** The 12 check-round couplings embed
   exactly as a 3×3 square patch. The logical H and P gates also need a direct (2,3)
   two-qutrit gate, which makes the interaction graph non-bipartite, so it cannot be a
   subgraph of any square lattice. In all 72 embeddings of the check graph into a 5×5
   grid, data qutrits 2 and 3 sit at grid distance 4 [numeric + analytic]. The repo
   already contains an 11-qutrit split-ancilla variant that uses relay ancillas for the
   "folded" geometry [code].
10. **Correction for these Clifford LRB circuits needs no real-time feedback.** A
    decoder's output is a Weyl (Pauli-frame) correction of the terminal readout and can
    be applied in software [analytic]. Ancilla reset can also be dropped: with a QND
    readout, an un-reset ancilla returns m + s (mod 3), so detectors become differences
    of consecutive outcomes [analytic]. The hardware requirements reduce to qutrit gates,
    ternary mid-circuit readout, and leakage control.
11. **SI1000 does not match transmon-qutrit hardware in its ratios.**
    - Gate locations correspond to p ≈ 0.03–0.07.
    - Readout, reset, and idle-during-readout correspond to p ≈ 0.001–0.014.
    - Idle during gate layers is absent from SI1000 but costs about 5–11% process
      infidelity per data qutrit per check round at current coherence [numeric under a
      stated Lindblad model].

    How this shifts the optimal check policy is open [conjecture]. A duration-aware noise
    model is needed before claiming a hardware advantage for correction.

## 2. Numbers at a glance

Abbreviations: CB = cycle benchmarking; QPT = process tomography; RO = readout; "n.r." =
not reported in the text I read.

| Source | Platform | 1-qutrit error | 2-qutrit gate | 3-level RO fidelity, duration | T1(2→1), T2(1–2) | MCM / feedback |
|---|---|---|---|---|---|---|
| Blok2021 | 5 of 8 fixed-freq. transmons, ring, resonator couplers | subspace RB f01 = 0.9997, f12 = 0.9994; gates ≤ 30 ns | CSUM (cross-Kerr) 0.889 QPT, ~1.5 μs; CR conditional-π 125 ns (2-level subspace only) | avg 0.96 ± 0.02 (|2>: 0.92–0.96), n.r. | 34.8 ± 4 μs; T2* 14 ± 5, T2echo 28 ± 5 μs | terminal only; teleportation heralded by postselection |
| Morvan2021 | same chip | qutrit RB r = 2.38–3.94×10⁻³ (arXiv); avg process infid. 3.8×10⁻³ (published) | CSUM CB process fidelity 0.82 (arXiv), 0.85 (published abstract) | as Blok2021 | as Blok2021 | none |
| Goss2022 | fixed-freq. transmons, 8-ring, two chips | n.r. | CZ† 97.3(1)% in 580 ns; CZ 95.2(3)% in 783 ns (CB) | n.r. | 28–63 μs; T2e 28–61, T2R 10–18 μs | none |
| Goss2024 | 3 of 8-ring | qutrit RB "F_RB" 0.989–0.992 isolated, 0.969–0.975 simultaneous | CZ† (as Goss2022) | P(i|i) 0.986–0.994 / 0.943–0.979 / 0.943–0.974 | 33–36 μs; T2e 32–33 μs | none |
| Luo2023 | 2 fixed-freq. + tunable coupler | single-qutrit gate 40 ns | CPhase 89.3 ± 1.9% (QPT incl. SPAM); coupler pulses 2×119 ns | assignment-matrix correction | 13.9 / 8.9 μs during gate | none |
| Roy2023 | 2 transmons, SQUID parametric coupler | H process fid. 97.1–99.0% | CPhase via |22>↔|31> (94 ns), |21>↔|30> (56 ns); no gate fidelity | no paramp; confusion-matrix inversion | 21.7 / 3.9 μs; T2R 2.0–2.4 μs | none |
| Liu2023 | single transmon qudit, EJ/EC ≈ 88 | SU(3) Clifford r = 5.4×10⁻³; π/2 pulses 35–85 ns | — | 4-state: 99.1 / 94.5 / 94.5 / 91.1%, n.r. | T1(|2>) 101 μs, T1(|3>) 73 μs; T2R(1-2) 37 μs | none |
| Nguyen2024 | coupled transmon qudits | native gate F = 0.99909 (d = 3) | two-photon interaction; qutrit Bell state 97.7% raw | n.r. | n.r. | none |
| Goss2026 | **IQM-fabricated** 20 flux-tunable transmons + 30 couplers, 15-qutrit chain | SU(3) gate 180 ns; RB in figure only | native tunable cross-Kerr from coupler flux pulses; no gate fidelity | avg 93%, 500 ns | figure only | none ("only terminal readout") |
| Chen2023 | Chalmers fixed-freq. transmons | — | — | **96.9% in 140 ns** (shelving with |3>, two tones, NN) | — | — |
| Vora2026 | AQT chip, QubiC FPGA | — | conditional X^m feed-forward | 88.0–99.8% per state, 500–750 ns; 40 ns inference | — | **yes, real-time** |
| LiuBJ2026 | transmon qutrit + ancilla qubit | Clifford ~10⁻⁴ (g–f qubit) | FWM erasure-SWAP 200 ns | ancilla RO 1.4 μs | 2·T1^ef ≈ 26 μs | repeated MCM, postselected |
| Vigneau2025 | **IQM** 6-qubit star QPU (qubits) | 99.89–99.96% | qubit-resonator CZ 98.0–99.0% | 3-state classifier >99 / >94 / >93% | — | repeated MCM, no reset, postselected; cycle 2.00 μs |

## 3. Per-source notes

### 3.1 Berkeley qutrit-processor line

#### Blok2021
- **Citation.** M. S. Blok, V. V. Ramasesh, T. Schuster, K. O'Brien, J. M. Kreikebaum,
  D. Dahlen, A. Morvan, B. Yoshida, N. Y. Yao, I. Siddiqi, "Quantum Information
  Scrambling on a Superconducting Qutrit Processor", Phys. Rev. X 11, 021010 (2021);
  arXiv:2003.03307 (arXiv title uses "in a"); DOI 10.1103/PhysRevX.11.021010.
- **Fetched.** https://arxiv.org/abs/2003.03307, https://arxiv.org/html/2003.03307
- **Device.**
  - Eight fixed-frequency transmons in a nearest-neighbour ring, coupled by resonators;
    five are used. Each has its own readout resonator, with a Purcell filter and a TWPA.
  - ω01/2π = 5.43–5.78 GHz and ω12/2π = 5.16–5.51 GHz (Table 2), so the anharmonicity
    is about -265 to -270 MHz [lit; arithmetic analytic].
  - EJ/EC ≈ 73 was chosen to suppress charge dispersion. It reduces the dispersion of
    |2> to 12 kHz and of |1> to 250 Hz [lit].
- **Coherence.**
  - T1(1→0) = 43–70 μs (mean 56.0 ± 10); T1(2→1) = 29–39 μs (mean 34.8 ± 4).
  - T2*: 39 ± 21 μs (0–1), 14 ± 5 μs (1–2), 6–26 μs (0–2).
  - T2echo: 61.2 ± 11 μs (0–1), 28 ± 5 μs (1–2), 26–45 μs (0–2).
  - The paper states that |2>→|1> decay is roughly twice as fast as 1→0 (bosonic
    enhancement), and that charge-noise dephasing is roughly an order of magnitude
    faster per level [lit].
- **Gates.**
  - Single-qutrit gates take ≤ 30 ns, with virtual phase gates. Subspace RB gives
    f01 = 0.9997 ± 0.0001 and f12 = 0.9994 ± 0.0001, and per-Clifford subspace errors of
    2.7–7.5×10⁻⁴ [lit].
  - Two-qutrit (a): a cross-resonance conditional-π gate (125 ns) that acts only in a
    two-level subspace; two applications prepare an EPR pair with F = 0.98 ± 0.002.
  - Two-qutrit (b): CSUM from the always-on cross-Kerr interaction (|α_ij| of about
    138–748 kHz) interleaved with single-qutrit pulses. It takes about 1.5 μs and has QPT
    fidelity 0.889, "primarily limited by decoherence"; the scrambler has fidelity 0.875
    [lit].
- **Readout.** Single-shot three-level dispersive readout, averaging 0.96 ± 0.02. Per
  state: |0> 0.97–0.99, |1> 0.94–0.97, |2> 0.92–0.96. The paper says it is "largely
  limited by decay during readout". The readout duration was not stated in the text I
  read [lit].
- **Measurement use.** The EPR measurement is the creation circuit run in reverse,
  followed by a terminal computational-basis measurement. Success is **heralded by
  postselecting** on Q2Q3 = |00>, giving F_avg = 0.568 ± 0.001 [lit]. There is no
  mid-circuit measurement or feedback.
- **Bearing on this study.**
  - It sets the 2020-era baseline for multi-qutrit chips.
  - The always-on cross-Kerr interaction makes idle errors coherent and
    neighbour-dependent, which the brief's SI1000 Weyl noise does not represent.
  - Postselection on terminal outcomes is the hardware analogue of LRB-D [conjecture on
    relevance].

#### Morvan2021
- **Citation.** A. Morvan, V. V. Ramasesh, M. S. Blok, J. M. Kreikebaum, K. O'Brien,
  L. Chen, B. K. Mitchell, R. K. Naik, D. I. Santiago, I. Siddiqi, "Qutrit Randomized
  Benchmarking", Phys. Rev. Lett. 126, 210504 (2021); arXiv:2008.09134;
  DOI 10.1103/PhysRevLett.126.210504.
- **Fetched.** https://arxiv.org/abs/2008.09134, https://arxiv.org/html/2008.09134
  (v1), https://dspace.mit.edu/handle/1721.1/143809 (published abstract).
- **Facts.**
  - The processor is the same five-qutrit device as Blok2021.
  - Qutrit Clifford RB fully depolarizes the noise: populations decay to 1/3 and the
    fitted observable is the qutrit Pauli ⟨Z⟩, whose imaginary part stays at 0.
  - arXiv Table 1, qutrit RB r = (d-1)/d·(1-p): 3.94, 2.38, 2.50, 3.75, 3.34 ×10⁻³.
  - Qubit-like RB: {0,1} 0.27–0.55×10⁻³, {1,2} 0.38–1.48×10⁻³, {0,2} 1.0–2.5×10⁻³.
  - Leakage out of the driven two-level subspace: 0.22–8×10⁻³. In {0,1} RB it comes from
    off-resonant driving of |2>; in the other subspaces it comes from T1 decay.
  - Qutrit Hadamard (interleaved RB): r_H = 4.6 ± 0.3×10⁻³ at 128 ns.
  - One qutrit became unusable during qubit-like RB because of a two-level fluctuator.
  - Cycle benchmarking: identity cycle 0.98. For CSUM, the arXiv v1 states process
    fidelity 0.82, "corresponds to a gate fidelity of 0.85". The **published** abstract
    (via MIT DSpace) states "average single-qutrit process infidelity of 3.8×10⁻³" and
    "two-qutrit process fidelity of 0.85". The two versions word this differently; I
    report both [lit].
- **Bearing on this study.**
  - The observable matches. The brief's per-circuit fidelity |Σ_k ω^k P(k)| is the
    modulus of the expectation of the clock operator Σ_k ω^k |k><k| on the logical
    outcome, the same Z-type observable Morvan fits [analytic + lit].
  - Qutrit Clifford twirling makes the effective noise depolarizing on average, which
    supports Weyl-noise modelling for the twirled average but not for individual
    sequences [lit; conjecture on the per-sequence part].

#### Goss2022
- **Citation.** N. Goss, A. Morvan, B. Marinelli, B. K. Mitchell, L. B. Nguyen,
  R. K. Naik, L. Chen, C. Jünger, J. M. Kreikebaum, D. I. Santiago, J. J. Wallman,
  I. Siddiqi, "High-fidelity qutrit entangling gates for superconducting circuits",
  Nat. Commun. 13, 7481 (2022); arXiv:2206.07216 (v3); DOI 10.1038/s41467-022-34851-z.
- **Fetched.** https://arxiv.org/abs/2206.07216, https://arxiv.org/html/2206.07216,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9722686 (journal metadata).
- **Mechanism.** A microwave-activated dynamic cross-Kerr interaction (differential AC
  Stark shift) between two fixed-frequency, statically coupled transmons. Flat-top
  cosine pulses limit leakage. There are two rounds of cross-Kerr evolution with
  {1,2}-subspace π pulses in between, followed by virtual-Z corrections.
- **Results** [lit]:
  - CB Weyl error rates are 2.7(1)% for CZ† and 4.8(3)% for CZ. The process fidelities
    are 97.3(1)% and 95.2(3)%, and the gate times 580 ns and 783 ns.
  - The worst-case Weyl channel error is below 8%.
  - XEB agrees with CB within one standard error.
  - Noise is dominated by single-qutrit phase errors: the CB purity estimate of 97.3(9)%
    exceeds the speckle-purity limit of 96.1(3)%.
  - The best previous transmon two-qutrit gate had 11.1% error.
  - The CZ and CZ† gates were done on two different chips.
- **Device** [lit]:
  - Anharmonicity -260 to -275 MHz.
  - T1^01 = 45–125 μs, T1^12 = 28–63 μs.
  - T2e^12 = 28–61 μs, T2R^12 = 10–18 μs.
  - Drive Ω ≈ 11–13 MHz.
- **Bearing on this study.**
  - This is the best two-qutrit gate in the fetched literature. In SI1000 units it is
    p ≈ 0.027 for the two-qutrit location [analytic mapping].
  - The gate is 15–40× slower than IQM's 20–40 ns qubit CZ [lit:Abdurakhimov2024].
  - Its residual noise is mostly coherent Z-type phase error. Randomized compiling turns
    this into stochastic Weyl noise that is probably biased toward Z^b components
    [conjecture]. A decoder prior biased toward Z^b could help [conjecture].

#### Goss2024
- **Citation.** N. Goss, S. Ferracin, A. Hashim, A. Carignan-Dugas, J. M. Kreikebaum,
  R. K. Naik, D. I. Santiago, I. Siddiqi, "Extending the computational reach of a
  superconducting qutrit processor", npj Quantum Information 10 (2024), published
  2024-10-14; DOI 10.1038/s41534-024-00892-z; arXiv:2305.16507.
- **Fetched.** https://arxiv.org/abs/2305.16507, https://arxiv.org/html/2305.16507,
  https://www.osti.gov/biblio/2479284 (journal metadata; the article number is not
  shown there, so I do not state it).
- **Facts** [lit]:
  - Randomized compiling with qutrit Weyl twirls tailors Markovian noise into stochastic
    Weyl–Heisenberg channels. The number of random compilations needed does not grow
    with dimension (Hoeffding bound).
  - RC is combined with noiseless output extrapolation (NOX) and readout calibration.
  - The device is an 8-qutrit ring of fixed-frequency transmons with resonator
    couplers; three qutrits are used.
  - T1^12 = 33–36 μs; T2e^12 = 32–33 μs.
  - Readout P(0|0) = 0.986–0.994, P(1|1) = 0.943–0.979, P(2|2) = 0.943–0.974.
  - Qutrit RB "F_RB" is 0.989–0.992 isolated and 0.969–0.975 simultaneous. The table
    does not say whether this is the decay parameter or the fidelity.
  - Static cross-Kerr between neighbours is 0.1–0.6 MHz.
  - The three-qutrit GHZ state has F = 0.818 raw and 0.951 with RC + NOX. The authors
    call this "the first-ever error mitigation experiment performed on qutrits".
- **Bearing on this study.**
  - RC is the hardware route to make the brief's Weyl-noise and DEM assumption hold on
    average [lit + conjecture].
  - The large drop from isolated to simultaneous RB shows that crosstalk is a leading
    error source on this platform; SI1000 has no crosstalk [lit + conjecture].

#### Goss2026 (qutrits on an IQM-fabricated chip)
- **Citation.** N. Goss, N. Suri, B. Marinelli, L. Chen, A. Hashim, S. Anand, A. Morvan,
  R. K. Naik, E. Rrapaj, D. I. Santiago, W. de Jong, N. Y. Yao, J. E. Moore, I. Siddiqi,
  "A Qutrit Time Crystal Stabilized with Native Chiral Interactions", arXiv:2605.14293
  (2026), preprint.
- **Fetched.** https://arxiv.org/abs/2605.14293, https://arxiv.org/html/2605.14293
- **Device** [lit]:
  - "a 2D grid of 20 superconducting flux-tunable transmon qubits with interactions
    mediated by 30 flux tunable couplers designed, fabricated, and packaged by IQM".
  - Housed at UC Berkeley and controlled with LBNL's open-source QubiC system, not IQM's
    control stack. TWPAs are from Silent Waves.
  - Sub-chains of the lattice are used as qutrit chains; the largest is 15 qutrits.
- **Single-qutrit gates** [lit]:
  - X_π/2 pulses in the 01 and 12 subspaces plus virtual Z(θ01, θ12) =
    diag(1, e^{iθ01}, e^{iθ12}).
  - An arbitrary SU(3) uses 3 rounds of subspace rotations and takes 180 ns; a qutrit
    Clifford uses 6 native pulses.
  - Simultaneous RB degrades with chain length, attributed mainly to microwave crosstalk
    and partly to residual ZZ. The RB numbers are in figures only, not extracted.
- **Couplers** [lit]:
  - Idle cross-Kerr nulling leaves a mean residual α22 of 24.7 kHz over all 30
    couplers.
  - Fast coupler flux pulses produce a large tunable cross-Kerr interaction. α22 grows
    much faster with coupler flux than the other terms, because the qutrits are
    negatively anharmonic and the coupler sits below them.
  - The phases were measured with a generalized JAZZ sequence. No two-qutrit gate
    fidelity is reported.
- **Readout** [lit]:
  - 500 ns pulse resolving all three states; average qutrit readout fidelity 93%.
  - Quote: "as our experiments only required a terminal readout and no mid-circuit
    measurement, we did not maximize the fidelity". The QND property was not
    characterized, which the authors call "critical ... in a qutrit quantum error
    correction code demonstration".
- **Coherence.** Figure 6b only; not extracted.
- **Bearing on this study.**
  - This is the strongest evidence that IQM-type square-lattice hardware works as a
    multi-qutrit device. The 15-qutrit chain exceeds the 9 qutrits this experiment needs
    [lit].
  - The coupler cross-Kerr control supplies the four phases a qutrit CZ needs
    [conjecture, by analogy with Goss2022 and Luo2023].
  - It was done with non-IQM electronics and without mid-circuit measurement [lit].

#### Nguyen2024
- **Citation.** L. B. Nguyen, N. Goss, K. Siva, Y. Kim, E. Younis, B. Qing, A. Hashim,
  D. I. Santiago, I. Siddiqi, "Empowering a qudit-based quantum processor by traversing
  the dual bosonic ladder", Nat. Commun. 15 (2024), DOI 10.1038/s41467-024-51434-2.
  arXiv:2312.17741, whose title is "Empowering high-dimensional quantum computing by
  traversing the dual bosonic ladder".
- **Fetched.** https://arxiv.org/abs/2312.17741, https://arxiv.org/html/2312.17741
- **Facts** [lit]:
  - Raman-assisted two-photon (double-excitation) hopping between coupled transmon
    qudits.
  - Native single-qudit gate fidelities: 0.99936 (d = 2), 0.99909 (d = 3), 0.9978
    (d = 4). A Clifford needs 2, 6, and 12 native gates respectively.
  - Qutrit Bell state: 97.7% raw (99.6% purified); ququart: 94.3% (99.3%).
  - Qutrit-level shelving gives CCZ at 96.0(3)% (CB) and CCCZ at 92(1)% (truth table).
- **Bearing on this study.** Another entangling route for qutrits, but there is no
  CZ/CSUM process fidelity to compare with Goss2022.

#### Vora2026 (QubiCML)
- **Citation.** N. R. Vora, Y. Xu, A. Hashim, N. Fruitwala, H. N. Nguyen, N. Goss,
  H. Liao, J. Balewski, A. Rajagopala, K. Nowrouzi, Q. Ji, K. B. Whaley, I. Siddiqi,
  P. Nguyen, G. Huang, "ML-Powered FPGA-based Real-Time Quantum State Discrimination
  Enabling Mid-circuit Measurements", arXiv:2406.18807. v1 is from 2024; I read v5,
  dated 2026-09-13.
- **Fetched.** https://arxiv.org/abs/2406.18807, https://arxiv.org/html/2406.18807 (v5)
- **Facts** [lit]:
  - A quantized MLP discriminator on an RFSoC FPGA (ZCU216) inside QubiC. Inference
    takes 40 ns (66 ns with softmax) and uses about 0.4% of the LUTs per pipeline.
  - Qutrit discrimination, readout 500–750 ns, eight transmons: P(0|0) = 99.1–99.8%,
    P(1|1) = 93.5–97.3%, P(2|2) = 88.0–95.0%, averages 94.5–97.3%.
  - Feed-forward on qutrits: mid-circuit measurement of a control qutrit, then X^m on
    the target conditioned on the outcome m ∈ {0,1,2}, as a precursor to CSUM.
  - Raw branch fidelities: |0,0>→|0,0> 98.9% and 99.0% (two pairs); m = 2 branches
    83.6–88.1%.
- **Bearing on this study.** This is the only transmon demonstration of qutrit
  mid-circuit measurement with real-time feed-forward that I found. The study only
  needs feedback for conditional reset or leakage handling; Pauli-frame corrections are
  classical (§5.4) [analytic].

### 3.2 Other two-qutrit gate platforms (2022–2026)

#### CerveraLierta2022
- **Citation.** A. Cervera-Lierta, M. Krenn, A. Aspuru-Guzik, A. Galda, "Experimental
  high-dimensional Greenberger-Horne-Zeilinger entanglement with superconducting
  transmon qutrits", Phys. Rev. Applied 17, 024062 (2022); arXiv:2104.05627 (v3);
  DOI 10.1103/PhysRevApplied.17.024062.
- **Fetched.** https://arxiv.org/abs/2104.05627, https://arxiv.org/pdf/2104.05627
- **Facts** [lit]:
  - Run on the IBM cloud device ibmq_rome through Qiskit Pulse. Custom R^(12) pulses
    are produced with a sideband at f12 - f01.
  - The default IBM CNOT is used. Its action on |2> was characterized: it adds phases
    and creates superpositions when the control is in |2>.
  - Readout uses the default 0–1 discriminator, which misclassifies |2> as |1>. Each
    basis-state probability is therefore measured by mapping that state to |000>, one
    experiment per basis element, which scales exponentially. The authors say future
    qutrit work "should program and calibrate a 0-1-2 discriminator".
  - Three-qutrit GHZ: F = 0.76 ± 0.01 mitigated, 0.69 ± 0.01 raw.
  - Device: ω12/2π = 4.44–4.93 GHz (anharmonicity about -330 to -340 MHz, arithmetic
    analytic); T1(1→0) = 36–118 μs; qubit CNOT 320–1145 ns with errors
    6.5–18.6×10⁻³; qubit readout error 2.0–4.3%.
- **Bearing on this study.** Qutrit use of a cloud qubit device with binary readout pays
  an exponential readout cost unless a ternary discriminator, or a two-measurement trick
  (Kanazawa2023), is available. IQM cloud has the same constraint (§4).

#### Luo2023
- **Citation.** K. Luo, W. Huang, Z. Tao, L. Zhang, Y. Zhou, J. Chu, W. Liu, B. Wang,
  J. Cui, S. Liu, F. Yan, M.-H. Yung, Y. Chen, T. Yan, D. Yu, "Experimental Realization
  of Two Qutrits Gate with Tunable Coupling in Superconducting Circuits",
  Phys. Rev. Lett. 130, 030603 (2023); arXiv:2206.11199;
  DOI 10.1103/PhysRevLett.130.030603.
- **Fetched.** https://arxiv.org/abs/2206.11199, https://arxiv.org/html/2206.11199
- **Facts** [lit]:
  - Two fixed-frequency transmon qutrits and a flux-tunable transmon coupler, on an
    8-transmon ring device. ω01/2π = 6.074 and 6.725 GHz; anharmonicities -256 and
    -236 MHz.
  - The coupler tunes the cross-Kerr interaction with an on/off ratio of "a few
    hundred" (above 100 MHz near resonance).
  - CPhase: two coupler flux pulses (50 ns half-sine edges, 19 ns flat top) and four
    single-qutrit gates of 40 ns each.
  - QPT fidelity 89.3 ± 1.9%, including SPAM. Simulation gives 91.2% with decoherence
    and 99.7% without.
  - EPR state: 95.5 ± 2.0%.
  - Coherence along the coupler trajectory: T1^12 = 13.9 and 8.9 μs, T2^12 = 12.3 and
    8.3 μs. The authors attribute the error mainly to decoherence.
- **Bearing on this study.** This is the closest published analogue to an IQM-style qutrit
  CZ: fixed or tunable transmons with a tunable coupler and coupler flux pulses. It
  shows that sub-μs qutrit CPhase gates on that architecture are coherence-limited
  [lit + conjecture].

#### Roy2023
- **Citation.** T. Roy, Z. Li, E. Kapit, D. I. Schuster, "Two-qutrit quantum algorithms
  on a programmable superconducting processor", Phys. Rev. Applied 19, 064024 (2023),
  published 2023-06-07; DOI 10.1103/PhysRevApplied.19.064024. arXiv:2211.06523 (v1),
  whose title is "Realization of two-qutrit quantum algorithms ...".
- **Fetched.** https://arxiv.org/abs/2211.06523, https://arxiv.org/html/2211.06523,
  https://link.aps.org/accepted/10.1103/PhysRevApplied.19.064024 (accepted manuscript
  PDF; its text was extracted).
- **Facts** [lit]:
  - Two transmons joined by a SQUID parametric coupler, driven by RF flux sidebands.
    ω/2π = 3.35 and 3.83 GHz; anharmonicities -115 and -160 MHz.
  - Native CPhase gates are 2π sideband pulses through **|3> states**: |22>↔|31> in
    94 ns and |21>↔|30> in 56 ns. |12>↔|03> exists but is not used because of lower
    fidelity.
  - T1^12 = 21.7 and 3.9 μs; T2R = 2.0–4.5 μs (dephasing-limited).
  - Single-qutrit π/2_01 ≈ 50 ns, π_12 ≈ 79–84 ns, H ≈ 142–148 ns; H process
    fidelity 98.96% and 97.06%.
  - No parametric amplifier: confusion-matrix inversion plus MLE.
  - Algorithm success probabilities: Deutsch-Jozsa 75.5% (constant) and 98.5%
    (balanced); Bernstein-Vazirani 78.3%; two-iteration Grover improves over one.
  - The deepest circuit (17 single-qutrit and 4 two-qutrit gates) lasts 2.11 μs. The
    state fidelity after one CPhase is 82.4%. No two-qutrit gate process fidelity is
    reported.
- **Bearing on this study.** Sideband gates are fast but leave the qutrit space by
  construction, so leakage to |3> is part of the gate mechanism [analytic from the
  stated transitions]. Any qutrit decoder on such hardware must treat |3> population as
  a separate error class [conjecture].

#### Fischer2023
- **Citation.** L. E. Fischer, A. Chiesa, F. Tacchino, D. J. Egger, S. Carretta,
  I. Tavernelli, "Universal qudit gate synthesis for transmons", PRX Quantum 4, 030327
  (2023); arXiv:2212.04496; DOI 10.1103/PRXQuantum.4.030327.
- **Fetched.** https://arxiv.org/abs/2212.04496, https://arxiv.org/html/2212.04496
- **Facts** (theory and simulation, IBM Zurich with Parma) [lit]:
  - The gate set is neighbouring-level single-qudit rotations plus a generalized echoed
    cross-resonance (ECR) gate.
  - Simulated: CR tone of 289 ns reaches 99.93% (up to local phases); the echo π pulse
    of 100 ns reaches 99.99%.
  - The full ECR(θ = π) has unitary fidelity 99.56%. Adding T1 = 310 μs and T2 = 170 μs
    gives 98.66%; with T1 = 474 μs and T2 = 666 μs it gives 99.22%. The ECR takes about
    1 μs.
  - The paper recommends EJ/EC ≈ 70. It notes that decay is predominantly sequential
    and that |3> lifetimes of about 30 μs are measured for qubit T1 of about 80 μs.
  - An error-correction sequence embedded in a single ququart protects against pure
    dephasing (simulation).
- **Bearing on this study.** This is the fixed-frequency (IBM-style) route to two-qudit
  gates, at about 1 μs. On tunable-coupler hardware the natural counterpart is
  coupler-flux cross-Kerr control (Luo2023, Goss2026) [conjecture].

#### Subramanian2023
- **Citation.** M. Subramanian, A. Lupascu, "Efficient two-qutrit gates in
  superconducting circuits using parametric coupling", Phys. Rev. A 108, 062616 (2023);
  arXiv:2309.05766 (v2); DOI 10.1103/PhysRevA.108.062616.
- **Fetched.** https://arxiv.org/abs/2309.05766, https://arxiv.org/pdf/2309.05766
- **Facts** (theory) [lit]:
  - The qutrit CZ is decomposed exactly into iSWAP01,10(2π/3) and iSWAP12,21(2π/3)
    partial swaps, driven by AC flux on a tunable coupler between two fixed-frequency
    transmons.
  - The simulated partial swaps reach 99.65% (315 ns) and 99.72% (435 ns).
  - The CZ reaches 99.41% assuming perfect single-qutrit rotations, with a total of
    about 750 ns excluding single-qutrit operations.
- **Bearing on this study.** A theory path to qutrit CZ above 99% on coupler hardware,
  but still slow.

### 3.3 Single-qudit control and coherence of higher levels

#### Liu2023
- **Citation.** P. Liu, R. Wang, J.-N. Zhang, Y. Zhang, X. Cai, H. Xu, Z. Li, J. Han,
  X. Li, G. Xue, W. Liu, L. You, Y. Jin, H. Yu, "Performing SU(d) operations and
  rudimentary algorithms in a superconducting transmon qudit for d = 3 and d = 4",
  Phys. Rev. X 13, 021028 (2023); arXiv:2304.11841; DOI 10.1103/PhysRevX.13.021028.
- **Fetched.** https://arxiv.org/abs/2304.11841, https://arxiv.org/html/2304.11841
- **Device and coherence** [lit]:
  - A single transmon with EJ/EC ≈ 88.
  - ω01/2π = 5.355 GHz, ω12/2π = 5.127 GHz, ω23/2π = 4.873 GHz.
  - T1: |1> 180 ± 3 μs, |2> 101 ± 1 μs, |3> 73 ± 1 μs.
  - T2 Ramsey: 76 ± 2 μs (01), 37 ± 3 μs (12), 22.8 ± 0.3 μs (23), ≥ 10 μs (34).
- **Readout** [lit]:
  - Simultaneous four-state single shot: P0 = 99.1%, P1 = 94.5%, P2 = 94.5%,
    P3 = 91.1%.
  - The largest off-diagonal entries are downward: P(0|1) = 5.4%, P(1|2) = 3.0%,
    P(0|2) = 2.4%, P(2|3) = 4.4%, P(0|3) = 3.8%.
- **Gates** [lit]:
  - π/2 cosine pulses of 45 ns (01), 35 ns (12), and 85 ns (23), each with a 10 ns
    buffer.
  - SU(3) Clifford error 5.4 ± 0.1×10⁻³, which is 1.02×10⁻³ per π/2 pulse at 5.25
    pulses per Clifford.
  - SU(4) Clifford error 5.8×10⁻² at about 14.3 pulses per Clifford.
  - Subspace SU(2) Clifford errors: {01} 4.6×10⁻⁴, {12} 1.7×10⁻³, {23} 2.3×10⁻³.
  - Leakage beyond the subspace is clearly visible in the RB populations. The authors
    note that RB is insensitive to the coherent part of the leakage.
- **Bearing on this study.** These are the best single-device |2> numbers (T1 ≈ 100 μs)
  among the sources I fetched. Because |3> is resolvable at 91%, leakage out of a
  qutrit could be flagged in readout on such a device [conjecture].

#### Peterer2015
- **Citation.** M. J. Peterer, S. J. Bader, X. Jin, F. Yan, A. Kamal, T. J. Gudmundsen,
  P. J. Leek, T. P. Orlando, W. D. Oliver, S. Gustavsson, "Coherence and Decay of Higher
  Energy Levels of a Superconducting Transmon Qubit", Phys. Rev. Lett. 114, 010501
  (2015); arXiv:1409.6031; DOI 10.1103/PhysRevLett.114.010501.
- **Fetched.** https://arxiv.org/abs/1409.6031, https://arxiv.org/html/1409.6031
- **Facts** [lit]:
  - A 3D transmon with EJ/EC = 58.
  - Sequential decay times: 84 μs (1→0), 41 μs (2→1), 30 μs (3→2), 22 μs (4→3).
    Γ(i→i-1) ≈ i·Γ(1→0), as for a harmonic oscillator.
  - Non-sequential decay times: 2→0 1812 μs, 3→1 1314 μs, 3→0 2631 μs, which is about
    two orders of magnitude suppressed.
  - T2: 72 μs (01), 32 μs (12), 12 μs (23), < 2 μs (34), each ± 20%.
  - Charge dispersion grows by more than an order of magnitude per level.
- **Bearing on this study.**
  - Amplitude damping on a qutrit acts as a downward shift |k>→|k-1| on the populated
    component. It is a biased channel, not uniform Weyl noise [analytic from the
    sequential-decay result].
  - Leakage to |3> cannot come from decay. It needs excitation by drives, measurement,
    or heating [analytic from the selection rule plus lit].

#### Wang2025
- **Citation.** Z. Wang, R. W. Parker, E. Champion, M. S. Blok, "Systematic study of
  High E_J/E_C transmon qudits up to d = 12", Phys. Rev. Applied 23, 034046 (2025);
  arXiv:2407.17407; DOI 10.1103/PhysRevApplied.23.034046.
- **Fetched.** https://arxiv.org/abs/2407.17407, https://arxiv.org/html/2407.17407
- **Facts** [lit]:
  - Six fixed-frequency transmons with EJ/EC = 88–325; up to 12 levels observed.
  - Adjacent-level process infidelity is below 3×10⁻³ in the lowest 10 levels.
  - Best 01 error per Clifford is 3.25×10⁻⁴ at 44 ns (leakage per Clifford was also
    measured).
  - 10-state readout fidelity 93.8% with a deep neural network on multi-tone readout of
    2.2 μs. The main error is decay during readout; higher levels have T1 of 10–20 μs.
  - T2E of the higher levels is close to the T1 limit.
  - Strong ZZ-like coupling between higher levels in a two-transmon system.
- **Bearing on this study.** Raising EJ/EC trades anharmonicity, and with it leakage at
  fast pulses, for charge-noise immunity. For qutrits, EJ/EC of about 70–90 is the range
  used by the multi-qutrit chips above [lit:Blok2021, Liu2023, Fischer2023].

### 3.4 Readout

#### Chen2023
- **Citation.** L. Chen, H.-X. Li, Y. Lu, C. W. Warren, C. J. Križan, S. Kosen,
  M. Rommel, S. Ahmed, A. Osman, J. Biznárová, A. Fadavi Roudsari, B. Lienhard,
  M. Caputo, K. Grigoras, L. Grönberg, J. Govenius, A. F. Kockum, P. Delsing,
  J. Bylander, G. Tancredi, "Transmon qubit readout fidelity at the threshold for
  quantum error correction without a quantum-limited amplifier", npj Quantum
  Information 9, 26 (2023); arXiv:2208.05879; DOI 10.1038/s41534-023-00689-6.
- **Fetched.** https://arxiv.org/abs/2208.05879, https://arxiv.org/html/2208.05879
- **Facts** [lit]:
  - Readout time 140 ns, without a quantum-limited amplifier.
  - Two-state assignment fidelity 99.5% using shelving (π12).
  - Three-state assignment fidelity 96.9% using π12 + π23 shelving (|1>→|3>, |2>→|1>),
    two multiplexed tones, and a feed-forward neural network. A single tone gives 94.2%.
  - In the secondary tone |2> and |3> overlap and are combined into one class.
  - Residual errors are mainly |2>→"0" (a shelving side effect) and |1>↔|2>.
  - Without shelving, direct |2> decay during 140 ns would cost 2.65%, against 0.035%
    for the |3>→|1> cascade.
- **Bearing on this study.** This is the fastest high-fidelity three-state readout
  found. Two caveats:
  - The qutrit version uses the |3> level, so the ancilla must have a calibrated 2–3
    transition [lit].
  - Leakage already in |3> is not separable from |2> in this scheme [lit], so it would
    be read as syndrome value 2 [conjecture].

#### Kanazawa2023
- **Citation.** N. Kanazawa, H. Emori, D. C. McKay, "Qutrit state discrimination with
  mid-circuit measurements", arXiv:2309.11303 (2023), preprint (IBM Quantum).
- **Fetched.** https://arxiv.org/abs/2309.11303, https://arxiv.org/html/2309.11303
- **Facts** [lit]:
  - The scheme: a binary measurement (|0> versus not-|0>), then an X01 pulse, then a
    second binary measurement. The outcomes (1,0) and (1,1) identify |1> and |2>.
  - It relies on the first measurement projecting onto the qutrit basis (QND) and needs
    no ternary IQ discriminator.
  - Its accuracy is comparable to, or slightly better than, a calibrated qutrit IQ
    discriminator on ibm_canberra and ibm_algiers.
  - The measurement durations are 1632 ns and 857 ns.
  - The spurious outcome "00" has mean probability 4.81×10⁻³ and 9.84×10⁻³. The
    authors attribute it to heating in the first measurement, or to X01 errors from
    residual photons.
  - It also works for leakage RB and population-decay measurements.
- **Bearing on this study.** On hardware with only binary mid-circuit readout (IQM
  today), this gives a ternary syndrome readout at twice the readout time
  [conjecture: requires that |2> is reliably binned with |1> and that readout is QND for
  |1> and |2>].

#### Kehrer2024
- **Citation.** T. Kehrer, T. Nadolny, C. Bruder, "Improving transmon qudit measurement
  on IBM Quantum hardware", Phys. Rev. Research 6, 013050 (2024); arXiv:2307.13504 (v2);
  DOI 10.1103/PhysRevResearch.6.013050.
- **Fetched.** https://arxiv.org/abs/2307.13504, https://arxiv.org/html/2307.13504
- **Facts** [lit]:
  - A mean-field readout model comparing single-frequency and multi-frequency
    strategies.
  - Ququart data from ibm_lagos Q4, with states prepared by two-photon j→j+2 X gates.
  - The default readout frequency is not optimal for four states, and the cloud stack
    does not let the user set the modulation frequency to its ideal value.
- **Bearing on this study.** Cloud users generally cannot re-optimize readout for
  qudits. On IQM, Pulla exposes calibration data (§4), but I found no documentation that
  readout pulses can be re-tuned for three-state discrimination [conjecture/gap].

### 3.5 Reset and leakage

#### Magnard2018
- **Citation.** P. Magnard, P. Kurpiers, B. Royer, T. Walter, J.-C. Besse,
  S. Gasparinetti, M. Pechal, J. Heinsoo, S. Storz, A. Blais, A. Wallraff, "Fast and
  Unconditional All-Microwave Reset of a Superconducting Qubit", Phys. Rev. Lett. 121,
  060502 (2018); arXiv:1801.07689; DOI 10.1103/PhysRevLett.121.060502.
- **Fetched.** https://arxiv.org/abs/1801.07689, https://arxiv.org/html/1801.07689
- **Facts** [lit]:
  - A drive on the |f,0>↔|g,1> transition, combined with e–f π pulses, unconditionally
    transfers any excitation in the three lowest levels into a resonator photon that
    then leaves the system ("resetting the transmon qutrit").
  - Residual excitation is 0.2% in under 500 ns, with no feedback.
  - The demonstration device had T1^ge = 5.5 μs, T1^ef = 2.1 μs.
- **Bearing on this study.** This is a measurement-free qutrit ancilla reset. IQM's pulse
  library ships an f0g1 leakage-reduction unit (§4), which is the same transition
  [lit:IQMpulseDocs].

#### McEwen2021
- **Citation.** M. McEwen, D. Kafri, Z. Chen, J. Atalaya, K. J. Satzinger, C. Quintana,
  P. V. Klimov, D. Sank, C. Gidney, A. G. Fowler, et al. (51 authors; last author
  R. Barends), "Removing leakage-induced correlated errors in superconducting quantum
  error correction", Nat. Commun. 12, 1761 (2021); arXiv:2102.06131;
  DOI 10.1038/s41467-021-21982-y.
- **Fetched.** https://arxiv.org/abs/2102.06131, https://arxiv.org/pdf/2102.06131
- **Facts** [lit]:
  - Multi-level reset on Sycamore: an adiabatic flux swap of all excitations into the
    readout resonator, which sits about 1.5 GHz below the qubit with g ≈ 120 MHz,
    followed by a fast return.
  - About 250 ns; "ground state with a fidelity of over 99%" from any of |1>, |2>, |3>.
    The error reaches the 0.2% readout floor at a 30 ns swap; the modelled dominant
    error is about 5×10⁻⁴.
  - In the bit-flip code, |2> leakage grows by γ↑ ≈ 0.09–0.11% per round and saturates
    near 1% without reset.
  - Reset on the measure qubits increases their leakage decay rate 40× and the data
    qubits' 2.4×, the latter by transport through the CZ's |21>/|03> resonance.
  - Leakage is long-lived and correlated in space and time.
- **Bearing on this study.** Two points:
  - It resets |3> as well, which a qutrit ancilla needs. It requires flux-tunable
    qubits; IQM qubits are flux-tunable [lit:Abdurakhimov2024]. Whether IQM readout
    resonators sit below the qubits, as this protocol requires, is not documented in
    what I read [gap].
  - Leakage is a physical source of time-correlated (non-Markovian) syndrome errors
    (§5.6) [lit + conjecture].

#### Haupt2025
- **Citation.** C. J. Haupt, A. Carrera Vazquez, L. E. Fischer, S. Woerner,
  D. J. Egger, "Mitigating errors in state preparation and measurement with
  noncomputational states", arXiv:2506.09145 (v2); DOI 10.1103/jv9h-spzw (APS; journal
  name not checked).
- **Fetched.** https://arxiv.org/abs/2506.09145, https://arxiv.org/html/2506.09145
- **Facts** [lit]:
  - A standard active reset (measure, then X if the result is 1) is "ineffective at
    removing the |2⟩-state population". The qubit discriminator bins |2> with |1>, and X
    does not act on |2>.
  - A qutrit reset needs three-way discrimination plus feed-forward I, X, or XX12. This
    is *simulated*; the hardware experiment (IBM Eagle) used a slow passive reset.
- **Bearing on this study.** IQM's `reset` is a measurement plus a conditional PRX on the
  0–1 transition (§4), so it is a qubit reset. It would leave |2> population on a qutrit
  ancilla [analytic + lit].

#### Hyyppa2024 (IQM)
- **Citation.** E. Hyyppä, A. Vepsäläinen, M. Papič, C. F. Chan, S. Inel, A. Landra,
  W. Liu, J. Luus, F. Marxer, C. Ockeloen-Korppi, S. Orbell, B. Tarasinski,
  J. Heinsoo, "Reducing leakage of single-qubit gates for superconducting quantum
  processors using analytical control pulse envelopes", PRX Quantum 5, 030353 (2024);
  arXiv:2402.17757 (v2); DOI 10.1103/PRXQuantum.5.030353.
- **Fetched.** https://arxiv.org/abs/2402.17757, https://arxiv.org/html/2402.17757
- **Facts** [lit]:
  - FAST DRAG and HD DRAG pulse envelopes.
  - Leakage below 3.0×10⁻⁵ per RX(π/2) down to 6.25 ns on a flux-tunable transmon at
    its sweet spot. The device has 4.417 GHz, anharmonicity -212 MHz, T1 ≈ 37 μs,
    T2e ≈ 22 μs (an IQM two-qubit test device).
  - Three-state readout for the leakage measurement: 1.5 μs without a quantum-limited
    amplifier; assignment-matrix diagonal 0.972 / 0.742 / 0.850.
- **Bearing on this study.** IQM's pulse-shaping work targets the 1–2 transition as a
  leakage channel. The same spectral shaping could protect a qutrit against 2–3 leakage
  [conjecture]. IQM's own three-state readout in that test setup was weak for |1> and
  |2> [lit].

### 3.6 Mid-circuit checks on qutrit levels

#### LiuBJ2026
- **Citation.** B.-J. Liu, Y.-Y. Wang, Y.-X. Wang, M. Badbaria, S. Puri, C. Wang,
  "Hardware-Efficient Erasure Qubits With Superconducting Transmon Qutrits",
  arXiv:2604.08672 (2026), preprint.
- **Fetched.** https://arxiv.org/abs/2604.08672, https://arxiv.org/html/2604.08672
- **Facts** [lit]:
  - The logical qubit lives in {|g>, |f>}. Decay to |e> is an erasure, detected by a
    200 ns microwave-activated four-wave-mixing "erasure SWAP" to an ancilla qubit and a
    1.4 μs ancilla readout.
  - One check round lasts 3.52 μs, and the checks are repeated mid-circuit.
  - Runs with a detected erasure are **discarded** (postselection).
  - False-positive rate about 2.0%; false-negative rate 6–8%, limited by ancilla
    coherence of 0.7–1 μs under the FWM drive and by ancilla decay during readout.
  - Post-selected bit-flip lifetime 549(56) μs and 610(46) μs, against T1^ge ≈ 55 μs.
    The erasure lifetime is 45 μs; 2·T1^ef ≈ 26 μs.
  - The abstract reports coherence beyond 300 μs with dynamical decoupling and
    single-qubit Clifford infidelity of order 10⁻⁴. The ancilla is reused for parity
    checks, and heralded Bell states are shown.
- **Bearing on this study.** This is the nearest hardware analogue of "repeated
  detection with postselection" on qutrit levels. The check-round time is set by the
  1.4 μs readout. The false-negative rate of a few percent shows that ternary-level
  detection is far from ideal today [lit + conjecture].

### 3.7 Bosonic qudits

#### Brock2025
- **Citation.** B. L. Brock, S. Singh, A. Eickbusch, V. V. Sivak, A. Z. Ding,
  L. Frunzio, S. M. Girvin, M. H. Devoret, "Quantum error correction of qudits beyond
  break-even", Nature 641, 612–618 (2025); arXiv:2409.15065;
  DOI 10.1038/s41586-025-08899-y.
- **Fetched.** https://arxiv.org/abs/2409.15065, https://arxiv.org/html/2409.15065
- **Device** [lit]: a GKP qutrit or ququart in a 3D cavity (T1,c = 631 μs,
  T2R,c = 1030 μs) with a tantalum transmon ancilla (T1,q = 295 μs, T2E,q = 286 μs).
- **Protocol** [lit]:
  - Generalized small-Big-small (sBs) stabilization, which is autonomous and needs only
    an ancilla reset between rounds, optimized by reinforcement learning.
  - Each round is 7 μs: 3632 ns of ancilla measurement and reset, 1808 ns of sBs gates,
    1336 ns idle, and 224 ns of FPGA instructions.
- **Results** [lit]:
  - Qutrit QEC gain G3 = 1.82 ± 0.03 (logical lifetime 886 ± 3 μs against 488 ± 7 μs
    for the best physical cavity Fock qutrit).
  - Ququart gain 1.87 ± 0.03 (logical lifetime 620 ± 2 μs); qubit gain 1.81.
- **Logical measurement** [lit]: a ternary GKP Pauli measurement made of two binary
  ancilla measurements, first {P0} versus {P1, P2}, then {P1} versus {P0, P2}. Its
  back-action was verified by Wigner tomography post-selected on the outcomes.
- **Bearing on this study.**
  - This is the only qudit QEC beyond break-even found, and it uses correction with
    repeated mid-circuit ancilla measurement and reset, not postselection [lit].
  - The two-binary-measurement construction of a ternary outcome is a generic pattern
    for qutrit syndromes on qubit-readout hardware [conjecture for transmon data
    qutrits].

## 4. IQM hardware: what it supports

Sources:

| Key | Citation | Fetched URLs |
|---|---|---|
| Abdurakhimov2024 | L. Abdurakhimov, J. Adam, H. Ahmad, et al. (114 authors), "Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer", arXiv:2408.12433 (2024) | https://arxiv.org/abs/2408.12433, https://arxiv.org/pdf/2408.12433 |
| Renger2025 | M. Renger, J. Verjauw, N. Wurz, et al. (65 authors), "A Superconducting Qubit-Resonator Quantum Processor with Effective All-to-All Connectivity", arXiv:2503.10903 (2025) | https://arxiv.org/abs/2503.10903 |
| Vigneau2025 | F. Vigneau, S. Majumder, A. Rath, et al. (17 authors), "Quantum error detection in qubit-resonator star architecture", arXiv:2503.12869 (v3), DOI 10.1103/8m33-wn4g | https://arxiv.org/abs/2503.12869, https://arxiv.org/html/2503.12869 |
| Hyyppa2024 | see §3.5 | |
| Goss2026 | see §3.1 | |
| IQMclientDocs | IQM client documentation, Release 36.0.0, `iqm.iqm_client.models` | https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.html (redirected from docs.meetiqm.com) |
| IQMpulseDocs | IQM pulse documentation, Release 15.0.0: `gates`, `gates.measure.Shelved_Measure_Constant`, `gates.lru`, `gates.prx`, `gates.reset.Reset_Conditional` | https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.html, https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.measure.Shelved_Measure_Constant.html, https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.lru.html, https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.prx.html, https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.reset.Reset_Conditional.html |
| IQMpullaDocs | IQM Pulla documentation, Release 15.0.0, "Custom Gates and Gate Implementations" | https://docs.iqm.tech/iqm-pulla/Custom%20Gates%20and%20Implementations.html |
| IQMpulseArticle2025 | IQM (M. Bruce), "How IQM's pulse-level access is fuelling value for end-users", 2025-02-28, updated 2026-04 | https://iqm.tech/?p=5957 |
| AWSGarnetDyn2025 | T. Chen, S. Smart, Z. Mohammad, "Experiment with dynamic circuits on IQM Garnet with Amazon Braket", AWS Quantum Technologies Blog, 2025-06-26 | https://aws.amazon.com/blogs/quantum-computing/experiment-with-dynamic-circuits-on-iqm-garnet-with-amazon-braket |
| AWSEmerald2025 | Z. Mohammad, C. Panchalam Govindarajan, P. Komar, S. Seegerer, "Amazon Braket launches new 54-qubit superconducting quantum processor from IQM", AWS blog, 2025-07-21 | https://aws.amazon.com/blogs/quantum-computing/amazon-braket-launches-new-54-qubit-superconducting-quantum-processor-from-iqm/ |

### 4.1 Connectivity and gates
- IQM "Crystal" QPUs have 5, 20, 54, or 150 computational qubits on a square lattice
  rotated by 45°, with nearest-neighbour tunable couplers [lit:Abdurakhimov2024].
- Qubits and couplers are flux-tunable transmons. Readout is frequency-multiplexed
  dispersive, with individual Purcell filters and three probe lines on Garnet
  [lit:Abdurakhimov2024].
- Garnet's native gates are PRX, virtual Z, and CZ by a coupler flux pulse "with a
  typical duration of 20 ns – 40 ns" [lit:Abdurakhimov2024]. The tunable ZZ reaches
  50 MHz and is fully off at idle. The architecture also supports cross-resonance,
  parametric, and iSWAP gates.
- Garnet medians: single-qubit error 9×10⁻⁴; CZ error 5×10⁻³; two-qubit Clifford error
  2×10⁻²; simultaneous readout error 3×10⁻². T1 is "of the order of 40 µs", with more
  than 100 µs shown in test devices [lit:Abdurakhimov2024].
- Emerald has 54 qubits, square lattice, tunable couplers, and median single-qubit and
  two-qubit fidelities of 99.93% and 99.5% [lit:AWSEmerald2025].
- The star architecture has six transmons around a central computational resonator,
  with a qubit-resonator CZ and a MOVE gate. A six-qubit GHZ state reaches 0.86
  (readout-error mitigated) [lit:Renger2025]. In the client API, `move` is "Star
  architecture only" [lit:IQMclientDocs].
- Whether a MOVE would transfer qutrit states (two-photon resonator states) was not
  documented in what I read [gap; conjecture that it is calibrated for single
  excitations only].

### 4.2 Mid-circuit measurement, feedback, reset (gate level)
- `measure` takes a `key` and an optional `feedback_key`. "Each qubit may be measured
  multiple times, i.e. mid-circuit measurements are allowed." [lit:IQMclientDocs]
- `cc_prx(angle, phase, feedback_qubit, feedback_key)`: "If the measurement result is 1,
  the PRX gate is applied. If it is 0, an identity gate of similar time duration gate is
  applied instead." The measurement must precede it [lit:IQMclientDocs]. The control is
  therefore one bit.
- `reset` "Resets the qubit(s) non-unitarily to the |0⟩ state", "inherits its
  calibration from cc_prx and is only available when cc_prx is" [lit:IQMclientDocs].
  `Reset_Conditional` in the pulse library "Uses a measurement followed by a conditional
  PRX gate with angle π". It states that "It is assumed the measurement projects the
  state into the computational basis" and that "The conditional PRX implementation
  handles any necessary waits to accommodate for the feedback result propagation delay"
  [lit:IQMpulseDocs]. No delay value is given. For a qutrit ancilla this does not clear
  |2> [analytic + lit:Haupt2025].
- Amazon Braket offers dynamic circuits (mid-circuit measurement, feed-forward, active
  reset) on Garnet as an "experimental capability" [lit:AWSGarnetDyn2025]. Two quoted
  restrictions: "conditional operations can only be applied to qubits in a local group"
  and "only supported as in verbatim compilation". Emerald supports the same features
  (`measure_ff`, `cc_prx`, active reset) [lit:AWSEmerald2025]. No latency figures are
  given in either source.
- Repeated mid-circuit ancilla readout has been used in IQM's own research, on the star
  QPU [lit:Vigneau2025]:
  - [[4,2,2]] error detection with ancillas measured every 2.00 μs cycle and **not
    reset**: "the stabilizer value is s_n = +1 if d_n ⊕ d_{n-1} = 0 and -1 otherwise".
    Results are postselected on no detection.
  - Logical error per cycle 0.25(2)%–0.91(3)%.
  - Device: binary readout fidelity 98.3–99.1%; single-qubit gates 99.89–99.96%;
    qubit-resonator CZ 98.0–99.0%; double MOVE 97.95–99.34%.
  - A separately optimized three-state classifier reaches > 99% for |0>, > 94% for |1>,
    and > 93% for |2> (one ancilla only > 84%). With it they saw leakage **accumulate in
    the ancillas measured every cycle**, attributed to measurement-induced transitions or
    MOVE gates.
  - With the default two-state classifier, "|2⟩ states are mostly classified as |1⟩".

### 4.3 Access to higher levels
- The circuit-level API exposes no gate on the 1–2 transition, and no ternary
  measurement result [lit:IQMclientDocs: the operation list is measure, prx, cc_prx,
  reset, cz, move, barrier, delay].
- The pulse library uses `prx_12` inside `Shelved_Measure_Constant`: "A measure gate
  implemented as a constant waveform is surrounded by two prx_12 gates". Acquisition
  type is `threshold`, which implies a binary result [lit:IQMpulseDocs; the binary
  reading is my inference]. The `prx` module page does not list `prx_12` as a separate
  class [lit:IQMpulseDocs].
- The pulse library has an `lru` gate, "a non-unitary quantum channel that removes
  population from the |2⟩ state of a qubit", with implementation `LRU_F0G1` "driving the
  |f0⟩ ↔ |g1⟩ transition" [lit:IQMpulseDocs]. This is the Magnard2018 mechanism.
- Pulse-level access (IQM Pulla) is offered through IQM Resonance. It lets users modify
  calibration data "including drive frequencies and pulse amplitudes", define custom
  gates and waveforms, and sweep parameters [lit:IQMpulseArticle2025, IQMpullaDocs].
  The Pulla custom-gate guide does not mention the 1–2 transition or drive-frequency
  changes [lit:IQMpullaDocs].
- Qutrit operation of an IQM-fabricated chip has been shown with non-IQM electronics
  [lit:Goss2026]. I found no IQM-authored qutrit or qudit computation paper [search
  result].
- Repo history: commit `1bd97c40` ("Restore IQM generation pipeline ...") added
  `external/iqm_qpu_circuits/[[5,1,2]]_3 Qutrit Surface Code/...` files (`.chp` and Cirq
  pickles for noiseless const-0 LRB circuits) [code/git, not inspected further].

### 4.4 Assessment for a 9-qutrit LRB on IQM hardware
- **Feasible today at gate level:** nothing qutrit-specific. The gate set is qubit-only
  [lit].
- **Feasible at pulse level, plausibly** [conjecture]:
  - single-qutrit gates from PRX-type pulses at the 1–2 frequency (the shelving gate
    shows the 1–2 drive is calibrated in the stack);
  - qutrit CZ from coupler flux pulses with four calibrated phases (Goss2026 shows the
    cross-Kerr control on an IQM chip; Luo2023 shows a coupler CPhase);
  - an unconditional qutrit ancilla reset (LRU_F0G1 plus a 1–2 π pulse, Magnard-style);
  - ternary readout either by a three-state classifier (used internally,
    Vigneau2025, Hyyppa2024) or by two binary measurements with an X01 pulse in between
    (Kanazawa2023).
- **Not documented anywhere I looked** [gap]:
  - whether Resonance users can calibrate readout for three states;
  - whether `cc_prx` can be conditioned on a ternary result;
  - feedback latency;
  - qutrit coherence numbers of Garnet or Emerald.

## 5. Analysis for this study

### 5.1 Connectivity [numeric + analytic]
Script: `L6-scripts/L6_grid_embed.py`.

- The check-round interaction graph (ancilla 5–{0,1,3}, 6–{1,2,4}, 7–{0,1,2},
  8–{3,1,4}; 12 edges) is bipartite. It embeds into a 3×3 grid, and since it has the
  same 12 edges it *is* the 3×3 grid: data 1 at the centre, ancillas on the edge
  midpoints, data 0, 2, 3, 4 at the corners [numeric].
- The logical H and P need SWAP(2,3) and CZ(2,3) [code: `src/lrb/code_definitions.py`,
  lines 498–505]. Adding the edge (2,3) creates odd cycles such as 3-2-6-4-8, so the
  graph cannot be a subgraph of any square lattice, which is bipartite [numeric +
  analytic].
- Over all 72 embeddings of the check graph into a 5×5 grid, data 2 and 3 are always at
  grid distance 4 [numeric].
  - Analytic reason: data 1 has degree 4, so the four ancillas sit at its four
    neighbours.
  - Data 0, 2, 3, 4 each need a pair of orthogonal ancillas.
  - This forces ancilla 5 opposite 6 and 7 opposite 8, so the corners {6,7} and {5,8}
    holding data 2 and 3 are diagonally opposite.
- Consequence for IQM Crystal hardware [conjecture]:
  - Checks are local.
  - Each logical H or P token costs routing of about 3 SWAP-equivalents, through an
    occupied data site or an extra qutrit outside the patch. In the brief's circuits
    this routing would sit in the noisy Clifford blocks.
  - Alternatively, use the repo's split-ancilla folded layout, which makes the logical
    gates local at the price of relay ancillas (11 qutrits) [code].
  - The star architecture gives all-to-all connectivity only for qubit states [lit +
    gap].
- Each check round needs at least 4 sequential two-qutrit gate layers, because data 1
  is in all four stabilizers. For a bipartite graph the edge chromatic number equals
  the maximum degree, here 4 (König) [analytic + numeric].

### 5.2 SI1000 versus hardware [numeric under a stated model]
Script: `L6-scripts/L6_si1000_map.py`.

- **Idle model.** A qutrit Lindbladian with sequential T1 (1→0 and 2→1) and
  independent level-projector dephasing fitted to the three pairwise T2 values. The
  equivalent p for the brief's measurement-layer idle (N1 with probability 2p) is
  e(t)/2.
- **Idle results** (Blok2021 average coherence, echo T2):

  | Idle time | e | Equivalent p |
  |---|---|---|
  | 0.14 μs | 0.32% | 0.0016 |
  | 0.5 μs | 1.13% | 0.0057 |
  | 1.0 μs | 2.25% | 0.011 |
  | 1.5 μs | 3.35% | 0.017 |
  | 2.2 μs | 4.87% | 0.024 |

  With Ramsey T2* these roughly double. With Liu2023 single-qudit coherence they are
  about 25% lower.
- **Equivalent p by location:**

  | SI1000 location | Hardware number | Equivalent p |
  |---|---|---|
  | 2-qutrit gate (N2, prob p) | CZ† 2.7% / CZ 4.8% [Goss2022]; CSUM 11.1% [Blok2021]; CPhase 10.7% [Luo2023]; CSUM 18% [Morvan2021, arXiv] | 0.027 / 0.048 / 0.11 / 0.11 / 0.18 |
  | 1-qutrit gate (N1, prob 0.1p) | published average 3.8×10⁻³ [Morvan2021]; per-qutrit r 2.38–3.94×10⁻³ → e = 3.2–5.3×10⁻³; SU(3) r = 5.4×10⁻³ [Liu2023] | 0.038; 0.032–0.053; 0.072 |
  | diagonal 1-qutrit gates (Z, P) | virtual frame updates | about 0 |
  | measurement flip (prob 5p) | 3-state readout 96.9% / 96% / 95.5% / 93% | 0.006 / 0.008 / 0.009 / 0.014 |
  | reset shift (prob 2p) | 0.2% residual [Magnard2018, McEwen2021] | 0.001 |
  | idle in gate layers | absent from SI1000 | — |

- **Idle within a check round** [numeric]: e = 5.1% at 2.3 μs, 7.6% at 3.5 μs, and
  10.7% at 5 μs per data qutrit (Blok2021 coherence). The 2.3 μs case is 4 layers of
  580 ns CZ†. Adding single-qutrit layers, about 0.5 μs of readout, and about 0.3 μs of
  reset gives about 3.3–5 μs per round [conjecture-level estimate from the cited
  durations].
- **Reading** [conjecture]:
  - Measured gate errors put hardware at p ≈ 0.03–0.07. Measurement, reset, and
    idle-during-readout put it at p ≈ 0.001–0.014.
  - SI1000 at p ≈ 0.03 therefore overstates the cost of the measurement and reset
    layers by about 3–30×. It understates the cost of a check round, because it has no
    idle noise during the ≥ 4 gate layers.
  - It also charges virtual Z and P gates.
  - Comparisons of check policies and decoders should be repeated with a duration-aware
    circuit noise model built from these numbers before drawing hardware conclusions.
- **Bias** [lit + conjecture]:
  - Hardware readout errors are dominated by P(i-1|i) [Liu2023, Blok2021, Wang2025] and
    amplitude damping by |k>→|k-1| [Peterer2015], whereas SI1000 uses symmetric X^a
    shifts.
  - Residual two-qutrit gate errors are mainly phase (Z-type) errors [Goss2022].
  - A decoder prior that uses the bias may matter more for qutrits than for qubits.

### 5.3 Measurement layout in the brief versus hardware [conjecture]
The brief's circuit measures the four ancillas in four separate layers [brief]. Hardware
qutrit readout is frequency-multiplexed and simultaneous [lit:Blok2021, Goss2026,
Abdurakhimov2024]. A hardware run would measure all four together, cutting the
measurement-layer idle exposure of each data qutrit from 4×2p to 1×2p per round. The
brief's sequential layout is pessimistic in this respect.

### 5.4 What correction requires on hardware
- **No real-time feedback** [analytic]. The LRB circuits are Clifford, and the brief's
  noise is Weyl. A Weyl error propagates through the remaining Clifford circuit to a
  Weyl operator, so its effect on the terminal X-basis readout of wires 0 and 3 is a
  fixed outcome shift.
  - The decoder's predicted logical shift can therefore be applied in post-processing
    (Pauli-frame tracking).
  - Hardware needs mid-circuit ternary readout and storage of outcomes, not
    feed-forward.
  - Feedback is needed only for conditional reset or active leakage removal.
- **Reset is optional** [analytic]. For a Z-check, CNOT from data to ancilla maps |m> to
  |m + s>. For an X-check (H, controlled-X^a kickback, H†), an ancilla starting in |m>
  becomes H|m> = Σ_a ω^{ma}|a>/√3, then Σ_a ω^{(m+s)a}|a>/√3 = H|m+s>, then |m+s>.
  - A QND readout that leaves the ancilla in |m> therefore gives the next round's
    outcome m + s (mod 3). Detectors become consecutive-outcome differences.
  - IQM uses the qubit version of this without reset [lit:Vigneau2025].
  - It needs a decoder detector-model change: difference detectors instead of the
    brief's raw-outcome detectors [conjecture on implementation].
  - It also lets ancilla leakage persist across rounds [lit:Vigneau2025, McEwen2021].
- **Ternary readout options:**
  1. A native three-state discriminator: 93–97% at 0.14–1.5 μs [lit].
  2. Two binary measurements with an X01 pulse between them [lit:Kanazawa2023;
     analogous logical construction in Brock2025]. This costs twice the readout time and
     depends on QND behaviour [conjecture].
- **Leakage to |3>**:
  - Sources: sideband CPhase gates use |3> by design [lit:Roy2023]; leakage-by-drive
    appears in qutrit RB [lit:Liu2023, Morvan2021]; repeated measurement can cause
    ancilla leakage [lit:Vigneau2025].
  - Removal: multi-level reset clears |3> on ancillas [lit:McEwen2021].
  - I found no qutrit leakage-reduction unit for data qutrits and no measured rate of
    leakage out of the qutrit space per qutrit gate [gap].
  - In Chen2023's fast readout |3> is read as |2> [lit], so ancilla leakage would turn
    into wrong syndromes rather than flags [conjecture].

### 5.5 Hardware sources of non-Markovian noise relevant to Q2
These are from the literature; their impact on LRB is my conjecture.

- Leakage: long-lived, correlated in space and time [lit:McEwen2021], and accumulating
  in repeatedly measured ancillas [lit:Vigneau2025].
- Always-on cross-Kerr between fixed-frequency qutrits: coherent, depends on the state
  of neighbours [lit:Blok2021, Goss2024]. Couplers reduce it to about 25 kHz
  [lit:Goss2026].
- Spectator entangling phases and coherent calibration errors of the four CZ phases
  [lit:Goss2024].
- Two-level-system fluctuators and drift: a qutrit lost to a TLS [lit:Morvan2021];
  TLS-driven fluctuations [lit:Brock2025]; drift between batches [lit:Goss2022].

Randomized compiling turns coherent errors into stochastic Weyl errors but does not
remove time correlations from drift, TLS, or leakage [lit:Goss2024 for the first part;
conjecture for the second]. Correction with a Markovian decoder prior would see these
as correlated detector events. Whether correction mitigates or amplifies their effect
on the LRB decay depends on that structure and is open [conjecture].

### 5.6 A sustainable hardware path (ranked, all [conjecture])
1. **Berkeley/LBNL-style stack**: fixed or tunable transmon qutrits with QubiC FPGA
   control. It already has qutrit CZ at 95–97%, 93–97% ternary readout, real-time
   ternary feed-forward (Vora2026), and RC (Goss2024). This is the shortest path to a
   9-qutrit LRB with mid-circuit checks.
2. **IQM-fabricated square-lattice chip with pulse-level control**: 15-qutrit operation
   has been shown (Goss2026). It needs a qutrit CZ from the coupler cross-Kerr, ternary
   readout, and an f0g1-based reset. The IQM cloud gate API does not support this
   today; Pulla might, subject to the access questions in §4.4.
3. **IQM cloud at gate level**: only a qubit encoding of the experiment is possible.
4. **Bosonic qudits (Brock2025)**: QEC beyond break-even for a single logical qutrit.
   This tests correction for a qutrit memory, but it is not a multi-qutrit stabilizer
   code and cannot run this LRB circuit as is.

## 6. Gaps, unverified leads, and dropped items
- **Not found:**
  - a transmon-qutrit stabilizer-code (QED or QEC) experiment;
  - a measured per-gate leakage rate out of the qutrit space;
  - a qutrit leakage-reduction unit;
  - a transmon qutrit CZ or CSUM above 97.3% process fidelity;
  - IQM-authored qutrit work;
  - documentation of ternary readout or 1–2 drives for IQM Resonance users;
  - feedback latency figures for IQM.
- **Figure-only, not extracted:** Goss2026 coherence statistics and simultaneous RB
  values.
- **Cited by fetched papers but not fetched by me** (not used as evidence):
  - Reuer et al. (three-level readout 88.7% with FPGA control, cited in Vora2026);
  - Champion et al., Phys. Rev. X 15, 021096 (2025), cited in Goss2026;
  - arXiv:2304.11159 and arXiv:2303.04796, which appeared only in search results.
- **Inconsistency recorded:** Morvan2021 arXiv v1 versus published abstract (2.38×10⁻³
  "as low as" versus 3.8×10⁻³ average; CSUM process fidelity 0.82 versus 0.85).
- **Search-snippet claims I did not use:** an "IQM Sirius" device (not found), and Garnet
  quantum volume or CLOPS beyond the CLOPS = 2600 stated in Abdurakhimov2024.

## 7. Files
- `notes/L6-scripts/L6_grid_embed.py`: square-lattice embedding check (networkx).
- `notes/L6-scripts/L6_si1000_map.py`: idle-channel process infidelity and SI1000
  equivalence table.
- `notes/L6-scripts/L6_round_idle.py`: idle infidelity over check-round durations, and
  the minimum layer count.
