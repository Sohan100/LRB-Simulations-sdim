# Lit review 02: Non-Markovian and gate-dependent noise in randomized benchmarking

Scope: formal definitions of non-Markovianity that matter for RB (process tensors,
temporally correlated noise, gate-dependent noise); when `A f^m + B` fails; how
correlations and memory reshape decay curves; tests for non-exponential decay;
leakage as a source of memory; and how postselection, heralding, conditioning or
QEC machinery can make the effective logical dynamics non-Markovian.

Compiled 2026-10-28 for the qubit `[[5,1,2]]` correction-vs-detection LRB study
(see `../CONTEXT.md`). Every source in the table below was fetched from its arXiv
abstract page, arXiv HTML/PDF, ar5iv, or publisher page in this session. Claims
attributed to a source were checked against its abstract and, where marked
"(body)", against the relevant section of the full text.

Labels used: **[lit]** = from literature (with citation key), **[analytic]** =
verified analytically (argument given), **[conj]** = conjecture / my inference,
**[ctx]** = observation already recorded in `../CONTEXT.md`.

---

## 1. Bibliography

Role: C = core (findings rest on it), S = supporting.

| Key | Title | Authors | Year | arXiv / DOI | URL | Fetched | Role |
|---|---|---|---|---|---|---|---|
| Wallman2018 | Randomized benchmarking with gate-dependent noise | J. J. Wallman | 2018 (arXiv 2017) | arXiv:1703.09835; Quantum 2, 47; doi:10.22331/q-2018-01-29-47 | https://arxiv.org/abs/1703.09835 | yes (abs, journal page, ar5iv body) | C |
| Proctor2017 | What randomized benchmarking actually measures | T. Proctor, K. Rudinger, K. Young, M. Sarovar, R. Blume-Kohout | 2017 | arXiv:1702.01853; PRL 119, 130502; doi:10.1103/PhysRevLett.119.130502 | https://arxiv.org/abs/1702.01853 | yes (abs) | S |
| Helsen2022 | A general framework for randomized benchmarking | J. Helsen, I. Roth, E. Onorati, A. H. Werner, J. Eisert | 2022 (arXiv 2020) | arXiv:2010.07974; PRX Quantum 3, 020357; doi:10.1103/PRXQuantum.3.020357 | https://arxiv.org/abs/2010.07974 | yes (abs) | C |
| Epstein2014 | Investigating the limits of randomized benchmarking protocols | J. M. Epstein, A. W. Cross, E. Magesan, J. M. Gambetta | 2014 (arXiv 2013) | arXiv:1308.2928; PRA 89, 062321; doi:10.1103/PhysRevA.89.062321 | https://arxiv.org/abs/1308.2928 | yes (abs, ar5iv body) | C |
| Ball2016 | The effect of noise correlations on randomized benchmarking | H. Ball, T. M. Stace, S. T. Flammia, M. J. Biercuk | 2016 (arXiv 2015) | arXiv:1504.05307; PRA 93, 022303; doi:10.1103/PhysRevA.93.022303 | https://arxiv.org/abs/1504.05307 | yes (abs) | C |
| Mavadia2018 | Experimental quantum verification in the presence of temporally correlated noise | S. Mavadia, C. L. Edmunds, C. Hempel, H. Ball, F. Roy, T. M. Stace, M. J. Biercuk | 2018 (arXiv 2017) | arXiv:1706.03787; npj Quantum Inf. 4, 7; doi:10.1038/s41534-017-0052-0 | https://arxiv.org/abs/1706.03787 | yes (abs) | S |
| FongMerkel2017 | Randomized Benchmarking, Correlated Noise, and Ising Models | B. H. Fong, S. T. Merkel | 2017 | arXiv:1703.09747 (no journal ref listed) | https://arxiv.org/abs/1703.09747 | yes (abs, ar5iv body) | C |
| Fogarty2015 | Non-exponential Fidelity Decay in Randomized Benchmarking with Low-Frequency Noise | M. A. Fogarty, M. Veldhorst, R. Harper, C. H. Yang, S. D. Bartlett, S. T. Flammia, A. S. Dzurak | 2015 | arXiv:1502.05119; PRA 92, 022326; doi:10.1103/PhysRevA.92.022326 | https://arxiv.org/abs/1502.05119 | yes (abs, ar5iv body) | C |
| Srivastava2026 | Blind-spots of Randomized Benchmarking Under Temporal Correlations | V. Srivastava, A. K. Roy, S. Mahanti, J. Kaur, S. Karuvade, A. Gilchrist | 2026 (arXiv 2025) | arXiv:2510.13051; PRR 8, 023258; doi:10.1103/r1z5-ycqc | https://arxiv.org/abs/2510.13051 | yes (abs) | S |
| Pollock2018 | Non-Markovian quantum processes: complete framework and efficient characterisation | F. A. Pollock, C. Rodríguez-Rosario, T. Frauenheim, M. Paternostro, K. Modi | 2018 (arXiv 2015) | arXiv:1512.00589; PRA 97, 012127; doi:10.1103/PhysRevA.97.012127 | https://arxiv.org/abs/1512.00589 | yes (abs) | S |
| FigueroaRomero2021 | Randomized Benchmarking for Non-Markovian Noise | P. Figueroa-Romero, K. Modi, T. M. Stace, M.-H. Hsieh | 2021 | arXiv:2107.05403; PRX Quantum 2, 040351; doi:10.1103/PRXQuantum.2.040351 | https://arxiv.org/abs/2107.05403 | yes (abs, ar5iv body) | C |
| FigueroaRomero2022 | Towards a general framework of Randomized Benchmarking incorporating non-Markovian Noise | P. Figueroa-Romero, K. Modi, M.-H. Hsieh | 2022 | arXiv:2202.11338; Quantum 6, 868; doi:10.22331/q-2022-12-01-868 | https://arxiv.org/abs/2202.11338 | yes (abs) | S |
| FigueroaRomero2024 | Operational Markovianization in Randomized Benchmarking | P. Figueroa-Romero, M. Papič, A. Auer, M.-H. Hsieh, K. Modi, I. de Vega | 2024 (arXiv 2023) | arXiv:2305.04704; Quantum Sci. Technol. 9, 035020; doi:10.1088/2058-9565/ad3f44 | https://arxiv.org/abs/2305.04704 | yes (abs) | C |
| WoodGambetta2018 | Quantification and Characterization of Leakage Errors | C. J. Wood, J. M. Gambetta | 2018 (arXiv 2017) | arXiv:1704.03081; PRA 97, 032306; doi:10.1103/PhysRevA.97.032306 | https://arxiv.org/abs/1704.03081 | yes (abs, ar5iv body) | C |
| Wallman2016 | Robust characterization of leakage errors (arXiv listing title: "Characterization of Leakage Errors via Randomized Benchmarking") | J. J. Wallman, M. Barnhill, J. Emerson | 2016 (arXiv 2014) | arXiv:1412.4126; New J. Phys. 18, 043021; doi:10.1088/1367-2630/18/4/043021 | https://arxiv.org/abs/1412.4126 | yes (abs, IOP page) | S |
| ChenBaldwin2025 | Randomized Benchmarking with Leakage Errors | Y.-H. Chen, C. H. Baldwin | 2025 | arXiv:2502.00154; PRR 7, 043065; doi:10.1103/j8sz-6ws9 | https://arxiv.org/abs/2502.00154 | yes (abs, HTML body) | C |
| Combes2017 | Logical Randomized Benchmarking | J. Combes, C. Granade, C. Ferrie, S. T. Flammia | 2017 | arXiv:1702.03688 (no journal ref listed) | https://arxiv.org/abs/1702.03688 | yes (abs, ar5iv body) | C |
| HarperFlammia2019 | Fault-Tolerant Logical Gates in the IBM Quantum Experience | R. Harper, S. T. Flammia | 2019 (arXiv 2018) | arXiv:1806.02359; PRL 122, 080504; doi:10.1103/PhysRevLett.122.080504 | https://arxiv.org/abs/1806.02359 | yes (abs, ar5iv body) | C |
| Kabir2025 | Sdim: A Qudit Stabilizer Simulator | A. Kabir, S. Nguyen, S. Ghosh, J. Keppens, T. Kiran, I. H. Kim, Y. Huang, B. Sorée | 2025 (v2 2026) | arXiv:2511.12777 | https://arxiv.org/abs/2511.12777 | yes (abs, HTML body Sec. VII-A) | C |
| Ceasura2022 | Non-Exponential Behaviour in Logical Randomized Benchmarking | A. Ceasura, P. Iyer, J. J. Wallman, H. Pashayan | 2022 | arXiv:2212.05488 (no journal ref listed) | https://arxiv.org/abs/2212.05488 | yes (abs, PDF body) | C |
| Ziyad2025 | Emergent Non-Markovianity in Logical Qubit Dynamics | J. A. Ziyad, R. Blume-Kohout, K. Rudinger | 2025 (v3 2026) | arXiv:2512.08893 | https://arxiv.org/abs/2512.08893 | yes (abs, PDF body) | C |
| Kwiatkowski2025 | Constructing an approximate logical Markovian model of consecutive QEC cycles of a stabilizer code | A. Kwiatkowski, A. J. Friedman, S. Geller, J. A. Ziyad, S. Glancy, E. Knill | 2025 | arXiv:2509.16887 | https://arxiv.org/abs/2509.16887 | yes (abs, PDF body) | C |
| Kumar2026 | Modeling Quantum Error Detection with Transition Matrices | R. S. Kumar, B. Foxman, Y. Ding | 2026 | arXiv:2609.12073 (extended version of IEEE QCE 2026 paper) | https://arxiv.org/abs/2609.12073 | yes (abs, HTML body) | C |
| Endo2025 | Non-Markovianity in Quantum Information Processing: Interplay with Quantum Error Mitigation | S. Endo, H. Hakoshima, T. Shitara | 2025 | arXiv:2510.20224 | https://arxiv.org/abs/2510.20224 | yes (abs only) | S |
| Girling2025 | Characterization of syndrome-dependent logical noise in detector regions | M. Girling, B. Criger, C. Cirstoiu | 2025 | arXiv:2508.08188 | https://arxiv.org/abs/2508.08188 | yes (abs only) | S |

25 sources (17 core, 8 supporting). This is above the target of 20 because the
scope covers six sub-topics. The 8 supporting entries each back a single
finding. Sources found but not cited (fetched, judged lower relevance):
Brillant et al. arXiv:2501.06172, Gandhari and Gullans arXiv:2502.14702,
Kam et al. arXiv:2410.23779, Kobayashi et al. arXiv:2412.13739, Budini
arXiv:1811.03448, Qi and Ng arXiv:2010.11498, Magesan et al. arXiv:1109.6887,
Carignan-Dugas et al. arXiv:1804.01122, Chasseur and Wilhelm arXiv:1505.00580,
Pollock et al. arXiv:1801.09811, Milz and Modi arXiv:2012.01894, Yang et al.
arXiv:2207.01542.

---

## 2. Key findings (all **[lit]**, one sentence each)

Q1 = decoders / correction vs detection, Q2 = non-Markovianity, Q3 = hardware.

### 2.1 Definitions of (non-)Markovianity relevant to RB

- **K1 [lit]** The process-tensor framework writes a multi-time process with memory as a many-body (Choi) state in which temporal correlations become spatial ones, and shows it can be reconstructed experimentally (Pollock2018). *Q2*
- **K2 [lit]** In RB, the noise is Markovian exactly when the process tensor factorizes into system-only noise maps between gates with an uncorrelated initial state, and RB non-Markovianity can be quantified as the q-norm distance between the measured average sequence fidelity (ASF) and a "Markovianized" ASF obtained by dissipating the environment after each step (FigueroaRomero2021, body). *Q2*
- **K3 [lit]** For programmable devices and logical qubits, "button-theoretic" Markovianity requires every press of a gate button to apply the same fixed CPTP map so that sequences compose as products, and logical Markovianity requires this for all logical operations including the logical idle (Ziyad2025, body). *Q2*
- **K4 [lit]** The answer to "is QEC non-Markovian?" depends on the definition: Endo et al. argue that negativity (a CP-divisibility-type signature) arises naturally in QEC from feedback conditioned on syndrome outcomes held in a gauge subsystem (Endo2025), while Ziyad et al. state their operational notion differs from CP-divisibility (Ziyad2025, body). *Q2*

### 2.2 When `A f^m + B` holds and when it fails

- **K5 [lit]** RB output is a linear combination of exponential decays when the implemented representation is multiplicity-free and the noise is time- and gate-independent, and under realistic conditions it is a linear combination of matrix-exponential decays (Ceasura2022, body; Helsen2022). *Q2*
- **K6 [lit]** For Markovian, time-independent but arbitrarily gate-dependent noise on a unitary 2-design, the mean survival probability is `A p^m + B t^m + eps_m` with `|eps_m| <= d1 * d2^m`, where `t` is the per-step trace (loss) and `d2` measures gate dependence, so the curve converges exponentially fast to a single exponential (essentially exact for m >= 10 when d2 <~ 1/2) and `p` still measures the fidelity of the noise between ideal gates (Wallman2018, body). *Q1, Q2*
- **K7 [lit]** Wallman argues that earlier perturbative analyses (expanding about a perfect twirl) and fixed-gauge bounds do not rigorously justify fitting RB decays to a single exponential (Wallman2018, body). *Q2*
- **K8 [lit]** Under gate-dependent noise the RB decay for small errors is still a simple exponential, but its rate need not equal the infidelity of any physically allowed (completely positive) representation of the gates, so "average gate infidelity" is gauge-dependent (Proctor2017). *Q1*
- **K9 [lit]** With non-Markovian noise the ASF is "not a simple function of sequence length" any more, but it remains stable under small gate-dependent perturbations, unlike the Markovian case where gate dependence enters as a perturbative term (FigueroaRomero2021, body; FigueroaRomero2022). *Q2*
- **K10 [lit]** Multiplicity in the representation plus a hidden memory register is enough to make gate- and time-independent noise produce oscillating ("uptick") RB output, i.e. memory and multiplicity break the exponential-decay theorems without any drift (Ceasura2022, body). *Q2*

### 2.3 How temporal correlations and memory reshape the curve

- **K11 [lit]** For temporally correlated Gaussian noise, the expected single-qubit RB fidelity maps to the partition function of a long-range spin-1 Ising model, and the decay crosses over from exponential (uncorrelated noise) to a power law (quasistatic noise) (FongMerkel2017). *Q2, Q3*
- **K12 [lit]** Under quasistatic noise, exponential fits give inconsistent noise estimates and underestimate the average gate error by factors of two to six, while fitting only short sequences can overestimate it (FongMerkel2017, body). *Q2*
- **K13 [lit]** Modeled as a random walk in Pauli space, RB outcomes are gamma-distributed with correlation-dependent parameters, and long correlations give large variance and skew toward high fidelity, causing finite-sampling pitfalls (Ball2016), as seen in a trapped-ion experiment where slow engineered noise gave broad, skewed fidelity distributions and fast noise near-Gaussian ones (Mavadia2018). *Q2, Q3*
- **K14 [lit]** Epstein et al. found the standard exponential model "not completely accurate" for 1/f noise (they fit a Gaussian-in-m form instead) and derived a different model for leakage, yet RB still estimated the average error within a factor of two in nearly all simulated cases (Epstein2014, body). *Q2, Q3*
- **K15 [lit]** Some system-environment interaction Hamiltonians make temporal correlations completely invisible in the ASF while still changing worst-case (diamond-norm) error, so an exponential-looking RB curve does not prove Markovianity (Srivastava2026). *Q2*
- **K16 [lit]** For finite-memory noise, non-Markovian effects act like SPAM after the memory length l, so the tail of the ASF looks exponential, and l can be estimated by setting Cliffords to identity where the decay is non-exponential and matching decay rates (FigueroaRomero2021, body). *Q2*

### 2.4 Tests for non-exponential decay and memory

- **K17 [lit]** In silicon quantum-dot RB, low-frequency noise produced non-exponential decays, and an Akaike-information-criterion comparison favored a two-rate model (a sum of two exponentials in m) over one rate plus an offset on all but one dataset, with the single-rate fit underestimating fidelity for m >= 100 (Fogarty2015, body). *Q2, Q3*
- **K18 [lit]** Interleaving identities is a proposed coherence test: non-exponential deviations that persist however many identities are added indicate coherent system-environment noise, and deviations that fade indicate dissipation (FigueroaRomero2021, body). *Q2*
- **K19 [lit]** For repeated QEC rounds, Markovian Pauli logical noise implies a straight line in log-polarization versus rounds, i.e. a constant per-round logical error rate computed from ratios of consecutive polarizations, so curvature is a direct witness of logical non-Markovianity (Ziyad2025, body). *Q2*
- **K20 [lit]** For error detection with postselection, the one- and two-cycle acceptance diagnostic equals `p_acc(2) - 2 p_acc(1) + 1 = -l + O(eps^2)`, where `l` is the "accepted leakage" into nontrivial-syndrome sectors, and a first-order `l` rules out any stationary logical-only description of the accepted shots (Kumar2026, body). *Q2*
- **K21 [lit]** Helsen et al. provide signal-processing methods with sample-complexity bounds for fitting sums of exponential decays, plus post-processing that isolates individual decays (Helsen2022). *Q2*

### 2.5 Leakage as a memory channel

- **K22 [lit]** Epstein et al. class leakage as non-Markovian because the noise at one moment depends on earlier gates, derive a two-exponential-plus-constant decay for it, and note that coherent leakage can produce collapse-and-revival in the qubit subspace (Epstein2014, body). *Q2, Q3*
- **K23 [lit]** In leakage RB the computational-subspace population decays as `A + B lambda1^m` with `lambda1 = 1 - L1 - L2` (leakage plus seepage), the survival probability is bi-exponential, and coherent leakage that twirling does not depolarize produces non-Markovian oscillations about the exponential (WoodGambetta2018, body). *Q2, Q3*
- **K24 [lit]** RB-type protocols can robustly estimate both coherent and incoherent leakage rates in a scalable way (Wallman2016). *Q3*
- **K25 [lit]** Significant leakage makes standard RB overestimate fidelity because the slope depends on the measurement operator; one remedy is leakage post-selection, where a gadget flags leaked shots, the retention rate gives the leakage parameter and the postselected decay gives the computational error (ChenBaldwin2025, body). *Q1, Q2, Q3*

### 2.6 QEC, syndrome memory, postselection and conditioning

- **K26 [lit]** Logical RB reduces to zeroth-order RB only if the logical noise is time-independent, Markovian and gate-independent at the logical-Clifford level and syndrome errors are uncorrelated between rounds, with each logical gate followed by error correction (with or without recovery) and an ancilla refresh (Combes2017, body). *Q1, Q2*
- **K27 [lit]** In LRB, syndrome qubits that are not reset can store a memory of the gate history, and a surjective decoder (e.g. minimum-weight decoding of a CSS code with d >= 3) feeds it back to the logical qubit, giving survival probabilities that oscillate between 0 and 1 under gate- and time-independent noise (Ceasura2022, body). *Q1, Q2*
- **K28 [lit]** Those oscillations decay when each syndrome qubit is reset with only 2% probability and survive high reset probability only when a classical repetition code is deliberately embedded in the syndrome register, and the authors expect realistic noise to produce milder failures (Ceasura2022, body). *Q2*
- **K29 [lit]** Logical qubits can be non-Markovian under purely Markovian physical noise, and this happens if and only if the data qubits are not necessarily returned to the code space after every QEC round, in which case the syndrome subsystem acts as a memory and the logical dynamics form a hidden Markov model (Ziyad2025, body). *Q2*
- **K30 [lit]** If every gate is followed by a perfect QEC round the logical gate set is composable (Markovian), so emergent logical non-Markovianity needs faulty QEC gadgets, for example ancilla readout errors that trigger wrong corrections and leave the data outside the code space (Ziyad2025, body). *Q1, Q2*
- **K31 [lit]** For the 3-qubit repetition and `[[5,1,3]]` codes with only syndrome-readout errors, the logical error rate is zero in round one, jumps, then relaxes exponentially fast to a constant, so non-Markovianity is strongest in the first rounds but never vanishes, and fault-tolerant syndrome extraction is expected to reduce but not remove it (Ziyad2025, body). *Q1, Q2*
- **K32 [lit]** Under Pauli-stochastic noise a QEC cycle updates the joint logical-syndrome state through a classical Markov chain on the syndrome register and syndrome-conditioned logical Pauli channels, independently of the logical state (Kwiatkowski2025, body, Prop. 1). *Q2*
- **K33 [lit]** If the cycle has syndrome-marginal independence (enforceable by syndrome randomization via Pauli twirling) and single-cycle logical entanglement fidelity of at least 1 - 1/64, repeated QEC cycles admit an approximate logical Markovian model whose deviations are exponentially suppressed in the number of cycles (Kwiatkowski2025, body, Thm. 1). *Q1, Q2, Q3*
- **K34 [lit]** In error detection with postselection, conditioning on acceptance makes the normalized state evolve nonlinearly, and imperfect checks produce false negatives ("accepted leakage" into nontrivial-syndrome sectors) at first order in check noise, while gate-induced syndrome leakage enters only at second order because ideal checks reject it (Kumar2026, body). *Q1, Q2*
- **K35 [lit]** In the same QED transition-matrix model the spectrum splits at small noise into slow accepted modes near 1 and fast leakage-rejection transients near 0, a first-order first-cycle transient appears, and less frequent checks improve the QED efficiency metric to leading order (Kumar2026, body). *Q1, Q2*
- **K36 [lit]** The first logical-RB experiment with an error-detecting code (`[[4,2,2]]` on IBM hardware, discarding odd-parity runs at the end) states that "there is at present no complete theory for fitting an RB decay curve with post-selection" (HarperFlammia2019, body). *Q2, Q3*
- **K37 [lit]** The Sdim paper defines detection-only LRB (LRB-D) as one terminal syndrome extraction followed by postselection on the all-zero syndrome, and warns that terminal detection can give decay curves with non-Markovian structure that a single exponential does not always summarize well (Kabir2025, body Sec. VII-A). *Q2*
- **K38 [lit]** Logical Pauli channels conditioned on a syndrome parity in detector regions can be estimated SPAM-robustly, as demonstrated on Quantinuum H1-1 with a small error-detecting code, where Pauli frame randomization and swapped measurements for leakage protection improved fault-tolerance diagnostics (Girling2025). *Q3*

### 2.7 Making RB behave Markovian

- **K39 [lit]** Fast dynamical decoupling turns non-Markovian RB into an exponential decay plus longer-time corrections, while randomized compiling leaves the average unchanged but always reduces the variance of RB outputs (FigueroaRomero2024). *Q2, Q3*

---

## 3. Implications for our qubit `[[5,1,2]]` LRB study

All items are **[conj]** (my inference from the literature above) unless a
different label is shown. Code facts (stabilizers, `X_L = X0 X3`,
`Z_L = Z0 Z2`, syndrome table) are from `../CONTEXT.md`, where they are
still marked "to be verified".

**I1. Acceptance does not depend on the logical state.** **[analytic]** In a
Clifford circuit with Pauli faults, every check outcome is a deterministic function
of the fault configuration: a Pauli error `E` either commutes or anticommutes
with each measured stabilizer, whatever the encoded state, and ancilla or readout
faults are classical flips. So the accepted/rejected record, and with it every
policy's acceptance, is independent of the logical input. This is the qubit-circuit
special case of the structure in K32 (Kwiatkowski2025, Prop. 1).

**I2. The leakage dictionary.** **[conj]** Kumar2026 already calls movement into
nontrivial-syndrome sectors "leakage". Taking that seriously, LRB-D maps onto
leakage RB:

| Leakage RB (K22 to K25) | LRB-D on `[[5,1,2]]` |
|---|---|
| computational subspace | code space (trivial syndrome) |
| leakage subspace | the 15 nontrivial syndrome sectors |
| leakage | a single detectable error, e.g. `Z0`, syndrome (1,0) on the X checks |
| seepage | a second error that restores the trivial syndrome, e.g. `Z2` after `Z0`, leaving `Z0 Z2 = Z_L`, which flips the `X_L` readout |
| leakage detection / LPS (ChenBaldwin2025) | rejection at a check whose outcome the policy uses |

On this reading, the accepted-shot signal needs at least two rates (K23). A
single-error seepage event (a second error with the same syndrome) carries a
logical flip about half the time: if the second error hits the partner qubit
(`Z0` then `Z2`) the result is `Z_L`, and if it hits the same qubit (`Z0` then
`Z0`) the error cancels. So the conditional fidelity should not be a single
exponential.

**I3. `const=1` and sparse `const=k` should be strongly non-Markovian.**
**[conj]** Between used checks the data are never returned to the code space,
which is exactly Ziyad's "only if" condition (K29). The intermediate checks are
physically present but their outcomes are ignored, so they add noise and give
no correction. Analytic sketch: the accumulated syndrome performs a random walk on
`Z_2^4` (4 stabilizers). Single-qubit X, Z and Y faults generate the whole group,
and logical Cliffords act on it as automorphisms, so the walk tends to the uniform
distribution. That gives terminal acceptance tending to 1/16 (rejection to 15/16),
the qubit analogue of the qutrit saturation at `1 - 1/81 = 1 - 3^-4` **[ctx]**.
The fidelity over accepted shots is a ratio `F(m) = N(m)/A(m)` of two sums of
exponentials (eigenmodes of the walk on syndrome x logical frame). It is flat while
single errors are filtered, falls once pairs of same-class errors start to
dominate the accepted mass, and tends to 0 as the logical frame of accepted shots
becomes uniform. This matches the shape of the qutrit `const=1` curve (0.96 to
0.05) **[ctx]**, and K36 and K37 already flag it as untheorized.

**I4. `unif=1` should be close to Markovian, apart from a first-round transient
and an O(p^2) memory.** **[conj]** With perfect checks, each accepted round
projects back to the code space, so Ziyad's condition holds and the accepted
logical process is a fixed trace-decreasing map per round. By K6's `B t^m` term
and I1, the normalized fidelity would then be close to a single exponential in
`p/t`. This assumes 2-design averaging and time-independent noise, and our
mirror-style inversion is not standard RB (see O2). SI1000 breaks the
perfect-check picture in two ways:

- readout flips `X_ERROR(5p)` are the largest single fault;
- checks are measured one ancilla at a time, so a data fault in a later
  measurement or reset layer of the same round is invisible to the checks that
  were already read.

Both produce Kumar's first-order "accepted leakage" (K34). Expect a first-round
transient followed by fast-mode decay (K35) and an almost constant F, which matches
the qutrit `unif=1` plateau near 0.95 **[ctx]**. The check-free inverse half is a
`const`-like segment of m logical gates carrying only gate noise (0.1p, p), so its
undetected same-class pairs are second order in gate noise.

**I5. What correction buys for Markovianity (Q2).** **[conj]** Per-round correction,
applied physically or in a Pauli frame, returns the state to the code space only
when the syndrome is right. Two effects need to be kept apart:

1. **Distance-2 guessing.** With d = 2, a correct syndrome for `Z0` or `Z2` still
   forces a coin-flip guess, so correction converts detectable errors into
   first-order logical errors. That is Markovian logical noise, just a lot of it.
2. **Readout errors.** A single readout error triggers a wrong correction and
   leaves the data outside the code space. This is Ziyad's mechanism for emergent
   non-Markovianity (K30, K31): strongest in the first rounds, then relaxing.

A decoder that uses consecutive rounds (detection events) addresses (2). K33
gives a sufficient condition for an approximately Markovian logical model:
single-cycle logical infidelity <= 1/64. A back-of-envelope estimate for our
model:

- each round puts about `5 x DEPOLARIZE1(2p)` on every data qubit (`../CONTEXT.md`);
- that gives about `7p` of Z-type error per qubit per round, and about the same
  of X-type, plus CNOT noise;
- four of the five data qubits sit in ambiguous pairs, so about 30p of each
  type lands on them, and about half of it is mis-corrected;
- that is about 15p of `Z_L` flips plus about 15p of `X_L` flips per round.

Since entanglement fidelity counts every logical Pauli, the single-cycle logical
infidelity is roughly 25-30p, which exceeds 1/64 somewhere around p = 5e-4 to
1e-3. So the theorem's sufficient condition probably fails over most of our p
grid. It
is only a sufficient condition, so this needs a direct numerical check, not
assumption. Net expectation: correction should remove the large `const`-policy
memory effect in I3, because it stops error pairs accumulating silently in
syndrome sectors. It replaces that effect with a larger but nearly Markovian
logical error rate plus a short readout-error-driven transient.

**I6. Fits and figures of merit.** **[conj]**

- The repo's `F = b f^m` fit (offset fixed at 0, grid scan) will return a
  single `f` even for the non-exponential `const` curves. K12 and K17 show that
  such an `f` depends on the depth window and can be off by large factors, so `f`
  should not be compared across policies whose curves have different shapes.
- Fidelity over accepted shots is a conditional quantity. Like the postselected
  decay in leakage RB (K25), it omits the rejected mass. Report the
  acceptance decay (a `t`-like rate) next to `f`, and compare correction (no
  rejection) with detection on that combined basis.

**I7. Cheap diagnostics on data we already have or will produce.** **[conj]**
None of these needs heavy compute.

1. **Local decay rates.** Compute `f_loc(m_i) = (F(m_{i+1})/F(m_i))^(1/(m_{i+1}-m_i))`
   over the depth grid. A constant `f_loc` is the K19-style Markovian signature;
   a drifting one is a witness of memory.
2. **One rate or two.** Compare `b f^m` with `b1 f1^m + b2 f2^m` by AIC or BIC
   (K17).
3. **Acceptance test.** Adapt Kumar's diagnostic (K20) to the acceptance curves:
   for a stationary per-round rejection, `A(m+2)/A(m)` is constant.
4. **Spread across sequences.** Inspect the distribution of per-sequence F over
   the 30 sequences (K13). In our Markovian simulation, any spread comes from
   gate dependence; on hardware it also probes temporal correlations.

**I8. Hardware (Q3).** **[conj]** Real devices add physical memory on top of the
emergent syndrome memory, such as transmon leakage (K22, K23, K25) and slow
drift or 1/f noise (K11, K14). The literature already offers matching tools:

- leakage-flagging gadgets with postselection (K25);
- swapped measurements and Pauli frame randomization (K38);
- randomized compiling and dynamical decoupling to Markovianize (K39);
- syndrome randomization to meet K33's assumptions.

A sustainable protocol would therefore keep active ancilla reset every round (the
repo already does), decode every round with a multi-round decoder (no silent
accumulation), randomize the Pauli frame of logical gates, and publish
acceptance alongside fidelity. K15 warns that an exponential-looking hardware
curve still does not certify Markovianity.

---

## 4. Seed corrections

| Seed as given | Status | Correction |
|---|---|---|
| Wallman, "Randomized benchmarking with gate-dependent noise", arXiv:1703.09835 | correct | Published Quantum 2, 47 (2018), doi:10.22331/q-2018-01-29-47. |
| Proctor et al., "What randomized benchmarking actually measures", arXiv:1702.01853 | correct | Full authors Proctor, Rudinger, Young, Sarovar, Blume-Kohout; PRL 119, 130502 (2017). |
| Figueroa-Romero, Modi, **Harris**, Stace, Hsieh, "Randomized benchmarking for non-Markovian noise", arXiv:2107.05403 | **author list wrong** | There is no "Harris". The authors are Figueroa-Romero, Modi, Stace, Hsieh. PRX Quantum 2, 040351 (2021). |
| Figueroa-Romero, Modi, Hsieh, "Towards a general framework of randomized benchmarking incorporating non-Markovian noise", arXiv:2202.11338 | correct | Quantum 6, 868 (2022). |
| Ball, Stace, Flammia, Biercuk, "Effect of noise correlations on randomized benchmarking", arXiv:1504.05307 | minor | Exact title begins "The effect of ..."; PRA 93, 022303 (2016). |
| Fong and Merkel, "Randomized benchmarking, correlated noise, and Ising models", arXiv:1703.09747 | correct | No journal reference on arXiv. |
| Mavadia et al. temporally correlated noise experiment, arXiv:1706.03787 | correct | Title "Experimental quantum verification in the presence of temporally correlated noise"; npj Quantum Inf. 4, 7 (2018). |
| Pollock et al., process tensor framework, arXiv:1512.00589 | correct | Title "Non-Markovian quantum processes: complete framework and efficient characterisation"; PRA 97, 012127 (2018). The companion "Operational Markov condition for quantum processes" is arXiv:1801.09811 (PRL 120, 040405), fetched but not cited. |
| Wood and Gambetta leakage RB, arXiv:1704.03081 | correct ID, title clarified | Title "Quantification and Characterization of Leakage Errors"; PRA 97, 032306 (2018). |
| Wallman et al. leakage characterization, arXiv:1412.4126 | correct ID, **title differs by version** | The arXiv listing reads "Characterization of Leakage Errors via Randomized Benchmarking". The published title is "Robust characterization of leakage errors", NJP 18, 043021 (2016), by Wallman, Barnhill, Emerson. |

---

## 5. Open questions

1. **No theory for postselected RB.** No complete theory exists for RB decays
   under postselection (K36). Can a Kumar-style transition matrix (K34, K35) over
   syndrome sector x logical frame x accept/reject, built for our circuits and SI1000,
   reproduce the qutrit `const=1` curve and predict the qubit one exactly? That
   would turn I3 from conjecture into verified-analytic.
2. **Mirror-style inversion.** The forward half has a check after every Clifford,
   while the reverse half inverts each Clifford individually with no checks. I
   found no literature on RB theory for this protocol, with or without
   postselection. Does it break the 2-design averaging behind K5 and K6, so that
   gate dependence (`H_L = SWAP(2,3)` + 5 H versus `S_L` with a CZ) leaves larger
   multi-exponential terms than in standard RB?
3. **When is correction approximately Markovian?** For d = 2 correction, does a
   multi-round decoder satisfy Kwiatkowski's syndrome-marginal-independence
   property, and at what p does the logical process stop being approximately
   Markovian (I5 guesses that the sufficient condition fails around p = 5e-4 to 1e-3)? Is the
   readout-error transient (K31) visible at our depths m <= 22?
4. **A fair figure of merit.** What metric compares correction (full acceptance,
   first-order logical errors) with detection (exponentially falling acceptance,
   second-order logical errors) without the nonlinearity of conditional fidelity?
   Candidates include Kumar's efficiency metric (K35) and a ChenBaldwin2025-style
   split into retention `t` and conditional decay.
5. **Definitions disagree.** Endo2025 (negativity from syndrome feedback) and
   Ziyad2025 (button-theoretic) can give opposite answers to "does correction
   remove non-Markovianity?". Which notion matters operationally for LRB
   decay fits? K19 and K20 are operational.
6. **Blind spots.** Could some policies, for example `unif=1`, hide the emergent
   memory in the ASF while it still raises worst-case logical error (the K15
   blind-spot scenario)?
7. **Hardware leakage and postselection.** On hardware, how do leakage events
   interact with syndrome postselection? Leaked data qubits can randomize checks
   and get rejected (helpful), or hide for several rounds and seep back (memory).
   No source I found treats leakage and code-level postselection together in an
   LRB setting.

---

## 6. Search log

Queries run (WebSearch, standard mode), in order:

1. non-Markovian randomized benchmarking process tensor average sequence fidelity
2. logical randomized benchmarking non-exponential decay error correction syndrome memory
3. randomized benchmarking postselection error detection heralded conditional fidelity decay
4. operational Markovianization randomized benchmarking Figueroa-Romero
5. leakage randomized benchmarking overestimate fidelity Chen Baldwin Quantinuum Physical Review Research 2025
6. gate-dependent noise randomized benchmarking Fourier convolution Merkel Pritchett Fong
7. Kwiatkowski approximate logical Markovian model consecutive QEC cycles stabilizer code arXiv 2509.16887
8. leakage non-Markovian error memory qubit subspace randomized benchmarking double exponential decay seepage
9. randomized benchmarking temporally correlated noise non-exponential decay 2024 OR 2025 OR 2026 quasistatic 1/f
10. testing for non-exponential decay randomized benchmarking model selection goodness of fit multi-exponential statistical analysis
11. post-selection induces non-Markovianity conditional quantum dynamics heralding memory process tensor
12. non-Markovian noise quantum error correction surface code memory process tensor decoder 2024 2025
13. erasure conversion randomized benchmarking postselected fidelity mid-circuit erasure check decay dual-rail
14. Brillant Groszkowski Seif Koch Clerk randomized benchmarking non-Markovian dephasing finite-duration gates arXiv
15. logical randomized benchmarking error detection code postselection experiment [[4,2,2]] decay acceptance rate trapped ion superconducting 2024 2025

Full texts of Ceasura2022, Ziyad2025 and Kwiatkowski2025 were read from their
arXiv PDFs with a standard-library text extractor in the session scratchpad; no
packages were installed.
