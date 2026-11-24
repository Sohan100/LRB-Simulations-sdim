# H2: other hardware platforms and real-time control for correction-based qutrit LRB

Agent H2, 2026-10-28. Scope: platforms other than superconducting transmon
qutrits for running correction-based LRB on the 9-qutrit `[[5,1,2]]_3` folded
block of `00-brief.md`, and the classical control stacks that real-time
decoding would need. Transmon qutrits appear only as a reference row.
`H1-sc-hardware.md` covers them in depth, and `lit-L5-realtime.md` and
`A3-pauli-frame-offline-causal.md` cover offline versus real-time decoding in
general. This note uses their results only as cross-references and labels them
that way.

This file replaces an earlier H2 draft from the same day. Section 8 lists what
changed and why. The two top-level scripts `H2_platform_budget.py` and
`H2_qutrit_algebra_checks.py` belong to that draft and are superseded. I left
them unmodified.

## Scripts and how to reproduce

The scripts sit in `notes/H2-scripts/`. Run each one after
`module load python/3.11`. Each script's stdout is saved next to it as
`*_output.txt`.

| Script | What it does | Run time |
|---|---|---|
| `h2_algebra_checks.py` | Exact checks A1 to A8 with sdim's own gate matrices and the repository's gate parser: resource counts, the light-shift gate decomposition, the interaction graph, Weyl normalization, syndrome transfer maps, the qutrit T gate, no-reset extraction on the full 9-qutrit block, lookup-table sizes, shot-noise formulas | 5 s, one core |
| `h2_platform_noise_study.py` | Exact LRB-D and sampled maximum-likelihood (MLD) correction on the baseline circuits under nine noise models. Loads, read-only, the chain structures cached by the decoder study (`.cache_correction/structures`) and the matching baseline circuits. Validates against the baseline CSVs and the decoder study's results. Writes `h2_platform_noise_study.csv` | 2 min, two cores |
| `h2_budget.py` | Time and shot budgets, decoder-latency ratios, photonic success probability, and the Rayleigh-floor check on the baseline CSVs | < 5 s |

Other files: PDFs were read with the existing worktree venv `.venv-h2pdf/`
(pypdf), which the earlier draft created; I installed nothing. Downloaded
papers went to my scratchpad.

Labels on every claim: **[VA]** verified analytically, **[VN]** verified
numerically (the script block is named), **[LIT key]** from a source I fetched
in this task (Section 9), **[CONJ]** conjecture. Vendor web pages are vendor
claims, not peer-reviewed measurements. Absence claims ("not demonstrated")
are conclusions from my searches on 2026-10-28 and are labelled [CONJ] unless a
fetched source states them.

Conventions. `r = 2(1 - f)/3` per depth step, as in the brief's fit. One depth
step is one forward logical Clifford with its check round plus one inverse
logical Clifford. `lambda = Re E[w^R]` is the signed fidelity of the residual
logical shift `R`. `a` is the acceptance factor per depth step.

## 0. Summary

1. **The binding constraint is the error per check round, not decoding
   speed.** A round has 12 two-qutrit gates, 4 one-qutrit gates and 4
   mid-circuit qutrit measurements. A depth-22 circuit holds about 413
   two-qutrit gates and 92 mid-circuit measurements [VN, A1 for the counts;
   CONJ for "binding", argued in Sections 3 to 5].
2. **Platform-specific noise matters a lot.** I applied ion-like noise
   location by location to the real baseline circuits:
   - two-qutrit gates from Hrmo et al.;
   - single-qutrit Clifford RB and qutrit readout from Ringbauer et al.;
   - the measured mid-circuit spectator error of Yu et al.;
   - one parallel camera readout per round.

   This `ion_now` model behaves like SI1000 at `p ≈ 3e-3 to 5e-3`, not at the
   `p ≈ 0.014` that a two-qutrit-gate-only mapping suggests. Its unif-1
   acceptance is `a = 0.67` per step. MLD correction gives `r = 0.078`. The
   conditional LRB-D error is `r = 7.2e-4` [VN, `h2_platform_noise_study.py`
   R2; inputs LIT H23, R22, Y25, M25 and CONJ].
3. **Correction is below break-even on every platform model; detection is
   not.** MLD correction gives `r = 0.018` to `0.14` per depth step. The
   physical reference is about `4e-3` per step for ions, two single-qutrit
   Cliffords at `2(2)e-3` each [LIT R22]. The conditional unif-1 rate on the
   ion models (`3.3e-5` to `1.5e-3`) is below that reference, but it costs
   rejection. The exception is a worst-case idle-dephasing model, at
   `4.8e-3` [VN, R2]. This is the same picture the decoder study finds at
   SI1000 `p = 1e-3`, which I reproduce [VN, V1].
4. **Correction is statistically cheap; detection is not.** For ±10% (one
   standard error) on `r` from depths 0 and `M` with 30 sequences:
   - MLD correction needs about 15 to 40 shots per circuit, minutes of ion
     beam time;
   - LRB-D unif 1 needs about `1e5` shots per circuit, about 117 h on native
     ion qutrits for two depths;
   - const 1 needs about 600 shots per circuit.

   The spread between Clifford sequences adds at most 4% [VN, `h2_budget.py`
   T3].
5. **Trapped-ion native qudits have the best published qutrit operations,**
   with one caveat on metrics: the LS fidelity is a SPAM-corrected per-gate
   decay and the transmon figures are process fidelities.
   - LS two-qutrit gate 98.7(2)% on two ions;
   - qutrit Clifford RB `2(2)e-3`;
   - worst-case qutrit readout error about `3e-3`;
   - about 100 ms coherence;
   - all-to-all connectivity;
   - a built-in leakage herald at readout.

   Two pieces are missing for this experiment: addressed native qutrit
   entangling gates in a 9-ion register, and mid-circuit qutrit readout with
   protected spectators. A qutrit readout takes about 4.5 ms [LIT R22, H23,
   M25, L23; absence CONJ].
6. **Apart from transmon qutrits, qubit-pair-encoded qutrits on a commercial
   QCCD ion machine are the only option that is realistic today.** Iqbal et
   al. ran 24 qutrits on Quantinuum H2 with mid-circuit measurement, real-time
   feed-forward and a leakage herald [LIT I25]. With their gate counts (about 10 two-qubit gates per
   qutrit CNOT) and `2e-3` per two-qubit gate, the per-round performance
   matches `ion_now` (`a = 0.66`, `r_MLD = 0.079`) [VN, R2; inputs CONJ].
7. **Neutral atoms, photonic, bosonic and spin or molecular qudits cannot host
   this 9-qutrit check-round circuit today.**
   - Neutral atoms: a 2026 paper states that no multi-qudit control in atom
     arrays has been demonstrated [LIT Bu26].
   - Photonic: two-qudit gates between photons succeed with probability
     `1/d`, which gives `10^-197` for a depth-22 shot at `d = 3` [LIT C22; VN
     T5].
   - Bosonic: GKP qudits are single-mode memories [LIT Br25].
   - Spin and molecular: the qudits are single objects [LIT F24, Yu25, Go17].
8. **Every gate in the circuit normalizes the qutrit Weyl group, and nothing
   is classically controlled.** Corrections are therefore an offline `Z_3`
   shift of the terminal outcome. The control system only has to store
   `4(m+1)` trits per shot. Reset can be dropped, because an ancilla that
   starts in `|k>` returns `k + s` [VA; VN A4, A6; LIT T15].
9. **Non-Clifford logical gates will need real-time decoding.** An X-type
   frame error before the qutrit T gate becomes a non-Weyl Clifford error,
   which a relabelling of the final readout cannot undo, and injection
   gadgets apply outcome-dependent Clifford corrections [VN A5; LIT T15,
   CAB12, HV12].
10. **Measured feedback and decoding latencies:**
    - FPGA loops: 0.2 to 0.55 µs (QICK, QubiC, Zurich Instruments, Qblox,
      an OPX-family controller).
    - FPGA decoder integrated in a Rigetti control system: 9.6 µs for 9
      rounds.
    - GPU links: about 4 µs.
    - GPU or workstation decoders: 63 to 67 µs (Google, Quantinuum Helios
      with NVIDIA).

    All of these are at most 1% of a 9.6 ms ion check round. On a 6 µs
    transmon round, GPU decoders take about 10 rounds [LIT St21, X23, ZI-FB,
    ZI-PQSC, QB, Cao26, Ca24, NV25b, A25; VN T4].
11. **Conclusion:** trapped ions are the most sustainable platform for this
    experiment. Start with qubit-pair qutrits on a QCCD machine, which is
    possible today. Move to native `40Ca+` qutrits once the two missing
    demonstrations exist.

    Use offline decoding, one parallel readout layer per round, no reset,
    and correction or sparse checks rather than `unif 1` postselection.
    Transmons remain the high-throughput option but have higher error per
    round (Section 6) [CONJ, built on items 2 to 10].

## 1. What the experiment asks of hardware

### 1.1 Counts

Counts come from the repository's own parser `elementary_single_qudit_logical_operator_gate_sequence`
and from the 216-entry catalog [VN, A1]:

- Physical content per logical token, as (two-qutrit, one-qutrit) gates:
  - H: (1, 5), namely `SWAP(2,3)`, H on {0,2,3,4} and `H_INV` on 1;
  - P: (1, 3), namely P on {0,4}, `P_INV` on 1 and `CZ(2,3)`;
  - X: (0, 2); Z: (0, 2); I: (0, 1).
- Catalog means per Clifford: 1.653 H, 1.463 P, 1.005 X, 1.005 Z and 0.005 I
  tokens. That gives 3.116 two-qutrit and 16.676 one-qutrit gates per logical
  Clifford.
  - If `SWAP(2,3)` is done by relabelling (Section 1.3), only `CZ(2,3)` is
    left: 1.463 two-qutrit gates.
  - If logical X and Z are done as frame updates (they are transversal
    Paulis), 12.657 one-qutrit gates are left.
  - The catalog has 216 entries but only 207 distinct strings. That matters
    for sampling and is outside this note's scope.
- One check round: 12 two-qutrit gates, 4 one-qutrit gates, 4 measurements
  and 4 resets.
- Depth `m = 22`: 44 Cliffords and 23 rounds, which is on average 413.1
  two-qutrit gates, 825.7 one-qutrit gates and 92 mid-circuit measurements.

### 1.2 SI1000 fault weight per round

The SI1000 weight is 118.4 p per round [VA from the brief; VN from the stored
circuits, `h2_platform_noise_study.py` V1]:

| Component | Weight per round |
|---|---|
| Gates | 12.4 p |
| Four sequential single-ancilla readout layers | 88 p |
| Reset layer | 18 p |

The stored circuits give, by location class:

| Class | Weight per round |
|---|---|
| idle of the other ancillas during readout | 24 p |
| idle of data in the first three readout layers | 30 p |
| idle of data in the last readout layer | 10 p |
| measured-ancilla depolarizing | 4 p |
| measurement flips | 20 p |

With one parallel readout layer the total drops to 64.4 p. Each data qutrit
then sees 4 p of idle noise per round instead of 10 p.

Camera readout of ions and atoms reads all ancillas at once, so all platform
models in Section 3 use one parallel readout layer [CONJ, modelling choice].

### 1.3 Interaction graph and routing

These results come from block A3 [VN]:

- The 12 check couplings form a graph isomorphic to the 3x3 square grid. It
  has 8 embeddings, and data 1 always sits at the centre.
- The logical edge `(2,3)`, used by `SWAP(2,3)` and `CZ(2,3)`, closes the odd
  5-cycle `0-7-2-3-5-0`. The full graph is therefore not bipartite: it is
  planar with maximum degree 4, but it fits no square or other bipartite
  lattice.
- In every grid embedding, data 2 and 3 are at distance 4.
- All six shortest 2–3 paths pass through another data qutrit (0, 1 or 4).
  Routing `CZ(2,3)` on a fixed grid therefore needs at least three
  intermediate sites and touches a third data qutrit. H1 treats the transmon
  layout.
- On an all-to-all register, which includes an ion chain, a QCCD processor
  and an atom array with transport, the full graph embeds with no extra gates
  [VA]. `SWAP(2,3)` becomes a relabelling, after which the next round's
  checks couple to the relabelled wires [VA; H1 reaches the same
  conclusion].

### 1.4 What runs between measurements

- Every gate in the circuits normalizes the qutrit Weyl group. The images of
  all 9 one-qutrit and 81 two-qutrit Weyl operators were extracted from
  sdim's matrices for H, H_INV, P, P_INV, X, X_INV, Z, Z_INV, I, CNOT,
  CNOT_INV, CZ, CZ_INV and SWAP [VN, A4].
- No gate is conditioned on a measurement outcome [VA, brief and circuit
  generator].
- Logical tokens act on syndromes by fixed 4x4 transfer matrices over `Z_3`
  [VN, A4]:
  - logical H swaps the Z-type and X-type syndromes with a sign,
    `T_H = [[0,0,1,0],[0,0,0,1],[2,0,0,0],[0,2,0,0]]`;
  - logical P adds `-s_Z` to `s_X`;
  - X and Z act trivially.

  A real-time frame tracker or decoder therefore needs the Clifford sequence.
  The sequence is fixed before the run, so this costs 216 small matrices
  [VA].
- **Reset is optional.** On the full 9-qutrit encoded block (statevector, 25
  random data Weyl errors x 4 ancilla starts), an ancilla that starts in
  `|k>` instead of `|0>` returns `k + s(E)` deterministically for all four
  checks. The sign matches the symplectic syndrome for every check
  [VN, A6]. Successive raw outcomes can therefore be differenced in software.
  Gehér et al. find that resetting gives no benefit in memory experiments.
  No-reset wins when reset takes longer than about 100 ns and the physical
  error exceeds about `10^-2.5` [LIT Ge25].
- Without mid-circuit measurement, depth `m` would need `4(m+1)` fresh
  ancillas, 92 at `m = 22` [VA]. Mid-circuit qutrit readout is therefore a
  hard requirement.

## 2. Platform assessment

### 2.1 Trapped-ion native qudits (40Ca+, 137Ba+)

**Encoding and single-qudit gates [LIT R22].**

- Ringbauer et al. encode qudits in the S1/2 and D5/2 Zeeman sublevels of
  40Ca+. That gives 8 levels and qudits up to `d = 7`, because one level is
  kept free for readout.
- With magnetic shielding, coherence on the order of 100 ms is reached on all
  transitions.
- Qutrit Clifford RB gives `2(2)e-3` per Clifford, which is `2.0e-4` per
  laser pulse.
- The AQT demonstrator quotes 15 µs single-qubit and about 200 µs two-qubit
  operations [LIT P21].

**Two-qutrit gates.**

- Gates built from MS gates and embedded in qudit space: `Cex` reaches
  97.5(2)% with 2 MS gates, `Cinc` 93.8(2)% with `2d` MS gates [LIT R22].
- The native light-shift (LS) gate `G(theta)` maps `|jj> -> |jj>` and
  `|jk> -> e^{i theta}|jk>` [LIT H23]:
  - it reaches 99.6(1)%, 98.7(2)%, 97.0(2)% and 93.7(3)% for
    `d = 2, 3, 4, 5`;
  - it was shown on two ions with global 401 nm beams, using LS pulses of
    about 35 µs interleaved with cyclic permutations;
  - the fidelities are "SPAM corrected average gate performance over a
    sequence of length n";
  - the authors state that slow non-Markovian noise makes repeated-gate
    decays non-exponential;
  - the simulated `d = 3` budget totals `1.3e-2`. Its largest items are gate
    Rabi frequency `4e-3`, local-operation frequency noise `2.3e-3`, motional
    coherence `2.2e-3` and gate-laser frequency noise `1.8e-3`;
  - D5/2 decay is listed with a 1 s lifetime.

**From the LS gate to the check CNOT [VA, VN A2].**

- At `theta = ±2pi/3`, `G = (D x D) CZ^{±1}` exactly, with
  `D = diag(w^{j^2}) = diag(1, w, w) = Z P^2`. This holds because
  `1 - delta_jk = (j - k)^2 mod 3`.
- Hence `CNOT = (I x H^dag)(D^dag x D^dag) G(+2pi/3) (I x H)` and
  `CNOT_INV = (I x H^dag)(D x D) G(-2pi/3) (I x H)`.
- Each check CNOT therefore costs one native LS gate plus local Cliffords on
  both qutrits: `D^dag` on the control, and `H` before and `H^dag D^dag`
  after on the target.
- Without the `D` corrections the product is not a CNOT. Whether `D` can be
  applied as a virtual phase update on ions is a [CONJ]. Ringbauer et al.
  note that two-level phase gauge freedom is lost for qudits [LIT R22].

**Mapping to a depolarizing parameter [VA].**

- For two-qutrit depolarizing noise with non-identity probability `q`, both
  the average gate fidelity and the fidelity of any pure state equal
  `1 - 0.9 q`. This follows from `sum_W |<psi|W|psi>|^2 = 9` over the 81 Weyl
  operators.
- For one qutrit the relation is `1 - 0.75 q`.
- So 98.7% corresponds to `q ≈ 0.0144`, and a Clifford error of `2e-3` to
  `q ≈ 2.7e-3`.
- Applying this to Hrmo's SPAM-corrected per-gate decay is a [CONJ].

**Register size and addressing.**

- Ringbauer et al. address ion subsets with focused beams. Their addressed
  entangling gates (MS gates) reach 0.993 ± 0.002 in a 3-ion string and
  0.983 ± 0.002 in a 10-ion string, with crosstalk suppressed to about
  `2e-3` [LIT R22].
- Meth et al. ran one qutrit plus four qubits with mixed-dimensional
  entangling gates in an all-to-all linear processor [LIT M25].
- Shi et al. used multi-tone control of a single trapped-ion qudit with up
  to 8 levels to run Grover search, without entangling gates [LIT S26].
- I found no register of 9 qutrits with addressed native qutrit entangling
  gates [CONJ, absence].

**Readout, mid-circuit measurement and their cost.**

- Fluorescence separates only S from D, so qudit readout shelves and detects
  sequentially [LIT R22]:
  - a qutrit needs two detections;
  - with 500 µs detection and 2500 µs of polarization-gradient recooling
    between detections, the worst-case qutrit readout error is about
    `3e-3`. The paper writes "readout error of ~99.7%", which must mean
    fidelity [LIT R22];
  - Meth et al. quote `(d - 2) 3.5 ms + 1 ms`, which is 4.5 ms for a qutrit.
    Each detection event is a 1 ms camera window followed by 2 ms of Doppler
    and 0.5 ms of polarization-gradient cooling [LIT M25].
- In their encoding, `d > 2` qudits live only in D5/2, with S1/2 as an
  auxiliary level [LIT M25]. The data qutrits are then dark during the
  ancillas' fluorescence and recooling. The remaining spectator errors are
  dephasing and D5/2 decay. At a 1 s lifetime [LIT H23], decay alone costs
  about `4.5e-3` per 4.5 ms readout [VA arithmetic; whether this suffices
  in practice is CONJ].
- Faster readout: high-NA optics detect qubits in 300 µs at about 99.9%, with
  negligible detection crosstalk [LIT P21]. Low et al. project qudit readout
  on the order of 1 ms or better [LIT L23].
- No mid-circuit measurement on a multi-qudit ion register was found
  [CONJ, absence]. The nearest data come from qubits:
  - in-situ mid-circuit measurement and reset (MCMR) in single-species
    `171Yb+` gives a data-qubit fidelity of 98.8(6)% or 98.2(7)% with the
    shelving method and 91.7(4)% with dressing. Shelving takes 26 µs each
    way and scattering 140 µs. Hands-off reset costs 0.9(3)% on the measured
    ion and 3.7(5)% on the data ion. The authors project below 0.1% with
    lower laser noise [LIT Y25];
  - the omg architecture reads metastable qubits through a ground-state
    qubit of the same species while cooling mid-circuit. I read only the
    abstract [LIT B26];
  - Quantinuum's 2021 QCCD QEC cycle took less than 200 ms, of which less
    than 70 ms was transport and the rest mostly cooling [LIT RA21].

**Leakage and erasure heralding.**

- In sequential readout, population outside the encoded levels never
  fluoresces. Low et al. treat "no bright state" as a directly detectable
  failure and remove those shots: the raw SPAM error of 13.1% becomes 8.3%
  postselected for 13 levels [LIT L23].
- Ringbauer's scheme repeats detection until every ion has been bright once
  [LIT R22].
- The herald covers only measured qudits: the ancillas in every round and the
  data at the end. Decay of a data qutrit's D5/2 population into S1/2 between
  readouts is heralded only at the final readout [CONJ].

**Verdict.**

- A 9-qutrit check-round circuit is not shown today. It needs addressed
  native LS gates in a chain of at least 9 ions and parallel mid-circuit
  qutrit readout with protected data [CONJ].
- Every ingredient exists in some ion experiment, and the gate, readout and
  coherence numbers are the best among native-qutrit platforms (Section 2.8)
  [LIT R22, H23, M25, P21].

### 2.2 Qubit-pair-encoded qutrits on QCCD ions (Quantinuum H2 and Helios)

These facts are from Iqbal et al. [LIT I25]:

- They prepared a `Z_3` toric code on 24 qutrits on Quantinuum H2, which has
  56 "effectively all-to-all connected qubits, with two-qubit gate fidelities
  exceeding 99.8%".
- Each qutrit is two qubits: `|0> = |00>`, `|1> = |10>`, `|2> = |11>`. The
  unused `|01>` heralds leakage, and about 11% of shots were discarded.
- Qutrit measurements were followed by real-time feed-forward.
- Native compilation:
  - qutrit Fourier: 3 ZZPhase gates;
  - controlled-Z: 4 controlled-RZ gates;
  - qutrit X: 1 ZZPhase gate (or two CNOTs);
  - charge conjugation: 1 ZZPhase gate.
- The fidelity per qutrit exceeds 96.5(3)%.

My estimates from these counts [CONJ]:

- a check CNOT is `F^dag CZ F`, about 10 two-qubit gates, so about `2e-2` at
  `2e-3` each;
- logical `CZ(2,3)` is about `8e-3`;
- a Fourier-type one-qutrit gate is about `6e-3`.

Verdict: realistic today. Every ingredient has been run at a larger scale.
The cost is about 10 native gates per qutrit CNOT and slow rounds. My guess is
of order 0.1 s per round, from the QCCD cycle times in RA21 [CONJ].

### 2.3 Rydberg and neutral-atom qudits

- **Qudit gates.** Burshtein et al. give a robust CZ recipe for any `d`,
  analysed for alkaline-earth qutrits. Their abstract states that no
  experimental demonstration of multi-qudit control in neutral atom arrays
  has been achieved to date [LIT Bu26]. My searches found none since
  [CONJ].
- **Qubit hardware.**
  - Parallel two-qubit gates reach 99.5% on up to 60 atoms [LIT E23].
  - Mid-circuit readout uses 500 µs imaging at about 99.8% fidelity, and
    data decoherence stays at or below 0.1% per 500 µs exposure. An FPGA
    (Xilinx ZCU102) does real-time image processing, decoding and
    feedforward, with a full readout-plus-feedforward cycle of slightly less
    than 1 ms [LIT Bl24].
  - In that work the Clifford-circuit corrections were "applied in software
    (with the exception of Fig. 4e, which is decoded real-time)" [LIT Bl24].
  - Coherence: `T2 > 1 s` [LIT Bl24].
- **Erasure heralding.** Metastable `171Yb` reaches 0.9990(1) single-qubit
  and 0.980(1) two-qubit fidelity. Mid-circuit erasure detection disturbs the
  remaining qubits by less than `1e-5` [LIT Ma23]. The original proposal
  estimates that 98% of errors can be converted to erasures and that the
  surface-code threshold rises from 0.937% to 4.15% [LIT W22]. A qutrit
  encoding in such atoms could inherit this heralding [CONJ].
- **Connectivity.** Atom transport gives arbitrary connectivity, so the odd
  cycle of Section 1.3 costs only movement time [VA for the graph; CONJ for
  the cost].
- **Verdict.** Not realistic for native qutrits until a two-qudit gate is
  demonstrated. A qubit-pair emulation as in 2.2 is possible in principle
  [CONJ].

### 2.4 Photonic qudits

- **The Chi et al. processor** is a silicon-photonic chip with two
  path-encoded ququarts [LIT C22]:
  - single-qudit gates reach classical fidelities of 0.988(13) for `X_4` and
    0.967(19) for `F_4`;
  - the multi-value controlled-X gate reaches a process fidelity of 0.952;
  - multiqudit controlled gates work "with a (1/d) success probability
    regardless of n";
  - the two-photon detection rate is about `10^3/s`;
  - the authors say scaling needs heralded multiplexed photon-pair sources;
  - outcomes "can be feed-forwarded into the d-QPU" for Kitaev's phase
    estimation. The text does not say whether this was done in real time.
- **Deterministic gates inside one photon.** Imany et al. built deterministic
  two-qudit gates between the time and frequency degrees of freedom of a
  single photon, with fidelities above 0.90 in the computational basis.
  They note that high-dimensional coding alone does not scale to universal
  computing [LIT Im19].
- **Shot success probability [VN T5; the d = 3 rate is CONJ].** At `1/3`
  success per two-qutrit gate, one check round succeeds with probability
  `1.9e-6` and a depth-22 shot (about 413 gates) with `10^-197`.
- **Measurement.** Photon detection is destructive, so ancillas cannot be
  measured mid-circuit and reused without fresh photons [CONJ, no fetched
  source].
- **Verdict:** not realistic.

### 2.5 Bosonic cavity qudits (GKP)

- **The Brock et al. experiment** error-corrected a GKP qutrit and a GKP
  ququart in one 3D cavity with a tantalum transmon ancilla [LIT Br25]:
  - gains beyond break-even are 1.82 ± 0.03 and 1.87 ± 0.03;
  - the qutrit logical lifetime is 886 ± 3 µs, against 488 ± 7 µs for the
    best Fock qutrit;
  - each stabilization round takes 7 µs. Of this, 3632 ns go to ancilla
    measurement and reset and 224 ns to FPGA sequencing;
  - a ternary Pauli measurement uses two binary ancilla measurements;
  - the dominant error is cavity dephasing from the transmon's 2.2%
    thermal population.
- **The same group's qubit work** ran a real-time QEC cycle of 2 x 4.924 µs
  on custom Xilinx Virtex-6 FPGA cards, "including ... real-time processing
  and decision making by the controller". A virtual oscillator rotation took
  448 ns [LIT Si23].
- **Two-mode logical gates.** The only one I found is for GKP qubits in the
  motion of one trapped ion, with a logical process fidelity of 0.680 [LIT
  Mt24].
- **What our circuit would need:** nine GKP qutrit modes with stabilization
  and two-mode CSUM gates between GKP qutrits. Neither is demonstrated
  [CONJ].
- **Verdict:** not realistic now. GKP qutrits are a candidate inner code for
  a later concatenated scheme [CONJ].

### 2.6 Spin and molecular qudits

- A single Sb donor in silicon gives a 16-dimensional electron-nuclear qudit
  controlled by electric and magnetic fields, with nuclear gate fidelity
  above 99.8% [LIT F24].
- Cat states of the spin-7/2 Sb nucleus act as an encoded qubit with logical
  Paulis implemented by multi-frequency control [LIT Yu25].
- Grover search ran on the nuclear spin 3/2 of a single-molecule-magnet
  transistor using electric fields only [LIT Go17].
- These are single objects. I found no multi-qudit register with
  entangling gates and mid-circuit readout [CONJ, absence].
- **Verdict:** not realistic for this experiment.

### 2.7 Reference: transmon qutrits

- Two-qutrit gates: CZ† reaches 97.3(1)% and CZ 95.2(3)% process fidelity
  [LIT Go22]. The gate times are 580 ns and 783 ns, and the Ramsey `T2` of
  the 1–2 manifold is 17(8) µs in the CZ pair [LIT Go22, full text].
- Single-qutrit gate infidelity goes as low as `2.38e-3`, and a CSUM gate
  reached 0.82 [LIT Mo21].
- An FPGA machine-learning classifier on QubiC reaches 97% qutrit
  classification from a 500 ns readout pulse, with 66 ns inference latency
  [LIT X23].
- Mid-circuit readout is routine. The fixed grid needs routing for
  `CZ(2,3)`, and idle noise is large; H1 covers both.

### 2.8 Comparison table

| Platform | Best two-qutrit gate | Mid-circuit readout | Connectivity for this circuit | Leakage or erasure herald | 9-qutrit check rounds today? |
|---|---|---|---|---|---|
| 40Ca+ native qudits | LS 98.7(2)%, 2 ions, global beams [H23] | qutrit ~3e-3 error, 3.5 to 4.5 ms [R22, M25]; not shown mid-circuit for qudits | all-to-all; no routing [VA] | "never bright" at readout [L23, R22] | no: needs addressed LS gates in 9+ ions and mid-circuit qutrit readout |
| Qubit pairs on QCCD (H2) | >99.8% qubit gates; ~10 per qutrit CNOT [I25; CONJ count] | mid-circuit readout and real-time feed-forward shown [I25] | all-to-all [I25] | unused two-qubit state 01 [I25] | **yes**, slow rounds |
| Neutral atoms | no qudit gate [Bu26]; qubit 99.5% [E23] | 0.5 ms imaging, <1 ms with feed-forward [Bl24] | reconfigurable | erasure conversion [Ma23, W22] | no |
| Photonic | probabilistic, 1/d [C22]; deterministic only within one photon [Im19] | destructive [CONJ] | on chip | loss shows as a missing coincidence [C22] | no (T5) |
| Bosonic GKP | none between qudit modes [Br25; Mt24 is qubits] | ancilla-based, 7 µs rounds [Br25] | one mode | not shown | no |
| Spin, molecular | none in multi-qudit registers [F24, Yu25, Go17] | single object | single object | not shown | no |
| Transmon qutrits | CZ† 97.3(1)% process [Go22] | ~1 µs, 97% qutrit classification [X23] | grid plus routing for `CZ(2,3)` (1.3) | readout of level 3 (see H1) | yes, high error (Section 3) |

## 3. Simulated LRB under platform noise models

### 3.1 Method

`h2_platform_noise_study.py` reuses the exact chain-model machinery of
`src/lrb/correction`, without modifying it:

1. Every noise location of a baseline circuit keeps its exact detector and
   logical response. The cached structures come from the `p = 1e-3`
   circuits, so every location is present.
2. A noise model gives each location an activation probability and a Weyl
   distribution. The distribution is uniform over the 8 or 80 non-identity
   Weyl operators, or Z-only dephasing for idles.
3. To assign them, each location is labelled by the operation it follows,
   using the stored circuit. For example: check CNOT, logical `CZ(2,3)`, the
   1q gate of an H token, data idle in the last readout layer, measurement
   flip.
4. Exact LRB-D (`unif 1`, `const 1`) and exact raw fidelity come from the
   forward recursion.
5. MLD correction over all rounds, and over the terminal round only, is
   sampled with 4000 shots per circuit and scored with the Rao-Blackwellized
   estimator of `study.evaluate_chain`.
6. Six Clifford sequences and depths 0, 2, 4, 6 and 10 are used.

Validation [VN, V1]:

- The relabelled SI1000 model reproduces `structure.at(1e-3)` kernels exactly
  (maximum difference 0).
- At `m = 0` the unif-1 rejection is 0.0611 against 0.0611 in the baseline
  CSV, and the legacy fidelity 0.99249 against 0.99250. At `m = 2` to `10`
  the 6-sequence means agree with the 30-sequence CSV to within 0.003 in
  rejection.
- MLD fidelities agree with the decoder study's 1e5-shot results on the same
  circuits to within about two standard errors.
- The fitted SI1000 rates agree with `results/key_summary.md`:
  - `p = 1e-3`: `r_MLD` 0.0121 (key summary 0.0118), `r_u1` `2.0e-5`
    (`2e-5`), `a_u1` 0.894 (0.894);
  - `p = 5e-3`: 0.0784 (0.076), `6.0e-4` (`6.01e-4`), 0.572 (0.573).

Models [inputs LIT where cited; CONJ where marked]. All non-SI1000 models use
one parallel readout layer, `SWAP(2,3)` as a relabelling and logical X, Z as
frame updates.

| Model | Two-qutrit (check CNOT / `CZ(2,3)`) | One-qutrit | Readout flip | Data spectator per round | Reset |
|---|---|---|---|---|---|
| `ion_now` | 0.0144 / 0.0144 (98.7%) [H23] | 2.7e-3 (2e-3 per Clifford) [R22] | 3e-3 [R22] | 0.015, Z-dephasing [Y25 qubit MCMR] | 5e-3 [CONJ] |
| `ion_slowread` | as `ion_now` | as `ion_now` | as `ion_now` | 0.029: 4.5 ms at `T2` = 100 ms, exponential [M25, R22; CONJ] | 5e-3 |
| `ion_worstidle` | as `ion_now` | as `ion_now` | as `ion_now` | 0.062: `ion_slowread` plus exponential dephasing during about 5 ms of sequential gates per round [CONJ, worst case] | 5e-3 |
| `ion_next` | 5.6e-3 [CONJ: technical noise removed from the H23 budget] | 6.7e-4 [CONJ] | 1e-3 [CONJ; R22 says an order of magnitude is possible] | 1e-3 [Y25 projection] | 1e-3 |
| `qubit_pairs` | 0.02 / 8e-3 [I25 counts x 2e-3; CONJ] | Fourier 6e-3, phase 2e-3 [CONJ] | 2e-3 [CONJ] | 2e-3 [CONJ] | 1e-3 |
| `transmon_opt` | 0.027 / 0.027 [Go22], native `CZ(2,3)` | 3.2e-3 [Mo21] | 0.03 [X23] | **0** (deliberately optimistic) | 0.01 [CONJ] |

### 3.2 Results

The table is from `h2_platform_noise_study.py` R1 and R2. Values are means
over 6 sequences [VN; inputs as in 3.1]. `r` comes from a log-linear fit of
`lambda(m)` over `m ≤ 10`. `a` is the per-step acceptance.

| Model | `a_u1` | `r_u1` (cond.) | `a_c1` | `r_c1` | `r_MLD` | `r_MLD,term` | `r_raw` | `acc_u1(10)` | `lambda_MLD(10)` |
|---|---|---|---|---|---|---|---|---|---|
| SI1000 1e-3 | 0.894 | 2.0e-5 | 0.932 | 3.1e-4 | 0.0121 | 0.0187 | 0.0284 | 0.307 | 0.815 |
| SI1000 2e-3 | 0.799 | 8.4e-5 | 0.870 | 1.6e-3 | 0.0263 | 0.0432 | 0.0557 | 0.094 | 0.640 |
| SI1000 5e-3 | 0.572 | 6.0e-4 | 0.735 | 0.025 | 0.0784 | 0.121 | 0.131 | 2.7e-3 | 0.255 |
| `ion_now` | 0.671 | 7.2e-4 | 0.734 | 0.021 | 0.0776 | 0.119 | 0.127 | 0.0157 | 0.265 |
| `ion_slowread` | 0.625 | 1.45e-3 | 0.706 | 0.038 | 0.0961 | 0.139 | 0.145 | 7.8e-3 | 0.184 |
| `ion_worstidle` | 0.528 | 4.8e-3 | 0.670 | 0.096 | 0.138 | 0.185 | 0.182 | 1.4e-3 | 0.077 |
| `ion_next` | 0.891 | 3.3e-5 | 0.905 | 6.2e-4 | 0.0181 | 0.0317 | 0.042 | 0.298 | 0.745 |
| `qubit_pairs` | 0.656 | 7.1e-4 | 0.723 | 0.027 | 0.0788 | 0.129 | 0.135 | 0.0118 | 0.263 |
| `transmon_opt` | 0.519 | 1.13e-3 | 0.698 | 0.060 | 0.112 | 0.168 | 0.176 | 9.6e-4 | 0.142 |

Readings [VN unless marked]:

- **Equivalent SI1000 `p`.**
  - `ion_now` matches SI1000 at `p = 5e-3` in `r_MLD` (0.078) and `r_u1`
    (`7e-4` against `6e-4`). Its acceptance (0.67) lies between the
    `p = 2e-3` and `p = 5e-3` values.
  - `ion_next` matches `p ≈ 1e-3 to 1.5e-3`, the regime the baseline study
    explores.
  - `qubit_pairs` matches `ion_now`.
  - `transmon_opt` is worse than SI1000 at `5e-3` even with no idle noise at
    all. H1 finds that idle noise is the dominant transmon error, so real
    transmon qutrits are worse still [CONJ, cross-reference].
- **Why the ion model sits far below the gate-only `p ≈ 0.014`.** Ions read
  out at about 0.3% where SI1000 assumes `5p ≈ 7%`. They need one readout
  layer instead of four. Neither this model nor SI1000 has idle noise during
  gates. `ion_now` has about 0.40 expected faults per depth step (R1
  `faults` column). Parallel-readout SI1000 at the same two-qutrit error has
  `64.4 x 0.0144 ≈ 0.93` for the check round alone.
- **Correction against detection.**
  - On every platform model, MLD correction gives `r` well above the
    physical per-step reference. For ions that reference is about `4e-3`,
    two single-qutrit Cliffords at `2(2)e-3` [LIT R22]. So correction on this
    `d = 2` code is below break-even.
  - The conditional unif-1 rate is below the physical reference on the ion
    and qubit-pair models: `3.3e-5` to `1.45e-3` against about `4e-3`. The
    acceptance per step is 0.63 to 0.89.
  - If rejected shots are scored as random outcomes, unif 1 on `ion_now`
    becomes `a f ≈ 0.67`, which is `r ≈ 0.22`, worse than MLD. This matches
    A2's general result that MAP correction is optimal in that scoring [VN
    arithmetic; cross-reference A2].
- **Offsets.** At `m = 0` the unif-1 accepted fidelity is 0.946 for
  `ion_now` and 0.905 for `ion_slowread`. Dephasing of data 0 and 3 in the
  final round's readout layer comes after the last check and flips the
  logical outcome undetected. It sets the offset `B`, not the decay rate. A
  hardware run would read the data together with the final ancillas and
  avoid most of it [CONJ].
- **Spectator error is the dominant uncertainty.**
  - Raising it from 1.5% to 2.9% (`ion_slowread`) moves `r_MLD` from 0.078
    to 0.096 and doubles `r_u1` [VN].
  - In the worst case, 6.2% per round (`ion_worstidle`: exponential
    dephasing over the whole 9.6 ms round at `T2` = 100 ms), `r_MLD` is
    0.138 and `a_u1` 0.53. The conditional unif-1 rate, `4.8e-3`, is then no
    longer below the physical reference of about `4e-3` [VN].
  - Quasi-static (Gaussian) dephasing or echo pulses would keep the real
    value nearer `ion_now` [CONJ]. Measuring the spectator dephasing of
    D5/2-encoded data during a qutrit readout is the most useful single
    calibration for this experiment [CONJ].

## 4. Time and shot budgets

### 4.1 Durations

Block T1 [arithmetic VN; inputs are LIT where cited in the script, otherwise
labelled estimates, so the budget as a whole is CONJ]:

| Platform | Check round | Round / `T2` | Logical Clifford | Shot at m = 10 | Shot at m = 22 | Coherent part of m = 22 / `T2` |
|---|---|---|---|---|---|---|
| 40Ca+ qutrits | 9.6 ms (12 sequential CNOTs 4.5 ms, readout 4.5 ms) | 0.096 | 1.4 ms | 0.14 s | 0.29 s | 2.8 |
| Qubit pairs, QCCD | ~134 ms (estimate) | 0.13 | ~34 ms | 2.2 s | 4.6 s | 4.6 |
| Transmon qutrits | 6.3 µs | 0.37 (`T2` = 17 µs) | 2.4 µs | 0.22 ms | 0.35 ms | 15 |
| Neutral atoms (qubit timings, hypothetical) | 1.5 ms | 0.0015 | 11 µs | 117 ms | 135 ms | 0.035 |

Wall clock for the baseline dataset shape, 30 sequences x 9 depths x N shots
(T2):

| Platform | N = 1e3 | N = 1e4 | N = 1e6 |
|---|---|---|---|
| 40Ca+ qutrits | 11 h | 4.8 d | 475 d |
| Qubit pairs, QCCD | 173 h | 72 d | 7,200 d |
| Transmon qutrits | 61 s | 10 min | 0.7 d |
| Neutral atoms (hypothetical) | 8.8 h | 3.7 d | 368 d |

The baseline's 1e6 shots per circuit are out of reach on any ion platform
[VN arithmetic].

### 4.2 Shots needed

Block T3 [VN; variance formula VA and VN A8]. The design is two points, depth
0 and the best `M ∈ {2, 4, 6, 10}`, with 30 sequences per depth. The target is
one standard error equal to 10% of `r`. The variance of the signed estimate is
`(1 - lambda)(1 + 2 lambda)/(2 N_acc)`. The floor from the spread between
sequences, estimated from 6 sequences, is at most 4% of `r` in every case, so
shot noise dominates.

| Model | MLD correction: shots per circuit | LRB-D unif 1: shots per circuit, beam time for two depths | LRB-D const 1: shots per circuit, beam time |
|---|---|---|---|
| SI1000 1e-3 | 39 | 1.75e6 | 8.2e3 |
| `ion_now` | 16 (about 1 min) | 1.2e5, 117 h | 645, 0.9 h |
| `ion_slowread` | 16 | 6.9e4, 51 h | 692, 0.9 h |
| `ion_worstidle` | 19 | 2.7e4, 20 h | 874, 0.8 h |
| `ion_next` | 26 | 6.5e5, 877 h | 3.2e3, 4.3 h |
| `qubit_pairs` | 15 (0.2 h) | 8.2e4, 1,030 h | 578, 11 h |
| `transmon_opt` | 15 | 1.1e5, 0.23 h | 621, 5 s |

Correction makes the slow platforms usable. Its decay is large, so tens to
hundreds of shots per circuit give `r` to 10%, once the decoder's noise model
has been learned. Learning that model takes its own memory-type runs; H1
describes them [CONJ].

Detection's conditional decay is tiny, so it needs `1e5` to `1e6` shots per
circuit. That means hundreds of hours on ions [VN].

### 4.3 The baseline at large `p` is a noise floor

Block T6 [VN]. At SI1000 `p ≥ 0.0125` and `m ≥ 10`:

- `const 1` rejects exactly `1 - 1/81 = 0.98765`. The terminal syndrome is
  then uniformly random.
- The accepted-shot legacy fidelity of 0.007 to 0.009 sits on the Rayleigh
  floor `sqrt(pi/(4 N_acc)) = 0.008` of the metric `|mean(w^k)|`. That floor
  is the value the metric takes for uniformly random outcomes [VA; VN A8:
  the Monte Carlo mean 0.0282 against the floor 0.0280 at N = 1000].

Fidelities near 0.01 in the baseline at these `p` carry no information.

## 5. Control stacks and real-time decoding

### 5.1 Measured or specified latencies

All rows are [LIT]; vendor pages are vendor claims.

| Stack | Figure | What it covers | Source |
|---|---|---|---|
| QICK (RFSoC, Fermilab) | 184 to 211 ns total | DAC+ADC round trip 90 to 117 ns, conditional evaluation and jump 42 ns, next pulse 52 ns | [St21] |
| QubiC 2.0 (RFSoC ZCU216, LBNL) | 71.2 ± 2.6 ns | one-way board-to-board, used for cross-board feed-forward | [X23] |
| QubiC 2.0 | 66 ns | inference of an FPGA ML classifier with 97% qutrit fidelity from a 500 ns readout pulse | [X23] |
| Zurich Instruments | 350 ns; < 550 ns; 380 ns (1st gen) | local feedback inside SHFQC; global feedback via PQSC "including all signal processing"; UHFQA to HDAWG. The PQSC page says decoders can run on its FPGA (no figure) | [ZI-FB, ZI-PQSC] |
| Qblox LINQ | 160 / 250 / 472 ns | thresholded bits self- / intra- / multi-cast, Q1 core to feedback queue, I/O paths excluded | [QB] |
| Quantum Machines (OPX-family controller, written "OPX++" in the paper's HTML) | 428 ns | 288 ns time of flight + 140 ns FPGA processing, readout to conditional pulse | [Cao26] |
| DGX Quantum (QM OPX + NVIDIA Grace Hopper) | < 4 µs | controller–GPU round trip | [NV25a] |
| NVIDIA NVQLink | 3.84 µs mean, 3.96 µs max | RFSoC FPGA to GPU loopback, 1000 samples | [NV25b] |
| Quantinuum Helios + NVIDIA GPU | 67 µs median | BP+OSD decode of Bring's code (8 logical qubits in 30) | [NV25b, QT26, QTdoc] |
| Quantinuum (2021 QCCD) | < 250 µs worst case; cycle < 200 ms | conditional-information distribution; two-stage lookup-table decoder with Pauli-frame updates | [RA21] |
| Google | 63 ± 17 µs | workstation decoder latency at `d = 5`, 1.1 µs cycles, up to `10^6` cycles, "does not yet include feedback" | [A25] |
| Riverlane FPGA decoder in Rigetti control | 0.44 to 0.79 µs per round; 9.6 µs response for 9 rounds | full response = 6.5 µs decoding + 3.1 µs communication and control; 1.4 µs for the final readout to reach the decoder | [Ca24] |
| Riverlane Collision Clustering | "MHz decoding speed" | up to 881 qubits on FPGA and 1057 on ASIC (0.06 mm², 8 mW), from the abstract | [Ba25] |
| AQT ion demonstrator | ~5 µs | master to peripheral-card latency for real-time feedback | [P21] |
| ARTIQ (ion and atom labs) | "sub-microsecond latency" (no number) | FPGA core device | [ARTIQ] |
| Neutral atoms (Harvard) | < 1 ms | readout plus FPGA decoding and feedforward cycle | [Bl24] |
| Yale bosonic | 2 x 4.924 µs cycle; 448 ns | QEC cycle including real-time decisions on a Virtex-6 FPGA; virtual oscillator rotation | [Si23] |

Quantum Machines' own pages returned HTTP 403 to automated fetches, so the OPX
row uses the peer-reviewable number in Cao26, whose coauthors include a QM
author.

### 5.2 Clifford LRB needs no real-time decoding

The argument [VA; VN A4, A6; LIT T15, Bl24, RA21, A25]:

1. Every gate in the circuits maps Weyl operators to Weyl operators (A4). The
   simulated noise is stochastic Weyl noise, and no operation is classically
   controlled.
2. A Weyl correction inferred at any time therefore propagates through the
   remaining Cliffords to a Weyl operator just before the terminal `H_INV`.
   The readout is in the X basis, so the Z exponents on wires 0 and 3 shift
   `x0 + x3` by a known amount, and the X exponents only add phases.
3. The decoder is a map from the `4(m+1)` detector trits of a shot to a `Z_3`
   shift, and optionally a reject flag, applied after the run.
4. Terhal states this for Clifford circuits: the Pauli frame "does not
   necessarily need to be available during the execution of the circuit" and
   "simply alters the interpretation of the final measurement outcome"
   [LIT T15].

A3 and L5 prove the exact equivalence of the offline frame with ideal physical
correction for this model, and show where it breaks: coherent noise,
amplitude damping and leakage [cross-reference].

Hardware practice agrees:

- Bluvstein et al. applied their Clifford-circuit corrections in software
  [LIT Bl24].
- Ryan-Anderson et al. tracked corrections in a software Pauli frame
  [LIT RA21].
- Google's offline neural-network decoder beat its real-time decoder, with
  `eps_5` of 0.269% against 0.35% [LIT A25]. Offline decoding is the more
  accurate choice when nothing needs feedback [CONJ].

What must still run in real time:

- **Storage of every mid-circuit outcome,** 92 trits at `m = 22` [VA].
- **Reset.** Unconditional reset or none, both feedback-free (A6 shows none
  is needed) [VN].
- **Leakage handling.** The ion and qubit-pair heralds are read out with the
  measurement and can be used offline [LIT L23, I25; CONJ that nothing else
  is needed].

On slow platforms, aborting a shot in real time when a used check fires would
save beam time. It fixes the postselection policy in advance, though, while
the baseline evaluates all policies on the same shots [CONJ].

### 5.3 Non-Clifford logical gates need it

Block A5 and the literature [VN; LIT]:

- **The qutrit T gate.** `T3 = diag(1, z, z^8)`, `z = e^{2 pi i/9}`, belongs
  to the Howard–Vala family `upsilon = (0, 6z'+2g'+3e', 6z'+g'+6e') mod 9`,
  with `(z', g', e') = (1, 2, 0)`. It is not a Clifford, but `T3 W T3^dag`
  is a Clifford for every Weyl `W`, the third level of the hierarchy
  [VN; LIT HV12].
- **Effect of a frame error.**
  - A Z-type frame error commutes with `T3` and only shifts the X-basis
    outcome distribution cyclically.
  - Every frame error with an X component produces a distribution that is
    not a cyclic shift. For a generic input, the ideal `[0.295, 0.686, 0.019]`
    becomes `[0.486, 0.107, 0.407]` after `X`.
  - No relabelling of the final outcome can repair this [VN].
- **Terhal's qubit argument.** A logical X before a T gadget "is mapped onto a
  logical Clifford error". One therefore wants to know the frame that affects
  the gadget's measurement before the next gate [LIT T15].
- **Qudit gadgets.** Qudit distillation applies a "Clifford correction
  operator ... tuned to outcomes" [LIT CAB12].
- **Requirements.**
  - The reaction time is measurement plus transport plus decoding plus
    control.
  - Decoding throughput must keep up with syndrome generation; otherwise the
    backlog grows and the computation slows exponentially [LIT T15; Ca24
    measure exactly this].
  - Slow diagnostics can be absorbed by waiting, a constant slow-down
    [LIT T15], or by working in a Clifford frame [LIT CIP18].
- **Per platform (T4, latency / round) [VN arithmetic of LIT numbers].**
  - On ions (9.6 ms round), every decoding and feedback entry of 5.1 is
    small: FPGA loops are about `5e-5` of a round, GPU or workstation
    decoders about `7e-3`, and even Quantinuum's 250 µs distribution is
    0.026. The neutral-atom figure (0.1 of an ion round) includes 500 µs of
    imaging and is not a decoder latency.
  - On transmons (6.3 µs round), FPGA loops are 0.03 to 0.09 of a round. The
    Riverlane response is 1.5 rounds, and GPU or workstation decoders are
    about 10 rounds.
  - Real-time non-Clifford operation on transmons therefore needs
    FPGA-resident decoding. On ions any decoder suffices [CONJ].
- **Decoder size for this code.**
  - A per-round lookup table has 81 entries, 6561 for a two-round window
    (2.4 KiB) and 531,441 for three rounds (195 KiB). The transfer maps of
    1.4 add one 4x4 `Z_3` matrix per Clifford [VN A7].
  - Every listed FPGA has room for this, so decoding costs one memory access
    and readout dominates the reaction time [CONJ].
- **For this `d = 2` code.** A T-gate experiment would postselect on the
  injection and the checks. In accepted shots the frame is trivial. What is
  needed in real time is then feed-forward of raw outcomes and a reject flag,
  not decoding proper. That begins with `d ≥ 3` qutrit codes, or with an
  MLD frame guess on the `d = 2` code [CONJ].

## 6. Conclusion: which platform is most sustainable

Criteria. This experiment is limited by four things:

- the error per check round relative to what a distance-2 code can flag or
  correct;
- the quality of mid-circuit qutrit readout and the protection of spectators;
- the shot budget;
- for a long campaign, calibration stability.

It is not limited by decoder speed (5.2).

Ranking [CONJ, built on Sections 2 to 5]:

1. **Trapped ions are the most sustainable platform.** Two stages:
   - *Now:* qubit-pair-encoded qutrits on a commercial all-to-all QCCD
     machine. Every ingredient has been demonstrated at 24 qutrits [LIT I25].
     Modelled performance equals the native-qutrit `ion_now` model. MLD
     correction then needs about 15 shots per circuit for 10% on `r`, under
     an hour of machine time for two depths. Detection-based LRB-D would
     need about 1000 h [VN T3, with CONJ inputs].
   - *Next:* native `40Ca+` qutrits. They need fewer native operations per
     qutrit gate. They have the best measured qutrit gates and readout (2.1)
     and a built-in leakage herald, and the experiment needs no routing. The
     prerequisites are addressed LS gates in a chain of at least 9 ions,
     and parallel mid-circuit qutrit readout with spectator error of about
     `1e-3`. Readout of about 1 ms or less would help, because D5/2 decay
     alone costs about `4.5e-3` per 4.5 ms readout. With the improvements the
     ion papers project (`ion_next`), the experiment would run in the regime
     the baseline study covers: equivalent SI1000 `p ≈ 1e-3 to 1.5e-3`, and
     `unif 1` acceptance 0.89 per step.
   - Until the spectator dephasing of idle qutrits is measured, today's
     native-ion prediction lies between `ion_now` and `ion_worstidle`:
     `r_MLD` 0.078 to 0.14, `a_u1` 0.67 to 0.53 [VN R2].
   - Caveat: ion qudit gates show slow non-Markovian noise [LIT H23]. That
     is relevant to study question 2 and should be characterized alongside
     the LRB data.
2. **Transmon qutrits are the throughput option.** They have the most mature
   open and commercial feedback stacks (5.1) and a 1e6-shot dataset takes
   0.7 days (4.1). But even the optimistic `transmon_opt` model, with no idle
   noise and native `CZ(2,3)`, is worse per round than `ion_now`
   (`a_u1` 0.52 against 0.67, `r_MLD` 0.11 against 0.078). It is only
   comparable to the ion worst case `ion_worstidle`, which includes all idle
   dephasing (0.53, 0.14). Real transmon devices add idle noise and routing
   (H1).
3. **Neutral atoms become a candidate when a two-qudit gate appears.**
   Reconfigurable connectivity, sub-ms mid-circuit imaging with FPGA
   feedforward, and erasure conversion already exist for qubits.
4. **Photonic, bosonic GKP, and spin or molecular qudits are not candidates
   for this experiment in the near term.**

**The protocol is the same on every platform:**

- store all mid-circuit outcomes;
- read all ancillas in one parallel layer per round;
- skip reset or reset unconditionally;
- decode offline into a `Z_3` frame shift;
- prefer correction or sparse checks over `unif 1` postselection, whose shot
  cost is prohibitive on slow machines;
- plan real-time decoding only for non-Clifford logical gates, and start
  with real-time feed-forward of raw injection outcomes on the `d = 2` code.

Real-time decoding is not needed for Clifford LRB. It is needed, with
latency below one round, only once non-Clifford gates are added. On ions that
latency is trivial to meet; on transmons it requires FPGA decoders
[CONJ, built on 5.2 and 5.3].

## 7. Caveats and open points

- **Platform models.**
  - The models replace each published number by a depolarizing or dephasing
    Weyl channel. They omit coherent errors, leakage, crosstalk, and
    correlated or slow noise [CONJ].
  - Idle noise during gates is omitted except in `ion_worstidle`. It is not
    negligible on ions: a round has about 5 ms of sequential gates, and with
    exponential dephasing at `T2` = 100 ms it would add up to 3.3% per data
    qutrit per round. That case is bracketed by `ion_worstidle` [VN].
  - The MLD decoder knows the exact noise model. With a learned and
    mismatched model it does worse [CONJ].
- **Mapping of the ion gate fidelity.** Hrmo's 98.7% is a SPAM-corrected
  per-gate decay of an entangled-state fidelity under non-Markovian noise,
  not a process fidelity. The mapping to `q = 0.0144` is an approximation
  [CONJ].
- **Estimated durations.** Gate durations for ions, the qubit-pair QCCD round
  time, transmon reset and per-shot preparation are my estimates and may be
  off by 2x or more [CONJ].
- **Absence claims** (no 9-qutrit ion register, no mid-circuit qudit readout
  on ions, no two-mode GKP qutrit gate, no multi-qudit spin register) come
  from my searches on 2026-10-28. For neutral atoms, Bu26 states the absence
  as of its writing.
- **Sources read only in part.** For Ba25, B26 and several vendor pages I
  read only the abstract or the page itself. Numbers from their full texts
  are not used.

## 8. Changes relative to the earlier H2 draft

The scripts in `notes/scripts/critic_H2-other-platforms_*.py` flagged several
items. I re-derived each one myself.

1. **The LS gate needs `D^dag` corrections on both qutrits.** The draft said
   "one LS gate plus two Fourier gates on the target". The exact identity is
   `CNOT = (I x H^dag)(D^dag x D^dag) G (I x H)`, and without the `D` factors
   the product is not a CNOT [VN A2].
2. **Shot counts.** The draft's "4e7 raw shots per circuit" for `unif 1` at
   `p = 1e-3` put a 10% shot-noise target on a single circuit's fidelity.
   The decay `F(0) - F(22)` it was compared with is a 30-sequence average.
   For that average the 4e7 is a total over the 30 sequences, i.e. 30x fewer
   shots per circuit [VA]. Section 4.2 uses an explicit target (10% on `r`
   with 30 sequences) and gives per-circuit numbers: `1.75e6` at SI1000
   `p = 1e-3` for `unif 1` [VN T3].
3. **The platform-matched `p ≈ 0.014` for ions overstated ion readout and idle
   noise.** A location-by-location ion model behaves like SI1000 at
   `p ≈ 3e-3 to 5e-3` [VN R2]. The draft's conclusion that the baseline
   rejects more than 99.9% of shots by `m = 4` on ions does not hold for the
   ion model, which keeps 18% at `m = 4` and 8% at `m = 6`.
4. **The fidelity of about 0.01 at `m = 22`** that the draft read from `const 1`
   at `p = 0.015` is the Rayleigh floor of the legacy metric [VN T6].
5. **"Shot / coherence"** mixed per-shot preparation with coherent evolution.
   T1 now reports the coherent part separately.
6. **Riverlane ASIC and FPGA per-round times** came from a full text I did not
   fetch this time. Only abstract-level claims are kept [LIT Ba25].
7. **Two additions:**
   - all shortest 2–3 routes on the grid pass through another data qutrit
     [VN A3];
   - logical gates act on syndromes through fixed transfer maps [VN A4].

## 9. Sources fetched in this task

I fetched every source below in this task; the URLs are given per entry.

### Trapped ions

- [R22] M. Ringbauer, M. Meth, L. Postler, R. Stricker, R. Blatt, P. Schindler, T. Monz, "A universal qudit quantum processor with trapped ions", Nat. Phys. 18, 1053 (2022), arXiv:2109.06903, DOI 10.1038/s41567-022-01658-0.
  - Fetched https://arxiv.org/abs/2109.06903 and https://arxiv.org/pdf/2109.06903 (v1).
- [H23] P. Hrmo, B. Wilhelm, L. Gerster, M. W. van Mourik, M. Huber, R. Blatt, P. Schindler, T. Monz, M. Ringbauer, "Native qudit entanglement in a trapped ion quantum processor", Nat. Commun. 14, 2242 (2023), arXiv:2206.04104, DOI 10.1038/s41467-023-37375-2.
  - Fetched https://arxiv.org/abs/2206.04104 and https://arxiv.org/pdf/2206.04104 (v1).
  - The nature.com page redirected to a login page.
- [M25] M. Meth, J. F. Haase, J. Zhang, C. Edmunds, L. Postler, A. Steiner, A. J. Jena, L. Dellantonio, R. Blatt, P. Zoller, T. Monz, P. Schindler, C. Muschik, M. Ringbauer, "Simulating 2D lattice gauge theories on a qudit quantum computer", Nat. Phys. 21, 570–576 (2025), arXiv:2310.12110, DOI 10.1038/s41567-025-02797-w.
  - Fetched https://arxiv.org/abs/2310.12110 and https://arxiv.org/pdf/2310.12110 (v3).
- [L23] P. J. Low, B. White, C. Senko, "Control and readout of a 13-level trapped ion qudit", arXiv:2306.03340 (2023).
  - Fetched https://arxiv.org/abs/2306.03340 and the PDF.
  - The arXiv page lists no journal reference.
- [S26] X. Shi, J. Sinanan-Singh, T. J. Burke, J. Chiaverini, I. L. Chuang, "Efficient implementation of a quantum algorithm with a trapped ion qudit", Nat. Commun. 17, 1911 (2026), arXiv:2506.09371, DOI 10.1038/s41467-026-68746-0.
  - Fetched https://arxiv.org/abs/2506.09371 and the Crossref record https://api.crossref.org/works/10.1038/s41467-026-68746-0.
- [Y25] Y. Yu, K. Yan, D. Biswas, V. N. Zhang, B. Harraz, C. Noel, C. Monroe, A. Kozhanov, "In situ midcircuit qubit measurement and reset in a single-species trapped-ion quantum computing system", Phys. Rev. Research 7, 043355 (2025), arXiv:2504.12544, DOI 10.1103/qfvd-93lw.
  - Fetched https://arxiv.org/abs/2504.12544, the PDF (v3) and the Crossref record.
- [B26] S. Brudney, C. Burns, G. J. Gregory, E. Ritchie, D. J. Wineland, D. T. C. Allcock, J. O'Reilly, "Mid-circuit ground-state cooling and ancilla readout in the omg architecture", arXiv:2608.13181 (2026).
  - Fetched https://arxiv.org/abs/2608.13181 (abstract only).
- [P21] I. Pogorelov, T. Feldker, Ch. D. Marciniak, et al. (16 authors), "A compact ion-trap quantum computing demonstrator", PRX Quantum 2, 020343 (2021), arXiv:2101.11390, DOI 10.1103/PRXQuantum.2.020343.
  - Fetched https://arxiv.org/abs/2101.11390 and the PDF.
- [RA21] C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, et al. (17 authors), "Realization of real-time fault-tolerant quantum error correction", Phys. Rev. X 11, 041058 (2021), arXiv:2107.07505, DOI 10.1103/PhysRevX.11.041058.
  - Fetched https://arxiv.org/abs/2107.07505, the PDF and the Crossref record.
- [I25] M. Iqbal, A. Lyons, C. F. B. Lo, N. Tantivasadakarn, et al. (19 authors), "Qutrit toric code and parafermions in trapped ions", Nat. Commun. 16, 6301 (2025), arXiv:2411.04185, DOI 10.1038/s41467-025-61391-z.
  - Fetched https://arxiv.org/abs/2411.04185 and https://arxiv.org/html/2411.04185v1.
- [Mt24] V. G. Matsos, C. H. Valahu, M. J. Millican, T. Navickas, X. C. Kolesnikow, M. J. Biercuk, T. R. Tan, "Universal quantum gate set for Gottesman-Kitaev-Preskill logical qubits", arXiv:2409.05455 (2024).
  - Fetched https://arxiv.org/abs/2409.05455.
  - The journal version was not confirmed by me.

### Neutral atoms

- [E23] S. J. Evered, D. Bluvstein, M. Kalinowski, et al. (15 authors), "High-fidelity parallel entangling gates on a neutral atom quantum computer", Nature 622, 268 (2023), arXiv:2304.05420, DOI 10.1038/s41586-023-06481-y.
  - Fetched https://arxiv.org/abs/2304.05420.
- [Bl24] D. Bluvstein, S. J. Evered, A. A. Geim, et al. (21 authors), "Logical quantum processor based on reconfigurable atom arrays", Nature (2024), arXiv:2312.03982, DOI 10.1038/s41586-023-06927-3.
  - Fetched https://arxiv.org/abs/2312.03982 and https://arxiv.org/pdf/2312.03982.
- [Ma23] S. Ma, G. Liu, P. Peng, et al. (10 authors), "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit", Nature 622, 279 (2023), arXiv:2305.05493, DOI 10.1038/s41586-023-06438-1.
  - Fetched https://arxiv.org/abs/2305.05493.
- [W22] Y. Wu, S. Kolkowitz, S. Puri, J. D. Thompson, "Erasure conversion for fault-tolerant quantum computing in alkaline earth Rydberg atom arrays", Nat. Commun. 13, 4657 (2022), arXiv:2201.03540, DOI 10.1038/s41467-022-32094-6.
  - Fetched https://arxiv.org/abs/2201.03540.
- [Bu26] A. Burshtein, S. Fraenkel, M. Goldstein, R. Finkelstein, "Robust control and entanglement of qudits in neutral atom arrays", Phys. Rev. Research 8, 013055 (2026), DOI 10.1103/4xcd-wyxx.
  - Fetched https://journals.aps.org/prresearch/abstract/10.1103/4xcd-wyxx.

### Photonic, bosonic, spin and molecular

- [C22] Y. Chi, J. Huang, Z. Zhang, J. Mao, Z. Zhou, X. Chen, C. Zhai, J. Bao, T. Dai, H. Yuan, M. Zhang, D. Dai, B. Tang, Y. Yang, Z. Li, Y. Ding, L. K. Oxenløwe, M. G. Thompson, J. L. O'Brien, Y. Li, Q. Gong, J. Wang, "A programmable qudit-based quantum processor", Nat. Commun. 13, 1166 (2022), DOI 10.1038/s41467-022-28767-x.
  - Fetched the full text from https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8897515/fullTextXML.
  - The PMC page returned a captcha.
- [Im19] P. Imany, J. A. Jaramillo-Villegas, M. S. Alshaykh, J. M. Lukens, O. D. Odele, A. J. Moore, D. E. Leaird, M. Qi, A. M. Weiner, "High-dimensional optical quantum logic in large operational spaces", npj Quantum Inf. 5, 59 (2019), arXiv:1805.04410, DOI 10.1038/s41534-019-0173-8.
  - Fetched https://arxiv.org/abs/1805.04410.
- [Br25] B. L. Brock, S. Singh, A. Eickbusch, V. V. Sivak, A. Z. Ding, L. Frunzio, S. M. Girvin, M. H. Devoret, "Quantum error correction of qudits beyond break-even", Nature 641, 612–618 (2025), arXiv:2409.15065, DOI 10.1038/s41586-025-08899-y.
  - Fetched https://arxiv.org/abs/2409.15065 and the PDF.
- [Si23] V. V. Sivak, A. Eickbusch, B. Royer, S. Singh, I. Tsioutsios, S. Ganjam, A. Miano, B. L. Brock, A. Z. Ding, L. Frunzio, S. M. Girvin, R. J. Schoelkopf, M. H. Devoret, "Real-time quantum error correction beyond break-even", Nature 616, 50–55 (2023), arXiv:2211.09116, DOI 10.1038/s41586-023-05782-6.
  - Fetched https://arxiv.org/abs/2211.09116, the PDF and the Crossref record.
- [F24] I. Fernández de Fuentes, T. Botzem, M. A. I. Johnson, et al. (14 authors), "Navigating the 16-dimensional Hilbert space of a high-spin donor qudit with electric and magnetic fields", Nat. Commun. 15, 1380 (2024), arXiv:2306.07453, DOI 10.1038/s41467-024-45368-y.
  - Fetched https://arxiv.org/abs/2306.07453.
- [Yu25] X. Yu, B. Wilhelm, D. Holmes, et al. (20 authors), "Schrödinger cat states of a nuclear spin qudit in silicon", Nat. Phys. (2025), arXiv:2405.15494, DOI 10.1038/s41567-024-02745-0.
  - Fetched https://arxiv.org/abs/2405.15494.
- [Go17] C. Godfrin, A. Ferhat, R. Ballou, S. Klyatskaya, M. Ruben, W. Wernsdorfer, F. Balestro, "Operating quantum states in single magnetic molecules: implementation of Grover's quantum algorithm", Phys. Rev. Lett. 119, 187702 (2017), arXiv:1710.11229, DOI 10.1103/PhysRevLett.119.187702.
  - Fetched https://arxiv.org/abs/1710.11229.

### Superconducting reference

- [Go22] N. Goss, A. Morvan, B. Marinelli, B. K. Mitchell, L. B. Nguyen, R. K. Naik, L. Chen, C. Jünger, J. M. Kreikebaum, D. I. Santiago, J. J. Wallman, I. Siddiqi, "High-fidelity qutrit entangling gates for superconducting circuits", arXiv:2206.07216 (v3 2023), related DOI 10.1038/s41467-022-34851-z.
  - Fetched https://arxiv.org/abs/2206.07216 and the PDF.
- [Mo21] A. Morvan, V. V. Ramasesh, M. S. Blok, J. M. Kreikebaum, K. O'Brien, L. Chen, B. K. Mitchell, R. K. Naik, D. I. Santiago, I. Siddiqi, "Qutrit randomized benchmarking", Phys. Rev. Lett. 126, 210504 (2021), arXiv:2008.09134, DOI 10.1103/PhysRevLett.126.210504.
  - Fetched https://arxiv.org/abs/2008.09134.

### Control stacks and decoders

- [St21] L. Stefanazzi, K. Treptow, N. Wilcer, C. Stoughton, S. Montella, C. Bradford, G. Cancelo, S. Saxena, H. Arnaldi, S. Sussman, A. Houck, A. Agrawal, H. Zhang, C. Ding, D. I. Schuster, "The QICK (Quantum Instrumentation Control Kit): Readout and control for qubits and detectors", arXiv:2110.00557 (v2 2022), related DOI 10.1063/5.0076249.
  - Fetched https://arxiv.org/abs/2110.00557 and the PDF.
- [X23] Y. Xu, G. Huang, N. Fruitwala, N. R. Vora, A. D. Rajagopala, A. Hashim, L. Chen, N. Goss, K. Nowrouzi, R. K. Naik, C. Spitzer, D. I. Santiago, I. Siddiqi, "QubiC 2.0: An extensible open-source qubit control system capable of mid-circuit measurement and feed-forward", arXiv:2309.10333 (v2, 22 Oct 2026).
  - Fetched https://arxiv.org/abs/2309.10333 and the PDF.
- [ZI-PQSC] Zurich Instruments, "PQSC", https://www.zhinst.com/en/products/pqsc (vendor page).
- [ZI-FB] Zurich Instruments, "Quantum feedback measurements", https://www.zhinst.com/en/applications/quantum-technologies/quantum-feedback-measurements (vendor page).
- [QB] Qblox documentation, "LINQ-based feedback", https://docs.qblox.com/en/main/products/architecture/sequencers/linq_based_feedback.html (vendor documentation).
- [Cao26] Q. Cao, U. Akhouri, S. Vizvary, N. Ofek, W. Dai, K. W. Murch, "Adaptive thresholding for scalable measurement-based qubit reset", arXiv:2609.25208 (2026).
  - Fetched https://arxiv.org/abs/2609.25208 and https://arxiv.org/html/2609.25208.
- [NV25a] NVIDIA Technical Blog, "Accelerating Quantum Error Correction Research with NVIDIA Quantum", 2025-03-20, https://developer.nvidia.com/blog/accelerating-quantum-error-correction-research-with-nvidia-quantum/ (vendor page).
- [NV25b] NVIDIA Technical Blog, "NVIDIA NVQLink Architecture Integrates Accelerated Computing with Quantum Processors", 2025-11-17, https://developer.nvidia.com/blog/nvidia-nvqlink-architecture-integrates-accelerated-computing-with-quantum-processors (vendor page).
- [QT26] Quantinuum blog, "Real Time Error Correction at Increased Scale", 2026-03-16, https://www.quantinuum.com/blog/real-time-error-correction-at-increased-scale (vendor page).
- [QTdoc] Quantinuum documentation, "GPU Decoding", https://docs.quantinuum.com/systems/trainings/helios/getting_started/gpu_decoding.html (vendor documentation).
- [ARTIQ] M-Labs, "ARTIQ", https://m-labs.hk/experiment-control/artiq/ (project page).
- [A25] R. Acharya et al. (Google Quantum AI and Collaborators), "Quantum error correction below the surface code threshold", Nature 638, 920–926 (2025), arXiv:2408.13687, DOI 10.1038/s41586-024-08449-y.
  - Fetched https://arxiv.org/abs/2408.13687 and the PDF.
- [Ba25] B. Barber, K. M. Barnes, T. Bialas, O. Buğdaycı, E. T. Campbell, N. I. Gillespie, K. Johar, R. Rajan, A. W. Richardson, L. Skoric, C. Topal, M. L. Turner, A. B. Ziad, "A real-time, scalable, fast and highly resource efficient decoder for a quantum computer", Nat. Electron. (2025), arXiv:2309.05558, DOI 10.1038/s41928-024-01319-5.
  - Fetched https://arxiv.org/abs/2309.05558 (abstract only).
- [Ca24] L. Caune, L. Skoric, N. S. Blunt, A. Ruban, et al. (25 authors), "Demonstrating real-time and low-latency quantum error correction with superconducting qubits", arXiv:2410.05202 (2024); Nat. Commun. 17, 7383 (2026), DOI 10.1038/s41467-026-73331-6.
  - Fetched https://arxiv.org/abs/2410.05202, the PDF and the Crossref record.

### Theory

- [T15] B. M. Terhal, "Quantum error correction for quantum memories", Rev. Mod. Phys. 87, 307 (2015), arXiv:1302.3428, DOI 10.1103/RevModPhys.87.307.
  - Fetched https://arxiv.org/abs/1302.3428 and the PDF (section II.G.3).
- [HV12] M. Howard, J. Vala, "Qudit versions of the qubit 'pi-over-eight' gate", Phys. Rev. A 86, 022316 (2012), arXiv:1206.1598.
  - Fetched https://arxiv.org/abs/1206.1598 and the PDF.
- [CAB12] E. T. Campbell, H. Anwar, D. E. Browne, "Magic state distillation in all prime dimensions using quantum Reed-Muller codes", Phys. Rev. X 2, 041021 (2012), arXiv:1205.3104.
  - Fetched https://arxiv.org/abs/1205.3104 and the PDF.
- [CIP18] C. Chamberland, P. Iyer, D. Poulin, "Fault-tolerant quantum computing in the Pauli or Clifford frame with slow error diagnostics", Quantum 2, 43 (2018), arXiv:1704.06662, DOI 10.22331/q-2018-01-04-43.
  - Fetched https://arxiv.org/abs/1704.06662 and the PDF.
- [Ge25] G. P. Gehér, M. Jastrzebski, E. T. Campbell, O. Crawford, "To reset, or not to reset—that is the question", npj Quantum Inf. 11, 39 (2025), arXiv:2408.00758, DOI 10.1038/s41534-025-00998-y.
  - Fetched https://arxiv.org/abs/2408.00758 and the Crossref record.

### Fetched but not usable

- Quantum Machines product and feedback pages returned HTTP 403.
- The Knill 2005 abstract was fetched but does not mention the Pauli frame, so it is not cited.

## 10. Claim list

Each claim carries one label. Where a body section mixes evidence types, the
claim is split here so that each row has a single label.

| ID | Claim | Label | Evidence |
|---|---|---|---|
| H2-other-platforms-C1 | A check round has 12 two-qutrit and 4 one-qutrit gates, 4 measurements and 4 resets. A logical Clifford averages 3.116 two-qutrit and 16.676 one-qutrit gates (1.463 with SWAP(2,3) as a relabelling; 12.657 with X and Z as frame updates). Depth 22 averages 413 two-qutrit gates and 92 mid-circuit measurements. | verified numerically | `h2_algebra_checks.py` A1 (repository parser, 216-entry catalog) |
| H2-other-platforms-C2 | SI1000 puts 118.4 p of fault weight per check round, 88 p of it in the four sequential single-ancilla readout layers. One parallel readout layer gives 64.4 p, and each data qutrit's idle weight drops from 10 p to 4 p. | verified analytically | derived from the brief's noise model (A1); class weights confirmed on the stored circuits (`h2_platform_noise_study.py` V1) |
| H2-other-platforms-C3 | The 12 check couplings form the 3x3 grid (8 embeddings, data 1 at the centre). Edge (2,3) closes the odd 5-cycle 0-7-2-3-5-0, so no bipartite lattice fits. Data 2 and 3 are at distance 4 in every embedding, and every shortest path between them passes through another data qutrit. | verified numerically | A3 (networkx on the repository's check circuits) |
| H2-other-platforms-C4 | Hrmo's LS gate at theta = ±2pi/3 equals (D x D) CZ^{±1} with D = diag(1,w,w) = Z P^2, and CNOT = (I x H^dag)(D^dag x D^dag) G(+2pi/3) (I x H). Each check CNOT is one native LS gate plus local Cliffords on both qutrits; without the D factors the product is not a CNOT. | verified analytically | 1 - delta_jk = (j-k)^2 mod 3; confirmed with sdim's matrices in A2 |
| H2-other-platforms-C5 | Every gate of the circuits normalizes the qutrit Weyl group. Logical H and P act on syndromes by fixed 4x4 Z_3 transfer maps (H swaps Z- and X-type syndromes with a sign; P adds -s_Z to s_X), so frame tracking needs the known Clifford sequence. | verified numerically | A4 (all 9 one-qutrit and 81 two-qutrit Weyl images from sdim matrices; transfer maps solved exactly) |
| H2-other-platforms-C6 | Reset is unnecessary: on the full 9-qutrit encoded block an ancilla starting in state k returns k + s(E) deterministically for all four checks (100 cases). | verified numerically | A6 statevector; no-reset practice discussed in Ge25 |
| H2-other-platforms-C7 | Native 40Ca+ qudits have: LS two-qutrit gate 98.7(2)% at d = 3 (two ions, global beams, SPAM-corrected per-gate decay, slow non-Markovian noise noted, simulated budget 1.3e-2); qutrit Clifford RB 2(2)e-3; worst-case qutrit readout error about 3e-3; about 100 ms coherence; addressed MS gates 0.993 (3 ions) and 0.983 (10 ions) with crosstalk about 2e-3. | from literature | H23, R22 (full texts) |
| H2-other-platforms-C8 | Ion qutrit readout takes 3.5 to 4.5 ms (sequential shelving with 500 µs to 1 ms detection windows and 2.5 ms recooling). In single-species 171Yb+, in-situ qubit MCMR with shelving leaves 98.8(6)% or 98.2(7)% data fidelity, with a projection below 0.1% error. | from literature | R22, M25, Y25 |
| H2-other-platforms-C9 | Ion qudit readout heralds out-of-code population as never-bright shots (13-level raw SPAM error 13.1% against 8.3% postselected). Qubit-pair qutrits on Quantinuum H2 herald leakage through the unused state 01 (about 11% of shots discarded) and support real-time feed-forward after qutrit measurements. | from literature | L23, R22, I25 |
| H2-other-platforms-C10 | Apart from transmon qutrits, qubit-pair-encoded qutrits on an all-to-all QCCD machine are the only option that can run the 9-qutrit check-round circuit today; a qutrit CNOT then costs about 10 two-qubit gates. | conjecture | I25 demonstrates the ingredients at 24 qutrits; gate count from I25's compilation (3 ZZPhase per Fourier, 4 controlled-RZ per controlled-Z) |
| H2-other-platforms-C11 | No multi-qudit control in neutral-atom arrays had been demonstrated as of Bu26. The atom qubit stack has 99.5% parallel two-qubit gates, 500 µs imaging at about 99.8% with at most 0.1% data error, an FPGA readout-plus-feedforward cycle under 1 ms, software-applied Clifford corrections, and erasure conversion (0.980 two-qubit fidelity, <1e-5 disturbance). | from literature | Bu26, E23, Bl24, Ma23, W22 |
| H2-other-platforms-C12 | Photonic two-qudit gates between photons succeed with probability 1/d (MVCX process fidelity 0.952, about 1e3 two-photon detections per second). Deterministic two-qudit gates exist only between degrees of freedom of a single photon. | from literature | C22 (Europe PMC full text), Im19 |
| H2-other-platforms-C13 | At success 1/3 per two-qutrit gate, one check round succeeds with probability 1.9e-6 and a depth-22 shot (about 413 gates) with 10^-197. | verified numerically | `h2_budget.py` T5; the 1/3 rate extrapolates C22's 1/d to d = 3 |
| H2-other-platforms-C14 | A GKP qutrit beats break-even (gain 1.82 ± 0.03; 886 ± 3 µs against 488 ± 7 µs; 7 µs rounds; ternary measurement by two binary ancilla measurements) as a single-mode memory. The only two-mode logical GKP gate found is between qubits in trapped-ion motion (process fidelity 0.680). | from literature | Br25, Si23, Mt24 |
| H2-other-platforms-C15 | Spin and molecular qudits are single-object demonstrations: a 16-dimensional Sb donor qudit, cat states of a spin-7/2 nucleus, and Grover search on a spin-3/2 molecular magnet. | from literature | F24, Yu25, Go17 |
| H2-other-platforms-C16 | Transmon qutrit reference: CZ† 97.3(1)% (580 ns) and CZ 95.2(3)% (783 ns) process fidelity, 1–2 manifold Ramsey T2 17(8) µs; single-qutrit infidelity down to 2.38e-3; FPGA qutrit classification 97% from a 500 ns pulse with 66 ns inference. | from literature | Go22 (full text), Mo21, X23 |
| H2-other-platforms-C17 | Applied location by location to the baseline circuits, ion_now behaves like SI1000 at p ≈ 3e-3 to 5e-3 (a_u1 0.67, conditional r_u1 7.2e-4, r_MLD 0.078), not like p ≈ 0.014. ion_next behaves like p ≈ 1e-3 to 1.5e-3, and qubit_pairs like ion_now. transmon_opt is worse than SI1000 at 5e-3 even with no idle noise. Spectator dephasing dominates the ion uncertainty (r_MLD 0.096 at 2.9%, 0.138 at 6.2%). | verified numerically | `h2_platform_noise_study.py` R1, R2; model inputs are literature values or conjectures as tabulated in Section 3.1 |
| H2-other-platforms-C18 | MLD correction gives r = 0.018 to 0.14 per depth step on every platform model, far above the about 4e-3 physical reference for ions (two qutrit Cliffords at 2e-3), so correction on this d = 2 code is below break-even. Conditional unif-1 detection gives 3.3e-5 to 1.45e-3 on the ion models at 0.63–0.89 acceptance per step (4.8e-3 in the worst-case idle model). | verified numerically | R2; physical reference from R22 |
| H2-other-platforms-C19 | The relabelled-model pipeline is validated: kernels identical to the cached structure at p = 1e-3, m = 0 LRB-D equal to the baseline CSV (rejection 0.0611), MLD within about two standard errors of the decoder study, and SI1000 rates at p = 1e-3 and 5e-3 matching results/key_summary.md. | verified numerically | `h2_platform_noise_study.py` V1, R2 |
| H2-other-platforms-C20 | At SI1000 p ≥ 0.0125 and m ≥ 10, const 1 rejects exactly 1 - 1/81 and the accepted-shot legacy fidelity (about 0.008) equals the Rayleigh floor sqrt(pi/(4 N_acc)) of the legacy metric, so those baseline points carry no information. | verified numerically | `h2_budget.py` T6 on the read-only CSVs; floor formula checked by Monte Carlo in A8 |
| H2-other-platforms-C21 | For ±10% (one standard error) on r from depths 0 and M with 30 sequences, MLD correction needs about 15 to 40 shots per circuit, LRB-D unif 1 needs 3e4 to 1.75e6 (117 h of ion time for two depths on ion_now), and const 1 needs about 600 to 8e3. The between-sequence spread adds at most 4%. | verified numerically | `h2_budget.py` T3; estimator variance (1-l)(1+2l)/(2N) checked in A8 |
| H2-other-platforms-C22 | An ion qutrit check round takes about 9.6 ms (12 sequential CNOTs plus a 4.5 ms readout). An m = 22 shot takes about 0.29 s, and the 270-circuit dataset 11 h per 1e3 shots per circuit. A transmon round takes about 6 µs, and 1e6 shots per circuit take 0.7 days. Qubit-pair QCCD rounds take about 0.13 s. | conjecture | `h2_budget.py` T1, T2; inputs P21, H23, M25, Go22, X23 plus labelled estimates |
| H2-other-platforms-C23 | Measured feedback and decoding latencies: FPGA loops 0.18 to 0.55 µs (QICK, Zurich Instruments, Qblox, an OPX-family controller); QubiC board-to-board 71 ns; Riverlane FPGA decoder response 9.6 µs for 9 rounds; GPU links about 4 µs; GPU and workstation decoders 63 to 67 µs; AQT ion controller about 5 µs; neutral-atom readout plus feedforward under 1 ms; Quantinuum 2021 conditional distribution under 250 µs. | from literature | St21, X23, ZI-FB, ZI-PQSC, QB, Cao26, Ca24, NV25a, NV25b, A25, P21, Bl24, RA21, Si23 |
| H2-other-platforms-C24 | Clifford LRB needs no real-time decoding. Every gate is Clifford, the noise is Weyl and nothing is conditional, so the decoder output is an offline Z_3 shift of x0 + x3 (from the Z exponents before the terminal H_INV). Only storage of 4(m+1) trits per shot must run in real time, and reset can be skipped. | verified analytically | argument in Section 5.2 using A4 and A6; T15 states the same for Clifford circuits; consistent with Bl24, RA21, A25 practice |
| H2-other-platforms-C25 | T3 = diag(1, z, z^8) is in the third Clifford level but not Clifford. A frame error with an X component before T3 gives X-basis statistics that are not a cyclic shift of the ideal ones, so no final relabelling repairs it; Z-type frame errors only shift them. | verified numerically | A5 (sdim matrices; membership in the Howard–Vala family checked) |
| H2-other-platforms-C26 | Non-Clifford gadgets need the frame before the next gate: an X before a T gadget becomes a Clifford error, and qudit distillation applies outcome-dependent Clifford corrections. Decoding throughput must keep up with syndrome generation, or the logical clock must wait. | from literature | T15, CAB12, HV12, CIP18, Ca24 |
| H2-other-platforms-C27 | Every listed decoding or feedback latency is at most 0.026 of a 9.6 ms ion round (FPGA loops about 5e-5). On a 6.3 µs transmon round, FPGA loops are 0.03 to 0.09 of a round and GPU or workstation decoders about 10 rounds. A per-round lookup table for this code has 81 entries (6561 entries, 2.4 KiB, for two rounds). | verified numerically | `h2_budget.py` T4 (ratios of C23's latencies to T1's rounds); A7 |
| H2-other-platforms-C28 | Not demonstrated as of my searches: a 9-qutrit ion register with addressed native qutrit entangling gates, mid-circuit readout on a multi-qudit ion register, a two-mode logical gate between GKP qutrits, and a multi-qudit spin or molecular register with entangling gates. | conjecture | absence in my searches on 2026-10-28 |
| H2-other-platforms-C29 | Real-time non-Clifford operation on transmons needs FPGA-resident decoding, while on ions any decoder suffices. On the d = 2 code a T-gate experiment needs only real-time feed-forward of raw outcomes and a reject flag, not decoding proper. | conjecture | built on C26, C27 |
| H2-other-platforms-C30 | Trapped ions are the most sustainable platform for this experiment: qubit-pair qutrits on a QCCD machine today, then native 40Ca+ qutrits once addressed LS gates in 9 or more ions and mid-circuit qutrit readout with about 1e-3 spectator error exist. The protocol on every platform is offline decoding, one parallel readout layer per round, no reset, and correction or sparse checks instead of unif-1 postselection. Transmons remain the throughput option. | conjecture | judgment built on C3 to C29 |
