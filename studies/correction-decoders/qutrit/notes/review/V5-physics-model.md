# V5 review: the physics model

Reviewer V5-physics-model, 2026-10-28.

**Lens.** Is the simulated model what the brief says? Do the decoders' conclusions depend on hidden assumptions?

**Where things are.**
- Scripts: `notes/review/V5/`. Run them from that folder after `module load python/3.11`.
- Outputs: `notes/review/V5/out/`.
- `lite_dem.py` caches compiled DEMs of the 270 + 270 baseline circuits in a scratch folder. Each check script rebuilds the cache if it is missing.
- Nothing under `src/`, `scripts/` or `tests/` was edited.

Labels used below: [VA] verified analytically, [VN] verified numerically, [LIT] from a source I fetched, [CONJ] conjecture.

## Verdict

1. **The circuits match the brief exactly [VN].** The 270 ordinary circuits and the 270 const0 circuits contain no noise location that breaks the brief's generalized-SI1000 rules. The encoding is noiseless, and so is the terminal readout.
2. **The compact DEM is exact for these circuits [VN].** An exact 9-qutrit statevector simulation, which shares no code with sdim, reproduces every unit-fault response. That is 3620 unit faults with 0 mismatches. RESET leaves no stale frame components.
3. **The 4.95 p floor is correct [VN].** In all 270 circuits it comes from exactly five final-round locations and does not depend on m. Reading all five data qutrits brings it to 0. It stays 0 even when that readout is noisy.
4. **No finding in my lens changes a reported number.** Five findings are minor: documentation issues and two latent sdim hazards that do not touch the production circuits. One critical finding is outside my lens: the m = 0 weight in the decay fits.
5. **The decoder ranking held in every test of hidden assumptions [VN].** The ranking is, from best r to worst: LRB-D, BFD, LRB-D const 1, MLD. The size of the gaps moves by up to about 3×.

## Findings (most severe first)

| id | severity | where | issue | evidence / repro |
|---|---|---|---|---|
| V5-F1 | critical (outside my lens: fits) | `src/lrb/correction/metrics.py:45-53`, `aggregate.py:177` | See the note below the table. | `aside_m0_weight.py`, `out/aside_m0_weight.txt` |
| V5-F2 | minor (documentation) | `notes/00-brief.md:102-104`; generator `src/lrb/circuit_generator.py:449-466` | See the note below the table. | `check1_noise_locations.py` (per-layer list); `check5_hidden_assumptions.py --part a`; fetched paper, see Sources |
| V5-F3 | minor (latent, sdim) | sdim 1.3.4 `program.py:704-714` (detector parser) | Expressions are rewritten by sequential `str.replace` of `[k]` with `[j]`. `rec[1]-rec[0]` therefore becomes `rec[1]-rec[1]` = 0. The DEM then silently drops responses. | `check4_frame_exactness.py`, section (d). In the micro-circuit, an X fault after RESET gives DEM (0,0) where the statevector gives (1,1). With `-rec[0]+rec[1]` both give (1,1). The audit in (d2) re-evaluated all 14129 DETECTOR/OBSERVABLE expressions of the baseline and variant circuits through sdim's own parsed lambdas: 0 differ, so production is unaffected. This is a hazard for new detectors, such as round differences written newest-first. |
| V5-F4 | minor (latent, sdim) | sdim 1.3.4 `program.py:163-166` (M_X in `simulate_frame`) | The frame is rotated into the X basis for readout and never rotated back. A later operation on that qutrit then sees a wrong frame. | `check4c_mx_frame.py`, `out/check4c.txt`. Two noiseless consecutive `M_X` on one qutrit disagree in 1318 of 2000 frame shots. `H_INV`+`M` gives 0 of 2000. No production circuit uses `M_X` (gate inventory in check4 (a)). The generator does support `M_X` (`circuit_generator.py:457-459`). |
| V5-F5 | minor (cosmetic, no numerical effect) | `src/lrb/correction/chain_dem.py:368-371` | The depth-0 fallback terminal map `[[0,0,1,0],[0,0,0,1]]` has the wrong sign. The map learned at every depth ≥ 1 is `[[0,0,2,0],[0,0,0,2]]`. An X-check ancilla reads −(z0+z1−z2); the terminal X-stabilizer reads +(z0+z1−z2) [VA]. | Printed `structure.terminal_map` for `term_c00_d0/d1/d4`, `par_term_c00_d3` and `par_term_chk_c00_d2`. It is never used numerically, because at depth 0 the only slot is entered with h = 0 (`forward.final_matrix`, `decoders.window_decode`). The comment "projection guess" is misleading. |
| V5-F6 | minor (documentation) | `notes/00-brief.md:163`; `results/key_summary.md:33,44` | (a) The brief's "`const 0` 0.99999 to 0.9989" is off at m = 0. The depth-0 const0 circuit has no noise location (check1), so F = 1 exactly and rejection is 0 (`const_check_data/0.csv`). The value 0.99999 is the m = 2 value. (b) `soft:0.99` shows "n/a" at p ≥ 2e-3 with no reason given. The undetectable floor caps every posterior confidence: P(L = 0 \| all rounds trivial, m = 0) = 0.989875 at p = 2e-3. So no shot reaches 0.99 and the strategy accepts nothing. This follows from reading only q0 and q3; it is not a decoder failure. | Production `c00_p0p002.npz`, `d0__soft_hist_rb`: no mass with 1−conf ≤ 0.01. At p = 1e-3 the cap is 0.994994. |

**V5-F1: the m = 0 point decides every decay fit.**
- For every exactly evaluated strategy, the across-sequence std at m = 0 is about 1e-16 or exactly 0. All 30 sequences share one depth-0 circuit.
- `_safe_weights` floors only errors ≤ 0, so this point gets a weight of about 1e31, or about 1e6 × the median weight when the error is exactly 0. The offset-0 fit `y = b f^m` is then forced through y(0).
- The baseline pipeline never had a zero-variance point, because its m = 0 data were sampled (std 8.6e-5 for unif 1 at p = 1e-3).
- Giving m = 0 the median weight of the other depths changes r beyond the quoted 95% CIs. At p = 1e-3:

  | strategy | r (pinned) | r (m = 0 at median weight) |
  |---|---|---|
  | `lrbd:all` | 2.00e-5 [1.95e-5, 2.04e-5] | 1.85e-5 |
  | `lrbd:const1` | 2.96e-4 [2.53e-4, 3.49e-4] | 4.99e-4 |

- Across p ∈ {5e-4, 1e-3, 2e-3}: `bfd:all` changes ×1.07, `mld:all` ×1.01, `mld:terminal` ×0.96.
- The bootstrap CI does not include this dependence on the estimator. The ranking does not change.

**V5-F2: the measurement rule is not the published SI1000.**
- The brief does not say where the "depolarizing p on the measured ancilla" sits. In the code it is applied **before** M.
- The published SI1000 has no such channel. Its MZ(5p) is only a 5p bit flip before the measurement [LIT].
- The extra channel adds 0.75 p to every ancilla outcome flip (5.75 p instead of 5 p) and 3 p to the 41 p of single-round blip mass per round.
- The published Idle(p/10) during gate steps is absent. The brief does say this.
- `N1-baseline-nonmarkov.md:605-607` says the SI1000 attribution is unconfirmed. I fetched the paper and it is confirmed (Sources).

## Checks run

### (1) Every noise location against the brief

`check1_noise_locations.py --all`, output in `out/check1.txt`. This is an independent classifier written from the circuit text; it does not call the generator.

**Depth-2 baseline circuit (Clifford 0, p_idx 4).**
- 398 ops and 246 noise locations, all listed with op index, block, type, wires, channel and prob_scale. There are 0 rule violations.
- The encoding (`H_INV 2,3,4; CNOT_INV 2→1; CNOT 1→0; CNOT 3→0; CNOT 4→2`) has no noise.
- The terminal readout (`H_INV 0, H_INV 3, M 0, M 3`) has no noise.

Per check round the summed scales are:

| block | locations | summed scale |
|---|---|---|
| CNOT and CNOT_INV, N2 | 12 | 12 p |
| H/H_INV on ancillas 7 and 8 | 4 | 0.4 p |
| idle in each of the 4 measurement layers, on all 8 other wires | 4 × 8 | 4 × 16 p |
| depolarizing on the measured ancilla, before M | 4 | 4 × p |
| X-shift before M | 4 | 4 × 5 p |
| reset layer: idle on data | 5 | 10 p |
| reset layer: X-shift after RESET | 4 | 8 p |

That totals 118.4 p per round plus reset.
- Each data qutrit gets 8.0 p of idle in the measurement layers and 2.0 p in the reset layer, as the brief says.
- Logical gates carry 0.1 p after every 1-qutrit gate, including the Paulis, the `_INV` forms and the identity token `I` on wire 0. They carry p after SWAP and CZ.
- prob equals p × prob_scale at every location.

**All 30 × 9 ordinary and 30 × 9 const0 circuits.**
- 0 violations. Each ordinary circuit has m + 1 rounds.
- The const0 circuits carry only gate noise. They have no mid-circuit M or RESET and a noiseless readout of all five qutrits. The depth-0 const0 circuit has no noise at all.
- The circuit structure is identical across p_idx 1, 4, 8 and 14; only the probabilities differ.

### (2) The logical observable

`check2_logical_observable.py`, output in `out/check2.txt`.
- Detector k reads `rec[k]`, which is the k-th ancilla M. RESET is not a `rec` target.
- `lrb_logical = rec[12]+rec[13]` is the terminal `M 0` and `M 3`, each preceded by `H_INV`. So it is the X-basis readout x0 + x3 [VN].
- In the frame, `H_INV` sends z to x, so a Z^b on q0 or q3 just before readout shifts the observable by +b [VA].

Single-fault examples from the DEM (final round of the depth-2 circuit):

| fault | X component | Z component |
|---|---|---|
| idle on q0 in the M8 layer | logical 0, no detector | logical 1, no detector |
| idle on q3 in the M8 layer | logical 0, no detector | logical 1, no detector |
| idle on q1, q2 or q4 in the M8 layer | 0 | 0 |
| idle on q0 in the M5 layer | 0 | logical 1, `r2_w7` = 2 (detected) |
| N2 after CNOT 7→0, on the target (q0) | logical 0 | logical 1, no detector |

For the N2 after CNOT 7→0, the Z component on the control (ancilla 7) flips `r2_w7` only.

Systematic check: 1890 data-qutrit locations lie after the last ancilla CNOT on their qutrit. In every one, X never shifts the observable, Z shifts it by exactly 1 on q0 and q3 and by 0 on q1, q2 and q4, and no detector fires. 0 deviations.

### (3) The 4.95 p floor

`check3_undetectable_floor.py`, output in `out/check3.txt`. This is my own enumeration over every (location, Weyl vector) from the DEM's unit responses, with weight prob_scale / n_vectors.

**All 270 baseline circuits give exactly 4.950000 p [VN].** The same five locations appear, at the same positions relative to the end of the circuit:

| location (final round) | vectors that flip the observable silently | contribution |
|---|---|---|
| N2 after CNOT 7→0 (Z on q0 and none on ancilla 7) | 18/80 | 0.225 p |
| idle on q0 in the M7 layer | 6/8 | 1.5 p |
| N2 after CNOT 8→3 (Z on q3 and none on ancilla 8) | 18/80 | 0.225 p |
| idle on q0 in the M8 layer | 6/8 | 1.5 p |
| idle on q3 in the M8 layer | 6/8 | 1.5 p |

- The claim's wording "the last data-ancilla CNOTs of the final X-checks" needs care. CNOT 7→0 and CNOT 8→3 are the **first** CNOTs of their X-checks. They are the **last** ancilla interaction of q0 and of q3.

**Protocol variants:** 3 sequences × depths {0, 1, 2, 4, 10}, with the terminal detectors counted. Each variant gave the same value in all 15 circuits.

| variant | floor (× p) |
|---|---|
| base | 4.95 |
| chk | 4.95 |
| par | 3.45 (idle on q0 and q3 in the single readout layer, 2 × 1.5, plus 2 × 0.225 from the CNOTs) |
| term | 0 |
| par_term | 0 |
| par_term_chk | 0 |

So reading all five data qutrits makes every such fault detectable, but parallel readout alone does not.

**Same with a noisy terminal readout** (`check6_noisy_terminal_readout.py`). The terminal readout layer gets the circuits' own ancilla measurement rule: p depolarizing and a 5p shift before M, 2p idle on the other wires, and 0.1p after each `H_INV`.
- base floor: 4.95 → 16.60 p.
- term floor: stays 0.
- First-order MLD failure rises by about 11.65 p in both, independent of m. This is SPAM only.

**BFD claim (`decoders.py:139-141`)** (`check3b_bfd_first_order.py`). Over all 270 circuits, the single-fault mass that BFD accepts, that fires a detector and that shifts the observable is 0.0. BFD forgives only blips (74 p at m = 2 up to 894 p at m = 22). Its silent logical mass is the same 4.95 p.

### (4) Is the frame simulation exact?

`check4_frame_exactness.py`, output in `out/check4.txt`; `check4b_tableau_vs_statevector.py` and `out/check4b.txt`; `check4d_pre_reset_faults.py` and `out/check4d.txt`.

**(a) Gate inventory.** The baseline circuits use CNOT, CNOT_INV, CZ, CZ_INV, H, H_INV, I, M, N1, N2, P, P_INV, RESET, SWAP, X, X_INV, Z, Z_INV, plus annotations. The variants use the same set without I. Nothing outside the Clifford + Pauli + M + RESET set, and no `M_X`.

**(b) Frame rules [VN].** I transcribed sdim's frame updates (`program.py:116-154`). For each of 486 (gate, input Pauli) pairs, U P U† is proportional to the Pauli sdim produces. U is the standard qutrit unitary: H|j⟩ = Σ_k ω^{jk}|k⟩/√3, P = diag(1, 1, ω), CNOT|a,b⟩ = |a,a+b⟩, CZ|a,b⟩ = ω^{ab}|a,b⟩. There are 0 mismatches.

**(c) Exact statevector versus DEM [VN].**
- The statevector simulation has 19683 amplitudes, projective mid-circuit M, and RESET implemented as measure then X^{−k}.
- Every unit fault of every location was inserted, in the depth 0, 2 and 4 circuits of sequences 0 and 7.
- 3620 unit faults gave 0 mismatches in both detector and observable shifts.
- In the noiseless circuits, every ancilla outcome has probability 1 and the observable is deterministically 0, matching sdim's reference.

**(d) RESET.**
- Micro-circuit: an X or Z fault between M and RESET leaves the next detector silent; an X fault after RESET flips it. The DEM and the statevector agree.
- All 270 circuits: 18900 ancilla N1 locations sit between that ancilla's M and its RESET, or the circuit end. 1620 of them are in final rounds. All have a zero response.

**(e) Randomization leakage.**
- With all noise set to zero, 2000 frame shots of the depth-2 and depth-22 circuits gave 0 non-zero detector or observable events.
- Two DEM compilations with different numpy seeds gave identical responses.

**(4b) sdim's tableau is the standard Clifford set [VN].** I replayed 300 random 3-qutrit circuits over the full gate set with M and RESET on the statevector, forcing each outcome to sdim's. All 2009 outcomes agree with sdim's deterministic/random flags. The noiseless depth-2 LRB circuit replays with 0 mismatches.

### (5) Hidden assumptions: tests and direction

**How these were tested.**
- Sensitivity tests (`check5_hidden_assumptions.py`, `check7_gate_layer_idle.py`) change the mechanisms of the production chain structures, which were read only. They then use the package's exact LRB-D and BFD and its sampled MLD.
- Setup: p = 2e-3; sequences 0-3; m ∈ {0, 2, 4, 6, 10}; 20000 shots.
- Each r is the package's own offset-0 fit of the signed metric, and a is the acceptance per step.
- Read these as relative changes, not as production numbers.

| model | LRB-D unif 1: r (a) | LRB-D const 1: r (a) | BFD: r (a) | MLD: r |
|---|---|---|---|---|
| baseline | 9.7e-5 (0.799) | 7.8e-4 (0.870) | 2.5e-4 (0.862) | 2.71e-2 |
| measurement-type faults ×2 (5p flip, p depol before M, 2p reset flip) | 1.16e-4 (0.751) | 9.4e-4 (0.870) | 2.7e-4 (0.853) | 2.82e-2 |
| no idle noise on ancillas (as for T1-limited \|0⟩) | 9.1e-5 (0.814) | 7.2e-4 (0.870) | 2.4e-4 (0.864) | 2.68e-2 |
| data idle pure Z (dephasing), same 2p rate | 2.67e-4 (0.800) | 1.69e-3 (0.872) | 5.9e-4 (0.863) | 3.76e-2 |
| data idle pure X, same 2p rate | 3.26e-4 (0.800) | 1.53e-3 (0.872) | 6.3e-4 (0.863) | 3.60e-2 |
| + SI1000 Idle(p/10) on every idle wire in each ASAP gate step | 1.42e-4 (0.779) | 9.9e-4 (0.855) | 3.4e-4 (0.845) | 3.07e-2 (baseline in the same run: 2.72e-2) |

In the gate-step test, ASAP scheduling places gates on disjoint wires in one step; measurement and reset layers are barriers. This adds 116 idle locations (+11.6 p of fault mass) at m = 0 and 1873 locations (+187 p) at m = 10.

**Decoder prior mismatch** (`check5 --part c`). Realized MLD failure at m = 0 / 2 / 4 / 6 / 10:

| truth → decoder prior | m = 0 | 2 | 4 | 6 | 10 |
|---|---|---|---|---|---|
| oracle prior (true model) | 0.0283 | 0.0822 | 0.1266 | 0.1698 | 0.2428 |
| flat prior, every prob_scale set to 1 | 0.0283 | 0.0833 | 0.1286 | 0.1726 | 0.2542 (+4.7%) |
| measurement rates divided by 5 | | | | | 0.2440 (+0.5%) |
| truth Z-biased idle, depolarizing prior | 0.0355 | 0.1081 | 0.1674 | 0.2277 | 0.3186 |
| Z-biased oracle (for comparison) | | | | | 0.3118 |

So MLD barely depends on knowing the exact model at d = 2. Its failures come from intrinsic ties, not from the prior.

**Where blips come from** (`check5 --part a`, bulk slot 5 of a depth-10 circuit).
- LRB-D first-order rejection starting in the slot is 107 p:
  - 66 p are persistent data faults; data idle alone accounts for 50 p.
  - 41 p are single-round blips.
- The blips break down as:

  | source | mass |
  |---|---|
  | 5p measurement flips | 20 p |
  | idle on ancillas still waiting in \|0⟩ during earlier measurement layers of the same round | 9 p |
  | reset flips | 8 p |
  | depolarizing before M | 3 p |
  | other | 1 p |

**Excluded or idealized effects, and which strategy each one favours.**

| effect | in the model? | evidence | favours |
|---|---|---|---|
| measurement and reset errors larger or non-uniform | uniform 5p flip plus p depol, 2p reset | [VN] ×2: LRB-D acceptance per step falls 0.799→0.751, BFD 0.862→0.853. r rises 20% (LRB-D), 11% (BFD), 4% (MLD). | correction and BFD, in acceptance. Ranking in r unchanged. State-dependent asymmetry (\|2⟩ read worst) cannot affect detection [VA] and barely affects MLD (prior test) [VN]. |
| idle depolarizing on waiting \|0⟩ ancillas (in the model; probably pessimistic for transmons, where T1 does not excite \|0⟩) | yes, 9 p of blips per round | [VN] removing it raises LRB-D acceptance by 0.015 per step and BFD by 0.002 | the model's choice inflates BFD's acceptance advantage over LRB-D by about 25%. On hardware, detection [CONJ] |
| idle noise during gate steps (SI1000 Idle(p/10)) | no | [VN] r rises 47% (LRB-D), 37% (BFD), 13% (MLD); MLD/LRB-D ratio 281→216 | narrows detection's lead, ranking unchanged |
| noise bias (dephasing-dominated T2, or the Pauli twirl of amplitude damping) | no, unbiased depolarizing | [VN] Pauli proxies: biased idle raises LRB-D r 2.8–3.4×, BFD 2.4–2.6×, MLD 1.3–1.4×. True amplitude damping is non-Pauli and \|2⟩→\|1⟩→\|0⟩ cascades [CONJ]. | narrows detection's lead in r. Ranking unchanged in the Pauli model. |
| leakage to \|3⟩ (larger in transmon qutrits) | no (Pauli frame only) | none | detection [CONJ]: leakage gives multi-round random syndromes. LRB-D and BFD (consecutive rounds) reject them. MLD turns them into wrong corrections. |
| correlated two-qutrit errors (ZZ or cross-Kerr crosstalk on idle neighbours) | no; N2 noise only follows gates, which act on (2,3) or ancilla-data pairs, so never on a logical pair | [VA] the weight-2 logicals are exactly X0X3, X2X4, Z0Z2, Z3Z4 and their powers (A1, re-derived). Correlated errors on those pairs are silent logical errors. | neither in absolute terms, but they remove detection's p² protection, so its relative advantage shrinks most [CONJ] |
| noiseless terminal readout and prep | yes, idealized | [VN] first order, see (3) | SPAM only. With the 2-qutrit readout the floor grows to 16.6 p; with full readout it stays 0. MLD gains about 11.65 p either way. |
| oracle decoder prior (Rao-Blackwellized scoring assumes it) | yes | [VN] mismatch costs ≤ 5% | not a material advantage for correction |
| drift and time-correlated (non-Markovian) noise | no | none | detection [CONJ]: it needs no model, while a stale prior mis-weights MLD ties |

**Conclusions that hold in every test.**
- Order in r: LRB-D unif 1 < BFD < LRB-D const 1 < MLD.
- MLD's r stays at least 100× LRB-D's.

**Conclusions that move by up to about 3×.**
- The absolute LRB-D and BFD decay rates: bias and idle structure.
- The acceptance gaps: measurement error rate and waiting-ancilla idle.

**Not re-tested.** I did not recompute physical RB under these changes, so whether "MLD is worse than physical RB" survives each change is [CONJ]. It has a 2.4–3× margin at p = 1e-3.

## Reproduction

```bash
module load python/3.11
cd studies/correction-decoders/qutrit/notes/review/V5
OMP_NUM_THREADS=3 python lite_dem.py --workers 3 --kinds lrb,const0   # ~5 min, DEM cache
python check1_noise_locations.py --all          # (1)  ~3 min
python check2_logical_observable.py             # (2)  ~10 min (recompiles 270 DEMs)
python check3_undetectable_floor.py             # (3)  ~2 min
python check3b_bfd_first_order.py               # (3b) ~1 min
python check6_noisy_terminal_readout.py         # (3)  ~1 min
python check4_frame_exactness.py --depth-indices 0,1,2 --cliffs 0,7   # (4) ~4 min
python check4b_tableau_vs_statevector.py; python check4c_mx_frame.py; python check4d_pre_reset_faults.py
python check5_hidden_assumptions.py --part a,b,c   # (5)  ~6 min
python check7_gate_layer_idle.py                # (5)  ~2 min
python aside_m0_weight.py                       # V5-F1
```

## Sources

- C. Gidney, M. Newman, A. Fowler, M. Broughton, "A Fault-Tolerant Honeycomb Memory", Quantum 5, 605 (2021). arXiv:2108.10457, DOI 10.22331/q-2021-12-20-605.
  - Fetched: https://arxiv.org/abs/2108.10457 and https://arxiv.org/pdf/2108.10457v2. I extracted the PDF text with pypdf from the worktree's existing `.venv-pdf`.
  - Table 1 defines MZ(p) as "Precede with a bitflip channel of strength p, and measure the qubit in the Z-basis", and ResonatorIdle(p) as "If the qubit is not measured or reset in a time step during which other qubits are being measured or reset, apply a 1-qubit depolarizing channel of strength p".
  - Table 2 lists SI1000 = CZ(p), AnyClifford1(p/10), InitZ(2p), MZ(5p), Idle(p/10), ResonatorIdle(2p).
