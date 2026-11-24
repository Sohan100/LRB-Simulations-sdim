"""H2 (other platforms): gate counts, time budgets and shot budgets.

Run with:  module load python/3.11 && python3 H2_platform_budget.py
Runtime: about 10 s on one core. Reads the 216-element Clifford catalog
from src/lrb/code_definitions.py of this worktree and the baseline LRB-D CSVs
of the read-only run folder. Writes nothing.

What it computes
  B1  Average physical gate content of one logical Clifford and of one check
      round of the [[5,1,2]]_3 LRB-D circuit (counts from code_definitions.py).
  B2  Per-round and per-shot durations for each platform from literature gate,
      readout and reset times (inputs listed in PLATFORMS with their sources;
      entries marked "estimate" are my assumptions, not measured values).
  B3  Wall-clock time for the baseline dataset size (30 sequences x 9 depths x
      N shots) on each platform, and the shot inflation 1/(1 - rejection) that
      detection (LRB-D) needs at the platform-matched SI1000 p, taken from the
      baseline results.
  B4  Photonic success probability for a probabilistic two-qudit gate with
      success 1/d per gate (Chi et al. 2022) in one check round and one shot.

All durations are order-of-magnitude budgets: no gate parallelism inside a
round unless stated, sequential single-ancilla readout as in the simulated
circuit ("seq") and one parallel readout layer per round ("par").
"""
from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
from collections import Counter

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# Load code_definitions.py directly so that the package __init__ (which pulls
# in the plotting modules) is not imported.
_spec = importlib.util.spec_from_file_location(
    "h2_code_definitions", os.path.join(WORKTREE, "src", "lrb", "code_definitions.py"))
_cd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cd  # dataclasses look the module up here
_spec.loader.exec_module(_cd)
CATALOG = _cd.single_qutrit_cliffords_exhaustive_strings

BASELINE = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
            "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000/results/LRB")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_GRID = [0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05]

# ---------------------------------------------------------------- B1 counts
tok = Counter("".join(CATALOG))
n_cliff = len(CATALOG)
avg = {g: tok.get(g, 0) / n_cliff for g in "HPXZ"}
# Physical content per logical token (FoldedQutritDetectionCode, read in this task):
#   H: SWAP(2,3) + H on {0,2,3,4} + H_INV on 1   -> 1 two-qutrit, 5 one-qutrit
#   P: P on {0,4} + P_INV on 1 + CZ(2,3)          -> 1 two-qutrit, 3 one-qutrit
#   X: X on {0,3};  Z: Z on {0,2}                 -> 0 two-qutrit, 2 one-qutrit
two_per_cliff = avg["H"] + avg["P"]
one_per_cliff = 5 * avg["H"] + 3 * avg["P"] + 2 * avg["X"] + 2 * avg["Z"]
two_per_cliff_virtual_swap = avg["P"]                     # SWAP done by relabelling
one_per_cliff_virtual_pauli = 5 * avg["H"] + 3 * avg["P"]  # X, Z tracked in software
ROUND = dict(two=12, one=4, meas=4, reset=4)

print("=" * 76)
print("B1  gate content")
print(f"  catalog size {n_cliff}; mean tokens per Clifford: "
      + ", ".join(f"{g} {avg[g]:.3f}" for g in "HPXZ"))
print(f"  per logical Clifford: {two_per_cliff:.3f} two-qutrit gates, {one_per_cliff:.2f} one-qutrit gates")
print(f"    with virtual SWAP(2,3): {two_per_cliff_virtual_swap:.3f} two-qutrit; with X/Z as frame "
      f"updates: {one_per_cliff_virtual_pauli:.2f} one-qutrit")
print(f"  per check round: {ROUND['two']} two-qutrit, {ROUND['one']} one-qutrit, "
      f"{ROUND['meas']} measurements, {ROUND['reset']} resets")
for m in (2, 10, 22):
    n2 = 2 * m * two_per_cliff + (m + 1) * ROUND["two"]
    n1 = 2 * m * one_per_cliff + (m + 1) * ROUND["one"]
    print(f"  depth m = {m:2d}: {2*m} Cliffords + {m+1} rounds -> {n2:.0f} two-qutrit, {n1:.0f} one-qutrit, "
          f"{(m+1)*ROUND['meas']} mid-circuit measurements")
# SI1000 fault-location budget per check round (brief, apply_si1000_noise_to_circuit)
gate_w = 12 * 1 + 4 * 0.1
meas_w_seq = 4 * (8 * 2 + 1 + 5)          # idle 2p on 8 other wires, depol p, flip 5p
meas_w_par = 1 * (5 * 2) + 4 * (1 + 5)    # one readout layer: idle only on 5 data wires
reset_w = 5 * 2 + 4 * 2
print(f"  SI1000 expected faults per round / p: gates {gate_w:.1f}, sequential readout {meas_w_seq}, "
      f"reset {reset_w}  -> total {gate_w + meas_w_seq + reset_w:.1f}")
print(f"    same with one parallel readout layer: readout {meas_w_par} -> total "
      f"{gate_w + meas_w_par + reset_w:.1f}")

# ------------------------------------------------------- B2 platform budgets
US = 1e-6
PLATFORMS = {
    "trapped-ion qudit (40Ca+)": dict(
        t1=150 * US,   # estimate: ~10 pulses per qutrit Clifford (Ringbauer 2022: 2e-3/Clifford vs 2e-4/pulse) x 15 us pi/2 (Pogorelov 2021)
        t2=600 * US,   # estimate: LS gate 3 x 35 us pulses + 6 pi-pulse permutations (Hrmo 2023) ~ 300 us, plus 2 Fourier on target
        tm=3800 * US,  # Ringbauer 2022: 2 x 500 us detection + 2500 us recooling; + ~300 us spectator hiding (estimate)
        tr=300 * US,   # estimate: optical pumping + spectator hiding
        tshot=10e-3,   # estimate: initial cooling and state preparation
        coh=100e-3,    # Ringbauer 2022 / Pogorelov 2021: ~100 ms coherence with shielding
        p_si=0.013 / 0.9,  # Hrmo 2023 qutrit LS gate 98.7(2)% -> p ~ (1-F)/0.9
        par_two=1,     # gates in one chain applied one at a time
    ),
    "transmon qutrit (reference)": dict(
        t1=128e-9,     # Morvan 2021: qutrit Hadamard 128 ns
        t2=1.0 * US,   # Goss 2022: qutrit CZ 783 ns; + Fourier on target
        tm=1.0 * US,   # QubiC 2.0: 500 ns qutrit readout pulse; + ~0.5 us latency/ring-down (estimate)
        tr=1.0 * US,   # estimate: conditional reset with ~0.2-0.6 us FPGA feedback (QICK, ZI, Qblox, OPX)
        tshot=100 * US,  # estimate: repetition delay
        coh=30 * US,   # Goss 2022 tables: 1-2 manifold T1 ~ 30-60 us, Ramsey T2 ~ 10-20 us; take 30 us
        p_si=0.027,    # Goss 2022: CZ^dag process fidelity 97.3% -> p ~ 1 - F_pro
        par_two=4,     # check CNOTs can be packed into 4 parallel layers (max degree 4)
    ),
    "neutral-atom (qubit hardware; qudit gates not demonstrated)": dict(
        t1=1.0 * US,   # estimate
        t2=1.0 * US,   # estimate (Rydberg CZ is sub-us; transport not included)
        tm=500 * US,   # Bluvstein 2024: 500 us mid-circuit imaging
        tr=1000 * US,  # estimate (re-cooling and re-initialisation; Bluvstein 2025 quotes 4 ms cycles)
        tshot=4e-3,    # Bluvstein 2025: 4 ms cycle time example
        coh=1.0,       # estimate (hyperfine/nuclear-spin qubits, order of 1 s)
        p_si=0.005,    # Evered 2023: CZ 99.5% (qubit gate)
        par_two=4,
    ),
}


def round_time(pp, readout="seq"):
    t_two = ROUND["two"] * pp["t2"] / pp["par_two"]
    t_one = ROUND["one"] * pp["t1"]
    t_meas = (ROUND["meas"] if readout == "seq" else 1) * pp["tm"]
    return t_two + t_one + t_meas + pp["tr"]


def clifford_time(pp):
    return two_per_cliff * pp["t2"] + one_per_cliff * pp["t1"]


def shot_time(pp, m, readout="seq"):
    return pp["tshot"] + 2 * m * clifford_time(pp) + (m + 1) * round_time(pp, readout)


print("=" * 76)
print("B2  durations (order of magnitude; see PLATFORMS for inputs)")
for name, pp in PLATFORMS.items():
    rs, rp = round_time(pp, "seq"), round_time(pp, "par")
    s22s, s22p = shot_time(pp, 22, "seq"), shot_time(pp, 22, "par")
    print(f"  {name}")
    print(f"    check round: {rs*1e3:9.3f} ms (sequential readout), {rp*1e3:9.3f} ms (one readout layer); "
          f"round / coherence = {rp/pp['coh']:.3g}")
    print(f"    logical Clifford: {clifford_time(pp)*1e3:.3f} ms; shot at m = 22: {s22s:.4g} s (seq), "
          f"{s22p:.4g} s (par); shot / coherence = {s22p/pp['coh']:.3g}")

# --------------------------------------------- B3 shot budgets and inflation
def read_rejection(p_idx, policy, k):
    path = os.path.join(BASELINE, str(p_idx), f"{policy}_check_data", f"{k}.csv")
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    rej = [float(x) for x in rows[3][1:]]
    fid = [float(x) for x in rows[1][1:]]
    return rej, fid


def nearest_p_index(p):
    return min(range(len(P_GRID)), key=lambda i: abs(math.log(max(P_GRID[i], 1e-12)) - math.log(p)))


print("=" * 76)
print("B3  dataset wall-clock (30 sequences x 9 depths x N shots, one readout layer per round)")
n_circ = 30
for name, pp in PLATFORMS.items():
    t_all = sum(shot_time(pp, m, "par") for m in DEPTHS) * n_circ
    print(f"  {name}: one shot of every circuit = {t_all:.4g} s")
    for n_shots in (1e3, 1e4, 1e6):
        t = t_all * n_shots
        print(f"    N = {n_shots:8.0e}: {t/3600:12.4g} h = {t/86400:10.4g} days")

print("  LRB-D shot inflation 1/(1 - rejection) at m = 22 from the baseline (SI1000)")
for p in (1e-3, 5e-3, 1.25e-2, 1.5e-2, 3e-2):
    i = nearest_p_index(p)
    for policy, k in (("unif", 1), ("const", 1), ("const", 0)):
        rej, fid = read_rejection(i, policy, k)
        r22 = rej[-1]
        infl = math.inf if r22 >= 1 else 1 / (1 - r22)
        print(f"    p = {P_GRID[i]:<7g} {policy:5s} {k}: rejection at m=22 {r22:.6f} -> inflation "
              f"{infl:.4g}; fidelity at m=22 {fid[-1]:.4f}")
for name, pp in PLATFORMS.items():
    i = nearest_p_index(pp["p_si"])
    rej, _ = read_rejection(i, "unif", 1)
    first_dead = next((DEPTHS[j] for j, r in enumerate(rej) if r > 0.999), None)
    print(f"  {name}: platform-matched p ~ {pp['p_si']:.3g} -> nearest baseline p = {P_GRID[i]}; "
          f"unif 1 rejection by depth {[round(r, 4) for r in rej]}; first depth with >99.9% rejection: {first_dead}")

# ------------------------------------------- B5 shots needed to see the decay
# Shot-noise standard error of the fidelity estimator F = |mean(w^k)| when the
# logical outcome is 0 with prob (1+2F)/3 and 1, 2 with prob (1-F)/3 each
# (delta method): sigma^2 = (1 - F)(1 + 2F) / (2 N_acc).
print("=" * 76)
print("B5  raw shots per circuit needed so that the m = 22 shot noise is 10% of the decay F(0) - F(22)")
print("    (sequence-to-sequence spread ignored; target is my choice, not a standard)")
for p in (5e-4, 1e-3, 2e-3):
    i = nearest_p_index(p)
    for policy, k in (("unif", 1), ("const", 1), ("const", 0)):
        rej, fid = read_rejection(i, policy, k)
        f0, f22, r22 = fid[0], fid[-1], rej[-1]
        decay = f0 - f22
        if not (decay > 0) or math.isnan(f22):
            print(f"    p = {P_GRID[i]:<7g} {policy:5s} {k}: no usable decay")
            continue
        sigma = 0.1 * decay
        n_acc = (1 - f22) * (1 + 2 * f22) / (2 * sigma ** 2)
        n_raw = n_acc / (1 - r22)
        t_ion = n_raw * shot_time(PLATFORMS["trapped-ion qudit (40Ca+)"], 22, "par")
        t_sc = n_raw * shot_time(PLATFORMS["transmon qutrit (reference)"], 22, "par")
        print(f"    p = {P_GRID[i]:<7g} {policy:5s} {k}: F(0) {f0:.5f}, F(22) {f22:.5f}, decay {decay:.2e}, "
              f"rejection {r22:.3f} -> N_acc {n_acc:.3g}, N_raw {n_raw:.3g} per m=22 circuit; "
              f"ion {t_ion/3600:.3g} h, transmon {t_sc/3600:.3g} h per circuit")

# ---------------------------------------------------- B4 photonic success
print("=" * 76)
print("B4  probabilistic two-qudit gates (success 1/d per gate, Chi et al. 2022 scheme)")
for dq in (3, 4):
    per_round = (1 / dq) ** ROUND["two"]
    n2_shot = 2 * 22 * two_per_cliff + 23 * ROUND["two"]
    print(f"  d = {dq}: one check round succeeds with prob {per_round:.3e}; a depth-22 shot "
          f"(~{n2_shot:.0f} two-qudit gates) with prob 10^{n2_shot*math.log10(1/dq):.1f}")
