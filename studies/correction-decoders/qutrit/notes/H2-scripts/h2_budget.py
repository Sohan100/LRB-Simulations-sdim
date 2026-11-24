"""H2 (other platforms): time budgets, shot budgets and decoder-latency ratios.

Run:  module load python/3.11 && python3 h2_budget.py
Runtime: < 5 s. Reads h2_platform_noise_study.csv (written by h2_platform_noise_study.py)
next to this script. Writes nothing.

Blocks
  T1  Duration of one check round, one logical Clifford and one shot per platform. Inputs are
      literature numbers where they exist and labelled estimates otherwise (see PLATFORM).
  T2  Wall clock for the baseline dataset shape (30 sequences x 9 depths x N shots).
  T3  Shots per circuit needed to estimate the per-Clifford error r to +-10% (one standard
      error) from a two-point design (depth 0 and the best depth M in {2, 4, 6, 10}), shot
      noise only, using the per-depth means of h2_platform_noise_study.csv, for MLD correction,
      LRB-D unif 1 (conditional decay) and LRB-D const 1, and the matching wall clock.
  T4  Decoder and feedback latencies from the literature relative to each platform's round.
  T5  Probabilistic photonic two-qudit gates: success probability of one round and one shot.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_SEQ = 30
DEPTHS_ALL = [0, 2, 4, 6, 10, 14, 18, 20, 22]
TWO_PER_CLIFF_VIRTUAL_SWAP = 1.463     # h2_algebra_checks.py A1: only CZ(2,3) left per Clifford
ONE_PER_CLIFF_VIRTUAL_PAULI = 12.657   # h2_algebra_checks.py A1: X, Z as frame updates
US, MS = 1e-6, 1e-3

# t_1q, t_2q: one single-qutrit Clifford and one check CNOT; par2: check CNOTs run in parallel;
# t_meas: one (parallel) qutrit readout layer; t_reset; t_prep: per-shot cooling/initialisation;
# t2: coherence time used only for the ratio column.
PLATFORM = {
    "ion (40Ca+ qutrits)": dict(
        t_1q=90 * US,       # estimate: ~6 local pulses x 15 us (Pogorelov 2021: single-qubit operations 15 us)
        t_2q=195 * US + 2 * 90 * US,  # estimate: LS gate = 3 x ~35 us LS pulses (Hrmo 2023) + 3 cyclic
                            # permutations (2 pulses each, ~15 us); + H and H^dag D^dag on the target
                            # (h2_algebra_checks A2); D^dag on the control assumed virtual
        par2=1,             # one entangling gate at a time in one chain (estimate)
        t_meas=4.5 * MS,    # Meth 2025: qudit readout ~ (d - 2) 3.5 ms + 1 ms
        t_reset=0.2 * MS,   # estimate (optical pumping); 0 if reset is skipped (A6)
        t_prep=10 * MS,     # estimate (Doppler + PGC + sideband cooling at the start of each shot)
        t2=100 * MS),       # Ringbauer 2022: ~100 ms with magnetic shielding
    "qubit pairs on QCCD ions": dict(
        t_1q=2 * MS, t_2q=10 * MS, par2=1,   # estimates: ~10 two-qubit gates per qutrit CNOT, with transport/cooling
        t_meas=5 * MS, t_reset=1 * MS, t_prep=20 * MS,  # estimates (Ryan-Anderson 2021: [[7,1,3]] QEC cycle < 200 ms)
        t2=1.0),            # estimate
    "transmon qutrits": dict(
        t_1q=0.1 * US,      # estimate
        t_2q=0.783 * US + 2 * 0.1 * US,      # Goss 2022: qutrit CZ 783 ns (CZ^dag 580 ns) + 2 local gates
        par2=3,             # estimate: 12 check CNOTs in 4 layers (data 1 has degree 4)
        t_meas=1.0 * US,    # QubiC 2.0: 500 ns qutrit readout pulse; + ~0.5 us ring-down (estimate)
        t_reset=1.0 * US,   # estimate
        t_prep=100 * US,    # estimate (repetition delay)
        t2=17 * US),        # Goss 2022 Supp. Table 1: Ramsey T2 of the 1-2 manifold 17(8) us
    "neutral atoms (qubit timings; no qudit gate exists)": dict(
        t_1q=1 * US, t_2q=1 * US, par2=4,    # estimates
        t_meas=1 * MS,      # Bluvstein 2024: readout + feedforward cycle slightly under 1 ms (500 us imaging)
        t_reset=0.5 * MS,   # estimate
        t_prep=100 * MS,    # estimate (loading and rearrangement)
        t2=1.0),            # Bluvstein 2024: T2 > 1 s (hyperfine qubit)
}


def round_time(pp):
    return 12 * pp["t_2q"] / pp["par2"] + 4 * pp["t_1q"] + pp["t_meas"] + pp["t_reset"]


def clifford_time(pp):
    return TWO_PER_CLIFF_VIRTUAL_SWAP * (pp["t_2q"] - 2 * pp["t_1q"]) + ONE_PER_CLIFF_VIRTUAL_PAULI * pp["t_1q"]


def shot_time(pp, m):
    return pp["t_prep"] + 2 * m * clifford_time(pp) + (m + 1) * round_time(pp)


print("=" * 100)
print("T1  durations (inputs: literature where cited in PLATFORM, otherwise labelled estimates)")
for name, pp in PLATFORM.items():
    rt, ct = round_time(pp), clifford_time(pp)
    print(f"  {name}")
    print(f"    check round {rt * 1e3:10.4f} ms (round / T2 = {rt / pp['t2']:.3g}); logical Clifford {ct * 1e3:9.4f} ms")
    print("    shot: " + ", ".join(f"m={m} {shot_time(pp, m) * 1e3:.4g} ms" for m in (0, 2, 6, 10, 22))
          + f"; coherent part of an m=22 shot / T2 = {(shot_time(pp, 22) - pp['t_prep']) / pp['t2']:.3g}")

print("=" * 100)
print(f"T2  wall clock for {N_SEQ} sequences x {len(DEPTHS_ALL)} depths x N shots per circuit")
for name, pp in PLATFORM.items():
    one = N_SEQ * sum(shot_time(pp, m) for m in DEPTHS_ALL)
    print(f"  {name:52s}: one shot of every circuit {one:9.3g} s; N=1e3 {one * 1e3 / 3600:9.3g} h; "
          f"N=1e4 {one * 1e4 / 86400:9.3g} d; N=1e6 {one * 1e6 / 86400:9.3g} d")

# ------------------------------------------------------------------ T3
rows = list(csv.DictReader(open(os.path.join(HERE, "h2_platform_noise_study.csv"))))
ORDER = ["si1000_1e-3", "si1000_2e-3", "si1000_5e-3", "ion_now", "ion_slowread", "ion_worstidle", "ion_next",
         "qubit_pairs", "transmon_opt"]
models = sorted({r["model"] for r in rows}, key=ORDER.index)
plat_of = {"si1000_1e-3": None, "si1000_2e-3": None, "si1000_5e-3": None,
           "ion_now": "ion (40Ca+ qutrits)", "ion_slowread": "ion (40Ca+ qutrits)",
           "ion_worstidle": "ion (40Ca+ qutrits)", "ion_next": "ion (40Ca+ qutrits)",
           "qubit_pairs": "qubit pairs on QCCD ions", "transmon_opt": "transmon qutrits"}
print("=" * 100)
print("T3  shots per circuit for +-10% (1 s.e.) on r = 2(1 - f)/3 from depths {0, M}, shot noise only,")
print("    30 sequences per depth; variance of the signed estimate (1 - lam)(1 + 2 lam)/(2 N_acc) (A8)")


def var_term(lam, acc):
    return (1 - lam) * (1 + 2 * lam) / (2.0 * N_SEQ * acc * lam * lam)


for model in models:
    sel = [r for r in rows if r["model"] == model]
    ms = sorted({int(r["m"]) for r in sel})
    mean = {m: {k: float(np.mean([float(r[k]) for r in sel if int(r["m"]) == m]))
                for k in ("acc_u1", "lam_u1", "acc_c1", "lam_c1", "lam_mld")} for m in ms}
    print(f"  model {model}")
    for label, lam_key, acc_key in (("MLD correction", "lam_mld", None), ("LRB-D unif 1", "lam_u1", "acc_u1"),
                                    ("LRB-D const 1", "lam_c1", "acc_c1")):
        best = None
        for M in ms[1:]:
            l0, lM = mean[0][lam_key], mean[M][lam_key]
            a0 = 1.0 if acc_key is None else mean[0][acc_key]
            aM = 1.0 if acc_key is None else mean[M][acc_key]
            if not (lM > 0.02 and lM < l0 and aM > 0):
                continue
            f = (lM / l0) ** (1.0 / M)
            r = 2 * (1 - f) / 3
            # sd(ln f) = sqrt(var0 + varM)/M; sd(r) = (2/3) f sd(ln f); require sd(r) = 0.1 r
            n_needed = (var_term(l0, a0) + var_term(lM, aM)) * ((2 * f / 3) / (0.1 * r * M)) ** 2
            if best is None or n_needed < best[0]:
                best = (n_needed, M, r, aM)
        if best is None:
            print(f"    {label:15s}: no usable two-point design")
            continue
        n_needed, M, r, aM = best
        line = (f"    {label:15s}: r = {r:.3g} (depths 0 and {M}); acceptance at M {aM:.3g}; "
                f"shots per circuit {n_needed:.3g}")
        pp = PLATFORM.get(plat_of[model]) if plat_of[model] else None
        if pp is not None:
            wall = N_SEQ * n_needed * (shot_time(pp, 0) + shot_time(pp, M))
            line += f"; wall clock for both depths {wall / 3600:.3g} h"
        # floor from the spread between Clifford sequences (exact or Rao-Blackwellized values per
        # sequence; for MLD the per-sequence value also carries a little sampling noise)
        sd0 = float(np.std([float(r[lam_key]) for r in sel if int(r["m"]) == 0], ddof=1)) if len(sel) else 0.0
        sdM = float(np.std([float(r[lam_key]) for r in sel if int(r["m"]) == M], ddof=1))
        f = (mean[M][lam_key] / mean[0][lam_key]) ** (1.0 / M)
        sd_lnf = math.sqrt((sd0 / mean[0][lam_key]) ** 2 + (sdM / mean[M][lam_key]) ** 2) / (math.sqrt(N_SEQ) * M)
        line += (f"; between-sequence sd of lam at M {sdM:.3g} -> relative floor on r with {N_SEQ} sequences "
                 f"{(2 * f / 3) * sd_lnf / r:.2g}")
        print(line)

# ------------------------------------------------------------------ T4
print("=" * 100)
print("T4  literature latencies relative to one check round of each platform")
LAT = [
    ("QICK branch on a measured value (42 ns) [St21]", 42e-9),
    ("QICK readout-to-next-pulse total (184-211 ns) [St21]", 211e-9),
    ("OPX-family readout-to-conditional pulse (428 ns) [Cao26]", 428e-9),
    ("ZI PQSC global feedback (< 550 ns) [ZI-PQSC]", 550e-9),
    ("Qblox LINQ multi-cast thresholded bits (472 ns, no I/O) [QB]", 472e-9),
    ("Riverlane FPGA decoder response, 9 rounds (9.6 us) [Ca24]", 9.6e-6),
    ("GPU link round trip (3.84 us mean) [NV25b]", 3.84e-6),
    ("Google workstation decoder latency (63 us) [A25]", 63e-6),
    ("GPU BP+OSD median decode, Helios (67 us) [NV25b]", 67e-6),
    ("Neutral-atom readout + FPGA feedforward cycle (< 1 ms) [Bl24]", 1e-3),
    ("AQT ion controller real-time feedback (~5 us) [P21]", 5e-6),
    ("Quantinuum conditional-information distribution (< 250 us) [RA21]", 250e-6),
]
names = ["ion (40Ca+ qutrits)", "transmon qutrits"]
print(f"    {'latency':70s} " + " ".join(f"{n[:18]:>18s}" for n in names))
for label, t in LAT:
    print(f"    {label:70s} " + " ".join(f"{t / round_time(PLATFORM[n]):18.3g}" for n in names))

# ------------------------------------------------------------------ T5
print("=" * 100)
print("T5  probabilistic two-qudit gates with success 1/d per gate (Chi et al. 2022 scheme at d = 3)")
n2_shot22 = 2 * 22 * 3.116 + 23 * 12
print(f"  one check round (12 gates): {(1 / 3) ** 12:.3e}; a depth-22 shot (~{n2_shot22:.0f} gates): "
      f"10^{-n2_shot22 * math.log10(3):.1f}")

# ------------------------------------------------------------------ T6
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
SHOTS = int(open(os.path.join(RUN, "shots.txt")).read().strip())
P_GRID = [float(x) for x in open(os.path.join(RUN, "probs.txt")).read().strip().split(",")]
print("=" * 100)
print(f"T6  baseline LRB-D at large p (read-only CSVs, {SHOTS} shots per circuit): accepted-shot fidelity versus")
print("    the Rayleigh floor sqrt(pi/(4 N_acc)) of the legacy metric |mean(w^k)| for uniformly random outcomes")
for pidx in (9, 10, 12, 14):
    for pol, k in (("const", 1), ("unif", 1)):
        with open(os.path.join(RUN, "results", "LRB", str(pidx), f"{pol}_check_data", f"{k}.csv")) as fh:
            rr = list(csv.reader(fh))
        fid = [float(x) for x in rr[1][1:]]
        rej = [float(x) for x in rr[3][1:]]
        parts = []
        for j, m in enumerate(DEPTHS_ALL):
            n_acc = SHOTS * (1 - rej[j])
            floor = math.sqrt(math.pi / (4 * n_acc)) if n_acc > 0 else float("nan")
            if m in (6, 10, 22):
                parts.append(f"m={m}: F {fid[j]:.4f}, rej {rej[j]:.5f}, floor {floor:.4f}")
        print(f"  p = {P_GRID[pidx]:<7g} {pol} {k}: " + "; ".join(parts) + f"  (1 - 1/81 = {1 - 1 / 81:.5f})")
