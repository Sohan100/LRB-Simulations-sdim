"""H1 (rewrite) - decay estimates, shot and time budgets from sc_dem_hw.log.

Parses the acceptance table and the sequence-spread table printed by
sc_dem_hw.py and computes:

 1. per-LRB-step decay estimates f = (y(m=10)/y(m=2))^(1/8) for the raw,
    unif-1 (LRB-D, all rounds) and const-1 fidelities, and the matching
    acceptance decay per step; r = 2(1 - f)/3.
 2. variance components: per shot, Re(omega^k) has variance
        v(F) = (1 + F)/2 - F^2
    (k = 0 w.p. (2F+1)/3, k = 1, 2 w.p. (1-F)/3 each).  With K sequences,
    N shots per sequence and acceptance a, the SE of the sequence-averaged
    fidelity is sqrt(s_seq^2/K + v/(K N a)).  N_eq = v/(a s_seq^2) is the
    shot number at which shot noise equals the sequence spread; shots
    beyond a few N_eq per sequence are wasted.
 3. wall-clock time of a stage-C dataset for several repetition times.

Run: module load python/3.11; python sc_budget.py   (instant)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

LOG = Path(__file__).resolve().parent / "sc_dem_hw.log"


def v_shot(F):
    return (1 + F) / 2 - F * F


def se_mean(F, s_seq, K, N, a=1.0):
    return math.sqrt(s_seq ** 2 / K + v_shot(F) / (K * N * a))


def parse():
    acc, spread = {}, {}
    pat_acc = re.compile(r"^\s+(\w+)\s+(SI1000 p=\S+|HW-\S+)\s+(\d+)\s+([\d.]+) / (\S+)\s+([\d.]+) / (\S+)\s+([\d.]+)")
    pat_spr = re.compile(r"^\s+(\w+)\s+(SI1000 p=\S+|HW-\S+)\s+m=\s*(\d+): raw mean ([\d.]+) sd ([\d.]+); unif1 acc mean ([\d.]+) sd ([\d.]+); unif1 fid sd ([\d.]+)")
    for line in LOG.read_text().splitlines():
        m = pat_acc.match(line)
        if m:
            s, mod, d = m.group(1), m.group(2), int(m.group(3))
            vals = [float(x) if x != "nan" else float("nan") for x in m.group(4, 5, 6, 7, 8)]
            acc[(s, mod, d)] = vals
            continue
        m = pat_spr.match(line)
        if m:
            s, mod, d = m.group(1), m.group(2), int(m.group(3))
            spread[(s, mod, d)] = [float(x) for x in m.group(4, 5, 6, 7, 8)]
    return acc, spread


def main():
    acc, spread = parse()
    print("1. per-step decay estimates from m=2 -> m=10 (f = ratio^(1/8), r = 2(1-f)/3)")
    print("   schedule model           raw f (r)          unif1 f (r) [acc/step]      const1 f (r)")
    for (s, mod) in sorted({(a, b) for a, b, _ in acc}):
        if (s, mod, 2) not in acc or (s, mod, 10) not in acc:
            continue
        a2, a10 = acc[(s, mod, 2)], acc[(s, mod, 10)]
        def fr(i):
            y2, y10 = a2[i], a10[i]
            if not (y2 > 0 and y10 > 0) or math.isnan(y2) or math.isnan(y10):
                return "      n/a       "
            f = (y10 / y2) ** (1 / 8)
            return f"{f:.4f} ({2 * (1 - f) / 3:.4f})"
        acc_step = (a10[0] / a2[0]) ** (1 / 8) if a2[0] > 0 and a10[0] > 0 else float("nan")
        print(f"   {s:8s} {mod:15s} {fr(4)}   {fr(1)} [{acc_step:.3f}]   {fr(3)}")

    print("\n2. variance components (relay11 / par9 / const0 at HW-target and HW-qubit)")
    print("   s_seq = sd over K=6 sequences of the exact raw fidelity; N_eq = v/(s^2)")
    for key in sorted(spread):
        s, mod, d = key
        if mod not in ("HW-target", "HW-qubit"):
            continue
        F, sd, a, asd, fsd = spread[key]
        print(f"   {s:8s} {mod:10s} m={d:2d}: raw F={F:.3f} s_seq={sd:.4f} v={v_shot(F):.3f}"
              f" N_eq(raw)={v_shot(F) / sd ** 2:7.0f}; unif1 acceptance a={a:.4f}")

    print("\n   SE of the K-averaged RAW fidelity, K*N = 3e5 shots per depth (relay11, HW-target):")
    for d in (2, 6, 10):
        F, sd, a, asd, fsd = spread[("relay11", "HW-target", d)]
        row = "; ".join(f"K={K:3d},N={N:5d}: {se_mean(F, sd, K, N):.4f}"
                        for K, N in ((30, 10000), (100, 3000), (300, 1000)))
        print(f"   m={d:2d}: {row}")

    print("\n   SE of the K-averaged unif-1 ACCEPTED fidelity (par9, HW-target), shot-noise part only:")
    for d in (2, 6, 10):
        F_acc = acc[("par9", "HW-target", d)][1]
        a = acc[("par9", "HW-target", d)][0]
        row = "; ".join(f"K={K:3d},N={N:5d}: {math.sqrt(v_shot(F_acc) / (K * N * a)):.4f}"
                        for K, N in ((30, 10000), (100, 3000)))
        print(f"   m={d:2d}: F={F_acc:.4f}, a={a:.4f}: {row}")

    print("\n3. wall-clock time (9 depths, K=100 sequences, N=3000 shots -> 2.7e6 shots):")
    for t_rep in (100e-6, 300e-6, 1e-3):
        print(f"   {t_rep * 1e6:6.0f} us per shot: {2.7e6 * t_rep / 3600:.2f} h")
    print("   circuit length at m=22 (22 LRB steps + inverse block, upper bound 2 x 22 steps):")
    for lab, t_step in (("par9 HW-now", 5.89), ("relay11 HW-now", 10.78),
                        ("par9 HW-target", 2.47), ("relay11 HW-target", 4.28)):
        print(f"   {lab:18s} step {t_step:5.2f} us -> m=22 circuit <= {2 * 22 * t_step:6.0f} us")
    print("\n   DEM learning: shots for 10% relative precision on a mechanism of"
          " probability q (Poisson counting, N ~ 100/q):")
    for q in (1e-2, 3e-3, 1e-3):
        print(f"   q = {q:g}: N ~ {100 / q:.0e}")


if __name__ == "__main__":
    main()
