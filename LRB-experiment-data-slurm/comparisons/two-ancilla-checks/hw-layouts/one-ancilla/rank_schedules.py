"""Rank the fault-safe schedules of oa_schedule_search.py (schedule_search.pkl).

Prints: counts by depth; where the generator round, PROD5 and PROD5B stand; the
SI1000 / SD6 Pareto front of the per-round leading-order logical error L2 (logical
error per round ~ L2 p^2); the top schedules by each model; and, for the chosen ones,
their gate moments.
Usage: rank_schedules.py -> schedule_search.out
"""
import pickle
from collections import Counter
from pathlib import Path

import oa_pair_score as PSC
import rounds as R

HERE = Path(__file__).resolve().parent
NAME = {5: "a1", 6: "a2", 7: "a3", 8: "a4"}


def orders_of(ops):
    gs = [op for op in ops if op[0] == "g" and len(op) == 4]
    anc = {}
    data = {}
    for op in gs:
        c, t = op[2], op[3]
        a, q = (t, c) if t in (5, 6) else (c, t)
        anc.setdefault(a, []).append(q)
        data.setdefault(q, []).append(a)
    return {a: tuple(v) for a, v in anc.items()}, {q: tuple(v) for q, v in data.items()}


def key(anc, data):
    return (tuple(sorted(anc.items())), tuple(sorted(data.items())))


def moment_table(ops):
    lines = []
    for i, (collapse, mom) in enumerate([m for m in PSC.moments(ops) if not m[0]]):
        cells = []
        for op in mom:
            if len(op) == 3:
                cells.append(f"{op[1]} {NAME[op[2]]}")
            else:
                c, t = op[2], op[3]
                if t in (5, 6):
                    cells.append(f"{NAME[t]}<-q{c}")
                else:
                    cells.append(f"{NAME[c]}->q{t}")
        lines.append(f"    m{i + 1}: " + ", ".join(cells))
    return "\n".join(lines)


def fmt(row):
    si, sd, depth = row[0], row[1], row[2]
    return (f"depth {depth}  SI1000 L2 {si[2]:.3f} A1 {si[0]:.2f}   SD6 L2 {sd[2]:.3f} A1 {sd[0]:.2f}")


if __name__ == "__main__":
    rows = pickle.load(open(HERE / "schedule_search.pkl", "rb"))
    index = {key(r[3], r[4]): r for r in rows}
    print(f"{len(rows)} fault-safe schedules of the current one-ancilla round (same 12 gates)")
    print("by gate-moment depth:", dict(sorted(Counter(r[2] for r in rows).items())))
    ref = {}
    for nm, ops in (("generator", PSC.generator_ops()), ("PROD5", R.PROD5), ("PROD5B", R.PROD5B)):
        a, d = orders_of(ops)
        hit = index.get(key(a, d))
        pr = PSC.per_round(ops)
        ref[nm] = pr
        print(f"  {nm:9s} in search: {hit is not None}  depth "
              f"{sum(1 for c, _ in PSC.moments(ops) if not c)}  SI1000 L2 {pr['si1000'][2]:.3f} "
              f"A1 {pr['si1000'][0]:.2f}   SD6 L2 {pr['sd6'][2]:.3f} A1 {pr['sd6'][0]:.2f}")
    g = ref["generator"]
    for model, idx in (("si1000", 0), ("sd6", 1)):
        L = sorted(r[idx][2] for r in rows)
        print(f"{model}: L2 per round min {L[0]:.3f} (x{L[0] / g[model][2]:.3f} of generator), "
              f"median {L[len(L) // 2]:.3f}, max {L[-1]:.3f}")
    # Pareto front in (SI1000 L2, SD6 L2)
    srt = sorted(rows, key=lambda r: (r[0][2], r[1][2]))
    front, best_sd = [], float("inf")
    for r in srt:
        if r[1][2] < best_sd - 1e-12:
            front.append(r)
            best_sd = r[1][2]
    print(f"\nPareto front (SI1000 L2, SD6 L2), ratios to the generator round:")
    for r in front:
        print(f"  {fmt(r)}   x{r[0][2] / g['si1000'][2]:.3f} / x{r[1][2] / g['sd6'][2]:.3f}")
        print(f"    OPS {r[5]}")
    for model, idx in (("SI1000", 0), ("SD6", 1)):
        print(f"\nTop 15 by {model} L2 per round:")
        for r in sorted(rows, key=lambda r: (r[idx][2], r[1 - idx][2]))[:15]:
            print(f"  {fmt(r)}")
    print("\nGate moments of the Pareto schedules:")
    for r in front:
        print(f"  {fmt(r)}")
        print(moment_table(r[5]))
