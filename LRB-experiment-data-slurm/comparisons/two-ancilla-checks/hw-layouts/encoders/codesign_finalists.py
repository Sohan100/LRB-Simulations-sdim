"""Build, fault-check, schedule and benchmark the best co-design candidates.

Reads the ranked candidate lines printed by codesign.py / codesign_extra.py,
keeps those with at most MAX_GATES two-qutrit gates per round (A1 has 25), and
for each one builds the round (codesign_rounds.design_round: per-check protocol
screening by single faults, then a full single-fault check of the assembled
round, no mid-round reset), schedules it, and benchmarks the zero-silent rounds
with the same 3-round memory harness used for A1 (no ancilla reset between
rounds' halves; resets only at round start).
Usage: codesign_finalists.py MAX_GATES out1 [out2 ...]
"""
import ast
import re
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codesign_rounds as CR  # noqa: E402

LINE = re.compile(r"^\s+(\d+) gates\s+tree (.+?)\s+data (\{.*?\})\s+extra (.+?)\s+safe (.+?)\s+"
                  r"anc1 (\(.*?\)) touches (\(.*?\)) reads (\(.*?\))\s+anc2 (\(.*?\)) touches (\(.*?\)) "
                  r"reads (\(.*?\))\s+per-check (\[.*?\])")


def parse(path):
    for line in Path(path).read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        total, tree, data, extra, safe, s1, t1, r1, s2, t2, r2, cs = m.groups()
        yield dict(total=int(total), tree=tree, data=ast.literal_eval(data), extra=extra,
                   safe=frozenset(frozenset((int(p[0]), int(p[1]))) for p in safe.split()),
                   s1=ast.literal_eval(s1), t1=tuple(ast.literal_eval(t1 if "," in t1 else t1[:-1] + ",)")),
                   r1=ast.literal_eval(r1), s2=ast.literal_eval(s2),
                   t2=tuple(ast.literal_eval(t2 if "," in t2 else t2[:-1] + ",)")),
                   r2=ast.literal_eval(r2), per=ast.literal_eval(cs))


if __name__ == "__main__":
    max_gates = int(sys.argv[1])
    cands = [c for p in sys.argv[2:] for c in parse(p) if c["total"] <= max_gates]
    print(f"{len(cands)} candidates with <= {max_gates} two-qutrit gates", flush=True)
    results = []
    with Pool(24) as pool:
        for c in cands:
            print(f"\n== tree {c['tree']} data {c['data']} anc {c['s1']}{c['t1']} reads {c['r1']}, "
                  f"anc {c['s2']}{c['t2']} reads {c['r2']} (min {c['total']} gates, per-check {c['per']})", flush=True)
            best = CR.design_round(c, pool)
            if best is None:
                print("  no zero-silent round", flush=True)
                continue
            ng, n2, nc, order, choice, ops = best
            cells = {}
            for model, p in (("si1000", 0.002), ("si1000", 0.005), ("sd6", 0.01)):
                cells[(model, p)] = CR.bench(ops, model, p, 200000)
            print(f"  BEST: {n2} two-qutrit gates, {ng} gate moments + {nc} collapse; order {order}", flush=True)
            for (model, p), cell in cells.items():
                print(f"    {model} p={p}: X acc {cell[0][0]:.4f} err {cell[0][1]:.2e} | "
                      f"Z acc {cell[1][0]:.4f} err {cell[1][1]:.2e}", flush=True)
            print(f"  ops {ops}", flush=True)
            results.append((ng, n2, c, cells))
    print("\nSUMMARY (gate moments, two-qutrit gates, tree, data, SI1000 0.002 / SD6 0.01 X-run acc, err):")
    for ng, n2, c, cells in sorted(results, key=lambda r: (r[0], r[1])):
        a, b = cells[("si1000", 0.002)][0], cells[("sd6", 0.01)][0]
        print(f"  {ng:2d} moments {n2:2d} gates  tree {c['tree']:12s} data {c['data']}  "
              f"SI1000 {a[0]:.3f}/{a[1]:.1e}  SD6 {b[0]:.3f}/{b[1]:.1e}")
