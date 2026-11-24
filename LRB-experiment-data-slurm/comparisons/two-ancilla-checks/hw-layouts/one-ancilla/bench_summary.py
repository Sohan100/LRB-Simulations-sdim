"""Pool the X- and Z-run cells of the bench_*.out files into one table row per round.

Cell = acceptance (mean of the X and Z runs) / logical error among accepted shots
(X and Z runs pooled) (event count). Usage: bench_summary.py FILE...
"""
import re
import sys
from collections import OrderedDict

LINE = re.compile(r"(\w+) p=([\d.]+) R=(\d) shots=(\d+) (\S+)\s+(\d+g\+\dc)\s+X: acc ([\d.]+) err \S+ \((\d+)\)\s+Z: acc ([\d.]+) err \S+ \((\d+)\)")
cells, cols, rows = {}, OrderedDict(), OrderedDict()
for path in sys.argv[1:]:
    for ln in open(path):
        m = LINE.search(ln)
        if not m:
            continue
        model, p, R, shots, name, mom, ax, ex, az, ez = m.groups()
        shots = int(shots)
        nacc = (float(ax) + float(az)) * shots
        nerr = int(ex) + int(ez)
        col = f"{model.upper()} {p} (R={R})"
        cols[col] = 1
        rows[name] = mom
        cells[(name, col)] = f"{(float(ax) + float(az)) / 2:.3f} / {nerr / nacc:.1e} ({nerr})"
print("| round | moments | " + " | ".join(cols) + " |")
print("|---|---|" + "---|" * len(cols))
for name, mom in rows.items():
    print(f"| {name} | {mom} | " + " | ".join(cells.get((name, c), "-") for c in cols) + " |")
