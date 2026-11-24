"""Markdown tables for REPORT.md Part 2, generated from the verdict CSVs.

results/part2_table_diagnostics.md   Table 2.2 (near-threshold / low-p maxima, k_max)
results/part2_table_fidelity.md      Table 2.3 (F = 1 - r at p = 1e-3, 2e-3, 5e-3)
"""
import importlib

import numpy as np
import pandas as pd

v31 = importlib.import_module("31_single_exp_verdict")

ORDER = [
    ("qutrit-one5-si1000", "qutrit, one-ancilla (five-moment round), SI1000"),
    ("qutrit-one5-sd6", "qutrit, one-ancilla (five-moment round), SD6"),
    ("qubit-one5-si1000", "qubit, one-ancilla (five-moment round), SI1000"),
    ("qubit-one5-sd6", "qubit, one-ancilla (five-moment round), SD6"),
    ("qutrit-two-si1000", "qutrit, two-ancilla, SI1000"),
    ("qutrit-two-sd6", "qutrit, two-ancilla, SD6"),
    ("qubit-two-si1000", "qubit, two-ancilla, SI1000"),
    ("qubit-two-sd6", "qubit, two-ancilla, SD6"),
    ("qutrit-one-si1000", "qutrit, one-ancilla (seven-moment round), SI1000"),
    ("qutrit-one-sd6", "qutrit, one-ancilla (seven-moment round), SD6"),
    ("qubit-one-si1000", "qubit, one-ancilla (seven-moment round), SI1000"),
    ("qubit-one-sd6", "qubit, one-ancilla (seven-moment round), SD6"),
]
TAU = 0.10


def fmt(x, bold_if=TAU):
    if not np.isfinite(x):
        return "n/a"
    s = f"{x:.2f}"
    return f"**{s}**" if x > bold_if else s


def main():
    v = pd.read_csv("results/single_exp_verdicts.csv")
    km = pd.read_csv("results/single_exp_kmax.csv").set_index("config")
    lines = ["| configuration | k = 1 | k = 2 | k = 3 | k = 4 | terminal check | k_max (all p ≤ p*) | k_max (p ≥ p*/2) |",
             "|---|---|---|---|---|---|---|---|"]
    for cfg, name in ORDER:
        g = v[v.config == cfg].set_index("policy")
        cells = []
        for k in (1, 2):
            r = g.loc[f"unif{k}"]
            cells.append(f"{fmt(r.top_max_window)} ; low {fmt(r.low_max_window)}")
        for k in (3, 4):
            r = g.loc[f"unif{k}"]
            cells.append(f"{fmt(r.top_max_window)} / {fmt(r.top_max_grid)}")
        cells.append(fmt(g.loc["const0"].top_max_window))
        a = int(km.loc[cfg, "k_max_all_p_main"])
        b = int(km.loc[cfg, "k_max_upper_half_main"])
        a12 = int(km.loc[cfg, "k_max_all_p_t12"])
        b12 = int(km.loc[cfg, "k_max_upper_half_t12"])
        sa = f"{a}" + (f" ({a12} at 12%)" if a12 != a else "")
        sb = f"{b}" + (f" ({b12} at 12%)" if b12 != b else "")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {sa} | {sb} |")
    open("results/part2_table_diagnostics.md", "w").write("\n".join(lines) + "\n")

    d = pd.concat([pd.read_csv(v31.ROWS), pd.read_csv(v31.ROWS5)])
    d = v31.merged_rows(d)
    ps = v31.pstars()
    out = ["| configuration | k | p*(k) | F at p = 1e-3 | F at p = 2e-3 | F at p = 5e-3 |",
           "|---|---|---|---|---|---|"]
    for cfg, name in ORDER:
        for k in (1, 2):
            pstar = ps[(cfg, k)][0]
            cells = []
            for p in (0.001, 0.002, 0.005):
                r = d[(d.config == cfg) & (d.policy == f"unif{k}") & np.isclose(d.p, p)]
                if p > pstar:
                    cells.append("not quoted, p > p*")
                    continue
                if not len(r):
                    cells.append("n/a")
                    continue
                ok = bool(v31.pass_mask(r, TAU, None).iloc[0])
                x = r.iloc[0]
                cells.append(f"{1 - x.r:.6f} ± {x.r_se:.6f}" + ("" if ok else " †"))
            out.append(f"| {name} | {k} | {pstar:.4f} | " + " | ".join(cells) + " |")
    open("results/part2_table_fidelity.md", "w").write("\n".join(out) + "\n")
    print("\n".join(lines))
    print()
    print("\n".join(out))


if __name__ == "__main__":
    main()
