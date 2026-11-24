"""Pseudo-thresholds by check layout: one ancilla, previous conventional sites, two ancillas.

The two-ancilla layout (square lattice, user's choice 2026-11-17; internal name layout B) replaces the
conventional-site two-ancilla layout in the paper. This script reads the
hand-off files written by scripts/merge_highp_threshold_extension.py (qutrit,
qubit, conv, qubit-conv, layout-b, qubit-layout-b families) and writes a table
and a figure of p* against the check interval k, qutrit and qubit, SI1000 and
SD6, to LRB-experiment-data-slurm/comparisons/sdim140-layout-b/, and the
adjacent-k gap significances of the layout-B configurations to
comparisons/sequence-bootstrap/interval_gap_significance-layout-b.csv. The
coherent-relay rows of the qutrit/qubit hand-off files are not used (the paper
no longer has that layout).

Usage (repo root):
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/compare_layout_b_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_conv_sites_thresholds as conv  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = PROJECT_ROOT / "LRB-experiment-data-slurm" / "comparisons"
OUT = COMPARISONS / "sdim140-layout-b"
BOOTSTRAP = COMPARISONS / "sequence-bootstrap"
SOURCES = {
    "qutrit": ("pseudo-thresholds-current.csv", "pseudo-thresholds-conv-sites-current.csv",
               "pseudo-thresholds-layout-b-current.csv"),
    "qubit": ("pseudo-thresholds-qubit-current.csv", "pseudo-thresholds-qubit-conv-sites-current.csv",
              "pseudo-thresholds-qubit-layout-b-current.csv"),
}
LAYOUTS = {
    "one-ancilla": "one ancilla",
    "conv-two-ancilla": "previous: two ancillas, conventional sites",
    "layout-b": "two ancillas",
}
STYLES = {
    "one ancilla": dict(color="#1f5fa8", marker="o", linestyle="-"),
    "previous: two ancillas, conventional sites": dict(color="#7a7a7a", marker="s", linestyle="--"),
    "two ancillas": dict(color="#b5452b", marker="D", linestyle="-"),
}


def load() -> pd.DataFrame:
    rows = []
    for code, files in SOURCES.items():
        for name in files:
            frame = pd.read_csv(COMPARISONS / name)
            for _, row in frame.iterrows():
                config = row["configuration"]
                if code == "qubit":
                    config = config.removeprefix("qubit-")
                layout, noise = config.rsplit("-", 1)
                if layout not in LAYOUTS:
                    continue
                rows.append({
                    "code": code,
                    "noise": noise,
                    "layout": LAYOUTS[layout],
                    "interval": int(row["interval"]),
                    "p_star": float(row["p_star"]),
                    "seq_boot_se": float(row["seq_boot_se"]),
                    "method": row["method"],
                    "status": row["status"],
                })
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "pseudo-thresholds-by-layout.csv", index=False)
    lines = [
        "# Pseudo-thresholds by check layout (the two-ancilla layout replaces the conventional sites)",
        "",
        "p* +- one standard error (sequence bootstrap, 68%); `pending` where the"
        " bootstrap has not been run for that exact p*. Sources: the"
        " `comparisons/pseudo-thresholds-*current.csv` hand-off files. 2/1 ="
        " two ancillas / one ancilla; 2/conv = two ancillas / previous"
        " conventional sites; gap = (one ancilla - two ancillas) / SE, the two"
        " SEs added in quadrature (independent runs). Two-ancilla rows marked"
        " provisional still wait for their extension or refinement.",
        "",
        "| code | noise | k | one ancilla | previous: conventional sites | two ancillas | 2/1 | 2/conv | gap | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (code, noise, k), group in table.groupby(["code", "noise", "interval"]):
        cell = {}
        for _, row in group.iterrows():
            se = (f" +- {row.seq_boot_se:.4f}" if np.isfinite(row.seq_boot_se)
                  else " (SE pending)")
            cell[row.layout] = (row.p_star, f"{row.p_star:.4f}{se}", row.seq_boot_se, row.status)
        one = cell["one ancilla"]
        cs = cell["previous: two ancillas, conventional sites"]
        b = cell.get("two ancillas")
        if b is None:
            continue
        gap = (one[0] - b[0]) / np.hypot(one[2], b[2])
        lines.append(
            f"| {code} | {noise.upper()} | {k} | {one[1]} | {cs[1]} | {b[1]} |"
            f" {b[0] / one[0]:.2f} | {b[0] / cs[0]:.2f} | {gap:.1f} sigma | {b[3]} |")
    (OUT / "pseudo-thresholds-by-layout.md").write_text("\n".join(lines) + "\n")


def plot(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True)
    for i, code in enumerate(("qutrit", "qubit")):
        for j, noise in enumerate(("si1000", "sd6")):
            ax = axes[i, j]
            for layout, style in STYLES.items():
                part = table[(table.code == code) & (table.noise == noise)
                             & (table.layout == layout)].sort_values("interval")
                err = np.where(np.isfinite(part.seq_boot_se), part.seq_boot_se, 0.0)
                ax.errorbar(part.interval, part.p_star, yerr=err, capsize=2,
                            markersize=4, linewidth=1.2, label=layout, **style)
            title = r"$[[5,1,2]]_3$" if code == "qutrit" else r"$[[5,1,2]]_2$"
            ax.set_title(f"{title}, {noise.upper()}")
            ax.set_xticks([1, 2, 3, 4])
            ax.grid(True, alpha=0.25)
            if j == 0:
                ax.set_ylabel("pseudo-threshold $p^*$")
            if i == 1:
                ax.set_xlabel("check interval $k$ (Cliffords)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pseudo-thresholds-by-layout.pdf")
    fig.savefig(OUT / "pseudo-thresholds-by-layout.png", dpi=160)


def write_gaps() -> pd.DataFrame:
    rows = []
    for name in (SOURCES["qutrit"][2], SOURCES["qubit"][2]):
        frame = pd.read_csv(COMPARISONS / name)
        for configuration, group in frame.groupby("configuration", sort=False):
            if not (BOOTSTRAP / configuration / "bootstrap_replicates.csv").exists():
                continue
            rows += conv.gap_significance(configuration, group.set_index("interval").p_star)
    gaps = pd.DataFrame(rows)
    if not gaps.empty:
        gaps.to_csv(BOOTSTRAP / "interval_gap_significance-layout-b.csv", index=False)
    return gaps


def main() -> int:
    table = load()
    write_table(table)
    plot(table)
    print((OUT / "pseudo-thresholds-by-layout.md").read_text())
    gaps = write_gaps()
    if not gaps.empty:
        print(gaps.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
