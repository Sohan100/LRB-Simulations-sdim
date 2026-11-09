"""Pseudo-thresholds by check layout: one ancilla, coherent relay, conventional sites.

Reads the hand-off files written by scripts/merge_highp_threshold_extension.py
(qutrit, qubit, conv, qubit-conv families) and writes a table and a figure of
p* against the check interval k for the three layouts, qutrit and qubit,
SI1000 and SD6, to LRB-experiment-data-slurm/comparisons/sdim140-conv-sites/,
and the adjacent-k gap significances of the conventional-site configurations to
comparisons/sequence-bootstrap/interval_gap_significance-conv-sites.csv.

Usage (repo root):
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/compare_conv_sites_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = PROJECT_ROOT / "LRB-experiment-data-slurm" / "comparisons"
OUT = COMPARISONS / "sdim140-conv-sites"
BOOTSTRAP = COMPARISONS / "sequence-bootstrap"
SOURCES = {
    "qutrit": ("pseudo-thresholds-current.csv", "pseudo-thresholds-conv-sites-current.csv"),
    "qubit": ("pseudo-thresholds-qubit-current.csv",
              "pseudo-thresholds-qubit-conv-sites-current.csv"),
}
LAYOUTS = {
    "one-ancilla": "one ancilla",
    "two-ancilla": "two ancillas, coherent relay",
    "conv-two-ancilla": "two ancillas, conventional sites",
}
STYLES = {
    "one ancilla": dict(color="#1f5fa8", marker="o", linestyle="-"),
    "two ancillas, coherent relay": dict(color="#7a7a7a", marker="s", linestyle="--"),
    "two ancillas, conventional sites": dict(color="#b5452b", marker="D", linestyle="-"),
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
        "# Pseudo-thresholds by check layout",
        "",
        "p* +- one standard error (sequence bootstrap, 68%); `pending` where the"
        " bootstrap has not been run for that exact p*. Sources: the four"
        " `comparisons/pseudo-thresholds-*current.csv` hand-off files. Ratio ="
        " conventional sites / coherent relay; gap = (coherent relay -"
        " conventional sites) / SE, the two SEs added in quadrature (independent"
        " runs).",
        "",
        "| code | noise | k | one ancilla | coherent relay | conventional sites | ratio | gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (code, noise, k), group in table.groupby(["code", "noise", "interval"]):
        cell = {}
        for _, row in group.iterrows():
            se = (f" +- {row.seq_boot_se:.4f}" if np.isfinite(row.seq_boot_se)
                  else " (SE pending)")
            cell[row.layout] = (row.p_star, f"{row.p_star:.4f}{se}", row.seq_boot_se)
        conv = cell["two ancillas, conventional sites"]
        relay = cell["two ancillas, coherent relay"]
        gap = (relay[0] - conv[0]) / np.hypot(relay[2], conv[2])
        lines.append(
            f"| {code} | {noise.upper()} | {k} | {cell['one ancilla'][1]} |"
            f" {relay[1]} | {conv[1]} | {conv[0] / relay[0]:.2f} | {gap:.1f} sigma |")
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


def gap_significance(configuration: str, published: pd.Series) -> list[dict]:
    """p*(k) - p*(k+1) against the SE of the paired replicate differences.

    Same rule as comparisons/sequence-bootstrap/interval_gap_significance.csv:
    both p* come from the same 1000 sequence-bootstrap replicates, so the SE is
    the spread of the per-replicate difference.
    """
    replicates = pd.read_csv(BOOTSTRAP / configuration / "bootstrap_replicates.csv")
    wide = replicates.pivot(index="rep", columns="check_num", values="p_star")
    rows = []
    for k in range(1, 4):
        diff = wide[k] - wide[k + 1]
        se = diff.std(ddof=1)
        gap = published[k] - published[k + 1]
        rows.append({
            "configuration": configuration,
            "comparison": f"p*({k}) - p*({k + 1})",
            "difference": gap,
            "se": se,
            "significance_sigma": gap / se,
            "replicates_positive_fraction": float((diff > 0).mean()),
            "replicates": int(diff.notna().sum()),
        })
    return rows


def write_gaps() -> pd.DataFrame:
    rows = []
    for name in (SOURCES["qutrit"][1], SOURCES["qubit"][1]):
        frame = pd.read_csv(COMPARISONS / name)
        for configuration, group in frame.groupby("configuration", sort=False):
            rows += gap_significance(configuration, group.set_index("interval").p_star)
    gaps = pd.DataFrame(rows)
    gaps.to_csv(BOOTSTRAP / "interval_gap_significance-conv-sites.csv", index=False)
    return gaps


def main() -> int:
    table = load()
    write_table(table)
    plot(table)
    print((OUT / "pseudo-thresholds-by-layout.md").read_text())
    print(write_gaps().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
