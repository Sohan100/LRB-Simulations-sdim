"""Plot p* versus check interval with sequence-bootstrap error bars.

Reads LRB-experiment-data-slurm/comparisons/sequence-bootstrap/<label>/
sequence_bootstrap_summary.csv (from scripts/sequence_bootstrap_thresholds.py)
and draws, per configuration, the pseudo-threshold against the uniform check
interval. Bold bars are the 68% interval of the bootstrap over Clifford
sequences (the error of the mean); thin grey bars are the notebook's
per-point 68% intervals, for comparison. Writes one PDF per configuration
and a four-panel overview next to the summaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = (Path(__file__).resolve().parents[1] / "LRB-experiment-data-slurm"
        / "comparisons" / "sequence-bootstrap")
PANELS = [
    ("one-ancilla-si1000", "One ancilla, SI1000"),
    ("two-ancilla-si1000", "Two ancillas, SI1000"),
    ("one-ancilla-sd6", "One ancilla, SD6"),
    ("two-ancilla-sd6", "Two ancillas, SD6"),
]
BLUE, GREY = "#2a78d6", "#a3a29b"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def draw(ax, frame: pd.DataFrame, title: str) -> None:
    k = frame["interval"].to_numpy()
    p = frame["p_star"].to_numpy()
    ax.errorbar(k + 0.08, p,
                yerr=[p - frame["previous_ci68_low"], frame["previous_ci68_high"] - p],
                fmt="none", ecolor=GREY, elinewidth=1.2, capsize=2,
                label="per-point 68% interval (previous)")
    ax.errorbar(k, p,
                yerr=[(p - frame["seq_boot_ci68_low"]).clip(lower=0),
                      (frame["seq_boot_ci68_high"] - p).clip(lower=0)],
                fmt="o-", color=BLUE, ms=6, lw=2, elinewidth=2, capsize=4,
                label="error of the mean, 68% (sequence bootstrap)")
    for kk, pp, se in zip(k, p, frame["seq_boot_se"]):
        ax.annotate(f"{pp:.4f}\n±{se:.4f}", (kk, pp), xytext=(10, 2),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.6, 4.7)
    ax.set_xlabel("Uniform check interval k")
    ax.set_ylabel("Pseudo-threshold p*")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, (label, title) in zip(axes, PANELS):
        frame = pd.read_csv(ROOT / label / "sequence_bootstrap_summary.csv")
        draw(ax, frame, title)
        single, single_ax = plt.subplots(figsize=(5.2, 3.8))
        draw(single_ax, frame, title)
        single_ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        single.tight_layout()
        single.savefig(ROOT / label / "pseudo-threshold-vs-interval-sequence-bootstrap.pdf")
        plt.close(single)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ROOT / "pseudo-threshold-vs-interval-sequence-bootstrap-all.pdf")
    fig.savefig(ROOT / "pseudo-threshold-vs-interval-sequence-bootstrap-all.png", dpi=160)
    print(ROOT / "pseudo-threshold-vs-interval-sequence-bootstrap-all.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
