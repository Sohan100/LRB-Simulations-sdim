"""Compare qubit and qutrit [[5,1,2]] folded-code LRB-D results.

Reads the two hand-off files written by scripts/merge_highp_threshold_extension.py,
comparisons/pseudo-thresholds-current.csv (qutrit) and
comparisons/pseudo-thresholds-qubit-current.csv (qubit), plus each
configuration's merged LRB-vs-RB table, and writes to
comparisons/qubit-vs-qutrit/:

- thresholds-vs-interval-qubit-vs-qutrit.{pdf,png}: p* against the uniform
  check interval k, one panel per configuration, error bars = error of the
  mean (sequence-bootstrap SE).
- lrb-vs-rb-unif-1-qubit-vs-qutrit.{pdf,png}: LRB error per Clifford with a
  check after every Clifford, and physical RB, against p.
- threshold-comparison.{csv,md}: p* for both codes, their ratio and the
  difference in units of the combined SE (the two campaigns use independent
  Clifford sequences, so their errors add in quadrature).

With --two-ancilla conv-sites the two-ancilla panels use the conventional-site
layout (comparisons/pseudo-thresholds-{,qubit-}conv-sites-current.csv) instead
of the coherent relay, and everything goes to
comparisons/qubit-vs-qutrit-conv-sites/; the default output is unchanged.

Usage (repo root):
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/compare_qubit_qutrit_thresholds.py
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/compare_qubit_qutrit_thresholds.py \
        --two-ancilla conv-sites
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_highp_threshold_extension as merge  # noqa: E402

# two-ancilla layout -> (output folder, panels, extra hand-off files per code)
LAYOUTS = {
    "relay": (
        merge.COMPARISONS / "qubit-vs-qutrit",
        [("one-ancilla-si1000", "One ancilla, SI1000"),
         ("two-ancilla-si1000", "Two ancillas, SI1000"),
         ("one-ancilla-sd6", "One ancilla, SD6"),
         ("two-ancilla-sd6", "Two ancillas, SD6")],
        {"qutrit": (), "qubit": ()},
    ),
    "conv-sites": (
        merge.COMPARISONS / "qubit-vs-qutrit-conv-sites",
        [("one-ancilla-si1000", "One ancilla, SI1000"),
         ("conv-two-ancilla-si1000", "Two ancillas (conventional sites), SI1000"),
         ("one-ancilla-sd6", "One ancilla, SD6"),
         ("conv-two-ancilla-sd6", "Two ancillas (conventional sites), SD6")],
        {"qutrit": (merge.CONV_CURRENT_SUMMARY,),
         "qubit": (merge.QUBIT_CONV_CURRENT_SUMMARY,)},
    ),
}
OUT, PANELS, EXTRA_SUMMARIES = LAYOUTS["relay"]
CODES = {
    "qutrit": dict(prefix="", color="#2a78d6", marker="o",
                   label=r"qutrit $[[5,1,2]]_3$"),
    "qubit": dict(prefix="qubit-", color="#eb6834", marker="s",
                  label=r"qubit $[[5,1,2]]_2$"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_thresholds() -> pd.DataFrame:
    frames = []
    for code, path in (("qutrit", merge.CURRENT_SUMMARY),
                       ("qubit", merge.QUBIT_CURRENT_SUMMARY)):
        for source in (path, *EXTRA_SUMMARIES[code]):
            frame = pd.read_csv(source.with_suffix(".csv"))
            frame["code"] = code
            frame["panel"] = frame["configuration"].str.removeprefix("qubit-")
            frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    return table[table["panel"].isin([panel for panel, _ in PANELS])]


def merged_table(label: str) -> pd.DataFrame:
    for campaign in merge.CAMPAIGNS.values():
        if label in campaign["pairs"]:
            return pd.read_csv(
                campaign["out"] / label / "unif_lrb_vs_rb_table_merged.csv")
    raise KeyError(label)


def plot_thresholds(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), sharey=True)
    for ax, (panel, title) in zip(axes, PANELS):
        for offset, (code, style) in zip((-0.06, 0.06), CODES.items()):
            rows = table[(table["panel"] == panel) & (table["code"] == code)]
            rows = rows.sort_values("interval")
            k = rows["interval"].to_numpy()
            p = rows["p_star"].to_numpy()
            ax.errorbar(k + offset, p, yerr=rows["seq_boot_se"].to_numpy(),
                        fmt="-", marker=style["marker"], color=style["color"],
                        ms=6, lw=2, elinewidth=1.6, capsize=3,
                        label=style["label"])
            last = rows.iloc[-1]
            ax.annotate(f"{last['p_star']:.4f}", (last["interval"] + offset,
                                                   last["p_star"]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color=INK, va="center")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.6, 4.8)
        ax.set_xlabel("Uniform check interval k")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Pseudo-threshold p*")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    stem = OUT / "thresholds-vs-interval-qubit-vs-qutrit"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(stem.with_suffix(".png"))


def plot_error_rates() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), sharey=True)
    for ax, (panel, title) in zip(axes, PANELS):
        for code, style in CODES.items():
            frame = merged_table(style["prefix"] + panel)
            frame = frame[(frame["check_num"] == 1) & (frame["p"] > 0)]
            frame = frame.sort_values("p")
            lrb = frame[np.isfinite(frame["lrb_error_rate"])
                        & (frame["lrb_error_rate"] > 0)]
            yerr = np.vstack([
                lrb["lrb_error_rate"] - lrb["lrb_error_rate_ci_low"],
                lrb["lrb_error_rate_ci_high"] - lrb["lrb_error_rate"],
            ]).clip(min=0)
            ax.errorbar(lrb["p"], lrb["lrb_error_rate"], yerr=yerr,
                        color=style["color"], marker=style["marker"], ms=4,
                        lw=1.6, capsize=2, label=f"LRB-D k=1, {style['label']}")
            rb = frame[np.isfinite(frame["rb_error_rate"])
                       & (frame["rb_error_rate"] > 0)]
            ax.plot(rb["p"], rb["rb_error_rate"], color=style["color"],
                    linestyle=":", marker="^", ms=4, lw=1.4,
                    label=f"physical RB, {style['label'].split()[0]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Two-qudit gate error rate p")
        ax.grid(True, which="both", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Error rate per Clifford")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    stem = OUT / "lrb-vs-rb-unif-1-qubit-vs-qutrit"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(stem.with_suffix(".png"))


def write_table(table: pd.DataFrame) -> None:
    wide = table.pivot_table(index=["panel", "interval"], columns="code",
                             values=["p_star", "seq_boot_se"])
    wide.columns = [f"{value}_{code}" for value, code in wide.columns]
    wide = wide.reset_index()
    wide["ratio_qubit_over_qutrit"] = wide["p_star_qubit"] / wide["p_star_qutrit"]
    combined = np.hypot(wide["seq_boot_se_qubit"], wide["seq_boot_se_qutrit"])
    wide["difference_sigma"] = (
        (wide["p_star_qubit"] - wide["p_star_qutrit"]) / combined)
    order = {panel: i for i, (panel, _) in enumerate(PANELS)}
    wide = wide.sort_values(["panel", "interval"],
                            key=lambda s: s.map(order) if s.name == "panel" else s)
    wide.to_csv(OUT / "threshold-comparison.csv", index=False)
    lines = [
        "# Qubit vs qutrit [[5,1,2]] LRB-D pseudo-thresholds",
        "",
        "Same circuits, check layouts, SI1000/SD6 noise, grids, fits and"
        " threshold estimator; only the local dimension differs. p* ± error"
        " of the mean (sequence-bootstrap SE, 68%). The difference is in"
        " units of the two SEs added in quadrature.",
        "",
        "| configuration | k | qutrit p* | qubit p* | qubit / qutrit |"
        " difference |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in wide.iterrows():
        lines.append(
            f"| {row['panel']} | {int(row['interval'])} |"
            f" {row['p_star_qutrit']:.4f} ± {row['seq_boot_se_qutrit']:.4f} |"
            f" {row['p_star_qubit']:.4f} ± {row['seq_boot_se_qubit']:.4f} |"
            f" {row['ratio_qubit_over_qutrit']:.2f} |"
            f" {row['difference_sigma']:+.1f}σ |")
    (OUT / "threshold-comparison.md").write_text("\n".join(lines) + "\n")
    print(OUT / "threshold-comparison.md")


def main(argv: list[str]) -> int:
    global OUT, PANELS, EXTRA_SUMMARIES
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--two-ancilla", choices=sorted(LAYOUTS), default="relay",
                        help="two-ancilla layout in panels 2 and 4 (default relay)")
    args = parser.parse_args(argv)
    OUT, PANELS, EXTRA_SUMMARIES = LAYOUTS[args.two_ancilla]
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_thresholds()
    plot_thresholds(table)
    plot_error_rates()
    write_table(table)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
