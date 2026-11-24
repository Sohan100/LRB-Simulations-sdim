"""Figures for the qubit [[5,1,2]] correction-vs-detection report.

Reads ``results/summary/{fits,curves}.csv`` (from ``summarize_grid.py``) and
``results/first_order/first_order_summary.json`` (from ``first_order.py``) and
writes PNG files to ``figures/main``.

    python studies/correction-decoders/qubit/scripts/plot_main.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
DOCS = ROOT / "studies/correction-decoders/qubit"
OUT = DOCS / "figures/main"

# Reference categorical palette, light mode, fixed slot order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SERIES = [  # (method key, label) in slot order
    ("lrbd_unif_1", "LRB-D, every check used (unif=1)"),
    ("lrbd_const_1", "LRB-D, terminal check only (const=1)"),
    ("mld5", "Full correction (exact MLD)"),
    ("mld5_gap4", "MLD + reject |LLR| < 4"),
    ("const0_mld", "No checks, decoded readout"),
    ("physical_rb_5p_a0", "Physical RB at 5p"),
]


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def load():
    curves = defaultdict(dict)
    for row in csv.DictReader(open(DOCS / "results/summary/curves.csv")):
        key = (float(row["p"]), row["method"])
        curves[key][int(row["depth"])] = (float(row["F_mean"]),
                                          float(row["F_std"]),
                                          float(row["rejection"]))
    fits = {}
    for row in csv.DictReader(open(DOCS / "results/summary/fits.csv")):
        fits[(float(row["p"]), row["method"])] = row
    return curves, fits


def fig_curves(curves, ps=(0.001, 0.005)):
    fig, axes = plt.subplots(2, len(ps), figsize=(10, 7), sharex=True)
    for col, p in enumerate(ps):
        for slot, (method, label) in enumerate(SERIES):
            d = curves.get((p, method))
            if not d:
                continue
            depths = sorted(d)
            F = np.array([d[m][0] for m in depths])
            A = np.array([1.0 - d[m][2] for m in depths])
            kw = dict(color=SLOTS[slot], linewidth=2, marker="o",
                      markersize=4.5, label=label)
            axes[0, col].plot(depths, F, **kw)
            if method != "physical_rb_5p_a0":
                # Depths with no accepted shot are left out, not clipped.
                axes[1, col].plot(depths, np.where(A > 0, A, np.nan), **kw)
        axes[0, col].set_title(f"p = {p:g}", color=INK, fontsize=10)
        axes[0, col].set_ylabel("Logical fidelity |<X_L>| (accepted shots)")
        axes[1, col].set_ylabel("Acceptance")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Benchmark depth m")
        for ax in axes[:, col]:
            style(ax)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("Qubit [[5,1,2]] LRB: postselection vs correction "
                 "(30 sequences; 1e6 shots LRB-D, 3e5 decoded)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_fidelity_acceptance_vs_depth.png", dpi=160)
    plt.close(fig)


def fig_error_vs_p(fits):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for slot, (method, label) in enumerate(SERIES):
        pts = sorted((p, float(r["f"]), float(r["f_lo"]), float(r["f_hi"]))
                     for (p, m), r in fits.items() if m == method and p > 0)
        pts = [x for x in pts if np.isfinite(x[1]) and x[1] < 1.0]
        if not pts:
            continue
        p = np.array([x[0] for x in pts])
        r = 0.5 * (1.0 - np.array([x[1] for x in pts]))
        lo = 0.5 * (1.0 - np.array([x[3] for x in pts]))
        hi = 0.5 * (1.0 - np.array([x[2] for x in pts]))
        ax.errorbar(p, r, yerr=[np.clip(r - lo, 0, None),
                                np.clip(hi - r, 0, None)],
                    color=SLOTS[slot], linewidth=2, marker="o",
                    markersize=4.5, capsize=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SI1000 noise parameter p")
    ax.set_ylabel("Error per depth step, (1 - f)/2")
    ax.set_title("Fitted decay (repo fit, offset 0; 68% sequence bootstrap)",
                 color=INK, fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_error_per_step_vs_p.png", dpi=160)
    plt.close(fig)


def fig_local_rates(curves, p=0.002):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for slot, (method, label) in enumerate(SERIES):
        d = curves.get((p, method))
        if not d:
            continue
        depths = sorted(d)
        mids, rates = [], []
        for a, b in zip(depths[:-1], depths[1:]):
            Fa, Fb = d[a][0], d[b][0]
            if Fa > 0 and Fb > 0 and np.isfinite(Fa) and np.isfinite(Fb):
                mids.append(0.5 * (a + b))
                rates.append((Fb / Fa) ** (1.0 / (b - a)))
        ax.plot(mids, 1.0 - np.array(rates), color=SLOTS[slot], linewidth=2,
                marker="o", markersize=4.5, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Depth window midpoint")
    ax.set_ylabel("1 - local decay rate per step")
    ax.set_title(f"Local decay rates, p = {p:g} "
                 "(constant = stationary, single exponential)",
                 color=INK, fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_local_decay_rates.png", dpi=160)
    plt.close(fig)


def fig_first_order():
    summary = json.load(open(DOCS / "results/first_order/"
                                    "first_order_summary.json"))
    s = summary["22"]
    comps = [("meas_idle", "Data idle during ancilla readout (2p)"),
             ("meas_flip", "Ancilla measurement flip (5p)"),
             ("two_qubit", "Two-qubit gate depolarizing (p)"),
             ("reset_idle", "Data idle during reset (2p)"),
             ("reset_flip", "Reset flip (2p)"),
             ("meas_depol", "Measured-ancilla depolarizing (p)"),
             ("one_qubit", "One-qubit gate depolarizing (0.1p)")]
    labels = [c[1] for c in comps]
    rej = np.array([s["R1_" + c[0]][0] for c in comps])
    err = np.array([s["E1_" + c[0]][0] for c in comps])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    y = np.arange(len(comps))[::-1]
    axes[0].barh(y, rej, color=SLOTS[0], height=0.6)
    axes[1].barh(y, err, color=SLOTS[2], height=0.6)
    axes[0].set_yticks(y, labels, fontsize=8, color=INK)
    axes[0].set_xlabel("First-order rejection weight / p (LRB-D, all checks)")
    axes[1].set_xlabel("First-order MLD logical-error weight / p")
    for ax, values in zip(axes, (rej, err)):
        for yi, v in zip(y, values):
            ax.text(v, yi, f" {v:.0f}", va="center", fontsize=8, color=INK)
        style(ax)
    fig.suptitle("Where the single faults go (depth 22, mean of 30 sequences, "
                 "exact DEM expansion)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_first_order_by_component.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    curves, fits = load()
    fig_curves(curves)
    fig_error_vs_p(fits)
    fig_local_rates(curves)
    fig_first_order()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
