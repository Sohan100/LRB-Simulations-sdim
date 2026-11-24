"""Figures for the correction-decoder report (reads the committed result CSVs).

All figures are static PNGs for the Markdown report. Every strategy has one
fixed colour (reference categorical palette, fixed slot order) and one fixed
marker in every figure, so identity never relies on colour alone and never
changes between figures; there is exactly one y-axis per panel and legends
sit outside the data region.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "docs" / "correction-decoders" / "qutrit-512" / "results"
RUN = Path("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
           "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# strategy -> (label, colour slot, marker)
STYLE = {
    "lrbd:unif1": ("LRB-D, every round (unif 1)", 0, "o"),
    "lrbd:const1": ("LRB-D, final round only (const 1)", 1, "s"),
    "bfd:all": ("Blip-forgiving detection (BFD)", 2, "^"),
    "soft": ("MLD + reject below a confidence threshold", 3, "D"),
    "mld:all": ("Full MLD correction (offline)", 4, "v"),
    "win:0": ("Hard-decision causal decoder, no lookahead", 5, "P"),
    "raw": ("No decoding, no rejection", 6, "X"),
    "lrbd:const0": ("LRB-D, no ancilla checks (const 0)", 7, "*"),
}
RB_STYLE = dict(color=INK_2, linestyle="--", linewidth=1.2, marker=".", label="Physical RB at p_RB = 5p")


def style_of(name: str):
    key = "soft" if name.startswith("soft:") else name
    label, slot, marker = STYLE[key]
    if name.startswith("soft:"):
        label = f"MLD + reject if confidence < {name.split(':')[1]}"
    return label, SLOTS[slot], marker


def style_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def read_csv(path: Path) -> list[dict]:
    """Read a CSV, falling back to its ``.gz`` sibling."""
    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return list(csv.DictReader(handle))


def f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def curves_by(rows, variant=None):
    out = defaultdict(list)
    for row in rows:
        if variant is not None and row.get("variant", "") != variant:
            continue
        out[(row["strategy"], round(f(row["p"]), 12))].append(row)
    for key in out:
        out[key].sort(key=lambda r: int(r["depth"]))
    return out


def rb_points(p_lrb: float):
    """Physical RB at p_RB = 5p from the production run: (depth, infidelity in survival terms)."""
    probs = [float(v) for v in (RUN / "probs.txt").read_text().strip().split(",")]
    index = int(np.argmin([abs(v - p_lrb) for v in probs]))
    path = RUN / "results" / "RB_comparison" / f"{index}.csv"
    if not path.exists():
        return None
    rows = list(csv.reader(path.read_text().strip().splitlines()))
    legacy = np.array([float(v) for v in rows[1][1:]])
    return np.array(DEPTHS), 2.0 * (1.0 - legacy) / 3.0


def legend_below(fig, axes, ncol=3):
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8)


def fig_decay(curves, p, path):
    strategies = ["lrbd:unif1", "lrbd:const1", "lrbd:const0", "bfd:all", "soft:0.9", "mld:all", "win:0", "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for name in strategies:
        rows = curves.get((name, round(p, 12)))
        if not rows:
            continue
        label, colour, marker = style_of(name)
        depth = np.array([int(r["depth"]) for r in rows])
        infid = 1 - np.array([f(r["survival_mean"]) for r in rows])
        acc = np.array([f(r["acc_mean"]) for r in rows])
        kw = dict(color=colour, marker=marker, markersize=5, linewidth=1.6, label=label)
        ok = infid > 0
        axes[0].plot(depth[ok], infid[ok], **kw)
        axes[1].plot(depth, acc, **kw)
    rb = rb_points(p)
    if rb is not None:
        ok = rb[1] > 0
        axes[0].plot(rb[0][ok], rb[1][ok], **RB_STYLE)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    style_axes(axes[0], "depth m (m forward + m inverse Cliffords)", "infidelity of accepted shots, 1 - P(correct)",
               f"Accepted-shot infidelity, p = {p:g}")
    style_axes(axes[1], "depth m", "acceptance (fraction of shots kept)", f"Acceptance, p = {p:g}")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    legend_below(fig, axes)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pareto(curves, frontier, p, depths, path):
    fig, axes = plt.subplots(1, len(depths), figsize=(5.4 * len(depths), 4.8), squeeze=False)
    for ax, depth in zip(axes[0], depths):
        pts = sorted([(f(r["acc_mean"]), f(r["infid_mean"])) for r in frontier
                      if np.isclose(f(r["p"]), p) and int(r["depth"]) == depth and f(r["infid_mean"]) > 0])
        if pts:
            label, colour, _marker = style_of("soft:0.9")
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=colour, linewidth=1.8,
                    label="MLD posterior threshold, all thresholds (Pareto frontier)")
        for name in ("lrbd:unif1", "lrbd:const1", "bfd:all", "mld:all", "win:0", "raw"):
            rows = curves.get((name, round(p, 12)))
            row = next((r for r in rows if int(r["depth"]) == depth), None) if rows else None
            if row:
                label, colour, marker = style_of(name)
                ax.plot(f(row["acc_mean"]), 1 - f(row["survival_mean"]), linestyle="none",
                        marker=marker, markersize=9, color=colour, label=label)
        ax.set_yscale("log")
        style_axes(ax, "acceptance", "infidelity of accepted shots", f"p = {p:g}, m = {depth}")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    legend_below(fig, axes[0])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_rate_vs_p(fits, path):
    strategies = ["lrbd:unif1", "lrbd:const1", "lrbd:const0", "bfd:all", "soft:0.9", "mld:all", "win:0", "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for name in strategies:
        rows = sorted([r for r in fits if r["strategy"] == name and r.get("variant", "") == ""], key=lambda r: f(r["p"]))
        rows = [r for r in rows if f(r["p"]) <= 0.0201]
        ps = np.array([f(r["p"]) for r in rows])
        rr = np.array([f(r.get("signed_r")) for r in rows])
        acc = np.array([f(r.get("acc_per_step")) for r in rows])
        label, colour, marker = style_of(name)
        kw = dict(color=colour, marker=marker, markersize=5, linewidth=1.6, label=label)
        ok = np.isfinite(rr) & (rr > 0)
        axes[0].plot(ps[ok], rr[ok], **kw)
        ok = np.isfinite(acc) & (acc < 1)
        if ok.any():
            axes[1].plot(ps[ok], 1 - acc[ok], **kw)
    rb = read_csv(RESULTS / "rb_reference.csv")
    rb = [r for r in rb if f(r["p_lrb"]) <= 0.0201]
    axes[0].plot([f(r["p_lrb"]) for r in rb], [f(r["rb_r"]) for r in rb], **RB_STYLE)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    style_axes(axes[0], "physical noise p (SI1000)", "logical error per depth step r = 2(1 - f)/3",
               "Fitted decay of the accepted shots (one depth step = 1 forward + 1 inverse Clifford)")
    style_axes(axes[1], "physical noise p (SI1000)", "rejection per depth step, 1 - a", "Rejection per depth step")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    legend_below(fig, axes)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_windows(fits, p, path, variant="", strategies=None, title=None):
    strategies = strategies or ["lrbd:unif1", "lrbd:const1", "lrbd:const0", "bfd:all", "soft:0.9", "mld:all"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name in strategies:
        row = next((r for r in fits if r["strategy"] == name and np.isclose(f(r["p"]), p)
                    and r.get("variant", "") == variant), None)
        if not row or not row.get("windows"):
            continue
        win = json.loads(row["windows"])
        mid = np.array([(w[0] + w[1]) / 2 for w in win])
        rate = 1 - np.array([w[2] for w in win])
        lo = 1 - np.array([w[4] for w in win])
        hi = 1 - np.array([w[3] for w in win])
        ok = rate > 0
        label, colour, marker = style_of(name)
        ax.errorbar(mid[ok], rate[ok], yerr=[np.clip(rate[ok] - lo[ok], 0, None), np.clip(hi[ok] - rate[ok], 0, None)],
                    color=colour, marker=marker, markersize=5, linewidth=1.4, capsize=2, label=label)
    ax.set_yscale("log")
    style_axes(ax, "window midpoint depth", "windowed decay 1 - f per depth step",
               title or f"Depth-windowed decay rate, p = {p:g}")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    legend_below(fig, ax, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_mixture(rows, path, p_low=5e-4, p_high=2e-3):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=False)
    strategies = ["lrbd:unif1", "bfd:all", "mld:all"]
    comp_style = {"low": (":", f"all shots at p = {p_low:g}"), "high": ("--", f"all shots at p = {p_high:g}"),
                  "mix": ("-", "each shot at p_low or p_high (50/50)")}
    for ax, name in zip(axes, strategies):
        label, colour, marker = style_of(name)
        for comp, (ls, comp_label) in comp_style.items():
            sel = [r for r in rows if r["strategy"] == name and r["component"] == comp
                   and np.isclose(f(r["p_low"]), p_low)]
            sel.sort(key=lambda r: int(r["depth"]))
            ax.plot([int(r["depth"]) for r in sel], [f(r["signed_mean"]) for r in sel], color=colour,
                    linestyle=ls, marker=marker, markersize=4, linewidth=1.6 if comp == "mix" else 1.2,
                    label=comp_label)
        style_axes(ax, "depth m", "accepted-shot polarization Re E[w^residual]", label)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    legend_below(fig, axes[:1], ncol=3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_dense(fits, p, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    variant_style = {"base": ("-", "baseline protocol"), "chk": ("--", "checks inside the inverse half"),
                     "par_term": (":", "parallel readout + full terminal readout")}
    for ax, name in zip(axes, ("lrbd:all", "bfd:all")):
        colour = SLOTS[0] if name == "lrbd:all" else SLOTS[2]
        marker = "o" if name == "lrbd:all" else "^"
        for variant, (ls, vlabel) in variant_style.items():
            row = next((r for r in fits if r["strategy"] == name and np.isclose(f(r["p"]), p)
                        and r.get("variant", "") == variant), None)
            if not row or not row.get("windows"):
                continue
            win = json.loads(row["windows"])
            mid = np.array([(w[0] + w[1]) / 2 for w in win])
            rate = 1 - np.array([w[2] for w in win])
            ok = rate > 0
            ax.plot(mid[ok], rate[ok], color=colour, linestyle=ls, marker=marker, markersize=3.5,
                    linewidth=1.4, label=vlabel)
        ax.set_yscale("log")
        title = "LRB-D, every round" if name == "lrbd:all" else "Blip-forgiving detection"
        style_axes(ax, "window midpoint depth (exact, every m)", "windowed decay 1 - f per depth step",
                   f"{title}, p = {p:g}")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    legend_below(fig, axes[:1], ncol=3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    results = Path(args.results)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if (results / "curves_baseline.csv.gz").exists() or (results / "curves_baseline.csv").exists():
        curves = curves_by(read_csv(results / "curves_baseline.csv"))
        fits = read_csv(results / "fits_baseline.csv")
        frontier = read_csv(results / "pareto_frontier.csv") if (results / "pareto_frontier.csv").exists() else []
        for p in (5e-4, 1e-3, 2e-3, 5e-3):
            fig_decay(curves, p, figures / f"decay_acceptance_p{p:g}.png")
            fig_pareto(curves, frontier, p, [6, 22], figures / f"pareto_p{p:g}.png")
            fig_windows(fits, p, figures / f"windows_baseline_p{p:g}.png")
        fig_rate_vs_p(fits, figures / "rate_vs_p.png")
    if (results / "mixture_curves.csv").exists():
        fig_mixture(read_csv(results / "mixture_curves.csv"), figures / "mixture_quasistatic.png")
    if (results / "fits_dense.csv.gz").exists() or (results / "fits_dense.csv").exists():
        dense = read_csv(results / "fits_dense.csv")
        for p in (1e-3, 2e-3, 5e-3):
            fig_dense(dense, p, figures / f"dense_windows_p{p:g}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
