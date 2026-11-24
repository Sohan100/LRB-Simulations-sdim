"""Figures for the memory study (written to figures/ next to this file).

memory-thresholds.pdf   logical error per round vs p, one panel per code and noise model
memory-footprint.pdf    logical error per d-round block vs physical qubits at p = 0.1%, with fits
memory-lambda.pdf       error suppression factor Lambda vs p
memory-gates.pdf        cost of a fold-transversal logical gate between every two rounds
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_memory as am  # noqa: E402

OUT = os.path.join(HERE, "figures")
# validated palette (dataviz reference, light surface): categorical slots 1-3, ordinal blue ramp
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CODE_COLOR = {"folded": "#2a78d6", "rotated": "#eb6834"}
CODE_MARK = {"folded": "o", "rotated": "s"}
CODE_LABEL = {"folded": "folded (unrotated)", "rotated": "rotated"}
D_RAMP = {3: "#86b6ef", 5: "#5598e7", 7: "#2a78d6", 9: "#1c5cab", 11: "#104281"}
GATE_COLOR = {"H": "#2a78d6", "S": "#eb6834"}
GATE_MARK = {"H": "o", "S": "s"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
})


def _percent_axis(ax, ticks=(0.0005, 0.001, 0.002, 0.005, 0.01)):
    """log p axis labelled in percent, no minor labels"""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: ("%g%%" % (100 * x))))
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_thresholds(rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
    for i, noise in enumerate(("SD6", "SI1000")):
        for j, code in enumerate(("folded", "rotated")):
            ax = axes[i, j]
            for d in (3, 5, 7, 9, 11):
                pts = sorted((k[3], r) for k, r in rows.items() if k[0] == noise and k[1] == code and k[2] == d
                             and min(r["errors_x"], r["errors_z"]) >= 5)
                if not pts:
                    continue
                ps = [p for p, _ in pts]
                es = [r["eps"] for _, r in pts]
                lo = [r["eps"] - r["eps_lo"] for _, r in pts]
                hi = [r["eps_hi"] - r["eps"] for _, r in pts]
                ax.errorbar(ps, es, yerr=[lo, hi], color=D_RAMP[d], marker="o", ms=3.5, lw=1.4, capsize=0,
                            label="d = %d" % d)
            th = am.threshold(rows, noise, code, 9, 11)
            if th:
                ax.axvline(th, color=INK2, lw=0.8, ls=":")
                ax.text(th, 1.5e-9, "%.2f%% " % (100 * th), color=INK2, fontsize=8, va="bottom", ha="right")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _percent_axis(ax)
            ax.set_ylim(1e-9, 0.3)
            ax.set_title("%s, %s" % (CODE_LABEL[code], noise), fontsize=9, color=INK)
            if i == 1:
                ax.set_xlabel("physical error rate p")
            if j == 0:
                ax.set_ylabel("logical error per round")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "memory-thresholds.pdf"))
    fig.savefig(os.path.join(OUT, "memory-thresholds.png"), dpi=160)
    plt.close(fig)


def fig_footprint(rows, p=0.001):
    """Error per d-round block (about d times the per-round error) against physical qubits."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, noise in zip(axes, ("SD6", "SI1000")):
        for code in ("folded", "rotated"):
            pts = sorted((r["qubits"], r) for k, r in rows.items()
                         if k[0] == noise and k[1] == code and k[3] == p and min(r["errors_x"], r["errors_z"]) >= 5)
            n = [q for q, _ in pts]
            e = [r["d"] * r["eps"] for _, r in pts]
            lo = [r["d"] * (r["eps"] - r["eps_lo"]) for _, r in pts]
            hi = [r["d"] * (r["eps_hi"] - r["eps"]) for _, r in pts]
            ax.errorbar(n, e, yerr=[lo, hi], color=CODE_COLOR[code], marker=CODE_MARK[code], ls="none",
                        ms=5, capsize=0, label=CODE_LABEL[code])
            fit = am.lambda_fit(rows, noise, code, p)
            if fit:
                L, a, _ = fit
                ds = np.arange(3, 61, 2)
                nq = [(2 * d - 1) ** 2 if code == "folded" else 2 * d * d - 1 for d in ds]
                ax.plot(nq, ds * np.exp(a) * L ** (-(ds + 1) / 2), color=CODE_COLOR[code], lw=1.0, alpha=0.7)
                t = am.teraquop(rows, noise, code, p)
                if t:
                    ax.plot([t[1]], [1e-12], marker=CODE_MARK[code], color=CODE_COLOR[code], ms=6,
                            mfc=SURFACE, mew=1.5)
                    ax.annotate("d = %d, %d qubits" % t, (t[1], 1e-12), textcoords="offset points",
                                xytext=(-8, 10 if code == "folded" else -14), ha="right", fontsize=8, color=INK2)
        ax.axhline(1e-12, color=INK2, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(12, 6000)
        ax.set_ylim(1e-14, 1e-1)
        ax.set_xlabel("physical qubits per logical qubit")
        ax.set_title("%s, p = %.1f%%" % (noise, 100 * p), fontsize=9)
    axes[0].set_ylabel("logical error per d-round block")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].text(14, 2e-12, "teraquop target 1e-12", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "memory-footprint.pdf"))
    fig.savefig(os.path.join(OUT, "memory-footprint.png"), dpi=160)
    plt.close(fig)


def fig_lambda(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, noise in zip(axes, ("SD6", "SI1000")):
        for code in ("folded", "rotated"):
            ps, Ls = [], []
            for p in sorted({k[3] for k in rows if k[0] == noise}):
                f = am.lambda_fit(rows, noise, code, p)
                if f:
                    ps.append(p)
                    Ls.append(f[0])
            ax.plot(ps, Ls, color=CODE_COLOR[code], marker=CODE_MARK[code], label=CODE_LABEL[code])
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _percent_axis(ax)
        ax.set_xlabel("physical error rate p")
        ax.set_title(noise, fontsize=9)
    axes[0].set_ylabel("Lambda = eps(d) / eps(d+2)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "memory-lambda.pdf"))
    fig.savefig(os.path.join(OUT, "memory-lambda.png"), dpi=160)
    plt.close(fig)


def fig_gates(rows, gates, p=0.001):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, noise in zip(axes, ("SD6", "SI1000")):
        for g in ("H", "S"):
            ds, ratio, err = [], [], []
            for d in (3, 5, 7, 9):
                r = rows.get((noise, "folded", d, p))
                sh, er = gates.get((noise, g, d, p, "Z"), (0, 0))
                if not r or not sh:
                    continue
                e_g = am.per_round(er, sh, d)
                ratio.append(e_g / r["eps_z"])
                err.append(ratio[-1] * math.sqrt(1 / max(er, 1) + 1 / max(r["errors_z"], 1)))
                ds.append(d)
            ax.errorbar(ds, ratio, yerr=err, color=GATE_COLOR[g], marker=GATE_MARK[g], ms=5, capsize=0,
                        label="logical %s between every two rounds" % g)
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xticks([3, 5, 7, 9])
        ax.set_ylim(0.6, 2.2)
        ax.set_xlabel("code distance d")
        ax.set_title("%s, p = %.1f%%, Z memory" % (noise, 100 * p), fontsize=9)
    axes[0].set_ylabel("error per round / plain memory")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "memory-gates.pdf"))
    fig.savefig(os.path.join(OUT, "memory-gates.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = am.load()
    gates = am.load_gates()
    fig_thresholds(rows)
    fig_footprint(rows)
    fig_lambda(rows)
    fig_gates(rows, gates)
    print("figures written to", OUT)
