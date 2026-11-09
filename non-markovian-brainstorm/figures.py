"""Figures for REPORT.md. Run after diagnose.py, estimators.py and run_variants.sh."""
from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nmlib as L

warnings.filterwarnings("ignore", category=RuntimeWarning)
FIG = os.path.join(L.OUT, "figures")
RES = os.path.join(L.OUT, "results")
# reference categorical palette, fixed order (dataviz skill, references/palette.md)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "lines.markersize": 4.5, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def model(m, fit):
    return fit["b"] * fit["f"] ** np.asarray(m, float)


def fig1_worked_example():
    """One-ancilla SD6, unif 2, p = 0.03: production vs extension."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    mm = np.linspace(0, 6, 100)
    for key, col, lab in [(("one", "sd6", "prod"), C[0], "production (1M shots)"),
                          (("one", "sd6", "ext"), C[1], "extension (20M shots)")]:
        i = L.p_index(key, 0.03)
        d = np.asarray(L.meta(key).depths)
        t, rj = L.counts(key, i, "unif", 2)
        c = L.curves(t, rj)
        okp = np.isfinite(c["pipeline"])
        fit = L.fit_fixed_a(d, c["pipeline"], c["pipeline_std"])
        a.plot(d[okp], c["pipeline"][okp], "^", color=col, mfc="none", mew=1.2,
               label=f"{lab}: pipeline |.|, fit r = {fit['r']:.3f}")
        a.plot(mm, model(mm, fit), "-", color=col, lw=1.1, alpha=0.8)
        okq = np.isfinite(c["pooled"]) & (c["n_acc_tot"] >= 50)
        a.plot(d[okq], c["pooled"][okq], "o", color=col, ms=4, label=f"{lab}: pooled signed")
        a.plot(d[okp], c["rice_floor"][okp], ":", color=col, lw=1.2)
        b.semilogy(d, np.where(c["n_acc_min"] > 0, c["n_acc_min"], np.nan), "o-", color=col, label=lab)
    a.text(2.6, 0.64, "dotted: mean |.| of pure noise\n(Rayleigh floor per sequence)", color=INK2, fontsize=6.5)
    a.set_xlabel("depth m (forward+inverse pairs)")
    a.set_ylabel("accepted polarization")
    a.set_ylim(-0.25, 0.75)
    a.legend(fontsize=6.3, loc="lower left")
    a.set_title("(a) one-ancilla SD6, unif 2, p = 0.03", fontsize=8.5, loc="left")
    b.axhline(30, color=INK2, lw=0.8, ls="--")
    b.text(0.1, 40, "~30 accepted: |.| bias ~0.16", color=INK2, fontsize=7)
    b.set_xlabel("depth m")
    b.set_ylabel("fewest accepted shots in any sequence")
    b.legend(fontsize=7)
    b.set_title("(b) shots behind each point", fontsize=8.5, loc="left")
    save(fig, "fig1_worked_example.png")


def fig2_windows():
    """unif 1 windowed decay rates with bootstrap errors, one-ancilla SI1000 vs SD6."""
    z = np.load(os.path.join(RES, "curves.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, noise in zip(axes, ["si1000", "sd6"]):
        for j, p in enumerate([0.001, 0.002, 0.005]):
            tag = f"one-{noise}-prod|{p}|unif1"
            d = z[f"{tag}|depths"]
            y = z[f"{tag}|pooled"]
            se = z[f"{tag}|pooled_se"]
            n = z[f"{tag}|n_acc_tot"]
            ok = np.isfinite(y) & (n >= 1000)
            d, y, se = d[ok], y[ok], se[ok]
            mids, w = L.windowed_rates(d, y)
            we = (2 / 3) * np.sqrt((se[1:] / y[1:]) ** 2 + (se[:-1] / y[:-1]) ** 2) / np.diff(d)
            ax.errorbar(mids + 0.25 * (j - 1), w / p ** 2, yerr=we / p ** 2, fmt="o-", color=C[j], capsize=2,
                        lw=1.2, ms=3.5, label=f"p = {p:g}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("window midpoint m")
        ax.set_title(f"({'a' if noise == 'si1000' else 'b'}) one-ancilla {noise.upper()}, unif 1",
                     fontsize=8.5, loc="left")
    axes[0].set_ylim(-15, 40)
    axes[0].set_ylabel("windowed error per step / p²")
    axes[1].legend(fontsize=7, loc="upper left")
    save(fig, "fig2_unif1_windowed_rates.png")


def fig3_c2_vs_k():
    d = pd.read_csv(os.path.join(RES, "curvature_scan.csv"))
    d = d[(d.source == "prod") & d.policy.str.startswith("unif")].copy()
    d["k"] = d.policy.str[4:].astype(int)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    sel = [("one-sd6", 0.0005, C[0]), ("one-sd6", 0.001, C[1]), ("one-sd6", 0.002, C[2]),
           ("two-sd6", 0.001, C[3]), ("one-si1000", 0.005, C[4]), ("one-si1000", 0.0075, C[6])]
    for cfg, p, col in sel:
        s = d[(d.config == cfg) & np.isclose(d.p, p)].sort_values("k")
        anc, noise = cfg.split("-")
        lab = f"{anc}-ancilla {noise.upper()}, p = {p:g}"
        a.errorbar(s.k, s.c2 / p ** 2, yerr=[(s.c2 - s.c2_lo) / p ** 2, (s.c2_hi - s.c2) / p ** 2],
                   fmt="o-", color=col, capsize=2, lw=1.2, label=lab)
        b.plot(s.k, s.c1 / p ** 2, "o-", color=col, lw=1.2, label=lab)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlabel("postselection interval k")
    a.set_ylabel("c2 / p²  (m² coefficient of -ln λ)")
    a.set_title("(a) memory term: flat in k", fontsize=8.5, loc="left")
    b.set_xlabel("postselection interval k")
    b.set_ylabel("c1 / p²  (linear coefficient)")
    b.set_title("(b) per-step term: grows with k", fontsize=8.5, loc="left")
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig3_c2_flat_in_k.png")


def fig4_variants():
    z = np.load(os.path.join(RES, "variant_sim_sd6_p0.002_shots1000000.npz"))
    d = z["depths"].astype(float)
    labels = {"mirror": "mirror inverse, unchecked (production)",
              "checked_inverse": "mirror inverse, check after each inverse",
              "compiled": "one compiled recovery Clifford"}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), gridspec_kw={"wspace": 0.42})
    for j, v in enumerate(["mirror", "checked_inverse", "compiled"]):
        t, rj = z[f"{v}_tallies"].astype(float), z[f"{v}_rejected"].astype(float)
        c = L.curves(t, rj)
        bs = L.bootstrap_curves(t, rj, n_boot=200, fields=("pooled",))
        se = np.nanstd(bs["pooled"], 0)
        ok = c["n_acc_tot"] >= 1000
        y = -np.log(c["pooled"] / c["pooled"][0])
        ye = se / c["pooled"]
        fit = L.fit_fixed_a(d[ok], c["pooled"][ok], se[ok], y_ceiling=None)
        axes[0].errorbar(d[ok], y[ok] * 1e3, yerr=ye[ok] * 1e3, fmt="o", color=C[j], ms=3.5, capsize=1.5,
                         label=labels[v])
        mm = np.linspace(0, 22, 50)
        axes[0].plot(mm, -np.log(model(mm, fit) / c["pooled"][0]) * 1e3, "-", color=C[j], lw=1, alpha=0.7)
        mids, w = L.windowed_rates(d[ok], c["pooled"][ok])
        axes[1].plot(mids, w * 1e4, "o-", color=C[j], ms=3.5)
        axes[2].semilogy(d, c["acc"], "o-", color=C[j], ms=3.5)
    axes[0].set_xlabel("depth m")
    axes[0].set_ylabel("-ln(λ(m)/λ(0))  ×10³")
    axes[0].set_title("(a) decay; lines: single exp.", fontsize=8.5, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[1].set_xlabel("window midpoint m")
    axes[1].set_ylabel("windowed error per step ×10⁴")
    axes[1].set_title("(b) local rate", fontsize=8.5, loc="left")
    axes[2].set_xlabel("depth m")
    axes[2].set_ylabel("acceptance A(m)")
    axes[2].set_title("(c) acceptance", fontsize=8.5, loc="left")
    fig.suptitle("Single-ancilla SD6, p = 0.002, unif 1, 30 sequences × 1M shots (login-node sdim DEM)",
                 fontsize=8.5, x=0.01, ha="left", y=1.03)
    save(fig, "fig4_protocol_variants_sd6_p0.002.png")


def fig5_estimators():
    d = pd.read_csv(os.path.join(RES, "estimator_table.csv"))
    t = pd.read_csv(os.path.join(RES, "pseudo_thresholds_by_estimator.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), gridspec_kw={"wspace": 0.3})
    panels = [("sd6 merged", "one-ancilla-sd6", 1), ("sd6 merged", "one-ancilla-sd6", 2),
              ("si1000+ext", "one-ancilla-si1000", 2)]
    names = {"pipe": "pipeline (production)", "pooled": "pooled signed", "aligned": "pooled, depths ∝ k",
             }
    for ax, (grid, cfg, k) in zip(axes, panels):
        s = d[(d.grid == grid) & (d.config == cfg) & (d.k == k) & (d.p >= 0.005)].sort_values("p")
        for j, e in enumerate(["pipe", "pooled", "aligned"]):
            if k == 1 and e == "aligned":
                continue
            ps = t[(t.grid == grid) & (t.config == cfg) & (t.k == k) & (t.estimator == e)].p_star.values[0]
            ax.loglog(s.p, s[e], "o-", color=C[j], ms=3, lw=1.2,
                      label=f"{names[e]}: p* = {ps:.4f}" if np.isfinite(ps) else f"{names[e]}: no p*")
        ax.loglog(s.p, s.rb, "s--", color=INK2, ms=3, lw=1.2, label="physical RB")
        ax.set_xlabel("p")
        anc, _, noise = cfg.split("-")
        ax.set_title(f"{anc}-ancilla {noise.upper()}, unif {k}", fontsize=8.5, loc="left")
        v = s[["pipe", "pooled", "aligned"]].values
        lo = np.nanmin(np.where(v > 0, v, np.nan))
        ax.set_ylim(bottom=lo * 0.06)
        ax.legend(fontsize=5.6, loc="lower right")
    axes[0].set_ylabel("error per depth step")
    save(fig, "fig5_estimators_vs_rb.png")


if __name__ == "__main__":
    fig1_worked_example()
    fig2_windows()
    fig3_c2_vs_k()
    fig4_variants()
    fig5_estimators()
