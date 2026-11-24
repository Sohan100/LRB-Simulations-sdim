"""Figures for REPORT.md (PNG, matplotlib).  Reads results/*.csv only.

fig1_pstar_vs_q.png        p*(q), k = 1, 2, simulated with SE, plus the model envelope
fig2_pair_coefficient.png  two-fault coefficient per depth step vs q
fig3_rates_vs_p.png        LRB-D (k = 1) and physical RB error versus p, q = 2, 3, 5, 7
fig4_acceptance_floor.png  depth-1 acceptance vs p: fault-free part and the q^-8 floor
fig5_window_rates.png      Part 2: fitted rate vs depth window, by policy
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qfit  # noqa: F401
import merge_highp_threshold_extension as merge

SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated (light), fixed order
QCOL = {2: SLOT[0], 3: SLOT[1], 5: SLOT[2], 7: SLOT[3]}
QMARK = {2: "o", 3: "s", 5: "D", 7: "^"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "lines.linewidth": 2.0, "lines.markersize": 7,
})
NOISE_TITLE = {"si1000": "SI1000", "sd6": "SD6"}


def fig1():
    d = pd.read_csv("results/pstar_vs_q.csv")
    models = {"seven": pd.read_csv("results/model_pstar.csv"),
              "five": pd.read_csv("results/model_pstar_five.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    style = {"five": (SLOT[0], "o", "five-moment round (paper)"),
             "seven": (SLOT[1], "s", "seven-moment round (published runs)")}
    for ax, noise in zip(axes, ("si1000", "sd6")):
        for rnd, (col, mk, lab) in style.items():
            mm = models[rnd]
            mm = mm[(mm.noise == noise) & (mm["compile"] == "production")].sort_values("q")
            ax.plot(mm.q, mm.pstar_model, "--", color=col, lw=1.2, label=f"model envelope, {lab}")
            g = d[(d.noise == noise) & (d.k == 1) & (d["round"] == rnd)].sort_values("q")
            ok = g.p_star_quotable.astype(bool)
            ax.errorbar(g.q[ok], g.p_star[ok], yerr=g.se[ok], fmt=mk + "-", color=col, lw=1.6,
                        capsize=3, label=f"simulated p*, k = 1, {lab}")
            for r in g[~ok].itertuples():
                suf = "" if rnd == "seven" else "_five"
                rates = pd.read_csv(f"results/final_q{r.q}_{noise}_production{suf}_rates.csv")
                lb = float(rates.p.max())  # no crossing up to the largest simulated p
                ax.annotate("", xy=(r.q, lb + 0.012), xytext=(r.q, lb),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
                ax.plot([r.q], [lb], mk, color=col, mfc=SURF)
                ax.annotate(f"> {lb:.2f}", (r.q, lb), textcoords="offset points", xytext=(7, -3),
                            fontsize=7.5, color=INK2)
            for r in g[ok].itertuples():
                ax.annotate(f"{r.p_star:.4f}", (r.q, r.p_star), textcoords="offset points",
                            xytext=(6, 6 if rnd == "five" else -12), fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 7, 11, 13, 17, 23, 31])
        ax.set_xticklabels(["2", "3", "5", "7", "11", "13", "17", "23", "31"])
        ax.set_xlabel("local dimension q")
        ax.set_title(f"{NOISE_TITLE[noise]}, one-ancilla layout, k = 1", loc="left")
    axes[0].set_ylabel("pseudo-threshold p*")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[1].text(0.98, 0.04, "open marker with arrow: no crossing up to the largest\nsimulated p (lower bound)",
                 transform=axes[1].transAxes, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig("figures/fig1_pstar_vs_q.png", dpi=160)


def fig2():
    s = pd.read_csv("results/pair_expansion_main.csv.gz")
    s = s[s.w >= 0]
    g = s.groupby(["noise", "compile", "q", "depth"]).S2b.sum().unstack("depth").fillna(0) / 10
    step = ((g[4] - g[1]) / 3).rename("s").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, noise in zip(axes, ("si1000", "sd6")):
        for mode, col, mk in (("production", SLOT[0], "o"), ("inverse", SLOT[1], "s")):
            h = step[(step.noise == noise) & (step["compile"] == mode)].sort_values("q")
            ax.plot(h.q, h.s, "-", color=col, marker=mk, label=f"{mode} compile")
        h = step[(step.noise == noise) & (step["compile"] == "production")].sort_values("q")
        q0, s0 = 5, float(h[h.q == 5].s.iloc[0])
        qq = np.array([3, 31])
        ax.plot(qq, s0 * (qq / q0) ** -2.0, ":", color=INK2, lw=1.2, label="slope q^-2")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([2, 3, 5, 7, 11, 13]); ax.set_xticklabels(["2", "3", "5", "7", "11", "13"])
        ax.set_xlabel("local dimension q")
        ax.set_title(f"{NOISE_TITLE[noise]}: two-fault coefficient per depth step", loc="left", fontsize=10)
    axes[0].set_ylabel("s(q)   [r_LRB = s p^2 at low p]")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig2_pair_coefficient.png", dpi=160)


def published_rates(q, noise):
    fam = {(2, "si1000"): ("qubit-si1000", "qubit-one-ancilla-si1000"),
           (2, "sd6"): ("qubit-sd6", "qubit-one-ancilla-sd6"),
           (3, "si1000"): ("si1000", "one-ancilla-si1000"),
           (3, "sd6"): ("sd6", "one-ancilla-sd6")}[(q, noise)]
    t = pd.read_csv(merge.CAMPAIGNS[fam[0]]["out"] / fam[1] / "unif_lrb_vs_rb_table_merged.csv")
    t = t[t.check_num == 1].sort_values("p")
    return t.p.to_numpy(), t.lrb_error_rate.to_numpy(), t.rb_error_rate.to_numpy()


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, noise in zip(axes, ("si1000", "sd6")):
        for q in (2, 3, 5, 7):
            if q in (2, 3):
                p, lrb, rb = published_rates(q, noise)
            else:
                try:
                    t = pd.read_csv(f"results/final_q{q}_{noise}_production_rates.csv").sort_values("p")
                except FileNotFoundError:
                    continue
                p, lrb, rb = t.p.to_numpy(), t.r_lrb_k1.to_numpy(), t.r_rb.to_numpy()
            ok = np.isfinite(lrb) & (lrb > 0)
            ax.plot(p[ok], lrb[ok], "-", color=QCOL[q], marker=QMARK[q], ms=4, label=f"q = {q}  LRB-D k = 1")
            ax.plot(p, rb, "--", color=QCOL[q], lw=1.4, label=f"q = {q}  physical RB")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-3, 0.12); ax.set_ylim(1e-6, 1.0)
        ax.set_xlabel("two-qudit gate error p")
        ax.set_title(f"{NOISE_TITLE[noise]}, one-ancilla layout", loc="left")
    axes[0].set_ylabel("error per forward+inverse pair (pipeline fit)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig3_rates_vs_p.png", dpi=160)


def fig4():
    curves = pd.read_csv("results/model_curves.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, noise in zip(axes, ("si1000", "sd6")):
        for q in (5, 7):
            try:
                t = pd.read_csv(f"results/final_q{q}_{noise}_production_rates.csv").sort_values("p")
            except FileNotFoundError:
                continue
            t = t[t.source == "ext"]
            a1 = [float(x.split(";")[1]) for x in t.acc_k1]
            ax.plot(t.p, a1, "o", color=QCOL[q], label=f"q = {q} simulated, depth 1")
            c = curves[(curves.q == q) & (curves.noise == noise) & (curves["compile"] == "production")]
            ax.plot(c.p, c.C1 + float(q) ** -8, "-", color=QCOL[q], lw=1.4, label=f"q = {q} model C1 + q^-8")
            ax.axhline(float(q) ** -8, color=QCOL[q], lw=1.0, ls=":")
            ax.annotate(f"q^-8 = {float(q) ** -8:.1e}", (0.031, float(q) ** -8), xytext=(0, 3),
                        textcoords="offset points", fontsize=8, color=INK2)
        ax.set_yscale("log"); ax.set_xlim(0.03, 0.1); ax.set_ylim(1e-8, 1e-2)
        ax.set_xlabel("two-qudit gate error p")
        ax.set_title(f"{NOISE_TITLE[noise]}: depth-1 acceptance, k = 1", loc="left")
    axes[0].set_ylabel("accepted fraction")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig4_acceptance_floor.png", dpi=160)


def fig5():
    d = pd.read_csv("results/single_exp_rows.csv.gz")
    cases = [("qutrit-one-si1000", 0.005, "production"), ("qutrit-one-sd6", 0.002, "production"),
             ("qubit-one-si1000", 0.002, "production"), ("qutrit-two-sd6", 0.002, "production")]
    pols = [("unif1", SLOT[0], "o"), ("unif2", SLOT[1], "s"), ("unif3", SLOT[2], "D"), ("const0", SLOT[3], "^")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (cfg, p, kind) in zip(axes, cases):
        for pol, col, mk in pols:
            r = d[(d.config == cfg) & (d.policy == pol) & np.isclose(d.p, p) & (d.run_kind == kind)]
            if not len(r) or not isinstance(r.window_rates.iloc[0], str):
                continue
            pts = [tuple(map(float, x.split(":"))) for x in r.window_rates.iloc[0].split(";")]
            m, rr = np.array(pts).T
            ax.plot(m, rr / rr[-1], "-", color=col, marker=mk, ms=5,
                    label={"unif1": "k = 1", "unif2": "k = 2", "unif3": "k = 3", "const0": "terminal check"}[pol])
        ax.axhspan(0.9, 1.1, color=GRID, alpha=0.6, lw=0)
        ax.set_ylim(0, 2.0)
        ax.set_xlabel("deepest depth in the window 0..m")
        ax.set_title(f"{cfg}, p = {p}", loc="left", fontsize=9)
    axes[0].set_ylabel("fitted rate / full-window rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig5_window_rates.png", dpi=160)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep going; report which failed
            print("FAILED", f.__name__, repr(e))
