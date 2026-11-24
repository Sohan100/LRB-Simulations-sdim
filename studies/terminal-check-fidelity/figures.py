"""Figures for REPORT.md, from the CSV/NPZ files in results/.

Static PNGs (matplotlib). Colours follow one fixed assignment in every
figure (validated categorical palette, light surface):

  blue    terminal-check logical infidelity / block-rate mode fit
  orange  free-rate mode fit
  aqua    direct quadratic fit of the accepted infidelity
  yellow  single-exponential fit (the production fit model)
  magenta physical RB
  green   LRB-D with a noisy check after every Clifford (unif-1)

Depth ramps in p use one hue, light to dark.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import tclib as T
import tcfit as F
import tcsteps as S
from tcref import reference

RES = os.path.join(T.HERE, "results")
FIG = os.path.join(T.HERE, "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # sequential blue, light -> dark
NAMES = {("qubit", "si1000"): "qubit code, SI1000", ("qubit", "sd6"): "qubit code, SD6",
         ("qutrit", "si1000"): "qutrit code, SI1000", ("qutrit", "sd6"): "qutrit code, SD6"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5, "axes.axisbelow": True, "figure.dpi": 150,
})


def read_csv(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    try:
        return float(row[key])
    except (ValueError, TypeError):
        return np.nan


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def plabel(p):
    return f"p = {p:g}"


# ------------------------------------------------------------------ figure 1
def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
    show = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    for ax, key in zip(axes.ravel(), NAMES):
        z = np.load(os.path.join(RES, f"curves_{key[0]}_{key[1]}.npz"))
        probs = list(z["probs"])
        dense, pd_ = z["dense"], z["prod_depths"]
        d = 2 if key[0] == "qubit" else 3
        for color, p in zip(RAMP, show):
            pi = probs.index(p)
            ax.plot(dense[1:], z[f"eps_{pi}"][1:], color=color, zorder=2, marker="o", markevery=[],
                    mec=SURFACE, mew=0.8, label=plabel(p))
            e, se = z[f"prod_eps_{pi}"], z[f"prod_eps_se_{pi}"]
            ok = (pd_ > 0) & np.isfinite(e) & (e > 0)
            ax.errorbar(pd_[ok], e[ok], yerr=se[ok], fmt="o", color=color, mec=SURFACE, mew=0.8,
                        ms=5, elinewidth=0.8, zorder=3)
        m = np.array([1.0, 40.0])
        ref = z[f"eps_{probs.index(1e-3)}"][1] * 0.45
        ax.plot(m, ref * m ** 2, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate("slope 2", (2.6, ref * 2.6 ** 2), xytext=(3, -11), textcoords="offset points",
                    fontsize=7.5, color=INK2)
        ax.axhline((d - 1) / d, color=INK2, lw=0.8, ls=":")
        ax.annotate("fully depolarized", (44, (d - 1) / d), xytext=(0, 3), textcoords="offset points",
                    fontsize=7, color=INK2, ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(NAMES[key], loc="left")
        ax.set_xlim(0.9, 45)
        ax.set_ylim(3e-7, 2.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for h in handles:
        h.set_markevery(None)
    axes[0, 0].legend(handles[::-1], labels[::-1], loc="lower right", fontsize=7.5, handlelength=2.2,
                      title="colour: p", title_fontsize=7.5)
    for h in handles:
        h.set_markevery([])
    for ax in axes[1]:
        ax.set_xlabel("depth m (forward + mirror-inverse pairs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accepted logical infidelity")
    fig.suptitle("Terminal check: the accepted logical infidelity grows as the square of the depth",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, 0.005, "Lines: exact average over all Clifford sequences. Points: production const-0 data, "
             "30 sequences x 1M shots, bootstrap standard errors.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "fig1_quadratic_law.png")


# ------------------------------------------------------------------ figure 2
def fig2():
    key, p = ("qubit", "si1000"), 5e-3
    sm = S.StepModel(*key)
    d = sm.d
    z = np.load(os.path.join(RES, f"curves_{key[0]}_{key[1]}.npz"))
    pi = list(z["probs"]).index(p)
    pd_ = z["prod_depths"]
    ref = reference(key[0], key[1], p)
    rows = [r for r in read_csv("characterization.csv") if r["code"] == key[0] and r["noise"] == key[1]
            and abs(fnum(r, "p") - p) < 1e-12][0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    dense = np.arange(0, 41)
    ex = sm.exact_curve(p, dense)
    lam_prod = 1 - d / (d - 1) * z[f"prod_eps_{pi}"]
    se_prod = d / (d - 1) * z[f"prod_eps_se_{pi}"]
    bfit, ffit = z[f"fit1_{pi}"]
    a.plot(dense, ex["lam"], color=BLUE, label="exact (sum of seven exponentials, ratio)")
    a.plot(dense, bfit * ffit ** dense, color=YELLOW, ls=(0, (5, 2)),
           label="single exponential fitted on depths 0 to 22")
    a.errorbar(pd_, lam_prod, yerr=se_prod, fmt="o", color=INK, mec=SURFACE, mew=0.8, ms=5,
               elinewidth=0.8, label="production data", zorder=3)
    a.set_xlabel("depth m")
    a.set_ylabel(r"accepted $\langle \bar X \rangle$")
    a.set_title("One decay constant does not describe the curve", loc="left")
    a.legend(loc="lower left", fontsize=7.5)
    # (b) what the single-exponential rate is, as a function of the deepest depth fitted
    mmax = np.arange(2, 41)
    r_fit = []
    for mm in mmax:
        dep = np.arange(0, mm + 1)
        r_fit.append(F.single_exponential(dep, sm.exact_curve(p, dep)["P"] * 3e7, d, min_accepted=1)["r"])
    b.axhline(fnum(rows, "rb_pair_official"), color=MAGENTA, label="physical RB, per pair")
    b.plot(mmax, r_fit, color=YELLOW, label="single-exponential fit, depths 0 to m_max")
    b.axhline(ref["true_unit"], color=BLUE, label="one logical pair between perfect checks (truth)")
    b.set_yscale("log")
    b.set_xlabel("deepest depth in the fit, m_max")
    b.set_ylabel("error per depth unit")
    b.set_title("A single-exponential rate grows with the fit window", loc="left")
    b.set_ylim(2.5e-5, 6e-3)
    b.legend(loc="upper left", fontsize=7.5, ncol=1, bbox_to_anchor=(0.0, 0.9))
    fig.suptitle(f"Qubit code, SI1000, p = {p:g}", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_single_exponential_fails.png")


# ------------------------------------------------------------------ figure 3
def operator_name(k, d):
    """X-type operator measured by character k = (a1, a2, b): S1^a1 S2^a2 L^b."""
    s1, s2, lo = np.array([1, 1, 1, 0, 0]), np.array([0, 1, 0, 1, 1]), np.array([1, 0, 0, 1, 0])
    v = (k[0] * s1 + k[1] * s2 + k[2] * lo) % 2
    return "".join(f"X{q}" for q in range(5) if v[q])


def fig3():
    key, pi = ("qubit", "si1000"), 6
    d = 2
    z = np.load(os.path.join(RES, f"resample_{key[0]}_{key[1]}_p{pi}.npz"))
    cnt, depths, p = z["counts"], z["depths"], float(z["p"])
    md = F.modes(d)
    chi = F.empirical_chars(cnt, d)
    fit = F.mode_fit(depths, cnt, d, "block")
    grid = T.char_grid(d)
    xs = np.linspace(0, depths.max(), 100)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.8, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    ends = []
    for i, k in enumerate(md.reps):
        fam = "logical" if grid[k][2] else "stabilizer"
        color = AQUA if fam == "logical" else VIOLET
        a.plot(xs, np.exp(fit["logB"][i] - fit["gamma"][i] * xs), color=color, lw=1.3)
        a.plot(depths, chi[:, k], "o", color=color, mec=SURFACE, mew=0.8, ms=5)
        ends.append((chi[-1, k], operator_name(grid[k], d), fam))
    ends.sort()
    ypos = np.geomspace(0.5, 0.82, len(ends))
    for (y, name, fam), yl in zip(ends, ypos):
        a.annotate(name, (depths.max(), y), xytext=(depths.max() + 2.2, yl), fontsize=7.5, color=INK2,
                   va="center", arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
    a.plot([], [], "o-", color=VIOLET, label="stabilizer modes (set the acceptance)")
    a.plot([], [], "o-", color=AQUA, label="logical-X modes (set the signal)")
    a.set_yscale("log")
    a.set_ylim(0.47, 1.03)
    a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlim(-0.5, depths.max() + 8)
    a.set_xlabel("depth m")
    a.set_ylabel("mean parity over all shots")
    a.set_title("Each Fourier mode of the readout decays exponentially", loc="left")
    a.legend(loc="lower left", fontsize=7.5)
    xs = np.linspace(1, depths.max(), 100)
    # (b) accepted infidelity: observed vs predicted from the mode rates
    st = F.accepted_stats(cnt, d)
    boot = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        boot.append(F.accepted_stats(cnt[rng.integers(0, 30, 30)], d)["eps"])
    se = np.std(boot, 0)
    chi_fit = np.exp(fit["logB"][None, :] - np.outer(xs, fit["gamma"]))
    Pfit = 1.0 / d ** 3 + md.expand(chi_fit)[:, 1:] @ md.F[:, 1:].T
    _, _, lam_fit = T.accept_and_polarization(Pfit, d)
    b.plot(xs, T.infidelity_from_polarization(lam_fit, d), color=BLUE,
           label="predicted from the mode rates")
    ok = depths > 0
    b.errorbar(depths[ok], st["eps"][ok], yerr=se[ok], fmt="o", color=INK, mec=SURFACE, mew=0.8, ms=5,
               elinewidth=0.8, label="observed on accepted shots", zorder=3)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("depth m")
    b.set_ylabel("accepted logical infidelity")
    b.set_title("They reproduce the postselected curve", loc="left")
    b.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Qubit code, SI1000, p = {p:g}: production const-0 circuits re-sampled with the full "
                 "readout record", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig3_mode_decays.png")


# ------------------------------------------------------------------ figure 4
def fig4():
    bias = [r for r in read_csv("estimator_bias.csv") if r["protocol"] == "mirror" and r["spam"] == "noisy"
            and r["depths"] == "designed" and r["noise"] == "si1000"]
    fs = [r for r in read_csv("finite_shots.csv") if r["protocol"] == "mirror" and r["noise"] == "si1000"
          and r["grid"] == "designed" and fnum(r, "reach") == 1.0 and r["nested"] == "True"
          and fnum(r, "meas_scale") == 1.0 and int(r["n_seq"]) == 30 and int(r["shots"]) == 1000]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=True)
    for ax, code in zip(axes, ("qubit", "qutrit")):
        rows = sorted([r for r in bias if r["code"] == code], key=lambda r: fnum(r, "p"))
        ps = np.array([fnum(r, "p") for r in rows])
        series = [
            ("single exponential", YELLOW, [fnum(r, "ratio_single_exp") for r in rows], "s"),
            ("direct quadratic", AQUA, [fnum(r, "direct_c2") / fnum(r, "true_unit_infidelity") for r in rows], "D"),
            ("free-rate mode fit", ORANGE, [fnum(r, "ratio_free_eps1") for r in rows], "o"),
            (r"block-rate mode fit $\times\, F_{\rm cov}$", BLUE,
             [fnum(r, "mode_block_beta") * fnum(r, "f_cov") / fnum(r, "true_unit_infidelity") for r in rows], "o"),
        ]
        for name, color, y, mk in series:
            ax.plot(ps, y, marker=mk, color=color, mec=SURFACE, mew=0.8, label=name)
        for r in fs:
            if r["code"] != code:
                continue
            p = fnum(r, "p")
            fc = reference(code, "si1000", p)["f_cov"]
            co = reference(code, "si1000", p)["coincidence"]
            tu = reference(code, "si1000", p)["true_unit"]
            ax.errorbar([p * 1.07], [fnum(r, "block_mean_over_ref") * co * fc / tu],
                        yerr=[fnum(r, "block_sd_over_ref") * co * fc / tu], fmt="none", ecolor=BLUE,
                        elinewidth=2.2, capsize=0)
            lo = max(fnum(r, "free_eps1_mean_over_ref") - fnum(r, "free_eps1_sd_over_ref"), 0.12)
            hi = fnum(r, "free_eps1_mean_over_ref") + fnum(r, "free_eps1_sd_over_ref")
            ax.plot([p * 0.93] * 2, [lo, hi], color=ORANGE, lw=2.2, solid_capstyle="butt")
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 300)
        ax.set_xlabel("two-qudit gate error p (SI1000)")
        ax.set_title(f"{code} code", loc="left")
    axes[0].set_ylabel("estimate / true logical infidelity of one pair")
    axes[1].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Estimators against the exact answer (noisy encoder and readout included)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, 0.005, "Markers: infinite-data limit. Thick bars at p = 0.002, 0.005, 0.01: one-standard-deviation "
             "scatter over 200 synthetic runs of 30 sequences x 1000 shots x 9 depths.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save(fig, "fig4_estimator_validation.png")


# ------------------------------------------------------------------ figure 5
def fig5():
    fs = [r for r in read_csv("finite_shots.csv") if r["protocol"] == "mirror" and r["noise"] == "si1000"
          and r["grid"] == "designed" and fnum(r, "reach") == 1.0 and r["nested"] == "True"
          and fnum(r, "meas_scale") == 1.0 and r["code"] == "qubit" and abs(fnum(r, "p") - 5e-3) < 1e-12]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True)
    shades = {10: 0, 30: 2, 100: 4}
    for n_seq, mk in ((10, "v"), (30, "o"), (100, "s")):
        rows = sorted([r for r in fs if int(r["n_seq"]) == n_seq], key=lambda r: fnum(r, "total_shots"))
        x = [fnum(r, "total_shots") for r in rows]
        a.plot(x, [100 * fnum(r, "block_sd_over_ref") for r in rows], marker=mk, color=RAMP[shades[n_seq]],
               mec=SURFACE, mew=0.8, label=f"{n_seq} sequences")
        b.plot(x, [100 * fnum(r, "free_eps1_sd_over_ref") for r in rows], marker=mk,
               color=["#f4a57f", ORANGE, "#a8421a"][list(shades).index(n_seq)], mec=SURFACE, mew=0.8,
               label=f"{n_seq} sequences")
    for ax, title in ((a, "Block-rate fit: limited by the number of sequences"),
                      (b, "Free-rate fit: limited by the number of shots")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total shots (all sequences and depths)")
        ax.set_title(title, loc="left")
        ax.legend(fontsize=7.5)
    for ax, ticks in ((a, [1.5, 2, 3, 5, 10]), (b, [20, 50, 100, 200, 500])):
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_ylabel("standard error of the estimate (%)")
    fig.suptitle("Shot budget, qubit code, SI1000, p = 0.005, nine depths up to m = 48, noisy SPAM",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig5_shot_budget.png")


# ------------------------------------------------------------------ figure 6
def fig6():
    xt = read_csv("robustness_crosstalk.csv")
    dr = read_csv("robustness_drift.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for code, ls in (("qubit", "-"), ("qutrit", (0, (4, 2)))):
        rows = [r for r in xt if r["code"] == code and fnum(r, "xt_per_clifford") > 0]
        x = [fnum(r, "xt_per_clifford") for r in rows]
        a.plot(x, [fnum(r, "block_times_fcov_over_true") for r in rows], "o", ls=ls, color=BLUE, mec=SURFACE,
               mew=0.8, label=rf"block-rate fit $\times\, F_{{\rm cov}}$, {code}")
        a.plot(x, [fnum(r, "free_eps1_over_true") for r in rows], "o", ls=ls, color=ORANGE, mec=SURFACE,
               mew=0.8, label=f"free-rate fit, {code}")
        rows = [r for r in dr if r["code"] == code]
        x = [fnum(r, "sigma_rel") for r in rows]
        b.plot(x, [fnum(r, "block_times_fcov_over_true") for r in rows], "o", ls=ls, color=BLUE, mec=SURFACE,
               mew=0.8, label=rf"block-rate fit $\times\, F_{{\rm cov}}$, {code}")
        b.plot(x, [fnum(r, "free_eps1_over_true") for r in rows], "o", ls=ls, color=ORANGE, mec=SURFACE,
               mew=0.8, label=f"free-rate fit, {code}")
    for ax in (a, b):
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.5, 1, 2, 5])
        ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_ylim(0.12, 8)
    a.set_xscale("log")
    a.set_xlabel("probability of a correlated error on qudits 0 and 2, per Clifford")
    a.set_ylabel("estimate / true logical infidelity of one pair")
    a.set_title("Correlated faults on a logical support", loc="left")
    a.legend(fontsize=7, loc="lower left")
    b.set_xlabel("relative shot-to-shot spread of p (log-normal)")
    b.set_title("Quasi-static drift of the noise strength", loc="left")
    fig.suptitle("Where the model behind the estimate breaks (SI1000, p = 0.005, infinite-data limit)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig6_failure_modes.png")


# ------------------------------------------------------------------ figure 7
def fig7():
    ch = read_csv("characterization.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), NAMES):
        rows = sorted([r for r in ch if (r["code"], r["noise"]) == key], key=lambda r: fnum(r, "p"))
        ps = np.array([fnum(r, "p") for r in rows])
        series = [
            ("physical RB", MAGENTA, "rb_pair_official", "s"),
            ("LRB-D, noisy check after every Clifford (unif-1)", GREEN, "unif1_pair_official", "D"),
            ("terminal check, single-exponential fit", YELLOW, "single_exp_r_prod_depths", "v"),
            ("logical pair between perfect checks (truth)", BLUE, "eps_pair_true", "o"),
        ]
        for name, color, col, mk in series:
            y = np.array([fnum(r, col) for r in rows])
            ok = y > 0
            ax.plot(ps[ok], y[ok], marker=mk, color=color, mec=SURFACE, mew=0.8, ms=4.5, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(NAMES[key], loc="left")
        ax.set_ylim(1e-8, 1)
    for ax in axes[1]:
        ax.set_xlabel("two-qudit gate error p")
    for ax in axes[:, 0]:
        ax.set_ylabel("infidelity per forward + inverse pair")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("What terminal-check data measures: the logical gates without syndrome-extraction noise",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig7_logical_vs_physical.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for w in which:
        globals()[f"fig{w}"]()
