"""When and where the terminal-check logical gates beat the physical qudit.

Exact, no sampling. For every (code, noise, p) it compares, depth by depth,

* logical:  the accepted infidelity of m Clifford pairs followed by the
            terminal check, eps_L(m), with ideal SPAM (tcsteps transfer matrix,
            uniformly random sequences), and the accepted fraction A(m);
* physical: (d-1)/d (1 - f^m) with f from the run's official physical RB rate
            per pair, i.e. the SPAM-free infidelity of m physical pairs.

Both are Clifford-group averages and both are free of SPAM, so the ratio
gain(m) = physical(m) / logical(m) is a like-for-like comparison of the gates.

Writes results/logical_vs_physical.csv and figures/fig8_when_logical_wins.png.
"""
from __future__ import annotations

import csv
import os

import numpy as np

import tclib as T
import tcsteps as S

RES = os.path.join(T.HERE, "results")
GAIN_DEPTHS = [1, 2, 5, 10, 22, 50, 100]
CAP = 60000


def curves(sm, p, r_rb, cap=CAP):
    """eps_L(m), A(m), physical(m) for m = 0.. until past break-even and A < 1/2."""
    d = sm.d
    f = 1 - d * r_rb / (d - 1)
    tm = sm.transfer(p, "mirror")
    v = np.zeros((sm.K, sm.nf))
    v[:, sm.f0] = 1.0
    chi = [v.sum(1)]
    crossed = half = False
    m = 0
    while m < cap:
        m += 1
        v = np.einsum("kab,kb->ka", tm, v)
        c = v.sum(1)
        chi.append(c)
        if m % 16 == 0 or m < 16:
            A, _, lam = T.accept_and_polarization(T.dist_from_chars(c[None], d), d)
            eps = float(T.infidelity_from_polarization(lam, d)[0])
            crossed = crossed or eps >= (d - 1) / d * (1 - f ** m)
            half = half or float(A[0]) < 0.5
            if crossed and half and m >= max(GAIN_DEPTHS):
                break
    chi = np.array(chi)
    A, _, lam = T.accept_and_polarization(np.clip(T.dist_from_chars(chi, d), 0.0, None), d)
    eps = T.infidelity_from_polarization(lam, d)
    ms = np.arange(len(chi))
    return ms, eps, A, (d - 1) / d * (1 - f ** ms)


def first_below(ms, y, level):
    """Depth (linear interpolation) where the decreasing curve y first falls below level; nan if never,
    0 if it is already below at depth 1."""
    idx = np.nonzero(y[1:] < level)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0] + 1
    if i == 1:
        return 0.0
    y0, y1 = y[i - 1], y[i]
    return float(ms[i - 1] + (y0 - level) / (y0 - y1))


def main():
    with open(os.path.join(RES, "characterization.csv")) as fh:
        ch = list(csv.DictReader(fh))
    rows = []
    for key in T.RUNS:
        code, noise = key
        sm = S.StepModel(code, noise)
        d = sm.d
        for r in ch:
            if (r["code"], r["noise"]) != key or not r["rb_pair_official"] or r["rb_pair_official"] == "nan":
                continue
            p, r_rb = float(r["p"]), float(r["rb_pair_official"])
            ms, eps, A, phys = curves(sm, p, r_rb)
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = phys / eps
            gain[0] = np.inf
            m_break = first_below(ms, gain, 1.0)
            gt = sm.ground_truth(p)
            row = {"code": code, "noise": noise, "p": p, "d": d,
                   "rb_pair": r_rb, "eps_pair": float(eps[1]), "discard_pair": float(1 - A[1]),
                   "eps_single": gt["single_stat"], "rb_single": r_rb / 2,
                   "gain_single": r_rb / 2 / gt["single_stat"],
                   "m_gain10": first_below(ms, gain, 10.0), "m_gain2": first_below(ms, gain, 2.0),
                   "m_break": m_break, "m_accept_half": first_below(ms, A, 0.5),
                   "accept_at_break": float(np.interp(m_break, ms, A)) if m_break == m_break else np.nan,
                   "eps_at_break": float(np.interp(m_break, ms, eps)) if m_break == m_break else np.nan}
            for g in GAIN_DEPTHS:
                row[f"gain_m{g}"] = float(gain[g]) if g < len(ms) else np.nan
                row[f"accept_m{g}"] = float(A[g]) if g < len(ms) else np.nan
            rows.append(row)
    with open(os.path.join(RES, "logical_vs_physical.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("| code | noise | p | physical, 1 Clifford | logical, 1 Clifford + check | gain | gain at 1 / 5 / 10 / 22 pairs | 10x better up to | break-even depth | accepted there | half the shots rejected at |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["p"] not in (0.001, 0.002, 0.005, 0.01):
            continue
        m10 = "-" if not r["m_gain10"] > 0 else f"{r['m_gain10']:.0f}"
        print(f"| {r['code']} | {r['noise'].upper()} | {r['p']:g} | {r['rb_single']:.1e} | {r['eps_single']:.1e} | "
              f"{r['gain_single']:.0f} | {r['gain_m1']:.0f} / {r['gain_m5']:.0f} / {r['gain_m10']:.1f} / {r['gain_m22']:.1f} | "
              f"{m10} | {r['m_break']:.0f} | {100 * r['accept_at_break']:.0f}% | {r['m_accept_half']:.0f} |")
    figure(rows)


def figure(rows):
    import matplotlib.pyplot as plt
    from figures import BLUE, RAMP, INK, INK2, SURFACE, NAMES, save

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), NAMES):
        rs = sorted([r for r in rows if (r["code"], r["noise"]) == key], key=lambda r: r["p"])
        ps = np.array([r["p"] for r in rs])
        mb = np.array([r["m_break"] for r in rs])
        m10 = np.array([r["m_gain10"] if r["m_gain10"] > 0 else np.nan for r in rs])
        mh = np.array([r["m_accept_half"] if r["m_accept_half"] >= 1 else np.nan for r in rs])
        ok = mb >= 1
        ax.fill_between(ps[ok], 0.5, mb[ok], color=RAMP[0], alpha=0.28, lw=0)
        ax.plot(ps[ok], mb[ok], color=RAMP[4], marker="o", mec=SURFACE, mew=0.8, ms=4.5,
                label="break-even: logical = physical")
        ok10 = m10 >= 1
        ax.plot(ps[ok10], m10[ok10], color=BLUE, marker="s", mec=SURFACE, mew=0.8, ms=4.2,
                label="logical 10 times better")
        ax.plot(ps, mh, color=INK2, ls=(0, (4, 2)), lw=1.2, label="half the shots rejected")
        ax.axhline(22, color=INK2, lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 2e4)
        ax.set_title(NAMES[key], loc="left")
        ax.text(1.15e-4, 1.6, "logical better", color=INK, fontsize=8, va="center")
        ax.text(0.05, 6e3, "physical better", color=INK, fontsize=8, va="center", ha="right")
    axes[0, 1].text(0.05, 26, "deepest production depth, 22", color=INK2, fontsize=7, va="bottom", ha="right")
    for ax in axes[1]:
        ax.set_xlabel("two-qudit gate error p")
    for ax in axes[:, 0]:
        ax.set_ylabel("Clifford pairs before the terminal check")
    axes[0, 1].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.suptitle("Where terminal-check logical gates beat the physical qudit",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig8_when_logical_wins.png")


if __name__ == "__main__":
    main()
