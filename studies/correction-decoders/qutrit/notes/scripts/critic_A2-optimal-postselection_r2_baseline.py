#!/usr/bin/env python3
"""Critic (round 2) of A2-optimal-postselection: baseline-CSV claims.

C14  const 1 acceptance saturation at 3^-4 (p=0.01): per-depth z-scores.
C13  legacy-metric Rayleigh floor (p=0.01, const 1): per-depth z-scores of
     the 30-sequence mean and of the per-sequence SD; residual signal needed
     at m=14 (Rice mean).
C16  unweighted log-linear fits (same depth selection as A2: >= 1000 accepted
     shots), max log-residuals, and residual z-scores using the CSV
     between-sequence standard deviations (SE = std/sqrt(30)).

Reads the baseline run folder read-only. Seconds of CPU.
Run: module load python/3.11 && python critic_A2-optimal-postselection_r2_baseline.py
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import i0e, i1e
from scipy.optimize import brentq

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
       "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEP = np.array([int(x) for x in open(f"{RUN}/depths.txt").read().split(",")])
PROBS = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]
SHOTS, NSEQ = 1_000_000, 30


def rows(p_idx, kind, k):
    out = {}
    with open(f"{RUN}/results/LRB/{p_idx}/{kind}_check_data/{k}.csv") as fh:
        for line in fh:
            key, *v = line.strip().split(",")
            out[key] = np.array([float(x) for x in v])
    return out


def rice_mean(nu, sigma):
    """E|nu + sigma (X + iY)|, X, Y ~ N(0,1) (Rice distribution)."""
    x = nu ** 2 / (4 * sigma ** 2)
    return sigma * math.sqrt(math.pi / 2) * (
        (1 + 2 * x) * i0e(x) + 2 * x * i1e(x))


def c14_c13():
    r = rows(8, "const", 1)
    acc = 1 - r["Rejected Runs"]
    se_acc = r["Rejected Standard Deviations"] / math.sqrt(NSEQ)
    print("C14: p=0.01 const 1 acceptance vs 3^-4 = %.7f" % 3 ** -4)
    for m, a, s in zip(DEP, acc, se_acc):
        if m >= 10:
            print(f"   m={m:<3d} acc={a:.7f}  SE={s:.1e}  z={(a - 3 ** -4) / s:+.2f}")
    print("C13: legacy floor (Rayleigh) vs CSV, p=0.01 const 1")
    for i, m in enumerate(DEP):
        if m < 10:
            continue
        N = acc[i] * SHOTS
        sig = math.sqrt(1 / (2 * N))           # per-component SD of the mean
        floor = rice_mean(0.0, sig)
        sd_r = sig * math.sqrt((4 - math.pi) / 2)
        F, Fs = r["Fidelity averages"][i], r["Fidelity Standard Deviations"][i]
        z = (F - floor) / (sd_r / math.sqrt(NSEQ))
        msg = ""
        if z > 2:
            nu = brentq(lambda v: rice_mean(v, sig) - F, 0, 1)
            msg = f"; Rice mean = CSV needs true |lambda| ~ {nu:.4f}"
        print(f"   m={m:<3d} N_acc={N:7.0f} CSV mean={F:.4f} floor={floor:.4f} "
              f"z={z:+.2f}; CSV per-seq SD={Fs:.4f} vs Rayleigh SD "
              f"{sd_r:.4f}{msg}")


def c16():
    print("C16: unweighted log-linear fits (depths with >= 1000 accepted)")
    for p_idx in (4, 5):
        for kind, k in (("unif", 1), ("unif", 2), ("const", 1)):
            r = rows(p_idx, kind, k)
            acc = 1 - r["Rejected Runs"]
            lam = r["Fidelity averages"]
            okm = (acc * SHOTS >= 1000) & np.isfinite(lam) & (lam > 0)
            m = DEP[okm].astype(float)
            out = []
            for y, s in ((acc[okm], r["Rejected Standard Deviations"][okm]),
                         (lam[okm], r["Fidelity Standard Deviations"][okm])):
                sl, ic = np.polyfit(m, np.log(y), 1)
                res = np.log(y) - (ic + sl * m)
                se = s / math.sqrt(NSEQ) / y          # SE of log(mean)
                se = np.where(se > 0, se, np.nan)
                chi2 = np.nansum((res / se) ** 2)
                dof = int(np.isfinite(se).sum()) - 2
                out.append((math.exp(sl), np.abs(res).max(),
                            np.nanmax(np.abs(res / se)), chi2, dof))
            (a, ra, za, ca, da), (f, rl, zl, cl, dl) = out
            print(f"   p={PROBS[p_idx]:g} {kind}{k}: a={a:.5f} (1-a={1 - a:.3f},"
                  f" -ln a={-math.log(a):.3f}) max|res| acc {ra:.1e} "
                  f"(max|z| {za:.1f}, chi2/dof {ca:.0f}/{da}); f={f:.7f} "
                  f"r_c={2 / 3 * (1 - f):.2e} max|res| lam {rl:.1e} "
                  f"(max|z| {zl:.1f}, chi2/dof {cl:.0f}/{dl})")


def c16_detail():
    print("C16 detail: unif 1, per-depth log-residuals of the acceptance fit "
          "and fits on m>=2 only; weighted fit of lam with SE of 1-f")
    for p_idx in (4, 5):
        r = rows(p_idx, "unif", 1)
        acc = 1 - r["Rejected Runs"]
        se = r["Rejected Standard Deviations"] / math.sqrt(NSEQ) / acc
        m = DEP.astype(float)
        sl, ic = np.polyfit(m, np.log(acc), 1)
        res = np.log(acc) - (ic + sl * m)
        print(f"   p={PROBS[p_idx]:g} all depths: z per depth "
              + " ".join(f"{z:+.1f}" for z in res / se))
        k = m >= 2
        sl2, ic2 = np.polyfit(m[k], np.log(acc[k]), 1)
        res2 = np.log(acc[k]) - (ic2 + sl2 * m[k])
        print(f"   p={PROBS[p_idx]:g} m>=2 only: a={math.exp(sl2):.5f} "
              f"max|res|={np.abs(res2).max():.1e} chi2/dof="
              f"{np.sum((res2 / se[k]) ** 2):.0f}/{k.sum() - 2}")
        lam = r["Fidelity averages"]
        s = r["Fidelity Standard Deviations"] / math.sqrt(NSEQ) / lam
        wts = 1 / s
        X = np.vstack([np.ones_like(m), m]).T
        Wm = np.diag(wts ** 2)
        cov = np.linalg.inv(X.T @ Wm @ X)
        beta = cov @ X.T @ Wm @ np.log(lam)
        res = np.log(lam) - X @ beta
        chi2 = float(np.sum((res / s) ** 2))
        scale = max(1.0, chi2 / (len(m) - 2))
        se_slope = math.sqrt(cov[1, 1] * scale)
        print(f"   p={PROBS[p_idx]:g} lam weighted fit: 1-f = "
              f"{-beta[1]:.3e} +/- {se_slope:.1e} (chi2/dof {chi2:.1f}/"
              f"{len(m) - 2}, SE scaled by sqrt(chi2/dof)); r_c = "
              f"{2 / 3 * (-beta[1]):.2e}; unweighted 1-f = "
              f"{-np.polyfit(m, np.log(lam), 1)[0]:.3e}")
    # repository fit table, if present
    import csv
    path = f"{RUN}/results/plots/unif_lrb_vs_rb_table_all_mixed_fits.csv"
    try:
        with open(path) as fh:
            rdr = list(csv.DictReader(fh))
        for row in rdr:
            if row["check_num"] == "1" and row["p_raw"] in ("0.001", "0.002"):
                print(f"   repository weighted fit (unif 1, p={row['p_raw']}): "
                      f"lrb_f={float(row['lrb_f']):.7f} lrb_b="
                      f"{float(row['lrb_b']):.5f} lrb_error_rate="
                      f"{float(row['lrb_error_rate']):.2e} (boot SD "
                      f"{float(row['lrb_error_rate_std_boot']):.1e})")
    except (OSError, KeyError) as exc:
        print(f"   (repository fit table not usable: {exc})")


def c19_variance():
    print("C19 caveat: per-sequence SD of the legacy fidelity vs shot-noise "
          "SD sqrt(V(lam)/N_acc)")
    for p_idx, kind, k in ((4, "unif", 1), (4, "const", 1), (8, "const", 1),
                           (8, "unif", 1)):
        r = rows(p_idx, kind, k)
        acc = 1 - r["Rejected Runs"]
        lam = r["Fidelity averages"]
        out = []
        for i, m in enumerate(DEP):
            N = acc[i] * SHOTS
            if N < 1000 or not np.isfinite(lam[i]):
                continue
            V = (1 + lam[i]) / 2 - lam[i] ** 2
            out.append(f"m={m}:{r['Fidelity Standard Deviations'][i] / math.sqrt(V / N):.2f}")
        print(f"   p={PROBS[p_idx]:g} {kind}{k}: ratio " + " ".join(out))


if __name__ == "__main__":
    c14_c13()
    c16()
    c16_detail()
    c19_variance()
