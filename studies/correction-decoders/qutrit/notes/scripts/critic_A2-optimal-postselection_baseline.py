#!/usr/bin/env python3
"""Critic checks of A2 claims that read the baseline CSVs: C13, C14, C16, C19.

Independent code (reads the read-only baseline run folder, writes nothing).

  - C16: unweighted log-linear fits of acceptance and of the legacy fidelity
    (= lam for detection policies) vs m; residuals compared with the
    statistical scatter actually present in the CSV (across-sequence SD).
  - between-sequence vs shot-noise variance of lam (matters for C19's shot
    cost, which counts shot noise only).
  - C14: const 1 acceptance at p=0.01, m>=14 vs 3^-4 with its standard error.
  - C13: legacy floor; per-circuit SD of the CSV vs the Rayleigh SD.
  - C19: N*(eps=0.1) with B known (A2's formula) and the eta definitions.

Run: module load python/3.11 && python critic_A2-optimal-postselection_baseline.py
"""
import math

import numpy as np

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
SHOTS = 1_000_000
NSEQ = 30
depths = np.array([int(x) for x in open(f"{RUN}/depths.txt").read().split(",")], float)
probs = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]


def read_csv(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            rows[parts[0]] = np.array([float(x) for x in parts[1:]])
    return rows


def V(lam):
    return (1 + lam) / 2 - lam ** 2


print("[C16] fits (unweighted, depths with >=1000 accepted shots), and noise")
for p_idx in (4, 5, 8):
    for kind, k in (("unif", 1), ("unif", 2), ("const", 1), ("const", 2)):
        r = read_csv(f"{RUN}/results/LRB/{p_idx}/{kind}_check_data/{k}.csv")
        lam, acc = r["Fidelity averages"], 1 - r["Rejected Runs"]
        sd_lam, sd_rej = r["Fidelity Standard Deviations"], r["Rejected Standard Deviations"]
        ok = (acc * SHOTS >= 1000) & np.isfinite(lam) & (lam > 0)
        ms = depths[ok]
        sa, ia = np.polyfit(ms, np.log(acc[ok]), 1)
        sl, il = np.polyfit(ms, np.log(lam[ok]), 1)
        ra = np.log(acc[ok]) - (ia + sa * ms)
        rl = np.log(lam[ok]) - (il + sl * ms)
        a, f = math.exp(sa), math.exp(sl)
        # standard errors of the 30-sequence means (empirical across-seq SD)
        se_la = sd_lam[ok] / math.sqrt(NSEQ) / lam[ok]
        se_ac = sd_rej[ok] / math.sqrt(NSEQ) / acc[ok]
        shot_sd = np.sqrt(V(lam[ok]) / (acc[ok] * SHOTS))
        ratio = sd_lam[ok] / shot_sd
        print(f"  p={probs[p_idx]:<6g} {kind}{k}: a={a:.5f} (1-a={1 - a:.4f}, -ln a={-sa:.4f})"
              f" f={f:.7f} r_c={2 / 3 * (1 - f):.2e} B={math.exp(il):.5f}"
              f" | max|log resid| acc {np.abs(ra).max():.1e} (SE {se_ac.max():.1e}),"
              f" lam {np.abs(rl).max():.1e} (SE {se_la.max():.1e})"
              f" | chi2/dof lam {np.sum((rl / se_la) ** 2) / max(ok.sum() - 2, 1):.1f}"
              f" | seqSD/shotSD of lam: {ratio.min():.2f}-{ratio.max():.2f}")

print("\n[C14] const 1 acceptance at p=0.01 vs 3^-4 = %.7f" % 3 ** -4)
r = read_csv(f"{RUN}/results/LRB/8/const_check_data/1.csv")
for i, m in enumerate(depths):
    acc = 1 - r["Rejected Runs"][i]
    se = r["Rejected Standard Deviations"][i] / math.sqrt(NSEQ)
    binom = math.sqrt(acc * (1 - acc) / (SHOTS * NSEQ))
    print(f"  m={int(m):2d}: acceptance {acc:.6f}  (acc-3^-4)/SE = "
          f"{(acc - 3 ** -4) / se:+6.1f}  [SE across seq {se:.1e}, binomial {binom:.1e}]")

print("\n[C13] legacy floor at p=0.01 const 1")
for i, m in enumerate(depths):
    if m < 10:
        continue
    acc = 1 - r["Rejected Runs"][i]
    N = acc * SHOTS
    sig = math.sqrt(1 / (2 * N))
    print(f"  m={int(m):2d}: CSV mean {r['Fidelity averages'][i]:.4f} (floor "
          f"{sig * math.sqrt(math.pi / 2):.4f}, 30-mean SE {sig * math.sqrt((4 - math.pi) / 2) / math.sqrt(NSEQ):.4f});"
          f" CSV per-seq SD {r['Fidelity Standard Deviations'][i]:.4f} vs Rayleigh SD"
          f" {sig * math.sqrt((4 - math.pi) / 2):.4f}")

print("\n[C19] single-depth CR shot cost (B known), unif 1")


def Nstar(A, a, B, f, eps=0.1):
    A, B = min(A, 1), min(B, 1)
    best = (math.inf, 0)
    for m in range(1, 2000):
        lam = B * f ** m
        n = V(lam) * f ** 2 / (A * a ** m * m ** 2 * lam ** 2 * eps ** 2 * (1 - f) ** 2)
        best = min(best, (n, m))
    return best


for p_idx in (4, 5):
    r = read_csv(f"{RUN}/results/LRB/{p_idx}/unif_check_data/1.csv")
    lam, acc = r["Fidelity averages"], 1 - r["Rejected Runs"]
    sa, ia = np.polyfit(depths, np.log(acc), 1)
    sl, il = np.polyfit(depths, np.log(lam), 1)
    A, a, B, f = math.exp(ia), math.exp(sa), math.exp(il), math.exp(sl)
    n, m = Nstar(A, a, B, f)
    eta, dl = -sa, -sl
    approx = 3 * math.e ** 2 / 8 * (1 - B) * (eta / dl) ** 2 / 0.01
    approx1 = 3 * math.e ** 2 / 8 * (1 - B) * ((1 - a) / (1 - f)) ** 2 / 0.01
    print(f"  p={probs[p_idx]:g}: A={A:.4f} a={a:.5f} B={B:.5f} f={f:.7f};"
          f" N*={n:.2e} at m*={m}; offset formula with eta=-ln a: {approx:.2e},"
          f" with eta=1-a: {approx1:.2e}; 4(1+eta/delta)/eps^2 = {4 * (1 + eta / dl) / 0.01:.2e}")
    # statistical uncertainty of f from a weighted fit using the CSV SDs
    se = r["Fidelity Standard Deviations"] / math.sqrt(NSEQ) / lam
    wfit = np.polyfit(depths, np.log(lam), 1, w=1 / se, cov="unscaled")
    print(f"     weighted fit: 1-f = {1 - math.exp(wfit[0][0]):.2e} +- "
          f"{math.sqrt(wfit[1][0, 0]):.1e} (from the 30M shots per depth actually used)")
