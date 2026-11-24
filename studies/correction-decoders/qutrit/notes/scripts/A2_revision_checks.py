#!/usr/bin/env python3
"""A2 revision: checks made while resolving the critique of
notes/A2-optimal-postselection.md (its Section 9, "Critique resolution").

Sections (one core each; times on the loaded login node):
  [V1] baseline CSVs (read-only): const 1 saturation at 3^-4 and the
       Rayleigh floor of the legacy metric with z-scores (C13, C14);
       unif 1 fits: per-depth acceptance residuals, both eta conventions,
       weighted fit of lam with an error bar, the repository fit table
       (C16); across-sequence versus shot-noise spread; shot cost with B
       known or fitted from an m=0 run, offset-formula variants (C19).
  [V2] theory: Re E w^R as a function of p0 for d=2..7 (C7); covariant
       rules under symmetric noise: the MAP rule maximises Fleg among
       covariant rules iff S* + S_min >= 2/3 (Sec. 3.5, C10); Example B
       made covariant (C11); loss bound at fixed alpha<1 by exact vertex
       enumeration on positive-loss cases (C12); unchecked inverse half and
       hidden syndrome state (C15); Forney's ratio with unequal priors
       (C3); mean of per-circuit ratios versus a common threshold.
  [V3] asymmetric noise: Pauli twirl of qutrit amplitude damping; code
       capacity [[5,1,2]]_3 with that noise in the computational and in the
       Fourier frame: conjugation symmetry, Fleg versus lam.
  [V4] circuit level, m=2 baseline circuits: non-tautological symmetry
       check (exact law from complex per-location characteristic functions
       built from the sampler's fault laws, with a power test; Monte Carlo
       with the repository sampler); model-mismatch regret (decoder built
       at the wrong p, for the wrong sequence, or from a sequence-averaged
       law).
  [V5] exact D=0 statistics at m=4 and m=6 from the critic's hidden-state
       contraction (critic_A2-optimal-postselection_r2_chain.py, imported
       read-only) for all 30 sequences, compared with the baseline CSV.

Run:  module load python/3.11 && python A2_revision_checks.py [--only V1 ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a2 = _load("a2", "A2_optimal_rule_check.py")   # also limits BLAS to 1 thread
W, RUN, WORKTREE = a2.W, a2.RUN, a2.WORKTREE
NSEQ, SHOTS = 30, 1_000_000
DEPTHS = [int(x) for x in open(f"{RUN}/depths.txt").read().split(",")]
PROBS = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]


def csv_rows(p_idx, kind, k):
    out = {}
    with open(f"{RUN}/results/LRB/{p_idx}/{kind}_check_data/{k}.csv") as fh:
        for line in fh:
            key, *v = line.strip().split(",")
            out[key] = np.array([float(x) for x in v])
    return out


def V(lam):
    """Variance of Re w^R per accepted shot."""
    return (1 + lam) / 2 - lam ** 2


# ======================================================================= V1
def rice_mean(nu, sig):
    from scipy.special import i0e, i1e
    y = nu ** 2 / (4 * sig ** 2)
    return sig * math.sqrt(math.pi / 2) * ((1 + 2 * y) * i0e(y)
                                           + 2 * y * i1e(y))


def wfit_log(m, y, se):
    """Weighted LS of log y = c + s m with SE(log y) = se / y. Returns
    (c, s, SE(s) scaled by sqrt(chi2/dof) when that exceeds 1, chi2, dof)."""
    s_log = se / y
    X = np.vstack([np.ones_like(m), m]).T
    w = 1 / s_log ** 2
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * np.log(y)))
    res = np.log(y) - X @ beta
    chi2, dof = float(np.sum(w * res ** 2)), len(m) - 2
    return beta[0], beta[1], math.sqrt(cov[1, 1] * max(1.0, chi2 / dof)), \
        chi2, dof


def n_single(A, a, B, f, eps, ms):
    """A2 single-depth Cramer-Rao count (B known), per depth."""
    lam = B * f ** ms
    c = V(lam) / (A * a ** ms * lam ** 2)
    return c * f ** 2 / (ms ** 2 * eps ** 2 * (1 - f) ** 2)


def n_two_depth(A, a, B, f, eps, ms):
    """B unknown: depths 0 and m1 with the optimal shot split."""
    c0 = V(B) / (A * B ** 2)
    lam = B * f ** ms
    c1 = V(lam) / (A * a ** ms * lam ** 2)
    return (np.sqrt(c0) + np.sqrt(c1)) ** 2 * f ** 2 / (
        ms ** 2 * eps ** 2 * (1 - f) ** 2)


def section_V1():
    print("\n[V1] baseline CSVs (read-only)")
    dep = np.array(DEPTHS, float)
    r = csv_rows(8, "const", 1)
    acc = 1 - r["Rejected Runs"]
    se_acc = r["Rejected Standard Deviations"] / math.sqrt(NSEQ)
    q81 = 3.0 ** -4
    se_bin = math.sqrt(q81 * (1 - q81) / (NSEQ * SHOTS))
    print("  C14/C13: p=0.01 const 1. Acceptance vs 3^-4 = %.7f (z with the "
          "CSV SE std/sqrt(30), and with the binomial SE %.1e);" % (q81, se_bin))
    print("  legacy metric vs the Rayleigh mean sqrt(pi/(4 N_acc)) (z with the "
          "Rayleigh SD/sqrt(30)); Rice inversion where z > 2")
    print("    m   acceptance  z_csv   z_bin   N_acc   CSV Fleg  floor   "
          "z_floor  SDseq/SDrayl  |lam| (Rice)")
    for i, m in enumerate(DEPTHS):
        if m < 10:
            continue
        N = acc[i] * SHOTS
        sig = math.sqrt(1 / (2 * N))
        floor = sig * math.sqrt(math.pi / 2)
        sd_r = sig * math.sqrt((4 - math.pi) / 2)
        F = r["Fidelity averages"][i]
        zf = (F - floor) / (sd_r / math.sqrt(NSEQ))
        nu = ""
        if zf > 2:
            from scipy.optimize import brentq
            nu = "%.4f" % brentq(lambda v: rice_mean(v, sig) - F, 0, 1)
        print(f"    {m:<3d} {acc[i]:.7f}  {(acc[i] - q81) / se_acc[i]:+6.2f}  "
              f"{(acc[i] - q81) / se_bin:+6.2f}  {N:6.0f}  {F:.4f}    "
              f"{floor:.4f}  {zf:+6.2f}   "
              f"{r['Fidelity Standard Deviations'][i] / sd_r:5.2f}         {nu}")

    print("  C16: unif 1 fits")
    import csv
    with open(f"{RUN}/results/plots/unif_lrb_vs_rb_table_all_mixed_fits.csv") as fh:
        table = list(csv.DictReader(fh))
    fits = {}
    for p_idx in (4, 5):
        r = csv_rows(p_idx, "unif", 1)
        acc = 1 - r["Rejected Runs"]
        lam = r["Fidelity averages"]
        sa = r["Rejected Standard Deviations"] / math.sqrt(NSEQ)
        sl = r["Fidelity Standard Deviations"] / math.sqrt(NSEQ)
        A, a = a2.fit_exp(dep, acc)
        B, f = a2.fit_exp(dep, lam)
        z = (np.log(acc) - np.log(A * a ** dep)) / (sa / acc)
        k = dep >= 2
        A2_, a2_ = a2.fit_exp(dep[k], acc[k])
        z2 = (np.log(acc[k]) - np.log(A2_ * a2_ ** dep[k])) / (sa[k] / acc[k])
        cw, sw, sew, chi2w, dofw = wfit_log(dep, lam, sl)
        zl = (np.log(lam) - np.log(B * f ** dep)) / (sl / lam)
        row = [t for t in table if t["check_num"] == "1"
               and t["p_raw"] == str(PROBS[p_idx])][0]
        fits[p_idx] = dict(A=A, a=a, B=B, f=f, fw=math.exp(sw), Bw=math.exp(cw))
        print(f"    p={PROBS[p_idx]:g}: unweighted (A2): a={a:.5f}, 1-a={1 - a:.4f},"
              f" -ln a={-math.log(a):.4f}; f={f:.7f}, r_c={2 / 3 * (1 - f):.3e}")
        print("      acceptance residual z per depth (m=" + ",".join(
            str(x) for x in DEPTHS) + "): " + " ".join(f"{x:+.1f}" for x in z))
        print(f"      acceptance fit on m>=2 only: a={a2_:.5f}, chi2="
              f"{np.sum(z2 ** 2):.1f} for {k.sum() - 2} dof, max|log res|="
              f"{np.abs(np.log(acc[k]) - np.log(A2_ * a2_ ** dep[k])).max():.1e}")
        print(f"      lam: unweighted-fit chi2 {np.sum(zl ** 2):.1f}/7; weighted "
              f"fit 1-f = {-sw:.3e} +/- {sew:.1e} (chi2 {chi2w:.1f}/{dofw}, SE "
              f"scaled), r_c = {2 / 3 * (-sw):.3e} +/- {2 / 3 * sew:.1e}")
        print(f"      repository table: lrb_f={float(row['lrb_f']):.7f}, r_c="
              f"{float(row['lrb_error_rate']):.3e} (bootstrap SD "
              f"{float(row['lrb_error_rate_std_boot']):.1e})")

    print("  C19: across-sequence SD / shot-noise SD sqrt(V(lam)/N_acc) per depth")
    for p_idx, kind, k in ((4, "unif", 1), (4, "const", 1), (8, "const", 1)):
        r = csv_rows(p_idx, kind, k)
        acc = 1 - r["Rejected Runs"]
        lam = r["Fidelity averages"]
        out = []
        for i, m in enumerate(DEPTHS):
            N = acc[i] * SHOTS
            if N >= 1000 and np.isfinite(lam[i]) and lam[i] > 0.05:
                out.append(f"{m}:{r['Fidelity Standard Deviations'][i] / math.sqrt(V(lam[i]) / N):.2f}")
        print(f"    p={PROBS[p_idx]:g} {kind} {k}: " + " ".join(out))
    F = fits[4]
    ms = np.arange(1, 400, dtype=float)
    eps = 0.1
    A, a, B, f = min(F["A"], 1), F["a"], min(F["B"], 1), F["f"]
    n1 = n_single(A, a, B, f, eps, ms)
    n2 = n_two_depth(A, a, B, f, eps, ms)
    i1, i2 = int(np.argmin(n1)), int(np.argmin(n2))
    print(f"  C19: unif 1, p=1e-3, eps=0.1, unweighted fit (A={A:.4f}, a={a:.5f},"
          f" B={B:.5f}, f={f:.7f}):")
    print(f"    B known, single depth: N*={n1[i1]:.3e} at m*={ms[i1]:.0f}; "
          f"B fitted, depths 0 and m1 with optimal split: N*={n2[i2]:.3e} at "
          f"m1={ms[i2]:.0f} (ratio {n2[i2] / n1[i1]:.2f})")
    fw = F["fw"]
    n1w = n_single(A, a, B, fw, eps, ms)
    n2w = n_two_depth(A, a, B, fw, eps, ms)
    print(f"    with the weighted f={fw:.7f}: B known {n1w.min():.3e} "
          f"(m*={ms[np.argmin(n1w)]:.0f}); B fitted {n2w.min():.3e} "
          f"(m1={ms[np.argmin(n2w)]:.0f})")
    for lab, eta in (("-ln a", -math.log(a)), ("1-a", 1 - a)):
        for labd, dl in (("unweighted", -math.log(f)), ("weighted", -math.log(fw))):
            approx = 3 * math.e ** 2 / 8 * (1 - B) * (eta / dl) ** 2 / eps ** 2
            print(f"    offset formula (3e^2/8)(1-B)(eta/delta)^2/eps^2 with "
                  f"eta={lab} ({eta:.4f}), delta {labd} ({dl:.3e}): {approx:.3e}")


# ======================================================================= V2
def frontier_S(J, alpha):
    fr = a2.frontier(J)
    return a2.beta_star(fr, alpha) / alpha


def resid(J, cmap, D0mix=False):
    """Accepted residual distribution, alpha=1, deterministic corrections
    cmap[D]; D0mix: D=0 (row 0) uses c=1 and c=2 with weight 1/2 each."""
    pr = np.zeros(3)
    for D in range(len(J)):
        cs = ((1, 0.5), (2, 0.5)) if (D == 0 and D0mix) else ((cmap[D], 1.0),)
        for c, w in cs:
            for l in range(3):
                pr[(l - c) % 3] += w * J[D, l]
    return pr


def sym_table(rng, npairs):
    """Rows: 0 = self-conjugate D=0 (posterior (x0, y, y)); then pairs
    (D_j, -D_j) with posteriors x and x[[0, 2, 1]] and equal mass."""
    mass = rng.dirichlet(np.ones(npairs + 1))
    x = rng.dirichlet([1, 1, 1])
    rows = [mass[0] * np.array([x[0], (x[1] + x[2]) / 2, (x[1] + x[2]) / 2])]
    for j in range(npairs):
        x = rng.dirichlet([1, 1, 1])
        rows += [mass[j + 1] / 2 * x, mass[j + 1] / 2 * x[[0, 2, 1]]]
    return np.array(rows)


def vertices_fixed_alpha(J, alpha):
    """All vertices of {delta: alpha(delta) = alpha}: every history rejected
    or accepted with one correction, except one history D* accepted with
    fraction theta and correction c*. Returns arrays (|z|/alpha, S)."""
    n = len(J)
    pD = J.sum(1)
    zc = np.array([[sum(J[D, l] * W ** ((l - c) % 3) for l in range(3))
                    for c in range(3)] for D in range(n)])
    bc = J.copy()
    Fs, Ss = [], []
    for Ds in range(n):
        oth = [d for d in range(n) if d != Ds]
        assign = np.array(list(itertools.product(range(4), repeat=n - 1)))
        accm = assign > 0
        cidx = np.clip(assign - 1, 0, 2)
        A_o = (accm * pD[oth][None, :]).sum(1)
        z_o = (accm * zc[oth][np.arange(n - 1)[None, :], cidx]).sum(1)
        b_o = (accm * bc[oth][np.arange(n - 1)[None, :], cidx]).sum(1)
        th = (alpha - A_o) / pD[Ds]
        ok = (th >= -1e-12) & (th <= 1 + 1e-12)
        th = np.clip(th, 0, 1)
        for cs in range(3):
            z = z_o + th * zc[Ds, cs]
            b = b_o + th * bc[Ds, cs]
            Fs.append(np.abs(z[ok]) / alpha)
            Ss.append(b[ok] / alpha)
    return np.concatenate(Fs), np.concatenate(Ss)


def section_V2(rng):
    print("\n[V2] theory checks")
    # (a) C7
    for d in range(2, 8):
        vals = {round(math.cos(2 * math.pi * r / d), 12) for r in range(1, d)}
        print(f"  C7: d={d}: distinct cos(2 pi r/d), r=1..d-1: {len(vals)} -> "
              f"Re E w^R is a function of p0 alone: {len(vals) == 1}")
    # (b) covariant rules, symmetric tables, alpha = 1
    n_tab, npairs = 3000, 3
    fails, agree, ident = 0, 0, 0
    for _ in range(n_tab):
        J = sym_table(rng, npairs)
        res = []
        for cs in itertools.product(range(3), repeat=npairs):
            cmap = np.zeros(len(J), dtype=int)
            for j, c in enumerate(cs):
                cmap[1 + 2 * j], cmap[2 + 2 * j] = c, (-c) % 3
            for mix in (False, True):
                m = a2.metrics(resid(J, cmap, mix))
                res.append((m["S"], m["lam"], m["Fleg"], abs(m["im"])))
        res = np.array(res)
        Sstar, Smin = res[:, 0].max(), res[:, 0].min()
        lam_star, lam_min = res[:, 1].max(), res[:, 1].min()
        Fmax = res[:, 2].max()
        ident += abs(Fmax - max(lam_star, -lam_min)) < 1e-12 and res[:, 3].max() < 1e-12
        map_opt = abs(lam_star) >= Fmax - 1e-12        # MAP: F = |lam*|
        cond = Sstar + Smin >= 2 / 3 - 1e-12
        agree += map_opt == cond
        fails += not map_opt
    print(f"  C10/Sec. 3.5: {n_tab} symmetric tables (D=0 + {npairs} mirror pairs, "
          f"alpha=1, all covariant maps incl. the 1/2-1/2 mix at D=0):")
    print(f"    Im z = 0 and max Fleg = max(lam*, -lam_min) in {ident}/{n_tab}; "
          f"MAP is Fleg-optimal among covariant maps iff S*+S_min>=2/3 in "
          f"{agree}/{n_tab}; MAP not Fleg-optimal in {fails} tables")
    # (c) Example B made covariant
    pr = 0.5 * np.array([0.65, 0.35, 0.0]) + 0.5 * np.array([0.65, 0.0, 0.35])
    m = a2.metrics(pr)
    print(f"  C11: Example B, accept D and -D (covariant): S={m['S']:.3f} "
          f"lam={m['lam']:.4f} Fleg={m['Fleg']:.4f} (vs 0.55 for the first "
          "history): no reversal")
    # (d) C12 loss bound at fixed alpha < 1
    n_tab, n = 400, 4
    pos, viol, ratio, nlam = 0, 0, 0.0, 0
    for _ in range(n_tab):
        pD = rng.dirichlet(np.ones(n))
        post = rng.dirichlet(np.ones(3), size=n)
        J = pD[:, None] * post
        alpha = rng.uniform(0.15, 1.0)
        Fv, Sv = vertices_fixed_alpha(J, alpha)
        FL = Fv.max()
        SL = Sv[Fv >= FL - 1e-12].max()
        Sstar = frontier_S(J, alpha)
        assert Sv.max() <= Sstar + 1e-12
        lamL = (3 * SL - 1) / 2
        loss = Sstar - SL
        if loss > 1e-12:
            pos += 1
            if lamL > 0:
                nlam += 1
                bound = (1 - SL) ** 2 / (4 * lamL)
                viol += loss > bound + 1e-12
                ratio = max(ratio, loss / bound)
    print(f"  C12: {n_tab} diffuse tables, {n} histories, alpha~U(0.15,1), exact "
          f"vertex enumeration: positive loss in {pos}, of which lam_L>0 in "
          f"{nlam}; bound violations {viol}; max loss/bound {ratio:.3f}")
    # (e) C15 unchecked inverse half; hidden syndrome state
    dep = np.array(DEPTHS, float)
    a, s = 0.9, 0.07
    phi = 1 - 1.5 * s
    step = np.array([1 - s, s / 2, s / 2])
    dist = np.array([1.0, 0, 0])
    exact = []
    for m in range(max(DEPTHS) + 1):
        if m in DEPTHS:
            exact.append(a ** m * dist[0])
        dist = np.array([sum(dist[j] * step[(i - j) % 3] for j in range(3))
                         for i in range(3)])
    exact = np.array(exact)
    formula = a ** dep * (1 + 2 * phi ** dep) / 3
    C, rr = a2.fit_exp(dep, exact)
    print(f"  C15: checked half (accept a={a}/step) + unchecked half (syndrome "
          f"trit increment nonzero w.p. {s}) + one final check: alpha(m) = "
          f"a^m (1+2 phi^m)/3, phi={phi:.3f}: max |exact-formula| = "
          f"{np.abs(exact - formula).max():.1e}; best single exponential: "
          f"max|log residual| = {np.abs(np.log(exact) - np.log(C * rr ** dep)).max():.2e}")
    for pdat, pmeas in ((0.05, 0.02), (0.01, 0.005)):
        stepd = np.array([1 - pdat, pdat / 2, pdat / 2])
        okm = np.array([1 - pmeas, pmeas / 2, pmeas / 2])  # P(e = -s') for s'
        M = np.array([[stepd[(t - s0) % 3] * okm[t] for t in range(3)]
                      for s0 in range(3)])
        vals = []
        for m in DEPTHS:
            v = np.zeros(3)
            v[0] = 1.0
            for _ in range(m + 1):          # m+1 checked rounds
                v = v @ M
            vals.append(v.sum())
        vals = np.array(vals)
        C, rr = a2.fit_exp(dep, vals)
        ev = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]
        print(f"  C15: every round checked, raw detectors, hidden syndrome trit, "
              f"p_data={pdat}, p_meas={pmeas}: eigenvalue ratio "
              f"{ev[1] / ev[0]:.3f}; single-exponential max|log residual| "
              f"{np.abs(np.log(vals) - np.log(C * rr ** dep)).max():.1e}")
    # (f) C3 Forney with unequal priors
    prior = np.array([0.98, 0.01, 0.01])
    like = np.array([0.02, 0.5, 0.001])
    joint = prior * like
    post = joint / joint.sum()
    ml = int(np.argmax(like))
    print(f"  C3: priors {prior.tolist()}, likelihoods {like.tolist()}: posterior "
          f"{np.round(post, 4).tolist()} (MAP class {int(post.argmax())}); "
          f"max-likelihood class {ml}, Forney ratio P(y|{ml})/sum_other = "
          f"{like[ml] / (like.sum() - like[ml]):.2f}; prior-weighted ratio for "
          f"class 0 = posterior odds {post[0] / (1 - post[0]):.2f}")
    # (g) mean of per-circuit ratios vs a common threshold
    def slope(fr, x):
        i = int(np.searchsorted(fr[0], x)) - 1
        i = min(max(i, 0), len(fr[0]) - 2)
        return (fr[1][i + 1] - fr[1][i]) / (fr[0][i + 1] - fr[0][i])
    print("  Remark 4: two circuits (400 histories each), mean acceptance fixed;"
          " objective = mean of per-circuit S (repository statistic)")
    for shapes, abar in ((((6, 2, 2), (6, 2, 2)), 0.5),
                         (((20, 2, 2), (3, 1, 1)), 0.5),
                         (((8, 1, 1), (4, 2, 2)), 0.6),
                         (((3, 1, 1), (3, 1, 1)), 0.4)):
        tabs = []
        for k, sh in enumerate(shapes):
            pD = rng.dirichlet(np.ones(400) * (0.3 if k == 0 else 3.0))
            tabs.append(pD[:, None] * rng.dirichlet(sh, size=400))
        frs = [a2.frontier(J) for J in tabs]
        a1 = np.linspace(max(0.0, 2 * abar - 1) + 1e-4,
                         min(1.0, 2 * abar) - 1e-4, 8001)
        a2v = 2 * abar - a1
        S1 = a2.beta_star(frs[0], a1) / a1
        S2 = a2.beta_star(frs[1], a2v) / a2v
        mS = (S1 + S2) / 2
        k = int(np.argmax(mS))
        pooled = np.vstack([0.5 * tabs[0], 0.5 * tabs[1]])
        pD = pooled.sum(1)
        q = pooled.max(1) / pD
        o = np.argsort(-q, kind="stable")
        t = q[o[int(np.searchsorted(np.cumsum(pD[o]), abar))]]
        acc1 = tabs[0].sum(1)[tabs[0].max(1) / tabs[0].sum(1) > t].sum()
        kc = int(np.argmin(np.abs(a1 - acc1)))
        g = [(slope(frs[0], a1[k]) - S1[k]) / a1[k],
             (slope(frs[1], a2v[k]) - S2[k]) / a2v[k]]
        pb = lambda i: (a2.beta_star(frs[0], a1[i]) + a2.beta_star(frs[1], a2v[i])) / 2
        print(f"    posteriors Dir{shapes[0]}/Dir{shapes[1]}, mean alpha {abar}: "
              f"common threshold t*={t:.4f} -> alphas ({a1[kc]:.3f}, {a2v[kc]:.3f}),"
              f" mean S {mS[kc]:.5f}, pooled beta {pb(kc):.5f}; max of mean S at "
              f"({a1[k]:.3f}, {a2v[k]:.3f}): mean S {mS[k]:.5f}, pooled beta "
              f"{pb(k):.5f}; (t_i-S_i)/alpha_i there: {g[0]:+.3f}, {g[1]:+.3f}")


# ======================================================================= V3
def weyl(a, b):
    X = np.roll(np.eye(3), 1, axis=0)                 # X|j> = |j+1>
    Z = np.diag(W ** np.arange(3))
    return np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)


def twirl(kraus):
    p = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            Wab = weyl(a, b)
            p[a, b] = sum(abs(np.trace(Wab.conj().T @ K)) ** 2 for K in kraus) / 9
    return p


def code_capacity_law(p1):
    """Exact P(D, l) of the [[5,1,2]]_3 code-capacity model (A2 section [C]
    conventions, Z+X checks) for an i.i.d. per-qutrit Weyl law p1[a, b]."""
    ab = np.array([(a, b) for a in range(3) for b in range(3)])
    w1 = p1.ravel()
    idx = np.array(list(itertools.product(range(9), repeat=5)))
    a, b = ab[idx, 0], ab[idx, 1]
    prob = w1[idx].prod(1)
    synd = [(a[:, 0] - a[:, 1] - a[:, 3]) % 3, (a[:, 1] + a[:, 2] - a[:, 4]) % 3,
            (b[:, 0] + b[:, 1] - b[:, 2]) % 3, (-b[:, 1] + b[:, 3] - b[:, 4]) % 3]
    D = np.zeros(len(prob), dtype=np.int64)
    for s in synd:
        D = 3 * D + s
    l = (b[:, 0] + b[:, 3]) % 3
    return np.bincount(D * 3 + l, weights=prob, minlength=3 * 81).reshape(81, 3)


def section_V3():
    print("\n[V3] amplitude damping: Pauli twirl and conjugation symmetry")
    g1, g2 = 0.01, 0.02
    K0 = np.diag([1, math.sqrt(1 - g1), math.sqrt(1 - g2)]).astype(complex)
    K1 = np.zeros((3, 3), complex)
    K1[0, 1] = math.sqrt(g1)
    K2 = np.zeros((3, 3), complex)
    K2[1, 2] = math.sqrt(g2)
    kr = [K0, K1, K2]
    assert np.allclose(sum(K.conj().T @ K for K in kr), np.eye(3))
    Fm = np.array([[W ** (j * k) for k in range(3)] for j in range(3)]) / math.sqrt(3)
    negD = a2.neg_index(4)
    for frame, kraus in (("computational", kr),
                         ("Fourier (F^dag K F)", [Fm.conj().T @ K @ Fm for K in kr])):
        p1 = twirl(kraus)
        asym = np.abs(p1 - p1[np.ix_([0, 2, 1], [0, 2, 1])]).max()
        print(f"  {frame} frame, gamma1={g1}, gamma2={g2}: twirl weights "
              f"p(a,b) (rows a=0,1,2; cols b=0,1,2):")
        for a in range(3):
            print("    " + "  ".join(f"{p1[a, b]:.3e}" for b in range(3)))
        print(f"    sum {p1.sum():.12f}; max |p(a,b) - p(-a,-b)| = {asym:.2e}")
        J = code_capacity_law(p1)
        sym = np.abs(J - J[negD][:, [0, 2, 1]]).max() / J.max()
        det = a2.metrics(J[0].copy())
        post = J / J.sum(1, keepdims=True)
        cmap = a2.label_map(post)
        mp = a2.metrics(a2.accepted_residual(J, a2.rule_from_corr(cmap, 1.0)))
        print(f"    code capacity: max|P(D,l)-P(-D,-l)|/max P = {sym:.2e}; "
              f"detection: alpha={det['alpha']:.5f} S={det['S']:.6f} "
              f"lam={det['lam']:.6f} Fleg={det['Fleg']:.6f} Im z={det['im']:+.2e} "
              f"(p1={J[0, 1] / J[0].sum():.2e}, p2={J[0, 2] / J[0].sum():.2e})")
        print(f"    MAP correction, alpha=1: S={mp['S']:.6f} lam={mp['lam']:.6f} "
              f"Fleg={mp['Fleg']:.6f} Im z={mp['im']:+.2e}")


# ======================================================================= V4
def location_law(loc):
    """Fault-vector law of one DEM location as implemented by the sampler
    (dem_simulation.sample_location_fault_vectors)."""
    w, p = loc.vector_width, loc.probability
    law = np.zeros((3,) * w)
    if loc.gate == "N1":
        if loc.channel == "d":
            law[...] = p / 8
            law[0, 0] = 1 - p
        elif loc.channel == "f":
            law[0, 0], law[1, 0], law[2, 0] = 1 - p, p / 2, p / 2
        elif loc.channel == "p":
            law[0, 0], law[0, 1], law[0, 2] = 1 - p, p / 2, p / 2
        else:
            raise ValueError(loc.channel)
    elif "prob_dist" in loc.params:
        law = np.asarray(loc.params["prob_dist"], float).reshape((3,) * 4)
    else:
        law[...] = p / 80
        law[(0,) * 4] = 1 - p
    return law


def chf_on_grid(Rs, law):
    """phi(k) = sum_v law(v) w^((Rs^T k).v) on the grid k in Z_3^s."""
    s, w = Rs.shape
    F3p = W ** np.outer(np.arange(3), np.arange(3))
    lhat = law.astype(complex)
    for ax in range(w):
        lhat = np.moveaxis(np.tensordot(F3p, lhat, axes=([1], [ax])), 0, ax)
    uidx = np.zeros((3,) * s, dtype=np.int64)
    for c in range(w):
        u = np.zeros((3,) * s, dtype=np.int64)
        for j in np.flatnonzero(Rs[:, c]):
            shp = [1] * s
            shp[j] = 3
            u = u + (np.arange(3) * int(Rs[j, c])).reshape(shp)
        uidx = uidx * 3 + (u % 3)
    return lhat.ravel()[uidx]


def exact_joint_general(model, perturb=None, override=None):
    """Exact P(x) from complex characteristic functions of the actual fault
    laws (no symmetry assumed). perturb = (location index, law) replaces one
    location's law (power test); override(loc) -> law or None replaces the
    law of every location for which it returns a law."""
    nd, nl = model.num_detectors, model.num_logical_observables
    n = nd + nl
    groups = {}
    for li, loc in enumerate(model.locations):
        R = np.zeros((n, loc.vector_width), dtype=np.int64)
        for c, t in enumerate(loc.component_terms):
            for i, v in t.detector:
                R[i, c] = v
            for i, v in t.logical:
                R[nd + i, c] = v
        law = location_law(loc)
        if perturb is not None and perturb[0] == li:
            law = perturb[1]
        if override is not None:
            new = override(loc)
            law = law if new is None else new
        supp = tuple(int(x) for x in np.flatnonzero(R.any(1)))
        if not supp:
            continue
        Rs = R[list(supp)]
        key = (supp, Rs.tobytes(), law.tobytes())
        if key in groups:
            groups[key][2] += 1
        else:
            groups[key] = [Rs, law, 1]
    by_supp = {}
    for (supp, _, _), (Rs, law, cnt) in groups.items():
        phi = chf_on_grid(Rs, law) ** cnt
        by_supp[supp] = by_supp[supp] * phi if supp in by_supp else phi
    hosts = []
    for supp, phi in sorted(by_supp.items(), key=lambda kv: -len(kv[0])):
        for h in hosts:
            if set(supp) <= set(h[0]):
                h[1] = h[1] * phi.reshape([3 if ax in supp else 1 for ax in h[0]])
                break
        else:
            hosts.append([supp, phi])
    Phat = np.ones((3,) * n, dtype=complex)
    for supp, phi in hosts:
        Phat = Phat * phi.reshape([3 if ax in supp else 1 for ax in range(n)])
    F3 = W ** (-np.outer(np.arange(3), np.arange(3)))
    for _ in range(n):
        Phat = np.tensordot(F3, Phat, axes=([1], [n - 1]))
    return Phat / 3 ** n


def load_model(p_idx, d_idx, cl):
    sys.path.insert(0, os.path.join(WORKTREE, "src"))
    from sdim.circuit_io import read_circuit
    from lrb.dem_simulation import build_compact_detector_error_model
    circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/{d_idx}.chp")
    return build_compact_detector_error_model(circ)


def posterior(J, cut=1e-13):
    pD = J.sum(1)
    post = np.full(J.shape, 1 / 3)
    ok = pD > cut
    post[ok] = J[ok] / pD[ok, None]
    return post


def threshold_beta(J_true, post_dec, alpha):
    """beta of the threshold rule built from a decoder posterior, with the
    acceptance tuned to alpha under the true law (randomised at the edge)."""
    pD = J_true.sum(1)
    q, c = post_dec.max(1), np.argmax(post_dec >= post_dec.max(1, keepdims=True)
                                      * (1 - 1e-12), axis=1)
    order = np.argsort(-q, kind="stable")
    cum = np.cumsum(pD[order])
    k = int(np.searchsorted(cum, alpha))
    acc = np.zeros(len(pD))
    acc[order[:k]] = 1.0
    if k < len(pD):
        prev = cum[k - 1] if k > 0 else 0.0
        acc[order[k]] = (alpha - prev) / pD[order[k]]
    return float((acc * J_true[np.arange(len(pD)), c]).sum())


def section_V4(quick):
    print("\n[V4] circuit level, m=2 baseline circuits (read-only)")
    sys.path.insert(0, os.path.join(WORKTREE, "src"))
    from lrb.dem_simulation import sample_detector_error_model
    for p_idx, cl in ((4, 0), (8, 3)):
        t0 = time.time()
        model = load_model(p_idx, 1, cl)
        P_a2, nd = a2.exact_joint_from_dem(model)
        Pg = exact_joint_general(model)
        n = Pg.ndim
        neg = Pg[np.ix_(*([[0, 2, 1]] * n))]
        big = np.abs(Pg.real) > 1e-12
        rel = (np.abs(Pg - neg)[big] / np.abs(Pg.real)[big]).max()
        print(f"  p={PROBS[p_idx]:g} seq {cl}: general-law exact P: max|Im P| = "
              f"{np.abs(Pg.imag).max():.1e}; max|P_general - P_A2| = "
              f"{np.abs(Pg.real - P_a2).max():.1e}; max|P(x)-P(-x)| = "
              f"{np.abs(Pg - neg).max():.1e} (relative, P>1e-12: {rel:.1e}) "
              f"[{time.time() - t0:.0f}s]")
        # power test: one N2 location made asymmetric (all its mass on v=(1,0,0,0))
        li = next(i for i, loc in enumerate(model.locations) if loc.gate == "N2")
        loc = model.locations[li]
        law = np.zeros((3,) * 4)
        law[(0,) * 4] = 1 - loc.probability
        law[1, 0, 0, 0] = loc.probability
        Pp = exact_joint_general(model, perturb=(li, law))
        negp = Pp[np.ix_(*([[0, 2, 1]] * n))]
        print(f"    power test (location {li}, N2, p={loc.probability:g}, all "
              f"fault mass on one vector): max|P(x)-P(-x)| = "
              f"{np.abs(Pp - negp).max():.1e}")
        # asymmetric single-qutrit noise: every N1 d location keeps its total
        # weight p but distributes it like the twirl of amplitude damping
        # (gamma2 = 2 gamma1), computational frame. Detection = unif 1.
        g1 = 0.01
        K0 = np.diag([1, math.sqrt(1 - g1), math.sqrt(1 - 2 * g1)]).astype(complex)
        K1 = np.zeros((3, 3), complex)
        K1[0, 1] = math.sqrt(g1)
        K2 = np.zeros((3, 3), complex)
        K2[1, 2] = math.sqrt(2 * g1)
        tw = twirl([K0, K1, K2])
        tw[0, 0] = 0.0
        tw /= tw.sum()

        def ad_like(loc):
            if loc.gate == "N1" and loc.channel == "d":
                law = loc.probability * tw
                law[0, 0] = 1 - loc.probability
                return law
            return None
        for lab, kw in (("symmetric (SI1000)", {}), ("AD-like N1 d", {"override": ad_like})):
            P = exact_joint_general(model, **kw)
            Jd = P.real.reshape(-1, 3)
            negx = P[np.ix_(*([[0, 2, 1]] * n))]
            md = a2.metrics(Jd[0].copy())
            pr = Jd[0] / Jd[0].sum()
            print(f"    {lab:20s}: max|P(x)-P(-x)| = {np.abs(P - negx).max():.1e}; "
                  f"unif 1: alpha={md['alpha']:.5f} S={md['S']:.6f} p1={pr[1]:.3e} "
                  f"p2={pr[2]:.3e} Im z={md['im']:+.2e} Fleg-lam={md['Fleg'] - md['lam']:.1e}")
        if p_idx == 8:
            N = 200_000 if quick else 1_000_000
            np.random.seed(20261028)
            t0 = time.time()
            out = sample_detector_error_model(model, extra_shots=N)
            x = np.stack([d["data"] for d in out["detectors"]]
                         + [out["logicals"][0]["data"]], 1) % 3
            idx = x @ (3 ** np.arange(n - 1, -1, -1))
            cnt = np.bincount(idx, minlength=3 ** n).astype(float)
            ng = a2.neg_index(n)
            sel = (np.arange(3 ** n) < ng) & (cnt + cnt[ng] >= 25)
            z = (cnt[sel] - cnt[ng][sel]) / np.sqrt(cnt[sel] + cnt[ng][sel])
            from scipy.stats import chi2 as chi2d
            Pr = P_a2.ravel()
            e = N * Pr
            ok = e >= 100
            ze = (cnt[ok] - e[ok]) / np.sqrt(e[ok] * (1 - Pr[ok]))
            print(f"    repository sampler, {N} shots ({time.time() - t0:.0f}s): "
                  f"symmetry test over {sel.sum()} pairs (x, -x) with >= 25 counts: "
                  f"chi2/dof = {np.sum(z ** 2):.1f}/{sel.sum()} (p-value "
                  f"{chi2d.sf(np.sum(z ** 2), sel.sum()):.2f}), max|z| = "
                  f"{np.abs(z).max():.2f}; MC vs exact over {ok.sum()} cells with "
                  f">= 100 expected: chi2/dof = {np.sum(ze ** 2):.1f}/{ok.sum()}, "
                  f"max|z| = {np.abs(ze).max():.2f}")

    print("  model mismatch: truth p=1e-3 (or 1e-2), m=2; regret "
          "S*(alpha) - S(decoder threshold rule) and the bound 2 Delta/alpha")
    cases = [(4, (3, 5, 8)), (8, (6, 11))]
    seqs = (0, 1) if quick else (0, 1, 2)
    for p_true, wrong in cases:
        laws = {}
        avg_n = 4 if quick else 8
        for cl in sorted(set(seqs) | set(range(avg_n)) | {s + 1 for s in seqs}):
            laws[(p_true, cl)] = a2.exact_joint_from_dem(
                load_model(p_true, 1, cl))[0].reshape(-1, 3)
        Javg = np.mean([laws[(p_true, cl)] for cl in range(avg_n)], 0)
        for cl in seqs:
            Jt = laws[(p_true, cl)]
            pD = Jt.sum(1)
            Pt = posterior(Jt)
            fr = a2.frontier(Jt, min_mass=1e-13)
            a_u1 = pD[0]
            alphas = (a_u1, 0.9, 0.95, 0.99, 1.0)
            decs = []
            for pw in wrong:
                decs.append((f"p'={PROBS[pw]:g}", a2.exact_joint_from_dem(
                    load_model(pw, 1, cl))[0].reshape(-1, 3)))
            decs.append((f"seq {cl + 1}", laws[(p_true, cl + 1)]))
            decs.append((f"avg seqs 0-{avg_n - 1}", Javg))
            print(f"    truth p={PROBS[p_true]:g} seq {cl}: alpha(unif 1)="
                  f"{a_u1:.4f}; S* at alpha = " + ", ".join(
                      f"{a:.3g}:{a2.beta_star(fr, a) / a:.6f}" for a in alphas))
            for name, Jd in decs:
                Pd = posterior(Jd)
                keep = pD > 1e-13
                Delta = float((pD[keep] * np.abs(Pt[keep] - Pd[keep]).max(1)).sum())
                regs = []
                for a in alphas:
                    regs.append((a2.beta_star(fr, a) - threshold_beta(Jt, Pd, a)) / a)
                top0 = int(np.argmax(Pd.max(1) * (pD > 1e-13)) == 0)
                print(f"      decoder {name:16s} Delta={Delta:.2e}  2Delta/alpha(1)="
                      f"{2 * Delta:.1e}  regret in S at those alphas: "
                      + " ".join(f"{x:.1e}" for x in regs)
                      + f"  (D=0 top-ranked: {bool(top0)})")


# ======================================================================= V5
def section_V5(quick):
    print("\n[V5] exact D=0 statistics from the critic's hidden-state "
          "contraction vs the baseline CSV (unif 1)")
    ch = _load("critic_chain", "critic_A2-optimal-postselection_r2_chain.py")
    jobs = [(4, 2), (4, 3), (8, 2)] if not quick else [(4, 2)]
    nseq = 6 if quick else NSEQ
    for p_idx, d_idx in jobs:
        t0 = time.time()
        m = DEPTHS[d_idx]
        P0s, q0s = [], []
        for cl in range(nseq):
            model, mm, cuts = ch.load(p_idx, d_idx, cl)
            chain = ch.Chain(model, mm, cuts)
            P0 = chain.prob(np.zeros((mm + 1, 4), dtype=np.int64))
            P0s.append(P0.sum())
            q0s.append(P0[0] / P0.sum())
            assert int(P0.argmax()) == 0
        P0s, q0s = np.array(P0s), np.array(q0s)
        r = csv_rows(p_idx, "unif", 1)
        acc_csv = 1 - r["Rejected Runs"][d_idx]
        S_csv = (1 + 2 * r["Fidelity averages"][d_idx]) / 3
        lam = (3 * q0s - 1) / 2
        se_acc = math.sqrt(np.sum(P0s * (1 - P0s) / SHOTS)) / nseq
        se_S = (2 / 3) * math.sqrt(np.sum(V(lam) / (P0s * SHOTS))) / nseq
        msg = ""
        if nseq == NSEQ:
            msg = (f"; CSV acceptance {acc_csv:.6f} (z={(acc_csv - P0s.mean()) / se_acc:+.2f}),"
                   f" CSV S=(1+2F)/3 {S_csv:.6f} (z={(S_csv - q0s.mean()) / se_S:+.2f}),"
                   f" shot-noise SEs {se_acc:.1e}, {se_S:.1e}")
        print(f"  p={PROBS[p_idx]:g} m={m} ({nseq} seqs, {time.time() - t0:.0f}s): "
              f"exact mean P(D=0)={P0s.mean():.6f}, mean q(0)={q0s.mean():.6f} "
              f"(range {q0s.min():.6f}-{q0s.max():.6f}){msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20261029)
    t0 = time.time()
    run = (lambda s: args.only is None or s in args.only)
    if run("V1"):
        section_V1()
    if run("V2"):
        section_V2(rng)
    if run("V3"):
        section_V3()
    if run("V4"):
        section_V4(args.quick)
    if run("V5"):
        section_V5(args.quick)
    print(f"\ntotal runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
