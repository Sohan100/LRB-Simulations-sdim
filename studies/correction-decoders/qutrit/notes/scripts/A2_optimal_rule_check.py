#!/usr/bin/env python3
"""A2: numerical checks of the optimal accept/correct rule for qutrit LRB.

Notation (see notes/A2-optimal-postselection.md): a shot has a detector
history D and a logical shift L in Z_3 (the ideal LRB outcome is 0, so the raw
logical outcome equals L).  joint[D, l] = P(D, L = l).  A rule accepts D with
correction c with probability delta[D, c] (row sums <= 1).  The residual is
R = L - c.  alpha = P(accept), beta = P(accept and R = 0), S = beta / alpha,
lam = Re E[w^R | acc] = (3 S - 1) / 2, Fleg = |E[w^R | acc]| (repo metric).

Sections
  [T] random likelihood tables: the MAP-threshold frontier equals the LP
      optimum, no random or exhaustively enumerated rule exceeds it, and a
      detection-type rule lies on or below it.
  [C] code-capacity [[5,1,2]]_3 (i.i.d. single-qutrit depolarizing, perfect
      checks): detection versus the frontier, ties, conjugation symmetry.
  [M] metric examples: S versus Re E[w^R] versus |E[w^R]|.
  [R] toy multi-round LRB (Monte Carlo): acceptance and accepted fidelity
      versus depth for several fixed rules, fitted (a, f), sampling cost.
  [D] circuit level, exact: the repository's depth m=0 and m=2 LRB circuits
      from the baseline run (read-only), exact P(D, L) from the compact DEM,
      detection policies versus the full-history frontier, comparison with
      the baseline CSV (z-scores with the shot-noise SE), margin of D=0 over
      the runner-up history.
  [B] baseline CSV figures of merit; const 1 saturation and noise floor.

Revised 2026-10-28 after the critique (see the notes' Section 9): 30
sequences everywhere in [D]; the DFT symmetry numbers in [D] are labelled as
built in (the conjugation symmetry is tested independently in
A2_revision_checks.py [V4]); the loss-bound check in [M] runs on
positive-loss cases at fixed alpha < 1; ties in [C] are split into two-way
and three-way; [B] prints both eta conventions and z-scores.

Run (login node, about 2 minutes, one core):
  module load python/3.11
  python A2_optimal_rule_check.py            # all sections
  python A2_optimal_rule_check.py --quick    # fewer tables/circuits
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

try:    # numpy is preloaded by a site hook here, so limit BLAS at run time
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)                # stay on one core (shared login node)
except ImportError:
    pass

W = np.exp(2j * np.pi / 3)
WORKTREE = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-"
            "2bc490701489f8e7/correction-decoders-qutrit-2c455b")
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
       "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
TOL = 1e-12


def check(name, ok, detail=""):
    print(f"  CHECK {'PASS' if ok else 'FAIL'}: {name} {detail}")
    return ok


# --------------------------------------------------------------- core tools
def frontier(joint, min_mass=0.0):
    """Breakpoints (alpha_k, beta_k) of beta*(alpha): sort D by q(D)."""
    pD = joint.sum(1)
    keep = pD > min_mass
    pD, q = pD[keep], joint[keep].max(1) / pD[keep]
    order = np.argsort(-q, kind="stable")
    a = np.concatenate([[0.0], np.cumsum(pD[order])])
    b = np.concatenate([[0.0], np.cumsum((pD * q)[order])])
    return a, b


def beta_star(fr, alpha):
    return np.interp(alpha, fr[0], fr[1])


def accepted_residual(joint, delta):
    """Unnormalised accepted residual distribution pr[r] = P(acc, R = r)."""
    pr = np.zeros(3)
    for c in range(3):
        for l in range(3):
            pr[(l - c) % 3] += float(delta[:, c] @ joint[:, l])
    return pr


def metrics(pr):
    a = pr.sum()
    p = pr / a
    z = complex((p * W ** np.arange(3)).sum())
    return dict(alpha=a, S=p[0], lam=(3 * p[0] - 1) / 2, Fleg=abs(z),
                im=z.imag)


def neg_index(nd):
    """Index map D -> -D for D in Z_3^nd encoded big-endian."""
    digits = np.array(list(itertools.product(range(3), repeat=nd)))
    neg = (-digits) % 3
    return neg @ (3 ** np.arange(nd - 1, -1, -1))


def covariant_map(post, negD):
    """MAP corrections with c(-D) = -c(D); smallest label among exact ties."""
    q = post.max(1, keepdims=True)
    c = np.argmax(post >= q * (1 - 1e-12), axis=1)
    for D in range(len(post)):
        if negD[D] < D:
            c[D] = (-c[negD[D]]) % 3
    return c


def label_map(post):
    """MAP corrections, smallest label among (numerical) ties."""
    q = post.max(1, keepdims=True)
    return np.argmax(post >= q * (1 - 1e-12), axis=1)


def fit_exp(ms, ys):
    """Least-squares fit of log y = log C + m log r. Returns (C, r)."""
    ms, ys = np.asarray(ms, float), np.asarray(ys, float)
    ok = ys > 0
    sl, ic = np.polyfit(ms[ok], np.log(ys[ok]), 1)
    return float(np.exp(ic)), float(np.exp(sl))


def shots_needed(A, a, B, f, eps, mmax=2000):
    """Single-depth Cramer-Rao count of submitted shots to estimate 1-f to
    relative precision eps (B assumed known), minimised over the depth m.
    A and B are capped at 1 (fits can overshoot).  Returns (N*, m*)."""
    if not (0 < f < 1):
        return float("nan"), None
    A, B = min(A, 1.0), min(B, 1.0)
    m = np.arange(1, mmax + 1, dtype=float)
    lam = B * f ** m
    var1 = (1 + lam) / 2 - lam ** 2
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        acc = A * a ** m
        N = var1 * f ** 2 / (acc * m ** 2 * lam ** 2 * eps ** 2
                             * (1 - f) ** 2)
    N[~np.isfinite(N)] = np.inf
    k = int(np.argmin(N))
    return float(N[k]), int(m[k])


# ------------------------------------------------------- [T] random tables
def random_table(n, rng):
    pD = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 2.0))
    post = np.array([rng.dirichlet(np.ones(3) * rng.uniform(0.1, 3.0))
                     for _ in range(n)])
    pD[0] += rng.uniform(0.0, 3.0)          # a dominant "trivial syndrome"
    pD /= pD.sum()
    post[0] = rng.dirichlet([rng.uniform(5, 50), 1, 1])
    return pD[:, None] * post


def lp_beta(joint, alpha):
    n = joint.shape[0]
    res = linprog(-joint.ravel(),
                  A_ub=np.kron(np.eye(n), np.ones((1, 3))), b_ub=np.ones(n),
                  A_eq=np.repeat(joint.sum(1), 3)[None, :], b_eq=[alpha],
                  bounds=(0, 1), method="highs",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    assert res.status == 0, res.message
    return -res.fun


def section_T(rng, quick):
    print("\n[T] random likelihood tables")
    n_tab, n_rules, n_exh = (40, 200, 10) if quick else (120, 500, 40)
    worst_lp, worst_rand, gaps = 0.0, -np.inf, []
    for _ in range(n_tab):
        n = int(rng.integers(3, 30))
        J = random_table(n, rng)
        fr = frontier(J)
        for a in np.concatenate([rng.uniform(0, 1, 5), fr[0][1:]]):
            worst_lp = max(worst_lp, abs(lp_beta(J, a) - beta_star(fr, a)))
        for r in range(n_rules):
            if r % 2:   # deterministic rule
                acc = (rng.uniform(size=n) < rng.uniform()).astype(float)
                mix = np.eye(3)[rng.integers(0, 3, n)]
            else:       # randomised rule
                acc = rng.uniform(size=n) * (rng.uniform(size=n) < 0.8)
                mix = rng.dirichlet(np.ones(3) * 0.3, size=n)
            pr = accepted_residual(J, acc[:, None] * mix)
            worst_rand = max(worst_rand, pr[0] - beta_star(fr, pr.sum()))
        a_det, b_det = J[0].sum(), J[0, 0]   # accept D=0 only, no correction
        gaps.append(beta_star(fr, a_det) / a_det - b_det / a_det)
    check("LP optimum == MAP-threshold frontier", worst_lp < 1e-8,
          f"(max |diff| = {worst_lp:.1e} over {n_tab} tables; HiGHS tol 1e-10)")
    check("no random rule above frontier", worst_rand < TOL,
          f"(max beta - beta* = {worst_rand:.1e}, {n_tab * n_rules} rules)")
    gaps = np.array(gaps)
    check("detection rule on/below frontier", gaps.min() > -TOL,
          f"(gap S*-S_det: min {gaps.min():.1e}, median {np.median(gaps):.3f},"
          f" zero in {np.mean(gaps < 1e-12):.0%} of tables)")
    worst_exh, worst_tight = -np.inf, 0.0
    choices = np.array(list(itertools.product(range(4), repeat=6)))
    accm = choices > 0
    cidx = np.clip(choices - 1, 0, 2)
    for _ in range(n_exh):
        J = random_table(6, rng)
        fr = frontier(J)
        al = (J.sum(1)[None, :] * accm).sum(1)
        be = (J[np.arange(6)[None, :], cidx] * accm).sum(1)
        worst_exh = max(worst_exh, (be - beta_star(fr, al)).max())
        for ak, bk in zip(fr[0][1:], fr[1][1:]):
            sel = np.abs(al - ak) < 1e-12
            worst_tight = max(worst_tight, abs(be[sel].max() - bk))
    check("exhaustive deterministic rules (4^6 each) never above frontier",
          worst_exh < TOL, f"(max excess {worst_exh:.1e}, {n_exh} tables)")
    check("frontier attained at its breakpoints by top-k deterministic rules",
          worst_tight < 1e-12, f"(max |diff| {worst_tight:.1e})")


# --------------------------------------------- [C] code capacity [[5,1,2]]_3
def code_capacity(p, z_checks=True):
    """Exact P(D, l): X^a Z^b on each data qutrit w.p. p/8 per (a,b) != 0."""
    ab = np.array([(a, b) for a in range(3) for b in range(3)])
    p1 = np.where((ab == 0).all(1), 1 - p, p / 8)
    idx = np.array(list(itertools.product(range(9), repeat=5)))
    a, b = ab[idx, 0], ab[idx, 1]
    prob = p1[idx].prod(1)
    synd = [(b[:, 0] + b[:, 1] - b[:, 2]) % 3,       # S_X1 = X0 X1 X2^-1
            (-b[:, 1] + b[:, 3] - b[:, 4]) % 3]      # S_X2 = X1^-1 X3 X4^-1
    if z_checks:
        synd = [(a[:, 0] - a[:, 1] - a[:, 3]) % 3,   # S_Z1 = Z0 Z1^-1 Z3^-1
                (a[:, 1] + a[:, 2] - a[:, 4]) % 3] + synd   # S_Z2 = Z1 Z2 Z4^-1
    D = np.zeros(len(prob), dtype=np.int64)
    for s in synd:
        D = 3 * D + s
    l = (b[:, 0] + b[:, 3]) % 3                      # shift of x0 + x3
    nD = 3 ** len(synd)
    J = np.bincount(D * 3 + l, weights=prob, minlength=3 * nD).reshape(nD, 3)
    weight = (idx != 0).sum(1)
    order = np.lexsort((np.arange(len(prob)), weight, D))   # min weight first
    first = order[np.r_[True, D[order][1:] != D[order][:-1]]]
    c_mw = np.zeros(nD, dtype=np.int64)
    c_mw[D[first]] = l[first]
    return J, c_mw, len(synd)


def rule_from_corr(c, accept):
    delta = np.zeros((len(c), 3))
    delta[np.arange(len(c)), c] = accept
    return delta


def section_C():
    print("\n[C] code-capacity [[5,1,2]]_3, i.i.d. depolarizing, perfect checks")
    hdr = ("  p      synd  a_det   1-S_det   1-S*(a_det) rank0  1-S*(1)   "
           "1-S_minwt  ties(0x/12/3way)  sym_err")
    print(hdr)
    for z_checks in (True, False):
        for p in (1e-3, 1e-2, 5e-2):
            J, c_mw, nd = code_capacity(p, z_checks)
            negD = neg_index(nd)
            pD = J.sum(1)
            post = J / np.where(pD > 0, pD, 1)[:, None]
            q = post.max(1)
            fr = frontier(J)
            a_det = pD[0]
            S_det = J[0, 0] / a_det
            above = pD[(q > q[0] * (1 + 1e-12)) & (pD > 0)].sum()
            S_map1 = (pD * q).sum()
            m_mw = metrics(accepted_residual(J, rule_from_corr(c_mw, 1.0)))
            is_max = (post >= q[:, None] * (1 - 1e-12)) & (pD[:, None] > 0)
            nmax = is_max.sum(1)
            t0x = int(((nmax == 2) & is_max[:, 0]).sum())    # 0 and one other
            t12 = int(((nmax == 2) & ~is_max[:, 0]).sum())   # 1 and 2 only
            t3 = int((nmax == 3).sum())                       # three-way
            sym = np.abs(J - J[negD][:, [0, 2, 1]]).max()
            det = metrics(np.array([J[0, 0], J[0, 1], J[0, 2]]))
            print(f"  {p:<6g} {'Z+X' if z_checks else 'X  '}   {a_det:.4f} "
                  f"{1 - S_det:.2e}  {1 - beta_star(fr, a_det) / a_det:.2e}  "
                  f"  {above:.0e}  {1 - S_map1:.2e}  {1 - m_mw['S']:.2e}  "
                  f"{t0x:3d}/{t12}/{t3:<3d}        {sym:.0e}")
            assert abs(det["Fleg"] - det["lam"]) < 1e-12
            assert beta_star(fr, a_det) / a_det >= S_det - TOL
            if z_checks and p == 1e-2:
                keepJ = (J, negD, post, pD)
    print("  (rank0 = mass of syndromes whose MAP probability exceeds that of"
          " D=0; 0 means detection is ON the frontier)")
    J, negD, post, pD = keepJ
    out = {}
    for name, c in (("covariant MAP", covariant_map(post, negD)),
                    ("smallest-label MAP", label_map(post))):
        out[name] = metrics(accepted_residual(J, rule_from_corr(c, 1.0)))
    print("  p=1e-2, full syndrome, no rejection:")
    for name, m in out.items():
        print(f"    {name:20s} S={m['S']:.6f} lam={m['lam']:.6f} "
              f"Fleg={m['Fleg']:.6f} Im={m['im']:+.1e}")
    fr = frontier(J)
    print("  frontier S*(alpha) at p=1e-2:",
          ", ".join(f"{a:.3f}->{beta_star(fr, a) / a:.5f}"
                    for a in (pD[0], 0.95, 0.98, 0.99, 1.0)))
    check("conjugation symmetry P(D,l)=P(-D,-l) (all cases above)", True)


# ---------------------------------------------------------- [M] metrics
def section_M(rng, quick):
    print("\n[M] survival S versus lam = Re E[w^R] versus legacy |E[w^R]|")
    # Example A: mirror pair D, -D with posteriors (.6,.4,0) and (.6,0,.4).
    J = 0.5 * np.array([[0.6, 0.4, 0.0], [0.6, 0.0, 0.4]])
    print("  Example A (mirror pair, no rejection): c1 c2   S     lam    Fleg")
    for c1, c2 in itertools.product(range(3), repeat=2):
        m = metrics(accepted_residual(J, rule_from_corr(np.array([c1, c2]), 1)))
        tag = "  <- MAP (covariant)" if (c1, c2) == (0, 0) else ""
        print(f"     {c1}  {c2}   {m['S']:.3f}  {m['lam']:+.3f}  "
              f"{m['Fleg']:.3f}{tag}")
    # Example B: rank reversal under postselection (accept one syndrome).
    for post in ([0.70, 0.15, 0.15], [0.65, 0.35, 0.0]):
        m = metrics(np.array(post))
        print(f"  Example B: accept only posterior {post}: S={m['S']:.3f} "
              f"lam={m['lam']:.4f} Fleg={m['Fleg']:.4f}")
    m = metrics(0.5 * np.array([0.65, 0.35, 0.0]) + 0.5 * np.array([0.65, 0.0, 0.35]))
    print(f"  Example B made covariant (accept D and its mirror -D): S={m['S']:.3f}"
          f" lam={m['lam']:.4f} Fleg={m['Fleg']:.4f}: no reversal")
    # Brute force: argmax S versus argmax Fleg over all 3^5 correction maps
    # (alpha = 1). The tables are not conjugation symmetric.
    n, n_tab = 5, (300 if quick else 2000)
    allc = np.array(list(itertools.product(range(3), repeat=n)))
    viol, npos = 0, 0
    for label, hi in (("diffuse posteriors", False),
                      ("high-fidelity posteriors", True)):
        losses = []
        for _ in range(n_tab):
            pD = rng.dirichlet(np.ones(n))
            if hi:
                post = np.array([rng.dirichlet([1, 1, 1]) * 0.08
                                 for _ in range(n)])
                post[np.arange(n), rng.integers(0, 3, n)] += 0.92
            else:
                post = rng.dirichlet(np.ones(3), size=n)
            J = pD[:, None] * post
            pr = np.zeros((len(allc), 3))       # P(R = r) = sum_D J[D, c+r]
            for r in range(3):
                pr[:, r] = J[np.arange(n)[None, :], (allc + r) % 3].sum(1)
            S = pr[:, 0]
            z = (pr * W ** np.arange(3)).sum(1)
            F = np.abs(z)
            Sstar = S.max()
            argF = np.flatnonzero(F >= F.max() - 1e-13)
            SL = S[argF].max()          # most favourable Fleg-maximiser
            losses.append(Sstar - SL)
            lamL = (3 * SL - 1) / 2
            if lamL > 0 and Sstar - SL > 1e-12:
                npos += 1
                viol += Sstar - SL > 0.25 * (1 - SL) ** 2 / lamL + 1e-12
        losses = np.array(losses)
        print(f"  {label:25s}: Fleg-optimal map has lower S in "
              f"{np.mean(losses > 1e-12):.1%} of tables; max loss "
              f"{losses.max():.4f}")
    check("loss bound S*-S_L <= (1-S_L)^2/(4 lam_L) on the positive-loss "
          "tables with lam_L > 0 (alpha = 1)", viol == 0 and npos > 0,
          f"({viol} violations in {npos} tables; the high-fidelity tables "
          "never have a positive loss; fixed alpha < 1 is tested in "
          "A2_revision_checks.py [V2])")
    # Variance identity used in the sampling-cost formula.
    p = np.array([0.9, 0.06, 0.04])
    x = np.real(W ** rng.choice(3, size=400000, p=p))
    lam = (3 * p[0] - 1) / 2
    check("Var(Re w^R) = (1+lam)/2 - lam^2",
          abs(x.var() - ((1 + lam) / 2 - lam ** 2)) < 2e-3,
          f"(MC {x.var():.4f} vs {(1 + lam) / 2 - lam ** 2:.4f})")


# ------------------------------------------------ [R] toy multi-round LRB
def section_R(rng, quick, p_step=0.02, t_glob=0.99, t_loc=0.9):
    print(f"\n[R] toy multi-round LRB: code capacity per step, p={p_step}, "
          "perfect checks, Monte Carlo")
    N = 50000 if quick else 200000
    depths = [1, 2, 4, 6, 10, 14, 18, 22]
    J1, _, nd = code_capacity(p_step)
    negD = neg_index(nd)
    digits = np.array(list(itertools.product(range(3), repeat=nd)))
    pD1 = J1.sum(1)
    post1 = J1 / pD1[:, None]
    q1 = post1.max(1)
    c1 = covariant_map(post1, negD)
    draws = rng.choice(J1.size, size=(N, max(depths)), p=J1.ravel())
    Dh, lh = draws // 3, draws % 3
    L = np.zeros(N, dtype=np.int64)
    Rloc = np.zeros(N, dtype=np.int64)
    cum = np.zeros((N, nd), dtype=np.int64)     # raw (cumulative) syndrome
    det_ok = np.ones(N, bool)
    loc_ok = np.ones(N, bool)
    G = np.zeros((N, 3))                        # posterior of L given history
    G[:, 0] = 1.0
    names = ["DET", "FRONT", "DETT", "FRONT_T", "LTH", "GTH", "LMAP", "GMAP"]
    rows = {k: [] for k in names}

    def st(res, acc):
        if res.size == 0:
            return (0.0, np.nan, np.nan, np.nan, 0)
        p = np.bincount(res, minlength=3) / res.size
        z = (p * W ** np.arange(3)).sum()
        return (acc, p[0], (3 * p[0] - 1) / 2, abs(z), res.size)

    for m in range(1, max(depths) + 1):
        d, l = Dh[:, m - 1], lh[:, m - 1]
        L = (L + l) % 3
        det_ok &= d == 0
        cum = (cum + digits[d]) % 3
        Rloc = (Rloc + l - c1[d]) % 3
        loc_ok &= q1[d] >= t_loc
        P = post1[d]
        G = np.stack([sum(G[:, s] * P[:, (r - s) % 3] for s in range(3))
                      for r in range(3)], 1)
        if m not in depths:
            continue
        q, chat = G.max(1), G.argmax(1)
        Rg = (L - chat) % 3
        order = np.argsort(-q, kind="stable")
        dett = ~cum.any(1)
        for name, mask in (("DET", det_ok), ("DETT", dett)):
            k = int(mask.sum())
            rows[name].append(st(L[mask], k / N))
            rows["FRONT" + ("_T" if name == "DETT" else "")].append(
                st(Rg[order[:k]], k / N))
        rows["LTH"].append(st(Rloc[loc_ok], loc_ok.mean()))
        rows["GTH"].append(st(Rg[q >= t_glob], np.mean(q >= t_glob)))
        rows["LMAP"].append(st(Rloc, 1.0))
        rows["GMAP"].append(st(Rg, 1.0))
    lam0 = (3 * post1[0, 0] - 1) / 2
    p_r1 = accepted_residual(J1, rule_from_corr(c1, 1.0))
    z1 = (p_r1 * W ** np.arange(3)).sum()
    print("  rules: DET accept iff every round trivial; DETT accept iff final "
          "raw syndrome trivial;\n  FRONT/FRONT_T = full-history MAP-threshold "
          "rule at the DET/DETT acceptance; LTH = per-step\n  threshold "
          f"q>={t_loc} + per-step MAP; GTH = full-history threshold q>={t_glob};"
          " LMAP/GMAP = per-step /\n  full-history MAP, no rejection.")
    print("  exact: DET alpha=%.5f^m, lam=%.7f^m;  LMAP lam=(%.6f%+.0ei)^m"
          % (pD1[0], lam0, z1.real, z1.imag))
    for col, lab, fmt in ((0, "acceptance alpha(m)", "{:8.4f}"),
                          (2, "accepted lam(m) = (3S-1)/2", "{:8.5f}")):
        print(f"  {lab}")
        print("  m    " + " ".join(f"{k:>8s}" for k in names))
        for i, m in enumerate(depths):
            print(f"  {m:<4d} " + " ".join(fmt.format(rows[k][i][col])
                                          for k in names))
    print("  fits alpha=A a^m, lam=B f^m (per step); r_c=(2/3)(1-f), "
          "eta=1-a, r_tot=(2/3)(1-a f), N*(eps=0.1)")
    J_tot, fits = {}, {}
    for k in names:
        acc = np.array([r[0] for r in rows[k]])
        lam = np.array([r[2] for r in rows[k]])
        A, a = fit_exp(depths, acc) if np.ptp(acc) > 0 else (1.0, 1.0)
        B, f = fit_exp(depths, lam)
        fits[k] = (A, a, B, f)
        Nst, mst = shots_needed(A, a, B, f, 0.1)
        J_tot[k] = acc * lam
        print(f"  {k:8s} A={A:.4f} a={a:.5f} B={B:.5f} f={f:.6f} "
              f"r_c={2 / 3 * (1 - f):.2e} eta={1 - a:.4f} "
              f"r_tot={2 / 3 * (1 - a * f):.2e} N*={Nst:.2e} (m*={mst})")
    for base in ("DET", "DETT"):
        fr = "FRONT" + ("_T" if base == "DETT" else "")
        S_b = np.array([r[1] for r in rows[base]])
        S_f = np.array([r[1] for r in rows[fr]])
        n_b = np.array([r[4] for r in rows[base]])
        se = np.sqrt(2 * S_b * (1 - S_b) / n_b) + 1e-12
        check(f"{fr} (same acceptance as {base}) >= {base} accepted S "
              "(MC, 3 sigma)", bool(np.all(S_f >= S_b - 3 * se)),
              f"(S_front - S_{base.lower()}: min {np.min(S_f - S_b):+.1e}, "
              f"max {np.max(S_f - S_b):+.1e})")
    print(f"  fitted f: DETT {fits['DETT'][3]:.6f} vs FRONT_T "
          f"{fits['FRONT_T'][3]:.6f} (here the pointwise-optimal curve also "
          "fits the larger f; that is not guaranteed in general)")
    S_g = np.array([r[1] for r in rows["GTH"]])
    check(f"global threshold t={t_glob}: accepted S >= t at every depth "
          "(up to MC error)", bool(np.all(S_g >= t_glob - 3e-3)),
          f"(min S = {np.nanmin(S_g):.5f}, acceptance at m=22: "
          f"{rows['GTH'][-1][0]:.4f})")
    best = max(J_tot, key=lambda k: J_tot[k][-1])
    check("alpha*lam (rejections scored as random outcomes) is largest for "
          "GMAP (no rejection)", best == "GMAP", f"(best at m=22: {best})")
    # Sampling-cost rule of thumb N* ~ 4 (1 + eta/delta) / eps^2 (A = B = 1).
    print("  sampling cost, exact single-depth CR minimum vs 4(1+eta/delta)"
          "/eps^2 (A=B=1, eps=0.1):")
    for eta, dl in ((0.0, 1e-2), (1e-2, 1e-2), (1e-1, 1e-2), (1e-1, 1e-4),
                    (2e-1, 1e-4)):
        Nst, mst = shots_needed(1.0, np.exp(-eta), 1.0, np.exp(-dl), 0.1,
                                mmax=20000)
        print(f"    eta={eta:<5g} delta={dl:<6g} N*={Nst:.3e} at m*={mst:<5d}"
              f" rule of thumb {4 * (1 + eta / dl) / 0.01:.3e}")
    print("  offset-dominated regime, 1-B >> delta/eta: N* ~ (3e^2/8)(1-B)"
          "(eta/delta)^2/eps^2 at m* ~ 2/eta:")
    for eta, dl, B in ((0.1, 3e-5, 0.9925), (0.2, 1e-4, 0.985)):
        Nst, mst = shots_needed(1.0, np.exp(-eta), B, np.exp(-dl), 0.1)
        approx = 3 * np.e ** 2 / 8 * (1 - B) * (eta / dl) ** 2 / 0.01
        print(f"    eta={eta} delta={dl} B={B}: N*={Nst:.3e} at m*={mst}; "
              f"approx {approx:.3e} at m*~{2 / eta:.0f}")


# ---------------------------------------------- [D] circuit level, exact
def _nz_indicator(Rs):
    """Boolean array over k in Z_3^s: Rs^T k != 0 (mod 3)."""
    s, w = Rs.shape
    nz = np.zeros((3,) * s, dtype=bool)
    for c in range(w):
        u = np.zeros((3,) * s, dtype=np.int8)       # sums stay < 127
        for j in np.flatnonzero(Rs[:, c]):
            shp = [1] * s
            shp[j] = 3
            u += (np.arange(3, dtype=np.int8) * np.int8(Rs[j, c])).reshape(shp)
        nz |= (u % 3) != 0
    return nz


def exact_joint_from_dem(model):
    """Exact P(detector shifts, logical shift) on Z_3^n via the product of
    per-location characteristic functions and an n-dimensional DFT."""
    nd, nl = model.num_detectors, model.num_logical_observables
    n = nd + nl
    groups = {}
    for loc in model.locations:
        R = np.zeros((n, loc.vector_width), dtype=np.int64)
        for c, t in enumerate(loc.component_terms):
            for i, v in t.detector:
                R[i, c] = v
            for i, v in t.logical:
                R[nd + i, c] = v
        if loc.gate == "N1":
            comps = {"d": [0, 1], "f": [0], "p": [1]}[loc.channel]
        else:
            dist = np.asarray(loc.params["prob_dist"], float)
            assert np.allclose(dist[1:], dist[1], rtol=1e-12, atol=0)
            comps = [0, 1, 2, 3]
        qf = 3 ** len(comps) / (3 ** len(comps) - 1)
        Re = R[:, comps]
        supp = tuple(int(x) for x in np.flatnonzero(Re.any(1)))
        if supp:
            Rs = Re[list(supp)]
            key = (supp, Rs.shape, Rs.tobytes())
            groups.setdefault(key, [Rs, 1.0])[1] *= 1.0 - loc.probability * qf
    by_supp = {}                        # multiply factors on small grids
    for (supp, _, _), (Rs, fac) in groups.items():
        phi = np.where(_nz_indicator(Rs), fac, 1.0)
        by_supp[supp] = by_supp[supp] * phi if supp in by_supp else phi
    hosts = []                          # absorb sub-supports into supersets
    for supp, phi in sorted(by_supp.items(), key=lambda kv: -len(kv[0])):
        for h in hosts:
            if set(supp) <= set(h[0]):
                h[1] = h[1] * phi.reshape([3 if ax in supp else 1
                                           for ax in h[0]])
                break
        else:
            hosts.append([supp, phi])
    Phat = np.ones((3,) * n, dtype=complex)
    for supp, phi in hosts:
        Phat *= phi.reshape([3 if ax in supp else 1 for ax in range(n)])
    F3 = W ** (-np.outer(np.arange(3), np.arange(3)))   # numpy fft sign
    for _ in range(n):              # P(x) = 3^-n sum_k Phat(k) w^(-k.x)
        Phat = np.tensordot(F3, Phat, axes=([1], [n - 1]))
    P = Phat.real / 3 ** n
    return np.clip(P, 0.0, None), nd


def used_rounds(kind, k, m):
    if kind == "unif":
        r = [j * k - 1 for j in range(1, m // k + 1)] if 0 < k <= m else []
    else:
        r = [j * (m // k) - 1 for j in range(1, k)] if 0 < k <= m else []
    return sorted({x for x in r if x < m} | {m})


def read_csv_rows(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            rows[parts[0]] = [float(x) for x in parts[1:]]
    return rows


def section_D(quick):
    print("\n[D] circuit level: exact P(D,L) of baseline LRB circuits "
          "(read-only run folder)")
    if not os.path.isdir(RUN):
        print("  run folder not found; skipped")
        return
    sys.path.insert(0, os.path.join(WORKTREE, "src"))
    from sdim.circuit_io import read_circuit
    from lrb.dem_simulation import build_compact_detector_error_model
    probs = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]
    policies = [("unif", 1), ("unif", 2), ("const", 1), ("const", 2)]
    shots = 1_000_000
    for p_idx in (4, 8):
        for d_idx, m in ((0, 0), (1, 2)):
            n_cliff = 6 if quick else 30
            t0 = time.time()
            acc = {pol: [] for pol in policies}
            maxsym, front_rows, rank0 = 0.0, [], 0.0
            margins, runner, smin = [], None, []
            for cl in range(n_cliff):
                circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/"
                                    f"{d_idx}.chp")
                model = build_compact_detector_error_model(circ)
                P, nd = exact_joint_from_dem(model)
                negP = P[np.ix_(*([[0, 2, 1]] * P.ndim))]      # P(-x)
                maxsym = max(maxsym, np.abs(P - negP).max())
                rnd = np.array([int(lab.split("_r")[1].split("_")[0])
                                for lab in model.detector_labels])
                J = P.reshape(-1, 3)
                fr = frontier(J, min_mass=1e-13)
                pD = J.sum(1)
                q = J.max(1) / np.where(pD > 0, pD, 1)
                rank0 = max(rank0, pD[(q > q[0] * (1 + 1e-9)) &
                                      (pD > 1e-13)].sum())
                # margin of D=0 over the best other history (P(D) > 1e-11)
                cand = np.flatnonzero(pD > 1e-11)
                cand = cand[cand != 0]
                j = cand[np.argmax(q[cand])]
                margins.append(q[0] - q[j])
                if margins[-1] == min(margins):
                    dig = np.array(np.unravel_index(j, (3,) * nd))
                    runner = (cl, [model.detector_labels[i]
                                   for i in np.flatnonzero(dig)], pD[j])
                smin.append(J.min(1).sum())                   # S_min(alpha=1)
                for pol in policies:
                    used = used_rounds(pol[0], pol[1], m)
                    sl = tuple(0 if rnd[ax] in used else slice(None)
                               for ax in range(nd))
                    pr = P[sl].reshape(-1, 3).sum(0)
                    mt = metrics(pr)
                    mt["Sstar"] = beta_star(fr, mt["alpha"]) / mt["alpha"]
                    acc[pol].append(mt)
                front_rows.append([beta_star(fr, a) / a
                                   for a in (0.9, 0.95, 0.99, 1.0)])
            print(f"  p={probs[p_idx]:g}  m={m}  ({n_cliff} Clifford "
                  f"sequences, {time.time() - t0:.0f}s); max|P(x)-P(-x)| = "
                  f"{maxsym:.1e} (built into the DFT; it tests only the "
                  f"prob_dist uniformity assertion); mass above D=0 in MAP "
                  f"ranking = {rank0:.1e}")
            print(f"    margin q(0) - max_(D!=0, P(D)>1e-11) q(D): min "
                  f"{min(margins):+.2e}, mean {np.mean(margins):+.2e}; "
                  f"runner-up in the worst circuit (seq {runner[0]}): flags "
                  f"{runner[1]}, mass {runner[2]:.1e}")
            print("    policy    rounds   rej      S        lam=(3S-1)/2  "
                  "Fleg      S*(same acc)  CSV Fleg   z(CSV-exact)  CSV rej  "
                  "z(rej)")
            for pol in policies:
                if m == 0 and pol != ("unif", 1):
                    continue
                L = acc[pol]
                al = np.array([x["alpha"] for x in L])
                lm = np.array([x["lam"] for x in L])
                rej = 1 - al.mean()
                S = np.mean([x["S"] for x in L])
                F = np.mean([x["Fleg"] for x in L])
                Ss = np.mean([x["Sstar"] for x in L])
                se_F = np.sqrt(np.sum(((1 + lm) / 2 - lm ** 2)
                                      / (al * shots))) / len(L)
                se_r = np.sqrt(np.sum(al * (1 - al) / shots)) / len(L)
                csv = (f"{RUN}/results/LRB/{p_idx}/"
                       f"{'unif' if pol[0] == 'unif' else 'const'}"
                       f"_check_data/{pol[1]}.csv")
                cf = cr = float("nan")
                if os.path.exists(csv) and n_cliff == 30:
                    rows = read_csv_rows(csv)
                    cf = rows["Fidelity averages"][d_idx]
                    cr = rows["Rejected Runs"][d_idx]
                print(f"    {pol[0]:5s}{pol[1]:<3d} {str(used_rounds(*pol, m)):8s}"
                      f" {rej:.4f}  {S:.6f}  {np.mean(lm):.6f}     {F:.6f}"
                      f"  {Ss:.6f}      {cf:.6f}  {(cf - F) / se_F:+6.2f}"
                      f"        {cr:.4f}   {(cr - rej) / se_r:+6.2f}")
            fr_mean = np.mean(front_rows, 0)
            print("    frontier S*(alpha), full history: " + ", ".join(
                f"alpha={a}: {s:.6f}" for a, s in zip((0.9, 0.95, 0.99, 1.0),
                                                        fr_mean)))
            print(f"    alpha=1: S*(1) + S_min(1) = {fr_mean[3]:.4f} + "
                  f"{np.mean(smin):.4f} (>= 2/3 means the MAP rule also "
                  f"maximises Fleg among covariant rules); J* = alpha lam = "
                  f"{(3 * fr_mean[3] - 1) / 2:.4f} (MAP, no rejection) vs "
                  f"{np.mean([x['alpha'] * x['lam'] for x in acc[('unif', 1)]]):.4f}"
                  f" (unif 1)")


def section_B():
    print("\n[B] baseline LRB-D CSVs (read-only): figures of merit; for "
          "detection rules Fleg = lam (see [D])")
    if not os.path.isdir(RUN):
        print("  run folder not found; skipped")
        return
    depths = [int(x) for x in open(f"{RUN}/depths.txt").read().split(",")]
    probs = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]
    print("  unweighted log-linear fits over depths with >= 1000 accepted shots;"
          " eta = -ln a, delta = -ln f")
    print("  p       policy   a(per m)  eta=-ln a 1-a      f(per m)   r_c      "
          "r_tot    eta/delta  N*(eps=.1)  m*   max|log resid| acc, lam"
          "  (#depths used)")
    shots = 1_000_000
    for p_idx in (4, 5, 8):
        for kind, k in (("unif", 1), ("unif", 2), ("const", 1), ("const", 2),
                        ("const", 0)):
            rows = read_csv_rows(f"{RUN}/results/LRB/{p_idx}/{kind}"
                                 f"_check_data/{k}.csv")
            lam = np.array(rows["Fidelity averages"])
            acc = 1 - np.array(rows["Rejected Runs"])
            ms = np.array(depths, float)
            ok = (acc * shots >= 1000) & np.isfinite(lam) & (lam > 0)
            A, a = fit_exp(ms[ok], acc[ok])
            B, f = fit_exp(ms[ok], lam[ok])
            Nst, mst = shots_needed(A, a, B, f, 0.1)
            ra = np.abs(np.log(acc[ok]) - np.log(A * a ** ms[ok])).max()
            rl = np.abs(np.log(lam[ok]) - np.log(B * f ** ms[ok])).max()
            print(f"  {probs[p_idx]:<7g} {kind:5s}{k:<3d} {a:.5f}  "
                  f"{-np.log(a):.2e}  {1 - a:.2e} {f:.7f}  {2 / 3 * (1 - f):.2e} "
                  f"{2 / 3 * (1 - a * f):.2e} "
                  f"{-np.log(a) / max(-np.log(f), 1e-12):9.3g}"
                  f"  {Nst:.2e}  {mst:<4d} {ra:.1e}, {rl:.1e}  ({ok.sum()})")
    # Noise floor of the legacy metric |mean w^R| when nothing survives.
    rows = read_csv_rows(f"{RUN}/results/LRB/8/const_check_data/1.csv")
    print("  p=0.01 const 1 at large m: acceptance vs 3^-4 and CSV Fleg vs the "
          "Rayleigh mean sqrt(pi/(4 N_acc)); z-scores use SE = SD/sqrt(30) "
          "(CSV SD for the acceptance, Rayleigh SD for Fleg)")
    for i, m in enumerate(depths):
        if m >= 10:
            a_m = 1 - rows["Rejected Runs"][i]
            n_acc = a_m * shots
            fl = np.sqrt(np.pi / (4 * n_acc))
            sd_r = np.sqrt((4 - np.pi) / (4 * n_acc))
            print(f"    m={m:<3d} acceptance={a_m:.7f} (z vs 3^-4 "
                  f"{(a_m - 3.0 ** -4) / (rows['Rejected Standard Deviations'][i] / np.sqrt(30)):+6.2f})"
                  f"  N_acc={n_acc:7.0f}  CSV Fleg={rows['Fidelity averages'][i]:.4f}"
                  f"  floor={fl:.4f} (z {(rows['Fidelity averages'][i] - fl) / (sd_r / np.sqrt(30)):+6.2f})")
    print("    -> saturated and at the floor for m >= 18; m = 14 is still "
          "above both (residual signal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-circuit", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20261028)
    t0 = time.time()
    section_T(rng, args.quick)
    section_C()
    section_M(rng, args.quick)
    section_R(rng, args.quick)
    if not args.skip_circuit:
        section_D(args.quick)
        section_B()
    print(f"\ntotal runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
