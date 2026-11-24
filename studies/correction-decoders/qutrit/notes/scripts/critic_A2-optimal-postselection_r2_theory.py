#!/usr/bin/env python3
"""Critic (round 2) of A2-optimal-postselection: independent theory checks.

Independent re-implementation (no code shared with A2_optimal_rule_check.py).
Notation: joint[D, l] = P(D, L=l); a rule delta[D, c] = P(accept and apply
correction c | D); residual R = L - c; alpha = P(acc); beta = P(acc, R=0).

Sections
  T1  Theorem 1 (C1): LP optimum vs sorted-q frontier on random tables that
      contain forced ties (equal q across histories, ties inside the argmax);
      LP vertex solutions satisfy optimality conditions (a) and (b).
  T2  Frontier shape (C2): concavity, S* monotone, endpoints, Chow relation.
  T3  Metric identities (C7, C8).
  T4  Examples A/B and random diffuse tables (C11), own RNG.
  T5  Legacy-metric maximiser vs survival at FIXED alpha < 1 by exact vertex
      enumeration; loss bound (C12) tested on cases with NONZERO loss;
      aligned-posterior statement tested at alpha < 1.
  T6  Estimators (C13): variance identity, Rayleigh floor, bias formula.
  T7  Local rules (C15): exact i.i.d. toy with a biased (non-covariant) rule;
      LRB-like structure with an unchecked second half is NOT exponential.
  T8  Finite-window counterexample (C18).
  T9  Shot-cost formulas (C19) incl. a two-depth design with B unknown.
  T10 Plug-in regret bound (C22) incl. tightness.
  T11 Forney's likelihood-ratio rule vs posterior threshold with unequal
      priors (C3 caveat).

Run: module load python/3.11 && python critic_A2-optimal-postselection_r2_theory.py
(one core, < 1 min)
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import linprog, minimize_scalar

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass

W = np.exp(2j * np.pi / 3)
rng = np.random.default_rng(424242)


def ok(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    return cond


def qstats(J):
    pD = J.sum(1)
    q = J.max(1) / pD
    return pD, q


def beta_star(J, alpha):
    """beta*(alpha) by sorting q decreasingly (frontier = LP relaxation)."""
    pD, q = qstats(J)
    o = np.argsort(-q, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(pD[o])])
    val = np.concatenate([[0.0], np.cumsum((pD * q)[o])])
    return float(np.interp(alpha, cum, val))


def residual_dist(J, delta):
    """pr[r] = P(acc, R=r); delta[D, c]."""
    n = J.shape[0]
    pr = np.zeros(3)
    for c in range(3):
        for r in range(3):
            pr[r] += float(delta[:, c] @ J[:, (c + r) % 3])
    return pr


def zval(pr):
    a = pr.sum()
    return complex((pr / a * W ** np.arange(3)).sum()), a


def lp_solve(J, alpha):
    n = J.shape[0]
    pD = J.sum(1)
    res = linprog(-J.ravel(), A_ub=np.kron(np.eye(n), np.ones((1, 3))),
                  b_ub=np.ones(n), A_eq=np.repeat(pD, 3)[None, :],
                  b_eq=[alpha], bounds=(0, 1), method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    assert res.status == 0
    return -res.fun, res.x.reshape(n, 3)


# ----------------------------------------------------------------- T1
def tied_table(rng):
    """Random table with forced ties: posteriors drawn from a small pool that
    contains argmax ties; masses drawn so that several histories share q."""
    pool = [np.array(v) for v in ([0.5, 0.25, 0.25], [0.4, 0.4, 0.2],
                                  [0.4, 0.2, 0.4], [0.9, 0.05, 0.05],
                                  [0.2, 0.7, 0.1], [1 / 3, 1 / 3, 1 / 3],
                                  [0.45, 0.1, 0.45])]
    n = int(rng.integers(3, 13))
    post = np.array([pool[i] for i in rng.integers(0, len(pool), n)])
    if rng.uniform() < 0.5:     # also some generic rows
        k = int(rng.integers(0, n))
        post[k] = rng.dirichlet(np.ones(3))
    pD = rng.dirichlet(np.ones(n))
    return pD[:, None] * post


def section_T1():
    print("\nT1  Theorem 1: LP optimum vs frontier, with forced ties")
    worst, viol_a, viol_b, nsol = 0.0, 0, 0, 0
    for _ in range(300):
        J = tied_table(rng)
        pD, q = qstats(J)
        for alpha in rng.uniform(0.02, 1.0, 4):
            lp, x = lp_solve(J, alpha)
            worst = max(worst, abs(lp - beta_star(J, alpha)))
            # optimality conditions on the LP solution
            o = np.argsort(-q, kind="stable")
            cum = np.cumsum(pD[o])
            tstar = q[o][np.searchsorted(cum, alpha - 1e-12)]
            acc = x.sum(1)
            post = J / pD[:, None]
            for D in range(len(pD)):
                nonarg = post[D] < q[D] - 1e-12
                if np.any(x[D, nonarg] > 1e-8):
                    viol_a += 1
                if q[D] > tstar + 1e-12 and acc[D] < 1 - 1e-8:
                    viol_b += 1
                if q[D] < tstar - 1e-12 and acc[D] > 1e-8:
                    viol_b += 1
            nsol += 1
    ok("LP optimum == sorted-q frontier (tied tables)", worst < 1e-9,
       f"(max |diff| {worst:.1e}, {nsol} LPs)")
    ok("LP optimal vertices satisfy (a) argmax support", viol_a == 0,
       f"({viol_a} violations)")
    ok("LP optimal vertices satisfy (b) accept q>t*, reject q<t*",
       viol_b == 0, f"({viol_b} violations)")
    # Random feasible rules never exceed the frontier (own generator).
    exc = -1.0
    for _ in range(200):
        J = tied_table(rng)
        n = J.shape[0]
        for _ in range(100):
            acc = rng.uniform(size=n) * (rng.uniform(size=n) < 0.7)
            mix = rng.dirichlet(np.ones(3) * 0.5, size=n)
            pr = residual_dist(J, acc[:, None] * mix)
            exc = max(exc, pr[0] - beta_star(J, pr.sum()))
    ok("20,000 random randomised rules never above frontier", exc < 1e-14,
       f"(max excess {exc:.1e})")


# ----------------------------------------------------------------- T2
def section_T2():
    print("\nT2  Frontier shape (C2) and Chow's error-reject relation")
    bad_conc = bad_mono = bad_end = bad_chow = 0
    for _ in range(300):
        n = int(rng.integers(3, 20))
        J = rng.dirichlet(np.ones(n))[:, None] * rng.dirichlet(
            np.ones(3) * 0.7, size=n)
        pD, q = qstats(J)
        o = np.argsort(-q)
        cum = np.concatenate([[0.0], np.cumsum(pD[o])])
        val = np.concatenate([[0.0], np.cumsum((pD * q)[o])])
        slopes = np.diff(val) / np.diff(cum)
        bad_conc += int(np.any(np.diff(slopes) > 1e-12))
        al = np.linspace(1e-6, 1, 400)
        S = np.array([beta_star(J, a) / a for a in al])
        bad_mono += int(np.any(np.diff(S) > 1e-12))
        bad_end += int(abs(S[-1] - (pD * q).sum()) > 1e-12
                       or abs(S[0] - q.max()) > 1e-6)
        # Chow: E(t) = sum_{q>=t} pD (1-q) equals int_[t,1] (1-s) dR(s), with
        # R(s) = P(q < s) (left-continuous); dR has atoms pD at the q values.
        for t in np.unique(q):
            lhs = (pD * (1 - q))[q >= t].sum()
            atoms = {}
            for qq, m in zip(q, pD):
                atoms[qq] = atoms.get(qq, 0.0) + m
            rhs = sum((1 - s) * m for s, m in atoms.items() if s >= t)
            bad_chow += int(abs(lhs - rhs) > 1e-14)
    ok("beta* concave (slopes nonincreasing)", bad_conc == 0)
    ok("S* nonincreasing in alpha", bad_mono == 0)
    ok("S*(0+) = max q and S*(1) = E[q]", bad_end == 0)
    ok("E(t) = int_[t,1](1-s)dR(s) (closed interval incl. atom at t)",
       bad_chow == 0)


# ----------------------------------------------------------------- T3
def section_T3():
    print("\nT3  Metric identities (C7, C8)")
    e1 = e2 = e3 = e4 = 0.0
    viol = 0
    for _ in range(20000):
        p = rng.dirichlet(np.ones(3) * rng.uniform(0.05, 3))
        z = (p * W ** np.arange(3)).sum()
        lam = (3 * p[0] - 1) / 2
        Q = (p ** 2).sum()
        e1 = max(e1, abs(z.real - lam))
        e2 = max(e2, abs(abs(z) ** 2 - (3 * Q - 1) / 2))
        e3 = max(e3, abs(abs(z) ** 2 - lam ** 2 - 0.75 * (p[1] - p[2]) ** 2))
        for s in (1, 2):   # shift invariance and conjugation invariance
            ps = np.roll(p, s)
            e4 = max(e4, abs(abs((ps * W ** np.arange(3)).sum()) - abs(z)))
        pc = p[[0, 2, 1]]
        e4 = max(e4, abs(abs((pc * W ** np.arange(3)).sum()) - abs(z)))
        if lam > 0 and abs(z) - lam > 3 / 8 * (1 - p[0]) ** 2 / lam + 1e-14:
            viol += 1
    ok("Re E w^R = (3S-1)/2", e1 < 1e-14, f"({e1:.1e})")
    ok("|E w^R|^2 = (3Q-1)/2", e2 < 1e-14, f"({e2:.1e})")
    ok("|z|^2 = lam^2 + (3/4)(p1-p2)^2", e3 < 1e-14, f"({e3:.1e})")
    ok("|z| invariant under r->r+s and r->-r", e4 < 1e-14, f"({e4:.1e})")
    ok("|z| - lam <= (3/8)(1-S)^2/lam for lam>0", viol == 0, f"({viol})")
    # d-dependence of Re E w^R being a function of p0 alone
    for d in (2, 3, 4, 5, 7):
        vals = {round(math.cos(2 * math.pi * r / d), 12) for r in range(1, d)}
        print(f"    d={d}: distinct cos(2 pi r/d), r!=0: {len(vals)} "
              f"-> Re E w^R is a function of p0 alone: {len(vals) == 1}")


# ----------------------------------------------------------------- T4
def section_T4():
    print("\nT4  Legacy metric vs survival (C11)")
    J = 0.5 * np.array([[0.6, 0.4, 0.0], [0.6, 0.0, 0.4]])
    rows = []
    for c1, c2 in itertools.product(range(3), repeat=2):
        d = np.zeros((2, 3))
        d[0, c1] = d[1, c2] = 1
        pr = residual_dist(J, d)
        z, a = zval(pr)
        rows.append(((c1, c2), pr[0] / a, z.real, abs(z)))
    best_F = max(r[3] for r in rows)
    print("    Example A maps with max Fleg:",
          [(r[0], round(r[1], 3), round(r[3], 3)) for r in rows
           if abs(r[3] - best_F) < 1e-12])
    ok("Example A: MAP (0,0) has S=0.6, Fleg=0.4; (2,1) has S=0, Fleg=0.5",
       abs(rows[0][1] - 0.6) < 1e-12 and abs(rows[0][3] - 0.4) < 1e-12
       and abs(rows[7][1]) < 1e-12 and abs(rows[7][3] - 0.5) < 1e-12)
    for post in ([0.70, 0.15, 0.15], [0.65, 0.35, 0.0]):
        z = (np.array(post) * W ** np.arange(3)).sum()
        print(f"    Example B posterior {post}: S={post[0]:.3f} "
              f"lam={(3 * post[0] - 1) / 2:.4f} Fleg={abs(z):.4f}")
    # Covariant version of Example B (accept D and -D together).
    pc = 0.5 * np.array([0.65, 0.35, 0]) + 0.5 * np.array([0.65, 0, 0.35])
    print(f"    Example B made covariant (accept D and -D): Fleg="
          f"{abs((pc * W ** np.arange(3)).sum()):.4f} < 0.55 -> no reversal")
    # Random diffuse tables, alpha = 1, all 3^5 maps (own RNG).
    n, ntab = 5, 2000
    allc = np.array(list(itertools.product(range(3), repeat=n)))
    lossy, losses = 0, []
    for _ in range(ntab):
        J = rng.dirichlet(np.ones(n))[:, None] * rng.dirichlet(np.ones(3),
                                                               size=n)
        pr = np.stack([J[np.arange(n)[None, :], (allc + r) % 3].sum(1)
                       for r in range(3)], 1)
        S = pr[:, 0]
        F = np.abs((pr * W ** np.arange(3)).sum(1))
        SL = S[F >= F.max() - 1e-13].max()
        losses.append(S.max() - SL)
    losses = np.array(losses)
    print(f"    diffuse tables (own RNG): Fleg maximiser has lower S in "
          f"{np.mean(losses > 1e-12):.1%} of {ntab}; max loss {losses.max():.3f}")


# ----------------------------------------------------------------- T5
def vertices_fixed_alpha(J, alpha):
    """All vertices of {delta: Sum_c delta <= 1, Sum_D pD a = alpha}.
    Yields (accept_vector, correction_vector) with fractional acceptance on
    at most one history."""
    n = J.shape[0]
    pD = J.sum(1)
    choices = list(itertools.product(range(4), repeat=n))   # 0=reject, k=c+1
    out = []
    for ch in choices:
        ch = np.array(ch)
        a = (ch > 0).astype(float)
        tot = (pD * a).sum()
        if abs(tot - alpha) < 1e-12:
            out.append((a, np.clip(ch - 1, 0, 2)))
        # fractional history f between reject and accept-with-c
        for f in range(n):
            if ch[f] == 0:
                continue
            rest = tot - pD[f]
            th = (alpha - rest) / pD[f]
            if 1e-12 < th < 1 - 1e-12:
                a2 = a.copy()
                a2[f] = th
                out.append((a2, np.clip(ch - 1, 0, 2)))
    return out


def section_T5():
    print("\nT5  Legacy maximiser at fixed alpha (exact vertex enumeration), "
          "C12")
    n = 4
    nlossy, viol, ntest, worst_ratio = 0, 0, 0, 0.0
    al_viol = 0
    for it in range(400):
        aligned = it % 4 == 0
        pD = rng.dirichlet(np.ones(n))
        if aligned:     # two non-maximal entries equal
            qv = rng.uniform(1 / 3, 1, n)
            post = np.zeros((n, 3))
            for D in range(n):
                c = rng.integers(0, 3)
                post[D] = (1 - qv[D]) / 2
                post[D, c] = qv[D]
        else:
            post = rng.dirichlet(np.ones(3) * rng.choice([0.5, 1.0, 3.0]),
                                 size=n)
        J = pD[:, None] * post
        alpha = float(rng.uniform(0.15, 1.0))
        best = []
        for a, c in vertices_fixed_alpha(J, alpha):
            d = np.zeros((n, 3))
            d[np.arange(n), c] = a
            pr = residual_dist(J, d)
            z, aa = zval(pr)
            best.append((abs(z), z.real, pr[0] / aa))
        if not best:
            continue
        best = np.array(best)
        Fmax = best[:, 0].max()
        lam_star = best[:, 1].max()
        Sstar = best[:, 2].max()
        assert abs(Sstar - beta_star(J, alpha) / alpha) < 1e-9
        sel = best[best[:, 0] >= Fmax - 1e-12]
        k = np.argmax(sel[:, 1])
        lamL, SL = sel[k, 1], sel[k, 2]
        if aligned:
            al_viol += int(abs(Fmax - lam_star) > 1e-10)
            continue
        ntest += 1
        loss = Sstar - SL
        if loss > 1e-10:
            nlossy += 1
            if lamL > 0:
                bound = 0.25 * (1 - SL) ** 2 / lamL
                worst_ratio = max(worst_ratio, loss / bound)
                viol += int(loss > bound + 1e-12)
    ok("aligned posteriors: max Fleg == lam* at random alpha<=1", al_viol == 0,
       f"({al_viol} violations in 100 tables)")
    ok("loss bound S*-S_L <= (1-S_L)^2/(4 lam_L) where loss>0", viol == 0,
       f"({nlossy} of {ntest} tables had loss>0 at fixed alpha; "
       f"max loss/bound = {worst_ratio:.3f})")


# ----------------------------------------------------------------- T6
def section_T6():
    print("\nT6  Estimators (C13)")
    p = np.array([0.8, 0.12, 0.08])
    lam = (3 * p[0] - 1) / 2
    x = np.real(W ** rng.choice(3, size=2_000_000, p=p))
    print(f"    Var(Re w^R): MC {x.var():.5f} vs (1+lam)/2-lam^2 = "
          f"{(1 + lam) / 2 - lam ** 2:.5f} (p1 != p2 case)")
    N, reps = 12300, 4000
    R = rng.choice(3, size=(reps, N))
    F = np.abs((W ** R).mean(1))
    floor = math.sqrt(math.pi / (4 * N))
    sd = math.sqrt(1 / (2 * N)) * math.sqrt((4 - math.pi) / 2)
    print(f"    lam=0, N={N}: mean |mean w^R| = {F.mean():.5f} vs "
          f"sqrt(pi/4N) = {floor:.5f}; per-circuit SD {F.std():.5f} vs "
          f"{sd:.5f}")
    lam1 = 0.05
    p1 = np.array([(1 + 2 * lam1) / 3, (1 - lam1) / 3, (1 - lam1) / 3])
    R = rng.choice(3, size=(reps, N), p=p1)
    F = np.abs((W ** R).mean(1))
    print(f"    lam={lam1}, N={N}: mean |mean w^R| - lam = "
          f"{F.mean() - lam1:.2e} vs (1-lam)/(4 N lam) = "
          f"{(1 - lam1) / (4 * N * lam1):.2e}")


# ----------------------------------------------------------------- T7
def section_T7():
    print("\nT7  Local rules (C15)")
    # per-step law: 3 histories, conjugation symmetric pair (D, -D) + D=0
    pi1 = np.array([[0.90, 0.01, 0.01],    # D = 0
                    [0.03, 0.02, 0.005],   # D
                    [0.03, 0.005, 0.02]])  # -D (mirror of D)
    pi1 /= pi1.sum()
    # covariant MAP (no rejection) and a biased rule (c=1 on D and on -D)
    for name, corr in (("covariant MAP", [0, 0, 0]),
                       ("biased: c=+1 on D and -D", [0, 1, 1])):
        t = sum(pi1[D, l] * W ** ((l - corr[D]) % 3)
                for D in range(3) for l in range(3))
        ms = np.arange(0, 41, 5)
        lam = np.real(t ** ms)
        Fl = np.abs(t ** ms)
        print(f"    {name}: t={t:.5f} |t|={abs(t):.5f} theta={np.angle(t):+.4f};"
              f" lam(m)={np.round(lam, 4).tolist()}")
        print(f"      Fleg(m)={np.round(Fl, 4).tolist()}")
    # LRB-like structure: m checked steps, then m unchecked steps, one final
    # check of the accumulated raw syndrome (toy per-step syndrome law).
    s_law = {0: 0.93, 1: 0.035, 2: 0.035}     # 1 trit syndrome increment
    a = s_law[0]
    ms = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22])
    phi = s_law[0] + (s_law[1] + s_law[2]) * (-0.5)   # E w^{k s}, k != 0
    acc = a ** ms * (1 + 2 * phi ** ms) / 3      # P(sum of m increments = 0)
    sl, ic = np.polyfit(ms, np.log(acc), 1)
    res = np.abs(np.log(acc) - (ic + sl * ms)).max()
    print(f"    checked half + unchecked half + final check: alpha(m) = "
          f"a^m (1+2 phi^m)/3, not one exponential; log-residual of best "
          f"exponential fit {res:.2e}")
    # Every round checked, raw detectors, measurement errors: hidden state s
    # (cumulative data syndrome); accept round j iff raw D_j = s_j + e_j = 0.
    for pdat, pmea in ((0.05, 0.02), (0.01, 0.005)):
        Pd = np.array([1 - pdat, pdat / 2, pdat / 2])      # increment law
        Pm = np.array([1 - pmea, pmea / 2, pmea / 2])      # meas-error law
        T = np.zeros((3, 3))
        for s in range(3):
            for s2 in range(3):
                T[s2, s] = Pd[(s2 - s) % 3] * Pm[(-s2) % 3]
        ms = np.arange(1, 23)
        v0 = np.array([1.0, 0, 0])
        acc = np.array([np.linalg.matrix_power(T, mm) @ v0 for mm in ms]).sum(1)
        sl, ic = np.polyfit(ms, np.log(acc), 1)
        res = np.abs(np.log(acc) - (ic + sl * ms)).max()
        ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        print(f"    every round checked, raw detectors, p_data={pdat}, "
              f"p_meas={pmea}: eigenvalue ratio {ev[1] / ev[0]:.3f}, "
              f"log-residual of single exponential {res:.1e}")


# ----------------------------------------------------------------- T8
def section_T8():
    print("\nT8  Finite-window counterexample (C18)")
    m = np.arange(0, 23)
    d = 0.99 ** m >= 0.8 * 0.995 ** m
    cross = math.log(0.8) / math.log(0.99 / 0.995)
    ok("0.99^m >= 0.8*0.995^m on [0,22]", bool(d.all()),
       f"(crossing at m={cross:.1f})")


# ----------------------------------------------------------------- T9
def Nm(m, A, a, B, f, eps):
    lam = B * f ** m
    V = (1 + lam) / 2 - lam ** 2
    return V * f ** 2 / (A * a ** m * m ** 2 * lam ** 2 * eps ** 2
                         * (1 - f) ** 2)


def section_T9():
    print("\nT9  Shot cost (C19)")
    g = lambda x: (math.exp(x) - 1) * (math.exp(x) + 2) / (2 * x * x)
    r = minimize_scalar(g, bounds=(0.1, 3), method="bounded")
    print(f"    eta=0, delta->0: min_x (e^x-1)(e^x+2)/(2x^2) = {r.fun:.4f} at "
          f"x = delta m* = {r.x:.3f}")
    # closed form vs general formula
    eta, dl, eps = 0.1, 1e-3, 0.1
    m = 7.0
    cf = (math.exp(eta * m) * (math.exp(dl * m) - 1) * (math.exp(dl * m) + 2)
          / (2 * eps ** 2 * m ** 2 * (math.exp(dl) - 1) ** 2))
    gen = Nm(m, 1.0, math.exp(-eta), 1.0, math.exp(-dl), eps)
    ok("closed form N(m) (A=B=1) == general formula", abs(cf / gen - 1) < 1e-12)
    # Baseline unif 1, p=1e-3: unweighted log-linear fits to the CSV (read
    # only), same depth selection as the notes (all 9 depths).
    run = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
           "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
    rows = {}
    with open(f"{run}/results/LRB/4/unif_check_data/1.csv") as fh:
        for line in fh:
            k, *v = line.strip().split(",")
            rows[k] = np.array([float(x) for x in v])
    dep = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22], float)
    sa, ia = np.polyfit(dep, np.log(1 - rows["Rejected Runs"]), 1)
    sb, ib = np.polyfit(dep, np.log(rows["Fidelity averages"]), 1)
    A, a, B, f = math.exp(ia), math.exp(sa), math.exp(ib), math.exp(sb)
    print(f"    baseline unif 1 p=1e-3 fit: A={A:.4f} a={a:.5f} B={B:.5f} "
          f"f={f:.7f}")
    A, B = min(A, 1), min(B, 1)
    ms = np.arange(1, 400)
    N1 = np.array([Nm(mm, A, a, B, f, 0.1) for mm in ms])
    k = int(np.argmin(N1))
    print(f"    single depth, B known: N* = {N1[k]:.3e} at m* = {ms[k]}")
    # Two-depth design, B unknown: slope from depths 0 and m1, optimal split.
    best = (np.inf, None)
    for m1 in range(2, 120):
        c = []
        for mm in (0, m1):
            lam = B * f ** mm
            V = (1 + lam) / 2 - lam ** 2
            c.append(V / (A * a ** mm * lam ** 2))   # Var(log lam_hat)*N_i
        # Var(slope) = (c0/N0 + c1/N1)/m1^2; min N0+N1 s.t. Var = target
        target = (0.1 * (1 - f) / f) ** 2       # Var(log f_hat) target
        Ntot = (math.sqrt(c[0]) + math.sqrt(c[1])) ** 2 / (m1 ** 2 * target)
        if Ntot < best[0]:
            best = (Ntot, m1)
    print(f"    two depths (0, m1), B unknown, optimal split: N* = "
          f"{best[0]:.3e} at m1 = {best[1]}  (ratio {best[0] / N1[k]:.2f})")


# ----------------------------------------------------------------- T10
def section_T10():
    print("\nT10 Plug-in regret bound (C22)")
    worst_ratio, viol = 0.0, 0
    for _ in range(3000):
        n = int(rng.integers(3, 15))
        pD = rng.dirichlet(np.ones(n))
        post = rng.dirichlet(np.ones(3) * rng.uniform(0.2, 3), size=n)
        J = pD[:, None] * post
        noise = rng.normal(0, rng.uniform(0.01, 1.5), size=(n, 3))
        pt = post * np.exp(noise)
        pt /= pt.sum(1, keepdims=True)
        Delta = (pD * np.abs(post - pt).max(1)).sum()
        alpha = rng.uniform(0.05, 1.0)
        # threshold rule built from the model posterior, acceptance alpha
        # under the TRUE marginal pD (randomise on the last history)
        qt = pt.max(1)
        ct = pt.argmax(1)
        o = np.argsort(-qt, kind="stable")
        acc = np.zeros(n)
        left = alpha
        for D in o:
            take = min(1.0, left / pD[D])
            acc[D] = take
            left -= take * pD[D]
            if left <= 1e-15:
                break
        beta_t = (acc * J[np.arange(n), ct]).sum()
        reg = beta_star(J, alpha) - beta_t
        if Delta > 0:
            worst_ratio = max(worst_ratio, reg / (2 * Delta))
        viol += int(reg > 2 * Delta + 1e-14)
    ok("beta*(alpha) - beta(tilde delta) <= 2 Delta", viol == 0,
       f"(max regret/(2 Delta) = {worst_ratio:.3f} over 3000 tables)")


# ----------------------------------------------------------------- T11
def section_T11():
    print("\nT11 Forney's rule with unequal priors (C3 caveat)")
    # Two histories, three logical classes, prior P(L=0)=0.98.
    prior = np.array([0.98, 0.01, 0.01])
    like = np.array([[0.90, 0.05, 0.05],     # P(y|l), y=y1
                     [0.30, 0.95, 0.20]])    # y=y2 (columns l)
    for y in range(2):
        post = prior * like[y] / (prior * like[y]).sum()
        m = int(np.argmax(post))
        forney = like[y, m] / (like[y].sum() - like[y, m])
        m_lik = int(np.argmax(like[y]))
        forney_ml = like[y, m_lik] / (like[y].sum() - like[y, m_lik])
        print(f"    y{y + 1}: posterior {np.round(post, 4)} MAP={m}; "
              f"max-likelihood class={m_lik}, Forney ratio={forney_ml:.3f}; "
              f"posterior odds q/(1-q)={post[m] / (1 - post[m]):.3f}")
    print("    -> without equal priors the likelihood-ratio form of Forney's "
          "rule ranks/labels differently; the posterior (prior-weighted) form "
          "is what Theorem 1 uses.")


def main():
    section_T1()
    section_T2()
    section_T3()
    section_T4()
    section_T5()
    section_T6()
    section_T7()
    section_T8()
    section_T9()
    section_T10()
    section_T11()


if __name__ == "__main__":
    main()
