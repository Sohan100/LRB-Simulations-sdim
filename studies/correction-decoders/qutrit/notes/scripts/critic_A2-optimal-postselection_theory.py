#!/usr/bin/env python3
"""Critic checks for A2-optimal-postselection: theory and metric claims.

Independent re-derivations (no code shared with A2_optimal_rule_check.py):

  [1] Theorem 1 (C1): LP optimum vs MAP-threshold frontier on random tables
      that include exact ties and atoms of q; the "iff" optimality condition
      (a rule that accepts a q<t* history or misses a q>t* one is strictly
      worse; a rule that breaks argmax ties differently is still optimal).
  [2] Chow error-reject identity E(t) = int_[t,1] (1-s) dR(s) (C2).
  [3] Metric identities (C7, C8) and the d=4 case (C7 says "fails for d>=5").
  [4] Example A (C11): the never-correct rule (2,1) is COVARIANT, so the
      notes' Sec. 3.5 bullet "within covariant rules under symmetric noise the
      covariant MAP-threshold rule maximises F_leg" is false in general.
  [5] Loss bound S*-S_L <= (1-S_L)^2/(4 lam_L) (C12) tested on tables where
      the loss is actually positive (the A2 test only used tables with zero
      loss, which makes it vacuous).
  [6] Variance of Re w^R (C13), exact; Rayleigh/Rice floor of |mean w^R|.
  [7] Shot-cost constants (C19): minima of the single-depth CR count, and
      the cost when B is NOT known (two-depth design, B and f both fitted).
  [8] C15 counterexample: raw-detector acceptance with measurement errors is
      a transfer-matrix sum of exponentials, not exactly A a^m.

Run: module load python/3.11 && python critic_A2-optimal-postselection_theory.py
(about 20 s, one core).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.special import i0e, i1e

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass

W3 = np.exp(2j * np.pi / 3)
rng = np.random.default_rng(777)


def hdr(s):
    print("\n" + s)


# ----------------------------------------------------------------- [1] C1
def frontier_beta(pD, post, alpha):
    """beta*(alpha) by sorting histories by q and filling greedily."""
    q = post.max(1)
    order = np.argsort(-q, kind="stable")
    beta, left = 0.0, alpha
    for i in order:
        take = min(pD[i], left)
        beta += take * q[i]
        left -= take
        if left <= 0:
            break
    return beta


def lp_beta(pD, post, alpha):
    """max sum_D sum_c x[D,c] pD q... as an LP over delta[D,c] in [0,1]."""
    n = len(pD)
    J = pD[:, None] * post
    c = -J.ravel()
    A_ub = np.zeros((n, 3 * n))
    for i in range(n):
        A_ub[i, 3 * i:3 * i + 3] = 1
    A_eq = np.repeat(pD, 3)[None, :]
    r = linprog(c, A_ub=A_ub, b_ub=np.ones(n), A_eq=A_eq, b_eq=[alpha],
                bounds=(0, 1), method="highs")
    assert r.status == 0
    return -r.fun


def rule_beta(pD, post, acc, corr):
    """beta and alpha of a rule with accept prob acc[D], correction corr[D]
    (corr may be a (n,3) mixing matrix)."""
    if corr.ndim == 1:
        corr = np.eye(3)[corr]
    alpha = float((pD * acc).sum())
    beta = float((pD * acc * (corr * post).sum(1)).sum())
    return alpha, beta


hdr("[1] Theorem 1: LP vs frontier, with ties and atoms in q")
worst = 0.0
for trial in range(300):
    n = int(rng.integers(2, 12))
    pD = rng.dirichlet(np.ones(n))
    # posteriors drawn from a small lattice so that exact ties/atoms occur
    post = rng.integers(0, 4, size=(n, 3)).astype(float) + 1e-300
    post[post.sum(1) < 0.5, 0] = 1.0
    post /= post.sum(1, keepdims=True)
    for alpha in rng.uniform(0, 1, 4):
        worst = max(worst, abs(lp_beta(pD, post, alpha)
                               - frontier_beta(pD, post, alpha)))
print(f"  max |LP - frontier| over 1200 (table, alpha) pairs with tied/atomic q:"
      f" {worst:.1e}")
# iff condition: perturb an optimal rule by swapping a little acceptance from
# a q>t* history to a q<t* history -> strictly lower beta at equal alpha.
pD = np.array([0.5, 0.3, 0.2])
post = np.array([[0.9, 0.05, 0.05], [0.6, 0.4, 0.0], [0.5, 0.25, 0.25]])
alpha = 0.6                       # t* = 0.6 (history 1 partially accepted)
b_opt = frontier_beta(pD, post, alpha)
a1, b1 = rule_beta(pD, post, np.array([1.0, 1 / 3, 0.0]), np.array([0, 0, 0]))
a2, b2 = rule_beta(pD, post, np.array([0.9, 0.5, 0.0]), np.array([0, 0, 0]))
a3, b3 = rule_beta(pD, post, np.array([1.0, 1 / 3, 0.0]),
                   np.array([[1, 0, 0], [1, 0, 0], [0.5, 0.5, 0]]))
print(f"  toy: beta*={b_opt:.6f}; threshold rule beta={b1:.6f} (alpha={a1:.3f});"
      f" rule taking from q>t* beta={b2:.6f} (alpha={a2:.3f}) -> strictly lower:"
      f" {b2 < b1 - 1e-12}")

# ----------------------------------------------------------------- [2] C2
hdr("[2] Chow identity: undetected error E(t) = int_[t,1](1-s) dR(s)")
n = 50
pD = rng.dirichlet(np.ones(n))
post = rng.dirichlet(np.ones(3) * 0.5, size=n)
post[:10] = post[10:20]           # duplicated rows -> atoms of the law of q
pD /= pD.sum()
q = post.max(1)
worst = 0.0
for t in np.concatenate([q, rng.uniform(1 / 3, 1, 20)]):
    E_direct = float((pD * (1 - q) * (q >= t)).sum())
    # R(s) = P(q < s); Lebesgue-Stieltjes integral over [t, 1]
    s_vals = np.unique(q)
    R = lambda s: float(pD[q < s].sum())
    E_stieltjes = sum((1 - s) * (R(np.nextafter(s, 2)) - R(s))
                      for s in s_vals if s >= t)
    worst = max(worst, abs(E_direct - E_stieltjes))
print(f"  max |E_direct - E_Stieltjes| = {worst:.1e} (atoms included)")
# slope of beta* = t*: numerical derivative
alphas = np.linspace(0.01, 0.99, 99)
bs = np.array([frontier_beta(pD, post, a) for a in alphas])
conc = np.all(np.diff(bs, 2) <= 1e-12)
S = bs / alphas
print(f"  beta* concave on grid: {conc}; S* nonincreasing: "
      f"{np.all(np.diff(S) <= 1e-12)}; S*(0+)={frontier_beta(pD, post, 1e-9) / 1e-9:.4f}"
      f" vs max q={q.max():.4f}; S*(1)={frontier_beta(pD, post, 1.0):.4f} vs"
      f" E[q]={(pD * q).sum():.4f}")

# ----------------------------------------------------------------- [3] C7/C8
hdr("[3] metric identities")
worst = [0.0, 0.0, 0.0]
for _ in range(10000):
    p = rng.dirichlet(np.ones(3) * rng.uniform(0.2, 5))
    z = (p * W3 ** np.arange(3)).sum()
    lam = (3 * p[0] - 1) / 2
    worst[0] = max(worst[0], abs(z.real - lam))
    worst[1] = max(worst[1], abs(abs(z) - math.sqrt((3 * (p ** 2).sum() - 1) / 2)))
    if lam > 0:
        worst[2] = max(worst[2], abs(z) - lam - 3 / 8 * (1 - p[0]) ** 2 / lam)
print(f"  max|Re z - (3S-1)/2| = {worst[0]:.1e}; max| |z| - sqrt((3Q-1)/2) |"
      f" = {worst[1]:.1e}; max(|z|-lam - (3/8)(1-S)^2/lam) = {worst[2]:.1e} (<=0)")
# d = 4: Re E[i^R] is NOT a function of p0 alone
W4 = 1j
pa, pb = np.array([0.7, 0.3, 0, 0]), np.array([0.7, 0, 0.3, 0])
print(f"  d=4: p=(0.7,0.3,0,0) -> Re E[i^R]={np.real((pa * W4 ** np.arange(4)).sum()):.2f};"
      f" p=(0.7,0,0.3,0) -> {np.real((pb * W4 ** np.arange(4)).sum()):.2f}"
      f"  (same S=0.7, different Re E): the identity fails for d=4 as well")
# d = 2
print("  d=2: Re E[(-1)^R] = 2S-1, a function of S: identity holds for d=2 too")

# ----------------------------------------------------------------- [4] C11
hdr("[4] Example A: covariance of the rules")
JA = 0.5 * np.array([[0.6, 0.4, 0.0], [0.6, 0.0, 0.4]])   # rows D, -D
for cD, cmD in itertools.product(range(3), repeat=2):
    pr = np.zeros(3)
    for row, c in ((0, cD), (1, cmD)):
        for l in range(3):
            pr[(l - c) % 3] += JA[row, l]
    z = (pr * W3 ** np.arange(3)).sum()
    cov = (cmD == (-cD) % 3)
    if cov:
        print(f"  covariant rule (c_D,c_-D)=({cD},{cmD}): S={pr[0]:.3f}"
              f" lam={(3 * pr[0] - 1) / 2:+.3f} Fleg={abs(z):.3f}")
print("  -> among COVARIANT rules, (2,1) has Fleg=0.5 > 0.4 of the MAP rule (0,0):"
      " Fleg=|lam| is maximised by the most NEGATIVE lam here.")

# ----------------------------------------------------------------- [5] C12
hdr("[5] loss bound S*-S_L <= (1-S_L)^2/(4 lam_L) on tables with positive loss")
nH = 5
allc = np.array(list(itertools.product(range(3), repeat=nH)))
n_pos, viol, ratio_max, n_lamneg = 0, 0, 0.0, 0
for _ in range(3000):
    pD = rng.dirichlet(np.ones(nH))
    post = rng.dirichlet(np.ones(3) * rng.uniform(0.3, 3), size=nH)
    J = pD[:, None] * post
    pr = np.stack([J[np.arange(nH)[None, :], (allc + r) % 3].sum(1)
                   for r in range(3)], 1)
    S = pr[:, 0]
    F = np.abs((pr * W3 ** np.arange(3)).sum(1))
    Sstar = S.max()
    for k in np.flatnonzero(F >= F.max() - 1e-13):   # every Fleg maximiser
        SL = S[k]
        lamL = (3 * SL - 1) / 2
        if lamL <= 0:
            n_lamneg += 1
            continue
        loss = Sstar - SL
        bound = (1 - SL) ** 2 / (4 * lamL)
        if loss > 1e-12:
            n_pos += 1
            ratio_max = max(ratio_max, loss / bound)
            viol += loss > bound + 1e-12
print(f"  maximisers with lam_L>0 and positive loss: {n_pos}; violations: {viol};"
      f" max loss/bound = {ratio_max:.3f}; maximisers with lam_L<=0: {n_lamneg}")

# ----------------------------------------------------------------- [6] C13
hdr("[6] estimator variance and legacy floor")
for p in ([0.9, 0.06, 0.04], [0.5, 0.25, 0.25], [0.995, 0.005, 0.0]):
    p = np.array(p)
    x = np.real(W3 ** np.arange(3))
    var_exact = (p * x ** 2).sum() - (p * x).sum() ** 2
    lam = (3 * p[0] - 1) / 2
    print(f"  p={p}: exact Var(Re w^R)={var_exact:.6f}  formula={(1 + lam) / 2 - lam ** 2:.6f}")
# Rice mean of |z_hat| for true |z| = nu, per-component variance 1/(2N)


def rice_mean(nu, sigma):
    x = nu ** 2 / (2 * sigma ** 2)
    # E = sigma sqrt(pi/2) L_{1/2}(-x),  L_{1/2}(-x)=e^{-x/2}[(1+x)I0(x/2)+x I1(x/2)]
    return sigma * math.sqrt(math.pi / 2) * ((1 + x) * i0e(x / 2) + x * i1e(x / 2))


N = 12406
sig = math.sqrt(1 / (2 * N))
print(f"  N_acc={N}: Rayleigh mean {rice_mean(0, sig):.5f} (=sqrt(pi/4N)="
      f"{math.sqrt(math.pi / (4 * N)):.5f}), Rayleigh SD {sig * math.sqrt((4 - math.pi) / 2):.5f}"
      f" per circuit, {sig * math.sqrt((4 - math.pi) / 2) / math.sqrt(30):.5f} for a 30-mean")
nu = minimize_scalar(lambda v: (rice_mean(v, sig) - 0.010473) ** 2,
                     bounds=(0, 0.05), method="bounded").x
print(f"  CSV const1 p=0.01 m=14 mean 0.010473 is matched by a Rice mean with"
      f" true |z| = {nu:.4f}, i.e. NOT a pure floor")

# ----------------------------------------------------------------- [7] C19
hdr("[7] shot-cost constants and the effect of an unknown B")


def N_single(m, eta, dl, B=1.0, A=1.0, eps=0.1):
    f = math.exp(-dl)
    lam = B * f ** m
    V = (1 + lam) / 2 - lam ** 2
    return V * f ** 2 / (A * math.exp(-eta * m) * m ** 2 * lam ** 2 * eps ** 2 * (1 - f) ** 2)


g = lambda x: (math.exp(x) - 1) * (math.exp(x) + 2) / (2 * x * x)
res = minimize_scalar(g, bounds=(0.1, 3), method="bounded")
print(f"  eta=0: min_x (e^x-1)(e^x+2)/(2x^2) = {res.fun:.4f} at x = {res.x:.3f}"
      "  (notes: 4.0 at 0.9/delta)")
print(f"  rejection-dominated constant 3e/2 = {1.5 * math.e:.4f}; offset constant"
      f" 3e^2/8 = {3 * math.e ** 2 / 8:.4f}")


def N_two_depth(eta, dl, B, A=1.0, eps=0.1, mmax=200):
    """Submitted shots for rel. precision eps on 1-f when B and f are both
    unknown: depths 0 and m, Fisher information summed, optimal split."""
    f = math.exp(-dl)
    best = (math.inf, None, None)
    for m in range(1, mmax + 1):
        for w0 in np.linspace(0.02, 0.98, 49):     # fraction of shots at m=0
            I = np.zeros((2, 2))
            for mm, w in ((0, w0), (m, 1 - w0)):
                lam = B * f ** mm
                V = (1 + lam) / 2 - lam ** 2
                acc = A * math.exp(-eta * mm)
                grad = np.array([f ** mm, B * mm * f ** (mm - 1) if mm else 0.0])
                I += w * acc / V * np.outer(grad, grad)
            var_f = np.linalg.inv(I)[1, 1]          # per submitted shot
            Nn = var_f / (eps * (1 - f)) ** 2
            if Nn < best[0]:
                best = (Nn, m, w0)
    return best


for (eta, dl, B) in ((0.1117, 2.73e-5, 0.99245), (0.2231, 1.146e-4, 0.985)):
    Nk = min((N_single(m, eta, dl, B), m) for m in range(1, 400))
    N2, m2, w0 = N_two_depth(eta, dl, B)
    print(f"  eta={eta} delta={dl} B={B}: B known N*={Nk[0]:.3e} (m*={Nk[1]});"
          f" B unknown (depths 0 and m) N*={N2:.3e} (m={m2}, {w0:.0%} of shots at m=0),"
          f" ratio {N2 / Nk[0]:.2f}")

# ----------------------------------------------------------------- [8] C15
hdr("[8] C15: raw-detector acceptance with measurement errors (one Z_3 check)")
pd, pm = 0.02, 0.05        # data-syndrome flip and measurement flip per round
Pdelta = np.array([1 - pd, pd / 2, pd / 2])
Pe = np.array([1 - pm, pm / 2, pm / 2])
# hidden state x = accumulated data syndrome; accept round iff x + e = 0
T = np.zeros((3, 3))
for x in range(3):
    for xp in range(3):
        T[x, xp] = Pdelta[(xp - x) % 3] * Pe[(-xp) % 3]
ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
ms = np.arange(0, 23)
v0 = np.array([1.0, 0, 0])
acc = np.array([v0 @ np.linalg.matrix_power(T, m + 1) @ np.ones(3) for m in ms])
sl, ic = np.polyfit(ms, np.log(acc), 1)
resid = np.log(acc) - (ic + sl * ms)
print(f"  eigenvalues |.| of transfer matrix: {np.round(ev, 6)}")
print(f"  acceptance(m) for m=0..22 vs best single exponential: max |log resid|"
      f" = {np.abs(resid).max():.2e}  (0 would mean exactly A a^m)")
pm = 0.0
Pe = np.array([1.0, 0, 0])
for x in range(3):
    for xp in range(3):
        T[x, xp] = Pdelta[(xp - x) % 3] * Pe[(-xp) % 3]
acc = np.array([v0 @ np.linalg.matrix_power(T, m + 1) @ np.ones(3) for m in ms])
sl, ic = np.polyfit(ms, np.log(acc), 1)
print(f"  same with perfect measurements: max |log resid| ="
      f" {np.abs(np.log(acc) - (ic + sl * ms)).max():.1e}")
