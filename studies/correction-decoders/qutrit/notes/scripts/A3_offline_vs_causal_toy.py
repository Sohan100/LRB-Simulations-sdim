#!/usr/bin/env python3
"""A3 toy model: offline (global) versus causal (committing, lookahead w) decoding.

Classical Z_3 repetition chain with measurement errors, used as a stand-in for
the time structure of the qutrit LRB-D circuit.  Nothing here reads or writes
repository data.

Model
  * n data trits, accumulated error x_t in Z_3^n, x_0 = 0 (ideal encoding).
  * schedule elements
      C : data-error step (each trit +-1 with prob q, each sign q/2) followed by
          a noisy check round s = H x + e_m, (H x)_j = x_j - x_{j+1}, every check
          independently wrong by +-1 with prob q_m;
      U : data-error step with prob q_u per trit and no check round (stand-in
          for an inverse Clifford, which has no check after it);
      Z : error on the readout trit only, prob q_post, never checked (stand-in
          for faults on wires 0 and 3 after their last check interaction in
          the final round);
      P : perfect readout of all data (reveals H x exactly).
  * readout: the logical shift is x[0] (error on the readout trit); a decoder
    outputs L_hat; fidelity F = |E w^(x[0] - L_hat)|, w = exp(2 pi i/3), the
    metric of the LRB-D runs.

Schedules (boundaries)
  memory-perfect : C^T P       (standard memory boundary, perfect last syndrome)
  memory-noisy   : C^T         (last round noisy, no successor, logical readout)
  lrb            : C^m U^m C Z (forward steps with checks, unchecked inverse
                                block, final noisy round m, invisible faults,
                                logical readout only; m+1 rounds as in LRB-D)

Decoders (all use the exact model, as the DEM decoder does)
  none     : L_hat = 0.
  offline  : exact Bayesian MAP of x_final[0] from the whole record.  Because
             only the final state is read out, this needs only the forward
             filter: it is also the causal decoder that never commits.
  causal w : sliding window with irrevocable commits: the state of the first
             uncommitted step is committed (MAP of its window marginal, the
             window starting from the previous commit treated as certain) as
             soon as w later rounds are available; w = 0 is the memoryless
             real-time decoder.  At the end the remaining window is decoded.
  postsel  : reject any shot with a non-zero syndrome in any round (LRB-D
             analogue); fidelity of accepted shots and acceptance.

Exact cross-checks: 'none' is exactly a product; for 'causal 0' on memory-noisy
the residual is a finite Markov chain, F(T) = e_0^T M^T f is computed exactly
and the eigenvalues with non-zero weight are printed (sum of exponentials).

Usage:  module load python/3.11; python A3_offline_vs_causal_toy.py [--quick]
Runtime: about one to three minutes on one core with the default settings.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import time

import numpy as np

W = np.exp(2j * np.pi / 3)


class Chain:
    def __init__(self, n, q, qm, qu, qpost):
        self.n, self.q, self.qm, self.qu, self.qpost = n, q, qm, qu, qpost
        self.S = 3 ** n
        self.K = 3 ** (n - 1)
        self.vec = np.array(list(itertools.product(range(3), repeat=n)))[:, ::-1]
        self.idx_of = {tuple(v): i for i, v in enumerate(self.vec)}
        self.svec = np.array(list(itertools.product(range(3), repeat=n - 1)))[:, ::-1]
        self.sidx_of = {tuple(v): i for i, v in enumerate(self.svec)}
        S, K = self.S, self.K
        self.ADD = np.array([[self.idx_of[tuple((self.vec[i] + self.vec[j]) % 3)]
                              for j in range(S)] for i in range(S)])
        self.SUB = np.array([[self.idx_of[tuple((self.vec[i] - self.vec[j]) % 3)]
                              for j in range(S)] for i in range(S)])
        self.SADD = np.array([[self.sidx_of[tuple((self.svec[i] + self.svec[j]) % 3)]
                               for j in range(K)] for i in range(K)])
        self.SSUB = np.array([[self.sidx_of[tuple((self.svec[i] - self.svec[j]) % 3)]
                               for j in range(K)] for i in range(K)])
        self.HX = np.array([self.sidx_of[tuple((v[:-1] - v[1:]) % 3)]
                            for v in self.vec])
        self.L0 = self.vec[:, 0].copy()
        pm = np.array([1 - qm, qm / 2, qm / 2])
        self.p_meas = np.array([np.prod(pm[v]) for v in self.svec])
        self.p_err = {}
        self.T = {}
        for kind, rates in (("C", [q] * n), ("U", [qu] * n),
                            ("Z", [qpost] + [0.0] * (n - 1))):
            pe = np.array([np.prod([[1 - r, r / 2, r / 2][vi]
                                    for vi, r in zip(v, rates)])
                           for v in self.vec])
            self.p_err[kind] = pe
            T = np.zeros((S, S))
            for x in range(S):
                for e in range(S):
                    T[x, self.ADD[x, e]] += pe[e]
            self.T[kind] = T
        self.L = np.zeros((K, S))
        for s in range(K):
            for x in range(S):
                self.L[s, x] = self.p_meas[self.SSUB[s, self.HX[x]]]
        self.Lperfect = np.zeros((K, S))
        for x in range(S):
            self.Lperfect[self.HX[x], x] = 1.0
        # memoryless single-round MAP: s -> argmax_e P(e) P(s | e); ties are
        # kept as sets (the decoders break ties uniformly at random, which keeps
        # them inversion-symmetric so that E[w^R] = (3 P_succ - 1)/2 is real)
        self.lookup_sets = []
        for s in range(K):
            sc = self.p_err["C"] * self.L[s]
            self.lookup_sets.append(np.flatnonzero(sc >= sc.max() * (1 - 1e-12)))
        self.lookup = np.array([ls[0] for ls in self.lookup_sets])

    def sample(self, rng, N, schedule):
        rates = {"C": [self.q] * self.n, "U": [self.qu] * self.n,
                 "Z": [self.qpost] + [0.0] * (self.n - 1)}
        x = np.zeros(N, int)
        obs = []
        for kind in schedule:
            if kind in rates:
                e = np.zeros(N, int)
                for i, r in enumerate(rates[kind]):
                    if r == 0:
                        continue
                    u = rng.random(N)
                    e += np.where(u < r / 2, 1, np.where(u < r, 2, 0)) * 3 ** i
                x = self.ADD[x, e]
            if kind == "C":
                m = np.zeros(N, int)
                for j in range(self.n - 1):
                    u = rng.random(N)
                    m += np.where(u < self.qm / 2, 1,
                                  np.where(u < self.qm, 2, 0)) * 3 ** j
                obs.append(("noisy", self.SADD[self.HX[x], m]))
            elif kind == "P":
                obs.append(("perfect", self.HX[x].copy()))
            else:
                obs.append(None)
        return x, obs


def schedule_for(name, T):
    if name == "memory-perfect":
        return ["C"] * T + ["P"]
    if name == "memory-noisy":
        return ["C"] * T
    if name == "lrb":
        return ["C"] * T + ["U"] * T + ["C", "Z"]
    raise KeyError(name)


def _rand_argmax(p, rng):
    """argmax along axis 1 with exact ties broken uniformly at random."""
    return np.argmax(p * (1.0 + 1e-9 * rng.random(p.shape)), axis=1)


def decode(chain, schedule, obs, w, rng):
    """Causal committing decoder with lookahead w; w=None gives offline MAP."""
    N = next(len(o[1]) for o in obs if o is not None)
    S = chain.S
    c = np.zeros(N, int)

    def lik(o):
        kind, s = o
        sp = chain.SSUB[s, chain.HX[c]]
        return (chain.Lperfect if kind == "perfect" else chain.L)[sp]

    def forward(win):
        alpha = np.zeros((N, S))
        alpha[:, 0] = 1.0
        alphas = []
        for k in win:
            kind = schedule[k]
            prev = alpha
            if kind in chain.T:
                alpha = alpha @ chain.T[kind]
            if obs[k] is not None:
                alpha = alpha * lik(obs[k])
                bad = alpha.sum(axis=1) <= 0
                if np.any(bad):   # commit inconsistent with a perfect readout:
                    # explain the discrepancy by one fresh data-error step
                    alpha[bad] = (prev[bad] @ chain.T["C"]) * lik(obs[k])[bad]
            alpha = alpha / alpha.sum(axis=1, keepdims=True)
            alphas.append(alpha)
        return alphas

    def first_marginal(win):
        alphas = forward(win)
        beta = np.ones((N, S))
        for j in range(len(win) - 1, 0, -1):
            k = win[j]
            b = beta
            if obs[k] is not None:
                b = b * lik(obs[k])
            kind = schedule[k]
            if kind in chain.T:
                b = b @ chain.T[kind].T
            beta = b / np.maximum(b.sum(axis=1, keepdims=True), 1e-300)
        return alphas[0] * beta

    window = []
    for k in range(len(schedule)):
        window.append(k)
        if w is None:
            continue
        while window and obs[window[0]] is not None and \
                sum(obs[j] is not None for j in window[1:]) >= w:
            rhat = _rand_argmax(first_marginal(window), rng)
            c = chain.ADD[c, rhat]
            window.pop(0)
    if not window:
        return chain.L0[c]
    post = forward(window)[-1]
    plog = np.stack([post[:, chain.L0 == v].sum(axis=1) for v in range(3)], 1)
    return (chain.L0[c] + _rand_argmax(plog, rng)) % 3


def fidelity(x, chain, lhat):
    r = (chain.L0[x] - lhat) % 3
    z = W ** r
    F = abs(z.mean())
    se = np.sqrt(max(1e-12, 1 - F ** 2) / (2 * len(z)))
    return F, se


def postselect(x, chain, obs):
    acc = np.ones(len(x), bool)
    for o in obs:
        if o is not None:
            acc &= (o[1] == 0)
    if acc.sum() == 0:
        return float("nan"), 0.0
    return abs((W ** chain.L0[x[acc]]).mean()), acc.mean()


def exact_w0_memory_noisy(chain, Tmax):
    S = chain.S
    M = np.zeros((S, S))
    pe = chain.p_err["C"]
    for r in range(S):
        for e in range(S):
            if pe[e] == 0:
                continue
            re = chain.ADD[r, e]
            for m in range(chain.K):
                sp = chain.SADD[chain.HX[re], m]
                ties = chain.lookup_sets[sp]
                for d in ties:
                    M[r, chain.SUB[re, d]] += pe[e] * chain.p_meas[m] / len(ties)
    f = W ** chain.L0
    v = np.zeros(S)
    v[0] = 1.0
    out = []
    for _ in range(Tmax + 1):
        out.append(abs(v @ f))
        v = v @ M
    evals, R = np.linalg.eig(M)
    weights = (R[0, :]) * (np.linalg.inv(R) @ f)
    groups = []
    for lam, wt in zip(evals, weights):
        for g in groups:
            if abs(g[0] - lam) < 1e-7:
                g[1] += wt
                break
        else:
            groups.append([lam, wt])
    groups = [g for g in groups if abs(g[1]) > 1e-9]
    groups.sort(key=lambda g: -abs(g[0]))
    return np.array(out), groups


def fit_rate(Ts, Fs, ses, tmin):
    Ts, Fs, ses = map(np.asarray, (Ts, Fs, ses))
    mask = (Ts >= tmin) & (Fs > 5 * ses)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    y = np.log(Fs[mask])
    wts = (Fs[mask] / ses[mask]) ** 2
    A = np.vstack([np.ones(mask.sum()), Ts[mask]]).T
    cov = np.linalg.inv(A.T @ (wts[:, None] * A))
    beta = cov @ A.T @ (wts * y)
    return np.exp(beta[1]), np.sqrt(cov[1, 1]) * np.exp(beta[1]), np.exp(beta[0])


def run(args):
    rng = np.random.default_rng(args.seed)
    chain = Chain(args.n, args.q, args.qm, args.qu, args.qpost)
    print(f"# toy chain n={args.n} (distance {args.n}), q={args.q}, q_m={args.qm}, "
          f"q_u={args.qu}, q_post={args.qpost}, N={args.N} shots per point")
    print("# memoryless single-round MAP lookup (syndrome -> committed error, "
          "ties in braces): " + ", ".join(
              f"{tuple(chain.svec[s])}->" + "{" + "|".join(
                  str(tuple(chain.vec[e])) for e in chain.lookup_sets[s]) + "}"
              for s in range(chain.K)))
    decs = ["none", "offline"] + [f"causal{w}" for w in args.ws]
    for sched in args.schedules:
        Ts = args.Ts_lrb if sched == "lrb" else args.Ts
        print(f"\n## schedule {sched}" + ("   (x-axis: m; 2m+1 error steps, m+1 rounds)"
                                         if sched == "lrb" else "   (x-axis: T)"))
        print("   T " + "".join(f"{d:>17s}" for d in decs) + "   postsel(F, acc)")
        rows = {d: [] for d in decs}
        for T in Ts:
            sch = schedule_for(sched, T)
            x, obs = chain.sample(rng, args.N, sch)
            line = f"{T:4d} "
            for d in decs:
                if d == "none":
                    lhat = np.zeros(args.N, int)
                else:
                    lhat = decode(chain, sch, obs,
                                  None if d == "offline" else int(d[6:]), rng)
                F, se = fidelity(x, chain, lhat)
                rows[d].append((T, F, se))
                line += f"  {F:.4f}+-{se:.4f}"
            Fp, acc = postselect(x, chain, obs)
            print(line + f"   ({Fp:.4f}, {acc:.3f})")
        tmin = args.tmin_lrb if sched == "lrb" else args.tmin
        print(f"  fit F ~ A f^T for T >= {tmin}; local slope r(T) = F(T')/F(T) per step:")
        for d in decs:
            arr = np.array(rows[d])
            f, fse, A = fit_rate(arr[:, 0], arr[:, 1], arr[:, 2], tmin)
            loc = [(arr[i + 1, 1] / arr[i, 1]) ** (1.0 / (arr[i + 1, 0] - arr[i, 0]))
                   for i in range(len(arr) - 1)]
            print(f"    {d:9s}: f = {f:.5f} +- {fse:.5f}, A = {A:.4f}, "
                  f"(2/3)(1-f) = {(2/3)*(1-f):.2e};  local: "
                  + " ".join(f"{v:.4f}" for v in loc))
        if sched == "lrb":
            ex = [(1 - 1.5 * args.q) ** (T + 1) * (1 - 1.5 * args.qu) ** T
                  * (1 - 1.5 * args.qpost) for T in Ts]
            print("    none exact: " + " ".join(f"{v:.4f}" for v in ex))
    if "memory-noisy" in args.schedules:
        ex, groups = exact_w0_memory_noisy(chain, max(args.Ts))
        print("\n## exact transfer matrix, causal w=0, memory-noisy")
        print("  F(T): " + " ".join(f"{t}:{ex[t]:.4f}" for t in args.Ts))
        print("  eigenvalues with non-zero weight (lambda, summed weight):")
        for lam, wt in groups[:10]:
            print(f"    {lam.real:+.6f}{lam.imag:+.6f}j   {wt.real:+.5f}{wt.imag:+.5f}j")
        print(f"  none exact: f = 1 - 3q/2 = {1 - 1.5 * args.q:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--qm", type=float, default=0.006)
    ap.add_argument("--qu", type=float, default=0.0005)
    ap.add_argument("--qpost", type=float, default=0.001)
    ap.add_argument("--N", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ws", type=int, nargs="*", default=[0, 1, 2, 4])
    ap.add_argument("--Ts", type=int, nargs="*",
                    default=[1, 2, 4, 6, 10, 15, 20, 30, 40])
    ap.add_argument("--Ts-lrb", type=int, nargs="*", dest="Ts_lrb",
                    default=[0, 1, 2, 4, 6, 10, 14, 18, 22])
    ap.add_argument("--tmin", type=int, default=6)
    ap.add_argument("--tmin-lrb", type=int, default=4, dest="tmin_lrb")
    ap.add_argument("--schedules", nargs="*",
                    default=["memory-perfect", "memory-noisy", "lrb"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.N = 5000
        args.Ts = [1, 4, 10, 20]
        args.Ts_lrb = [0, 2, 6, 10]
        args.ws = [0, 1]
    t0 = time.time()
    run(args)
    print(f"\n# elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
