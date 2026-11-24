#!/usr/bin/env python3
"""Critic check for N1, part 4: independent re-derivation of headline numbers
from the result CSVs (own fitting code, not N1_baseline_fits.py), plus code
algebra and the REML identity.

  [1] model A (b f^m, SEM weights, N1's usable-point rule) and the miss at m=22
      for const1 (C8); windowed rates for const1 / const0 at p = 1e-3 (C8, C9);
  [2] unif1 windowed acceptance hazard, first window vs later windows (C7);
  [3] saturation of const1 / const0 acceptance, const1 fidelity floor (C12);
  [4] pipeline fit at p = 5e-3 for unif1 with / without points that have
      < 1000 mean accepted shots per sequence (C17);
  [5] const1 windowed rate at fixed tail growth g = 2 versus m (C13);
  [6] code algebra in the symplectic representation over Z_3 (C10, C12);
  [7] REML / difference-operator identity on random matrices (C15).
Usage: module load python/3.11 && python critic_N1-baseline-nonmarkov_misc.py
"""
import csv
import itertools
import math
import os

import numpy as np
from scipy import optimize, stats

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
M = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22], dtype=float)
PROBS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 1e-2, 1.25e-2,
         1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2]
NS, SHOTS = 30, 1_000_000


def load(pidx, kind, k):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    with open(os.path.join(RUN, "results", "LRB", str(pidx), sub, f"{k}.csv")) as fh:
        rows = list(csv.reader(fh))
    y, sd, rej, rsd = [np.array([float(v) for v in r[1:]]) for r in rows[1:5]]
    return y, sd, rej, rsd


def usable(y, sd, rej):
    nacc = (1 - rej) * SHOTS
    ray = np.sqrt(np.pi / (4 * np.maximum(nacc, 1)))
    ok = np.isfinite(y) & np.isfinite(sd) & (nacc >= 1000) & (y >= 5 * ray)
    sem = np.maximum(sd / math.sqrt(NS), 1.5 / (NS * np.maximum(nacc, 1)))
    return ok, sem


def fit_A(m, y, s):
    f = lambda th: (y - th[0] * th[1] ** m) / s
    r = optimize.least_squares(f, [y[0], 0.999], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return r.x, float(r.fun @ r.fun)


def windowed(m, y):
    f = (y[1:] / y[:-1]) ** (1 / np.diff(m))
    return (2 / 3) * (1 - f)


def part1():
    print("[1] const1 / const0 at p = 1e-3 (pidx 4)")
    for kind, k in (("const", 1), ("const", 0)):
        y, sd, rej, _ = load(4, kind, k)
        ok, sem = usable(y, sd, rej)
        (b, f), c2 = fit_A(M[ok], y[ok], sem[ok])
        pred = b * f ** M
        print(f"    {kind}{k}: A chi2 = {c2:.4g} on {ok.sum() - 2} dof; miss at m=22: "
              f"{pred[-1] - y[-1]:.3e} ({(pred[-1] - y[-1]) / sem[-1]:.1f} sigma); miss at m=0: "
              f"{pred[0] - y[0]:.3e} ({(pred[0] - y[0]) / sem[0]:.1f} sigma)")
        print(f"      windowed r: {np.array2string(windowed(M, y), precision=3)}")


def part2():
    print("[2] unif1 per-step rejection hazard / p by window (m=0->2 first)")
    for pidx in (1, 2, 3, 4, 5):
        y, sd, rej, _ = load(pidx, "unif", 1)
        acc = 1 - rej
        h = -np.log(acc[1:] / acc[:-1]) / np.diff(M) / PROBS[pidx]
        print(f"    p={PROBS[pidx]:g}: " + " ".join(f"{v:.2f}" for v in h))


def part3():
    print("[3] saturation")
    rejs, fids, rays = [], [], []
    for pidx in (11, 12, 13, 14):
        y, sd, rej, _ = load(pidx, "const", 1)
        for i in range(3, 9):         # m >= 6
            rejs.append(rej[i]); fids.append(y[i])
            rays.append(math.sqrt(math.pi / (4 * (1 - rej[i]) * SHOTS)))
    rejs, fids, rays = map(np.array, (rejs, fids, rays))
    print(f"    const1, p >= 2e-2, m >= 6 (n={len(rejs)}): rejected {rejs.mean():.6f} "
          f"(80/81 = {80/81:.6f}, spread {rejs.min():.6f}..{rejs.max():.6f}); fidelity "
          f"{fids.mean():.4f} +- {fids.std(ddof=1)/math.sqrt(len(fids)):.4f}; Rayleigh {rays.mean():.4f}")
    vals = []
    for pidx in (13, 14):
        y, sd, rej, _ = load(pidx, "const", 0)
        vals += list(rej[6:])
    print(f"    const0, p >= 4e-2, m >= 18: rejected {min(vals):.4f}..{max(vals):.4f} (8/9 = {8/9:.4f})")


def pipeline_fit(x, y, e):
    """Independent re-implementation of the pipeline grid scan (offset 0)."""
    e = np.where(e <= 0, max(np.median(e[np.isfinite(e)]) * 1e-3, 1e-12), e)
    w = 1 / e ** 2
    lo, hi = 1e-8, 1.0
    for _ in range(5):
        g = np.linspace(lo, hi, 2001)
        z = g[None, :] ** x[:, None]
        b = (w[:, None] * z * y[:, None]).sum(0) / (w[:, None] * z * z).sum(0)
        c2 = (w[:, None] * (y[:, None] - b * z) ** 2).sum(0)
        i = int(np.argmin(c2)); fb = g[i]
        span = (hi - lo) / 10; lo, hi = max(1e-8, fb - span), min(1.0, fb + span)
    return fb


def part4():
    print("[4] pipeline unif1 F with all points vs only points with N_acc >= 1000")
    for pidx in (6, 7, 8):
        y, sd, rej, _ = load(pidx, "unif", 1)
        nacc = (1 - rej) * SHOTS
        ok = np.isfinite(y) & np.isfinite(sd) & (y <= 1)
        f_all = pipeline_fit(M[ok], y[ok], sd[ok])
        ok2 = ok & (nacc >= 1000)
        f_cut = pipeline_fit(M[ok2], y[ok2], sd[ok2])
        print(f"    p={PROBS[pidx]:g}: r_all = {(2/3)*(1-f_all):.4e} ({ok.sum()} pts), "
              f"r_cut = {(2/3)*(1-f_cut):.4e} ({ok2.sum()} pts); min N_acc used {nacc[ok].min():.1f}")


def part5():
    print("[5] const1 windowed rate (g = 2 in every window) vs m, p = 1e-4 .. 2e-3")
    for pidx in (1, 3, 4, 5):
        y, sd, rej, _ = load(pidx, "const", 1)
        r = windowed(M, y)
        print(f"    p={PROBS[pidx]:g}: r/p^2 = " + " ".join(f"{v / PROBS[pidx]**2:.1f}" for v in r))


# ---------------------------------------------------------------- algebra
NQ = 5


def vec(x=None, z=None):
    v = np.zeros(2 * NQ, dtype=int)
    for q, a in (x or {}).items():
        v[q] = a % 3
    for q, a in (z or {}).items():
        v[NQ + q] = a % 3
    return v


def sp(u, v):
    return int((u[:NQ] @ v[NQ:] - u[NQ:] @ v[:NQ]) % 3)


S = [vec(z={0: 1, 1: -1, 3: -1}), vec(z={1: 1, 2: 1, 4: -1}),
     vec(x={0: 1, 1: 1, 2: -1}), vec(x={1: -1, 3: 1, 4: -1})]
XL, ZL = vec(x={0: 1, 3: 1}), vec(z={0: 1, 2: 1})


def syndrome(e):
    return tuple(sp(e, s) for s in S)


def conj(v, gates):
    """Forward conjugation U v U^dag for a list of (name, qubits)."""
    v = v.copy()
    for name, qs in gates:
        x, z = v[:NQ], v[NQ:]
        if name == "H":
            q = qs[0]; x[q], z[q] = (-z[q]) % 3, x[q]
        elif name == "H_INV":
            q = qs[0]; x[q], z[q] = z[q], (-x[q]) % 3
        elif name == "P":
            q = qs[0]; z[q] = (z[q] + x[q]) % 3
        elif name == "P_INV":
            q = qs[0]; z[q] = (z[q] - x[q]) % 3
        elif name == "CZ":
            a, b = qs; za, zb = z[a] + x[b], z[b] + x[a]; z[a], z[b] = za % 3, zb % 3
        elif name == "SWAP":
            a, b = qs; x[a], x[b] = x[b], x[a]; z[a], z[b] = z[b], z[a]
    return v


H_L = [("SWAP", (2, 3))] + [("H", (q,)) for q in (0, 2, 3, 4)] + [("H_INV", (1,))]
P_L = [("P", (0,)), ("P", (4,)), ("P_INV", (1,)), ("CZ", (2, 3))]


def in_stab_group(v):
    """v commutes with all stabilizers and both logicals -> in the stabilizer group."""
    return syndrome(v) == (0, 0, 0, 0) and sp(v, XL) == 0 and sp(v, ZL) == 0


def part6():
    print("[6] code algebra (symplectic, Z_3)")
    print(f"    stabilizers commute: {all(sp(a, b) == 0 for a in S for b in S)}; "
          f"logicals commute with S: {all(sp(L, s) == 0 for L in (XL, ZL) for s in S)}; "
          f"<XL,ZL> = {sp(XL, ZL)}")
    w1 = [vec(x={q: a}, z={q: b}) for q in range(5) for a in range(3) for b in range(3) if (a, b) != (0, 0)]
    print(f"    all 40 weight-1 data Paulis detected: {all(syndrome(e) != (0,0,0,0) for e in w1)}")
    w23 = [vec(x={2: a, 3: c}, z={2: b, 3: d}) for a, b, c, d in itertools.product(range(3), repeat=4)
           if (a, b, c, d) != (0, 0, 0, 0)]
    print(f"    all 80 non-identity Paulis on (2,3) detected: {all(syndrome(e) != (0,0,0,0) for e in w23)}")
    zonly = [e for e in w23 if not any(e[:NQ])]
    print(f"    every non-zero pure-Z Pauli on (2,3) flags an X-stabilizer (const 0 check): "
          f"{all(syndrome(e)[2:] != (0, 0) for e in zonly)}")
    x0, x3i = vec(x={0: 1}), vec(x={3: -1})
    print(f"    syndrome(X0) = {syndrome(x0)}, syndrome(X3^-1) = {syndrome(x3i)}; X0 X3 is logical X: "
          f"{np.array_equal((x0 - x3i) % 3, XL)}")
    for nm, G in (("H_L", H_L), ("P_L", P_L)):
        imgs = [conj(s, G) for s in S]
        ok = all(in_stab_group(v) for v in imgs)
        # induced linear map on syndromes: syndrome(U e U^dag) for unit errors
        basis = [vec(x={q: 1}) for q in range(5)] + [vec(z={q: 1}) for q in range(5)]
        A = np.array([syndrome(conj(e, G)) for e in basis]).T     # 4 x 10
        B = np.array([syndrome(e) for e in basis]).T
        # the syndrome map T satisfies A = T B; solve over Z_3 by brute force on 3^16 is too big;
        # instead check rank of B and A (both 4) -> T is invertible on the syndrome space
        def rank3(Mx):
            Mx = Mx.copy() % 3; r = 0
            for c in range(Mx.shape[1]):
                piv = next((i for i in range(r, Mx.shape[0]) if Mx[i, c]), None)
                if piv is None:
                    continue
                Mx[[r, piv]] = Mx[[piv, r]]
                Mx[r] = (Mx[r] * (1 if Mx[r, c] == 1 else 2)) % 3
                for i in range(Mx.shape[0]):
                    if i != r and Mx[i, c]:
                        Mx[i] = (Mx[i] - Mx[i, c] * Mx[r]) % 3
                r += 1
            return r
        print(f"    {nm}: maps every stabilizer generator into the stabilizer group: {ok}; "
              f"syndrome map rank {rank3(A)} (of 4)")
    print(f"    H_L(XL) = ZL: {np.array_equal(conj(XL, H_L), ZL)}; P_L(XL) = XL ZL: "
          f"{np.array_equal(conj(XL, P_L), (XL + ZL) % 3)}")


def part7():
    print("[7] REML identity S^-1 - S^-1 1 (1'S^-1 1)^-1 1'S^-1 = D'(D S D')^-1 D")
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in range(3, 10):
        for _ in range(50):
            A = rng.normal(size=(n, n)); Sg = A @ A.T + n * np.eye(n)
            m = np.sort(rng.choice(np.arange(40), n, replace=False)).astype(float)
            D = np.zeros((n - 1, n))
            for i in range(n - 1):
                D[i, i], D[i, i + 1] = -1 / (m[i + 1] - m[i]), 1 / (m[i + 1] - m[i])
            Si = np.linalg.inv(Sg); one = np.ones(n)
            lhs = Si - np.outer(Si @ one, one @ Si) / (one @ Si @ one)
            rhs = D.T @ np.linalg.inv(D @ Sg @ D.T) @ D
            worst = max(worst, np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs)))
    print(f"    max relative deviation over 350 random (S, m): {worst:.2e}")


def part8():
    """Exact 30-sequence RB curve (from the stored circuits): sign of its curvature."""
    import re
    print("[8] RB (p_RB = p): curvature of the exact finite-sample mean vs the data")
    N = np.zeros((30, 9))
    for s in range(30):
        for d in range(9):
            with open(os.path.join(RUN, "experiments", "RB", str(s), "1", f"{d}.chp")) as fh:
                N[s, d] = sum(1 for ln in fh if ln.startswith("N1"))
    for pidx in (1, 4, 8, 14):
        with open(os.path.join(RUN, "results", "RB", f"{pidx}.csv")) as fh:
            rows = list(csv.reader(fh))
        y = np.array([float(v) for v in rows[1][1:]]); sd = np.array([float(v) for v in rows[2][1:]])
        lam = 1 - 9 * (0.1 * PROBS[pidx]) / 8
        ye = (lam ** N).mean(axis=0)
        sem = np.maximum(sd / math.sqrt(30), 1e-12)
        sel = M > 0
        X = np.column_stack([np.ones(sel.sum()), M[sel], M[sel] ** 2])
        def dcoef(yy):
            w = yy[sel] / sem[sel]
            th, *_ = np.linalg.lstsq(X * w[:, None], np.log(yy[sel]) * w, rcond=None)
            cov = np.linalg.inv((X * w[:, None]).T @ (X * w[:, None]))
            return -th[2], math.sqrt(cov[2, 2])
        d_obs, s_obs = dcoef(y)
        d_ex, _ = dcoef(ye)
        print(f"    p={PROBS[pidx]:g}: z_d(data, m>0) = {d_obs/s_obs:+.2f}; z_d(exact sequence mean) = {d_ex/s_obs:+.2f}")


if __name__ == "__main__":
    part1(); part2(); part3(); part4(); part5(); part6(); part7(); part8()
