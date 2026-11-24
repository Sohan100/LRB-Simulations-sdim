#!/usr/bin/env python3
"""Critic checks of A2 circuit-level claims C5, C9, C10, C20, C23, C24.

Independent of A2_optimal_rule_check.py except in step [V], which imports
its exact_joint_from_dem only to compare outputs.

  [V] validation of the exact law P(D, L) of baseline circuits:
      (i)  my own characteristic-function code, which reads each location's
           fault law from the DEM (N2: the actual prob_dist, NOT assumed
           uniform; N1: channel d/f/p as sampled by dem_simulation.py);
      (ii) a direct probability-space convolution (no Fourier transform) for a
           few circuits;
      (iii) A2's exact_joint_from_dem;
      (iv) Monte Carlo with the repository sampler sample_detector_error_model.
  [P] policies, frontier, D=0 ranking margin, MAP and J = alpha*lam, compared
      with the baseline CSVs, for m=0 and m=2 at p=1e-3 and p=1e-2 (30
      sequences each, including p=1e-2, m=2, where A2 used 8 and did not
      compare with the CSV).

Run: module load python/3.11 && python critic_A2-optimal-postselection_circuit.py
(about 3-4 minutes on one core).  Reads the baseline run folder, writes nothing
there.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass

WT = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/"
      "correction-decoders-qutrit-2c455b")
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
sys.path.insert(0, os.path.join(WT, "src"))
sys.path.insert(0, os.path.join(WT, "docs/correction-decoders/qutrit-512/notes/scripts"))
from sdim.circuit_io import read_circuit                      # noqa: E402
from lrb.dem_simulation import (build_compact_detector_error_model,  # noqa: E402
                                sample_detector_error_model)

W = np.exp(2j * np.pi / 3)


def location_law(loc, nd, n):
    """Return (R, vs, probs): response matrix R (n x w) and the fault law
    {v: prob} of this location exactly as dem_simulation.py samples it."""
    R = np.zeros((n, loc.vector_width), dtype=np.int64)
    for c, t in enumerate(loc.component_terms):
        for i, v in t.detector:
            R[i, c] = v
        for i, v in t.logical:
            R[nd + i, c] = v
    if loc.gate == "N1":
        p = loc.probability
        if loc.channel == "d":
            vs = [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]
            pr = [p / 8] * 8
        elif loc.channel == "f":
            vs, pr = [(1, 0), (2, 0)], [p / 2] * 2
        elif loc.channel == "p":
            vs, pr = [(0, 1), (0, 2)], [p / 2] * 2
        else:
            raise ValueError(loc.channel)
        vs = np.array(vs)
        pr = np.array(pr)
    else:
        dist = np.asarray(loc.params["prob_dist"], float)
        idx = np.arange(81)
        vs = np.stack([idx // 27, (idx // 9) % 3, (idx // 3) % 3, idx % 3], 1)
        vs, pr = vs[1:], dist[1:]          # sample_dense_two_qudit_distribution
    return R, vs, pr                       # digit order: column 0 most significant


def joint_charfun(model):
    """My characteristic-function computation of P(D, L) (complex-safe)."""
    nd, nl = model.num_detectors, model.num_logical_observables
    n = nd + nl
    Phat = np.ones((3,) * n, dtype=complex)
    by_support = {}
    for loc in model.locations:
        R, vs, pr = location_law(loc, nd, n)
        U = (vs @ R.T) % 3                       # shift vector per fault v
        S = np.flatnonzero(U.any(0))
        if S.size == 0:
            continue
        # law of the shift u restricted to the support S, on a small grid
        dist = np.zeros((3,) * S.size)
        dist[(0,) * S.size] = 1 - pr.sum()
        np.add.at(dist, tuple(U[:, S].T), pr)
        # fac(k) = sum_u dist(u) w^{k.u} = 3^|S| * ifftn(dist)(k)
        fac = np.fft.ifftn(dist) * 3 ** S.size
        key = tuple(S.tolist())
        by_support[key] = by_support[key] * fac if key in by_support else fac
    for key, fac in by_support.items():
        shape = [3 if ax in set(key) else 1 for ax in range(n)]
        Phat *= fac.reshape(shape)
    P = np.fft.fftn(Phat).real / 3 ** n      # P(x) = 3^-n sum_k Phat w^{-k.x}
    return P


def joint_direct(model):
    """Direct convolution in probability space (no Fourier transform)."""
    nd, nl = model.num_detectors, model.num_logical_observables
    n = nd + nl
    P = np.zeros((3,) * n)
    P[(0,) * n] = 1.0
    for loc in model.locations:
        R, vs, pr = location_law(loc, nd, n)
        U = (vs @ R.T) % 3
        shifts = {}
        for u, w in zip(map(tuple, U), pr):
            shifts[u] = shifts.get(u, 0.0) + w
        new = (1 - pr.sum()) * P
        for u, w in shifts.items():
            ax = [i for i in range(n) if u[i]]
            if not ax:
                new += w * P
            else:
                new += w * np.roll(P, [u[i] for i in ax], axis=ax)
        P = new
    return P


def read_csv(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            rows[parts[0]] = [float(x) for x in parts[1:]]
    return rows


def rounds_used(kind, k, m):
    if kind == "unif":
        r = {j * k - 1 for j in range(1, m // k + 1)} if k <= m else set()
    else:
        r = {j * (m // k) - 1 for j in range(1, k)} if k <= m else set()
    return sorted({x for x in r if x < m} | {m})


def evaluate(P, labels):
    nd = P.ndim - 1
    J = P.reshape(-1, 3)
    pD = J.sum(1)
    q = np.where(pD > 0, J.max(1) / np.where(pD > 0, pD, 1), 0)
    out = {}
    keep = pD > 1e-13
    order = np.argsort(-q[keep], kind="stable")
    a = np.concatenate([[0], np.cumsum(pD[keep][order])])
    b = np.concatenate([[0], np.cumsum((pD * q)[keep][order])])
    out["Sstar"] = lambda al: np.interp(al, a, b) / al
    out["EQ"] = float((pD * q).sum())
    q0 = q[0]
    for thr in (1e-13, 1e-10, 1e-8):
        m = (pD > thr)
        m[0] = False
        out[f"margin_{thr:g}"] = q0 - q[m].max()
        out[f"above_{thr:g}"] = float(pD[m & (q > q0)].sum())
    out["mass_unresolved"] = float(pD[pD <= 1e-13].sum())
    out["q0"] = float(q0)
    rnd = np.array([int(s.split("_r")[1].split("_")[0]) for s in labels])
    pols = {}
    m_depth = rnd.max()
    for kind, k in (("unif", 1), ("unif", 2), ("const", 1), ("const", 2)):
        used = rounds_used(kind, k, m_depth)
        sl = tuple(0 if rnd[ax] in used else slice(None) for ax in range(nd))
        pr = P[sl].reshape(-1, 3).sum(0)
        al = pr.sum()
        z = (pr / al * W ** np.arange(3)).sum()
        pols[(kind, k)] = dict(alpha=al, S=pr[0] / al, lam=z.real, F=abs(z),
                               im=z.imag, Sstar=out["Sstar"](al))
    out["pols"] = pols
    return out


def main():
    t_start = time.time()
    probs = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]
    # ---------------------------------------------------------------- [V]
    print("[V] validation of the exact joint law")
    from A2_optimal_rule_check import exact_joint_from_dem   # their code
    for p_idx, d_idx, cl in ((4, 1, 0), (8, 1, 3)):
        circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/{d_idx}.chp")
        model = build_compact_detector_error_model(circ)
        t0 = time.time()
        Pc = joint_charfun(model)
        t1 = time.time()
        Pd = joint_direct(model)
        t2 = time.time()
        Pa, _ = exact_joint_from_dem(model)
        print(f"  p={probs[p_idx]:g} m=2 seq {cl}: charfun {t1 - t0:.1f}s, direct "
              f"{t2 - t1:.1f}s; max|Pc-Pd|={np.abs(Pc - Pd).max():.1e}, "
              f"max|Pa-Pd|={np.abs(Pa - Pd).max():.1e}, sum Pd={Pd.sum():.15f}, "
              f"min Pd={Pd.min():.1e}")
        negPd = Pd[np.ix_(*([[0, 2, 1]] * Pd.ndim))]
        rel = np.abs(Pd - negPd) / np.maximum(Pd, 1e-300)
        print(f"    direct convolution, conjugation symmetry max|P(x)-P(-x)| ="
              f" {np.abs(Pd - negPd).max():.1e}, max relative {rel[Pd > 0].max():.1e}"
              f" (not built in here)")
        # ranking of D=0 over ALL histories, using the positive-terms result
        Jd = Pd.reshape(-1, 3)
        pDd = Jd.sum(1)
        qd = np.where(pDd > 0, Jd.max(1) / np.where(pDd > 0, pDd, 1), 0)
        pos = pDd > 0
        pos[0] = False
        above = pos & (qd > qd[0] * (1 + 1e-12))
        print(f"    direct: histories with P(D)>0: {int(pos.sum()) + 1};"
              f" smallest P(D) {pDd[pDd > 0].min():.1e}; q(0)={qd[0]:.8f};"
              f" max over ALL D!=0 of q = {qd[pos].max():.8f}; #D above q(0):"
              f" {int(above.sum())} (mass {pDd[above].sum():.1e});"
              f" mass with P(D)<=1e-13: {pDd[pDd <= 1e-13].sum():.1e}")
        i2 = np.flatnonzero(pos)[np.argmax(qd[pos])]
        dig = np.array(np.unravel_index(i2, (3,) * (Pd.ndim - 1)))
        print(f"    runner-up history (raw detectors by round): "
              f"{dig.reshape(-1, 4).tolist()}, P(D)={pDd[i2]:.2e}")
        if p_idx == 8:
            np.random.seed(12345)
            N = 400_000
            t3 = time.time()
            pay = sample_detector_error_model(model, extra_shots=N)
            det = np.stack([d["data"] for d in pay["detectors"]], 1)
            lg = pay["logicals"][0]["data"]
            x = np.concatenate([det, lg[:, None]], 1)
            flat = x @ (3 ** np.arange(x.shape[1] - 1, -1, -1))
            emp = np.bincount(flat, minlength=Pd.size) / N
            Pf = Pd.ravel()
            top = np.argsort(-Pf)[:40]
            exp = Pf[top] * N
            chi2 = float((((emp[top] * N) - exp) ** 2 / exp).sum())
            # also the empirical unif-1 survival and acceptance
            acc = ~det.any(1)
            print(f"    MC (repository sampler, {N} shots, {time.time() - t3:.0f}s):"
                  f" chi2 over the 40 most likely (D,L) cells = {chi2:.1f} (40 dof);"
                  f" unif1 acceptance MC {acc.mean():.5f} vs exact"
                  f" {Pd.reshape(-1, 3)[0].sum():.5f}; S MC {np.mean(lg[acc] == 0):.5f}"
                  f" vs exact {Pd.reshape(-1, 3)[0, 0] / Pd.reshape(-1, 3)[0].sum():.5f}")
    # ---------------------------------------------------------------- [P]
    print("\n[P] policies vs frontier vs CSV (my code; 30 sequences each)")
    for p_idx in (4, 8):
        for d_idx, m in ((0, 0), (1, 2)):
            t0 = time.time()
            res = []
            for cl in range(30):
                circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/{d_idx}.chp")
                model = build_compact_detector_error_model(circ)
                P = joint_charfun(model)
                res.append(evaluate(P, model.detector_labels))
            print(f"  p={probs[p_idx]:g} m={m} ({time.time() - t0:.0f}s): q(0) mean"
                  f" {np.mean([r['q0'] for r in res]):.6f}; min over seqs of"
                  f" q(0)-max_(D!=0,pD>1e-13) q = {min(r['margin_1e-13'] for r in res):.2e},"
                  f" (pD>1e-10) {min(r['margin_1e-10'] for r in res):.2e},"
                  f" (pD>1e-8) {min(r['margin_1e-08'] for r in res):.2e};"
                  f" mass ranked above D=0: {max(r['above_1e-13'] for r in res):.1e};"
                  f" unresolved mass (pD<=1e-13) max {max(r['mass_unresolved'] for r in res):.1e}")
            pols = [("unif", 1)] if m == 0 else [("unif", 1), ("unif", 2), ("const", 1), ("const", 2)]
            for pol in pols:
                L = [r["pols"][pol] for r in res]
                rej = 1 - np.mean([x["alpha"] for x in L])
                S = np.mean([x["S"] for x in L])
                lam = np.mean([x["lam"] for x in L])
                F = np.mean([x["F"] for x in L])
                Ss = np.mean([x["Sstar"] for x in L])
                dF = max(abs(x["F"] - abs(x["lam"])) for x in L)
                im = max(abs(x["im"]) for x in L)
                lam_sd = np.std([x["lam"] for x in L])
                csv = read_csv(f"{RUN}/results/LRB/{p_idx}/{pol[0]}_check_data/{pol[1]}.csv")
                cf, cr = csv["Fidelity averages"][d_idx], csv["Rejected Runs"][d_idx]
                csd = csv["Fidelity Standard Deviations"][d_idx]
                # shot-noise SE of the 30-sequence mean of lam_hat
                se = np.sqrt(np.mean([((1 + x["lam"]) / 2 - x["lam"] ** 2)
                                      / (x["alpha"] * 1e6) for x in L]) / 30)
                print(f"    {pol[0]:5s}{pol[1]} rounds {rounds_used(pol[0], pol[1], m)}:"
                      f" rej {rej:.4f} (CSV {cr:.4f}); S {S:.6f}; S* {Ss:.6f};"
                      f" lam {lam:.6f} vs CSV {cf:.6f} (diff {lam - cf:+.1e} ="
                      f" {(lam - cf) / se:+.1f} shot-SE); max|F-|lam||={dF:.0e},"
                      f" max|Im z|={im:.0e}; between-seq SD of lam {lam_sd:.2e}"
                      f" (CSV SD {csd:.2e})")
            EQ = np.mean([r["EQ"] for r in res])
            u1 = [r["pols"][("unif", 1)] for r in res]
            J_det = np.mean([x["alpha"] * x["lam"] for x in u1])
            print(f"    MAP no rejection: S*(1)=E[q]={EQ:.6f}; J_MAP={(3 * EQ - 1) / 2:.4f}"
                  f" vs J_unif1={J_det:.4f}")
    print(f"\ntotal {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
