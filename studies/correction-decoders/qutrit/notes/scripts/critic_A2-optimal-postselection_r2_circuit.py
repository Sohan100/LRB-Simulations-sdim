#!/usr/bin/env python3
"""Critic (round 2) of A2-optimal-postselection: circuit-level claims.

Mode "conv" (C5, C9, C10, C24 cross-check):
  * exact P(D, L) by DIRECT sequential convolution over the compact-DEM noise
    locations (sum of nonnegative terms, so even histories of tiny mass get
    accurate posteriors), independent of A2's characteristic-function/DFT
    method; then compare with A2's function on the same circuits;
  * the q-ranking of the all-trivial history over ALL histories (no mass
    cutoff), conjugation symmetry, policy and frontier numbers;
  * Monte Carlo with the repository's own DEM sampler
    (lrb.correction.compact_dem.sample_detector_error_model) vs the exact law.
Mode "dft" (C5 generality): A2's fast DFT function (validated in mode conv)
  on all 30 sequences at p=0.01, m=2 and on p=0.02, 0.05 at m=0, 2.

Reads the baseline run folder read-only. One core.
Run: module load python/3.11
     python critic_A2-optimal-postselection_r2_circuit.py conv
     python critic_A2-optimal-postselection_r2_circuit.py dft
"""
from __future__ import annotations

import importlib.util
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

WT = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-"
      "2bc490701489f8e7/correction-decoders-qutrit-2c455b")
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
       "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
sys.path.insert(0, os.path.join(WT, "src"))
from lrb.correction.compact_dem import read_run_circuit as read_circuit  # noqa: E402
from lrb.correction.compact_dem import (build_compact_detector_error_model,  # noqa
                                sample_detector_error_model)

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "a2", os.path.join(HERE, "A2_optimal_rule_check.py"))
a2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2)          # only defines functions; main() not run

PROBS = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]


def response(loc, nd, n):
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
        assert len(dist) == 81
        nz = dist[1:]
        assert np.all(nz == nz[0]), "N2 law not uniform over nonzero vectors"
        comps = [0, 1, 2, 3]
    return R[:, comps]


def exact_joint_conv(model):
    nd, nl = model.num_detectors, model.num_logical_observables
    n = nd + nl
    P = np.zeros((3,) * n)
    P[(0,) * n] = 1.0
    for loc in model.locations:
        Rc = response(loc, nd, n)
        if not Rc.any():
            continue
        w = Rc.shape[1]
        imgs = {}
        for v in itertools.product(range(3), repeat=w):
            if any(v):
                u = tuple(int(x) for x in (Rc @ np.array(v)) % 3)
                imgs[u] = imgs.get(u, 0) + 1
        p, tot = loc.probability, 3 ** w - 1
        new = (1.0 - p) * P
        for u, cnt in imgs.items():
            axes = [i for i in range(n) if u[i]]
            if not axes:
                new += (p * cnt / tot) * P
            else:
                new += (p * cnt / tot) * np.roll(
                    P, shift=[u[i] for i in axes], axis=axes)
        P = new
    return P, nd


def rounds_of(model):
    return np.array([int(lab.split("_r")[1].split("_")[0])
                     for lab in model.detector_labels])


def used_rounds(kind, k, m):
    # repository: postselect_uniform_interval / postselect_constant_number
    r = []
    if kind == "unif" and 0 < k <= m:
        r = [j * k - 1 for j in range(1, m // k + 1) if j * k - 1 < m]
    if kind == "const" and 0 < k <= m:
        r = [j * (m // k) - 1 for j in range(1, k)]
    return sorted(set(r) | {m})


def analyse(P, nd, model, m, verbose=True):
    J = P.reshape(-1, 3)
    pD = J.sum(1)
    pos = pD > 0
    q = np.where(pos, J.max(1) / np.where(pos, pD, 1), 0)
    q0 = q[0]
    others = np.arange(len(q)) != 0
    imax = int(np.argmax(np.where(others & pos, q, -1)))
    res = dict(q0=q0, qmax_other=q[imax], mass_imax=pD[imax],
               mass_above=pD[(q > q0) & pos].sum(),
               n_above=int(((q > q0) & pos).sum()),
               min_mass=pD[pos].min())
    negP = P[np.ix_(*([[0, 2, 1]] * P.ndim))]
    big = P > 1e-30
    res["sym_abs"] = np.abs(P - negP).max()
    res["sym_rel"] = (np.abs(P - negP)[big] / P[big]).max()
    # frontier
    o = np.argsort(-q, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(pD[o])])
    val = np.concatenate([[0.0], np.cumsum((pD * q)[o])])
    Sst = lambda a: np.interp(a, cum, val) / a
    rnd = rounds_of(model)
    pol = {}
    for kind, k in (("unif", 1), ("unif", 2), ("const", 1), ("const", 2)):
        used = used_rounds(kind, k, m)
        sl = tuple(0 if rnd[ax] in used else slice(None) for ax in range(nd))
        pr = P[sl].reshape(-1, 3).sum(0)
        a = pr.sum()
        z = (pr / a * np.exp(2j * np.pi * np.arange(3) / 3)).sum()
        pol[(kind, k)] = dict(alpha=a, S=pr[0] / a, lam=(3 * pr[0] / a - 1) / 2,
                              Fleg=abs(z), im=z.imag, Sstar=Sst(a))
    res["pol"] = pol
    res["S1"] = Sst(1.0)
    return res


def mode_conv():
    print("mode conv: direct convolution vs A2 DFT, all-history ranking")
    cases = [(4, 0, 0), (4, 0, 7), (8, 0, 0), (14, 0, 0),
             (4, 1, 0), (8, 1, 3)]
    for p_idx, d_idx, cl in cases:
        m = (0, 2)[d_idx]
        circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/{d_idx}.chp")
        model = build_compact_detector_error_model(circ)
        t0 = time.time()
        Pc, nd = exact_joint_conv(model)
        tc = time.time() - t0
        Pd, _ = a2.exact_joint_from_dem(model)
        big = Pc > 1e-12
        rel = (np.abs(Pc - Pd)[big] / Pc[big]).max()
        r = analyse(Pc, nd, model, m)
        print(f" p={PROBS[p_idx]:g} m={m} seq={cl}: conv {tc:.0f}s; "
              f"sum={Pc.sum():.15f}; |conv-dft| max abs "
              f"{np.abs(Pc - Pd).max():.1e}, max rel (P>1e-12) {rel:.1e}; "
              f"min P(D)={r['min_mass']:.1e}")
        print(f"   q(0)={r['q0']:.12f}; max_(D!=0) q={r['qmax_other']:.12f} "
              f"(mass {r['mass_imax']:.1e}); #histories with q>q(0): "
              f"{r['n_above']} (mass {r['mass_above']:.1e}); "
              f"sym abs {r['sym_abs']:.1e} rel {r['sym_rel']:.1e}")
        for key, v in r["pol"].items():
            if m == 0 and key != ("unif", 1):
                continue
            print(f"   {key[0]}{key[1]}: rej={1 - v['alpha']:.4f} "
                  f"S={v['S']:.6f} lam={v['lam']:.6f} Fleg={v['Fleg']:.6f} "
                  f"Im={v['im']:+.0e} S*(same a)={v['Sstar']:.6f}")
        print(f"   S*(1) (MAP, no rejection) = {r['S1']:.6f}")
    # Monte Carlo with the repository's DEM sampler vs exact law (m=2)
    np.random.seed(12345)
    p_idx, d_idx, cl = 4, 1, 0
    circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/{d_idx}.chp")
    model = build_compact_detector_error_model(circ)
    Pc, nd = exact_joint_conv(model)
    shots = 2_000_000
    out = sample_detector_error_model(model, extra_shots=shots)
    Dm = np.stack([d["data"] for d in out["detectors"]], 1)
    Lm = out["logicals"][0]["data"]
    idx = (Dm * 3 ** np.arange(nd - 1, -1, -1)).sum(1)
    Jc = Pc.reshape(-1, 3)
    emp = np.zeros_like(Jc)
    np.add.at(emp, (idx, Lm), 1.0)
    emp /= shots
    zs = (emp - Jc) / np.sqrt(Jc * (1 - Jc) / shots + 1e-300)
    top = np.argsort(-Jc.ravel())[:40]
    print(f" MC check (repo DEM sampler, {shots} shots, p=1e-3 m=2 seq 0): "
          f"P(D=0,L=0) exact {Jc[0, 0]:.6f} MC {emp[0, 0]:.6f}; "
          f"max |z-score| over the 40 most likely (D,L) cells "
          f"{np.abs(zs.ravel()[top]).max():.2f}")


def mode_dft():
    print("mode dft: A2's DFT function on more sequences / larger p")
    for p_idx, d_idx, ncl in ((8, 1, 30), (11, 0, 30), (11, 1, 30),
                              (14, 0, 30), (14, 1, 30)):
        m = (0, 2)[d_idx]
        t0 = time.time()
        acc = {}
        worst_above, worst_ratio, S1 = 0.0, -np.inf, []
        for cl in range(ncl):
            circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/"
                                f"{d_idx}.chp")
            model = build_compact_detector_error_model(circ)
            P, nd = a2.exact_joint_from_dem(model)
            J = P.reshape(-1, 3)
            pD = J.sum(1)
            keep = pD > 1e-11          # DFT values below this are unreliable
            q = np.where(keep, J.max(1) / np.where(keep, pD, 1), 0)
            above = pD[keep & (q > q[0] * (1 + 1e-7))].sum()
            worst_above = max(worst_above, above)
            qo = np.where(keep & (np.arange(len(q)) > 0), q, 0).max()
            worst_ratio = max(worst_ratio, qo - q[0])
            r = analyse(P, nd, model, m, verbose=False)
            S1.append(r["S1"])
            for key, v in r["pol"].items():
                acc.setdefault(key, []).append(v)
        print(f" p={PROBS[p_idx]:g} m={m} ({ncl} seqs, {time.time() - t0:.0f}s):"
              f" mass ranked above D=0 (P(D)>1e-11) max {worst_above:.1e}; "
              f"max_D (q(D)-q(0)) = {worst_ratio:+.2e}; mean S*(1) "
              f"{np.mean(S1):.6f}")
        for key, L in acc.items():
            if m == 0 and key != ("unif", 1):
                continue
            mean = lambda f: np.mean([x[f] for x in L])
            print(f"   {key[0]}{key[1]}: rej={1 - mean('alpha'):.4f} "
                  f"S={mean('S'):.6f} S*(same a)={mean('Sstar'):.6f} "
                  f"gap={mean('Sstar') - mean('S'):.2e} "
                  f"max|Fleg-lam|={max(abs(x['Fleg'] - abs(x['lam'])) for x in L):.0e}")


def mode_ref():
    """Noiseless reference shot: logical value x0 + x3 and raw ancilla
    values (A2 checked K = L on six circuits only)."""
    from sdim.program import Program, SimulationOptions
    print("mode ref: noiseless reference values, p=1e-3 circuits")
    for d_idx in (1, 4, 8):
        m = [0, 2, 4, 6, 10, 14, 18, 20, 22][d_idx]
        logical, anc_nonzero = set(), 0
        for cl in range(30):
            circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/4/{d_idx}.chp")
            prog = Program(circ)
            prog._simulate_tableau(SimulationOptions(shots=1))
            res = prog.measurement_results
            logical.add((res[0][-1][0].measurement_value
                         + res[3][-1][0].measurement_value) % 3)
            anc_nonzero += sum(int(r[0].measurement_value % 3 != 0)
                               for w in (5, 6, 7, 8) for r in res[w])
        print(f"   m={m}: reference x0+x3 over 30 sequences: {sorted(logical)};"
              f" nonzero reference ancilla records: {anc_nonzero}")


if __name__ == "__main__":
    {"conv": mode_conv, "dft": mode_dft, "ref": mode_ref}[sys.argv[1]]()
