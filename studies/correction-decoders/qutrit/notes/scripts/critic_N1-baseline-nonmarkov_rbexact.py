#!/usr/bin/env python3
"""Critic check for N1, part 2: exact prediction for physical RB (results/RB).

The stored RB circuits experiments/RB/<cliff>/<p_idx>/<depth_idx>.chp are
single-qutrit Clifford strings with an `N1 ... noise_channel="d"` after every
gate and a noiseless `M 0`.  For single-qutrit depolarizing noise of total
probability q (uniform over the 8 non-identity Weyl operators) the channel is
rho -> lam rho + (1 - lam) I/3 with lam = 1 - 9 q / 8, it commutes with every
unitary, and the metric |sum_k omega^k P(k)| of the mirrored circuit is
exactly lam^N(s) (N = number of noise locations).  So the exact expected
30-sequence mean is (1/30) sum_s lam^N_s(m), with no fit parameter.

Checks:
  * which probability convention the simulator used (q = prob, or prob*scale);
  * observed RB means vs the exact finite-sample prediction, in units of the
    exact shot-noise SEM (independent binomial shots, 1e4 per circuit);
  * observed across-sequence std vs the exact std of lam^N_s plus shot noise;
  * whether the 30 sequences are shared across p (same N_s at every p);
  * how far the exact 30-sequence mean is from a pure exponential g^m.
Usage: module load python/3.11 && python critic_N1-baseline-nonmarkov_rbexact.py
"""
import csv
import math
import os
import re

import numpy as np
from scipy import stats

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22], dtype=float)
PROBS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 1e-2, 1.25e-2,
         1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2]
NSEQ, SHOTS = 30, 10_000
PROB_RE = re.compile(r'prob="([^"]+)"\s+prob_scale="([^"]+)"')


def parse(path):
    n_noise, probs, scales, gates = 0, set(), set(), []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("N1"):
                n_noise += 1
                mt = PROB_RE.search(line)
                probs.add(float(mt.group(1))); scales.add(float(mt.group(2)))
            elif line and not line.startswith(("#", "d ", "Random", "M ", "LOGICAL")):
                gates.append(line.split()[0])
    assert len(probs) <= 1 and len(scales) <= 1, path
    return n_noise, (probs.pop() if probs else 0.0), (scales.pop() if scales else 1.0), gates


def read_csv(pidx):
    with open(os.path.join(RUN, "results", "RB", f"{pidx}.csv"), newline="") as fh:
        rows = list(csv.reader(fh))
    return [np.array([float(v) for v in r[1:]]) for r in rows[1:3]]


def main():
    N = np.zeros((15, NSEQ, 9), dtype=int)
    gate_seq = {}
    prob_file = {}
    for pidx in range(1, 15):
        for s in range(NSEQ):
            for d in range(9):
                n, pr, sc, gates = parse(os.path.join(RUN, "experiments", "RB", str(s),
                                                      str(pidx), f"{d}.chp"))
                N[pidx, s, d] = n
                prob_file[pidx] = (pr, sc)
                if d == 8:
                    gate_seq[(pidx, s)] = tuple(gates)
    shared = all(gate_seq[(pidx, s)] == gate_seq[(1, s)] for pidx in range(1, 15)
                 for s in range(NSEQ))
    distinct = len({gate_seq[(1, s)] for s in range(NSEQ)})
    print(f"sequences identical across all 14 p: {shared}; distinct sequences at one p: {distinct}")
    # nested? depth-m circuit = first m Cliffords: check N_s(m) monotone
    print(f"N_s(m) non-decreasing in m for all s: {bool(np.all(np.diff(N[1], axis=1) >= 0))};"
          f" mean gates per Clifford (fwd+inv)/m at m=22: {N[1, :, 8].mean()/22:.3f}")
    print(f"file prob / p: {[round(prob_file[i][0]/PROBS[i], 6) for i in range(1, 15)]}; "
          f"prob_scale: {sorted({prob_file[i][1] for i in range(1, 15)})}")

    allz, rows = [], []
    for conv in ("q = prob", "q = prob*scale"):
        chi2_tot, dof_tot = 0.0, 0
        for pidx in range(1, 15):
            pr, sc = prob_file[pidx]
            q = pr if conv == "q = prob" else pr * sc
            lam = 1 - 9 * q / 8
            ys = lam ** N[pidx]                              # (30, 9) exact per sequence
            mean_exact = ys.mean(axis=0)
            shot_var = (1 - ys) * (ys + 0.5) / SHOTS          # per sequence
            sem_shot = np.sqrt(shot_var.sum(axis=0)) / NSEQ
            mean_obs, std_obs = read_csv(pidx)
            ok = sem_shot > 0
            z = (mean_obs[ok] - mean_exact[ok]) / sem_shot[ok]
            chi2_tot += float(np.sum(z ** 2)); dof_tot += int(ok.sum())
            if conv == "q = prob":
                allz.append(z)
                std_exact = np.sqrt(ys.var(axis=0) + shot_var.mean(axis=0))
                # pure exponential through the exact finite-sample curve (log-linear, b free)
                L = np.log(mean_exact[1:])
                X = np.column_stack([np.ones(8), DEPTHS[1:]])
                th, *_ = np.linalg.lstsq(X, L, rcond=None)
                dev = mean_exact[1:] - np.exp(X @ th)
                # same with b fixed to 1 (the a = 0, b = 1 claim)
                g_fit = math.exp(np.sum(DEPTHS[1:] * L) / np.sum(DEPTHS[1:] ** 2))
                dev1 = mean_exact - g_fit ** DEPTHS
                rows.append((PROBS[pidx], q, float(np.max(np.abs(z))), float(np.mean(z)),
                             float(np.median(std_obs[1:] / std_exact[1:])),
                             float(np.max(np.abs(dev / sem_shot[1:]))),
                             float(np.max(np.abs(dev1[1:] / sem_shot[1:]))),
                             float(math.exp(th[0]))))
        print(f"[{conv}] chi2(obs - exact, shot SEM) = {chi2_tot:.1f} on {dof_tot} points, "
              f"p = {stats.chi2.sf(chi2_tot, dof_tot):.3g}")
    print("p      q_gate   max|z|  mean z  median std_obs/std_exact  max|exact - expfit|/SEM (b free, b=1)  b_free")
    for r in rows:
        print(f"{r[0]:<7g}{r[1]:<9.2e}{r[2]:6.2f}  {r[3]:6.2f}   {r[4]:6.3f}                   "
              f"{r[5]:6.2f}  {r[6]:6.2f}                      {r[7]:.6f}")
    zz = np.concatenate(allz)
    print(f"all z: mean {zz.mean():.3f}, sd {zz.std(ddof=1):.3f}, n = {len(zz)}")


if __name__ == "__main__":
    main()
