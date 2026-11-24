#!/usr/bin/env python3
"""Critic (v2): re-run parts of A3_pauli_frame_statevector_check.py with extra
diagnostics that the original does not print.

  1. Pauli mode (depth 2, Clifford sequences 0,1,2, p rescaled to 0.02): besides
     the shot-by-shot record agreement, count how many check rounds actually
     triggered a lookup correction (a non-trivial syndrome does not imply a
     correction: the lookup only knows weight-1 data syndromes), and how many
     shots ended with at least one correction.
  2. Coherent mode (theta = 0.1, 150 shots, as in A3): besides the paired mean
     difference, report the mean absolute per-shot difference of Re E[w^k], so
     that "averaged difference << per-shot difference" can be judged, and the
     difference per Clifford sequence.

The A3 module is imported read-only; its functions are called unchanged.
Usage: module load python/3.11; python critic_A3-pauli-frame_v2_rerun_a3.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "a3sv", os.path.join(HERE, "A3_pauli_frame_statevector_check.py"))
a3 = importlib.util.module_from_spec(spec)
sys.modules["a3sv"] = a3
spec.loader.exec_module(a3)
a3.SYMP = a3.symplectic_table()


def pauli(cliffs=(0, 1, 2), depth_idx=1, shots=20, p=0.02, seed=12345):
    tot = mism = rounds_tot = rounds_corr = shots_corr = 0
    for c in cliffs:
        path = f"{a3.RUN}/experiments/LRB/{c}/8/{depth_idx}.chp"
        ops = a3.parse_chp(path)
        lookup = a3.build_lookup(ops)
        rng = np.random.default_rng(seed)
        nround = len(a3.round_end_indices(ops))
        for _ in range(shots):
            draws = a3.draw_noise(ops, rng, p)
            rec_p, log_p, _ = a3.run_shot(ops, draws, lookup, "physical")
            rec_f, log_f, _ = a3.run_shot(ops, draws, lookup, "frame")
            tot += 1
            mism += (rec_p != rec_f) or (log_p != log_f)
            ncorr = 0
            for r in range(nround):
                synd = tuple(rec_p[4 * r:4 * r + 4])
                rounds_tot += 1
                if lookup.get(synd) is not None:
                    rounds_corr += 1
                    ncorr += 1
            shots_corr += ncorr > 0
    print(f"Pauli mode p={p}, depth_idx={depth_idx}: {tot} shots, mismatching shots {mism}; "
          f"rounds with a lookup correction {rounds_corr}/{rounds_tot}; "
          f"shots with >= 1 correction {shots_corr}/{tot}; "
          f"lookup table covers {len(lookup)} of 80 non-zero round syndromes")


PHYS, FRM = [], []


def coherent(cliffs=(0, 1, 2), shots=50, theta=0.1, seed=12345):
    per = {}
    PHYS.clear()
    FRM.clear()
    for c in cliffs:
        path = f"{a3.RUN}/experiments/LRB/{c}/8/1.chp"
        ops = a3.parse_chp(path)
        lookup = a3.build_lookup(ops)
        rng = np.random.default_rng(seed + 1)
        diffs, phys, frm = [], [], []
        for _ in range(shots):
            draws = a3.draw_noise(ops, rng, p_base=None)
            _, _, dp = a3.run_shot(ops, draws, lookup, "physical",
                                   coherent_theta=theta, exact_logical=True)
            _, _, df = a3.run_shot(ops, draws, lookup, "frame",
                                   coherent_theta=theta, exact_logical=True)
            cp = np.dot(dp, a3.W ** np.arange(3)).real
            cf = np.dot(df, a3.W ** np.arange(3)).real
            diffs.append(cp - cf)
            phys.append(cp)
            frm.append(cf)
        per[c] = np.array(diffs)
        PHYS.extend(phys)
        FRM.extend(frm)
    alld = np.concatenate(list(per.values()))
    n = len(alld)
    ph, fr = np.array(PHYS), np.array(FRM)
    print(f"Coherent mode theta={theta}: Re E[w^k] physical {ph.mean():.4f} +- "
          f"{ph.std(ddof=1) / np.sqrt(n):.4f}, frame {fr.mean():.4f} +- "
          f"{fr.std(ddof=1) / np.sqrt(n):.4f}")
    print(f"Coherent mode theta={theta}: {n} shots; paired mean diff "
          f"{alld.mean():+.4f} +- {alld.std(ddof=1) / np.sqrt(n):.4f}; "
          f"mean |diff| {np.abs(alld).mean():.4f}; rms diff {np.sqrt((alld ** 2).mean()):.4f}; "
          f"fraction |diff|>1e-9 {np.mean(np.abs(alld) > 1e-9):.3f}")
    for c, d in per.items():
        print(f"   sequence {c}: mean diff {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, "
              f"mean |diff| {np.abs(d).mean():.4f}")


if __name__ == "__main__":
    t = time.time()
    pauli()
    coherent()                      # A3's setting: 150 shots
    coherent(shots=200, seed=777)   # independent, larger sample (600 shots)
    print(f"elapsed {time.time() - t:.1f} s")
