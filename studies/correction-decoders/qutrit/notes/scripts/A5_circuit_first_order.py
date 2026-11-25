#!/usr/bin/env python3
"""A5: first-order (single-fault) circuit-level coefficients for the
[[5,1,2]]_3 LRB circuits of the baseline SI1000 run.

For each stored circuit (read-only, from the baseline run folder) the
repository's CompactDetectorErrorModel gives, for every noise location, the
linear Z_3 response of every detector and of the logical observable.  Every
single fault (location x non-zero Weyl vector, probability prob/(3^w - 1),
or prob/2 for the 'f'/'p' shift channels) is enumerated and grouped by its
full detector signature D (all rounds).  At first order in p:

  detection (unif 1, all rounds used):
      rejection      R1 = sum of masses with D != 0
      accepted error U1 = mass with D = 0 and a non-zero observable shift
  detection, terminal round only (const 1):
      same with D restricted to the final round's four detectors
  optimal correction (the p -> 0 limit of maximum-likelihood decoding of the
  whole detector history; outcome = shift of the observable):
      C1 = U1 + sum_{D != 0} (mass(D) - max_L mass(D, L))
  hybrid (correct a signature whose single-fault explanations all give the
  same observable shift, reject the others):
      rejection H1 = sum_{D ambiguous} mass(D), accepted error = U1

All coefficients are divided by p and fitted linearly in the depth m (number
of forward Cliffords; each depth step adds one forward Clifford, one check
round and one inverse Clifford).  The slope is the per-round coefficient.
The fidelity metric |sum_k w^k P(k)| equals 1 - 1.5 P(shift != 0) at first
order, so 1 - f per round = 1.5 x slope(C1) for correction.

Run (login node, about one minute):
    module load python/3.11
    python A5_circuit_first_order.py [--cliffs 10] [--depth-idx 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(WORKTREE, "src"))

from lrb.correction.compact_dem import read_run_circuit as read_circuit  # noqa: E402
from lrb.correction.compact_dem import build_compact_detector_error_model  # noqa: E402

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_IDX = 4
P_VALUE = 1e-3
DATA = set(range(5))


def fault_vectors(loc):
    """Yield (vector, probability) for every non-zero fault of a location."""
    w = loc.vector_width
    if loc.gate == "N1":
        if loc.channel == "d":
            vecs = [v for v in itertools.product(range(3), repeat=2) if any(v)]
        elif loc.channel == "f":
            vecs = [(1, 0), (2, 0)]
        elif loc.channel == "p":
            vecs = [(0, 1), (0, 2)]
        else:
            raise ValueError(loc.channel)
    else:
        vecs = [v for v in itertools.product(range(3), repeat=w) if any(v)]
    pr = loc.probability / len(vecs)
    return np.array(vecs, dtype=np.int64), pr


def category(loc, seg):
    ps = float(loc.params.get("prob_scale", 0) or 0)
    if loc.gate == "N2":
        if loc.qudit in DATA and loc.target in DATA:
            return f"logical 2q gate ({seg})"
        return "check CNOT"
    if loc.channel == "f":
        return "measurement flip (5p)" if ps == 5.0 else "reset error (2p)"
    if abs(ps - 0.1) < 1e-9:
        return f"logical 1q gate ({seg})" if loc.qudit in DATA else "ancilla H gate"
    if abs(ps - 2.0) < 1e-9:
        return "data idle (meas/reset layers)" if loc.qudit in DATA else "ancilla idle"
    if abs(ps - 1.0) < 1e-9:
        return "pre-measurement depolarizing (ancilla)"
    return "other"


def segments(circuit, depth):
    """Label each operation index: 'fwd' / 'inv' logical gates, else 'check'."""
    n_meas = 0
    lab = []
    for op in circuit.operations:
        if op.gate_name == "M" and op.qudit_index >= 5:
            n_meas += 1
        lab.append("inv" if n_meas >= 4 * depth else "fwd")
    return lab


def analyse(path, depth):
    circ = read_circuit(path)
    seg = segments(circ, depth)
    model = build_compact_detector_error_model(circ)
    ndet = model.num_detectors
    final_cols = [i for i, lab in enumerate(model.detector_labels)
                  if lab.startswith(f"lrb_stab_r{depth}_")]
    sig = {}            # full signature -> mass per observable shift
    sig_final = {}      # terminal-round signature -> mass per shift
    cat_mass = {}       # category -> [total, detected, undetected-logical]
    entries = []
    for loc in model.locations:
        vecs, pr = fault_vectors(loc)
        T = np.zeros((loc.vector_width, ndet), dtype=np.int64)
        TL = np.zeros(loc.vector_width, dtype=np.int64)
        for k, terms in enumerate(loc.component_terms):
            for idx, val in terms.detector:
                T[k, idx] = val
            for idx, val in terms.logical:
                TL[k] = val
        Dm = (vecs @ T) % 3
        Lm = (vecs @ TL) % 3
        cat = category(loc, seg[loc.operation_index])
        cm = cat_mass.setdefault(cat, np.zeros(3))
        for D, L in zip(Dm, Lm):
            key = D.astype(np.int8).tobytes()
            sig.setdefault(key, np.zeros(3))[L] += pr
            kf = D[final_cols].astype(np.int8).tobytes()
            sig_final.setdefault(kf, np.zeros(3))[L] += pr
            cm[0] += pr
            if D.any():
                cm[1] += pr
            elif L:
                cm[2] += pr
            entries.append((cat, key, int(L), pr))
    zero = np.zeros(ndet, dtype=np.int8).tobytes()
    zero_f = np.zeros(len(final_cols), dtype=np.int8).tobytes()
    U1 = sig.get(zero, np.zeros(3))[1:].sum()
    R1 = sum(v.sum() for k, v in sig.items() if k != zero)
    C1 = U1 + sum(v.sum() - v.max() for k, v in sig.items() if k != zero)
    amb = {k for k, v in sig.items() if k != zero and np.count_nonzero(v > 0) > 1}
    H1 = sum(sig[k].sum() for k in amb)
    U1f = sig_final.get(zero_f, np.zeros(3))[1:].sum()
    R1f = sum(v.sum() for k, v in sig_final.items() if k != zero_f)
    C1f = U1f + sum(v.sum() - v.max() for k, v in sig_final.items() if k != zero_f)
    # correction-failure attribution: a fault fails if its shift differs from
    # the decoder's choice for its signature (or if undetected with a shift)
    best = {k: int(np.argmax(v)) for k, v in sig.items()}
    cat_fail = {}
    for cat, key, L, pr in entries:
        fail = (L != 0) if key == zero else (L != best[key])
        if fail:
            cat_fail[cat] = cat_fail.get(cat, 0.0) + pr
    return dict(depth=depth, n_locations=len(model.locations), n_detectors=ndet,
                R1=R1 / P_VALUE, U1=U1 / P_VALUE, C1=C1 / P_VALUE, H1=H1 / P_VALUE,
                R1_final=R1f / P_VALUE, U1_final=U1f / P_VALUE, C1_final=C1f / P_VALUE,
                total_mass=sum(v.sum() for v in sig.values()) / P_VALUE,
                cat_total={k: v[0] / P_VALUE for k, v in cat_mass.items()},
                cat_detected={k: v[1] / P_VALUE for k, v in cat_mass.items()},
                cat_undetected_logical={k: v[2] / P_VALUE for k, v in cat_mass.items()},
                cat_correction_fail={k: v / P_VALUE for k, v in cat_fail.items()})


def fit(ms, ys):
    A = np.vstack([np.ones(len(ms)), ms]).T
    coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return float(coef[1]), float(coef[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cliffs", type=int, default=10)
    ap.add_argument("--depth-idx", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for cl in range(args.cliffs):
        for di in args.depth_idx:
            path = os.path.join(RUN, "experiments", "LRB", str(cl), str(P_IDX), f"{di}.chp")
            r = analyse(path, DEPTHS[di])
            r["cliff"] = cl
            rows.append(r)
    ms = sorted({r["depth"] for r in rows})
    keys = ["R1", "U1", "C1", "H1", "R1_final", "U1_final", "C1_final", "total_mass"]
    mean = {k: [np.mean([r[k] for r in rows if r["depth"] == m]) for m in ms] for k in keys}
    std = {k: [np.std([r[k] for r in rows if r["depth"] == m]) for m in ms] for k in keys}
    fits = {k: fit(ms, mean[k]) for k in keys}
    print(f"first-order coefficients (units of p; p = {P_VALUE}), mean over "
          f"{args.cliffs} Clifford sequences")
    print("  m   " + "  ".join(f"{k:>10s}" for k in keys))
    for i, m in enumerate(ms):
        print(f"  {m:2d}  " + "  ".join(f"{mean[k][i]:10.3f}" for k in keys))
    print("  linear fit (slope per round, intercept):")
    for k in keys:
        print(f"    {k:11s} slope {fits[k][0]:8.3f}  intercept {fits[k][1]:8.3f}")
    cats = sorted({c for r in rows for c in r["cat_total"]})
    catfit = {}
    for name in ("cat_total", "cat_detected", "cat_undetected_logical", "cat_correction_fail"):
        catfit[name] = {}
        for c in cats:
            ys = [np.mean([r[name].get(c, 0.0) for r in rows if r["depth"] == m]) for m in ms]
            catfit[name][c] = fit(ms, ys)
    print("  per-round slope by fault category (units of p):")
    print(f"    {'category':42s} {'total':>8s} {'detected':>9s} {'undet.log':>9s} {'corr.fail':>9s}")
    for c in cats:
        print(f"    {c:42s} {catfit['cat_total'][c][0]:8.3f} {catfit['cat_detected'][c][0]:9.3f}"
              f" {catfit['cat_undetected_logical'][c][0]:9.3f} {catfit['cat_correction_fail'][c][0]:9.3f}")
    s_corr = fits["C1"][0]
    print(f"  correction: per-round observable-flip slope {s_corr:.3f} p -> 1-f per round "
          f"= {1.5 * s_corr:.2f} p (twirled eps = {4 / 3 * s_corr:.2f} p); physical RB "
          f"comparison: 1-f = 5.625 p (eps = 5 p) per round")
    out = dict(p=P_VALUE, cliffs=args.cliffs, depths=ms, mean=mean, std=std, fits=fits,
               category_fits=catfit, runtime_s=time.time() - t0,
               rows=[{k: v for k, v in r.items()} for r in rows])
    os.makedirs(os.path.join(HERE, "A5_out"), exist_ok=True)
    with open(os.path.join(HERE, "A5_out", "A5_circuit_first_order.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
