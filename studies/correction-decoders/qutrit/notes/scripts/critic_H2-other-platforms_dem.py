"""Critic of H2-other-platforms.md: DEM-based checks of the platform claims.

Run:  module load python/3.11 && python3 critic_H2-other-platforms_dem.py
Runtime: about 3-4 min on one core. Writes nothing. Read-only use of the
baseline run folder (stored circuits of Clifford sequences 0-3).

The compact DEM of the repository (src/lrb/dem_simulation.py) is used only
to obtain each noise location's linear detector/logical response. Sampling
is done here with location probabilities chosen per noise model, so the
same circuits can be run under SI1000 at any p, under a parallel-readout
emulation, and under an ion-tailored model. The sampler is validated
against the baseline CSVs at m = 0 (identical circuits for all sequences)
and approximately at m > 0.

Quantities per depth m (averaged over 4 sequences, 20,000 shots each):
  rej_u1   unif 1 rejection;  rej_c1, F_c1  const 1 rejection and fidelity
  F_raw    fidelity of ALL shots, no correction
  F_dec    terminal-syndrome lookup decoder (trained on half, tested on half)
  B_or     oracle upper bound on ANY decoder without postselection:
           an oracle that knows every fault before the inverse block can
           reach at most sum_s P(s)|E[w^L | s]| over late faults (inverse
           block + final round); the true optimum is below this (Jensen).
Noise models
  si1000(p)       the brief's model, sequential single-ancilla readout
  si1000par(p)    same, readout layers merged into one: idle noise kept
                  only on data wires in the last readout layer of a round
  ion             two-qutrit 0.0144 (Hrmo 98.7% / 0.9), one-qutrit 1e-3,
                  readout flip 3e-3 (Ringbauer 99.7%), no extra ancilla
                  depolarization, ONE readout layer per round with data
                  idle 0.04 (3.8 ms / 100 ms), reset idle 0.003, reset
                  flip 1e-3.  ion_opt: data idle during readout 0.01.
                  These rates are my assumptions (conjecture), chosen to be
                  favourable to ions relative to SI1000 at p = 0.015.
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import os
import re
import sys
import time

import numpy as np
from sdim.circuit_io import read_circuit

WT = "/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/correction-decoders-qutrit-2c455b"
BASE = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
        "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
_spec = importlib.util.spec_from_file_location("crit_dem", os.path.join(WT, "src", "lrb", "dem_simulation.py"))
dem = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dem
_spec.loader.exec_module(dem)

DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
USE_DI = [0, 1, 2, 3, 4, 8]          # m = 0, 2, 4, 6, 10, 22
SEQS = [0, 1, 2, 3]
SHOTS = 20000
w = np.exp(2j * np.pi / 3)
rng = np.random.default_rng(2026)

N1D = np.array([v for v in itertools.product(range(3), repeat=2) if any(v)])
N1F = np.array([[1, 0], [2, 0]])
N2D = np.array([v for v in itertools.product(range(3), repeat=4) if any(v)])


def compile_circuit(cl, di):
    c = read_circuit(f"{BASE}/experiments/LRB/{cl}/10/{di}.chp")
    model = dem.build_compact_detector_error_model(c)
    ops = c.operations
    labels = model.detector_labels
    rounds = np.array([int(re.match(r"lrb_stab_r(\d+)_w(\d+)", lb).group(1)) for lb in labels])
    nd = len(labels)
    last_reset = max([i for i, op in enumerate(ops) if op.gate_name == "RESET"], default=-1)
    locs = []
    for loc in model.locations:
        pr = loc.params
        scale = float(pr.get("prob_scale", "nan"))
        R = np.zeros((loc.vector_width, nd + 1), dtype=np.int64)
        for ci, terms in enumerate(loc.component_terms):
            for di_, coef in terms.detector:
                R[ci, di_] = coef % 3
            for li, coef in terms.logical:
                R[ci, nd + li] = coef % 3
        if loc.gate == "N2":
            cat, V = "2q", N2D
        elif loc.channel == "f":
            cat, V = ("meas_flip" if scale == 5.0 else "reset_flip"), N1F
        elif scale == 0.1:
            cat, V = "1q", N1D
        elif scale == 1.0:
            cat, V = "anc_depol", N1D
        else:
            j = loc.operation_index
            while ops[j].gate_name not in ("M", "RESET"):
                j += 1
            if ops[j].gate_name == "RESET":
                cat, V = "idle_reset", N1D
            else:
                anc = int(ops[j].qudit_index)
                cat = f"idle_meas_{anc}_{'data' if loc.qudit < 5 else 'anc'}"
                V = N1D
        resp = (V @ R) % 3
        locs.append(dict(cat=cat, resp=resp, late=loc.operation_index > last_reset))
    return dict(rounds=rounds, nd=nd, locs=locs, m=DEPTHS[di])


def probs_for(model_name, locs, p=None):
    q = np.zeros(len(locs))
    for i, L in enumerate(locs):
        c = L["cat"]
        if model_name in ("si1000", "si1000par"):
            base = {"2q": p, "1q": 0.1 * p, "anc_depol": p, "meas_flip": 5 * p,
                    "reset_flip": 2 * p, "idle_reset": 2 * p}
            if c in base:
                q[i] = base[c]
            else:                                   # idle in a measurement layer
                _, _, anc, kind = c.split("_")
                if model_name == "si1000":
                    q[i] = 2 * p
                else:                               # one readout layer: data idle once per round
                    q[i] = 2 * p if (anc == "8" and kind == "data") else 0.0
        else:                                       # ion-tailored, one readout layer per round
            idle = 0.04 if model_name == "ion" else 0.01
            base = {"2q": 0.013 / 0.9, "1q": 1e-3, "anc_depol": 0.0, "meas_flip": 3e-3,
                    "reset_flip": 1e-3, "idle_reset": 3e-3}
            if c in base:
                q[i] = base[c]
            else:
                _, _, anc, kind = c.split("_")
                q[i] = idle if (anc == "8" and kind == "data") else 0.0
    return q


def sample(comp, q, shots, late_only=False):
    nd = comp["nd"]
    acc = np.zeros((shots, nd + 1), dtype=np.int64)
    for L, qi in zip(comp["locs"], q):
        if qi <= 0 or (late_only and not L["late"]):
            continue
        fired = np.flatnonzero(rng.random(shots) < qi)
        if fired.size == 0:
            continue
        rows = L["resp"][rng.integers(0, len(L["resp"]), fired.size)]
        np.add.at(acc, fired, rows)
    acc %= 3
    return acc[:, :nd], acc[:, nd]


def fid(k):
    return float(abs(np.mean(w ** k))) if k.size else float("nan")


def analyse(comp, det, lg):
    m = comp["m"]
    rounds = comp["rounds"]
    passed = np.ones(det.shape[0], dtype=bool)
    for r in range(m + 1):
        passed &= np.all(det[:, rounds == r] == 0, axis=1)
    term = det[:, rounds == m]
    term_ok = np.all(term == 0, axis=1)
    key = term @ np.array([27, 9, 3, 1])
    half = det.shape[0] // 2
    guess = np.zeros(81, dtype=int)
    for s in range(81):
        sel = key[:half] == s
        if sel.any():
            guess[s] = int(np.argmax(np.bincount(lg[:half][sel], minlength=3)))
    corr = (lg[half:] - guess[key[half:]]) % 3
    return dict(rej_u1=1 - passed.mean(), F_u1=fid(lg[passed]), rej_c1=1 - term_ok.mean(),
                F_c1=fid(lg[term_ok]), F_raw=fid(lg), F_dec=fid(corr))


def oracle_bound(comp, q, shots):
    det, lg = sample(comp, q, shots, late_only=True)
    m = comp["m"]
    key = det[:, comp["rounds"] == m] @ np.array([27, 9, 3, 1])
    tot = 0.0
    for s in np.unique(key):
        sel = key == s
        tot += sel.mean() * abs(np.mean(w ** lg[sel]))
    return tot


def read_csv(pidx, pol, k):
    with open(f"{BASE}/results/LRB/{pidx}/{pol}_check_data/{k}.csv") as fh:
        r = list(csv.reader(fh))
    return [float(x) for x in r[1][1:]], [float(x) for x in r[3][1:]]


t0 = time.time()
comps = {(cl, di): compile_circuit(cl, di) for cl in SEQS for di in USE_DI}
print(f"compiled {len(comps)} circuits in {time.time() - t0:.0f} s")

variants = [("si1000", 0.015, 10), ("si1000", 1e-3, 4), ("si1000par", 1e-3, None), ("si1000par", 0.015, None),
            ("ion", None, None), ("ion_opt", None, None)]
for name, p, pidx in variants:
    print("=" * 100)
    print(f"model {name}" + (f", p = {p}" if p is not None else ""))
    if pidx is not None:
        fu, ru = read_csv(pidx, "unif", 1)
        fc, rc = read_csv(pidx, "const", 1)
    print(f"  {'m':>3s} {'rej_u1':>9s} {'F_u1':>7s} {'rej_c1':>8s} {'F_c1':>7s} {'F_raw':>7s} {'F_dec':>7s} {'B_or':>7s}"
          + ("   | CSV (30 seq): rej_u1   F_u1   rej_c1  F_c1" if pidx is not None else ""))
    for di in USE_DI:
        res = []
        for cl in SEQS:
            comp = comps[(cl, di)]
            q = probs_for(name, comp["locs"], p)
            det, lg = sample(comp, q, SHOTS)
            r = analyse(comp, det, lg)
            r["B_or"] = oracle_bound(comp, q, SHOTS)
            res.append(r)
        avg = {k: np.nanmean([r[k] for r in res]) for k in res[0]}
        line = (f"  {DEPTHS[di]:3d} {avg['rej_u1']:9.5f} {avg['F_u1']:7.4f} {avg['rej_c1']:8.5f} {avg['F_c1']:7.4f} "
                f"{avg['F_raw']:7.4f} {avg['F_dec']:7.4f} {avg['B_or']:7.4f}")
        if pidx is not None:
            j = DEPTHS.index(DEPTHS[di])
            line += f"   | {ru[j]:9.5f} {fu[j]:7.4f} {rc[j]:8.5f} {fc[j]:7.4f}"
        print(line)
print(f"total {time.time() - t0:.0f} s")
