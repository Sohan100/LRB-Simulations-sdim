#!/usr/bin/env python3
"""Critic check (round 2): posterior structure of the 'ambiguous' signatures
at circuit level (relevant to A1 claims C8, C14, C19).

For each detector signature reachable by a single fault, the leading-order
posterior of the most likely class is (max class weight)/(total weight).
The strict hybrid of A1 rejects every signature with more than one class.
Here the rejected weight is binned by that posterior, and a posterior
threshold tau (accept if posterior >= tau, decode to the majority class) is
evaluated at leading order: rejection and first-order failure (minority
weight of accepted groups, plus the undetected 4.95p).

Uses the propagator of critic_A1-syndromes_r2_circuit.py (validated there
against the repository CompactDetectorErrorModel).
Run: module load python/3.11; python critic_A1-syndromes_r2_lopsided.py
"""
import importlib.util
import sys
import types

import numpy as np

HERE = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/"
        "correction-decoders-qutrit-2c455b/studies/correction-decoders/qutrit/notes/scripts/")
# load only the function definitions of the circuit script (skip its main body)
src = open(HERE + "critic_A1-syndromes_r2_circuit.py").read()
cut = src.index("# ============================================================ 1. DEM validation")
mod = types.ModuleType("crit")
exec(compile(src[:cut], "crit_defs", "exec"), mod.__dict__)
RUN = mod.RUN
read_circuit = mod.read_circuit


def posterior_bins(circ, view="V1"):
    prop = mod.propagate(circ)
    nd = prop["det"].shape[0]
    extra = (mod.FUNC @ prop["data_final"]) % 3
    resp, w, L = mod.expand(prop, [prop["det"], prop["obs"], extra])
    if view == "V1":
        sig = resp[:, :nd]
        cls = resp[:, nd].astype(int)
        ncl = 3
    else:
        sig = np.hstack([resp[:, :nd], resp[:, nd + 1:nd + 5]])
        cls = (resp[:, nd + 5] + 3 * resp[:, nd + 6]).astype(int)
        ncl = 9
    s8 = np.ascontiguousarray(sig.astype(np.int8))
    kv = s8.view(np.dtype((np.void, s8.shape[1]))).ravel()
    _, first, g = np.unique(kv, return_index=True, return_inverse=True)
    g = g.reshape(-1)
    zero = ~np.any(s8[first] != 0, axis=1)
    Wm = np.zeros((len(first), ncl))
    np.add.at(Wm, (g, cls), w)
    tot = Wm.sum(1)
    mx = Wm.max(1)
    post = np.where(tot > 0, mx / np.where(tot > 0, tot, 1), 1)
    amb = ((Wm > 0).sum(1) > 1) & ~zero
    undet = Wm[zero][:, 1:].sum()
    bins = [(0.95, 1.0001), (0.8, 0.95), (0.6, 0.8), (0.0, 0.6)]
    rows = []
    for lo, hi in bins:
        m = amb & (post >= lo) & (post < hi)
        rows.append((lo, hi, tot[m].sum(), (tot - mx)[m].sum()))
    thr = []
    for tau in (1.0, 0.99, 0.95, 0.9, 0.8, 0.6, 0.0):
        acc = ~zero & (post >= tau - 1e-12)
        rej = tot[~zero & ~acc].sum()
        fail = (tot - mx)[acc].sum() + undet
        thr.append((tau, rej, fail))
    return rows, thr, tot[amb].sum()


cases = [("memory R=4", None, "mem4"), ("LRB seq 0 m=0", (0, 0), None), ("LRB seq 0 m=2", (0, 1), None),
         ("LRB seq 0 m=22", (0, 8), None), ("LRB seq 5 m=22", (5, 8), None)]
import dataclasses  # noqa: E402
cd = mod.CodeSimulationProfileRegistry._build_code_definition("folded_qutrit")
cd_mem = dataclasses.replace(cd, clifford_to_gate_sequence=lambda s: [], clifford_inverse_map={"E": []})
gen = mod.LRBCircuitGenerator(code_definition=cd_mem, noise_model=mod.SI1000_NOISE_MODEL)
for name, cl_di, tag in cases:
    if tag == "mem4":
        circ = gen.generate_lrb_clifford_sequence(depths=[4], clifford_strings=["E"] * 4)[0]
        gen.update_noise_param(circ, 1e-3)
    else:
        circ = read_circuit(f"{RUN}/experiments/LRB/{cl_di[0]}/1/{cl_di[1]}.chp")
    for view in ("V1", "V2"):
        rows, thr, hyb = posterior_bins(circ, view)
        print(f"{name} {view}: strict-hybrid rejection {hyb:.3f}")
        for lo, hi, rj, mn in rows:
            print(f"     ambiguous groups with posterior in [{lo:.2f},{min(hi, 1):.2f}): weight {rj:8.3f}, minority {mn:7.3f}")
        print("     threshold tau -> (rejection, first-order failure incl. undetected):",
              [(t, round(r, 2), round(f, 3)) for t, r, f in thr])
