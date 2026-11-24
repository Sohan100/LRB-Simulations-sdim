#!/usr/bin/env python3
"""Critic check (round 2) of A1 claim C18, memory part: second-order
coefficients of P(fail|acc) per bulk round in the V2 view (detectors plus a
perfect final syndrome, full logical class (a,b)), for memory windows R=0..3
built with the repository generator.  Own pair enumeration with exact base-3
signature keys.  Detection: accept iff all-zero signature.  Hybrid: accept a
signature iff all single-fault explanations share one class (zero signature
-> class 0); signatures without single-fault explanation cannot fail at p^2.
Run: module load python/3.11; python critic_A1-syndromes_r2_order2v2.py
"""
import dataclasses
import time
import types

import numpy as np

HERE = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/"
        "correction-decoders-qutrit-2c455b/docs/correction-decoders/qutrit-512/notes/scripts/")
src = open(HERE + "critic_A1-syndromes_r2_circuit.py").read()
cut = src.index("# ============================================================ 1. DEM validation")
mod = types.ModuleType("crit")
exec(compile(src[:cut], "crit_defs", "exec"), mod.__dict__)


def order2_v2(circ):
    prop = mod.propagate(circ)
    nd = prop["det"].shape[0]
    extra = (mod.FUNC @ prop["data_final"]) % 3
    resp, w, L = mod.expand(prop, [prop["det"], extra])
    sig = resp[:, :nd + 4].astype(np.int64)
    cls = (resp[:, nd + 4] + 3 * resp[:, nd + 5]).astype(np.int64)
    ns = sig.shape[1]
    assert ns <= 39
    p3 = 3 ** np.arange(ns, dtype=np.int64)
    key = sig @ p3
    sc = np.array([l["sc"] for l in prop["locs"]])
    Ttot = sc.sum()
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    Wg = np.zeros((len(uk), 9))
    np.add.at(Wg, (inv, cls), w)
    acc = (Wg > 0).sum(1) == 1
    best = Wg.argmax(1)
    iz = np.searchsorted(uk, 0)
    if iz < len(uk) and uk[iz] == 0:
        acc[iz] = True
        best[iz] = 0
    U1 = w[(key == 0) & (cls != 0)].sum()
    D1 = w[key != 0].sum()
    H1 = w[~acc[inv]].sum()
    A_det = A_hyb = 0.0
    a_, b_ = cls % 3, cls // 3
    for i in range(len(w) - 1):
        j = np.arange(i + 1, len(w))
        j = j[L[j] != L[i]]
        k2 = ((sig[i] + sig[j]) % 3) @ p3
        c2 = (a_[i] + a_[j]) % 3 + 3 * ((b_[i] + b_[j]) % 3)
        w2 = w[i] * w[j]
        A_det += w2[(k2 == 0) & (c2 != 0)].sum()
        pos = np.minimum(np.searchsorted(uk, k2), len(uk) - 1)
        hit = (uk[pos] == k2) & acc[pos]
        A_hyb += w2[hit & (c2 != best[pos])].sum()
    assert U1 == 0
    return dict(D1=D1, H1=H1, det2=A_det, hyb2=A_hyb, n=len(w))


cd = mod.CodeSimulationProfileRegistry._build_code_definition("folded_qutrit")
cd_mem = dataclasses.replace(cd, clifford_to_gate_sequence=lambda s: [], clifford_inverse_map={"E": []})
gen = mod.LRBCircuitGenerator(code_definition=cd_mem, noise_model=mod.SI1000_NOISE_MODEL)
prev = None
for R in range(0, 4):
    t = time.time()
    circ = gen.generate_lrb_clifford_sequence(depths=[R], clifford_strings=["E"] * R)[0]
    gen.update_noise_param(circ, 1e-3)
    r = order2_v2(circ)
    inc = "" if prev is None else (f"  increments: det {r['det2'] - prev['det2']:.3f},"
                                   f" hyb {r['hyb2'] - prev['hyb2']:.3f}")
    print(f"R={R}: faults {r['n']}, det rej {r['D1']:.3f}, hyb rej {r['H1']:.3f},"
          f" det P(f|a) p^2 {r['det2']:.3f}, hyb P(f|a) p^2 {r['hyb2']:.3f}{inc} ({time.time() - t:.0f} s)",
          flush=True)
    prev = r
