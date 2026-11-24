#!/usr/bin/env python3
"""Critic (v2) of A3-boundary-C1/C4/C6/C7 using the repository's own DEM.

Read-only on the baseline run folder.  Circuits are read with sdim's reader and
compiled with the repository's `src/lrb/dem_simulation.py`
(`build_compact_detector_error_model`), i.e. the exact linear responses the
production runs sampled from.  Nothing is written outside stdout.

Part A  Fault budget per segment (own segmentation: data-only gate blocks vs
        check rounds), in units of p.
Part B  Exact joint distribution P(syndrome, logical shift) by convolution over
        all noise locations, for m = 0 (4 detectors) and m = 2 (12 detectors),
        at p = 1e-3 and 1e-2.  Reports LRB-D acceptance / accepted fidelity
        (cross-check against the baseline CSVs), and the exact corrected
        fidelity of: no correction, MAP (symmetric ties), and the exact
        optimum of |E w^R|.  Compares with A3's bound F(0) <= P_acc F_acc + 1 - P_acc.
        Also the invisible-fault factor prod_loc (1 - 3/2 P_loc(D=0, L!=0)).
Part C  First-order (single-fault) budget of 1 - Re E[w^R] in units of p for
        m = 2, 6, 10, 14, 22 and three Clifford sequences: no correction, MAP
        on all rounds, MAP on the final round only, LRB-D unif-1 and const-1
        (accepted shots), and the attribution of the MAP failure to segments.
        Per-step coefficients are slopes between depths.

Usage: module load python/3.11; python critic_A3-pauli-frame_v2_dem.py
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import time

import numpy as np
from sdim.circuit_io import read_circuit

ROOT = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/"
        "correction-decoders-qutrit-2c455b")
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_CIRC = 0.01                                   # p of the p_idx = 8 circuit files
w = np.exp(2j * np.pi / 3)

_spec = importlib.util.spec_from_file_location(
    "dem_simulation", os.path.join(ROOT, "src/lrb/dem_simulation.py"))
dem = importlib.util.module_from_spec(_spec)
sys.modules["dem_simulation"] = dem
_spec.loader.exec_module(dem)


def load_model(cliff, depth_idx):
    circ = read_circuit(f"{RUN}/experiments/LRB/{cliff}/8/{depth_idx}.chp")
    model = dem.build_compact_detector_error_model(circ)
    return circ, model


def segment_labels(circ):
    """Noise op index -> ('C', k) data-only block k, or ('R', r) check round r."""
    labels, state, k, r = {}, None, -1, -1
    for i, op in enumerate(circ.operations):
        gid = op.gate_id
        if 0 <= gid <= 13:
            qs = [op.qudit_index]
            if gid >= 9 and op.target_index is not None:
                qs.append(op.target_index)
            anc = any(q >= 5 for q in qs)
            if anc and state != "R":
                state, r = "R", r + 1
            elif not anc and state != "C":
                state, k = "C", k + 1
        if gid in (17, 18):
            labels[i] = (state, k if state == "C" else r)
    return labels


def location_faults(loc):
    """List of (probability, Weyl vector) for one compiled location (absolute p)."""
    P = loc.probability
    if loc.gate == "N1":
        if loc.channel == "d":
            vs = [v for v in itertools.product(range(3), repeat=2) if any(v)]
            return [(P / 8, v) for v in vs]
        if loc.channel == "f":
            return [(P / 2, (1, 0)), (P / 2, (2, 0))]
        if loc.channel == "p":
            return [(P / 2, (0, 1)), (P / 2, (0, 2))]
        raise ValueError(loc.channel)
    dist = np.asarray(loc.params["prob_dist"], float)
    out = []
    for idx in range(1, 81):
        if dist[idx] > 0:
            v = (idx // 27 % 3, idx // 9 % 3, idx // 3 % 3, idx % 3)
            out.append((dist[idx], v))
    return out


def response(loc, v, ndet):
    det = np.zeros(ndet, int)
    L = 0
    for c, vc in enumerate(v):
        if vc == 0:
            continue
        for di, coef in loc.component_terms[c].detector:
            det[di] = (det[di] + vc * coef) % 3
        for li, coef in loc.component_terms[c].logical:
            L = (L + vc * coef) % 3
    return det, L


# ---------------------------------------------------------------- Part A

def part_a():
    print("Part A: fault mass per segment (units of p), own segmentation")
    for cliff, di in ((0, 1), (0, 8), (5, 8)):
        circ, model = load_model(cliff, di)
        m = DEPTHS[di]
        seg = segment_labels(circ)
        tot = {}
        for loc in model.locations:
            key = seg[loc.operation_index]
            tot[key] = tot.get(key, 0.0) + loc.probability / P_CIRC
        fwd = [tot.get(("C", k), 0.0) for k in range(m)]
        rounds = [tot.get(("R", r), 0.0) for r in range(m)]
        inv = tot.get(("C", m), 0.0)
        fin = tot.get(("R", m), 0.0)
        print(f"  seq {cliff} m={m}: forward Cliffords {sum(fwd):.1f} p "
              f"(min {min(fwd):.1f}, max {max(fwd):.1f}, mean {np.mean(fwd):.2f}); "
              f"rounds 0..m-1 mean {np.mean(rounds):.1f} p; inverse block {inv:.1f} p "
              f"({inv / m:.2f} p per inverse Clifford); final round {fin:.1f} p; "
              f"ratio round/Clifford {np.mean(rounds) / np.mean(fwd):.1f}")


# ---------------------------------------------------------------- Part B

class Joint:
    """Exact P(detectors, logical) on Z_3^(ndet+1), flattened, by convolution."""

    def __init__(self, nd):
        self.n = nd + 1
        self.size = 3 ** self.n
        self.pw = 3 ** np.arange(self.n - 1, -1, -1)
        idx = np.arange(self.size)
        self.dig = ((idx[:, None] // self.pw[None, :]) % 3).astype(np.int8)
        self.cache = {}

    def perm_minus(self, r):
        key = tuple(r)
        if key not in self.cache:
            sh = ((self.dig - np.asarray(r, dtype=np.int8)) % 3).astype(np.int64)
            self.cache[key] = sh @ self.pw
        return self.cache[key]

    def run(self, model, p):
        P = np.zeros(self.size)
        P[0] = 1.0
        nd = self.n - 1
        for loc in model.locations:
            scale = p / P_CIRC
            agg = {}
            for prob, v in location_faults(loc):
                det, L = response(loc, v, nd)
                key = tuple(det) + (L,)
                agg[key] = agg.get(key, 0.0) + prob * scale
            ptot = sum(agg.values())
            new = (1 - ptot) * P
            for key, pr in agg.items():
                if not any(key):
                    new = new + pr * P
                else:
                    new = new + pr * P[self.perm_minus(key)]
            P = new
        return P.reshape(-1, 3)              # rows: syndrome index, cols: L


def decoders_exact(PL):
    a = PL @ (w ** np.arange(3))
    res = {"none": a.sum()}
    mx = PL.max(axis=1, keepdims=True)
    mask = (PL >= mx * (1 - 1e-12)) & (mx > 0)
    wts = mask / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    res["MAP"] = np.sum((wts @ (w ** (-np.arange(3)))) * a)
    best = 0.0
    nz = np.flatnonzero(PL.sum(axis=1) > 0)
    rot = np.stack([w ** (-k) * a[nz] for k in range(3)])     # (3, S)
    for phi in np.linspace(0, 2 * np.pi / 3, 241):
        val = np.max((np.exp(-1j * phi) * rot).real, axis=0).sum()
        best = max(best, val)
    res["opt|.|"] = best
    acc = PL[0].sum()
    res["Pacc"] = acc
    res["Facc"] = abs(a[0]) / acc
    return res


def invisible_factor(model, p, nd):
    fac = 1.0
    for loc in model.locations:
        mass = 0.0
        for prob, v in location_faults(loc):
            det, L = response(loc, v, nd)
            if not det.any() and L != 0:
                mass += prob * p / P_CIRC
        fac *= 1 - 1.5 * mass
    return fac


def part_b():
    print("\nPart B: exact joint distributions from the repository DEM")
    csv = {}
    for pidx in (4, 8):
        rows = open(f"{RUN}/results/LRB/{pidx}/unif_check_data/1.csv").read().split("\n")
        csv[pidx] = rows[:5]
    for di in (0, 1):
        m = DEPTHS[di]
        for cliff in (0,):
            circ, model = load_model(cliff, di)
            nd = model.num_detectors
            J = Joint(nd)
            for p in (1e-3, 1e-2):
                t0 = time.time()
                PL = J.run(model, p)
                r = decoders_exact(PL)
                bound = r["Pacc"] * r["Facc"] + 1 - r["Pacc"]
                lam = invisible_factor(model, p, nd)
                print(f"  m={m} seq {cliff} p={p:g}: LRB-D unif1 P_acc={r['Pacc']:.6f} "
                      f"F_acc={r['Facc']:.6f} | corrected |E w^R|: none={abs(r['none']):.5f} "
                      f"MAP={abs(r['MAP']):.5f} (Re {r['MAP'].real:.5f}) "
                      f"opt|.|={r['opt|.|']:.5f} | A3-type bound={bound:.5f} | "
                      f"invisible factor={lam:.5f}  [{time.time() - t0:.1f}s]")
    print("  baseline CSV rows (unif_check_data/1.csv; Fidelity / Rejected at each depth):")
    for pidx, rows in csv.items():
        print(f"   p_idx {pidx}: {rows[1][:80]} ... | {rows[3][:80]} ...")


# ---------------------------------------------------------------- Part C

def first_order(cliff, di):
    circ, model = load_model(cliff, di)
    m = DEPTHS[di]
    nd = model.num_detectors
    seg = segment_labels(circ)
    final_idx = [i for i, lab in enumerate(model.detector_labels)
                 if lab.startswith(f"lrb_stab_r{m}_")]
    groups, groups_fin = {}, {}
    faults = []
    for loc in model.locations:
        lab = seg[loc.operation_index]
        for prob, v in location_faults(loc):
            det, L = response(loc, v, nd)
            mass = prob / P_CIRC                       # units of p
            key = tuple(det)
            kfin = tuple(det[final_idx])
            faults.append((mass, key, kfin, L, lab))
            groups.setdefault(key, np.zeros(3))[L] += mass
            groups_fin.setdefault(kfin, np.zeros(3))[L] += mass
    zero = tuple([0] * nd)
    zfin = tuple([0] * len(final_idx))

    def fail(L, row, is_zero):
        if is_zero:                                    # no-fault event dominates: decide 0
            return 1.5 * (L != 0)
        ties = np.flatnonzero(row >= row.max() * (1 - 1e-12))
        return np.mean([1.5 * (L != t) for t in ties])

    out = {"none": 0.0, "MAP": 0.0, "MAPfinal": 0.0, "unif1": 0.0, "const1": 0.0,
           "rej1": 0.0}
    attrib = {}
    for mass, key, kfin, L, lab in faults:
        out["none"] += mass * 1.5 * (L != 0)
        fm = mass * fail(L, groups[key], key == zero)
        out["MAP"] += fm
        out["MAPfinal"] += mass * fail(L, groups_fin[kfin], kfin == zfin)
        if key == zero:
            out["unif1"] += mass * 1.5 * (L != 0)
        else:
            out["rej1"] += mass
        if kfin == zfin:
            out["const1"] += mass * 1.5 * (L != 0)
        if lab[0] == "C":
            seg_name = "fwd-Clifford" if lab[1] < m else (
                "inverse-block" if lab[1] == m else "other")
        else:
            seg_name = "check-round" if lab[1] < m else "final-round"
        attrib[seg_name] = attrib.get(seg_name, 0.0) + fm
    return m, out, attrib


def part_c():
    print("\nPart C: first-order 1 - Re E[w^R] in units of p (single-fault enumeration)")
    dis = [1, 3, 4, 5, 8]
    for cliff in (0, 1, 2):
        res = []
        for di in dis:
            t0 = time.time()
            m, out, attrib = first_order(cliff, di)
            res.append((m, out, attrib))
            print(f"  seq {cliff} m={m:2d}: none {out['none']:7.2f}  MAP {out['MAP']:7.2f}  "
                  f"MAP-final-only {out['MAPfinal']:7.2f}  LRB-D unif1(acc) {out['unif1']:6.2f}"
                  f" (reject mass {out['rej1']:7.1f})  const1(acc) {out['const1']:6.2f}   "
                  f"MAP attribution: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(attrib.items()))
                  + f"  [{time.time() - t0:.1f}s]")
        (m0, o0, a0), (m1, o1, a1) = res[0], res[-1]
        dm = m1 - m0
        print(f"  seq {cliff} per-step slopes between m={m0} and m={m1}: "
              + ", ".join(f"{k} {(o1[k] - o0[k]) / dm:.2f}" for k in
                          ("none", "MAP", "MAPfinal", "unif1", "const1"))
              + " ; MAP slope by segment: "
              + ", ".join(f"{k} {(a1.get(k, 0) - a0.get(k, 0)) / dm:.2f}"
                          for k in ("fwd-Clifford", "check-round", "inverse-block",
                                    "final-round")))


if __name__ == "__main__":
    t = time.time()
    part_a()
    part_b()
    part_c()
    print(f"elapsed {time.time() - t:.1f} s")
