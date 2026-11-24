#!/usr/bin/env python3
"""Critic of A3: fault budget and exact first-order (single-fault) logical error
budget of the real baseline LRB circuits.

Read-only on the baseline run folder.  The circuits are parsed directly and every
unit Weyl component of every noise location is propagated with a linear frame
propagator.  The same propagator reproduces the repository's compiled DEM
exactly (critic_A3-pauli-frame_gates_dem_check.py: 0 mismatches on 5 circuits).

For each circuit all single faults are enumerated (8 Weyl vectors per N1 'd',
2 per N1 'f', 80 per N2, each with probability p_loc / count).  First-order
quantities, in units of p (the .chp files at p_idx 8 hold p = 0.01; we divide):

  none        : sum over faults with logical shift L != 0
  MAP(all)    : faults grouped by the full detector pattern; failure of a
                group = total - max_k mass_k (pattern 0: all L != 0 faults)
  MAP(final)  : same, but the decoder sees only the round-m detectors
  LRB-D unif1 : accepted logical faults (pattern 0, L != 0), and rejection mass
  LRB-D const1: accepted iff round-m detectors are zero

Per-step coefficients are the slopes between depths.  The MAP failure is also
attributed to circuit segments (forward Clifford, check round 0..m-1, inverse
block, final round).  Individual forward-Clifford fault masses are collected
over all 30 Clifford sequences at m = 22.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python critic_A3-pauli-frame_first_order.py
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import defaultdict

import numpy as np

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_FILE = 0.01           # p_idx 8
ANC = {5, 6, 7, 8}
G1 = {"I", "X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV"}
G2 = {"CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"}
RULE1 = {"H": lambda a, b: (-b, a), "H_INV": lambda a, b: (b, -a),
         "P": lambda a, b: (a, a + b), "P_INV": lambda a, b: (a, b - a)}
NZ2 = [v for v in itertools.product(range(3), repeat=2) if any(v)]
NZ4 = [v for v in itertools.product(range(3), repeat=4) if any(v)]


def parse(path):
    ops = []
    for t in (ln.split() for ln in open(path)):
        if not t or t[0] in ("#", "Randomly-generated", "d", "DETECTOR",
                             "LOGICAL_OBSERVABLE", "TICK"):
            continue
        args = [int(x) for x in t[1:] if "=" not in x]
        params = dict(x.split("=", 1) for x in t[1:] if "=" in x)
        ops.append((t[0], args, {k: v.strip('"') for k, v in params.items()}))
    return ops


def segment_labels(ops, m):
    """Label every op: ('fwd', k) forward Clifford k (1..m), ('round', r),
    ('inv', 0) inverse block, ('final', m), or ('enc'/'term', -)."""
    labels = []
    completed = 0          # number of finished check rounds
    in_round = False
    m8_seen = False
    cur = None
    for name, args, _ in ops:
        is_gate = name in G1 or name in G2
        touches_anc = any(a in ANC for a in args)
        if is_gate and touches_anc and (not in_round or m8_seen):
            if in_round and m8_seen:          # empty Clifford between rounds
                completed += 1
            in_round, m8_seen, cur = True, False, completed
        elif is_gate and not touches_anc and in_round and m8_seen:
            in_round = False
            completed += 1
        if in_round:
            lab = ("final", m) if cur == m else ("round", cur)
        else:
            r = completed
            lab = ("fwd", r + 1) if r < m else (("inv", 0) if r == m else ("term", -1))
        if name == "M" and args[0] == 8 and in_round:
            m8_seen = True
        labels.append(lab)
    return labels


def unit_responses(ops):
    """Batched linear propagation of unit X and Z components of every noise op."""
    members = []
    for i, (n, a, pr) in enumerate(ops):
        if n == "N1" and float(pr["prob"]) > 0:
            members += [(i, c) for c in range(2)]
        elif n == "N2" and float(pr["prob"]) > 0:
            members += [(i, c) for c in range(4)]
    B = len(members)
    fa = np.zeros((B, 9), np.int64)
    fb = np.zeros((B, 9), np.int64)
    by_op = defaultdict(list)
    for k, (i, c) in enumerate(members):
        by_op[i].append((k, c))
    meas = []
    for i, (n, a, pr) in enumerate(ops):
        if n in RULE1:
            q = a[0]
            na, nb = RULE1[n](fa[:, q], fb[:, q])
            fa[:, q], fb[:, q] = na % 3, nb % 3
        elif n in G2:
            c, t = a
            ac, bc, at, bt = fa[:, c].copy(), fb[:, c].copy(), fa[:, t].copy(), fb[:, t].copy()
            if n == "CNOT":
                fb[:, c] = bc - bt; fa[:, t] = at + ac
            elif n == "CNOT_INV":
                fb[:, c] = bc + bt; fa[:, t] = at - ac
            elif n == "CZ":
                fb[:, c] = bc + at; fb[:, t] = bt + ac
            elif n == "CZ_INV":
                fb[:, c] = bc - at; fb[:, t] = bt - ac
            else:
                fa[:, c], fb[:, c], fa[:, t], fb[:, t] = at, bt, ac, bc
            fa %= 3; fb %= 3
        elif n == "M":
            meas.append(fa[:, a[0]].copy())
        elif n == "RESET":
            fa[:, a[0]] = 0; fb[:, a[0]] = 0
        elif n in ("N1", "N2"):
            for k, c in by_op.get(i, []):
                q = a[c // 2]
                if c % 2 == 0:
                    fa[k, q] = (fa[k, q] + 1) % 3
                else:
                    fb[k, q] = (fb[k, q] + 1) % 3
    meas = np.array(meas).T
    det = meas[:, :-2] % 3
    log = (meas[:, -2] + meas[:, -1]) % 3
    return members, det, log


def single_faults(ops, labels):
    members, det, log = unit_responses(ops)
    comp = defaultdict(dict)
    for k, (i, c) in enumerate(members):
        comp[i][c] = k
    P, DET, LOG, SEG = [], [], [], []
    for i, (n, a, pr) in enumerate(ops):
        if i not in comp:
            continue
        prob = float(pr["prob"]) / P_FILE
        ch = pr.get("noise_channel", pr.get("channel", "d"))
        if n == "N1":
            vecs = NZ2 if ch == "d" else ([(1, 0), (2, 0)] if ch == "f" else [(0, 1), (0, 2)])
        else:
            vecs = NZ4
        ks = [comp[i][c] for c in range(len(vecs[0]))]
        V = np.array(vecs)                       # (nv, ncomp)
        D = (V @ det[ks]) % 3                    # (nv, ndet)
        L = (V @ log[ks]) % 3
        P.append(np.full(len(V), prob / len(V)))
        DET.append(D)
        LOG.append(L)
        SEG += [labels[i]] * len(V)
    return (np.concatenate(P), np.concatenate(DET), np.concatenate(LOG), SEG)


def first_order(P, DET, LOG, SEG, m):
    nd = DET.shape[1]
    res = {}
    res["none"] = P[LOG != 0].sum()
    zero = ~DET.any(axis=1)
    res["unif1_acc_fail"] = P[zero & (LOG != 0)].sum()
    res["unif1_reject"] = P[~zero].sum()
    fin = DET[:, nd - 4:]
    zf = ~fin.any(axis=1)
    res["const1_acc_fail"] = P[zf & (LOG != 0)].sum()

    def map_fail(keys):
        groups = defaultdict(lambda: np.zeros(3))
        idx = defaultdict(list)
        for j, key in enumerate(keys):
            groups[key][LOG[j]] += P[j]
            idx[key].append(j)
        total = 0.0
        attr = defaultdict(float)
        for key, mass in groups.items():
            if not any(key):
                win = {0}
            else:
                mx = mass.max()
                win = set(np.flatnonzero(mass >= mx * (1 - 1e-12)).tolist())
            for j in idx[key]:
                # expected failure with uniform random tie-breaking among winners
                pf = 1.0 - (1.0 / len(win) if LOG[j] in win else 0.0)
                total += P[j] * pf
                attr[SEG[j][0]] += P[j] * pf
        return total, attr

    keys_all = [r.tobytes() for r in DET]
    # zero-pattern detection must use the array, not the bytes; handle via any()
    keys_all = [k if DET[j].any() else b"" for j, k in enumerate(keys_all)]
    res["map_all"], res["map_attr"] = map_fail(keys_all)
    keys_fin = [r.tobytes() if r.any() else b"" for r in fin]
    res["map_final"], _ = map_fail(keys_fin)
    seg_mass = defaultdict(float)
    for j, s in enumerate(SEG):
        seg_mass[s[0]] += P[j]
    res["seg_mass"] = seg_mass
    return res


def clifford_masses(ops, labels, m):
    mass = defaultdict(float)
    for (n, a, pr), lab in zip(ops, labels):
        if n in ("N1", "N2") and lab[0] == "fwd":
            mass[lab[1]] += float(pr["prob"]) / P_FILE
    return [mass[k] for k in range(1, m + 1)]


def main():
    t0 = time.time()
    cliffs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 5]
    dis = [0, 1, 3, 4, 5, 8]
    print("# first-order coefficients in units of p (per circuit)")
    print("# cliff m | mass: fwdC rounds inv final | none  MAP(all)  MAP(final)  "
          "unif1-acc-fail unif1-rej  const1-acc-fail | MAP attribution fwd/round/inv/final")
    table = defaultdict(dict)
    for c in cliffs:
        for di in dis:
            m = DEPTHS[di]
            ops = parse(f"{RUN}/experiments/LRB/{c}/8/{di}.chp")
            labels = segment_labels(ops, m)
            P, DET, LOG, SEG = single_faults(ops, labels)
            r = first_order(P, DET, LOG, SEG, m)
            table[c][m] = r
            sm, at = r["seg_mass"], r["map_attr"]
            print(f"  {c:2d} {m:3d} | {sm.get('fwd', 0):7.1f} {sm.get('round', 0):7.1f} "
                  f"{sm.get('inv', 0):6.1f} {sm.get('final', 0):6.1f} | {r['none']:7.2f} "
                  f"{r['map_all']:8.3f} {r['map_final']:9.3f}   {r['unif1_acc_fail']:8.3f} "
                  f"{r['unif1_reject']:9.1f}   {r['const1_acc_fail']:8.3f} | "
                  f"{at.get('fwd', 0):.3f}/{at.get('round', 0):.3f}/{at.get('inv', 0):.3f}/"
                  f"{at.get('final', 0):.3f}")
    print("\n# per-step slopes (units of p per logical step) between m=6 and m=22")
    for c in cliffs:
        a, b = table[c][6], table[c][22]
        sl = {k: (b[k] - a[k]) / 16 for k in ("none", "map_all", "map_final",
                                              "unif1_acc_fail", "const1_acc_fail")}
        att = {k: (b["map_attr"].get(k, 0) - a["map_attr"].get(k, 0)) / 16
               for k in ("fwd", "round", "inv", "final")}
        print(f"  cliff {c}: " + ", ".join(f"{k} {v:.3f}" for k, v in sl.items())
              + " | MAP slope by segment: " + ", ".join(f"{k} {v:.3f}" for k, v in att.items()))
    # individual forward Clifford masses over all 30 sequences at m = 22
    allm = []
    per_seq = []
    for c in range(30):
        ops = parse(f"{RUN}/experiments/LRB/{c}/8/8.chp")
        labels = segment_labels(ops, 22)
        ms = clifford_masses(ops, labels, 22)
        allm += ms
        per_seq.append(np.mean(ms))
    allm = np.array(allm)
    vals, cnts = np.unique(np.round(allm, 2), return_counts=True)
    print(f"\n# individual forward-Clifford fault mass over 30 sequences x 22 Cliffords "
          f"(units of p): min {allm.min():.2f}, max {allm.max():.2f}, mean {allm.mean():.2f}")
    print("  histogram: " + ", ".join(f"{v:.1f}p x{n}" for v, n in zip(vals, cnts)))
    print(f"  per-sequence mean: min {min(per_seq):.2f}, max {max(per_seq):.2f}")
    print(f"elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
