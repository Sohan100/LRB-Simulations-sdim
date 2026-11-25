#!/usr/bin/env python3
"""Critic check for N1, part 3: exact low-order fault analysis of the stored circuits.

Independent of src/lrb/correction/compact_dem.py.  For each stored LRB circuit
(experiments/LRB/<seq>/<p_idx>/<d>.chp and experiments/LRB_const0/...):

  1. parse gates, noise locations (N1 d: 8 Weyl ops; N1 f: X^a, a = 1, 2;
     N2: 80 two-qutrit Weyl ops; probability = the `prob` attribute, which the
     RB check showed is the probability actually used), measurements, resets,
     DETECTOR / LOGICAL_OBSERVABLE expressions;
  2. back-propagate every measurement record (Z on the measured qutrit) through
     the circuit (Heisenberg picture, conjugation by U^-1; RESET zeroes the
     qutrit) and so obtain, for every fault component, its exact Z_3 shift of
     every detector and of the logical observable;
  3. for a policy (= set of used detectors) compute
       R1  first-order rejection probability,
       U1  first-order probability of an accepted logical error,
       U2  second-order (two-fault) probability of an accepted logical error,
     by aggregating fault components into shift classes and pairing classes
     whose shifts cancel on every used detector but not on the logical.
     Pairs at the same location are excluded.  Pairs can be restricted to
     both faults in the inverse half, both in the forward half, etc.
Predictions (per sequence, averaged over the 30 sequences) are compared with
the baseline CSVs at small p.
Usage: module load python/3.11 && python critic_N1-baseline-nonmarkov_faults.py s0 s1 out_dir
       (sequences s0..s1-1; run three chunks of 10 in parallel, ~3 min each)
"""
import csv
import itertools
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
NQ = 9
PIDX = 4            # circuits at p = 1e-3; coefficients are divided by P_REF
P_REF = 1e-3
ATTR = re.compile(r'(\w+)="([^"]*)"')
REC = re.compile(r'([+-]?)\s*rec\[(\d+)\]')

N1D = [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]
N1F = [(1, 0), (2, 0)]
N2C = [c for c in itertools.product(range(3), repeat=4) if any(c)]   # (x1,z1,x2,z2)


def parse(path):
    ops = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(("#", "Random")) or line.startswith("d "):
                continue
            head = line.split()[0]
            attrs = dict(ATTR.findall(line))
            body = ATTR.sub("", line).split()[1:]
            if head in ("DETECTOR", "LOGICAL_OBSERVABLE"):
                terms = [(-1 if s == "-" else 1, int(i)) for s, i in REC.findall(attrs["expr"])]
                ops.append((head, terms, attrs.get("label", "")))
            else:
                ops.append((head, [int(q) for q in body], attrs))
    return ops


# conjugation by U^-1 (backward propagation of an observable through gate U)
def back_gate(O, name, qs):
    X, Z = O[:, :NQ], O[:, NQ:]
    if name == "H":                      # U^-1 = H_INV: (x,z) -> (z,-x)
        q = qs[0]; x, z = X[:, q].copy(), Z[:, q].copy(); X[:, q] = z; Z[:, q] = -x
    elif name == "H_INV":                # U^-1 = H: (x,z) -> (-z, x)
        q = qs[0]; x, z = X[:, q].copy(), Z[:, q].copy(); X[:, q] = -z; Z[:, q] = x
    elif name == "P":                    # U^-1 = P_INV: z -> z - x
        q = qs[0]; Z[:, q] -= X[:, q]
    elif name == "P_INV":
        q = qs[0]; Z[:, q] += X[:, q]
    elif name == "CNOT":                 # U^-1 = CNOT_INV: x_t -= x_c, z_c += z_t
        c, t = qs; X[:, t] -= X[:, c]; Z[:, c] += Z[:, t]
    elif name == "CNOT_INV":             # U^-1 = CNOT: x_t += x_c, z_c -= z_t
        c, t = qs; X[:, t] += X[:, c]; Z[:, c] -= Z[:, t]
    elif name == "CZ":                   # U^-1 = CZ_INV: z_a -= x_b, z_b -= x_a
        a, b = qs; xa, xb = X[:, a].copy(), X[:, b].copy(); Z[:, a] -= xb; Z[:, b] -= xa
    elif name == "CZ_INV":
        a, b = qs; xa, xb = X[:, a].copy(), X[:, b].copy(); Z[:, a] += xb; Z[:, b] += xa
    elif name == "SWAP":
        a, b = qs
        X[:, [a, b]] = X[:, [b, a]]; Z[:, [a, b]] = Z[:, [b, a]]
    elif name in ("X", "Z", "X_INV", "Z_INV", "I"):
        pass
    else:
        raise ValueError(name)
    O %= 3


def fault_table(ops):
    """Return (locations, det_coef, n_det): per noise location its prob, kind,
    qubits, op index, and the shift matrix (components x (n_det + 1))."""
    nrec = sum(1 for o in ops if o[0] == "M")
    dets = [o for o in ops if o[0] == "DETECTOR"]
    logi = [o for o in ops if o[0] == "LOGICAL_OBSERVABLE"]
    C = np.zeros((len(dets) + 1, nrec), dtype=np.int64)
    for i, d in enumerate(dets + logi):
        for s, r in d[1]:
            C[i, r] = (C[i, r] + s) % 3
    O = np.zeros((nrec, 2 * NQ), dtype=np.int64)
    rec = nrec
    locs = []
    for idx in range(len(ops) - 1, -1, -1):
        name, qs, attrs = ops[idx]
        if name in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            continue
        if name == "M":
            rec -= 1
            O[rec, :] = 0
            O[rec, NQ + qs[0]] = 1          # record observable: Z on the measured qutrit
        elif name == "RESET":
            O[:, qs[0]] = 0; O[:, NQ + qs[0]] = 0
        elif name in ("N1", "N2"):
            p = float(attrs["prob"])
            if name == "N1":
                comps = N1D if attrs["noise_channel"] == "d" else (N1F if attrs["noise_channel"] == "f" else None)
                q = qs[0]
                V = np.array(comps)                         # (K, 2): (x, z)
                # record shift = x_v * Oz - z_v * Ox
                shift = (np.outer(V[:, 0], O[:, NQ + q]) - np.outer(V[:, 1], O[:, q])) % 3
            else:
                a, b = qs
                V = np.array(N2C)
                shift = (np.outer(V[:, 0], O[:, NQ + a]) - np.outer(V[:, 1], O[:, a])
                         + np.outer(V[:, 2], O[:, NQ + b]) - np.outer(V[:, 3], O[:, b])) % 3
            S = (shift @ C.T) % 3                           # (K, n_det + 1)
            locs.append(dict(idx=idx, p=p, K=len(V), S=S.astype(np.int8), qs=tuple(qs)))
        else:
            back_gate(O, name, qs)
    assert rec == 0
    locs.reverse()
    return locs, len(dets)


def round_marks(ops):
    """Op index of each measurement of ancilla 5 (start of each round's Z-block
    measurement) -- used to split forward / inverse halves."""
    return [i for i, o in enumerate(ops) if o[0] == "M" and o[1] == [5]]


_HASH_RNG = np.random.default_rng(12345)
_R1 = _HASH_RNG.integers(1, 2**31 - 1, size=4096, dtype=np.int64)
_R2 = _HASH_RNG.integers(1, 2**31 - 1, size=4096, dtype=np.int64)
_PM = np.int64(2**31 - 1)


def _key(S):
    """Collision-resistant int64 identifier of each row of S (two random
    integer hashes mod 2^31 - 1; used only as a dictionary key)."""
    n = S.shape[1]
    S64 = S.astype(np.int64)
    return ((S64 @ _R1[:n]) % _PM) * np.int64(2**31) + (S64 @ _R2[:n]) % _PM


def _target_rows(S, l):
    """Rows (-c + l e_L) mod 3, e_L = last coordinate (the logical)."""
    T = (2 * S.astype(np.int16)) % 3
    T[:, -1] = (T[:, -1] + l) % 3
    return T.astype(np.int8)


def analyse(locs, used, region_of, regions):
    """First- and second-order quantities for one policy, resolved by region.
    used: boolean mask over detectors.  Returns a flat dict (units: probability)."""
    keyidx = np.concatenate([np.flatnonzero(used), [len(used)]])
    out = {}
    blocks = {r: ([], []) for r in regions}     # shift rows, weights
    for r in regions:
        out[f"P_{r}"] = out[f"R1_{r}"] = out[f"U1_{r}"] = out[f"H_{r}"] = 0.0
    same = {r: 0.0 for r in regions}
    for L in locs:
        r = region_of(L)
        S = L["S"][:, keyidx]
        pi = L["p"] / L["K"]
        det = np.any(S[:, :-1] != 0, axis=1)
        log = S[:, -1] != 0
        out[f"P_{r}"] += L["p"]
        out[f"R1_{r}"] += pi * det.sum()
        out[f"U1_{r}"] += pi * np.sum(~det & log)
        out[f"H_{r}"] += pi * np.sum(~det & ~log)
        blocks[r][0].append(S); blocks[r][1].append(np.full(len(S), pi))
        # same-location ordered pairs whose sum is an accepted logical error
        k = _key(S)
        for l in (1, 2):
            t = _key(_target_rows(S, l))
            same[r] += pi * pi * float(np.sum(t[:, None] == k[None, :]))
    agg = {}
    for r in regions:
        if not blocks[r][0]:
            agg[r] = None
            continue
        S = np.concatenate(blocks[r][0]); w = np.concatenate(blocks[r][1])
        k = _key(S)
        uk, first, inv = np.unique(k, return_index=True, return_inverse=True)
        W = np.bincount(inv.ravel(), weights=w)
        agg[r] = (S[first], W, uk)

    def pair_sum(ra, rb):
        if agg[ra] is None or agg[rb] is None:
            return 0.0
        Sa, Wa, _ = agg[ra]
        _, Wb, ukb = agg[rb]
        tot = 0.0
        for l in (1, 2):
            t = _key(_target_rows(Sa, l))
            pos = np.clip(np.searchsorted(ukb, t), 0, len(ukb) - 1)
            hit = ukb[pos] == t
            tot += float(np.sum(Wa[hit] * Wb[pos[hit]]))
        return tot
    for i, ra in enumerate(regions):
        for rb in regions[i:]:
            if ra == rb:
                u2 = 0.5 * (pair_sum(ra, ra) - same[ra])
                net = u2 - out[f"U1_{ra}"] * (out[f"H_{ra}"] + out[f"U1_{ra}"])
            else:
                u2 = pair_sum(ra, rb)
                net = (u2 - out[f"U1_{ra}"] * (out[f"H_{rb}"] + out[f"U1_{rb}"])
                       - out[f"U1_{rb}"] * (out[f"H_{ra}"] + out[f"U1_{ra}"]))
            out[f"U2_{ra}_{rb}"] = u2
            out[f"N2_{ra}_{rb}"] = net       # net 2nd-order term of P(err | acc)
    out["R1"] = sum(out[f"R1_{r}"] for r in regions)
    out["U1"] = sum(out[f"U1_{r}"] for r in regions)
    out["N2"] = sum(v for kk, v in out.items() if kk.startswith("N2_"))
    return out


REGIONS = ("fwd", "last", "inv", "term")


def regions_for(ops):
    """Op-index boundaries: 'fwd' = forward part before the last forward check
    round, 'last' = the last forward check round incl. its reset layer,
    'inv' = inverse logical blocks, 'term' = terminal check round."""
    resets = [i for i, o in enumerate(ops) if o[0] == "RESET"]
    z_starts = [i for i, o in enumerate(ops) if o[0] == "CNOT" and o[1] == [0, 5]]
    term_start = z_starts[-1]
    if resets:
        inv_start = next(i for i, o in enumerate(ops) if i > resets[-1]
                         and o[0] not in ("N1", "N2", "RESET"))
        last_start = z_starts[-2]
    else:
        inv_start = last_start = term_start
    def region_of(L):
        i = L["idx"]
        if i < last_start:
            return "fwd"
        if i < inv_start:
            return "last"
        if i < term_start:
            return "inv"
        return "term"
    return region_of


def run_range(s0, s1, out_path):
    t0 = time.time()
    rows = []
    for s in range(s0, s1):
        for d, m in enumerate(DEPTHS):
            ops = parse(os.path.join(RUN, "experiments", "LRB", str(s), str(PIDX), f"{d}.chp"))
            locs, ndet = fault_table(ops)
            region_of = regions_for(ops)
            used_all = np.ones(ndet, bool)
            used_term = np.zeros(ndet, bool); used_term[-4:] = True
            used_u2 = np.zeros(ndet, bool); used_u2[-4:] = True
            for j in range(1, m // 2 + 1):           # unif 2: rounds 2j - 1
                used_u2[4 * (2 * j - 1): 4 * (2 * j - 1) + 4] = True
            for pol, used in (("unif1", used_all), ("unif2", used_u2), ("const1", used_term)):
                out = analyse(locs, used, region_of, REGIONS)
                rows.append(dict(seq=s, policy=pol, m=m, **out))
            ops0 = parse(os.path.join(RUN, "experiments", "LRB_const0", str(s), str(PIDX), f"{d}.chp"))
            locs0, ndet0 = fault_table(ops0)
            out0 = analyse(locs0, np.ones(ndet0, bool), lambda L: "all", ("all",))
            rows.append(dict(seq=s, policy="const0", m=m, **out0))
        print(f"seq {s} done ({time.time() - t0:.0f} s)", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", out_path)


def main():
    s0, s1 = int(sys.argv[1]), int(sys.argv[2])
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    run_range(s0, s1, os.path.join(out_dir, f"faults_seq{s0}-{s1 - 1}.csv"))


if __name__ == "__main__":
    main()
