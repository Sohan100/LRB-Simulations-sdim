#!/usr/bin/env python3
"""Critic check for A1-syndromes, circuit level (claims C2, C13, C14, C16,
C17, C18, C20, C23).

Own parser and own batched Pauli-frame propagator (sdim frame rules, which
critic_A1-syndromes_code.py cross-checks against explicit unitaries).
The propagator is validated against the repository CompactDetectorErrorModel
(sdim) on circuits not used by A1 (depth 10 LRB, a depth-6 const0 circuit,
a memory circuit), and the repository DEM is used directly for the C2
convention check on modified depth-0 circuits.

Run (login node, a few minutes, 2 threads):
  module load python/3.11
  OMP_NUM_THREADS=2 python critic_A1-syndromes_circuit.py [--part all]
"""
import argparse
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
SCR = os.environ.get("CRITIC_SCRATCH", "/tmp")
NQ = 9
OUT = []


def say(*a):
    s = " ".join(str(t) for t in a)
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ parsing
def read_ops(path):
    ops = []
    started = False
    for line in open(path):
        t = line.strip()
        if not t:
            continue
        if not started:
            if t == "#":
                started = True
            continue
        if t.split()[0] == "d":
            continue
        name = t.split()[0]
        kv = dict(re.findall(r'(\w+)="([^"]*)"', t))
        rest = re.sub(r'\w+="[^"]*"', "", t).split()[1:]
        qs = tuple(int(x) for x in rest)
        ops.append((name, qs, kv, t))
    return ops


# ------------------------------------------------------------------ frames
def conj(Fx, Fz, g, a, b):
    if g in ("I", "X", "X_INV", "Z", "Z_INV"):
        return
    if g == "H":
        Fx[a], Fz[a] = -Fz[a], Fx[a].copy()
    elif g == "H_INV":
        Fx[a], Fz[a] = Fz[a].copy(), -Fx[a]
    elif g == "P":
        Fz[a] += Fx[a]
    elif g == "P_INV":
        Fz[a] -= Fx[a]
    elif g == "CNOT":
        Fx[b] += Fx[a]
        Fz[a] -= Fz[b]
    elif g == "CNOT_INV":
        Fx[b] -= Fx[a]
        Fz[a] += Fz[b]
    elif g == "CZ":
        Fz[b] += Fx[a]
        Fz[a] += Fx[b]
    elif g == "CZ_INV":
        Fz[b] -= Fx[a]
        Fz[a] -= Fx[b]
    elif g == "SWAP":
        Fx[[a, b]] = Fx[[b, a]]
        Fz[[a, b]] = Fz[[b, a]]
    else:
        raise ValueError(g)


GATES = {"I", "X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV", "CNOT",
         "CNOT_INV", "CZ", "CZ_INV", "SWAP"}


def propagate(ops, snapshot=None):
    """Unit columns: N1 -> (X, Z) on its qudit; N2 -> (Xa, Za, Xb, Zb)."""
    locs = []
    col = 0
    for i, (nm, qs, kv, _) in enumerate(ops):
        if nm in ("N1", "N2"):
            w = 2 if nm == "N1" else 4
            locs.append(dict(i=i, name=nm, qs=qs, ch=kv.get("noise_channel", "d"),
                             scale=float(kv.get("prob_scale", "1")), c0=col, w=w))
            col += w
    nc = col
    Fx = np.zeros((NQ, nc), dtype=np.int64)
    Fz = np.zeros((NQ, nc), dtype=np.int64)
    rec = []
    det, obs, dlab = [], [], []
    snap = None
    li = 0
    for i, (nm, qs, kv, _) in enumerate(ops):
        if snapshot is not None and i == snapshot:
            snap = (Fx % 3).copy(), (Fz % 3).copy()
        if nm in ("N1", "N2"):
            L = locs[li]
            li += 1
            Fx[qs[0], L["c0"]] += 1
            Fz[qs[0], L["c0"] + 1] += 1
            if nm == "N2":
                Fx[qs[1], L["c0"] + 2] += 1
                Fz[qs[1], L["c0"] + 3] += 1
        elif nm in GATES:
            conj(Fx, Fz, nm, qs[0], qs[1] if len(qs) > 1 else None)
            Fx %= 3
            Fz %= 3
        elif nm == "M":
            rec.append(Fx[qs[0]] % 3)
        elif nm == "RESET":
            Fx[qs[0]] = 0
            Fz[qs[0]] = 0
        elif nm in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            row = np.zeros(nc, dtype=np.int64)
            for sgn, idx in re.findall(r'([+-]?)rec\[(\d+)\]', kv["expr"]):
                row += (-1 if sgn == "-" else 1) * rec[int(idx)]
            (det if nm == "DETECTOR" else obs).append(row % 3)
            if nm == "DETECTOR":
                dlab.append(kv.get("label"))
        else:
            raise ValueError(nm)
    if snap is None:
        snap = (Fx % 3, Fz % 3)
    return dict(locs=locs, nc=nc, det=np.array(det).reshape(len(det), nc),
                obs=np.array(obs).reshape(len(obs), nc), snap=snap, rec=rec,
                dlab=dlab)


# syndrome/class functionals on (x0..x4, z0..z4)
FUN = np.array([[1, 2, 0, 2, 0, 0, 0, 0, 0, 0],      # sigma_Z1 = x0-x1-x3
                [0, 1, 1, 0, 2, 0, 0, 0, 0, 0],      # sigma_Z2 = x1+x2-x4
                [0, 0, 0, 0, 0, 2, 2, 1, 0, 0],      # sigma_X1 = -z0-z1+z2
                [0, 0, 0, 0, 0, 0, 1, 0, 2, 1],      # sigma_X2 = z1-z3+z4
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],      # a = x0+x2
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])     # b = z0+z3


def comps(name, ch):
    if name == "N1":
        if ch == "d":
            v = [(x, z) for x in range(3) for z in range(3) if (x, z) != (0, 0)]
        elif ch == "f":
            v = [(1, 0), (2, 0)]
        else:
            raise ValueError(ch)
    else:
        v = [(a, b, c, d) for a in range(3) for b in range(3) for c in range(3)
             for d in range(3) if (a, b, c, d) != (0, 0, 0, 0)]
    return np.array(v, dtype=np.int64)


def faults(prop, with_v2=True):
    """Expand into single faults: response rows [det..., obs, (perfect syn x4,
    a, b)], weight (coefficient of p), location index."""
    blocks = [prop["det"], prop["obs"]]
    if with_v2:
        sx, sz = prop["snap"]
        data = np.vstack([sx[:5], sz[:5]])
        blocks.append((FUN @ data) % 3)
    U = np.vstack(blocks)
    R, Wt, Lc, Rl = [], [], [], []
    for k, L in enumerate(prop["locs"]):
        if L["scale"] == 0:
            continue
        v = comps(L["name"], L["ch"])
        R.append((v @ U[:, L["c0"]:L["c0"] + L["w"]].T) % 3)
        Wt.append(np.full(len(v), L["scale"] / len(v)))
        Lc.append(np.full(len(v), k))
    return np.vstack(R), np.concatenate(Wt), np.concatenate(Lc)


def keyrows(A):
    A = np.ascontiguousarray(A.astype(np.int8))
    return A.view(np.dtype((np.void, A.shape[1]))).ravel()


def lead(resp, w, nsig, ccols):
    """Leading order: detection rejection, undetected failure, MLD failure,
    strict-hybrid rejection (signature with >= 2 classes)."""
    cls = np.zeros(len(resp), dtype=np.int64)
    for k, c in enumerate(ccols):
        cls += resp[:, c] * 3 ** k
    if nsig:
        _, gi = np.unique(keyrows(resp[:, :nsig]), return_inverse=True)
        gi = gi.ravel()
        zero = ~np.any(resp[:, :nsig] != 0, axis=1)
    else:
        gi = np.zeros(len(resp), dtype=np.int64)
        zero = np.ones(len(resp), dtype=bool)
    ng = gi.max() + 1
    Wg = np.zeros((ng, 3 ** len(ccols)))
    np.add.at(Wg, (gi, cls), w)
    gzero = np.zeros(ng, dtype=bool)
    gzero[gi[zero]] = True
    tot = Wg.sum(1)
    mx = Wg.max(1)
    ncl = (Wg > 0).sum(1)
    undet = Wg[gzero][:, 1:].sum()
    amb = (ncl > 1) & ~gzero
    return dict(det=tot[~gzero].sum(), undet=undet,
                mld=(tot - mx)[~gzero].sum() + undet, hyb=tot[amb].sum(),
                amb_fault=amb[gi], gi=gi, cls=cls, Wg=Wg, gzero=gzero)


def terminal_index(ops):
    """index of the first terminal op (after the last ancilla measurement
    for ordinary circuits; first H_INV of the const0 readout otherwise)."""
    last_anc_m = max([i for i, o in enumerate(ops) if o[0] == "M" and o[1][0] >= 5],
                     default=None)
    if last_anc_m is not None:
        return last_anc_m + 1
    # const0: the 5 H_INV before the 5 M
    i = next(k for k, o in enumerate(ops) if o[0] == "M")
    return i - 5


def analyse(path_or_ops):
    ops = read_ops(path_or_ops) if isinstance(path_or_ops, str) else path_or_ops
    prop = propagate(ops, snapshot=terminal_index(ops))
    R, w, L = faults(prop)
    nd = prop["det"].shape[0]
    v1 = lead(R, w, nd, [nd])
    R2 = np.hstack([R[:, :nd], R[:, nd + 1:nd + 5], R[:, nd + 5:nd + 7]])
    v2 = lead(R2, w, nd + 4, [nd + 4, nd + 5])
    return ops, prop, R, w, L, v1, v2, nd


# ------------------------------------------------------------------ roles
def roles(ops, prop):
    out = []
    for L in prop["locs"]:
        i = L["i"]
        nm, qs = L["name"], L["qs"]
        prev = ops[i - 1][0]
        if nm == "N2":
            r = "check_CNOT" if max(qs) >= 5 else "gate_2q"
        elif L["ch"] == "f":
            r = "reset_flip" if prev == "RESET" else "meas_flip"
        elif L["scale"] == 0.1:
            r = "gate_1q" if qs[0] < 5 else "anc_H"
        elif L["scale"] == 1.0:
            r = "meas_depol"
        else:
            j = i
            while ops[j][0] == "N1":
                j += 1
            r = ("idle_reset_" if ops[j][0] == "RESET" else "idle_meas_") + \
                ("data" if qs[0] < 5 else "anc")
        out.append(r)
    return out


# ------------------------------------------------------------------ memory
def strip_logical(ops):
    """Remove logical-gate ops (data-only gates after the encoder, their
    0.1p N1 and data-data N2) -> memory circuit with identity Cliffords."""
    gate_idx = [i for i, o in enumerate(ops) if o[0] in GATES]
    enc = set(gate_idx[:7])
    term = terminal_index(ops)
    keep = []
    for i, o in enumerate(ops):
        nm, qs = o[0], o[1]
        if i in enc or i >= term:
            keep.append(o)
            continue
        if nm in GATES and max(qs) < 5:
            continue
        if nm == "N1" and float(o[2].get("prob_scale", 1)) == 0.1 and qs[0] < 5:
            continue
        if nm == "N2" and max(qs) < 5:
            continue
        keep.append(o)
    return keep


def write_ops(ops, path, header):
    with open(path, "w") as fh:
        fh.write(header)
        for o in ops:
            fh.write(o[3] + "\n")


def header_of(path):
    h = []
    for line in open(path):
        h.append(line)
        if line.strip().split() and line.strip().split()[0] == "d":
            break
    return "".join(h)


# ------------------------------------------------------------------ DEM
def dem_compare(path):
    sys.path.insert(0, os.path.join(WT, "src"))
    from sdim.circuit_io import read_circuit
    from lrb.dem_simulation import build_compact_detector_error_model
    t = time.time()
    model = build_compact_detector_error_model(read_circuit(path))
    ops = read_ops(path)
    prop = propagate(ops)
    by_i = {L["i"]: L for L in prop["locs"]}
    nbad = ncmp = 0
    for loc in model.locations:
        L = by_i[loc.operation_index]
        for k, terms in enumerate(loc.component_terms):
            d = np.zeros(prop["det"].shape[0], dtype=np.int64)
            for a, v in terms.detector:
                d[a] = v % 3
            o = np.zeros(prop["obs"].shape[0], dtype=np.int64)
            for a, v in terms.logical:
                o[a] = v % 3
            ncmp += 1
            if not (np.array_equal(d, prop["det"][:, L["c0"] + k])
                    and np.array_equal(o, prop["obs"][:, L["c0"] + k])):
                nbad += 1
    return dict(path=os.path.basename(os.path.dirname(os.path.dirname(path))) + "/"
                + os.path.basename(path), n_loc=len(model.locations),
                n_comp=ncmp, mismatches=nbad, seconds=round(time.time() - t, 1))


def dem_conventions():
    """C2 with the repository DEM: insert unit noise on each data qutrit right
    after the encoder of the depth-0 circuits, read the DEM responses."""
    sys.path.insert(0, os.path.join(WT, "src"))
    from sdim.circuit_io import read_circuit
    from lrb.dem_simulation import build_compact_detector_error_model
    res = {}
    for sub in ("LRB", "LRB_const0"):
        src = os.path.join(RUN, "experiments", sub, "0", "4", "0.chp")
        lines = open(src).read().splitlines()
        # encoder = first 7 gate lines after '#' and 'd 3'
        start = lines.index("#") + 2
        ins = start + 7
        new = lines[:ins] + [f'N1 {q} noise_channel="d" prob="0.001" prob_scale="1.0"'
                             for q in range(5)] + lines[ins:]
        dst = os.path.join(SCR, f"critic_conv_{sub}.chp")
        open(dst, "w").write("\n".join(new) + "\n")
        model = build_compact_detector_error_model(read_circuit(dst))
        rows = {}
        for loc in model.locations:
            if loc.operation_index in range(ins - start + 0, ins - start + 5):
                pass
        # operation indices in the sdim circuit: count ops after header
        for loc in model.locations:
            if loc.gate == "N1" and loc.operation_index < 12:
                q = loc.qudit
                for k, terms in enumerate(loc.component_terms):
                    rows[("X" if k == 0 else "Z") + str(q)] = (
                        {a: v % 3 for a, v in terms.detector},
                        {a: v % 3 for a, v in terms.logical})
        res[sub] = rows
    # compare with functionals
    ok_lrb = ok_c0 = True
    for key, (dd, ll) in res["LRB"].items():
        typ, q = key[0], int(key[1])
        e = np.zeros(10, dtype=int)
        e[q if typ == "X" else 5 + q] = 1
        s = (FUN @ e) % 3
        want = {j: int(s[j]) for j in range(4) if s[j]}
        ok_lrb &= (dd == want) and (ll.get(0, 0) == int(s[5]))
    for key, (dd, ll) in res["LRB_const0"].items():
        typ, q = key[0], int(key[1])
        e = np.zeros(10, dtype=int)
        e[q if typ == "X" else 5 + q] = 1
        s = (FUN @ e) % 3
        want = {j - 2: int((-s[j]) % 3) for j in (2, 3) if s[j]}
        ok_c0 &= (dd == want) and (ll.get(0, 0) == int(s[5]))
    return res, ok_lrb, ok_c0


# ------------------------------------------------------------------ order 2
def order2(R, w, L, locp, nsig, ccols, hybrid=True, chunk=4096):
    """Second-order coefficient of P(fail|acc) for detection and the strict
    hybrid (accept iff all single-fault explanations of the signature share
    one class; pair-only signatures accepted only if unanimous -> no failure).
    Signatures are hashed with two independent random 62-bit linear keys."""
    rng = np.random.default_rng(12345)
    sig = R[:, :nsig].astype(np.int64)
    cl = np.zeros(len(R), dtype=np.int64)
    for k, c in enumerate(ccols):
        cl = cl * 3 + R[:, c]
    ncl = 3 ** len(ccols)
    h1 = rng.integers(1, 2 ** 40, size=nsig)
    h2 = rng.integers(1, 2 ** 40, size=nsig)
    k1 = sig @ h1
    k2 = sig @ h2
    key = k1 * 1 + 0  # combined below
    # unique single-fault signatures
    pairkey = np.stack([k1, k2], axis=1)
    uk, inv = np.unique(pairkey, axis=0, return_inverse=True)
    inv = inv.ravel()
    Wg = np.zeros((len(uk), ncl))
    np.add.at(Wg, (inv, cl), w)
    zero_g = np.all(uk == 0, axis=1)
    acc = (Wg > 0).sum(1) == 1
    acc[zero_g] = True
    best = Wg.argmax(1)
    best[zero_g] = 0
    T = locp.sum()
    is0 = np.all(sig == 0, axis=1)
    undet = w[is0 & (cl != 0)].sum()
    det_rej = w[~is0].sum()
    hyb_rej = w[~acc[inv]].sum()
    f1 = acc[inv] & (cl != best[inv])           # = undetected failures
    dep = float(np.sum(w[f1] * (T - locp[L[f1]])))
    # lookup structure for combined signatures (mod 3 reduction needed)
    order = np.lexsort((uk[:, 1], uk[:, 0]))
    uks = uk[order]
    A2det = 0.0
    A2hyb = 0.0
    n = len(R)
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        j = j[L[j] != L[i]]
        if len(j) == 0:
            continue
        s2 = (sig[i] + sig[j]) % 3
        c2 = np.zeros(len(j), dtype=np.int64)
        # class digits add mod 3 digit-wise
        ci, cj = cl[i], cl[j]
        pw = 1
        for _ in ccols:
            c2 += (((ci // pw) % 3 + (cj // pw) % 3) % 3) * pw
            pw *= 3
        ww = w[i] * w[j]
        z = ~np.any(s2 != 0, axis=1)
        A2det += ww[z & (c2 != 0)].sum()
        if hybrid:
            a1 = s2 @ h1
            a2 = s2 @ h2
            pos = np.searchsorted(uks[:, 0], a1)
            pos = np.minimum(pos, len(uks) - 1)
            # resolve ties on the first key by a small linear probe
            hit = np.zeros(len(j), dtype=bool)
            gidx = np.zeros(len(j), dtype=np.int64)
            for off in range(4):
                p_ = np.minimum(pos + off, len(uks) - 1)
                m = (~hit) & (uks[p_, 0] == a1) & (uks[p_, 1] == a2)
                hit |= m
                gidx[m] = order[p_[m]]
            good = hit & ~z
            g2 = gidx[good]
            A2hyb += ww[good][acc[g2] & (c2[good] != best[g2])].sum() + \
                ww[z & (c2 != 0)].sum()
    return dict(det_rej1=det_rej, undet1=undet,
                det_fa2=A2det - dep + undet * det_rej,
                hyb_rej1=hyb_rej,
                hyb_fa2=(A2hyb - dep + undet * hyb_rej) if hybrid else None)


def det_order2_fast(R, w, L, locp, nsig, ccols):
    """Detection only: pairs with zero combined signature are pairs between
    the group of signature s and the group of -s (mod 3)."""
    sig = R[:, :nsig]
    cl = R[:, ccols]                          # class digits
    kb = keyrows(sig)
    kneg = keyrows((-sig) % 3)
    uniq, gi = np.unique(kb, return_inverse=True)
    gi = gi.ravel()
    pos = {u: k for k, u in enumerate(uniq)}
    members = defaultdict(list)
    for idx, g in enumerate(gi):
        members[g].append(idx)
    is0 = ~np.any(sig != 0, axis=1)
    undet = w[is0 & np.any(cl != 0, axis=1)].sum()
    det_rej = w[~is0].sum()
    T = locp.sum()
    f1 = is0 & np.any(cl != 0, axis=1)
    dep = float(np.sum(w[f1] * (T - locp[L[f1]])))
    A2 = 0.0
    done = set()
    for g, mem in members.items():
        h = pos.get(kneg[mem[0]])
        if h is None or (min(g, h), max(g, h)) in done:
            continue
        done.add((min(g, h), max(g, h)))
        a = np.array(mem)
        b = np.array(members[h])
        cs = (cl[a][:, None, :] + cl[b][None, :, :]) % 3
        bad = np.any(cs != 0, axis=2) & (L[a][:, None] != L[b][None, :])
        ww = w[a][:, None] * w[b][None, :]
        if g == h:                            # zero group: unordered pairs once
            bad = np.triu(bad, k=1)
        A2 += ww[bad].sum()
    return dict(det_rej1=det_rej, undet1=undet, det_fa2=A2 - dep + undet * det_rej)


def run_order2(path_or_ops, views=("V1",), hybrid=True):
    ops, prop, R, w, L, v1, v2, nd = analyse(path_or_ops)
    locp = np.array([Lc["scale"] for Lc in prop["locs"]])
    out = {}
    if not hybrid:
        if "V1" in views:
            out["V1"] = det_order2_fast(R, w, L, locp, nd, [nd])
        if "V2" in views:
            R2 = np.hstack([R[:, :nd], R[:, nd + 1:nd + 5], R[:, nd + 5:nd + 7]])
            out["V2"] = det_order2_fast(R2, w, L, locp, nd + 4, [nd + 4, nd + 5])
        return out
    if "V1" in views:
        out["V1"] = order2(R, w, L, locp, nd, [nd], hybrid=hybrid)
    if "V2" in views:
        R2 = np.hstack([R[:, :nd], R[:, nd + 1:nd + 5], R[:, nd + 5:nd + 7]])
        out["V2"] = order2(R2, w, L, locp, nd + 4, [nd + 4, nd + 5], hybrid=hybrid)
    return out


def csv_rows(p_idx, policy, key):
    rows = {}
    for line in open(os.path.join(RUN, "results", "LRB", str(p_idx), policy,
                                  f"{key}.csv")):
        parts = line.strip().split(",")
        rows[parts[0]] = [float(x) for x in parts[1:]]
    return rows


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    args = ap.parse_args()
    part = args.part
    depths = [int(x) for x in open(os.path.join(RUN, "depths.txt")).read().split(",")]
    t0 = time.time()

    if part in ("all", "conv"):
        say("=" * 70)
        say("C2 with the repository DEM (unit noise after the encoder, depth 0)")
        res, ok1, ok2 = dem_conventions()
        for sub, rows in res.items():
            for k in sorted(rows):
                say(f"  {sub:10s} {k}: detectors {rows[k][0]} logical {rows[k][1]}")
        say("  ancilla detectors == <S,E>, observable == z0+z3:", ok1)
        say("  const0 detectors == -sigma_X, observable == z0+z3:", ok2)

    if part in ("all", "dem"):
        say("=" * 70)
        say("Own propagator vs repository DEM (circuits not used by A1)")
        mem = strip_logical(read_ops(os.path.join(RUN, "experiments", "LRB", "5", "4",
                                                  "2.chp")))
        mp = os.path.join(SCR, "critic_memory_R4.chp")
        write_ops(mem, mp, header_of(os.path.join(RUN, "experiments", "LRB", "5", "4",
                                                  "2.chp")))
        for p in (os.path.join(RUN, "experiments", "LRB", "7", "4", "4.chp"),
                  os.path.join(RUN, "experiments", "LRB_const0", "11", "4", "3.chp"),
                  mp):
            say("  ", dem_compare(p))

    if part in ("all", "memory"):
        say("=" * 70)
        say("C16: memory windows (identity Cliffords) from stripped baseline files")
        base = os.path.join(RUN, "experiments", "LRB", "0", "4")
        rows = []
        for di in range(0, 4):   # depths 0,2,4,6 -> R = m bulk rounds
            ops = strip_logical(read_ops(os.path.join(base, f"{di}.chp")))
            ops_, prop, R, w, L, v1, v2, nd = analyse(ops)
            R_ = depths[di]
            rows.append((R_, v1, v2, ops_, prop, R, w, L, nd))
            say(f"  R={R_}: V1 det {v1['det']:.3f} undet {v1['undet']:.3f} MLD "
                f"{v1['mld']:.3f} hyb {v1['hyb']:.3f} | V2 det {v2['det']:.3f} "
                f"MLD {v2['mld']:.3f} hyb {v2['hyb']:.3f}")
        for k in ("det", "mld", "hyb"):
            inc1 = [(b[1][k] - a[1][k]) / (b[0] - a[0]) for a, b in zip(rows, rows[1:])]
            inc2 = [(b[2][k] - a[2][k]) / (b[0] - a[0]) for a, b in zip(rows, rows[1:])]
            say(f"  per-round increment {k}: V1 {np.round(inc1, 4).tolist()}  "
                f"V2 {np.round(inc2, 4).tolist()}")
        # V2 ambiguous breakdown: R=4 minus R=2, per bulk round, by role/qutrit/type
        def amb_breakdown(row):
            R_, v1, v2, ops_, prop, Rr, w, L, nd = row
            rl = roles(ops_, prop)
            comps_all = []
            for Lc in prop["locs"]:
                comps_all.append(comps(Lc["name"], Lc["ch"]))
            bd = defaultdict(float)
            kidx = 0
            amb = v2["amb_fault"]
            # rebuild per-fault (loc, comp) in the order used by faults()
            pos = 0
            for li, Lc in enumerate(prop["locs"]):
                v = comps_all[li]
                for cvec in v:
                    if amb[pos]:
                        r = rl[li]
                        if r.startswith("idle") and Lc["qs"][0] < 5:
                            q = Lc["qs"][0]
                            x, z = int(cvec[0]), int(cvec[1])
                            t = "mixed" if (x and z) else ("pureX" if x else "pureZ")
                            bd[(r, f"q{q}", t)] += w[pos]
                        else:
                            bd[(r,)] += w[pos]
                    pos += 1
            return bd
        b4 = amb_breakdown(rows[3])
        b2 = amb_breakdown(rows[2])
        say("  V2 ambiguous weight per bulk round, (R=6 - R=4)/2, by role/qutrit/type:")
        keys = sorted(set(b4) | set(b2))
        for k in keys:
            d = (b4.get(k, 0) - b2.get(k, 0)) / 2
            if abs(d) > 1e-9:
                say(f"     {k}: {d:.4f}")
        # same for V1
        say("  V1 per-round MLD/hyb increments were computed above.")

    if part in ("all", "lrb"):
        say("=" * 70)
        say("C13/C17/C23: all 30 Clifford sequences x 9 depths, LRB and const0")
        t1 = time.time()
        tab = defaultdict(lambda: defaultdict(list))
        for c in range(30):
            for di in range(9):
                p = os.path.join(RUN, "experiments", "LRB", str(c), "4", f"{di}.chp")
                ops, prop, R, w, L, v1, v2, nd = analyse(p)
                for k in ("det", "undet", "mld", "hyb"):
                    tab[("V1", k)][di].append(v1[k])
                    tab[("V2", k)][di].append(v2[k])
                p0 = os.path.join(RUN, "experiments", "LRB_const0", str(c), "4",
                                  f"{di}.chp")
                ops0 = read_ops(p0)
                prop0 = propagate(ops0)
                if prop0["nc"] == 0:
                    for k in ("det", "undet", "mld", "hyb"):
                        tab[("C0", k)][di].append(0.0)
                else:
                    R0, w0, L0 = faults(prop0, with_v2=False)
                    nd0 = prop0["det"].shape[0]
                    v0 = lead(R0, w0, nd0, [nd0])
                    for k in ("det", "undet", "mld", "hyb"):
                        tab[("C0", k)][di].append(v0[k])
        say(f"  analysed 540 circuits in {time.time() - t1:.1f} s")
        b1 = csv_rows(1, "unif_check_data", 1)
        b0 = csv_rows(1, "const_check_data", 0)
        p = 1e-4
        ms = np.array(depths)
        hdr = ("m", "det(30)", "base", "det(0-9)", "undet min/max", "MLD(30)",
               "MLD(0-9)", "hyb(30)", "hyb(0-9)", "C0det(30)", "C0base", "C0det(0-9)")
        say("  " + " | ".join(hdr))
        for di in range(9):
            a = lambda key, sl=slice(None): float(np.mean(tab[key][di][sl]))
            say(f"  {depths[di]:2d} | {a(('V1','det')):8.2f} | "
                f"{-np.log(1 - b1['Rejected Runs'][di]) / p:8.2f} | "
                f"{a(('V1','det'), slice(0,10)):8.2f} | "
                f"{min(tab[('V1','undet')][di]):.4f}/{max(tab[('V1','undet')][di]):.4f} | "
                f"{a(('V1','mld')):7.2f} | {a(('V1','mld'), slice(0,10)):7.2f} | "
                f"{a(('V1','hyb')):7.2f} | {a(('V1','hyb'), slice(0,10)):7.2f} | "
                f"{a(('C0','det')):7.2f} | "
                f"{-np.log(1 - b0['Rejected Runs'][di]) / p:7.2f} | "
                f"{a(('C0','det'), slice(0,10)):7.2f}")
        sel = ms >= 2
        A = np.vstack([np.ones(sel.sum()), ms[sel]]).T
        def fit(vals):
            co, *_ = np.linalg.lstsq(A, np.array(vals)[sel], rcond=None)
            return co
        for key, nm in ((("V1", "det"), "V1 det"), (("V1", "mld"), "V1 MLD"),
                        (("V1", "hyb"), "V1 hyb"), (("V2", "mld"), "V2 MLD"),
                        (("V2", "hyb"), "V2 hyb"), (("C0", "det"), "C0 det"),
                        (("C0", "mld"), "C0 MLD"), (("C0", "hyb"), "C0 hyb")):
            for sl, lab in ((slice(None), "30 seq"), (slice(0, 10), "seq 0-9")):
                co = fit([np.mean(tab[key][di][sl]) for di in range(9)])
                say(f"  fit {nm:7s} {lab:7s}: c0 {co[0]:8.3f} c1 {co[1]:8.3f}")
        co_b = fit([-np.log(1 - b1['Rejected Runs'][di]) / p for di in range(9)])
        co_b0 = fit([-np.log(1 - b0['Rejected Runs'][di]) / p for di in range(9)])
        say(f"  baseline unif1 -ln(acc)/p fit: c0 {co_b[0]:.3f} c1 {co_b[1]:.3f}")
        say(f"  baseline const0 -ln(acc)/p fit: c0 {co_b0[0]:.3f} c1 {co_b0[1]:.3f}")
        # per-sequence spread of slopes
        sl_seq = []
        for c in range(30):
            co, *_ = np.linalg.lstsq(A, np.array([tab[("V1", "mld")][di][c]
                                                  for di in range(9)])[sel], rcond=None)
            co2, *_ = np.linalg.lstsq(A, np.array([tab[("V2", "mld")][di][c]
                                                   for di in range(9)])[sel], rcond=None)
            sl_seq.append((co[1], co2[1]))
        sl_seq = np.array(sl_seq)
        say(f"  per-sequence MLD slopes V1 mean {sl_seq[:, 0].mean():.3f} sd "
            f"{sl_seq[:, 0].std():.3f}; V2 mean {sl_seq[:, 1].mean():.3f} sd "
            f"{sl_seq[:, 1].std():.3f}; ratio of means "
            f"{sl_seq[:, 0].mean() / sl_seq[:, 1].mean():.4f}")
        # baseline (1-F)/(1.5p) at p = 1e-4
        say("  baseline unif1 (1-F)/(1.5p) at p=1e-4 per depth:",
            [round((1 - f) / 1.5 / p, 3) for f in b1["Fidelity averages"]])

    if part in ("all", "order2"):
        say("=" * 70)
        say("C18: second-order coefficients (own pair enumeration)")
        t1 = time.time()
        res = {}
        # depth 0 (identical for all sequences), V1 and V2
        p0 = os.path.join(RUN, "experiments", "LRB", "0", "4", "0.chp")
        res["m0"] = run_order2(p0, views=("V1", "V2"))
        say("  m=0:", {v: {k: (round(x, 3) if x is not None else None)
                           for k, x in d.items()} for v, d in res["m0"].items()})
        # memory R=1, R=2 V2 (bulk-round increments)
        for di in (1,):
            ops = strip_logical(read_ops(os.path.join(RUN, "experiments", "LRB", "0",
                                                      "4", f"{di}.chp")))
            res[f"mem{depths[di]}"] = run_order2(ops, views=("V1", "V2"))
            say(f"  memory R={depths[di]}:",
                {v: {k: (round(x, 3) if x is not None else None) for k, x in d.items()}
                 for v, d in res[f"mem{depths[di]}"].items()})
        say(f"  (elapsed {time.time() - t1:.1f} s)")
        # m = 2 for sequences 0..5 (V1, detection + hybrid)
        c2 = []
        for c in range(6):
            p2 = os.path.join(RUN, "experiments", "LRB", str(c), "4", "1.chp")
            r2 = run_order2(p2, views=("V1",))["V1"]
            c2.append(r2)
            say(f"  m=2 seq {c}: det f|a {r2['undet1']:.3f} p + {r2['det_fa2']:.2f} p^2;"
                f" hyb f|a {r2['undet1']:.3f} p + {r2['hyb_fa2']:.1f} p^2;"
                f" hyb rej {r2['hyb_rej1']:.2f} p  ({time.time() - t1:.0f} s)")
        # detection only, all 30 sequences at m = 2 (pairs, detection part)
        det2 = [c2[c]["det_fa2"] for c in range(6)]
        for c in range(6, 30):
            p2 = os.path.join(RUN, "experiments", "LRB", str(c), "4", "1.chp")
            det2.append(run_order2(p2, views=("V1",), hybrid=False)["V1"]["det_fa2"])
        say(f"  m=2 detection c2 over 30 sequences: mean {np.mean(det2):.2f}, "
            f"min {np.min(det2):.2f}, max {np.max(det2):.2f}")
        say(f"  (elapsed {time.time() - t1:.1f} s)")
        # baseline comparison of unif-1 fidelities
        d0 = res["m0"]["V1"]
        for p_idx, p in ((3, 5e-4), (4, 1e-3), (5, 2e-3)):
            b = csv_rows(p_idx, "unif_check_data", 1)
            F0 = b["Fidelity averages"][0]
            F2 = b["Fidelity averages"][1]
            pr0 = 1 - 1.5 * (d0["undet1"] * p + d0["det_fa2"] * p * p)
            pr2 = 1 - 1.5 * (4.95 * p + np.mean(det2) * p * p)
            pr2_A1 = [1 - 1.5 * (4.95 * p + c_ * p * p) for c_ in (97.239, 119.834)]
            say(f"  p={p}: m=0 pred {pr0:.6f} base {F0:.6f} diff {pr0 - F0:+.2e} | "
                f"m=2 pred(30-seq c2) {pr2:.6f} base {F2:.6f} diff {pr2 - F2:+.2e} | "
                f"A1 seq0/seq1 c2 preds {pr2_A1[0]:.6f}/{pr2_A1[1]:.6f}")
        # hybrid at m=4 for one sequence: slope check
        p4 = os.path.join(RUN, "experiments", "LRB", "0", "4", "2.chp")
        r4 = run_order2(p4, views=("V1",))["V1"]
        say(f"  m=4 seq 0: det c2 {r4['det_fa2']:.2f}, hyb c2 {r4['hyb_fa2']:.1f},"
            f" hyb rej {r4['hyb_rej1']:.2f}  ({time.time() - t1:.0f} s)")

    if part in ("all", "cycle"):
        say("=" * 70)
        say("C14: one bulk check round + reset, taken from a stripped baseline file")
        import itertools as it
        ops = strip_logical(read_ops(os.path.join(RUN, "experiments", "LRB", "0", "4",
                                                  "1.chp")))
        # segment = ops strictly after the encoder up to the end of the first
        # reset layer (incl. its flips)
        gate_idx = [i for i, o in enumerate(ops) if o[0] in GATES]
        start = gate_idx[6] + 1
        resets = [i for i, o in enumerate(ops) if o[0] == "RESET"]
        end = resets[3] + 2                    # last RESET and its 'f' noise
        seg = [o for o in ops[start:end] if o[0] not in ("DETECTOR",
                                                          "LOGICAL_OBSERVABLE")]
        prop = propagate(seg)
        sx, sz = prop["snap"]
        rec = np.array(prop["rec"])            # 4 x nc  (this-round flips)
        nxt = sx[5:9]                          # ancilla X after reset -> next flips
        data = np.vstack([sx[:5], sz[:5]])
        # min-weight table over all data Paulis
        allv = np.array(list(it.product(range(3), repeat=10)))
        wt = np.count_nonzero((allv[:, :5] != 0) | (allv[:, 5:] != 0), axis=1)
        S = allv @ FUN.T % 3
        si = S[:, 0] + 3 * S[:, 1] + 9 * S[:, 2] + 27 * S[:, 3]
        ci = S[:, 4] + 3 * S[:, 5]
        minw = np.full((81, 9), 99)
        np.minimum.at(minw, (si, ci), wt)
        nw1 = np.zeros(81, dtype=int)          # number of weight-1 errors per syndrome
        np.add.at(nw1, si[wt == 1], 1)
        w1cos = defaultdict(set)
        for k in np.nonzero(wt == 1)[0]:
            w1cos[int(si[k])].add(int(ci[k]))
        tot = defaultdict(float)
        flips_measonly = np.zeros(4)
        total_w = 0.0
        for L in prop["locs"]:
            v = comps(L["name"], L["ch"])
            cols = slice(L["c0"], L["c0"] + L["w"])
            res = (v @ data[:, cols].T) % 3
            fl = (v @ rec[:, cols].T) % 3
            nx = (v @ nxt[:, cols].T) % 3
            sres = res @ FUN.T % 3
            for k in range(len(v)):
                wgt = L["scale"] / len(v)
                total_w += wgt
                s_ = int(sres[k, 0] + 3 * sres[k, 1] + 9 * sres[k, 2] + 27 * sres[k, 3])
                c_ = int(sres[k, 4] + 3 * sres[k, 5])
                if s_ == 0 and c_ == 0:
                    if np.any(fl[k]) or np.any(nx[k]):
                        tot["meas_only"] += wgt
                        flips_measonly += (fl[k] != 0) * wgt
                    else:
                        tot["no_effect"] += wgt
                    continue
                wmin = minw[s_, c_]
                if wmin == 1:
                    amb = len(w1cos[s_]) > 1
                    tot["w1_ambiguous" if amb else "w1_correctable"] += wgt
                else:
                    tot[f"w{wmin}"] += wgt
        say(f"  total fault weight {total_w:.4f} p; categories:",
            {k: round(v_, 4) for k, v_ in sorted(tot.items())})
        say("  measurement-only flips per check:", np.round(flips_measonly, 4).tolist())

    if part in ("all", "lopsided"):
        say("=" * 70)
        say("Circuit-level posterior structure of 'ambiguous' signatures")

        def post_bins(v, tag):
            Wg = v["Wg"]
            tot = Wg.sum(1)
            ncl = (Wg > 0).sum(1)
            amb = (ncl > 1) & ~v["gzero"]
            post = np.where(tot > 0, Wg.max(1) / np.where(tot > 0, tot, 1), 0)
            bins = [(0.95, 1.01), (0.8, 0.95), (0.6, 0.8), (0.0, 0.6)]
            parts = []
            for lo, hi in bins:
                m = amb & (post >= lo) & (post < hi)
                parts.append(f"post in [{lo},{min(hi, 1)}): rej {tot[m].sum():.3f}, "
                             f"minority {(tot - Wg.max(1))[m].sum():.3f}")
            return f"{tag}: " + " | ".join(parts)

        base = os.path.join(RUN, "experiments", "LRB", "0", "4")
        vv = {}
        for di in (2, 3):
            ops = strip_logical(read_ops(os.path.join(base, f"{di}.chp")))
            _, _, _, _, _, v1, v2, _ = analyse(ops)
            vv[di] = (v1, v2)
        # per-bulk-round differences of the binned totals (R=6 minus R=4)/2
        for view, k in (("V1", 0), ("V2", 1)):
            a = post_bins(vv[3][k], f"memory R=6 {view}")
            b = post_bins(vv[2][k], f"memory R=4 {view}")
            say("  " + a)
            say("  " + b)
        for c in (0, 1, 2):
            for di in (0, 8):
                p = os.path.join(RUN, "experiments", "LRB", str(c), "4", f"{di}.chp")
                _, _, _, _, _, v1, v2, _ = analyse(p)
                say("  " + post_bins(v1, f"LRB seq {c} m={depths[di]} V1"))

    say(f"total {time.time() - t0:.1f} s")
    with open(os.path.join(HERE, f"critic_A1-syndromes_circuit_output_{part}.txt"),
              "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
