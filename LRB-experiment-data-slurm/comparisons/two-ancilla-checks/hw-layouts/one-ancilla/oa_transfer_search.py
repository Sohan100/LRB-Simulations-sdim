"""One ancilla per stabilizer on a square lattice: data frames + ancilla-ancilla transfers.

Extends oa_frame_search.py. Besides reading lattice-neighbour data in data frames
(windows of data-data CNOTs, pushed and later popped), the two ancillas of one type
(Z: a1 reads G1, a2 reads G2; X: a3 reads G3, a4 reads G4) may be coupled and
transfer partial sums: a_Q += c * a_P (Z type: CNOT^c(P -> Q); X type, ancillas in
the Fourier basis: CNOT^-c(Q -> P)). A transfer touches no data qutrit.

Per type ("sector") and timeline, a breadth-first search over (segment, acc_P,
acc_Q) gives the minimum number of reads + transfers for every pair of touched sets
(T_P, T_Q) and coupling flag. Segments: I, W1 prefixes (push), ..., I, W2 ..., I;
reads in a segment use its frame, transfers may happen in any segment.

Combination: total = reads + transfers + 2 * (data gates). Necessary geometric
filters (square lattice): bipartite colouring (data couplers bichromatic, touched
sets monochromatic, coupled ancillas of opposite colour); two ancillas share at most
2 data neighbours, and two that share 2 cannot be coupled; ancilla degree <= 4; data
degree <= 4. Survivors are embedded exactly on Z^2.

Usage: oa_transfer_search.py 'SHAPES' MAXGATES [--no-transfer] [--safe]
"""
import itertools
import sys
from collections import defaultdict
from multiprocessing import Pool

import numba
import numpy as np

import oa_frame_search as FS

D = 3
NV = 243
NN = NV * NV
POW = np.array([3 ** i for i in range(5)])


def enc(v):
    return int(sum((int(x) % D) * 3 ** i for i, x in enumerate(v)))


def dec(n):
    return tuple((n // 3 ** i) % 3 for i in range(5))


ADD = np.array([[enc([(a + b) % D for a, b in zip(dec(x), dec(y))]) for y in range(NV)] for x in range(NV)],
               dtype=np.int32)
MUL = np.array([[enc([(c * a) % D for a in dec(x)]) for x in range(NV)] for c in range(3)], dtype=np.int32)
SUBSETS = [frozenset(s) for k in range(1, 5) for s in itertools.combinations(range(5), k)]


@numba.njit(cache=True)
def sector_bfs(nseg, pmoves, pseg, qmoves, qseg, transfer, target, ADD, MUL, maxc):
    """pmoves[i]: vector (int) added to acc_P in segment pseg[i]; same for Q.
    Returns the min cost to reach state `target` in the last segment (99 if > maxc)."""
    INF = 99
    dist = np.full((nseg, NN), INF, dtype=np.int8)
    dist[0, 0] = 0
    for k in range(nseg):
        if k > 0:
            for s in range(NN):
                if dist[k - 1, s] < dist[k, s]:
                    dist[k, s] = dist[k - 1, s]
        for d in range(maxc):
            any_ = False
            for s in range(NN):
                if dist[k, s] != d:
                    continue
                any_ = True
                p = s // NV
                q = s % NV
                for i in range(pmoves.shape[0]):
                    if pseg[i] == k:
                        t = ADD[p, pmoves[i]] * NV + q
                        if dist[k, t] > d + 1:
                            dist[k, t] = d + 1
                for i in range(qmoves.shape[0]):
                    if qseg[i] == k:
                        t = p * NV + ADD[q, qmoves[i]]
                        if dist[k, t] > d + 1:
                            dist[k, t] = d + 1
                if transfer:
                    for c in range(1, 3):
                        t = p * NV + ADD[q, MUL[c, p]]
                        if dist[k, t] > d + 1:
                            dist[k, t] = d + 1
                        t = ADD[p, MUL[c, q]] * NV + q
                        if dist[k, t] > d + 1:
                            dist[k, t] = d + 1
            if not any_ and d > 0:
                pass
    return dist[nseg - 1, target]


def segments(timeline):
    """Frames of the segments in time order: I, W1[:1], ..., W1, ..., W1[:1], I, W2..., I."""
    segs = [()]
    for w in timeline:
        L = len(w)
        segs += [w[:i] for i in range(1, L + 1)] + [w[:i] for i in range(L - 1, 0, -1)] + [()]
    return segs


def read_moves(kind, segs, T):
    vec, seg = [], []
    for k, fr in enumerate(segs):
        for q in sorted(T):
            e = tuple(int(i == q) for i in range(5))
            r = FS.pull_back(kind, e, fr)
            for c in (1, 2):
                vec.append(enc([(c * x) % D for x in r]))
                seg.append(k)
    return np.array(vec, dtype=np.int32), np.array(seg, dtype=np.int32)


def sector_table(kind, tP, tQ, segs, maxc, allow_transfer=True):
    """(T_P, T_Q, coupled) -> min cost, for pairs passing the cheap geometric filters."""
    target = enc(tP) * NV + enc(tQ)
    out = {}
    cache = {T: read_moves(kind, segs, T) for T in SUBSETS}
    for TP in SUBSETS:
        for TQ in SUBSETS:
            if len(TP & TQ) > 2:
                continue
            for coupled in ((False, True) if allow_transfer else (False,)):
                if coupled and (len(TP & TQ) >= 1 or len(TP) > 3 or len(TQ) > 3):
                    # coupled ancillas are lattice neighbours: opposite colours, so no common data
                    continue
                pm, ps = cache[TP]
                qm, qs = cache[TQ]
                c = sector_bfs(len(segs), pm, ps, qm, qs, coupled, target, ADD, MUL, maxc)
                if c < 99:
                    out[(TP, TQ, coupled)] = int(c)
    return out


def colour_ok(edges, groups, opposite_pairs):
    """Union-find with parity: edges and opposite_pairs (between groups) bichromatic."""
    parent = list(range(5))
    par = [0] * 5

    def find(x):
        if parent[x] == x:
            return x, 0
        r, p = find(parent[x])
        parent[x] = r
        par[x] = (par[x] + p) % 2
        return r, par[x]

    def union(a, b, d):
        (ra, pa), (rb, pb) = find(a), find(b)
        if ra == rb:
            return (pa + pb) % 2 == d
        parent[ra] = rb
        par[ra] = (pa + pb + d) % 2
        return True

    for a, b in edges:
        if not union(a, b, 1):
            return False
    for g in groups:
        g = sorted(g)
        for x in g[1:]:
            if not union(g[0], x, 0):
                return False
    for g1, g2 in opposite_pairs:
        if not union(min(g1), min(g2), 1):
            return False
    return True


NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def embed(edges, groups, anc_adj, size=9):
    """Data sites + 4 distinct ancilla sites; ancilla j adjacent to every qutrit of
    groups[j]; ancilla pairs in anc_adj adjacent; data couplers at distance 1."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    close = defaultdict(set)
    for g in groups:
        for a, b in itertools.combinations(g, 2):
            close[a].add(b)
            close[b].add(a)
    order = sorted(range(5), key=lambda q: -(len(adj[q]) * 10 + len(close[q])))
    pts = [(x, y) for x in range(size) for y in range(size)]
    pos = {}

    def ok(q, p):
        for r, pr in pos.items():
            d = abs(p[0] - pr[0]) + abs(p[1] - pr[1])
            if d == 0 or (r in adj[q] and d != 1) or (r in close[q] and d != 2):
                return False
        return True

    def ancillas():
        occ = set(pos.values())
        cand = []
        for g in groups:
            sites = None
            for q in g:
                s = {(pos[q][0] + dx, pos[q][1] + dy) for dx, dy in NB} - occ
                sites = s if sites is None else sites & s
            if not sites:
                return None
            cand.append(sorted(sites))
        for pick in itertools.product(*cand):
            if len(set(pick)) < 4:
                continue
            if all(abs(pick[i][0] - pick[j][0]) + abs(pick[i][1] - pick[j][1]) == 1 for i, j in anc_adj):
                return pick
        return None

    def rec(i):
        if i == 5:
            a = ancillas()
            return (dict(pos), a) if a else None
        q = order[i]
        for p in (pts if pos else [(size // 2, size // 2)]):
            if ok(q, p):
                pos[q] = p
                r = rec(i + 1)
                if r:
                    return r
                del pos[q]
        return None

    return rec(0)


def analyse(timeline, max_gates, allow_transfer):
    L = sum(len(w) for w in timeline)
    edges = sorted({(2, 3)} | {tuple(sorted(g[:2])) for w in timeline for g in w})
    if not colour_ok(edges, [], []):
        return []
    budget = max_gates - 2 * L
    segs = segments(timeline)
    # cheap lower bound: both ancillas of a sector touch every data qutrit and are coupled
    full = frozenset(range(5))
    lbs = []
    for kind, a, b in (("Z", "G1", "G2"), ("X", "G3", "G4")):
        m, s = read_moves(kind, segs, full)
        tgt = enc(FS.CHECKS[a][1]) * NV + enc(FS.CHECKS[b][1])
        lbs.append(int(sector_bfs(len(segs), m, s, m, s, allow_transfer, tgt, ADD, MUL, max(budget, 1))))
    if sum(lbs) > budget:
        return []
    zt = sector_table("Z", FS.CHECKS["G1"][1], FS.CHECKS["G2"][1], segs, min(max(budget - 2, 1), 30), allow_transfer)
    xt = sector_table("X", FS.CHECKS["G3"][1], FS.CHECKS["G4"][1], segs, min(max(budget - 2, 1), 30), allow_transfer)
    zl = sorted(zt.items(), key=lambda kv: kv[1])
    xl = sorted(xt.items(), key=lambda kv: kv[1])
    if not zl or not xl or zl[0][1] + xl[0][1] > budget:
        return []
    out, seen = [], set()
    for (T1, T2, c12), zc in zl:
        if zc + xl[0][1] > budget:
            break
        for (T3, T4, c34), xc in xl:
            if zc + xc > budget:
                break
            groups = [T1, T2, T3, T4]
            if any(len(groups[i] & groups[j]) > 2 for i in range(4) for j in range(i + 1, 4)):
                continue
            # data degree: ancillas touching q + data couplers at q <= 4
            deg = [sum(q in g for g in groups) + sum(q in e for e in edges) for q in range(5)]
            if max(deg) > 4:
                continue
            opp = ([(T1, T2)] if c12 else []) + ([(T3, T4)] if c34 else [])
            if not colour_ok(edges, groups, opp):
                continue
            key = (tuple(edges), tuple(tuple(sorted(g)) for g in groups), c12, c34)
            if key in seen:
                continue
            seen.add(key)
            anc_adj = ([(0, 1)] if c12 else []) + ([(2, 3)] if c34 else [])
            e = embed(edges, groups, anc_adj)
            if e:
                out.append((2 * L + zc + xc, timeline, groups, c12, c34, zc, xc, edges, e))
    return out


def _work(args):
    return analyse(*args)


def fmt_timeline(t):
    return " | ".join(" ".join(f"{'CNOT' if g[2] == 1 else 'CNOT_INV'}({g[0]}->{g[1]})" for g in w) for w in t)


if __name__ == "__main__":
    shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1].split(";")]
    max_gates = int(sys.argv[2])
    allow_transfer = "--no-transfer" not in sys.argv
    if "--safe" in sys.argv:
        # data windows only on pairs that carry no weight-2 logical: a fault on the target
        # after the last in-window read becomes a two-qutrit error on the pair at the pop,
        # which is X0X3 / X2X4 / Z0Z2 / Z3Z4 (silent) for the pairs 03, 24, 02, 34
        unsafe = {frozenset(p) for p in [(0, 2), (0, 3), (2, 4), (3, 4)]}
        FS.GATES[:] = [g for g in FS.GATES if frozenset(g[:2]) not in unsafe]
        print(f"--safe: data windows on {sorted({tuple(sorted(g[:2])) for g in FS.GATES})}", flush=True)
    rows = []
    for shape in shapes:
        jobs = [(t, max_gates, allow_transfer) for t in FS.timelines(shape)] if shape != (0,) else [((), max_gates, allow_transfer)]
        with Pool(min(110, len(jobs))) as pool:
            for good in pool.imap_unordered(_work, jobs, chunksize=1):
                rows += good
        best = min((r[0] for r in rows), default=None)
        print(f"windows {shape}: {len(jobs)} timelines, {len(rows)} embeddable candidates so far, best {best}",
              flush=True)
    rows.sort(key=lambda r: (r[0], sum(len(g) for g in r[2]) + len(r[7]) + r[3] + r[4]))
    print(f"\nbest candidates (<= {max_gates} two-qutrit gates incl. data-data gates):")
    for total, t, groups, c12, c34, zc, xc, edges, (pos, anc) in rows[:300]:
        ncoup = len(edges) + sum(len(g) for g in groups) + c12 + c34
        print(f"  {total:2d} gates (Z {zc}, X {xc}, data {total - zc - xc})  frames [{fmt_timeline(t)}]  "
              f"data couplers {edges}  couplers {ncoup}  "
              + "  ".join(f"{FS.NAMES[j]}:{sorted(g)}" for j, g in enumerate(groups))
              + f"  a1-a2 {'Y' if c12 else 'n'} a3-a4 {'Y' if c34 else 'n'}  data {pos} anc {anc}", flush=True)
