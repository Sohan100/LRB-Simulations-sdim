"""Build, schedule and verify one-ancilla square-lattice rounds found by the searches.

A candidate = (timeline of data-CNOT windows, touched sets T1..T4, coupling flags
a1-a2 / a3-a4). For each sector (Z: a1 G1, a2 G2; X: a3 G3, a4 G4) every
minimum-cost event sequence (reads per segment, transfers) is enumerated from the
breadth-first distance table. Sequences are merged with the data-gate windows into
op lists (random interleavings), scheduled ASAP with the production scheduler, and
the shallowest are checked with ../hw_layouts_check.py: zero-noise rounds (with and
without H-bar), the 40 single data Paulis (deterministic, QND, right syndrome), and
exhaustive single faults (2 rounds, with and without H-bar).

Wires: data 0..4, a1 = 5 (G1), a2 = 6 (G2), a3 = 7 (G3), a4 = 8 (G4).
Gate conventions: Z read CNOT^c(q -> a); X read CNOT^c(a -> q) between H and H_INV
on a; Z transfer a_Q += c a_P: CNOT^c(P -> Q); X transfer: CNOT^-c(Q -> P).
"""
import itertools
import random
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import oa_frame_search as FS  # noqa: E402
import oa_transfer_search as TS  # noqa: E402

WIRE = {"G1": 5, "G2": 6, "G3": 7, "G4": 8}
NV = 243


def cn(c):
    return "CNOT" if c % 3 == 1 else "CNOT_INV"


def sector_paths(kind, tP, tQ, TP, TQ, coupled, segs, limit=4000, seed=0):
    """All (up to limit) minimum-cost event sequences: list of (seg, event) with
    event = ('rP', q, c) | ('rQ', q, c) | ('tPQ', c) | ('tQP', c)."""
    moves = []   # (seg, event, func(p, q) -> (p, q))
    for k, fr in enumerate(segs):
        for who, T in (("rP", TP), ("rQ", TQ)):
            for q in sorted(T):
                e = tuple(int(i == q) for i in range(5))
                r = FS.pull_back(kind, e, fr)
                for c in (1, 2):
                    v = TS.enc([(c * x) % 3 for x in r])
                    moves.append((k, (who, q, c), v))
        if coupled:
            for c in (1, 2):
                moves.append((k, ("tPQ", c), None))
                moves.append((k, ("tQP", c), None))

    def apply(state, mv):
        k, ev, v = mv
        p, q = state
        if ev[0] == "rP":
            return TS.ADD[p, v], q
        if ev[0] == "rQ":
            return p, TS.ADD[q, v]
        if ev[0] == "tPQ":
            return p, TS.ADD[q, TS.MUL[ev[1], p]]
        return TS.ADD[p, TS.MUL[ev[1], q]], q

    nseg = len(segs)
    # forward BFS distances over (seg, p, q); moving to the next segment is free
    INF = 99
    dist = {}
    start = (0, 0, 0)
    dist[start] = 0
    for k in range(nseg):
        if k > 0:
            for (kk, p, q), d in list(dist.items()):
                if kk == k - 1 and dist.get((k, p, q), INF) > d:
                    dist[(k, p, q)] = d
        frontier = sorted(((d, s) for s, d in dist.items() if s[0] == k))
        dq = deque(s for _, s in frontier)
        while dq:
            s = dq.popleft()
            d = dist[s]
            for mv in moves:
                if mv[0] != k:
                    continue
                p2, q2 = apply((s[1], s[2]), mv)
                t = (k, int(p2), int(q2))
                if dist.get(t, INF) > d + 1:
                    dist[t] = d + 1
                    dq.append(t)
    goal = (nseg - 1, TS.enc(tP), TS.enc(tQ))
    best = dist.get(goal, INF)
    if best >= INF:
        return best, []
    # backward enumeration of shortest paths
    paths = []
    rng = random.Random(seed)

    def back(state, suffix):
        if len(paths) >= limit:
            return
        k, p, q = state
        d = dist[state]
        if d == 0 and k == 0 and p == 0 and q == 0:
            paths.append(list(reversed(suffix)))
            return
        preds = []
        # free segment advance
        if k > 0 and dist.get((k - 1, p, q), INF) == d:
            preds.append(((k - 1, p, q), None))
        for mv in moves:
            if mv[0] != k:
                continue
            # find predecessor states s with apply(s, mv) == (p, q) and dist == d-1
            for (kk, pp, qq), dd in []:
                pass
        # predecessors by inverting the move
        for mv in moves:
            if mv[0] != k:
                continue
            kind_ev = mv[1][0]
            if kind_ev == "rP":
                pp, qq = int(TS.ADD[p, TS.MUL[2, mv[2]]]), q
            elif kind_ev == "rQ":
                pp, qq = p, int(TS.ADD[q, TS.MUL[2, mv[2]]])
            elif kind_ev == "tPQ":
                pp, qq = p, int(TS.ADD[q, TS.MUL[(-mv[1][1]) % 3, p]])
            else:
                pp, qq = int(TS.ADD[p, TS.MUL[(-mv[1][1]) % 3, q]]), q
            if dist.get((k, pp, qq), INF) == d - 1:
                preds.append(((k, pp, qq), (k, mv[1])))
        rng.shuffle(preds)
        for st, ev in preds:
            back(st, suffix + ([ev] if ev else []))

    back(goal, [])
    # dedupe
    uniq = []
    seen = set()
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return best, uniq


def sector_ops(kind, P, Q, path):
    out = []
    for k, ev in path:
        if ev[0] == "rP" or ev[0] == "rQ":
            a = P if ev[0] == "rP" else Q
            q, c = ev[1], ev[2]
            out.append((k, ("g", cn(c), q, a) if kind == "Z" else ("g", cn(c), a, q)))
        else:
            src, dst = (P, Q) if ev[0] == "tPQ" else (Q, P)
            c = ev[1]
            out.append((k, ("g", cn(c), src, dst) if kind == "Z" else ("g", cn(-c), dst, src)))
    return out


def window_ops(timeline):
    """Segment boundaries: data gates between segment k and k+1."""
    bounds = []
    for w in timeline:
        L = len(w)
        for g in w:
            bounds.append([("g", cn(g[2]), g[0], g[1])])
        for g in reversed(w):
            bounds.append([("g", cn(-g[2]), g[0], g[1])])
    return bounds


def assemble(zpath, xpath, timeline, rng):
    """Random interleaving respecting segment order; X ancillas get H / H_INV."""
    zops = sector_ops("Z", 5, 6, zpath)
    xops = sector_ops("X", 7, 8, xpath)
    bounds = window_ops(timeline)
    nseg = len(bounds) + 1
    ops = [("r", w) for w in (5, 6, 7, 8)] + [("g", "H", 7), ("g", "H", 8)]
    for k in range(nseg):
        zs = [op for kk, op in zops if kk == k]
        xs = [op for kk, op in xops if kk == k]
        # random merge keeping each sector's own order
        merged = []
        i = j = 0
        while i < len(zs) or j < len(xs):
            if j >= len(xs) or (i < len(zs) and rng.random() < 0.5):
                merged.append(zs[i]); i += 1
            else:
                merged.append(xs[j]); j += 1
        ops += merged
        if k < nseg - 1:
            ops += bounds[k]
    ops += [("g", "H_INV", 7), ("g", "H_INV", 8)]
    ops += [("m", 5, "G1"), ("m", 6, "G2"), ("m", 7, "G3"), ("m", 8, "G4")]
    return ops
