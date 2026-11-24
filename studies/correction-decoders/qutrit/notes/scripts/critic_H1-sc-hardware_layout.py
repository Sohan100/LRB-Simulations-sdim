"""Critic check of H1 layout claims C1, C2, C3, C5, C7, C22 (independent code).

Independent of h1_layout_analysis.py: own backtracking subgraph-monomorphism
counter (no networkx VF2), own BFS, own placement search with a search
radius chosen so that a simple bound makes it exhaustive (see notes below).
networkx is used only for the planarity test (C7).

Run:  module load python/3.11; python critic_H1-sc-hardware_layout.py
Runtime: about 1-2 minutes on one core.
"""
from __future__ import annotations

import itertools
from collections import deque

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# Couplings read from src/lrb/code_definitions.py (z_ALT and x blocks).
CHECK = [(5, 0), (5, 1), (5, 3), (6, 1), (6, 2), (6, 4),
         (7, 0), (7, 1), (7, 2), (8, 3), (8, 1), (8, 4)]
SPLIT = [(5, 0), (5, 1), (3, 9), (9, 5), (6, 1), (6, 2), (6, 4),
         (7, 0), (7, 1), (7, 2), (8, 3), (8, 1), (4, 10), (10, 8)]
FOLD = (2, 3)
ENCODER = [(2, 1), (1, 0), (3, 0), (4, 2)]   # logical_plus_initial_state
SUPP = {5: (0, 1, 3), 6: (1, 2, 4), 7: (0, 1, 2), 8: (1, 3, 4)}

# Garnet coupling map, transcribed from the fetched fake_garnet.py.
GARNET = [(1, 2), (1, 4), (2, 5), (3, 4), (8, 3), (4, 5), (9, 4), (5, 6),
          (10, 5), (6, 7), (11, 6), (12, 7), (8, 9), (8, 13), (9, 10),
          (9, 14), (10, 11), (10, 15), (11, 12), (16, 11), (17, 12),
          (13, 14), (14, 15), (18, 14), (16, 15), (19, 15), (16, 17),
          (16, 20), (18, 19), (19, 20)]


def adj(edges):
    a = {}
    for u, v in edges:
        a.setdefault(u, set()).add(v)
        a.setdefault(v, set()).add(u)
    return a


def grid_edges(n):
    e = []
    for r in range(n):
        for c in range(n):
            if r + 1 < n:
                e.append(((r, c), (r + 1, c)))
            if c + 1 < n:
                e.append(((r, c), (r, c + 1)))
    return e


def two_colourable(edges, parity=None):
    """2-colouring with per-edge parity (1 = endpoints differ, 0 = equal)."""
    a = {}
    for (u, v) in edges:
        p = 1 if parity is None else parity[(u, v)]
        a.setdefault(u, []).append((v, p))
        a.setdefault(v, []).append((u, p))
    col = {}
    for s in a:
        if s in col:
            continue
        col[s] = 0
        q = [s]
        while q:
            u = q.pop()
            for v, p in a[u]:
                want = col[u] ^ p
                if v not in col:
                    col[v] = want
                    q.append(v)
                elif col[v] != want:
                    return False
    return True


def monomorphisms(pattern_edges, host_edges, limit=None):
    """Count (or list up to limit) injective maps preserving pattern edges."""
    P = adj(pattern_edges)
    H = adj(host_edges)
    order = sorted(P, key=lambda v: -len(P[v]))
    # BFS-like order: next vertex has most already-placed neighbours.
    placed_order = [order[0]]
    rest = set(order[1:])
    while rest:
        best = max(rest, key=lambda v: (len(P[v] & set(placed_order)),
                                        len(P[v])))
        placed_order.append(best)
        rest.remove(best)
    found = []
    count = 0

    def rec(i, m, used):
        nonlocal count
        if limit is not None and count >= limit:
            return
        if i == len(placed_order):
            count += 1
            if limit is not None:
                found.append(dict(m))
            return
        v = placed_order[i]
        nbrs = [m[w] for w in P[v] if w in m]
        cands = set(H) if not nbrs else set.intersection(*(H[x] for x in nbrs))
        for h in cands:
            if h in used or len(H[h]) < len(P[v]):
                continue
            m[v] = h
            used.add(h)
            rec(i + 1, m, used)
            used.discard(h)
            del m[v]

    rec(0, {}, set())
    return count, found


def odd_cycle_check(edges, cyc):
    es = {frozenset(e) for e in edges}
    ok = all(frozenset((cyc[i], cyc[(i + 1) % len(cyc)])) in es
             for i in range(len(cyc)))
    return ok and len(cyc) % 2 == 1


def bfs_avoid(src, data_sites, lo, hi):
    """Grid distances from src on [lo,hi]^2; data sites never transited."""
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        if u != src and u in data_sites:
            continue
        r, c = u
        for v in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if lo <= v[0] <= hi and lo <= v[1] <= hi and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def layout_cost(data_pos, lo, hi, anc_may_relay=True):
    """Min total check relay hops over ancilla placements (assignment)."""
    data_sites = set(data_pos.values())
    dists = {d: bfs_avoid(p, data_sites, lo, hi) for d, p in data_pos.items()}
    sites = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)
             if (r, c) not in data_sites]
    cost = np.full((4, len(sites)), 1e9)
    for k, a in enumerate(sorted(SUPP)):
        for j, s in enumerate(sites):
            tot = 0
            for d in SUPP[a]:
                L = dists[d].get(s)
                if L is None:
                    tot = 1e9
                    break
                tot += L - 1
            cost[k, j] = tot
    if not anc_may_relay:
        return None
    rows, cols = linear_sum_assignment(cost)
    return int(round(cost[rows, cols].sum())), {sorted(SUPP)[k]: sites[j]
                                                 for k, j in zip(rows, cols)}


def lower_bound(data_pos):
    lb = 0
    for a, s in SUPP.items():
        xs = [data_pos[d][0] for d in s]
        ys = [data_pos[d][1] for d in s]
        lb += max(xs) - min(xs) + max(ys) - min(ys) - 3
    return lb


def search(pos3, R, target, lo=-9, hi=10):
    """All placements (2 at origin, 3 at pos3) with data 0,1,4 within L1
    distance R of 2 or 3 and lower bound <= target; exact cost for each."""
    md = lambda u, v: abs(u[0] - v[0]) + abs(u[1] - v[1])
    p2 = (0, 0)
    near = [(r, c) for r in range(-R - 2, R + 3) for c in range(-R - 2, R + 3)
            if (r, c) not in (p2, pos3)
            and min(md((r, c), p2), md((r, c), pos3)) <= R]
    best = None
    n_eval = 0
    for q0, q1, q4 in itertools.permutations(near, 3):
        dp = {0: q0, 1: q1, 2: p2, 3: pos3, 4: q4}
        if lower_bound(dp) > target:
            continue
        if md(p2, pos3) == 2:
            # one-relay fold: some common neighbour of 2 and 3 must be free
            mids = [(r, c) for (r, c) in ((0, 1), (1, 0), (0, -1), (-1, 0))
                    if md((r, c), pos3) == 1]
            if all(m in (q0, q1, q4) for m in mids):
                continue
        n_eval += 1
        c, anc = layout_cost(dp, lo, hi)
        if best is None or c < best[0]:
            best = (c, dp, anc)
    return best, n_eval, len(near)


def main():
    print("== C1: checks-only graph ==")
    print("  2-colourable (bipartite):", two_colourable(CHECK))
    g3 = grid_edges(3)
    n3, ex = monomorphisms(CHECK, g3, limit=1)
    n3_all, _ = monomorphisms(CHECK, g3)
    print(f"  monomorphisms into 3x3 grid: {n3_all} (8 = dihedral group, so"
          f" the placement is unique up to symmetry); example {ex[0]}")
    ng, exg = monomorphisms(CHECK, GARNET, limit=1)
    ng_all, _ = monomorphisms(CHECK, GARNET)
    print(f"  Garnet: bipartite={two_colourable(GARNET)}, monomorphisms of "
          f"checks-only graph: {ng_all}; example {exg[0] if exg else None}")
    H1_PLACEMENT = {3: (0, 0), 5: (0, 1), 0: (0, 2), 8: (1, 0), 1: (1, 1),
                    7: (1, 2), 4: (2, 0), 6: (2, 1), 2: (2, 2)}
    ge = {frozenset(e) for e in g3}
    print("  H1's printed 3x3 placement valid:",
          all(frozenset((H1_PLACEMENT[u], H1_PLACEMENT[v])) in ge
              for u, v in CHECK))

    print("\n== C2: checks + fold ==")
    print("  bipartite:", two_colourable(CHECK + [FOLD]),
          "; 7-0-5-3-2 is an odd cycle:",
          odd_cycle_check(CHECK + [FOLD], [7, 0, 5, 3, 2]))
    data_sites = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}
    d = bfs_avoid((2, 2), data_sites, -6, 8)
    print("  planar patch: L1 distance 2-3 =", 4,
          "; shortest data-avoiding path length =", d[(0, 0)])

    print("\n== C3: split-ancilla variant ==")
    print("  bipartite:", two_colourable(SPLIT),
          "; 8-1-5-9-3 is an odd cycle:", odd_cycle_check(SPLIT, [8, 1, 5, 9, 3]))
    # Parity-restoring relay sets: edges given EVEN parity (one relay each).
    sets = {}
    for k in (1, 2, 3):
        for S in itertools.combinations(CHECK, k):
            par = {e: (0 if e in S else 1) for e in CHECK}
            par[FOLD] = 1
            if two_colourable(CHECK + [FOLD], par):
                sets.setdefault(k, []).append(S)
        if k in sets:
            break
    print("  minimal even-parity (single-relay) sets with native fold:",
          sets)
    # Repo's split choice {(5,3),(8,4)}: parity-consistent?
    par = {e: (0 if e in [(5, 3), (8, 4)] else 1) for e in CHECK}
    par[FOLD] = 1
    print("  repo split set {S_Z1-3, S_X2-4} parity-consistent with fold:",
          two_colourable(CHECK + [FOLD], par))
    for S in sets[min(sets)]:
        edges = [e for e in CHECK if e not in S] + [FOLD]
        r = 100
        for (a, dd) in S:
            edges += [(a, r), (r, dd)]
            r += 1
        n6, _ = monomorphisms(edges, grid_edges(6), limit=1)
        n7, _ = monomorphisms(edges, grid_edges(7), limit=1)
        ngar, _ = monomorphisms(edges, GARNET, limit=1)
        print(f"  single-hop relays on {S}: embeds in 6x6 {n6 > 0}, "
              f"7x7 {n7 > 0}, Garnet {ngar > 0}")

    print("\n== C5: cheapest square-lattice layouts (own search) ==")
    print("  Exhaustiveness: if data q in {0,1,4} is farther than R from both"
          " 2 and 3, the two (data 0, 4) or four (data 1) ancillas pairing q"
          " with 2 or 3 cost >= 2(R+1-3); R=5 gives >= 6, so any layout with"
          " total <= 5 has all of 0,1,4 within R=5. H1 used R=3, for which"
          " this bound gives only >= 2.")
    for pos3, tgt, name in (((0, 1), 5, "native fold (2-3 adjacent)"),
                            ((0, 2), 4, "fold at distance 2, straight"),
                            ((1, 1), 4, "fold at distance 2, diagonal")):
        best, n_eval, n_near = search(pos3, 5, tgt)
        print(f"  {name}: placements with lower bound <= {tgt}: {n_eval}; "
              f"best exact cost among them: "
              f"{None if best is None else best[0]}")
    # The 6-hop optimum itself (search at target 6, R=5 is exhaustive for
    # cost <= 5 only; we just exhibit a 6 and confirm H1's layout costs 6).
    H1_GRID = {"D0": (2, 3), "A5": (3, 3), "R1": (3, 4), "R2": (3, 5),
               "A7": (4, 3), "D2": (4, 4), "D3": (4, 5), "D1": (5, 3),
               "A6": (5, 4), "A8": (5, 5), "D4": (5, 6)}
    dp = {k: H1_GRID[f"D{k}"] for k in range(5)}
    ds = set(dp.values())
    tot = 0
    for a, s in SUPP.items():
        for dd in s:
            L = bfs_avoid(dp[dd], ds, -3, 12)[H1_GRID[f"A{a}"]]
            tot += L - 1
    print(f"  H1's drawn native-fold layout: total check relay hops = {tot};"
          f" fold D2-D3 adjacent: {abs(dp[2][1] - dp[3][1]) + abs(dp[2][0] - dp[3][0]) == 1}")
    # Garnet placement given in the notes.
    QB = {"D0": 17, "D1": 14, "D2": 10, "D3": 5, "D4": 1, "A5": 16, "A6": 9,
          "A7": 15, "A8": 4, "R1": 11, "R2": 6}
    used = [("A5", "D0"), ("A6", "D1"), ("A6", "D2"), ("A7", "D1"),
            ("A7", "D2"), ("A8", "D3"), ("A8", "D4"), ("D2", "D3"),
            ("A5", "A7"), ("A5", "R1"), ("R1", "R2"), ("R2", "D3"),
            ("A6", "A8")]
    gset = {frozenset(e) for e in GARNET}
    print("  13 couplers of the notes' layout all present on Garnet with the"
          " stated QB map:",
          all(frozenset((QB[u], QB[v])) in gset for u, v in used))
    gpos = {frozenset((H1_GRID[u], H1_GRID[v])) for u, v in used}
    print("  same 13 couplers are grid-adjacent in the drawn layout:",
          all(abs(H1_GRID[u][0] - H1_GRID[v][0]) + abs(H1_GRID[u][1] - H1_GRID[v][1]) == 1
              for u, v in used), "; distinct:", len(gpos) == 13)

    print("\n== C7: planarity ==")
    for name, e in (("checks+fold", CHECK + [FOLD]),
                    ("checks+fold+encoder", CHECK + [FOLD] + ENCODER)):
        g = nx.Graph(e)
        print(f"  {name}: planar={nx.check_planarity(g)[0]}, max degree="
              f"{max(dict(g.degree()).values())}, degrees={dict(sorted(g.degree()))}")

    print("\n== C22: Stage A graph on Garnet ==")
    stageA = ENCODER + [FOLD]
    print("  bipartite:", two_colourable(stageA))
    n, ex = monomorphisms(stageA, GARNET, limit=1)
    nall, _ = monomorphisms(stageA, GARNET)
    exA = {0: 2, 1: 1, 2: 4, 3: 5, 4: 9}
    print(f"  labelled monomorphisms into Garnet: {nall}; notes' example "
          f"valid: {all(frozenset((exA[u], exA[v])) in gset for u, v in stageA)}")
    n4cyc = 0
    A = adj(GARNET)
    for a, b, c, dd in itertools.permutations(A, 4):
        if b in A[a] and c in A[b] and dd in A[c] and a in A[dd]:
            n4cyc += 1
    print(f"  4-cycles in Garnet (labelled, /8 = unlabelled): {n4cyc} "
          f"({n4cyc // 8} plaquettes)")


if __name__ == "__main__":
    main()
