"""H1 layout analysis for the [[5,1,2]]_3 folded qutrit code on a square lattice.

Questions answered (all by exhaustive graph search, no physics assumptions):

1. Is the two-qutrit interaction graph of each circuit variant bipartite?
   A square lattice with nearest-neighbour couplers is bipartite, so a
   non-bipartite interaction graph cannot be embedded with every coupling
   nearest-neighbour.
2. Does the graph embed (as a subgraph monomorphism) into an n x n grid and
   into the IQM Garnet coupling map (30 couplers, taken from IQM's
   qiskit-on-iqm fake Garnet backend file)?
3. Which minimal sets of stabilizer couplings must be routed through relay
   ancillas so that the folded code, with its fold coupling 2-3 kept native,
   embeds in a square lattice?
4. Expected number of logical H and P tokens per random Clifford of the
   repo's 216-entry catalogue (sets the cost of a non-native fold coupling).

Run:  module load python/3.11; python h1_layout_analysis.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import networkx as nx
from networkx.algorithms import isomorphism as iso

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

DATA = [0, 1, 2, 3, 4]

# Stabilizer couplings of the single-ancilla circuits (ancilla, data).
SINGLE_CHECK_EDGES = [
    (5, 0), (5, 1), (5, 3),      # S_Z1 = Z0 Z1^-1 Z3^-1
    (6, 1), (6, 2), (6, 4),      # S_Z2 = Z1 Z2 Z4^-1
    (7, 0), (7, 1), (7, 2),      # S_X1 = X0 X1 X2^-1
    (8, 3), (8, 1), (8, 4),      # S_X2 = X1^-1 X3 X4^-1
]
# Repo split-ancilla variant (FoldedQutritSplitAncillaUniformDetectionCode):
# relay 9 carries the -z3 contribution into 5, relay 10 the -x4 into 8.
SPLIT_CHECK_EDGES = [
    (5, 0), (5, 1), (3, 9), (9, 5),
    (6, 1), (6, 2), (6, 4),
    (7, 0), (7, 1), (7, 2),
    (8, 3), (8, 1), (4, 10), (10, 8),
]
# Logical H uses SWAP(2,3); logical P uses CZ(2,3): the fold coupling.
FOLD_EDGE = (2, 3)

# IQM Garnet coupling map as listed in IQM's qiskit-on-iqm fake backend
# (src/iqm/qiskit_iqm/fake_backends/fake_garnet.py, fetched 2026-10-28).
GARNET_EDGES = [
    (1, 2), (1, 4), (2, 5), (3, 4), (8, 3), (4, 5), (9, 4), (5, 6),
    (10, 5), (6, 7), (11, 6), (12, 7), (8, 9), (8, 13), (9, 10), (9, 14),
    (10, 11), (10, 15), (11, 12), (16, 11), (17, 12), (13, 14), (14, 15),
    (18, 14), (16, 15), (19, 15), (16, 17), (16, 20), (18, 19), (19, 20),
]


def graph(edges) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def odd_cycle(g: nx.Graph):
    """Return one odd cycle (as a vertex list) or None if bipartite."""
    if nx.is_bipartite(g):
        return None
    # Shortest odd cycle through BFS layering from every vertex.
    best = None
    for root in g.nodes:
        dist = {root: 0}
        parent = {root: None}
        queue = [root]
        for u in queue:
            for v in g.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] == dist[u] and parent[u] != v:
                    # Same BFS layer: odd cycle root..u v..root.
                    pu, pv = [u], [v]
                    while pu[-1] != pv[-1]:
                        pu.append(parent[pu[-1]])
                        pv.append(parent[pv[-1]])
                    cyc = pu + pv[-2::-1]
                    if best is None or len(cyc) < len(best):
                        best = cyc
    return best


def embeds(host: nx.Graph, g: nx.Graph):
    """Return one monomorphism g -> host as {g_node: host_node}, or None."""
    gm = iso.GraphMatcher(host, g)
    for mapping in gm.subgraph_monomorphisms_iter():
        return {gv: hv for hv, gv in mapping.items()}
    return None


def with_relays(base_edges, removed, fold=True):
    """Replace each removed (ancilla, data) coupling by a 2-step relay path."""
    edges = [e for e in base_edges if e not in removed]
    next_relay = 100
    relay_of = {}
    for (a, d) in removed:
        r = next_relay
        next_relay += 1
        edges += [(d, r), (r, a)]
        relay_of[(a, d)] = r
    if fold:
        edges.append(FOLD_EDGE)
    return edges, relay_of


ANCILLA_SUPPORT = {5: (0, 1, 3), 6: (1, 2, 4), 7: (0, 1, 2), 8: (3, 1, 4)}


def bfs_no_data_transit(size, source, data_sites):
    """Grid distances from source; data sites are reachable but not crossed.

    Relay chains may pass through free sites or ancilla sites (which hold
    |0> outside their own extraction step) but never through a data qutrit.
    """
    dist = {source: 0}
    frontier = [source]
    while frontier:
        nxt = []
        for (r, c) in frontier:
            for v in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if not (0 <= v[0] < size and 0 <= v[1] < size):
                    continue
                if v in dist:
                    continue
                dist[v] = dist[(r, c)] + 1
                if v not in data_sites:
                    nxt.append(v)
        frontier = nxt
    return dist


class _FastGrid:
    """Integer-indexed square grid for fast repeated BFS."""

    def __init__(self, size):
        import numpy as np
        self.size = size
        self.n = size * size
        self.nbrs = []
        for r in range(size):
            for c in range(size):
                lst = []
                for (rr, cc) in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= rr < size and 0 <= cc < size:
                        lst.append(rr * size + cc)
                self.nbrs.append(lst)
        self.np = np

    def idx(self, rc):
        return rc[0] * self.size + rc[1]

    def rc(self, i):
        return divmod(i, self.size)

    def bfs(self, src, data_mask):
        """Distances from src; data sites reachable but never transited."""
        INF = 10 ** 6
        dist = [INF] * self.n
        dist[src] = 0
        frontier = [src]
        nbrs = self.nbrs
        while frontier:
            nxt = []
            for u in frontier:
                du = dist[u] + 1
                for v in nbrs[u]:
                    if dist[v] == INF:
                        dist[v] = du
                        if not data_mask[v]:
                            nxt.append(v)
            frontier = nxt
        return self.np.array(dist, dtype=float)


def _best_ancilla_assignment(fg, data_idx):
    """Optimal ancilla sites for fixed data sites (assignment problem)."""
    np = fg.np
    from scipy.optimize import linear_sum_assignment

    data_mask = [False] * fg.n
    for i in data_idx.values():
        data_mask[i] = True
    dist = {d: fg.bfs(i, data_mask) for d, i in data_idx.items()}
    anc = sorted(ANCILLA_SUPPORT)
    cost = np.empty((len(anc), fg.n))
    for k, a in enumerate(anc):
        cost[k] = sum(dist[d] - 1.0 for d in ANCILLA_SUPPORT[a])
    cost[:, [i for i in range(fg.n) if data_mask[i]]] = 1e9
    rows, cols = linear_sum_assignment(cost)
    total = int(round(cost[rows, cols].sum()))
    layout = {d: fg.rc(i) for d, i in data_idx.items()}
    for k, j in zip(rows, cols):
        layout[anc[k]] = fg.rc(j)
    return total, layout, dist


def _manhattan_lower_bound(data_rc):
    """Lower bound on total check relay hops ignoring blocking/collisions.

    For three points the L1 sum of distances to the best site equals the
    x-spread plus the y-spread, so each ancilla needs at least
    spread_x + spread_y - 3 relay hops.
    """
    lb = 0
    for supp in ANCILLA_SUPPORT.values():
        xs = [data_rc[d][0] for d in supp]
        ys = [data_rc[d][1] for d in supp]
        lb += max(xs) - min(xs) + max(ys) - min(ys) - 3
    return lb


def _search(size, pos3_list, radius, threshold=6):
    """Exhaustive placement search, pruned by the Manhattan lower bound.

    Placements whose lower bound exceeds ``threshold`` cannot beat the
    native-fold optimum (6 hops, found without pruning in an earlier run of
    this script) and are skipped; this keeps the search to about a minute.
    """
    fg = _FastGrid(size)
    c = size // 2
    pos2 = (c, c)
    md = lambda u, v: abs(u[0] - v[0]) + abs(u[1] - v[1])
    result = {}
    for pos3 in pos3_list:
        near = [(r, cc) for r in range(size) for cc in range(size)
                if (r, cc) not in (pos2, pos3)
                and min(md((r, cc), pos2), md((r, cc), pos3)) <= radius]
        i2, i3 = fg.idx(pos2), fg.idx(pos3)
        for q0, q1, q4 in itertools.permutations(near, 3):
            data_rc = {0: q0, 1: q1, 2: pos2, 3: pos3, 4: q4}
            if _manhattan_lower_bound(data_rc) > threshold:
                continue
            data_idx = {0: fg.idx(q0), 1: fg.idx(q1), 2: i2, 3: i3,
                        4: fg.idx(q4)}
            total, layout, dist = _best_ancilla_assignment(fg, data_idx)
            L = dist[2][i3]
            if L >= 10 ** 6:
                continue
            h_fold = int(L) - 1
            key = (h_fold, pos3)
            if key not in result or total < result[key][0]:
                result[key] = (total, layout)
    return result


def min_relay_layout_search(size: int = 9, radius: int = 3):
    """Native fold (2-3 adjacent): minimal check relay hops per round.

    Each stabilizer coupling (ancilla a, data d) at relay-path length L > 1 is
    realised by a coherent relay chain of L-1 relay qutrits (compute along
    the chain, couple, uncompute), as in the repo's split variant; this costs
    2(L-1) extra two-qutrit gates per check round. Relay paths may cross free
    sites or ancilla sites (|0> outside their own extraction step) but never
    data sites.
    """
    c = size // 2
    res = _search(size, [(c, c + 1)], radius)
    (h_fold, _), (total, layout) = min(res.items(), key=lambda kv: kv[1][0])
    return total, [layout]


def pareto_fold_distance_search(size: int = 9, radius: int = 3):
    """Minimal check-relay hops for fold relay-path length 1 or 2.

    Logical H needs no physical SWAP(2,3) (relabelling; see notes), so only
    logical P's CZ(2,3) needs the coupling; at relay-path length L it costs
    h_fold = L - 1 relay hops (2 h_fold extra two-qutrit gates) per logical
    P token. Data 3 is placed adjacent to 2, or at distance 2 (straight or
    diagonal); longer fold distances cannot win (see notes for the bound).
    Returns {h_fold: (min h_check, example layout)}.
    """
    c = size // 2
    res = _search(size, [(c, c + 1), (c, c + 2), (c + 1, c + 1)], radius)
    best = {}
    for (h_fold, pos3), (total, layout) in res.items():
        if h_fold not in best or total < best[h_fold][0]:
            best[h_fold] = (total, layout)
    return dict(sorted(best.items()))


def shortest_relay_path(src, dst, data_sites, size: int = 11):
    """Shortest grid path src -> dst whose interior avoids data sites."""
    from collections import deque
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        if u != src and u in data_sites:
            continue
        r, c = u
        for v in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= v[0] < size and 0 <= v[1] < size and v not in prev:
                prev[v] = u
                q.append(v)
    path = [dst]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def describe_layout(layout, size: int = 9):
    """Per-coupling relay-path lengths (data never crossed) for a layout."""
    data_sites = {layout[d] for d in DATA}
    out = []
    for a, supp in ANCILLA_SUPPORT.items():
        for d in supp:
            dist = bfs_no_data_transit(size, layout[d], data_sites)
            out.append(((a, d), dist.get(layout[a])))
    return out


def clifford_token_stats():
    from lrb.code_definitions import (QutritCliffordLibrary,
                                      single_qutrit_cliffords_exhaustive_strings)
    cat = single_qutrit_cliffords_exhaustive_strings
    inv = QutritCliffordLibrary.build_inverse_lookup()
    n_h = n_p = n_x = n_z = 0
    n_h_inv = n_p_inv = 0
    for c in cat:
        seq = QutritCliffordLibrary.qutrit_gate_seq_list(c)
        n_h += sum(t.startswith("H") for t in seq)
        n_p += sum(t.startswith("P") for t in seq)
        n_x += sum(t.startswith("X") for t in seq)
        n_z += sum(t.startswith("Z") for t in seq)
        iseq = inv[c]
        n_h_inv += sum(t.startswith("H") for t in iseq)
        n_p_inv += sum(t.startswith("P") for t in iseq)
    n = len(cat)
    return {
        "catalogue_entries": n,
        "distinct_entries": len(set(cat)),
        "mean_H_per_forward": n_h / n,
        "mean_P_per_forward": n_p / n,
        "mean_X_per_forward": n_x / n,
        "mean_Z_per_forward": n_z / n,
        "mean_H_per_inverse": n_h_inv / n,
        "mean_P_per_inverse": n_p_inv / n,
    }


def main() -> None:
    grids = {n: nx.grid_2d_graph(n, n) for n in (3, 4, 5, 6)}
    garnet = graph(GARNET_EDGES)
    print(f"Garnet graph: {garnet.number_of_nodes()} nodes, "
          f"{garnet.number_of_edges()} edges, bipartite="
          f"{nx.is_bipartite(garnet)}, max degree="
          f"{max(d for _, d in garnet.degree())}")

    variants = {
        "single-ancilla checks only": SINGLE_CHECK_EDGES,
        "single-ancilla checks + fold edge 2-3": SINGLE_CHECK_EDGES + [FOLD_EDGE],
        "split-ancilla (repo) checks only": SPLIT_CHECK_EDGES,
        "split-ancilla (repo) checks + fold edge 2-3": SPLIT_CHECK_EDGES + [FOLD_EDGE],
    }
    print("\n== Variant embeddability ==")
    for name, edges in variants.items():
        g = graph(edges)
        cyc = odd_cycle(g)
        res = {n: embeds(h, g) is not None for n, h in grids.items()}
        on_garnet = embeds(garnet, g)
        print(f"{name}: nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
              f"bipartite={cyc is None} shortest_odd_cycle={cyc} "
              f"grid_embeds={res} garnet_embeds={on_garnet is not None}")
        if name == "single-ancilla checks only":
            emb = embeds(grids[3], g)
            print("  3x3 embedding (wire -> (row, col)):",
                  dict(sorted(emb.items())))
            emb_g = on_garnet
            print("  one Garnet embedding (wire -> QB):",
                  dict(sorted(emb_g.items())))

    # Distance between 2 and 3 in the planar (unfolded) 3x3 layout.
    g_planar = graph(SINGLE_CHECK_EDGES)
    emb = embeds(grids[3], g_planar)
    d23 = nx.shortest_path_length(grids[3], emb[2], emb[3])
    print(f"\nPlanar 3x3 layout: lattice distance between data 2 and 3 = {d23}")

    # Minimal frustrated-edge sets with the fold edge native.
    print("\n== Minimal relay sets with native fold edge 2-3 ==")
    found = {}
    for k in (1, 2):
        for removed in itertools.combinations(SINGLE_CHECK_EDGES, k):
            base = [e for e in SINGLE_CHECK_EDGES if e not in removed]
            g_nb = graph(base + [FOLD_EDGE])
            if not nx.is_bipartite(g_nb):
                continue
            edges, relay_of = with_relays(SINGLE_CHECK_EDGES, removed)
            g_relay = graph(edges)
            ok_grid = {n: embeds(h, g_relay) is not None
                       for n, h in grids.items() if n >= 4}
            ok_garnet = embeds(garnet, g_relay)
            found[removed] = (ok_grid, ok_garnet is not None)
            print(f"remove {removed}: bipartite with fold edge; relay graph "
                  f"bipartite={nx.is_bipartite(g_relay)} "
                  f"grid_embeds={ok_grid} garnet_embeds={ok_garnet is not None}")
        if found:
            break
    # Explicit coordinates for the first embeddable two-relay solution.
    for removed, (ok_grid, ok_garnet) in found.items():
        if ok_grid.get(5) or ok_grid.get(4):
            edges, relay_of = with_relays(SINGLE_CHECK_EDGES, removed)
            n = 4 if ok_grid.get(4) else 5
            emb = embeds(grids[n], graph(edges))
            print(f"\nExample embedding for relays on {removed} in a {n}x{n} "
                  f"grid (relay ids {relay_of}):")
            for v in sorted(emb, key=lambda x: (x >= 100, x)):
                print(f"  wire {v} -> {emb[v]}")
            if ok_garnet:
                emb_g = embeds(garnet, graph(edges))
                print("  Garnet placement:", dict(sorted(emb_g.items())))
            break

    stats = clifford_token_stats()
    p_per_step = stats["mean_P_per_forward"] + stats["mean_P_per_inverse"]
    print("\n== Pareto front: fold relay hops vs check relay hops ==")
    print("   (extra two-qutrit gates per LRB step = 2*h_check + "
          f"2*h_fold*{p_per_step:.3f} logical P tokens per step)")
    for h_fold, (h_check, lay) in pareto_fold_distance_search().items():
        extra = 2 * h_check + 2 * h_fold * p_per_step
        far = [(e, L) for e, L in describe_layout(lay) if L > 1]
        print(f"  h_fold={h_fold}: min h_check={h_check}, extra 2q gates per "
              f"step={extra:.2f}; layout "
              + ", ".join(f"{w}:{lay[w]}" for w in sorted(lay)))
        print(f"    non-native checks: {far}")

    # Explicit relay paths for the native-fold optimum and a Garnet fit test.
    best_native = pareto_fold_distance_search()[0][1]
    print("\n== Native-fold optimum: explicit relay paths ==")
    used_edges = {tuple(sorted((best_native[2], best_native[3])))}
    data_sites = {best_native[d] for d in DATA}
    anc_sites = {best_native[a] for a in ANCILLA_SUPPORT}
    relay_only_sites = set()
    for a, supp in ANCILLA_SUPPORT.items():
        for d in supp:
            path = shortest_relay_path(best_native[d], best_native[a],
                                       data_sites)
            for u, v in zip(path, path[1:]):
                used_edges.add(tuple(sorted((u, v))))
            inner = path[1:-1]
            relay_only_sites |= {s for s in inner if s not in anc_sites}
            if inner:
                print(f"  coupling ancilla {a} - data {d}: path {path} "
                      f"({len(inner)} relay hop(s); relay sites that are "
                      f"other ancillas: "
                      f"{[s for s in inner if s in anc_sites]})")
    print(f"  extra relay-only qutrits: {len(relay_only_sites)} at "
          f"{sorted(relay_only_sites)}; total qutrits "
          f"{9 + len(relay_only_sites)}")
    g_used = nx.Graph()
    g_used.add_edges_from(used_edges)
    emb_g = embeds(garnet, g_used)
    print(f"  physical coupling graph: {g_used.number_of_nodes()} sites, "
          f"{g_used.number_of_edges()} couplers; fits in Garnet: "
          f"{emb_g is not None}")
    if emb_g is not None:
        name = {best_native[w]: w for w in best_native}
        print("  Garnet placement (wire or relay site -> QB): " + ", ".join(
            f"{name.get(s, 'relay' + str(s))}->QB{emb_g[s]}"
            for s in sorted(emb_g, key=lambda s: (s not in name,
                                                  name.get(s, 99)))))

    # Planar layout: fold path length avoiding data sites.
    planar = {0: (0, 2), 1: (1, 1), 2: (2, 2), 3: (0, 0), 4: (2, 0)}
    shift = lambda rc: (rc[0] + 3, rc[1] + 3)
    planar = {k: shift(v) for k, v in planar.items()}
    path = shortest_relay_path(planar[2], planar[3], set(planar.values()))
    print(f"\nPlanar 3x3 layout: shortest data-avoiding path from data 2 to "
          f"data 3 has length {len(path) - 1} ({len(path) - 2} relay hops)")

    print("\n== Clifford catalogue token statistics ==")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
