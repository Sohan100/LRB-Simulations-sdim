"""Can a coupling graph be laid out on a square grid with only nearest-neighbour
couplers (horizontal, vertical, or diagonal of one cell), with no two couplers
crossing? Exhaustive backtracking on a small grid. Also reports the number of
frequency groups needed (chromatic number) and triangles."""
import itertools
import sys
import networkx as nx

LAYOUTS = {
    "square lattice: a{012}, c{124}": {5: {0, 1, 2}, 6: {1, 2, 4}},
    "production: 4 ancillas, one per check": {5: {0, 1, 3}, 6: {1, 2, 4}, 7: {0, 1, 2}, 8: {1, 3, 4}},
    "2a: a{0123} G1,G3 | b{1234} G2,G4": {5: {0, 1, 2, 3}, 6: {1, 2, 3, 4}},
    "2b: a{0134} G1,G4 | b{0124} G2,G3": {5: {0, 1, 3, 4}, 6: {0, 1, 2, 4}},
    "user LEFT": {5: {0, 1, 2, 3}, 6: {2, 4}},
    "user RIGHT": {5: {0, 1, 2}, 6: {0, 3}, 7: {1, 2, 4}},
}


def graph(anc):
    G = nx.Graph()
    G.add_edge(2, 3)
    for a, nb in anc.items():
        G.add_edges_from((a, q) for q in nb)
    return G


def chromatic(G):
    for k in range(1, 6):
        for col in itertools.product(range(k), repeat=G.number_of_nodes()):
            c = dict(zip(G.nodes, col))
            if all(c[u] != c[v] for u, v in G.edges):
                return k


def embed(G, size=5, diag=True):
    nodes = list(nx.bfs_tree(G, max(G.nodes, key=G.degree)))
    pts = [(x, y) for x in range(size) for y in range(size)]
    steps = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0) and (diag or dx * dy == 0)]
    pos, used, sols = {}, set(), []

    def ok_edge(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        return (dx, dy) in steps

    def crossing(e, placed_edges):
        (p, q) = e
        if abs(p[0] - q[0]) == 1 and abs(p[1] - q[1]) == 1:
            other = ((p[0], q[1]), (q[0], p[1]))
            return other in placed_edges or other[::-1] in placed_edges
        return False

    def rec(i, edges):
        if i == len(nodes):
            sols.append(dict(pos))
            return True
        v = nodes[i]
        nb_placed = [u for u in G[v] if u in pos]
        cands = pts if not pos else {(pos[u][0] + dx, pos[u][1] + dy) for u in nb_placed for dx, dy in steps}
        for p in sorted(cands):
            if p in used or not (0 <= p[0] < size and 0 <= p[1] < size):
                continue
            if not all(ok_edge(pos[u], p) for u in nb_placed):
                continue
            new = [(pos[u], p) for u in nb_placed]
            es = set(edges)
            bad = False
            for e in new:
                if crossing(e, es):
                    bad = True
                    break
                es.add(e)
            if bad:
                continue
            pos[v] = p
            used.add(p)
            if rec(i + 1, es):
                return True
            del pos[v]
            used.discard(p)
        return False

    rec(0, set())
    return sols[0] if sols else None


for name, anc in LAYOUTS.items():
    G = graph(anc)
    sq = embed(G, diag=False)
    dg = embed(G, diag=True)
    tri = sum(nx.triangles(G).values()) // 3
    print(f"{name}: frequency groups {chromatic(G)}, triangles {tri}, "
          f"square grid only: {'yes' if sq else 'no'}, grid + diagonals, no crossings: "
          f"{'yes ' + str(dg) if dg else 'no'}", flush=True)
