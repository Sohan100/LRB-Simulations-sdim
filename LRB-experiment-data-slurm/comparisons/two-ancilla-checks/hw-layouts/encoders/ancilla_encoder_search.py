"""Fewest-gate encoder of |+_L> on a layout whose data qutrits are not coupled as a tree.

CSS (CNOT-network) encoders with helper ancillas: every qutrit starts in |0> or
|+> and the circuit is CNOT / CNOT_INV on couplers. The state is the uniform
superposition over a subspace W of F_3^n; the circuit is an encoder iff it maps a
coordinate subspace span(e_S) onto V (data) + span(e_a : ancillas a left in |+>),
ancillas left in |0> otherwise, so the ancillas end disentangled. Breadth-first
search backwards from every target lists all minimal circuits.

Lower bound for ANY circuit: the gates must connect all five data qutrits (no
data split is unentangled), so #gates >= edges of a minimum Steiner tree of the
data in the coupler graph.

Usage: ancilla_encoder_search.py one-ancilla [max_depth]
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
import encoder_trees as ET  # noqa: E402

D = 3
LAYOUTS = {
    # paper (Overleaf) one-ancilla layout: q1 centre, a_j next to the data of G_j,
    # long-range q2-q3; wires a1..a4 = 5..8
    "one-ancilla": {"n": 9, "anc": (5, 6, 7, 8),
                    "couplers": [(5, 0), (5, 1), (5, 3), (6, 1), (6, 2), (6, 4), (7, 0), (7, 1), (7, 2),
                                 (8, 1), (8, 3), (8, 4), (2, 3)]},
    # paper's conventional grid (panels a, c): q2 centre, corners q0 q1 q3 q4, sites
    # N=5 (q0,q1,q2), W=6 (q0,q2,q3), E=7 (q1,q2,q4), S=8 (q2,q3,q4); on Garnet
    # q2=QB9, q0=QB3, q1=QB5, q3=QB13, q4=QB15, N=QB4, W=QB8, E=QB10, S=QB14; no data-data coupler
    "conv-grid": {"n": 9, "anc": (5, 6, 7, 8),
                  "couplers": [(5, 0), (5, 1), (5, 2), (6, 0), (6, 2), (6, 3), (7, 1), (7, 2), (7, 4),
                               (8, 2), (8, 3), (8, 4)]},
}


def rref_t(rows, n):
    M = [list(r) for r in rows]
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, len(M)) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = 1 if M[r][c] == 1 else 2
        M[r] = [(v * inv) % D for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % D for a, b in zip(M[i], M[r])]
        r += 1
    return tuple(tuple(x) for x in M[:r])


def is_coordinate(W):
    return all(sum(1 for v in row if v) == 1 for row in W)


def steiner_lower_bound(n, couplers, terminals):
    """Minimum number of edges of a tree connecting all terminals (brute force)."""
    nodes = range(n)
    others = [v for v in nodes if v not in terminals]
    E = [tuple(e) for e in couplers]
    best = None
    for k in range(len(others) + 1):
        for extra in itertools.combinations(others, k):
            keep = set(terminals) | set(extra)
            sub = [e for e in E if e[0] in keep and e[1] in keep]
            parent = {v: v for v in keep}
            def f(x):
                while parent[x] != x:
                    x = parent[x]
                return x
            comps = len(keep)
            for a, b in sub:
                ra, rb = f(a), f(b)
                if ra != rb:
                    parent[ra] = rb
                    comps -= 1
            if comps == 1:
                cost = len(keep) - 1
                best = cost if best is None else min(best, cost)
        if best is not None:
            return best
    return best


def search(layout, max_depth):
    n, anc, couplers = layout["n"], layout["anc"], layout["couplers"]
    gates = [(nm, c, t) for a, b in couplers for c, t in ((a, b), (b, a)) for nm in ("CNOT", "CNOT_INV")]
    Vrows = [list(v) + [0] * (n - 5) for v in ET.V.tolist()]
    targets = {}
    for k in range(len(anc) + 1):
        for plus in itertools.combinations(anc, k):
            rows = Vrows + [[int(i == a) for i in range(n)] for a in plus]
            targets[rref_t(rows, n)] = plus
    frontier = {W: (targets[W], ()) for W in targets}
    seen = set(frontier)
    for depth in range(1, max_depth + 1):
        nxt = {}
        hits = []
        for W, (plus, path) in frontier.items():
            for g in gates:
                nm, c, t = g
                s = 1 if nm == "CNOT" else -1
                # backward step: undo g, i.e. apply its inverse to the subspace basis
                rows = [list(r) for r in W]
                for r in rows:
                    r[t] = (r[t] - s * r[c]) % D
                W2 = rref_t(rows, n)
                if W2 in seen:
                    continue
                seen.add(W2)
                p2 = (g,) + path
                nxt[W2] = (plus, p2)
                if is_coordinate(W2):
                    hits.append((W2, plus, p2))
        print(f"  depth {depth}: {len(nxt)} new subspaces, {len(hits)} encoders", flush=True)
        if hits:
            return depth, hits
        frontier = nxt
    return None, []


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "one-ancilla"
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    L = LAYOUTS[name]
    lb = steiner_lower_bound(L["n"], L["couplers"], list(range(5)))
    print(f"{name}: couplers {L['couplers']}")
    print(f"lower bound for any encoder (Steiner tree of the data): {lb} two-qutrit gates")
    depth, hits = search(L, max_depth)
    if depth is None:
        print(f"no CNOT-network encoder with <= {max_depth} gates")
    else:
        print(f"minimal CNOT-network encoder: {depth} two-qutrit gates ({len(hits)} found)")
        for W, plus, path in hits[:10]:
            start = sorted(next(i for i, v in enumerate(row) if v) for row in W)
            print(f"  |+> on {start}, ancillas left in |+>: {list(plus)}; gates {list(path)}")
