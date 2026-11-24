"""Fewest-two-qudit-gate encoders of |+_L> for the [[5,1,2]]_d code on a layout.

Model (exact within its class). |+_L> is a CSS state: the uniform
superposition over V = span(G3, G4, Xbar) in F_d^5 (equivalently the kernel of
G1, G2). Every qudit starts in |0> or, after a free one-qudit Fourier gate, in
|+>; a CNOT / CNOT_INV c->t maps the support subspace W by x_t += +-x_c. So a
circuit of CNOT-type gates from a product state is a walk on subspaces of
F_d^n, from a coordinate subspace span{e_i : i in S} to a target
V (+) {helpers in |0> or |+>}. Helpers (ancillas) may be used as scratch and
must end unentangled, in |0> or |+> (the check rounds reset them anyway).
A breadth-first search over all subspaces of the target dimension gives the
exact minimum number of two-qudit gates in this class; every two-qudit gate
sits on a coupler of the layout.

Lower bound for ANY circuit (not only CNOT circuits): |+_L> has no product
cut (checked here), so the two-qudit gates must connect all five data qudits,
through helpers if needed: at least (5 + helpers used - 1) gates, the size of
a Steiner tree of the data in the coupling graph.

Usage: encoder_search.py LAYOUT D [--helpers=a,b,...] [--all-optimal]
LAYOUT: a1 | one-ancilla | data-complete
"""
import itertools
import sys
from collections import deque

import numpy as np

CODE_X = [(1, 1, -1, 0, 0), (0, -1, 0, 1, -1), (1, 0, 0, 1, 0)]   # G3, G4, Xbar
CODE_Z = [(1, -1, 0, -1, 0), (0, 1, 1, 0, -1)]                  # G1, G2

LAYOUTS = {
    # A1: q0(0,0) q1(1,0) E5(2,0) q3(0,1) q2(1,1) q4(2,1) N6(1,2)
    "a1": {"n": 7, "couplers": [(0, 1), (0, 3), (1, 2), (2, 3), (2, 4), (1, 5), (4, 5), (2, 6)]},
    # Paper one-ancilla layout: q1 centre, q3 TL, q4 TR, q0 BL, q2 BR; ancilla
    # sites 5 (G1: q0 q1 q3, left), 6 (G2: q1 q2 q4, right), 7 (G3: q0 q1 q2,
    # bottom), 8 (G4: q1 q3 q4, top); long-range coupler q2-q3.
    "one-ancilla": {"n": 9, "couplers": [(5, 0), (5, 1), (5, 3), (6, 1), (6, 2), (6, 4),
                                         (7, 0), (7, 1), (7, 2), (8, 1), (8, 3), (8, 4), (2, 3)]},
    # All data pairs, no helpers (to classify which data trees allow 4 gates).
    "data-complete": {"n": 5, "couplers": list(itertools.combinations(range(5), 2))},
}


def rref(rows, d):
    M = [list(r) for r in rows]
    n = len(M[0]) if M else 0
    piv_row = 0
    for col in range(n):
        p = next((i for i in range(piv_row, len(M)) if M[i][col] % d), None)
        if p is None:
            continue
        M[piv_row], M[p] = M[p], M[piv_row]
        inv = pow(M[piv_row][col], -1, d)
        M[piv_row] = [(v * inv) % d for v in M[piv_row]]
        for i in range(len(M)):
            if i != piv_row and M[i][col] % d:
                f = M[i][col]
                M[i] = [(a - f * b) % d for a, b in zip(M[i], M[piv_row])]
        piv_row += 1
        if piv_row == len(M):
            break
    return tuple(tuple(r) for r in M[:piv_row])


def moves(couplers, d):
    signs = (1,) if d == 2 else (1, -1)
    return [(c, t, s) for a, b in couplers for c, t in ((a, b), (b, a)) for s in signs]


def apply(W, mv, d):
    c, t, s = mv
    rows = []
    for r in W:
        r = list(r)
        r[t] = (r[t] + s * r[c]) % d
        rows.append(r)
    return rref(rows, d)


def data_subspace(n):
    return [tuple(list(v) + [0] * (n - 5)) for v in CODE_X]


def targets(n, d, helpers, k_plus):
    """Target subspaces: data in |+_L>, k_plus of the helpers in |+>, rest |0>."""
    out = []
    for plus in itertools.combinations(helpers, k_plus):
        rows = data_subspace(n) + [tuple(1 if i == h else 0 for i in range(n)) for h in plus]
        out.append(rref(rows, d))
    return out


def starts(n, d, active, dim):
    """Coordinate subspaces of dimension dim on the active qudits."""
    return {rref([tuple(1 if i == q else 0 for i in range(n)) for q in S], d): S
            for S in itertools.combinations(active, dim)}


def bfs(sources, mvs, d, stop_at=None, max_depth=None):
    """Multi-source BFS. With stop_at (a set of states), finish the level on
    which the first of them is reached and stop; max_depth caps the levels."""
    dist = {s: 0 for s in sources}
    frontier = list(sources)
    level = 0
    hit = stop_at is not None and any(s in stop_at for s in sources)
    while frontier and not hit and (max_depth is None or level < max_depth):
        level += 1
        nxt = []
        for W in frontier:
            for mv in mvs:
                V = apply(W, mv, d)
                if V not in dist:
                    dist[V] = level
                    nxt.append(V)
                    if stop_at is not None and V in stop_at:
                        hit = True
        frontier = nxt
    return dist


def optimal_paths(W, dist, mvs, d, limit=20000):
    """All shortest move sequences from W down to a target (dist 0)."""
    out = []

    def rec(X, path):
        if len(out) >= limit:
            return
        if dist[X] == 0:
            out.append(list(path))
            return
        for mv in mvs:
            Y = apply(X, mv, d)
            if dist.get(Y, 1e9) == dist[X] - 1:
                path.append(mv)
                rec(Y, path)
                path.pop()
    rec(W, [])
    return out


def layers(path):
    """ASAP two-qudit-gate depth."""
    free = {}
    depth = 0
    for c, t, _ in path:
        lv = max(free.get(c, 0), free.get(t, 0)) + 1
        free[c] = free[t] = lv
        depth = max(depth, lv)
    return depth


def no_product_cut(d):
    """|+_L> has no product cut: every bipartition has positive entanglement."""
    n = 5
    stab = [("X", v) for v in CODE_X] + [("Z", v) for v in CODE_Z]
    full = []
    for kind, v in stab:
        full.append(tuple((x % d) for x in v) + (0,) * 5 if kind == "X" else (0,) * 5 + tuple(x % d for x in v))
    worst = None
    for k in range(1, 3):
        for A in itertools.combinations(range(n), k):
            # entropy S(A) = |A| - dim(stabilizers supported on A)
            B = [i for i in range(n) if i not in A]
            # subgroup supported on A: combinations vanishing on B (X and Z parts)
            M = np.array(full) % d
            cols = [i for i in B] + [5 + i for i in B]
            sub = M[:, cols]
            # kernel of sub^T over F_d -> number of independent combos
            rank = len(rref([tuple(r) for r in sub.T.tolist()], d)) if sub.size else 0
            dimA = len(full) - rank
            S = k - dimA
            worst = S if worst is None else min(worst, S)
    return worst


def run(layout, d, helpers=None, all_optimal=False, verbose=True):
    L = LAYOUTS[layout]
    n = L["n"]
    allh = [q for q in range(5, n)]
    helpers = allh if helpers is None else helpers
    active = list(range(5)) + list(helpers)
    cps = [(a, b) for a, b in L["couplers"] if a in active and b in active]
    mvs = moves(cps, d)
    best = None
    for k_plus in range(0, len(helpers) + 1):
        dim = 3 + k_plus
        T = targets(n, d, helpers, k_plus)
        # inactive helpers stay |0>: they are never touched (no moves on them)
        S = starts(n, d, active, dim)
        cap = None if best is None else best[0]
        dist = bfs(T, mvs, d, stop_at=set(S), max_depth=cap)
        for W, Sset in S.items():
            if W in dist:
                D = dist[W]
                if best is None or D < best[0]:
                    best = (D, k_plus, Sset, W, dist)
        if verbose:
            reach = min((dist[W] for W in S if W in dist), default=None)
            print(f"  {layout} d={d} helpers={helpers} helpers ending |+>: {k_plus}: "
                  f"{len(dist)} subspaces reached, min gates {reach}", flush=True)
    if best is None:
        return None
    D, k_plus, Sset, W, dist = best
    paths = optimal_paths(W, dist, mvs, d) if all_optimal else optimal_paths(W, dist, mvs, d, limit=1)
    # Also scan every start with the optimal count for the shallowest circuit.
    if all_optimal:
        S = starts(n, d, active, 3 + k_plus)
        cands = []
        for W2, S2 in S.items():
            if dist.get(W2) == D:
                for p in optimal_paths(W2, dist, mvs, d):
                    cands.append((layers(p), S2, p))
        cands.sort(key=lambda c: c[0])
        return D, k_plus, cands
    return D, k_plus, [(layers(paths[0]), Sset, paths[0])]


if __name__ == "__main__":
    layout, d = sys.argv[1], int(sys.argv[2])
    hel = next((a.split("=")[1] for a in sys.argv if a.startswith("--helpers=")), None)
    hel = None if hel is None else [int(x) for x in hel.split(",") if x != ""]
    print(f"|+_L> min entanglement over cuts (d={d}): {no_product_cut(d)} (>0: no product cut)")
    res = run(layout, d, hel, all_optimal="--all-optimal" in sys.argv)
    if res is None:
        print("no encoder found")
        sys.exit()
    D, k_plus, cands = res
    print(f"RESULT {layout} d={d} helpers={hel}: min two-qudit gates {D} "
          f"(helpers ending |+>: {k_plus}); {len(cands)} optimal circuits listed")
    for depth, S, p in cands[:5]:
        print(f"  depth {depth}: |+> on {S}, gates " +
              ", ".join(f"{'CNOT' if s == 1 else 'CNOT_INV'} {c}->{t}" for c, t, s in p))
