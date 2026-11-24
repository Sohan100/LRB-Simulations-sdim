"""H1 (rewrite) - interaction graphs of the [[5,1,2]]_3 LRB circuits.

Questions answered (all couplings are parsed from src/lrb/code_definitions.py,
not typed by hand):

G1  Which coupling graphs are bipartite (embeddable in a square lattice with
    every coupling nearest-neighbour)?  Shortest odd cycle when not.
G2  Is there ANY choice of stabilizer generators (one ancilla per generator,
    direct ancilla-data couplings) for which checks + fold coupling (2,3) is
    bipartite?  Exhaustive over all 4-element bases of the stabilizer group.
G3  Repo split-ancilla variant: which single-relay edge sets restore
    bipartiteness together with the fold; is the repo's choice one of them?
G4  Embeddings: checks-only graph into 3x3 grid and into IQM Garnet; the
    data-only Stage-A graph (encoder + fold) into Garnet; the proposed
    11-qutrit relay layout into Garnet.
G5  Exhaustive minimum number of relay hops per check round on an infinite
    square lattice, for fold distance 1 (native CZ(2,3)) and 2.
G6  Planarity / degrees of checks + fold (custom chip).
G7  Average number of logical H and P tokens per catalogue Clifford.

Run: module load python/3.11; OMP_NUM_THREADS=1 python sc_graphs.py
Runtime: about 1-2 minutes on one core.
"""
from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
from lrb import code_definitions as defns  # noqa: E402

FQ = defns.FoldedQutritDetectionCode
SPL = defns.FoldedQutritSplitAncillaUniformDetectionCode


def couplings(circ):
    out = []
    for op in circ.operations:
        if op.target_index is not None:
            out.append((int(op.qudit_index), int(op.target_index)))
    return out


def und(edges):
    return sorted({tuple(sorted(e)) for e in edges})


CHECK = und(couplings(FQ.z_ALT_stabilizer_measurement_ancillae_circuit())
            + couplings(FQ.x_stabilizer_measurement_ancillae_circuit()))
SPLIT = und(couplings(SPL.z_boundary_split_stabilizer_measurement_ancillae_circuit())
            + couplings(SPL.z_middle_stabilizer_measurement_ancillae_circuit())
            + couplings(SPL.x_middle_stabilizer_measurement_ancillae_circuit())
            + couplings(SPL.x_boundary_split_stabilizer_measurement_ancillae_circuit()))
ENC = und(couplings(FQ.logical_plus_initial_state()))
FOLD_H = und(couplings(FQ.single_qudit_logical_gate_circuit("H")))
FOLD_P = und(couplings(FQ.single_qudit_logical_gate_circuit("P")))
FOLD = (2, 3)
SUPP = {5: (0, 1, 3), 6: (1, 2, 4), 7: (0, 1, 2), 8: (1, 3, 4)}

# IQM Garnet coupling map (30 pairs) transcribed from qiskit-on-iqm
# src/iqm/qiskit_iqm/fake_backends/fake_garnet.py (fetched 2026-10-28).
GARNET = [(1, 2), (1, 4), (2, 5), (3, 4), (8, 3), (4, 5), (9, 4), (5, 6),
          (10, 5), (6, 7), (11, 6), (12, 7), (8, 9), (8, 13), (9, 10),
          (9, 14), (10, 11), (10, 15), (11, 12), (16, 11), (17, 12),
          (13, 14), (14, 15), (18, 14), (16, 15), (19, 15), (16, 17),
          (16, 20), (18, 19), (19, 20)]


def shortest_odd_cycle(edges):
    g = nx.Graph(edges)
    if nx.is_bipartite(g):
        return None
    best = None
    for s in g.nodes:
        # BFS layers; an edge inside one layer closes an odd cycle
        dist = {s: 0}
        par = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in g[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                elif dist[v] == dist[u] and (best is None or 2 * dist[u] + 1 < len(best)):
                    pu, pv = [u], [v]
                    while par[pu[-1]] is not None:
                        pu.append(par[pu[-1]])
                    while par[pv[-1]] is not None:
                        pv.append(par[pv[-1]])
                    if pu[-1] == pv[-1] and not (set(pu[:-1]) & set(pv[:-1])):
                        cyc = pu[::-1] + pv[:-1]
                        best = cyc
    return best


def count_monos(pattern_edges, host_edges, limit=None):
    G = nx.Graph(host_edges)
    P = nx.Graph(pattern_edges)
    gm = isomorphism.GraphMatcher(G, P)
    n = 0
    ex = None
    for m in gm.subgraph_monomorphisms_iter():
        n += 1
        if ex is None:
            ex = {v: k for k, v in m.items()}
        if limit and n >= limit:
            break
    return n, ex


def grid(n):
    return [((r, c), (r + 1, c)) for r in range(n - 1) for c in range(n)] + \
           [((r, c), (r, c + 1)) for r in range(n) for c in range(n - 1)]


# ------------------------------------------------------------------ G2
def stabilizer_vectors():
    """Symplectic vectors (x[0..4], z[0..4]) of the four generators."""
    v = {
        "SZ1": [0] * 5 + [1, 2, 0, 2, 0],
        "SZ2": [0] * 5 + [0, 1, 1, 0, 2],
        "SX1": [1, 1, 2, 0, 0] + [0] * 5,
        "SX2": [0, 2, 0, 1, 2] + [0] * 5,
    }
    return np.array(list(v.values()))


def rank_mod3(rows):
    m = np.array(rows, dtype=int) % 3
    r = 0
    rows_n, cols = m.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows_n):
            if m[i, c] % 3:
                piv = i
                break
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        inv = 1 if m[r, c] == 1 else 2
        m[r] = (m[r] * inv) % 3
        for i in range(rows_n):
            if i != r and m[i, c]:
                m[i] = (m[i] - m[i, c] * m[r]) % 3
        r += 1
    return r


def g2_all_bases():
    gens = stabilizer_vectors()
    lines = []
    seen = set()
    for coef in itertools.product(range(3), repeat=4):
        if not any(coef):
            continue
        v = tuple((np.array(coef) @ gens) % 3)
        key = min(v, tuple((2 * np.array(v)) % 3))
        if key in seen:
            continue
        seen.add(key)
        lines.append(np.array(v))
    supports = [tuple(q for q in range(5) if v[q] or v[5 + q]) for v in lines]
    n_bases = n_bip = 0
    for combo in itertools.combinations(range(len(lines)), 4):
        if rank_mod3([lines[i] for i in combo]) < 4:
            continue
        n_bases += 1
        edges = [FOLD]
        for k, i in enumerate(combo):
            edges += [(100 + k, q) for q in supports[i]]
        if nx.is_bipartite(nx.Graph(edges)):
            n_bip += 1
    # key lemma: every element acting on exactly one of {2,3} also acts on 1,
    # and no element avoids both 2 and 3 entirely except none
    one_of = [s for s in supports if (2 in s) != (3 in s)]
    lemma = all(1 in s for s in one_of)
    neither = [s for s in supports if 2 not in s and 3 not in s]
    return len(lines), n_bases, n_bip, (len(one_of), lemma, sorted(set(one_of))), neither


# ------------------------------------------------------------------ G3
def two_col_parity(edges, parity):
    adj = {}
    for e in edges:
        p = parity.get(e, 1)
        adj.setdefault(e[0], []).append((e[1], p))
        adj.setdefault(e[1], []).append((e[0], p))
    col = {}
    for s in adj:
        if s in col:
            continue
        col[s] = 0
        st = [s]
        while st:
            u = st.pop()
            for v, p in adj[u]:
                w = col[u] ^ p
                if v not in col:
                    col[v] = w
                    st.append(v)
                elif col[v] != w:
                    return False
    return True


# ------------------------------------------------------------------ G5
def bfs_avoid(src, blocked, lo, hi):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        if u != src and u in blocked:
            continue
        r, c = u
        for v in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if lo <= v[0] <= hi and lo <= v[1] <= hi and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def exact_cost(dp, lo=-10, hi=11):
    blocked = set(dp.values())
    d = {q: bfs_avoid(p, blocked, lo, hi) for q, p in dp.items()}
    sites = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)
             if (r, c) not in blocked]
    cost = np.full((4, len(sites)), 10 ** 6)
    for k, a in enumerate(sorted(SUPP)):
        for j, s in enumerate(sites):
            tot = 0
            for q in SUPP[a]:
                L = d[q].get(s)
                if L is None:
                    tot = 10 ** 6
                    break
                tot += L - 1
            cost[k, j] = tot
    rows, cols = linear_sum_assignment(cost)
    return int(cost[rows, cols].sum()), {sorted(SUPP)[k]: sites[j] for k, j in zip(rows, cols)}


def lower_bound(dp):
    lb = 0
    for s in SUPP.values():
        xs = [dp[q][0] for q in s]
        ys = [dp[q][1] for q in s]
        lb += max(xs) - min(xs) + max(ys) - min(ys) - 3
    return lb


def min_hops(pos3, target, R=5):
    md = lambda u, v: abs(u[0] - v[0]) + abs(u[1] - v[1])
    p2 = (0, 0)
    near = [(r, c) for r in range(-R - 2, R + 3) for c in range(-R - 2, R + 3)
            if (r, c) not in (p2, pos3) and min(md((r, c), p2), md((r, c), pos3)) <= R]
    best = None
    n_eval = 0
    for q0, q1, q4 in itertools.permutations(near, 3):
        dp = {0: q0, 1: q1, 2: p2, 3: pos3, 4: q4}
        if lower_bound(dp) > target:
            continue
        if md(p2, pos3) == 2:
            mids = [m for m in ((0, 1), (1, 0), (0, -1), (-1, 0)) if md(m, pos3) == 1]
            if all(m in (q0, q1, q4) for m in mids):
                continue
        n_eval += 1
        c, anc = exact_cost(dp)
        if best is None or c < best[0]:
            best = (c, dp, anc)
    return best, n_eval


# Proposed layout (native fold).  Grid coordinates (row, col).
LAYOUT = {"D0": (2, 3), "A5": (3, 3), "R1": (3, 4), "R2": (3, 5),
          "A7": (4, 3), "D2": (4, 4), "D3": (4, 5),
          "D1": (5, 3), "A6": (5, 4), "A8": (5, 5), "D4": (5, 6)}
LAYOUT_COUPLERS = [("A5", "D0"), ("A6", "D1"), ("A6", "D2"), ("A7", "D1"),
                   ("A7", "D2"), ("A8", "D3"), ("A8", "D4"), ("D2", "D3"),
                   ("A5", "A7"), ("A5", "R1"), ("R1", "R2"), ("R2", "D3"),
                   ("A6", "A8")]


def main():
    t0 = time.time()
    print("parsed check couplings:", CHECK)
    print("parsed split-variant couplings:", SPLIT)
    print("encoder couplings:", ENC, "; logical H:", FOLD_H, "; logical P:", FOLD_P)

    print("\n== G1 bipartiteness ==")
    graphs = {
        "checks": CHECK,
        "checks+fold": CHECK + [FOLD],
        "split": SPLIT,
        "split+fold": SPLIT + [FOLD],
        "stageA (encoder+fold)": ENC + [FOLD],
        "checks+fold+encoder": CHECK + [FOLD] + ENC,
    }
    for name, e in graphs.items():
        cyc = shortest_odd_cycle(e)
        print(f"  {name:24s} bipartite={cyc is None}"
              + ("" if cyc is None else f", shortest odd cycle {cyc} (length {len(cyc)})"))

    print("\n== G2 any generator choice? ==")
    nl, nb, nbip, t3, t2 = g2_all_bases()
    print(f"  {nl} one-dimensional subgroups (generator lines), {nb} bases;"
          f" bases whose ancilla graph + fold is bipartite: {nbip}")
    print(f"  lines acting on exactly one of data 2, 3: {t3[0]}; all of them act on data 1: "
          f"{t3[1]}; their supports: {t3[2]}")
    print(f"  lines acting on neither 2 nor 3: {t2}")

    print("\n== G3 relay sets restoring bipartiteness with the fold ==")
    for k in (1, 2):
        sols = []
        for S in itertools.combinations(CHECK, k):
            par = {e: (0 if e in S else 1) for e in CHECK}
            par[FOLD] = 1
            if two_col_parity(CHECK + [FOLD], par):
                sols.append(S)
        print(f"  single-relay sets of size {k}: {sols}")
    repo = [e for e in CHECK if e in ((3, 5), (4, 8))]
    par = {e: (0 if e in repo else 1) for e in CHECK}
    par_f = dict(par)
    par_f[FOLD] = 1
    print("  repo split choice", repo, "with fold parity-consistent:",
          two_col_parity(CHECK + [FOLD], par_f),
          "; without fold:", two_col_parity(CHECK, par))

    print("\n== G4 embeddings ==")
    n, ex = count_monos(CHECK, grid(3))
    print(f"  checks into 3x3 grid: {n} monomorphisms (dihedral group has 8); example {ex}")
    n, ex = count_monos(CHECK, GARNET)
    print(f"  checks into Garnet: {n}; example {ex}")
    n, ex = count_monos(ENC + [FOLD], GARNET)
    print(f"  Stage-A (encoder+fold) into Garnet: {n}; example {ex}")
    adj_ok = all(abs(LAYOUT[a][0] - LAYOUT[b][0]) + abs(LAYOUT[a][1] - LAYOUT[b][1]) == 1
                 for a, b in LAYOUT_COUPLERS)
    print(f"  proposed layout: {len(LAYOUT)} qutrits, {len(LAYOUT_COUPLERS)} couplers,"
          f" all grid-adjacent: {adj_ok}")
    n, ex = count_monos(LAYOUT_COUPLERS, GARNET)
    print(f"  proposed layout into Garnet: {n} monomorphisms; example "
          f"{dict(sorted((k, f'QB{v}') for k, v in ex.items())) if ex else None}")
    # check relay hop count of the proposed layout
    dp = {int(k[1]): LAYOUT[k] for k in LAYOUT if k[0] == "D"}
    blocked = set(dp.values())
    tot = 0
    for a, s in SUPP.items():
        for q in s:
            L = bfs_avoid(dp[q], blocked, -2, 12)[LAYOUT[f"A{a}"]]
            tot += L - 1
    print(f"  proposed layout: check relay hops per round = {tot}")

    print("\n== G5 minimum relay hops (exhaustive, R=5) ==")
    print("  bound: a data qutrit q in {0,1,4} farther than R from both 2 and 3"
          " forces >= 2(R-2) hops (q=0,4) or 4(R-2) (q=1); R=5 -> >= 6")
    for pos3, tgt, name in (((0, 1), 5, "fold native (distance 1)"),
                            ((0, 2), 4, "fold distance 2, straight"),
                            ((1, 1), 4, "fold distance 2, diagonal")):
        best, n_eval = min_hops(pos3, tgt)
        print(f"  {name}: placements with lower bound <= {tgt}: {n_eval}; "
              f"best exact cost {None if best is None else best[0]}"
              + ("" if best is None else f"; e.g. data {best[1]} ancillas {best[2]}"))
    print(f"  [{time.time() - t0:.0f}s]")

    print("\n== G6 custom chip ==")
    g = nx.Graph(CHECK + [FOLD])
    print("  checks+fold planar:", nx.check_planarity(g)[0], "; degrees:",
          dict(sorted(g.degree())))

    print("\n== G7 logical token statistics of the 216-element catalogue ==")
    cat = defns.single_qutrit_cliffords_exhaustive_strings
    lib = defns.QutritCliffordLibrary
    inv = lib.build_inverse_lookup()
    fw = [lib.qutrit_gate_seq_list(c) for c in cat]
    iv = [inv[c] for c in cat]
    def count(seqs, letter):
        return np.mean([sum(1 for t in s if t[0] == letter) for s in seqs])
    print(f"  {len(cat)} Cliffords; mean tokens per forward Clifford: "
          f"H {count(fw, 'H'):.3f}, P {count(fw, 'P'):.3f}, X {count(fw, 'X'):.3f},"
          f" Z {count(fw, 'Z'):.3f}; per inverse: H {count(iv, 'H'):.3f}, P {count(iv, 'P'):.3f}")
    print(f"  CZ(2,3) per LRB step (forward + inverse): {count(fw, 'P') + count(iv, 'P'):.3f};"
          f" SWAP(2,3) per step: {count(fw, 'H') + count(iv, 'H'):.3f}")
    par_ok = all((sum(1 for t in a if t[0] == 'H') - sum(1 for t in b if t[0] == 'H')) % 2 == 0
                 for a, b in zip(fw, iv))
    print(f"  forward and inverse H-token counts have equal parity for every Clifford: {par_ok}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
