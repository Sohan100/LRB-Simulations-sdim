"""Co-design: square-lattice layout + 4-gate native encoder + two-ancilla check rounds.

For every data tree T that hosts a 4-gate encoder of |+_L> (encoder_trees.py,
clifford_tree_search.py) the data couplers are D = T + {q2-q3} (q2-q3 carries the
logical gates). D must sit on the square lattice with unit-length edges, which
needs D bipartite (its one cycle, if any, a unit square). Two ancillas sit on empty
lattice sites; each couples to its lattice-neighbour data qutrits. Inside check
rounds, data-data gates may use only couplers on pairs without a weight-2 logical
(unsafe: 02, 03, 24, 34); other couplers are encoder-only.

Per check and ancilla, a uniform-cost search over reads of touched qutrits and
LIFO push/pop of safe data CNOTs ("difference trick", Heisenberg accumulation, as
in ../encoder_layout_search.py) gives the minimum two-qutrit gate count. Each
ancilla measures two checks (one stabilizer at a time per ancilla). Layouts are
ranked by total two-qutrit gates per round; the best are built into full rounds
and fault-checked by codesign_rounds.py.
"""
import heapq
import itertools
import sys
from functools import lru_cache

D = 3
CHECKS = {"G1": ("Z", (1, 2, 0, 2, 0)), "G2": ("Z", (0, 1, 1, 0, 2)),
          "G3": ("X", (1, 1, 2, 0, 0)), "G4": ("X", (0, 2, 0, 1, 2))}
UNSAFE = {frozenset(p) for p in [(0, 2), (0, 3), (2, 4), (3, 4)]}
SYMS = [lambda x, y: (x, y), lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (-x, -y),
        lambda x, y: (y, x), lambda x, y: (-y, x), lambda x, y: (y, -x), lambda x, y: (-y, -x)]


def mat_id():
    return tuple(tuple(int(i == j) for j in range(5)) for i in range(5))


def conj(kind, gate):
    name, c, t = gate
    s = 1 if name == "CNOT" else -1
    M = [list(r) for r in mat_id()]
    if kind == "Z":
        M[c][t] = s % D
    else:
        M[t][c] = (-s) % D
    return tuple(map(tuple, M))


def mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(5)) % D for j in range(5)) for i in range(5))


def inverse(g):
    return ("CNOT_INV" if g[0] == "CNOT" else "CNOT", g[1], g[2])


@lru_cache(maxsize=None)
def protocols(kind, target, touch, safe, max_pushes=4, max_cost=11, keep=40):
    """Minimal protocols (cost, list of op tuples). safe: frozenset of frozenset pairs."""
    gates = [(n, c, t) for p in sorted(map(tuple, map(sorted, safe))) for c, t in (p, p[::-1])
             for n in ("CNOT", "CNOT_INV")]
    start = ((), (0,) * 5)
    best, sols = None, []
    heap = [(0, 0, start, ())]
    seen = {start: 0}
    tie = itertools.count(1)
    Mcache = {(): mat_id()}
    while heap:
        cost, _, (stack, acc), ops = heapq.heappop(heap)
        if best is not None and cost > best:
            break
        if not stack and acc == target and ops:
            best = cost
            if len(sols) < keep:
                sols.append(ops)
            continue
        if seen.get((stack, acc), 99) < cost:
            continue
        M = Mcache[stack]
        last = ops[-1] if ops else None
        nxt = []
        for q in touch:
            if last is not None and last[0] == "read" and last[1] == q:
                continue
            for e in (1, 2):
                acc2 = tuple((acc[i] + M[i][q] * e) % D for i in range(5))
                nxt.append(((stack, acc2), ops + (("read", q, e),)))
        if len(stack) < max_pushes:
            for g in gates:
                if stack and g == inverse(stack[-1]):
                    continue
                st = stack + (g,)
                if st not in Mcache:
                    Mcache[st] = mul(M, conj(kind, g))
                nxt.append(((st, acc), ops + (("push",) + g,)))
        if stack:
            nxt.append(((stack[:-1], acc), ops + (("pop",) + inverse(stack[-1]),)))
        c2 = cost + 1
        if c2 > max_cost:
            continue
        for st, o2 in nxt:
            done = (not st[0]) and st[1] == target
            if seen.get(st, 99) <= c2 and not done:
                continue
            seen[st] = min(seen.get(st, 99), c2)
            heapq.heappush(heap, (c2, next(tie), st, o2))
    return (best, sols) if best is not None else (None, [])


def to_ops(kind, anc, proto, tag):
    ops = [("g", "H", anc)] if kind == "X" else []
    for step in proto:
        if step[0] == "read":
            _, q, e = step
            name = "CNOT" if e == 1 else "CNOT_INV"
            ops.append(("g", name, q, anc) if kind == "Z" else ("g", name, anc, q))
        else:
            ops.append(("g", step[1], step[2], step[3]))
    if kind == "X":
        ops.append(("g", "H_INV", anc))
    return ops + [("m", anc, tag), ("r", anc)]


def embeddings(edges):
    """All unit-edge placements of the 5 data qutrits on Z^2, up to lattice symmetry."""
    adj = {q: set() for q in range(5)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    order, seen = [], set()
    root = max(range(5), key=lambda q: len(adj[q]))
    todo = [root]
    while todo:
        u = todo.pop(0)
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        todo += sorted(adj[u] - seen)
    out, canon = [], set()

    def rec(i, pos):
        if i == len(order):
            ok = all(abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1 for a, b in edges)
            if not ok:
                return
            forms = []
            for S in SYMS:
                P = {q: S(*pos[q]) for q in pos}
                mx = min(p[0] for p in P.values())
                my = min(p[1] for p in P.values())
                forms.append(tuple(sorted((q, (P[q][0] - mx, P[q][1] - my)) for q in P)))
            c = min(forms)
            if c not in canon:
                canon.add(c)
                out.append(dict(c))
            return
        u = order[i]
        placed_nb = [v for v in adj[u] if v in pos]
        if not placed_nb:
            cands = [(0, 0)]
        else:
            v = placed_nb[0]
            cands = [(pos[v][0] + dx, pos[v][1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        for c in cands:
            if c in pos.values():
                continue
            pos[u] = c
            rec(i + 1, pos)
            del pos[u]

    rec(0, {})
    return out


def nb(p):
    return [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]


def layouts_for(tree_edges):
    Dedges = set(map(frozenset, tree_edges)) | {frozenset((2, 3))}
    Dl = [tuple(sorted(e)) for e in Dedges]
    for pos in embeddings(Dl):
        occ = {p: q for q, p in pos.items()}
        # data-data couplers: D edges, plus extra safe lattice-adjacent pairs (optional, safe only)
        adjacent = {frozenset((a, b)) for a in range(5) for b in range(a + 1, 5)
                    if abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1}
        extra = {e for e in adjacent - Dedges if e not in UNSAFE}
        safe = frozenset(e for e in (Dedges | extra) if e not in UNSAFE)
        sites = {}
        for q, p in pos.items():
            for s in nb(p):
                if s not in occ:
                    sites.setdefault(s, set()).add(q)
        yield pos, Dedges, extra, safe, {s: tuple(sorted(t)) for s, t in sites.items()}


def best_assignments(sites, safe):
    """Over ancilla-site pairs and 2+2 check splits, minimal total protocol cost."""
    res = []
    for (s1, t1), (s2, t2) in itertools.combinations(sorted(sites.items()), 2):
        cost = {}
        for chk, (kind, vec) in CHECKS.items():
            for a, t in ((0, t1), (1, t2)):
                cost[(chk, a)] = protocols(kind, vec, t, safe)[0]
        for pair in itertools.combinations(CHECKS, 2):
            other = [c for c in CHECKS if c not in pair]
            cs = [cost[(c, 0)] for c in pair] + [cost[(c, 1)] for c in other]
            if any(x is None for x in cs):
                continue
            res.append((sum(cs), (s1, t1, pair), (s2, t2, tuple(other)), cs))
    return res


def _tree_rows(T):
    lbl = lambda X: " ".join(sorted(f"{min(e)}{max(e)}" for e in X))
    rows, n_emb = [], 0
    for pos, Dedges, extra, safe, sites in layouts_for([tuple(e) for e in T]):
        n_emb += 1
        for total, A, B, cs in best_assignments(sites, safe):
            rows.append((total, lbl(T), pos, lbl(extra), lbl(safe), A, B, cs))
    return lbl(T), n_emb, rows


if __name__ == "__main__":
    from multiprocessing import Pool
    import encoder_trees as ET
    extra_trees = sys.argv[1:]  # star trees confirmed by clifford_tree_search_fast, e.g. "02,12,23,24"
    allowed = {frozenset(frozenset(g[1:]) for g in p) for S, p in ET.cnot_encoders()[4]}
    for s in extra_trees:
        allowed.add(frozenset(frozenset((int(e[0]), int(e[1]))) for e in s.split(",")))
    rows = []
    with Pool(min(12, len(allowed))) as pool:
        for name, n_emb, r in pool.imap_unordered(_tree_rows, sorted(allowed, key=lambda T: sorted(map(sorted, T)))):
            print(f"tree {name:12s}: {n_emb} lattice embeddings of T+23, {len(r)} candidates", flush=True)
            rows += r
    rows.sort(key=lambda r: r[0])
    print(f"\n{len(rows)} (layout, ancilla pair, split) candidates; best totals (per-check cap 11):", flush=True)
    seen = set()
    for r in rows:
        k = (r[1], tuple(sorted(r[2].items())), r[5][0], r[6][0])
        if k in seen:
            continue
        seen.add(k)
        total, T, pos, extra, safe, A, B, cs = r
        print(f"  {total:2d} gates  tree {T:12s} data {pos}  extra {extra or '-'}  safe {safe}  "
              f"anc1 {A[0]} touches {A[1]} reads {A[2]}  anc2 {B[0]} touches {B[1]} reads {B[2]}  per-check {cs}")
        if len(seen) >= 40:
            break
