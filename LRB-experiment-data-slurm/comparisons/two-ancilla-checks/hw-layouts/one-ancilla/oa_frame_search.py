"""One ancilla per stabilizer on a square lattice: exhaustive search over data frames.

Setting. 5 data qutrits and 4 ancillas; ancilla a_j reads only G_j, all four in one
round with a single readout layer (as in the paper's one-ancilla round). Couplers are
lattice-adjacent pairs only (no diagonals), so the coupling graph is bipartite, and
q2-q3 is a coupler (logical gates). No direct-read layout exists (odd cycle, see
../README.md), so some checks must be read in a different data frame.

Frame model. A data frame is a sequence F = (U_1, ..., U_L) of data-data CNOT^s gates
(s = +-1). The round applies U_1..U_L (push), reads, and undoes them (pop), so it
costs 2L data-data gates. Reading Z_q^e (or X_q^e) while the frame F is applied
measures F^dag Z_q^e F on the input state. Every prefix of F is also a frame, so a
check may be read partly in each prefix frame (the "difference trick" of the
two-ancilla layouts is the case L = 1 with reads in both frames). For each check we
enumerate every split G_j = sum_f F_f^dag B_f F_f over the timeline frames and keep,
for every touched data set T, the minimum number of reads sum_f |supp B_f|.

Layout. The data coupler set is E = {q2-q3} + the frame's pairs. A layout needs a
2-colouring of the data with every coupler in E bichromatic and every touched set
monochromatic (ancilla sites are the other colour), and an embedding: injective data
sites on Z^2 with every coupler in E at distance 1, plus 4 distinct empty sites,
site j adjacent to every qutrit of T_j.

Output: ranked candidates by total two-qutrit gates (reads + 2L), with the layout.
Usage: oa_frame_search.py 'SHAPES' MAXGATES   (SHAPES e.g. '1;2;1,1;2,1': window lengths)
"""
import itertools
import sys
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool

D = 3
CHECKS = {"G1": ("Z", (1, 2, 0, 2, 0)), "G2": ("Z", (0, 1, 1, 0, 2)),
          "G3": ("X", (1, 1, 2, 0, 0)), "G4": ("X", (0, 2, 0, 1, 2))}
NAMES = list(CHECKS)
PAIRS = [(a, b) for a in range(5) for b in range(5) if a != b]
GATES = [(c, t, s) for c, t in PAIRS for s in (1, 2)]   # CNOT^s, c -> t


def conj(kind, v, gate, inverse=False):
    """Vector of U P U^dag (inverse=True: U^dag P U) for P = Z^v or X^v."""
    c, t, s = gate
    if inverse:
        s = (-s) % D
    v = list(v)
    if kind == "Z":
        v[c] = (v[c] - s * v[t]) % D
    else:
        v[t] = (v[t] + s * v[c]) % D
    return tuple(v)


def pull_back(kind, v, frame):
    """F^dag P F for frame F = U_1..U_L applied in that order."""
    for g in reversed(frame):
        v = conj(kind, v, g, inverse=True)
    return v


def push(kind, v, frame):
    for g in frame:
        v = conj(kind, v, g)
    return v


ALLV = list(itertools.product(range(D), repeat=5))


def supp(v):
    return frozenset(i for i in range(5) if v[i])


def check_options(kind, target, frames):
    """touched set -> min reads, over all ways to write target = sum_f sum_q c_fq F_f^dag P_q F_f
    (breadth-first search over (accumulated vector, touched set))."""
    moves = []
    for fr in frames:
        for q in range(5):
            e = tuple(int(i == q) for i in range(5))
            r = pull_back(kind, e, fr)
            for c in (1, 2):
                moves.append((q, tuple((c * x) % D for x in r)))
    start = ((0,) * 5, frozenset())
    dist = {start: 0}
    layer = [start]
    best = {}
    while layer:
        nxt = []
        for acc, T in layer:
            d = dist[(acc, T)]
            for q, r in moves:
                st = (tuple((a + b) % D for a, b in zip(acc, r)), T | {q})
                if st not in dist:
                    dist[st] = d + 1
                    nxt.append(st)
        layer = nxt
    for (acc, T), d in dist.items():
        if acc == tuple(target) and d:
            best[T] = min(best.get(T, 99), d)
    return best


def colourable(edges, groups):
    """2-colouring with edges bichromatic and every group monochromatic."""
    parent = {i: i for i in range(5)}
    parity = {i: 0 for i in range(5)}

    def find(x):
        if parent[x] == x:
            return x, 0
        r, p = find(parent[x])
        parent[x] = r
        parity[x] = (parity[x] + p) % 2
        return r, parity[x]

    def union(a, b, d):
        (ra, pa), (rb, pb) = find(a), find(b)
        if ra == rb:
            return (pa + pb) % 2 == d
        parent[ra] = rb
        parity[ra] = (pa + pb + d) % 2
        return True

    for a, b in edges:
        if not union(a, b, 1):
            return False
    for g in groups:
        g = sorted(g)
        for x in g[1:]:
            if not union(g[0], x, 0):
                return False
    return True


NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def embed(edges, groups, size=7):
    """First lattice embedding (data sites, ancilla sites) or None."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    # data that share a group must be within distance 2 (common neighbour site)
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
            if d == 0:
                return False
            if r in adj[q] and d != 1:
                return False
            if r in close[q] and d != 2:
                # two data in one touched set must share an empty neighbour site: distance 2
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
            if len(set(pick)) == len(pick):
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


def windows_of_len(L):
    for seq in itertools.product(GATES, repeat=L):
        # skip a gate followed by another on the same ordered pair (reducible)
        if any(seq[i][:2] == seq[i + 1][:2] for i in range(L - 1)):
            continue
        yield seq


def timelines(shape):
    """shape: tuple of window lengths, e.g. (1, 1) = two separate one-gate windows."""
    pools = [list(windows_of_len(L)) for L in shape]
    for combo in itertools.product(*pools):
        # windows of equal length: unordered (they commute as far as the frame set goes)
        if any(shape[i] == shape[i + 1] and combo[i] >= combo[i + 1] for i in range(len(shape) - 1)):
            continue
        yield combo


def analyse(seq, max_gates, prefix_only=False):
    """seq: tuple of windows (each a tuple of gates)."""
    frames = [()] + [w[:i] for w in seq for i in range(1, len(w) + 1)]
    if prefix_only:
        frames = [()] + list(seq)
    L = sum(len(w) for w in seq)
    edges = {(2, 3)} | {tuple(sorted(g[:2])) for w in seq for g in w}
    if not colourable(edges, []):
        return []
    opts = []
    for nm in NAMES:
        kind, tgt = CHECKS[nm]
        o = check_options(kind, tgt, frames)
        opts.append(sorted(o.items(), key=lambda kv: kv[1]))
    budget = max_gates - 2 * L
    out = []
    mins = [o[0][1] for o in opts]
    if sum(mins) > budget:
        return []

    def rec(j, chosen, cost):
        if j == 4:
            if colourable(edges, [T for T, _ in chosen]):
                out.append((cost + 2 * L, seq, list(chosen)))
            return
        for T, c in opts[j]:
            if cost + c + sum(mins[j + 1:]) > budget:
                break
            if len(T) > 4:
                continue
            if not colourable(edges, [t for t, _ in chosen] + [T]):
                continue
            chosen.append((T, c))
            rec(j + 1, chosen, cost + c)
            chosen.pop()

    rec(0, [], 0)
    return out


def _work(args):
    seq, max_gates, prefix_only = args
    res = analyse(seq, max_gates, prefix_only)
    good = []
    seen = set()
    for total, s, chosen in res:
        edges = sorted({(2, 3)} | {tuple(sorted(g[:2])) for w in s for g in w})
        groups = [T for T, _ in chosen]
        key = (tuple(edges), tuple(map(lambda t: tuple(sorted(t)), groups)))
        if key in seen:
            continue
        seen.add(key)
        e = embed(edges, groups)
        if e:
            good.append((total, s, chosen, edges, e))
    return good


if __name__ == "__main__":
    shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1].split(";")]
    max_gates = int(sys.argv[2])
    prefix_only = "--prefix-only" in sys.argv
    rows = []
    for shape in shapes:
        jobs = [(seq, max_gates, prefix_only) for seq in timelines(shape)]
        with Pool(min(120, len(jobs))) as pool:
            for good in pool.imap_unordered(_work, jobs, chunksize=8):
                rows += good
        print(f"windows {shape}: {len(jobs)} timelines, {len(rows)} embeddable candidates so far", flush=True)
    rows.sort(key=lambda r: (r[0], len(r[3]) + sum(len(T) for T, _ in r[2])))
    print(f"\nbest candidates (<= {max_gates} two-qutrit gates incl. 2L data gates):")
    seen = set()
    for total, s, chosen, edges, (pos, anc) in rows[:400]:
        k = (total, tuple(edges), tuple(tuple(sorted(T)) for T, _ in chosen))
        if k in seen:
            continue
        seen.add(k)
        ncoup = len(edges) + sum(len(T) for T, _ in chosen)
        gs = " | ".join(" ".join(f"{'CNOT' if g[2] == 1 else 'CNOT_INV'}({g[0]}->{g[1]})" for g in w) for w in s)
        print(f"  {total:2d} gates  frame [{gs}]  data couplers {edges}  couplers {ncoup}  "
              + "  ".join(f"{NAMES[j]}:{sorted(T)}/{c}" for j, (T, c) in enumerate(chosen))
              + f"  data {pos} anc {anc}", flush=True)
