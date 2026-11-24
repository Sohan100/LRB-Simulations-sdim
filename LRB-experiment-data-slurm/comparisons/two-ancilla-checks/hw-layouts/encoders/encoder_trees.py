"""Which data-qutrit couplings can host a minimal encoder of |+_L>?

|+_L> of the [[5,1,2]]_3 code is the uniform superposition over the subspace
V = span(G3, G4, Xbar) of F_3^5 (x-exponent vectors), stabilized by Z^w for w in
V^perp = span(G1, G2). Five qutrits that are all entangled need >= 4 two-qutrit
gates, and with exactly 4 the gates form a spanning tree T of the data.

1. Cut test (necessary for any Clifford circuit): every edge of T splits the data
   into A|B with exactly one gate across, so S(A) <= 1 (in log 3 units).
   S(A) = |A| - dim(V_A) - dim(Vperp_A), V_A = vectors of V supported in A.
2. CNOT encoders: qutrits in S (|S| = 3) start in |+> (H on |0>), the rest in |0>,
   then CNOT / CNOT_INV gates; the circuit maps span(e_S) to M span(e_S), and is
   an encoder iff that equals V. Breadth-first search over subspaces lists every
   4-gate encoder and the trees they use.
"""
import itertools
import sys
from collections import defaultdict

import numpy as np

D = 3
G3, G4, XL = (1, 1, -1, 0, 0), (0, -1, 0, 1, -1), (1, 0, 0, 1, 0)
G1, G2 = (1, -1, 0, -1, 0), (0, 1, 1, 0, -1)
V = np.array([G3, G4, XL]) % D
VP = np.array([G1, G2]) % D
UNSAFE = {frozenset(p) for p in [(0, 3), (2, 4), (0, 2), (3, 4)]}   # pairs holding a weight-2 logical


def rref(M):
    M = np.array(M, dtype=int) % D
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), -1, D)) % D
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % D
        r += 1
        if r == rows:
            break
    return M[:r]


def key(M):
    return tuple(map(tuple, rref(M)))


def span_all(B):
    out = set()
    for coeffs in itertools.product(range(D), repeat=len(B)):
        out.add(tuple((np.array(coeffs) @ B) % D))
    return out


def cut_entropy(A):
    A = set(A)
    def dim_supported(B):
        vecs = [v for v in span_all(B) if all(v[i] == 0 for i in range(5) if i not in A)]
        return int(round(np.log(len(vecs)) / np.log(D)))
    return len(A) - dim_supported(V) - dim_supported(VP)


def spanning_trees(nodes=range(5)):
    pairs = list(itertools.combinations(nodes, 2))
    for es in itertools.combinations(pairs, 4):
        parent = list(range(5))
        def f(x):
            while parent[x] != x:
                x = parent[x]
            return x
        ok = True
        for a, b in es:
            ra, rb = f(a), f(b)
            if ra == rb:
                ok = False
                break
            parent[ra] = rb
        if ok:
            yield frozenset(frozenset(e) for e in es)


def tree_cut_ok(T):
    for e in T:
        rest = [f for f in T if f != e]
        # component of one endpoint without e
        a = min(e)
        comp, todo = {a}, [a]
        while todo:
            u = todo.pop()
            for f in rest:
                if u in f:
                    w = next(iter(f - {u}))
                    if w not in comp:
                        comp.add(w)
                        todo.append(w)
        if cut_entropy(comp) != 1:
            return False
    return True


GATES = [(name, c, t) for c, t in itertools.permutations(range(5), 2) for name in ("CNOT", "CNOT_INV")]


def apply(M, gate):
    name, c, t = gate
    s = 1 if name == "CNOT" else -1
    M = M.copy()
    M[:, t] = (M[:, t] + s * M[:, c]) % D
    return M


def cnot_encoders(max_gates=4, allowed=None):
    """All CNOT encoders with <= max_gates gates (gates restricted to allowed pairs)."""
    target = key(V)
    starts = {}
    for S in itertools.combinations(range(5), 3):
        M = np.zeros((3, 5), dtype=int)
        for i, q in enumerate(S):
            M[i, q] = 1
        starts[key(M)] = S
    layer = {k: [(S, ())] for k, S in starts.items()}
    seen = set(layer)
    found = defaultdict(list)
    for depth in range(1, max_gates + 1):
        nxt = defaultdict(list)
        for k, paths in layer.items():
            M = np.array(k)
            for g in GATES:
                if allowed is not None and frozenset(g[1:]) not in allowed:
                    continue
                k2 = key(apply(M, g))
                for S, p in paths:
                    if k2 == target:
                        found[depth].append((S, p + (g,)))
                    if k2 not in seen:
                        nxt[k2].append((S, p + (g,)))
        seen |= set(nxt)
        layer = {k: v[:50] for k, v in nxt.items()}
        if found:
            break
    return found


if __name__ == "__main__":
    print("entanglement S(A) (log3 units) for every split A|B of the data:")
    for r in (1, 2):
        for A in itertools.combinations(range(5), r):
            print(f"  A={set(A)}: S={cut_entropy(A)}")
    trees = list(spanning_trees())
    ok_trees = [T for T in trees if tree_cut_ok(T)]
    print(f"\n{len(trees)} spanning trees of 5 data qutrits; {len(ok_trees)} pass the cut test")
    found = cnot_encoders()
    depth = min(found)
    sols = found[depth]
    by_tree = defaultdict(list)
    for S, p in sols:
        by_tree[frozenset(frozenset(g[1:]) for g in p)].append((S, p))
    print(f"\nminimal CNOT encoder: {depth} gates; {len(sols)} circuits found, on {len(by_tree)} distinct trees")
    lbl = lambda T: " ".join(sorted(f"{min(e)}{max(e)}" for e in T))
    cnot_trees = set(by_tree)
    for T in sorted(ok_trees, key=lbl):
        safe = all(e not in UNSAFE for e in T)
        tag = "CNOT encoder found" if T in cnot_trees else "no CNOT encoder"
        ex = by_tree[T][0] if T in by_tree else None
        print(f"  tree {lbl(T):12s} {'all edges safe' if safe else 'uses unsafe pair(s) ' + lbl([e for e in T if e in UNSAFE]):30s} {tag}"
              + (f"  e.g. |+> on {ex[0]}: {ex[1]}" if ex else ""))
    print(f"\ntrees with a CNOT encoder but failing the cut test (should be none): "
          f"{[lbl(T) for T in cnot_trees if T not in ok_trees]}")
