"""Meet-in-the-middle check: is there a 7-gate CNOT-network encoder on the one-ancilla layout?

Forward BFS (gates applied) from every coordinate start subspace up to depth F,
backward BFS (gates undone) from every target up to depth K-F; an encoder with K
gates exists iff some subspace is reached by both at depths summing to K
(dimensions must agree, which they do automatically). All four ancillas allowed.
Same CSS model as ancilla_encoder_search.py. Usage: ancilla_encoder_mitm.py K F
"""
import itertools
import sys

sys.path.insert(0, ".")
import ancilla_encoder_search as AS  # noqa: E402

import os
L = AS.LAYOUTS[os.environ.get("LAYOUT", "one-ancilla")]
n, anc = L["n"], L["anc"]
GATES = [(nm, c, t) for a, b in L["couplers"] for c, t in ((a, b), (b, a)) for nm in ("CNOT", "CNOT_INV")]


def step(W, g, sign):
    nm, c, t = g
    s = (1 if nm == "CNOT" else -1) * sign
    rows = [list(r) for r in W]
    for r in rows:
        r[t] = (r[t] + s * r[c]) % AS.D
    return AS.rref_t(rows, n)


def bfs(starts, depth, sign):
    seen = {W: 0 for W in starts}
    frontier = list(starts)
    for d in range(1, depth + 1):
        nxt = []
        for W in frontier:
            for g in GATES:
                W2 = step(W, g, sign)
                if W2 not in seen:
                    seen[W2] = d
                    nxt.append(W2)
        frontier = nxt
        print(f"  {'forward' if sign > 0 else 'backward'} depth {d}: {len(nxt)} new", flush=True)
    return seen


if __name__ == "__main__":
    K, F = int(sys.argv[1]), int(sys.argv[2])
    Vrows = [list(v) + [0] * (n - 5) for v in AS.ET.V.tolist()]
    targets, starts = [], []
    for k in range(len(anc) + 1):
        for plus in itertools.combinations(anc, k):
            targets.append(AS.rref_t(Vrows + [[int(i == a) for i in range(n)] for a in plus], n))
    for dim in range(3, 3 + len(anc) + 1):
        for S in itertools.combinations(range(n), dim):
            starts.append(AS.rref_t([[int(i == q) for i in range(n)] for q in S], n))
    fw = bfs(starts, F, +1)
    bw = bfs(targets, K - F, -1)
    best = min((fw[W] + bw[W] for W in fw.keys() & bw.keys()), default=None)
    print(f"shortest encoder found by meeting in the middle (<= {K} gates): {best}")
