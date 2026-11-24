"""All 4-gate CNOT encoders of |+_L> on one data tree, with depth and qubit validity.

Depth counts the H layer on the |+>-start qutrits plus the CNOT layers when gates
on disjoint qutrits run together (ASAP). Qubit validity: the same gate list with
CNOT_INV -> CNOT must map span(e_S) onto V mod 2 (the qubit code's |+_L>).
Usage: tree_encoders.py 01,03,12,14
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
import encoder_trees as ET  # noqa: E402


def rref_mod(M, d):
    M = np.array(M, dtype=int) % d
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), -1, d)) % d
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % d
        r += 1
    return tuple(map(tuple, M[:r]))


def maps_to_V(S, gates, d):
    M = np.zeros((3, 5), dtype=int)
    for i, q in enumerate(S):
        M[i, q] = 1
    for name, c, t in gates:
        s = 1 if (name == "CNOT" or d == 2) else -1
        M[:, t] = (M[:, t] + s * M[:, c]) % d
    Vd = np.array([ET.G3, ET.G4, ET.XL]) % d   # signed vectors, reduced mod d
    return rref_mod(M, d) == rref_mod(Vd, d)


def depth(S, gates):
    busy = {q: 1 for q in S}          # H layer on the |+> qutrits
    for _, c, t in gates:
        layer = max(busy.get(c, 0), busy.get(t, 0)) + 1
        busy[c] = busy[t] = layer
    return max(busy.values())


if __name__ == "__main__":
    T = {frozenset((int(e[0]), int(e[1]))) for e in sys.argv[1].split(",")}
    found = ET.cnot_encoders()[4]
    rows = []
    for S, p in found:
        if {frozenset(g[1:]) for g in p} != T:
            continue
        rows.append((depth(S, p), S, p, maps_to_V(S, p, 3), maps_to_V(S, p, 2)))
    # cnot_encoders keeps at most 50 paths per subspace; enumerate exhaustively on this tree too
    gates = [(n, c, t) for e in T for c, t in (tuple(e), tuple(e)[::-1]) for n in ("CNOT", "CNOT_INV")]
    seen = {(S, p) for _, S, p, _, _ in rows}
    for S in itertools.combinations(range(5), 3):
        for p in itertools.product(gates, repeat=4):
            if {frozenset(g[1:]) for g in p} != T or (S, p) in seen:
                continue
            if maps_to_V(S, p, 3):
                rows.append((depth(S, p), S, p, True, maps_to_V(S, p, 2)))
                seen.add((S, p))
    rows.sort(key=lambda r: (not r[4], r[0]))
    print(f"tree {sys.argv[1]}: {len(rows)} four-CNOT encoders; "
          f"{sum(1 for r in rows if r[4])} also valid for the qubit code")
    for dep, S, p, q3, q2 in rows[:12]:
        print(f"  depth {dep} (H layer + CNOT layers)  |+> on {S}  gates {list(p)}  qutrit {q3} qubit {q2}")
