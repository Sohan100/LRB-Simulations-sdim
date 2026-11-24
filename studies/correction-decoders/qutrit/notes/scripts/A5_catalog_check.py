#!/usr/bin/env python3
"""A5 side check: does the repository's 216-string single-qutrit Clifford
catalog cover the qutrit Clifford group uniformly?

Counts (i) distinct unitaries modulo global phase, built from sdim.unitary
matrices (H = DFT, P = diag(w^{j(j-1)/2}), X, Z), and (ii) the multiplicity
of each symplectic part (the only part that matters when twirling Pauli
noise).  A uniform twirl needs 216 distinct elements, i.e. every one of the
24 symplectic parts exactly 9 times.

Run:  module load python/3.11; python A5_catalog_check.py   (a second or two)
"""
import collections
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "src")))
from lrb import code_definitions as defs  # noqa: E402
from sdim import unitary as U  # noqa: E402

G = {"X": U.generate_x_matrix(3), "Z": U.generate_z_matrix(3),
     "H": U.generate_h_matrix(3), "P": U.generate_p_matrix(3), "I": np.eye(3)}
SYM = {"H": np.array([[0, -1], [1, 0]]), "P": np.array([[1, 0], [1, 1]]),
       "X": np.eye(2, dtype=int), "Z": np.eye(2, dtype=int), "I": np.eye(2, dtype=int)}


def unitary(word):
    M = np.eye(3, dtype=complex)
    for ch in reversed(word):          # descriptors act right to left
        M = G[ch] @ M
    return M


def key(M):
    flat = M.ravel()
    idx = int(np.argmax(np.abs(flat) > 1e-9))
    return tuple(np.round(flat / (flat[idx] / abs(flat[idx])), 6))


def main():
    cat = defs.single_qutrit_cliffords_exhaustive_strings
    keys = [key(unitary(s)) for s in cat]
    group = {key(np.eye(3)): ""}
    queue = deque([""])
    while queue:
        w = queue.popleft()
        for g in "HPXZ":
            k = key(unitary(g + w))
            if k not in group:
                group[k] = g + w
                queue.append(g + w)
    missing = sorted((w for k, w in group.items() if k not in set(keys)), key=len)
    sym = collections.Counter()
    for s in cat:
        M = np.eye(2, dtype=int)
        for ch in reversed(s):
            M = (SYM[ch] @ M) % 3
        sym[tuple(M.ravel())] += 1
    print(f"catalog strings: {len(cat)}, distinct strings: {len(set(cat))}, "
          f"distinct Clifford elements mod phase: {len(set(keys))} of {len(group)}")
    print("missing elements (shortest words, rightmost acts first):", missing)
    dup = [v for v in collections.defaultdict(list,
           {k: [s for s, kk in zip(cat, keys) if kk == k] for k in set(keys)}).values()
           if len(v) > 1]
    print("elements listed more than once:", dup)
    print("multiplicity of the 24 symplectic parts (uniform = 9 each):",
          dict(collections.Counter(sym.values())))


if __name__ == "__main__":
    main()
