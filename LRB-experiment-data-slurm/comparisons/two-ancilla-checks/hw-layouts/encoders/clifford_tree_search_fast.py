"""Faster exhaustive 4-gate Clifford encoder search on given data trees.

Same model as clifford_tree_search.py (CZ after per-endpoint axis choices, final
local Cliffords free, exact LC-equivalence test), but: intermediate states are
deduplicated per edge order, final states are deduplicated, a two-entropy filter
(S({0,3}) = S({2,4}) = 1) runs before the full profile and the exact LC test, and
edge orders run in parallel.
"""
import itertools
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
import clifford_tree_search as CS  # noqa: E402

START = np.zeros((5, 10), dtype=int)
for _i in range(5):
    START[_i, 5 + _i] = 1


def step_states(S, i, j):
    out = {}
    for a in range(4):
        Si = CS.local(S, i, CS.REPS[a])
        for b in range(4):
            S2 = CS.cz(CS.local(Si, j, CS.REPS[b]), i, j)
            out[CS.ET.key(S2)] = S2
    return out


def run_order(order):
    layer = {CS.ET.key(START): START}
    for (i, j) in order:
        nxt = {}
        for S in layer.values():
            nxt.update(step_states(S, i, j))
        layer = nxt
    hits = 0
    for S in layer.values():
        if CS.entropy(S, (0, 3)) != 1 or CS.entropy(S, (2, 4)) != 1:
            continue
        if tuple(CS.entropy(S, A) for A in CS.SUBSETS) != CS.TPROF:
            continue
        if CS.lc_equivalent(S, CS.TARGET):
            hits += 1
    return order, len(layer), hits


if __name__ == "__main__":
    trees = {"paper tree (sanity check)": [(0, 1), (0, 3), (1, 2), (2, 4)],
             "star at q1": [(0, 1), (1, 2), (1, 3), (1, 4)],
             "star at q2": [(0, 2), (1, 2), (2, 3), (2, 4)],
             "star at q3": [(0, 3), (1, 3), (2, 3), (3, 4)],
             "star at q4": [(0, 4), (1, 4), (2, 4), (3, 4)]}
    with Pool(16) as pool:
        for name, T in trees.items():
            orders = list(itertools.permutations(T))
            res = pool.map(run_order, orders)
            total = sum(h for _, _, h in res)
            nstates = sum(n for _, n, _ in res)
            good = [o for o, _, h in res if h]
            print(f"{name:28s} edges {T}: {nstates} distinct final states over {len(orders)} orders; "
                  + (f"4-gate Clifford encoder EXISTS ({total} LC-equivalent finals), e.g. order {good[0]}"
                     if total else "NO 4-gate Clifford encoder (exhaustive)"), flush=True)
