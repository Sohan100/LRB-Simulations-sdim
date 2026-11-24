"""Fewest-gate CNOT-network encoder of |+_L> on the square-lattice one-ancilla layouts.

Same model and search as ../encoders/ancilla_encoder_search.py (helper ancillas end
in |0> or |+>, CNOT / CNOT_INV on the round's couplers only), run per subset of helper
ancillas (as ../encoders/ancilla_encoder_pairs.py) to keep the subspace BFS small.
Usage: oa_encoder.py LAYOUT MAX_DEPTH SUBSET_SIZE   (LAYOUT: sq14 | sq01)
"""
import contextlib
import io
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "encoders"))
import ancilla_encoder_search as AS  # noqa: E402

# couplers used by the rounds in rounds.py (wires: data 0..4, a1..a4 = 5..8)
LAYOUTS = {
    "sq14": [(1, 4), (2, 3), (5, 4), (6, 0), (6, 2), (7, 3), (8, 0), (8, 1), (5, 6), (7, 8)],
    "sq01": [(0, 1), (2, 3), (5, 1), (5, 4), (6, 3), (7, 2), (7, 4), (8, 0), (5, 6), (7, 8)],
}


def run(args):
    name, sub, max_depth = args
    couplers = [e for e in LAYOUTS[name] if all(w < 5 or w in sub for w in e)]
    lay = {"n": 9, "anc": tuple(sub), "couplers": couplers}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        depth, hits = AS.search(lay, max_depth)
    return sub, depth, hits


if __name__ == "__main__":
    name, max_depth, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    lb = AS.steiner_lower_bound(9, LAYOUTS[name], list(range(5)))
    print(f"{name}: couplers {LAYOUTS[name]}; Steiner lower bound {lb} two-qutrit gates", flush=True)
    import networkx as nx
    subs = []
    for sub in itertools.combinations(range(5, 9), size):
        G = nx.Graph([e for e in LAYOUTS[name] if all(w < 5 or w in sub for w in e)])
        G.add_nodes_from(range(5))
        # an encoder must entangle all five data, so its gates must connect them
        if nx.is_connected(G.subgraph([v for v in G if v < 5 or v in sub])):
            subs.append(sub)
    print(f"helper subsets of size {size} that connect the data: {subs}", flush=True)
    best = None
    with Pool(max(1, len(subs))) as pool:
        for sub, depth, hits in pool.imap_unordered(run, [(name, s, max_depth) for s in subs]):
            print(f"  helper ancillas {sub}: " + (f"minimal {depth} gates, {len(hits)} encoders" if depth
                                                  else f"none with <= {max_depth} gates"), flush=True)
            if depth and (best is None or depth < best[0]):
                best = (depth, sub, hits)
    if best:
        depth, sub, hits = best
        print(f"best: {depth} two-qutrit gates through {sub}")
        for W, plus, path in hits[:5]:
            start = sorted(next(i for i, v in enumerate(row) if v) for row in W)
            print(f"  |+> on {start}, ancillas left in |+>: {list(plus)}; gates {list(path)}")
