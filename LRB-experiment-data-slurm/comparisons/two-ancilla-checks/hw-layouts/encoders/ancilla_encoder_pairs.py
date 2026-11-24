"""One-ancilla layout encoder search, restricted to subsets of the helper ancillas.

The data of the paper's one-ancilla layout share only the q2-q3 coupler, so an
encoder must route through ancillas; a Steiner tree joining the five data needs
two ancillas (6 edges). Searching each ancilla subset separately keeps the
subspace BFS small. Same CSS model and target set as ancilla_encoder_search.py.
Usage: ancilla_encoder_pairs.py [max_depth] [subset size]
"""
import contextlib
import io
import itertools
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")
import ancilla_encoder_search as AS  # noqa: E402

import os
L = AS.LAYOUTS[os.environ.get("LAYOUT", "one-ancilla")]
NAMES = {5: "a1(G1:q0q1q3)", 6: "a2(G2:q1q2q4)", 7: "a3(G3:q0q1q2)", 8: "a4(G4:q1q3q4)"} if L is AS.LAYOUTS["one-ancilla"] else {5: "N(q0q1q2)", 6: "W(q0q2q3)", 7: "E(q1q2q4)", 8: "S(q2q3q4)"}


def run(args):
    sub, max_depth = args
    couplers = [e for e in L["couplers"] if all(w < 5 or w in sub for w in e)]
    lay = {"n": L["n"], "anc": tuple(sub), "couplers": couplers}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        depth, hits = AS.search(lay, max_depth)
    return sub, depth, hits, buf.getvalue()


if __name__ == "__main__":
    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    subs = list(itertools.combinations(L["anc"], size))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else len(subs)) as pool:
        for sub, depth, hits, log in pool.imap_unordered(run, [(s, max_depth) for s in subs]):
            print(f"ancillas {[NAMES[a] for a in sub]}: "
                  + (f"minimal {depth} two-qutrit gates, {len(hits)} encoders" if depth else "none within depth"),
                  flush=True)
            for line in log.strip().splitlines():
                print("    " + line, flush=True)
            for W, plus, path in hits[:4]:
                start = sorted(next(i for i, v in enumerate(row) if v) for row in W)
                print(f"    |+> on {start}, ancillas left in |+>: {list(plus)}; gates {list(path)}", flush=True)
