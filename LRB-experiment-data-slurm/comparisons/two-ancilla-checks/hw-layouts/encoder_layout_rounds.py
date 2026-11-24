"""Build and fault-check full check rounds for encoder-compatible square layouts.

Layouts (square lattice, data plaquette q0 (0,0), q1 (1,0), q2 (1,1), q3 (0,1)):
  A1: q4 at (2,1); ancilla E at (2,0) touches q1, q4 (reads G2, G4); ancilla N at
      (1,2) touches q2 (reads G1, G3).
  A2: same E; ancilla S at (1,-1) touches q1 (reads G1, G3).
  A3: E alone reads all four.
Usage: encoder_layout_rounds.py [--layout=A1] [--slack=N] [--validate]
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import encoder_layout_search as ES  # noqa: E402
import hw_layouts_check as H  # noqa: E402

E, N = 5, 6
LAYOUTS = {
    "A1": {"G1": (N, (2,)), "G2": (E, (1, 4)), "G3": (N, (2,)), "G4": (E, (1, 4))},
    "A2": {"G1": (N, (1,)), "G2": (E, (1, 4)), "G3": (N, (1,)), "G4": (E, (1, 4))},
    "A3": {"G1": (E, (1, 4)), "G2": (E, (1, 4)), "G3": (E, (1, 4)), "G4": (E, (1, 4))},
}


def candidates(layout, slack):
    per = {}
    for chk, (anc, touch) in LAYOUTS[layout].items():
        kind, vec = ES.CHECKS[chk]
        sols = ES.search(kind, vec, touch, max_pushes=4, max_cost=16, slack=slack)
        per[chk] = [ES.to_ops(kind, anc, proto, chk) for _, proto in sols]
    return per


def grouped(g1, g2, g3, g4):
    """Z checks, then their readouts together, then X checks, then their readouts:
    two readout layers per round instead of one per check."""
    def split(ops):
        gates = [op for op in ops if op[0] == "g"]
        coll = [op for op in ops if op[0] != "g"]
        return gates, coll
    out = []
    for a, b in ((g1, g2), (g3, g4)):
        ga, ca = split(a)
        gb, cb = split(b)
        out += ga + gb + [op for op in ca + cb if op[0] == "m"] + [op for op in ca + cb if op[0] == "r"]
    return out


def rounds(layout, slack):
    per = candidates(layout, slack)
    for combo in itertools.product(*(range(len(per[c])) for c in ("G1", "G2", "G3", "G4"))):
        ops = grouped(per["G1"][combo[0]], per["G2"][combo[1]], per["G3"][combo[2]], per["G4"][combo[3]])
        yield f"{layout} combo {combo}", ops


if __name__ == "__main__":
    layout = next((a.split("=")[1] for a in sys.argv if a.startswith("--layout=")), "A1")
    slack = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--slack=")), "0"))
    allr = list(rounds(layout, slack))
    print(f"{layout}: {len(allr)} candidate rounds (slack {slack})", flush=True)
    if "--validate" in sys.argv:
        name, ops = allr[0]
        print(name, H.stats(ops), flush=True)
        for line in H.noiseless_tests(name, ops):
            print(line, flush=True)
        sys.exit()
    with Pool(min(48, len(allr))) as pool:
        results = []
        for name, lines in pool.imap_unordered(H.fault_test, allr):
            silent = [int(l.split(" silent")[0].split()[-1]) for l in lines]
            results.append((sum(silent), name, lines))
            print(name, "silent (X, Z):", silent, flush=True)
    results.sort()
    best = [r for r in results if r[0] == 0]
    print(f"\n{len(best)} of {len(results)} rounds have 0 silent single faults", flush=True)
    for s, name, lines in results[:5]:
        ops = dict(allr)[name]
        print(name, "two-qutrit gates, readouts:", H.stats(ops), flush=True)
