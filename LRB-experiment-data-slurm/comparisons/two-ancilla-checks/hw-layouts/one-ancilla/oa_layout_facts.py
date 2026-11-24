"""Hardware facts for one-ancilla square-lattice layouts: drawing, couplers, degrees,
frequency groups, IQM Garnet placements, encoder options.

Encoder options:
- native 4-gate encoder: some 4-gate encoder tree (encoders/encoder_trees.out and
  clifford_tree_search.out: 17 CNOT trees + stars at q2, q3, q4) is made of
  lattice-adjacent data pairs; the pairs not already couplers would be encoder-only
  couplers (extra hardware);
- Steiner lower bound for any encoder on the layout's own couplers (all data
  connected), as in encoders/ancilla_encoder_search.py.
Usage: oa_layout_facts.py CANDIDATE_FILE [indices]
"""
import itertools
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "encoders"))
sys.path.insert(0, str(HERE))
import garnet_place as GP  # noqa: E402
import oa_finalists as F  # noqa: E402

NAME = {0: "q0", 1: "q1", 2: "q2", 3: "q3", 4: "q4", 5: "a1", 6: "a2", 7: "a3", 8: "a4"}
TREES = ["01 02 03 04", "01 02 03 24", "01 03 04 24", "01 03 12 14", "01 03 12 24", "01 03 14 24",
         "01 12 13 24", "01 13 14 24", "02 03 12 24", "03 04 14 24", "03 12 13 14", "03 12 13 24",
         "03 12 23 24", "03 13 14 24", "03 13 23 24", "03 13 24 34", "03 14 24 34",
         "02 12 23 24", "03 13 23 34", "04 14 24 34"]          # last three: stars (CZ-type gates)
TREES = [frozenset(frozenset((int(e[0]), int(e[1]))) for e in t.split()) for t in TREES]
PAPER_TREE = frozenset(frozenset(p) for p in [(0, 1), (0, 3), (1, 2), (2, 4)])


def facts(c):
    pos = dict(c["data"])
    for j, s in enumerate(c["anc"]):
        pos[5 + j] = tuple(s)
    coup = {frozenset(e) for e in c["edges"]}
    for j, g in enumerate(c["groups"]):
        coup |= {frozenset((5 + j, q)) for q in g}
    if c["c12"]:
        coup.add(frozenset((5, 6)))
    if c["c34"]:
        coup.add(frozenset((7, 8)))
    for e in coup:
        a, b = tuple(e)
        assert abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1, e
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    occ = {p: w for w, p in pos.items()}
    rows = []
    for y in range(max(ys), min(ys) - 1, -1):
        rows.append("  ".join(NAME[occ[(x, y)]] if (x, y) in occ else " .".rjust(2)
                              for x in range(min(xs), max(xs) + 1)))
    G = nx.Graph([tuple(e) for e in coup])
    deg = dict(G.degree())
    adj_data = {frozenset((a, b)) for a in range(5) for b in range(a + 1, 5)
                if abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1}
    native = [t for t in TREES if t <= adj_data]
    named = [(NAME[a], NAME[b]) for a, b in (tuple(sorted(e)) for e in coup)]
    gp = GP.placements(named)
    steiner = None
    for k in range(4, 9):
        for sub in itertools.combinations(range(5, 9), k - 4):
            H = G.subgraph(list(range(5)) + list(sub))
            if nx.is_connected(H):
                steiner = k
                break
        if steiner:
            break
    return dict(rows=rows, couplers=sorted(tuple(sorted(e)) for e in coup), ncoup=len(coup),
                maxdeg_data=max(deg.get(q, 0) for q in range(5)), maxdeg_anc=max(deg.get(a, 0) for a in range(5, 9)),
                bipartite=nx.is_bipartite(G), adj_data=sorted(tuple(sorted(e)) for e in adj_data),
                native=[" ".join(f"{min(e)}{max(e)}" for e in sorted(map(sorted, t))) for t in native],
                paper_native=PAPER_TREE <= adj_data, garnet=len(gp), garnet_example=gp[0] if gp else None,
                steiner=steiner)


if __name__ == "__main__":
    lines = Path(sys.argv[1]).read_text().splitlines()
    cands = [c for c in map(F.parse_line, lines) if c]
    idx = [int(i) for i in sys.argv[2:]] or range(len(cands))
    for i in idx:
        c = cands[i]
        f = facts(c)
        print(f"[{i}] frames {c['timeline']} groups {[sorted(g) for g in c['groups']]} a1-a2 {c['c12']} a3-a4 {c['c34']}")
        for r in f["rows"]:
            print("      " + r)
        print(f"    couplers {f['ncoup']}: {f['couplers']}; max degree data {f['maxdeg_data']} / ancilla "
              f"{f['maxdeg_anc']}; bipartite {f['bipartite']}")
        print(f"    lattice-adjacent data pairs {f['adj_data']}; 4-gate encoder trees on them: {f['native'] or 'none'}"
              f"; paper encoder tree native: {f['paper_native']}")
        print(f"    Steiner lower bound for an encoder on the round's couplers: {f['steiner']} two-qutrit gates")
        print(f"    IQM Garnet placements: {f['garnet']}" + (f", e.g. {f['garnet_example']}" if f['garnet'] else ""))
