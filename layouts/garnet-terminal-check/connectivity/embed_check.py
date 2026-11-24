"""Which LRB-D circuits fit IQM Garnet's coupler graph without SWAP routing?

Garnet edges: IQM's qiskit-on-iqm fake backend
(src/iqm/qiskit_iqm/fake_backends/fake_garnet.py, `qubit_connectivity`), CZ on
these 30 pairs only. Required couplings come from the circuits in
src/lrb/code_definitions.py (FoldedQutritDetectionCode and
FoldedQutritSplitAncillaUniformDetectionCode): every two-qutrit gate needs a
coupler between its two qutrits. A layout fits if some injective placement maps
every required pair onto a Garnet pair (subgraph monomorphism).
"""
import itertools

import networkx as nx
from networkx.algorithms import isomorphism

GARNET = [(1, 2), (1, 4), (2, 5), (3, 4), (8, 3), (4, 5), (9, 4), (5, 6), (10, 5), (6, 7),
          (11, 6), (12, 7), (8, 9), (8, 13), (9, 10), (9, 14), (10, 11), (10, 15), (11, 12),
          (16, 11), (17, 12), (13, 14), (14, 15), (18, 14), (16, 15), (19, 15), (16, 17),
          (16, 20), (18, 19), (19, 20)]
G = nx.Graph([(f"QB{a}", f"QB{b}") for a, b in GARNET])
SQUARE = nx.grid_2d_graph(9, 9)  # infinite-lattice stand-in

STABS = {"G1": (0, 1, 3), "G2": (1, 2, 4), "G3": (0, 1, 2), "G4": (1, 3, 4)}
GATE = [("q2", "q3")]                                    # CZ_{2->3} in S, SWAP_{2,3} in H
ENCODER = [("q2", "q1"), ("q1", "q0"), ("q3", "q0"), ("q4", "q2")]


def one_ancilla(name, data):
    return [(f"a{name}", f"q{d}") for d in data]


def relay(name, data, relayed):
    a, r = f"a{name}", f"r{name}"
    return [(a, f"q{d}") for d in data if d != relayed] + [(r, f"q{relayed}"), (a, r)]


ONE_ANC = sum((one_ancilla(n, s) for n, s in STABS.items()), [])
# Simulated two-ancilla layout: G1 relays q3 (wire 9), G4 relays q4 (wire 10).
TWO_ANC = (relay("G1", STABS["G1"], 3) + one_ancilla("G2", STABS["G2"])
           + one_ancilla("G3", STABS["G3"]) + relay("G4", STABS["G4"], 4))


def report(label, edges, host=G):
    P = nx.Graph(edges)
    bip = nx.is_bipartite(P)
    odd = None
    if not bip:
        odd = min((c for c in nx.simple_cycles(P, length_bound=7) if len(c) % 2), key=len)
    gm = isomorphism.GraphMatcher(host, P)
    n = sum(1 for _ in gm.subgraph_monomorphisms_iter())
    ex = next(isomorphism.GraphMatcher(host, P).subgraph_monomorphisms_iter(), None)
    place = {v: k for k, v in ex.items()} if ex else None
    print(f"{label}: {P.number_of_nodes()} qutrits, {P.number_of_edges()} couplings, "
          f"bipartite={bip}, placements={n}")
    if odd:
        print(f"    shortest odd cycle: {' - '.join(odd)}")
    if place:
        print("    example:", ", ".join(f"{k}->{place[k]}" for k in sorted(place)))
    return n, place


print("Garnet: bipartite =", nx.is_bipartite(G), ", max degree =", max(d for _, d in G.degree()))
print()
report("terminal check (const-0): encoder + gate", ENCODER + GATE)
report("terminal check, gate only", GATE + [("q0", "q0")][:0] + [(f"q{i}", f"q{i}") for i in []] or GATE)
print()
report("one-ancilla checks + gate", ONE_ANC + GATE)
report("one-ancilla checks alone", ONE_ANC)
report("one-ancilla checks + gate on 9x9 square lattice", ONE_ANC + GATE, SQUARE)
print()
report("two-ancilla (simulated) checks + gate", TWO_ANC + GATE)
report("two-ancilla (simulated) checks alone", TWO_ANC)
report("two-ancilla (simulated) + gate on 9x9 square lattice", TWO_ANC + GATE, SQUARE)
print()
print("All relay assignments: each stabilizer one ancilla, or ancilla + relay on one of its data")
rows = []
for choice in itertools.product([None, 0, 1, 2], repeat=4):
    edges = list(GATE)
    for (name, data), c in zip(STABS.items(), choice):
        edges += one_ancilla(name, data) if c is None else relay(name, data, data[c])
    P = nx.Graph(edges)
    if not nx.is_bipartite(P):
        continue
    gm = isomorphism.GraphMatcher(G, P)
    ok = next(gm.subgraph_monomorphisms_iter(), None) is not None
    gs = isomorphism.GraphMatcher(SQUARE, P)
    ok_sq = next(gs.subgraph_monomorphisms_iter(), None) is not None
    desc = ", ".join(n if c is None else f"{n}:relay q{STABS[n][c]}" for n, c in zip(STABS, choice))
    rows.append((P.number_of_nodes(), desc, ok, ok_sq))
for nq, desc, ok, ok_sq in sorted(rows):
    print(f"  {nq:2d} qutrits  garnet={ok!s:5}  square={ok_sq!s:5}  {desc}")
