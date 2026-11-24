"""Sanity checks for relay_chains.py.

1. Without the q2-q3 gate, chain layouts must fit Garnet (the one-ancilla checks alone do).
2. With the gate, test every chain layout on a 13x13 square lattice with q1 pinned to the
   centre (allowed by translation symmetry), so Garnet's finite size is not the reason.
"""
import itertools

import networkx as nx
from networkx.algorithms import isomorphism

from embed_check import G, STABS, GATE, one_ancilla
from relay_chains import chain

SQ = nx.grid_2d_graph(13, 13)
nx.set_node_attributes(SQ, {v: ("q1" if v == (6, 6) else "") for v in SQ}, "tag")


def layouts(gate, hops):
    for choice in itertools.product([None] + [(i, h) for i in range(3) for h in hops], repeat=4):
        edges = list(gate)
        for (name, data), c in zip(STABS.items(), choice):
            edges += one_ancilla(name, data) if c is None else chain(name, data, data[c[0]], c[1])
        yield choice, nx.Graph(edges)


def fits(P, host, pin):
    if not nx.is_bipartite(P):
        return False
    if pin:
        nx.set_node_attributes(P, {v: ("q1" if v == "q1" else "") for v in P}, "tag")
        gm = isomorphism.GraphMatcher(host, P, node_match=lambda a, b: a["tag"] == b["tag"])
    else:
        gm = isomorphism.GraphMatcher(host, P)
    return next(gm.subgraph_monomorphisms_iter(), None) is not None


n = sum(fits(P, G, False) for _, P in layouts([], (1, 2, 3)) if P.number_of_nodes() <= 20)
print("no gate, Garnet, layouts that fit:", n, flush=True)
tested = hit = 0
for choice, P in layouts(GATE, (1, 2, 3)):
    if P.number_of_nodes() > 20 or not nx.is_bipartite(P):
        continue
    tested += 1
    hit += fits(P, SQ, True)
print(f"gate, 13x13 square lattice: {hit} of {tested} bipartite layouts fit", flush=True)
