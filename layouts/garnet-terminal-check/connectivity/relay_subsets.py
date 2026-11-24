"""Relay variants where the relay carries one or two of a stabilizer's data qutrits.

Each stabilizer: one ancilla, or syndrome ancilla a + relay r (a - r coupled) with r
coupled to a nonempty proper subset of the three data qutrits and a to the rest.
Tested on Garnet with the q2-q3 logical-gate coupler.
"""
import itertools

import networkx as nx
from networkx.algorithms import isomorphism

from embed_check import G, STABS, GATE, one_ancilla


def split(name, data, on_relay):
    a, r = f"a{name}", f"r{name}"
    return ([(a, f"q{d}") for d in data if d not in on_relay]
            + [(r, f"q{d}") for d in on_relay] + [(a, r)])


options = [None] + [s for k in (1, 2) for s in itertools.combinations(range(3), k)]
tested = fit = 0
for choice in itertools.product(options, repeat=4):
    edges = list(GATE)
    for (name, data), c in zip(STABS.items(), choice):
        edges += one_ancilla(name, data) if c is None else split(name, data, {data[i] for i in c})
    P = nx.Graph(edges)
    if not nx.is_bipartite(P):
        continue
    tested += 1
    if next(isomorphism.GraphMatcher(G, P).subgraph_monomorphisms_iter(), None):
        fit += 1
        print("fits:", choice)
print(f"relay carrying 1 or 2 data qutrits: {fit} of {tested} bipartite layouts fit Garnet")
