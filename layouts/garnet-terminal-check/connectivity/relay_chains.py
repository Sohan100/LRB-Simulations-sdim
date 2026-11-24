"""Do longer relay chains let a check-based LRB-D fit Garnet with the q2-q3 gate?

Same conventions as embed_check.py. A chain of h relays carries one data qutrit
of a stabilizer to its syndrome ancilla: a - r1 - ... - rh - q. Parity on a
bipartite graph allows only odd h for the qutrit of the other sublattice.
"""
import itertools

import networkx as nx
from networkx.algorithms import isomorphism

from embed_check import G, STABS, GATE, one_ancilla


def chain(name, data, relayed, hops):
    a = f"a{name}"
    nodes = [a] + [f"r{name}{i}" for i in range(1, hops + 1)]
    return ([(a, f"q{d}") for d in data if d != relayed]
            + list(zip(nodes, nodes[1:])) + [(nodes[-1], f"q{relayed}")])


def fits(edges, host):
    P = nx.Graph(edges)
    if not nx.is_bipartite(P):
        return None
    return next(isomorphism.GraphMatcher(host, P).subgraph_monomorphisms_iter(), None)


found = []
for choice in itertools.product([None] + [(i, h) for i in range(3) for h in (1, 2, 3)], repeat=4):
    edges = list(GATE)
    for (name, data), c in zip(STABS.items(), choice):
        edges += one_ancilla(name, data) if c is None else chain(name, data, data[c[0]], c[1])
    n = nx.Graph(edges).number_of_nodes()
    if n > 20:
        continue
    m = fits(edges, G)
    if m:
        desc = ", ".join(n_ if c is None else f"{n_}:q{STABS[n_][c[0]]} via {c[1]} relay(s)"
                         for n_, c in zip(STABS, choice))
        found.append((n, desc, {v: k for k, v in m.items()}))
found.sort()
print(f"{len(found)} chain layouts fit Garnet with the q2-q3 gate")
for n, desc, place in found[:12]:
    print(f"  {n} qutrits: {desc}")
if found:
    print("smallest, example placement:", found[0][2])
