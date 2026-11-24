"""Place a layout's coupler graph on IQM Garnet (20 sites) and list placements.

Garnet couplers from paper-rework/garnet-connectivity/embed_check.py (IQM
fake_garnet). A placement maps every layout qutrit to a distinct Garnet site so
that every layout coupler is a Garnet coupler (subgraph monomorphism).
"""
import itertools

import networkx as nx
from networkx.algorithms import isomorphism

GARNET = [(1, 2), (1, 4), (2, 5), (3, 4), (8, 3), (4, 5), (9, 4), (5, 6), (10, 5), (6, 7),
          (11, 6), (12, 7), (8, 9), (8, 13), (9, 10), (9, 14), (10, 11), (10, 15), (11, 12),
          (16, 11), (17, 12), (13, 14), (14, 15), (18, 14), (16, 15), (19, 15), (16, 17),
          (16, 20), (18, 19), (19, 20)]
G = nx.Graph([(f"QB{a}", f"QB{b}") for a, b in GARNET])


def placements(couplers):
    L = nx.Graph(couplers)
    gm = isomorphism.GraphMatcher(G, L)
    out = []
    for m in gm.subgraph_monomorphisms_iter():
        out.append({v: k for k, v in m.items()})
    return out


if __name__ == "__main__":
    import sys
    A1 = [("q0", "q1"), ("q1", "q2"), ("q2", "q3"), ("q0", "q3"), ("q2", "q4"),
          ("E", "q1"), ("E", "q4"), ("N", "q2")]
    p = placements(A1)
    print(f"A1: {len(p)} Garnet placements; e.g. {p[0] if p else None}")
