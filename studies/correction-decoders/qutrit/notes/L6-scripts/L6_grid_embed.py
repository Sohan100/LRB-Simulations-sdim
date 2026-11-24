"""Check embeddability of the [[5,1,2]]_3 LRB-D interaction graph into a square lattice."""
import networkx as nx
from networkx.algorithms import isomorphism as iso
checks = [(5,0),(5,1),(5,3),(6,1),(6,2),(6,4),(7,0),(7,1),(7,2),(8,3),(8,1),(8,4)]
G12 = nx.Graph(checks)                 # check-round couplings only
G13 = nx.Graph(checks + [(2,3)])       # plus logical SWAP(2,3)/CZ(2,3)
print("bipartite(checks only):", nx.is_bipartite(G12), " bipartite(+2-3):", nx.is_bipartite(G13))
print("odd cycle example in G13:", [c for c in nx.cycle_basis(G13) if len(c)%2==1][:2])
for n in (3,5):
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(n,n), label_attribute="xy")
    for name,G in (("checks",G12),("checks+2-3",G13)):
        GM = iso.GraphMatcher(grid, G)
        found = GM.subgraph_is_monomorphic()
        print(f"{n}x{n} grid, {name}: monomorphism exists = {found}")
# all embeddings of the check graph into a 5x5 grid: distance between images of data 2 and 3
grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5,5), label_attribute="xy")
xy = nx.get_node_attributes(grid,"xy")
dists=set(); cnt=0
for m in iso.GraphMatcher(grid, G12).subgraph_monomorphisms_iter():
    inv = {v:k for k,v in m.items()}
    (a,b),(c,d) = xy[inv[2]], xy[inv[3]]
    dists.add(abs(a-c)+abs(b-d)); cnt+=1
    if cnt==1: print("example placement:", {q: xy[inv[q]] for q in range(9)})
print("number of embeddings into 5x5:", cnt, " grid distance(2,3) over all embeddings:", sorted(dists))
