import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", (sys.argv[1] if len(sys.argv)>1 else __file__.replace("L6_round_idle.py","L6_si1000_map.py"))); 
src = open((sys.argv[1] if len(sys.argv)>1 else __file__.replace("L6_round_idle.py","L6_si1000_map.py"))).read().split('sets = {')[0]   # reuse only the model functions
ns = {}; exec(src, ns)
S = ns['lindblad_super'](T1_10=56.0, T1_21=34.8, T2_01=61.2, T2_12=28.0, T2_02=36.6)
S2 = ns['lindblad_super'](T1_10=180., T1_21=101., T2_01=76., T2_12=37., T2_02=37.)
for t in (2.3, 3.5, 5.0, 10.0):
    e = 1-ns['F_pro'](S,t); e2 = 1-ns['F_pro'](S2,t)
    print(f"t={t:5.1f} us  Blok-avg(echo) e={e:.3f}   Liu2023-qudit e={e2:.3f}")
# minimum number of sequential two-qutrit layers per check round (edge colouring of check graph)
import networkx as nx
G = nx.Graph([(5,0),(5,1),(5,3),(6,1),(6,2),(6,4),(7,0),(7,1),(7,2),(8,3),(8,1),(8,4)])
print("max degree =", max(dict(G.degree()).values()), "(bipartite => edge chromatic number = max degree, Konig)")
