"""Qubit A1: search longer G4 protocols (slack 1, 2) and fault-test full rounds.

Every minimal qubit A1 round has two silent single faults, both a Z on q2/q3
inside G4's data-CNOT frame (the residue is Z1 Z2 Z3 = G1 Zbar). G2 and G4
both contain q4, which only E reaches without q2-q4, so only G4's protocol can
change. Output: a1_qubit_g4_slack.out.
"""
import sys
from multiprocessing import Pool
import a1_round_search_d as S

d = 2
per = {}
for chk, (kind, vec) in S.CHECKS.items():
    anc, touch = S.TOUCH[chk]
    slack = 2 if chk == "G4" else 0
    sols = S.search(kind, vec, touch, d, max_pushes=4, max_cost=18, slack=slack)
    per[chk] = [(c, S.to_gates(kind, anc, p, d)) for c, p in sols]
    print(f"{chk}: {len(sols)} protocols, costs {sorted(set(c for c, _ in sols))}", flush=True)
safe = []
with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 14) as pool:
    for g1i in range(len(per["G1"])):
        for j, (c4, g4) in enumerate(per["G4"]):
            rc = S.round_circuit(d, per["G1"][g1i][1], per["G2"][0][1], per["G3"][0][1], g4)
            ops = S.ops_of(rc)
            jobs = [(d, ops, sw, s) for sw in (False, True) for s in S.sites(rc, d)]
            bad = sum(pool.map(S.silent, jobs, chunksize=16))
            mom = S.GEN.schedule_moments(rc)
            ng = sum(1 for col, _ in mom if not col)
            two = sum(1 for nm, w in ops if len(w) == 2)
            print(f"G1#{g1i} G4#{j} (G4 cost {c4}): {bad} silent of {len(jobs)}, {two} two-qubit gates, {ng} gate moments", flush=True)
            if bad == 0:
                safe.append((two, ng, g1i, j))
safe.sort()
print(f"\n{len(safe)} safe rounds", flush=True)
for two, ng, g1i, j in safe[:5]:
    print(f"best: {two} two-qubit gates, {ng} gate moments")
    print(f"  G1: {per['G1'][g1i][1]}\n  G2: {per['G2'][0][1]}\n  G3: {per['G3'][0][1]}\n  G4: {per['G4'][j][1]}")
