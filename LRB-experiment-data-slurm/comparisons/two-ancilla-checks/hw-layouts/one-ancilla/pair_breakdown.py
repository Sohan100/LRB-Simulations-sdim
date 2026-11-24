"""The schedules tied at the minimum of oa_schedule_search.py, and the per-round L2 of
the generator round and PROD5B split by the noise kinds of the two faults.
Usage: pair_breakdown.py -> breakdown.out
"""
import pickle

import oa_pair_score as P
import rank_schedules as RS
import rounds as R

rows = pickle.load(open("schedule_search.pkl", "rb"))
best = sorted(rows, key=lambda r: (r[0][2], r[1][2]))[:4]
print("== the 4 schedules tied at the minimum")
for r in best:
    print(f"  SI1000 {r[0][2]:.4f} SD6 {r[1][2]:.4f}  anc orders {r[3]}  data orders {r[4]}")
    print(RS.moment_table(r[5]))
gen = P.generator_ops()
LBL = {"g2": "CNOT", "g1": "H gate", "reset": "reset", "meas": "readout", "idle": "idle (gate moment)",
       "mr_idle": "data idle (reset/readout moment)"}
for model in ("si1000", "sd6"):
    print(f"\n== {model}: per-round L2 by fault-pair kind (sum of X and Z runs, L(3)-L(2))")
    tabs = {}
    for nm, ops in (("generator", gen), ("PROD5B", R.PROD5B)):
        t = {}
        for run in "XZ":
            b3, b2 = P.breakdown(ops, 3, model, run), P.breakdown(ops, 2, model, run)
            for k in set(b3) | set(b2):
                t[k] = t.get(k, 0) + b3.get(k, 0) - b2.get(k, 0)
        tabs[nm] = t
    keys = sorted(set(tabs["generator"]) | set(tabs["PROD5B"]), key=lambda k: -tabs["generator"].get(k, 0))
    print(f"  {'pair':55s} {'generator':>10s} {'PROD5B':>10s}")
    for k in keys:
        g, b = tabs["generator"].get(k, 0), tabs["PROD5B"].get(k, 0)
        if max(abs(g), abs(b)) < 1e-3:
            continue
        print(f"  {LBL[k[0]] + ' + ' + LBL[k[1]]:55s} {g:10.3f} {b:10.3f}")
    print(f"  {'total':55s} {sum(tabs['generator'].values()):10.3f} {sum(tabs['PROD5B'].values()):10.3f}")
