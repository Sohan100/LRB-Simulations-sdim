"""Rank the zero-silent 5-moment schedules of the current one-ancilla round
(prod_reschedule_T5.out lists only 5; regenerate all 52) by silent fault pairs.

Uses prod_reschedule.schedules/ops_of and the tableau screen (oa_symbolic) instead of
sdim, then oa_fault_spread.analyse for the exact count of silent fault pairs (both on
two-qutrit gates, and all locations). Fewer pairs = lower leading-order logical error.
Usage: rank_prod5.py -> rank_prod5.out
"""
import oa_fault_spread as FSP
import oa_symbolic as SY
import prod_reschedule as P

seen, rows = set(), []
for a in P.schedules(5):
    key = P.canonical_orders(a)
    if key in seen:
        continue
    seen.add(key)
    ops = P.ops_of(a, 5)
    if not SY.round_ok(ops) or SY.silent(ops)[0]:
        continue
    _, _, _, _, pairs = FSP.analyse("x", ops)
    two = sum(c for kp, c in pairs.items() if kp == ("ancilla-data CNOT", "ancilla-data CNOT"))
    rows.append((two, sum(pairs.values()), ops))
rows.sort(key=lambda r: (r[0], r[1]))
print(f"{len(seen)} distinct 5-moment schedules, {len(rows)} correct with 0 silent single faults")
prod = FSP.analyse("x", FSP.production_ops())[4]
print(f"current generator round: {sum(c for kp, c in prod.items() if kp == ('ancilla-data CNOT', 'ancilla-data CNOT'))} "
      f"2q-gate silent pairs, {sum(prod.values())} all")
for two, allp, ops in rows:
    print(f"  {two:4d} 2q-gate pairs, {allp:4d} all  OPS {ops}")
