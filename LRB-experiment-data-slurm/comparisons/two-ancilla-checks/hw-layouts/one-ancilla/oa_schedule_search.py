"""Every schedule of the paper's one-ancilla round, ranked by the exact leading-order
logical error per round (oa_pair_score.per_round).

Same layout and same 12 gates as the LRB generator's round; a schedule is the gate
order on every wire: the order of the 3 data qutrits on each ancilla (6^4) and the
order of the ancillas on each data qutrit (q1: 4!, the others 2). The generator packs
any op list ASAP (oa_pair_score.moments), so these orders fix the circuit, its depth
and its idle locations. Each order with no cycle is built as an op list and kept if
the round is correct (oa_symbolic.round_ok: deterministic, QND, right checks) and no
single fault is silent (per-round L1 = 0). It is then scored under SI1000 and SD6.

Usage: oa_schedule_search.py [WORKERS]  -> schedule_search.out, schedule_search.pkl
"""
import itertools
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import oa_pair_score as PSC
import oa_symbolic as SY

HERE = Path(__file__).resolve().parent
A1, A2, A3, A4 = 5, 6, 7, 8
CHECKS = [(A1, "Z", [(0, 1), (1, -1), (3, -1)]), (A2, "Z", [(1, 1), (2, 1), (4, -1)]),
          (A3, "X", [(0, 1), (1, 1), (2, -1)]), (A4, "X", [(3, 1), (1, -1), (4, -1)])]
TYPE = {a: t for a, t, _ in CHECKS}
COEF = {(a, q): k for a, _, qs in CHECKS for q, k in qs}
ANC_OF = {q: [a for a, _, qs in CHECKS if q in [d for d, _ in qs]] for q in range(5)}
TAGS = {A1: "G1", A2: "G2", A3: "G3", A4: "G4"}


def gate(a, q):
    name = "CNOT" if COEF[(a, q)] % 3 == 1 else "CNOT_INV"
    return ("g", name, q, a) if TYPE[a] == "Z" else ("g", name, a, q)


def topo(anc_orders, data_orders):
    """Op list for the given wire orders, or None if they contain a cycle."""
    nodes = [(a, q) for a, _, qs in CHECKS for q, _ in qs]
    succ = {n: [] for n in nodes}
    indeg = {n: 0 for n in nodes}
    for a, order in anc_orders.items():
        for u, v in zip(order, order[1:]):
            succ[(a, u)].append((a, v)); indeg[(a, v)] += 1
    for q, order in data_orders.items():
        for u, v in zip(order, order[1:]):
            succ[(u, q)].append((v, q)); indeg[(v, q)] += 1
    ready = [n for n in nodes if indeg[n] == 0]
    seq = []
    while ready:
        n = ready.pop()
        seq.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    if len(seq) < len(nodes):
        return None
    ops = [("r", a) for a in (A1, A2, A3, A4)] + [("g", "H", A3), ("g", "H", A4)]
    ops += [gate(a, q) for a, q in seq]
    ops += [("g", "H_INV", A3), ("g", "H_INV", A4)]
    ops += [("m", a, TAGS[a]) for a in (A1, A2, A3, A4)]
    return ops


def all_orders():
    anc_sets = [(a, [q for q, _ in qs]) for a, _, qs in CHECKS]
    anc_choices = [[(a, p) for p in itertools.permutations(qs)] for a, qs in anc_sets]
    data_choices = [[(q, p) for p in itertools.permutations(ANC_OF[q])] for q in range(5)]
    anc_all = [dict(asel) for asel in itertools.product(*anc_choices)]
    for dsel in itertools.product(*data_choices):
        for i in range(0, len(anc_all), 108):
            yield dict(dsel), anc_all[i:i + 108]


def _job(args):
    data_orders, anc_list = args
    out, n_acyclic, n_correct, n_safe = [], 0, 0, 0
    correct = None
    for anc_orders in anc_list:
        ops = topo(anc_orders, data_orders)
        if ops is None:
            continue
        n_acyclic += 1
        ok = SY.round_ok(ops)
        if correct is None:
            correct = ok
        elif ok != correct:
            raise RuntimeError("correctness depends on ancilla orders")
        if not ok:
            continue
        n_correct += 1
        pr = PSC.per_round(ops)
        if pr["si1000"][1] > 1e-12 or pr["sd6"][1] > 1e-12:
            continue
        n_safe += 1
        depth = sum(1 for c, _ in PSC.moments(ops) if not c)
        out.append((pr["si1000"], pr["sd6"], depth, anc_orders, data_orders, ops))
    return out, n_acyclic, n_correct, n_safe


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    jobs = list(all_orders())
    print(f"{len(jobs) // 12} data-order patterns x {12 * len(jobs[0][1])} ancilla-order patterns", flush=True)
    rows, tot = [], [0, 0, 0]
    with Pool(workers) as pool:
        for k, (out, na, nc, ns) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            rows += out
            tot[0] += na; tot[1] += nc; tot[2] += ns
            if k % 384 == 383:
                print(f"  {k + 1}/{len(jobs)} jobs done, {tot[2]} fault-safe so far", flush=True)
    print(f"acyclic {tot[0]}, correct {tot[1]}, fault-safe {tot[2]}", flush=True)
    pickle.dump(rows, open(HERE / "schedule_search.pkl", "wb"))
