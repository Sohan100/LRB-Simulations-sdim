"""Shallowest schedules of the paper's one-ancilla round (same layout, same 12 gates).

The LRB generator's round (noise_bench.production_round) packs its op list ASAP into
7 gate moments: H(a3) H(a4) | 4 CNOT layers ... | H_INV(a3) | H_INV(a4). q1 sits in
all four checks, so 4 CNOT layers are the minimum, and the X ancillas need one H and
one H_INV moment around their three CNOTs, so 5 gate moments is the lower bound.

This script enumerates every assignment of the 12 CNOTs to gate moments 1..T (each
qutrit at most one gate per moment; X ancillas: H in a moment before their first
CNOT, H_INV after the last), keeps those whose op list passes the noiseless tests of
../hw_layouts_check.py (zero noise 5 rounds, 40 single data Paulis: deterministic,
QND, correct syndrome), and fault-checks them exhaustively (single faults, 2 rounds,
with and without the H-bar sandwich). Readout and reset layers are unchanged.

Usage: prod_reschedule.py T [MAX_CHECKED]
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
import hw_layouts_check as H  # noqa: E402
import protocol_search as PS  # noqa: E402

A1, A2, A3, A4 = 5, 6, 7, 8
# (ancilla, type, [(data, coeff)]) exactly as in the LRB generator round
CHECKS = [(A1, "Z", [(0, 1), (1, -1), (3, -1)]), (A2, "Z", [(1, 1), (2, 1), (4, -1)]),
          (A3, "X", [(0, 1), (1, 1), (2, -1)]), (A4, "X", [(3, 1), (1, -1), (4, -1)])]


def cn(k):
    return "CNOT" if k % 3 == 1 else "CNOT_INV"


def gate(anc, typ, q, k):
    return ("g", cn(k), q, anc) if typ == "Z" else ("g", cn(k), anc, q)


def schedules(T):
    """Yield {(anc, q): moment} for every valid packing into T gate moments."""
    items = [(anc, typ, q, k) for anc, typ, qs in CHECKS for q, k in qs]
    lo = {it: (2 if it[1] == "X" else 1) for it in items}
    hi = {it: (T - 1 if it[1] == "X" else T) for it in items}
    assign = {}
    busy = set()

    def rec(i):
        if i == len(items):
            yield dict(assign)
            return
        anc, typ, q, k = items[i]
        for t in range(lo[items[i]], hi[items[i]] + 1):
            if (anc, t) in busy or (q, t) in busy:
                continue
            busy.add((anc, t)); busy.add((q, t))
            assign[(anc, q)] = t
            yield from rec(i + 1)
            busy.discard((anc, t)); busy.discard((q, t))
            del assign[(anc, q)]

    yield from rec(0)


def ops_of(assign, T):
    ops = [("r", a) for a in (A1, A2, A3, A4)]
    for t in range(1, T + 1):
        for anc, typ, qs in CHECKS:
            ts = [assign[(anc, q)] for q, _ in qs]
            if typ == "X" and t == min(ts) - 1:
                ops.append(("g", "H", anc))
            for q, k in qs:
                if assign[(anc, q)] == t:
                    ops.append(gate(anc, typ, q, k))
            if typ == "X" and t == max(ts) + 1:
                ops.append(("g", "H_INV", anc))
    tags = {A1: "G1", A2: "G2", A3: "G3", A4: "G4"}
    ops += [("m", a, tags[a]) for a in (A1, A2, A3, A4)]
    return ops


def canonical_orders(assign):
    """Gate order on every wire: per ancilla the data order, per data qutrit the ancilla
    order. Schedules with equal keys are the same circuit up to idle placement."""
    per_anc = tuple(tuple(q for q, _ in sorted(qs, key=lambda qk: assign[(anc, qk[0])])) for anc, _, qs in CHECKS)
    per_q = tuple(tuple(a for (a, qq), t in sorted(assign.items(), key=lambda kv: kv[1]) if qq == q) for q in range(5))
    return per_anc, per_q


def _test(args):
    ops, T = args
    lines = H.noiseless_tests("x", ops)
    ok = all("OK" in ln for ln in lines if ln.strip().startswith(("zero", "40")))
    if not ok:
        return ops, None
    _, fl = H.fault_test(("x", ops))
    silent = [int(ln.split(" faults, ")[1].split(" silent")[0]) for ln in fl]
    return ops, (fl, silent)


if __name__ == "__main__":
    T = int(sys.argv[1])
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    seen, jobs = set(), []
    for a in schedules(T):
        key = canonical_orders(a)
        if key in seen:
            continue
        seen.add(key)
        jobs.append((ops_of(a, T), T))
    print(f"T={T}: {len(jobs)} distinct schedules with {T} gate moments", flush=True)
    jobs = jobs[:cap]
    good = []
    with Pool(64) as pool:
        for ops, res in pool.imap_unordered(_test, jobs):
            if res is None:
                continue
            fl, silent = res
            if sum(silent) == 0:
                good.append(ops)
                if len(good) <= 5:
                    print(f"  zero-silent: {ops}", flush=True)
                    for ln in fl:
                        print("   ", ln, flush=True)
    print(f"T={T}: {len(good)} of {len(jobs)} checked schedules pass noiseless tests with 0 silent single faults",
          flush=True)
