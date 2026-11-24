"""Search A1 check rounds in dimension d (2 or 3) and keep the fault-tolerant ones.

Same protocol search as encoder_layout_search.py (reads of touched qudits
interleaved with push/pop of data CNOTs on the safe data couplers q0-q1,
q1-q2, q2-q3, Heisenberg accumulation), written for any prime d. Rounds are
assembled exactly as the generator builds them for layout A1: reset of both
ancillas, G1 (N) and G2 (E) gates, readouts N, E, then G3 (N) and G4 (E)
gates, readouts N, E (no reset between the two readouts of an ancilla).

Each candidate round gets the single-fault enumeration of
scripts/check_a1_single_faults.py (worktree a1-square-layout): every Pauli
after every operation, before and after each readout, with and without a
logical-H sandwich. Prints the safe combinations, best (fewest gate moments)
first.

Usage: a1_round_search_d.py D [--slack=N] [--processes=16]
"""
import heapq
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

WT = Path("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/a1-square-layout")
sys.path.insert(0, str(WT / "src"))
from sdim.circuit import Circuit  # noqa: E402
from sdim.program import Program  # noqa: E402
from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_definitions import (FoldedQubitA1SquareDetectionCode,  # noqa: E402
                                  FoldedQutritA1SquareDetectionCode)

E, N = 5, 6
CHECKS = {"G1": ("Z", (1, -1, 0, -1, 0)), "G2": ("Z", (0, 1, 1, 0, -1)),
          "G3": ("X", (1, 1, -1, 0, 0)), "G4": ("X", (0, -1, 0, 1, -1))}
TOUCH = {"G1": (N, (2,)), "G2": (E, (1, 4)), "G3": (N, (2,)), "G4": (E, (1, 4))}
SAFE = [(0, 1), (1, 2), (2, 3)]
GEN = LRBCircuitGenerator(noise_model="si1000")


def gates(d):
    names = ("CNOT",) if d == 2 else ("CNOT", "CNOT_INV")
    return [(nm, c, t) for a, b in SAFE for c, t in ((a, b), (b, a)) for nm in names]


def conj(kind, gate, d):
    name, c, t = gate
    s = 1 if name == "CNOT" else -1
    D = np.eye(5, dtype=int)
    if kind == "Z":
        D[c, t] = s
    else:
        D[t, c] = -s
    return D % d


def inverse(gate, d):
    name, c, t = gate
    if d == 2:
        return gate
    return ("CNOT_INV" if name == "CNOT" else "CNOT", c, t)


def search(kind, target, touch, d, max_pushes=4, max_cost=16, slack=0):
    target = tuple(int(v) % d for v in target)
    best, sols, seen = None, [], {}
    tie = itertools.count()
    heap = [(0, 0, ((), (0,) * 5), ())]
    G = gates(d)
    while heap:
        cost, _, (stack, acc), ops = heapq.heappop(heap)
        if best is not None and cost > best + slack:
            break
        if not stack and acc == target and ops:
            best = cost if best is None else best
            sols.append((cost, ops))
            continue
        if seen.get((stack, acc), 99) < cost:
            continue
        M = np.eye(5, dtype=int)
        for g in stack:
            M = (M @ conj(kind, g, d)) % d
        nxt = []
        last = ops[-1] if ops else None
        for q in touch:
            if last is not None and last[0] == "read" and last[1] == q:
                continue
            for e in range(1, d):
                v = np.zeros(5, dtype=int)
                v[q] = e
                acc2 = tuple(((np.array(acc) + M @ v) % d).tolist())
                nxt.append((cost + 1, (stack, acc2), ops + (("read", q, e),)))
        if len(stack) < max_pushes:
            for g in G:
                if stack and g == inverse(stack[-1], d):
                    continue
                nxt.append((cost + 1, (stack + (g,), acc), ops + (("push",) + g,)))
        if stack:
            nxt.append((cost + 1, (stack[:-1], acc), ops + (("pop",) + inverse(stack[-1], d),)))
        for c2, st, o2 in nxt:
            if c2 > max_cost:
                continue
            done = not st[0] and st[1] == target
            if seen.get(st, 99) <= c2 and not done:
                continue
            seen[st] = min(seen.get(st, 99), c2)
            heapq.heappush(heap, (c2, next(tie), st, o2))
    return sols


def to_gates(kind, anc, proto, d):
    out = [("H", anc)] if kind == "X" else []
    for step in proto:
        if step[0] == "read":
            _, q, e = step
            name = "CNOT" if e == 1 else "CNOT_INV"
            out.append((name, q, anc) if kind == "Z" else (name, anc, q))
        else:
            out.append(tuple(step[1:]))
    if kind == "X":
        out.append(("H_INV", anc))
    return out


def round_circuit(d, g1, g2, g3, g4):
    c = Circuit(dimension=d, num_qudits=7)
    c.add_gate("RESET", E)
    c.add_gate("RESET", N)
    for blk in (g1, g2):
        for g in blk:
            c.add_gate(g[0], *g[1:])
    c.add_gate("M", N)
    c.add_gate("M", E)
    for blk in (g3, g4):
        for g in blk:
            c.add_gate(g[0], *g[1:])
    c.add_gate("M", N)
    c.add_gate("M", E)
    return c


def code(d):
    return FoldedQubitA1SquareDetectionCode if d == 2 else FoldedQutritA1SquareDetectionCode


def paulis(d, n):
    return [p for p in itertools.product(range(d), repeat=2 * n) if any(p)]


def sites(rc, d):
    out = []
    for i, op in enumerate(rc.operations):
        w = tuple(GEN._operation_qudits(op))
        if op.gate_name == "M":
            out += [(i, "before", w, (k, 0)) for k in range(1, d)]
            out += [(i, "after", w, p) for p in paulis(d, 1)]
        else:
            out += [(i, "after", w, p) for p in paulis(d, len(w))]
    return out


def silent(args):
    d, rc_ops, sandwich, site = args
    C = code(d)
    rc = Circuit(dimension=d, num_qudits=7)
    for nm, w in rc_ops:
        rc.add_gate(nm, *w)
    c = C.logical_plus_initial_state()
    if sandwich:
        c = c + C.single_qudit_logical_gate_circuit("H")
    faulty = Circuit(dimension=d, num_qudits=7)
    for i, (nm, w) in enumerate(rc_ops):
        if site and site[0] == i and site[1] == "before":
            add(faulty, site[2], site[3])
        faulty.add_gate(nm, *w)
        if site and site[0] == i and site[1] == "after":
            add(faulty, site[2], site[3])
    c = c + faulty + rc
    if sandwich:
        c = c + C.single_qudit_logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H_INV", q)
        c.add_gate("M", q)
    m = Program(c).simulate(shots=2)[0]
    v = lambda w, k: int(m[w][k][0].measurement_value) % d
    anc = any(v(w, k) for w in (E, N) for k in range(len(m[w])))
    x = [v(q, 0) for q in range(5)]
    dat = (x[0] + x[1] - x[2]) % d or (-x[1] + x[3] - x[4]) % d
    return bool((x[0] + x[3]) % d and not anc and not dat)


def add(c, wires, exps):
    for j, q in enumerate(wires):
        for _ in range(exps[2 * j]):
            c.add_gate("X", q)
        for _ in range(exps[2 * j + 1]):
            c.add_gate("Z", q)


def ops_of(rc):
    return [(op.gate_name, tuple(GEN._operation_qudits(op))) for op in rc.operations]


if __name__ == "__main__":
    d = int(sys.argv[1])
    slack = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--slack=")), "0"))
    procs = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--processes=")), "16"))
    per = {}
    for chk, (kind, vec) in CHECKS.items():
        anc, touch = TOUCH[chk]
        sols = search(kind, vec, touch, d, slack=slack)
        per[chk] = [to_gates(kind, anc, proto, d) for _, proto in sols]
        print(f"d={d} {chk}: {len(sols)} protocols, two-qudit gates {sols[0][0]}", flush=True)
    combos = list(itertools.product(*(range(len(per[c])) for c in CHECKS)))
    print(f"{len(combos)} candidate rounds", flush=True)
    safe = []
    with Pool(procs) as pool:
        for combo in combos:
            rc = round_circuit(d, *(per[c][i] for c, i in zip(CHECKS, combo)))
            ops = ops_of(rc)
            jobs = [(d, ops, sw, s) for sw in (False, True) for s in sites(rc, d)]
            # noiseless sanity: no fault, nothing fires and logical is right
            assert not silent((d, ops, False, None))
            bad = sum(pool.map(silent, jobs, chunksize=16))
            mom = GEN.schedule_moments(rc)
            ng = sum(1 for col, _ in mom if not col)
            print(f"combo {combo}: {len(jobs)} faults, {bad} silent, {ng} gate moments", flush=True)
            if bad == 0:
                safe.append((ng, combo, ops))
    safe.sort()
    print(f"\n{len(safe)} of {len(combos)} rounds have 0 silent single faults", flush=True)
    for ng, combo, ops in safe[:3]:
        print(f"best {combo} ({ng} gate moments):")
        for chk, i in zip(CHECKS, combo):
            print(f"  {chk}: {per[chk][i]}")
