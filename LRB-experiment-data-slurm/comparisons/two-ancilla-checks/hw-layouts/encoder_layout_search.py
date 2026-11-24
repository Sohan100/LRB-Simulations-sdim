"""Check circuits for a square-lattice chip that also runs the paper's encoder.

The encoder's data CNOTs (2-1, 1-0, 3-0, 4-2) plus the logical gates' q2-q3 make
the data coupling graph a plaquette q0-q1-q2-q3 with q4 on q2. On a square
lattice no empty site touches three data qutrits, so every check moves parity
through data-data gates ("difference trick"). Inside a check only the data pairs
without a weight-2 logical may be used: q0-q1, q1-q2, q2-q3 (X0X3 and X2X4 are
logicals, so one fault on a q0-q3 or q2-q4 gate is a silent logical error).

Single-ancilla protocol search: an ancilla touching data set T performs reads of
touched qutrits (Z_q^e or X_q^e) interleaved with pushes of data CNOTs on the safe
edges and LIFO pops (undo). Reading P after data unitary U measures U^dag P U, so
the readings add up (Heisenberg picture); a protocol is valid when the stack is
empty and the accumulated operator equals the check. Uniform-cost search lists all
minimal protocols; the fault checker then picks hook-safe ones.
"""
import heapq
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CHECKS = {"G1": ("Z", (1, -1, 0, -1, 0)), "G2": ("Z", (0, 1, 1, 0, -1)),
          "G3": ("X", (1, 1, -1, 0, 0)), "G4": ("X", (0, -1, 0, 1, -1))}
SAFE = [(0, 1), (1, 2), (2, 3)]
GATES = [(name, c, t) for a, b in SAFE for c, t in ((a, b), (b, a)) for name in ("CNOT", "CNOT_INV")]


def conj_matrix(kind, gate):
    """Matrix D with V^dag P(v) V = P(D v) for exponent vectors (columns)."""
    name, c, t = gate
    s = 1 if name == "CNOT" else -1
    D = np.eye(5, dtype=int)
    if kind == "Z":      # V^dag Z_t V = Z_c^s Z_t
        D[c, t] = s
    else:                # V^dag X_c V = X_c X_t^-s
        D[t, c] = -s
    return D % 3


def inverse(gate):
    name, c, t = gate
    return ("CNOT_INV" if name == "CNOT" else "CNOT", c, t)


def search(kind, target, touch, max_pushes=3, max_cost=14, slack=0):
    target = np.array(target) % 3
    start = (tuple(), tuple([0] * 5))
    best = None
    sols = []
    heap = [(0, 0, start, ())]
    seen = {}
    tie = itertools.count()
    while heap:
        cost, _, (stack, acc), ops = heapq.heappop(heap)
        if best is not None and cost > best + slack:
            break
        if not stack and np.array_equal(np.array(acc), target) and ops:
            best = cost if best is None else best
            sols.append((cost, ops))
            continue
        key = (stack, acc)
        if seen.get(key, 99) < cost:
            continue
        M = np.eye(5, dtype=int)
        for gte in stack:            # cumulative M = D1 D2 ... (reading after all pushes)
            M = (M @ conj_matrix(kind, gte)) % 3
        nxt = []
        last = ops[-1] if ops else None
        for q in touch:
            if last is not None and last[0] == "read" and last[1] == q:
                continue
            for e in (1, 2):
                v = np.zeros(5, dtype=int); v[q] = e
                acc2 = tuple(((np.array(acc) + M @ v) % 3).tolist())
                nxt.append((cost + 1, (stack, acc2), ops + (("read", q, e),)))
        if len(stack) < max_pushes:
            for gte in GATES:
                if stack and gte == inverse(stack[-1]):
                    continue
                nxt.append((cost + 1, (stack + (gte,), acc), ops + (("push",) + gte,)))
        if stack:
            nxt.append((cost + 1, (stack[:-1], acc), ops + (("pop",) + inverse(stack[-1]),)))
        for c2, st, o2 in nxt:
            if c2 > max_cost:
                continue
            if seen.get(st, 99) <= c2 and not (not st[0] and np.array_equal(np.array(st[1]), target)):
                continue
            seen[st] = min(seen.get(st, 99), c2)
            heapq.heappush(heap, (c2, next(tie), st, o2))
    return sols


def to_ops(kind, anc, proto, tag):
    """Protocol -> hw_layouts_check ops."""
    ops = [("g", "H", anc)] if kind == "X" else []
    for step in proto:
        if step[0] == "read":
            _, q, e = step
            name = "CNOT" if e == 1 else "CNOT_INV"
            ops.append(("g", name, q, anc) if kind == "Z" else ("g", name, anc, q))
        else:
            ops.append(("g", step[1], step[2], step[3]))
    if kind == "X":
        ops.append(("g", "H_INV", anc))
    return ops + [("m", anc, tag), ("r", anc)]


if __name__ == "__main__":
    touch_sets = {"{1,4}": (1, 4), "{3,4}": (3, 4), "{1}": (1,), "{2}": (2,), "{0}": (0,), "{3}": (3,), "{4}": (4,)}
    for tname, T in touch_sets.items():
        for chk, (kind, vec) in CHECKS.items():
            sols = search(kind, vec, T)
            if sols:
                c = sols[0][0]
                pushes = sum(1 for s in sols[0][1] if s[0] == "push")
                print(f"touch {tname:6s} {chk}: min two-qutrit gates {c} ({pushes} data CNOT pairs), "
                      f"{len(sols)} minimal protocols; e.g. {sols[0][1]}", flush=True)
            else:
                print(f"touch {tname:6s} {chk}: none within limits", flush=True)
