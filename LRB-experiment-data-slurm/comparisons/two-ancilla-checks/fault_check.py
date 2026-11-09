"""Exhaustive single-fault check of one check round, three ways of measuring.

Each fault is one Pauli X^a Z^b on one qutrit (after a one-qutrit gate, a
reset, or at the start of the round), or one two-qutrit Pauli on the pair
right after a two-qutrit gate (80 per gate). It is put in round 1 of two
rounds, followed by a noiseless readout of all five data qutrits in the X
basis. Detectors: every ancilla readout in both rounds must be 0 (the relay
ancillas included) and G3, G4 computed from the final data must be 0. A fault
is a silent logical error if no detector fires but X0 + X3 at the end is not 0.
Run on |+L> (catches logical-X flips) and with logical H before the rounds and
H^-1 after (catches logical-Z flips).

Schemes:
- one ancilla: the simulated one-ancilla round (9 qutrits);
- coherent relay: the simulated two-ancilla round (11 qutrits, CNOT 9->5 and
  CNOT_INV 8->10 between ancillas);
- data relay: four conventional sites N=5 W=6 E=7 S=8, schedule G2+G3, then G4,
  then G1, partial values lent through q4 (G4) and q0 (G1).
"""
import itertools
from collections import Counter

from sdim.program import Program
from lrb.code_definitions import (FoldedQutritDetectionCode,
                                  FoldedQutritSplitAncillaUniformDetectionCode)

PAULIS = [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]


def g(name, *w): return ("g", name, *w)
def m(w): return ("m", w)
def r(*ws): return [("r", w) for w in ws]


ONE_ANCILLA = [
    g("CNOT", 0, 5), g("CNOT_INV", 1, 5), g("CNOT_INV", 3, 5), m(5),
    g("CNOT", 1, 6), g("CNOT", 2, 6), g("CNOT_INV", 4, 6), m(6),
    g("H", 7), g("CNOT", 7, 0), g("CNOT", 7, 1), g("CNOT_INV", 7, 2), g("H_INV", 7), m(7),
    g("H", 8), g("CNOT", 8, 3), g("CNOT_INV", 8, 1), g("CNOT_INV", 8, 4), g("H_INV", 8), m(8),
    *r(5, 6, 7, 8)]

COHERENT_RELAY = [
    g("CNOT", 0, 5), g("CNOT_INV", 1, 5), g("CNOT_INV", 3, 9), g("CNOT", 9, 5),
    g("CNOT", 3, 9), m(5), m(9),
    g("CNOT", 1, 6), g("CNOT", 2, 6), g("CNOT_INV", 4, 6), m(6),
    g("H", 7), g("CNOT", 7, 0), g("CNOT", 7, 1), g("CNOT_INV", 7, 2), g("H_INV", 7), m(7),
    g("H", 8), g("H", 10), g("CNOT", 8, 3), g("CNOT_INV", 8, 1), g("CNOT_INV", 10, 4),
    g("CNOT_INV", 8, 10), g("CNOT", 10, 4), g("H_INV", 8), g("H_INV", 10), m(8), m(10),
    *r(5, 6, 7, 8, 9, 10)]

N, W, E, S = 5, 6, 7, 8
DATA_RELAY = [
    g("H", N), g("CNOT", N, 0), g("CNOT", N, 1), g("CNOT_INV", N, 2), g("H_INV", N), m(N),
    g("CNOT", 1, E), g("CNOT", 2, E), g("CNOT_INV", 4, E), m(E),
    *r(N, E),
    g("H", E), g("H", S), g("CNOT", S, 3), g("CNOT_INV", 4, S), g("CNOT_INV", E, 1),
    g("CNOT_INV", E, 4), g("CNOT", 4, S), g("CNOT_INV", S, 3), g("H_INV", E), g("H_INV", S),
    m(E), m(S),
    g("CNOT_INV", 3, W), g("CNOT", W, 0), g("CNOT", 0, N), g("CNOT_INV", 1, N),
    g("CNOT_INV", W, 0), g("CNOT", 3, W), m(N), m(W),
    *r(N, W, E, S)]

SCHEMES = {
    "one ancilla": (FoldedQutritDetectionCode, ONE_ANCILLA, set()),
    "coherent relay": (FoldedQutritSplitAncillaUniformDetectionCode, COHERENT_RELAY, {9, 10}),
    "data relay": (FoldedQutritDetectionCode, DATA_RELAY, {W, S}),
}


def fault_sites(ops, nq):
    """(index to insert after, wires) for every fault location in one round;
    index -1 is the start of the round."""
    sites = [(-1, (q,)) for q in range(nq)]
    for i, op in enumerate(ops):
        if op[0] == "g":
            sites.append((i, tuple(op[2:])))
        elif op[0] == "r":
            sites.append((i, (op[1],)))
    return sites


def faults(ops, nq):
    for i, wires in fault_sites(ops, nq):
        for combo in itertools.product([(0, 0)] + PAULIS, repeat=len(wires)):
            if any(p != (0, 0) for p in combo):
                yield i, dict(zip(wires, combo))


def build(code, ops, fault, sandwich):
    c = code.logical_plus_initial_state()
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H")
    labels, count = [], Counter()

    def apply(op, rnd):
        if op[0] == "g":
            c.add_gate(op[1], *op[2:])
        elif op[0] == "m":
            c.add_gate("M", op[1])
            labels.append((rnd, op[1], count[op[1]]))
            count[op[1]] += 1
        else:
            c.add_gate("RESET", op[1])
            count[op[1]] += 1

    def inject(paulis):
        for w, (a, b) in paulis.items():
            for _ in range(a):
                c.add_gate("X", w)
            for _ in range(b):
                c.add_gate("Z", w)

    for rnd in range(2):
        if rnd == 0 and fault and fault[0] == -1:
            inject(fault[1])
        for i, op in enumerate(ops):
            apply(op, rnd)
            if rnd == 0 and fault and fault[0] == i:
                inject(fault[1])
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H_INV", q); c.add_gate("M", q)
        labels.append(("final", q, count[q])); count[q] += 1
    return c, labels


def outcome(code, ops, relays, fault, sandwich):
    c, labels = build(code, ops, fault, sandwich)
    res = Program(c).simulate(shots=2)[0]  # shots=1 returns a flat list
    val = {lab: int(res[lab[1]][lab[2]][0].measurement_value) for lab in labels}
    x = [val[("final", q, max(k for (t, w, k) in labels if t == "final" and w == q))]
         for q in range(5)]
    anc = {lab: v for lab, v in val.items() if lab[0] != "final"}
    checks_fire = any(v for lab, v in anc.items() if lab[1] not in relays)
    relay_fire = any(v for lab, v in anc.items() if lab[1] in relays)
    data_fire = (x[0] + x[1] - x[2]) % 3 or (-x[1] + x[3] - x[4]) % 3
    logical = (x[0] + x[3]) % 3
    return checks_fire, relay_fire, bool(data_fire), logical


def main():
    print(f"{'scheme':16s} {'faults':>7s} {'silent logical':>15s} "
          f"{'silent if relay readouts ignored':>34s}  silent-logical faults")
    for name, (code, ops, relays) in SCHEMES.items():
        nq = code.num_qudits
        for sandwich in (False, True):
            ref = outcome(code, ops, relays, None, sandwich)
            assert ref == (False, False, False, 0), (name, ref)
            total, silent, silent_wo_relay, examples = 0, 0, 0, []
            for fault in faults(ops, nq):
                total += 1
                chk, rel, dat, lg = outcome(code, ops, relays, fault, sandwich)
                if lg and not chk and not dat:
                    silent_wo_relay += 1
                    if not rel:
                        silent += 1
                        examples.append(fault)
            tag = f"{name}{' (H sandwich)' if sandwich else ''}"
            ex = "; ".join(f"after op {i}: {p}" for i, p in examples[:4])
            print(f"{tag:30s} {total:5d} {silent:8d} {silent_wo_relay:20d}   {ex}")


if __name__ == "__main__":
    main()
