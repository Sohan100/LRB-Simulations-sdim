"""Zero-noise tests of scheduled check rounds on the four conventional sites.

Wires: data q0..q4 = 0..4, sites N = 5, W = 6, E = 7, S = 8 (9 qutrits).
Every round: step 1 measures G3 on N and G2 on E; the sites are reset; step 2
measures G1 on N + W and G4 on E + S; all sites are reset. Two rounds, then
the terminal logical-X readout.

Step-2 variants:
- separate readout: each site measures its part, readouts added
  (several gate orders, including all couplings before any measurement);
- data relay: W passes -z3 to N through q0, S passes x3 to E through q4,
  with data-ancilla gates only; N and E read the whole stabilizer, W and S
  must read 0.
Optionally sandwich the rounds between logical H and H^-1 so the logical
Z information is tested as well.
"""
from __future__ import annotations

import itertools
from collections import Counter

from sdim.circuit import Circuit
from sdim.program import Program
from lrb.code_definitions import FoldedQutritDetectionCode as Code

NQ, SHOTS = 9, 3000
N, W, E, S = 5, 6, 7, 8


class Builder:
    """Circuit plus bookkeeping of which result index each readout lands on
    (SDIM records a result for RESET as well as for M)."""

    def __init__(self, circuit: Circuit):
        self.c = circuit
        self.count = Counter()
        self.reads: dict[str, list[tuple[int, int]]] = {}

    def g(self, name, *wires):
        self.c.add_gate(name, *wires)

    def m(self, label, wire):
        self.c.add_gate("M", wire)
        self.reads.setdefault(label, []).append((wire, self.count[wire]))
        self.count[wire] += 1

    def reset(self, *wires):
        for w in wires:
            self.c.add_gate("RESET", w)
            self.count[w] += 1

    def add(self, other: Circuit):
        self.c = self.c + other


# ----- step 1 (one site each) -----
def g3_on_n(b):
    b.g("H", N); b.g("CNOT", N, 0); b.g("CNOT", N, 1); b.g("CNOT_INV", N, 2); b.g("H_INV", N)
    b.m("G3", N)


def g2_on_e(b):
    b.g("CNOT", 1, E); b.g("CNOT", 2, E); b.g("CNOT_INV", 4, E); b.m("G2", E)


# ----- step 2, separate readout: coupling pieces -----
def n_q0(b): b.g("CNOT", 0, N)
def n_q1(b): b.g("CNOT_INV", 1, N)
def w_q3(b): b.g("CNOT_INV", 3, W)
def e_q1(b): b.g("CNOT_INV", E, 1)
def e_q4(b): b.g("CNOT_INV", E, 4)
def s_q3(b): b.g("CNOT", S, 3)


def separate(order_q1: str, order_q3: str, measure_between: str | None):
    """order_q1 / order_q3: which site couples to q1 / q3 first.
    measure_between: None = all couplings, then all readouts;
    'G1' / 'G4' = finish that stabilizer (couple + read) before the other."""
    def step2(b):
        b.g("H", E); b.g("H", S)
        if measure_between == "G1":
            n_q0(b); n_q1(b); w_q3(b); b.m("G1a", N); b.m("G1b", W)
            e_q1(b); e_q4(b); s_q3(b)
        elif measure_between == "G4":
            e_q1(b); e_q4(b); s_q3(b)
            b.g("H_INV", E); b.g("H_INV", S); b.m("G4a", E); b.m("G4b", S)
            n_q0(b); n_q1(b); w_q3(b); b.m("G1a", N); b.m("G1b", W)
            return
        else:
            n_q0(b); e_q4(b)
            for site in order_q1:
                (n_q1 if site == "N" else e_q1)(b)
            for site in order_q3:
                (w_q3 if site == "W" else s_q3)(b)
            b.m("G1a", N); b.m("G1b", W)
        b.g("H_INV", E); b.g("H_INV", S); b.m("G4a", E); b.m("G4b", S)
    return step2


# ----- step 2, relay through a shared data qutrit -----
def data_relay(transfer: str):
    def step2(b):
        # G1: W collects -z3, lends it to q0, N reads z0 - z1 - z3, then undo.
        b.g("CNOT_INV", 3, W); b.g("CNOT", W, 0)
        b.g("CNOT", 0, N); b.g("CNOT_INV", 1, N)
        b.g("CNOT_INV", W, 0); b.g("CNOT", 3, W)
        b.m("G1a", N); b.m("G1b", W)
        # G4: S collects x3, lends it to q4, E reads -x1 - x4 + x3, then undo.
        inverse = {"CNOT": "CNOT_INV", "CNOT_INV": "CNOT"}[transfer]
        b.g("H", E); b.g("H", S)
        b.g("CNOT", S, 3); b.g(transfer, 4, S)
        b.g("CNOT_INV", E, 1); b.g("CNOT_INV", E, 4)
        b.g(inverse, 4, S); b.g("CNOT_INV", S, 3)
        b.g("H_INV", E); b.g("H_INV", S)
        b.m("G4a", E); b.m("G4b", S)
    return step2


def run(step2, rounds=2, sandwich=False, error=None, shots=SHOTS):
    b = Builder(Code.logical_plus_initial_state())
    if error:
        b.g(error[1], error[0])
    if sandwich:
        b.add(Code.single_qudit_logical_gate_circuit("H"))
    for _ in range(rounds):
        g3_on_n(b); g2_on_e(b); b.reset(N, E)
        step2(b); b.reset(N, W, E, S)
    if sandwich:
        b.add(Code.single_qudit_logical_gate_circuit("H_INV"))
    b.g("H_INV", 0); b.g("H_INV", 3); b.m("x0", 0); b.m("x3", 3)
    m = Program(b.c).simulate(shots=shots)[0]
    val = lambda label, r, s: int(m[b.reads[label][r][0]][b.reads[label][r][1]][s].measurement_value)
    out = {"G1": [], "G2": [], "G3": [], "G4": []}
    for r in range(rounds):
        out["G1"].append([(val("G1a", r, s) + val("G1b", r, s)) % 3 for s in range(shots)])
        out["G4"].append([(val("G4a", r, s) + val("G4b", r, s)) % 3 for s in range(shots)])
        out["G2"].append([val("G2", r, s) for s in range(shots)])
        out["G3"].append([val("G3", r, s) for s in range(shots)])
    relay_zero = [all(val(lbl, r, s) == 0 for r in range(rounds) for lbl in ("G1b", "G4b"))
                  for s in range(shots)]
    xbar = [(val("x0", 0, s) + val("x3", 0, s)) % 3 for s in range(shots)]
    passed = [s for s in range(shots)
              if all(out[g][r][s] == 0 for g in out for r in range(rounds))]
    return out, passed, xbar, relay_zero


def summary(name, step2, **kw):
    out, passed, xbar, relay_zero = run(step2, **kw)
    rnds = " | ".join(
        f"R{r + 1} " + " ".join(f"{g}:{len(set(out[g][r]))}" for g in ("G1", "G2", "G3", "G4"))
        for r in range(len(out["G1"])))
    xb = dict(sorted(Counter(xbar[s] for s in passed).items()))
    print(f"{name:44s} pass {len(passed):4d}/{SHOTS}  X-bar among passed {xb}   "
          f"[distinct values per check: {rnds}]")
    return out, passed, xbar, relay_zero


if __name__ == "__main__":
    print("== separate readout, scheduled on reused sites ==")
    summary("G1 read, then G4", separate("", "", "G1"))
    summary("G4 read, then G1", separate("", "", "G4"))
    for q1, q3 in itertools.product(("NE", "EN"), ("WS", "SW")):
        summary(f"all couplings first, q1 {q1[0]}->{q1[1]}, q3 {q3[0]}->{q3[1]}",
                separate(q1, q3, None))

    print("\n== relay through a shared data qutrit, scheduled ==")
    for transfer in ("CNOT", "CNOT_INV"):
        summary(f"transfer {transfer} q4->S", data_relay(transfer))
