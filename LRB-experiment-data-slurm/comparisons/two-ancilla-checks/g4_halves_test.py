"""Zero noise: measure parts of G4 = X1^-1 X3 X4^-1 on |+_L>, then measure
the whole G1 = Z0 Z1^-1 Z3^-1 on one ancilla and see whether it stays 0.

Ancilla 8 measures the q1,q4 part (-x1 - x4), ancilla 10 measures x3,
ancilla 5 then measures all of G1.
"""
from collections import Counter

from sdim.circuit import Circuit
from sdim.program import Program
from lrb.code_definitions import FoldedQutritSplitAncillaUniformDetectionCode as Code

N, SHOTS = Code.num_qudits, 3000


def q1q4_part(c):
    c.add_gate("H", 8); c.add_gate("CNOT_INV", 8, 1); c.add_gate("CNOT_INV", 8, 4)
    c.add_gate("H_INV", 8); c.add_gate("M", 8)


def q3_part(c):
    c.add_gate("H", 10); c.add_gate("CNOT", 10, 3); c.add_gate("H_INV", 10); c.add_gate("M", 10)


def whole_g4(c):
    c.add_gate("H", 8); c.add_gate("CNOT_INV", 8, 1); c.add_gate("CNOT", 8, 3)
    c.add_gate("CNOT_INV", 8, 4); c.add_gate("H_INV", 8); c.add_gate("M", 8)


def whole_g1(c):
    c.add_gate("CNOT", 0, 5); c.add_gate("CNOT_INV", 1, 5); c.add_gate("CNOT_INV", 3, 5)
    c.add_gate("M", 5)


CASES = {
    "q1,q4 part only": [q1q4_part],
    "q3 part only": [q3_part],
    "both parts, readouts added": [q1q4_part, q3_part],
    "whole G4 on one ancilla": [whole_g4],
}
for name, steps in CASES.items():
    c = Code.logical_plus_initial_state()
    for step in steps:
        step(c)
    whole_g1(c)
    m = Program(c).simulate(shots=SHOTS)[0]
    v = lambda w, s: int(m[w][0][s].measurement_value)
    g1 = Counter(v(5, s) for s in range(SHOTS))
    extra = ""
    if len(steps) == 2:
        extra = f"   G4 from the sum {dict(Counter((v(8, s) + v(10, s)) % 3 for s in range(SHOTS)))}"
    elif name.startswith("whole"):
        extra = f"   G4 {dict(Counter(v(8, s) for s in range(SHOTS)))}"
    print(f"{name:28s} then G1: {dict(sorted(g1.items()))}{extra}")
