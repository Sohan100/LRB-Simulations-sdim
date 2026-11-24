"""Can the square-lattice layout run the paper's LRB-D versions?

Layout: a = wire 5 on {q0,q1,q2}, c = wire 6 on {q1,q2,q4}, data coupler q2-q3
(layout_candidates.SQUARE). Checks:
  1. every two-qutrit gate of each protocol piece sits on a coupler;
  2. the paper's encoder (data CNOTs 2-1, 1-0, 3-0, 4-2) does not, so a routed
     encoder through the ancillas is built and compared with it exactly
     (statevector), and preparation by measurement is also tested;
  3. noiseless sdim runs of random mirror sequences of logical Cliffords (H, H_INV,
     P, P_INV) for the terminal-check version (no check rounds) and for interval
     checks k = 1, 2, 3: every ancilla readout 0, terminal G3 = G4 = 0, and logical
     x0 + x3 = 0.
"""
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from sdim.circuit import Circuit  # noqa: E402
from sdim.program import Program  # noqa: E402
from lrb.code_definitions import FoldedQutritDetectionCode as CODE  # noqa: E402
import layout_candidates as L  # noqa: E402
from hw_layouts_check import g  # noqa: E402

A, C = 5, 6
COUPLERS = {frozenset(e) for e in [(A, 0), (A, 1), (A, 2), (C, 1), (C, 2), (C, 4), (2, 3)]}
SQUARE = L.SQUARE
Z_HALF = SQUARE[:SQUARE.index(("r", C)) + 1]

# Routed encoder: same unitary as CODE.logical_plus_initial_state on the data,
# ancillas start and end in |0>. CNOT 3->0 is done first, while q2 is still |0>,
# using q2 and a as a bus; the other data CNOTs go through a or c.
ROUTED_ENCODER = [
    g("H_INV", 3), g("H_INV", 4),
    g("CNOT", 3, 2), g("CNOT", 2, A), g("CNOT", A, 0), g("CNOT_INV", 2, A), g("CNOT_INV", 3, 2),  # CNOT 3->0
    g("H_INV", 2),
    g("CNOT", 2, A), g("CNOT_INV", A, 1), g("CNOT_INV", 2, A),                                     # CNOT_INV 2->1
    g("CNOT", 1, A), g("CNOT", A, 0), g("CNOT_INV", 1, A),                                         # CNOT 1->0
    g("CNOT", 4, C), g("CNOT", C, 2), g("CNOT_INV", 4, C),                                         # CNOT 4->2
]


def pairs_of_circuit(c):
    out = set()
    for o in c.operations:
        w = L_GEN._operation_qudits(o)
        if len(w) == 2:
            out.add(frozenset(w))
    return out


def pairs_of_ops(ops):
    return {frozenset(op[2:]) for op in ops if op[0] == "g" and len(op) == 4}


def coupler_report():
    pieces = {"paper encoder (logical_plus_initial_state)": pairs_of_circuit(CODE.logical_plus_initial_state()),
              "routed encoder": pairs_of_ops(ROUTED_ENCODER),
              "check round (SQUARE)": pairs_of_ops(SQUARE),
              "terminal X readout": pairs_of_circuit(CODE.terminal_const0_logical_x_measurement_circuit())}
    for tok in ("H", "H_INV", "P", "P_INV", "X", "Z"):
        pieces[f"logical {tok}"] = pairs_of_circuit(CODE.single_qudit_logical_gate_circuit(tok))
    for name, prs in pieces.items():
        bad = sorted(tuple(sorted(p)) for p in prs if p not in COUPLERS)
        print(f"  {name:44s} two-qutrit pairs {sorted(tuple(sorted(p)) for p in prs)}  "
              f"{'all on couplers' if not bad else 'NOT on couplers: ' + str(bad)}")


def encoder_statevector_check():
    import hw_layouts_statevector as SV
    psi = np.zeros((3,) * SV.NW, dtype=complex)
    psi[(0,) * SV.NW] = 1
    ref = psi.copy()
    for o in CODE.logical_plus_initial_state().operations:
        w = L_GEN._operation_qudits(o)
        ref = SV.gate(ref, o.gate_name, *w)
    for op in ROUTED_ENCODER:
        psi = SV.gate(psi, op[1], *op[2:])
    print(f"  routed encoder vs paper encoder: |<ref|psi>|^2 = {abs(np.vdot(ref, psi)) ** 2:.12f} "
          f"(ancillas back in |0>: {np.isclose(np.sum(np.abs(psi[(slice(None),) * 5 + (0,) * 4]) ** 2), 1)})")


def build(prep, tokens, k, shots):
    """prep: 'paper', 'routed' or 'measure'. k: None (terminal check only) or interval."""
    c = Circuit(dimension=3, num_qudits=9)
    labels, count = [], Counter()

    def add_ops(ops, tag):
        nonlocal c
        for op in ops:
            if op[0] == "g":
                c.add_gate(op[1], *op[2:])
            elif op[0] == "m":
                c.add_gate("M", op[1])
                labels.append((tag, op[2], op[1], count[op[1]]))
                count[op[1]] += 1
            else:
                c.add_gate("RESET", op[1])
                count[op[1]] += 1

    if prep == "paper":
        c = c + CODE.logical_plus_initial_state()
    elif prep == "routed":
        add_ops(ROUTED_ENCODER, "enc")
    else:
        for q in range(5):
            c.add_gate("H", q)
        add_ops(Z_HALF, "prep")
    seq = tokens + [t[:-4] if t.endswith("_INV") else t + "_INV" for t in reversed(tokens)]
    for i, tok in enumerate(seq, 1):
        c = c + CODE.single_qudit_logical_gate_circuit(tok)
        if k is not None and i % k == 0:
            add_ops(SQUARE, f"round{i}")
    c = c + CODE.terminal_const0_logical_x_measurement_circuit()
    for q in range(5):
        labels.append(("final", "data", q, count[q]))
        count[q] += 1
    res = Program(c).simulate(shots=shots)[0]
    vals = {lab: [int(res[lab[2]][lab[3]][s].measurement_value) for s in range(shots)] for lab in labels}
    ok, kept = 0, 0
    for s in range(shots):
        if prep == "measure" and any(v[s] for lab, v in vals.items() if lab[0] == "prep"):
            continue
        kept += 1
        x = [vals[("final", "data", q, count[q] - 1)][s] for q in range(5)]
        anc = [v[s] for lab, v in vals.items() if lab[0] not in ("final", "prep")]
        good = (not any(anc) and (x[0] + x[1] - x[2]) % 3 == 0 and (-x[1] + x[3] - x[4]) % 3 == 0
                and (x[0] + x[3]) % 3 == 0)
        ok += good
    return ok, kept


if __name__ == "__main__":
    import protocol_search as PS
    L_GEN = PS.GEN
    print("two-qutrit gates vs the layout's couplers:")
    coupler_report()
    print("\nencoder:")
    encoder_statevector_check()
    print("\nnoiseless mirror sequences (20 random sequences of 6 logical Cliffords + inverse):")
    rng = random.Random(1116)
    seqs = [[rng.choice(["H", "H_INV", "P", "P_INV"]) for _ in range(6)] for _ in range(20)]
    for prep in ("paper", "routed", "measure"):
        for k in (None, 1, 2, 3):
            tot_ok = tot_kept = 0
            for tokens in seqs:
                ok, kept = build(prep, tokens, k, shots=60 if prep == "measure" else 8)
                tot_ok += ok
                tot_kept += kept
            name = "terminal check only" if k is None else f"interval k={k}"
            print(f"  prep {prep:8s} {name:20s}: {tot_ok}/{tot_kept} kept shots pass"
                  + (" (prep readouts postselected on 0)" if prep == "measure" else ""), flush=True)
