"""Do the square-lattice one-ancilla rounds run the paper's LRB-D versions?

Same tests as ../square_protocol_compat.py, for rounds.py SQ14 / SQ01:
  1. every two-qutrit gate of the logical Cliffords, terminal readout and check
     round sits on a layout coupler (the paper's encoder is listed for reference);
  2. noiseless sdim runs of 20 random mirror sequences of 6 logical Cliffords (H,
     H_INV, P, P_INV) + inverse, terminal check only and check rounds every
     k = 1, 2, 3 Cliffords; preparation either the paper's encoder (as in the LRB
     simulations, where preparation is ideal) or by measurement (|+>^5, one round,
     prep readouts postselected on 0). Pass: every ancilla readout 0, terminal
     G3 = G4 = 0 and logical x0 + x3 = 0.
Usage: oa_protocol_compat.py sq14 sq01
"""
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from sdim.circuit import Circuit  # noqa: E402
from sdim.program import Program  # noqa: E402
from lrb.code_definitions import FoldedQutritDetectionCode as CODE  # noqa: E402
import protocol_search as PS  # noqa: E402
import rounds as R  # noqa: E402


def couplers_of(ops):
    return {frozenset(op[2:]) for op in ops if op[0] == "g" and len(op) == 4}


def pairs_of_circuit(c):
    return {frozenset(PS.GEN._operation_qudits(o)) for o in c.operations if len(PS.GEN._operation_qudits(o)) == 2}


def coupler_report(rnd):
    coup = couplers_of(rnd) | {frozenset((2, 3))}
    pieces = {"paper encoder (reference)": pairs_of_circuit(CODE.logical_plus_initial_state()),
              "terminal X readout": pairs_of_circuit(CODE.terminal_const0_logical_x_measurement_circuit())}
    for tok in ("H", "H_INV", "P", "P_INV", "X", "Z"):
        pieces[f"logical {tok}"] = pairs_of_circuit(CODE.single_qudit_logical_gate_circuit(tok))
    for name, prs in pieces.items():
        bad = sorted(tuple(sorted(p)) for p in prs if p not in coup)
        print(f"  {name:28s} pairs {sorted(tuple(sorted(p)) for p in prs)}  "
              f"{'all on couplers' if not bad else 'NOT on couplers: ' + str(bad)}", flush=True)
    print(f"  check round couplers {sorted(tuple(sorted(p)) for p in couplers_of(rnd))}", flush=True)


def build(rnd, prep, tokens, k, shots):
    c = Circuit(dimension=3, num_qudits=9)
    labels, count = [], Counter()

    def add_ops(ops, tag):
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
    else:
        for q in range(5):
            c.add_gate("H", q)
        add_ops(rnd, "prep")
    seq = tokens + [t[:-4] if t.endswith("_INV") else t + "_INV" for t in reversed(tokens)]
    for i, tok in enumerate(seq, 1):
        c = c + CODE.single_qudit_logical_gate_circuit(tok)
        if k is not None and i % k == 0:
            add_ops(rnd, f"round{i}")
    c = c + CODE.terminal_const0_logical_x_measurement_circuit()
    for q in range(5):
        labels.append(("final", "data", q, count[q]))
        count[q] += 1
    res = Program(c).simulate(shots=shots)[0]
    vals = {lab: [int(res[lab[2]][lab[3]][s].measurement_value) for s in range(shots)] for lab in labels}
    ok = kept = 0
    for s in range(shots):
        if prep == "measure" and any(v[s] for lab, v in vals.items() if lab[0] == "prep"):
            continue
        kept += 1
        x = [vals[("final", "data", q, count[q] - 1)][s] for q in range(5)]
        anc = [v[s] for lab, v in vals.items() if lab[0] not in ("final", "prep")]
        ok += (not any(anc) and (x[0] + x[1] - x[2]) % 3 == 0 and (-x[1] + x[3] - x[4]) % 3 == 0
               and (x[0] + x[3]) % 3 == 0)
    return ok, kept


if __name__ == "__main__":
    for name in sys.argv[1:]:
        rnd = R.ROUNDS[name]
        print(f"== {name}\ntwo-qutrit gates vs the layout's couplers:", flush=True)
        coupler_report(rnd)
        rng = random.Random(1117)
        seqs = [[rng.choice(["H", "H_INV", "P", "P_INV"]) for _ in range(6)] for _ in range(20)]
        print("noiseless mirror sequences (20 random sequences of 6 logical Cliffords + inverse):", flush=True)
        for prep in ("paper", "measure"):
            for k in (None, 1, 2, 3):
                tot_ok = tot_kept = 0
                for tokens in seqs:
                    ok, kept = build(rnd, prep, tokens, k, shots=60 if prep == "measure" else 8)
                    tot_ok += ok
                    tot_kept += kept
                label = "terminal check only" if k is None else f"interval k={k}"
                print(f"  prep {prep:8s} {label:20s}: {tot_ok}/{tot_kept} kept shots pass"
                      + (" (prep readouts postselected on 0)" if prep == "measure" else ""), flush=True)
