"""One-stop audit of layout A1 against every constraint the user set.

A1 on the square grid (x right, y up):
          N(1,2)
  q3(0,1) q2(1,1) q4(2,1)
  q0(0,0) q1(1,0) E(2,0)
Couplers are exactly the grid-adjacent pairs of placed qutrits.

Checks:
  1. every coupler is a nearest-neighbour grid edge (distance 1, no diagonals);
  2. every two-qutrit gate of each piece (paper encoder, logical Cliffords, check
     round, terminal readout) sits on a coupler, and no gate acts on 3+ qutrits;
  3. the same on the composed circuits as run (encoder + mirror sequence + rounds
     every k Cliffords or none + terminal readout), op by op;
  4. two ancillas, each readout one stabilizer (injected single data Paulis);
  5. bipartite coupling graph (2 frequency groups);
  6. noiseless terminal-check and interval k = 1, 2, 3 runs pass.
"""
import ast
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import encoder_layout_rounds as ER  # noqa: E402
import hw_layouts_check as H  # noqa: E402
import protocol_search as PS  # noqa: E402
import square_protocol_compat as SPC  # noqa: E402
from lrb.code_definitions import FoldedQutritDetectionCode as CODE  # noqa: E402

E, N = 5, 6
NAME = {0: "q0", 1: "q1", 2: "q2", 3: "q3", 4: "q4", E: "E", N: "N"}
POS = {0: (0, 0), 1: (1, 0), E: (2, 0), 3: (0, 1), 2: (1, 1), 4: (2, 1), N: (1, 2)}
COUPLERS = {frozenset((a, b)) for a in POS for b in POS
            if a < b and abs(POS[a][0] - POS[b][0]) + abs(POS[a][1] - POS[b][1]) == 1}
QUD = PS.GEN._operation_qudits


def dist(a, b):
    return abs(POS[a][0] - POS[b][0]) + abs(POS[a][1] - POS[b][1])


def lbl(p):
    a, b = sorted(p)
    return f"{NAME[a]}-{NAME[b]}"


def pairs_circuit(c):
    two, big = Counter(), 0
    for o in c.operations:
        w = QUD(o)
        if len(w) == 2:
            two[frozenset(w)] += 1
        elif len(w) > 2:
            big += 1
    return two, big


def pairs_ops(ops):
    return Counter(frozenset(op[2:]) for op in ops if op[0] == "g" and len(op) == 4)


def verdict(two, big=0):
    bad = [p for p in two if p not in COUPLERS]
    far = [p for p in two if all(q in POS for q in p) and dist(*p) != 1]
    ok = not bad and not far and not big
    return ok, ("OK" if ok else f"FAIL off-coupler {[lbl(p) for p in bad]} 3+qutrit ops {big}")


def a1_round():
    per = ER.candidates("A1", 0)
    return ER.grouped(per["G1"][0], per["G2"][0], per["G3"][0], per["G4"][1])


def bipartite():
    adj = {q: set() for q in POS}
    for a, b in map(tuple, COUPLERS):
        adj[a].add(b)
        adj[b].add(a)
    col = {0: 0}
    todo = [0]
    while todo:
        u = todo.pop()
        for v in adj[u]:
            if v not in col:
                col[v] = 1 - col[u]
                todo.append(v)
            elif col[v] == col[u]:
                return False, None
    return True, col


if __name__ == "__main__":
    allok = True
    print("1. placement and couplers")
    for y in (2, 1, 0):
        row = {POS[q][0]: NAME[q] for q in POS if POS[q][1] == y}
        print("     " + "".join(f"{row.get(x, '.'):>4s}" for x in range(3)))
    for p in sorted(COUPLERS, key=lambda p: sorted(p)):
        print(f"     coupler {lbl(p):6s} grid distance {dist(*p)}")
    print(f"   {len(POS)} qutrits, {len(COUPLERS)} couplers, all nearest-neighbour grid edges "
          f"(diagonals: {sum(1 for p in COUPLERS if dist(*p) != 1)})")

    rnd = a1_round()
    txt = (HERE / "encoder_A1_circuits.txt").read_text()
    print("\n2. two-qutrit gates of each piece vs couplers")
    pieces = {"paper encoder (state prep)": pairs_circuit(CODE.logical_plus_initial_state()),
              "terminal readout (const0)": pairs_circuit(CODE.terminal_const0_logical_x_measurement_circuit()),
              "terminal readout (plain)": pairs_circuit(CODE.terminal_logical_x_measurement_circuit()),
              "A1 check round": (pairs_ops(rnd), 0)}
    for tok in ("H", "H_INV", "P", "P_INV", "X", "Z"):
        pieces[f"logical {tok}"] = pairs_circuit(CODE.single_qudit_logical_gate_circuit(tok))
    for name, (two, big) in pieces.items():
        ok, v = verdict(two, big)
        allok &= ok
        uses = ", ".join(f"{lbl(p)} x{n}" for p, n in sorted(two.items(), key=lambda t: sorted(t[0]))) or "none"
        print(f"   {name:28s} {v:4s}  pairs: {uses}")
    enc_ops = [(o.gate_name, QUD(o)) for o in CODE.logical_plus_initial_state().operations]
    print("   paper encoder gates, with grid positions:")
    for gname, w in enc_ops:
        where = "  ".join(f"{NAME[q]}{POS[q]}" for q in w)
        print(f"     {gname:9s} on {where}" + (f"   (distance {dist(*w)})" if len(w) == 2 else ""))
    saved = [op for line in txt.splitlines() if line[:1] == "G"
             for op in ast.literal_eval(line.split(" ", 1)[1])]
    in_round = [tuple(op[1:]) for op in rnd if op[0] == "g"]
    same = Counter(saved) == Counter(in_round)
    allok &= same
    print(f"   A1 round has the same gates as encoder_A1_circuits.txt: {same}")

    print("\n3. composed circuits as run (op by op)")
    SPC.L_GEN = PS.GEN
    rng = random.Random(1116)
    seqs = [[rng.choice(["H", "H_INV", "P", "P_INV"]) for _ in range(6)] for _ in range(20)]
    for k in (None, 1, 2, 3):
        two_all, big_all, nops = Counter(), 0, 0
        for tokens in seqs:
            c = CODE.logical_plus_initial_state()
            seq = tokens + [t[:-4] if t.endswith("_INV") else t + "_INV" for t in reversed(tokens)]
            for i, tok in enumerate(seq, 1):
                c = c + CODE.single_qudit_logical_gate_circuit(tok)
                if k is not None and i % k == 0:
                    c = c + PS.to_circuit([op[:2] if op[0] == "m" else op for op in rnd], 9)
            c = c + CODE.terminal_const0_logical_x_measurement_circuit()
            two, big = pairs_circuit(c)
            two_all += two
            big_all += big
            nops += len(c.operations)
        ok, v = verdict(two_all, big_all)
        allok &= ok
        name = "terminal check only" if k is None else f"interval k={k}"
        print(f"   {name:20s} {v}  ({nops} ops over 20 sequences; couplers used: "
              f"{', '.join(sorted(lbl(p) for p in two_all))})")

    print("\n4. ancillas and one stabilizer per readout")
    anc = sorted({op[1] for op in rnd if op[0] == "m"})
    tags = [(NAME[op[1]], op[2]) for op in rnd if op[0] == "m"]
    print(f"   ancillas used: {[NAME[a] for a in anc]}; readouts per round: {tags}")
    for line in H.noiseless_tests("A1 round", rnd):
        print("  " + line)
        allok &= "OK" in line or "A1 round" in line

    print("\n5. frequency groups")
    bip, col = bipartite()
    allok &= bip
    print(f"   bipartite: {bip}; group 0: {[NAME[q] for q in POS if col[q] == 0]}, "
          f"group 1: {[NAME[q] for q in POS if col[q] == 1]}")

    print("\n6. noiseless LRB-D runs with the paper encoder (20 mirror sequences x 8 shots)")
    SPC.SQUARE = rnd
    for k in (None, 1, 2, 3):
        tot_ok = tot = 0
        for tokens in seqs:
            ok_, kept = SPC.build("paper", tokens, k, shots=8)
            tot_ok += ok_
            tot += kept
        allok &= tot_ok == tot
        name = "terminal check only" if k is None else f"interval k={k}"
        print(f"   {name:20s}: {tot_ok}/{tot} shots pass")

    print("\nsingle-fault check of this round: encoder_A1_grouped_check.out "
          "(A1 grouped (0, 0, 0, 1): 0 of 2136 silent, with and without logical H)")
    print(f"\nALL CHECKS PASS: {allok}")
