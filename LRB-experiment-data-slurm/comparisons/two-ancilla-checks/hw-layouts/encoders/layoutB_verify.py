"""Full verification of layout B (co-design winner) for the qutrit and qubit codes.

Layout B (square lattice, x right, y up):
    q4(0,2)  A(1,2)
    q1(0,1)  q2(1,1)  B(2,1)
    q0(0,0)  q3(1,0)
Couplers = lattice-adjacent pairs: q0-q1, q0-q3, q1-q2, q2-q3, q1-q4, A-q4, A-q2, B-q2.
Ancilla A = wire 5 reads G2, G4; ancilla B = wire 6 reads G1, G3. Encoder tree
q0-q1, q0-q3, q1-q2, q1-q4: H on q2, q3, q4; CNOT 3->0, CNOT 4->1, CNOT_INV 2->1,
CNOT 1->0. q0-q3 is encoder-only (X0 X3 is logical); checks use q0-q1, q1-q2,
q1-q4, q2-q3. Round: reset A, B; G2 (A), G1 (B); read; G4 (A), G3 (B); read
(no mid-round reset).
Checks: 1 couplers; 2 encoder state = paper encoder state (d = 3, 2); 3 every
two-qudit gate on a coupler; 4 noiseless LRB-D (terminal, k = 1, 2, 3) with the
new encoder, d = 3 and 2; 5 one stabilizer per readout (d = 3); 6 exact
statevector non-destructiveness (d = 3); 7 single faults d = 3 and d = 2;
8 Garnet placements.
"""
import ast
import itertools
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from sdim.circuit import Circuit  # noqa: E402
from sdim.program import Program  # noqa: E402
from lrb.code_definitions import FoldedQutritDetectionCode as Q3, FoldedQubitDetectionCode as Q2  # noqa: E402
import protocol_search as PS  # noqa: E402
import garnet_place as GP  # noqa: E402

A, B = 5, 6
NAME = {0: "q0", 1: "q1", 2: "q2", 3: "q3", 4: "q4", A: "A", B: "B"}
POS = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (0, 2), A: (1, 2), B: (2, 1)}
COUPLERS = {frozenset((a, b)) for a in POS for b in POS
            if a < b and abs(POS[a][0] - POS[b][0]) + abs(POS[a][1] - POS[b][1]) == 1}
ENCODER = [("H", 2), ("H", 3), ("H", 4), ("CNOT", 3, 0), ("CNOT", 4, 1), ("CNOT_INV", 2, 1), ("CNOT", 1, 0)]
ROUND = ast.literal_eval((HERE / "layoutB_round_ops.txt").read_text())
QUD = PS.GEN._operation_qudits


def enc_circuit(d):
    c = Circuit(dimension=d, num_qudits=9)
    for g in ENCODER:
        c.add_gate(g[0], *g[1:])
    return c


def lbl(p):
    a, b = sorted(p)
    return f"{NAME.get(a, a)}-{NAME.get(b, b)}"


def statevector(c, d, nq=7):
    """Exact statevector of a unitary circuit on the first nq wires (gates H, H_INV, CNOT, CNOT_INV, X, Z, P...)."""
    psi = np.zeros((d,) * nq, dtype=complex)
    psi[(0,) * nq] = 1
    w = np.exp(2j * np.pi / d)
    F = np.array([[w ** (j * k) for j in range(d)] for k in range(d)]) / np.sqrt(d)
    for o in c.operations:
        q = QUD(o)
        name = o.gate_name
        if name in ("CNOT", "CNOT_INV"):
            s = 1 if name == "CNOT" else -1
            cq, tq = q
            out = np.empty_like(psi)
            for v in range(d):
                idx = [slice(None)] * nq
                idx[cq] = v
                out[tuple(idx)] = np.roll(psi[tuple(idx)], s * v, axis=tq if tq < cq else tq - 1)
            psi = out
        elif name in ("H", "H_INV"):
            M = F if name == "H" else F.conj().T
            psi = np.moveaxis(np.tensordot(M, psi, axes=([1], [q[0]])), 0, q[0])
        else:
            raise ValueError(name)
    return psi


def lrb_build(code, d, tokens, k, shots):
    c = enc_circuit(d)
    labels, count = [], Counter()
    seq = tokens + [t[:-4] if t.endswith("_INV") else t + "_INV" for t in reversed(tokens)]
    for i, tok in enumerate(seq, 1):
        c = c + code.single_qudit_logical_gate_circuit(tok)
        if k is not None and i % k == 0:
            for op in ROUND:
                if op[0] == "g":
                    c.add_gate(op[1], *op[2:])
                elif op[0] == "m":
                    c.add_gate("M", op[1])
                    labels.append((op[1], count[op[1]]))
                    count[op[1]] += 1
                else:
                    c.add_gate("RESET", op[1])
                    count[op[1]] += 1
    c = c + code.terminal_const0_logical_x_measurement_circuit()
    res = Program(c).simulate(shots=shots)[0]
    ok = 0
    for s in range(shots):
        x = [int(res[q][count[q]][s].measurement_value) for q in range(5)]
        anc = [int(res[w][n][s].measurement_value) for w, n in labels]
        if d == 3:
            good = (x[0] + x[1] - x[2]) % 3 == 0 and (-x[1] + x[3] - x[4]) % 3 == 0 and (x[0] + x[3]) % 3 == 0
        else:
            good = (x[0] + x[1] + x[2]) % 2 == 0 and (x[1] + x[3] + x[4]) % 2 == 0 and (x[0] + x[3]) % 2 == 0
        ok += good and not any(anc)
    return ok


# ---- d-generic single-fault test (mirrors ../hw_layouts_check.fault_test) ----
def fbuild(code, d, ops, fault, sandwich):
    c = code.logical_plus_initial_state()
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H")
    labels, count = [], Counter()
    def inject(paulis):
        for w, (x, z) in paulis.items():
            for _ in range(x):
                c.add_gate("X", w)
            for _ in range(z):
                c.add_gate("Z", w)
    for rnd in range(2):
        if rnd == 0 and fault[0] == -1:
            inject(fault[1])
        for i, op in enumerate(ops):
            if op[0] == "g":
                c.add_gate(op[1], *op[2:])
            elif op[0] == "m":
                c.add_gate("M", op[1])
                labels.append((op[1], count[op[1]]))
                count[op[1]] += 1
            else:
                c.add_gate("RESET", op[1])
                count[op[1]] += 1
            if rnd == 0 and fault[0] == i:
                inject(fault[1])
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H_INV", q)
        c.add_gate("M", q)
    return c, labels, count


def fone(args):
    d, ops, fault, sandwich = args
    code = Q3 if d == 3 else Q2
    c, labels, count = fbuild(code, d, ops, fault, sandwich)
    res = Program(c).simulate(shots=2)[0]
    x = [int(res[q][count[q]][0].measurement_value) for q in range(5)]
    anc = any(int(res[w][n][0].measurement_value) for w, n in labels)
    if d == 3:
        dat = (x[0] + x[1] - x[2]) % 3 or (-x[1] + x[3] - x[4]) % 3
        lg = (x[0] + x[3]) % 3
    else:
        dat = (x[0] + x[1] + x[2]) % 2 or (x[1] + x[3] + x[4]) % 2
        lg = (x[0] + x[3]) % 2
    return bool(lg) and not anc and not dat


def faults(d, ops):
    paulis = [(x, z) for x in range(d) for z in range(d) if (x, z) != (0, 0)]
    sites = [(-1, (q,)) for q in range(7)]
    for i, op in enumerate(ops):
        if op[0] == "g":
            sites.append((i, tuple(op[2:])))
        elif op[0] in ("r", "m"):
            sites.append((i, (op[1],)))
    for i, wires in sites:
        for combo in itertools.product([(0, 0)] + paulis, repeat=len(wires)):
            if any(p != (0, 0) for p in combo):
                yield i, dict(zip(wires, combo))


if __name__ == "__main__":
    print("1. couplers (lattice-adjacent pairs):", ", ".join(sorted(lbl(p) for p in COUPLERS)),
          f"({len(COUPLERS)}; all grid distance 1)")
    print("\n2. encoder state vs the paper encoder (exact statevector, ancillas idle):")
    for d, code in ((3, Q3), (2, Q2)):
        ref = statevector(code.logical_plus_initial_state(), d)
        new = statevector(enc_circuit(d), d)
        print(f"   d={d}: |<paper|new>|^2 = {abs(np.vdot(ref, new)) ** 2:.12f}")
    print("\n3. two-qudit gates vs couplers:")
    pieces = {"new encoder": [frozenset(g[1:]) for g in ENCODER if len(g) == 3],
              "check round": [frozenset(op[2:]) for op in ROUND if op[0] == "g" and len(op) == 4]}
    for tok in ("H", "H_INV", "P", "P_INV", "X", "Z"):
        pieces[f"logical {tok}"] = [frozenset(QUD(o)) for o in Q3.single_qudit_logical_gate_circuit(tok).operations
                                    if len(QUD(o)) == 2]
    pieces["terminal readout"] = [frozenset(QUD(o)) for o in Q3.terminal_const0_logical_x_measurement_circuit().operations
                                  if len(QUD(o)) == 2]
    for name, prs in pieces.items():
        cnt = Counter(prs)
        bad = [p for p in cnt if p not in COUPLERS]
        print(f"   {name:18s} {'OK' if not bad else 'FAIL ' + str([lbl(p) for p in bad])}  "
              + ", ".join(f"{lbl(p)} x{n}" for p, n in sorted(cnt.items(), key=lambda t: sorted(t[0]))))
    print("\n4. noiseless LRB-D with the new encoder (20 mirror sequences of 6 Cliffords x 8 shots):")
    rng = random.Random(1117)
    seqs = [[rng.choice(["H", "H_INV", "P", "P_INV"]) for _ in range(6)] for _ in range(20)]
    for d, code in ((3, Q3), (2, Q2)):
        for k in (None, 1, 2, 3):
            ok = sum(lrb_build(code, d, t, k, 8) for t in seqs)
            print(f"   d={d} {'terminal check only' if k is None else f'interval k={k}':20s}: {ok}/160 pass", flush=True)
    print("\n5. one stabilizer per readout: see layoutB_readout_check.py / .out (mid-round reset variant)")
    print("\n6. exact statevector, non-destructive (d = 3): see layoutB_statevector below")
    import hw_layouts_statevector as SV
    rngn = np.random.default_rng(1117)
    ev, evec = np.linalg.eigh(SV.P)
    codesp = evec[:, ev > 0.5]
    worst_f, worst_p = 1.0, 0.0
    for trial in range(10):
        coeff = rngn.normal(size=3) + 1j * rngn.normal(size=3)
        psi0 = codesp @ (coeff / np.linalg.norm(coeff))
        psi = np.zeros((3,) * SV.NW, dtype=complex)
        psi[(slice(None),) * 5 + (0,) * 4] = psi0.reshape((3,) * 5)
        for rnd in range(3):
            psi, log = SV.run(psi, ROUND, rngn)
            worst_p = max([worst_p] + [1 - p[0] for _, _, p in log])
            worst_f = min(worst_f, abs(np.vdot(psi0, SV.data_state(psi))) ** 2)
    print(f"   10 random logical states x 3 rounds: min fidelity {worst_f:.12f}, max P(readout != 0) {worst_p:.1e}")
    print("\n7. single faults (2 rounds, with and without logical-H sandwich):", flush=True)
    with Pool(24) as pool:
        for d in (3, 2):
            fl = list(faults(d, ROUND))
            res = pool.map(fone, [(d, ROUND, f, s) for s in (False, True) for f in fl], chunksize=16)
            print(f"   d={d}: {len(fl)} faults per run, silent {sum(res[:len(fl)])} (no H) and "
                  f"{sum(res[len(fl):])} (H sandwich)", flush=True)
    print("\n8. IQM Garnet placements:")
    p = GP.placements([(NAME[a], NAME[b]) for a, b in map(sorted, COUPLERS)])
    print(f"   {len(p)} placements; e.g. {p[0] if p else None}")
