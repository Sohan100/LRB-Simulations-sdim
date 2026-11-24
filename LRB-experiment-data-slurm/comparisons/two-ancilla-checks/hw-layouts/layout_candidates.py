"""Which chip layout suits the paper's [[5,1,2]]_3 code with nearest-neighbour logical gates?

The paper's logical gates (Sec. "[[5,1,2]]_3 Qutrit Error Detection Code",
src/lrb/code_definitions.py) use one data-data two-qutrit gate pair, q2-q3
(SWAP_{2,3} in H, CZ_{2->3} in S), so q2-q3 is a required coupler. This script
compares check layouts built on top of it.

Usage:
  layout_candidates.py                 graph facts + noiseless tests + virtual-SWAP check
  layout_candidates.py --faults        ... + exhaustive single-fault check
  layout_candidates.py --bench MODEL P SHOTS   noisy 3-round memory benchmark
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hw_layouts_check as H  # noqa: E402
from hw_layouts_check import g, m, r, A, B  # noqa: E402

SUPPORT = {"G1": {0, 1, 3}, "G2": {1, 2, 4}, "G3": {0, 1, 2}, "G4": {1, 3, 4}}
LOGICAL_COUPLER = (2, 3)

# ---- candidate rounds (ops in hw_layouts_check format; ancilla wires 5..8) ----
Z_HALF = [g("CNOT", 0, A), g("CNOT", 1, B),
          g("CNOT_INV", 1, A), g("CNOT", 2, B),
          g("CNOT_INV", 3, A), g("CNOT_INV", 4, B),
          m(A, "G1"), m(B, "G2"), *r(A, B)]
# 2a: a = {q0,q1,q2,q3} reads G1, G3; b = {q1,q2,q3,q4} reads G2, G4
TWO_A = Z_HALF + [g("H", A), g("H", B),
                  g("CNOT", A, 0), g("CNOT_INV", B, 1),
                  g("CNOT", A, 1), g("CNOT", B, 3),
                  g("CNOT_INV", A, 2), g("CNOT_INV", B, 4),
                  g("H_INV", A), g("H_INV", B), m(A, "G3"), m(B, "G4"), *r(A, B)]
# 2b: a = {q0,q1,q3,q4} reads G1, G4; b = {q0,q1,q2,q4} reads G2, G3
TWO_B = Z_HALF + [g("H", A), g("H", B),
                  g("CNOT", A, 3), g("CNOT", B, 1),
                  g("CNOT_INV", A, 1), g("CNOT", B, 0),
                  g("CNOT_INV", A, 4), g("CNOT_INV", B, 2),
                  g("H_INV", A), g("H_INV", B), m(A, "G4"), m(B, "G3"), *r(A, B)]
# 1: one ancilla touching all five data reads the four checks in turn
ONE_STAR = (H.z_direct(A, [(0, 1), (1, -1), (3, -1)], "G1") + H.z_direct(A, [(1, 1), (2, 1), (4, -1)], "G2")
            + H.x_direct(A, [(0, 1), (1, 1), (2, -1)], "G3") + H.x_direct(A, [(3, 1), (1, -1), (4, -1)], "G4"))

# square lattice only (no diagonals): a = {q0,q1,q2}, c = {q1,q2,q4}, plus q2-q3.
# a reads G3 directly and G1 with the q2-q3 difference trick; c reads G2 directly
# and G4 with the trick (as in the user's right layout).
#   G1: a reads Z0 Z1^-1 Z2, CNOT 3->2, a reads Z2^-1, CNOT_INV 3->2
#       (CNOT_INV(3->2) Z2^-1 CNOT(3->2) = Z2^-1 Z3^-1, so the total is G1)
#   G4: c reads X1^-1 X4^-1 X2, CNOT 2->3, c reads X2^-1, CNOT_INV 2->3
SQUARE = [g("CNOT", 0, A), g("CNOT", 2, B),
          g("CNOT_INV", 1, A), g("CNOT_INV", 4, B),
          g("CNOT", 2, A), g("CNOT", 1, B),
          g("CNOT", 3, 2),
          g("CNOT_INV", 2, A),
          g("CNOT_INV", 3, 2),
          m(A, "G1"), m(B, "G2"), *r(A, B),
          g("H", A), g("H", B),
          g("CNOT_INV", A, 2), g("CNOT_INV", B, 1),
          g("CNOT", A, 0), g("CNOT_INV", B, 4),
          g("CNOT", A, 1), g("CNOT", B, 2),
          g("CNOT", 2, 3),
          g("CNOT_INV", B, 2),
          g("CNOT_INV", 2, 3),
          g("H_INV", A), g("H_INV", B), m(A, "G3"), m(B, "G4"), *r(A, B)]

CANDIDATES = {
    "square lattice: a{012} G3,G1* | c{124} G2,G4* (*via q2-q3)": SQUARE,
    "4 ancillas, one per check (production)": H.ONE_ANCILLA,
    "2 ancillas: a{0123} G1,G3 | b{1234} G2,G4": TWO_A,
    "2 ancillas: a{0134} G1,G4 | b{0124} G2,G3": TWO_B,
    "1 ancilla on all five data": ONE_STAR,
    "user LEFT (hybrid relay, q2-q3)": H.SCHEMES["LEFT  G1,G2(a reads),G3,G3G4^-1 hybrid q2-q3"],
    "user RIGHT (relay + q2-q3 difference)": H.SCHEMES["RIGHT G1(a reads),G2,G3,G4 via q2-q3"],
}

# A short, crossing-free drawing of 2a (unit grid with diagonals): q1 q2 q3 in a row.
POS_2A = {0: (1, 2), 5: (1, 1), 1: (0, 0), 2: (1, 0), 3: (2, 0), 6: (1, -1), 4: (1, -2)}


def coupling_graph(ops):
    G = nx.Graph()
    G.add_edge(*LOGICAL_COUPLER)
    for op in ops:
        if op[0] == "g" and len(op) == 4:
            G.add_edge(op[2], op[3])
    return G


def readers(ops):
    """check tag -> ancilla wire that reads it."""
    return {op[2]: op[1] for op in ops if op[0] == "m" and op[2] != "relay"}


def virtual_swap_ok(G):
    """After H's single-qutrit layer without the SWAP, the stabilizers are the
    paper's with q2 <-> q3 relabelled. Can every relabelled check still be read
    directly by one ancilla?"""
    swap = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
    anc = [v for v in G if v >= 5]
    return all(any({swap[q] for q in s} <= set(G[a]) for a in anc) for s in SUPPORT.values())


def segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    if len({p1, p2, p3, p4}) < 4:
        return False
    return (orient(p1, p2, p3) != orient(p1, p2, p4)) and (orient(p3, p4, p1) != orient(p3, p4, p2))


def drawing_report(G, pos):
    edges = list(G.edges)
    lengths = [np.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]) for u, v in edges]
    cross = [(e, f) for e, f in itertools.combinations(edges, 2)
             if segments_cross(pos[e[0]], pos[e[1]], pos[f[0]], pos[f[1]])]
    return max(lengths), min(lengths), cross


def graph_facts():
    print("graph facts (coupler q2-q3 for the logical gates always included):")
    for name, ops in CANDIDATES.items():
        G = coupling_graph(ops)
        anc = sorted(v for v in G if v >= 5)
        dd = max(G.degree(q) for q in range(5))
        da = max(G.degree(a) for a in anc)
        n2, nm = H.stats(ops)
        gm, cm, _ = H_depth(ops)
        print(f"  {name}")
        print(f"    qutrits {5 + len(anc)}, couplers {G.number_of_edges()}, max degree data {dd} / ancilla {da}, "
              f"planar {nx.check_planarity(G)[0]}, square-lattice (bipartite) {nx.is_bipartite(G)}")
        print(f"    per round: {n2} two-qutrit gates, {nm} readouts, {gm} gate + {cm} collapse moments; "
              f"SWAP_23 can be virtual: {virtual_swap_ok(G)}")
    if "2 ancillas: a{0123} G1,G3 | b{1234} G2,G4" in CANDIDATES:
        G = coupling_graph(TWO_A)
        mx, mn, cross = drawing_report(G, POS_2A)
        print(f"  drawing of 2a on a grid with diagonals: coupler lengths {mn:.2f}..{mx:.2f}, crossings {cross}")


def H_depth(ops):
    import protocol_search as PS
    return PS.depth(H.strip(ops) if hasattr(H, "strip") else [op[:2] if op[0] == "m" else op for op in ops])


# ---- statevector checks: virtual SWAP in H, and preparation by measurement ----
def statevector_checks():
    import hw_layouts_statevector as SV
    rng = np.random.default_rng(7)
    evals, evecs = np.linalg.eigh(SV.P)
    code = evecs[:, evals > 0.5]
    swap = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}

    def relabel(ops):
        out = []
        for op in ops:
            if op[0] == "g":
                out.append(("g", op[1], *[swap[w] for w in op[2:]]))
            else:
                out.append(op)
        return out

    def embed(psi5):
        psi = np.zeros((3,) * SV.NW, dtype=complex)
        psi[(slice(None),) * SV.ND + (0,) * (SV.NW - SV.ND)] = psi5.reshape((3,) * SV.ND)
        return psi

    coeff = rng.normal(size=3) + 1j * rng.normal(size=3)
    psi0 = code @ (coeff / np.linalg.norm(coeff))
    for name in ("2 ancillas: a{0123} G1,G3 | b{1234} G2,G4", "4 ancillas, one per check (production)"):
        # H-bar without its SWAP: single-qutrit layer only, H on 0,2,3,4 and H_INV on 1
        psi = embed(psi0)
        for q in (0, 2, 3, 4):
            psi = SV.gate(psi, "H", q)
        psi = SV.gate(psi, "H_INV", 1)
        worst = 0.0
        for _ in range(2):
            psi, log = SV.run(psi, relabel(CANDIDATES[name]), rng)
            worst = max([worst] + [1 - p[0] for _, _, p in log])
        # apply the skipped SWAP at the end and compare with H-bar |psi0> (SWAP first, as in the code)
        psi = np.swapaxes(psi, 2, 3)
        ref = np.swapaxes(embed(psi0), 2, 3)
        for q in (0, 2, 3, 4):
            ref = SV.gate(ref, "H", q)
        ref = SV.gate(ref, "H_INV", 1)
        fid = abs(np.vdot(SV.data_state(ref), SV.data_state(psi))) ** 2
        print(f"  {name}: H-bar with virtual SWAP, relabelled checks for 2 rounds: "
              f"max P(readout != 0) {worst:.1e}, fidelity with H-bar|psi> {fid:.12f}")
    # preparation of |+_L> by measurement: |+>^5, read G1, G2 once, fix with X3, X4 powers
    plus = np.ones(3) / np.sqrt(3)
    psi5 = plus
    for _ in range(4):
        psi5 = np.kron(psi5, plus)
    psi = embed(psi5)
    psi, log = SV.run(psi, TWO_A[:10], rng)
    s1, s2 = log[0][1], log[1][1]
    d = SV.data_state(psi)
    # X_3^k shifts G1 = Z0 Z1^-1 Z3^-1 by -k; X_4^k shifts G2 = Z1 Z2 Z4^-1 by -k
    fix = np.kron(np.kron(np.eye(27), np.linalg.matrix_power(SV.X1, s1 % 3)),
                  np.linalg.matrix_power(SV.X1, s2 % 3))
    d = fix @ d
    target = SV.P @ psi5
    target /= np.linalg.norm(target)
    print(f"  |+_L> by measurement (|+>^5, read G1={s1}, G2={s2}, apply X3^{s1 % 3} X4^{s2 % 3}): "
          f"fidelity with the code's |+_L> {abs(np.vdot(target, d)) ** 2:.12f}, "
          f"<X_L> = {np.vdot(d, SV.XL @ d).real:+.3f}")


def bench(model, p, shots, rounds=3, noreset=False):
    """noreset: drop every RESET (ancillas start each check in the state they were
    read in). With all syndromes 0 every readout stays 0, so 'accept if all
    readouts are 0' is still the right postselection."""
    import noise_bench as NB
    import protocol_search as PS
    from sdim.circuit import Circuit
    from lrb.code_definitions import FoldedQutritDetectionCode as C9
    prod, _ = NB.production_round("folded_qutrit", model)
    if noreset:
        bare = Circuit(dimension=3, num_qudits=prod.num_qudits)
        for o in prod.operations:
            if o.gate_name != "RESET":
                bare.add_gate(o.gate_name, *PS.GEN._operation_qudits(o))
        prod = bare
    rows = {"production round (LRB runs, 4 ancillas)" + (" no reset" if noreset else ""): prod}
    for name, ops in CANDIDATES.items():
        if noreset and not name.startswith(("2 ancillas", "square")):
            continue
        ops = [op[:2] if op[0] == "m" else op for op in ops if not (noreset and op[0] == "r")]
        rows[name + (" no reset" if noreset else "")] = PS.to_circuit(ops, 9)
    for name, ideal in rows.items():
        if only and not name.startswith("production") and not any(o in name for o in only):
            continue
        moments = PS.GEN.schedule_moments(ideal)
        ng = sum(1 for collapse, _ in moments if not collapse)
        cells = []
        for sandwich in (False, True):
            acc, err = NB.memory(C9, ideal, model, p, rounds, sandwich, shots)
            cells.append(f"acc {acc / shots:.4f} err {err / max(acc, 1):.2e} ({err})")
        print(f"{model} p={p} R={rounds} {name:44s} {ng}g+{len(moments) - ng}c  "
              f"X: {cells[0]}   Z: {cells[1]}", flush=True)


if __name__ == "__main__":
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    if only:
        CANDIDATES = {k: v for k, v in CANDIDATES.items() if any(o in k for o in only)}
    if "--bench" in sys.argv:
        i = sys.argv.index("--bench")
        bench(sys.argv[i + 1], float(sys.argv[i + 2]), int(sys.argv[i + 3]), noreset="--noreset" in sys.argv)
        sys.exit()
    graph_facts()
    if not only:
        print("\nstatevector checks:")
        statevector_checks()
    print()
    for name, ops in CANDIDATES.items():
        print(name, flush=True)
        for line in H.noiseless_tests(name, ops):
            print(line, flush=True)
    if "--faults" in sys.argv:
        with Pool(len(CANDIDATES)) as pool:
            for name, lines in pool.imap(H.fault_test, CANDIDATES.items()):
                print(name, flush=True)
                for line in lines:
                    print(line, flush=True)
