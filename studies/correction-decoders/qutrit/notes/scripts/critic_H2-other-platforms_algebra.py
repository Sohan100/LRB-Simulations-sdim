"""Critic of H2-other-platforms.md: independent algebra, counting and budget checks.

Run:  module load python/3.11 && python3 critic_H2-other-platforms_algebra.py
Runtime: about 30-60 s on one core (reads 30 stored depth-22 circuits).
Writes nothing. Read-only use of the baseline run folder.

Independent of H2's scripts: gate matrices come from sdim.unitary (the
package's own generators), gate counts come from the repository's own
FoldedQutritDetectionCode methods and from the stored baseline circuits,
and Weyl propagation uses a symplectic propagator that is itself checked
against the sdim matrices.

Blocks
  K1  catalog: per-token gate content, catalog means, duplicates, group
      coverage, inverse-map correctness
  K2  actual gate/measurement/reset counts in the 30 stored m = 22 circuits
  K3  SI1000 nominal fault weight per round, read from stored circuits
  K4  light-shift gate identity, D = Z P^2, LS-based CNOT with explicit
      local corrections
  K5  interaction graph from the repository's check circuits; routing of
      CZ(2,3) on the 3x3 grid (explicit nearest-neighbour construction,
      verified by statevector) and single-fault analysis of that routing
  K6  no-reset check on the full 9-qutrit encoded block (statevector)
  K7  symplectic propagation: code sanity, Clifford dependence of
      syndromes (relevant to the per-round lookup-table claim)
  K8  T3 conjugation in closed form
  K9  shot-noise budget: sequence averaging (factor 30) and Rayleigh floor
  K10 time-budget arithmetic incl. coherent part of a shot
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import math
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

import sdim.unitary as su
from sdim.circuit_io import read_circuit

WT = "/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/correction-decoders-qutrit-2c455b"
BASE = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
        "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
_spec = importlib.util.spec_from_file_location("crit_cd", os.path.join(WT, "src", "lrb", "code_definitions.py"))
cd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cd
_spec.loader.exec_module(cd)
Code = cd.FoldedQutritDetectionCode
CAT = cd.single_qutrit_cliffords_exhaustive_strings
INV = cd.QutritCliffordLibrary.build_inverse_lookup()

d = 3
w = np.exp(2j * np.pi / 3)
X = su.generate_x_matrix(3)
Z = su.generate_z_matrix(3)
H = su.generate_h_matrix(3)
P = su.generate_p_matrix(3)
I3 = np.eye(3, dtype=complex)
CNOT = su.generate_cnot_matrix(3)            # |c,t> -> |c, t+c>, control = first factor
CZ = np.kron(I3, H) @ CNOT @ np.kron(I3, H.conj().T)   # sdim's own definition (unitary.py)
SWAP = np.zeros((9, 9), dtype=complex)
for a, b in itertools.product(range(3), repeat=2):
    SWAP[b * 3 + a, a * 3 + b] = 1
ONE = {"H": H, "H_INV": H.conj().T, "P": P, "P_INV": P.conj().T, "X": X, "X_INV": X.conj().T,
       "Z": Z, "Z_INV": Z.conj().T, "I": I3}
TWO = {"CNOT": CNOT, "CNOT_INV": CNOT.conj().T, "CZ": CZ, "CZ_INV": CZ.conj().T, "SWAP": SWAP}


def eq_phase(a, b, tol=1e-9):
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(b[k]) < tol:
        return False
    ph = a[k] / b[k]
    return abs(abs(ph) - 1) < 1e-7 and np.allclose(a, ph * b, atol=1e-8)


def hr(title):
    print("=" * 78)
    print(title)


# ---------------------------------------------------------------- K1 catalog
hr("K1  Clifford catalog: token content, duplicates, group coverage, inverses")


def token_content(tok):
    gates, controls, targets = Code.elementary_single_qudit_logical_operator_gate_sequence(tok)
    n2 = n1 = 0
    for g, c, t in zip(gates, controls, targets):
        if t is not None:
            n2 += 1
        else:
            n1 += len(c) if isinstance(c, list) else 1
    return n2, n1


content = {t: token_content(t) for t in ("I", "H", "P", "X", "Z", "H_INV", "P_INV", "X_INV", "Z_INV")}
print("  per-token (two-qutrit, one-qutrit) gates from the repository:", content)
tok = Counter("".join(CAT))
n = len(CAT)
mean_tok = {g: tok[g] / n for g in "IHPXZ"}
two = sum(mean_tok[g] * content[g][0] for g in "IHPXZ")
one = sum(mean_tok[g] * content[g][1] for g in "IHPXZ")
print(f"  catalog entries {n}; mean tokens/Clifford: " + ", ".join(f"{g} {mean_tok[g]:.4f}" for g in "IHPXZ"))
print(f"  mean per logical Clifford: two-qutrit {two:.4f}, one-qutrit {one:.4f} (incl. the 'I' token)")
print(f"  virtual SWAP: two-qutrit {two - mean_tok['H']:.4f}; X,Z in software: one-qutrit "
      f"{one - 2 * mean_tok['X'] - 2 * mean_tok['Z']:.4f}")
dups = [s for s, c in Counter(CAT).items() if c > 1]
print(f"  distinct strings {len(set(CAT))}; duplicated strings: {dups}")


def string_unitary(s):
    U = I3.copy()
    for ch in s:                      # descriptor read as an operator product (rightmost acts first)
        U = U @ ONE[ch]
    return U


def canon(U):
    k = np.argmax(np.abs(U.flatten()) > 1e-9)
    ph = U.flatten()[k] / abs(U.flatten()[k])
    return tuple(np.round((U / ph).flatten(), 6))


# full single-qutrit Clifford group mod phase from H and P
group = {canon(I3): I3}
frontier = [I3]
while frontier:
    nxt = []
    for U in frontier:
        for G in (H, P):
            V = G @ U
            k = canon(V)
            if k not in group:
                group[k] = V
                nxt.append(V)
    frontier = nxt
cat_keys = [canon(string_unitary(s)) for s in CAT]
mult = Counter(cat_keys)
print(f"  Clifford group (mod phase) generated by H, P: {len(group)} elements")
print(f"  distinct Cliffords represented by the catalog: {len(mult)}; "
      f"missing: {len(group) - len(mult)}; multiplicity histogram: {dict(Counter(mult.values()))}")
inv_ok = 0
for s in CAT:
    fwd = list(reversed(s))           # QutritCliffordLibrary.qutrit_gate_seq_list
    U = I3.copy()
    for g in fwd + INV[s]:            # execution order: forward then inverse
        U = ONE[g] @ U
    inv_ok += eq_phase(U, I3)
print(f"  inverse map gives the identity for {inv_ok}/{n} catalog entries")

# ---------------------------------------------------- K2 counts in stored circuits
hr("K2  counts in the 30 stored m = 22 circuits (p index 10)")


def classify_noise(op):
    pr = op.params or {}
    scale = float(pr.get("prob_scale", "nan"))
    ch = pr.get("noise_channel", pr.get("channel", "d"))
    return op.gate_name, ch, scale


rows = []
for cl in range(30):
    c = read_circuit(f"{BASE}/experiments/LRB/{cl}/10/8.chp")
    cnt = Counter()
    for op in c.operations:
        if op.gate_name in ("N1", "N2"):
            cnt[classify_noise(op)] += 1
        elif op.gate_name in ("M", "RESET"):
            cnt[op.gate_name] += 1
    n2 = cnt[("N2", "d", 1.0)]
    n1 = cnt[("N1", "d", 0.1)]
    mcirc = cnt[("N1", "f", 5.0)]
    rows.append((n2, n1, mcirc, cnt["RESET"], cnt["M"]))
arr = np.array(rows)
for j, name in enumerate(("noisy two-qutrit gates", "noisy one-qutrit gates", "noisy (mid-circuit) measurements",
                          "resets", "all M ops incl. terminal")):
    print(f"  {name:34s}: mean {arr[:, j].mean():7.2f}, min {arr[:, j].min()}, max {arr[:, j].max()}")
print(f"  model prediction (catalog mean, 44 Cliffords + 23 rounds): two-qutrit {44 * two + 23 * 12:.1f}, "
      f"one-qutrit {44 * one + 23 * 4:.1f}")

# ------------------------------------------------ K3 SI1000 weights from circuits
hr("K3  SI1000 nominal fault weight (sum of location probabilities / p) from stored circuits")
c0 = read_circuit(f"{BASE}/experiments/LRB/0/10/0.chp")      # m = 0: one round, no reset
w0 = sum(float((op.params or {}).get("prob_scale", 0)) for op in c0.operations if op.gate_name in ("N1", "N2"))
print(f"  depth-0 circuit (final round only, no reset layer): {w0:.2f} p")
c22 = read_circuit(f"{BASE}/experiments/LRB/0/10/8.chp")
cat_w = Counter()
ops = c22.operations
for i, op in enumerate(ops):
    if op.gate_name not in ("N1", "N2"):
        continue
    g, ch, sc = classify_noise(op)
    if g == "N2":
        pair = {op.qudit_index, op.target_index}
        cat_w["2q check gate" if pair & {5, 6, 7, 8} else "2q logical gate"] += sc
    elif sc == 0.1:
        cat_w["1q check gate" if op.qudit_index in (5, 6, 7, 8) else "1q logical gate"] += sc
    elif sc == 2.0 and ch == "d":
        # find what follows: measurement layer or reset layer
        j = i
        while ops[j].gate_name not in ("M", "RESET"):
            j += 1
        cat_w["idle in measurement layer" if ops[j].gate_name == "M" else "idle in reset layer"] += sc
    elif sc == 1.0:
        cat_w["measured-ancilla depol"] += sc
    elif sc == 5.0:
        cat_w["measurement flip"] += sc
    elif sc == 2.0 and ch == "f":
        cat_w["reset flip"] += sc
rounds, resets = 23, 22
per_round = (cat_w["2q check gate"] + cat_w["1q check gate"] + cat_w["idle in measurement layer"]
             + cat_w["measured-ancilla depol"] + cat_w["measurement flip"]) / rounds
per_reset = (cat_w["idle in reset layer"] + cat_w["reset flip"]) / resets
print("  m = 22 circuit, totals by category / p:", {k: round(v, 2) for k, v in cat_w.items()})
print(f"  per check round {per_round:.2f} p, per reset layer {per_reset:.2f} p, sum {per_round + per_reset:.2f} p")
print(f"  readout layers alone: {(cat_w['idle in measurement layer'] + cat_w['measured-ancilla depol'] + cat_w['measurement flip']) / rounds:.2f} p per round")

# ------------------------------------------------------- K4 LS gate algebra
hr("K4  light-shift gate G(theta) of Hrmo et al. Eq. (3) vs sdim CZ")
D = np.diag([w ** ((j * j) % 3) for j in range(3)])
for th, sgn in ((2 * np.pi / 3, 1), (-2 * np.pi / 3, -1)):
    G = np.diag([1.0 if j == k else np.exp(1j * th) for j in range(3) for k in range(3)])
    Dp = D if sgn == 1 else D.conj().T
    CZp = CZ if sgn == 1 else CZ.conj().T
    print(f"  theta = {sgn:+d}*2pi/3: G == (D x D) CZ^{sgn:+d} exactly: {np.allclose(G, np.kron(Dp, Dp) @ CZp)}")
print(f"  D == Z P^2 exactly: {np.allclose(D, Z @ P @ P)};  D is diagonal with phases (1, w, w)")
G = np.diag([1.0 if j == k else np.exp(2j * np.pi / 3) for j in range(3) for k in range(3)])
cnot_ls = np.kron(I3, H.conj().T) @ np.kron(D.conj().T, D.conj().T) @ G @ np.kron(I3, H)
print(f"  CNOT == (I x H^dag)(D^dag x D^dag) G (I x H): {eq_phase(CNOT, cnot_ls)}  "
      "(needs D^dag on BOTH qutrits in addition to the two Fourier gates)")
print(f"  without the D corrections, (I x H^dag) G (I x H) == CNOT: "
      f"{eq_phase(CNOT, np.kron(I3, H.conj().T) @ G @ np.kron(I3, H))}")
HD = H.conj().T @ D.conj().T
print(f"  merged target gate H^dag D^dag equals H^dag up to phase: {eq_phase(HD, H.conj().T)} "
      "(so it is a different single-qutrit Clifford, not a plain Fourier gate)")

# -------------------------------------------- K5 interaction graph and routing
hr("K5  interaction graph from the repository's check circuits; CZ(2,3) routing")
edges = set()
for fac in (Code.z_ALT_stabilizer_measurement_ancillae_circuit, Code.x_stabilizer_measurement_ancillae_circuit):
    for op in fac().operations:
        if op.target_index is not None:
            edges.add(tuple(sorted((op.qudit_index, op.target_index))))
logical_edges = set()
for t in ("H", "P", "H_INV", "P_INV", "X", "Z"):
    for op in Code.single_qudit_logical_gate_circuit(t).operations:
        if op.target_index is not None:
            logical_edges.add(tuple(sorted((op.qudit_index, op.target_index))))
print(f"  check edges ({len(edges)}): {sorted(edges)}; logical two-qutrit edges: {sorted(logical_edges)}")
Gc = nx.Graph(sorted(edges))
grid = nx.grid_2d_graph(3, 3)
print(f"  check graph isomorphic to 3x3 grid: {nx.is_isomorphic(Gc, grid)}; "
      f"isomorphisms: {sum(1 for _ in isomorphism.GraphMatcher(grid, Gc).isomorphisms_iter())}")
Gf = nx.Graph(sorted(edges | logical_edges))
print(f"  full graph bipartite: {nx.is_bipartite(Gf)}; check-graph distance d(2,3) = {nx.shortest_path_length(Gc, 2, 3)}")
paths = list(nx.all_shortest_paths(Gc, 2, 3))
print(f"  all shortest 2-3 paths in the check graph: {paths}")
print(f"  every shortest path passes through a third data qutrit: "
      f"{all(any(v in (0, 1, 4) for v in p[1:-1]) for p in paths)}")

# explicit nearest-neighbour CZ(2,3) on wires ordered (2, 7, 1, 8, 3), ancillas 7, 8 in |0>
order = [2, 7, 1, 8, 3]
pos = {q: i for i, q in enumerate(order)}
seq = [("CNOT", 2, 7), ("CNOT", 7, 1), ("CNOT", 1, 8), ("CNOT_INV", 7, 1), ("CNOT_INV", 1, 8),
       ("CZ", 8, 3),
       ("CNOT", 1, 8), ("CNOT", 7, 1), ("CNOT_INV", 1, 8), ("CNOT_INV", 7, 1), ("CNOT_INV", 2, 7)]
assert all(tuple(sorted((a, b))) in edges for _, a, b in seq), "construction must be nearest-neighbour"


def apply2(psi, U, a, b, nq):
    psi = psi.reshape([3] * nq)
    U4 = U.reshape(3, 3, 3, 3)
    psi = np.tensordot(U4, psi, axes=([2, 3], [a, b]))
    psi = np.moveaxis(psi, [0, 1], [a, b])
    return psi.reshape(-1)


def apply1(psi, U, a, nq):
    psi = psi.reshape([3] * nq)
    psi = np.tensordot(U, psi, axes=([1], [a]))
    psi = np.moveaxis(psi, 0, a)
    return psi.reshape(-1)


ok = True
rng = np.random.default_rng(5)
for _ in range(5):
    v = rng.normal(size=27) + 1j * rng.normal(size=27)          # random state of data 2, 1, 3
    v /= np.linalg.norm(v)
    psi = np.zeros(3 ** 5, dtype=complex)
    for x2, x1, x3 in itertools.product(range(3), repeat=3):
        idx = np.ravel_multi_index((x2, 0, x1, 0, x3), [3] * 5)
        psi[idx] = v[np.ravel_multi_index((x2, x1, x3), [3] * 3)]
    target = psi.copy()
    out = psi.copy()
    for g, a, b in seq:
        out = apply2(out, TWO[g], pos[a], pos[b], 5)
    target = apply2(target, CZ, pos[2], pos[3], 5)
    ok &= np.allclose(out, target)
print(f"  11-gate nearest-neighbour construction == CZ(2,3) with ancillas 7, 8 restored to |0>: {ok}")
print("  light-cone lower bound for any construction: >= 4 two-qutrit gates (distance 4)")

# ------------------------------------------- symplectic propagator (used in K5b, K7)
NQ = 9


def wvec():
    return np.zeros(2 * NQ, dtype=int)  # [x_0..x_8, z_0..z_8]


def conj1(v, g, q):
    v = v.copy()
    x, z = v[q], v[NQ + q]
    if g == "H":
        v[q], v[NQ + q] = (-z) % 3, x % 3
    elif g == "H_INV":
        v[q], v[NQ + q] = z % 3, (-x) % 3
    elif g == "P":
        v[NQ + q] = (z + x) % 3
    elif g == "P_INV":
        v[NQ + q] = (z - x) % 3
    return v


def conj2(v, g, c, t):
    v = v.copy()
    xc, xt, zc, zt = v[c], v[t], v[NQ + c], v[NQ + t]
    if g == "CNOT":
        v[t], v[NQ + c] = (xt + xc) % 3, (zc - zt) % 3
    elif g == "CNOT_INV":
        v[t], v[NQ + c] = (xt - xc) % 3, (zc + zt) % 3
    elif g == "CZ":
        v[NQ + t], v[NQ + c] = (zt + xc) % 3, (zc + xt) % 3
    elif g == "CZ_INV":
        v[NQ + t], v[NQ + c] = (zt - xc) % 3, (zc - xt) % 3
    elif g == "SWAP":
        v[c], v[t], v[NQ + c], v[NQ + t] = xt, xc, zt, zc
    return v


def weyl_matrix(xs, zs):
    M = np.array([[1.0 + 0j]])
    for a, b in zip(xs, zs):
        M = np.kron(M, np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b))
    return M


# check propagator against sdim matrices on 1 and 2 qutrits
good = True
for g, U in ONE.items():
    if g in ("X", "X_INV", "Z", "Z_INV", "I"):
        continue
    for a, b in itertools.product(range(3), repeat=2):
        v = np.zeros(2 * NQ, dtype=int); v[0], v[NQ] = a, b
        u = conj1(v, g, 0)
        good &= eq_phase(U @ weyl_matrix([a], [b]) @ U.conj().T, weyl_matrix([u[0]], [u[NQ]]))
for g, U in TWO.items():
    for a0, a1, b0, b1 in itertools.product(range(3), repeat=4):
        v = np.zeros(2 * NQ, dtype=int); v[0], v[1], v[NQ], v[NQ + 1] = a0, a1, b0, b1
        u = conj2(v, g, 0, 1)
        good &= eq_phase(U @ weyl_matrix([a0, a1], [b0, b1]) @ U.conj().T,
                         weyl_matrix([u[0], u[1]], [u[NQ], u[NQ + 1]]))
print(f"  symplectic propagator agrees with sdim matrices for all gates: {good}")

STAB = {"SZ1": ({}, {0: 1, 1: -1, 3: -1}), "SZ2": ({}, {1: 1, 2: 1, 4: -1}),
        "SX1": ({0: 1, 1: 1, 2: -1}, {}), "SX2": ({1: -1, 3: 1, 4: -1}, {})}


def mk(xd, zd):
    v = wvec()
    for q, e in xd.items():
        v[q] = e % 3
    for q, e in zd.items():
        v[NQ + q] = e % 3
    return v


def symp(u, v):
    return int((u[:NQ] @ v[NQ:] - u[NQ:] @ v[:NQ]) % 3)


S = {k: mk(*val) for k, val in STAB.items()}
XL, ZL = mk({0: 1, 3: 1}, {}), mk({}, {0: 1, 2: 1})


def syndrome(v):
    return tuple(symp(v, S[k]) for k in ("SZ1", "SZ2", "SX1", "SX2"))


# K5b: single faults in the 11-gate routing that leave a trivial syndrome but a logical error
dangerous, total = 0, 0
for gi, (g, a, b) in enumerate(seq):
    for e in itertools.product(range(3), repeat=4):
        if not any(e):
            continue
        v = mk({a: e[0], b: e[1]}, {a: e[2], b: e[3]})
        for g2, a2, b2 in seq[gi + 1:]:
            v = conj2(v, g2, a2, b2)
        total += 1
        anc_x = any(v[q] for q in (5, 6, 7, 8))   # X part on ancillas would be caught next round
        data = v.copy()
        for q in (5, 6, 7, 8):
            data[q] = 0; data[NQ + q] = 0
        if not anc_x and syndrome(data) == (0, 0, 0, 0) and (symp(data, XL) or symp(data, ZL)):
            dangerous += 1
print(f"  single two-qutrit faults inside the routing that give an undetectable logical error: "
      f"{dangerous}/{total} (fault-tolerance cost of routing through data qutrit 1)")

# ----------------------------------------------------- K6 no-reset, full block
hr("K6  no-reset check on the full 9-qutrit encoded block (statevector, sdim matrices)")
enc = [("H_INV", 2), ("H_INV", 3), ("H_INV", 4), ("CNOT_INV", 2, 1), ("CNOT", 1, 0), ("CNOT", 3, 0), ("CNOT", 4, 2)]
check_ops = []
for fac in (Code.z_ALT_stabilizer_measurement_ancillae_circuit, Code.x_stabilizer_measurement_ancillae_circuit):
    for op in fac().operations:
        if op.gate_name == "M":
            continue
        check_ops.append((op.gate_name, op.qudit_index, op.target_index))


def run(ops_list, psi):
    for op in ops_list:
        if len(op) == 2 or op[2] is None:
            psi = apply1(psi, ONE[op[0]], op[1], 9)
        else:
            psi = apply2(psi, TWO[op[0]], op[1], op[2], 9)
    return psi


psi0 = np.zeros(3 ** 9, dtype=complex); psi0[0] = 1
plus_L = run(enc, psi0)


def apply_weyl(psi, v):
    """Apply X^x Z^z qutrit by qutrit (no dense 3^9 x 3^9 matrices)."""
    for q in range(NQ):
        Mq = np.linalg.matrix_power(X, int(v[q]) % 3) @ np.linalg.matrix_power(Z, int(v[NQ + q]) % 3)
        if v[q] % 3 or v[NQ + q] % 3:
            psi = apply1(psi, Mq, q, 9)
    return psi


# sanity: stabilizer and X_L eigenvalues of the encoded state
def expval(v, psi):
    return np.vdot(psi, apply_weyl(psi.copy(), v))
print("  <S> for SZ1, SZ2, SX1, SX2 on encoded |+>_L:",
      [np.round(expval(S[k], plus_L), 6) for k in ("SZ1", "SZ2", "SX1", "SX2")],
      " <X_L> =", np.round(expval(XL, plus_L), 6))
rng = np.random.default_rng(11)
allok, cnt_cases = True, 0
outcome_vs_symp = Counter()
for trial in range(40):
    xs = rng.integers(0, 3, 5); zs = rng.integers(0, 3, 5)
    ev = mk({q: int(xs[q]) for q in range(5)}, {q: int(zs[q]) for q in range(5)})
    base_state = apply_weyl(plus_L.copy(), ev)
    ref = None
    for k in [(0, 0, 0, 0)] + [tuple(rng.integers(0, 3, 4)) for _ in range(3)]:
        st = base_state.copy()
        for qi, kk in zip((5, 6, 7, 8), k):
            for _ in range(kk):
                st = apply1(st, X, qi, 9)
        out = run(check_ops, st).reshape([3] * 9)
        probs = (np.abs(out) ** 2).sum(axis=(0, 1, 2, 3, 4)).reshape(-1)
        idx = int(np.argmax(probs))
        det = probs[idx] > 1 - 1e-9
        res = np.unravel_index(idx, [3] * 4)
        if ref is None:
            ref = np.array(res)
            v = mk({q: int(xs[q]) for q in range(5)}, {q: int(zs[q]) for q in range(5)})
            outcome_vs_symp[(tuple(int(r) for r in ref), syndrome(v))] += 1
        allok &= det and all((res[i] - ref[i] - k[i]) % 3 == 0 for i in range(4))
        cnt_cases += 1
print(f"  {cnt_cases} (error, ancilla start) cases: outcome deterministic and equal to k + s(E): {allok}")
lin = all(tuple(o) == tuple(int(s_) for s_ in s) or True for (o, s) in outcome_vs_symp)
pairs = list(outcome_vs_symp)
# fit a sign per detector between measured outcome and symplectic syndrome
signs = []
for i in range(4):
    sg = {(o[i] * pow(s[i], 1, 3)) for (o, s) in pairs if s[i]}
    ok_plus = all(o[i] == s[i] for o, s in pairs)
    ok_minus = all(o[i] == (-s[i]) % 3 for o, s in pairs)
    signs.append("+" if ok_plus else ("-" if ok_minus else "?"))
print(f"  measured outcome vs symplectic syndrome, sign per check (SZ1, SZ2, SX1, SX2): {signs}")

# ------------------------------------------------- K7 Clifford dependence
hr("K7  symplectic sanity of the code and Clifford dependence of syndromes")


def apply_token(v, tok_):
    for op in Code.single_qudit_logical_gate_circuit(tok_).operations:
        if op.target_index is None:
            if op.gate_name in ("H", "H_INV", "P", "P_INV"):
                v = conj1(v, op.gate_name, op.qudit_index)
        else:
            v = conj2(v, op.gate_name, op.qudit_index, op.target_index)
    return v


for t in ("H", "P", "H_INV", "P_INV"):
    imgs = {k: apply_token(S[k], t) for k in S}
    in_group = all(syndrome(u) == (0, 0, 0, 0) and symp(u, XL) == 0 and symp(u, ZL) == 0 for u in imgs.values())
    xl, zl = apply_token(XL, t), apply_token(ZL, t)
    print(f"  token {t:6s}: stabilizers mapped into the stabilizer group: {in_group}; "
          f"X_L -> (<.,X_L>,<.,Z_L>) = ({symp(xl, XL)},{symp(xl, ZL)}), Z_L -> ({symp(zl, XL)},{symp(zl, ZL)})")
for e_name, e in (("X0", mk({0: 1}, {})), ("Z1", mk({}, {1: 1})), ("X3", mk({3: 1}, {}))):
    s0 = syndrome(e)
    sH = syndrome(apply_token(e, "H"))
    sP = syndrome(apply_token(e, "P"))
    print(f"  data error {e_name}: syndrome {s0}; after logical H {sH}; after logical P {sP}")

# ----------------------------------------------------------- K8 T3
hr("K8  T3 = diag(1, z, z^8): closed form of T3 X T3^dag")
zeta = np.exp(2j * np.pi / 9)
T3 = np.diag([1, zeta, zeta ** 8])
lhs = T3 @ X @ T3.conj().T
rhs = zeta * X @ np.diag([w ** ((j * j + j) % 3) for j in range(3)])
print(f"  T3 X T3^dag == z * X * diag(w^(j^2+j)) exactly: {np.allclose(lhs, rhs)} "
      "(quadratic phase => Clifford, not Weyl)")

# ------------------------------------------------------------- K9 statistics
hr("K9  shot-noise budget vs the baseline CSVs")


def read(pidx, pol, k):
    with open(f"{BASE}/results/LRB/{pidx}/{pol}_check_data/{k}.csv") as fh:
        r = list(csv.reader(fh))
    f = np.array([float(x) for x in r[1][1:]]); fs = np.array([float(x) for x in r[2][1:]])
    rej = np.array([float(x) for x in r[3][1:]])
    return f, fs, rej


SH = 1e6
f, fs, rej = read(4, "unif", 1)
nacc = SH * (1 - rej[-1])
pred = math.sqrt((1 - f[-1]) * (1 + 2 * f[-1]) / (2 * nacc))
print(f"  p=1e-3 unif 1, m=22: N_acc/circuit {nacc:.0f}; predicted shot-noise sd {pred:.3e}; "
      f"observed sd over 30 sequences {fs[-1]:.3e}")
excess = math.sqrt(max(fs[-1] ** 2 - pred ** 2, 0))
print(f"    => sequence-to-sequence (non-shot) sd about {excess:.1e}; standard error of the 30-sequence mean "
      f"{fs[-1] / math.sqrt(30):.2e} = {fs[-1] / math.sqrt(30) / (f[0] - f[-1]):.0%} of the decay {f[0] - f[-1]:.2e}")
for pol, k, pidx in (("unif", 1, 4), ("const", 1, 4), ("const", 0, 4)):
    f, fs, rej = read(pidx, pol, k)
    sig = 0.1 * (f[0] - f[-1])
    n_acc_tot = (1 - f[-1]) * (1 + 2 * f[-1]) / (2 * sig ** 2)
    n_raw_tot = n_acc_tot / (1 - rej[-1])
    print(f"  p=1e-3 {pol} {k}: raw shots for 10% of decay: {n_raw_tot:.3g} in TOTAL over the m=22 sequences, "
          f"i.e. {n_raw_tot / 30:.3g} per circuit with 30 sequences (H2 quotes the total as 'per circuit')")
for pidx, p in ((9, 0.0125), (10, 0.015), (12, 0.03)):
    f, fs, rej = read(pidx, "const", 1)
    nacc = SH * (1 - rej[-1])
    floor = math.sqrt(math.pi / (4 * nacc))
    sd = math.sqrt((4 - math.pi) / (4 * nacc))
    print(f"  p={p} const 1 m=22: rejection {rej[-1]:.6f} (1 - 1/81 = {1 - 1 / 81:.6f}); F {f[-1]:.4f} vs Rayleigh "
          f"floor sqrt(pi/4N) {floor:.4f}; sd {fs[-1]:.4f} vs Rayleigh sd {sd:.4f}; "
          f"z = {(f[-1] - floor) / (sd / math.sqrt(30)):+.2f}")
f, fs, rej = read(10, "const", 1)
print(f"  p=0.015 const 1 by depth: F {np.round(f, 4).tolist()}; rejection {np.round(rej, 5).tolist()}")

# --------------------------------------------------------------- K10 time budget
hr("K10 time-budget arithmetic (H2 inputs), incl. coherent part of a shot")
US = 1e-6
plat = {"ion": dict(t1=150 * US, t2=600 * US, tm=3800 * US, tr=300 * US, tshot=10e-3, coh=100e-3, par=1),
        "transmon": dict(t1=128e-9, t2=1 * US, tm=1 * US, tr=1 * US, tshot=100 * US, coh=30 * US, par=4)}
two_h2, one_h2 = 3.116, 16.67
for nm, pp in plat.items():
    rnd_par = 12 * pp["t2"] / pp["par"] + 4 * pp["t1"] + pp["tm"] + pp["tr"]
    rnd_seq = rnd_par + 3 * pp["tm"]
    cl = two_h2 * pp["t2"] + one_h2 * pp["t1"]
    shot = pp["tshot"] + 44 * cl + 23 * rnd_par
    coh_part = shot - pp["tshot"]
    depths = [0, 2, 4, 6, 10, 14, 18, 20, 22]
    tall = 30 * sum(pp["tshot"] + 2 * m * cl + (m + 1) * rnd_par for m in depths)
    print(f"  {nm}: round {rnd_seq * 1e3:.4g} ms seq / {rnd_par * 1e3:.4g} ms par; shot(m=22) {shot:.4g} s; "
          f"shot/coh {shot / pp['coh']:.3g} (coherent part only {coh_part / pp['coh']:.3g}); "
          f"270 circuits x 1e3/1e4/1e6 shots: {tall * 1e3 / 3600:.3g} h / {tall * 1e4 / 86400:.3g} d / "
          f"{tall * 1e6 / 86400 / 365.25:.3g} yr")
print(f"  photonic: (1/3)^12 = {(1 / 3) ** 12:.3e}; (1/3)^413 = 10^{-413 * math.log10(3):.1f}")
print(f"  lookup tables: 3^4 = {3 ** 4}, 3^8 = {3 ** 8} ({3 ** 8 * 3 / 8 / 1024:.2f} KiB at 3 bits), "
      f"3^12 = {3 ** 12} ({3 ** 12 * 3 / 8 / 1024:.1f} KiB)")
