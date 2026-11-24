"""H2 (other platforms): small exact checks used in H2-other-platforms.md.

Run with:  module load python/3.11 && python3 H2_qutrit_algebra_checks.py
Runtime: about 3 s, one core. Writes nothing.

Gate conventions follow sdim/unitary.py (read in this task):
  X|j> = |j+1>, Z|j> = w^j |j>, H[m,n] = w^(mn)/sqrt(3),
  P = diag(w^(j(j-1)/2)) = diag(1, 1, w) for d = 3,
  CNOT|c,t> = |c, t+c>, CZ|c,t> = w^(ct)|c,t>, w = exp(2 pi i / 3).

Checks
  C1  Hrmo et al. light-shift gate G(theta) (|jj> -> |jj>, |jk> -> e^{i theta}|jk>)
      at theta = +-2pi/3 equals (D x D) CZ^(+-1) up to a global phase, with
      D = diag(w^(j^2)) a single-qutrit Clifford; hence one native LS gate plus
      single-qutrit Fourier gates gives each CNOT/CNOT_INV of the check circuit.
  C2  Every gate used by the LRB circuits (H, H_INV, P, P_INV, X, Z, CNOT,
      CNOT_INV, CZ, CZ_INV, SWAP) maps every Weyl operator to a Weyl operator
      up to phase (so a Pauli frame can be pushed to the terminal readout).
  C3  A qutrit diagonal third-level gate T3 = diag(1, z, z^8), z = exp(2 pi i/9)
      (the d = 3 "pi/8" gate of Howard and Vala, arXiv:1206.1598) is not a
      Clifford, and T3 X T3^dag is a Clifford but not a Weyl operator; a frame
      error X before T3 changes the X-basis outcome distribution by more than a
      cyclic shift, so it cannot be fixed by relabelling the final readout.
  C4  No-reset syndrome extraction: with the ancilla starting in |k> instead
      of |0>, the Z-check and X-check circuits of FoldedQutritDetectionCode
      return k + s (mod 3) deterministically, so successive raw outcomes can
      be differenced; reset is optional for the Clifford LRB experiment.
  C5  Interaction graph of the circuit (check CNOTs + logical SWAP/CZ(2,3)):
      degrees, planarity, bipartiteness, smallest square grid containing it.
  C6  Size of a real-time lookup-table decoder for this code.
  C7  Monte Carlo check of the shot-noise formula for |mean(w^k)| used in
      H2_platform_budget.py block B5.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

d = 3
w = np.exp(2j * np.pi / d)
I3 = np.eye(d, dtype=complex)
X = np.zeros((d, d), dtype=complex)
for j in range(d):
    X[(j + 1) % d, j] = 1
Z = np.diag([w ** j for j in range(d)])
H = np.array([[w ** (m * n) for n in range(d)] for m in range(d)]) / np.sqrt(d)
P = np.diag([w ** (j * (j - 1) // 2) for j in range(d)])
CNOT = np.zeros((d * d, d * d), dtype=complex)
CZ = np.zeros((d * d, d * d), dtype=complex)
SWAP = np.zeros((d * d, d * d), dtype=complex)
for c, t in itertools.product(range(d), repeat=2):
    CNOT[c * d + (t + c) % d, c * d + t] = 1
    CZ[c * d + t, c * d + t] = w ** (c * t)
    SWAP[t * d + c, c * d + t] = 1


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def equal_up_to_phase(a, b, tol=1e-9):
    idx = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(b[idx]) < tol:
        return False
    ph = a[idx] / b[idx]
    return abs(abs(ph) - 1) < 1e-7 and np.allclose(a, ph * b, atol=1e-7)


def weyl(n):
    """All n-qutrit Weyl operators X^a Z^b (tensor products), with labels."""
    ops = []
    for exps in itertools.product(range(d), repeat=2 * n):
        mats = [np.linalg.matrix_power(X, exps[2 * i]) @ np.linalg.matrix_power(Z, exps[2 * i + 1]) for i in range(n)]
        ops.append((exps, kron(*mats)))
    return ops


W1 = weyl(1)
W2 = weyl(2)


def maps_weyl_to_weyl(U, n):
    W = W1 if n == 1 else W2
    for _, A in W:
        B = U @ A @ U.conj().T
        if not any(equal_up_to_phase(B, C) for _, C in W):
            return False
    return True


def is_weyl(A, n):
    W = W1 if n == 1 else W2
    return any(equal_up_to_phase(A, C) for _, C in W)


print("=" * 72)
print("C1  light-shift gate G(theta) vs CZ (Hrmo et al., Eq. 3)")
D = np.diag([w ** ((j * j) % d) for j in range(d)])
for theta, power in ((2 * np.pi / 3, 1), (-2 * np.pi / 3, -1)):
    G = np.diag([1.0 if j == k else np.exp(1j * theta) for j in range(d) for k in range(d)])
    Dp = D if power == 1 else D.conj().T
    CZp = CZ if power == 1 else CZ.conj().T
    ok = equal_up_to_phase(G, kron(Dp, Dp) @ CZp)
    print(f"  theta = {theta:+.4f}: G == phase * (D x D) CZ^({power:+d})  -> {ok}")
print(f"  D = diag(w^(j^2)) is a Clifford: {maps_weyl_to_weyl(D, 1)}; "
      f"D == Z P^2 up to phase: {equal_up_to_phase(D, Z @ P @ P)}")
cnot_from_cz = kron(I3, H) @ CZ @ kron(I3, H.conj().T)
cnot_from_cz_b = kron(I3, H.conj().T) @ CZ @ kron(I3, H)
print(f"  CNOT == (I x H) CZ (I x H^dag): {equal_up_to_phase(CNOT, cnot_from_cz)};  "
      f"CNOT == (I x H^dag) CZ (I x H): {equal_up_to_phase(CNOT, cnot_from_cz_b)}")
print(f"  CNOT_INV == (I x H^dag) CZ^-1 (I x H): "
      f"{equal_up_to_phase(CNOT.conj().T, kron(I3, H.conj().T) @ CZ.conj().T @ kron(I3, H))}")

print("=" * 72)
print("C2  every gate of the LRB circuits normalizes the Weyl group")
gates1 = {"H": H, "H_INV": H.conj().T, "P": P, "P_INV": P.conj().T, "X": X, "Z": Z}
gates2 = {"CNOT": CNOT, "CNOT_INV": CNOT.conj().T, "CZ": CZ, "CZ_INV": CZ.conj().T, "SWAP": SWAP}
for name, U in gates1.items():
    print(f"  {name:8s} Weyl->Weyl: {maps_weyl_to_weyl(U, 1)}")
for name, U in gates2.items():
    print(f"  {name:8s} Weyl->Weyl: {maps_weyl_to_weyl(U, 2)}")

print("=" * 72)
print("C3  qutrit T gate (third level of the Clifford hierarchy)")
zeta = np.exp(2j * np.pi / 9)
T3 = np.diag([1, zeta, zeta ** 8])
TXT = T3 @ X @ T3.conj().T
print(f"  T3 is Clifford: {maps_weyl_to_weyl(T3, 1)}")
print(f"  T3 X T3^dag is a Weyl operator: {is_weyl(TXT, 1)}")
print(f"  T3 X T3^dag is a Clifford:      {maps_weyl_to_weyl(TXT, 1)}")
# Frame-error demonstration: a fixed generic input state -> T3 -> X-basis
# readout (H_INV then Z basis), with and without a Weyl frame error X^a Z^b
# before T3. A Weyl frame error that commutes through to the readout can only
# permute the outcome distribution cyclically. (Highly symmetric inputs such
# as P H|0> hide the effect, so a generic state is used.)
_v = np.array([0.8, 0.5 + 0.2j, 0.1 - 0.3j], dtype=complex)
psi_in = _v / np.linalg.norm(_v)
def xbasis_probs(psi):
    amp = H.conj().T @ psi
    return np.abs(amp) ** 2
def is_cyclic_shift(p, q):
    return any(np.allclose(p, np.roll(q, s), atol=1e-9) for s in range(d))
p_ideal = xbasis_probs(T3 @ psi_in)
print(f"  X-basis probs ideal (T3 |psi_in>): {np.round(p_ideal, 4)}")
for (a, b), A in [((exps[0], exps[1]), A) for exps, A in W1 if exps != (0, 0)]:
    p_err = xbasis_probs(T3 @ A @ psi_in)
    print(f"    frame error X^{a} Z^{b} before T3 -> {np.round(p_err, 4)}  cyclic shift of ideal: "
          f"{is_cyclic_shift(p_err, p_ideal)}")
# Contrast: Clifford P in place of T3 -> every Weyl frame error is a cyclic shift.
p_ideal_c = xbasis_probs(P @ psi_in)
ok_shift = all(is_cyclic_shift(xbasis_probs(P @ A @ psi_in), p_ideal_c) for _, A in W1)
print(f"  same test with Clifford P in place of T3: every Weyl frame error is a cyclic shift: {ok_shift}")

print("=" * 72)
print("C4  no-reset syndrome extraction (ancilla starts in |k>)")
# Four-qutrit register: wires (q0, q1, q2, anc). Build gates on 4 qutrits.
def op_on(U, wires, n=4):
    """Embed a 1- or 2-qutrit unitary U acting on `wires` into n qutrits."""
    dim = d ** n
    out = np.zeros((dim, dim), dtype=complex)
    k = len(wires)
    for col in range(dim):
        digits = list(np.unravel_index(col, [d] * n))
        sub_in = int(np.ravel_multi_index([digits[wi] for wi in wires], [d] * k))
        for sub_out in range(d ** k):
            amp = U[sub_out, sub_in]
            if abs(amp) < 1e-15:
                continue
            new = digits.copy()
            out_digits = np.unravel_index(sub_out, [d] * k)
            for wi, val in zip(wires, out_digits):
                new[wi] = int(val)
            out[int(np.ravel_multi_index(new, [d] * n)), col] += amp
    return out

rng = np.random.default_rng(7)
def random_state(n):
    v = rng.normal(size=d ** n) + 1j * rng.normal(size=d ** n)
    return v / np.linalg.norm(v)

Xinv = X.conj().T
Zinv = Z.conj().T
# X-check S_X1 = X0 X1 X2^-1 measured by ancilla wire 3 with
# H a, CNOT a->0, CNOT a->1, CNOT_INV a->2, H_INV a, M a  (code_definitions.py)
SX = kron(X, X, Xinv, I3)
UX = (op_on(H.conj().T, [3]) @ op_on(CNOT.conj().T, [3, 2]) @ op_on(CNOT, [3, 1])
      @ op_on(CNOT, [3, 0]) @ op_on(H, [3]))
# Z-check S_Z1-like: CNOT 0->a, CNOT_INV 1->a, CNOT_INV 2->a (pattern of ancilla 5)
SZ = kron(Z, Zinv, Zinv, I3)
UZ = op_on(CNOT.conj().T, [2, 3]) @ op_on(CNOT.conj().T, [1, 3]) @ op_on(CNOT, [0, 3])

def eig_projector(S, s):
    return sum((w ** (-s * t)) * np.linalg.matrix_power(S, t) for t in range(d)) / d

def ancilla_distribution(state):
    probs = np.abs(state.reshape([d] * 4)) ** 2
    return np.round(probs.sum(axis=(0, 1, 2)), 9)

for label, S, U in (("X-check", SX, UX), ("Z-check", SZ, UZ)):
    table = {}
    for s in range(d):
        data = random_state(3)
        for k in range(d):
            anc = np.zeros(d, dtype=complex); anc[k] = 1
            psi = np.kron(data, anc)
            psi = eig_projector(S, s) @ psi
            psi /= np.linalg.norm(psi)
            dist = ancilla_distribution(U @ psi)
            outcome = int(np.argmax(dist))
            table[(s, k)] = (outcome, float(dist[outcome]))
    det = all(p > 1 - 1e-9 for (_, p) in table.values())
    offset_ok = all((table[(s, k)][0] - table[(s, 0)][0]) % d == k for s in range(d) for k in range(d))
    print(f"  {label}: deterministic outcome for every (s, k): {det}; "
          f"outcome(s, k) = outcome(s, 0) + k mod 3: {offset_ok}; "
          f"outcome(s, 0) for s = 0, 1, 2: {[table[(s, 0)][0] for s in range(d)]}")

print("=" * 72)
print("C5  two-qutrit interaction graph of the LRB circuit")
check_edges = [(0, 5), (1, 5), (3, 5), (1, 6), (2, 6), (4, 6), (7, 0), (7, 1), (7, 2), (8, 3), (8, 1), (8, 4)]
logical_edges = [(2, 3)]  # SWAP(2,3) in logical H, CZ(2,3) in logical P
G = nx.Graph()
G.add_nodes_from(range(9))
G.add_edges_from(check_edges + logical_edges)
print(f"  nodes 9, edges {G.number_of_edges()}, degrees {dict(G.degree())}")
print(f"  planar: {nx.check_planarity(G)[0]}, bipartite: {nx.is_bipartite(G)}, "
      f"girth: {min(len(c) for c in nx.minimum_cycle_basis(G))}")
Gc = nx.Graph(); Gc.add_edges_from(check_edges)
print(f"  check-only graph: edges {Gc.number_of_edges()}, bipartite {nx.is_bipartite(Gc)}")
def embeds_in_grid(graph, rows, cols):
    grid = nx.grid_2d_graph(rows, cols)
    gm = isomorphism.GraphMatcher(grid, graph)
    return gm.subgraph_is_monomorphic()
for rows, cols in ((3, 3), (3, 4), (4, 4), (3, 5), (4, 5)):
    print(f"  embeds (as a subgraph, no SWAPs) in a {rows}x{cols} square grid: "
          f"full {embeds_in_grid(G, rows, cols)}, checks-only {embeds_in_grid(Gc, rows, cols)}")
odd = [c for c in nx.cycle_basis(G) if len(c) % 2 == 1]
print(f"  an odd cycle created by the logical (2,3) edge: {odd[0] if odd else None} "
      f"(any bipartite lattice, e.g. square or heavy-hex, then needs routing)")
# All 3x3 embeddings of the check graph: where do data 2 and 3 sit?
grid = nx.grid_2d_graph(3, 3)
gm = isomorphism.GraphMatcher(grid, Gc)
dists, n_emb, swap_ok = set(), 0, True
# Stabilizer supports after a *virtual* SWAP(2,3) (relabel wires 2 <-> 3):
supports_after = [{0, 1, 2}, {1, 3, 4}, {0, 1, 3}, {1, 2, 4}]  # Z1', Z2', X1', X2'
for mapping in gm.subgraph_monomorphisms_iter():
    inv = {v: k for k, v in mapping.items()}  # code wire -> grid node
    n_emb += 1
    dists.add(nx.shortest_path_length(grid, inv[2], inv[3]))
    anc_nbrs = [{mapping[nb] for nb in grid.neighbors(inv[a])} for a in (5, 6, 7, 8)]
    if not all(any(sup == nb for nb in anc_nbrs) for sup in supports_after):
        swap_ok = False
print(f"  3x3 embeddings of the check graph: {n_emb}; grid distance between data 2 and 3: {sorted(dists)}")
print(f"  after a virtual SWAP(2,3) every new stabilizer support is the neighbour set of some ancilla "
      f"site (checks stay nearest-neighbour if ancilla roles are permuted): {swap_ok}")

print("=" * 72)
print("C6  lookup-table sizes for a real-time decoder")
for wnd in (1, 2, 3):
    n = 3 ** (4 * wnd)
    print(f"  window of {wnd} round(s): 3^{4 * wnd} = {n} syndrome patterns; "
          f"table of 2-bit Z_3 shift + 1-bit reject flag = {n * 3 / 8 / 1024:.2f} KiB")

print("=" * 72)
print("C7  shot noise of the fidelity estimator |mean(w^k)| (used in H2_platform_budget.py B5)")
# Outcome 0 with prob (1+2F)/3, outcomes 1 and 2 with prob (1-F)/3 each.
# Delta method: Var = (1 - F)(1 + 2F) / (2 N).
rng_mc = np.random.default_rng(11)
for F_true, N in ((0.99, 1000), (0.9, 1000), (0.97, 10000)):
    probs = [(1 + 2 * F_true) / 3, (1 - F_true) / 3, (1 - F_true) / 3]
    reps = 4000
    counts = rng_mc.multinomial(N, probs, size=reps)
    est = np.abs(counts @ np.array([1, w, w ** 2]) / N)
    sd_mc = est.std(ddof=1)
    sd_th = np.sqrt((1 - F_true) * (1 + 2 * F_true) / (2 * N))
    print(f"  F = {F_true}, N = {N}: Monte Carlo sd {sd_mc:.3e}, delta-method sd {sd_th:.3e}, "
          f"ratio {sd_mc / sd_th:.3f}")
