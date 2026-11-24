"""H2 (other platforms): exact algebra, graph and counting checks.

Run:  module load python/3.11 && python3 h2_algebra_checks.py
Runtime: about 20 s on one core. Writes nothing. Reads only this worktree's
src/lrb/code_definitions.py and sdim's own gate matrices (sdim.unitary), so the
conventions are the simulator's: X|j> = |j+1>, Z = diag(w^j),
H[m, n] = w^(mn)/sqrt(3), P = diag(w^(j(j-1)/2)), CNOT|c,t> = |c, t+c>,
CZ = (I x H) CNOT (I x H^dag), w = exp(2 pi i/3).

Blocks
  A1  Physical gate content of each logical token, catalog means, per check
      round and per depth-m circuit (counts from the repository's own
      gate-sequence parser).
  A2  Light-shift (LS) gate of Hrmo et al.: G(+-2pi/3) versus sdim's CZ, and
      the exact local gates needed to turn one G into each CNOT / CNOT_INV.
  A3  Interaction graph: check-only graph versus the 3x3 grid, the odd cycle
      closed by the logical edge (2,3), distances and shortest paths.
  A4  Every gate of the circuits normalizes the Weyl group (symplectic images
      extracted from the matrices); syndrome transfer maps of the logical
      tokens (how a data error's syndrome changes across a logical gate).
  A5  Qutrit T gate T3 = diag(1, z, z^8), z = exp(2 pi i/9) (a member of the
      Howard-Vala family): third level of the Clifford hierarchy, not
      Clifford; a frame error with an X component before T3 changes the
      X-basis statistics by more than a cyclic shift.
  A6  No reset needed: on the full 9-qutrit encoded block, an ancilla that
      starts in |k> returns k + s(E) deterministically for every check.
  A7  Lookup-table sizes for a per-round decoder.
  A8  Shot noise of the fidelity estimators (signed and legacy) and the
      Rayleigh floor sqrt(pi/(4N)) of the legacy metric at zero fidelity.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

import sdim.unitary as su

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "h2_code_definitions", os.path.join(WORKTREE, "src", "lrb", "code_definitions.py"))
cd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cd          # dataclasses resolve the module here
_spec.loader.exec_module(cd)
Code = cd.FoldedQutritDetectionCode
CATALOG = cd.single_qutrit_cliffords_exhaustive_strings

d = 3
w = np.exp(2j * np.pi / 3)
I3 = np.eye(3, dtype=complex)
X = su.generate_x_matrix(3).astype(complex)
Z = su.generate_z_matrix(3).astype(complex)
H = su.generate_h_matrix(3).astype(complex)
P = su.generate_p_matrix(3).astype(complex)
CNOT = su.generate_cnot_matrix(3).astype(complex)      # first tensor factor = control
CZ = np.kron(I3, H) @ CNOT @ np.kron(I3, H.conj().T)   # sdim's GeneralizedCZGate
SWAP = np.zeros((9, 9), dtype=complex)
for a, b in itertools.product(range(3), repeat=2):
    SWAP[3 * b + a, 3 * a + b] = 1.0
ONE = {"H": H, "H_INV": H.conj().T, "P": P, "P_INV": P.conj().T,
       "X": X, "X_INV": X.conj().T, "Z": Z, "Z_INV": Z.conj().T, "I": I3}
TWO = {"CNOT": CNOT, "CNOT_INV": CNOT.conj().T, "CZ": CZ, "CZ_INV": CZ.conj().T, "SWAP": SWAP}


def hr(title: str) -> None:
    print("=" * 78)
    print(title)


def equal_up_to_phase(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(b[k]) < tol:
        return False
    ph = a[k] / b[k]
    return abs(abs(ph) - 1) < 1e-7 and np.allclose(a, ph * b, atol=1e-8)


def weyl1(a: int, b: int) -> np.ndarray:
    return np.linalg.matrix_power(X, a % 3) @ np.linalg.matrix_power(Z, b % 3)


# --------------------------------------------------------------------- A1
hr("A1  physical gate content (repository parser) and catalog means")


def token_content(tok: str) -> tuple[int, int]:
    gates, controls, targets = Code.elementary_single_qudit_logical_operator_gate_sequence(tok)
    n2 = n1 = 0
    for g, c, t in zip(gates, controls, targets):
        if t is not None:
            n2 += 1
        else:
            n1 += len(c) if isinstance(c, list) else 1
    return n2, n1


content = {t: token_content(t) for t in ("H", "P", "X", "Z", "I")}
print("  per token (two-qutrit gates, one-qutrit gates):", content)
tok = Counter("".join(CATALOG))
n_cat = len(CATALOG)
mean_tok = {g: tok[g] / n_cat for g in "HPXZI"}
two_per = sum(mean_tok[g] * content[g][0] for g in "HPXZI")
one_per = sum(mean_tok[g] * content[g][1] for g in "HPXZI")
print(f"  catalog: {n_cat} entries, {len(set(CATALOG))} distinct strings; mean tokens per Clifford "
      + ", ".join(f"{g} {mean_tok[g]:.3f}" for g in "HPXZI"))
print(f"  mean per logical Clifford: two-qutrit {two_per:.3f}, one-qutrit {one_per:.3f}")
print(f"    SWAP(2,3) done by relabelling: two-qutrit {two_per - mean_tok['H']:.3f} (only CZ(2,3) left)")
print(f"    logical X, Z as frame updates: one-qutrit {one_per - 2 * mean_tok['X'] - 2 * mean_tok['Z']:.3f}")
check_ops = [op for fac in (Code.z_ALT_stabilizer_measurement_ancillae_circuit,
                            Code.x_stabilizer_measurement_ancillae_circuit) for op in fac().operations]
n2_round = sum(op.target_index is not None for op in check_ops)
n1_round = sum(op.target_index is None and op.gate_name not in ("M",) for op in check_ops)
n_meas = sum(op.gate_name == "M" for op in check_ops)
print(f"  per check round: {n2_round} two-qutrit, {n1_round} one-qutrit, {n_meas} measurements, "
      f"{len(Code.reset_measurement_wires().operations)} resets")
for m in (2, 6, 10, 22):
    n2 = 2 * m * two_per + (m + 1) * n2_round
    n1 = 2 * m * one_per + (m + 1) * n1_round
    print(f"  depth m = {m:2d}: {2 * m} Cliffords + {m + 1} rounds -> {n2:6.1f} two-qutrit, {n1:6.1f} one-qutrit, "
          f"{(m + 1) * n_meas} mid-circuit measurements (expected values over the catalog)")
# SI1000 nominal fault weight per round (brief): sum of location probabilities / p
gate_w = n2_round * 1.0 + n1_round * 0.1
meas_seq = n_meas * (8 * 2 + 1 + 5)          # idle 2p on 8 other wires, ancilla depol p, flip 5p
meas_par = 5 * 2 + n_meas * (1 + 5)          # one readout layer: idle 2p on the 5 data wires only
reset_w = 5 * 2 + 4 * 2
print(f"  SI1000 weight per round / p: gates {gate_w:.1f}, sequential readout {meas_seq}, reset {reset_w}, "
      f"total {gate_w + meas_seq + reset_w:.1f}; with one parallel readout layer {gate_w + meas_par + reset_w:.1f}")
print(f"  idle weight per data qutrit per round / p: sequential {4 * 2 + 2}, parallel {1 * 2 + 2}")

# --------------------------------------------------------------------- A2
hr("A2  light-shift gate G(theta) (Hrmo et al. Eq. 3) versus sdim's CZ")
D = np.diag([w ** ((j * j) % 3) for j in range(3)])
print(f"  sdim CZ is diagonal: {np.allclose(CZ, np.diag(np.diag(CZ)))}; "
      f"CZ = diag(w^(c t)): {np.allclose(CZ, np.diag([w ** ((c * t) % 3) for c in range(3) for t in range(3)]))}")
print(f"  identity 1 - delta_jk = (j - k)^2 mod 3 for all j, k: "
      f"{all((1 - (j == k)) == ((j - k) ** 2) % 3 for j in range(3) for k in range(3))}")
for th, s in ((2 * np.pi / 3, +1), (-2 * np.pi / 3, -1)):
    G = np.diag([1.0 if j == k else np.exp(1j * th) for j in range(3) for k in range(3)])
    Ds = D if s > 0 else D.conj().T
    CZs = CZ if s > 0 else CZ.conj().T
    print(f"  theta = {s:+d}*2pi/3: G == (D x D) CZ^({s:+d}) exactly: {np.allclose(G, np.kron(Ds, Ds) @ CZs)}")
print(f"  D = diag(w^(j^2)) = diag(1, w, w); D == Z P^2 exactly: {np.allclose(D, Z @ P @ P)}")
Gp = np.diag([1.0 if j == k else w for j in range(3) for k in range(3)])
Gm = Gp.conj()
c1 = np.kron(I3, H.conj().T) @ np.kron(D.conj().T, D.conj().T) @ Gp @ np.kron(I3, H)
c2 = np.kron(I3, H.conj().T) @ np.kron(D, D) @ Gm @ np.kron(I3, H)
print(f"  CNOT     == (I x H^dag)(D^dag x D^dag) G(+2pi/3) (I x H): {np.allclose(CNOT, c1)}")
print(f"  CNOT_INV == (I x H^dag)(D x D) G(-2pi/3) (I x H):         {np.allclose(CNOT.conj().T, c2)}")
print(f"  without the D corrections (I x H^dag) G (I x H) == CNOT up to phase: "
      f"{equal_up_to_phase(CNOT, np.kron(I3, H.conj().T) @ Gp @ np.kron(I3, H))}")
print("  => each check CNOT = one native LS gate + D^dag (diagonal Clifford) on the control"
      " + H before and H^dag D^dag after on the target")

# --------------------------------------------------------------------- A3
hr("A3  interaction graph of the circuit")
edges = set()
for op in check_ops:
    if op.target_index is not None:
        edges.add(tuple(sorted((op.qudit_index, op.target_index))))
logical_edges = set()
for t in ("H", "P", "H_INV", "P_INV", "X", "Z"):
    for op in Code.single_qudit_logical_gate_circuit(t).operations:
        if op.target_index is not None:
            logical_edges.add(tuple(sorted((op.qudit_index, op.target_index))))
print(f"  check edges ({len(edges)}): {sorted(edges)}")
print(f"  logical two-qutrit edges: {sorted(logical_edges)} (SWAP(2,3) in H, CZ(2,3) in P)")
Gc = nx.Graph(sorted(edges))
grid = nx.grid_2d_graph(3, 3)
gm = isomorphism.GraphMatcher(grid, Gc)
isos = list(gm.isomorphisms_iter())
print(f"  check graph isomorphic to the 3x3 grid: {nx.is_isomorphic(Gc, grid)}; number of embeddings {len(isos)}")
dist23 = {nx.shortest_path_length(grid, next(g for g, c in iso.items() if c == 2),
                                  next(g for g, c in iso.items() if c == 3)) for iso in isos}
centre = {iso[(1, 1)] for iso in isos}
print(f"  grid distance between data 2 and 3 over all embeddings: {sorted(dist23)}; wire at the centre: {centre}")
Gf = nx.Graph(sorted(edges | logical_edges))
odd = [c for c in nx.minimum_cycle_basis(Gf) if len(c) % 2 == 1]
print(f"  full graph bipartite: {nx.is_bipartite(Gf)}; an odd cycle through (2,3): {sorted(odd[0]) if odd else None} "
      f"(length {len(odd[0]) if odd else 0})")
paths = list(nx.all_shortest_paths(Gc, 2, 3))
print(f"  shortest 2-3 paths in the check graph (length {len(paths[0]) - 1}): {paths}")
print(f"  every shortest path passes through another data qutrit: "
      f"{all(any(v in (0, 1, 4) for v in p[1:-1]) for p in paths)}")
print(f"  max degree of the full graph: {max(dict(Gf.degree()).values())}; planar: {nx.check_planarity(Gf)[0]}")

# --------------------------------------------------------------------- A4
hr("A4  Clifford normalization and syndrome transfer maps of the logical tokens")
W1 = {(a, b): weyl1(a, b) for a in range(3) for b in range(3)}
W2 = {(a0, b0, a1, b1): np.kron(weyl1(a0, b0), weyl1(a1, b1))
      for a0, b0, a1, b1 in itertools.product(range(3), repeat=4)}


def sympl_image(U: np.ndarray, table: dict) -> dict:
    img = {}
    for key, A in table.items():
        B = U @ A @ U.conj().T
        hit = [k2 for k2, C in table.items() if equal_up_to_phase(B, C)]
        if len(hit) != 1:
            return {}
        img[key] = hit[0]
    return img


IMG1 = {g: sympl_image(U, W1) for g, U in ONE.items()}
IMG2 = {g: sympl_image(U, W2) for g, U in TWO.items()}
print("  one-qutrit gates map every Weyl operator to a Weyl operator: "
      + ", ".join(f"{g} {bool(v)}" for g, v in IMG1.items()))
print("  two-qutrit gates map every Weyl operator to a Weyl operator: "
      + ", ".join(f"{g} {bool(v)}" for g, v in IMG2.items()))
NQ = 9


def prop(v: np.ndarray, gate: str, q: int, t: int | None) -> np.ndarray:
    """Propagate a 9-qutrit Weyl vector v = (a_0..a_8, b_0..b_8) through one gate."""
    v = v.copy()
    if t is None:
        a, b = IMG1[gate][(int(v[q]), int(v[NQ + q]))]
        v[q], v[NQ + q] = a, b
    else:
        a0, b0, a1, b1 = IMG2[gate][(int(v[q]), int(v[NQ + q]), int(v[t]), int(v[NQ + t]))]
        v[q], v[NQ + q], v[t], v[NQ + t] = a0, b0, a1, b1
    return v


def mk(xd: dict, zd: dict) -> np.ndarray:
    v = np.zeros(2 * NQ, dtype=int)
    for q, e in xd.items():
        v[q] = e % 3
    for q, e in zd.items():
        v[NQ + q] = e % 3
    return v


def symp(u: np.ndarray, v: np.ndarray) -> int:
    return int((u[:NQ] @ v[NQ:] - u[NQ:] @ v[:NQ]) % 3)


STAB = [mk({}, {0: 1, 1: -1, 3: -1}), mk({}, {1: 1, 2: 1, 4: -1}),
        mk({0: 1, 1: 1, 2: -1}, {}), mk({1: -1, 3: 1, 4: -1}, {})]
XL, ZL = mk({0: 1, 3: 1}, {}), mk({}, {0: 1, 2: 1})
print(f"  stabilizers commute pairwise: {all(symp(s, t) == 0 for s in STAB for t in STAB)}; "
      f"commute with X_L, Z_L: {all(symp(s, XL) == 0 and symp(s, ZL) == 0 for s in STAB)}; "
      f"<X_L, Z_L> = {symp(XL, ZL)} (non-zero)")


def syndrome(v: np.ndarray) -> np.ndarray:
    return np.array([symp(v, s) for s in STAB], dtype=int)


def through_token(v: np.ndarray, tok_: str) -> np.ndarray:
    for op in Code.single_qudit_logical_gate_circuit(tok_).operations:
        v = prop(v, op.gate_name, op.qudit_index, op.target_index)
    return v


units = [mk({q: 1}, {}) for q in range(5)] + [mk({}, {q: 1}) for q in range(5)]
for t in ("H", "P", "X", "Z"):
    # syndrome after the token as a linear function of the syndrome before it
    pairs = [(syndrome(u), syndrome(through_token(u, t))) for u in units]
    A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
    rows = []
    for i in range(4):               # solve B[:, i] = A @ T[i, :] (mod 3) row by row
        sol = [c for c in itertools.product(range(3), repeat=4) if np.array_equal((A @ np.array(c)) % 3, B[:, i])]
        rows.append(sol[0] if len(sol) == 1 else None)
    T = None if any(r is None for r in rows) else np.array(rows)
    print(f"  token {t}: syndrome transfer map (s_after = T s_before mod 3) exists: {T is not None}; "
          f"T = {T.tolist() if T is not None else None}")
print("  (a non-identity T means a data error keeps its Weyl type only up to the logical gate: a real-time"
      " decoder or frame tracker needs the known Clifford sequence; T is a 4x4 matrix over Z_3)")

# --------------------------------------------------------------------- A5
hr("A5  qutrit T gate T3 = diag(1, z, z^8), z = exp(2 pi i/9)")
zeta = np.exp(2j * np.pi / 9)
T3 = np.diag([1, zeta, zeta ** 8])
upsilon = lambda zp, gp, ep: ((0, (6 * zp + 2 * gp + 3 * ep) % 9, (6 * zp + gp + 6 * ep) % 9))
print(f"  T3 exponents (0,1,8) are in the Howard-Vala family upsilon = (0, 6z'+2g'+3e', 6z'+g'+6e') mod 9: "
      f"{any(upsilon(zp, gp, ep) == (0, 1, 8) for zp in range(3) for gp in (1, 2) for ep in range(3))}")
print(f"  T3 is a Clifford: {bool(sympl_image(T3, W1))}")
third = all(bool(sympl_image(T3 @ A @ T3.conj().T, W1)) for A in W1.values())
print(f"  T3 W T3^dag is a Clifford for every Weyl W (third level): {third}")
TXT = T3 @ X @ T3.conj().T
print(f"  T3 X T3^dag is a Weyl operator: {any(equal_up_to_phase(TXT, C) for C in W1.values())}")
v = np.array([0.8, 0.5 + 0.2j, 0.1 - 0.3j]); psi = v / np.linalg.norm(v)


def xprobs(state: np.ndarray) -> np.ndarray:
    return np.abs(H.conj().T @ state) ** 2


ideal = xprobs(T3 @ psi)
print(f"  X-basis statistics after T3 on a generic state: {np.round(ideal, 4)}")
for (a, b), A in W1.items():
    if (a, b) == (0, 0):
        continue
    pe = xprobs(T3 @ A @ psi)
    shift = any(np.allclose(pe, np.roll(ideal, s), atol=1e-9) for s in range(3))
    print(f"    frame error X^{a} Z^{b} before T3: {np.round(pe, 4)}; a cyclic shift of the ideal: {shift}")
ok_cliff = all(any(np.allclose(xprobs(P @ A @ psi), np.roll(xprobs(P @ psi), s), atol=1e-9) for s in range(3))
               for A in W1.values())
print(f"  control: with the Clifford P in place of T3 every frame error is a cyclic shift: {ok_cliff}")

# --------------------------------------------------------------------- A6
hr("A6  no-reset syndrome extraction on the full 9-qutrit block (statevector)")


def apply1(psi9: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    s = psi9.reshape([3] * NQ)
    s = np.moveaxis(np.tensordot(U, s, axes=([1], [q])), 0, q)
    return s.reshape(-1)


def apply2(psi9: np.ndarray, U: np.ndarray, q: int, t: int) -> np.ndarray:
    s = psi9.reshape([3] * NQ)
    s = np.tensordot(U.reshape(3, 3, 3, 3), s, axes=([2, 3], [q, t]))
    s = np.moveaxis(s, [0, 1], [q, t])
    return s.reshape(-1)


def run_ops(psi9: np.ndarray, ops) -> np.ndarray:
    for op in ops:
        if op.gate_name == "M":
            continue
        qs = op.qudit_index if isinstance(op.qudit_index, (list, tuple)) else [op.qudit_index]
        for q in qs:
            if op.target_index is None:
                psi9 = apply1(psi9, ONE[op.gate_name], q)
            else:
                psi9 = apply2(psi9, TWO[op.gate_name], q, op.target_index)
    return psi9


psi0 = np.zeros(3 ** NQ, dtype=complex); psi0[0] = 1.0
plus_L = run_ops(psi0, Code.logical_plus_initial_state().operations)


def apply_weyl(psi9: np.ndarray, vec: np.ndarray) -> np.ndarray:
    for q in range(NQ):
        if vec[q] % 3 or vec[NQ + q] % 3:
            psi9 = apply1(psi9, weyl1(int(vec[q]), int(vec[NQ + q])), q)
    return psi9


ev = [np.vdot(plus_L, apply_weyl(plus_L.copy(), s)).round(9) for s in STAB]
print(f"  encoded |+>_L: <S> for (S_Z1, S_Z2, S_X1, S_X2) = {ev}; <X_L> = {np.vdot(plus_L, apply_weyl(plus_L.copy(), XL)).round(9)}")
rng = np.random.default_rng(7)
ok_det = ok_shift = True
n_cases = 0
sign_fit = Counter()
for trial in range(25):
    e = mk({q: int(rng.integers(3)) for q in range(5)}, {q: int(rng.integers(3)) for q in range(5)})
    base = apply_weyl(plus_L.copy(), e)
    ref = None
    for kvec in [(0, 0, 0, 0)] + [tuple(int(x) for x in rng.integers(0, 3, 4)) for _ in range(3)]:
        st = base.copy()
        for anc, kk in zip((5, 6, 7, 8), kvec):
            st = apply1(st, np.linalg.matrix_power(X, kk), anc)
        out = run_ops(st, check_ops)
        probs = (np.abs(out.reshape([3] * NQ)) ** 2).sum(axis=(0, 1, 2, 3, 4)).reshape(-1)
        idx = int(np.argmax(probs))
        ok_det &= bool(probs[idx] > 1 - 1e-9)
        res = np.array(np.unravel_index(idx, [3] * 4))
        if ref is None:
            ref = res
            s_e = syndrome(e)
            for i in range(4):
                if s_e[i]:
                    sign_fit[(i, "+" if res[i] == s_e[i] else ("-" if res[i] == (-s_e[i]) % 3 else "?"))] += 1
        ok_shift &= bool(all((res[i] - ref[i] - kvec[i]) % 3 == 0 for i in range(4)))
        n_cases += 1
print(f"  {n_cases} (error, ancilla start) cases: outcome deterministic: {ok_det}; outcome = k + s(E): {ok_shift}")
signs = {i: sorted({k[1] for k, c in sign_fit.items() if k[0] == i and c > 0}) for i in range(4)}
print(f"  sign of measured outcome versus symplectic syndrome per check (S_Z1, S_Z2, S_X1, S_X2): {signs}")

# --------------------------------------------------------------------- A7
hr("A7  lookup-table sizes (2-bit Z_3 shift + 1 reject bit per entry)")
for wnd in (1, 2, 3):
    n = 3 ** (4 * wnd)
    print(f"  {wnd} round(s) of syndrome: 3^{4 * wnd} = {n} entries = {n * 3 / 8 / 1024:.2f} KiB")
print(f"  with the Clifford-dependent transfer maps: 216 catalog Cliffords x one 4x4 matrix over Z_3 each")

# --------------------------------------------------------------------- A8
hr("A8  shot noise of the fidelity estimators")
rng2 = np.random.default_rng(11)
for lam, N in ((0.98, 1000), (0.9, 1000), (0.5, 2000), (0.0, 1000), (0.0, 10000)):
    S = (1 + 2 * lam) / 3
    probs3 = [S, (1 - S) / 2, (1 - S) / 2]
    counts = rng2.multinomial(N, probs3, size=20000)
    z = counts @ np.array([1, w, w ** 2]) / N
    sd_signed = z.real.std(ddof=1)
    th_signed = np.sqrt((1 - lam) * (1 + 2 * lam) / (2 * N))
    leg = np.abs(z)
    print(f"  lambda = {lam:4.2f}, N = {N:5d}: signed sd MC {sd_signed:.3e} vs (1-l)(1+2l)/(2N) {th_signed:.3e}; "
          f"legacy mean {leg.mean():.4f}" + (f" vs Rayleigh floor sqrt(pi/(4N)) {np.sqrt(np.pi / (4 * N)):.4f}"
                                             if lam == 0 else f" (true {lam})"))
