"""Exhaustive search for 4-gate Clifford encoders of |+_L> on a given data tree.

Any circuit with one two-qutrit gate per tree edge equals, up to local Cliffords
(free), a sequence CZ_e o (R_i x R_j) where R only matters through which Pauli axis
it maps onto Z (4 axes per qutrit): diagonal Cliffords commute through CZ and a
CZ power is a local rescaling. So the search runs over the 24 edge orders and
4^8 axis choices from |0>^5, and tests local-Clifford (LC) equivalence of the
result with |+_L> exactly (entropy-profile filter, then the linear test
S1 Q^T Lambda S2^T = 0 for block-diagonal Q with det Q_i = 1).
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
import encoder_trees as ET  # noqa: E402

D, N = 3, 5
# target stabilizer matrix rows (x | z)
TARGET = np.zeros((5, 10), dtype=int)
for r, v in enumerate([ET.G3, ET.G4, ET.XL]):
    TARGET[r, :5] = np.array(v) % D
for r, v in enumerate([ET.G1, ET.G2]):
    TARGET[3 + r, 5:] = np.array(v) % D

AXES = [(0, 1), (1, 0), (1, 1), (1, 2)]   # Pauli axis (x, z) to be mapped to Z


def axis_rep(p):
    """A det-1 matrix M on (x, z) column vectors with M p = (0, 1) (up to scale)."""
    for a, b, c, d in itertools.product(range(D), repeat=4):
        if (a * d - b * c) % D != 1:
            continue
        x = (a * p[0] + b * p[1]) % D
        z = (c * p[0] + d * p[1]) % D
        if x == 0 and z != 0:
            return np.array([[a, b], [c, d]])
    raise ValueError


REPS = [axis_rep(p) for p in AXES]


def local(S, i, M):
    S = S.copy()
    x, z = S[:, i].copy(), S[:, 5 + i].copy()
    S[:, i] = (M[0, 0] * x + M[0, 1] * z) % D
    S[:, 5 + i] = (M[1, 0] * x + M[1, 1] * z) % D
    return S


def cz(S, i, j):
    S = S.copy()
    S[:, 5 + i] = (S[:, 5 + i] + S[:, j]) % D
    S[:, 5 + j] = (S[:, 5 + j] + S[:, i]) % D
    return S


def entropy(S, A):
    """|A| - dim of stabilizer subgroup supported in A."""
    cols = [q for q in range(N) if q not in A] + [5 + q for q in range(N) if q not in A]
    # subgroup supported in A = combinations whose restriction outside A vanishes
    M = S[:, cols] % D
    rank = ET.rref(M).shape[0] if M.size else 0
    return len(A) - (S.shape[0] - rank)


SUBSETS = [A for r in (1, 2) for A in itertools.combinations(range(N), r)]
TPROF = tuple(entropy(TARGET, A) for A in SUBSETS)


def nullspace_mod(A):
    A = np.array(A, dtype=int) % D
    R = ET.rref(A)
    rows, cols = R.shape[0], A.shape[1]
    piv = []
    for r in range(rows):
        piv.append(next(c for c in range(cols) if R[r, c]))
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=int)
        v[f] = 1
        for r, pc in enumerate(piv):
            v[pc] = (-R[r, f]) % D
        basis.append(v)
    return basis


LAM = np.block([[np.zeros((5, 5), int), np.eye(5, dtype=int)], [-np.eye(5, dtype=int), np.zeros((5, 5), int)]])


def lc_equivalent(S1, S2):
    """Exists Q = diag(Q_i in SL(2,3)) with rowspace(S1 Q^T) = rowspace(S2)?"""
    # unknowns q_i = (a_i, b_i, c_i, d_i): x'_i = a x_i + b z_i, z'_i = c x_i + d z_i
    eqs = []
    W = (S2 @ LAM.T) % D        # (S1 Q^T) Lam S2^T = 0  <=>  (S1 Q^T) (S2 Lam^T)^T = 0
    for r1 in range(5):
        for r2 in range(5):
            row = np.zeros(4 * N, dtype=int)
            for i in range(N):
                x1, z1 = S1[r1, i], S1[r1, 5 + i]
                wx, wz = W[r2, i], W[r2, 5 + i]
                row[4 * i + 0] += x1 * wx
                row[4 * i + 1] += z1 * wx
                row[4 * i + 2] += x1 * wz
                row[4 * i + 3] += z1 * wz
            eqs.append(row % D)
    basis = nullspace_mod(eqs)
    if len(basis) > 12:
        raise RuntimeError("solution space too large")
    for coeffs in itertools.product(range(D), repeat=len(basis)):
        q = (np.array(coeffs) @ np.array(basis)) % D if basis else np.zeros(4 * N, int)
        if all((q[4 * i] * q[4 * i + 3] - q[4 * i + 1] * q[4 * i + 2]) % D == 1 for i in range(N)):
            return True
    return False


def search(tree, stop_after=None):
    edges = [tuple(sorted(e)) for e in tree]
    start = np.zeros((5, 10), dtype=int)
    for i in range(N):
        start[i, 5 + i] = 1
    sols = []
    seen_orders = set()
    for order in itertools.permutations(edges):
        for axes in itertools.product(range(4), repeat=8):
            S = start
            for k, (i, j) in enumerate(order):
                S = local(S, i, REPS[axes[2 * k]])
                S = local(S, j, REPS[axes[2 * k + 1]])
                S = cz(S, i, j)
            prof = tuple(entropy(S, A) for A in SUBSETS)
            if prof != TPROF:
                continue
            if lc_equivalent(S, TARGET):
                sols.append((order, axes))
                if stop_after and len(sols) >= stop_after:
                    return sols
    return sols


if __name__ == "__main__":
    lbl = lambda T: " ".join(sorted(f"{min(e)}{max(e)}" for e in T))
    trees = {"paper tree (sanity check)": [(0, 1), (0, 3), (1, 2), (2, 4)],
             "star at q1": [(0, 1), (1, 2), (1, 3), (1, 4)],
             "star at q2": [(0, 2), (1, 2), (2, 3), (2, 4)],
             "star at q3": [(0, 3), (1, 3), (2, 3), (3, 4)],
             "star at q4": [(0, 4), (1, 4), (2, 4), (3, 4)]}
    for name, T in trees.items():
        sols = search([frozenset(e) for e in T], stop_after=1)
        print(f"{name:28s} edges {lbl([frozenset(e) for e in T])}: "
              + (f"4-gate Clifford encoder EXISTS, e.g. order {sols[0][0]} axes {sols[0][1]}" if sols
                 else "no 4-gate Clifford encoder (exhaustive)"), flush=True)
