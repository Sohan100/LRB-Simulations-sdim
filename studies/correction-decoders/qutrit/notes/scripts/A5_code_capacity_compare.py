#!/usr/bin/env python3
"""A5: is a distance-3 qutrit code needed before correction beats detection in
qutrit LRB, and which code is the best next step?

Exact (and, where marked, Monte Carlo) comparison of small qutrit stabilizer
codes.  Everything is symplectic linear algebra mod 3 plus exact discrete
Fourier transforms over Z_3^m; numpy only.

Noise models (one "round" = one syndrome-extraction round):
  * code capacity: each data qutrit independently suffers, with probability q,
    one of the 8 non-identity Weyl operators X^a Z^b, chosen uniformly;
    syndromes are perfect.
  * phenomenological: the same data noise, then every check outcome is
    shifted by +1 or by -1 with probability q_m/2 each; after T noisy rounds a
    final perfect syndrome round (no extra data noise).

Codes (n data qutrits, r checks, one ancilla per check):
  512  [[5,1,2]]_3  the repository's folded surface code (brief, code_definitions.py)
  513  [[5,1,3]]_3  cyclic shifts of X Z Z^-1 X^-1 I
  713  [[7,1,3]]_3  qutrit Steane / 2D colour code with a star bipartition
  913  [[9,1,3]]_3  rotated surface code (checkerboard Z exponents)
  1313 [[13,1,3]]_3 unrotated surface code folded along its diagonal; built by
                    the same rule that reproduces the repository code at D = 2

Sections
  S1  constructions: commutation, rank, k, distance (brute force), counts of
      minimum-weight logical operators, quantum Hamming / Singleton bounds
  S2  logical gates (sdim frame conventions): repository H_L/P_L, colour-code
      star-transversal gates, fold-transversal gates of the folded codes,
      local-Clifford x permutation automorphism searches (meet in the middle),
      diagonal (phase-type) logical gates over every qutrit pairing
  S3  one-ancilla-per-check syndrome extraction: hook and fold-pair fault sets,
      safe coupling orders, Tanner-graph planarity
  S4  code capacity, exact: (syndrome, logical) distribution by DFT, weight
      enumerators, MLD / detection / hybrid, leading coefficients, break-even
  S5  phenomenological, exact: transfer matrices for detection and for the
      memoryless hybrid / MLD-lookup decoders; exact leading-order
      coefficients of space-time decoders (MLD, detection, hybrid) by
      enumerating all fault sets of weight <= 2
  S6  phenomenological, Monte Carlo: space-time MLD by an exact Bayesian
      forward filter (validation of S5 and break-even search)

Run (login node; module load python/3.11):
    python A5_code_capacity_compare.py            # all sections, about 4 min
    python A5_code_capacity_compare.py --quick    # short grids, no slow searches, small MC (~1.5 min)
    python A5_code_capacity_compare.py --mc913    # adds the joint space-time MC of [[9,1,3]]_3 (~2.5 min more)
Outputs: A5_out/A5_results.json, A5_out/A5_run_log.txt, A5_out/A5_eps_vs_q.png
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "A5_out")
LOG: list[str] = []
RESULTS: dict = {}
OMEGA = np.exp(2j * np.pi / 3)


def log(*args) -> None:
    s = " ".join(str(a) for a in args)
    print(s, flush=True)
    LOG.append(s)


def frac_str(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ---------------------------------------------------------------------------
# GF(3) linear algebra
# ---------------------------------------------------------------------------

def rref3(A):
    A = np.atleast_2d(np.array(A, dtype=np.int64) % 3)
    rows, cols = A.shape
    piv: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        if A[r, c] == 2:
            A[r] = (2 * A[r]) % 3
        col = A[:, c].copy()
        col[r] = 0
        A = (A - np.outer(col, A[r])) % 3
        piv.append(c)
        r += 1
    return A[:r], piv


def rank3(A) -> int:
    A = np.atleast_2d(A)
    if A.size == 0:
        return 0
    return len(rref3(A)[1])


def nullspace3(A):
    """Rows spanning {x : A x = 0 mod 3}."""
    A = np.atleast_2d(np.array(A, dtype=np.int64) % 3)
    R, piv = rref3(A)
    n = A.shape[1]
    free = [c for c in range(n) if c not in piv]
    B = np.zeros((len(free), n), dtype=np.int64)
    for k, f in enumerate(free):
        B[k, f] = 1
        for i, pc in enumerate(piv):
            B[k, pc] = (-R[i, f]) % 3
    return B


def omega_rows(G, n):
    """Rows (-g_z | g_x): (omega_rows(G) @ e) % 3 are the symplectic products <g, e>."""
    G = np.atleast_2d(np.array(G, dtype=np.int64))
    return np.hstack([(-G[:, n:]) % 3, G[:, :n] % 3])


def sp(u, v, n) -> int:
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    return int((u[:n] @ v[n:] - u[n:] @ v[:n]) % 3)


def pauli(n, X=None, Z=None):
    v = np.zeros(2 * n, dtype=np.int64)
    for i, a in (X or {}).items():
        v[i] = (v[i] + a) % 3
    for i, b in (Z or {}).items():
        v[n + i] = (v[n + i] + b) % 3
    return v


def pstr(v, n) -> str:
    out = []
    for i in range(n):
        a, b = int(v[i]) % 3, int(v[n + i]) % 3
        if a == 0 and b == 0:
            continue
        t = ""
        if a:
            t += f"X{i}" + ("" if a == 1 else "^-1")
        if b:
            t += f"Z{i}" + ("" if b == 1 else "^-1")
        out.append(t)
    return " ".join(out) if out else "I"


LOCAL8 = np.array([(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)],
                  dtype=np.int64)
LOCAL9 = np.array([(a, b) for a in range(3) for b in range(3)], dtype=np.int64)


def errors_on_support(n, supp, local=LOCAL8):
    w = len(supp)
    idx = np.array(list(itertools.product(range(len(local)), repeat=w)), dtype=np.int64)
    E = np.zeros((idx.shape[0], 2 * n), dtype=np.int64)
    for j, q in enumerate(supp):
        E[:, q] = local[idx[:, j], 0]
        E[:, n + q] = local[idx[:, j], 1]
    return E


def weight_w_errors(n, w):
    return np.vstack([errors_on_support(n, s) for s in itertools.combinations(range(n), w)])


# ---------------------------------------------------------------------------
# Codes
# ---------------------------------------------------------------------------

class Code:
    def __init__(self, name, n, gens, XL, ZL, meta=None):
        self.name = name
        self.n = n
        self.G = np.array(gens, dtype=np.int64) % 3
        self.r = self.G.shape[0]
        self.XL = np.array(XL, dtype=np.int64) % 3
        self.ZL = np.array(ZL, dtype=np.int64) % 3
        self.meta = meta or {}
        self.A = omega_rows(np.vstack([self.G, self.XL, self.ZL]), n)  # (r+2, 2n)
        self.c = sp(self.XL, self.ZL, n)
        self.Nstab = nullspace3(self.G)          # v in span(G)  <=>  Nstab @ v = 0
        self.normalizer = nullspace3(omega_rows(self.G, n))  # {e : <g,e> = 0 for all g}

    def in_stab(self, V):
        V = np.atleast_2d(np.asarray(V, dtype=np.int64))
        if self.Nstab.size == 0:
            return np.ones(V.shape[0], bool)
        return np.all((V @ self.Nstab.T) % 3 == 0, axis=1)

    def syn_cls(self, E):
        """(N, 2n) errors -> (N, r+2): r syndrome trits then 2 class trits
        (<X_L, e>, <Z_L, e>)."""
        return (np.atleast_2d(E) @ self.A.T) % 3

    def supports(self):
        n = self.n
        return [tuple(int(i) for i in np.nonzero((g[:n] != 0) | (g[n:] != 0))[0]) for g in self.G]


def code_512():
    n = 5
    gens = [pauli(n, Z={0: 1, 1: 2, 3: 2}),      # S_Z1 = Z0 Z1^-1 Z3^-1  (ancilla 5)
            pauli(n, Z={1: 1, 2: 1, 4: 2}),      # S_Z2 = Z1 Z2 Z4^-1     (ancilla 6)
            pauli(n, X={0: 1, 1: 1, 2: 2}),      # S_X1 = X0 X1 X2^-1     (ancilla 7)
            pauli(n, X={1: 2, 3: 1, 4: 2})]      # S_X2 = X1^-1 X3 X4^-1  (ancilla 8)
    return Code("[[5,1,2]]_3", n, gens, pauli(n, X={0: 1, 3: 1}), pauli(n, Z={0: 1, 2: 1}),
                meta={"fold_pairs": [(2, 3)]})


def cyclic_gens(n, pattern):
    """pattern: list of (x, z) exponents for sites 0..len-1, shifted cyclically."""
    gens = []
    for j in range(n):
        v = np.zeros(2 * n, dtype=np.int64)
        for k, (a, b) in enumerate(pattern):
            v[(j + k) % n] = a % 3
            v[n + (j + k) % n] = b % 3
        gens.append(v)
    return np.array(gens)


def code_513():
    n = 5
    allg = cyclic_gens(n, [(1, 0), (0, 1), (0, 2), (2, 0), (0, 0)])   # X Z Z^-1 X^-1 I
    return Code("[[5,1,3]]_3", n, allg[:4], pauli(n, X={i: 1 for i in range(n)}),
                pauli(n, Z={i: 1 for i in range(n)}), meta={"all_shifts": allg})


def code_713():
    n = 7
    faces = [(0, 2, 4, 6), (1, 2, 5, 6), (3, 4, 5, 6)]
    unstar = (0, 1, 3, 6)
    eps = {i: (1 if i in unstar else 2) for i in range(n)}
    gens = [pauli(n, X={i: 1 for i in f}) for f in faces] + \
           [pauli(n, Z={i: eps[i] for i in f}) for f in faces]
    return Code("[[7,1,3]]_3", n, gens, pauli(n, X={i: 1 for i in range(n)}),
                pauli(n, Z=eps), meta={"faces": faces, "unstarred": unstar,
                                       "starred": tuple(i for i in range(n) if i not in unstar)})


def code_913():
    n = 9
    q = lambda r, c: 3 * r + c  # noqa: E731
    zexp = lambda i: 1 if ((i // 3 + i % 3) % 2 == 0) else 2  # noqa: E731
    xchk = [(0, 1, 3, 4), (4, 5, 7, 8), (1, 2), (6, 7)]
    zchk = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 3), (5, 8)]
    gens = [pauli(n, X={i: 1 for i in s}) for s in xchk] + \
           [pauli(n, Z={i: zexp(i) for i in s}) for s in zchk]
    XL = pauli(n, X={q(0, 0): 1, q(1, 0): 1, q(2, 0): 1})
    ZL = pauli(n, Z={q(0, 0): 1, q(0, 1): 2, q(0, 2): 1})
    coords = {q(r, c): (r, c) for r in range(3) for c in range(3)}
    return Code("[[9,1,3]]_3", n, gens, XL, ZL, meta={"coords": coords})


def folded_square(D):
    """Unrotated distance-D surface code on a (2D-1)x(2D-1) grid, data at (i,j)
    with i+j even, Z checks at (odd, even), X checks at (even, odd).  Exponent
    rule (it reproduces the repository [[5,1,2]]_3 exactly at D = 2):
      Z check: +1 on the up and left neighbours, -1 on down and right;
      X check: +1 on the left and down neighbours, -1 on right and up."""
    L = 2 * D - 1
    sites = [(i, j) for i in range(L) for j in range(L) if (i + j) % 2 == 0]
    if D == 2:  # repository labels 0=(0,0) 1=(1,1) 2=(0,2) 3=(2,0) 4=(2,2)
        sites = [(0, 0), (1, 1), (0, 2), (2, 0), (2, 2)]
    idx = {s: k for k, s in enumerate(sites)}
    n = len(sites)
    gens = []
    zpos = [(i, j) for i in range(L) for j in range(L) if i % 2 == 1 and j % 2 == 0]
    xpos = [(i, j) for i in range(L) for j in range(L) if i % 2 == 0 and j % 2 == 1]
    for (i, j) in zpos:
        z = {}
        for (di, dj, e) in [(-1, 0, 1), (0, -1, 1), (1, 0, 2), (0, 1, 2)]:
            s = (i + di, j + dj)
            if s in idx:
                z[idx[s]] = e
        gens.append(pauli(n, Z=z))
    for (i, j) in xpos:
        x = {}
        for (di, dj, e) in [(0, -1, 1), (1, 0, 1), (0, 1, 2), (-1, 0, 2)]:
            s = (i + di, j + dj)
            if s in idx:
                x[idx[s]] = e
        gens.append(pauli(n, X=x))
    XL = pauli(n, X={idx[(i, 0)]: 1 for i in range(0, L, 2)})
    ZL = pauli(n, Z={idx[(0, j)]: 1 for j in range(0, L, 2)})
    pairs = sorted({tuple(sorted((idx[(i, j)], idx[(j, i)]))) for (i, j) in sites if i != j})
    fixed = [idx[(i, i)] for i in range(L) if (i, i) in idx]
    meta = {"sites": sites, "pairs": pairs, "fixed": fixed, "fold_pairs": pairs,
            "even": [idx[s] for s in sites if s[0] % 2 == 0],
            "odd": [idx[s] for s in sites if s[0] % 2 == 1]}
    name = f"[[{n},1,{D}]]_3 folded" if D != 2 else "[[5,1,2]]_3 (folded rule)"
    return Code(name, n, gens, XL, ZL, meta=meta)


def code_1313():
    return folded_square(3)


def find_logicals(G, n):
    """Return (XL, ZL) for a stabilizer group with k = 1, by search in the normalizer."""
    C = Code("tmp", n, G, np.zeros(2 * n, dtype=np.int64), np.zeros(2 * n, dtype=np.int64))
    Nb = C.normalizer
    cand = [v for v in Nb if not C.in_stab(v)[0]]
    for a in cand:
        for b in cand:
            if sp(a, b, n) != 0:
                return a % 3, b % 3
    raise RuntimeError("no logical pair found")


# ---------------------------------------------------------------------------
# S1: verification of constructions
# ---------------------------------------------------------------------------

def verify_code(code, maxw=3):
    n, r = code.n, code.r
    allops = np.vstack([code.G, code.XL, code.ZL])
    comm = (omega_rows(allops, n) @ allops.T) % 3
    res = {
        "n": n, "r": r,
        "stabilizers_commute_with_all": bool(np.all(comm[:r, :] == 0)),
        "rank": rank3(code.G),
        "XL_ZL_symplectic_product": code.c,
        "logicals_independent_of_stabilizers": rank3(allops) == r + 2,
    }
    res["k"] = n - res["rank"]
    res["check_weights"] = [len(s) for s in code.supports()]
    dist = None
    counts = {}
    for w in range(1, maxw + 1):
        E = weight_w_errors(n, w)
        SL = code.syn_cls(E)
        nontriv = np.all(SL[:, :r] == 0, axis=1) & np.any(SL[:, r:] != 0, axis=1)
        counts[w] = int(nontriv.sum())
        stab_w = np.all(SL[:, :r] == 0, axis=1) & np.all(SL[:, r:] == 0, axis=1)
        counts[f"stab{w}"] = int(stab_w.sum())
        if dist is None and counts[w] > 0:
            dist = w
    res["distance"] = dist if dist is not None else f">{maxw}"
    res["nontrivial_logicals_by_weight"] = {w: counts[w] for w in range(1, maxw + 1)}
    res["stabilizer_elements_by_weight"] = {w: counts[f"stab{w}"] for w in range(1, maxw + 1)}
    # single-error syndromes: are all weight-1 errors distinguishable up to logical class?
    E1 = weight_w_errors(n, 1)
    SL1 = code.syn_cls(E1)
    keys = {}
    clash = 0
    for row in SL1:
        s, l = row[:r].tobytes(), row[r:].tobytes()
        if s in keys and keys[s] != l:
            clash += 1
        keys.setdefault(s, l)
    res["weight1_syndromes_used"] = len(keys)
    res["weight1_colliding_errors"] = clash
    res["total_syndromes"] = 3 ** r
    return res


def section_S1(codes):
    log("\n=== S1: code constructions (commutation, rank, k, distance by brute force) ===")
    out = {}
    for key, code in codes.items():
        maxw = 3 if code.n <= 13 else 3
        res = verify_code(code, maxw=maxw)
        out[key] = res
        log(f"{code.name}: n={res['n']} r={res['r']} rank={res['rank']} k={res['k']} "
            f"d={res['distance']} commute={res['stabilizers_commute_with_all']} "
            f"<XL,ZL>={res['XL_ZL_symplectic_product']} weights={res['check_weights']}")
        log(f"   nontrivial logicals by weight {res['nontrivial_logicals_by_weight']}; "
            f"stabilizer elements by weight {res['stabilizer_elements_by_weight']}; "
            f"weight-1 errors use {res['weight1_syndromes_used']} of {res['total_syndromes']} "
            f"syndromes, {res['weight1_colliding_errors']} collide with another class")
    # [[5,1,3]]_3 generator derivation: qubit pattern XZZXI versus X Z Z^-1 X^-1 I
    n = 5
    naive = cyclic_gens(n, [(1, 0), (0, 1), (0, 1), (1, 0), (0, 0)])
    comm_naive = (omega_rows(naive, n) @ naive.T) % 3
    qud = code_513().meta["all_shifts"]
    comm_q = (omega_rows(qud, n) @ qud.T) % 3
    d513 = {
        "XZZXI_5shifts_commute": bool(np.all(comm_naive == 0)),
        "XZZXI_5shifts_rank": rank3(naive),
        "XZZXI_sum_of_5_shifts": pstr(naive.sum(0) % 3, n),
        "XZ^-1..._5shifts_commute": bool(np.all(comm_q == 0)),
        "X_Z_Zinv_Xinv_I_5shifts_rank": rank3(qud),
        "X_Z_Zinv_Xinv_I_sum_of_5_shifts_is_identity": bool(np.all(qud.sum(0) % 3 == 0)),
    }
    # code generated by the first four XZZXI shifts (not cyclic)
    XLn, ZLn = find_logicals(naive[:4], n)
    cn = Code("XZZXI first 4 shifts", n, naive[:4], XLn, ZLn)
    vn = verify_code(cn, 3)
    d513["XZZXI_first4_code_k"] = vn["k"]
    d513["XZZXI_first4_code_distance"] = vn["distance"]
    log("[[5,1,3]]_3 derivation: qubit pattern XZZXI -> 5 cyclic shifts commute mod 3: "
        f"{d513['XZZXI_5shifts_commute']}, rank {d513['XZZXI_5shifts_rank']} (k = 0: a stabilizer "
        f"state, the sum of the shifts is {d513['XZZXI_sum_of_5_shifts']}); "
        f"X Z Z^-1 X^-1 I -> shifts commute {d513['XZ^-1..._5shifts_commute']}, rank "
        f"{d513['X_Z_Zinv_Xinv_I_5shifts_rank']}, sum of shifts = I: "
        f"{d513['X_Z_Zinv_Xinv_I_sum_of_5_shifts_is_identity']}")
    log(f"   (the code from only the first four XZZXI shifts has k={vn['k']}, d={vn['distance']}, "
        "and is not cyclic)")
    # bounds
    hb = {}
    for nn in range(4, 11):
        hb[nn] = {"3^(n-1)": 3 ** (nn - 1), "1+8n": 1 + 8 * nn}
    log("Quantum Hamming bound for [[n,1,3]]_3 (nondegenerate count): 3^(n-1) >= 1 + 8n; "
        "n = 5: 81 >= 41, so [[5,1,3]]_3 is not perfect; it meets the quantum Singleton "
        "bound n - k = 2(d - 1) = 4 (MDS)")
    out["513_derivation"] = d513
    out["hamming"] = hb
    # folded rule at D = 2 equals the repository code
    f2 = folded_square(2)
    rep = code_512()
    same = (rank3(np.vstack([f2.G, rep.G])) == rank3(rep.G) == rank3(f2.G)
            and np.array_equal(f2.XL, rep.XL) and np.array_equal(f2.ZL, rep.ZL))
    same_gens = bool(np.array_equal(np.sort(f2.G, axis=0), np.sort(rep.G, axis=0)))
    out["folded_rule_D2_equals_repository"] = {"same_stabilizer_group_and_logicals": bool(same),
                                               "identical_generator_list": same_gens}
    log(f"Folded-square rule at D = 2 reproduces the repository code (same stabilizer group and "
        f"logical operators): {same}; identical generators incl. exponents: {same_gens}")
    RESULTS["S1"] = out
    return out


# ---------------------------------------------------------------------------
# S2: logical gates
# ---------------------------------------------------------------------------

G1 = {
    "I": np.eye(2, dtype=np.int64),
    "H": np.array([[0, 2], [1, 0]], dtype=np.int64),       # sdim H:  x' = -z, z' = x
    "H_INV": np.array([[0, 1], [2, 0]], dtype=np.int64),   # x' = z, z' = -x
    "P": np.array([[1, 0], [1, 1]], dtype=np.int64),       # z' = z + x
    "P_INV": np.array([[1, 0], [2, 1]], dtype=np.int64),
}
SL23 = [np.array([[a, b], [c, d]], dtype=np.int64)
        for a in range(3) for b in range(3) for c in range(3) for d in range(3)
        if (a * d - b * c) % 3 == 1]
assert len(SL23) == 24


def apply_gate(V, n, name, a, b=None):
    V = np.array(V, dtype=np.int64, copy=True)
    if b is None:
        M = G1[name] if isinstance(name, str) else name
        x = V[:, a].copy()
        z = V[:, n + a].copy()
        V[:, a] = (M[0, 0] * x + M[0, 1] * z) % 3
        V[:, n + a] = (M[1, 0] * x + M[1, 1] * z) % 3
    elif name == "CNOT":        # sdim: x_t += x_c ; z_c -= z_t
        V[:, b] = (V[:, b] + V[:, a]) % 3
        V[:, n + a] = (V[:, n + a] - V[:, n + b]) % 3
    elif name == "CNOT_INV":
        V[:, b] = (V[:, b] - V[:, a]) % 3
        V[:, n + a] = (V[:, n + a] + V[:, n + b]) % 3
    elif name == "CZ":          # z_b += x_a ; z_a += x_b
        V[:, n + b] = (V[:, n + b] + V[:, a]) % 3
        V[:, n + a] = (V[:, n + a] + V[:, b]) % 3
    elif name == "CZ_INV":
        V[:, n + b] = (V[:, n + b] - V[:, a]) % 3
        V[:, n + a] = (V[:, n + a] - V[:, b]) % 3
    elif name == "SWAP":
        V[:, [a, b]] = V[:, [b, a]]
        V[:, [n + a, n + b]] = V[:, [n + b, n + a]]
    else:
        raise ValueError(name)
    return V


def logical_action(code, circuit):
    """circuit: list of (name, a) or (name, a, b), applied in order (Heisenberg
    action E -> U E U^dag, sdim frame rules).  Returns the 2x2 logical
    symplectic matrix (columns = images of X_L, Z_L) or None if the stabilizer
    group is not preserved."""
    n = code.n
    V = np.vstack([code.G, code.XL, code.ZL])
    for g in circuit:
        V = apply_gate(V, n, *g)
    return logical_matrix_from_images(code, V[:code.r], V[code.r], V[code.r + 1])


def logical_matrix_from_images(code, imgG, imgX, imgZ):
    n = code.n
    if not code.in_stab(imgG).all():
        return None
    cinv = 1 if code.c == 1 else 2
    cols = []
    for v in (imgX, imgZ):
        a = (sp(v, code.ZL, n) * cinv) % 3
        b = (-sp(v, code.XL, n) * cinv) % 3
        resid = (v - a * code.XL - b * code.ZL) % 3
        if not code.in_stab(resid)[0]:
            return None
        cols.append((a, b))
    return np.array([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]], dtype=np.int64) % 3


def mat_key(M):
    return tuple(int(x) for x in np.asarray(M).ravel())


def gen_group(mats):
    I2 = np.eye(2, dtype=np.int64)
    seen = {mat_key(I2): I2}
    frontier = [I2]
    while frontier:
        new = []
        for A in frontier:
            for M in mats:
                B = (M @ A) % 3
                k = mat_key(B)
                if k not in seen:
                    seen[k] = B
                    new.append(B)
        frontier = new
    return list(seen.values())


def mat_order(M):
    A = np.eye(2, dtype=np.int64)
    for k in range(1, 25):
        A = (M @ A) % 3
        if np.array_equal(A, np.eye(2, dtype=np.int64)):
            return k
    return None


def name_of(M):
    if M is None:
        return "not logical"
    for k, v in G1.items():
        if np.array_equal(M % 3, v):
            return k
    if np.array_equal(M % 3, 2 * np.eye(2, dtype=np.int64)):
        return "C (=H^2)"
    return f"{mat_key(M)} order {mat_order(M)}"


def repo_H(code):
    return [("SWAP", 2, 3)] + [("H", q) for q in (0, 2, 3, 4)] + [("H_INV", 1)]


def repo_P(code):
    return [("P", 0), ("P", 4), ("P_INV", 1), ("CZ", 2, 3)]


def fold_H(code):
    m = code.meta
    circ = [("SWAP", a, b) for (a, b) in m["pairs"]]
    circ += [("H", q) for q in m["even"]] + [("H_INV", q) for q in m["odd"]]
    return circ


def fold_P(code):
    m = code.meta
    even = set(m["even"])
    circ = [("P" if q in even else "P_INV", q) for q in m["fixed"]]
    circ += [("CZ" if a in even else "CZ_INV", a, b) for (a, b) in m["pairs"]]
    return circ


def mitm_automorphisms(code, perms, split=None):
    """All (pi, M_0..M_{n-1}) with pi in perms and M_i in SL(2,3) such that the
    map (site i, local vector v) -> (site pi(i), M_i v) preserves the stabilizer
    group S.  Test: v in S  <=>  <v, u> = 0 for every u in the normalizer N(S).
    The condition is sum_i T_i[M_i] = 0 (mod 3) for per-site constraint vectors
    T_i[M].  All T vectors lie in a subspace of small dimension rho; projecting
    onto the rho pivot coordinates of that subspace is injective on it, so the
    base-3 integer of the projection is an exact key.  Meet in the middle over
    the sites.  Returns a list of (pi, (m_0..m_{n-1}) into SL23, logical matrix)."""
    n, r = code.n, code.r
    U = code.normalizer
    dimN = U.shape[0]
    K = r * dimN
    split = split if split is not None else n // 2
    left, right = list(range(split)), list(range(split, n))
    found = []
    for pi in perms:
        T = np.zeros((n, 24, K), dtype=np.int64)
        for i in range(n):
            gi = np.stack([code.G[:, i], code.G[:, n + i]], axis=1)          # (r, 2)
            ui = np.stack([U[:, pi[i]], U[:, n + pi[i]]], axis=1)            # (dimN, 2)
            for mi, Mm in enumerate(SL23):
                img = (gi @ Mm.T) % 3
                val = (np.outer(img[:, 0], ui[:, 1]) - np.outer(img[:, 1], ui[:, 0])) % 3
                T[i, mi] = val.ravel()
        _, piv = rref3(T.reshape(-1, K))
        rho = len(piv)
        Tp = T[:, :, piv].astype(np.int8)                                     # (n, 24, rho)
        nch = (rho + 32) // 33                                                # exact chunks of <= 33 trits
        PW = np.zeros((rho, nch))
        for j in range(rho):
            PW[j, j // 33] = 3.0 ** (j % 33)
        mults = np.array([1, 0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9][:nch],
                         dtype=np.uint64)

        def chunk_keys(S):
            return (S.astype(np.float64) @ PW).astype(np.int64)               # exact, (N, nch)

        def keys(S, ck=None):
            ck = chunk_keys(S) if ck is None else ck
            h = np.zeros(ck.shape[0], dtype=np.uint64)
            for c in range(nch):
                h = h + ck[:, c].astype(np.uint64) * mults[c]
            return h.view(np.int64)

        def all_sums(sites):
            S = np.zeros((1, rho), dtype=np.int8)
            for i in sites:
                S = ((S[:, None, :] + Tp[i][None, :, :]) % 3).reshape(-1, rho).astype(np.int8)
            return S

        SLft = all_sums(left)
        ckl = chunk_keys(SLft)
        kl = keys(SLft, ckl)
        order = np.argsort(kl, kind="stable")
        kl_sorted = kl[order]
        Rrest = all_sums(right[1:])
        for m0 in range(24):
            need = ((-(Rrest + Tp[right[0], m0][None, :])) % 3).astype(np.int8)
            ckr = chunk_keys(need)
            kr = keys(need, ckr)
            lo = np.searchsorted(kl_sorted, kr, side="left")
            hi = np.searchsorted(kl_sorted, kr, side="right")
            for h in np.nonzero(hi > lo)[0]:
                for li in order[lo[h]:hi[h]]:
                    if not np.array_equal(ckl[li], ckr[h]):                # hash collision
                        continue
                    lidx = np.unravel_index(li, (24,) * len(left))
                    ridx = (m0,) + tuple(np.unravel_index(h, (24,) * (len(right) - 1)))
                    found.append((tuple(pi), tuple(int(x) for x in tuple(lidx) + ridx)))
    outl = []
    for pi, ms in found:
        V = np.vstack([code.G, code.XL, code.ZL])
        Wv = np.zeros_like(V)
        for i in range(n):
            loc = (np.stack([V[:, i], V[:, n + i]], axis=1) @ SL23[ms[i]].T) % 3
            Wv[:, pi[i]] = loc[:, 0]
            Wv[:, n + pi[i]] = loc[:, 1]
        Lm = logical_matrix_from_images(code, Wv[:code.r], Wv[code.r], Wv[code.r + 1])
        assert Lm is not None, "MITM returned a map that does not preserve the code"
        outl.append((pi, ms, Lm))
    return outl


def involutions(n):
    """All involutions of range(n) as lists of 2-cycles."""
    def rec(rem):
        if not rem:
            yield []
            return
        a = rem[0]
        rest = rem[1:]
        for p in rec(rest):          # a fixed
            yield p
        for k, b in enumerate(rest):  # a paired with b
            for p in rec(rest[:k] + rest[k + 1:]):
                yield [(a, b)] + p
    return list(rec(list(range(n))))


def diagonal_P_system(code, pairs):
    """Linear constraints on theta = (Q_ii, Q_ab for (a,b) in pairs, c) such
    that the diagonal Clifford z += Q x (P^Q_ii on qutrit i, CZ^Q_ab on each
    pair) preserves the code and acts as logical P^c: (0, Q g_x) in S for every
    generator g, (0, Q z_x) in S for Z_L, and (0, Q x_x) - c Z_L in S for X_L."""
    n = code.n
    U = code.normalizer
    Ux = U[:, :n]
    var = [("d", i) for i in range(n)] + [("p", a, b) for (a, b) in pairs]
    nv = len(var) + 1
    cons = []
    targets = [(g[:n], False) for g in code.G] + [(code.ZL[:n], False), (code.XL[:n], True)]
    for vx, isX in targets:
        Qv = np.zeros((n, nv), dtype=np.int64)
        for k, vr in enumerate(var):
            if vr[0] == "d":
                Qv[vr[1], k] += vx[vr[1]]
            else:
                a, b = vr[1], vr[2]
                Qv[a, k] += vx[b]
                Qv[b, k] += vx[a]
        for j in range(U.shape[0]):
            row = (-(Ux[j] @ Qv)) % 3          # <(0,w), u> = -w . u_x
            if isX:
                row[-1] = (-sp(code.ZL, U[j], n)) % 3
            cons.append(row)
    return var, np.array(cons, dtype=np.int64) % 3


def diagonal_P_solution(code, pairs):
    """Return an explicit circuit for a diagonal logical P^c (c != 0) supported on
    the given pairs, verified with logical_action, or None."""
    var, A = diagonal_P_system(code, pairs)
    if rank3(A) != rank3(A[:, :-1]):
        return None
    N = nullspace3(A)
    for coeffs in itertools.product(range(3), repeat=N.shape[0]):
        v = (np.array(coeffs, dtype=np.int64) @ N) % 3
        if v[-1] == 0:
            continue
        circ = []
        for k, vr in enumerate(var):
            t = int(v[k])
            if t == 0:
                continue
            if vr[0] == "d":
                circ.append(("P" if t == 1 else "P_INV", vr[1]))
            else:
                circ.append(("CZ" if t == 1 else "CZ_INV", vr[1], vr[2]))
        Lm = logical_action(code, circ)
        if Lm is not None and np.array_equal(Lm[:, 0], np.array([1, v[-1]])) and \
                np.array_equal(Lm[:, 1], np.array([0, 1])):
            return circ
        raise RuntimeError("diagonal solution failed verification")
    return None


def diagonal_P_search(code, pairings):
    hits = []
    for pairs in pairings:
        var, A = diagonal_P_system(code, pairs)
        if rank3(A) == rank3(A[:, :-1]):
            hits.append(pairs)
    return hits


def section_S2(codes, args):
    log("\n=== S2: logical gates (sdim frame conventions) ===")
    out = {}
    F, P = G1["H"], G1["P"]
    log(f"single-qutrit references: F (sdim H) = {mat_key(F)}, P = {mat_key(P)}, "
        f"<F,P> has order {len(gen_group([F, P]))} (SL(2,3))")
    # repository code
    c512 = codes["512"]
    mH, mP = logical_action(c512, repo_H(c512)), logical_action(c512, repo_P(c512))
    grp = gen_group([mH, mP])
    out["512"] = {"H_L": name_of(mH), "P_L": name_of(mP), "group_order": len(grp)}
    log(f"[[5,1,2]]_3 repository H_L -> {name_of(mH)}, P_L -> {name_of(mP)}; generated logical "
        f"group order {len(grp)}")
    # folded D=3
    c13 = codes["1313"]
    mH, mP = logical_action(c13, fold_H(c13)), logical_action(c13, fold_P(c13))
    grp = gen_group([mH, mP])
    out["1313"] = {"H_L_fold": name_of(mH), "P_L_fold": name_of(mP), "group_order": len(grp),
                   "H_L_circuit": fold_H(c13), "P_L_circuit": fold_P(c13),
                   "pairs": c13.meta["pairs"], "fixed": c13.meta["fixed"]}
    n2q_H = sum(1 for g in fold_H(c13) if len(g) == 3)
    n2q_P = sum(1 for g in fold_P(c13) if len(g) == 3)
    log(f"[[13,1,3]]_3 folded: fold-transversal H_L (SWAP on {len(c13.meta['pairs'])} mirror pairs, "
        f"H on even rows, H_INV on odd rows) -> {name_of(mH)}; P_L (P/P_INV on the "
        f"{len(c13.meta['fixed'])} diagonal qutrits, CZ/CZ_INV on the pairs) -> {name_of(mP)}; "
        f"group order {len(grp)}; two-qutrit gates per H_L/P_L: {n2q_H}/{n2q_P}")
    # same rule check at D=2
    f2 = folded_square(2)
    out["folded_rule_D2_circuits_equal_repo"] = {
        "H": sorted(map(str, fold_H(f2))) == sorted(map(str, repo_H(f2))),
        "P": sorted(map(str, fold_P(f2))) == sorted(map(str, repo_P(f2)))}
    log(f"   the same rule at D = 2 gives exactly the repository circuits: "
        f"{out['folded_rule_D2_circuits_equal_repo']}")
    # colour code
    c7 = codes["713"]
    us, st = c7.meta["unstarred"], c7.meta["starred"]
    h7 = [("H", q) for q in us] + [("H_INV", q) for q in st]
    p7 = [("P", q) for q in us] + [("P_INV", q) for q in st]
    mH, mP = logical_action(c7, h7), logical_action(c7, p7)
    uniF = logical_action(c7, [("H", q) for q in range(7)])
    uniP = logical_action(c7, [("P", q) for q in range(7)])
    grp = gen_group([mH, mP]) if (mH is not None and mP is not None) else []
    out["713"] = {"H_star": name_of(mH), "P_star": name_of(mP), "group_order": len(grp),
                  "uniform_H": name_of(uniF), "uniform_P": name_of(uniP)}
    log(f"[[7,1,3]]_3 colour: H on unstarred {us} + H_INV on starred {st} -> {name_of(mH)}; "
        f"P/P_INV on the same split -> {name_of(mP)}; group order {len(grp)}; uniform H^(x7) -> "
        f"{name_of(uniF)}, uniform P^(x7) -> {name_of(uniP)}")
    # [[5,1,3]]_3 automorphisms (all permutations x local SL(2,3))
    c5 = codes["513"]
    t0 = time.time()
    perms = list(itertools.permutations(range(5)))
    auts = mitm_automorphisms(c5, perms, split=3)
    logs = {}
    for pi, ms, Lm in auts:
        logs.setdefault(mat_key(Lm), []).append(pi)
    lgroup = [np.array(k).reshape(2, 2) for k in logs]
    orders = sorted(mat_order(M) for M in lgroup)
    noperm = [a for a in auts if a[0] == tuple(range(5))]
    lg_noperm = {mat_key(a[2]) for a in noperm}
    out["513"] = {"automorphisms_with_permutations": len(auts),
                  "distinct_logical_actions": len(lgroup), "element_orders": orders,
                  "has_order3_element(P-type)": any(o in (3, 6) for o in orders),
                  "local_only_automorphisms": len(noperm),
                  "local_only_logical_actions": [name_of(np.array(k).reshape(2, 2)) for k in lg_noperm],
                  "permutations_used": sorted({str(a[0]) for a in auts}),
                  "contains_F": mat_key(F) in logs, "contains_P": mat_key(P) in logs}
    if mat_key(F) in logs:
        out["513"]["permutation_for_F"] = str(logs[mat_key(F)][0])
    log(f"[[5,1,3]]_3: {len(auts)} local-Clifford x permutation automorphisms "
        f"({len(out['513']['permutations_used'])} permutations); logical actions form a group of "
        f"order {len(lgroup)} with element orders {orders}; P-type (order 3) element: "
        f"{out['513']['has_order3_element(P-type)']}; logical F available: {out['513']['contains_F']}"
        f" (e.g. with permutation {out['513'].get('permutation_for_F')}); without permutations: "
        f"{len(noperm)} automorphisms, logical actions {out['513']['local_only_logical_actions']} "
        f"[{time.time() - t0:.1f}s]")
    # Q8 2-design check: Weyl x| group is a unitary 2-design iff the symplectic part acts
    # transitively on the 8 nonzero points of Z_3^2
    pts = [np.array(p) for p in [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]]
    orb = {tuple((M @ pts[0]) % 3) for M in lgroup}
    out["513"]["orbit_of_X_under_logical_group"] = len(orb)
    log(f"   orbit of logical X under that group has {len(orb)} of 8 nonzero points "
        f"(8 = transitive, so Weyl x| group is a unitary 2-design)")
    # colour code local automorphisms (identity permutation)
    if not args.quick:
        t0 = time.time()
        a7 = mitm_automorphisms(c7, [tuple(range(7))], split=3)
        lg7 = {mat_key(a[2]) for a in a7}
        out["713"]["local_automorphisms"] = len(a7)
        out["713"]["local_logical_group_order"] = len(lg7)
        log(f"[[7,1,3]]_3: {len(a7)} local-Clifford automorphisms (no permutation), "
            f"{len(lg7)} distinct logical actions [{time.time() - t0:.1f}s]")
    # rotated code: local Cliffords x the 8 lattice symmetries
    c9 = codes["913"]
    syms = {
        "id": lambda r, c: (r, c), "rot90": lambda r, c: (c, 2 - r),
        "rot180": lambda r, c: (2 - r, 2 - c), "rot270": lambda r, c: (2 - c, r),
        "transpose": lambda r, c: (c, r), "antitranspose": lambda r, c: (2 - c, 2 - r),
        "flip_lr": lambda r, c: (r, 2 - c), "flip_ud": lambda r, c: (2 - r, c)}
    permd = {k: tuple(3 * f(i // 3, i % 3)[0] + f(i // 3, i % 3)[1] for i in range(9))
             for k, f in syms.items()}
    out["913"] = {}
    if not args.quick:
        t0 = time.time()
        res9 = {}
        for k, pi in permd.items():
            a9 = mitm_automorphisms(c9, [pi], split=4)
            acts = sorted({name_of(a[2]) for a in a9})
            res9[k] = {"automorphisms": len(a9), "logical_actions": acts}
        out["913"]["local_x_lattice_symmetry"] = res9
        log("[[9,1,3]]_3: local-Clifford automorphisms composed with each lattice symmetry "
            f"[{time.time() - t0:.1f}s]:")
        for k, v in res9.items():
            log(f"   {k:13s}: {v['automorphisms']:3d} automorphisms, logical actions {v['logical_actions']}")
    # diagonal (phase-type) P over all pairings, with explicit verified circuits and the
    # fault test for the pairs (a single CZ fault = an arbitrary two-qutrit Pauli on its pair)
    t0 = time.time()
    for key in ("512", "513", "713", "913"):
        code = codes[key]
        n = code.n
        invs = involutions(n)
        hits = diagonal_P_search(code, invs)
        hs = [frozenset(h) for h in hits]
        minimal = [sorted(h) for h in hs if not any(g < h for g in hs)]
        singles = np.vstack([np.zeros((1, 2 * n), dtype=np.int64), weight_w_errors(n, 1)])
        rec = {"pairings_tested": len(invs), "pairings_with_diagonal_P": len(hits),
               "minimal_pairings": [str(m) for m in minimal]}
        ft = []
        for mp in minimal:
            circ = diagonal_P_solution(code, mp)
            Epair = (np.vstack([pair_set(n, a2, b2) for (a2, b2) in mp])
                     if mp else np.zeros((0, 2 * n), dtype=np.int64))
            if key == "512":
                safe = detection_safe(code, Epair) if mp else True
            else:
                safe = collision_free(code, np.vstack([singles, Epair])) if mp else True
            ft.append({"pairs": str(mp), "circuit": str(circ), "pair_faults_safe": bool(safe)})
        rec["minimal_solutions"] = ft
        full_pairs = [(a2, b2) for a2 in range(n) for b2 in range(a2 + 1, n)]
        rec["any_CZ_network_possible"] = bool(diagonal_P_search(code, [full_pairs]))
        if key == "513":
            best = None
            for kk in range(len(full_pairs) + 1):
                subs = [sub for sub in itertools.combinations(full_pairs, kk)
                        if diagonal_P_search(code, [list(sub)])]
                if subs:
                    best = (kk, subs)
                    break
            if best:
                circ = diagonal_P_solution(code, list(best[1][0]))
                deg = np.bincount(np.array(best[1][0]).ravel(), minlength=n)
                rec["min_CZ_network"] = {"number_of_CZ": best[0], "networks": len(best[1]),
                                         "example": str(best[1][0]), "circuit": str(circ),
                                         "max_CZ_per_qutrit": int(deg.max())}
        out.setdefault(key, {})["diagonal_P"] = rec
        nsafe = sum(1 for f in ft if f["pair_faults_safe"])
        log(f"{code.name}: diagonal logical P (P^t on qutrits + CZ^u on one pairing): "
            f"{len(hits)} of {len(invs)} pairings admit one; {len(minimal)} minimal pairings, "
            f"{nsafe} of them with {'detection-safe' if key == '512' else 'correctable'} pair faults")
        for f in ft[:4]:
            log(f"      pairs {f['pairs']}: {f['circuit']} -> pair faults safe: {f['pair_faults_safe']}")
        if key == "513" and "min_CZ_network" in rec:
            m = rec["min_CZ_network"]
            log(f"      no pairing works; the smallest CZ network has {m['number_of_CZ']} CZ gates "
                f"({m['networks']} network(s), max {m['max_CZ_per_qutrit']} CZ per qutrit): {m['circuit']}")
    log(f"   [diagonal searches {time.time() - t0:.1f}s]")
    RESULTS["S2"] = out
    return out


# ---------------------------------------------------------------------------
# S3: one-ancilla syndrome extraction: hooks, fold pairs, safe orders
# ---------------------------------------------------------------------------

def hook_set(code, k, order):
    """Data errors left by one fault in the one-ancilla circuit that measures
    generator k by coupling to the qutrits in `order`: an arbitrary Pauli on
    the qutrit just coupled times any power of the not-yet-coupled part of the
    check (the ancilla error propagated by the remaining two-qutrit gates)."""
    n = code.n
    g = code.G[k]
    errs = []
    for pos, q in enumerate(order):
        tail = np.zeros(2 * n, dtype=np.int64)
        for s in order[pos + 1:]:
            tail[s] = g[s]
            tail[n + s] = g[n + s]
        for (a, b) in LOCAL9:
            base = np.zeros(2 * n, dtype=np.int64)
            base[q] = a
            base[n + q] = b
            for e in range(3):
                errs.append((base + e * tail) % 3)
    return np.unique(np.array(errs), axis=0)


def pair_set(n, a, b):
    E = errors_on_support(n, (a, b), local=LOCAL9)
    return E[np.any(E != 0, axis=1)]


def collision_free(code, E):
    """True if no two errors in E share a syndrome but differ in logical class."""
    SL = code.syn_cls(E)
    r = code.r
    d = {}
    for row in SL:
        s = row[:r].tobytes()
        l = row[r:].tobytes()
        if d.setdefault(s, l) != l:
            return False
    return True


def detection_safe(code, E):
    SL = code.syn_cls(E)
    r = code.r
    bad = np.all(SL[:, :r] == 0, axis=1) & np.any(SL[:, r:] != 0, axis=1)
    return not bool(bad.any())


def section_S3(codes, args):
    log("\n=== S3: one-ancilla-per-check extraction: hook / fold-pair faults ===")
    out = {}
    try:
        import networkx as nx
    except Exception:  # pragma: no cover
        nx = None
    for key in ("512", "513", "713", "913", "1313"):
        code = codes[key]
        n = code.n
        singles = np.vstack([np.zeros((1, 2 * n), dtype=np.int64), weight_w_errors(n, 1)])
        supp = code.supports()
        res = {"check_weights": [len(s) for s in supp],
               "two_qutrit_gates_per_round": int(sum(len(s) for s in supp)),
               "ancillas": code.r}
        fold = code.meta.get("fold_pairs", [])
        fold_E = np.vstack([pair_set(n, a, b) for (a, b) in fold]) if fold else np.zeros((0, 2 * n), dtype=np.int64)
        if key == "512":
            # distance 2: test detection-safety of single faults
            E = singles
            items = []
            for k, s in enumerate(supp):
                for order in itertools.permutations(s):
                    items.append(hook_set(code, k, order))
            allE = np.vstack([singles, fold_E] + items)
            res["single_faults_all_orders_detection_safe"] = detection_safe(code, allE)
            res["fold_pair_paulis_detection_safe"] = detection_safe(code, fold_E)
            log(f"{code.name}: every single fault (any qutrit Pauli, any hook for any coupling "
                f"order, any two-qutrit Pauli on fold pair {fold}) is detected or harmless: "
                f"{res['single_faults_all_orders_detection_safe']}")
        else:
            safe_orders = {}
            items = {}
            for k, s in enumerate(supp):
                ok = []
                for order in itertools.permutations(s):
                    H = hook_set(code, k, order)
                    if collision_free(code, np.vstack([singles, H])):
                        ok.append(order)
                    items[(k, order)] = H
                safe_orders[k] = ok
            res["safe_orders_per_check"] = {k: f"{len(v)}/{math.factorial(len(supp[k]))}"
                                            for k, v in safe_orders.items()}
            if fold:
                res["fold_pair_faults_correctable"] = collision_free(code, np.vstack([singles, fold_E]))
            base = np.vstack([singles, fold_E]) if fold else singles
            # pairwise compatibility among safe (check, order) items (+ fold faults)
            keys = [(k, o) for k in safe_orders for o in safe_orders[k]]
            compat = {}
            for i1 in range(len(keys)):
                for i2 in range(i1 + 1, len(keys)):
                    k1, k2 = keys[i1], keys[i2]
                    if k1[0] == k2[0]:
                        continue
                    compat[(k1, k2)] = collision_free(code, np.vstack([base, items[k1], items[k2]]))
            if fold:
                for kk in keys:
                    compat[("fold", kk)] = collision_free(code, np.vstack([base, items[kk]]))
            all_ok = all(compat.values()) if compat else False
            res["all_pairs_of_safe_items_compatible"] = bool(all_ok) if keys else False
            # backtracking for one joint assignment
            checks = list(safe_orders)
            choice = {}

            def bt(i):
                if i == len(checks):
                    return True
                k = checks[i]
                for o in safe_orders[k]:
                    if fold and not compat[("fold", (k, o))]:
                        continue
                    if all(compat.get(((kk, choice[kk]), (k, o)), compat.get(((k, o), (kk, choice[kk])), True))
                           for kk in choice):
                        choice[k] = o
                        if bt(i + 1):
                            return True
                        del choice[k]
                return False
            found = bt(0) if all(len(v) > 0 for v in safe_orders.values()) else False
            res["joint_safe_assignment_exists"] = bool(found)
            if found:
                res["example_joint_orders"] = {k: list(map(int, v)) for k, v in choice.items()}
                Etot = np.vstack([base] + [items[(k, choice[k])] for k in choice])
                res["joint_assignment_verified_directly"] = collision_free(code, Etot)
            log(f"{code.name}: weights {res['check_weights']}, {res['two_qutrit_gates_per_round']} "
                f"two-qutrit gates/round; safe coupling orders per check {res['safe_orders_per_check']}; "
                f"joint safe assignment: {res['joint_safe_assignment_exists']}"
                + (f" (verified directly: {res.get('joint_assignment_verified_directly')})" if found else "")
                + (f"; every pair of safe orders compatible: {res['all_pairs_of_safe_items_compatible']}"
                   if keys else "")
                + (f"; fold-pair two-qutrit faults correctable: {res.get('fold_pair_faults_correctable')}"
                   if fold else ""))
        # Tanner graph planarity (data + one ancilla per check), with and without fold edges
        if nx is not None:
            Gt = nx.Graph()
            for k, s in enumerate(supp):
                for q in s:
                    Gt.add_edge(("a", k), ("d", q))
            res["tanner_planar"] = bool(nx.check_planarity(Gt)[0])
            if fold:
                Gf = Gt.copy()
                for (a, b) in fold:
                    Gf.add_edge(("d", a), ("d", b))
                res["tanner_plus_fold_edges_planar"] = bool(nx.check_planarity(Gf)[0])
            if key == "713":
                Gc = nx.Graph()
                for fi, f in enumerate(code.meta["faces"]):
                    for q in f:
                        Gc.add_edge(("f", fi), ("d", q))
                res["tanner_planar_one_ancilla_per_face"] = bool(nx.check_planarity(Gc)[0])
            degs = {}
            for k, s in enumerate(supp):
                for q in s:
                    degs[q] = degs.get(q, 0) + 1
            res["data_degree_max"] = max(degs.values())
            log(f"   Tanner graph planar: {res['tanner_planar']}"
                + (f"; with fold edges: {res.get('tanner_plus_fold_edges_planar')}" if fold else "")
                + (f"; with one ancilla per face: {res.get('tanner_planar_one_ancilla_per_face')}"
                   if key == "713" else "")
                + f"; max checks per data qutrit {res['data_degree_max']}")
        out[key] = res
    RESULTS["S3"] = out
    return out


# ---------------------------------------------------------------------------
# S4: code capacity (exact)
# ---------------------------------------------------------------------------

def kgrid_weight(code):
    """w(k) for every k in Z_3^m (m = r + 2): the number of qutrits on which the
    Weyl vector A^T k acts, as an int8 array of shape (3,)*m (axis j <-> k_j)."""
    m, n = code.r + 2, code.n
    shape = (3,) * m
    ar = np.arange(3, dtype=np.int16)
    w = np.zeros(shape, dtype=np.int8)
    for i in range(n):
        f = []
        for col in (i, n + i):
            acc = np.zeros(shape, dtype=np.int16)
            for j in range(m):
                c = int(code.A[j, col])
                if c:
                    sh = [1] * m
                    sh[j] = 3
                    acc = acc + ((ar * c) % 3).reshape(sh)
            f.append(acc % 3)
        w += ((f[0] != 0) | (f[1] != 0)).astype(np.int8)
    return w


def poly_coeffs(n):
    """C[w, j] = coefficient of t^j in (1 + 8t)^(n - w) (1 - t)^w."""
    C = np.zeros((n + 1, n + 1))
    for w in range(n + 1):
        p = np.array([1.0])
        for _ in range(n - w):
            p = np.convolve(p, [1.0, 8.0])
        for _ in range(w):
            p = np.convolve(p, [1.0, -1.0])
        C[w, :len(p)] = p
    return C


def series_from_counts(counts, n, order=3):
    """sum_j counts[j] (q/8)^j (1-q)^(n-j), expanded as Fractions for q^0..q^order."""
    out = [Fraction(0)] * (order + 1)
    for j, cnt in enumerate(counts):
        if j > order or cnt == 0:
            continue
        for i in range(0, order - j + 1):
            out[j + i] += Fraction(int(cnt), 8 ** j) * math.comb(n - j, i) * (-1) ** i
    return out


def series_div_accept(F, R, order=3):
    """F(q) / (1 - R(q)) with R(0) = 0."""
    inv = [Fraction(0)] * (order + 1)
    inv[0] = Fraction(1)
    # 1/(1-R) = sum_k R^k
    Rk = [Fraction(1)] + [Fraction(0)] * order
    for _ in range(order):
        Rk = [sum(Rk[a] * R[b - a] for a in range(b + 1)) for b in range(order + 1)]
        inv = [inv[b] + Rk[b] for b in range(order + 1)]
    return [sum(F[a] * inv[b - a] for a in range(b + 1)) for b in range(order + 1)]


def fmt_series(c, order=3):
    terms = []
    for k in range(order + 1):
        if c[k] != 0:
            terms.append(f"{frac_str(c[k])} q^{k}" if k > 1 else (f"{frac_str(c[k])} q" if k == 1 else frac_str(c[k])))
    return " + ".join(terms) if terms else "0"


class CodeCapacity:
    """Exact (syndrome, class) distribution of the code-capacity model."""

    def __init__(self, code, J=None):
        self.code = code
        self.n, self.r, self.m = code.n, code.r, code.r + 2
        self.N = 3 ** self.m
        self.w = kgrid_weight(code)
        self.J = code.n if J is None else J
        C = poly_coeffs(self.n)
        wi = self.w.astype(np.int64)
        self.Aj = []
        for j in range(self.J + 1):
            Aj = np.rint(np.fft.fftn(C[wi, j]).real / self.N).astype(np.int64)
            Aj = Aj.reshape(3 ** self.r, 9)
            assert Aj.sum() == math.comb(self.n, j) * 8 ** j, (code.name, j)
            assert Aj.min() >= 0
            self.Aj.append(Aj)
        self.Aj = np.array(self.Aj)                                   # (J+1, S, 9)
        # leading-order (small-q) MLD table: lexicographic argmax over (A_0, A_1, ...)
        cand = np.ones(self.Aj.shape[1:], dtype=bool)
        for j in range(self.J + 1):
            vals = np.where(cand, self.Aj[j], -1)
            cand &= vals == vals.max(axis=1, keepdims=True)
        self.lstar = np.argmax(cand, axis=1)
        self.tied = cand.sum(axis=1) > 1
        S = self.Aj.shape[1]
        self.Nfail = [int(self.Aj[j].sum() - self.Aj[j][np.arange(S), self.lstar].sum())
                      for j in range(self.J + 1)]
        # minimum weight per (s, l); strict hybrid
        nz = self.Aj > 0
        wmin = np.where(nz.any(axis=0), nz.argmax(axis=0), self.J + 1)   # (S, 9)
        wmin_s = wmin.min(axis=1)
        nmin = (wmin == wmin_s[:, None]).sum(axis=1)
        self.hyb_accept = (nmin == 1) & (wmin_s <= self.J)
        self.hyb_class = np.argmin(wmin, axis=1)

    def Pq(self, q):
        arr = (1.0 - 9.0 * q / 8.0) ** self.w.astype(np.float64)
        P = np.fft.fftn(arr).real / self.N
        return np.clip(P, 0.0, None).reshape(3 ** self.r, 9)

    def mld_eps(self, q, P=None):
        P = self.Pq(q) if P is None else P
        return float(1.0 - P.max(axis=1).sum())

    def table_eps(self, q, P=None):
        P = self.Pq(q) if P is None else P
        return float(1.0 - P[np.arange(P.shape[0]), self.lstar].sum())

    def detection(self, q, P=None):
        P = self.Pq(q) if P is None else P
        acc = float(P[0].sum())
        return acc, float(1.0 - P[0, 0] / acc)

    def hybrid(self, q, P=None):
        P = self.Pq(q) if P is None else P
        acc_rows = P[self.hyb_accept]
        acc = float(acc_rows.sum())
        ok = float(P[np.nonzero(self.hyb_accept)[0], self.hyb_class[self.hyb_accept]].sum())
        return acc, float(1.0 - ok / acc)

    def series(self, order=3):
        n = self.n
        S = self.Aj.shape[1]
        res = {"MLD_fail": series_from_counts(self.Nfail, n, order)}
        if self.n <= self.J:
            rej_counts = [int(self.Aj[j][1:].sum()) for j in range(self.J + 1)]
            fail_acc = [int(self.Aj[j][0, 1:].sum()) for j in range(self.J + 1)]
            R = series_from_counts(rej_counts, n, order)
            res["detect_rej"] = R
            res["detect_fail_given_acc"] = series_div_accept(series_from_counts(fail_acc, n, order), R, order)
            hr = [int(self.Aj[j][~self.hyb_accept].sum()) for j in range(self.J + 1)]
            acc_idx = np.nonzero(self.hyb_accept)[0]
            hf = [int(self.Aj[j][acc_idx].sum() - self.Aj[j][acc_idx, self.hyb_class[acc_idx]].sum())
                  for j in range(self.J + 1)]
            R2 = series_from_counts(hr, n, order)
            res["hybrid_rej"] = R2
            res["hybrid_fail_given_acc"] = series_div_accept(series_from_counts(hf, n, order), R2, order)
        return res


def bisect_root(f, lo, hi, tol=1e-5, it=60):
    """Root of f in [lo, hi] (sign change required), Brent's method."""
    from scipy.optimize import brentq
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-7, rtol=tol, maxiter=it))


def break_even(f, grid):
    """First q in the grid interval where f changes sign (f = eps - q); bisection."""
    vals = [f(q) for q in grid]
    for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if va < 0 <= vb or va > 0 >= vb:
            return bisect_root(f, a, b), vals
    return None, vals


def section_S4(codes, args):
    log("\n=== S4: code capacity (exact DFT over Z_3^(r+2)) ===")
    out = {}
    qgrid = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7.5e-2, 0.1, 0.15, 0.2, 0.3]
    if args.quick:
        qgrid = [1e-3, 1e-2, 5e-2, 0.1, 0.2]
    CC = {}
    for key in ("512", "513", "713", "913", "1313"):
        code = codes[key]
        t0 = time.time()
        J = None if code.n <= 9 else 3
        cc = CodeCapacity(code, J=J)
        CC[key] = cc
        ser = cc.series(order=3)
        rec = {"syndromes": 3 ** code.r, "Nfail_by_weight": cc.Nfail[:4],
               "tied_syndromes_leading_order": int(cc.tied.sum()),
               "series": {k: [frac_str(x) for x in v] for k, v in ser.items()}}
        curve = {}
        for q in (qgrid if code.n <= 9 or args.quick else [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3]):
            P = cc.Pq(q)
            row = {"MLD": cc.mld_eps(q, P), "table": cc.table_eps(q, P)}
            if key == "512":
                row["detect_acc"], row["detect_eps_acc"] = cc.detection(q, P)
                row["hybrid_acc"], row["hybrid_eps_acc"] = cc.hybrid(q, P)
            curve[q] = row
        rec["curve"] = curve
        bgrid = list(np.geomspace(1e-3, 0.45, 40)) if code.n <= 9 else [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.45]
        qs, vals = break_even(lambda q: cc.mld_eps(q) - q, bgrid)
        rec["break_even_MLD"] = qs
        rec["eps_minus_q_sign_on_grid"] = "all positive" if all(v > 0 for v in vals) else (
            "all negative" if all(v < 0 for v in vals) else "changes sign")
        if key == "512":
            qd, _ = break_even(lambda q: cc.detection(q)[1] - q, bgrid)
            qh, _ = break_even(lambda q: cc.hybrid(q)[1] - q, bgrid)
            rec["break_even_detect_acc"] = qd
            rec["break_even_hybrid_acc"] = qh
        out[key] = rec
        log(f"{code.name}: MLD eps = {fmt_series(ser['MLD_fail'])} + O(q^4) "
            f"[N_fail(w=1,2,3) = {cc.Nfail[1:4]}]; break-even eps = q at q* = "
            f"{('%.4f' % qs) if qs else 'none (' + rec['eps_minus_q_sign_on_grid'] + ' for q in [1e-3, 0.45])'}"
            f" [{time.time() - t0:.1f}s]")
        if key == "512":
            log(f"   detection: P_rej = {fmt_series(ser['detect_rej'])}, P(fail|acc) = "
                f"{fmt_series(ser['detect_fail_given_acc'])}; strict hybrid: P_rej = "
                f"{fmt_series(ser['hybrid_rej'])}, P(fail|acc) = {fmt_series(ser['hybrid_fail_given_acc'])}")
            log(f"   accepted-error break-even: detection {rec['break_even_detect_acc']}, hybrid "
                f"{rec['break_even_hybrid_acc']}")
        for q in ([1e-3, 1e-2, 5e-2] if not args.quick else [1e-3, 1e-2, 5e-2]):
            row = curve[q]
            extra = ""
            if key == "512":
                extra = (f"; detect acc {row['detect_acc']:.5f} eps|acc {row['detect_eps_acc']:.3e}; hybrid acc "
                         f"{row['hybrid_acc']:.5f} eps|acc {row['hybrid_eps_acc']:.3e}")
            log(f"   q={q:g}: MLD eps {row['MLD']:.4e} (eps/q {row['MLD'] / q:.3f}, eps/q^2 "
                f"{row['MLD'] / q ** 2:.2f}){extra}")
    RESULTS["S4"] = out
    return out, CC


# ---------------------------------------------------------------------------
# S5: phenomenological noise, exact
# ---------------------------------------------------------------------------

CHARS = [(a, b) for a in range(3) for b in range(3)]


def char_vec(a, b):
    l1, l2 = np.divmod(np.arange(9), 3)
    return OMEGA ** ((a * l1 + b * l2) % 3)


def meas_dist(r, qm):
    w = np.array([1 - qm, qm / 2, qm / 2])
    P = np.ones((1,))
    for _ in range(r):
        P = np.multiply.outer(P, w)
    return P.reshape(-1)


def neg_index(r):
    """index of -s for every syndrome index s (C order over (3,)*r)."""
    idx = np.arange(3 ** r)
    digs = np.array(np.unravel_index(idx, (3,) * r))
    return np.ravel_multi_index((-digs) % 3, (3,) * r)


def conv(a, v, r):
    sh = (3,) * r
    return np.fft.ifftn(np.fft.fftn(a.reshape(sh)) * np.fft.fftn(v.reshape(sh))).reshape(-1)


def xcorr(a, b, r):
    """g(u) = sum_x a(x) b(x + u)."""
    sh = (3,) * r
    N = 3 ** r
    A_ = np.fft.ifftn(a.reshape(sh)) * N
    return np.fft.ifftn(A_ * np.fft.fftn(b.reshape(sh))).reshape(-1)


def lead_eig(apply, dim, it=400, tol=1e-14, v0=None):
    v = (np.ones(dim, dtype=complex) / dim) if v0 is None else np.asarray(v0, dtype=complex)
    lam = 0
    for _ in range(it):
        u = apply(v)
        lam_new = np.vdot(v, u) / np.vdot(v, v)
        nrm = np.linalg.norm(u)
        if nrm == 0:
            return 0.0
        v = u / nrm
        if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
            lam = lam_new
            break
        lam = lam_new
    return complex(lam)


def mld_table(Pn, r, lam=None):
    """Per-syndrome correction class, symmetrised so that lambda(-s) = -lambda(s).
    With lam=None the per-q code-capacity MLD argmax of Pn is used; the
    memoryless decoders below pass the fixed leading-order table cc.lstar, which
    does not depend on q (the per-q table jumps between tied classes as q
    changes, which makes eps(q) of a memoryless decoder non-monotonic)."""
    lam = np.argmax(Pn, axis=1) if lam is None else np.array(lam, copy=True)
    neg = neg_index(r)
    l1, l2 = np.divmod(lam, 3)
    for s in range(3 ** r):
        t = neg[s]
        if t > s:
            lam[t] = 3 * ((-l1[s]) % 3) + ((-l2[s]) % 3)
    lam[0] = 0
    return lam


def phen_memoryless(cc, q, qm, decoder):
    """Exact per-round quantities of a memoryless decoder under phenomenological
    noise, from the leading eigenvalues of the character transfer matrices.
    decoder: 'detect' (reject if any raw outcome is non-zero), 'lookup'
    (apply the fixed leading-order MLD table to every observed syndrome), 'hybrid' (lookup, but
    reject syndromes whose minimum-weight explanations span >1 class).
    Returns (acceptance per round, accepted eps per round, eigenvalues)."""
    r = cc.r
    Pn = cc.Pq(q)
    Pm = meas_dist(r, qm)
    neg = neg_index(r)
    mus = {}
    if decoder == "lookup":
        lam = mld_table(Pn, r, cc.lstar)
        keep = np.ones(3 ** r, dtype=bool)
    elif decoder == "hybrid":
        lam = cc.hyb_class.copy()
        keep = cc.hyb_accept.copy()
    for (a, b) in CHARS:
        chi = char_vec(a, b)
        av = Pn @ chi                                   # a_chi(ds)
        if ((-a) % 3, (-b) % 3) in mus:                 # conjugate character
            mus[(a, b)] = np.conj(mus[((-a) % 3, (-b) % 3)])
            continue
        sh = (3,) * r
        if decoder == "detect":
            kern = np.fft.fftn(av.reshape(sh))
            weight = Pm[neg]
        else:
            bv = np.conj(chi[lam]) * keep               # b_chi(o)
            kern = np.fft.fftn(xcorr(av, bv, r).reshape(sh))
            weight = Pm

        def apply(v, kern=kern, weight=weight, sh=sh):
            return weight * np.fft.ifftn(kern * np.fft.fftn(v.reshape(sh))).reshape(-1)
        mus[(a, b)] = lead_eig(apply, 3 ** r, v0=weight)
    mu1 = mus[(0, 0)].real
    f = np.mean([mus[c].real for c in CHARS if c != (0, 0)]) / mu1
    return mu1, (8.0 / 9.0) * (1.0 - f), mus


def phen_memoryless_mc(cc, q, qm, decoder, T, shots, rng):
    """Direct simulation of the same memoryless decoders (validation)."""
    r = cc.r
    Pn = cc.Pq(q)
    flat = Pn.reshape(-1)
    flat = flat / flat.sum()
    if decoder == "lookup":
        lam = mld_table(Pn, r, cc.lstar)
        keep = np.ones(3 ** r, dtype=bool)
    elif decoder == "hybrid":
        lam, keep = cc.hyb_class, cc.hyb_accept
    pw = 3 ** np.arange(r - 1, -1, -1)
    s = np.zeros((shots, r), dtype=np.int64)       # accumulated syndrome (detect) / residual
    l = np.zeros((shots, 2), dtype=np.int64)
    alive = np.ones(shots, dtype=bool)
    for _ in range(T):
        u = rng.choice(flat.size, size=shots, p=flat)
        ds = np.array(np.unravel_index(u // 9, (3,) * r)).T
        dl = np.stack(np.divmod(u % 9, 3), axis=1)
        mm = rng.choice(3, size=(shots, r), p=[1 - qm, qm / 2, qm / 2])
        s = (s + ds) % 3
        l = (l + dl) % 3
        o = (s + mm) % 3
        if decoder == "detect":
            alive &= np.all(o == 0, axis=1)
        else:
            oi = o @ pw
            alive &= keep[oi]
            c = lam[oi]
            l = (l - np.stack(np.divmod(c, 3), axis=1)) % 3
            s = (s - o) % 3
    # final perfect round
    if decoder == "detect":
        alive &= np.all(s == 0, axis=1)
    else:
        oi = s @ pw
        alive &= keep[oi]
        c = lam[oi]
        l = (l - np.stack(np.divmod(c, 3), axis=1)) % 3
    vals = {}
    for (a, b) in CHARS:
        vals[(a, b)] = np.sum(alive * OMEGA ** ((a * l[:, 0] + b * l[:, 1]) % 3)) / shots
    return vals


def phen_memoryless_exact_T(cc, q, qm, decoder, T):
    """Exact E[chi(l_final) 1_accept] after T noisy rounds + final perfect round."""
    r = cc.r
    Pn = cc.Pq(q)
    Pm = meas_dist(r, qm)
    neg = neg_index(r)
    if decoder == "lookup":
        lam = mld_table(Pn, r, cc.lstar)
        keep = np.ones(3 ** r, dtype=bool)
    elif decoder == "hybrid":
        lam, keep = cc.hyb_class, cc.hyb_accept
    vals = {}
    for (a, b) in CHARS:
        chi = char_vec(a, b)
        av = Pn @ chi
        v = np.zeros(3 ** r, dtype=complex)
        v[0] = 1.0
        if decoder == "detect":
            for _ in range(T):
                v = Pm[neg] * conv(av, v, r)
            vals[(a, b)] = v[0]
        else:
            bv = np.conj(chi[lam]) * keep
            g = xcorr(av, bv, r)
            for _ in range(T):
                v = Pm * conv(g, v, r)
            vals[(a, b)] = np.sum(v * bv[neg])        # final correction of the residual -m
    return vals


def spacetime_faults(code, T, rho):
    """All fault sets of weight 1 and 2 of the phenomenological model with T
    noisy rounds and a final perfect round.  Signature = raw outcome shifts of
    rounds 0..T (r trits each).  Data fault weight 1/8 (x q), measurement fault
    weight rho/2 (x q, q_m = rho q).  Returns grouped masses
    M1[sig, class], M2[sig, class] and the index of the empty signature."""
    n, r = code.n, code.r
    E1 = weight_w_errors(n, 1)
    SL = code.syn_cls(E1)
    sites = np.array([int(np.nonzero((e[:n] != 0) | (e[n:] != 0))[0][0]) for e in E1])
    L = (T + 1) * r
    sig, cls, wt, loc = [], [], [], []
    for t in range(T):
        for k, e in enumerate(E1):
            v = np.zeros(L, dtype=np.int8)
            for tt in range(t, T + 1):
                v[tt * r:(tt + 1) * r] = SL[k, :r]
            sig.append(v)
            cls.append(SL[k, r] * 3 + SL[k, r + 1])
            wt.append(1.0 / 8.0)
            loc.append(("d", t, sites[k]))
        if rho > 0:
            for c in range(r):
                for a in (1, 2):
                    v = np.zeros(L, dtype=np.int8)
                    v[t * r + c] = a
                    sig.append(v)
                    cls.append(0)
                    wt.append(rho / 2.0)
                    loc.append(("m", t, c))
    sig = np.array(sig, dtype=np.int8)
    cls = np.array(cls, dtype=np.int64)
    wt = np.array(wt)
    locid = {x: i for i, x in enumerate(sorted(set(loc)))}
    loc = np.array([locid[x] for x in loc])
    F = len(wt)
    ii, jj = np.triu_indices(F, k=1)
    keep = loc[ii] != loc[jj]
    ii, jj = ii[keep], jj[keep]
    sig2 = (sig[ii] + sig[jj]) % 3
    l1a, l2a = np.divmod(cls[ii], 3)
    l1b, l2b = np.divmod(cls[jj], 3)
    cls2 = ((l1a + l1b) % 3) * 3 + (l2a + l2b) % 3
    wt2 = wt[ii] * wt[jj]
    allsig = np.vstack([np.zeros((1, L), dtype=np.int8), sig, sig2])
    uniq, inv = np.unique(allsig, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    zero = inv[0]
    M1 = np.zeros((len(uniq), 9))
    M2 = np.zeros((len(uniq), 9))
    np.add.at(M1, (inv[1:1 + F], cls), wt)
    np.add.at(M2, (inv[1 + F:], cls2), wt2)
    return M1, M2, zero


def spacetime_coeffs(M1, M2, zero):
    """Leading-order coefficients of space-time decoders from grouped masses."""
    has1 = M1.sum(axis=1) > 0
    out = {}
    # MLD
    key1 = np.where(has1[:, None], M1 * 1e6 + M2, M2)        # lexicographic (M1, then M2)
    cstar = np.argmax(key1, axis=1)
    cstar[zero] = 0
    rows = np.arange(len(M1))
    out["MLD_first"] = float((M1.sum(1) - M1[rows, cstar]).sum())
    out["MLD_second"] = float((M2.sum(1) - M2[rows, cstar]).sum())
    # detection
    nz = np.ones(len(M1), dtype=bool)
    nz[zero] = False
    out["detect_rej_first"] = float(M1[nz].sum())
    out["detect_fail_second"] = float(M2[zero].sum() - M2[zero, 0])
    # strict hybrid
    n1 = (M1 > 0).sum(axis=1)
    n2 = (M2 > 0).sum(axis=1)
    rej = np.where(has1, n1 > 1, n2 > 1)
    rej[zero] = False
    c_h = np.where(has1, np.argmax(M1, axis=1), np.argmax(M2, axis=1))
    c_h[zero] = 0
    acc = ~rej
    out["hybrid_rej_first"] = float(M1[rej].sum())
    out["hybrid_rej_second"] = float(M2[rej & ~has1].sum())
    out["hybrid_fail_second"] = float((M2[acc].sum(1) - M2[acc][np.arange(acc.sum()), c_h[acc]]).sum())
    return out


def section_S5(codes, CC, args):
    log("\n=== S5: phenomenological noise, exact ===")
    out = {"memoryless": {}, "spacetime_leading": {}, "validation": {}}
    rng = np.random.default_rng(2026)
    # (a) validation of the transfer matrices against direct simulation
    t0 = time.time()
    for key, dec in (("512", "detect"), ("512", "hybrid"), ("512", "lookup"), ("513", "lookup")):
        cc = CC[key]
        q = qm = 0.05
        T = 6
        ex = phen_memoryless_exact_T(cc, q, qm, dec, T)
        mc = phen_memoryless_mc(cc, q, qm, dec, T, 200000, rng)
        diff = max(abs(ex[c] - mc[c]) for c in CHARS)
        out["validation"][f"{key}_{dec}"] = {"max_abs_diff": float(diff),
                                             "exact_acc": float(ex[(0, 0)].real),
                                             "mc_acc": float(mc[(0, 0)].real)}
        log(f"validation {codes[key].name} {dec}: exact vs 2e5-shot simulation of E[chi(l) 1_acc] after "
            f"T={T} rounds at q=q_m=0.05: max |diff| = {diff:.4f} (1-sigma MC error ~ {1 / math.sqrt(200000):.4f})")
    log(f"   [{time.time() - t0:.1f}s]")
    # (b) memoryless decoders vs q
    qlist = [1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7.5e-2, 0.1]
    if args.quick:
        qlist = [1e-3, 1e-2, 5e-2]
    t0 = time.time()
    for key in ("512", "513", "713", "913", "1313"):
        cc = CC[key]
        decs = ["detect", "hybrid", "lookup"] if key == "512" else ["lookup"]
        ql = qlist if key != "1313" else ([1e-2] if not args.quick else [])
        for dec in decs:
            rec = {}
            for rho in (1.0, 0.5, 0.0):
                for q in ql:
                    if key == "1313" and rho != 1.0:
                        continue
                    acc, eps, _ = phen_memoryless(cc, q, rho * q, dec)
                    rec[f"rho={rho},q={q}"] = {"acc": acc, "eps": eps}
            # leading coefficient at small q
            lead = {}
            for rho in (1.0, 0.5, 0.0):
                if key == "1313" and rho != 1.0:
                    continue
                qs = 1e-4 if key != "1313" else 1e-3
                acc, eps, _ = phen_memoryless(cc, qs, rho * qs, dec)
                lead[rho] = {"acc_loss_over_q": (1 - acc) / qs, "eps_over_q": eps / qs,
                             "eps_over_q2": eps / qs ** 2}
            rec["small_q"] = lead
            # break-even (rho = 1)
            if dec == "lookup" and key != "1313":
                qb, _ = break_even(lambda q, cc=cc: phen_memoryless(cc, q, q, "lookup")[1] - q,
                                   list(np.geomspace(2e-3, 0.3, 16)))
                rec["break_even_rho1"] = qb
            if dec in ("detect", "hybrid"):
                qb, _ = break_even(lambda q, cc=cc, dec=dec: phen_memoryless(cc, q, q, dec)[1] - q,
                                   list(np.geomspace(2e-3, 0.45, 16)))
                rec["break_even_acc_rho1"] = qb
            out["memoryless"][f"{key}_{dec}"] = rec
            L1 = lead[1.0]
            if key == "512":
                log(f"{codes[key].name} memoryless {dec}: q_m=q: per-round rejection {L1['acc_loss_over_q']:.3f} q, "
                    f"accepted eps {L1['eps_over_q']:.5f} q = {L1['eps_over_q2']:.4f} q^2 (q=1e-4); "
                    + (f"q_m=0: rejection {lead[0.0]['acc_loss_over_q']:.3f} q, eps {lead[0.0]['eps_over_q2']:.2f} q^2; "
                       if dec != "lookup" else f"q_m=0: eps {lead[0.0]['eps_over_q']:.4f} q; ")
                    + f"break-even {rec.get('break_even_rho1', rec.get('break_even_acc_rho1'))}")
            else:
                log(f"{codes[key].name} memoryless MLD lookup: eps/q^2 at small q = "
                    + ", ".join(f"{lead[rh]['eps_over_q2']:.2f} (q_m={rh}q)" for rh in lead)
                    + (f"; break-even (q_m=q) q* = {rec.get('break_even_rho1')}" if key != "1313" else ""))
            for q in [1e-2, 5e-2]:
                kq = f"rho=1.0,q={q}"
                if kq in rec:
                    log(f"     q=q_m={q}: acc {rec[kq]['acc']:.5f}, eps {rec[kq]['eps']:.4e} "
                        f"(eps/q {rec[kq]['eps'] / q:.3f}, eps/q^2 {rec[kq]['eps'] / q ** 2:.2f})")
    log(f"   [memoryless {time.time() - t0:.1f}s]")
    # (c) space-time decoders, exact leading orders by fault enumeration
    t0 = time.time()
    for key in ("512", "513", "713", "913", "1313"):
        code = codes[key]
        rec = {}
        for rho in (1.0, 0.5, 0.0):
            per = {}
            for T in ((3, 4, 5) if rho == 1.0 else (4, 5)):
                M1, M2, zero = spacetime_faults(code, T, rho)
                per[T] = spacetime_coeffs(M1, M2, zero)
            d = {k: per[5][k] - per[4][k] for k in per[5]}
            d_prev = ({k: per[4][k] - per[3][k] for k in per[4]} if 3 in per else d)
            rec[rho] = {"per_round": d, "per_round_T3to4": d_prev, "T5_total": per[5]}
        out["spacetime_leading"][key] = rec
        R1 = rec[1.0]["per_round"]
        keys_ok = (["MLD_first", "detect_rej_first", "detect_fail_second", "hybrid_rej_first",
                    "hybrid_fail_second"] if key == "512" else list(R1))
        stable = all(abs(rec[1.0]["per_round"][k] - rec[1.0]["per_round_T3to4"][k]) < 1e-9
                     for k in keys_ok)
        if key == "512":
            log(f"{code.name} space-time, per bulk round (q_m = q): MLD fails {R1['MLD_first']:.4f} q; "
                f"detection rejects {R1['detect_rej_first']:.3f} q, accepted error {R1['detect_fail_second']:.4f} q^2; "
                f"hybrid rejects {R1['hybrid_rej_first']:.3f} q (+{R1['hybrid_rej_second']:.2f} q^2), "
                f"accepted error {R1['hybrid_fail_second']:.3f} q^2 "
                f"[q_m = 0: MLD {rec[0.0]['per_round']['MLD_first']:.4f} q]; increments stable from T=3: {stable}")
        else:
            log(f"{code.name} space-time MLD, per bulk round: eps = A q^2 with A = "
                + ", ".join(f"{rec[rh]['per_round']['MLD_second']:.3f} (q_m={rh}q)" for rh in (1.0, 0.5, 0.0))
                + f"; first order {R1['MLD_first']:.1e}; detection rejects {R1['detect_rej_first']:.2f} q with "
                f"accepted error {R1['detect_fail_second']:.3f} q^2; hybrid accepted error "
                f"{R1['hybrid_fail_second']:.3f} q^2, rejection {R1['hybrid_rej_second']:.3f} q^2; stable: {stable}")
    log(f"   [space-time enumeration {time.time() - t0:.1f}s]")
    RESULTS["S5"] = out
    return out


# ---------------------------------------------------------------------------
# S6: phenomenological noise, Monte Carlo with the exact space-time MLD
# ---------------------------------------------------------------------------

def mc_spacetime(cc, q, qm, T, shots, rng, batch):
    """Sample T noisy rounds; after every round t decode with the exact Bayesian
    filter as if a perfect round revealing s_t followed (the prefix of a T-round
    run is distributed as a t-round run).  Returns P_L(t), t = 1..T: the
    probability that the MLD class differs from the true accumulated class."""
    r, m = cc.r, cc.r + 2
    sh = (3,) * m
    axes = tuple(range(1, m + 1))
    Pq = cc.Pq(q)
    Khat = np.fft.fftn(Pq.reshape(sh))
    flat = Pq.reshape(-1) / Pq.sum()
    wv = np.array([1 - qm, qm / 2, qm / 2])
    ar = np.arange(3)
    pw = 3 ** np.arange(r - 1, -1, -1)
    fails = np.zeros(T)
    done = 0
    while done < shots:
        B = min(batch, shots - done)
        alpha = np.zeros((B,) + sh)
        alpha[(slice(None),) + (0,) * m] = 1.0
        s = np.zeros((B, r), dtype=np.int64)
        lcls = np.zeros(B, dtype=np.int64)
        for t in range(T):
            u = rng.choice(flat.size, size=B, p=flat)
            ds = np.array(np.unravel_index(u // 9, (3,) * r)).T
            dl1, dl2 = np.divmod(u % 9, 3)
            l1, l2 = np.divmod(lcls, 3)
            lcls = ((l1 + dl1) % 3) * 3 + (l2 + dl2) % 3
            s = (s + ds) % 3
            o = (s + rng.choice(3, size=(B, r), p=wv)) % 3
            alpha = np.fft.ifftn(np.fft.fftn(alpha, axes=axes) * Khat, axes=axes).real
            np.clip(alpha, 0.0, None, out=alpha)
            Lk = np.ones((B,) + (1,) * m)
            for c in range(r):
                vals = wv[(o[:, c][:, None] - ar[None, :]) % 3]
                shp = [B] + [1] * m
                shp[1 + c] = 3
                Lk = Lk * vals.reshape(shp)
            alpha = alpha * Lk
            alpha /= alpha.reshape(B, -1).sum(axis=1).reshape((B,) + (1,) * m)
            post = alpha.reshape(B, 3 ** r, 9)[np.arange(B), s @ pw]
            fails[t] += np.sum(post.argmax(axis=1) != lcls)
        done += B
    return fails / shots


def per_round_from_PL(PL, shots, t0):
    """eps per round from f_L(t) = 1 - 9/8 P_L(t): f = (f_L(T)/f_L(t0))^(1/(T-t0))."""
    T = len(PL)
    fL = 1 - 9.0 / 8.0 * np.asarray(PL)
    f = (fL[T - 1] / fL[t0 - 1]) ** (1.0 / (T - t0))
    eps = 8.0 / 9.0 * (1 - f)
    # crude 1-sigma band from binomial errors of the two endpoints
    se = [math.sqrt(max(PL[i] * (1 - PL[i]), 1e-12) / shots) * 9.0 / 8.0 for i in (t0 - 1, T - 1)]
    lo = 8.0 / 9.0 * (1 - ((fL[T - 1] + se[1]) / (fL[t0 - 1] - se[0])) ** (1.0 / (T - t0)))
    hi = 8.0 / 9.0 * (1 - ((fL[T - 1] - se[1]) / (fL[t0 - 1] + se[0])) ** (1.0 / (T - t0)))
    return float(eps), float(lo), float(hi)


def section_S6(codes, CC, args):
    log("\n=== S6: phenomenological noise, Monte Carlo of the exact space-time MLD (q_m = q) ===")
    out = {}
    rng = np.random.default_rng(7)
    plan = [("512", [0.03], 6, 5000, 5000),
            ("513", [0.01, 0.08, 0.12], 6, 6000, 6000),
            ("713", [0.02], 5, 1500, 500)]
    if args.mc913:
        plan.append(("913", [0.03, 0.06], 5, 600, 60))
    if args.quick:
        plan = [("512", [0.03], 6, 4000, 2000), ("513", [0.05], 6, 4000, 2000)]
    for key, qs, T, shots, batch in plan:
        cc = CC[key]
        out[key] = {}
        for q in qs:
            t0 = time.time()
            PL = mc_spacetime(cc, q, q, T, shots, rng, batch)
            eps, lo, hi = per_round_from_PL(PL, shots, 2)
            out[key][q] = {"PL": [float(x) for x in PL], "eps": eps, "eps_lo": lo, "eps_hi": hi,
                           "shots": shots, "T": T}
            lead = RESULTS.get("S5", {}).get("spacetime_leading", {}).get(key, {}).get(1.0, {}).get("per_round", {})
            if not lead:
                pred = float("nan")
            elif lead.get("MLD_first", 0) > 0:          # distance 2: only the first order is exact
                pred = lead["MLD_first"] * q
            else:
                pred = lead["MLD_second"] * q * q
            log(f"{codes[key].name} q=q_m={q}: space-time MLD eps per round {eps:.4e} "
                f"[{lo:.3e}, {hi:.3e}] ({shots} shots, T={T}); eps/q {eps / q:.3f}, eps/q^2 {eps / q ** 2:.2f}; "
                f"leading-order prediction {pred:.3e} [{time.time() - t0:.1f}s]")
        # space-time break-even (eps = q) by log-linear interpolation of eps/q between MC points
        qs_sorted = sorted(out[key])
        ratio = [out[key][q]["eps"] / q for q in qs_sorted]
        be = None
        for (qa, ra), (qb, rb) in zip(zip(qs_sorted, ratio), zip(qs_sorted[1:], ratio[1:])):
            if (ra - 1) * (rb - 1) <= 0 and ra != rb:
                x = math.log(qa) + (math.log(qb) - math.log(qa)) * (0 - math.log(ra)) / (math.log(rb) - math.log(ra))
                be = math.exp(x)
        out[key]["break_even_interp"] = be
        if be:
            log(f"   {codes[key].name}: space-time MLD break-even (q_m = q) by interpolation: q* ~ {be:.3f}")
    RESULTS["S6"] = out
    return out


# ---------------------------------------------------------------------------
# plot, main
# ---------------------------------------------------------------------------

def make_plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log(f"(plot skipped: {exc})")
        return
    S4 = RESULTS.get("S4", {})
    S5 = RESULTS.get("S5", {}).get("memoryless", {})
    S6 = RESULTS.get("S6", {})
    lab = {"512": "[[5,1,2]]", "513": "[[5,1,3]]", "713": "[[7,1,3]]", "913": "[[9,1,3]]", "1313": "[[13,1,3]] folded"}
    col = {"512": "#4c4c4c", "513": "#1f77b4", "713": "#2ca02c", "913": "#d62728", "1313": "#9467bd"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    for key, rec in S4.items():
        qs = sorted(rec["curve"])
        ax.loglog(qs, [rec["curve"][q]["MLD"] for q in qs], "-o", ms=3, color=col[key], label=f"{lab[key]} MLD")
        if key == "512":
            ax.loglog(qs, [rec["curve"][q]["detect_eps_acc"] for q in qs], ":", color=col[key], label="[[5,1,2]] detection (accepted)")
            ax.loglog(qs, [rec["curve"][q]["hybrid_eps_acc"] for q in qs], "--", color=col[key], label="[[5,1,2]] hybrid (accepted)")
    qq = np.geomspace(1e-4, 0.3, 50)
    ax.loglog(qq, qq, "k-", lw=2, alpha=0.4, label="unencoded qutrit (eps = q)")
    ax.set_xlabel("q (depolarizing probability per data qutrit per round)")
    ax.set_ylabel("logical error per round eps")
    ax.set_title("code capacity (exact)")
    ax.set_ylim(1e-9, 1)
    ax.legend(fontsize=7)
    ax = axs[1]
    for k, rec in S5.items():
        key, dec = k.split("_")
        pts = sorted((float(kk.split("q=")[1]), v["eps"]) for kk, v in rec.items()
                     if kk.startswith("rho=1.0,q="))
        if not pts:
            continue
        style = {"lookup": "-o", "detect": ":", "hybrid": "--"}[dec]
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], style, ms=3, color=col[key],
                  label=f"{lab[key]} {'memoryless lookup' if dec == 'lookup' else dec + ' (accepted)'}")
    for key, rec in S6.items():
        qs = sorted(k for k in rec if isinstance(k, float))
        ax.errorbar(qs, [rec[q]["eps"] for q in qs],
                    yerr=[[rec[q]["eps"] - rec[q]["eps_lo"] for q in qs], [rec[q]["eps_hi"] - rec[q]["eps"] for q in qs]],
                    fmt="s", ms=5, mfc="none", color=col[key], label=f"{lab[key]} space-time MLD (MC)")
    ax.loglog(qq, qq, "k-", lw=2, alpha=0.4, label="unencoded qutrit")
    ax.set_xlabel("q = q_m")
    ax.set_title("phenomenological (q_m = q)")
    ax.set_ylim(1e-7, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    log(f"plot written: {path}")


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, Fraction):
        return frac_str(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip slow searches, short grids, small MC")
    ap.add_argument("--mc913", action="store_true", help="also run the joint space-time MC for [[9,1,3]]_3 (~2 min)")
    ap.add_argument("--sections", default="1,2,3,4,5,6")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    log(f"A5_code_capacity_compare.py  numpy {np.__version__}  args {vars(args)}")
    codes = {"512": code_512(), "513": code_513(), "713": code_713(), "913": code_913(), "1313": code_1313()}
    secs = set(args.sections.split(","))
    if "1" in secs:
        section_S1(codes)
    if "2" in secs:
        section_S2(codes, args)
    if "3" in secs:
        section_S3(codes, args)
    CC = None
    if secs & {"4", "5", "6"}:
        _, CC = section_S4(codes, args)
    if "5" in secs:
        section_S5(codes, CC, args)
    if "6" in secs:
        section_S6(codes, CC, args)
    log(f"\ntotal runtime {time.time() - t_all:.1f}s")
    suffix = "_quick" if args.quick else ""
    with open(os.path.join(OUT, f"A5_results{suffix}.json"), "w") as fh:
        json.dump(to_jsonable(RESULTS), fh, indent=1)
    if "4" in secs:
        make_plot(os.path.join(OUT, f"A5_eps_vs_q{suffix}.png"))
    with open(os.path.join(OUT, f"A5_run_log{suffix}.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
