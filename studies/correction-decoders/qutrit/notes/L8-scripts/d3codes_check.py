#!/usr/bin/env python3
"""Symplectic (mod-3) checks of distance-3 qutrit code candidates for LRB.

Pauli X^x Z^z on n qutrits is the vector [x_0..x_{n-1} | z_0..z_{n-1}] mod 3
(phases ignored; for odd prime d a Clifford that preserves the stabilizer
space symplectically preserves the code up to a known Pauli frame).
Gate conventions follow sdim: F (=H): X->Z, Z->X^-1;  P: X->XZ, Z->Z;
SUM (=CNOT) c->t: X_c->X_c X_t, Z_t->Z_c^-1 Z_t;  CZ: X_i->X_i Z_j.
"""
import itertools
import os
import sys
import time
from collections import defaultdict

# keep the run on one core (login-node rule); must be set before numpy is imported
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

p = 3
INV = {1: 1, 2: 2}
T0 = time.time()


def mod(a):
    return np.mod(a, p)


def rref(M):
    M = mod(np.array(M, dtype=np.int64))
    if M.ndim == 1:
        M = M[None, :]
    R = M.copy()
    rows, cols = R.shape
    piv = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(R[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + nz[0]
        if i != r:
            R[[r, i]] = R[[i, r]]
        R[r] = mod(R[r] * INV[int(R[r, c])])
        for j in range(rows):
            if j != r and R[j, c]:
                R[j] = mod(R[j] - R[j, c] * R[r])
        piv.append(c)
        r += 1
    return R[:r], piv


def rank(M):
    M = np.asarray(M)
    if M.size == 0:
        return 0
    return len(rref(M)[1])


def nullspace(M):
    M = mod(np.asarray(M, dtype=np.int64))
    n = M.shape[1]
    R, piv = rref(M)
    free = [c for c in range(n) if c not in piv]
    B = []
    for f in free:
        v = np.zeros(n, dtype=np.int64)
        v[f] = 1
        for i, pc in enumerate(piv):
            v[pc] = mod(-R[i, f])
        B.append(v)
    return np.array(B, dtype=np.int64).reshape(-1, n)


def in_rowspace_many(B, W):
    R, piv = rref(B)
    Wr = mod(np.array(W, dtype=np.int64))
    for i, pc in enumerate(piv):
        Wr = mod(Wr - np.outer(Wr[:, pc], R[i]))
    return ~Wr.any(axis=1)


def symp(U, V, n):
    U = np.atleast_2d(U)
    V = np.atleast_2d(V)
    return mod(U[:, :n] @ V[:, n:].T - U[:, n:] @ V[:, :n].T)


def pauli(n, xs=None, zs=None):
    v = np.zeros(2 * n, dtype=np.int64)
    for q, a in (xs or {}).items():
        v[q] = a % 3
    for q, b in (zs or {}).items():
        v[n + q] = b % 3
    return v


class Code:
    def __init__(self, name, S, n, LX=None, LZ=None):
        self.name = name
        self.n = n
        self.S = mod(np.array(S, dtype=np.int64))
        assert np.all(symp(self.S, self.S, n) == 0), name + ": stabilizers do not commute"
        self.r = rank(self.S)
        self.k = n - self.r
        if LX is None or LZ is None:
            LX, LZ = self.find_logicals()
        self.LX = mod(np.array(LX, dtype=np.int64))
        self.LZ = mod(np.array(LZ, dtype=np.int64))
        assert np.all(symp(self.LX, self.S, n) == 0), name + ": LX not in normalizer"
        assert np.all(symp(self.LZ, self.S, n) == 0), name + ": LZ not in normalizer"
        self.c = int(symp(self.LX, self.LZ, n)[0, 0])
        assert self.c != 0, name + ": LX, LZ commute"

    def find_logicals(self):
        n = self.n
        A = mod(np.concatenate([self.S[:, n:], -self.S[:, :n]], axis=1))
        N = nullspace(A)
        cands = [v for v in N if rank(np.vstack([self.S, v])) > self.r]
        for u in cands:
            for w in cands:
                if symp(u, w, n)[0, 0] != 0:
                    return u, w
        raise ValueError("no logicals")

    def syndrome(self, V):
        return symp(V, self.S, self.n)

    def logical_class(self, V):
        cinv = INV[self.c]
        a = mod(symp(V, self.LZ, self.n)[:, 0] * cinv)
        b = mod(-symp(V, self.LX, self.n)[:, 0] * cinv)
        return np.stack([a, b], axis=1)

    def is_nontrivial_logical(self, V):
        V = np.atleast_2d(V)
        z = np.all(self.syndrome(V) == 0, axis=1)
        return z & self.logical_class(V).any(axis=1)

    def distance(self, wmax=4):
        for w in range(1, wmax + 1):
            V = paulis_weight(self.n, w)
            bad = self.is_nontrivial_logical(V)
            if bad.any():
                return w, V[bad]
        return None, None

    def weights(self):
        n = self.n
        return [int(np.count_nonzero(self.S[i, :n] + 3 * self.S[i, n:])) for i in range(len(self.S))]


LOCAL = [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]


def paulis_weight(n, w, sites=None):
    sites = list(range(n)) if sites is None else list(sites)
    supps = list(itertools.combinations(sites, w))
    locs = list(itertools.product(LOCAL, repeat=w))
    out = np.zeros((len(supps) * len(locs), 2 * n), dtype=np.int64)
    t = 0
    for supp in supps:
        for loc in locs:
            for q, (a, b) in zip(supp, loc):
                out[t, q] = a
                out[t, n + q] = b
            t += 1
    return out


def all_on(n, sites):
    """All Paulis (including identity) supported on `sites`."""
    sites = list(sites)
    out = []
    for loc in itertools.product([(a, b) for a in range(3) for b in range(3)], repeat=len(sites)):
        v = np.zeros(2 * n, dtype=np.int64)
        for q, (a, b) in zip(sites, loc):
            v[q] = a
            v[n + q] = b
        out.append(v)
    return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------- gates
MF = ((0, 2), (1, 0))      # X->Z, Z->X^-1
MFd = ((0, 1), (2, 0))     # X->Z^-1, Z->X
MP = ((1, 0), (1, 1))      # X->XZ
MPd = ((1, 0), (2, 1))     # X->XZ^-1
MI = ((1, 0), (0, 1))
LOCMAT = {"F": MF, "Fd": MFd, "P": MP, "Pd": MPd, "I": MI}


def apply_local(V, n, q, M):
    V = V.copy()
    x = V[:, q].copy()
    z = V[:, n + q].copy()
    V[:, q] = mod(M[0][0] * x + M[0][1] * z)
    V[:, n + q] = mod(M[1][0] * x + M[1][1] * z)
    return V


def apply_gate(V, n, g, qs):
    V = np.atleast_2d(V).copy()
    if g in LOCMAT:
        return apply_local(V, n, qs[0], LOCMAT[g])
    i, j = qs
    if g in ("CZ", "CZd"):
        s = 1 if g == "CZ" else -1
        xi = V[:, i].copy()
        xj = V[:, j].copy()
        V[:, n + i] = mod(V[:, n + i] + s * xj)
        V[:, n + j] = mod(V[:, n + j] + s * xi)
    elif g == "SWAP":
        V[:, [i, j]] = V[:, [j, i]]
        V[:, [n + i, n + j]] = V[:, [n + j, n + i]]
    elif g in ("SUM", "SUMd"):
        s = 1 if g == "SUM" else -1
        V[:, j] = mod(V[:, j] + s * V[:, i])
        V[:, n + i] = mod(V[:, n + i] - s * V[:, n + j])
    else:
        raise ValueError(g)
    return V


def apply_circuit(V, n, circ):
    V = np.atleast_2d(V)
    for g, qs in circ:
        V = apply_gate(V, n, g, qs)
    return V


def apply_monomial(V, n, mats, perm):
    """site i -> perm[i] with local 2x2 matrix mats[i]."""
    V = np.atleast_2d(V)
    W = np.zeros_like(V)
    for i in range(n):
        M = mats[i]
        x = V[:, i]
        z = V[:, n + i]
        W[:, perm[i]] = mod(M[0][0] * x + M[0][1] * z)
        W[:, n + perm[i]] = mod(M[1][0] * x + M[1][1] * z)
    return W


def logical_action_from_images(code, imgS, imgL):
    if np.any(code.syndrome(imgS) != 0) or np.any(code.logical_class(imgS).any(axis=1)):
        return None
    if np.any(code.syndrome(imgL) != 0):
        return None
    lc = code.logical_class(imgL)
    return tuple(int(v) for v in (lc[0, 0], lc[1, 0], lc[0, 1], lc[1, 1]))  # row-major of [img(X) img(Z)]


def logical_action(code, circ):
    imgS = apply_circuit(code.S, code.n, circ)
    imgL = apply_circuit(np.vstack([code.LX, code.LZ]), code.n, circ)
    return logical_action_from_images(code, imgS, imgL)


# --------------------------------------------------------- SL(2,3) helpers
def m2(t):
    return np.array(t, dtype=np.int64).reshape(2, 2)


SL23 = [tuple(int(v) for v in (a, b, c, d))
        for a, b, c, d in itertools.product(range(3), repeat=4) if (a * d - b * c) % 3 == 1]
NAMES = {(0, 2, 1, 0): "F", (1, 0, 1, 1): "P", (1, 0, 0, 1): "I", (2, 0, 0, 2): "F^2(-I)"}


def gen_group(gens):
    G = {(1, 0, 0, 1)}
    frontier = list(G)
    gens = [m2(g) for g in gens]
    while frontier:
        new = []
        for h in frontier:
            for g in gens:
                prod = tuple(int(v) for v in mod(g @ m2(h)).reshape(-1))
                if prod not in G:
                    G.add(prod)
                    new.append(prod)
        frontier = new
    return G


def order_of(t):
    M = m2(t)
    A = np.eye(2, dtype=np.int64)
    for k in range(1, 30):
        A = mod(M @ A)
        if np.array_equal(A, np.eye(2, dtype=np.int64)):
            return k
    return None


def transitive_on_nonzero(G):
    orbit = set()
    for g in G:
        v = mod(m2(g) @ np.array([1, 0]))
        orbit.add(tuple(int(x) for x in v))
    return len(orbit) == 8


# ------------------------------------------------------ fault-set checks
def signature_map(code, V):
    syn = code.syndrome(V)
    lc = code.logical_class(V)
    d = {}
    for s, l in zip(map(tuple, syn), map(tuple, lc)):
        if s in d and d[s] != l:
            return None
        d[s] = l
    return d


def merge_ok(d1, d2):
    if d1 is None or d2 is None:
        return False
    for k, v in d2.items():
        if k in d1 and d1[k] != v:
            return False
    return True


def singles(n):
    return np.vstack([np.zeros((1, 2 * n), dtype=np.int64), paulis_weight(n, 1)])


def check_faults(code, g, order):
    """Data errors from one fault in a single-ancilla check of generator g with the
    given qudit order: (any single-qudit error on order[j]) x (g restricted to the
    qudits after j)^a, a in {0,1,2}."""
    n = code.n
    g = mod(np.array(g, dtype=np.int64))
    rows = []
    for j in range(len(order)):
        suffix = np.zeros(2 * n, dtype=np.int64)
        for q in order[j + 1:]:
            suffix[q] = g[q]
            suffix[n + q] = g[n + q]
        for a in range(3):
            for (ex, ez) in [(x, z) for x in range(3) for z in range(3)]:
                v = mod(a * suffix)
                v[order[j]] = mod(v[order[j]] + ex)
                v[n + order[j]] = mod(v[n + order[j]] + ez)
                rows.append(v)
    return np.unique(np.array(rows, dtype=np.int64), axis=0)


def support(g, n):
    return [q for q in range(n) if g[q] % 3 or g[n + q] % 3]


def order_search(code, gens, label, max_report=3):
    """For each generator choose a qudit order; find assignments for which all single
    faults (singles + hooks) are correctable by a syndrome lookup (no two single-fault
    data errors with equal syndrome and different logical class)."""
    n = code.n
    base = signature_map(code, singles(n))
    assert base is not None, "single-qudit errors not correctable"
    opts = []
    for g in gens:
        supp = support(g, n)
        lst = []
        for o in itertools.permutations(supp):
            # orders that are cyclic rotations/reversals give different hooks, keep all
            d = signature_map(code, np.vstack([singles(n), check_faults(code, g, o)]))
            lst.append((o, d))
        opts.append(lst)
    m = len(gens)
    ok_single = [[merge_ok(base, d) and d is not None for (o, d) in lst] for lst in opts]
    # pairwise compatibility
    pair_ok = {}
    for i in range(m):
        for j in range(i + 1, m):
            T = np.zeros((len(opts[i]), len(opts[j])), dtype=bool)
            for a, (oa, da) in enumerate(opts[i]):
                if not ok_single[i][a]:
                    continue
                for b, (ob, db) in enumerate(opts[j]):
                    if ok_single[j][b]:
                        T[a, b] = merge_ok(da, db)
            pair_ok[(i, j)] = T
    sols = []
    count = [0]

    CAP = 5000

    def bt(i, chosen):
        if count[0] >= CAP:
            return
        if i == m:
            count[0] += 1
            if len(sols) < max_report:
                sols.append([opts[t][c][0] for t, c in enumerate(chosen)])
            return
        for a in range(len(opts[i])):
            if not ok_single[i][a]:
                continue
            if all(pair_ok[(t, i)][chosen[t], a] for t in range(i)):
                bt(i + 1, chosen + [a])

    bt(0, [])
    total = 1
    for lst in opts:
        total *= len(lst)
    frac_single = [sum(x) / len(x) for x in ok_single]
    cnt = f">={CAP}" if count[0] >= CAP else str(count[0])
    print(f"  [{label}] hook-safe order assignments: {cnt} of {total}; "
          f"per-check fraction of orders individually safe: {[round(f, 3) for f in frac_single]}")
    for s in sols:
        print("     example:", s)
    return count[0], total, sols, opts


def correctable(code, V):
    return signature_map(code, np.vstack([singles(code.n), V])) is not None


def detect_report(code, V, label):
    """For a detection code: single faults that are undetectable nontrivial logicals."""
    bad = code.is_nontrivial_logical(V)
    print(f"  [{label}] single faults giving undetectable logical errors: {int(bad.sum())} of {len(V)}")
    return V[bad]


def fmt(v, n):
    terms = []
    for q in range(n):
        a, b = int(v[q]) % 3, int(v[n + q]) % 3
        if a or b:
            s = ""
            if a:
                s += "X" + ("" if a == 1 else "^-1")
            if b:
                s += "Z" + ("" if b == 1 else "^-1")
            terms.append(f"{s}[{q}]")
    return " ".join(terms) if terms else "I"


# ------------------------------------------------- meet-in-the-middle search
SL23M = [m2(t) for t in SL23]


def local_symp(u, w):
    return (u[0] * w[1] - u[1] * w[0]) % 3


def monomial_automorphisms(code, perms, split=None, target=None):
    """All (perm, local SL(2,3) per site) mapping code.S into target.S (default: itself).
    Returns list of (perm, mats_idx, logical action or None)."""
    tgt = code if target is None else target
    n = code.n
    Sg = code.S
    Nb = np.vstack([tgt.S, tgt.LX, tgt.LZ])  # spans normalizer of target
    split = split if split is not None else n // 2 + 1
    res = []
    for perm in perms:
        # contribution of site i with matrix M: entries (j,l) = <M g_j[i], Nb_l[perm i]>
        contrib = []
        for i in range(n):
            gi = np.stack([Sg[:, i], Sg[:, n + i]], axis=1)          # (r,2)
            ni = np.stack([Nb[:, perm[i]], Nb[:, n + perm[i]]], axis=1)  # (L,2)
            C = np.zeros((24, gi.shape[0] * ni.shape[0]), dtype=np.int8)
            for t, M in enumerate(SL23M):
                img = mod(gi @ M.T)                                   # (r,2)
                C[t] = mod(np.outer(img[:, 0], ni[:, 1]) - np.outer(img[:, 1], ni[:, 0])).reshape(-1)
            contrib.append(C)
        A_sites = list(range(split))
        B_sites = list(range(split, n))
        L = contrib[0].shape[1]

        def sums(sites):
            idx = list(itertools.product(range(24), repeat=len(sites)))
            tot = np.zeros((len(idx), L), dtype=np.int8)
            arr = np.array(idx, dtype=np.int64)
            for c, st in enumerate(sites):
                tot += contrib[st][arr[:, c]]
            return arr, mod(tot).astype(np.int8)

        Aidx, Asum = sums(A_sites)
        Bidx, Bsum = sums(B_sites)
        Bkeys = defaultdict(list)
        for t, row in enumerate(Bsum):
            Bkeys[row.tobytes()].append(t)
        negA = mod(-Asum.astype(np.int16)).astype(np.int8)
        for t, row in enumerate(negA):
            hits = Bkeys.get(row.tobytes())
            if hits:
                for h in hits:
                    mats_idx = list(Aidx[t]) + list(Bidx[h])
                    res.append((tuple(perm), tuple(int(m) for m in mats_idx)))
    out = []
    for perm, mats_idx in res:
        mats = [SL23[m] for m in mats_idx]
        Ms = [((mm[0], mm[1]), (mm[2], mm[3])) for mm in mats]
        imgS = apply_monomial(code.S, n, Ms, perm)
        imgL = apply_monomial(np.vstack([code.LX, code.LZ]), n, Ms, perm)
        act = logical_action_from_images(tgt, imgS, imgL) if target is None else None
        out.append((perm, mats_idx, act))
    return out


def summarize_group(label, acts):
    acts = set(a for a in acts if a is not None)
    G = gen_group(list(acts)) if acts else {(1, 0, 0, 1)}
    orders = sorted(order_of(g) for g in G)
    print(f"  [{label}] distinct logical actions: {len(acts)}; generated group order {len(G)}; "
          f"element orders {orders}; transitive on F_3^2\\0 (2-design with Paulis): {transitive_on_nonzero(G)}")
    return G


# ======================================================================
def section(t):
    print("\n" + "=" * 78 + f"\n{t}   (t={time.time() - T0:.1f}s)\n" + "=" * 78)


def brief_512():
    section("1. [[5,1,2]]_3 folded surface code from the brief (LRB-D baseline)")
    n = 5
    S = [pauli(n, zs={0: 1, 1: -1, 3: -1}), pauli(n, zs={1: 1, 2: 1, 4: -1}),
         pauli(n, xs={0: 1, 1: 1, 2: -1}), pauli(n, xs={1: -1, 3: 1, 4: -1})]
    LX = pauli(n, xs={0: 1, 3: 1})
    LZ = pauli(n, zs={0: 1, 2: 1})
    C = Code("512", S, n, LX, LZ)
    d, ex = C.distance()
    print(f"  n={n} k={C.k} d={d}; stabilizer weights {C.weights()}")
    # all weight-2 nontrivial logical supports
    V = paulis_weight(n, 2)
    bad = C.is_nontrivial_logical(V)
    supps = sorted(set(tuple(support(v, n)) for v in V[bad]))
    print(f"  supports of weight-2 nontrivial logicals: {supps}")
    H = [("SWAP", (2, 3))] + [("F", (q,)) for q in (0, 2, 3, 4)] + [("Fd", (1,))]
    Pc = [("P", (0,)), ("P", (4,)), ("Pd", (1,)), ("CZ", (2, 3))]
    print(f"  logical action of brief's H circuit: {logical_action(C, H)}  (F = (0,2,1,0))")
    print(f"  logical action of brief's P circuit: {logical_action(C, Pc)}  (P = (1,0,1,1))")
    # single faults: fold-gate faults on (2,3), check-circuit hooks, singles
    pair = all_on(n, [2, 3])
    detect_report(C, pair, "arbitrary 2-qutrit error on fold pair (2,3) after SWAP/CZ")
    bad23 = pair[C.is_nontrivial_logical(pair)]
    print("     e.g.", [fmt(v, n) for v in bad23[:4]])
    checks = [(S[0], (0, 1, 3)), (S[1], (1, 2, 4)), (S[2], (0, 1, 2)), (S[3], (3, 1, 4))]
    hooks = np.vstack([check_faults(C, g, o) for g, o in checks])
    hb = detect_report(C, hooks, "single faults inside the four check circuits (brief's CNOT order)")
    print("     e.g.", [fmt(v, n) for v in hb[:4]])
    # compare with the D=2 member of the folded-square construction used in section 6
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        Csq, qpos = square13(2)
    brief_pos = {0: (0, 0), 1: (1, 1), 2: (2, 0), 3: (0, 2), 4: (2, 2)}
    perm = [qpos.index(brief_pos[q]) for q in range(n)]   # brief qutrit q -> square index
    Sb = apply_monomial(C.S, n, [MI] * n, perm)
    same = rank(np.vstack([Sb, Csq.S])) == rank(Csq.S) == rank(Sb)
    Lb = apply_monomial(np.vstack([C.LX, C.LZ]), n, [MI] * n, perm)
    lc = Csq.logical_class(Lb)
    print(f"  brief code equals the D=2 folded-square construction (same stabilizer space): {same}; "
          f"brief X-bar, Z-bar map to square logical classes {lc.tolist()}")
    return C


def chau_code():
    section("2. Chau 1997 five-register code for N=3: numerical stabilizer")
    N = 3
    w = np.exp(2j * np.pi / N)
    digits = np.array(list(itertools.product(range(N), repeat=5)), dtype=np.int64)
    pw = N ** np.arange(4, -1, -1)
    cw = []
    for k in range(N):
        psi = np.zeros(N ** 5, dtype=complex)
        for pp, q, r in itertools.product(range(N), repeat=3):
            regs = np.array([(pp + q + k) % N, (pp + r) % N, (q + r) % N, pp, q])
            psi[int(regs @ pw)] += w ** ((k * (pp + q + r) + pp * r) % N)
        cw.append(psi / N ** 1.5)
    Psi = np.array(cw)
    G = Psi.conj() @ Psi.T
    assert np.allclose(G, np.eye(N)), "codewords not orthonormal"
    stab = []
    for a in itertools.product(range(N), repeat=5):
        perm = (((digits + np.array(a)) % N) @ pw)
        for b in itertools.product(range(N), repeat=5):
            ph = w ** ((digits @ np.array(b)) % N)
            WPsi = np.zeros_like(Psi)
            WPsi[:, perm] = Psi * ph
            Mx = Psi.conj() @ WPsi.T
            lam = Mx[0, 0]
            if abs(abs(lam) - 1) < 1e-9 and np.allclose(Mx, lam * np.eye(N)):
                stab.append(list(a) + list(b))
    stab = np.array(stab, dtype=np.int64)
    print(f"  Weyl operators acting as a scalar on the Chau code: {len(stab)} (expect 81 = 3^4)")
    R, piv = rref(stab)
    C = Code("chau", R, 5)
    d, ex = C.distance()
    print(f"  rank {C.r}, k={C.k}, d={d}; generators: {[fmt(g, 5) for g in R]}")
    return C


def five_qutrit():
    section("3. [[5,1,3]]_3 cyclic five-qutrit code: transversal / permutation automorphisms, hooks")
    n = 5
    g0x = [1, 0, 0, -1, 0]
    g0z = [0, 1, -1, 0, 0]
    S = []
    for s in range(4):
        x = np.roll(g0x, s)
        z = np.roll(g0z, s)
        S.append(np.concatenate([x, z]))
    LX = np.array([1] * 5 + [0] * 5)
    LZ = np.array([0] * 5 + [1] * 5)
    C = Code("513", S, n, LX, LZ)
    d, ex = C.distance()
    print(f"  generators X Z Z^-1 X^-1 I (cyclic): k={C.k} d={d}; weights {C.weights()}")
    # min weight of stabilizer elements
    allS = []
    for coef in itertools.product(range(3), repeat=4):
        allS.append(mod(np.array(coef) @ C.S))
    allS = np.array(allS)
    wts = sorted(set(int(np.count_nonzero(v[:n] + 3 * v[n:])) for v in allS[1:]))
    print(f"  weights of nonidentity stabilizer elements: {wts}")
    # uniform transversal
    uni = []
    for t in SL23:
        M = ((t[0], t[1]), (t[2], t[3]))
        act = logical_action_from_images(C, apply_monomial(C.S, n, [M] * n, list(range(n))),
                                         apply_monomial(np.vstack([C.LX, C.LZ]), n, [M] * n, list(range(n))))
        if act is not None:
            uni.append((t, act))
    print(f"  uniform transversal U^{{x5}} that are logical: {len(uni)}: "
          + ", ".join(f"{t}->{a}" for t, a in uni))
    summarize_group("uniform transversal", [a for _, a in uni])
    # nonuniform, identity permutation
    res_id = monomial_automorphisms(C, [tuple(range(n))], split=3)
    summarize_group("local (non-uniform) transversal, no permutation", [a for _, _, a in res_id])
    print(f"     number of local-Clifford automorphisms (perm=id): {len(res_id)}")
    # with all permutations
    perms = list(itertools.permutations(range(n)))
    res_all = monomial_automorphisms(C, perms, split=3)
    nperm = len(set(r[0] for r in res_all))
    print(f"     monomial automorphisms (perm x local): {len(res_all)} over {nperm} permutations")
    G = summarize_group("local Clifford x qudit permutation", [a for _, _, a in res_all])
    # Keppens et al. claim T_L = (S F)^{x5} on the sdim/Keppens convention code
    S2 = []
    gx = [0, 0, 1, -1, 0]
    gz = [0, 1, 0, 0, -1]
    for s in range(4):
        S2.append(np.concatenate([np.roll(gx, s), np.roll(gz, s)]))
    C2 = Code("513sdim", S2, n, LX, LZ)
    TF = [("F", (q,)) for q in range(n)] + [("P", (q,)) for q in range(n)]
    print(f"  sdim/Keppens convention (I Z X X^-1 Z^-1 cyclic): d={C2.distance()[0]}; "
          f"(P F)^x5 logical action: {logical_action(C2, TF)}")
    uni2 = []
    for t in SL23:
        M = ((t[0], t[1]), (t[2], t[3]))
        act = logical_action_from_images(C2, apply_monomial(C2.S, n, [M] * n, list(range(n))),
                                         apply_monomial(np.vstack([C2.LX, C2.LZ]), n, [M] * n, list(range(n))))
        if act is not None:
            uni2.append((t, act))
    print(f"  sdim/Keppens convention: uniform transversal logical gates: "
          + ", ".join(f"{t}->{a}" for t, a in uni2))
    # hooks for single-ancilla extraction
    order_search(C, list(C.S), "[[5,1,3]]_3 single ancilla per generator")
    return C


def css_from(name, n, HX, HZ, LX, LZ):
    S = [np.concatenate([mod(np.array(h)), np.zeros(n, dtype=np.int64)]) for h in HX] + \
        [np.concatenate([np.zeros(n, dtype=np.int64), mod(np.array(h))]) for h in HZ]
    return Code(name, S, n, np.concatenate([mod(np.array(LX)), np.zeros(n, dtype=np.int64)]),
                np.concatenate([np.zeros(n, dtype=np.int64), mod(np.array(LZ))]))


def color7():
    section("4. [[7,1,3]]_3 qutrit Steane / 2D colour code with star bipartition")
    n = 7
    # 0=A,1=B,2=C corners, 3=O centre (unstarred); 4=AB,5=BC,6=CA midpoints (starred)
    faces = [(0, 4, 3, 6), (1, 4, 3, 5), (2, 5, 3, 6)]
    star = {4, 5, 6}
    sgn = [(-1 if q in star else 1) for q in range(n)]
    HX = [[1 if q in f else 0 for q in range(n)] for f in faces]
    HZ = [[sgn[q] if q in f else 0 for q in range(n)] for f in faces]
    C = css_from("713", n, HX, HZ, [1] * n, sgn)
    d, ex = C.distance()
    print(f"  k={C.k} d={d}; weights {C.weights()}; two-qutrit gates per round (1 ancilla/check): {sum(C.weights())}")
    Fst = [("F" if q not in star else "Fd", (q,)) for q in range(n)]
    Pst = [("P" if q not in star else "Pd", (q,)) for q in range(n)]
    aF = logical_action(C, Fst)
    aP = logical_action(C, Pst)
    print(f"  star-transversal F (F unstarred, F^-1 starred): {aF};  star-transversal P: {aP}")
    G = gen_group([aF, aP])
    print(f"  group generated by these two logical actions: order {len(G)} (|SL(2,3)| = 24)")
    print(f"  uniform F^x7 logical? {logical_action(C, [('F', (q,)) for q in range(n)])}; "
          f"uniform P^x7 logical? {logical_action(C, [('P', (q,)) for q in range(n)])}")
    res = monomial_automorphisms(C, [tuple(range(n))], split=4)
    summarize_group("all local-Clifford automorphisms (perm=id)", [a for _, _, a in res])
    print(f"     number of local-Clifford automorphisms (perm=id): {len(res)}")
    order_search(C, list(C.S), "[[7,1,3]]_3 single ancilla per check")
    return C


def rotated9():
    section("5. [[9,1,3]]_3 rotated surface code")
    n = 9
    q = lambda r, c: 3 * r + c
    s = {q(r, c): (1 if (r + c) % 2 == 0 else -1) for r in range(3) for c in range(3)}
    Xf = [[(0, 0), (0, 1), (1, 0), (1, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)], [(0, 1), (0, 2)], [(2, 0), (2, 1)]]
    Zf = [[(0, 1), (0, 2), (1, 1), (1, 2)], [(1, 0), (1, 1), (2, 0), (2, 1)], [(0, 0), (1, 0)], [(1, 2), (2, 2)]]
    HX = [[1 if any(q(*rc) == t for rc in f) else 0 for t in range(n)] for f in Xf]
    HZ = [[s[t] if any(q(*rc) == t for rc in f) else 0 for t in range(n)] for f in Zf]
    S = [np.concatenate([mod(np.array(h)), np.zeros(n, dtype=np.int64)]) for h in HX] + \
        [np.concatenate([np.zeros(n, dtype=np.int64), mod(np.array(h))]) for h in HZ]
    C = Code("913", S, n)
    d, ex = C.distance()
    print(f"  k={C.k} d={d}; weights {C.weights()}; two-qutrit gates per round: {sum(C.weights())}")
    # dihedral symmetries of the 3x3 grid
    def tr(f):
        return [q(*f(r, c)) for r in range(3) for c in range(3)]
    dih = {"id": lambda r, c: (r, c), "rot90": lambda r, c: (c, 2 - r), "rot180": lambda r, c: (2 - r, 2 - c),
           "rot270": lambda r, c: (2 - c, r), "diag": lambda r, c: (c, r), "anti": lambda r, c: (2 - c, 2 - r),
           "horiz": lambda r, c: (2 - r, c), "vert": lambda r, c: (r, 2 - c)}
    for name, f in dih.items():
        perm = tr(f)
        inv = all(perm[perm[t]] == t for t in range(n))
        found = []
        for signs in itertools.product([0, 1], repeat=n):
            mats = [MF if sgn == 0 else MFd for sgn in signs]
            imgS = apply_monomial(C.S, n, mats, perm)
            imgL = apply_monomial(np.vstack([C.LX, C.LZ]), n, mats, perm)
            act = logical_action_from_images(C, imgS, imgL)
            if act is not None:
                found.append(act)
        print(f"  ZX-duality candidate {name:6s} (involution={inv}): F/F^-1 sign patterns giving a logical gate: "
              f"{len(found)}; actions {sorted(set(found))[:3]}")
    # phase-type: diagonal local P^{+-1} plus CZ^{+-1} on orbits of an involution: try all involutive dihedral perms
    for name in ["diag", "anti", "horiz", "vert", "rot180"]:
        perm = tr(dih[name])
        fixed = [t for t in range(n) if perm[t] == t]
        pairs = sorted(set(tuple(sorted((t, perm[t]))) for t in range(n) if perm[t] != t))
        cnt = 0
        for fs in itertools.product([0, 1, 2], repeat=len(fixed)):
            for ps in itertools.product([1, 2], repeat=len(pairs)):
                circ = [("P" if a == 1 else "Pd", (t,)) for t, a in zip(fixed, fs) if a]
                circ += [("CZ" if b == 1 else "CZd", pr) for pr, b in zip(pairs, ps)]
                act = logical_action(C, circ)
                if act is not None and act != (1, 0, 0, 1):
                    cnt += 1
        print(f"  phase-type fold gate on orbits of {name:6s}: nontrivial logical solutions {cnt}")
    order_search(C, list(C.S), "[[9,1,3]]_3 single ancilla per check")
    return C


def square13(D=3):
    section(f"6. Folded square (unrotated) surface code, D={D}: [[{2*D*D-2*D+1},1,{D}]]_3 (Moussa 2016)")
    L = 2 * D - 2
    qpos = [(X, Y) for Y in range(L + 1) for X in range(L + 1) if (X + Y) % 2 == 0]
    idx = {pos: t for t, pos in enumerate(qpos)}
    n = len(qpos)
    verts = [(X, Y) for Y in range(0, L + 1, 2) for X in range(1, L, 2)]
    facs = [(X, Y) for Y in range(1, L, 2) for X in range(0, L + 1, 2)]
    HX, HZ = [], []
    for (X, Y) in verts:   # left +1, right -1, below(+Y) +1, above -1
        h = [0] * n
        for (dx, dy, e) in [(-1, 0, 1), (1, 0, -1), (0, 1, 1), (0, -1, -1)]:
            if (X + dx, Y + dy) in idx:
                h[idx[(X + dx, Y + dy)]] = e
        HX.append(h)
    for (X, Y) in facs:    # top(-Y) +1, bottom -1, left +1, right -1
        h = [0] * n
        for (dx, dy, e) in [(0, -1, 1), (0, 1, -1), (-1, 0, 1), (1, 0, -1)]:
            if (X + dx, Y + dy) in idx:
                h[idx[(X + dx, Y + dy)]] = e
        HZ.append(h)
    LX = [1 if (pos[0] == 0) else 0 for pos in qpos]
    LZ = [1 if (pos[1] == 0) else 0 for pos in qpos]
    C = css_from(f"sq{D}", n, HX, HZ, LX, LZ)
    d, ex = C.distance(wmax=D)
    print(f"  n={n} k={C.k} d={d}; weights {C.weights()}; two-qutrit gates per round: {sum(C.weights())}")
    ee = lambda pos: pos[0] % 2 == 0
    tau = [idx[(pos[1], pos[0])] for pos in qpos]
    fixed = [t for t in range(n) if tau[t] == t]
    pairs = sorted(set(tuple(sorted((t, tau[t]))) for t in range(n) if tau[t] != t))
    print(f"  fold qudits {len(fixed)}, fold pairs {len(pairs)}")
    Hc = [("SWAP", pr) for pr in pairs] + [("F" if ee(qpos[t]) else "Fd", (t,)) for t in range(n)]
    Pc = [("P" if ee(qpos[t]) else "Pd", (t,)) for t in fixed] + \
         [("CZ" if ee(qpos[pr[0]]) else "CZd", pr) for pr in pairs]
    aH = logical_action(C, Hc)
    aP = logical_action(C, Pc)
    print(f"  fold-transversal F (SWAP pairs; F even-even, F^-1 odd-odd): {aH}")
    print(f"  fold-transversal P (P/P^-1 on fold qudits; CZ/CZ^-1 on pairs): {aP}")
    print(f"  group generated: order {len(gen_group([a for a in (aH, aP) if a]))}")
    # pair-site errors
    pairerr = np.vstack([all_on(n, list(pr)) for pr in pairs])
    ok = correctable(C, pairerr)
    print(f"  all single-pair 2-qutrit errors (fold-gate faults) correctable: {ok}")
    if not ok:
        # find a nontrivial logical supported on <=2 pair sites
        sites = [[t] for t in fixed] + [list(pr) for pr in pairs]
        found = None
        for A, B in itertools.combinations_with_replacement(range(len(sites)), 2):
            supp = sorted(set(sites[A]) | set(sites[B]))
            V = all_on(n, supp)
            bad = C.is_nontrivial_logical(V)
            if bad.any():
                found = (supp, fmt(V[bad][0], n))
                break
        print(f"     nontrivial logical on <=2 pair sites: {found}")
    gens = list(C.S)
    if D == 3:
        cnt, tot, sols, opts = order_search(C, gens, f"square D={D} single ancilla per check", max_report=1)
        if sols:
            hk = np.vstack([check_faults(C, g, o) for g, o in zip(gens, sols[0])])
            print(f"  joint: hooks (first safe order) + fold-pair errors correctable: "
                  f"{correctable(C, np.vstack([hk, pairerr]))}")
        # exhaustive pairwise test: if every pair of (check, order) fault sets is mutually
        # consistent, every one of the order assignments is hook-safe
        maps = [[o_d[1] for o_d in lst] for lst in opts]
        allpairs = all(merge_ok(da, db) for i in range(len(maps)) for j in range(i + 1, len(maps))
                       for da in maps[i] for db in maps[j])
        npairs = sum(len(maps[i]) * len(maps[j]) for i in range(len(maps)) for j in range(i + 1, len(maps)))
        pe = signature_map(C, np.vstack([singles(n), pairerr]))
        withfold = pe is not None and all(merge_ok(pe, dd) for lst in maps for dd in lst)
        print(f"  every pair of (check, order) choices consistent: {allpairs} ({npairs} pairs); "
              f"fold-pair errors consistent with every (check, order): {withfold}")
    return C, qpos


def css_exhaustive(n):
    section(f"7. Exhaustive search: does any CSS [[{n},1,3]]_3 exist?")
    W = np.vstack([np.eye(n, dtype=np.int64)[[i]] * a for i in range(n) for a in (1, 2)] +
                  [np.array([[a if t == i else (b if t == j else 0) for t in range(n)]])
                   for i, j in itertools.combinations(range(n), 2) for a in (1, 2) for b in (1, 2)])

    def subspaces(m, k):
        for piv in itertools.combinations(range(m), k):
            free = [(i, c) for i in range(k) for c in range(piv[i] + 1, m) if c not in piv]
            for vals in itertools.product(range(3), repeat=len(free)):
                M = np.zeros((k, m), dtype=np.int64)
                for i, pc in enumerate(piv):
                    M[i, pc] = 1
                for (i, c), v in zip(free, vals):
                    M[i, c] = v
                yield M

    found = 0
    tried = 0
    for kx in range(1, n - 1):
        kz = n - 1 - kx
        if kz < 1 or kx > kz:
            continue  # (kx,kz) and (kz,kx) related by X<->Z
        for HX in subspaces(n, kx):
            Dsp = nullspace(HX)            # (n-kx) x n, = ker HX = HX^perp
            lowD = W[np.all(mod(W @ HX.T) == 0, axis=1)]
            for Cc in subspaces(Dsp.shape[0], kz):
                tried += 1
                HZ = mod(Cc @ Dsp)
                if lowD.size and not np.all(in_rowspace_many(HZ, lowD)):
                    continue
                lowX = W[np.all(mod(W @ HZ.T) == 0, axis=1)]
                if lowX.size and not np.all(in_rowspace_many(HX, lowX)):
                    continue
                found += 1
                if found == 1:
                    print(f"  found example kx={kx}: HX={HX.tolist()} HZ={HZ.tolist()}")
    print(f"  (HX,HZ) pairs tried: {tried}; CSS [[{n},1,>=3]]_3 codes found: {found}")
    return found


def five_qutrit_perms():
    section("3b. [[5,1,3]]_3: permutation part of the monomial automorphisms")
    n = 5
    g0x = [1, 0, 0, -1, 0]
    g0z = [0, 1, -1, 0, 0]
    S = [np.concatenate([np.roll(g0x, s), np.roll(g0z, s)]) for s in range(4)]
    C = Code("513", S, n, np.array([1] * 5 + [0] * 5), np.array([0] * 5 + [1] * 5))
    res = monomial_automorphisms(C, list(itertools.permutations(range(n))), split=3)
    byperm = defaultdict(set)
    for perm, mi, act in res:
        byperm[perm].add((tuple(SL23[m] for m in mi), act))
    aff = {tuple((a * i + b) % 5 for i in range(5)) for a in range(1, 5) for b in range(5)}
    print(f"  permutations occurring: {len(byperm)}; all affine i->a*i+b mod 5: {set(byperm) <= aff}")
    for a in range(1, 5):
        perm = tuple((a * i) % 5 for i in range(5))
        pairs = sorted(set((x[0][0], x[1]) for x in byperm[perm] if len(set(x[0])) == 1))
        print(f"  i->{a}i: (uniform local map -> logical action): "
              + ", ".join(f"{l}->{act}" for l, act in pairs))
    words = {}
    gens = {"F": m2((0, 2, 1, 0)), "Fd": m2((0, 1, 2, 0)), "P": m2((1, 0, 1, 1)), "Pd": m2((1, 0, 2, 1))}
    for L in range(1, 5):
        for word in itertools.product(gens, repeat=L):
            M = np.eye(2, dtype=np.int64)
            for g in word:
                M = mod(gens[g] @ M)
            t = tuple(int(v) for v in M.reshape(-1))
            words.setdefault(t, word)
    for t in [(1, 2, 2, 2), (2, 1, 1, 1), (1, 1, 1, 2), (2, 2, 2, 1)]:
        print(f"  {t} = time-ordered gate word {words[t]}")


def anwar7():
    section("4b. Anwar-Campbell-Browne 2012 seven-qutrit code (signed Steane generators)")
    n = 7
    HX = [[0, 0, 0, -1, 1, 1, -1], [1, 0, -1, 0, -1, 0, 1], [0, 1, -1, 0, 0, -1, 1]]
    HZ = [[0, 0, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1]]
    C = css_from("anwar7", n, HX, HZ, [1, -1, 1, 1, -1, 1, -1], [1] * 7)
    print(f"  k={C.k} d={C.distance()[0]}")
    nm = {(0, 2, 1, 0): "F", (0, 1, 2, 0): "Fd", (1, 0, 1, 1): "P", (1, 0, 2, 1): "Pd", (1, 0, 0, 1): "I"}
    res = monomial_automorphisms(C, [tuple(range(n))], split=4)
    summarize_group("Anwar7 local automorphisms", [a for _, _, a in res])
    for perm, mi, act in res:
        if act in [(0, 2, 1, 0), (1, 0, 1, 1)]:
            print(f"  logical {nm[act]} <- local gates {[nm.get(SL23[m], SL23[m]) for m in mi]}")


def two_design():
    section("8. Qutrit Clifford subgroups: frame potential (unitary 2-design iff = 2)")
    w = np.exp(2j * np.pi / 3)
    Xm = np.roll(np.eye(3), 1, axis=0)
    Zm = np.diag([1, w, w ** 2])
    Fm = np.array([[w ** (j * k) for j in range(3)] for k in range(3)]) / np.sqrt(3)
    Pm = np.diag([w ** ((j * (j - 1) // 2) % 3) for j in range(3)])
    Wd = {(a, b): np.linalg.matrix_power(Xm, a) @ np.linalg.matrix_power(Zm, b) for a in range(3) for b in range(3)}

    def symp_of(U):
        out = []
        for G in (Xm, Zm):
            Cc = U @ G @ U.conj().T
            for (a, b), Wab in Wd.items():
                if (a, b) != (0, 0) and abs(abs(np.trace(Wab.conj().T @ Cc) / 3) - 1) < 1e-9:
                    out.append((a, b))
                    break
        (a1, b1), (a2, b2) = out
        return (a1, a2, b1, b2)

    def key(U):
        v = U.reshape(-1)
        i = int(np.argmax(np.abs(v) > 1e-9))
        return tuple(np.round(v / (v[i] / abs(v[i])), 6))

    G = {key(np.eye(3)): np.eye(3)}
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for U in frontier:
            for g in (Fm, Pm, Xm, Zm):
                V = g @ U
                k = key(V)
                if k not in G:
                    G[k] = V
                    new.append(V)
        frontier = new
    G = list(G.values())
    Q8 = {(1, 0, 0, 1), (2, 0, 0, 2), (0, 2, 1, 0), (0, 1, 2, 0), (1, 2, 2, 2), (2, 1, 1, 1), (1, 1, 1, 2), (2, 2, 2, 1)}
    Z4 = {(1, 0, 0, 1), (2, 0, 0, 2), (1, 2, 2, 2), (2, 1, 1, 1)}

    def fp(Sset):
        A = np.array(Sset)
        T = np.einsum("aij,bij->ab", A.conj(), A)
        return float(np.mean(np.abs(T) ** 4))

    print(f"  sdim-convention F symplectic {symp_of(Fm)}, P symplectic {symp_of(Pm)}")
    print(f"  full Clifford group mod phase: {len(G)} elements, frame potential {fp(G):.6f}")
    q8 = [U for U in G if symp_of(U) in Q8]
    z4 = [U for U in G if symp_of(U) in Z4]
    print(f"  Weyl x Q8: {len(q8)} elements, frame potential {fp(q8):.6f}")
    print(f"  Weyl x Z4: {len(z4)} elements, frame potential {fp(z4):.6f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["512", "chau", "513", "513p", "713", "a7", "913", "sq3", "css5", "css6", "design"]
    if "512" in which:
        brief_512()
    if "chau" in which:
        Cch = chau_code()
    if "513" in which:
        C5 = five_qutrit()
        if "chau" in which:
            eq = monomial_automorphisms(Cch, list(itertools.permutations(range(5))), split=3, target=C5)
            print(f"  Chau code -> cyclic code local-Clifford x permutation equivalences found: {len(eq)}")
    if "513p" in which:
        five_qutrit_perms()
    if "713" in which:
        color7()
    if "a7" in which:
        anwar7()
    if "913" in which:
        rotated9()
    if "sq3" in which:
        square13(3)
    if "sq4" in which:
        square13(4)
    if "css5" in which:
        css_exhaustive(5)
    if "css6" in which:
        css_exhaustive(6)
    if "design" in which:
        two_design()
    print(f"\nTotal time {time.time() - T0:.1f}s")
