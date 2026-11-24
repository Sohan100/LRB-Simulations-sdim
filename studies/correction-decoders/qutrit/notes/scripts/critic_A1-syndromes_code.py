#!/usr/bin/env python3
"""Critic check for A1-syndromes, code level (claims C1, C3-C11, C21).

Independent of A1_syndrome_classification.py: conventions are derived from
explicit 243 x 243 matrices (sdim's X, Z, H=DFT, P=diag(1,1,w), CNOT, CZ,
SWAP definitions), not from the symplectic frame rules. Counting uses exact
Fraction polynomials in q (not integer polynomials in u = q/8).

Run: module load python/3.11; python critic_A1-syndromes_code.py
"""
import itertools
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction as Fr

import numpy as np

W = np.exp(2j * np.pi / 3)
N = 5
DIM = 3 ** N
I3 = np.eye(3, dtype=complex)
X3 = np.zeros((3, 3), dtype=complex)
for j in range(3):
    X3[(j + 1) % 3, j] = 1            # X|j> = |j+1>  (sdim generate_x_matrix)
Z3 = np.diag([1, W, W * W])           # sdim generate_z_matrix
F3 = np.array([[W ** (m * n) for n in range(3)] for m in range(3)]) / np.sqrt(3)
P3 = np.diag([1, 1, W])               # sdim generate_p_matrix, d = 3


def kron_list(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def one(op, q):
    ops = [I3] * N
    ops[q] = op
    return kron_list(ops)


def two(fun, a, b):
    """Two-qutrit diagonal/permutation gate given by fun(ja, jb) ->
    (phase, ja', jb') on qutrits a, b of the 5-qutrit register."""
    U = np.zeros((DIM, DIM), dtype=complex)
    for idx in range(DIM):
        digs = [(idx // 3 ** (N - 1 - k)) % 3 for k in range(N)]
        ph, na, nb = fun(digs[a], digs[b])
        d2 = list(digs)
        d2[a], d2[b] = na, nb
        jdx = sum(d2[k] * 3 ** (N - 1 - k) for k in range(N))
        U[jdx, idx] = ph
    return U


GATES1 = {"H": F3, "H_INV": F3.conj().T, "P": P3, "P_INV": P3.conj().T,
          "X": X3, "X_INV": X3.conj().T, "Z": Z3, "Z_INV": Z3.conj().T,
          "I": I3}


def gate_unitary(name, a, b=None):
    if name in GATES1:
        return one(GATES1[name], a)
    if name == "CNOT":
        return two(lambda x, y: (1, x, (y + x) % 3), a, b)
    if name == "CNOT_INV":
        return two(lambda x, y: (1, x, (y - x) % 3), a, b)
    if name == "CZ":
        return two(lambda x, y: (W ** (x * y), x, y), a, b)
    if name == "CZ_INV":
        return two(lambda x, y: (W ** (-x * y), x, y), a, b)
    if name == "SWAP":
        return two(lambda x, y: (1, y, x), a, b)
    raise ValueError(name)


def pauli_matrix(x, z):
    """prod_q X_q^{x_q} Z_q^{z_q} (X written left of Z on each qutrit)."""
    ops = [np.linalg.matrix_power(X3, int(x[q]) % 3)
           @ np.linalg.matrix_power(Z3, int(z[q]) % 3) for q in range(N)]
    return kron_list(ops)


def comm_exp(A, B):
    """k with A B = w^k B A (A, B monomial unitaries that commute up to a
    third root of unity)."""
    M = A @ B
    Nn = B @ A
    i, j = np.unravel_index(np.argmax(np.abs(Nn)), Nn.shape)
    lam = M[i, j] / Nn[i, j]
    k = int(np.round(np.angle(lam) / (2 * np.pi / 3))) % 3
    assert np.allclose(M, (W ** k) * Nn)
    return k


def vec(xd=None, zd=None):
    x = np.zeros(N, dtype=int)
    z = np.zeros(N, dtype=int)
    for q, e in (xd or {}).items():
        x[q] = e % 3
    for q, e in (zd or {}).items():
        z[q] = e % 3
    return x, z


out = []


def say(*a):
    s = " ".join(str(t) for t in a)
    print(s, flush=True)
    out.append(s)


# ---------------------------------------------------------------- C1
say("=" * 70)
say("C1: commutation, logicals, encoder (matrix level)")
SZ1 = vec(zd={0: 1, 1: -1, 3: -1})
SZ2 = vec(zd={1: 1, 2: 1, 4: -1})
SX1 = vec(xd={0: 1, 1: 1, 2: -1})
SX2 = vec(xd={1: -1, 3: 1, 4: -1})
XL = vec(xd={0: 1, 3: 1})
ZL = vec(zd={0: 1, 2: 1})
names = ["SZ1", "SZ2", "SX1", "SX2", "XL", "ZL"]
mats = [pauli_matrix(*v) for v in (SZ1, SZ2, SX1, SX2, XL, ZL)]
tab = [[comm_exp(A, B) for B in mats] for A in mats]
for nm, row in zip(names, tab):
    say("  ", nm, row)
say("  Z X = w X Z on one qutrit:", comm_exp(Z3, X3) == 1)
# encoder statevector
psi = np.zeros(DIM, dtype=complex)
psi[0] = 1
enc = [("H_INV", 2), ("H_INV", 3), ("H_INV", 4), ("CNOT_INV", 2, 1),
       ("CNOT", 1, 0), ("CNOT", 3, 0), ("CNOT", 4, 2)]
for g in enc:
    psi = gate_unitary(*g) @ psi
for nm, Mt in zip(names, mats):
    say(f"  <psi|{nm}|psi> = {np.vdot(psi, Mt @ psi):.6f}")

# ---------------------------------------------------------------- syndrome
# functionals from matrix commutation of each stabilizer with unit Paulis
units = []
for q in range(N):
    units.append(("X", q, pauli_matrix(*vec(xd={q: 1}))))
for q in range(N):
    units.append(("Z", q, pauli_matrix(*vec(zd={q: 1}))))
FUN = np.array([[comm_exp(S, U) for (_, _, U) in units] for S in mats])
# rows SZ1,SZ2,SX1,SX2,XL,ZL ; cols x0..x4, z0..z4 ; value k with S E = w^k E S
say("syndrome functionals k(S,E) [S E = w^k E S], cols x0..x4 z0..z4:")
for nm, row in zip(names, FUN):
    say("  ", nm, row.tolist())
# logical labels: a = k(ZL, E), b = -k(XL, E)
SYNF = FUN[:4] % 3
AF = FUN[5] % 3
BF = (-FUN[4]) % 3

# ---------------------------------------------------------------- enumerate
allv = np.array(list(itertools.product(range(3), repeat=2 * N)), dtype=int)
xs, zs = allv[:, :N], allv[:, N:]
wt = np.count_nonzero((xs != 0) | (zs != 0), axis=1)
syn = (allv @ SYNF.T) % 3
sidx = syn @ np.array([1, 3, 9, 27])
aa = (allv @ AF) % 3
bb = (allv @ BF) % 3
cidx = aa + 3 * bb
say("check b(E) == z0+z3 for all E:", bool(np.all(bb == (zs[:, 0] + zs[:, 3]) % 3)))
say("check a(E) == x0+x2 for all E:", bool(np.all(aa == (xs[:, 0] + xs[:, 2]) % 3)))


def lab(i):
    v = allv[i]
    parts = []
    for q in range(N):
        x, z = v[q], v[N + q]
        s = ""
        if x:
            s += f"X{q}" + ("" if x == 1 else "^-1")
        if z:
            s += f"Z{q}" + ("" if z == 1 else "^-1")
        if s:
            parts.append(s)
    return " ".join(parts) or "I"


# ---------------------------------------------------------------- C3
say("=" * 70)
say("C3: stabilizer group, logical cosets")
stab = np.nonzero((sidx == 0) & (cidx == 0))[0]
say("  |stabilizer group| =", len(stab), " weight distribution",
    dict(sorted(Counter(wt[stab]).items())))
# cross-check the group by explicit generation from the four generators
gens = [np.concatenate(v) for v in (SZ1, SZ2, SX1, SX2)]
grp = {tuple((sum(c * g for c, g in zip(cs, gens))) % 3)
       for cs in itertools.product(range(3), repeat=4)}
say("  group generated by S (81?)", len(grp),
    " equal to commutant-with-logicals set:",
    grp == {tuple(allv[i]) for i in stab})
for c in range(1, 9):
    sel = np.nonzero((sidx == 0) & (cidx == c))[0]
    m = wt[sel].min()
    reps = [lab(i) for i in sel if wt[i] == m]
    say(f"  coset (a,b)=({c % 3},{c // 3}): min weight {m}, #reps {len(reps)}:",
        reps)
w2log = [lab(i) for i in np.nonzero((sidx == 0) & (cidx != 0) & (wt == 2))[0]]
say("  all weight-2 nontrivial logicals:", len(w2log), w2log)

# ---------------------------------------------------------------- C4/C5
say("=" * 70)
say("C4/C5: syndrome classification from weight<=2 errors")
cnt = np.zeros((81, 9, N + 1), dtype=int)
np.add.at(cnt, (sidx, cidx, wt), 1)
cls = {}
post = {}
for s in range(81):
    wmin = next(w for w in range(N + 1) if cnt[s, :, w].sum() > 0)
    lead = cnt[s, :, wmin]
    nz = lead[lead > 0]
    if s == 0:
        cls[s] = "T"
    elif len(nz) == 1:
        cls[s] = f"U{wmin}"
    elif len(set(nz.tolist())) == 1:
        cls[s] = f"A{wmin}"
    else:
        cls[s] = f"A{wmin}*"
    post[s] = Fr(int(nz.max()), int(nz.sum()))
say("  classes:", dict(Counter(cls.values())))
say("  leading posteriors:", dict(Counter(str(v) for v in post.values())))
w2syn = [s for s in range(81) if cls[s].startswith("A2")]
say("  #syndromes with wmin=2:", len(w2syn),
    "; any with unique coset at weight 2:",
    any(np.count_nonzero(cnt[s, :, 2]) == 1 for s in w2syn))
say("  A2 syndromes whose top count is unique (count-hybrid accepts):",
    sum(1 for s in w2syn if sorted(cnt[s, :, 2])[-1] > sorted(cnt[s, :, 2])[-2]))
w1 = np.nonzero(wt == 1)[0]
amb = [i for i in w1 if cls[int(sidx[i])] == "A1"]
uni = [i for i in w1 if cls[int(sidx[i])] == "U1"]
say("  weight-1 errors: U1", len(uni), " A1", len(amb))
pairs = defaultdict(list)
for i in amb:
    pairs[int(sidx[i])].append(lab(i))
say("  tied pairs:", sorted(tuple(sorted(v)) for v in pairs.values()))
by_q = Counter()
for i in uni:
    q = int(np.nonzero((xs[i] != 0) | (zs[i] != 0))[0][0])
    typ = "mixed" if (xs[i][q] and zs[i][q]) else ("pureX" if xs[i][q] else "pureZ")
    by_q[(q, typ)] += 1
say("  U1 weight-1 errors by (qutrit, type):", dict(sorted(by_q.items())))

# ---------------------------------------------------------------- C6..C10
say("=" * 70)
say("C6-C10: exact code capacity, Fraction polynomials in q")
DEG = N + 1


def pmul(a, b):
    r = [Fr(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] += x * y
    return r


def padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(n)]


def psc(c, a):
    return [c * x for x in a]


def pser_div(num, den, order=4):
    num = list(num) + [Fr(0)] * (order + 2)
    den = list(den) + [Fr(0)] * (order + 2)
    o = []
    for k in range(order + 1):
        s = num[k] - sum(o[j] * den[k - j] for j in range(k))
        o.append(s / den[0])
    return o


def fmt(p, n=3):
    t = []
    for k, c in enumerate(p):
        if c != 0:
            t.append(f"{c}" + ("" if k == 0 else (" q" if k == 1 else f" q^{k}")))
        if len(t) >= n:
            break
    return " + ".join(t) if t else "0"


def site_probs(kind):
    """per qutrit: dict k(=x+3z) -> probability polynomial coefficient of q
    (single-qutrit error probabilities are linear in q)."""
    dep = {k: Fr(1, 8) for k in range(1, 9)}
    px = {1: Fr(1, 2), 2: Fr(1, 2)}
    pz = {3: Fr(1, 2), 6: Fr(1, 2)}
    pxz = {1: Fr(1, 4), 2: Fr(1, 4), 3: Fr(1, 4), 6: Fr(1, 4)}
    mix = {4: Fr(1, 4), 5: Fr(1, 4), 7: Fr(1, 4), 8: Fr(1, 4)}
    if kind == "depolarizing":
        return [dep] * N
    if kind == "pure_X":
        return [px] * N
    if kind == "pure_Z":
        return [pz] * N
    if kind == "pure_X_or_Z":
        return [pxz] * N
    if kind == "mixed_only":
        return [mix] * N
    if kind == "centre_only":
        return [{}, dep, {}, {}, {}]
    if kind == "corners_only":
        return [dep, {}, dep, dep, dep]
    m = re.match(r"qutrit_(\d)_only", kind)
    j = int(m.group(1))
    return [dep if i == j else {} for i in range(N)]


# per-error single-qutrit index k_q = x_q + 3 z_q
kq = xs + 3 * zs


def table(kind, transform=None):
    """T[(s,c)] = probability polynomial (list of Fractions in q).

    P(E) = prod_{identity sites} (1 - t_q q) * prod_{error sites} c_{q,k} q.
    Aggregate the integer weights 8^5 prod c_{q,k} per (s, c, identity set)
    with numpy, then multiply by the 32 identity-set polynomials exactly."""
    sp = site_probs(kind)
    SCALE = 8
    cint = np.zeros((N, 9), dtype=np.int64)       # SCALE * c_{q,k}
    allowed = np.zeros((N, 9), dtype=bool)
    tq = []
    for q in range(N):
        allowed[q, 0] = True
        for k, c in sp[q].items():
            cint[q, k] = int(c * SCALE)
            assert Fr(cint[q, k], SCALE) == c
            allowed[q, k] = True
        tq.append(sum(sp[q].values(), Fr(0)))
    if transform is None:
        S_, C_ = sidx, cidx
    else:
        S_, C_ = transform
    ok = np.all(allowed[np.arange(N)[None, :], kq], axis=1)
    wint = np.ones(len(allv), dtype=np.int64)
    for q in range(N):
        wint *= np.where(kq[:, q] == 0, SCALE, cint[q, kq[:, q]])
    idmask = (kq == 0) @ (1 << np.arange(N))
    acc = np.zeros((81, 9, 32), dtype=np.int64)
    np.add.at(acc, (S_[ok], C_[ok], idmask[ok]), wint[ok])
    idpoly = {}
    for m in range(32):
        p = [Fr(1)]
        for q in range(N):
            if (m >> q) & 1:
                p = pmul(p, [Fr(1), -tq[q]])
        nerr = N - bin(m).count("1")
        idpoly[m] = [Fr(0)] * nerr + p          # times q^{#error sites}
    T = {}
    for s, c, m in zip(*np.nonzero(acc)):
        p = psc(Fr(int(acc[s, c, m]), SCALE ** N), idpoly[m])
        key = (int(s), int(c))
        T[key] = padd(T.get(key, [Fr(0)] * DEG), p)
    return T


def collapse(T, mode):
    """mode 'full': class c; 'obs': class b; 'xsyn': key (sX, b)."""
    U = defaultdict(lambda: defaultdict(lambda: [Fr(0)] * DEG))
    for (s, c), p in T.items():
        if mode == "full":
            U[s][c] = padd(U[s][c], p)
        elif mode == "obs":
            U[s][c // 3] = padd(U[s][c // 3], p)
        elif mode == "xsyn":
            sv = [(s // 3 ** j) % 3 for j in range(4)]
            U[sv[2] + 3 * sv[3]][c // 3] = padd(U[sv[2] + 3 * sv[3]][c // 3], p)
    return U


def lowest(p):
    for k, c in enumerate(p):
        if c != 0:
            return k
    return None


def lexmax(polys):
    """index of the coset with the largest probability for small q."""
    best = None
    for h, p in polys.items():
        if best is None:
            best = h
            continue
        pb = polys[best]
        for k in range(DEG):
            a, b = (p[k] if k < len(p) else 0), (pb[k] if k < len(pb) else 0)
            if a != b:
                if a > b:
                    best = h
                break
    return best


def strategies(U, zero_key=0, dec_tables=None):
    """detect / mld / hybrid strict. dec_tables: decoder's own tables (for
    the naive prior); default = true prior."""
    D = U if dec_tables is None else dec_tables
    res = {}
    for st in ("detect", "mld", "hyb"):
        acc = [Fr(0)] * DEG
        fail = [Fr(0)] * DEG
        for s, polys in U.items():
            tot = [Fr(0)] * DEG
            for p in polys.values():
                tot = padd(tot, p)
            if all(t == 0 for t in tot):
                continue
            dpol = D.get(s, {})
            if st == "detect":
                ok, dec = (s == zero_key), 0
            else:
                if dpol and any(any(c != 0 for c in p) for p in dpol.values()):
                    dec = lexmax(dpol)
                    lo = min(lowest(p) for p in dpol.values() if lowest(p) is not None)
                    uniq = sum(1 for p in dpol.values() if lowest(p) == lo) == 1
                else:            # decoder never expects this syndrome
                    dec, uniq = 0, False
                ok = True if st == "mld" else (uniq or s == zero_key)
                if s == zero_key and st == "hyb":
                    ok = True
            if not ok:
                continue
            acc = padd(acc, tot)
            fail = padd(fail, padd(tot, psc(-1, polys.get(dec, [Fr(0)] * DEG))))
        rej = padd([Fr(1)], psc(-1, acc))
        res[st] = (rej, pser_div(fail, acc), acc, fail)
    return res


def report(kind, modes=("full", "obs", "xsyn")):
    T = table(kind)
    for mode in modes:
        U = collapse(T, mode)
        r = strategies(U)
        say(f"  {kind:14s} {mode:5s} | det rej {fmt(r['detect'][0])}; "
            f"f|a {fmt(r['detect'][1])} | MLD fail {fmt(r['mld'][1])} | "
            f"hyb rej {fmt(r['hyb'][0])}; f|a {fmt(r['hyb'][1])}")
    return T


Tdep = report("depolarizing")
for kind in ("pure_X", "pure_Z", "pure_X_or_Z", "mixed_only", "centre_only",
             "corners_only", "qutrit_0_only", "qutrit_3_only"):
    report(kind, modes=("full", "obs"))

# numbers at q = 1e-3, 1e-2, 0.05, 0.1 (depolarizing, full class)
U = collapse(Tdep, "full")
r = strategies(U)


def ev(p, q):
    return float(sum(float(c) * q ** k for k, c in enumerate(p)))


for q in (1e-3, 1e-2, 0.05, 0.1):
    pa_d, pf_d = ev(r["detect"][2], q), ev(r["detect"][3], q)
    pa_h, pf_h = ev(r["hyb"][2], q), ev(r["hyb"][3], q)
    say(f"  q={q}: detect P_acc {pa_d:.5f} P(f|a) {pf_d / pa_d:.3e} | hybrid "
        f"P_acc {pa_h:.5f} P(f|a) {pf_h / pa_h:.3e} | MLD {ev(r['mld'][3], q):.4e}")
# exact argmax vs lexicographic choice
for q in (1e-3, 1e-2, 0.05, 0.1):
    nd = 0
    for s, polys in U.items():
        vals = {h: ev(p, q) for h, p in polys.items()}
        b = max(vals, key=vals.get)
        if abs(vals[b] - vals[lexmax(polys)]) > 1e-15:
            nd += 1
    say(f"  q={q}: syndromes where exact argmax != small-q choice: {nd}")

# ---------------------------------------------------------------- C8
say("=" * 70)
say("C8: posterior-threshold family, depolarizing, full class")
for q in (1e-3, 1e-2, 5e-2):
    rows = []
    for s, polys in U.items():
        vals = np.array([ev(p, q) for p in polys.values()])
        rows.append((vals.max() / vals.sum(), vals.sum(), vals.sum() - vals.max()))
    rows.sort(key=lambda t: -t[0])
    acc = fail = 0.0
    lev = []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and abs(rows[j][0] - rows[i][0]) < 1e-12:
            acc += rows[j][1]
            fail += rows[j][2]
            j += 1
        lev.append((round(rows[i][0], 4), j - i, round(acc, 5), f"{fail / acc:.3e}"))
        i = j
    say(f"  q={q}: (posterior, #syndromes at level, cum P_acc, cum P(f|a))")
    for t in lev:
        say("     ", t)

# ---------------------------------------------------------------- C11
say("=" * 70)
say("C11: logical gates from unitaries")
LOG = {"H": [("SWAP", 2, 3), ("H", 0), ("H", 2), ("H", 3), ("H", 4), ("H_INV", 1)],
       "P": [("P", 0), ("P", 4), ("P_INV", 1), ("CZ", 2, 3)]}
INV = {"H": "H_INV", "H_INV": "H", "P": "P_INV", "P_INV": "P", "CZ": "CZ_INV",
       "CZ_INV": "CZ", "SWAP": "SWAP", "CNOT": "CNOT_INV", "CNOT_INV": "CNOT"}
# the repository reverses the GROUP order and inverts each gate; inside a
# group the qudit order is kept. Groups: H: [SWAP], [H 0,2,3,4], [H_INV 1];
# P: [P 0,4], [P_INV 1], [CZ 2 3].
GROUPS = {"H": [[("SWAP", 2, 3)], [("H", 0), ("H", 2), ("H", 3), ("H", 4)],
                [("H_INV", 1)]],
          "P": [[("P", 0), ("P", 4)], [("P_INV", 1)], [("CZ", 2, 3)]],
          "X": [[("X", 0), ("X", 3)]], "Z": [[("Z", 0), ("Z", 2)]],
          "I": [[("I", 0)]]}


def token_ops(tok):
    base = tok[0]
    groups = GROUPS[base]
    if len(tok) > 1:
        groups = [[(INV[g[0]],) + g[1:] for g in grp] for grp in reversed(groups)]
    return [g for grp in groups for g in grp]


def token_unitary(tok):
    U = np.eye(DIM, dtype=complex)
    for g in token_ops(tok):
        U = gate_unitary(*g) @ U
    return U


# Pauli candidates supported on <= 2 qutrits, for identifying images
cand = np.nonzero(wt <= 2)[0]
cand_mats = {}


def identify(V):
    """Return index i (into allv) with V = phase * Pauli(allv[i])."""
    # V is monomial: find its permutation part and diagonal phases
    for i in cand:
        if i not in cand_mats:
            cand_mats[i] = pauli_matrix(allv[i][:N], allv[i][N:])
        Pm = cand_mats[i]
        t = np.vdot(Pm, V)
        if abs(abs(t) - DIM) < 1e-6:
            return i
    return None


def symplectic_from_unitary(Uu):
    cols = []
    for (_, q, Um) in units:
        img = Uu @ Um @ Uu.conj().T
        i = identify(img)
        cols.append(allv[i])
    return np.array(cols).T % 3      # 10 x 10, column = image of unit


gate_G = {}
for tok in ("H", "H_INV", "P", "P_INV"):
    Uu = token_unitary(tok)
    G = symplectic_from_unitary(Uu)
    gate_G[tok] = G
    # images of stabilizers / logicals
    def img_lab(v):
        w_ = (G @ v) % 3
        i = int(np.nonzero(np.all(allv == w_, axis=1))[0][0])
        return i, lab(i)
    ims = {nm: img_lab(np.concatenate(v))[1] for nm, v in
           zip(names, (SZ1, SZ2, SX1, SX2, XL, ZL))}
    # syndrome map M and logical map on (a,b)
    Ssyn = SYNF
    # sigma(G E) = SYNF @ G @ E ; find M with SYNF G = M SYNF (+ K on a,b)
    rows = (SYNF @ G) % 3
    A6 = np.vstack([SYNF, AF, BF]) % 3
    cands6 = np.array(list(itertools.product(range(3), repeat=6)))
    img6 = (cands6 @ A6) % 3
    def solve(row):
        h = np.nonzero(np.all(img6 == row % 3, axis=1))[0]
        assert len(h) == 1
        return cands6[h[0]]
    Mrows = np.array([solve(r_) for r_ in rows])
    Lrows = np.array([solve(r_) for r_ in ((np.vstack([AF, BF]) @ G) % 3)])
    stab_ok = all(np.all((SYNF @ ((G @ np.concatenate(v)) % 3)) % 3 == 0) and
                  ((AF @ ((G @ np.concatenate(v)) % 3)) % 3 == 0) and
                  ((BF @ ((G @ np.concatenate(v)) % 3)) % 3 == 0)
                  for v in (SZ1, SZ2, SX1, SX2))
    say(f"  {tok}: ops {token_ops(tok)}")
    say(f"     maps S to S: {stab_ok}; images {ims}")
    say(f"     sigma' rows (coeffs on sigma_Z1,Z2,X1,X2 | a,b): {Mrows.tolist()}")
    say(f"     (a',b') rows (coeffs on sigma | a,b): {Lrows.tolist()}")
    # weight-1 image census
    imgs = (allv[w1] @ G.T) % 3
    wimg = np.count_nonzero((imgs[:, :N] != 0) | (imgs[:, N:] != 0), axis=1)
    sim = (imgs @ SYNF.T) % 3 @ np.array([1, 3, 9, 27])
    census = Counter((cls[int(sidx[i])], int(wimg[k]), cls[int(sim[k])])
                     for k, i in enumerate(w1))
    say("     weight-1 census (class before, weight after, syndrome class after):",
        dict(sorted(census.items())))
    # exact decoding statistics, noise before the gate, decoding after it
    V2 = (allv @ G.T) % 3
    S2 = (V2 @ SYNF.T) % 3 @ np.array([1, 3, 9, 27])
    C2 = (V2 @ AF) % 3 + 3 * ((V2 @ BF) % 3)
    T2 = table("depolarizing", transform=(S2, C2))
    U2 = collapse(T2, "full")
    r_true = strategies(U2)
    r_naive = strategies(U2, dec_tables=U)
    say(f"     true prior : MLD {fmt(r_true['mld'][1])} | hyb rej "
        f"{fmt(r_true['hyb'][0])} f|a {fmt(r_true['hyb'][1])}")
    say(f"     naive prior: MLD {fmt(r_naive['mld'][1])} | hyb rej "
        f"{fmt(r_naive['hyb'][0])} f|a {fmt(r_naive['hyb'][1])}")

# ---------------------------------------------------------------- C21
say("=" * 70)
say("C21: Clifford catalog, physical-symbol and folded-code logical level")
here = os.path.dirname(os.path.abspath(__file__))
wt_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
txt = open(os.path.join(wt_root, "src", "lrb", "code_definitions.py")).read()
blk = txt[txt.index("single_qutrit_cliffords_exhaustive_with_S_strings = ["):]
blk = blk[:blk.index("]")]
strs = [s.replace("S", "P") for s in re.findall(r'"([A-Z]+)"', blk)]
say("  #strings", len(strs), " #distinct strings", len(set(strs)))


def norm_key(U):
    flat = U.ravel()
    k = np.flatnonzero(np.abs(flat) > 1e-9)[0]
    V = U / (flat[k] / abs(flat[k]))
    return tuple(np.round(V.ravel(), 6))


# (i) single-qutrit symbols, execution order = reversed string
M1 = {"H": F3, "P": P3, "X": X3, "Z": Z3, "I": I3}
keys_phys = []
for s in strs:
    U1 = I3.copy()
    for ch in reversed(s):          # execution order (rightmost acts first)
        U1 = M1[ch] @ U1
    keys_phys.append(norm_key(U1))
mult = Counter(Counter(keys_phys).values())
say("  physical-symbol unitaries: distinct", len(set(keys_phys)),
    " multiplicities", dict(mult))
# (ii) logical action of the folded-code circuits on the code space
basis = np.stack([np.linalg.matrix_power(pauli_matrix(*ZL), k) @ psi
                  for k in range(3)], axis=1)          # 243 x 3, orthonormal
say("  code basis orthonormal:", np.allclose(basis.conj().T @ basis, np.eye(3)))
TOKU = {t: token_unitary(t) for t in ("H", "P", "X", "Z", "I")}
TOKL = {}
for t, Uu in TOKU.items():
    L = basis.conj().T @ Uu @ basis
    assert np.allclose(L.conj().T @ L, np.eye(3), atol=1e-9), t
    TOKL[t] = L
keys_log = []
for s in strs:
    L = np.eye(3, dtype=complex)
    for ch in reversed(s):
        L = TOKL[ch] @ L
    keys_log.append(norm_key(L))
multL = Counter(Counter(keys_log).values())
say("  folded-code logical unitaries: distinct", len(set(keys_log)),
    " multiplicities", dict(multL))
# group generated by the logical tokens
grpL = {norm_key(np.eye(3, dtype=complex))}
front = [np.eye(3, dtype=complex)]
while front:
    nw = []
    for Uu in front:
        for t in ("H", "P", "X", "Z"):
            V = TOKL[t] @ Uu
            k = norm_key(V)
            if k not in grpL:
                grpL.add(k)
                nw.append(V)
    front = nw
say("  |<H_L,P_L,X_L,Z_L>| mod phase:", len(grpL),
    " never sampled:", len(grpL - set(keys_log)))
# action on logical Paulis (mod phase) of each catalog element: uniformity
# of the image of the logical Z (relevant for the 3/4 argument, C23)
XLm, ZLm = (basis.conj().T @ pauli_matrix(*XL) @ basis,
            basis.conj().T @ pauli_matrix(*ZL) @ basis)
lp = {}
for a in range(3):
    for b in range(3):
        lp[(a, b)] = np.linalg.matrix_power(XLm, a) @ np.linalg.matrix_power(ZLm, b)


def which_logical(V):
    for k_, Pm in lp.items():
        if abs(abs(np.vdot(Pm, V)) - 3) < 1e-6:
            return k_
    return None


imgZ = Counter()
for s in strs:
    L = np.eye(3, dtype=complex)
    for ch in reversed(s):
        L = TOKL[ch] @ L
    for k_ in [(0, 1), (1, 0), (1, 1)]:
        imgZ[(k_, which_logical(L @ lp[k_] @ L.conj().T))] += 1
for k_ in [(0, 1), (1, 0), (1, 1)]:
    d = {kk[1]: v for kk, v in imgZ.items() if kk[0] == k_}
    nZ = sum(v for (a_, b_), v in d.items() if b_ != 0)
    say(f"  image of logical X^{k_[0]}Z^{k_[1]} over the 216 strings:",
        dict(sorted(d.items())), f"-> fraction with Z part {nZ}/216 = {nZ / 216:.4f}")

with open(os.path.join(here, "critic_A1-syndromes_code_output.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
