#!/usr/bin/env python3
"""Critic check (round 2) of A1-syndromes claims C1-C11, C21, C23.

Independent of A1_syndrome_classification.py: own Pauli algebra, own
statevector simulation with the explicit sdim gate matrices (sdim/unitary.py:
X|j>=|j+1>, Z=diag(w^j), H = Fourier with H[m,n]=w^{mn}/sqrt3, P=diag(1,1,w),
CNOT|a,b>=|a,a+b>), own enumeration and own exact series arithmetic.

Run:  module load python/3.11; python critic_A1-syndromes_r2_code.py
Runtime: about 1 minute on one core.
"""
import itertools
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

W = np.exp(2j * np.pi / 3)
REPO = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/"
        "correction-decoders-qutrit-2c455b")

# ---------------------------------------------------------------- matrices
X1 = np.zeros((3, 3), complex)
for j in range(3):
    X1[(j + 1) % 3, j] = 1
Z1 = np.diag([W ** j for j in range(3)])
H1 = np.array([[W ** (m * n) for n in range(3)] for m in range(3)]) / np.sqrt(3)
P1 = np.diag([1, 1, W])
assert np.allclose(H1 @ X1 @ H1.conj().T, Z1)
assert np.allclose(H1 @ Z1 @ H1.conj().T, X1.conj().T)      # X^-1
ONEQ = {"H": H1, "H_INV": H1.conj().T, "P": P1, "P_INV": P1.conj().T,
        "X": X1, "X_INV": X1.conj().T, "Z": Z1, "Z_INV": Z1.conj().T,
        "I": np.eye(3)}


def ap1(psi, U, q):
    psi = np.tensordot(U, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q)


def ap2(psi, g, c, t):
    n = psi.ndim
    if g == "SWAP":
        return np.swapaxes(psi, c, t)
    idx = np.indices(psi.shape)
    a, b = idx[c], idx[t]
    if g in ("CNOT", "CNOT_INV"):
        s = 1 if g == "CNOT" else -1
        # |a,b> -> |a, b + s a> : new[a, b'] = old[a, b' - s a]
        src = list(idx)
        src[t] = (b - s * a) % 3
        return psi[tuple(src)]
    if g in ("CZ", "CZ_INV"):
        s = 1 if g == "CZ" else -1
        return psi * W ** (s * a * b % 3)
    raise ValueError(g)


def apply(psi, ops):
    for o in ops:
        if len(o) == 2:
            psi = ap1(psi, ONEQ[o[0]], o[1])
        else:
            psi = ap2(psi, o[0], o[1], o[2])
    return psi


def pauli_op(psi, x, z):
    """Apply X^x Z^z (Z first) on each qutrit (x, z dicts or lists)."""
    for q in range(len(x)):
        if z[q] % 3:
            psi = ap1(psi, np.linalg.matrix_power(Z1, z[q] % 3), q)
        if x[q] % 3:
            psi = ap1(psi, np.linalg.matrix_power(X1, x[q] % 3), q)
    return psi


ENC = [("H_INV", 2), ("H_INV", 3), ("H_INV", 4), ("CNOT_INV", 2, 1),
       ("CNOT", 1, 0), ("CNOT", 3, 0), ("CNOT", 4, 2)]
ZCHK = [("CNOT", 0, 5), ("CNOT_INV", 1, 5), ("CNOT_INV", 3, 5),
        ("CNOT", 1, 6), ("CNOT", 2, 6), ("CNOT_INV", 4, 6)]
XCHK = [("H", 7), ("CNOT", 7, 0), ("CNOT", 7, 1), ("CNOT_INV", 7, 2), ("H_INV", 7),
        ("H", 8), ("CNOT", 8, 3), ("CNOT_INV", 8, 1), ("CNOT_INV", 8, 4), ("H_INV", 8)]

out = {}


def section(t):
    print("=" * 70)
    print(t)


# ======================================================== C1: statevector
section("C1/C2 statevector checks (9 qutrits, explicit sdim matrices)")
psi0 = np.zeros((3,) * 9, complex)
psi0[(0,) * 9] = 1
enc = apply(psi0, ENC)
data = enc[(Ellipsis,) + (0,) * 4]                    # 5-qutrit data state
data5 = data.copy()


def expval(psi5, x, z):
    return np.vdot(psi5, pauli_op(psi5, x, z))


stabs = {"S_Z1": ([0] * 5, [1, -1, 0, -1, 0]), "S_Z2": ([0] * 5, [0, 1, 1, 0, -1]),
         "S_X1": ([1, 1, -1, 0, 0], [0] * 5), "S_X2": ([0, -1, 0, 1, -1], [0] * 5),
         "X_L": ([1, 0, 0, 1, 0], [0] * 5)}
for k, (x, z) in stabs.items():
    print(f"  <{k}> on encoded state = {np.round(expval(data5, x, z), 10)}")
zl = expval(data5, [0] * 5, [1, 0, 1, 0, 0])
print("  <Z_L> =", np.round(zl, 10), "(should be 0 for |+>_L)")


def ancilla_dist(psi, anc):
    p = np.abs(psi) ** 2
    axes = tuple(i for i in range(9) if i != anc)
    return p.sum(axis=axes)


def readouts(psi):
    """Deterministic outcomes of ancillas 5..8 after the check round, the
    terminal x0+x3 and the const0 detectors."""
    s = apply(psi, ZCHK + XCHK)
    anc = []
    for a in (5, 6, 7, 8):
        d = ancilla_dist(s, a)
        assert np.isclose(d.max(), 1.0), (a, d)
        anc.append(int(np.argmax(d)))
    # terminal (ordinary): H_INV on 0,3 then measure; distribution of m0+m3
    t = apply(psi, [("H_INV", 0), ("H_INV", 3)])
    p = np.abs(t) ** 2
    p03 = p.sum(axis=(1, 2, 4, 5, 6, 7, 8))
    dist = np.zeros(3)
    for i in range(3):
        for j in range(3):
            dist[(i + j) % 3] += p03[i, j]
    assert np.isclose(dist.max(), 1.0), dist
    obs = int(np.argmax(dist))
    # const0 terminal: H_INV on all data, outcome distributions of detectors
    t = apply(psi, [("H_INV", q) for q in range(5)])
    p = np.abs(t) ** 2
    pd = p.sum(axis=(5, 6, 7, 8))
    det = np.zeros((3, 3))
    lo = np.zeros(3)
    for v in itertools.product(range(3), repeat=5):
        if pd[v] < 1e-12:
            continue
        d1 = (v[0] + v[1] - v[2]) % 3
        d2 = (-v[1] + v[3] - v[4]) % 3
        det[d1, d2] += pd[v]
        lo[(v[0] + v[3]) % 3] += pd[v]
    assert np.isclose(det.max(), 1.0) and np.isclose(lo.max(), 1.0)
    d = np.unravel_index(np.argmax(det), det.shape)
    return tuple(anc), obs, (int(d[0]), int(d[1])), int(np.argmax(lo))


ref = readouts(enc)
print("  noiseless outcomes (anc 5..8, obs, const0 dets, const0 obs):", ref)
shift_unit = {}
for kind in ("X", "Z"):
    for q in range(5):
        x = [0] * 5
        z = [0] * 5
        (x if kind == "X" else z)[q] = 1
        psi = pauli_op(enc, x + [0] * 4, z + [0] * 4)
        r = readouts(psi)
        sh = tuple((a - b) % 3 for a, b in zip(r[0], ref[0]))
        ob = (r[1] - ref[1]) % 3
        c0 = tuple((a - b) % 3 for a, b in zip(r[2], ref[2]))
        c0o = (r[3] - ref[3]) % 3
        shift_unit[(kind, q)] = (sh, ob, c0, c0o)
        print(f"  {kind}{q}: ancilla shifts {sh}  obs {ob}  const0 dets {c0} const0 obs {c0o}")

# syndrome functional derived from the statevector (columns = x0..x4, z0..z4)
F_sv = np.zeros((4, 10), int)
B_sv = np.zeros(10, int)
C0_sv = np.zeros((2, 10), int)
for q in range(5):
    F_sv[:, q] = shift_unit[("X", q)][0]
    F_sv[:, 5 + q] = shift_unit[("Z", q)][0]
    B_sv[q] = shift_unit[("X", q)][1]
    B_sv[5 + q] = shift_unit[("Z", q)][1]
    C0_sv[:, q] = shift_unit[("X", q)][2]
    C0_sv[:, 5 + q] = shift_unit[("Z", q)][2]


def symp(a, b):
    """<A,B> = z_A.x_B - x_A.z_B, so that A B = w^<A,B> B A."""
    return int((np.dot(a[5:], b[:5]) - np.dot(a[:5], b[5:])) % 3)


def vec(x=(), z=()):
    v = np.zeros(10, int)
    for q, e in x:
        v[q] = e % 3
    for q, e in z:
        v[5 + q] = e % 3
    return v


S = [vec(z=[(0, 1), (1, -1), (3, -1)]), vec(z=[(1, 1), (2, 1), (4, -1)]),
     vec(x=[(0, 1), (1, 1), (2, -1)]), vec(x=[(1, -1), (3, 1), (4, -1)])]
XL = vec(x=[(0, 1), (3, 1)])
ZL = vec(z=[(0, 1), (2, 1)])
# commutation phase check of the symplectic convention against matrices
A = np.kron(X1, np.eye(3))
B = np.kron(Z1, np.eye(3))
assert np.allclose(B @ A, W * A @ B)     # Z X = w X Z  -> <Z,X> = 1
F_alg = np.array([[symp(s, np.eye(10, dtype=int)[i]) for i in range(10)] for s in S])
B_alg = np.array([(-symp(XL, np.eye(10, dtype=int)[i])) % 3 for i in range(10)])
print("  statevector syndrome functional == <S,E>:", np.array_equal(F_sv % 3, F_alg % 3))
print("  statevector observable shift == -<X_L,E> = z0+z3:", np.array_equal(B_sv % 3, B_alg))
print("  const0 detectors == -sigma_X:", np.array_equal(C0_sv % 3, (-F_alg[2:]) % 3))
print("  S_Z1 functional", F_sv[0].tolist(), " S_X1 functional", F_sv[2].tolist())
tab = [[symp(a, b) for b in S + [XL, ZL]] for a in S + [XL, ZL]]
print("  symplectic table:", tab)


def rank3(M):
    M = np.array(M, int) % 3
    r = 0
    for c in range(M.shape[1]):
        piv = [i for i in range(r, M.shape[0]) if M[i, c]]
        if not piv:
            continue
        M[[r, piv[0]]] = M[[piv[0], r]]
        M[r] = (M[r] * (1 if M[r, c] == 1 else 2)) % 3
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % 3
        r += 1
    return r


print("  rank(S) =", rank3(S), " rank(S,XL,ZL) =", rank3(S + [XL, ZL]))

# ======================================================== C3-C5 enumeration
section("C3-C5 enumeration of all 9^5 data Paulis")
KS = np.array(list(itertools.product(range(9), repeat=5)))
V = np.concatenate([KS % 3, KS // 3], axis=1)         # k = x + 3 z
SYN = (V @ F_alg.T) % 3
AA = (V @ np.array([symp(ZL, np.eye(10, dtype=int)[i]) for i in range(10)])) % 3
BB = (V @ B_alg) % 3
WT = np.count_nonzero(KS, axis=1)
sidx = SYN @ np.array([1, 3, 9, 27])
cidx = AA + 3 * BB
norm = (sidx == 0)
stab = norm & (cidx == 0)
print("  normalizer size", norm.sum(), " stabilizer group size", stab.sum())
print("  stabilizer weight distribution", dict(sorted(Counter(WT[stab]).items())))
w2log = np.nonzero(norm & (cidx != 0) & (WT == 2))[0]


def lab(k):
    s = []
    for q in range(5):
        x, z = k[q] % 3, k[q] // 3
        if x or z:
            s.append((f"X{q}" + ("^-1" if x == 2 else "") if x else "") +
                     (f"Z{q}" + ("^-1" if z == 2 else "") if z else ""))
    return " ".join(s) or "I"


print("  weight-2 nontrivial logicals:", [lab(KS[i]) for i in w2log])
for c in range(1, 9):
    sel = norm & (cidx == c)
    wm = WT[sel].min()
    print(f"  coset a={c % 3} b={c // 3}: min weight {wm}, #reps {np.sum(sel & (WT == wm))}")

# syndrome classes
cnt = np.zeros((81, 9, 6), int)
np.add.at(cnt, (sidx, cidx, WT), 1)
classes = Counter()
post = Counter()
cls_of = {}
for s in range(81):
    wmin = next(w for w in range(6) if cnt[s, :, w].sum())
    lead = cnt[s, :, wmin]
    ncos = np.count_nonzero(lead)
    if wmin == 0:
        c = "T"
    elif ncos == 1:
        c = f"U{wmin}"
    elif len(set(lead[lead > 0])) == 1:
        c = f"A{wmin}"
    else:
        c = f"A{wmin}*"
    cls_of[s] = c
    classes[c] += 1
    post[(c, str(Fraction(int(lead.max()), int(lead.sum()))))] += 1
print("  syndrome classes:", dict(classes))
print("  leading posteriors by class:", dict(post))
uniq_max_w2 = sum(1 for s in range(81) if cls_of[s] == "A2*" and
                  np.sum(cnt[s, :, 2] == cnt[s, :, 2].max()) == 1)
print("  A2* syndromes with a unique max coset at weight 2:", uniq_max_w2)
# single-qutrit errors
w1 = np.nonzero(WT == 1)[0]
percls = Counter()
pairs = set()
for i in w1:
    s = sidx[i]
    q = int(np.nonzero(KS[i])[0][0])
    k = KS[i][q]
    typ = "pureX" if k in (1, 2) else ("pureZ" if k in (3, 6) else "mixed")
    percls[(q, typ, cls_of[s])] += 1
    if cls_of[s] == "A1":
        partners = [lab(KS[j]) for j in w1 if sidx[j] == s and j != i]
        pairs.add(tuple(sorted([lab(KS[i])] + partners)))
print("  per qutrit/type/class:", dict(sorted(percls.items())))
print("  tied pairs:", sorted(pairs))

# ======================================================== exact polynomials
section("C6-C10 exact code-capacity series (own implementation)")


def table(wq, D, sidx=sidx, cidx=cidx, ns=81, nc=9):
    """P[s,c] as integer polynomials in u=q/D (length 6)."""
    coef = np.zeros((len(KS), 6), dtype=object)
    coef[:, 0] = 1
    for q in range(5):
        Wq = sum(wq[q][1:])
        newc = np.zeros_like(coef)
        for k in range(9):
            m = KS[:, q] == k
            if k == 0:
                newc[m, :] = coef[m, :]
                newc[m, 1:] += -Wq * coef[m, :-1]
            else:
                newc[m, 1:] = wq[q][k] * coef[m, :-1]
        coef = newc
    P = np.zeros((ns, nc, 6), dtype=object)
    for i in range(len(KS)):
        P[sidx[i], cidx[i]] += coef[i]
    return P


def to_q(p, D):
    return [Fraction(int(c)) / Fraction(D) ** k for k, c in enumerate(p)]


def series_div(num, den, n=5):
    num = list(num) + [Fraction(0)] * n
    den = list(den) + [Fraction(0)] * n
    out = []
    for k in range(n):
        out.append((num[k] - sum(out[j] * den[k - j] for j in range(k))) / den[0])
    return out


def lexgt(a, b):
    for x, y in zip(a, b):
        if x != y:
            return x > y
    return False


def strategies(P, D):
    G, H, _ = P.shape
    res = {}
    for st in ("detect", "mld", "hyb_strict", "hyb_count"):
        acc = [Fraction(0)] * 6
        fail = [Fraction(0)] * 6
        for g in range(G):
            polys = [to_q(P[g, h], D) for h in range(H)]
            if all(all(c == 0 for c in p) for p in polys):
                continue
            orders = [next((k for k, c in enumerate(p) if c != 0), None) for p in polys]
            m = min(o for o in orders if o is not None)
            lead = {h: polys[h][m] for h in range(H) if orders[h] == m}
            best = 0
            for h in range(H):
                if lexgt(polys[h], polys[best]):
                    best = h
            if st == "detect":
                ok, dec = (g == 0), 0
            elif st == "mld":
                ok, dec = True, best
            elif st == "hyb_strict":
                ok, dec = len(lead) == 1, best
            else:
                mx = max(lead.values())
                ok, dec = sum(1 for v in lead.values() if v == mx) == 1, best
            if not ok:
                continue
            tot = [sum(p[k] for p in polys) for k in range(6)]
            acc = [a + t for a, t in zip(acc, tot)]
            fail = [f + t - polys[dec][k] for k, (f, t) in enumerate(zip(fail, tot))]
        rej = [-a for a in acc]
        rej[0] += 1
        res[st] = dict(rej=rej[:4], fail_given_acc=series_div(fail, acc, 4))
    return res


def fmt(c, n=3):
    t = []
    for k, x in enumerate(c[:n + 1]):
        if x:
            t.append(f"{x}q^{k}")
    return " + ".join(t) or "0"


wdep = [[0] + [1] * 8] * 5
P_dep = table(wdep, 8)
R = strategies(P_dep, 8)
for st, v in R.items():
    print(f"  depolarizing {st:10s} P_rej = {fmt(v['rej'])};  P(f|a) = {fmt(v['fail_given_acc'])}")

# observable-only (class b) and X-syndrome + b
Pb = np.zeros((81, 3, 6), dtype=object)
for c in range(9):
    Pb[:, c // 3] += P_dep[:, c]
Rb = strategies(Pb, 8)
for st, v in Rb.items():
    print(f"  obs-only     {st:10s} P_rej = {fmt(v['rej'])};  P(f|a) = {fmt(v['fail_given_acc'])}")
Px = np.zeros((9, 3, 6), dtype=object)
for s in range(81):
    d = [(s // 3 ** j) % 3 for j in range(4)]
    Px[d[2] + 3 * d[3]] += Pb[s]
Rx = strategies(Px, 8)
for st, v in Rx.items():
    print(f"  xsyn+obs     {st:10s} P_rej = {fmt(v['rej'])};  P(f|a) = {fmt(v['fail_given_acc'])}")

# numeric table at q values
print("  numeric (detect P_acc, P(f|a); hybrid P_acc, P(f|a); MLD P_fail):")
for q in (1e-3, 1e-2, 0.05, 0.1):
    u = q / 8
    vals = np.array([[sum(float(c) * u ** k for k, c in enumerate(P_dep[s, c_]))
                      for c_ in range(9)] for s in range(81)])
    tot = vals.sum(1)
    dacc = tot[0]
    dfa = (tot[0] - vals[0, 0]) / dacc
    hy = [s for s in range(81) if cls_of[s] in ("T", "U1")]
    hacc = tot[hy].sum()
    hf = sum(tot[s] - vals[s].max() for s in hy) / hacc
    mld = sum(tot[s] - vals[s].max() for s in range(81))
    print(f"   q={q}: detect {dacc:.5f} {dfa:.3e} | hybrid {hacc:.5f} {hf:.3e} | MLD {mld:.4e}")

# biased/restricted noise
kinds = {
    "pure_X": ([[0, 1, 1, 0, 0, 0, 0, 0, 0]] * 5, 2),
    "pure_Z": ([[0, 0, 0, 1, 0, 0, 1, 0, 0]] * 5, 2),
    "pure_X_or_Z": ([[0, 1, 1, 1, 0, 0, 1, 0, 0]] * 5, 4),
    "mixed_only": ([[0, 0, 0, 0, 1, 1, 0, 1, 1]] * 5, 4),
    "centre_only": ([[0] * 9, [0] + [1] * 8, [0] * 9, [0] * 9, [0] * 9], 8),
    "corners_only": ([[0] + [1] * 8, [0] * 9, [0] + [1] * 8, [0] + [1] * 8, [0] + [1] * 8], 8),
    "qutrit0_only": ([[0] + [1] * 8] + [[0] * 9] * 4, 8),
    "qutrit3_only": ([[0] * 9] * 3 + [[0] + [1] * 8] + [[0] * 9], 8),
    # one tied pair with unequal rates: X0 at 2 units, X3^-1 at 1 unit, others 0
    "unequal_pair": ([[0, 2, 0, 0, 0, 0, 0, 0, 0], [0] * 9, [0] * 9,
                      [0, 0, 1, 0, 0, 0, 0, 0, 0], [0] * 9], 3),
}
for kname, (wq, D) in kinds.items():
    P = table(wq, D)
    Rk = strategies(P, D)
    Pbk = np.zeros((81, 3, 6), dtype=object)
    for c in range(9):
        Pbk[:, c // 3] += P[:, c]
    Rkb = strategies(Pbk, D)
    print(f"  {kname:13s} det rej {fmt(Rk['detect']['rej'], 1)} f|a {fmt(Rk['detect']['fail_given_acc'], 3)}"
          f" | MLD {fmt(Rk['mld']['fail_given_acc'], 2)} | hyb rej {fmt(Rk['hyb_strict']['rej'], 2)}"
          f" f|a {fmt(Rk['hyb_strict']['fail_given_acc'], 2)} | b-only MLD {fmt(Rkb['mld']['fail_given_acc'], 2)}"
          f" hyb rej {fmt(Rkb['hyb_strict']['rej'], 2)}")

# ======================================================== C8 threshold family
section("C8 posterior-threshold family")
for q in (1e-3, 1e-2, 0.05):
    u = q / 8
    vals = np.array([[sum(float(c) * u ** k for k, c in enumerate(P_dep[s, c_]))
                      for c_ in range(9)] for s in range(81)])
    tot = vals.sum(1)
    postv = vals.max(1) / tot
    lv = defaultdict(list)
    for s in range(81):
        lv[round(postv[s], 6)].append(cls_of[s])
    print(f"  q={q}: levels (posterior: classes):")
    for k in sorted(lv, reverse=True):
        print(f"      {k:.6f}: {dict(Counter(lv[k]))}")

# ======================================================== C11 gates
section("C11 logical gates (own symplectic maps from sdim frame rules)")


def frame(ops, v):
    x, z = list(v[:5]), list(v[5:])
    for o in ops:
        g, a = o[0], o[1]
        b = o[2] if len(o) > 2 else None
        if g == "H":
            x[a], z[a] = -z[a], x[a]
        elif g == "H_INV":
            x[a], z[a] = z[a], -x[a]
        elif g == "P":
            z[a] += x[a]
        elif g == "P_INV":
            z[a] -= x[a]
        elif g == "CZ":
            z[b] += x[a]
            z[a] += x[b]
        elif g == "CZ_INV":
            z[b] -= x[a]
            z[a] -= x[b]
        elif g == "SWAP":
            x[a], x[b] = x[b], x[a]
            z[a], z[b] = z[b], z[a]
        elif g in ("CNOT", "CNOT_INV"):
            s = 1 if g == "CNOT" else -1
            x[b] += s * x[a]
            z[a] -= s * z[b]
    return np.array([t % 3 for t in x + z])


TOK = {"H": [("SWAP", 2, 3), ("H", 0), ("H", 2), ("H", 3), ("H", 4), ("H_INV", 1)],
       "P": [("P", 0), ("P", 4), ("P_INV", 1), ("CZ", 2, 3)],
       "X": [], "Z": []}
INV = {"H": "H_INV", "H_INV": "H", "P": "P_INV", "P_INV": "P", "CZ": "CZ_INV",
       "CZ_INV": "CZ", "SWAP": "SWAP"}
TOK["H_INV"] = [(INV[o[0]],) + o[1:] for o in reversed(TOK["H"])]
TOK["P_INV"] = [(INV[o[0]],) + o[1:] for o in reversed(TOK["P"])]
Fa = np.array([symp(ZL, np.eye(10, dtype=int)[i]) for i in range(10)])
for tk in ("H", "H_INV", "P", "P_INV"):
    G = np.array([frame(TOK[tk], np.eye(10, dtype=int)[i]) for i in range(10)]).T
    symplectic = all(symp(G[:, i], G[:, j]) == symp(np.eye(10, dtype=int)[i], np.eye(10, dtype=int)[j])
                     for i in range(10) for j in range(10))
    # syndrome and coset of G e for unit e: sigma' rows
    Ms = (F_alg @ G) % 3          # functional of image
    # express Ms rows in span of F_alg rows and coset functionals
    basis = np.vstack([F_alg, Fa, B_alg]) % 3
    sol = []
    for row in np.vstack([Ms, (Fa @ G) % 3, (B_alg @ G) % 3]):
        found = None
        for cvec in itertools.product(range(3), repeat=6):
            if np.array_equal((np.array(cvec) @ basis) % 3, row % 3):
                found = cvec
                break
        sol.append(found)
    sol = np.array(sol)
    M = sol[:4, :4]
    LamK = sol[4:]
    print(f"  {tk}: symplectic {symplectic}; M = {M.tolist()}; (a',b') rows over (sig,a,b) = {LamK.tolist()}")
    # true vs naive prior decoding, depolarizing noise before the gate
    V2 = (V @ G.T) % 3
    s2 = ((V2 @ F_alg.T) % 3) @ np.array([1, 3, 9, 27])
    c2 = ((V2 @ Fa) % 3) + 3 * ((V2 @ B_alg) % 3)
    P2 = table(wdep, 8, sidx=s2, cidx=c2)
    perm = sorted(map(tuple, P2.reshape(-1, 6).tolist())) == sorted(map(tuple, P_dep.reshape(-1, 6).tolist()))
    # naive decoder: decisions from P_dep, scored with P2
    def naive(st):
        acc = [Fraction(0)] * 6
        fail = [Fraction(0)] * 6
        for g in range(81):
            polys0 = [to_q(P_dep[g, h], 8) for h in range(9)]
            orders = [next((k for k, c in enumerate(p) if c != 0), None) for p in polys0]
            m = min(o for o in orders if o is not None)
            lead = [h for h in range(9) if orders[h] == m]
            best = 0
            for h in range(9):
                if lexgt(polys0[h], polys0[best]):
                    best = h
            ok = True if st == "mld" else (len(lead) == 1)
            if not ok:
                continue
            pol = [to_q(P2[g, h], 8) for h in range(9)]
            tot = [sum(p[k] for p in pol) for k in range(6)]
            acc = [a + t for a, t in zip(acc, tot)]
            fail = [f + t - pol[best][k] for k, (f, t) in enumerate(zip(fail, tot))]
        rej = [-a for a in acc]
        rej[0] += 1
        return rej, series_div(fail, acc, 3)
    r_m = naive("mld")
    r_h = naive("hyb")
    print(f"     table permuted: {perm}; naive MLD P_fail = {fmt(r_m[1], 2)};"
          f" naive hybrid P_rej = {fmt(r_h[0], 1)}, P(f|a) = {fmt(r_h[1], 2)}")
    # weight-1 images
    cen = Counter()
    for i in w1:
        e2 = (G @ V[i]) % 3
        wgt = int(np.count_nonzero((e2[:5] != 0) | (e2[5:] != 0)))
        s_ = int(((F_alg @ e2) % 3) @ np.array([1, 3, 9, 27]))
        cen[(cls_of[sidx[i]], wgt, cls_of[s_])] += 1
    print("     weight-1 images (class before, weight after, syndrome class after):", dict(cen))

# ======================================================== C21 / C23 catalog
section("C21/C23 Clifford catalog, physical and logical level")
src = open(REPO + "/src/lrb/code_definitions.py").read()
blk = src[src.index("single_qutrit_cliffords_exhaustive_with_S_strings = ["):]
blk = blk[:blk.index("]")]
strs = [s.replace("S", "P") for s in re.findall(r'"([A-Z]+)"', blk)]
print("  strings:", len(strs), " distinct strings:", len(set(strs)))
print("  duplicated:", {s: c for s, c in Counter(strs).items() if c > 1})


def key(U):
    k = np.flatnonzero(np.abs(U.ravel()) > 1e-9)[0]
    U = U / (U.ravel()[k] / abs(U.ravel()[k]))
    return tuple(np.round(U.ravel(), 6))


M1 = {"H": H1, "P": P1, "X": X1, "Z": Z1, "I": np.eye(3)}


def word_unitary(s, Mdict):
    U = np.eye(Mdict["I"].shape[0], dtype=complex)
    for ch in s:            # rightmost symbol acts first (repo convention)
        U = U @ Mdict[ch]
    return U


phys = [key(word_unitary(s, M1)) for s in strs]
mult = Counter(Counter(phys).values())
grp = {key(np.eye(3))}
front = [np.eye(3, dtype=complex)]
while front:
    new = []
    for U in front:
        for g in (H1, P1, X1, Z1):
            U2 = U @ g
            k = key(U2)
            if k not in grp:
                grp.add(k)
                new.append(U2)
    front = new
print(f"  physical: distinct unitaries mod phase {len(set(phys))}, multiplicities {dict(mult)},"
      f" group order {len(grp)}, never sampled {len(grp - set(phys))}")
# also the reversed reading (leftmost acts first), for robustness
rev = [key(word_unitary(s[::-1], M1)) for s in strs]
print("  (reversed reading would give", len(set(rev)), "distinct)")

# logical level: 5-qutrit code space
psi5 = np.zeros((3,) * 5, complex)
psi5[(0,) * 5] = 1
plus = apply(psi5, ENC)


def tokop(psi, tk):
    if tk == "X":
        return apply(psi, [("X", 0), ("X", 3)])
    if tk == "Z":
        return apply(psi, [("Z", 0), ("Z", 2)])
    if tk == "I":
        return psi
    return apply(psi, TOK[tk])


# logical basis: |0>_L = projection of |+>_L on Z_L = 1, |j>_L = X_L^j |0>_L
ZLop = lambda p: apply(p, [("Z", 0), ("Z", 2)])
proj = (plus + ZLop(plus) + ZLop(ZLop(plus))) / 3
zero = proj / np.linalg.norm(proj)
basis = [zero, tokop(zero, "X"), tokop(tokop(zero, "X"), "X")]
for j in range(3):
    assert np.allclose(ZLop(basis[j]), W ** j * basis[j])
Lmat = {}
for tk in ("H", "P", "X", "Z", "I"):
    Lm = np.array([[np.vdot(basis[i], tokop(basis[j], tk)) for j in range(3)] for i in range(3)])
    assert np.allclose(Lm @ Lm.conj().T, np.eye(3)), tk     # preserves code space
    Lmat[tk] = Lm
for tk in ("H", "P"):
    # identify the logical gate as physical gate times a Pauli (mod phase)
    found = []
    for a, b in itertools.product(range(3), repeat=2):
        cand = np.linalg.matrix_power(X1, a) @ np.linalg.matrix_power(Z1, b) @ M1[tk]
        if key(cand) == key(Lmat[tk]):
            found.append((a, b))
    print(f"  logical {tk}_L = X^a Z^b {tk} with (a,b) in {found}")
logi = [key(word_unitary(s, Lmat)) for s in strs]
print(f"  logical: distinct logical unitaries mod phase {len(set(logi))},"
      f" multiplicities {dict(Counter(Counter(logi).values()))}")
# C23: images of fixed logical Paulis under catalog elements
Pauli = {(a, b): np.linalg.matrix_power(X1, a) @ np.linalg.matrix_power(Z1, b)
         for a in range(3) for b in range(3)}
pk = {key(v): ab for ab, v in Pauli.items()}


def conj_image(U, ab):
    return pk[key(U @ Pauli[ab] @ U.conj().T)]


Us = [word_unitary(s, Lmat) for s in strs]
for ab in ((1, 0), (0, 1), (1, 1)):
    for lab_, Ulist in (("V = C^-1 (one catalog draw)", [U.conj().T for U in Us]),
                        ("V = C (one catalog draw)", Us)):
        img = Counter(conj_image(U, ab) for U in Ulist)
        pz = Fraction(sum(v for k, v in img.items() if k[1] != 0), len(Ulist))
        print(f"  Pauli X^{ab[0]}Z^{ab[1]}: P(image has Z part) under {lab_}: {pz} = {float(pz):.4f}")
# two draws
ab = (1, 0)
tot = 0
hit = 0
Uinv = [U.conj().T for U in Us]
imgs1 = [conj_image(U, ab) for U in Uinv]
for i, U in enumerate(Uinv):
    for j, U2 in enumerate(Uinv):
        # V = C1^-1 C2^-1 (error after C2 is conjugated by (C2 C1)^-1)
        im = conj_image(Uinv[i] @ Uinv[j], ab)
        tot += 1
        hit += (im[1] != 0)
print(f"  two draws, X_L: P(Z part) = {Fraction(hit, tot)} = {hit / tot:.4f}")
# which group elements are missing: test the strings the catalog pattern implies
cand = ["XZHPP", "XZPPH", "XXZ", "XXZHPP", "XXZPPH", "ZZX", "ZZXHPP", "ZZXPPH",
        "XXZZHPP", "XXZZPPH"]
ck = [key(word_unitary(s, M1)) for s in cand]
print("  candidate missing strings (P = S):", cand)
print("    all absent from the catalog:", all(k not in set(phys) for k in ck),
      "| mutually distinct:", len(set(ck)) == len(ck), "| in group:", all(k in grp for k in ck))
# symplectic-part distribution of the catalog (Pauli prefix stripped)
def symp_key(U):
    ims = []
    for ab in ((1, 0), (0, 1)):
        ims.append(pk[key(U @ Pauli[ab] @ U.conj().T)])
    return tuple(ims)
sd = Counter(symp_key(word_unitary(s, M1)) for s in strs)
print("  catalog weight per SL(2,3) element (24 elements, uniform would be 9 each):",
      sorted(Counter(sd.values()).items()))
print("done")
