"""Exact density-matrix check: real-time physical correction vs offline Pauli(Weyl)-frame
relabelling, for a small qutrit Clifford circuit with mid-circuit syndrome measurement.

Checks
 (1) every qutrit Clifford used by the LRB circuits (sdim matrices for H, P, CNOT and their
     inverses; standard CZ, CZ_INV, SWAP) maps every Weyl operator to a phase times a Weyl
     operator (frame propagation closes);
 (2) the qutrit T gate diag(1, z, z^8), z = exp(2 pi i / 9), maps X to a non-Weyl operator that
     is itself Clifford (third level of the hierarchy);
 (3) Weyl channels are Weyl-covariant, ladder amplitude damping is not;
 (4) toy protocol on 3 data qutrits (repetition code |+>_L) + 2 ancillas measuring
     Z0 Z1^-1 and Z1 Z2^-1: noise -> syndrome measurement -> lookup correction ->
     more noise -> Clifford U -> computational-basis readout of the data.
     R = correction applied physically (perfect gate) right after the syndrome;
     F = no physical action, Weyl frame propagated through U, readout relabelled.
     Compare the exact joint distributions P(s1, s2, k0, k1, k2).
"""
import itertools
import numpy as np

d = 3
w = np.exp(2j * np.pi / d)
# sdim conventions (sdim/unitary.py): X|j> = |j+1>, Z = diag(w^j), H_mn = w^{mn}/sqrt3,
# P = diag(w^{j(j-1)/2}), CNOT|i,j> = |i, i+j>.
X = np.zeros((d, d), complex)
for i in range(d):
    X[(i + 1) % d, i] = 1
Z = np.diag([w ** j for j in range(d)])
H = np.array([[w ** (m * n) for n in range(d)] for m in range(d)]) / np.sqrt(d)
P = np.diag([w ** (j * (j - 1) // 2) for j in range(d)])
I3 = np.eye(d)


def kron(*ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


CNOT = np.zeros((d * d, d * d), complex)
CZ = np.zeros((d * d, d * d), complex)
SWAP = np.zeros((d * d, d * d), complex)
for i, j in itertools.product(range(d), repeat=2):
    CNOT[d * i + (i + j) % d, d * i + j] = 1
    CZ[d * i + j, d * i + j] = w ** (i * j)
    SWAP[d * j + i, d * i + j] = 1
gates1 = {"H": H, "H_INV": H.conj().T, "P": P, "P_INV": P.conj().T, "X": X, "Z": Z}
gates2 = {"CNOT": CNOT, "CNOT_INV": CNOT.conj().T, "CZ": CZ, "CZ_INV": CZ.conj().T, "SWAP": SWAP}


def weyl(a, b):
    return np.linalg.matrix_power(X, a % d) @ np.linalg.matrix_power(Z, b % d)


def weyl_n(avec, bvec):
    return kron(*[weyl(a, b) for a, b in zip(avec, bvec)])


def as_weyl(M, n):
    """Return (avec, bvec, phase) if M = phase * X^a Z^b (n qutrits), else None.  O(D^2)."""
    D = d ** n
    col = M[:, 0]
    idx = int(np.argmax(np.abs(col)))
    if abs(col[idx]) < 1e-9:
        return None
    avec = np.unravel_index(idx, [d] * n)
    Xa = weyl_n(avec, [0] * n)
    R = Xa.conj().T @ M
    if not np.allclose(R, np.diag(np.diag(R)), atol=1e-9):
        return None
    r = np.diag(R)
    bvec = []
    for i in range(n):
        e = [0] * n; e[i] = 1
        j = np.ravel_multi_index(e, [d] * n)
        ang = np.angle(r[j] / r[0]) / (2 * np.pi / d)
        bvec.append(int(round(ang)) % d)
    ph = r[0]
    if np.allclose(M, ph * weyl_n(avec, bvec), atol=1e-9) and abs(abs(ph) - 1) < 1e-9:
        return tuple(int(a) for a in avec), tuple(bvec), ph
    return None


# ---------------------------------------------------------------- (1) Clifford closure
print("(1) Clifford conjugation maps Weyl -> phase * Weyl")
for name, G in gates1.items():
    ok = all(as_weyl(G @ weyl(a, b) @ G.conj().T, 1) is not None for a in range(d) for b in range(d))
    print(f"   {name:8s} closure on all 9 one-qutrit Weyl operators: {ok}")
for name, G in gates2.items():
    ok = all(as_weyl(G @ weyl_n(av, bv) @ G.conj().T, 2) is not None
             for av in itertools.product(range(d), repeat=2) for bv in itertools.product(range(d), repeat=2))
    print(f"   {name:8s} closure on all 81 two-qutrit Weyl operators: {ok}")

# ---------------------------------------------------------------- (2) qutrit T gate
z = np.exp(2j * np.pi / 9)
T = np.diag([1, z, z ** 8])
TXT = T @ X @ T.conj().T
print("(2) qutrit T = diag(1, z, z^8): T X T^dag is Weyl?", as_weyl(TXT, 1) is not None)
is_cliff = all(as_weyl(TXT @ weyl(a, b) @ TXT.conj().T, 1) is not None for a in range(d) for b in range(d))
print("    ... but T X T^dag is Clifford (maps all Weyl to Weyl):", is_cliff)

# ---------------------------------------------------------------- (3) covariance
rng = np.random.default_rng(7)


def rand_rho(D):
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    r = A @ A.conj().T
    return r / np.trace(r)


def weyl_channel_kraus(probs):
    return [np.sqrt(probs[a, b]) * weyl(a, b) for a in range(d) for b in range(d) if probs[a, b] > 0]


def amp_damp_kraus(g):
    K0 = np.diag([1, np.sqrt(1 - g), np.sqrt(1 - 2 * g)])
    K1 = np.zeros((d, d)); K1[0, 1] = np.sqrt(g)
    K2 = np.zeros((d, d)); K2[1, 2] = np.sqrt(2 * g)
    return [K0.astype(complex), K1.astype(complex), K2.astype(complex)]


def apply_kraus(rho, ks):
    return sum(k @ rho @ k.conj().T for k in ks)


pw = rng.random((d, d)); pw[0, 0] = 10.0; pw /= pw.sum()
kr_w, kr_ad = weyl_channel_kraus(pw), amp_damp_kraus(0.05)
print("(3) covariance  || N(V rho V^dag) - V N(rho) V^dag ||_1 for V = X, random rho")
rho = rand_rho(d)
for lab, ks in [("random Weyl channel", kr_w), ("ladder amplitude damping g=0.05", kr_ad)]:
    diff = apply_kraus(X @ rho @ X.conj().T, ks) - X @ apply_kraus(rho, ks) @ X.conj().T
    print(f"   {lab:34s}: {np.abs(np.linalg.eigvalsh((diff + diff.conj().T) / 2)).sum():.3e}")

# ---------------------------------------------------------------- (4) toy protocol
n = 5          # qutrits 0,1,2 data; 3,4 ancillas
DATA = [0, 1, 2]


def op_on(U, qs):
    """Embed a k-qutrit operator U acting on qutrits qs (in that order) into n qutrits."""
    k = len(qs)
    Ut = U.reshape([d] * (2 * k))
    def f(rho):
        r = rho.reshape([d] * (2 * n))
        # act on ket indices
        r = np.moveaxis(r, qs, list(range(k)))
        r = np.tensordot(Ut, r, axes=(list(range(k, 2 * k)), list(range(k))))
        r = np.moveaxis(r, list(range(k)), qs)
        # act on bra indices with U^*
        bq = [q + n for q in qs]
        r = np.moveaxis(r, bq, list(range(k)))
        r = np.tensordot(Ut.conj(), r, axes=(list(range(k, 2 * k)), list(range(k))))
        r = np.moveaxis(r, list(range(k)), bq)
        return r.reshape(d ** n, d ** n)
    return f


def kraus_on(ks, q):
    fs = [op_on(k, [q]) for k in ks]
    return lambda rho: sum(f(rho) for f in fs)


def project(rho, q, val):
    Pj = np.zeros((d, d), complex); Pj[val, val] = 1
    return op_on(Pj, [q])(rho)


def reset_anc(rho, q):
    # trace out q and re-prepare |0>: sum_v |0><v| rho |v><0|
    out = 0
    for v in range(d):
        K = np.zeros((d, d), complex); K[0, v] = 1
        out = out + op_on(K, [q])(rho)
    return out


def ket0():
    v = np.zeros(d ** n, complex); v[0] = 1
    return np.outer(v, v.conj())


# lookup decoder: syndrome (s1, s2) = (x0 - x1, x1 - x2) of single X^a errors
lut = {(0, 0): (0, 0, 0)}
for q in range(3):
    for a in (1, 2):
        e = [0, 0, 0]; e[q] = a
        s = ((e[0] - e[1]) % d, (e[1] - e[2]) % d)
        lut[s] = tuple((-x) % d for x in e)   # correction = inverse of the error


PREP = [H]

def run(noise_ks, U_list, U_label, nonclifford=False):
    """Return exact joint distributions for protocols R and F (relabelled)."""
    rho = ket0()
    rho = op_on(PREP[0], [0])(rho)
    rho = op_on(CNOT, [0, 1])(rho)
    rho = op_on(CNOT, [0, 2])(rho)
    for q in DATA:
        rho = kraus_on(noise_ks, q)(rho)
    # syndrome extraction: anc3 <- x0 - x1 ; anc4 <- x1 - x2
    rho = op_on(CNOT, [0, 3])(rho); rho = op_on(CNOT.conj().T, [1, 3])(rho)
    rho = op_on(CNOT, [1, 4])(rho); rho = op_on(CNOT.conj().T, [2, 4])(rho)
    distR, distF = {}, {}
    for s1 in range(d):
        for s2 in range(d):
            br = project(project(rho, 3, s1), 4, s2)
            ps = np.trace(br).real
            if ps < 1e-15:
                continue
            br = reset_anc(reset_anc(br, 3), 4)
            corr = lut.get((s1, s2), (0, 0, 0))
            C = kron(*[np.linalg.matrix_power(X, c) for c in corr])
            # protocol R: physical correction now
            rR = op_on(C, DATA)(br)
            # protocol F: nothing physical; frame = C
            rF = br
            for q in DATA:
                rR = kraus_on(noise_ks, q)(rR)
                rF = kraus_on(noise_ks, q)(rF)
            Utot = np.eye(d ** 3, dtype=complex)
            for (U, qs) in U_list:
                # build operator on data register
                rR = op_on(U, qs)(rR)
                rF = op_on(U, qs)(rF)
                # accumulate data-register unitary for frame propagation
                emb = np.zeros((d ** 3, d ** 3), complex)
                Ud = U.reshape([d] * (2 * len(qs)))
                for idx in itertools.product(range(d), repeat=3):
                    vec = np.zeros([d] * 3, complex); vec[idx] = 1
                    t = np.moveaxis(vec, qs, list(range(len(qs))))
                    t = np.tensordot(Ud, t, axes=(list(range(len(qs), 2 * len(qs))), list(range(len(qs)))))
                    t = np.moveaxis(t, list(range(len(qs))), qs)
                    emb[:, np.ravel_multi_index(idx, [d] * 3)] = t.reshape(-1)
                if not any(U is G for G in NONCLIFFORD):
                    Utot = emb @ Utot
            Fprop = Utot @ C @ Utot.conj().T
            wf = as_weyl(Fprop, 3)
            assert wf is not None
            avec = wf[0]
            pR = np.real(np.diag(rR)).reshape([d] * n).sum(axis=(3, 4))
            pF = np.real(np.diag(rF)).reshape([d] * n).sum(axis=(3, 4))
            for k in itertools.product(range(d), repeat=3):
                distR[(s1, s2) + k] = distR.get((s1, s2) + k, 0) + pR[k]
                kr = tuple((k[i] + avec[i]) % d for i in range(3))
                distF[(s1, s2) + kr] = distF.get((s1, s2) + kr, 0) + pF[k]
    keys = set(distR) | set(distF)
    tv = 0.5 * sum(abs(distR.get(k, 0) - distF.get(k, 0)) for k in keys)
    return tv


NONCLIFFORD = [T]
p = 0.02
dep = np.full((d, d), p / 8); dep[0, 0] = 1 - p
# generic single-qutrit input state |psi> on qutrit 0, encoded in the repetition code
v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v)
M = np.column_stack([v, rng.normal(size=(d, d - 1)) + 1j * rng.normal(size=(d, d - 1))])
Q, Rm = np.linalg.qr(M)
V = Q * (v[0] / Q[0, 0])          # unitary whose first column is |psi>
assert np.allclose(V[:, 0], v) and np.allclose(V.conj().T @ V, np.eye(d))
CINV = CNOT.conj().T
cases = [
    ("|+>_L, U = H0 CNOT02 P1 H1^dag CZ12 (Clifford)", H, [(H, [0]), (CNOT, [0, 2]), (P, [1]), (H.conj().T, [1]), (CZ, [1, 2])]),
    ("|psi>_L, U = decode, P0, H0 (Clifford)", V, [(CINV, [0, 2]), (CINV, [0, 1]), (P, [0]), (H, [0])]),
    ("|psi>_L, U = decode, T0, H0 (non-Clifford, naive frame)", V, [(CINV, [0, 2]), (CINV, [0, 1]), (T, [0]), (H, [0])]),
]
noises = [("uniform Weyl depolarizing p=0.02", weyl_channel_kraus(dep)),
          ("random biased Weyl channel", kr_w),
          ("ladder amplitude damping g=0.05", kr_ad)]
print("(4) total-variation distance between R (physical correction, perfect gate) and F (frame relabelling)")
for clab, prep, U in cases:
    PREP = [prep]
    for nlab, ks in noises:
        print(f"   {clab:56s} | {nlab:34s} | TV = {run(ks, U, clab):.3e}")

print("(5) scaling of TV(R,F) with ladder amplitude-damping strength g (Clifford cases)")
for clab, prep, U in cases[:2]:
    PREP = [prep]
    for g in (0.05, 0.025, 0.0125):
        print(f"   {clab:48s} | g = {g:<7} | TV = {run(amp_damp_kraus(g), U, clab):.3e}")
