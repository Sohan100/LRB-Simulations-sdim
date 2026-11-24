"""Exact (enumerative) check of the mirrored-sequence decay for a single qutrit with
gate-independent Weyl noise. Enumerates all SL(2,3)^m symplectic sequences (the Weyl
part of each Clifford does not affect Weyl-vector propagation) for small m and computes
E[<Z>] exactly, then compares with
   mirrored:  lamZ(E_b) * gamma(E_f) * gamma(E_f o E_b)^(m-1)
   standard:  lamZ(E_b) * gamma(E_f)^m        (single recovery Clifford)
and with the naive product lamZ * gamma_f * (gamma_f*gamma_b)^(m-1)."""
import itertools
import numpy as np

w = np.exp(2j * np.pi / 3)
mats = [np.array([[a, b], [c, e]]) for a, b, c, e in itertools.product(range(3), repeat=4)
        if (a * e - b * c) % 3 == 1]
G = len(mats)


def inv(M):
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) % 3


vecs = [np.array(v) for v in itertools.product(range(3), repeat=2)]


def chi(v):
    return w ** (v[0] % 3)


def phi_table(q):
    """phi[S index] = sum_e q(e) chi(S^{-1} e)"""
    return np.array([sum(p * chi(inv(M) @ np.array(e) % 3) for e, p in q.items()) for M in mats])


def mul_index():
    keyd = {tuple(M.flatten()): i for i, M in enumerate(mats)}
    tab = np.zeros((G, G), dtype=int)
    for i, A in enumerate(mats):
        for j, B in enumerate(mats):
            tab[i, j] = keyd[tuple(((A @ B) % 3).flatten())]
    return tab


MUL = mul_index()
ID = [i for i, M in enumerate(mats) if (M == np.eye(2, dtype=int)).all()][0]


def gamma(q):
    return (9 * q.get((0, 0), 0) - 1) / 8


def compose(q1, q2):
    out = {}
    for (a1, b1), p1 in q1.items():
        for (a2, b2), p2 in q2.items():
            v = ((a1 + a2) % 3, (b1 + b2) % 3)
            out[v] = out.get(v, 0) + p1 * p2
    return out


def lamZ(q):
    return sum(p * chi(np.array(e)) for e, p in q.items())


def exact(qf, qb, m, mirrored=True):
    pf, pb = phi_table(qf), phi_table(qb)
    lab_b = lamZ(qb)
    total = 0.0
    for seq in itertools.product(range(G), repeat=m):
        S = ID
        frames = []
        for i in seq:
            S = MUL[i, S]
            frames.append(S)
        val = 1.0
        for k in range(m):
            val *= pf[frames[k]]                 # forward error after C_{k+1}
        if mirrored:
            for k in range(m - 1):               # backward error after C_{k+2}^{-1} in frame S_{k+1}
                val *= pb[frames[k]]
            val *= lab_b                         # backward error after C_1^{-1} in lab frame
        else:
            val *= lab_b                         # error after the single recovery Clifford (lab frame)
        total += val
    return total / G ** m


eps = 0.2
q_dep = {(0, 0): 1 - eps, **{v: eps / 8 for v in itertools.product(range(3), repeat=2) if v != (0, 0)}}
q_z = {(0, 0): 1 - eps, (0, 1): eps / 2, (0, 2): eps / 2}
q_x = {(0, 0): 1 - eps, (1, 0): eps / 2, (2, 0): eps / 2}
for name, qf, qb in (("depolarizing", q_dep, q_dep), ("Z-dephasing", q_z, q_z),
                     ("fwd Z-dephasing, bwd X-flip", q_z, q_x)):
    for m in (1, 2, 3, 4):
        ex = exact(qf, qb, m, True).real
        pred = (lamZ(qb) * gamma(qf) * gamma(compose(qf, qb)) ** (m - 1)).real
        naive = (lamZ(qb) * gamma(qf) * (gamma(qf) * gamma(qb)) ** (m - 1)).real
        exs = exact(qf, qb, m, False).real
        preds = (lamZ(qb) * gamma(qf) ** m).real
        print(f"{name:28s} m={m}: mirrored exact={ex:.6f} frame-pred={pred:.6f} naive={naive:.6f} | "
              f"standard exact={exs:.6f} pred={preds:.6f}")
