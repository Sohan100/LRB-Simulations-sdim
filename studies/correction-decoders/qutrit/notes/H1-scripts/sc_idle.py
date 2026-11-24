"""H1 (rewrite) - Markovian qutrit idle channel and its SI1000 equivalent.

Model: amplitude damping 1->0 (rate g10), 2->1 (g21), optional 2->0 (g20),
plus pure dephasing chosen to reproduce measured T2 of the 01, 12 and
(optionally) 02 coherences.  For this channel (no Hamiltonian part)

    F_pro(t) = (1/9) [ 1 + e^{-g10 t} + e^{-(g21+g20) t}
                       + 2 e^{-t/T2_01} + 2 e^{-t/T2_12} + 2 e^{-t/T2_02} ],

and the SI1000 'd' location with the same process infidelity has
probability e(t) = 1 - F_pro(t).  The closed form is cross-checked against a
matrix-exponential Lindblad solution, and the Weyl-twirl split into X-type
(X^a Z^b, a != 0) and pure-Z errors is reported.

Coherence sets (all fetched in this task):
  Goss2022 Q4 / Q5: Goss et al., arXiv:2206.07216, Supp. Tables 1-2
      (T1_01, T1_12, T2 echo and Ramsey for 01, 12, 02).
  Blok2021 avg: Blok et al., arXiv:2003.03307 (T1 56.0 / 34.8 us,
      T2* 39 / 14 us, T2E 61.2 / 28 us; no 02 value -> T2_02 derived).
  Liu2023: Liu et al., arXiv:2304.11841 (T1 180 / 101 us, Ramsey
      T2 76 / 37 us).
  IQM-like (assumption): T1 from the IQM Spark |2> demonstration,
      arXiv:2402.07315 (1/G10 = 44.4, 1/G21 = 35.0, 1/G20 = 69.2 us),
      T2_01 = 35 us from the IQM star-QPU error model (arXiv:2503.12869,
      T2* = 35 us), T2_12 = 17 us ASSUMED (not measured in any source).

Run: module load python/3.11; python sc_idle.py   (seconds)
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.linalg import expm

D = 3
W = np.exp(2j * np.pi / 3)
XOP = np.roll(np.eye(D), 1, axis=0)
ZOP = np.diag([W ** j for j in range(D)])

COH = {
    # name: (T1_10, T1_21, T1_20 or None, T2_01, T2_12, T2_02 or None)
    "Goss2022 Q4 echo": (78, 47, None, 138, 45, 62),
    "Goss2022 Q4 Ramsey": (78, 47, None, 99, 17, 21),
    "Goss2022 Q5 echo": (45, 33, None, 63, 28, 37),
    "Goss2022 Q5 Ramsey": (45, 33, None, 36, 10, 11),
    "Blok2021 avg echo": (56.0, 34.8, None, 61.2, 28.0, None),
    "Blok2021 avg Ramsey": (56.0, 34.8, None, 39.0, 14.0, None),
    "Liu2023 Ramsey": (180, 101, None, 76, 37, None),
    "IQM-like (assumed T2_12)": (44.4, 35.0, 69.2, 35.0, 17.0, None),
    "target (2x Goss Q4 echo)": (156, 94, None, 276, 90, 124),
}


def rates(T1_10, T1_21, T1_20, T2_01, T2_12, T2_02):
    g10, g21 = 1 / T1_10, 1 / T1_21
    g20 = 0.0 if T1_20 is None else 1 / T1_20
    G01 = 1 / T2_01
    G12 = 1 / T2_12
    if T2_02 is None:
        # no measured 02 coherence: assume independent 01 and 12 transition-
        # frequency noise, so the 02 pure-dephasing rate is the sum.  (For the
        # Goss sets, where T2_02 was measured, this rule gives 0.0060/us vs the
        # measured 0.0055/us (echo) and 0.046/us vs 0.037/us (Ramsey).)
        phi01 = max(G01 - g10 / 2, 0.0)
        phi12 = max(G12 - (g10 + g21 + g20) / 2, 0.0)
        phi02 = phi01 + phi12
        G02 = (g21 + g20) / 2 + phi02
    else:
        G02 = 1 / T2_02
    return g10, g21, g20, G01, G12, G02


def fpro_closed(t, coh):
    g10, g21, g20, G01, G12, G02 = rates(*coh)
    return (1 + np.exp(-g10 * t) + np.exp(-(g21 + g20) * t)
            + 2 * np.exp(-G01 * t) + 2 * np.exp(-G12 * t)
            + 2 * np.exp(-G02 * t)) / 9


def e_idle(t, coh):
    """SI1000-equivalent depolarizing probability for an idle of t us."""
    if t <= 0:
        return 0.0
    return float(1 - fpro_closed(t, coh))


def lindblad_superop(coh):
    g10, g21, g20, G01, G12, G02 = rates(*coh)
    P = lambda i, j: np.outer(np.eye(D)[i], np.eye(D)[j])
    Ls = [np.sqrt(g10) * P(0, 1), np.sqrt(g21) * P(1, 2)]
    if g20:
        Ls.append(np.sqrt(g20) * P(0, 2))
    # pure dephasing rates of the three coherences
    ph01 = G01 - g10 / 2
    ph12 = G12 - (g10 + g21 + g20) / 2
    ph02 = G02 - (g21 + g20) / 2
    # covariance of level-energy noise (d0 = 0, d1, d2)
    M = np.array([[2 * ph01, ph01 + ph02 - ph12], [ph01 + ph02 - ph12, 2 * ph02]])
    lam, U = np.linalg.eigh(M)
    clipped = lam.min() < -1e-12
    for k in range(2):
        if lam[k] > 0:
            Ls.append(np.sqrt(lam[k]) * np.diag([0, U[0, k], U[1, k]]))
    # column-stacking superoperator: vec(A X B) = (B^T kron A) vec(X)
    I = np.eye(D)
    S = np.zeros((D * D, D * D), complex)
    for L in Ls:
        LdL = L.conj().T @ L
        S += np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I)
    return S, clipped


def channel_apply(Sexp, rho):
    v = rho.reshape(-1, order="F")
    return (Sexp @ v).reshape(D, D, order="F")


def fpro_and_twirl(t, coh):
    S, clipped = lindblad_superop(coh)
    Sexp = expm(S * t)
    # Choi J = sum_ij E_ij (x) L(E_ij)
    J = np.zeros((D * D, D * D), complex)
    for i in range(D):
        for j in range(D):
            E = np.zeros((D, D))
            E[i, j] = 1
            J += np.kron(E, channel_apply(Sexp, E))
    probs = {}
    for a, b in itertools.product(range(D), repeat=2):
        Wk = np.linalg.matrix_power(XOP, a) @ np.linalg.matrix_power(ZOP, b)
        v = np.zeros(D * D, complex)
        for i in range(D):
            v += np.kron(np.eye(D)[i], Wk @ np.eye(D)[i])
        probs[(a, b)] = float(np.real(v.conj() @ J @ v)) / D ** 2
    return probs[(0, 0)], probs, clipped


def main():
    ts = [0.035, 0.1, 0.25, 0.3, 0.5, 0.58, 1.0]
    print("idle error e(t) = 1 - F_pro (SI1000 'd' probability), closed form")
    print("  " + "set".ljust(28) + "".join(f"{t:>9.3f}" for t in ts) + "  (us)")
    for name, coh in COH.items():
        print("  " + name.ljust(28) + "".join(f"{e_idle(t, coh):9.2e}" for t in ts))
    print("\ncross-check vs matrix exponential, Weyl-twirl split at t = 0.5 us:")
    for name, coh in COH.items():
        f, probs, clipped = fpro_and_twirl(0.5, coh)
        px = sum(v for (a, b), v in probs.items() if a)
        pz = sum(v for (a, b), v in probs.items() if a == 0 and b)
        print(f"  {name:28s} 1-F closed {e_idle(0.5, coh):.4e}, expm {1 - f:.4e};"
              f" X-type {px:.2e}, pure-Z {pz:.2e}, X share {px / (px + pz):.2f}"
              + ("  (dephasing covariance clipped)" if clipped else ""))
    print("\nper-microsecond idle loss (e(1us)):")
    for name, coh in COH.items():
        print(f"  {name:28s} {e_idle(1.0, coh):.4f}")


if __name__ == "__main__":
    main()
