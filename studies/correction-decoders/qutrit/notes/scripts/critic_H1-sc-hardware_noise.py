"""Critic check of H1 claims C11, C12, C20 (idle error and round timing).

Independent re-implementation (row-stacking superoperator, different code)
of a Markovian qutrit idle channel, and its Weyl-twirled error probability
1 - F_pro. Adds the sensitivity H1 does not report:
  * echo T2 (H1's choice for the Goss sets) versus Ramsey T2 from the same
    Goss et al. tables (T2r_12 = 17 us for Q3/Q4, 10 us for Q5);
  * Markovian (exponential) versus quasi-static Gaussian dephasing for the
    same T2 (Ramsey decay of transmons is often Gaussian-like);
  * round timing including the two single-qutrit gates that turn each CZ
    into a CNOT (which h1_noise_budget.round_schedule omits).

Run:  module load python/3.11; python critic_H1-sc-hardware_noise.py
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

D = 3


def proj(i, j):
    m = np.zeros((D, D), complex)
    m[i, j] = 1
    return m


def superop_rowstack(Ls):
    I = np.eye(D)
    S = np.zeros((D * D, D * D), complex)
    for L in Ls:
        LdL = L.conj().T @ L
        S += np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T)
    return S


def channel(t, T1_10, T1_21, T2_01, T2_12, T1_20=None):
    g10, g21 = 1 / T1_10, 1 / T1_21
    g20 = 0.0 if T1_20 is None else 1 / T1_20
    G1, G2 = g10, g21 + g20
    # pure dephasing rates of the 01 and 12 coherences
    phi01 = max(1 / T2_01 - G1 / 2, 0.0)
    phi12 = max(1 / T2_12 - (G1 + G2) / 2, 0.0)
    a = 2 * phi01
    b = max(2 * phi12 - a, 0.0)
    Ls = [np.sqrt(g10) * proj(0, 1), np.sqrt(g21) * proj(1, 2),
          np.sqrt(a) * proj(1, 1), np.sqrt(b) * proj(2, 2)]
    if g20:
        Ls.append(np.sqrt(g20) * proj(0, 2))
    return expm(superop_rowstack(Ls) * t)


def err_markov(t, **kw):
    S = channel(t, **kw)
    # F_pro = (1/d^2) sum_ij <i|L(|i><j|)|j>; row-stacking index i*D+j
    f = sum(S[i * D + j, i * D + j] for i in range(D) for j in range(D)).real / D ** 2
    return 1 - f


def err_gaussian_dephasing(t, T1_10, T1_21, T2_01, T2_12, T1_20=None):
    """Amplitude damping Markovian; dephasing quasi-static Gaussian with
    exp(-(t/Tphi)^2) envelopes fitted so that the pure-dephasing factor at
    t = T2 matches the exponential one (same T2 definition, 1/e point)."""
    S = channel(t, T1_10, T1_21, 1e12, 1e12, T1_20)   # damping only
    g1 = 1 / T1_10
    g2 = 1 / T1_21 + (0 if T1_20 is None else 1 / T1_20)
    # pure-dephasing 1/e times
    tp01 = 1 / max(1 / T2_01 - g1 / 2, 1e-12)
    tp12 = 1 / max(1 / T2_12 - (g1 + g2) / 2, 1e-12)
    tp02 = 1 / max(1 / (1 / (1 / tp12 - 1 / tp01) if tp12 < tp01 else 1e12), 1e-12)
    env = {(0, 1): np.exp(-(t / tp01) ** 2), (1, 2): np.exp(-(t / tp12) ** 2),
           (0, 2): np.exp(-(t / tp02) ** 2)}
    f = 0.0
    for i in range(D):
        for j in range(D):
            e = 1.0 if i == j else env[tuple(sorted((i, j)))]
            f += S[i * D + j, i * D + j].real * e
    return 1 - f / D ** 2


SETS = {
    # Goss et al. 2022 Supplementary Tables 1-2 (fetched): echo and Ramsey
    "Goss Q4 echo (H1)": dict(T1_10=78, T1_21=47, T2_01=138, T2_12=45),
    "Goss Q4 Ramsey": dict(T1_10=78, T1_21=47, T2_01=99, T2_12=17),
    "Goss Q5 echo (H1)": dict(T1_10=45, T1_21=33, T2_01=63, T2_12=28),
    "Goss Q5 Ramsey": dict(T1_10=45, T1_21=33, T2_01=36, T2_12=10),
    "Liu23 Ramsey (H1)": dict(T1_10=180, T1_21=101, T2_01=76, T2_12=37),
    "IQM-like (H1 assumption)": dict(T1_10=44.4, T1_21=35.0, T1_20=69.2,
                                     T2_01=10.0, T2_12=5.0),
}


def main():
    print("== idle error 1 - F_pro (Markovian) ==")
    for t in (0.035, 0.5, 0.58, 1.0):
        row = f"t={t:5.3f} us: " + ", ".join(
            f"{k}: {err_markov(t, **v):.2e}" for k, v in SETS.items())
        print(row)
    print("\n== 500 ns readout layer: Markovian vs quasi-static Gaussian "
          "dephasing (same T2) ==")
    for k, v in SETS.items():
        print(f"  {k}: Markov {err_markov(0.5, **v):.2e}, Gaussian "
              f"{err_gaussian_dephasing(0.5, **v):.2e}")

    print("\n== C11 ratios at p = 0.027 ==")
    for lo, hi, what in ((0.022, 0.031, "readout 5p vs 3-state error"),):
        print(f"  {what}: {5 * 0.027 / hi:.1f}x to {5 * 0.027 / lo:.1f}x")
    e_lo = err_markov(0.5, **SETS["Goss Q4 echo (H1)"])
    e_hi = err_markov(0.5, **SETS["Goss Q5 Ramsey"])
    print(f"  idle 2p = 0.054 vs 500 ns readout idle {e_lo:.2e} (Q4 echo) "
          f"... {e_hi:.2e} (Q5 Ramsey): {0.054 / e_hi:.1f}x to "
          f"{0.054 / e_lo:.1f}x")

    print("\n== C12 round timing (us) ==")
    t1q, t2q, tm, tr = 0.035, 0.580, 0.500, 0.250
    seq = 12 * t2q + 4 * t1q + 4 * tm + tr
    seq_cnot = seq + 12 * 2 * t1q        # CNOT = H^-1 . CZ . H on target
    par = 4 * t2q + 2 * t1q + tm + tr
    par_fast = 4 * 0.100 + 2 * t1q + tm + tr
    seq_fast = 12 * 0.100 + 4 * t1q + 4 * tm + tr
    q4 = SETS["Goss Q4 echo (H1)"]
    for name, T in (("sequential, 580 ns CZ", seq),
                    ("sequential, 580 ns CZ, + CNOT basis changes", seq_cnot),
                    ("parallel, 580 ns CZ", par),
                    ("sequential, 100 ns CZ", seq_fast),
                    ("parallel, 100 ns CZ", par_fast)):
        print(f"  {name}: round {T:.2f} us, whole-round idle (Q4 echo) "
              f"{err_markov(T, **q4):.3f}, (Q4 Ramsey) "
              f"{err_markov(T, **SETS['Goss Q4 Ramsey']):.3f}")
    # gate-layer part only (excluding readout and reset already in SI1000),
    # and excluding the data qutrit's own two gates (data 0, 2, 3, 4)
    own = 2 * t2q
    g_seq = seq - 4 * tm - tr - own
    g_par = par - tm - tr - own
    print(f"  gate-layer idle of data 0/2/3/4 net of own gates: sequential "
          f"{g_seq:.2f} us -> {err_markov(g_seq, **q4):.3f}; parallel "
          f"{g_par:.2f} us -> {err_markov(g_par, **q4):.3f}")
    print(f"  compare: SI1000-style gate marginal for HW-A on data 0/2/3/4 per"
          f" round = 2 x 0.9 x 0.027 = {2 * 0.9 * 0.027:.3f}")

    print("\n== C20 per-us loss ==")
    for k in ("Goss Q4 echo (H1)", "Goss Q4 Ramsey", "IQM-like (H1 assumption)"):
        print(f"  {k}: {err_markov(1.0, **SETS[k]):.3f} per us")


if __name__ == "__main__":
    main()
