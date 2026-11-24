"""H1 hardware noise budget for one LRB-D check round on transmon qutrits.

1. Idle error of a transmon qutrit for a duration t, from a Lindblad model
   with level decay |1>->|0>, |2>->|1> (and optionally |2>->|0>) and pure
   dephasing fitted to measured echo/Ramsey times. The figure reported is
   the Weyl-twirled error probability 1 - F_pro, which is exactly the
   probability of the repo's N1 "d" location (non-identity Weyl error), so
   it can be compared one-to-one with the SI1000 idle rate 2p.
2. Round timing and per-round idle exposure of a data qutrit for the repo's
   sequential schedule (one ancilla per measurement layer) and for a
   parallel schedule, with literature gate/readout/reset durations.
3. SI1000 location rates at p = two-qutrit gate error, against measured
   hardware rates (readout, single-qutrit gates, reset).

All literature numbers are hard-coded with their source in the comments;
see H1-sc-hardware.md for full citations.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

D = 3


def ket(j):
    v = np.zeros((D, 1), complex)
    v[j] = 1
    return v


def lindblad_superop(c_ops):
    """Column-stacking superoperator of sum_k D[L_k]."""
    I = np.eye(D)
    S = np.zeros((D * D, D * D), complex)
    for L in c_ops:
        LdL = L.conj().T @ L
        S += np.kron(L.conj(), L)
        S -= 0.5 * np.kron(I, LdL)
        S -= 0.5 * np.kron(LdL.T, I)
    return S


def idle_error(t_us, T1_1, T1_2, T2_01, T2_12, T2_02=None, T1_20=None):
    """Weyl-twirled error probability 1 - F_pro of a qutrit idle of t_us.

    T1_1: lifetime of |1> (decay to |0>); T1_2: decay time |2>->|1>;
    T1_20: optional direct |2>->|0> decay time; T2_jk: echo/Ramsey times of
    the jk coherences. Pure dephasing is modelled with two independent
    collapse operators sqrt(a)|1><1| and sqrt(b)|2><2| (rates a/2, b/2,
    (a+b)/2 on the 01, 02, 12 coherences), with a, b fitted to T2_01 and
    T2_12 (T2_02 then follows; if given it is only reported).
    """
    G1 = 1.0 / T1_1
    G21 = 1.0 / T1_2
    G20 = 0.0 if T1_20 is None else 1.0 / T1_20
    G2 = G21 + G20
    # Coherence decay rates: (G_j + G_k)/2 + pure dephasing.
    phi01 = max(1.0 / T2_01 - G1 / 2, 0.0)
    phi12 = max(1.0 / T2_12 - (G1 + G2) / 2, 0.0)
    # phi01 = a/2, phi12 = (a+b)/2  ->  a = 2 phi01, b = 2 phi12 - a
    a = 2 * phi01
    b = max(2 * phi12 - a, 0.0)
    c_ops = [np.sqrt(G1) * ket(0) @ ket(1).T,
             np.sqrt(G21) * ket(1) @ ket(2).T,
             np.sqrt(a) * ket(1) @ ket(1).T,
             np.sqrt(b) * ket(2) @ ket(2).T]
    if G20 > 0:
        c_ops.append(np.sqrt(G20) * ket(0) @ ket(2).T)
    S = expm(lindblad_superop(c_ops) * t_us)
    f_pro = np.trace(S).real / D ** 2
    implied_T2_02 = 1.0 / (G2 / 2 + b / 2)
    return 1.0 - f_pro, implied_T2_02


# ---- coherence parameter sets (microseconds) ---------------------------
PARAM_SETS = {
    # Goss et al. 2022 (arXiv:2206.07216), Supplementary table, CZ pair Q3/Q4
    "Goss22 Q3 (best)": dict(T1_1=125, T1_2=63, T2_01=190, T2_12=61),
    "Goss22 Q4": dict(T1_1=78, T1_2=47, T2_01=138, T2_12=45),
    # Goss et al. 2022, CZ-dagger pair Q5/Q6
    "Goss22 Q5": dict(T1_1=45, T1_2=33, T2_01=63, T2_12=28),
    # Liu et al. 2023 PRX (arXiv:2304.11841) Table S1, Ramsey T2
    "Liu23 (Ramsey)": dict(T1_1=180, T1_2=101, T2_01=76, T2_12=37),
    # IQM Spark prototype (Ronkko et al. 2024, arXiv:2402.07315) gives
    # Gamma10^-1=44.4, Gamma21^-1=35.0, Gamma20^-1=69.2 us but no T2; the
    # IQM fake-Garnet backend lists qubit T2 ~ 9-12 us. ASSUMPTION: T2_01 =
    # 10 us and T2_12 = 5 us (ratio ~0.5 as in Goss22/Liu23).
    "IQM-like (assumed T2)": dict(T1_1=44.4, T1_2=35.0, T1_20=69.2,
                                  T2_01=10.0, T2_12=5.0),
}

DURATIONS_US = {
    "single-qutrit gate 35 ns (Yu25/Liu23)": 0.035,
    "single-qutrit H 128 ns (Morvan21)": 0.128,
    "flux CZ 40 ns (IQM qubit CZ, fake Garnet)": 0.040,
    "CZdag 580 ns (Goss22)": 0.580,
    "CZ 783 ns (Goss22)": 0.783,
    "3-state readout 140 ns (Chen23)": 0.140,
    "3-state readout ~500 ns (Krinner22 pulse+integration)": 0.500,
    "readout ~1 us (IBM MCM, Govia23)": 1.0,
    "multi-level reset 250 ns (McEwen21)": 0.250,
}


def round_schedule(t1q, t2q, tm, treset, sequential=True,
                   relay_hops=0, two_q_layers=None):
    """Duration of one check round and the idle time of a data qutrit.

    sequential=True mimics the repo order: each ancilla's three couplings
    are applied one after another and the ancilla is read out before the
    next ancilla starts (4 measurement layers), then one reset layer.
    sequential=False: couplings packed into ``two_q_layers`` parallel layers
    (4 is the minimum because data qutrit 1 couples to all four ancillas),
    two single-qutrit layers for the X-ancilla basis changes, one parallel
    readout layer and one reset layer. Relay hops add 2 two-qutrit layers
    each when sequential, 2 extra layers in total when parallel.
    """
    n2q = 12 + 2 * relay_hops
    if sequential:
        t_gates = n2q * t2q + 4 * t1q  # H and H_INV on ancillas 7 and 8
        t_meas = 4 * tm
    else:
        layers = (two_q_layers or 4) + (2 if relay_hops else 0)
        t_gates = layers * t2q + 2 * t1q
        t_meas = tm
    total = t_gates + t_meas + treset
    return total, n2q


def main():
    print("== 1. Qutrit idle error (Weyl-twirled, = N1 'd' probability) ==")
    header = "duration".ljust(52) + "".join(k[:22].rjust(24)
                                            for k in PARAM_SETS)
    print(header)
    for name, t in DURATIONS_US.items():
        row = name.ljust(52)
        for ps in PARAM_SETS.values():
            e, _ = idle_error(t, **ps)
            row += f"{e:24.2e}"
        print(row)
    for k, ps in PARAM_SETS.items():
        _, t202 = idle_error(1.0, **ps)
        print(f"  implied T2_02 for {k}: {t202:.1f} us")

    print("\n== 2. Check-round timing and data idle exposure ==")
    scenarios = {
        # (t1q, t2q, tm, treset) in us
        "microwave cross-Kerr CZ (Goss22) + 500 ns readout":
            (0.035, 0.580, 0.500, 0.250),
        "assumed fast flux CZ 100 ns + 500 ns readout":
            (0.035, 0.100, 0.500, 0.250),
        "assumed fast flux CZ 100 ns + 140 ns readout":
            (0.035, 0.100, 0.140, 0.250),
    }
    for sname, (t1q, t2q, tm, tr) in scenarios.items():
        for seq in (True, False):
            for hops in (0, 6):
                T, n2q = round_schedule(t1q, t2q, tm, tr, sequential=seq,
                                        relay_hops=hops)
                errs = [idle_error(T, **PARAM_SETS[k])[0]
                        for k in ("Goss22 Q4", "IQM-like (assumed T2)")]
                print(f"{sname} | {'sequential' if seq else 'parallel'} | "
                      f"relay hops {hops}: round {T:.2f} us, {n2q} 2q gates; "
                      f"data idle error over whole round: Goss22-Q4 "
                      f"{errs[0]:.3f}, IQM-like {errs[1]:.3f}")

    print("\n== 3. SI1000 location rates vs hardware (per location) ==")
    for p in (0.027, 0.048, 0.01):
        print(f"p = {p}: 1q gate 0.1p = {0.1 * p:.4f}; 2q gate p = {p:.4f}; "
              f"measurement flip 5p = {5 * p:.3f}; idle per measurement "
              f"layer 2p = {2 * p:.3f}; reset flip 2p = {2 * p:.3f}")
    print("hardware: 2q CZdag Weyl error 0.027, CZ 0.048 (Goss22); qutrit "
          "Clifford error 2.4e-3 (Morvan21), 5.4e-3 (Liu23); 3-state readout "
          "error 0.022 (Krinner22 mean), 0.031 (Chen23); multi-level reset "
          "error ~1e-3 (McEwen21); idle per 500 ns readout layer: see table 1")

    print("\n== 4. SI1000 sequential-vs-parallel idle budget per round ==")
    print("sequential (repo): data idle locations per round = 4 measurement "
          "layers x 2p + reset 2p = 10p; parallel: 1 x 2p + 2p = 4p; "
          "2q-gate depolarizing marginal on a data qutrit = 0.9p per gate "
          "(72 of 80 pairs act non-trivially on it)")
    deg = {0: 2, 1: 4, 2: 2, 3: 2, 4: 2}
    for q, n in deg.items():
        print(f"  data {q}: gate marginal {0.9 * n:.1f}p, idle sequential "
              f"10p, idle parallel 4p")


if __name__ == "__main__":
    main()
