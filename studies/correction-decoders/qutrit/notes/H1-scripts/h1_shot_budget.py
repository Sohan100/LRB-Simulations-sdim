"""Shot budgets for a hardware LRB run of the [[5,1,2]]_3 folded code.

Estimator: F_hat = |mean(w^k)| over the N_acc accepted shots of one circuit,
w = exp(2 pi i/3). With P(0) = q and the error mass split evenly between
k = 1, 2 (no preferred direction), Re(w^k) has mean F = (3q - 1)/2 and
variance q + (1 - q)/4 - F^2, so SE(F_hat) ~ sqrt(var / N_acc) (the
imaginary part only adds an O(1/N) bias). The fit y_m = b f^m is weighted
least squares with Fisher information from those SEs; r = 1 - (1 + 2f)/3.

Part 0 checks the variance formula against the production baseline: the
reported std over 30 Cliffords at p = 0.005 (unif 1) should match the shot
noise of the accepted shots if between-sequence variation is small.
Acceptance/fidelity inputs for parts 1-2 are the HW-C (projected) parallel
readout numbers from h1_dem_readout_schedule.py (4 sequences, 2e4 shots).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

BASE = Path("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
            "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-"
            "si1000/results/LRB")
DEPTHS_BASE = [0, 2, 4, 6, 10, 14, 18, 20, 22]


def var_re(F):
    q = (2 * F + 1) / 3
    return q + (1 - q) / 4 - F ** 2


def read_rows(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    return {r[0]: [float(x) for x in r[1:]] for r in rows}


def fit_sigma_f(depths, b, f, se):
    """Std of f from weighted LS of y = b f^m with per-depth SE."""
    depths = np.asarray(depths, float)
    J = np.stack([f ** depths, b * depths * f ** (depths - 1)], axis=1)
    W = np.diag(1 / np.asarray(se) ** 2)
    cov = np.linalg.inv(J.T @ W @ J)
    return float(np.sqrt(cov[1, 1]))


def main():
    print("== 0. Baseline check: std over 30 Cliffords vs shot noise ==")
    rows = read_rows(BASE / "6" / "unif_check_data" / "1.csv")
    fids = rows["Fidelity averages"]
    stds = rows["Fidelity Standard Deviations"]
    rej = rows["Rejected Runs"]
    for m, F, s, r in zip(DEPTHS_BASE, fids, stds, rej):
        if np.isnan(F):
            continue
        n_acc = 1e6 * (1 - r)
        pred = np.sqrt(var_re(F) / n_acc) if n_acc > 0 else np.nan
        print(f"  m={m:2d}: F={F:.4f} accepted/circuit={n_acc:9.0f} "
              f"reported std={s:.2e} shot-noise prediction={pred:.2e}")

    print("\n== 1. Per-circuit and per-depth precision (K=30 sequences) ==")
    depths = [0, 2, 6, 10]
    acc_unif1 = [0.913, 0.597, 0.248, 0.104]   # HW-C parallel, unif 1
    fid_unif1 = [0.9822, 0.9808, 0.9811, 0.9829]
    fid_raw = [0.9617, 0.7796, 0.5006, 0.3170]  # HW-C parallel, no rejection
    K = 30
    for N in (1e4, 3e4, 1e5):
        se_u = [np.sqrt(var_re(F) / (K * N * a))
                for F, a in zip(fid_unif1, acc_unif1)]
        se_r = [np.sqrt(var_re(F) / (K * N)) for F in fid_raw]
        print(f"  N={N:.0e} shots/circuit: SE of sequence-averaged fidelity "
              f"unif1 {['%.1e' % s for s in se_u]}, raw {['%.1e' % s for s in se_r]}")
        f_raw = (fid_raw[-1] / fid_raw[0]) ** (1 / depths[-1])
        sf = fit_sigma_f(depths, fid_raw[0], f_raw, se_r)
        print(f"    raw-decay fit: f={f_raw:.4f} +- {sf:.1e}  ->  "
              f"r={(2 / 3) * (1 - f_raw):.4f} +- {(2 / 3) * sf:.1e}")

    print("\n== 2. Wall-clock budget ==")
    n_depths_full = 9
    for N in (1e4, 3e4, 1e5):
        shots = K * n_depths_full * N
        for t_rep in (100e-6, 300e-6, 1e-3):
            print(f"  K={K}, 9 depths, N={N:.0e}: {shots:.1e} shots; at "
                  f"{t_rep * 1e6:.0f} us/shot -> {shots * t_rep / 3600:.2f} h")

    print("\n== 3. Detector-correlation (DEM) learning: rough shot need ==")
    for pij in (1e-3, 1e-2):
        for rel in (0.1, 0.3):
            n = pij / (rel * pij) ** 2
            print(f"  edge probability {pij:.0e} to {int(rel * 100)}% relative "
                  f"precision: ~{n:.1e} shots (Poisson count of joint events)")


if __name__ == "__main__":
    main()
