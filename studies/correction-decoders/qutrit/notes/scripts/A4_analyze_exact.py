#!/usr/bin/env python3
"""A4: analysis of the exact LRB-D results written by A4_exact_lrbd.py.

Prints
  1. validation of the exact values against the per-sequence shot counts of
     the baseline run (z-scores restricted to cells with >= 1000 accepted
     shots, where the normal approximation holds);
  2. small-p expansion coefficients, per depth m (mean over sequences):
        -ln F(m; p) = a1(m) p + a2(m) p^2 + O(p^3)
        -ln A(m; p) = c1(m) p + c2(m) p^2 + O(p^3)
     and least-squares fits a2(m) = alpha + beta m + gamma m^2 (m >= 2);
  3. exact-curve shape diagnostics at p = 1e-3 and 5e-3: depth-windowed
     rates, curvature of ln F, single-exponential misfit, for detection and
     for the hybrid (postselected intermediate rounds, ML-decoded final
     round).

    module load python/3.11
    python .../A4_analyze_exact.py <path to A4_exact.npz>
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import A4_exact_lrbd as X  # noqa: E402


def zscores_masked(A_pred, F_pred, accc, rejc, min_acc=1000):
    A_obs, F_obs, n_acc, tot = X.observed(accc, rejc)
    ok = (A_pred * tot >= min_acc) & (n_acc >= min_acc) & (F_pred < 1)
    zA = (A_obs - A_pred) / np.sqrt(A_pred * (1 - A_pred) / tot)
    q = (2.0 / 3.0) * (1 - F_pred)
    var = (1 - q) + q / 4 - F_pred ** 2
    zF = (F_obs - F_pred) / np.sqrt(np.maximum(var, 1e-300) /
                                    np.maximum(n_acc, 1))
    return zA[ok], zF[ok]


def poly_coeffs(p, y, deg=3):
    """Fit y(p) = sum_{k=1..deg} c_k p^k (no constant); return c."""
    A = np.stack([p ** k for k in range(1, deg + 1)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def main(path: str) -> None:
    z = np.load(path)
    A, F, Adec, Fdec = z["A"], z["F"], z["Adec"], z["Fdec"]
    As, Fs = z["As"], z["Fs"]
    A0, F0, F0dec, A0s, F0s = z["A0"], z["F0"], z["F0dec"], z["A0s"], z["F0s"]
    seqs, depths = z["seqs"], z["depths"]
    p_eval, p_small = z["p_eval"], z["p_small"]
    pols = [str(s) for s in z["policies"]]
    polt = X.POLICIES
    print(f"sequences {list(seqs)}; depths {list(depths)}; policies {pols}")

    # ---------------------------------------------------------------- 1
    print("\n1. Validation (cells with >= 1000 accepted shots)")
    for p_val in (1e-3, 5e-3, 1e-2, 2e-2):
        k = list(p_eval).index(p_val)
        p_idx = X.PROBS.index(p_val)
        zA_all, zF_all = [], []
        for pi, pol in enumerate(polt):
            accc, rejc = X.load_counts(p_idx, pol)
            zA, zF = zscores_masked(A[pi, :, :, k].T, F[pi, :, :, k].T,
                                    accc[:, seqs, :], rejc[:, seqs])
            zA_all.append(zA), zF_all.append(zF)
        accc, rejc = X.load_counts(p_idx, ("const", 0))
        zA, zF = zscores_masked(A0[:, :, k].T, F0[:, :, k].T,
                                accc[:, seqs, :], rejc[:, seqs])
        zA_all.append(zA), zF_all.append(zF)
        zA_all, zF_all = np.concatenate(zA_all), np.concatenate(zF_all)
        print(f"  p={p_val:6.4f}: acceptance z mean {zA_all.mean():+.3f} sd "
              f"{zA_all.std():.3f} max|z| {np.abs(zA_all).max():.2f} (n="
              f"{zA_all.size}); fidelity z mean {zF_all.mean():+.3f} sd "
              f"{zF_all.std():.3f} max|z| {np.abs(zF_all).max():.2f} (n="
              f"{zF_all.size})")
    print("  (cells are counted once per policy; policies that use the same"
          " rounds share shots, so the z's are not independent)")

    # ---------------------------------------------------------------- 2
    print("\n2. Small-p coefficients (mean over sequences; m = depth)")
    ms = depths.astype(float)
    sel = ms >= 2
    for pi, lab in list(enumerate(pols)) + [(-1, "const0")]:
        lnF = -np.log(F0s if pi < 0 else Fs[pi])        # (S, D, P)
        lnA = -np.log(A0s if pi < 0 else As[pi])
        a = np.array([[poly_coeffs(p_small, lnF[s, d]) for d in
                       range(len(depths))] for s in range(len(seqs))])
        c = np.array([[poly_coeffs(p_small, lnA[s, d]) for d in
                       range(len(depths))] for s in range(len(seqs))])
        a1, a2 = a[..., 0].mean(0), a[..., 1].mean(0)
        c1, c2 = c[..., 0].mean(0), c[..., 1].mean(0)
        Q = np.stack([np.ones(sel.sum()), ms[sel], ms[sel] ** 2], 1)
        g, *_ = np.linalg.lstsq(Q, a2[sel], rcond=None)
        h, *_ = np.linalg.lstsq(Q[:, :2], c1[sel], rcond=None)
        print(f"  {lab:7s} a1(m) = {np.array2string(a1, precision=3)}")
        print(f"          a2(m) = {np.array2string(a2, precision=1)}")
        print(f"          a2 fit alpha={g[0]:.1f} beta={g[1]:.2f} "
              f"gamma={g[2]:.3f}  (gamma m^2/beta m at m=22: "
              f"{g[2] * 22 / g[1] if g[1] else np.inf:.2f})")
        print(f"          c1(m) = {np.array2string(c1, precision=2)}; "
              f"linear fit c1 = {h[0]:.2f} + {h[1]:.3f} m;  c2(m=22) = "
              f"{c2[-1]:.1f}")

    # ---------------------------------------------------------------- 3
    print("\n3. Exact-curve shape at p = 1e-3 and 5e-3 (mean over sequences"
          " of per-sequence curves)")
    for p_val in (1e-3, 5e-3):
        k = list(p_eval).index(p_val)
        print(f"  p = {p_val}")
        print("   policy   F(2)      F(22)     rate 2->4  rate 20->22 "
              "curv d [b e^{-cm-dm^2}]  hybrid F(2) F(22) rate2->4 rate20->22"
              "  acc_det(22) acc_hyb(22)")
        for pi, lab in list(enumerate(pols)) + [(-1, "const0")]:
            Fm = (F0 if pi < 0 else F[pi])[:, :, k].mean(0)
            Fh = (F0dec if pi < 0 else Fdec[pi])[:, :, k].mean(0)
            Am = (A0 if pi < 0 else A[pi])[:, :, k].mean(0)
            Ah = (np.ones_like(Am) if pi < 0 else Adec[pi][:, :, k].mean(0))
            r24 = (Fm[2] / Fm[1]) ** 0.5
            r2022 = (Fm[8] / Fm[7]) ** 0.5
            h24 = (Fh[2] / Fh[1]) ** 0.5
            h2022 = (Fh[8] / Fh[7]) ** 0.5
            Q = np.stack([np.ones(sel.sum()), ms[sel], ms[sel] ** 2], 1)
            beta, *_ = np.linalg.lstsq(Q, np.log(Fm[sel]), rcond=None)
            print(f"   {lab:7s} {Fm[1]:.6f}  {Fm[8]:.6f}  {r24:.6f}   "
                  f"{r2022:.6f}    {-beta[2]:+.2e}            "
                  f"{Fh[1]:.5f} {Fh[8]:.5f} {h24:.5f} {h2022:.5f}   "
                  f"{Am[8]:.4f}     {Ah[8]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("A4_SCRATCH", "/tmp"), "A4_exact.npz"))
