"""Semi-analytic k = 1 pseudo-threshold versus q (no sampling).

Near threshold the pipeline fit rests on depths 0 and 1, so the fitted decay is
f = lambda_1 / lambda_0 and r_LRB = (q-1)/q (1 - f).  Model of the accepted
polarization at depth m (D_m = 4(m+1) detectors):

  accepted shots = clean shots (no fault visible on its own, probability C_m)
                 + random shots (fully scrambled, pass by chance: q^-D_m)
  lambda_m = c_m (1 - q/(q-1) eps_m),   c_m = C_m / (C_m + q^-D_m),
  formed per random sequence and then averaged over sequences, as the
  pipeline averages per-sequence polarizations,

with eps_1 - eps_0 = s(q) p^2 the two-fault silent-logical term per depth
step (exact low-p coefficient from 20_pair_expansion.py; power-law
extrapolation in q beyond q = 13).  C_m(p) is exact from the DEM (product over
mechanisms of 1 - pi_i(p)(1 - z_i)), averaged over random sequences.
Physical RB is exact: r_RB = (q-1)/q [1 - E_C lambda^(2 n(C))], lambda =
1 - s p q^2/(q^2-1), over the full catalogue word-length distribution.

Outputs per (q, noise, compile): p*_model (both terms), p*_rand (random term
only, eps = 0) and p*_pair (pair term only, c_m = 1).
Writes results/model_pstar.csv and results/model_curves.csv.gz.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import qslib

P_GRID = np.round(np.arange(0.001, 0.2001, 0.0005), 6)


def rb_error(q, noise, mode, p):
    """Exact physical RB error per forward+inverse pair (pipeline unit)."""
    s = 0.1 if noise == "si1000" else 1.0
    lam = 1.0 - s * p * q * q / (q * q - 1.0)
    sym = qslib.symplectic_words(q, "HS" if mode == "production" else "HShs")
    sym_len = np.array([0 if w == "I" else len(w) for w in sym])
    if mode == "production":
        pa = np.arange(q)
    else:
        pa = np.array([min(a, q - a) for a in range(q)])
    pauli_len = (pa[:, None] + pa[None, :]).reshape(-1)
    # E over independent Pauli and symplectic parts (identity word "I" = 1 token)
    e_sym = np.mean(lam ** (2 * sym_len))
    e_pau = np.mean(lam ** (2 * pauli_len))
    # correction for the single word "I" (length 1 instead of 0)
    n_tot = q * q * len(sym)
    e = e_sym * e_pau + (lam ** 2 - 1.0) / n_tot
    return (q - 1) / q * (1 - e)


def clean_probs(args):
    q, noise, mode, seed = args
    import importlib
    ra = importlib.import_module("22_random_acceptance")
    words = qslib.random_words(q, mode, 1, seed)
    c1, v1, _ = ra.lam_terms(q, noise, mode, words, 1)
    P = P_GRID[:, None]
    return np.exp(np.sum(np.log1p(-np.minimum(c1[None, :] * P, 1.0) * v1[None, :]), axis=1))


def crossing(p, a, b):
    """First p where a(p) >= b(p) (linear interpolation)."""
    diff = a - b
    idx = np.where(diff >= 0)[0]
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return p[0]
    t = diff[i - 1] / (diff[i - 1] - diff[i])
    return p[i - 1] + t * (p[i] - p[i - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="2,3,5,7,11,13,17,19,23,29,31")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compiles", default="production,inverse")
    a = ap.parse_args()
    import importlib
    ra = importlib.import_module("22_random_acceptance")
    pairs = pd.read_csv("results/pair_expansion_main.csv.gz" if not qslib.ROUND_SUFFIX else "results/pair_expansion_five.csv.gz")
    pairs = pairs[pairs.w >= 0]
    step = (pairs.groupby(["noise", "compile", "q", "depth"]).S2b.sum().unstack("depth").fillna(0) / 10)
    step = ((step[4] - step[1]) / 3).rename("s").reset_index()
    rows, curves = [], []
    qs = [int(x) for x in a.q.split(",")]
    for noise in ("si1000", "sd6"):
        for mode in a.compiles.split(","):
            st = step[(step.noise == noise) & (step["compile"] == mode)]
            fit_q = st[st.q >= 5]
            slope, icpt = np.polyfit(np.log(fit_q.q), np.log(fit_q.s), 1)
            for q in qs:
                if q == 2 and mode == "inverse":
                    continue
                c0w, v0w, _ = ra.lam_terms(q, noise, mode, [], 0)
                C0 = np.exp(np.sum(np.log1p(-np.minimum(c0w[None, :] * P_GRID[:, None], 1.0)
                                           * v0w[None, :]), axis=1))
                jobs = [(q, noise, mode, 1000 * q + s) for s in range(a.n)]
                with ProcessPoolExecutor(max_workers=a.workers) as ex:
                    C1_seq = np.array(list(ex.map(clean_probs, jobs)))  # (n_seq, n_p)
                C1 = C1_seq.mean(axis=0)
                row_s = st[st.q == q]
                s_q = float(row_s.s.iloc[0]) if len(row_s) else float(np.exp(icpt) * q ** slope)
                c0 = C0 / (C0 + float(q) ** -4)
                # the pipeline averages per-sequence polarizations, so the
                # clean fraction is formed per sequence and then averaged
                c1 = np.mean(C1_seq / (C1_seq + float(q) ** -8), axis=0)
                eps = s_q * P_GRID ** 2
                pol_ratio = c1 / c0 * np.clip(1 - q / (q - 1) * eps, 0, 1)
                r_model = (q - 1) / q * (1 - pol_ratio)
                r_rand = (q - 1) / q * (1 - c1 / c0)
                r_pair = (q - 1) / q * (1 - np.clip(1 - q / (q - 1) * eps, 0, 1))
                r_rb = np.array([rb_error(q, noise, mode, p) for p in P_GRID])
                rows.append(dict(q=q, noise=noise, compile=mode, nbar=qslib.mean_tokens_exact(q, mode),
                                 s_step=s_q, s_from_fit=not len(row_s),
                                 pstar_model=crossing(P_GRID, r_model, r_rb),
                                 pstar_rand=crossing(P_GRID, r_rand, r_rb),
                                 pstar_pair=crossing(P_GRID, r_pair, r_rb),
                                 A1_model_at_pstar=np.nan))
                i = np.searchsorted(P_GRID, rows[-1]["pstar_model"]) if np.isfinite(rows[-1]["pstar_model"]) else None
                if i is not None and i < len(P_GRID):
                    rows[-1]["A1_model_at_pstar"] = float(C1[i] + q ** -8.0)
                    rows[-1]["step_acceptance_at_pstar"] = float((C1[i] + q ** -8.0) / (C0[i] + q ** -4.0))
                for j, p in enumerate(P_GRID):
                    curves.append(dict(q=q, noise=noise, compile=mode, p=p, r_model=r_model[j],
                                       r_rand=r_rand[j], r_pair=r_pair[j], r_rb=r_rb[j],
                                       C0=C0[j], C1=C1[j]))
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()},
                      flush=True)
    pd.DataFrame(rows).to_csv(f"results/model_pstar{qslib.ROUND_SUFFIX}.csv", index=False)
    pd.DataFrame(curves).to_csv(f"results/model_curves{qslib.ROUND_SUFFIX}.csv.gz", index=False)


if __name__ == "__main__":
    main()
