"""Characterize the terminal-check (const-0) decay across p, noise model and code.

Everything is exact (no sampling): the outcome distribution of every circuit
is the product of its DEM characters, and averages over uniformly random
sequences use the frame transfer matrix (tcsteps). The sampled production
tallies are read only for the overlay in the figures.

Writes
  results/characterization.csv   one row per (code, noise, p)
  results/curves_<code>_<noise>.npz  exact curves on depths 0..40 and the
                                     production points with bootstrap errors
"""
from __future__ import annotations

import csv
import os

import numpy as np

import tclib as T
import tcfit as F
import tcsteps as S

PROD_DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
DENSE = list(range(0, 41))


def official(run, col):
    """p -> value from the run's official unif-1 fit table (per forward+inverse pair)."""
    path = os.path.join(T.MAIN_DATA, run, "results", "plots", "unif-1-lrb-vs-rb-table-mixed-fits.csv")
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[round(float(row["p"]), 10)] = float(row[col])
    return out


def production_points(key, p_idx, d, n_boot=300, seed=1):
    base = os.path.join(T.MAIN_DATA, T.RUNS[key], "progress", str(p_idx), "const_check_data")
    tal = np.load(os.path.join(base, "0.npy")).astype(float)        # depth x seq x d
    rej = np.load(os.path.join(base, "0_rejected.npy")).astype(float)

    def stat(idx):
        t, r = tal[:, idx].sum(1), rej[:, idx].sum(1)
        n_acc = t.sum(1)
        lam = (d * t[:, 0] / n_acc - 1) / (d - 1)
        return (d - 1) / d * (1 - lam), n_acc / (n_acc + r)

    eps, acc = stat(np.arange(tal.shape[1]))
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.integers(0, tal.shape[1], tal.shape[1]))[0] for _ in range(n_boot)])
    return eps, bs.std(0), acc


def main():
    rows = []
    for key in T.RUNS:
        code, noise = key
        sm = S.StepModel(code, noise)
        d = sm.d
        md = F.modes(d)
        run = os.path.join(T.MAIN_DATA, T.RUNS[key])
        probs = [float(x) for x in open(os.path.join(run, "probs.txt")).read().split(",")]
        rb = official(T.RUNS[key], "rb_error_rate")
        u1 = official(T.RUNS[key], "lrb_error_rate")
        store = {"dense": np.array(DENSE), "prod_depths": np.array(PROD_DEPTHS), "probs": np.array(probs)}
        for pi, p in enumerate(probs):
            if p == 0:
                continue
            gt = sm.ground_truth(p)
            gam = sm.asymptotic_rates(p)[md.reps]
            der = F.derived(gam, d)
            ex = sm.exact_curve(p, DENSE)
            exp_ = sm.exact_curve(p, PROD_DEPTHS)
            se = F.single_exponential(PROD_DEPTHS, exp_["P"] * 1e6 * 30, d, min_accepted=30)
            eps_prod, eps_prod_se, acc_prod = production_points(key, pi, d)
            store[f"eps_{pi}"] = ex["eps"]
            store[f"A_{pi}"] = ex["A"]
            store[f"chi_{pi}"] = ex["chi"]
            store[f"prod_eps_{pi}"] = eps_prod
            store[f"prod_eps_se_{pi}"] = eps_prod_se
            store[f"prod_A_{pi}"] = acc_prod
            store[f"fit1_{pi}"] = np.array([se["b"], se["f"]])
            store[f"gamma_{pi}"] = gam
            r_rb = rb.get(round(p, 10), np.nan)
            rows.append({
                "code": code, "noise": noise, "p": p,
                "eps_pair_true": gt["pair_stat"],            # one mirror pair, perfect prep and check
                "eps_pair_true_Xframe": gt["pair_X"],        # = depth 1 of the simulated protocol
                "eps_single_true": gt["single_stat"],        # one Clifford, perfect prep and check
                "alpha": der["alpha"], "beta": der["beta"],
                "alpha_plus_beta": der["alpha"] + der["beta"],
                "eps1_from_rates": der["eps1"],
                "cov_factor": (der["alpha"] + der["beta"]) / der["beta"],
                "discard_per_pair": gt["pair_stat_discard"],
                "eps_m22": float(exp_["eps"][-1]), "accept_m22": float(exp_["A"][-1]),
                "single_exp_r_prod_depths": se["r"], "single_exp_chi2_per_dof": se["chi2"] / max(se["dof"], 1),
                "rb_pair_official": r_rb,
                "unif1_pair_official": u1.get(round(p, 10), np.nan),
                "gain_pair": r_rb / gt["pair_stat"] if r_rb == r_rb else np.nan,
                "breakeven_depth": r_rb / der["beta"] if r_rb == r_rb else np.nan,
            })
        np.savez_compressed(os.path.join(T.HERE, "results", f"curves_{code}_{noise}.npz"), **store)
    path = os.path.join(T.HERE, "results", "characterization.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['code']:6s} {r['noise']:6s} p={r['p']:<7g} pair={r['eps_pair_true']:.3e} beta={r['beta']:.3e} "
              f"cov={r['cov_factor']:.3f} disc={r['discard_per_pair']:.3f} 1exp={r['single_exp_r_prod_depths']:.3e} "
              f"(chi2/dof {r['single_exp_chi2_per_dof']:.0f}) RB={r['rb_pair_official']:.3e} unif1={r['unif1_pair_official']:.3e} "
              f"gain={r['gain_pair']:.1f} m*={r['breakeven_depth']:.0f} A22={r['accept_m22']:.3f}")


if __name__ == "__main__":
    main()
