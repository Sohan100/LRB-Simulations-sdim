"""Two ways hardware noise can break the model behind the block estimate.

(a) Correlated logical faults. A two-qudit depolarizing error is injected on
    data qudits 0 and 2 (the support of the weight-2 logical Z) after every
    noisy Clifford, with probability ``xt``. Such faults are undetectable
    logical errors at first order. The block model cannot represent them; the
    free mode fit can (its linear coefficient alpha).

(b) Quasi-static drift. p is constant within a shot and log-normally
    distributed between shots (relative width sigma). Faults on different
    qudits are then positively correlated between shots, so coincidences are
    more frequent than the product of the marginal rates.

For each case the script reports the exact truth (logical infidelity of one
mirror pair between a perfect preparation and a perfect check, pooled over
the shot ensemble), the estimates from exact data, and for (a) the detection
power of the free fit's alpha at finite shots.

Writes results/robustness_crosstalk.csv and results/robustness_drift.csv.
"""
from __future__ import annotations

import csv
import os
import numpy as np

import tclib as T
import tcfit as F
import tcsteps as S

from tcref import design_depths, reference
P0 = 5e-3


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def crosstalk():
    rows = []
    for code in ("qubit", "qutrit"):
        sm = S.StepModel(code, "si1000", crosstalk=((0, 2),))
        d = sm.d
        F.modes(d)  # fills the mode cache
        gt0 = sm.ground_truth(P0)
        fcov = reference(code, "si1000", P0)["f_cov"]
        depths = design_depths(gt0["pair_stat_discard"])
        spam = dict(prep=True, meas=True, meas_scale=1.0)
        # null distribution of alpha-hat at two shot budgets
        null = {}
        for shots in (1000, 10000):
            rng = np.random.default_rng(7 + shots)
            a = []
            for _ in range(150):
                P = sm.experiment_dists(P0, 30, depths, rng, spam=spam)
                a.append(F.mode_fit(depths, sm.sample_counts(P, shots, rng), d, "free")["alpha"])
            null[shots] = (float(np.mean(a)), float(np.std(a)))
        for xt in (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
            gt = sm.ground_truth(P0, xt)
            ex = sm.exact_curve(P0, depths, spam=sm.spam(P0), xt=xt)
            mb = F.mode_fit(depths, ex["P"] * 1e12, d, "block")
            mf = F.mode_fit(depths, ex["P"] * 1e12, d, "free")
            row = {"code": code, "p": P0, "xt_per_clifford": xt, "true_pair": gt["pair_stat"],
                   "true_pair_over_no_xt": gt["pair_stat"] / gt0["pair_stat"],
                   "block_beta_exact": mb["eps1"], "block_times_fcov_over_true": mb["eps1"] * fcov / gt["pair_stat"],
                   "free_eps1_over_true": mf["eps1"] / gt["pair_stat"], "free_alpha_exact": mf["alpha"],
                   "pearson_block_per_1e6_shots": None, "f_cov": fcov}
            for shots in (1000, 10000):
                rng = np.random.default_rng(11 + shots)
                hits = 0
                n = 60
                for _ in range(n):
                    P = sm.experiment_dists(P0, 30, depths, rng, spam=spam, xt=xt)
                    al = F.mode_fit(depths, sm.sample_counts(P, shots, rng), d, "free")["alpha"]
                    hits += (al - null[shots][0]) / null[shots][1] > 2.0
                row[f"alpha_null_sd_{shots}"] = null[shots][1]
                row[f"detect_frac_2sigma_{shots}"] = hits / n
            row["total_shots_1000"] = 1000 * 30 * len(depths)
            row.pop("pearson_block_per_1e6_shots")
            rows.append(row)
            print(row, flush=True)
    write(os.path.join(T.HERE, "results", "robustness_crosstalk.csv"), rows)


def drift():
    rows = []
    z, w = np.polynomial.hermite_e.hermegauss(11)
    w = w / w.sum()
    for code in ("qubit", "qutrit"):
        sm = S.StepModel(code, "si1000")
        d = sm.d
        md = F.modes(d)
        gt0 = sm.ground_truth(P0)
        fcov = reference(code, "si1000", P0)["f_cov"]
        depths = design_depths(gt0["pair_stat_discard"])
        for sigma in (0.0, 0.1, 0.2, 0.3, 0.5):
            pj = P0 * np.exp(sigma * z - sigma ** 2 / 2)
            chi_pair = sum(wi * sm.factors(p, "mirror").mean((0, 1)) for wi, p in zip(w, pj))
            true_pair = sm._eps_of(chi_pair)
            chi = sum(wi * sm.exact_chars(p, depths, "mirror", sm.spam(p)) for wi, p in zip(w, pj))
            P = np.clip(T.dist_from_chars(chi, d), 0, None)
            cnt = P * 1e12
            mb = F.mode_fit(depths, cnt, d, "block")
            mf = F.mode_fit(depths, cnt, d, "free")
            # curvature of the log mode decays: ratio of late to early decay rate of the slowest S mode
            st = F.accepted_stats(cnt, d)
            # model-predicted accepted infidelity at the deepest depth vs the exact one
            gam = mb["gamma"]
            chi_fit = np.exp(mb["logB"][None, :] - np.outer(depths, gam))
            Pfit = 1.0 / d ** 3 + md.expand(chi_fit)[:, 1:] @ md.F[:, 1:].T
            _, _, lam_fit = T.accept_and_polarization(Pfit, d)
            eps_fit = T.infidelity_from_polarization(lam_fit, d)
            rows.append({"code": code, "p_mean": P0, "sigma_rel": sigma,
                         "true_pair": true_pair, "true_pair_over_no_drift": true_pair / gt0["pair_stat"],
                         "block_times_fcov_over_true": mb["eps1"] * fcov / true_pair,
                         "free_eps1_over_true": mf["eps1"] / true_pair,
                         "observed_over_block_predicted_eps_at_mmax": float(st["eps"][-1] / eps_fit[-1]),
                         "observed_over_block_predicted_eps_at_mid": float(st["eps"][len(depths) // 2] / eps_fit[len(depths) // 2])})
            print(rows[-1], flush=True)
    write(os.path.join(T.HERE, "results", "robustness_drift.csv"), rows)


if __name__ == "__main__":
    drift()
    crosstalk()
