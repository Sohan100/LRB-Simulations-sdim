"""Bias of each estimator on exact (infinite-shot, infinite-sequence) data.

For every (code, noise, p) the exact sequence-averaged outcome distribution
of the terminal-check protocol is computed at a set of depths (transfer
matrix, tcsteps) and handed to the estimators as if it were data with 1e12
shots per depth. The reference values are exact too:

  eps_pair  logical infidelity of one mirror pair between a perfect
            preparation and a perfect check (random frame)
  beta      its pair-coincidence part: quadratic coefficient of the
            SPAM-free infidelity, from the exact asymptotic mode rates

Scenarios: mirror or compiled closing, ideal or noisy SPAM (generator's
noisy encoder and readout), production depths or a depth grid designed from
the discard rate. Writes results/estimator_bias.csv.
"""
from __future__ import annotations

import csv
import os

import numpy as np

import tclib as T
import tcfit as F
import tcsteps as S
from tcref import design_depths, reference

PROD_DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
PS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]


def main():
    rows = []
    for code in ("qubit", "qutrit"):
        for noise in ("si1000", "sd6"):
            sm = S.StepModel(code, noise)
            d = sm.d
            md = F.modes(d)
            for p in PS:
                gt = sm.ground_truth(p)
                beta_pair = F.derived(sm.asymptotic_rates(p, "mirror")[md.reps], d)["beta"]
                beta_single = F.derived(sm.asymptotic_rates(p, "forward")[md.reps], d)["beta"]
                for protocol in ("mirror", "compiled"):
                    unit_true = gt["pair_stat"] if protocol == "mirror" else gt["single_stat"]
                    unit_beta = beta_pair if protocol == "mirror" else beta_single
                    unit_co = reference(code, noise, p, protocol)["coincidence"]
                    disc = gt["pair_stat_discard"] if protocol == "mirror" else gt["single_stat_discard"]
                    grids = {"designed": design_depths(disc)}
                    if protocol == "mirror":
                        grids["production"] = PROD_DEPTHS
                    for spam_name, spam in (("ideal", None), ("noisy", sm.spam(p))):
                        for gname, depths in grids.items():
                            ex = sm.exact_curve(p, depths, protocol, spam)
                            cnt = ex["P"] * 1e12
                            # x = number of noisy depth units; compiled: m forward + 1 recovery Clifford
                            x = np.array(depths, float)
                            if protocol == "compiled":
                                x = np.where(x > 0, x + 1, 0.0)
                            se = F.single_exponential(x, cnt, d, min_accepted=1e9)
                            dq = F.direct_quadratic(x, cnt, d)
                            mf = F.mode_fit(x, cnt, d, "free")
                            mb = F.mode_fit(x, cnt, d, "block")
                            rows.append({
                                "code": code, "noise": noise, "p": p, "protocol": protocol, "spam": spam_name,
                                "depths": gname, "m_max": max(depths), "n_depths": len(depths),
                                "accept_m0": float(ex["A"][0]), "accept_mmax": float(ex["A"][-1]),
                                "true_unit_infidelity": unit_true, "true_beta": unit_beta,
                                "coincidence_reference": unit_co, "f_cov": unit_true / unit_co,
                                "single_exp_r": se["r"], "direct_c2": dq["c2"],
                                "mode_free_eps1": mf["eps1"], "mode_free_beta": mf["beta"],
                                "mode_block_beta": mb["eps1"],
                                "ratio_single_exp": se["r"] / unit_true,
                                "ratio_direct_c2_to_beta": dq["c2"] / unit_beta,
                                "ratio_free_eps1": mf["eps1"] / unit_true,
                                "ratio_block_to_beta": mb["eps1"] / unit_beta,
                                "ratio_block_to_coincidence": mb["eps1"] / unit_co,
                                "ratio_block_to_true": mb["eps1"] / unit_true,
                            })
    path = os.path.join(T.HERE, "results", "estimator_bias.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), "rows")
    for r in rows:
        if r["p"] in (2e-3, 1e-2) and r["spam"] == "noisy":
            print(f"{r['code']:6s} {r['noise']:6s} p={r['p']:<6g} {r['protocol']:8s} {r['depths']:10s} mmax={r['m_max']:<3d} "
                  f"A0={r['accept_m0']:.2f} Amax={r['accept_mmax']:.2f} | 1exp/true={r['ratio_single_exp']:6.2f} "
                  f"c2/beta={r['ratio_direct_c2_to_beta']:5.2f} free/true={r['ratio_free_eps1']:5.3f} "
                  f"block/co={r['ratio_block_to_coincidence']:5.3f} block/true={r['ratio_block_to_true']:5.3f} F={r['f_cov']:.3f}")


if __name__ == "__main__":
    main()
