"""Print the Markdown tables of REPORT.md from the files in results/."""
from __future__ import annotations

import csv
import os

import numpy as np

import tclib as T
from tcref import reference

RES = os.path.join(T.HERE, "results")


def rows(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def f(r, k):
    try:
        return float(r[k])
    except (ValueError, TypeError):
        return float("nan")


def e(x, nd=2):
    if not np.isfinite(x):
        return "n/a"
    m, ex = f"{x:.{nd - 1}e}".split("e")
    return f"{m}e{int(ex)}"


def table_characterization():
    print("\n### Table A. What the terminal check measures (exact)\n")
    print("| code | noise | p | logical pair, truth | coincidence part | F_cov | discard per pair | physical RB per pair | "
          "RB / logical | break-even depth | unif-1 per pair (noisy checks) | single-exp fit of const-0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    ch = rows("characterization.csv")
    for r in ch:
        p = f(r, "p")
        if p not in (1e-3, 2e-3, 5e-3, 1e-2):
            continue
        ref = reference(r["code"], r["noise"], p)
        print(f"| {r['code']} | {r['noise'].upper()} | {p:g} | {e(ref['true_unit'])} | {e(ref['coincidence'])} | "
              f"{ref['f_cov']:.2f} | {100 * f(r, 'discard_per_pair'):.1f}% | {e(f(r, 'rb_pair_official'))} | "
              f"{f(r, 'rb_pair_official') / ref['true_unit']:.0f} | {f(r, 'rb_pair_official') / ref['coincidence']:.0f} | "
              f"{e(f(r, 'unif1_pair_official'))} | {e(f(r, 'single_exp_r_prod_depths'))} |")


def table_bias():
    print("\n### Table B. Estimators on exact data (infinite shots and sequences), noisy SPAM\n")
    print("| code | noise | closing | depth grid | single exp / truth | direct c2 / truth | free fit / truth | "
          "block fit / coincidence | block fit x F_cov / truth |")
    print("|---|---|---|---|---|---|---|---|---|")
    b = [r for r in rows("estimator_bias.csv") if r["spam"] == "noisy" and f(r, "p") <= 1e-2]
    keys = sorted({(r["code"], r["noise"], r["protocol"], r["depths"]) for r in b})
    for k in keys:
        sel = [r for r in b if (r["code"], r["noise"], r["protocol"], r["depths"]) == k]

        def rng(col, fn=lambda r, c: f(r, c), fmt="{:.2f}"):
            v = np.array([fn(r, col) for r in sel])
            v = v[np.isfinite(v)]
            if len(v) == 0:
                return "n/a"
            return f"{fmt.format(v.min())} to {fmt.format(v.max())}"

        print(f"| {k[0]} | {k[1].upper()} | {k[2]} | {k[3]} | {rng('ratio_single_exp', fmt='{:.0f}')} | "
              f"{rng('direct_c2', lambda r, c: f(r, c) / f(r, 'true_unit_infidelity'))} | {rng('ratio_free_eps1', fmt='{:.3f}')} | "
              f"{rng('ratio_block_to_coincidence', fmt='{:.3f}')} | "
              f"{rng('mode_block_beta', lambda r, c: f(r, c) * f(r, 'f_cov') / f(r, 'true_unit_infidelity'), '{:.3f}')} |")


def table_shots():
    print("\n### Table C. Standard error at finite shots (200 synthetic experiments per row), mirror closing, noisy SPAM\n")
    print("| code | p | sequences | shots per circuit | depths (max) | total shots | block fit: mean / ref | "
          "block fit: SE | bootstrap SE | free fit: mean / truth | free fit: SE |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    fs = [r for r in rows("finite_shots.csv") if r["protocol"] == "mirror" and r["noise"] == "si1000"
          and r["grid"] == "designed" and f(r, "reach") == 1.0 and r["nested"] == "True" and f(r, "meas_scale") == 1.0]
    for r in fs:
        if int(r["shots"]) not in (100, 1000, 10000):
            continue
        boot = f"{100 * f(r, 'block_boot_se_over_ref'):.1f}%" if np.isfinite(f(r, "block_boot_se_over_ref")) else ""
        print(f"| {r['code']} | {f(r, 'p'):g} | {r['n_seq']} | {r['shots']} | {r['n_depths']} ({r['m_max']}) | "
              f"{e(f(r, 'total_shots'))} | {f(r, 'block_mean_over_ref'):.3f} | {100 * f(r, 'block_sd_over_ref'):.1f}% | {boot} | "
              f"{f(r, 'free_eps1_mean_over_ref'):.2f} | {100 * f(r, 'free_eps1_sd_over_ref'):.0f}% |")
    print("\n### Table D. Design variants at p = 0.005 (SI1000 unless noted), 30 sequences, 1000 shots per circuit\n")
    print("| code | variant | depths (max) | total shots | block fit: mean / ref | block fit: SE | free fit: SE |")
    print("|---|---|---|---|---|---|---|")
    allr = rows("finite_shots.csv")
    for r in allr:
        if int(r["shots"]) != 1000 or int(r["n_seq"]) != 30:
            continue
        base = (r["protocol"] == "mirror" and r["noise"] == "si1000" and r["grid"] == "designed"
                and f(r, "reach") == 1.0 and r["nested"] == "True" and f(r, "meas_scale") == 1.0)
        if base and abs(f(r, "p") - 5e-3) > 1e-12:
            continue
        if base:
            name = "baseline: mirror, nested prefixes, designed grid"
        elif r["nested"] == "False":
            name = "independent sequences at every depth"
        elif r["protocol"] == "compiled":
            name = "compiled recovery Clifford (x = number of Cliffords)"
        elif r["grid"] == "production":
            name = "production depths 0, 2, ..., 22"
        elif f(r, "reach") != 1.0:
            name = f"designed grid, reach {f(r, 'reach'):g}"
        elif f(r, "meas_scale") != 1.0:
            name = "readout flip 20 p instead of 5 p"
        elif r["noise"] == "sd6":
            name = "SD6, p = 0.002"
        else:
            continue
        print(f"| {r['code']} | {name} | {r['n_depths']} ({r['m_max']}) | {e(f(r, 'total_shots'))} | "
              f"{f(r, 'block_mean_over_ref'):.3f} | {100 * f(r, 'block_sd_over_ref'):.1f}% | {100 * f(r, 'free_eps1_sd_over_ref'):.0f}% |")


def table_resample():
    print("\n### Table E. Production const-0 circuits re-sampled with the full record (30 sequences x 1M shots)\n")
    print("| code | noise | p | truth (all sequences) | block fit x F_cov | free fit | single-exponential fit | "
          "discard per pair: fit / truth |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows("production_resample.csv"):
        print(f"| {r['code']} | {r['noise'].upper()} | {f(r, 'p'):g} | {e(f(r, 'true_pair'), 3)} | "
              f"{e(f(r, 'block_times_fcov'), 3)} ± {e(f(r, 'block_times_fcov_se'), 1)} | "
              f"{e(f(r, 'free_eps1'), 3)} ± {e(f(r, 'free_eps1_se'), 1)} | {e(f(r, 'single_exp_r'), 3)} ± {e(f(r, 'single_exp_r_se'), 1)} | "
              f"{100 * f(r, 'discard1'):.2f}% ± {100 * f(r, 'discard1_se'):.2f}% / {100 * f(r, 'true_discard_pair'):.2f}% |")


def table_robust():
    print("\n### Table F. Correlated faults on the logical support {0, 2} (SI1000, p = 0.005)\n")
    print("| code | correlated-error probability per Clifford | truth / truth without it | block fit x F_cov / truth | "
          "free fit / truth | detected at 2 sigma, 30 seq x 1000 shots | detected, 30 seq x 10000 shots |")
    print("|---|---|---|---|---|---|---|")
    for r in rows("robustness_crosstalk.csv"):
        print(f"| {r['code']} | {f(r, 'xt_per_clifford'):g} | {f(r, 'true_pair_over_no_xt'):.2f} | "
              f"{f(r, 'block_times_fcov_over_true'):.2f} | {f(r, 'free_eps1_over_true'):.2f} | "
              f"{100 * f(r, 'detect_frac_2sigma_1000'):.0f}% | {100 * f(r, 'detect_frac_2sigma_10000'):.0f}% |")
    print("\n### Table G. Quasi-static drift of p between shots (log-normal, SI1000, mean p = 0.005)\n")
    print("| code | relative spread | truth / truth without drift | block fit x F_cov / truth | free fit / truth | "
          "observed / predicted accepted infidelity at mid depth |")
    print("|---|---|---|---|---|---|")
    for r in rows("robustness_drift.csv"):
        print(f"| {r['code']} | {f(r, 'sigma_rel'):g} | {f(r, 'true_pair_over_no_drift'):.2f} | "
              f"{f(r, 'block_times_fcov_over_true'):.2f} | {f(r, 'free_eps1_over_true'):.2f} | "
              f"{f(r, 'observed_over_block_predicted_eps_at_mid'):.3f} |")


if __name__ == "__main__":
    table_characterization()
    table_bias()
    table_shots()
    table_resample()
    table_robust()
