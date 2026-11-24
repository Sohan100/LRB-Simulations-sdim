"""Mirror-inverse (production) vs compiled-recovery (alternate) pseudo-thresholds.

Every LRB and RB fit includes depth m = 0 (user rule, 2026-10-31). Each
protocol uses its production grid for p < split and its high-p grid for
p >= split (SD6 0.025, SI1000 0.015). Estimators:

  official   mirror only: comparisons/pseudo-thresholds-current.csv, the
             production estimator (per-sequence |.| mean, 1/std^2 weights,
             monotone-trim threshold), quoted as is;
  pipeline   compiled only: the same per-sequence |.| statistic and weighted
             offset-0 fit on results/LRB/<p>/unif_check_data/<k>.csv, simple
             last-crossing threshold, depths that are multiples of k;
  pooled     both protocols: src/lrb/pooled_lrb_fits.py (pooled signed
             polarization, binomial weights, >= 1000 accepted shots per depth,
             depths that are multiples of k, sequence bootstrap).

Mirror rates are per forward+inverse pair against physical RB per pair;
compiled rates are per Clifford plus check round against compiled physical RB
per Clifford. Writes results/compiled_vs_mirror_thresholds.csv and
results/compiled_vs_mirror_rates.csv.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # studies/non-markovian -> repo root
sys.path.insert(0, os.path.join(REPO, "src"))

from lrb.pooled_lrb_fits import build_pooled_tables, fit_offset0_decay, pseudo_threshold, read_run_params, rb_rate  # noqa: E402

DATA = os.path.join(REPO, "LRB-experiment-data-slurm")
CONFIGS = {
    # config: (split, mirror prod, mirror high-p, compiled prod, compiled high-p)
    "one-ancilla-si1000": (0.015,
                           "Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140",
                           "Run-2026-10-30-10-01-06-folded_qutrit-single-ancilla-si1000-sdim140-highp-ext",
                           "Run-2026-10-31-00-39-01-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery",
                           "Run-2026-10-31-00-45-00-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery-highp"),
    "one-ancilla-sd6": (0.025,
                        "Run-2026-10-28-10-24-46-folded_qutrit-single-ancilla-sd6-sdim140",
                        "Run-2026-10-30-05-58-58-folded_qutrit-single-ancilla-sd6-sdim140-highp-ext",
                        "Run-2026-10-31-00-40-37-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery",
                        "Run-2026-10-31-00-45-57-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery-highp"),
    "two-ancilla-si1000": (0.015,
                           "Run-2026-10-28-10-26-06-folded_qutrit_split_unif-split-ancilla-si1000-sdim140",
                           "Run-2026-10-30-10-01-28-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-highp-ext",
                           "Run-2026-10-31-00-41-58-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery",
                           "Run-2026-10-31-00-45-27-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery-highp"),
    "two-ancilla-sd6": (0.025,
                        "Run-2026-10-28-10-27-37-folded_qutrit_split_unif-split-ancilla-sd6-sdim140",
                        "Run-2026-10-30-05-59-13-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-highp-ext",
                        "Run-2026-10-31-00-43-31-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery",
                        "Run-2026-10-31-00-46-19-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery-highp"),
}
CHECKS = (1, 2, 3, 4)


def official_thresholds() -> dict:
    out = {}
    with open(os.path.join(DATA, "comparisons", "pseudo-thresholds-current.csv")) as fh:
        for row in csv.DictReader(fh):
            out[(row["configuration"], int(row["interval"]))] = (float(row["p_star"]), float(row["ci68_low"]),
                                                                 float(row["ci68_high"]), row["status"])
    return out


def pipeline_rate(run: str, p_idx: int, k: int) -> float:
    """Production statistic and fit on the run's stats CSV, depths that are multiples of k."""
    path = os.path.join(run, "results", "LRB", str(p_idx), "unif_check_data", f"{k}.csv")
    if not os.path.exists(path):
        return np.nan
    rows = [line.strip().split(",") for line in open(path) if line.strip()]
    mean = np.asarray([float(v) for v in rows[1][1:]])
    std = np.asarray([float(v) for v in rows[2][1:]])
    depths = read_run_params(run).depths
    use = np.isfinite(mean) & np.isfinite(std) & (mean <= 1.0) & (depths % k == 0)
    fit = fit_offset0_decay(depths[use], mean[use], std[use])
    return fit["r"] if fit else np.nan


def merged_points(prod: str, highp: str, split: float):
    for run, keep in ((prod, lambda p: 0 < p < split), (highp, lambda p: p >= split)):
        for i, p in enumerate(read_run_params(run).probs):
            if keep(p):
                yield run, i, p


def main():
    official = official_thresholds()
    thr_rows, rate_rows = [], []
    for cfg, (split, m_prod, m_hp, c_prod, c_hp) in CONFIGS.items():
        paths = [os.path.join(DATA, r) for r in (m_prod, m_hp, c_prod, c_hp)]
        _, m_thr = build_pooled_tables(paths[0], checks=CHECKS, highp_run=paths[1], split=split, n_boot=100)
        c_rows, c_thr = build_pooled_tables(paths[2], checks=CHECKS, highp_run=paths[3], split=split, n_boot=100)
        m_rows, _ = build_pooled_tables(paths[0], checks=(1,), highp_run=paths[1], split=split, n_boot=0)
        for k in CHECKS:
            pts = list(merged_points(paths[2], paths[3], split))
            p_pipe, note = pseudo_threshold([p for _, _, p in pts], [pipeline_rate(r, i, k) for r, i, _ in pts],
                                            [rb_rate(r, i, read_run_params(r).depths) for r, i, _ in pts])
            off = official.get((cfg, k), (np.nan, np.nan, np.nan, ""))
            mt = next(t for t in m_thr if t["k"] == k)
            ct = next(t for t in c_thr if t["k"] == k)
            thr_rows.append({
                "config": cfg, "k": k,
                "mirror_official_p_star": off[0], "mirror_official_ci68": f"[{off[1]:.4f}, {off[2]:.4f}]",
                "mirror_pooled_p_star": mt["p_star_r_L"],
                "mirror_pooled_ci68": f"[{mt['ci68_low']:.4f}, {mt['ci68_high']:.4f}]",
                "compiled_pipeline_p_star": p_pipe, "compiled_pipeline_note": note,
                "compiled_pooled_p_star": ct["p_star_r_L"],
                "compiled_pooled_ci68": f"[{ct['ci68_low']:.4f}, {ct['ci68_high']:.4f}]",
                "compiled_pooled_note": ct["note"],
            })
            print(cfg, k, {c: (round(v, 4) if isinstance(v, float) else v) for c, v in thr_rows[-1].items()
                           if c.endswith("p_star")}, flush=True)
        mir = {r["p"]: r for r in m_rows}
        for r in c_rows:
            if r["k"] != 1 or r["p"] not in mir:
                continue
            rate_rows.append({
                "config": cfg, "p": r["p"],
                "mirror_r_per_pair_pooled": mir[r["p"]]["r_L"], "mirror_rb_per_pair": mir[r["p"]]["r_RB"],
                "mirror_chi2_dof": mir[r["p"]]["chi2_dof"], "mirror_window_ratio": mir[r["p"]]["window_ratio"],
                "compiled_r_L_per_clifford": r["r_L"], "compiled_r_L_se": r["r_L_se"],
                "compiled_rb_per_clifford": r["r_RB"], "compiled_d_L": r["d_L"],
                "compiled_chi2_dof": r["chi2_dof"], "compiled_window_ratio": r["window_ratio"],
                "pair_over_cycle": mir[r["p"]]["r_L"] / r["r_L"] if r["r_L"] else np.nan,
            })

    def dump(name, rows):
        cols = list(dict.fromkeys(c for row in rows for c in row))
        with open(os.path.join(HERE, "results", name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in rows:
                w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in row.items()})

    dump("compiled_vs_mirror_thresholds.csv", thr_rows)
    dump("compiled_vs_mirror_rates.csv", rate_rows)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
