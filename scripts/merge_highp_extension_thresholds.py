"""Pseudo-thresholds from a production run merged with a high-p extension run.

At high p with frequent checks almost every shot is rejected after depth 2, so
the production depth grid (0, 2, 4, ...) leaves two-point LRB fits there. An
extension run with odd depths and more shots replaces the production rows at
p >= --split-p; production rows below it are kept. Fit and threshold settings
match ``notebooks/04-qutrit-one-and-relay-two-ancilla-si1000-sd6.ipynb``. Outputs go to a separate
folder, so the production run's plots and tables are untouched.

Usage (from the repository root):

    .venv-sdim140/bin/python scripts/merge_highp_extension_thresholds.py \\
        <production-run> <extension-run> <output-label>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lrb.lrb_plotting import (  # noqa: E402
    LRBPlotFitConfig,
    LRBResultsPlotter,
    LRBThresholdConfig,
)

RUNS_ROOT = PROJECT_ROOT / "LRB-experiment-data-slurm"
OUT_ROOT = RUNS_ROOT / "comparisons" / "sdim140-highp-merged"
TABLE_NAME = "unif_lrb_vs_rb_table_all_mixed_fits.csv"

FIT_CONFIG = LRBPlotFitConfig(
    a_fixed=0.0,
    y_ceiling=1.0,
    min_fit_points_lrb=2,
    min_fit_points_rb=3,
    f_min=1e-8,
    f_max=1.0,
    rb_a_fixed=0.0,
    p_axis_scale=1.0,
    p_axis_label="Two-qudit gate error rate $p$",
    p_axis_symbol="p",
    p_axis_column=None,
)
THRESHOLD_CONFIG = LRBThresholdConfig(
    tol=5e-4,
    require_consecutive=2,
    ignore_first_n=3,
    err_floor=2e-3,
    tail_min_prefix=6,
    tail_drop_tol_abs=0.01,
    tail_drop_tol_rel=0.25,
    p_min=None,
    p_max=None,
    min_lrb_n_points_keep=0,
    min_rb_n_points_keep=0,
    zoom_half_window_points=4,
    zoom_pad_frac=0.25,
    min_zoom_span=0.01,
    p_window_before_points=8,
    p_window_after_points=5,
    min_points_in_view=6,
    view_pad_frac=0.15,
    bootstrap_reps_error=500,
    bootstrap_reps_threshold=1000,
    bootstrap_ci_level=0.68,
    bootstrap_use_sem=True,
    bootstrap_seed=12345,
    paper_mode=True,
    paper_cols=2,
    paper_panel_width_in=3.2,
    paper_panel_height_in=2.5,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("production_run")
    parser.add_argument("extension_run")
    parser.add_argument("label")
    parser.add_argument("--split-p", type=float, default=0.0225,
                        help="Extension rows replace production rows at "
                             "p >= this value.")
    parser.add_argument("--check-max", type=int, default=4)
    args = parser.parse_args()

    production = LRBResultsPlotter(
        working_folder=str(RUNS_ROOT / args.production_run) + "/",
        fit_config=FIT_CONFIG)
    extension = LRBResultsPlotter(
        working_folder=str(RUNS_ROOT / args.extension_run) + "/",
        fit_config=FIT_CONFIG)

    extension_table = extension.build_unif_lrb_vs_rb_table_mixed_fits(
        write_per_check_tables=True,
        check_min=1,
        check_max=args.check_max,
        bootstrap_reps=THRESHOLD_CONFIG.bootstrap_reps_error,
        bootstrap_ci_level=THRESHOLD_CONFIG.bootstrap_ci_level,
        bootstrap_use_sem=THRESHOLD_CONFIG.bootstrap_use_sem,
        bootstrap_seed=THRESHOLD_CONFIG.bootstrap_seed,
    )
    prod = pd.read_csv(Path(production.out_dir) / TABLE_NAME)
    ext = pd.read_csv(extension_table)
    checks = range(1, args.check_max + 1)
    prod = prod[prod["check_num"].isin(checks)]
    ext = ext[ext["check_num"].isin(checks)]
    merged = pd.concat([
        prod[prod["p"] < args.split_p].assign(source="production"),
        ext[ext["p"] >= args.split_p].assign(source="extension"),
    ], ignore_index=True).sort_values(["check_num", "p"])

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = out_dir / "unif_lrb_vs_rb_table_merged.csv"
    merged.to_csv(merged_csv, index=False)

    # Cross-check at p values both runs cover: the fits differ in depth grid
    # and shot count, so large disagreements flag fit-range sensitivity.
    overlap = prod.merge(ext, on=["check_num", "p"],
                         suffixes=("_prod", "_ext"))
    overlap[["check_num", "p", "lrb_error_rate_prod", "lrb_n_points_prod",
             "lrb_error_rate_ext", "lrb_n_points_ext", "rb_error_rate_prod",
             "rb_error_rate_ext"]].to_csv(out_dir / "overlap_check.csv",
                                          index=False)

    production.out_dir = str(out_dir)
    summary_csv = production.plot_all_unif_threshold_graphs(
        threshold_config=THRESHOLD_CONFIG,
        table_csv_path=str(merged_csv),
        show=False,
        check_min=1,
        check_max=args.check_max,
    )
    production.plot_unif_pseudo_thresholds_vs_interval_check(
        check_min=1,
        check_max=args.check_max,
        summary_csv_path=summary_csv,
        do_fit=False,
        paper_mode=True,
        show=False,
    )
    summary = pd.read_csv(summary_csv)
    print(f"Merged table: {merged_csv}")
    print(f"Summary: {summary_csv}")
    print(summary[["check_num", "threshold_p", "threshold_p_ci_low",
                   "threshold_p_ci_high", "method"]].to_string(index=False))
    print(overlap[["check_num", "p", "lrb_error_rate_prod",
                   "lrb_n_points_prod", "lrb_error_rate_ext",
                   "lrb_n_points_ext"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
