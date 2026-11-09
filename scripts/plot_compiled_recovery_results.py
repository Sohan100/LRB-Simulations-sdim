"""Plot the compiled-recovery runs with the production analysis pipeline.

The compiled-recovery runs (`--inverse-mode compiled`) close each sequence
with one recovery Clifford instead of the mirror inverse. This script gives
them the same figures and thresholds as the production (mirror) runs:

- const=0 and uniform-interval 1..4 summary plots (circuit-level p panels);
- LRB-vs-RB tables, merged with the high-p extension at the production
  split points, pseudo-threshold grids and p* versus interval.

Same fit and bootstrap settings as the notebook, with depth 0 in every fit.
For checks every k >= 2 Cliffords only even depths are fitted (odd depths
carry an extra unchecked Clifford), the same rule as the production merge.
Everything is written under LRB-experiment-data-slurm/comparisons/
compiled-recovery/<label>/; the run folders are not modified.

Usage (repo root):
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python \\
        scripts/plot_compiled_recovery_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_highp_threshold_extension as merge  # noqa: E402
from lrb.lrb_plotting import LRBPaperPlotConfig  # noqa: E402

OUT_ROOT = merge.COMPARISONS / "compiled-recovery"
CHECK_MIN, CHECK_MAX = merge.CHECK_MIN, merge.CHECK_MAX
# label -> (compiled run, compiled high-p extension, p_split)
RUNS = {
    "one-ancilla-si1000": (
        "Run-2026-10-31-00-39-01-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery",
        "Run-2026-10-31-00-45-00-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery-highp",
        0.015,
    ),
    "two-ancilla-si1000": (
        "Run-2026-10-31-00-41-58-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery",
        "Run-2026-10-31-00-45-27-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery-highp",
        0.015,
    ),
    "one-ancilla-sd6": (
        "Run-2026-10-31-00-40-37-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery",
        "Run-2026-10-31-00-45-57-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery-highp",
        0.025,
    ),
    "two-ancilla-sd6": (
        "Run-2026-10-31-00-43-31-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery",
        "Run-2026-10-31-00-46-19-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery-highp",
        0.025,
    ),
}

# Same settings as `Visualize LRB DEM SI1000 SD6 Results.ipynb`.
PAPER_CONFIG = LRBPaperPlotConfig(
    enabled=True,
    column_layout="double",
    rows_per_page=3,
    pairs_per_row=2,
    num_prob_samples=6,
    prob_sample_indices=(1, 3, 4, 5, 6, 8),
    row_height_in=1.6,
    panel_title_fontsize=7.0,
    axis_label_fontsize=6.6,
    tick_fontsize=6.0,
    legend_fontsize=5.7,
    line_width=0.85,
    marker_size=2.6,
    errorbar_capsize=1.2,
    show_legend_first_row_only=False,
    compact_fit_legend=True,
    legend_on_rejected_panel=True,
)


class CompiledRunPlotter(merge.EvenDepthsForSparseChecks):
    """Apply the even-depth rule to the summary plots as well."""

    def _read_pair(self, check_type, check_num, prob_index):
        ok_lrb, lrb, ok_rb, rb = super()._read_pair(
            check_type, check_num, prob_index)
        if ok_lrb and check_type == "unif" and int(check_num) >= 2:
            first, fidelity, rejection = lrb
            fidelity = [
                entry if entry is None or depth % 2 == 0
                else dict(entry, mean=float("nan"), std=float("nan"))
                for depth, entry in zip(self.depths, fidelity)
            ]
            lrb = (first, fidelity, rejection)
        return ok_lrb, lrb, ok_rb, rb


def main() -> int:
    for label, (run, highp, p_split) in RUNS.items():
        print(f"=== {label}")
        out_dir = OUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        plotter = merge.plotter_for(run, CompiledRunPlotter)
        plotter.out_dir = str(out_dir)
        plotter.plot_all_const_checks(
            show=False, check_min=0, check_max=0, paper_config=PAPER_CONFIG)
        plotter.plot_all_unif_checks(
            show=False, check_min=CHECK_MIN, check_max=CHECK_MAX,
            paper_config=PAPER_CONFIG)

        production = merge.build_table(
            merge.plotter_for(run, CompiledRunPlotter),
            out_dir / "standard-table")
        extension = merge.build_table(
            merge.plotter_for(highp, CompiledRunPlotter),
            out_dir / "extension-table")
        merge.overlap_report(production, extension, p_split)
        merged = pd.concat(
            [production[production["p"] < p_split],
             extension[extension["p"] >= p_split]],
            ignore_index=True,
        ).sort_values(["check_num", "p"]).reset_index(drop=True)
        merged["source"] = np.where(merged["p"] < p_split, run, highp)
        merged_csv = out_dir / "unif_lrb_vs_rb_table_merged.csv"
        merged.to_csv(merged_csv, index=False)

        summary_csv = plotter.plot_all_unif_threshold_graphs(
            threshold_config=merge.THRESHOLD_CONFIG,
            table_csv_path=str(merged_csv),
            show=False,
            check_min=CHECK_MIN,
            check_max=CHECK_MAX,
        )
        plotter.plot_unif_pseudo_thresholds_vs_interval_check(
            check_min=CHECK_MIN,
            check_max=CHECK_MAX,
            summary_csv_path=str(summary_csv),
            do_fit=False,
            fit_model="exp",
            fit_degree=1,
            paper_mode=True,
            show=False,
        )
        for _, row in pd.read_csv(summary_csv).iterrows():
            print(f"  k={int(row.check_num)}: p*={row.threshold_p:.4f} "
                  f"[{row.threshold_p_ci_low:.4f}, {row.threshold_p_ci_high:.4f}]"
                  f" {row.method}")
        print(f"  outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
