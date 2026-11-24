"""Provisional pseudo-thresholds from one production run (production rows only).

Builds the run's LRB-vs-RB table with the notebook/merge settings into
comparisons/<out>/<label>/ (the run's own results/ folder is untouched), runs the
threshold estimator, and prints p*(k) with the number of LRB fit points near the
crossing, to decide where extension or refinement runs are needed.
Usage: provisional_thresholds.py OUT LABEL RUN [scale-free]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_highp_threshold_extension as M  # noqa: E402

out, label, run = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = M.SCALE_FREE_THRESHOLD_CONFIG if len(sys.argv) > 4 and sys.argv[4] == "scale-free" else M.THRESHOLD_CONFIG
out_dir = M.COMPARISONS / out / label
table = M.build_table(M.plotter_for(run), out_dir / "production-table")
table = table[table["check_num"].between(M.CHECK_MIN, M.CHECK_MAX)]
csv = out_dir / "unif_lrb_vs_rb_table_production.csv"
table.to_csv(csv, index=False)
plotter = M.plotter_for(run)
plotter.out_dir = str(out_dir)
summary = pd.read_csv(plotter.plot_all_unif_threshold_graphs(
    threshold_config=cfg, table_csv_path=str(csv), show=False,
    check_min=M.CHECK_MIN, check_max=M.CHECK_MAX))
print(f"=== {label} ({run})")
for _, row in summary.iterrows():
    k = int(row.check_num)
    sub = table[table.check_num == k].sort_values("p")
    near = sub.iloc[(sub["p"] - row.threshold_p).abs().argsort()[:3]].sort_values("p")
    pts = ", ".join(f"p={r.p:.4g}: LRB {r.lrb_error_rate:.4f} (n={r.lrb_n_points:.0f}) RB {r.rb_error_rate:.4f}"
                    for r in near.itertuples())
    print(f"  k={k}: p*={row.threshold_p:.4f} [{row.threshold_p_ci_low:.4f}, {row.threshold_p_ci_high:.4f}] "
          f"{row.method} | {pts}")
