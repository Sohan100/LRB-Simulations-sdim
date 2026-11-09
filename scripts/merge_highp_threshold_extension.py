"""Merge high-p extension runs into production threshold tables.

At high p with checks every one or two Cliffords, the production runs
(depths 0..22, 1M shots) keep shots only at depths 0 and 2, so the LRB fits
have two points and some pseudo-thresholds have no clean crossing. Extension
runs re-simulate the high-p range at depths 0,1,2,3,4,6 with 20M shots. This
script builds each extension's LRB-vs-RB table with the notebook's fit and
bootstrap settings, takes production rows for p < p_split and extension rows
for p >= p_split, and reruns the notebook's threshold estimator on the merged
table. Outputs go to LRB-experiment-data-slurm/comparisons/<out>/<label>/ so
the production plots are left untouched.

For checks every k >= 2 Cliffords, extension fits use even depths only.
Depths that are not multiples of k carry an extra unchecked Clifford, so the
odd depths 1 and 3 sit on a different curve (see
non-markovian-brainstorm/REPORT.md). Production depths are all even, so
this keeps the two datasets' depth structure the same; k = 1 uses every
depth.

It then rewrites CURRENT_SUMMARY (.md and .csv): every pseudo-threshold of
the four production configurations, taken from the merged result when its
extension is complete and from the production run otherwise, with its status.
That file is the hand-off to whoever writes the results up.

Usage (repo root):
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python \
        scripts/merge_highp_threshold_extension.py [sd6] [si1000]
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python \
        scripts/merge_highp_threshold_extension.py qubit-sd6 qubit-si1000
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python \
        scripts/merge_highp_threshold_extension.py conv-sd6 conv-si1000
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python \
        scripts/merge_highp_threshold_extension.py qubit-conv-sd6 qubit-conv-si1000
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lrb.lrb_plotting import (  # noqa: E402
    LRBPlotFitConfig,
    LRBResultsPlotter,
    LRBThresholdConfig,
)

RUNS_ROOT = PROJECT_ROOT / "LRB-experiment-data-slurm"
COMPARISONS = RUNS_ROOT / "comparisons"
CURRENT_SUMMARY = COMPARISONS / "pseudo-thresholds-current"
# The qubit campaign has its own hand-off file so the qutrit one is never
# rewritten by a qubit merge.
QUBIT_CURRENT_SUMMARY = COMPARISONS / "pseudo-thresholds-qubit-current"
# The conventional-site two-ancilla campaigns (data relay on the four
# conventional ancilla sites) get their own hand-off files too.
CONV_CURRENT_SUMMARY = COMPARISONS / "pseudo-thresholds-conv-sites-current"
QUBIT_CONV_CURRENT_SUMMARY = (
    COMPARISONS / "pseudo-thresholds-qubit-conv-sites-current")
CHECK_MIN, CHECK_MAX = 1, 4
# Same settings as `Visualize LRB DEM SI1000 SD6 Results.ipynb`.
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
# The qubit crossings sit where LRB and RB errors are about 1e-3. There the
# absolute tol (5e-4) and err_floor (2e-3) above dominate: the floor starts
# the search above the true crossing and the estimator returns the floor
# point. The qubit campaigns therefore use the scale-free rule, p* where the
# LRB error equals the RB error, linearly interpolated (tol 0, no floor). On
# the qutrit merged tables this rule reproduces 15 of the 16 published p*
# exactly and the 16th (two-ancilla SI1000, k=3) within 0.0002.
SCALE_FREE_THRESHOLD_CONFIG = dataclasses.replace(
    THRESHOLD_CONFIG, tol=0.0, err_floor=None)

# Each campaign: output folder, the p at which extension rows take over, and
# label -> (production run, extension run).
CAMPAIGNS = {
    "sd6": {
        "out": COMPARISONS / "sdim140-sd6-highp-merged",
        "p_split": 0.025,
        "pairs": {
            "one-ancilla-sd6": (
                "Run-2026-10-28-10-24-46-folded_qutrit-single-ancilla-sd6-sdim140",
                "Run-2026-10-30-05-58-58-folded_qutrit-single-ancilla-sd6-sdim140-highp-ext",
            ),
            "two-ancilla-sd6": (
                "Run-2026-10-28-10-27-37-folded_qutrit_split_unif-split-ancilla-sd6-sdim140",
                "Run-2026-10-30-05-59-13-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-highp-ext",
            ),
        },
    },
    "si1000": {
        "out": COMPARISONS / "sdim140-si1000-highp-merged",
        "p_split": 0.015,
        "pairs": {
            "one-ancilla-si1000": (
                "Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140",
                "Run-2026-10-30-10-01-06-folded_qutrit-single-ancilla-si1000-sdim140-highp-ext",
            ),
            "two-ancilla-si1000": (
                "Run-2026-10-28-10-26-06-folded_qutrit_split_unif-split-ancilla-si1000-sdim140",
                "Run-2026-10-30-10-01-28-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-highp-ext",
            ),
        },
    },
    # Qubit [[5,1,2]] counterpart. Its crossings sit at p = 0.003-0.021,
    # below the high-p extension ranges, so the "extension" here is a
    # near-threshold refinement: production settings (depths 0..22, 1M
    # shots, 30 new sequences) on a fine p grid, used inside its own window
    # p_split <= p <= p_max.
    "qubit-sd6": {
        "out": COMPARISONS / "sdim140-qubit-sd6-refine-merged",
        "p_split": 0.006,
        "p_max": 0.026,
        "threshold_config": SCALE_FREE_THRESHOLD_CONFIG,
        "pairs": {
            "qubit-one-ancilla-sd6": (
                "Run-2026-10-31-10-13-03-folded_qubit-single-ancilla-sd6-sdim140",
                "Run-2026-10-31-11-09-31-folded_qubit-single-ancilla-sd6-sdim140-threshold-refine",
            ),
            "qubit-two-ancilla-sd6": (
                "Run-2026-10-31-10-13-07-folded_qubit_split_unif-split-ancilla-sd6-sdim140",
                "Run-2026-10-31-11-09-33-folded_qubit_split_unif-split-ancilla-sd6-sdim140-threshold-refine",
            ),
        },
    },
    "qubit-si1000": {
        "out": COMPARISONS / "sdim140-qubit-si1000-refine-merged",
        "p_split": 0.002,
        "p_max": 0.013,
        "threshold_config": SCALE_FREE_THRESHOLD_CONFIG,
        "pairs": {
            "qubit-one-ancilla-si1000": (
                "Run-2026-10-31-10-12-55-folded_qubit-single-ancilla-si1000-sdim140",
                "Run-2026-10-31-11-09-27-folded_qubit-single-ancilla-si1000-sdim140-threshold-refine",
            ),
            "qubit-two-ancilla-si1000": (
                "Run-2026-10-31-10-12-58-folded_qubit_split_unif-split-ancilla-si1000-sdim140",
                "Run-2026-10-31-11-09-29-folded_qubit_split_unif-split-ancilla-si1000-sdim140-threshold-refine",
            ),
        },
    },
    # Conventional-site two-ancilla layout, same grids and merge rules as the
    # campaigns above.
    "conv-sd6": {
        "out": COMPARISONS / "sdim140-conv-sites-sd6-highp-merged",
        "p_split": 0.025,
        "pairs": {
            "conv-two-ancilla-sd6": (
                "Run-2026-11-01-09-24-10-folded_qutrit_conv_sites-conv-sites-sd6-sdim140",
                "Run-2026-11-01-09-26-20-folded_qutrit_conv_sites-conv-sites-sd6-sdim140-highp-ext",
            ),
        },
    },
    "conv-si1000": {
        "out": COMPARISONS / "sdim140-conv-sites-si1000-highp-merged",
        "p_split": 0.015,
        "pairs": {
            "conv-two-ancilla-si1000": (
                "Run-2026-11-01-09-22-32-folded_qutrit_conv_sites-conv-sites-si1000-sdim140",
                "Run-2026-11-01-09-25-50-folded_qutrit_conv_sites-conv-sites-si1000-sdim140-highp-ext",
            ),
        },
    },
    # Its k=4 crossing fell below the 0.006-0.026 refinement used for the
    # coherent relay (0.0040 when interpolated over the coarse production
    # grid), so this refinement starts at 0.003 (0.003-0.0055 in steps of
    # 0.0005, then 0.006-0.026 in steps of 0.001); on it the crossing is
    # 0.0047. The first refinement,
    # Run-2026-11-01-09-31-12-...-threshold-refine, is superseded.
    "qubit-conv-sd6": {
        "out": COMPARISONS / "sdim140-qubit-conv-sites-sd6-refine-merged",
        "p_split": 0.003,
        "p_max": 0.026,
        "threshold_config": SCALE_FREE_THRESHOLD_CONFIG,
        "pairs": {
            "qubit-conv-two-ancilla-sd6": (
                "Run-2026-11-01-09-28-00-folded_qubit_conv_sites-conv-sites-sd6-sdim140",
                "Run-2026-11-02-01-15-57-folded_qubit_conv_sites-conv-sites-sd6-sdim140-threshold-refine-wide",
            ),
        },
    },
    "qubit-conv-si1000": {
        "out": COMPARISONS / "sdim140-qubit-conv-sites-si1000-refine-merged",
        "p_split": 0.002,
        "p_max": 0.013,
        "threshold_config": SCALE_FREE_THRESHOLD_CONFIG,
        "pairs": {
            "qubit-conv-two-ancilla-si1000": (
                "Run-2026-11-01-09-26-44-folded_qubit_conv_sites-conv-sites-si1000-sdim140",
                "Run-2026-11-01-09-29-15-folded_qubit_conv_sites-conv-sites-si1000-sdim140-threshold-refine",
            ),
        },
    },
}
QUTRIT_CAMPAIGNS = ("sd6", "si1000")
QUBIT_CAMPAIGNS = ("qubit-sd6", "qubit-si1000")
CONV_CAMPAIGNS = ("conv-sd6", "conv-si1000")
QUBIT_CONV_CAMPAIGNS = ("qubit-conv-sd6", "qubit-conv-si1000")
# family -> (campaigns, hand-off file, qubit arithmetic and refine wording)
FAMILIES = {
    "qutrit": (QUTRIT_CAMPAIGNS, CURRENT_SUMMARY, False),
    "qubit": (QUBIT_CAMPAIGNS, QUBIT_CURRENT_SUMMARY, True),
    "conv": (CONV_CAMPAIGNS, CONV_CURRENT_SUMMARY, False),
    "qubit-conv": (QUBIT_CONV_CAMPAIGNS, QUBIT_CONV_CURRENT_SUMMARY, True),
}


class EvenDepthsForSparseChecks(LRBResultsPlotter):
    """Mask odd depths out of uniform-check stats for checks every k >= 2."""

    def _read_fidelity_stats_file(self, path: str):
        stats = LRBResultsPlotter._read_fidelity_stats_file(path)
        parts = Path(path).parts
        if stats is None or "unif_check_data" not in parts:
            return stats
        if int(Path(path).stem) < 2:
            return stats
        return [
            entry if entry is None or depth % 2 == 0
            else dict(entry, mean=float("nan"), std=float("nan"))
            for depth, entry in zip(self.depths, stats)
        ]


def plotter_for(run: str,
                plotter_class: type = LRBResultsPlotter) -> LRBResultsPlotter:
    return plotter_class(
        working_folder=str(RUNS_ROOT / run) + "/", fit_config=FIT_CONFIG)


def build_table(plotter: LRBResultsPlotter, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.out_dir = str(out_dir)
    path = plotter.build_unif_lrb_vs_rb_table_mixed_fits(
        write_per_check_tables=False,
        check_min=CHECK_MIN,
        check_max=CHECK_MAX,
        bootstrap_reps=THRESHOLD_CONFIG.bootstrap_reps_error,
        bootstrap_ci_level=THRESHOLD_CONFIG.bootstrap_ci_level,
        bootstrap_use_sem=THRESHOLD_CONFIG.bootstrap_use_sem,
        bootstrap_seed=THRESHOLD_CONFIG.bootstrap_seed,
    )
    return pd.read_csv(path)


def from_extension(p, p_split: float, p_max: float = np.inf):
    """True where the extension supplies the row: p_split <= p <= p_max.

    High-p extensions have no p_max. The qubit near-threshold refinements
    replace production only inside their own p window.
    """
    return (p >= p_split) & (p <= p_max)


def overlap_report(production: pd.DataFrame, extension: pd.DataFrame,
                   p_split: float, p_max: float = np.inf) -> None:
    """Compare the two datasets where their p grids overlap."""
    keys = ["check_num", "p"]
    cols = ["lrb_error_rate", "lrb_n_points", "rb_error_rate"]
    both = production[keys + cols].merge(
        extension[keys + cols], on=keys, suffixes=("_prod", "_ext"))
    both = both[from_extension(both["p"], p_split, p_max)].sort_values(keys)
    print("  overlap (production vs extension):")
    for _, row in both.iterrows():
        print(
            f"    k={int(row.check_num)} p={row.p:.4f}  "
            f"LRB {row.lrb_error_rate_prod:.4f} (n={row.lrb_n_points_prod:.0f})"
            f" vs {row.lrb_error_rate_ext:.4f} (n={row.lrb_n_points_ext:.0f})"
            f"  RB {row.rb_error_rate_prod:.4f} vs {row.rb_error_rate_ext:.4f}")


def run_complete(run: str) -> bool:
    progress = RUNS_ROOT / run / "progress"
    markers = list(progress.glob("*/done.txt"))
    return bool(markers) and all(
        m.read_text().strip() == "1" for m in markers)


def merge_campaign(pairs: dict[str, tuple[str, str]], out_root: Path,
                   p_split: float, p_max: float = np.inf,
                   threshold_config: LRBThresholdConfig = THRESHOLD_CONFIG,
                   ) -> None:
    for label, (production_run, extension_run) in pairs.items():
        if not run_complete(extension_run):
            print(f"=== {label}: extension {extension_run} incomplete; skipped")
            continue
        print(f"=== {label}")
        out_dir = out_root / label
        production = pd.read_csv(
            RUNS_ROOT / production_run / "results" / "plots"
            / "unif_lrb_vs_rb_table_all_mixed_fits.csv")
        production = production[
            production["check_num"].between(CHECK_MIN, CHECK_MAX)]
        extension = build_table(
            plotter_for(extension_run, EvenDepthsForSparseChecks),
            out_dir / "extension-table")
        overlap_report(production, extension, p_split, p_max)

        merged = pd.concat(
            [production[~from_extension(production["p"], p_split, p_max)],
             extension[from_extension(extension["p"], p_split, p_max)]],
            ignore_index=True,
        ).sort_values(["check_num", "p"]).reset_index(drop=True)
        merged["source"] = np.where(
            from_extension(merged["p"], p_split, p_max),
            extension_run, production_run)
        merged_csv = out_dir / "unif_lrb_vs_rb_table_merged.csv"
        merged.to_csv(merged_csv, index=False)

        # The production plotter supplies code titles and the p axis; its
        # outputs are redirected so production plots stay untouched.
        plotter = plotter_for(production_run)
        plotter.out_dir = str(out_dir)
        summary_csv = plotter.plot_all_unif_threshold_graphs(
            threshold_config=threshold_config,
            table_csv_path=str(merged_csv),
            show=False,
            check_min=CHECK_MIN,
            check_max=CHECK_MAX,
        )
        summary = pd.read_csv(summary_csv)
        for _, row in summary.iterrows():
            print(
                f"  k={int(row.check_num)}: p*={row.threshold_p:.4f} "
                f"[{row.threshold_p_ci_low:.4f}, {row.threshold_p_ci_high:.4f}]"
                f" {row.method}")
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
        print(f"  outputs: {out_dir}")


SUMMARY_NAME = "unif_thresholds_summary_monotone_trim_zoom_pwindow.csv"
SEQUENCE_BOOTSTRAP = COMPARISONS / "sequence-bootstrap"


def sequence_bootstrap(label: str, interval: int, p_star: float) -> dict:
    """Error of the mean from scripts/sequence_bootstrap_thresholds.py.

    Used only when it was computed for this exact p*, so a stale bootstrap
    never sits next to a refreshed threshold.
    """
    path = SEQUENCE_BOOTSTRAP / label / "sequence_bootstrap_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    match = frame[frame["interval"] == interval]
    if match.empty or abs(float(match["p_star"].iloc[0]) - p_star) > 1e-9:
        return {}
    row = match.iloc[0]
    return {
        "seq_boot_se": float(row["seq_boot_se"]),
        "seq_ci68_low": float(row["seq_boot_ci68_low"]),
        "seq_ci68_high": float(row["seq_boot_ci68_high"]),
        "seq_jackknife_se": float(row["seq_jackknife_se"]),
    }


def write_current_summary(family: str = "qutrit") -> None:
    """Collect every current pseudo-threshold into one hand-off file."""
    names, summary_path, qubit = FAMILIES[family]
    sd6, si1000 = (CAMPAIGNS[name] for name in names)
    rows = []
    for campaign in (CAMPAIGNS[name] for name in names):
        for label, (production_run, extension_run) in campaign["pairs"].items():
            merged = campaign["out"] / label / SUMMARY_NAME
            if run_complete(extension_run) and merged.exists():
                path, source = merged, "merged"
                status = "final"
            else:
                path = RUNS_ROOT / production_run / "results" / "plots" / SUMMARY_NAME
                source = "production"
                status = ("provisional: refinement still running" if qubit
                          else "provisional: high-p extension still running")
            frame = pd.read_csv(path)
            for _, row in frame[frame["check_num"].between(
                    CHECK_MIN, CHECK_MAX)].iterrows():
                p_star = float(row["threshold_p"])
                rows.append({
                    "configuration": label,
                    "interval": int(row["check_num"]),
                    "p_star": p_star,
                    "ci68_low": float(row["threshold_p_ci_low"]),
                    "ci68_high": float(row["threshold_p_ci_high"]),
                    "seq_boot_se": np.nan,
                    "seq_ci68_low": np.nan,
                    "seq_ci68_high": np.nan,
                    "seq_jackknife_se": np.nan,
                    **sequence_bootstrap(label, int(row["check_num"]), p_star),
                    "method": row["method"],
                    "source": source,
                    "status": status,
                    "summary_csv": str(path.relative_to(PROJECT_ROOT)),
                })
    table = pd.DataFrame(rows)
    table.to_csv(summary_path.with_suffix(".csv"), index=False)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    lines = [
        "# Current LRB-D pseudo-thresholds (sdim 1.4.0, SI1000/SD6"
        + (", qubit [[5,1,2]] folded code" if qubit else "")
        + (", two ancillas on the conventional sites)" if family.endswith("conv")
           else ")"),
        "",
        f"Generated {stamp} by `scripts/merge_highp_threshold_extension.py`;"
        " rerun it to refresh. Machine-readable copy: "
        f"`{summary_path.name}.csv` next to this file.",
        "",
        "p* is the two-qudit gate error p where the LRB error per Clifford"
        " equals physical RB at the same p, same noise model. Intervals are"
        " 68% bootstrap. "
        + ("`source=merged` means production data outside the refinement"
           " window plus the near-threshold refinement (production depths,"
           " 1M shots, 30 new Clifford sequences, fine p grid) inside it:"
           f" SD6 {sd6['p_split']} <= p <= {sd6['p_max']} in steps of"
           + (" 0.0005 below 0.006 and 0.001 above," if sd6["p_split"] < 0.006
              else " 0.001,")
           + f" SI1000 {si1000['p_split']} <= p <= {si1000['p_max']} in"
           " steps of 0.0005. p* is where the LRB error equals the RB error,"
           " linearly interpolated (tol 0, no error floor; the qutrit"
           " settings tol 5e-4 and floor 2e-3 are on the scale of the qubit"
           " errors themselves). On the qutrit tables this rule gives every"
           " published one-ancilla and conventional-site p* exactly (16 of"
           " 16); of the eight coherent-relay p* it gives six exactly and"
           " two within 0.0002 (SD6 k=4 +0.00001, SI1000 k=3 +0.00015)."
           if qubit else
           "`source=merged` means production data for"
           " p < p_split plus the high-p extension (depths 0,1,2,3,4,6, 20M"
           " shots, new Clifford sequences; for k >= 2 only its even depths"
           " are fitted, matching production) for p >= p_split: SD6 p_split ="
           f" {sd6['p_split']}, SI1000 p_split ="
           f" {si1000['p_split']}.")
        + " `method` ending in `no_interp`"
        " means the crossing sits on a grid point; `no_permanent_regime`"
        " means no real crossing (not quotable).",
        "",
        "Two uncertainty columns. `68% interval (per point)` is the"
        " notebook's interval: each depth and each p resampled"
        " independently. `error of the mean` is the sequence bootstrap"
        " (`scripts/sequence_bootstrap_thresholds.py`, 1000 replicates):"
        " whole Clifford sequences are resampled with one draw shared by"
        " every depth and every p, every curve is refit and the crossing"
        " recomputed, so the correlations from reusing the same 30 sequences"
        " are included. It is the standard error over the random-sequence"
        " ensemble, given as SE and as its 68% percentile interval. Quote"
        " the error of the mean.",
        "",
        "| configuration | interval | p* | error of the mean (SE) |"
        " error-of-mean 68% interval | 68% interval (per point) | method |"
        " source | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if np.isfinite(row["seq_boot_se"]):
            se = f"{row['seq_boot_se']:.4f}"
            seq = f"[{row['seq_ci68_low']:.4f}, {row['seq_ci68_high']:.4f}]"
        else:
            se = seq = "pending"
        lines.append(
            f"| {row['configuration']} | {row['interval']} |"
            f" {row['p_star']:.4f} | {se} | {seq} | [{row['ci68_low']:.4f},"
            f" {row['ci68_high']:.4f}] | {row['method']} | {row['source']}"
            f" | {row['status']} |")
    lines += ["", "Per-configuration summary CSVs:", ""]
    for path in sorted({row["summary_csv"] for row in rows}):
        lines.append(f"- `{path}`")
    summary_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {summary_path.with_suffix('.md')}")


def main(argv: list[str]) -> int:
    names = argv or list(QUTRIT_CAMPAIGNS)
    families = [family for family, (members, _, _) in FAMILIES.items()
                if all(name in members for name in names)]
    if not families:
        raise SystemExit(
            "Merge each family (qutrit, qubit, conv, qubit-conv) in its own call.")
    for name in names:
        campaign = CAMPAIGNS[name]
        merge_campaign(campaign["pairs"], campaign["out"], campaign["p_split"],
                       campaign.get("p_max", np.inf),
                       campaign.get("threshold_config", THRESHOLD_CONFIG))
    write_current_summary(families[0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
