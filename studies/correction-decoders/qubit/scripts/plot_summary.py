"""At-a-glance comparison at p = 1e-3 under both noise models.

Left panel: error per depth step (1 - f)/2 divided by p (log scale); right
panel: acceptance at m = 22. Data: results/summary/fits.csv (repo SI1000,
main grid) and results/lit_grid/lit_fits.json (literature SI1000).

    python studies/correction-decoders/qubit/scripts/plot_summary.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
DOCS = ROOT / "studies/correction-decoders/qubit"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SLOT1, SLOT2 = "#2a78d6", "#eb6834"
P = 0.001

ROWS = [  # (label, main-grid method, literature method)
    ("LRB-D, every check + final data check", "lrbd_unif_1_final",
     "lrbd_unif_1_final"),
    ("LRB-D, every check (repo 2-qubit readout)", "lrbd_unif_1",
     "lrbd_unif_1"),
    ("LRB-D, check every 4 Cliffords", "lrbd_unif_4", "lrbd_unif_4"),
    ("LRB-D, terminal check only (const=1)", "lrbd_const_1", "lrbd_const_1"),
    ("MLD + accept |LLR| >= 4", "mld5_gap4", "mld5_gap4"),
    ("MLD + accept |LLR| >= 2", "mld5_gap2", "mld5_gap2"),
    ("Full correction (exact MLD)", "mld5", "mld5"),
    ("No checks, postselect final X checks", "const0", "const0"),
    ("No checks, decode final readout", "const0_mld", "const0_mld"),
    ("Physical RB at p_RB = p", None, "physical_rb_1p"),
    ("Physical RB at p_RB = 5p", "physical_rb_5p_a0", "physical_rb_5p"),
]


def main() -> int:
    main_fit, main_rej = {}, {}
    for row in csv.DictReader(open(DOCS / "results/summary/fits.csv")):
        if abs(float(row["p"]) - P) < 1e-12:
            main_fit[row["method"]] = (1 - float(row["f"])) / 2 / P
            main_rej[row["method"]] = float(row["rej_mmax"])
    lit_fit, lit_acc = {}, {}
    for row in json.load(open(DOCS / "results/lit_grid/lit_fits.json")):
        if abs(row["p"] - P) < 1e-12:
            lit_fit[row["method"]] = row["err_per_step_over_p"]
            lit_acc[row["method"]] = row["A22"]
    # Physical RB at p_RB = p under the repo model equals the literature one
    # (a single qubit is never idle), so reuse it for the main-grid column.
    main_fit.setdefault("physical_rb_1p", lit_fit.get("physical_rb_1p"))

    labels = [r[0] for r in ROWS]
    y = np.arange(len(ROWS))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    for offset, color, name, fits, accs in (
            (0.18, SLOT1, "repo SI1000 (Aug 2026 run)", main_fit,
             {k: 1 - v for k, v in main_rej.items()}),
            (-0.18, SLOT2, "literature SI1000 (Oct 2026 runs)", lit_fit,
             lit_acc)):
        e = [fits.get(r[1] if color == SLOT1 else r[2]) or
             (fits.get("physical_rb_1p") if r[0].endswith("= p") else None)
             for r in ROWS]
        a = [accs.get(r[1] if color == SLOT1 else r[2], 1.0)
             if r[1] or r[2] else 1.0 for r in ROWS]
        ev = np.array([np.nan if v is None else v for v in e], float)
        av = np.array([np.nan if v is None else v for v in a], float)
        axes[0].barh(y + offset, ev, height=0.34, color=color, label=name)
        axes[1].barh(y + offset, av, height=0.34, color=color, label=name)
    axes[0].set_xscale("log")
    axes[0].axvline(main_fit["physical_rb_1p"], color=MUTED, lw=1,
                    ls="--")
    axes[0].set_yticks(y, labels, fontsize=8.5, color=INK)
    axes[0].set_xlabel("Error per depth step / p, (1 - f)/2p (log)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Acceptance at depth m = 22 (log)")
    for ax in axes:
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=8.5, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Qubit [[5,1,2]] LRB at p = 1e-3: what each protocol costs "
                 "(dashed: bare qubit at the same p)", color=INK,
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = DOCS / "figures/main/fig0_summary_p1e-3.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
