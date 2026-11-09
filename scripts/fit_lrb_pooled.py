"""Write pooled signed-polarization fits (r_L, d_L, r_uncond) for one run.

An alternative to the notebook's mixed-fit tables, which it does not touch.
Output goes to --out-dir only:

    pooled_lrb_table.csv           one row per (interval k, p)
    pooled_pseudo_thresholds.csv   one row per k: p where r_L = r_RB

Usage:
    python scripts/fit_lrb_pooled.py RUN_FOLDER --out-dir DIR [--checks 1,2,3,4]
        [--min-accepted 1000] [--no-aligned] [--n-boot 200]
        [--highp-run HIGHP_RUN_FOLDER --split P]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lrb.pooled_lrb_fits import build_pooled_tables  # noqa: E402


def _write(path, rows):
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checks", default="1,2,3,4")
    ap.add_argument("--min-accepted", type=int, default=1000)
    ap.add_argument("--no-aligned", action="store_true", help="also fit depths that are not multiples of k")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--highp-run", help="take rows with p >= --split from this run instead")
    ap.add_argument("--split", type=float)
    a = ap.parse_args()
    rows, thr = build_pooled_tables(
        os.path.abspath(a.run), checks=[int(x) for x in a.checks.split(",")],
        min_accepted=a.min_accepted, aligned=not a.no_aligned, n_boot=a.n_boot,
        highp_run=os.path.abspath(a.highp_run) if a.highp_run else None, split=a.split)
    os.makedirs(a.out_dir, exist_ok=True)
    _write(os.path.join(a.out_dir, "pooled_lrb_table.csv"), rows)
    _write(os.path.join(a.out_dir, "pooled_pseudo_thresholds.csv"), thr)
    for t in thr:
        lo, hi = t["ci68_low"], t["ci68_high"]
        print(f"k={t['k']}: p*(r_L) = {t['p_star_r_L']:.4g} [{lo:.4g}, {hi:.4g}] ({t['note']}); "
              f"r_uncond: {t['note_uncond']}")


if __name__ == "__main__":
    main()
