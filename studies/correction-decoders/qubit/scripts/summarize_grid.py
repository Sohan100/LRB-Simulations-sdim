"""Summarize the main qubit [[5,1,2]] grid: F(m), rejection and fitted decay.

Reads ``<grid>/p<idx>.npz`` and ``<grid>/rb<idx>.npz`` written by
``run_grid.py`` and writes ``<out>/fits.csv`` (one row per probability and
method) and ``<out>/curves.csv`` (F(m) and rejection per depth). Gap-threshold
methods ``mld5_gapT`` accept a shot when the exact-MLD ``|LLR| >= T``.

    python studies/correction-decoders/qubit/scripts/summarize_grid.py \
        --grid studies/correction-decoders/qubit/results/grid_main \
        --out studies/correction-decoders/qubit/results/summary
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import fits as Fz  # noqa: E402

METHODS = [
    "lrbd_unif_1", "lrbd_unif_2", "lrbd_unif_4", "lrbd_unif_11",
    "lrbd_unif_22", "lrbd_const_1", "lrbd_const_2", "lrbd_const_4",
    "lrbd_const_22", "lrbd_unif_1_final", "lrbd_const_1_final", "mld5",
    "mld2", "const0", "const0_mld",
]
GAPS = [2.0, 4.0, 6.0, 8.0]


def gap_counts(hist: np.ndarray, threshold: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Accepted and error counts for ``|LLR| >= threshold`` (bin edges)."""
    start = int(np.searchsorted(E.GAP_EDGES, threshold, side="left"))
    kept = hist[:, :, start:, :]
    accepted = kept.sum(axis=(2, 3))
    errors = kept[..., 1].sum(axis=2)
    return accepted, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-boot", type=int, default=200)
    args = parser.parse_args()
    grid = Path(args.grid)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fit_rows, curve_rows = [], []
    for path in sorted(grid.glob("p*.npz"),
                       key=lambda q: int(q.stem[1:])):
        p_index = int(path.stem[1:])
        res = E.PointResult.from_npz(str(path))
        depths = res.depths
        entries = {}
        for name in METHODS:
            if name in res.tables:
                t = res.tables[name]
                entries[name] = (t.shots, t.accepted, t.errors, "lrb")
        if "mld5" in res.gap_hist:
            shots = res.tables["mld5"].shots
            for g in GAPS:
                acc, err = gap_counts(res.gap_hist["mld5"], g)
                entries[f"mld5_gap{g:g}"] = (shots, acc, err, "lrb")
        rb_path = grid / f"rb{p_index}.npz"
        if rb_path.exists():
            rb = np.load(rb_path)
            entries["physical_rb_5p"] = (rb["shots"], rb["accepted"],
                                         rb["errors"], "rb")
            # The repo's free-offset RB fit is ill-conditioned when F(m) is
            # nearly flat; |<Z>| decays to 0, so also fit with offset 0.
            entries["physical_rb_5p_a0"] = (rb["shots"], rb["accepted"],
                                            rb["errors"], "lrb")
        for name, (shots, acc, err, kind) in entries.items():
            fid = Fz.per_sequence_fidelity(acc, err)
            with np.errstate(invalid="ignore"):
                f_mean = np.mean(fid, axis=0)
                f_std = np.std(fid, axis=0)
            rej = 1.0 - acc.sum(axis=0) / np.maximum(shots.sum(axis=0), 1)
            fit = Fz.bootstrap_fit(depths, acc, err, kind=kind,
                                   num_boot=args.num_boot)
            fit_rows.append({
                "p_index": p_index, "p": res.p, "method": name, **fit,
                "error_per_depth": Fz.error_per_clifford(fit["f"]),
                "rej_m0": rej[0], "rej_mmax": rej[-1],
                "F_m0": f_mean[0], "F_mmax": f_mean[-1]})
            for d_index, depth in enumerate(depths):
                curve_rows.append({
                    "p_index": p_index, "p": res.p, "method": name,
                    "depth": depth, "F_mean": f_mean[d_index],
                    "F_std": f_std[d_index], "rejection": rej[d_index],
                    "accepted": int(acc[:, d_index].sum()),
                    "shots": int(shots[:, d_index].sum())})

    for name, rows in (("fits.csv", fit_rows), ("curves.csv", curve_rows)):
        with open(out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    show = ["lrbd_unif_1", "lrbd_unif_1_final", "lrbd_const_1",
            "lrbd_unif_4", "mld5", "mld5_gap4", "mld5_gap8", "const0",
            "const0_mld", "physical_rb_5p", "physical_rb_5p_a0"]
    print(f"{'p':>8s} " + " ".join(f"{m[:14]:>15s}" for m in show))
    for p_index in sorted({r["p_index"] for r in fit_rows}):
        sel = {r["method"]: r for r in fit_rows if r["p_index"] == p_index}
        p = next(iter(sel.values()))["p"]
        cells = []
        for m in show:
            r = sel.get(m)
            cells.append("-" if r is None or not np.isfinite(r["f"])
                         else f"{r['f']:.4f}/{r['rej_mmax']:.2f}")
        print(f"{p:8.4f} " + " ".join(f"{c:>15s}" for c in cells))
    print("(cells: fitted f / rejection at the largest depth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
