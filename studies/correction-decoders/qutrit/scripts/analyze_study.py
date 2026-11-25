"""Aggregate decoder-study outputs into the CSV tables and figures of the report.

Reads the raw per-circuit ``.npz`` outputs of ``run_decoder_study.py`` and
``run_variant_study.py`` (kept outside git under ``.cache_correction``) and
writes small, committed result files:

* ``baseline_reproduction.csv``: exact LRB-D of this study vs the 1M-shot
  production CSVs, for every p, policy and depth;
* ``curves_<study>.csv``: Clifford-averaged acceptance and fidelity per
  strategy, p and depth;
* ``fits_<study>.csv``: decay fits (signed metric, offset 0, bootstrapped
  over sequences), per-step acceptance, curvature of log-fidelity, and the
  first/last depth-windowed decay factors;
* ``rb_reference.csv``: physical RB at p_RB = 5p from the production run;
* figures under ``figures/``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrb.correction import aggregate, metrics  # noqa: E402

BASELINE_DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
PRODUCTION_SHOTS = 1_000_000


def read_stats_csv(path: Path) -> dict:
    rows = list(csv.reader(path.read_text().strip().splitlines()))
    return {
        "p": float(rows[0][1]),
        "fid": np.array([float(v) for v in rows[1][1:]]),
        "fid_std": np.array([float(v) for v in rows[2][1:]]),
        "rej": np.array([float(v) for v in rows[3][1:]]),
        "rej_std": np.array([float(v) for v in rows[4][1:]]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as CSV; a ``.gz`` suffix writes gzip-compressed text."""
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()})


def curve_rows(curves, study: str, variant: str = "") -> list[dict]:
    rows = []
    for curve in curves:
        for di, depth in enumerate(curve.depths):
            row = {"study": study, "variant": variant, "strategy": curve.strategy, "p": curve.p,
                   "depth": int(depth), "n_seq": int(np.sum(np.isfinite(curve.acceptance[:, di]))),
                   "acc_mean": float(np.nanmean(curve.acceptance[:, di])),
                   "acc_std": float(np.nanstd(curve.acceptance[:, di]))}
            for m in aggregate.METRICS:
                row[f"{m}_mean"] = float(np.nanmean(curve.fidelity[m][:, di]))
                row[f"{m}_std"] = float(np.nanstd(curve.fidelity[m][:, di]))
            rows.append(row)
    return rows


def _fit_one(args):
    curve, study, variant, boot = args
    return _fit_curve_row(curve, study, variant, boot)


def fit_rows(curves, study: str, variant: str = "", boot: int = 200, workers: int = 4) -> list[dict]:
    from multiprocessing import Pool

    with Pool(workers) as pool:
        return pool.map(_fit_one, [(c, study, variant, boot) for c in curves], chunksize=8)


def _fit_curve_row(curve, study: str, variant: str, boot: int) -> dict:
    if True:
        depths = curve.depths
        row = {"study": study, "variant": variant, "strategy": curve.strategy, "p": curve.p}
        for m in ("signed", "legacy"):
            fit = aggregate.fit_curve(depths, curve.fidelity[m], metric=m, boot=boot)
            for key, value in fit.items():
                row[f"{m}_{key}"] = value
        acc = aggregate.per_step_acceptance(depths, curve.acceptance)
        row["acc_per_step"] = acc["a"]
        row["acc_c"] = acc["c"]
        signed = curve.fidelity["signed"]
        if np.all(np.isfinite(signed)) and np.all(np.nanmean(signed, axis=0) > 0):
            curv = aggregate.curvature(depths, signed, min_depth=2, boot=boot)
            row.update({f"signed_{k}": v for k, v in curv.items()})
            win = aggregate.windowed(depths, signed, boot=boot)
            row["win_first"] = win[1]["rate"] if len(win) > 1 else np.nan
            row["win_first_lo"] = win[1]["lo"] if len(win) > 1 else np.nan
            row["win_first_hi"] = win[1]["hi"] if len(win) > 1 else np.nan
            row["win_last"] = win[-1]["rate"]
            row["win_last_lo"] = win[-1]["lo"]
            row["win_last_hi"] = win[-1]["hi"]
            row["windows"] = json.dumps([[w["m0"], w["m1"], round(w["rate"], 8), round(w["lo"], 8), round(w["hi"], 8)] for w in win])
        return row


def baseline_reproduction(curves, run_folder: Path, probs: list[float]) -> list[dict]:
    rows = []
    lookup = {(c.strategy, round(c.p, 12)): c for c in curves}
    for p_index, p in enumerate(probs):
        if p == 0:
            continue
        for policy, folder in (("unif", "unif_check_data"), ("const", "const_check_data")):
            for k in range(0, 23):
                path = run_folder / "results" / "LRB" / str(p_index) / folder / f"{k}.csv"
                if not path.exists():
                    continue
                name = "lrbd:const0" if (policy == "const" and k == 0) else f"lrbd:{policy}{k}"
                curve = lookup.get((name, round(p, 12)))
                if curve is None:
                    continue
                stats = read_stats_csv(path)
                for di, depth in enumerate(curve.depths):
                    acc = curve.acceptance[:, di]
                    fid = curve.fidelity["legacy"][:, di]
                    n_seq = acc.size
                    rej_prod = float(stats["rej"][di])
                    fid_prod = float(stats["fid"][di])
                    # Production used 1e6 shots per circuit: add binomial shot noise to the
                    # across-sequence spread, and compare fidelities only where the
                    # production run accepted enough shots for its |E w^k| estimate to be
                    # unbiased (the legacy metric is biased upward by ~1/sqrt(N_acc)).
                    n_acc = (1.0 - rej_prod) * PRODUCTION_SHOTS
                    rej_se = np.sqrt((stats["rej_std"][di] ** 2 + rej_prod * (1 - rej_prod) / PRODUCTION_SHOTS) / n_seq)
                    fid_se = np.sqrt((stats["fid_std"][di] ** 2 + 1.5 * max(1 - fid_prod, 0.0) / max(n_acc, 1.0)) / n_seq)
                    comparable = n_acc >= 1000
                    rows.append({
                        "p": p, "policy": f"{policy}{k}", "depth": int(depth),
                        "rej_study": float(1 - acc.mean()), "rej_prod": rej_prod,
                        "rej_seq_std": float(stats["rej_std"][di]),
                        "fid_study": float(np.nanmean(fid)), "fid_prod": fid_prod,
                        "fid_seq_std": float(stats["fid_std"][di]),
                        "n_acc_prod_per_circuit": n_acc,
                        "rej_z": float((1 - acc.mean() - rej_prod) / max(rej_se, 1e-15)),
                        "fid_z": float((np.nanmean(fid) - fid_prod) / max(fid_se, 1e-15)) if comparable else float("nan"),
                        "fid_comparable": comparable,
                    })
    return rows


MIXTURE_PAIRS = ((5e-4, 2e-3), (1e-3, 5e-3), (2e-4, 1e-3))
MIXTURE_STRATEGIES = ("lrbd:unif1", "lrbd:const1", "bfd:all", "soft:0.99", "mld:all", "raw")


def mixture_rows(data: dict, depths: list[int], boot: int = 200) -> list[dict]:
    """
    Quasi-static noise fluctuations: each shot draws p_low or p_high (50/50).

    The joint (logical shift, accept) distribution is linear in the mixture
    weights, so the mixed curve of every strategy is exact given the per-p
    results: acceptance mixes linearly and the accepted residual distribution
    is the acceptance-weighted mixture. Postselection therefore reweights the
    mixture toward the quieter component as depth grows.
    """
    rows = []
    cliffs = sorted({c for (_v, c, _p) in data})
    for p_low, p_high in MIXTURE_PAIRS:
        key_low = [k for k in data if np.isclose(k[2], p_low)]
        key_high = [k for k in data if np.isclose(k[2], p_high)]
        if not key_low or not key_high:
            continue
        for strategy in MIXTURE_STRATEGIES:
            for label, weights in (("low", (1.0, 0.0)), ("high", (0.0, 1.0)), ("mix", (0.5, 0.5))):
                signed = np.full((len(cliffs), len(depths)), np.nan)
                acc = np.full((len(cliffs), len(depths)), np.nan)
                for ci, cliff in enumerate(cliffs):
                    entries = [data.get(("", cliff, p)) for p in (p_low, p_high)]
                    if any(e is None for e in entries):
                        continue
                    for di in range(len(depths)):
                        points = [aggregate.strategy_points(e, di).get(strategy) for e in entries]
                        if any(pt is None for pt in points):
                            continue
                        joint = sum(w * pt[0] * pt[1] / max(pt[1].sum(), 1e-300) for w, pt in zip(weights, points))
                        acc[ci, di] = sum(w * pt[0] for w, pt in zip(weights, points))
                        signed[ci, di] = metrics.fidelity_metrics(joint)["signed"]
                if not np.all(np.isfinite(signed)):
                    continue
                fit = aggregate.fit_curve(depths, signed, metric="signed", boot=boot)
                curv = aggregate.curvature(depths, signed, min_depth=2, boot=boot)
                win = aggregate.windowed(depths, signed, boot=boot)
                rows.append({"p_low": p_low, "p_high": p_high, "component": label, "strategy": strategy,
                             "f": fit.get("f"), "r": fit.get("r"), "c2": curv["c2"], "c2_lo": curv["c2_lo"],
                             "c2_hi": curv["c2_hi"], "win_first": win[1]["rate"], "win_last": win[-1]["rate"],
                             "acc_m_max": float(np.nanmean(acc[:, -1]))})
    return rows


def rb_reference(run_folder: Path, probs: list[float]) -> list[dict]:
    rows = []
    for p_index, p in enumerate(probs):
        path = run_folder / "results" / "RB_comparison" / f"{p_index}.csv"
        if not path.exists() or p == 0:
            continue
        stats = read_stats_csv(path)
        # The pipeline fits RB with a free offset; with 22 depths and a slow decay that
        # fit is ill-conditioned at small p (it can return an offset near 0.9). The
        # metric's true asymptote is 0, so the offset-0 fit (as for LRB) is reported
        # as the reference and the free-offset fit is kept for comparison.
        fixed = metrics.fit_decay(BASELINE_DEPTHS, stats["fid"], stats["fid_std"], free_offset=False)
        free = metrics.fit_decay(BASELINE_DEPTHS, stats["fid"], stats["fid_std"], free_offset=True)
        rows.append({"p_lrb": p, "p_rb": stats["p"], "rb_f": fixed["f"], "rb_r": fixed["r"],
                     "rb_b": fixed["b"], "rb_free_f": free["f"], "rb_free_r": free["r"],
                     "rb_free_a": free["a"], "rb_free_b": free["b"]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-results", default=str(REPO_ROOT / ".cache_correction" / "results_baseline"))
    parser.add_argument("--variant-results", default=str(REPO_ROOT / ".cache_correction" / "results_variants"))
    parser.add_argument("--dense-results", default=str(REPO_ROOT / ".cache_correction" / "results_dense"))
    parser.add_argument("--run-folder", default=os.environ.get("LRB_BASELINE_RUN", "/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000"))
    parser.add_argument("--out", default=str(STUDY / "results"))
    parser.add_argument("--boot", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_folder = Path(args.run_folder)
    probs = [float(v) for v in (run_folder / "probs.txt").read_text().strip().split(",")]

    write_csv(out / "rb_reference.csv", rb_reference(run_folder, probs))

    base_dir = Path(args.baseline_results)
    if base_dir.exists():
        data = aggregate.load_npz_dir(str(base_dir))
        curves = aggregate.build_curves(data, BASELINE_DEPTHS)
        write_csv(out / "baseline_reproduction.csv.gz", baseline_reproduction(curves, run_folder, probs))
        write_csv(out / "curves_baseline.csv.gz", curve_rows(curves, "baseline"))
        write_csv(out / "fits_baseline.csv.gz", fit_rows(curves, "baseline", boot=args.boot))
        write_csv(out / "mixture_nonmarkov.csv", mixture_rows(data, BASELINE_DEPTHS, boot=args.boot))
        print(f"baseline: {len(data)} files, {len(curves)} curves")

    dense_dir = Path(args.dense_results)
    if dense_dir.exists():
        data = aggregate.load_npz_dir(str(dense_dir))
        config = json.loads((dense_dir / "variant_config.json").read_text())
        depths = config["depths"]
        rows_c, rows_f = [], []
        for variant in sorted({key[0] for key in data}):
            curves = aggregate.build_curves(data, depths, variant)
            rows_c += curve_rows(curves, "dense", variant)
            rows_f += fit_rows(curves, "dense", variant, boot=args.boot)
        write_csv(out / "curves_dense.csv.gz", rows_c)
        write_csv(out / "fits_dense.csv.gz", rows_f)
        print(f"dense: {len(data)} files")

    var_dir = Path(args.variant_results)
    if var_dir.exists():
        data = aggregate.load_npz_dir(str(var_dir))
        config = json.loads((var_dir / "variant_config.json").read_text())
        depths = config["depths"]
        rows_c, rows_f = [], []
        for variant in sorted({key[0] for key in data}):
            curves = aggregate.build_curves(data, depths, variant)
            rows_c += curve_rows(curves, "variants", variant)
            rows_f += fit_rows(curves, "variants", variant, boot=args.boot)
        write_csv(out / "curves_variants.csv.gz", rows_c)
        write_csv(out / "fits_variants.csv.gz", rows_f)
        print(f"variants: {len(data)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
