"""Stationarity tests on exact (shot-noise-free) LRB-D curves.

For every policy in ``exact_lrbd_p<p>.npz`` this computes, with a 2000-sample
bootstrap over Clifford sequences:

* ``F(m)``, the mean over sequences of ``|1 - 2 P(flip | accept)|`` (the repo
  estimator), and the acceptance ``A(m)``;
* the curvature witness ``c2``: the quadratic coefficient of a least-squares
  fit ``log F(m) = c0 + c1 m + c2 m^2`` over ``m in [m_min, m_max]``. A single
  exponential has ``c2 = 0``. ``c2 < 0`` means the decay accelerates, so the
  per-step logical error grows with depth. The same witness is computed for
  ``log A(m)``;
* the decay rate ``f`` and the per-step acceptance from fits over an early
  window ``[2, 12]`` and a late window ``[20, 40]``.

    python studies/correction-decoders/qubit/scripts/exact_lrbd_analysis.py \
        --exact studies/correction-decoders/qubit/results/exact_lrbd
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def quad_coeff(m: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    return float(np.polyfit(m[ok], y[ok], 2)[0])


def window_rate(m: np.ndarray, y: np.ndarray, lo: int, hi: int) -> float:
    sel = (m >= lo) & (m <= hi) & np.isfinite(y)
    if sel.sum() < 2:
        return np.nan
    return float(np.exp(np.polyfit(m[sel], y[sel], 1)[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exact", required=True)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--m-min", type=int, default=2)
    parser.add_argument("--m-max", type=int, default=40)
    args = parser.parse_args()
    folder = Path(args.exact)
    rng = np.random.default_rng(1)
    rows = []
    for path in sorted(folder.glob("exact_lrbd_p*.npz")):
        data = np.load(path)
        p = float(path.stem.split("_p")[1])
        m = data["depths"].astype(float)
        keys = sorted(k[3:] for k in data.files if k.startswith("A__"))
        for key in keys:
            acc = data[f"A__{key}"]
            flip = data[f"E__{key}"]
            n_seq = acc.shape[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                fid = np.abs(1.0 - 2.0 * flip / acc)

            def stats(pick):
                F = np.nanmean(fid[pick], axis=0)
                A = np.mean(acc[pick], axis=0)
                sel = (m >= args.m_min) & (m <= args.m_max)
                logF = np.log(np.where(F > 0, F, np.nan))
                logA = np.log(np.where(A > 0, A, np.nan))
                return (quad_coeff(m[sel], logF[sel]),
                        quad_coeff(m[sel], logA[sel]),
                        window_rate(m, logF, 2, 12),
                        window_rate(m, logF, 20, 40),
                        window_rate(m, logA, 2, 12),
                        window_rate(m, logA, 20, 40))

            point = stats(np.arange(n_seq))
            boots = np.array([stats(rng.integers(0, n_seq, n_seq))
                              for _ in range(args.boot)])
            lo = np.nanpercentile(boots, 2.5, axis=0)
            hi = np.nanpercentile(boots, 97.5, axis=0)
            names = ["c2_logF", "c2_logA", "f_early", "f_late", "a_early",
                     "a_late"]
            row = {"p": p, "policy": key, "n_seq": n_seq}
            for i, name in enumerate(names):
                row[name] = point[i]
                row[name + "_lo"] = lo[i]
                row[name + "_hi"] = hi[i]
            # Difference late - early with its own bootstrap interval.
            diff = boots[:, 3] - boots[:, 2]
            row["f_late_minus_early"] = point[3] - point[2]
            row["f_diff_lo"], row["f_diff_hi"] = np.nanpercentile(diff,
                                                                  [2.5, 97.5])
            rows.append(row)
    out = folder / "stationarity.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'p':>7s} {'policy':>14s} {'c2(logF) [95% CI]':>34s} "
          f"{'f early':>9s} {'f late':>9s} {'late-early [95% CI]':>30s} "
          f"{'a early':>8s} {'a late':>8s}")
    for r in rows:
        print(f"{r['p']:7g} {r['policy']:>14s} "
              f"{r['c2_logF']:+.2e} [{r['c2_logF_lo']:+.1e},{r['c2_logF_hi']:+.1e}] "
              f"{r['f_early']:9.5f} {r['f_late']:9.5f} "
              f"{r['f_late_minus_early']:+.2e} [{r['f_diff_lo']:+.1e},{r['f_diff_hi']:+.1e}] "
              f"{r['a_early']:8.4f} {r['a_late']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
