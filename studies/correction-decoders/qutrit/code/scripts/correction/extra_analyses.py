"""Pareto frontiers and quasi-static-noise mixture curves from the raw study outputs.

Writes two small CSVs next to the other committed results:

* ``pareto_frontier.csv``: for selected (p, depth), the acceptance and the
  accepted-shot infidelity of the maximum-likelihood posterior-threshold rule
  at every confidence-histogram bin edge (sequence-averaged), i.e. the full
  acceptance/fidelity frontier rather than a few fixed thresholds;
* ``mixture_curves.csv``: sequence-averaged signed fidelity and acceptance vs
  depth when every shot independently runs at p_low or p_high (50/50), for
  LRB-D (all rounds), blip-forgiving detection and full MLD correction, next
  to the two pure components.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrb.correction import aggregate, metrics  # noqa: E402
from lrb.correction.study import CONF_EDGES  # noqa: E402

DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()})


def pareto_rows(data: dict, probs, depth_indices) -> list[dict]:
    rows = []
    upper = np.concatenate([CONF_EDGES, [np.inf]])
    for p in probs:
        keys = sorted(k for k in data if np.isclose(k[2], p) and k[0] == "")
        if not keys:
            continue
        for di in depth_indices:
            hists = []
            for key in keys:
                entry = data[key]
                shots = float(entry[f"d{di}__shots"])
                hists.append(entry[f"d{di}__soft_hist_rb"] / shots)
            hists = np.array(hists)  # (n_seq, bins, 3) per-shot expected residual mass
            # Accept every bin whose upper edge is <= log10(1 - threshold).
            for b in range(len(upper)):
                accepted = hists[:, : b + 1, :].sum(axis=1)  # (n_seq, 3)
                acc = accepted.sum(axis=1)
                ok = acc > 0
                if not np.any(ok):
                    continue
                survival = np.where(ok, accepted[:, 0] / np.where(ok, acc, 1), np.nan)
                rows.append({"p": p, "depth": DEPTHS[di], "log10_one_minus_threshold": float(upper[b]),
                             "acc_mean": float(acc.mean()), "infid_mean": float(np.nanmean(1 - survival)),
                             "n_seq_accepting": int(ok.sum())})
    return rows


def mixture_curve_rows(data: dict, pairs, strategies) -> list[dict]:
    rows = []
    cliffs = sorted({c for (v, c, _p) in data if v == ""})
    for p_low, p_high in pairs:
        for strategy in strategies:
            for label, weights in (("low", (1.0, 0.0)), ("high", (0.0, 1.0)), ("mix", (0.5, 0.5))):
                for di, depth in enumerate(DEPTHS):
                    signed, accs = [], []
                    for cliff in cliffs:
                        entries = [data.get(("", cliff, p)) for p in (p_low, p_high)]
                        points = [aggregate.strategy_points(e, di).get(strategy) if e is not None else None
                                  for e in entries]
                        if any(pt is None for pt in points):
                            continue
                        joint = sum(w * pt[0] * pt[1] / max(pt[1].sum(), 1e-300) for w, pt in zip(weights, points))
                        accs.append(sum(w * pt[0] for w, pt in zip(weights, points)))
                        signed.append(metrics.fidelity_metrics(joint)["signed"])
                    if signed:
                        rows.append({"p_low": p_low, "p_high": p_high, "strategy": strategy, "component": label,
                                     "depth": depth, "signed_mean": float(np.nanmean(signed)),
                                     "signed_sem": float(np.nanstd(signed) / np.sqrt(len(signed))),
                                     "acc_mean": float(np.mean(accs))})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", default=str(REPO_ROOT / ".cache_correction" / "results_baseline"))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" / "correction-decoders" / "qutrit-512" / "results"))
    args = parser.parse_args()
    data = aggregate.load_npz_dir(args.baseline_results)
    out = Path(args.out)
    write_csv(out / "pareto_frontier.csv", pareto_rows(data, (5e-4, 1e-3, 2e-3, 5e-3), (3, 8)))
    write_csv(out / "mixture_curves.csv", mixture_curve_rows(
        data, ((5e-4, 2e-3), (1e-3, 5e-3)), ("lrbd:unif1", "bfd:all", "mld:all")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
