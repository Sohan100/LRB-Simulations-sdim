"""Crossing estimates from a sampled folder set (cache/<tag>/q*-*-*/).

For every configuration: per-p LRB fits (pipeline statistic, offset 0, all
depths incl. m = 0; even depths only for k >= 2 when the grid is the dense
0..6 extension grid, as in the production merge) and physical-RB fits on the
exact per-sequence RB polarization of 30 independent RB sequences (same
depths), then p*(k) with the notebook crossing chain (scale-free rule).

    python 11_scan_thresholds.py --tag scan
Writes results/<tag>_rates.csv and results/<tag>_thresholds.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import qfit
import qsim

KS = (1, 2, 3, 4)


def load(folder: Path):
    meta = json.loads((folder / "words.json").read_text())
    files = sorted(folder.glob("p*.npz"), key=lambda f: float(np.load(f)["p"]))
    data = {float(np.load(f)["p"]): dict(np.load(f)) for f in files}
    return meta, data


def keep_mask(depths, k, dense):
    d = np.asarray(depths)
    if dense and k >= 2:
        return d % 2 == 0
    return np.ones(len(d), dtype=bool)


def analyse(folder: Path, rb_seed: int, out_rows, thr_rows):
    meta, data = load(folder)
    q, noise, mode, depths = meta["q"], meta["noise"], meta["compile"], meta["depths"]
    dense = depths == [0, 1, 2, 3, 4, 6]
    probs = sorted(data)
    plotter = qfit.fake_plotter(Path("cache/plotters") / folder.name, q, depths, probs,
                                meta["n_seq"], noise)
    rb_words = qsim.draw_sequences(q, mode, meta["n_seq"], max(depths), rb_seed)
    lrb = {k: [] for k in KS}
    rb = []
    for p in probs:
        d = data[p]
        rb_pol = qfit.rb_exact_polarization(q, noise, rb_words, depths, p, mode)
        _, r_rb = qfit.fit_rb(plotter, rb_pol)
        rb.append(r_rb)
        row = dict(q=q, noise=noise, compile=mode, p=p, r_rb=r_rb)
        for k in KS:
            pol = qsim.polarization(d["unif"][k - 1])
            fit, r = qfit.fit_lrb(plotter, pol, keep_mask(depths, k, dense))
            lrb[k].append(r)
            acc = d["unif"][k - 1].sum(axis=(1, 2)) / (meta["shots"] * meta["n_seq"])
            row[f"r_lrb_k{k}"] = r
            row[f"acc_k{k}"] = json.dumps([float(f"{x:.3g}") for x in acc])
        out_rows.append(row)
    for k in KS:
        thr, method = qfit.threshold_from_errors(plotter, qfit.SCALE_FREE, probs, lrb[k], rb)
        thr_rows.append(dict(q=q, noise=noise, compile=mode, k=k, p_star=thr, method=method))
        print(f"q={q} {noise:6s} {mode:10s} k={k}: p* = {thr:.4f} ({method})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="scan")
    a = ap.parse_args()
    root = Path("cache") / a.tag
    out_rows, thr_rows = [], []
    for folder in sorted(root.glob("q*-*-*")):
        if not (folder / "words.json").exists():
            continue
        q = int(folder.name.split("-")[0][1:])
        analyse(folder, 900 + q, out_rows, thr_rows)
    pd.DataFrame(out_rows).to_csv(f"results/{a.tag}_rates.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(f"results/{a.tag}_thresholds.csv", index=False)


if __name__ == "__main__":
    main()
