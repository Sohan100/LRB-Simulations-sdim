"""p*(k) with sequence-bootstrap standard errors for one q = 5 or 7 configuration.

Mirrors the production chain (merge + sequence_bootstrap_thresholds.py):
* rows with p < p_split come from the production-grid set (depths
  0,2,...,22, 1M shots), rows with p >= p_split from the extension set
  (depths 0,1,2,3,4,6, many shots, other Clifford sequences; even depths only
  for k >= 2);
* LRB rate: pipeline statistic and fit (offset 0, m = 0 included);
* RB rate: pipeline RB fit (offset 0) on DEM-sampled RB, 10,000 shots per
  circuit, own sequences per set;
* crossing: the notebook chain with the scale-free rule (tol 0, no floor),
  as for the qubit campaigns;
* error of the mean: 1000 replicates resampling whole sequences, one draw per
  sequence set (LRB prod, LRB ext, RB prod, RB ext) shared by every depth and p.

    python 13_final_thresholds.py --q 5 --noise si1000 --p-split 0.03
Writes results/final_q<q>_<noise>_<compile>_{rates,thresholds}.csv.
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


class Set:
    def __init__(self, tag, q, noise, mode, label, suffix=""):
        self.folder = Path("cache") / tag / f"q{q}-{noise}-{mode}{suffix}"
        meta = json.loads((self.folder / "words.json").read_text())
        self.depths = meta["depths"]
        self.n_seq = meta["n_seq"]
        self.lrb = {}
        self.rb = {}
        self.acc = {}
        for f in self.folder.glob("p*.npz"):
            z = np.load(f)
            p = float(z["p"])
            self.lrb[p] = {k: qsim.polarization(z["unif"][k - 1]) for k in KS}
            sd = z["shots_per_depth"] if "shots_per_depth" in z.files else np.full(len(self.depths), z["shots"])
            self.acc[p] = z["unif"][0].sum(axis=(1, 2)) / (sd * self.n_seq)
        for f in self.folder.glob("rb_p*.npz"):
            z = np.load(f)
            self.rb[float(z["p"])] = z["pol"]
        self.dense = self.depths == [0, 1, 2, 3, 4, 6]
        self.plotter = qfit.fake_plotter(Path("cache/plotters") / f"{tag}-{label}-q{q}-{noise}-{mode}{suffix}",
                                         q, self.depths, sorted(self.lrb), self.n_seq, noise)

    def keep(self, k):
        d = np.asarray(self.depths)
        return (d % 2 == 0) if (self.dense and k >= 2) else np.ones(len(d), bool)


def evaluate(rows, draws):
    rb, lrb = [], {k: [] for k in KS}
    for src, name, p in rows:
        s_rb = draws[f"rb_{name}"]
        s_lrb = draws[f"lrb_{name}"]
        pol = src.rb[p] if s_rb is None else src.rb[p][:, s_rb]
        rb.append(qfit.fit_rb(src.plotter, pol)[1])
        for k in KS:
            pk = src.lrb[p][k] if s_lrb is None else src.lrb[p][k][:, s_lrb]
            lrb[k].append(qfit.fit_lrb(src.plotter, pk, src.keep(k))[1])
    return rb, lrb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--compile", default="production")
    ap.add_argument("--prod-tag", default="final-prod")
    ap.add_argument("--ext-tag", default="final-ext")
    ap.add_argument("--p-split", type=float, required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--round", default="seven", choices=("seven", "five"))
    a = ap.parse_args()
    suffix = "" if a.round == "seven" else "-five"
    prod = Set(a.prod_tag, a.q, a.noise, a.compile, "prod", suffix)
    ext = Set(a.ext_tag, a.q, a.noise, a.compile, "ext", suffix)
    rows = [(prod, "prod", p) for p in sorted(prod.lrb) if p < a.p_split and p in prod.rb]
    rows += [(ext, "ext", p) for p in sorted(ext.lrb) if p >= a.p_split and p in ext.rb]
    ps = [r[2] for r in rows]
    full = {k: None for k in ("lrb_prod", "lrb_ext", "rb_prod", "rb_ext")}
    rb, lrb = evaluate(rows, full)
    pstar = {}
    for k in KS:
        pstar[k] = qfit.threshold_from_errors(prod.plotter, qfit.SCALE_FREE, ps, lrb[k], rb)
    rate_rows = []
    for i, (src, name, p) in enumerate(rows):
        rate_rows.append(dict(q=a.q, noise=a.noise, compile=a.compile, p=p, source=name, r_rb=rb[i],
                              **{f"r_lrb_k{k}": lrb[k][i] for k in KS},
                              acc_k1=";".join(f"{x:.3g}" for x in src.acc[p])))
    stem = f"results/final_q{a.q}_{a.noise}_{a.compile}" + ("" if a.round == "seven" else "_five")
    pd.DataFrame(rate_rows).to_csv(stem + "_rates.csv", index=False)
    rng = np.random.default_rng(20261120)
    reps = {k: [] for k in KS}
    for _ in range(a.reps):
        draws = {"lrb_prod": rng.integers(0, prod.n_seq, prod.n_seq),
                 "lrb_ext": rng.integers(0, ext.n_seq, ext.n_seq),
                 "rb_prod": rng.integers(0, prod.n_seq, prod.n_seq),
                 "rb_ext": rng.integers(0, ext.n_seq, ext.n_seq)}
        rbb, lrbb = evaluate(rows, draws)
        for k in KS:
            reps[k].append(qfit.threshold_from_errors(prod.plotter, qfit.SCALE_FREE, ps, lrbb[k], rbb)[0])
    out = []
    for k in KS:
        v = np.asarray(reps[k], float)
        ok = v[np.isfinite(v)]
        lo, hi = np.percentile(ok, [16, 84]) if len(ok) else (np.nan, np.nan)
        out.append(dict(q=a.q, noise=a.noise, compile=a.compile, round=a.round, k=k, p_star=pstar[k][0],
                        method=pstar[k][1], seq_boot_se=float(np.std(ok, ddof=1)) if len(ok) > 1 else np.nan,
                        ci68_low=lo, ci68_high=hi, valid_fraction=len(ok) / len(v),
                        p_split=a.p_split))
        print(f"q={a.q} {a.noise} {a.compile} k={k}: p* = {pstar[k][0]:.4f} +- "
              f"{out[-1]['seq_boot_se']:.4f} ({pstar[k][1]}; valid {len(ok)}/{len(v)})", flush=True)
    pd.DataFrame(out).to_csv(stem + "_thresholds.csv", index=False)


if __name__ == "__main__":
    main()
