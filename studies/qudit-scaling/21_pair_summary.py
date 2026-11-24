"""Summarise the exact two-fault expansion (results/pair_expansion_<tag>.csv.gz).

Per (q, noise, compile): the bulk two-fault coefficient S2(m), averaged over
the random sequences, is fitted as S2(m) = s0 + s1 m + s2 m^2 for m = 1..4.
At low p the accepted logical error is eps(m) = S1 p + S2(m) p^2, the logical
errors are uniform over the q-1 non-zero values (Clifford twirl), so the
pipeline rate per depth step is r_LRB = d eps/dm = (s1 + 2 s2 m) p^2.

Physical RB at low p (exact, every token gets DEP1 at s p):
    r_RB = (q-1)/q [1 - E_C lambda^(2 n(C))] ~ 2 nbar s p q/(q+1),
so the leading-order crossing is p*_LO = 2 nbar s q / ((q+1) s1_eff) with
s1_eff = s1 + 2 s2 m_eff; m_eff = 0 (forward slope only) and m_eff = 5.

Also splits s1 by the rank w of the pair (silent fraction (q-1) q^-w) and by
where the two faults sit (F forward Clifford, C forward check, I inverse
Clifford, E final check).

Writes results/pair_summary_<tag>.csv and results/pair_classes_<tag>.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import qslib


def slope_fit(m, y):
    A = np.vstack([np.ones_like(m), m, m * m]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    d = pd.read_csv(f"results/pair_expansion_{a.tag}.csv.gz")
    d = d[d.w >= 0]
    rows, cls = [], []
    for (q, noise, mode), g in d.groupby(["q", "noise", "compile"]):
        n_seq = g.seq.nunique()
        per = g.groupby(["depth", "seq"]).S2b.sum().unstack("seq").fillna(0.0)
        m = per.index.to_numpy(float)
        mean = per.mean(axis=1).to_numpy()
        coef = slope_fit(m, mean)
        # sequence bootstrap of s1, s2
        rng = np.random.default_rng(1)
        boots = []
        for _ in range(400):
            s = rng.integers(0, per.shape[1], per.shape[1])
            boots.append(slope_fit(m, per.iloc[:, s].mean(axis=1).to_numpy()))
        boots = np.array(boots)
        nbar = qslib.mean_tokens(q, mode)
        s = 0.1 if noise == "si1000" else 1.0
        rb_slope = 2 * nbar * s * q / (q + 1)
        row = dict(q=q, noise=noise, compile=mode, n_seq=n_seq, nbar=nbar,
                   s0=coef[0], s1=coef[1], s2=coef[2],
                   s1_se=boots[:, 1].std(ddof=1), s2_se=boots[:, 2].std(ddof=1),
                   rb_slope=rb_slope,
                   pstar_LO_m0=rb_slope / coef[1] if coef[1] > 0 else np.nan,
                   pstar_LO_m5=rb_slope / (coef[1] + 10 * coef[2]))
        rows.append(row)
        # class decomposition of the slope: per class, slope of mean S2b vs m
        for key, gg in g.groupby(["w", "seg_pair"]):
            pc = gg.groupby("depth").S2b.sum().reindex(per.index, fill_value=0.0) / n_seq
            c = slope_fit(m, pc.to_numpy())
            cls.append(dict(q=q, noise=noise, compile=mode, w=key[0], seg_pair=key[1],
                            c0=c[0], c1=c[1], c2=c[2], S2b_m4=float(pc.iloc[-1])))
    out = pd.DataFrame(rows)
    out.to_csv(f"results/pair_summary_{a.tag}.csv", index=False)
    pd.DataFrame(cls).to_csv(f"results/pair_classes_{a.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
