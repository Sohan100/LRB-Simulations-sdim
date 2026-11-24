"""Who carries the signal at the model crossing for large q?

At p = p*_model(q) (results/model_pstar.csv, production compile), for 96
random depth-1 sequences: the share of sequences whose depth-1 accepted shots
are still mostly clean (c_s > 1/2), their token counts, and for physical RB the
share of E_C lambda^(2n) contributed by words with at most the median token
count of those clean sequences.  Writes results/short_word_tail.csv.
"""
import importlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import qslib

m = importlib.import_module("23_model_pstar")
rows = []
mod = pd.read_csv("results/model_pstar.csv")
for q in (5, 7, 13, 31):
    for noise in ("si1000", "sd6"):
        ps = float(mod[(mod.q == q) & (mod.noise == noise) & (mod["compile"] == "production")].pstar_model.iloc[0])
        i = int(np.argmin(np.abs(m.P_GRID - ps)))
        jobs = [(q, noise, "production", 1000 * q + s) for s in range(96)]
        with ProcessPoolExecutor(max_workers=10) as ex:
            C1 = np.array(list(ex.map(m.clean_probs, jobs)))[:, i]
        words = [qslib.random_words(q, "production", 1, 1000 * q + s)[0] for s in range(96)]
        n = np.array([len(w) for w in words])
        c = C1 / (C1 + float(q) ** -8)
        clean = c > 0.5
        # RB: share of E lambda^(2n) from words with n <= max token count of clean LRB sequences
        s = 0.1 if noise == "si1000" else 1.0
        lam = 1 - s * ps * q * q / (q * q - 1)
        sym = qslib.symplectic_words(q, "HS")
        sl = np.array([0 if w == "I" else len(w) for w in sym])
        pa = np.arange(q)
        pl = (pa[:, None] + pa[None, :]).reshape(-1)
        tot = (pl[:, None] + sl[None, :]).reshape(-1)
        wts = lam ** (2 * tot)
        nmax = int(n[clean].max()) if clean.any() else -1
        rows.append(dict(q=q, noise=noise, p_model=ps, frac_clean_seq=float(clean.mean()),
                         mean_c=float(c.mean()), median_tokens_all=float(np.median(n)),
                         median_tokens_clean=float(np.median(n[clean])) if clean.any() else np.nan,
                         rb_share_from_short=float(wts[tot <= nmax].sum() / wts.sum()) if nmax >= 0 else np.nan,
                         catalogue_frac_short=float(np.mean(tot <= nmax)) if nmax >= 0 else np.nan))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/short_word_tail.csv", index=False)
