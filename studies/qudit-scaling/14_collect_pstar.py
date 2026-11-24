"""Assemble p*(q) for the one-ancilla layout, production compile.

q = 2, 3: the published values and sequence-bootstrap SEs from the canonical
hand-off files (pseudo-thresholds-qubit-current.csv, pseudo-thresholds-current.csv).
q = 5, 7: this study (results/final_q*_thresholds.csv, 13_final_thresholds.py).
Adds the model envelope (results/model_pstar.csv), the per-step acceptance at
p* measured in the simulations, physical RB at p*, and local growth exponents
d ln p* / d ln q with propagated SEs.

Writes results/pstar_vs_q.csv and prints a table.
"""
import numpy as np
import pandas as pd

import qfit  # noqa: F401  (puts the repository's scripts/ on sys.path)
import merge_highp_threshold_extension as merge

rows = []
canon = {("seven", 2): ("pseudo-thresholds-qubit-current.csv", "qubit-one-ancilla-{n}"),
         ("seven", 3): ("pseudo-thresholds-current.csv", "one-ancilla-{n}"),
         ("five", 2): ("pseudo-thresholds-qubit-prod5b-current.csv", "qubit-prod5b-one-ancilla-{n}"),
         ("five", 3): ("pseudo-thresholds-prod5b-current.csv", "prod5b-one-ancilla-{n}")}
for (rnd, q), (fname, pat) in canon.items():
    t = pd.read_csv(merge.COMPARISONS / fname)
    for noise in ("si1000", "sd6"):
        for r in t[t.configuration == pat.format(n=noise)].itertuples():
            rows.append(dict(round=rnd, q=q, noise=noise, k=int(r.interval), p_star=float(r.p_star),
                             se=float(r.seq_boot_se), source=f"published ({fname})",
                             method=r.method))
for rnd, suf in (("seven", ""), ("five", "_five")):
    for q in (5, 7):
        for noise in ("si1000", "sd6"):
            try:
                t = pd.read_csv(f"results/final_q{q}_{noise}_production{suf}_thresholds.csv")
            except FileNotFoundError:
                continue
            for r in t.itertuples():
                rows.append(dict(round=rnd, q=q, noise=noise, k=int(r.k), p_star=float(r.p_star),
                                 se=float(r.seq_boot_se), source="this study (13_final_thresholds.py)",
                                 method=r.method))
d = pd.DataFrame(rows)
models = []
for rnd, fname in (("seven", "results/model_pstar.csv"), ("five", "results/model_pstar_five.csv")):
    m = pd.read_csv(fname)
    m = m[m["compile"] == "production"][["q", "noise", "pstar_model", "pstar_rand"]].assign(round=rnd)
    models.append(m)
d = d.merge(pd.concat(models), on=["round", "q", "noise"], how="left")
# crossings without a permanent regime are not quotable
d.loc[d.method.str.startswith("no_permanent"), "p_star_quotable"] = False
d["p_star_quotable"] = d.get("p_star_quotable", True)
d["p_star_quotable"] = d["p_star_quotable"].fillna(True)
d["ratio_to_model"] = d.p_star / d.pstar_model
# local growth exponent between successive q (k = 1, 2)
out = []
for (rnd, noise, k), g in d[d.p_star_quotable.astype(bool)].groupby(["round", "noise", "k"]):
    g = g.sort_values("q")
    for a, b in zip(g.itertuples(), list(g.itertuples())[1:]):
        if np.isfinite(a.p_star) and np.isfinite(b.p_star):
            alpha = np.log(b.p_star / a.p_star) / np.log(b.q / a.q)
            se = np.sqrt((a.se / a.p_star) ** 2 + (b.se / b.p_star) ** 2) / np.log(b.q / a.q)
            out.append(dict(round=rnd, noise=noise, k=k, q_from=a.q, q_to=b.q, exponent=alpha, exponent_se=se))
pd.DataFrame(out).to_csv("results/pstar_growth_exponents.csv", index=False)
d.to_csv("results/pstar_vs_q.csv", index=False)
pd.set_option("display.width", 220)
print(d[d.k <= 2].sort_values(["round", "noise", "k", "q"]).round(4).to_string(index=False))
print(pd.DataFrame(out).round(3).to_string(index=False))
