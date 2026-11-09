"""Summarise the login-node protocol-variant runs (results/variant_sim_*.npz).

Per variant: fitted error per step on all usable depths (depth 0 always included),
log-quadratic curvature c2 with a 95% sequence-bootstrap interval, chi2/dof
of a single exponential, the late/early window ratio (fit on the deeper half
of the depths over fit on the shallower half), acceptance per step and at the
deepest depth. ``per_clifford`` divides the mirror variants' per-pair rate by
2 so all three share one unit: one logical Clifford plus one check round.
Writes results/variant_summary.csv.
"""
from __future__ import annotations

import csv
import glob
import os
import warnings

import numpy as np

import nmlib as L

warnings.filterwarnings("ignore", category=RuntimeWarning)


def summarise(t, rj, d, lo):
    c = L.curves(t, rj)
    b = L.bootstrap_curves(t, rj, n_boot=200, fields=("pooled", "acc"))
    se = np.nanstd(b["pooled"], 0)
    ok = (c["n_acc_tot"] >= 1000) & (d >= lo) & (c["pooled"] > 0) & (se > 0)
    dd, y, s = d[ok], c["pooled"][ok], se[ok]
    f = L.fit_fixed_a(dd, y, s, y_ceiling=None)
    rb = [L.fit_fixed_a(dd, bp[ok], s, y_ceiling=None)["r"] for bp in b["pooled"] if np.all(bp[ok] > 0)]
    out = {"depths": " ".join(map(str, dd)), "r": f["r"], "r_se": float(np.std(rb))}
    lin = L.fit_log_poly(dd, y, s, 1)
    out["chi2_dof_exp"] = lin["chi2"] / lin["dof"] if lin and lin["dof"] > 0 else np.nan
    if len(dd) >= 4:
        q = L.fit_log_poly(dd, y, s, 2)
        c2b = [L.fit_log_poly(dd, bp[ok], s, 2)["c"][2] for bp in b["pooled"] if np.all(bp[ok] > 0)]
        out.update(c2=q["c"][2], c2_lo=np.percentile(c2b, 2.5), c2_hi=np.percentile(c2b, 97.5))
        h = len(dd) // 2
        f1 = L.fit_fixed_a(dd[:h + 1], y[:h + 1], s[:h + 1], y_ceiling=None)
        f2 = L.fit_fixed_a(dd[h:], y[h:], s[h:], y_ceiling=None)
        out["late_over_early"] = f2["r"] / f1["r"] if f1["r"] > 0 else np.nan
    okA = c["n_acc_tot"] >= 1000
    out["acc_per_step"] = float(np.exp(np.polyfit(d[okA], np.log(c["acc"][okA]), 1)[0]))
    out["acc_at_deepest"] = float(c["acc"][-1])
    return out


def main():
    rows = []
    for fn in sorted(glob.glob(os.path.join(L.OUT, "results", "variant_sim_*.npz"))):
        z = np.load(fn)
        d = z["depths"]
        variants = [k[: -len("_tallies")] for k in z.files if k.endswith("_tallies")]
        for v in variants:
            t, rj = z[f"{v}_tallies"].astype(float), z[f"{v}_rejected"].astype(float)
            for lo in (0,):
                r = summarise(t, rj, d, lo)
                r["per_clifford"] = r["r"] / (1 if v.startswith("compiled") else 2)
                rows.append({"file": os.path.basename(fn), "noise": str(z["noise"]), "p": float(z["p"]),
                             "shots": int(z["shots"]),
                             "variant": v, "m_min": lo, **r})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = os.path.join(L.OUT, "results", "variant_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
