#!/usr/bin/env python3
"""A4: attribute the second-order accepted infidelity to circuit regions.

For the cached DEMs of A4_exact_lrbd.py, remove the faults of one region of
the depth-m circuit and recompute the small-p coefficients
    -ln F(m; p) = a1(m) p + a2(m) p^2 + O(p^3).
Regions: forward part (forward Cliffords and check rounds 0..m-1 with their
resets), inverse block (the m unchecked inverse Cliffords), final check round.
Because the second-order coefficient is a sum over pairs of faults, removing
a region removes every pair that touches it; comparing the variants
separates forward-forward, inverse-inverse and mixed pairs.

    module load python/3.11
    python .../A4_attribution.py [cache_dir]      (~1 minute, one core)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import A4_exact_lrbd as X  # noqa: E402

P_SMALL = np.array([2e-5, 4e-5, 6e-5, 8e-5, 1e-4])


def subset(dem: dict, keep: np.ndarray) -> dict:
    """Restrict a DEM dict to the faults with keep[f] True."""
    rows_keep = keep[dem["owner"]]
    new_index = -np.ones(len(keep), dtype=np.int64)
    new_index[keep] = np.arange(keep.sum())
    out = dict(dem)
    out["rows"] = dem["rows"][rows_keep]
    out["owner"] = new_index[dem["owner"][rows_keep]].astype(np.int32)
    for k in ("scale", "kappa", "opidx", "qudit", "target"):
        out[k] = dem[k][keep]
    return out


def regions(dem: dict, m: int):
    op = dem["opidx"]
    start_final = dem["starts"][m]
    reset_f = (dem["qudit"] >= 5) & (dem["scale"] == 2.0) & \
        (np.isclose(dem["kappa"], 1.5)) & (op < start_final)
    end_fwd = op[reset_f].max() if (m > 0 and reset_f.any()) else -1
    fwd = op <= end_fwd
    inv = (op > end_fwd) & (op < start_final)
    fin = op >= start_final
    return fwd, inv, fin


def a2_of(dem: dict, rounds: list[int]) -> tuple[float, float]:
    prep = X.Prepared(dem, rounds)
    y = np.array([-np.log(prep.evaluate(p)[1]) for p in P_SMALL])
    Q = np.stack([P_SMALL, P_SMALL ** 2, P_SMALL ** 3], 1)
    c, *_ = np.linalg.lstsq(Q, y, rcond=None)
    return c[0], c[1]


def main() -> None:
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("A4_SCRATCH", "/tmp"), "A4_dem_cache")
    seqs = [0, 1, 2]
    for pol in (("unif", 1), ("const", 1)):
        print(f"\npolicy {pol[0]}{pol[1]}: a2(m) averaged over sequences "
              f"{seqs}")
        print("   m    full   no-inverse  no-forward  only-final | "
              "fwd-pairs(+final)  inv-pairs(+final)  fwd x inv")
        rows = []
        for di, m in enumerate(X.DEPTHS):
            if m == 0:
                continue
            vals = []
            for seq in seqs:
                dem = X.load_dem("LRB", seq, di, cache)
                rnds = X.used_rounds(pol, m)
                fwd, inv, fin = regions(dem, m)
                v = [a2_of(dem, rnds)[1],
                     a2_of(subset(dem, ~inv), rnds)[1],
                     a2_of(subset(dem, ~fwd), rnds)[1],
                     a2_of(subset(dem, fin), rnds)[1]]
                vals.append(v)
            v = np.mean(vals, axis=0)
            # pairs: full = FF + II + ff + FI + Ff + If (+ single-region)
            rows.append((m, *v))
            ff = v[1] - v[3]
            ii = v[2] - v[3]
            fi = v[0] - v[1] - v[2] + v[3]
            print(f"  {m:2d}  {v[0]:8.1f}  {v[1]:9.1f}  {v[2]:9.1f}  "
                  f"{v[3]:9.1f} | {ff:12.1f}  {ii:16.1f}  {fi:12.1f}")
        r = np.array(rows)
        ms = r[:, 0]
        for col, lab in ((1, "full"), (2, "no-inverse"), (3, "no-forward")):
            Q = np.stack([np.ones_like(ms), ms, ms ** 2], 1)
            g, *_ = np.linalg.lstsq(Q, r[:, col], rcond=None)
            print(f"   fit {lab:10s}: a2 = {g[0]:8.1f} + {g[1]:7.2f} m + "
                  f"{g[2]:6.3f} m^2")


if __name__ == "__main__":
    main()
