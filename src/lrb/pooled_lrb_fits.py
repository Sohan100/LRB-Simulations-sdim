"""Pooled signed-polarization LRB fits: logical cycle error r_L and discard rate d_L.

An alternative to the production fit in ``lrb_plotting.py``, which it does
not replace or modify. The production statistic averages
``|sum_i w^i P(i | accepted)|`` over sequences with equal weight; with few
accepted shots that absolute value is biased upward, and equal weighting
does not match the ratio-of-sums form a lossy, twirled step map predicts.
This module instead uses, per depth,

    lambda(m) = (3 P(0 | accepted) - 1) / 2,  pooled over sequences,

which needs the ideal outcome to be 0 (true for every LRB/RB circuit here),
and reads the per-sequence counts the pipeline checkpoints in
``<run>/progress/<p>/{unif,const}_check_data/<k>.npy`` and
``<k>_rejected.npy``.

Per (policy, p) it reports:

* ``r_L = 2 (1 - f) / 3`` from ``lambda = b f^m`` (offset 0, binomial
  weights), on depths with at least ``min_accepted`` accepted shots and, if
  ``aligned``, multiples of the interval k. Depth 0 is always in the fit;
  there is no option to leave it out;
* ``d_L = 1 - a``, the discard rate per step from ``A(m) ~ a^m``;
* ``r_uncond = 2 (1 - a f) / 3``, the decay of ``A(m) lambda(m)``, which
  scores a discarded shot as a random guess;
* a sequence-bootstrap standard error, the single-exponential chi^2/dof and
  the ratio of rates fitted on the deeper and shallower halves of the depths
  (near 1 for a decay that does not depend on the depth window).

One step is one forward Clifford plus its check round in ``compiled`` runs
(``inverse_mode.txt`` present), and a forward+inverse pair in mirror runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

MIRROR = "mirror"


@dataclass
class RunParams:
    path: str
    depths: np.ndarray
    probs: list[float]
    unif_checks: list[int]
    inverse_mode: str

    @property
    def step_unit(self) -> str:
        return "clifford+check" if self.inverse_mode == "compiled" else "forward+inverse pair"


def _csv_line(path: str) -> list[str]:
    with open(path) as fh:
        return [x for x in fh.read().strip().split(",") if x.strip()]


def read_run_params(run: str) -> RunParams:
    mode_file = os.path.join(run, "inverse_mode.txt")
    mode = open(mode_file).read().strip() if os.path.exists(mode_file) else MIRROR
    return RunParams(
        path=run,
        depths=np.asarray([int(x) for x in _csv_line(os.path.join(run, "depths.txt"))]),
        probs=[float(x) for x in _csv_line(os.path.join(run, "probs.txt"))],
        unif_checks=[int(x) for x in _csv_line(os.path.join(run, "check_unif.txt"))],
        inverse_mode=mode,
    )


def load_policy_counts(run: str, p_idx: int, kind: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Accepted tallies (n_depths, n_seq, 3) and rejected counts (n_depths, n_seq)."""
    base = os.path.join(run, "progress", str(p_idx), f"{kind}_check_data")
    return (np.load(os.path.join(base, f"{k}.npy")).astype(float),
            np.load(os.path.join(base, f"{k}_rejected.npy")).astype(float))


def pooled_curves(tallies: np.ndarray, rejected: np.ndarray, seq_idx: np.ndarray | None = None) -> dict:
    """Pooled acceptance, signed polarization and its binomial SE per depth."""
    if seq_idx is not None:
        tallies, rejected = tallies[:, seq_idx], rejected[:, seq_idx]
    n_acc = tallies.sum(axis=(1, 2))
    shots = n_acc + rejected.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = tallies[..., 0].sum(axis=1) / n_acc
        lam = (3.0 * p0 - 1.0) / 2.0
        se = np.sqrt(np.clip(9.0 * p0 * (1.0 - p0) / 4.0, 1e-12, None) / np.maximum(n_acc, 1))
    acc = n_acc / shots
    return {"acc": acc, "lam": lam, "lam_se": se, "n_acc": n_acc, "shots": shots, "joint": acc * lam}


def fit_offset0_decay(depths, y, sigma, f_min: float = 1e-8, f_max: float = 1.0) -> dict | None:
    """Weighted ``y = b f^m`` fit by the production code's coarse-to-fine f scan."""
    x, y, e = (np.asarray(v, dtype=float) for v in (depths, y, sigma))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < 2:
        return None
    fin = e[np.isfinite(e) & (e > 0)]
    floor = max((float(np.median(fin)) if len(fin) else 1.0) * 1e-3, 1e-12)
    w = 1.0 / np.where(e <= 0, floor, e) ** 2
    lo, hi, best = f_min, f_max, None
    for _ in range(5):
        fg = np.linspace(lo, hi, 2001)
        z = fg[None, :] ** x[:, None]
        den = np.sum(w[:, None] * z * z, axis=0)
        b = np.sum(w[:, None] * z * y[:, None], axis=0) / np.where(den <= 0, np.nan, den)
        chi2 = np.sum(w[:, None] * (y[:, None] - b[None, :] * z) ** 2, axis=0)
        i = int(np.argmin(np.where(np.isfinite(chi2), chi2, np.inf)))
        best = {"f": float(fg[i]), "b": float(b[i]), "chi2": float(chi2[i]), "n": len(x)}
        span = (hi - lo) / 10.0
        lo, hi = max(f_min, fg[i] - span), min(f_max, fg[i] + span)
    best["r"] = 2.0 * (1.0 - best["f"]) / 3.0
    return best


def _log_linear_chi2(depths, y, se) -> float:
    ly, sl = np.log(y), se / y
    A = np.vstack([np.ones_like(depths, dtype=float), depths]).T
    W = 1.0 / sl
    coef, *_ = np.linalg.lstsq(A * W[:, None], ly * W, rcond=None)
    dof = len(depths) - 2
    return float(np.sum(((ly - A @ coef) / sl) ** 2) / dof) if dof > 0 else np.nan


@dataclass
class PooledFit:
    k: int
    depths_used: list[int] = field(default_factory=list)
    r_L: float = np.nan
    r_L_se: float = np.nan
    d_L: float = np.nan
    r_uncond: float = np.nan
    chi2_dof: float = np.nan
    window_ratio: float = np.nan
    min_accepted_used: float = np.nan
    boot_r_L: np.ndarray | None = None


def _select(depths, curves, k, min_accepted, aligned):
    use = np.isfinite(curves["lam"]) & (curves["lam"] > 0) & (curves["n_acc"] >= min_accepted)
    if aligned and k > 0:
        use &= depths % k == 0
    return use


def fit_pooled_lrb(depths, tallies, rejected, k: int, min_accepted: int = 1000,
                   aligned: bool = True, n_boot: int = 200, seed: int = 0) -> PooledFit:
    depths = np.asarray(depths)
    c = pooled_curves(tallies, rejected)
    use = _select(depths, c, k, min_accepted, aligned)
    out = PooledFit(k=k, depths_used=[int(d) for d in depths[use]])
    if use.sum() < 2:
        return out
    fit = fit_offset0_decay(depths[use], c["lam"][use], c["lam_se"][use])
    out.r_L = fit["r"]
    out.min_accepted_used = float(c["n_acc"][use].min())
    slope = np.polyfit(depths[use], np.log(c["acc"][use]), 1)[0]
    a = float(np.exp(slope))
    out.d_L = 1.0 - a
    out.r_uncond = 2.0 * (1.0 - a * fit["f"]) / 3.0
    if use.sum() >= 3:
        out.chi2_dof = _log_linear_chi2(depths[use], c["lam"][use], c["lam_se"][use])
    if use.sum() >= 4:
        d, y, s = depths[use], c["lam"][use], c["lam_se"][use]
        h = len(d) // 2
        early = fit_offset0_decay(d[:h + 1], y[:h + 1], s[:h + 1])
        late = fit_offset0_decay(d[h:], y[h:], s[h:])
        if early and late and early["r"] > 0:
            out.window_ratio = late["r"] / early["r"]
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        n_seq = tallies.shape[1]
        boots = []
        for _ in range(n_boot):
            cb = pooled_curves(tallies, rejected, rng.integers(0, n_seq, n_seq))
            yb = cb["lam"][use]
            fb = fit_offset0_decay(depths[use], yb, c["lam_se"][use]) if np.all(yb > 0) else None
            boots.append(fb["r"] if fb else np.nan)
        out.boot_r_L = np.asarray(boots)
        if np.isfinite(out.boot_r_L).sum() > 10:
            out.r_L_se = float(np.nanstd(out.boot_r_L))
    return out


def rb_rate(run: str, p_idx: int, depths) -> float:
    """Physical RB error per step from ``results/RB/<p>.csv`` (offset 0, std weights, all depths)."""
    path = os.path.join(run, "results", "RB", f"{p_idx}.csv")
    if not os.path.exists(path):
        return np.nan
    rows = [line.strip().split(",") for line in open(path) if line.strip()]
    mean = np.asarray([float(v) for v in rows[1][1:]])
    std = np.asarray([float(v) for v in rows[2][1:]])
    fit = fit_offset0_decay(np.asarray(depths), mean, std)
    return fit["r"] if fit else np.nan


def pseudo_threshold(ps, r_lrb, r_rb) -> tuple[float, str]:
    """p where r_LRB = r_RB: the last crossing from below, interpolated in log ratio."""
    ps, r_lrb, r_rb = (np.asarray(v, dtype=float) for v in (ps, r_lrb, r_rb))
    ok = np.isfinite(r_lrb) & np.isfinite(r_rb) & (r_lrb > 0) & (r_rb > 0)
    ps, lr = ps[ok], np.log(r_lrb[ok] / r_rb[ok])
    if len(ps) < 2:
        return np.nan, "too few points"
    worse = lr >= 0
    if not worse[-1]:
        return np.nan, f"no crossing below p={ps[-1]:g}"
    if worse.all():
        return np.nan, f"LRB worse at every p >= {ps[0]:g}"
    i = int(np.max(np.where(~worse)[0]))
    return float(ps[i] + (0.0 - lr[i]) * (ps[i + 1] - ps[i]) / (lr[i + 1] - lr[i])), "interp"


def _per_p_fits(run: str, k: int, min_accepted: int, aligned: bool, n_boot: int, seed: int,
                p_range=(-np.inf, np.inf)) -> list[tuple]:
    params = read_run_params(run)
    out = []
    for p_idx, p in enumerate(params.probs):
        if p == 0 or not (p_range[0] <= p < p_range[1]):
            continue
        try:
            t, rj = load_policy_counts(run, p_idx, "unif", k)
        except FileNotFoundError:
            continue
        fit = fit_pooled_lrb(params.depths, t, rj, k, min_accepted=min_accepted,
                             aligned=aligned, n_boot=n_boot, seed=seed + 1000 * k + p_idx)
        out.append((p, fit, rb_rate(run, p_idx, params.depths), p_idx, params))
    return out


def build_pooled_tables(run: str, checks=(1, 2, 3, 4), min_accepted: int = 1000,
                        aligned: bool = True, n_boot: int = 200, seed: int = 12345,
                        highp_run: str | None = None,
                        split: float | None = None) -> tuple[list[dict], list[dict]]:
    """Per-(k, p) rows and per-k pseudo-thresholds with a bootstrap 68% interval.

    Every LRB and RB fit includes depth 0. With ``highp_run`` and ``split``,
    rows with p >= split come from ``highp_run`` instead, as in
    ``scripts/merge_highp_threshold_extension.py``.
    """
    if (highp_run is None) != (split is None):
        raise ValueError("Pass both highp_run and split, or neither.")
    rows, thresholds = [], []
    for k in checks:
        kw = dict(k=k, min_accepted=min_accepted, aligned=aligned, n_boot=n_boot, seed=seed)
        if highp_run is None:
            per_p = _per_p_fits(run, **kw)
        else:
            per_p = (_per_p_fits(run, p_range=(-np.inf, split), **kw)
                     + _per_p_fits(highp_run, p_range=(split, np.inf), **kw))
        for p, fit, r_rb, p_idx, params in per_p:
            rows.append({
                "k": k, "p_index": p_idx, "p": p, "source": os.path.basename(params.path.rstrip("/")),
                "step_unit": params.step_unit,
                "r_L": fit.r_L, "r_L_se": fit.r_L_se, "d_L": fit.d_L, "r_uncond": fit.r_uncond,
                "r_RB": r_rb, "chi2_dof": fit.chi2_dof, "window_ratio": fit.window_ratio,
                "depths_used": " ".join(map(str, fit.depths_used)),
                "min_accepted_used": fit.min_accepted_used,
            })
        ps = [x[0] for x in per_p]
        r_rb = [x[2] for x in per_p]
        p_star, note = pseudo_threshold(ps, [x[1].r_L for x in per_p], r_rb)
        lo = hi = np.nan
        if n_boot > 0 and per_p:
            bs = []
            for b in range(n_boot):
                vals = [x[1].boot_r_L[b] if x[1].boot_r_L is not None else np.nan for x in per_p]
                bs.append(pseudo_threshold(ps, vals, r_rb)[0])
            bs = np.asarray(bs)
            if np.isfinite(bs).sum() > 0.8 * n_boot:
                lo, hi = np.nanpercentile(bs, [16, 84])
        p_unc, note_unc = pseudo_threshold(ps, [x[1].r_uncond for x in per_p], r_rb)
        step = per_p[0][4].step_unit if per_p else ""
        thresholds.append({"k": k, "step_unit": step, "p_star_r_L": p_star, "ci68_low": lo,
                           "ci68_high": hi, "note": note, "p_star_r_uncond": p_unc, "note_uncond": note_unc})
    return rows, thresholds
