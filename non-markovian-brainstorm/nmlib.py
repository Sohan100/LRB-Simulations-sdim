"""Shared loaders, estimators and fits for the non-Markovian LRB-D brainstorm.

Reads only. Every number comes from the per-sequence count arrays the
production pipeline checkpoints in ``<run>/progress/<p_idx>/{unif,const}_check_data``:
``<k>.npy`` holds accepted tallies with shape (n_depths, n_seq, 3) and
``<k>_rejected.npy`` the rejected-shot counts with shape (n_depths, n_seq).
The ideal terminal outcome is 0 for every sequence (mirror inverse back to
|+>_L, read in X), which the p = 0 rows confirm.

Conventions match ``src/lrb/lrb_plotting.py``: depth m means 2m noisy
Cliffords (forward plus mirror inverse), the fit is y = b f^m with offset 0,
and the reported error is r = 2(1 - f)/3, i.e. per forward+inverse pair.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "LRB-experiment-data-slurm")
OUT = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    ("one", "si1000", "prod"): "Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140",
    ("one", "sd6", "prod"): "Run-2026-10-28-10-24-46-folded_qutrit-single-ancilla-sd6-sdim140",
    ("two", "si1000", "prod"): "Run-2026-10-28-10-26-06-folded_qutrit_split_unif-split-ancilla-si1000-sdim140",
    ("two", "sd6", "prod"): "Run-2026-10-28-10-27-37-folded_qutrit_split_unif-split-ancilla-sd6-sdim140",
    ("one", "sd6", "ext"): "Run-2026-10-30-05-58-58-folded_qutrit-single-ancilla-sd6-sdim140-highp-ext",
    ("two", "sd6", "ext"): "Run-2026-10-30-05-59-13-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-highp-ext",
    # SI1000 extensions were still running when this was written (~3M of 20M
    # shots per sequence); counts are valid but partial.
    ("one", "si1000", "ext"): "Run-2026-10-30-10-01-06-folded_qutrit-single-ancilla-si1000-sdim140-highp-ext",
    ("two", "si1000", "ext"): "Run-2026-10-30-10-01-28-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-highp-ext",
}

OMEGA = np.exp(2j * np.pi / 3)


def _read_list(path: str, cast=float) -> list:
    with open(path) as fh:
        txt = fh.read().strip()
    return [cast(x) for x in txt.split(",") if x.strip()]


@dataclass(frozen=True)
class RunMeta:
    key: tuple
    path: str
    depths: tuple
    probs: tuple


@lru_cache(maxsize=None)
def meta(key: tuple) -> RunMeta:
    path = os.path.join(DATA, RUNS[key])
    return RunMeta(
        key=key,
        path=path,
        depths=tuple(_read_list(os.path.join(path, "depths.txt"), int)),
        probs=tuple(_read_list(os.path.join(path, "probs.txt"), float)),
    )


@lru_cache(maxsize=None)
def counts(key: tuple, p_idx: int, kind: str, k: int):
    """Return (tallies[n_depth, n_seq, 3], rejected[n_depth, n_seq]) as float."""
    base = os.path.join(meta(key).path, "progress", str(p_idx), f"{kind}_check_data")
    t = np.load(os.path.join(base, f"{k}.npy")).astype(float)
    rj = np.load(os.path.join(base, f"{k}_rejected.npy")).astype(float)
    return t, rj


def p_index(key: tuple, p: float) -> int:
    probs = np.asarray(meta(key).probs)
    i = int(np.argmin(np.abs(probs - p)))
    if not np.isclose(probs[i], p):
        raise KeyError(f"p={p} not in {key}: {probs}")
    return i


# ---------------------------------------------------------------- estimators

def per_seq_stats(t: np.ndarray, rj: np.ndarray):
    """Per-(depth, sequence) accepted count, survival, signed and |.| polarization."""
    n_acc = t.sum(-1)
    shots = n_acc + rj
    with np.errstate(invalid="ignore", divide="ignore"):
        P = t / n_acc[..., None]
        pipeline = np.abs(P @ OMEGA ** np.arange(3))  # legacy |sum omega^i P_i|
        signed = (3.0 * P[..., 0] - 1.0) / 2.0  # Re E[omega^residual]
    return n_acc, shots, pipeline, signed


def curves(t: np.ndarray, rj: np.ndarray, seq_idx: np.ndarray | None = None) -> dict:
    """Depth curves for every estimator; ``seq_idx`` resamples sequences."""
    if seq_idx is not None:
        t = t[:, seq_idx]
        rj = rj[:, seq_idx]
    n_acc, shots, pipe, sgn = per_seq_stats(t, rj)
    tot_acc = n_acc.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = (3.0 * t[..., 0].sum(1) / tot_acc - 1.0) / 2.0
        joint = (t[..., 0].sum(1) - 0.5 * (t[..., 1].sum(1) + t[..., 2].sum(1))) / shots.sum(1)
        rice_floor = np.sqrt(np.pi / (4.0 * np.maximum(n_acc, 1))).mean(1)
    ok = n_acc > 0
    pipe_mean = np.where(ok.all(1), np.nanmean(np.where(ok, pipe, np.nan), 1), np.nan)
    pipe_std = np.where(ok.all(1), np.nanstd(np.where(ok, pipe, np.nan), 1), np.nan)
    sgn_mean = np.where(ok.all(1), np.nanmean(np.where(ok, sgn, np.nan), 1), np.nan)
    return {
        "acc": tot_acc / shots.sum(1),
        "n_acc_min": n_acc.min(1),
        "n_acc_tot": tot_acc,
        "pipeline": pipe_mean,
        "pipeline_std": pipe_std,
        "signed_mean": sgn_mean,
        "pooled": pooled,
        "joint": joint,
        "rice_floor": rice_floor,
    }


def bootstrap_curves(t, rj, n_boot=300, seed=0, fields=("pooled", "acc", "joint", "pipeline")):
    rng = np.random.default_rng(seed)
    n_seq = t.shape[1]
    out = {f: [] for f in fields}
    for _ in range(n_boot):
        idx = rng.integers(0, n_seq, n_seq)
        c = curves(t, rj, idx)
        for f in fields:
            out[f].append(c[f])
    return {f: np.asarray(v) for f, v in out.items()}


# ---------------------------------------------------------------------- fits

def err_from_f(f):
    return 2.0 * (1.0 - np.asarray(f, dtype=float)) / 3.0


def fit_fixed_a(depths, y, sigma, a=0.0, f_min=1e-8, f_max=1.0, y_ceiling=1.0):
    """Exact mirror of LRBResultsPlotter._fit_decay_parameter_fixed_a.

    Weighted grid scan for y = a + b f^m with weight 1/sigma^2 (zero or NaN
    sigma floored the way the pipeline does), points above ``y_ceiling``
    dropped, at least two points required.
    """
    x = np.asarray(depths, float)
    y = np.asarray(y, float)
    e = np.asarray(sigma, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    if y_ceiling is not None:
        ok &= y <= y_ceiling
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < 2:
        return None
    err = np.where(np.isnan(e), np.inf, e)
    fin = err[np.isfinite(err)]
    med = float(np.median(fin)) if len(fin) else 1.0
    floor = max(med * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    w = 1.0 / (err * err)
    y0 = y - a
    lo, hi = f_min, f_max
    best = None
    for _ in range(5):
        fg = np.linspace(lo, hi, 2001)
        z = np.power(fg[None, :], x[:, None])
        num = np.sum(w[:, None] * z * y0[:, None], 0)
        den = np.sum(w[:, None] * z * z, 0)
        den = np.where(den <= 0, np.nan, den)
        b = num / den
        chi2 = np.sum(w[:, None] * (y0[:, None] - b[None, :] * z) ** 2, 0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        i = int(np.argmin(chi2))
        best = {"f": float(fg[i]), "b": float(b[i]), "chi2": float(chi2[i]), "n": len(x)}
        span = (hi - lo) / 10.0
        lo, hi = max(f_min, fg[i] - span), min(f_max, fg[i] + span)
    best["r"] = float(err_from_f(best["f"]))
    return best


def fit_log_poly(depths, y, sigma, deg=1):
    """Weighted LSQ of ln y on a polynomial in m; returns per-step coefficients.

    ln y = c0 - c1 m - c2 m^2. sigma_ln = sigma / y. c1 is the initial
    (m -> 0) decay rate of ln y per depth step.
    """
    x = np.asarray(depths, float)
    y = np.asarray(y, float)
    s = np.asarray(sigma, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (y > 0) & (s > 0)
    x, y, s = x[ok], y[ok], s[ok]
    if len(x) < deg + 2:
        return None
    ly = np.log(y)
    sl = s / y
    A = np.vstack([x ** j for j in range(deg + 1)]).T
    W = 1.0 / sl ** 2
    coef, *_ = np.linalg.lstsq(A * np.sqrt(W)[:, None], ly * np.sqrt(W), rcond=None)
    resid = (ly - A @ coef) / sl
    return {"c": -coef, "chi2": float(resid @ resid), "dof": len(x) - deg - 1}


def windowed_rates(depths, y):
    """Per-depth-step error rate between consecutive depths: 2/3 (1 - ratio^(1/dm))."""
    d = np.asarray(depths, float)
    y = np.asarray(y, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = y[1:] / y[:-1]
        f = np.sign(ratio) * np.abs(ratio) ** (1.0 / np.diff(d))
    return 0.5 * (d[1:] + d[:-1]), err_from_f(f)


def check_slots_unif(m: int, k: int) -> list:
    """Same rule as LRBSimulationPipeline.postselect_uniform_interval."""
    slots = [m]
    if m >= k and k > 0:
        slots.extend(j * k - 1 for j in range(1, m // k + 1) if j * k - 1 < m)
    return slots


def unchecked_tail(m: int, k: int) -> int:
    """Noisy Cliffords after the last postselected forward check (forward rest + m inverses)."""
    return (m % k if k > 0 else m) + m


def rb_curve(key: tuple, p_idx: int):
    """Physical RB mean/std per depth from results/RB/<p>.csv (pipeline output)."""
    path = os.path.join(meta(key).path, "results", "RB", f"{p_idx}.csv")
    rows = [line.strip().split(",") for line in open(path) if line.strip()]
    mean = np.array([float(v) for v in rows[1][1:]])
    std = np.array([float(v) for v in rows[2][1:]])
    return mean, std
