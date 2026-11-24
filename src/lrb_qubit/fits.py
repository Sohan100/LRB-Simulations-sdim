"""Decay fits replicating ``LRBResultsPlotter`` plus sequence-bootstrap errors.

``fit_fixed_offset`` reproduces ``_fit_decay_parameter_fixed_a`` from
``src/lrb/lrb_plotting.py`` (LRB: ``y = a + b f^m`` with ``a = 0`` fixed,
points above ``y_ceiling = 1`` dropped, weights ``1/sigma^2`` from the
standard deviation over Clifford sequences with the repo's floor, five rounds
of a 2001-point coarse-to-fine scan of ``f``). ``fit_free_offset`` reproduces
``_fit_decay_parameter_free_a`` (physical RB).

``bootstrap_fit`` resamples Clifford sequences with replacement and refits, so
the spread of ``f`` reflects sequence-to-sequence variation, the dominant
uncertainty in RB.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def _safe_weights(errors: np.ndarray) -> np.ndarray:
    err = np.asarray(errors, dtype=float)
    err = np.where(np.isnan(err), np.inf, err)
    finite = err[np.isfinite(err)]
    median = float(np.median(finite)) if len(finite) else 1.0
    floor = max(median * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    return 1.0 / (err * err)


def _clean(depths, means, errors, y_ceiling):
    x = np.asarray(depths, dtype=float)
    y = np.asarray(means, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    if y_ceiling is not None:
        ok &= y <= y_ceiling
    return x[ok], y[ok], e[ok]


def fit_fixed_offset(depths: Sequence[float], means: Sequence[float],
                     errors: Sequence[float], *, a_fixed: float = 0.0,
                     y_ceiling: Optional[float] = 1.0, f_min: float = 1e-8,
                     f_max: float = 1.0, min_points: int = 2
                     ) -> Optional[dict[str, float]]:
    """Repo LRB fit ``y = a_fixed + b f^m``."""
    x, y, e = _clean(depths, means, errors, y_ceiling)
    if len(x) < min_points:
        return None
    w = _safe_weights(e)
    y0 = y - a_fixed
    low, high = f_min, f_max
    best = None
    for _ in range(5):
        grid = np.linspace(low, high, 2001)
        z = np.power(grid[None, :], x[:, None])
        num = np.sum(w[:, None] * z * y0[:, None], axis=0)
        den = np.sum(w[:, None] * z * z, axis=0)
        den = np.where(den <= 0, np.nan, den)
        b = num / den
        chi2 = np.sum(w[:, None] * (y0[:, None] - b[None, :] * z) ** 2, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        i = int(np.argmin(chi2))
        best = {"f": float(grid[i]), "a": a_fixed, "b": float(b[i]),
                "chi2": float(chi2[i]), "n_points": float(len(x))}
        span = (high - low) / 10.0
        low = max(f_min, grid[i] - span)
        high = min(f_max, grid[i] + span)
    return best


def fit_free_offset(depths: Sequence[float], means: Sequence[float],
                    errors: Sequence[float], *, f_min: float = 1e-8,
                    f_max: float = 1.0, min_points: int = 2
                    ) -> Optional[dict[str, float]]:
    """Repo RB fit ``y = a + b f^m`` with ``a`` free."""
    x, y, e = _clean(depths, means, errors, None)
    if len(x) < min_points:
        return None
    w = _safe_weights(e)
    s, sy = float(np.sum(w)), float(np.sum(w * y))
    low, high = f_min, f_max
    best = None
    for _ in range(5):
        grid = np.linspace(low, high, 2001)
        z = np.power(grid[None, :], x[:, None])
        sz = np.sum(w[:, None] * z, axis=0)
        szz = np.sum(w[:, None] * z * z, axis=0)
        szy = np.sum(w[:, None] * z * y[:, None], axis=0)
        det = s * szz - sz * sz
        det = np.where(np.abs(det) < 1e-20, np.nan, det)
        a = (szz * sy - sz * szy) / det
        b = (s * szy - sz * sy) / det
        chi2 = np.sum(w[:, None] * (y[:, None] - (a[None, :] + b[None, :] * z))
                      ** 2, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        i = int(np.argmin(chi2))
        best = {"f": float(grid[i]), "a": float(a[i]), "b": float(b[i]),
                "chi2": float(chi2[i]), "n_points": float(len(x))}
        span = (high - low) / 10.0
        low = max(f_min, grid[i] - span)
        high = min(f_max, grid[i] + span)
    return best


def per_sequence_fidelity(accepted: np.ndarray, errors: np.ndarray
                          ) -> np.ndarray:
    """``|1 - 2 e|`` per sequence and depth (NaN where nothing accepted)."""
    accepted = np.asarray(accepted, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(1.0 - 2.0 * np.asarray(errors, float) / accepted)


def fit_from_counts(depths: Sequence[int], accepted: np.ndarray,
                    errors: np.ndarray, kind: str = "lrb"
                    ) -> Optional[dict[str, float]]:
    """Fit the sequence-averaged fidelity the way the repo notebook does."""
    fid = per_sequence_fidelity(accepted, errors)
    with np.errstate(invalid="ignore"):
        mean = np.mean(fid, axis=0)
        std = np.std(fid, axis=0)
    fitter: Callable = fit_fixed_offset if kind == "lrb" else fit_free_offset
    return fitter(depths, mean, std)


def bootstrap_fit(depths: Sequence[int], accepted: np.ndarray,
                  errors: np.ndarray, kind: str = "lrb",
                  num_boot: int = 200, seed: int = 0
                  ) -> dict[str, float]:
    """Point estimate plus sequence-bootstrap 68% interval of ``f``."""
    point = fit_from_counts(depths, accepted, errors, kind)
    rng = np.random.default_rng(seed)
    n = accepted.shape[0]
    values = []
    for _ in range(num_boot):
        pick = rng.integers(0, n, size=n)
        fit = fit_from_counts(depths, accepted[pick], errors[pick], kind)
        if fit is not None:
            values.append(fit["f"])
    out = {"f": np.nan if point is None else point["f"],
           "b": np.nan if point is None else point["b"],
           "a": np.nan if point is None else point["a"],
           "n_points": 0 if point is None else point["n_points"]}
    if values:
        lo, hi = np.percentile(values, [16, 84])
        out.update(f_lo=float(lo), f_hi=float(hi),
                   f_std=float(np.std(values)))
    else:
        out.update(f_lo=np.nan, f_hi=np.nan, f_std=np.nan)
    return out


def error_per_clifford(f: float, dimension: int = 2) -> float:
    """Average logical infidelity per Clifford from the decay parameter."""
    return (dimension - 1) * (1.0 - f) / dimension
