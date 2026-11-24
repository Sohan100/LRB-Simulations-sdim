"""Fidelity metrics and decay fits for corrected and postselected LRB data.

Three per-circuit fidelity metrics are computed from the distribution of the
residual logical shift ``k`` (0 = correct) among accepted shots:

* ``legacy``: ``|sum_k omega^k P(k)|``, the repository's LRB metric. It is
  blind to a systematic shift, so a decoder that is consistently wrong by +1
  would score 1; it is reported only for continuity with the baseline.
* ``signed``: ``Re sum_k omega^k P(k)``, the logical-X polarization.
* ``survival``: ``P(k = 0)``. For a depolarizing logical channel
  ``survival = (1 + 2 * signed) / 3``.

Decay fits mirror ``lrb.lrb_plotting`` (weighted grid scan over ``f`` with
the offset either fixed at 0 or free), and report ``F = (1 + 2 f) / 3``.
"""

from __future__ import annotations

import numpy as np

OMEGA = np.exp(2j * np.pi * np.arange(3) / 3)


def fidelity_metrics(counts: np.ndarray) -> dict[str, float]:
    """
    Compute the three fidelity metrics from residual counts or probabilities.

    Args:
        counts (np.ndarray): ``(3,)`` counts or probabilities of residual
            logical shift 0, 1, 2 among accepted shots.

    Returns:
        dict[str, float]: ``legacy``, ``signed`` and ``survival``; NaN when
        nothing was accepted.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return {"legacy": np.nan, "signed": np.nan, "survival": np.nan}
    probs = counts / total
    value = probs @ OMEGA
    return {"legacy": float(abs(value)), "signed": float(value.real), "survival": float(probs[0])}


def _safe_weights(errors: np.ndarray) -> np.ndarray:
    """Mirror ``LRBResultsPlotter._safe_weights`` (median-based error floor)."""
    err = np.asarray(errors, dtype=float)
    err = np.where(np.isnan(err), np.inf, err)
    finite = err[np.isfinite(err)]
    median = float(np.median(finite)) if len(finite) else 1.0
    floor = max(median * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    return 1.0 / (err * err)


def fit_decay(depths, means, errors, free_offset: bool = False, f_min: float = 1e-8, f_max: float = 1.0) -> dict | None:
    """
    Fit ``y = a + b f^m`` by a weighted coarse-to-fine grid scan over ``f``.

    With ``free_offset=False`` the offset ``a`` is fixed at 0, as in the
    pipeline's LRB fits; otherwise ``a`` and ``b`` are solved analytically at
    each ``f`` as in the pipeline's RB fits.

    Args:
        depths: Depth axis.
        means: Mean values.
        errors: Uncertainties used as weights ``1/err^2``.
        free_offset (bool): Fit a free offset.
        f_min (float): Lower bound on ``f``.
        f_max (float): Upper bound on ``f``.

    Returns:
        dict | None: ``f``, ``a``, ``b``, ``chi2``, ``n_points``, ``F``,
        ``r`` (= 1 - F), or ``None`` with fewer than two usable points.
    """
    x = np.asarray(depths, dtype=float)
    y = np.asarray(means, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < 2:
        return None
    w = _safe_weights(e)
    low, high = float(f_min), float(f_max)
    best = None
    for _ in range(5):
        grid = np.linspace(low, high, 2001)
        z = np.power(grid[None, :], x[:, None])
        if free_offset:
            s = w.sum()
            sy = (w * y).sum()
            sz = (w[:, None] * z).sum(axis=0)
            szz = (w[:, None] * z * z).sum(axis=0)
            szy = (w[:, None] * z * y[:, None]).sum(axis=0)
            det = s * szz - sz * sz
            det = np.where(np.abs(det) < 1e-20, np.nan, det)
            a = (szz * sy - sz * szy) / det
            b = (s * szy - sz * sy) / det
        else:
            num = (w[:, None] * z * y[:, None]).sum(axis=0)
            den = (w[:, None] * z * z).sum(axis=0)
            den = np.where(den <= 0, np.nan, den)
            a = np.zeros_like(grid)
            b = num / den
        resid = y[:, None] - (a[None, :] + b[None, :] * z)
        chi2 = (w[:, None] * resid * resid).sum(axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        idx = int(np.argmin(chi2))
        f_best = float(grid[idx])
        best = {"f": f_best, "a": float(a[idx]), "b": float(b[idx]),
                "chi2": float(chi2[idx]), "n_points": int(len(x))}
        span = (high - low) / 10.0
        low, high = max(f_min, f_best - span), min(f_max, f_best + span)
    best["F"] = (1.0 + 2.0 * best["f"]) / 3.0
    best["r"] = 1.0 - best["F"]
    return best


def windowed_rates(depths, values) -> list[dict]:
    """
    Depth-windowed decay factors between consecutive depths.

    ``rate_i = (y_{i+1} / y_i) ** (1 / (m_{i+1} - m_i))``.

    Args:
        depths: Depth axis.
        values: Positive values (fidelity or acceptance).

    Returns:
        list[dict]: One entry per consecutive depth pair.
    """
    out = []
    for i in range(len(depths) - 1):
        m0, m1 = depths[i], depths[i + 1]
        y0, y1 = values[i], values[i + 1]
        rate = (y1 / y0) ** (1.0 / (m1 - m0)) if y0 > 0 and y1 > 0 else np.nan
        out.append({"m0": m0, "m1": m1, "rate": float(rate)})
    return out
