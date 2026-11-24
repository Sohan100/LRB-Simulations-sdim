"""Soft-output acceptance rules for qubit ``[[5,1,2]]`` LRB (work package E).

Three groups of tools live here.

Gap (``|LLR|``) postselection
    The exact MLD in ``mld.py`` returns ``LLR = log P(L=0|s) - log P(L=1|s)``.
    A shot with ``|LLR| = g`` is decoded wrongly with conditional probability
    ``q(g) = 1 / (1 + exp(g))``. Accepting shots with ``|LLR| >= t`` (and
    decoding them) is the Neyman-Pearson / Chow rule: among all (possibly
    randomized) acceptance rules that look at the same detection events and
    accept the same fraction of shots, it has the fewest accepted logical
    errors (fractional-knapsack argument, see the E note). ``experiment.py``
    stores ``|LLR|`` histograms (``GAP_EDGES``) split into correct and wrong
    decisions; ``cumulative_counts`` turns them into threshold rules, and
    ``ParetoCurve`` into pooled acceptance-vs-error curves. Points between bin
    edges are randomized rules (accept a fraction ``lam`` of the boundary bin),
    so every interpolated point is achievable.

Repo-style statistics and fits
    ``repo_fidelity_stats`` reproduces ``CountTable.summary`` (mean and
    population std over sequences of ``F = |1 - 2 e / a|``).
    ``fit_fixed_offset`` reproduces ``LRBResultsPlotter._fit_lrb_from_stats``
    (``_mask_invalid``, ``_apply_ceiling_cut`` and
    ``_fit_decay_parameter_fixed_a`` with ``_safe_weights``, default
    ``LRBPlotFitConfig``), ``fit_free_offset`` the physical-RB fit
    (``_fit_decay_parameter_free_a``). ``fit_fixed_offset_batch`` is a
    vectorized version for sequence bootstraps.

Rule-based hybrids
    ``SlotLayout`` groups detectors into time slots (check rounds, plus the
    final data-readout slot when it is used). ``evaluate_cluster_rule`` splits
    each shot's detection events into clusters of consecutive non-empty slots
    and looks every cluster up in a table. A cluster spanning three or more
    slots, or one missing from the table, rejects the shot; accepted clusters
    contribute a logical-flip correction (XOR over clusters). Tables:

    ``structural_tables``
        Decoder-free tables from the code structure only: isolated ancilla
        measurement errors (time-like detection pairs, using the logical
        gates' stabilizer maps ``LRBCircuit.back``), lone terminal Z-check
        events, and optionally qubit-1 syndromes. None of them flips
        ``x0 + x3`` (data qubit 1 is never coupled to other data qubits by the
        logical gates, and ancilla faults that flip only a measurement do not
        touch the data), so the correction is always the identity.
    ``dem_lookup_tables``
        Single-fault lookup from the detector error model: a cluster must be
        exactly the signature of one error mechanism, and the mechanisms with
        that signature must agree on the logical flip (``rho = 0``) or have
        first-order posterior odds of at least ``1 / rho``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import circuits as Q
from . import code as C
from . import mld as M
from .experiment import GAP_EDGES

# Default ``LRBPlotFitConfig`` values used by the repo's plots.
REPO_A_FIXED = 0.0
REPO_Y_CEILING = 1.0
REPO_MIN_POINTS = 2
REPO_F_MIN = 1e-8
REPO_F_MAX = 1.0
_SCAN_POINTS = 2001
_SCAN_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Soft output basics
# ---------------------------------------------------------------------------

def gap_error_probability(gap) -> np.ndarray:
    """Conditional error probability ``1 / (1 + exp(|LLR|))`` of an MLD shot."""
    gap = np.abs(np.asarray(gap, dtype=float))
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(gap))


def cumulative_counts(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accepted and wrong counts of every bin-edge threshold rule.

    Args:
        hist: ``(..., bins, 2)`` counts of (correct, wrong) MLD decisions per
            ``|LLR|`` bin (``PointResult.gap_hist`` layout).

    Returns:
        ``(accepted, errors)`` of shape ``(..., bins + 1)``. Index ``k`` is the
        rule "accept bins ``>= k``", i.e. ``|LLR| >= edges[k]``; index ``0``
        accepts everything and index ``bins`` accepts nothing.
    """
    hist = np.asarray(hist)
    total = hist.sum(axis=-1)
    wrong = hist[..., 1]
    zeros = np.zeros(total.shape[:-1] + (1,), dtype=total.dtype)
    acc = np.concatenate([np.cumsum(total[..., ::-1], axis=-1)[..., ::-1],
                          zeros], axis=-1)
    err = np.concatenate([np.cumsum(wrong[..., ::-1], axis=-1)[..., ::-1],
                          zeros], axis=-1)
    return acc, err


def interpolate_counts(acc: np.ndarray, err: np.ndarray, k: int, lam: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Counts of the randomized rule (all bins ``>= k``) + ``lam`` x bin ``k-1``.

    Works on any leading shape; the last axis is the threshold index.
    """
    acc = np.asarray(acc, dtype=float)
    err = np.asarray(err, dtype=float)
    if k <= 0 or lam == 0.0:
        return acc[..., k], err[..., k]
    return (acc[..., k] + lam * (acc[..., k - 1] - acc[..., k]),
            err[..., k] + lam * (err[..., k - 1] - err[..., k]))


def match_acceptance(acc: np.ndarray, target: float) -> tuple[int, float]:
    """Randomized bin-edge rule whose (pooled) accepted count equals ``target``.

    Args:
        acc: 1-D non-increasing accepted counts per threshold index.
        target: Desired accepted count (same units as ``acc``).

    Returns:
        ``(k, lam)`` for ``interpolate_counts``.
    """
    acc = np.asarray(acc, dtype=float)
    if target >= acc[0]:
        return 0, 0.0
    if target <= acc[-1]:
        return len(acc) - 1, 0.0
    # Smallest k with acc[k] <= target; then acc[k-1] > target.
    k = int(np.argmax(acc <= target))
    lo, hi = acc[k], acc[k - 1]
    lam = 0.0 if hi == lo else float((target - lo) / (hi - lo))
    return k, min(max(lam, 0.0), 1.0)


def match_error_rate(acc: np.ndarray, err: np.ndarray, target_rate: float
                     ) -> tuple[int, float, float]:
    """Largest-acceptance randomized bin-edge rule with error rate ``<= target``.

    Along each segment (adding a fraction ``lam`` of one bin) the error rate
    moves monotonically between its end points, so every segment is solved in
    closed form and the best segment wins.

    Returns:
        ``(k, lam, accepted)``; ``accepted`` is 0 when no rule qualifies.
    """
    acc = np.asarray(acc, dtype=float)
    err = np.asarray(err, dtype=float)
    best = (len(acc) - 1, 0.0, 0.0)
    for k in range(len(acc) - 1, 0, -1):
        a0, e0 = acc[k], err[k]
        da, de = acc[k - 1] - a0, err[k - 1] - e0
        if da <= 0:
            continue
        # rate(lam) = (e0 + lam de) / (a0 + lam da) <= target
        #   <=> lam * slope <= slack
        slack = target_rate * a0 - e0
        slope = de - target_rate * da
        if slope > 0:
            if slack < 0:
                continue
            lam = min(1.0, slack / slope)
        elif slope < 0:
            if slack < 0 and slack / slope > 1.0:
                continue
            lam = 1.0
        else:
            if slack < 0:
                continue
            lam = 1.0
        accepted = a0 + lam * da
        if accepted > best[2]:
            best = (k, float(lam), float(accepted))
    return best


@dataclass
class ParetoCurve:
    """Pooled acceptance-vs-error curve of the gap rule at one (p, depth).

    Attributes:
        edges: Bin edges (``GAP_EDGES``); threshold index ``k`` accepts
            ``|LLR| >= edges[k]``.
        shots: Total shots.
        accepted, errors: Pooled counts per threshold index (``bins + 1``).
    """

    edges: np.ndarray
    shots: float
    accepted: np.ndarray
    errors: np.ndarray

    @classmethod
    def from_hist(cls, hist: np.ndarray, edges: np.ndarray = GAP_EDGES
                  ) -> "ParetoCurve":
        """Pool ``(..., bins, 2)`` histograms over all leading axes."""
        hist = np.asarray(hist)
        pooled = hist.reshape(-1, hist.shape[-2], 2).sum(axis=0)
        acc, err = cumulative_counts(pooled)
        return cls(np.asarray(edges), float(acc[0]), acc.astype(float),
                   err.astype(float))

    @property
    def acceptance(self) -> np.ndarray:
        return self.accepted / self.shots

    @property
    def error_rate(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.errors / self.accepted

    def error_at_acceptance(self, alpha: float) -> dict[str, float]:
        """Gap-rule errors at acceptance ``alpha``.

        Returns the achievable randomized rule (``rate``, conservative: it
        accepts a random fraction of the boundary bin, an upper bound on the
        exact threshold rule there) and an optimistic interpolation
        (``rate_low``) that charges the boundary-bin shots the smallest error
        probability ``q(upper edge)`` allowed by their ``|LLR|``.
        """
        target = alpha * self.shots
        k, lam = match_acceptance(self.accepted, target)
        acc, err = interpolate_counts(self.accepted, self.errors, k, lam)
        out = {"k": k, "lam": lam, "accepted": float(acc),
               "errors": float(err),
               "rate": float(err / acc) if acc > 0 else np.nan}
        if k > 0 and lam > 0:
            extra = lam * (self.accepted[k - 1] - self.accepted[k])
            q_low = float(gap_error_probability(self.edges[k]))
            low = self.errors[k] + extra * q_low
        else:
            low = err
        out["rate_low"] = float(low / acc) if acc > 0 else np.nan
        return out

    def acceptance_at_error(self, rate: float) -> dict[str, float]:
        k, lam, accepted = match_error_rate(self.accepted, self.errors, rate)
        acc, err = interpolate_counts(self.accepted, self.errors, k, lam)
        return {"k": k, "lam": lam, "acceptance": float(acc / self.shots),
                "rate": float(err / acc) if acc > 0 else np.nan}


# ---------------------------------------------------------------------------
# Repo-style statistics and decay fits
# ---------------------------------------------------------------------------

def fidelity(accepted, errors) -> np.ndarray:
    """``|1 - 2 errors / accepted|`` (NaN where nothing was accepted)."""
    accepted = np.asarray(accepted, dtype=float)
    errors = np.asarray(errors, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.abs(1.0 - 2.0 * errors / accepted)
    return np.where(accepted > 0, out, np.nan)


def repo_fidelity_stats(accepted, errors, axis: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std over sequences, as in ``CountTable.summary``."""
    fid = fidelity(accepted, errors)
    with np.errstate(invalid="ignore"):
        return np.mean(fid, axis=axis), np.std(fid, axis=axis)


def safe_weights(errors) -> np.ndarray:
    """Copy of ``LRBResultsPlotter._safe_weights``."""
    err = np.asarray(errors, dtype=float)
    err = np.where(np.isnan(err), np.inf, err)
    finite = err[np.isfinite(err)]
    median = float(np.median(finite)) if len(finite) else 1.0
    floor = max(median * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    return 1.0 / (err * err)


def _prepare(depths, means, errors, y_ceiling):
    x = np.asarray(depths, dtype=float)
    y = np.asarray(means, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(e)          # _mask_invalid
    if y_ceiling is not None:
        ok &= y <= y_ceiling                    # _apply_ceiling_cut
    x, y, e = x[ok], y[ok], e[ok]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    return x[ok], y[ok], e[ok]


def fit_fixed_offset(depths, means, errors, *, a_fixed: float = REPO_A_FIXED,
                     y_ceiling: Optional[float] = REPO_Y_CEILING,
                     min_points: int = REPO_MIN_POINTS,
                     f_min: float = REPO_F_MIN, f_max: float = REPO_F_MAX
                     ) -> Optional[dict[str, float]]:
    """Repo LRB fit ``F = a_fixed + b f^m`` (weighted coarse-to-fine scan)."""
    x, y, e = _prepare(depths, means, errors, y_ceiling)
    if len(x) < min_points:
        return None
    w = safe_weights(e)
    y0 = y - float(a_fixed)
    low, high = float(f_min), float(f_max)
    best = None
    for _ in range(_SCAN_ITERATIONS):
        f_grid = np.linspace(low, high, _SCAN_POINTS)
        z = np.power(f_grid[None, :], x[:, None])
        num = np.sum(w[:, None] * z * y0[:, None], axis=0)
        den = np.sum(w[:, None] * z * z, axis=0)
        den = np.where(den <= 0, np.nan, den)
        b = num / den
        resid = y0[:, None] - b[None, :] * z
        chi2 = np.sum(w[:, None] * resid * resid, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        idx = int(np.argmin(chi2))
        f_best = float(f_grid[idx])
        best = {"f": f_best, "a": float(a_fixed), "b": float(b[idx]),
                "chi2": float(chi2[idx]), "n_points": float(len(x))}
        span = (high - low) / 10.0
        low = max(f_min, f_best - span)
        high = min(f_max, f_best + span)
    return best


def fit_free_offset(depths, means, errors, *, min_points: int = REPO_MIN_POINTS,
                    f_min: float = REPO_F_MIN, f_max: float = REPO_F_MAX
                    ) -> Optional[dict[str, float]]:
    """Repo physical-RB fit ``F = a + b f^m`` (no ceiling cut, as in the repo)."""
    x, y, e = _prepare(depths, means, errors, None)
    if len(x) < min_points:
        return None
    w = safe_weights(e)
    s = float(np.sum(w))
    sy = float(np.sum(w * y))
    low, high = float(f_min), float(f_max)
    best = None
    for _ in range(_SCAN_ITERATIONS):
        f_grid = np.linspace(low, high, _SCAN_POINTS)
        z = np.power(f_grid[None, :], x[:, None])
        sz = np.sum(w[:, None] * z, axis=0)
        szz = np.sum(w[:, None] * z * z, axis=0)
        szy = np.sum(w[:, None] * z * y[:, None], axis=0)
        det = (s * szz) - (sz * sz)
        det = np.where(np.abs(det) < 1e-20, np.nan, det)
        a = (szz * sy - sz * szy) / det
        b = (s * szy - sz * sy) / det
        resid = y[:, None] - (a[None, :] + b[None, :] * z)
        chi2 = np.sum(w[:, None] * resid * resid, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        idx = int(np.argmin(chi2))
        f_best = float(f_grid[idx])
        best = {"f": f_best, "a": float(a[idx]), "b": float(b[idx]),
                "chi2": float(chi2[idx]), "n_points": float(len(x))}
        span = (high - low) / 10.0
        low = max(f_min, f_best - span)
        high = min(f_max, f_best + span)
    return best


def fit_fixed_offset_batch(depths, means: np.ndarray, errors: np.ndarray, *,
                           a_fixed: float = REPO_A_FIXED,
                           y_ceiling: Optional[float] = REPO_Y_CEILING,
                           min_points: int = REPO_MIN_POINTS,
                           f_min: float = REPO_F_MIN, f_max: float = REPO_F_MAX,
                           chunk: int = 64) -> np.ndarray:
    """Vectorized ``fit_fixed_offset`` over rows of ``means`` / ``errors``.

    Returns the fitted ``f`` per row (NaN where the repo fit returns None).
    Dropped points get zero weight, which is identical to removing them.
    The same coarse-to-fine grids are scanned, but ``chi2`` is evaluated in
    the closed form ``sum w y0^2 - num^2 / den`` (equal to the repo's residual
    sum up to rounding), so ``f`` agrees with ``fit_fixed_offset`` to within
    a final grid step. Used for bootstrap replicates only.
    """
    x = np.asarray(depths, dtype=float)
    means = np.atleast_2d(np.asarray(means, dtype=float))
    errors = np.atleast_2d(np.asarray(errors, dtype=float))
    out = np.full(means.shape[0], np.nan)
    steps = np.arange(_SCAN_POINTS, dtype=float)
    for start in range(0, means.shape[0], chunk):
        y = means[start:start + chunk]
        e = errors[start:start + chunk]
        ok = ~np.isnan(y) & ~np.isnan(e) & np.isfinite(x)[None, :]
        if y_ceiling is not None:
            with np.errstate(invalid="ignore"):
                ok &= np.where(np.isnan(y), False, y <= y_ceiling)
        ok &= np.isfinite(y) & np.isfinite(e)
        count = ok.sum(axis=1)
        # _safe_weights per row: the median runs over the kept points only.
        e_kept = np.where(ok, e, np.nan)
        med = np.ones(len(y))
        for row in np.flatnonzero(count > 0):
            med[row] = float(np.median(e_kept[row][ok[row]]))
        floor = np.maximum(med * 1e-3, 1e-12)
        e_safe = np.where(e_kept <= 0, floor[:, None], e_kept)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(ok, 1.0 / (e_safe * e_safe), 0.0)
        y0 = np.where(ok, y - a_fixed, 0.0)
        rows = len(y)
        low = np.full(rows, float(f_min))
        high = np.full(rows, float(f_max))
        f_best = np.full(rows, np.nan)
        wy = w * y0
        syy = np.sum(wy * y0, axis=1)
        for _ in range(_SCAN_ITERATIONS):
            # Same arithmetic as np.linspace(low, high, n).
            step = (high - low) / (_SCAN_POINTS - 1)
            f_grid = steps[None, :] * step[:, None] + low[:, None]
            f_grid[:, -1] = high
            with np.errstate(divide="ignore"):
                log_f = np.log(f_grid)
            z = np.exp(log_f[:, :, None] * x[None, None, :])
            z[:, :, x == 0] = 1.0
            num = np.einsum("bgd,bd->bg", z, wy)
            den = np.einsum("bgd,bd->bg", z * z, w)
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = syy[:, None] - num * num / den
            chi2 = np.where((den > 0) & np.isfinite(chi2), chi2, np.inf)
            idx = np.argmin(chi2, axis=1)
            f_best = f_grid[np.arange(rows), idx]
            span = (high - low) / 10.0
            low = np.maximum(f_min, f_best - span)
            high = np.minimum(f_max, f_best + span)
        out[start:start + rows] = np.where(count >= min_points, f_best, np.nan)
    return out


def decay_to_error_per_clifford(f) -> np.ndarray:
    """Qubit (d = 2) conversion used by the repo: ``r = 1 - (1 + f) / 2``."""
    return 1.0 - (1.0 + np.asarray(f, dtype=float)) / 2.0


# ---------------------------------------------------------------------------
# Sequence bootstrap
# ---------------------------------------------------------------------------

def bootstrap_multiplicities(num_sequences: int, num_boot: int,
                             rng: np.random.Generator) -> np.ndarray:
    """``(num_boot, num_sequences)`` counts of each sequence in a resample."""
    picks = rng.integers(0, num_sequences, size=(num_boot, num_sequences))
    mult = np.zeros((num_boot, num_sequences), dtype=np.int64)
    for b in range(num_boot):
        mult[b] = np.bincount(picks[b], minlength=num_sequences)
    return mult


def weighted_fidelity_stats(fid: np.ndarray, mult: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Repo mean/std over sequences for every resample.

    Args:
        fid: ``(num_sequences, num_depths)`` per-sequence fidelities (NaN
            allowed: a resample containing a NaN sequence gets a NaN mean).
        mult: ``(num_boot, num_sequences)`` multiplicities.

    Returns:
        ``(means, stds)`` of shape ``(num_boot, num_depths)``.
    """
    n = mult.sum(axis=1, keepdims=True).astype(float)
    nan = np.isnan(fid)
    f0 = np.where(nan, 0.0, fid)
    mean = (mult @ f0) / n
    second = (mult @ (f0 * f0)) / n
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    bad = (mult @ nan.astype(float)) > 0
    mean = np.where(bad, np.nan, mean)
    std = np.where(bad, np.nan, std)
    return mean, std


def bootstrap_fixed_offset_fit(depths, accepted: np.ndarray,
                               errors: np.ndarray, mult: np.ndarray
                               ) -> dict[str, float]:
    """Repo fit plus sequence-bootstrap spread of ``f``.

    Args:
        accepted, errors: ``(num_sequences, num_depths)`` counts.
        mult: bootstrap multiplicities.
    """
    mean, std = repo_fidelity_stats(accepted, errors)
    fit = fit_fixed_offset(depths, mean, std)
    fid = fidelity(accepted, errors)
    bmean, bstd = weighted_fidelity_stats(fid, mult)
    boots = fit_fixed_offset_batch(depths, bmean, bstd)
    good = boots[np.isfinite(boots)]
    out = {"f": np.nan if fit is None else fit["f"],
           "b": np.nan if fit is None else fit["b"],
           "f_boot_std": float(np.std(good)) if len(good) > 1 else np.nan,
           "f_ci_low": float(np.quantile(good, 0.025)) if len(good) else np.nan,
           "f_ci_high": float(np.quantile(good, 0.975)) if len(good) else np.nan,
           "boot_ok": float(len(good)) / max(len(boots), 1)}
    out["boot_samples"] = boots
    return out


# ---------------------------------------------------------------------------
# Vectorized matching over bootstrap replicates (rows)
# ---------------------------------------------------------------------------

def batch_error_at_acceptance(acc: np.ndarray, err: np.ndarray,
                              target: np.ndarray,
                              edges: np.ndarray = GAP_EDGES
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray]:
    """Row-wise ``match_acceptance`` + ``interpolate_counts``.

    Args:
        acc, err: ``(rows, K+1)`` pooled cumulative counts.
        target: ``(rows,)`` accepted counts to match.
        edges: Bin edges, for the optimistic boundary-bin bound.

    Returns:
        ``(k, lam, rate, rate_low)``: the randomized rule, its error rate and
        the optimistic bound (see ``ParetoCurve.error_at_acceptance``).
    """
    q_edges = gap_error_probability(edges)
    rows, width = acc.shape
    target = np.asarray(target, dtype=float)
    below = acc <= target[:, None]
    k = np.where(below.any(axis=1), np.argmax(below, axis=1), width - 1)
    k = np.where(target >= acc[:, 0], 0, k)
    r = np.arange(rows)
    km1 = np.maximum(k - 1, 0)
    hi = acc[r, km1]
    lo = acc[r, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = np.where((k > 0) & (hi > lo), (target - lo) / (hi - lo), 0.0)
    lam = np.clip(lam, 0.0, 1.0)
    a = lo + lam * (hi - lo)
    e = err[r, k] + lam * (err[r, km1] - err[r, k])
    e_low = err[r, k] + lam * (hi - lo) * q_edges[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        return k, lam, e / a, e_low / a


def batch_acceptance_at_error(acc: np.ndarray, err: np.ndarray,
                              rate: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise ``match_error_rate``: returns ``(k, lam, accepted)``."""
    a0 = acc[:, 1:]
    e0 = err[:, 1:]
    da = acc[:, :-1] - a0
    de = err[:, :-1] - e0
    t = np.asarray(rate, dtype=float)[:, None]
    slack = t * a0 - e0
    slope = de - t * da
    with np.errstate(invalid="ignore", divide="ignore"):
        lam_pos = np.where(slope > 0, np.minimum(1.0, slack / slope), np.nan)
        ratio = np.where(slope < 0, slack / slope, np.nan)
    feasible = np.where(
        slope > 0, slack >= 0,
        np.where(slope < 0, (slack >= 0) | (ratio <= 1.0), slack >= 0))
    lam = np.where(slope > 0, lam_pos, 1.0)
    feasible &= da > 0
    accepted = np.where(feasible, a0 + np.nan_to_num(lam) * da, 0.0)
    best = np.argmax(accepted, axis=1)
    r = np.arange(acc.shape[0])
    out_acc = accepted[r, best]
    out_lam = np.where(out_acc > 0, np.nan_to_num(lam[r, best]), 0.0)
    k = np.where(out_acc > 0, best + 1, acc.shape[1] - 1)
    return k, out_lam, out_acc


# ---------------------------------------------------------------------------
# Matched gap rule on identical shots
# ---------------------------------------------------------------------------

class MatchedGapScorer:
    """Score any acceptance rule against the gap rule on the same shots.

    For a rule with accepted mask ``A`` and per-shot error indicator ``e_R``,
    the matched gap rule accepts the same number of shots with the largest
    ``|LLR|`` (boundary ties randomized, see ``matched_gap_weights``) and
    decodes them by MLD. ``score`` returns the rule's errors, the matched gap
    rule's expected errors, and the paired difference with its shot-noise
    variance ``sum z_i^2 - (sum z_i)^2 / n``, where
    ``z_i = 1[i in A] e_R(i) - w_G(i) e_G(i)``.
    """

    def __init__(self, llr: np.ndarray, flip: np.ndarray, decimals: int = 9):
        llr = np.asarray(llr, dtype=float)
        self.gap = np.round(np.abs(llr), decimals)
        self.wrong = (llr < 0) != np.asarray(flip, dtype=bool)
        order = np.argsort(-self.gap, kind="stable")
        self.sorted_gap = self.gap[order]
        self.cum_wrong = np.concatenate(
            [[0], np.cumsum(self.wrong[order])]).astype(float)
        self.n = len(llr)

    def boundary(self, n_accept: int) -> tuple[float, int, int, float]:
        """``(boundary gap, start, end, frac)`` of the tie group."""
        if n_accept <= 0:
            return np.inf, 0, 0, 0.0
        if n_accept >= self.n:
            return -np.inf, self.n, self.n, 0.0
        b = self.sorted_gap[n_accept - 1]
        neg = -self.sorted_gap
        start = int(np.searchsorted(neg, -b, side="left"))
        end = int(np.searchsorted(neg, -b, side="right"))
        frac = (n_accept - start) / (end - start)
        return float(b), start, end, float(frac)

    def gap_errors(self, n_accept: int) -> float:
        _, start, end, frac = self.boundary(n_accept)
        if n_accept >= self.n:
            return float(self.cum_wrong[-1])
        return float(self.cum_wrong[start]
                     + frac * (self.cum_wrong[end] - self.cum_wrong[start]))

    def score(self, accept: np.ndarray, rule_wrong: np.ndarray
              ) -> dict[str, float]:
        accept = np.asarray(accept, dtype=bool)
        e_r = accept & np.asarray(rule_wrong, dtype=bool)
        n_acc = int(accept.sum())
        b, start, end, frac = self.boundary(n_acc)
        if n_acc >= self.n:
            w_above = np.ones(self.n, dtype=bool)
            w_tied = np.zeros(self.n, dtype=bool)
        else:
            w_above = self.gap > b
            w_tied = self.gap == b
        eg_above = self.wrong & w_above
        eg_tied = self.wrong & w_tied
        e_gap = float(eg_above.sum() + frac * eg_tied.sum())
        s1 = float(e_r.sum())
        s2 = float(eg_above.sum() + frac * frac * eg_tied.sum())
        s3 = float((e_r & eg_above).sum() + frac * (e_r & eg_tied).sum())
        diff = s1 - e_gap
        sumsq = s1 + s2 - 2.0 * s3
        return {"accepted": float(n_acc), "rule_errors": s1,
                "gap_errors": e_gap, "diff": diff,
                "diff_var": max(sumsq - diff * diff / max(self.n, 1), 0.0),
                "mld_errors_on_accepted": float((accept & self.wrong).sum())}


def matched_gap_weights(gap: np.ndarray, n_accept: float,
                        decimals: int = 9) -> np.ndarray:
    """Inclusion weights of the gap rule that accepts ``n_accept`` shots.

    Shots are ranked by ``|LLR|`` (rounded to ``decimals`` so that identical
    syndromes tie exactly); the boundary tie group is accepted with equal
    probability (randomized threshold), so the weights sum to ``n_accept``.
    """
    gap = np.round(np.abs(np.asarray(gap, dtype=float)), decimals)
    n = len(gap)
    if n_accept <= 0:
        return np.zeros(n)
    if n_accept >= n:
        return np.ones(n)
    sorted_gap = np.sort(gap)[::-1]
    boundary = sorted_gap[int(np.ceil(n_accept)) - 1]
    above = gap > boundary
    tied = gap == boundary
    n_above = int(above.sum())
    frac = (n_accept - n_above) / int(tied.sum())
    return above.astype(float) + frac * tied


# ---------------------------------------------------------------------------
# Rule-based hybrids: slot layout and cluster rules
# ---------------------------------------------------------------------------

@dataclass
class SlotLayout:
    """Detector columns grouped into time slots.

    Attributes:
        columns: For every slot, detector columns ordered by bit position.
        labels: Round label of every slot (``depth + 1`` for the final
            data-readout slot).
        stabilizers: For every slot, the stabilizer index of each bit.
        has_final: Whether the last slot is the final data-readout slot.
    """

    columns: list[np.ndarray]
    labels: list[int]
    stabilizers: list[tuple[int, ...]]
    has_final: bool
    rounds_back: list[tuple[tuple[int, ...], ...]] = field(default_factory=list)

    @classmethod
    def from_circuit(cls, lc: "Q.LRBCircuit", use_final: bool = True
                     ) -> "SlotLayout":
        final_label = lc.depth + 1
        columns, labels, stabs = [], [], []
        for label in lc.rounds:
            cols, ss = [], []
            for index, (r, s) in enumerate(lc.detector_keys):
                if r == label:
                    cols.append(index)
                    ss.append(s)
            order = np.argsort(ss)
            columns.append(np.asarray(cols)[order])
            stabs.append(tuple(int(ss[i]) for i in order))
            labels.append(label)
        has_final = bool(use_final and lc.terminal == "data5")
        back = [lc.back[label] for label in lc.rounds]
        if has_final:
            cols, ss = [], []
            for index, (r, s) in enumerate(lc.detector_keys):
                if r == final_label:
                    cols.append(index)
                    ss.append(s)
            order = np.argsort(ss)
            columns.append(np.asarray(cols)[order])
            stabs.append(tuple(int(ss[i]) for i in order))
            labels.append(final_label)
            final_back = Q._back_subsets(lc.segments[-1], C.X_STABILIZERS)
            back.append(tuple(final_back))
        return cls(columns, labels, stabs, has_final, back)

    @property
    def num_slots(self) -> int:
        return len(self.columns)

    @property
    def bits(self) -> list[int]:
        return [len(c) for c in self.columns]

    @property
    def num_detectors(self) -> int:
        return int(sum(len(c) for c in self.columns))

    def codes(self, dets: np.ndarray) -> np.ndarray:
        """``(shots, slots)`` integer pattern of every slot."""
        dets = np.asarray(dets, dtype=bool)
        out = np.zeros((dets.shape[0], self.num_slots), dtype=np.int64)
        for t, cols in enumerate(self.columns):
            weights = (1 << np.arange(len(cols))).astype(np.int64)
            out[:, t] = dets[:, cols].astype(np.int64) @ weights
        return out

    def code_of(self, slot: int, stabilizers: Sequence[int]) -> int:
        code = 0
        for s in stabilizers:
            code ^= 1 << self.stabilizers[slot].index(s)
        return code

    def measurement_error_pattern(self, slot: int, s: int
                                  ) -> tuple[int, Optional[int]]:
        """Detection pattern of a flipped measurement of ``s`` in ``slot``.

        Returns ``(code at slot, code at slot + 1)``; the second entry is
        ``None`` when nothing later compares with this measurement (terminal
        Z checks, or every terminal check without the final slot).
        """
        first = self.code_of(slot, [s])
        nxt = slot + 1
        if nxt >= self.num_slots:
            return first, None
        image = [s2 for s2, subset in zip(self.stabilizers[nxt],
                                          self.rounds_back[nxt])
                 if s in subset]
        if not image:
            return first, None
        return first, self.code_of(nxt, image)


@dataclass
class ClusterTables:
    """Lookup tables for ``evaluate_cluster_rule``.

    ``single[t][code]`` and ``pair[t][code_t | code_{t+1} << bits_t]`` hold
    ``-1`` (reject), ``0`` (accept, no logical flip) or ``1`` (accept, flip).
    """

    single: list[np.ndarray]
    pair: list[np.ndarray]
    name: str = ""

    @classmethod
    def empty(cls, layout: SlotLayout, name: str = "") -> "ClusterTables":
        bits = layout.bits
        single = [np.full(1 << b, -1, dtype=np.int8) for b in bits]
        pair = [np.full(1 << (bits[t] + bits[t + 1]), -1, dtype=np.int8)
                for t in range(layout.num_slots - 1)]
        return cls(single, pair, name)

    def set_single(self, slot: int, code: int, value: int) -> None:
        self.single[slot][code] = value

    def set_pair(self, layout: SlotLayout, slot: int, code0: int, code1: int,
                 value: int) -> None:
        self.pair[slot][code0 | (code1 << layout.bits[slot])] = value


def structural_tables(layout: SlotLayout, qubit1: bool = True,
                      terminal_z: bool = True) -> ClusterTables:
    """Decoder-free tables from the code structure (no correction applied).

    Accepted cluster types (each with the identity correction):

    * a time-like pair: one flipped ancilla measurement in slot ``t`` and the
      detectors of slot ``t + 1`` that compare with it;
    * a lone terminal Z-check event (``terminal_z``): nothing later compares
      with the terminal Z checks, and an error that is X-type at the end of the
      circuit does not flip the X-basis readout;
    * a qubit-1 syndrome (``qubit1``): both Z checks, both X checks, or all
      four in one round (``X1``, ``Z1``, ``Y1``), or both final X detectors.

    Everything else rejects. Without the final slot, lone terminal X-check
    events are ambiguous between a measurement error and a data error that
    flips ``x0 + x3``, so they reject.
    """
    tables = ClusterTables.empty(
        layout, f"structural(q1={qubit1},tz={terminal_z},"
        f"final={layout.has_final})")
    for t in range(layout.num_slots):
        is_final_slot = layout.has_final and t == layout.num_slots - 1
        if is_final_slot:
            continue
        for s in layout.stabilizers[t]:
            first, second = layout.measurement_error_pattern(t, s)
            if second is not None:
                tables.set_pair(layout, t, first, second, 0)
    terminal = layout.num_slots - 1 - (1 if layout.has_final else 0)
    if terminal_z:
        for s in C.Z_STABILIZERS:
            tables.set_single(terminal, layout.code_of(terminal, [s]), 0)
    if qubit1:
        for t in range(layout.num_slots):
            is_final_slot = layout.has_final and t == layout.num_slots - 1
            if is_final_slot:
                tables.set_single(t, layout.code_of(t, C.X_STABILIZERS), 0)
                continue
            for group in (C.Z_STABILIZERS, C.X_STABILIZERS,
                          C.Z_STABILIZERS + C.X_STABILIZERS):
                tables.set_single(t, layout.code_of(t, group), 0)
    return tables


def dem_signature_weights(layout: SlotLayout, dem
                          ) -> dict[tuple, list[float]]:
    """First-order weights ``[no flip, flip]`` of every single-fault cluster.

    Keys are ``("single", t, code)`` or ``("pair", t, code)``; mechanisms
    without detectors are keyed ``("none",)``. The DEM must have exactly the
    layout's detectors (use the ``data2`` circuit's DEM for a layout without
    the final slot: it is the exact marginal over the final detectors).
    """
    position = {}
    for t, cols in enumerate(layout.columns):
        for bit, col in enumerate(cols):
            position[int(col)] = (t, bit)
    if dem.num_detectors != len(position):
        raise ValueError("DEM detectors do not match the slot layout.")
    weights: dict[tuple, list[float]] = {}
    for mech in M.dem_mechanisms(dem):
        if mech.probability <= 0:
            continue
        if not mech.detectors:
            key = ("none",)
        else:
            per_slot: dict[int, int] = {}
            for d in mech.detectors:
                t, bit = position[int(d)]
                per_slot[t] = per_slot.get(t, 0) ^ (1 << bit)
            per_slot = {t: c for t, c in per_slot.items() if c}
            if not per_slot:
                key = ("none",)
            elif len(per_slot) == 1:
                (t, c), = per_slot.items()
                key = ("single", t, c)
            else:
                ts = sorted(per_slot)
                if len(ts) != 2 or ts[1] != ts[0] + 1:
                    raise ValueError("Mechanism spans non-adjacent slots.")
                t = ts[0]
                key = ("pair", t,
                       per_slot[t] | (per_slot[t + 1] << layout.bits[t]))
        entry = weights.setdefault(key, [0.0, 0.0])
        entry[int(mech.flips_observable)] += mech.probability
    return weights


def dem_lookup_tables(layout: SlotLayout, dem, rho: float = 0.0,
                      weights: Optional[dict] = None) -> ClusterTables:
    """Single-fault lookup tables with ambiguity rejection.

    A cluster is accepted when it equals the signature of at least one DEM
    mechanism and ``min(w0, w1) <= rho * max(w0, w1)``; the correction is the
    logical flip of the heavier class. ``weights`` may pass a precomputed
    ``dem_signature_weights(layout, dem)``.
    """
    if weights is None:
        weights = dem_signature_weights(layout, dem)
    tables = ClusterTables.empty(layout, f"dem_lookup(rho={rho},"
                                 f"final={layout.has_final})")
    for key, (w0, w1) in weights.items():
        if key[0] == "none":
            continue
        if min(w0, w1) > rho * max(w0, w1):
            continue
        value = int(w1 > w0)
        if key[0] == "single":
            tables.single[key[1]][key[2]] = value
        else:
            tables.pair[key[1]][key[2]] = value
    return tables


def evaluate_cluster_rule(codes: np.ndarray, layout: SlotLayout,
                          tables: ClusterTables
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Apply a cluster rule to slot codes.

    Args:
        codes: ``(shots, slots)`` from ``SlotLayout.codes``.

    Returns:
        ``(accept, flip)`` boolean arrays; ``flip`` is the rule's correction of
        the raw logical outcome (only meaningful where accepted).
    """
    shots, num_slots = codes.shape
    occ = codes != 0
    accept = np.ones(shots, dtype=bool)
    flip = np.zeros(shots, dtype=bool)
    if num_slots >= 3:
        triple = occ[:, :-2] & occ[:, 1:-1] & occ[:, 2:]
        accept &= ~triple.any(axis=1)
    false = np.zeros(shots, dtype=bool)
    for t in range(num_slots):
        left = occ[:, t - 1] if t > 0 else false
        right = occ[:, t + 1] if t + 1 < num_slots else false
        right2 = occ[:, t + 2] if t + 2 < num_slots else false
        single = occ[:, t] & ~left & ~right
        if single.any():
            value = tables.single[t][codes[single, t]]
            accept[single] &= value >= 0
            flip[single] ^= value == 1
        if t + 1 < num_slots:
            pair = occ[:, t] & right & ~left & ~right2
            if pair.any():
                code = codes[pair, t] | (codes[pair, t + 1]
                                         << layout.bits[t])
                value = tables.pair[t][code]
                accept[pair] &= value >= 0
                flip[pair] ^= value == 1
    return accept, flip


def reference_cluster_rule(codes_row: Sequence[int], layout: SlotLayout,
                           tables: ClusterTables) -> tuple[bool, bool]:
    """Slow per-shot reference implementation (used by the tests)."""
    runs, current = [], []
    for t, code in enumerate(codes_row):
        if code:
            current.append(t)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    flip = False
    for run in runs:
        if len(run) == 1:
            value = int(tables.single[run[0]][codes_row[run[0]]])
        elif len(run) == 2:
            t = run[0]
            value = int(tables.pair[t][codes_row[t] | (codes_row[t + 1]
                                                       << layout.bits[t])])
        else:
            return False, False
        if value < 0:
            return False, False
        flip ^= value == 1
    return True, flip


def single_fault_audit(layout: SlotLayout, tables: ClusterTables, dem,
                       weights: Optional[dict] = None) -> dict[str, float]:
    """Apply a rule to every single DEM mechanism on its own.

    Returns the total probability of single faults that are rejected,
    accepted and corrected properly, or accepted with a logical error (the
    first-order error budget of the rule), plus the count of the latter.
    """
    if weights is None:
        weights = dem_signature_weights(layout, dem)
    out = {"rejected": 0.0, "accepted_ok": 0.0, "accepted_wrong": 0.0,
           "wrong_signatures": 0, "undetectable_flip": 0.0}
    for key, (w0, w1) in weights.items():
        if key[0] == "none":
            out["accepted_ok"] += w0
            out["accepted_wrong"] += w1
            out["undetectable_flip"] += w1
            continue
        if key[0] == "single":
            value = int(tables.single[key[1]][key[2]])
        else:
            value = int(tables.pair[key[1]][key[2]])
        if value < 0:
            out["rejected"] += w0 + w1
            continue
        good, bad = (w0, w1) if value == 0 else (w1, w0)
        out["accepted_ok"] += good
        out["accepted_wrong"] += bad
        if bad > 0:
            out["wrong_signatures"] += 1
    return out
