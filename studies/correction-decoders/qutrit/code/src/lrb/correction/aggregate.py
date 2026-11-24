"""Aggregate per-circuit study outputs into Clifford-averaged curves and fits.

The study drivers write one ``.npz`` per (sequence, p) [baseline study] or
(variant, sequence, p) [variant study] with keys ``d<depth_index>__<name>``.
This module turns them into tidy records:

``curve record``: strategy, p, depth, acceptance (mean, std over sequences),
and the three fidelity metrics (mean, std) of the accepted shots.

Strategies are named ``family:detail``, e.g. ``lrbd:unif1``, ``bfd:all``,
``mld:all``, ``soft:0.99`` (accept iff the MLD posterior confidence is at
least 0.99), ``win:2``, ``raw``.

Decay fits use the signed metric ``Re E[omega^residual]`` with the offset
fixed at 0 (as the pipeline does for its legacy metric) and are bootstrapped
over sequences. Non-Markovianity diagnostics are the depth-windowed decay
factors and the curvature of ``log y`` in ``m``.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from . import metrics
from .study import CONF_EDGES

SOFT_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9995, 0.9999)
METRICS = ("legacy", "signed", "survival")


def _p_from_name(name: str) -> float:
    match = re.search(r"_p([0-9pme]+)\.npz$", name)
    return float(match.group(1).replace("p", ".").replace("m", "-"))


def load_npz_dir(directory: str, prefix: str = "") -> dict:
    """Load every ``.npz`` in a directory keyed by (prefix, cliff, p)."""
    data = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{prefix}*.npz"))):
        if ".tmp" in path:
            continue
        name = os.path.basename(path)
        match = re.match(r"^(?:(?P<variant>[a-z_]+)_)?c(?P<cliff>\d+)_p", name)
        variant = match.group("variant") or ""
        with np.load(path) as handle:
            data[(variant, int(match.group("cliff")), _p_from_name(name))] = {k: handle[k] for k in handle.files}
    return data


def _soft_point(hist_rb: np.ndarray, shots: float, threshold: float) -> tuple[float, np.ndarray]:
    """Acceptance and expected residual distribution for confidence >= threshold."""
    # Bin b collects log10(1-c) in (edges[b-1], edges[b]]; accept bins whose
    # upper edge is <= log10(1 - threshold) (conservative: whole bins only).
    limit = np.log10(1.0 - threshold)
    upper = np.concatenate([CONF_EDGES, [np.inf]])
    keep = upper <= limit + 1e-12
    accepted = hist_rb[keep].sum(axis=0)
    return accepted.sum() / shots, accepted


def strategy_points(entry: dict, depth_index: int) -> dict[str, tuple[float, np.ndarray]]:
    """
    Acceptance and residual distribution of every strategy at one depth.

    Args:
        entry (dict): Loaded arrays of one (sequence, p) file.
        depth_index (int): Depth index.

    Returns:
        dict: strategy name -> (acceptance, residual probabilities among
        accepted shots, unnormalized).
    """
    pre = f"d{depth_index}__"
    out: dict[str, tuple[float, np.ndarray]] = {}

    def exact(name: str, joint: np.ndarray):
        out[name] = (float(joint.sum()), joint)

    for key, name in (("lrbd_all", "lrbd:all"), ("lrbd_xtriv", "lrbd:xtriv"),
                      ("bfd_all", "bfd:all"), ("bfd_xtriv", "bfd:xtriv"),
                      ("lrbd_all_noterm", "lrbd:all_noterm"), ("bfd_all_noterm", "bfd:all_noterm")):
        if pre + key in entry:
            exact(name, entry[pre + key])
    for family in ("lrbd", "bfd"):
        for policy in ("unif", "const"):
            key = f"{pre}{family}_{policy}"
            if key in entry:
                for index, joint in enumerate(entry[key], start=1):
                    exact(f"{family}:{policy}{index}", joint)
    if pre + "const0_joint" in entry:
        joint = entry[pre + "const0_joint"]  # [l, s]
        exact("lrbd:const0", joint[:, 0])
        l_hat = joint.argmax(axis=0)
        corrected = np.array([sum(joint[(l + l_hat[s]) % 3, s] for s in range(joint.shape[1])) for l in range(3)])
        exact("mld:const0", corrected)
    shots_key = pre + "shots"
    if shots_key in entry:
        shots = float(entry[shots_key])
        for key in entry:
            if key.startswith(pre) and key.endswith("_rb") and key != pre + "soft_hist_rb":
                base = key[len(pre):-3]
                if base == "bfd_sampled":
                    out["bfd:all_sampled"] = (float(entry[pre + "bfd_sampled_accepted"]) / shots, entry[key])
                    continue
                family, _, detail = base.partition("_")
                if family.startswith("win"):
                    out[f"win:{family[3:]}"] = (1.0, entry[key])
                else:
                    out[f"{family}:{detail}"] = (1.0, entry[key])
        out["raw"] = (1.0, entry[pre + "raw_counts"].astype(float))
        hist = entry[pre + "soft_hist_rb"]
        for threshold in SOFT_THRESHOLDS:
            out[f"soft:{threshold}"] = _soft_point(hist, shots, threshold)
    return out


@dataclass
class Curve:
    strategy: str
    p: float
    depths: np.ndarray
    acceptance: np.ndarray  # (n_cliff, n_depth)
    fidelity: dict  # metric -> (n_cliff, n_depth)


def build_curves(data: dict, depths: list[int], variant: str = "") -> list[Curve]:
    """Collect per-sequence curves for every (strategy, p)."""
    by_p: dict[float, list[int]] = {}
    for (var, cliff, p) in data:
        if var == variant:
            by_p.setdefault(p, []).append(cliff)
    curves = []
    for p, cliffs in sorted(by_p.items()):
        cliffs = sorted(cliffs)
        per_strategy: dict[str, list] = {}
        for ci, cliff in enumerate(cliffs):
            entry = data[(variant, cliff, p)]
            for di in range(len(depths)):
                for name, (acc, residual) in strategy_points(entry, di).items():
                    per_strategy.setdefault(name, {})[(ci, di)] = (acc, residual)
        for name, values in per_strategy.items():
            acc = np.full((len(cliffs), len(depths)), np.nan)
            fid = {m: np.full((len(cliffs), len(depths)), np.nan) for m in METRICS}
            for (ci, di), (a, residual) in values.items():
                acc[ci, di] = a
                for m, v in metrics.fidelity_metrics(residual).items():
                    fid[m][ci, di] = v
            curves.append(Curve(name, p, np.asarray(depths), acc, fid))
    return curves


def fit_curve(depths, values: np.ndarray, metric: str = "signed", boot: int = 200, seed: int = 0) -> dict:
    """
    Fit ``y = b f^m`` to the sequence-averaged curve and bootstrap over sequences.

    For ``survival`` the asymptote 1/3 is subtracted first. Weights follow
    the pipeline (across-sequence standard deviation).

    Returns:
        dict: ``f``, ``b``, ``r`` and bootstrap ``f_lo``, ``f_hi``, ``r_lo``, ``r_hi``.
    """
    offset = 1.0 / 3.0 if metric == "survival" else 0.0
    ok_cols = np.all(np.isfinite(values), axis=0)
    x = np.asarray(depths)[ok_cols]
    vals = values[:, ok_cols] - offset
    if x.size < 2:
        return {}
    scale = 1.5 if metric == "survival" else 1.0

    def one(sample):
        mean = sample.mean(axis=0) * scale
        std = sample.std(axis=0) * scale
        return metrics.fit_decay(x, mean, std)

    best = one(vals)
    if best is None:
        return {}
    rng = np.random.default_rng(seed)
    fs = []
    for _ in range(boot):
        idx = rng.integers(0, vals.shape[0], size=vals.shape[0])
        res = one(vals[idx])
        if res is not None:
            fs.append(res["f"])
    fs = np.array(fs)
    out = {"f": best["f"], "b": best["b"], "r": best["r"], "chi2": best["chi2"]}
    if fs.size:
        lo, hi = np.percentile(fs, [2.5, 97.5])
        out.update({"f_lo": lo, "f_hi": hi, "r_lo": 2 * (1 - hi) / 3, "r_hi": 2 * (1 - lo) / 3})
    return out


def per_step_acceptance(depths, acceptance: np.ndarray) -> dict:
    """Fit ``acc(m) = c a^m`` (log-linear least squares on the mean curve)."""
    mean = np.nanmean(acceptance, axis=0)
    x = np.asarray(depths, dtype=float)
    ok = mean > 0
    if ok.sum() < 2:
        return {"a": np.nan, "c": np.nan}
    slope, intercept = np.polyfit(x[ok], np.log(mean[ok]), 1)
    return {"a": float(np.exp(slope)), "c": float(np.exp(intercept))}


def windowed(depths, values: np.ndarray, boot: int = 200, seed: int = 1) -> list[dict]:
    """Depth-windowed decay factors of the mean curve, bootstrapped over sequences."""
    rng = np.random.default_rng(seed)
    mean = np.nanmean(values, axis=0)
    rows = []
    for i in range(len(depths) - 1):
        m0, m1 = depths[i], depths[i + 1]
        def rate(v):
            return (v[i + 1] / v[i]) ** (1.0 / (m1 - m0)) if v[i] > 0 and v[i + 1] > 0 else np.nan
        samples = []
        for _ in range(boot):
            idx = rng.integers(0, values.shape[0], size=values.shape[0])
            samples.append(rate(np.nanmean(values[idx], axis=0)))
        samples = np.array(samples)
        rows.append({"m0": int(m0), "m1": int(m1), "rate": rate(mean),
                     "lo": float(np.nanpercentile(samples, 2.5)), "hi": float(np.nanpercentile(samples, 97.5))})
    return rows


def curvature(depths, values: np.ndarray, min_depth: int = 0, boot: int = 200, seed: int = 2) -> dict:
    """
    Quadratic coefficient of ``log y`` vs ``m`` with a bootstrap CI.

    A Markovian (single-exponential) decay has zero curvature; a negative
    value means the decay accelerates with depth (errors accumulate
    super-linearly, as for pairs of undetected faults).
    """
    x = np.asarray(depths, dtype=float)
    keep = x >= min_depth

    def coeff(v):
        mean = np.nanmean(v, axis=0)[keep]
        if np.any(mean <= 0) or keep.sum() < 3:
            return np.nan
        return float(np.polyfit(x[keep], np.log(mean), 2)[0])

    rng = np.random.default_rng(seed)
    samples = np.array([coeff(values[rng.integers(0, values.shape[0], size=values.shape[0])]) for _ in range(boot)])
    return {"c2": coeff(values), "c2_lo": float(np.nanpercentile(samples, 2.5)),
            "c2_hi": float(np.nanpercentile(samples, 97.5))}
