"""WP-E tasks 1-2: gap (|LLR|) postselection versus LRB-D on the main grid.

Reads ``results/grid_main/p<idx>.npz`` (``PointResult``) and ``rb<idx>.npz``
and writes CSVs to ``results/E`` and figures to ``figures/E``:

* ``calibration.csv``: soft-output calibration per |LLR| bin (observed wrong
  fraction against the band ``[q(upper edge), q(lower edge)]``).
* ``pareto_curves.csv``: pooled acceptance and error rate of gap thresholds.
* ``np_test_grid.csv`` / ``np_test_summary.csv``: every LRB-D point against
  the gap curve at the same pooled acceptance (Neyman-Pearson test).
* ``fits_threshold.csv``, ``fits_lrbd.csv``, ``fits_rb.csv``: repo-style
  decay fits with sequence-bootstrap errors.
* ``matched_points.csv``, ``matched_fits.csv``: operating points that match
  LRB-D acceptance (error reduction) or LRB-D error rate (acceptance gain).
* ``eff_fidelity.csv``: acceptance x fidelity check (rejection charged as a
  random guess).

Usage (light; a few minutes on one core)::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python studies/correction-decoders/qubit/scripts/pareto.py --n-boot 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import acceptance_rules as A  # noqa: E402
from lrb_qubit.experiment import (GAP_EDGES, PointResult,  # noqa: E402
                                  run_physical_rb)

DOCS = ROOT / "studies" / "correction-decoders" / "qubit"
THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
CURVE_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                    6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0)
MATCH_POLICIES = ("lrbd_unif_1", "lrbd_unif_2", "lrbd_unif_3", "lrbd_unif_5",
                  "lrbd_unif_11", "lrbd_const_1", "lrbd_const_2",
                  "lrbd_const_3", "lrbd_const_5", "lrbd_const_11",
                  "lrbd_unif_1_final", "lrbd_const_1_final")
DETAIL_POLICIES = MATCH_POLICIES + ("lrbd_unif_22", "lrbd_const_22", "const0")

# Colors (dataviz reference palette): information set, not method.
BLUE = "#2a78d6"     # uses the final data detectors (data-5 information)
ORANGE = "#eb6834"   # check rounds only (data-2 information)
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b")


def threshold_index(t: float) -> int:
    k = int(np.flatnonzero(np.isclose(GAP_EDGES, t))[0])
    return k


def decoder_for(policy: str) -> tuple[str, ...]:
    if policy == "const0":
        return ("const0_mld",)
    if policy.endswith("_final"):
        return ("mld5",)
    return ("mld2", "mld5")


# ---------------------------------------------------------------------------
# Vectorized helpers on bootstrap replicates
# ---------------------------------------------------------------------------

def per_sequence_at(acc_s: np.ndarray, err_s: np.ndarray, k: np.ndarray,
                    lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sequence counts of rows' randomized rules.

    Args:
        acc_s, err_s: ``(num_seq, K+1)`` per-sequence cumulative counts.
        k, lam: ``(rows,)``.

    Returns:
        ``(rows, num_seq)`` accepted and errors.
    """
    km1 = np.maximum(k - 1, 0)
    a = acc_s[:, k].T + lam[:, None] * (acc_s[:, km1].T - acc_s[:, k].T)
    e = err_s[:, k].T + lam[:, None] * (err_s[:, km1].T - err_s[:, k].T)
    return a, e


def boot_quantiles(values: np.ndarray) -> tuple[float, float, float]:
    good = values[np.isfinite(values)]
    if len(good) < 2:
        return np.nan, np.nan, np.nan
    return (float(np.std(good)), float(np.quantile(good, 0.025)),
            float(np.quantile(good, 0.975)))


# ---------------------------------------------------------------------------
# Analysis per probability
# ---------------------------------------------------------------------------

class PointAnalysis:
    def __init__(self, idx: int, point: PointResult, rb: dict,
                 mult: np.ndarray, rb1: "dict | None" = None):
        self.idx = idx
        self.point = point
        self.p = point.p
        self.depths = point.depths
        self.rb = rb
        self.rb1 = rb1
        self.mult = mult
        self.q_edges = A.gap_error_probability(GAP_EDGES)
        self.cum = {}
        for name, hist in point.gap_hist.items():
            acc, err = A.cumulative_counts(hist)   # (seq, depth, K+1)
            self.cum[name] = (acc.astype(float), err.astype(float))

    # -- calibration -------------------------------------------------------
    def calibration(self) -> tuple[list[dict], list[dict]]:
        rows, summary = [], []
        for name, hist in sorted(self.point.gap_hist.items()):
            for d_index in [None] + list(range(len(self.depths))):
                h = (hist.sum(axis=(0, 1)) if d_index is None
                     else hist[:, d_index].sum(axis=0))
                n = h.sum(axis=1).astype(float)
                w = h[:, 1].astype(float)
                q_hi = self.q_edges[:-1]
                q_lo = self.q_edges[1:]
                z = np.zeros_like(n)
                with np.errstate(invalid="ignore", divide="ignore"):
                    over = (w > n * q_hi) & (n > 0)
                    under = (w < n * q_lo) & (n > 0)
                    z = np.where(over, (w - n * q_hi)
                                 / np.sqrt(n * q_hi * (1 - q_hi)), z)
                    z = np.where(under, (w - n * q_lo)
                                 / np.sqrt(np.maximum(n * q_lo * (1 - q_lo),
                                                      1e-300)), z)
                used = n > 0
                summary.append({
                    "p_idx": self.idx, "p": self.p, "decoder": name,
                    "depth": "all" if d_index is None
                    else self.depths[d_index],
                    "bins": int(used.sum()), "shots": int(n.sum()),
                    "max_abs_z": float(np.max(np.abs(z[used])))
                    if used.any() else 0.0,
                    "bins_abs_z_gt3": int(np.sum(np.abs(z[used]) > 3)),
                    "bins_abs_z_gt4": int(np.sum(np.abs(z[used]) > 4)),
                    "observed_errors": int(w.sum()),
                    "expected_errors_low": float(np.sum(n * q_lo)),
                    "expected_errors_high": float(np.sum(n * q_hi))})
                if d_index is None:
                    for j in np.flatnonzero(used):
                        if GAP_EDGES[j] >= 20:
                            continue
                        rows.append({
                            "p_idx": self.idx, "p": self.p, "decoder": name,
                            "gap_low": GAP_EDGES[j],
                            "gap_high": GAP_EDGES[j + 1], "shots": int(n[j]),
                            "wrong": int(w[j]), "observed": w[j] / n[j],
                            "q_low": q_lo[j], "q_high": q_hi[j],
                            "z_outside_band": z[j]})
        return rows, summary

    # -- curves ------------------------------------------------------------
    def curve_rows(self) -> list[dict]:
        rows = []
        for name in ("mld2", "mld5", "const0_mld"):
            if name not in self.cum:
                continue
            acc, err = self.cum[name]
            for d_index, depth in enumerate(self.depths):
                pa = acc[:, d_index].sum(axis=0)
                pe = err[:, d_index].sum(axis=0)
                shots = pa[0]
                for t in CURVE_THRESHOLDS:
                    k = threshold_index(t)
                    a = pa[k]
                    rate = pe[k] / a if a > 0 else np.nan
                    se = (np.sqrt(rate * (1 - rate) / a) if a > 0 else np.nan)
                    rows.append({"p_idx": self.idx, "p": self.p,
                                 "decoder": name, "depth": depth,
                                 "threshold": t, "acceptance": a / shots,
                                 "error_rate": rate, "error_rate_se": se,
                                 "accepted": int(a), "errors": int(pe[k]),
                                 "shots": int(shots)})
        return rows

    # -- Neyman-Pearson test against LRB-D ---------------------------------
    def np_test(self) -> list[dict]:
        rows = []
        mult = self.mult
        for policy in sorted(self.point.tables):
            if not (policy.startswith("lrbd_") or policy == "const0"):
                continue
            table = self.point.tables[policy]
            for decoder in decoder_for(policy):
                if decoder not in self.cum:
                    continue
                acc_c, err_c = self.cum[decoder]
                for d_index, depth in enumerate(self.depths):
                    sh = table.shots[:, d_index].astype(float)
                    ac = table.accepted[:, d_index].astype(float)
                    er = table.errors[:, d_index].astype(float)
                    if ac.sum() == 0:
                        continue
                    alpha = ac.sum() / sh.sum()
                    eps = er.sum() / ac.sum()
                    ca = acc_c[:, d_index]
                    ce = err_c[:, d_index]
                    pa = ca.sum(axis=0)[None, :]
                    pe = ce.sum(axis=0)[None, :]
                    _, _, rate, rate_low = A.batch_error_at_acceptance(
                        pa, pe, np.array([alpha * pa[0, 0]]))
                    # paired sequence bootstrap
                    b_sh = mult @ sh
                    b_ac = mult @ ac
                    b_er = mult @ er
                    b_pa = mult @ ca
                    b_pe = mult @ ce
                    with np.errstate(invalid="ignore", divide="ignore"):
                        b_alpha = b_ac / b_sh
                        b_eps = b_er / b_ac
                    _, _, b_rate, b_rate_low = A.batch_error_at_acceptance(
                        b_pa, b_pe, b_alpha * b_pa[:, 0])
                    diff = eps - rate[0]
                    diff_low = eps - rate_low[0]
                    se_boot = np.nanstd(b_eps - b_rate)
                    n_gap = alpha * pa[0, 0]
                    se_shot = np.sqrt(eps * (1 - eps) / ac.sum()
                                      + rate[0] * (1 - rate[0])
                                      / max(n_gap, 1.0))
                    # Floor at one error in the accepted sample, so that a
                    # handful of accepted shots cannot signal a violation.
                    se = max(se_boot, se_shot, 1.0 / ac.sum())
                    rows.append({
                        "p_idx": self.idx, "p": self.p, "policy": policy,
                        "decoder": decoder, "depth": depth,
                        "acceptance": alpha, "lrbd_error_rate": eps,
                        "gap_error_rate": rate[0],
                        "gap_error_rate_optimistic": rate_low[0],
                        "diff": diff, "diff_optimistic": diff_low,
                        "se_boot": se_boot, "se_shot": se_shot,
                        "z": diff / se if se > 0 else np.nan,
                        "z_optimistic": diff_low / se if se > 0 else np.nan,
                        "lrbd_beats_curve_3sigma": bool(diff < -3 * se),
                        "lrbd_beats_optimistic_3sigma":
                            bool(diff_low < -3 * se)})
        return rows

    # -- fits --------------------------------------------------------------
    def _fit_row(self, method: str, decoder: str, threshold, acc: np.ndarray,
                 err: np.ndarray, shots: np.ndarray) -> dict:
        fit = A.bootstrap_fixed_offset_fit(self.depths, acc, err, self.mult)
        mean, std = A.repo_fidelity_stats(acc, err)
        row = {"p_idx": self.idx, "p": self.p, "method": method,
               "decoder": decoder, "threshold": threshold, "f": fit["f"],
               "f_boot_std": fit["f_boot_std"], "f_ci_low": fit["f_ci_low"],
               "f_ci_high": fit["f_ci_high"], "b": fit["b"],
               "r_per_clifford": float(A.decay_to_error_per_clifford(fit["f"]))
               if np.isfinite(fit["f"]) else np.nan}
        accp = acc.sum(axis=0) / shots.sum(axis=0)
        for d_index, depth in enumerate(self.depths):
            row[f"F_m{depth}"] = mean[d_index]
            row[f"Fstd_m{depth}"] = std[d_index]
            row[f"acc_m{depth}"] = accp[d_index]
        return row

    def fits(self) -> tuple[list[dict], list[dict], list[dict]]:
        thr_rows, lrbd_rows, rb_rows = [], [], []
        for decoder in ("mld2", "mld5"):
            acc_c, err_c = self.cum[decoder]
            shots = acc_c[:, :, 0]
            for t in THRESHOLDS:
                k = threshold_index(t)
                thr_rows.append(self._fit_row(
                    f"gap_t{t:g}", decoder, t, acc_c[:, :, k],
                    err_c[:, :, k], shots))
        for name in sorted(self.point.tables):
            if not (name.startswith("lrbd_") or name.startswith("const0")):
                continue
            t = self.point.tables[name]
            lrbd_rows.append(self._fit_row(name, "", "", t.accepted.astype(
                float), t.errors.astype(float), t.shots.astype(float)))
        # physical RB, repo free-offset fit, at p_RB = 5p (grid) and p_RB = p
        for scale, rb in ((5.0, self.rb), (1.0, self.rb1)):
            if rb is None:
                continue
            acc = rb["accepted"].astype(float)
            err = rb["errors"].astype(float)
            mean, std = A.repo_fidelity_stats(acc, err)
            fit = A.fit_free_offset(self.depths, mean, std)
            fid = A.fidelity(acc, err)
            bmean, bstd = A.weighted_fidelity_stats(fid, self.mult)
            boots = np.array([
                (lambda r: np.nan if r is None else r["f"])(
                    A.fit_free_offset(self.depths, bmean[b], bstd[b]))
                for b in range(len(bmean))])
            sd, lo, hi = boot_quantiles(boots)
            fixed = A.bootstrap_fixed_offset_fit(self.depths, acc, err,
                                                 self.mult)
            row = {"p_idx": self.idx, "p": self.p, "rb_scale": scale,
                   "p_rb": float(rb["p_rb"]),
                   "f_free": np.nan if fit is None else fit["f"],
                   "a_free": np.nan if fit is None else fit["a"],
                   "f_free_boot_std": sd, "f_free_ci_low": lo,
                   "f_free_ci_high": hi,
                   "r_free": float(A.decay_to_error_per_clifford(fit["f"]))
                   if fit else np.nan,
                   "f_fixed0": fixed["f"],
                   "f_fixed0_boot_std": fixed["f_boot_std"],
                   "r_fixed0": float(A.decay_to_error_per_clifford(
                       fixed["f"])) if np.isfinite(fixed["f"]) else np.nan}
            for d_index, depth in enumerate(self.depths):
                row[f"F_m{depth}"] = mean[d_index]
            rb_rows.append(row)
        return thr_rows, lrbd_rows, rb_rows

    # -- matched operating points -------------------------------------------
    def matched(self) -> tuple[list[dict], list[dict]]:
        point_rows, fit_rows = [], []
        mult = self.mult
        n_boot = mult.shape[0]
        for policy in MATCH_POLICIES:
            if policy not in self.point.tables:
                continue
            table = self.point.tables[policy]
            for decoder in decoder_for(policy):
                acc_c, err_c = self.cum[decoder]
                n_d = len(self.depths)
                # per-depth matched rules: full data (row 0) + replicates
                k_a = np.zeros((n_boot + 1, n_d), int)
                l_a = np.zeros((n_boot + 1, n_d))
                k_e = np.zeros((n_boot + 1, n_d), int)
                l_e = np.zeros((n_boot + 1, n_d))
                alpha_d = np.full((n_boot + 1, n_d), np.nan)
                eps_d = np.full((n_boot + 1, n_d), np.nan)
                eps_g = np.full((n_boot + 1, n_d), np.nan)
                alpha_g = np.full((n_boot + 1, n_d), np.nan)
                ones = np.ones((1, mult.shape[1]), dtype=mult.dtype)
                weights = np.vstack([ones, mult])
                for d_index in range(n_d):
                    sh = weights @ table.shots[:, d_index].astype(float)
                    ac = weights @ table.accepted[:, d_index].astype(float)
                    er = weights @ table.errors[:, d_index].astype(float)
                    pa = weights @ acc_c[:, d_index]
                    pe = weights @ err_c[:, d_index]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        alpha = ac / sh
                        eps = er / ac
                    alpha_d[:, d_index] = alpha
                    eps_d[:, d_index] = eps
                    k, lam, rate, _ = A.batch_error_at_acceptance(
                        pa, pe, alpha * pa[:, 0])
                    k_a[:, d_index], l_a[:, d_index] = k, lam
                    eps_g[:, d_index] = rate
                    target = np.where(np.isfinite(eps), eps, 0.0)
                    k2, lam2, accepted = A.batch_acceptance_at_error(
                        pa, pe, target)
                    k_e[:, d_index], l_e[:, d_index] = k2, lam2
                    alpha_g[:, d_index] = accepted / pa[:, 0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    reduction = eps_d / eps_g
                    gain = alpha_g / alpha_d
                for d_index, depth in enumerate(self.depths):
                    r_sd, r_lo, r_hi = boot_quantiles(reduction[1:, d_index])
                    g_sd, g_lo, g_hi = boot_quantiles(gain[1:, d_index])
                    point_rows.append({
                        "p_idx": self.idx, "p": self.p, "policy": policy,
                        "decoder": decoder, "depth": depth,
                        "lrbd_acceptance": alpha_d[0, d_index],
                        "lrbd_error_rate": eps_d[0, d_index],
                        "gap_error_rate_at_lrbd_acceptance":
                            eps_g[0, d_index],
                        "error_reduction": reduction[0, d_index],
                        "error_reduction_ci_low": r_lo,
                        "error_reduction_ci_high": r_hi,
                        "gap_acceptance_at_lrbd_error": alpha_g[0, d_index],
                        "acceptance_gain": gain[0, d_index],
                        "acceptance_gain_ci_low": g_lo,
                        "acceptance_gain_ci_high": g_hi})
                # fits along the matched paths (row 0 = full data)
                f_rows = {}
                n_seq = acc_c.shape[0]
                n_w = weights.sum(axis=1)[:, None].astype(float)
                for kind, kk, ll in (("matched_acceptance", k_a, l_a),
                                     ("matched_error", k_e, l_e)):
                    a_s = np.zeros((n_boot + 1, n_seq, n_d))
                    e_s = np.zeros((n_boot + 1, n_seq, n_d))
                    for d_index in range(n_d):
                        a, e = per_sequence_at(acc_c[:, d_index],
                                               err_c[:, d_index],
                                               kk[:, d_index], ll[:, d_index])
                        a_s[:, :, d_index] = a
                        e_s[:, :, d_index] = e
                    fid = A.fidelity(a_s, e_s)
                    nan = np.isnan(fid)
                    f0 = np.where(nan, 0.0, fid)
                    wf = weights.astype(float)
                    mean = np.einsum("rs,rsd->rd", wf, f0) / n_w
                    second = np.einsum("rs,rsd->rd", wf, f0 * f0) / n_w
                    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
                    bad = np.einsum("rs,rsd->rd", wf, nan.astype(float)) > 0
                    mean = np.where(bad, np.nan, mean)
                    std = np.where(bad, np.nan, std)
                    f_vals = A.fit_fixed_offset_batch(self.depths, mean, std)
                    exact = A.fit_fixed_offset(self.depths, mean[0], std[0])
                    f_vals[0] = np.nan if exact is None else exact["f"]
                    acc0 = a_s[0].sum(axis=0) / acc_c[:, :, 0].sum(axis=0)
                    f_rows[kind] = (f_vals, mean[0], std[0], acc0)
                lr = A.fidelity(table.accepted, table.errors)
                ml, sl = A.weighted_fidelity_stats(lr, weights)
                f_lrbd = A.fit_fixed_offset_batch(self.depths, ml, sl)
                exact = A.fit_fixed_offset(self.depths, ml[0], sl[0])
                f_lrbd[0] = np.nan if exact is None else exact["f"]
                for kind, (f_vals, mean0, std0, acc0) in f_rows.items():
                    diff = f_vals - f_lrbd
                    d_sd, d_lo, d_hi = boot_quantiles(diff[1:])
                    f_sd, f_lo, f_hi = boot_quantiles(f_vals[1:])
                    l_sd, _, _ = boot_quantiles(f_lrbd[1:])
                    row = {"p_idx": self.idx, "p": self.p, "policy": policy,
                           "decoder": decoder, "kind": kind,
                           "f_gap": f_vals[0], "f_gap_boot_std": f_sd,
                           "f_gap_ci_low": f_lo, "f_gap_ci_high": f_hi,
                           "f_lrbd": f_lrbd[0], "f_lrbd_boot_std": l_sd,
                           "f_diff": diff[0], "f_diff_boot_std": d_sd,
                           "f_diff_ci_low": d_lo, "f_diff_ci_high": d_hi}
                    for d_index, depth in enumerate(self.depths):
                        row[f"Fgap_m{depth}"] = mean0[d_index]
                        row[f"Flrbd_m{depth}"] = ml[0, d_index]
                        row[f"accgap_m{depth}"] = acc0[d_index]
                        row[f"acclrbd_m{depth}"] = (
                            table.accepted[:, d_index].sum()
                            / table.shots[:, d_index].sum())
                    fit_rows.append(row)
        return point_rows, fit_rows

    # -- acceptance x fidelity -------------------------------------------
    def effective_fidelity(self) -> list[dict]:
        """Rejected shots scored as random guesses: F_eff = acceptance x F."""
        rows = []
        mld = self.point.tables.get("mld5")
        f_mld = A.fidelity(mld.accepted, mld.errors)
        candidates = {}
        for name, table in self.point.tables.items():
            if name.startswith("lrbd_"):
                candidates[name] = (table.accepted / table.shots,
                                    A.fidelity(table.accepted, table.errors))
        for decoder in ("mld2", "mld5"):
            acc_c, err_c = self.cum[decoder]
            for t in THRESHOLDS[1:]:
                k = threshold_index(t)
                candidates[f"gap_t{t:g}_{decoder}"] = (
                    acc_c[:, :, k] / acc_c[:, :, 0],
                    A.fidelity(acc_c[:, :, k], err_c[:, :, k]))
        for d_index, depth in enumerate(self.depths):
            base = float(np.mean(f_mld[:, d_index]))
            best_name, best_val = None, -np.inf
            for name, (acc, fid) in candidates.items():
                val = float(np.mean(np.nan_to_num(acc[:, d_index]
                                                  * fid[:, d_index])))
                if val > best_val:
                    best_name, best_val = name, val
            se = float(np.std(f_mld[:, d_index]) / np.sqrt(len(f_mld)))
            rows.append({"p_idx": self.idx, "p": self.p, "depth": depth,
                         "F_mld5": base, "F_mld5_seq_se": se,
                         "best_other_method": best_name,
                         "best_other_acceptance_x_F": best_val,
                         "excess": best_val - base})
        return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=7)


def _tidy_axis(ax, axis: str = "x") -> None:
    """Readable ticks: linear for spans under a decade, decade labels else."""
    from matplotlib import ticker
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    target = ax.xaxis if axis == "x" else ax.yaxis
    if lo > 0 and hi / lo < 10.0:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("linear")
        target.set_major_locator(ticker.MaxNLocator(4))
        target.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        target.set_minor_locator(ticker.NullLocator())
        return
    target.set_major_locator(ticker.LogLocator(base=10, numticks=7))
    target.set_major_formatter(ticker.LogFormatterSciNotation())
    target.set_minor_formatter(ticker.NullFormatter())


def _positive(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan)


def plot_pareto_page(plt, an: PointAnalysis, title_extra: str = ""):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0), sharex=False)
    for d_index, depth in enumerate(an.depths):
        ax = axes.flat[d_index]
        _style(ax)
        for decoder, color, ls in (("mld2", ORANGE, "--"),
                                   ("mld5", BLUE, "-")):
            acc, err = an.cum[decoder]
            pa = acc[:, d_index].sum(axis=0)
            pe = err[:, d_index].sum(axis=0)
            ok = (pa > 0) & (pe > 0)   # zero-error points cannot be drawn
            ax.plot(pa[ok] / pa[0], pe[ok] / pa[ok],
                    color=color, ls=ls, lw=1.6,
                    label=f"gap rule ({'check rounds only' if decoder == 'mld2' else 'all detectors'})")
        for name, table in an.point.tables.items():
            if not name.startswith("lrbd_"):
                continue
            ac = table.accepted[:, d_index].sum()
            if ac == 0:
                continue
            alpha = ac / table.shots[:, d_index].sum()
            eps = table.errors[:, d_index].sum() / ac
            if eps <= 0:
                continue
            final = name.endswith("_final")
            marker = "o" if "_unif_" in name else "s"
            color = BLUE if final else ORANGE
            ax.plot(alpha, eps, marker=marker, ms=4.5 if not final else 6,
                    mfc="white" if not final else color, mec=color,
                    mew=1.0, ls="none", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _tidy_axis(ax, "x")
        _tidy_axis(ax, "y")
        ax.set_title(f"m = {depth}", fontsize=9, color=INK)
        if d_index % 3 == 0:
            ax.set_ylabel("logical error rate of accepted shots",
                          fontsize=7.5, color=INK2)
        if d_index >= 6:
            ax.set_xlabel("acceptance", fontsize=7.5, color=INK2)
    handles = [
        plt.Line2D([], [], color=ORANGE, ls="--", lw=1.6,
                   label="gap rule, check rounds only (mld2)"),
        plt.Line2D([], [], color=BLUE, ls="-", lw=1.6,
                   label="gap rule, with final data detectors (mld5)"),
        plt.Line2D([], [], color=ORANGE, marker="o", mfc="white", ls="none",
                   label="LRB-D unif=k (data-2)"),
        plt.Line2D([], [], color=ORANGE, marker="s", mfc="white", ls="none",
                   label="LRB-D const=k (data-2)"),
        plt.Line2D([], [], color=BLUE, marker="o", ls="none",
                   label="LRB-D unif=1 + final data check"),
        plt.Line2D([], [], color=BLUE, marker="s", ls="none",
                   label="LRB-D const=1 + final data check")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False)
    fig.suptitle(f"Gap-threshold Pareto curves vs LRB-D, p = {an.p:g}"
                 f"{title_extra}", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def plot_calibration(plt, analyses, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    picks = [a for a in analyses if a.p in (0.001, 0.005, 0.02)]
    for ax, decoder in zip(axes, ("mld5", "mld2")):
        _style(ax)
        g = np.linspace(0, 12, 200)
        ax.plot(g, A.gap_error_probability(g), color=INK, lw=1.2,
                label="exact posterior: 1/(1+e^|LLR|)")
        for a, marker in zip(picks, ("o", "s", "^")):
            hist = a.point.gap_hist[decoder].sum(axis=(0, 1))
            n = hist.sum(axis=1)
            ok = n >= 200
            centers = 0.5 * (GAP_EDGES[:-1] + np.minimum(GAP_EDGES[1:], 40))
            ok &= centers < 12
            ax.plot(centers[ok], hist[ok, 1] / n[ok], marker=marker, ms=3.5,
                    ls="none", mfc="white",
                    mec=BLUE if decoder == "mld5" else ORANGE,
                    label=f"p = {a.p:g}")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 0.7)
        ax.set_xlabel("|LLR| (bin centre)", fontsize=8, color=INK2)
        ax.set_ylabel("observed wrong fraction", fontsize=8, color=INK2)
        ax.set_title(f"{decoder}: soft-output calibration", fontsize=9,
                     color=INK)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=150)
    plt.close(fig)


def plot_fits_vs_threshold(plt, thr_rows, lrbd_rows, rb_rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax in axes:
        _style(ax)
    for decoder, color, ls in (("mld5", BLUE, "-"), ("mld2", ORANGE, "--")):
        for t_index, t in enumerate((0.0, 2.0, 4.0, 8.0)):
            sel = [r for r in thr_rows if r["decoder"] == decoder
                   and r["threshold"] == t and r["p"] > 0]
            x = [r["p"] for r in sel]
            r_val = [max(r["r_per_clifford"], 1e-9) for r in sel]
            axes[0].plot(x, r_val, color=color, ls=ls,
                         alpha=0.35 + 0.65 * t_index / 3, lw=1.3,
                         marker="o", ms=3,
                         label=f"gap t={t:g} ({decoder})")
            accs = [r["acc_m22"] for r in sel]
            axes[1].plot(x, accs, color=color, ls=ls,
                         alpha=0.35 + 0.65 * t_index / 3, lw=1.3, marker="o",
                         ms=3)
    for name, marker in (("lrbd_unif_1", "o"), ("lrbd_const_1", "s"),
                         ("lrbd_unif_1_final", "D")):
        sel = [r for r in lrbd_rows if r["method"] == name and r["p"] > 0]
        axes[0].plot([r["p"] for r in sel],
                     [max(r["r_per_clifford"], 1e-9) for r in sel],
                     color=INK2, marker=marker, ms=4, mfc="white", lw=0.8,
                     label=name.replace("lrbd_", "LRB-D "))
        axes[1].plot([r["p"] for r in sel], [r["acc_m22"] for r in sel],
                     color=INK2, marker=marker, ms=4, mfc="white", lw=0.8)
    for scale, ls in ((5.0, "-"), (1.0, "-.")):
        sel = [r for r in rb_rows if r["p"] > 0 and r["rb_scale"] == scale]
        axes[0].plot([r["p"] for r in sel],
                     [max(r["r_free"], 1e-9) for r in sel], color=AQUA,
                     lw=1.6, ls=ls,
                     label=f"physical RB, p_RB = {scale:g}p (free offset)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("p", fontsize=8, color=INK2)
    axes[0].set_ylabel("fitted r = (1 - f)/2 (offset fixed 0)", fontsize=8,
                       color=INK2)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("p", fontsize=8, color=INK2)
    axes[1].set_ylabel("acceptance at m = 22", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fits_vs_p.png", dpi=150)
    plt.close(fig)


def plot_fm_curves(plt, analyses, rb_by_p, out):
    picks = [a for a in analyses if a.p in (0.002, 0.005, 0.01)]
    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 6.5),
                             squeeze=False)
    for col, an in enumerate(picks):
        ax_f, ax_a = axes[0, col], axes[1, col]
        _style(ax_f)
        _style(ax_a)
        acc_c, err_c = an.cum["mld5"]
        for t_index, t in enumerate((0.0, 1.0, 2.0, 4.0, 8.0)):
            k = threshold_index(t)
            mean, _ = A.repo_fidelity_stats(acc_c[:, :, k], err_c[:, :, k])
            accp = acc_c[:, :, k].sum(0) / acc_c[:, :, 0].sum(0)
            color = RAMP[min(2 * t_index + 1, len(RAMP) - 1)]
            ax_f.plot(an.depths, mean, color=color, marker="o", ms=3, lw=1.4,
                      label=f"gap t={t:g} (mld5)")
            ax_a.plot(an.depths, _positive(accp), color=color, marker="o",
                      ms=3, lw=1.4)
        for name, marker in (("lrbd_unif_1", "o"), ("lrbd_const_1", "s"),
                             ("lrbd_unif_1_final", "D")):
            table = an.point.tables[name]
            mean, _ = A.repo_fidelity_stats(table.accepted, table.errors)
            ax_f.plot(an.depths, mean, color=ORANGE if "final" not in name
                      else INK2, marker=marker, mfc="white", ms=4, lw=1.0,
                      label=name.replace("lrbd_", "LRB-D "))
            ax_a.plot(an.depths,
                      _positive(table.accepted.sum(0) / table.shots.sum(0)),
                      color=ORANGE if "final" not in name else INK2,
                      marker=marker, mfc="white", ms=4, lw=1.0)
        rb5, rb1 = rb_by_p.get(an.idx, (None, None))
        for rb, ls, lab in ((rb5, ":", "physical RB at p_RB = 5p"),
                            (rb1, "-.", "physical RB at p_RB = p")):
            if rb is None:
                continue
            mean, _ = A.repo_fidelity_stats(rb["accepted"], rb["errors"])
            ax_f.plot(an.depths, mean, color=AQUA, lw=1.6, ls=ls, label=lab)
        ax_f.set_title(f"p = {an.p:g}", fontsize=9, color=INK)
        ax_f.set_ylim(0, 1.02)
        ax_a.set_yscale("log")
        ax_a.set_xlabel("depth m", fontsize=8, color=INK2)
        if col == 0:
            ax_f.set_ylabel("F = |1 - 2 e/a| (mean over sequences)",
                            fontsize=8, color=INK2)
            ax_a.set_ylabel("acceptance", fontsize=8, color=INK2)
            ax_f.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "F_and_acceptance_vs_m.png", dpi=150)
    plt.close(fig)


def plot_matched(plt, point_rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax in axes.flat:
        _style(ax)
    combos = (("lrbd_unif_1", "mld2", ORANGE, "o"),
              ("lrbd_const_1", "mld2", ORANGE, "s"),
              ("lrbd_unif_1_final", "mld5", BLUE, "o"),
              ("lrbd_const_1_final", "mld5", BLUE, "s"))
    for col, p in enumerate((0.002, 0.01)):
        for policy, decoder, color, marker in combos:
            sel = [r for r in point_rows if r["p"] == p
                   and r["policy"] == policy and r["decoder"] == decoder]
            x = np.array([r["depth"] for r in sel], float)
            red = np.array([r["error_reduction"] for r in sel])
            lo = np.array([r["error_reduction_ci_low"] for r in sel])
            hi = np.array([r["error_reduction_ci_high"] for r in sel])
            gain = np.array([r["acceptance_gain"] for r in sel])
            glo = np.array([r["acceptance_gain_ci_low"] for r in sel])
            ghi = np.array([r["acceptance_gain_ci_high"] for r in sel])
            lab = f"{policy.replace('lrbd_', '')} vs {decoder}"
            axes[0, col].errorbar(x, red, yerr=[np.abs(red - lo),
                                                np.abs(hi - red)],
                                  color=color, marker=marker,
                                  mfc="white" if "final" not in policy
                                  else color, ms=4, lw=1.2, capsize=2,
                                  label=lab)
            axes[1, col].errorbar(x, gain, yerr=[np.abs(gain - glo),
                                                 np.abs(ghi - gain)],
                                  color=color, marker=marker,
                                  mfc="white" if "final" not in policy
                                  else color, ms=4, lw=1.2, capsize=2,
                                  label=lab)
        axes[0, col].set_title(f"p = {p:g}: error reduction at equal "
                               "acceptance", fontsize=9, color=INK)
        axes[1, col].set_title(f"p = {p:g}: acceptance gain at equal "
                               "error rate", fontsize=9, color=INK)
        for row in range(2):
            axes[row, col].set_yscale("log")
            axes[row, col].axhline(1.0, color=MUTED, lw=0.8)
            axes[row, col].set_xlabel("depth m", fontsize=8, color=INK2)
        axes[0, col].set_ylabel("eps_LRB-D / eps_gap", fontsize=8, color=INK2)
        axes[1, col].set_ylabel("acc_gap / acc_LRB-D", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "matched_operating_points.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v)
                             for k, v in row.items()})


RESULT_KEYS = ("cal", "cal_sum", "curves", "np", "thr", "lrbd", "rb", "mpts",
               "mfits", "eff")


RB1_SEED = 4242


def physical_rb_at_p(grid: Path, p: float) -> dict:
    """Matched physical RB at p_RB = p (the newer convention), same sequences
    and depths as the grid, ``rb_shots`` shots, seed ``RB1_SEED``."""
    meta = json.loads((grid / "meta.json").read_text())
    table = run_physical_rb(p, meta["sequences"], meta["depths"],
                            meta["rb_shots"], seed=RB1_SEED)
    return {"p_rb": np.array(p), "shots": table.shots,
            "accepted": table.accepted, "errors": table.errors}


def load_analysis(grid: Path, idx: int, mult: np.ndarray,
                  with_rb1: bool = True) -> "PointAnalysis | None":
    path = grid / f"p{idx}.npz"
    rb_path = grid / f"rb{idx}.npz"
    if not path.exists() or not rb_path.exists():
        return None
    point = PointResult.from_npz(str(path))
    rb1 = physical_rb_at_p(grid, point.p) if with_rb1 else None
    return PointAnalysis(idx, point, dict(np.load(rb_path)), mult, rb1)


def analyse_index(task) -> tuple[int, dict]:
    grid, idx, mult = task
    out = {key: [] for key in RESULT_KEYS}
    an = load_analysis(grid, idx, mult)
    if an is None:
        return idx, out
    out["rb1_table"] = an.rb1
    out["cal"], out["cal_sum"] = an.calibration()
    out["curves"] = an.curve_rows()
    if an.p > 0:
        out["np"] = an.np_test()
        out["thr"], out["lrbd"], out["rb"] = an.fits()
        out["mpts"], out["mfits"] = an.matched()
        out["eff"] = an.effective_fidelity()
    return idx, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", default=str(DOCS / "results" / "grid_main"))
    parser.add_argument("--out", default=str(DOCS / "results" / "E"))
    parser.add_argument("--figures", default=str(DOCS / "figures" / "E"))
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--p-indices", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    grid = Path(args.grid)
    out = Path(args.out)
    figs = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((grid / "meta.json").read_text())
    indices = (list(range(len(meta["probs"]))) if args.p_indices == "all"
               else [int(i) for i in args.p_indices.split(",")])
    rng = np.random.default_rng(args.seed)
    mult = A.bootstrap_multiplicities(meta["num_seq"], args.n_boot, rng)

    results = {key: [] for key in RESULT_KEYS}
    start = time.perf_counter()
    tasks = [(grid, idx, mult) for idx in indices]
    parts: dict[int, dict] = {}
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for idx, part in pool.map(analyse_index, tasks):
                parts[idx] = part
                print(f"[{time.perf_counter() - start:7.1f}s] p{idx} done",
                      flush=True)
    else:
        for task in tasks:
            idx, part = analyse_index(task)
            parts[idx] = part
            print(f"[{time.perf_counter() - start:7.1f}s] p{idx} done",
                  flush=True)
    for idx in sorted(parts):
        for key in RESULT_KEYS:
            results[key] += parts[idx][key]
    analyses = [an for an in (load_analysis(grid, idx, mult[:1],
                                            with_rb1=False)
                              for idx in indices) if an is not None]
    for an in analyses:
        an.rb1 = parts.get(an.idx, {}).get("rb1_table")
    rb_by_p = {an.idx: (an.rb, an.rb1) for an in analyses}

    write_csv(out / "calibration.csv", results["cal"])
    write_csv(out / "calibration_summary.csv", results["cal_sum"])
    write_csv(out / "pareto_curves.csv", results["curves"])
    detail = [r for r in results["np"] if r["policy"] in DETAIL_POLICIES]
    write_csv(out / "np_test_grid.csv", detail)
    summary = []
    for decoder in ("mld2", "mld5", "const0_mld"):
        for idx in sorted({r["p_idx"] for r in results["np"]}):
            sel = [r for r in results["np"] if r["decoder"] == decoder
                   and r["p_idx"] == idx]
            if not sel:
                continue
            z = np.array([r["z"] for r in sel], float)
            zo = np.array([r["z_optimistic"] for r in sel], float)
            summary.append({
                "p_idx": idx, "p": sel[0]["p"], "decoder": decoder,
                "comparisons": len(sel),
                "min_z": float(np.nanmin(z)),
                "min_z_optimistic": float(np.nanmin(zo)),
                "beats_curve_3sigma": int(sum(r["lrbd_beats_curve_3sigma"]
                                              for r in sel)),
                "beats_optimistic_3sigma": int(sum(
                    r["lrbd_beats_optimistic_3sigma"] for r in sel)),
                "median_error_ratio": float(np.nanmedian(
                    [r["lrbd_error_rate"] / r["gap_error_rate"]
                     for r in sel if r["gap_error_rate"] > 0]))})
    write_csv(out / "np_test_summary.csv", summary)
    write_csv(out / "fits_threshold.csv", results["thr"])
    write_csv(out / "fits_lrbd.csv", results["lrbd"])
    write_csv(out / "fits_rb.csv", results["rb"])
    write_csv(out / "matched_points.csv", results["mpts"])
    write_csv(out / "matched_fits.csv", results["mfits"])
    write_csv(out / "eff_fidelity.csv", results["eff"])

    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(figs / "pareto_all.pdf") as pdf:
            for an in analyses:
                if an.p == 0:
                    continue
                fig = plot_pareto_page(plt, an)
                pdf.savefig(fig)
                if an.p in (0.001, 0.002, 0.005, 0.01, 0.02):
                    fig.savefig(figs / f"pareto_p{an.idx}.png", dpi=130)
                plt.close(fig)
        plot_calibration(plt, analyses, figs)
        plot_fits_vs_threshold(plt, results["thr"], results["lrbd"],
                               results["rb"], figs)
        plot_fm_curves(plt, analyses, rb_by_p, figs)
        plot_matched(plt, results["mpts"], figs)
    print(f"done in {time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
