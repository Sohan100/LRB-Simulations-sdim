"""WP-F analysis: Markovianity tests for LRB-D versus LRB with correction.

Reads the outputs of ``studies/correction-decoders/qubit/scripts/nonmarkov_runs.py`` (default
``studies/correction-decoders/qubit/results/F``) and writes

* ``results/F/analysis.json``: every statistic below, per p and method;
* ``results/F/summary_<ptag>.md``: the per-method tables used in the note;
* ``results/F/twirl_toy.json``: numerical check of the twirl identity;
* figures in ``studies/correction-decoders/qubit/figures/F/``.

Estimator: per sequence the signed polarization ``F_s(m) = 1 - 2 e/a``
(``a`` accepted shots, ``e`` logical errors among them), averaged over the 30
sequences (the repo averages per-sequence ``|P0 - P1|``; the sign is kept here
because the ideal outcome is known and the absolute value biases small
signals upward). A depth is *usable* for a method if every sequence has at
least ``N_MIN`` accepted shots and the mean exceeds 4 bootstrap standard
errors.

Uncertainty: sequence-level bootstrap (resample the 30 sequences with
replacement, whole depth family together, B = 2000; the same resampling
indices for every method at one p, so method comparisons are paired). Shot
noise is included because each resampled sequence carries its own sampled
counts. Intervals are percentile 95% intervals.

Tests (per p and method):

1. windowed local decay rates ``r = -d log F / dm`` (width 2) with 95% CIs;
2. curvature ``c2`` of a weighted quadratic fit of ``log F`` on ``[2, m_hi]``;
   early/late rate ratio;
3. weighted fits of ``A f^m`` (M0), ``A f^m + B``, two exponentials,
   stretched exponential ``A exp(-l m^b)`` and ``A exp(-a m - g m^2)``;
   chi^2, AIC, BIC, and likelihood-ratio p-values calibrated by a null
   residual bootstrap (refits on ``M0_hat(m) + (Fbar*_b(m) - Fbar(m))``);
4. ``f`` versus the fit window start and end;
5. Markovian surrogate: ``A f^m`` fitted on the short window ``[1, 5]`` and
   extrapolated; deviation ``Delta(m) = log Fbar(m) - log F_surr(m)``;
6. check spacing (LRB-D ``unif=1,2,4`` against information-restricted MLD on
   the same rounds) and the inverse half (``inverse_check_every``).

Usage::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python studies/correction-decoders/qubit/scripts/nonmarkov_analysis.py [--res DIR] [--fig DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import experiment as E  # noqa: E402

DOC = ROOT / "studies" / "correction-decoders" / "qubit"
P_TAGS = (("p5e-4", 5e-4), ("p2e-3", 2e-3), ("p5e-3", 5e-3))
N_MIN = 100
N_BOOT = 2000
N_NULL = 300
SEED = 20261028
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
SHORT = (1, 6)

LABELS = {
    "mld5": "MLD (all rounds + final data)",
    "mld2": "MLD (all rounds, data-2 readout)",
    "raw": "no decoding, no rejection",
    "const0": "const=0 LRB-D (no checks)",
    "const0_mld": "const=0 decoded",
    "rb": "physical RB at 5p",
}


def label(name: str) -> str:
    if "@" in name:
        base, k = name.split("@")
        return f"{label(base)}, depths = 0 mod {k}"
    if name.startswith("x:"):
        base = "no decoding, no rejection" if name == "x:null" else label(
            name[2:])
        return base + " [exact]"
    if name == "null200":
        return "no decoding, no rejection [exact, 200 seq]"
    if name == "null200_nochecks":
        return "no checks, no decoding [exact, 200 seq]"
    if name in LABELS:
        return LABELS[name]
    if name.startswith("lrbd_"):
        rest = name[5:]
        final = rest.endswith("_final")
        rest = rest[:-6] if final else rest
        if rest == "sym":
            core = "LRB-D all rounds"
        else:
            policy, k = rest.rsplit("_", 1)
            core = f"LRB-D {policy}={k}"
        return core + (" + final data check" if final else " (data-2)")
    if name.startswith("mld5_t"):
        return f"MLD, accept |LLR| >= {name[6:]}"
    if name.startswith("mld5_"):
        rest = name[5:]
        if rest == "fwd":
            return "MLD, forward + terminal rounds only"
        policy, k = rest.rsplit("_", 1)
        return f"MLD on {policy}={k} rounds + final data"
    return name


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Curve:
    name: str
    depths: np.ndarray
    shots: np.ndarray      # (seq, depth)
    accepted: np.ndarray
    errors: np.ndarray
    seq_ids: Optional[np.ndarray] = None   # global sequence indices
    F_s: np.ndarray = field(init=False)
    A_s: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.seq_ids is None:
            self.seq_ids = np.arange(self.accepted.shape[0])
        a = self.accepted.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.F_s = np.where(a > 0, 1.0 - 2.0 * self.errors / a, np.nan)
        self.A_s = a / np.maximum(self.shots, 1)

    @property
    def n_seq(self) -> int:
        return self.F_s.shape[0]


def load_point(path: Path) -> dict[str, Curve]:
    pr = E.PointResult.from_npz(str(path))
    depths = np.asarray(pr.depths)
    first = next(iter(pr.tables.values()))
    keep = first.shots.sum(axis=1) > 0      # drop sequences never run
    curves = {}
    ids = np.flatnonzero(keep)
    for name, t in pr.tables.items():
        curves[name] = Curve(name, depths, t.shots[keep], t.accepted[keep],
                             t.errors[keep], ids)
    if "mld5" in pr.gap_hist and "mld5" in pr.tables:
        hist = pr.gap_hist["mld5"][keep]  # (seq, depth, bins, 2)
        shots = pr.tables["mld5"].shots[keep]
        for t in THRESHOLDS:
            idx = int(np.searchsorted(E.GAP_EDGES, t))
            assert abs(E.GAP_EDGES[idx] - t) < 1e-12
            acc = hist[:, :, idx:, :].sum(axis=(2, 3))
            err = hist[:, :, idx:, 1].sum(axis=2)
            name = f"mld5_t{t:g}"
            curves[name] = Curve(name, depths, shots, acc, err, ids)
    return curves


# Exact curves carry no shot noise. They are stored as Curve objects with a
# nominal 1e15 shots per cell (so the accepted-shot threshold and the
# binomial floor on the standard error are inactive); F_s and A_s are exact.
NOMINAL = 1e15


def exact_curve(name: str, depths, accept: np.ndarray,
                acc_flip: np.ndarray) -> Curve:
    shots = np.full(accept.shape, NOMINAL)
    return Curve(name, np.asarray(depths), shots, accept * NOMINAL,
                 acc_flip * NOMINAL)


def load_exact(path: Path) -> dict[str, Curve]:
    data = np.load(path)
    depths = np.asarray(data["depths"])
    out = {}
    for m in data["methods"]:
        m = str(m)
        out[f"x:{m}"] = exact_curve(f"x:{m}", depths, data[f"{m}__accept"],
                                    data[f"{m}__acc_flip"])
    return out


def load_null200(res_root: Path, p: float) -> dict[str, Curve]:
    """Lead's exact null-decoder curves (200 sequences, depths 0..40)."""
    path = res_root / "exact_null" / f"exact_null_p{p:g}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    depths = np.asarray(data["depths"])
    ones = np.ones_like(data["flip_checks_data5"])
    return {"null200": exact_curve("null200", depths, ones,
                                   data["flip_checks_data5"]),
            "null200_nochecks": exact_curve("null200_nochecks", depths, ones,
                                            data["flip_nochecks"])}


def lattice(c: Curve, k: int) -> Curve:
    """Restrict a curve to depths that are multiples of ``k`` (unif=k
    segments are then complete; a period-k sawtooth is removed)."""
    keep = (c.depths % k) == 0
    return Curve(f"{c.name}@{k}", c.depths[keep], c.shots[:, keep],
                 c.accepted[:, keep], c.errors[:, keep], c.seq_ids)


def sampled_sequences(path: Path) -> np.ndarray:
    """Boolean mask of the sequences present in a merged sampled file."""
    pr = E.PointResult.from_npz(str(path))
    first = next(iter(pr.tables.values()))
    return first.shots.sum(axis=1) > 0


def load_rb(path: Path) -> Curve:
    data = np.load(path)
    return Curve("rb", np.asarray(data["depths"]), data["shots"],
                 data["accepted"], data["errors"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wls(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares; ``Y`` is (n,) or (B, n). Returns coefficients."""
    Wx = X * w[:, None]
    G = np.linalg.solve(X.T @ Wx, Wx.T)  # (k, n)
    return (G @ Y.T).T if Y.ndim == 2 else G @ Y


def ci(samples: np.ndarray, axis: int = 0) -> tuple:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=axis)
    return lo, hi


# Model fits by profiled least squares on parameter grids. For every grid
# point the amplitudes enter linearly and are solved in closed form, so the
# same code fits the observed mean curve and hundreds of bootstrap replicates
# at once. Grids are nested (the exponential grid is contained in the
# stretched (beta = 1) and quadratic-log (gamma = 0) grids), so likelihood-ratio
# statistics are non-negative.

NPAR = {"exp": 2, "exp+B": 3, "2exp": 4, "stretched": 3, "quadlog": 3}


def _k1(Y, w, basis):
    """One-amplitude fits: Y (B, n), basis (c, n) -> chi2 (B, c), amp (B, c)."""
    S = (basis ** 2 * w).sum(axis=1)
    R = Y @ (basis * w).T
    YY = (Y ** 2 * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2 = YY[:, None] - R ** 2 / S[None, :]
        amp = R / S[None, :]
    return chi2, amp


def _k2(Y, w, e1, e2):
    """Two-amplitude fits for paired bases e1, e2 (c, n)."""
    S11 = (e1 ** 2 * w).sum(axis=1)
    S22 = (e2 ** 2 * w).sum(axis=1)
    S12 = (e1 * e2 * w).sum(axis=1)
    R1 = Y @ (e1 * w).T
    R2 = Y @ (e2 * w).T
    YY = (Y ** 2 * w).sum(axis=1)
    det = S11 * S22 - S12 ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        a1 = (S22 * R1 - S12 * R2) / det
        a2 = (S11 * R2 - S12 * R1) / det
        chi2 = YY[:, None] - (a1 * R1 + a2 * R2)
    chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
    return chi2, a1, a2


def grid_fits(m, Y, w, r0):
    """Fit all models to every row of ``Y`` (B, n); weights ``w`` = 1/sigma^2.

    Returns ``{model: (chi2 (B,), params (B, k))}``.
    """
    r0 = max(float(r0), 1e-5)
    R = r0 * np.exp(np.linspace(-3.0, 3.0, 401))
    E1 = np.exp(-np.outer(R, m))
    out = {}
    chi2, amp = _k1(Y, w, E1)
    i = np.argmin(chi2, axis=1)
    b = np.arange(len(Y))
    out["exp"] = (chi2[b, i], np.stack([amp[b, i], np.exp(-R[i])], axis=1))
    chi2, a1, a2 = _k2(Y, w, E1, np.ones_like(E1))
    i = np.argmin(chi2, axis=1)
    out["exp+B"] = (chi2[b, i], np.stack([a1[b, i], np.exp(-R[i]),
                                          a2[b, i]], axis=1))
    R2 = r0 * np.exp(np.linspace(-4.0, 4.0, 121))
    G = np.exp(-np.outer(R2, m))
    ii, jj = np.triu_indices(len(R2), k=1)
    chi2, a1, a2 = _k2(Y, w, G[ii], G[jj])
    i = np.argmin(chi2, axis=1)
    c2 = chi2[b, i]
    p2 = np.stack([a1[b, i], np.exp(-R2[ii[i]]), a2[b, i],
                   np.exp(-R2[jj[i]])], axis=1)
    worse = c2 > out["exp"][0]   # nesting: 2exp contains exp (A2 = 0)
    c2 = np.where(worse, out["exp"][0], c2)
    p2[worse] = np.concatenate([out["exp"][1][worse],
                                np.zeros((worse.sum(), 1)),
                                np.ones((worse.sum(), 1))], axis=1)
    out["2exp"] = (c2, p2)
    mref = float(np.mean(m)) if np.mean(m) > 0 else 1.0
    betas = np.round(np.arange(0.3, 3.0001, 0.05), 10)
    Rs = R[::2]
    rr, bb = np.meshgrid(Rs, betas, indexing="ij")
    rr, bb = rr.ravel(), bb.ravel()
    basis = np.exp(-rr[:, None] * mref * np.power(
        np.maximum(m, 0)[None, :] / mref, bb[:, None]))
    chi2, amp = _k1(Y, w, basis)
    i = np.argmin(chi2, axis=1)
    out["stretched"] = (chi2[b, i], np.stack([amp[b, i], rr[i], bb[i]],
                                             axis=1))
    span = max(float(m.max() - m.min()), 1.0)
    gammas = (r0 / span) * np.linspace(-3.0, 6.0, 91)
    gammas = np.concatenate([gammas, [0.0]])
    alphas = R[::2]
    aa, gg = np.meshgrid(alphas, gammas, indexing="ij")
    aa, gg = aa.ravel(), gg.ravel()
    basis = np.exp(-aa[:, None] * m[None, :] - gg[:, None] * m[None, :] ** 2)
    chi2, amp = _k1(Y, w, basis)
    i = np.argmin(chi2, axis=1)
    out["quadlog"] = (chi2[b, i], np.stack([amp[b, i], aa[i], gg[i]],
                                           axis=1))
    return out


def usable_depths(c: Curve, mean, sd) -> np.ndarray:
    ok = (c.accepted >= N_MIN).all(axis=0) & np.isfinite(mean) & (
        mean > 4 * sd)
    # contiguous from depth 0
    mask = np.zeros_like(ok)
    for i, flag in enumerate(ok):
        if not flag:
            break
        mask[i] = True
    return mask


def analyze_curve(c: Curve, idx: np.ndarray, rng: np.random.Generator,
                  null: bool = True, short=SHORT) -> dict:
    depths = c.depths.astype(float)
    Fb = np.nanmean(c.F_s[idx], axis=1)              # (B, D)
    mean = np.mean(c.F_s, axis=0)
    sd = np.nanstd(Fb, axis=0)
    # floor: a cell with no observed error still has binomial uncertainty
    n_acc = c.accepted.sum(axis=0).astype(float)
    sd = np.maximum(sd, 1.0 / np.sqrt(np.maximum(n_acc, 1.0)))
    Ab = np.mean(c.A_s[idx], axis=1)
    A_mean = c.A_s.mean(axis=0)
    use = usable_depths(c, mean, sd)
    out = {"name": c.name, "label": label(c.name), "n_seq": int(c.n_seq),
           "depths": c.depths.tolist(), "F": mean.tolist(),
           "F_sd": sd.tolist(), "F_lo": ci(Fb)[0].tolist(),
           "F_hi": ci(Fb)[1].tolist(),
           "F_abs": np.mean(np.abs(c.F_s), axis=0).tolist(),
           "A": A_mean.tolist(), "A_lo": ci(Ab)[0].tolist(),
           "A_hi": ci(Ab)[1].tolist(),
           "min_accepted": c.accepted.min(axis=0).tolist(),
           "usable": use.tolist()}
    m_use = depths[use]
    if len(m_use) < 6:
        out["status"] = "too few usable depths"
        return out
    m_hi = float(m_use.max())
    out["m_hi"] = m_hi
    with np.errstate(invalid="ignore", divide="ignore"):
        logF = np.log(mean)
        logFb = np.log(np.where(Fb > 0, Fb, np.nan))
    sd_log = sd / mean

    # --- windowed local rates (width 2 and 1) ---------------------------
    rates = {}
    for width in (1, 2, 4):
        rows = []
        for i, m1 in enumerate(depths):
            m2 = m1 + width
            j = np.flatnonzero(depths == m2)
            if not len(j) or not use[i] or not use[j[0]]:
                continue
            j = j[0]
            r = -(logF[j] - logF[i]) / width
            rb = -(logFb[:, j] - logFb[:, i]) / width
            lo, hi = ci(rb)
            rows.append([m1, m2, r, float(lo), float(hi)])
        if rows and width in (2, 4):
            # non-overlapping windows only
            first = rows[0][0]
            rows = [r for r in rows if (r[0] - first) % width == 0]
        rates[str(width)] = rows
    out["local_rates"] = rates

    # --- main window fits -------------------------------------------------
    sel = use & (depths >= 2)
    m = depths[sel]
    y = mean[sel]
    s = sd[sel]
    wlog = 1.0 / sd_log[sel] ** 2
    X1 = np.stack([np.ones_like(m), m], axis=1)
    X2 = np.stack([np.ones_like(m), m, m ** 2], axis=1)
    beta = wls(X1, logF[sel], wlog)
    beta_b = wls(X1, logFb[:, sel], wlog)
    f_ll = math.exp(beta[1])
    out["loglin"] = {"window": [float(m.min()), float(m.max())],
                     "f": f_ll, "f_ci": [float(np.exp(x)) for x in
                                         ci(beta_b[:, 1])],
                     "r": -beta[1], "r_ci": [float(-x) for x in
                                            ci(beta_b[:, 1])[::-1]],
                     "A": math.exp(beta[0])}
    gam = wls(X2, logF[sel], wlog)
    gam_b = wls(X2, logFb[:, sel], wlog)
    c2lo, c2hi = ci(gam_b[:, 2])
    out["curvature"] = {"c2": float(gam[2]), "c2_ci": [float(c2lo),
                                                       float(c2hi)],
                        "c2_sd": float(np.nanstd(gam_b[:, 2])),
                        # relative change of the local rate across the window
                        "rel_rate_change": float(
                            -2 * gam[2] * (m.max() - m.min()) / max(
                                -beta[1], 1e-12))}
    rel_b = -2 * gam_b[:, 2] * (m.max() - m.min()) / np.maximum(
        -beta_b[:, 1], 1e-12)
    out["curvature"]["rel_rate_change_ci"] = [float(x) for x in ci(rel_b)]

    # early / late rates
    L = max(4.0, math.floor((m.max() - 2) / 3))
    early = sel & (depths <= 2 + L)
    late = sel & (depths >= m.max() - L)
    re_, rl_ = None, None
    if early.sum() >= 3 and late.sum() >= 3:
        Xe = np.stack([np.ones(early.sum()), depths[early]], axis=1)
        Xl = np.stack([np.ones(late.sum()), depths[late]], axis=1)
        we = 1 / sd_log[early] ** 2
        wl = 1 / sd_log[late] ** 2
        re_ = -wls(Xe, logF[early], we)[1]
        rl_ = -wls(Xl, logF[late], wl)[1]
        reb = -wls(Xe, logFb[:, early], we)[:, 1]
        rlb = -wls(Xl, logFb[:, late], wl)[:, 1]
        out["early_late"] = {
            "early_window": [2.0, float(depths[early].max())],
            "late_window": [float(depths[late].min()), float(m.max())],
            "r_early": float(re_), "r_early_ci": [float(x) for x in ci(reb)],
            "r_late": float(rl_), "r_late_ci": [float(x) for x in ci(rlb)],
            "ratio": float(rl_ / re_) if re_ > 0 else None,
            "ratio_ci": [float(x) for x in ci(rlb / reb)],
            "diff": float(rl_ - re_),
            "diff_ci": [float(x) for x in ci(rlb - reb)]}

    # --- f versus fit window ------------------------------------------------
    win = []
    for m0 in (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22):
        w_sel = use & (depths >= m0)
        if w_sel.sum() < 4 or depths[w_sel].max() - m0 < 4:
            continue
        Xw = np.stack([np.ones(w_sel.sum()), depths[w_sel]], axis=1)
        ww = 1 / sd_log[w_sel] ** 2
        b = wls(Xw, logF[w_sel], ww)
        bb = wls(Xw, logFb[:, w_sel], ww)
        lo, hi = ci(-bb[:, 1])
        win.append(["start", m0, float(depths[w_sel].max()), float(-b[1]),
                    float(lo), float(hi)])
    for m1 in (4, 6, 8, 10, 14, 18, 22, 26, 30):
        w_sel = use & (depths <= m1)
        if w_sel.sum() < 4 or m1 > m_hi:
            continue
        Xw = np.stack([np.ones(w_sel.sum()), depths[w_sel]], axis=1)
        ww = 1 / sd_log[w_sel] ** 2
        b = wls(Xw, logF[w_sel], ww)
        bb = wls(Xw, logFb[:, w_sel], ww)
        lo, hi = ci(-bb[:, 1])
        win.append(["end", 0.0, float(m1), float(-b[1]), float(lo),
                    float(hi)])
    out["window_rates"] = win

    # --- Markovian surrogate from short depths ------------------------------
    s_sel = use & (depths >= short[0]) & (depths <= short[1])
    if s_sel.sum() >= 3:
        Xs = np.stack([np.ones(s_sel.sum()), depths[s_sel]], axis=1)
        ws = 1 / sd_log[s_sel] ** 2
        bs = wls(Xs, logF[s_sel], ws)
        bsb = wls(Xs, logFb[:, s_sel], ws)
        pred = bs[0] + bs[1] * depths
        predb = bsb[:, :1] + bsb[:, 1:2] * depths[None, :]
        delta = logF - pred
        deltab = logFb - predb
        dlo, dhi = ci(deltab)
        k_hi = int(np.flatnonzero(depths == m_hi)[0])
        out["surrogate"] = {
            "window": list(short), "r_short": float(-bs[1]),
            "r_short_ci": [float(-x) for x in ci(bsb[:, 1])[::-1]],
            "delta": np.where(use, delta, np.nan).tolist(),
            "delta_lo": np.where(use, dlo, np.nan).tolist(),
            "delta_hi": np.where(use, dhi, np.nan).tolist(),
            "delta_m_hi": float(delta[k_hi]),
            "delta_m_hi_ci": [float(dlo[k_hi]), float(dhi[k_hi])],
            "delta_m0": float(delta[0]),
            "delta_m0_ci": [float(dlo[0]), float(dhi[0])]}

    # --- model comparison (F space, weighted) ------------------------------
    wF = 1.0 / s ** 2
    obs = grid_fits(m, y[None, :], wF, -beta[1])
    fits = {}
    n = len(m)
    for name, (chi2, params) in obs.items():
        k = NPAR[name]
        fits[name] = {"params": [float(x) for x in params[0]],
                      "chi2": float(chi2[0]), "dof": n - k,
                      "aic": float(chi2[0]) + 2 * k,
                      "bic": float(chi2[0]) + k * math.log(n)}
    out["fits"] = fits
    if null:
        th0 = np.asarray(fits["exp"]["params"])
        base = th0[0] * th0[1] ** m
        dev = Fb[:, sel] - mean[sel][None, :]
        nb = min(N_NULL, dev.shape[0])
        pick = rng.choice(dev.shape[0], nb, replace=False)
        nul = grid_fits(m, base[None, :] + dev[pick], wF, -beta[1])
        chi0 = nul["exp"][0]
        out["gof_exp"] = {
            "chi2": fits["exp"]["chi2"], "dof": fits["exp"]["dof"],
            "p_null_bootstrap": float(
                (1 + np.sum(chi0 >= fits["exp"]["chi2"])) / (1 + len(chi0))),
            "chi2_null_median": float(np.median(chi0))}
        lr = {}
        for k in ("exp+B", "2exp", "stretched", "quadlog"):
            vals = chi0 - nul[k][0]
            obs_d = fits["exp"]["chi2"] - fits[k]["chi2"]
            lr[k] = {"delta_chi2": float(obs_d),
                     "p_null_bootstrap": float(
                         (1 + np.sum(vals >= obs_d)) / (1 + len(vals))),
                     "null_median": float(np.median(vals)),
                     "null_q95": float(np.quantile(vals, 0.95))}
        out["lr_tests"] = lr
    out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------
# Extra tests: drift on a common window, check spacing, two-sector law,
# threshold floors, exact low-order expansion
# ---------------------------------------------------------------------------

def _logF_boot(c: Curve, idx):
    Fb = np.nanmean(c.F_s[idx], axis=1)
    mean = np.mean(c.F_s, axis=0)
    sd = np.maximum(np.nanstd(Fb, axis=0), 1.0 / np.sqrt(np.maximum(
        c.accepted.sum(axis=0).astype(float), 1.0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.log(mean), np.log(np.where(Fb > 0, Fb, np.nan)),
                sd / mean, usable_depths(c, mean, sd))


def drift_stats(c: Curve, idx, lo: float, hi: float) -> dict:
    """Rates on [lo, hi], its first and last thirds, and the relative drift
    rho = (r_late - r_early) / r_all, with bootstrap samples."""
    logF, logFb, sdl, use = _logF_boot(c, idx)
    d = c.depths.astype(float)
    sel = use & (d >= lo) & (d <= hi)
    third = (hi - lo) / 3.0
    parts = {"all": sel, "early": sel & (d <= lo + third),
             "late": sel & (d >= hi - third)}
    res, boot = {}, {}
    for key, msk in parts.items():
        if msk.sum() < 2:
            return {}
        X = np.stack([np.ones(msk.sum()), d[msk]], axis=1)
        w = 1.0 / sdl[msk] ** 2
        res[key] = -wls(X, logF[msk], w)[1]
        boot[key] = -wls(X, logFb[:, msk], w)[:, 1]
    rho = (res["late"] - res["early"]) / res["all"]
    rhob = (boot["late"] - boot["early"]) / boot["all"]
    return {"r": res, "r_boot": boot, "rho": rho, "rho_boot": rhob}


def align(curves, names):
    """Restrict curves to their common sequence ids; return (dict, idx)."""
    common = None
    for n in names:
        ids = set(int(i) for i in curves[n].seq_ids)
        common = ids if common is None else common & ids
    common = np.array(sorted(common))
    out = {}
    for n in names:
        c = curves[n]
        rows = np.array([int(np.flatnonzero(c.seq_ids == i)[0])
                         for i in common])
        out[n] = Curve(c.name, c.depths, c.shots[rows], c.accepted[rows],
                       c.errors[rows], common)
    idx = np.random.default_rng(SEED + 23).integers(
        0, len(common), (N_BOOT, len(common)))
    return out, idx


def paired_drift(curves, idx, a: str, b: str) -> Optional[dict]:
    if a not in curves or b not in curves:
        return None
    curves, idx = align(curves, [a, b])
    ua = _logF_boot(curves[a], idx)[3]
    ub = _logF_boot(curves[b], idx)[3]
    d = curves[a].depths
    common = ua & ub
    if common.sum() < 8:
        return None
    hi = float(d[common].max())
    sa = drift_stats(curves[a], idx, 2.0, hi)
    sb = drift_stats(curves[b], idx, 2.0, hi)
    if not sa or not sb:
        return None
    diff = sa["rho_boot"] - sb["rho_boot"]
    return {"a": a, "b": b, "window": [2.0, hi],
            "n_seq": int(len(curves[a].seq_ids)),
            "rho_a": float(sa["rho"]),
            "rho_a_ci": [float(x) for x in ci(sa["rho_boot"])],
            "rho_b": float(sb["rho"]),
            "rho_b_ci": [float(x) for x in ci(sb["rho_boot"])],
            "abs_rho_a_minus_abs_rho_b": float(abs(sa["rho"]) - abs(sb["rho"])),
            "abs_diff_ci": [float(x) for x in ci(np.abs(sa["rho_boot"])
                                                  - np.abs(sb["rho_boot"]))],
            "rho_diff_ci": [float(x) for x in ci(diff)],
            "r_a": float(sa["r"]["all"]), "r_b": float(sb["r"]["all"])}


def spacing_table(curves, idx, family: list[str]) -> Optional[dict]:
    """Per-block rates of a family on a common window, ratios to the first."""
    if any(n not in curves for n in family):
        return None
    curves, idx = align(curves, family)
    use = np.ones(len(curves[family[0]].depths), bool)
    for n in family:
        use &= _logF_boot(curves[n], idx)[3]
    d = curves[family[0]].depths.astype(float)
    sel = use & (d >= 2)
    if sel.sum() < 5:
        return None
    rates, boots = {}, {}
    for n in family:
        logF, logFb, sdl, _ = _logF_boot(curves[n], idx)
        X = np.stack([np.ones(sel.sum()), d[sel]], axis=1)
        w = 1.0 / sdl[sel] ** 2
        rates[n] = float(-wls(X, logF[sel], w)[1])
        boots[n] = -wls(X, logFb[:, sel], w)[:, 1]
    base = family[0]
    return {"window": [2.0, float(d[sel].max())],
            "n_seq": int(len(curves[base].seq_ids)),
            "rates": rates,
            "rates_ci": {n: [float(x) for x in ci(boots[n])] for n in family},
            "ratio_to_first": {n: rates[n] / rates[base] for n in family},
            "ratio_ci": {n: [float(x) for x in ci(boots[n] / boots[base])]
                         for n in family}}


def two_sector(c: Curve, idx, D: float = 16.0) -> Optional[dict]:
    """const=1 two-sector law: A = A0 [e^{-rm} + (1 - e^{-rm})/D] and
    F = Fc / (1 + (e^{rm} - 1)/D); r fitted separately to A and to F."""
    d = c.depths.astype(float)
    Fb = np.nanmean(c.F_s[idx], axis=1)
    Ab = np.mean(c.A_s[idx], axis=1)
    F = np.mean(c.F_s, axis=0)
    A = c.A_s.mean(axis=0)
    sF = np.maximum(np.nanstd(Fb, axis=0), 1e-6)
    sA = np.maximum(np.std(Ab, axis=0), 1e-9)
    ok = (c.accepted >= N_MIN).all(axis=0) & (d >= 1)
    if ok.sum() < 6:
        return None
    m = d[ok]
    R = np.exp(np.linspace(np.log(1e-4), np.log(3.0), 3000))
    bA = np.exp(-np.outer(R, m)) + (1 - np.exp(-np.outer(R, m))) / D
    bF = 1.0 / (1.0 + (np.exp(np.outer(R, m)) - 1.0) / D)
    out = {}
    for key, basis, Y, Yb, s in (("A", bA, A[ok], Ab[:, ok], sA[ok]),
                                 ("F", bF, F[ok], Fb[:, ok], sF[ok])):
        w = 1.0 / s ** 2
        chi2, amp = _k1(np.vstack([Y[None, :], Yb]), w, basis)
        i = np.argmin(chi2, axis=1)
        r_hat = R[i[0]]
        out[f"r_{key}"] = float(r_hat)
        out[f"r_{key}_ci"] = [float(x) for x in ci(R[i[1:]])]
        out[f"chi2_{key}"] = float(chi2[0, i[0]])
        out[f"amp_{key}"] = float(amp[0, i[0]])
        out[f"dof_{key}"] = int(ok.sum() - 2)
        out[f"_boot_{key}"] = R[i[1:]]
    diff = out.pop("_boot_F") - out.pop("_boot_A")
    out["r_F_minus_r_A_ci"] = [float(x) for x in ci(diff)]
    out["D"] = D
    out["A_last"] = float(A[ok][-1])
    out["m_last"] = float(m[-1])
    return out


def threshold_floor(res: dict) -> dict:
    out = {}
    for t in THRESHOLDS:
        r = res.get(f"mld5_t{t:g}")
        if r is None:
            continue
        use = np.asarray(r["usable"], bool)
        F = np.asarray(r["F"])[use]
        lo = np.asarray(r["F_lo"])[use]
        k = int(np.argmin(F))
        out[f"{t:g}"] = {"tanh_t_over_2": math.tanh(t / 2),
                         "min_F": float(F[k]), "min_F_lo": float(lo[k]),
                         "at_m": int(np.asarray(r["depths"])[use][k]),
                         "A_at_m_hi": float(np.asarray(r["A"])[use][-1])}
    return out


LOW_DEC = ["mld5", "mld5_unif_2", "mld5_unif_4", "mld5_const_4",
           "mld5_const_1", "raw", "const0_mld"]
LOW_DET = ["lrbd_unif_1", "lrbd_unif_1_final", "lrbd_unif_2",
           "lrbd_unif_2_final", "lrbd_unif_4", "lrbd_unif_4_final",
           "lrbd_const_4_final", "lrbd_const_1_final", "const0", "lrbd_sym",
           "lrbd_sym_final"]


def low_order_analysis(path: Path) -> dict:
    """Affine-in-m test of the leading-order failure weight per method."""
    data = np.load(path)
    p = float(data["p"])
    d = np.asarray(data["depths"], float)
    keys = [str(k) for k in data["keys"]]
    n_seq = data[keys[0].replace(":", "__")].shape[0]
    idx = np.random.default_rng(SEED + 5).integers(0, n_seq, (N_BOOT, n_seq))
    out = {"p": p, "depths": d.tolist()}
    sel = d >= 2
    X = np.stack([np.ones(sel.sum()), d[sel], d[sel] ** 2], axis=1)
    X1 = X[:, :2]
    w = np.ones(sel.sum())
    span = d[sel].max() - d[sel].min()

    def summarize(arr, scale):
        y = arr.mean(axis=0) / scale
        yb = arr[idx].mean(axis=1) / scale
        g = wls(X, y[sel], w)
        gb = wls(X, yb[:, sel], w)
        b = wls(X1, y[sel], w)
        bb = wls(X1, yb[:, sel], w)
        e = (d >= 2) & (d <= 2 + span / 3)
        lt = (d >= d.max() - span / 3)
        Xe = np.stack([np.ones(e.sum()), d[e]], axis=1)
        Xl = np.stack([np.ones(lt.sum()), d[lt]], axis=1)
        se = wls(Xe, y[e], np.ones(e.sum()))[1]
        sl = wls(Xl, y[lt], np.ones(lt.sum()))[1]
        seb = wls(Xe, yb[:, e], np.ones(e.sum()))[:, 1]
        slb = wls(Xl, yb[:, lt], np.ones(lt.sum()))[:, 1]
        return {"mean": y.tolist(), "lo": ci(yb)[0].tolist(),
                "hi": ci(yb)[1].tolist(),
                "slope": float(b[1]), "slope_ci": [float(x) for x in
                                                   ci(bb[:, 1])],
                "intercept": float(b[0]),
                "quad": float(g[2]), "quad_ci": [float(x) for x in
                                                 ci(gb[:, 2])],
                "slope_early": float(se), "slope_early_ci": [
                    float(x) for x in ci(seb)],
                "slope_late": float(sl), "slope_late_ci": [
                    float(x) for x in ci(slb)],
                "late_over_early": float(sl / se) if se else None,
                "late_over_early_ci": [float(x) for x in ci(slb / seb)]}

    for name in LOW_DEC:
        key = f"{name}__S1"
        if key in data.files:
            out[f"{name}:S1/p"] = summarize(data[key], p)
        key = f"{name}__S1_late"
        if key in data.files:
            out[f"{name}:S1_late/p"] = summarize(data[key], p)
    for name in LOW_DET:
        for part in ("S2u", "S2u_late"):
            key = f"{name}__{part}"
            if key in data.files:
                out[f"{name}:{part}/p2"] = summarize(data[key], p * p)
        if f"{name}__S2u" in data.files and f"{name}__S2u_late" in data.files:
            out[f"{name}:S2u_forward/p2"] = summarize(
                data[f"{name}__S2u"] - data[f"{name}__S2u_late"], p * p)
        key = f"{name}__S1u"
        if key in data.files:
            out[f"{name}:S1u/p"] = summarize(data[key], p)
    return out


def plot_low_order(lo: dict, figdir: Path) -> None:
    tags = [t for t, _ in P_TAGS if t in lo]
    if not tags:
        return
    fig, axes = _fig(1, 2, w=4.8)
    tag = "p5e-4" if "p5e-4" in lo else tags[0]
    L = lo[tag]
    d = np.asarray(L["depths"])
    ax = axes[0][0]
    for i, name in enumerate(["mld5", "mld5_unif_4", "mld5_const_1", "raw",
                              "const0_mld"]):
        key = f"{name}:S1/p"
        if key not in L:
            continue
        col, mk = style(name, i)
        ax.plot(d, L[key]["mean"], color=col, marker=mk, ms=3.5, lw=1.6,
                label=label(name), markeredgecolor=SURFACE,
                markeredgewidth=0.6)
        ax.fill_between(d, L[key]["lo"], L[key]["hi"], color=col, alpha=0.12,
                        lw=0)
    key = "mld5:S1_late/p"
    if key in L:
        ax.plot(d, L[key]["mean"], color=SLOTS[0], ls="--", lw=1.4,
                label="MLD, late-detected faults only")
    ax.set_xlabel("depth m")
    ax.set_ylabel("first-order failure weight S1 / p")
    ax.set_title(f"decoders, exact first order (p = {L['p']:g})",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    ax = axes[0][1]
    for i, name in enumerate(["lrbd_unif_1_final", "lrbd_unif_2_final",
                              "lrbd_unif_4_final", "lrbd_const_4_final",
                              "lrbd_const_1_final", "const0"]):
        key = f"{name}:S2u/p2"
        if key not in L:
            continue
        col, mk = style(name.replace("_final", ""), i)
        ax.plot(d, L[key]["mean"], color=col, marker=mk, ms=3.5, lw=1.6,
                label=label(name), markeredgecolor=SURFACE,
                markeredgewidth=0.6)
        ax.fill_between(d, L[key]["lo"], L[key]["hi"], color=col, alpha=0.12,
                        lw=0)
    key = "lrbd_unif_1_final:S2u_late/p2"
    if key in L:
        ax.plot(d, L[key]["mean"], color=SLOTS[1], ls="--", lw=1.4,
                label="unif=1 + final, terminal-only pairs")
    ax.set_yscale("log")
    ax.set_xlabel("depth m")
    ax.set_ylabel("second-order undetected weight S2u / p^2")
    ax.set_title(f"LRB-D, exact second order (p = {L['p']:g})", fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(figdir / "low_order.png", dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Twirl identity check (toy, exact Clifford action on Pauli labels)
# ---------------------------------------------------------------------------

def _clifford_label_perms() -> np.ndarray:
    """(24, 4) array: perm[c, P] = label of C P C^dagger (0=I,1=X,2=Y,3=Z)."""
    import stim
    from lrb_qubit import code as C
    names = {"I": "I", "H": "H", "P": "S", "X": "X", "Z": "Z"}
    perms = np.zeros((24, 4), dtype=np.int64)
    for i, desc in enumerate(C.SINGLE_QUBIT_CLIFFORDS):
        circ = stim.Circuit()
        circ.append("I", [0])
        for token in C.clifford_gate_sequence(desc):
            circ.append(names[token], [0])
        tab = stim.Tableau.from_circuit(circ)
        for lab, pauli in ((1, "X"), (2, "Y"), (3, "Z")):
            out = str(tab(stim.PauliString(pauli)))[-1]
            perms[i, lab] = {"X": 1, "Y": 2, "Z": 3}[out]
    return perms


def twirl_toy(n_traj: int = 400_000, max_m: int = 40, seed: int = 7) -> dict:
    """Monte Carlo check of F(m) = E[(-1/3)^K] for correlated, biased errors.

    Each trajectory draws its own uniformly random Clifford sequence
    C_1..C_m from the 24-element catalog. A logical Pauli E_j (always Z: a
    maximally biased error) is applied after C_j with probability eps(h_j),
    where h_j is a two-state Markov chain (calm/burst). The error propagates
    to the start as U_j^dagger E_j U_j with U_j = C_j ... C_1; <X_L> of the
    mirror circuit is the product of the commutation signs. A frozen-rate
    model (eps fixed per trajectory, never mixing) is the non-Markovian
    counterexample.
    """
    rng = np.random.default_rng(seed)
    perms = _clifford_label_perms()
    inv = np.zeros_like(perms)
    for c in range(24):
        inv[c, perms[c]] = np.arange(4)
    anti_x = np.array([0, 0, 1, 1])  # Y, Z anticommute with X
    P = np.array([[0.95, 0.05], [0.30, 0.70]])
    eps = np.array([0.01, 0.15])
    pi = np.array([P[1, 0], P[0, 1]]) / (P[0, 1] + P[1, 0])
    models = {}
    # HMM-correlated errors
    h = (rng.random(n_traj) < pi[1]).astype(np.int64)
    uinv = np.tile(np.arange(4), (n_traj, 1))  # labels of U^dagger P U
    sign = np.ones(n_traj)
    K = np.zeros(n_traj, dtype=np.int64)
    F_mc, F_K = [1.0], [1.0]
    se = [0.0]
    for _ in range(max_m):
        c = rng.integers(0, 24, n_traj)
        # U_j = C_j U_{j-1}: U_j^dag P U_j = U_{j-1}^dag (C_j^dag P C_j) U_{j-1}
        uinv = np.take_along_axis(uinv, inv[c], axis=1)
        h = np.where(rng.random(n_traj) < P[h, 1], 1, 0)
        err = rng.random(n_traj) < eps[h]
        lab = uinv[:, 3]  # image of Z
        sign = np.where(err & (anti_x[lab] == 1), -sign, sign)
        K += err
        F_mc.append(float(sign.mean()))
        se.append(float(sign.std() / math.sqrt(n_traj)))
        F_K.append(float(np.mean((-1.0 / 3.0) ** K)))
    Tz = P * (1 - eps + (-1 / 3) * eps)[None, :]
    exact = [float(pi @ np.linalg.matrix_power(Tz, k) @ np.ones(2))
             for k in range(max_m + 1)]
    evals, evecs = np.linalg.eig(Tz.T)
    order = np.argsort(-np.abs(evals))
    lam1, lam2 = evals[order[0]].real, evals[order[1]].real
    eps_bar = float(pi @ eps)
    # connected correlations of the error indicators: a_0 a_d
    Ta = P * eps[None, :]
    cov = []
    for d in range(1, 60):
        joint = pi @ np.diag(eps) @ np.linalg.matrix_power(P, d - 1) @ Ta @ \
            np.ones(2)
        cov.append(float(joint - eps_bar ** 2))
    f_ind = 1 - 4 * eps_bar / 3
    log_lam_2nd = math.log(f_ind) + (16 / 9) * sum(cov) / f_ind ** 2
    models["hmm_bursts"] = {
        "P": P.tolist(), "eps": eps.tolist(), "eps_bar": eps_bar,
        "F_mc": F_mc, "F_mc_se": se, "F_from_K": F_K, "F_exact_gf": exact,
        "lambda1": lam1, "lambda2": lam2, "f_independent": f_ind,
        "log_lambda1_second_order": log_lam_2nd,
        "max_abs_z_mc_vs_exact": float(np.max(np.abs(
            (np.array(F_mc[1:]) - np.array(exact[1:]))
            / np.maximum(np.array(se[1:]), 1e-12))))}
    # frozen per-trajectory rate (non-mixing): a mixture of exponentials
    eps_f = np.where(rng.random(n_traj) < 0.5, 0.01, 0.12)
    uinv = np.tile(np.arange(4), (n_traj, 1))
    sign = np.ones(n_traj)
    F_mc, se = [1.0], [0.0]
    for _ in range(max_m):
        c = rng.integers(0, 24, n_traj)
        uinv = np.take_along_axis(uinv, inv[c], axis=1)
        err = rng.random(n_traj) < eps_f
        sign = np.where(err & (anti_x[uinv[:, 3]] == 1), -sign, sign)
        F_mc.append(float(sign.mean()))
        se.append(float(sign.std() / math.sqrt(n_traj)))
    exact = [0.5 * (1 - 4 * 0.01 / 3) ** k + 0.5 * (1 - 4 * 0.12 / 3) ** k
             for k in range(max_m + 1)]
    models["frozen_rate"] = {"eps": [0.01, 0.12], "F_mc": F_mc,
                             "F_mc_se": se, "F_exact": exact,
                             "max_abs_z_mc_vs_exact": float(np.max(np.abs(
                                 (np.array(F_mc[1:]) - np.array(exact[1:]))
                                 / np.maximum(np.array(se[1:]), 1e-12))))}
    return models


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#6250d6", "#e34948"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, ls="-")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _fig(nrows, ncols, w=4.2, h=3.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def _legend(fig, axes):
    """One shared legend below the panels (union over panels, in order)."""
    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc="lower center",
                   ncol=min(3, len(seen)), fontsize=8, frameon=False,
                   labelcolor=INK2)


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Colour follows the method across figures (slot order of the reference
# palette); methods that never share a figure may reuse a slot.
METHOD_STYLE = {
    "mld5": (0, "o"), "lrbd_unif_1": (1, "s"), "lrbd_const_1": (2, "^"),
    "const0_mld": (3, "D"), "rb": (4, "v"), "raw": (6, "X"),
    "lrbd_unif_1_final": (7, "P"), "lrbd_unif_2": (5, "*"),
    "lrbd_unif_4": (6, "D"), "lrbd_const_4": (7, "v"), "const0": (3, "o"),
    "mld5_unif_2": (2, "s"), "mld5_unif_4": (4, "^"),
    "mld5_const_4": (7, "D"), "mld5_const_1": (1, "v"),
    "mld5_t0.5": (1, "s"), "mld5_t1": (2, "^"), "mld5_t2": (3, "D"),
    "mld5_t4": (4, "v"), "mld5_fwd": (2, "s"), "lrbd_sym": (5, "*"),
    "mld2": (6, "o"), "lrbd_unif_2_final": (5, "P"),
}


def style(name, i):
    name = name.split("@")[0]
    if name.startswith("x:"):
        name = "raw" if name == "x:null" else name[2:]
    if name.startswith("null200"):
        name = "raw"
    slot, marker = METHOD_STYLE.get(name, (i % 8, MARKERS[i % 8]))
    return SLOTS[slot], marker


def plot_family(results, methods, fname, figdir, ylab="mean signed F",
                kind="F"):
    fig, axes = _fig(1, len(P_TAGS))
    for ax, (tag, p) in zip(axes[0], P_TAGS):
        res = results.get(tag, {})
        for i, name in enumerate(methods):
            r = res.get(name)
            if r is None:
                continue
            d = np.asarray(r["depths"], float)
            use = np.asarray(r["usable"], bool)
            if kind == "F":
                y = np.asarray(r["F"])
                lo, hi = np.asarray(r["F_lo"]), np.asarray(r["F_hi"])
            else:
                y = np.asarray(r["A"])
                lo, hi = np.asarray(r["A_lo"]), np.asarray(r["A_hi"])
                use = np.asarray(r["min_accepted"]) > 0
            col, mk = style(name, i)
            ax.plot(d[use], y[use], color=col, lw=1.6,
                    marker=mk, ms=3.5, label=label(name),
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
            ax.fill_between(d[use], lo[use], hi[use], color=col, alpha=0.12,
                            lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("depth m")
        ax.set_title(f"p = {p:g}", fontsize=10)
    axes[0][0].set_ylabel(ylab)
    _legend(fig, axes)
    nleg = len({l for ax in axes.flat
                for l in ax.get_legend_handles_labels()[1]})
    fig.tight_layout(rect=(0, 0.06 + 0.045 * math.ceil(nleg / 3), 1, 1))
    fig.savefig(figdir / fname, dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_local_rates(results, methods, fname, figdir, width="2",
                     normalize=True):
    fig, axes = _fig(1, len(P_TAGS))
    for ax, (tag, p) in zip(axes[0], P_TAGS):
        res = results.get(tag, {})
        for i, name in enumerate(methods):
            r = res.get(name)
            if r is None or r.get("status") != "ok":
                continue
            rows = np.asarray(r["local_rates"][width])
            if not len(rows):
                continue
            mid = 0.5 * (rows[:, 0] + rows[:, 1])
            scale = r["loglin"]["r"] if normalize else 1.0
            col, mk = style(name, i)
            ax.errorbar(mid, rows[:, 2] / scale,
                        yerr=[(rows[:, 2] - rows[:, 3]) / scale,
                              (rows[:, 4] - rows[:, 2]) / scale],
                        color=col, lw=1.4, marker=mk, ms=3.5,
                        capsize=0, label=label(name), elinewidth=0.8)
        if normalize:
            ax.axhline(1.0, color=INK2, lw=0.8)
            ax.set_ylim(-0.5, 3.0)
        ax.set_xlabel("window centre (depth)")
        ax.set_title(f"p = {p:g}", fontsize=10)
    axes[0][0].set_ylabel("local rate / fitted rate" if normalize
                          else "local rate -dlogF/dm")
    _legend(fig, axes)
    nleg = len({l for ax in axes.flat
                for l in ax.get_legend_handles_labels()[1]})
    fig.tight_layout(rect=(0, 0.06 + 0.045 * math.ceil(nleg / 3), 1, 1))
    fig.savefig(figdir / fname, dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_window(results, methods, fname, figdir):
    fig, axes = _fig(1, len(P_TAGS))
    for ax, (tag, p) in zip(axes[0], P_TAGS):
        res = results.get(tag, {})
        for i, name in enumerate(methods):
            r = res.get(name)
            if r is None or r.get("status") != "ok":
                continue
            rows = [w for w in r["window_rates"] if w[0] == "start"]
            if not rows:
                continue
            arr = np.asarray([w[1:] for w in rows], float)
            scale = r["loglin"]["r"]
            col, mk = style(name, i)
            ax.errorbar(arr[:, 0], arr[:, 2] / scale,
                        yerr=[(arr[:, 2] - arr[:, 3]) / scale,
                              (arr[:, 4] - arr[:, 2]) / scale],
                        color=col, lw=1.4, marker=mk, ms=3.5,
                        label=label(name), elinewidth=0.8)
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_xlabel("fit window start m0 (end = m_hi)")
        ax.set_title(f"p = {p:g}", fontsize=10)
        ax.set_ylim(0, 3)
    axes[0][0].set_ylabel("fitted rate / rate on [2, m_hi]")
    _legend(fig, axes)
    nleg = len({l for ax in axes.flat
                for l in ax.get_legend_handles_labels()[1]})
    fig.tight_layout(rect=(0, 0.06 + 0.045 * math.ceil(nleg / 3), 1, 1))
    fig.savefig(figdir / fname, dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_surrogate(results, methods, fname, figdir):
    fig, axes = _fig(1, len(P_TAGS))
    for ax, (tag, p) in zip(axes[0], P_TAGS):
        res = results.get(tag, {})
        for i, name in enumerate(methods):
            r = res.get(name)
            if r is None or "surrogate" not in r:
                continue
            d = np.asarray(r["depths"], float)
            s = r["surrogate"]
            y = np.asarray(s["delta"], float)
            lo = np.asarray(s["delta_lo"], float)
            hi = np.asarray(s["delta_hi"], float)
            ok = np.isfinite(y)
            col, mk = style(name, i)
            ax.plot(d[ok], y[ok], color=col, lw=1.6, marker=mk,
                    ms=3.5, label=label(name), markeredgecolor=SURFACE,
                    markeredgewidth=0.6)
            ax.fill_between(d[ok], lo[ok], hi[ok], color=col, alpha=0.12,
                            lw=0)
        ax.axhline(0.0, color=INK2, lw=0.8)
        ax.axvspan(SHORT[0], SHORT[1], color=GRID, alpha=0.6, lw=0)
        ax.set_xlabel("depth m")
        ax.set_title(f"p = {p:g}", fontsize=10)
    axes[0][0].set_ylabel("log F - log F_Markov(short-depth fit)")
    _legend(fig, axes)
    nleg = len({l for ax in axes.flat
                for l in ax.get_legend_handles_labels()[1]})
    fig.tight_layout(rect=(0, 0.06 + 0.045 * math.ceil(nleg / 3), 1, 1))
    fig.savefig(figdir / fname, dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_toy(toy, figdir):
    fig, axes = _fig(1, 2, w=4.6)
    ax = axes[0][0]
    t = toy["hmm_bursts"]
    m = np.arange(len(t["F_mc"]))
    ax.errorbar(m, t["F_mc"], yerr=2 * np.asarray(t["F_mc_se"]), color=SLOTS[0],
                marker="o", ms=3.5, lw=0, elinewidth=0.8,
                label="Monte Carlo <X_L> (twirled, Z-only errors)")
    ax.plot(m, t["F_exact_gf"], color=SLOTS[1], lw=1.6,
            label="generating function pi T(-1/3)^m 1")
    ax.plot(m, np.asarray(t["f_independent"]) ** m, color=SLOTS[2], lw=1.6,
            ls="-", label="independent blocks (1-4 eps/3)^m")
    ax.set_yscale("log")
    ax.set_xlabel("depth m")
    ax.set_ylabel("F(m)")
    ax.set_title("bursty (two-state HMM) logical errors", fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    ax = axes[0][1]
    t = toy["frozen_rate"]
    m = np.arange(len(t["F_mc"]))
    ax.errorbar(m, t["F_mc"], yerr=2 * np.asarray(t["F_mc_se"]), color=SLOTS[0],
                marker="o", ms=3.5, lw=0, elinewidth=0.8, label="Monte Carlo")
    ax.plot(m, t["F_exact"], color=SLOTS[1], lw=1.6,
            label="mixture 0.5 f_a^m + 0.5 f_b^m")
    ax.set_yscale("log")
    ax.set_xlabel("depth m")
    ax.set_title("frozen per-shot rate (non-mixing)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(figdir / "twirl_toy.png", dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}" if ax < 1 else f"{x:.{digits}g}"


def table_rows(res: dict, order: list[str]) -> list[str]:
    head = ("| method | usable m | f on [2, m_hi] [95% CI] | r = -ln f [95% CI]"
            " | c2 per block^2 [95% CI] | r_late - r_early [95% CI] (ratio)"
            " | Delta(m_hi) vs short-depth Markov [95% CI] | chi2/dof (exp),"
            " p_GOF | best alternative: Delta chi2, p | acceptance at m_hi |")
    rows = [head, "|" + "---|" * 10]
    for name in order:
        r = res.get(name)
        if r is None or name.startswith("__"):
            continue
        if r.get("status") != "ok":
            rows.append(f"| {label(name)} | {r.get('status')} |" + " |" * 8)
            continue
        ll = r["loglin"]
        cu = r["curvature"]
        el = r.get("early_late", {})
        su = r.get("surrogate", {})
        gof = r.get("gof_exp", {})
        lr = r.get("lr_tests", {})
        best = None
        for k, v in lr.items():
            if best is None or v["delta_chi2"] > lr[best]["delta_chi2"]:
                best = k
        best_s = (f"{best}: {fmt(lr[best]['delta_chi2'])}, "
                  f"{fmt(lr[best]['p_null_bootstrap'], 2)}" if best else "n/a")
        k_hi = r["depths"].index(int(r["m_hi"]))
        el_s = "n/a"
        if el:
            el_s = (f"{fmt(el['diff'])} [{fmt(el['diff_ci'][0])}, "
                    f"{fmt(el['diff_ci'][1])}] ({fmt(el.get('ratio'), 2)})")
        su_s = "n/a"
        if su:
            su_s = (f"{fmt(su['delta_m_hi'])} [{fmt(su['delta_m_hi_ci'][0])}, "
                    f"{fmt(su['delta_m_hi_ci'][1])}]")
        chi = gof.get("chi2", float("nan")) / max(gof.get("dof", 1), 1)
        rows.append(
            f"| {label(name)} | 0-{int(r['m_hi'])} | {ll['f']:.5f} "
            f"[{ll['f_ci'][0]:.5f}, {ll['f_ci'][1]:.5f}] | {fmt(ll['r'])} "
            f"[{fmt(ll['r_ci'][0])}, {fmt(ll['r_ci'][1])}] | {fmt(cu['c2'])} "
            f"[{fmt(cu['c2_ci'][0])}, {fmt(cu['c2_ci'][1])}] | {el_s} | "
            f"{su_s} | {fmt(chi)}, {fmt(gof.get('p_null_bootstrap'), 2)} | "
            f"{best_s} | {fmt(r['A'][k_hi])} |")
    return rows


def _ci_s(v, lo, hi, d=3):
    return f"{fmt(v, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


KEY_ROWS = ["rb", "x:null", "mld5", "mld2", "const0_mld", "mld5_unif_2@2",
            "mld5_unif_4@4", "mld5_const_4", "mld5_const_1", "mld5_t1",
            "mld5_t2", "x:lrbd_unif_1", "x:lrbd_unif_1_final",
            "x:lrbd_unif_2_final@2", "x:lrbd_unif_4_final@4",
            "x:lrbd_unif_2_final", "x:lrbd_const_4_final",
            "x:lrbd_const_1_final", "const0"]


def note_tables(results: dict) -> str:
    """Markdown tables quoted in notes/F-non-markovianity.md."""
    lines = []
    std = results.get("std", {})
    for tag, p in P_TAGS:
        res = std.get(tag)
        if not res:
            continue
        lines += [f"### Shape tests at p = {p:g}", "",
                  "| method | seq | depths | r = -ln f on [2, m_hi] |"
                  " c2 (per block^2) | drift rho = (r_late - r_early)/r |"
                  " Delta(m_hi) vs short-depth Markov | p_GOF (exp) |"
                  " LR p: 2exp / stretched / quadlog |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in KEY_ROWS:
            r = res.get(name)
            if not r or r.get("status") != "ok":
                continue
            ll, cu = r["loglin"], r["curvature"]
            el, su = r.get("early_late"), r.get("surrogate")
            rho = "n/a"
            if el and el.get("r_early"):
                rr = ll["r"]
                rho = _ci_s(el["diff"] / rr, el["diff_ci"][0] / rr,
                            el["diff_ci"][1] / rr, 2)
            dlt = (_ci_s(su["delta_m_hi"], *su["delta_m_hi_ci"], 2)
                   if su else "n/a")
            lr = r.get("lr_tests", {})
            lrs = " / ".join(fmt(lr[k]["p_null_bootstrap"], 2)
                             if k in lr else "n/a"
                             for k in ("2exp", "stretched", "quadlog"))
            lines.append(
                f"| {label(name)} | {r['n_seq']} | "
                f"{r['depths'][0]}-{int(r['m_hi'])} | "
                f"{_ci_s(ll['r'], *ll['r_ci'])} | "
                f"{_ci_s(cu['c2'], *cu['c2_ci'], 2)} | {rho} | {dlt} | "
                f"{fmt(r.get('gof_exp', {}).get('p_null_bootstrap'), 2)} | "
                f"{lrs} |")
        lines.append("")
        ex = res.get("__extra__", {})
        lines += [f"### Paired drift comparisons at p = {p:g} (common"
                  " sequences, same bootstrap resamples)", "",
                  "| A | B | sequences | window | rho_A | rho_B |"
                  " abs(rho_A) - abs(rho_B) |",
                  "|---|---|---|---|---|---|---|"]
        for pr in ex.get("pairs_exact", []) + ex.get("pairs", []):
            lines.append(
                f"| {label(pr['a'])} | {label(pr['b'])} | {pr['n_seq']} | "
                f"{int(pr['window'][0])}-{int(pr['window'][1])} | "
                f"{_ci_s(pr['rho_a'], *pr['rho_a_ci'], 2)} | "
                f"{_ci_s(pr['rho_b'], *pr['rho_b_ci'], 2)} | "
                f"{_ci_s(pr['abs_rho_a_minus_abs_rho_b'], *pr['abs_diff_ci'], 2)} |")
        lines.append("")
        lines += [f"### Check spacing at p = {p:g}", "",
                  "| family | window | r_1 | r_2 / r_1 | r_4 / r_1 |",
                  "|---|---|---|---|---|"]
        for key in ("spacing_lrbd_exact", "spacing_lrbd", "spacing_mld"):
            s = ex.get(key)
            if not s:
                continue
            names = list(s["rates"])
            lines.append(
                f"| {key.replace('spacing_', '')} ({label(names[0])} ...) | "
                f"{int(s['window'][0])}-{int(s['window'][1])} | "
                f"{_ci_s(s['rates'][names[0]], *s['rates_ci'][names[0]])} | "
                f"{_ci_s(s['ratio_to_first'][names[1]], *s['ratio_ci'][names[1]])} | "
                f"{_ci_s(s['ratio_to_first'][names[2]], *s['ratio_ci'][names[2]])} |")
        lines.append("")
        for key in ("two_sector_x:lrbd_const_1_final",
                    "two_sector_lrbd_const_1_final"):
            t = ex.get(key)
            if t:
                lines.append(
                    f"Two-sector law ({key[11:]}, D = 16): r_A = "
                    f"{_ci_s(t['r_A'], *t['r_A_ci'])}, r_F = "
                    f"{_ci_s(t['r_F'], *t['r_F_ci'])}, r_F - r_A in "
                    f"[{fmt(t['r_F_minus_r_A_ci'][0])}, "
                    f"{fmt(t['r_F_minus_r_A_ci'][1])}], chi2_A/dof = "
                    f"{fmt(t['chi2_A'] / t['dof_A'])}, chi2_F/dof = "
                    f"{fmt(t['chi2_F'] / t['dof_F'])}, A(30) = "
                    f"{fmt(t['A_last'])}.")
        tf = ex.get("threshold_floor", {})
        if tf:
            lines += ["", "| t | tanh(t/2) | min over m of F (95% lower) | at m |"
                      " acceptance at m_hi |", "|---|---|---|---|---|"]
            for t, v in tf.items():
                lines.append(f"| {t} | {v['tanh_t_over_2']:.3f} | "
                             f"{v['min_F']:.3f} ({v['min_F_lo']:.3f}) | "
                             f"{v['at_m']} | {fmt(v['A_at_m_hi'])} |")
        lines.append("")
    lo = results.get("low_order", {})
    if lo:
        lines += ["### Exact low-order expansion (30 sequences; slope and"
                  " curvature of the leading-order weight on m = 2..30)", "",
                  "| p | method | order | slope per block | quadratic coeff."
                  " | late/early slope |", "|---|---|---|---|---|---|"]
        rows = [("mld5:S1/p", "1 (p)"), ("mld5:S1_late/p", "1 (p)"),
                ("mld5_unif_4:S1/p", "1 (p)"), ("mld5_const_1:S1/p", "1 (p)"),
                ("raw:S1/p", "1 (p)"), ("const0_mld:S1/p", "1 (p)"),
                ("lrbd_unif_1_final:S2u/p2", "2 (p^2)"),
                ("lrbd_unif_1_final:S2u_late/p2", "2 (p^2)"),
                ("lrbd_unif_1_final:S2u_forward/p2", "2 (p^2)"),
                ("lrbd_unif_2_final:S2u/p2", "2 (p^2)"),
                ("lrbd_unif_4_final:S2u/p2", "2 (p^2)"),
                ("lrbd_const_4_final:S2u/p2", "2 (p^2)"),
                ("lrbd_const_1_final:S2u/p2", "2 (p^2)"),
                ("const0:S2u/p2", "2 (p^2)")]
        inv_rows = [("mld5:S1/p", "1 (p)"), ("mld5:S1_late/p", "1 (p)"),
                    ("raw:S1/p", "1 (p)"),
                    ("lrbd_sym_final:S2u/p2", "2 (p^2)"),
                    ("lrbd_sym_final:S2u_late/p2", "2 (p^2)"),
                    ("lrbd_unif_1_final:S2u/p2", "2 (p^2)"),
                    ("lrbd_unif_1_final:S2u_late/p2", "2 (p^2)")]
        keys = [(tag, p, rows, "") for tag, p in P_TAGS]
        keys += [(f"inv1_{tag}", p, inv_rows, "inv1 ") for tag, p in P_TAGS]
        keys += [(f"n150_{tag}", p, rows, "150 seq ") for tag, p in P_TAGS]
        keys += [(f"n150_inv1_{tag}", p, inv_rows, "150 seq inv1 ")
                 for tag, p in P_TAGS]
        for tag, p, rws, pre in keys:
            L = lo.get(tag)
            if not L:
                continue
            for key, order in rws:
                v = L.get(key)
                if not v:
                    continue
                lines.append(
                    f"| {p:g} | {pre}{key} | {order} | "
                    f"{_ci_s(v['slope'], *v['slope_ci'])} | "
                    f"{_ci_s(v['quad'], *v['quad_ci'], 2)} | "
                    f"{_ci_s(v['late_over_early'], *v['late_over_early_ci'])} |")
        lines.append("")
    for v in ("inv1",):
        inv = results.get(v, {})
        for tag, p in P_TAGS:
            res = inv.get(tag)
            if not res:
                continue
            lines += [f"### Inverse half with checks ({v}) at p = {p:g}", "",
                      "| method | depths | r | c2 | drift rho | p_GOF |",
                      "|---|---|---|---|---|---|"]
            for name in INV_ORDER + EXACT_INV:
                r = res.get(name)
                if not r or r.get("status") != "ok":
                    continue
                ll, cu, el = r["loglin"], r["curvature"], r.get("early_late")
                rho = "n/a"
                if el and el.get("r_early"):
                    rr = ll["r"]
                    rho = _ci_s(el["diff"] / rr, el["diff_ci"][0] / rr,
                                el["diff_ci"][1] / rr, 2)
                lines.append(
                    f"| {label(name)} | {r['depths'][0]}-{int(r['m_hi'])} | "
                    f"{_ci_s(ll['r'], *ll['r_ci'])} | "
                    f"{_ci_s(cu['c2'], *cu['c2_ci'], 2)} | {rho} | "
                    f"{fmt(r.get('gof_exp', {}).get('p_null_bootstrap'), 2)} |")
            lines.append("")
    toy = results.get("twirl_toy_summary")
    if toy:
        h = toy["hmm_bursts"]
        lines += ["### Twirl identity toy", "",
                  f"Bursty two-state HMM, Z-only logical errors: max |z| of Monte"
                  f" Carlo vs generating function over m = 1..40: "
                  f"{h['max_abs_z_mc_vs_exact']:.2f}; lambda1 = "
                  f"{h['lambda1']:.5f} vs independent-block f = "
                  f"{h['f_independent']:.5f}; second-order cumulant estimate of"
                  f" ln lambda1 = {h['log_lambda1_second_order']:.5f} (exact "
                  f"{math.log(h['lambda1']):.5f}); |lambda2/lambda1| = "
                  f"{abs(h['lambda2'] / h['lambda1']):.3f}. Frozen-rate mixture:"
                  f" max |z| = {toy['frozen_rate']['max_abs_z_mc_vs_exact']:.2f}.",
                  ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STD_ORDER = ["rb", "raw", "mld5", "mld2", "mld5_t0.5", "mld5_t1", "mld5_t2",
             "mld5_t4", "mld5_unif_2", "mld5_unif_4", "mld5_const_4",
             "mld5_const_1", "lrbd_unif_1", "lrbd_unif_1_final",
             "lrbd_unif_2", "lrbd_unif_2_final", "lrbd_unif_4",
             "lrbd_unif_4_final", "lrbd_const_4", "lrbd_const_4_final",
             "lrbd_const_1", "lrbd_const_1_final", "const0", "const0_mld"]
EXACT_STD = ["x:null", "x:lrbd_unif_1", "x:lrbd_unif_1_final",
             "x:lrbd_unif_2", "x:lrbd_unif_2_final", "x:lrbd_unif_4",
             "x:lrbd_unif_4_final", "x:lrbd_const_4", "x:lrbd_const_4_final",
             "x:lrbd_const_1", "x:lrbd_const_1_final", "null200",
             "null200_nochecks"]
LATTICE = ["lrbd_unif_2@2", "lrbd_unif_2_final@2", "x:lrbd_unif_2@2",
           "x:lrbd_unif_2_final@2", "mld5_unif_2@2", "lrbd_unif_4@4",
           "lrbd_unif_4_final@4", "x:lrbd_unif_4@4", "x:lrbd_unif_4_final@4",
           "mld5_unif_4@4"]
EXACT_INV = ["x:null", "x:lrbd_unif_1", "x:lrbd_unif_1_final", "x:lrbd_sym",
             "x:lrbd_sym_final"]
INV_ORDER = ["raw", "mld5", "mld5_fwd", "lrbd_unif_1", "lrbd_unif_1_final",
             "lrbd_sym", "lrbd_sym_final"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--res", default=str(DOC / "results" / "F"))
    parser.add_argument("--fig", default=str(DOC / "figures" / "F"))
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--no-toy", action="store_true")
    parser.add_argument("--variants", default="std,inv1,inv2")
    args = parser.parse_args()
    res_dir = Path(args.res)
    fig_dir = Path(args.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results: dict = {}
    for variant in args.variants.split(","):
        results[variant] = {}
        for tag, p in P_TAGS:
            path = res_dir / f"{variant}_{tag}.npz"
            xpath0 = res_dir / f"exact_{variant}_{tag}.npz"
            if not path.exists() and not xpath0.exists():
                continue
            curves = load_point(path) if path.exists() else {}
            if variant == "std" and (res_dir / f"rb_{tag}.npz").exists():
                curves["rb"] = load_rb(res_dir / f"rb_{tag}.npz")
            xpath = res_dir / f"exact_{variant}_{tag}.npz"
            if xpath.exists():
                # Exact curves cover all 30 sequences; paired comparisons
                # intersect sequence ids (see ``align``).
                curves.update(load_exact(xpath))
            if variant == "std":
                curves.update(load_null200(res_dir.parent, p))
                for base in ("lrbd_unif_2", "lrbd_unif_2_final",
                             "x:lrbd_unif_2", "x:lrbd_unif_2_final",
                             "mld5_unif_2"):
                    if base in curves:
                        curves[f"{base}@2"] = lattice(curves[base], 2)
                for base in ("lrbd_unif_4", "lrbd_unif_4_final",
                             "x:lrbd_unif_4", "x:lrbd_unif_4_final",
                             "mld5_unif_4"):
                    if base in curves:
                        curves[f"{base}@4"] = lattice(curves[base], 4)
            n_seq = next(iter(curves.values())).n_seq
            idx_cache = {}

            def idx_for(n):
                # the same resampling of the shared sequences for every method
                if n not in idx_cache:
                    idx_cache[n] = np.random.default_rng(SEED + 17).integers(
                        0, n, (N_BOOT, n))
                return idx_cache[n]

            idx = idx_for(n_seq)
            out = {}
            order = (STD_ORDER + EXACT_STD + LATTICE if variant == "std"
                     else INV_ORDER + EXACT_INV)
            for name in order:
                if name not in curves:
                    continue
                out[name] = analyze_curve(curves[name],
                                          idx_for(curves[name].n_seq), rng,
                                          null=not args.no_null)
                print(f"{variant} {tag} {name}: {out[name].get('status')}",
                      flush=True)
            extra = {}
            if variant == "std":
                extra["pairs"] = [x for x in (
                    paired_drift(curves, idx, a, b) for a, b in (
                        ("mld5", "lrbd_unif_1_final"),
                        ("mld5", "lrbd_unif_1"),
                        ("mld5_unif_2", "lrbd_unif_2_final"),
                        ("mld5_unif_4", "lrbd_unif_4_final"),
                        ("mld5_const_4", "lrbd_const_4_final"),
                        ("mld5_const_1", "lrbd_const_1_final"),
                        ("const0_mld", "const0"),
                        ("mld5", "rb"))) if x is not None]
                extra["spacing_lrbd"] = spacing_table(
                    curves, idx, ["lrbd_unif_1_final", "lrbd_unif_2_final",
                                  "lrbd_unif_4_final"])
                extra["spacing_mld"] = spacing_table(
                    curves, idx, ["mld5", "mld5_unif_2", "mld5_unif_4"])
                extra["spacing_lrbd_data2"] = spacing_table(
                    curves, idx, ["lrbd_unif_1", "lrbd_unif_2",
                                  "lrbd_unif_4"])
                extra["spacing_lrbd_exact"] = spacing_table(
                    curves, idx, ["x:lrbd_unif_1_final",
                                  "x:lrbd_unif_2_final",
                                  "x:lrbd_unif_4_final"])
                extra["pairs_exact"] = [x for x in (
                    paired_drift(curves, idx, a, b) for a, b in (
                        ("mld5", "x:lrbd_unif_1_final"),
                        ("mld5", "x:lrbd_unif_1"),
                        ("mld5_unif_2", "x:lrbd_unif_2_final"),
                        ("mld5_unif_4", "x:lrbd_unif_4_final"),
                        ("mld5_const_4", "x:lrbd_const_4_final"),
                        ("mld5_const_1", "x:lrbd_const_1_final"),
                        ("mld5", "x:null"), ("mld5", "raw"))) if x is not None]
                for nm in ("lrbd_const_1", "lrbd_const_1_final",
                           "x:lrbd_const_1", "x:lrbd_const_1_final"):
                    if nm in curves:
                        extra[f"two_sector_{nm}"] = two_sector(curves[nm],
                                                                idx)
                extra["threshold_floor"] = threshold_floor(out)
            else:
                extra["pairs"] = [x for x in (
                    paired_drift(curves, idx, a, b) for a, b in (
                        ("mld5", "lrbd_sym_final"),
                        ("mld5", "lrbd_unif_1_final"),
                        ("mld5", "x:lrbd_sym_final"),
                        ("mld5", "x:lrbd_unif_1_final"))) if x is not None]
            out["__extra__"] = extra
            results[variant][tag] = out
            md = [f"# {variant} {tag} (p = {p:g})", ""]
            md += table_rows(out, order)
            (res_dir / f"summary_{variant}_{tag}.md").write_text(
                "\n".join(md) + "\n")
    low = {}
    for tag, p in P_TAGS:
        for variant in ("std", "inv1"):
            path = res_dir / f"loworder_{variant}_{tag}.npz"
            if path.exists():
                low[f"{variant}_{tag}" if variant != "std" else tag] = \
                    low_order_analysis(path)
        path = res_dir / f"loworder_std_{tag}_n150.npz"
        if path.exists():
            low[f"n150_{tag}"] = low_order_analysis(path)
        path = res_dir / f"loworder_inv1_{tag}_n150.npz"
        if path.exists():
            low[f"n150_inv1_{tag}"] = low_order_analysis(path)
    if low:
        results["low_order"] = low
        plot_low_order(low, fig_dir)
    if not args.no_toy:
        toy = twirl_toy()
        (res_dir / "twirl_toy.json").write_text(json.dumps(toy, indent=1))
        plot_toy(toy, fig_dir)
        results["twirl_toy_summary"] = {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
            for k, v in toy.items()}
    (res_dir / "analysis.json").write_text(json.dumps(results, indent=1))
    (res_dir / "note_tables.md").write_text(note_tables(results))

    std = results.get("std", {})
    if std:
        plot_family(std, ["mld5", "lrbd_unif_1", "lrbd_unif_1_final",
                          "lrbd_const_1", "const0_mld", "rb", "raw"],
                    "F_main.png", fig_dir)
        plot_family(std, ["lrbd_unif_1", "lrbd_unif_2", "lrbd_unif_4",
                          "lrbd_const_4", "lrbd_const_1", "const0"],
                    "F_lrbd.png", fig_dir)
        plot_family(std, ["x:lrbd_unif_1_final", "x:lrbd_unif_2_final",
                          "x:lrbd_unif_4_final", "x:lrbd_const_4_final",
                          "x:lrbd_const_1_final", "x:null"],
                    "F_lrbd_exact.png", fig_dir)
        plot_local_rates(std, ["mld5", "x:lrbd_unif_1_final",
                               "x:lrbd_const_1_final", "x:null",
                               "const0_mld", "rb"],
                         "local_rates_exact.png", fig_dir)
        plot_surrogate(std, ["mld5", "x:lrbd_unif_1_final",
                             "x:lrbd_unif_2_final", "x:lrbd_const_4_final",
                             "x:null", "rb"],
                       "surrogate_exact.png", fig_dir)
        plot_local_rates(std, ["x:lrbd_unif_2_final", "x:lrbd_unif_4_final",
                               "mld5_unif_2", "mld5_unif_4"],
                         "local_rates_spacing_exact.png", fig_dir,
                         width="1")
        plot_family(std, ["mld5", "mld5_unif_2", "mld5_unif_4",
                          "mld5_const_4", "mld5_const_1", "const0_mld"],
                    "F_mld_info.png", fig_dir)
        plot_family(std, ["mld5", "mld5_t0.5", "mld5_t1", "mld5_t2",
                          "mld5_t4"], "F_thresholds.png", fig_dir)
        plot_family(std, ["mld5_t0.5", "mld5_t1", "mld5_t2", "mld5_t4",
                          "lrbd_unif_1", "lrbd_const_1"],
                    "A_postselected.png", fig_dir, ylab="acceptance",
                    kind="A")
        plot_local_rates(std, ["mld5", "lrbd_unif_1", "lrbd_const_1",
                               "const0_mld", "rb"], "local_rates_main.png",
                         fig_dir)
        plot_local_rates(std, ["lrbd_unif_2", "lrbd_unif_4", "mld5_unif_2",
                               "mld5_unif_4"], "local_rates_spacing.png",
                         fig_dir, width="1")
        plot_window(std, ["mld5", "lrbd_unif_1", "lrbd_unif_2", "const0_mld",
                          "rb"], "rate_vs_window.png", fig_dir)
        plot_surrogate(std, ["mld5", "lrbd_unif_1", "lrbd_unif_2",
                             "lrbd_const_1", "const0_mld", "rb"],
                       "surrogate_deviation.png", fig_dir)
    for v in ("inv1", "inv2"):
        if results.get(v):
            plot_family(results[v], ["mld5", "mld5_fwd",
                                     "x:lrbd_unif_1_final",
                                     "x:lrbd_sym_final", "x:null"],
                        f"F_{v}.png", fig_dir)
            plot_local_rates(results[v], ["mld5", "x:lrbd_unif_1_final",
                                          "x:lrbd_sym_final", "x:null"],
                             f"local_rates_{v}.png", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
