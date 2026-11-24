"""Estimators of a logical gate infidelity from terminal-check LRB-D data.

Input everywhere: ``counts[depth, outcome]`` pooled over sequences (or
``counts[seq, depth, outcome]``, pooled here), outcome = (s1, s2, l) flattened
as in tclib, and the depth coordinate ``x`` (m for the mirror protocol).
Every fit includes depth 0 when the data has it; no function drops it.

Estimators
----------
single_exponential   the production model  lam_acc(m) = b f^m  (for reference)
direct_quadratic     eps_acc(m) = c0 + c1 m + c2 m^2 on the accepted shots
mode_fit             "syndrome-resolved" fit: each Fourier mode (character) of
                     the full outcome distribution is B_k exp(-gamma_k m);
                     joint multinomial maximum likelihood. Two rate models:
                       'free'   one rate per mode,
                       'block'  rates additive over the code's independent
                                error blocks (qudits 0, 1, 4 and the pair 2-3).
                     The SPAM-free logical infidelity of n depth units between
                     a perfect preparation and a perfect check follows from
                     the rates alone:  eps*(n) = (d-1)/d (1 - Lambda(n)),
                     Lambda(n) = sum_L e^{-gamma_k n} / (1 + sum_S e^{-gamma_k n}).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import tclib as T


# --------------------------------------------------------------- structure

class Modes:
    """Character classes {k, -k} of Z_d^3 and their block decomposition."""

    def __init__(self, d: int):
        self.d = d
        g = T.char_grid(d)
        K = d ** 3
        cls = -np.ones(K, dtype=int)
        reps = []
        for k in range(1, K):
            if cls[k] >= 0:
                continue
            neg = int(np.ravel_multi_index(tuple((-g[k]) % d), (d, d, d)))
            cls[k] = cls[neg] = len(reps)
            reps.append(k)
        self.cls, self.reps, self.n = cls, reps, len(reps)
        self.S, self.L, self.acc = T.index_sets(d)
        self.F = np.real(T.fourier_matrix(d))  # used with real, inversion-symmetric chi
        self.Fc = T.fourier_matrix(d)
        # blocks: columns of the X-type check matrix (rows S1, S2, L) per data qudit
        sgn = 1 if d == 2 else -1
        col = {0: (1, 0, 1), 1: (1, sgn, 0), 2: (sgn, 0, 0), 3: (0, 1, 1), 4: (0, sgn, 0)}
        block_params = {}
        M = []
        for k in reps:
            row = {}
            for q in (0, 1, 4):
                if int(g[k] @ np.array(col[q])) % d:
                    row[("q", q)] = 1
            xy = (int(g[k] @ np.array(col[2])) % d, int(g[k] @ np.array(col[3])) % d)
            if xy != (0, 0):
                neg = ((-xy[0]) % d, (-xy[1]) % d)
                row[("q23", min(xy, neg))] = 1
            for key in row:
                block_params.setdefault(key, len(block_params))
            M.append(row)
        self.block_names = sorted(block_params, key=block_params.get)
        self.M_block = np.zeros((self.n, len(block_params)))
        for i, row in enumerate(M):
            for key in row:
                self.M_block[i, block_params[key]] = 1.0

    def expand(self, per_class: np.ndarray) -> np.ndarray:
        """Per-class values -> all K characters (value 0 slot for k = 0 set by caller)."""
        out = np.zeros(per_class.shape[:-1] + (self.d ** 3,))
        out[..., 1:] = per_class[..., self.cls[1:]]
        return out


_MODES = {}


def modes(d: int) -> Modes:
    if d not in _MODES:
        _MODES[d] = Modes(d)
    return _MODES[d]


def pool(counts: np.ndarray) -> np.ndarray:
    return counts.sum(0) if counts.ndim == 3 else counts


# ------------------------------------------------------- accepted-shot fits

def accepted_stats(counts: np.ndarray, d: int):
    """Acceptance, accepted polarization and its binomial standard error per depth."""
    c = pool(counts).astype(float)
    md = modes(d)
    na = c[:, md.acc]
    n_acc = na.sum(1)
    n_tot = c.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = na[:, 0] / n_acc
        lam = (d * p0 - 1.0) / (d - 1.0)
        se = (d / (d - 1.0)) * np.sqrt(np.clip(p0 * (1 - p0), 0, None) / n_acc)
        # avoid zero weights when no error was seen: one-count floor
        se = np.maximum(se, (d / (d - 1.0)) / np.maximum(n_acc, 1.0))
    return {"A": n_acc / n_tot, "n_acc": n_acc, "lam": lam, "se": se,
            "eps": (d - 1.0) / d * (1 - lam), "eps_se": (d - 1.0) / d * se}


def single_exponential(x, counts, d, min_accepted=1):
    """Production model lam = b f^m, weighted by binomial errors, all depths (m = 0 included)."""
    st = accepted_stats(counts, d)
    x = np.asarray(x, float)
    ok = st["n_acc"] >= min_accepted
    xs, y, s = x[ok], st["lam"][ok], st["se"][ok]
    best = None
    lo, hi = 1e-8, 1.0
    for _ in range(6):
        fg = np.linspace(lo, hi, 2001)
        z = fg[None, :] ** xs[:, None]
        w = 1.0 / s[:, None] ** 2
        b = (w * z * y[:, None]).sum(0) / (w * z * z).sum(0)
        chi2 = (w * (y[:, None] - b * z) ** 2).sum(0)
        i = int(np.argmin(chi2))
        best = (fg[i], b[i], chi2[i])
        span = (hi - lo) / 10
        lo, hi = max(1e-8, fg[i] - span), min(1.0, fg[i] + span)
    f, b, chi2 = best
    return {"f": f, "b": b, "r": (d - 1.0) / d * (1 - f), "chi2": chi2, "dof": int(ok.sum()) - 2}


def direct_quadratic(x, counts, d, min_rel_acceptance=0.5, linear=True):
    """eps_acc(m) = c0 + c1 m + c2 m^2 by weighted least squares.

    Uses every depth, m = 0 included, at which the acceptance is at least
    ``min_rel_acceptance`` times the depth-0 acceptance (deeper points leave
    the range where the quadratic law holds). ``linear=False`` fixes c1 = 0.
    """
    st = accepted_stats(counts, d)
    x = np.asarray(x, float)
    ok = st["A"] >= min_rel_acceptance * st["A"][np.argmin(x)]
    xs, y, s = x[ok], st["eps"][ok], st["eps_se"][ok]
    cols = [np.ones_like(xs), xs, xs ** 2] if linear else [np.ones_like(xs), xs ** 2]
    if len(xs) < len(cols):
        return {"c0": np.nan, "c1": np.nan, "c2": np.nan, "c2_se_fit": np.nan, "chi2": np.nan,
                "dof": -1, "n_depths": int(ok.sum()), "x_max": float(xs.max())}
    Amat = np.vstack(cols).T / s[:, None]
    coef, *_ = np.linalg.lstsq(Amat, y / s, rcond=None)
    cov = np.linalg.pinv(Amat.T @ Amat)
    resid = (y - np.vstack(cols).T @ coef) / s
    out = {"c0": coef[0], "c2": coef[-1], "c2_se_fit": float(np.sqrt(cov[-1, -1])),
           "c1": coef[1] if linear else 0.0, "chi2": float(resid @ resid),
           "dof": len(xs) - len(cols), "n_depths": int(ok.sum()), "x_max": float(xs.max())}
    return out


# ------------------------------------------------------------------ mode fit

def empirical_chars(counts, d):
    c = pool(counts).astype(float)
    P = c / c.sum(1, keepdims=True)
    return T.chars_from_dist(P, d)


def _initial(x, counts, d):
    """Per-mode log-linear least squares as a starting point."""
    md = modes(d)
    chi = empirical_chars(counts, d)[:, md.reps]
    x = np.asarray(x, float)
    logB, gam = np.zeros(md.n), np.zeros(md.n)
    for i in range(md.n):
        ok = chi[:, i] > 0.05
        if ok.sum() >= 2:
            w = chi[ok, i] ** 2
            Amat = np.vstack([np.ones(ok.sum()), -x[ok]]).T * np.sqrt(w)[:, None]
            sol, *_ = np.linalg.lstsq(Amat, np.log(chi[ok, i]) * np.sqrt(w), rcond=None)
            logB[i], gam[i] = min(sol[0], 0.0), max(sol[1], 1e-9)
        else:
            gam[i] = 1.0
    return logB, gam


def mode_fit(x, counts, d, rates="free", fix_amplitudes=False):
    """Joint multinomial maximum-likelihood fit of chi_k(m) = B_k exp(-gamma_k m).

    rates='free': one gamma per mode class. rates='block': gamma = M g with one
    g >= 0 per independent error block. ``fix_amplitudes`` sets B_k = 1
    (ideal SPAM, as in the simulations with noiseless encoder and readout).
    Returns the parameters and the derived SPAM-free quantities.
    """
    md = modes(d)
    x = np.asarray(x, float)
    c = pool(counts).astype(float)
    ntot = c.sum(1)
    Mr = np.eye(md.n) if rates == "free" else md.M_block
    nr = Mr.shape[1]
    logB0, gam0 = _initial(x, counts, d)
    g0, *_ = np.linalg.lstsq(Mr, gam0, rcond=None)
    g0 = np.clip(g0, 1e-9, None)
    Fk = md.F[:, 1:]                 # P = 1/K + Fk @ chi[1:]
    cls1 = md.cls[1:]
    scale = max(gam0.mean(), 1e-6)   # rates are optimized in units of their typical size

    def unpack(theta):
        g = theta[:nr] * scale
        logB = np.zeros(md.n) if free_amp is False else theta[nr:]
        return g, logB

    delta = 1e-9  # below this probability -log is continued quadratically, so the
    #               objective stays finite and smooth if a trial point leaves the simplex

    def nll(theta):
        g, logB = unpack(theta)
        gam = Mr @ g
        chi = np.exp(logB[None, :] - np.outer(x, gam))          # depth x class
        chik = chi[:, cls1]                                       # depth x (K-1)
        P = 1.0 / d ** 3 + chik @ Fk.T                           # depth x K
        low = P < delta
        Ps = np.where(low, delta, P)
        u = (P - delta) / delta
        logP = np.where(low, np.log(delta) + u - 0.5 * u * u, np.log(Ps))
        dlog = np.where(low, (1.0 - u) / delta, 1.0 / Ps)
        val = -(c * logP).sum()
        dchi = (c * dlog) @ Fk                                    # depth x (K-1)
        dcls = np.zeros((len(x), md.n))
        np.add.at(dcls.T, cls1, (dchi * chik).T)
        grad_logB = -dcls.sum(0)
        grad_gam = (dcls * x[:, None]).sum(0)
        grad = np.concatenate([Mr.T @ grad_gam * scale, [] if free_amp is False else grad_logB])
        return val, grad

    def solve(theta0, with_amp):
        nonlocal free_amp
        free_amp = with_amp
        bounds = [(0.0, None)] * nr + ([(None, 0.0)] * md.n if with_amp else [])
        return minimize(nll, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 20000, "maxfun": 100000, "ftol": 1e-14, "gtol": 1e-10})

    free_amp = False
    fixed = solve(g0 / scale, False)                      # rates only, B = 1
    if fix_amplitudes:
        res = fixed
    else:
        # two starts (per-mode least squares; rates-only optimum with B = 1), keep the better
        cands = [solve(np.concatenate([g0 / scale, logB0]), True),
                 solve(np.concatenate([fixed.x, np.zeros(md.n)]), True)]
        res = min(cands, key=lambda r: r.fun)
        free_amp = True
    g, logB = unpack(res.x)
    gam = Mr @ g
    out = {"gamma": gam, "logB": logB, "g": g, "nll": float(res.fun), "success": bool(res.success)}
    out.update(derived(gam, d))
    # goodness of fit: Pearson chi2 over cells with expected count >= 5
    chi = np.exp(logB[None, :] - np.outer(x, gam))
    P = 1.0 / d ** 3 + chi[:, cls1] @ Fk.T
    exp = ntot[:, None] * P
    big = exp >= 5
    out["pearson"] = float((((c - exp) ** 2) / np.where(big, exp, 1.0))[big].sum())
    out["pearson_cells"] = int(big.sum())
    out["n_par"] = len(res.x)
    return out


def Lambda(gam_class: np.ndarray, n, d: int):
    """SPAM-free accepted polarization after n depth units, from mode rates."""
    md = modes(d)
    n = np.atleast_1d(np.asarray(n, float))
    u = np.exp(-np.outer(n, gam_class))            # n x class
    uk = md.expand(u)                              # n x K (k = 0 slot is 0)
    num = uk[:, md.L].sum(1)
    den = 1.0 + uk[:, md.S].sum(1)
    return num / den


def derived(gam_class: np.ndarray, d: int) -> dict:
    md = modes(d)
    gk = md.expand(gam_class)
    L1, S1 = gk[md.L].sum(), gk[md.S].sum()
    L2, S2 = (gk[md.L] ** 2).sum(), (gk[md.S] ** 2).sum()
    pref = (d - 1.0) / d
    return {
        "eps1": float(pref * (1 - Lambda(gam_class, 1.0, d)[0])),   # one depth unit
        "alpha": float(pref * (L1 - S1) / d ** 2),                   # linear coefficient of eps*(n)
        "beta": float(pref * ((S2 - L2) / (2 * d ** 2) + S1 * (L1 - S1) / d ** 4)),  # quadratic
        "discard1": float(1.0 - (1.0 + np.exp(-gk[md.S]).sum()) / d ** 2),  # rejection of one unit
    }


def eps_star(gam_class, n, d):
    return (d - 1.0) / d * (1 - Lambda(gam_class, n, d))


# ---------------------------------------------------------------- bootstrap

def bootstrap(fn, counts_seq: np.ndarray, n_boot: int = 200, seed: int = 0):
    """One-standard-error bootstrap over whole sequences (axis 0 of counts_seq).

    ``fn(pooled_counts) -> dict of floats``; returns {key: (mean, std)} over
    replicates, sequences resampled with replacement and kept whole across
    depths.
    """
    rng = np.random.default_rng(seed)
    n_seq = counts_seq.shape[0]
    vals = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n_seq, n_seq)
        r = fn(counts_seq[idx].sum(0))
        if r is None:
            continue
        for k, v in r.items():
            if np.isscalar(v):
                vals.setdefault(k, []).append(float(v))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in vals.items()}


# ------------------------------------------------- hardware-facing entry point

def record_to_outcome(x: np.ndarray, d: int) -> np.ndarray:
    """Map terminal X-basis data readouts x[..., 5] (values 0..d-1, data qudits
    0..4) to the outcome index s1 d^2 + s2 d + l used everywhere here.

    Qutrit: s1 = x0 + x1 - x2, s2 = x3 - x1 - x4, l = x0 + x3 (mod 3), the
    detector and observable expressions of the production const-0 circuits.
    Qubit: the same sums modulo 2.
    """
    x = np.asarray(x, dtype=np.int64)
    s1 = (x[..., 0] + x[..., 1] - x[..., 2]) % d
    s2 = (x[..., 3] - x[..., 1] - x[..., 4]) % d
    lo = (x[..., 0] + x[..., 3]) % d
    return s1 * d * d + s2 * d + lo


def analyze(x, counts_seq: np.ndarray, d: int, f_cov: float = 1.0, n_boot: int = 300, seed: int = 0) -> dict:
    """Terminal-check analysis of one experiment.

    x           depth coordinate per depth index (mirror: m; compiled: number of noisy Cliffords)
    counts_seq  counts[sequence, depth, outcome] with outcome from ``record_to_outcome``.
                With independent sequences at every depth, axis 0 is just a circuit index.
    f_cov       covariance factor of the compiled Clifford set (tcref.reference), 1 to leave it out.

    Returns point estimates with one-standard-error sequence-bootstrap errors:
      eps_block      block-rate fit: logical infidelity of one depth unit from coincidences
                     of independent errors (lower estimate; times f_cov in eps_block_cov)
      eps_free       free-rate fit: no independence assumption (upper estimate, noisy)
      discard        rejected fraction of one depth unit, SPAM-free
      pred_over_obs  per depth, block-model prediction / observed accepted infidelity
    """
    md = modes(d)
    x = np.asarray(x, float)

    def est(c):
        mb = mode_fit(x, c, d, "block")
        mf = mode_fit(x, c, d, "free")
        return {"eps_block": mb["eps1"], "eps_block_cov": mb["eps1"] * f_cov, "eps_free": mf["eps1"],
                "alpha_free": mf["alpha"], "discard": mb["discard1"]}

    pooled = counts_seq.sum(0)
    point = est(pooled)
    boot = bootstrap(est, counts_seq, n_boot=n_boot, seed=seed)
    out = {k: (point[k], boot[k][1]) for k in point}
    mb = mode_fit(x, pooled, d, "block")
    chi = np.exp(mb["logB"][None, :] - np.outer(x, mb["gamma"]))
    P = 1.0 / d ** 3 + md.expand(chi)[:, 1:] @ md.F[:, 1:].T
    _, _, lam = T.accept_and_polarization(P, d)
    st = accepted_stats(pooled, d)
    out["depth_x"] = x
    out["accept_observed"] = st["A"]
    out["eps_observed"] = st["eps"]
    out["eps_observed_se"] = st["eps_se"]
    out["eps_predicted"] = T.infidelity_from_polarization(lam, d)
    out["block_rates"] = dict(zip([str(n) for n in md.block_names], mb["g"]))
    return out
