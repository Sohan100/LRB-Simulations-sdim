#!/usr/bin/env python3
"""Critic check for N1 (baseline non-Markovian signatures), part 1.

Uses the PER-SEQUENCE outcome counts that the run folder keeps in
progress/<p_idx>/{unif,const}_check_data/<k>.npy (shape 9 depths x 30
sequences x 3 logical outcomes) and <k>_rejected.npy (9 x 30).  N1 states
that only means and stds over sequences exist; these files allow exact tests.

Checks:
  [A] the npy counts reproduce the result CSVs (mean and np.std ddof=0).
  [B] m = 0 shot noise: across-sequence dispersion of the fidelity, of the
      rejected count and of the logical-error count, against multinomial /
      binomial predictions, per p and pooled over the 14 p values (C19).
  [C] empirical depth-depth correlation of per-sequence fidelities against
      the "Brownian" model of N1 section 3.4 (C18).
  [D] sequence bootstrap (resample the 30 Clifford sequences, keeping each
      sequence's 9 depths together) of: the log-quadratic curvature d, the
      model-E0 quadratic coefficient b2 (m > 0), the unif-1 m = 0 offset, and
      the windowed-rate ratio.  Compared with N1's diagonal-SEM z values.
  [E] exact per-sequence shot variance (binomial, own N_acc) against the
      across-sequence variance, i.e. the "shot fraction" of N1 section 3.4.

Read-only on the run folder.  Runtime ~1 min on one core.
Usage: module load python/3.11 && python critic_N1-baseline-nonmarkov_persequence.py
"""
import csv
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import stats

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22], dtype=float)
PROBS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 1e-2, 1.25e-2,
         1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2]
NSEQ = 30
SHOTS = 1_000_000
OMEGA = np.exp(2j * np.pi / 3)
RNG = np.random.default_rng(20261028)
NBOOT = 4000


def load_counts(pidx, kind, k):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    base = os.path.join(RUN, "progress", str(pidx), sub)
    cnt = np.load(os.path.join(base, f"{k}.npy"))           # (9, 30, 3)
    rej = np.load(os.path.join(base, f"{k}_rejected.npy"))  # (9, 30)
    return cnt.astype(np.int64), rej.astype(np.int64)


def load_csv(pidx, kind, k):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    path = os.path.join(RUN, "results", "LRB", str(pidx), sub, f"{k}.csv")
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    arr = [np.array([float(v) for v in r[1:]]) for r in rows[1:5]]
    return arr  # mean, std, rej mean, rej std


def per_seq_fid(cnt):
    nacc = cnt.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (cnt[..., 0] + cnt[..., 1] * OMEGA + cnt[..., 2] * OMEGA**2) / nacc
    return np.abs(z), nacc


# ------------------------------------------------------------------ [A]
def check_A():
    worst_mean, worst_std, worst_rej, n = 0.0, 0.0, 0.0, 0
    for pidx in range(1, 15):
        for kind, ks in (("unif", range(1, 23)), ("const", range(0, 23))):
            for k in ks:
                cnt, rej = load_counts(pidx, kind, k)
                assert np.all(cnt.sum(axis=2) + rej == SHOTS)
                fid, _ = per_seq_fid(cnt)
                mean, std, rmean, _ = load_csv(pidx, kind, k)
                with np.errstate(invalid="ignore"):
                    m_mine = fid.mean(axis=1)
                    s_mine = fid.std(axis=1)
                ok = np.isfinite(mean)
                worst_mean = max(worst_mean, np.nanmax(np.abs(m_mine[ok] - mean[ok])))
                worst_std = max(worst_std, np.nanmax(np.abs(s_mine[ok] - std[ok])))
                worst_rej = max(worst_rej, np.max(np.abs(rej.mean(axis=1) / SHOTS - rmean)))
                n += 1
    print(f"[A] {n} policy/p files: max|mean diff| = {worst_mean:.2e}, "
          f"max|std diff| = {worst_std:.2e}, max|rejected diff| = {worst_rej:.2e}")


# ------------------------------------------------------------------ [B]
def check_B():
    print("[B] m = 0 dispersion (30 identical circuits per p; unif1 file)")
    rows = []
    sum_df = sum_dr = sum_de = 0.0
    dof = 0
    for pidx in range(1, 15):
        cnt, rej = load_counts(pidx, "unif", 1)
        c0, r0 = cnt[0], rej[0]                   # (30, 3), (30,)
        nacc = c0.sum(axis=1)
        fid = np.abs((c0[:, 0] + c0[:, 1] * OMEGA + c0[:, 2] * OMEGA**2) / nacc)
        e = (c0[:, 1] + c0[:, 2]).astype(float)
        ehat = e.sum() / nacc.sum()
        F = 1 - 1.5 * ehat
        var_pred = (1 - F) * (F + 0.5) / nacc            # per sequence
        # fidelity dispersion statistic: sum (x - xbar)^2 / sigma_i^2 (approx chi2_29)
        xbar = np.sum(fid / var_pred) / np.sum(1 / var_pred)
        Df = float(np.sum((fid - xbar) ** 2 / var_pred))
        # rejected count, binomial(SHOTS, r)
        rhat = r0.sum() / (NSEQ * SHOTS)
        Dr = float(np.sum((r0 - SHOTS * rhat) ** 2) / (SHOTS * rhat * (1 - rhat)))
        # logical-error count given N_acc, binomial(N_acc, e)
        De = float(np.sum((e - nacc * ehat) ** 2 / (nacc * ehat * (1 - ehat))))
        ratio = fid.std() / math.sqrt(np.mean(var_pred))
        rows.append((PROBS[pidx], ratio, Df, Dr, De))
        sum_df += Df; sum_dr += Dr; sum_de += De; dof += NSEQ - 1
    for p, ratio, Df, Dr, De in rows:
        print(f"    p={p:<7g} std/pred={ratio:.3f}  D_fid={Df:6.1f}  D_rej={Dr:6.1f}  "
              f"D_err={De:6.1f}  (each ~ chi2_29, mean 29)")
    ratios = np.array([r[1] for r in rows])
    print(f"    ratio: mean {ratios.mean():.3f} +- {ratios.std(ddof=1)/math.sqrt(len(ratios)):.3f}"
          f" (sd {ratios.std(ddof=1):.3f}); expected E[std_ddof0]/sigma = "
          f"{math.sqrt(2/30)*math.exp(math.lgamma(15)-math.lgamma(14.5)):.4f}")
    for name, s in (("fidelity", sum_df), ("rejected", sum_dr), ("log.error", sum_de)):
        lo = stats.chi2.cdf(s, dof)
        print(f"    pooled D_{name:9s} = {s:7.1f} on {dof} dof: P(chi2 <= D) = {lo:.4f}, "
              f"var ratio = {s/dof:.3f}")
    # correlation of the two error outcomes at m = 0: n1 vs n2 split
    return rows


# --------------------------------------------------------- fits (own code)
def wls_poly(m, y, sig, cols):
    """Weighted LS y = X theta, X built from cols(m). Returns theta, cov, chi2."""
    X = cols(m)
    w = 1 / sig
    Xw, yw = X * w[:, None], y * w
    theta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.inv(Xw.T @ Xw)
    chi2 = float(np.sum((yw - Xw @ theta) ** 2))
    return theta, cov, chi2


def logquad_cols(m):
    return np.column_stack([np.ones_like(m), m, m * m])


def quad_cols(m):
    return np.column_stack([np.ones_like(m), m, m * m])


def usable(fid_mean, nacc_mean):
    ray = np.sqrt(np.pi / (4 * np.maximum(nacc_mean, 1)))
    return np.isfinite(fid_mean) & (nacc_mean >= 1000) & (fid_mean >= 5 * ray)


def boot_stats(fid, nacc, sel, stat_fn):
    """Sequence bootstrap of stat_fn(mean_fid_over_seq) on depth subset sel."""
    out = []
    idx_all = np.arange(NSEQ)
    for _ in range(NBOOT):
        idx = RNG.choice(idx_all, NSEQ, replace=True)
        ym = fid[sel][:, idx].mean(axis=1)
        out.append(stat_fn(ym))
    return np.array(out)


def check_D():
    print("[D] sequence bootstrap vs diagonal-SEM errors (NBOOT=%d)" % NBOOT)
    cases = [("const", 0, [1, 2, 3, 4, 5]), ("const", 1, [1, 2, 3, 4, 5, 6]),
             ("unif", 1, [3, 4, 5, 6, 7]), ("unif", 2, [4, 5])]
    out_rows = []
    for kind, k, pidxs in cases:
        for pidx in pidxs:
            cnt, rej = load_counts(pidx, kind, k)
            fid, nacc = per_seq_fid(cnt)
            ym = fid.mean(axis=1)
            sem = fid.std(axis=1) / math.sqrt(NSEQ)
            sem = np.maximum(sem, 1.5 / (NSEQ * np.maximum(nacc.mean(axis=1), 1)))
            sel = usable(ym, nacc.mean(axis=1))
            m = DEPTHS[sel]
            if len(m) < 5:
                continue
            # (1) log-quadratic curvature on ln y, weights sem/y (as N1's d_log)
            sL = sem[sel] / ym[sel]

            def dlog(yy, m=m, sL=sL):
                th, _, _ = wls_poly(m, np.log(yy), sL, logquad_cols)
                return -th[2]          # N1 convention: ln y = lnb - c m - d m^2
            th, cov, _ = wls_poly(m, np.log(ym[sel]), sL, logquad_cols)
            d0, sd_diag = -th[2], math.sqrt(cov[2, 2])
            bs = boot_stats(fid, nacc, sel, dlog)
            sd_boot = bs.std(ddof=1)
            row = dict(policy=f"{kind}{k}", p=PROBS[pidx], n=len(m), d=d0,
                       z_diag=d0 / sd_diag, z_boot=d0 / sd_boot,
                       boot_frac_d_le_0=float(np.mean(bs <= 0)) if d0 > 0 else float(np.mean(bs >= 0)))
            # (2) model E0 (m > 0) quadratic coefficient b2: y = b0 - b1 m - b2 m^2
            s0 = sel & (DEPTHS > 0)
            m0 = DEPTHS[s0]
            if len(m0) >= 4:
                def b2fun(yy, m0=m0, s0=s0):
                    th, _, _ = wls_poly(m0, yy, sem[s0], quad_cols)
                    return -th[2]
                th, cov, c2 = wls_poly(m0, ym[s0], sem[s0], quad_cols)
                b2, sb2 = -th[2], math.sqrt(cov[2, 2])
                bs2 = boot_stats(fid, nacc, s0, b2fun)
                row.update(b2_over_p2=b2 / PROBS[pidx] ** 2,
                           sig_b2_diag_over_p2=sb2 / PROBS[pidx] ** 2,
                           sig_b2_boot_over_p2=bs2.std(ddof=1) / PROBS[pidx] ** 2,
                           E0_chi2=c2, E0_dof=len(m0) - 3)
            out_rows.append(row)
            print("    " + "  ".join(f"{kk}={vv:.4g}" if isinstance(vv, float) else f"{kk}={vv}"
                                     for kk, vv in row.items()))
    return out_rows


def check_D_offset():
    """unif1 m = 0 offset relative to the m > 0 single exponential (C5, C21)."""
    print("[D2] unif1 m=0 offset: log-linear fit to m>0, sequence bootstrap")
    for pidx in (4, 5, 6, 7):
        cnt, rej = load_counts(pidx, "unif", 1)
        fid, nacc = per_seq_fid(cnt)
        ym = fid.mean(axis=1)
        sem = fid.std(axis=1) / math.sqrt(NSEQ)
        sel = usable(ym, nacc.mean(axis=1))
        s0 = sel & (DEPTHS > 0)
        m0 = DEPTHS[s0]
        sL = sem[s0] / ym[s0]

        def off(yy_all):
            th, _, _ = wls_poly(m0, np.log(yy_all[1:][s0[1:]]), sL,
                                lambda mm: np.column_stack([np.ones_like(mm), mm]))
            return yy_all[0] - math.exp(th[0])
        allsel = np.ones(9, bool)
        o = off(ym)
        bs = boot_stats(fid, nacc, allsel, off)
        th, cov, c2 = wls_poly(m0, np.log(ym[s0]), sL,
                               lambda mm: np.column_stack([np.ones_like(mm), mm]))
        r_m0 = (2 / 3) * (1 - math.exp(th[1]))
        print(f"    p={PROBS[pidx]:g}: offset={o:.3e} ({o/PROBS[pidx]**2:.1f} p^2), "
              f"boot sd={bs.std(ddof=1):.2e}, z_boot={o/bs.std(ddof=1):.2f}, "
              f"logfit chi2(m>0)={c2:.1f}/{len(m0)-2}, r(m>0)={r_m0:.3e}, usable m={DEPTHS[sel].astype(int).tolist()}")


# ------------------------------------------------------------------ [C]
def check_C():
    print("[C] empirical correlation of per-sequence fidelity across depths vs Brownian model")
    for kind, k, pidx in (("const", 0, 4), ("const", 1, 4), ("unif", 1, 4),
                          ("const", 0, 3), ("const", 1, 3), ("unif", 2, 4)):
        cnt, rej = load_counts(pidx, kind, k)
        fid, nacc = per_seq_fid(cnt)
        sel = DEPTHS > 0
        Y = fid[sel]                                     # (8, 30)
        C = np.corrcoef(Y)
        ym = Y.mean(axis=1)
        var_tot = Y.var(axis=1)
        shot = np.clip((1 - ym) * (ym + 0.5), 0, None) / nacc[sel].mean(axis=1)
        seq_sd = np.sqrt(np.clip(var_tot - shot, 0, None))
        mm = DEPTHS[sel]
        lo, hi = np.minimum.outer(mm, mm), np.maximum.outer(mm, mm)
        Cb = np.outer(seq_sd, seq_sd) * np.sqrt(lo / hi)
        np.fill_diagonal(Cb, var_tot)
        Cb = Cb / np.sqrt(np.outer(var_tot, var_tot))
        iu = np.triu_indices(len(mm), 1)
        adj = [(i, i + 1) for i in range(len(mm) - 1)]
        emp_adj = np.array([C[i, j] for i, j in adj])
        bro_adj = np.array([Cb[i, j] for i, j in adj])
        # Fisher-z sd for a correlation from 30 samples: 1/sqrt(27)
        diff = np.arctanh(np.clip(C[iu], -0.999, 0.999)) - np.arctanh(np.clip(Cb[iu], -0.999, 0.999))
        print(f"    {kind}{k} p={PROBS[pidx]:g}: corr(m=2,m=22) emp {C[0,-1]:.2f} vs Brownian {Cb[0,-1]:.2f};"
              f" corr(m=20,m=22) emp {C[-2,-1]:.2f} vs {Cb[-2,-1]:.2f}; "
              f"mean adjacent emp {emp_adj.mean():.2f} vs {bro_adj.mean():.2f}; "
              f"rms Fisher-z diff {np.sqrt(np.mean(diff**2)):.2f} (noise ~{1/math.sqrt(27):.2f}); "
              f"shot fraction (m>0, median) {np.median(shot/var_tot):.2f}")


# ------------------------------------------------------------------ [E]
def check_E():
    print("[E] exact per-sequence binomial shot variance vs across-sequence variance (median over m>0)")
    for kind, k in (("unif", 1), ("unif", 2), ("const", 0), ("const", 1)):
        out = []
        for pidx in range(1, 8):
            cnt, rej = load_counts(pidx, kind, k)
            fid, nacc = per_seq_fid(cnt)
            e = (cnt[..., 1] + cnt[..., 2]) / np.maximum(nacc, 1)
            shot_i = (9 / 4) * e * (1 - e) / np.maximum(nacc, 1)  # per sequence, Re part
            tot = fid.var(axis=1)
            fr = shot_i.mean(axis=1)[1:] / np.where(tot[1:] > 0, tot[1:], np.nan)
            out.append(f"{PROBS[pidx]:g}:{np.nanmedian(fr):.2f}")
        print(f"    {kind}{k}: " + "  ".join(out))


def check_saturation():
    print("[F] saturation of const1 / const0 acceptance and const1 fidelity floor")
    rejs, fl, flp = [], [], []
    for pidx in range(1, 15):
        cnt, rej = load_counts(pidx, "const", 1)
        fid, nacc = per_seq_fid(cnt)
        rf = rej.mean(axis=1) / SHOTS
        for d in range(9):
            if abs(rf[d] - 80 / 81) < 2e-4:
                rejs.append(rf[d]); fl.append(fid[d].mean())
                flp.append(np.mean(np.sqrt(np.pi / (4 * nacc[d]))))
    rejs, fl, flp = map(np.array, (rejs, fl, flp))
    print(f"    const1 points within 2e-4 of 80/81: n={len(rejs)}, mean rejected {rejs.mean():.6f} "
          f"(80/81={80/81:.6f}), fidelity mean {fl.mean():.4f} +- {fl.std(ddof=1)/math.sqrt(len(fl)):.4f}, "
          f"Rayleigh mean {flp.mean():.4f}")


if __name__ == "__main__":
    check_A()
    check_B()
    check_E()
    check_C()
    check_D()
    check_D_offset()
    check_saturation()
