#!/usr/bin/env python3
"""A4 revision: checks requested by the adversarial critique of
A4-nonmarkov-theory.md, run on the exact values of ALL 30 Clifford sequences.

Inputs
  * exact acceptance / accepted fidelity from A4_exact_lrbd.py, run in pieces,
        A4_exact_lrbd.py --seqs 0 1 2 3 4 5           --out $A4_SCRATCH/A4_exact.npz
        A4_exact_lrbd.py --seqs 6 ... 17              --out $A4_SCRATCH/A4_exact_s06-17.npz
        A4_exact_lrbd.py --seqs 18 ... 29             --out $A4_SCRATCH/A4_exact_s18-29.npz
    (about 6 minutes per 12 sequences on one core); section `merge` joins
    them into $A4_SCRATCH/A4_exact_all30.npz;
  * per-sequence outcome counts of the read-only baseline run,
        progress/<p_idx>/{unif,const}_check_data/<k>{,_rejected}.npy.

Sections (default: all, in this order)
  merge   join the three exact files (sequence axis) -> A4_exact_all30.npz
  valid   C7: exact vs baseline z-scores, 30 sequences, cells with >= 1000
          accepted shots; unif 1 alone (one cell per circuit, independent)
  coeff   C8-C12: small-p coefficients a1, a2, c1, c2 (mean over 30
          sequences); a2 fits; per-sequence unif-1 m^2 coefficients
          (gate-dependence spread); c1 line intercepts and the round-0 rule
  rise    C14: const 4, paired F(4) - F(2), data vs exact for the SAME 30
          sequences, both estimators
  t5      C15: m = 0 residual against b f^m fitted on m >= 2, with the
          extrapolation error (cluster bootstrap that refits each replicate)
          and the exact expectation for the same 30 sequences
  sat     C16: const 1 acceptance at m = 22 and exact windowed decay rates
          -d ln F/dm at p = 1e-2, 2e-2 (crossover and overshoot)
  rice    C17: modulus vs real-part estimator, z against exact (const 1,
          p = 1e-2, 2e-2, 30 sequences)
  power   C18: unif 1 curvature d of ln F = a - c m - d m^2 with ONE
          estimator (weighted log-quadratic, weights from the data SEM) for
          data, bootstrap and exact; data factor for a 3 sigma detection;
          per-sequence exact d (gate-dependence spread)
  eta     C19/C20: detection vs hybrid (intermediate rounds postselected,
          final round ML-decoded) for every policy: rate-drift index eta of
          the 30-sequence mean curve with jackknife SE over sequences, and
          absolute shape measures (max relative misfit of the best single
          exponential on m >= 2, curvature d, per-step rate)

Estimators: real part (n0 - (n1 + n2)/2)/n_acc unless stated (identical to
the modulus to all printed digits wherever F >> its SE).

    module load python/3.11
    export OMP_NUM_THREADS=1 A4_SCRATCH=<dir>
    python A4_revision_checks.py [sections...]          (~1 minute)
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import A4_exact_lrbd as X  # noqa: E402

SCR = os.environ.get("A4_SCRATCH", "/tmp")
ALL = os.path.join(SCR, "A4_exact_all30.npz")
PIECES = [os.path.join(SCR, f) for f in
          ("A4_exact.npz", "A4_exact_s06-17.npz", "A4_exact_s18-29.npz")]
DEPTHS = np.array(X.DEPTHS)
MS = DEPTHS.astype(float)
SEL = DEPTHS >= 2
W = np.exp(2j * np.pi / 3)
POLS = [f"{a}{b}" for a, b in X.POLICIES]          # 9 policies (no const0)
RNG = np.random.default_rng(20261028)
NBOOT = 2000


# ---------------------------------------------------------------------------
def load_all():
    return np.load(ALL)


def counts(p, kind, val):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    base = f"{X.RUN}/progress/{X.PROBS.index(p)}/{sub}/{val}"
    acc = np.load(base + ".npy").astype(float)          # (D, S, 3)
    rej = np.load(base + "_rejected.npy").astype(float)
    return acc, rej


def estimators(acc, rej):
    """Per-sequence (S, D) arrays: modulus F, real-part F, acceptance, n_acc."""
    n = acc.sum(-1)
    z = acc[..., 0] + W * acc[..., 1] + W ** 2 * acc[..., 2]
    nn = np.maximum(n, 1)
    Fabs = np.abs(z) / nn
    Fre = (acc[..., 0] - 0.5 * (acc[..., 1] + acc[..., 2])) / nn
    A = n / (n + rej)
    return Fabs.T, Fre.T, A.T, n.T


def var_re(F):
    """Shot variance x n_acc of the real-part estimator for symmetric
    outcomes P(1) = P(2) = (1 - F)/3."""
    return (1 + 2 * F) / 3 + (1 - F) / 6 - F ** 2


def wls_poly(y, s, deg, sel=SEL):
    """Weighted LS of ln y on (1, m, ..., m^deg) over the selected depths,
    weights (y/s)^2 (delta method).  Returns coefficients (low order first)."""
    m = MS[sel]
    Q = np.stack([m ** k for k in range(deg + 1)], 1)
    wts = (y[sel] / s[sel]) ** 2
    WQ = Q * wts[:, None]
    return np.linalg.solve(Q.T @ WQ, WQ.T @ np.log(y[sel]))


def shape(F):
    """Shape measures of one curve F(m) (m >= 2, unweighted in ln F):
    eta = ln f(20->22)/ln f(2->4) - 1, max relative misfit of the best
    single exponential, curvature d, per-step 1 - f, rises anywhere."""
    lnF = np.log(F[SEL])
    Q1 = np.stack([np.ones(SEL.sum()), MS[SEL]], 1)
    b1, *_ = np.linalg.lstsq(Q1, lnF, rcond=None)
    dev = np.max(np.abs(np.exp(Q1 @ b1) / F[SEL] - 1))
    Q2 = np.stack([np.ones(SEL.sum()), MS[SEL], MS[SEL] ** 2], 1)
    b2, *_ = np.linalg.lstsq(Q2, lnF, rcond=None)
    r24 = (F[2] / F[1]) ** 0.5
    r2022 = (F[8] / F[7]) ** 0.5
    eta = np.log(r2022) / np.log(r24) - 1
    return eta, dev, -b2[2], 1 - np.exp(b1[1]), bool(np.any(np.diff(F[SEL]) > 0))


def jackknife(fun, arr):
    """Jackknife SE over the first axis of arr for a statistic of the mean."""
    n = arr.shape[0]
    vals = np.array([fun(np.delete(arr, i, 0).mean(0)) for i in range(n)])
    return np.sqrt((n - 1) / n * np.sum((vals - vals.mean(0)) ** 2, axis=0))


# ---------------------------------------------------------------------------
def sec_merge():
    print("\n== merge")
    parts = [np.load(f) for f in PIECES]
    out = {}
    seq_axis = {"A": 1, "F": 1, "Adec": 1, "Fdec": 1, "As": 1, "Fs": 1,
                "A0": 0, "F0": 0, "F0dec": 0, "A0s": 0, "F0s": 0, "seqs": 0}
    for k in parts[0].files:
        if k in seq_axis:
            out[k] = np.concatenate([z[k] for z in parts], axis=seq_axis[k])
        else:
            out[k] = parts[0][k]
            for z in parts[1:]:
                assert np.array_equal(z[k], out[k]), k
    assert list(out["seqs"]) == list(range(30)), out["seqs"]
    np.savez(ALL, **out)
    print(f"  wrote {ALL}: sequences {out['seqs'].min()}..{out['seqs'].max()}"
          f", F shape {out['F'].shape}")


def sec_valid():
    print("\n== valid (C7): exact vs baseline counts, all 30 sequences")
    z = load_all()
    pe = list(z["p_eval"])
    seqs = list(z["seqs"])
    for p in (1e-3, 5e-3, 1e-2, 2e-2):
        k = pe.index(p)
        zA_all, zF_all, zFr_all = [], [], []
        for pi, (kind, val) in enumerate(X.POLICIES + [("const", 0)]):
            acc, rej = counts(p, kind, val)
            Fa, Fr, A, n = estimators(acc, rej)
            if kind == "const" and val == 0:
                Ap, Fp = z["A0"][:, :, k], z["F0"][:, :, k]
            else:
                Ap, Fp = z["A"][pi][:, :, k], z["F"][pi][:, :, k]
            Fa, Fr, A, n = Fa[seqs], Fr[seqs], A[seqs], n[seqs]
            tot = n / np.maximum(A, 1e-300)
            ok = (Ap * tot >= 1000) & (n >= 1000) & (Fp < 1)
            zA = (A - Ap) / np.sqrt(Ap * (1 - Ap) / tot)
            se = np.sqrt(var_re(Fp) / np.maximum(n, 1))
            zA_all.append(zA[ok]), zF_all.append(((Fa - Fp) / se)[ok])
            zFr_all.append(((Fr - Fp) / se)[ok])
        zA_all, zF_all, zFr_all = map(np.concatenate,
                                      (zA_all, zF_all, zFr_all))
        print(f"  p={p:.0e}: {zA_all.size} cells: acceptance z mean "
              f"{zA_all.mean():+.3f} sd {zA_all.std():.3f}; fidelity z "
              f"(modulus) mean {zF_all.mean():+.3f} sd {zF_all.std():.3f}; "
              f"(real part) mean {zFr_all.mean():+.3f} sd {zFr_all.std():.3f}")
    # unif 1 only: one cell per circuit, independent shots
    for p in (1e-3, 2e-3, 5e-3):
        k = pe.index(p)
        acc, rej = counts(p, "unif", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        Ap, Fp = z["A"][0][:, :, k], z["F"][0][:, :, k]
        tot = n / A
        ok = (n >= 1000)[:, 1:]
        zA = ((A - Ap) / np.sqrt(Ap * (1 - Ap) / tot))[:, 1:][ok]
        zF = ((Fr - Fp) / np.sqrt(var_re(Fp) / n))[:, 1:][ok]
        rel = ((A - Ap) / Ap)[:, 1:][ok]
        print(f"  unif 1 only, p={p:.0e}, m >= 2, {zA.size} independent "
              f"circuits: acceptance z mean {zA.mean():+.3f} sd {zA.std():.3f}"
              f" (mean relative deviation {rel.mean():+.1e}); fidelity z mean"
              f" {zF.mean():+.3f} sd {zF.std():.3f} (SE of a mean of "
              f"{zA.size} unit z's = {1 / np.sqrt(zA.size):.3f})")


def small_p(arr, p_small):
    """Cubic fit (no constant) of -ln arr in p, per (seq, depth)."""
    Q = np.stack([p_small, p_small ** 2, p_small ** 3], 1)
    y = -np.log(arr)                                   # (S, D, P)
    S, D, _ = y.shape
    c = np.linalg.lstsq(Q, y.reshape(S * D, -1).T, rcond=None)[0]
    return c.T.reshape(S, D, 3)


def sec_coeff():
    print("\n== coeff (C8-C12): small-p coefficients, mean over 30 sequences")
    z = load_all()
    ps = z["p_small"]
    Q3 = np.stack([np.ones(SEL.sum()), MS[SEL], MS[SEL] ** 2], 1)
    Q2 = Q3[:, :2]
    labs = POLS + ["const0"]
    res = {}
    for pi, lab in enumerate(labs):
        Fs = z["F0s"] if lab == "const0" else z["Fs"][pi]
        As = z["A0s"] if lab == "const0" else z["As"][pi]
        cF, cA = small_p(Fs, ps), small_p(As, ps)
        a1, a2 = cF[..., 0], cF[..., 1]                 # (S, D)
        c1, c2 = cA[..., 0], cA[..., 1]
        g, *_ = np.linalg.lstsq(Q3, a2.mean(0)[SEL], rcond=None)
        h, *_ = np.linalg.lstsq(Q2, c1.mean(0)[SEL], rcond=None)
        gc, *_ = np.linalg.lstsq(Q3, c2.mean(0)[SEL], rcond=None)
        per = np.array([np.linalg.lstsq(Q3, a2[s, SEL], rcond=None)[0]
                        for s in range(a2.shape[0])])   # (S, 3)
        res[lab] = dict(a2=a2, g=g, per=per)
        print(f"  {lab:7s} a1 range [{a1.min():.4f}, {a1.max():.4f}]; "
              f"a2 fit {g[0]:7.1f} + {g[1]:7.3f} m + {g[2]:7.4f} m^2; "
              f"per-sequence m^2 coeff mean {per[:, 2].mean():+.4f} sd "
              f"{per[:, 2].std(ddof=1):.4f} (SE {per[:, 2].std(ddof=1) / np.sqrt(len(per)):.4f}),"
              f" range [{per[:, 2].min():+.3f}, {per[:, 2].max():+.3f}]")
        print(f"          c1(0) = {c1.mean(0)[0]:.3f}; c1 line (m>=2) "
              f"{h[0]:.3f} + {h[1]:.4f} m; c2 fit {gc[0]:.1f} + {gc[1]:.2f} m"
              f" + {gc[2]:.3f} m^2; c2 range over m>=2 "
              f"[{c2.mean(0)[SEL].min():.1f}, {c2.mean(0)[SEL].max():.1f}]")
        if lab == "const0":
            print(f"          a2/m^2 (m>=2): "
                  f"{np.round(a2.mean(0)[SEL] / MS[SEL] ** 2, 3)}")
    # beta_k = a2 slope for unif k with free alpha + beta m (+ gamma m^2)
    print("  unif k slopes beta_k / k (quadratic fits above): " + ", ".join(
        f"k={k}: {res[f'unif{k}']['g'][1] / k:.2f}" for k in (1, 2, 3, 4, 6)))
    print("  const c curvature gamma_c * c: " + ", ".join(
        f"c={c}: {res[f'const{c}']['g'][2] * c:.2f}" for c in (1, 2, 3, 4)))
    # first-order acceptance rule:
    #   c1(policy, m) = c1(const 1, m) + 41.0 n_used(m) - 8 [round 0 used]
    c1_ref = small_p(z["As"][POLS.index("const1")], ps)[..., 0]      # (S, D)
    h, *_ = np.linalg.lstsq(Q2, c1_ref.mean(0)[SEL], rcond=None)
    print(f"  const 1 (no intermediate round): c1(0) = {c1_ref.mean(0)[0]:.3f};"
          f" c1 line on m >= 2: {h[0]:.3f} + {h[1]:.4f} m; per-sequence "
          f"slopes: mean {np.mean([np.polyfit(MS[SEL], c, 1)[0] for c in c1_ref[:, SEL]]):.3f},"
          f" range [{np.min([np.polyfit(MS[SEL], c, 1)[0] for c in c1_ref[:, SEL]]):.3f}, "
          f"{np.max([np.polyfit(MS[SEL], c, 1)[0] for c in c1_ref[:, SEL]]):.3f}]")
    print("  rule c1(policy) - c1(const 1) = 41.0 n_used(m) - 8 [round 0 "
          "used], every sequence and depth: max |violation|")
    for pi, lab in enumerate(POLS):
        c1 = small_p(z["As"][pi], ps)[..., 0]
        nu, r0 = [], []
        for m in DEPTHS:
            rr = X.used_rounds(X.POLICIES[pi], int(m))
            nu.append(len(rr) - 1)
            r0.append(1.0 if (0 in rr and m > 0) else 0.0)
        nu, r0 = np.array(nu), np.array(r0)
        viol = np.max(np.abs(c1 - c1_ref - (41.0 * nu - 8.0 * r0)[None, :]))
        print(f"     {lab:7s} n_used = {nu.tolist()}, round 0 used = "
              f"{r0.astype(int).tolist()}: max |violation| = {viol:.2e}")
    np.savez(os.path.join(SCR, "A4rev_coeff.npz"),
             **{f"{k}_a2": v["a2"] for k, v in res.items()},
             **{f"{k}_per": v["per"] for k, v in res.items()})


def sec_rise():
    print("\n== rise (C14): const 4, paired F(4) - F(2) over the 30 sequences")
    z = load_all()
    pe = list(z["p_eval"])
    pi = POLS.index("const4")
    for p in (5e-4, 1e-3, 2e-3, 5e-3):
        k = pe.index(p)
        ex = z["F"][pi][:, 2, k] - z["F"][pi][:, 1, k]
        acc, rej = counts(p, "const", 4)
        Fa, Fr, A, n = estimators(acc, rej)
        for lab, F in (("real", Fr), ("modulus", Fa)):
            dd = F[:, 2] - F[:, 1]
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            # shot-noise-only SE of the mean difference given the exact curve
            se_shot = np.sqrt(np.sum(var_re(z["F"][pi][:, 1, k]) / n[:, 1] +
                                     var_re(z["F"][pi][:, 2, k]) / n[:, 2])) / 30
            print(f"  p={p:.0e} {lab:7s}: data {dd.mean():+.3e} (SE {se:.1e},"
                  f" z = {dd.mean() / se:+.2f}); exact, same 30 seqs "
                  f"{ex.mean():+.3e} (6-seq {ex[:6].mean():+.2e}; per-seq "
                  f"[{ex.min():+.1e}, {ex.max():+.1e}]); (data - exact)/SE = "
                  f"{(dd.mean() - ex.mean()) / se:+.2f}; expected z = "
                  f"exact/SE = {ex.mean() / se:+.2f}; shot-only SE "
                  f"{se_shot:.1e}")


def m1_fit(y, s):
    return wls_poly(y, s, 1)


def sec_t5():
    print("\n== t5 (C15): m = 0 point vs b f^m fitted on m >= 2 (unif 1)")
    z = load_all()
    pe = list(z["p_eval"])
    for p in (5e-4, 1e-3, 2e-3):
        acc, rej = counts(p, "unif", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        F = Fr
        y, s = F.mean(0), F.std(0, ddof=1) / np.sqrt(30)
        b = m1_fit(y, s)
        res0 = y[0] - np.exp(b[0])
        boot = []
        for _ in range(NBOOT):
            idx = RNG.integers(0, 30, 30)
            yb = F[idx].mean(0)
            sb = F[idx].std(0, ddof=1) / np.sqrt(30)
            bb = m1_fit(yb, np.where(sb > 0, sb, s))
            boot.append(yb[0] - np.exp(bb[0]))
        sd = np.std(boot, ddof=1)
        Fe = z["F"][0][:, :, pe.index(p)].mean(0)       # exact, same 30 seqs
        be = m1_fit(Fe, s)
        rex = Fe[0] - np.exp(be[0])
        print(f"  p={p:.0e}: residual {res0:+.2e}; z with the m=0 SE only "
              f"{res0 / s[0]:+.2f}; z with the cluster bootstrap (refit per "
              f"replicate) {res0 / sd:+.2f}; exact expected residual (30 seqs)"
              f" {rex:+.2e} (expected z {rex / sd:+.2f}); (data - exact)/sd "
              f"= {(res0 - rex) / sd:+.2f}")


def sec_sat():
    print("\n== sat (C16): const 1 saturation and crossover (exact, 30 seqs)")
    z = load_all()
    pe = list(z["p_eval"])
    pi = POLS.index("const1")
    for p in (2e-3, 5e-3, 1e-2, 2e-2):
        k = pe.index(p)
        F = z["F"][pi][:, :, k].mean(0)
        A = z["A"][pi][:, :, k].mean(0)
        rate = -np.diff(np.log(F)) / np.diff(MS)
        print(f"  p={p:.0e}: exact A(22) = {A[-1]:.5f} (1/81 = {1 / 81:.5f});"
              f" -d lnF/dm per window (0-2, 2-4, ..., 20-22) = "
              f"{np.array2string(rate, precision=3)}")
    for p in (1e-2, 2e-2):
        acc, rej = counts(p, "const", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        print(f"  data p={p:.0e}: A(22) mean {A[:, 8].mean():.5f} (range "
              f"{A[:, 8].min():.5f}-{A[:, 8].max():.5f}); real-part F(22) "
              f"mean {Fr[:, 8].mean():+.4f} +- {Fr[:, 8].std(ddof=1) / np.sqrt(30):.4f}")


def sec_rice():
    print("\n== rice (C17): estimator z-scores against exact (const 1, 30 seqs)")
    z = load_all()
    pe = list(z["p_eval"])
    pi = POLS.index("const1")
    for p in (1e-2, 2e-2):
        k = pe.index(p)
        acc, rej = counts(p, "const", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        Fx = z["F"][pi][:, :, k]
        se = np.sqrt(var_re(Fx) / np.maximum(n, 1))
        za, zr = (Fa - Fx) / se, (Fr - Fx) / se
        low = Fx < 5 * se
        print(f"  p={p:.0e}: n = {za.size} circuits: modulus z mean "
              f"{za.mean():+.2f} (sd {za.std():.2f}); real-part z mean "
              f"{zr.mean():+.2f} (sd {zr.std():.2f}); {low.sum()} cells with "
              f"F < 5 SE: modulus {za[low].mean():+.2f}, real part "
              f"{zr[low].mean():+.2f}; modulus F(22) mean {Fa[:, 8].mean():.4f}"
              f" vs Rice floor sqrt(pi/(4 n_acc)) = "
              f"{np.mean(np.sqrt(np.pi / 4 / n[:, 8])):.4f}")


def sec_power():
    print("\n== power (C18): unif 1 curvature d in ln F = a - c m - d m^2")
    z = load_all()
    pe = list(z["p_eval"])
    for p in (5e-4, 1e-3, 2e-3):
        acc, rej = counts(p, "unif", 1)
        Fa, Fr, A, n = estimators(acc, rej)
        F = Fr
        y, s = F.mean(0), F.std(0, ddof=1) / np.sqrt(30)
        d0 = -wls_poly(y, s, 2)[2]
        db = []
        for _ in range(NBOOT):
            idx = RNG.integers(0, 30, 30)
            yb = F[idx].mean(0)
            sb = F[idx].std(0, ddof=1) / np.sqrt(30)
            db.append(-wls_poly(yb, np.where(sb > 0, sb, s), 2)[2])
        db = np.array(db)
        Fe = z["F"][0][:, :, pe.index(p)]                # (30, D)
        dex = -wls_poly(Fe.mean(0), s, 2)[2]
        dex_unw = shape(Fe.mean(0))[2]
        dper = np.array([-wls_poly(Fe[i], s, 2)[2] for i in range(30)])
        ratio = dex / db.std(ddof=1)
        print(f"  p={p:.0e}: data d = {d0:+.2e} (bootstrap SE {db.std(ddof=1):.1e},"
              f" 95% half-width {0.5 * np.diff(np.percentile(db, [2.5, 97.5]))[0]:.1e});"
              f" exact d of the 30-seq mean curve {dex:+.2e} (unweighted fit "
              f"{dex_unw:+.2e}); exact/SE = {ratio:+.2f}; data factor for 3 "
              f"sigma = {(3 / ratio) ** 2 if ratio != 0 else np.inf:.0f}x; "
              f"per-sequence exact d mean {dper.mean():+.2e}, sd "
              f"{dper.std(ddof=1):.1e}, range [{dper.min():+.1e}, "
              f"{dper.max():+.1e}]; A(22) = {A[:, 8].mean():.4f}")


def sec_eta():
    print("\n== eta (C19, C20): detection vs hybrid (final round ML-decoded),"
          " 30-sequence mean curves, jackknife SE over sequences")
    z = load_all()
    pe = list(z["p_eval"])
    labs = POLS + ["const0"]
    for p in (1e-3, 5e-3):
        k = pe.index(p)
        print(f"  p = {p:.0e}: policy | detection: eta +- SE, max rel misfit,"
              f" d, 1-f | hybrid: eta +- SE, max rel misfit, d, 1-f, "
              f"acceptance(22)")
        for pi, lab in enumerate(labs):
            if lab == "const0":
                Fd, Fh = z["F0"][:, :, k], z["F0dec"][:, :, k]
                Ah = np.ones(30)
            else:
                Fd, Fh = z["F"][pi][:, :, k], z["Fdec"][pi][:, :, k]
                Ah = z["Adec"][pi][:, 8, k]
            out = []
            for arr in (Fd, Fh):
                e, dev, d, r, up = shape(arr.mean(0))
                se = jackknife(lambda f: shape(f)[0], arr)
                out.append((e, se, dev, d, r, up))
            (e1, s1, v1, d1, r1, u1), (e2, s2, v2, d2, r2, u2) = out
            print(f"    {lab:7s} | {e1:+8.2f} +- {s1:5.2f} {v1:8.1e} "
                  f"{d1:+9.2e} {r1:8.1e} | {e2:+8.2f} +- {s2:5.2f} {v2:8.1e} "
                  f"{d2:+9.2e} {r2:8.1e}  {Ah.mean():.3f}")


SECTIONS = {"merge": sec_merge, "valid": sec_valid, "coeff": sec_coeff,
            "rise": sec_rise, "t5": sec_t5, "sat": sec_sat, "rice": sec_rice,
            "power": sec_power, "eta": sec_eta}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        SECTIONS[name]()
