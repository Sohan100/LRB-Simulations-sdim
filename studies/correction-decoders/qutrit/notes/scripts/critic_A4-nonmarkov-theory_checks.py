#!/usr/bin/env python3
"""Critic checks for A4-nonmarkov-theory.md (independent re-derivations).

Sections (run one or several by name; default: all except 'brute12'):

  brute4    independent brute-force character sums (no transfer matrices,
            no round maps, no banding) for policies whose used rounds give
            at most 8 detectors; compared with A4's Prepared.evaluate.
  brute12   same for 12-detector cases (unif 1 at m = 2, unif 2 at m = 4);
            531441 dual vectors each, ~1 minute.
  first     exact first-order coefficients a1(m), c1(m) from single-fault
            enumeration over all non-zero fault vectors; per-fault breakdown
            of a1 in the final round; test of the formula
            c1 = 63.2 + 71.1 m + 41.0 n_used(m).
  second    a2(m), c2(m) re-extracted at p = 1e-6..4e-6 (A4 used
            2e-5..1e-4); sign of c2 at every depth; per-sequence unif-1
            curvature.
  mask      sawtooth / final-interval mechanism: a2 of unif 4 and const 1
            with final-round ancilla-only (measurement-type) faults removed.
  data      checks on the baseline counts: const 0 (1-F)/m^2, const 1
            saturation, const 4 paired rise, T5 with extrapolation error,
            modulus vs real-part estimator z-scores, low-count cells.

Reads the A4 DEM cache (A4_SCRATCH/A4_dem_cache) and the read-only baseline
run folder.  Writes nothing.  One core, a few minutes in total.

    module load python/3.11
    export OMP_NUM_THREADS=1 A4_SCRATCH=<dir holding A4_dem_cache, A4_exact.npz>
    python critic_A4-nonmarkov-theory_checks.py first second mask data brute4
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import A4_exact_lrbd as X  # noqa: E402  (only Prepared/used_rounds/load_dem)

SCR = os.environ.get("A4_SCRATCH", "/tmp")
CACHE = os.path.join(SCR, "A4_dem_cache")
NPZ = os.path.join(SCR, "A4_exact.npz")
OMEGA = np.exp(2j * np.pi / 3)
SEQS = [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------------------------
# helpers independent of A4's transfer-matrix code
# --------------------------------------------------------------------------
def fault_blocks(dem):
    """List of (row indices, scale, kappa) per fault (owner is sorted)."""
    owner = dem["owner"]
    out = []
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    ends = np.r_[starts[1:], len(owner)]
    for f, (a, b) in enumerate(zip(starts, ends)):
        assert owner[a] == f
        out.append((np.arange(a, b), float(dem["scale"][f]),
                    float(dem["kappa"][f])))
    return out


def det_cols(rounds, block=4):
    return [block * r + i for r in rounds for i in range(block)]


def brute_Z(dem, cols, p_list, chunk=6561):
    """Z_w = 3^-|U| sum_u prod_f phi_f(A_f^T u + w l_f), w = 0, 1."""
    rows = dem["rows"].astype(np.int64)
    ndet = int(dem["ndet"])
    A = rows[:, cols]                       # (C, nU)
    Lr = rows[:, ndet]                      # (C,)
    owner = dem["owner"]
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    nf = len(starts)
    lf = np.array([[np.log1p(-k * s * p) for s, k in
                    zip(dem["scale"], dem["kappa"])] for p in p_list])  # (P,F)
    nU = len(cols)
    U = np.array(list(itertools.product(range(3), repeat=nU)), dtype=np.int64)
    Z = np.zeros((2, len(p_list)))
    for a in range(0, len(U), chunk):
        u = U[a:a + chunk]
        base = (u @ A.T) % 3                  # (n, C)
        for w in (0, 1):
            val = (base + w * Lr[None, :]) % 3
            nz = np.logical_or.reduceat(val != 0, starts, axis=1)  # (n, F)
            Z[w] += np.exp(nz.astype(float) @ lf.T).sum(0)
    return Z / 3 ** nU, nf


def nonzero_vectors(k):
    return np.array([v for v in itertools.product(range(3), repeat=k)
                     if any(v)], dtype=np.int64)


VEC = {k: nonzero_vectors(k) for k in (1, 2, 4)}


def first_order(dem, cols):
    """Exact c1 (acceptance) and a1 (accepted -ln F) coefficients.

    For every fault f (k components, scale s) and every non-zero fault
    vector e: c1 += s P(e) 1{D_U(e) != 0};  a1 += s P(e) (3/2) 1{D_U(e) = 0,
    L(e) != 0}, with P(e) = 1/(3^k - 1).  Vectorised per k.
    """
    rows = dem["rows"].astype(np.int64)
    ndet = int(dem["ndet"])
    owner = dem["owner"]
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    ends = np.r_[starts[1:], len(owner)]
    ncomp = ends - starts
    sub = rows[:, list(cols) + [ndet]]                 # (C, nU+1)
    per_fault = np.zeros((len(starts), 2))
    for k in (1, 2, 4):
        fsel = np.flatnonzero(ncomp == k)
        if len(fsel) == 0:
            continue
        R = np.stack([sub[starts[fsel] + c] for c in range(k)], 1)  # (f,k,c)
        E = VEC[k]
        V = np.einsum("ek,fkc->fec", E, R) % 3       # (f, nvec, nU+1)
        det = np.any(V[..., :-1] != 0, axis=2)
        L = V[..., -1]
        s = dem["scale"][fsel]
        per_fault[fsel, 0] = s * det.sum(1) / len(E)
        per_fault[fsel, 1] = s * 1.5 * ((~det) & (L != 0)).sum(1) / len(E)
    return per_fault[:, 0].sum(), per_fault[:, 1].sum(), per_fault


def subset(dem, keep):
    rows_keep = keep[dem["owner"]]
    new_index = -np.ones(len(keep), dtype=np.int64)
    new_index[keep] = np.arange(keep.sum())
    out = dict(dem)
    out["rows"] = dem["rows"][rows_keep]
    out["owner"] = new_index[dem["owner"][rows_keep]].astype(np.int32)
    for k in ("scale", "kappa", "opidx", "qudit", "target"):
        out[k] = dem[k][keep]
    return out


def coeffs(prep, ps, which):
    """Fit -ln Y(p) = c1 p + c2 p^2 + c3 p^3 (Y = F or A)."""
    vals = []
    for p in ps:
        A, F = prep.evaluate(p)
        vals.append(-np.log(F if which == "F" else A))
    Q = np.stack([ps, ps ** 2, ps ** 3], 1)
    c, *_ = np.linalg.lstsq(Q, np.array(vals), rcond=None)
    return c


# --------------------------------------------------------------------------
def sec_brute(max_det):
    print(f"\n[brute{max_det}] brute-force character sums vs A4 Prepared")
    p_list = [1e-3, 5e-3, 2e-2]
    cases = []
    for di, m in enumerate(X.DEPTHS):
        cases.append((0, di, m, ("const", 1)))
    if max_det >= 8:
        cases += [(0, 2, 4, ("const", 2)), (1, 3, 6, ("const", 2)),
                  (2, 1, 2, ("unif", 2)), (3, 2, 4, ("const", 4))]
    if max_det >= 12:
        cases = [(0, 1, 2, ("unif", 1)), (4, 2, 4, ("unif", 2)),
                 (5, 3, 6, ("const", 3))]
    worst = 0.0
    for seq, di, m, pol in cases:
        dem = X.load_dem("LRB", seq, di, CACHE)
        rnds = X.used_rounds(pol, m)
        cols = det_cols(rnds)
        if len(cols) > max_det:
            continue
        Z, nf = brute_Z(dem, cols, p_list)
        prep = X.Prepared(dem, rnds)
        for k, p in enumerate(p_list):
            A, F = prep.evaluate(p)
            dA = abs(Z[0, k] - A) / A
            dF = abs(Z[1, k] / Z[0, k] - F) / max(abs(F), 1e-300)
            worst = max(worst, dA, dF)
        print(f"  seq {seq} m={m:2d} {pol[0]}{pol[1]} rounds {rnds}: "
              f"faults {nf}; p=5e-3 brute A={Z[0, 1]:.10f} F="
              f"{Z[1, 1] / Z[0, 1]:.10f} | A4 A={prep.evaluate(5e-3)[0]:.10f}"
              f" F={prep.evaluate(5e-3)[1]:.10f}")
    print(f"  max relative deviation over cases and p in {p_list}: "
          f"{worst:.1e}")


def sec_first():
    print("\n[first] exact first-order coefficients from single faults")
    pols = [("unif", 1), ("unif", 2), ("unif", 3), ("unif", 4),
            ("unif", 6), ("const", 1), ("const", 2), ("const", 3),
            ("const", 4)]
    a1_all, c1_tab = [], {pol: np.zeros((len(SEQS), len(X.DEPTHS)))
                          for pol in pols}
    for si, seq in enumerate(SEQS):
        for di, m in enumerate(X.DEPTHS):
            dem = X.load_dem("LRB", seq, di, CACHE)
            for pol in pols:
                c1, a1, _ = first_order(dem, det_cols(X.used_rounds(pol, m)))
                c1_tab[pol][si, di] = c1
                a1_all.append(a1)
    a1_all = np.array(a1_all)
    print(f"  a1 over all (seq, m, policy): min {a1_all.min():.6f} "
          f"max {a1_all.max():.6f}")
    # per-fault breakdown of a1 in the final round, seq 0, m = 2
    dem = X.load_dem("LRB", 0, 1, CACHE)
    c1, a1, pf = first_order(dem, det_cols(X.used_rounds(("unif", 1), 2)))
    nz = np.flatnonzero(pf[:, 1] > 0)
    print(f"  seq 0, m=2, unif1: a1 = {a1:.4f}; contributing faults:")
    for f in nz:
        print(f"    op {dem['opidx'][f]:5d} qudit {dem['qudit'][f]} target "
              f"{dem['target'][f]:2d} scale {dem['scale'][f]:.1f} kappa "
              f"{dem['kappa'][f]:.4f}: a1 part {pf[f, 1]:.4f} "
              f"(= 1.5 x {pf[f, 1] / 1.5:.4f})")
    # const 0
    a10 = []
    for seq in SEQS:
        for di, m in enumerate(X.DEPTHS):
            dem0 = X.load_dem("LRB_const0", seq, di, CACHE)
            if len(dem0["scale"]) == 0:
                continue
            c1, a1, _ = first_order(dem0, list(range(int(dem0["ndet"]))))
            a10.append(a1)
    print(f"  const0: max a1 over (seq, m) = {max(a10):.2e}")
    # acceptance formula check
    ms = np.array(X.DEPTHS, dtype=float)
    print("  c1(m) mean over 6 seqs vs formula 63.2 + 71.1 m + 41.0 n_used:")
    for pol in pols:
        c1m = c1_tab[pol].mean(0)
        n_used = np.array([len(X.used_rounds(pol, int(m))) - 1 for m in ms])
        form = 63.2 + 71.1 * ms + 41.0 * n_used
        sel = ms >= 2
        slope = np.polyfit(ms[sel], c1m[sel], 1)
        print(f"   {pol[0]}{pol[1]:<2d} c1 = {np.round(c1m, 2)}; formula-"
              f"exact = {np.round(form - c1m, 2)}; slope {slope[0]:.3f} "
              f"intercept {slope[1]:.2f}")


def sec_second():
    print("\n[second] a2, c2 re-extracted at p = 1e-6..4e-6")
    ps = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    z = np.load(NPZ)
    pols = [str(s) for s in z["policies"]]
    psm = z["p_small"]
    for pol, pname in ((("unif", 1), "unif1"), (("const", 1), "const1"),
                       (("unif", 4), "unif4")):
        pi = pols.index(pname)
        for di in (1, 8):
            m = X.DEPTHS[di]
            a2n, c2n, a2o, c2o = [], [], [], []
            for si, seq in enumerate(SEQS[:3]):
                dem = X.load_dem("LRB", seq, di, CACHE)
                prep = X.Prepared(dem, X.used_rounds(pol, m))
                a2n.append(coeffs(prep, ps, "F")[1])
                c2n.append(coeffs(prep, ps, "A")[1])
                Q = np.stack([psm, psm ** 2, psm ** 3], 1)
                a2o.append(np.linalg.lstsq(Q, -np.log(z["Fs"][pi, si, di]),
                                           rcond=None)[0][1])
                c2o.append(np.linalg.lstsq(Q, -np.log(z["As"][pi, si, di]),
                                           rcond=None)[0][1])
            print(f"  {pname} m={m:2d}: a2 (small p) {np.round(a2n, 1)} vs "
                  f"A4 {np.round(a2o, 1)}; c2 {np.round(c2n, 1)} vs A4 "
                  f"{np.round(c2o, 1)}")
    # sign of c2 at every depth, every policy (A4's npz, mean over seqs)
    Q = np.stack([psm, psm ** 2, psm ** 3], 1)
    print("  c2(m) (A4 small-p data, mean over 6 seqs):")
    for pi, pname in enumerate(pols):
        c2 = np.array([[np.linalg.lstsq(Q, -np.log(z["As"][pi, s, d]),
                                        rcond=None)[0][1]
                        for d in range(len(X.DEPTHS))] for s in range(6)])
        print(f"   {pname:7s} {np.round(c2.mean(0), 1)}")
    # per-sequence unif 1 a2 fit and curvature of the exact curve at 1e-3
    pi = pols.index("unif1")
    ms = np.array(X.DEPTHS, dtype=float)
    sel = ms >= 2
    gam = []
    for s in range(6):
        a2 = np.array([np.linalg.lstsq(Q, -np.log(z["Fs"][pi, s, d]),
                                       rcond=None)[0][1]
                       for d in range(len(X.DEPTHS))])
        g = np.polyfit(ms[sel], a2[sel], 2)
        gam.append(g[0])
    k = list(z["p_eval"]).index(1e-3)
    dcurv = []
    for s in range(6):
        g = np.polyfit(ms[sel], np.log(z["F"][pi, s, sel, k]), 2)
        dcurv.append(-g[0])
    print(f"  unif1 per-sequence gamma (m^2 coeff of a2): "
          f"{np.round(gam, 3)}; mean {np.mean(gam):.3f}, sd "
          f"{np.std(gam, ddof=1):.3f}")
    print(f"  unif1 per-sequence exact curvature d at 1e-3: "
          f"{np.array2string(np.array(dcurv), precision=2)}; mean "
          f"{np.mean(dcurv):.2e}, sd {np.std(dcurv, ddof=1):.2e}")


def sec_mask():
    print("\n[mask] final-round measurement-type faults and the a2 terms")
    ps = np.array([2e-6, 4e-6, 6e-6, 8e-6])
    for pol, dis in ((("unif", 4), (6, 7, 8)), (("const", 1), (1, 4, 8)),
                     (("unif", 1), (1, 4, 8))):
        print(f"  policy {pol[0]}{pol[1]}")
        for di in dis:
            m = X.DEPTHS[di]
            full, nomeas, nofinal = [], [], []
            for seq in SEQS:
                dem = X.load_dem("LRB", seq, di, CACHE)
                rnds = X.used_rounds(pol, m)
                start_final = dem["starts"][m]
                fin = dem["opidx"] >= start_final
                anc_only = fin & (dem["qudit"] >= 5) & (dem["target"] < 0)
                full.append(coeffs(X.Prepared(dem, rnds), ps, "F")[1])
                nomeas.append(coeffs(X.Prepared(subset(dem, ~anc_only),
                                                rnds), ps, "F")[1])
                nofinal.append(coeffs(X.Prepared(subset(dem, ~fin), rnds),
                                      ps, "F")[1])
            print(f"   m={m:2d} r=m mod k: a2 full {np.mean(full):8.1f} | "
                  f"without final-round ancilla-only faults "
                  f"{np.mean(nomeas):8.1f} | without any final-round fault "
                  f"{np.mean(nofinal):8.1f}")


def sec_attr():
    """A4_attribution.py regions, but per sequence and for all 6 sequences."""
    print("\n[attr] unif1 a2 attribution per sequence (A4 used seqs 0-2)")
    ps = np.array([2e-6, 4e-6, 6e-6, 8e-6])
    ms = np.array([m for m in X.DEPTHS if m > 0], dtype=float)
    for seq in SEQS:
        tab = []
        for di, m in enumerate(X.DEPTHS):
            if m == 0:
                continue
            dem = X.load_dem("LRB", seq, di, CACHE)
            rnds = X.used_rounds(("unif", 1), m)
            op = dem["opidx"]
            start_final = dem["starts"][m]
            reset_f = (dem["qudit"] >= 5) & (dem["scale"] == 2.0) & \
                np.isclose(dem["kappa"], 1.5) & (op < start_final)
            end_fwd = op[reset_f].max()
            fwd = op <= end_fwd
            inv = (op > end_fwd) & (op < start_final)
            v = [coeffs(X.Prepared(dem, rnds), ps, "F")[1],
                 coeffs(X.Prepared(subset(dem, ~inv), rnds), ps, "F")[1],
                 coeffs(X.Prepared(subset(dem, ~fwd), rnds), ps, "F")[1]]
            tab.append(v)
        tab = np.array(tab)
        g = [np.polyfit(ms, tab[:, j], 2) for j in range(3)]
        print(f"  seq {seq}: m^2 coeff full {g[0][0]:+.3f} | no-inverse "
              f"{g[1][0]:+.3f} | no-forward (inverse+final) {g[2][0]:+.3f};"
              f" linear coeff full {g[0][1]:.2f}")


def load_counts(p_idx, kind, val):
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    base = f"{X.RUN}/progress/{p_idx}/{sub}/{val}"
    acc = np.load(base + ".npy").astype(float)
    rej = np.load(base + "_rejected.npy").astype(float)
    return acc, rej


def sec_data():
    print("\n[data] checks on the baseline per-sequence counts")
    ms = np.array(X.DEPTHS, dtype=float)
    # const 0 (1 - F)/m^2 at p = 1e-3
    acc, rej = load_counts(X.PROBS.index(1e-3), "const", 0)
    n = acc.sum(-1)
    Fre = (acc[..., 0] - 0.5 * (acc[..., 1] + acc[..., 2])) / n   # (D, S)
    Fm = Fre.mean(1)
    se = Fre.std(1, ddof=1) / np.sqrt(Fre.shape[1])
    print("  const0 p=1e-3 (1-F)/m^2 [real-part estimator], m=2..22: "
          + ", ".join(f"{(1 - Fm[i]) / ms[i] ** 2:.2e}+-{se[i] / ms[i] ** 2:.1e}"
                      for i in range(1, 9)))
    # const 1 saturation
    for p in (1e-2, 2e-2, 3e-2, 5e-2):
        acc, rej = load_counts(X.PROBS.index(p), "const", 1)
        n = acc.sum(-1)
        A = n / (n + rej)
        Fre = (acc[..., 0] - 0.5 * (acc[..., 1] + acc[..., 2])) / n
        Fab = np.abs(acc[..., 0] + OMEGA * acc[..., 1]
                     + OMEGA ** 2 * acc[..., 2]) / n
        print(f"  const1 p={p}: A(22) mean {A[-1].mean():.5f} (1/81 = "
              f"{1 / 81:.5f}); F_re(22) mean {Fre[-1].mean():+.2e} sd "
              f"{Fre[-1].std(ddof=1):.1e}; F_abs(22) mean {Fab[-1].mean():.2e}")
    # const 4 paired rise m=2 -> 4
    for p in (1e-3, 5e-3):
        acc, rej = load_counts(X.PROBS.index(p), "const", 4)
        n = acc.sum(-1)
        Fab = np.abs(acc[..., 0] + OMEGA * acc[..., 1]
                     + OMEGA ** 2 * acc[..., 2]) / n
        dlt = Fab[2] - Fab[1]
        print(f"  const4 p={p}: mean F(4)-F(2) = {dlt.mean():+.2e}, paired z "
              f"= {dlt.mean() / (dlt.std(ddof=1) / np.sqrt(len(dlt))):+.2f}")
    # T5 with the extrapolation uncertainty (cluster bootstrap)
    rng = np.random.default_rng(11)
    for p in (5e-4, 1e-3, 2e-3):
        acc, rej = load_counts(X.PROBS.index(p), "unif", 1)
        n = acc.sum(-1)
        F = (np.abs(acc[..., 0] + OMEGA * acc[..., 1]
                    + OMEGA ** 2 * acc[..., 2]) / n).T          # (S, D)
        S = F.shape[0]

        def resid0(Fs):
            y = Fs.mean(0)
            s = Fs.std(0, ddof=1) / np.sqrt(S)
            sel = ms >= 2
            w = (y[sel] / s[sel]) ** 2
            Q = np.stack([np.ones(sel.sum()), ms[sel]], 1)
            cb = np.linalg.solve(Q.T @ (Q * w[:, None]),
                                 (Q * w[:, None]).T @ np.log(y[sel]))
            return y[0] - np.exp(cb[0]), s[0]

        d0, s0 = resid0(F)
        boot = [resid0(F[rng.integers(0, S, S)])[0] for _ in range(2000)]
        sb = np.std(boot, ddof=1)
        print(f"  T5 unif1 p={p}: residual {d0:+.2e}; z with SE(m=0) only "
              f"{d0 / s0:+.2f}; z with cluster-bootstrap SE (fit refitted) "
              f"{d0 / sb:+.2f}")
    # modulus vs real-part estimator at low SNR, const 1
    z = np.load(NPZ)
    pols = [str(s) for s in z["policies"]]
    pi = pols.index("const1")
    for p in (1e-2, 2e-2):
        k = list(z["p_eval"]).index(p)
        acc, rej = load_counts(X.PROBS.index(p), "const", 1)
        acc = acc[:, SEQS, :]
        n = acc.sum(-1)
        Fp = z["F"][pi, :, :, k].T                      # (D, S)
        q = (2.0 / 3.0) * (1 - Fp)
        var = (1 - q) + q / 4 - Fp ** 2
        se = np.sqrt(var / n)
        Fab = np.abs(acc[..., 0] + OMEGA * acc[..., 1]
                     + OMEGA ** 2 * acc[..., 2]) / n
        Fre = (acc[..., 0] - 0.5 * (acc[..., 1] + acc[..., 2])) / n
        zab, zre = (Fab - Fp) / se, (Fre - Fp) / se
        low = Fp < 5 * se
        print(f"  const1 p={p}: n={zab.size}; modulus z mean {zab.mean():+.2f}"
              f" (F<5SE cells: {zab[low].mean():+.2f}, n={low.sum()}); "
              f"real-part z mean {zre.mean():+.2f} sd {zre.std():.2f}")
    # largest acceptance |z| at high p in low-count cells
    p = 2e-2
    k = list(z["p_eval"]).index(p)
    worst = (0, None)
    for pi, pname in enumerate(pols):
        kind, val = (pname[:-1], int(pname[-1]))
        acc, rej = load_counts(X.PROBS.index(p), kind, val)
        n = acc[:, SEQS, :].sum(-1)
        tot = n + rej[:, SEQS]
        Ap = z["A"][pi, :, :, k].T
        zA = (n / tot - Ap) / np.sqrt(Ap * (1 - Ap) / tot)
        i = np.unravel_index(np.nanargmax(np.abs(zA)), zA.shape)
        if abs(zA[i]) > abs(worst[0]):
            worst = (zA[i], (pname, X.DEPTHS[i[0]], SEQS[i[1]], n[i],
                             Ap[i] * tot[i]))
    print(f"  p=2e-2 largest |z_A| = {worst[0]:+.1f} at (policy, m, seq, "
          f"observed accepted, expected accepted) = {worst[1]}")


def main():
    secs = sys.argv[1:] or ["first", "second", "mask", "data", "brute4"]
    for s in secs:
        {"brute4": lambda: sec_brute(8), "brute12": lambda: sec_brute(12),
         "first": sec_first, "second": sec_second, "mask": sec_mask,
         "data": sec_data, "attr": sec_attr}[s]()


if __name__ == "__main__":
    main()
