#!/usr/bin/env python3
"""Adversarial critic of A4-nonmarkov-theory.md: exact-DEM checks.

Independent re-derivations.  A4's module A4_exact_lrbd is imported only
(i) for its DEM cache loader (which calls the repository DEM compiler) and
used_rounds(), and (ii) as the object under test (Prepared.evaluate).  The
evaluators below do not use A4's round maps, banding or transfer matrices.

Sections (pass names on the command line; default: all):
  extract  re-extract the fault rows directly from the repository DEM
           (lrb.correction.compact_dem.build_compact_detector_error_model) for a few
           circuits; compare with A4's cached rows; check that every
           location's probability equals p * prob_scale and that only the
           channels d/f/p and gates N1/N2 occur.
  brute    brute-force character sums Z_w = 3^-|U| sum_u prod_j phi_j(...)
           over all dual vectors u (no round maps, no banding, no transfer
           matrices) for const 0, const 1, const 2 (all depths) and for
           unif 1 at m = 2 and unif 2 at m = 4 (12 detectors, 531441 dual
           vectors); compare acceptance and fidelity with Prepared.evaluate.
  first    exact first-order coefficients by single-fault enumeration:
           a1 (accepted infidelity) and c1 (rejection) per policy/depth;
           contributors to a1 in the final round; per-round transient
           rate c1(unif1)-c1(const1) by fault category; c1 line intercepts.
  second   a2(m), c2(m) at p = 1e-6..4e-6 (brute force for const 0/1,
           A4 Prepared for unif 1): m^2 coefficient of c2 (log-convexity).
  hybrid   rate-drift index eta and absolute shape measures for ALL
           policies, detection vs hybrid, from A4_exact.npz.
  attr     A4's region attribution re-run on sequences 3, 4, 5 (unif 1).
  const4   compile seqs 0..29 at m = 2, 4; exact const 4 F(2), F(4) and
           unif 1 F(0) (all sequences); writes critic_A4_const4.npz.

Reads the read-only baseline run folder (circuits) and the A4 DEM cache in
$A4_SCRATCH/A4_dem_cache (created by A4_exact_lrbd.py); writes only into
$A4_SCRATCH.  One core; about 3 minutes for everything.

    module load python/3.11
    export OMP_NUM_THREADS=1 A4_SCRATCH=<dir>
    python critic_A4-nonmarkov-theory_exact.py [sections...]
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import A4_exact_lrbd as X  # noqa: E402

SCR = os.environ.get("A4_SCRATCH", "/tmp")
CACHE = os.path.join(SCR, "A4_dem_cache")
NPZ = os.path.join(SCR, "A4_exact.npz")
SEQS = [0, 1, 2, 3, 4, 5]
DEPTHS = X.DEPTHS
OMEGA = np.exp(2j * np.pi / 3)


# ---------------------------------------------------------------------------
# independent helpers
# ---------------------------------------------------------------------------
def fault_offsets(owner: np.ndarray) -> np.ndarray:
    off = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    assert np.all(owner[off] == np.arange(len(off))), "owner not sorted"
    return off


def type_table(dem):
    keys = sorted({(float(s), float(k)) for s, k in
                   zip(dem["scale"], dem["kappa"])})
    idx = {k: i for i, k in enumerate(keys)}
    ft = np.array([idx[(float(s), float(k))] for s, k in
                   zip(dem["scale"], dem["kappa"])], dtype=np.int64)
    return keys, ft


def brute_counts(dem, U, w, chunk=40000):
    """For every dual vector u in Z_3^|U|: per fault type, the number of
    faults whose character argument is non-zero.  Returns (counts, keys)."""
    rows = dem["rows"].astype(np.float32)
    ndet = int(dem["ndet"])
    owner = dem["owner"]
    keys, ft = type_table(dem)
    nt = len(keys)
    nf = len(dem["scale"])
    if nf == 0:
        return np.zeros((3 ** len(U), 0), dtype=np.int32), keys
    off = fault_offsets(owner)
    onehot = np.zeros((nf, nt), dtype=np.float32)
    onehot[np.arange(nf), ft] = 1
    RU = rows[:, U]                          # (C, |U|)
    Lw = (w * rows[:, ndet]) % 3            # (C,)
    allu = np.array(list(itertools.product(range(3), repeat=len(U))),
                    dtype=np.float32)
    out = np.zeros((len(allu), nt), dtype=np.int32)
    for a in range(0, len(allu), chunk):
        u = allu[a:a + chunk]
        val = (u @ RU.T + Lw[None, :]) % 3   # exact small integers
        nz = val != 0                        # (n_u, C)
        nzf = np.logical_or.reduceat(nz, off, axis=1)   # (n_u, nf)
        out[a:a + chunk] = np.rint(nzf.astype(np.float32) @ onehot)
    return out, keys


def brute_eval(dem, U, plist):
    """Exact acceptance and accepted fidelity by brute force."""
    res = []
    c0, keys = brute_counts(dem, U, 0)
    c1, _ = brute_counts(dem, U, 1)
    for p in plist:
        lf = np.array([np.log1p(-k * s * p) for s, k in keys])
        Z0 = np.exp(c0 @ lf).sum() / 3 ** len(U) if len(keys) else 1.0
        Z1 = np.exp(c1 @ lf).sum() / 3 ** len(U) if len(keys) else 1.0
        res.append((Z0, Z1 / Z0))
    return np.array(res)


def used_dets(rounds):
    return [4 * r + i for r in rounds for i in range(4)]


def single_fault_coeffs(dem, U):
    """Exact first-order coefficients by enumeration of every non-zero fault
    vector of every location: returns per-fault arrays
    (pdet, pflip) with pdet = P(D_U != 0 | fault), pflip = P(D_U = 0 and
    L != 0 | fault)."""
    rows = dem["rows"].astype(np.int64)
    ndet = int(dem["ndet"])
    owner = dem["owner"]
    off = fault_offsets(owner)
    ends = np.r_[off[1:], len(owner)]
    nf = len(off)
    pdet = np.zeros(nf)
    pflip = np.zeros(nf)
    evecs = {k: np.array([e for e in itertools.product(range(3), repeat=k)
                          if any(e)], dtype=np.int64) for k in (1, 2, 4)}
    for f in range(nf):
        R = rows[off[f]:ends[f]]
        k = R.shape[0]
        E = evecs[k]
        D = (E @ R[:, U]) % 3
        L = (E @ R[:, ndet]) % 3
        det = np.any(D != 0, axis=1)
        pdet[f] = det.mean()
        pflip[f] = np.mean((~det) & (L != 0))
    return pdet, pflip


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def sec_extract():
    print("\n== extract: independent row extraction from the repository DEM")
    sys.path.insert(0, os.path.join(X.WORKTREE, "src"))
    from lrb.correction.compact_dem import read_run_circuit as read_circuit
    from lrb.correction.compact_dem import build_compact_detector_error_model
    p_circ = X.PROBS[4]
    for kind, seq, di in (("LRB", 0, 1), ("LRB", 3, 8), ("LRB_const0", 1, 8)):
        path = f"{X.RUN}/experiments/{kind}/{seq}/4/{di}.chp"
        t0 = time.time()
        model = build_compact_detector_error_model(read_circuit(path))
        ndet = model.num_detectors
        rows, scales, chans, gates, bad_p, dup = [], [], set(), set(), 0, 0
        for loc in model.locations:
            chans.add(loc.channel)
            gates.add(loc.gate)
            s = float(loc.params.get("prob_scale", np.nan))
            scales.append(s)
            if not np.isclose(loc.probability, p_circ * s, rtol=1e-9):
                bad_p += 1
            if loc.gate == "N1":
                comps = {"d": [0, 1], "f": [0], "p": [1]}[loc.channel]
            else:
                comps = [0, 1, 2, 3]
                dist = np.asarray(loc.params["prob_dist"], float)
                assert np.allclose(dist[1:], dist[1]), "N2 not uniform"
            for c in comps:
                r = np.zeros(ndet + 1, dtype=np.int64)
                dets = [d for d, _ in loc.component_terms[c].detector]
                dup += len(dets) - len(set(dets))
                for d, v in loc.component_terms[c].detector:
                    r[d] += v
                for _, v in loc.component_terms[c].logical:
                    r[ndet] += v
                rows.append(r % 3)
        rows = np.array(rows)
        cached = X.load_dem(kind, seq, di, CACHE)
        same = rows.shape == cached["rows"].shape and \
            np.array_equal(rows, cached["rows"].astype(np.int64))
        print(f"  {kind} seq {seq} depth_idx {di}: {len(model.locations)} "
              f"locations, gates {sorted(gates)}, channels {sorted(chans)}, "
              f"prob != p*scale: {bad_p}, duplicate detector terms: {dup}, "
              f"rows identical to A4 cache: {same} ({time.time()-t0:.1f}s)")


def sec_brute():
    print("\n== brute: brute-force character sums vs A4 Prepared.evaluate")
    plist = [1e-4, 1e-3, 5e-3, 2e-2]
    worst = 0.0
    for seq in (0, 1, 2):
        for di, m in enumerate(DEPTHS):
            dem = X.load_dem("LRB", seq, di, CACHE)
            for pol in (("const", 1), ("const", 2)):
                rnds = X.used_rounds(pol, m)
                if len(rnds) > 2:
                    continue
                b = brute_eval(dem, used_dets(rnds), plist)
                prep = X.Prepared(dem, rnds)
                a = np.array([prep.evaluate(p) for p in plist])
                rel = np.max(np.abs(a / b - 1))
                worst = max(worst, rel)
            dem0 = X.load_dem("LRB_const0", seq, di, CACHE)
            b = brute_eval(dem0, [0, 1], plist)
            prep0 = X.Prepared(dem0, [0], block=2, det_index=[[0, 1]])
            a = np.array([prep0.evaluate(p) for p in plist])
            worst = max(worst, np.max(np.abs(a / b - 1)))
    print(f"  const 0/1/2, seqs 0-2, all depths, p in {plist}: max relative "
          f"difference (A and F) = {worst:.1e}")
    for pol, m in ((("unif", 1), 2), (("unif", 2), 4), (("unif", 3), 6)):
        di = DEPTHS.index(m)
        for seq in (0, 1):
            t0 = time.time()
            dem = X.load_dem("LRB", seq, di, CACHE)
            rnds = X.used_rounds(pol, m)
            b = brute_eval(dem, used_dets(rnds), plist)
            prep = X.Prepared(dem, rnds)
            a = np.array([prep.evaluate(p) for p in plist])
            print(f"  {pol[0]}{pol[1]} m={m} seq {seq} rounds {rnds} "
                  f"({len(used_dets(rnds))} detectors): max rel diff "
                  f"{np.max(np.abs(a / b - 1)):.1e}; A(1e-3)={b[1,0]:.6f} "
                  f"F(1e-3)={b[1,1]:.8f} ({time.time()-t0:.0f}s)")


def sec_first():
    print("\n== first: exact first-order coefficients (single-fault "
          "enumeration), mean over seqs 0-5")
    pols = [("unif", 1), ("unif", 2), ("unif", 3), ("unif", 4), ("unif", 6),
            ("const", 1), ("const", 2), ("const", 3), ("const", 4)]
    c1 = np.zeros((len(pols), len(SEQS), len(DEPTHS)))
    a1 = np.zeros_like(c1)
    cat_rate = {}
    for si, seq in enumerate(SEQS):
        for di, m in enumerate(DEPTHS):
            dem = X.load_dem("LRB", seq, di, CACHE)
            s = dem["scale"]
            per = {}
            for pi, pol in enumerate(pols):
                rnds = X.used_rounds(pol, m)
                key = tuple(rnds)
                if key not in per:
                    per[key] = single_fault_coeffs(dem, used_dets(rnds))
                pdet, pflip = per[key]
                c1[pi, si, di] = np.sum(s * pdet)
                a1[pi, si, di] = 1.5 * np.sum(s * pflip)
            if m >= 2:
                # transient decomposition: unif 1 minus const 1
                pd_u, _ = per[tuple(X.used_rounds(("unif", 1), m))]
                pd_c, _ = per[tuple(X.used_rounds(("const", 1), m))]
                diff = s * (pd_u - pd_c)
                q, k, sc = dem["qudit"], dem["kappa"], dem["scale"]
                anc = q >= 5
                cats = {
                    "meas shift f (5p)": anc & (sc == 5.0),
                    "reset shift f (2p)": anc & (sc == 2.0) & np.isclose(k, 1.5),
                    "measured-ancilla depol (p)": anc & (sc == 1.0) & np.isclose(k, 1.125),
                    "ancilla idle depol (2p)": anc & (sc == 2.0) & np.isclose(k, 1.125),
                    "ancilla 1q gate (0.1p)": anc & (sc == 0.1),
                    "N2 (CNOT)": np.isclose(k, 81 / 80),
                    "data 1q / idle": (~anc) & ~np.isclose(k, 81 / 80),
                }
                for name, sel in cats.items():
                    cat_rate.setdefault(name, []).append(diff[sel].sum() / m)
            if seq == 0 and m == 4:
                # contributors to a1 (final round), const 1 rounds
                pdet, pflip = per[tuple(X.used_rounds(("const", 1), m))]
                contrib = 1.5 * s * pflip
                nzc = np.flatnonzero(contrib > 1e-12)
                print("  a1 contributors (seq 0, m=4): (op index, qudit, "
                      "target, scale, kappa, P(flip & undetected), "
                      "contribution to a1)")
                for f in nzc:
                    print(f"     op {dem['opidx'][f]:5d} q{dem['qudit'][f]} "
                          f"t{dem['target'][f]:2d} s={s[f]:.1f} "
                          f"k={dem['kappa'][f]:.4f} P={pflip[f]:.4f} "
                          f"-> {contrib[f]:.4f}")
                print(f"     total a1 = {contrib.sum():.4f} "
                      f"(rate r0 = {contrib.sum() / 1.5:.4f})")
    ms = np.array(DEPTHS, float)
    sel = ms >= 2
    print("  policy   a1 range over (seq, m)       c1(m=0)  c1 line (m>=2): "
          "intercept + slope m")
    Q = np.stack([np.ones(sel.sum()), ms[sel]], 1)
    lines = [np.linalg.lstsq(Q, c1[pi].mean(0)[sel], rcond=None)[0]
             for pi in range(len(pols))]
    s_c1 = lines[pols.index(("const", 1))][1]
    for pi, pol in enumerate(pols):
        cm = c1[pi].mean(0)
        g = lines[pi]
        print(f"  {pol[0]}{pol[1]:<2d}  [{a1[pi].min():.6f}, "
              f"{a1[pi].max():.6f}]   {cm[0]:7.3f}   {g[0]:7.3f} + "
              f"{g[1]:8.4f} m   (slope - const1 slope = "
              f"{g[1] - s_c1:+.4f}; 41/k = "
              f"{41 / pol[1] if pol[0] == 'unif' else float('nan'):.4f})")
    # the notes' formula: c1 = 63.2 + 71.1 m + 41.0 n_used(m)
    print("  notes' formula 63.2 + 71.1 m + 41.0 n_used(m) vs exact c1 "
          "(mean of seqs), m >= 2: max |difference| per policy")
    for pi, pol in enumerate(pols):
        cm = c1[pi].mean(0)
        nu = np.array([len(X.used_rounds(pol, int(m))) - 1 for m in ms])
        pred = 63.2 + 71.1 * ms + 41.0 * nu
        dd = (cm - pred)[sel]
        print(f"     {pol[0]}{pol[1]:<2d} n_used={nu.tolist()}  exact-formula"
              f" = {np.round(dd, 1).tolist()}")
    print("  per-round transient rate c1(unif1)-c1(const1), divided by m, "
          "by fault category (mean over seqs, m >= 2; units of p):")
    tot = 0.0
    for name, v in cat_rate.items():
        v = np.array(v)
        tot += v.mean()
        print(f"     {name:28s} mean {v.mean():8.4f}  range "
              f"[{v.min():.4f}, {v.max():.4f}]")
    print(f"     {'total':28s} mean {tot:8.4f}")
    np.savez(os.path.join(SCR, "critic_A4_first.npz"), c1=c1, a1=a1)


def sec_second():
    print("\n== second: a2, c2 at p = 1e-6..4e-6 (cubic fits)")
    ps = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    Q3 = np.stack([ps, ps ** 2, ps ** 3], 1)
    out = {}
    for lab in ("const1", "const0", "unif1"):
        a2 = np.zeros((len(SEQS), len(DEPTHS)))
        c2 = np.zeros_like(a2)
        for si, seq in enumerate(SEQS):
            for di, m in enumerate(DEPTHS):
                if lab == "const0":
                    dem = X.load_dem("LRB_const0", seq, di, CACHE)
                    r = brute_eval(dem, [0, 1], ps)
                elif lab == "const1":
                    dem = X.load_dem("LRB", seq, di, CACHE)
                    r = brute_eval(dem, used_dets([m]), ps)
                else:
                    dem = X.load_dem("LRB", seq, di, CACHE)
                    prep = X.Prepared(dem, X.used_rounds(("unif", 1), m))
                    r = np.array([prep.evaluate(p) for p in ps])
                cA, *_ = np.linalg.lstsq(Q3, -np.log(r[:, 0]), rcond=None)
                cF, *_ = np.linalg.lstsq(Q3, -np.log(r[:, 1]), rcond=None)
                a2[si, di], c2[si, di] = cF[1], cA[1]
        out[lab] = (a2, c2)
        ms = np.array(DEPTHS, float)
        sel = ms >= 2
        Q = np.stack([np.ones(sel.sum()), ms[sel], ms[sel] ** 2], 1)
        ga, *_ = np.linalg.lstsq(Q, a2.mean(0)[sel], rcond=None)
        gc, *_ = np.linalg.lstsq(Q, c2.mean(0)[sel], rcond=None)
        print(f"  {lab}: a2(m) = {np.round(a2.mean(0), 1)}")
        print(f"          a2 fit {ga[0]:.1f} + {ga[1]:.2f} m + {ga[2]:.3f} m^2;"
              f" a2/m^2 (m>=2) = {np.round(a2.mean(0)[sel] / ms[sel]**2, 2)}")
        print(f"          c2(m) = {np.round(c2.mean(0), 1)}")
        print(f"          c2 fit {gc[0]:.1f} + {gc[1]:.2f} m + {gc[2]:.3f} m^2"
              f"  (ln A convex in m at O(p^2) iff the m^2 coefficient < 0)")
        if lab == "unif1":
            g6 = []
            for si in range(len(SEQS)):
                g, *_ = np.linalg.lstsq(Q, a2[si, sel], rcond=None)
                g6.append(g[2])
            print(f"          per-sequence a2 m^2 coefficient: "
                  f"{np.round(g6, 3)}")
    np.savez(os.path.join(SCR, "critic_A4_second.npz"),
             **{f"{k}_a2": v[0] for k, v in out.items()},
             **{f"{k}_c2": v[1] for k, v in out.items()})


def shape(F, ms):
    sel = ms >= 2
    lnF = np.log(F[sel])
    Q1 = np.stack([np.ones(sel.sum()), ms[sel]], 1)
    b1, *_ = np.linalg.lstsq(Q1, lnF, rcond=None)
    dev = np.max(np.abs(np.exp(Q1 @ b1) / F[sel] - 1))
    Q2 = np.stack([np.ones(sel.sum()), ms[sel], ms[sel] ** 2], 1)
    b2, *_ = np.linalg.lstsq(Q2, lnF, rcond=None)
    r24 = (F[2] / F[1]) ** 0.5
    r2022 = (F[8] / F[7]) ** 0.5
    eta = np.log(r2022) / np.log(r24) - 1
    return eta, dev, -b2[2], 1 - np.exp(b1[1]), np.any(np.diff(F[sel]) > 0)


def sec_hybrid():
    print("\n== hybrid: shape of detection vs hybrid (intermediate "
          "postselected, final round ML-decoded), mean of seqs 0-5")
    z = np.load(NPZ)
    ms = np.array(DEPTHS, float)
    pols = [str(s) for s in z["policies"]] + ["const0"]
    for p_val in (1e-3, 5e-3):
        k = list(z["p_eval"]).index(p_val)
        print(f"  p = {p_val}:  policy | det: eta, max|rel misfit| of best "
              f"exp, curvature d, per-step 1-f, rises? | hybrid: same")
        for pi, lab in enumerate(pols):
            if lab == "const0":
                Fd, Fh = z["F0"][:, :, k].mean(0), z["F0dec"][:, :, k].mean(0)
            else:
                Fd, Fh = z["F"][pi][:, :, k].mean(0), \
                    z["Fdec"][pi][:, :, k].mean(0)
            ed, dd, cd, rd, ud = shape(Fd, ms)
            eh, dh, ch, rh, uh = shape(Fh, ms)
            print(f"    {lab:7s} | {ed:+8.2f} {dd:8.1e} {cd:+9.2e} {rd:8.1e} "
                  f"{str(ud):5s} | {eh:+8.2f} {dh:8.1e} {ch:+9.2e} {rh:8.1e}"
                  f" {str(uh):5s}")


def sec_attr():
    print("\n== attr: A4 attribution (unif 1) on sequences 3, 4, 5")
    import A4_attribution as AT
    rows = []
    for di, m in enumerate(DEPTHS):
        if m == 0:
            continue
        vals = []
        for seq in (3, 4, 5):
            dem = X.load_dem("LRB", seq, di, CACHE)
            rnds = X.used_rounds(("unif", 1), m)
            fwd, inv, fin = AT.regions(dem, m)
            vals.append([AT.a2_of(dem, rnds)[1],
                         AT.a2_of(AT.subset(dem, ~inv), rnds)[1],
                         AT.a2_of(AT.subset(dem, ~fwd), rnds)[1]])
        rows.append((m, *np.mean(vals, 0)))
    r = np.array(rows)
    Q = np.stack([np.ones(len(r)), r[:, 0], r[:, 0] ** 2], 1)
    for col, lab in ((1, "full"), (2, "no-inverse"), (3, "no-forward")):
        g, *_ = np.linalg.lstsq(Q, r[:, col], rcond=None)
        print(f"   seqs 3-5 fit {lab:10s}: a2 = {g[0]:7.1f} + {g[1]:6.2f} m "
              f"+ {g[2]:6.3f} m^2")


def sec_const4():
    print("\n== const4: all 30 sequences, exact const 4 at m = 2, 4 and "
          "unif 1 at m = 0")
    t0 = time.time()
    plist = [5e-4, 1e-3, 2e-3, 5e-3]
    F2 = np.zeros((30, len(plist)))
    F4 = np.zeros_like(F2)
    A2 = np.zeros_like(F2)
    A4 = np.zeros_like(F2)
    for seq in range(30):
        d2 = X.load_dem("LRB", seq, 1, CACHE)
        d4 = X.load_dem("LRB", seq, 2, CACHE)
        b = brute_eval(d2, used_dets(X.used_rounds(("const", 4), 2)), plist)
        prep = X.Prepared(d4, X.used_rounds(("const", 4), 4))
        a = np.array([prep.evaluate(p) for p in plist])
        A2[seq], F2[seq] = b[:, 0], b[:, 1]
        A4[seq], F4[seq] = a[:, 0], a[:, 1]
    d0 = X.load_dem("LRB", 0, 0, CACHE)
    b0 = brute_eval(d0, used_dets([0]), plist)
    for k, p in enumerate(plist):
        d = F4[:, k] - F2[:, k]
        print(f"  p={p:.0e}: exact F(4)-F(2) mean over 30 seqs {d.mean():+.2e}"
              f" (6-seq mean {d[:6].mean():+.2e}; range [{d.min():+.1e}, "
              f"{d.max():+.1e}]); F(0) unif1 = {b0[k, 1]:.7f}")
    np.savez(os.path.join(SCR, "critic_A4_const4.npz"), p=np.array(plist),
             F2=F2, F4=F4, A2=A2, A4=A4, F0=b0[:, 1], A0=b0[:, 0])
    print(f"  ({time.time() - t0:.0f}s)")


def sec_misc():
    print("\n== misc")
    z = np.load(NPZ)
    pe = list(z["p_eval"])
    ms = np.array(DEPTHS, float)
    # (1) C16: const 1 windowed decay rates at large p (exact, 6 seqs)
    for p in (2e-3, 5e-3, 1e-2, 2e-2):
        F = z["F"][5][:, :, pe.index(p)].mean(0)
        A = z["A"][5][:, :, pe.index(p)].mean(0)
        rate = -np.diff(np.log(np.abs(F))) / np.diff(ms)
        print(f"  C16 const1 p={p:.0e}: F(m) = {np.array2string(F, precision=4)}"
              f"; -dlnF/dm per window = {np.array2string(rate, precision=3)};"
              f" A(22) = {A[-1]:.5f}")
    # (2) C20: eta per sequence and jackknife of the mean-curve eta
    for p in (1e-3, 5e-3):
        k = pe.index(p)
        for pi, lab in ((0, "unif1"), (2, "unif3"), (5, "const1")):
            for nm, arr in (("det", z["F"]), ("hyb", z["Fdec"])):
                Fs = arr[pi][:, :, k]
                per = [shape(Fs[i], ms)[0] for i in range(Fs.shape[0])]
                jk = [shape(np.delete(Fs, i, 0).mean(0), ms)[0]
                      for i in range(Fs.shape[0])]
                n = len(jk)
                se = np.sqrt((n - 1) / n * np.sum((np.array(jk) -
                                                   np.mean(jk)) ** 2))
                print(f"  C20 p={p:.0e} {lab:6s} {nm}: eta(mean curve) = "
                      f"{shape(Fs.mean(0), ms)[0]:+.2f} (jackknife SE "
                      f"{se:.2f}); per-sequence eta "
                      f"{np.array2string(np.array(per), precision=2)}")
    # (3) C7: unif 1 only z-scores at p = 1e-3 (independent circuits)
    acc = np.load(f"{X.RUN}/progress/4/unif_check_data/1.npy").astype(float)
    rej = np.load(f"{X.RUN}/progress/4/unif_check_data/1_rejected.npy")
    seqs = list(z["seqs"])
    acc, rej = acc[:, seqs, :], rej[:, seqs].astype(float)
    Ap = z["A"][0][:, :, pe.index(1e-3)].T
    Fp = z["F"][0][:, :, pe.index(1e-3)].T
    n_acc = acc.sum(-1)
    tot = n_acc + rej
    zA = (n_acc / tot - Ap) / np.sqrt(Ap * (1 - Ap) / tot)
    zz = acc[..., 0] + OMEGA * acc[..., 1] + OMEGA ** 2 * acc[..., 2]
    Fo = np.abs(zz) / n_acc
    var = (1 + 2 * Fp) / 3 + (1 - Fp) / 6 - Fp ** 2
    zF = (Fo - Fp) / np.sqrt(var / n_acc)
    print(f"  C7 unif1 only, p=1e-3, {zA.size} circuits: acceptance z mean "
          f"{zA.mean():+.2f} sd {zA.std():.2f}; fidelity z mean "
          f"{zF.mean():+.2f} sd {zF.std():.2f}; SE of a mean of n=54 "
          f"unit-variance z's = {1 / np.sqrt(zA.size):.2f}")


SECTIONS = {"extract": sec_extract, "brute": sec_brute, "first": sec_first,
            "second": sec_second, "hybrid": sec_hybrid, "attr": sec_attr,
            "const4": sec_const4, "misc": sec_misc}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SECTIONS)
    for n in names:
        SECTIONS[n]()
