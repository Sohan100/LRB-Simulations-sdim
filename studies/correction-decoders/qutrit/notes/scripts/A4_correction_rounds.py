#!/usr/bin/env python3
"""A4 revision: exact LRB with Pauli-frame CORRECTION at the used check
rounds of the real baseline circuits (nothing rejected), plus a Monte Carlo
validation that does not use any of the algebra below.

Decoder ("memoryless lookup with a Pauli frame").  At every used round r_i
the decoder computes the residual syndrome s_i = D_{r_i} - (syndrome that the
frame predicts at r_i), looks up a minimum-weight data Weyl operator c(s_i)
with static syndrome s_i (fixed table, deterministic tie-breaking, odd:
c(-s) = -c(s)), and adds it to the frame right after the round.  The
terminal readout x0 + x3 is relabelled by the logical effect of the frame.
The final round m is decoded either with the same table ('lut') or by a
genie maximum-likelihood rule on the final residual syndrome ('ml', as in
A4_exact_lrbd.terminal_decode).  Intermediate rounds can instead be
postselected ('post'), which reproduces A4_exact_lrbd exactly.

Exact evaluation.  Because the table always returns an operator with the
observed syndrome, the frame's predicted syndrome at r_i equals
A_i D_{r_{i-1}}, so s_i is exactly A4's difference detector delta_i
(A4 notes, section 3.1), whatever the earlier corrections were.  The
corrected logical shift is L - sum_i g_i(delta_i), with g_i(s) the readout
effect of c(s) inserted after round r_i (from the repository DEM rows of
the data idle locations in the reset layer after r_i, or in the M8 layer of
the final round).  A4's Fourier tables (Prepared, w = 0, 1, 2) are inverted
to probability tables: single-round faults give a measure on Z_3^4, faults
that touch rounds i and i+1 give a measure on Z_3^4 x Z_3^4.  A forward pass
with an 81-state carry then gives E[omega^{w (L - sum g)} ...] exactly.

Sections
  sigma    static syndromes of the 10 data Weyl components from the DEM
           (response after one round mapped back through A4's round map);
           consistency over rounds, depths and sequences; the lookup table
  check    (a) all intermediate rounds postselected + final 'post' equals
           A4 Prepared.evaluate; (b) + final 'ml' equals
           A4 Prepared.terminal_decode; (c) Monte Carlo of the lookup decoder
           on raw detectors with explicit frame propagation (independent of
           the difference-detector identity) equals the exact value
  run      correction at the used rounds of unif 1, 2, 4 and const 2, 4
           (final 'ml' and 'lut') for all 30 sequences at p = 1e-3, 5e-3;
           writes $A4_SCRATCH/A4_correction_rounds.npz
  shape    eta (+- jackknife SE over sequences), max relative misfit of the
           best single exponential, curvature, per-step rate, for detection,
           hybrid and every-round correction (needs A4_exact_all30.npz from
           A4_revision_checks.py merge)

    module load python/3.11
    export OMP_NUM_THREADS=1 A4_SCRATCH=<dir>
    python A4_correction_rounds.py [sections...]
Needs the DEM cache $A4_SCRATCH/A4_dem_cache (A4_exact_lrbd.py).  One core.
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
OUT = os.path.join(SCR, "A4_correction_rounds.npz")
ALL = os.path.join(SCR, "A4_exact_all30.npz")
OMEGA = np.exp(2j * np.pi / 3)
V = X.all_vectors(4)                      # (81, 4), index = v . POW
POW = np.array([27, 9, 3, 1])
NV = 81


def vidx(v):
    return (np.asarray(v) % 3) @ POW


ADD = vidx(V[:, None, :] + V[None, :, :])        # ADD[t, x] = idx(t + x)
SUB = vidx(V[:, None, :] - V[None, :, :])        # SUB[t, y] = idx(t - y)
NEG = vidx(-V)
CH = OMEGA ** (-((V @ V.T) % 3))                 # CH[s, v] = omega^{-s.v}
RESET_OFF, M8_OFF = 76, 65     # op offsets from the round's first CNOT 0->5


# ---------------------------------------------------------------------------
# GF(3) and DEM helpers
# ---------------------------------------------------------------------------
def gf3_inv(B):
    n = B.shape[0]
    M = np.concatenate([B % 3, np.eye(n, dtype=np.int64)], 1).astype(np.int64)
    for col in range(n):
        piv = col + np.flatnonzero(M[col:, col] % 3)[0]
        M[[col, piv]] = M[[piv, col]]
        M[col] = (M[col] * (1 if M[col, col] == 1 else 2)) % 3
        for r in range(n):
            if r != col and M[r, col]:
                M[r] = (M[r] - M[r, col] * M[col]) % 3
    Bi = M[:, n:]
    assert np.array_equal((B @ Bi) % 3, np.eye(n, dtype=np.int64))
    return Bi


def location_rows(dem, op_index, qudit):
    """The component rows (ndet + 1 columns) of the unique data idle
    location at operation op_index on the given qudit."""
    f = np.flatnonzero((dem["opidx"] == op_index) & (dem["qudit"] == qudit))
    assert len(f) == 1, (op_index, qudit, f)
    f = int(f[0])
    assert dem["scale"][f] == 2.0 and np.isclose(dem["kappa"][f], 1.125)
    r = np.flatnonzero(dem["owner"] == f)
    assert len(r) == 2
    return dem["rows"][r].astype(np.int64) % 3          # (2, ndet + 1)


def frame_rows(dem, r, final):
    """(10, ndet+1) response rows of the 10 data Weyl components inserted
    right after check round r (reset layer; M8 layer for the final round)."""
    off = M8_OFF if final else RESET_OFF
    st = int(dem["starts"][r])
    return np.concatenate([location_rows(dem, st + off + q, q)
                           for q in range(5)], 0)


def static_sigma(dem, m):
    """Static syndromes (10, 4) of the data components, one estimate per
    intermediate round r < m, via sigma = response(r+1) @ inv(B_r)."""
    prep = X.Prepared(dem, list(range(m + 1)))
    out = []
    for r in range(m):
        resp = frame_rows(dem, r, False)[:, 4 * (r + 1):4 * (r + 1) + 4]
        out.append((resp @ gf3_inv(prep.maps[r])) % 3)
    return out


def build_lut(sig):
    """Minimum-weight table: syndrome index -> (10,) component vector c with
    c @ sig = s (mod 3).  Candidates in a fixed order (weight 0, 1, 2; qudit
    order; Weyl order); odd symmetry c(-s) = -c(s) enforced."""
    cands = [np.zeros(10, dtype=np.int64)]
    singles = []
    for q in range(5):
        for a, b in itertools.product(range(3), repeat=2):
            if (a, b) != (0, 0):
                c = np.zeros(10, dtype=np.int64)
                c[2 * q], c[2 * q + 1] = a, b
                singles.append((q, c))
    cands += [c for _, c in singles]
    for (q1, c1), (q2, c2) in itertools.combinations(singles, 2):
        if q1 < q2:
            cands.append(c1 + c2)
    lut = -np.ones((NV, 10), dtype=np.int64)
    wt = -np.ones(NV, dtype=np.int64)
    for c in cands:
        s = int(vidx((c @ sig) % 3))
        if wt[s] < 0:
            w_ = int(np.count_nonzero(c.reshape(5, 2).any(1)))
            lut[s], wt[s] = c, w_
            sn = int(NEG[s])
            if wt[sn] < 0:
                lut[sn], wt[sn] = (-c) % 3, w_
    assert np.all(wt >= 0), "some syndrome has no weight <= 2 representative"
    return lut, wt


# ---------------------------------------------------------------------------
# exact forward pass
# ---------------------------------------------------------------------------
class Corrector:
    """Exact corrected / postselected LRB for one circuit and one set of
    used rounds."""

    def __init__(self, dem, rounds, lut, need_w2=True):
        self.rounds = rounds
        self.R = len(rounds)
        self.prep = X.Prepared(dem, rounds, nw=3 if need_w2 else 2)
        m = rounds[-1]
        # readout effect (mod 3) of every table entry inserted after r_i
        self.g = []
        for i, r in enumerate(rounds):
            L = frame_rows(dem, r, final=(i == self.R - 1))[:, -1]
            self.g.append((lut @ L) % 3)                  # (81,)
        assert m == rounds[-1]

    def final_vector(self, p, w, modes):
        """Twisted measure G_w * beta_R(t) over the final difference
        syndrome t; modes[i] in {'post', 'lut'} for i < R - 1."""
        pr = self.prep
        lf = pr._logfac(p)
        alpha = np.zeros(NV, dtype=complex)
        alpha[0] = 1.0
        beta = None
        for i in range(self.R):
            Sp = CH @ np.exp(pr.S[w, i].T @ lf) / NV
            beta = (Sp[SUB] * alpha[None, :]).sum(1)
            if i == self.R - 1:
                break
            if modes[i] == "post":
                h = beta[NEG]
            else:
                ph = OMEGA ** (-w * self.g[i])
                h = (beta[:, None] * ph[ADD]).sum(0)
            Th = np.exp(np.tensordot(lf, pr.T[w, i], axes=(0, 0)))
            Mp = CH @ Th @ CH / NV ** 2
            alpha = h @ Mp
        return np.exp(pr.G[w] @ lf) * beta

    def evaluate(self, p, modes, final):
        """(probability of not rejecting, fidelity) for final in
        {'post', 'lut', 'ml'}."""
        if final == "ml":
            bw = [self.final_vector(p, w, modes) for w in range(3)]
            P = np.stack([sum(OMEGA ** (-w * l) * bw[w] for w in range(3)) / 3
                          for l in range(3)], 1)          # (t, l)
            assert np.max(np.abs(P.imag)) < 1e-9
            P = P.real
            lhat = P.argmax(1)
            tot = P.sum()
            num = sum(P[:, l] * OMEGA ** ((l - lhat) % 3)
                      for l in range(3)).sum()
            return tot, num.real / tot, abs(num.imag) / tot
        b0 = self.final_vector(p, 0, modes)
        b1 = self.final_vector(p, 1, modes)
        if final == "post":
            z0, z1 = b0[0], b1[0]
        else:
            ph = OMEGA ** (-self.g[-1])
            z0, z1 = b0.sum(), (b1 * ph).sum()
        return z0.real, (z1 / z0).real, abs((z1 / z0).imag)


# ---------------------------------------------------------------------------
# Monte Carlo of the lookup decoder on raw detectors (independent check)
# ---------------------------------------------------------------------------
def mc_lookup(dem, rounds, lut, p, nshots, rng, chunk=4000):
    rows = dem["rows"].astype(np.int64) % 3
    owner = dem["owner"]
    nf = len(dem["scale"])
    rstart = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    ncomp = np.diff(np.r_[rstart, len(owner)])
    probs = p * dem["scale"]
    ndet = int(dem["ndet"])
    R = len(rounds)
    resp = [frame_rows(dem, r, final=(i == R - 1)) for i, r in
            enumerate(rounds)]
    vals = []
    for a in range(0, nshots, chunk):
        n = min(chunk, nshots - a)
        D = np.zeros((n, ndet + 1), dtype=np.int64)
        si, fi = np.nonzero(rng.random((n, nf)) < probs[None, :])
        for k in (1, 2, 4):
            sel = ncomp[fi] == k
            if not np.any(sel):
                continue
            u = rng.integers(1, 3 ** k, size=sel.sum())
            dig = (u[:, None] // 3 ** np.arange(k)[None, :]) % 3
            r0 = rstart[fi[sel]]
            contrib = sum(dig[:, c][:, None] * rows[r0 + c] for c in range(k))
            np.add.at(D, si[sel], contrib)
        D %= 3
        pred = np.zeros_like(D)
        for i, r in enumerate(rounds):
            s = (D[:, 4 * r:4 * r + 4] - pred[:, 4 * r:4 * r + 4]) % 3
            c = lut[vidx(s)]                                # (n, 10)
            pred = (pred + c @ resp[i]) % 3
        Lc = (D[:, ndet] - pred[:, ndet]) % 3
        vals.append(OMEGA ** Lc)
    v = np.concatenate(vals)
    return v.real.mean(), v.real.std() / np.sqrt(len(v))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def get_lut(verbose=False):
    sigs = []
    for seq, di in ((0, 8), (7, 5), (19, 3)):
        dem = X.load_dem("LRB", seq, di, CACHE)
        sigs += static_sigma(dem, X.DEPTHS[di])
    sig = sigs[0]
    same = all(np.array_equal(s, sig) for s in sigs)
    lut, wt = build_lut(sig)
    if verbose:
        print(f"  static syndrome estimates: {len(sigs)} (rounds of three "
              f"circuits); all identical: {same}")
        for q in range(5):
            print(f"    qudit {q}: component 0 -> {sig[2 * q]}, component 1"
                  f" -> {sig[2 * q + 1]}")
        print(f"  table: syndromes by weight of the chosen operator: "
              f"{np.bincount(wt).tolist()} (weight 0, 1, 2)")
    assert same
    return lut, sig


def sec_sigma():
    print("\n== sigma: static syndromes and the lookup table")
    get_lut(verbose=True)


def sec_check():
    print("\n== check: exact evaluator against A4 and against Monte Carlo")
    lut, _ = get_lut()
    worst_a = worst_b = 0.0
    for seq in (0, 3):
        for di in (1, 3, 6, 8):
            m = X.DEPTHS[di]
            dem = X.load_dem("LRB", seq, di, CACHE)
            for pol in (("unif", 1), ("unif", 3), ("const", 2), ("const", 1)):
                rnds = X.used_rounds(pol, m)
                C = Corrector(dem, rnds, lut)
                modes = ["post"] * (len(rnds) - 1)
                for p in (1e-3, 5e-3, 2e-2):
                    a, f, _ = C.evaluate(p, modes, "post")
                    a0, f0 = C.prep.evaluate(p)
                    worst_a = max(worst_a, abs(a / a0 - 1), abs(f / f0 - 1))
                    a, f, _ = C.evaluate(p, modes, "ml")
                    a0, f0, _ = C.prep.terminal_decode(p)
                    worst_b = max(worst_b, abs(a / a0 - 1), abs(f / f0 - 1))
    print(f"  (a) postselect all used rounds vs A4 evaluate: max relative "
          f"difference {worst_a:.1e}")
    print(f"  (b) postselect intermediate, ML final vs A4 terminal_decode: "
          f"max relative difference {worst_b:.1e}")
    rng = np.random.default_rng(5)
    for seq, di, pol, p, n in ((0, 2, ("unif", 1), 5e-3, 200000),
                               (3, 4, ("unif", 1), 5e-3, 200000),
                               (5, 4, ("unif", 2), 1e-2, 200000),
                               (1, 3, ("const", 2), 1e-2, 200000),
                               (2, 6, ("unif", 1), 2e-3, 200000)):
        m = X.DEPTHS[di]
        dem = X.load_dem("LRB", seq, di, CACHE)
        rnds = X.used_rounds(pol, m)
        C = Corrector(dem, rnds, lut, need_w2=False)
        modes = ["lut"] * (len(rnds) - 1)
        t0 = time.time()
        _, fex, im = C.evaluate(p, modes, "lut")
        fmc, se = mc_lookup(dem, rnds, lut, p, n, rng)
        print(f"  (c) seq {seq} m={m:2d} {pol[0]}{pol[1]} p={p:.0e}: exact "
              f"lookup-decoded F = {fex:.5f} (|Im| {im:.1e}); Monte Carlo "
              f"{fmc:.5f} +- {se:.5f} ({n} shots), z = {(fmc - fex) / se:+.2f}"
              f"  [{time.time() - t0:.0f}s]")


RUN_POLS = [("unif", 1), ("unif", 2), ("unif", 4), ("const", 2), ("const", 4)]
RUN_P = [1e-3, 5e-3]


def sec_run():
    print("\n== run: exact correction at the used rounds, 30 sequences")
    lut, _ = get_lut()
    t0 = time.time()
    nS, nD = 30, len(X.DEPTHS)
    Fml = np.zeros((len(RUN_POLS), nS, nD, len(RUN_P)))
    Flut = np.zeros_like(Fml)
    for seq in range(nS):
        for di, m in enumerate(X.DEPTHS):
            dem = X.load_dem("LRB", seq, di, CACHE)
            done = {}
            for pi, pol in enumerate(RUN_POLS):
                rnds = tuple(X.used_rounds(pol, m))
                if rnds not in done:
                    C = Corrector(dem, list(rnds), lut)
                    modes = ["lut"] * (len(rnds) - 1)
                    done[rnds] = [(C.evaluate(p, modes, "ml")[1],
                                   C.evaluate(p, modes, "lut")[1])
                                  for p in RUN_P]
                for k in range(len(RUN_P)):
                    Fml[pi, seq, di, k], Flut[pi, seq, di, k] = done[rnds][k]
        print(f"  seq {seq} done  t={time.time() - t0:6.1f}s", flush=True)
    np.savez(OUT, Fml=Fml, Flut=Flut, p=np.array(RUN_P),
             policies=np.array([f"{a}{b}" for a, b in RUN_POLS]),
             depths=np.array(X.DEPTHS))
    print(f"  wrote {OUT}")


def sec_shape():
    import A4_revision_checks as RC
    print("\n== shape: detection vs hybrid vs correction at the used rounds"
          " (30-sequence mean curves; jackknife SE over sequences)")
    z = np.load(OUT)
    e = np.load(ALL)
    pe = list(e["p_eval"])
    pols_all = [f"{a}{b}" for a, b in X.POLICIES]
    for k, p in enumerate(z["p"]):
        kk = pe.index(p)
        print(f"  p = {p:.0e}: policy / protocol: eta +- SE | max rel misfit"
              f" | curvature d | per-step 1-f | F(2) F(22) | any rise")
        for pi, lab in enumerate(z["policies"]):
            lab = str(lab)
            pa = pols_all.index(lab)
            rows = (("detection", e["F"][pa][:, :, kk]),
                    ("hybrid (ML final)", e["Fdec"][pa][:, :, kk]),
                    ("correct, ML final", z["Fml"][pi][:, :, k]),
                    ("correct, lookup final", z["Flut"][pi][:, :, k]))
            for name, arr in rows:
                mean = arr.mean(0)
                eta, dev, d, r, up = RC.shape(mean)
                se = RC.jackknife(lambda f: RC.shape(f)[0], arr)
                print(f"    {lab:6s} {name:22s} {eta:+7.2f} +- {se:5.2f} | "
                      f"{dev:8.1e} | {d:+9.2e} | {r:8.1e} | {mean[1]:.5f} "
                      f"{mean[8]:.5f} | {up}")


SECTIONS = {"sigma": sec_sigma, "check": sec_check, "run": sec_run,
            "shape": sec_shape}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        SECTIONS[name]()
