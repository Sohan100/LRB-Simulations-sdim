#!/usr/bin/env python3
"""A4: exact (no sampling) LRB-D acceptance and accepted fidelity from the DEM.

Idea (derived in A4-nonmarkov-theory.md, section 3.1).  For the Pauli
(Weyl) noise of the SI1000 model every noise location j fires with
probability p_j and then draws a uniformly random non-zero vector e_j in
Z_3^{k_j} (k = 1 for the 'f' shift channel, 2 for N1 depolarizing, 4 for N2).
The repository DEM gives the linear response of every detector and of the
logical observable to every unit component, so

    E[ omega^{w L} * 1{D_U = 0} ]
        = 3^{-|U|} sum_{u in Z_3^{|U|}} prod_j phi_j(A_j^T (u, w)),
    phi_j(v) = 1                      if v = 0,
             = 1 - kappa_j p_j        otherwise,   kappa_j = 3^k/(3^k - 1).

w = 0 gives the acceptance, w = 1 gives E[omega^L 1_acc] (real for this
symmetric noise), and the accepted fidelity is their ratio.  The sum over u
is exponential in the number of used detectors, but after the triangular
change of variables to "difference detectors" between consecutive used
rounds every fault touches at most two consecutive used rounds, so the sum
is a product of 81 x 81 transfer matrices.  The script

  * compiles (and caches) the repository DEM for selected circuits of the
    read-only baseline run folder,
  * evaluates acceptance and accepted fidelity exactly for any check policy
    and any list of p values (the DEM structure does not depend on p),
  * optionally evaluates terminal maximum-likelihood correction instead of
    terminal rejection ("hybrid": intermediate rounds postselected, final
    round decoded), via an 81-point Fourier inversion of the last vector,
  * compares the exact values with the per-sequence shot counts of the
    baseline run (z-scores),
  * extracts small-p expansion coefficients of -ln F and -ln A.

Run (login node, one core with OMP_NUM_THREADS=1; ~8 minutes for 6 sequences):
    module load python/3.11
    python studies/correction-decoders/qutrit/notes/scripts/A4_exact_lrbd.py \
        --seqs 0 1 2 3 4 5 --out <scratch>/A4_exact.npz

Reads only; writes the cache and the output file to the paths given
(default: the session scratch directory if $A4_SCRATCH is set, else /tmp).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "..", "..", ".."))
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
       "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
PROBS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 0.01, 0.0125,
         0.015, 0.02, 0.03, 0.04, 0.05]
OMEGA = np.exp(2j * np.pi / 3)


# ----------------------------------------------------------------------------
# DEM compilation and caching
# ----------------------------------------------------------------------------
def compile_circuit(path: str) -> dict:
    """Compile one .chp circuit into dense integer response arrays."""
    sys.path.insert(0, os.path.join(WORKTREE, "src"))
    from lrb.correction.compact_dem import read_run_circuit as read_circuit  # noqa: E402
    from lrb.correction.compact_dem import build_compact_detector_error_model

    circuit = read_circuit(path)
    model = build_compact_detector_error_model(circuit)
    ops = circuit.operations
    # first gate of each ordinary check round: CNOT 0 -> 5
    starts = [i for i, o in enumerate(ops)
              if o.gate_name == "CNOT" and o.qudit_index == 0
              and o.target_index == 5]
    ndet = model.num_detectors
    rows, owner, scale, kappa, opidx, qd, tg = [], [], [], [], [], [], []
    for f, loc in enumerate(model.locations):
        if loc.gate == "N1" and loc.channel == "f":
            comps = [0]
        elif loc.gate == "N1" and loc.channel == "p":
            comps = [1]
        elif loc.gate == "N1":
            comps = [0, 1]
        else:
            comps = [0, 1, 2, 3]
        for cidx in comps:
            r = np.zeros(ndet + 1, dtype=np.int8)
            terms = loc.component_terms[cidx]
            for d, v in terms.detector:
                r[d] = v % 3
            for _, v in terms.logical:
                r[ndet] = (r[ndet] + v) % 3
            rows.append(r)
            owner.append(f)
        s = float(loc.params.get("prob_scale", 1.0))
        scale.append(s)
        kappa.append(3 ** len(comps) / (3 ** len(comps) - 1))
        opidx.append(loc.operation_index)
        qd.append(loc.qudit)
        tg.append(loc.target)
    if not rows:                       # noiseless circuit (const0, m = 0)
        rows = np.zeros((0, ndet + 1), dtype=np.int8)
    return {
        "rows": np.array(rows, dtype=np.int8).reshape(-1, ndet + 1),
        "owner": np.array(owner, dtype=np.int32),
        "scale": np.array(scale), "kappa": np.array(kappa),
        "opidx": np.array(opidx, dtype=np.int64),
        "qudit": np.array(qd, dtype=np.int64),
        "target": np.array(tg, dtype=np.int64),
        "starts": np.array(starts, dtype=np.int64),
        "labels": np.array(model.detector_labels),
        "ndet": ndet,
    }


def load_dem(kind: str, seq: int, depth_idx: int, cache: str,
             p_idx: int = 4) -> dict:
    """kind = 'LRB' or 'LRB_const0'.  DEM structure is p independent."""
    os.makedirs(cache, exist_ok=True)
    fn = os.path.join(cache, f"{kind}_{seq}_{depth_idx}.npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=False)
        return {k: (z[k] if k != "ndet" else int(z[k])) for k in z.files}
    dem = compile_circuit(f"{RUN}/experiments/{kind}/{seq}/{p_idx}/"
                          f"{depth_idx}.chp")
    np.savez_compressed(fn, **dem)
    return dem


# ----------------------------------------------------------------------------
# GF(3) helpers
# ----------------------------------------------------------------------------
INV3 = {1: 1, 2: 2}


def solve_gf3(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return B (b x b) with X @ B = Y (mod 3) for X, Y of shape (N, b).

    Free variables are set to zero; raises if the system is inconsistent.
    """
    b = X.shape[1]
    M = np.unique(np.concatenate([X % 3, Y % 3], axis=1).astype(np.int64),
                  axis=0)
    row = 0
    pivots = []
    for col in range(b):
        cand = np.flatnonzero(M[row:, col] % 3) + row
        if len(cand) == 0:
            continue
        piv = cand[0]
        M[[row, piv]] = M[[piv, row]]
        M[row] = (M[row] * INV3[int(M[row, col] % 3)]) % 3
        others = np.flatnonzero(M[:, col] % 3)
        others = others[others != row]
        M[others] = (M[others] - np.outer(M[others, col], M[row])) % 3
        pivots.append((row, col))
        row += 1
        if row == M.shape[0]:
            break
    if np.any(M[row:, b:] % 3):
        raise RuntimeError("inconsistent GF(3) system for round map")
    B = np.zeros((b, b), dtype=np.int64)
    for r, col in pivots:
        B[col] = M[r, b:]
    return B


def all_vectors(b: int) -> np.ndarray:
    return np.array(list(itertools.product(range(3), repeat=b)),
                    dtype=np.int64)


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
def used_rounds(policy: tuple, m: int) -> list[int]:
    """Rounds used by the repository postselection (lrb_simulation.py)."""
    kind, val = policy
    rounds = {m}
    if kind == "unif":
        k = val
        if m >= k > 0:
            rounds |= {j * k - 1 for j in range(1, m // k + 1)
                       if j * k - 1 < m}
    elif kind == "const":
        c = val
        if 0 < c <= m:
            interval = m // c
            rounds |= {j * interval - 1 for j in range(1, c)}
    else:
        raise ValueError(policy)
    return sorted(rounds)


# ----------------------------------------------------------------------------
# Exact evaluation
# ----------------------------------------------------------------------------
class Prepared:
    """Fault classification and count tables for one circuit and policy."""

    def __init__(self, dem: dict, rounds: list[int], block: int = 4,
                 det_index=None, nw: int = 2):
        # nw = number of character orders w = 0..nw-1 tabulated (2 for
        # acceptance and fidelity; 3 when the full joint law of the logical
        # shift is needed, e.g. by A4_correction_rounds.py)
        rows, owner = dem["rows"], dem["owner"]
        ndet = int(dem["ndet"])
        nf = len(dem["scale"])
        self.R = len(rounds)
        self.block = block
        self.nw = nw
        V = all_vectors(block)
        self.V = V
        nv = len(V)
        if nf == 0:                     # noiseless circuit
            self.types = []
            self.G = np.zeros((nw, 0))
            self.S = np.zeros((nw, self.R, 0, nv))
            self.T = np.zeros((nw, max(self.R - 1, 0), 0, nv, nv),
                              dtype=np.int32)
            self.nfaults = self.n_pairs = self.n_single = self.n_global = 0
            self.maps = []
            return
        if det_index is None:
            det_index = [[block * r + i for i in range(block)]
                         for r in rounds]
        D = np.stack([rows[:, idx] for idx in det_index], axis=1).astype(
            np.int64)                          # (C, R, b)
        L = rows[:, ndet].astype(np.int64)     # (C,)
        # round maps A_i : D_{r_{i-1}} -> D_{r_i} for persistent faults
        delta = D.copy()
        self.maps = []
        for i in range(1, self.R):
            if block == 4 and len(dem["starts"]) > rounds[i - 1]:
                # data-only faults born before round r_{i-1} starts; their
                # data error is persistent and fully seen at r_{i-1}
                start = dem["starts"][rounds[i - 1]]
                q, tq = dem["qudit"][owner], dem["target"][owner]
                data_only = (q < 5) & ((tq < 0) | (tq < 5))
                pers = (dem["opidx"][owner] < start) & data_only
            else:
                pers = np.zeros(len(owner), dtype=bool)
            if np.any(pers):
                B = solve_gf3(D[pers, i - 1, :], D[pers, i, :])
            else:
                B = np.zeros((block, block), dtype=np.int64)
            self.maps.append(B)
            delta[:, i, :] = (D[:, i, :] - D[:, i - 1, :] @ B) % 3
        nz = np.any(delta != 0, axis=2)                       # (C, R)
        offsets = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        fnz = np.logical_or.reduceat(nz, offsets, axis=0)     # (F, R)
        lnz = np.logical_or.reduceat(L != 0, offsets)          # (F,)
        assert len(offsets) == nf
        first = np.where(fnz.any(1), fnz.argmax(1), -1)
        last = np.where(fnz.any(1),
                        self.R - 1 - fnz[:, ::-1].argmax(1), -1)
        span = last - first
        if np.any(span > 1):
            bad = np.flatnonzero(span > 1)[:5]
            raise RuntimeError(f"banding violated for faults {bad}")
        # fault types (scale, kappa)
        keys = sorted({(float(s), float(k)) for s, k in
                       zip(dem["scale"], dem["kappa"])})
        self.types = keys
        tindex = {k: t for t, k in enumerate(keys)}
        ftype = np.array([tindex[(float(s), float(k))] for s, k in
                          zip(dem["scale"], dem["kappa"])])
        nt = len(keys)
        # count tables for w in {0, ..., nw - 1}
        self.G = np.zeros((nw, nt))
        self.S = np.zeros((nw, self.R, nt, nv))
        self.T = np.zeros((nw, max(self.R - 1, 0), nt, nv, nv),
                          dtype=np.int32)
        onehot = np.zeros((nf, nt))
        onehot[np.arange(nf), ftype] = 1.0
        rowf = owner                                   # fault of each row
        # global faults: no used-detector response
        glob = first[rowf] < 0
        for w in range(nw):
            hit = np.zeros(nf, dtype=bool)
            np.logical_or.at(hit, rowf[glob], (w * L[glob]) % 3 != 0)
            self.G[w] = hit[first < 0] @ onehot[first < 0]
        # faults grouped by (first used round, span); vectorised per group
        lut = np.array([False, True, True, False, True, True, False])
        V8 = V.astype(np.int64)
        for i in range(self.R):
            for sp in (0, 1):
                fsel = np.flatnonzero((first == i) & (span == sp))
                if len(fsel) == 0:
                    continue
                rsel = np.flatnonzero(np.isin(rowf, fsel))
                loc = np.searchsorted(fsel, rowf[rsel])
                starts_loc = np.flatnonzero(np.r_[True, loc[1:] != loc[:-1]])
                a = ((delta[rsel, i, :] @ V8.T) % 3).astype(np.int8)
                oh = onehot[fsel].astype(np.float32)        # (nf_g, nt)
                if sp == 1:
                    bv = ((delta[rsel, i + 1, :] @ V8.T) % 3).astype(np.int8)
                    ab = a[:, :, None] + bv[:, None, :]      # (rows, nv, nv)
                for w in range(nw):
                    lw = ((w * L[rsel]) % 3).astype(np.int8)
                    if sp == 0:
                        nzr = lut[a + lw[:, None]]          # (rows, nv)
                        nzf = np.logical_or.reduceat(nzr, starts_loc, axis=0)
                        self.S[w, i] += oh.T @ nzf.astype(np.float32)
                    else:
                        nzr = lut[ab + lw[:, None, None]]   # (rows, nv, nv)
                        nzf = np.logical_or.reduceat(nzr, starts_loc, axis=0)
                        cnt = np.tensordot(oh, nzf.astype(np.float32),
                                           axes=(0, 0))      # (nt, nv, nv)
                        self.T[w, i] += np.rint(cnt).astype(np.int32)
        self.nfaults = nf
        self.n_pairs = int(np.sum(span == 1))
        self.n_single = int(np.sum(span == 0))
        self.n_global = int(np.sum(first < 0))

    def _logfac(self, p: float) -> np.ndarray:
        return np.array([np.log1p(-k * s * p) for s, k in self.types])

    def last_vector(self, p: float, w: int) -> tuple[np.ndarray, float]:
        """x_R(v_R) and global factor; Z = G * sum(x_R)."""
        lf = self._logfac(p)
        nv = len(self.V)
        x = np.exp(self.S[w, 0].T @ lf) / nv
        for i in range(self.R - 1):
            Ti = np.exp(np.tensordot(lf, self.T[w, i], axes=(0, 0)))
            x = (x @ Ti) / nv * np.exp(self.S[w, i + 1].T @ lf)
        return x, float(np.exp(self.G[w] @ lf))

    def evaluate(self, p: float) -> tuple[float, float]:
        """(acceptance, accepted fidelity) exactly."""
        x0, g0 = self.last_vector(p, 0)
        x1, g1 = self.last_vector(p, 1)
        acc = g0 * x0.sum()
        num = g1 * x1.sum()
        return acc, num / acc

    def terminal_decode(self, p: float) -> tuple[float, float, float]:
        """Hybrid: intermediate used rounds postselected, final decoded (ML).

        Returns (acceptance of intermediate rounds, decoded fidelity,
        detected fidelity for comparison).
        """
        x0, g0 = self.last_vector(p, 0)
        x1, g1 = self.last_vector(p, 1)
        V = self.V
        phase = OMEGA ** (-(V @ V.T % 3))         # (s, v): omega^{-s.v}
        c0 = g0 * (phase @ x0)                     # P(acc', D_m = s)
        c1 = g1 * (phase @ x1)                     # E[w^L 1{acc', D_m=s}]
        c2 = np.conj(c1)
        # P(L = l, s) = (1/3) sum_w omega^{-w l} c_w(s)
        P = np.stack([(c0 + OMEGA ** (-l) * c1 + OMEGA ** (-2 * l) * c2) / 3
                      for l in range(3)], axis=1).real   # (s, l)
        lhat = P.argmax(axis=1)
        acc = c0.real.sum()
        num = np.sum(OMEGA ** (-lhat) * c1)
        return acc, abs(num) / acc, (c1[0].real / c0[0].real)


# ----------------------------------------------------------------------------
# Baseline data
# ----------------------------------------------------------------------------
def load_counts(p_idx: int, policy: tuple):
    kind, val = policy
    sub = "unif_check_data" if kind == "unif" else "const_check_data"
    base = f"{RUN}/progress/{p_idx}/{sub}/{val}"
    acc = np.load(base + ".npy").astype(np.float64)        # (9, 30, 3)
    rej = np.load(base + "_rejected.npy").astype(np.float64)
    return acc, rej


def observed(acc_counts: np.ndarray, rej_counts: np.ndarray):
    n_acc = acc_counts.sum(-1)
    tot = n_acc + rej_counts
    z = acc_counts[..., 0] + OMEGA * acc_counts[..., 1] + \
        OMEGA ** 2 * acc_counts[..., 2]
    F = np.abs(z) / np.maximum(n_acc, 1)
    return n_acc / tot, F, n_acc, tot


def zscores(A_pred, F_pred, acc_counts, rej_counts):
    A_obs, F_obs, n_acc, tot = observed(acc_counts, rej_counts)
    zA = (A_obs - A_pred) / np.sqrt(A_pred * (1 - A_pred) / tot)
    q = (2.0 / 3.0) * (1 - F_pred)
    var = (1 - q) + q / 4 - F_pred ** 2
    zF = (F_obs - F_pred) / np.sqrt(np.maximum(var, 1e-300) / n_acc)
    return zA, zF


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
POLICIES = [("unif", 1), ("unif", 2), ("unif", 3), ("unif", 4),
            ("unif", 6), ("const", 1), ("const", 2), ("const", 3),
            ("const", 4)]
P_EVAL = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
P_SMALL = [2e-5, 4e-5, 6e-5, 8e-5, 1e-4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--cache", default=os.path.join(
        os.environ.get("A4_SCRATCH", "/tmp"), "A4_dem_cache"))
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("A4_SCRATCH", "/tmp"), "A4_exact.npz"))
    args = ap.parse_args()
    t0 = time.time()
    nS, nD, nP = len(args.seqs), len(DEPTHS), len(POLICIES)
    A = np.zeros((nP, nS, nD, len(P_EVAL)))
    F = np.zeros_like(A)
    Fdec = np.zeros_like(A)      # hybrid: postselect intermediate, decode last
    Adec = np.zeros_like(A)
    As = np.zeros((nP, nS, nD, len(P_SMALL)))
    Fs = np.zeros_like(As)
    A0 = np.zeros((nS, nD, len(P_EVAL)))
    F0 = np.zeros_like(A0)
    F0dec = np.zeros_like(A0)
    A0s = np.zeros((nS, nD, len(P_SMALL)))
    F0s = np.zeros_like(A0s)
    for si, seq in enumerate(args.seqs):
        for di, m in enumerate(DEPTHS):
            dem = load_dem("LRB", seq, di, args.cache)
            for pi, pol in enumerate(POLICIES):
                prep = Prepared(dem, used_rounds(pol, m))
                for k, p in enumerate(P_EVAL):
                    A[pi, si, di, k], F[pi, si, di, k] = prep.evaluate(p)
                    Adec[pi, si, di, k], Fdec[pi, si, di, k], _ = \
                        prep.terminal_decode(p)
                for k, p in enumerate(P_SMALL):
                    As[pi, si, di, k], Fs[pi, si, di, k] = prep.evaluate(p)
            dem0 = load_dem("LRB_const0", seq, di, args.cache)
            prep0 = Prepared(dem0, [0], block=2,
                             det_index=[list(range(int(dem0["ndet"])))])
            for k, p in enumerate(P_EVAL):
                A0[si, di, k], F0[si, di, k] = prep0.evaluate(p)
                _, F0dec[si, di, k], _ = prep0.terminal_decode(p)
            for k, p in enumerate(P_SMALL):
                A0s[si, di, k], F0s[si, di, k] = prep0.evaluate(p)
            print(f"seq {seq} m={m:2d} done  t={time.time() - t0:6.1f}s",
                  flush=True)
    np.savez(args.out, A=A, F=F, Adec=Adec, Fdec=Fdec, As=As, Fs=Fs,
             A0=A0, F0=F0, F0dec=F0dec, A0s=A0s, F0s=F0s,
             seqs=np.array(args.seqs), depths=np.array(DEPTHS),
             p_eval=np.array(P_EVAL), p_small=np.array(P_SMALL),
             policies=np.array([f"{a}{b}" for a, b in POLICIES]))
    # --- validation against baseline counts at the run's p values ---------
    print("\nValidation against baseline per-sequence counts")
    for p_val in (1e-3, 5e-3, 1e-2, 2e-2):
        k = P_EVAL.index(p_val)
        p_idx = PROBS.index(p_val)
        zAs, zFs = [], []
        for pi, pol in enumerate(POLICIES):
            accc, rejc = load_counts(p_idx, pol)
            zA, zF = zscores(A[pi, :, :, k].T, F[pi, :, :, k].T,
                             accc[:, args.seqs, :], rejc[:, args.seqs])
            zAs.append(zA[1:]), zFs.append(zF[1:])
        accc, rejc = load_counts(p_idx, ("const", 0))
        zA, zF = zscores(A0[:, :, k].T, F0[:, :, k].T,
                         accc[:, args.seqs, :], rejc[:, args.seqs])
        zAs.append(zA[1:]), zFs.append(zF[1:])
        zA_all, zF_all = np.concatenate([z.ravel() for z in zAs]), \
            np.concatenate([z.ravel() for z in zFs])
        print(f" p={p_val:7.4f}: acceptance z mean {zA_all.mean():+.3f} "
              f"sd {zA_all.std():.3f} max|z| {np.abs(zA_all).max():.2f} | "
              f"fidelity z mean {zF_all.mean():+.3f} sd {zF_all.std():.3f} "
              f"max|z| {np.abs(zF_all).max():.2f}  (n={zA_all.size})")
    print(f"total time {time.time() - t0:.1f}s; wrote {args.out}")


if __name__ == "__main__":
    main()
