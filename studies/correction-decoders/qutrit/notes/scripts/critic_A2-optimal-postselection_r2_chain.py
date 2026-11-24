#!/usr/bin/env python3
"""Critic (round 2) of A2-optimal-postselection: exact posteriors of chosen
detector histories at depth m > 2 (conjecture C23).

Method (hidden-Markov contraction over check rounds, raw detectors).
Cut the circuit after the last ancilla measurement of each round r. A fault
before cut r affects the later raw detectors (rounds > r) and the logical
shift L only through the class of the data error modulo stabilizers, so the
"future-effect" vectors of all faults before cut r span a GF(3) subspace V_r
of small dimension k_r (<= 6 expected; computed, not assumed). With a basis
B_r of V_r, the state after cut r is the coordinate vector c_r in GF(3)^k_r.
A state c_{r-1} contributes U_r c_{r-1} to round r's detectors and maps
linearly to c_r = T_r c_{r-1}; faults of segment r (between cuts r-1 and r)
contribute an independent (round-r shift, coordinate) pair whose exact law
K_r is built by direct nonnegative convolution. Then
    P(D = d, L = l) = [ s_{-1} M_0(d_0) M_1(d_1) ... M_m(d_m) ](l)
with M_r(d)[c, c'] = K_r[d - U_r c, c' - T_r c]. The only approximation is
floating point; no DFT, no truncation.

Validation: at m = 2 every history of the candidate sets is compared with
the full exact law (A2's DFT function, itself validated by direct
convolution in critic_A2-optimal-postselection_r2_circuit.py).

Candidates compared with D = 0:
  (a) all single-round flag patterns (one round, any of 80 nonzero values);
  (b) all histories produced by a single fault (each DEM location, each
      nonzero fault vector), deduplicated;
  (c) all pairs of single-ancilla flags in two different rounds.

Reads the baseline run folder read-only. One core.
Run: module load python/3.11 && python critic_A2-optimal-postselection_r2_chain.py
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import time

import numpy as np

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass

WT = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-"
      "2bc490701489f8e7/correction-decoders-qutrit-2c455b")
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
       "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
sys.path.insert(0, os.path.join(WT, "src"))
from sdim.circuit_io import read_circuit  # noqa: E402
from lrb.dem_simulation import build_compact_detector_error_model  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
PROBS = [float(x) for x in open(f"{RUN}/probs.txt").read().split(",")]
POW4 = 3 ** np.arange(3, -1, -1)


# ------------------------------------------------------------ GF(3) algebra
def rref3(rows):
    """Row-reduced echelon basis over GF(3). Returns (basis, pivots)."""
    A = np.array(rows, dtype=np.int64) % 3 if len(rows) else np.zeros((0, 0))
    if A.size == 0:
        return A.reshape(0, A.shape[1] if A.ndim == 2 else 0), []
    A = A.copy()
    piv, r = [], 0
    for c in range(A.shape[1]):
        nz = np.flatnonzero(A[r:, c]) + r
        if len(nz) == 0:
            continue
        A[[r, nz[0]]] = A[[nz[0], r]]
        A[r] = (A[r] * (1 if A[r, c] == 1 else 2)) % 3   # 2*2 = 1 mod 3
        for i in range(A.shape[0]):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % 3
        piv.append(c)
        r += 1
        if r == A.shape[0]:
            break
    return A[:r], piv


def coords(basis, piv, v):
    c = v[piv] % 3
    assert np.all((c @ basis - v) % 3 == 0), "vector not in span"
    return c


# ------------------------------------------------------------- the model
def load(p_idx, d_idx, cl):
    circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/{p_idx}/{d_idx}.chp")
    model = build_compact_detector_error_model(circ)
    m = DEPTHS[d_idx]
    # cut r = op index of the last ancilla measurement of round r
    anc_meas = [i for i, op in enumerate(circ.operations)
                if op.gate_name == "M" and int(op.qudit_index) in (5, 6, 7, 8)]
    assert len(anc_meas) == 4 * (m + 1)
    cuts = [anc_meas[4 * r + 3] for r in range(m + 1)]
    return model, m, cuts


def fault_images(loc, nd):
    R = np.zeros((nd + 1, loc.vector_width), dtype=np.int64)
    for c, t in enumerate(loc.component_terms):
        for i, v in t.detector:
            R[i, c] = v
        for i, v in t.logical:
            R[nd + i, c] = v
    if loc.gate == "N1":
        comps = {"d": [0, 1], "f": [0], "p": [1]}[loc.channel]
    else:
        dist = np.asarray(loc.params["prob_dist"], float)
        assert len(dist) == 81 and np.all(dist[1:] == dist[1])
        comps = [0, 1, 2, 3]
    Rc = R[:, comps]
    imgs = {}
    for v in itertools.product(range(3), repeat=len(comps)):
        if any(v):
            u = tuple(int(x) for x in (Rc @ np.array(v)) % 3)
            imgs[u] = imgs.get(u, 0) + 1
    return imgs, 3 ** len(comps) - 1


class Chain:
    def __init__(self, model, m, cuts):
        nd = model.num_detectors
        self.m, self.nd = m, nd
        rnd = np.array([int(l.split("_r")[1].split("_")[0])
                        for l in model.detector_labels])
        anc = np.array([int(l.split("_w")[1]) - 5
                        for l in model.detector_labels])
        # coordinate order: round-major, ancilla 0..3, then L
        self.col = {(r, a): i for i, (r, a) in enumerate(zip(rnd, anc))}
        order = [self.col[(r, a)] for r in range(m + 1) for a in range(4)]
        self.order = np.array(order + [nd])
        segs = [[] for _ in range(m + 2)]
        for loc in model.locations:
            s = int(np.searchsorted(cuts, loc.operation_index))  # 0..m+1
            imgs, tot = fault_images(loc, nd)
            imgs = {tuple(np.array(u)[self.order]): c for u, c in imgs.items()}
            if any(any(u) for u in imgs):
                segs[s].append((loc.probability, tot, imgs))
        self.segs = segs
        # check causality and build the bases V_r of future effects
        def fut(u, r):           # restriction to rounds > r and L
            return np.array(u[4 * (r + 1):])
        self.B, self.piv, self.k = [], [], []
        prev_rows = []
        for r in range(m + 1):
            rows = [fut(u, r) for (_, _, imgs) in segs[r] for u in imgs]
            for (_, _, imgs) in segs[r]:
                for u in imgs:
                    assert not any(u[:4 * r]), "fault affects an earlier round"
            # propagate the previous basis
            if r > 0:
                rows += [row[4:] for row in self.B[r - 1]]
            B, piv = rref3([x for x in rows if x.any()]) if any(
                x.any() for x in rows) else (np.zeros((0, len(fut(rows[0], r)) if rows else 0), dtype=np.int64), [])
            self.B.append(B)
            self.piv.append(piv)
            self.k.append(B.shape[0])
        # linear maps U_r (state -> round-r shift) and T_r (state -> new state)
        self.U, self.T = [], []
        for r in range(m + 1):
            kprev = self.k[r - 1] if r > 0 else 0
            U = np.zeros((4, kprev), dtype=np.int64)
            T = np.zeros((self.k[r], kprev), dtype=np.int64)
            for i in range(kprev):
                row = self.B[r - 1][i]          # over rounds >= r and L
                U[:, i] = row[:4]
                if self.k[r]:
                    T[:, i] = coords(self.B[r], self.piv[r], row[4:])
                else:
                    assert not row[4:].any()
            self.U.append(U)
            self.T.append(T)
        # segment kernels over (round-r shift [81], new coords [3^k_r])
        self.K = []
        for r in range(m + 1):
            k = self.k[r]
            K = np.zeros((81, 3 ** k))
            K[0, 0] = 1.0
            K = K.reshape((3,) * 4 + (3,) * k)
            for p, tot, imgs in segs[r]:
                new = (1 - p) * K
                for u, cnt in imgs.items():
                    sh = list(u[4 * r:4 * r + 4])
                    cc = list(coords(self.B[r], self.piv[r],
                                     np.array(u[4 * (r + 1):]))) if k else []
                    shift = sh + cc
                    axes = [i for i, x in enumerate(shift) if x]
                    if axes:
                        new += (p * cnt / tot) * np.roll(
                            K, [shift[i] for i in axes], axis=axes)
                    else:
                        new += (p * cnt / tot) * K
                K = new
            self.K.append(K.reshape(81, 3 ** k))
        # final segment (after the last cut): L only
        self.final = np.zeros(3)
        self.final[0] = 1.0
        for p, tot, imgs in segs[m + 1]:
            new = (1 - p) * self.final
            for u, cnt in imgs.items():
                assert not any(u[:-1])
                new += (p * cnt / tot) * np.roll(self.final, u[-1])
            self.final = new
        assert self.k[m] <= 1
        # precomputed index maps for M_r(d)
        self.rowmap, self.colmap = [], []
        for r in range(m + 1):
            kprev = self.k[r - 1] if r > 0 else 0
            cprev = np.array(list(itertools.product(range(3), repeat=kprev)),
                             dtype=np.int64).reshape(3 ** kprev, kprev)
            cnew = np.array(list(itertools.product(range(3), repeat=self.k[r])),
                            dtype=np.int64).reshape(3 ** self.k[r], self.k[r])
            Uc = (cprev @ self.U[r].T) % 3                   # (Nprev, 4)
            Tc = (cprev @ self.T[r].T) % 3 if self.k[r] else np.zeros(
                (len(cprev), 0), dtype=np.int64)
            pw = 3 ** np.arange(self.k[r] - 1, -1, -1)
            colmap = (((cnew[None, :, :] - Tc[:, None, :]) % 3) @ pw
                      if self.k[r] else np.zeros((len(cprev), 1), dtype=np.int64))
            self.rowmap.append(Uc)
            self.colmap.append(colmap)
        # L value of the final coordinate
        if self.k[m] == 1:
            self.Lsign = int(self.B[m][0][-1])
        else:
            self.Lsign = 0

    def prob(self, d):
        """d: (m+1, 4) raw detector history -> P(D = d, L = l), l=0,1,2."""
        s = np.ones(1)
        for r in range(self.m + 1):
            rows = ((d[r] - self.rowmap[r]) % 3) @ POW4      # (Nprev,)
            M = self.K[r][rows[:, None], self.colmap[r]]     # (Nprev, Nnew)
            s = s @ M
        out = np.zeros(3)
        if self.k[self.m] == 1:
            for c in range(3):
                out[(c * self.Lsign) % 3] += s[c]
        else:
            out[0] = s[0]
        # convolve with final L-only noise
        res = np.zeros(3)
        for a in range(3):
            for b in range(3):
                res[(a + b) % 3] += out[a] * self.final[b]
        return res


# ----------------------------------------------------------- candidates
def candidates(chain):
    m = chain.m
    cands = {}
    a = []
    for r in range(m + 1):
        for v in itertools.product(range(3), repeat=4):
            if any(v):
                d = np.zeros((m + 1, 4), dtype=np.int64)
                d[r] = v
                a.append(d)
    cands["single-round flag pattern"] = a
    seen, b = set(), []
    for seg in chain.segs:
        for (_, _, imgs) in seg:
            for u in imgs:
                det = tuple(u[:-1])
                if any(det) and det not in seen:
                    seen.add(det)
                    b.append(np.array(det).reshape(m + 1, 4))
    cands["single-fault history"] = b
    singles = []
    for r in range(m + 1):
        for an in range(4):
            for val in (1, 2):
                d = np.zeros((m + 1, 4), dtype=np.int64)
                d[r, an] = val
                singles.append((r, d))
    c = [d1 + d2 for (r1, d1), (r2, d2) in itertools.combinations(singles, 2)
         if r1 != r2]
    cands["two single-ancilla flags, different rounds"] = c
    return cands


def raw_index(d, chain):
    # A2's joint array is ordered by model.detector_labels (big-endian)
    flat = np.zeros(chain.nd, dtype=np.int64)
    for (r, a), i in chain.col.items():
        flat[i] = d[r, a]
    return int(flat @ (3 ** np.arange(chain.nd - 1, -1, -1)))


def run(p_idx, d_idx, cl, validate=False):
    t0 = time.time()
    model, m, cuts = load(p_idx, d_idx, cl)
    ch = Chain(model, m, cuts)
    tb = time.time() - t0
    zero = np.zeros((m + 1, 4), dtype=np.int64)
    P0 = ch.prob(zero)
    q0 = P0.max() / P0.sum()
    print(f" p={PROBS[p_idx]:g} m={m} seq={cl}: state dims k_r={ch.k} "
          f"(build {tb:.0f}s); P(D=0)={P0.sum():.6f} q(0)={q0:.10f} "
          f"argmax {int(P0.argmax())}")
    cands = candidates(ch)
    if validate:
        spec = importlib.util.spec_from_file_location(
            "a2", os.path.join(HERE, "A2_optimal_rule_check.py"))
        a2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a2)
        Pf, _ = a2.exact_joint_from_dem(model)
        J = Pf.reshape(-1, 3)
        err, n = abs(P0 - J[0]).max() / J[0].max(), 1
        for name, L in cands.items():
            for d in L:
                ref = J[raw_index(d, ch)]
                pr = ch.prob(d)
                err = max(err, np.max(np.abs(pr - ref)) / max(ref.sum(), 1e-300))
                n += 1
        print(f"   validation vs full exact law: max |P_chain - P_full| / "
              f"P(D) over {n} histories = {err:.1e}")
    for name, L in cands.items():
        t1 = time.time()
        arr = np.array([ch.prob(d) for d in L])
        tot = arr.sum(1)
        q = arr.max(1) / tot
        k = int(np.argmax(q))
        above = q > q0
        flags = [tuple(x) for x in np.argwhere(L[k])]
        print(f"   {name}: {len(L)} histories ({time.time() - t1:.0f}s); max q "
              f"{q[k]:.10f}, margin q(0)-max = {q0 - q[k]:+.2e} (mass "
              f"{tot[k]:.1e}, flags (round, ancilla) {flags}); #q>q(0): "
              f"{int(above.sum())}, mass {tot[above].sum():.1e}")


def margins(p_idx, d_idx, cl, sets=None):
    model, m, cuts = load(p_idx, d_idx, cl)
    ch = Chain(model, m, cuts)
    P0 = ch.prob(np.zeros((m + 1, 4), dtype=np.int64))
    q0 = P0.max() / P0.sum()
    out = {}
    for name, L in candidates(ch).items():
        if sets is not None and name not in sets:
            continue
        arr = np.array([ch.prob(d) for d in L])
        q = arr.max(1) / arr.sum(1)
        out[name] = (q0 - q.max(), int((q > q0).sum()), len(L))
    return q0, out


def sweep():
    print("sweep: margin q(0) - max_candidates q(D) (positive = D=0 on top)")
    jobs = [(4, 2, range(30), None), (8, 2, range(10), None),
            (4, 4, range(3), None),
            (4, 8, range(1), ("single-round flag pattern",
                              "single-fault history"))]
    for p_idx, d_idx, seqs, sets in jobs:
        t0 = time.time()
        worst, n_above, n_hist, q0s = np.inf, 0, 0, []
        for cl in seqs:
            q0, out = margins(p_idx, d_idx, cl, sets)
            q0s.append(q0)
            for name, (mg, na, nh) in out.items():
                worst = min(worst, mg)
                n_above += na
                n_hist += nh
        print(f"  p={PROBS[p_idx]:g} m={DEPTHS[d_idx]} ({len(seqs)} seqs, "
              f"{time.time() - t0:.0f}s): q(0) in [{min(q0s):.6f}, "
              f"{max(q0s):.6f}]; min margin {worst:+.2e}; candidates above "
              f"D=0: {n_above} of {n_hist}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "sweep":
        sweep()
    if which in ("validate", "all"):
        run(4, 1, 0, validate=True)
        run(8, 1, 3, validate=True)
    if which in ("m4", "all"):
        run(4, 2, 0)
        run(4, 2, 1)
        run(8, 2, 0)
    if which in ("m6", "all"):
        run(4, 3, 0)
