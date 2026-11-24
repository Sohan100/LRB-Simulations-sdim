"""Exact low-order expansion of the accepted logical error from sdim's DEM.

sdim's compact DEM writes every noise gate as one mechanism i: with
probability pi_i it adds a uniformly random element of a subgroup V_i of
Z_q^(D+1) (D detectors, one logical observable).  For postselection on a set
S of detectors, a shot is accepted when every detector in S is 0.

Single mechanism:  a_i  = Pr[v in V_i has det_S(v) = 0 and log(v) != 0]
Pair of mechanisms: a_ij = Pr[v_i + v_j has det_S = 0 and log != 0],
v_i, v_j uniform and independent.  Writing w_ij for the rank of V_i + V_j,
a_ij = (q - 1) q^(-w_ij) when e_L lies in V_i + V_j (S-restricted), else 0:
the silent fraction of a fault pair is q^-(w_ij - 1) (1 - 1/q).  Every
q-dependence of the low-order logical error therefore sits in integer ranks
and in pi_i (linear in p; pi = p q^2/(q^2-1) for DEP1, p q^4/(q^4-1) for
DEP2, p q/(q-1) for flips).

The accepted logical-error probability to second order is
    eps = sum_i pi_i a_i + sum_{i<j} pi_i pi_j (a_ij - a_i - a_j) + O(pi^3)
(the conditional error differs from it at third order).  This module counts
a_ij exactly by enumerating every element of every V_i (q^rank of them; rank
<= 4) and matching detector images v_i = -v_j on S.
"""
from __future__ import annotations

import itertools

import numpy as np


def mechanism_arrays(dem, det_select: np.ndarray | None = None):
    """Dense generator matrices restricted to the selected detectors + logical.

    Returns list of (prob, G) with G an int array (rank, |S| + 1), last column
    the logical observable.  Mechanisms whose restricted image is zero are
    dropped (they can neither be detected nor flip the logical).
    """
    q = dem.dimension
    D = dem.num_detectors
    cols = np.arange(D) if det_select is None else np.asarray(det_select)
    colmap = {int(c): i for i, c in enumerate(cols)}
    out = []
    for m in dem.mechanisms:
        rows = []
        for g in m.generators:
            row = np.zeros(len(cols) + 1, dtype=np.int64)
            for t, v in g.items():
                t = int(t)
                if t >= D:
                    row[-1] = (row[-1] + int(v)) % q
                elif t in colmap:
                    row[colmap[t]] = (row[colmap[t]] + int(v)) % q
            rows.append(row)
        G = np.array(rows, dtype=np.int64).reshape(len(rows), len(cols) + 1)
        G = row_reduce(G, q)
        if G.shape[0] == 0:
            continue
        out.append((float(m.probability), G, m.source))
    return out


def row_reduce(G: np.ndarray, q: int) -> np.ndarray:
    """Basis of the row space of G over Z_q (q prime)."""
    A = np.array(G, dtype=np.int64) % q
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, c] % q:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), -1, q)
        A[r] = (A[r] * inv) % q
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % q
        r += 1
        if r == rows:
            break
    return A[:r]


def rank_mod(G: np.ndarray, q: int) -> int:
    return row_reduce(G, q).shape[0]


def elements(G: np.ndarray, q: int) -> np.ndarray:
    r = G.shape[0]
    coeffs = np.array(list(itertools.product(range(q), repeat=r)), dtype=np.int64)
    return (coeffs @ G) % q


def second_order(dem, det_select=None, p: float = 1.0, mech_labels=None):
    """First-order coefficient and bulk two-fault coefficient, vectorized.

    Every element of every V_i is enumerated once (q^rank rows, rank <= 4);
    pairs are matched through their detector images with sparse products:
      N_ij = #{(v, u) in V_i x V_j : det(v) = -det(u) != 0, log(v)+log(u) != 0}
           = (A B^T)_ij - (C D^T)_ij,
    A[i,K] = #{v in V_i: det v = K}, B[j,K] = #{u in V_j: det u = -K},
    C[i,(K,l)] = #{v: det K, log l}, D[j,(K,l)] = #{u: det -K, log -l}.
    Then a_ij = N_ij / (|V_i||V_j|) = (q-1) q^-w_ij for every bulk pair
    (both mechanisms detectable on their own), and
      S1  = sum_i (pi_i/p) a_i            (first-order silent mechanisms),
      S2b = sum_{i<j bulk} (pi_i pi_j/p^2) a_ij.
    Returns dict(S1, S2b, n_mech, pairs=DataFrame(i, j, w, weight, a)).
    """
    import pandas as pd
    from scipy import sparse

    q = dem.dimension
    mechs = mechanism_arrays(dem, det_select)
    n = len(mechs)
    probs = np.array([m[0] for m in mechs]) / p
    sizes = np.array([q ** m[1].shape[0] for m in mechs], dtype=float)
    k_size = np.ones(n)
    det_rows, logs, mids = [], [], []
    for i, (_, G, _) in enumerate(mechs):
        E = elements(G, q)
        det = E[:, :-1]
        zero = ~det.any(axis=1)
        k_size[i] = zero.sum()
        nz = ~zero
        det_rows.append(det[nz].astype(np.int16))
        logs.append(E[nz, -1])
        mids.append(np.full(int(nz.sum()), i, dtype=np.int64))
    det_all = np.concatenate(det_rows)
    log_all = np.concatenate(logs).astype(np.int64)
    mid_all = np.concatenate(mids)
    neg_all = ((-det_all) % q).astype(np.int16)
    both = np.concatenate([det_all, neg_all])
    _, inv = np.unique(both, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    key = inv[: len(det_all)].astype(np.int64)
    negkey = inv[len(det_all):].astype(np.int64)
    nk = int(inv.max()) + 1
    ones = np.ones(len(key), dtype=np.int64)
    A = sparse.csr_matrix((ones, (mid_all, key)), shape=(n, nk))
    B = sparse.csr_matrix((ones, (mid_all, negkey)), shape=(n, nk))
    C = sparse.csr_matrix((ones, (mid_all, key * q + log_all)), shape=(n, nk * q))
    D = sparse.csr_matrix((ones, (mid_all, negkey * q + (-log_all) % q)), shape=(n, nk * q))
    N = (A @ B.T - C @ D.T).tocoo()
    a1 = (k_size - 1) / sizes
    S1 = float(np.sum(probs * a1))
    sel = (N.row < N.col) & (N.data != 0)
    i, j, cnt = N.row[sel], N.col[sel], N.data[sel].astype(float)
    bulk = (k_size[i] == 1) & (k_size[j] == 1)
    i, j, cnt = i[bulk], j[bulk], cnt[bulk]
    a = cnt / (sizes[i] * sizes[j])
    w = np.rint(np.log((q - 1) / a) / np.log(q)).astype(int)
    assert np.allclose(a * float(q) ** w / (q - 1), 1.0), "a_ij not of the form (q-1) q^-w"
    weight = probs[i] * probs[j]
    pairs = pd.DataFrame(dict(i=i, j=j, w=w, weight=weight, a=a))
    return dict(S1=S1, S2b=float(np.sum(weight * a)), n_mech=n, pairs=pairs,
                sources=[m[2] for m in mechs])
