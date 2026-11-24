#!/usr/bin/env python3
"""Adversarial critic of A4-nonmarkov-theory.md: one-qutrit toy checks.

Independent of A4's scripts.  Sections:
  catalog  C26: the repository's 216-string Clifford catalog
           (lrb.code_definitions.single_qutrit_cliffords_exhaustive_strings),
           mapped with sdim's own H, P, X, Z matrices: number of distinct
           elements mod phase, closure, frame potential, and the twirl of a
           random trace-decreasing map (is it a Id + b Dep?).
  limits   C3: counterexamples to "tends to a constant when the Perron term
           enters O, otherwise decays as (lambda_O/lambda_1)^m":
           (i) Jordan block at the Perron eigenvalue -> r(m) ~ 1/m;
           (ii) peripheral spectrum lambda_1 * omega^k -> r(m) periodic;
           (iii) rho with no overlap on the left Perron vector.
  ratio    C5: exact enumeration over all 216^m sequences for m = 1, 2:
           ratio of averages E[N]/E[D] vs (a/(a+b))^m, and the mean of
           per-sequence ratios E[N/D]; exact log-curvature
           ln r(0) - 2 ln r(1) + ln r(2) for state-independent and
           state-dependent acceptance; Monte Carlo with independent
           sequences per depth (no reuse across m).
  mirror   C6: exact sequence averages of mirrored sequences (m random
           Cliffords followed by their m inverses, as in LRB-D) vs standard
           RB (one inversion gate) with gate-dependent noise (stochastic
           and coherent).  Is the mirrored average one exponential plus a
           fast transient?

    module load python/3.11
    python critic_A4-nonmarkov-theory_toys.py [sections...]   (~1 minute)
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
d = 3
w = np.exp(2j * np.pi / 3)
Hm = np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)
Pm = np.diag([w ** (j * (j - 1) // 2) for j in range(d)])   # sdim, odd d
Xm = np.zeros((d, d), complex)
for i in range(d):
    Xm[i, (i - 1) % d] = 1                                  # sdim X
Zm = np.diag([w ** j for j in range(d)])
TOK = {"H": Hm, "P": Pm, "X": Xm, "Z": Zm, "I": np.eye(d, dtype=complex)}
rng = np.random.default_rng(424242)


def key(U):
    f = U.ravel()
    k = int(np.flatnonzero(np.abs(f) > 1e-9)[0])
    return tuple(np.round((U * np.conj(f[k]) / abs(f[k])).ravel(), 6))


def sup(U):
    return np.kron(U, U.conj())


def kraus_sup(K):
    return sum(np.kron(k, k.conj()) for k in K)


def catalog():
    sys.path.insert(0, os.path.join(WORKTREE, "src"))
    from lrb.code_definitions import single_qutrit_cliffords_exhaustive_strings as S
    mats = []
    for s in S:
        U = np.eye(d, dtype=complex)
        for t in reversed(s):          # rightmost symbol acts first
            U = TOK[t] @ U
        mats.append(U)
    return S, mats


def sec_catalog():
    print("\n== catalog (C26)")
    S, mats = catalog()
    keys = {key(U) for U in mats}
    closed = all(key(A @ B) in keys for A in mats for B in mats)
    fp = np.mean([[abs(np.trace(A.conj().T @ B)) ** 4 for B in mats]
                  for A in mats])
    fp3 = np.mean([[abs(np.trace(A.conj().T @ B)) ** 6 for B in mats]
                   for A in mats])
    print(f"  {len(S)} strings -> {len(keys)} distinct unitaries mod phase; "
          f"closed under products: {closed}")
    print(f"  frame potential t=2: {fp:.12f} (unitary 2-design iff 2); "
          f"t=3: {fp3:.6f} (3-design iff 6)")
    # twirl of a random trace-decreasing (non-unital) map
    G = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    Q, _ = np.linalg.qr(G)
    K = [Q[3 * i:3 * i + 3] for i in range(3)]
    M = np.diag([1.0, 0.7, 0.4])
    E = kraus_sup([k @ np.sqrt(M) for k in K])
    tw = sum(sup(U).conj().T @ E @ sup(U) for U in mats) / len(mats)
    Id = np.eye(9)
    Dep = np.outer(np.eye(3).ravel() / 3, np.eye(3).ravel())
    A = np.stack([Id.ravel(), Dep.ravel()], 1)
    c, *_ = np.linalg.lstsq(A, tw.ravel(), rcond=None)
    print(f"  twirl of a random trace-decreasing map = a Id + b Dep: "
          f"residual {np.abs(A @ c - tw.ravel()).max():.1e}, "
          f"a={c[0].real:.5f}, b={c[1].real:.5f}")


def sec_limits():
    print("\n== limits (C3): r(m) = tr[O E^m rho] / tr[E^m rho]")
    # classical (diagonal) CP trace-non-increasing maps: measure-and-prepare
    def mp_map(T):
        """E(rho) = sum_ij T_ij <j|rho|j> |i><i| (CP, trace non-increasing
        iff column sums <= 1)."""
        S = np.zeros((9, 9))
        for i in range(3):
            for j in range(3):
                S[i * 3 + i, j * 3 + j] = T[i, j]
        return S
    a, b = 0.6, 0.4
    T = np.array([[a, b, 0], [0, a, 0], [0, 0, 0.3]])
    E = mp_map(T)
    rho = np.diag([0, 1.0, 0]).astype(complex).ravel()
    O = np.diag([0, 1.0, 0])
    r = []
    for m in (1, 10, 100, 1000):
        v = np.linalg.matrix_power(E, m) @ rho
        r.append(np.trace(O @ v.reshape(3, 3)).real /
                 np.trace(v.reshape(3, 3)).real)
    print(f"  (i) Jordan block at the Perron eigenvalue {a}: column sums "
          f"{T.sum(0)}; r(1,10,100,1000) = {np.round(r, 6)}; "
          f"m r(m) -> {1000 * r[-1]:.4f} (= a/b = {a / b:.4f}): decays like"
          f" 1/m, neither constant nor exponential, although the Perron "
          f"eigenvalue's term enters N(m) = a^m")
    P = np.roll(np.eye(3), 1, axis=0)
    E2 = mp_map(0.9 * P)
    rho2 = np.diag([1.0, 0, 0]).astype(complex).ravel()
    O2 = np.diag([1.0, 0.0, -1.0])
    rr = []
    for m in range(1, 8):
        v = np.linalg.matrix_power(E2, m) @ rho2
        rr.append(np.trace(O2 @ v.reshape(3, 3)).real /
                  np.trace(v.reshape(3, 3)).real)
    ev = np.linalg.eigvals(E2)
    print(f"  (ii) E = 0.9 x cyclic permutation (acceptance state "
          f"independent): peripheral |lambda| = "
          f"{np.round(np.sort(np.abs(ev))[-3:], 3)}; r(1..7) = "
          f"{np.round(rr, 3)}: periodic, no limit")
    T3 = np.array([[0.9, 0, 0], [0, 0.5, 0.2], [0, 0.3, 0.6]])
    E3 = mp_map(T3)
    rho3 = np.diag([0, 0.5, 0.5]).astype(complex).ravel()
    O3 = np.diag([1.0, 1.0, -2.0])
    v = np.linalg.matrix_power(E3, 200) @ rho3
    print(f"  (iii) rho supported off the Perron block (lambda_1 = 0.9 on "
          f"|0>): r(200) = {np.trace(O3 @ v.reshape(3, 3)).real / np.trace(v.reshape(3, 3)).real:.4f};"
          f" the limit is set by the dominant eigenvalue that rho reaches "
          f"(0.8), not by lambda_1")


def per_sequence_ratio(E, mats, seqs, O, rho):
    """N/D for given sequences (list of index arrays), ideal inversion."""
    out = []
    for s in seqs:
        st = rho.copy()
        U = np.eye(3, dtype=complex)
        for g in s:
            st = E @ (sup(mats[g]) @ st)
            U = mats[g] @ U
        st = sup(U.conj().T) @ st
        R = st.reshape(3, 3)
        out.append((np.trace(O @ R), np.trace(R).real))
    return np.array(out)


def full_group():
    grp, keys = [np.eye(3, dtype=complex)], {key(np.eye(3)): 0}
    front = [0]
    while front:
        new = []
        for i in front:
            for G in (Hm, Pm):
                V = G @ grp[i]
                kk = key(V)
                if kk not in keys:
                    keys[kk] = len(grp)
                    grp.append(V)
                    new.append(keys[kk])
        front = new
    return grp


def sec_ratio():
    for which in ("group", "catalog"):
        _sec_ratio(which)


def _sec_ratio(which):
    print(f"\n== ratio (C5): exact enumeration over all sequences "
          f"[{which}: {'full Clifford group, uniform' if which == 'group' else 'repository 216-string catalog'}]")
    mats = full_group() if which == "group" else catalog()[1]
    nG = len(mats)
    sups = np.stack([sup(U) for U in mats])
    rho = (np.ones((3, 3)) / 3).astype(complex).ravel()        # |+><+|
    O = Xm.conj().T
    G = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
    Q, _ = np.linalg.qr(G)
    K = [np.sqrt(0.95) * np.eye(3)] + [np.sqrt(0.05) * Q[3 * i:3 * i + 3]
                                       for i in range(2)]
    for lab, M in (("state-independent acceptance", 0.8 * np.eye(3)),
                   ("state-dependent acceptance", np.diag([1.0, 0.6, 0.3]))):
        E = kraus_sup([k @ np.sqrt(M) for k in K])
        tw = np.einsum("gji,jk,gkl->il", sups.conj(), E, sups) / nG
        Id = np.eye(9)
        Dep = np.outer(np.eye(3).ravel() / 3, np.eye(3).ravel())
        A = np.stack([Id.ravel(), Dep.ravel()], 1)
        c, *_ = np.linalg.lstsq(A, tw.ravel(), rcond=None)
        f = (c[0] / (c[0] + c[1])).real
        res = {}
        for m in (1, 2):
            idx = np.array(np.meshgrid(*[np.arange(nG)] * m,
                                       indexing="ij")).reshape(m, -1).T
            nd = per_sequence_ratio(E, mats, idx, O, rho)
            N, D = nd[:, 0], nd[:, 1].real
            res[m] = (N.mean() / D.mean(), (N / D).mean())
        ra = [1.0] + [res[m][0] for m in (1, 2)]
        rm = [1.0] + [res[m][1] for m in (1, 2)]
        curv_a = np.log(abs(ra[0])) - 2 * np.log(abs(ra[1])) + np.log(abs(ra[2]))
        curv_m = np.log(abs(rm[0])) - 2 * np.log(abs(rm[1])) + np.log(abs(rm[2]))
        print(f"  {lab}: f = a/(a+b) = {f:.6f}")
        print(f"     ratio of averages m=1,2: {np.round(ra[1:], 8)} vs f^m = "
              f"{np.round([f, f * f], 8)}")
        print(f"     mean of per-sequence ratios m=1,2: {np.round(rm[1:], 8)};"
              f" log-curvature ln r0 - 2 ln r1 + ln r2: ratio-of-averages "
              f"{curv_a:+.2e}, mean-of-ratios {curv_m:+.2e} (0 for one "
              f"exponential through m = 0, 1, 2)")
    # Monte Carlo with independent sequences per depth
    M = np.diag([1.0, 0.6, 0.3])
    E = kraus_sup([k @ np.sqrt(M) for k in K])
    ms = np.arange(1, 21)
    nseq = 20000
    mean, se = [], []
    for m in ms:
        idx = rng.integers(0, nG, size=(nseq, m))
        st = np.tile(rho, (nseq, 1))
        Ucum = np.tile(np.eye(3, dtype=complex), (nseq, 1, 1))
        for t in range(m):
            st = np.einsum("nij,nj->ni", sups[idx[:, t]], st) @ E.T
            Ucum = np.einsum("nij,njk->nik", np.stack(mats)[idx[:, t]], Ucum)
        Ui = np.conj(np.transpose(Ucum, (0, 2, 1)))
        R = np.einsum("nij,njk,nlk->nil", Ui, st.reshape(nseq, 3, 3),
                      Ui.conj())
        rr = np.einsum("ij,nji->n", O, R) / np.einsum("nii->n", R).real
        mean.append(rr.real.mean())
        se.append(rr.real.std() / np.sqrt(nseq))
    mean, se = np.array(mean), np.array(se)
    Q = np.stack([np.ones(len(ms)), ms], 1)
    W = (mean / se) ** 2
    beta = np.linalg.solve(Q.T @ (Q * W[:, None]), (Q * W[:, None]).T @
                           np.log(mean))
    fit = np.exp(Q @ beta)
    chi2 = np.sum(((mean - fit) / se) ** 2)
    Q2 = np.stack([np.ones(len(ms)), ms, ms ** 2], 1)
    b2 = np.linalg.solve(Q2.T @ (Q2 * W[:, None]), (Q2 * W[:, None]).T @
                         np.log(mean))
    chi2q = np.sum(((mean - np.exp(Q2 @ b2)) / se) ** 2)
    print(f"  MC, independent sequences per depth (n={nseq} each), "
          f"state-dependent acceptance, Re of mean ratio, m=1..20: "
          f"single-exp chi2/dof = {chi2 / (len(ms) - 2):.2f}, log-quadratic "
          f"chi2/dof = {chi2q / (len(ms) - 3):.2f}, curvature {-b2[2]:+.2e}")


def sec_mirror():
    print("\n== mirror (C6): mirrored vs standard RB, gate-dependent noise;"
          " uniform over the group vs the repository catalog distribution")
    S, cmats = catalog()
    # full group by breadth-first search from H and P (word length = cost)
    grp, keys, wl = [np.eye(3, dtype=complex)], {key(np.eye(3)): 0}, [0]
    front = [0]
    while front:
        new = []
        for i in front:
            for G in (Hm, Pm):
                V = G @ grp[i]
                kk = key(V)
                if kk not in keys:
                    keys[kk] = len(grp)
                    grp.append(V)
                    wl.append(wl[i] + 1)
                    new.append(keys[kk])
        front = new
    nG = len(grp)
    inv = np.array([keys[key(U.conj().T)] for U in grp])
    mult = np.array([[keys[key(A @ B)] for B in grp] for A in grp])
    cat_idx = np.array([keys[key(U)] for U in cmats])
    w_cat = np.bincount(cat_idx, minlength=nG) / len(cat_idx)
    w_uni = np.ones(nG) / nG
    Kh = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Kh = (Kh + Kh.conj().T) / 2
    lens = np.array(wl)

    def noise(g, kind):
        n = lens[g] + 1
        if kind == "stochastic":         # depolarising, strength ~ length
            q = 0.004 * n
            return (1 - q) * np.eye(9) + q * np.outer(np.eye(3).ravel() / 3,
                                                      np.eye(3).ravel())
        th = 0.02 * n                    # coherent over-rotation ~ length
        ev, V = np.linalg.eigh(Kh)
        U = V @ np.diag(np.exp(-1j * th * ev)) @ V.conj().T
        return sup(U)

    rho = (np.ones((3, 3)) / 3).astype(complex).ravel()
    O = Xm.conj().T
    print(f"  group order {nG}; catalog covers {np.count_nonzero(w_cat)} "
          f"elements; max catalog weight {w_cat.max() * 216:.0f}/216")
    for wlab, wts in (("uniform", w_uni), ("catalog", w_cat)):
      for kind in ("stochastic", "coherent"):
        Nf = np.stack([noise(g, kind) @ sup(grp[g]) for g in range(nG)])
        Ni = np.stack([noise(inv[g], kind) @ sup(grp[inv[g]])
                       for g in range(nG)])
        # mirrored: Y_k = sum_g w_g Ni_g Y_{k-1} Nf_g, Y_0 = Id
        Y = np.eye(9, dtype=complex)
        Fmir = [np.trace(O @ (Y @ rho).reshape(3, 3)).real]
        for m in range(1, 31):
            Y = np.einsum("g,gij,jk,gkl->il", wts, Ni, Y, Nf)
            Fmir.append(np.trace(O @ (Y @ rho).reshape(3, 3)).real)
        # standard RB: Markov chain on the cumulative element, noisy inverse
        st = np.zeros((nG, 9), dtype=complex)
        st[0] = rho
        Fstd = [1.0]
        for m in range(1, 31):
            new = np.zeros_like(st)
            for g in range(nG):
                if wts[g] > 0:
                    new[mult[g]] += wts[g] * (st @ Nf[g].T)   # c -> g c
            st = new
            tot = sum(Ni[c] @ st[c] for c in range(nG))   # noisy inverse
            Fstd.append(np.trace(O @ tot.reshape(3, 3)).real)
        for lab, F in (("mirrored", np.array(Fmir)), ("standard", np.array(Fstd))):
            print(f"  [{wlab}]", end="")
            m = np.arange(len(F))
            sel = m >= 4
            Q = np.stack([np.ones(sel.sum()), m[sel]], 1)
            bb, *_ = np.linalg.lstsq(Q, np.log(F[sel]), rcond=None)
            dev = np.max(np.abs(np.exp(Q @ bb) / F[sel] - 1))
            rates = (F[1:] / F[:-1])
            print(f"  {kind:10s} {lab:8s}: F(1,2,5,30) = "
                  f"{np.round(F[[1, 2, 5, 30]], 6)}; windowed rates m=1->2 "
                  f"{rates[1]:.6f}, 10->11 {rates[10]:.6f}, 29->30 "
                  f"{rates[29]:.6f}; max rel dev of one exponential on m>=4:"
                  f" {dev:.1e}")


SECTIONS = {"catalog": sec_catalog, "limits": sec_limits,
            "ratio": sec_ratio, "mirror": sec_mirror}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(SECTIONS):
        SECTIONS[n]()
